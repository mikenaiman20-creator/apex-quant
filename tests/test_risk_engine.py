import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from server.risk_engine import (
    AuditIntegrityError,
    RiskAuditLog,
    RiskEngineService,
    RiskGateClient,
    RiskGateUnavailable,
    RiskRequestHandler,
)

GATE_TOKEN = "g" * 64
ADMIN_TOKEN = "a" * 64


def make_service(tmp_path):
    return RiskEngineService(
        gate_token=GATE_TOKEN,
        admin_token=ADMIN_TOKEN,
        audit_path=tmp_path / "risk_audit.jsonl",
        state_path=tmp_path / "risk_engine_state.json",
        heartbeat_path=tmp_path / "risk_engine_heartbeat.json",
        allowed_symbols={"SPY", "QQQ", "NVDA"},
    )


def valid_batch(**overrides):
    request = {
        "request_id": "test-request-1",
        "operation": "submit_batch",
        "mode": "PAPER",
        "account_equity": 100000,
        "open_symbols": [],
        "orders": [{
            "symbol": "SPY",
            "qty": 10,
            "side": "buy",
            "limit_price": 100,
            "stop_price": 98,
            "take_profit_price": 103,
            "client_order_id": "apx-spy-1",
        }],
    }
    request.update(overrides)
    return request


def test_risk_engine_approves_with_independent_risk_and_reward_checks(tmp_path):
    service = make_service(tmp_path)
    result = service.authorize(valid_batch())
    assert result["approved"] is True
    assert result["details"]["risk_pct"] == pytest.approx(0.0002)
    assert result["details"]["reward_risk"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (valid_batch(account_equity=1000), "per-trade risk limit"),
        (valid_batch(mode="LIVE"), "PAPER mode"),
        (valid_batch(orders=[{
            "symbol": "XYZ", "qty": 1, "side": "buy", "limit_price": 10,
            "stop_price": 9, "take_profit_price": 12, "client_order_id": "x",
        }]), "allowlist"),
        (valid_batch(orders=[{
            "symbol": "SPY", "qty": 1, "side": "buy", "limit_price": 10,
            "stop_price": 9, "take_profit_price": 10.1, "client_order_id": "x",
        }]), "reward/risk"),
        (valid_batch(open_symbols=["QQQ", "NVDA", "AAPL"]), "position count"),
    ],
)
def test_risk_engine_rejects_unsafe_batches_and_audits_each(tmp_path, payload, reason):
    service = make_service(tmp_path)
    result = service.authorize(payload)
    assert result["approved"] is False
    assert reason in result["reason"]
    audit = RiskAuditLog(tmp_path / "risk_audit.jsonl")
    assert audit.sequence == 1


def test_override_is_never_an_order_policy_bypass_and_is_logged(tmp_path):
    service = make_service(tmp_path)
    result = service.authorize(valid_batch(override=True))
    assert result["approved"] is False
    record = json.loads((tmp_path / "risk_audit.jsonl").read_text().splitlines()[0])
    assert record["event"] == "risk_override_rejected"


def test_reconciliation_failure_latches_and_persists_risk_service(tmp_path):
    service = make_service(tmp_path)
    failed = service.reconcile({
        "request_id": "reconcile-1",
        "mode": "PAPER",
        "account": {"equity": "110000", "cash": "98000"},
        "positions": [{"symbol": "SPY", "qty": "20", "market_value": "4000"}],
        "target_symbols": ["SPY"],
        "managed_symbols": ["SPY"],
    })
    assert failed["approved"] is False
    assert service.health()["status"] == "blocked"
    denied = service.authorize(valid_batch())
    assert denied["approved"] is False
    restored = make_service(tmp_path)
    assert restored.health()["status"] == "blocked"


def test_risk_audit_detects_edits_and_truncation(tmp_path):
    path = tmp_path / "risk_audit.jsonl"
    audit = RiskAuditLog(path)
    audit.append("risk_check", {"approved": True})
    audit.append("risk_rejection", {"approved": False})
    text = path.read_text()
    path.write_text(text.replace('"approved":true', '"approved":false', 1))
    with pytest.raises(AuditIntegrityError):
        RiskAuditLog(path)

    path.write_text(text.splitlines()[0] + "\n")
    with pytest.raises(AuditIntegrityError):
        RiskAuditLog(path)


def test_operator_latch_reset_is_authenticated_and_audited(tmp_path):
    service = make_service(tmp_path)
    service._latch("reconciliation mismatch")
    with pytest.raises(PermissionError):
        service.reset_block(
            {"actor": "operator", "reason": "reviewed"}, "bad-token",
        )
    result = service.reset_block(
        {"request_id": "reset-1", "actor": "operator", "reason": "reviewed"},
        ADMIN_TOKEN,
    )
    assert result["approved"] is True
    assert service.health()["status"] == "ready"
    audit = RiskAuditLog(tmp_path / "risk_audit.jsonl")
    assert audit.sequence == 2
    records = [
        json.loads(line) for line in (tmp_path / "risk_audit.jsonl").read_text().splitlines()
    ]
    assert records[-1]["event"] == "risk_override"


def test_risk_gateway_http_auth_health_and_loopback_requirement(tmp_path):
    service = make_service(tmp_path)
    service.refresh_heartbeat()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RiskRequestHandler)
    server.risk_service = service
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RiskGateClient(
        f"http://127.0.0.1:{server.server_port}", GATE_TOKEN,
    )
    try:
        assert client.health()["status"] == "ready"
        assert client.authorize(valid_batch())["approved"] is True
        assert client.reconcile({
            "mode": "PAPER",
            "account": {"equity": 100000, "cash": 100000},
            "positions": [],
        })["approved"] is True
        with pytest.raises(ValueError, match="loopback"):
            RiskGateClient("http://example.com:18765", GATE_TOKEN)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_risk_gate_reports_local_configuration_without_contacting_gateway():
    missing = RiskGateClient("http://127.0.0.1:18765", "")
    status = missing.configuration_status()
    assert status["configured"] is False
    assert status["token_configured"] is False
    assert "not configured" in status["reason"].lower()

    configured = RiskGateClient("http://127.0.0.1:18765", GATE_TOKEN)
    assert configured.configuration_status()["configured"] is True


def test_risk_gate_rejects_stale_heartbeat(monkeypatch):
    client = RiskGateClient(
        "http://127.0.0.1:18765", GATE_TOKEN,
        max_heartbeat_age=1,
    )
    monkeypatch.setattr(
        client, "_request",
        lambda *_args, **_kwargs: {
            "status": "ready", "heartbeat_at": time.time() - 60,
        },
    )
    with pytest.raises(RiskGateUnavailable, match="stale"):
        client.health()
