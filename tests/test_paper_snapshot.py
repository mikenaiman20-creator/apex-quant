import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest

import server.paper_snapshot as paper_snapshot


def _env_file(path: Path, **values: str) -> Path:
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return path


class _FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode("utf-8"))


def test_refresh_writes_only_allowlisted_paper_fields(monkeypatch, tmp_path):
    secret = "test-$SECRET-value"
    env_file = _env_file(
        tmp_path / ".env",
        TRADING_MODE="PAPER",
        BROKER="ALPACA",
        ALPACA_API_KEY="test-key",
        ALPACA_API_SECRET=secret,
    )
    opener = _FakeOpener([
        {
            "id": "private-account-id",
            "status": "ACTIVE",
            "currency": "USD",
            "equity": "25000",
            "private_field": "not published",
        },
        [{
            "symbol": "SPY",
            "qty": "2",
            "side": "long",
            "market_value": "1000",
            "asset_id": "private-asset-id",
        }],
        [{
            "symbol": "SPY",
            "side": "buy",
            "qty": "2",
            "status": "filled",
            "client_order_id": "private-client-order-id",
        }],
    ])
    monkeypatch.setattr(paper_snapshot, "build_opener", lambda *args: opener)

    output = tmp_path / "logs" / "paper_dashboard_snapshot.json"
    result = paper_snapshot.refresh_snapshot(env_file, output)

    assert result["paper_api_status"] == "connected"
    assert result["mode"] == "PAPER"
    assert result["account"] == {
        "status": "ACTIVE",
        "currency": "USD",
        "equity": "25000",
    }
    assert result["positions"] == [{
        "symbol": "SPY",
        "qty": "2",
        "side": "long",
        "market_value": "1000",
    }]
    assert result["orders"] == [{
        "symbol": "SPY",
        "side": "buy",
        "qty": "2",
        "status": "filled",
    }]
    assert len(opener.requests) == 3
    assert all(request.method == "GET" for request, _ in opener.requests)
    assert all(
        request.full_url.startswith(paper_snapshot.PAPER_API_URL)
        for request, _ in opener.requests
    )
    assert all(
        request.get_header("Apca-api-key-id") == "test-key"
        for request, _ in opener.requests
    )
    assert all(
        request.get_header("Apca-api-secret-key") == secret
        for request, _ in opener.requests
    )
    serialized = output.read_text(encoding="utf-8")
    assert secret not in serialized
    assert "private-account-id" not in serialized
    assert "private-asset-id" not in serialized
    assert "private-client-order-id" not in serialized
    if os.name == "posix":
        assert output.stat().st_mode & 0o777 == 0o600


def test_live_mode_is_rejected_without_making_api_requests(monkeypatch, tmp_path):
    env_file = _env_file(
        tmp_path / ".env",
        TRADING_MODE="LIVE",
        BROKER="ALPACA",
        ALPACA_API_KEY="test-key",
        ALPACA_API_SECRET="test-secret",
    )
    monkeypatch.setattr(
        paper_snapshot,
        "build_opener",
        lambda *args: pytest.fail("API client must not be constructed in LIVE mode"),
    )

    with pytest.raises(ValueError, match="TRADING_MODE=PAPER"):
        paper_snapshot.refresh_snapshot(env_file, tmp_path / "snapshot.json")


def test_api_failure_does_not_publish_success_snapshot(monkeypatch, tmp_path):
    env_file = _env_file(
        tmp_path / ".env",
        TRADING_MODE="PAPER",
        BROKER="ALPACA",
        ALPACA_API_KEY="test-key",
        ALPACA_API_SECRET="test-secret",
    )
    opener = _FakeOpener([
        HTTPError("https://paper-api.alpaca.markets/v2/account", 401, "unauthorized", {}, None),
    ])
    monkeypatch.setattr(paper_snapshot, "build_opener", lambda *args: opener)
    output = tmp_path / "snapshot.json"

    with pytest.raises(RuntimeError, match="HTTP 401"):
        paper_snapshot.refresh_snapshot(env_file, output)

    assert not output.exists()
