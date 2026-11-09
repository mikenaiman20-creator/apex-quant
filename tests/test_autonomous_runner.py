import asyncio
import json
import logging
import threading
import time
from datetime import datetime, time as datetime_time, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import server.autonomous_runner as runner_module
from server.dashboard import DashboardState
from engine.data_pipeline import Bar, Quote, SymbolState
from server.autonomous_runner import (
    AlpacaClient, AutonomousRunner, _adx, _feed_bar, _opening_range_breakout,
    _relative_volume, load_crypto_symbols, load_paper_settings,
    write_dashboard_market_snapshot, write_dashboard_paper_snapshot,
)
from engine.strategy import Check, Decision, Side
from server.ingestion.brain import KnowledgeBrain
from server.broker_adapters import AlpacaPaperAdapter

NY = ZoneInfo("America/New_York")


class ApprovingRiskGate:
    def health(self):
        return {"status": "ready", "heartbeat_at": time.time()}

    def authorize(self, _request):
        return {"approved": True, "reason": "test approval"}

    def reconcile(self, _request):
        return {"approved": True, "reason": "test reconciliation"}


def test_runner_rejects_live_mode(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    with pytest.raises(RuntimeError):
        load_paper_settings()


def test_single_instance_lock_releases_only_a_lock_it_acquired(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    lock_path = tmp_path / "logs" / "autonomous.lock"
    with runner_module.single_instance_lock():
        assert lock_path.exists()
    with runner_module.single_instance_lock():
        assert lock_path.exists()


def test_runner_accepts_configured_universe_seed(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("WATCHLIST", "SPY,QQQ")
    assert load_paper_settings().watchlist == ("SPY", "QQQ")


def test_runner_halts_before_broker_access_when_risk_heartbeat_is_unavailable(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()

    class UnavailableRiskGate:
        def health(self):
            raise runner_module.RiskGateUnavailable("risk heartbeat unavailable")

    class Broker:
        called = False

        def clock(self):
            self.called = True
            raise AssertionError("broker must not be contacted")

    broker = Broker()
    runner = AutonomousRunner(
        settings, broker, risk_gate=UnavailableRiskGate(),
    )
    assert runner.run_once() == runner_module.POLL_SECONDS
    assert runner.risk_gateway_halted is True
    assert broker.called is False


def test_order_routing_requires_explicit_enablement_and_promoted_brain_strategy(
    monkeypatch, tmp_path,
):
    runner = AutonomousRunner.__new__(AutonomousRunner)
    runner.settings = SimpleNamespace(mode=runner_module.TradingMode.PAPER)
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    monkeypatch.delenv("ENABLE_ORDER_ROUTING", raising=False)
    assert not runner._order_routing_authorized("stock")

    monkeypatch.setenv("ENABLE_ORDER_ROUTING", "true")
    assert not runner._order_routing_authorized("stock")
    brain = KnowledgeBrain(tmp_path / "data" / "knowledge_brain.sqlite")
    extraction = {
        "title": "Validated equity rules",
        "category": "stocks",
        "storage_category": "stock",
        "rules": {
            "indicators": ["ATR(14)"],
            "entry_condition": "Close breaks 0.1% above range.",
            "exit_condition": "Exit at 1.25 ATR or 2R.",
            "stop_loss_atr": 1.25,
            "take_profit_mult": 2.0,
        },
        "conviction_score": 0.82,
        "source_url": "https://example.org/validated",
    }
    strategy_id = brain.add_strategy(extraction)
    assert strategy_id is not None
    brain.record_simulation(strategy_id, 20, 11, 150.0, 100.0)
    brain.promote_strategy(strategy_id)
    assert runner._order_routing_authorized("stock")


def test_alpaca_adapter_routes_only_protected_equity_orders():
    class Client:
        def submit_bracket(self, payload):
            return {"id": payload["client_order_id"]}

    adapter = AlpacaPaperAdapter(Client())
    assert adapter.submit_protected_order("stock", {"client_order_id": "paper-1"}) == {
        "id": "paper-1",
    }
    with pytest.raises(NotImplementedError, match="not enabled for crypto"):
        adapter.submit_protected_order("crypto", {})


def test_crypto_pair_configuration_is_validated(monkeypatch):
    monkeypatch.setenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,BTC/USD")
    assert load_crypto_symbols() == ("BTC/USD", "ETH/USD")
    monkeypatch.setenv("CRYPTO_SYMBOLS", "BTCUSD")
    with pytest.raises(RuntimeError, match="slash-pair"):
        load_crypto_symbols()


def test_crypto_research_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CRYPTO_SYMBOLS", raising=False)
    assert load_crypto_symbols() == ()
    monkeypatch.setenv("CRYPTO_SYMBOLS", "")
    assert load_crypto_symbols() == ()


def test_bar_time_accepts_nanosecond_market_timestamps():
    parsed = runner_module._bar_time("2026-11-08T18:21:00.788764427+00:00")
    assert parsed == datetime(2026, 11, 8, 18, 21, 0, 788764, tzinfo=timezone.utc)


def test_crypto_research_failures_are_rate_limited_warnings(monkeypatch, caplog):
    timestamps = iter((1000.0, 1010.0, 1300.0))
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(timestamps))
    runner = AutonomousRunner.__new__(AutonomousRunner)
    runner.crypto_error_last_logged_at = None

    with caplog.at_level(logging.WARNING, logger="apex.autonomous"):
        runner._log_crypto_research_failure(RuntimeError("feed unavailable"))
        runner._log_crypto_research_failure(RuntimeError("feed unavailable"))
        runner._log_crypto_research_failure(RuntimeError("feed unavailable"))

    assert len(caplog.records) == 2
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    assert all("equity processing will continue" in record.message for record in caplog.records)


def test_alpaca_client_routes_crypto_market_data_to_crypto_endpoints():
    client = AlpacaClient("key", "secret")
    requests = []

    def request(base, path, method="GET", params=None, payload=None):
        requests.append((base, path, params))
        if path.endswith("/bars"):
            return {"bars": {"BTC/USD": []}}
        return {"quotes": {"BTC/USD": {"ap": 100, "bp": 99}}}

    client.request = request
    start = datetime(2026, 10, 29, tzinfo=timezone.utc)
    client.bars(("BTC/USD",), "1Min", start, start + timedelta(minutes=1), 10)
    quotes = client.latest_quotes(("BTC/USD",))
    assert requests[0][1] == "/v1beta3/crypto/us/bars"
    assert requests[0][2]["symbols"] == "BTC/USD"
    assert requests[1][1] == "/v1beta3/crypto/us/latest/quotes"
    assert "BTC/USD" in quotes


def test_paper_runner_clamps_risk_and_session_drawdown(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.015")
    monkeypatch.setenv("DAILY_DRAWDOWN_KILL_PCT", "0.03")
    settings = load_paper_settings()
    assert settings.risk.risk_per_trade_pct == pytest.approx(0.01)
    assert settings.risk.daily_drawdown_kill_pct == pytest.approx(0.02)


def test_session_phases_and_intraday_indicators():
    assert AutonomousRunner._phase(datetime_time(9, 44)) == "opening"
    assert AutonomousRunner._phase(datetime_time(11, 0)) == "regular"
    assert AutonomousRunner._phase(datetime_time(12, 30)) == "midday_lull"
    assert AutonomousRunner._phase(datetime_time(15, 15)) == "power_hour"
    bars = [
        Bar(datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i),
            100 + i, 101 + i, 99 + i, 100.5 + i, 1000)
        for i in range(40)
    ]
    assert _adx(bars) is not None
    assert _adx(bars) > 0


def test_opening_range_and_time_slot_relative_volume():
    state = SymbolState("SPY")
    session_day = datetime(2026, 10, 29, 9, 30, tzinfo=NY)
    for day_offset in range(8, 0, -1):
        prior = session_day - timedelta(days=day_offset)
        if prior.weekday() >= 5:
            continue
        state.agg.bars["1m"].append(Bar(
            prior.replace(minute=45).astimezone(timezone.utc), 100, 101, 99, 100.5, 100
        ))
    current_time = session_day.replace(minute=45)
    current = Bar(current_time.astimezone(timezone.utc), 100, 102, 100, 101.5, 200)
    state.agg.bars["1m"].append(current)
    assert _relative_volume(state, current) == pytest.approx(2)

    state.agg.bars["1m"].clear()
    for minute in range(15):
        state.agg.bars["1m"].append(Bar(
            (session_day + timedelta(minutes=minute)).astimezone(timezone.utc),
            100, 101, 99, 100.5, 100,
        ))
    state.agg.bars["1m"].append(Bar(
        (session_day + timedelta(minutes=15)).astimezone(timezone.utc),
        101, 102, 100.5, 101.8, 200,
    ))
    assert _opening_range_breakout(
        state, session_day + timedelta(minutes=16), Side.LONG
    )
    assert not _opening_range_breakout(
        state, session_day + timedelta(minutes=16), Side.SHORT
    )


def test_signal_uses_paper_limit_bracket_tranches_and_heartbeat_telemetry(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()
    heartbeat = tmp_path / "logs" / "runner_heartbeat.json"
    monkeypatch.setattr(runner_module, "HEARTBEAT_PATH", heartbeat)

    class FakeClient:
        def __init__(self):
            self.payloads = []

        def submit_bracket(self, payload):
            self.payloads.append(payload)
            return {
                "id": f"parent-{len(self.payloads)}",
                "legs": [{"id": f"stop-{len(self.payloads)}", "type": "stop"}],
            }

    client = FakeClient()
    runner = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    runner.symbols = ("SPY",)
    state = SymbolState("SPY")
    runner.states = {"SPY": state}
    for minute in range(40):
        state.agg.bars["5m"].append(Bar(
            datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * minute),
            100, 101, 99, 100.2, 1000,
        ))
    state.agg.bars["1m"].append(Bar(
        datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc),
        100, 101, 99, 100.5, 1000,
    ))
    runner.last_quotes["SPY"] = Quote(
        datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc), "SPY", 100, 100, 100.01, 100
    )

    class AcceptingAlpha:
        min_score = 0.7

        def evaluate(self, _state, _equity):
            return Decision(
                "SPY", Side.LONG, 0.9, True,
                checks=[Check("trend_d50", True, 0.2, "trend")],
                entry=100, stop=99, target=102, qty=100,
            )

    runner.alpha = AcceptingAlpha()
    runner.mtf_filter = SimpleNamespace(assess=lambda *_args: SimpleNamespace(
        allowed=True, reason="aligned", hourly={}, daily={}, intraday={},
    ))
    monkeypatch.setattr(runner, "_order_routing_authorized", lambda _asset_class: False)
    runner._submit_signal(
        "SPY", {"t": "2026-10-29T15:00:00+00:00", "c": 100.5},
        100_000, {}, {}, bar_time=datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc),
    )
    assert client.payloads == []
    locked = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert locked["telemetry"]["accepted"] is False
    assert "order routing locked" in locked["telemetry"]["reason"]

    monkeypatch.setattr(runner, "_order_routing_authorized", lambda _asset_class: True)
    runner._submit_signal(
        "SPY", {"t": "2026-10-29T15:00:00+00:00", "c": 100.5},
        100_000, {}, {}, bar_time=datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc),
    )
    assert len(client.payloads) == 2
    assert sum(int(payload["qty"]) for payload in client.payloads) == runner.managed["SPY"].qty
    assert all(payload["type"] == "limit" for payload in client.payloads)
    assert all(payload["order_class"] == "bracket" for payload in client.payloads)
    assert runner.managed["SPY"].stop_order_ids == ["stop-1", "stop-2"]
    record = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert record["telemetry"]["accepted"] is True
    assert record["telemetry"]["rvol"] is None

    class RejectingRiskGate:
        def authorize(self, _request):
            return {"approved": False, "reason": "independent risk limit"}

    runner.risk_gate = RejectingRiskGate()
    runner.managed.pop("SPY")
    runner._submit_signal(
        "SPY", {"t": "2026-10-29T15:00:00+00:00", "c": 100.5},
        100_000, {}, {}, bar_time=datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc),
    )
    assert len(client.payloads) == 2
    rejected = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert rejected["telemetry"]["accepted"] is False
    assert rejected["telemetry"]["reason"] == "independent risk limit"

    runner.risk_gate = ApprovingRiskGate()
    runner._submit_signal(
        "SPY", {"t": "2026-10-29T15:00:00+00:00", "c": 100.5},
        100_000, {}, {}, bar_time=datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc),
        entry_blocker="macro event window: CPI",
    )
    assert len(client.payloads) == 2
    rejected = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert rejected["telemetry"]["accepted"] is False
    assert "CPI" in rejected["telemetry"]["reason"]

    runner.alpha.evaluate = lambda *_args: Decision(
        "SPY", Side.LONG, 0.2, False, reject_reason="signal score below threshold",
    )
    runner._submit_signal(
        "SPY", {"t": "2026-10-29T15:00:00+00:00", "c": 100.5},
        100_000, {}, {}, bar_time=datetime(2026, 10, 29, 15, 1, tzinfo=timezone.utc),
    )
    no_signal = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert no_signal["telemetry"]["accepted"] is False
    assert no_signal["telemetry"]["required_score"] == pytest.approx(0.7)


def test_partial_take_profit_moves_all_runner_stops_to_breakeven(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()

    class FakeClient:
        def __init__(self):
            self.updated = []

        def orders(self, symbol):
            return [{"id": "entry-1", "status": "filled"},
                    {"id": "entry-2", "status": "filled"}]

        def open_orders(self, symbols):
            return [
                {"legs": [{"id": "stop-1", "type": "stop"}]},
                {"legs": [{"id": "stop-2", "type": "stop"}]},
            ]

        def update_order(self, order_id, payload):
            self.updated.append((order_id, payload))

    client = FakeClient()
    runner = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    state = SymbolState("SPY")
    state.last_price = 103
    for index in range(20):
        state.agg.bars["5m"].append(Bar(
            datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * index),
            100, 101, 99, 100, 1000,
        ))
    runner.states = {"SPY": state}
    runner.managed = {"SPY": runner_module.ManagedPosition(
        "SPY", Side.LONG, 50, 100, 98, 98, 103,
        stop_order_ids=["stop-1", "stop-2"], initial_qty=100,
        partial_qty=50, runner_qty=50, entry_order_ids=["entry-1", "entry-2"],
        entry_tranches_expected=2,
    )}
    runner._manage_stops({"SPY": {
        "avg_entry_price": "100", "qty": "50", "side": "long",
    }})
    assert runner.managed["SPY"].partial_taken
    assert {order for order, _ in client.updated} == {"stop-1", "stop-2"}
    assert all(payload["stop_price"] >= 100 for _, payload in client.updated)


def test_runner_settings_and_market_data_feed(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("WATCHLIST", "SPY,QQQ,NVDA")
    settings = load_paper_settings()
    assert settings.watchlist == ("SPY", "QQQ", "NVDA")

    client = AlpacaClient("key", "secret")
    assert client.paper_url == "https://paper-api.alpaca.markets"

    state = SymbolState("SPY")
    _feed_bar(state, {
        "t": datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc).isoformat(),
        "o": 100, "h": 102, "l": 99, "c": 101, "v": 400,
    })
    assert state.last_price == 101
    assert state.agg.bars["1m"][-1].volume == pytest.approx(400)


def test_dry_run_defers_dynamic_selection_when_market_is_closed(monkeypatch, capsys):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()

    class FakeClient:
        def clock(self):
            return {"is_open": False, "next_open": "2026-10-29T09:30:00-04:00",
                    "next_close": "2026-10-29T16:00:00-04:00"}

        def account(self):
            return {"status": "ACTIVE", "equity": "1000", "currency": "USD"}

        def positions(self):
            return []

        def open_orders(self, symbols):
            assert symbols == settings.watchlist
            return []

    monkeypatch.setattr(
        runner_module, "AutonomousRunner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selection should be deferred")),
    )
    monkeypatch.setattr(
        runner_module, "RiskGateClient", lambda: ApprovingRiskGate(),
    )
    assert runner_module.dry_run(FakeClient(), settings) == 2
    assert "deferred until premarket" in capsys.readouterr().out


def test_runner_closes_only_identified_filled_paper_test_position(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("WATCHLIST", "SPY,QQQ,NVDA")
    settings = load_paper_settings()

    class FakeClient:
        closed = []

        def positions(self):
            return [{"symbol": "SPY", "qty": "1", "side": "long"}]

        def orders(self, symbol):
            return [{
                "client_order_id": "apex-spy-paper-test-20261027T192638Z",
                "status": "filled", "filled_qty": "1", "side": "buy",
            }]

        def open_orders(self, symbols):
            return []

        def close_position(self, symbol):
            self.closed.append(symbol)
            return {"id": "close-order"}

    client = FakeClient()
    runner = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    assert runner._cleanup_identified_test_positions(market_open=True)
    assert client.closed == ["SPY"]


def test_runner_does_not_liquidate_unknown_existing_position(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    monkeypatch.setenv("WATCHLIST", "SPY,QQQ,NVDA")
    settings = load_paper_settings()

    class FakeClient:
        def positions(self):
            return [{"symbol": "SPY", "qty": "1", "side": "long"}]

        def orders(self, symbol):
            return []

    runner = AutonomousRunner(settings, FakeClient())
    with pytest.raises(RuntimeError, match="refusing to liquidate"):
        runner._cleanup_identified_test_positions(market_open=True)


def test_daily_loss_baseline_survives_runner_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    now = datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc)

    first = AutonomousRunner(settings, object())
    first._start_session(now, {"equity": "100000"})
    restarted = AutonomousRunner(settings, object())
    restarted._start_session(now, {"equity": "97000"})
    assert restarted.session_equity == 100000


def test_daily_drawdown_halt_persists_across_runner_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    monkeypatch.setattr(runner_module, "HEARTBEAT_PATH", tmp_path / "logs" / "heartbeat.json")
    now = datetime(2026, 10, 28, 14, 0, tzinfo=timezone.utc)

    class FakeClient:
        def __init__(self):
            self.cancelled = []
            self.closed = []

        def account(self):
            return {"equity": "97500"}

        def cancel_orders(self, symbols):
            self.cancelled.append(symbols)

        def close_position(self, symbol):
            self.closed.append(symbol)
            return {"id": "close-order"}

    client = FakeClient()
    first = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    first._start_session(now, {"equity": "100000"})
    assert first._kill_switch(now, {"SPY": {"symbol": "SPY"}})
    assert first.halted
    persisted = json.loads((tmp_path / "logs" / "session_guard.json").read_text())
    assert persisted["halted"] is True
    restarted = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    restarted._start_session(now, {"equity": "100000"})
    assert restarted.halted
    assert restarted._kill_switch(now, {})
    assert client.cancelled


def test_closing_bell_flatten_submits_cancels_and_closes(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("BROKER", "ALPACA")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-secret")
    settings = load_paper_settings()

    class FakeClient:
        def __init__(self):
            self.cancelled = []
            self.closed = []

        def cancel_orders(self, symbols):
            self.cancelled.append(symbols)

        def close_position(self, symbol):
            self.closed.append(symbol)
            return {"id": "close-order"}

    client = FakeClient()
    runner = AutonomousRunner(settings, client, risk_gate=ApprovingRiskGate())
    session_day = datetime(2026, 10, 28).date()
    runner._flatten_positions({"SPY": {"symbol": "SPY"}}, session_day)
    runner._flatten_positions({"SPY": {"symbol": "SPY"}}, session_day)
    assert client.cancelled == [("SPY",)]
    assert client.closed == ["SPY"]
    assert runner.flattened_day == session_day


def test_heartbeat_uses_recent_unix_timestamp_and_dashboard_reads_active(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "logs" / "runner_heartbeat.json"
    monkeypatch.setattr(runner_module, "HEARTBEAT_PATH", heartbeat_path)
    runner_module.write_heartbeat("active", "Processing market data.", ("SPY", "QQQ"))

    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert abs(time.time() - heartbeat["timestamp"]) < 2
    status = DashboardState(tmp_path, client_factory=lambda: None)._runner_status()
    assert status["status"] == "active"
    assert status["symbols"] == ["SPY", "QQQ"]


def test_heartbeat_retries_atomic_replace_after_transient_windows_lock(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "logs" / "runner_heartbeat.json"
    monkeypatch.setattr(runner_module, "HEARTBEAT_PATH", heartbeat_path)
    replace = runner_module.os.replace
    attempts = 0

    def replace_with_transient_lock(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary file lock")
        replace(source, destination)

    monkeypatch.setattr(runner_module.os, "replace", replace_with_transient_lock)
    runner_module.write_heartbeat("active", "Processing market data.", ("SPY",))

    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "active"
    assert heartbeat["symbols"] == ["SPY"]
    assert attempts == 3
    assert list(heartbeat_path.parent.glob("*.tmp")) == []


def test_paper_dashboard_snapshot_is_atomic_and_excludes_sensitive_account_fields(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    write_dashboard_paper_snapshot(
        {
            "status": "ACTIVE",
            "equity": "25000",
            "cash": "12000",
            "buying_power": "48000",
            "account_number": "DO-NOT-PUBLISH",
            "id": "DO-NOT-PUBLISH",
        },
        [{
            "symbol": "SPY",
            "qty": "2",
            "side": "long",
            "market_value": "1100",
            "unrealized_pl": "20",
            "secret": "DO-NOT-PUBLISH",
        }],
        [{
            "symbol": "QQQ",
            "side": "buy",
            "qty": "1",
            "status": "filled",
            "filled_at": "2026-11-06T15:00:00Z",
            "id": "DO-NOT-PUBLISH",
        }],
    )

    snapshot_path = tmp_path / "logs" / "paper_dashboard_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["mode"] == "PAPER"
    assert snapshot["paper_api_status"] == "connected"
    assert snapshot["account"] == {
        "status": "ACTIVE",
        "equity": "25000",
        "cash": "12000",
        "buying_power": "48000",
    }
    assert snapshot["positions"] == [{
        "symbol": "SPY",
        "qty": "2",
        "side": "long",
        "market_value": "1100",
        "unrealized_pl": "20",
    }]
    assert snapshot["orders"] == [{
        "symbol": "QQQ",
        "side": "buy",
        "qty": "1",
        "status": "filled",
        "filled_at": "2026-11-06T15:00:00Z",
    }]
    assert list(snapshot_path.parent.glob("*.tmp")) == []


def test_market_dashboard_snapshot_is_allowlisted_and_limits_candles(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "ROOT", tmp_path)
    write_dashboard_market_snapshot({
        "SPY": {
            "asset_class": "equity",
            "display_symbol": "SPY",
            "price": 600.0,
            "bid": 599.9,
            "ask": 600.1,
            "timestamp": "2026-11-07T14:30:00+00:00",
            "trend_50d": "above",
            "sma50": 590.0,
            "atr": 2.0,
            "atr_pct": 0.33,
            "bbwp": 12.0,
            "regime": "above 50-day trend",
            "api_key": "must-not-publish",
            "candles": [
                {"time": f"2026-11-07T14:{index % 60:02d}:00+00:00", "close": 600 + index}
                for index in range(125)
            ],
        },
    })

    snapshot_path = tmp_path / "logs" / "market_dashboard_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["symbols"]["SPY"]["price"] == 600.0
    assert "api_key" not in snapshot["symbols"]["SPY"]
    assert len(snapshot["symbols"]["SPY"]["candles"]) == 120
    assert "must-not-publish" not in snapshot_path.read_text(encoding="utf-8")
    assert list(snapshot_path.parent.glob("*.tmp")) == []


def test_closed_market_dashboard_refresh_is_read_only(monkeypatch):
    runner = AutonomousRunner.__new__(AutonomousRunner)
    calls = []

    class Client:
        def account(self):
            calls.append("account")
            return {"equity": "1000"}

        def positions(self):
            calls.append("positions")
            return []

        def recent_orders(self, limit):
            calls.append(("orders", limit))
            return []

    runner.client = Client()
    monkeypatch.setattr(
        runner_module,
        "write_dashboard_paper_snapshot",
        lambda account, positions, orders: calls.append("snapshot"),
    )

    runner._publish_dashboard_snapshot_while_closed()

    assert calls == ["account", "positions", ("orders", 20), "snapshot"]


def test_closed_market_market_snapshot_uses_read_only_recent_bars(monkeypatch):
    now = datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc)
    runner = AutonomousRunner.__new__(AutonomousRunner)
    runner.symbols = ("SPY",)
    runner.crypto_market_data = {}
    runner.dashboard_market_last_refresh_at = None
    calls = []

    def row(timestamp, close):
        return {
            "t": timestamp.isoformat(),
            "o": close - 0.1,
            "h": close + 0.2,
            "l": close - 0.2,
            "c": close,
            "v": 1000,
        }

    minute_rows = [
        row(now - timedelta(days=2, minutes=20 - index), 600 + index * 0.1)
        for index in range(20)
    ]
    daily_rows = [
        row(now - timedelta(days=index + 1), 590 + index * 0.1)
        for index in range(55)
    ]

    class Client:
        def bars(self, symbols, timeframe, _start, _end, limit):
            calls.append((symbols, timeframe, limit))
            return {"SPY": minute_rows if timeframe == "1Min" else daily_rows}

    runner.client = Client()
    published = {}
    monkeypatch.setattr(
        runner_module,
        "write_dashboard_market_snapshot",
        lambda market: published.update(market),
    )

    runner._publish_dashboard_market_snapshot_while_closed(now)

    assert calls == [
        (("SPY",), "1Min", 10000),
        (("SPY",), "1Day", 500),
    ]
    assert published["SPY"]["price"] == pytest.approx(601.9)
    assert published["SPY"]["trend_50d"] in {"above", "below"}
    assert len(published["SPY"]["candles"]) == 20
    assert runner.dashboard_market_last_refresh_at == now


def test_run_forever_keeps_transient_alpaca_failures_in_retry_state(monkeypatch):
    runner = AutonomousRunner.__new__(AutonomousRunner)
    runner.symbols = ("SPY",)
    runner.heartbeat_status = "active"
    runner.heartbeat_message = "Market open; runner cycle connected."
    calls = 0

    def run_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise runner_module.urllib.error.URLError("temporary network failure")
        raise asyncio.CancelledError

    runner.run_once = run_once
    heartbeat_writes = []
    monkeypatch.setattr(
        runner_module,
        "write_heartbeat",
        lambda status, message="", symbols=(), **kwargs: heartbeat_writes.append(
            (status, message)
        ),
    )
    delays = []

    class StopEvent:
        def is_set(self):
            return False

        def wait(self, delay):
            delays.append(delay)
            return False

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_forever(StopEvent()))

    assert calls == 2
    assert delays and delays[0] >= 1
    assert ("waiting", "Alpaca poll delayed; retrying next cycle.") in heartbeat_writes


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_alpaca_rate_limits_and_server_errors_are_transient(status):
    error = runner_module.AlpacaAPIError(status, "temporary")
    assert runner_module._is_transient_runner_error(error)


def test_run_forever_refreshes_heartbeat_each_loop(monkeypatch, tmp_path):
    heartbeat_path = tmp_path / "logs" / "runner_heartbeat.json"
    monkeypatch.setattr(runner_module, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(runner_module, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    runner = AutonomousRunner.__new__(AutonomousRunner)
    runner.symbols = ("SPY",)
    runner.heartbeat_status = "active"
    runner.heartbeat_message = "Market open; runner cycle connected."
    writes = []
    heartbeat_writes_ready = threading.Event()
    write = runner_module.write_heartbeat

    def record_write(status, message="", symbols=()):
        writes.append(time.time())
        if len(writes) >= 3:
            heartbeat_writes_ready.set()
        write(status, message, symbols)

    monkeypatch.setattr(runner_module, "write_heartbeat", record_write)

    def stop_after_cycle():
        assert heartbeat_writes_ready.wait(timeout=2)
        raise asyncio.CancelledError

    runner.run_once = stop_after_cycle
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_forever())

    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "active"
    assert heartbeat["message"] == "Market open; runner cycle connected."
    assert len(writes) >= 3
    status = DashboardState(tmp_path, client_factory=lambda: None)._runner_status()
    assert status["status"] == "active"


def test_dashboard_marks_stale_and_millisecond_heartbeats_disconnected(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    heartbeat = logs / "runner_heartbeat.json"
    heartbeat.write_text(json.dumps({
        "timestamp": (time.time() - 180) * 1000,
        "status": "active",
    }), encoding="utf-8")
    status = DashboardState(tmp_path, client_factory=lambda: None)._runner_status()
    assert status["status"] == "disconnected"
    assert "stale" in status["message"]
