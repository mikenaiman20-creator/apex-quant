import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.data_pipeline import Bar
from server.evolution import (
    ArsenalEvolution, approve_candidate, deflated_sharpe, evolve_arsenal,
    register_trials, stage_candidate,
)
from server.healer import diagnose_exception
from server.researcher import STRATEGY_CATALOG, StrategyResearcher, classify_regime
from server.event_guard import MacroEventGuard
import server.unified_master as master_module
import backtest.holdout as holdout_module
from backtest.engine import BTConfig


def _bars(count=80):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for index in range(count):
        price += 0.25 if index % 8 else -0.05
        bars.append(Bar(start + timedelta(days=index), price - 0.1, price + 0.8,
                        price - 0.8, price, 1000))
    return bars


def _closed_bracket_orders(count, win_rate=0.7):
    orders = []
    winners = round(count * win_rate)
    for index in range(count):
        day = datetime(2026, 1, 5 + index % 3, 15, 0, tzinfo=timezone.utc)
        exit_price = 102.0 if index < winners else 99.5
        orders.append({
            "id": f"entry-{index}",
            "symbol": "SPY",
            "status": "filled",
            "side": "buy",
            "filled_qty": "10",
            "filled_avg_price": "100",
            "filled_at": day.isoformat(),
            "legs": [{
                "id": f"exit-{index}",
                "symbol": "SPY",
                "status": "filled",
                "side": "sell",
                "filled_qty": "10",
                "filled_avg_price": str(exit_price),
                "filled_at": (day + timedelta(minutes=30)).isoformat(),
            }],
        })
    return orders


def _backtest_trial_samples(count, win_rate):
    return [{"win_rate": win_rate, "trade_count": 20} for _ in range(count)]


def test_live_readiness_uses_closed_friction_adjusted_paper_trades():
    readiness = master_module.calculate_live_readiness(
        _closed_bracket_orders(50),
        researcher={},
        total_trials=142,
        risk_settings=SimpleNamespace(
            risk_per_trade_pct=0.005,
            daily_drawdown_kill_pct=0.02,
        ),
        recent_trial_metrics=_backtest_trial_samples(142, 0.7),
        api_rejection_breaker_enabled=True,
    )
    breakdown = readiness["readiness_breakdown"]
    assert readiness["readiness_score"] == 100
    assert readiness["readiness_status"] == "FULL DEPLOYMENT"
    assert breakdown["execution_sample_size"]["score"] == 25
    assert breakdown["execution_sample_size"]["market_sessions"] == 3
    assert breakdown["friction_adjusted_expectancy"]["expectancy_per_trade_usd"] > 2
    assert breakdown["friction_adjusted_expectancy"]["profit_factor"] > 1.5
    assert breakdown["walk_forward_stability"]["score"] == 25
    assert breakdown["missing_criteria"] == []
    assert "does not authorize" in breakdown["paper_only_notice"]


def test_live_readiness_pairs_standalone_partial_exits_into_one_closed_trade():
    entry_time = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    orders = [
        {
            "id": "entry",
            "symbol": "SPY",
            "status": "filled",
            "side": "buy",
            "filled_qty": "10",
            "filled_avg_price": "100",
            "filled_at": entry_time.isoformat(),
        },
        {
            "id": "partial-exit-a",
            "symbol": "SPY",
            "status": "filled",
            "side": "sell",
            "filled_qty": "5",
            "filled_avg_price": "101",
            "filled_at": (entry_time + timedelta(minutes=10)).isoformat(),
        },
        {
            "id": "partial-exit-b",
            "symbol": "SPY",
            "status": "filled",
            "side": "sell",
            "filled_qty": "5",
            "filled_avg_price": "102",
            "filled_at": (entry_time + timedelta(minutes=20)).isoformat(),
        },
    ]
    metrics = master_module._closed_trade_metrics(orders)
    assert metrics["closed_trade_count"] == 1
    assert metrics["sessions"] == 1
    assert metrics["expectancy"] > 0


def test_live_readiness_fails_closed_for_missing_evidence_and_breaker():
    readiness = master_module.calculate_live_readiness(
        orders=[],
        researcher={},
        total_trials=0,
        risk_settings=None,
    )
    breakdown = readiness["readiness_breakdown"]
    assert readiness["readiness_score"] == 0
    assert readiness["readiness_status"] == "UNSAFE"
    assert breakdown["execution_sample_size"]["score"] == 0
    assert breakdown["risk_guardrails"]["score"] == 0
    assert any("API rejection breaker" in item for item in breakdown["missing_criteria"])
    assert any("142 research trials with recorded walk-forward win rates" in item
               for item in breakdown["missing_criteria"])


def test_missing_api_rejection_breaker_prevents_full_deployment_status():
    readiness = master_module.calculate_live_readiness(
        _closed_bracket_orders(50),
        researcher={},
        total_trials=142,
        risk_settings=SimpleNamespace(
            risk_per_trade_pct=0.005,
            daily_drawdown_kill_pct=0.02,
        ),
        recent_trial_metrics=_backtest_trial_samples(142, 0.7),
    )
    assert readiness["readiness_score"] == 95
    assert readiness["readiness_status"] == "LIVE CANDIDATE"


def test_live_readiness_applies_sample_gate_and_stability_deviation_penalty():
    readiness = master_module.calculate_live_readiness(
        _closed_bracket_orders(10),
        researcher={},
        total_trials=142,
        risk_settings=SimpleNamespace(
            risk_per_trade_pct=0.01,
            daily_drawdown_kill_pct=0.03,
        ),
        recent_trial_metrics=_backtest_trial_samples(142, 0.91),
    )
    breakdown = readiness["readiness_breakdown"]
    assert breakdown["execution_sample_size"]["score"] == 0
    assert breakdown["execution_sample_size"]["market_sessions"] == 3
    assert breakdown["risk_guardrails"]["score"] == 20
    assert breakdown["walk_forward_stability"]["score"] == 20
    assert breakdown["walk_forward_stability"]["paper_win_rate"] == 0.7
    assert readiness["readiness_status"] == "PAPER STABLE"


def test_regime_classifier_calculates_indicators_and_rejects_short_history():
    assert classify_regime("SPY", _bars(20)) is None
    result = classify_regime("SPY", _bars())
    assert result is not None
    assert result.symbol == "SPY"
    assert result.label in {
        "high_volatility_trending", "trending",
        "low_volatility_choppy", "mean_reverting", "high_volatility_choppy",
    }
    assert result.adx >= 0
    assert result.atr > 0
    assert result.atr_pct > 0


def test_research_cycle_persists_equity_only_regime_advice(tmp_path: Path):
    bars = _bars()

    class FakeClient:
        def bars(self, symbols, timeframe, start, end, limit):
            assert timeframe in {"1Day", "1Hour"}
            return {symbol: [{
                "t": item.start.isoformat(), "o": item.open, "h": item.high,
                "l": item.low, "c": item.close, "v": item.volume,
            } for item in bars] for symbol in symbols}

    state = StrategyResearcher(FakeClient(), lambda: ("SPY",), tmp_path).research_once()
    assert state["execution_enabled"] is False
    assert state["asset_class_scope"] == ["equities"]
    assert "SPY" in state["regimes"]
    assert "1h" in state["timeframe_context"]["SPY"]
    assert "1d" in state["timeframe_context"]["SPY"]
    assert any(item["asset_class"] == "crypto" for item in state["catalog"])
    persisted = json.loads((tmp_path / "logs" / "research_state.json").read_text())
    assert persisted["regimes"]["SPY"]["symbol"] == "SPY"


def test_researcher_runs_local_discovery_after_market_context_failure(tmp_path: Path):
    class StopAfterCycle:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _seconds):
            self.stopped = True
            return True

    researcher = StrategyResearcher(
        object(), lambda: ("SPY",), tmp_path, interval_seconds=0,
    )

    def fail_market_context():
        raise RuntimeError("paper data endpoint unavailable")

    researcher.research_once = fail_market_context
    researcher.run(StopAfterCycle())

    state = json.loads(
        (tmp_path / "logs" / "research_agent_state.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (tmp_path / "logs" / "strategy_trial_ledger.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (tmp_path / "logs" / "strategy_audit_report.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "waiting_for_data"
    assert "paper data endpoint unavailable" in state["context_error"]
    assert state["execution_enabled"] is False
    assert ledger["latest_activity"]["status"] == "waiting_for_data"
    assert audit["research_activity"]["status"] == "waiting_for_data"


def test_researcher_repairs_legacy_trial_ledger_before_discovery(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    ledger_path = logs / "strategy_trial_ledger.json"
    ledger_path.write_text('{"total_trials": 9}', encoding="utf-8")

    state = StrategyResearcher(
        object(), lambda: ("SPY",), tmp_path,
    ).discovery_once()

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert state["status"] == "waiting_for_data"
    assert state["cumulative_trials"] == 9
    assert ledger["cycles"]["legacy_import"] == 9
    assert ledger["total_trials"] == 9
    assert ledger["latest_activity"]["status"] == "waiting_for_data"


def test_strategy_promotions_require_all_gates_and_quarantine_is_immediate():
    qualified = {
        "id": "qualified", "name": "Qualified", "status": "research_only",
        "paper_trades": 30, "sharpe": 1.51, "profit_factor": 1.31,
        "max_drawdown_pct": 4.99,
    }
    under_sampled = {**qualified, "id": "too-few", "paper_trades": 29}
    losing = {
        "id": "losing", "status": "promoted", "recent_outcomes_r": [-1, -0.5, -0.2],
    }
    arsenal, messages = evolve_arsenal({
        "strategies": [qualified, under_sampled, losing],
    })
    statuses = {item["id"]: item["status"] for item in arsenal["strategies"]}
    assert statuses == {
        "qualified": "promoted", "too-few": "research_only", "losing": "quarantined",
    }
    assert len(messages) == 2


def test_alpha_decay_quarantines_strategy():
    arsenal, messages = evolve_arsenal({"strategies": [{
        "id": "decaying", "status": "research_only",
        "recent_expectancy_r": -0.01, "baseline_expectancy_r": 0.2,
    }]})
    assert arsenal["strategies"][0]["status"] == "quarantined"
    assert messages


def test_arsenal_evolution_seeds_and_persists_safely(tmp_path: Path):
    path = tmp_path / "data" / "strategy_arsenal.json"
    arsenal = ArsenalEvolution(path).run_once()
    assert len(arsenal["strategies"]) == 5
    assert all(item["status"] == "research_only" for item in arsenal["strategies"])
    assert json.loads(path.read_text())["version"] == 1


def test_trial_ledger_is_cumulative_and_deduplicates_cycles(tmp_path: Path):
    ledger = tmp_path / "trial-ledger.json"
    assert register_trials("cycle-a", 16, ledger) == 16
    assert register_trials("cycle-a", 16, ledger) == 16
    assert register_trials("cycle-b", 16, ledger) == 32
    persisted = json.loads(ledger.read_text(encoding="utf-8"))
    assert persisted["num_trials"] == 32
    assert persisted["total_trials"] == 32


def test_trial_counter_persists_failed_and_historical_sweeps(tmp_path: Path):
    ledger = tmp_path / "logs" / "trial-ledger.json"
    assert register_trials("sweep-before-failure", 4, ledger) == 4
    assert register_trials("later-sweep", 7, ledger) == 11
    assert register_trials("later-sweep", 7, ledger) == 11
    restored = json.loads(ledger.read_text(encoding="utf-8"))
    assert restored["cycles"] == {
        "sweep-before-failure": 4,
        "later-sweep": 7,
    }
    assert restored["num_trials"] == 11


def test_trial_registration_fails_closed_on_corrupt_history(tmp_path: Path):
    ledger = tmp_path / "trial-ledger.json"
    original = "{ invalid json"
    ledger.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to reset trial history"):
        register_trials("new-cycle", 16, ledger)
    assert ledger.read_text(encoding="utf-8") == original


def test_trial_ledger_retains_only_142_unique_walk_forward_win_rate_samples(tmp_path: Path):
    ledger_path = tmp_path / "trial-ledger.json"
    records = [{"win_rate": 0.6, "trade_count": 10} for _ in range(150)]
    assert register_trials("cycle-a", 150, ledger_path, win_rate_records=records) == 150
    assert register_trials("cycle-a", 150, ledger_path, win_rate_records=records) == 150
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["recent_win_rates"]) == 142
    assert all(sample["cycle_id"] == "cycle-a" for sample in ledger["recent_win_rates"])


def test_deflated_sharpe_hurdle_increases_with_variants():
    one = deflated_sharpe(3.0, observations=300, n_trials=1)
    many = deflated_sharpe(3.0, observations=300, n_trials=100)
    assert one["required_sharpe"] < many["required_sharpe"]
    assert one["passed"] is True
    assert many["passed"] is False


def test_candidate_requires_holdout_before_staging_and_human_approval(tmp_path: Path):
    candidate = {
        "id": "candidate-2026-01",
        "sharpe": 5.0,
        "observations": 300,
        "paper_trades": 40,
        "profit_factor": 2.0,
        "max_drawdown_pct": 3.0,
        "n_trials": 1,
        "cpcv": {
            "method": "CPCV-CSCV",
            "pbo": 0.1,
            "pbo_passed": True,
            "valid_splits": 20,
            "n_trials": 1,
            "dsr": {
                "raw_sharpe": 5.0,
                "observations": 300,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
        },
        "params": {"min_signal_score": 0.75},
        "forward_paper_expectancy_r": None,
    }
    audit_path = tmp_path / "logs" / "audit.json"
    staging_dir = tmp_path / "staging"
    blocked = stage_candidate(
        candidate, {"available": False, "passed": False, "reason": "missing"},
        audit_path, staging_dir,
    )
    assert blocked["status"] == "blocked"
    assert not (staging_dir / "candidate-2026-01.json").exists()

    staged = stage_candidate(
        candidate, {"available": True, "passed": True, "reason": "passed"},
        audit_path, staging_dir,
    )
    assert staged["status"] == "staged_candidate"
    assert staged["cpcv_gate_passed"] is True
    assert staged["pbo"] == 0.1
    assert (staging_dir / "candidate-2026-01.json").exists()
    active = tmp_path / "active.json"
    approved = approve_candidate(
        "candidate-2026-01", staging_dir, active,
    )
    assert approved["human_approved"] is True
    assert approved["status"] == "approved_for_paper"
    assert json.loads(active.read_text())["candidate_id"] == "candidate-2026-01"


def test_candidate_without_cpcv_evidence_cannot_be_staged(tmp_path: Path):
    candidate = {
        "id": "missing-cpcv",
        "sharpe": 5.0,
        "observations": 300,
        "paper_trades": 40,
        "profit_factor": 2.0,
        "max_drawdown_pct": 3.0,
        "n_trials": 1,
    }
    audit = stage_candidate(
        candidate,
        {"available": True, "passed": True, "reason": "passed"},
        tmp_path / "audit.json",
        tmp_path / "staging",
    )
    assert audit["status"] == "blocked"
    assert audit["cpcv_gate_passed"] is False
    assert not (tmp_path / "staging" / "missing-cpcv.json").exists()


def test_final_holdout_is_consumed_once_per_dataset(tmp_path: Path, monkeypatch):
    directory = tmp_path / "final_holdout"
    directory.mkdir()
    (directory / "SPY.csv").write_text("immutable test fixture", encoding="utf-8")
    calls = []
    monkeypatch.setattr(holdout_module, "load_dir", lambda *_args: {"SPY": [object()]})
    monkeypatch.setattr(
        holdout_module, "run_backtest",
        lambda *_args: calls.append("run") or type("Result", (), {"trades": []})(),
    )
    monkeypatch.setattr(
        holdout_module, "trade_equity_curve", lambda *_args: [(datetime.now(timezone.utc).date(), 5000)],
    )
    monkeypatch.setattr(holdout_module, "summarize", lambda *_args: {
        "n": 30, "sharpe_daily_ann": 2.0, "profit_factor": 1.5,
        "max_dd_pct": 4.0, "mean_r": 0.1,
    })
    ledger = tmp_path / "consumption.json"
    first = holdout_module.validate_final_holdout(
        ["SPY"], BTConfig(), {}, "candidate-a", directory, ledger,
    )
    same = holdout_module.validate_final_holdout(
        ["SPY"], BTConfig(), {}, "candidate-a", directory, ledger,
    )
    different = holdout_module.validate_final_holdout(
        ["SPY"], BTConfig(), {}, "candidate-b", directory, ledger,
    )
    assert first["passed"] is True
    assert same == first
    assert different["passed"] is False
    assert "already consumed" in different["reason"]
    assert calls == ["run"]


def test_healer_diagnosis_is_structured_and_strategy_catalog_is_advisory():
    try:
        raise ValueError("sample component failure")
    except ValueError as exc:
        diagnosis = diagnose_exception("research", exc)
    assert diagnosis["component"] == "research"
    assert diagnosis["exception"] == "ValueError"
    assert "sample component failure" in diagnosis["traceback"]
    assert STRATEGY_CATALOG
    assert all(strategy["execution_enabled"] is False for strategy in STRATEGY_CATALOG)


def test_unified_dry_run_delegates_to_paper_runner_without_starting_components(monkeypatch):
    observed = []
    monkeypatch.setattr(master_module, "runner_main", lambda args: observed.append(args) or 2)
    assert master_module.main(["--dry-run"]) == 2
    assert observed == [["--dry-run"]]


def test_macro_calendar_blocks_within_configured_window(tmp_path: Path):
    event_time = datetime(2026, 10, 29, 13, 30, tzinfo=timezone.utc)
    calendar = tmp_path / "macro_events.json"
    calendar.write_text(json.dumps({"events": [{
        "name": "CPI", "impact": "high", "timestamp": event_time.isoformat(),
    }]}), encoding="utf-8")
    guard = MacroEventGuard(calendar, buffer_minutes=15)
    blocked = guard.check(event_time - timedelta(minutes=10))
    assert blocked.blocked
    assert blocked.event == "CPI"
    assert not guard.check(event_time - timedelta(minutes=16)).blocked


def test_macro_guard_detects_broad_index_volatility_spike(tmp_path: Path):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=20)
    bars = [
        Bar(start + timedelta(minutes=index), 100, 101, 99, 100, 1000)
        for index in range(20)
    ]
    bars[-1].close = bars[-1].open * 1.02
    calendar = tmp_path / "macro_events.json"
    calendar.write_text('{"events":[]}', encoding="utf-8")
    result = MacroEventGuard(calendar, volatility_spike_pct=0.01).check(
        now, bars
    )
    assert result.blocked
    assert "SPY moved" in result.reason
    cooling = MacroEventGuard(calendar, volatility_spike_pct=0.01)
    cooling.check(now, bars)
    assert cooling.check(now + timedelta(minutes=5)).blocked
    assert not cooling.check(now + timedelta(minutes=16)).blocked
