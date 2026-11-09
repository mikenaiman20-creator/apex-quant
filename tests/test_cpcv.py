from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest

from backtest.cpcv import evaluate_cpcv, generate_synthetic_calibration_signal
from backtest.data import NY, MinuteBar
from backtest.engine import BTConfig
from backtest.metrics import deflated_sharpe_from_returns
from backtest.strategies import opening_range_breakout


def _fixture():
    first = date(2025, 1, 2)
    days = []
    current = first
    while len(days) < 24:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    bars = []
    trades_a, trades_b = [], []
    for index, day in enumerate(days):
        start = datetime.combine(day, time(9, 30), tzinfo=NY)
        bars.append(MinuteBar(start, 100, 101, 99, 100, 1000))
        pnl_a = 20.0 if index % 4 < 2 else -10.0
        pnl_b = -10.0 if index % 4 < 2 else 20.0
        for trades, pnl in ((trades_a, pnl_a), (trades_b, pnl_b)):
            trades.append(SimpleNamespace(
                entry_ts=start,
                exit_ts=start + timedelta(minutes=1),
                pnl_net=pnl,
            ))
    return {"SPY": bars}, {(0.65,): trades_a, (0.75,): trades_b}


def _mock_candidate_replays(monkeypatch, results):
    import backtest.cpcv as cpcv

    def replay(_data, config, *, signal_fn=None, reset_equity_days):
        assert reset_equity_days
        assert signal_fn is None or signal_fn is opening_range_breakout
        key = (config.min_signal_score,)
        return SimpleNamespace(trades=results[key])

    monkeypatch.setattr(cpcv, "run_backtest", replay)


def _evaluate(data, results, **kwargs):
    return evaluate_cpcv(
        data,
        BTConfig(initial_equity=5000, warmup_days=0),
        results,
        n_trials=kwargs.pop("n_trials", 37),
        grid={"min_signal_score": [0.65, 0.75]},
        workers=1,
        **kwargs,
    )


def test_cpcv_reports_pbo_dsr_and_full_trial_count(monkeypatch):
    data, results = _fixture()
    _mock_candidate_replays(monkeypatch, results)
    report = _evaluate(data, results, group_count=6, embargo_days=1)
    assert report["method"] == "CPCV-CSCV"
    assert report["splits"] == 20
    assert report["valid_splits"] == 20
    assert 0 <= report["pbo"] <= 1
    assert report["n_trials"] == 37
    assert report["trials_in_run"] == 2
    assert report["dsr"]["n_trials"] == 37
    assert report["dsr"]["observations"] == 24
    assert len(report["split_results"]) == 20
    assert set(report["split_results"][0]) == {
        "split", "test_groups", "winner_params", "train_sharpe",
        "test_sharpe", "test_rank_fraction",
    }
    assert isinstance(report["passed"], bool)


def test_cpcv_replays_custom_research_strategy(monkeypatch):
    data, results = _fixture()
    _mock_candidate_replays(monkeypatch, results)
    report = _evaluate(
        data, results, n_trials=2, signal_fn=opening_range_breakout,
    )
    assert report["valid_splits"] == 20


def test_cpcv_purges_trades_that_cross_group_boundaries(monkeypatch):
    data, results = _fixture()
    _mock_candidate_replays(monkeypatch, results)
    days = [bar.ts.date() for bar in data["SPY"]]
    cross_boundary = SimpleNamespace(
        entry_ts=datetime.combine(days[3], time(15, 59), tzinfo=NY),
        exit_ts=datetime.combine(days[4], time(9, 31), tzinfo=NY),
        pnl_net=1_000_000.0,
    )
    results[(0.65,)].append(cross_boundary)
    report = _evaluate(data, results, n_trials=2, group_count=6, embargo_days=1)
    assert report["valid_splits"] == 20
    assert report["oos_observations"] == 24
    assert report["dsr"]["raw_sharpe"] != float("inf")


def test_cpcv_requires_even_groups_and_fails_closed_on_short_history(monkeypatch):
    data, results = _fixture()
    _mock_candidate_replays(monkeypatch, results)
    with pytest.raises(ValueError, match="even integer"):
        _evaluate(data, results, n_trials=2, group_count=5)

    short_data = {"SPY": data["SPY"][:3]}
    report = _evaluate(short_data, results, n_trials=2, group_count=6)
    assert report["pbo"] is None
    assert report["dsr"]["passed"] is False
    assert report["passed"] is False


def test_dsr_probability_declines_when_the_trial_count_increases():
    returns = [0.002, -0.001, 0.003, -0.0005, 0.0015, -0.001] * 30
    one_trial = deflated_sharpe_from_returns(returns, n_trials=1)
    many_trials = deflated_sharpe_from_returns(returns, n_trials=100)
    assert one_trial["raw_sharpe"] == pytest.approx(many_trials["raw_sharpe"])
    assert one_trial["adjusted_sharpe"] > many_trials["adjusted_sharpe"]
    assert one_trial["required_sharpe"] < many_trials["required_sharpe"]


def test_synthetic_modest_edge_passes_cpcv_and_dsr_calibration(monkeypatch):
    import random
    import statistics

    import backtest.cpcv as cpcv

    calibration = generate_synthetic_calibration_signal()
    assert statistics.correlation(calibration.signal, calibration.net_returns) == pytest.approx(
        0.15, abs=1e-10,
    )
    realized_sharpe = deflated_sharpe_from_returns(
        calibration.net_returns, n_trials=2,
    )
    assert realized_sharpe["raw_sharpe"] == pytest.approx(0.50, abs=1e-10)
    assert sum(
        gross - net for gross, net in zip(
            calibration.gross_returns, calibration.net_returns,
        )
    ) / len(calibration.net_returns) == pytest.approx(
        calibration.transaction_cost_per_observation,
    )

    start_day = date(1980, 1, 1)
    days = []
    current = start_day
    while len(days) < len(calibration.net_returns):
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    bars = [
        MinuteBar(
            datetime.combine(day, time(9, 30), tzinfo=NY),
            100, 100.01, 99.99, 100, 1000,
        )
        for day in days
    ]
    base = BTConfig(initial_equity=5000, warmup_days=0)

    def make_trades(returns):
        return [
            SimpleNamespace(
                entry_ts=bar.ts,
                exit_ts=bar.ts + timedelta(minutes=1),
                pnl_net=value * base.initial_equity,
            )
            for bar, value in zip(bars, returns)
        ]

    rng = random.Random(20261110)
    decoy = [rng.gauss(0, 0.01) for _ in days]
    decoy_mean = sum(decoy) / len(decoy)
    decoy_std = (
        sum((value - decoy_mean) ** 2 for value in decoy) / (len(decoy) - 1)
    ) ** 0.5
    decoy = [(value - decoy_mean) / decoy_std * 0.01 for value in decoy]
    candidates = {
        (0.65,): make_trades(calibration.net_returns),
        (0.75,): make_trades(decoy),
    }

    def replay(_data, config, *, signal_fn=None, reset_equity_days):
        assert reset_equity_days
        assert signal_fn is None
        return SimpleNamespace(trades=candidates[(config.min_signal_score,)])

    monkeypatch.setattr(cpcv, "run_backtest", replay)
    report = evaluate_cpcv(
        {"SPY": bars},
        base,
        candidates,
        n_trials=2,
        grid={"min_signal_score": [0.65, 0.75]},
        group_count=6,
        embargo_days=1,
        workers=1,
    )

    assert report["pbo"] < 0.20
    assert report["pbo_passed"] is True
    assert report["dsr"]["raw_sharpe"] == pytest.approx(0.50, abs=0.15)
    assert report["dsr"]["adjusted_sharpe"] >= 0.95
    assert report["dsr"]["passed"] is True
    assert report["passed"] is True
