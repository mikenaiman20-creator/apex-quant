from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

from backtest.data import NY, MinuteBar
from backtest.engine import BTConfig
from backtest.metrics import trade_equity_curve
from backtest.walkforward import walk_forward


def test_walk_forward_reports_unselected_windows_and_keeps_zero_exposure_days():
    days = []
    current = date(2025, 1, 6)
    while len(days) < 10:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    bars = {
        "SPY": [
            MinuteBar(
                datetime.combine(day, time(9, 30), tzinfo=NY),
                100, 101, 99, 100, 1000,
            )
            for day in days
        ]
    }

    def trade(day, pnl):
        ts = datetime.combine(day, time(10), tzinfo=NY)
        return SimpleNamespace(
            entry_ts=ts,
            exit_ts=ts + timedelta(minutes=1),
            pnl_net=pnl,
            pnl_gross=pnl,
            costs=0.0,
            r_multiple=pnl / 100,
        )

    results = {
        (1.0,): [
            trade(days[0], 10),
            trade(days[1], 10),
            trade(days[2], 100),
        ]
    }
    folds, oos, _ = walk_forward(
        bars,
        BTConfig(initial_equity=5000, warmup_days=0),
        grid={"min_signal_score": [1.0]},
        train_days=2,
        test_days=2,
        min_train_trades=2,
        results=results,
    )

    assert len(folds) == 4
    assert [fold["selected"] for fold in folds] == [True, False, False, False]
    assert len(oos) == 1
    oos_days = [day for fold in folds for day in fold["test_days"]]
    curve = trade_equity_curve(oos, 5000, calendar_days=oos_days)
    assert [day for day, _ in curve] == days[2:]
    assert [equity for _, equity in curve] == [5100, 5100, 5100, 5100, 5100, 5100, 5100, 5100]
