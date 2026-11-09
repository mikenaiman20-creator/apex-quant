"""Mechanics tests for the backtester: fills, stop priority, gaps, kill-switch, costs, no-lookahead, metrics."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import pytest

from backtest.data import NY, MinuteBar, data_quality, normalize, synthetic_bars
from backtest.engine import BTConfig, run_backtest
from backtest.metrics import bootstrap_ci, summarize, trade_equity_curve
from backtest.strategies import opening_range_breakout
from backtest.strategies import short_term_zscore_mean_reversion
from backtest.walkforward import run_parameter_grid
from config.settings import RiskLimits
from engine.data_pipeline import SymbolState, Trade
from engine.strategy import Decision, Side

DAY = date(2025, 3, 3)  # a Monday


def day_bars(d=DAY, price=lambda i: 100.0, overrides=None, vol=1000.0):
    overrides = overrides or {}
    base = datetime.combine(d, time(9, 30), tzinfo=NY)
    out = []
    for i in range(390):
        ts = (base + timedelta(minutes=i)).astimezone(timezone.utc)
        if i in overrides:
            o, h, l, c = overrides[i]
        else:
            o = h = l = c = price(i)
        out.append(MinuteBar(ts, o, h, l, c, vol))
    return out


def orb_day_bars(day, breakout=None, breakout_volume=1000.0, regular_range=0.04):
    base = datetime.combine(day, time(9, 30), tzinfo=NY)
    bars = []
    for index in range(390):
        ts = (base + timedelta(minutes=index)).astimezone(timezone.utc)
        if index < 15:
            o, h, l, c, volume = 100.0, 100.1, 99.9, 100.0, 1000.0
        elif index == 20 and breakout is not None:
            o, h, l, c = breakout
            volume = breakout_volume
        else:
            half_range = regular_range / 2
            o, h, l, c, volume = (
                100.0, 100.0 + half_range, 100.0 - half_range, 100.0, 1000.0,
            )
        bars.append(MinuteBar(ts, o, h, l, c, volume))
    return bars


def cfg(**kw):
    base = dict(initial_equity=5000.0, warmup_days=0, entry_start=time(9, 30), slippage_bps=1.0,
                spread_bps=0.0, commission_per_share=0.0035, commission_min=0.35,
                risk=RiskLimits(risk_per_trade_pct=0.01, max_position_notional_pct=1.0))
    base.update(kw)
    return BTConfig(**base)


def long_once(stop):
    fired = {"n": 0}

    def fn(st, eq):
        if fired["n"]:
            return None
        fired["n"] += 1
        px = st.last_price
        return Decision(st.symbol, Side.LONG, 1.0, True, [], px, stop, px + 10, 1)
    return fn


def test_fill_at_next_open_plus_slippage_and_eod_flat():
    bars = day_bars(overrides={1: (101, 101, 101, 101)}, price=lambda i: 100.0 if i != 1 else 101)
    # after bar 1 prices return to 100; stop 99 not hit; target far away
    r = run_backtest({"X": bars}, cfg(), long_once(99.0))
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.entry == pytest.approx(101 * (1 + 1e-4), abs=1e-9)
    assert t.exit_reason == "eod"
    assert t.exit == pytest.approx(100 * (1 - 1e-4), abs=1e-9)
    assert t.exit_ts.astimezone(NY).time() == time(15, 55)
    assert t.pnl_net < 0
    assert t.pnl_net == pytest.approx(t.pnl_gross - t.costs)
    assert t.slippage_cost > 0


def test_backtester_applies_explicit_bid_ask_spread_to_each_fill():
    bars = day_bars(overrides={1: (101, 101, 101, 101)})
    result = run_backtest({"X": bars}, cfg(spread_bps=4.0), long_once(99.0))

    assert result.trades[0].entry == pytest.approx(101 * (1 + 3e-4))
    assert result.trades[0].exit == pytest.approx(100 * (1 - 3e-4))


def test_backtester_flattens_at_last_session_bar_even_if_eod_bar_is_missing():
    first = day_bars()[:385]
    second = day_bars(d=DAY + timedelta(days=1))
    fired = False

    def signal(st, _equity):
        nonlocal fired
        if (
            not fired
            and st.agg.bars["1m"][-1].start.astimezone(NY).time() == time(15, 53)
        ):
            fired = True
            return Decision(st.symbol, Side.LONG, 1.0, True, [], 100, 90, 120, 1)
        return None

    result = run_backtest(
        {"SPY": first + second}, cfg(entry_end=time(15, 54)), signal,
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "eod"
    assert result.trades[0].exit_ts.date() == first[-1].ts.date()


def _zscore_states(*, panic=False):
    spy, qqq = SymbolState("SPY"), SymbolState("QQQ")
    qqq.market_state = spy
    base = datetime.combine(date(2025, 3, 3), time(9, 30), tzinfo=NY)
    for index in range(81):
        ts = base + timedelta(minutes=index)
        if panic and index >= 61:
            spy_price = 100.0 - (index - 60) * 0.09
        else:
            spy_price = 100.0 + (0.02 if index % 2 else 0.0)
        spy.on_trade(Trade(ts, "SPY", spy_price, 1000, 1))

        if index < 61:
            qqq_price = 100.0
        elif index < 80:
            qqq_price = 99.8 if index % 2 else 100.2
        else:
            qqq_price = 99.4
        qqq.on_trade(Trade(ts, "QQQ", qqq_price, 1000, 1))
    return spy, qqq


def test_zscore_mean_reversion_fades_extreme_and_targets_rolling_mean():
    _, qqq = _zscore_states()
    decision = short_term_zscore_mean_reversion(qqq, 5000)

    assert decision is not None
    assert decision.side is Side.LONG
    assert decision.target > decision.entry
    assert decision.stop < decision.entry
    assert any(check.name == "spy_market_regime" for check in decision.checks)


def test_zscore_mean_reversion_blocks_spy_panic_regime():
    _, qqq = _zscore_states(panic=True)

    assert short_term_zscore_mean_reversion(qqq, 5000) is None


def test_stop_beats_target_when_bar_touches_both():
    bars = day_bars(overrides={1: (101, 101, 101, 101), 3: (101, 120, 90, 101)})
    r = run_backtest({"X": bars}, cfg(), long_once(99.0))
    t = r.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit == pytest.approx(99 * (1 - 1e-4), abs=1e-9)


def test_gap_through_stop_fills_at_open():
    bars = day_bars(overrides={1: (101, 101, 101, 101), 3: (95, 96, 94, 95)})
    t = run_backtest({"X": bars}, cfg(), long_once(99.0)).trades[0]
    assert t.exit_reason == "stop_gap"
    assert t.exit == pytest.approx(95 * (1 - 1e-4), abs=1e-9)


def test_target_fills_at_limit_without_slippage():
    # fill ~101.0101, stop 99 -> dist ~2.01 -> target ~ 105.03 at rr=2
    bars = day_bars(overrides={1: (101, 101, 101, 101), 3: (101, 110, 101, 109)})
    t = run_backtest({"X": bars}, cfg(), long_once(99.0)).trades[0]
    assert t.exit_reason == "target"
    assert t.exit > 105 and t.pnl_gross > 0


def test_commission_minimum_applies_both_sides():
    bars = day_bars(overrides={1: (101, 101, 101, 101), 3: (101, 110, 101, 109)})
    t = run_backtest({"X": bars}, cfg(), long_once(99.0)).trades[0]
    assert t.costs >= 0.70 - 1e-9


def test_kill_switch_flattens_and_halts_for_the_day():
    # big position (1% risk over a 11-pt stop distance), price sinks ~8% but stays above the stop
    px = lambda i: 101.0 if i < 3 else 92.0
    bars = day_bars(price=px)
    calls = {"n": 0}

    def fn(st, eq):
        calls["n"] += 1
        p = st.last_price
        return Decision(st.symbol, Side.LONG, 1.0, True, [], p, 80.0, p + 50, 1)

    # oversized on purpose (notional cap binds at ~49 sh) so a drop to 92 loses ~9% > 3% while the stop at 80 stays untouched
    c = cfg(risk=RiskLimits(risk_per_trade_pct=0.5, max_position_notional_pct=1.0, daily_drawdown_kill_pct=0.03))
    r = run_backtest({"X": bars}, c, fn)
    assert len(r.trades) == 1 and r.trades[0].exit_reason == "kill_switch"
    assert r.trades[0].exit_ts.astimezone(NY).time() < time(10, 0)


def test_signal_sees_only_the_past():
    seen = []
    closes = {}
    bars = day_bars(price=lambda i: 100 + i * 0.01)
    for i, b in enumerate(bars):
        closes[b.ts] = b.c

    def fn(st, eq):
        seen.append(st.last_price)
        return None

    run_backtest({"X": bars}, cfg(), fn)
    ordered = [b.c for b in bars]
    # at evaluation k the state must reflect exactly bar k's close, never bar k+1's
    assert all(abs(s - c) < 1e-9 for s, c in zip(seen, ordered))


@pytest.mark.parametrize(
    ("side", "breakout"),
    [
        ("BUY", (100.05, 100.40, 99.95, 100.20)),
        ("SHORT", (100.05, 100.10, 99.60, 99.80)),
    ],
)
def test_opening_range_breakout_applies_volume_and_atr_gates(side, breakout):
    bars = []
    first_day = date(2025, 1, 6)
    trading_days = []
    day = first_day
    while len(trading_days) < 11:
        if day.weekday() < 5:
            trading_days.append(day)
        day += timedelta(days=1)
    for day_index, current_day in enumerate(trading_days):
        day_data = orb_day_bars(
            current_day,
            breakout if day_index == len(trading_days) - 1 else None,
            breakout_volume=2000.0,
        )
        bars.extend(day_data)
    result = run_backtest({"SPY": bars}, cfg(), opening_range_breakout)

    assert len(result.trades) == 1
    assert result.trades[0].side == side
    assert result.trades[0].entry_ts.astimezone(NY).date() == trading_days[-1]
    assert result.trades[0].entry_ts.astimezone(NY).time() == time(9, 51)
    assert result.trades[0].rationale.startswith("SPY")
    assert "prior 20-bar average" in result.trades[0].rationale
    assert "10-session same-time average" in result.trades[0].rationale
    assert result.trades[0].rationale.startswith("SPY")


def test_opening_range_rejects_breakout_below_relative_volume_threshold():
    bars = []
    first_day = date(2025, 1, 6)
    trading_days = []
    day = first_day
    while len(trading_days) < 11:
        if day.weekday() < 5:
            trading_days.append(day)
        day += timedelta(days=1)
    for day_index, current_day in enumerate(trading_days):
        day_data = orb_day_bars(
            current_day,
            (100.05, 100.40, 99.95, 100.20)
            if day_index == len(trading_days) - 1 else None,
        )
        bars.extend(day_data)

    result = run_backtest({"SPY": bars}, cfg(), opening_range_breakout)

    assert result.trades == []


def test_parameter_grid_passes_custom_signal_to_each_backtest():
    bars = day_bars(price=lambda i: 100.0 if i < 15 else 101.0)
    results = run_parameter_grid(
        {"SPY": bars},
        cfg(),
        grid={"min_reward_risk": [2.0]},
        workers=1,
        signal_fn=long_once(99.0),
    )

    assert len(results[(2.0,)]) == 1


def test_parameter_grid_reserves_trial_count_before_failed_replays(tmp_path, monkeypatch):
    import json

    import backtest.walkforward as walkforward

    ledger_path = tmp_path / "logs" / "trial-ledger.json"

    def failed_replay(_job):
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(walkforward, "_run", failed_replay)
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        run_parameter_grid(
            {"SPY": day_bars()},
            cfg(),
            grid={"min_signal_score": [0.65, 0.75]},
            workers=1,
            trial_cycle_id="failed-sweep",
            trial_ledger_path=ledger_path,
        )

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["num_trials"] == 2
    assert ledger["total_trials"] == 2
    assert ledger["cycles"]["failed-sweep"] == 2


def test_opening_range_rejects_breakout_below_intraday_atr_regime():
    bars = []
    first_day = date(2025, 1, 6)
    trading_days = []
    day = first_day
    while len(trading_days) < 11:
        if day.weekday() < 5:
            trading_days.append(day)
        day += timedelta(days=1)
    for day_index, current_day in enumerate(trading_days):
        bars.extend(orb_day_bars(
            current_day,
            (100.05, 100.40, 99.95, 100.20)
            if day_index == len(trading_days) - 1 else None,
            breakout_volume=2000.0,
            regular_range=0.04 if day_index == len(trading_days) - 1 else 0.4,
        ))

    result = run_backtest({"SPY": bars}, cfg(), opening_range_breakout)

    assert result.trades == []


def test_opening_range_width_trailing_stop_tightens_after_one_r():
    bars = day_bars(overrides={
        1: (100, 100, 100, 100),
        2: (100, 102, 100, 102),
        3: (102, 103, 101.4, 102),
    })

    def signal(st, _equity):
        if st.last_price == 100:
            return Decision(
                st.symbol, Side.LONG, 1.0, True, [], 100, 99, 120, 1,
                trail_distance=0.5,
            )
        return None

    trade = run_backtest(
        {"SPY": bars},
        cfg(risk=replace(cfg().risk, min_reward_risk=10.0)),
        signal,
    ).trades[0]

    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(101.5 * (1 - 1e-4))


def test_session_vwap_resets_and_daily_close_banked():
    st = SymbolState("X")
    d1 = datetime(2025, 3, 3, 15, 0, tzinfo=timezone.utc)
    d2 = datetime(2025, 3, 4, 15, 0, tzinfo=timezone.utc)
    st.on_trade(Trade(d1, "X", 100.0, 10, 1))
    st.on_trade(Trade(d1 + timedelta(seconds=1), "X", 102.0, 10, 1))
    assert st.session_vwap() == pytest.approx(101.0, abs=1e-9)
    st.on_trade(Trade(d2, "X", 90.0, 5, 1))
    assert st.session_vwap() == pytest.approx(90.0, abs=1e-9)
    assert st.daily_closes == [102.0]


def test_normalize_drops_premarket_zero_volume_and_dupes():
    pre = datetime(2025, 3, 3, 8, 0, tzinfo=NY).astimezone(timezone.utc)
    rth = datetime(2025, 3, 3, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    bars = [MinuteBar(pre, 1, 1, 1, 1, 10), MinuteBar(rth, 1, 1, 1, 1, 0),
            MinuteBar(rth, 1, 1, 1, 1, 5), MinuteBar(rth, 1, 1, 1, 1, 5)]
    assert len(normalize(bars)) == 1


def test_normalize_rejects_nonfinite_and_inconsistent_ohlc():
    ts = datetime(2025, 3, 3, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    bars = [
        MinuteBar(ts, 100, 99, 98, 100, 100),
        MinuteBar(ts + timedelta(minutes=1), float("nan"), 101, 99, 100, 100),
        MinuteBar(ts + timedelta(minutes=2), 100, 102, 99, 101, 100),
    ]
    assert [bar.ts for bar in normalize(bars)] == [bars[-1].ts]


def test_data_quality_reports_flat_volume_and_missing_candles():
    ts = datetime(2025, 3, 3, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    bars = [
        MinuteBar(ts, 100, 100, 100, 100, 1000),
        MinuteBar(ts + timedelta(minutes=1), 100, 101, 99, 100, 10),
        MinuteBar(ts + timedelta(minutes=3), 100, 101, 99, 100, 1000),
    ]
    quality = data_quality(bars)
    assert quality["flatline_bars"] == 1
    assert quality["low_volume_bars"] == 1
    assert quality["missing_intervals"] == 1


def test_atr_slippage_is_dynamic_and_reported():
    bars = day_bars(
        overrides={i: (100, 102, 98, 100) for i in range(390)},
    )

    def signal_after_warmup(state, _equity):
        if len(state.agg.bars["1m"]) == 70:
            return Decision(state.symbol, Side.LONG, 1.0, True, [], 100, 95, 120, 1)
        return None

    fixed = run_backtest(
        {"X": bars}, cfg(slippage_atr_mult=0, commission_per_share=0, commission_min=0),
        signal_after_warmup,
    ).trades[0]
    atr_slipped = run_backtest(
        {"X": bars}, cfg(slippage_atr_mult=0.5, commission_per_share=0, commission_min=0),
        signal_after_warmup,
    ).trades[0]
    assert atr_slipped.slippage_cost > fixed.slippage_cost
    assert atr_slipped.costs > fixed.costs


def test_short_entries_include_sell_side_regulatory_fees():
    bars = day_bars(overrides={1: (101, 101, 101, 101), 3: (101, 90, 90, 90)})

    def short_once(_state, _equity):
        return Decision("X", Side.SHORT, 1.0, True, [], 101, 103, 90, 1)

    trade = run_backtest({"X": bars}, cfg(), short_once).trades[0]
    assert trade.regulatory_fees > 0
    assert trade.commission_cost >= 0.70


def test_synthetic_is_deterministic_and_rth_only():
    a, b = synthetic_bars(3, seed=5), synthetic_bars(3, seed=5)
    assert [x.c for x in a] == [x.c for x in b]
    assert len(a) == 3 * 390
    assert all(time(9, 30) <= x.ts.astimezone(NY).time() < time(16, 0) for x in a)


def test_bootstrap_ci_brackets_mean():
    xs = [1.0, -1.0, 2.0, -0.5, 0.5, 1.5, -1.0, 0.0, 1.0, -0.2] * 5
    lo, hi = bootstrap_ci(xs)
    mean = sum(xs) / len(xs)
    assert lo <= mean <= hi


def test_summarize_no_trades_safe():
    m = summarize([], [], 5000.0)
    assert m["n"] == 0 and m["net_pnl"] == 0


def test_realized_equity_curve_and_cagr_metrics():
    first = datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    second = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)
    trades = [
        SimpleNamespace(entry_ts=first - timedelta(minutes=1), exit_ts=first, pnl_net=100.0,
                        pnl_gross=100.0, costs=0.0, r_multiple=1.0),
        SimpleNamespace(entry_ts=second - timedelta(minutes=1), exit_ts=second, pnl_net=-50.0,
                        pnl_gross=-50.0, costs=0.0, r_multiple=-0.5),
    ]
    curve = trade_equity_curve(trades, 5000.0)
    assert [equity for _, equity in curve] == [5100.0, 5050.0]
    metrics = summarize(trades, curve, 5000.0)
    assert metrics["mean_r"] == pytest.approx(0.25)
    assert metrics["cagr_pct"] > 0
