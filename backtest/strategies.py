"""Independent research-only strategies for the event-driven backtester."""
from __future__ import annotations

from collections import deque
from datetime import date, time
from math import isfinite
from itertools import islice
from typing import Any
from zoneinfo import ZoneInfo

from engine.data_pipeline import SymbolState
from engine.strategy import Check, Decision, Side


NY = ZoneInfo("America/New_York")
OPENING_RANGE_MINUTES = 15
ORB_ENTRY_CUTOFF = time(11, 0)
ORB_VOLUME_LOOKBACK = 20
ORB_VOLUME_MULTIPLE = 1.5
ORB_ATR_PERIOD = 14
ORB_ATR_REGIME_DAYS = 10
ORB_TRAIL_RANGE_MULTIPLE = 0.5
ORB_PARAMETER_GRID = {"min_reward_risk": [1.5, 2.5]}
ZSCORE_LOOKBACK = 20
ZSCORE_ENTRY_THRESHOLD = 2.0
ZSCORE_STOP_THRESHOLD = 3.5
ZSCORE_MIN_ENTRY_TIME = time(9, 50)
ZSCORE_LAST_ENTRY_TIME = time(15, 30)
SPY_REGIME_LOOKBACK = 20
SPY_VOLATILITY_BASELINE = 60
SPY_PANIC_ZSCORE = -2.5
SPY_PANIC_DRAWDOWN = 0.015
SPY_PANIC_VOL_MULTIPLE = 2.0
SPY_MIN_PANIC_VOL = 0.0005
ZSCORE_PARAMETER_GRID = {"zscore_entry_threshold": [2.0, 2.5]}


def _recent_bars(state: SymbolState, count: int) -> list[Any]:
    return list(islice(reversed(state.agg.bars["1m"]), count))


def _rolling_zscore(state: SymbolState, lookback: int = ZSCORE_LOOKBACK):
    bars = _recent_bars(state, lookback)
    if len(bars) != lookback:
        return None
    if any(
        (newer.start - older.start).total_seconds() != 60
        for newer, older in zip(bars, bars[1:])
    ):
        return None
    closes = [bar.close for bar in bars]
    if any(not isfinite(close) or close <= 0 for close in closes):
        return None
    mean = sum(closes) / lookback
    variance = sum((close - mean) ** 2 for close in closes) / lookback
    deviation = variance ** 0.5
    if deviation <= 0 or not isfinite(deviation):
        return None
    return mean, deviation, (closes[0] - mean) / deviation


def _spy_market_panic(state: SymbolState, timestamp) -> bool:
    market = state.market_state
    if market is None or market.symbol != "SPY":
        return True
    required_bars = SPY_REGIME_LOOKBACK + SPY_VOLATILITY_BASELINE + 1
    market_bars = _recent_bars(market, required_bars)
    if len(market_bars) < required_bars:
        return True
    if market_bars[0].start != timestamp:
        return True
    if any(
        (newer.start - older.start).total_seconds() != 60
        for newer, older in zip(market_bars, market_bars[1:])
    ):
        return True

    zscore = _rolling_zscore(market)
    if zscore is None:
        return True
    closes = [bar.close for bar in market_bars]
    recent = closes[:SPY_REGIME_LOOKBACK]
    peak = max(recent)
    drawdown = recent[0] / peak - 1.0 if peak > 0 else -1.0
    returns = [
        current / previous - 1.0
        for previous, current in zip(closes[1:], closes[:-1])
        if previous > 0
    ]
    if len(returns) != SPY_REGIME_LOOKBACK + SPY_VOLATILITY_BASELINE:
        return True
    short_vol = _population_std(returns[:SPY_REGIME_LOOKBACK])
    baseline_vol = _population_std(returns[SPY_REGIME_LOOKBACK:])
    latest_drop = returns[0]
    return bool(
        zscore[2] <= SPY_PANIC_ZSCORE
        or drawdown <= -SPY_PANIC_DRAWDOWN
        or latest_drop <= -0.005
        or (
            baseline_vol > 0
            and short_vol >= SPY_MIN_PANIC_VOL
            and short_vol > SPY_PANIC_VOL_MULTIPLE * baseline_vol
        )
    )


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def short_term_zscore_mean_reversion(
    state: SymbolState,
    equity: float,
) -> Decision | None:
    """Fade 20-bar price extremes intraday, gated by synchronous SPY panic telemetry."""
    del equity
    bars = state.agg.bars["1m"]
    if not bars:
        return None
    current = bars[-1]
    local = current.start.astimezone(NY)
    if not ZSCORE_MIN_ENTRY_TIME <= local.time() <= ZSCORE_LAST_ENTRY_TIME:
        return None
    prior = bars[-2] if len(bars) >= 2 else None
    if prior is None or (current.start - prior.start).total_seconds() != 60:
        return None

    market = state.market_state
    if market is None or market.symbol != "SPY" or not market.agg.bars["1m"]:
        return None
    if market.agg.bars["1m"][-1].start != current.start:
        return None
    if _spy_market_panic(state, current.start):
        return None

    stats = _rolling_zscore(state)
    if stats is None:
        return None
    mean, deviation, zscore = stats
    threshold = getattr(state, "zscore_entry_threshold", ZSCORE_ENTRY_THRESHOLD)
    stop_z = ZSCORE_STOP_THRESHOLD
    if zscore <= -threshold:
        side = Side.LONG
        stop = mean - stop_z * deviation
    elif zscore >= threshold:
        side = Side.SHORT
        stop = mean + stop_z * deviation
    else:
        return None

    target = mean
    if (side is Side.LONG and not current.close < target) or (
        side is Side.SHORT and not current.close > target
    ):
        return None
    return Decision(
        state.symbol,
        side,
        min(abs(zscore) / threshold, 1.0),
        True,
        [
            Check(
                "short_term_zscore", True, 1.0,
                f"20-bar z-score={zscore:+.2f}; entry threshold={threshold:.2f}; "
                f"mean={mean:.4f}",
            ),
            Check(
                "spy_market_regime", True, 1.0,
                "synchronous SPY regime passed: no panic z-score, drawdown, "
                "one-minute crash, or volatility expansion",
            ),
        ],
        entry=current.close,
        stop=stop,
        target=target,
    )


def _session_bars(state: SymbolState, day: date) -> list[Any]:
    same_day = []
    for bar in reversed(state.agg.bars["1m"]):
        if bar.start.astimezone(NY).date() != day:
            break
        if time(9, 30) <= bar.start.astimezone(NY).time() < time(16, 0):
            same_day.append(bar)
    same_day.reverse()
    return same_day


def _set_opening_range(cache: dict[str, Any], session: list[Any]) -> None:
    opening = [
        bar for bar in session
        if time(9, 30) <= bar.start.astimezone(NY).time() < time(9, 45)
    ]
    opening_times = {bar.start.astimezone(NY).time() for bar in opening}
    expected_times = {time(9, minute) for minute in range(30, 45)}
    cache.update({
        "opening_high": max((bar.high for bar in opening), default=None),
        "opening_low": min((bar.low for bar in opening), default=None),
        "opening_range_valid": len(opening) == OPENING_RANGE_MINUTES
        and opening_times == expected_times,
    })


def _intraday_atr(window: deque) -> float | None:
    if len(window) != ORB_ATR_PERIOD + 1:
        return None
    bars = list(window)
    if any(
        (current.start - previous.start).total_seconds() != 60
        for previous, current in zip(bars, bars[1:])
    ):
        return None
    true_ranges = [
        max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        for previous, current in zip(bars, bars[1:])
    ]
    value = sum(true_ranges) / len(true_ranges)
    return value if isfinite(value) and value > 0 else None


def _update_orb_indicators(state: SymbolState, bars: Any) -> dict[str, Any]:
    current = bars[-1]
    current_local = current.start.astimezone(NY)
    day = current_local.date()
    cache = getattr(state, "_orb_research_cache", None)
    if cache is None:
        cache = {"atr_history": {}}
        state._orb_research_cache = cache

    if cache.get("day") != day:
        session = _session_bars(state, day)
        cache.update({
            "day": day,
            "opening_high": None,
            "opening_low": None,
            "opening_range_valid": False,
            "opening_range_checked": False,
            "volume_history": deque(
                (bar.volume for bar in session[:-1][-ORB_VOLUME_LOOKBACK:]),
                maxlen=ORB_VOLUME_LOOKBACK,
            ),
            "atr_window": deque(session[-(ORB_ATR_PERIOD + 1):], maxlen=ORB_ATR_PERIOD + 1),
            "last_bar_ts": None,
            "last_atr_minute": None,
        })

    if not cache["opening_range_checked"] and current_local.time() >= time(9, 45):
        _set_opening_range(cache, _session_bars(state, day))
        cache["opening_range_checked"] = True

    if current.start != cache["last_bar_ts"]:
        if cache["last_bar_ts"] is not None:
            cache["volume_history"].append(cache["last_bar_volume"])
            cache["atr_window"].append(current)
        cache["last_bar_ts"] = current.start
        cache["last_bar_volume"] = current.volume

    average_volume = (
        sum(cache["volume_history"]) / len(cache["volume_history"])
        if len(cache["volume_history"]) == ORB_VOLUME_LOOKBACK else None
    )
    relative_volume_passed = (
        average_volume is not None
        and average_volume > 0
        and current.volume >= ORB_VOLUME_MULTIPLE * average_volume
    )

    intraday_atr = _intraday_atr(cache["atr_window"])
    minute_index = current_local.hour * 60 + current_local.minute
    historical_atr = cache["atr_history"].setdefault(minute_index, deque(maxlen=ORB_ATR_REGIME_DAYS))
    atr_average = (
        sum(historical_atr) / ORB_ATR_REGIME_DAYS
        if len(historical_atr) == ORB_ATR_REGIME_DAYS else None
    )
    atr_regime_passed = (
        intraday_atr is not None
        and atr_average is not None
        and intraday_atr > atr_average
    )
    if (
        cache["opening_range_valid"]
        and intraday_atr is not None
        and cache["last_atr_minute"] != minute_index
    ):
        historical_atr.append(intraday_atr)
        cache["last_atr_minute"] = minute_index

    return {
        "current_local": current_local,
        "average_volume": average_volume,
        "relative_volume_passed": relative_volume_passed,
        "intraday_atr": intraday_atr,
        "atr_average": atr_average,
        "atr_regime_passed": atr_regime_passed,
    }


def opening_range_breakout(state: SymbolState, equity: float) -> Decision | None:
    """Research ORB: 15-minute range break, relative-volume and intraday ATR gates."""
    del equity  # Position sizing is applied by the backtest risk engine at the next-bar fill.
    bar_deque = state.agg.bars["1m"]
    if not bar_deque:
        return None

    current = bar_deque[-1]
    indicators = _update_orb_indicators(state, bar_deque)
    current_local = indicators["current_local"]

    if (
        current_local.time() < time(9, 45)
        or current_local.time() > ORB_ENTRY_CUTOFF
        or not state._orb_research_cache["opening_range_valid"]
    ):
        return None
    if not indicators["relative_volume_passed"] or not indicators["atr_regime_passed"]:
        return None

    opening_high = state._orb_research_cache["opening_high"]
    opening_low = state._orb_research_cache["opening_low"]
    prior = bar_deque[-2] if len(bar_deque) >= 2 else None
    if prior is None or prior.start.astimezone(NY).date() != current_local.date():
        return None

    if prior.close <= opening_high < current.close:
        side, stop, detail = (
            Side.LONG,
            opening_low,
            f"close crossed the {OPENING_RANGE_MINUTES}-minute high {opening_high:.4f}",
        )
    elif prior.close >= opening_low > current.close:
        side, stop, detail = (
            Side.SHORT,
            opening_high,
            f"close crossed the {OPENING_RANGE_MINUTES}-minute low {opening_low:.4f}",
        )
    else:
        return None

    opening_range_width = opening_high - opening_low
    if opening_range_width <= 0 or stop is None:
        return None
    volume_ratio = current.volume / indicators["average_volume"]
    atr_ratio = indicators["intraday_atr"] / indicators["atr_average"]
    return Decision(
        state.symbol,
        side,
        1.0,
        True,
        [
            Check(
                "opening_range_breakout", True, 1.0,
                f"{detail}; volume={volume_ratio:.2f}x prior 20-bar average "
                f"(minimum {ORB_VOLUME_MULTIPLE:.1f}x)",
            ),
            Check(
                "intraday_atr_regime", True, 1.0,
                f"14-bar ATR={indicators['intraday_atr']:.4f}, "
                f"10-session same-time average={indicators['atr_average']:.4f} "
                f"({atr_ratio:.2f}x)",
            ),
        ],
        entry=current.close,
        stop=stop,
        trail_distance=opening_range_width * ORB_TRAIL_RANGE_MULTIPLE,
        trail_activation_r=1.0,
    )
