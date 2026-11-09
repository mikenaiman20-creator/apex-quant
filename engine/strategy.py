"""Behavioral + technical alpha engine and risk engine.

Every evaluation returns a Decision carrying a full rationale so the dashboard and journal
can stream *why* a setup was taken or skipped. Rules are numeric; nothing is "looks bullish".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import List, Optional, Tuple

from config.settings import RiskLimits
from engine.data_pipeline import (
    SymbolState, atr, bbwp, ema, swing_highs, swing_lows, vwap, volume_profile,
)


class Side(str, Enum):
    LONG = "BUY"
    SHORT = "SHORT"


@dataclass
class Check:
    name: str
    passed: bool
    weight: float
    detail: str


@dataclass
class Decision:
    symbol: str
    side: Optional[Side]
    score: float
    take: bool
    checks: List[Check] = field(default_factory=list)
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    qty: int = 0
    reject_reason: str = ""
    behavioral_story: str = ""
    trail_distance: Optional[float] = None
    trail_activation_r: float = 1.0

    def rationale(self) -> str:
        lines = [f"{c.name}: {'PASS' if c.passed else 'FAIL'} (w={c.weight:.2f}) {c.detail}" for c in self.checks]
        head = f"{self.symbol} {self.side.value if self.side else '-'} score={self.score:.2f}"
        tail = self.behavioral_story or self.reject_reason
        return head + " | " + " ; ".join(lines) + (f" || {tail}" if tail else "")


# ----------------------------------------------------------------------------- risk engine
class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.l = limits

    def size(self, equity: float, entry: float, stop: float) -> int:
        """Shares such that (entry-stop)*qty == risk_per_trade_pct*equity, capped by notional limit."""
        per_share = abs(entry - stop)
        if per_share <= 0 or entry <= 0:
            return 0
        by_risk = (equity * self.l.risk_per_trade_pct) / per_share
        by_notional = (equity * self.l.max_position_notional_pct) / entry
        return max(int(min(by_risk, by_notional)), 0)

    def levels(self, side: Side, entry: float, atr_val: float, structure_stop: Optional[float]) -> Tuple[float, float]:
        """Stop = farther of ATR-multiple stop and structure stop (beyond swing), target = R:R multiple."""
        atr_dist = self.l.atr_stop_mult * atr_val
        if side is Side.LONG:
            stop = entry - atr_dist
            if structure_stop is not None:
                stop = min(stop, structure_stop - 0.1 * atr_val)
            target = entry + self.l.min_reward_risk * (entry - stop)
        else:
            stop = entry + atr_dist
            if structure_stop is not None:
                stop = max(stop, structure_stop + 0.1 * atr_val)
            target = entry - self.l.min_reward_risk * (stop - entry)
        return stop, target

    def manage_stop(self, side: Side, entry: float, initial_stop: float, current_stop: float,
                    price: float, extreme: float, atr_val: float) -> float:
        """Monotonic stop: break-even at +1R, then Chandelier (extreme -/+ 3*ATR). Never loosens."""
        r = abs(entry - initial_stop)
        new = current_stop
        if side is Side.LONG:
            if price - entry >= self.l.breakeven_at_r * r:
                new = max(new, entry)
                new = max(new, extreme - self.l.chandelier_atr_mult * atr_val)
            return max(new, current_stop)
        if entry - price >= self.l.breakeven_at_r * r:
            new = min(new, entry)
            new = min(new, extreme + self.l.chandelier_atr_mult * atr_val)
        return min(new, current_stop)


def simple_moving_average(values: list[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    window = values[-period:]
    if any(not isfinite(value) for value in window):
        return None
    return sum(window) / period


def average_directional_index(bars, period: int = 14) -> float | None:
    if period <= 0 or len(bars) < period * 2 + 1:
        return None
    plus_dm, minus_dm, true_ranges = [], [], []
    for previous, current in zip(bars[:-1], bars[1:]):
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        true_ranges.append(max(current.high - current.low, abs(current.high - previous.close),
                               abs(current.low - previous.close)))
    smoothed_tr = sum(true_ranges[:period])
    smoothed_plus = sum(plus_dm[:period])
    smoothed_minus = sum(minus_dm[:period])
    dx_values = []
    for index in range(period - 1, len(true_ranges)):
        if index >= period:
            smoothed_tr += true_ranges[index] - smoothed_tr / period
            smoothed_plus += plus_dm[index] - smoothed_plus / period
            smoothed_minus += minus_dm[index] - smoothed_minus / period
        if smoothed_tr <= 0:
            dx_values.append(0.0)
            continue
        plus_di = 100 * smoothed_plus / smoothed_tr
        minus_di = 100 * smoothed_minus / smoothed_tr
        denominator = plus_di + minus_di
        dx_values.append(100 * abs(plus_di - minus_di) / denominator if denominator else 0.0)
    if len(dx_values) < period:
        return None
    adx_value = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        adx_value = (adx_value * (period - 1) + dx) / period
    return adx_value


@dataclass(frozen=True)
class TimeframeAssessment:
    allowed: bool
    reason: str
    hourly: dict[str, float | bool | None]
    daily: dict[str, float | bool | None]
    intraday: dict[str, float | bool | None]


class MultiTimeframeFilter:
    """Require aligned 1h/1d trend and 15m/5m confirmation before an entry."""

    def __init__(self, min_adx: float = 18.0):
        if not isfinite(min_adx) or min_adx < 0:
            raise ValueError("min_adx must be a finite non-negative value")
        self.min_adx = min_adx

    @staticmethod
    def _alignment(closes: list[float], price: float, side: Side) -> dict[str, float | bool | None]:
        sma20 = simple_moving_average(closes, 20)
        sma50 = simple_moving_average(closes, 50)
        sma200 = simple_moving_average(closes, 200)
        if None in (sma20, sma50, sma200):
            return {"sma20": sma20, "sma50": sma50, "sma200": sma200, "aligned": None}
        bullish = price > sma20 > sma50 > sma200
        bearish = price < sma20 < sma50 < sma200
        return {
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "aligned": bullish if side is Side.LONG else bearish,
        }

    def assess(self, price: float, side: Side, hourly_bars, daily_bars, bars_15m,
               bars_5m) -> TimeframeAssessment:
        if not isfinite(price) or price <= 0:
            missing = {"sma20": None, "sma50": None, "sma200": None, "aligned": None, "adx": None}
            return TimeframeAssessment(False, "invalid price for MTF assessment",
                                       missing.copy(), missing.copy(), missing.copy())
        hourly = self._alignment([bar.close for bar in hourly_bars], price, side)
        daily = self._alignment([bar.close for bar in daily_bars], price, side)
        intraday: dict[str, float | bool | None] = {}
        for label, bars in (("15m", bars_15m), ("5m", bars_5m)):
            closes = [bar.close for bar in bars]
            average = simple_moving_average(closes, 20)
            intraday[label] = (
                closes[-1] > average if average is not None and side is Side.LONG
                else closes[-1] < average if average is not None
                else None
            )
        hourly["adx"] = average_directional_index(hourly_bars)
        daily["adx"] = average_directional_index(daily_bars)
        if hourly["aligned"] is None or daily["aligned"] is None:
            reason = "insufficient 200-bar 1h/1d history for MTF alignment"
        elif not hourly["aligned"] or not daily["aligned"]:
            reason = "higher-timeframe SMA20/50/200 alignment opposes signal"
        elif hourly["adx"] is None or hourly["adx"] < self.min_adx:
            reason = f"1h ADX below trend threshold {self.min_adx:.1f}"
        elif daily["adx"] is None or daily["adx"] < self.min_adx:
            reason = f"1d ADX below trend threshold {self.min_adx:.1f}"
        elif intraday["15m"] is not True or intraday["5m"] is not True:
            reason = "5m/15m trend confirmation opposes signal or is unavailable"
        else:
            reason = "1h/1d SMA alignment, ADX, and 5m/15m confirmation aligned"
        return TimeframeAssessment(
            reason.startswith("1h/1d "), reason, hourly, daily, intraday
        )


# ----------------------------------------------------------------------------- alpha engine
class AlphaEngine:
    """Weighted confluence. Hard gates (regime, spread, ATR availability) veto; the rest score."""

    WEIGHTS = {
        "trend_d50": 0.20,
        "vwap_retest": 0.20,
        "stop_hunt_reversal": 0.20,
        "book_imbalance": 0.15,
        "volume_delta": 0.15,
        "fomo_exhaustion": 0.10,
    }

    def __init__(self, risk: RiskEngine, min_score: float = 0.70):
        self.risk = risk
        self.min_score = min_score

    def evaluate(self, st: SymbolState, equity: float) -> Decision:
        m5 = list(st.agg.bars["5m"])[-300:]   # enough for ATR(14) and BBWP(13, 252)
        m1 = list(st.agg.bars["1m"])[-400:]
        px = st.last_price
        if px is None or len(m5) < 20 or len(m1) < 10:
            return Decision(st.symbol, None, 0.0, False, reject_reason="warming up: insufficient bars")

        a = atr(m5, 14)
        if a is None or a <= 0:
            return Decision(st.symbol, None, 0.0, False, reject_reason="ATR unavailable")

        # ---- hard gates
        q = st.last_quote
        if q and q.ask > q.bid and (q.ask - q.bid) / px > 0.0015:
            return Decision(st.symbol, None, 0.0, False, reject_reason=f"spread {(q.ask - q.bid) / px:.4%} > 0.15%")
        bw = bbwp([b.close for b in m5])
        if bw is not None and bw < 10:
            return Decision(st.symbol, None, 0.0, False, reject_reason=f"BBWP {bw:.0f} < 10: squeeze, direction undefined")

        long_d = self._score(st, m5, m1, px, a, Side.LONG)
        short_d = self._score(st, m5, m1, px, a, Side.SHORT)
        best = long_d if long_d.score >= short_d.score else short_d
        best.take = best.score >= self.min_score

        if not best.take:
            best.reject_reason = f"score {best.score:.2f} < {self.min_score:.2f}"
            return best

        structure = self._structure_stop(m5, best.side)
        stop, target = self.risk.levels(best.side, px, a, structure)
        qty = self.risk.size(equity, px, stop)
        if qty < 1:
            best.take, best.reject_reason = False, "position size rounds to 0 shares"
            return best
        best.entry, best.stop, best.target, best.qty = px, stop, target, qty
        best.behavioral_story = self._story(best)
        return best

    # ---- scoring
    def _score(self, st, m5, m1, px, a, side: Side) -> Decision:
        w = self.WEIGHTS
        L = side is Side.LONG
        checks: List[Check] = []

        # 1. Daily 50 EMA trend (needs history seeded into st.daily_closes)
        d = st.daily_closes
        if len(d) >= 50:
            e50 = ema(d, 50)[-1]
            ok = (px > e50) if L else (px < e50)
            checks.append(Check("trend_d50", ok, w["trend_d50"], f"px={px:.2f} vs D50={e50:.2f}"))
        else:
            checks.append(Check("trend_d50", False, w["trend_d50"], f"only {len(d)} daily closes; need 50"))

        # 2. VWAP retest: within 0.25 ATR of VWAP, on the correct side, and was >=0.5 ATR away in last 30 1m bars
        v = st.session_vwap()  # session-anchored (resets each NY trading day)
        if v is not None:
            near = abs(px - v) <= 0.25 * a
            side_ok = (px >= v) if L else (px <= v)
            was_away = any(abs(b.close - v) >= 0.5 * a for b in m1[-30:-3])
            checks.append(Check("vwap_retest", near and side_ok and was_away, w["vwap_retest"],
                                f"vwap={v:.2f} dist={abs(px - v) / a:.2f}ATR away_before={was_away}"))
        else:
            checks.append(Check("vwap_retest", False, w["vwap_retest"], "no VWAP"))

        # 3. Stop-hunt reversal: last closed 5m bar pierced a prior swing extreme then closed back inside
        hunt, hd = self._stop_hunt(m5, side)
        checks.append(Check("stop_hunt_reversal", hunt, w["stop_hunt_reversal"], hd))

        # 4. Book imbalance (top-5 levels): >= +0.25 for longs, <= -0.25 for shorts
        imb = st.book.imbalance(5)
        ok = imb >= 0.25 if L else imb <= -0.25
        checks.append(Check("book_imbalance", ok, w["book_imbalance"], f"imb={imb:+.2f}"))

        # 5. Volume delta: last 3 closed 1m bars net delta aligned and > 0.1 of volume
        last3 = m1[-4:-1]
        dv = sum(b.delta for b in last3)
        tv = sum(b.volume for b in last3) or 1.0
        frac = dv / tv
        ok = frac >= 0.10 if L else frac <= -0.10
        checks.append(Check("volume_delta", ok, w["volume_delta"], f"delta/vol={frac:+.2f}"))

        # 6. FOMO exhaustion (fade): stretched >=2 ATR from VWAP with falling delta & a block print into the extreme.
        #    Fade direction: stretched UP -> SHORT candidate; stretched DOWN -> LONG candidate (capitulation).
        fomo, fd = self._fomo(st, m1, px, v, a, side)
        checks.append(Check("fomo_exhaustion", fomo, w["fomo_exhaustion"], fd))

        score = sum(c.weight for c in checks if c.passed)
        return Decision(st.symbol, side, score, False, checks)

    @staticmethod
    def _stop_hunt(m5, side: Side):
        if len(m5) < 15:
            return False, "not enough bars"
        last, hist = m5[-2], m5[:-2]  # last CLOSED bar vs prior history
        if side is Side.LONG:
            lows = swing_lows(hist[-40:])
            if not lows:
                return False, "no swing lows"
            lvl = lows[-1][1]
            ok = last.low < lvl and last.close > lvl and last.delta > 0
            return ok, f"swingLow={lvl:.2f} barLow={last.low:.2f} close={last.close:.2f} delta={last.delta:+.0f}"
        highs = swing_highs(hist[-40:])
        if not highs:
            return False, "no swing highs"
        lvl = highs[-1][1]
        ok = last.high > lvl and last.close < lvl and last.delta < 0
        return ok, f"swingHigh={lvl:.2f} barHigh={last.high:.2f} close={last.close:.2f} delta={last.delta:+.0f}"

    @staticmethod
    def _fomo(st, m1, px, v, a, side: Side):
        if v is None or len(m1) < 8:
            return False, "insufficient data"
        stretch = (px - v) / a
        blocks = st.tape_block_trades() if abs(stretch) >= 2.0 else []
        d_now = sum(b.delta for b in m1[-3:])
        d_prev = sum(b.delta for b in m1[-6:-3])
        if side is Side.SHORT:
            ok = stretch >= 2.0 and d_now < d_prev and bool(blocks)
        else:
            ok = stretch <= -2.0 and d_now > d_prev and bool(blocks)
        return ok, f"stretch={stretch:+.1f}ATR delta {d_prev:+.0f}->{d_now:+.0f} blocks={len(blocks)}"

    @staticmethod
    def _structure_stop(m5, side: Side) -> Optional[float]:
        if side is Side.LONG:
            s = swing_lows(m5[-40:])
            return s[-1][1] if s else None
        s = swing_highs(m5[-40:])
        return s[-1][1] if s else None

    @staticmethod
    def _story(d: Decision) -> str:
        names = {c.name for c in d.checks if c.passed}
        bits = []
        if "stop_hunt_reversal" in names:
            bits.append("liquidity sweep beyond a swing level, then reclaimed: trapped stops fuel the move")
        if "fomo_exhaustion" in names:
            bits.append("retail chase exhausted: price stretched from VWAP while aggressive buying/selling fades")
        if "book_imbalance" in names:
            bits.append("resting liquidity leans our way")
        if "vwap_retest" in names:
            bits.append("institutional reference price (VWAP) defended")
        return "; ".join(bits) or "technical confluence only"
