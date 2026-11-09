"""Microstructure engine: bars, tape, L2 book, volume profile, and indicator math.

Pure computation lives here (testable offline). The async feed wrapper at the bottom
adds auto-reconnect with exponential backoff; broker-specific streams plug into it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

log = logging.getLogger("apex.data")
NY = ZoneInfo("America/New_York")

TIMEFRAMES = {"1s": 1, "1m": 60, "5m": 300, "4h": 14400}


# ----------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class Trade:
    ts: datetime
    symbol: str
    price: float
    size: float
    side: int  # +1 buyer-initiated (lifted ask), -1 seller-initiated (hit bid), 0 unknown


@dataclass(frozen=True)
class Quote:
    ts: datetime
    symbol: str
    bid: float
    bid_size: float
    ask: float
    ask_size: float


@dataclass
class Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    pv: float = 0.0  # sum(price*size) for VWAP

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass
class BookSnapshot:
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size), best first
    asks: List[Tuple[float, float]] = field(default_factory=list)

    def imbalance(self, depth: int = 5) -> float:
        """(bid - ask) / (bid + ask) over top `depth` levels, in [-1, 1]."""
        b = sum(s for _, s in self.bids[:depth])
        a = sum(s for _, s in self.asks[:depth])
        return 0.0 if (a + b) == 0 else (b - a) / (b + a)


# ----------------------------------------------------------------------------- aggregation
class BarAggregator:
    """Builds bars for every timeframe from the trade tape, with delta per bar."""

    def __init__(self, symbol: str, maxlen: int = 2000):
        self.symbol = symbol
        self.bars: Dict[str, Deque[Bar]] = {tf: deque(maxlen=maxlen) for tf in TIMEFRAMES}

    @staticmethod
    def _bucket(ts: datetime, secs: int) -> datetime:
        epoch = int(ts.timestamp())
        return datetime.fromtimestamp(epoch - epoch % secs, tz=timezone.utc)

    def on_trade(self, t: Trade) -> List[str]:
        """Returns the list of timeframes whose bar just CLOSED (a new bucket began)."""
        closed: List[str] = []
        for tf, secs in TIMEFRAMES.items():
            start = self._bucket(t.ts, secs)
            dq = self.bars[tf]
            if not dq or dq[-1].start != start:
                if dq:
                    closed.append(tf)
                dq.append(Bar(start, t.price, t.price, t.price, t.price))
            b = dq[-1]
            b.high, b.low, b.close = max(b.high, t.price), min(b.low, t.price), t.price
            b.volume += t.size
            b.pv += t.price * t.size
            if t.side > 0:
                b.buy_volume += t.size
            elif t.side < 0:
                b.sell_volume += t.size
        return closed


def classify_side(price: float, bid: Optional[float], ask: Optional[float], prev_price: Optional[float]) -> int:
    """Lee-Ready style: quote rule first, tick rule as fallback."""
    if bid is not None and ask is not None and ask > bid:
        mid = (bid + ask) / 2
        if price > mid:
            return 1
        if price < mid:
            return -1
    if prev_price is not None:
        return 1 if price > prev_price else -1 if price < prev_price else 0
    return 0


# ----------------------------------------------------------------------------- indicators
def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def atr(bars: List[Bar], period: int = 14) -> Optional[float]:
    """Wilder ATR. None until period+1 bars exist."""
    if len(bars) < period + 1:
        return None
    trs = [
        max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
        for p, b in zip(bars[:-1], bars[1:])
    ]
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def vwap(bars: List[Bar]) -> Optional[float]:
    vol = sum(b.volume for b in bars)
    return None if vol == 0 else sum(b.pv for b in bars) / vol


def bbwp(closes: List[float], bb_len: int = 13, lookback: int = 252) -> Optional[float]:
    """Bollinger Band Width Percentile (0-100): where current BB width ranks in its own history."""
    if len(closes) < bb_len + 2:
        return None
    widths: List[float] = []
    for i in range(bb_len, len(closes) + 1):
        w = closes[i - bb_len:i]
        m = sum(w) / bb_len
        sd = math.sqrt(sum((x - m) ** 2 for x in w) / bb_len)
        widths.append(0.0 if m == 0 else (4 * sd) / m)
    hist = widths[-lookback:]
    cur = hist[-1]
    return 100.0 * sum(1 for x in hist[:-1] if x <= cur) / max(len(hist) - 1, 1)


def swing_lows(bars: List[Bar], left: int = 3, right: int = 3) -> List[Tuple[int, float]]:
    out = []
    for i in range(left, len(bars) - right):
        if bars[i].low < min(b.low for b in bars[i - left:i]) and bars[i].low <= min(
            b.low for b in bars[i + 1:i + 1 + right]
        ):
            out.append((i, bars[i].low))
    return out


def swing_highs(bars: List[Bar], left: int = 3, right: int = 3) -> List[Tuple[int, float]]:
    out = []
    for i in range(left, len(bars) - right):
        if bars[i].high > max(b.high for b in bars[i - left:i]) and bars[i].high >= max(
            b.high for b in bars[i + 1:i + 1 + right]
        ):
            out.append((i, bars[i].high))
    return out


# ----------------------------------------------------------------------------- volume profile
@dataclass(frozen=True)
class VolumeProfile:
    poc: float
    vah: float
    val: float


def volume_profile(bars: List[Bar], tick: float = 0.05, value_area: float = 0.70) -> Optional[VolumeProfile]:
    """POC/VAH/VAL. Volume is spread uniformly across each bar's high-low range."""
    if not bars:
        return None
    hist: Dict[int, float] = defaultdict(float)
    for b in bars:
        lo, hi = round(b.low / tick), round(b.high / tick)
        n = max(hi - lo + 1, 1)
        for lvl in range(lo, hi + 1):
            hist[lvl] += b.volume / n
    total = sum(hist.values())
    if total == 0:
        return None
    poc = max(hist, key=hist.get)
    lo = hi = poc
    acc = hist[poc]
    while acc < value_area * total:
        up = hist.get(hi + 1, 0.0)
        dn = hist.get(lo - 1, 0.0)
        if up == 0 and dn == 0:
            break
        if up >= dn:
            hi += 1
            acc += up
        else:
            lo -= 1
            acc += dn
    return VolumeProfile(poc=poc * tick, vah=hi * tick, val=lo * tick)


# ----------------------------------------------------------------------------- symbol state
class SymbolState:
    """Everything the strategy needs about one symbol, updated from live events."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.agg = BarAggregator(symbol)
        self.book = BookSnapshot()
        self.last_quote: Optional[Quote] = None
        self.last_price: Optional[float] = None
        self.market_state: Optional["SymbolState"] = None
        self.daily_closes: List[float] = []  # seeded from history; used for the Daily 50 EMA
        self.tape: Deque[Trade] = deque(maxlen=5000)
        self._recent: Deque[Trade] = deque(maxlen=200)  # cheap window for block-trade detection
        self.session_date = None
        self._s_pv = 0.0
        self._s_vol = 0.0

    def on_quote(self, q: Quote) -> None:
        self.last_quote = q

    def on_book(self, snap: BookSnapshot) -> None:
        self.book = snap

    def _roll_session(self, ts: datetime) -> None:
        """New NY trading date: bank yesterday's last price as the daily close, reset session VWAP."""
        d = ts.astimezone(NY).date()
        if self.session_date != d:
            if self.session_date is not None and self.last_price is not None:
                self.daily_closes.append(self.last_price)
            self.session_date = d
            self._s_pv = 0.0
            self._s_vol = 0.0

    def session_vwap(self) -> Optional[float]:
        return None if self._s_vol <= 0 else self._s_pv / self._s_vol

    def on_trade(self, t: Trade) -> List[str]:
        self._roll_session(t.ts)
        if t.side == 0:
            bid = self.last_quote.bid if self.last_quote else None
            ask = self.last_quote.ask if self.last_quote else None
            t = Trade(t.ts, t.symbol, t.price, t.size, classify_side(t.price, bid, ask, self.last_price))
        self.last_price = t.price
        self.tape.append(t)
        self._recent.append(t)
        self._s_pv += t.price * t.size
        self._s_vol += t.size
        return self.agg.on_trade(t)

    def tape_block_trades(self, window: int = 200, sigma: float = 4.0) -> List[Trade]:
        """Prints whose size is > mean + sigma*sd of the recent tape: crude institutional-block proxy."""
        recent = list(self._recent)[-window:]
        if len(recent) < 30:
            return []
        sizes = [t.size for t in recent]
        m = sum(sizes) / len(sizes)
        sd = math.sqrt(sum((s - m) ** 2 for s in sizes) / len(sizes))
        return [t for t in recent[-20:] if sd > 0 and t.size > m + sigma * sd]


# ----------------------------------------------------------------------------- resilient feed
Handler = Callable[[object], Awaitable[None]]


async def run_feed_forever(
    connect: Callable[[], AsyncIterator[object]],
    handler: Handler,
    max_backoff: float = 60.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Consume an async event stream; on any failure reconnect with exponential backoff + jitter."""
    backoff = 1.0
    while not (stop and stop.is_set()):
        try:
            async for event in connect():
                backoff = 1.0
                await handler(event)
                if stop and stop.is_set():
                    return
            log.warning("feed ended cleanly; reconnecting")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("feed error; reconnecting in %.1fs", backoff)
        await asyncio.sleep(backoff * (1 + 0.2 * random.random()))
        backoff = min(backoff * 2, max_backoff)
