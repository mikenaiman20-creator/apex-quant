"""Market data loading for the backtester: CSV files, Alpaca REST (free IEX feed), synthetic random walk.

All bars are normalized to: tz-aware UTC start time, regular trading hours only (09:30-16:00 New York),
volume > 0, sorted, de-duplicated.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from zoneinfo import ZoneInfo

from server.io_utils import ensure_directory, secure_file

NY = ZoneInfo("America/New_York")
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)
log = logging.getLogger("apex.backtest.data")


@dataclass(frozen=True)
class MinuteBar:
    ts: datetime  # bar START, tz-aware UTC
    o: float
    h: float
    l: float
    c: float
    v: float


def _parse_ts(raw: str, naive_tz: str = "UTC") -> datetime:
    raw = raw.strip()
    if raw.replace(".", "", 1).isdigit():
        x = float(raw)
        if x > 1e11:  # milliseconds
            x /= 1000.0
        return datetime.fromtimestamp(x, tz=timezone.utc)
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(naive_tz))
    return dt.astimezone(timezone.utc)


def normalize(bars: Iterable[MinuteBar]) -> List[MinuteBar]:
    seen, out = set(), []
    rejected = 0
    for b in sorted(bars, key=lambda x: x.ts):
        values = (b.o, b.h, b.l, b.c, b.v)
        if (b.ts in seen or not all(math.isfinite(value) for value in values)
                or b.v <= 0 or min(b.o, b.h, b.l, b.c) <= 0 or b.h < b.l
                or b.h < max(b.o, b.c) or b.l > min(b.o, b.c)):
            rejected += 1
            continue
        t = b.ts.astimezone(NY).time()
        if not (RTH_OPEN <= t < RTH_CLOSE):
            continue
        seen.add(b.ts)
        out.append(b)
    if rejected:
        log.warning("Rejected %d duplicate, malformed, non-finite, or zero-volume candles.", rejected)
    return out


def data_quality(bars: Iterable[MinuteBar], interval_minutes: int = 1) -> dict[str, int | float]:
    """Report flat candles, low-volume bars, and missing-bar gaps without mutating input."""
    ordered = sorted(bars, key=lambda bar: bar.ts)
    if not ordered:
        return {
            "bars": 0, "flatline_bars": 0, "low_volume_bars": 0,
            "missing_intervals": 0, "median_volume": 0.0,
        }
    volumes = sorted(bar.v for bar in ordered if math.isfinite(bar.v) and bar.v > 0)
    median_volume = volumes[len(volumes) // 2] if volumes else 0.0
    flatline_bars = sum(
        1 for bar in ordered if bar.o == bar.h == bar.l == bar.c
    )
    low_volume_bars = sum(
        1 for bar in ordered if median_volume > 0 and bar.v < median_volume * 0.1
    )
    expected = timedelta(minutes=interval_minutes)
    missing_intervals = sum(
        max(int((current.ts - previous.ts) / expected) - 1, 0)
        for previous, current in zip(ordered, ordered[1:])
        if previous.ts.date() == current.ts.date()
    )
    return {
        "bars": len(ordered),
        "flatline_bars": flatline_bars,
        "low_volume_bars": low_volume_bars,
        "missing_intervals": missing_intervals,
        "median_volume": median_volume,
    }


def load_csv(path: str, naive_tz: str = "UTC") -> List[MinuteBar]:
    """CSV with a header. Accepted columns: timestamp|time|t, open|o, high|h, low|l, close|c, volume|v.
    Timestamps: ISO-8601 or epoch seconds/ms. Naive ISO timestamps are interpreted in `naive_tz`."""
    def pick(row, *names):
        for n in names:
            if n in row and row[n] != "":
                return row[n]
        raise KeyError(f"{path}: missing column among {names}")

    bars = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = {k.strip().lower(): v for k, v in row.items()}
            bars.append(MinuteBar(
                _parse_ts(pick(r, "timestamp", "time", "t", "datetime"), naive_tz),
                float(pick(r, "open", "o")), float(pick(r, "high", "h")),
                float(pick(r, "low", "l")), float(pick(r, "close", "c")),
                float(pick(r, "volume", "v")),
            ))
    return normalize(bars)


def load_dir(directory: str, symbols: List[str], naive_tz: str = "UTC") -> Dict[str, List[MinuteBar]]:
    out = {}
    for s in symbols:
        p = os.path.join(directory, f"{s}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        out[s] = load_csv(p, naive_tz)
    return out


def save_csv(path: str, bars: List[MinuteBar]) -> None:
    target = Path(path)
    if str(target.parent) not in {"", "."}:
        ensure_directory(target.parent)
    with target.open("w", newline="") as f:
        secure_file(target)
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.ts.isoformat(), b.o, b.h, b.l, b.c, b.v])


def fetch_alpaca(symbol: str, start: str, end: str, feed: str = "iex") -> List[MinuteBar]:
    """1-minute bars from Alpaca's REST API (split-adjusted). Reads ALPACA_API_KEY / ALPACA_API_SECRET.
    Free plan: IEX feed only, and `end` must be >= 15 minutes in the past. NOTE: written to the documented
    v2 API shape but not exercised from the build sandbox (no outbound access to it)."""
    key, secret = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_API_SECRET", "")
    if not (key and secret):
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET.")
    bars: List[MinuteBar] = []
    token = None
    while True:
        q = {"symbols": symbol, "timeframe": "1Min", "start": start, "end": end,
             "limit": "10000", "adjustment": "split", "feed": feed}
        if token:
            q["page_token"] = token
        req = urllib.request.Request(
            "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q),
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.load(r)
        for x in payload.get("bars", {}).get(symbol, []):
            bars.append(MinuteBar(_parse_ts(x["t"]), x["o"], x["h"], x["l"], x["c"], x["v"]))
        token = payload.get("next_page_token")
        if not token:
            break
    return normalize(bars)


def synthetic_bars(n_days: int = 160, seed: int = 11, start_price: float = 100.0,
                   daily_vol: float = 0.01, start: date = date(2025, 1, 2)) -> List[MinuteBar]:
    """Driftless random walk with a U-shaped intraday volatility/volume profile.
    By construction it has NO exploitable edge: a correct backtester must not find one."""
    rng = random.Random(seed)
    px, out, d, days = start_price, [], start, 0
    while days < n_days:
        if d.weekday() < 5:
            base = datetime.combine(d, RTH_OPEN, tzinfo=NY)
            for i in range(390):
                frac = i / 389
                shape = 0.6 + 1.8 * (2 * frac - 1) ** 2
                sigma = daily_vol / math.sqrt(390) / 1.2 * shape
                p0, path = px, []
                for _ in range(4):
                    px *= math.exp(rng.gauss(0, sigma / 2))
                    path.append(px)
                vol = max(int(rng.lognormvariate(math.log(2000 * shape), 0.4)), 1)
                ts = (base + timedelta(minutes=i)).astimezone(timezone.utc)
                out.append(MinuteBar(ts, p0, max(path + [p0]), min(path + [p0]), path[-1], float(vol)))
            days += 1
        d += timedelta(days=1)
    return out
