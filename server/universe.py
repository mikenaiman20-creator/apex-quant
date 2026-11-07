"""Daily, paper-runner universe selection from liquid Alpaca US equities."""
from __future__ import annotations

import csv
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("apex.autonomous")
NY = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
PREMARKET_OPEN = time(4, 0)
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")
NON_COMMON_STOCK_NAME = re.compile(
    r"\b(?:ETF|ETN|EXCHANGE[ -]TRADED|FUND|TRUST|INDEX|ULTRA\w*|LEVERAGED|INVERSE|SHORT|BEAR|BULL|2X|3X)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UniverseCandidate:
    symbol: str
    average_dollar_volume: float
    atr_pct: float
    premarket_volume: float
    relative_premarket_volume: float
    spread_pct: float
    rank_score: float


def _chunks(items: list[str], size: int):
    for offset in range(0, len(items), size):
        yield tuple(items[offset:offset + size])


def _read_constituents(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if not rows:
        return []
    header = [value.strip().lower() for value in rows[0]]
    column = next((header.index(name) for name in ("symbol", "ticker") if name in header), None)
    if column is None:
        values = [row[0].strip().upper() for row in rows if row and row[0].strip()]
    else:
        values = [row[column].strip().upper() for row in rows[1:] if len(row) > column and row[column].strip()]
    return list(dict.fromkeys(symbol for symbol in values if SYMBOL_PATTERN.fullmatch(symbol)))


class DynamicUniverseSelector:
    def __init__(
        self,
        client,
        top_n: int = 10,
        constituents_file: Path | None = None,
        min_average_dollar_volume: float = 20_000_000,
        min_atr_pct: float = 0.005,
        max_atr_pct: float = 0.08,
        max_spread_pct: float = 0.0025,
        min_premarket_volume: float = 100,
        min_relative_premarket_volume: float = 0.0005,
        candidate_batch_size: int = 100,
    ):
        self.client = client
        self.top_n = top_n
        self.constituents_file = constituents_file
        self.min_average_dollar_volume = min_average_dollar_volume
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.max_spread_pct = max_spread_pct
        self.min_premarket_volume = min_premarket_volume
        self.min_relative_premarket_volume = min_relative_premarket_volume
        self.candidate_batch_size = candidate_batch_size

    @staticmethod
    def _active_symbols(assets: list[dict[str, Any]]) -> list[str]:
        allowed_exchanges = {"NASDAQ", "NYSE", "AMEX"}
        return sorted({
            asset["symbol"].upper() for asset in assets
            if isinstance(asset.get("symbol"), str)
            if asset.get("status") == "active"
            and asset.get("tradable")
            and asset.get("class") == "us_equity"
            and asset.get("exchange") in allowed_exchanges
            and SYMBOL_PATTERN.fullmatch(asset.get("symbol", "").upper())
            and not NON_COMMON_STOCK_NAME.search(str(asset.get("name") or ""))
        })

    def _candidate_symbols(self) -> list[str]:
        assets = self.client.assets()
        active_symbols = set(self._active_symbols(assets))
        file_symbols = _read_constituents(self.constituents_file) if self.constituents_file else []
        try:
            most_active = self.client.most_actives()
        except Exception:
            log.exception("Alpaca most-active screener failed; using configured constituents/assets.")
            most_active = []
        screened_symbols = [
            row.get("symbol", "").upper() for row in most_active
            if SYMBOL_PATTERN.fullmatch(row.get("symbol", "").upper())
            and row.get("symbol", "").upper() in active_symbols
        ]
        if file_symbols:
            symbols = list(dict.fromkeys(
                [symbol for symbol in file_symbols if symbol in active_symbols] + screened_symbols
            ))
        else:
            symbols = list(dict.fromkeys(screened_symbols))
        if not symbols:
            limit = int(os.getenv("UNIVERSE_ASSET_FALLBACK_LIMIT", "500"))
            symbols = sorted(active_symbols)[:limit]
        if not symbols:
            raise RuntimeError("Alpaca returned no eligible US equity universe.")
        return symbols

    def select(self, now: datetime) -> list[UniverseCandidate]:
        if now.tzinfo is None:
            raise ValueError("Universe selection time must be timezone-aware.")
        symbols = self._candidate_symbols()
        end = now.astimezone(NY)
        daily_start = end - timedelta(days=50)
        daily: dict[str, list[dict[str, Any]]] = {}
        for batch in _chunks(symbols, self.candidate_batch_size):
            daily.update(self.client.bars(batch, "1Day", daily_start, end, limit=10000))

        quality: dict[str, tuple[float, float, float]] = {}
        for symbol in symbols:
            bars = [
                row for row in sorted(daily.get(symbol, []), key=lambda row: row["t"])
                if datetime.fromisoformat(row["t"].replace("Z", "+00:00")).astimezone(NY).date() < end.date()
            ][-21:]
            if len(bars) < 15:
                continue
            closes = [float(row["c"]) for row in bars]
            avg_dollar_volume = sum(float(row["v"]) * close for row, close in zip(bars, closes)) / len(bars)
            true_ranges = []
            for index, row in enumerate(bars):
                high, low = float(row["h"]), float(row["l"])
                previous_close = closes[index - 1] if index else float(row["o"])
                true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
            atr_pct = (sum(true_ranges[-14:]) / min(len(true_ranges[-14:]), 14)) / closes[-1]
            if (closes[-1] < 10 or avg_dollar_volume < self.min_average_dollar_volume
                    or not self.min_atr_pct <= atr_pct <= self.max_atr_pct):
                continue
            quality[symbol] = (avg_dollar_volume, atr_pct, sum(float(row["v"]) for row in bars) / len(bars))
        if not quality:
            raise RuntimeError("No candidates passed minimum daily price, dollar-volume, and ATR filters.")

        today = end.date()
        premarket_start = datetime.combine(today, PREMARKET_OPEN, tzinfo=NY)
        market_open = datetime.combine(today, MARKET_OPEN, tzinfo=NY)
        premarket: dict[str, float] = {}
        ranked_symbols = list(quality)
        for batch in _chunks(ranked_symbols, self.candidate_batch_size):
            minute_bars = self.client.bars(batch, "1Min", premarket_start, min(end, market_open), limit=10000)
            for symbol, values in minute_bars.items():
                premarket[symbol] = sum(float(row["v"]) for row in values)

        candidates = []
        for batch in _chunks(ranked_symbols, self.candidate_batch_size):
            quotes = self.client.latest_quotes(batch)
            for symbol in batch:
                raw = quotes.get(symbol)
                if not raw:
                    continue
                quote_time = datetime.fromisoformat(raw["t"].replace("Z", "+00:00"))
                quote_age = (end.astimezone(quote_time.tzinfo) - quote_time).total_seconds()
                if quote_age < -30 or quote_age > 300:
                    continue
                bid, ask = float(raw.get("bp", 0)), float(raw.get("ap", 0))
                if bid <= 0 or ask < bid:
                    continue
                midpoint = (bid + ask) / 2
                spread_pct = (ask - bid) / midpoint
                avg_dollar_volume, atr_pct, avg_daily_volume = quality[symbol]
                pm_volume = premarket.get(symbol, 0.0)
                relative_pm_volume = pm_volume / avg_daily_volume if avg_daily_volume else 0
                if (spread_pct > self.max_spread_pct or pm_volume < self.min_premarket_volume
                        or relative_pm_volume < self.min_relative_premarket_volume):
                    continue
                liquidity_score = min(math.log10(avg_dollar_volume) / 12, 1)
                volatility_score = min(atr_pct / 0.04, 1)
                activity_score = min(relative_pm_volume / 0.01, 1)
                spread_score = max(0, 1 - spread_pct / self.max_spread_pct)
                score = 0.4 * liquidity_score + 0.2 * volatility_score + 0.2 * activity_score + 0.2 * spread_score
                candidates.append(UniverseCandidate(
                    symbol, avg_dollar_volume, atr_pct, pm_volume, relative_pm_volume, spread_pct, score,
                ))
        candidates.sort(key=lambda candidate: (-candidate.rank_score, candidate.symbol))
        selected = candidates[:self.top_n]
        if not selected:
            raise RuntimeError("No candidates passed premarket volume, relative volume, and spread filters.")
        log.info(
            "Dynamic universe selected %d/%d candidates: %s",
            len(selected), len(candidates),
            ",".join(f"{candidate.symbol}(ADV=${candidate.average_dollar_volume:,.0f},"
                     f"ATR={candidate.atr_pct:.2%},PM={candidate.relative_premarket_volume:.2%},"
                     f"spread={candidate.spread_pct:.3%})" for candidate in selected),
        )
        return selected
