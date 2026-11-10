"""Paper-only Alpaca runner for the existing intraday alpha and risk engines."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from backtest.data import MinuteBar, data_quality, normalize
from backtest.cpcv import evaluate_cpcv
from backtest.engine import BTConfig
from backtest.holdout import validate_final_holdout
from backtest.metrics import summarize, trade_equity_curve
from backtest.walkforward import apply_params, walk_forward
from config.settings import (
    BrokerName, Settings, TradingMode, load_project_env, load_settings,
)
from engine.data_pipeline import Quote, SymbolState, Trade, atr, bbwp
from engine.strategy import (
    AlphaEngine, MultiTimeframeFilter, RiskEngine, Side, simple_moving_average,
)
from server.broker_adapters import AlpacaPaperAdapter, BaseBrokerAdapter
from server.ingestion.brain import KnowledgeBrain
from server.io_utils import (
    apply_shared_umask,
    atomic_write_json,
    ensure_directory,
    secure_file,
)
from server.time_utils import parse_iso_timestamp
from server.risk_engine import RiskGateClient, RiskGateRejected, RiskGateUnavailable
from server.universe import DynamicUniverseSelector
from server.event_guard import GuardDecision, MacroEventGuard
from server.evolution import register_trials, stage_candidate

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
BAR_INTERVAL = timedelta(minutes=1)
POLL_SECONDS = 15
MAX_BACKOFF_SECONDS = 60
HEARTBEAT_PATH = ROOT / "logs" / "runner_heartbeat.json"
HEARTBEAT_INTERVAL_SECONDS = 15
ENTRY_START = datetime_time(9, 45)
ENTRY_END = datetime_time(15, 30)
OPENING_PHASE_END = datetime_time(10, 15)
MIDDAY_PHASE_START = datetime_time(12, 0)
MIDDAY_PHASE_END = datetime_time(14, 0)
POWER_HOUR_START = datetime_time(15, 0)
FLATTEN_AT = datetime_time(15, 50)
MAX_RISK_PER_TRADE_PCT = 0.01
MAX_DAILY_DRAWDOWN_PCT = 0.02
OPENING_MIN_RVOL = 1.5
OPENING_ATR_STOP_MULT = 1.0
CRYPTO_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,12}/(?:USD|USDC|USDT)$")
log = logging.getLogger("apex.autonomous")
_heartbeat_lock = threading.Lock()
_heartbeat_telemetry: dict[str, Any] | None = None
_heartbeat_crypto_symbols: tuple[str, ...] = ()
_heartbeat_risk_gate: dict[str, Any] | None = None


def write_heartbeat(status: str, message: str = "", symbols: tuple[str, ...] = (),
                    telemetry: dict[str, Any] | None = None,
                    crypto_symbols: tuple[str, ...] | None = None,
                    risk_gate: dict[str, Any] | None = None) -> bool:
    """Publish runner heartbeat best-effort; telemetry failures never crash the runner."""
    global _heartbeat_telemetry, _heartbeat_crypto_symbols, _heartbeat_risk_gate
    with _heartbeat_lock:
        if telemetry is not None:
            _heartbeat_telemetry = telemetry
        if crypto_symbols is not None:
            _heartbeat_crypto_symbols = crypto_symbols
        if risk_gate is not None:
            _heartbeat_risk_gate = risk_gate
        record = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "status": status,
            "message": message,
            "symbols": list(symbols),
            "crypto_symbols": list(_heartbeat_crypto_symbols),
            "telemetry": _heartbeat_telemetry,
            "risk_gate": _heartbeat_risk_gate,
        }
        try:
            atomic_write_json(HEARTBEAT_PATH, record)
            return True
        except (OSError, TypeError, ValueError):
            log.exception("Unable to publish runner heartbeat to %s.", HEARTBEAT_PATH)
            return False


def write_dashboard_paper_snapshot(
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> None:
    """Publish a sanitized, read-only view for the isolated dashboard container."""
    account_fields = (
        "status", "currency", "equity", "last_equity", "cash", "buying_power",
        "portfolio_value", "trading_blocked", "account_blocked",
    )
    position_fields = (
        "symbol", "qty", "side", "market_value", "cost_basis", "unrealized_pl",
        "unrealized_plpc", "current_price", "avg_entry_price",
    )
    order_fields = (
        "submitted_at", "symbol", "side", "qty", "filled_qty", "type", "status",
        "filled_at",
    )

    def select_fields(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            key: value for key, value in item.items()
            if key in fields and (value is None or isinstance(value, (str, int, float, bool)))
        }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "PAPER",
        "paper_api_status": "connected",
        "account": select_fields(account, account_fields),
        "positions": [select_fields(item, position_fields) for item in positions],
        "orders": [select_fields(item, order_fields) for item in orders],
    }
    path = ROOT / "logs" / "paper_dashboard_snapshot.json"
    try:
        atomic_write_json(path, record)
    except (OSError, TypeError, ValueError):
        log.exception("Unable to publish paper account snapshot for the read-only dashboard.")


def write_dashboard_market_snapshot(market: dict[str, dict[str, Any]]) -> None:
    """Atomically publish sanitized quote, candle, and indicator data for the read-only UI."""
    allowed_fields = {
        "asset_class", "display_symbol", "price", "bid", "ask", "timestamp",
        "trend_50d", "sma50", "atr", "atr_pct", "bbwp", "regime", "candles",
    }
    clean_market = {}
    for symbol, values in market.items():
        if not isinstance(symbol, str) or not isinstance(values, dict):
            continue
        clean = {
            key: value for key, value in values.items()
            if key in allowed_fields and isinstance(value, (str, int, float, list))
            and not isinstance(value, bool)
        }
        candles = clean.get("candles")
        clean["candles"] = [
            {"time": candle["time"], "close": candle["close"]}
            for candle in (candles if isinstance(candles, list) else [])
            if isinstance(candle, dict)
            and isinstance(candle.get("time"), str)
            and isinstance(candle.get("close"), (int, float))
            and math.isfinite(candle["close"])
        ][-120:]
        for key in ("price", "bid", "ask", "sma50", "atr", "atr_pct", "bbwp"):
            if key in clean and (
                not isinstance(clean[key], (int, float))
                or not math.isfinite(clean[key])
            ):
                clean.pop(key)
        clean_market[symbol] = clean

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": clean_market,
    }
    path = ROOT / "logs" / "market_dashboard_snapshot.json"
    try:
        atomic_write_json(path, record)
    except (OSError, TypeError, ValueError):
        log.exception("Unable to publish sanitized market data for the read-only dashboard.")


class AlpacaAPIError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"Alpaca HTTP {status}: {detail}")
        self.status = status


def _is_transient_runner_error(error: Exception) -> bool:
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    return isinstance(error, AlpacaAPIError) and (
        error.status in {408, 429} or 500 <= error.status < 600
    )


class AlpacaClient:
    """Small standard-library REST client; order requests always use the paper host."""

    def __init__(self, key: str, secret: str, feed: str = "iex"):
        self.paper_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.feed = feed

    def request(self, base: str, path: str, method: str = "GET",
                params: dict[str, Any] | None = None, payload: dict[str, Any] | None = None):
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                data = response.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise AlpacaAPIError(exc.code, detail) from exc

    def account(self):
        return self.request(self.paper_url, "/v2/account")

    def clock(self):
        return self.request(self.paper_url, "/v2/clock")

    def positions(self):
        return self.request(self.paper_url, "/v2/positions")

    def assets(self):
        return self.request(self.paper_url, "/v2/assets", params={
            "status": "active", "asset_class": "us_equity",
        })

    def most_actives(self):
        response = self.request(self.data_url, "/v1beta1/screener/stocks/most-actives",
                                params={"top": 100})
        return response.get("most_actives", [])

    def open_orders(self, symbols: tuple[str, ...]):
        return self.request(self.paper_url, "/v2/orders", params={
            "status": "open", "nested": "true", "symbols": ",".join(symbols), "limit": 500,
        })

    def orders(self, symbol: str):
        return self.request(self.paper_url, "/v2/orders", params={
            "status": "all", "nested": "true", "symbols": symbol, "limit": 500,
            "direction": "desc",
        })

    def recent_orders(self, limit: int = 20):
        return self.request(self.paper_url, "/v2/orders", params={
            "status": "all", "nested": "true", "limit": min(limit, 100),
            "direction": "desc",
        })

    def bars(self, symbols: tuple[str, ...], timeframe: str, start: datetime, end: datetime,
             limit: int):
        crypto_symbols = tuple(symbol for symbol in symbols if "/" in symbol)
        equity_symbols = tuple(symbol for symbol in symbols if "/" not in symbol)
        if crypto_symbols and equity_symbols:
            result = defaultdict(list)
            result.update(self.bars(equity_symbols, timeframe, start, end, limit))
            result.update(self.bars(crypto_symbols, timeframe, start, end, limit))
            return result
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        token = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(symbols),
                "timeframe": timeframe,
                "start": start.astimezone(timezone.utc).isoformat(),
                "end": end.astimezone(timezone.utc).isoformat(),
                "limit": min(limit, 10000),
            }
            if crypto_symbols:
                base, path = self.data_url, "/v1beta3/crypto/us/bars"
            else:
                base, path = self.data_url, "/v2/stocks/bars"
                params.update({"feed": self.feed, "adjustment": "split"})
            if token:
                params["page_token"] = token
            page = self.request(base, path, params=params)
            for symbol, values in page.get("bars", {}).items():
                result[symbol].extend(values)
            token = page.get("next_page_token")
            if not token:
                return result

    def latest_quotes(self, symbols: tuple[str, ...]):
        crypto_symbols = tuple(symbol for symbol in symbols if "/" in symbol)
        equity_symbols = tuple(symbol for symbol in symbols if "/" not in symbol)
        quotes = {}
        if equity_symbols:
            response = self.request(self.data_url, "/v2/stocks/quotes/latest", params={
                "symbols": ",".join(equity_symbols), "feed": self.feed,
            })
            quotes.update(response.get("quotes", {}))
        if crypto_symbols:
            response = self.request(self.data_url, "/v1beta3/crypto/us/latest/quotes", params={
                "symbols": ",".join(crypto_symbols),
            })
            quotes.update(response.get("quotes", {}))
        return quotes

    def submit_bracket(self, payload: dict[str, Any]):
        client_id = payload["client_order_id"]
        params = {"client_order_id": client_id}
        try:
            return self.request(self.paper_url, "/v2/orders:by_client_order_id", params=params)
        except AlpacaAPIError as exc:
            if exc.status != 404:
                raise
        try:
            return self.request(self.paper_url, "/v2/orders", method="POST", payload=payload)
        except (urllib.error.URLError, TimeoutError):
            # A timeout can happen after Alpaca accepted the order; reconcile by id before retrying.
            try:
                return self.request(self.paper_url, "/v2/orders:by_client_order_id", params=params)
            except AlpacaAPIError as exc:
                if exc.status == 404:
                    raise
                raise

    def update_order(self, order_id: str, payload: dict[str, Any]):
        return self.request(self.paper_url, f"/v2/orders/{urllib.parse.quote(order_id, safe='')}",
                            method="PATCH", payload=payload)

    def close_position(self, symbol: str):
        return self.request(self.paper_url, f"/v2/positions/{urllib.parse.quote(symbol, safe='')}",
                            method="DELETE")

    def cancel_orders(self, symbols: tuple[str, ...]):
        for order in self.open_orders(symbols):
            order_id = order.get("id")
            if order_id:
                self.request(self.paper_url, f"/v2/orders/{urllib.parse.quote(order_id, safe='')}",
                             method="DELETE")


@dataclass
class ManagedPosition:
    symbol: str
    side: Side
    qty: int
    entry: float
    initial_stop: float
    stop: float
    extreme: float
    stop_order_id: str | None = None
    stop_order_ids: list[str] | None = None
    initial_qty: int = 0
    partial_qty: int = 0
    runner_qty: int = 0
    partial_taken: bool = False
    entry_order_ids: list[str] | None = None
    entry_tranches_expected: int = 0


def configure_logging() -> None:
    apply_shared_umask()
    log_dir = ensure_directory(ROOT / "logs")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            entry = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(entry, ensure_ascii=True)

    handlers: list[logging.Handler] = []
    try:
        file_handler = RotatingFileHandler(
            log_dir / "autonomous.log", maxBytes=5_000_000,
            backupCount=5, encoding="utf-8",
        )
        secure_file(log_dir / "autonomous.log")
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    except OSError:
        log.exception("Unable to open the persistent runner log; continuing on journald only.")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handlers.append(console_handler)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root_logger.addHandler(handler)
    log.handlers.clear()
    log.propagate = True


@contextmanager
def single_instance_lock():
    lock_path = ROOT / "logs" / "autonomous.lock"
    ensure_directory(lock_path.parent)
    lock_file = lock_path.open("a+b")
    secure_file(lock_path)
    acquired = False
    try:
        if os.name == "nt":
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "Another autonomous runner instance is already active."
                ) from exc
        else:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    "Another autonomous runner instance is already active."
                ) from exc
        acquired = True
        yield
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def load_paper_settings() -> Settings:
    load_project_env(ROOT)
    settings = load_settings(require_credentials=False)
    if settings.mode is not TradingMode.PAPER:
        raise RuntimeError("Autonomous runner is paper-only: set TRADING_MODE=PAPER.")
    if settings.broker is not BrokerName.ALPACA:
        raise RuntimeError("Autonomous runner requires BROKER=ALPACA.")
    if not settings.alpaca_key or not settings.alpaca_secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env.")
    if not settings.watchlist or any(not symbol.isalpha() for symbol in settings.watchlist):
        raise RuntimeError("WATCHLIST must contain valid US-equity symbols.")
    risk = settings.risk
    if not math.isfinite(risk.risk_per_trade_pct) or risk.risk_per_trade_pct <= 0:
        raise RuntimeError("RISK_PER_TRADE_PCT must be a finite positive value.")
    if risk.risk_per_trade_pct > MAX_RISK_PER_TRADE_PCT:
        log.warning("Clamping configured per-trade risk to the paper runner maximum of %.2f%%.",
                    MAX_RISK_PER_TRADE_PCT * 100)
        risk = replace(risk, risk_per_trade_pct=MAX_RISK_PER_TRADE_PCT)
    if not math.isfinite(risk.daily_drawdown_kill_pct) or risk.daily_drawdown_kill_pct <= 0:
        raise RuntimeError("DAILY_DRAWDOWN_KILL_PCT must be a finite positive value.")
    if risk.daily_drawdown_kill_pct > MAX_DAILY_DRAWDOWN_PCT:
        log.warning("Clamping configured session drawdown guard to the %.2f%% maximum.",
                    MAX_DAILY_DRAWDOWN_PCT * 100)
        risk = replace(risk, daily_drawdown_kill_pct=MAX_DAILY_DRAWDOWN_PCT)
    if risk.breakeven_at_r != 1.5:
        log.info("Setting runner break-even trigger to 1.5R, aligned with the first profit tranche.")
        risk = replace(risk, breakeven_at_r=1.5)
    return replace(settings, risk=risk)


def load_crypto_symbols() -> tuple[str, ...]:
    """Return explicitly configured crypto symbols; an empty setting disables polling."""
    raw_symbols = os.getenv("CRYPTO_SYMBOLS", "").split(",")
    symbols = tuple(dict.fromkeys(value.strip().upper() for value in raw_symbols if value.strip()))
    invalid = [symbol for symbol in symbols if not CRYPTO_SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid:
        raise RuntimeError(
            "CRYPTO_SYMBOLS must use supported slash-pair notation such as BTC/USD: "
            + ", ".join(invalid)
        )
    return symbols


def _bar_time(raw: str) -> datetime:
    return parse_iso_timestamp(raw).astimezone(timezone.utc)


def _feed_bar(state: SymbolState, bar: dict[str, Any]) -> None:
    """Approximate tape delta from OHLCV, matching the backtest's documented bar model."""
    ts = _bar_time(bar["t"])
    open_, high, low, close, volume = (float(bar[k]) for k in ("o", "h", "l", "c", "v"))
    buy_fraction = 0.5 if high <= low else (close - low) / (high - low)
    points = [open_, low, high, close] if close >= open_ else [open_, high, low, close]
    for index, price in enumerate(points):
        size = volume / 4
        trade_time = ts + timedelta(seconds=15 * index)
        state.on_trade(Trade(trade_time, state.symbol, price, size * buy_fraction, 1))
        state.on_trade(Trade(trade_time, state.symbol, price, size * (1 - buy_fraction), -1))


def _adx(bars, period: int = 14) -> float | None:
    if len(bars) < period * 2 + 1:
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
    value = sum(dx_values[:period]) / period
    for current in dx_values[period:]:
        value = (value * (period - 1) + current) / period
    return value


def _relative_volume(state: SymbolState, current_bar, lookback: int = 20) -> float | None:
    current_time = current_bar.start.astimezone(NY)
    minute_of_day = current_time.hour * 60 + current_time.minute
    history = [
        bar.volume for bar in state.agg.bars["1m"]
        if (
            bar.start.astimezone(NY).date() < current_time.date()
            and abs(
                bar.start.astimezone(NY).hour * 60
                + bar.start.astimezone(NY).minute - minute_of_day
            ) <= 2
        )
        and bar.volume > 0
    ][-lookback:]
    if len(history) < 5:
        return None
    baseline = median(history)
    return current_bar.volume / baseline if baseline > 0 else None


def _opening_range_breakout(state: SymbolState, now: datetime, side: Side) -> bool:
    local_now = now.astimezone(NY)
    session_bars = [
        bar for bar in state.agg.bars["1m"]
        if bar.start.astimezone(NY).date() == local_now.date()
    ]
    opening = [
        bar for bar in session_bars
        if datetime_time(9, 30) <= bar.start.astimezone(NY).time() < datetime_time(9, 45)
    ]
    if len(opening) < 15 or not session_bars:
        return False
    opening_high = max(bar.high for bar in opening)
    opening_low = min(bar.low for bar in opening)
    last = session_bars[-1]
    return last.close > opening_high if side is Side.LONG else last.close < opening_low


def _raw_bars(rows: list[dict[str, Any]], interval: timedelta, now: datetime):
    from engine.data_pipeline import Bar

    completed = []
    for row in sorted(rows, key=lambda item: item["t"]):
        start = _bar_time(row["t"])
        if start + interval > now:
            continue
        completed.append(Bar(
            start,
            float(row["o"]),
            float(row["h"]),
            float(row["l"]),
            float(row["c"]),
            float(row["v"]),
        ))
    return completed


def _resample_bars(bars, interval_minutes: int):
    grouped = {}
    interval_seconds = interval_minutes * 60
    for bar in bars:
        bucket_epoch = int(bar.start.timestamp())
        bucket_epoch -= bucket_epoch % interval_seconds
        bucket = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
        current = grouped.get(bucket)
        if current is None:
            grouped[bucket] = [
                bar.open, bar.high, bar.low, bar.close, bar.volume, bar.pv,
            ]
        else:
            current[1] = max(current[1], bar.high)
            current[2] = min(current[2], bar.low)
            current[3] = bar.close
            current[4] += bar.volume
            current[5] += bar.pv
    from engine.data_pipeline import Bar

    return [
        Bar(start, values[0], values[1], values[2], values[3], values[4], pv=values[5])
        for start, values in sorted(grouped.items())
    ]


def _quote(raw: dict[str, Any], symbol: str) -> Quote:
    return Quote(
        ts=_bar_time(raw["t"]),
        symbol=symbol,
        bid=float(raw["bp"]),
        bid_size=float(raw["bs"]),
        ask=float(raw["ap"]),
        ask_size=float(raw["as"]),
    )


class AutonomousRunner:
    def __init__(
        self,
        settings: Settings,
        client: AlpacaClient,
        broker_adapter: BaseBrokerAdapter | None = None,
        risk_gate: RiskGateClient | None = None,
    ):
        self.settings = settings
        self.client = client
        self.broker_adapter = broker_adapter or AlpacaPaperAdapter(client)
        self.risk_gate = risk_gate or RiskGateClient()
        self.symbols = settings.watchlist
        self.crypto_symbols = load_crypto_symbols()
        self.risk = RiskEngine(settings.risk)
        self.alpha = AlphaEngine(self.risk, settings.min_signal_score)
        self._load_approved_candidate()
        self.mtf_filter = MultiTimeframeFilter()
        self.states: dict[str, SymbolState] = {}
        self.hourly_bars: dict[str, list[Any]] = {}
        self.daily_bars: dict[str, list[Any]] = {}
        self.mtf_refreshed_at: datetime | None = None
        self.crypto_last_research_at: datetime | None = None
        self.crypto_last_bar: dict[str, datetime] = {}
        self.crypto_mtf_refreshed_at: datetime | None = None
        self.crypto_error_last_logged_at: float | None = None
        self.crypto_market_data: dict[str, dict[str, Any]] = {}
        self.dashboard_market_last_refresh_at: datetime | None = None
        self.dashboard_market_error_last_logged_at: float | None = None
        self.watermarks: dict[str, datetime] = {}
        self.managed: dict[str, ManagedPosition] = {}
        self.last_quotes: dict[str, Quote] = {}
        self.initialized = False
        self.session_day = None
        self.session_equity = 0.0
        self.halted = False
        self.risk_gateway_halted = False
        self.risk_gateway_halt_reason = ""
        self.risk_gate_restricted = False
        self.risk_gate_restricted_reason = ""
        self.last_reconciliation_at = 0.0
        self.flattened_day = None
        self.prepared_day: date | None = None
        self.heartbeat_status = "waiting"
        self.heartbeat_message = "Runner started; checking Alpaca market clock."
        macro_buffer = int(os.getenv("MACRO_EVENT_BUFFER_MINUTES", "15"))
        volatility_spike_pct = float(os.getenv("SPY_VOLATILITY_SPIKE_PCT", "0.01"))
        self.event_guard = MacroEventGuard(
            ROOT / os.getenv("MACRO_EVENT_FILE", "data/macro_events.json"),
            buffer_minutes=macro_buffer,
            volatility_spike_pct=volatility_spike_pct,
        )
        self.selector = DynamicUniverseSelector(
            client,
            top_n=int(os.getenv("UNIVERSE_TOP_N", "10")),
            constituents_file=ROOT / os.getenv("UNIVERSE_FILE", "data/universe.csv"),
            min_average_dollar_volume=float(os.getenv("UNIVERSE_MIN_ADV", "20000000")),
            min_atr_pct=float(os.getenv("UNIVERSE_MIN_ATR_PCT", "0.005")),
            max_atr_pct=float(os.getenv("UNIVERSE_MAX_ATR_PCT", "0.08")),
            max_spread_pct=float(os.getenv("UNIVERSE_MAX_SPREAD_PCT", "0.0025")),
            min_premarket_volume=float(os.getenv("UNIVERSE_MIN_PREMARKET_VOLUME", "100")),
            min_relative_premarket_volume=float(os.getenv("UNIVERSE_MIN_RELATIVE_PREMARKET", "0.0005")),
        )

    def _order_routing_authorized(self, asset_class: str) -> bool:
        if getattr(self, "risk_gate_restricted", False):
            log.info(
                "Order routing locked: risk gateway is in read-only configuration fallback."
            )
            return False
        if os.getenv("ENABLE_ORDER_ROUTING", "false").strip().lower() != "true":
            log.info("Order routing locked: ENABLE_ORDER_ROUTING is not true.")
            return False
        if self.settings.mode is not TradingMode.PAPER:
            log.error("Order routing locked: autonomous runner supports PAPER mode only.")
            return False
        try:
            brain = KnowledgeBrain(ROOT / "data" / "knowledge_brain.sqlite")
            promoted = brain.has_promoted_strategy(asset_class)
        except Exception:
            log.exception(
                "Order routing locked for %s: Knowledge Brain is not readable.", asset_class,
            )
            return False
        if not promoted:
            log.info(
                "Order routing locked for %s: no Gate 1-3 promoted strategy is available.",
                asset_class,
            )
            return False
        return True

    def _risk_gate_configuration(self) -> dict[str, Any]:
        getter = getattr(self.risk_gate, "configuration_status", None)
        if callable(getter):
            try:
                status = getter()
            except Exception:
                log.exception("Unable to read risk gateway configuration status.")
                status = None
            if isinstance(status, dict):
                return status
        # Unknown client implementations are assumed configured; health() decides.
        return {"configured": True, "reason": None}

    def _degraded_read_only_allowed(self) -> bool:
        """Allow a read-only fallback only for non-production PAPER deployments."""
        if self.settings.mode is not TradingMode.PAPER:
            return False
        environment = os.getenv("APEX_ENV", "development").strip().lower()
        if environment in {"prod", "production", "live"}:
            return False
        return os.getenv(
            "ALLOW_RISK_GATE_DEGRADED", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _is_risk_configuration_error(reason: str) -> bool:
        return "not configured" in reason.strip().lower()

    def _risk_gate_telemetry(self) -> dict[str, Any]:
        if self.risk_gateway_halted:
            state = "halted"
        elif self.risk_gate_restricted:
            state = "restricted"
        else:
            state = "ok"
        configuration = self._risk_gate_configuration()
        return {
            "state": state,
            "reason": (
                self.risk_gateway_halt_reason
                or self.risk_gate_restricted_reason
                or ""
            ),
            "configured": configuration.get("configured"),
            "base_url": configuration.get("base_url"),
        }

    def _enter_read_only_mode(self, reason: str) -> None:
        if self.risk_gate_restricted and self.risk_gate_restricted_reason == reason:
            return
        self.risk_gate_restricted = True
        self.risk_gate_restricted_reason = reason
        self.heartbeat_status = "restricted"
        self.heartbeat_message = (
            f"RISK CONFIG WARNING: {reason} Runner is in read-only paper-safe mode; "
            "no broker orders will be submitted."
        )
        log.warning(self.heartbeat_message)
        try:
            write_heartbeat(
                self.heartbeat_status, self.heartbeat_message, self.symbols,
                risk_gate=self._risk_gate_telemetry(),
                telemetry={
                    "event": "risk_gate_configuration_warning",
                    "reason": reason,
                    "mode": "read_only",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except OSError:
            log.exception("Unable to publish the risk configuration warning heartbeat.")

    def _exit_read_only_mode(self) -> None:
        if not self.risk_gate_restricted:
            return
        self.risk_gate_restricted = False
        self.risk_gate_restricted_reason = ""
        log.info("Risk gateway authentication restored; leaving read-only mode.")

    def _halt_for_risk_gateway(self, reason: str) -> None:
        self.risk_gateway_halted = True
        self.risk_gateway_halt_reason = reason
        self.halted = True
        self.risk_gate_restricted = False
        self.risk_gate_restricted_reason = ""
        self.heartbeat_status = "disconnected"
        self.heartbeat_message = (
            f"RISK GATE HALT: {reason}; all broker order changes are stopped."
        )
        log.critical(self.heartbeat_message)
        if self.session_day is not None:
            try:
                self._persist_session_guard()
            except OSError:
                log.exception("Unable to persist the risk gateway halt.")
        try:
            write_heartbeat(
                self.heartbeat_status, self.heartbeat_message, self.symbols,
                risk_gate=self._risk_gate_telemetry(),
                telemetry={
                    "event": "risk_gateway_halt",
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except OSError:
            log.exception("Unable to publish the risk gateway halt heartbeat.")

    def _check_risk_gateway(self) -> bool:
        if self.risk_gateway_halted:
            return False
        configuration = self._risk_gate_configuration()
        if configuration.get("configured") is False:
            reason = str(
                configuration.get("reason")
                or "Risk gateway authentication is not configured."
            )
            if self._degraded_read_only_allowed():
                self._enter_read_only_mode(reason)
                return True
            self._halt_for_risk_gateway(reason)
            return False
        try:
            self.risk_gate.health()
        except RiskGateUnavailable as exc:
            reason = str(exc)
            if (
                self._is_risk_configuration_error(reason)
                and self._degraded_read_only_allowed()
            ):
                self._enter_read_only_mode(reason)
                return True
            self._halt_for_risk_gateway(reason)
            return False
        self._exit_read_only_mode()
        return True

    def _authorize_order_mutation(self, operation: str, **details: Any) -> None:
        if self.risk_gateway_halted or self.risk_gate_restricted:
            raise RiskGateUnavailable(
                self.risk_gateway_halt_reason
                or self.risk_gate_restricted_reason
                or "risk gateway is not available for order mutations"
            )
        request = {
            "request_id": str(uuid.uuid4()),
            "operation": operation,
            **details,
        }
        try:
            decision = self.risk_gate.authorize(request)
        except RiskGateUnavailable as exc:
            self._halt_for_risk_gateway(str(exc))
            raise
        if decision.get("approved") is not True:
            reason = str(decision.get("reason", "risk gateway rejected order mutation"))
            self._halt_for_risk_gateway(reason)
            raise RiskGateRejected(reason)

    def _load_approved_candidate(self) -> None:
        path = ROOT / "data" / "active_paper_candidate.json"
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning("Ignoring unreadable approved candidate %s: %s", path, exc)
            return
        if (candidate.get("status") != "approved_for_paper"
                or candidate.get("human_approved") is not True):
            log.warning("Ignoring candidate profile without explicit paper approval: %s", path)
            return
        params = candidate.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"Approved candidate has no valid parameter profile: {path}")
        base = BTConfig(
            risk=self.settings.risk,
            min_signal_score=self.settings.min_signal_score,
        )
        profile = apply_params(base, params)
        self.risk = RiskEngine(profile.risk)
        self.alpha = AlphaEngine(self.risk, profile.min_signal_score)
        log.info("Loaded human-approved paper candidate %s.", candidate.get("candidate_id"))

    def _ensure_no_existing_exposure(self) -> None:
        scope = tuple(dict.fromkeys(self.settings.watchlist + self.symbols))
        existing_positions = {p["symbol"] for p in self.client.positions()}
        existing_orders = {o["symbol"] for o in self.client.open_orders(scope)}
        overlap = (existing_positions | existing_orders) & set(scope)
        if overlap:
            raise RuntimeError(
                "Refusing to start with existing target-symbol positions/orders: "
                + ", ".join(sorted(overlap))
            )

    def _cleanup_identified_test_positions(self, market_open: bool) -> bool:
        """Close only filled positions attributable to the explicitly named paper test-order IDs."""
        if self.risk_gate_restricted:
            return False
        positions = {
            position["symbol"]: position for position in self.client.positions()
            if position.get("symbol") in self.settings.watchlist
        }
        waiting = False
        for symbol, position in positions.items():
            tests = [
                order for order in self.client.orders(symbol)
                if str(order.get("client_order_id", "")).startswith("apex-spy-paper-test-")
                and order.get("status") == "filled"
            ]
            qty = abs(float(position.get("qty", 0)))
            matching = [
                order for order in tests
                if abs(float(order.get("filled_qty", 0)) - qty) < 1e-8
                and (order.get("side") == "buy") == (position.get("side") == "long")
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"Existing {symbol} position is not uniquely attributable to a known paper test order; "
                    "refusing to liquidate or trade."
                )
            if not market_open:
                log.warning("Identified orphan test position %s qty=%s; waiting for market open to close.",
                            symbol, position.get("qty"))
                waiting = True
                continue
            closing_orders = [
                order for order in self.client.open_orders((symbol,))
                if order.get("side") == ("sell" if position.get("side") == "long" else "buy")
                and order.get("type") == "market"
            ]
            if closing_orders:
                log.info("Close order already pending for identified test position %s.", symbol)
                waiting = True
                continue
            self._authorize_order_mutation(
                "close_position", symbol=symbol, purpose="identified_test_cleanup",
            )
            result = self.client.close_position(symbol)
            log.warning("Submitted close for identified paper test position %s qty=%s order=%s",
                        symbol, position.get("qty"), result.get("id", "unknown"))
            waiting = True
        return waiting

    def _refresh_mtf_context(self, now: datetime, force: bool = False) -> None:
        if self.mtf_refreshed_at and not force and now - self.mtf_refreshed_at < timedelta(minutes=15):
            return
        if not self.symbols:
            return
        daily_start = now - timedelta(days=400)
        hourly_start = now - timedelta(days=90)
        daily = self.client.bars(self.symbols, "1Day", daily_start, now, limit=500)
        hourly = self.client.bars(self.symbols, "1Hour", hourly_start, now, limit=3000)
        self.daily_bars = {
            symbol: _raw_bars(daily.get(symbol, []), timedelta(days=1), now)
            for symbol in self.symbols
        }
        self.hourly_bars = {
            symbol: _raw_bars(hourly.get(symbol, []), timedelta(hours=1), now)
            for symbol in self.symbols
        }
        self.mtf_refreshed_at = now
        log.info(
            "MTF context refreshed: hourly SMA20/50/200 and ADX; daily SMA20/50/200 and ADX."
        )

    def _run_crypto_research(self, now: datetime) -> None:
        if not self.crypto_symbols:
            return
        if (self.crypto_last_research_at
                and now - self.crypto_last_research_at < timedelta(minutes=1)):
            return
        self.crypto_last_research_at = now
        if (self.crypto_mtf_refreshed_at is None
                or now - self.crypto_mtf_refreshed_at >= timedelta(hours=1)):
            daily = self.client.bars(
                self.crypto_symbols, "1Day", now - timedelta(days=400), now, limit=500
            )
            hourly = self.client.bars(
                self.crypto_symbols, "1Hour", now - timedelta(days=90), now, limit=3000
            )
            for symbol in self.crypto_symbols:
                self.daily_bars[symbol] = _raw_bars(
                    daily.get(symbol, []), timedelta(days=1), now
                )
                self.hourly_bars[symbol] = _raw_bars(
                    hourly.get(symbol, []), timedelta(hours=1), now
                )
            self.crypto_mtf_refreshed_at = now
        start = now - timedelta(minutes=45)
        raw_bars = self.client.bars(self.crypto_symbols, "1Min", start, now, limit=100)
        quotes = self.client.latest_quotes(self.crypto_symbols)
        for symbol in self.crypto_symbols:
            rows = raw_bars.get(symbol, [])
            latest = _raw_bars(rows, BAR_INTERVAL, now)
            quote_data = quotes.get(symbol)
            if not latest or not quote_data:
                log.info("CRYPTO RESEARCH %s data unavailable; execution remains disabled.", symbol)
                continue
            quote = _quote(quote_data, symbol)
            latest_bar = latest[-1]
            prior_time = self.crypto_last_bar.get(symbol)
            if prior_time is None or latest_bar.start > prior_time:
                self.crypto_last_bar[symbol] = latest_bar.start
                self.crypto_market_data[symbol] = {
                    "asset_class": "crypto",
                    "display_symbol": symbol.replace("/", "-"),
                    "price": latest_bar.close,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "timestamp": quote.ts.isoformat(),
                    "candles": [
                        {"time": bar.start.isoformat(), "close": bar.close}
                        for bar in latest[-120:]
                    ],
                    "regime": "24/7 · order routing disabled",
                }
                log.info(
                    "CRYPTO RESEARCH %s 24/7 mark=%.4f bid=%.4f ask=%.4f spread=%.3f%%; "
                    "hourly_bars=%d daily_bars=%d; paper execution disabled pending "
                    "broker-native protective exits.",
                    symbol, latest_bar.close, quote.bid, quote.ask,
                    (quote.ask - quote.bid) / quote.ask * 100 if quote.ask > 0 else 0,
                    len(self.hourly_bars.get(symbol, [])), len(self.daily_bars.get(symbol, [])),
                )

    def _log_crypto_research_failure(self, error: Exception) -> None:
        monotonic_now = time.monotonic()
        if (
            self.crypto_error_last_logged_at is None
            or monotonic_now - self.crypto_error_last_logged_at >= 300
        ):
            log.warning(
                "Crypto research data is unavailable; equity processing will continue "
                "(%s: %s).",
                type(error).__name__, error,
            )
            self.crypto_error_last_logged_at = monotonic_now

    def _prepare_session(self, now: datetime, next_open: datetime) -> None:
        candidates = self.selector.select(now)
        if not candidates:
            raise RuntimeError("Dynamic universe selection returned no candidates.")
        self.symbols = tuple(candidate.symbol for candidate in candidates)
        self.states = {symbol: SymbolState(symbol) for symbol in self.symbols}
        self.watermarks.clear()
        self.managed.clear()
        self.initialized = False
        self.prepared_day = next_open.astimezone(NY).date()
        self._ensure_no_existing_exposure()
        self._refresh_mtf_context(now, force=True)
        self._optimize_daily_profile(now)

    def _optimize_daily_profile(self, now: datetime) -> None:
        symbols = self.symbols[:min(3, len(self.symbols))]
        start = now - timedelta(days=int(os.getenv("OPTIMIZATION_HISTORY_DAYS", "180")))
        raw = self.client.bars(symbols, "1Min", start, now, limit=10000)
        data: dict[str, list[MinuteBar]] = {}
        for symbol in symbols:
            bars = [
                MinuteBar(_bar_time(row["t"]), float(row["o"]), float(row["h"]),
                          float(row["l"]), float(row["c"]), float(row["v"]))
                for row in raw.get(symbol, [])
            ]
            quality = data_quality(bars)
            if any(quality[name] for name in (
                "flatline_bars", "low_volume_bars", "missing_intervals"
            )):
                log.warning("HISTORICAL DATA QUALITY %s: %s", symbol, quality)
            data[symbol] = normalize(bars)
        if not any(len(bars) for bars in data.values()):
            log.warning("Daily optimization skipped: no historical minute bars; retaining baseline profile.")
            return
        account = self.client.account()
        base = BTConfig(
            initial_equity=float(account["equity"]),
            risk=self.settings.risk,
            min_signal_score=self.settings.min_signal_score,
            warmup_days=50,
            slippage_bps=float(os.getenv("BACKTEST_SLIPPAGE_BPS", "1.0")),
            slippage_atr_mult=float(os.getenv("BACKTEST_SLIPPAGE_ATR_MULT", "0.1")),
            commission_per_share=float(os.getenv("BACKTEST_COMMISSION_PER_SHARE", "0.0035")),
            commission_min=float(os.getenv("BACKTEST_COMMISSION_MIN", "0.35")),
            sell_fee_bps=float(os.getenv("BACKTEST_SELL_FEE_BPS", "0.3")),
        )
        cycle_id = f"{now.date().isoformat()}:{','.join(symbols)}"
        trial_ledger_path = ROOT / "logs" / "strategy_trial_ledger.json"
        folds, oos_trades, results = walk_forward(
            data, base, train_days=30, test_days=10,
            min_train_trades=8, workers=int(os.getenv("OPTIMIZATION_WORKERS", "2")),
            trial_cycle_id=cycle_id, trial_ledger_path=trial_ledger_path,
        )
        if not folds:
            log.warning("Daily optimization found no qualifying walk-forward folds; retaining baseline profile.")
            return
        latest = folds[-1]
        profile = apply_params(base, latest["params"])
        trial_count = register_trials(
            cycle_id, len(results), trial_ledger_path
        )
        cpcv_report = evaluate_cpcv(
            data, base, results, n_trials=trial_count,
            workers=int(os.getenv("OPTIMIZATION_WORKERS", "2")),
        )
        params_fingerprint = hashlib.sha256(
            json.dumps(latest["params"], sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        candidate_id = f"equity-daily-{now.date().strftime('%Y%m%d')}-{params_fingerprint}"
        oos_curve = trade_equity_curve(oos_trades, base.initial_equity)
        oos_metrics = summarize(oos_trades, oos_curve, base.initial_equity)
        holdout = validate_final_holdout(
            symbols, base, latest["params"], candidate_id,
        )
        slippage_drag = sum(trade.slippage_cost for trade in oos_trades)
        candidate = {
            "id": candidate_id,
            "candidate_id": candidate_id,
            "generated_at": now.isoformat(),
            "symbols": list(symbols),
            "params": latest["params"],
            "paper_trades": len(oos_trades),
            "sharpe": oos_metrics["sharpe_daily_ann"],
            "observations": len(oos_curve),
            "profit_factor": oos_metrics["profit_factor"],
            "max_drawdown_pct": oos_metrics["max_dd_pct"],
            "n_trials": trial_count,
            "pbo": cpcv_report["pbo"],
            "cpcv": cpcv_report,
            "expected_slippage_drag": slippage_drag,
            "simulated_slippage_drag": slippage_drag,
            "simulated_transaction_costs": oos_metrics["costs"],
            "forward_paper_expectancy_r": None,
        }
        audit = stage_candidate(
            candidate, holdout,
            ROOT / "logs" / "strategy_audit_report.json",
            ROOT / "data" / "staging",
        )

        def finite_or_none(value):
            return value if isinstance(value, (int, float)) and math.isfinite(value) else None

        profile_record = {
            "as_of": now.isoformat(),
            "symbols": list(symbols),
            "params": latest["params"],
            "train": latest["train"],
            "test": latest["test"],
            "train_mean_r": finite_or_none(latest["train_meanR"]),
            "test_mean_r": finite_or_none(latest["test_meanR"]),
            "test_trades": latest["test_n"],
            "test_win_rate": finite_or_none(latest["test_win_rate"]),
            "test_profit_factor": finite_or_none(latest["test_profit_factor"]),
            "status": audit["status"],
            "candidate_id": candidate_id,
            "human_approval_required": True,
            "n_trials": trial_count,
            "cpcv": cpcv_report,
            "pbo": finite_or_none(cpcv_report["pbo"]),
            "raw_sharpe": finite_or_none(oos_metrics["sharpe_daily_ann"]),
            "deflated_sharpe_probability": audit["deflated_sharpe_probability"],
            "holdout": holdout,
            "expected_slippage_drag": slippage_drag,
            "simulated_slippage_drag": slippage_drag,
            "simulated_transaction_costs": oos_metrics["costs"],
        }
        profile_path = ROOT / "logs" / "daily_profile.json"
        atomic_write_json(profile_path, profile_record, indent=2)
        log.info(
            "Daily walk-forward candidate %s; no active parameters changed without human approval: %s",
            audit["status"], json.dumps(profile_record, allow_nan=False),
        )
        write_heartbeat(
            "active",
            f"Optimization {audit['status']}; {trial_count} cumulative variants; "
            f"final holdout: {holdout.get('reason')}.",
            self.symbols,
            {
                "event": "strategy_candidate_audit",
                "candidate_id": candidate_id,
                "status": audit["status"],
                "n_trials": trial_count,
                "pbo": finite_or_none(cpcv_report["pbo"]),
                "cpcv_passed": cpcv_report["passed"],
                "raw_sharpe": audit["raw_sharpe"],
                "deflated_sharpe_probability": audit["deflated_sharpe_probability"],
                "expected_slippage_drag": slippage_drag,
                "simulated_slippage_drag": slippage_drag,
                "holdout_passed": holdout.get("passed", False),
                "holdout_reason": holdout.get("reason"),
            },
        )

    def preflight(self) -> None:
        self._ensure_no_existing_exposure()
        log.info("Startup target-exposure preflight passed.")

    def _initialize_market_state(self, now: datetime) -> None:
        start = now - timedelta(days=12)
        intraday = self.client.bars(self.symbols, "1Min", start, now, limit=10000)
        earliest = {}
        for symbol in self.symbols:
            values = sorted(intraday.get(symbol, []), key=lambda row: row["t"])
            completed = [b for b in values if _bar_time(b["t"]) + BAR_INTERVAL <= now]
            if not completed:
                raise RuntimeError(f"No completed intraday bars returned for {symbol}.")
            earliest[symbol] = _bar_time(completed[0]["t"]).astimezone(NY).date()
            for bar in completed:
                _feed_bar(self.states[symbol], bar)
            self.watermarks[symbol] = _bar_time(completed[-1]["t"])

        daily = self.client.bars(self.symbols, "1Day", now - timedelta(days=400), now, limit=500)
        for symbol in self.symbols:
            prior_closes = [
                float(row["c"]) for row in sorted(daily.get(symbol, []), key=lambda row: row["t"])
                if _bar_time(row["t"]).astimezone(NY).date() < earliest[symbol]
            ]
            self.states[symbol].daily_closes.extend(prior_closes[-50:])
            if len(self.states[symbol].daily_closes) < 50:
                raise RuntimeError(f"Need 50 prior daily closes to warm up {symbol}'s trend filter.")
        self.initialized = True
        log.info("Market state initialized from completed bars; watchlist=%s feed=%s",
                 ",".join(self.symbols), self.settings.alpaca_data_feed)

    def _refresh_quotes(self, now: datetime) -> None:
        for symbol, raw in self.client.latest_quotes(self.symbols).items():
            quote = _quote(raw, symbol)
            self.last_quotes[symbol] = quote
            self.states[symbol].on_quote(quote)
        for symbol, quote in list(self.last_quotes.items()):
            if (now - quote.ts).total_seconds() > 90:
                self.last_quotes.pop(symbol, None)
                self.states[symbol].last_quote = None

    def _publish_dashboard_market_snapshot(self) -> None:
        market = dict(self.crypto_market_data)
        for symbol in self.symbols:
            state = self.states.get(symbol)
            if state is None:
                continue
            quote = self.last_quotes.get(symbol)
            minute_bars = list(state.agg.bars["1m"])[-120:]
            five_minute_bars = list(state.agg.bars["5m"])
            daily_closes = state.daily_closes
            current_price = (
                quote.mid if quote is not None
                else minute_bars[-1].close if minute_bars
                else state.last_price
            )
            if current_price is None or not math.isfinite(current_price):
                continue
            sma50 = simple_moving_average(daily_closes, 50)
            trend = (
                "above" if sma50 is not None and current_price >= sma50
                else "below" if sma50 is not None else "unavailable"
            )
            atr_value = atr(five_minute_bars, 14)
            closes = [bar.close for bar in five_minute_bars]
            bbwp_value = bbwp(closes)
            regime = f"{trend} 50-day trend" if trend != "unavailable" else "trend unavailable"
            if bbwp_value is not None and bbwp_value < 20:
                regime += " · volatility squeeze"
            market[symbol] = {
                "asset_class": "equity",
                "display_symbol": symbol,
                "price": current_price,
                "bid": quote.bid if quote is not None else None,
                "ask": quote.ask if quote is not None else None,
                "timestamp": quote.ts.isoformat() if quote is not None else None,
                "trend_50d": trend,
                "sma50": sma50,
                "atr": atr_value,
                "atr_pct": (
                    atr_value / current_price * 100
                    if atr_value is not None and current_price > 0 else None
                ),
                "bbwp": bbwp_value,
                "regime": regime,
                "candles": [
                    {"time": bar.start.isoformat(), "close": bar.close}
                    for bar in minute_bars
                ],
            }
        write_dashboard_market_snapshot(market)

    def _publish_dashboard_market_snapshot_while_closed(self, now: datetime) -> None:
        if (
            self.dashboard_market_last_refresh_at is not None
            and now - self.dashboard_market_last_refresh_at < timedelta(minutes=5)
        ):
            return
        self.dashboard_market_last_refresh_at = now
        start = now - timedelta(days=7)
        minute_rows = self.client.bars(self.symbols, "1Min", start, now, limit=10000)
        daily_rows = self.client.bars(
            self.symbols, "1Day", now - timedelta(days=400), now, limit=500
        )
        market = dict(self.crypto_market_data)
        for symbol in self.symbols:
            minute_bars = _raw_bars(minute_rows.get(symbol, []), BAR_INTERVAL, now)[-120:]
            daily_bars = _raw_bars(
                daily_rows.get(symbol, []), timedelta(days=1), now
            )
            current_price = (
                minute_bars[-1].close if minute_bars
                else daily_bars[-1].close if daily_bars
                else None
            )
            if current_price is None or not math.isfinite(current_price):
                continue
            sma50 = simple_moving_average([bar.close for bar in daily_bars], 50)
            trend = (
                "above" if sma50 is not None and current_price >= sma50
                else "below" if sma50 is not None else "unavailable"
            )
            atr_value = atr(minute_bars, 14)
            bbwp_value = bbwp([bar.close for bar in minute_bars])
            regime = f"{trend} 50-day trend" if trend != "unavailable" else "trend unavailable"
            if bbwp_value is not None and bbwp_value < 20:
                regime += " · volatility squeeze"
            market[symbol] = {
                "asset_class": "equity",
                "display_symbol": symbol,
                "price": current_price,
                "timestamp": (
                    minute_bars[-1].start.isoformat()
                    if minute_bars else daily_bars[-1].start.isoformat()
                ),
                "trend_50d": trend,
                "sma50": sma50,
                "atr": atr_value,
                "atr_pct": (
                    atr_value / current_price * 100
                    if atr_value is not None and current_price > 0 else None
                ),
                "bbwp": bbwp_value,
                "regime": regime,
                "candles": [
                    {"time": bar.start.isoformat(), "close": bar.close}
                    for bar in minute_bars
                ],
            }
        write_dashboard_market_snapshot(market)

    def _log_dashboard_market_failure(self, error: Exception) -> None:
        monotonic_now = time.monotonic()
        if (
            self.dashboard_market_error_last_logged_at is None
            or monotonic_now - self.dashboard_market_error_last_logged_at >= 300
        ):
            log.warning(
                "Closed-market dashboard data is unavailable; paper execution is unaffected "
                "(%s: %s).",
                type(error).__name__, error,
            )
            self.dashboard_market_error_last_logged_at = monotonic_now

    def _account_and_positions(self):
        account = self.client.account()
        if account.get("trading_blocked") or account.get("account_blocked"):
            raise RuntimeError("Alpaca reports a blocked account; refusing to trade.")
        all_positions = self.client.positions()
        positions = {
            position["symbol"]: position for position in all_positions
            if position["symbol"] in self.symbols
        }
        orders = {o["symbol"]: o for o in self.client.open_orders(self.symbols)}
        try:
            recent_orders = self.client.recent_orders(limit=20)
        except Exception:
            log.exception("Unable to refresh recent paper orders for the dashboard snapshot.")
            recent_orders = list(orders.values())
        write_dashboard_paper_snapshot(account, all_positions, recent_orders)
        now = time.time()
        if now - self.last_reconciliation_at >= float(
            os.getenv("RISK_RECONCILIATION_INTERVAL_SECONDS", "60")
        ):
            try:
                report = self.risk_gate.reconcile({
                    "request_id": str(uuid.uuid4()),
                    "mode": self.settings.mode.value,
                    "account": account,
                    "positions": all_positions,
                    "target_symbols": list(self.symbols),
                    "managed_symbols": list(self.managed),
                    "source": "autonomous_runner",
                })
            except RiskGateUnavailable as exc:
                self._halt_for_risk_gateway(str(exc))
                raise
            if report.get("approved") is not True:
                reason = str(report.get("reason", "broker reconciliation failed"))
                self._halt_for_risk_gateway(reason)
                raise RiskGateRejected(reason)
            self.last_reconciliation_at = now
        return account, positions, orders

    def _publish_dashboard_snapshot_while_closed(self) -> None:
        try:
            write_dashboard_paper_snapshot(
                self.client.account(),
                self.client.positions(),
                self.client.recent_orders(limit=20),
            )
        except Exception:
            log.exception("Unable to refresh read-only paper dashboard data while market is closed.")

    def _macro_entry_guard(self, now: datetime) -> GuardDecision:
        scheduled = self.event_guard.check(now)
        if scheduled.blocked:
            log.warning("MACRO CIRCUIT BREAKER: %s until %s.",
                        scheduled.reason, scheduled.until)
            return scheduled
        index_rows = self.client.bars(
            ("SPY",), "1Min", now - timedelta(minutes=20), now, limit=30
        )
        index_bars = _raw_bars(index_rows.get("SPY", []), BAR_INTERVAL, now)
        decision = self.event_guard.check(now, index_bars)
        if decision.blocked:
            log.critical("MACRO VOLATILITY CIRCUIT BREAKER: %s until %s.",
                         decision.reason, decision.until)
        return decision

    def _start_session(self, now: datetime, account: dict[str, Any]) -> None:
        day = now.astimezone(NY).date()
        if day != self.session_day:
            self.session_day = day
            state_path = ROOT / "logs" / "session_guard.json"
            previous = {}
            try:
                previous = json.loads(state_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                previous = {}
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                log.warning("Ignoring unreadable session guard %s: %s", state_path, exc)
                previous = {}
            if previous.get("session_date") == day.isoformat():
                self.session_equity = float(previous["starting_equity"])
                self.halted = bool(previous.get("halted", False))
                log.info("Restored persisted session loss limit for %s: $%.2f",
                         day, self.session_equity)
            else:
                self.session_equity = float(account["equity"])
                self.halted = False
                self._persist_session_guard()
            log.info("NY session %s; starting equity=$%.2f", day, self.session_equity)

    def _persist_session_guard(self) -> None:
        state_path = ROOT / "logs" / "session_guard.json"
        try:
            atomic_write_json(state_path, {
                "session_date": self.session_day.isoformat(),
                "starting_equity": self.session_equity,
                "halted": self.halted,
            })
        except OSError:
            log.exception("Unable to persist the session guard state.")

    def _kill_switch(self, now: datetime, positions: dict[str, dict[str, Any]]) -> bool:
        if self.halted:
            if self.risk_gate_restricted:
                log.warning(
                    "Session halt active, but risk configuration fallback is read-only; "
                    "broker cancellation is suppressed."
                )
                return True
            self._authorize_order_mutation(
                "cancel_orders", symbols=list(self.symbols), purpose="session_halt",
            )
            self.client.cancel_orders(self.symbols)
            for symbol in positions:
                self._authorize_order_mutation(
                    "close_position", symbol=symbol, purpose="session_halt",
                )
                self.client.close_position(symbol)
                log.warning("Persisted session halt re-submitted close for %s", symbol)
            return True
        account = self.client.account()
        equity = float(account["equity"])
        limit = self.session_equity * (1 - self.settings.risk.daily_drawdown_kill_pct)
        if equity > limit:
            return False
        self.halted = True
        drawdown_pct = (self.session_equity - equity) / self.session_equity * 100
        message = (
            f"CRITICAL [CIRCUIT BREAKER] session drawdown {drawdown_pct:.2f}% reached; "
            f"equity=${equity:.2f}, threshold=${limit:.2f}. Halting entries and flattening."
        )
        self.heartbeat_message = message
        log.critical(message)
        self._persist_session_guard()
        telemetry = {
            "event": "daily_drawdown_halt",
            "drawdown_pct": drawdown_pct,
            "equity": equity,
            "threshold": limit,
            "timestamp": now.isoformat(),
        }
        write_heartbeat("active", message[:240], self.symbols, telemetry)
        if self.risk_gate_restricted:
            log.warning(
                "Read-only risk configuration fallback; kill-switch broker actions suppressed."
            )
            return True
        self._authorize_order_mutation(
            "cancel_orders", symbols=list(self.symbols), purpose="daily_drawdown_kill",
        )
        self.client.cancel_orders(self.symbols)
        for symbol in positions:
            self._authorize_order_mutation(
                "close_position", symbol=symbol, purpose="daily_drawdown_kill",
            )
            self.client.close_position(symbol)
            log.warning("Kill switch submitted close for %s", symbol)
        return True

    def _flatten_positions(self, positions: dict[str, dict[str, Any]], session_day: date) -> None:
        if self.flattened_day == session_day:
            return
        if self.risk_gate_restricted:
            log.warning(
                "Read-only risk configuration fallback; closing-bell flatten suppressed for %s.",
                session_day,
            )
            return
        for symbol in positions:
            self._authorize_order_mutation(
                "cancel_orders", symbols=[symbol], purpose="closing_bell_flatten",
            )
            self.client.cancel_orders((symbol,))
            self._authorize_order_mutation(
                "close_position", symbol=symbol, purpose="closing_bell_flatten",
            )
            result = self.client.close_position(symbol)
            log.warning("Closing-bell flatten submitted for %s order=%s",
                        symbol, result.get("id", "unknown"))
        self.flattened_day = session_day
        log.info("Closing-bell protection engaged for %s; no overnight exposure permitted.",
                 session_day)

    def _stop_leg_id(self, order: dict[str, Any]) -> str | None:
        for leg in order.get("legs", []):
            if leg.get("type") == "stop" or leg.get("stop_price") is not None:
                return leg.get("id")
        return None

    def _manage_stops(self, positions: dict[str, dict[str, Any]]) -> None:
        if self.risk_gate_restricted:
            return
        for symbol, position in positions.items():
            managed = self.managed.get(symbol)
            state = self.states[symbol]
            if managed is None or state.last_price is None:
                continue
            managed.entry = float(position["avg_entry_price"])
            managed.qty = abs(int(float(position["qty"])))
            bars_5m = list(state.agg.bars["5m"])[-60:]
            atr_value = atr(bars_5m, 14)
            if not atr_value:
                continue
            if (
                managed.partial_qty > 0
                and managed.entry_order_ids
                and len(managed.entry_order_ids) == managed.entry_tranches_expected
                and managed.qty <= managed.runner_qty
            ):
                order_records = self.client.orders(symbol)
                filled_entries = {
                    order.get("id") for order in order_records
                    if order.get("status") == "filled"
                }
                if set(managed.entry_order_ids).issubset(filled_entries):
                    managed.partial_taken = True
            long_ = managed.side is Side.LONG
            managed.extreme = max(managed.extreme, state.last_price) if long_ else min(
                managed.extreme, state.last_price
            )
            new_stop = self.risk.manage_stop(managed.side, managed.entry, managed.initial_stop,
                                             managed.stop, state.last_price, managed.extreme, atr_value)
            if managed.partial_taken:
                if long_:
                    new_stop = max(
                        new_stop, managed.entry,
                        managed.extreme - self.risk.l.chandelier_atr_mult * atr_value,
                    )
                else:
                    new_stop = min(
                        new_stop, managed.entry,
                        managed.extreme + self.risk.l.chandelier_atr_mult * atr_value,
                    )
            if abs(new_stop - managed.stop) < 0.01:
                continue
            if (long_ and new_stop >= state.last_price) or (not long_ and new_stop <= state.last_price):
                continue
            open_orders = self.client.open_orders((symbol,))
            active_stop_ids = {
                leg_id for order in open_orders
                if (leg_id := self._stop_leg_id(order)) is not None
            }
            if not managed.stop_order_ids and active_stop_ids:
                managed.stop_order_ids = sorted(active_stop_ids)
                managed.stop_order_id = managed.stop_order_ids[0]
            stop_order_ids = [
                order_id for order_id in (managed.stop_order_ids or [managed.stop_order_id])
                if order_id and order_id in active_stop_ids
            ]
            if not stop_order_ids:
                log.error("Cannot trail %s: bracket stop order ID unavailable.", symbol)
                continue
            self._authorize_order_mutation(
                "update_stop",
                symbol=symbol,
                side="long" if long_ else "short",
                current_stop=managed.stop,
                new_stop=round(new_stop, 2),
                current_price=state.last_price,
                order_ids=stop_order_ids,
            )
            for stop_order_id in stop_order_ids:
                self.client.update_order(stop_order_id, {"stop_price": round(new_stop, 2)})
            log.info("Trailed %s stop %.2f -> %.2f across %d bracket tranche(s).",
                     symbol, managed.stop, new_stop, len(stop_order_ids))
            managed.stop = new_stop
            if managed.partial_taken:
                log.info("MTF EXIT %s first 1.5R tranche confirmed; remaining stop protected at breakeven/Chandelier.",
                         symbol)

    @staticmethod
    def _phase(local_time: datetime_time) -> str:
        if local_time < OPENING_PHASE_END:
            return "opening"
        if MIDDAY_PHASE_START <= local_time < MIDDAY_PHASE_END:
            return "midday_lull"
        if local_time >= POWER_HOUR_START:
            return "power_hour"
        return "regular"

    def _submit_signal(self, symbol: str, bar: dict[str, Any], equity: float,
                       positions: dict[str, dict[str, Any]], orders: dict[str, dict[str, Any]],
                       bar_time: datetime | None = None,
                       entry_blocker: str | None = None) -> None:
        if symbol in positions or symbol in orders or len(positions) + len(orders) >= self.settings.risk.max_open_positions:
            return
        quote = self.last_quotes.get(symbol)
        decision = self.alpha.evaluate(self.states[symbol], equity)
        ts = bar_time or _bar_time(bar["t"])
        local_time = ts.astimezone(NY).time()
        phase = self._phase(local_time)
        state = self.states[symbol]
        m1 = list(state.agg.bars["1m"])
        m5 = list(state.agg.bars["5m"])
        current_bar = m1[-1] if m1 else None
        rvol = _relative_volume(state, current_bar) if current_bar is not None else None
        feed_rejection = ""
        if current_bar is None:
            feed_rejection = "data quality: no current minute candle"
        elif (
            not all(math.isfinite(value) for value in (
                current_bar.open, current_bar.high, current_bar.low,
                current_bar.close, current_bar.volume,
            ))
            or min(current_bar.open, current_bar.high, current_bar.low, current_bar.close) <= 0
            or current_bar.volume <= 0
            or current_bar.high < max(current_bar.open, current_bar.close)
            or current_bar.low > min(current_bar.open, current_bar.close)
        ):
            feed_rejection = "data quality: malformed or zero-volume current candle"
        elif rvol is not None and rvol < 0.1:
            feed_rejection = f"data quality: current volume too thin (RVOL {rvol:.2f}x)"
        elif len(m1) >= 5 and all(
            item.open == item.high == item.low == item.close
            for item in m1[-5:]
        ):
            feed_rejection = "data quality: five consecutive flat-line candles"
        elif len(m1) >= 2 and m1[-1].start.date() == m1[-2].start.date() and (
            m1[-1].start - m1[-2].start > timedelta(minutes=2)
        ):
            feed_rejection = "data quality: missing recent intraday candle interval"
        atr_value = atr(m5[:-1], 14) if len(m5) > 1 else None
        adx_value = _adx(m5[:-1]) if len(m5) > 1 else None
        completed_5m = [
            item for item in m5
            if item.start + timedelta(minutes=5) <= ts
        ]
        completed_15m = [
            item for item in _resample_bars(m1, 15)
            if item.start + timedelta(minutes=15) <= ts
        ]
        mtf = None
        if decision.take and decision.side is not None:
            mtf = self.mtf_filter.assess(
                float(bar.get("c", state.last_price or 0)),
                decision.side,
                self.hourly_bars.get(symbol, []),
                self.daily_bars.get(symbol, []),
                completed_15m,
                completed_5m,
            )
        rr = None
        if decision.stop is not None and decision.target is not None and decision.entry is not None:
            risk_distance = abs(decision.entry - decision.stop)
            rr = abs(decision.target - decision.entry) / risk_distance if risk_distance else None

        rejection = feed_rejection or (decision.reject_reason if not decision.take else "")
        if decision.take and mtf is not None and not mtf.allowed:
            rejection = f"MTF filter: {mtf.reason}"
        if decision.take and entry_blocker:
            rejection = entry_blocker
        required_score = self.alpha.min_score
        if decision.take and decision.side is not None:
            if phase == "opening":
                required_score = max(required_score, 0.75)
                if atr_value is None or atr_value <= 0:
                    rejection = "opening filter: ATR unavailable for tight-stop sizing"
                orb = _opening_range_breakout(state, ts, decision.side)
                momentum = bool(
                    current_bar and rvol is not None and rvol >= OPENING_MIN_RVOL
                    and ((decision.side is Side.LONG and current_bar.close > current_bar.open)
                         or (decision.side is Side.SHORT and current_bar.close < current_bar.open))
                )
                if not rejection and not orb and not momentum:
                    rejection = (
                        f"opening filter: no ORB breakout and RVOL momentum below "
                        f"{OPENING_MIN_RVOL:.1f}x"
                    )
            elif phase == "midday_lull":
                required_score = max(required_score, 0.85)
                checks = {check.name: check.passed for check in decision.checks}
                if not (checks.get("vwap_retest") or checks.get("fomo_exhaustion")):
                    rejection = "midday filter: no VWAP mean-reversion confirmation"
            elif phase == "power_hour":
                required_score = max(required_score, 0.80)
                checks = {check.name: check.passed for check in decision.checks}
                if not checks.get("trend_d50") or adx_value is None or adx_value < 20:
                    rejection = "power-hour filter: trend continuation confirmation unavailable"
            if not rejection and decision.score < required_score:
                rejection = f"{phase} score {decision.score:.2f} < required {required_score:.2f}"
        if not rejection and quote is None:
            rejection = "no fresh quote available for slippage-controlled entry"
        if not rejection and quote is not None and (
            quote.bid <= 0 or quote.ask < quote.bid
        ):
            rejection = "invalid or crossed quote; refusing limit entry"
        if not rejection and (decision.side is None or decision.stop is None or decision.entry is None):
            rejection = "signal is missing side or protective stop"

        if rejection or not decision.take:
            accepted = False
        else:
            accepted = True
        telemetry = {
            "symbol": symbol,
            "phase": phase,
            "current_price": current_bar.close if current_bar else float(bar.get("c", 0)),
            "bar_open": current_bar.open if current_bar else None,
            "bar_change_pct": (
                (current_bar.close - current_bar.open) / current_bar.open * 100
                if current_bar and current_bar.open > 0 else None
            ),
            "session_vwap": state.session_vwap(),
            "rvol": rvol,
            "adx": adx_value,
            "atr": atr_value,
            "score": decision.score,
            "required_score": required_score,
            "reward_risk": rr,
            "protective_stop": decision.stop,
            "profit_target": decision.target,
            "mtf": {
                "hourly": mtf.hourly,
                "daily": mtf.daily,
                "intraday": mtf.intraday,
                "allowed": mtf.allowed,
                "reason": mtf.reason,
            } if mtf else None,
            "side": decision.side.value if decision.side else None,
            "accepted": accepted,
            "reason": rejection or "eligible; preparing paper bracket",
            "timestamp": ts.isoformat(),
        }
        details = (
            f"RVOL {rvol:.2f}x" if rvol is not None else "RVOL n/a",
            f"ADX {adx_value:.1f}" if adx_value is not None else "ADX n/a",
            f"ATR {atr_value:.4f}" if atr_value is not None else "ATR n/a",
            f"score {decision.score:.2f}",
            f"R:R {rr:.2f}:1" if rr is not None else "R:R n/a",
        )
        if not accepted:
            message = (
                f"[INTRADAY SCAN] {symbol} phase={phase} " + ", ".join(details)
                + f" -> Rejected: {rejection or decision.reject_reason or 'signal score below threshold'}; "
                + decision.rationale()
            )
            log.info("%s", message)
            write_heartbeat("active", message[:240], self.symbols, telemetry)
            return

        if not self._order_routing_authorized("stock"):
            telemetry.update({
                "accepted": False,
                "reason": "order routing locked: explicit enablement and promoted research evidence required",
            })
            write_heartbeat(
                "active",
                f"{symbol}: signal qualified, but order routing is locked.",
                self.symbols,
                telemetry,
            )
            return

        side = "buy" if decision.side is Side.LONG else "sell"
        quote_price = quote.ask if decision.side is Side.LONG else quote.bid
        chase_bps = float(os.getenv("MAX_ENTRY_SLIPPAGE_BPS", "5"))
        if not math.isfinite(chase_bps) or not 0 <= chase_bps <= 25:
            raise RuntimeError("MAX_ENTRY_SLIPPAGE_BPS must be between 0 and 25.")
        chase_factor = chase_bps / 10_000
        limit_price = quote_price * (1 + chase_factor if decision.side is Side.LONG else 1 - chase_factor)
        stop_distance = abs(decision.entry - decision.stop)
        if phase == "opening" and atr_value is not None:
            stop_distance = min(stop_distance, OPENING_ATR_STOP_MULT * atr_value)
        entry_reference = round(limit_price, 2)
        stop_price = round(
            entry_reference - stop_distance if decision.side is Side.LONG
            else entry_reference + stop_distance, 2
        )
        stop_distance = abs(entry_reference - stop_price)
        qty = self.risk.size(equity, entry_reference, stop_price)
        if qty < 1:
            telemetry.update({"accepted": False, "reason": "risk-sized quantity is below one share"})
            log.info("[INTRADAY SCAN] %s phase=%s %s -> Rejected: risk-sized quantity is below one share.",
                     symbol, phase, ", ".join(details))
            write_heartbeat("active", f"{symbol}: risk-sized quantity is below one share.",
                            self.symbols, telemetry)
            return
        stop_ids = []
        entry_ids = []
        partial_qty = qty // 2 if qty > 1 else 0
        runner_qty = qty - partial_qty
        if qty == 1:
            tranches = [(qty, 1.5)]
        else:
            first_qty = qty // 2
            tranches = [
                (first_qty, 1.5),
                (qty - first_qty, max(1.5, self.settings.risk.min_reward_risk)),
            ]
        epoch = int(ts.timestamp())
        order_plans = []
        risk_orders = []
        for index, (tranche_qty, target_r) in enumerate(tranches, start=1):
            target_price = round(
                entry_reference + target_r * stop_distance if decision.side is Side.LONG
                else entry_reference - target_r * stop_distance, 2
            )
            client_id = f"apx-{symbol.lower()}-{epoch}-{side}-{index}"
            payload = {
                "symbol": symbol,
                "qty": str(tranche_qty),
                "side": side,
                "type": "limit",
                "limit_price": round(limit_price, 2),
                "time_in_force": "day",
                "order_class": "bracket",
                "take_profit": {"limit_price": target_price},
                "stop_loss": {"stop_price": stop_price},
                "client_order_id": client_id,
            }
            order_plans.append((tranche_qty, target_r, client_id, payload))
            risk_orders.append({
                "symbol": symbol,
                "qty": tranche_qty,
                "side": side,
                "limit_price": round(limit_price, 2),
                "stop_price": stop_price,
                "take_profit_price": target_price,
                "client_order_id": client_id,
            })
        try:
            risk_decision = self.risk_gate.authorize({
                "request_id": str(uuid.uuid4()),
                "operation": "submit_batch",
                "mode": self.settings.mode.value,
                "account_equity": equity,
                "open_symbols": sorted(set(positions) | set(orders)),
                "orders": risk_orders,
                "phase": phase,
                "signal_score": decision.score,
            })
        except RiskGateUnavailable as exc:
            self._halt_for_risk_gateway(str(exc))
            raise
        if risk_decision.get("approved") is not True:
            reason = str(risk_decision.get("reason", "independent risk engine rejected the order"))
            telemetry.update({"accepted": False, "reason": reason})
            log.warning("Independent risk engine rejected %s order batch: %s", symbol, reason)
            write_heartbeat(
                "active", f"{symbol}: independent risk engine rejected the order batch.",
                self.symbols, telemetry,
            )
            return
        self.managed[symbol] = ManagedPosition(
            symbol=symbol,
            side=decision.side,
            qty=qty,
            entry=entry_reference,
            initial_stop=stop_price,
            stop=stop_price,
            extreme=entry_reference,
            stop_order_ids=stop_ids,
            initial_qty=qty,
            partial_qty=partial_qty,
            runner_qty=runner_qty,
            entry_order_ids=entry_ids,
            entry_tranches_expected=len(tranches),
        )
        for tranche_qty, target_r, client_id, payload in order_plans:
            order = self.broker_adapter.submit_protected_order("stock", payload)
            orders[symbol] = order
            if order.get("id"):
                entry_ids.append(order["id"])
            stop_id = self._stop_leg_id(order)
            if stop_id:
                stop_ids.append(stop_id)
                if self.managed[symbol].stop_order_id is None:
                    self.managed[symbol].stop_order_id = stop_id
            log.info(
                "[INTRADAY SCAN] %s RVOL %s, ADX %s, ATR %s -> %s Setup Triggered. "
                "Score %.2f, R:R %.2f:1 -> Executing Paper %s qty=%d limit=$%.2f "
                "target=%.1fR order=%s",
                symbol,
                f"{rvol:.2f}x" if rvol is not None else "n/a",
                f"{adx_value:.1f}" if adx_value is not None else "n/a",
                f"{atr_value:.4f}" if atr_value is not None else "n/a",
                phase.upper(), decision.score, rr or 0.0, side.upper(), tranche_qty,
                limit_price, target_r, order.get("id", client_id),
            )
        telemetry.update({
            "accepted": True,
            "reason": "paper bracket limit order submitted",
            "qty": qty,
            "limit_price": round(limit_price, 2),
            "stop": stop_price,
            "targets_r": [target for _, target in tranches],
        })
        write_heartbeat(
            "active", f"{symbol}: paper {side} limit brackets submitted.", self.symbols, telemetry
        )

    def _process_bars(self, bars: dict[str, list[dict[str, Any]]], now: datetime,
                      account: dict[str, Any], positions: dict[str, dict[str, Any]],
                      orders: dict[str, dict[str, Any]],
                      entry_blocker: str | None = None) -> None:
        pending = []
        for symbol, values in bars.items():
            watermark = self.watermarks.get(symbol)
            for bar in values:
                ts = _bar_time(bar["t"])
                if (watermark is None or ts > watermark) and ts + BAR_INTERVAL <= now:
                    pending.append((ts, symbol, bar))
        pending.sort(key=lambda item: (item[0], item[1]))
        for ts, symbol, bar in pending:
            _feed_bar(self.states[symbol], bar)
            self.watermarks[symbol] = ts
            local = ts.astimezone(NY)
            if local.time() >= FLATTEN_AT and self.flattened_day != local.date():
                self._flatten_positions(positions, local.date())
                return
            is_fresh = now - (ts + BAR_INTERVAL) <= timedelta(minutes=2)
            if (is_fresh and ENTRY_START <= local.time() <= ENTRY_END
                    and not self.halted and self.flattened_day != local.date()):
                self._submit_signal(
                    symbol, bar, float(account["equity"]), positions, orders,
                    bar_time=ts + BAR_INTERVAL,
                    entry_blocker=entry_blocker,
                )

    def run_once(self) -> float:
        if self.risk_gateway_halted or not self._check_risk_gateway():
            return POLL_SECONDS
        clock = self.client.clock()
        now = datetime.now(timezone.utc)
        if self.crypto_symbols:
            try:
                self._run_crypto_research(now)
            except (urllib.error.URLError, TimeoutError, AlpacaAPIError, ValueError) as exc:
                self._log_crypto_research_failure(exc)
        if not clock.get("is_open"):
            self.heartbeat_status = "restricted" if self.risk_gate_restricted else "waiting"
            crypto_state = (
                f" Crypto research is active for {', '.join(self.crypto_symbols)}; order routing disabled."
                if self.crypto_symbols else ""
            )
            risk_state = (
                " Risk configuration fallback: read-only paper-safe mode."
                if self.risk_gate_restricted else ""
            )
            self.heartbeat_message = (
                f"Equity market closed; next open {clock.get('next_open', 'unknown')}."
                + crypto_state + risk_state
            )
            write_heartbeat(
                self.heartbeat_status, self.heartbeat_message, self.symbols,
                crypto_symbols=self.crypto_symbols,
            )
            self._publish_dashboard_snapshot_while_closed()
            try:
                self._publish_dashboard_market_snapshot_while_closed(now)
            except Exception as exc:
                self._log_dashboard_market_failure(exc)
            if self._cleanup_identified_test_positions(market_open=False):
                return 60.0
            local_now = now.astimezone(NY)
            if (
                self.session_day == local_now.date()
                and local_now.time() >= FLATTEN_AT
                and self.flattened_day != local_now.date()
            ):
                positions = {
                    position["symbol"]: position for position in self.client.positions()
                    if position.get("symbol") in self.symbols
                }
                self._flatten_positions(positions, local_now.date())
                return 60.0
            next_open = _bar_time(clock["next_open"])
            scan_at = next_open - timedelta(minutes=15)
            next_day = next_open.astimezone(NY).date()
            if self.prepared_day != next_day and now >= scan_at:
                self._prepare_session(now, next_open)
            if self.prepared_day != next_day:
                seconds = max((scan_at - now).total_seconds(), 1)
                log.info("Market closed; premarket scan scheduled for %s.", scan_at.isoformat())
                return min(seconds, 60.0 if self.crypto_symbols else 300.0)
            log.info("Market closed; next open=%s.", clock.get("next_open"))
            return 60.0
        self.heartbeat_status = "restricted" if self.risk_gate_restricted else "active"
        self.heartbeat_message = (
            "Market open; runner cycle connected in read-only paper-safe mode."
            if self.risk_gate_restricted
            else "Market open; runner cycle connected."
        )
        write_heartbeat(self.heartbeat_status, self.heartbeat_message, self.symbols,
                        crypto_symbols=self.crypto_symbols,
                        risk_gate=self._risk_gate_telemetry())
        if self._cleanup_identified_test_positions(market_open=True):
            return POLL_SECONDS
        session_open = _bar_time(clock.get("timestamp", now.isoformat()))
        session_day = session_open.astimezone(NY).date()
        if self.prepared_day != session_day:
            self._prepare_session(now, session_open)
        if not self.initialized:
            self._initialize_market_state(now)
        account, positions, orders = self._account_and_positions()
        self._start_session(now, account)
        if self._kill_switch(now, positions):
            return POLL_SECONDS
        self._refresh_mtf_context(now)
        macro_guard = self._macro_entry_guard(now)
        if macro_guard.blocked:
            message = f"MACRO CIRCUIT BREAKER: {macro_guard.reason}; new entries paused."
            write_heartbeat("active", message, self.symbols, {
                "event": "macro_circuit_breaker",
                "reason": macro_guard.reason,
                "event_name": macro_guard.event,
                "until": macro_guard.until,
                "timestamp": now.isoformat(),
            }, crypto_symbols=self.crypto_symbols)
        local_now = now.astimezone(NY)
        if local_now.time() >= FLATTEN_AT:
            self._flatten_positions(positions, local_now.date())
            self.heartbeat_message = (
                f"Closing-bell protection active; session positions flattened for {local_now.date()}."
            )
            write_heartbeat("active", self.heartbeat_message, self.symbols)
            return POLL_SECONDS
        try:
            self._refresh_quotes(now)
            self._manage_stops(positions)
            start = min(self.watermarks.values()) + BAR_INTERVAL
            bars = self.client.bars(self.symbols, "1Min", start, now, limit=10000)
            self._process_bars(
                bars, now, account, positions, orders,
                entry_blocker=macro_guard.reason if macro_guard.blocked else None,
            )
        except Exception:
            self.heartbeat_status = "restricted" if self.risk_gate_restricted else "active"
            self.heartbeat_message = (
                "Strategy cycle failed; telemetry, heartbeat, and account snapshot remain active."
            )
            write_heartbeat(
                self.heartbeat_status, self.heartbeat_message, self.symbols,
                risk_gate=self._risk_gate_telemetry(),
                crypto_symbols=self.crypto_symbols,
            )
            log.exception("Strategy cycle failed; telemetry continues and the runner will retry.")
        self._publish_dashboard_market_snapshot()
        return POLL_SECONDS

    async def run_forever(self, stop_event: threading.Event | None = None) -> None:
        backoff = 1.0
        heartbeat_stop = threading.Event()

        def refresh_heartbeat():
            while not heartbeat_stop.is_set():
                try:
                    write_heartbeat(self.heartbeat_status, self.heartbeat_message, self.symbols)
                except OSError:
                    log.exception("Unable to refresh runner heartbeat.")
                if heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                    return

        heartbeat_thread = threading.Thread(target=refresh_heartbeat, name="runner-heartbeat", daemon=True)
        heartbeat_thread.start()
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    delay = self.run_once()
                    backoff = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if _is_transient_runner_error(exc):
                        self.heartbeat_status = "waiting"
                        self.heartbeat_message = "Alpaca poll delayed; retrying next cycle."
                        try:
                            write_heartbeat(
                                self.heartbeat_status, self.heartbeat_message, self.symbols,
                            )
                        except OSError:
                            log.exception("Unable to persist delayed-poll heartbeat.")
                        log.warning(
                            "Alpaca poll delayed; retrying next cycle in %.1fs: %s",
                            backoff, exc,
                        )
                        delay = backoff * (1 + random.random() * 0.2)
                        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    else:
                        self.heartbeat_status = "disconnected"
                        self.heartbeat_message = "Runner cycle failed; reconnecting."
                        try:
                            write_heartbeat(
                                self.heartbeat_status, self.heartbeat_message, self.symbols,
                            )
                        except OSError:
                            log.exception("Unable to persist disconnected heartbeat.")
                        log.exception("Runner cycle failed; reconnecting in %.1fs.", backoff)
                        delay = backoff * (1 + random.random() * 0.2)
                        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                if stop_event is None:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.to_thread(stop_event.wait, delay)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)


def dry_run(client: AlpacaClient, settings: Settings) -> int:
    try:
        RiskGateClient().health()
    except RiskGateUnavailable as exc:
        print(f"Risk engine readiness: BLOCKED ({exc})")
        return 2
    print("Independent risk engine heartbeat: OK")
    clock = client.clock()
    account = client.account()
    if account.get("trading_blocked") or account.get("account_blocked"):
        raise RuntimeError("Alpaca reports a blocked paper account.")
    print("Alpaca paper API connection: OK")
    print(f"Account status: {account.get('status')}")
    print(f"Equity: ${float(account['equity']):,.2f} {account.get('currency', 'USD')}")
    print(f"Market open: {clock.get('is_open')}")
    print(f"Next open: {clock.get('next_open')}")
    print(f"Next close: {clock.get('next_close')}")
    market_open = bool(clock.get("is_open"))
    candidates = []
    if market_open:
        runner = AutonomousRunner(settings, client)
        candidates = runner.selector.select(datetime.now(timezone.utc))
        symbols = tuple(candidate.symbol for candidate in candidates)
        print(f"Dynamic universe ({len(candidates)}): " + ", ".join(symbols))
    else:
        symbols = ()
        print("Market is closed; dynamic liquidity selection is deferred until premarket.")
    scope = tuple(dict.fromkeys(settings.watchlist + symbols))
    existing_positions = {p["symbol"]: p for p in client.positions() if p.get("symbol") in scope}
    existing_orders = {
        order["symbol"] for order in client.open_orders(scope) if order.get("symbol") in scope
    }
    print("Daily optimizer configured; dry-run does not download optimization history or submit trades.")
    if existing_positions or existing_orders:
        exposure = [
            f"{symbol} position qty={position.get('qty')}"
            for symbol, position in sorted(existing_positions.items())
        ]
        exposure.extend(f"{symbol} has an open order" for symbol in sorted(existing_orders))
        print("Startup blocked until target exposure is resolved: " + "; ".join(exposure))
        log.warning("Dry-run found existing target exposure: %s", "; ".join(exposure))
        print("Dry run complete; no orders were submitted. Readiness: BLOCKED.")
        return 2
    else:
        print("Target-symbol position/order preflight: clear")
        log.info("Dry-run connectivity check succeeded; no orders submitted.")
    if not market_open:
        print("Dry run complete; paper API is connected. Trading readiness is pending market-open universe selection.")
        return 2
    print("Dry run complete; no orders were submitted.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paper-only ApexQuant Alpaca trader.")
    parser.add_argument("--dry-run", action="store_true", help="check paper account and market clock; submit no orders")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        settings = load_paper_settings()
        client = AlpacaClient(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_data_feed)
        if args.dry_run:
            return dry_run(client, settings)
        log.info("Starting paper-only runner for %s; clock follows US exchange hours/DST.",
                 ",".join(settings.watchlist))
        with single_instance_lock():
            runner = AutonomousRunner(settings, client)
            write_heartbeat("waiting", "Runner started; checking Alpaca market clock.", settings.watchlist,
                            crypto_symbols=runner.crypto_symbols)
            asyncio.run(runner.run_forever())
    except KeyboardInterrupt:
        log.info("Runner stopped by user.")
        return 0
    except Exception as exc:
        log.exception("Runner startup failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
