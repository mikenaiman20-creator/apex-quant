"""ApexQuant configuration. Fails closed: PAPER unless LIVE is explicitly double-confirmed."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(root: Path | None = None) -> bool:
    """Load project configuration from its absolute path, independent of the current directory."""
    env_path = (Path(root) if root is not None else PROJECT_ROOT).resolve() / ".env"
    return load_dotenv(env_path, override=False)


load_project_env()

LIVE_CONFIRM_PHRASE = "I_ACCEPT_REAL_LOSSES"


class TradingMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class BrokerName(str, Enum):
    ALPACA = "ALPACA"
    IBKR = "IBKR"


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _list(name: str, default: str) -> tuple[str, ...]:
    return tuple(s.strip().upper() for s in os.getenv(name, default).split(",") if s.strip())


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_pct: float = 0.005       # equity risked per trade (0.5%)
    max_position_notional_pct: float = 0.20  # cap on single-position notional
    max_open_positions: int = 3
    daily_drawdown_kill_pct: float = 0.03    # liquidate + halt at -3% on the day
    min_reward_risk: float = 2.0
    atr_stop_mult: float = 1.5
    breakeven_at_r: float = 1.0
    chandelier_atr_mult: float = 3.0


@dataclass(frozen=True)
class Settings:
    mode: TradingMode
    broker: BrokerName
    alpaca_key: str
    alpaca_secret: str
    alpaca_data_feed: str          # "iex" (free, thin volume) or "sip" (paid, consolidated)
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    watchlist: tuple[str, ...]
    risk: RiskLimits
    min_signal_score: float
    webhook_secret: str
    webhook_port: int
    log_dir: str
    db_path: str
    market_tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("America/New_York"))
    display_tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("Asia/Jerusalem"))

    @property
    def is_live(self) -> bool:
        return self.mode is TradingMode.LIVE


def load_settings(require_credentials: bool = True) -> Settings:
    load_project_env()
    mode = TradingMode(os.getenv("TRADING_MODE", "PAPER").upper())
    if mode is TradingMode.LIVE and os.getenv("LIVE_CONFIRM") != LIVE_CONFIRM_PHRASE:
        raise RuntimeError(
            f"TRADING_MODE=LIVE requires LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}. Refusing to start."
        )

    broker = BrokerName(os.getenv("BROKER", "ALPACA").upper())
    s = Settings(
        mode=mode,
        broker=broker,
        alpaca_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret=os.getenv("ALPACA_API_SECRET", ""),
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex").lower(),
        ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1"),
        ibkr_port=_int("IBKR_PORT", 7497 if mode is TradingMode.PAPER else 7496),
        ibkr_client_id=_int("IBKR_CLIENT_ID", 17),
        # Equities only: BTC trades 24/7 and conflicts with the market-hours scheduler.
        watchlist=_list("WATCHLIST", "SPY,QQQ,NVDA"),
        risk=RiskLimits(
            risk_per_trade_pct=_float("RISK_PER_TRADE_PCT", 0.005),
            max_position_notional_pct=_float("MAX_POSITION_NOTIONAL_PCT", 0.20),
            max_open_positions=_int("MAX_OPEN_POSITIONS", 3),
            daily_drawdown_kill_pct=_float("DAILY_DRAWDOWN_KILL_PCT", 0.03),
            min_reward_risk=_float("MIN_REWARD_RISK", 2.0),
            atr_stop_mult=_float("ATR_STOP_MULT", 1.5),
            breakeven_at_r=_float("BREAKEVEN_AT_R", 1.0),
            chandelier_atr_mult=_float("CHANDELIER_ATR_MULT", 3.0),
        ),
        min_signal_score=_float("MIN_SIGNAL_SCORE", 0.70),
        webhook_secret=os.getenv("WEBHOOK_SECRET", ""),
        webhook_port=_int("WEBHOOK_PORT", 8000),
        log_dir=os.getenv("LOG_DIR", "logs"),
        db_path=os.getenv("DB_PATH", "logs/apex.sqlite"),
    )

    if require_credentials:
        if broker is BrokerName.ALPACA and not (s.alpaca_key and s.alpaca_secret):
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing.")
        if not s.webhook_secret:
            raise RuntimeError("WEBHOOK_SECRET missing (webhook must never be unauthenticated).")
    return s
