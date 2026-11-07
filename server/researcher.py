"""Advisory regime classification and strategy research; never submits orders."""
from __future__ import annotations

import json
import hashlib
import itertools
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from engine.data_pipeline import Bar
from engine.strategy import average_directional_index, simple_moving_average

log = logging.getLogger("apex.research")
ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
DISCOVERY_GRID = {
    "min_signal_score": (0.65, 0.75),
    "atr_stop_mult": (1.25, 1.75),
}
DISCOVERY_INTERVAL_SECONDS = 900
DISCOVERY_MAX_SYMBOLS = 3

STRATEGY_CATALOG = (
    {"name": "Opening Range Breakout", "family": "ORB", "asset_class": "equities",
     "frequency": "day", "regimes": ["high_volatility_trending"], "execution_enabled": False},
    {"name": "VWAP Mean Reversion", "family": "VWAP", "asset_class": "equities",
     "frequency": "day", "regimes": ["mean_reverting", "low_volatility_choppy"], "execution_enabled": False},
    {"name": "RVOL Momentum", "family": "RVOL", "asset_class": "equities",
     "frequency": "day", "regimes": ["high_volatility_trending"], "execution_enabled": False},
    {"name": "ATR Volatility Swing", "family": "ATR", "asset_class": "equities",
     "frequency": "swing", "regimes": ["high_volatility_trending"], "execution_enabled": False},
    {"name": "Multi-Timeframe Trend", "family": "MTF", "asset_class": "equities",
     "frequency": "swing", "regimes": ["trending"], "execution_enabled": False},
    {"name": "Crypto VWAP Research", "family": "VWAP", "asset_class": "crypto",
     "frequency": "day", "regimes": ["mean_reverting"], "execution_enabled": False},
    {"name": "Forex Trend Research", "family": "MTF", "asset_class": "forex",
     "frequency": "swing", "regimes": ["trending"], "execution_enabled": False},
)


@dataclass(frozen=True)
class Regime:
    symbol: str
    label: str
    adx: float
    atr: float
    atr_pct: float
    relative_atr: float
    ma_slope_pct: float
    as_of: str


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
    adx = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        adx = ((adx * (period - 1)) + dx) / period
    return adx


def _bar_quality_issue(bars, timeframe: str) -> str | None:
    for bar in bars:
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if (
            not all(math.isfinite(value) for value in values)
            or min(bar.open, bar.high, bar.low, bar.close) <= 0
            or bar.volume <= 0
            or bar.high < max(bar.open, bar.close)
            or bar.low > min(bar.open, bar.close)
        ):
            return "malformed or zero-volume candle"
    recent_volumes = [bar.volume for bar in bars[-20:]]
    median_volume = sorted(recent_volumes)[len(recent_volumes) // 2] if recent_volumes else 0
    if bars and median_volume > 0 and bars[-1].volume < median_volume * 0.1:
        return "latest candle volume is below 10% of recent median"
    if len(bars) >= 5 and all(
        bar.open == bar.high == bar.low == bar.close for bar in bars[-5:]
    ):
        return "five consecutive flat-line candles"
    for previous, current in zip(bars, bars[1:]):
        gap = current.start - previous.start
        if timeframe == "1d" and gap.total_seconds() > timedelta(days=4).total_seconds():
            return "missing daily candle interval"
        if (timeframe == "1h"
                and current.start.astimezone(NY).date() == previous.start.astimezone(NY).date()
                and gap.total_seconds() > timedelta(hours=2).total_seconds()):
            return "missing hourly candle interval"
    return None


def classify_regime(symbol: str, bars, as_of: datetime | None = None) -> Regime | None:
    if len(bars) < 40:
        return None
    quality_issue = _bar_quality_issue(bars, "1d")
    if quality_issue:
        log.warning("REGIME DATA QUALITY %s: %s; skipping classification.", symbol, quality_issue)
        return None
    adx_value = _adx(bars)
    if adx_value is None:
        return None
    recent = bars[-14:]
    ranges = [max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))
              for prev, bar in zip(bars[-15:-1], recent)]
    atr_value = sum(ranges) / len(ranges)
    close = bars[-1].close
    if close <= 0 or atr_value <= 0:
        return None
    atr_pct = atr_value / close
    historical_atrs = []
    for index in range(max(14, len(bars) - 60), len(bars) - 1):
        windows = bars[index - 13:index + 1]
        if len(windows) == 14:
            prior_bars = bars[index - 14:index]
            trs = [
                max(item.high - item.low, abs(item.high - previous.close),
                    abs(item.low - previous.close))
                for previous, item in zip(prior_bars, windows)
            ]
            historical_atrs.append(sum(trs) / len(trs))
    median_atr = sorted(historical_atrs)[len(historical_atrs) // 2] if historical_atrs else atr_value
    relative_atr = atr_value / median_atr if median_atr > 0 else 1.0
    closes = [bar.close for bar in bars]
    fast = sum(closes[-10:]) / 10
    slow = sum(closes[-30:-20]) / 10
    slope = (fast - slow) / slow if slow else 0.0
    trending = adx_value >= 25 and abs(slope) >= 0.001
    high_vol = relative_atr >= 1.25
    if trending and high_vol:
        label = "high_volatility_trending"
    elif trending:
        label = "trending"
    elif adx_value < 18 and relative_atr < 0.8:
        label = "low_volatility_choppy"
    else:
        label = "mean_reverting" if adx_value < 22 else "high_volatility_choppy"
    timestamp = as_of or datetime.now(timezone.utc)
    return Regime(symbol, label, adx_value, atr_value, atr_pct, relative_atr, slope,
                  timestamp.astimezone(timezone.utc).isoformat())


class StrategyResearcher:
    def __init__(self, client, symbols_provider, root: Path = ROOT,
                 interval_seconds: int = DISCOVERY_INTERVAL_SECONDS):
        self.client = client
        self.symbols_provider = symbols_provider
        self.root = root
        self.interval_seconds = interval_seconds
        self._last_discovery_fingerprint: str | None = None
        self._context_error: str | None = None

    @property
    def _agent_state_path(self) -> Path:
        return self.root / "logs" / "research_agent_state.json"

    @staticmethod
    def _json_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _write_agent_state(self, state: dict[str, Any]) -> None:
        path = self._agent_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
        temp_path.replace(path)

    def _discovery_files(self) -> list[tuple[str, Path]]:
        data_dir = self.root / "data"
        symbols = [
            symbol for symbol in self.symbols_provider()
            if isinstance(symbol, str) and symbol.isalpha()
        ]
        files = [
            (symbol.upper(), data_dir / f"{symbol.upper()}.csv")
            for symbol in dict.fromkeys(symbols)
            if (data_dir / f"{symbol.upper()}.csv").is_file()
        ]
        return files[:DISCOVERY_MAX_SYMBOLS]

    @staticmethod
    def _dataset_fingerprint(files: list[tuple[str, Path]]) -> str:
        digest = hashlib.sha256()
        for symbol, path in files:
            stat = path.stat()
            digest.update(f"{symbol}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        digest.update(json.dumps(DISCOVERY_GRID, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def discovery_once(self, force: bool = False) -> dict[str, Any]:
        """Run bounded, cost-adjusted WFO on local equity data, independently of market hours."""
        from backtest.data import load_dir
        from backtest.engine import BTConfig
        from backtest.holdout import validate_final_holdout
        from backtest.metrics import summarize, trade_equity_curve
        from backtest.walkforward import walk_forward
        from server.evolution import (
            load_trial_ledger, register_trials, stage_candidate,
            update_audit_research_activity, update_trial_activity,
        )

        now = datetime.now(timezone.utc)
        trial_ledger_path = self.root / "logs" / "strategy_trial_ledger.json"
        trial_ledger = load_trial_ledger(trial_ledger_path)
        files = self._discovery_files()
        cycle_id = ""
        grid_size = math.prod(len(values) for values in DISCOVERY_GRID.values())
        audit_path = self.root / "logs" / "strategy_audit_report.json"
        if not files:
            state = {
                "status": "waiting_for_data",
                "updated_at": now.isoformat(),
                "current_task": "Looking for local equity history",
                "message": (
                    "The researcher is running continuously, Sir, but no configured equity "
                    "symbols have local CSV history in data/ yet."
                ),
                "symbols": [],
                "variants_tested": 0,
                "cumulative_trials": trial_ledger["total_trials"],
                "variants": [],
                "candidate": None,
                "context_error": self._context_error,
                "execution_enabled": False,
            }
            self._write_agent_state(state)
            update_audit_research_activity(
                "", "waiting_for_data", state["message"], path=audit_path,
            )
            update_trial_activity(
                "", 0, "waiting_for_data", state["message"],
                trial_ledger_path,
            )
            return state

        symbols = [symbol for symbol, _ in files]
        fingerprint = self._dataset_fingerprint(files)
        cycle_id = f"research-{fingerprint}"
        if self._last_discovery_fingerprint is None:
            previous = self._read_agent_state()
            if isinstance(previous, dict):
                self._last_discovery_fingerprint = previous.get("dataset_fingerprint")
        if not force and fingerprint == self._last_discovery_fingerprint:
            existing = self._read_agent_state()
            if existing:
                state = {
                    **existing,
                    "status": "idle_waiting_for_new_data",
                    "updated_at": now.isoformat(),
                    "current_task": "Monitoring local history for new bars",
                    "message": (
                        f"Research is online, Sir. The current {len(symbols)}-symbol dataset "
                        f"has already been evaluated across {existing.get('variants_tested', 0)} "
                        "parameter variants; the agent is waiting for new data to avoid "
                        "counting repeated tests as new evidence."
                    ),
                }
                self._write_agent_state(state)
                update_audit_research_activity(
                    cycle_id, "idle_waiting_for_new_data", state["message"],
                    grid_size, int(state.get("variants_tested", 0)), audit_path,
                )
                update_trial_activity(
                    cycle_id, int(state.get("variants_tested", 0)),
                    "idle_waiting_for_new_data", state["message"],
                    trial_ledger_path,
                )
                return state

        try:
            data = load_dir(str(self.root / "data"), symbols)
        except (OSError, ValueError, KeyError) as exc:
            state = {
                "status": "data_error",
                "updated_at": now.isoformat(),
                "current_task": "Validating local historical bars",
                "message": f"Research data validation failed, Sir: {exc}",
                "symbols": symbols,
                "variants_tested": 0,
                "cumulative_trials": 0,
                "variants": [],
                "candidate": None,
                "context_error": self._context_error,
                "execution_enabled": False,
            }
            self._write_agent_state(state)
            update_audit_research_activity(
                cycle_id, "data_error", state["message"], grid_size, 0, audit_path,
            )
            update_trial_activity(
                cycle_id, 0, "data_error", state["message"],
                trial_ledger_path,
            )
            return state

        grid = {name: list(values) for name, values in DISCOVERY_GRID.items()}
        planned_trials = math.prod(len(values) for values in grid.values())
        running_state = {
            "status": "evaluating_variants",
            "updated_at": now.isoformat(),
            "current_task": "Running cost-adjusted walk-forward tests",
            "message": (
                f"The researcher is evaluating {planned_trials} parameter variants across "
                f"{', '.join(symbols)}, Sir. Equity market hours do not gate this offline replay."
            ),
            "symbols": symbols,
            "dataset_fingerprint": fingerprint,
            "variants_tested": 0,
            "planned_variants": planned_trials,
            "cumulative_trials": None,
            "variants": [],
            "candidate": None,
            "context_error": self._context_error,
            "execution_enabled": False,
        }
        self._write_agent_state(running_state)
        update_audit_research_activity(
            cycle_id, "evaluating_variants", running_state["message"],
            planned_trials, 0, audit_path,
        )
        update_trial_activity(
            cycle_id, planned_trials, "evaluating_variants", running_state["message"],
            trial_ledger_path,
        )

        config = BTConfig(
            slippage_bps=float(os.getenv("BACKTEST_SLIPPAGE_BPS", "1.0")),
            slippage_atr_mult=float(os.getenv("BACKTEST_SLIPPAGE_ATR_MULT", "0.1")),
            commission_per_share=float(os.getenv("BACKTEST_COMMISSION_PER_SHARE", "0.0035")),
            commission_min=float(os.getenv("BACKTEST_COMMISSION_MIN", "0.35")),
            sell_fee_bps=float(os.getenv("BACKTEST_SELL_FEE_BPS", "0.3")),
            warmup_days=int(os.getenv("RESEARCH_WARMUP_DAYS", "50")),
        )
        folds, _, results = walk_forward(
            data,
            config,
            grid=grid,
            train_days=int(os.getenv("RESEARCH_TRAIN_DAYS", "30")),
            test_days=int(os.getenv("RESEARCH_TEST_DAYS", "10")),
            min_train_trades=int(os.getenv("RESEARCH_MIN_TRAIN_TRADES", "8")),
            workers=1,
        )
        variant_reports = []
        for key, trades in results.items():
            params = dict(zip(grid, key))
            curve = trade_equity_curve(trades, config.initial_equity)
            metrics = summarize(trades, curve, config.initial_equity)
            variant_reports.append({
                "params": params,
                "trade_count": metrics["n"],
                "win_rate": self._json_number(metrics["win_rate"]),
                "raw_sharpe": self._json_number(metrics["sharpe_daily_ann"]),
                "profit_factor": self._json_number(metrics["profit_factor"]),
                "max_drawdown_pct": self._json_number(metrics["max_dd_pct"]),
                "expectancy_r": self._json_number(metrics["mean_r"]),
                "simulated_slippage_drag": self._json_number(
                    sum(getattr(trade, "slippage_cost", 0.0) for trade in trades)
                ),
                "status": "tested_not_promoted",
            })
        variant_reports.sort(
            key=lambda item: (
                item["expectancy_r"] if item["expectancy_r"] is not None else -math.inf
            ),
            reverse=True,
        )
        trial_count = register_trials(
            cycle_id, len(results), trial_ledger_path,
            win_rate_records=variant_reports,
        )
        candidate_report = None
        if folds:
            latest = folds[-1]
            latest_params = latest["params"]
            matching_test_days = set()
            for fold in folds:
                if fold["params"] == latest_params:
                    start_text, end_text = fold["test"].split("..")
                    start_day = datetime.fromisoformat(start_text).date()
                    end_day = datetime.fromisoformat(end_text).date()
                    matching_test_days.update(
                        trade.entry_ts.astimezone(NY).date()
                        for trade in results.get(
                            tuple(latest_params[name] for name in grid), []
                        )
                        if start_day <= trade.entry_ts.astimezone(NY).date() <= end_day
                    )
            selected_key = tuple(latest_params[name] for name in grid)
            selected_trades = [
                trade for trade in results.get(selected_key, [])
                if trade.entry_ts.astimezone(NY).date() in matching_test_days
            ]
            selected_curve = trade_equity_curve(selected_trades, config.initial_equity)
            metrics = summarize(selected_trades, selected_curve, config.initial_equity)
            candidate_id = (
                f"research-{fingerprint[:12]}-"
                f"{hashlib.sha256(json.dumps(latest_params, sort_keys=True).encode()).hexdigest()[:8]}"
            )
            holdout = validate_final_holdout(
                symbols, config, latest_params, candidate_id,
                directory=self.root / "data" / "final_holdout",
                ledger_path=self.root / "logs" / "final_holdout_consumption.json",
            )
            slippage_drag = sum(
                getattr(trade, "slippage_cost", 0.0) for trade in selected_trades
            )
            candidate = {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "generated_at": now.isoformat(),
                "source": "24/7_researcher",
                "symbols": symbols,
                "params": latest_params,
                "paper_trades": len(selected_trades),
                "sharpe": self._json_number(metrics["sharpe_daily_ann"]),
                "observations": len(selected_curve),
                "profit_factor": self._json_number(metrics["profit_factor"]),
                "max_drawdown_pct": self._json_number(metrics["max_dd_pct"]),
                "n_trials": trial_count,
                "expected_slippage_drag": self._json_number(slippage_drag),
                "simulated_slippage_drag": self._json_number(slippage_drag),
                "simulated_transaction_costs": self._json_number(metrics["costs"]),
                "forward_paper_expectancy_r": None,
            }
            audit = stage_candidate(
                candidate,
                holdout,
                self.root / "logs" / "strategy_audit_report.json",
                self.root / "data" / "staging",
            )
            candidate_report = {
                "candidate_id": candidate_id,
                "status": audit["status"],
                "params": latest_params,
                "paper_trades": len(selected_trades),
                "holdout": holdout,
                "reason": audit["reason"],
            }

        cumulative = trial_count
        completed_at = datetime.now(timezone.utc)
        status = (
            "candidate_staged" if candidate_report and candidate_report["status"] == "staged_candidate"
            else "completed"
        )
        message = (
            f"Walk-forward evaluated {len(results)} new parameter variants across {len(symbols)} "
            f"equities, Sir. {len(folds)} folds qualified; "
            f"{'candidate ' + str(candidate_report['candidate_id']) + ' is staged for review' if candidate_report and candidate_report['status'] == 'staged_candidate' else 'no candidate cleared every promotion gate'}."
        )
        if candidate_report and candidate_report["status"] != "staged_candidate":
            message += f" Gate result: {candidate_report['reason']}."
        message += " Research remains isolated from order execution, Sir."
        if self._context_error:
            message += f" Market-context warning: {self._context_error}"
        state = {
            **running_state,
            "status": status,
            "updated_at": completed_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "current_task": "Monitoring for new historical bars",
            "message": message,
            "variants_tested": len(results),
            "cumulative_trials": cumulative,
            "qualified_folds": len(folds),
            "variants": variant_reports,
            "candidate": candidate_report,
            "execution_enabled": False,
        }
        self._write_agent_state(state)
        update_audit_research_activity(
            cycle_id, status, message, planned_trials, len(results), audit_path,
        )
        update_trial_activity(
            cycle_id, len(results), status, message,
            trial_ledger_path,
        )
        self._last_discovery_fingerprint = fingerprint
        log.info("RESEARCHER AGENT %s", message)
        return state

    def _read_agent_state(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self._agent_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"Invalid researcher state schema: {self._agent_state_path}")
        return value

    def research_once(self) -> dict[str, Any]:
        symbols = tuple(self.symbols_provider())
        now = datetime.now(timezone.utc)
        regimes = {}
        timeframe_context = {}
        if symbols:
            daily = self.client.bars(symbols, "1Day", now - timedelta(days=400), now, limit=500)
            hourly = self.client.bars(symbols, "1Hour", now - timedelta(days=90), now, limit=3000)

            for symbol in symbols:
                def convert(rows, interval):
                    converted = []
                    for item in rows:
                        start = datetime.fromisoformat(item["t"].replace("Z", "+00:00"))
                        if start + interval <= now:
                            converted.append(Bar(
                                start, float(item["o"]), float(item["h"]), float(item["l"]),
                                float(item["c"]), float(item["v"]),
                            ))
                    return sorted(converted, key=lambda bar: bar.start)

                bars = convert(daily.get(symbol, []), timedelta(days=1))
                hourly_bars = convert(hourly.get(symbol, []), timedelta(hours=1))
                for timeframe, values in (("1h", hourly_bars), ("1d", bars)):
                    quality_issue = _bar_quality_issue(values, timeframe)
                    if quality_issue:
                        log.warning(
                            "RESEARCH DATA QUALITY %s %s: %s; suppressing indicators.",
                            symbol, timeframe, quality_issue,
                        )
                        if timeframe == "1h":
                            hourly_bars = []
                        else:
                            bars = []
                regime = classify_regime(symbol, bars, now)
                if regime:
                    regimes[symbol] = asdict(regime)
                    log.info("REGIME DETECTED %s: %s ADX=%.1f ATR%%=%.2f slope=%+.3f%%",
                             symbol, regime.label, regime.adx, 100 * regime.atr_pct,
                             100 * regime.ma_slope_pct)
                timeframe_context[symbol] = {}
                for name, values in (("1h", hourly_bars), ("1d", bars)):
                    closes = [bar.close for bar in values]
                    averages = {
                        f"sma{period}": simple_moving_average(closes, period)
                        for period in (20, 50, 200)
                    }
                    current = closes[-1] if closes else None
                    aligned = None
                    if current is not None and all(value is not None for value in averages.values()):
                        aligned = current > averages["sma20"] > averages["sma50"] > averages["sma200"]
                    timeframe_context[symbol][name] = {
                        **averages,
                        "adx": average_directional_index(values),
                        "bullish_alignment": aligned,
                        "bars": len(values),
                    }
        eligible = [
            {**strategy, "eligible_symbols": [
                symbol for symbol, regime in regimes.items()
                if regime["label"] in strategy["regimes"]
            ]}
            for strategy in STRATEGY_CATALOG
        ]
        state = {
            "updated_at": now.isoformat(),
            "regimes": regimes,
            "timeframe_context": timeframe_context,
            "catalog": eligible,
            "execution_enabled": False,
            "asset_class_scope": sorted({
                "crypto" if "/" in symbol else "equities" for symbol in symbols
            }),
            "research_only_symbols": [symbol for symbol in symbols if "/" in symbol],
            "note": "Crypto pairs are monitored 24/7 as research data; crypto order execution remains disabled "
                    "until broker-native protective exits are verified.",
        }
        state_path = self.root / "logs" / "research_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp_path.replace(state_path)
        log.info("RESEARCH LOOP completed for %d symbols; research-only strategies=%d",
                 len(regimes), len(eligible))
        return state

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                previous = self._read_agent_state() or {}
                self._write_agent_state({
                    **previous,
                    "status": "refreshing_market_context",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "current_task": "Refreshing hourly and daily market context",
                    "message": (
                        "The 24/7 Researcher is refreshing market context before its "
                        "independent local-data discovery cycle, Sir."
                    ),
                    "execution_enabled": False,
                })
                self.research_once()
                self._context_error = None
            except Exception as exc:
                log.exception("Research context cycle failed; discovery will still run.")
                self._context_error = f"{type(exc).__name__}: {exc}"
                prior = self._read_agent_state() or {}
                self._write_agent_state({
                    **prior,
                    "status": "research_context_error",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "current_task": "Refreshing market context",
                    "message": (
                        f"Market-context refresh failed, Sir: {self._context_error}. "
                        "The independent historical-data discovery cycle will continue, Sir."
                    ),
                    "context_error": self._context_error,
                    "execution_enabled": False,
                })
            try:
                self.discovery_once()
            except Exception as exc:
                log.exception("Research discovery cycle failed.")
                message = (
                    f"Background discovery failed, Sir: {type(exc).__name__}: {exc}. "
                    "The paper execution thread remains separate, Sir."
                )
                self._write_agent_state({
                    "status": "error",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "current_task": "Recovering from discovery error",
                    "message": message,
                    "variants_tested": 0,
                    "variants": [],
                    "candidate": None,
                    "context_error": self._context_error,
                    "execution_enabled": False,
                })
            stop_event.wait(self.interval_seconds)


def strategy_catalog() -> list[dict[str, Any]]:
    return [dict(strategy) for strategy in STRATEGY_CATALOG]
