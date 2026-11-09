"""Single-process supervisor for the paper runner, dashboard, research, and arsenal."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any
from typing import Callable

from server.autonomous_runner import (
    AlpacaClient,
    AutonomousRunner,
    configure_logging,
    load_paper_settings,
    load_crypto_symbols,
    main as runner_main,
    single_instance_lock,
    write_heartbeat,
)
from server.dashboard import create_server
from server.evolution import ARSENAL_PATH, ArsenalEvolution
from server.healer import diagnose_exception, install_exception_hooks
from server.ingestion import ResearchIngestionWorker
from server.io_utils import atomic_write_json
from server.researcher import StrategyResearcher

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("apex.master")
READINESS_SPREAD_BPS = 1
READINESS_SLIPPAGE_BPS = 2
READINESS_ROUND_TRIP_FRICTION_BPS = 2 * (
    READINESS_SPREAD_BPS + READINESS_SLIPPAGE_BPS
)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flatten_orders(orders: Any) -> list[dict[str, Any]]:
    flattened = []
    seen = set()

    def visit(order: Any) -> None:
        if not isinstance(order, dict):
            return
        order_id = order.get("id")
        key = str(order_id) if order_id is not None else f"anonymous:{id(order)}"
        if key in seen:
            return
        seen.add(key)
        flattened.append(order)
        legs = order.get("legs")
        if isinstance(legs, list):
            for leg in legs:
                visit(leg)

    if isinstance(orders, list):
        for order in orders:
            visit(order)
    return flattened


def _order_datetime(order: dict[str, Any]):
    from datetime import datetime, timezone

    for key in ("filled_at", "updated_at", "submitted_at", "created_at"):
        value = order.get(key)
        if isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
            except ValueError:
                continue
    return None


def _order_time(order: dict[str, Any]) -> str | None:
    timestamp = _order_datetime(order)
    return timestamp.date().isoformat() if timestamp else None


def _closed_trade_metrics(orders: Any) -> dict[str, Any]:
    records = _flatten_orders(orders)
    orders_by_id = {}
    for order in records:
        if order.get("id") is not None:
            orders_by_id[str(order["id"])] = order

    def fill_order(order: dict[str, Any]):
        timestamp = _order_datetime(order)
        qty = _finite_number(order.get("filled_qty"))
        price = _finite_number(order.get("filled_avg_price"))
        side = str(order.get("side", "")).lower()
        if qty is None or qty <= 0 or price is None or side not in {"buy", "sell"}:
            return None
        symbol = order.get("symbol")
        parent_id = order.get("parent_order_id")
        if not symbol and parent_id:
            parent = orders_by_id.get(str(parent_id), {})
            symbol = parent.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            return None
        return {
            "symbol": symbol,
            "direction": 1 if side == "buy" else -1,
            "qty": qty,
            "price": price,
            "timestamp": timestamp,
            "parent_order_id": parent_id,
        }

    fills = [fill for order in records if (fill := fill_order(order)) is not None]
    fills.sort(key=lambda fill: (
        fill["timestamp"] is None,
        fill["timestamp"],
        1 if fill["parent_order_id"] else 0,
    ))
    open_lots: dict[str, list[dict[str, Any]]] = {}
    closed = []
    for fill in fills:
        remaining = fill["qty"]
        lots = open_lots.setdefault(fill["symbol"], [])
        while remaining > 1e-8 and lots and lots[0]["direction"] != fill["direction"]:
            lot = lots[0]
            matched = min(remaining, lot["remaining_qty"])
            gross = lot["direction"] * (fill["price"] - lot["entry_price"]) * matched
            friction = (
                (lot["entry_price"] + fill["price"]) * matched
                * READINESS_ROUND_TRIP_FRICTION_BPS / 10_000
            )
            if not all(math.isfinite(value) for value in (gross, friction)):
                break
            lot["gross_pnl"] += gross
            lot["friction"] += friction
            lot["net_pnl"] += gross - friction
            lot["remaining_qty"] -= matched
            remaining -= matched
            lot["session"] = (
                fill["timestamp"].date().isoformat() if fill["timestamp"] else lot["session"]
            )
            if lot["remaining_qty"] <= 1e-8:
                closed.append({
                    "gross_pnl": lot["gross_pnl"],
                    "friction": lot["friction"],
                    "net_pnl": lot["net_pnl"],
                    "session": lot["session"],
                })
                lots.pop(0)
        if remaining > 1e-8:
            lots.append({
                "direction": fill["direction"],
                "remaining_qty": remaining,
                "entry_price": fill["price"],
                "gross_pnl": 0.0,
                "friction": 0.0,
                "net_pnl": 0.0,
                "session": (
                    fill["timestamp"].date().isoformat() if fill["timestamp"] else None
                ),
            })

    wins = [trade["net_pnl"] for trade in closed if trade["net_pnl"] > 0]
    losses = [-trade["net_pnl"] for trade in closed if trade["net_pnl"] < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    return {
        "trades": closed,
        "closed_trade_count": len(closed),
        "sessions": len({trade["session"] for trade in closed if trade["session"]}),
        "win_rate": len(wins) / len(closed) if closed else None,
        "profit_factor": (
            gross_profit / gross_loss if gross_loss
            else math.inf if gross_profit else None
        ),
        "expectancy": (
            sum(trade["net_pnl"] for trade in closed) / len(closed)
            if closed else None
        ),
    }


def calculate_live_readiness(
    orders: Any,
    researcher: Any,
    total_trials: int,
    risk_settings: Any,
    recent_trial_metrics: Any = None,
    api_rejection_breaker_enabled: bool = False,
    order_history_status: str = "available",
) -> dict[str, Any]:
    """Calculate a deterministic, read-only readiness index from persisted evidence."""
    trial_count = (
        total_trials if isinstance(total_trials, int)
        and not isinstance(total_trials, bool) and total_trials >= 0 else 0
    )
    trade_metrics = _closed_trade_metrics(orders)
    trade_count = trade_metrics["closed_trade_count"]
    sessions = trade_metrics["sessions"]
    execution_score = 0.0 if trade_count < 10 else min(25.0, (trade_count - 10) * 0.625)
    execution_missing = []
    if order_history_status != "available":
        execution_missing.append(f"Paper order history is {order_history_status}.")
    if trade_count < 10:
        execution_missing.append(f"Need at least 10 fully closed paper trades; found {trade_count}.")
    elif trade_count < 50:
        execution_missing.append(f"Need 50 closed paper trades for full sample credit; found {trade_count}.")
    if sessions < 3:
        execution_missing.append(f"Need trades across 3 market sessions; found {sessions}.")
    if trade_count >= 50 and sessions < 3:
        execution_score = min(execution_score, 20.0)

    expectancy = trade_metrics["expectancy"]
    profit_factor = trade_metrics["profit_factor"]
    friction_score = 0.0
    friction_missing = []
    if expectancy is None:
        friction_missing.append("Closed trade fills are unavailable for friction-adjusted expectancy.")
    elif expectancy <= 0:
        friction_missing.append("Friction-adjusted expectancy must be positive.")
    else:
        pf_factor = min(profit_factor / 1.5, 1.0) if profit_factor is not None else 0.0
        expectancy_factor = min(expectancy / 2.0, 1.0)
        friction_score = 25.0 * expectancy_factor * pf_factor
        if expectancy <= 2:
            friction_score = min(friction_score, 24.99)
            friction_missing.append("Raise friction-adjusted expectancy above $2.00 per trade.")
        if profit_factor is None or profit_factor < 1.5:
            friction_missing.append("Raise friction-adjusted profit factor to at least 1.5.")

    risk = getattr(risk_settings, "risk", risk_settings)
    daily_limit = _finite_number(getattr(risk, "daily_drawdown_kill_pct", None))
    per_trade_risk = _finite_number(getattr(risk, "risk_per_trade_pct", None))
    effective_daily_limit = min(daily_limit, 0.02) if daily_limit is not None and daily_limit > 0 else None
    dd_verified = effective_daily_limit is not None and effective_daily_limit <= 0.02
    sizing_verified = per_trade_risk is not None and 0 < per_trade_risk <= 0.01
    risk_score = (10.0 if dd_verified else 0.0) + (10.0 if sizing_verified else 0.0)
    risk_missing = []
    if not dd_verified:
        risk_missing.append("Configure and verify a hard daily drawdown stop at or below 2%.")
    if not sizing_verified:
        risk_missing.append("Enforce position risk at or below 1% of account equity per trade.")
    if not api_rejection_breaker_enabled:
        risk_missing.append("Add and verify an API rejection breaker that halts after 3 consecutive errors.")
    else:
        risk_score += 5.0

    if recent_trial_metrics is None and isinstance(researcher, dict):
        recent_trial_metrics = researcher.get("recent_win_rates")
    valid_samples = []
    if isinstance(recent_trial_metrics, list):
        for item in recent_trial_metrics:
            if not isinstance(item, dict):
                continue
            rate = _finite_number(item.get("win_rate"))
            count = item.get("trade_count")
            if (
                rate is not None and not isinstance(item.get("win_rate"), bool)
                and 0 <= rate <= 1 and isinstance(count, int)
                and not isinstance(count, bool) and count > 0
            ):
                valid_samples.append({"win_rate": rate, "trade_count": count})
    sample = valid_samples[-142:]
    sample_trades = sum(item["trade_count"] for item in sample)
    backtest_rate = (
        sum(float(item["win_rate"]) * item["trade_count"] for item in sample) / sample_trades
        if len(sample) == 142 and sample_trades else None
    )

    stability_score = 0.0
    stability_missing = []
    paper_rate = trade_metrics["win_rate"]
    if trial_count < 142 or len(sample) < 142:
        stability_missing.append(
            f"Need 142 research trials with recorded walk-forward win rates; "
            f"found {min(trial_count, len(sample))} usable samples."
        )
    if backtest_rate is None:
        stability_missing.append("A walk-forward backtest win rate is not available.")
    if paper_rate is None:
        stability_missing.append("Closed paper trades are required to calculate live win rate.")
    if total_trials >= 142 and backtest_rate is not None and paper_rate is not None:
        deviation = abs(paper_rate - backtest_rate)
        excess = max(0.0, deviation - 0.15)
        deductions = math.floor((excess + 1e-12) / 0.05)
        stability_score = max(0.0, 25.0 - deductions * 5.0)
        if deviation > 0.15:
            stability_missing.append(
                f"Paper win rate differs from WFO by {deviation:.1%}; keep deviation within 15%."
            )

    pillars = {
        "execution_sample_size": {
            "score": round(execution_score, 2), "max_score": 25,
            "closed_trades": trade_count, "market_sessions": sessions,
            "missing_criteria": execution_missing,
        },
        "friction_adjusted_expectancy": {
            "score": round(friction_score, 2), "max_score": 25,
            "expectancy_per_trade_usd": round(expectancy, 2) if expectancy is not None else None,
            "profit_factor": (
                round(profit_factor, 3) if profit_factor is not None and math.isfinite(profit_factor)
                else "infinite" if profit_factor == math.inf else None
            ),
            "modeled_spread_bps": READINESS_SPREAD_BPS,
            "modeled_slippage_bps": READINESS_SLIPPAGE_BPS,
            "modeled_round_trip_cost_bps": READINESS_ROUND_TRIP_FRICTION_BPS,
            "missing_criteria": friction_missing,
        },
        "risk_guardrails": {
            "score": round(risk_score, 2), "max_score": 25,
            "effective_daily_drawdown_limit": effective_daily_limit,
            "risk_per_trade": per_trade_risk,
            "api_rejection_breaker_enabled": api_rejection_breaker_enabled,
            "missing_criteria": risk_missing,
        },
        "walk_forward_stability": {
            "score": round(stability_score, 2), "max_score": 25,
            "paper_win_rate": round(paper_rate, 4) if paper_rate is not None else None,
            "backtest_win_rate": round(backtest_rate, 4) if backtest_rate is not None else None,
            "research_trials": trial_count,
            "win_rate_samples": len(sample),
            "missing_criteria": stability_missing,
        },
    }
    score = max(0, min(100, round(sum(pillar["score"] for pillar in pillars.values()))))
    status = (
        "UNSAFE" if score < 40 else
        "PAPER STABLE" if score < 70 else
        "LIVE CANDIDATE" if score < 90 or risk_score < 25 else
        "FULL DEPLOYMENT"
    )
    return {
        "readiness_score": score,
        "readiness_status": status,
        "readiness_breakdown": {
            **pillars,
            "total_score": score,
            "max_score": 100,
            "missing_criteria": [
                criterion for pillar in pillars.values()
                for criterion in pillar["missing_criteria"]
            ],
            "status_thresholds": {
                "UNSAFE": "0-39",
                "PAPER STABLE": "40-69",
                "LIVE CANDIDATE": "70-89",
                "FULL DEPLOYMENT": "90-100 with all critical risk guardrails satisfied",
            },
            "paper_only_notice": "Readiness is informational and does not authorize or enable live orders.",
        },
    }


def _supervise(name: str, target: Callable[[], None], stop_event: threading.Event) -> None:
    delay = 1.0
    while not stop_event.is_set():
        try:
            target()
            if not stop_event.is_set():
                raise RuntimeError(f"{name} component exited unexpectedly.")
            return
        except Exception as exc:
            diagnose_exception(name, exc)
            log.exception("%s component failed; restarting in %.1f seconds.", name, delay)
            if stop_event.wait(delay):
                return
            delay = min(delay * 2, 60.0)


def _read_research_symbols(settings) -> tuple[str, ...]:
    try:
        heartbeat = json.loads(
            (ROOT / "logs" / "runner_heartbeat.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        heartbeat = {}
    try:
        profile = json.loads((ROOT / "logs" / "daily_profile.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        profile = {}
    heartbeat_symbols = heartbeat.get("symbols") if isinstance(heartbeat, dict) else None
    if isinstance(heartbeat_symbols, list):
        heartbeat_symbols = tuple(dict.fromkeys(
            str(item).upper() for item in heartbeat_symbols if str(item).isalpha()
        ))
    else:
        heartbeat_symbols = ()
    candidates = profile.get("universe") or profile.get("symbols") if isinstance(profile, dict) else None
    equity_symbols = ()
    if isinstance(candidates, list):
        equity_symbols = tuple(dict.fromkeys(
            str(item).upper() for item in candidates if str(item).isalpha()
        ))
    return tuple(dict.fromkeys(
        (heartbeat_symbols or equity_symbols or settings.watchlist) + load_crypto_symbols()
    ))


def _watchdog(stop_event: threading.Event, components: dict[str, threading.Thread]) -> None:
    health_path = ROOT / "logs" / "master_health.json"
    while not stop_event.is_set():
        statuses = {name: "running" if thread.is_alive() else "stopped"
                    for name, thread in components.items()}
        failed = [name for name, status in statuses.items() if status == "stopped"]
        if failed:
            log.error("WATCHDOG detected stopped components: %s", ", ".join(failed))
        try:
            atomic_write_json(health_path, {
                "timestamp": time.time(),
                "status": "active" if not failed else "degraded",
                "components": statuses,
            }, indent=2)
        except OSError:
            log.exception("Unable to publish master health telemetry.")
        stop_event.wait(15)


def run_master() -> int:
    configure_logging()
    install_exception_hooks()
    settings = load_paper_settings()
    client = AlpacaClient(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_data_feed)
    stop_event = threading.Event()
    dashboard = create_server("127.0.0.1", 5000)
    runner = AutonomousRunner(settings, client)
    researcher = StrategyResearcher(
        AlpacaClient(settings.alpaca_key, settings.alpaca_secret, settings.alpaca_data_feed),
        lambda: _read_research_symbols(settings),
    )
    ingestion_worker = ResearchIngestionWorker(ROOT)
    evolution = ArsenalEvolution(ARSENAL_PATH)

    def run_dashboard() -> None:
        log.info("Dashboard listening at http://127.0.0.1:5000")
        dashboard.serve_forever(poll_interval=0.25)

    def run_trader() -> None:
        with single_instance_lock():
            write_heartbeat("waiting", "Unified master started; checking Alpaca market clock.",
                            settings.watchlist)
            asyncio.run(runner.run_forever(stop_event))

    def run_research() -> None:
        researcher.run(stop_event)

    def run_ingestion() -> None:
        ingestion_worker.run(stop_event)

    def run_evolution() -> None:
        evolution.run(stop_event)

    component_targets = {
        "dashboard": run_dashboard,
        "runner": run_trader,
        "research": run_research,
        "ingestion": run_ingestion,
        "evolution": run_evolution,
    }
    components = {
        name: threading.Thread(
            target=_supervise, args=(name, target, stop_event),
            name=f"apex-{name}", daemon=False,
        )
        for name, target in component_targets.items()
    }
    watchdog = threading.Thread(
        target=_watchdog, args=(stop_event, components), name="apex-watchdog", daemon=False,
    )
    log.info("Starting unified PAPER-only supervisor for %s.", ",".join(settings.watchlist))
    for thread in components.values():
        thread.start()
    watchdog.start()
    try:
        while any(thread.is_alive() for thread in components.values()):
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Shutdown requested; stopping all unified components.")
    finally:
        stop_event.set()
        dashboard.shutdown()
        for thread in components.values():
            thread.join()
        watchdog.join()
        dashboard.server_close()
        try:
            write_heartbeat("disconnected", "Unified master stopped.", settings.watchlist)
        except OSError:
            log.exception("Unable to persist stopped runner heartbeat.")
        log.info("Unified master stopped cleanly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unified ApexQuant paper-trading supervisor.")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the paper API and target exposure without starting components")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            return runner_main(["--dry-run"])
        return run_master()
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.getLogger("apex.master").exception("Unified master startup failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
