"""One-candidate validation against a fixed, separately maintained final holdout."""
from __future__ import annotations

import math
import json
import logging
from pathlib import Path
from typing import Any

from backtest.data import load_dir
from backtest.engine import BTConfig, SignalFn, run_backtest
from backtest.holdout_lock import (
    DEFAULT_LOCK_PATH as DEFAULT_HOLDOUT_LOCK,
    canonical_symbols,
    fingerprint_directory,
    verify as verify_holdout_lock,
)
from backtest.metrics import summarize, trade_equity_curve
from backtest.walkforward import apply_params

ROOT = Path(__file__).resolve().parents[1]
FINAL_HOLDOUT_DIR = ROOT / "data" / "final_holdout"
log = logging.getLogger("apex.backtest.holdout")


def validate_final_holdout(
    symbols: list[str] | tuple[str, ...],
    base: BTConfig,
    params: dict[str, float],
    candidate_id: str,
    directory: Path = FINAL_HOLDOUT_DIR,
    ledger_path: Path | None = None,
    signal_fn: SignalFn | None = None,
    lock_path: Path | None = None,
    require_lock: bool | None = None,
) -> dict[str, Any]:
    """Load the isolated holdout only after parameters are frozen; never tune on its results."""
    if not symbols:
        return {
            "available": False,
            "passed": False,
            "reason": "final holdout data unavailable; provide untouched CSVs in data/final_holdout",
            "missing_files": [],
        }
    symbols = canonical_symbols(symbols)
    expected_files = [directory / f"{symbol}.csv" for symbol in symbols]
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        return {
            "available": False,
            "passed": False,
            "reason": "final holdout data unavailable; provide untouched CSVs in data/final_holdout",
            "missing_files": missing,
        }
    if require_lock is None:
        require_lock = directory.resolve() == FINAL_HOLDOUT_DIR.resolve()
    if require_lock:
        lock_result = verify_holdout_lock(
            directory,
            lock_path or DEFAULT_HOLDOUT_LOCK,
            require_locked=True,
        )
        if not lock_result["ok"]:
            return {
                "available": False,
                "passed": False,
                "reason": "final holdout lock rejected: " + lock_result["reason"],
                "missing_files": [],
            }
    ledger_path = ledger_path or (
        ROOT / "logs" / "final_holdout_consumption.json"
        if directory.resolve() == FINAL_HOLDOUT_DIR.resolve()
        else directory.parent / "logs" / "final_holdout_consumption.json"
    )
    try:
        dataset_id = fingerprint_directory(directory, symbols)["dataset_id"]
    except OSError as exc:
        return {
            "available": False,
            "passed": False,
            "reason": f"final holdout data could not be fingerprinted: {exc}",
            "missing_files": [],
        }
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        ledger = {"version": 1, "datasets": {}}
    datasets = ledger.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid final holdout consumption ledger: {ledger_path}")
    prior = datasets.get(dataset_id)
    if prior:
        if prior.get("candidate_id") == candidate_id and isinstance(prior.get("result"), dict):
            return prior["result"]
        return {
            "available": False,
            "passed": False,
            "reason": "this final holdout dataset was already consumed by a different candidate",
            "missing_files": [],
        }
    datasets[dataset_id] = {"candidate_id": candidate_id, "status": "in_progress"}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temp = ledger_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    temp.replace(ledger_path)
    try:
        data = load_dir(str(directory), list(symbols))
    except (OSError, ValueError, KeyError) as exc:
        log.error("Final holdout could not be loaded: %s", exc)
        result = {
            "available": False,
            "passed": False,
            "reason": f"final holdout data invalid: {exc}",
            "missing_files": [],
        }
        datasets[dataset_id] = {"candidate_id": candidate_id, "result": result}
        temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temp.replace(ledger_path)
        return result
    if any(not bars for bars in data.values()):
        result = {
            "available": False,
            "passed": False,
            "reason": "final holdout contains no valid normalized regular-session bars",
            "missing_files": [],
        }
        datasets[dataset_id] = {"candidate_id": candidate_id, "result": result}
        temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temp.replace(ledger_path)
        return result
    config = apply_params(base, params)
    result = (
        run_backtest(data, config)
        if signal_fn is None
        else run_backtest(data, config, signal_fn=signal_fn)
    )
    metrics = summarize(
        result.trades, trade_equity_curve(result.trades, base.initial_equity),
        base.initial_equity,
    )
    finite_sharpe = math.isfinite(metrics["sharpe_daily_ann"])
    passed = bool(
        metrics["n"] >= 30
        and finite_sharpe
        and metrics["sharpe_daily_ann"] > 0
        and metrics["profit_factor"] > 1.3
        and metrics["max_dd_pct"] < 5
        and metrics["mean_r"] > 0
    )
    result = {
        "available": True,
        "passed": passed,
        "reason": "holdout gates passed" if passed else "holdout gates failed",
        "metrics": {
            key: value if math.isfinite(value) else None
            for key, value in metrics.items()
        },
    }
    datasets[dataset_id] = {"candidate_id": candidate_id, "result": result}
    temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    temp.replace(ledger_path)
    return result
