"""Statistical strategy promotion and quarantine; this module never routes orders."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest.metrics import deflated_sharpe_probability
from server.io_utils import atomic_write_json, ensure_directory, secure_file

log = logging.getLogger("apex.evolution")
ROOT = Path(__file__).resolve().parents[1]
ARSENAL_PATH = ROOT / "data" / "strategy_arsenal.json"
TRIAL_LEDGER_PATH = ROOT / "logs" / "strategy_trial_ledger.json"
AUDIT_PATH = ROOT / "logs" / "strategy_audit_report.json"
STAGING_DIR = ROOT / "data" / "staging"
ACTIVE_CANDIDATE_PATH = ROOT / "data" / "active_paper_candidate.json"
BASE_SHARPE_HURDLE = 1.5
_trial_ledger_lock = threading.Lock()
_candidate_audit_lock = threading.Lock()

SEED_ARSENAL = {
    "version": 1,
    "updated_at": None,
    "strategies": [
        {
            "id": "equity-orb-v1", "name": "Opening Range Breakout", "asset_class": "equities",
            "status": "research_only", "paper_trades": 0, "sharpe": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "recent_outcomes_r": [], "recent_expectancy_r": None, "baseline_expectancy_r": None,
        },
        {
            "id": "equity-vwap-v1", "name": "VWAP Mean Reversion", "asset_class": "equities",
            "status": "research_only", "paper_trades": 0, "sharpe": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "recent_outcomes_r": [], "recent_expectancy_r": None, "baseline_expectancy_r": None,
        },
        {
            "id": "equity-rvol-v1", "name": "RVOL Momentum", "asset_class": "equities",
            "status": "research_only", "paper_trades": 0, "sharpe": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "recent_outcomes_r": [], "recent_expectancy_r": None, "baseline_expectancy_r": None,
        },
        {
            "id": "equity-atr-swing-v1", "name": "ATR Volatility Swing", "asset_class": "equities",
            "status": "research_only", "paper_trades": 0, "sharpe": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "recent_outcomes_r": [], "recent_expectancy_r": None, "baseline_expectancy_r": None,
        },
        {
            "id": "equity-mtf-v1", "name": "Multi-Timeframe Trend", "asset_class": "equities",
            "status": "research_only", "paper_trades": 0, "sharpe": None,
            "profit_factor": None, "max_drawdown_pct": None,
            "recent_outcomes_r": [], "recent_expectancy_r": None, "baseline_expectancy_r": None,
        },
    ],
}


def normalize_trial_ledger(value: Any) -> dict[str, Any]:
    """Return a valid ledger, preserving usable counts from older ledger shapes."""
    if not isinstance(value, dict):
        return {"version": 1, "cycles": {}, "num_trials": 0, "total_trials": 0}
    raw_cycles = value.get("cycles")
    if raw_cycles is not None and not isinstance(raw_cycles, dict):
        return {"version": 1, "cycles": {}, "num_trials": 0, "total_trials": 0}
    cycles: dict[str, int] = {}
    if isinstance(raw_cycles, dict):
        for cycle_id, count in raw_cycles.items():
            if (
                isinstance(cycle_id, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
            ):
                cycles[cycle_id] = count
    legacy_total = value.get("num_trials", value.get("total_trials"))
    if (
        not cycles
        and isinstance(legacy_total, int)
        and not isinstance(legacy_total, bool)
        and legacy_total > 0
    ):
        cycles["legacy_import"] = legacy_total
    result = {
        "version": 1,
        "cycles": cycles,
        "num_trials": sum(cycles.values()),
        "total_trials": sum(cycles.values()),
    }
    recent_win_rates = value.get("recent_win_rates")
    if isinstance(recent_win_rates, list):
        valid_samples = []
        for sample in recent_win_rates:
            if not isinstance(sample, dict):
                continue
            cycle_id = sample.get("cycle_id")
            win_rate = _finite_number(sample.get("win_rate"))
            trade_count = sample.get("trade_count")
            if (
                isinstance(cycle_id, str)
                and win_rate is not None and 0 <= win_rate <= 1
            ) and isinstance(trade_count, int) and not isinstance(trade_count, bool) and trade_count > 0:
                valid_samples.append({
                    "cycle_id": cycle_id,
                    "win_rate": win_rate,
                    "trade_count": trade_count,
                })
        if valid_samples:
            result["recent_win_rates"] = valid_samples[-142:]
    for key in ("last_updated_at", "latest_cycle_id", "latest_activity"):
        if key in value:
            result[key] = value[key]
    return result


@contextmanager
def _trial_ledger_file_lock(path: Path):
    """Serialize ledger updates across CLI and worker processes."""
    ensure_directory(path.parent)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        secure_file(lock_path)
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_trial_ledger(path: Path, *, strict: bool) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if strict:
            raise ValueError(f"Trial ledger is invalid; refusing to reset trial history: {path}") from exc
        log.warning("Invalid strategy trial ledger; reinitializing %s.", path)
        raw = None

    if strict and raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"Trial ledger has an invalid schema: {path}")
        cycles = raw.get("cycles")
        if cycles is not None and (
            not isinstance(cycles, dict)
            or any(
                not isinstance(cycle_id, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for cycle_id, count in cycles.items()
            )
        ):
            raise ValueError(f"Trial ledger has invalid cycle counts: {path}")
        totals = [
            raw[key] for key in ("num_trials", "total_trials") if key in raw
        ]
        if any(
            not isinstance(total, int) or isinstance(total, bool) or total < 0
            for total in totals
        ):
            raise ValueError(f"Trial ledger has an invalid total: {path}")
        if len(totals) == 2 and totals[0] != totals[1]:
            raise ValueError(f"Trial ledger cumulative totals disagree: {path}")
        total = totals[0] if totals else None
        if cycles is None and total is None:
            raise ValueError(f"Trial ledger has no trial counts: {path}")
        if isinstance(cycles, dict) and total is not None and cycles:
            if total != sum(cycles.values()):
                raise ValueError(f"Trial ledger total does not match its cycles: {path}")

    return normalize_trial_ledger(raw)


def load_trial_ledger(path: Path = TRIAL_LEDGER_PATH, *, strict: bool = False) -> dict[str, Any]:
    """Load the trial ledger; strict mode refuses to discard or reset trial history."""
    with _trial_ledger_lock, _trial_ledger_file_lock(path):
        existed = path.exists()
        ledger = _read_trial_ledger(path, strict=strict)
        if strict:
            return ledger
        if existed:
            try:
                original = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                original = None
        else:
            original = None
        if existed and original != ledger:
            atomic_write_json(path, ledger, indent=2)
        return ledger


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def register_trials(cycle_id: str, trial_count: int,
                    path: Path = TRIAL_LEDGER_PATH,
                    win_rate_records: list[dict[str, Any]] | None = None) -> int:
    """Persist unique sweep counts, including failed sweeps reserved before execution."""
    if not cycle_id or not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count < 0:
        raise ValueError("A cycle ID and non-negative integer trial count are required.")
    with _trial_ledger_lock, _trial_ledger_file_lock(path):
        ledger = _read_trial_ledger(path, strict=True)
        cycles = ledger["cycles"]
        if cycle_id not in cycles:
            cycles[cycle_id] = trial_count
        ledger["num_trials"] = sum(cycles.values())
        ledger["total_trials"] = ledger["num_trials"]
        valid_records = []
        for record in win_rate_records or []:
            if not isinstance(record, dict):
                continue
            win_rate = _finite_number(record.get("win_rate"))
            trade_count = record.get("trade_count")
            if (
                win_rate is not None and 0 <= win_rate <= 1
                and isinstance(trade_count, int)
                and not isinstance(trade_count, bool) and trade_count > 0
            ):
                valid_records.append({
                    "cycle_id": cycle_id,
                    "win_rate": win_rate,
                    "trade_count": trade_count,
                })
        if valid_records:
            samples = ledger.get("recent_win_rates", [])
            samples = [sample for sample in samples if sample.get("cycle_id") != cycle_id]
            ledger["recent_win_rates"] = (samples + valid_records)[-142:]
        ledger["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        ledger["latest_cycle_id"] = cycle_id
        atomic_write_json(path, ledger, indent=2)
        return int(ledger["num_trials"])


def update_trial_activity(cycle_id: str, trial_count: int, status: str, message: str,
                          path: Path = TRIAL_LEDGER_PATH) -> dict[str, Any]:
    """Publish the research worker's current task alongside cumulative trial counts."""
    with _trial_ledger_lock, _trial_ledger_file_lock(path):
        ledger = _read_trial_ledger(path, strict=True)
        cycles = ledger["cycles"]
        if cycle_id and cycle_id not in cycles and status == "completed":
            cycles[cycle_id] = trial_count
        ledger["num_trials"] = sum(cycles.values())
        ledger["total_trials"] = ledger["num_trials"]
        ledger["latest_activity"] = {
            "cycle_id": cycle_id,
            "status": status,
            "trial_count": trial_count,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(path, ledger, indent=2)
        return ledger["latest_activity"]


def update_audit_research_activity(
    cycle_id: str,
    status: str,
    message: str,
    planned_trials: int = 0,
    variants_tested: int = 0,
    path: Path = AUDIT_PATH,
) -> dict[str, Any]:
    """Persist researcher progress without replacing the latest candidate audit."""
    with _candidate_audit_lock:
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            audit = {}
        if not isinstance(audit, dict):
            raise ValueError(f"Invalid strategy audit report schema: {path}")
        activity = {
            "cycle_id": cycle_id,
            "status": status,
            "message": message,
            "planned_trials": planned_trials,
            "variants_tested": variants_tested,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        audit["research_activity"] = activity
        atomic_write_json(path, audit, indent=2)
        return activity


def deflated_sharpe(sharpe: float | None, observations: int,
                    n_trials: int, skewness: float = 0.0,
                    kurtosis: float = 3.0) -> dict[str, float | int | bool | None]:
    """Apply the Deflated Sharpe Ratio with an all-trials multiple-testing correction."""
    value = _finite_number(sharpe)
    return deflated_sharpe_probability(
        value, observations, n_trials, skewness=skewness, kurtosis=kurtosis,
        minimum_sharpe=BASE_SHARPE_HURDLE,
    )


def stage_candidate(candidate: dict[str, Any], holdout: dict[str, Any],
                    path: Path = AUDIT_PATH, staging_dir: Path = STAGING_DIR) -> dict[str, Any]:
    """Write an audit record and stage only a statistically and blindly validated candidate."""
    trials = int(candidate.get("n_trials", 1))
    cpcv = candidate.get("cpcv")
    cpcv_dsr = cpcv.get("dsr") if isinstance(cpcv, dict) else None
    cpcv_sharpe = _finite_number(
        cpcv_dsr.get("raw_sharpe") if isinstance(cpcv_dsr, dict) else None
    )
    skewness = _finite_number(
        cpcv_dsr.get("skewness") if isinstance(cpcv_dsr, dict) else None
    )
    kurtosis = _finite_number(
        cpcv_dsr.get("kurtosis") if isinstance(cpcv_dsr, dict) else None
    )
    sharpe_result = deflated_sharpe(
        cpcv_sharpe if cpcv_sharpe is not None else _finite_number(candidate.get("sharpe")),
        int(cpcv_dsr.get("observations", 0)) if isinstance(cpcv_dsr, dict) else 0,
        trials,
        skewness if skewness is not None else 0.0,
        kurtosis if kurtosis is not None else 3.0,
    )
    basic_gates = bool(
        int(candidate.get("paper_trades", 0)) >= 30
        and (_finite_number(candidate.get("profit_factor")) or 0) > 1.3
        and (_finite_number(candidate.get("max_drawdown_pct")) is not None)
        and float(candidate["max_drawdown_pct"]) < 5
    )
    holdout_passed = bool(holdout.get("available") and holdout.get("passed"))
    pbo = _finite_number(cpcv.get("pbo")) if isinstance(cpcv, dict) else None
    cpcv_passed = bool(
        isinstance(cpcv, dict)
        and cpcv.get("method") == "CPCV-CSCV"
        and cpcv.get("n_trials") == trials
        and int(cpcv.get("valid_splits", 0)) > 0
        and cpcv.get("pbo_passed") is True
        and pbo is not None and 0 <= pbo < 0.20
        and sharpe_result["passed"]
    )
    passed = (
        basic_gates and bool(sharpe_result["passed"])
        and cpcv_passed and holdout_passed
    )
    candidate_id = str(candidate.get("id") or "candidate")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate_id):
        raise ValueError("Candidate ID must be a plain filename identifier.")
    audit = {
        "candidate_id": candidate_id,
        "generated_at": candidate.get("generated_at"),
        "n_trials": sharpe_result["n_trials"],
        "raw_sharpe": sharpe_result["raw_sharpe"],
        "deflated_sharpe_probability": sharpe_result["adjusted_sharpe"],
        "required_sharpe": sharpe_result["required_sharpe"],
        "sharpe_gate_passed": sharpe_result["passed"],
        "pbo": pbo,
        "pbo_gate_passed": pbo is not None and 0 <= pbo < 0.20,
        "cpcv_gate_passed": cpcv_passed,
        "cpcv": candidate.get("cpcv"),
        "basic_gates_passed": basic_gates,
        "holdout": holdout,
        "expected_slippage_drag": candidate.get("expected_slippage_drag"),
        "simulated_slippage_drag": candidate.get("simulated_slippage_drag"),
        "simulated_transaction_costs": candidate.get("simulated_transaction_costs"),
        "forward_paper_expectancy_r": candidate.get("forward_paper_expectancy_r"),
        "status": "staged_candidate" if passed else "blocked",
        "reason": (
            "all statistical gates and final blind holdout passed; human approval required"
            if passed else "one or more statistical gates or final holdout validation failed"
        ),
    }
    with _candidate_audit_lock:
        try:
            prior_audit = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            prior_audit = {}
        if isinstance(prior_audit, dict) and "research_activity" in prior_audit:
            audit["research_activity"] = prior_audit["research_activity"]
        atomic_write_json(path, audit, indent=2)
        if passed:
            ensure_directory(staging_dir)
            staged = {**candidate, **audit, "human_approved": False}
            target = staging_dir / f"{candidate_id}.json"
            atomic_write_json(target, staged, indent=2)
    log.info(
        "CANDIDATE AUDIT %s status=%s trials=%s raw_sharpe=%s deflated_probability=%s holdout=%s",
        candidate_id, audit["status"], audit["n_trials"], audit["raw_sharpe"],
        audit["deflated_sharpe_probability"], holdout.get("reason"),
    )
    return audit


def approve_candidate(candidate_id: str, staging_dir: Path = STAGING_DIR,
                      active_path: Path = ACTIVE_CANDIDATE_PATH,
                      audit_path: Path = AUDIT_PATH) -> dict[str, Any]:
    """Explicitly approve a staged candidate for paper use; never enables live execution."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate_id):
        raise ValueError("Candidate ID must be a plain filename identifier.")
    path = staging_dir / f"{candidate_id}.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("status") != "staged_candidate" or candidate.get("human_approved"):
        raise ValueError("Candidate is not awaiting human approval.")
    candidate["status"] = "approved_for_paper"
    candidate["human_approved"] = True
    from datetime import datetime, timezone
    candidate["approved_at"] = datetime.now(timezone.utc).isoformat()
    ensure_directory(active_path.parent)
    atomic_write_json(active_path, candidate, indent=2)
    atomic_write_json(path, candidate, indent=2)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        audit = None
    if isinstance(audit, dict) and audit.get("candidate_id") == candidate_id:
        audit["status"] = "approved_for_paper"
        audit["human_approved"] = True
        audit["approved_at"] = candidate["approved_at"]
        atomic_write_json(audit_path, audit, indent=2)
    return candidate


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Review a staged ApexQuant paper strategy candidate.")
    parser.add_argument("--approve", metavar="CANDIDATE_ID", required=True)
    args = parser.parse_args(argv)
    print(f"Candidate {args.approve} will be approved for PAPER trading only.")
    confirmation = input(f"Type {args.approve} to confirm: ").strip()
    if confirmation != args.approve:
        print("Approval cancelled.")
        return 2
    approve_candidate(args.approve)
    print(f"Approved {args.approve} for paper-only use. Restart the runner to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def evolve_arsenal(arsenal: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply explicit statistical gates to supplied, already-recorded paper metrics."""
    messages = []
    for strategy in arsenal.get("strategies", []):
        if not isinstance(strategy, dict):
            continue
        if strategy.get("status") == "quarantined":
            continue
        outcomes = strategy.get("recent_outcomes_r", [])
        if isinstance(outcomes, list) and len(outcomes) >= 3 and all(
            (value := _finite_number(item)) is not None and value < 0 for item in outcomes[-3:]
        ):
            strategy["status"] = "quarantined"
            strategy["quarantine_reason"] = "three_consecutive_losses"
            messages.append(f"{strategy.get('name', strategy.get('id', 'strategy'))} quarantined: 3 consecutive losses")
            continue
        recent = _finite_number(strategy.get("recent_expectancy_r"))
        baseline = _finite_number(strategy.get("baseline_expectancy_r"))
        if recent is not None and baseline is not None and baseline > 0 and recent <= 0:
            strategy["status"] = "quarantined"
            strategy["quarantine_reason"] = "alpha_decay"
            messages.append(f"{strategy.get('name', strategy.get('id', 'strategy'))} quarantined: alpha decay")
            continue
        trades = int(strategy.get("paper_trades", 0) or 0)
        sharpe = _finite_number(strategy.get("sharpe"))
        profit_factor = _finite_number(strategy.get("profit_factor"))
        drawdown = _finite_number(strategy.get("max_drawdown_pct"))
        if (trades >= 30 and sharpe is not None and sharpe > 1.5
                and profit_factor is not None and profit_factor > 1.3
                and drawdown is not None and drawdown < 5):
            if strategy.get("status") != "promoted":
                strategy["status"] = "promoted"
                messages.append(f"{strategy.get('name', strategy.get('id', 'strategy'))} passed paper statistical gates")
        elif strategy.get("status") == "promoted":
            strategy["status"] = "research_only"
            messages.append(f"{strategy.get('name', strategy.get('id', 'strategy'))} no longer meets promotion gates")
    return arsenal, messages


class ArsenalEvolution:
    def __init__(self, path: Path = ARSENAL_PATH, interval_seconds: int = 3600):
        self.path = path
        self.interval_seconds = interval_seconds

    def run_once(self) -> dict[str, Any]:
        try:
            arsenal = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            arsenal = json.loads(json.dumps(SEED_ARSENAL))
        if not isinstance(arsenal, dict) or not isinstance(arsenal.get("strategies"), list):
            raise ValueError(f"Invalid strategy arsenal schema: {self.path}")
        arsenal, messages = evolve_arsenal(arsenal)
        # This persisted path is staging-only; the legacy pure helper remains compatible.
        for strategy in arsenal["strategies"]:
            if strategy.get("status") == "promoted":
                strategy["status"] = "research_only"
                strategy["promotion_requires"] = [
                    "multiple-testing-adjusted Sharpe",
                    "final blind holdout",
                    "explicit human approval",
                ]
                strategy["promotion_blocked_reason"] = (
                    "arsenal metrics lack the required candidate audit and holdout evidence"
                )
        from datetime import datetime, timezone
        arsenal["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self.path, arsenal, indent=2)
        for message in messages:
            log.warning("STATISTICAL VERDICT %s", message)
        if not messages:
            log.info("EVOLUTION LOOP checked %d strategies; no status transitions.",
                     len(arsenal["strategies"]))
        return arsenal

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception("Arsenal evaluation failed.")
                raise
            stop_event.wait(self.interval_seconds)
