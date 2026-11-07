"""Statistical strategy promotion and quarantine; this module never routes orders."""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from datetime import datetime, timezone
from statistics import NormalDist
from pathlib import Path
from typing import Any

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
        return {"version": 1, "cycles": {}, "total_trials": 0}
    raw_cycles = value.get("cycles")
    if raw_cycles is not None and not isinstance(raw_cycles, dict):
        return {"version": 1, "cycles": {}, "total_trials": 0}
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
    legacy_total = value.get("total_trials")
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


def load_trial_ledger(path: Path = TRIAL_LEDGER_PATH) -> dict[str, Any]:
    """Load and repair the local ledger under the same lock used by its writers."""
    with _trial_ledger_lock:
        exists = True
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            exists = False
            raw = None
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Invalid strategy trial ledger; reinitializing %s.", path)
            raw = None
        ledger = normalize_trial_ledger(raw)
        if exists and raw != ledger:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
            temp.replace(path)
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
    """Persist unique discovery-cycle trial counts and return the cumulative count."""
    if not cycle_id or trial_count < 0:
        raise ValueError("A cycle ID and non-negative trial count are required.")
    with _trial_ledger_lock:
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            ledger = None
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Invalid strategy trial ledger JSON; reinitializing %s.", path)
            ledger = None
        ledger = normalize_trial_ledger(ledger)
        cycles = ledger["cycles"]
        if cycle_id not in cycles:
            cycles[cycle_id] = trial_count
            ledger["total_trials"] = sum(cycles.values())
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
                ledger["recent_win_rates"] = (samples + valid_records)[-142:]
        ledger["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        ledger["latest_cycle_id"] = cycle_id
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temp.replace(path)
        return int(ledger.get("total_trials", sum(int(value) for value in cycles.values())))


def update_trial_activity(cycle_id: str, trial_count: int, status: str, message: str,
                          path: Path = TRIAL_LEDGER_PATH) -> dict[str, Any]:
    """Publish the research worker's current task alongside cumulative trial counts."""
    with _trial_ledger_lock:
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            ledger = None
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("Invalid strategy trial ledger JSON; reinitializing %s.", path)
            ledger = None
        ledger = normalize_trial_ledger(ledger)
        cycles = ledger["cycles"]
        if cycle_id and cycle_id not in cycles and status == "completed":
            cycles[cycle_id] = trial_count
        ledger["total_trials"] = sum(cycles.values())
        ledger["latest_activity"] = {
            "cycle_id": cycle_id,
            "status": status,
            "trial_count": trial_count,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temp.replace(path)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)
        return activity


def deflated_sharpe(sharpe: float | None, observations: int,
                    n_trials: int) -> dict[str, float | int | bool | None]:
    """Approximate a multiple-testing-adjusted Sharpe significance hurdle."""
    value = _finite_number(sharpe)
    trials = max(int(n_trials), 1)
    n_obs = max(int(observations), 1)
    if value is None or n_obs < 2:
        return {
            "raw_sharpe": value, "adjusted_sharpe": None,
            "required_sharpe": None, "n_trials": trials,
            "observations": n_obs, "passed": False,
        }
    standard_error = math.sqrt(252.0 / n_obs)
    expected_max = (
        NormalDist().inv_cdf(0.5 ** (1.0 / trials)) * standard_error
        if trials > 1 else 0.0
    )
    required = max(BASE_SHARPE_HURDLE, expected_max + NormalDist().inv_cdf(0.95) * standard_error)
    adjusted = NormalDist().cdf((value - expected_max) / standard_error)
    return {
        "raw_sharpe": value, "adjusted_sharpe": adjusted,
        "required_sharpe": required, "n_trials": trials,
        "observations": n_obs,
        "passed": bool(value >= required and adjusted >= 0.95),
    }


def stage_candidate(candidate: dict[str, Any], holdout: dict[str, Any],
                    path: Path = AUDIT_PATH, staging_dir: Path = STAGING_DIR) -> dict[str, Any]:
    """Write an audit record and stage only a statistically and blindly validated candidate."""
    sharpe_result = deflated_sharpe(
        _finite_number(candidate.get("sharpe")),
        int(candidate.get("observations", 0)),
        int(candidate.get("n_trials", 1)),
    )
    basic_gates = bool(
        int(candidate.get("paper_trades", 0)) >= 30
        and (_finite_number(candidate.get("profit_factor")) or 0) > 1.3
        and (_finite_number(candidate.get("max_drawdown_pct")) is not None)
        and float(candidate["max_drawdown_pct"]) < 5
    )
    holdout_passed = bool(holdout.get("available") and holdout.get("passed"))
    passed = basic_gates and bool(sharpe_result["passed"]) and holdout_passed
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
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)
        if passed:
            staging_dir.mkdir(parents=True, exist_ok=True)
            staged = {**candidate, **audit, "human_approved": False}
            target = staging_dir / f"{candidate_id}.json"
            staged_temp = target.with_suffix(".json.tmp")
            staged_temp.write_text(json.dumps(staged, indent=2, allow_nan=False), encoding="utf-8")
            staged_temp.replace(target)
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
    active_path.parent.mkdir(parents=True, exist_ok=True)
    temp = active_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(candidate, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(active_path)
    staged_temp = path.with_suffix(".json.tmp")
    staged_temp.write_text(json.dumps(candidate, indent=2, allow_nan=False), encoding="utf-8")
    staged_temp.replace(path)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        audit = None
    if isinstance(audit, dict) and audit.get("candidate_id") == candidate_id:
        audit["status"] = "approved_for_paper"
        audit["human_approved"] = True
        audit["approved_at"] = candidate["approved_at"]
        audit_temp = audit_path.with_suffix(".json.tmp")
        audit_temp.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        audit_temp.replace(audit_path)
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(arsenal, indent=2), encoding="utf-8")
        temp_path.replace(self.path)
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
