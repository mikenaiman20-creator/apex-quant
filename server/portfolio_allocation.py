"""Fail-closed, research-only equal-risk budgeting for independent strategy sleeves."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Sequence


MAX_PORTFOLIO_RISK_PCT = 0.005
MAX_COST_CALIBRATION_ERROR = 0.20
DEFAULT_EQUITY_SYMBOLS = ("SPY", "QQQ", "NVDA")


def build_equal_risk_plan(
    evidence_by_sleeve: Mapping[str, Mapping[str, Any]],
    *,
    portfolio_risk_cap_pct: float = MAX_PORTFOLIO_RISK_PCT,
    symbols_by_sleeve: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Split a capped risk budget evenly; missing or failed evidence receives zero budget.

    Validation evidence must include passing CPCV+DSR, a passing locked holdout,
    and a finite relative cost-calibration error no greater than 20%.
    The result is reporting-only: it never authorizes or routes orders.
    """
    if not evidence_by_sleeve:
        raise ValueError("At least one strategy sleeve is required.")
    if (
        not math.isfinite(portfolio_risk_cap_pct)
        or portfolio_risk_cap_pct <= 0
        or portfolio_risk_cap_pct > MAX_PORTFOLIO_RISK_PCT
    ):
        raise ValueError(
            f"portfolio_risk_cap_pct must be in (0, {MAX_PORTFOLIO_RISK_PCT}]."
        )
    if any(not isinstance(name, str) or not name.strip() for name in evidence_by_sleeve):
        raise ValueError("Sleeve names must be non-empty strings.")

    equal_weight = 1.0 / len(evidence_by_sleeve)
    per_sleeve_cap = portfolio_risk_cap_pct * equal_weight
    symbol_sets = (
        {name: DEFAULT_EQUITY_SYMBOLS for name in evidence_by_sleeve}
        if symbols_by_sleeve is None else symbols_by_sleeve
    )
    if not isinstance(symbol_sets, Mapping) or set(symbol_sets) != set(evidence_by_sleeve):
        raise ValueError("symbols_by_sleeve must specify exactly the configured sleeves.")
    sleeves: dict[str, Any] = {}
    for name, evidence in evidence_by_sleeve.items():
        if not isinstance(evidence, Mapping):
            raise TypeError(f"Validation evidence for {name} must be a mapping.")
        symbols = symbol_sets[name]
        if (
            isinstance(symbols, str)
            or not symbols
            or any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError(f"Symbols for {name} must be unique non-empty strings.")

        cpcv = evidence.get("cpcv")
        holdout = evidence.get("holdout")
        cost_error = evidence.get("cost_calibration_error")
        expectancy_r = evidence.get("oos_expectancy_r")
        oos_net_pnl = evidence.get("oos_net_pnl")
        reasons = []
        dsr = cpcv.get("dsr") if isinstance(cpcv, Mapping) else None
        pbo = cpcv.get("pbo") if isinstance(cpcv, Mapping) else None
        if (
            not isinstance(cpcv, Mapping)
            or cpcv.get("passed") is not True
            or not isinstance(pbo, (int, float))
            or isinstance(pbo, bool)
            or not math.isfinite(pbo)
            or pbo >= 0.20
            or not isinstance(dsr, Mapping)
            or dsr.get("passed") is not True
            or not isinstance(dsr.get("adjusted_sharpe"), (int, float))
            or isinstance(dsr.get("adjusted_sharpe"), bool)
            or not math.isfinite(dsr["adjusted_sharpe"])
            or dsr["adjusted_sharpe"] < 0.95
        ):
            reasons.append("CPCV/DSR gates have not passed")
        if (
            not isinstance(expectancy_r, (int, float))
            or isinstance(expectancy_r, bool)
            or not math.isfinite(expectancy_r)
            or expectancy_r <= 0
            or not isinstance(oos_net_pnl, (int, float))
            or isinstance(oos_net_pnl, bool)
            or not math.isfinite(oos_net_pnl)
            or oos_net_pnl <= 0
        ):
            reasons.append("walk-forward expectancy after costs is not positive")
        if not isinstance(holdout, Mapping) or holdout.get("passed") is not True:
            reasons.append("locked holdout gate has not passed")
        if (
            not isinstance(cost_error, (int, float))
            or isinstance(cost_error, bool)
            or not math.isfinite(cost_error)
            or cost_error < 0
            or cost_error > MAX_COST_CALIBRATION_ERROR
        ):
            reasons.append("cost calibration evidence is missing or exceeds 20%")

        eligible = not reasons
        per_symbol_cap = per_sleeve_cap / len(symbols)
        sleeves[name] = {
            "equal_weight": equal_weight,
            "proposed_risk_cap_pct": per_sleeve_cap,
            "validation_eligible": eligible,
            "research_risk_budget_pct": per_sleeve_cap if eligible else 0.0,
            "execution_risk_pct": 0.0,
            "reasons": reasons,
            "assets": {
                symbol: {
                    "equal_weight": 1.0 / len(symbols),
                    "proposed_risk_cap_pct": per_symbol_cap,
                    "research_risk_budget_pct": per_symbol_cap if eligible else 0.0,
                    "execution_risk_pct": 0.0,
                }
                for symbol in symbols
            },
        }

    return {
        "version": 1,
        "mode": "research_only",
        "portfolio_risk_cap_pct": portfolio_risk_cap_pct,
        "dynamic_weights_enabled": False,
        "order_routing_enabled": False,
        "sleeves": sleeves,
        "approved_research_risk_pct": sum(
            sleeve["research_risk_budget_pct"] for sleeve in sleeves.values()
        ),
        "execution_risk_pct": 0.0,
    }
