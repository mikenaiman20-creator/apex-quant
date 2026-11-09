import pytest

from server.portfolio_allocation import build_equal_risk_plan


def _passing_evidence(cost_error=0.1):
    return {
        "cpcv": {
            "passed": True,
            "pbo": 0.1,
            "dsr": {"passed": True, "adjusted_sharpe": 0.97},
        },
        "holdout": {"passed": True},
        "cost_calibration_error": cost_error,
        "oos_expectancy_r": 0.1,
        "oos_net_pnl": 100.0,
    }


def test_equal_risk_budgets_are_capped_and_research_only():
    plan = build_equal_risk_plan({
        "equity_primary": _passing_evidence(),
        "equity_orb": _passing_evidence(),
    })

    assert plan["portfolio_risk_cap_pct"] == pytest.approx(0.005)
    assert plan["approved_research_risk_pct"] == pytest.approx(0.005)
    assert plan["execution_risk_pct"] == 0
    assert plan["order_routing_enabled"] is False
    assert plan["dynamic_weights_enabled"] is False
    assert sorted(
        sleeve["research_risk_budget_pct"]
        for sleeve in plan["sleeves"].values()
    ) == pytest.approx([0.0025, 0.0025])
    assert set(plan["sleeves"]["equity_orb"]["assets"]) == {"SPY", "QQQ", "NVDA"}
    assert plan["sleeves"]["equity_orb"]["assets"]["SPY"][
        "research_risk_budget_pct"
    ] == pytest.approx(0.0025 / 3)


def test_missing_or_failed_validation_receives_no_risk_budget():
    plan = build_equal_risk_plan({
        "equity_primary": _passing_evidence(),
        "equity_orb": {
            "cpcv": {
                "passed": False,
                "pbo": 0.4,
                "dsr": {"passed": False, "adjusted_sharpe": 0.0},
            },
            "holdout": {"passed": True},
            "cost_calibration_error": 0.1,
            "oos_expectancy_r": 0.1,
            "oos_net_pnl": 100.0,
        },
    })

    assert plan["approved_research_risk_pct"] == pytest.approx(0.0025)
    assert plan["sleeves"]["equity_orb"]["research_risk_budget_pct"] == 0
    assert plan["sleeves"]["equity_orb"]["reasons"] == [
        "CPCV/DSR gates have not passed",
    ]


@pytest.mark.parametrize("error", [0.20001, -0.01, float("nan"), None])
def test_cost_calibration_must_be_present_finite_and_within_limit(error):
    plan = build_equal_risk_plan({
        "equity_orb": _passing_evidence(cost_error=error),
    })

    assert plan["sleeves"]["equity_orb"]["validation_eligible"] is False
    assert plan["sleeves"]["equity_orb"]["research_risk_budget_pct"] == 0


@pytest.mark.parametrize(
    ("expectancy", "net_pnl"),
    [(0.0, 10.0), (-0.1, 10.0), (0.1, 0.0), (0.1, -1.0), (None, None)],
)
def test_nonpositive_after_cost_oos_expectancy_gets_zero_budget(expectancy, net_pnl):
    evidence = _passing_evidence()
    evidence["oos_expectancy_r"] = expectancy
    evidence["oos_net_pnl"] = net_pnl
    plan = build_equal_risk_plan({"zscore_mean_reversion": evidence})

    assert plan["sleeves"]["zscore_mean_reversion"]["research_risk_budget_pct"] == 0


def test_portfolio_budget_cannot_exceed_risk_engine_cap():
    with pytest.raises(ValueError, match="must be in"):
        build_equal_risk_plan(
            {"equity_orb": _passing_evidence()},
            portfolio_risk_cap_pct=0.01,
        )
