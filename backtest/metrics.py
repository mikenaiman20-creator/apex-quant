"""Performance statistics, with a bootstrap confidence interval on mean R (the number that matters)."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

from backtest.data import NY


def deflated_sharpe_probability(
    annualized_sharpe: float | None,
    observations: int,
    n_trials: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization: float = 252.0,
    minimum_sharpe: float = 0.0,
) -> Dict[str, float | int | bool | None]:
    """Compute DSR significance against the trial-adjusted null.

    The default has no absolute Sharpe hurdle; promotion policies may impose a
    separate minimum after the statistical DSR gate.
    """
    trials = int(n_trials)
    count = int(observations)
    empty = {
        "raw_sharpe": None,
        "adjusted_sharpe": None,
        "required_sharpe": None,
        "n_trials": trials,
        "observations": count,
        "skewness": None,
        "kurtosis": None,
        "passed": False,
    }
    if (
        trials < 1 or count < 3 or annualized_sharpe is None
        or not math.isfinite(annualized_sharpe)
        or not math.isfinite(skewness) or not math.isfinite(kurtosis)
        or not math.isfinite(annualization) or annualization <= 0
        or not math.isfinite(minimum_sharpe) or minimum_sharpe < 0
        or kurtosis < 1
    ):
        return empty

    sharpe = annualized_sharpe / math.sqrt(annualization)
    variance = (
        1.0 + 0.5 * sharpe ** 2 - skewness * sharpe
        + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    ) / (count - 1)
    if not math.isfinite(variance) or variance <= 0:
        return empty
    standard_error = math.sqrt(variance)
    if trials == 1:
        expected_max = 0.0
    else:
        normal = NormalDist()
        gamma = 0.5772156649015329
        expected_max = standard_error * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trials)
            + gamma * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
        )
    probability = NormalDist().cdf((sharpe - expected_max) / standard_error)
    required = max(
        minimum_sharpe / math.sqrt(annualization),
        expected_max + NormalDist().inv_cdf(0.95) * standard_error,
    )
    return {
        "raw_sharpe": annualized_sharpe,
        "adjusted_sharpe": probability,
        "required_sharpe": required * math.sqrt(annualization),
        "n_trials": trials,
        "observations": count,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "passed": bool(annualized_sharpe >= required * math.sqrt(annualization)
                       and probability >= 0.95),
    }


def deflated_sharpe_from_returns(
    returns: Sequence[float],
    n_trials: int,
    *,
    annualization: float = 252.0,
    minimum_sharpe: float = 0.0,
) -> Dict[str, float | int | bool | None]:
    """Calculate DSR from a daily return series, including its skew and kurtosis."""
    count = len(returns)
    trials = int(n_trials)
    if (
        trials < 1 or count < 3
        or not all(math.isfinite(value) for value in returns)
    ):
        return deflated_sharpe_probability(
            None, count, trials, annualization=annualization,
            minimum_sharpe=minimum_sharpe,
        )
    mean = sum(returns) / count
    variance = sum((value - mean) ** 2 for value in returns) / (count - 1)
    if not math.isfinite(variance) or variance <= 0:
        return deflated_sharpe_probability(
            None, count, trials, annualization=annualization,
            minimum_sharpe=minimum_sharpe,
        )
    deviation = math.sqrt(variance)
    skewness = sum((value - mean) ** 3 for value in returns) / count / deviation ** 3
    kurtosis = sum((value - mean) ** 4 for value in returns) / count / deviation ** 4
    sharpe = mean / deviation * math.sqrt(annualization)
    return deflated_sharpe_probability(
        sharpe, count, trials, skewness=skewness, kurtosis=kurtosis,
        annualization=annualization, minimum_sharpe=minimum_sharpe,
    )


def bootstrap_ci(xs: Sequence[float], n_boot: int = 4000, seed: int = 1, alpha: float = 0.05) -> Tuple[float, float]:
    m = len(xs)
    if m < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def trade_equity_curve(trades, initial_equity: float, calendar_days: Sequence | None = None):
    """Build a realized, end-of-day equity curve when replay marks are unavailable."""
    daily_pnl = defaultdict(float)
    for trade in trades:
        daily_pnl[trade.exit_ts.astimezone(NY).date()] += trade.pnl_net
    equity = initial_equity
    curve = []
    days = sorted(set(calendar_days)) if calendar_days is not None else sorted(daily_pnl)
    for day in days:
        pnl = daily_pnl.get(day, 0.0)
        equity += pnl
        curve.append((day, equity))
    return curve


def summarize(trades, equity_curve, initial_equity: float) -> Dict[str, float]:
    n = len(trades)
    rs = [t.r_multiple for t in trades]
    gross = sum(t.pnl_gross for t in trades)
    costs = sum(t.costs for t in trades)
    net = sum(t.pnl_net for t in trades)
    gp = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gl = -sum(t.pnl_net for t in trades if t.pnl_net < 0)

    eq = [initial_equity] + [e for _, e in equity_curve]
    peak, mdd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak if peak > 0 else 0.0)
    cagr = float("nan")
    if equity_curve and initial_equity > 0 and eq[-1] > 0:
        elapsed_days = (equity_curve[-1][0] - equity_curve[0][0]).days
        if elapsed_days > 0:
            cagr = 100 * ((eq[-1] / initial_equity) ** (365.25 / elapsed_days) - 1)
    rets = [(b - a) / a for a, b in zip(eq[:-1], eq[1:]) if a > 0]
    sharpe = float("nan")
    if len(rets) > 2:
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
        sharpe = mu / sd * math.sqrt(252) if sd > 0 else float("nan")

    mean_r = sum(rs) / n if n else float("nan")
    sd_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / (n - 1)) if n > 1 else float("nan")
    t_stat = mean_r / (sd_r / math.sqrt(n)) if n > 1 and sd_r > 0 else float("nan")
    lo, hi = bootstrap_ci(rs)
    holds = [(t.exit_ts - t.entry_ts).total_seconds() / 60 for t in trades]
    return {
        "n": n,
        "win_rate": sum(1 for t in trades if t.pnl_net > 0) / n if n else float("nan"),
        "mean_r": mean_r,
        "mean_r_ci_lo": lo,
        "mean_r_ci_hi": hi,
        "t_stat": t_stat,
        "profit_factor": gp / gl if gl > 0 else float("inf") if gp > 0 else float("nan"),
        "gross_pnl": gross,
        "costs": costs,
        "slippage_cost": sum(getattr(t, "slippage_cost", 0.0) for t in trades),
        "commission_cost": sum(getattr(t, "commission_cost", 0.0) for t in trades),
        "regulatory_fees": sum(getattr(t, "regulatory_fees", 0.0) for t in trades),
        "net_pnl": net,
        "return_pct": 100 * net / initial_equity,
        "max_dd_pct": 100 * mdd,
        "cagr_pct": cagr,
        "sharpe_daily_ann": sharpe,
        "avg_hold_min": sum(holds) / n if n else float("nan"),
    }


def verdict(m: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    n = int(m["n"])
    if n == 0:
        return ["NO TRADES: the strategy never fired. Check the check-pass table below."]
    if n < 100:
        notes.append(f"Only {n} trades: statistically meaningless (want 100 minimum, 300+ to trust).")
    lo, hi = m["mean_r_ci_lo"], m["mean_r_ci_hi"]
    if lo > 0:
        notes.append("95% bootstrap CI of mean R is above 0. Necessary, not sufficient: verify out-of-sample.")
    elif hi < 0:
        notes.append("95% bootstrap CI of mean R is below 0: loses money after costs.")
    else:
        notes.append("95% bootstrap CI of mean R straddles 0: no demonstrated edge.")
    if m["gross_pnl"] > 0 and m["costs"] > 0.5 * m["gross_pnl"]:
        notes.append(f"Costs eat {100 * m['costs'] / m['gross_pnl']:.0f}% of gross profit: fragile to execution.")
    if m["gross_pnl"] <= 0 < m["costs"]:
        notes.append("Gross P&L is already <= 0 before costs: no signal to protect.")
    return notes


def fmt(m: Dict[str, float]) -> str:
    cagr = f"{m['cagr_pct']:+.2f}%" if math.isfinite(m["cagr_pct"]) else "n/a"
    return (
        f"trades={int(m['n'])}  win={m['win_rate']:.1%}  expectancy={m['mean_r']:+.3f}R "
        f"(95% CI {m['mean_r_ci_lo']:+.3f}..{m['mean_r_ci_hi']:+.3f}, t={m['t_stat']:+.2f})\n"
        f"net=${m['net_pnl']:+,.2f} ({m['return_pct']:+.2f}%)  gross=${m['gross_pnl']:+,.2f}  costs=${m['costs']:,.2f}  "
        f"(slippage=${m['slippage_cost']:,.2f}, commission=${m['commission_cost']:,.2f}, fees=${m['regulatory_fees']:,.2f})  "
        f"PF={m['profit_factor']:.2f}  maxDD={m['max_dd_pct']:.2f}%  CAGR={cagr}  "
        f"Sharpe={m['sharpe_daily_ann']:.2f}  "
        f"avgHold={m['avg_hold_min']:.0f}m"
    )
