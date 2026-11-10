"""Deterministic synthetic panel used by the Phase 1 gate-calibration tests.

The panel deliberately mirrors the production research problem while keeping the
gate itself untouched:

* 500 symbols, 15 years of daily observations (3,780 return observations).
* Cross-sectional 5-day reversal signal with a target information coefficient.
* Student-t(5) innovations with mild AR(1) autocorrelation.
* Dollar-neutral long/short quintile portfolio with turnover-aware costs using
  the production ``BTConfig`` cost parameters.
* A calibrated zero-alpha overlay so the *asset-level* IC can be 0.15 while the
  *portfolio-level* after-cost Sharpe is the requested 0.50.  A fully invested
  500-name portfolio with IC 0.15 would have a far higher Sharpe; the overlay is
  the same "plant the signal and the portfolio Sharpe independently" device the
  existing ``generate_synthetic_calibration_signal`` uses.

The gate under test is the real ``backtest.cpcv.evaluate_cpcv`` (CPCV splits,
purging, embargo, PBO, DSR).  Only the intraday fill engine is bypassed: this is
a daily cross-sectional strategy, and ``run_backtest`` is replaced with the
pre-costed daily trade records produced here.

Why 15 years rather than the 3 years in the Phase 1 brief: with 756 daily
observations a true annualized Sharpe of 0.50 yields DSR 0.806 under the
production formula.  Roughly 2,773 observations (~11 years) are required to
reach DSR 0.95 at Sharpe 0.50.  The panel uses 15 years (3,780 observations) for
margin.  The PBO < 0.20 and DSR >= 0.95 thresholds are unchanged.  See
``DECISIONS.md`` ("Gate Calibration Baseline").
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from operator import itemgetter
from types import SimpleNamespace
from typing import Any, Sequence

import backtest.cpcv as cpcv
from backtest.cpcv import evaluate_cpcv
from backtest.data import NY, MinuteBar
from backtest.engine import BTConfig

SYMBOLS = 500
YEARS = 15
TRADING_DAYS_PER_YEAR = 252
AUTOCORRELATION = 0.02
DEGREES_OF_FREEDOM = 5
QUANTILE = 0.20
TARGET_SHARPE = 0.50
LOOKBACKS = (5, 10)
LOOKBACK_BY_SCORE = {0.65: 5, 0.75: 10}
DEFAULT_SEED = 20261110
BOUNDARY_ICS = (0.00, 0.05, 0.10, 0.15, 0.20)
BOUNDARY_SEEDS = tuple(range(20261110, 20261130))
GROUP_COUNT = 6
EMBARGO_DAYS = 1
N_TRIALS = 1


@dataclass(frozen=True)
class SeedPanels:
    seed: int
    observations: int
    risk_budget: float
    overlay: tuple[float, ...]
    panels: dict[float, dict[int, tuple[float, ...]]]
    realized_ic: dict[float, float]


@dataclass(frozen=True)
class GateResult:
    pbo: float
    dsr_probability: float
    raw_sharpe: float
    net_sharpe: float
    passed: bool
    observations: int
    realized_ic: float
    report: dict[str, Any]


def _build_t_table(size: int = 20_000, df: int = DEGREES_OF_FREEDOM, seed: int = 12345) -> tuple[float, ...]:
    """Standardized Student-t quantiles, built once and sampled by index."""
    rng = random.Random(seed)
    values = []
    for _ in range(size):
        z = rng.gauss(0.0, 1.0)
        chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))
        values.append(z / math.sqrt(chi2 / df))
    values.sort()
    mean = sum(values) / size
    variance = sum((value - mean) ** 2 for value in values) / (size - 1)
    std = math.sqrt(variance)
    return tuple((value - mean) / std for value in values)


_T_TABLE = _build_t_table()


def _cost_bps_per_side() -> float:
    """One-way cost approximation using the production cost parameters.

    The engine charges slippage plus half the quoted spread on every fill and a
    sell-side regulatory fee.  A long/short portfolio sells roughly half of its
    turnover, so half the sell fee is applied to each side.
    """
    config = BTConfig()
    return config.slippage_bps + config.spread_bps / 2.0 + config.sell_fee_bps / 2.0


def _generate_noise(symbols: int, days: int, rho: float, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    table_size = len(_T_TABLE)
    scale = math.sqrt(1.0 - rho * rho)
    noise: list[list[float]] = [[0.0] * symbols for _ in range(days)]
    previous = [0.0] * symbols
    for day in range(days):
        row = noise[day]
        for index in range(symbols):
            previous[index] = rho * previous[index] + scale * _T_TABLE[rng.randrange(table_size)]
            row[index] = previous[index]
    return noise


def _rolling_signal(noise: Sequence[Sequence[float]], lookback: int) -> list[list[float]]:
    """Negative rolling ``lookback`` sum of past noise, standardized by day."""
    days = len(noise)
    symbols = len(noise[0])
    signal: list[list[float]] = [[0.0] * symbols for _ in range(days)]
    rolling = [0.0] * symbols
    for day in range(1, days):
        current = noise[day - 1]
        for index in range(symbols):
            rolling[index] += current[index]
        stale_day = day - 1 - lookback
        if stale_day >= 0:
            stale = noise[stale_day]
            for index in range(symbols):
                rolling[index] -= stale[index]
        if day < lookback:
            continue
        total = sum(rolling)
        total_square = sum(value * value for value in rolling)
        variance = (total_square - total * total / symbols) / (symbols - 1)
        std = math.sqrt(variance) if variance > 0 else 1.0
        mean = total / symbols
        row = signal[day]
        for index in range(symbols):
            row[index] = -(rolling[index] - mean) / std
    return signal


def _ranking_getters(signal: Sequence[Sequence[float]], start: int, quantile: float = QUANTILE):
    days = len(signal)
    symbols = len(signal[0])
    size = int(symbols * quantile)
    if size < 2:
        raise ValueError("Calibration portfolio needs at least two names per side.")
    getters = []
    for day in range(start, days):
        row = signal[day]
        order = sorted(range(symbols), key=row.__getitem__)
        top = order[-size:]
        bottom = order[:size]
        getters.append((itemgetter(*top), itemgetter(*bottom), set(top), set(bottom), size))
    return getters


def _asset_returns(
    noise: Sequence[Sequence[float]],
    signal: Sequence[Sequence[float]],
    information_coefficient: float,
    volatility: float = 0.01,
) -> list[list[float]]:
    days = len(noise)
    symbols = len(noise[0])
    root = math.sqrt(1.0 - information_coefficient * information_coefficient)
    returns: list[list[float]] = [[0.0] * symbols for _ in range(days)]
    for day in range(DEGREES_OF_FREEDOM, days):
        signal_row = signal[day]
        noise_row = noise[day]
        row = returns[day]
        for index in range(symbols):
            row[index] = volatility * (
                information_coefficient * signal_row[index] + root * noise_row[index]
            )
    return returns


def _portfolio_returns(
    getters: Sequence[tuple[Any, Any, set[int], set[int], int]],
    returns: Sequence[Sequence[float]],
    start: int,
    cost_bps: float,
) -> list[float]:
    cost_rate = cost_bps / 1e4
    output = []
    previous_long: set[int] = set()
    previous_short: set[int] = set()
    for offset, (top_get, bottom_get, long_side, short_side, size) in enumerate(getters):
        row = returns[start + offset]
        gross = sum(top_get(row)) / size - sum(bottom_get(row)) / size
        absolute_change = (
            len(long_side - previous_long)
            + len(previous_long - long_side)
            + len(short_side - previous_short)
            + len(previous_short - short_side)
            + 2 * len(long_side & previous_short)
            + 2 * len(short_side & previous_long)
        )
        turnover = absolute_change / (2.0 * size)
        output.append(gross - turnover * cost_rate)
        previous_long, previous_short = long_side, short_side
    return output


def _overlay(noise: Sequence[Sequence[float]], start: int, target_std: float) -> tuple[float, ...]:
    """Diversified zero-alpha basket: cross-sectional mean of the same t(5) noise."""
    symbols = len(noise[0])
    values = [sum(noise[day]) / symbols for day in range(start, len(noise))]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return tuple(((value - mean) / std) * target_std for value in values)


def _sharpe(returns: Sequence[float], annualization: float = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2:
        return float("-inf")
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    if not math.isfinite(variance) or variance <= 0:
        return float("-inf")
    return mean / math.sqrt(variance) * math.sqrt(annualization)


def _calibrate_risk_budget(
    signal_returns: Sequence[float],
    overlay: Sequence[float],
    target_sharpe: float,
) -> float:
    def sharpe(weight: float) -> float:
        total = [
            weight * signal + (1.0 - weight) * noise
            for signal, noise in zip(signal_returns, overlay)
        ]
        return _sharpe(total)

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if sharpe(middle) < target_sharpe:
            low = middle
        else:
            high = middle
    weight = 0.5 * (low + high)
    if not 0.0 < weight <= 1.0:
        raise ValueError("Risk-budget calibration did not converge inside (0, 1].")
    return weight


def _realized_ic(
    signal: Sequence[Sequence[float]],
    returns: Sequence[Sequence[float]],
    start: int,
) -> float:
    symbols = len(signal[0])
    correlations = []
    for day in range(start, len(signal)):
        signal_row = signal[day]
        return_row = returns[day]
        signal_mean = sum(signal_row) / symbols
        return_mean = sum(return_row) / symbols
        covariance = sum(
            (left - signal_mean) * (right - return_mean)
            for left, right in zip(signal_row, return_row)
        ) / (symbols - 1)
        signal_variance = sum((value - signal_mean) ** 2 for value in signal_row) / (symbols - 1)
        return_variance = sum((value - return_mean) ** 2 for value in return_row) / (symbols - 1)
        if signal_variance > 0 and return_variance > 0:
            correlations.append(covariance / math.sqrt(signal_variance * return_variance))
    return sum(correlations) / len(correlations)


def build_seed(
    seed: int,
    information_coefficients: Sequence[float],
    *,
    symbols: int = SYMBOLS,
    years: int = YEARS,
    autocorrelation: float = AUTOCORRELATION,
    target_sharpe: float = TARGET_SHARPE,
    compute_realized_ic: bool = False,
) -> SeedPanels:
    """Build the panel and calibrate the risk budget once per seed.

    The risk budget is always calibrated on the IC=0.15 panel so every IC in a
    sweep is evaluated with the same portfolio sizing.
    """
    start = 10
    days = years * TRADING_DAYS_PER_YEAR + start
    noise = _generate_noise(symbols, days, autocorrelation, seed)
    true_signal = _rolling_signal(noise, 5)
    getters = {lookback: _ranking_getters(_rolling_signal(noise, lookback), start) for lookback in LOOKBACKS}
    cost_bps = _cost_bps_per_side()

    def panel_for(information_coefficient: float) -> dict[int, tuple[float, ...]]:
        returns = _asset_returns(noise, true_signal, information_coefficient)
        return {
            lookback: tuple(_portfolio_returns(getters[lookback], returns, start, cost_bps))
            for lookback in LOOKBACKS
        }

    reference = panel_for(0.15)
    reference_std = math.sqrt(
        sum((value - sum(reference[5]) / len(reference[5])) ** 2 for value in reference[5])
        / (len(reference[5]) - 1)
    )
    overlay = _overlay(noise, start, reference_std)
    risk_budget = _calibrate_risk_budget(reference[5], overlay, target_sharpe)

    panels: dict[float, dict[int, tuple[float, ...]]] = {}
    realized: dict[float, float] = {}
    for information_coefficient in information_coefficients:
        panels[information_coefficient] = (
            reference if information_coefficient == 0.15 else panel_for(information_coefficient)
        )
        if compute_realized_ic:
            returns = _asset_returns(noise, true_signal, information_coefficient)
            realized[information_coefficient] = _realized_ic(true_signal, returns, start)

    return SeedPanels(
        seed=seed,
        observations=days - start,
        risk_budget=risk_budget,
        overlay=overlay,
        panels=panels,
        realized_ic=realized,
    )


def _daily_bars(count: int) -> list[MinuteBar]:
    bars = []
    current = date(1990, 1, 1)
    while len(bars) < count:
        if current.weekday() < 5:
            start = datetime.combine(current, time(9, 30), tzinfo=NY)
            bars.append(MinuteBar(start, 100.0, 100.01, 99.99, 100.0, 1000.0))
        current += timedelta(days=1)
    return bars


def _trade_series(
    signal_returns: Sequence[float],
    overlay: Sequence[float],
    risk_budget: float,
    bars: Sequence[MinuteBar],
    initial_equity: float,
):
    total = [
        risk_budget * signal + (1.0 - risk_budget) * noise
        for signal, noise in zip(signal_returns, overlay)
    ]
    return [
        SimpleNamespace(
            entry_ts=bar.ts,
            exit_ts=bar.ts + timedelta(minutes=1),
            pnl_net=value * initial_equity,
        )
        for bar, value in zip(bars, total)
    ]


def evaluate_gate(
    seed_panels: SeedPanels,
    information_coefficient: float,
    monkeypatch: Any,
    *,
    n_trials: int = N_TRIALS,
    group_count: int = GROUP_COUNT,
    embargo_days: int = EMBARGO_DAYS,
) -> GateResult:
    """Run the real CPCV/PBO/DSR gate on one panel."""
    panel = seed_panels.panels[information_coefficient]
    bars = _daily_bars(seed_panels.observations)
    base = BTConfig(initial_equity=5000.0, warmup_days=0)
    candidates = {
        (score,): _trade_series(
            panel[lookback],
            seed_panels.overlay,
            seed_panels.risk_budget,
            bars,
            base.initial_equity,
        )
        for score, lookback in LOOKBACK_BY_SCORE.items()
    }

    def replay(_data, config, *, signal_fn=None, reset_equity_days):
        return SimpleNamespace(trades=candidates[(config.min_signal_score,)])

    monkeypatch.setattr(cpcv, "run_backtest", replay)
    report = evaluate_cpcv(
        {"SYNTH": bars},
        base,
        candidates,
        n_trials=n_trials,
        grid={"min_signal_score": sorted(LOOKBACK_BY_SCORE)},
        group_count=group_count,
        embargo_days=embargo_days,
        workers=1,
    )

    pbo_raw = report.get("pbo")
    pbo = float("inf") if pbo_raw is None else float(pbo_raw)
    dsr = report["dsr"]
    net_returns = [
        seed_panels.risk_budget * signal + (1.0 - seed_panels.risk_budget) * noise
        for signal, noise in zip(panel[5], seed_panels.overlay)
    ]
    return GateResult(
        pbo=pbo,
        dsr_probability=float(dsr.get("adjusted_sharpe") or 0.0),
        raw_sharpe=float(dsr.get("raw_sharpe") or 0.0),
        net_sharpe=_sharpe(net_returns),
        passed=bool(report["passed"]),
        observations=int(report["oos_observations"]),
        realized_ic=seed_panels.realized_ic.get(information_coefficient, float("nan")),
        report=report,
    )
