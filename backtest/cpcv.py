"""Combinatorial purged cross-validation with embargo and PBO/DSR reporting."""
from __future__ import annotations

import itertools
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from backtest.data import MinuteBar, NY
from backtest.engine import BTConfig, SignalFn, TradeRecord, run_backtest
from backtest.metrics import deflated_sharpe_from_returns
from backtest.walkforward import DEFAULT_GRID, apply_params


@dataclass(frozen=True)
class SyntheticSignalCalibration:
    signal: tuple[float, ...]
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    transaction_cost_per_observation: float
    information_coefficient: float
    annualized_sharpe: float


def generate_synthetic_calibration_signal(
    observations: int = 30_000,
    information_coefficient: float = 0.15,
    annualized_sharpe: float = 0.50,
    *,
    daily_volatility: float = 0.01,
    transaction_cost_per_observation: float = 0.0002,
    seed: int = 20261109,
) -> SyntheticSignalCalibration:
    """Generate a reproducible signal/return pair with controlled net Sharpe and IC."""
    if observations < 3:
        raise ValueError("Synthetic calibration requires at least three observations.")
    if not math.isfinite(information_coefficient) or not 0 <= information_coefficient < 1:
        raise ValueError("information_coefficient must be finite and in [0, 1).")
    if not math.isfinite(annualized_sharpe) or annualized_sharpe < 0:
        raise ValueError("annualized_sharpe must be finite and non-negative.")
    if not math.isfinite(daily_volatility) or daily_volatility <= 0:
        raise ValueError("daily_volatility must be finite and positive.")
    if (
        not math.isfinite(transaction_cost_per_observation)
        or transaction_cost_per_observation < 0
    ):
        raise ValueError("transaction_cost_per_observation must be finite and non-negative.")

    rng = random.Random(seed)
    target = [rng.gauss(0.0, 1.0) for _ in range(observations)]
    target_mean = sum(target) / observations
    target_variance = sum((value - target_mean) ** 2 for value in target) / (observations - 1)
    target_std = math.sqrt(target_variance)
    target_z = [(value - target_mean) / target_std for value in target]

    noise = [rng.gauss(0.0, 1.0) for _ in range(observations)]
    projection = sum(x * y for x, y in zip(noise, target_z)) / sum(
        value * value for value in target_z
    )
    orthogonal = [x - projection * y for x, y in zip(noise, target_z)]
    orthogonal_mean = sum(orthogonal) / observations
    orthogonal_variance = sum(
        (value - orthogonal_mean) ** 2 for value in orthogonal
    ) / (observations - 1)
    orthogonal_std = math.sqrt(orthogonal_variance)
    orthogonal_z = [
        (value - orthogonal_mean) / orthogonal_std for value in orthogonal
    ]

    signal = tuple(
        information_coefficient * outcome
        + math.sqrt(1 - information_coefficient ** 2) * independent
        for outcome, independent in zip(target_z, orthogonal_z)
    )
    net_mean = daily_volatility * annualized_sharpe / math.sqrt(252.0)
    net_returns = tuple(net_mean + daily_volatility * value for value in target_z)
    gross_returns = tuple(
        value + transaction_cost_per_observation for value in net_returns
    )
    return SyntheticSignalCalibration(
        signal=signal,
        gross_returns=gross_returns,
        net_returns=net_returns,
        transaction_cost_per_observation=transaction_cost_per_observation,
        information_coefficient=information_coefficient,
        annualized_sharpe=annualized_sharpe,
    )


def _trading_days(data: dict[str, list[MinuteBar]]) -> list[date]:
    return sorted({
        bar.ts.astimezone(NY).date()
        for bars in data.values()
        for bar in bars
    })


def _make_groups(days: Sequence[date], group_count: int) -> list[list[date]]:
    if group_count < 4 or group_count % 2:
        raise ValueError("CPCV group_count must be an even integer of at least four.")
    if len(days) < group_count:
        raise ValueError("CPCV requires at least one trading day in every group.")
    quotient, remainder = divmod(len(days), group_count)
    groups, offset = [], 0
    for index in range(group_count):
        size = quotient + (index < remainder)
        groups.append(list(days[offset:offset + size]))
        offset += size
    return groups


def _group_trades(
    results: dict[tuple[Any, ...], list[TradeRecord]],
    group_by_day: dict[date, int],
) -> dict[tuple[Any, ...], dict[int, list[TradeRecord]]]:
    grouped = {}
    for params, trades in results.items():
        by_group: dict[int, list[TradeRecord]] = {}
        for trade in trades:
            entry_day = trade.entry_ts.astimezone(NY).date()
            exit_day = trade.exit_ts.astimezone(NY).date()
            entry_group = group_by_day.get(entry_day)
            exit_group = group_by_day.get(exit_day)
            if entry_group is not None and entry_group == exit_group:
                by_group.setdefault(entry_group, []).append(trade)
        grouped[params] = by_group
    return grouped


def _run_isolated_group(args):
    params, data, config, reset_days, signal_fn = args
    result = run_backtest(
        data, config, signal_fn=signal_fn, reset_equity_days=reset_days,
    )
    return params, result.trades


def _returns(
    trades: Sequence[TradeRecord],
    selected_days: Sequence[date],
    initial_equity: float,
) -> list[float]:
    pnl_by_day: dict[date, float] = {}
    selected = set(selected_days)
    for trade in trades:
        day = trade.exit_ts.astimezone(NY).date()
        if day in selected:
            pnl_by_day[day] = pnl_by_day.get(day, 0.0) + trade.pnl_net
    equity = initial_equity
    output = []
    for day in selected_days:
        pnl = pnl_by_day.get(day, 0.0)
        if equity <= 0 or not math.isfinite(equity):
            return []
        output.append(pnl / equity)
        equity += pnl
    return output


def _sharpe(returns: Sequence[float], annualization: float = 252.0) -> float:
    if len(returns) < 2:
        return float("-inf")
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    if not math.isfinite(variance) or variance <= 0:
        return float("-inf")
    return mean / math.sqrt(variance) * math.sqrt(annualization)


def evaluate_cpcv(
    data: dict[str, list[MinuteBar]],
    base: BTConfig,
    results: dict[tuple[Any, ...], list[TradeRecord]],
    *,
    n_trials: int,
    grid: dict[str, Sequence[float]] | None = None,
    group_count: int = 6,
    embargo_days: int = 1,
    workers: int = 0,
    signal_fn: SignalFn | None = None,
) -> dict[str, Any]:
    """Replay candidates with independent group equity, then purge and embargo every split."""
    if not results:
        raise ValueError("CPCV requires results for at least one tested parameter set.")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if group_count < 4 or group_count % 2:
        raise ValueError("CPCV group_count must be an even integer of at least four.")
    days = _trading_days(data)
    if len(days) < group_count:
        return {
            "method": "CPCV-CSCV",
            "groups": group_count,
            "test_groups": group_count // 2,
            "embargo_days": embargo_days,
            "splits": 0,
            "valid_splits": 0,
            "pbo": None,
            "pbo_threshold": 0.20,
            "pbo_passed": False,
            "n_trials": n_trials,
            "trials_in_run": len(results),
            "dsr": deflated_sharpe_from_returns([], n_trials),
            "oos_observations": 0,
            "split_results": [],
            "passed": False,
            "reason": "insufficient trading days for the configured CPCV groups",
        }
    groups = _make_groups(days, group_count)
    grid = grid or DEFAULT_GRID
    names = list(grid)
    if not names or any(not values for values in grid.values()):
        raise ValueError("CPCV parameter grid must contain non-empty values.")
    if any(len(params) != len(names) for params in results):
        raise ValueError("CPCV parameter keys do not match the parameter grid.")
    reset_days = frozenset(group[0] for group in groups)
    jobs = [
        (
            params,
            data,
            apply_params(base, dict(zip(names, params))),
            reset_days,
            signal_fn,
        )
        for params in results
    ]
    worker_count = workers or min(4, os.cpu_count() or 1)
    if worker_count < 1:
        raise ValueError("workers must be positive.")
    if worker_count > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            isolated_results = dict(executor.map(_run_isolated_group, jobs))
    else:
        isolated_results = dict(_run_isolated_group(job) for job in jobs)
    day_position = {day: index for index, day in enumerate(days)}
    group_by_day = {
        day: index
        for index, group in enumerate(groups)
        for day in group
    }
    grouped_trades = _group_trades(isolated_results, group_by_day)
    grouped_daily_pnl: dict[tuple[Any, ...], dict[int, dict[date, float]]] = {}
    for params, by_group in grouped_trades.items():
        grouped_daily_pnl[params] = {}
        for group_index, trades in by_group.items():
            daily_pnl: dict[date, float] = {}
            for trade in trades:
                day = trade.exit_ts.astimezone(NY).date()
                daily_pnl[day] = daily_pnl.get(day, 0.0) + trade.pnl_net
            grouped_daily_pnl[params][group_index] = daily_pnl
    test_group_count = group_count // 2
    split_count = 0
    valid_split_count = 0
    overfit_count = 0
    split_results: list[dict[str, Any]] = []
    oos_pnl: dict[date, float] = {}
    oos_observations: dict[date, int] = {}
    all_params = tuple(results)

    for test_indices_tuple in itertools.combinations(range(group_count), test_group_count):
        split_count += 1
        test_indices = set(test_indices_tuple)
        test_days = [
            day for index in test_indices
            for day in groups[index]
        ]
        train_indices = set(range(group_count)) - test_indices
        embargoed: set[date] = set()
        for index in test_indices:
            if index + 1 in test_indices:
                continue
            end_position = day_position[groups[index][-1]]
            embargoed.update(days[end_position + 1:end_position + 1 + embargo_days])
        train_days = [
            day for index in train_indices
            for day in groups[index]
            if day not in embargoed
        ]
        if len(train_days) < 2 or len(test_days) < 2:
            continue

        train_scores: dict[tuple[Any, ...], float] = {}
        test_scores: dict[tuple[Any, ...], float] = {}
        for params in all_params:
            candidate_trades = grouped_trades[params]
            train_trades = [
                trade for index in train_indices
                for trade in candidate_trades.get(index, [])
                if trade.entry_ts.astimezone(NY).date() not in embargoed
                and trade.exit_ts.astimezone(NY).date() not in embargoed
            ]
            test_trades = [
                trade for index in test_indices
                for trade in candidate_trades.get(index, [])
            ]
            train_scores[params] = _sharpe(
                _returns(train_trades, train_days, base.initial_equity)
            )
            test_scores[params] = _sharpe(
                _returns(test_trades, test_days, base.initial_equity)
            )

        eligible = [params for params in all_params if math.isfinite(train_scores[params])]
        if not eligible:
            continue
        valid_split_count += 1
        winner = max(eligible, key=lambda params: train_scores[params])
        winner_test_score = test_scores[winner]
        lower = sum(score < winner_test_score for score in test_scores.values())
        equal = sum(score == winner_test_score for score in test_scores.values())
        rank_fraction = (lower + 0.5 * equal) / len(all_params)
        if rank_fraction <= 0.5:
            overfit_count += 1
        split_results.append({
            "split": split_count,
            "test_groups": sorted(test_indices),
            "winner_params": dict(zip(names, winner)),
            "train_sharpe": (
                train_scores[winner] if math.isfinite(train_scores[winner]) else None
            ),
            "test_sharpe": (
                winner_test_score if math.isfinite(winner_test_score) else None
            ),
            "test_rank_fraction": rank_fraction,
        })
        for day in test_days:
            group_index = group_by_day[day]
            day_pnl = grouped_daily_pnl[winner].get(group_index, {}).get(day, 0.0)
            oos_pnl[day] = oos_pnl.get(day, 0.0) + day_pnl
            oos_observations[day] = oos_observations.get(day, 0) + 1

    pbo = overfit_count / valid_split_count if valid_split_count else None
    oos_days = sorted(oos_observations)
    averaged_returns = [
        (oos_pnl[day] / oos_observations[day]) / base.initial_equity
        for day in oos_days
    ]
    dsr = deflated_sharpe_from_returns(averaged_returns, n_trials)
    pbo_passed = pbo is not None and pbo < 0.20
    return {
        "method": "CPCV-CSCV",
        "groups": group_count,
        "test_groups": test_group_count,
        "embargo_days": embargo_days,
        "splits": split_count,
        "valid_splits": valid_split_count,
        "pbo": pbo,
        "pbo_threshold": 0.20,
        "pbo_passed": pbo_passed,
        "n_trials": int(n_trials),
        "trials_in_run": len(results),
        "dsr": dsr,
        "oos_observations": len(averaged_returns),
        "split_results": split_results,
        "passed": bool(pbo_passed and dsr["passed"]),
    }
