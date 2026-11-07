"""Walk-forward parameter selection.

Each parameter combo is replayed ONCE over the full history (indicator state stays warm). Trades are then bucketed
by entry date: for every fold, positive-expectancy TRAIN candidates are ranked on capped profit factor, win rate,
and mean R; only the selected candidate's trades on the following TEST dates count toward out-of-sample results.
Approximation: sizing/position interplay is continuous across folds, which is fine for judging R-multiples.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Tuple

from backtest.data import MinuteBar, NY
from backtest.engine import BTConfig, BTResult, TradeRecord, run_backtest

DEFAULT_GRID = {
    "min_signal_score": [0.65, 0.75],
    "atr_stop_mult": [1.25, 1.75],
    "min_reward_risk": [2.0, 2.5],
    "chandelier_atr_mult": [2.5, 3.5],
}
_RISK_KEYS = {
    "atr_stop_mult", "min_reward_risk", "chandelier_atr_mult",
    "risk_per_trade_pct", "max_position_notional_pct",
}


def apply_params(cfg: BTConfig, params: Dict[str, float]) -> BTConfig:
    risk_kw = {k: v for k, v in params.items() if k in _RISK_KEYS}
    top_kw = {k: v for k, v in params.items() if k not in _RISK_KEYS}
    return replace(cfg, risk=replace(cfg.risk, **risk_kw), **top_kw)


def _run(args) -> Tuple[Tuple, List[TradeRecord]]:
    key, data, cfg = args
    return key, run_backtest(data, cfg).trades


def _mean_r(trades: List[TradeRecord]) -> float:
    return sum(t.r_multiple for t in trades) / len(trades) if trades else float("-inf")


def _quality(trades: List[TradeRecord]) -> float:
    """Reward profitable, consistent training-window performance; reject non-positive expectancy."""
    mean_r = _mean_r(trades)
    if mean_r <= 0:
        return float("-inf")
    wins = [t.pnl_net for t in trades if t.pnl_net > 0]
    losses = [-t.pnl_net for t in trades if t.pnl_net < 0]
    if not wins:
        return float("-inf")
    profit_factor = sum(wins) / sum(losses) if losses else float("inf")
    win_rate = len(wins) / len(trades)
    return 0.5 * min(profit_factor, 3.0) / 3.0 + 0.3 * win_rate + 0.2 * min(mean_r, 1.0)


def walk_forward(data: Dict[str, List[MinuteBar]], base: BTConfig, grid=None, train_days: int = 60,
                 test_days: int = 20, min_train_trades: int = 20, workers: int = 0,
                 min_positive_train: bool = True):
    grid = grid or DEFAULT_GRID
    names = list(grid)
    combos = list(itertools.product(*[grid[n] for n in names]))
    jobs = [(c, data, apply_params(base, dict(zip(names, c)))) for c in combos]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = dict(ex.map(_run, jobs))
    else:
        results = dict(_run(j) for j in jobs)

    days = sorted({b.ts.astimezone(NY).date() for bars in data.values() for b in bars})
    folds, oos = [], []
    i = base.warmup_days  # no trades exist before warmup
    while i + train_days + test_days <= len(days):
        tr = set(days[i:i + train_days])
        te = set(days[i + train_days:i + train_days + test_days])
        best_key, best_val, best_n = None, float("-inf"), 0
        best_train_trades = []
        for key, trades in results.items():
            train_trades = [t for t in trades if t.entry_ts.astimezone(NY).date() in tr]
            if len(train_trades) < min_train_trades:
                continue
            quality = _quality(train_trades) if min_positive_train else _mean_r(train_trades)
            if quality > best_val:
                best_key, best_val, best_n = key, quality, len(train_trades)
                best_train_trades = train_trades
        if best_key is not None:
            test_trades = [t for t in results[best_key] if t.entry_ts.astimezone(NY).date() in te]
            oos.extend(test_trades)
            test_wins = [t for t in test_trades if t.pnl_net > 0]
            test_losses = [-t.pnl_net for t in test_trades if t.pnl_net < 0]
            folds.append({
                "train": f"{min(tr)}..{max(tr)}", "test": f"{min(te)}..{max(te)}",
                "params": dict(zip(names, best_key)), "train_meanR": _mean_r(best_train_trades),
                "train_quality": best_val, "train_n": best_n,
                "test_meanR": _mean_r(test_trades) if test_trades else float("nan"), "test_n": len(test_trades),
                "test_win_rate": len(test_wins) / len(test_trades) if test_trades else float("nan"),
                "test_profit_factor": (sum(t.pnl_net for t in test_trades if t.pnl_net > 0) /
                                       sum(test_losses)) if test_losses else float("nan"),
            })
        i += test_days
    return folds, oos, results
