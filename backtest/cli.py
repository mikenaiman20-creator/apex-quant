"""Usage:
  python -m backtest.cli run --synthetic                     # sanity check on a no-edge random walk
  python -m backtest.cli run --csv-dir data --symbols SPY QQQ [--walk-forward]
  python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --strategy orb --walk-forward
  python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --strategy zscore --walk-forward-only
  python -m backtest.cli fetch --symbols SPY QQQ --start 2025-01-01 --end 2026-09-30 --out data   # Alpaca, free IEX
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from server.io_utils import ensure_directory, secure_file

from backtest.cpcv import evaluate_cpcv
from backtest.data import data_quality, fetch_alpaca, load_dir, save_csv, synthetic_bars
from backtest.engine import BTConfig, run_backtest
from backtest.metrics import fmt, summarize, trade_equity_curve, verdict
from backtest.strategies import (
    ORB_PARAMETER_GRID,
    ZSCORE_PARAMETER_GRID,
    opening_range_breakout,
    short_term_zscore_mean_reversion,
)
from backtest.walkforward import DEFAULT_GRID, run_parameter_grid, walk_forward, parameter_grid_size
from config.settings import load_project_env

BLIND_SPOTS = (
    "NOT testable on bar data: book_imbalance (no L2, scored 0), real tape prints (delta approximated from bar close\n"
    "location; 'block trades' are volume-spike artifacts), 1s timeframe. Treat those weights as untested."
)
log = logging.getLogger("apex.backtest.cli")


def _report(title, trades, curve, equity):
    m = summarize(trades, curve, equity)
    print(f"\n== {title} ==\n{fmt(m)}")
    for n in verdict(m):
        print(" -", n)
    return m


def _report_pooled_null_model(trades, initial_equity: float, seed_count: int = 3):
    """Report pooled independent null runs against their combined starting capital."""
    pooled_equity = initial_equity * seed_count
    curve = trade_equity_curve(trades, pooled_equity)
    return _report(
        f"NULL MODEL (random entries, same exits & costs, {seed_count} seeds pooled)",
        trades,
        curve,
        pooled_equity,
    )


def cmd_run(a) -> int:
    if a.strategy == "orb":
        signal_fn, grid = opening_range_breakout, ORB_PARAMETER_GRID
    elif a.strategy == "zscore":
        signal_fn, grid = short_term_zscore_mean_reversion, ZSCORE_PARAMETER_GRID
    else:
        signal_fn, grid = None, None
    workers = a.workers if a.workers else (1 if a.strategy in {"orb", "zscore"} else 0)
    if a.synthetic:
        data = {s: synthetic_bars(n_days=a.days, seed=11 + i) for i, s in enumerate(a.symbols)}
        print(f"Synthetic random-walk data (no edge by construction): {a.days} days x {len(a.symbols)} symbols")
    else:
        data = load_dir(a.csv_dir, a.symbols, a.naive_tz)
        for symbol, bars in data.items():
            if not bars:
                raise ValueError(
                    f"No valid regular-session bars loaded for {symbol} from "
                    f"{a.csv_dir}/{symbol}.csv."
                )
            quality = data_quality(bars)
            log.info(
                "Validated %s: %d bars, %s through %s; flat=%d low_volume=%d gaps=%d",
                symbol, len(bars), bars[0].ts.isoformat(), bars[-1].ts.isoformat(),
                quality["flatline_bars"], quality["low_volume_bars"],
                quality["missing_intervals"],
            )
    cfg = BTConfig(
        initial_equity=a.equity,
        slippage_bps=a.slippage_bps,
        slippage_atr_mult=a.slippage_atr_mult,
        spread_bps=a.spread_bps,
        commission_per_share=a.commission_per_share,
        commission_min=a.commission_min,
        sell_fee_bps=a.sell_fee_bps,
        min_signal_score=a.min_score,
        zscore_entry_threshold=(
            ZSCORE_PARAMETER_GRID["zscore_entry_threshold"][0]
            if a.strategy == "zscore" else 2.0
        ),
    )

    from server.evolution import TRIAL_LEDGER_PATH, register_trials

    cycle_id = f"backtest-cli:{datetime.now(timezone.utc).isoformat()}:{uuid4().hex}"
    trial_count = register_trials(
        cycle_id, parameter_grid_size(grid or DEFAULT_GRID), TRIAL_LEDGER_PATH,
    )
    results = run_parameter_grid(
        data, cfg, grid=grid, workers=workers, signal_fn=signal_fn,
        trial_cycle_id=cycle_id, trial_ledger_path=TRIAL_LEDGER_PATH,
    )
    cpcv = evaluate_cpcv(
        data,
        cfg,
        results,
        n_trials=trial_count,
        grid=grid,
        group_count=a.cpcv_groups,
        embargo_days=a.cpcv_embargo_days,
        workers=workers,
        signal_fn=signal_fn,
    )
    pbo = cpcv["pbo"]
    dsr = cpcv["dsr"]
    pbo_text = f"{pbo:.3f}" if pbo is not None else "n/a"
    dsr_text = (
        f"{dsr['adjusted_sharpe']:.3f}"
        if dsr["adjusted_sharpe"] is not None else "n/a"
    )
    print("\n== COMBINATORIAL PURGED CROSS-VALIDATION ==")
    print(
        f"method={cpcv['method']}  splits={cpcv['valid_splits']}/{cpcv['splits']}  "
        f"groups={cpcv['groups']}  test_groups={cpcv['test_groups']}  "
        f"embargo_days={cpcv['embargo_days']}"
    )
    print(
        f"PBO={pbo_text} (required < {cpcv['pbo_threshold']:.2f}; "
        f"{'PASS' if cpcv['pbo_passed'] else 'FAIL'})  "
        f"DSR_probability={dsr_text} (required >= 0.950; "
        f"{'PASS' if dsr['passed'] else 'FAIL'})"
    )
    print(
        f"trials_in_run={cpcv['trials_in_run']}  cumulative_trials={cpcv['n_trials']}  "
        f"DSR_observations={cpcv['oos_observations']}"
    )
    if cpcv.get("reason"):
        print(f"CPCV unavailable: {cpcv['reason']}")

    if not a.walk_forward_only:
        res = run_backtest(data, cfg, signal_fn=signal_fn)
        if signal_fn is None:
            print(BLIND_SPOTS)
        _report(
            "STRATEGY" if signal_fn is None else f"STRATEGY (INDEPENDENT {a.strategy.upper()})",
            res.trades, res.equity_curve, a.equity,
        )
        print(f"\nevaluations={res.evals}  bars={res.bars}")
        print("check pass counts (how often each component fired on the best side):")
        for k, v in sorted(res.check_pass.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<20}{v:>8}  ({v / max(res.evals, 1):.1%})")
        print("top reject reasons:")
        for k, v in res.rejects.most_common(5):
            print(f"  {v:>8}  {k}")

        # Null model: random direction, same exits/costs, matched trade frequency.
        p = max(len(res.trades) / max(res.evals, 1), 1e-4)
        rnd_trades = []
        for seed in range(3):
            r = run_backtest(data, replace(cfg, entry_mode="random", random_entry_prob=p, seed=seed))
            rnd_trades.extend(r.trades)
        _report_pooled_null_model(rnd_trades, a.equity)
        print("The strategy only has an edge if it beats the null model, not merely if it is positive.")

        if a.out:
            out_path = Path(a.out)
            if str(out_path.parent) not in {"", "."}:
                ensure_directory(out_path.parent)
            with out_path.open("w", newline="") as f:
                secure_file(out_path)
                w = csv.writer(f)
                w.writerow(["symbol", "side", "entry_ts", "exit_ts", "entry", "exit", "qty", "stop", "gross", "costs",
                            "net", "R", "reason", "score", "rationale"])
                for t in res.trades:
                    w.writerow([t.symbol, t.side, t.entry_ts, t.exit_ts, f"{t.entry:.4f}", f"{t.exit:.4f}", t.qty,
                                f"{t.initial_stop:.4f}", f"{t.pnl_gross:.2f}", f"{t.costs:.2f}", f"{t.pnl_net:.2f}",
                                f"{t.r_multiple:.3f}", t.exit_reason, f"{t.score:.2f}", t.rationale])
            print(f"\ntrade log -> {a.out}")

    if a.walk_forward or a.walk_forward_only:
        folds, oos, _ = walk_forward(
            data, cfg, train_days=a.train_days, test_days=a.test_days,
            workers=workers, results=results, grid=grid, signal_fn=signal_fn,
        )
        print("\n== WALK-FORWARD folds ==")
        for f in folds:
            if not f["selected"]:
                print(
                    f"  train {f['train']} -> test {f['test']}  "
                    f"NO QUALIFYING TRAIN CANDIDATE (test exposure=0)"
                )
                continue
            print(f"  train {f['train']} -> test {f['test']}  {f['params']}  "
                  f"trainR={f['train_meanR']:+.3f}(n={f['train_n']})  "
                  f"testR={f['test_meanR']:+.3f}(n={f['test_n']}) "
                  f"testWR={f['test_win_rate']:.1%} testPF={f['test_profit_factor']:.2f}")
        oos_days = [day for fold in folds for day in fold["test_days"]]
        oos_curve = trade_equity_curve(oos, a.equity, calendar_days=oos_days)
        _report("WALK-FORWARD OUT-OF-SAMPLE (concatenated test windows)", oos, oos_curve, a.equity)
        if a.strategy == "zscore":
            from server.portfolio_allocation import build_equal_risk_plan

            expectancy = summarize(oos, oos_curve, a.equity)
            plan = build_equal_risk_plan({
                "equity-zscore-mean-reversion-v1": {
                    "cpcv": cpcv,
                    "holdout": {"passed": False},
                    "cost_calibration_error": None,
                    "oos_expectancy_r": expectancy["mean_r"],
                    "oos_net_pnl": expectancy["net_pnl"],
                }
            })
            budget_dollars = a.equity * plan["approved_research_risk_pct"]
            sleeve = plan["sleeves"]["equity-zscore-mean-reversion-v1"]
            print("\n== FAIL-CLOSED RESEARCH ALLOCATION ==")
            print(
                f"strategy=equity-zscore-mean-reversion-v1  "
                f"validation_eligible={'YES' if sleeve['validation_eligible'] else 'NO'}  "
                f"research_risk_budget=${budget_dollars:.2f}  "
                f"execution_risk=${a.equity * plan['execution_risk_pct']:.2f}"
            )
            for reason in sleeve["reasons"]:
                print(f" - {reason}")
            print(" - locked holdout and independent cost calibration remain required for allocation")
            print(" - live and paper order routing remain disabled")
    return 0


def cmd_fetch(a) -> int:
    for s in a.symbols:
        bars = fetch_alpaca(s, a.start, a.end, a.feed)
        save_csv(f"{a.out}/{s}.csv", bars)
        print(f"{s}: {len(bars)} RTH bars -> {a.out}/{s}.csv")
    return 0


def main(argv=None) -> int:
    load_project_env()
    p = argparse.ArgumentParser(prog="backtest")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--csv-dir")
    g.add_argument("--synthetic", action="store_true")
    r.add_argument(
        "--symbols", nargs="+", default=["SPY", "QQQ", "NVDA"],
        help="symbols to evaluate (default: SPY QQQ NVDA)",
    )
    r.add_argument("--days", type=int, default=160, help="synthetic only")
    r.add_argument("--equity", type=float, default=5000.0)
    r.add_argument("--slippage-bps", type=float, default=1.0)
    r.add_argument("--slippage-atr-mult", type=float, default=0.1)
    r.add_argument(
        "--spread-bps", type=float, default=2.0,
        help="full quoted spread in bps; half-spread is charged on each fill",
    )
    r.add_argument("--commission-per-share", type=float, default=0.0035)
    r.add_argument("--commission-min", type=float, default=0.35)
    r.add_argument("--sell-fee-bps", type=float, default=0.3)
    r.add_argument("--min-score", type=float, default=0.70)
    r.add_argument(
        "--strategy", choices=("alpha", "orb", "zscore"), default="alpha",
        help="strategy to evaluate; ORB and zscore are independent research-only signals",
    )
    r.add_argument("--naive-tz", default="UTC")
    r.add_argument("--walk-forward", action="store_true")
    r.add_argument("--walk-forward-only", action="store_true",
                   help="run only walk-forward evaluation; skip full-history and null-model replays")
    r.add_argument("--train-days", type=int, default=60)
    r.add_argument("--test-days", type=int, default=20)
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--cpcv-groups", type=int, default=6,
                   help="even number of chronological CPCV groups (default: 6)")
    r.add_argument("--cpcv-embargo-days", type=int, default=1,
                   help="training sessions embargoed after each test block (default: 1)")
    r.add_argument("--out", help="write trade log CSV")
    r.set_defaults(fn=cmd_run)

    f = sub.add_parser("fetch")
    f.add_argument("--symbols", nargs="+", required=True)
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--feed", default="iex")
    f.add_argument("--out", default="data")
    f.set_defaults(fn=cmd_fetch)

    a = p.parse_args(argv)
    if a.cmd == "run" and not a.synthetic and not a.csv_dir:
        a.csv_dir = "data"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(line_buffering=True)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        log.info("Backtest interrupted.")
        return 130
    except Exception:
        log.exception("Backtest command '%s' failed.", a.cmd)
        return 1


if __name__ == "__main__":
    sys.exit(main())
