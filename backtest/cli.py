"""Usage:
  python -m backtest.cli run --synthetic                     # sanity check on a no-edge random walk
  python -m backtest.cli run --csv-dir data --symbols SPY QQQ [--walk-forward]
  python -m backtest.cli fetch --symbols SPY QQQ --start 2025-01-01 --end 2026-09-30 --out data   # Alpaca, free IEX
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import replace

from backtest.data import data_quality, fetch_alpaca, load_dir, save_csv, synthetic_bars
from backtest.engine import BTConfig, run_backtest
from backtest.metrics import fmt, summarize, trade_equity_curve, verdict
from backtest.walkforward import walk_forward
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


def cmd_run(a) -> int:
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
        commission_per_share=a.commission_per_share,
        commission_min=a.commission_min,
        sell_fee_bps=a.sell_fee_bps,
        min_signal_score=a.min_score,
    )

    if not a.walk_forward_only:
        res = run_backtest(data, cfg)
        print(BLIND_SPOTS)
        _report("STRATEGY", res.trades, res.equity_curve, a.equity)
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
        _report("NULL MODEL (random entries, same exits & costs, 3 seeds pooled)", rnd_trades, [], a.equity)
        print("The strategy only has an edge if it beats the null model, not merely if it is positive.")

        if a.out:
            with open(a.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["symbol", "side", "entry_ts", "exit_ts", "entry", "exit", "qty", "stop", "gross", "costs",
                            "net", "R", "reason", "score", "rationale"])
                for t in res.trades:
                    w.writerow([t.symbol, t.side, t.entry_ts, t.exit_ts, f"{t.entry:.4f}", f"{t.exit:.4f}", t.qty,
                                f"{t.initial_stop:.4f}", f"{t.pnl_gross:.2f}", f"{t.costs:.2f}", f"{t.pnl_net:.2f}",
                                f"{t.r_multiple:.3f}", t.exit_reason, f"{t.score:.2f}", t.rationale])
            print(f"\ntrade log -> {a.out}")

    if a.walk_forward or a.walk_forward_only:
        folds, oos, _ = walk_forward(data, cfg, train_days=a.train_days, test_days=a.test_days, workers=a.workers)
        print("\n== WALK-FORWARD folds ==")
        for f in folds:
            print(f"  train {f['train']} -> test {f['test']}  {f['params']}  "
                  f"trainR={f['train_meanR']:+.3f}(n={f['train_n']})  "
                  f"testR={f['test_meanR']:+.3f}(n={f['test_n']}) "
                  f"testWR={f['test_win_rate']:.1%} testPF={f['test_profit_factor']:.2f}")
        oos_curve = trade_equity_curve(oos, a.equity)
        _report("WALK-FORWARD OUT-OF-SAMPLE (concatenated test windows)", oos, oos_curve, a.equity)
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
    r.add_argument("--commission-per-share", type=float, default=0.0035)
    r.add_argument("--commission-min", type=float, default=0.35)
    r.add_argument("--sell-fee-bps", type=float, default=0.3)
    r.add_argument("--min-score", type=float, default=0.70)
    r.add_argument("--naive-tz", default="UTC")
    r.add_argument("--walk-forward", action="store_true")
    r.add_argument("--walk-forward-only", action="store_true",
                   help="run only walk-forward evaluation; skip full-history and null-model replays")
    r.add_argument("--train-days", type=int, default=60)
    r.add_argument("--test-days", type=int, default=20)
    r.add_argument("--workers", type=int, default=0)
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
