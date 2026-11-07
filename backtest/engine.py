"""Event-driven bar-replay backtester that reuses the LIVE AlphaEngine / RiskEngine / SymbolState code.

Honesty rules baked in:
  * Signals are computed after a 1m bar closes and filled at the NEXT bar's open (+ adverse slippage). No same-bar fills.
  * If a bar touches both stop and target, the STOP is assumed to hit first. Gaps through a stop fill at the open.
  * Costs: per-share commission with a minimum, slippage in bps, and a small regulatory fee on sells.
  * Positions are flattened at `flat_at` (default 15:55 NY). Daily-drawdown kill-switch mirrors the live firewall.
  * Bar data has no order book or real tape. Delta is approximated from the close's position in the bar's range;
    book_imbalance is 0 by construction; block-trade detection reacts to volume spikes, not real prints.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from backtest.data import MinuteBar, NY
from config.settings import RiskLimits
from engine.data_pipeline import SymbolState, Trade, atr
from engine.strategy import AlphaEngine, Decision, RiskEngine, Side

SignalFn = Callable[[SymbolState, float], Optional[Decision]]


@dataclass(frozen=True)
class BTConfig:
    initial_equity: float = 5000.0
    risk: RiskLimits = field(default_factory=RiskLimits)
    min_signal_score: float = 0.70
    warmup_days: int = 50                 # Daily-50-EMA needs 50 closed days
    entry_start: time = time(9, 45)
    entry_end: time = time(15, 30)
    flat_at: time = time(15, 55)
    slippage_bps: float = 1.0
    slippage_atr_mult: float = 0.1
    commission_per_share: float = 0.0035  # IBKR tiered-ish; set 0 to model Alpaca commission-free
    commission_min: float = 0.35
    sell_fee_bps: float = 0.3             # SEC + TAF approx, sells only
    entry_mode: str = "strategy"          # "strategy" | "random" (null-model baseline, same exits/costs)
    random_entry_prob: float = 0.01
    seed: int = 7

    def __post_init__(self):
        cost_values = (
            self.slippage_bps, self.slippage_atr_mult, self.commission_per_share,
            self.commission_min, self.sell_fee_bps,
        )
        if not all(math.isfinite(value) and value >= 0 for value in cost_values):
            raise ValueError("Transaction cost configuration must be finite and non-negative.")


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_ts: datetime
    exit_ts: datetime
    entry: float
    exit: float
    qty: int
    initial_stop: float
    pnl_gross: float
    costs: float
    pnl_net: float
    r_multiple: float
    exit_reason: str
    score: float
    rationale: str
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    regulatory_fees: float = 0.0


@dataclass
class _Pos:
    symbol: str
    side: Side
    qty: int
    entry: float
    market_entry: float
    initial_stop: float
    stop: float
    target: float
    extreme: float
    entry_ts: datetime
    entry_cost: float
    score: float
    rationale: str
    slippage_rate: float
    entry_slippage_cost: float
    entry_commission_cost: float
    entry_regulatory_fees: float


@dataclass
class BTResult:
    cfg: BTConfig
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Tuple[date, float]] = field(default_factory=list)
    evals: int = 0
    check_pass: Counter = field(default_factory=Counter)
    rejects: Counter = field(default_factory=Counter)
    bars: int = 0


def _feed(st: SymbolState, b: MinuteBar) -> None:
    """Turn one OHLCV bar into 4 price prints (O, then L/H in the direction-consistent order, then C)."""
    rng = b.h - b.l
    buy_frac = 0.5 if rng <= 0 else (b.c - b.l) / rng
    pts = [b.o, b.l, b.h, b.c] if b.c >= b.o else [b.o, b.h, b.l, b.c]
    per = b.v / 4
    for k, px in enumerate(pts):
        t = b.ts + timedelta(seconds=15 * k)
        bs = per * buy_frac
        ss = per - bs
        if bs > 0:
            st.on_trade(Trade(t, st.symbol, px, bs, 1))
        if ss > 0:
            st.on_trade(Trade(t, st.symbol, px, ss, -1))


def _normalize_reject(msg: str) -> str:
    return "score below threshold" if msg.startswith("score") else re.sub(r"[\d\.\-%]+", "#", msg)


def run_backtest(data: Dict[str, List[MinuteBar]], cfg: BTConfig, signal_fn: Optional[SignalFn] = None) -> BTResult:
    rng = random.Random(cfg.seed)
    risk = RiskEngine(cfg.risk)
    alpha = AlphaEngine(risk, cfg.min_signal_score)
    states = {s: SymbolState(s) for s in data}
    by_ts: Dict[datetime, List[Tuple[str, MinuteBar]]] = defaultdict(list)
    for sym, bars in data.items():
        for b in bars:
            by_ts[b.ts].append((sym, b))

    res = BTResult(cfg=cfg)
    positions: Dict[str, _Pos] = {}
    pending: Dict[str, Tuple[Decision, date]] = {}
    last_close: Dict[str, float] = {}
    realized = 0.0
    minimum_slip = cfg.slippage_bps / 1e4
    cur_day: Optional[date] = None
    day_start = cfg.initial_equity
    halted = False

    def equity() -> float:
        u = 0.0
        for s, p in positions.items():
            px = last_close.get(s, p.entry)
            u += (px - p.entry) * p.qty * (1 if p.side is Side.LONG else -1)
        return cfg.initial_equity + realized + u

    def commission(qty: int) -> float:
        return max(qty * cfg.commission_per_share, cfg.commission_min) if cfg.commission_per_share > 0 else 0.0

    def close(sym: str, px: float, reason: str, ts: datetime, adverse: bool) -> None:
        nonlocal realized
        p = positions.pop(sym)
        market_exit = px
        exit_slippage_cost = 0.0
        if adverse:
            px = px * (1 - p.slippage_rate) if p.side is Side.LONG else px * (1 + p.slippage_rate)
            exit_slippage_cost = abs(px - market_exit) * p.qty
        direction = 1 if p.side is Side.LONG else -1
        gross = (market_exit - p.market_entry) * p.qty * direction
        exit_commission = commission(p.qty)
        exit_regulatory = px * p.qty * cfg.sell_fee_bps / 1e4 if p.side is Side.LONG else 0.0
        exit_cost = exit_commission + exit_regulatory
        realized += gross - exit_cost - exit_slippage_cost
        costs = p.entry_cost + exit_cost + exit_slippage_cost
        net = gross - costs
        risk_amt = abs(p.entry - p.initial_stop) * p.qty
        res.trades.append(TradeRecord(
            sym, p.side.value, p.entry_ts, ts, p.entry, px, p.qty, p.initial_stop, gross, costs, net,
            net / risk_amt if risk_amt > 0 else 0.0, reason, p.score, p.rationale,
            p.entry_slippage_cost + exit_slippage_cost,
            p.entry_commission_cost + exit_commission,
            p.entry_regulatory_fees + exit_regulatory,
        ))

    def try_open(sym: str, dec: Decision, b: MinuteBar) -> None:
        nonlocal realized
        long_ = dec.side is Side.LONG
        five_minute_atr = atr(list(states[sym].agg.bars["5m"])[-60:], 14)
        dynamic_slip = (
            cfg.slippage_atr_mult * five_minute_atr / b.o
            if five_minute_atr is not None and b.o > 0 else 0.0
        )
        slip = max(minimum_slip, dynamic_slip)
        fill = b.o * (1 + slip) if long_ else b.o * (1 - slip)
        stop = dec.stop
        if stop is None or (long_ and fill <= stop) or (not long_ and fill >= stop):
            return
        qty = risk.size(equity(), fill, stop)
        if qty < 1:
            return
        dist = abs(fill - stop)
        target = fill + cfg.risk.min_reward_risk * dist if long_ else fill - cfg.risk.min_reward_risk * dist
        entry_commission = commission(qty)
        entry_regulatory = 0.0 if long_ else fill * qty * cfg.sell_fee_bps / 1e4
        entry_slippage = abs(fill - b.o) * qty
        cost = entry_commission + entry_regulatory
        realized -= cost + entry_slippage
        positions[sym] = _Pos(
            sym, dec.side, qty, fill, b.o, stop, stop, target, fill, b.ts, cost,
            dec.score, dec.rationale() if dec.checks else "", slip, entry_slippage,
            entry_commission, entry_regulatory,
        )

    for ts in sorted(by_ts):
        local = ts.astimezone(NY)
        d, tod = local.date(), local.time()
        if d != cur_day:
            if cur_day is not None:
                res.equity_curve.append((cur_day, equity()))
            cur_day, halted, day_start = d, False, equity()
            pending.clear()

        for sym, b in by_ts[ts]:
            res.bars += 1
            st = states[sym]
            _feed(st, b)
            last_close[sym] = b.c

            if sym in pending:
                dec, sig_day = pending.pop(sym)
                if sig_day == d and not halted and tod < cfg.flat_at and sym not in positions:
                    try_open(sym, dec, b)

            p = positions.get(sym)
            if p is None:
                continue
            long_ = p.side is Side.LONG
            if tod >= cfg.flat_at:
                close(sym, b.o, "eod", ts, True)
                continue
            exit_px, reason, adverse = None, "", False
            if long_:
                if b.o <= p.stop:
                    exit_px, reason, adverse = b.o, "stop_gap", True
                elif b.l <= p.stop:
                    exit_px, reason, adverse = p.stop, "stop", True
                elif b.o >= p.target:
                    exit_px, reason = b.o, "target_gap"
                elif b.h >= p.target:
                    exit_px, reason = p.target, "target"
            else:
                if b.o >= p.stop:
                    exit_px, reason, adverse = b.o, "stop_gap", True
                elif b.h >= p.stop:
                    exit_px, reason, adverse = p.stop, "stop", True
                elif b.o <= p.target:
                    exit_px, reason = b.o, "target_gap"
                elif b.l <= p.target:
                    exit_px, reason = p.target, "target"
            if exit_px is not None:
                close(sym, exit_px, reason, ts, adverse)
                continue
            p.extreme = max(p.extreme, b.h) if long_ else min(p.extreme, b.l)
            a5 = atr(list(st.agg.bars["5m"])[-60:], 14)
            if a5:
                p.stop = risk.manage_stop(p.side, p.entry, p.initial_stop, p.stop, b.c, p.extreme, a5)

        # daily drawdown kill-switch (marks open positions at the bar close)
        if not halted and equity() <= day_start * (1 - cfg.risk.daily_drawdown_kill_pct):
            halted = True
            for sym in list(positions):
                close(sym, last_close[sym], "kill_switch", ts, True)
            pending.clear()

        if halted or not (cfg.entry_start <= tod <= cfg.entry_end):
            continue
        eq = equity()
        for sym, b in by_ts[ts]:
            if sym in positions or sym in pending:
                continue
            if len(positions) + len(pending) >= cfg.risk.max_open_positions:
                break
            st = states[sym]
            if signal_fn is None and len(st.daily_closes) < cfg.warmup_days:
                continue
            res.evals += 1
            if signal_fn is not None:
                dec = signal_fn(st, eq)
            elif cfg.entry_mode == "random":
                dec = _random_decision(st, eq, risk, rng, cfg.random_entry_prob)
            else:
                dec = alpha.evaluate(st, eq)
                if dec.checks:
                    for c in dec.checks:
                        if c.passed:
                            res.check_pass[c.name] += 1
                if not dec.take:
                    res.rejects[_normalize_reject(dec.reject_reason)] += 1
            if dec is not None and dec.take:
                pending[sym] = (dec, d)

    if cur_day is not None:
        res.equity_curve.append((cur_day, equity()))
    return res


def _random_decision(st: SymbolState, eq: float, risk: RiskEngine, rng: random.Random, p: float) -> Optional[Decision]:
    if rng.random() > p or st.last_price is None:
        return None
    a = atr(list(st.agg.bars["5m"])[-60:], 14)
    if not a:
        return None
    side = rng.choice([Side.LONG, Side.SHORT])
    px = st.last_price
    stop, target = risk.levels(side, px, a, None)
    return Decision(st.symbol, side, 1.0, True, [], px, stop, target, risk.size(eq, px, stop))
