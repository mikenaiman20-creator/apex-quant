"""Offline smoke tests: indicator math, sizing, stop monotonicity, live-mode guard. Run: python -m pytest -q"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from config.settings import LIVE_CONFIRM_PHRASE, RiskLimits, load_settings
import config.settings as settings_module
from engine.data_pipeline import (
    BookSnapshot, Bar, SymbolState, Trade, atr, bbwp, ema, volume_profile,
)
from engine.strategy import AlphaEngine, RiskEngine, Side
from engine.strategy import MultiTimeframeFilter

T0 = datetime(2026, 10, 26, 14, 30, tzinfo=timezone.utc)


def mk_bars(n, base=100.0, step=0.1):
    out = []
    for i in range(n):
        c = base + i * step
        out.append(Bar(T0 + timedelta(minutes=5 * i), c - 0.05, c + 0.3, c - 0.3, c, 1000, 600, 400, c * 1000))
    return out


def test_live_mode_requires_confirmation(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)
    with pytest.raises(RuntimeError):
        load_settings(require_credentials=False)
    monkeypatch.setenv("LIVE_CONFIRM", LIVE_CONFIRM_PHRASE)
    assert load_settings(require_credentials=False).is_live


def test_paper_is_default(monkeypatch):
    monkeypatch.delenv("TRADING_MODE", raising=False)
    assert not load_settings(require_credentials=False).is_live


def test_settings_load_project_env_from_absolute_root_outside_current_directory(
    monkeypatch, tmp_path,
):
    project_root = tmp_path / "project"
    working_directory = tmp_path / "elsewhere"
    project_root.mkdir()
    working_directory.mkdir()
    (project_root / ".env").write_text(
        "ALPACA_API_KEY=test-key-from-file\n"
        "ALPACA_API_SECRET=test-secret-from-file\n"
        "WEBHOOK_SECRET=test-webhook-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", project_root)
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "WEBHOOK_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(working_directory)

    settings = load_settings()

    assert settings.alpaca_key == "test-key-from-file"
    assert settings.alpaca_secret == "test-secret-from-file"


def test_atr_and_ema():
    bars = mk_bars(30)
    assert atr(bars, 14) == pytest.approx(0.6, abs=0.15)
    assert ema([1, 2, 3, 4, 5], 3)[-1] > 3.5


def test_volume_profile_ordering():
    vp = volume_profile(mk_bars(40))
    assert vp.val <= vp.poc <= vp.vah


def test_bbwp_range():
    closes = [100 + (i % 7) * 0.2 for i in range(300)]
    assert 0 <= bbwp(closes) <= 100


def test_position_size_risks_exact_fraction():
    r = RiskEngine(RiskLimits())
    qty = r.size(100_000, 100.0, 99.0)  # 0.5% = $500 risk, $1/share -> 500 shares, notional cap 200 shares
    assert qty == 200  # notional cap binds
    assert r.size(100_000, 100.0, 90.0) == 50  # $10/share -> 50 shares by risk


def test_stop_never_loosens():
    r = RiskEngine(RiskLimits())
    stop = 98.0
    s1 = r.manage_stop(Side.LONG, 100, 98, stop, 102.5, 102.5, 0.5)   # +1.25R -> BE+ chandelier
    assert s1 >= 100
    s2 = r.manage_stop(Side.LONG, 100, 98, s1, 101.0, 102.5, 0.5)     # price pulls back: stop must hold
    assert s2 >= s1
    s3 = r.manage_stop(Side.SHORT, 100, 102, 102, 97.5, 97.5, 0.5)
    assert s3 <= 100


def test_rr_at_least_target():
    r = RiskEngine(RiskLimits())
    stop, tgt = r.levels(Side.LONG, 100.0, 0.5, structure_stop=None)
    assert (tgt - 100.0) / (100.0 - stop) >= 2.0 - 1e-9


def test_mtf_filter_requires_daily_hourly_and_intraday_alignment():
    bars = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(220):
        close = 80 + index * 0.5
        bars.append(Bar(start + timedelta(hours=index), close - 0.1, close + 0.3,
                        close - 0.3, close, 1000))
    intraday = []
    for index in range(30):
        close = 180 + index * 0.5
        intraday.append(Bar(start + timedelta(minutes=index * 15), close - 0.1,
                            close + 0.3, close - 0.3, close, 1000))
    engine = MultiTimeframeFilter(min_adx=10)
    assessment = engine.assess(190, Side.LONG, bars, bars, intraday, intraday)
    assert assessment.allowed
    assert assessment.hourly["aligned"] is True
    assert assessment.daily["aligned"] is True
    counter_trend = engine.assess(190, Side.SHORT, bars, bars, intraday, intraday)
    assert not counter_trend.allowed
    assert "opposes" in counter_trend.reason


def test_mtf_filter_fails_closed_when_200_bar_history_is_missing():
    bars = mk_bars(50)
    result = MultiTimeframeFilter().assess(110, Side.LONG, bars, bars, bars, bars)
    assert not result.allowed
    assert "insufficient" in result.reason


def test_warmup_rejects_without_crashing():
    st = SymbolState("SPY")
    d = AlphaEngine(RiskEngine(RiskLimits())).evaluate(st, 100_000)
    assert not d.take and "warming" in d.reject_reason


def test_tape_side_classification_and_bars():
    st = SymbolState("SPY")
    for i in range(5):
        st.on_trade(Trade(T0 + timedelta(seconds=i), "SPY", 100 + i * 0.01, 100, 0))
    assert st.agg.bars["1m"][-1].volume == 500
    assert st.agg.bars["1m"][-1].buy_volume > 0
