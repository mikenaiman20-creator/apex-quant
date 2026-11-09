import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _app_test(path: Path):
    testing = pytest.importorskip("streamlit.testing.v1")
    return testing.AppTest.from_file(str(path)).run(timeout=30)


def test_streamlit_dashboard_shows_sections_even_without_telemetry(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.setenv("APEX_DATA_ROOT", str(root / "missing-production-data"))
    monkeypatch.setenv("APEX_SOURCE_ROOT", str(root / "missing-production-source"))
    app = _app_test(root / "app.py")

    assert not app.exception
    assert [item.value for item in app.title] == ["Apex Quant"]
    assert {item.label for item in app.tabs} == {
        "◈ Overview",
        "⚡ Execution Engine",
        "⌕ Research & Strategy Arsenal",
        "◉ CPCV / DSR Validation",
        "✦ Apex Quant Assistant (RAG)",
        "☷ Activity",
    }
    assert any("Production telemetry is not mounted" in item.value for item in app.error)
    assert any(
        item.value.startswith("One or more project modules are missing")
        for item in app.warning
    )


def test_streamlit_dashboard_uses_placeholders_for_unpublished_snapshots(
    monkeypatch, tmp_path,
):
    root = Path(__file__).parents[1]
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "logs" / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "symbols": ["SPY", "QQQ", "NVDA"],
    }), encoding="utf-8")
    source = tmp_path / "source"
    for module in ("backtest", "config", "engine", "server"):
        (source / module).mkdir(parents=True)
    monkeypatch.setenv("APEX_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("APEX_SOURCE_ROOT", str(source))
    app = _app_test(root / "app.py")

    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert any("snapshots have not been published yet" in item.value for item in app.info)
    assert any("No daily strategy profile is available." in item.value for item in app.info)
    assert any("No researcher state has been published." in item.value for item in app.info)
    assert any("No strategy audit report is available." in item.value for item in app.info)


def test_streamlit_rag_chat_returns_local_evidence_when_ollama_is_unavailable(
    monkeypatch, tmp_path,
):
    root = Path(__file__).parents[1]
    logs = tmp_path / "logs"
    data = tmp_path / "data"
    source = tmp_path / "source"
    logs.mkdir()
    data.mkdir()
    for module in ("backtest", "config", "engine", "server"):
        (source / module).mkdir(parents=True)
    (source / "backtest" / "validation.py").write_text(
        "CPCV requires PBO below the configured DSR gate.\n",
        encoding="utf-8",
    )
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "symbols": ["SPY"],
    }), encoding="utf-8")
    monkeypatch.setenv("APEX_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("APEX_SOURCE_ROOT", str(source))
    monkeypatch.delenv("APEX_UI_OLLAMA_URL", raising=False)
    app = _app_test(root / "app.py")
    assert not app.exception
    assert len(app.chat_input) == 1

    app.chat_input[0].set_value("CPCV PBO DSR").run(timeout=30)

    assert not app.exception
    assert any("Ollama is not configured" in item.value for item in app.markdown)
    assert any("validation.py" in item.value for item in app.caption)


def test_streamlit_dashboard_renders_published_paper_research_data(monkeypatch, tmp_path):
    root = tmp_path
    logs = root / "logs"
    data = root / "data"
    source = root / "source"
    logs.mkdir()
    data.mkdir()
    for module in ("backtest", "config", "engine", "server"):
        (source / module).mkdir(parents=True)
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "message": "Market open",
        "symbols": ["SPY"],
        "telemetry": {"symbol": "SPY", "accepted": True, "score": 0.81, "rvol": 2.0},
    }), encoding="utf-8")
    (logs / "paper_dashboard_snapshot.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "PAPER",
        "account": {"equity": "25000", "cash": "12000", "buying_power": "48000"},
        "positions": [{"symbol": "SPY", "qty": "2", "side": "long"}],
        "orders": [{"symbol": "SPY", "side": "buy", "status": "filled"}],
    }), encoding="utf-8")
    (logs / "market_dashboard_snapshot.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": {
            "SPY": {
                "asset_class": "equity",
                "display_symbol": "SPY",
                "price": 600,
                "trend_50d": "above",
                "sma50": 590,
                "atr_pct": 0.5,
                "bbwp": 12,
                "regime": "above 50-day trend · volatility squeeze",
                "candles": [
                    {"time": "2026-11-07T14:30:00+00:00", "close": 599},
                    {"time": "2026-11-07T14:31:00+00:00", "close": 600},
                ],
            },
            "BTC/USD": {
                "asset_class": "crypto",
                "display_symbol": "BTC-USD",
                "price": 90000,
                "regime": "24/7 · order routing disabled",
                "candles": [],
            },
        },
    }), encoding="utf-8")
    (logs / "daily_profile.json").write_text(json.dumps({
        "universe": ["SPY"],
        "params": {"min_signal_score": 0.75},
    }), encoding="utf-8")
    (logs / "research_state.json").write_text(json.dumps({
        "regimes": {"SPY": {"label": "bullish_trend"}},
    }), encoding="utf-8")
    (logs / "research_agent_state.json").write_text(json.dumps({
        "status": "running",
        "variants_tested": 4,
        "planned_variants": 12,
        "symbols": ["SPY"],
    }), encoding="utf-8")
    (logs / "research_brain_state.json").write_text(json.dumps({
        "status": "completed",
        "knowledge_brain": {"strategies": 3, "untested": 2, "sources": 5},
    }), encoding="utf-8")
    (logs / "strategy_audit_report.json").write_text(json.dumps({
        "status": "blocked",
        "n_trials": 12,
        "raw_sharpe": 1.2,
        "deflated_sharpe_probability": 0.4,
        "holdout": {"passed": False, "reason": "missing"},
    }), encoding="utf-8")
    (logs / "strategy_trial_ledger.json").write_text(json.dumps({
        "total_trials": 12,
    }), encoding="utf-8")
    (logs / "autonomous.log").write_text(
        '{"timestamp":"2026-11-07T00:00:00Z","level":"INFO","message":"Filled SPY order"}\n',
        encoding="utf-8",
    )
    (data / "strategy_arsenal.json").write_text(json.dumps({
        "strategies": [{"id": "s1", "status": "research_only"}],
    }), encoding="utf-8")
    (data / "macro_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    connection = sqlite3.connect(data / "knowledge_brain.sqlite")
    connection.executescript("""
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            source_url TEXT NOT NULL,
            category TEXT NOT NULL,
            strategy_json TEXT NOT NULL,
            conviction_score REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE ingestion_sources (
            source_url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            extraction_status TEXT NOT NULL,
            error TEXT
        );
        INSERT INTO strategies VALUES (
            1, 'https://example.test/strategy', 'stock',
            '{"title":"Test strategy"}', 0.82, 'untested', '2026-11-07T00:00:00Z'
        );
        INSERT INTO ingestion_sources VALUES (
            'https://example.test/strategy', 'Test source', 'rss',
            '2026-11-07T00:00:00Z', 'completed', NULL
        );
    """)
    connection.commit()
    connection.close()

    monkeypatch.setenv("APEX_DATA_ROOT", str(root))
    monkeypatch.setenv("APEX_SOURCE_ROOT", str(source))
    app = _app_test(Path(__file__).parents[1] / "app.py")

    assert not app.exception
    assert [item.value for item in app.title] == ["Apex Quant"]
    assert any(item.value == "Market open" for item in app.success)
    assert any(
        item.label == "Paper equity" and item.value == "$25,000.00"
        for item in app.metric
    )
    assert any(item.label == "SQLite integrity" and item.value == "OK" for item in app.metric)
    assert any(item.label == "SPY" and item.value == "$600.00" for item in app.metric)
    assert any(item.label == "BTC-USD" and item.value == "$90,000.00" for item in app.metric)
    assert not any(
        item.value.startswith("One or more project modules are missing")
        for item in app.warning
    )
