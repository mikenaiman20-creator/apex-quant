import json

import app as dashboard_ui


def test_rag_indexes_only_selected_local_sources_and_redacts_credentials(
    monkeypatch, tmp_path,
):
    source = tmp_path / "source"
    logs = tmp_path / "logs"
    data = tmp_path / "data"
    source.mkdir()
    logs.mkdir()
    data.mkdir()
    (source / "risk_gate.py").write_text(
        'def authorize():\n    return "CPCV requires a passing DSR"\n'
        'API_KEY="secret-value"\n'
        'APCA_API_KEY_ID="alpaca-secret-value"\n',
        encoding="utf-8",
    )
    (logs / "strategy_audit_report.json").write_text(json.dumps({
        "status": "blocked",
        "reason": "DSR probability below threshold",
    }), encoding="utf-8")
    (logs / ".env").write_text("API_KEY=never-index-this", encoding="utf-8")
    monkeypatch.setattr(dashboard_ui, "SOURCE_ROOT", source)
    monkeypatch.setattr(dashboard_ui, "LOGS", logs)
    monkeypatch.setattr(dashboard_ui, "DATA", data)

    paths = dashboard_ui._rag_paths()
    documents = dashboard_ui._load_rag_documents(
        tuple(str(path) for path in paths),
        dashboard_ui._rag_fingerprint(paths),
    )
    matches = dashboard_ui._retrieve_rag_documents("CPCV DSR promotion", documents)

    assert matches
    assert matches[0]["source"] == "risk_gate.py"
    assert "secret-value" not in "\n".join(item["content"] for item in documents)
    assert "alpaca-secret-value" not in "\n".join(item["content"] for item in documents)
    assert all(".env" not in item["source"] for item in documents)


def test_rag_requires_explicit_local_ollama_and_rejects_remote_endpoints(monkeypatch):
    monkeypatch.delenv("APEX_UI_OLLAMA_URL", raising=False)
    answer, error = dashboard_ui._local_ollama_answer("question", "evidence")
    assert answer is None
    assert "not configured" in error

    monkeypatch.setenv("APEX_UI_OLLAMA_URL", "https://example.com")
    answer, error = dashboard_ui._local_ollama_answer("question", "evidence")
    assert answer is None
    assert "local endpoint" in error


def test_runtime_asset_questions_use_authoritative_runner_heartbeat(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "symbols": ["SPY", "QQQ", "NVDA"],
        "crypto_symbols": ["BTC/USD", "ETH/USD"],
        "message": "Crypto research is active; order routing disabled.",
    }), encoding="utf-8")
    monkeypatch.setattr(dashboard_ui, "LOGS", logs)

    question = "Which assets are monitored, and are crypto orders enabled?"
    evidence = dashboard_ui._runtime_status_document()

    assert dashboard_ui._is_runtime_status_question(question)
    assert evidence is not None
    assert "SPY, QQQ, NVDA" in evidence["content"]
    assert "BTC/USD, ETH/USD" in evidence["content"]
    assert "crypto order routing is disabled" in evidence["content"]
    assert evidence["source"] == "logs/runner_heartbeat.json"
