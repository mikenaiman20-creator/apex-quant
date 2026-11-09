import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server.dashboard import DashboardState, _jarvis_bullets, create_server
from server.evolution import load_trial_ledger, register_trials


def test_dashboard_survives_missing_logs_profile_and_api(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "research_brain_state.json").write_text(json.dumps({
        "status": "completed",
        "knowledge_brain": {"strategies": 3, "untested": 3},
        "generated_variants": [],
        "execution_enabled": False,
    }), encoding="utf-8")
    state = DashboardState(tmp_path, client_factory=lambda: (_ for _ in ()).throw(
        RuntimeError("paper API offline")
    ))
    data = state.snapshot()
    assert data["runner"]["status"] == "disconnected"
    assert data["profile"] is None
    assert data["account"]["connected"] is False
    assert data["account"]["error"]["message"] == "paper API offline"
    assert data["positions"] == []
    assert data["orders"] == []
    assert data["activity"] == []
    assert data["research_brain"]["knowledge_brain"]["strategies"] == 3
    assert 0 <= data["readiness_score"] <= 100
    assert data["readiness_status"] in {
        "UNSAFE", "PAPER STABLE", "LIVE CANDIDATE", "FULL DEPLOYMENT",
    }
    assert "risk_guardrails" in data["readiness_breakdown"]


def test_dashboard_loads_standard_alpaca_keys_from_absolute_env_path(
    monkeypatch, tmp_path,
):
    (tmp_path / ".env").write_text(
        "ALPACA_API_KEY=dashboard-test-key\n"
        "ALPACA_API_SECRET=dashboard-test-secret\n"
        "ALPACA_DATA_FEED=sip\n",
        encoding="utf-8",
    )
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_DATA_FEED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path.parent)
    received = {}

    def fake_client(key, secret, feed):
        received.update(key=key, secret=secret, feed=feed)
        return "client"

    monkeypatch.setattr("server.dashboard.AlpacaClient", fake_client)
    assert DashboardState(tmp_path)._make_client() == "client"
    assert received == {
        "key": "dashboard-test-key",
        "secret": "dashboard-test-secret",
        "feed": "sip",
    }


def test_trial_ledger_repairs_invalid_and_legacy_schemas(tmp_path: Path):
    ledger_path = tmp_path / "logs" / "strategy_trial_ledger.json"
    ledger_path.parent.mkdir()
    ledger_path.write_text('{"cycles": "broken", "total_trials": "many"}', encoding="utf-8")

    repaired = load_trial_ledger(ledger_path)
    persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert repaired == {
        "version": 1, "cycles": {}, "num_trials": 0, "total_trials": 0,
    }
    assert persisted == repaired

    ledger_path.write_text('{"total_trials": 12}', encoding="utf-8")
    legacy = load_trial_ledger(ledger_path)
    assert legacy["num_trials"] == 12
    assert legacy["total_trials"] == 12
    assert legacy["cycles"] == {"legacy_import": 12}
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == legacy
    assert register_trials("new-cycle", 3, ledger_path) == 15


def test_dashboard_normalizes_corrupt_trial_ledger_without_failing(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "strategy_trial_ledger.json").write_text("{ invalid json", encoding="utf-8")

    data = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    assert data["trial_ledger"] == {
        "version": 1, "cycles": {}, "num_trials": 0, "total_trials": 0,
    }
    assert json.loads((logs / "strategy_trial_ledger.json").read_text(encoding="utf-8")) == data["trial_ledger"]


def test_dashboard_normalizes_legacy_ollama_error_to_idle_retry_status(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "research_brain_state.json").write_text(json.dumps({
        "status": "error",
        "error": "OllamaUnavailable: Ollama was unavailable for every harvested research source.",
        "knowledge_brain": {"strategies": 0, "sources": 7},
    }), encoding="utf-8")
    data = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()

    assert data["research_brain"]["status"] == "idle"
    assert data["research_brain"]["health_status"] == "waiting_for_dependency"
    assert data["research_brain"]["retrying"] is True
    assert data["research_brain"]["knowledge_brain"]["strategies"] == 0


def test_jarvis_bullets_are_bounded_deduplicated_and_single_honorific():
    summary = _jarvis_bullets([
        ("Market", "Market is closed, Sir."),
        ("Duplicate", "Market is closed."),
        ("Researcher", "Testing variants. Sir. Sir."),
        ("Trials", "20 trials"),
        ("Staged", "None"),
        ("Ignored", "This fifth point is omitted"),
    ])
    lines = summary.splitlines()
    assert len(lines) == 4
    assert sum("Market is closed" in line for line in lines) == 1
    assert all(line.startswith("• **") and line.count("Sir.") == 1 for line in lines)
    assert "Sir. Sir." not in summary


def test_dashboard_reads_runner_profile_activity_and_read_only_account(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": 42,
        "status": "waiting",
        "message": "Market closed",
        "symbols": ["AAA", "BBB"],
    }), encoding="utf-8")
    (logs / "daily_profile.json").write_text(json.dumps({
        "params": {"min_signal_score": 0.75},
        "universe": ["AAA", "BBB"],
    }), encoding="utf-8")
    (logs / "autonomous.log").write_text(
        '{"timestamp":"2026-10-28T15:00:00Z","level":"INFO","message":"signal check"}\n'
        "legacy log message\n",
        encoding="utf-8",
    )

    class FakeClient:
        def account(self):
            return {"equity": "1000", "cash": "800", "buying_power": "1000"}

        def positions(self):
            return [{"symbol": "AAA", "qty": "2"}]

        def recent_orders(self, limit):
            assert limit == 20
            return [{"symbol": "AAA", "status": "filled"}]

    data = DashboardState(tmp_path, client_factory=FakeClient).snapshot()
    assert data["runner"]["status"] == "waiting"
    assert data["symbols"] == ["AAA", "BBB"]
    assert data["profile"]["params"]["min_signal_score"] == 0.75
    assert data["account"]["connected"] is True
    assert data["positions"][0]["symbol"] == "AAA"
    assert data["orders"][0]["status"] == "filled"
    assert [event["message"] for event in data["activity"]] == ["legacy log message", "signal check"]


def test_dashboard_api_and_index_are_served_on_localhost():
    state = DashboardState(client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/", timeout=2) as response:
            assert response.status == 200
            assert b"ApexQuant Monitor" in response.read()
            assert response.headers.get("Access-Control-Allow-Origin") is None
        with urlopen(base + "/api/state", timeout=2) as response:
            payload = json.load(response)
            assert response.status == 200
            assert response.headers.get("Access-Control-Allow-Origin") is None
            assert payload["account"]["connected"] is False
            assert isinstance(payload["readiness_score"], int)
            assert payload["readiness_status"] == "UNSAFE"
            assert "missing_criteria" in payload["readiness_breakdown"]
        try:
            urlopen(Request(base + "/api/state", method="OPTIONS"), timeout=2)
            raise AssertionError("Cross-origin preflight unexpectedly succeeded.")
        except HTTPError as exc:
            assert exc.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_renders_mobile_readiness_ring_without_changing_refresh_interval():
    html = (Path(__file__).parents[1] / "server" / "dashboard.html").read_text(encoding="utf-8")
    assert 'id="readiness-ring"' in html
    assert 'id="readiness-score"' in html
    assert 'id="readiness-status"' in html
    assert 'style.setProperty("--readiness-progress"' in html
    assert "setInterval(refresh, 3000)" in html
    assert 'id="jarvis-log-container"' in html
    assert 'fetch("/api/jarvis"' in html
    assert "JSON.stringify({prompt: promptText})" in html
    assert 'payload.reply' in html
    assert "payload.answer" in html
    assert "new AbortController()" in html
    assert "Preparing a concise answer" in html


def test_jarvis_offline_diagnostic_and_legacy_researcher_answer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("server.dashboard.query_local_llm", lambda _context, _prompt: None)
    logs = tmp_path / "logs"
    logs.mkdir()
    state = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    staging = tmp_path / "data" / "staging"
    staging.mkdir(parents=True)
    (staging / "candidate-jarvis.json").write_text(json.dumps({
        "candidate_id": "candidate-jarvis",
        "status": "staged_candidate",
        "human_approved": False,
    }), encoding="utf-8")
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        (logs / "research_agent_state.json").write_text(json.dumps({
            "status": "evaluating_variants",
            "current_task": "Running cost-adjusted walk-forward tests",
            "symbols": ["SPY", "QQQ"],
            "planned_variants": 4,
            "variants_tested": 2,
            "message": "Testing updated parameter sets, Sir.",
        }), encoding="utf-8")
        (logs / "strategy_trial_ledger.json").write_text(json.dumps({
            "version": 1,
            "cycles": {"cycle-a": 11},
            "total_trials": 11,
        }), encoding="utf-8")
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/jarvis/?t=123",
            data=json.dumps({
                "prompt": "How about the researcher, what is he doing right now?"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            answer = json.load(response)["reply"]
            assert response.status == 200
        assert "**Local Ollama Unavailable**" in answer
        assert "offline or unresponsive" in answer
        assert "ollama run llama3" in answer

        legacy_request = Request(
            f"http://127.0.0.1:{server.server_port}/api/ask",
            data=json.dumps({
                "question": "How about the researcher, what is he doing right now?"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(legacy_request, timeout=2) as response:
            legacy_answer = json.load(response)["answer"]
        assert "**Researcher:** evaluating variants" in legacy_answer
        assert "Running cost-adjusted walk-forward tests" in legacy_answer
        assert "2/4 variants" in legacy_answer
        assert "11 cumulative unique trials" in legacy_answer
        assert "candidate-jarvis" in legacy_answer
        assert legacy_answer.count("\n") <= 3
        assert all(line.startswith("• **") for line in legacy_answer.splitlines())
        assert legacy_answer.endswith("Sir.")
        assert "Sir. Sir." not in legacy_answer

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_jarvis_endpoint_uses_local_llm_with_cached_live_context(tmp_path: Path, monkeypatch):
    state = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "message": "Scanning",
        "symbols": ["SPY"],
        "telemetry": {
            "symbol": "SPY",
            "score": 0.42,
            "required_score": 0.7,
            "reason": "score below threshold",
        },
    }), encoding="utf-8")
    (tmp_path / "logs" / "strategy_trial_ledger.json").write_text(json.dumps({
        "version": 1,
        "cycles": {"cycle-a": 158},
        "total_trials": 158,
    }), encoding="utf-8")
    state.snapshot()
    calls = []

    def fake_llm(system_prompt, user_query):
        calls.append((system_prompt, user_query))
        return "**Status:** Scanning SPY.\n- **Safety:** Paper-only, Sir."

    monkeypatch.setattr("server.dashboard.query_local_llm", fake_llm)
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/jarvis/?cache=1",
            data=json.dumps({"prompt": "Summarize live readiness"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        assert payload["reply"] == payload["answer"]
        assert "**Status:** Scanning SPY. Sir." in payload["reply"]
        assert "**Safety:** Paper-only. Sir." in payload["reply"]
        assert len(calls) == 1
        assert calls[0][1] == "Summarize live readiness"
        assert "Equity:" in calls[0][0]
        assert "158" in calls[0][0]
        assert "paper-only" in calls[0][0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_llm_keeps_model_warm_and_limits_generation_for_quick_replies(monkeypatch):
    import io

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((json.loads(request.data), timeout))
        return io.BytesIO(json.dumps({"response": "• **Answer:** Ready, Sir."}).encode())

    monkeypatch.setattr("server.dashboard.urllib.request.urlopen", fake_urlopen)
    from server.dashboard import query_local_llm

    assert query_local_llm("Answer directly.", "What is the status?") == "• **Answer:** Ready, Sir."
    payload, timeout = calls[0]
    assert payload["keep_alive"] == "10m"
    assert payload["options"]["num_predict"] == 100
    assert "User Question: What is the status?" in payload["prompt"]
    assert timeout == 90.0


def test_jarvis_live_readiness_and_execution_answers_use_current_telemetry(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "message": "Latest setup rejected",
        "telemetry": {
            "symbol": "NVDA",
            "score": 0.4,
            "required_score": 0.65,
            "accepted": False,
            "reason": "score below required threshold",
        },
    }), encoding="utf-8")
    (logs / "strategy_trial_ledger.json").write_text(json.dumps({
        "version": 1,
        "cycles": {"cycle-a": 158},
        "total_trials": 158,
    }), encoding="utf-8")
    state = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    execution_reply = state.answer_question("When will it open a position?")
    assert "0.40" in execution_reply
    assert "0.65" in execution_reply
    assert "paper-only" in execution_reply
    assert "score below required threshold" in execution_reply

    readiness_reply = state.answer_question("What is my live readiness score?")
    assert "Readiness" in readiness_reply
    assert "UNSAFE" in readiness_reply
    assert "live orders remain disabled" in readiness_reply

    shadow_reply = state.answer_question("How does shadow testing work?")
    assert "Live scan" in shadow_reply
    assert "158 cumulative trials" in shadow_reply
    assert "paper-only orders" in shadow_reply


def test_dashboard_rejects_non_local_bind():
    try:
        create_server("0.0.0.0", 0)
    except ValueError as exc:
        assert "localhost" in str(exc)
    else:
        raise AssertionError("Dashboard accepted a non-local bind address.")


def test_dashboard_exposes_latest_intraday_scan_telemetry(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "telemetry": {
            "symbol": "NVDA", "phase": "power_hour", "rvol": 3.2,
            "adx": 31.5, "atr": 2.4, "score": 0.78,
            "reward_risk": 2.4, "accepted": False, "reason": "ADX below gate",
        },
    }), encoding="utf-8")
    status = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()["runner"]
    assert status["telemetry"]["symbol"] == "NVDA"
    assert status["telemetry"]["rvol"] == 3.2
    assert status["telemetry"]["reason"] == "ADX below gate"


def test_dashboard_reads_candidate_audit(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "strategy_audit_report.json").write_text(json.dumps({
        "candidate_id": "candidate-1",
        "status": "blocked",
        "n_trials": 100,
        "raw_sharpe": 1.8,
        "deflated_sharpe_probability": 0.72,
        "holdout": {"available": False, "passed": False, "reason": "missing"},
    }), encoding="utf-8")
    data = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    assert data["audit"]["candidate_id"] == "candidate-1"
    assert data["audit"]["n_trials"] == 100


def test_jarvis_reports_runner_and_staged_candidates_without_side_effects(tmp_path: Path):
    logs = tmp_path / "logs"
    staging = tmp_path / "data" / "staging"
    logs.mkdir()
    staging.mkdir(parents=True)
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "symbols": ["SPY", "QQQ"],
        "telemetry": {
            "symbol": "SPY",
            "accepted": False,
            "reason": "No qualifying entry, Sir.",
        },
    }), encoding="utf-8")
    (staging / "candidate-1.json").write_text(json.dumps({
        "candidate_id": "candidate-1",
        "status": "staged_candidate",
        "human_approved": False,
        "paper_trades": 40,
        "holdout": {"passed": True},
    }), encoding="utf-8")
    (staging / "approved.json").write_text(json.dumps({
        "candidate_id": "approved",
        "status": "approved_for_paper",
        "human_approved": True,
    }), encoding="utf-8")
    data = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    assert data["staged_candidates"][0]["candidate_id"] == "candidate-1"
    assert len(data["staged_candidates"]) == 1
    assert "**runner:** active" in data["jarvis"]["status"].lower()
    assert data["jarvis"]["status"].count("\n") <= 3
    assert "staged for your review" in data["jarvis"]["candidates"]
    assert "not active" in data["jarvis"]["candidates"].lower()
    assert all(value.endswith("Sir.") for key, value in data["jarvis"].items()
               if isinstance(value, str)
               if key not in {"read_only_notice"})
    assert "Sir." in data["jarvis"]["read_only_notice"]


def test_jarvis_thoughts_explain_scan_gates_trials_holdout_and_roadmap(tmp_path: Path):
    logs = tmp_path / "logs"
    data_dir = tmp_path / "data"
    logs.mkdir()
    data_dir.mkdir()
    (logs / "runner_heartbeat.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "active",
        "symbols": ["SPY", "QQQ"],
        "telemetry": {
            "symbol": "SPY",
            "phase": "regular",
            "current_price": 520.25,
            "bar_change_pct": 0.12,
            "session_vwap": 519.8,
            "rvol": 1.4,
            "adx": 24.0,
            "atr": 1.25,
            "score": 0.68,
            "side": "long",
            "accepted": False,
            "reason": "MTF filter: hourly trend is not aligned",
            "mtf": {"allowed": False, "reason": "hourly trend is not aligned"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }), encoding="utf-8")
    (logs / "strategy_trial_ledger.json").write_text(json.dumps({
        "version": 1, "cycles": {"cycle-a": 20, "cycle-b": 10}, "total_trials": 30,
    }), encoding="utf-8")
    (logs / "strategy_audit_report.json").write_text(json.dumps({
        "candidate_id": "candidate-a",
        "status": "blocked",
        "n_trials": 30,
        "raw_sharpe": 1.7,
        "deflated_sharpe_probability": 0.82,
        "required_sharpe": 2.4,
        "sharpe_gate_passed": False,
        "basic_gates_passed": True,
        "reason": "Sharpe gate failed",
        "holdout": {"available": False, "passed": False, "reason": "untouched data missing"},
        "expected_slippage_drag": 4.0,
        "simulated_slippage_drag": 5.0,
        "simulated_transaction_costs": 8.0,
    }), encoding="utf-8")
    (data_dir / "macro_events.json").write_text(json.dumps({"events": [{
        "name": "CPI",
        "impact": "high",
        "timestamp": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
    }]}), encoding="utf-8")

    snapshot = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    thoughts = snapshot["jarvis"]["thoughts"]
    assert "SPY" in thoughts["operations"]
    assert "RVOL 1.40x" in thoughts["operations"]
    assert "hourly trend is not aligned" in thoughts["operations"]
    assert "30 trials" in thoughts["why"]
    assert "required Sharpe hurdle is 2.40" in thoughts["why"]
    assert "untouched data missing" in thoughts["why"]
    assert "ATR" in thoughts["why"]
    assert "CPI" in thoughts["roadmap"]
    assert all(value.endswith("Sir.") for value in thoughts.values())
    assert all(
        1 <= len(value.splitlines()) <= 4
        and all(line.startswith("• **") for line in value.splitlines())
        and all(line.count("Sir.") == 1 for line in value.splitlines())
        for value in thoughts.values()
    )
    assert snapshot["trial_ledger"]["total_trials"] == 30


def test_jarvis_explains_empty_telemetry_and_candidate_state(tmp_path: Path):
    data = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    thoughts = data["jarvis"]["thoughts"]
    assert "No per-symbol scan is available" in thoughts["operations"]
    assert "heartbeat found" in thoughts["operations"]
    assert "no audit report" in thoughts["why"].lower()
    assert "None" in thoughts["candidate_reason"]
    assert "No upcoming event listed" in thoughts["roadmap"]
    assert all(value.endswith("Sir.") for value in thoughts.values())


def test_jarvis_reports_background_researcher_progress(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "research_agent_state.json").write_text(json.dumps({
        "status": "completed",
        "current_task": "Monitoring local history for new bars",
        "symbols": ["SPY", "QQQ"],
        "variants_tested": 4,
        "variants": [{
            "params": {"min_signal_score": 0.75},
            "trade_count": 35,
            "profit_factor": 1.4,
            "expectancy_r": 0.22,
        }],
        "candidate": {
            "status": "blocked",
            "reason": "final holdout data is unavailable",
        },
        "execution_enabled": False,
    }), encoding="utf-8")

    snapshot = DashboardState(
        tmp_path, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    ).snapshot()
    thoughts = snapshot["jarvis"]["thoughts"]
    assert "24/7 Researcher completed 4 variants" in thoughts["operations"]
    assert "final holdout data is unavailable" in thoughts["roadmap"]
    assert snapshot["researcher"]["execution_enabled"] is False
    assert all(value.endswith("Sir.") for value in thoughts.values())


def test_macro_outlook_identifies_active_pre_release_pause(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "macro_events.json").write_text(json.dumps({"events": [{
        "name": "FOMC",
        "impact": "high",
        "timestamp": (datetime.now(timezone.utc) + timedelta(minutes=8)).isoformat(),
    }]}), encoding="utf-8")
    result = DashboardState(tmp_path)._macro_outlook()
    assert result["next_event"]["name"] == "FOMC"
    assert result["next_event"]["pause_active"] is True
    assert "pre-release pause is active now" in result["message"]
    assert result["message"].endswith("Sir.")
