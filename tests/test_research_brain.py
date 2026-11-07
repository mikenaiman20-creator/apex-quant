import io
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from server.ingestion.brain import (
    KnowledgeBrain,
    compile_research_variant,
    evaluate_order_routing_eligibility,
)
from server.ingestion.filter import (
    GeminiUnavailable,
    OllamaUnavailable,
    extract_strategy,
    validate_extraction,
)
from server.ingestion.sources import (
    ResearchDocument,
    SourcesUnavailable,
    _parse_vtt,
    fetch_feed,
    fetch_web_article,
    harvest,
    parse_feed,
)
from server.ingestion.worker import ResearchIngestionWorker
from server.ingestion import cli as ingestion_cli


def _extraction(source_url="https://example.org/strategy"):
    return {
        "title": "ATR breakout",
        "category": "stocks",
        "rules": {
            "indicators": ["ATR(14)", "volume > 1.5x average"],
            "entry_condition": "Close breaks above opening range high by 0.1%.",
            "exit_condition": "Exit at 1.25 ATR stop or 2R target.",
            "stop_loss_atr": 1.25,
            "take_profit_mult": 2.0,
        },
        "conviction_score": 0.82,
        "source_url": source_url,
    }


def test_rss_parser_extracts_rss_and_atom_entries_without_markup():
    rss = b"""<rss><channel><item><title>ATR Rules</title>
      <link>https://example.org/atr</link>
      <description><![CDATA[<p>Use ATR(14).</p><script>secret()</script>]]></description>
      </item></channel></rss>"""
    document = parse_feed(rss, "https://example.org/feed")[0]
    assert document.title == "ATR Rules"
    assert document.source_url == "https://example.org/atr"
    assert "ATR(14)" in document.content
    assert "secret" not in document.content

    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <title>Trend Rules</title><link href="https://example.org/trend"/>
      <summary>Use SMA(20).</summary></entry></feed>"""
    atom_document = parse_feed(atom, "https://example.org/atom")[0]
    assert atom_document.source_url == "https://example.org/trend"
    assert atom_document.content == "Use SMA(20)."


def test_article_fetch_and_youtube_transcript_parsing_are_bounded(monkeypatch):
    class Response(io.BytesIO):
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        "server.ingestion.sources.urlopen",
        lambda *_args, **_kwargs: Response(
            b"<html><script>ignore me</script><main>Quant rules <b>here</b></main></html>"
        ),
    )
    doc = ResearchDocument("title", "https://example.org/article", "excerpt", "rss")
    article = fetch_web_article(doc)
    assert article is not None
    assert article.source_type == "web_article"
    assert article.content == "Quant rules here"
    assert _parse_vtt(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nATR(14) breakout\n"
        "00:00:02.000 --> 00:00:03.000\nATR(14) breakout"
    ) == "ATR(14) breakout"


def test_harvest_logs_unavailable_sources_and_raises_if_all_fail(monkeypatch):
    monkeypatch.setattr(
        "server.ingestion.sources.fetch_feed",
        lambda _url: (_ for _ in ()).throw(TimeoutError("network timeout")),
    )
    with pytest.raises(RuntimeError, match="All configured research sources"):
        harvest(["https://example.org/feed"], [])


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/feed.xml", "http://localhost/feed.xml", "http://127.0.0.1/feed.xml"],
)
def test_research_source_urls_reject_nonpublic_targets(url):
    with pytest.raises(ValueError, match="Unsupported research source URL"):
        fetch_feed(url)


def test_extraction_schema_validation_normalizes_category_and_source_url():
    result = validate_extraction(_extraction("https://wrong.example/"), "https://source.example/a")
    assert result["storage_category"] == "stock"
    assert result["category"] == "stocks"
    assert result["source_url"] == "https://source.example/a"
    assert result["rules"]["stop_loss_atr"] == 1.25

    invalid = _extraction()
    invalid["rules"]["take_profit_mult"] = float("nan")
    with pytest.raises(ValueError, match="safe supported ranges"):
        validate_extraction(invalid, invalid["source_url"])

    invalid = _extraction()
    invalid["rules"]["unreviewed_field"] = "ignored"
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_extraction(invalid, invalid["source_url"])


def test_extraction_numeric_fields_accept_numeric_strings_and_reject_invalid_values():
    extraction = _extraction()
    extraction["conviction_score"] = "0.82"
    extraction["rules"]["stop_loss_atr"] = "1.25"
    extraction["rules"]["take_profit_mult"] = "2.0"

    result = validate_extraction(extraction, extraction["source_url"])
    assert result["conviction_score"] == 0.82
    assert result["rules"]["stop_loss_atr"] == 1.25
    assert result["rules"]["take_profit_mult"] == 2.0

    extraction["rules"]["stop_loss_atr"] = "not-a-number"
    with pytest.raises(
        ValueError,
        match="Conviction, stop loss, and take profit must be numeric",
    ):
        validate_extraction(extraction, extraction["source_url"])

    extraction["rules"]["stop_loss_atr"] = "1.25 ATR"
    extraction["rules"]["take_profit_mult"] = "2R"
    result = validate_extraction(extraction, extraction["source_url"])
    assert result["rules"]["stop_loss_atr"] == 1.25
    assert result["rules"]["take_profit_mult"] == 2.0

    extraction["rules"]["take_profit_mult"] = "2 ATR"
    with pytest.raises(ValueError, match="must be numeric"):
        validate_extraction(extraction, extraction["source_url"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("entry_condition", "Enter when momentum looks strong.", "numerical threshold"),
        ("exit_condition", "Exit when the trend weakens.", "numerical threshold"),
        ("indicators", ["ATR"], "numerical periods"),
        ("conviction_score", 0.79, "Logical rigor"),
    ],
)
def test_extraction_gate_rejects_vague_or_low_rigor_rules(field, value, message):
    extraction = _extraction()
    if field == "indicators":
        extraction["rules"][field] = value
    elif field in {"entry_condition", "exit_condition"}:
        extraction["rules"][field] = value
    else:
        extraction[field] = value
    with pytest.raises(ValueError, match=message):
        validate_extraction(extraction, extraction["source_url"])


def test_ollama_extractor_uses_loopback_json_mode_and_validates_reply(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response_payload = {"response": json.dumps(_extraction())}
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr("server.ingestion.filter.urllib.request.urlopen", fake_urlopen)
    document = ResearchDocument("A useful article", "https://example.org/a", "rules", "rss")
    result = extract_strategy(document)
    request, timeout = calls[0]
    body = json.loads(request.data)
    assert result["source_url"] == document.source_url
    assert body["format"] == "json"
    assert body["keep_alive"] == "10m"
    assert timeout == 45.0
    assert request.full_url == "http://127.0.0.1:11434/api/generate"


@pytest.mark.parametrize(
    "url", ["https://example.org", "http://localhost:11434", "http://[::1]:11434"],
)
def test_ollama_extractor_rejects_non_ipv4_loopback_endpoint(monkeypatch, url):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("APEX_OLLAMA_URL", url)
    document = ResearchDocument("title", "https://example.org/a", "text", "rss")
    with pytest.raises(ValueError, match="IPv4 loopback"):
        extract_strategy(document)


def test_knowledge_brain_deduplicates_ranks_and_tracks_status(tmp_path: Path):
    brain = KnowledgeBrain(tmp_path / "knowledge_brain.sqlite")
    first = brain.add_strategy(validate_extraction(_extraction(), "https://example.org/a"))
    assert first is not None
    assert brain.add_strategy(validate_extraction(_extraction(), "https://example.org/a")) is None

    higher = _extraction("https://example.org/b")
    higher["conviction_score"] = 0.95
    second = brain.add_strategy(validate_extraction(higher, higher["source_url"]))
    assert second is not None
    assert [item["id"] for item in brain.top_untested()] == [second, first]

    brain.set_status(second, "rejected")
    assert [item["id"] for item in brain.top_untested()] == [first]
    assert brain.summary() == {
        "untested": 1, "simulating": 0, "promoted": 0, "rejected": 1,
        "sources": 0, "strategies": 2,
    }
    assert brain.get_stats() == brain.summary()
    with sqlite3.connect(brain.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_asset_routing_eligibility_requires_sample_performance_and_rigor(tmp_path: Path):
    brain = KnowledgeBrain(tmp_path / "knowledge_brain.sqlite")
    extraction = validate_extraction(_extraction(), "https://example.org/crypto")
    extraction["category"] = "crypto"
    extraction["storage_category"] = "crypto"
    strategy_id = brain.add_strategy(extraction)
    assert strategy_id is not None
    assert not evaluate_order_routing_eligibility(brain, "crypto")

    brain.record_simulation(strategy_id, 19, 14, 150.0, 100.0)
    assert not evaluate_order_routing_eligibility(brain, "crypto")

    brain.record_simulation(strategy_id, 20, 11, 150.0, 100.0)
    stats = brain.get_asset_performance_stats("crypto")
    assert stats["total_trades"] == 20
    assert stats["win_rate"] == pytest.approx(0.55)
    assert stats["profit_factor"] == pytest.approx(1.5)
    assert evaluate_order_routing_eligibility(brain, "crypto")
    assert brain.top_untested() == []
    assert brain.get_stats()["simulating"] == 1
    assert not evaluate_order_routing_eligibility(brain, "forex")
    assert not brain.has_promoted_strategy("crypto")
    brain.promote_strategy(strategy_id)
    assert brain.has_promoted_strategy("crypto")
    with pytest.raises(ValueError, match="Only a simulating strategy"):
        brain.promote_strategy(strategy_id)


def test_rejected_extraction_is_persisted_without_overwriting_existing_strategy(tmp_path: Path):
    brain = KnowledgeBrain(tmp_path / "knowledge_brain.sqlite")
    extraction = validate_extraction(_extraction(), "https://example.org/rejected")
    strategy_id = brain.add_strategy(extraction)
    assert strategy_id is not None
    brain.record_simulation(strategy_id, 20, 12, 150.0, 100.0)

    brain.reject_strategy(
        "https://example.org/rejected", "Ambiguous rules", "universal",
        "No explicit numerical entry rule.",
    )
    assert brain.get_stats()["simulating"] == 1
    assert brain.get_stats()["rejected"] == 0

    rejected_id = brain.reject_strategy(
        "https://example.org/vague", "Vague rules", "universal",
        "No explicit numerical entry rule.",
    )
    assert rejected_id != strategy_id
    assert brain.get_stats()["rejected"] == 1
    assert evaluate_order_routing_eligibility(brain, "stock")


@pytest.mark.parametrize(
    ("wins", "gross_profit", "gross_loss"),
    [(10, 200.0, 100.0), (15, 100.0, 100.0)],
)
def test_asset_routing_gate_rejects_bad_win_rate_or_profit_factor(
    tmp_path, wins, gross_profit, gross_loss,
):
    brain = KnowledgeBrain(tmp_path / "knowledge_brain.sqlite")
    extraction = validate_extraction(_extraction(), "https://example.org/weak")
    extraction["category"] = "crypto"
    extraction["storage_category"] = "crypto"
    strategy_id = brain.add_strategy(extraction)
    assert strategy_id is not None
    brain.record_simulation(strategy_id, 20, wins, gross_profit, gross_loss)
    assert not evaluate_order_routing_eligibility(brain, "crypto")


def test_compiled_python_variant_is_literal_data_and_execution_disabled(tmp_path: Path):
    candidate = {
        "id": 7,
        "strategy": _extraction(),
    }
    artifact = compile_research_variant(candidate, tmp_path)
    source = artifact.read_text(encoding="utf-8")
    compile(source, str(artifact), "exec")
    assert artifact.name == "variant_7.py"
    assert "EXECUTION_ENABLED = False" in source
    assert "SIMULATION_STATUS = 'untested'" in source


def test_ingestion_worker_persists_state_and_research_artifacts(tmp_path: Path, monkeypatch):
    document = ResearchDocument(
        "ATR breakout", "https://example.org/atr", "transient source text", "rss",
    )
    monkeypatch.setattr("server.ingestion.worker.harvest", lambda *_args: [document])
    monkeypatch.setattr(
        "server.ingestion.worker.extract_strategy",
        lambda _document: validate_extraction(_extraction(), document.source_url),
    )
    worker = ResearchIngestionWorker(tmp_path, interval_seconds=60, feed_urls=[], video_urls=[])
    first = worker.run_once()
    second = worker.run_once()

    assert first["new_strategy_extractions"] == 1
    assert first["knowledge_brain"]["strategies"] == 1
    assert first["generated_variants"] == ["data\\knowledge_variants\\variant_1.py"]
    assert first["execution_enabled"] is False
    assert second["new_strategy_extractions"] == 0
    assert second["duplicate_strategies"] == 1
    assert (tmp_path / "data" / "knowledge_brain.sqlite").is_file()
    assert (tmp_path / "logs" / "research_brain_state.json").is_file()


def test_ingestion_worker_persists_extraction_gate_failures_as_rejected(
    tmp_path: Path, monkeypatch,
):
    document = ResearchDocument(
        "Vague strategy", "https://example.org/vague", "generic source text", "rss",
    )
    monkeypatch.setattr("server.ingestion.worker.harvest", lambda *_args: [document])
    monkeypatch.setattr("server.ingestion.worker.extract_strategy", lambda _document: None)
    worker = ResearchIngestionWorker(tmp_path, interval_seconds=60, feed_urls=[], video_urls=[])

    state = worker.run_once()

    assert state["filtered_sources"] == 1
    assert state["knowledge_brain"]["rejected"] == 1
    assert state["knowledge_brain"]["untested"] == 0
    assert worker.brain.top_untested() == []
    with sqlite3.connect(worker.brain.path) as connection:
        assert connection.execute(
            "SELECT extraction_status FROM ingestion_sources "
            "WHERE source_url = ?", (document.source_url,),
        ).fetchone()[0] == "rejected"


def test_ingestion_worker_retries_failures_with_exponential_backoff(tmp_path, monkeypatch):
    worker = ResearchIngestionWorker(tmp_path, interval_seconds=60, feed_urls=[], video_urls=[])
    monkeypatch.setattr(
        worker, "run_once",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated source failure")),
    )
    monkeypatch.setattr("server.ingestion.worker.random.random", lambda: 0.0)

    class StopAfterWait:
        def __init__(self):
            self.delays = []

        def is_set(self):
            return False

        def wait(self, delay):
            self.delays.append(delay)
            return True

    stop = StopAfterWait()
    worker.run(stop)
    assert stop.delays == [120]
    state = json.loads((tmp_path / "logs" / "research_brain_state.json").read_text())
    assert state["status"] == "error"
    assert state["health_status"] == "error"
    assert state["execution_enabled"] is False


def test_ingestion_worker_uses_configured_cloud_schedule(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_DEPLOYMENT", "true")
    monkeypatch.setenv("INGESTION_SCHEDULE_MINUTES", "180")
    worker = ResearchIngestionWorker(tmp_path, feed_urls=[], video_urls=[])
    assert worker.interval_seconds == 10_800
    assert worker.cloud_deployment is True


def test_scheduled_ingestion_cli_loads_env_and_reports_failure(monkeypatch, caplog):
    calls = []

    class FakeWorker:
        def __init__(self, root):
            calls.append(("worker", root))

        def run_once(self):
            raise RuntimeError("simulated ingestion failure")

    monkeypatch.setattr(ingestion_cli, "ROOT", Path("test-root"))
    monkeypatch.setattr(ingestion_cli, "load_project_env", lambda root: calls.append(
        ("env", root)
    ))
    monkeypatch.setattr(ingestion_cli, "ResearchIngestionWorker", FakeWorker)
    assert ingestion_cli.main() == 1
    assert "Scheduled research ingestion failed." in caplog.text
    assert calls == [
        ("env", Path("test-root")),
        ("worker", Path("test-root")),
    ]


def test_scheduled_ingestion_cli_fails_when_no_documents_were_harvested(
    monkeypatch, caplog,
):
    class EmptyWorker:
        def __init__(self, _root):
            pass

        def run_once(self):
            return {"sources_harvested": 0, "new_strategy_extractions": 0}

    monkeypatch.setattr(ingestion_cli, "load_project_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ingestion_cli, "ResearchIngestionWorker", EmptyWorker)
    assert ingestion_cli.main() == 1
    assert "No research documents were harvested" in caplog.text


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (GeminiUnavailable("Gemini offline"), "gemini_unavailable"),
        (OllamaUnavailable("Ollama offline"), "ollama_unavailable"),
        (SourcesUnavailable("RSS offline"), "sources_unavailable"),
    ],
)
def test_ingestion_worker_reports_missing_dependencies_as_idle_and_retries(
    tmp_path, monkeypatch, failure, reason,
):
    worker = ResearchIngestionWorker(tmp_path, interval_seconds=60, feed_urls=[], video_urls=[])
    monkeypatch.setattr(
        worker, "run_once",
        lambda: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr("server.ingestion.worker.random.random", lambda: 0.0)

    class StopAfterWait:
        waited = False

        def is_set(self):
            return self.waited

        def wait(self, delay):
            assert delay == 120
            self.waited = True
            return True

    worker.run(StopAfterWait())
    state = json.loads((tmp_path / "logs" / "research_brain_state.json").read_text())
    assert state["status"] == "idle"
    assert state["health_status"] == "waiting_for_dependency"
    assert state["reason"] == reason
    assert state["retrying"] is True
    assert state["knowledge_brain"]["strategies"] == 0
    assert state["execution_enabled"] is False


def test_extractor_service_outage_does_not_reject_strategy_source(tmp_path, monkeypatch):
    worker = ResearchIngestionWorker(tmp_path, feed_urls=[], video_urls=[])
    document = ResearchDocument("candidate", "https://example.org/candidate", "rules", "rss")
    monkeypatch.setattr("server.ingestion.worker.harvest", lambda *_args: [document])
    monkeypatch.setattr(
        "server.ingestion.worker.extract_strategy",
        lambda _document: (_ for _ in ()).throw(GeminiUnavailable("Gemini offline")),
    )
    with pytest.raises(GeminiUnavailable, match="every harvested source"):
        worker.run_once()
    assert worker.brain.get_stats()["rejected"] == 0
    with sqlite3.connect(worker.brain.path) as connection:
        assert connection.execute(
            "SELECT extraction_status FROM ingestion_sources WHERE source_url = ?",
            (document.source_url,),
        ).fetchone()[0] == "unavailable"
