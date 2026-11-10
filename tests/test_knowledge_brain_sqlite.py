"""Regression tests for Knowledge Brain SQLite persistence.

These cover the production failure where the dashboard opened a WAL database
through a read-only bind mount and SQLite returned "unable to open database
file" because it could not create the ``-shm`` sidecar.  They also cover the
writer side: parent-directory provisioning, concurrent access, permission
errors, and corrupt-file quarantine.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from server.ingestion import brain as brain_module
from server.ingestion.brain import KnowledgeBrain


def _extraction(source_url: str) -> dict:
    return {
        "title": "ATR breakout",
        "category": "stocks",
        "storage_category": "stock",
        "rules": {
            "indicators": ["ATR(14)"],
            "entry_condition": "Close breaks above the opening range high.",
            "exit_condition": "Exit at 1.25 ATR stop or 2R target.",
            "stop_loss_atr": 1.25,
            "take_profit_mult": 2.0,
        },
        "conviction_score": 0.82,
        "source_url": source_url,
    }


def test_initializes_database_and_creates_missing_parent_directories(tmp_path: Path):
    path = tmp_path / "missing" / "nested" / "state" / "knowledge_brain.sqlite"
    assert not path.parent.exists()

    brain = KnowledgeBrain(path)

    assert path.is_file()
    assert path.parent.is_dir()
    assert brain.summary()["strategies"] == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_concurrent_reads_and_writes_do_not_raise_unable_to_open(tmp_path: Path):
    path = tmp_path / "data" / "knowledge_brain.sqlite"
    KnowledgeBrain(path).record_source(
        "https://example.org/seed", "seed", "rss", "extracted",
    )

    errors: list[BaseException] = []

    def writer(worker: int) -> None:
        try:
            brain = KnowledgeBrain(path)
            for index in range(10):
                brain.record_source(
                    f"https://example.org/{worker}/{index}",
                    f"title-{worker}-{index}",
                    "rss",
                    "extracted",
                )
                brain.summary()
        except BaseException as exc:  # noqa: BLE001 - the assertion reports it
            errors.append(exc)

    def raw_reader() -> None:
        try:
            for _ in range(50):
                connection = sqlite3.connect(path, timeout=30)
                try:
                    connection.execute("SELECT COUNT(*) FROM ingestion_sources").fetchone()
                finally:
                    connection.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(4)]
    threads += [threading.Thread(target=raw_reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, [f"{type(exc).__name__}: {exc}" for exc in errors]
    assert "unable to open database file" not in " ".join(str(exc) for exc in errors).lower()


def test_permission_failure_is_logged_and_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
):
    path = tmp_path / "data" / "knowledge_brain.sqlite"

    def deny(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(brain_module.sqlite3, "connect", deny)
    monkeypatch.setattr(brain_module.time, "sleep", lambda _seconds: None)

    with caplog.at_level(logging.WARNING, logger="apex.research.brain"):
        with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
            KnowledgeBrain(path)

    assert "Knowledge Brain SQLite open failed" in caplog.text
    assert "Unable to open Knowledge Brain database" in caplog.text


def test_corrupt_database_is_quarantined_and_reinitialized(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    path = tmp_path / "data" / "knowledge_brain.sqlite"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a sqlite database")

    with caplog.at_level(logging.WARNING, logger="apex.research.brain"):
        brain = KnowledgeBrain(path)

    assert brain.summary()["strategies"] == 0
    assert path.is_file()
    quarantined = list(path.parent.glob("knowledge_brain.sqlite.corrupt-*"))
    assert quarantined, "the corrupt database must be preserved, not deleted"
    assert "corrupt" in caplog.text.lower()


def test_dashboard_reads_wal_database_through_immutable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("streamlit")
    import app as dashboard_app

    path = tmp_path / "data" / "knowledge_brain.sqlite"
    brain = KnowledgeBrain(path)
    brain.add_strategy(_extraction("https://example.org/strategy"))
    brain.record_source("https://example.org/strategy", "ATR", "rss", "extracted")

    real_connect = sqlite3.connect

    def read_only_mount_connect(database, *args, **kwargs):
        if isinstance(database, str) and "mode=ro" in database and "immutable" not in database:
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(dashboard_app.sqlite3, "connect", read_only_mount_connect)
    dashboard_app._read_knowledge_database.clear()

    result = dashboard_app._read_knowledge_database(
        str(path), path.stat().st_mtime_ns,
    )

    assert "error" not in result, result
    assert result["integrity"] == "ok"
    assert result["source_count"] == 1
    assert len(result["strategies"]) == 1


def test_dashboard_returns_graceful_error_when_all_open_modes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
):
    pytest.importorskip("streamlit")
    import app as dashboard_app

    path = tmp_path / "data" / "knowledge_brain.sqlite"
    KnowledgeBrain(path)
    dashboard_app._read_knowledge_database.clear()

    def deny(*_args, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(dashboard_app.sqlite3, "connect", deny)

    with caplog.at_level(logging.ERROR, logger="apex.dashboard"):
        result = dashboard_app._read_knowledge_database(
            str(path), path.stat().st_mtime_ns,
        )

    assert "error" in result
    assert result["error"].startswith("Knowledge Brain SQLite read failed:")
    assert "Knowledge Brain SQLite read failed" in caplog.text
