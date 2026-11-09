from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from server.ingestion.brain import KnowledgeBrain, compile_research_variant
from server.ingestion.filter import (
    GeminiUnavailable, OllamaUnavailable, extract_strategy,
)
from server.ingestion.sources import ResearchDocument, SourcesUnavailable, harvest
from server.io_utils import atomic_write_json

log = logging.getLogger("apex.research.factory")
DEFAULT_INTERVAL_SECONDS = 180 * 60
MAX_BACKOFF_SECONDS = 3600


class ResearchIngestionWorker:
    def __init__(
        self,
        root: Path,
        interval_seconds: int | None = None,
        feed_urls: Iterable[str] | None = None,
        video_urls: Iterable[str] | None = None,
    ):
        self.root = Path(root)
        configured_interval = (
            int(os.getenv(
                "INGESTION_SCHEDULE_MINUTES", str(DEFAULT_INTERVAL_SECONDS // 60)
            )) * 60
            if interval_seconds is None else int(interval_seconds)
        )
        self.interval_seconds = max(60, configured_interval)
        self.cloud_deployment = os.getenv("CLOUD_DEPLOYMENT", "false").strip().lower() == "true"
        self.feed_urls = tuple(feed_urls) if feed_urls is not None else None
        self.video_urls = tuple(video_urls) if video_urls is not None else None
        self.brain = KnowledgeBrain(self.root / "data" / "knowledge_brain.sqlite")
        self.state_path = self.root / "logs" / "research_brain_state.json"

    def _write_state(self, state: dict) -> None:
        atomic_write_json(self.state_path, state, indent=2)

    def run_once(self) -> dict:
        started = datetime.now(timezone.utc)
        self._write_state({
            "status": "harvesting",
            "health_status": "active",
            "retrying": False,
            "started_at": started.isoformat(),
            "message": "Harvesting configured public research feeds.",
            "execution_enabled": False,
            "cloud_deployment": self.cloud_deployment,
            "schedule_minutes": self.interval_seconds // 60,
        })
        documents = harvest(self.feed_urls, self.video_urls)
        new_strategies = 0
        duplicate_strategies = 0
        failed_extractions = 0
        llm_failures = 0
        last_extractor_failure: OllamaUnavailable | GeminiUnavailable | None = None
        for document in documents:
            service_failure = False
            try:
                extraction = extract_strategy(document)
            except (OllamaUnavailable, GeminiUnavailable) as exc:
                log.warning("%s", exc)
                extraction = None
                llm_failures += 1
                last_extractor_failure = exc
                service_failure = True
                error = f"{type(exc).__name__}; source will be retried."
            except Exception:
                log.exception("Unexpected strategy extraction failure for %s.", document.source_url)
                extraction = None
                service_failure = True
                error = "Unexpected extraction failure; details are in local logs."
            else:
                error = None if extraction else "No valid quantitative strategy was extracted."
            if extraction:
                inserted_id = self.brain.add_strategy(extraction)
                self.brain.record_source(
                    document.source_url, document.title, document.source_type,
                    "extracted" if inserted_id is not None else "duplicate",
                )
                if inserted_id is not None:
                    new_strategies += 1
                else:
                    duplicate_strategies += 1
            else:
                failed_extractions += 1
                if not service_failure:
                    self.brain.reject_strategy(
                        document.source_url,
                        document.title,
                        "universal",
                        "No explicit numerical entry, exit, indicator, or sufficient logical rigor.",
                    )
                self.brain.record_source(
                    document.source_url, document.title, document.source_type,
                    "unavailable" if service_failure else "rejected",
                    error,
                )

        if documents and llm_failures == len(documents):
            if isinstance(last_extractor_failure, GeminiUnavailable):
                raise GeminiUnavailable("Gemini was unavailable for every harvested source.")
            raise OllamaUnavailable("Ollama was unavailable for every harvested source.")
        candidates = self.brain.top_untested(minimum_conviction=0.70, limit=20)
        variant_dir = self.root / "data" / "knowledge_variants"
        variants = [
            str(compile_research_variant(candidate, variant_dir).relative_to(self.root))
            for candidate in candidates
        ]
        summary = self.brain.summary()
        completed = datetime.now(timezone.utc)
        state = {
            "status": "completed",
            "started_at": started.isoformat(),
            "updated_at": completed.isoformat(),
            "sources_harvested": len(documents),
            "new_strategy_extractions": new_strategies,
            "duplicate_strategies": duplicate_strategies,
            "filtered_sources": failed_extractions,
            "knowledge_brain": summary,
            "generated_variants": variants,
            "simulation_status": "untested; no generic strategy simulator is enabled",
            "execution_enabled": False,
            "cloud_deployment": self.cloud_deployment,
            "schedule_minutes": self.interval_seconds // 60,
            "message": (
                f"Processed {len(documents)} sources; extracted {new_strategies} structured "
                "strategies. High-conviction artifacts remain untested and cannot place orders."
            ),
        }
        self._write_state(state)
        log.info(
            "Research factory processed %d documents, extracted %d strategies, brain total=%d.",
            len(documents), new_strategies, summary["strategies"],
        )
        return state

    def run(self, stop_event: threading.Event) -> None:
        delay = self.interval_seconds
        while not stop_event.is_set():
            try:
                self.run_once()
                delay = self.interval_seconds
            except (OllamaUnavailable, GeminiUnavailable, SourcesUnavailable) as exc:
                reason = (
                    "gemini_unavailable" if isinstance(exc, GeminiUnavailable)
                    else "ollama_unavailable" if isinstance(exc, OllamaUnavailable)
                    else "sources_unavailable"
                )
                log.warning("Research ingestion is idle (%s); retrying in background: %s", reason, exc)
                state = {
                    "status": "idle",
                    "health_status": "waiting_for_dependency",
                    "reason": reason,
                    "retrying": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "message": (
                        "Waiting for Gemini; research ingestion will retry automatically."
                        if reason == "gemini_unavailable"
                        else "Waiting for local Ollama; research ingestion will retry automatically."
                        if reason == "ollama_unavailable"
                        else "Research sources are unreachable; ingestion will retry automatically."
                    ),
                    "knowledge_brain": self.brain.get_stats(),
                    "generated_variants": [],
                    "execution_enabled": False,
                    "cloud_deployment": self.cloud_deployment,
                    "schedule_minutes": self.interval_seconds // 60,
                }
                try:
                    self._write_state(state)
                except OSError:
                    log.exception("Unable to persist research dependency status.")
                delay = min(delay * 2, MAX_BACKOFF_SECONDS)
            except Exception as exc:
                log.exception("Research ingestion cycle failed; retrying with backoff.")
                state = {
                    "status": "error",
                    "health_status": "error",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "message": "Research ingestion failed; retrying in the background.",
                    "knowledge_brain": self.brain.get_stats(),
                    "generated_variants": [],
                    "execution_enabled": False,
                    "cloud_deployment": self.cloud_deployment,
                    "schedule_minutes": self.interval_seconds // 60,
                }
                try:
                    self._write_state(state)
                except OSError:
                    log.exception("Unable to persist research factory error state.")
                delay = min(delay * 2, MAX_BACKOFF_SECONDS)
            wait = delay * (1 + random.random() * 0.1)
            if stop_event.wait(wait):
                return
