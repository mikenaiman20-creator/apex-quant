from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from config.settings import load_project_env
from server.ingestion.worker import ResearchIngestionWorker

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("apex.research.cli")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    load_project_env(ROOT)
    if not os.getenv("GEMINI_API_KEY", "").strip():
        log.warning(
            "GEMINI_API_KEY is not configured; extraction will use local Ollama fallback."
        )
    try:
        state = ResearchIngestionWorker(ROOT).run_once()
    except Exception:
        log.exception("Scheduled research ingestion failed.")
        return 1
    log.info(
        "Scheduled research ingestion completed: sources=%d extracted=%d",
        state["sources_harvested"], state["new_strategy_extractions"],
    )
    if not state["sources_harvested"]:
        log.error("No research documents were harvested; reporting the scheduled job as failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
