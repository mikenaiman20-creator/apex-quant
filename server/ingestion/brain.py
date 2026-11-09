from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Any

from server.io_utils import atomic_write_text, ensure_directory, secure_file

log = logging.getLogger("apex.research.brain")
VALID_STATUSES = {"untested", "simulating", "promoted", "rejected"}
MIN_SIMULATION_TRADES = 20
MIN_SIMULATION_WIN_RATE = 0.55
MIN_SIMULATION_PROFIT_FACTOR = 1.50
MIN_LOGICAL_RIGOR = 0.80
MIN_COMBINED_CONVICTION = 0.80
_BRAIN_LOCK = threading.RLock()
DEFAULT_BRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_brain.sqlite"


class KnowledgeBrain:
    def __init__(self, path: Path = DEFAULT_BRAIN_PATH):
        self.path = Path(path)
        ensure_directory(self.path.parent)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        ensure_directory(self.path.parent)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            log.warning("SQLite WAL is unavailable for %s; falling back to rollback journal.",
                        self.path)
            try:
                connection.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.Error:
                pass
        try:
            connection.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        secure_file(self.path)
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with _BRAIN_LOCK, self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY,
                    source_url TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL CHECK(category IN ('stock', 'forex', 'crypto')),
                    strategy_json TEXT NOT NULL,
                    conviction_score REAL NOT NULL CHECK(conviction_score >= 0 AND conviction_score <= 1),
                    status TEXT NOT NULL DEFAULT 'untested'
                        CHECK(status IN ('untested', 'simulating', 'promoted', 'rejected')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_sources (
                    source_url TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    extraction_status TEXT NOT NULL,
                    error TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS strategy_simulations (
                    strategy_id INTEGER PRIMARY KEY REFERENCES strategies(id),
                    completed_trades INTEGER NOT NULL CHECK(completed_trades >= 0),
                    winning_trades INTEGER NOT NULL CHECK(
                        winning_trades >= 0 AND winning_trades <= completed_trades
                    ),
                    gross_profit REAL NOT NULL CHECK(gross_profit >= 0),
                    gross_loss REAL NOT NULL CHECK(gross_loss >= 0),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS strategies_status_score "
                "ON strategies(status, conviction_score DESC)"
            )
        for suffix in ("", "-wal", "-shm"):
            secure_file(self.path.with_name(self.path.name + suffix))

    def record_source(
        self, source_url: str, title: str, source_type: str,
        extraction_status: str, error: str | None = None,
    ) -> None:
        with _BRAIN_LOCK, self._connection() as connection:
            connection.execute(
                """INSERT INTO ingestion_sources
                   (source_url, title, source_type, fetched_at, extraction_status, error)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                   ON CONFLICT(source_url) DO UPDATE SET
                     title=excluded.title, source_type=excluded.source_type,
                     fetched_at=excluded.fetched_at,
                     extraction_status=excluded.extraction_status,
                     error=excluded.error""",
                (source_url, title[:300], source_type, extraction_status, error),
            )

    def add_strategy(self, extraction: dict[str, Any]) -> int | None:
        category = extraction.get("storage_category")
        if category not in {"stock", "forex", "crypto"}:
            raise ValueError("Strategy extraction has no supported storage category.")
        conviction = float(extraction["conviction_score"])
        if not math.isfinite(conviction) or not MIN_LOGICAL_RIGOR <= conviction <= 1:
            raise ValueError(
                f"Logical rigor score must be between {MIN_LOGICAL_RIGOR:.2f} and 1."
            )
        strategy_json = json.dumps({
            key: value for key, value in extraction.items() if key != "storage_category"
        }, ensure_ascii=True, allow_nan=False)
        with _BRAIN_LOCK, self._connection() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO strategies
                   (source_url, category, strategy_json, conviction_score, status)
                   VALUES (?, ?, ?, ?, 'untested')""",
                (extraction["source_url"], category, strategy_json, conviction),
            )
            if cursor.rowcount:
                return int(cursor.lastrowid)
            return None

    def reject_strategy(
        self, source_url: str, title: str, category: str, reason: str,
    ) -> int:
        normalized_category = (
            "crypto" if category == "crypto" else "forex" if category == "forex" else "stock"
        )
        strategy_json = json.dumps({
            "title": title[:200],
            "category": category,
            "gate": "mathematical_extraction",
            "rejection_reason": reason[:600],
            "execution_enabled": False,
        }, ensure_ascii=True)
        with _BRAIN_LOCK, self._connection() as connection:
            connection.execute(
                """INSERT INTO strategies
                   (source_url, category, strategy_json, conviction_score, status)
                   VALUES (?, ?, ?, 0, 'rejected')
                   ON CONFLICT(source_url) DO NOTHING""",
                (source_url, normalized_category, strategy_json),
            )
            row = connection.execute(
                "SELECT id FROM strategies WHERE source_url = ?", (source_url,)
            ).fetchone()
            return int(row["id"])

    def record_simulation(
        self,
        strategy_id: int,
        completed_trades: int,
        winning_trades: int,
        gross_profit: float,
        gross_loss: float,
    ) -> None:
        if (
            isinstance(completed_trades, bool)
            or isinstance(winning_trades, bool)
            or not isinstance(completed_trades, int)
            or not isinstance(winning_trades, int)
            or completed_trades < 0
            or winning_trades < 0
            or winning_trades > completed_trades
        ):
            raise ValueError("Simulation trade counts are invalid.")
        try:
            gross_profit = float(gross_profit)
            gross_loss = float(gross_loss)
        except (TypeError, ValueError) as exc:
            raise ValueError("Simulation gross P&L values must be numeric.") from exc
        if (
            not math.isfinite(gross_profit)
            or not math.isfinite(gross_loss)
            or gross_profit < 0
            or gross_loss < 0
        ):
            raise ValueError("Simulation gross P&L values must be finite and non-negative.")
        with _BRAIN_LOCK, self._connection() as connection:
            strategy = connection.execute(
                "SELECT status FROM strategies WHERE id = ?", (strategy_id,)
            ).fetchone()
            if not strategy:
                raise KeyError(f"Unknown strategy id: {strategy_id}")
            if strategy["status"] not in {"untested", "simulating"}:
                raise ValueError("Simulation metrics can only update untested/simulating strategies.")
            connection.execute(
                """INSERT INTO strategy_simulations
                   (strategy_id, completed_trades, winning_trades, gross_profit, gross_loss, updated_at)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(strategy_id) DO UPDATE SET
                     completed_trades=excluded.completed_trades,
                     winning_trades=excluded.winning_trades,
                     gross_profit=excluded.gross_profit,
                     gross_loss=excluded.gross_loss,
                     updated_at=CURRENT_TIMESTAMP""",
                (strategy_id, completed_trades, winning_trades, gross_profit, gross_loss),
            )
            connection.execute(
                "UPDATE strategies SET status='simulating' WHERE id = ?", (strategy_id,)
            )

    def get_asset_performance_stats(self, asset_class: str) -> dict[str, float | int]:
        if asset_class not in {"stock", "forex", "crypto"}:
            raise ValueError(f"Unsupported asset class: {asset_class}")
        with _BRAIN_LOCK, self._connection() as connection:
            rows = connection.execute(
                """SELECT s.strategy_json, p.completed_trades, p.winning_trades,
                          p.gross_profit, p.gross_loss
                   FROM strategies s
                   JOIN strategy_simulations p ON p.strategy_id = s.id
                   WHERE s.category = ? AND s.status = 'simulating'""",
                (asset_class,),
            ).fetchall()
        trades = sum(int(row["completed_trades"]) for row in rows)
        wins = sum(int(row["winning_trades"]) for row in rows)
        gross_profit = sum(float(row["gross_profit"]) for row in rows)
        gross_loss = sum(float(row["gross_loss"]) for row in rows)
        rigor_weighted = 0.0
        for row in rows:
            strategy = json.loads(row["strategy_json"])
            rigor_weighted += float(strategy.get("conviction_score", 0)) * int(
                row["completed_trades"]
            )
        return {
            "total_trades": trades,
            "winning_trades": wins,
            "win_rate": wins / trades if trades else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0
                else math.inf if gross_profit > 0 else 0.0
            ),
            "logical_rigor": rigor_weighted / trades if trades else 0.0,
            "strategy_count": len(rows),
        }

    def top_untested(self, minimum_conviction: float = 0.70, limit: int = 20) -> list[dict[str, Any]]:
        with _BRAIN_LOCK, self._connection() as connection:
            rows = connection.execute(
                """SELECT id, source_url, category, strategy_json, conviction_score,
                          status, created_at
                   FROM strategies
                   WHERE status = 'untested' AND conviction_score >= ?
                   ORDER BY conviction_score DESC, created_at ASC
                   LIMIT ?""",
                (minimum_conviction, max(0, min(limit, 100))),
            ).fetchall()
        return [
            {**dict(row), "strategy": json.loads(row["strategy_json"])}
            for row in rows
        ]

    def set_status(self, strategy_id: int, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid strategy status: {status}")
        if status == "promoted":
            self.promote_strategy(strategy_id)
            return
        with _BRAIN_LOCK, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE strategies SET status = ? WHERE id = ?",
                (status, strategy_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"Unknown strategy id: {strategy_id}")

    def promote_strategy(self, strategy_id: int) -> None:
        with _BRAIN_LOCK, self._connection() as connection:
            row = connection.execute(
                """SELECT s.status, s.conviction_score, p.completed_trades,
                          p.winning_trades, p.gross_profit, p.gross_loss
                   FROM strategies s
                   JOIN strategy_simulations p ON p.strategy_id = s.id
                   WHERE s.id = ?""",
                (strategy_id,),
            ).fetchone()
            if not row:
                exists = connection.execute(
                    "SELECT 1 FROM strategies WHERE id = ?", (strategy_id,)
                ).fetchone()
                if not exists:
                    raise KeyError(f"Unknown strategy id: {strategy_id}")
                raise ValueError("A strategy requires recorded simulation metrics before promotion.")
            if row["status"] != "simulating":
                raise ValueError("Only a simulating strategy can be promoted.")
            trades = int(row["completed_trades"])
            win_rate = int(row["winning_trades"]) / trades if trades else 0.0
            profit_factor = (
                float(row["gross_profit"]) / float(row["gross_loss"])
                if float(row["gross_loss"]) > 0
                else math.inf if float(row["gross_profit"]) > 0 else 0.0
            )
            performance_score = min(
                1.0,
                0.5 * (win_rate / MIN_SIMULATION_WIN_RATE)
                + 0.5 * (profit_factor / MIN_SIMULATION_PROFIT_FACTOR),
            )
            combined_conviction = 0.5 * float(row["conviction_score"]) + 0.5 * performance_score
            if (
                trades < MIN_SIMULATION_TRADES
                or win_rate < MIN_SIMULATION_WIN_RATE
                or profit_factor < MIN_SIMULATION_PROFIT_FACTOR
                or float(row["conviction_score"]) < MIN_LOGICAL_RIGOR
                or combined_conviction < MIN_COMBINED_CONVICTION
            ):
                raise ValueError("Strategy does not meet the Knowledge Brain promotion gates.")
            connection.execute(
                "UPDATE strategies SET status = 'promoted' WHERE id = ?", (strategy_id,)
            )

    def has_promoted_strategy(self, asset_class: str) -> bool:
        if asset_class not in {"stock", "forex", "crypto"}:
            raise ValueError(f"Unsupported asset class: {asset_class}")
        with _BRAIN_LOCK, self._connection() as connection:
            row = connection.execute(
                """SELECT 1 FROM strategies s
                   JOIN strategy_simulations p ON p.strategy_id = s.id
                   WHERE s.category = ? AND s.status = 'promoted'
                     AND p.completed_trades >= ?
                     AND 1.0 * p.winning_trades / p.completed_trades >= ?
                     AND (p.gross_profit / NULLIF(p.gross_loss, 0) >= ?
                          OR (p.gross_loss = 0 AND p.gross_profit > 0))
                     AND s.conviction_score >= ?
                   LIMIT 1""",
                (
                    asset_class, MIN_SIMULATION_TRADES, MIN_SIMULATION_WIN_RATE,
                    MIN_SIMULATION_PROFIT_FACTOR, MIN_LOGICAL_RIGOR,
                ),
            ).fetchone()
        return row is not None

    def summary(self) -> dict[str, int]:
        with _BRAIN_LOCK, self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM strategies GROUP BY status"
            ).fetchall()
            sources = connection.execute("SELECT COUNT(*) AS count FROM ingestion_sources").fetchone()
        result = {status: 0 for status in VALID_STATUSES}
        result.update({str(row["status"]): int(row["count"]) for row in rows})
        result["sources"] = int(sources["count"])
        result["strategies"] = sum(result[status] for status in VALID_STATUSES)
        return result

    def get_stats(self) -> dict[str, int]:
        return self.summary()


def compile_research_variant(candidate: dict[str, Any], directory: Path) -> Path:
    """Emit a literal-only Python research artifact; it is never imported or executed."""
    strategy = candidate.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("Candidate has no parsed strategy.")
    variant_id = int(candidate["id"])
    source = (
        "# Generated declarative research artifact; not executable trading logic.\n"
        "EXECUTION_ENABLED = False\n"
        "SIMULATION_STATUS = 'untested'\n"
        f"STRATEGY_ID = {variant_id}\n"
        f"STRATEGY_SPEC = {pformat(strategy, sort_dicts=True, width=88)}\n"
    )
    compile(source, f"variant_{variant_id}.py", "exec")
    ensure_directory(directory)
    target = directory / f"variant_{variant_id}.py"
    if target.is_file() and target.read_text(encoding="utf-8") == source:
        return target
    atomic_write_text(target, source)
    return target


def evaluate_order_routing_eligibility(
    knowledge_brain: KnowledgeBrain, asset_class: str,
) -> bool:
    """Return whether research evidence clears the metrics gate; never enables routing."""
    stats = knowledge_brain.get_asset_performance_stats(asset_class)
    if stats["total_trades"] < MIN_SIMULATION_TRADES:
        return False
    if stats["win_rate"] < MIN_SIMULATION_WIN_RATE:
        return False
    if stats["profit_factor"] < MIN_SIMULATION_PROFIT_FACTOR:
        return False
    if stats["logical_rigor"] < MIN_LOGICAL_RIGOR:
        return False
    performance_score = min(
        1.0,
        0.5 * (stats["win_rate"] / MIN_SIMULATION_WIN_RATE)
        + 0.5 * (stats["profit_factor"] / MIN_SIMULATION_PROFIT_FACTOR),
    )
    combined_conviction = 0.5 * stats["logical_rigor"] + 0.5 * performance_score
    return combined_conviction >= MIN_COMBINED_CONVICTION
