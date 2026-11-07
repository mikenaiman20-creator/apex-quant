from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(os.environ.get("APEX_DATA_ROOT", "/app"))
LOGS = ROOT / "logs"
DATA = ROOT / "data"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError:
        return None, f"{path.name} has not been published yet."
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object."
    return value, None


def _read_events(path: Path, limit: int = 100) -> list[dict[str, str]]:
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            end = file.tell()
            file.seek(max(0, end - 256_000))
            lines = file.read().decode("utf-8", "replace").splitlines()[-limit:]
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [{"timestamp": "", "level": "ERROR", "message": f"Could not read {path.name}: {exc}"}]

    events = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            events.append({"timestamp": "", "level": "INFO", "message": line})
            continue
        if isinstance(item, dict):
            events.append({
                "timestamp": str(item.get("timestamp", "")),
                "level": str(item.get("level", "INFO")),
                "message": str(item.get("message", line)),
            })
    return list(reversed(events))


@st.cache_data(ttl=30, show_spinner=False)
def _read_knowledge_database(path_text: str, modified_ns: int) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_file():
        return {"error": "Knowledge Brain SQLite database is not present in the production data mount."}
    uri = f"{path.as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            counts = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM strategies GROUP BY status"
                )
            }
            source_count = connection.execute(
                "SELECT COUNT(*) FROM ingestion_sources"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT id, category, conviction_score, status, created_at, strategy_json
                FROM strategies
                ORDER BY created_at DESC, id DESC
                LIMIT 100
                """
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return {"error": f"Knowledge Brain SQLite read failed: {exc}"}

    strategies = []
    for row in rows:
        try:
            details = json.loads(row["strategy_json"])
        except (TypeError, json.JSONDecodeError):
            details = {}
        strategies.append({
            "id": row["id"],
            "title": details.get("title", f"Strategy {row['id']}"),
            "category": row["category"],
            "status": row["status"],
            "conviction": row["conviction_score"],
            "created_at": row["created_at"],
            "rejection_reason": details.get("rejection_reason", ""),
        })
    return {
        "integrity": integrity,
        "status_counts": counts,
        "source_count": source_count,
        "strategies": strategies,
    }


def _heartbeat_age(timestamp: Any) -> float | None:
    try:
        if isinstance(timestamp, (int, float)):
            seconds = float(timestamp)
            if seconds > 100_000_000_000:
                seconds /= 1000
        elif isinstance(timestamp, str):
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = parsed.timestamp()
        else:
            return None
        return time.time() - seconds
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any, suffix: str = "") -> str:
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _table(items: Any, columns: tuple[str, ...] | None = None) -> None:
    if not isinstance(items, list) or not items:
        st.info("No records are available in the latest production snapshot.")
        return
    rows = [item for item in items if isinstance(item, dict)]
    if not rows:
        st.info("No records are available in the latest production snapshot.")
        return
    if columns:
        rows = [{key: item.get(key) for key in columns} for item in rows]
    st.dataframe(rows, width="stretch", hide_index=True)


def _json_panel(title: str, value: Any) -> None:
    with st.expander(title):
        if value is None:
            st.info("No data has been published yet.")
        else:
            st.json(value)


def _render_dashboard() -> None:
    heartbeat, heartbeat_error = _read_json(LOGS / "runner_heartbeat.json")
    profile, profile_error = _read_json(LOGS / "daily_profile.json")
    research, research_error = _read_json(LOGS / "research_state.json")
    researcher, researcher_error = _read_json(LOGS / "research_agent_state.json")
    brain, brain_error = _read_json(LOGS / "research_brain_state.json")
    audit, audit_error = _read_json(LOGS / "strategy_audit_report.json")
    trials, trials_error = _read_json(LOGS / "strategy_trial_ledger.json")
    paper, paper_error = _read_json(LOGS / "paper_dashboard_snapshot.json")
    arsenal, arsenal_error = _read_json(DATA / "strategy_arsenal.json")
    if arsenal is None:
        arsenal, fallback_arsenal_error = _read_json(LOGS / "strategy_arsenal.json")
        if arsenal_error and fallback_arsenal_error:
            arsenal_error = fallback_arsenal_error
    macro, macro_error = _read_json(DATA / "macro_events.json")
    events = _read_events(LOGS / "autonomous.log")

    telemetry_missing = all(
        value is None for value in (heartbeat, profile, research, researcher, brain, audit, paper)
    )
    st.title("Apex Quant")
    st.caption("Production monitor · read-only · PAPER execution only")
    if telemetry_missing:
        st.error(
            "Production telemetry is not mounted or has not been generated. "
            "Mount the VPS data/ and logs/ directories read-only at /app/data and /app/logs."
        )
        for error in (
            heartbeat_error, profile_error, research_error, researcher_error,
            brain_error, audit_error, paper_error,
        ):
            if error:
                st.caption(error)

    runner_status = str((heartbeat or {}).get("status", "disconnected")).upper()
    age = _heartbeat_age((heartbeat or {}).get("timestamp"))
    if age is None or age > 120 or age < -30:
        runner_status = "DISCONNECTED"
    status_text = (
        f"{runner_status} · heartbeat {max(0, int(age))}s ago"
        if age is not None else f"{runner_status} · heartbeat unavailable"
    )
    st.caption(f"Runner: {status_text} · refreshed {datetime.now(timezone.utc):%H:%M:%S UTC}")

    account = paper.get("account", {}) if isinstance(paper, dict) else {}
    positions = paper.get("positions", []) if isinstance(paper, dict) else []
    orders = paper.get("orders", []) if isinstance(paper, dict) else []
    telemetry = (heartbeat or {}).get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    knowledge = (brain or {}).get("knowledge_brain", {})
    knowledge = knowledge if isinstance(knowledge, dict) else {}

    top = st.columns(6)
    top[0].metric("Runner", runner_status)
    top[1].metric(
        "Paper API",
        str((paper or {}).get("paper_api_status", "unavailable")).upper(),
    )
    top[2].metric("Paper equity", _money(account.get("equity")))
    top[3].metric("Cash", _money(account.get("cash")))
    top[4].metric("Open positions", str(len(positions)) if isinstance(positions, list) else "—")
    top[5].metric("Recent orders", str(len(orders)) if isinstance(orders, list) else "—")
    if paper_error:
        st.warning(f"Paper account snapshot unavailable: {paper_error}")
    elif isinstance(paper, dict) and paper:
        snapshot_age = _heartbeat_age(paper.get("timestamp"))
        if snapshot_age is None or snapshot_age > 600 or snapshot_age < -30:
            st.warning("Paper account snapshot is stale; values below are last known, not live.")

    if runner_status == "DISCONNECTED":
        st.error((heartbeat or {}).get("message") or "Runner heartbeat is missing or stale.")
    elif heartbeat:
        st.success((heartbeat or {}).get("message") or "Runner heartbeat is current.")

    tabs = st.tabs(["Overview", "Trading", "Research", "Strategy validation", "Activity"])

    with tabs[0]:
        st.subheader("Latest signal scan")
        if telemetry:
            cols = st.columns(5)
            cols[0].metric("Symbol", str(telemetry.get("symbol", "—")))
            cols[1].metric("Decision", "ACCEPTED" if telemetry.get("accepted") else "REJECTED")
            cols[2].metric("Score", _number(telemetry.get("score")))
            cols[3].metric("RVOL", _number(telemetry.get("rvol"), "x"))
            cols[4].metric("ADX", _number(telemetry.get("adx")))
            st.write("Reason:", telemetry.get("reason") or "No reason recorded.")
            st.write("Phase:", telemetry.get("phase", "—"), "· ATR:", _number(telemetry.get("atr")))
        else:
            st.info("No per-symbol scan telemetry has been reported yet.")

        st.subheader("Active universe")
        symbols = (profile or {}).get("universe") or (heartbeat or {}).get("symbols") or []
        if symbols:
            st.write(" · ".join(str(symbol) for symbol in symbols))
        else:
            st.info("The runner has not published a symbol universe.")
        st.subheader("Strategy profile")
        if profile:
            params = profile.get("params")
            if isinstance(params, dict) and params:
                st.json(params)
            else:
                st.info("Profile exists but contains no selected parameters.")
            st.caption(
                f"Generated: {profile.get('timestamp') or profile.get('created_at') or 'not recorded'}"
            )
        else:
            st.info(profile_error or "No daily strategy profile is available.")

        st.subheader("Read-only safety")
        st.info(
            "The dashboard receives no broker credentials and cannot submit or modify orders. "
            "Displayed account state is published by the paper runner."
        )

    with tabs[1]:
        st.subheader("Paper account")
        if account:
            account_cols = st.columns(4)
            account_cols[0].metric("Equity", _money(account.get("equity")))
            account_cols[1].metric("Cash", _money(account.get("cash")))
            account_cols[2].metric("Buying power", _money(account.get("buying_power")))
            account_cols[3].metric("Portfolio value", _money(account.get("portfolio_value")))
            st.caption(
                f"Broker status: {account.get('status', 'unknown')} · "
                f"Trading blocked: {account.get('trading_blocked', 'unknown')}"
            )
        else:
            st.info("No paper-account snapshot is available. Start the runner to publish it.")
        st.subheader("Open positions")
        _table(positions, (
            "symbol", "qty", "side", "market_value", "avg_entry_price",
            "current_price", "unrealized_pl", "unrealized_plpc",
        ))
        st.subheader("Recent orders")
        _table(orders, (
            "submitted_at", "symbol", "side", "qty", "filled_qty", "type",
            "status", "filled_at",
        ))
        snapshot_time = (paper or {}).get("timestamp") if isinstance(paper, dict) else None
        st.caption(f"Snapshot time: {snapshot_time or 'not recorded'}")

    with tabs[2]:
        st.subheader("Research worker")
        if researcher:
            progress = st.columns(4)
            progress[0].metric("Status", str(researcher.get("status", "unknown")))
            progress[1].metric("Variants tested", str(researcher.get("variants_tested", 0)))
            progress[2].metric("Planned variants", str(researcher.get("planned_variants", "—")))
            progress[3].metric("Cumulative trials", str((trials or {}).get("total_trials", 0)))
            st.write("Current task:", researcher.get("current_task", "—"))
            st.write("Symbols:", ", ".join(map(str, researcher.get("symbols", []))))
        else:
            st.info(researcher_error or "No researcher state has been published.")

        st.subheader("Market regimes")
        regimes = (research or {}).get("regimes", {}) if isinstance(research, dict) else {}
        if isinstance(regimes, dict) and regimes:
            _table([
                {"symbol": symbol, **(value if isinstance(value, dict) else {"regime": value})}
                for symbol, value in regimes.items()
            ])
        else:
            st.info(research_error or "No market-regime report is available.")

        st.subheader("Knowledge research")
        if brain:
            brain_cols = st.columns(4)
            brain_cols[0].metric("Ingestion", str(brain.get("status", "unknown")))
            brain_cols[1].metric("Strategies", str(knowledge.get("strategies", 0)))
            brain_cols[2].metric("Untested", str(knowledge.get("untested", 0)))
            brain_cols[3].metric("Sources", str(knowledge.get("sources", 0)))
            st.caption(
                f"{brain.get('message') or brain.get('last_error') or 'Research artifacts remain advisory.'}"
            )
        else:
            st.info(brain_error or "No knowledge-ingestion state is available.")
        st.subheader("Knowledge Brain database")
        database_path = DATA / "knowledge_brain.sqlite"
        try:
            database_mtime = database_path.stat().st_mtime_ns
        except OSError:
            database_mtime = 0
        database = _read_knowledge_database(str(database_path), database_mtime)
        if "error" in database:
            st.warning(database["error"])
        else:
            db_metrics = st.columns(3)
            db_metrics[0].metric("SQLite integrity", str(database["integrity"]).upper())
            db_metrics[1].metric("Research sources", str(database["source_count"]))
            db_metrics[2].metric(
                "Stored strategies",
                str(sum(database["status_counts"].values())),
            )
            st.json(database["status_counts"])
            _table(database["strategies"], (
                "id", "title", "category", "status", "conviction", "created_at",
                "rejection_reason",
            ))
        _json_panel("Full research state", research)
        _json_panel("Full ingestion state", brain)

    with tabs[3]:
        st.subheader("Out-of-sample and audit status")
        if audit:
            holdout = audit.get("holdout") if isinstance(audit.get("holdout"), dict) else {}
            metrics = st.columns(5)
            metrics[0].metric("Audit status", str(audit.get("status", "unknown")))
            metrics[1].metric("Trials", str(audit.get("n_trials", (trials or {}).get("total_trials", 0))))
            metrics[2].metric("Raw Sharpe", _number(audit.get("raw_sharpe")))
            metrics[3].metric("DSR probability", _number(audit.get("deflated_sharpe_probability")))
            metrics[4].metric("Holdout", "PASS" if holdout.get("passed") else "NOT PASSED")
            st.write("Candidate:", audit.get("candidate_id", "—"))
            st.write(
                "Required Sharpe:", _number(audit.get("required_sharpe")),
                "· simulated costs:", _money(audit.get("simulated_transaction_costs")),
                "· slippage drag:", _money(audit.get("simulated_slippage_drag")),
            )
            st.caption(f"Holdout note: {holdout.get('reason') or 'not reported'}")
        else:
            st.info(audit_error or "No strategy audit report is available.")
        st.subheader("Strategy arsenal")
        strategies = (arsenal or {}).get("strategies", []) if isinstance(arsenal, dict) else []
        _table(strategies)
        if arsenal_error and arsenal is None:
            st.caption(arsenal_error)
        st.subheader("Candidates awaiting human review")
        candidate_dir = DATA / "staging"
        candidates = []
        if candidate_dir.is_dir():
            for path in sorted(candidate_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                candidate, _ = _read_json(path)
                if (
                    candidate
                    and candidate.get("status") == "staged_candidate"
                    and candidate.get("human_approved") is not True
                ):
                    candidates.append(candidate)
        _table(candidates, ("candidate_id", "status", "paper_trades", "profit_factor", "sharpe"))
        _json_panel("Full audit report", audit)
        _json_panel("Trial ledger", trials)

    with tabs[4]:
        st.subheader("Execution and system activity")
        if events:
            st.dataframe(events, width="stretch", hide_index=True)
        else:
            st.info("No autonomous log entries are available.")
        st.subheader("Macro calendar")
        macro_events = (macro or {}).get("events", []) if isinstance(macro, dict) else []
        if macro_events:
            _table(macro_events)
        else:
            st.info(macro_error or "No local macro events have been configured.")

    with st.expander("Raw telemetry files"):
        st.caption("These are the read-only files mounted from the production data/logs directories.")
        for title, value in (
            ("Runner heartbeat", heartbeat),
            ("Paper account snapshot", paper),
            ("Daily profile", profile),
            ("Researcher", researcher),
            ("Audit", audit),
        ):
            _json_panel(title, value)


st.set_page_config(page_title="Apex Quant", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #11151b; }
    [data-testid="stMetric"] {
      background: #1b222c; border: 1px solid #293342; padding: 14px; border-radius: 10px;
    }
    [data-testid="stAlert"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.fragment(run_every="15s")
def dashboard() -> None:
    _render_dashboard()


dashboard()
