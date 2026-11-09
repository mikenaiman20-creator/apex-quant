"""Tests for the committed SHA-256 holdout lock and canonical fingerprinting."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import backtest.holdout as holdout_module
from backtest.engine import BTConfig
from backtest.holdout_lock import (
    DEFAULT_LOCK_PATH,
    build_manifest,
    fingerprint_directory,
    main,
    pending_manifest,
    read_lock,
    verify,
    write_lock,
)


def test_committed_lock_manifest_is_valid():
    lock = read_lock(DEFAULT_LOCK_PATH)
    assert lock["algorithm"].startswith("sha256(")
    if lock["status"] == "locked":
        assert lock["files"]
        assert lock["dataset_id"]
    else:
        assert lock["status"] == "pending"
        assert lock["files"] == {}


def _write_holdout(directory: Path, name: str = "SPY", body: str = "immutable fixture") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_pending_lock_passes_when_holdout_is_absent(tmp_path: Path):
    result = verify(tmp_path / "final_holdout", tmp_path / "lock.json")
    assert result == {
        "ok": False,
        "status": "error",
        "reason": f"Holdout lock manifest is missing: {tmp_path / 'lock.json'}",
    }

    write_lock(pending_manifest(), tmp_path / "lock.json")
    result = verify(tmp_path / "final_holdout", tmp_path / "lock.json")
    assert result["ok"] is True
    assert result["status"] == "absent"

    missing = verify(
        tmp_path / "final_holdout", tmp_path / "lock.json", require_locked=True
    )
    assert missing["ok"] is False
    assert missing["status"] == "pending"


def test_pending_lock_fails_closed_when_unlocked_holdout_appears(tmp_path: Path):
    write_lock(pending_manifest(), tmp_path / "lock.json")
    _write_holdout(tmp_path / "final_holdout")

    result = verify(tmp_path / "final_holdout", tmp_path / "lock.json")
    assert result["ok"] is False
    assert result["status"] == "unlocked"
    assert "no locked fingerprint" in result["reason"]


def test_record_then_verify_roundtrip(tmp_path: Path):
    directory = tmp_path / "final_holdout"
    _write_holdout(directory, "SPY", "spy fixture")
    _write_holdout(directory, "QQQ", "qqq fixture")
    lock_path = tmp_path / "lock.json"

    manifest = build_manifest(directory, ["QQQ", "SPY"])
    write_lock(manifest, lock_path)

    assert manifest["status"] == "locked"
    assert manifest["symbols"] == ["QQQ", "SPY"]
    assert set(manifest["files"]) == {"QQQ.csv", "SPY.csv"}
    assert manifest["locked_at"]

    result = verify(directory, lock_path, require_locked=True)
    assert result["ok"] is True
    assert result["status"] == "verified"
    assert result["dataset_id"] == manifest["dataset_id"]


def test_tampered_or_extra_holdout_files_fail(tmp_path: Path):
    directory = tmp_path / "final_holdout"
    path = _write_holdout(directory)
    lock_path = tmp_path / "lock.json"
    write_lock(build_manifest(directory, ["SPY"]), lock_path)

    path.write_text("revised after seeing the score", encoding="utf-8")
    tampered = verify(directory, lock_path)
    assert tampered["ok"] is False
    assert tampered["status"] == "modified"
    assert "SPY.csv" in tampered["reason"]

    path.write_text("immutable fixture", encoding="utf-8")
    _write_holdout(directory, "QQQ", "sneaked in later")
    extra = verify(directory, lock_path)
    assert extra["ok"] is False
    assert "QQQ.csv" in extra["reason"]


def test_dataset_id_is_independent_of_symbol_order(tmp_path: Path):
    directory = tmp_path / "final_holdout"
    _write_holdout(directory, "SPY", "spy")
    _write_holdout(directory, "QQQ", "qqq")
    _write_holdout(directory, "NVDA", "nvda")

    forward = fingerprint_directory(directory, ["SPY", "QQQ", "NVDA"])
    reverse = fingerprint_directory(directory, ["NVDA", "QQQ", "SPY"])
    assert forward["dataset_id"] == reverse["dataset_id"]
    assert forward["files"] == reverse["files"]


def _stub_backtest(monkeypatch) -> None:
    monkeypatch.setattr(
        holdout_module, "load_dir", lambda *_args: {"SPY": [object()], "QQQ": [object()]}
    )
    monkeypatch.setattr(
        holdout_module,
        "run_backtest",
        lambda *_args: type("Result", (), {"trades": []})(),
    )
    monkeypatch.setattr(
        holdout_module,
        "trade_equity_curve",
        lambda *_args: [(datetime.now(timezone.utc).date(), 5000)],
    )
    monkeypatch.setattr(
        holdout_module,
        "summarize",
        lambda *_args: {
            "n": 30, "sharpe_daily_ann": 2.0, "profit_factor": 1.5,
            "max_dd_pct": 4.0, "mean_r": 0.1,
        },
    )


def test_holdout_consumption_is_order_independent(tmp_path: Path, monkeypatch):
    directory = tmp_path / "final_holdout"
    _write_holdout(directory, "QQQ", "qqq")
    _write_holdout(directory, "SPY", "spy")
    _stub_backtest(monkeypatch)
    ledger = tmp_path / "consumption.json"

    first = holdout_module.validate_final_holdout(
        ["SPY", "QQQ"], BTConfig(), {}, "candidate-a", directory, ledger,
    )
    reordered = holdout_module.validate_final_holdout(
        ["QQQ", "SPY"], BTConfig(), {}, "candidate-b", directory, ledger,
    )

    assert first["passed"] is True
    assert reordered["passed"] is False
    assert "already consumed" in reordered["reason"]


def test_holdout_runtime_rejects_unlocked_or_tampered_data(tmp_path: Path, monkeypatch):
    directory = tmp_path / "final_holdout"
    path = _write_holdout(directory)
    lock_path = tmp_path / "lock.json"
    write_lock(pending_manifest(), lock_path)
    _stub_backtest(monkeypatch)

    def run(candidate: str, ledger: Path):
        return holdout_module.validate_final_holdout(
            ["SPY"], BTConfig(), {}, candidate, directory, ledger,
            lock_path=lock_path, require_lock=True,
        )

    pending = run("candidate-a", tmp_path / "a.json")
    assert pending["available"] is False
    assert "lock rejected" in pending["reason"]
    assert "no locked fingerprint" in pending["reason"]

    write_lock(build_manifest(directory, ["SPY"]), lock_path)
    passed = run("candidate-a", tmp_path / "b.json")
    assert passed["passed"] is True

    path.write_text("tampered", encoding="utf-8")
    tampered = run("candidate-b", tmp_path / "c.json")
    assert tampered["available"] is False
    assert "lock rejected" in tampered["reason"]
    assert "SPY.csv" in tampered["reason"]


def test_cli_record_and_verify_exit_codes(tmp_path: Path, capsys):
    directory = tmp_path / "final_holdout"
    _write_holdout(directory)
    lock_path = tmp_path / "lock.json"

    assert main(["record", "--dir", str(directory), "--lock", str(lock_path)]) == 0
    assert main(["verify", "--dir", str(directory), "--lock", str(lock_path)]) == 0
    _write_holdout(directory, body="changed")
    assert main(["verify", "--dir", str(directory), "--lock", str(lock_path)]) == 1
    assert "FAIL" in capsys.readouterr().err
