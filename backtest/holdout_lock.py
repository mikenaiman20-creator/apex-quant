"""SHA-256 lock manifest for the isolated final holdout dataset.

The holdout CSVs are deliberately *not* stored in the repository.  This module
records a canonical fingerprint of the exact files that were frozen for the
blind test, so CI and the production host can prove that the dataset being
evaluated is the untouched snapshot rather than a revision chosen after seeing
results.

Usage:
    python -m backtest.holdout_lock record --dir data/final_holdout
    python -m backtest.holdout_lock verify
    python -m backtest.holdout_lock verify --require-locked   # production host
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from server.io_utils import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "final_holdout"
DEFAULT_LOCK_PATH = ROOT / "config" / "holdout.lock.json"
CHUNK_BYTES = 1024 * 1024
LOCK_VERSION = 1
ALGORITHM = "sha256(canonical-json-v1: sorted filename + per-file sha256 + bytes)"


def canonical_symbols(symbols: Iterable[str]) -> list[str]:
    """Return symbols sorted by filename so the dataset id never depends on order."""
    cleaned = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        raise ValueError("At least one holdout symbol is required.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Holdout symbols must be unique.")
    return sorted(cleaned, key=str.upper)


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def dataset_id_from_files(files: dict[str, dict[str, Any]]) -> str:
    """Stable id over sorted filenames and their content hashes (order independent)."""
    payload = json.dumps(
        {"version": LOCK_VERSION, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_directory(
    directory: Path, symbols: Sequence[str] | None = None
) -> dict[str, Any]:
    """Fingerprint every ``*.csv`` in ``directory`` (or only the given symbols)."""
    directory = Path(directory)
    if symbols:
        paths = [directory / f"{symbol}.csv" for symbol in canonical_symbols(symbols)]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing holdout files: " + ", ".join(sorted(missing))
            )
    else:
        paths = sorted(directory.glob("*.csv"), key=lambda path: path.name)
        if not paths:
            raise FileNotFoundError(f"No holdout CSV files found in {directory}")
    files = {path.name: _hash_file(path) for path in paths}
    return {
        "symbols": [Path(name).stem for name in files],
        "files": files,
        "dataset_id": dataset_id_from_files(files),
    }


def read_lock(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Holdout lock manifest is missing: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Holdout lock manifest is unreadable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("version") != LOCK_VERSION:
        raise ValueError(f"Unsupported holdout lock manifest: {path}")
    if lock.get("status") not in {"pending", "locked"}:
        raise ValueError(f"Invalid holdout lock status in {path}")
    return lock


def write_lock(manifest: dict[str, Any], path: Path = DEFAULT_LOCK_PATH) -> None:
    atomic_write_json(Path(path), manifest, indent=2)


def build_manifest(directory: Path, symbols: Sequence[str]) -> dict[str, Any]:
    fingerprint = fingerprint_directory(directory, symbols)
    return {
        "version": LOCK_VERSION,
        "algorithm": ALGORITHM,
        "status": "locked",
        "locked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbols": fingerprint["symbols"],
        "dataset_id": fingerprint["dataset_id"],
        "files": fingerprint["files"],
    }


def pending_manifest() -> dict[str, Any]:
    return {
        "version": LOCK_VERSION,
        "algorithm": ALGORITHM,
        "status": "pending",
        "locked_at": None,
        "symbols": [],
        "dataset_id": None,
        "files": {},
    }


def verify(
    directory: Path = DEFAULT_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    *,
    require_locked: bool = False,
) -> dict[str, Any]:
    """Verify the holdout against the committed lock without consuming it.

    * No files present: pass for CI when the lock is ``pending`` or ``locked``
      (private data is not checked out in a clean CI job).
    * Files present: every file must match the manifest byte-for-byte and the
      manifest must be ``locked``; otherwise fail closed.
    """
    try:
        lock = read_lock(lock_path)
    except ValueError as exc:
        return {"ok": False, "status": "error", "reason": str(exc)}

    directory = Path(directory)
    present = sorted(
        (path for path in directory.glob("*.csv") if path.is_file()),
        key=lambda path: path.name,
    ) if directory.is_dir() else []

    if not present:
        if require_locked and lock.get("status") != "locked":
            return {
                "ok": False,
                "status": "pending",
                "reason": "holdout lock is pending; record the frozen dataset first",
            }
        return {
            "ok": True,
            "status": "absent",
            "reason": "no holdout files present; nothing to verify",
            "dataset_id": lock.get("dataset_id"),
        }

    if lock.get("status") != "locked" or not lock.get("files"):
        return {
            "ok": False,
            "status": "unlocked",
            "reason": "holdout files are present but no locked fingerprint is recorded",
        }

    actual = fingerprint_directory(directory)
    expected_files = lock.get("files") or {}
    mismatched = sorted(
        name
        for name in set(expected_files) | set(actual["files"])
        if expected_files.get(name) != actual["files"].get(name)
    )
    if mismatched:
        return {
            "ok": False,
            "status": "modified",
            "reason": "holdout files do not match the locked snapshot: "
            + ", ".join(mismatched),
            "expected_dataset_id": lock.get("dataset_id"),
            "actual_dataset_id": actual["dataset_id"],
        }
    if actual["dataset_id"] != lock.get("dataset_id"):
        return {
            "ok": False,
            "status": "modified",
            "reason": "combined holdout dataset id does not match the lock",
            "expected_dataset_id": lock.get("dataset_id"),
            "actual_dataset_id": actual["dataset_id"],
        }
    return {
        "ok": True,
        "status": "verified",
        "reason": "holdout matches the locked SHA-256 snapshot",
        "dataset_id": actual["dataset_id"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="holdout_lock")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="fingerprint and lock the frozen holdout")
    record.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    record.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    record.add_argument("--symbols", nargs="+", default=None)

    check = sub.add_parser("verify", help="verify the holdout against the lock")
    check.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    check.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    check.add_argument(
        "--require-locked",
        action="store_true",
        help="fail when the lock is still pending (use on the production host)",
    )

    args = parser.parse_args(argv)
    if args.command == "record":
        try:
            manifest = build_manifest(args.dir, args.symbols or [])
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        write_lock(manifest, args.lock)
        print(
            f"Locked {len(manifest['files'])} holdout files "
            f"dataset_id={manifest['dataset_id']}"
        )
        return 0

    result = verify(args.dir, args.lock, require_locked=args.require_locked)
    stream = sys.stdout if result["ok"] else sys.stderr
    print(f"{'PASS' if result['ok'] else 'FAIL'}: {result['reason']}", file=stream)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
