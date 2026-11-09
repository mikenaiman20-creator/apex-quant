#!/usr/bin/env python3
"""Container bootstrap: create shared runtime directories, relax bind-mount
permissions, then drop privileges before executing the service command.

Bind mounts are created by the Docker daemon as ``root`` and may be owned by a
host UID that differs from the container application UID.  chmod/chown are
therefore attempted best-effort: read-only dashboard mounts intentionally fail
and the service still starts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SHARED_DIR_MODE = 0o777
SHARED_FILE_MODE = 0o666
SHARED_UMASK = 0o002
# The risk gateway deliberately keeps its tamper-evident audit log and state
# private (0o600).  Relaxing the shared runtime tree must never widen these.
SENSITIVE_FILE_NAMES = {
    "risk_audit.jsonl",
    "risk_audit.jsonl.anchor",
    "risk_engine_state.json",
}


def _is_sensitive(name: str) -> bool:
    return name in SENSITIVE_FILE_NAMES


def _relax(path: Path) -> None:
    try:
        if path.is_dir():
            os.chmod(path, SHARED_DIR_MODE)
    except OSError:
        return
    try:
        for root, directories, files in os.walk(path):
            for name in directories:
                try:
                    os.chmod(Path(root) / name, SHARED_DIR_MODE)
                except OSError:
                    pass
            for name in files:
                if _is_sensitive(name):
                    continue
                try:
                    os.chmod(Path(root) / name, SHARED_FILE_MODE)
                except OSError:
                    pass
    except OSError:
        pass


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    for root, directories, files in os.walk(path):
        for name in (None, *directories, *files):
            target = Path(root) if name is None else Path(root) / name
            try:
                os.chown(target, uid, gid)
            except OSError:
                pass


def main(argv: list[str]) -> int:
    if not argv:
        print("entrypoint: no command supplied", file=sys.stderr)
        return 2
    app_root = Path(os.getenv("APEX_APP_ROOT", "/app"))
    for name in ("logs", "data"):
        directory = app_root / name
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        _relax(directory)

    os.umask(SHARED_UMASK)

    uid_text = os.getenv("APEX_RUNTIME_UID", "").strip()
    gid_text = os.getenv("APEX_RUNTIME_GID", "").strip()
    if getattr(os, "geteuid", lambda: 1)() == 0 and uid_text.isdigit():
        uid = int(uid_text)
        gid = int(gid_text) if gid_text.isdigit() else uid
        for name in ("logs", "data"):
            _chown_tree(app_root / name, uid, gid)
        try:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
        except OSError as exc:
            print(f"entrypoint: unable to drop privileges: {exc}", file=sys.stderr)
            return 1

    try:
        os.execvp(argv[0], argv)
    except OSError as exc:
        print(f"entrypoint: unable to execute {argv[0]}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
