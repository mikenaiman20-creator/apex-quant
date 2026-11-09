"""Shared, bind-mount-safe filesystem helpers.

Containers and host services write the same runtime directories under
``logs/`` and ``data/`` with different UIDs.  Every writer therefore:

* creates missing parent directories;
* writes through a same-directory temporary file and ``os.replace``;
* applies an explicit shared mode after creation (instead of inheriting a
  restrictive umask), and
* keeps failures best-effort where telemetry must never crash a service.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, IO

SHARED_FILE_MODE = 0o666
SHARED_DIR_MODE = 0o777
SENSITIVE_FILE_MODE = 0o600


def apply_shared_umask(mask: int = 0o002) -> None:
    """Prefer group-writable files for shared volumes; ignore platforms without umask."""
    try:
        os.umask(mask)
    except (AttributeError, OSError, ValueError):
        pass


def chmod_best_effort(path: str | Path, mode: int) -> bool:
    try:
        os.chmod(path, mode)
        return True
    except (NotImplementedError, OSError):
        return False


def ensure_directory(path: str | Path, mode: int = SHARED_DIR_MODE) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    chmod_best_effort(directory, mode)
    return directory


def secure_file(path: str | Path, mode: int = SHARED_FILE_MODE) -> Path:
    target = Path(path)
    chmod_best_effort(target, mode)
    return target


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    mode: int = SHARED_FILE_MODE,
    encoding: str = "utf-8",
) -> None:
    target = Path(path)
    if str(target.parent) not in {"", "."}:
        ensure_directory(target.parent)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        secure_file(temporary, mode)
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        secure_file(target, mode)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = None,
    allow_nan: bool = False,
    mode: int = SHARED_FILE_MODE,
) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, allow_nan=allow_nan) + "\n",
        mode=mode,
    )


def open_append_text(
    path: str | Path,
    *,
    mode: int = SHARED_FILE_MODE,
    encoding: str = "utf-8",
) -> IO[str]:
    target = Path(path)
    if str(target.parent) not in {"", "."}:
        ensure_directory(target.parent)
    handle = target.open("a", encoding=encoding)
    secure_file(target, mode)
    return handle


def safe_is_file(path: str | Path) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


def safe_is_dir(path: str | Path) -> bool:
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def safe_stat(path: str | Path):
    try:
        return Path(path).stat()
    except OSError:
        return None
