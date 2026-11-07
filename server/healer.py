"""Failure diagnosis hooks for supervised components; never changes source or risk policy."""
from __future__ import annotations

import logging
import threading
import traceback
from types import TracebackType

log = logging.getLogger("apex.healer")
_install_lock = threading.Lock()
_installed = False


def diagnose_exception(component: str, exception: BaseException,
                       tb: TracebackType | None = None) -> dict[str, str]:
    trace = "".join(traceback.format_exception(type(exception), exception, tb or exception.__traceback__))
    frames = traceback.extract_tb(tb or exception.__traceback__)
    module = frames[-1].filename if frames else "unknown"
    diagnosis = {
        "component": component,
        "module": module,
        "exception": type(exception).__name__,
        "message": str(exception)[:500],
        "traceback": trace[-12000:],
    }
    log.error("SELF-HEALING diagnosis component=%s module=%s exception=%s: %s",
              component, module, diagnosis["exception"], diagnosis["message"])
    log.debug("Failure traceback:\n%s", diagnosis["traceback"])
    return diagnosis


def install_exception_hooks() -> None:
    """Log otherwise-unhandled main-thread and worker-thread exceptions."""
    global _installed
    with _install_lock:
        if _installed:
            return
        previous_sys_hook = __import__("sys").excepthook
        previous_thread_hook = threading.excepthook

        def sys_hook(exc_type, exc_value, exc_tb):
            diagnose_exception("main", exc_value, exc_tb)
            previous_sys_hook(exc_type, exc_value, exc_tb)

        def thread_hook(args):
            diagnose_exception(args.thread.name or "worker", args.exc_value, args.exc_traceback)
            previous_thread_hook(args)

        __import__("sys").excepthook = sys_hook
        threading.excepthook = thread_hook
        _installed = True
