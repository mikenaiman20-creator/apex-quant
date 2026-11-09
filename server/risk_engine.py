"""Independent fail-closed risk gateway and tamper-evident audit log."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from server.reconciliation import reconcile_snapshot


log = logging.getLogger("apex.risk")
EMPTY_HASH = "0" * 64
SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
AUDIT_FIELDS = (
    "operation", "mode", "symbol", "side", "qty", "limit_price", "stop_price",
    "take_profit_price", "client_order_id", "order_count", "total_qty", "equity",
    "risk_usd", "risk_pct", "notional_usd", "reward_risk", "open_symbols",
    "order_ids", "symbols", "current_stop", "new_stop", "current_price", "purpose",
    "source", "phase", "signal_score", "position_count", "cash",
    "position_market_value", "calculated_equity", "equity_delta", "tolerance",
    "unmanaged_symbols", "actor", "reason", "override_reason",
)


class RiskGateError(RuntimeError):
    """Base error for a failed risk gateway call."""


class RiskGateUnavailable(RiskGateError):
    """The risk service could not be authenticated or its heartbeat was stale."""


class RiskGateRejected(RiskGateError):
    """The risk service rejected an operation."""


class AuditIntegrityError(RuntimeError):
    """The append-only audit log or its integrity anchor is invalid."""


def _canonical_json(record: dict[str, Any]) -> bytes:
    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _write_atomic(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temp, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic file write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temp, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


class RiskAuditLog:
    """Append-only JSONL audit log with a hash chain and separately stored tail anchor."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.anchor_path = self.path.with_name(f"{self.path.name}.anchor")
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence, self.last_hash = self._verify()
        self._failed = False

    def _verify(self) -> tuple[int, str]:
        sequence = 0
        previous_hash = EMPTY_HASH
        if self.path.exists():
            with self.path.open("rb") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.endswith(b"\n"):
                        raise AuditIntegrityError(
                            f"Risk audit line {line_number} is incomplete."
                        )
                    try:
                        envelope = json.loads(raw_line)
                        entry_hash = envelope.pop("entry_hash")
                        record = envelope
                        expected_sequence = sequence + 1
                        if record.get("sequence") != expected_sequence:
                            raise ValueError("non-contiguous sequence")
                        if record.get("prev_hash") != previous_hash:
                            raise ValueError("previous hash mismatch")
                        computed_hash = hashlib.sha256(
                            previous_hash.encode("ascii") + _canonical_json(record)
                        ).hexdigest()
                        if not hmac.compare_digest(str(entry_hash), computed_hash):
                            raise ValueError("entry hash mismatch")
                    except (
                        AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError
                    ) as exc:
                        raise AuditIntegrityError(
                            f"Risk audit integrity check failed at line {line_number}."
                        ) from exc
                    sequence = expected_sequence
                    previous_hash = computed_hash

        if self.anchor_path.exists():
            try:
                anchor = json.loads(self.anchor_path.read_text(encoding="utf-8"))
                if anchor != {"sequence": sequence, "entry_hash": previous_hash}:
                    raise ValueError("audit tail does not match anchor")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise AuditIntegrityError("Risk audit tail anchor is invalid.") from exc
        elif sequence:
            raise AuditIntegrityError("Risk audit tail anchor is missing.")
        return sequence, previous_hash

    def append(self, event_type: str, details: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._failed:
                raise AuditIntegrityError("Risk audit writer is latched failed.")
            record = {
                "sequence": self.sequence + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event_type,
                "details": details,
                "prev_hash": self.last_hash,
            }
            entry_hash = hashlib.sha256(
                self.last_hash.encode("ascii") + _canonical_json(record)
            ).hexdigest()
            envelope = {**record, "entry_hash": entry_hash}
            line = _canonical_json(envelope) + b"\n"
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, flags, 0o600)
                try:
                    os.fchmod(descriptor, 0o600)
                    view = memoryview(line)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("risk audit append made no progress")
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _write_atomic(
                    self.anchor_path,
                    _canonical_json({
                        "sequence": record["sequence"], "entry_hash": entry_hash,
                    }),
                )
            except OSError:
                self._failed = True
                raise
            self.sequence = record["sequence"]
            self.last_hash = entry_hash
            return envelope


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _symbol(value: Any) -> str:
    if not isinstance(value, str) or not SYMBOL_PATTERN.fullmatch(value):
        raise ValueError("symbol is invalid.")
    return value


def _audit_safe(details: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in AUDIT_FIELDS:
        value = details.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [
                item for item in value
                if (
                    not isinstance(item, float) or math.isfinite(item)
                ) and (isinstance(item, (str, int, float, bool)) or item is None)
            ][:100]
    return safe


class RiskEngineService:
    def __init__(
        self,
        *,
        gate_token: str,
        admin_token: str,
        audit_path: str | Path,
        state_path: str | Path,
        heartbeat_path: str | Path,
        allowed_symbols: set[str],
        max_risk_pct: float = 0.005,
        max_notional_pct: float = 0.20,
        max_open_positions: int = 3,
        min_reward_risk: float = 1.5,
        tolerance_usd: float = 5.0,
        tolerance_pct: float = 0.001,
    ):
        if len(gate_token) < 32 or len(admin_token) < 32 or hmac.compare_digest(
            gate_token, admin_token,
        ):
            raise ValueError("Risk gateway and admin tokens must be distinct and at least 32 characters.")
        self.gate_token = gate_token
        self.admin_token = admin_token
        self.audit = RiskAuditLog(audit_path)
        self.state_path = Path(state_path)
        self.heartbeat_path = Path(heartbeat_path)
        self.allowed_symbols = allowed_symbols
        self.max_risk_pct = max_risk_pct
        self.max_notional_pct = max_notional_pct
        self.max_open_positions = max_open_positions
        self.min_reward_risk = min_reward_risk
        self.tolerance_usd = tolerance_usd
        self.tolerance_pct = tolerance_pct
        self._state_lock = threading.Lock()
        self._heartbeat_lock = threading.Lock()
        self.heartbeat_at = 0.0
        self.blocked = False
        self.block_reason = ""
        if self.state_path.exists():
            try:
                persisted = json.loads(self.state_path.read_text(encoding="utf-8"))
                if persisted.get("blocked") is not True or not isinstance(
                    persisted.get("reason"), str
                ):
                    raise ValueError("invalid risk state")
                self.blocked = True
                self.block_reason = persisted["reason"]
            except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError("Risk engine state is invalid; refusing to start.") from exc

    @classmethod
    def from_env(cls) -> "RiskEngineService":
        def number(name: str, default: float) -> float:
            try:
                value = float(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ValueError(f"{name} must be numeric.") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")
            return value

        allowed = {
            item.strip().upper() for item in
            os.getenv("RISK_ALLOWED_SYMBOLS", "SPY,QQQ,NVDA").split(",")
            if item.strip()
        }
        if not allowed or any(not SYMBOL_PATTERN.fullmatch(item) for item in allowed):
            raise ValueError("RISK_ALLOWED_SYMBOLS must contain valid uppercase symbols.")
        max_risk_pct = number("RISK_MAX_PER_TRADE_PCT", 0.005)
        max_notional_pct = number("RISK_MAX_POSITION_NOTIONAL_PCT", 0.20)
        max_positions = int(os.getenv("RISK_MAX_OPEN_POSITIONS", "3"))
        min_reward_risk = number("RISK_MIN_REWARD_RISK", 1.5)
        tolerance_usd = number("RISK_RECONCILIATION_TOLERANCE_USD", 5.0)
        tolerance_pct = number("RISK_RECONCILIATION_TOLERANCE_PCT", 0.001)
        if (
            max_risk_pct > 0.01
            or max_notional_pct > 0.20
            or not 1 <= max_positions <= 3
            or min_reward_risk < 1.0
            or tolerance_usd > 100.0
            or tolerance_pct > 0.01
        ):
            raise ValueError("Risk limits exceed the documented hard limits.")
        return cls(
            gate_token=os.getenv("RISK_GATE_TOKEN", ""),
            admin_token=os.getenv("RISK_ADMIN_TOKEN", ""),
            audit_path=os.getenv("RISK_AUDIT_PATH", "/app/logs/risk_audit.jsonl"),
            state_path=os.getenv("RISK_STATE_PATH", "/app/logs/risk_engine_state.json"),
            heartbeat_path=os.getenv(
                "RISK_HEARTBEAT_PATH", "/app/logs/risk_engine_heartbeat.json",
            ),
            allowed_symbols=allowed,
            max_risk_pct=max_risk_pct,
            max_notional_pct=max_notional_pct,
            max_open_positions=max_positions,
            min_reward_risk=min_reward_risk,
            tolerance_usd=tolerance_usd,
            tolerance_pct=tolerance_pct,
        )

    def refresh_heartbeat(self) -> None:
        with self._heartbeat_lock:
            now = time.time()
            _write_atomic(
                self.heartbeat_path,
                _canonical_json({
                    "timestamp": now,
                    "status": "blocked" if self.blocked else "ready",
                }),
            )
            self.heartbeat_at = now

    def start_heartbeat(self, interval: float = 2.0) -> threading.Event:
        self.refresh_heartbeat()
        stop = threading.Event()

        def run() -> None:
            while not stop.wait(interval):
                try:
                    self.refresh_heartbeat()
                except OSError:
                    log.exception("Risk engine heartbeat persistence failed.")

        threading.Thread(target=run, name="risk-engine-heartbeat", daemon=True).start()
        return stop

    def _persist_block(self) -> None:
        _write_atomic(self.state_path, _canonical_json({
            "blocked": True,
            "reason": self.block_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    def _latch(self, reason: str) -> None:
        with self._state_lock:
            self.blocked = True
            self.block_reason = reason
            self._persist_block()
            self.refresh_heartbeat()

    def health(self) -> dict[str, Any]:
        return {
            "status": "blocked" if self.blocked else "ready",
            "heartbeat_at": self.heartbeat_at,
            "blocked_reason": self.block_reason if self.blocked else None,
        }

    def _decision(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        approved: bool,
        reason: str,
        details: dict[str, Any],
        event_type: str = "risk_check",
    ) -> dict[str, Any]:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            request_id = str(uuid.uuid4())
        event_details = {
            "request_id": request_id,
            "operation": operation,
            "approved": approved,
            "reason": reason,
            **_audit_safe(details),
        }
        self.audit.append(event_type, event_details)
        return {
            "request_id": request_id,
            "approved": approved,
            "reason": reason,
            "details": _audit_safe(details),
        }

    def authorize(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if not isinstance(operation, str):
            operation = "invalid"
        if request.get("override") is True or request.get("manual_override") is True:
            return self._decision(
                request, operation=operation, approved=False,
                reason="manual overrides are disabled; the request was logged",
                details={
                    "actor": request.get("actor"),
                    "override_reason": request.get("override_reason"),
                    "reason": "manual override requested",
                },
                event_type="risk_override_rejected",
            )
        if self.blocked:
            return self._decision(
                request, operation=operation, approved=False,
                reason=f"risk engine is latched blocked: {self.block_reason}",
                details={"reason": self.block_reason},
            )
        try:
            details = self._evaluate(request)
            return self._decision(
                request, operation=operation, approved=True,
                reason="risk policy approved the request", details=details,
            )
        except ValueError as exc:
            return self._decision(
                request, operation=operation, approved=False,
                reason=str(exc), details={"reason": str(exc)},
            )

    def _evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "submit_batch":
            return self._evaluate_order_batch(request)
        if operation == "update_stop":
            symbol = _symbol(request.get("symbol"))
            if symbol not in self.allowed_symbols:
                raise ValueError("symbol is not in the risk allowlist.")
            side = request.get("side")
            if side not in {"long", "short"}:
                raise ValueError("position side must be long or short.")
            current_stop = _finite_number(request.get("current_stop"), "current_stop")
            new_stop = _finite_number(request.get("new_stop"), "new_stop")
            current_price = _finite_number(request.get("current_price"), "current_price")
            if side == "long" and not current_stop < new_stop < current_price:
                raise ValueError("long stop update must tighten risk and stay below market.")
            if side == "short" and not current_price < new_stop < current_stop:
                raise ValueError("short stop update must tighten risk and stay above market.")
            order_ids = request.get("order_ids")
            if not isinstance(order_ids, list) or not order_ids or not all(
                isinstance(item, str) and item for item in order_ids
            ):
                raise ValueError("stop update requires broker order identifiers.")
            return {
                "symbol": symbol, "side": side, "current_stop": current_stop,
                "new_stop": new_stop, "current_price": current_price,
                "order_ids": order_ids,
            }
        if operation == "close_position":
            symbol = _symbol(request.get("symbol"))
            if symbol not in self.allowed_symbols:
                raise ValueError("symbol is not in the risk allowlist.")
            return {"symbol": symbol, "purpose": request.get("purpose", "risk_reduction")}
        if operation == "cancel_orders":
            symbols = request.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ValueError("cancel request requires at least one symbol.")
            checked = [_symbol(item) for item in symbols]
            return {"symbols": checked, "purpose": request.get("purpose", "risk_reduction")}
        raise ValueError("unsupported order operation.")

    def _evaluate_order_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("mode") != "PAPER":
            raise ValueError("autonomous order routing is restricted to PAPER mode.")
        orders = request.get("orders")
        if not isinstance(orders, list) or not 1 <= len(orders) <= 10:
            raise ValueError("order batch must contain between 1 and 10 orders.")
        equity = _finite_number(request.get("account_equity"), "account_equity")
        if equity <= 0:
            raise ValueError("account equity must be greater than zero.")
        open_symbols_value = request.get("open_symbols", [])
        if not isinstance(open_symbols_value, list):
            raise ValueError("open_symbols must be a list.")
        open_symbols = {_symbol(item) for item in open_symbols_value}
        symbols: set[str] = set()
        sides: set[str] = set()
        client_ids: set[str] = set()
        total_qty = 0
        total_risk = 0.0
        total_notional = 0.0
        total_reward = 0.0
        for order in orders:
            if not isinstance(order, dict):
                raise ValueError("order batch contains an invalid order.")
            symbol = _symbol(order.get("symbol"))
            if symbol not in self.allowed_symbols:
                raise ValueError(f"{symbol} is not in the risk allowlist.")
            side = order.get("side")
            if side not in {"buy", "sell"}:
                raise ValueError("order side must be buy or sell.")
            qty_value = order.get("qty")
            if isinstance(qty_value, bool):
                raise ValueError("order quantity must be a positive whole number.")
            try:
                qty = int(qty_value)
                if float(qty_value) != qty or qty <= 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("order quantity must be a positive whole number.") from exc
            entry = _finite_number(order.get("limit_price"), "limit_price")
            stop = _finite_number(order.get("stop_price"), "stop_price")
            target = _finite_number(order.get("take_profit_price"), "take_profit_price")
            if min(entry, stop, target) <= 0:
                raise ValueError("order prices must be greater than zero.")
            if side == "buy" and not stop < entry < target:
                raise ValueError("buy bracket must have stop < entry < target.")
            if side == "sell" and not target < entry < stop:
                raise ValueError("sell bracket must have target < entry < stop.")
            client_id = order.get("client_order_id")
            if not isinstance(client_id, str) or not client_id or client_id in client_ids:
                raise ValueError("each order requires a unique client_order_id.")
            symbols.add(symbol)
            sides.add(side)
            client_ids.add(client_id)
            risk = qty * abs(entry - stop)
            reward = qty * abs(target - entry)
            total_qty += qty
            total_risk += risk
            total_reward += reward
            total_notional += qty * entry
        if len(symbols) != 1 or len(sides) != 1:
            raise ValueError("all orders in a batch must share one symbol and side.")
        symbol = next(iter(symbols))
        if symbol in open_symbols:
            raise ValueError("symbol already has an open position or order.")
        if len(open_symbols) >= self.max_open_positions:
            raise ValueError("maximum open position count would be exceeded.")
        risk_pct = total_risk / equity
        notional_pct = total_notional / equity
        reward_risk = total_reward / total_risk if total_risk else 0.0
        if risk_pct > self.max_risk_pct:
            raise ValueError("order batch exceeds the independent per-trade risk limit.")
        if notional_pct > self.max_notional_pct:
            raise ValueError("order batch exceeds the independent notional limit.")
        if reward_risk < self.min_reward_risk:
            raise ValueError("order batch does not meet the minimum reward/risk ratio.")
        return {
            "operation": "submit_batch",
            "mode": "PAPER",
            "symbol": symbol,
            "side": next(iter(sides)),
            "order_count": len(orders),
            "total_qty": total_qty,
            "equity": equity,
            "risk_usd": total_risk,
            "risk_pct": risk_pct,
            "notional_usd": total_notional,
            "reward_risk": reward_risk,
            "open_symbols": sorted(open_symbols),
            "client_order_id": ",".join(sorted(client_ids)),
        }

    def reconcile(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            report = reconcile_snapshot(
                request.get("account"),
                request.get("positions"),
                mode=request.get("mode"),
                target_symbols=request.get("target_symbols", ()),
                managed_symbols=request.get("managed_symbols"),
                tolerance_usd=self.tolerance_usd,
                tolerance_pct=self.tolerance_pct,
            )
            approved = report["approved"] is True
            reason = report["reason"]
        except (TypeError, ValueError) as exc:
            report = {"approved": False, "reason": str(exc)}
            approved = False
            reason = str(exc)
        response = self._decision(
            request,
            operation="reconcile",
            approved=approved,
            reason=reason,
            details={
                "mode": report.get("mode"),
                "source": request.get("source"),
                "position_count": report.get("position_count"),
                "equity": report.get("equity"),
                "cash": report.get("cash"),
                "position_market_value": report.get("position_market_value"),
                "calculated_equity": report.get("calculated_equity"),
                "equity_delta": report.get("equity_delta"),
                "tolerance": report.get("tolerance"),
                "unmanaged_symbols": report.get("unmanaged_symbols", []),
                "reason": reason,
            },
            event_type="reconciliation",
        )
        response["details"] = _audit_safe({
            **response["details"],
            "position_count": report.get("position_count"),
            "equity": report.get("equity"),
            "cash": report.get("cash"),
            "position_market_value": report.get("position_market_value"),
            "calculated_equity": report.get("calculated_equity"),
            "equity_delta": report.get("equity_delta"),
            "tolerance": report.get("tolerance"),
            "unmanaged_symbols": report.get("unmanaged_symbols", []),
        })
        if not approved:
            self._latch(reason)
        return response

    def reset_block(
        self, request: dict[str, Any], supplied_admin_token: str,
    ) -> dict[str, Any]:
        actor = request.get("actor")
        reason = request.get("reason")
        if not isinstance(actor, str) or not actor.strip() or not isinstance(
            reason, str
        ) or not reason.strip() or len(actor) > 120 or len(reason) > 500:
            raise ValueError("an operator actor and reason are required to reset the risk latch.")
        if not hmac.compare_digest(supplied_admin_token, self.admin_token):
            self._decision(
                request, operation="risk_latch_reset", approved=False,
                reason="invalid admin credential",
                details={"actor": actor, "reason": "invalid admin credential"},
                event_type="risk_override_rejected",
            )
            raise PermissionError("risk latch reset was not authorized.")
        self.audit.append("risk_override", {
            "request_id": request.get("request_id", str(uuid.uuid4())),
            "operation": "risk_latch_reset",
            "approved": True,
            "actor": actor.strip(),
            "reason": reason.strip(),
            "previous_block_reason": self.block_reason,
        })
        with self._state_lock:
            try:
                self.state_path.unlink()
            except FileNotFoundError:
                pass
            self.blocked = False
            self.block_reason = ""
            self.refresh_heartbeat()
        return {
            "request_id": request.get("request_id"),
            "approved": True,
            "reason": "risk latch reset; operator override recorded",
        }


class RiskRequestHandler(BaseHTTPRequestHandler):
    server: "RiskHTTPServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("risk_http %s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required.")
        length = int(raw_length)
        if not 0 < length <= 1_048_576:
            raise ValueError("request body size is invalid.")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object.")
        return payload

    def _authorized(self, expected: str) -> bool:
        authorization = self.headers.get("Authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied, expected)

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            if not self._authorized(self.server.risk_service.gate_token):
                self._send(401, {"error": "unauthorized"})
                return
        elif self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.server.risk_service.health())

    def do_POST(self) -> None:
        service = self.server.risk_service
        is_reset = self.path == "/v1/admin/reset"
        if self.path not in {"/v1/authorize", "/v1/reconcile", "/v1/admin/reset"}:
            self._send(404, {"error": "not found"})
            return
        expected = service.admin_token if is_reset else service.gate_token
        if not self._authorized(expected):
            if is_reset:
                try:
                    service.audit.append("risk_override_rejected", {
                        "operation": "risk_latch_reset",
                        "approved": False,
                        "reason": "invalid admin credential",
                    })
                except OSError:
                    log.exception("Unable to audit an unauthorized risk reset attempt.")
                    self._send(503, {"error": "risk audit unavailable"})
                    return
            self._send(401, {"error": "unauthorized"})
            return
        try:
            request = self._read_json()
            if self.path == "/v1/authorize":
                response = service.authorize(request)
            elif self.path == "/v1/reconcile":
                response = service.reconcile(request)
            else:
                response = service.reset_block(
                    request, self.headers.get("Authorization", "")[7:],
                )
            self._send(200, response)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except PermissionError as exc:
            self._send(403, {"error": str(exc)})
        except (OSError, RuntimeError) as exc:
            log.exception("Risk service operation failed.")
            self._send(503, {"error": f"risk service unavailable: {type(exc).__name__}"})


class RiskHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: RiskEngineService):
        self.risk_service = service
        super().__init__(address, RiskRequestHandler)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class RiskGateClient:
    """Host-side client restricted to a loopback risk endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        admin_token: str | None = None,
        *,
        timeout: float = 2.0,
        max_heartbeat_age: float = 10.0,
    ):
        self.base_url = (base_url or os.getenv(
            "RISK_GATE_URL", "http://127.0.0.1:18765"
        )).rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "http" or not parsed.hostname or parsed.port is None
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
        ):
            raise ValueError("Risk gateway URL must be a plain HTTP loopback address.")
        try:
            is_loopback = parsed.hostname.lower() == "localhost" or ipaddress.ip_address(
                parsed.hostname
            ).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("Risk gateway URL must resolve to a loopback address.")
        self.token = token if token is not None else os.getenv("RISK_GATE_TOKEN", "")
        self.admin_token = (
            admin_token if admin_token is not None else os.getenv("RISK_ADMIN_TOKEN", "")
        )
        self.timeout = timeout
        self.max_heartbeat_age = max_heartbeat_age
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None,
        *, token: str = "",
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body = None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise RiskGateUnavailable("Risk gateway request is not valid JSON.") from exc
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=headers, method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            status = getattr(exc, "code", None)
            suffix = f" (HTTP {status})" if status else ""
            raise RiskGateUnavailable(
                f"Risk gateway request failed{suffix}."
            ) from exc
        if not isinstance(result, dict):
            raise RiskGateUnavailable("Risk gateway returned an invalid response.")
        return result

    def _heartbeat(self, *, allow_blocked: bool) -> dict[str, Any]:
        if len(self.token) < 32:
            raise RiskGateUnavailable("Risk gateway authentication is not configured.")
        response = self._request("GET", "/v1/health", token=self.token)
        try:
            heartbeat = _finite_number(response.get("heartbeat_at"), "heartbeat_at")
        except ValueError as exc:
            raise RiskGateUnavailable("Risk gateway returned an invalid heartbeat.") from exc
        age = time.time() - heartbeat
        if (
            response.get("status") not in {"ready", "blocked"}
            or age < -3.0
            or age > self.max_heartbeat_age
        ):
            reason = response.get("blocked_reason") or "risk heartbeat is stale"
            raise RiskGateUnavailable(str(reason))
        if response.get("status") == "blocked" and not allow_blocked:
            raise RiskGateUnavailable(
                str(response.get("blocked_reason") or "risk gateway is blocked")
            )
        return response

    def health(self) -> dict[str, Any]:
        return self._heartbeat(allow_blocked=False)

    def authorize(self, request: dict[str, Any]) -> dict[str, Any]:
        if len(self.token) < 32:
            raise RiskGateUnavailable("Risk gateway authentication is not configured.")
        self.health()
        response = self._request(
            "POST", "/v1/authorize", request, token=self.token,
        )
        if not isinstance(response.get("approved"), bool):
            raise RiskGateUnavailable("Risk gateway returned no authorization decision.")
        return response

    def reconcile(self, request: dict[str, Any]) -> dict[str, Any]:
        if len(self.token) < 32:
            raise RiskGateUnavailable("Risk gateway authentication is not configured.")
        self._heartbeat(allow_blocked=True)
        response = self._request(
            "POST", "/v1/reconcile", request, token=self.token,
        )
        if not isinstance(response.get("approved"), bool):
            raise RiskGateUnavailable("Risk gateway returned no reconciliation decision.")
        return response

    def reset(self, *, actor: str, reason: str) -> dict[str, Any]:
        if len(self.admin_token) < 32:
            raise RiskGateUnavailable("Risk admin authentication is not configured.")
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required.")
        request = urllib.request.Request(
            self.base_url + "/v1/admin/reset",
            data=json.dumps({
                "request_id": str(uuid.uuid4()),
                "actor": actor.strip(),
                "reason": reason.strip(),
            }).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.admin_token}",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RiskGateUnavailable("Risk latch reset request failed.") from exc
        if not isinstance(result, dict) or result.get("approved") is not True:
            raise RiskGateRejected("Risk latch reset was not approved.")
        return result


def _serve(host: str, port: int) -> int:
    service = RiskEngineService.from_env()
    heartbeat_stop = service.start_heartbeat()
    try:
        server = RiskHTTPServer((host, port), service)
    except OSError:
        heartbeat_stop.set()
        raise
    log.info("Risk gateway listening on %s:%d.", host, port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        heartbeat_stop.set()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ApexQuant independent risk gateway.")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="start the risk gateway")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    subparsers.add_parser("check", help="check the authenticated gateway heartbeat")
    reset_parser = subparsers.add_parser("reset", help="audited operator reset of a risk latch")
    reset_parser.add_argument("--actor", required=True)
    reset_parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "check":
            from config.settings import load_project_env
            load_project_env()
            print(json.dumps(RiskGateClient().health(), sort_keys=True))
            return 0
        if args.command == "reset":
            from config.settings import load_project_env
            load_project_env()
            print(json.dumps(
                RiskGateClient().reset(actor=args.actor, reason=args.reason), sort_keys=True,
            ))
            return 0
        host = args.host if args.command == "serve" else "127.0.0.1"
        port = args.port if args.command == "serve" else 8765
        return _serve(host, port)
    except (RiskGateError, OSError, RuntimeError, ValueError) as exc:
        log.exception("Risk gateway command failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
