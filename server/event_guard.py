"""Local high-impact event and broad-index volatility circuit breaker."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GuardDecision:
    blocked: bool
    reason: str = ""
    event: str | None = None
    until: str | None = None


class MacroEventGuard:
    def __init__(
        self,
        calendar_path: Path,
        buffer_minutes: int = 15,
        volatility_spike_pct: float = 0.01,
    ):
        if buffer_minutes < 0:
            raise ValueError("buffer_minutes must be non-negative")
        if not math.isfinite(volatility_spike_pct) or volatility_spike_pct <= 0:
            raise ValueError("volatility_spike_pct must be finite and positive")
        self.calendar_path = calendar_path
        self.buffer = timedelta(minutes=buffer_minutes)
        self.volatility_spike_pct = volatility_spike_pct
        self.volatility_pause_until: datetime | None = None
        self.last_spike_bar: datetime | None = None

    def _events(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self.calendar_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
            raise ValueError(f"Macro calendar must be an object with an events list: {self.calendar_path}")
        return raw["events"]

    @staticmethod
    def _event_time(event: dict[str, Any]) -> datetime:
        value = event.get("timestamp")
        if not isinstance(value, str):
            raise ValueError("Macro event requires an ISO-8601 timestamp")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Macro event timestamps must include an explicit timezone")
        return parsed.astimezone(timezone.utc)

    def check(self, now: datetime, index_bars: list[Any] | None = None) -> GuardDecision:
        if now.tzinfo is None:
            raise ValueError("Macro guard time must be timezone-aware")
        now = now.astimezone(timezone.utc)
        for event in self._events():
            if not isinstance(event, dict):
                raise ValueError("Macro calendar event entries must be objects")
            impact = str(event.get("impact", "high")).lower()
            if impact != "high":
                continue
            timestamp = self._event_time(event)
            if timestamp - self.buffer <= now <= timestamp + self.buffer:
                label = str(event.get("name", "high-impact macro release"))
                return GuardDecision(
                    True,
                    f"macro event window: {label}",
                    label,
                    (timestamp + self.buffer).isoformat(),
                )

        if self.volatility_pause_until and now < self.volatility_pause_until:
            return GuardDecision(
                True,
                "broad-index volatility cooldown after SPY spike",
                "SPY volatility",
                self.volatility_pause_until.isoformat(),
            )
        if index_bars and len(index_bars) >= 2:
            recent = index_bars[-16:]
            first = recent[0].close
            last = recent[-1].close
            latest_start = recent[-1].start
            latest_start = (
                latest_start.astimezone(timezone.utc)
                if latest_start.tzinfo else latest_start.replace(tzinfo=timezone.utc)
            )
            if first > 0 and timedelta(0) <= now - latest_start <= timedelta(minutes=2):
                move = abs(last - first) / first
                if move >= self.volatility_spike_pct:
                    latest_bar = latest_start
                    if self.last_spike_bar != latest_bar:
                        self.volatility_pause_until = now + self.buffer
                        self.last_spike_bar = latest_bar
                    return GuardDecision(
                        True,
                        f"broad-index volatility spike: SPY moved {move:.2%} across recent bars",
                        "SPY volatility",
                        self.volatility_pause_until.isoformat(),
                    )
        return GuardDecision(False)
