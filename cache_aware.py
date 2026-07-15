"""Explicit hot-cache signal contract.

Observed provider usage counters are intentionally absent here.  Only an
explicit host cache-write or cache-break event can change this state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class CacheAwareSignalTracker:
    state: str = "unknown"
    route_fingerprint: str = ""
    conversation_id: str = ""
    source: str = "none"
    recorded_at: float = 0.0
    expires_at: float = 0.0

    def record(
        self,
        event: str,
        *,
        route_fingerprint: str,
        source: str,
        ttl_seconds: float = 0.0,
        conversation_id: str = "",
    ) -> None:
        normalized = str(event or "").strip().lower()
        if normalized not in {"write", "break"}:
            raise ValueError("cache signal event must be write or break")
        normalized_source = str(source or "").strip()
        if not normalized_source:
            raise ValueError("cache signal source is required")
        now = time.monotonic()
        self.route_fingerprint = str(route_fingerprint or "")
        self.conversation_id = str(conversation_id or "")
        self.source = normalized_source
        self.recorded_at = now
        if normalized == "break":
            self.state = "broken"
            self.expires_at = 0.0
            return
        ttl = max(0.0, float(ttl_seconds or 0.0))
        self.state = "hot" if ttl > 0 else "unknown"
        self.expires_at = now + ttl if ttl > 0 else 0.0

    def status(
        self, *, route_fingerprint: str, conversation_id: str = ""
    ) -> dict[str, object]:
        current_route = str(route_fingerprint or "")
        state = self.state
        now = time.monotonic()
        remaining = 0.0
        if self.conversation_id and self.conversation_id != str(conversation_id or ""):
            state = "scope-mismatch"
        elif self.route_fingerprint and self.route_fingerprint != current_route:
            state = "route-mismatch"
        elif state == "hot":
            if self.expires_at <= now:
                state = "expired"
            else:
                remaining = self.expires_at - now
        return {
            "state": state,
            "source": self.source,
            "route_fingerprint": self.route_fingerprint,
            "conversation_id": self.conversation_id,
            "recorded_at_monotonic": self.recorded_at,
            "expires_at_monotonic": self.expires_at,
            "remaining_seconds": max(0.0, remaining),
            "contract": "explicit-host-write-or-break",
            "observed_usage_drives_state": False,
        }
