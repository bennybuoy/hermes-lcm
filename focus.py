"""Persisted immutable focus-overlay records."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from .db_bootstrap import configure_connection, run_versioned_migrations


@dataclass(frozen=True)
class FocusBrief:
    focus_id: int
    conversation_id: str
    session_id: str
    prompt: str
    content: str
    source_node_ids: tuple[int, ...]
    covered_generation: int
    covered_store_id: int
    token_count: int
    created_at: float
    active: bool
    supersedes_focus_id: int | None = None

    def metadata(self, *, preview_chars: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = {
            "focus_id": self.focus_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "source_node_ids": list(self.source_node_ids),
            "covered_generation": self.covered_generation,
            "covered_store_id": self.covered_store_id,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "active": self.active,
            "supersedes_focus_id": self.supersedes_focus_id,
        }
        if preview_chars > 0:
            limit = max(1, min(int(preview_chars), 1000))
            result["preview"] = self.content[:limit]
            result["preview_truncated"] = len(self.content) > limit
        return result


class FocusStore:
    def __init__(self, db_path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        run_versioned_migrations(self._conn)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _row(row) -> FocusBrief | None:
        if row is None:
            return None
        try:
            source_ids = tuple(int(value) for value in json.loads(row[5] or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            source_ids = ()
        return FocusBrief(
            focus_id=int(row[0]),
            conversation_id=str(row[1]),
            session_id=str(row[2]),
            prompt=str(row[3]),
            content=str(row[4]),
            source_node_ids=source_ids,
            covered_generation=int(row[6] or 0),
            covered_store_id=int(row[7] or 0),
            token_count=int(row[8] or 0),
            created_at=float(row[9] or 0),
            active=bool(row[10]),
            supersedes_focus_id=int(row[11]) if row[11] is not None else None,
        )

    def get_active(self, conversation_id: str) -> FocusBrief | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT focus_id, conversation_id, session_id, prompt, content,
                       source_node_ids, covered_generation, covered_store_id,
                       token_count, created_at, active, supersedes_focus_id
                FROM lcm_focus_briefs
                WHERE conversation_id = ? AND active = 1
                ORDER BY focus_id DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return self._row(row)

    def get(self, focus_id: int) -> FocusBrief | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT focus_id, conversation_id, session_id, prompt, content,
                       source_node_ids, covered_generation, covered_store_id,
                       token_count, created_at, active, supersedes_focus_id
                FROM lcm_focus_briefs WHERE focus_id = ?
                """,
                (int(focus_id),),
            ).fetchone()
        return self._row(row)

    def publish(
        self,
        *,
        conversation_id: str,
        session_id: str,
        prompt: str,
        content: str,
        source_node_ids: list[int],
        covered_generation: int,
        covered_store_id: int,
        token_count: int,
        supersedes_focus_id: int | None = None,
    ) -> FocusBrief:
        """Deactivate the prior record and insert one immutable replacement."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                active = self._conn.execute(
                    "SELECT focus_id FROM lcm_focus_briefs WHERE conversation_id = ? AND active = 1",
                    (conversation_id,),
                ).fetchone()
                active_id = int(active[0]) if active else None
                if supersedes_focus_id is not None and active_id != int(supersedes_focus_id):
                    raise ValueError("focus changed before replacement publication")
                self._conn.execute(
                    "UPDATE lcm_focus_briefs SET active = 0 WHERE conversation_id = ? AND active = 1",
                    (conversation_id,),
                )
                cur = self._conn.execute(
                    """
                    INSERT INTO lcm_focus_briefs
                        (conversation_id, session_id, prompt, content,
                         source_node_ids, covered_generation, covered_store_id,
                         token_count, created_at, active, supersedes_focus_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        conversation_id,
                        session_id,
                        prompt,
                        content,
                        json.dumps([int(value) for value in source_node_ids]),
                        int(covered_generation),
                        int(covered_store_id),
                        int(token_count),
                        time.time(),
                        supersedes_focus_id,
                    ),
                )
                focus_id = int(cur.lastrowid)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        brief = self.get(focus_id)
        if brief is None:  # pragma: no cover - defensive
            raise RuntimeError("published focus brief could not be reloaded")
        return brief

    def unfocus(self, conversation_id: str) -> FocusBrief | None:
        with self._lock:
            active = self.get_active(conversation_id)
            if active is None:
                return None
            self._conn.execute(
                "UPDATE lcm_focus_briefs SET active = 0 WHERE focus_id = ? AND active = 1",
                (active.focus_id,),
            )
            self._conn.commit()
            return self.get(active.focus_id)

    def history(self, conversation_id: str, *, limit: int = 20) -> list[FocusBrief]:
        bounded = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT focus_id, conversation_id, session_id, prompt, content,
                       source_node_ids, covered_generation, covered_store_id,
                       token_count, created_at, active, supersedes_focus_id
                FROM lcm_focus_briefs WHERE conversation_id = ?
                ORDER BY focus_id DESC LIMIT ?
                """,
                (conversation_id, bounded),
            ).fetchall()
        return [brief for brief in (self._row(row) for row in rows) if brief is not None]
