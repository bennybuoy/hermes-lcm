"""Persistent ordered active frontier and prepared-batch management.

Implements the candidate-based async compaction workflow:

1. ``prepare_background_compaction_once`` — snapshot the current source
   boundary, build leaf summaries privately, store a "ready" batch.
2. ``promote_prepared_compaction`` — CAS promotion at a turn boundary:
   validate source identity, policy fingerprint, and route fingerprint,
   then atomically insert canonical DAG nodes, advance the frontier
   generation, and mark the batch as promoted.
3. ``reject_prepared_compaction`` — mark a batch as rejected with a reason.

Candidates are invisible to canonical DAG/search/assembly until promoted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .db_bootstrap import ensure_frontier_tables, configure_connection, run_versioned_migrations

logger = logging.getLogger(__name__)


@dataclass
class PreparedBatch:
    """Result of a ``prepare_background_compaction_once`` call."""
    batch_id: int
    conversation_id: str
    session_id: str
    base_generation: int
    source_end_store_id: int
    source_identity_hash: str
    source_ids: list[int]
    policy_fingerprint: str
    route_fingerprint: str
    state: str  # preparing | ready | promoted | rejected | failed
    expected_leaf_count: int
    frontier_end_store_id: int
    failure_reason: str = ""


@dataclass
class PromotionResult:
    """Result of a ``promote_prepared_compaction`` call."""
    promoted: bool
    reason: str = ""
    batch_id: int = 0


class FrontierStore:
    """Manages the three frontier tables in the LCM SQLite database."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        configure_connection(self._conn)
        run_versioned_migrations(self._conn)
        ensure_frontier_tables(self._conn)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None
        return self._conn

    # -- Active frontiers -------------------------------------------------

    def get_active_frontier(self, conversation_id: str) -> dict[str, Any] | None:
        """Return the latest (highest generation) active frontier row."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT conversation_id, generation, session_id,
                       source_end_store_id, policy_fingerprint,
                       route_fingerprint, created_at, updated_at
                FROM lcm_active_frontiers
                WHERE conversation_id = ?
                ORDER BY generation DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "conversation_id": row[0],
            "generation": row[1],
            "session_id": row[2],
            "source_end_store_id": row[3],
            "policy_fingerprint": row[4],
            "route_fingerprint": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def ensure_frontier(
        self,
        conversation_id: str,
        session_id: str,
        source_end_store_id: int = 0,
        policy_fingerprint: str = "",
        route_fingerprint: str = "",
    ) -> int:
        """Get or create the initial generation=1 frontier. Returns generation."""
        with self._lock:
            existing = self.get_active_frontier(conversation_id)
            if existing is not None:
                return existing["generation"]
            now = time.time()
            self._conn.execute(
                """
                INSERT INTO lcm_active_frontiers
                    (conversation_id, generation, session_id,
                     source_end_store_id, policy_fingerprint,
                     route_fingerprint, created_at, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, session_id, source_end_store_id,
                 policy_fingerprint, route_fingerprint, now, now),
            )
            self._conn.commit()
            return 1

    def advance_frontier_generation(
        self,
        conversation_id: str,
        session_id: str,
        new_source_end: int,
        policy_fingerprint: str,
        route_fingerprint: str,
        base_generation: int,
    ) -> int:
        """CAS-promote: insert a new generation only if base_generation matches.

        Returns the new generation on success, or 0 on CAS failure.
        """
        with self._lock:
            current = self.get_active_frontier(conversation_id)
            current_gen = current["generation"] if current else 0
            if current_gen != base_generation:
                return 0
            now = time.time()
            new_gen = base_generation + 1
            self._conn.execute(
                """
                INSERT INTO lcm_active_frontiers
                    (conversation_id, generation, session_id,
                     source_end_store_id, policy_fingerprint,
                     route_fingerprint, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, new_gen, session_id,
                 new_source_end, policy_fingerprint,
                 route_fingerprint, now, now),
            )
            self._conn.commit()
            return new_gen

    # -- Frontier items ---------------------------------------------------

    def set_frontier_items(
        self,
        conversation_id: str,
        generation: int,
        items: list[dict[str, Any]],
    ) -> None:
        """Replace all items for a generation."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM lcm_frontier_items WHERE conversation_id = ? AND generation = ?",
                (conversation_id, generation),
            )
            for ordinal, item in enumerate(items):
                self._conn.execute(
                    """
                    INSERT INTO lcm_frontier_items
                        (conversation_id, generation, ordinal, kind,
                         ref_id, source_start, source_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (conversation_id, generation, ordinal,
                     item.get("kind", "message"),
                     item.get("ref_id", 0),
                     item.get("source_start", 0),
                     item.get("source_end", 0)),
                )
            self._conn.commit()

    def get_frontier_items(
        self,
        conversation_id: str,
        generation: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ordinal, kind, ref_id, source_start, source_end
                FROM lcm_frontier_items
                WHERE conversation_id = ? AND generation = ?
                ORDER BY ordinal
                """,
                (conversation_id, generation),
            ).fetchall()
        return [
            {"ordinal": r[0], "kind": r[1], "ref_id": r[2],
             "source_start": r[3], "source_end": r[4]}
            for r in rows
        ]

    # -- Prepared batches -------------------------------------------------

    def create_batch(
        self,
        conversation_id: str,
        session_id: str,
        base_generation: int,
        source_end_store_id: int,
        source_identity_hash: str,
        source_ids: list[int],
        policy_fingerprint: str,
        route_fingerprint: str,
        state: str = "preparing",
    ) -> int:
        """Insert a new prepared batch and return its batch_id."""
        with self._lock:
            now = time.time()
            cur = self._conn.execute(
                """
                INSERT INTO lcm_prepared_batches
                    (conversation_id, session_id, base_generation,
                     source_end_store_id, source_identity_hash, source_ids,
                     policy_fingerprint, route_fingerprint, state,
                     expected_leaf_count, frontier_end_store_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                """,
                (conversation_id, session_id, base_generation,
                 source_end_store_id, source_identity_hash,
                 json.dumps(source_ids),
                 policy_fingerprint, route_fingerprint, state,
                 now, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def update_batch_state(
        self,
        batch_id: int,
        state: str,
        *,
        expected_leaf_count: Optional[int] = None,
        frontier_end_store_id: Optional[int] = None,
        failure_reason: str = "",
    ) -> None:
        with self._lock:
            sets = ["state = ?", "updated_at = ?"]
            params: list[Any] = [state, time.time()]
            if expected_leaf_count is not None:
                sets.append("expected_leaf_count = ?")
                params.append(expected_leaf_count)
            if frontier_end_store_id is not None:
                sets.append("frontier_end_store_id = ?")
                params.append(frontier_end_store_id)
            if failure_reason:
                sets.append("failure_reason = ?")
                params.append(failure_reason)
            params.append(batch_id)
            self._conn.execute(
                f"UPDATE lcm_prepared_batches SET {', '.join(sets)} WHERE batch_id = ?",
                params,
            )
            self._conn.commit()

    def get_batch(self, batch_id: int) -> PreparedBatch | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT batch_id, conversation_id, session_id, base_generation,
                       source_end_store_id, source_identity_hash, source_ids,
                       policy_fingerprint, route_fingerprint, state,
                       expected_leaf_count, frontier_end_store_id, failure_reason
                FROM lcm_prepared_batches WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
        if not row:
            return None
        return PreparedBatch(
            batch_id=row[0],
            conversation_id=row[1],
            session_id=row[2],
            base_generation=row[3],
            source_end_store_id=row[4],
            source_identity_hash=row[5],
            source_ids=json.loads(row[6]) if row[6] else [],
            policy_fingerprint=row[7],
            route_fingerprint=row[8],
            state=row[9],
            expected_leaf_count=row[10],
            frontier_end_store_id=row[11],
            failure_reason=row[12] or "",
        )

    def get_batch_counts_by_state(self, conversation_id: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT state, COUNT(*) FROM lcm_prepared_batches
                WHERE conversation_id = ?
                GROUP BY state
                """,
                (conversation_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_ready_batch(self, conversation_id: str) -> PreparedBatch | None:
        """Return the most recent ready batch for a conversation, if any."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT batch_id, conversation_id, session_id, base_generation,
                       source_end_store_id, source_identity_hash, source_ids,
                       policy_fingerprint, route_fingerprint, state,
                       expected_leaf_count, frontier_end_store_id, failure_reason
                FROM lcm_prepared_batches
                WHERE conversation_id = ? AND state = 'ready'
                ORDER BY batch_id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return PreparedBatch(
            batch_id=row[0],
            conversation_id=row[1],
            session_id=row[2],
            base_generation=row[3],
            source_end_store_id=row[4],
            source_identity_hash=row[5],
            source_ids=json.loads(row[6]) if row[6] else [],
            policy_fingerprint=row[7],
            route_fingerprint=row[8],
            state=row[9],
            expected_leaf_count=row[10],
            frontier_end_store_id=row[11],
            failure_reason=row[12] or "",
        )

    def reap_stale_preparing(self, conversation_id: str) -> int:
        """Mark any 'preparing' batches as 'failed' (for restart recovery)."""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE lcm_prepared_batches
                SET state = 'failed', failure_reason = 'stale_preparing_on_restart',
                    updated_at = ?
                WHERE conversation_id = ? AND state = 'preparing'
                """,
                (time.time(), conversation_id),
            )
            self._conn.commit()
            return cur.rowcount

    def supersede_pending_batches(
        self,
        conversation_id: str,
        *,
        reason: str = "foreground_compaction",
    ) -> int:
        """Mark ready/preparing batches as superseded (foreground race won)."""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE lcm_prepared_batches
                SET state = 'superseded', failure_reason = ?, updated_at = ?
                WHERE conversation_id = ? AND state IN ('ready', 'preparing')
                """,
                (reason, time.time(), conversation_id),
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# -- Helpers ---------------------------------------------------------------

def compute_source_identity_hash(
    conn: sqlite3.Connection,
    session_id: str,
    source_ids: Sequence[int],
) -> str:
    """Compute a SHA-256 hash of the content of the given source message IDs."""
    h = hashlib.sha256()
    for sid in source_ids:
        row = conn.execute(
            "SELECT store_id, role, content, timestamp FROM messages WHERE store_id = ? AND session_id = ?",
            (sid, session_id),
        ).fetchone()
        if row:
            h.update(f"{row[0]}|{row[1]}|{row[2]}|{row[3]}".encode())
        else:
            h.update(f"{sid}|missing".encode())
    return h.hexdigest()[:32]


def compute_route_fingerprint(summary_model: str, fallback_models: tuple[str, ...]) -> str:
    """Stable fingerprint of the summary route (model + fallbacks)."""
    raw = json.dumps({"model": summary_model, "fallbacks": list(fallback_models)}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]