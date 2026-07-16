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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .db_bootstrap import (
    ensure_frontier_tables,
    ensure_prepared_batch_payload_columns,
    configure_connection,
    run_versioned_migrations,
)

logger = logging.getLogger(__name__)

# payload_version >= PREPARED_PAYLOAD_VERSION means prepare stored a full
# summary payload and promote must not re-run the summarizer.
PREPARED_PAYLOAD_VERSION = 2


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
    state: str  # preparing | ready | promoted | rejected | failed | superseded
    expected_leaf_count: int
    frontier_end_store_id: int
    failure_reason: str = ""
    summary_payload: str = ""
    payload_version: int = 0
    resolved_policy_json: str = "{}"

    def resolved_policy(self) -> dict[str, Any]:
        try:
            value = json.loads(self.resolved_policy_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @property
    def has_summary_payload(self) -> bool:
        """True when prepare stored a publishable summary (promote is zero-LLM)."""
        if int(self.payload_version or 0) < PREPARED_PAYLOAD_VERSION:
            return False
        if not (self.summary_payload or "").strip():
            return False
        try:
            data = json.loads(self.summary_payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return bool(isinstance(data, dict) and (data.get("summary_text") or "").strip())

    def parsed_summary_payload(self) -> dict[str, Any] | None:
        if not self.has_summary_payload:
            return None
        try:
            data = json.loads(self.summary_payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None


@dataclass
class PromotionResult:
    """Result of a ``promote_prepared_compaction`` call."""
    promoted: bool
    reason: str = ""
    batch_id: int = 0
    node_id: int = 0
    covered_source_ids: list[int] = field(default_factory=list)
    # Wall-clock telemetry (milliseconds). validation = CAS checks only;
    # publication = DAG insert + frontier advance + lifecycle; wall = total.
    validation_ms: float = 0.0
    publication_ms: float = 0.0
    wall_ms: float = 0.0


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

    @contextmanager
    def publication_transaction(self):
        """Yield the coordinator connection inside one immediate transaction.

        All LCM publication tables share this database.  Callers must use only
        no-commit primitives while inside the context; normal return commits
        the whole publication and every exception rolls it all back.
        """
        with self._lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

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

    def advance_frontier_generation_with_items(
        self,
        conversation_id: str,
        session_id: str,
        new_source_end: int,
        policy_fingerprint: str,
        route_fingerprint: str,
        base_generation: int,
        items: list[dict[str, Any]],
    ) -> int:
        """CAS-advance a generation and publish its items atomically.

        Both tables live on this FrontierStore connection, so one SQLite write
        transaction can guarantee that an active generation with a positive
        source boundary is never committed without its ordered items.
        Returns the new generation, or 0 on CAS mismatch.
        """
        if new_source_end > 0 and not items:
            raise ValueError("frontier items required for positive source boundary")
        with self._lock:
            conn = self.conn
            owns_transaction = not conn.in_transaction
            try:
                if owns_transaction:
                    conn.execute("BEGIN IMMEDIATE")
                new_gen = self.advance_frontier_generation_with_items_no_commit(
                    conn,
                    conversation_id,
                    session_id,
                    new_source_end,
                    policy_fingerprint,
                    route_fingerprint,
                    base_generation,
                    items,
                )
                if new_gen == 0:
                    if owns_transaction:
                        conn.rollback()
                    return 0
                if owns_transaction:
                    conn.commit()
                return new_gen
            except Exception:
                if owns_transaction:
                    conn.rollback()
                raise

    def advance_frontier_generation_with_items_no_commit(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
        new_source_end: int,
        policy_fingerprint: str,
        route_fingerprint: str,
        base_generation: int,
        items: list[dict[str, Any]],
    ) -> int:
        """CAS-publish a generation and items on a caller-owned transaction."""
        if new_source_end > 0 and not items:
            raise ValueError("frontier items required for positive source boundary")
        row = conn.execute(
            """
            SELECT generation
            FROM lcm_active_frontiers
            WHERE conversation_id = ?
            ORDER BY generation DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        current_gen = int(row[0]) if row else 0
        if current_gen != int(base_generation):
            return 0

        now = time.time()
        new_gen = int(base_generation) + 1
        conn.execute(
            """
            INSERT INTO lcm_active_frontiers
                (conversation_id, generation, session_id,
                 source_end_store_id, policy_fingerprint,
                 route_fingerprint, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                new_gen,
                session_id,
                new_source_end,
                policy_fingerprint,
                route_fingerprint,
                now,
                now,
            ),
        )
        phase_hook = getattr(self, "_publication_phase_hook", None)
        if callable(phase_hook):
            phase_hook("after_frontier_generation")
        for ordinal, item in enumerate(items):
            conn.execute(
                """
                INSERT INTO lcm_frontier_items
                    (conversation_id, generation, ordinal, kind,
                     ref_id, source_start, source_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    new_gen,
                    ordinal,
                    item.get("kind", "message"),
                    item.get("ref_id", 0),
                    item.get("source_start", 0),
                    item.get("source_end", 0),
                ),
            )
        if callable(phase_hook):
            phase_hook("after_frontier_items")
        return new_gen

    def publish_generation_state_no_commit(
        self,
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        source_end_store_id: int,
        policy_fingerprint: str,
        route_fingerprint: str,
        base_generation: int,
        items: list[dict[str, Any]],
        batch_reason: str,
        winner_batch_id: int | None = None,
        phase_hook=None,
    ) -> int:
        """Atomically publish the complete canonical winner state.

        The caller owns ``BEGIN IMMEDIATE`` and performs any DAG/message writes
        first on ``conn``.  This shared finalizer advances the ordered frontier,
        settles every losing batch for the replaced generation, and aligns the
        lifecycle marker before the transaction may commit.
        """
        new_generation = self.advance_frontier_generation_with_items_no_commit(
            conn,
            conversation_id,
            session_id,
            int(source_end_store_id),
            policy_fingerprint,
            route_fingerprint,
            int(base_generation),
            items,
        )
        if not new_generation:
            return 0
        finalize_generation_winner_no_commit(
            conn,
            conversation_id=conversation_id,
            session_id=session_id,
            source_end_store_id=source_end_store_id,
            base_generation=base_generation,
            batch_reason=batch_reason,
            winner_batch_id=winner_batch_id,
            phase_hook=phase_hook,
        )
        return int(new_generation)

    def rollback_frontier_generation(self, conversation_id: str, generation: int) -> bool:
        """Remove the just-published generation when a later publish step fails.

        The rollback is deliberately conditional: it only removes ``generation``
        when it is still the active tip, so a concurrent promotion can never be
        erased by an older failing caller. The tip check and both deletions run
        inside one ``BEGIN IMMEDIATE`` transaction; acquiring the write lock
        before reading prevents a newer generation from committing between the
        check and deletion.
        """
        with self._lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """
                    SELECT generation
                    FROM lcm_active_frontiers
                    WHERE conversation_id = ?
                    ORDER BY generation DESC
                    LIMIT 1
                    """,
                    (conversation_id,),
                ).fetchone()
                current_generation = int(row[0]) if row else 0
                if current_generation != int(generation):
                    conn.rollback()
                    return False
                conn.execute(
                    "DELETE FROM lcm_frontier_items WHERE conversation_id = ? AND generation = ?",
                    (conversation_id, generation),
                )
                conn.execute(
                    "DELETE FROM lcm_active_frontiers WHERE conversation_id = ? AND generation = ?",
                    (conversation_id, generation),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

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
        resolved_policy_json: str = "{}",
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
                     created_at, updated_at, resolved_policy_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                """,
                (conversation_id, session_id, base_generation,
                 source_end_store_id, source_identity_hash,
                 json.dumps(source_ids),
                 policy_fingerprint, route_fingerprint, state,
                 now, now, resolved_policy_json),
            )
            self._conn.commit()
            return cur.lastrowid

    def create_batch_bounded(
        self,
        *,
        conversation_id: str,
        session_id: str,
        base_generation: int,
        source_end_store_id: int,
        source_identity_hash: str,
        source_ids: list[int],
        policy_fingerprint: str,
        route_fingerprint: str,
        max_conversation_candidates: int,
        max_profile_candidates: int,
        resolved_policy_json: str = "{}",
    ) -> tuple[int, str]:
        """Atomically enforce speculative candidate caps and create a batch."""
        with self._lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                duplicate = conn.execute(
                    """
                    SELECT 1 FROM lcm_prepared_batches
                    WHERE conversation_id = ?
                      AND state IN ('preparing', 'ready')
                      AND base_generation = ?
                      AND policy_fingerprint = ?
                      AND route_fingerprint = ?
                      AND source_end_store_id >= ?
                    LIMIT 1
                    """,
                    (
                        conversation_id,
                        int(base_generation),
                        policy_fingerprint,
                        route_fingerprint,
                        int(source_end_store_id),
                    ),
                ).fetchone()
                if duplicate:
                    conn.rollback()
                    return 0, "candidate-already-covers-range"
                conversation_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM lcm_prepared_batches
                        WHERE conversation_id = ?
                          AND state IN ('preparing', 'ready')
                        """,
                        (conversation_id,),
                    ).fetchone()[0]
                )
                if conversation_count >= max(1, int(max_conversation_candidates)):
                    conn.rollback()
                    return 0, "conversation-candidate-limit"
                profile_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM lcm_prepared_batches
                        WHERE state IN ('preparing', 'ready')
                        """
                    ).fetchone()[0]
                )
                if profile_count >= max(1, int(max_profile_candidates)):
                    conn.rollback()
                    return 0, "profile-candidate-limit"
                now = time.time()
                cur = conn.execute(
                    """
                    INSERT INTO lcm_prepared_batches
                        (conversation_id, session_id, base_generation,
                         source_end_store_id, source_identity_hash, source_ids,
                         policy_fingerprint, route_fingerprint, state,
                         expected_leaf_count, frontier_end_store_id,
                         created_at, updated_at, resolved_policy_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'preparing', 0, 0, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        session_id,
                        int(base_generation),
                        int(source_end_store_id),
                        source_identity_hash,
                        json.dumps(source_ids),
                        policy_fingerprint,
                        route_fingerprint,
                        now,
                        now,
                        resolved_policy_json,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid), ""
            except Exception:
                conn.rollback()
                raise

    def create_batch_cas(
        self,
        *,
        conversation_id: str,
        session_id: str,
        base_generation: int,
        source_end_store_id: int,
        source_identity_hash: str,
        source_ids: list[int],
        policy_fingerprint: str,
        route_fingerprint: str,
        resolved_policy_json: str = "{}",
        max_conversation_candidates: int | None = None,
        max_profile_candidates: int | None = None,
    ) -> tuple[int, str]:
        """Create a preparing row only while ``base_generation`` is still active.

        The generation check and insert share SQLite's writer lock.  A publisher
        that wins before this transaction therefore makes the speculative work
        stale without allowing a post-winner row to appear.
        """
        with self.publication_transaction() as conn:
            row = conn.execute(
                """
                SELECT generation FROM lcm_active_frontiers
                WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            current_generation = int(row[0]) if row else 0
            if current_generation != int(base_generation):
                return 0, "generation-superseded"

            if max_conversation_candidates is not None:
                duplicate = conn.execute(
                    """
                    SELECT 1 FROM lcm_prepared_batches
                    WHERE conversation_id = ?
                      AND state IN ('preparing', 'ready')
                      AND base_generation = ?
                      AND policy_fingerprint = ?
                      AND route_fingerprint = ?
                      AND source_end_store_id >= ?
                    LIMIT 1
                    """,
                    (
                        conversation_id,
                        int(base_generation),
                        policy_fingerprint,
                        route_fingerprint,
                        int(source_end_store_id),
                    ),
                ).fetchone()
                if duplicate:
                    return 0, "candidate-already-covers-range"
                conversation_count = int(conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_batches
                    WHERE conversation_id = ? AND state IN ('preparing', 'ready')
                    """,
                    (conversation_id,),
                ).fetchone()[0])
                if conversation_count >= max(1, int(max_conversation_candidates)):
                    return 0, "conversation-candidate-limit"
                profile_count = int(conn.execute(
                    """SELECT COUNT(*) FROM lcm_prepared_batches
                       WHERE state IN ('preparing', 'ready')"""
                ).fetchone()[0])
                if profile_count >= max(1, int(max_profile_candidates or 1)):
                    return 0, "profile-candidate-limit"

            now = time.time()
            cur = conn.execute(
                """
                INSERT INTO lcm_prepared_batches
                    (conversation_id, session_id, base_generation,
                     source_end_store_id, source_identity_hash, source_ids,
                     policy_fingerprint, route_fingerprint, state,
                     expected_leaf_count, frontier_end_store_id,
                     created_at, updated_at, resolved_policy_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'preparing', 0, 0, ?, ?, ?)
                """,
                (
                    conversation_id,
                    session_id,
                    int(base_generation),
                    int(source_end_store_id),
                    source_identity_hash,
                    json.dumps([int(source_id) for source_id in source_ids]),
                    policy_fingerprint,
                    route_fingerprint,
                    now,
                    now,
                    resolved_policy_json,
                ),
            )
            return int(cur.lastrowid), ""

    def finalize_batch_cas(
        self,
        batch_id: int,
        *,
        base_generation: int,
        state: str,
        expected_leaf_count: int,
        frontier_end_store_id: int,
        summary_payload: str,
        payload_version: int,
        source_end_store_id: int,
        source_identity_hash: str,
        source_ids: Sequence[int],
    ) -> tuple[bool, str]:
        """Finalize preparation only if its row and base generation are current."""
        with self.publication_transaction() as conn:
            batch_row = conn.execute(
                """SELECT conversation_id, base_generation, state
                   FROM lcm_prepared_batches WHERE batch_id = ?""",
                (int(batch_id),),
            ).fetchone()
            if batch_row is None:
                return False, "batch-not-found"
            current_state = str(batch_row[2] or "")
            if current_state != "preparing":
                return False, f"batch-state-{current_state}"
            row = conn.execute(
                """
                SELECT generation FROM lcm_active_frontiers
                WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1
                """,
                (str(batch_row[0]),),
            ).fetchone()
            current_generation = int(row[0]) if row else 0
            if (
                int(batch_row[1]) != int(base_generation)
                or current_generation != int(base_generation)
            ):
                self.update_batch_state_no_commit(
                    conn,
                    int(batch_id),
                    "superseded",
                    failure_reason="generation-superseded-during-preparation",
                )
                return False, "generation-superseded"
            self.update_batch_state_no_commit(
                conn,
                int(batch_id),
                state,
                expected_leaf_count=expected_leaf_count,
                frontier_end_store_id=frontier_end_store_id,
                summary_payload=summary_payload,
                payload_version=payload_version,
                source_end_store_id=source_end_store_id,
                source_identity_hash=source_identity_hash,
                source_ids=source_ids,
            )
            return True, ""

    def settle_batch_if_preparing(
        self,
        batch_id: int,
        state: str,
        *,
        failure_reason: str,
    ) -> bool:
        """Settle an owned preparing row without reviving a winner-settled row."""
        with self.publication_transaction() as conn:
            cur = conn.execute(
                """
                UPDATE lcm_prepared_batches
                SET state = ?, failure_reason = ?, updated_at = ?
                WHERE batch_id = ? AND state = 'preparing'
                """,
                (state, failure_reason, time.time(), int(batch_id)),
            )
            return int(cur.rowcount or 0) > 0

    def list_pending_batches(self, conversation_id: str) -> list[PreparedBatch]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT batch_id, conversation_id, session_id, base_generation,
                       source_end_store_id, source_identity_hash, source_ids,
                       policy_fingerprint, route_fingerprint, state,
                       expected_leaf_count, frontier_end_store_id, failure_reason,
                       COALESCE(summary_payload, ''), COALESCE(payload_version, 0),
                       COALESCE(resolved_policy_json, '{}')
                FROM lcm_prepared_batches
                WHERE conversation_id = ? AND state IN ('preparing', 'ready')
                ORDER BY source_end_store_id DESC, batch_id DESC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._row_to_batch(row) for row in rows]

    def cleanup_pending_batches(
        self,
        *,
        conversation_id: str,
        current_generation: int,
        policy_fingerprint: str,
        route_fingerprint: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> dict[str, int]:
        """Supersede expired or incompatible speculative work only."""
        current_time = time.time() if now is None else float(now)
        counts = {
            "ttl-expired": 0,
            "generation-superseded": 0,
            "policy-superseded": 0,
        }
        with self._lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                if ttl_seconds > 0:
                    cur = conn.execute(
                        """
                        UPDATE lcm_prepared_batches
                        SET state = 'superseded', failure_reason = 'ttl-expired', updated_at = ?
                        WHERE state IN ('preparing', 'ready') AND updated_at < ?
                        """,
                        (current_time, current_time - float(ttl_seconds)),
                    )
                    counts["ttl-expired"] = max(0, int(cur.rowcount))
                cur = conn.execute(
                    """
                    UPDATE lcm_prepared_batches
                    SET state = 'superseded', failure_reason = 'generation-superseded', updated_at = ?
                    WHERE conversation_id = ? AND state IN ('preparing', 'ready')
                      AND base_generation != ?
                    """,
                    (current_time, conversation_id, int(current_generation)),
                )
                counts["generation-superseded"] = max(0, int(cur.rowcount))
                cur = conn.execute(
                    """
                    UPDATE lcm_prepared_batches
                    SET state = 'superseded', failure_reason = 'policy-superseded', updated_at = ?
                    WHERE conversation_id = ? AND state IN ('preparing', 'ready')
                      AND (policy_fingerprint != ? OR route_fingerprint != ?)
                    """,
                    (
                        current_time,
                        conversation_id,
                        policy_fingerprint,
                        route_fingerprint,
                    ),
                )
                counts["policy-superseded"] = max(0, int(cur.rowcount))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return counts

    def update_batch_state(
        self,
        batch_id: int,
        state: str,
        *,
        expected_leaf_count: Optional[int] = None,
        frontier_end_store_id: Optional[int] = None,
        failure_reason: str = "",
        summary_payload: Optional[str] = None,
        payload_version: Optional[int] = None,
        source_end_store_id: Optional[int] = None,
        source_identity_hash: Optional[str] = None,
        source_ids: Optional[Sequence[int]] = None,
    ) -> None:
        with self._lock:
            owns_transaction = not self._conn.in_transaction
            self.update_batch_state_no_commit(
                self._conn,
                batch_id,
                state,
                expected_leaf_count=expected_leaf_count,
                frontier_end_store_id=frontier_end_store_id,
                failure_reason=failure_reason,
                summary_payload=summary_payload,
                payload_version=payload_version,
                source_end_store_id=source_end_store_id,
                source_identity_hash=source_identity_hash,
                source_ids=source_ids,
            )
            if owns_transaction:
                self._conn.commit()

    @staticmethod
    def update_batch_state_no_commit(
        conn: sqlite3.Connection,
        batch_id: int,
        state: str,
        *,
        expected_leaf_count: Optional[int] = None,
        frontier_end_store_id: Optional[int] = None,
        failure_reason: str = "",
        summary_payload: Optional[str] = None,
        payload_version: Optional[int] = None,
        source_end_store_id: Optional[int] = None,
        source_identity_hash: Optional[str] = None,
        source_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Update a prepared batch on a caller-owned transaction."""
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
        if summary_payload is not None:
            sets.append("summary_payload = ?")
            params.append(summary_payload)
        if payload_version is not None:
            sets.append("payload_version = ?")
            params.append(int(payload_version))
        if source_end_store_id is not None:
            sets.append("source_end_store_id = ?")
            params.append(int(source_end_store_id))
        if source_identity_hash is not None:
            sets.append("source_identity_hash = ?")
            params.append(str(source_identity_hash))
        if source_ids is not None:
            sets.append("source_ids = ?")
            params.append(json.dumps([int(source_id) for source_id in source_ids]))
        params.append(batch_id)
        conn.execute(
            f"UPDATE lcm_prepared_batches SET {', '.join(sets)} WHERE batch_id = ?",
            params,
        )

    def _row_to_batch(self, row: Sequence[Any]) -> PreparedBatch:
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
            summary_payload=(row[13] if len(row) > 13 else "") or "",
            payload_version=int(row[14]) if len(row) > 14 else 0,
            resolved_policy_json=(row[15] if len(row) > 15 else "{}") or "{}",
        )

    def get_batch(self, batch_id: int) -> PreparedBatch | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT batch_id, conversation_id, session_id, base_generation,
                       source_end_store_id, source_identity_hash, source_ids,
                       policy_fingerprint, route_fingerprint, state,
                       expected_leaf_count, frontier_end_store_id, failure_reason,
                       COALESCE(summary_payload, ''), COALESCE(payload_version, 0),
                       COALESCE(resolved_policy_json, '{}')
                FROM lcm_prepared_batches WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_batch(row)

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
        """Return the most recent ready batch for a conversation, if any.

        Prefers payload-bearing (v2+) batches. Legacy v1 ready rows without a
        summary payload are superseded in place so promote never re-summarizes.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT batch_id, conversation_id, session_id, base_generation,
                       source_end_store_id, source_identity_hash, source_ids,
                       policy_fingerprint, route_fingerprint, state,
                       expected_leaf_count, frontier_end_store_id, failure_reason,
                       COALESCE(summary_payload, ''), COALESCE(payload_version, 0),
                       COALESCE(resolved_policy_json, '{}')
                FROM lcm_prepared_batches
                WHERE conversation_id = ? AND state = 'ready'
                ORDER BY batch_id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        batch = self._row_to_batch(row)
        if not batch.has_summary_payload:
            # Reject silently at the ready-lookup boundary so callers fall back
            # to foreground compaction instead of re-running the LLM.
            self.update_batch_state(
                batch.batch_id,
                "superseded",
                failure_reason="legacy_v1_batch_without_payload",
            )
            return None
        return batch

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

    @staticmethod
    def supersede_competing_batches_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        base_generation: int,
        *,
        winner_batch_id: int | None = None,
        reason: str = "canonical_generation_published",
    ) -> int:
        """Settle every losing candidate tied to a published base generation."""
        params: list[Any] = [reason, time.time(), conversation_id, int(base_generation)]
        winner_clause = ""
        if winner_batch_id is not None:
            winner_clause = " AND batch_id != ?"
            params.append(int(winner_batch_id))
        cur = conn.execute(
            f"""
            UPDATE lcm_prepared_batches
            SET state = 'superseded', failure_reason = ?, updated_at = ?
            WHERE conversation_id = ? AND base_generation = ?
              AND state IN ('preparing', 'ready'){winner_clause}
            """,
            params,
        )
        return max(0, int(cur.rowcount or 0))

    def list_itemless_active_generations(
        self,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return active frontier generations that have no frontier items.

        Used by promotion compensation reconciliation and doctor-style repair.
        """
        with self._lock:
            if conversation_id:
                rows = self._conn.execute(
                    """
                    SELECT f.conversation_id, f.generation, f.session_id,
                           f.source_end_store_id
                    FROM lcm_active_frontiers f
                    WHERE f.conversation_id = ?
                      AND f.source_end_store_id > 0
                      AND NOT EXISTS (
                          SELECT 1 FROM lcm_frontier_items i
                          WHERE i.conversation_id = f.conversation_id
                            AND i.generation = f.generation
                      )
                    ORDER BY f.generation
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT f.conversation_id, f.generation, f.session_id,
                           f.source_end_store_id
                    FROM lcm_active_frontiers f
                    WHERE f.source_end_store_id > 0
                      AND NOT EXISTS (
                          SELECT 1 FROM lcm_frontier_items i
                          WHERE i.conversation_id = f.conversation_id
                            AND i.generation = f.generation
                      )
                    ORDER BY f.conversation_id, f.generation
                    """
                ).fetchall()
        return [
            {
                "conversation_id": r[0],
                "generation": r[1],
                "session_id": r[2],
                "source_end_store_id": r[3],
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass


# -- Helpers ---------------------------------------------------------------

def finalize_generation_winner_no_commit(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    session_id: str,
    source_end_store_id: int,
    base_generation: int,
    batch_reason: str,
    winner_batch_id: int | None = None,
    phase_hook=None,
) -> None:
    """Settle batch/lifecycle winner state on a caller-owned transaction."""
    if callable(phase_hook):
        phase_hook("after_frontier")
    FrontierStore.supersede_competing_batches_no_commit(
        conn,
        conversation_id,
        int(base_generation),
        winner_batch_id=winner_batch_id,
        reason=batch_reason,
    )
    if callable(phase_hook):
        phase_hook("after_batches_superseded")
    now = time.time()
    conn.execute(
        """
        INSERT INTO lcm_lifecycle_state(
            conversation_id, current_session_id,
            current_frontier_store_id, current_bound_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            current_session_id = excluded.current_session_id,
            current_frontier_store_id = MAX(
                lcm_lifecycle_state.current_frontier_store_id,
                excluded.current_frontier_store_id
            ),
            current_bound_at = COALESCE(
                lcm_lifecycle_state.current_bound_at,
                excluded.current_bound_at
            ),
            updated_at = excluded.updated_at
        """,
        (
            conversation_id,
            session_id,
            int(source_end_store_id),
            now,
            now,
        ),
    )
    if callable(phase_hook):
        phase_hook("after_lifecycle")


def compute_source_identity_hash(
    conn: sqlite3.Connection,
    session_id: str,
    source_ids: Sequence[int],
) -> str:
    """Hash every durable field that can change summarizer semantics."""
    h = hashlib.sha256()
    normalized_ids = [int(source_id) for source_id in source_ids]
    if not normalized_ids:
        return h.hexdigest()[:32]
    placeholders = ",".join("?" for _ in normalized_ids)
    rows = conn.execute(
        f"""SELECT store_id, session_id, role, content, tool_call_id, tool_calls
            FROM messages WHERE session_id = ? AND store_id IN ({placeholders})""",
        (session_id, *normalized_ids),
    ).fetchall()
    rows_by_id = {int(row[0]): row for row in rows}
    for sid in normalized_ids:
        row = rows_by_id.get(sid)
        if row:
            tool_calls = row[5]
            if tool_calls:
                try:
                    tool_calls = json.dumps(
                        json.loads(tool_calls),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    tool_calls = str(tool_calls)
            else:
                tool_calls = ""
            h.update(json.dumps(
                [
                    int(row[0]),
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[3] or ""),
                    str(row[4] or ""),
                    tool_calls,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
        else:
            h.update(f"{sid}|missing".encode())
    return h.hexdigest()[:32]


def compute_route_fingerprint(summary_model: str, fallback_models: tuple[str, ...]) -> str:
    """Stable fingerprint of the summary route (model + fallbacks)."""
    raw = json.dumps({"model": summary_model, "fallbacks": list(fallback_models)}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
