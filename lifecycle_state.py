"""Durable lifecycle/checkpoint state for hermes-lcm.

This is the smallest viable substrate for cross-turn/session lifecycle state:
- which logical conversation a session belongs to
- which session is currently bound
- which session was last finalized
- the active session frontier/checkpoint marker
- the last finalized frontier marker
"""

from __future__ import annotations

import functools
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .db_bootstrap import configure_connection, refuse_schema_version_too_new, run_versioned_migrations


def _synchronized(method):
    """Serialize a read-modify-write method on the store's reentrant lock.

    The lifecycle connection is shared across threads (check_same_thread=False,
    autocommit). Without serialization, two callers reading state and then
    writing can interleave and clobber each other's update.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class LifecycleState:
    conversation_id: str
    current_session_id: str | None
    last_finalized_session_id: str | None
    current_frontier_store_id: int
    last_finalized_frontier_store_id: int
    debt_kind: str | None
    debt_size_estimate: int
    current_bound_at: float | None
    last_finalized_at: float | None
    debt_updated_at: float | None
    last_maintenance_attempt_at: float | None
    last_rollover_at: float | None
    last_reset_at: float | None
    rollover_carry_over_context: bool | None
    binding_generation: int
    updated_at: float


class LifecycleStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # The connection is opened check_same_thread=False in autocommit mode
        # and is shared across the gateway thread, dispatcher, and sub-agents.
        # Serialize read-modify-write flows so concurrent binds/frontier
        # advances cannot interleave and regress the checkpoint.
        self._lock = threading.RLock()
        self._transaction_local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._conn.commit()

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live SQLite connection, or ``None`` once :meth:`close` has run.

        Exposed for read-oriented diagnostics -- for example the doctor's
        maintenance-debt scan -- that need ad-hoc queries the store does not wrap
        in a purpose-built method. Callers must treat it as read-only; writes go
        through the store's own methods.
        """
        return getattr(self, "_conn", None)

    def row_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM lcm_lifecycle_state").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_state(self, row: sqlite3.Row | None) -> LifecycleState | None:
        if row is None:
            return None
        return LifecycleState(
            conversation_id=row["conversation_id"],
            current_session_id=row["current_session_id"],
            last_finalized_session_id=row["last_finalized_session_id"],
            current_frontier_store_id=int(row["current_frontier_store_id"] or 0),
            last_finalized_frontier_store_id=int(row["last_finalized_frontier_store_id"] or 0),
            debt_kind=row["debt_kind"],
            debt_size_estimate=int(row["debt_size_estimate"] or 0),
            current_bound_at=row["current_bound_at"],
            last_finalized_at=row["last_finalized_at"],
            debt_updated_at=row["debt_updated_at"],
            last_maintenance_attempt_at=row["last_maintenance_attempt_at"],
            last_rollover_at=row["last_rollover_at"],
            last_reset_at=row["last_reset_at"],
            rollover_carry_over_context=(
                None
                if row["rollover_carry_over_context"] is None
                else bool(row["rollover_carry_over_context"])
            ),
            binding_generation=int(row["binding_generation"] or 0),
            updated_at=float(row["updated_at"] or 0.0),
        )

    @contextmanager
    def publication_connection(self, conn: sqlite3.Connection):
        """Route lifecycle writes to a caller-owned publication transaction."""
        previous = getattr(self._transaction_local, "connection", None)
        self._transaction_local.connection = conn
        try:
            yield
        finally:
            self._transaction_local.connection = previous

    def get_by_conversation(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return self._row_to_state(row)

    def get_by_session(self, session_id: str | None) -> LifecycleState | None:
        if not session_id:
            return None
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_lifecycle_state
            WHERE current_session_id = ? OR last_finalized_session_id = ?
            ORDER BY CASE WHEN current_session_id = ? THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (session_id, session_id, session_id),
        ).fetchone()
        return self._row_to_state(row)

    @_synchronized
    def bind_session(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> LifecycleState:
        # Capture the canonical owner/generation before reading lifecycle. The
        # conditional write below requires this exact snapshot still to be the
        # active tip, so a rollover that wins after this read makes the bind a
        # harmless no-op instead of letting stale startup state steal ownership.
        observed_generation = 0
        observed_frontier_session = ""
        if conversation_id:
            frontier = self._conn.execute(
                """SELECT generation, session_id FROM lcm_active_frontiers
                   WHERE conversation_id = ?
                   ORDER BY generation DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if frontier is not None:
                observed_generation = int(frontier[0] or 0)
                observed_frontier_session = str(frontier[1] or "")
        existing = self.get_by_conversation(conversation_id) if conversation_id else self.get_by_session(session_id)
        conversation_id = conversation_id or (existing.conversation_id if existing else session_id)
        if observed_generation == 0:
            frontier = self._conn.execute(
                """SELECT generation, session_id FROM lcm_active_frontiers
                   WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1""",
                (conversation_id,),
            ).fetchone()
            if frontier is not None:
                observed_generation = int(frontier[0] or 0)
                observed_frontier_session = str(frontier[1] or "")
        head = self._conn.execute(
            """SELECT current_session_id, last_finalized_session_id,
                      carry_over_context, rollover_epoch, frontier_generation
               FROM lcm_rollover_heads WHERE conversation_id = ?""",
            (conversation_id,),
        ).fetchone()
        head_consistent = bool(
            head is not None
            and observed_frontier_session == str(head[0] or "")
            and observed_generation >= int(head[4] or 0)
        )
        if head_consistent:
            # The head is authoritative only while it agrees with the latest
            # active frontier. Reconstruct or reconcile lifecycle from it;
            # stale bind/rewrite calls cannot clear or replace the head.
            now = time.time()
            protected = self._conn.execute(
                """SELECT finalized_boundary_store_id
                   FROM lcm_protected_sessions
                   WHERE conversation_id = ? AND finalized_session_id = ?""",
                (conversation_id, str(head[1])),
            ).fetchone()
            finalized_boundary = int(protected[0] or 0) if protected else (
                existing.last_finalized_frontier_store_id if existing else 0
            )
            self._conn.execute(
                """INSERT INTO lcm_lifecycle_state(
                       conversation_id, current_session_id,
                       last_finalized_session_id,
                       current_frontier_store_id,
                       last_finalized_frontier_store_id,
                       current_bound_at, last_finalized_at,
                       last_rollover_at, rollover_carry_over_context,
                       binding_generation,
                       updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(conversation_id) DO UPDATE SET
                       current_session_id = excluded.current_session_id,
                       last_finalized_session_id = excluded.last_finalized_session_id,
                       current_frontier_store_id = excluded.current_frontier_store_id,
                       last_finalized_frontier_store_id = MAX(
                           lcm_lifecycle_state.last_finalized_frontier_store_id,
                           excluded.last_finalized_frontier_store_id
                       ),
                       current_bound_at = excluded.current_bound_at,
                       last_finalized_at = excluded.last_finalized_at,
                       last_rollover_at = excluded.last_rollover_at,
                       rollover_carry_over_context = excluded.rollover_carry_over_context,
                       binding_generation = lcm_lifecycle_state.binding_generation + 1,
                       updated_at = excluded.updated_at
                   WHERE lcm_lifecycle_state.current_session_id IS NOT excluded.current_session_id
                      OR lcm_lifecycle_state.last_finalized_session_id IS NOT excluded.last_finalized_session_id
                      OR lcm_lifecycle_state.rollover_carry_over_context IS NOT excluded.rollover_carry_over_context""",
                (
                    conversation_id, str(head[0]), str(head[1]),
                    int(self._conn.execute(
                        """SELECT source_end_store_id FROM lcm_active_frontiers
                           WHERE conversation_id = ? AND generation = ?""",
                        (conversation_id, observed_generation),
                    ).fetchone()[0] or 0),
                    finalized_boundary, now, now, now, int(head[2]), now,
                ),
            )
            self._conn.commit()
            restored = self.get_by_conversation(conversation_id)
            assert restored is not None
            return restored
        now = time.time()
        current_frontier = 0
        current_bound_at = now
        last_finalized_session_id = None
        last_finalized_frontier = 0
        debt_kind = None
        debt_size_estimate = 0
        last_finalized_at = None
        debt_updated_at = None
        last_maintenance_attempt_at = None
        last_rollover_at = None
        last_reset_at = None
        rollover_carry_over_context = None

        if existing is not None:
            if existing.current_session_id == session_id:
                return existing
            if self._conn.execute(
                """SELECT 1 FROM lcm_protected_sessions
                   WHERE conversation_id = ? AND finalized_session_id = ?""",
                (conversation_id, session_id),
            ).fetchone():
                return existing
            current_frontier = (
                existing.current_frontier_store_id if existing.current_session_id == session_id else 0
            )
            current_bound_at = (
                existing.current_bound_at if existing.current_session_id == session_id else now
            )
            last_finalized_session_id = existing.last_finalized_session_id
            last_finalized_frontier = existing.last_finalized_frontier_store_id
            debt_kind = existing.debt_kind
            debt_size_estimate = existing.debt_size_estimate
            last_finalized_at = existing.last_finalized_at
            debt_updated_at = existing.debt_updated_at
            last_maintenance_attempt_at = existing.last_maintenance_attempt_at
            last_rollover_at = (
                now
                if (
                    (existing.current_session_id and existing.current_session_id != session_id)
                    or (
                        existing.current_session_id is None
                        and existing.last_finalized_session_id
                        and existing.last_finalized_session_id != session_id
                    )
                )
                else existing.last_rollover_at
            )
            last_reset_at = existing.last_reset_at

        if existing is None:
            self._conn.execute(
                """INSERT INTO lcm_lifecycle_state(
                       conversation_id, current_session_id,
                       last_finalized_session_id, current_frontier_store_id,
                       last_finalized_frontier_store_id, debt_kind,
                       debt_size_estimate, current_bound_at, last_finalized_at,
                       debt_updated_at, last_maintenance_attempt_at,
                       last_rollover_at, last_reset_at,
                       rollover_carry_over_context, binding_generation, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(conversation_id) DO NOTHING""",
                (
                    conversation_id, session_id, last_finalized_session_id,
                    current_frontier, last_finalized_frontier, debt_kind,
                    debt_size_estimate, current_bound_at, last_finalized_at,
                    debt_updated_at, last_maintenance_attempt_at,
                    last_rollover_at, last_reset_at,
                    rollover_carry_over_context, now,
                ),
            )
        else:
            # Preserve finalized metadata and the winning carry policy. Only
            # ownership fields change, and only while both observed owner and
            # active generation still match.
            self._conn.execute(
                """UPDATE lcm_lifecycle_state
                   SET current_session_id = ?, current_frontier_store_id = 0,
                       current_bound_at = ?, last_rollover_at = ?,
                       binding_generation = binding_generation + 1,
                       updated_at = ?
                   WHERE conversation_id = ?
                     AND current_session_id IS ?
                     AND binding_generation = ?
                     AND (
                         (? = 0 AND NOT EXISTS (
                             SELECT 1 FROM lcm_active_frontiers
                             WHERE conversation_id = ?
                         ))
                         OR EXISTS (
                             SELECT 1 FROM lcm_active_frontiers AS f
                             WHERE f.conversation_id = ?
                               AND f.generation = ? AND f.session_id = ?
                               AND NOT EXISTS (
                                   SELECT 1 FROM lcm_active_frontiers AS newer
                                   WHERE newer.conversation_id = f.conversation_id
                                     AND newer.generation > f.generation
                               )
                         )
                     )""",
                (
                    session_id, now, now, now, conversation_id,
                    existing.current_session_id, existing.binding_generation,
                    observed_generation,
                    conversation_id, conversation_id, observed_generation,
                    observed_frontier_session,
                ),
            )
        self._conn.commit()
        state = self.get_by_conversation(conversation_id)
        assert state is not None
        return state

    @_synchronized
    def finalize_session(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int = 0,
    ) -> LifecycleState | None:
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        current_session_id = state.current_session_id
        current_frontier = state.current_frontier_store_id
        if current_session_id == session_id:
            current_session_id = None
            current_frontier = 0
        finalized_frontier = max(
            int(frontier_store_id or 0),
            state.last_finalized_frontier_store_id,
        )
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_session_id = ?,
                last_finalized_session_id = ?,
                current_frontier_store_id = ?,
                last_finalized_frontier_store_id = ?,
                rollover_carry_over_context = NULL,
                binding_generation = binding_generation + 1,
                debt_kind = debt_kind,
                debt_size_estimate = debt_size_estimate,
                last_finalized_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                current_session_id,
                session_id,
                current_frontier,
                finalized_frontier,
                now,
                now,
                state.conversation_id,
            ),
        )
        self._conn.commit()
        return self.get_by_conversation(state.conversation_id)

    @_synchronized
    def record_rollover(
        self,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
        carry_over_context: bool = True,
    ) -> LifecycleState:
        state = self.get_by_conversation(conversation_id)
        if (
            state is not None
            and state.current_session_id == new_session_id
            and state.last_finalized_session_id == old_session_id
        ):
            return state

        now = time.time()
        last_finalized_frontier = max(
            int(finalized_frontier_store_id or 0),
            state.last_finalized_frontier_store_id if state else 0,
        )
        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                rollover_carry_over_context, binding_generation,
                updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = 0,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                rollover_carry_over_context = excluded.rollover_carry_over_context,
                binding_generation = lcm_lifecycle_state.binding_generation + 1,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                new_session_id,
                old_session_id,
                last_finalized_frontier,
                now,
                now,
                now,
                now,
                1 if carry_over_context else 0,
                now,
            ),
        )
        self._conn.commit()
        updated = self.get_by_conversation(conversation_id)
        assert updated is not None
        return updated

    @staticmethod
    def record_rollover_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        current_frontier_store_id: int,
        finalized_frontier_store_id: int,
        carry_over_context: bool,
        frozen_generation: int = 0,
    ) -> None:
        """Publish the rollover lifecycle row on a caller-owned transaction."""
        now = time.time()
        LifecycleStateStore.record_rollover_head_no_commit(
            conn,
            conversation_id,
            old_session_id=old_session_id,
            new_session_id=new_session_id,
            carry_over_context=carry_over_context,
            rollover_generation=frozen_generation,
            finalized_boundary_store_id=finalized_frontier_store_id,
        )
        conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id, current_session_id, last_finalized_session_id,
                current_frontier_store_id, last_finalized_frontier_store_id,
                current_bound_at, last_finalized_at, last_rollover_at,
                last_reset_at, rollover_carry_over_context,
                binding_generation, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = excluded.current_frontier_store_id,
                last_finalized_frontier_store_id = MAX(
                    lcm_lifecycle_state.last_finalized_frontier_store_id,
                    excluded.last_finalized_frontier_store_id
                ),
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                rollover_carry_over_context = excluded.rollover_carry_over_context,
                binding_generation = lcm_lifecycle_state.binding_generation + 1,
                debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = excluded.updated_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                new_session_id,
                old_session_id,
                max(0, int(current_frontier_store_id or 0)),
                max(0, int(finalized_frontier_store_id or 0)),
                now, now, now, now, 1 if carry_over_context else 0, now,
            ),
        )
        if not carry_over_context:
            # Retain the v11 row only for mixed-version writer compatibility;
            # v12 enforcement never reads its scalar cutoff.
            LifecycleStateStore.record_no_carry_policy_no_commit(
                conn,
                conversation_id,
                old_session_id=old_session_id,
                new_session_id=new_session_id,
                finalized_cutoff_store_id=finalized_frontier_store_id,
                frozen_generation=frozen_generation,
            )

    @staticmethod
    def record_rollover_head_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        carry_over_context: bool,
        rollover_generation: int,
        finalized_boundary_store_id: int,
    ) -> None:
        """CAS the authoritative latest rollover head and historical set."""
        now = time.time()
        generation = max(0, int(rollover_generation or 0))
        conn.execute(
            """INSERT INTO lcm_rollover_heads(
                   conversation_id, current_session_id,
                   last_finalized_session_id, carry_over_context,
                   rollover_epoch, frontier_generation, created_at, updated_at
               )
               SELECT ?, ?, ?, ?, 1, ?, ?, ?
               WHERE EXISTS (
                   SELECT 1 FROM lcm_active_frontiers AS f
                   WHERE f.conversation_id = ? AND f.generation = ?
                     AND f.session_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM lcm_active_frontiers AS newer
                         WHERE newer.conversation_id = f.conversation_id
                           AND newer.generation > f.generation
                     )
               )
               ON CONFLICT(conversation_id) DO UPDATE SET
                   current_session_id = excluded.current_session_id,
                   last_finalized_session_id = excluded.last_finalized_session_id,
                   carry_over_context = excluded.carry_over_context,
                   rollover_epoch = lcm_rollover_heads.rollover_epoch + 1,
                   frontier_generation = excluded.frontier_generation,
                   updated_at = excluded.updated_at
               WHERE lcm_rollover_heads.current_session_id = ?
                 AND excluded.frontier_generation > lcm_rollover_heads.frontier_generation""",
            (
                conversation_id, new_session_id, old_session_id,
                1 if carry_over_context else 0, generation, now, now,
                conversation_id, generation, new_session_id, old_session_id,
            ),
        )
        head = conn.execute(
            """SELECT current_session_id, last_finalized_session_id,
                      carry_over_context, rollover_epoch, frontier_generation
               FROM lcm_rollover_heads WHERE conversation_id = ?""",
            (conversation_id,),
        ).fetchone()
        if not head or (
            str(head[0]) != new_session_id
            or str(head[1]) != old_session_id
            or int(head[2]) != int(bool(carry_over_context))
            or int(head[4]) != generation
        ):
            raise RuntimeError("rollover head owner/generation CAS lost")
        if not carry_over_context:
            conn.execute(
                """INSERT INTO lcm_protected_sessions(
                       conversation_id, finalized_session_id,
                       finalized_boundary_store_id, protected_at_generation,
                       rollover_epoch, created_at, updated_at
                   ) VALUES(?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(conversation_id, finalized_session_id) DO UPDATE SET
                       finalized_boundary_store_id = MAX(
                           lcm_protected_sessions.finalized_boundary_store_id,
                           excluded.finalized_boundary_store_id
                       ),
                       protected_at_generation = MIN(
                           lcm_protected_sessions.protected_at_generation,
                           excluded.protected_at_generation
                       ),
                       rollover_epoch = MIN(
                           lcm_protected_sessions.rollover_epoch,
                           excluded.rollover_epoch
                       ),
                       updated_at = excluded.updated_at""",
                (
                    conversation_id, old_session_id,
                    max(0, int(finalized_boundary_store_id or 0)), generation,
                    int(head[3]), now, now,
                ),
            )

    @staticmethod
    def record_no_carry_policy_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_cutoff_store_id: int,
        frozen_generation: int,
    ) -> None:
        """Publish the lifecycle-independent no-carry cutoff policy."""
        now = time.time()
        conn.execute(
            """INSERT INTO lcm_rollover_policies(
                   conversation_id, finalized_session_id, current_session_id,
                   finalized_cutoff_store_id, frozen_generation,
                   carry_over_context, created_at, updated_at
               ) VALUES(?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(conversation_id) DO UPDATE SET
                   finalized_session_id = excluded.finalized_session_id,
                   current_session_id = excluded.current_session_id,
                   finalized_cutoff_store_id = MAX(
                       lcm_rollover_policies.finalized_cutoff_store_id,
                       excluded.finalized_cutoff_store_id
                   ),
                   frozen_generation = MAX(
                       lcm_rollover_policies.frozen_generation,
                       excluded.frozen_generation
                   ),
                   carry_over_context = 0,
                   updated_at = excluded.updated_at""",
            (
                conversation_id, old_session_id, new_session_id,
                max(0, int(finalized_cutoff_store_id or 0)),
                max(0, int(frozen_generation or 0)), now, now,
            ),
        )

    @staticmethod
    def extend_no_carry_finalized_boundary_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        current_session_id: str,
        frontier_store_id: int,
    ) -> None:
        """Advance durable old-session coverage without changing frontier."""
        now = time.time()
        boundary = max(0, int(frontier_store_id or 0))
        authorization = conn.execute(
            """SELECT h.rollover_epoch
               FROM lcm_rollover_heads AS h
               JOIN lcm_protected_sessions AS p
                 ON p.conversation_id = h.conversation_id
                AND p.finalized_session_id = ?
               JOIN lcm_active_frontiers AS f
                 ON f.conversation_id = h.conversation_id
                AND f.session_id = h.current_session_id
                AND f.generation >= h.frontier_generation
               WHERE h.conversation_id = ? AND h.current_session_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM lcm_active_frontiers AS newer
                     WHERE newer.conversation_id = f.conversation_id
                       AND newer.generation > f.generation
                 )""",
            (old_session_id, conversation_id, current_session_id),
        ).fetchone()
        if authorization is None:
            raise RuntimeError("rollover head no longer authorizes protected storage extension")
        conn.execute(
            """UPDATE lcm_lifecycle_state
               SET last_finalized_frontier_store_id = MAX(
                       last_finalized_frontier_store_id, ?
                   ), updated_at = ?
               WHERE conversation_id = ? AND current_session_id = ?
                 AND last_finalized_session_id = ?""",
            (boundary, now, conversation_id, current_session_id, old_session_id),
        )
        protected = conn.execute(
            """UPDATE lcm_protected_sessions
               SET finalized_boundary_store_id = MAX(finalized_boundary_store_id, ?),
                   updated_at = ?
               WHERE conversation_id = ? AND finalized_session_id = ?
            """,
            (boundary, now, conversation_id, old_session_id),
        )
        # Best-effort compatibility for a schema-v11 process that was already
        # open during migration. No v12 trigger or reader treats this scalar as
        # authority, so interleaved newer-session store ids cannot freeze it.
        conn.execute(
            """UPDATE lcm_rollover_policies
               SET finalized_cutoff_store_id = MAX(finalized_cutoff_store_id, ?),
                   updated_at = ?
               WHERE conversation_id = ? AND finalized_session_id = ?""",
            (boundary, now, conversation_id, old_session_id),
        )
        if int(protected.rowcount or 0) != 1:
            raise RuntimeError("protected session disappeared during storage extension")

    @staticmethod
    def extend_finalized_rollover_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        current_session_id: str,
        frontier_store_id: int,
    ) -> None:
        """Advance both winner and finalized boundaries under locked ownership."""
        now = time.time()
        authorized = conn.execute(
            """SELECT 1 FROM lcm_rollover_heads AS h
               JOIN lcm_active_frontiers AS f
                 ON f.conversation_id = h.conversation_id
                AND f.session_id = h.current_session_id
                AND f.generation >= h.frontier_generation
               WHERE h.conversation_id = ? AND h.current_session_id = ?
                 AND h.last_finalized_session_id = ?
                 AND h.carry_over_context = 1
                 AND NOT EXISTS (
                     SELECT 1 FROM lcm_active_frontiers AS newer
                     WHERE newer.conversation_id = f.conversation_id
                       AND newer.generation > f.generation
                 )""",
            (conversation_id, current_session_id, old_session_id),
        ).fetchone()
        if authorized is None:
            raise RuntimeError("rollover head no longer authorizes finalized-session extension")
        cur = conn.execute(
            """UPDATE lcm_lifecycle_state
               SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                   last_finalized_frontier_store_id = MAX(last_finalized_frontier_store_id, ?),
                   updated_at = ?
               WHERE conversation_id = ?
                 AND current_session_id = ?
                 AND last_finalized_session_id = ?""",
            (
                max(0, int(frontier_store_id or 0)),
                max(0, int(frontier_store_id or 0)),
                now,
                conversation_id,
                current_session_id,
                old_session_id,
            ),
        )
        # Lifecycle is a reconstructable cache. A stale rewrite may make this
        # update a no-op, but cannot invalidate the head-authorized frontier.

    @staticmethod
    def finalize_session_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        session_id: str,
        frontier_store_id: int,
    ) -> None:
        """Finalize a current session on a caller-owned writer transaction."""
        now = time.time()
        cur = conn.execute(
            """UPDATE lcm_lifecycle_state
               SET current_session_id = NULL,
                   last_finalized_session_id = ?,
                   current_frontier_store_id = 0,
                   rollover_carry_over_context = NULL,
                   binding_generation = binding_generation + 1,
                   last_finalized_frontier_store_id = MAX(
                       last_finalized_frontier_store_id, ?
                   ),
                   last_finalized_at = ?,
                   updated_at = ?
               WHERE conversation_id = ? AND current_session_id = ?""",
            (
                session_id,
                max(0, int(frontier_store_id or 0)),
                now,
                now,
                conversation_id,
                session_id,
            ),
        )
        if int(cur.rowcount or 0) != 1:
            raise RuntimeError("lifecycle no longer authorizes session finalization")

    def get_fragmentation_stats(self, state_db_path: str | Path | None = None) -> dict[str, Any]:
        """Return read-only lifecycle/session fragmentation diagnostics.

        This intentionally reports mismatches only. It does not infer that every
        mismatch is corrupt, and it never rewrites lifecycle, message, DAG, or
        Hermes host state. Repair/cleanup flows must stay explicit and separate.
        """
        conn = self._conn
        assert conn is not None

        def _count(query: str, params: tuple[Any, ...] = ()) -> int:
            row = conn.execute(query, params).fetchone()
            return int(row[0] if row else 0)

        def _session_ids(query: str) -> set[str]:
            return {
                str(row[0])
                for row in conn.execute(query).fetchall()
                if row[0]
            }

        message_sessions = _session_ids("SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL")
        node_sessions = _session_ids("SELECT DISTINCT session_id FROM summary_nodes WHERE session_id IS NOT NULL")
        lcm_any_sessions = message_sessions | node_sessions
        state_sessions: set[str] = set()
        state_db_read_success = False
        lifecycle_current_sessions = _session_ids(
            "SELECT DISTINCT current_session_id FROM lcm_lifecycle_state WHERE current_session_id IS NOT NULL"
        )
        lifecycle_last_finalized_sessions = _session_ids(
            "SELECT DISTINCT last_finalized_session_id FROM lcm_lifecycle_state WHERE last_finalized_session_id IS NOT NULL"
        )
        lifecycle_referenced_sessions = lifecycle_current_sessions | lifecycle_last_finalized_sessions

        empty_lifecycle_rows = 0
        for row in conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id
            FROM lcm_lifecycle_state
            """
        ).fetchall():
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or refs.isdisjoint(lcm_any_sessions):
                empty_lifecycle_rows += 1

        stats: dict[str, Any] = {
            "read_only": True,
            "lifecycle_rows": _count("SELECT COUNT(*) FROM lcm_lifecycle_state"),
            "empty_lifecycle_rows": empty_lifecycle_rows,
            "messages_total": _count("SELECT COUNT(*) FROM messages"),
            "summary_nodes_total": _count("SELECT COUNT(*) FROM summary_nodes"),
            "distinct_message_sessions": len(message_sessions),
            "distinct_node_sessions": len(node_sessions),
            "distinct_lcm_any_sessions": len(lcm_any_sessions),
            "lifecycle_current_sessions": len(lifecycle_current_sessions),
            "lifecycle_last_finalized_sessions": len(lifecycle_last_finalized_sessions),
            "lifecycle_current_missing_in_messages": len(lifecycle_current_sessions - message_sessions),
            "lifecycle_current_missing_in_nodes": len(lifecycle_current_sessions - node_sessions),
            "lifecycle_current_missing_in_lcm_any": len(lifecycle_current_sessions - lcm_any_sessions),
            "lifecycle_last_finalized_missing_in_messages": len(lifecycle_last_finalized_sessions - message_sessions),
            "lifecycle_last_finalized_missing_in_nodes": len(lifecycle_last_finalized_sessions - node_sessions),
            "lifecycle_last_finalized_missing_in_lcm_any": len(lifecycle_last_finalized_sessions - lcm_any_sessions),
            "message_sessions_without_lifecycle_current": len(message_sessions - lifecycle_current_sessions),
            "message_sessions_without_lifecycle_reference": len(message_sessions - lifecycle_referenced_sessions),
            "node_sessions_without_lifecycle_reference": len(node_sessions - lifecycle_referenced_sessions),
            "state_db_checked": False,
            "state_db_error": "",
            "state_sessions_total": 0,
            "lifecycle_current_missing_in_state": 0,
            "lifecycle_last_finalized_missing_in_state": 0,
            "lcm_message_sessions_missing_in_state": 0,
            "lcm_node_sessions_missing_in_state": 0,
            "state_sessions_missing_in_lcm_messages": 0,
            "state_sessions_missing_in_lcm_any": 0,
        }

        if state_db_path:
            path = Path(state_db_path).expanduser()
            if path.exists():
                stats["state_db_checked"] = True
                try:
                    state_uri = path.resolve().as_uri() + "?mode=ro"
                    state_conn = sqlite3.connect(state_uri, uri=True)
                    try:
                        state_rows = state_conn.execute("SELECT id FROM sessions WHERE id IS NOT NULL").fetchall()
                    finally:
                        state_conn.close()
                    state_sessions = {str(row[0]) for row in state_rows if row[0]}
                    state_db_read_success = True
                    stats.update({
                        "state_sessions_total": len(state_sessions),
                        "lifecycle_current_missing_in_state": len(lifecycle_current_sessions - state_sessions),
                        "lifecycle_last_finalized_missing_in_state": len(
                            lifecycle_last_finalized_sessions - state_sessions
                        ),
                        "lcm_message_sessions_missing_in_state": len(message_sessions - state_sessions),
                        "lcm_node_sessions_missing_in_state": len(node_sessions - state_sessions),
                        "state_sessions_missing_in_lcm_messages": len(state_sessions - message_sessions),
                        "state_sessions_missing_in_lcm_any": len(state_sessions - lcm_any_sessions),
                    })
                except Exception as exc:  # pragma: no cover - defensive
                    stats["state_db_error"] = str(exc)
            else:
                stats["state_db_error"] = f"state database not found: {path}"

        stats["classification"] = self._classify_fragmentation(
            lifecycle_rows=stats["lifecycle_rows"],
            lifecycle_current_sessions=lifecycle_current_sessions,
            lifecycle_last_finalized_sessions=lifecycle_last_finalized_sessions,
            message_sessions=message_sessions,
            node_sessions=node_sessions,
            lcm_any_sessions=lcm_any_sessions,
            lifecycle_referenced_sessions=lifecycle_referenced_sessions,
            state_sessions=state_sessions,
            state_db_read_success=state_db_read_success,
        )

        return stats

    @staticmethod
    def _classify_fragmentation(
        *,
        lifecycle_rows: int,
        lifecycle_current_sessions: set[str],
        lifecycle_last_finalized_sessions: set[str],
        message_sessions: set[str],
        node_sessions: set[str],
        lcm_any_sessions: set[str],
        lifecycle_referenced_sessions: set[str],
        state_sessions: set[str],
        state_db_read_success: bool,
    ) -> dict[str, Any]:
        """Bucket lifecycle mismatches into operator-readable read-only categories."""

        def sample(session_ids: set[str], limit: int = 5) -> list[str]:
            return sorted(session_ids)[:limit]

        categories: list[dict[str, Any]] = []

        def add_category(
            name: str,
            session_ids: set[str],
            *,
            severity: str,
            description: str,
            recommended_action: str,
        ) -> None:
            if not session_ids:
                return
            categories.append({
                "name": name,
                "severity": severity,
                "count": len(session_ids),
                "sample_session_ids": sample(session_ids),
                "description": description,
                "recommended_action": recommended_action,
            })

        add_category(
            "stale_lifecycle_current",
            lifecycle_current_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle current-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; these are often old or ephemeral lifecycle rows, not automatic corruption.",
        )
        add_category(
            "stale_lifecycle_finalized",
            lifecycle_last_finalized_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle finalized-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; only remove with an explicit backup-first lifecycle cleanup flow.",
        )
        if lifecycle_rows > 0:
            add_category(
                "lcm_message_sessions_without_lifecycle_reference",
                message_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Raw-message sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; investigate only if the sessions should belong to an active conversation.",
            )
            add_category(
                "lcm_node_sessions_without_lifecycle_reference",
                node_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Summary-node sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; verify expand/search still work before considering cleanup.",
            )

        if state_db_read_success:
            add_category(
                "lcm_message_sessions_missing_in_state",
                message_sessions - state_sessions,
                severity="notice",
                description="LCM raw-message sessions are absent from the Hermes session database.",
                recommended_action="Treat as retained or imported context unless the session should still be browsable in host session history.",
            )
            add_category(
                "lcm_node_sessions_missing_in_state",
                node_sessions - state_sessions,
                severity="notice",
                description="LCM summary-node sessions are absent from the Hermes session database.",
                recommended_action="Keep read-only; this can happen after host session pruning while LCM retained summaries remain useful.",
            )
            add_category(
                "state_only_sessions",
                state_sessions - lcm_any_sessions,
                severity="notice",
                description="Hermes host sessions exist without raw messages or summary nodes in LCM.",
                recommended_action="Usually benign for sessions outside LCM scope, ignored sessions, or sessions that never reached durable LCM ingest.",
            )

        warn_count = sum(1 for item in categories if item["severity"] == "warn")
        status = "warn" if warn_count else ("notice" if categories else "pass")
        summary = (
            "no lifecycle fragmentation categories detected"
            if not categories
            else f"{len(categories)} lifecycle fragmentation categories need review"
        )
        return {
            "read_only": True,
            "status": status,
            "summary": summary,
            "categories": categories,
        }

    @_synchronized
    def record_debt(
        self,
        conversation_id: str | None,
        *,
        kind: str,
        size_estimate: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = ?,
                debt_size_estimate = ?,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (kind, max(0, int(size_estimate or 0)), now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    def clear_debt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_maintenance_attempt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_maintenance_attempt_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_reset(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_reset_at = ?,
                debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def prune_empty_sessions(
        self,
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        max_age_hours: float | None = None,
    ) -> int:
        """Delete lifecycle rows for sessions with no stored data.

        A row is eligible when BOTH referenced session IDs
        (``current_session_id`` and ``last_finalized_session_id``)
        have zero messages AND zero summary_nodes in the main store.

        Only the lifecycle table is modified — messages, nodes, and FTS
        indexes are untouched (they already contain no data for these sessions).

        Args:
            protected_session_ids: Sessions that must never be deleted
                (typically the actively-bound engine session).
            max_age_hours: Only delete rows older than this many hours.
                ``None`` means delete all eligible rows regardless of age.

        Returns:
            Number of rows deleted.
        """
        conn = self._conn
        assert conn is not None
        protected = {str(s) for s in (protected_session_ids or ()) if s}

        conn.execute("BEGIN IMMEDIATE")
        try:
            sessions_with_data: set[str] = set()
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            def _session_has_data(session_id: str) -> bool:
                if not session_id:
                    return False
                if "messages" in tables and conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone():
                    return True
                if "summary_nodes" in tables and conn.execute(
                    "SELECT 1 FROM summary_nodes WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone():
                    return True
                return False

            if "messages" in tables:
                for row in conn.execute(
                    "SELECT DISTINCT session_id FROM messages"
                ).fetchall():
                    sessions_with_data.add(str(row[0]))
            if "summary_nodes" in tables:
                for row in conn.execute(
                    "SELECT DISTINCT session_id FROM summary_nodes"
                ).fetchall():
                    sessions_with_data.add(str(row[0]))

            now = time.time()
            max_age_seconds = (
                float(max_age_hours) * 3600.0
                if max_age_hours is not None
                else None
            )
            deleted = 0

            rows = conn.execute(
                "SELECT * FROM lcm_lifecycle_state"
            ).fetchall()
            for row in rows:
                cur = str(row["current_session_id"] or "")
                fin = str(row["last_finalized_session_id"] or "")

                if ((cur and cur in sessions_with_data)
                        or (fin and fin in sessions_with_data)):
                    continue

                refs = {r for r in (cur, fin) if r}
                if refs & protected:
                    continue

                if max_age_seconds is not None:
                    row_age = (
                        row["current_bound_at"]
                        or row["last_finalized_at"]
                        or row["updated_at"]
                    )
                    if row_age is not None and (now - float(row_age)) < max_age_seconds:
                        continue

                # Recheck against the tables right before deletion. BEGIN
                # IMMEDIATE blocks concurrent writers while this transaction is
                # open; this fresh query also keeps the safety check honest if
                # the broad snapshot logic above changes later.
                if _session_has_data(cur) or _session_has_data(fin):
                    continue

                conn.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                    (row["conversation_id"],),
                )
                deleted += 1

            if deleted:
                conn.commit()
            else:
                conn.rollback()
            return deleted
        except Exception:
            conn.rollback()
            raise

    def delete_safe_rows_for_sessions(
        self,
        session_ids: set[str] | list[str] | tuple[str, ...],
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[int, int]:
        candidates = {str(s) for s in session_ids if s}
        if not candidates:
            return 0, 0
        protected = {str(s) for s in (protected_session_ids or ()) if s}
        deleted = 0
        skipped = 0
        rows = self._conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall()
        for row in rows:
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or not (refs & candidates):
                continue
            if refs & protected:
                skipped += 1
                continue
            if refs <= candidates:
                self._conn.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                    (row["conversation_id"],),
                )
                deleted += 1
                continue
            skipped += 1
        if deleted:
            self._conn.commit()
        return deleted, skipped

    def advance_frontier(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        publication_conn = getattr(self._transaction_local, "connection", None)
        if publication_conn is not None:
            # Global lock order: a caller that already owns SQLite's writer lock
            # must never acquire a sibling store lock. Lifecycle operations take
            # ``_lock`` before beginning their own SQL transaction; publication
            # therefore performs its lifecycle SQL directly on the coordinator
            # connection and lets SQLite serialize concurrent lifecycle writers.
            advanced = self.advance_frontier_no_commit(
                publication_conn,
                conversation_id,
                session_id,
                frontier_store_id,
            )
            if not advanced:
                return None
            row = publication_conn.execute(
                """
                SELECT conversation_id, current_session_id,
                       last_finalized_session_id, current_frontier_store_id,
                       last_finalized_frontier_store_id, debt_kind,
                       debt_size_estimate, current_bound_at, last_finalized_at,
                       debt_updated_at, last_maintenance_attempt_at,
                       last_rollover_at, last_reset_at,
                       rollover_carry_over_context, binding_generation, updated_at
                FROM lcm_lifecycle_state WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return LifecycleState(
                conversation_id=str(row[0]),
                current_session_id=row[1],
                last_finalized_session_id=row[2],
                current_frontier_store_id=int(row[3] or 0),
                last_finalized_frontier_store_id=int(row[4] or 0),
                debt_kind=row[5],
                debt_size_estimate=int(row[6] or 0),
                current_bound_at=row[7],
                last_finalized_at=row[8],
                debt_updated_at=row[9],
                last_maintenance_attempt_at=row[10],
                last_rollover_at=row[11],
                last_reset_at=row[12],
                rollover_carry_over_context=(
                    None if row[13] is None else bool(row[13])
                ),
                binding_generation=int(row[14] or 0),
                updated_at=float(row[15] or 0.0),
            )
        with self._lock:
            state = self.get_by_conversation(conversation_id)
            if state is None or state.current_session_id != session_id:
                return state
            conn = self._conn
            assert conn is not None
            advanced = self.advance_frontier_no_commit(
                conn,
                conversation_id,
                session_id,
                frontier_store_id,
            )
            if advanced:
                conn.commit()
            return self.get_by_conversation(conversation_id)

    @staticmethod
    def advance_frontier_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
        frontier_store_id: int,
    ) -> bool:
        """Advance lifecycle on a caller-owned transaction, without commit.

        Returns whether the currently-bound session acknowledged the marker.
        SQL ``MAX`` preserves the normal method's monotonicity guarantee.
        """
        cursor = conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                updated_at = ?
            WHERE conversation_id = ? AND current_session_id = ?
            """,
            (
                int(frontier_store_id or 0),
                time.time(),
                conversation_id,
                session_id,
            ),
        )
        return cursor.rowcount > 0
