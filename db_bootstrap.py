"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sqlite3
import time
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


class SchemaVersionTooNewError(RuntimeError):
    """Raised when a database was written by a newer LCM schema than this build.

    Opening and migrating such a database with older code risks silently
    corrupting data written under semantics this build does not understand, so
    we refuse rather than degrade.
    """


class SQLiteStartupBusyError(RuntimeError):
    """Raised when bounded SQLite startup lock waiting is exhausted."""


SCHEMA_VERSION = 12
SQLITE_BUSY_TIMEOUT_MS = 30_000
SQLITE_STARTUP_BACKOFF_INITIAL_SECONDS = 0.01
SQLITE_STARTUP_BACKOFF_MAX_SECONDS = 0.25
# Bounded busy wait for foreground cutover so compress() cannot hang
# indefinitely behind a concurrent writer (gateway host holds compression_locks).
FOREGROUND_COMPRESS_BUSY_TIMEOUT_MS = 2_000
# Hard wall-clock budget for a single compress() invocation.
FOREGROUND_COMPRESS_DEADLINE_SECONDS = 45.0
_MIN_DISK_SPACE_BYTES = 50 * 1024 * 1024
REQUIRED_CORE_TABLES = (
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "lcm_protected_sessions",
    "lcm_rollover_heads",
    "lcm_session_end_receipts",
    "lcm_migration_state",
    "lcm_focus_briefs",
    "messages_fts",
    "nodes_fts",
)

# Test-only subprocess crash injection. Production callers never set this;
# tests assign a phase name in the child process and verify SQLite rolls the
# whole versioned migration back after an abrupt exit.
_MIGRATION_CRASH_PHASE: str | None = None


def _migration_crash_boundary(phase: str) -> None:
    if _MIGRATION_CRASH_PHASE == phase:
        os._exit(88)  # noqa: PLW1510 - deliberate migration crash injection


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_sqls: Sequence[str],
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_sqls = tuple(trigger_sqls)


def _is_sqlite_busy_error(exc: BaseException) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    if error_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(exc).lower()
    return isinstance(exc, sqlite3.Error) and (
        "locked" in message or "busy" in message
    )


def _execute_startup_pragma_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    *,
    deadline: float,
) -> None:
    """Execute a startup PRAGMA with bounded retry for SQLITE_BUSY/LOCKED.

    SQLite's journal-mode transition does not consistently invoke the busy
    handler when another connection is publishing WAL/DDL frames.  Retrying
    that one connection-local startup operation closes the gap left by
    ``busy_timeout`` without weakening the database writer lock that protects
    migrations and FTS repair.
    """
    delay = SQLITE_STARTUP_BACKOFF_INITIAL_SECONDS
    while True:
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_busy_error(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SQLiteStartupBusyError(
                    f"SQLite startup remained busy while executing {sql!r}"
                ) from exc
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, SQLITE_STARTUP_BACKOFF_MAX_SECONDS)


def configure_connection(conn: sqlite3.Connection) -> None:
    """Configure SQLite connection for WAL durability and hygiene.

    In a multi-agent deployment (gateway process + CLI sessions + sub-agents),
    every process opens its own sqlite3.Connection pointing at the same
    lcm.db file.  These settings improve committed-write durability and WAL
    hygiene, but do NOT make sibling processes safe from an unexpected process
    death.  Abnormal exit still depends on normal SQLite WAL recovery;
    application-level checkpoints only run during graceful shutdown (see
    ``MessageStore.close()`` etc.).

    Key design decisions:
    - journal_mode=WAL  : writes go to a separate log; readers never block.
    - synchronous=FULL  : fsync both the WAL and the WAL index before every
                          write transaction commit.  WAL + FULL is the only
                          combination SQLite guarantees survives power loss
                          without data loss (NORMAL may lose the WAL index).
    - wal_autocheckpoint=500 : after 500 WAL pages (~2 MB) SQLite will try
                               an automatic passive checkpoint.  This is a
                               best-effort hint — it is silently skipped when
                               another connection holds a read transaction.
                               Under checkpoint starvation WAL can grow well
                               beyond this trigger.
    - journal_size_limit=67108864 (64 MiB) : limits the WAL file size after
                                             a successful checkpoint or reset.
                                             It does NOT force a checkpoint
                                             or cap growth while another
                                             connection holds an old WAL
                                             end mark.
    - mmap_size=268435456 (256 MiB)        : memory-map reads so concurrent
                                              readers cache WAL pages in RAM.
    """
    # Install the wait policy before journal-mode negotiation: concurrent
    # process startups can otherwise fail immediately while one connection is
    # publishing migration/DDL frames.
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    deadline = time.monotonic() + (SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    for pragma in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=FULL",
        "PRAGMA wal_autocheckpoint=500",
        "PRAGMA journal_size_limit=67108864",
        "PRAGMA mmap_size=268435456",
    ):
        _execute_startup_pragma_with_retry(conn, pragma, deadline=deadline)


def add_column_if_missing(
    conn: sqlite3.Connection,
    existing_columns: set[str],
    column: str,
    alter_sql: str,
) -> None:
    """Idempotently add a column, tolerating a concurrent process that won the race.

    In the multi-agent deployment (gateway + CLI sessions + sub-agents) every
    process opens its own connection to the same ``lcm.db`` and runs startup
    migrations concurrently.  A plain check-``PRAGMA table_info``-then-``ALTER``
    races: two processes both observe the column as absent (each within its own
    connection snapshot) and both issue ``ALTER TABLE ... ADD COLUMN``.  The loser
    then raised ``sqlite3.OperationalError: duplicate column name``, which
    propagated out of ``_init_db`` and crashed store construction.  Swallowing
    exactly that error makes the migration idempotent under concurrency; any other
    OperationalError still propagates.
    """
    if column in existing_columns:
        return
    try:
        conn.execute(alter_sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def ensure_schema_version_monotonic_guard(conn: sqlite3.Connection) -> None:
    """Prevent any writer, including older binaries, from lowering the marker."""
    ensure_metadata_table(conn)
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_schema_version_monotonic
        BEFORE UPDATE OF value ON metadata
        WHEN OLD.key = 'schema_version'
          AND NEW.key = 'schema_version'
          AND CAST(NEW.value AS INTEGER) < CAST(OLD.value AS INTEGER)
        BEGIN
            SELECT RAISE(IGNORE);
        END
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0




def read_existing_schema_version(conn: sqlite3.Connection) -> int:
    """Return schema_version without creating or modifying schema objects."""
    metadata_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()
    if not metadata_exists:
        return 0
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def refuse_schema_version_too_new(conn: sqlite3.Connection) -> None:
    """Raise before any startup DDL when a newer build owns the DB."""
    current_version = read_existing_schema_version(conn)
    if current_version > SCHEMA_VERSION:
        raise SchemaVersionTooNewError(
            f"LCM database schema version {current_version} is newer than this "
            f"build supports (v{SCHEMA_VERSION}). Refusing to open to avoid "
            f"corrupting data written by a newer hermes-lcm. Upgrade the plugin "
            f"or restore a pre-upgrade backup (.db/-wal/-shm)."
        )

def ensure_migration_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        )
        """
    )


def ensure_lifecycle_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT,
            last_finalized_session_id TEXT,
            current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            debt_kind TEXT,
            debt_size_estimate INTEGER NOT NULL DEFAULT 0,
            current_bound_at REAL,
            last_finalized_at REAL,
            debt_updated_at REAL,
            last_maintenance_attempt_at REAL,
            last_rollover_at REAL,
            last_reset_at REAL,
            rollover_carry_over_context INTEGER
                CHECK (rollover_carry_over_context IN (0, 1)),
            binding_generation INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id)"
    )


def ensure_lifecycle_state_columns(conn: sqlite3.Connection) -> None:
    ensure_lifecycle_state_table(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "debt_kind",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_kind TEXT",
    )
    add_column_if_missing(
        conn, columns, "debt_size_estimate",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_size_estimate INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, columns, "debt_updated_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_updated_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_maintenance_attempt_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_rollover_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_rollover_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_reset_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_reset_at REAL",
    )
    add_column_if_missing(
        conn, columns, "rollover_carry_over_context",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN rollover_carry_over_context INTEGER "
        "CHECK (rollover_carry_over_context IN (0, 1))",
    )
    add_column_if_missing(
        conn, columns, "binding_generation",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN binding_generation "
        "INTEGER NOT NULL DEFAULT 0",
    )


def ensure_rollover_policy_table(conn: sqlite3.Connection) -> None:
    """Keep the schema-v11 ledger available to already-running old writers.

    Schema v12 never reads this scalar-cutoff table as authority.  It remains
    solely as a compatibility landing zone until a deployment can guarantee
    that no schema-v11 process is still alive.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_rollover_policies (
            conversation_id TEXT PRIMARY KEY,
            finalized_session_id TEXT NOT NULL,
            current_session_id TEXT NOT NULL,
            finalized_cutoff_store_id INTEGER NOT NULL DEFAULT 0
                CHECK (finalized_cutoff_store_id >= 0),
            frozen_generation INTEGER NOT NULL DEFAULT 0
                CHECK (frozen_generation >= 0),
            carry_over_context INTEGER NOT NULL DEFAULT 0
                CHECK (carry_over_context IN (0, 1)),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_lcm_rollover_policy_finalized_session
           ON lcm_rollover_policies(finalized_session_id)"""
    )


def ensure_rollover_v12_tables(conn: sqlite3.Connection) -> None:
    """Create the v12 historical protection, head, and ingest receipt schema."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_protected_sessions (
            conversation_id TEXT NOT NULL,
            finalized_session_id TEXT NOT NULL,
            finalized_boundary_store_id INTEGER NOT NULL DEFAULT 0
                CHECK (finalized_boundary_store_id >= 0),
            protected_at_generation INTEGER NOT NULL DEFAULT 0
                CHECK (protected_at_generation >= 0),
            rollover_epoch INTEGER NOT NULL DEFAULT 0
                CHECK (rollover_epoch >= 0),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (conversation_id, finalized_session_id)
        )
        """
    )
    _migration_crash_boundary("v12_after_protected_table")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_lcm_protected_session
           ON lcm_protected_sessions(finalized_session_id, conversation_id)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_rollover_heads (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT NOT NULL,
            last_finalized_session_id TEXT NOT NULL,
            carry_over_context INTEGER NOT NULL
                CHECK (carry_over_context IN (0, 1)),
            rollover_epoch INTEGER NOT NULL DEFAULT 1
                CHECK (rollover_epoch > 0),
            frontier_generation INTEGER NOT NULL DEFAULT 0
                CHECK (frontier_generation >= 0),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    _migration_crash_boundary("v12_after_head_table")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_lcm_rollover_head_current
           ON lcm_rollover_heads(current_session_id)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_session_end_receipts (
            conversation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            payload_fingerprint TEXT NOT NULL,
            rollover_epoch INTEGER NOT NULL DEFAULT 0
                CHECK (rollover_epoch >= 0),
            prefix_count INTEGER NOT NULL DEFAULT 0
                CHECK (prefix_count >= 0),
            retained_count INTEGER NOT NULL DEFAULT 0
                CHECK (retained_count >= 0),
            created_at REAL NOT NULL,
            PRIMARY KEY (
                conversation_id, session_id, payload_fingerprint, rollover_epoch
            )
        )
        """
    )
    _migration_crash_boundary("v12_after_receipt_table")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_lcm_session_end_receipts_retention
           ON lcm_session_end_receipts(conversation_id, created_at DESC)"""
    )


def _drop_schema_v11_cutoff_triggers(conn: sqlite3.Connection) -> None:
    for name in (
        "lcm_no_carry_frontier_insert",
        "lcm_no_carry_frontier_update",
        "lcm_no_carry_item_insert",
        "lcm_no_carry_item_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def backfill_rollover_v12(conn: sqlite3.Connection) -> None:
    """Atomically translate v10 lifecycle and v11 policy state into v12."""
    ensure_rollover_v12_tables(conn)
    now = time.time()
    # Every v11 no-carry policy is historical protection, even when its
    # current owner later became stale after a carry rollover.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_policies'"
    ).fetchone():
        conn.execute(
            """INSERT INTO lcm_protected_sessions(
                   conversation_id, finalized_session_id,
                   finalized_boundary_store_id, protected_at_generation,
                   rollover_epoch, created_at, updated_at
               )
               SELECT conversation_id, finalized_session_id,
                      MAX(0, finalized_cutoff_store_id),
                      MAX(0, frozen_generation), 1, created_at, updated_at
               FROM lcm_rollover_policies WHERE carry_over_context = 0
               ON CONFLICT(conversation_id, finalized_session_id) DO UPDATE SET
                   finalized_boundary_store_id = MAX(
                       lcm_protected_sessions.finalized_boundary_store_id,
                       excluded.finalized_boundary_store_id
                   ),
                   protected_at_generation = MAX(
                       lcm_protected_sessions.protected_at_generation,
                       excluded.protected_at_generation
                   ),
                   updated_at = MAX(lcm_protected_sessions.updated_at, excluded.updated_at)"""
        )
        _migration_crash_boundary("v12_after_policy_protected_backfill")

    # Use lifecycle policy only when its owner agrees with the latest active
    # frontier. This both backfills v10 carry=0 rows and avoids reviving a stale
    # v11 policy owner after a later carry rollover.
    conn.execute(
        """INSERT INTO lcm_rollover_heads(
               conversation_id, current_session_id,
               last_finalized_session_id, carry_over_context,
               rollover_epoch, frontier_generation, created_at, updated_at
           )
           SELECT l.conversation_id, l.current_session_id,
                  l.last_finalized_session_id,
                  l.rollover_carry_over_context, 1, f.generation, ?, ?
           FROM lcm_lifecycle_state AS l
           JOIN lcm_active_frontiers AS f
             ON f.conversation_id = l.conversation_id
            AND f.session_id = l.current_session_id
            AND NOT EXISTS (
                SELECT 1 FROM lcm_active_frontiers AS newer
                WHERE newer.conversation_id = f.conversation_id
                  AND newer.generation > f.generation
            )
           WHERE l.current_session_id IS NOT NULL
             AND l.last_finalized_session_id IS NOT NULL
             AND l.rollover_carry_over_context IN (0, 1)
           ON CONFLICT(conversation_id) DO NOTHING""",
        (now, now),
    )
    _migration_crash_boundary("v12_after_lifecycle_head_backfill")
    # A policy may be the only durable lifecycle-independent evidence left.
    conn.execute(
        """INSERT INTO lcm_rollover_heads(
               conversation_id, current_session_id,
               last_finalized_session_id, carry_over_context,
               rollover_epoch, frontier_generation, created_at, updated_at
           )
           SELECT p.conversation_id, p.current_session_id,
                  p.finalized_session_id, 0, 1, f.generation,
                  p.created_at, p.updated_at
           FROM lcm_rollover_policies AS p
           JOIN lcm_active_frontiers AS f
             ON f.conversation_id = p.conversation_id
            AND f.session_id = p.current_session_id
            AND NOT EXISTS (
                SELECT 1 FROM lcm_active_frontiers AS newer
                WHERE newer.conversation_id = f.conversation_id
                  AND newer.generation > f.generation
            )
           WHERE p.carry_over_context = 0
           ON CONFLICT(conversation_id) DO NOTHING"""
    )
    _migration_crash_boundary("v12_after_policy_head_backfill")
    # v10 did not have the v11 table at all.
    conn.execute(
        """INSERT INTO lcm_protected_sessions(
               conversation_id, finalized_session_id,
               finalized_boundary_store_id, protected_at_generation,
               rollover_epoch, created_at, updated_at
           )
           SELECT h.conversation_id, h.last_finalized_session_id,
                  MAX(0, l.last_finalized_frontier_store_id),
                  h.frontier_generation, h.rollover_epoch, ?, ?
           FROM lcm_rollover_heads AS h
           JOIN lcm_lifecycle_state AS l USING(conversation_id)
           WHERE h.carry_over_context = 0
           ON CONFLICT(conversation_id, finalized_session_id) DO UPDATE SET
               finalized_boundary_store_id = MAX(
                   lcm_protected_sessions.finalized_boundary_store_id,
                   excluded.finalized_boundary_store_id
               ), updated_at = excluded.updated_at""",
        (now, now),
    )
    _migration_crash_boundary("v12_after_lifecycle_protected_backfill")


def ensure_rollover_v12_triggers(conn: sqlite3.Connection) -> None:
    """Enforce historical session provenance against legacy frontier SQL."""
    ensure_rollover_v12_tables(conn)
    _drop_schema_v11_cutoff_triggers(conn)
    source_tables_ready = all(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        for table in ("messages", "summary_nodes")
    )
    for operation, reference in (("insert", "NEW"), ("update", "NEW")):
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS lcm_protected_frontier_{operation}
            BEFORE {operation.upper()} ON lcm_active_frontiers
            WHEN EXISTS (
                SELECT 1 FROM lcm_protected_sessions AS p
                WHERE p.conversation_id = {reference}.conversation_id
                  AND p.finalized_session_id = {reference}.session_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'no-carry protected session blocks frontier ownership');
            END
            """
        )
        if source_tables_ready:
            conn.execute(
                f"""
            CREATE TRIGGER IF NOT EXISTS lcm_protected_item_{operation}
            BEFORE {operation.upper()} ON lcm_frontier_items
            WHEN EXISTS (
                SELECT 1 FROM lcm_protected_sessions AS p
                WHERE p.conversation_id = {reference}.conversation_id
                  AND (
                      ({reference}.kind = 'message' AND EXISTS (
                          SELECT 1 FROM messages AS m
                          WHERE m.store_id = {reference}.ref_id
                            AND m.conversation_id = {reference}.conversation_id
                            AND m.session_id = p.finalized_session_id
                      ))
                      OR ({reference}.kind = 'node' AND EXISTS (
                          WITH RECURSIVE closure(
                              node_id, session_id, source_type, source_ids
                          ) AS (
                              SELECT n.node_id, n.session_id,
                                     n.source_type, n.source_ids
                              FROM summary_nodes AS n
                              WHERE n.node_id = {reference}.ref_id
                              UNION
                              SELECT child.node_id, child.session_id,
                                     child.source_type, child.source_ids
                              FROM closure AS parent
                              JOIN json_each(parent.source_ids) AS edge
                                ON parent.source_type = 'nodes'
                              JOIN summary_nodes AS child
                                ON child.node_id = CAST(edge.value AS INTEGER)
                          )
                          SELECT 1 FROM closure AS c
                          WHERE c.session_id = p.finalized_session_id
                             OR (
                                 c.source_type = 'messages'
                                 AND EXISTS (
                                     SELECT 1
                                     FROM json_each(c.source_ids) AS source
                                     JOIN messages AS m
                                       ON m.store_id = CAST(source.value AS INTEGER)
                                     WHERE m.conversation_id = {reference}.conversation_id
                                       AND m.session_id = p.finalized_session_id
                                 )
                             )
                      ))
                      OR EXISTS (
                          SELECT 1 FROM messages AS covered
                          WHERE covered.conversation_id = {reference}.conversation_id
                            AND covered.session_id = p.finalized_session_id
                            AND covered.store_id BETWEEN {reference}.source_start
                                                     AND {reference}.source_end
                      )
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'no-carry protected session blocks frontier item provenance');
            END
                """
            )
        else:
            conn.execute(f"DROP TRIGGER IF EXISTS lcm_protected_item_{operation}")
    _migration_crash_boundary("v12_after_provenance_triggers")

    # Compatibility for a schema-v9/v10 process that was already open during
    # migration. Only a lifecycle rollover consistent with the newest frontier
    # may advance the head; same-generation rewrites are inert.
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_v12_legacy_lifecycle_rollover
        AFTER INSERT ON lcm_lifecycle_state
        WHEN NEW.current_session_id IS NOT NULL
          AND NEW.last_finalized_session_id IS NOT NULL
          AND NEW.rollover_carry_over_context IN (0, 1)
        BEGIN
            INSERT INTO lcm_rollover_heads(
                conversation_id, current_session_id,
                last_finalized_session_id, carry_over_context,
                rollover_epoch, frontier_generation, created_at, updated_at
            )
            SELECT NEW.conversation_id, NEW.current_session_id,
                   NEW.last_finalized_session_id,
                   NEW.rollover_carry_over_context, 1,
                   f.generation, NEW.updated_at, NEW.updated_at
            FROM lcm_active_frontiers AS f
            WHERE f.conversation_id = NEW.conversation_id
              AND f.session_id = NEW.current_session_id
              AND NOT EXISTS (
                  SELECT 1 FROM lcm_active_frontiers AS newer
                  WHERE newer.conversation_id = f.conversation_id
                    AND newer.generation > f.generation
              )
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                carry_over_context = excluded.carry_over_context,
                rollover_epoch = lcm_rollover_heads.rollover_epoch + 1,
                frontier_generation = excluded.frontier_generation,
                updated_at = excluded.updated_at
            WHERE excluded.frontier_generation > lcm_rollover_heads.frontier_generation;
            INSERT INTO lcm_protected_sessions(
                conversation_id, finalized_session_id,
                finalized_boundary_store_id, protected_at_generation,
                rollover_epoch, created_at, updated_at
            )
            SELECT NEW.conversation_id, NEW.last_finalized_session_id,
                   MAX(0, NEW.last_finalized_frontier_store_id),
                   h.frontier_generation, h.rollover_epoch,
                   NEW.updated_at, NEW.updated_at
            FROM lcm_rollover_heads AS h
            WHERE h.conversation_id = NEW.conversation_id
              AND h.current_session_id = NEW.current_session_id
              AND h.last_finalized_session_id = NEW.last_finalized_session_id
              AND h.carry_over_context = 0
            ON CONFLICT(conversation_id, finalized_session_id) DO UPDATE SET
                finalized_boundary_store_id = MAX(
                    lcm_protected_sessions.finalized_boundary_store_id,
                    excluded.finalized_boundary_store_id
                ), updated_at = excluded.updated_at;
        END
        """
    )
    _migration_crash_boundary("v12_after_legacy_insert_trigger")
    conn.execute("DROP TRIGGER IF EXISTS lcm_v12_legacy_lifecycle_rollover_update")
    conn.execute(
        """CREATE TRIGGER lcm_v12_legacy_lifecycle_rollover_update
           AFTER UPDATE OF current_session_id, last_finalized_session_id,
                           rollover_carry_over_context
           ON lcm_lifecycle_state
           WHEN NEW.current_session_id IS NOT NULL
             AND NEW.last_finalized_session_id IS NOT NULL
             AND NEW.rollover_carry_over_context IN (0, 1)
           BEGIN
             INSERT INTO lcm_rollover_heads(
                 conversation_id, current_session_id,
                 last_finalized_session_id, carry_over_context,
                 rollover_epoch, frontier_generation, created_at, updated_at
             )
             SELECT NEW.conversation_id, NEW.current_session_id,
                    NEW.last_finalized_session_id,
                    NEW.rollover_carry_over_context, 1,
                    f.generation, NEW.updated_at, NEW.updated_at
             FROM lcm_active_frontiers AS f
             WHERE f.conversation_id = NEW.conversation_id
               AND f.session_id = NEW.current_session_id
               AND NOT EXISTS (
                   SELECT 1 FROM lcm_active_frontiers AS newer
                   WHERE newer.conversation_id = f.conversation_id
                     AND newer.generation > f.generation
               )
             ON CONFLICT(conversation_id) DO UPDATE SET
                 current_session_id = excluded.current_session_id,
                 last_finalized_session_id = excluded.last_finalized_session_id,
                 carry_over_context = excluded.carry_over_context,
                 rollover_epoch = lcm_rollover_heads.rollover_epoch + 1,
                 frontier_generation = excluded.frontier_generation,
                 updated_at = excluded.updated_at
             WHERE excluded.frontier_generation > lcm_rollover_heads.frontier_generation;
             INSERT INTO lcm_protected_sessions(
                 conversation_id, finalized_session_id,
                 finalized_boundary_store_id, protected_at_generation,
                 rollover_epoch, created_at, updated_at
             )
             SELECT NEW.conversation_id, NEW.last_finalized_session_id,
                    MAX(0, NEW.last_finalized_frontier_store_id),
                    h.frontier_generation, h.rollover_epoch,
                    NEW.updated_at, NEW.updated_at
             FROM lcm_rollover_heads AS h
             WHERE h.conversation_id = NEW.conversation_id
               AND h.current_session_id = NEW.current_session_id
               AND h.last_finalized_session_id = NEW.last_finalized_session_id
               AND h.carry_over_context = 0
             ON CONFLICT(conversation_id, finalized_session_id) DO UPDATE SET
                 finalized_boundary_store_id = MAX(
                     lcm_protected_sessions.finalized_boundary_store_id,
                     excluded.finalized_boundary_store_id
                 ), updated_at = excluded.updated_at;
           END"""
    )
    _migration_crash_boundary("v12_after_legacy_update_trigger")
def ensure_message_origin_columns(conn: sqlite3.Connection) -> None:
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
    ).fetchone()
    if not table_row:
        return
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "conversation_id",
        "ALTER TABLE messages ADD COLUMN conversation_id TEXT DEFAULT ''",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_msg_conversation_session ON messages(conversation_id, session_id, store_id)"
    )


def ensure_content_scan_checkpoint_schema(
    conn: sqlite3.Connection, *, replace_legacy_triggers: bool = False
) -> None:
    """Install the v9 bounded-content paging schema.

    This function is called only while ``run_versioned_migrations`` owns its
    ``BEGIN IMMEDIATE`` transaction.  In particular, the DDL, triggers,
    migration-step marker, and schema-version publication become visible in one
    crash-atomic commit.  Legacy message revisions are deliberately created on
    first expansion instead of scanning the entire messages table at startup.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_content_revisions (
            store_id INTEGER PRIMARY KEY,
            content_fingerprint TEXT NOT NULL,
            content_chars INTEGER NOT NULL,
            content_bytes INTEGER NOT NULL DEFAULT 0,
            storage_version INTEGER NOT NULL DEFAULT 2,
            scan_byte_offset INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    revision_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_content_revisions)").fetchall()
    }
    add_column_if_missing(
        conn,
        revision_columns,
        "content_bytes",
        "ALTER TABLE lcm_content_revisions ADD COLUMN content_bytes INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn,
        revision_columns,
        "storage_version",
        "ALTER TABLE lcm_content_revisions ADD COLUMN storage_version INTEGER NOT NULL DEFAULT 1",
    )
    add_column_if_missing(
        conn,
        revision_columns,
        "scan_byte_offset",
        "ALTER TABLE lcm_content_revisions ADD COLUMN scan_byte_offset INTEGER NOT NULL DEFAULT 0",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_content_scan_checkpoints (
            store_id INTEGER NOT NULL,
            content_fingerprint TEXT NOT NULL,
            char_offset INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL,
            quote TEXT,
            quote_backslashes INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (store_id, content_fingerprint, char_offset)
        )
        """
    )
    checkpoint_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(lcm_content_scan_checkpoints)").fetchall()
    }
    add_column_if_missing(
        conn,
        checkpoint_columns,
        "byte_offset",
        "ALTER TABLE lcm_content_scan_checkpoints ADD COLUMN byte_offset INTEGER NOT NULL DEFAULT 0",
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_lcm_content_scan_checkpoint_lookup
           ON lcm_content_scan_checkpoints(
               store_id, content_fingerprint, char_offset DESC
           )"""
    )

    messages_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
    ).fetchone()
    if not messages_exists:
        return

    # Replace any pre-v9 unversioned triggers so all new revisions contain the
    # immutable character/byte lengths used by incremental BLOB paging.
    if replace_legacy_triggers:
        for trigger_name in (
            "lcm_content_revision_insert",
            "lcm_content_revision_update",
            "lcm_content_revision_delete",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_content_revision_insert
        AFTER INSERT ON messages BEGIN
            INSERT OR REPLACE INTO lcm_content_revisions
                (store_id, content_fingerprint, content_chars, content_bytes,
                 storage_version, scan_byte_offset)
            VALUES (
                new.store_id,
                lower(hex(randomblob(16))),
                COALESCE(length(CAST(new.content AS TEXT)), 0),
                COALESCE(length(CAST(new.content AS BLOB)), 0),
                2,
                COALESCE(length(CAST(new.content AS BLOB)), 0)
            );
            DELETE FROM lcm_content_scan_checkpoints WHERE store_id = new.store_id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_content_revision_update
        AFTER UPDATE OF content ON messages BEGIN
            INSERT OR REPLACE INTO lcm_content_revisions
                (store_id, content_fingerprint, content_chars, content_bytes,
                 storage_version, scan_byte_offset)
            VALUES (
                new.store_id,
                lower(hex(randomblob(16))),
                COALESCE(length(CAST(new.content AS TEXT)), 0),
                COALESCE(length(CAST(new.content AS BLOB)), 0),
                2,
                COALESCE(length(CAST(new.content AS BLOB)), 0)
            );
            DELETE FROM lcm_content_scan_checkpoints WHERE store_id = new.store_id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_content_revision_delete
        AFTER DELETE ON messages BEGIN
            DELETE FROM lcm_content_scan_checkpoints WHERE store_id = old.store_id;
            DELETE FROM lcm_content_revisions WHERE store_id = old.store_id;
        END
        """
    )


def ensure_frontier_tables(conn: sqlite3.Connection) -> None:
    """Create the persistent ordered active frontier tables (schema v6).

    Three tables implement the candidate-based async compaction workflow:

    - ``lcm_active_frontiers`` — one row per conversation per generation.
      Tracks the current canonical frontier (source boundary, policy
      fingerprint, generation counter).

    - ``lcm_frontier_items`` — ordered items in a frontier generation.
      Each item references either a raw message store_id or a DAG node_id,
      with an ordinal for deterministic ordering.

    - ``lcm_prepared_batches`` — candidate compaction results built
      off-context.  Each batch references the base generation it was
      prepared against, carries source identity hashes and policy/route
      fingerprints for CAS validation at promotion time, and transitions
      through states: preparing → ready → promoted | rejected | failed.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_active_frontiers (
            conversation_id TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 1,
            session_id TEXT NOT NULL,
            source_end_store_id INTEGER NOT NULL DEFAULT 0,
            policy_fingerprint TEXT NOT NULL DEFAULT '',
            route_fingerprint TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (conversation_id, generation)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_frontiers_conversation
            ON lcm_active_frontiers(conversation_id, generation DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_frontier_items (
            conversation_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'message',
            ref_id INTEGER NOT NULL,
            source_start INTEGER NOT NULL DEFAULT 0,
            source_end INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (conversation_id, generation, ordinal)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_frontier_items_gen
            ON lcm_frontier_items(conversation_id, generation, ordinal)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_prepared_batches (
            batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            base_generation INTEGER NOT NULL,
            source_end_store_id INTEGER NOT NULL,
            source_identity_hash TEXT NOT NULL DEFAULT '',
            source_ids TEXT NOT NULL DEFAULT '[]',
            policy_fingerprint TEXT NOT NULL DEFAULT '',
            route_fingerprint TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'preparing',
            expected_leaf_count INTEGER NOT NULL DEFAULT 0,
            frontier_end_store_id INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            failure_reason TEXT DEFAULT '',
            summary_payload TEXT NOT NULL DEFAULT '',
            payload_version INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_batches_conv_state
            ON lcm_prepared_batches(conversation_id, state)
        """
    )
    ensure_prepared_batch_payload_columns(conn)


def ensure_prepared_batch_payload_columns(conn: sqlite3.Connection) -> None:
    """Add prepared-summary payload columns (schema v7) when missing.

    ``payload_version`` 0/1 = legacy v1 batches without a persisted summary
    (must be rejected/superseded, never re-summarized at promote).
    ``payload_version`` 2 = prepare stored the full summary payload so promote
    is a pure publish path with zero LLM calls.
    """
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lcm_prepared_batches'"
    ).fetchone()
    if not table_row:
        return
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_prepared_batches)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "summary_payload",
        "ALTER TABLE lcm_prepared_batches ADD COLUMN summary_payload TEXT NOT NULL DEFAULT ''",
    )
    add_column_if_missing(
        conn, columns, "payload_version",
        "ALTER TABLE lcm_prepared_batches ADD COLUMN payload_version INTEGER NOT NULL DEFAULT 0",
    )


def ensure_focus_and_policy_metadata(conn: sqlite3.Connection) -> None:
    """Create schema-v8 immutable focus briefs and prepared-policy metadata."""
    ensure_frontier_tables(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_prepared_batches)").fetchall()
    }
    add_column_if_missing(
        conn,
        columns,
        "resolved_policy_json",
        "ALTER TABLE lcm_prepared_batches ADD COLUMN resolved_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_focus_briefs (
            focus_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            content TEXT NOT NULL,
            source_node_ids TEXT NOT NULL DEFAULT '[]',
            covered_generation INTEGER NOT NULL DEFAULT 0,
            covered_store_id INTEGER NOT NULL DEFAULT 0,
            token_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
            supersedes_focus_id INTEGER,
            FOREIGN KEY(supersedes_focus_id) REFERENCES lcm_focus_briefs(focus_id)
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_focus_one_active
        ON lcm_focus_briefs(conversation_id) WHERE active = 1
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_focus_conversation_history
        ON lcm_focus_briefs(conversation_id, focus_id DESC)
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS lcm_focus_briefs_immutable
        BEFORE UPDATE ON lcm_focus_briefs
        WHEN OLD.conversation_id != NEW.conversation_id
          OR OLD.session_id != NEW.session_id
          OR OLD.prompt != NEW.prompt
          OR OLD.content != NEW.content
          OR OLD.source_node_ids != NEW.source_node_ids
          OR OLD.covered_generation != NEW.covered_generation
          OR OLD.covered_store_id != NEW.covered_store_id
          OR OLD.token_count != NEW.token_count
          OR OLD.created_at != NEW.created_at
          OR COALESCE(OLD.supersedes_focus_id, 0) != COALESCE(NEW.supersedes_focus_id, 0)
        BEGIN
            SELECT RAISE(ABORT, 'focus briefs are immutable');
        END
        """
    )


def supersede_legacy_v1_ready_batches(conn: sqlite3.Connection) -> int:
    """Mark ready/preparing batches that lack a summary payload as superseded.

    v1 batches discarded the LLM result at prepare time and re-ran summarization
    at promote. After the v7 payload migration those rows must not silently
    re-summarize — supersede them so the next prepare builds a payload-bearing
    batch.
    """
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lcm_prepared_batches'"
    ).fetchone()
    if not table_row:
        return 0
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_prepared_batches)").fetchall()
    }
    if "payload_version" not in columns or "summary_payload" not in columns:
        return 0
    cur = conn.execute(
        """
        UPDATE lcm_prepared_batches
        SET state = 'superseded',
            failure_reason = 'legacy_v1_batch_without_payload',
            updated_at = ?
        WHERE state IN ('ready', 'preparing')
          AND (COALESCE(payload_version, 0) < 2
               OR COALESCE(summary_payload, '') = '')
        """,
        (time.time(),),
    )
    return int(cur.rowcount or 0)


def mark_migration_step_complete(conn: sqlite3.Connection, step_name: str) -> None:
    ensure_migration_state_table(conn)
    conn.execute(
        """
        INSERT INTO lcm_migration_state(step_name, completed_at)
        VALUES(?, strftime('%s','now'))
        ON CONFLICT(step_name) DO UPDATE SET completed_at = excluded.completed_at
        """,
        (step_name,),
    )


def set_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        WHERE CAST(metadata.value AS INTEGER) <= CAST(excluded.value AS INTEGER)
        """,
        (str(version),),
    )


def get_existing_table_names(conn: sqlite3.Connection, names: Iterable[str]) -> set[str]:
    existing: set[str] = set()
    for name in names:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row and row[0]:
            existing.add(row[0])
    return existing


def _database_path_for_connection(conn: sqlite3.Connection | None, fallback: str = "") -> str:
    if conn is None:
        return fallback
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.DatabaseError:
        return fallback
    for row in rows:
        if len(row) >= 3 and row[1] == "main" and row[2]:
            return str(row[2])
    return fallback


def inspect_lcm_schema_health(
    conn: sqlite3.Connection | None,
    *,
    database_path: str = "",
    required_tables: Iterable[str] = REQUIRED_CORE_TABLES,
) -> dict[str, object]:
    """Return read-only health metadata for the core hermes-lcm SQLite schema."""
    required = tuple(required_tables)
    resolved_path = _database_path_for_connection(conn, database_path)
    detail: dict[str, object] = {
        "database_path": resolved_path,
        "required_tables": list(required),
        "existing_tables": [],
        "missing_tables": [],
    }
    if conn is None:
        detail["error"] = "LCM store connection is not initialized"
        return detail

    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        detail["error"] = str(exc)
        return detail

    existing = sorted(str(row[0]) for row in rows if row and row[0])
    existing_set = set(existing)
    missing = [name for name in required if name not in existing_set]
    detail["existing_tables"] = existing
    detail["missing_tables"] = missing
    return detail


def get_fts_shadow_table_names(table_name: str) -> list[str]:
    return [
        f"{table_name}_data",
        f"{table_name}_idx",
        f"{table_name}_docsize",
        f"{table_name}_config",
    ]


def quote_sql_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "a").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def _fts_needs_rebuild_structural(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    shadow_tables = get_fts_shadow_table_names(spec.table_name)
    existing_tables = get_existing_table_names(conn, [spec.table_name, *shadow_tables])
    if spec.table_name not in existing_tables:
        return True
    if any(name not in existing_tables for name in shadow_tables):
        return True

    try:
        info = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (spec.table_name,),
        ).fetchone()
        sql = (info[0] if info else "") or ""
        normalized = sql.lower()
        if "virtual table" not in normalized or "using fts5" not in normalized:
            return True

        columns = conn.execute(
            f"PRAGMA table_info({quote_sql_identifier(spec.table_name)})"
        ).fetchall()
        column_names = {row[1] for row in columns if len(row) > 1}
        if spec.indexed_column not in column_names:
            return True

        content_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.content_table)}"
        ).fetchone()[0]
        # For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads
        # through to the content table (so it can never reveal a lagging index)
        # and is O(index size). The ``<fts>_docsize`` shadow table holds the
        # true indexed-document count and is a cheap ordinary-table count. Its
        # existence is already guaranteed by the shadow-table check above.
        docsize_table = f"{spec.table_name}_docsize"
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(docsize_table)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True
    except sqlite3.DatabaseError:
        return True

    return False


INTEGRITY_CHECK_INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS = 24.0


def _integrity_check_interval_hours() -> float:
    """Hours between startup FTS deep integrity-checks.

    ``0`` checks on every startup (previous behavior); a negative value never
    checks on startup (relies on structural checks + LIKE fallback + doctor).
    """
    raw = os.environ.get(INTEGRITY_CHECK_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    if not math.isfinite(value):
        # nan/inf would suppress startup checks indefinitely once a marker
        # exists; treat non-finite values as invalid.
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    return value


def _integrity_marker_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_checked_at:{spec.table_name}"


def _load_integrity_checked_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_marker_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_integrity_checked(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_marker_key(spec), str(current)),
    )


def _should_run_integrity_check(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    hours = _integrity_check_interval_hours()
    if hours == 0:
        return True
    if hours < 0:
        return False
    last = _load_integrity_checked_at(conn, spec)
    if last is None:
        return True
    current = time.time() if now is None else now
    return (current - last) >= hours * 3600.0


def _fts_needs_rebuild(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> bool:
    if _fts_needs_rebuild_structural(conn, spec):
        return True
    # Structurally sound: the FTS5 integrity-check is O(index size) and was the
    # dominant startup cost on large databases (issue #235). On the startup path
    # (``throttle=True``) skip it when already checked within the interval.
    # Explicit repair (e.g. ``/lcm doctor repair apply``) uses ``throttle=False``
    # so it always runs the deep check and can fix same-row-count drift that the
    # structural checks cannot see.
    if throttle and not _should_run_integrity_check(conn, spec, now=now):
        return False
    result = check_external_content_fts_integrity(conn, spec)
    if result["status"] == "pass":
        _record_integrity_checked(conn, spec, now=now)
    return result["status"] == "fail"


def check_external_content_fts_integrity(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
) -> dict[str, str]:
    """Run SQLite's FTS5 integrity-check for an external-content table.

    FTS5 exposes this as a special INSERT command. Wrap it in a savepoint and
    roll it back so diagnostics can verify the index without leaving any state
    behind on the shared connection.
    """

    if _fts_needs_rebuild_structural(conn, spec):
        return {"status": "fail", "detail": "structural repair needed"}

    savepoint = f"lcm_fts_integrity_{spec.table_name}"
    savepoint_sql = quote_sql_identifier(savepoint)
    try:
        conn.execute(f"SAVEPOINT {savepoint_sql}")
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint_sql}")
            conn.execute(f"RELEASE {savepoint_sql}")
        except sqlite3.DatabaseError:
            pass
        detail = str(exc)
        lowered = detail.lower()
        if "readonly" in lowered or "read-only" in lowered:
            return {"status": "unchecked", "detail": detail}
        return {"status": "fail", "detail": detail}

    try:
        conn.execute(f"ROLLBACK TO {savepoint_sql}")
        conn.execute(f"RELEASE {savepoint_sql}")
    except sqlite3.DatabaseError as exc:
        return {"status": "fail", "detail": str(exc)}

    return {"status": "pass", "detail": "ok"}


def _drop_fts_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(table_name)}")
    for shadow_name in get_fts_shadow_table_names(table_name):
        conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(shadow_name)}")


def _extract_trigger_name(trigger_sql: str) -> str | None:
    match = re.search(
        r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))",
        trigger_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


def _drop_fts_triggers(conn: sqlite3.Connection, trigger_sqls: Sequence[str]) -> None:
    for trigger_sql in trigger_sqls:
        trigger_name = _extract_trigger_name(trigger_sql)
        if trigger_name:
            conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(trigger_name)}")


def _drop_fts_artifacts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    _drop_fts_triggers(conn, spec.trigger_sqls)
    _drop_fts_table(conn, spec.table_name)


def _check_disk_space(db_path: str) -> bool:
    try:
        parent = os.path.dirname(os.path.abspath(db_path)) or "."
        return shutil.disk_usage(parent).free >= _MIN_DISK_SPACE_BYTES
    except (OSError, AttributeError):
        return True


def _fts_missing_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    expected = {
        trigger_name
        for trigger_name in (_extract_trigger_name(sql) for sql in spec.trigger_sqls)
        if trigger_name
    }
    if not expected:
        return False
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        tuple(sorted(expected)),
    ).fetchall()
    existing = {str(row[0]) for row in rows if row and row[0]}
    return bool(expected - existing)


def external_content_fts_needs_repair(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    return _fts_needs_rebuild_structural(conn, spec) or _fts_missing_triggers(conn, spec)


def repair_external_content_fts(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
    commit: bool = True,
) -> dict[str, bool]:
    rebuilt = False
    degraded = False
    if _fts_needs_rebuild(conn, spec, now=now, throttle=throttle):
        db_path = conn.execute("PRAGMA database_list").fetchone()
        if db_path:
            db_file = db_path[2]
            if db_file and not _check_disk_space(db_file):
                logger.warning(
                    "Low disk space for FTS rebuild of '%s' (%d MB needed), degrading to LIKE search",
                    spec.table_name,
                    _MIN_DISK_SPACE_BYTES // (1024 * 1024),
                )
                _drop_fts_artifacts(conn, spec)
                if commit:
                    conn.commit()
                return {"rebuilt": False, "degraded": True, "triggers_recreated": False}
        _drop_fts_table(conn, spec.table_name)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
                {quote_sql_identifier(spec.indexed_column)},
                content={quote_sql_identifier(spec.content_table)},
                content_rowid={quote_sql_identifier(spec.content_rowid)}
            )
            """
        )
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}) VALUES('rebuild')"
        )
        rebuilt = True

    triggers_were_missing = _fts_missing_triggers(conn, spec)
    for trigger_sql in spec.trigger_sqls:
        conn.execute(trigger_sql)
    if rebuilt:
        # A freshly rebuilt index is known-consistent; record the marker so the
        # next startup can skip the deep integrity-check within the interval.
        _record_integrity_checked(conn, spec, now=now)
    if commit:
        conn.commit()
    return {"rebuilt": rebuilt, "degraded": degraded, "triggers_recreated": triggers_were_missing}


def ensure_external_content_fts(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    # Startup path: throttle the deep integrity-check. Explicit repair callers
    # use ``repair_external_content_fts(..., throttle=False)`` for a forced check.
    repair_external_content_fts(conn, spec, now=now, throttle=True)


def run_versioned_migrations(
    conn: sqlite3.Connection,
    *,
    fts_specs: Sequence[ExternalContentFtsSpec] = (),
) -> None:
    # Preserve the refusal-before-DDL contract for databases already marked by
    # a newer build, then serialize the authoritative re-check, every migration
    # step, and the marker update behind one SQLite writer lock. Without this,
    # a base-version connection can cache the old marker while a newer writer
    # migrates, wait, and then overwrite the committed marker with its stale
    # version.
    refuse_schema_version_too_new(conn)
    if conn.in_transaction:
        raise RuntimeError("run_versioned_migrations requires no active transaction")
    try:
        conn.execute("BEGIN IMMEDIATE")
        refuse_schema_version_too_new(conn)
        ensure_metadata_table(conn)
        # Install while the migration owns the writer lock, before publishing
        # v8 and releasing older processes waiting to run unconditional marker
        # UPSERTs. The trigger is durable database policy, not caller policy.
        ensure_schema_version_monotonic_guard(conn)
        ensure_migration_state_table(conn)

        current_version = get_schema_version(conn)
        if current_version < 2:
            mark_migration_step_complete(conn, "v2_external_content_fts_triggers")
            current_version = 2

        if current_version < 3:
            ensure_lifecycle_state_table(conn)
            mark_migration_step_complete(conn, "v3_lifecycle_state")
            current_version = 3
        else:
            ensure_lifecycle_state_table(conn)

        ensure_lifecycle_state_columns(conn)
        if current_version < 4:
            mark_migration_step_complete(conn, "v4_lifecycle_debt_columns")
            current_version = 4

        ensure_message_origin_columns(conn)
        if current_version < 5:
            mark_migration_step_complete(conn, "v5_message_conversation_id")
            current_version = 5

        ensure_frontier_tables(conn)
        if current_version < 6:
            mark_migration_step_complete(conn, "v6_active_frontier_tables")
            current_version = 6

        if current_version < 7:
            supersede_legacy_v1_ready_batches(conn)
            mark_migration_step_complete(conn, "v7_prepared_batch_summary_payload")
            current_version = 7

        ensure_focus_and_policy_metadata(conn)
        if current_version < 8:
            mark_migration_step_complete(conn, "v8_focus_and_resolved_policy_metadata")
            current_version = 8

        ensure_content_scan_checkpoint_schema(
            conn, replace_legacy_triggers=current_version < 9
        )
        if current_version < 9:
            mark_migration_step_complete(conn, "v9_content_scan_checkpoints")
            current_version = 9

        if current_version < 10:
            mark_migration_step_complete(conn, "v10_rollover_carry_policy")
            current_version = 10

        migrating_to_v11 = current_version < 11
        if migrating_to_v11:
            # The v10 lifecycle policy column and every v11 object remain
            # inside this same writer transaction. Abrupt exits at any phase
            # therefore recover as wholly v10, never as a partial v11 schema.
            _migration_crash_boundary("v11_after_column")
            ensure_rollover_policy_table(conn)
            _migration_crash_boundary("v11_after_table")
            # v11's scalar triggers are intentionally not installed by v12;
            # the migration transaction proceeds directly to provenance-based
            # protection below.
            _migration_crash_boundary("v11_after_trigger")
            mark_migration_step_complete(conn, "v11_no_carry_frontier_policy")
            _migration_crash_boundary("v11_after_migration_step")
            current_version = 11
        else:
            ensure_rollover_policy_table(conn)

        migrating_to_v12 = current_version < 12
        if migrating_to_v12:
            ensure_rollover_v12_tables(conn)
            _migration_crash_boundary("v12_after_ddl")
            backfill_rollover_v12(conn)
            _migration_crash_boundary("v12_after_backfill")
            ensure_rollover_v12_triggers(conn)
            _migration_crash_boundary("v12_after_triggers")
            mark_migration_step_complete(
                conn, "v12_protected_sessions_heads_and_ingest_receipts"
            )
            _migration_crash_boundary("v12_after_migration_step")
            current_version = 12
        else:
            ensure_rollover_v12_tables(conn)
            ensure_rollover_v12_triggers(conn)

        # Startup FTS inspection, repair, rebuild, and trigger installation are
        # part of the same cross-connection writer transaction as schema
        # migration.  A concurrent opener therefore cannot observe v10, release
        # the migration lock, and race this connection's drop/create sequence.
        for spec in fts_specs:
            repair_external_content_fts(
                conn,
                spec,
                throttle=True,
                commit=False,
            )

        set_schema_version(conn, current_version)
        if migrating_to_v11:
            _migration_crash_boundary("v11_after_schema_version")
        if migrating_to_v12:
            _migration_crash_boundary("v12_after_schema_version")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
