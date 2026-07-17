"""Tests for FTS startup integrity-check throttling (issue #235).

The FTS5 ``integrity-check`` is O(index size) and was run unconditionally on
every startup where the index already exists and is structurally sound,
dominating launch time on large databases. These tests pin the throttled
behavior: the deep check runs at most once per configurable interval, while the
cheap structural checks always run.

Note on behavior model: a brand-new database takes the ``structural -> rebuild``
path and does NOT run integrity-check; the expensive check only fires on
subsequent startups of an existing, structurally-sound index. The tests build
the index first, then exercise the existing-index path.
"""

import sqlite3
import threading
import time
import types

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.db_bootstrap import (
    ExternalContentFtsSpec,
    ensure_external_content_fts,
)

INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
MARKER_KEY = "fts_integrity_checked_at:messages_fts"


def _make_conn(tmp_path, name="t.db"):
    conn = sqlite3.connect(str(tmp_path / name))
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        );
        INSERT INTO messages(content) VALUES ('hello world');
        INSERT INTO messages(content) VALUES ('second searchable message');
        """
    )
    return conn


def _spec():
    return ExternalContentFtsSpec(
        table_name="messages_fts",
        content_table="messages",
        content_rowid="store_id",
        indexed_column="content",
        trigger_sqls=(),
    )


def _make_future_schema_db(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(db_bootstrap.SCHEMA_VERSION + 1),),
        )
        conn.commit()
    finally:
        conn.close()


def _journal_mode(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def _table_names(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def _reconstruct_v9_database(db_path):
    """Create a structurally genuine v9 lifecycle schema from current DDL."""
    conn = sqlite3.connect(str(db_path))
    try:
        db_bootstrap.run_versioned_migrations(conn)
        conn.execute(
            """INSERT INTO lcm_lifecycle_state(
                   conversation_id, current_session_id, current_frontier_store_id
               ) VALUES('legacy-conversation', 'legacy-session', 17)"""
        )
        conn.execute("DROP TRIGGER lcm_schema_version_monotonic")
        for (trigger,) in conn.execute(
            """SELECT name FROM sqlite_master WHERE type='trigger'
               AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%')"""
        ).fetchall():
            conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        for table in (
            "lcm_session_end_receipts",
            "lcm_rollover_heads",
            "lcm_protected_sessions",
        ):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state DROP COLUMN rollover_carry_over_context"
        )
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state DROP COLUMN binding_generation"
        )
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name = 'v10_rollover_carry_policy'"
        )
        for trigger in (
            "lcm_no_carry_frontier_insert",
            "lcm_no_carry_frontier_update",
            "lcm_no_carry_item_insert",
            "lcm_no_carry_item_update",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE IF EXISTS lcm_rollover_policies")
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name = 'v11_no_carry_frontier_policy'"
        )
        conn.execute(
            """DELETE FROM lcm_migration_state
               WHERE step_name = 'v12_protected_sessions_heads_and_ingest_receipts'"""
        )
        conn.execute(
            "UPDATE metadata SET value = '9' WHERE key = 'schema_version'"
        )
        db_bootstrap.ensure_schema_version_monotonic_guard(conn)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def integrity_calls(monkeypatch):
    """Spy that counts real integrity-check invocations by table name."""
    calls = []
    real = db_bootstrap.check_external_content_fts_integrity

    def spy(conn, spec):
        calls.append(spec.table_name)
        return real(conn, spec)

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", spy)
    return calls


def _marker(conn):
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", (MARKER_KEY,)
    ).fetchone()
    return row[0] if row else None


def test_existing_index_without_marker_runs_check_and_records_marker(tmp_path, integrity_calls):
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # builds index (rebuild path)
    # Simulate an existing DB upgraded to the throttling version: no marker yet.
    conn.execute("DELETE FROM metadata WHERE key = ?", (MARKER_KEY,))
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts"]
    assert _marker(conn) is not None
    conn.close()


def test_fresh_marker_skips_integrity_check(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build records a fresh marker
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == []  # fresh marker -> deep check skipped
    conn.close()


def test_expired_marker_reruns_integrity_check(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())
    # Age the marker well past the 24h interval.
    conn.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (str(time.time() - 100 * 3600), MARKER_KEY),
    )
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts"]
    conn.close()


def test_interval_zero_checks_every_init(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "0")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())
    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts", "messages_fts"]
    conn.close()


def test_negative_interval_never_checks_on_startup(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == []
    conn.close()


def test_structural_mismatch_rebuilds_despite_fresh_marker(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker, index has 2 docs

    # Insert a row without a trigger (spec has none): the FTS index now lags
    # content. Marker is fresh, so the deep integrity-check is throttled, but
    # the structural check must still detect the desync and rebuild.
    conn.execute("INSERT INTO messages(content) VALUES ('untracked row')")
    integrity_calls.clear()

    ensure_external_content_fts(conn, spec)

    assert integrity_calls == []  # repaired via structural path, not deep check
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False
    conn.close()


def test_external_content_desync_detected_via_docsize(tmp_path):
    """Content-vs-index row-count comparison must detect real desync.

    For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads through to
    the content table and cannot reveal a lagging index; ``<fts>_docsize`` holds
    the true indexed-document count. This guards the switch to docsize.
    """
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False

    # Insert without a trigger: indexed doc count (2) now lags content (3).
    conn.execute("INSERT INTO messages(content) VALUES ('untracked row')")
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is True
    conn.close()


def test_explicit_repair_fixes_same_count_corruption_despite_fresh_marker(tmp_path, monkeypatch):
    """`/lcm doctor repair apply` must deep-check/repair regardless of throttle.

    Regression for review on PR #236: the startup throttle must not leak into
    the explicit repair path. Same-row-count stale drift passes structural
    checks but fails the FTS5 integrity-check; with a fresh marker the throttle
    would otherwise skip the repair entirely.
    """
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker (startup path)

    # Content changes but the index does not (spec has no update trigger): the
    # row count is unchanged, so structural checks pass, but the indexed tokens
    # are stale and the integrity-check fails.
    conn.execute(
        "UPDATE messages SET content = 'completely different searchable text' WHERE store_id = 1"
    )
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False
    assert db_bootstrap.check_external_content_fts_integrity(conn, spec)["status"] == "fail"

    # Explicit repair (doctor path) is unthrottled and must rebuild + fix it.
    repaired = db_bootstrap.repair_external_content_fts(conn, spec)
    assert repaired["rebuilt"] is True
    assert db_bootstrap.check_external_content_fts_integrity(conn, spec)["status"] == "pass"
    conn.close()


def test_startup_throttle_still_skips_explicitly(tmp_path, monkeypatch, integrity_calls):
    """The throttle remains available on the startup path via throttle=True."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker
    integrity_calls.clear()

    db_bootstrap.repair_external_content_fts(conn, spec, throttle=True)

    assert integrity_calls == []  # fresh marker -> throttled path skips deep check
    conn.close()


def test_non_finite_interval_falls_back_to_default(monkeypatch):
    """nan/inf must not parse as a valid interval (would suppress checks forever)."""
    for value in ("nan", "inf", "-inf", "Infinity"):
        monkeypatch.setenv(INTERVAL_ENV, value)
        assert (
            db_bootstrap._integrity_check_interval_hours()
            == db_bootstrap.DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
        )


def test_check_disk_space_uses_portable_fallback_when_statvfs_is_unavailable(monkeypatch, tmp_path):
    """Windows lacks os.statvfs, so startup FTS repair must not crash there."""
    monkeypatch.delattr(db_bootstrap.os, "statvfs", raising=False)
    monkeypatch.setattr(
        db_bootstrap,
        "shutil",
        types.SimpleNamespace(
            disk_usage=lambda path: types.SimpleNamespace(
                free=db_bootstrap._MIN_DISK_SPACE_BYTES
            )
        ),
        raising=False,
    )

    assert db_bootstrap._check_disk_space(str(tmp_path / "lcm.db")) is True


def test_run_versioned_migrations_refuses_newer_schema_before_migration_state_ddl(tmp_path):
    conn = sqlite3.connect(tmp_path / "future-no-ddl.db")
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(db_bootstrap.SCHEMA_VERSION + 1),),
        )
        conn.commit()

        with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
            db_bootstrap.run_versioned_migrations(conn)

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert tables == {"metadata"}
    finally:
        conn.close()


def test_run_versioned_migrations_refuses_newer_schema(tmp_path):
    from hermes_lcm.db_bootstrap import (
        SchemaVersionTooNewError,
        ensure_metadata_table,
        run_versioned_migrations,
    )

    conn = sqlite3.connect(tmp_path / "future.db")
    try:
        ensure_metadata_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '99')"
        )
        conn.commit()
        with pytest.raises(SchemaVersionTooNewError):
            run_versioned_migrations(conn)
    finally:
        conn.close()


def test_run_versioned_migrations_accepts_current_schema(tmp_path):
    from hermes_lcm.db_bootstrap import run_versioned_migrations, get_schema_version, SCHEMA_VERSION

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        run_versioned_migrations(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_fresh_database_is_schema_v13_with_durable_rollover_state(tmp_path):
    conn = sqlite3.connect(tmp_path / "fresh-v11.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)

        assert db_bootstrap.SCHEMA_VERSION == 13
        assert db_bootstrap.get_schema_version(conn) == 13
        lifecycle_columns = {
            row[1]: row
            for row in conn.execute(
                "PRAGMA table_info(lcm_lifecycle_state)"
            ).fetchall()
        }
        assert "rollover_carry_over_context" in lifecycle_columns
        assert "binding_generation" in lifecycle_columns
        assert lifecycle_columns["rollover_carry_over_context"][4] is None
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v10_rollover_carry_policy'"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v11_no_carry_frontier_policy'"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_policies'"
        ).fetchone() == (1,)
        conn.execute(
            "INSERT INTO lcm_lifecycle_state(conversation_id) VALUES('fresh-v10')"
        )
        assert conn.execute(
            "SELECT rollover_carry_over_context FROM lcm_lifecycle_state WHERE conversation_id = 'fresh-v10'"
        ).fetchone() == (None,)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO lcm_lifecycle_state(
                       conversation_id, rollover_carry_over_context
                   ) VALUES('invalid-v10', 2)"""
            )
    finally:
        conn.close()


def test_v9_rollover_carry_policy_migrates_to_v11_and_restarts_idempotently(tmp_path):
    db_path = tmp_path / "v9-to-v11.db"
    _reconstruct_v9_database(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert db_bootstrap.get_schema_version(conn) == 9
        assert "rollover_carry_over_context" not in {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(lcm_lifecycle_state)"
            ).fetchall()
        }

        db_bootstrap.run_versioned_migrations(conn)

        assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION == 13
        assert "rollover_carry_over_context" in {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(lcm_lifecycle_state)"
            ).fetchall()
        }
        assert conn.execute(
            """SELECT current_session_id, current_frontier_store_id,
                      rollover_carry_over_context
               FROM lcm_lifecycle_state
               WHERE conversation_id = 'legacy-conversation'"""
        ).fetchone() == ("legacy-session", 17, None)
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v10_rollover_carry_policy'"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v11_no_carry_frontier_policy'"
        ).fetchone() == (1,)
    finally:
        conn.close()

    restarted = sqlite3.connect(db_path)
    try:
        db_bootstrap.run_versioned_migrations(restarted)
        assert db_bootstrap.get_schema_version(restarted) == 13
        assert restarted.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert restarted.execute(
            "SELECT COUNT(*) FROM lcm_lifecycle_state WHERE conversation_id = 'legacy-conversation'"
        ).fetchone() == (1,)
    finally:
        restarted.close()


def test_migration_serializes_version_read_and_prevents_marker_downgrade(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "serialized-schema-migration.db"
    setup = sqlite3.connect(db_path)
    setup.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    setup.execute(
        "INSERT INTO metadata(key, value) VALUES ('schema_version', '7')"
    )
    setup.commit()
    setup.close()

    migrator = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    older = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    db_bootstrap.configure_connection(migrator)
    db_bootstrap.configure_connection(older)
    version_read_started = threading.Event()
    allow_migration = threading.Event()
    original_get_schema_version = db_bootstrap.get_schema_version

    def pause_after_version_lock(conn):
        value = original_get_schema_version(conn)
        if conn is migrator:
            version_read_started.set()
            assert allow_migration.wait(timeout=5.0)
        return value

    monkeypatch.setattr(
        db_bootstrap,
        "get_schema_version",
        pause_after_version_lock,
    )
    outcomes: dict[str, object] = {}

    def migrate():
        try:
            db_bootstrap.run_versioned_migrations(migrator)
        except Exception as exc:
            outcomes["migration_error"] = exc

    def stale_v7_marker_write():
        try:
            db_bootstrap.set_schema_version(older, 7)
            older.commit()
            outcomes["older_finished"] = True
        except Exception as exc:
            outcomes["older_error"] = exc

    migration_thread = threading.Thread(target=migrate)
    older_thread = threading.Thread(target=stale_v7_marker_write)
    try:
        migration_thread.start()
        assert version_read_started.wait(timeout=5.0)
        older_thread.start()
        time.sleep(0.1)
        assert older_thread.is_alive(), (
            "schema marker writer must wait until the version check and migration commit"
        )

        allow_migration.set()
        migration_thread.join(timeout=5.0)
        older_thread.join(timeout=5.0)

        assert not migration_thread.is_alive()
        assert not older_thread.is_alive()
        assert "migration_error" not in outcomes
        assert "older_error" not in outcomes
        check = sqlite3.connect(db_path)
        try:
            assert db_bootstrap.get_schema_version(check) == db_bootstrap.SCHEMA_VERSION
        finally:
            check.close()
    finally:
        allow_migration.set()
        migration_thread.join(timeout=5.0)
        older_thread.join(timeout=5.0)
        migrator.close()
        older.close()


def test_v8_migration_blocks_base_v7_unconditional_schema_upsert(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "mixed-v7-v8-schema-migration.db"
    setup = sqlite3.connect(db_path)
    setup.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    setup.execute(
        "INSERT INTO metadata(key, value) VALUES ('schema_version', '7')"
    )
    setup.commit()
    setup.close()

    migrator = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    base_v7 = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    db_bootstrap.configure_connection(migrator)
    db_bootstrap.configure_connection(base_v7)
    cached_v7 = base_v7.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert cached_v7 == "7"

    migration_holds_writer_lock = threading.Event()
    allow_migration = threading.Event()
    original_get_schema_version = db_bootstrap.get_schema_version

    def pause_v8_after_locked_version_read(conn):
        version = original_get_schema_version(conn)
        if conn is migrator:
            migration_holds_writer_lock.set()
            assert allow_migration.wait(timeout=5.0)
        return version

    monkeypatch.setattr(
        db_bootstrap,
        "get_schema_version",
        pause_v8_after_locked_version_read,
    )
    outcomes: dict[str, object] = {}

    def migrate_to_v8():
        try:
            db_bootstrap.run_versioned_migrations(migrator)
        except Exception as exc:
            outcomes["migration_error"] = exc

    def execute_base_v7_cached_marker_write():
        try:
            # This is the unconditional SQL shipped by the base-v7 process,
            # deliberately not the current guarded set_schema_version().
            base_v7.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (cached_v7,),
            )
            base_v7.commit()
            outcomes["base_v7_finished"] = True
        except Exception as exc:
            outcomes["base_v7_error"] = exc

    migration_thread = threading.Thread(target=migrate_to_v8)
    base_v7_thread = threading.Thread(target=execute_base_v7_cached_marker_write)
    try:
        migration_thread.start()
        assert migration_holds_writer_lock.wait(timeout=5.0)
        base_v7_thread.start()
        time.sleep(0.1)
        assert base_v7_thread.is_alive(), (
            "base-v7 UPSERT must wait behind the v8 migration writer lock"
        )

        allow_migration.set()
        migration_thread.join(timeout=5.0)
        base_v7_thread.join(timeout=5.0)

        assert not migration_thread.is_alive()
        assert not base_v7_thread.is_alive()
        assert "migration_error" not in outcomes
        assert "base_v7_error" not in outcomes
        assert outcomes.get("base_v7_finished") is True

        check = sqlite3.connect(db_path)
        try:
            assert db_bootstrap.get_schema_version(check) == db_bootstrap.SCHEMA_VERSION
            focus_table = check.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'lcm_focus_briefs'"
            ).fetchone()
            assert focus_table is not None
            prepared_columns = {
                row[1]
                for row in check.execute(
                    "PRAGMA table_info(lcm_prepared_batches)"
                ).fetchall()
            }
            assert "resolved_policy_json" in prepared_columns
        finally:
            check.close()
    finally:
        allow_migration.set()
        migration_thread.join(timeout=5.0)
        base_v7_thread.join(timeout=5.0)
        migrator.close()
        base_v7.close()


def test_v11_migration_blocks_concurrent_base_v9_schema_downgrade(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "mixed-v9-v10-schema-migration.db"
    _reconstruct_v9_database(db_path)

    migrator = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    base_v9 = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    db_bootstrap.configure_connection(migrator)
    db_bootstrap.configure_connection(base_v9)
    cached_v9 = base_v9.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert cached_v9 == "9"

    migration_holds_writer_lock = threading.Event()
    allow_migration = threading.Event()
    original_get_schema_version = db_bootstrap.get_schema_version

    def pause_v10_after_locked_version_read(conn):
        version = original_get_schema_version(conn)
        if conn is migrator:
            migration_holds_writer_lock.set()
            assert allow_migration.wait(timeout=5.0)
        return version

    monkeypatch.setattr(
        db_bootstrap,
        "get_schema_version",
        pause_v10_after_locked_version_read,
    )
    outcomes: dict[str, object] = {}

    def migrate_to_v10():
        try:
            db_bootstrap.run_versioned_migrations(migrator)
        except Exception as exc:
            outcomes["migration_error"] = exc

    def execute_base_v9_cached_marker_write():
        try:
            base_v9.execute(
                """INSERT INTO metadata(key, value)
                   VALUES('schema_version', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (cached_v9,),
            )
            base_v9.commit()
            outcomes["base_v9_version"] = base_v9.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        except Exception as exc:
            outcomes["base_v9_error"] = exc

    migration_thread = threading.Thread(target=migrate_to_v10)
    base_v9_thread = threading.Thread(target=execute_base_v9_cached_marker_write)
    try:
        migration_thread.start()
        assert migration_holds_writer_lock.wait(timeout=5.0)
        base_v9_thread.start()
        time.sleep(0.1)
        assert base_v9_thread.is_alive(), (
            "base-v9 marker write must wait behind the v10 migration lock"
        )

        allow_migration.set()
        migration_thread.join(timeout=5.0)
        base_v9_thread.join(timeout=5.0)

        assert not migration_thread.is_alive()
        assert not base_v9_thread.is_alive()
        assert "migration_error" not in outcomes
        assert "base_v9_error" not in outcomes
        assert outcomes["base_v9_version"] == "13"
        check = sqlite3.connect(db_path)
        try:
            assert db_bootstrap.get_schema_version(check) == 13
            assert "rollover_carry_over_context" in {
                row[1]
                for row in check.execute(
                    "PRAGMA table_info(lcm_lifecycle_state)"
                ).fetchall()
            }
            assert check.execute("PRAGMA quick_check").fetchone() == ("ok",)
        finally:
            check.close()
    finally:
        allow_migration.set()
        migration_thread.join(timeout=5.0)
        base_v9_thread.join(timeout=5.0)
        migrator.close()
        base_v9.close()


def test_message_store_refuses_newer_schema_before_startup_ddl(tmp_path):
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "newer-message.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        MessageStore(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}


def test_summary_dag_refuses_newer_schema_before_startup_ddl(tmp_path):
    from hermes_lcm.dag import SummaryDAG

    db_path = tmp_path / "newer-dag.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        SummaryDAG(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}


def test_lifecycle_state_store_refuses_newer_schema_before_writable_pragmas_or_ddl(tmp_path):
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    db_path = tmp_path / "newer-lifecycle.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        LifecycleStateStore(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}

def test_message_store_refuses_newer_schema_before_configuring_connection(tmp_path, monkeypatch):
    from hermes_lcm.store import MessageStore
    import hermes_lcm.store as store_module

    db_path = tmp_path / "newer-before-pragmas.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    called = False

    def fail_if_called(conn):
        nonlocal called
        called = True
        raise AssertionError("configure_connection should not run for future schemas")

    monkeypatch.setattr(store_module, "configure_connection", fail_if_called)

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        MessageStore(db_path)
    assert called is False
