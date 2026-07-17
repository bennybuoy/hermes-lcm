"""Regression tests for the five publication/safety blockers at 27d1f5d."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sqlite3
import threading
import time

import pytest

import hermes_lcm.db_bootstrap as db_bootstrap
import hermes_lcm.store as store_module
import hermes_lcm.tools as tools_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _crash_during_v9_schema_publication(db_path: str) -> None:
    """Child-process crash hook used to verify SQLite DDL rollback."""
    conn = sqlite3.connect(db_path)
    db_bootstrap.configure_connection(conn)

    def create_partial_schema_then_crash(connection, **_kwargs):
        connection.execute(
            "CREATE TABLE lcm_content_revisions (store_id INTEGER PRIMARY KEY)"
        )
        os._exit(77)

    db_bootstrap.ensure_content_scan_checkpoint_schema = create_partial_schema_then_crash
    db_bootstrap.run_versioned_migrations(conn)


def _open_message_store_process(db_path: str, start_gate, outcomes) -> None:
    """Open one independent MessageStore connection after a process barrier."""
    try:
        start_gate.wait(timeout=10)
        store = store_module.MessageStore(db_path)
        try:
            version = store._conn.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()[0]
            outcomes.put(("ok", version))
        finally:
            store.close()
    except BaseException as exc:  # noqa: BLE001 - reported to parent process
        outcomes.put(("error", type(exc).__name__, str(exc)))


def _engine(tmp_path, **overrides) -> LCMEngine:
    values = {
        "database_path": str(tmp_path / "open-issues.db"),
        "large_output_externalization_enabled": True,
        "large_output_externalization_path": str(tmp_path / "payloads"),
    }
    values.update(overrides)
    engine = LCMEngine(config=LCMConfig(**values), hermes_home=str(tmp_path))
    engine.on_session_start(
        "current", conversation_id="conversation", platform="test", context_length=100_000
    )
    return engine


def test_expand_uses_bounded_blob_reads_progress_budget_and_restores_handler(
    tmp_path, monkeypatch
):
    progress_installs: list[tuple[object, int]] = []

    class TrackingConnection(store_module._LCMSQLiteConnection):
        def set_progress_handler(self, callback, n):
            progress_installs.append((callback, n))
            return super().set_progress_handler(callback, n)

    real_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(store_module.sqlite3, "connect", tracked_connect)
    engine = _engine(tmp_path)
    content = ("utf8-π-bounded-record\n" * 80_000) + "TAIL"
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    prior_calls = 0

    def prior_handler():
        nonlocal prior_calls
        prior_calls += 1
        return 0

    engine._store._conn.set_progress_handler(prior_handler, 7)
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        result = json.loads(
            tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": 1_000_000, "max_tokens": 64},
                engine=engine,
            )
        )
        assert result["store_id"] == store_id
        expansion_sql = "\n".join(statements).lower()
        assert "substr(cast(content" not in expansion_sql
        assert "length(cast(content as text))" not in expansion_sql
        assert len(progress_installs) >= 3  # caller, temporary budget, restoration

        before = prior_calls
        engine._store._conn.execute(
            "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<500) "
            "SELECT sum(x) FROM n"
        ).fetchone()
        assert prior_calls > before
    finally:
        engine._store._conn.set_trace_callback(None)
        engine._store._conn.set_progress_handler(None, 0)
        engine.shutdown()


def test_externalized_grep_streams_body_in_bounded_reads_and_honors_body_deadline(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    path = payload_dir / "bounded.json"
    path.write_text(
        json.dumps(
            {
                "session_id": "current",
                "created_at": 1.0,
                "content": ("ordinary body line\n" * 40_000) + "deadline needle",
            }
        ),
        encoding="utf-8",
    )
    engine = _engine(tmp_path)
    real_open = tools_module.Path.open
    largest_read = 0
    body_bytes_read = 0
    deadline_mode = False
    virtual_now = 0.0

    class BoundedReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal body_bytes_read, largest_read, virtual_now
            largest_read = max(largest_read, size)
            assert 0 <= size <= 16 * 1024, "external body was read in one large operation"
            chunk = self._handle.read(size)
            if size > 1:
                body_bytes_read += len(chunk)
                if deadline_mode:
                    virtual_now += 0.3
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def bounded_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        if candidate.resolve() == path.resolve() and "b" in str(
            args[0] if args else kwargs.get("mode", "r")
        ):
            return BoundedReader(handle)
        return handle

    monkeypatch.setattr(tools_module.Path, "open", bounded_open)
    try:
        result = json.loads(
            tools_module.lcm_grep(
                {
                    "query": "deadline needle",
                    "content_scope": "externalized",
                    "ref": path.name,
                    "max_payload_chars": 1_000_000,
                },
                engine=engine,
            )
        )
        assert result["total_results"] == 1
        assert 1 < largest_read <= 16 * 1024

        deadline_mode = True
        virtual_now = 0.0
        body_bytes_before = body_bytes_read
        monkeypatch.setattr(tools_module.time, "monotonic", lambda: virtual_now)
        expired = json.loads(
            tools_module.lcm_grep(
                {
                    "query": "deadline needle",
                    "content_scope": "externalized",
                    "ref": path.name,
                    "max_payload_chars": 1_000_000,
                },
                engine=engine,
            )
        )
        assert expired["total_results"] == 0
        assert expired["scan"]["scan_truncated"] is True
        assert body_bytes_read - body_bytes_before < path.stat().st_size
    finally:
        engine.shutdown()


def test_checkpoint_schema_is_v9_atomic_lazy_and_concurrent(tmp_path, monkeypatch):
    assert db_bootstrap.SCHEMA_VERSION >= 9
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            conversation_id TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata(key, value) VALUES('schema_version', '8');
        """
    )
    legacy.executemany(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES('s','user',?,1)",
        [("legacy" * 1000,) for _ in range(200)],
    )
    legacy.commit()
    legacy.close()

    statements: list[str] = []
    real_connect = store_module.sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(store_module.sqlite3, "connect", traced_connect)
    rebuild_gate = threading.Barrier(2)
    rebuild_checks: list[int] = []
    real_needs_rebuild = db_bootstrap._fts_needs_rebuild

    def synchronize_missing_fts(connection, spec, **kwargs):
        needed = real_needs_rebuild(connection, spec, **kwargs)
        if needed and spec.table_name == "messages_fts":
            rebuild_checks.append(id(connection))
            try:
                rebuild_gate.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
        return needed

    monkeypatch.setattr(
        db_bootstrap, "_fts_needs_rebuild", synchronize_missing_fts
    )
    stores = []
    errors = []

    def open_store():
        try:
            stores.append(store_module.MessageStore(db_path))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    try:
        assert not errors
        assert len(stores) == 2
        assert len(rebuild_checks) == 1
        sql = "\n".join(statements).lower()
        assert "from messages\n               where store_id not in" not in sql
        conn = stores[0]._conn
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == str(db_bootstrap.SCHEMA_VERSION)
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='v9_content_scan_checkpoints'"
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM lcm_content_revisions").fetchone()[0] == 0
    finally:
        for store in stores:
            store.close()


def test_message_store_repeated_concurrent_process_startup(tmp_path):
    """Independent startup losers wait through WAL negotiation and migration."""
    ctx = multiprocessing.get_context("fork")
    process_count = 4

    for iteration in range(5):
        db_path = tmp_path / f"concurrent-process-{iteration}.db"
        legacy = sqlite3.connect(db_path)
        legacy.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source TEXT DEFAULT '',
                conversation_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO metadata(key, value) VALUES('schema_version', '8');
            INSERT INTO messages(session_id, role, content, timestamp)
            VALUES('s', 'user', 'concurrent process FTS rebuild', 1);
            """
        )
        legacy.commit()
        legacy.close()

        start_gate = ctx.Barrier(process_count)
        outcomes = ctx.Queue()
        processes = [
            ctx.Process(
                target=_open_message_store_process,
                args=(str(db_path), start_gate, outcomes),
            )
            for _ in range(process_count)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)

            assert all(not process.is_alive() for process in processes)
            assert all(process.exitcode == 0 for process in processes)
            try:
                results = [outcomes.get(timeout=2) for _ in processes]
            except queue.Empty:  # pragma: no cover - explicit regression failure
                pytest.fail("concurrent startup process exited without an outcome")
            assert results == [
                ("ok", str(db_bootstrap.SCHEMA_VERSION))
            ] * process_count
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
            outcomes.close()
            outcomes.join_thread()


def test_message_store_startup_does_not_leak_raw_operational_error(
    tmp_path, monkeypatch
):
    def fail_configuration(_connection):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store_module, "configure_connection", fail_configuration)
    with pytest.raises(store_module.MessageStoreStartupError) as caught:
        store_module.MessageStore(tmp_path / "locked-startup.db")

    assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    assert "database is locked" in str(caught.value)


def test_sqlite_startup_pragma_lock_retries_with_backoff(monkeypatch):
    statements: list[str] = []
    sleeps: list[float] = []
    journal_attempts = 0

    class FlakyConnection:
        def execute(self, sql):
            nonlocal journal_attempts
            statements.append(sql)
            if sql == "PRAGMA journal_mode=WAL":
                journal_attempts += 1
                if journal_attempts < 3:
                    raise sqlite3.OperationalError("database is locked")
            return self

    monkeypatch.setattr(db_bootstrap.time, "sleep", sleeps.append)
    db_bootstrap.configure_connection(FlakyConnection())

    assert journal_attempts == 3
    assert sleeps == [
        db_bootstrap.SQLITE_STARTUP_BACKOFF_INITIAL_SECONDS,
        db_bootstrap.SQLITE_STARTUP_BACKOFF_INITIAL_SECONDS * 2,
    ]
    assert statements[-1] == "PRAGMA mmap_size=268435456"


def test_checkpoint_schema_and_marker_survive_process_crash_atomically(tmp_path):
    db_path = tmp_path / "crash-v9.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata(key, value) VALUES('schema_version', '8');
        """
    )
    conn.commit()
    conn.close()

    process = multiprocessing.get_context("fork").Process(
        target=_crash_during_v9_schema_publication, args=(str(db_path),)
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 77

    check = sqlite3.connect(db_path)
    try:
        assert check.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "8"
        assert check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_content_revisions'"
        ).fetchone() is None
        assert check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_content_scan_checkpoints'"
        ).fetchone() is None
    finally:
        check.close()

    recovered = sqlite3.connect(db_path)
    try:
        db_bootstrap.configure_connection(recovered)
        db_bootstrap.run_versioned_migrations(recovered)
        assert db_bootstrap.get_schema_version(recovered) == db_bootstrap.SCHEMA_VERSION
        assert recovered.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='v9_content_scan_checkpoints'"
        ).fetchone()
        assert recovered.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_content_scan_checkpoints'"
        ).fetchone()
    finally:
        recovered.close()


def test_incremental_blob_paging_preserves_multibyte_utf8_losslessly(tmp_path):
    engine = _engine(tmp_path)
    content = ("π🙂漢字 bounded utf8 page\n" * 20_000) + "UTF8-TAIL"
    store_id = engine._store.append("current", {"role": "user", "content": "placeholder"})
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    offset = 0
    pages: list[str] = []
    try:
        for _ in range(20):
            page = json.loads(
                tools_module.lcm_expand(
                    {"store_id": store_id, "content_offset": offset, "max_tokens": 65_536},
                    engine=engine,
                )
            )
            pages.append(page["content"])
            if not page["has_more"]:
                break
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        else:  # pragma: no cover - a regression fails explicitly
            pytest.fail("UTF-8 paging did not terminate")
        assert "".join(pages) == content
    finally:
        engine.shutdown()


def test_reported_checkpoint_is_persisted_and_restart_advances_after_offset_timeout(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    content = ("ordinary durable checkpoint record\n" * 80_000) + "TAIL"
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    requested_offset = 1_500_000
    monkeypatch.setattr(
        tools_module, "_EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS", 10.0
    )
    try:
        with monkeypatch.context() as scoped:
            original_byte_offset = store_module._ContentBlobReader.byte_offset

            def expire_offset_conversion(reader, char_offset):
                if char_offset > 0:
                    raise TimeoutError("forced byte-offset deadline")
                return original_byte_offset(reader, char_offset)

            scoped.setattr(
                store_module._ContentBlobReader,
                "byte_offset",
                expire_offset_conversion,
            )
            first = json.loads(
                tools_module.lcm_expand(
                    {
                        "store_id": store_id,
                        "content_offset": requested_offset,
                        "max_tokens": 64,
                    },
                    engine=engine,
                )
            )

        persisted = engine._store._conn.execute(
            """SELECT char_offset FROM lcm_content_scan_checkpoints
               WHERE store_id=? ORDER BY char_offset DESC LIMIT 1""",
            (store_id,),
        ).fetchone()
        assert first.get("content_boundary_scan_pending") is True, first
        assert persisted is not None
        assert first["content_scan_checkpoint_offset"] == persisted[0]

        engine.shutdown()
        engine = _engine(tmp_path)
        restarted = json.loads(
            tools_module.lcm_expand(
                {
                    "store_id": store_id,
                    "content_offset": requested_offset,
                    "max_tokens": 64,
                },
                engine=engine,
            )
        )
        restarted_persisted = engine._store._conn.execute(
            """SELECT char_offset FROM lcm_content_scan_checkpoints
               WHERE store_id=? ORDER BY char_offset DESC LIMIT 1""",
            (store_id,),
        ).fetchone()
        assert restarted_persisted is not None
        assert restarted_persisted[0] > first["content_scan_checkpoint_offset"]
        if restarted.get("content_boundary_scan_pending"):
            assert restarted["content_scan_checkpoint_offset"] == restarted_persisted[0]
        else:
            assert restarted["content_offset"] == requested_offset
    finally:
        engine.shutdown()


def test_legacy_v8_first_expand_128mib_respects_one_ms_and_returns_scan_pending(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    conn = engine._store._conn
    for trigger_name in (
        "lcm_content_revision_insert",
        "lcm_content_revision_update",
        "lcm_content_revision_delete",
        "msg_fts_insert",
        "msg_fts_update",
        "msg_fts_delete",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    size = 128 * 1024 * 1024
    cursor = conn.execute(
        """INSERT INTO messages(session_id, role, content, timestamp)
           VALUES('current', 'user', zeroblob(?), 1)""",
        (size,),
    )
    store_id = int(cursor.lastrowid)
    conn.commit()
    blob = conn.blobopen("messages", "content", store_id, readonly=False)
    try:
        chunk = b"x" * (1024 * 1024)
        for _ in range(size // len(chunk)):
            blob.write(chunk)
    finally:
        blob.close()
    conn.execute(
        "UPDATE messages SET content=CAST(content AS TEXT) WHERE store_id=?",
        (store_id,),
    )
    conn.commit()
    assert conn.execute(
        "SELECT typeof(content) FROM messages WHERE store_id=?", (store_id,)
    ).fetchone()[0] == "text"
    db_bootstrap.ensure_content_scan_checkpoint_schema(conn)
    for trigger_sql in store_module.build_message_fts_spec().trigger_sqls:
        conn.execute(trigger_sql)
    conn.commit()

    monkeypatch.setattr(
        tools_module, "_EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS", 0.001
    )
    started = time.monotonic()
    try:
        result = json.loads(
            tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": 0, "max_tokens": 64},
                engine=engine,
            )
        )
        elapsed = time.monotonic() - started
        assert elapsed < 0.1
        assert result["content_scan_pending"] is True
        revision = conn.execute(
            """SELECT content_chars, content_bytes, storage_version, scan_byte_offset
               FROM lcm_content_revisions WHERE store_id=?""",
            (store_id,),
        ).fetchone()
        assert revision is not None
        assert revision[1] == size
        assert revision[2] == 1
        assert 0 <= revision[3] < size
        retry = json.loads(
            tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": 0, "max_tokens": 64},
                engine=engine,
            )
        )
        assert retry["content_scan_pending"] is True
        assert retry["content_scan_byte_offset"] > result["content_scan_byte_offset"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "credential",
    [
        "sk-proj-" + ("A" * 1_300_000),
        "github_pat_" + ("B" * 1_300_000),
        "Bearer " + ("eyJhbGciOiJIUzI1NiJ9." + ("C" * 1_300_000)),
    ],
    ids=["openai", "github", "bearer"],
)
def test_deep_checkpoint_never_marks_inside_standalone_credential_normal(
    tmp_path, credential
):
    engine = _engine(tmp_path)
    content = "benign prefix\n" + credential + "\nBENIGN-SUFFIX"
    store_id = engine._store.append("current", {"role": "user", "content": "placeholder"})
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    offset = content.index(credential) + 100_000
    try:
        page = json.loads(
            tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": offset, "max_tokens": 64},
                engine=engine,
            )
        )
        assert credential[20_000:20_100] not in json.dumps(page)
        checkpoint = engine._store._conn.execute(
            """SELECT char_offset, mode FROM lcm_content_scan_checkpoints
               WHERE store_id=? ORDER BY char_offset DESC LIMIT 1""",
            (store_id,),
        ).fetchone()
        assert checkpoint is not None
        credential_start = content.index(credential)
        credential_end = content.index("\nBENIGN-SUFFIX")
        assert not (
            credential_start <= checkpoint[0] < credential_end
            and checkpoint[1] == "normal"
        )
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "suffix",
    ['"session_id":"foreign"', '"unexpected_override":"foreign"'],
    ids=["duplicate-security-key", "unknown-key"],
)
def test_externalized_grep_rejects_trailing_duplicate_security_key(tmp_path, suffix):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "duplicate-session.json"
    (payload_dir / ref).write_text(
        '{"session_id":"current","content":"DUPLICATE-OWNER-SECRET",'
        + suffix
        + "}",
        encoding="utf-8",
    )
    engine = _engine(tmp_path)
    try:
        result = json.loads(
            tools_module.lcm_grep(
                {
                    "query": "DUPLICATE-OWNER-SECRET",
                    "content_scope": "externalized",
                    "ref": ref,
                },
                engine=engine,
            )
        )
        assert result["results"] == []
        assert result["diagnostics"] == [{"ref": ref, "error": "ambiguous_metadata"}]
    finally:
        engine.shutdown()


def test_externalized_grep_accepts_old_canonical_trailing_metadata_losslessly(
    tmp_path,
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "old-canonical.json"
    content = "first line\nOLD-CANONICAL-π🙂-NEEDLE\nlast escaped \\\" line"
    payload = {
        "kind": "ingest_payload",
        "role": "tool",
        "session_id": "current",
        "field_path": "result.content",
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": 1234.5,
    }
    (payload_dir / ref).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    engine = _engine(tmp_path)
    try:
        result = json.loads(
            tools_module.lcm_grep(
                {
                    "query": "OLD-CANONICAL-π🙂-NEEDLE",
                    "content_scope": "externalized",
                    "ref": ref,
                    "max_payload_chars": len(content) + 1,
                },
                engine=engine,
            )
        )
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["results"][0]["matched_text"] == "OLD-CANONICAL-π🙂-NEEDLE"
        assert result["results"][0]["session_id"] == "current"
    finally:
        engine.shutdown()


def _historical_persisted_output_payload(
    *, kind: str, role: str, tool_call_id: str, content: str
) -> dict:
    """Exact raw/tool field order written by ``41b43c8:externalize.py``."""
    preview_sha256 = "a" * 64
    redacted_preview_sha256 = "b" * 64
    marker = {
        "source_path": "/tmp/hermes-results/historical-output.txt",
        "expected_chars": len(content),
        "preview_sha256": preview_sha256,
        "redacted_preview_sha256": redacted_preview_sha256,
        "file_size": len(content.encode("utf-8")),
        "file_mtime_ns": 1_725_000_000_000_000_001,
        "file_ctime_ns": 1_725_000_000_000_000_002,
    }
    return {
        "kind": kind,
        "tool_call_id": tool_call_id,
        "role": role,
        "session_id": "current",
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": 1_725_000_000.25,
        "persisted_output_source_path": marker["source_path"],
        "persisted_output_expected_chars": marker["expected_chars"],
        "persisted_output_preview_sha256": marker["preview_sha256"],
        "persisted_output_redacted_preview_sha256": marker[
            "redacted_preview_sha256"
        ],
        "persisted_output_file_size": marker["file_size"],
        "persisted_output_file_mtime_ns": marker["file_mtime_ns"],
        "persisted_output_file_ctime_ns": marker["file_ctime_ns"],
        "persisted_output_markers": [marker],
    }


def _accumulate_historical_markers(payload: dict, count: int) -> None:
    """Reproduce the distinct-marker merge shape written by ``41b43c8``."""
    first = payload["persisted_output_markers"][0]
    markers = [first]
    for index in range(1, count):
        marker = dict(first)
        marker["source_path"] = (
            f"/tmp/hermes-results/historical-output-{index:04d}.txt"
        )
        marker["preview_sha256"] = f"{index:064x}"
        marker["file_mtime_ns"] += index
        marker["file_ctime_ns"] += index
        markers.append(marker)
    payload["persisted_output_markers"] = markers


@pytest.mark.parametrize(
    ("kind", "role", "tool_call_id", "needle"),
    [
        ("raw_payload", "assistant", "", "HISTORICAL-RAW-PERSISTED-NEEDLE"),
        ("tool_result", "tool", "call-historical", "HISTORICAL-TOOL-PERSISTED-NEEDLE"),
    ],
    ids=["raw-payload", "tool-result"],
)
def test_externalized_grep_accepts_exact_historical_persisted_output_shapes(
    tmp_path, kind, role, tool_call_id, needle
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"{kind}.json"
    content = f"prefix π🙂\n{needle}\nsuffix"
    payload = _historical_persisted_output_payload(
        kind=kind,
        role=role,
        tool_call_id=tool_call_id,
        content=content,
    )
    (payload_dir / ref).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {
                "query": needle,
                "content_scope": "externalized",
                "ref": ref,
                "max_payload_chars": len(content) + 1,
            },
            engine=engine,
        ))
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["results"][0]["matched_text"] == needle
        assert result["results"][0]["session_id"] == "current"
        assert result["results"][0]["content_chars_scanned"] == len(content)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("marker_count", [100, 1_000], ids=["100", "1000"])
def test_externalized_grep_losslessly_streams_accumulated_historical_markers(
    tmp_path, marker_count
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"historical-{marker_count}-markers.json"
    needle = f"HISTORICAL-{marker_count}-MARKER-NEEDLE"
    content = f"prefix π🙂\n{needle}\nsuffix"
    payload = _historical_persisted_output_payload(
        kind="tool_result",
        role="tool",
        tool_call_id=f"historical-{marker_count}",
        content=content,
    )
    _accumulate_historical_markers(payload, marker_count)
    path = payload_dir / ref
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    assert path.stat().st_size > 16 * 1024

    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {
                "query": needle,
                "content_scope": "externalized",
                "ref": ref,
                "max_payload_chars": len(content) + 1,
            },
            engine=engine,
        ))
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["results"][0]["matched_text"] == needle
        assert result["results"][0]["content_chars_scanned"] == len(content)
        assert result["scan"]["bytes_scanned"] == path.stat().st_size
        assert result["scan"]["max_persisted_output_markers"] == 4_096
        assert result["scan"]["max_suffix_depth"] == 3
    finally:
        engine.shutdown()


def test_externalized_grep_marker_cap_is_retryable_not_ambiguous(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "historical-marker-cap.json"
    needle = "HISTORICAL-MARKER-CAP-RETRY-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id="marker-cap", content=needle
    )
    _accumulate_historical_markers(payload, 101)
    (payload_dir / ref).write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(tools_module, "_EXTERNALIZED_SUFFIX_MAX_MARKERS", 100)
        bounded = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert bounded["results"] == []
        assert bounded["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        assert bounded["scan"]["scan_truncated"] is True
        assert bounded["scan"]["max_persisted_output_markers"] == 100

        monkeypatch.setattr(tools_module, "_EXTERNALIZED_SUFFIX_MAX_MARKERS", 4_096)
        retry = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
    finally:
        engine.shutdown()


def test_externalized_grep_marker_cap_is_operation_wide_and_ref_retryable(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    needle = "HISTORICAL-OPERATION-WIDE-MARKER-CAP-NEEDLE"
    refs = []
    for index in range(2):
        ref = f"operation-marker-cap-{index}.json"
        payload = _historical_persisted_output_payload(
            kind="tool_result",
            role="tool",
            tool_call_id=f"operation-cap-{index}",
            content=needle,
        )
        _accumulate_historical_markers(payload, 60)
        (payload_dir / ref).write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        refs.append(ref)

    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(tools_module, "_EXTERNALIZED_SUFFIX_MAX_MARKERS", 100)
        bounded = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "max_files": 2},
            engine=engine,
        ))
        assert bounded["total_results"] == 1
        assert len(bounded["diagnostics"]) == 1
        assert bounded["diagnostics"][0]["ref"] in refs
        assert bounded["diagnostics"][0]["error"] == "payload_truncated"
        assert bounded["scan"]["scan_truncated"] is True
        assert bounded["scan"]["persisted_output_markers_scanned"] == 100
        truncated_ref = bounded["diagnostics"][0]["ref"]

        retry = json.loads(tools_module.lcm_grep(
            {
                "query": needle,
                "content_scope": "externalized",
                "ref": truncated_ref,
            },
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
        assert retry["scan"]["persisted_output_markers_scanned"] == 60
    finally:
        engine.shutdown()


def test_externalized_grep_suffix_byte_budget_is_retryable_and_eventual(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "historical-byte-budget.json"
    needle = "HISTORICAL-SUFFIX-BYTE-BUDGET-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id="byte-budget", content=needle
    )
    _accumulate_historical_markers(payload, 1_000)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    engine = _engine(tmp_path)
    try:
        original_budget = tools_module._LCM_GREP_OPERATION_MAX_BYTES
        monkeypatch.setattr(
            tools_module,
            "_LCM_GREP_OPERATION_MAX_BYTES",
            path.stat().st_size // 2,
        )
        bounded = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert bounded["results"] == []
        assert bounded["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        assert bounded["scan"]["scan_truncated"] is True
        assert bounded["scan"]["bytes_scanned"] <= bounded["scan"]["max_total_bytes"]

        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", original_budget
        )
        retry = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
    finally:
        engine.shutdown()


def test_externalized_grep_suffix_deadline_stops_between_bounded_chunks(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "historical-suffix-deadline.json"
    needle = "HISTORICAL-SUFFIX-DEADLINE-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id="suffix-deadline", content=needle
    )
    _accumulate_historical_markers(payload, 1_000)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    engine = _engine(tmp_path)
    real_open = tools_module.Path.open
    virtual_now = 0.0
    deadline_mode = True
    largest_read = 0

    class DeadlineReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal virtual_now, largest_read
            largest_read = max(largest_read, size)
            chunk = self._handle.read(size)
            if deadline_mode and size > 1 and chunk:
                virtual_now += 0.3
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def deadline_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        if candidate.resolve() == path.resolve() and "b" in mode:
            return DeadlineReader(handle)
        return handle

    monkeypatch.setattr(tools_module.Path, "open", deadline_open)
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: virtual_now)
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: virtual_now)
    try:
        bounded = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert bounded["results"] == []
        assert bounded["diagnostics"] == [{"ref": ref, "error": "body_deadline"}]
        assert bounded["scan"]["scan_truncated"] is True
        assert largest_read <= 16 * 1024

        deadline_mode = False
        virtual_now = 0.0
        retry = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
    finally:
        engine.shutdown()


def test_externalized_grep_accepts_explicit_legacy_persisted_preview_shape(
    tmp_path,
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "legacy-preview.json"
    needle = "HISTORICAL-LEGACY-PREVIEW-NEEDLE"
    content = f"prefix\n{needle}\nsuffix"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id="legacy-preview", content=content
    )
    preview = content[:16]
    payload.pop("persisted_output_preview_sha256")
    payload["persisted_output_preview_prefix"] = preview
    marker = payload["persisted_output_markers"][0]
    marker.pop("preview_sha256")
    marker["preview_prefix"] = preview
    (payload_dir / ref).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "variant",
    [
        "duplicate-persisted-key",
        "duplicate-marker-key",
        "unknown-persisted-key",
        "unknown-marker-key",
        "source-path-type",
        "expected-chars-bool",
        "expected-chars-too-large",
        "digest-noncanonical",
        "file-size-negative",
        "marker-file-time-type",
        "marker-not-list",
        "marker-missing-required",
        "marker-top-level-mismatch",
        "duplicate-marker-entry",
        "malformed-marker-json",
        "pathological-nesting",
    ],
)
def test_externalized_grep_rejects_malicious_historical_persisted_metadata(
    tmp_path, variant
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"malicious-{variant}.json"
    needle = "MALICIOUS-HISTORICAL-PERSISTED-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id="malicious", content=needle
    )

    if variant == "unknown-persisted-key":
        payload["persisted_output_future_override"] = "foreign"
    elif variant == "unknown-marker-key":
        payload["persisted_output_markers"][0]["session_id"] = "foreign"
    elif variant == "source-path-type":
        payload["persisted_output_source_path"] = 7
    elif variant == "expected-chars-bool":
        payload["persisted_output_expected_chars"] = True
    elif variant == "expected-chars-too-large":
        payload["persisted_output_expected_chars"] = 1 << 63
    elif variant == "digest-noncanonical":
        payload["persisted_output_preview_sha256"] = "A" * 64
    elif variant == "file-size-negative":
        payload["persisted_output_file_size"] = -1
    elif variant == "marker-file-time-type":
        payload["persisted_output_markers"][0]["file_mtime_ns"] = "now"
    elif variant == "marker-not-list":
        payload["persisted_output_markers"] = {"source_path": "/tmp/override"}
    elif variant == "marker-missing-required":
        payload["persisted_output_markers"][0].pop("expected_chars")
    elif variant == "marker-top-level-mismatch":
        payload["persisted_output_markers"][0]["source_path"] = "/tmp/foreign"
    elif variant == "duplicate-marker-entry":
        payload["persisted_output_markers"].append(
            dict(payload["persisted_output_markers"][0])
        )
    elif variant == "pathological-nesting":
        payload["persisted_output_markers"][0]["source_path"] = [[["nested"]]]

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if variant == "duplicate-persisted-key":
        target = '"persisted_output_expected_chars":' + str(len(needle))
        serialized = serialized.replace(target, target + "," + target, 1)
    elif variant == "duplicate-marker-key":
        target = '"source_path":"/tmp/hermes-results/historical-output.txt"'
        serialized = serialized.replace(target, target + "," + target, 1)
    elif variant == "malformed-marker-json":
        target = '"source_path":"/tmp/hermes-results/historical-output.txt",'
        serialized = serialized.replace(target, target[:-1], 1)
    (payload_dir / ref).write_text(serialized, encoding="utf-8")

    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert result["results"] == []
        assert result["diagnostics"] == [{
            "ref": ref,
            "error": (
                "ambiguous_metadata"
                if variant in {
                    "duplicate-persisted-key",
                    "duplicate-marker-key",
                    "unknown-persisted-key",
                    "unknown-marker-key",
                }
                else "invalid_payload"
            ),
        }]
    finally:
        engine.shutdown()
