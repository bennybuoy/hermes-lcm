"""Regression tests for the five publication/safety blockers at 27d1f5d."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import sqlite3
import sys
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


def _recursive_python_size(value, seen=None) -> int:
    """Count retained Python containers/slots without trusting diagnostics."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _recursive_python_size(key, seen) + _recursive_python_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_recursive_python_size(item, seen) for item in value)
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"} or not hasattr(value, slot):
                continue
            size += _recursive_python_size(getattr(value, slot), seen)
    if hasattr(value, "__dict__"):
        size += _recursive_python_size(vars(value), seen)
    return size


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

        # Completed immutable parses remain reusable until stat invalidation.
        # Change only the stat identity so this probe still exercises bounded
        # body I/O and its deadline rather than the completed cache fast path.
        path_stat = path.stat()
        os.utime(path, ns=(path_stat.st_atime_ns, path_stat.st_mtime_ns + 1))
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


def _move_persisted_metadata_before_content(payload: dict) -> dict:
    """Reproduce the current writer's accumulated-marker field order."""
    return {
        key: value
        for key, value in payload.items()
        if key != "content"
    } | {"content": payload["content"]}


def _write_compact_marker_payload(
    path, *, marker_count: int, pre_content: bool, needle: str
) -> None:
    """Write a large canonical marker list without constructing it in memory."""
    before_content = (
        '{"session_id":"current","persisted_output_source_path":"/p/0",'
        '"persisted_output_expected_chars":1,"persisted_output_markers":['
    )
    after_markers = f'],"content":"{needle}"}}'
    if not pre_content:
        before_content = (
            f'{{"session_id":"current","content":"{needle}",'
            '"persisted_output_source_path":"/p/0",'
            '"persisted_output_expected_chars":1,"persisted_output_markers":['
        )
        after_markers = "]}"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(before_content)
        for index in range(marker_count):
            if index:
                handle.write(",")
            handle.write(
                '{"source_path":"/p/' + str(index) + '","expected_chars":1}'
            )
        handle.write(after_markers)


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


@pytest.mark.parametrize(
    "marker_count", [50, 1_000, 4_097], ids=["50", "1000", "4097"]
)
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
        assert result["scan"]["max_suffix_depth"] == 3
        assert result["scan"]["persisted_output_markers_scanned"] == marker_count
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "marker_count", [50, 1_000, 4_097], ids=["50", "1000", "4097"]
)
def test_externalized_grep_streams_current_precontent_accumulated_markers(
    tmp_path, marker_count
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"current-{marker_count}-markers.json"
    needle = f"CURRENT-{marker_count}-MARKER-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, marker_count)
    payload = _move_persisted_metadata_before_content(payload)
    path = payload_dir / ref
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    assert payload["persisted_output_markers"][-1]["source_path"].endswith(
        f"-{marker_count - 1:04d}.txt"
    )
    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["results"][0]["matched_text"] == needle
        assert result["scan"]["bytes_scanned"] == path.stat().st_size
        assert result["scan"]["persisted_output_markers_scanned"] == marker_count
    finally:
        engine.shutdown()


@pytest.mark.parametrize("layout", ["pre-content", "post-content"])
def test_externalized_grep_processes_substantially_more_than_4097_markers(
    tmp_path, layout
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    marker_count = 12_000
    ref = f"large-{layout}.json"
    needle = f"LARGE-{layout}-MARKER-NEEDLE"
    first = {"source_path": "/p/0", "expected_chars": len(needle)}
    payload = {
        "kind": "tool_result",
        "tool_call_id": ref,
        "role": "tool",
        "session_id": "current",
        "content": needle,
        "persisted_output_source_path": first["source_path"],
        "persisted_output_expected_chars": first["expected_chars"],
        "persisted_output_markers": [first] + [
            {"source_path": f"/p/{index}", "expected_chars": len(needle)}
            for index in range(1, marker_count)
        ],
    }
    if layout == "pre-content":
        payload = _move_persisted_metadata_before_content(payload)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    assert path.stat().st_size < tools_module._LCM_GREP_OPERATION_MAX_BYTES
    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["scan"]["persisted_output_markers_scanned"] == marker_count
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


@pytest.mark.parametrize("marker_count", [60_000])
@pytest.mark.parametrize("layout", ["pre-content", "post-content"])
def test_externalized_grep_exact_canonical_markers_progress_at_production_cap(
    tmp_path, monkeypatch, marker_count, layout
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"canonical-{marker_count}-{layout}.json"
    needle = f"CANONICAL-{marker_count}-{layout}-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, marker_count)
    if layout == "pre-content":
        payload = _move_persisted_metadata_before_content(payload)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    assert path.stat().st_size > tools_module._LCM_GREP_OPERATION_MAX_BYTES

    real_open = tools_module.Path.open
    transport_bytes_read = 0
    read_ranges: list[tuple[int, int]] = []

    class CountingReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal transport_bytes_read
            start = self._handle.tell()
            chunk = self._handle.read(size)
            transport_bytes_read += len(chunk)
            if chunk:
                read_ranges.append((start, start + len(chunk)))
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def counted_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        if candidate.resolve() == path.resolve() and "b" in mode:
            return CountingReader(handle)
        return handle

    monkeypatch.setattr(tools_module.Path, "open", counted_open)
    engine = _engine(tmp_path)
    try:
        offsets = []
        for attempt in range(1, 16):
            bytes_before = transport_bytes_read
            result = json.loads(tools_module.lcm_grep(
                {"query": needle, "content_scope": "externalized", "ref": ref},
                engine=engine,
            ))
            bytes_this_call = transport_bytes_read - bytes_before
            assert result["scan"]["bytes_scanned"] == bytes_this_call
            assert 0 < bytes_this_call <= tools_module._LCM_GREP_OPERATION_MAX_BYTES
            if result["total_results"] == 1:
                break
            assert result["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
            assert result["scan"]["continuations_pending"] == 1
            offsets.append(result["scan"]["continuation_reused_bytes"] + bytes_this_call)
            assert offsets == sorted(set(offsets))
            assert result["scan"]["continuation_memory_bytes"] < 3_000_000
            continuation_cache = engine._externalized_grep_continuations
            checkpoint = next(iter(continuation_cache.values()))
            assert isinstance(
                checkpoint, tools_module._ExternalizedPayloadContinuation
            )
            assert checkpoint.offset == offsets[-1]
            assert checkpoint.retained_bytes() == result["scan"][
                "continuation_memory_bytes"
            ]
            assert checkpoint.retained_bytes() < checkpoint.offset
        else:
            pytest.fail("production-cap retries did not complete the canonical payload")

        assert result["diagnostics"] == []
        assert result["total_results"] == 1
        assert result["scan"]["continuations_pending"] == 0
        assert 10 <= attempt <= 12
        assert path.stat().st_size > 16 * 1024 * 1024
        assert transport_bytes_read == path.stat().st_size
        assert all(end <= next_start for (_, end), (next_start, _) in zip(
            read_ranges, read_ranges[1:]
        ))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("pre_content", [True, False], ids=["pre-content", "post-content"])
def test_externalized_grep_streams_100001_writer_markers_without_count_failure(
    tmp_path, pre_content
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    marker_count = 100_001
    ref = f"uncapped-writer-{pre_content}.json"
    needle = f"UNCAPPED-WRITER-{pre_content}-NEEDLE"
    path = payload_dir / ref
    _write_compact_marker_payload(
        path,
        marker_count=marker_count,
        pre_content=pre_content,
        needle=needle,
    )
    assert path.stat().st_size > tools_module._LCM_GREP_OPERATION_MAX_BYTES

    engine = _engine(tmp_path)
    try:
        total_bytes_read = 0
        max_retained = 0
        for _attempt in range(30):
            result = json.loads(tools_module.lcm_grep(
                {"query": needle, "content_scope": "externalized", "ref": ref},
                engine=engine,
            ))
            bytes_this_call = result["scan"]["bytes_scanned"]
            assert 0 <= bytes_this_call <= tools_module._LCM_GREP_OPERATION_MAX_BYTES
            total_bytes_read += bytes_this_call
            if result["total_results"] == 1:
                break
            assert result["diagnostics"] in (
                [{"ref": ref, "error": "payload_truncated"}],
                [{"ref": ref, "error": "metadata_deadline"}],
                [{"ref": ref, "error": "body_deadline"}],
            )
            checkpoint = next(iter(engine._externalized_grep_continuations.values()))
            max_retained = max(max_retained, checkpoint.retained_bytes())
            assert len(checkpoint.prefix_parser.buffer) < 200_000
            assert checkpoint.retained_bytes() < 3_500_000
        else:
            pytest.fail("100001-marker payload did not complete under bounded retries")

        assert result["diagnostics"] == []
        assert result["results"][0]["matched_text"] == needle
        assert total_bytes_read == path.stat().st_size
        assert max_retained > 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("marker_count", [100_000, 500_000])
def test_externalized_marker_dedup_retains_bounded_real_python_heap(
    tmp_path, monkeypatch, marker_count
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    state_dir = tmp_path / "marker-state"
    monkeypatch.setattr(
        tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_dir
    )
    monkeypatch.setattr(
        tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0
    )
    ref = f"heap-{marker_count}.json"
    path = payload_dir / ref
    _write_compact_marker_payload(
        path,
        marker_count=marker_count,
        pre_content=True,
        needle="BOUNDED-HEAP-NEEDLE",
    )

    engine = _engine(tmp_path)
    try:
        max_recursive = 0
        for _attempt in range(80):
            result = json.loads(tools_module.lcm_grep(
                {
                    "query": "BOUNDED-HEAP-NEEDLE",
                    "content_scope": "externalized",
                    "ref": ref,
                },
                engine=engine,
            ))
            cache = getattr(engine, "_externalized_grep_continuations", {})
            if cache:
                checkpoint = next(iter(cache.values()))
                measured = _recursive_python_size(checkpoint)
                max_recursive = max(max_recursive, measured)
                assert checkpoint.retained_bytes() >= measured
            if result["total_results"] == 1:
                break
        else:
            pytest.fail(f"{marker_count}-marker payload did not complete")

        # The old digest set retained ~100 bytes per marker, so recursive
        # sizing fails by a wide margin at both 100k and 500k.
        assert max_recursive < 8 * 1024 * 1024
        assert result["results"][0]["matched_text"] == "BOUNDED-HEAP-NEEDLE"
    finally:
        engine.shutdown()


def test_externalized_marker_state_is_private_exact_and_cleaned(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    state_dir = tmp_path / "marker-state"
    monkeypatch.setattr(
        tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_dir
    )
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 32 * 1024)
    ref = "disk-backed-duplicate.json"
    path = payload_dir / ref
    _write_compact_marker_payload(
        path, marker_count=20_000, pre_content=True, needle="never-returned"
    )

    engine = _engine(tmp_path)
    try:
        first = json.loads(tools_module.lcm_grep(
            {"query": "never-returned", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert first["scan"]["continuations_pending"] == 1
        state_files = list(state_dir.rglob("markers-*.sqlite3"))
        assert len(state_files) == 1
        assert not state_files[0].is_relative_to(payload_dir)
        assert state_files[0].resolve() != (tmp_path / "open-issues.db").resolve()
        with sqlite3.connect(state_files[0]) as state_connection:
            stored_identity = tuple(json.loads(state_connection.execute(
                "SELECT stat_identity FROM state_metadata"
            ).fetchone()[0]))
        assert stored_identity == tools_module._externalized_file_identity(path.stat())

        # Stat mutation must invalidate and delete the old private index.
        old_state = state_files[0]
        path.write_text(
            '{"session_id":"current","content":"x",'
            '"persisted_output_source_path":"/p/0",'
            '"persisted_output_expected_chars":1,'
            '"persisted_output_markers":['
            '{"source_path":"/p/0","expected_chars":1},'
            '{"source_path":"/p/0","expected_chars":1}]}',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 2 * 1024 * 1024
        )
        duplicate = json.loads(tools_module.lcm_grep(
            {"query": "x", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert duplicate["results"] == []
        assert duplicate["diagnostics"] == [{"ref": ref, "error": "invalid_payload"}]
        assert not old_state.exists()
        assert list(state_dir.rglob("markers-*.sqlite3")) == []
    finally:
        engine.shutdown()


def test_externalized_marker_state_preserves_unprovable_files_and_cleans_own_state(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "marker-state"
    state_dir.mkdir()
    stale = state_dir / "markers-dead-process.sqlite3"
    stale.write_bytes(b"crashed")
    stale_journal = state_dir / "markers-dead-process.sqlite3-journal"
    stale_journal.write_bytes(b"partial")
    unprovable = state_dir / "markers-99999999-fresh.sqlite3"
    unprovable.write_bytes(b"unprovable-owner")
    old = time.time() - tools_module._EXTERNALIZED_MARKER_STATE_TTL_SECONDS - 2
    os.utime(stale, (old, old))
    os.utime(stale_journal, (old, old))
    monkeypatch.setattr(
        tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_dir
    )

    state = tools_module._ExternalizedMarkerIdentityStore()
    live = state.path
    assert live.exists()
    # Legacy/malformed files have no strong owner identity and lease. They are
    # not proof of death, so the safe reaper leaves them alone.
    assert stale.exists()
    assert stale_journal.exists()
    assert unprovable.exists()
    state.close()
    assert not live.exists()
    assert stale.exists() and stale_journal.exists() and unprovable.exists()


def test_externalized_marker_state_count_pressure_never_deletes_unprovable_owners(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "marker-state"
    state_dir.mkdir()
    oldest = None
    for index in range(tools_module._EXTERNALIZED_MARKER_STATE_MAX_ORPHANS + 8):
        path = state_dir / f"markers-crashed-{index:03d}.sqlite3"
        path.write_bytes(b"orphan")
        modified = time.time() - 20 + (index / 100)
        os.utime(path, (modified, modified))
        oldest = oldest or path
    monkeypatch.setattr(
        tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_dir
    )

    state = tools_module._ExternalizedMarkerIdentityStore()
    try:
        assert len(list(state_dir.glob("*.sqlite3"))) == (
            tools_module._EXTERNALIZED_MARKER_STATE_MAX_ORPHANS + 8
        )
        assert oldest is not None and oldest.exists()
    finally:
        state.close()


def test_externalized_grep_file_mutation_invalidates_parser_continuation(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "mutated-continuation.json"
    needle = "MUTATED-CONTINUATION-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, 1_000)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    first_budget = path.stat().st_size // 2

    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", first_budget
        )
        first = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert first["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        assert first["scan"]["continuations_pending"] == 1

        mutated = dict(payload)
        mutated["session_id"] = "foreign"
        path.write_text(
            json.dumps(mutated, separators=(",", ":")), encoding="utf-8"
        )
        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 2 * 1024 * 1024
        )
        mutated_result = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert mutated_result["results"] == []
        assert mutated_result["diagnostics"] == [
            {"ref": ref, "error": "session_mismatch"}
        ]
        assert mutated_result["scan"]["continuation_reused_bytes"] == 0
        assert mutated_result["scan"]["continuations_pending"] == 0
    finally:
        engine.shutdown()


def test_externalized_grep_continuation_cache_enforces_count_and_ttl(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    refs = []
    for index in range(tools_module._EXTERNALIZED_CONTINUATION_MAX_FILES + 1):
        ref = f"bounded-cache-{index}.json"
        refs.append(ref)
        (payload_dir / ref).write_text(json.dumps({
            "session_id": "current",
            "content": f"needle-{index}-" + ("x" * 20_000),
        }, separators=(",", ":")), encoding="utf-8")

    virtual_now = 10.0
    monkeypatch.setattr(
        tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 8 * 1024
    )
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: virtual_now)
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: virtual_now)
    engine = _engine(tmp_path)
    try:
        for index, ref in enumerate(refs):
            result = json.loads(tools_module.lcm_grep(
                {
                    "query": f"needle-{index}",
                    "content_scope": "externalized",
                    "ref": ref,
                },
                engine=engine,
            ))
            assert result["diagnostics"] == [
                {"ref": ref, "error": "payload_truncated"}
            ]
            virtual_now += 1.0

        cache = engine._externalized_grep_continuations
        assert len(cache) == tools_module._EXTERNALIZED_CONTINUATION_MAX_FILES
        assert str((payload_dir / refs[0]).resolve()) not in cache

        virtual_now += tools_module._EXTERNALIZED_CONTINUATION_TTL_SECONDS + 1.0
        expired_retry = json.loads(tools_module.lcm_grep(
            {
                "query": f"needle-{len(refs) - 1}",
                "content_scope": "externalized",
                "ref": refs[-1],
            },
            engine=engine,
        ))
        assert expired_retry["scan"]["continuation_reused_bytes"] == 0
        assert len(engine._externalized_grep_continuations) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("file_count", [5, 20])
def test_externalized_no_ref_discovery_is_eventual_beyond_cache_slots(
    tmp_path, monkeypatch, file_count
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    monkeypatch.setattr(
        tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0
    )
    needle = f"NO-REF-LAST-OF-{file_count}"
    for index in range(file_count):
        content = ((needle + "-") if index == file_count - 1 else "no-match-")
        content += "x" * (tools_module._LCM_GREP_OPERATION_MAX_BYTES + 32_768)
        (payload_dir / f"payload-{index:03d}.json").write_text(
            json.dumps({"session_id": "current", "content": content}, separators=(",", ":")),
            encoding="utf-8",
        )

    engine = _engine(tmp_path)
    try:
        offsets = []
        for attempt in range(1, (file_count * 3) + 6):
            result = json.loads(tools_module.lcm_grep(
                {
                    "query": needle,
                    "content_scope": "externalized",
                    "limit": file_count,
                    "max_files": file_count,
                },
                engine=engine,
            ))
            scheduler = next(iter(engine._externalized_grep_schedulers.values()))
            offsets.append((scheduler.cursor, scheduler.active_ref))
            assert len(engine._externalized_grep_schedulers) <= (
                tools_module._EXTERNALIZED_SCHEDULER_MAX_QUERIES
            )
            mutable = [
                checkpoint
                for checkpoint in engine._externalized_grep_continuations.values()
                if not checkpoint.completed
            ]
            assert len(mutable) <= 1
            assert len(engine._externalized_grep_continuations) == len(mutable)
            if result["total_results"]:
                break
        else:
            pytest.fail("no-ref scheduler never reached the final payload")

        assert result["results"][0]["ref"] == f"payload-{file_count - 1:03d}.json"
        assert result["results"][0]["matched_text"] == needle
        assert attempt <= (file_count * 2) + 2
        assert len(set(offsets)) > file_count
    finally:
        engine.shutdown()


def test_externalized_no_ref_schedulers_isolate_queries_and_survive_deletion(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    monkeypatch.setattr(
        tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0
    )
    large = "x" * (tools_module._LCM_GREP_OPERATION_MAX_BYTES + 16_384)
    for index in range(5):
        prefixes = []
        if index == 0:
            prefixes.append("QUERY-B-FIRST")
        if index == 4:
            prefixes.append("QUERY-A-LAST")
        (payload_dir / f"isolation-{index}.json").write_text(
            json.dumps({
                "session_id": "current",
                "content": " ".join(prefixes) + large,
            }, separators=(",", ":")),
            encoding="utf-8",
        )

    engine = _engine(tmp_path)
    try:
        first_a = json.loads(tools_module.lcm_grep(
            {
                "query": "QUERY-A-LAST", "content_scope": "externalized",
                "limit": 5, "max_files": 5,
            },
            engine=engine,
        ))
        assert first_a["total_results"] == 0
        (payload_dir / "isolation-0.json").unlink()

        result_b = json.loads(tools_module.lcm_grep(
            {
                "query": "QUERY-B-FIRST", "content_scope": "externalized",
                "limit": 5, "max_files": 5,
            },
            engine=engine,
        ))
        assert result_b["total_results"] == 0  # deleted candidate cannot leak
        assert len(engine._externalized_grep_schedulers) == 2

        for _attempt in range(20):
            result_a = json.loads(tools_module.lcm_grep(
                {
                    "query": "QUERY-A-LAST", "content_scope": "externalized",
                    "limit": 5, "max_files": 5,
                },
                engine=engine,
            ))
            if result_a["total_results"]:
                break
        else:
            pytest.fail("query A was lost after active-candidate deletion")
        assert result_a["results"][0]["ref"] == "isolation-4.json"
    finally:
        engine.shutdown()


def test_externalized_no_ref_scheduler_cache_enforces_count_and_ttl(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    (payload_dir / "one.json").write_text(
        '{"session_id":"current","content":"haystack"}', encoding="utf-8"
    )
    virtual_now = 100.0
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: virtual_now)
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: virtual_now)
    engine = _engine(tmp_path)
    try:
        for index in range(tools_module._EXTERNALIZED_SCHEDULER_MAX_QUERIES + 1):
            json.loads(tools_module.lcm_grep(
                {
                    "query": f"query-{index}",
                    "content_scope": "externalized",
                    "max_files": 1,
                },
                engine=engine,
            ))
            virtual_now += 0.01
        assert len(engine._externalized_grep_schedulers) == (
            tools_module._EXTERNALIZED_SCHEDULER_MAX_QUERIES
        )

        virtual_now += tools_module._EXTERNALIZED_SCHEDULER_TTL_SECONDS + 1
        json.loads(tools_module.lcm_grep(
            {
                "query": "after-ttl",
                "content_scope": "externalized",
                "max_files": 1,
            },
            engine=engine,
        ))
        assert len(engine._externalized_grep_schedulers) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    [
        ("malformed", "invalid_payload"),
        ("duplicate-marker", "invalid_payload"),
        ("unknown-key", "ambiguous_metadata"),
        ("session-override", "ambiguous_metadata"),
    ],
)
def test_externalized_grep_continuation_still_fails_closed_on_late_metadata(
    tmp_path, variant, expected_error
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"continued-{variant}.json"
    needle = "CONTINUED-FAIL-CLOSED-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, 1_000)
    if variant == "duplicate-marker":
        payload["persisted_output_markers"].append(
            dict(payload["persisted_output_markers"][0])
        )
    serialized = json.dumps(payload, separators=(",", ":"))
    if variant == "malformed":
        serialized = serialized[:-1]
    elif variant == "unknown-key":
        serialized = serialized[:-1] + ',"future_override":true}'
    elif variant == "session-override":
        serialized = serialized[:-1] + ',"session_id":"foreign"}'
    path = payload_dir / ref
    path.write_text(serialized, encoding="utf-8")

    engine = _engine(tmp_path)
    try:
        _, diagnostics, scan = tools_module._search_externalized_payloads(
            engine,
            query=needle,
            regex_mode=False,
            allowed_session_ids=frozenset({"current"}),
            ref=ref,
            limit=1,
            max_files=1,
            max_payload_chars=len(needle) + 1,
            max_total_bytes=path.stat().st_size // 2,
            deadline=time.monotonic() + 10,
        )
        assert diagnostics == [{"ref": ref, "error": "payload_truncated"}]
        assert scan["continuations_pending"] == 1

        hits, diagnostics, scan = tools_module._search_externalized_payloads(
            engine,
            query=needle,
            regex_mode=False,
            allowed_session_ids=frozenset({"current"}),
            ref=ref,
            limit=1,
            max_files=1,
            max_payload_chars=len(needle) + 1,
            max_total_bytes=tools_module._LCM_GREP_OPERATION_MAX_BYTES,
            deadline=time.monotonic() + 10,
        )
        assert hits == []
        assert diagnostics == [{"ref": ref, "error": expected_error}]
        assert scan["continuation_reused_bytes"] > 0
        assert scan["continuations_pending"] == 0
    finally:
        engine.shutdown()


def test_externalized_grep_charges_every_rejected_byte_under_shared_40kb_cap(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    variants = [
        b'{"session_id":"current","content":"' + (b"A" * 24_000) + b'\xff"}',
        b'{"session_id":"current","content":"x","unknown":' + (b"0" * 24_000),
        b'{"session_id":"foreign","padding":"' + (b"F" * 24_000) + b'","content":"x"}',
        b'{"session_id":"current","content":"x","created_at":' + (b"9" * 24_000),
        b'{"session_id":"current","content":"x","session_id":"current"}'
        + (b"Z" * 24_000),
    ]
    paths = []
    for index, raw in enumerate(variants):
        path = payload_dir / f"rejected-{index}.json"
        path.write_bytes(raw)
        paths.append(path.resolve())

    real_open = tools_module.Path.open
    transport_bytes_read = 0

    class CountingReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal transport_bytes_read
            chunk = self._handle.read(size)
            transport_bytes_read += len(chunk)
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def counted_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        if candidate.resolve() in paths and "b" in str(
            args[0] if args else kwargs.get("mode", "r")
        ):
            return CountingReader(handle)
        return handle

    monkeypatch.setattr(tools_module.Path, "open", counted_open)
    engine = _engine(tmp_path)
    try:
        _, diagnostics, scan = tools_module._search_externalized_payloads(
            engine,
            query="never-matches",
            regex_mode=False,
            allowed_session_ids=frozenset({"current"}),
            ref="",
            limit=10,
            max_files=10,
            max_payload_chars=1_000,
            max_total_bytes=40_000,
            deadline=time.monotonic() + 10,
        )
        assert diagnostics
        assert scan["scan_truncated"] is True
        assert scan["byte_budget_exhausted"] is True
        assert scan["max_total_bytes"] == 40_000
        assert transport_bytes_read == scan["bytes_scanned"] == 40_000
    finally:
        engine.shutdown()


@pytest.mark.parametrize("layout", ["pre-content", "post-content"])
@pytest.mark.parametrize(
    "variant",
    ["duplicate-top", "unknown-top", "duplicate-marker", "unknown-marker"],
)
def test_externalized_grep_streaming_marker_metadata_rejects_ambiguity(
    tmp_path, layout, variant
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"{layout}-{variant}.json"
    needle = "STREAMING-METADATA-REJECTION-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, 50)
    if layout == "pre-content":
        payload = _move_persisted_metadata_before_content(payload)
    serialized = json.dumps(payload, separators=(",", ":"))
    if variant == "duplicate-top":
        target = '"persisted_output_expected_chars":' + str(len(needle))
        serialized = serialized.replace(target, target + "," + target, 1)
    elif variant == "unknown-top":
        serialized = serialized.replace(
            '"persisted_output_expected_chars":',
            '"persisted_output_override":0,"persisted_output_expected_chars":',
            1,
        )
    elif variant == "duplicate-marker":
        target = '"source_path":"/tmp/hermes-results/historical-output.txt"'
        serialized = serialized.replace(target, target + "," + target, 1)
    else:
        target = '"source_path":"/tmp/hermes-results/historical-output.txt"'
        serialized = serialized.replace(target, target + ',"override":true', 1)
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
            "error": "ambiguous_metadata",
        }]
    finally:
        engine.shutdown()


def test_externalized_grep_precontent_deadline_checks_unaligned_authorized_chunks(
    tmp_path, monkeypatch,
):
    marker_count = 12_000
    payload = {
        "session_id": "current",
        "persisted_output_source_path": "/p/0",
        "persisted_output_expected_chars": 1,
        "persisted_output_markers": [
            {"source_path": f"/p/{index}", "expected_chars": 1}
            for index in range(marker_count)
        ],
        "content": "x",
    }
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "unaligned-precontent-deadline.json"
    path = payload_dir / ref
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    virtual_now = 0.0
    bytes_read = 0
    deadline_mode = True
    real_open = tools_module.Path.open

    class DeadlineReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal virtual_now, bytes_read
            chunk = self._handle.read(size)
            bytes_read += len(chunk)
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
    engine = _engine(tmp_path)
    try:
        bounded = json.loads(tools_module.lcm_grep(
            {"query": "x", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert bounded["results"] == []
        assert bounded["diagnostics"] == [
            {"ref": ref, "error": "metadata_deadline"}
        ]
        assert 0 < bytes_read < len(raw)
        assert bounded["scan"]["continuations_pending"] == 1

        deadline_mode = False
        virtual_now = 0.0
        retry = json.loads(tools_module.lcm_grep(
            {"query": "x", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
        assert retry["scan"]["continuation_reused_bytes"] > 0
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
    transport_bytes_read = 0

    class DeadlineReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal virtual_now, largest_read, transport_bytes_read
            largest_read = max(largest_read, size)
            chunk = self._handle.read(size)
            transport_bytes_read += len(chunk)
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
        assert 0 < bounded["scan"]["bytes_scanned"] < path.stat().st_size
        assert bounded["scan"]["bytes_scanned"] == transport_bytes_read
        assert largest_read <= 16 * 1024
        assert bounded["scan"]["continuations_pending"] == 1

        retry = bounded
        for attempt in range(1, 20):
            virtual_now = 0.0
            retry_bytes_before = transport_bytes_read
            retry = json.loads(tools_module.lcm_grep(
                {"query": needle, "content_scope": "externalized", "ref": ref},
                engine=engine,
            ))
            assert retry["scan"]["bytes_scanned"] == (
                transport_bytes_read - retry_bytes_before
            )
            assert retry["scan"]["continuation_reused_bytes"] > 0
            if retry["total_results"] == 1:
                break
            assert retry["diagnostics"] == [{"ref": ref, "error": "body_deadline"}]
            assert retry["scan"]["continuations_pending"] == 1
        else:
            pytest.fail("bounded retries did not make reusable per-file progress")

        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
        assert retry["scan"]["continuations_pending"] == 0
        assert transport_bytes_read == path.stat().st_size
    finally:
        engine.shutdown()


def test_externalized_grep_reuses_completed_parse_after_outer_deadline(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "completed-before-outer-deadline.json"
    needle = "COMPLETED-PARSE-OUTER-DEADLINE-NEEDLE"
    path = payload_dir / ref
    path.write_text(json.dumps({
        "session_id": "current",
        "content_chars": len(needle),
        "content_bytes": len(needle.encode("utf-8")),
        "content": needle,
    }, separators=(",", ":")), encoding="utf-8")

    real_open = tools_module.Path.open
    real_mandatory_redact = tools_module._mandatory_redact_grep_response
    bytes_read = 0
    virtual_now = 0.0
    expire_outer_response = True

    class CountingReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal bytes_read
            chunk = self._handle.read(size)
            bytes_read += len(chunk)
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def counted_open(candidate, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        if candidate.resolve() == path.resolve() and "b" in mode:
            return CountingReader(handle)
        return handle

    def expire_after_response_metadata(value):
        nonlocal virtual_now
        protected = real_mandatory_redact(value)
        if expire_outer_response and isinstance(value, dict) and "query" in value:
            virtual_now = 2.0
        return protected

    monkeypatch.setattr(tools_module.Path, "open", counted_open)
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: virtual_now)
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: virtual_now)
    monkeypatch.setattr(
        tools_module, "_mandatory_redact_grep_response", expire_after_response_metadata
    )
    engine = _engine(tmp_path)
    try:
        expired = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert expired["results"] == []
        assert expired["operation_budget"]["deadline_exhausted"] is True
        assert expired["scan"]["bytes_scanned"] > 0
        assert expired["scan"]["continuations_pending"] == 1
        assert bytes_read == path.stat().st_size
        checkpoint = next(iter(engine._externalized_grep_continuations.values()))
        assert checkpoint.completed is True

        expire_outer_response = False
        virtual_now = 0.0
        before_retry = bytes_read
        retry = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
        assert retry["scan"]["continuation_reused_bytes"] == path.stat().st_size
        assert bytes_read == before_retry
        assert next(iter(engine._externalized_grep_continuations.values())) is checkpoint
    finally:
        engine.shutdown()


def test_externalized_grep_rejects_cross_retry_five_megabyte_scalar_boundedly(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "five-megabyte-marker-scalar.json"
    path = payload_dir / ref
    malicious_scalar = "x" * (5 * 1024 * 1024)
    path.write_text(
        '{"session_id":"current",'
        '"persisted_output_source_path":"/safe",'
        '"persisted_output_expected_chars":1,'
        '"persisted_output_markers":[{"source_path":"'
        + malicious_scalar
        + '","expected_chars":1}],"content":"needle"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 128 * 1024
    )
    engine = _engine(tmp_path)
    try:
        first = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert first["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        assert first["scan"]["continuations_pending"] == 1
        assert first["scan"]["continuation_memory_bytes"] < 210_000

        second = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert second["results"] == []
        assert second["diagnostics"] == [{"ref": ref, "error": "invalid_payload"}]
        assert second["scan"]["continuation_reused_bytes"] > 0
        assert second["scan"]["continuations_pending"] == 0
        assert second["scan"]["continuation_memory_bytes"] == 0
        assert (
            first["scan"]["bytes_scanned"] + second["scan"]["bytes_scanned"]
            < 300_000
        )
    finally:
        engine.shutdown()


@pytest.mark.parametrize("layout", ["pre-content", "post-content"])
def test_externalized_grep_discards_long_delimiter_whitespace_across_retries(
    tmp_path, monkeypatch, layout
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = f"long-delimiter-whitespace-{layout}.json"
    needle = f"LONG-DELIMITER-WHITESPACE-{layout}-NEEDLE"
    short_whitespace = " " * 20_000
    long_whitespace = " " * (5 * 1024 * 1024)
    marker_zero = '{"source_path":"/p/0","expected_chars":1}'
    marker_one = '{"source_path":"/p/1","expected_chars":1}'
    if layout == "pre-content":
        raw = (
            '{"session_id":"current"' + short_whitespace
            + ',"persisted_output_source_path":"/p/0",'
            '"persisted_output_expected_chars":1,"persisted_output_markers":['
            + marker_zero + long_whitespace + "]" + short_whitespace
            + ',"content":"' + needle + '"}'
        )
    else:
        raw = (
            '{"session_id":"current","content":"' + needle + '",'
            '"persisted_output_source_path":"/p/0"' + short_whitespace
            + ',"persisted_output_expected_chars":1,"persisted_output_markers":['
            + marker_zero + short_whitespace + "," + marker_one
            + long_whitespace + "]" + short_whitespace + "}"
        )
    path = payload_dir / ref
    path.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: 0.0)

    engine = _engine(tmp_path)
    try:
        total_bytes_read = 0
        for _attempt in range(12):
            result = json.loads(tools_module.lcm_grep(
                {"query": needle, "content_scope": "externalized", "ref": ref},
                engine=engine,
            ))
            total_bytes_read += result["scan"]["bytes_scanned"]
            if result["total_results"] == 1:
                break
            assert result["diagnostics"] == [
                {"ref": ref, "error": "payload_truncated"}
            ]
            checkpoint = next(iter(engine._externalized_grep_continuations.values()))
            parser = (
                checkpoint.prefix_parser
                if checkpoint.phase == "metadata"
                else checkpoint.content_state.suffix_parser
            )
            assert parser.state in {
                "value_delimiter", "marker_delimiter", "marker_or_end", "key",
            }
            assert len(parser.buffer) <= 16 * 1024
            # Includes the explicitly bounded native SQLite page-cache budget,
            # not only Python string payload bytes.
            assert checkpoint.retained_bytes() < 600_000
        else:
            pytest.fail("legal delimiter whitespace did not complete across retries")

        assert result["diagnostics"] == []
        assert result["results"][0]["matched_text"] == needle
        assert total_bytes_read == path.stat().st_size
    finally:
        engine.shutdown()


def test_externalized_grep_long_delimiter_wait_is_stat_invalidated(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "mutated-delimiter-wait.json"
    path = payload_dir / ref
    whitespace = " " * 200_000
    path.write_text(
        '{"session_id":"current"' + whitespace + ',"content":"OLD-NEEDLE"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 64 * 1024)
    engine = _engine(tmp_path)
    try:
        first = json.loads(tools_module.lcm_grep(
            {"query": "NEW-NEEDLE", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert first["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        checkpoint = next(iter(engine._externalized_grep_continuations.values()))
        assert checkpoint.prefix_parser.state == "value_delimiter"
        assert len(checkpoint.prefix_parser.buffer) <= 16 * 1024

        path.write_text(
            '{"session_id":"current"' + whitespace + ',"content":"NEW-NEEDLE"}',
            encoding="utf-8",
        )
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 512 * 1024)
        retry = json.loads(tools_module.lcm_grep(
            {"query": "NEW-NEEDLE", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["diagnostics"] == []
        assert retry["total_results"] == 1
        assert retry["scan"]["continuation_reused_bytes"] == 0
    finally:
        engine.shutdown()


def test_externalized_grep_long_delimiter_wait_rejects_malformed_delimiter(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "malformed-delimiter-wait.json"
    path = payload_dir / ref
    path.write_text(
        '{"session_id":"current"' + (" " * 200_000) + '!"content":"needle"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 64 * 1024)
    engine = _engine(tmp_path)
    try:
        first = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert first["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        checkpoint = next(iter(engine._externalized_grep_continuations.values()))
        assert checkpoint.prefix_parser.state == "value_delimiter"
        assert len(checkpoint.prefix_parser.buffer) <= 16 * 1024

        monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 512 * 1024)
        retry = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert retry["results"] == []
        assert retry["diagnostics"] == [{"ref": ref, "error": "invalid_payload"}]
        assert retry["scan"]["continuation_reused_bytes"] > 0
    finally:
        engine.shutdown()


def test_externalized_grep_concurrent_same_ref_uses_distinct_mutable_checkouts(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "concurrent-same-ref.json"
    needle = "CONCURRENT-SAME-REF-NEEDLE"
    payload = _historical_persisted_output_payload(
        kind="tool_result", role="tool", tool_call_id=ref, content=needle
    )
    _accumulate_historical_markers(payload, 2_000)
    path = payload_dir / ref
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 64 * 1024
        )
        initial = json.loads(tools_module.lcm_grep(
            {"query": needle, "content_scope": "externalized", "ref": ref},
            engine=engine,
        ))
        assert initial["diagnostics"] == [{"ref": ref, "error": "payload_truncated"}]
        assert initial["scan"]["continuations_pending"] == 1

        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 2 * 1024 * 1024
        )
        monkeypatch.setattr(
            tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0
        )
        original_resume = tools_module._ExternalizedPayloadContinuation.resume
        start_gate = threading.Barrier(2)
        observed_ids: list[int] = []
        observed_lock = threading.Lock()
        outcomes: queue.Queue = queue.Queue()

        def synchronized_resume(continuation, *args, **kwargs):
            with observed_lock:
                observed_ids.append(id(continuation))
            start_gate.wait(timeout=10)
            return original_resume(continuation, *args, **kwargs)

        monkeypatch.setattr(
            tools_module._ExternalizedPayloadContinuation,
            "resume",
            synchronized_resume,
        )

        def grep_once():
            try:
                outcomes.put(json.loads(tools_module.lcm_grep(
                    {"query": needle, "content_scope": "externalized", "ref": ref},
                    engine=engine,
                )))
            except BaseException as exc:  # noqa: BLE001 - surfaced to assertion
                outcomes.put(exc)

        threads = [threading.Thread(target=grep_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()

        concurrent_results = [outcomes.get_nowait() for _ in threads]
        assert not any(isinstance(result, BaseException) for result in concurrent_results)
        assert len(observed_ids) == 2
        assert len(set(observed_ids)) == 2
        assert all(result["total_results"] == 1 for result in concurrent_results)
        assert all(result["scan"]["bytes_scanned"] <= 2 * 1024 * 1024 for result in concurrent_results)
        cached = list(engine._externalized_grep_continuations.values())
        assert len(cached) == 1
        assert cached[0].completed is True
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "ordering", ["preserve-then-commit", "commit-then-preserve"]
)
def test_completed_continuation_mixed_acknowledgements_never_delete_checkpoint(
    tmp_path, ordering
):
    engine = _engine(tmp_path)
    try:
        key = str((tmp_path / "payloads" / "shared-completed.json").resolve())
        checkpoint = tools_module._ExternalizedPayloadContinuation(
            identity=(1, 2, 3, 4, 5),
            allowed_session_ids=frozenset({"current"}),
            max_payload_chars=100,
            operation_budget=tools_module._ExternalizedSuffixOperationBudget(),
        )
        checkpoint.completed = True
        checkpoint.offset = 3
        tools_module._store_externalized_continuation(engine, key, checkpoint)
        first = tools_module._checkout_externalized_continuation(
            engine,
            key,
            identity=checkpoint.identity,
            allowed_session_ids=checkpoint.allowed_session_ids,
            max_payload_chars=checkpoint.max_payload_chars,
            file_size=checkpoint.offset,
        )
        second = tools_module._checkout_externalized_continuation(
            engine,
            key,
            identity=checkpoint.identity,
            allowed_session_ids=checkpoint.allowed_session_ids,
            max_payload_chars=checkpoint.max_payload_chars,
            file_size=checkpoint.offset,
        )
        assert first is checkpoint and second is checkpoint

        start_gate = threading.Barrier(2)
        first_done = threading.Event()
        failures: queue.Queue = queue.Queue()

        def acknowledge(kind: str) -> None:
            try:
                completion = tools_module._ExternalizedContinuationCompletion(
                    engine, key, checkpoint
                )
                start_gate.wait(timeout=10)
                goes_first = kind == ordering.split("-then-")[0]
                if not goes_first:
                    assert first_done.wait(timeout=10)
                getattr(completion, kind)()
                if goes_first:
                    first_done.set()
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                failures.put(exc)

        threads = [
            threading.Thread(target=acknowledge, args=(kind,))
            for kind in ("commit", "preserve")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()
        assert failures.empty(), list(failures.queue)
        assert engine._externalized_grep_continuations.get(key) is checkpoint
        assert checkpoint.completed is True
    finally:
        engine.shutdown()


def test_completed_acknowledgement_does_not_resurrect_evicted_checkpoint(tmp_path):
    engine = _engine(tmp_path)
    try:
        key = str((tmp_path / "payloads" / "evicted-completed.json").resolve())
        checkpoint = tools_module._ExternalizedPayloadContinuation(
            identity=(10, 20, 30, 40, 50),
            allowed_session_ids=frozenset({"current"}),
            max_payload_chars=100,
            operation_budget=tools_module._ExternalizedSuffixOperationBudget(),
        )
        checkpoint.completed = True
        completion = tools_module._ExternalizedContinuationCompletion(
            engine, key, checkpoint
        )

        tools_module._store_externalized_continuation(engine, key, checkpoint)
        checkpoint.cached_at = (
            time.monotonic()
            - tools_module._EXTERNALIZED_CONTINUATION_TTL_SECONDS
            - 1
        )
        completion.preserve()
        assert key not in engine._externalized_grep_continuations

        tools_module._store_externalized_continuation(engine, key, checkpoint)
        incompatible = tools_module._checkout_externalized_continuation(
            engine,
            key,
            identity=(10, 20, 31, 40, 50),
            allowed_session_ids=checkpoint.allowed_session_ids,
            max_payload_chars=checkpoint.max_payload_chars,
            file_size=checkpoint.offset,
        )
        assert incompatible is None
        completion.preserve()
        assert key not in engine._externalized_grep_continuations

        tools_module._store_externalized_continuation(engine, key, checkpoint)
        for index in range(tools_module._EXTERNALIZED_CONTINUATION_MAX_FILES):
            replacement = tools_module._ExternalizedPayloadContinuation(
                identity=(index, index, index, index, index),
                allowed_session_ids=frozenset({"current"}),
                max_payload_chars=100,
                operation_budget=tools_module._ExternalizedSuffixOperationBudget(),
            )
            replacement.completed = True
            tools_module._store_externalized_continuation(
                engine, f"replacement-{index}", replacement
            )
        assert key not in engine._externalized_grep_continuations
        completion.preserve()
        assert key not in engine._externalized_grep_continuations
        assert len(engine._externalized_grep_continuations) == (
            tools_module._EXTERNALIZED_CONTINUATION_MAX_FILES
        )
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
