"""Regression tests for the five publication/safety blockers at 27d1f5d."""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import threading

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
    store_id = engine._store.append("current", {"role": "user", "content": content})
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


def test_externalized_grep_rejects_trailing_duplicate_security_key(tmp_path):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    ref = "duplicate-session.json"
    (payload_dir / ref).write_text(
        '{"session_id":"current","content":"DUPLICATE-OWNER-SECRET",'
        '"session_id":"foreign"}',
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
