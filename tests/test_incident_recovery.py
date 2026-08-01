from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "incident-recovery.db"),
            async_background_compaction_enabled=False,
            async_background_compaction_worker_enabled=False,
        ),
        hermes_home=str(tmp_path / "hermes-home"),
    )
    engine.on_session_start(
        "incident-session",
        conversation_id="incident-session",
        platform="test",
        context_length=500_000,
    )
    return engine


def test_restart_reconciliation_computes_each_active_identity_once(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    size = 400
    messages = [{"role": "user", "content": f"incoming-no-match-{i}"} for i in range(size)]
    stored_tail = [("assistant", f"durable-no-match-{i}", "", "") for i in range(size)]
    message_ids = {id(message) for message in messages}
    calls: dict[int, int] = {}
    original = engine._message_replay_identity

    def counted(message, *args, **kwargs):
        if id(message) in message_ids:
            calls[id(message)] = calls.get(id(message), 0) + 1
        return original(message, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", counted)
    try:
        assert engine._find_reconciled_cursor_for_store_tail(
            messages, stored_tail, allow_empty_prefix=False,
            session_count=size, raw_session_count=size,
        ) is None
        assert calls == {id(message): 1 for message in messages}
    finally:
        engine.shutdown()


def test_restart_reconciliation_honours_foreground_deadline(tmp_path):
    engine = _engine(tmp_path)
    size = 400
    messages = [{"role": "user", "content": f"incoming-no-match-{i}"} for i in range(size)]
    stored_tail = [("assistant", f"durable-no-match-{i}", "", "") for i in range(size)]
    try:
        with pytest.raises(RuntimeError, match="reconciliation deadline"):
            engine._find_reconciled_cursor_for_store_tail(
                messages, stored_tail, allow_empty_prefix=False,
                session_count=size, raw_session_count=size, deadline_at=0.0,
            )
    finally:
        engine.shutdown()


def test_ingest_threads_foreground_deadline_into_restart_reconciliation(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    seen: list[float | None] = []

    def reconcile(_messages, *, deadline_at=None):
        seen.append(deadline_at)
        return 0

    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", reconcile)
    engine._ingest_cursor_needs_reconcile = True
    try:
        engine._ingest_messages([{"role": "user", "content": "resume me once"}], deadline_at=123.0)
        assert seen == [123.0]
    finally:
        engine.shutdown()


def test_foreground_compress_passes_its_deadline_into_ingest(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "bounded foreground ingest"}]
    seen: list[float] = []

    def ingest(candidate, *, deadline_at):
        seen.append(deadline_at)
        return list(candidate)

    monkeypatch.setattr(engine, "_ingest_messages", ingest)
    try:
        engine.compress(messages, current_tokens=1, force=True)
        assert len(seen) == 1
        assert seen[0] > 0
    finally:
        engine.shutdown()


def test_ingest_serializes_concurrent_callers_inside_storage_boundary(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def ingest_locked(candidate, *, deadline_at=None):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return list(candidate)

    monkeypatch.setattr(engine, "_ingest_messages_locked", ingest_locked)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(engine._ingest_messages, [{"role": "user", "content": f"concurrent-{i}"}])
                for i in range(2)
            ]
            assert all(future.result() for future in futures)
        assert max_active == 1
    finally:
        engine.shutdown()
