"""End-to-end integration tests for async background compaction.

Exercises the real daemon worker thread through prepare → promote / reject /
failure-backoff lifecycles. These are intentionally separate from the design
gate tests, which call prepare/promote synchronously without the worker.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _make_async_engine(
    tmp_path,
    *,
    session_id: str = "async-integ-session",
    conversation_id: str = "async-integ-conversation",
    worker_interval: float = 0.5,
    **extra_config,
) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
        async_background_compaction_worker_interval_seconds=worker_interval,
        **extra_config,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=50_000,
    )
    return engine


def _messages(count: int = 20, *, tokens_each: int = 40) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    filler = "x " * max(1, tokens_each // 2)
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"msg {idx} {filler}"})
    return messages


def _stub_leaf_summary(monkeypatch, engine, *, text: str = "deterministic async summary"):
    """Replace leaf summarization with a deterministic stub (no LLM)."""

    def fake_summarize(initial_chunk, focus_topic=None):
        source_tokens = max(1, len(initial_chunk) * 10)
        return (list(initial_chunk), source_tokens, text, 1, 1)

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_summarize)


def _stub_leaf_summary_failure(monkeypatch, engine):
    """Force every leaf summary attempt to fail."""

    def boom(initial_chunk, focus_topic=None):
        raise RuntimeError("injected async prepare failure")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", boom)


def _wait_for_ready_batch(engine: LCMEngine, *, timeout: float = 3.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = engine.get_async_compaction_status()
        if int(last.get("prepared_batches", 0) or 0) >= 1:
            return last
        time.sleep(0.1)
    raise AssertionError(
        f"timed out waiting for ready batch; last status={last!r}"
    )


def _wait_for_failed_batches(
    engine: LCMEngine, *, timeout: float = 5.0, min_failed: int = 1
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = engine.get_async_compaction_status()
        if int(last.get("failed_batches", 0) or 0) >= min_failed:
            # Give the worker a couple more ticks so stuck preparing/ready
            # states have a chance to appear if they would.
            time.sleep(0.3)
            return engine.get_async_compaction_status()
        time.sleep(0.1)
    raise AssertionError(
        f"timed out waiting for failed_batches>={min_failed}; last status={last!r}"
    )


def test_worker_prepare_then_promote_on_compress(tmp_path, monkeypatch):
    """Worker prepares a ready batch; compress() promotes it and shuts down cleanly."""
    engine = _make_async_engine(tmp_path, session_id="promote-e2e")
    _stub_leaf_summary(monkeypatch, engine)
    worker = engine._async_worker_thread
    assert worker is not None and worker.is_alive()
    assert worker.daemon is True

    try:
        messages = _messages(20, tokens_each=40)
        engine.ingest(messages)

        status = _wait_for_ready_batch(engine, timeout=3.0)
        assert status["prepared_batches"] >= 1
        assert status["preparing_batches"] == 0

        before_count = engine.compression_count
        compacted = engine.compress(
            messages, current_tokens=engine.threshold_tokens + 1
        )

        assert engine.compression_count == before_count + 1
        assert compacted is not None
        nodes = engine._dag.get_session_nodes(engine.current_session_id)
        assert len(nodes) >= 1
        async_status = engine.get_async_compaction_status()
        assert async_status["promoted_batches"] >= 1
        assert async_status["total_promote_succeeded"] >= 1
        assert async_status["last_prepare_at"] is not None
        assert async_status["last_promote_at"] is not None
    finally:
        engine.shutdown()

    assert not worker.is_alive()


def test_reject_prepared_batch_then_foreground_compress(tmp_path, monkeypatch):
    """Rejecting a ready batch does not block normal foreground compression."""
    engine = _make_async_engine(tmp_path, session_id="reject-e2e")
    _stub_leaf_summary(monkeypatch, engine)
    worker = engine._async_worker_thread
    assert worker is not None and worker.is_alive()

    try:
        messages = _messages(20, tokens_each=40)
        engine.ingest(messages)
        _wait_for_ready_batch(engine, timeout=3.0)

        ready = engine._frontier.get_ready_batch(engine.current_conversation_id)
        assert ready is not None
        # Stop the worker so it cannot race a new ready batch before compress.
        engine._config.async_background_compaction_worker_enabled = False
        engine._stop_async_worker()

        engine.reject_prepared_compaction(ready.batch_id, reason="operator_reject")

        batch = engine._frontier.get_batch(ready.batch_id)
        assert batch is not None
        assert batch.state == "rejected"
        assert engine.get_async_compaction_status()["rejected_batches"] >= 1
        assert engine.get_async_compaction_status()["prepared_batches"] == 0

        before_nodes = len(engine._dag.get_session_nodes(engine.current_session_id))
        compacted = engine.compress(
            messages, current_tokens=engine.threshold_tokens + 1
        )

        assert compacted != messages
        assert engine._last_compression_status == "compacted"
        after_nodes = engine._dag.get_session_nodes(engine.current_session_id)
        assert len(after_nodes) > before_nodes
        # Rejected path must not count as a successful promote.
        assert engine.get_async_compaction_status()["promoted_batches"] == 0
    finally:
        engine.shutdown()

    assert not worker.is_alive()


def test_prepare_failures_backoff_keeps_worker_alive(tmp_path, monkeypatch):
    """Repeated prepare failures mark batches failed without killing the worker."""
    engine = _make_async_engine(
        tmp_path,
        session_id="backoff-e2e",
        # Trip the circuit breaker quickly so cooldown is exercised in-window.
        async_background_compaction_worker_max_consecutive_failures=2,
        async_background_compaction_worker_cooldown_seconds=30.0,
    )
    _stub_leaf_summary_failure(monkeypatch, engine)
    worker = engine._async_worker_thread
    assert worker is not None and worker.is_alive()

    try:
        messages = _messages(20, tokens_each=40)
        engine.ingest(messages)

        status = _wait_for_failed_batches(engine, timeout=5.0, min_failed=1)
        assert status["failed_batches"] >= 1
        assert status["prepared_batches"] == 0
        assert status["preparing_batches"] == 0
        assert status["pending_batches"] == 0
        # Worker must survive failures / cooldown entry.
        assert worker.is_alive()
        assert engine._async_worker_thread is worker
        assert engine._async_worker_thread.is_alive()
        # Telemetry should reflect at least one prepare attempt.
        assert status["total_prepare_attempts"] >= 1
        assert status["worker_last_tick_at"] is not None
    finally:
        engine.shutdown()

    assert not worker.is_alive()
