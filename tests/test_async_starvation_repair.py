"""Regression tests for async-lane starvation repair.

Locks four repairs:
1. A foreground frontier advance supersedes the pending batches it
   retro-invalidates (``supersede_pending_batches`` was written but never
   wired — stranded ``ready`` batches jammed the worker gate forever).
2. The async worker tick reaps generation-stale pending batches instead of
   starving the conversation lane.
3. A successful promote supersedes sibling batches stranded below the new
   generation (previously each straggler cost one wasted
   promote-on-compress rejection per compress).
4. The foreground compress deadline auto-derives from the summary timeout
   (a 45s wall clock under a 300s summary budget guaranteed a one-pass
   truncation on every turn), and an explicit value still wins.
5. A batch swept mid-prepare cannot be resurrected to ``ready`` by its own
   straggler completion write.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from hermes_lcm.compaction import resolve_foreground_deadline_seconds
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _engine(
    tmp_path,
    *,
    session_id="starvation-session",
    conversation_id="starvation-conversation",
    worker=False,
):
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=worker,
        # High interval: the daemon thread is stopped immediately after
        # session start so manual ticks are fully deterministic.
        async_background_compaction_worker_interval_seconds=60.0,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=50_000,
    )
    if worker:
        # Deterministic tests: no daemon thread racing the manual ticks.
        engine._stop_async_worker()
    return engine


def _messages(count: int = 20, *, prefix="message") -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": "system prompt"}]
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(
            {"role": role, "content": f"{prefix} {idx} " + ("x " * 12)}
        )
    return messages


def _stub_leaf_summary(monkeypatch, engine, *, text="deterministic summary"):
    def fake_summarize(initial_chunk, focus_topic=None):
        source_tokens = max(1, len(initial_chunk) * 10)
        return (list(initial_chunk), source_tokens, text, 1, 1)

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_summarize)


def test_foreground_advance_supersedes_pending_batches(tmp_path, monkeypatch):
    """A frontier advance retro-invalidates ready batches; they must be swept."""
    engine = _engine(tmp_path)
    try:
        _stub_leaf_summary(monkeypatch, engine)
        messages = _messages()
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(messages)
        assert batch is not None and batch.state == "ready"

        now = time.time()
        node = SummaryNode(
            session_id=engine.current_session_id,
            depth=0,
            summary="foreground leaf summary",
            token_count=8,
            source_token_count=400,
            source_ids=list(batch.source_ids),
            source_type="messages",
            created_at=now,
            earliest_at=now,
            latest_at=now,
            expand_hint="",
        )
        node_id = engine._dag.add_node(node)

        new_gen = engine._note_foreground_async_frontier_advance(
            source_end_store_id=int(batch.source_end_store_id or 0),
            node_id=int(node_id),
            covered_source_ids=list(batch.source_ids),
            consumed_source_ids=list(batch.source_ids),
        )
        assert int(new_gen) >= 2

        swept = engine._frontier.get_batch(batch.batch_id)
        assert swept is not None
        assert swept.state == "superseded"
        assert swept.failure_reason == "foreground_compaction"
    finally:
        engine.shutdown()


def test_worker_tick_reaps_generation_stale_pending_batches(tmp_path, monkeypatch):
    """A ready batch stranded below the live generation must not starve the lane."""
    engine = _engine(tmp_path, worker=True)
    try:
        _stub_leaf_summary(monkeypatch, engine)
        messages = _messages()
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(messages)
        assert batch is not None and batch.state == "ready"

        # Simulate stranding: the frontier advanced past this batch after it
        # was prepared (restart, older build, or a race the sweep missed).
        conn: sqlite3.Connection = engine._frontier._conn
        conn.execute(
            "UPDATE lcm_prepared_batches SET base_generation = 0 WHERE batch_id = ?",
            (batch.batch_id,),
        )
        conn.commit()

        result = engine._async_worker_tick_body()

        swept = engine._frontier.get_batch(batch.batch_id)
        assert swept is not None
        assert swept.state == "superseded"
        assert swept.failure_reason == "stale_generation_reaped"

        # The lane un-jammed: the tick proceeded instead of returning early.
        assert result is True
        counts = engine._frontier.get_batch_counts_by_state(
            engine.current_conversation_id
        )
        assert counts.get("ready", 0) >= 1
        ready = engine._frontier.get_ready_batch(engine.current_conversation_id)
        assert ready is not None
        assert ready.batch_id != batch.batch_id
        assert ready.base_generation >= 1
    finally:
        engine.shutdown()


def test_promote_supersedes_sibling_batches_below_new_generation(tmp_path, monkeypatch):
    """A successful promote sweeps sibling stragglers stranded by the advance."""
    engine = _engine(tmp_path)
    try:
        _stub_leaf_summary(monkeypatch, engine)
        messages = _messages()
        engine.ingest(messages)

        older = engine.prepare_background_compaction_once(messages)
        assert older is not None and older.state == "ready"
        newer = engine.prepare_background_compaction_once(messages)
        assert newer is not None and newer.state == "ready"
        assert newer.batch_id != older.batch_id

        result = engine.promote_prepared_compaction(newer.batch_id, messages)
        assert result.promoted is True

        promoted_row = engine._frontier.get_batch(newer.batch_id)
        assert promoted_row.state == "promoted"

        swept = engine._frontier.get_batch(older.batch_id)
        assert swept is not None
        assert swept.state == "superseded"
        assert swept.failure_reason == "frontier_advanced"
    finally:
        engine.shutdown()


def test_foreground_deadline_auto_derives_from_summary_timeout():
    """Deadline defaults to at least one full summary round-trip plus margin."""
    default_config = LCMConfig()
    assert default_config.foreground_compress_deadline_seconds == 0.0

    # Default 60s summary budget: 60 + 15 margin beats the 45s floor.
    assert resolve_foreground_deadline_seconds(LCMConfig()) == 75.0

    # A 300s summary budget must never truncate a foreground pass at 45s.
    tuned = LCMConfig(summary_timeout_ms=300_000)
    assert resolve_foreground_deadline_seconds(tuned) == 315.0

    # Explicit values still win (existing tests pin 1.0 / 5.0 deadlines).
    explicit = LCMConfig(foreground_compress_deadline_seconds=1.5)
    assert resolve_foreground_deadline_seconds(explicit) == 1.5


def test_swept_batch_cannot_be_resurrected_to_ready(tmp_path):
    """A straggler prepare completion must not resurrect a swept batch."""
    engine = _engine(tmp_path)
    try:
        conv = engine.current_conversation_id
        sess = engine.current_session_id
        batch_id = engine._frontier.create_batch(
            conv, sess, 1, 10, "identity-hash", [1, 2, 3], "policy-fp", "route-fp",
            state="preparing",
        )

        # A frontier advance sweeps the in-flight batch mid-prepare.
        engine._frontier.update_batch_state(
            batch_id, "superseded", failure_reason="foreground_compaction"
        )

        # The straggler completion write (guarded by from_state) is a no-op.
        engine._frontier.update_batch_state(
            batch_id, "ready",
            from_state="preparing",
            summary_payload='{"summary_text": "late"}',
            payload_version=2,
        )
        row = engine._frontier.get_batch(batch_id)
        assert row.state == "superseded"

        # The normal preparing -> ready transition still works.
        live_id = engine._frontier.create_batch(
            conv, sess, 1, 10, "identity-hash", [4, 5], "policy-fp", "route-fp",
            state="preparing",
        )
        engine._frontier.update_batch_state(
            live_id, "ready",
            from_state="preparing",
            summary_payload='{"summary_text": "ok"}',
            payload_version=2,
        )
        assert engine._frontier.get_batch(live_id).state == "ready"
    finally:
        engine.shutdown()