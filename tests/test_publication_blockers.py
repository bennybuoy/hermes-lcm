"""Deterministic regressions for canonical publication concurrency blockers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import threading

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.maintenance import flush_engine_connections


SESSION_ID = "publication-blocker-session"
CONVERSATION_ID = "publication-blocker-conversation"


def _messages(count: int = 12) -> list[dict[str, str]]:
    rows = [{"role": "system", "content": "system prompt"}]
    for index in range(count):
        rows.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"publication source {index} " + ("x " * 20),
            }
        )
    return rows


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "publication-blockers.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=False,
        )
    )
    engine.on_session_start(
        SESSION_ID,
        conversation_id=CONVERSATION_ID,
        platform="test",
        context_length=50_000,
    )
    return engine


def _prepare(engine: LCMEngine, messages: list[dict[str, str]], monkeypatch):
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda initial_chunk, focus_topic=None: (
            list(initial_chunk),
            100,
            "async winner summary",
            1,
            1,
        ),
    )
    engine.ingest(messages)
    batch = engine.prepare_background_compaction_once(messages)
    assert batch is not None and batch.state == "ready"
    return batch


def test_async_and_foreground_publishers_have_one_sqlite_lock_winner(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    messages = _messages()
    batch = _prepare(engine, messages, monkeypatch)

    async_has_writer = threading.Event()
    foreground_attempting_publication = threading.Event()
    original_publication_transaction = engine._frontier.publication_transaction

    @contextmanager
    def observed_publication_transaction():
        if threading.current_thread().name == "foreground-publisher":
            foreground_attempting_publication.set()
        with original_publication_transaction() as conn:
            yield conn

    def async_boundary(phase: str):
        if phase == "after_begin":
            async_has_writer.set()
            assert foreground_attempting_publication.wait(timeout=2)

    monkeypatch.setattr(
        engine._frontier,
        "publication_transaction",
        observed_publication_transaction,
    )
    engine._async_compaction_publish_failure_hook = async_boundary
    promotion_result = {}
    foreground_result = {}

    def promote():
        promotion_result["value"] = engine.promote_prepared_compaction(
            batch.batch_id, messages
        )

    def foreground():
        foreground_result["value"] = engine._publish_foreground_leaf(
            node=SummaryNode(
                session_id=SESSION_ID,
                depth=0,
                summary="foreground loser summary",
                token_count=10,
                source_token_count=100,
                source_ids=list(batch.source_ids),
                source_type="messages",
                created_at=1.0,
            ),
            source_end_store_id=int(batch.frontier_end_store_id),
            covered_source_ids=list(batch.source_ids),
        )

    async_thread = threading.Thread(target=promote, name="async-publisher")
    foreground_thread = threading.Thread(
        target=foreground, name="foreground-publisher"
    )
    try:
        async_thread.start()
        assert async_has_writer.wait(timeout=2)
        foreground_thread.start()
        async_thread.join(timeout=3)
        foreground_thread.join(timeout=3)
        assert not async_thread.is_alive()
        assert not foreground_thread.is_alive()
        assert promotion_result["value"].promoted is True
        assert foreground_result["value"]["published"] is False
        assert foreground_result["value"]["reason"] == "canonical_source_overlap"

        nodes = engine._dag.get_session_nodes(SESSION_ID)
        active = engine._frontier.get_active_frontier(CONVERSATION_ID)
        assert active is not None
        active_node_ids = {
            int(item["ref_id"])
            for item in engine._frontier.get_frontier_items(
                CONVERSATION_ID, int(active["generation"])
            )
            if item["kind"] == "node"
        }
        assert len(nodes) == 1
        assert active_node_ids == {int(nodes[0].node_id)}
        assert "async winner summary" in nodes[0].summary
        grep = json.loads(
            engine.handle_tool_call(
                "lcm_grep", {"query": "foreground loser summary"}
            )
        )
        assert grep.get("results", []) == []
    finally:
        engine._async_compaction_publish_failure_hook = None
        engine.shutdown()


def test_maintenance_flush_cannot_commit_publication_owned_by_another_thread(
    tmp_path,
):
    engine = _engine(tmp_path)
    transaction_open = threading.Event()
    release_publication = threading.Event()
    maintenance_done = threading.Event()
    publication_errors: list[BaseException] = []

    def publication_owner():
        try:
            with engine._frontier.publication_transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO summary_nodes
                        (session_id, depth, summary, token_count, source_token_count,
                         source_ids, source_type, created_at)
                    VALUES (?, 0, ?, 1, 1, '[]', 'messages', 1.0)
                    """,
                    (SESSION_ID, "must roll back"),
                )
                transaction_open.set()
                assert release_publication.wait(timeout=2)
                raise RuntimeError("rollback owner transaction")
        except RuntimeError as exc:
            publication_errors.append(exc)

    def maintenance():
        flush_engine_connections(engine)
        maintenance_done.set()

    owner = threading.Thread(target=publication_owner, name="publication-owner")
    flusher = threading.Thread(target=maintenance, name="maintenance-flusher")
    try:
        owner.start()
        assert transaction_open.wait(timeout=2)
        flusher.start()
        release_publication.set()
        owner.join(timeout=2)
        flusher.join(timeout=2)
        assert not owner.is_alive()
        assert not flusher.is_alive()
        assert maintenance_done.is_set()
        assert publication_errors
        assert engine._dag.get_session_nodes(SESSION_ID) == []
    finally:
        release_publication.set()
        owner.join(timeout=2)
        flusher.join(timeout=2)
        engine.shutdown()


def test_publication_does_not_take_lifecycle_lock_after_sqlite_writer_lock(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    messages = _messages()
    batch = _prepare(engine, messages, monkeypatch)
    writer_locked = threading.Event()
    lifecycle_lock_held = threading.Event()
    promotion_done = threading.Event()
    lifecycle_done = threading.Event()
    results = {}

    engine._lifecycle._conn.execute("PRAGMA busy_timeout=800")

    def publication_boundary(phase: str):
        if phase == "after_begin":
            writer_locked.set()
            assert lifecycle_lock_held.wait(timeout=2)

    engine._async_compaction_publish_failure_hook = publication_boundary

    def promote():
        results["promotion"] = engine.promote_prepared_compaction(
            batch.batch_id, messages
        )
        promotion_done.set()

    def lifecycle_writer():
        assert writer_locked.wait(timeout=2)
        conn = engine._lifecycle._conn
        with engine._lifecycle._lock:
            lifecycle_lock_held.set()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                results["lifecycle_error"] = exc
            else:
                conn.rollback()
        lifecycle_done.set()

    promotion = threading.Thread(target=promote, name="promotion")
    lifecycle = threading.Thread(target=lifecycle_writer, name="lifecycle")
    try:
        promotion.start()
        lifecycle.start()
        assert promotion_done.wait(timeout=0.30), (
            "publication waited on lifecycle lock while holding SQLite writer lock"
        )
        assert lifecycle_done.wait(timeout=1.5)
        promotion.join(timeout=1)
        lifecycle.join(timeout=1)
        assert results["promotion"].promoted is True
        assert "lifecycle_error" not in results
    finally:
        engine._async_compaction_publish_failure_hook = None
        promotion.join(timeout=2)
        lifecycle.join(timeout=2)
        engine.shutdown()


@pytest.mark.parametrize("mutation", ["delete", "reassign", "rewrite"])
def test_source_identity_is_revalidated_under_publication_writer_lock(
    tmp_path, monkeypatch, mutation
):
    engine = _engine(tmp_path)
    messages = _messages()
    batch = _prepare(engine, messages, monkeypatch)
    validation_complete = threading.Event()
    mutation_complete = threading.Event()
    original_hash = engine_module.compute_source_identity_hash
    calls = 0

    def synchronize_after_optimistic_hash(conn, session_id, source_ids):
        nonlocal calls
        calls += 1
        value = original_hash(conn, session_id, source_ids)
        if calls == 1:
            validation_complete.set()
            assert mutation_complete.wait(timeout=2)
        return value

    monkeypatch.setattr(
        engine_module,
        "compute_source_identity_hash",
        synchronize_after_optimistic_hash,
    )

    def mutate_source():
        assert validation_complete.wait(timeout=2)
        conn = sqlite3.connect(str(engine._store.db_path), timeout=2)
        try:
            source_id = int(batch.source_ids[0])
            if mutation == "delete":
                conn.execute("DELETE FROM messages WHERE store_id = ?", (source_id,))
            elif mutation == "reassign":
                conn.execute(
                    "UPDATE messages SET session_id = 'reassigned-session' WHERE store_id = ?",
                    (source_id,),
                )
            else:
                conn.execute(
                    "UPDATE messages SET content = content || ' rewritten' WHERE store_id = ?",
                    (source_id,),
                )
            conn.commit()
        finally:
            conn.close()
            mutation_complete.set()

    mutator = threading.Thread(target=mutate_source, name=f"source-{mutation}")
    try:
        mutator.start()
        result = engine.promote_prepared_compaction(batch.batch_id, messages)
        mutator.join(timeout=2)
        assert not mutator.is_alive()
        assert calls >= 2
        assert result.promoted is False
        assert result.reason == "source_identity_mismatch"
        assert engine._dag.get_session_nodes(SESSION_ID) == []
        active = engine._frontier.get_active_frontier(CONVERSATION_ID)
        assert active is not None
        assert int(active["generation"]) == int(batch.base_generation)
    finally:
        mutation_complete.set()
        mutator.join(timeout=2)
        engine.shutdown()
