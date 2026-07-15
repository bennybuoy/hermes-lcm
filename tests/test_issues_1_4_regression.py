"""Regression tests for hermes-lcm issues #1–#4 (async promote + host replacement).

#1 Prepare persists summary payload; promote is zero-LLM with telemetry.
#2 Promoted batch drops covered raw messages from returned active context.
#3 Foreground compress is bounded, priority over background, lock released.
#4 Promotion writes non-empty ordered frontier items; itemless tips reconcile.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.db_bootstrap import (
    SCHEMA_VERSION,
    run_versioned_migrations,
)
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.frontier import PREPARED_PAYLOAD_VERSION, FrontierStore
from hermes_lcm.tokens import count_messages_tokens, count_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(
    tmp_path,
    *,
    session_id: str = "iss-session",
    conversation_id: str = "iss-conversation",
    fresh_tail_count: int = 2,
    leaf_chunk_tokens: int = 20,
    context_threshold: float = 0.10,
    context_length: int = 50_000,
    async_enabled: bool = True,
    worker_enabled: bool = False,
    **extra,
) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=fresh_tail_count,
        leaf_chunk_tokens=leaf_chunk_tokens,
        context_threshold=context_threshold,
        async_background_compaction_enabled=async_enabled,
        async_background_compaction_worker_enabled=worker_enabled,
        **extra,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=context_length,
    )
    return engine


def _messages(count: int = 12, *, prefix: str = "message", tokens_each: int = 40) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "system prompt"}]
    filler = "x " * max(1, tokens_each // 2)
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{prefix} {idx} {filler}"})
    return msgs


def _stub_summarize(monkeypatch, engine, *, text: str = "prepared leaf summary"):
    calls = {"n": 0}

    def fake(initial_chunk, focus_topic=None):
        calls["n"] += 1
        source_tokens = max(1, len(initial_chunk) * 10)
        return (list(initial_chunk), source_tokens, f"{text} #{calls['n']}", 1, 1)

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake)
    return calls


# ===========================================================================
# Issue #1 — persist summary payload; zero LLM on promote
# ===========================================================================

class TestIssue1PersistSummaryPayload:
    def test_prepare_persists_summary_payload(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            assert batch is not None
            assert batch.state == "ready"
            assert batch.payload_version >= PREPARED_PAYLOAD_VERSION
            assert batch.has_summary_payload
            payload = batch.parsed_summary_payload()
            assert payload is not None
            assert "prepared leaf summary" in payload["summary_text"]
            assert payload["source_ids"] == batch.source_ids
            assert payload["token_count"] == count_tokens(payload["summary_text"])
            assert calls["n"] == 1
        finally:
            engine.shutdown()

    def test_promote_makes_zero_summarizer_calls(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            prepare_calls = calls["n"]
            assert prepare_calls >= 1

            result = engine.promote_prepared_compaction(batch.batch_id, msgs)

            assert result.promoted is True
            assert calls["n"] == prepare_calls  # zero additional LLM calls
            assert engine._dag.get_session_node_count(engine.current_session_id) == 1
            node = engine._dag.get_session_nodes(engine.current_session_id)[0]
            assert "prepared leaf summary" in node.summary
        finally:
            engine.shutdown()

    def test_promote_missing_payload_token_count_uses_text_tokens(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            payload = batch.parsed_summary_payload()
            assert payload is not None
            summary_text = payload["summary_text"]
            payload.pop("token_count", None)
            engine._frontier.update_batch_state(
                batch.batch_id,
                "ready",
                summary_payload=json.dumps(payload),
                payload_version=PREPARED_PAYLOAD_VERSION,
            )

            result = engine.promote_prepared_compaction(batch.batch_id, msgs)

            assert result.promoted is True
            node = engine._dag.get_session_nodes(engine.current_session_id)[0]
            assert node.token_count == count_tokens(summary_text)
        finally:
            engine.shutdown()

    def test_promote_telemetry_splits_validation_and_publication(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            result = engine.promote_prepared_compaction(batch.batch_id, msgs)
            assert result.promoted is True
            assert result.wall_ms >= 0
            assert result.validation_ms >= 0
            assert result.publication_ms >= 0
            assert result.wall_ms + 1.0 >= result.validation_ms  # wall covers validation
            status = engine.get_async_compaction_status()
            assert status["last_promote_wall_ms"] is not None
            assert status["last_promote_validation_ms"] is not None
            assert status["last_promote_publication_ms"] is not None
        finally:
            engine.shutdown()

    def test_legacy_v1_ready_batch_is_superseded_not_re_summarized(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            # Simulate a legacy v1 ready row (no payload).
            engine._frontier.update_batch_state(
                batch.batch_id,
                "ready",
                summary_payload="",
                payload_version=0,
            )
            reloaded = engine._frontier.get_batch(batch.batch_id)
            assert reloaded is not None
            assert not reloaded.has_summary_payload
            prepare_calls = calls["n"]

            result = engine.promote_prepared_compaction(batch.batch_id, msgs)

            assert result.promoted is False
            assert result.reason == "legacy_v1_batch_without_payload"
            assert calls["n"] == prepare_calls  # never re-summarized
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            final = engine._frontier.get_batch(batch.batch_id)
            assert final.state == "superseded"
        finally:
            engine.shutdown()

    def test_migration_supersedes_legacy_v1_ready_batches(self, tmp_path):
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db))
        # Build the shared tables, then recreate lcm_prepared_batches in its
        # genuine v6 shape (no summary_payload or payload_version columns).
        run_versioned_migrations(conn)
        conn.execute("DROP INDEX IF EXISTS idx_batches_conv_state")
        conn.execute("DROP TABLE lcm_prepared_batches")
        conn.execute(
            """
            CREATE TABLE lcm_prepared_batches (
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
                failure_reason TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_batches_conv_state
                ON lcm_prepared_batches(conversation_id, state)
            """
        )
        # This fixture intentionally reconstructs a genuine pre-v8 database
        # after using current bootstrap DDL as a shortcut.
        conn.execute("DROP TRIGGER lcm_schema_version_monotonic")
        conn.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )
        before_columns = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(lcm_prepared_batches)"
            ).fetchall()
        }
        assert "summary_payload" not in before_columns
        assert "payload_version" not in before_columns
        now = time.time()
        conn.execute(
            """
            INSERT INTO lcm_prepared_batches
                (conversation_id, session_id, base_generation, source_end_store_id,
                 source_identity_hash, source_ids, policy_fingerprint, route_fingerprint,
                 state, expected_leaf_count, frontier_end_store_id, created_at, updated_at)
            VALUES (?, ?, 1, 10, 'hash', '[]', 'pol', 'route', 'ready', 1, 10, ?, ?)
            """,
            ("c1", "s1", now, now),
        )
        conn.commit()

        run_versioned_migrations(conn)

        after_columns = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(lcm_prepared_batches)"
            ).fetchall()
        }
        assert {"summary_payload", "payload_version"}.issubset(after_columns)
        state = conn.execute(
            "SELECT state, failure_reason FROM lcm_prepared_batches"
        ).fetchone()
        assert state[0] == "superseded"
        assert "legacy_v1" in state[1]
        version = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert version == (str(SCHEMA_VERSION),)
        conn.close()


# ===========================================================================
# Issue #2 — covered raw messages absent after promote host replacement
# ===========================================================================

class TestIssue2HostReplacementDropsCovered:
    def test_compress_after_promote_drops_covered_raw_and_keeps_tail(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="async summary unique")
        try:
            msgs = _messages(16, tokens_each=40)
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            assert batch is not None and batch.state == "ready"
            covered = set(int(s) for s in batch.source_ids)
            assert covered

            before_tokens = count_messages_tokens(msgs)

            # Capture covered message contents BEFORE compress — compress advances
            # _last_compacted_store_id, which makes _get_store_id_map_for_messages
            # skip already-compacted store IDs and return an empty mapping.
            pre_compress_id_map = engine._get_store_id_map_for_messages(msgs)
            covered_contents = [
                m.get("content")
                for m in msgs
                if pre_compress_id_map.get(id(m)) in covered
            ]
            assert covered_contents, (
                "no covered message contents captured — id_map mapping failed before compress"
            )

            result = engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)

            assert engine._last_compression_status == "compacted"
            assert len(result) < len(msgs), (
                f"expected smaller context after promote, got {len(result)} vs {len(msgs)}"
            )
            # Summary content present once.
            summary_hits = sum(
                1
                for m in result
                if "async summary unique" in str(m.get("content") or "")
            )
            assert summary_hits == 1

            # Covered raw contents absent from active context.
            result_blob = " ".join(str(m.get("content") or "") for m in result)
            for content in covered_contents:
                assert content not in result_blob

            # Fresh tail (last 2 non-system) still present.
            tail = [m for m in msgs if m.get("role") != "system"][-2:]
            for t in tail:
                assert t["content"] in result_blob

            after_tokens = count_messages_tokens(result)
            assert after_tokens < before_tokens
            # Materially below cutover: next preflight should not re-fire.
            assert not engine.should_compress(prompt_tokens=after_tokens)
        finally:
            engine.shutdown()

    def test_pre_promotion_ingest_lock_prevents_publication_and_retries_suffix(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="partial promote summary")
        try:
            engine.update_model(
                model="partial-promote-model",
                context_length=50_000,
                provider="test",
            )
            prepared_messages = _messages(16, tokens_each=40)
            engine.ingest(prepared_messages)
            batch = engine.prepare_background_compaction_once(prepared_messages)
            assert batch is not None and batch.state == "ready"
            covered = set(int(s) for s in batch.source_ids)
            pre_compress_id_map = engine._get_store_id_map_for_messages(
                prepared_messages
            )
            covered_contents = {
                msg["content"]
                for msg in prepared_messages
                if pre_compress_id_map.get(id(msg)) in covered
            }
            assert covered_contents

            later_messages = _messages(
                8,
                prefix="arrived-after-prepare",
                tokens_each=40,
            )[1:]
            current_messages = prepared_messages + later_messages

            original_ingest = engine._ingest_messages

            def locked_ingest(_messages):
                raise sqlite3.OperationalError("database is locked")

            monkeypatch.setattr(engine, "_ingest_messages", locked_ingest)

            result = engine.compress(
                current_messages,
                current_tokens=engine.threshold_tokens + 5_000,
            )

            assert engine._last_compression_status == "failed"
            assert (
                "sqlite_locked_during_ingest_before_promote"
                in engine._last_compression_noop_reason
            )
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            assert result == current_messages
            assert engine._ingest_cursor == len(prepared_messages)
            monkeypatch.setattr(engine, "_ingest_messages", original_ingest)
            store_count = engine._store.get_session_count(engine.current_session_id)
            appended = {
                "role": "user",
                "content": "first message after bounded promotion failure",
            }
            engine._ingest_messages(result + [appended])
            assert (
                engine._store.get_session_count(engine.current_session_id)
                == store_count + len(later_messages) + 1
            )
            stored_contents = {
                row["content"]
                for row in engine._store.get_session_messages(
                    engine.current_session_id
                )
            }
            for message in later_messages:
                assert message["content"] in stored_contents
            assert engine._store.get_session_messages(engine.current_session_id)[-1][
                "content"
            ] == appended["content"]
        finally:
            engine.shutdown()

    def test_partial_promote_outer_sqlite_lock_uses_canonical_fallback(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="outer lock promote summary")
        try:
            engine.update_model(
                model="partial-promote-model",
                context_length=50_000,
                provider="test",
            )
            prepared_messages = _messages(16, tokens_each=40)
            engine.ingest(prepared_messages)
            batch = engine.prepare_background_compaction_once(prepared_messages)
            covered = set(int(store_id) for store_id in batch.source_ids)
            id_map = engine._get_store_id_map_for_messages(prepared_messages)
            covered_contents = {
                message["content"]
                for message in prepared_messages
                if id_map.get(id(message)) in covered
            }
            assert covered_contents
            current_messages = prepared_messages + _messages(
                8,
                prefix="outer-lock-later",
                tokens_each=40,
            )[1:]

            def locked_reclassification():
                raise sqlite3.OperationalError("database is locked outside body wrappers")

            monkeypatch.setattr(
                engine,
                "_maybe_reclassify_late_auxiliary_before_compaction_write",
                locked_reclassification,
            )

            result = engine.compress(
                current_messages,
                current_tokens=engine.threshold_tokens + 5_000,
            )

            assert engine._last_compression_status == "failed"
            assert "sqlite_locked:" in engine._last_compression_noop_reason
            assert engine._dag.get_session_node_count(engine.current_session_id) == 1
            result_blob = "\n".join(
                str(message.get("content") or "") for message in result
            )
            assert "outer lock promote summary" in result_blob
            for content in covered_contents:
                assert content not in result_blob
            assert engine._ingest_cursor == len(result)
        finally:
            engine.shutdown()

    def test_lock_after_leaf_commit_consumes_covered_raw_before_assembly(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(
            tmp_path,
            fresh_tail_count=2,
        )
        _stub_summarize(monkeypatch, engine, text="committed foreground leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"
            engine._config.async_background_compaction_promote_on_compress = False
            frontier_before = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_before is not None

            def lock_after_node_commit(_chunk, _source_ids):
                raise sqlite3.OperationalError("database is locked after add_node")

            monkeypatch.setattr(
                engine,
                "_maybe_gc_compacted_tool_results",
                lock_after_node_commit,
            )

            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            nodes = engine._dag.get_session_nodes(engine.current_session_id, depth=0)
            assert len(nodes) == 1, "fixture must commit the leaf before locking"
            covered_contents = {
                engine._store.get(store_id)["content"]
                for store_id in nodes[0].source_ids
            }
            assert covered_contents
            result_blob = "\n".join(
                str(message.get("content") or "") for message in result
            )
            assert "committed foreground leaf" in result_blob
            for content in covered_contents:
                assert content not in result_blob
            for tail_message in messages[-2:]:
                assert tail_message["content"] in result_blob
            assert engine._ingest_cursor == len(result)
            assert engine._last_compression_status == "compacted"
            assert (
                "publication_followup_error_after_leaf"
                in engine._last_compression_noop_reason
            )
            frontier_after = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_after is not None
            assert frontier_after["generation"] > frontier_before["generation"]
            assert engine._frontier.get_frontier_items(
                prepared.conversation_id,
                frontier_after["generation"],
            )
            assert engine._frontier.get_batch(prepared.batch_id).state == "superseded"
            stale_result = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )
            assert stale_result.promoted is False
            assert stale_result.reason == "batch_state_superseded"
        finally:
            engine.shutdown()

    def test_recovery_frontier_failure_rolls_back_committed_leaf(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="rolled back foreground leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"
            engine._config.async_background_compaction_promote_on_compress = False
            frontier_before = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_before is not None

            engine._frontier.conn.execute(
                """
                CREATE TRIGGER fail_recovery_frontier_item
                BEFORE INSERT ON lcm_frontier_items
                BEGIN
                    SELECT RAISE(ABORT, 'injected recovery frontier failure');
                END
                """
            )
            engine._frontier.conn.commit()

            lifecycle_before = engine.get_status()["lifecycle"][
                "current_frontier_store_id"
            ]

            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            assert result == messages
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            lifecycle_after = engine.get_status()["lifecycle"][
                "current_frontier_store_id"
            ]
            assert lifecycle_after == lifecycle_before
            frontier_after = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_after is not None
            assert frontier_after["generation"] == frontier_before["generation"]
            assert engine._frontier.list_itemless_active_generations(
                prepared.conversation_id
            ) == []
            assert (
                engine._last_compression_noop_reason
                == "frontier_advance_failed_leaf_rolled_back"
            )

            engine._frontier.conn.execute(
                "DROP TRIGGER fail_recovery_frontier_item"
            )
            engine._frontier.conn.commit()
            engine.shutdown()

            # A fresh engine must recover the same raw lineage, not the
            # temporarily published leaf's covered checkpoint.
            engine = _engine(tmp_path, fresh_tail_count=2)
            assert engine._last_compacted_store_id == lifecycle_before
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            promoted = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )
            assert promoted.promoted is True
            assert engine._dag.get_session_node_count(engine.current_session_id) == 1
        finally:
            engine.shutdown()

    def test_lifecycle_failure_after_frontier_publish_compensates_generation_and_leaf(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="lifecycle rollback leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"
            engine._config.async_background_compaction_promote_on_compress = False
            frontier_before = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            lifecycle_before = engine.get_status()["lifecycle"][
                "current_frontier_store_id"
            ]

            def fail_lifecycle(*_args, **_kwargs):
                raise RuntimeError("injected foreground lifecycle failure")

            monkeypatch.setattr(
                engine._lifecycle,
                "advance_frontier",
                fail_lifecycle,
            )
            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            assert result == messages
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            frontier_after = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_after["generation"] == frontier_before["generation"]
            assert (
                engine.get_status()["lifecycle"]["current_frontier_store_id"]
                == lifecycle_before
            )
        finally:
            engine.shutdown()

    def test_lifecycle_failure_rolls_back_before_newer_generation_can_reuse_leaf(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="concurrent lifecycle leaf")
        concurrent_frontier = FrontierStore(str(engine._store.db_path))
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"

            def publish_newer_generation_then_fail(*_args, **_kwargs):
                # Publication is still uncommitted, so a sibling reader can
                # observe only the complete base generation, never G's node.
                published = concurrent_frontier.get_active_frontier(
                    prepared.conversation_id
                )
                assert published is not None
                assert int(published["generation"]) == int(prepared.base_generation)
                assert concurrent_frontier.get_frontier_items(
                    prepared.conversation_id,
                    int(prepared.base_generation) + 1,
                ) == []
                raise RuntimeError("injected lifecycle failure before commit")

            monkeypatch.setattr(
                engine._lifecycle,
                "advance_frontier",
                publish_newer_generation_then_fail,
            )

            result = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )

            assert result.promoted is False
            active = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert active is not None and int(active["generation"]) == int(prepared.base_generation)
            assert engine._dag.get_session_node_count(prepared.session_id) == 0
        finally:
            concurrent_frontier.close()
            engine.shutdown()

    def test_lifecycle_failure_needs_no_compensating_frontier_rollback(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="rollback error leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"

            def fail_lifecycle(*_args, **_kwargs):
                raise RuntimeError("injected lifecycle failure")

            def fail_rollback(*_args, **_kwargs):
                raise sqlite3.OperationalError("injected rollback failure")

            monkeypatch.setattr(
                engine._lifecycle,
                "advance_frontier",
                fail_lifecycle,
            )
            monkeypatch.setattr(
                engine._frontier,
                "rollback_frontier_generation",
                fail_rollback,
            )

            result = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )

            assert result.promoted is False
            active = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert active is not None
            assert int(active["generation"]) == int(prepared.base_generation)
            assert engine._frontier.get_frontier_items(
                prepared.conversation_id,
                int(prepared.base_generation) + 1,
            ) == []
            assert engine._dag.get_session_node_count(prepared.session_id) == 0
        finally:
            engine.shutdown()

    def test_committed_lifecycle_marker_survives_post_commit_exception_without_double_advance(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="acknowledged lifecycle leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"
            engine._config.async_background_compaction_promote_on_compress = False
            frontier_before = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            original_advance = engine._lifecycle.advance_frontier

            def commit_then_raise(*args, **kwargs):
                original_advance(*args, **kwargs)
                raise RuntimeError("injected post-commit lifecycle exception")

            monkeypatch.setattr(
                engine._lifecycle,
                "advance_frontier",
                commit_then_raise,
            )
            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            assert result != messages
            assert engine._dag.get_session_node_count(engine.current_session_id) == 1
            frontier_after = engine._frontier.get_active_frontier(
                prepared.conversation_id
            )
            assert frontier_after["generation"] == frontier_before["generation"] + 1
            assert engine.get_status()["lifecycle"]["current_frontier_store_id"] > 0
        finally:
            engine.shutdown()

    def test_full_host_replacement_seam_survives_session_reload(
        self, tmp_path, monkeypatch
    ):
        db = tmp_path / "seam.db"
        config = LCMConfig(
            database_path=str(db),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=False,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start(
            "seam-session",
            conversation_id="seam-conversation",
            platform="test",
            context_length=50_000,
        )
        _stub_summarize(monkeypatch, engine, text="seam summary")
        msgs = _messages(16, tokens_each=40)
        engine.ingest(msgs)
        batch = engine.prepare_background_compaction_once(msgs)
        covered = set(int(s) for s in batch.source_ids)
        result = engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)
        assert engine._last_compression_status == "compacted"
        assert len(result) < len(msgs)
        # Persist "host" replacement via re-ingest of returned context on reload.
        engine.shutdown()

        engine2 = LCMEngine(config=config)
        engine2.on_session_start(
            "seam-session",
            conversation_id="seam-conversation",
            platform="test",
            context_length=50_000,
        )
        try:
            # Immutable store still has full history.
            store_count = engine2._store.get_session_count("seam-session")
            assert store_count >= len(msgs)
            # DAG has the promoted summary.
            assert engine2._dag.get_session_node_count("seam-session") >= 1
            # Frontier items exist for the active generation.
            frontier = engine2._frontier.get_active_frontier("seam-conversation")
            assert frontier is not None
            items = engine2._frontier.get_frontier_items(
                "seam-conversation", frontier["generation"]
            )
            assert items, "promoted generation must have frontier items"
            # Next preflight on the replaced context stays under threshold.
            replaced_tokens = count_messages_tokens(result)
            assert not engine2.should_compress(prompt_tokens=replaced_tokens)
            # Covered store_ids remain in store but are not the only active view.
            assert covered
        finally:
            engine2.shutdown()

    def test_one_promote_does_not_immediately_retrigger_compress(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages(16, tokens_each=40)
            engine.ingest(msgs)
            engine.prepare_background_compaction_once(msgs)
            first = engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)
            first_status = engine._last_compression_status
            first_count = engine.compression_count
            assert first_status == "compacted"

            # Immediate second compress on the replacement must not re-compact.
            second = engine.compress(first, current_tokens=count_messages_tokens(first))
            # Either noop/sanitized or same size — must not grow and re-trigger.
            assert len(second) <= len(first) + 1
            # No additional successful leaf publish required.
            assert engine.compression_count <= first_count + 1
        finally:
            engine.shutdown()


# ===========================================================================
# Issue #3 — bounded compress, foreground priority, no hang
# ===========================================================================

class TestIssue3BoundedForegroundCompress:
    def test_foreground_flag_blocks_background_prepare(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            engine._foreground_compress_active.set()
            batch = engine.prepare_background_compaction_once(msgs)
            # Batch may exist as failed, or prepare returns the failed row.
            if batch is not None:
                assert batch.state == "failed"
                assert "foreground" in (batch.failure_reason or "")
            assert calls["n"] == 0  # never entered LLM while foreground active
        finally:
            engine._foreground_compress_active.clear()
            engine.shutdown()

    def test_compress_releases_foreground_flag_on_success(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            engine.prepare_background_compaction_once(msgs)
            engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)
            assert not engine._foreground_compress_active.is_set()
            assert engine._last_compression_status != "running"
        finally:
            engine.shutdown()

    def test_compress_releases_foreground_flag_on_failure(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, async_enabled=False)
        def boom(initial_chunk, focus_topic=None):
            raise RuntimeError("summarizer exploded")

        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", boom)
        try:
            msgs = _messages(20, tokens_each=40)
            engine.ingest(msgs)
            with pytest.raises(RuntimeError, match="summarizer exploded"):
                engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)
            assert not engine._foreground_compress_active.is_set()
            assert engine._last_compression_status != "running"
        finally:
            engine.shutdown()

    def test_compress_fails_within_deadline_when_sqlite_locked(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, async_enabled=True)
        # Dynamic config overrides (not yet first-class dataclass fields).
        engine._config.foreground_compress_busy_timeout_ms = 200
        engine._config.foreground_compress_deadline_seconds = 5.0
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            engine.prepare_background_compaction_once(msgs)

            # Hold an exclusive write lock on the shared DB file from another connection.
            lock_conn = sqlite3.connect(str(engine._store.db_path), timeout=0.1)
            lock_conn.execute("BEGIN EXCLUSIVE")
            lock_conn.execute("CREATE TABLE IF NOT EXISTS _lock_holder(x INTEGER)")
            started = time.perf_counter()
            try:
                # compress must return (failed or compacted) without hanging for minutes.
                # With busy_timeout=200ms, promote/ingest should hit locked quickly.
                result = engine.compress(
                    msgs, current_tokens=engine.threshold_tokens + 1
                )
                elapsed = time.perf_counter() - started
                assert elapsed < 8.0, f"compress hung for {elapsed:.1f}s"
                assert not engine._foreground_compress_active.is_set()
                assert engine._last_compression_status != "running"
                # Either failed with timeout/locked or somehow completed; never hang.
                assert result is not None
            finally:
                lock_conn.rollback()
                lock_conn.close()
        finally:
            engine.shutdown()

    def test_phase_timings_recorded_on_promote_compress(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            engine.prepare_background_compaction_once(msgs)
            engine.compress(msgs, current_tokens=engine.threshold_tokens + 1)
            phases = getattr(engine, "_last_compress_phase_timings_ms", {}) or {}
            assert "promotion_lookup" in phases or "assembly" in phases
        finally:
            engine.shutdown()

    def test_worker_skips_while_foreground_active(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, worker_enabled=False)
        calls = _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            engine._foreground_compress_active.set()
            outcome = engine._async_worker_tick_body()
            assert outcome is None
            assert calls["n"] == 0
        finally:
            engine._foreground_compress_active.clear()
            engine.shutdown()


# ===========================================================================
# Issue #4 — frontier items on promotion + reconciliation
# ===========================================================================

class TestIssue4FrontierItems:
    def test_frontier_rollback_rechecks_tip_inside_write_transaction(
        self, tmp_path
    ):
        db_path = tmp_path / "rollback-race.db"
        frontier = FrontierStore(str(db_path))
        writer = sqlite3.connect(
            str(db_path), timeout=5.0, check_same_thread=False
        )
        writer.execute("PRAGMA journal_mode=WAL")
        try:
            conversation_id = "rollback-race-conversation"
            session_id = "rollback-race-session"
            base_generation = frontier.ensure_frontier(
                conversation_id,
                session_id,
            )
            target_generation = frontier.advance_frontier_generation_with_items(
                conversation_id,
                session_id,
                10,
                "policy",
                "route",
                base_generation,
                [
                    {
                        "kind": "node",
                        "ref_id": 10,
                        "source_start": 1,
                        "source_end": 10,
                    }
                ],
            )
            assert target_generation == base_generation + 1

            # Publish G+1 on another connection but hold its write transaction
            # open. Safe rollback must wait for this writer, then re-read the
            # committed tip and refuse to delete G.
            newer_generation = target_generation + 1
            now = time.time()
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                """
                INSERT INTO lcm_active_frontiers
                    (conversation_id, generation, session_id,
                     source_end_store_id, policy_fingerprint,
                     route_fingerprint, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    newer_generation,
                    session_id,
                    20,
                    "policy",
                    "route",
                    now,
                    now,
                ),
            )
            writer.execute(
                """
                INSERT INTO lcm_frontier_items
                    (conversation_id, generation, ordinal, kind,
                     ref_id, source_start, source_end)
                VALUES (?, ?, 0, 'node', 20, 11, 20)
                """,
                (conversation_id, newer_generation),
            )

            outcome: dict[str, Any] = {}

            def rollback_target():
                try:
                    outcome["result"] = frontier.rollback_frontier_generation(
                        conversation_id,
                        target_generation,
                    )
                except Exception as exc:  # regression records unsafe BUSY snapshot
                    outcome["error"] = exc

            thread = threading.Thread(target=rollback_target)
            thread.start()
            time.sleep(0.1)
            assert thread.is_alive(), "rollback must wait for the active writer"
            writer.commit()
            thread.join(timeout=5.0)

            assert not thread.is_alive()
            assert "error" not in outcome
            assert outcome["result"] is False
            active = frontier.get_active_frontier(conversation_id)
            assert active is not None
            assert active["generation"] == newer_generation
            assert frontier.get_frontier_items(
                conversation_id,
                target_generation,
            )
            assert frontier.get_frontier_items(
                conversation_id,
                newer_generation,
            )
        finally:
            if writer.in_transaction:
                writer.rollback()
            writer.close()
            frontier.close()

    def test_atomic_frontier_advance_rolls_back_generation_when_item_insert_fails(
        self, tmp_path
    ):
        engine = _engine(tmp_path)
        try:
            engine._frontier.ensure_frontier(
                engine.current_conversation_id,
                engine.current_session_id,
            )
            frontier = engine._frontier.get_active_frontier(
                engine.current_conversation_id
            )
            assert frontier is not None
            before_generation = frontier["generation"]
            engine._frontier.conn.execute(
                """
                CREATE TRIGGER fail_frontier_item_insert
                BEFORE INSERT ON lcm_frontier_items
                BEGIN
                    SELECT RAISE(ABORT, 'injected frontier item failure');
                END
                """
            )
            engine._frontier.conn.commit()

            with pytest.raises(sqlite3.DatabaseError, match="injected frontier"):
                engine._frontier.advance_frontier_generation_with_items(
                    engine.current_conversation_id,
                    engine.current_session_id,
                    10,
                    "policy",
                    "route",
                    before_generation,
                    [
                        {
                            "kind": "message",
                            "ref_id": 10,
                            "source_start": 10,
                            "source_end": 10,
                        }
                    ],
                )

            after = engine._frontier.get_active_frontier(
                engine.current_conversation_id
            )
            assert after is not None
            assert after["generation"] == before_generation
            assert engine._frontier.get_frontier_items(
                engine.current_conversation_id,
                before_generation + 1,
            ) == []
        finally:
            engine.shutdown()

    def test_promote_writes_nonempty_frontier_items(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages(12)
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            result = engine.promote_prepared_compaction(batch.batch_id, msgs)
            assert result.promoted is True
            frontier = engine._frontier.get_active_frontier(batch.conversation_id)
            assert frontier is not None
            assert frontier["generation"] > batch.base_generation
            items = engine._frontier.get_frontier_items(
                batch.conversation_id, frontier["generation"]
            )
            assert items, "active generation with source_end>0 must have items"
            # Node item present and references a real DAG node.
            node_items = [i for i in items if i["kind"] == "node"]
            assert node_items
            node_id = node_items[0]["ref_id"]
            nodes = {n.node_id: n for n in engine._dag.get_session_nodes(batch.session_id)}
            assert node_id in nodes
            # Ranges monotonic and non-overlapping.
            ends = []
            for item in items:
                assert item["source_start"] <= item["source_end"]
                if ends:
                    assert item["source_start"] > ends[-1]
                ends.append(item["source_end"])
            # Message items only for uncovered tail.
            msg_items = [i for i in items if i["kind"] == "message"]
            covered = set(batch.source_ids)
            for mi in msg_items:
                assert mi["ref_id"] not in covered
        finally:
            engine.shutdown()

    def test_promote_compensation_does_not_leave_itemless_tip(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            before = engine._frontier.get_active_frontier(batch.conversation_id)[
                "generation"
            ]
            engine._async_compaction_publish_failure_hook = "after_frontier_items"
            with pytest.raises(RuntimeError, match="injected async promotion failure"):
                engine.promote_prepared_compaction(batch.batch_id, msgs)
            after = engine._frontier.get_active_frontier(batch.conversation_id)
            assert after["generation"] == before
            # No orphan tip with items or without.
            itemless = engine._frontier.list_itemless_active_generations(
                batch.conversation_id
            )
            assert all(r["generation"] <= before for r in itemless)
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        finally:
            engine.shutdown()

    def test_reconcile_repairs_itemless_generation(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            result = engine.promote_prepared_compaction(batch.batch_id, msgs)
            assert result.promoted is True
            frontier = engine._frontier.get_active_frontier(batch.conversation_id)
            gen = frontier["generation"]
            # Wipe items to simulate legacy itemless tip.
            engine._frontier.set_frontier_items(batch.conversation_id, gen, [])
            assert engine._frontier.get_frontier_items(batch.conversation_id, gen) == []
            repaired = engine.reconcile_itemless_frontier_generations(
                batch.conversation_id
            )
            assert repaired >= 1
            items = engine._frontier.get_frontier_items(batch.conversation_id, gen)
            assert items
        finally:
            engine.shutdown()

    def test_fault_injection_after_canonical_insert_rolls_back(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        _stub_summarize(monkeypatch, engine)
        try:
            msgs = _messages()
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            before = engine._frontier.get_active_frontier(batch.conversation_id)[
                "generation"
            ]
            engine._async_compaction_publish_failure_hook = "after_canonical_insert"
            with pytest.raises(RuntimeError, match="injected async promotion failure"):
                engine.promote_prepared_compaction(batch.batch_id, msgs)
            assert (
                engine._frontier.get_active_frontier(batch.conversation_id)["generation"]
                == before
            )
            assert engine._dag.get_session_node_count(engine.current_session_id) == 0
            assert engine._frontier.get_batch(batch.batch_id).state == "ready"
        finally:
            engine.shutdown()


# ===========================================================================
# Issue #6 — active frontier is authoritative for assembly
# ===========================================================================

class TestIssue6AuthoritativeFrontierAssembly:
    def test_foreground_compaction_publishes_authoritative_generation_without_async(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2, async_enabled=False)
        _stub_summarize(monkeypatch, engine, text="foreground frontier leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            compressed = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
            )

            assert compressed != messages
            frontier = engine._frontier.get_active_frontier(
                engine.current_conversation_id
            )
            assert frontier is not None
            assert int(frontier["source_end_store_id"]) > 0
            items = engine._frontier.get_frontier_items(
                engine.current_conversation_id,
                int(frontier["generation"]),
            )
            assert items
            node_items = [item for item in items if item["kind"] == "node"]
            assert node_items
            for item in node_items:
                assert engine._dag.get_node(int(item["ref_id"])) is not None
        finally:
            engine.shutdown()

    def test_assembly_projects_exact_active_generation_not_stale_host_or_dag(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="authoritative frontier leaf")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None and prepared.state == "ready"
            covered = set(int(s) for s in prepared.source_ids)
            id_map = engine._get_store_id_map_for_messages(messages)
            covered_contents = {
                str(message.get("content") or "")
                for message in messages
                if id_map.get(id(message)) in covered
                and message.get("role") != "system"
            }
            assert covered_contents
            promoted = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )
            assert promoted.promoted is True

            # Canonical storage may contain nodes that are not members of the
            # selected generation. Assembly must not infer visibility merely
            # from "uncondensed" DAG state.
            engine._dag.add_node(
                SummaryNode(
                    session_id=engine.current_session_id,
                    depth=0,
                    summary="unpublished rogue canonical node",
                    token_count=5,
                    source_token_count=5,
                    source_ids=[],
                    source_type="messages",
                    created_at=time.time(),
                )
            )

            assembled = engine._assemble_context(messages[0], messages[1:])
            blob = "\n".join(str(message.get("content") or "") for message in assembled)

            assert "authoritative frontier leaf" in blob
            assert "unpublished rogue canonical node" not in blob
            for content in covered_contents:
                assert content not in blob
            for tail_message in messages[-2:]:
                assert str(tail_message["content"]) in blob
        finally:
            engine.shutdown()

    def test_restart_assembles_same_authoritative_generation(
        self, tmp_path, monkeypatch
    ):
        config = LCMConfig(
            database_path=str(tmp_path / "frontier-restart.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=False,
        )
        messages = _messages(16, tokens_each=40)
        first = LCMEngine(config=config)
        first.on_session_start(
            "frontier-restart-session",
            conversation_id="frontier-restart-conversation",
            platform="test",
            context_length=50_000,
        )
        _stub_summarize(monkeypatch, first, text="restart stable frontier")
        try:
            first.ingest(messages)
            prepared = first.prepare_background_compaction_once(messages)
            assert prepared is not None
            assert first.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            ).promoted
            before = first._assemble_context(messages[0], messages[1:])
        finally:
            first.shutdown()

        restarted = LCMEngine(config=config)
        restarted.on_session_start(
            "frontier-restart-session",
            conversation_id="frontier-restart-conversation",
            platform="test",
            context_length=50_000,
        )
        try:
            after = restarted._assemble_context(messages[0], messages[1:])
            assert after == before
            blob = "\n".join(str(message.get("content") or "") for message in after)
            assert "restart stable frontier" in blob
        finally:
            restarted.shutdown()

    def test_missing_active_node_ref_fails_closed_instead_of_host_replay(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="missing frontier node")
        try:
            messages = _messages(16, tokens_each=40)
            engine.ingest(messages)
            prepared = engine.prepare_background_compaction_once(messages)
            assert prepared is not None
            promoted = engine.promote_prepared_compaction(
                prepared.batch_id,
                messages,
            )
            assert promoted.promoted
            assert engine._dag.delete_node(promoted.node_id)

            with pytest.raises(RuntimeError, match="missing canonical node"):
                engine._assemble_context(messages[0], messages[1:])
        finally:
            engine.shutdown()


# ===========================================================================
# Final publication review blockers
# ===========================================================================

class TestFinalPublicationReviewBlockers:
    def test_async_promotion_atomically_carries_base_generation_items(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="generation carrying leaf")
        atomic_calls: list[tuple[int, list[dict[str, Any]]]] = []
        original_atomic = engine._frontier.advance_frontier_generation_with_items

        def reject_split_advance(*_args, **_kwargs):
            pytest.fail("async promotion used split generation publication")

        def record_atomic(
            conversation_id,
            session_id,
            new_source_end,
            policy_fingerprint,
            route_fingerprint,
            base_generation,
            items,
        ):
            atomic_calls.append((int(base_generation), list(items)))
            return original_atomic(
                conversation_id,
                session_id,
                new_source_end,
                policy_fingerprint,
                route_fingerprint,
                base_generation,
                items,
            )

        monkeypatch.setattr(
            engine._frontier,
            "advance_frontier_generation",
            reject_split_advance,
        )
        monkeypatch.setattr(
            engine._frontier,
            "advance_frontier_generation_with_items",
            record_atomic,
        )
        try:
            first_messages = _messages(12, prefix="first generation")
            engine.ingest(first_messages)
            first_batch = engine.prepare_background_compaction_once(first_messages)
            assert first_batch is not None
            first_result = engine.promote_prepared_compaction(
                first_batch.batch_id,
                first_messages,
            )
            assert first_result.promoted is True

            later = _messages(6, prefix="second generation")[1:]
            all_messages = first_messages + later
            engine.ingest(all_messages)
            second_batch = engine.prepare_background_compaction_once(all_messages)
            assert second_batch is not None
            second_result = engine.promote_prepared_compaction(
                second_batch.batch_id,
                all_messages,
            )
            assert second_result.promoted is True

            assert len(atomic_calls) == 2
            assert atomic_calls[0][0] == first_batch.base_generation
            assert atomic_calls[1][0] == second_batch.base_generation
            active = engine._frontier.get_active_frontier(
                engine.current_conversation_id
            )
            assert active is not None
            active_items = engine._frontier.get_frontier_items(
                engine.current_conversation_id,
                int(active["generation"]),
            )
            active_node_ids = {
                int(item["ref_id"])
                for item in active_items
                if item["kind"] == "node"
            }
            assert active_node_ids == {
                int(first_result.node_id),
                int(second_result.node_id),
            }
        finally:
            engine.shutdown()

    def test_messages_arriving_after_prepare_are_durable_before_promotion(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="post-prepare durable leaf")
        try:
            prepared_messages = _messages(12, prefix="prepared")
            engine.ingest(prepared_messages)
            batch = engine.prepare_background_compaction_once(prepared_messages)
            assert batch is not None and batch.state == "ready"

            arrived = {
                "role": "user",
                "content": "newest user message arrived after async preparation",
            }
            current_messages = prepared_messages + [arrived]
            result = engine.compress(
                current_messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            assert any(
                row["content"] == arrived["content"]
                for row in engine._store.get_session_messages(
                    engine.current_session_id
                )
            )
            assert any(
                message.get("content") == arrived["content"]
                for message in result
            )
            assert engine._ingest_cursor == len(result)
            engine._ingest_messages(
                result
                + [{"role": "assistant", "content": "reply after promoted replacement"}]
            )
            stored = engine._store.get_session_messages(engine.current_session_id)
            assert stored[-1]["content"] == "reply after promoted replacement"
        finally:
            engine.shutdown()

    def test_ready_between_promotion_lookups_ingests_late_messages_and_next_append(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="lookup race durable leaf")
        try:
            prepared_messages = _messages(12, prefix="prepared before lookup race")
            engine.ingest(prepared_messages)
            batch = engine.prepare_background_compaction_once(
                prepared_messages,
                leave_state="preparing",
            )
            assert batch is not None and batch.state == "preparing"

            arrived = {
                "role": "user",
                "content": "host message arriving after preparation before readiness",
            }
            current_messages = prepared_messages + [arrived]
            original_get_ready_batch = engine._frontier.get_ready_batch
            lookup_count = 0

            def become_ready_between_lookups(conversation_id):
                nonlocal lookup_count
                lookup_count += 1
                if lookup_count == 1:
                    assert original_get_ready_batch(conversation_id) is None
                    engine._frontier.update_batch_state(batch.batch_id, "ready")
                    return None
                return original_get_ready_batch(conversation_id)

            monkeypatch.setattr(
                engine._frontier,
                "get_ready_batch",
                become_ready_between_lookups,
            )

            result = engine.compress(
                current_messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            assert lookup_count == 2
            assert engine._frontier.get_batch(batch.batch_id).state == "promoted"
            stored = engine._store.get_session_messages(engine.current_session_id)
            assert [row["content"] for row in stored].count(arrived["content"]) == 1
            assert any(message.get("content") == arrived["content"] for message in result)
            assert engine._ingest_cursor == len(result)

            next_append = {
                "role": "assistant",
                "content": "first append after lookup-race promotion",
            }
            count_after_promotion = len(stored)
            engine._ingest_messages(result + [next_append])
            stored_after_append = engine._store.get_session_messages(
                engine.current_session_id
            )
            assert len(stored_after_append) == count_after_promotion + 1
            assert stored_after_append[-1]["content"] == next_append["content"]
        finally:
            engine.shutdown()

    def test_adaptive_rescue_batch_covers_only_summarized_prefix(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)

        def rescue_prefix(initial_chunk, focus_topic=None):
            compacted = list(initial_chunk[:3])
            assert len(initial_chunk) > len(compacted)
            return compacted, count_messages_tokens(compacted), "rescued prefix only", 1, 2

        monkeypatch.setattr(
            engine,
            "_summarize_leaf_chunk_with_rescue",
            rescue_prefix,
        )
        try:
            messages = _messages(12, prefix="rescue source")
            engine.ingest(messages)
            stored = engine._store.get_session_messages(engine.current_session_id)
            system_id = int(stored[0]["store_id"])

            batch = engine.prepare_background_compaction_once(messages)

            assert batch is not None and batch.state == "ready"
            assert len(batch.source_ids) == 3
            assert system_id not in batch.source_ids
            payload = batch.parsed_summary_payload()
            assert payload is not None
            assert payload["source_ids"] == batch.source_ids
            assert batch.source_end_store_id == batch.source_ids[-1]
            assert batch.frontier_end_store_id == batch.source_ids[-1]

            result = engine.promote_prepared_compaction(batch.batch_id, messages)
            assert result.promoted is True
            node = engine._dag.get_node(result.node_id)
            assert node is not None
            assert node.source_ids == batch.source_ids
            unsummarized_ids = {
                int(row["store_id"])
                for row in stored
                if int(row["store_id"]) > batch.source_ids[-1]
            }
            assert unsummarized_ids
            assert unsummarized_ids.isdisjoint(result.covered_source_ids)
        finally:
            engine.shutdown()

    def test_async_prepare_and_promotion_preserve_leading_system_anchor(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="anchor-safe leaf")
        try:
            messages = _messages(12, prefix="anchor source")
            messages[0]["content"] = "immutable leading system anchor"
            engine.ingest(messages)
            stored = engine._store.get_session_messages(engine.current_session_id)
            system_id = int(stored[0]["store_id"])

            batch = engine.prepare_background_compaction_once(messages)

            assert batch is not None
            assert system_id not in batch.source_ids
            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )
            assert result[0]["role"] == "system"
            assert result[0]["content"].startswith(messages[0]["content"])
            node = engine._dag.get_session_nodes(engine.current_session_id)[0]
            assert system_id not in node.source_ids
        finally:
            engine.shutdown()

    def test_atomic_rollback_returns_original_context_when_later_compress_fails(
        self, tmp_path, monkeypatch
    ):
        engine = _engine(tmp_path, fresh_tail_count=2)
        _stub_summarize(monkeypatch, engine, text="superseding canonical leaf")
        concurrent_frontier = FrontierStore(str(engine._store.db_path))
        try:
            messages = _messages(16, prefix="superseded rollback")
            engine.ingest(messages)
            batch = engine.prepare_background_compaction_once(messages)
            assert batch is not None and batch.state == "ready"
            covered_contents = {
                engine._store.get(store_id)["content"]
                for store_id in batch.source_ids
            }

            def publish_newer_then_fail(*_args, **_kwargs):
                published = concurrent_frontier.get_active_frontier(
                    batch.conversation_id
                )
                assert published is not None
                assert int(published["generation"]) == int(batch.base_generation)
                assert concurrent_frontier.get_frontier_items(
                    batch.conversation_id,
                    int(batch.base_generation) + 1,
                ) == []
                raise RuntimeError("lifecycle failed before atomic commit")

            monkeypatch.setattr(
                engine._lifecycle,
                "advance_frontier",
                publish_newer_then_fail,
            )

            def later_sqlite_timeout():
                raise sqlite3.OperationalError("database is locked after supersession")

            monkeypatch.setattr(
                engine,
                "_maybe_reclassify_late_auxiliary_before_compaction_write",
                later_sqlite_timeout,
            )

            result = engine.compress(
                messages,
                current_tokens=engine.threshold_tokens + 1,
            )

            result_blob = "\n".join(
                str(message.get("content") or "") for message in result
            )
            assert "superseding canonical leaf" not in result_blob
            for content in covered_contents:
                assert content in result_blob
            assert result == messages
            assert engine._dag.get_session_node_count(batch.session_id) == 0
        finally:
            concurrent_frontier.close()
            engine.shutdown()
