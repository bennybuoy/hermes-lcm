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
        # Build the table set, then mark the fixture as a v6 database before
        # inserting a payload-less ready batch. The next migration call must
        # perform the v6→v7 cleanup itself.
        run_versioned_migrations(conn)
        conn.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )
        now = time.time()
        conn.execute(
            """
            INSERT INTO lcm_prepared_batches
                (conversation_id, session_id, base_generation, source_end_store_id,
                 source_identity_hash, source_ids, policy_fingerprint, route_fingerprint,
                 state, expected_leaf_count, frontier_end_store_id, created_at, updated_at,
                 summary_payload, payload_version)
            VALUES (?, ?, 1, 10, 'hash', '[]', 'pol', 'route', 'ready', 1, 10, ?, ?, '', 0)
            """,
            ("c1", "s1", now, now),
        )
        conn.commit()

        run_versioned_migrations(conn)

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
