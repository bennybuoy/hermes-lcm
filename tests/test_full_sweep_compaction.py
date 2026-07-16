"""Issue #8: bounded full-sweep candidate construction and publication."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

import hermes_lcm.sweep as sweep_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _messages(count: int = 20) -> list[dict]:
    messages = [{"role": "system", "content": "system anchor"}]
    for index in range(count):
        messages.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"turn {index} " + ("payload " * 24),
            }
        )
    return messages


def _engine(tmp_path, **overrides) -> LCMEngine:
    values = dict(
        database_path=str(tmp_path / "sweep.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=32,
        condensation_fanin=4,
        condensation_min_fanin=2,
        incremental_max_depth=3,
        full_sweep_compaction_enabled=True,
        full_sweep_max_passes=64,
        full_sweep_deadline_seconds=30.0,
        summary_prefix_target_tokens=8,
        context_threshold=0.10,
    )
    values.update(overrides)
    engine = LCMEngine(config=LCMConfig(**values))
    engine.on_session_start(
        "sweep-session",
        conversation_id="sweep-conversation",
        platform="test",
        context_length=2_000,
    )
    return engine


def _stub_summaries(monkeypatch, engine):
    calls = {"leaf": 0, "condense": 0}

    def leaf(chunk, focus_topic=None, timeout_seconds=None):
        calls["leaf"] += 1
        return list(chunk), 100, f"leaf-{calls['leaf']}", 1, 1

    def condense(**kwargs):
        calls["condense"] += 1
        return f"parent-{calls['condense']}", 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", leaf)
    monkeypatch.setattr(sweep_module, "summarize_with_escalation", condense)
    return calls


def _raw_descendants(engine, node_id: int) -> set[int]:
    node = engine._dag.get_node(node_id)
    assert node is not None
    if node.source_type == "messages":
        return set(node.source_ids)
    result: set[int] = set()
    for child_id in node.source_ids:
        result.update(_raw_descendants(engine, child_id))
    return result


def test_full_sweep_builds_16_leaves_depth2_and_publishes_once(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = _stub_summaries(monkeypatch, engine)
    publication_calls = []
    original = engine._frontier.publish_generation_state_no_commit

    def counted(*args, **kwargs):
        publication_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._frontier, "publish_generation_state_no_commit", counted)
    try:
        result = engine.compress(_messages(20), current_tokens=1_900)

        nodes = engine._dag.get_session_nodes("sweep-session")
        leaves = [node for node in nodes if node.depth == 0]
        depth2 = [node for node in nodes if node.depth == 2]
        assert len(leaves) >= 16
        assert depth2
        assert calls["condense"] >= 5
        assert len(publication_calls) == 1

        frontier = engine._frontier.get_active_frontier("sweep-conversation")
        items = engine._frontier.get_frontier_items(
            "sweep-conversation", frontier["generation"]
        )
        node_items = [item for item in items if item["kind"] == "node"]
        assert node_items
        covered = set()
        for item in node_items:
            covered.update(_raw_descendants(engine, item["ref_id"]))
        stored = engine._store.get_session_messages("sweep-session")
        returned_text = "\n".join(str(message.get("content", "")) for message in result)
        assert covered
        assert all(any(row["store_id"] == source_id for row in stored) for source_id in covered)
        assert "turn 0" not in returned_text
        assert engine._last_full_sweep_status["publication_count"] == 1
        assert engine.get_status()["full_sweep"]["leaf_count"] >= 16
    finally:
        engine.shutdown()


def test_full_sweep_pass_limit_publishes_one_consistent_partial_candidate(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, full_sweep_max_passes=3)
    _stub_summaries(monkeypatch, engine)
    try:
        engine.compress(_messages(20), current_tokens=1_900)

        status = engine._last_full_sweep_status
        assert status["reason"] == "pass-limit"
        assert status["partial"] is True
        assert status["passes"] == 3
        assert status["publication_count"] == 1
        frontier = engine._frontier.get_active_frontier("sweep-conversation")
        assert engine._frontier.get_frontier_items(
            "sweep-conversation", frontier["generation"]
        )
    finally:
        engine.shutdown()


def test_full_sweep_deadline_is_operation_wide_and_keeps_partial_source_closure(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, full_sweep_deadline_seconds=0.01)
    calls = _stub_summaries(monkeypatch, engine)
    clock = {"value": 10.0}

    def monotonic():
        value = clock["value"]
        clock["value"] += 0.004
        return value

    monkeypatch.setattr(sweep_module.time, "monotonic", monotonic)
    try:
        engine.compress(_messages(20), current_tokens=1_900)

        status = engine._last_full_sweep_status
        assert status["reason"] == "deadline"
        assert status["partial"] is True
        assert status["publication_count"] in {0, 1}
        assert calls["leaf"] <= 3
        for node in engine._dag.get_session_nodes("sweep-session"):
            if node.source_type == "nodes":
                assert all(engine._dag.get_node(source_id) is not None for source_id in node.source_ids)
    finally:
        engine.shutdown()


def test_full_sweep_detects_non_reducing_no_progress_without_publication(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)

    def non_reducing(chunk, focus_topic=None, timeout_seconds=None):
        return list(chunk), 1, "oversized " * 200, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", non_reducing)
    try:
        original = _messages(8)
        result = engine.compress(original, current_tokens=1_900)

        assert result == original
        assert engine._last_full_sweep_status["reason"] == "no-progress"
        assert engine._last_full_sweep_status["publication_count"] == 0
        assert engine._dag.get_session_node_count("sweep-session") == 0
    finally:
        engine.shutdown()


def test_full_sweep_uses_minimum_fanin_only_under_prefix_pressure(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path,
        condensation_fanin=4,
        condensation_min_fanin=2,
        full_sweep_max_passes=16,
        summary_prefix_target_tokens=1,
    )
    _stub_summaries(monkeypatch, engine)
    try:
        engine.compress(_messages(5), current_tokens=1_900)

        parents = [
            node
            for node in engine._dag.get_session_nodes("sweep-session")
            if node.depth == 1
        ]
        assert parents
        assert any(len(parent.source_ids) in {2, 3} for parent in parents)
        assert engine._last_full_sweep_status["used_minimum_fanin"] is True
    finally:
        engine.shutdown()


def test_full_sweep_next_generation_carries_prior_canonical_nodes(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    try:
        first = engine.compress(_messages(10), current_tokens=1_900)
        first_frontier = engine._frontier.get_active_frontier("sweep-conversation")
        first_node_ids = {
            item["ref_id"]
            for item in engine._frontier.get_frontier_items(
                "sweep-conversation", first_frontier["generation"]
            )
            if item["kind"] == "node"
        }
        appended = first + _messages(8)[1:]

        engine.compress(appended, current_tokens=1_900)

        second_frontier = engine._frontier.get_active_frontier("sweep-conversation")
        second_node_ids = {
            item["ref_id"]
            for item in engine._frontier.get_frontier_items(
                "sweep-conversation", second_frontier["generation"]
            )
            if item["kind"] == "node"
        }
        assert second_frontier["generation"] == first_frontier["generation"] + 1
        assert first_node_ids <= second_node_ids
        assert all(engine._dag.get_node(node_id) is not None for node_id in second_node_ids)
    finally:
        engine.shutdown()


def test_full_sweep_ambiguous_post_publication_failure_returns_canonical_replacement(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    original = _messages(10)

    engine._full_sweep_publish_failure_hook = "after_commit"
    try:
        result = engine.compress(original, current_tokens=1_900)

        frontier = engine._frontier.get_active_frontier("sweep-conversation")
        items = engine._frontier.get_frontier_items(
            "sweep-conversation", frontier["generation"]
        )
        covered = {
            source_id
            for item in items
            if item["kind"] == "node"
            for source_id in _raw_descendants(engine, item["ref_id"])
        }
        returned = "\n".join(str(message.get("content") or "") for message in result)
        assert covered
        assert "turn 0" not in returned
        assert engine._ingest_cursor == len(result)
        assert engine._last_full_sweep_status["reason"] == "post-publication-failure"
        assert engine._last_full_sweep_status["publication_count"] == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("max_passes", [1, 2, 4, 8])
def test_full_sweep_pass_bound_is_hard(tmp_path, monkeypatch, max_passes):
    engine = _engine(tmp_path, full_sweep_max_passes=max_passes)
    _stub_summaries(monkeypatch, engine)
    try:
        engine.compress(_messages(20), current_tokens=1_900)
        assert engine._last_full_sweep_status["passes"] <= max_passes
    finally:
        engine.shutdown()


def test_full_sweep_revalidates_exact_sources_and_settles_base_batches(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    messages = _messages(10)
    engine.ingest(messages)
    engine._frontier.ensure_frontier(
        "sweep-conversation", "sweep-session", source_end_store_id=0
    )
    stored = engine._store.get_session_messages("sweep-session")
    source_ids = [int(row["store_id"]) for row in stored[:-2] if row["role"] != "system"]
    batch_id, _ = engine._frontier.create_batch_cas(
        conversation_id="sweep-conversation",
        session_id="sweep-session",
        base_generation=1,
        source_end_store_id=max(source_ids),
        source_identity_hash="stale",
        source_ids=source_ids,
        policy_fingerprint="",
        route_fingerprint="",
    )
    assert batch_id > 0
    try:
        engine.compress(messages, current_tokens=1_900)
        active = engine._frontier.get_active_frontier("sweep-conversation")
        lifecycle = engine._lifecycle.get_by_conversation("sweep-conversation")
        batch = engine._frontier.get_batch(batch_id)
        assert active is not None and lifecycle is not None and batch is not None
        assert batch.state == "superseded"
        assert lifecycle.current_frontier_store_id == active["source_end_store_id"]
    finally:
        engine.shutdown()


def test_full_sweep_rejects_source_rewrite_between_summary_and_publication(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    original_identity = engine._foreground_source_identity_for_messages
    rewritten = {"done": False}

    def identity_then_rewrite(messages, source_ids):
        identity = original_identity(messages, source_ids)
        if not rewritten["done"]:
            rewritten["done"] = True
            engine._store._conn.execute(
                "UPDATE messages SET content='concurrent rewrite' WHERE store_id=?",
                (int(source_ids[0]),),
            )
            engine._store._conn.commit()
        return identity

    monkeypatch.setattr(
        engine, "_foreground_source_identity_for_messages", identity_then_rewrite
    )
    try:
        engine.compress(_messages(10), current_tokens=1_900)
        assert engine._dag.get_session_node_count("sweep-session") == 0
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
        assert engine._last_full_sweep_status["publication_count"] == 0
    finally:
        engine.shutdown()


def test_full_sweep_rollback_never_deletes_waiting_writer_reused_node_id(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    writer_started = threading.Event()
    writer_committed = threading.Event()
    writer_result: dict[str, int] = {}
    writer_thread: list[threading.Thread] = []
    original_delete = engine._dag.delete_node

    def waiting_writer():
        conn = sqlite3.connect(str(engine._store.db_path), timeout=5.0)
        try:
            writer_started.set()
            cur = conn.execute(
                """INSERT INTO summary_nodes
                   (session_id, depth, summary, token_count, source_token_count,
                    source_ids, source_type, created_at)
                   VALUES ('winner-session', 0, 'waiting writer winner', 1, 1,
                           '[]', 'messages', 99)"""
            )
            conn.commit()
            writer_result["node_id"] = int(cur.lastrowid)
        finally:
            conn.close()
            writer_committed.set()

    def fail_after_nodes(phase):
        if phase == "after_nodes" and not writer_thread:
            thread = threading.Thread(target=waiting_writer, name="waiting-writer")
            writer_thread.append(thread)
            thread.start()
            assert writer_started.wait(timeout=2)
            raise RuntimeError("force full-sweep rollback")

    def expose_reused_id(node_id):
        assert writer_committed.wait(timeout=5)
        return original_delete(node_id)

    monkeypatch.setattr(engine._dag, "delete_node", expose_reused_id)
    engine._full_sweep_publish_failure_hook = fail_after_nodes
    try:
        engine.compress(_messages(10), current_tokens=1_900)
        assert writer_thread
        writer_thread[0].join(timeout=5)
        assert not writer_thread[0].is_alive()
        winner_id = writer_result["node_id"]
        row = engine._dag.connection.execute(
            "SELECT summary FROM summary_nodes WHERE node_id=?", (winner_id,)
        ).fetchone()
        assert row == ("waiting writer winner",)
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
    finally:
        engine._full_sweep_publish_failure_hook = None
        if writer_thread:
            writer_thread[0].join(timeout=5)
        engine.shutdown()


def test_full_sweep_aborts_losslessly_when_locked_tail_bounds_are_exceeded(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    monkeypatch.setattr(sweep_module, "_FULL_SWEEP_MAX_MESSAGE_ROWS", 1, raising=False)
    try:
        original = _messages(10)
        result = engine.compress(original, current_tokens=1_900)
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
        assert engine._dag.get_session_nodes("sweep-session") == []
        assert engine._last_full_sweep_status["reason"] == "publication-rolled-back"
        assert [message["content"] for message in result] == [
            message["content"] for message in original
        ]
    finally:
        engine.shutdown()


def test_full_sweep_aborts_losslessly_when_locked_frontier_bounds_are_exceeded(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    legacy_id = engine._store.append(
        "sweep-session", {"role": "user", "content": "prior frontier source"}
    )
    engine._frontier.set_frontier_items(
        "sweep-conversation",
        1,
        [{
            "kind": "message",
            "ref_id": legacy_id,
            "source_start": legacy_id,
            "source_end": legacy_id,
        }],
    )
    monkeypatch.setattr(sweep_module, "_FULL_SWEEP_MAX_FRONTIER_ROWS", 0)
    try:
        original = _messages(10)
        result = engine.compress(original, current_tokens=1_900)
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
        assert engine._dag.get_session_nodes("sweep-session") == []
        assert engine._last_full_sweep_status["reason"] == "publication-rolled-back"
        assert [message["content"] for message in result] == [
            message["content"] for message in original
        ]
    finally:
        engine.shutdown()


def test_full_sweep_aborts_losslessly_on_shared_locked_byte_budget(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    phases: list[str] = []
    engine._full_sweep_publish_crash_hook = phases.append
    monkeypatch.setattr(
        sweep_module, "_FULL_SWEEP_MAX_LOCKED_SERIALIZED_BYTES", 500
    )
    try:
        original = _messages(10)
        result = engine.compress(original, current_tokens=1_900)
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
        assert engine._dag.get_session_nodes("sweep-session") == []
        assert engine._last_full_sweep_status["reason"] == "publication-rolled-back"
        assert "after_nodes" not in phases
        assert [message["content"] for message in result] == [
            message["content"] for message in original
        ]
    finally:
        engine.shutdown()


def test_full_sweep_aborts_losslessly_on_locked_deadline(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _stub_summaries(monkeypatch, engine)
    expired = {"value": False}
    real_monotonic = time.monotonic

    def controlled_monotonic():
        value = real_monotonic()
        return value + 1_000.0 if expired["value"] else value

    monkeypatch.setattr(sweep_module.time, "monotonic", controlled_monotonic)
    original_canonical = engine._canonical_message_source_ids_no_commit

    def expire_after_canonical(*args, **kwargs):
        result = original_canonical(*args, **kwargs)
        expired["value"] = True
        return result

    monkeypatch.setattr(
        engine, "_canonical_message_source_ids_no_commit", expire_after_canonical
    )
    phases: list[str] = []
    engine._full_sweep_publish_crash_hook = phases.append
    try:
        original = _messages(10)
        result = engine.compress(original, current_tokens=1_900)
        assert engine._frontier.get_active_frontier("sweep-conversation")["generation"] == 1
        assert engine._dag.get_session_nodes("sweep-session") == []
        assert engine._last_full_sweep_status["reason"] == "publication-rolled-back"
        assert "after_nodes" not in phases
        assert [message["content"] for message in result] == [
            message["content"] for message in original
        ]
    finally:
        engine._full_sweep_publish_crash_hook = None
        engine.shutdown()


@pytest.mark.parametrize(
    ("phase", "expected_generation"),
    [("after_nodes", 1), ("after_commit", 2)],
)
def test_full_sweep_process_death_restarts_at_wholly_old_or_new_state(
    tmp_path, phase, expected_generation
):
    db_path = tmp_path / f"sweep-crash-{phase}.db"
    script = r'''
import os, sys
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.sweep as sweep_module

engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1], fresh_tail_count=2, leaf_chunk_tokens=32,
    condensation_fanin=4, condensation_min_fanin=2, incremental_max_depth=3,
    full_sweep_compaction_enabled=True, full_sweep_max_passes=64,
    full_sweep_deadline_seconds=30.0, summary_prefix_target_tokens=8,
    context_threshold=0.10,
))
engine.on_session_start('sweep-session', conversation_id='sweep-conversation', platform='test', context_length=2000)
def leaf(chunk, focus_topic=None, timeout_seconds=None):
    return list(chunk), 100, 'crash-safe-sweep-summary', 1, 1
engine._summarize_leaf_chunk_with_rescue = leaf
sweep_module.summarize_with_escalation = lambda **kwargs: ('crash-safe-parent', 1)
engine._full_sweep_publish_crash_hook = sys.argv[2]
messages = [{'role':'system','content':'anchor'}] + [
    {'role':'user' if i % 2 == 0 else 'assistant', 'content':f'turn {i} ' + 'payload '*24}
    for i in range(10)
]
engine.compress(messages, current_tokens=1900)
raise SystemExit('crash hook did not fire')
'''
    package_root = tmp_path / f"package-{phase}"
    package_root.mkdir()
    (package_root / "hermes_lcm").symlink_to(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), "/home/ben/hermes-agent-gil-pr"]
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert crashed.returncode == 90, (crashed.stdout, crashed.stderr)
    conn = sqlite3.connect(db_path)
    active = conn.execute(
        """SELECT generation, source_end_store_id FROM lcm_active_frontiers
           WHERE conversation_id='sweep-conversation' ORDER BY generation DESC LIMIT 1"""
    ).fetchone()
    node_count = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
    lifecycle = conn.execute(
        """SELECT current_frontier_store_id FROM lcm_lifecycle_state
           WHERE conversation_id='sweep-conversation'"""
    ).fetchone()
    conn.close()
    assert active[0] == expected_generation
    if phase == "after_nodes":
        assert node_count == 0
        assert lifecycle[0] == 0
    else:
        assert node_count > 0
        assert lifecycle[0] == active[1]
