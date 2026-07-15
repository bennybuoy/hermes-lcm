"""Issue #8: bounded full-sweep candidate construction and publication."""

from __future__ import annotations

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
    original = engine._frontier.advance_frontier_generation_with_items

    def counted(*args, **kwargs):
        publication_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine._frontier, "advance_frontier_generation_with_items", counted)
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

    def persist_failure():
        raise RuntimeError("injected lifecycle failure")

    monkeypatch.setattr(engine, "_persist_frontier_marker", persist_failure)
    monkeypatch.setattr(engine._frontier, "rollback_frontier_generation", lambda *_: False)
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
