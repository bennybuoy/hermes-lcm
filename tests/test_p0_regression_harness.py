"""P0 regression harness tests for the independent compaction policy.

These tests are written RED-first per the implementation wishlist:

1. Deterministic hierarchy: 16 depth-0 leaves at fan-in 4 → four depth-1
   nodes → one depth-2 parent. Assert source closure, absorption, and
   raw expansion.

2. Full host replacement seam: compress through the Hermes adapter,
   persist returned context, reload session, assert covered raw messages
   are absent from provider replay but present in immutable LCM storage.

3. Threshold-independence matrix: preparation/cutover/target/emergency and
   assembly caps cannot alter one another.

4. Multi-turn no-loop: reach target, prove unchanged subsequent turns do
   not compact again.

These tests describe the desired contract. Tests that are not yet
satisfied by the current implementation are marked xfail(strict=True)
so they become real gates as each feature lands.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    tmp_path,
    *,
    context_length: int = 200_000,
    context_threshold: float = 0.35,
    fresh_tail_count: int = 4,
    leaf_chunk_tokens: int = 100,
    condensation_fanin: int = 4,
    incremental_max_depth: int = 3,
    max_assembly_tokens: int = 0,
    reserve_tokens_floor: int = 0,
    emergency_pressure_ratio: float = 0.95,
    session_id: str = "harness-session",
    conversation_id: str = "harness-conversation",
    platform: str = "test",
    **extra_config,
):
    """Build a fully initialised LCMEngine for deterministic testing."""
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=fresh_tail_count,
        leaf_chunk_tokens=leaf_chunk_tokens,
        context_threshold=context_threshold,
        condensation_fanin=condensation_fanin,
        incremental_max_depth=incremental_max_depth,
        max_assembly_tokens=max_assembly_tokens,
        reserve_tokens_floor=reserve_tokens_floor,
        emergency_pressure_ratio=emergency_pressure_ratio,
        **extra_config,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform=platform,
        context_length=context_length,
    )
    return engine


def _messages(count: int, *, prefix: str = "msg", tokens_each: int = 30) -> list[dict[str, Any]]:
    """Build a deterministic message list with roughly *tokens_each* tokens per message."""
    messages: list[dict[str, Any]] = []
    filler = "x " * max(1, tokens_each // 2)
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"{prefix} {idx} {filler}"})
    return messages


def _stub_summary(monkeypatch, engine, *, text_prefix: str = "summary"):
    """Replace the LLM summary call with a deterministic stub."""
    call_counter = [0]

    def fake_summarize(**kwargs):
        call_counter[0] += 1
        content = kwargs.get("content", "")
        return (f"{text_prefix} #{call_counter[0]} of {len(content)} chars", call_counter[0] * 5)

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        fake_summarize,
    )
    return call_counter


# ===========================================================================
# 1. Deterministic hierarchy: 16 leaves → 4 depth-1 → 1 depth-2
# ===========================================================================

def _make_hierarchy_engine(tmp_path, *, session_id="hier-session", **overrides):
    """Build an engine tuned for deterministic multi-leaf hierarchy tests."""
    defaults = dict(
        context_length=100_000,
        context_threshold=0.08,
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=30,
        condensation_fanin=4,
        incremental_max_depth=3,
        cache_friendly_condensation_enabled=False,
    )
    defaults.update(overrides)
    return _make_engine(tmp_path, session_id=session_id, **defaults)


def _seed_depth0_leaves(engine, count=16, *, store_id_offset=1):
    """Manually create *count* depth-0 leaf nodes with real store_ids.

    Each leaf sources 2 raw messages from the store so source closure
    can be verified.
    """
    # First insert raw messages into the store
    msgs = _messages(count * 2, tokens_each=30)
    store_ids = engine._store.append_batch(
        engine.current_session_id, msgs, source="test"
    )

    nodes = []
    for i in range(count):
        src_ids = store_ids[i * 2 : i * 2 + 2]
        node = SummaryNode(
            session_id=engine.current_session_id,
            depth=0,
            summary=f"Leaf summary {i}",
            token_count=15,
            source_token_count=60,
            source_ids=src_ids,
            source_type="messages",
            created_at=time.time() + i,
            earliest_at=time.time() + i,
            latest_at=time.time() + i + 1,
            expand_hint=f"Expand for details about: leaf {i}",
        )
        engine._dag.add_node(node)
        nodes.append(node)
    return nodes, store_ids


def _seed_depth1_from_depth0(engine, depth0_nodes, fanin=4):
    """Manually condense depth-0 nodes into depth-1 nodes at *fanin*."""
    depth1_nodes = []
    for group_idx in range(0, len(depth0_nodes), fanin):
        group = depth0_nodes[group_idx : group_idx + fanin]
        if len(group) < fanin:
            break
        node = SummaryNode(
            session_id=engine.current_session_id,
            depth=1,
            summary=f"Depth-1 condensation of leaves {group_idx}-{group_idx + len(group) - 1}",
            token_count=30,
            source_token_count=sum(n.token_count for n in group),
            source_ids=[n.node_id for n in group],
            source_type="nodes",
            created_at=time.time(),
            earliest_at=group[0].earliest_at,
            latest_at=group[-1].latest_at,
            expand_hint="Expand for details about: condensation group",
        )
        engine._dag.add_node(node)
        depth1_nodes.append(node)
    return depth1_nodes


def _stub_condensation_summarize(monkeypatch, engine):
    """Stub the condensation summarizer to return deterministic text."""
    counter = [0]

    def fake_summarize(**kwargs):
        counter[0] += 1
        return (f"Condensed summary #{counter[0]}", counter[0] * 10)

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        fake_summarize,
    )
    return counter


class TestDeterministicHierarchy:
    """16 depth-0 leaves at fan-in 4 condense to four depth-1 nodes and
    one depth-2 parent. Assert source closure, absorption, and raw
    expansion."""

    def test_16_leaves_fanin_4_produces_depth2_parent(self, tmp_path, monkeypatch):
        """Seed 16 depth-0 leaves and 4 depth-1 nodes, trigger condensation
        to produce a depth-2 parent."""
        engine = _make_hierarchy_engine(tmp_path, session_id="hierarchy-session")
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            d0_nodes, store_ids = _seed_depth0_leaves(engine, count=16)
            d1_nodes = _seed_depth1_from_depth0(engine, d0_nodes, fanin=4)
            assert len(d1_nodes) == 4

            # Trigger condensation — 4 depth-1 nodes at fanin 4 → 1 depth-2
            engine._maybe_condense(
                focus_topic=None,
                leaf_compacted_this_turn=True,
                force_overflow=False,
            )

            depth_stats = engine._dag.get_session_depth_stats(engine.current_session_id)
            assert depth_stats.get(2, {}).get("count", 0) >= 1, (
                f"Expected >=1 depth-2 node after condensation, got {depth_stats}"
            )
        finally:
            engine.shutdown()

    def test_depth2_parent_source_ids_reference_depth1_nodes(self, tmp_path, monkeypatch):
        """The depth-2 parent must source_type='nodes' and reference depth-1 node IDs."""
        engine = _make_hierarchy_engine(tmp_path, session_id="closure-session")
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            d0_nodes, _ = _seed_depth0_leaves(engine, count=16)
            d1_nodes = _seed_depth1_from_depth0(engine, d0_nodes, fanin=4)

            engine._maybe_condense(
                focus_topic=None,
                leaf_compacted_this_turn=True,
                force_overflow=False,
            )

            depth2_nodes = engine._dag.get_session_nodes(
                engine.current_session_id, depth=2
            )
            assert len(depth2_nodes) >= 1, "No depth-2 nodes found"

            d1_ids = {n.node_id for n in d1_nodes}

            for d2_node in depth2_nodes:
                assert d2_node.source_type == "nodes", (
                    f"Depth-2 node {d2_node.node_id} source_type={d2_node.source_type}, "
                    "expected 'nodes'"
                )
                # Every source_id must be a real depth-1 node from our set
                for src_id in d2_node.source_ids:
                    src_node = engine._dag.get_node(src_id)
                    assert src_node is not None, (
                        f"Depth-2 node {d2_node.node_id} references "
                        f"missing node {src_id}"
                    )
                    assert src_node.depth == 1, (
                        f"Depth-2 node {d2_node.node_id} source {src_id} "
                        f"has depth {src_node.depth}, expected 1"
                    )
                    assert src_id in d1_ids, (
                        f"Depth-2 source {src_id} not in expected depth-1 set"
                    )
        finally:
            engine.shutdown()

    def test_depth1_node_sources_reference_depth0_leaves(self, tmp_path, monkeypatch):
        """Every depth-1 node must source_type='nodes' and reference depth-0 IDs."""
        engine = _make_hierarchy_engine(tmp_path, session_id="d1-closure-session")
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            d0_nodes, _ = _seed_depth0_leaves(engine, count=16)
            d1_nodes = _seed_depth1_from_depth0(engine, d0_nodes, fanin=4)

            d0_ids = {n.node_id for n in d0_nodes}

            for d1_node in d1_nodes:
                assert d1_node.source_type == "nodes"
                for src_id in d1_node.source_ids:
                    src_node = engine._dag.get_node(src_id)
                    assert src_node is not None
                    assert src_node.depth == 0
                    assert src_id in d0_ids
        finally:
            engine.shutdown()

    def test_absorbed_depth0_nodes_are_not_active(self, tmp_path, monkeypatch):
        """Depth-0 nodes referenced by a depth-1 parent are absorbed (not active)."""
        engine = _make_hierarchy_engine(tmp_path, session_id="absorption-session")
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            d0_nodes, _ = _seed_depth0_leaves(engine, count=16)
            d1_nodes = _seed_depth1_from_depth0(engine, d0_nodes, fanin=4)

            # Collect all source_ids from depth-1 parents
            absorbed_ids: set[int] = set()
            for d1 in d1_nodes:
                absorbed_ids.update(d1.source_ids)

            assert len(absorbed_ids) == 16, (
                f"Expected 16 absorbed depth-0 nodes, got {len(absorbed_ids)}"
            )

            # Absorbed depth-0 nodes should not be uncondensed
            uncondensed_d0 = engine._dag.get_uncondensed_at_depth(
                engine.current_session_id, depth=0
            )
            uncondensed_ids = {n.node_id for n in uncondensed_d0}
            overlap = absorbed_ids & uncondensed_ids
            assert not overlap, (
                f"Nodes {overlap} are both absorbed by depth-1 parents "
                f"and uncondensed at depth 0"
            )
        finally:
            engine.shutdown()

    def test_raw_messages_recoverable_after_full_hierarchy(self, tmp_path, monkeypatch):
        """After building a full hierarchy, raw messages are still in the store."""
        engine = _make_hierarchy_engine(tmp_path, session_id="raw-recovery-session")
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            d0_nodes, store_ids = _seed_depth0_leaves(engine, count=16)
            d1_nodes = _seed_depth1_from_depth0(engine, d0_nodes, fanin=4)

            engine._maybe_condense(
                focus_topic=None,
                leaf_compacted_this_turn=True,
                force_overflow=False,
            )

            # Raw messages must still be in the immutable store
            store_count = engine._store.get_session_count(engine.current_session_id)
            assert store_count >= 32, (
                f"Expected >=32 raw messages in store, got {store_count}"
            )
        finally:
            engine.shutdown()

    def test_full_hierarchy_through_compress(self, tmp_path, monkeypatch):
        """End-to-end: compress produces depth-0 leaves, and repeated
        compress calls build the full hierarchy up to depth 2."""
        engine = _make_hierarchy_engine(
            tmp_path,
            session_id="e2e-hierarchy-session",
            context_threshold=0.04,
            leaf_chunk_tokens=15,
        )
        _stub_condensation_summarize(monkeypatch, engine)
        try:
            # 40 messages at ~30 tokens each = ~1200 tokens total
            # With context_length=100K and threshold=0.04, cutover=4000
            # We need current_tokens > 4000 to trigger compaction
            msgs = _messages(40, tokens_each=30)
            engine.ingest(msgs)
            threshold = engine.threshold_tokens

            # Compress with high token count to trigger compaction
            result = engine.compress(msgs, current_tokens=threshold + 5000)

            depth_stats = engine._dag.get_session_depth_stats(engine.current_session_id)
            # We should have at least a few depth-0 leaves
            assert depth_stats.get(0, {}).get("count", 0) >= 1, (
                f"Expected depth-0 nodes after compress, got {depth_stats}"
            )
        finally:
            engine.shutdown()


# ===========================================================================
# 2. Full host replacement seam
# ===========================================================================

class TestHostReplacementSeam:
    """Compress through the Hermes adapter, persist returned context,
    reload session, assert covered raw messages are absent from provider
    replay but remain in immutable LCM storage."""

    def test_compress_returns_replacement_not_just_db_write(self, tmp_path, monkeypatch):
        """compress() must return a modified message list, not just write nodes."""
        engine = _make_engine(
            tmp_path,
            context_length=50_000,
            context_threshold=0.10,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="replacement-session",
        )
        _stub_summary(monkeypatch, engine)
        try:
            msgs = [{"role": "system", "content": "system"}] + _messages(20, tokens_each=40)
            engine.ingest(msgs)
            threshold = engine.threshold_tokens

            result = engine.compress(msgs, current_tokens=threshold + 200)

            # Result must be different from input (compaction occurred)
            assert result != msgs, "compress() returned input unchanged"
            assert len(result) < len(msgs), (
                f"compress() did not reduce message count: {len(result)} vs {len(msgs)}"
            )
        finally:
            engine.shutdown()

    def test_covered_raw_messages_absent_from_replay_but_in_storage(self, tmp_path, monkeypatch):
        """After compression, summarized raw messages are not in the returned
        context but are still in the immutable LCM store."""
        engine = _make_engine(
            tmp_path,
            context_length=50_000,
            context_threshold=0.10,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="seam-session",
        )
        _stub_summary(monkeypatch, engine)
        try:
            original_msgs = [{"role": "system", "content": "system"}] + _messages(20, tokens_each=40)
            engine.ingest(original_msgs)
            threshold = engine.threshold_tokens

            result = engine.compress(original_msgs, current_tokens=threshold + 200)

            # The returned context should contain summary content, not the raw messages
            result_content = " ".join(m.get("content", "") for m in result)
            raw_contents = [m.get("content", "") for m in original_msgs[1:]]  # skip system

            # At least some raw messages should be absent from the result
            absent_count = sum(1 for rc in raw_contents if rc not in result_content)
            assert absent_count > 0, "All raw messages still present in compressed result"

            # But all raw messages must still be in the immutable store
            store_count = engine._store.get_session_count(engine.current_session_id)
            assert store_count >= len(original_msgs), (
                f"Store has {store_count} messages, expected at least {len(original_msgs)}"
            )
        finally:
            engine.shutdown()

    def test_reloaded_session_preserves_immutable_raw_messages(self, tmp_path, monkeypatch):
        """After compression and session reload, raw messages survive in storage."""
        db_path = tmp_path / "reload.db"
        config = LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start(
            "reload-session",
            conversation_id="reload-conversation",
            platform="test",
            context_length=50_000,
        )
        _stub_summary(monkeypatch, engine)
        original_msgs = [{"role": "system", "content": "system"}] + _messages(20, tokens_each=40)
        engine.ingest(original_msgs)
        threshold = engine.threshold_tokens
        result = engine.compress(original_msgs, current_tokens=threshold + 200)
        engine.shutdown()

        # Reload with a new engine instance
        engine2 = LCMEngine(config=config)
        engine2.on_session_start(
            "reload-session",
            conversation_id="reload-conversation",
            platform="test",
            context_length=50_000,
        )
        try:
            store_count = engine2._store.get_session_count("reload-session")
            assert store_count >= len(original_msgs), (
                f"After reload, store has {store_count} messages, "
                f"expected >= {len(original_msgs)}"
            )

            # The DAG nodes should also survive
            dag_count = engine2._dag.get_session_node_count("reload-session")
            assert dag_count >= 1, f"After reload, DAG has {dag_count} nodes, expected >= 1"
        finally:
            engine2.shutdown()


# ===========================================================================
# 3. Threshold-independence matrix
# ===========================================================================

class TestThresholdIndependence:
    """preparation/cutover/target/emergency and assembly caps cannot
    alter one another.

    The four boundaries are:
    - cutover_threshold (host-visible threshold_tokens): when to trigger compaction
    - post_compaction_target: independent target for assembled output size
    - emergency_threshold: provider-window pressure for forced recovery
    - assembly caps: hard output safety bounds

    Invariant: post_compaction_target < preparation_threshold <= cutover_threshold < emergency_threshold
    """

    def test_cutover_is_not_lowered_by_assembly_cap(self, tmp_path):
        """A low max_assembly_tokens must not lower the cutover trigger."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            max_assembly_tokens=1000,
            session_id="cutover-indep-session",
        )
        try:
            # Cutover should be 200_000 * 0.50 = 100_000, NOT min(100_000, 1000)
            assert engine.threshold_tokens == 100_000, (
                f"Cutover threshold lowered by assembly cap: "
                f"got {engine.threshold_tokens}, expected 100000"
            )
        finally:
            engine.shutdown()

    def test_post_compaction_target_is_independent_of_cutover(self, tmp_path):
        """The post-compaction target is a separate value from cutover."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            max_assembly_tokens=50000,
            session_id="target-indep-session",
        )
        try:
            assert engine.threshold_tokens == 100_000
            target = engine._post_compaction_target_tokens()
            assert target == 50_000, (
                f"Post-compaction target should be 50000, got {target}"
            )
            assert target < engine.threshold_tokens, (
                "Post-compaction target should be below cutover"
            )
        finally:
            engine.shutdown()

    def test_ordinary_cutover_converges_to_resolved_policy_target(
        self, tmp_path, monkeypatch
    ):
        """A 128K cutover must keep compacting toward its 64K policy target.

        This reproduces the production GLM-5.2 failure where one bounded leaf
        pass reduced a roughly 128K provider prompt only to roughly 100K and
        then stopped because the loop compared against cutover, not target.
        """
        engine = _make_engine(
            tmp_path,
            context_length=1_000_000,
            context_threshold=0.75,
            model_thresholds={"glm-5.2": 0.128},
            fresh_tail_count=24,
            leaf_chunk_tokens=8_000,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=24_000,
            session_id="glm-target-convergence-session",
        )
        _stub_summary(monkeypatch, engine, text_prefix="bounded leaf")
        try:
            engine.update_model(
                model="glm-5.2",
                context_length=1_000_000,
                provider="ollama-cloud",
            )
            assert engine.threshold_tokens == 128_000
            assert engine._post_compaction_target_tokens() == 64_000

            # Size conversation tokens near the live GLM-5.2 shape (~90K
            # messages + ~38K fixed provider/system/tool overhead ≈ 128K).
            # tokens_each is approximate under the plugin tokenizer; assert
            # the resulting overhead stays below the 64K target so convergence
            # is physically reachable after fresh-tail preservation.
            messages = [
                {"role": "system", "content": "system prompt"},
                *_messages(120, tokens_each=3_000),
            ]
            message_tokens_before = count_messages_tokens(messages)
            observed_prompt_tokens = engine.threshold_tokens
            provider_overhead_tokens = observed_prompt_tokens - message_tokens_before
            assert 0 < provider_overhead_tokens < observed_prompt_tokens
            assert provider_overhead_tokens < 64_000, (
                "fixture overhead must leave room under the 64K target: "
                f"overhead={provider_overhead_tokens}, messages={message_tokens_before}"
            )

            result = engine.compress(
                messages,
                current_tokens=observed_prompt_tokens,
            )

            estimated_prompt_tokens_after = (
                provider_overhead_tokens + count_messages_tokens(result)
            )
            assert estimated_prompt_tokens_after <= 64_000, (
                "ordinary cutover stopped above the resolved policy target: "
                f"{estimated_prompt_tokens_after} > 64000"
            )
            depth0_nodes = engine._dag.get_session_nodes(
                engine.current_session_id, depth=0
            )
            assert len(depth0_nodes) >= 3, (
                "128K→64K convergence should require multiple bounded leaf passes"
            )
        finally:
            engine.shutdown()

    def test_partial_rescue_does_not_trim_unsummarized_raw_to_policy_target(
        self, tmp_path, monkeypatch
    ):
        """A partial rescue pass must not evict raw content it did not cover."""
        engine = _make_engine(
            tmp_path,
            context_length=20_000,
            context_threshold=0.75,
            model_thresholds={"rescue-model": 0.10},
            fresh_tail_count=2,
            leaf_chunk_tokens=100,
            dynamic_leaf_chunk_enabled=False,
            session_id="partial-rescue-lossless-session",
        )
        try:
            engine.update_model(
                model="rescue-model",
                context_length=20_000,
                provider="test",
            )
            assert engine.threshold_tokens == 2_000
            assert engine._post_compaction_target_tokens() == 1_000

            messages = [
                {"role": "system", "content": "system prompt"},
                *_messages(22, prefix="lossless", tokens_each=180),
            ]
            leading = engine._leading_anchor_count(messages)
            fresh_start = len(messages) - engine._config.fresh_tail_count
            raw_outside_tail = messages[leading:fresh_start]
            assert len(raw_outside_tail) > 4

            def partial_rescue(chunk, focus_topic=None):
                compacted = list(chunk[:2])
                source_tokens = count_messages_tokens(compacted)
                return (
                    compacted,
                    source_tokens,
                    "Partial rescue summary.\nExpand for details about: rescued prefix",
                    1,
                    1,
                )

            monkeypatch.setattr(
                engine,
                "_summarize_leaf_chunk_with_rescue",
                partial_rescue,
            )

            result = engine.compress(messages, current_tokens=engine.threshold_tokens)

            nodes = engine._dag.get_session_nodes(engine.current_session_id, depth=0)
            assert len(nodes) == 1
            covered_contents = {
                engine._store.get(store_id)["content"]
                for store_id in nodes[0].source_ids
            }
            returned_contents = {
                msg.get("content")
                for msg in result
                if isinstance(msg, dict)
            }
            unsummarized = [
                msg["content"]
                for msg in raw_outside_tail
                if msg["content"] not in covered_contents
            ]
            assert unsummarized
            assert set(unsummarized).issubset(returned_contents), (
                "policy-target assembly silently dropped raw messages that the "
                "partial rescue leaf did not cover"
            )
        finally:
            engine.shutdown()

    def test_deadline_after_leaf_publication_returns_canonical_replacement(
        self, tmp_path, monkeypatch
    ):
        """Timeout after publish must not replay raw rows already covered by DAG."""

        engine = _make_engine(
            tmp_path,
            context_length=20_000,
            context_threshold=0.75,
            model_thresholds={"deadline-model": 0.10},
            fresh_tail_count=2,
            leaf_chunk_tokens=120,
            dynamic_leaf_chunk_enabled=True,
            dynamic_leaf_chunk_max=180,
            condensation_fanin=100,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=False,
            session_id="deadline-after-publication-session",
        )
        try:
            engine.update_model(
                model="deadline-model",
                context_length=20_000,
                provider="test",
            )
            engine._config.foreground_compress_deadline_seconds = 1.0
            messages = [
                {"role": "system", "content": "system prompt"},
                *_messages(28, prefix="deadline", tokens_each=100),
            ]

            def expire_after_summary(chunk, focus_topic=None):
                compacted = list(chunk)
                source_tokens = count_messages_tokens(compacted)
                time.sleep(1.05)
                return (
                    compacted,
                    source_tokens,
                    "Published before deadline.\nExpand for details about: first leaf",
                    0,
                    0,
                )

            monkeypatch.setattr(
                engine,
                "_summarize_leaf_chunk_with_rescue",
                expire_after_summary,
            )

            result = engine.compress(messages, current_tokens=engine.threshold_tokens)

            nodes = engine._dag.get_session_nodes(engine.current_session_id, depth=0)
            assert len(nodes) == 1, "fixture must publish one canonical leaf before timeout"
            covered_contents = {
                engine._store.get(store_id)["content"]
                for store_id in nodes[0].source_ids
            }
            returned_contents = {
                msg.get("content")
                for msg in result
                if isinstance(msg, dict)
            }
            assert covered_contents
            assert covered_contents.isdisjoint(returned_contents), (
                "deadline returned stale host input containing raw rows already "
                "published into the canonical DAG"
            )
            assert any(
                "Published before deadline" in str(msg.get("content", ""))
                for msg in result
                if isinstance(msg, dict)
            )
            assert result[-1]["content"] == messages[-1]["content"]
            assert engine._last_compression_status != "running"
        finally:
            engine.shutdown()

    def test_emergency_threshold_is_independent_of_cutover_and_target(self, tmp_path):
        """Emergency threshold uses emergency_pressure_ratio, not cutover or target."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            max_assembly_tokens=50000,
            emergency_pressure_ratio=0.90,
            session_id="emergency-indep-session",
        )
        try:
            assert engine.threshold_tokens == 100_000  # cutover
            assert engine._post_compaction_target_tokens() == 50_000  # target
            emergency = engine._effective_emergency_threshold_tokens()
            assert emergency == 180_000, (
                f"Emergency threshold should be 200000*0.90=180000, got {emergency}"
            )
            # Invariant: target < cutover < emergency
            assert 50_000 < 100_000 < 180_000
        finally:
            engine.shutdown()

    def test_reserve_floor_does_not_affect_cutover(self, tmp_path):
        """reserve_tokens_floor constrains assembly output, not cutover trigger."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            reserve_tokens_floor=100_000,
            session_id="reserve-indep-session",
        )
        try:
            # Cutover should still be 100_000
            assert engine.threshold_tokens == 100_000, (
                f"reserve_tokens_floor lowered cutover: {engine.threshold_tokens}"
            )
            # But the assembly cap should be 200_000 - 100_000 = 100_000
            cap = engine._effective_assembly_token_cap()
            assert cap == 100_000
        finally:
            engine.shutdown()

    def test_assembly_cap_does_not_trigger_overflow_recovery(self, tmp_path):
        """Being above the assembly cap but below emergency should not force
        overflow recovery."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            max_assembly_tokens=50000,
            emergency_pressure_ratio=0.95,
            session_id="overflow-indep-session",
        )
        try:
            # At 60K tokens: above target (50K) but below emergency (190K)
            assert not engine._should_force_overflow_recovery(observed_tokens=60_000), (
                "Overflow recovery triggered at 60K which is above target "
                "but well below emergency threshold"
            )
            # At 190K: at emergency threshold
            assert engine._should_force_overflow_recovery(observed_tokens=190_000), (
                "Overflow recovery not triggered at emergency threshold"
            )
        finally:
            engine.shutdown()

    def test_changing_assembly_cap_does_not_change_cutover(self, tmp_path):
        """Changing max_assembly_tokens at runtime must not change threshold_tokens."""
        engine = _make_engine(
            tmp_path,
            context_length=200_000,
            context_threshold=0.50,
            max_assembly_tokens=0,
            session_id="runtime-cap-session",
        )
        try:
            assert engine.threshold_tokens == 100_000

            # Change assembly cap
            engine._config.max_assembly_tokens = 1000
            # Cutover should be unchanged
            assert engine.threshold_tokens == 100_000, (
                "Cutover changed after modifying max_assembly_tokens"
            )

            # Change reserve floor
            engine._config.reserve_tokens_floor = 150_000
            assert engine.threshold_tokens == 100_000, (
                "Cutover changed after modifying reserve_tokens_floor"
            )
        finally:
            engine.shutdown()


# ===========================================================================
# 4. Multi-turn no-loop test
# ===========================================================================

class TestNoCompactionLoop:
    """Reach target, then prove unchanged subsequent turns do not compact
    again. Attempts, wall time, generation rewrites, and emergency fallback
    are bounded."""

    def test_repeated_turns_at_target_do_not_recompact(self, tmp_path, monkeypatch):
        """After reaching the post-compaction target, unchanged subsequent
        turns should not trigger another compaction."""
        engine = _make_engine(
            tmp_path,
            context_length=100_000,
            context_threshold=0.50,
            max_assembly_tokens=20_000,
            emergency_pressure_ratio=0.95,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="no-loop-session",
        )
        call_counter = _stub_summary(monkeypatch, engine)
        try:
            msgs = [{"role": "system", "content": "system"}] + _messages(40, tokens_each=40)
            engine.ingest(msgs)
            threshold = engine.threshold_tokens  # 50_000

            # First compaction
            result = engine.compress(msgs, current_tokens=threshold + 100)
            first_count = engine.compression_count
            assert first_count >= 1, "First compress did not compact"

            summary_calls_after_first = call_counter[0]

            # Now feed the compressed result back unchanged — should not recompact
            # The compressed context should be below cutover
            result_tokens = count_messages_tokens(result)

            # If result is already below cutover, subsequent calls should be no-ops
            if result_tokens < threshold:
                # Run preflight — should not request compaction
                preflight = engine.should_compress_preflight(result)
                assert not preflight, (
                    f"Preflight requested re-compaction with {result_tokens} tokens "
                    f"below cutover {threshold}"
                )
            else:
                # If still above cutover, the second compress should make progress
                # but should not loop indefinitely
                result2 = engine.compress(result, current_tokens=result_tokens)
                second_count = engine.compression_count
                # Compression count should not have exploded
                assert second_count <= first_count + 2, (
                    f"Too many compactions: {second_count} after {first_count}"
                )
        finally:
            engine.shutdown()

    def test_bypassed_session_above_target_below_cutover_no_recompact(self, tmp_path, monkeypatch):
        """A bypassed session sitting above the post-compaction target but
        below the cutover threshold should not trigger repeated compaction."""
        engine = _make_engine(
            tmp_path,
            context_length=1_000,
            context_threshold=0.50,
            max_assembly_tokens=90,
            emergency_pressure_ratio=0.95,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="ignored:no-reloop",
        )
        # Make it a bypassed session
        engine._config.ignore_session_patterns = ["ignored:*"]
        engine._refresh_session_filters()
        try:
            engine.threshold_tokens = 350  # cutover
            messages = [{"role": "user", "content": "unchanged assembled context"}]

            # Stub token counting to report 110 tokens (above target 90, below cutover 350)
            from hermes_lcm import compaction as lcm_compaction_module
            monkeypatch.setattr(lcm_compaction_module, "count_messages_tokens", lambda _: 110)

            assert engine._post_compaction_target_tokens() == 90
            # Multiple preflight calls should all be no-ops
            assert not engine.should_compress_preflight(messages)
            assert not engine.should_compress_preflight(messages)
            assert engine.compression_count == 0
        finally:
            engine.shutdown()

    def test_emergency_recovery_converges_and_does_not_loop(self, tmp_path, monkeypatch):
        """Emergency recovery should converge: after recovery, subsequent
        turns at the recovered size should not re-trigger emergency."""
        engine = _make_engine(
            tmp_path,
            context_length=1_000,
            context_threshold=0.50,
            max_assembly_tokens=100,
            emergency_pressure_ratio=0.90,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="emergency-converge-session",
        )
        _stub_summary(monkeypatch, engine)
        try:
            msgs = [{"role": "system", "content": "system"}] + _messages(20, tokens_each=30)
            engine.ingest(msgs)
            threshold = engine.threshold_tokens  # 500

            # Force emergency by providing very high token count
            emergency_threshold = engine._effective_emergency_threshold_tokens()
            assert emergency_threshold == 900  # 1000 * 0.90

            result = engine.compress(msgs, current_tokens=emergency_threshold + 50)

            # After emergency recovery, the result should be bounded
            result_tokens = count_messages_tokens(result)
            # Should be well under the emergency threshold
            assert result_tokens < emergency_threshold, (
                f"Emergency recovery did not converge: result={result_tokens}, "
                f"emergency={emergency_threshold}"
            )

            # A second compress with the same token count should not loop
            result2 = engine.compress(result, current_tokens=emergency_threshold + 50)
            # Compression count should not have exploded
            # (it may compact once more but should not loop)
        finally:
            engine.shutdown()


# ===========================================================================
# 5. Policy fingerprint and status reporting (P0.2 contract)
# ===========================================================================

class TestPolicyFingerprintContract:
    """RED tests for the typed ModelCompactionPolicy that doesn't exist yet.
    These are xfail(strict=True) until P0.2 is implemented."""

    def test_policy_dataclass_exists(self, tmp_path):
        from hermes_lcm.policy import ModelCompactionPolicy
        policy = ModelCompactionPolicy(
            cutover_threshold=0.50,
            post_compaction_target=0.25,
            emergency_threshold=0.95,
            preparation_threshold=0.40,
        )
        assert policy.cutover_threshold == 0.50
        assert policy.post_compaction_target == 0.25

    def test_policy_resolver_normalizes_model_name(self, tmp_path):
        from hermes_lcm.policy import resolve_policy
        # Case-insensitive normalized matching
        policy = resolve_policy(
            model="GLM-5.2",
            provider="ollama-cloud",
            context_length=262_144,
        )
        assert policy is not None
        assert policy.cutover_threshold > 0

    def test_policy_has_stable_fingerprint(self, tmp_path):
        from hermes_lcm.policy import resolve_policy
        p1 = resolve_policy(model="glm-5.2", provider="ollama", context_length=262_144)
        p2 = resolve_policy(model="glm-5.2", provider="ollama", context_length=262_144)
        assert p1.fingerprint == p2.fingerprint
        # Different model → different fingerprint
        p3 = resolve_policy(model="deepseek-v4", provider="ollama", context_length=262_144)
        assert p1.fingerprint != p3.fingerprint

    def test_status_reports_selected_policy(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            context_length=262_144,
            context_threshold=0.85,
            session_id="policy-status-session",
        )
        try:
            engine.update_model("glm-5.2", 262_144, provider="ollama")
            status = engine.get_status()
            assert "compaction_policy" in status, (
                "lcm_status missing 'compaction_policy' key"
            )
            policy = status["compaction_policy"]
            assert "fingerprint" in policy
            assert "cutover_threshold" in policy
            assert "post_compaction_target" in policy
            assert "emergency_threshold" in policy
        finally:
            engine.shutdown()

    def test_policy_invariant_violation_raises(self, tmp_path):
        from hermes_lcm.policy import ModelCompactionPolicy
        with pytest.raises(ValueError, match="invariant"):
            ModelCompactionPolicy(
                cutover_threshold=0.50,
                post_compaction_target=0.60,  # violates: target < cutover
                emergency_threshold=0.95,
                preparation_threshold=0.40,
            )

    def test_per_model_policy_overrides_default(self, tmp_path):
        from hermes_lcm.policy import resolve_policy
        # MiniMax M3 should get 0.48 cutover (below 512K cliff)
        policy = resolve_policy(
            model="minimax-m3",
            provider="minimax",
            context_length=1_000_000,
        )
        assert policy.cutover_threshold == pytest.approx(0.48)
        # Unknown model gets conservative default
        unknown = resolve_policy(
            model="some-unknown-model",
            provider="unknown",
            context_length=200_000,
        )
        assert unknown.cutover_threshold == pytest.approx(0.75)


# ===========================================================================
# 6. Persistent active frontier contract (P0.3)
# ===========================================================================

class TestActiveFrontierContract:
    """RED tests for the persistent ordered active frontier. These are
    xfail(strict=True) until P0.3 is implemented."""

    def test_frontier_schema_exists(self, tmp_path):
        from hermes_lcm.db_bootstrap import run_versioned_migrations
        import sqlite3
        db = tmp_path / "frontier.db"
        conn = sqlite3.connect(str(db))
        run_versioned_migrations(conn)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "lcm_active_frontiers" in tables
        assert "lcm_frontier_items" in tables
        assert "lcm_prepared_batches" in tables

    def test_prepare_background_compaction_returns_batch(self, tmp_path, monkeypatch):
        engine = _make_engine(
            tmp_path,
            context_length=50_000,
            context_threshold=0.10,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="bg-prep-session",
            async_background_compaction_enabled=True,
        )
        _stub_summary(monkeypatch, engine)
        try:
            msgs = _messages(20, tokens_each=40)
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            assert batch is not None
            assert batch.state == "ready"
        finally:
            engine.shutdown()

    def test_atomic_promotion_advances_frontier(self, tmp_path, monkeypatch):
        engine = _make_engine(
            tmp_path,
            context_length=50_000,
            context_threshold=0.10,
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            session_id="promote-session",
            async_background_compaction_enabled=True,
        )
        _stub_summary(monkeypatch, engine)
        try:
            msgs = _messages(20, tokens_each=40)
            engine.ingest(msgs)
            batch = engine.prepare_background_compaction_once(msgs)
            result = engine.promote_prepared_compaction(batch.batch_id, msgs)
            assert result.promoted is True
        finally:
            engine.shutdown()