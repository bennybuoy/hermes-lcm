"""Adversarial regressions for the final publication/security review blockers."""

from __future__ import annotations

import json
import io
import os
import sqlite3
import time

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.config as config_module
import hermes_lcm.frontier as frontier_module
import hermes_lcm.store as store_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import redact_sensitive_text
from hermes_lcm.preparation import active_profile_admissions
from hermes_lcm.schemas import LCM_EXPAND, LCM_EXPAND_QUERY
import hermes_lcm.tools as tools_module


def _engine(tmp_path, **overrides) -> LCMEngine:
    values = {
        "database_path": str(tmp_path / "blockers.db"),
        "fresh_tail_count": 1,
        "async_background_compaction_enabled": True,
        "async_preparation_utility_policy_enabled": True,
        "async_preparation_min_reduction_tokens": 1,
        "context_threshold": 0.01,
    }
    values.update(overrides)
    engine = LCMEngine(config=LCMConfig(**values))
    engine.on_session_start(
        "current", conversation_id="conversation", platform="test", context_length=10_000
    )
    return engine


def test_unterminated_quoted_password_is_redacted_before_any_truncation(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    try:
        value = 'prefix password="supersecret-without-closing-quote'
        redacted = redact_sensitive_text(value, engine._config)
        assert "supersecret" not in redacted
        assert "LCM sensitive redaction" in redacted
    finally:
        engine.shutdown()


def test_escaped_assignment_password_is_redacted_in_linear_time(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    try:
        value = 'prefix password=\\"escaped-assignment-secret suffix'
        started = time.monotonic()
        redacted = redact_sensitive_text((value + "\n") * 2_000, engine._config)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert "escaped-assignment-secret" not in redacted
        assert "LCM sensitive redaction" in redacted
    finally:
        engine.shutdown()


def test_escaped_json_password_is_independently_redacted_in_linear_time(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    try:
        value = 'prefix \\"password\\":\\"escaped-json-secret\\" suffix'
        started = time.monotonic()
        redacted = redact_sensitive_text((value + "\n") * 2_000, engine._config)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert "escaped-json-secret" not in redacted
        assert "LCM sensitive redaction" in redacted
    finally:
        engine.shutdown()


def test_cross_session_high_fanout_is_rejected_before_json_list_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, cross_session_expansion_enabled=True)
    raw = "[" + ",".join(str(index) for index in range(20_000)) + "]"
    conn = engine._dag.connection
    assert conn is not None
    conn.execute(
        """INSERT INTO summary_nodes
           (session_id, depth, summary, token_count, source_token_count,
            source_ids, source_type, created_at)
           VALUES ('archive', 1, 'fanout archive', 1, 1, ?, 'nodes', 1)""",
        (raw,),
    )
    node_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.commit()
    original_loads = tools_module.json.loads

    def guarded_loads(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("unbounded source_ids JSON was materialized")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(tools_module.json, "loads", guarded_loads)
    capability = engine.issue_cross_session_capability(["archive"])
    try:
        result = json.loads(tools_module.lcm_expand_query(
            {
                "prompt": "find archive",
                "node_ids": [node_id],
                "cross_session": True,
            },
            engine=engine,
            cross_session_capability=capability,
        ))
        assert result.get("authorization_truncated") is True or "error" in result
    finally:
        engine.shutdown()


def test_prepared_batch_lineage_rejects_before_oversized_json_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    source_id = engine._store.append(
        "current", {"role": "user", "content": "prepared source"}
    )
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    batch_id, _ = engine._frontier.create_batch_cas(
        conversation_id="conversation",
        session_id="current",
        base_generation=int(active["generation"]),
        source_end_store_id=source_id,
        source_identity_hash="identity",
        source_ids=[source_id],
        policy_fingerprint="",
        route_fingerprint="",
    )
    assert batch_id > 0
    raw = "[" + ",".join(str(index) for index in range(20_000)) + "]"
    engine._frontier._conn.execute(
        "UPDATE lcm_prepared_batches SET source_ids=?, state='ready' WHERE batch_id=?",
        (raw, batch_id),
    )
    engine._frontier._conn.commit()
    original_loads = tools_module.json.loads

    def guarded_loads(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("oversized prepared-batch lineage was decoded")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(tools_module.json, "loads", guarded_loads)
    import hermes_lcm.frontier as frontier_module
    monkeypatch.setattr(frontier_module.json, "loads", guarded_loads)
    try:
        batch = engine._frontier.get_batch(batch_id)
        assert batch is not None
        assert batch.state == "rejected"
        assert batch.source_ids == []
        assert "source_ids" in batch.failure_reason
        assert engine._frontier.get_ready_batch("conversation") is None
    finally:
        engine.shutdown()


def test_prepared_batch_payload_is_length_checked_before_sql_substring_fetch(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    batch_id, _ = engine._frontier.create_batch_cas(
        conversation_id="conversation",
        session_id="current",
        base_generation=int(active["generation"]),
        source_end_store_id=1,
        source_identity_hash="identity",
        source_ids=[1],
        policy_fingerprint="policy",
        route_fingerprint="route",
    )
    engine._frontier._conn.execute(
        """UPDATE lcm_prepared_batches
           SET state='ready', summary_payload=? WHERE batch_id=?""",
        ("payload-secret-" + ("x" * 4_000), batch_id),
    )
    engine._frontier._conn.commit()
    monkeypatch.setattr(frontier_module, "_BATCH_LOAD_MAX_SUMMARY_BYTES", 1_024)
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    try:
        batch = engine._frontier.get_batch(batch_id)
        assert batch is not None and batch.state == "rejected"
        assert batch.summary_payload == ""
        selects = [
            statement
            for statement in statements
            if "FROM lcm_prepared_batches" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert len(selects) == 1
        assert "LENGTH(CAST(SUMMARY_PAYLOAD AS BLOB))" in selects[0].upper()
        assert "payload-secret" not in selects[0]
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_inspection_lineage_is_sql_row_edge_and_deadline_bounded(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    raw = "[" + ",".join(str(index) for index in range(20_000)) + "]"
    conn = engine._dag.connection
    assert conn is not None
    conn.execute(
        """INSERT INTO summary_nodes
           (session_id, depth, summary, token_count, source_token_count,
            source_ids, source_type, created_at)
           VALUES ('current', 0, 'poisoned lineage', 1, 1, ?, 'messages', 1)""",
        (raw,),
    )
    conn.commit()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    original_loads = tools_module.json.loads

    def guarded_loads(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("oversized inspection lineage was decoded")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(tools_module.json, "loads", guarded_loads)
    try:
        assert tools_module._inspect_highest_compacted_source_store_id(
            engine, "current"
        ) == 0
        assert any("LIMIT" in statement.upper() for statement in statements)
    finally:
        conn.set_trace_callback(None)
        engine.shutdown()


def test_inspection_lineage_aggregate_row_cap_stops_before_later_rows(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    conn = engine._dag.connection
    assert conn is not None
    for source_id in (11, 99):
        conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at)
               VALUES ('current', 0, ?, 1, 1, ?, 'messages', 1)""",
            (f"row-{source_id}", json.dumps([source_id])),
        )
    conn.commit()
    monkeypatch.setattr(tools_module, "_LCM_INSPECT_LINEAGE_MAX_ROWS", 1)
    try:
        assert tools_module._inspect_highest_compacted_source_store_id(
            engine, "current"
        ) == 11
    finally:
        engine.shutdown()


def test_inspection_lineage_aggregate_edge_cap_stops_before_later_edges(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    conn = engine._dag.connection
    assert conn is not None
    for raw in ("[11,12]", "[99]"):
        conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at)
               VALUES ('current', 0, 'edge row', 1, 1, ?, 'messages', 1)""",
            (raw,),
        )
    conn.commit()
    monkeypatch.setattr(tools_module, "_LCM_INSPECT_LINEAGE_MAX_EDGES", 2)
    try:
        assert tools_module._inspect_highest_compacted_source_store_id(
            engine, "current"
        ) == 12
    finally:
        engine.shutdown()


def test_inspection_lineage_deadline_stops_before_decode(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    conn = engine._dag.connection
    assert conn is not None
    conn.execute(
        """INSERT INTO summary_nodes
           (session_id, depth, summary, token_count, source_token_count,
            source_ids, source_type, created_at)
           VALUES ('current', 0, 'deadline row', 1, 1, '[99]', 'messages', 1)"""
    )
    conn.commit()
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(tools_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        tools_module,
        "decode_source_ids",
        lambda _raw: (_ for _ in ()).throw(AssertionError("decoded after deadline")),
    )
    try:
        assert tools_module._inspect_highest_compacted_source_store_id(
            engine, "current"
        ) == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("limit_kind", ["rows", "edges", "bytes", "deadline"])
def test_foreground_publication_canonical_limits_abort_without_partial_truth(
    tmp_path, monkeypatch, limit_kind
):
    engine = _engine(tmp_path)
    first = engine._store.append(
        "current", {"role": "user", "content": "canonical one"}
    )
    second = engine._store.append(
        "current", {"role": "assistant", "content": "canonical two"}
    )
    candidate_id = engine._store.append(
        "current", {"role": "user", "content": "candidate"}
    )
    for source_id in (first, second):
        engine._dag.add_node(SummaryNode(
            session_id="current",
            depth=0,
            summary=f"existing-{source_id}",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        ))
    engine._frontier.ensure_frontier("conversation", "current")
    if limit_kind == "rows":
        monkeypatch.setattr(engine_module, "_CANONICAL_LINEAGE_MAX_ROWS", 1)
    elif limit_kind == "edges":
        monkeypatch.setattr(engine_module, "_CANONICAL_LINEAGE_MAX_EDGES", 1)
    elif limit_kind == "bytes":
        monkeypatch.setattr(engine_module, "_CANONICAL_LINEAGE_MAX_BYTES", 4)
    else:
        monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_DEADLINE_SECONDS", 0.0)
    before_nodes = len(engine._dag.get_session_nodes("current"))
    try:
        with pytest.raises(RuntimeError, match="lineage|deadline"):
            engine._publish_foreground_leaf(
                node=SummaryNode(
                    session_id="current",
                    depth=0,
                    summary="must roll back",
                    token_count=1,
                    source_token_count=1,
                    source_ids=[candidate_id],
                    source_type="messages",
                    created_at=2.0,
                ),
                source_end_store_id=candidate_id,
                covered_source_ids=[candidate_id],
            )
        assert len(engine._dag.get_session_nodes("current")) == before_nodes
        assert engine._frontier.get_active_frontier("conversation")["generation"] == 1
    finally:
        engine.shutdown()


def test_canonical_lineage_page_is_byte_sized_before_python_materialization(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    conn = engine._dag.connection
    assert conn is not None
    for source_id in (1, 2, 3):
        raw = "[" + (" " * 60_000) + str(source_id) + "]"
        conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at)
               VALUES ('current', 0, ?, 1, 1, ?, 'messages', 1)""",
            (f"existing-{source_id}", raw),
        )
    conn.commit()
    candidate_id = engine._store.append(
        "current", {"role": "user", "content": "candidate"}
    )
    engine._frontier.ensure_frontier("conversation", "current")
    monkeypatch.setattr(engine_module, "_CANONICAL_LINEAGE_MAX_BYTES", 70_000)
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(RuntimeError, match="lineage.*byte"):
            engine._publish_foreground_leaf(
                node=SummaryNode(
                    session_id="current",
                    depth=0,
                    summary="must roll back",
                    token_count=1,
                    source_token_count=1,
                    source_ids=[candidate_id],
                    source_type="messages",
                    created_at=2.0,
                ),
                source_end_store_id=candidate_id,
                covered_source_ids=[candidate_id],
            )
        lineage_selects = [
            statement
            for statement in statements
            if "FROM summary_nodes" in statement
            and "source_type = 'messages'" in statement
        ]
        assert lineage_selects
        assert all("SUBSTR" in statement.upper() for statement in lineage_selects)
        assert "LIMIT 1" in lineage_selects[0].upper()
        assert engine._dag.get_session_node_count("current") == 3
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_pending_batch_enumeration_pages_and_omits_unbounded_payload_columns(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    for source_end in range(1, 6):
        batch_id, _ = engine._frontier.create_batch_cas(
            conversation_id="conversation",
            session_id="current",
            base_generation=int(active["generation"]),
            source_end_store_id=source_end,
            source_identity_hash=f"hash-{source_end}",
            source_ids=[source_end],
            policy_fingerprint="policy",
            route_fingerprint="route",
        )
        assert batch_id > 0
        engine._frontier._conn.execute(
            """UPDATE lcm_prepared_batches
               SET summary_payload=?, resolved_policy_json=? WHERE batch_id=?""",
            ("payload-secret-" + "x" * 20_000, "policy-secret-" + "y" * 20_000, batch_id),
        )
        engine._frontier._conn.commit()
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(frontier_module, "_PENDING_BATCH_QUERY_PAGE", 2)
    try:
        batches = engine._frontier.list_pending_batches("conversation")
        assert [batch.source_end_store_id for batch in batches] == [5, 4, 3, 2, 1]
        assert all(batch.summary_payload == "" for batch in batches)
        selects = [
            statement for statement in statements
            if "FROM lcm_prepared_batches" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert len(selects) >= 3
        assert all("LIMIT" in statement.upper() for statement in selects)
        assert all("COALESCE(summary_payload" not in statement for statement in selects)
        assert all("COALESCE(resolved_policy_json" not in statement for statement in selects)
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_pending_batch_page_is_byte_sized_and_substring_capped_before_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    padded_values = []
    for source_end in range(1, 4):
        batch_id, _ = engine._frontier.create_batch_cas(
            conversation_id="conversation",
            session_id="current",
            base_generation=int(active["generation"]),
            source_end_store_id=source_end,
            source_identity_hash=f"hash-{source_end}",
            source_ids=[source_end],
            policy_fingerprint="policy",
            route_fingerprint="route",
        )
        padded = "[" + (" " * 60_000) + str(source_end) + "]"
        padded_values.append(padded)
        engine._frontier._conn.execute(
            "UPDATE lcm_prepared_batches SET source_ids=? WHERE batch_id=?",
            (padded, batch_id),
        )
        engine._frontier._conn.commit()
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        frontier_module, "_PENDING_BATCH_MAX_SERIALIZED_BYTES", 70_000
    )
    original_decode = frontier_module.decode_source_ids

    def guarded_decode(raw):
        if raw == padded_values[1]:
            raise AssertionError("second oversized page row reached Python decode")
        return original_decode(raw)

    monkeypatch.setattr(frontier_module, "decode_source_ids", guarded_decode)
    try:
        with pytest.raises(RuntimeError, match="serialized byte bound"):
            engine._frontier.list_pending_batches("conversation")
        selects = [
            statement
            for statement in statements
            if "FROM lcm_prepared_batches" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert selects
        assert all("SUBSTR" in statement.upper() for statement in selects)
        assert "LIMIT 1" in selects[0].upper()
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_pending_batch_deadline_is_rechecked_after_sql_before_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    engine._frontier.create_batch_cas(
        conversation_id="conversation",
        session_id="current",
        base_generation=int(active["generation"]),
        source_end_store_id=1,
        source_identity_hash="hash",
        source_ids=[1],
        policy_fingerprint="policy",
        route_fingerprint="route",
    )
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(frontier_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(frontier_module, "_PENDING_BATCH_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(
        frontier_module,
        "decode_source_ids",
        lambda _raw: (_ for _ in ()).throw(AssertionError("decoded after deadline")),
    )
    try:
        with pytest.raises(RuntimeError, match="deadline"):
            engine._frontier.list_pending_batches("conversation")
    finally:
        engine.shutdown()


def test_pending_batch_enumeration_row_cap_is_hard(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine._frontier.ensure_frontier("conversation", "current")
    active = engine._frontier.get_active_frontier("conversation")
    assert active is not None
    for source_end in (1, 2):
        engine._frontier.create_batch_cas(
            conversation_id="conversation",
            session_id="current",
            base_generation=int(active["generation"]),
            source_end_store_id=source_end,
            source_identity_hash=f"hash-{source_end}",
            source_ids=[source_end],
            policy_fingerprint="policy",
            route_fingerprint="route",
        )
    monkeypatch.setattr(frontier_module, "_PENDING_BATCH_MAX_ROWS", 1)
    try:
        with pytest.raises(RuntimeError, match="row bound"):
            engine._frontier.list_pending_batches("conversation")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("limit_kind", ["bytes", "deadline"])
def test_foreground_identity_budget_exhaustion_rolls_back(
    tmp_path, monkeypatch, limit_kind
):
    engine = _engine(tmp_path)
    source_id = engine._store.append(
        "current", {"role": "user", "content": "identity " + ("x" * 4_000)}
    )
    engine._frontier.ensure_frontier("conversation", "current")
    before_generation = engine._frontier.get_active_frontier("conversation")["generation"]

    def tighten_after_lock(phase):
        if phase != "after_begin":
            return
        if limit_kind == "bytes":
            monkeypatch.setattr(
                engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 256
            )
        else:
            monkeypatch.setattr(
                engine_module, "_PUBLICATION_LOCKED_DEADLINE_SECONDS", 0.0
            )

    engine._foreground_publish_crash_hook = tighten_after_lock
    try:
        with pytest.raises(RuntimeError, match="byte bound|deadline"):
            engine._publish_foreground_leaf(
                node=SummaryNode(
                    session_id="current",
                    depth=0,
                    summary="must not publish",
                    token_count=1,
                    source_token_count=1,
                    source_ids=[source_id],
                    source_type="messages",
                    created_at=1.0,
                ),
                source_end_store_id=source_id,
                covered_source_ids=[source_id],
            )
        assert engine._dag.get_session_node_count("current") == 0
        assert engine._frontier.get_active_frontier("conversation")["generation"] == before_generation
    finally:
        engine._foreground_publish_crash_hook = None
        engine.shutdown()


@pytest.mark.parametrize("column", ["content", "tool_calls"])
def test_foreground_identity_rejects_nested_representation_before_decode(
    tmp_path, column
):
    engine = _engine(tmp_path)
    source_id = engine._store.append(
        "current", {"role": "assistant", "content": "safe"}
    )
    nested = "0"
    for _ in range(engine_module._PUBLICATION_LOCKED_MAX_NESTED_DEPTH + 2):
        nested = "[" + nested + "]"
    engine._store._conn.execute(
        f"UPDATE messages SET {column}=? WHERE store_id=?", (nested, source_id)
    )
    engine._store._conn.commit()
    engine._frontier.ensure_frontier("conversation", "current")
    before_generation = engine._frontier.get_active_frontier("conversation")["generation"]
    try:
        with pytest.raises(RuntimeError, match="nested-depth"):
            engine._publish_foreground_leaf(
                node=SummaryNode(
                    session_id="current",
                    depth=0,
                    summary="must not publish nested source",
                    token_count=1,
                    source_token_count=1,
                    source_ids=[source_id],
                    source_type="messages",
                    created_at=1.0,
                ),
                source_end_store_id=source_id,
                covered_source_ids=[source_id],
            )
        assert engine._dag.get_session_node_count("current") == 0
        assert engine._frontier.get_active_frontier("conversation")["generation"] == before_generation
    finally:
        engine.shutdown()


@pytest.mark.parametrize("limit_kind", ["bytes", "deadline"])
def test_async_identity_budget_exhaustion_rolls_back(
    tmp_path, monkeypatch, limit_kind
):
    engine = _engine(tmp_path, fresh_tail_count=1)
    messages = [
        {"role": "system", "content": "system"},
        *[
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"async identity {index} " + ("x" * 1_000),
            }
            for index in range(8)
        ],
    ]
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda initial_chunk, focus_topic=None: (
            list(initial_chunk), 100, "prepared summary", 1, 1
        ),
    )
    engine.ingest(messages)
    batch = engine.prepare_background_compaction_once(messages, force=True)
    assert batch is not None and batch.state == "ready"
    before_generation = engine._frontier.get_active_frontier("conversation")["generation"]

    def tighten_after_lock(phase):
        if phase != "after_begin":
            return
        if limit_kind == "bytes":
            monkeypatch.setattr(
                engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 256
            )
        else:
            monkeypatch.setattr(
                engine_module, "_PUBLICATION_LOCKED_DEADLINE_SECONDS", 0.0
            )

    engine._async_compaction_publish_crash_hook = tighten_after_lock
    try:
        result = engine.promote_prepared_compaction(batch.batch_id, messages)
        assert result.promoted is False
        assert "byte bound" in result.reason or "deadline" in result.reason
        assert engine._dag.get_session_node_count("current") == 0
        assert engine._frontier.get_active_frontier("conversation")["generation"] == before_generation
    finally:
        engine._async_compaction_publish_crash_hook = None
        engine.shutdown()


def test_async_canonical_lineage_is_byte_paged_before_materialization(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, fresh_tail_count=1)
    messages = [
        {"role": "system", "content": "system"},
        *[
            {"role": "user", "content": f"async canonical {index} " + ("x" * 200)}
            for index in range(8)
        ],
    ]
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda initial_chunk, focus_topic=None: (
            list(initial_chunk), 100, "prepared summary", 1, 1
        ),
    )
    engine.ingest(messages)
    batch = engine.prepare_background_compaction_once(messages, force=True)
    assert batch is not None and batch.state == "ready"
    conn = engine._dag.connection
    assert conn is not None
    for source_id in (100_001, 100_002, 100_003):
        raw = "[" + (" " * 60_000) + str(source_id) + "]"
        conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at)
               VALUES ('current', 0, ?, 1, 1, ?, 'messages', 1)""",
            (f"unrelated-{source_id}", raw),
        )
    conn.commit()
    monkeypatch.setattr(engine_module, "_CANONICAL_LINEAGE_MAX_BYTES", 70_000)
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    try:
        result = engine.promote_prepared_compaction(batch.batch_id, messages)
        assert result.promoted is False
        assert "lineage byte bound" in result.reason
        lineage_selects = [
            statement
            for statement in statements
            if "FROM summary_nodes" in statement
            and "source_type = 'messages'" in statement
        ]
        assert lineage_selects
        assert all("SUBSTR" in statement.upper() for statement in lineage_selects)
        assert "LIMIT 1" in lineage_selects[0].upper()
        assert engine._dag.get_session_node_count("current") == 3
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


@pytest.mark.parametrize("limit_kind", ["rows", "bytes", "deadline"])
def test_rollover_preflight_budget_exhaustion_is_lossless(
    tmp_path, monkeypatch, limit_kind
):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "rollover " + ("x" * 4_000)}]
    engine.ingest(messages)
    before_session = engine.current_session_id
    before_rows = len(engine._store.get_range("current"))
    before_frontier = engine._frontier.get_active_frontier("conversation")
    monkeypatch.setattr(
        engine,
        "_publish_rollover_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rollover publication reached after exhausted preflight")
        ),
    )
    if limit_kind == "rows":
        monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_ROWS", 0)
    elif limit_kind == "bytes":
        monkeypatch.setattr(
            engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 256
        )
    else:
        ticks = iter((0.0, 0.0, 3.0))
        monkeypatch.setattr(engine_module.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(
            engine_module, "_PUBLICATION_LOCKED_DEADLINE_SECONDS", 2.0
        )
    try:
        with pytest.raises(RuntimeError, match="row bound|byte bound|deadline"):
            engine.rollover_session("current", "next", previous_messages=messages)
        assert engine.current_session_id == before_session
        assert len(engine._store.get_range("current")) == before_rows
        assert engine._frontier.get_active_frontier("conversation") == before_frontier
    finally:
        engine.shutdown()


def test_load_session_rejects_oversized_roles_before_storage_read(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine._store,
        "count_session_load_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("storage read")),
    )
    try:
        too_many = [f"role-{index}" for index in range(
            tools_module._LCM_LOAD_SESSION_MAX_ROLES + 1
        )]
        result = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current", "roles": too_many}
        ))
        assert "roles count" in result["error"]
        too_long = "r" * (tools_module._LCM_LOAD_SESSION_MAX_ROLE_CHARS + 1)
        result = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current", "roles": [too_long]}
        ))
        assert "role string" in result["error"]
    finally:
        engine.shutdown()


def test_load_session_oversized_tool_calls_is_guarded_before_json_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "assistant", "content": "safe", "tool_calls": [{"id": "ok"}]}
    )
    raw = json.dumps([{"arguments": "x" * 80_000}])
    engine._store._conn.execute(
        "UPDATE messages SET tool_calls=? WHERE store_id=?", (raw, store_id)
    )
    engine._store._conn.commit()
    original_loads = store_module.json.loads

    def guarded_loads(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("oversized tool_calls decoded")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(store_module.json, "loads", guarded_loads)
    try:
        response_text = engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current"}
        )
        assert len(response_text.encode("utf-8")) <= tools_module._LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES
        response = json.loads(response_text)
        assert response["messages"][0]["tool_calls_truncated"] is True
        assert "tool_calls" not in response["messages"][0]
        assert response["serialized_bytes"] <= response["serialized_byte_limit"]
    finally:
        engine.shutdown()


def test_foreground_publication_pages_locked_frontier_and_tail(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    store_ids = [
        engine._store.append(
            "current", {"role": "user", "content": f"message-{index}"}
        )
        for index in range(5)
    ]
    node_ids = []
    for source_id in store_ids[:2]:
        node_ids.append(engine._dag.add_node(SummaryNode(
            session_id="current",
            depth=0,
            summary=f"existing-{source_id}",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        )))
    engine._frontier.ensure_frontier(
        "conversation", "current", source_end_store_id=store_ids[1]
    )
    engine._frontier.set_frontier_items(
        "conversation",
        1,
        [
            {
                "kind": "node",
                "ref_id": node_id,
                "source_start": source_id,
                "source_end": source_id,
            }
            for node_id, source_id in zip(node_ids, store_ids[:2])
        ],
    )
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_QUERY_BATCH", 1)
    try:
        result = engine._publish_foreground_leaf(
            node=SummaryNode(
                session_id="current",
                depth=0,
                summary="paged candidate",
                token_count=1,
                source_token_count=1,
                source_ids=[store_ids[2]],
                source_type="messages",
                created_at=2.0,
            ),
            source_end_store_id=store_ids[2],
            covered_source_ids=[store_ids[2]],
        )
        assert result["published"] is True
        frontier_selects = [
            statement for statement in statements
            if "FROM lcm_frontier_items AS i" in statement
        ]
        tail_selects = [
            statement for statement in statements
            if "FROM messages" in statement
            and "store_id >" in statement
            and "ORDER BY store_id" in statement
        ]
        assert len(frontier_selects) >= 2
        assert len(tail_selects) >= 2
        assert all("LIMIT" in statement.upper() for statement in frontier_selects + tail_selects)
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_rollover_publication_pages_locked_frontier(tmp_path, monkeypatch):
    engine = _engine(tmp_path, new_session_retain_depth=-1)
    store_ids = [
        engine._store.append(
            "current", {"role": "user", "content": f"rollover-{index}"}
        )
        for index in range(3)
    ]
    node_ids = [
        engine._dag.add_node(SummaryNode(
            session_id="current",
            depth=0,
            summary=f"rollover-node-{source_id}",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
            source_type="messages",
            created_at=1.0,
        ))
        for source_id in store_ids
    ]
    engine._frontier.ensure_frontier(
        "conversation", "current", source_end_store_id=store_ids[-1]
    )
    engine._frontier.set_frontier_items(
        "conversation",
        1,
        [
            {
                "kind": "node",
                "ref_id": node_id,
                "source_start": source_id,
                "source_end": source_id,
            }
            for node_id, source_id in zip(node_ids, store_ids)
        ],
    )
    statements: list[str] = []
    engine._frontier._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_QUERY_BATCH", 1)
    try:
        moved = engine._publish_rollover_state(
            "conversation",
            "current",
            "next",
            carry_over_context=True,
            final_tail=[],
        )
        assert moved == 3
        selects = [
            statement for statement in statements
            if "FROM lcm_frontier_items AS i" in statement
        ]
        assert len(selects) >= 3
        assert all("LIMIT" in statement.upper() for statement in selects)
        active = engine._frontier.get_active_frontier("conversation")
        assert active["generation"] == 2
        assert active["session_id"] == "next"
    finally:
        engine._frontier._conn.set_trace_callback(None)
        engine.shutdown()


def test_expand_schema_and_runtime_caps_apply_before_storage_reads(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    maximum = LCM_EXPAND["parameters"]["properties"]["max_tokens"]["maximum"]
    source_maximum = LCM_EXPAND["parameters"]["properties"]["source_limit"]["maximum"]
    monkeypatch.setattr(
        engine._store,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DB read occurred")),
    )
    try:
        result = json.loads(tools_module.lcm_expand(
            {"store_id": 1, "max_tokens": maximum + 1, "source_limit": source_maximum + 1},
            engine=engine,
        ))
        assert "hard cap" in result["error"]
    finally:
        engine.shutdown()


def test_expand_query_rejects_oversized_current_inputs_before_search_or_synthesis(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    prompt_max = LCM_EXPAND_QUERY["parameters"]["properties"]["prompt"]["maxLength"]
    monkeypatch.setattr(
        engine._dag,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DB search occurred")),
    )
    monkeypatch.setattr(
        tools_module,
        "_synthesize_expansion_answer",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("synthesis occurred")),
    )
    try:
        result = json.loads(tools_module.lcm_expand_query(
            {"prompt": "p" * (prompt_max + 1), "query": "archive"}, engine=engine
        ))
        assert "exceeds" in result["error"]
    finally:
        engine.shutdown()


def test_cross_session_grep_and_store_expand_require_host_capability(tmp_path):
    engine = _engine(tmp_path)
    foreign_id = engine._store.append(
        "foreign", {"role": "user", "content": "foreign capability canary"}
    )
    try:
        grep = json.loads(tools_module.lcm_grep(
            {"query": "canary", "session_scope": "session", "session_id": "foreign"},
            engine=engine,
        ))
        assert "trusted host capability" in grep["error"]
        expanded = json.loads(tools_module.lcm_expand({"store_id": foreign_id}, engine=engine))
        assert "trusted host capability" in expanded["error"]

        capability = engine.issue_cross_session_capability(["foreign"])
        allowed = json.loads(tools_module.lcm_expand(
            {"store_id": foreign_id}, engine=engine, cross_session_capability=capability
        ))
        assert allowed["content"] == "foreign capability canary"
    finally:
        engine.shutdown()


def test_grep_mandatory_redacts_current_and_authorized_cross_session_fields(tmp_path):
    engine = _engine(tmp_path)
    token = "sk-proj-" + ("A" * 48)
    current_id = engine._store.append(
        "current",
        {"role": "user", "content": f"credential canary {token}"},
        source=token,
    )
    foreign_id = engine._store.append(
        "foreign",
        {"role": "user", "content": f"credential canary {token}"},
        source=token,
    )
    capability = engine.issue_cross_session_capability(["foreign"])
    try:
        current = json.loads(tools_module.lcm_grep(
            {"query": "credential", "session_scope": "current"}, engine=engine
        ))
        foreign = json.loads(tools_module.lcm_grep(
            {
                "query": "credential",
                "session_scope": "session",
                "session_id": "foreign",
            },
            engine=engine,
            cross_session_capability=capability,
        ))
        assert current["results"][0]["store_id"] == current_id
        assert foreign["results"][0]["store_id"] == foreign_id
        for response in (current, foreign):
            encoded = json.dumps(response, sort_keys=True)
            assert token not in encoded
            assert "LCM sensitive redaction" in encoded
            for hit in response["results"]:
                for field in ("snippet", "content", "title", "source"):
                    assert token not in str(hit.get(field, ""))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("query", ["boundedgrep", "bounded-grep"])
def test_grep_database_search_bounds_content_and_tool_calls_before_python(
    tmp_path, monkeypatch, query
):
    engine = _engine(tmp_path)
    secret = "sk-proj-" + ("A" * 48)
    content = f"boundedgrep bounded-grep credential {secret} " + ("x" * 1_000_000)
    store_id = engine._store.append(
        "current", {"role": "assistant", "content": "placeholder"}
    )
    tool_calls = json.dumps([
        {
            "id": "call_canary",
            "type": "function",
            "function": {"name": "canary", "arguments": "y" * 1_000_000},
        }
    ])
    engine._store._conn.execute(
        "UPDATE messages SET content=?, tool_calls=? WHERE store_id=?",
        (content, tool_calls, store_id),
    )
    engine._store._conn.commit()

    original_row_to_dict = engine._store._row_to_dict

    def guarded_row_to_dict(row):
        if row is not None:
            assert not (len(row) > 4 and isinstance(row[4], str)
                        and len(row[4]) > 32_000), "1 MB content reached Python"
            assert not (len(row) > 6 and isinstance(row[6], str)
                        and len(row[6]) > 32_000), "1 MB tool_calls reached Python"
        return original_row_to_dict(row)

    monkeypatch.setattr(engine._store, "_row_to_dict", guarded_row_to_dict)
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        response = json.loads(tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        ))
        assert response["results"][0]["store_id"] == store_id
        snippet = response["results"][0]["snippet"]
        assert "bounded" in snippet
        assert secret not in snippet
        assert "sk-proj-" not in snippet
        assert "LCM sensitive redaction" in snippet
        assert len(snippet) <= 300

        selects = [
            statement.upper()
            for statement in statements
            if "FROM MESSAGES" in statement.upper()
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert selects
        assert any("LENGTH(CAST(M.CONTENT AS BLOB))" in statement
                   or "LENGTH(CAST(CONTENT AS BLOB))" in statement
                   for statement in selects)
        assert any("LENGTH(CAST(M.TOOL_CALLS AS BLOB))" in statement
                   or "LENGTH(CAST(TOOL_CALLS AS BLOB))" in statement
                   for statement in selects)
        assert any("SUBSTR" in statement and "CONTENT" in statement
                   for statement in selects)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


@pytest.mark.parametrize(
    ("query", "content"),
    [
        ("longcredentialfts", "api_key=" + ("A" * 20_000) + " longcredentialfts"),
        ("long-credential-like", "api_key=" + ("B" * 20_000) + " long-credential-like"),
    ],
)
def test_grep_fail_closes_window_start_inside_long_credential(
    tmp_path, query, content
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    try:
        response_text = tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        )
        response = json.loads(response_text)
        assert response["results"][0]["store_id"] == store_id
        assert query in response["results"][0]["snippet"]
        assert "A" * 64 not in response_text
        assert "B" * 64 not in response_text
        assert "LCM sensitive redaction" in response_text
        assert len(response_text) < 10_000
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("query", "secret_body"),
    [
        (
            "quotedwindowfts",
            'password="' + (("swordfish password words\n") * 700)
            + "quotedwindowfts" + (("\nswordfish password words") * 20) + '"',
        ),
        (
            "quoted-window-like",
            'password="' + (("swordfish password words\n") * 700)
            + "quoted-window-like" + (("\nswordfish password words") * 20) + '"',
        ),
        (
            "pemwindowfts",
            "-----BEGIN PRIVATE KEY-----\n"
            + (("Qk9EWUxJTkVDQU5BUlk=" + "A" * 48 + "\n") * 220)
            + "pemwindowfts\n-----END PRIVATE KEY-----",
        ),
        (
            "pem-window-like",
            "-----BEGIN PRIVATE KEY-----\n"
            + (("Qk9EWUxJTkVDQU5BUlk=" + "B" * 48 + "\n") * 220)
            + "pem-window-like\n-----END PRIVATE KEY-----",
        ),
    ],
)
def test_grep_left_boundary_fail_closes_quoted_and_pem_credentials(
    tmp_path, query, secret_body
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": secret_body + "\nafter credential"}
    )
    try:
        response_text = tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        )
        response = json.loads(response_text)
        assert response["results"][0]["store_id"] == store_id
        snippet = response["results"][0]["snippet"]
        assert "swordfish" not in snippet.casefold()
        assert "password words" not in snippet.casefold()
        assert "Qk9EWUxJTkVDQU5BUlk" not in snippet
        assert "PRIVATE KEY" not in snippet
        assert "LCM sensitive redaction" in snippet
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("query", "secret_body", "leak_canary"),
    [
        (
            "oppositequotefts",
            'password="'
            + ("D" * 20_000)
            + "it's-still-secret \\\"escaped-double-quote\\\" "
            + "oppositequotefts DOUBLE-QUOTED-LEAK-CANARY"
            + '"',
            "DOUBLE-QUOTED-LEAK-CANARY",
        ),
        (
            "opposite-quote-like",
            "password='"
            + ("S" * 20_000)
            + 'say "still-secret" \\\'escaped-single-quote\\\' '
            + "opposite-quote-like SINGLE-QUOTED-LEAK-CANARY"
            + "'",
            "SINGLE-QUOTED-LEAK-CANARY",
        ),
    ],
)
def test_grep_truncated_quoted_window_ignores_opposite_quote_type(
    tmp_path, query, secret_body, leak_canary
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current",
        {"role": "user", "content": secret_body + "\nBENIGN-GREP-SUFFIX"},
    )
    try:
        response_text = tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        )
        response = json.loads(response_text)
        assert response["results"][0]["store_id"] == store_id
        assert leak_canary not in response_text
        assert "it's-still-secret" not in response_text
        assert 'say \\"still-secret\\"' not in response_text
        assert "escaped-double-quote" not in response_text
        assert "escaped-single-quote" not in response_text
        assert "LCM sensitive redaction" in response_text
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "nested_decoy",
    ["api_key='inner-decoy'", "password='same-key-decoy'"],
)
def test_grep_truncated_outer_credential_ignores_nested_assignment_decoys(
    tmp_path, nested_decoy
):
    engine = _engine(tmp_path)
    query = "nested-assignment-window-canary"
    leak_canary = "STILL-OUTER-SECRET"
    content = (
        'password="'
        + ("A" * 20_000)
        + nested_decoy
        + " "
        + leak_canary
        + " "
        + query
        + '"\nBENIGN-AFTER-OUTER-CREDENTIAL'
    )
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    try:
        response_text = tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        )
        response = json.loads(response_text)
        assert response["results"][0]["store_id"] == store_id
        assert leak_canary not in response_text
        assert "inner-decoy" not in response_text
        assert "same-key-decoy" not in response_text
        assert "LCM sensitive redaction" in response_text
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("query", "credential"),
    [
        (
            "quoted-leading-token-window",
            'password="' + ("A" * 96)
            + (("\nswordfish password words") * 400)[:8_120]
            + "\nquoted-leading-token-window\n"
            + (("swordfish password words\n") * 800)
            + '"',
        ),
        (
            "pem-leading-token-window",
            "-----BEGIN PRIVATE KEY-----\n"
            + ("B" * 96)
            + (("\nswordfish password words") * 400)[:8_120]
            + "\npem-leading-token-window\n"
            + (("swordfish password words\n") * 800)
            + "-----END PRIVATE KEY-----",
        ),
    ],
)
def test_grep_long_leading_token_redacts_through_credential_terminator(
    tmp_path, query, credential
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": credential + "\nbenign suffix"}
    )
    try:
        response = json.loads(tools_module.lcm_grep(
            {"query": query, "session_scope": "current", "limit": 1},
            engine=engine,
        ))
        assert response["results"][0]["store_id"] == store_id
        snippet = response["results"][0]["snippet"]
        assert "swordfish" not in snippet.casefold()
        assert "password words" not in snippet.casefold()
        assert "A" * 64 not in snippet
        assert "B" * 64 not in snippet
        assert "LCM sensitive redaction" in snippet
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "field",
    ["source", "conversation_id", "session_scope", "session_id", "content_scope", "ref", "role", "sort"],
)
def test_grep_rejects_oversized_metadata_before_discovery(tmp_path, monkeypatch, field):
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine._store,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized grep metadata reached discovery")
        ),
    )
    args = {"query": "metadata-cap-canary", field: "x" * 20_000}
    if field == "session_id":
        args["session_scope"] = "session"
    try:
        response = json.loads(tools_module.lcm_grep(args, engine=engine))
        assert "error" in response
        assert "hard" in response["error"] or "limit" in response["error"]
        assert "x" * 1_000 not in json.dumps(response)
    finally:
        engine.shutdown()


def test_grep_charges_bounded_echo_metadata_to_operation_budget(tmp_path):
    engine = _engine(tmp_path)
    source = "source-" + ("s" * 400)
    conversation_id = "conversation-" + ("c" * 400)
    try:
        response = json.loads(tools_module.lcm_grep(
            {
                "query": "no-such-budget-result",
                "source": source,
                "conversation_id": conversation_id,
            },
            engine=engine,
        ))
        minimum = len("no-such-budget-result".encode()) + len(source.encode()) + len(conversation_id.encode())
        assert response["operation_budget"]["bytes_materialized"] >= minimum
    finally:
        engine.shutdown()


@pytest.mark.parametrize("regex_mode", [False, True])
def test_grep_rejects_oversized_query_before_any_discovery(tmp_path, monkeypatch, regex_mode):
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine._store,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized query reached raw discovery")
        ),
    )
    try:
        response = json.loads(tools_module.lcm_grep(
            {"query": "q" * 2_001, "regex": regex_mode}, engine=engine
        ))
        assert "query" in response["error"] or "pattern" in response["error"]
        assert "limit" in response["error"]
    finally:
        engine.shutdown()


def test_grep_uses_one_discovery_and_response_budget_across_many_sessions(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    session_ids = [f"archive-{index}" for index in range(10)]
    for session_id in session_ids:
        for row_index in range(30):
            engine._store.append(
                session_id,
                {
                    "role": "user",
                    "content": (
                        f"operationwidegrep {session_id} {row_index} "
                        + ("payload " * 2_000)
                    ),
                },
            )
    capability = engine.issue_cross_session_capability(session_ids)
    original_search = engine._store.search
    requested_rows = 0

    def counted_search(*args, **kwargs):
        nonlocal requested_rows
        requested_rows += int(kwargs["limit"])
        return original_search(*args, **kwargs)

    monkeypatch.setattr(engine._store, "search", counted_search)
    try:
        response_text = tools_module.lcm_grep(
            {
                "query": "operationwidegrep",
                "session_scope": "all",
                "limit": 200,
            },
            engine=engine,
            cross_session_capability=capability,
        )
        response = json.loads(response_text)
        assert response["results"]
        assert requested_rows <= 1_000
        assert response.get("operation_budget", {}).get("rows_limit") == 1_000
        assert len(response_text.encode("utf-8")) <= 2 * 1024 * 1024
    finally:
        engine.shutdown()


def test_grep_deadline_is_rechecked_after_sort_before_redaction_and_echo(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    expired = False

    class FakeClock:
        @staticmethod
        def monotonic():
            return 2.0 if expired else 0.0

        @staticmethod
        def time():
            return time.time()

    monkeypatch.setattr(tools_module, "time", FakeClock)
    monkeypatch.setattr(
        engine._store,
        "search",
        lambda *_args, **_kwargs: [{
            "store_id": 1,
            "session_id": "current",
            "source": "test",
            "conversation_id": "conversation",
            "role": "user",
            "timestamp": 1.0,
            "snippet": "deadlinecanary",
            "_grep_window_start": 1,
        }],
    )
    original_sort_key = tools_module._combined_result_sort_key

    def expire_during_sort(result, sort):
        nonlocal expired
        expired = True
        return original_sort_key(result, sort)

    monkeypatch.setattr(tools_module, "_combined_result_sort_key", expire_during_sort)
    try:
        response = json.loads(tools_module.lcm_grep(
            {
                "query": "deadlinecanary",
                "session_scope": "current",
                "role": "user",
                "limit": 1,
            },
            engine=engine,
        ))
        assert response["results"] == []
        assert response["operation_budget"]["exhausted"] is True
    finally:
        engine.shutdown()


def test_unauthorized_store_expand_never_reads_large_payload_columns(tmp_path):
    engine = _engine(tmp_path)
    foreign_id = engine._store.append(
        "foreign", {"role": "user", "content": "X" * 1_000_000}
    )

    def deny_payload_read(action, _arg1, column, _db_name, _trigger):
        if action == sqlite3.SQLITE_READ and column in {"content", "tool_calls"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    engine._store._conn.set_authorizer(deny_payload_read)
    try:
        response = json.loads(tools_module.lcm_expand(
            {"store_id": foreign_id}, engine=engine
        ))
        assert "trusted host capability" in response["error"]
    finally:
        engine._store._conn.set_authorizer(None)
        engine.shutdown()


def test_authorized_store_expand_materializes_only_bounded_redacted_payload(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    secret = "api_key=" + ("X" * 1_000_000)
    foreign_id = engine._store.append(
        "foreign", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (secret, foreign_id)
    )
    engine._store._conn.commit()
    original_row_to_dict = engine._store._row_to_dict

    def guarded_row_to_dict(row):
        assert all(
            not isinstance(value, str) or len(value) < 150_000 for value in row
        ), "unbounded expansion payload reached Python"
        return original_row_to_dict(row)

    monkeypatch.setattr(engine._store, "_row_to_dict", guarded_row_to_dict)
    capability = engine.issue_cross_session_capability(["foreign"])
    try:
        response_text = tools_module.lcm_expand(
            {"store_id": foreign_id, "max_tokens": 128},
            engine=engine,
            cross_session_capability=capability,
        )
        response = json.loads(response_text)
        assert response["store_id"] == foreign_id
        assert response["content_chars"] == len(secret)
        assert response["content_truncated"] is True
        assert "X" * 64 not in response_text
        assert "api_key=" not in response_text
        assert "LCM sensitive redaction" in response_text
    finally:
        engine.shutdown()


@pytest.mark.parametrize("credential_kind", ["quoted", "pem"])
def test_store_expand_arbitrary_deep_offset_fail_closes_and_recovers_suffix(
    tmp_path, credential_kind
):
    engine = _engine(tmp_path)
    if credential_kind == "quoted":
        credential = 'password="' + ("QUOTED-BODY-CANARY\n" * 8_000) + '"'
    else:
        credential = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("PEM-BODY-CANARY-0123456789\n" * 8_000)
            + "-----END PRIVATE KEY-----"
        )
    suffix = "\nBENIGN-SUFFIX-RECOVERED"
    content = "benign prefix\n" + credential + suffix
    store_id = engine._store.append(
        "current", {"role": "user", "content": content}
    )
    offset = content.index("BODY-CANARY") + 20_000
    pages = []
    try:
        for _ in range(80):
            page = json.loads(tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": offset, "max_tokens": 64},
                engine=engine,
            ))
            pages.append(page["content"])
            encoded = json.dumps(page)
            assert "QUOTED-BODY-CANARY" not in encoded
            assert "PEM-BODY-CANARY" not in encoded
            if not page["has_more"]:
                break
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        else:
            pytest.fail("bounded credential scan did not reach a terminal page")
        assert "BENIGN-SUFFIX-RECOVERED" in "".join(pages)
        assert pages and any("LCM sensitive redaction" in item for item in pages)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("credential", "leak_canary"),
    [
        (
            'password="'
            + ("Q" * 24_000)
            + "it's-still-secret \\\"escaped-matching-quote\\\" "
            + "QUOTED-LEAK-CANARY"
            + ("R" * 2_000)
            + '"',
            "QUOTED-LEAK-CANARY",
        ),
        (
            "password=" + ("U" * 24_000) + "UNQUOTED-LEAK-CANARY" + ("V" * 2_000),
            "UNQUOTED-LEAK-CANARY",
        ),
    ],
)
def test_store_expand_arbitrary_offset_detects_deep_assignment_and_recovers_suffix(
    tmp_path, credential, leak_canary
):
    engine = _engine(tmp_path)
    suffix = "\nBENIGN-DEEP-OFFSET-SUFFIX"
    content = "benign prefix\n" + credential + suffix
    store_id = engine._store.append(
        "current", {"role": "user", "content": content}
    )
    offset = content.index(leak_canary)
    assert offset > 20_000
    pages = []
    try:
        for _ in range(80):
            page = json.loads(tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": offset, "max_tokens": 64},
                engine=engine,
            ))
            pages.append(page["content"])
            assert leak_canary not in json.dumps(page)
            if not page["has_more"]:
                break
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        else:
            pytest.fail("bounded assignment scan did not reach the benign suffix")
        assert "BENIGN-DEEP-OFFSET-SUFFIX" in "".join(pages)
        assert any("LCM sensitive redaction" in item for item in pages)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("credential", "leak_canary"),
    [
        (
            r'{\"password\":\"'
            + ("J" * 24_000)
            + "JSON-ESCAPED-LEAK-CANARY"
            + r'\"}',
            "JSON-ESCAPED-LEAK-CANARY",
        ),
        (
            "password="
            + ("U" * 24_000)
            + "!punctuation:still-secret?UNQUOTED-PUNCTUATION-LEAK-CANARY",
            "UNQUOTED-PUNCTUATION-LEAK-CANARY",
        ),
    ],
    ids=["json-escaped", "unquoted-punctuation"],
)
def test_store_expand_deep_offset_handles_escaped_json_and_punctuation(
    tmp_path, credential, leak_canary
):
    engine = _engine(tmp_path)
    suffix = "\nBENIGN-CONSERVATIVE-TERMINATOR-SUFFIX"
    content = "benign prefix\n" + credential + suffix
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    offset = content.index(leak_canary)
    assert offset > 20_000
    pages = []
    try:
        for _ in range(80):
            page = json.loads(tools_module.lcm_expand(
                {"store_id": store_id, "content_offset": offset, "max_tokens": 64},
                engine=engine,
            ))
            pages.append(page["content"])
            assert page["store_id"] == store_id
            assert leak_canary not in json.dumps(page)
            if not page["has_more"]:
                break
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        else:
            pytest.fail("deep-offset credential scan did not reach a safe suffix")
        assert "BENIGN-CONSERVATIVE-TERMINATOR-SUFFIX" in "".join(pages)
        assert any("LCM sensitive redaction" in item for item in pages)
    finally:
        engine.shutdown()


def test_store_expand_sequential_benign_pages_cross_two_mib_losslessly(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    unit = "ordinary benign paging record 0123456789\n"
    content = (unit * ((2_400_000 // len(unit)) + 1)) + "BENIGN-TAIL-CANARY"
    assert len(content) > 2_300_000
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    original_row_to_dict = engine._store._row_to_dict

    def bounded_row_to_dict(row):
        assert all(
            not isinstance(value, str) or len(value) < 150_000 for value in row
        ), "whole benign row was materialized during paging"
        return original_row_to_dict(row)

    monkeypatch.setattr(engine._store, "_row_to_dict", bounded_row_to_dict)
    offset = 0
    recovered = []
    try:
        for _ in range(80):
            page = json.loads(tools_module.lcm_expand(
                {
                    "store_id": store_id,
                    "content_offset": offset,
                    "max_tokens": 65_536,
                },
                engine=engine,
            ))
            assert page["store_id"] == store_id
            assert "LCM sensitive redaction" not in page["content"]
            recovered.append(page["content"])
            if not page["has_more"]:
                break
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        else:
            pytest.fail("benign paging did not reach EOF")
        assert "".join(recovered) == content
        assert recovered[-1].endswith("BENIGN-TAIL-CANARY")
    finally:
        engine.shutdown()


def test_store_expand_twenty_mb_deep_offset_is_per_call_bounded_and_resumable(
    tmp_path,
):
    engine = _engine(tmp_path)
    content = ("ordinary-20mb-record\n" * 1_000_001)[:20_000_000]
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    statements = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        page = json.loads(tools_module.lcm_expand(
            {
                "store_id": store_id,
                "content_offset": 19_000_000,
                "max_tokens": 64,
            },
            engine=engine,
        ))
        content_reads = [
            statement for statement in statements
            if "substr(cast(content" in statement.lower()
        ]
        assert len(content_reads) <= 140
        assert page["content_boundary_scan_pending"] is True
        assert page["content"] and "LCM sensitive redaction" in page["content"]
        assert 0 < page["content_scan_checkpoint_offset"] < 19_000_000

        engine._store._conn.set_trace_callback(None)
        engine.shutdown()
        engine = _engine(tmp_path)
        statements.clear()
        engine._store._conn.set_trace_callback(statements.append)
        retry = json.loads(tools_module.lcm_expand(
            {
                "store_id": store_id,
                "content_offset": 19_000_000,
                "max_tokens": 64,
            },
            engine=engine,
        ))
        retry_reads = [
            statement for statement in statements
            if "substr(cast(content" in statement.lower()
        ]
        assert len(retry_reads) <= 140
        assert retry["content_scan_checkpoint_offset"] > page["content_scan_checkpoint_offset"]
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_store_expand_twenty_mb_deadline_fails_closed_before_deep_scan(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    content = ("deadline-20mb-record\n" * 1_000_001)[:20_000_000]
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    statements = []
    engine._store._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        tools_module, "_EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS", 0.0
    )
    try:
        page = json.loads(tools_module.lcm_expand(
            {"store_id": store_id, "content_offset": 19_000_000, "max_tokens": 64},
            engine=engine,
        ))
        content_reads = [
            statement for statement in statements
            if "substr(cast(content" in statement.lower()
        ]
        assert len(content_reads) <= 3
        assert page["content_boundary_scan_pending"] is True
        assert page["content_scan_checkpoint_offset"] == 0
        assert "LCM sensitive redaction" in page["content"]
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_store_expand_twenty_mb_sequential_pages_do_not_rescan_from_zero(tmp_path):
    engine = _engine(tmp_path)
    content = ("sequential-20mb-record\n" * 1_000_001)[:20_000_000]
    store_id = engine._store.append(
        "current", {"role": "user", "content": "placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    offset = 0
    page_query_counts = []
    try:
        for _ in range(4):
            statements = []
            engine._store._conn.set_trace_callback(statements.append)
            page = json.loads(tools_module.lcm_expand(
                {
                    "store_id": store_id,
                    "content_offset": offset,
                    "max_tokens": 65_536,
                },
                engine=engine,
            ))
            page_query_counts.append(sum(
                "substr(cast(content" in statement.lower()
                for statement in statements
            ))
            assert page["has_more"] is True
            assert page["next_content_offset"] > offset
            offset = page["next_content_offset"]
        assert max(page_query_counts) <= 20
        assert page_query_counts[-1] <= page_query_counts[1] + 2
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_store_expand_checkpoint_is_invalidated_after_same_length_rewrite(tmp_path):
    engine = _engine(tmp_path)
    original = ("ordinary-safe-line\n" * 20_000)[:300_000] + "ORIGINAL-SAFE-SUFFIX"
    store_id = engine._store.append(
        "current", {"role": "user", "content": original}
    )
    try:
        first = json.loads(tools_module.lcm_expand(
            {"store_id": store_id, "content_offset": 250_000, "max_tokens": 64},
            engine=engine,
        ))
        assert "ordinary-safe-line" in first["content"]
        old_fingerprint = engine._store._conn.execute(
            "SELECT content_fingerprint FROM lcm_content_revisions WHERE store_id=?",
            (store_id,),
        ).fetchone()[0]
        old_checkpoint = engine._store._conn.execute(
            """SELECT MAX(char_offset) FROM lcm_content_scan_checkpoints
               WHERE store_id=? AND content_fingerprint=?""",
            (store_id, old_fingerprint),
        ).fetchone()[0]
        assert old_checkpoint == 250_000

        rewritten = (
            'password="' + ("REWRITE-SECRET-CANARY" * 14_285) + '"\nSAFE'
        )[:len(original)]
        rewritten = rewritten.ljust(len(original), "Z")
        engine._store._conn.execute(
            "UPDATE messages SET content=? WHERE store_id=?",
            (rewritten, store_id),
        )
        engine._store._conn.commit()
        new_fingerprint = engine._store._conn.execute(
            "SELECT content_fingerprint FROM lcm_content_revisions WHERE store_id=?",
            (store_id,),
        ).fetchone()[0]
        assert new_fingerprint != old_fingerprint
        assert engine._store._conn.execute(
            "SELECT COUNT(*) FROM lcm_content_scan_checkpoints WHERE store_id=?",
            (store_id,),
        ).fetchone()[0] == 0
        second = json.loads(tools_module.lcm_expand(
            {"store_id": store_id, "content_offset": 250_000, "max_tokens": 64},
            engine=engine,
        ))
        assert "REWRITE-SECRET-CANARY" not in json.dumps(second)
        assert "LCM sensitive redaction" in second["content"]
    finally:
        engine.shutdown()


def test_grep_combined_sources_never_exceed_operation_row_budget(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)

    def database_hits(*_args, **_kwargs):
        return [
            {
                "store_id": index + 1,
                "session_id": "current",
                "source": "test",
                "conversation_id": "conversation",
                "role": "user",
                "timestamp": float(index),
                "snippet": "row-budget-canary",
                "_grep_window_start": 1,
            }
            for index in range(800)
        ]

    def external_hits(*_args, **_kwargs):
        hits = [
            {
                "type": "externalized",
                "ref": f"payload-{index}.json",
                "session_id": "current",
                "line": 1,
                "char_offset": 0,
                "byte_offset": 0,
                "matched_text": "row-budget-canary",
                "snippet": "row-budget-canary",
                "payload_truncated": False,
                "content_chars_scanned": 20,
                "_sort_ts": 0.0,
                "_sort_rank": 0.0,
                "_sort_directness": 0.0,
            }
            for index in range(600)
        ]
        return hits, [], {
            "files_scanned": 600,
            "entries_scanned": 600,
            "bytes_scanned": 1,
            "matches": 600,
            "scan_truncated": False,
        }

    monkeypatch.setattr(engine._store, "search", database_hits)
    monkeypatch.setattr(tools_module, "_search_externalized_payloads", external_hits)
    try:
        response = json.loads(tools_module.lcm_grep(
            {
                "query": "row-budget-canary",
                "content_scope": "all",
                "role": "user",
                "limit": 200,
            },
            engine=engine,
        ))
        budget = response["operation_budget"]
        assert response["results"]
        assert budget["rows_materialized"] <= budget["rows_limit"] == 1_000
        assert budget["rows_reserved"] <= budget["rows_limit"]
        assert budget["exhausted"] is True
    finally:
        engine.shutdown()


def test_grep_external_entries_are_charged_before_discovery(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    charged = {"database": 0, "external": 0}

    def database_hits(*_args, **kwargs):
        charged["database"] += kwargs["max_candidate_rows"]
        return []

    def external_hits(*_args, **kwargs):
        charged["external"] += kwargs["max_files"]
        return [], [], {
            "files_scanned": 0,
            "entries_scanned": kwargs["max_files"],
            "bytes_scanned": 0,
            "matches": 0,
            "scan_truncated": True,
        }

    monkeypatch.setattr(engine._store, "search", database_hits)
    monkeypatch.setattr(tools_module, "_search_externalized_payloads", external_hits)
    try:
        response = json.loads(tools_module.lcm_grep(
            {
                "query": "charged-before-open",
                "content_scope": "all",
                "role": "user",
                "limit": 200,
                "max_files": 500,
            },
            engine=engine,
        ))
        assert charged["database"] + charged["external"] <= 1_000
        assert charged == {"database": 500, "external": 500}
        assert response["operation_budget"]["rows_reserved"] == 1_000
        assert response["operation_budget"]["exhausted"] is True
    finally:
        engine.shutdown()


@pytest.mark.parametrize("session_first", [True, False])
def test_externalized_grep_authorizes_metadata_before_foreign_payload_read(
    tmp_path, monkeypatch, session_first
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    foreign_path = payload_dir / f"foreign-{session_first}.json"
    foreign_canary = "FOREIGN-PAYLOAD-BODY-MUST-NOT-BE-READ"
    payload = (
        {
            "kind": "ingest_payload",
            "session_id": "foreign-session",
            "content": foreign_canary,
        }
        if session_first
        else {
            "content": foreign_canary,
            "session_id": "foreign-session",
        }
    )
    foreign_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    content_byte_offset = foreign_path.read_bytes().index(foreign_canary.encode())
    engine = _engine(
        tmp_path, large_output_externalization_path=str(payload_dir)
    )
    original_open = tools_module.Path.open
    bytes_read = 0

    class GuardedForeignReader:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size=-1):
            nonlocal bytes_read
            if size < 0 or size > 1:
                raise AssertionError("foreign payload body read before authorization")
            chunk = self._handle.read(size)
            bytes_read += len(chunk)
            if bytes_read > content_byte_offset:
                raise AssertionError("foreign content bytes were read before authorization")
            return chunk

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def guarded_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.resolve() == foreign_path.resolve() and "b" in str(args[0] if args else kwargs.get("mode", "r")):
            return GuardedForeignReader(handle)
        return handle

    monkeypatch.setattr(tools_module.Path, "open", guarded_open)
    try:
        response = json.loads(tools_module.lcm_grep(
            {
                "query": "MUST-NOT-BE-READ",
                "content_scope": "externalized",
                "ref": foreign_path.name,
            },
            engine=engine,
        ))
        assert response["results"] == []
        assert response["diagnostics"] == [{
            "ref": foreign_path.name,
            "error": (
                "session_mismatch"
                if session_first
                else "session_metadata_unavailable"
            ),
        }]
        assert 0 < bytes_read <= content_byte_offset
    finally:
        engine.shutdown()


def test_externalized_metadata_reader_is_chunk_linear_and_deadline_bounded(
    monkeypatch,
):
    padding = "M" * 15_900
    payload = json.dumps({
        "padding": padding,
        "session_id": "current",
        "content": "BODY-MUST-NOT-BE-READ-BEFORE-AUTHORIZATION",
    }).encode("utf-8")
    body_offset = payload.index(b"BODY-MUST-NOT-BE-READ")

    class CountedReader(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.read_calls = 0

        def read(self, size=-1):
            self.read_calls += 1
            assert size == 1
            return super().read(size)

    reader = CountedReader(payload)
    started = time.perf_counter()
    prefix, content_seen, truncated = (
        tools_module._read_externalized_payload_metadata_prefix_from_handle(
            reader,
            deadline=time.monotonic() + 1.0,
        )
    )
    elapsed = time.perf_counter() - started
    assert content_seen is True
    assert truncated is False
    assert '"session_id": "current"' in prefix
    assert reader.tell() <= body_offset
    assert reader.read_calls <= body_offset
    assert elapsed < 0.5

    expired_reader = CountedReader(payload)
    ticks = iter([0.0, 0.1, 0.6, 0.7])
    monkeypatch.setattr(tools_module, "_external_metadata_now", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        tools_module._read_externalized_payload_metadata_prefix_from_handle(
            expired_reader,
            deadline=0.5,
        )
    assert expired_reader.read_calls <= 512


def test_store_expand_metadata_and_payload_reads_share_one_snapshot(tmp_path):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "snapshot-original"}
    )
    other = sqlite3.connect(str(engine._config.database_path), timeout=5)
    try:
        def authorize_and_rewrite(session_id):
            assert session_id == "current"
            other.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("snapshot-concurrent-rewrite", store_id),
            )
            other.commit()
            return True

        loaded = engine._store.get_for_expansion(
            store_id,
            authorize_session=authorize_and_rewrite,
            content_offset=0,
            max_content_chars=1_000,
            content_lookahead_chars=0,
        )
        assert loaded["status"] == "ok"
        assert loaded["message"]["content"] == "snapshot-original"
        assert other.execute(
            "SELECT content FROM messages WHERE store_id=?", (store_id,)
        ).fetchone()[0] == "snapshot-concurrent-rewrite"
    finally:
        other.close()
        engine.shutdown()


@pytest.mark.parametrize("session_scope", ["current", "session"])
def test_grep_summary_projection_is_bounded_redacted_and_explicit(
    tmp_path, monkeypatch, session_scope
):
    engine = _engine(tmp_path)
    target_session = "current" if session_scope == "current" else "archive"
    secret = "github_pat_" + ("Z" * 64)
    store_id = engine._store.append(
        target_session, {"role": "user", "content": "summary source"}
    )
    node_id = engine._dag.add_node(SummaryNode(
        session_id=target_session,
        depth=0,
        summary="summaryprojection " + secret + ("s" * 1_000_000),
        token_count=10,
        source_token_count=20,
        source_ids=[store_id],
        source_type="messages",
        created_at=1.0,
        expand_hint="api_key=" + secret + ("h" * 1_000_000),
    ))
    original_row_to_node = engine._dag._row_to_node

    def guarded_row_to_node(row):
        if row is not None:
            assert all(
                not isinstance(value, str) or len(value) < 64_000 for value in row
            ), "unbounded summary metadata reached Python"
        return original_row_to_node(row)

    monkeypatch.setattr(engine._dag, "_row_to_node", guarded_row_to_node)
    statements: list[str] = []
    engine._dag._conn.set_trace_callback(statements.append)
    kwargs = {"engine": engine}
    args = {
        "query": "summaryprojection",
        "session_scope": session_scope,
        "source": "unknown",
        "limit": 1,
    }
    if session_scope == "session":
        args["session_id"] = target_session
        kwargs["cross_session_capability"] = engine.issue_cross_session_capability(
            [target_session]
        )
    try:
        response_text = tools_module.lcm_grep(args, **kwargs)
        response = json.loads(response_text)
        summary_hits = [hit for hit in response["results"] if hit["type"] == "summary"]
        assert summary_hits and summary_hits[0]["node_id"] == node_id
        assert secret not in response_text
        assert len(summary_hits[0]["snippet"]) <= 300
        assert len(summary_hits[0]["expand_hint"]) <= 2_048
        assert len(response_text) < 20_000
        selects = [
            statement.upper() for statement in statements
            if "SUMMARY_NODES" in statement.upper()
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert selects
        assert all("SELECT *" not in statement and "SELECT N.*" not in statement
                   for statement in selects)
        assert any("SUBSTR" in statement and "LENGTH(CAST" in statement
                   for statement in selects)
    finally:
        engine._dag._conn.set_trace_callback(None)
        engine.shutdown()


def test_load_session_requires_capability_and_bounds_redacts_nested_rows(tmp_path):
    engine = _engine(tmp_path)
    current_id = engine._store.append(
        "current", {"role": "user", "content": "current transcript remains usable"}
    )
    foreign_id = engine._store.append(
        "foreign", {"role": "assistant", "content": "placeholder"}
    )
    poisoned_tool_calls = json.dumps([
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "legacy",
                "arguments": '\\"password\\":\\"tool-call-secret ' + ("x" * 50_000),
            },
        }
    ])
    engine._store._conn.execute(
        "UPDATE messages SET content=?, tool_calls=? WHERE store_id=?",
        ('password=\\"content-secret', poisoned_tool_calls, foreign_id),
    )
    engine._store._conn.commit()
    try:
        denied = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "foreign"}
        ))
        assert "trusted host capability" in denied["error"]

        current = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current"}
        ))
        assert current["messages"][0]["store_id"] == current_id
        assert "remains usable" in current["messages"][0]["content"]

        allowed = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {"session_id": "foreign", "max_content_chars": 20_000},
            cross_session_capability=engine.issue_cross_session_capability(["foreign"]),
        ))
        encoded = json.dumps(allowed, sort_keys=True)
        assert "content-secret" not in encoded
        assert "tool-call-secret" not in encoded
        assert "LCM sensitive redaction" in encoded
        assert allowed["messages"][0]["serialized_truncated"] is True
        assert allowed["serialized_bytes"] <= allowed["serialized_byte_limit"]
    finally:
        engine.shutdown()


def test_load_session_shared_serialized_budget_stops_without_skipping_cursor(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    store_ids = [
        engine._store.append(
            "current",
            {"role": "user", "content": f"row-{index} " + ("x" * 1_000)},
        )
        for index in range(20)
    ]
    monkeypatch.setattr(
        tools_module, "_LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES", 20_000
    )
    monkeypatch.setattr(
        tools_module, "_LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES", 2_000
    )
    try:
        page = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current", "limit": 20}
        ))
        assert page["serialized_budget_exhausted"] is True
        assert 0 < page["returned_messages"] < len(store_ids)
        assert page["serialized_bytes"] <= page["serialized_byte_limit"]
        assert page["next_cursor"] == page["messages"][-1]["store_id"]

        next_page = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {
                "session_id": "current",
                "limit": 1,
                "after_store_id": page["next_cursor"],
            },
        ))
        assert next_page["messages"][0]["store_id"] == store_ids[page["returned_messages"]]
    finally:
        engine.shutdown()


def test_externalized_enumeration_does_not_glob_or_sort_the_whole_directory(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    for index in range(50):
        (payload_dir / f"{index}.json").write_text(
            json.dumps({"session_id": "current", "content": "needle"}),
            encoding="utf-8",
        )
    engine = _engine(tmp_path, large_output_externalization_path=str(payload_dir))
    monkeypatch.setattr(
        tools_module.Path,
        "glob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full glob used")),
    )
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "max_files": 3},
            engine=engine,
        ))
        assert result["scan"]["files_scanned"] <= 3
        assert result["scan"]["scan_truncated"] is True
    finally:
        engine.shutdown()


def test_externalized_scandir_counts_non_json_entries_at_the_hard_cap(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    engine = _engine(tmp_path, large_output_externalization_path=str(payload_dir))
    seen = {"count": 0}

    class _Entry:
        def __init__(self, name):
            self.name = name

    class _Scandir:
        def __init__(self):
            self._names = iter([*(f"junk-{index}.txt" for index in range(20)), "hit.json"])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            seen["count"] += 1
            if seen["count"] > 3:
                raise AssertionError("scandir advanced beyond the hard entry cap")
            return _Entry(next(self._names))

    monkeypatch.setattr(tools_module.os, "scandir", lambda _root: _Scandir())
    try:
        result = json.loads(tools_module.lcm_grep(
            {"query": "needle", "content_scope": "externalized", "max_files": 3},
            engine=engine,
        ))
        assert seen["count"] == 3
        assert result["scan"]["entries_scanned"] == 3
        assert result["scan"]["files_scanned"] == 0
        assert result["scan"]["scan_truncated"] is True
    finally:
        engine.shutdown()


def test_prompt_aware_terms_and_scored_summaries_have_fixed_cardinality(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, prompt_aware_eviction_enabled=True, max_assembly_tokens=50)
    calls = {"count": 0}
    original = engine._prompt_aware_relevance_score

    def counted(text, terms):
        calls["count"] += 1
        assert len(terms) <= engine_module._PROMPT_AWARE_MAX_TERMS
        return original(text, terms)

    monkeypatch.setattr(engine, "_prompt_aware_relevance_score", counted)
    conn = engine._dag.connection
    assert conn is not None
    conn.executemany(
        """INSERT INTO summary_nodes
           (session_id, depth, summary, token_count, source_token_count,
            source_ids, source_type, created_at)
           VALUES ('current', 0, ?, 1, 1, '[1]', 'messages', ?)""",
        [(f"summary {index}", float(index)) for index in range(800)],
    )
    conn.commit()
    prompt = " ".join(f"term{index}" for index in range(500))
    try:
        engine._assemble_context(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": prompt}],
            assembly_cap_override=50,
        )
        assert calls["count"] <= engine_module._PROMPT_AWARE_MAX_SUMMARIES
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("env_name", "yaml_key", "yaml_value"),
    [
        ("LCM_MODEL_THRESHOLDS", "model_thresholds", {"good": 0.4}),
        (
            "LCM_MODEL_POLICIES",
            "model_policies",
            {"good": {"compaction_mode": "inline"}},
        ),
    ],
)
def test_invalid_structured_env_preserves_yaml_and_records_warning(
    monkeypatch, env_name, yaml_key, yaml_value
):
    monkeypatch.setattr(
        config_module,
        "_load_hermes_config_yaml",
        lambda: {"lcm": {yaml_key: yaml_value}},
    )
    monkeypatch.setenv(env_name, "definitely-invalid")
    config = LCMConfig.from_env()
    assert getattr(config, yaml_key) == yaml_value
    assert any(env_name in warning for warning in config.config_source_warnings)


def test_preparation_admission_released_when_batch_creation_raises(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "payload " * 30}
        for index in range(8)
    ]
    engine.ingest(messages)
    monkeypatch.setattr(
        engine._frontier,
        "create_batch_cas",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("boom")),
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            engine.prepare_background_compaction_once(messages, force=True)
        assert active_profile_admissions(engine._store.db_path) == 0
    finally:
        engine.shutdown()
