"""Adversarial regressions for the final publication/security review blockers."""

from __future__ import annotations

import json
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
