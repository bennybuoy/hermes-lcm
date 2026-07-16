"""Non-vacuous regressions for the final publication/security blockers."""

from __future__ import annotations

import json
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.externalize as externalize_module
import hermes_lcm.frontier as frontier_module
import hermes_lcm.maintenance as maintenance_module
import hermes_lcm.tools as tools_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.maintenance import _source_inventory, apply_dag_maintenance

from .test_dag_maintenance import _fixture as maintenance_fixture


def _engine(tmp_path, **overrides) -> LCMEngine:
    values = {
        "database_path": str(tmp_path / "final-review.db"),
        "fresh_tail_count": 1,
        "async_background_compaction_enabled": True,
        "async_background_compaction_worker_enabled": False,
    }
    values.update(overrides)
    engine = LCMEngine(
        config=LCMConfig(**values), hermes_home=str(tmp_path / "hermes-home")
    )
    engine.on_session_start(
        "current", conversation_id="conversation", platform="test", context_length=10_000
    )
    return engine


def test_foreground_identity_uses_incremental_bounded_sql_not_get_batch(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user", "content": "bounded identity one"},
        {"role": "assistant", "content": "bounded identity two"},
    ]
    engine.ingest(messages)
    source_ids = engine._get_store_ids_for_messages(messages)
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        engine._store,
        "get_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreground identity called unbounded get_batch")
        ),
    )
    try:
        identity = engine._foreground_source_identity_for_messages(messages, source_ids)
        assert len(identity) == 64
        engine._store._conn.execute(
            "DELETE FROM messages WHERE store_id=?", (source_ids[-1],)
        )
        engine._store._conn.commit()
        assert engine._foreground_source_identity_for_messages(messages, source_ids) == ""
        identity_selects = [
            statement for statement in statements
            if "FROM messages" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert identity_selects
        assert any("SUBSTR" in statement.upper() for statement in identity_selects)
        assert all("SELECT STORE_ID, SESSION_ID, SOURCE" not in statement.upper()
                   for statement in identity_selects)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def _ready_batch(engine: LCMEngine) -> int:
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
    assert batch_id > 0
    return batch_id


def test_valid_oversized_ready_row_is_rejected_directly_before_payload_decode(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    batch_id = _ready_batch(engine)
    raw_summary = json.dumps({"summary_text": "ready-canary-" + ("x" * 8_000)})
    engine._frontier._conn.execute(
        """UPDATE lcm_prepared_batches
           SET state='ready', payload_version=2, summary_payload=? WHERE batch_id=?""",
        (raw_summary, batch_id),
    )
    engine._frontier._conn.commit()
    monkeypatch.setattr(frontier_module, "_BATCH_LOAD_MAX_SUMMARY_BYTES", 1_024)
    original_loads = frontier_module.json.loads

    def guarded_loads(raw, *args, **kwargs):
        if raw == raw_summary:
            raise AssertionError("oversized ready summary reached JSON decode")
        return original_loads(raw, *args, **kwargs)

    monkeypatch.setattr(frontier_module.json, "loads", guarded_loads)
    try:
        assert engine._frontier.get_ready_batch("conversation") is None
        state, reason = engine._frontier._conn.execute(
            "SELECT state, failure_reason FROM lcm_prepared_batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        assert state == "rejected"
        assert "encoded-size" in reason
    finally:
        engine.shutdown()


def test_pending_numeric_affinity_text_is_rejected_without_python_int(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    batch_id = _ready_batch(engine)
    numeric_canary = "not-a-number-" + ("9" * 1_000_000)
    engine._frontier._conn.execute(
        "UPDATE lcm_prepared_batches SET base_generation=? WHERE batch_id=?",
        (numeric_canary, batch_id),
    )
    engine._frontier._conn.commit()
    original_row_to_batch = engine._frontier._row_to_batch

    def guarded_row_to_batch(row):
        assert numeric_canary not in row
        return original_row_to_batch(row)

    monkeypatch.setattr(engine._frontier, "_row_to_batch", guarded_row_to_batch)
    try:
        assert engine._frontier.list_pending_batches("conversation") == []
        assert engine._frontier._conn.execute(
            "SELECT state FROM lcm_prepared_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()[0] == "rejected"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["escape-bytes", "deadline"])
def test_source_identity_multirow_escape_amplification_is_page_bounded(
    tmp_path, monkeypatch, failure
):
    engine = _engine(tmp_path)
    source_ids = [
        engine._store.append(
            "current", {"role": "user", "content": ("\\\"" * 8_000) + str(index)}
        )
        for index in range(3)
    ]
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    seen: list[int] = []
    budget = {
        "rows": 0,
        "bytes": 0,
        "max_rows": 10,
        "max_bytes": 30_000,
        "deadline_at": 10.0,
    }
    if failure == "deadline":
        ticks = iter((0.0, 0.0, 0.0, 11.0, 11.0, 11.0))
        monkeypatch.setattr(frontier_module.time, "monotonic", lambda: next(ticks, 11.0))
    else:
        monkeypatch.setattr(frontier_module.time, "monotonic", lambda: 0.0)
    try:
        with pytest.raises(RuntimeError, match="byte bound|deadline"):
            frontier_module.compute_source_identity_hash(
                engine._store._conn,
                "current",
                source_ids,
                read_budget=budget,
                row_validator=lambda source_id, _row: seen.append(source_id),
            )
        selects = [
            statement for statement in statements
            if "FROM messages" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert selects
        assert all(
            "JSON_QUOTE" not in statement.upper()
            or "LIMIT 1" in statement.upper()
            for statement in selects
        )
        assert any("LIMIT 1" in statement.upper() for statement in selects)
        assert len(seen) < len(source_ids)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_maintenance_root_read_is_explicit_and_does_not_materialize_summary(tmp_path):
    path, parent, _ = maintenance_fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "UPDATE summary_nodes SET summary=? WHERE node_id=?",
        ("root-summary-canary-" + ("x" * 1_000_000), parent),
    )
    conn.commit()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        nodes, messages = _source_inventory(conn, parent)
        assert parent in nodes and messages == {1, 2}
        root_reads = [
            statement for statement in statements
            if f"NODE_ID={parent}" in statement.replace(" ", "").upper()
            and "FROM SUMMARY_NODES" in statement.upper()
        ]
        assert root_reads
        assert all("SELECT *" not in statement.upper() for statement in root_reads)
        assert all("SUMMARY" not in statement.upper().split("FROM", 1)[0]
                   for statement in root_reads[:1])
    finally:
        conn.close()


def test_copy_subtree_node_payload_shares_message_budget_and_rolls_back(
    tmp_path, monkeypatch
):
    path, parent, _ = maintenance_fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE summary_nodes SET summary=?, expand_hint=? WHERE node_id=?",
        ("summary-canary-" + ("x" * 5_000), "hint-canary-" + ("y" * 5_000), parent),
    )
    before = conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
    conn.commit()
    conn.close()
    monkeypatch.setattr(maintenance_module, "_MAINTENANCE_COPY_MAX_BYTES", 4_096)
    with pytest.raises(ValueError, match="copy.*byte.*budget"):
        apply_dag_maintenance(
            path,
            operation="copy-subtree",
            conversation_id="source-conv",
            node_id=parent,
            target_session_id="target",
            target_conversation_id="target-conv",
            confirmation="APPLY copy-subtree",
        )
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == before
    finally:
        conn.close()


def test_promoted_frontier_rejects_declared_range_overlap_without_dropping_node(
    tmp_path
):
    engine = _engine(tmp_path)
    first = engine._store.append("current", {"role": "user", "content": "first"})
    middle = engine._store.append("current", {"role": "user", "content": "middle"})
    last = engine._store.append("current", {"role": "user", "content": "last"})
    sparse = engine._dag.add_node(SummaryNode(
        session_id="current", depth=0, summary="sparse canonical", token_count=1,
        source_token_count=1, source_ids=[first, last], source_type="messages", created_at=1,
    ))
    candidate = engine._dag.add_node(SummaryNode(
        session_id="current", depth=0, summary="candidate", token_count=1,
        source_token_count=1, source_ids=[middle], source_type="messages", created_at=2,
    ))
    engine._frontier.ensure_frontier("conversation", "current", source_end_store_id=last)
    engine._frontier.set_frontier_items("conversation", 1, [{
        "kind": "node", "ref_id": sparse, "source_start": first, "source_end": last,
    }])
    try:
        with pytest.raises(RuntimeError, match="range overlap"):
            engine._build_promoted_frontier_items_no_commit(
                engine._frontier._conn,
                conversation_id="conversation",
                session_id="current",
                node_id=candidate,
                covered_source_ids=[middle],
                frontier_end_store_id=last,
                base_generation=1,
                read_budget=engine._new_locked_publication_read_budget(),
            )
        items = engine._frontier.get_frontier_items("conversation", 1)
        assert [int(item["ref_id"]) for item in items] == [sparse]
    finally:
        engine.shutdown()


def test_full_sweep_frontier_rejects_overlapping_declared_ranges_losslessly():
    with pytest.raises(RuntimeError, match="range overlap"):
        engine_module.FullSweepMixin._full_sweep_frontier_items(
            [],
            [{"store_id": 5}],
            [{"kind": "node", "ref_id": 7, "source_start": 1, "source_end": 10}],
        )


def test_rollover_checks_active_encoded_size_before_normalization(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    message = {"role": "user", "content": "active-canary-" + ("x" * 20_000)}
    engine._store.append(
        "current",
        {"role": "user", "content": "small stored row"},
        conversation_id="conversation",
    )
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 1_024)
    monkeypatch.setattr(
        engine,
        "_session_end_prefix_compare_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normalization reached before active-value size check")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session("current", "next", previous_messages=[message])
        assert engine.current_session_id == "current"
    finally:
        engine.shutdown()


def test_rollover_externalized_placeholder_stats_before_file_read(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    payload_dir = externalize_module.get_large_output_storage_dir(
        engine._config, hermes_home=engine._hermes_home, create=True
    )
    payload_dir.mkdir(parents=True, exist_ok=True)
    ref = "rollover-large.json"
    path = payload_dir / ref
    path.write_text(
        json.dumps({"session_id": "current", "content": "x" * 50_000}),
        encoding="utf-8",
    )
    placeholder = f"[Externalized tool output: tool_call_id=t; ref={ref}]"
    engine._store.append(
        "current",
        {"role": "tool", "content": placeholder},
        conversation_id="conversation",
    )
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 4_096)
    original_loads = externalize_module.json.loads

    def guarded_loads(raw, *args, **kwargs):
        if isinstance(raw, str) and len(raw) > 4_096:
            raise AssertionError("oversized externalized file was parsed")
        return original_loads(raw, *args, **kwargs)

    monkeypatch.setattr(externalize_module.json, "loads", guarded_loads)
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session(
                "current", "next", previous_messages=[{"role": "tool", "content": placeholder}]
            )
        assert engine.current_session_id == "current"
    finally:
        engine.shutdown()


def test_load_session_sql_substrings_valid_oversized_content_before_python(
    tmp_path
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "load-canary-" + ("x" * 1_000_000)}
    )
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        response_text = engine.handle_tool_call(
            "lcm_load_session",
            {"session_id": "current", "limit": 1, "max_content_chars": 128},
        )
        assert len(response_text.encode("utf-8")) <= tools_module._LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES
        response = json.loads(response_text)
        assert response["messages"][0]["store_id"] == store_id
        assert response["messages"][0]["content_truncated"] is True
        selects = [
            statement for statement in statements
            if "FROM messages" in statement
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert selects
        assert any("SUBSTR" in statement.upper() for statement in selects)
        assert any("LENGTH(CAST(CONTENT AS BLOB))" in statement.upper()
                   for statement in selects)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_load_session_rejects_numeric_affinity_text_before_materialization(tmp_path):
    engine = _engine(tmp_path)
    store_id = engine._store.append("current", {"role": "user", "content": "safe"})
    numeric_canary = "numeric-canary-" + ("9" * 1_000_000)
    engine._store._conn.execute(
        "UPDATE messages SET token_estimate=? WHERE store_id=?", (numeric_canary, store_id)
    )
    engine._store._conn.commit()
    try:
        response = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current", "limit": 1}
        ))
        assert "error" in response
        assert "invalid stored message scalar" in response["error"]
        assert numeric_canary not in json.dumps(response)
    finally:
        engine.shutdown()
