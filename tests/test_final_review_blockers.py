"""Non-vacuous regressions for the final publication/security blockers."""

from __future__ import annotations

import json
import sqlite3
import tempfile

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.externalize as externalize_module
import hermes_lcm.frontier as frontier_module
import hermes_lcm.maintenance as maintenance_module
import hermes_lcm.reconcile as reconcile_module
import hermes_lcm.tools as tools_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import externalize_ingest_payload
from hermes_lcm.maintenance import (
    _source_inventory,
    apply_dag_maintenance,
    plan_dag_maintenance,
)

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


def _persisted_output_marker(
    tmp_path, monkeypatch, content: str, name: str = "result.txt"
) -> str:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    result_dir = tmp_path / "hermes-results"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / name
    result_path.write_text(content, encoding="utf-8")
    preview = content[:30]
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({len(content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {result_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )


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


def test_source_mapper_bounds_rows_before_real_full_sweep_mapper_invocation(
    tmp_path, monkeypatch
):
    engine = _engine(
        tmp_path,
        full_sweep_compaction_enabled=True,
        full_sweep_max_passes=8,
        full_sweep_deadline_seconds=30.0,
        fresh_tail_count=1,
        leaf_chunk_tokens=16,
        condensation_fanin=2,
        condensation_min_fanin=2,
        summary_prefix_target_tokens=1,
        context_threshold=0.10,
    )
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": f"bounded-full-sweep-{index} " + ("payload " * 20)}
        for index in range(8)
    ]
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        engine._store,
        "get_session_messages_after",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("real full sweep called the unbounded mapper reader")
        ),
    )
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda chunk, **_kwargs: (list(chunk), 100, "leaf", 1, 1),
    )
    monkeypatch.setattr(
        engine_module,
        "summarize_with_escalation",
        lambda **_kwargs: ("parent", 1),
    )
    try:
        engine.compress(messages, current_tokens=9_000)
        assert engine._last_full_sweep_status["publication_count"] == 1
        mapper_selects = [
            statement for statement in statements
            if "FROM messages" in statement
            and "store_id >" in statement
            and statement.lstrip().upper().startswith(("SELECT", "WITH"))
        ]
        assert mapper_selects
        assert any("SUBSTR" in statement.upper() for statement in mapper_selects)
        assert any("LENGTH(CAST" in statement.upper() for statement in mapper_selects)
        assert all("LIMIT" in statement.upper() for statement in mapper_selects)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_source_mapper_bounds_rows_on_real_foreground_compaction_path(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=16)
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"bounded-foreground-{index} " + ("payload " * 20),
        }
        for index in range(6)
    ]
    monkeypatch.setattr(
        engine._store,
        "get_session_messages_after",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("real foreground compaction called the unbounded mapper reader")
        ),
    )
    monkeypatch.setattr(
        engine_module,
        "summarize_with_escalation",
        lambda **_kwargs: ("foreground bounded summary", 1),
    )
    try:
        engine.compress(
            messages,
            current_tokens=engine.threshold_tokens + 1,
            force=True,
        )
        assert engine._dag.get_session_node_count("current") >= 1
    finally:
        engine.shutdown()


def test_source_mapper_rejects_oversized_row_before_replay_identity(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    message = {"role": "user", "content": "mapper-canary-" + ("x" * 20_000)}
    store_id = engine._store.append(
        "current", {"role": "user", "content": "small"},
        conversation_id="conversation",
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?",
        (message["content"], store_id),
    )
    engine._store._conn.commit()
    monkeypatch.setattr(
        engine,
        "_new_locked_publication_read_budget",
        lambda: {
            "rows": 0,
            "bytes": 0,
            "max_rows": 10,
            "max_bytes": 1_024,
            "deadline_at": 1e30,
        },
    )
    monkeypatch.setattr(
        engine._store,
        "get_session_messages_after",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized mapper used full-row session loading")
        ),
    )
    original_identity = engine._message_replay_identity

    def guarded_identity(candidate, *args, **kwargs):
        if kwargs.get("stored_row") and "mapper-canary" in str(candidate.get("content")):
            raise AssertionError("oversized stored row reached replay identity")
        return original_identity(candidate, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", guarded_identity)
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine._get_store_ids_for_messages([{"role": "user", "content": "small"}])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["nesting", "deadline"])
def test_source_mapper_preflights_shared_nested_and_deadline_budget(
    tmp_path, monkeypatch, failure
):
    engine = _engine(tmp_path)
    nested: object = "leaf"
    for _ in range(reconcile_module._RECONCILIATION_MAX_NESTED_DEPTH + 2):
        nested = {"child": nested}
    message = {"role": "user", "content": nested}
    if failure == "deadline":
        message = {"role": "user", "content": "deadline-canary"}
        monkeypatch.setattr(
            engine,
            "_new_locked_publication_read_budget",
            lambda: {
                "rows": 0,
                "bytes": 0,
                "max_rows": 10,
                "max_bytes": 10_000,
                "deadline_at": 0.0,
            },
        )
        monkeypatch.setattr(reconcile_module.time, "monotonic", lambda: 1.0)
    original_identity = engine._message_replay_identity

    def guarded_identity(candidate, *args, **kwargs):
        if candidate is message:
            raise AssertionError("unpreflighted active row reached replay identity")
        return original_identity(candidate, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", guarded_identity)
    try:
        with pytest.raises(RuntimeError, match="nested-depth|deadline"):
            engine._get_store_ids_for_messages([message])
    finally:
        engine.shutdown()


def test_source_mapper_computes_each_active_identity_once_with_shared_budget(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user", "content": f"generated-placeholder-{index}"}
        for index in range(40)
    ]
    engine._generated_ignored_active_replay_placeholder_message_ids = {
        id(message) for message in messages
    }
    calls: dict[int, int] = {}
    original = engine._message_replay_identity

    def counted(message, *args, **kwargs):
        if message in messages:
            assert kwargs.get("read_budget") is not None
            calls[id(message)] = calls.get(id(message), 0) + 1
        return original(message, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", counted)
    try:
        engine._get_store_ids_for_messages(messages)
        assert calls == {id(message): 1 for message in messages}
    finally:
        engine.shutdown()


def test_restart_reconciliation_computes_each_identity_once_under_one_budget(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    messages = [
        {"role": "system", "content": "bounded restart"},
        {"role": "user", "content": "restart question"},
        {"role": "assistant", "content": "restart answer"},
    ]
    engine._store._append_protected_batch(
        "current", messages, [1, 1, 1], conversation_id="conversation"
    )
    active_ids = {id(message) for message in messages}
    active_calls: dict[int, int] = {}
    stored_calls: dict[int, int] = {}
    budget_ids: set[int] = set()
    original = engine._message_replay_identity

    def counted(message, *args, **kwargs):
        budget = kwargs.get("read_budget")
        assert budget is not None
        budget_ids.add(id(budget))
        if id(message) in active_ids:
            active_calls[id(message)] = active_calls.get(id(message), 0) + 1
        elif kwargs.get("stored_row"):
            store_id = int(message["store_id"])
            stored_calls[store_id] = stored_calls.get(store_id, 0) + 1
        return original(message, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", counted)
    try:
        assert engine._reconcile_ingest_cursor_from_store(messages) == len(messages)
        assert active_calls == {id(message): 1 for message in messages}
        assert stored_calls == {1: 1, 2: 1, 3: 1}
        assert len(budget_ids) == 1
    finally:
        engine.shutdown()


def test_restart_reconciliation_sql_bounds_tail_before_python_materialization(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "small"},
        conversation_id="conversation",
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?",
        ("restart-tail-canary-" + ("x" * 1_000_000), store_id),
    )
    engine._store._conn.commit()
    monkeypatch.setattr(
        engine,
        "_new_locked_publication_read_budget",
        lambda: {
            "rows": 0, "bytes": 0, "files": 0,
            "max_rows": 8, "max_bytes": 512, "max_files": 8,
            "deadline_at": 1e30,
        },
    )
    monkeypatch.setattr(
        engine._store,
        "get_session_tail",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restart reconciliation used unbounded tail loading")
        ),
    )
    monkeypatch.setattr(
        engine._store,
        "get_session_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restart reconciliation used unbounded head loading")
        ),
    )
    original_identity = engine._message_replay_identity

    def guarded_identity(row, *args, **kwargs):
        if kwargs.get("stored_row") and len(str(row.get("content") or "")) > 128:
            raise AssertionError("oversized restart row reached Python identity mapping")
        return original_identity(row, *args, **kwargs)

    monkeypatch.setattr(engine, "_message_replay_identity", guarded_identity)
    statements: list[str] = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine._reconcile_ingest_cursor_from_store(
                [{"role": "user", "content": "incoming"}]
            )
        selects = [s.upper() for s in statements if "FROM MESSAGES" in s.upper()]
        assert any("LENGTH(CAST" in s for s in selects)
        assert all("SELECT STORE_ID, SESSION_ID, SOURCE, ROLE, CONTENT" not in s for s in selects)
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()


def test_reconciliation_ingest_placeholder_restoration_uses_shared_budget(
    tmp_path
):
    engine = _engine(tmp_path)
    externalized = externalize_ingest_payload(
        "I" * 1_000_000,
        role="user",
        session_id="current",
        field_path="content",
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert externalized is not None
    budget = {
        "rows": 0, "bytes": 0, "files": 0,
        "max_rows": 8, "max_bytes": 128, "max_files": 8,
        "deadline_at": 1e30,
    }
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine._message_replay_identity(
                {
                    "session_id": "current",
                    "role": "user",
                    "content": externalized["placeholder"],
                },
                stored_row=True,
                read_budget=budget,
            )
        assert budget["bytes"] <= budget["max_bytes"]
    finally:
        engine.shutdown()


def test_restart_reconciliation_large_no_match_checks_deadline_after_identity_capture(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    size = 1_200
    messages = [
        {"role": "user", "content": f"incoming-no-match-{index}"}
        for index in range(size)
    ]
    active_identities = {
        id(message): ("user", str(message["content"]), "", "")
        for message in messages
    }
    stored_tail = [
        ("assistant", f"durable-no-match-{index}", "", "")
        for index in range(size)
    ]
    ticks = 0
    # Identity capture is already complete. Allow the three linear setup
    # passes, then expire during suffix-overlap matching. The old reverse
    # cursor loop never consulted the deadline here and scanned all prefixes.
    expire_after = (3 * size) + 25

    def matching_clock():
        nonlocal ticks
        ticks += 1
        return 2.0 if ticks > expire_after else 0.0

    monkeypatch.setattr(reconcile_module.time, "monotonic", matching_clock)
    monkeypatch.setattr(
        engine,
        "_message_replay_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("matching loop recaptured an active identity")
        ),
    )
    budget = {
        "rows": size,
        "bytes": size * 64,
        "files": 0,
        "max_rows": size * 4,
        "max_bytes": size * 1024,
        "max_files": size * 4,
        "deadline_at": 1.0,
    }
    try:
        with pytest.raises(RuntimeError, match="reconciliation deadline"):
            engine._find_reconciled_cursor_for_store_tail(
                messages,
                stored_tail,
                allow_empty_prefix=False,
                session_count=size,
                raw_session_count=size,
                read_budget=budget,
                active_identities=active_identities,
            )
        assert ticks > expire_after
    finally:
        engine.shutdown()


def test_source_mapper_persisted_output_recovery_uses_shared_byte_budget(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, large_output_externalization_enabled=True)
    marker = _persisted_output_marker(tmp_path, monkeypatch, "P" * 8_000)
    message = {"role": "tool", "tool_call_id": "bounded", "content": marker}
    monkeypatch.setattr(
        engine,
        "_new_locked_publication_read_budget",
        lambda: {
            "rows": 0,
            "bytes": 0,
            "files": 0,
            "max_rows": 10,
            "max_bytes": 2_048,
            "max_files": 10,
            "deadline_at": 1e30,
        },
    )
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine._get_store_ids_for_messages([message])
    finally:
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


def test_maintenance_lineage_budget_charges_and_memoizes_message_validation(
    tmp_path
):
    path, parent, _children = maintenance_fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    budget = maintenance_module._new_maintenance_lineage_budget()
    try:
        first = _source_inventory(conn, parent, lineage_budget=budget)
        first_usage = (budget["rows"], budget["bytes"])
        second = _source_inventory(conn, parent, lineage_budget=budget)
        assert second == first
        assert (budget["rows"], budget["bytes"]) == first_usage
        assert int(budget["message_rows"]) == 2
        message_payload_reads = [
            statement for statement in statements
            if "FROM messages" in statement
            and "SUBSTR" in statement.upper()
            and statement.lstrip().upper().startswith("SELECT")
        ]
        assert len(message_payload_reads) == 1
    finally:
        conn.close()


def test_maintenance_plan_threads_one_lineage_budget_through_all_closures(
    tmp_path, monkeypatch
):
    path, parent, _children = maintenance_fixture(tmp_path)
    seen_budgets: list[object] = []
    original = maintenance_module._source_inventory

    def recorded(*args, **kwargs):
        budget = kwargs.get("lineage_budget")
        assert budget is not None
        seen_budgets.append(budget)
        return original(*args, **kwargs)

    monkeypatch.setattr(maintenance_module, "_source_inventory", recorded)
    plan = plan_dag_maintenance(
        path,
        operation="rewrite-subtree",
        conversation_id="source-conv",
        node_id=parent,
        rewrites={parent: "bounded rewrite"},
    )
    assert plan["dry_run"] is True
    assert len(seen_budgets) >= 2
    assert len({id(budget) for budget in seen_budgets}) == 1


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


def _many_node_rewrite_fixture(tmp_path) -> tuple[object, int, dict[int, str]]:
    path, parent, _ = maintenance_fixture(tmp_path)
    conn = sqlite3.connect(path)
    try:
        previous = parent
        node_ids = [parent]
        for depth in range(2, 19):
            cur = conn.execute(
                """INSERT INTO summary_nodes
                   (session_id, depth, summary, token_count, source_token_count,
                    source_ids, source_type, created_at, expand_hint)
                   VALUES ('source', ?, ?, 10, 10, ?, 'nodes', ?, ?)""",
                (
                    depth,
                    f"many-node-summary-{depth}-" + ("s" * 700),
                    json.dumps([previous]),
                    float(depth),
                    f"many-node-hint-{depth}-" + ("h" * 700),
                ),
            )
            previous = int(cur.lastrowid)
            node_ids.append(previous)
        conn.execute(
            """UPDATE summary_nodes SET summary=?, expand_hint=? WHERE node_id=?""",
            ("many-node-root-" + ("r" * 700), "many-node-root-hint-" + ("q" * 700), parent),
        )
        conn.execute(
            """UPDATE lcm_frontier_items SET ref_id=?
               WHERE conversation_id='source-conv' AND generation=1 AND ordinal=0""",
            (previous,),
        )
        conn.commit()
        return path, previous, {
            node_id: f"replacement-{index}" for index, node_id in enumerate(node_ids)
        }
    finally:
        conn.close()


def test_rewrite_plan_shares_payload_budget_across_many_nodes(tmp_path, monkeypatch):
    path, root, rewrites = _many_node_rewrite_fixture(tmp_path)
    monkeypatch.setattr(maintenance_module, "_MAINTENANCE_COPY_MAX_BYTES", 10_000)
    with pytest.raises(ValueError, match="shared byte budget"):
        plan_dag_maintenance(
            path,
            operation="rewrite-subtree",
            conversation_id="source-conv",
            node_id=root,
            rewrites=rewrites,
        )


def test_rewrite_apply_shares_payload_budget_and_aborts_atomically(
    tmp_path, monkeypatch
):
    path, root, rewrites = _many_node_rewrite_fixture(tmp_path)
    conn = sqlite3.connect(path)
    before = {
        "nodes": conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
        "generation": conn.execute(
            """SELECT generation FROM lcm_active_frontiers
               WHERE conversation_id='source-conv'"""
        ).fetchone()[0],
        "items": conn.execute("SELECT COUNT(*) FROM lcm_frontier_items").fetchone()[0],
    }
    conn.close()

    def lower_budget(phase):
        if phase == "before_begin":
            monkeypatch.setattr(
                maintenance_module, "_MAINTENANCE_COPY_MAX_BYTES", 10_000
            )

    with pytest.raises(ValueError, match="shared byte budget"):
        apply_dag_maintenance(
            path,
            operation="rewrite-subtree",
            conversation_id="source-conv",
            node_id=root,
            rewrites=rewrites,
            confirmation="APPLY rewrite-subtree",
            snapshot_hook=lower_budget,
        )
    conn = sqlite3.connect(path)
    try:
        after = {
            "nodes": conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            "generation": conn.execute(
                """SELECT generation FROM lcm_active_frontiers
                   WHERE conversation_id='source-conv'"""
            ).fetchone()[0],
            "items": conn.execute("SELECT COUNT(*) FROM lcm_frontier_items").fetchone()[0],
        }
        assert after == before
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


def test_rollover_empty_store_preflights_unmatched_host_suffix(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    huge = {"role": "user", "content": "empty-store-suffix-" + ("x" * 20_000)}
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 1_024)
    monkeypatch.setattr(
        engine_module,
        "protect_messages_for_ingest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unbounded empty-store suffix reached normalization")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session("current", "next", previous_messages=[huge])
        assert engine.current_session_id == "current"
        assert engine._store.get_session_count("current") == 0
    finally:
        engine.shutdown()


def test_rollover_matched_prefix_preflights_huge_unmatched_suffix_and_rolls_back(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    prefix = {"role": "user", "content": "durable-prefix"}
    engine._store.append("current", prefix, conversation_id="conversation")
    huge = {"role": "assistant", "content": "matched-prefix-suffix-" + ("y" * 20_000)}
    before_frontier = engine._frontier.get_active_frontier("conversation")
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 2_048)
    monkeypatch.setattr(
        engine_module,
        "protect_messages_for_ingest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unbounded matched-prefix suffix reached normalization")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session(
                "current", "next", previous_messages=[dict(prefix), huge]
            )
        assert engine.current_session_id == "current"
        assert engine._store.get_session_count("current") == 1
        assert engine._frontier.get_active_frontier("conversation") == before_frontier
    finally:
        engine.shutdown()


def test_rollover_matched_ingest_placeholder_uses_budget_before_expansion(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    externalized = externalize_ingest_payload(
        "R" * 1_000_000,
        role="user",
        session_id="current",
        field_path="content",
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert externalized is not None
    placeholder = externalized["placeholder"]
    engine._store.append(
        "current", {"role": "user", "content": placeholder},
        conversation_id="conversation",
    )
    resolver_budget = {
        "rows": 0, "bytes": 0, "files": 0,
        "max_rows": 8, "max_bytes": 128, "max_files": 8,
        "deadline_at": 1e30,
    }
    with pytest.raises(RuntimeError, match="byte bound"):
        engine._session_end_prefix_compare_value(
            placeholder,
            session_id="current",
            read_budget=resolver_budget,
        )
    assert resolver_budget["bytes"] <= resolver_budget["max_bytes"]
    monkeypatch.setattr(engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 128)
    before = engine._store.get_session_messages("current")
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session(
                "current", "next", previous_messages=[{"role": "user", "content": placeholder}]
            )
        assert engine.current_session_id == "current"
        assert engine._store.get_session_messages("current") == before
        assert engine._store.get_session_count("next") == 0
    finally:
        engine.shutdown()


def test_rollover_unmatched_persisted_output_expansion_aborts_losslessly(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    marker = _persisted_output_marker(
        tmp_path, monkeypatch, "rollover-secret-" + ("R" * 12_000), "rollover.txt"
    )
    message = {"role": "tool", "tool_call_id": "rollover", "content": marker}
    before_frontier = engine._frontier.get_active_frontier("conversation")
    monkeypatch.setattr(
        engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 2_048
    )
    try:
        with pytest.raises(RuntimeError, match="byte bound"):
            engine.rollover_session(
                "current", "next", previous_messages=[message]
            )
        assert engine.current_session_id == "current"
        assert engine._store.get_session_count("current") == 0
        assert engine._frontier.get_active_frontier("conversation") == before_frontier
    finally:
        engine.shutdown()


@pytest.mark.parametrize("payload_state", ["oversized", "missing"])
def test_rollover_unmatched_externalized_marker_resolves_with_shared_budget(
    tmp_path, monkeypatch, payload_state
):
    engine = _engine(tmp_path)
    ref = f"unmatched-{payload_state}.json"
    if payload_state == "oversized":
        payload_dir = externalize_module.get_large_output_storage_dir(
            engine._config, hermes_home=engine._hermes_home, create=True
        )
        (payload_dir / ref).write_text(
            json.dumps({
                "kind": "tool_result",
                "session_id": "current",
                "role": "tool",
                "tool_call_id": "externalized-rollover",
                "content": "E" * 12_000,
            }),
            encoding="utf-8",
        )
    placeholder = (
        "[Externalized tool output: tool_call_id=externalized-rollover; "
        f"chars=12000; ref={ref}]"
    )
    message = {
        "role": "tool",
        "tool_call_id": "externalized-rollover",
        "content": placeholder,
    }
    before_frontier = engine._frontier.get_active_frontier("conversation")
    monkeypatch.setattr(
        engine_module, "_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES", 2_048
    )
    try:
        expected = "byte bound" if payload_state == "oversized" else "could not be resolved"
        with pytest.raises(RuntimeError, match=expected):
            engine.rollover_session(
                "current", "next", previous_messages=[message]
            )
        assert engine.current_session_id == "current"
        assert engine._store.get_session_count("current") == 0
        assert engine._frontier.get_active_frontier("conversation") == before_frontier
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


def test_load_session_redacts_secret_spanning_requested_content_boundary(tmp_path):
    engine = _engine(tmp_path)
    max_chars = 128
    secret = "github_pat_" + ("A" * 40)
    content = ("x" * (max_chars - 10)) + " " + secret + " visible-tail"
    engine._store.append("current", {"role": "user", "content": content})
    try:
        response = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {
                "session_id": "current",
                "limit": 1,
                "max_content_chars": max_chars,
            },
        ))
        returned = response["messages"][0]["content"]
        assert secret not in returned
        assert "github_pat" not in returned
        assert "LCM sens" in returned
        assert len(returned) <= max_chars
    finally:
        engine.shutdown()


def test_load_session_redaction_shortening_preserves_source_truncation_metadata(
    tmp_path
):
    engine = _engine(tmp_path)
    secret = "sk-proj-" + ("S" * 20_000)
    content = "prefix " + secret + " suffix " + ("x" * 10_000)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "legacy placeholder"}
    )
    engine._store._conn.execute(
        "UPDATE messages SET content=? WHERE store_id=?", (content, store_id)
    )
    engine._store._conn.commit()
    try:
        response = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {"session_id": "current", "limit": 1, "max_content_chars": 128},
        ))
        item = response["messages"][0]
        assert secret not in item["content"]
        assert "sk-proj-" not in item["content"]
        assert item["content_redacted"] is True
        assert item["content_source_truncated"] is True
        assert item["content_truncated"] is True
        assert item["content_returned_chars"] == len(item["content"])
        assert item["content_returned_chars"] <= item["content_chars"] == len(content)
        assert item["serialized_truncated"] is True

        # A redacted output length is not a raw-source cursor.  The load API
        # must disable mapped continuation and advertise a safe restart boundary;
        # following that boundary through lcm_expand must not expose a suffix of
        # the credential that no longer matches the redaction pattern.
        assert item["next_content_offset"] is None
        assert item["content_continuation_disabled"] is True
        assert item["safe_expand_content_offset"] == 0
        expanded = json.loads(engine.handle_tool_call(
            "lcm_expand",
            {
                "store_id": item["store_id"],
                "content_offset": item["safe_expand_content_offset"],
                "max_tokens": 32,
            },
        ))
        expanded_text = json.dumps(expanded)
        assert secret not in expanded_text
        assert ("S" * 100) not in expanded_text
        assert expanded["has_more"] is False
    finally:
        engine.shutdown()


def test_load_session_non_sensitive_raw_cursor_is_lossless_and_deterministic(tmp_path):
    engine = _engine(tmp_path)
    content = "".join(str(index % 10) for index in range(1_000))
    store_id = engine._store.append(
        "current", {"role": "user", "content": content}
    )
    try:
        loaded = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {"session_id": "current", "limit": 1, "max_content_chars": 128},
        ))["messages"][0]
        assert loaded["content_redacted"] is False
        assert loaded["next_content_offset"] == len(loaded["content"]) == 128
        assert loaded.get("content_continuation_disabled") is not True

        first = json.loads(engine.handle_tool_call(
            "lcm_expand",
            {
                "store_id": store_id,
                "content_offset": loaded["next_content_offset"],
                "max_tokens": 65_536,
            },
        ))
        second = json.loads(engine.handle_tool_call(
            "lcm_expand",
            {
                "store_id": store_id,
                "content_offset": loaded["next_content_offset"],
                "max_tokens": 65_536,
            },
        ))
        assert first["content"] == second["content"]
        assert loaded["content"] + first["content"] == content
        assert first["has_more"] is False
    finally:
        engine.shutdown()


def test_load_session_count_and_page_share_snapshot_during_concurrent_insert(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._store.append("current", {"role": "user", "content": "snapshot-one"})
    original = engine._store._load_session_page_locked
    raced = {"value": False}

    def insert_between_count_and_page(*args, **kwargs):
        if not raced["value"]:
            raced["value"] = True
            writer = sqlite3.connect(engine._store.db_path, timeout=2.0)
            try:
                writer.execute(
                    """INSERT INTO messages
                       (session_id, source, role, content, timestamp, token_estimate,
                        pinned, conversation_id)
                       VALUES ('current', 'test', 'user', 'snapshot-two', 0, 1, 0,
                               'conversation')"""
                )
                writer.commit()
            finally:
                writer.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        engine._store, "_load_session_page_locked", insert_between_count_and_page
    )
    try:
        response = json.loads(engine.handle_tool_call(
            "lcm_load_session", {"session_id": "current", "limit": 10}
        ))
        assert raced["value"] is True
        assert response["total_messages"] == response["returned_messages"] == 1
        assert response["messages"][0]["content"] == "snapshot-one"
        assert response["has_more"] is False
        assert response["next_cursor"] is None
    finally:
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


def test_load_session_equal_length_reassignment_cannot_cross_filtered_payload_read(
    tmp_path
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "current", {"role": "user", "content": "current-secret"}
    )
    assert len("current") == len("foreign")
    assert len("current-secret") == len("foreign-secret")
    raced = {"value": False}

    def race_unfiltered_payload(statement):
        normalized = " ".join(statement.upper().split())
        if (
            not raced["value"]
            and "FROM MESSAGES WHERE" in normalized
            and "SUBSTR" in normalized
        ):
            raced["value"] = True
            writer = sqlite3.connect(engine._store.db_path, timeout=2.0)
            try:
                writer.execute(
                    "UPDATE messages SET session_id='foreign', content='foreign-secret' WHERE store_id=?",
                    (store_id,),
                )
                writer.commit()
            finally:
                writer.close()

    engine._store._conn.set_trace_callback(race_unfiltered_payload)
    try:
        response = json.loads(engine.handle_tool_call(
            "lcm_load_session",
            {"session_id": "current", "roles": ["user"], "limit": 1},
        ))
        encoded = json.dumps(response)
        assert raced["value"] is True
        assert "foreign-secret" not in encoded
        assert all(message["session_id"] == "current" for message in response["messages"])
    finally:
        engine._store._conn.set_trace_callback(None)
        engine.shutdown()
