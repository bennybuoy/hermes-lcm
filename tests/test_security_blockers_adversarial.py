"""Adversarial regressions for the final publication/security review blockers."""

from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.config as config_module
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


def test_escaped_quoted_password_assignments_are_redacted_in_linear_time(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    try:
        secrets = ("escaped-assignment-secret", "escaped-json-secret")
        value = (
            'prefix password=\\"escaped-assignment-secret and '
            '\\"password\\":\\"escaped-json-secret suffix'
        )
        started = time.monotonic()
        redacted = redact_sensitive_text((value + "\n") * 2_000, engine._config)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert all(secret not in redacted for secret in secrets)
        assert redacted.count("LCM sensitive redaction") >= 2
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
