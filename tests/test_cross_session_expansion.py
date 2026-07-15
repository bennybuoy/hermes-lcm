"""Issue #12: bounded, explicitly authorized cross-session DAG synthesis."""

from __future__ import annotations

import json
import threading

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.schemas import LCM_EXPAND_QUERY


def _engine(tmp_path, *, enabled=True, max_sessions=2, per_session=2):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "cross-session.db"),
            cross_session_expansion_enabled=enabled,
            cross_session_max_sessions=max_sessions,
            cross_session_max_summaries_per_session=per_session,
            cross_session_expansion_deadline_ms=5_000,
            expansion_timeout_ms=5_000,
        )
    )
    engine.on_session_start("current", conversation_id="conversation", platform="test")
    return engine


def _node(engine, session_id, summary, *, content="raw evidence"):
    store_id = engine._store.append(session_id, {"role": "user", "content": content})
    return engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=10,
            source_token_count=20,
            source_ids=[store_id],
            source_type="messages",
            created_at=0,
        )
    )


def _args(**overrides):
    values = {
        "prompt": "What happened?",
        "query": "archive",
        "cross_session": True,
        "max_tokens": 100,
        "context_max_tokens": 500,
    }
    values.update(overrides)
    return values


def _invoke(engine, args, *, session_ids=None):
    capability = engine.issue_cross_session_capability(
        session_ids or ["archive", "archive-a", "archive-b", "archive-c", "one", "two"]
    )
    return lcm_tools.lcm_expand_query(
        args,
        engine=engine,
        cross_session_capability=capability,
    )


def test_cross_session_mode_requires_profile_gate_and_trusted_host_capability(tmp_path):
    disabled = _engine(tmp_path / "disabled", enabled=False)
    enabled = _engine(tmp_path / "enabled")
    try:
        assert "disabled" in json.loads(_invoke(disabled, _args()))["error"]
        self_authorized = _args(
            authorize_cross_session=True,
            session_scope="all",
            session_ids=["archive"],
        )
        error = json.loads(
            lcm_tools.lcm_expand_query(self_authorized, engine=enabled)
        )["error"]
        assert "trusted host capability" in error
    finally:
        disabled.shutdown()
        enabled.shutdown()


def test_current_session_default_never_silently_widens(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _node(engine, "archive-a", "archive project result")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "should not run")
    try:
        result = json.loads(
            lcm_tools.lcm_expand_query(
                {"prompt": "What happened?", "query": "archive"},
                engine=engine,
            )
        )
        assert result["node_ids"] == []
        assert "current session" in result["answer"]
        assert "cross_session" not in result
    finally:
        engine.shutdown()


def test_sessions_are_ranked_before_bounded_expansion(tmp_path, monkeypatch):
    engine = _engine(tmp_path, max_sessions=2, per_session=1)
    node_b1 = engine._dag.get_node(_node(engine, "archive-b", "archive b first"))
    node_b2 = engine._dag.get_node(_node(engine, "archive-b", "archive b second"))
    node_a = engine._dag.get_node(_node(engine, "archive-a", "archive a"))
    node_c = engine._dag.get_node(_node(engine, "archive-c", "archive c"))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node_b1, node_b2, node_a, node_c])
    captured = {}

    def synthesize(**kwargs):
        captured["blocks"] = kwargs["context_blocks"]
        return "bounded archive answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(engine, _args()))
        assert result["answer"] == "bounded archive answer"
        assert result["contributing_session_ids"] == ["archive-b", "archive-a"]
        assert [match["node_id"] for match in result["matches"]] == [node_b1.node_id, node_a.node_id]
        assert result["skipped_buckets"] == [{"session_id": "archive-c", "reason": "max-sessions"}]
        assert {block["session_id"] for block in captured["blocks"]} == {"archive-a", "archive-b"}
    finally:
        engine.shutdown()


def test_session_allowlist_is_authorization_boundary(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    allowed = _node(engine, "allowed", "archive allowed")
    denied = _node(engine, "denied", "archive denied")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "allowed only")
    try:
        result = json.loads(
            _invoke(
                engine,
                _args(
                    node_ids=[allowed, denied],
                    query="",
                ),
                session_ids=["allowed"],
            )
        )
        assert result["node_ids"] == [allowed]
        assert result["contributing_session_ids"] == ["allowed"]
        assert all(match["session_id"] != "denied" for match in result["matches"])
    finally:
        engine.shutdown()


def test_context_and_answer_budgets_are_shared_not_per_bucket(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    first = engine._dag.get_node(_node(engine, "one", "archive one", content="x " * 100))
    second = engine._dag.get_node(_node(engine, "two", "archive two", content="y " * 100))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [first, second])
    captured = {}

    def synthesize(**kwargs):
        captured.update(kwargs)
        return "answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(
            _invoke(
                engine,
                _args(context_max_tokens=5, max_tokens=7),
            )
        )
        assert captured["max_tokens"] == 7
        assert result["context_tokens_used"] <= 5
        assert result["context_max_tokens"] == 5
        assert result["context_truncated"] is True
    finally:
        engine.shutdown()


def test_completed_bucket_survives_later_operation_deadline(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    first = engine._dag.get_node(_node(engine, "one", "archive one"))
    second = engine._dag.get_node(_node(engine, "two", "archive two"))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [first, second])
    clock = {"now": 0.0}
    monkeypatch.setattr(lcm_tools.time, "monotonic", lambda: clock["now"])
    original_collect = lcm_tools._collect_context_blocks_for_node

    def collect(*args, **kwargs):
        blocks = original_collect(*args, **kwargs)
        clock["now"] = 2.0
        return blocks

    monkeypatch.setattr(lcm_tools, "_collect_context_blocks_for_node", collect)
    try:
        result = json.loads(
            _invoke(engine, _args(deadline_ms=1_000))
        )
        assert result["degraded"] is True
        assert result["timed_out"] is True
        assert result["contributing_session_ids"] == ["one"]
        assert result["node_ids"] == [first.node_id]
        assert {item["session_id"] for item in result["skipped_buckets"]} == {"two"}
    finally:
        engine.shutdown()


def test_externalized_payloads_are_metadata_only_in_archive_mode(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    node = engine._dag.get_node(
        _node(
            engine,
            "archive",
            "archive payload",
            content="[GC'd externalized tool output: chars=999; ref=payload.json]",
        )
    )
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])
    captured = {}

    def synthesize(**kwargs):
        captured["serialized"] = json.dumps(kwargs["context_blocks"])
        return "metadata answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(engine, _args()))
        assert result["externalized_refs"] == "metadata-only"
        assert "payload.json" in captured["serialized"]
        assert "externalized_payload" not in captured["serialized"]
    finally:
        engine.shutdown()


def test_concurrent_reentry_is_rejected_deterministically(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    node = engine._dag.get_node(_node(engine, "archive", "archive result"))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])
    entered = threading.Event()
    release = threading.Event()

    def synthesize(**kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return "first answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    first_result = {}

    def run_first():
        first_result.update(json.loads(_invoke(engine, _args())))

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        second = json.loads(_invoke(engine, _args()))
        assert second["reentry_blocked"] is True
        assert "already active" in second["error"]
    finally:
        release.set()
        thread.join(timeout=2)
        engine.shutdown()
    assert first_result["answer"] == "first answer"


def test_tool_schema_does_not_expose_self_authorization_or_scope_controls():
    properties = LCM_EXPAND_QUERY["parameters"]["properties"]
    assert "authorize_cross_session" not in properties
    assert "session_scope" not in properties
    assert "session_ids" not in properties


def test_allowed_node_cannot_expand_raw_source_from_denied_session(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    denied_store_id = engine._store.append(
        "denied", {"role": "user", "content": "DENIED RAW SECRET"}
    )
    allowed_node_id = engine._dag.add_node(SummaryNode(
        session_id="allowed",
        depth=0,
        summary="archive allowed summary",
        token_count=4,
        source_token_count=4,
        source_ids=[denied_store_id],
        source_type="messages",
        created_at=0,
    ))
    captured = {}

    def synthesize(**kwargs):
        captured["context"] = json.dumps(kwargs["context_blocks"])
        return "safe answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(
            engine,
            _args(node_ids=[allowed_node_id], query=""),
            session_ids=["allowed"],
        ))
        assert result["answer"] == "safe answer"
        assert "DENIED RAW SECRET" not in captured["context"]
        assert '"session_id": "denied"' not in captured["context"]
    finally:
        engine.shutdown()


def test_cross_session_hard_bounds_override_caller_and_profile_values(tmp_path, monkeypatch):
    engine = _engine(tmp_path, max_sessions=1_000_000, per_session=1_000_000)
    node = engine._dag.get_node(_node(engine, "archive", "archive bounded"))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])
    captured = {}

    def synthesize(**kwargs):
        captured.update(kwargs)
        return "answer " * 100_000

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(
            engine,
            _args(
                max_results=1_000_000,
                max_tokens=1_000_000,
                context_max_tokens=1_000_000,
                max_sessions=1_000_000,
                deadline_ms=1_000_000_000,
            ),
            session_ids=["archive"],
        ))
        assert captured["max_tokens"] <= 8_192
        assert result["context_max_tokens"] <= 65_536
        assert result["max_sessions"] <= 10
        assert result["max_summaries_per_session"] <= 20
        assert result["deadline_ms"] <= 120_000
        assert result["answer_truncated"] is True
        assert len(result["answer"]) < 100_000
    finally:
        engine.shutdown()
