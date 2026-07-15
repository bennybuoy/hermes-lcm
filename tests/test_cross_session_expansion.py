"""Issue #12: bounded, explicitly authorized cross-session DAG synthesis."""

from __future__ import annotations

import json
import threading
import time

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.schemas import LCM_EXPAND_QUERY
from hermes_lcm.tokens import count_tokens


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


def _node(
    engine,
    session_id,
    summary,
    *,
    content="raw evidence",
    expand_hint="",
    source="",
):
    store_id = engine._store.append(
        session_id,
        {"role": "user", "content": content},
        source=source,
    )
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
            expand_hint=expand_hint,
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


def test_cross_session_metadata_is_redacted_bounded_and_charged_to_context(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    secret = "sk-proj-cross-session-super-secret"
    node = engine._dag.get_node(_node(
        engine,
        "archive",
        "x",
        expand_hint=f"api_key: {secret} " + ("h" * 1_000_000),
        source="custom-source-" + ("s" * 250_000),
    ))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])
    captured = {}

    def synthesize(**kwargs):
        captured.update(kwargs)
        return "answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(engine, _args(context_max_tokens=64)))
        serialized_context = json.dumps(captured["context_blocks"])
        serialized_result = json.dumps(result)

        assert result["context_tokens_used"] <= 64
        assert result["context_tokens_used"] == lcm_tools._context_content_token_count(
            captured["context_blocks"]
        )
        assert secret not in serialized_context
        assert secret not in serialized_result
        assert len(serialized_context) < 10_000
        assert len(serialized_result) < 10_000
        assert len(result["matches"][0]["expand_hint"]) < 1_000
    finally:
        engine.shutdown()


def test_cross_session_model_metadata_is_redacted_and_bounded_on_success_and_failure(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    labeled_secret = "cross-session-model-credential"
    standalone_secret = "sk-proj-cross-session-standalone-credential-123456789"
    oversized_model = (
        f"provider api_key={labeled_secret} fallback={standalone_secret} "
        + ("oversized-model " * 10_000)
    )
    node = engine._dag.get_node(_node(engine, "archive", "archive model metadata"))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])

    def assert_safe_model_metadata(result):
        assert result["model_truncated"] is True
        assert labeled_secret not in result["model"]
        assert standalone_secret not in result["model"]
        assert "oversized-model " * 1_000 not in result["model"]
        assert len(result["model"]) <= lcm_tools._CROSS_SESSION_METADATA_MAX_CHARS
        assert count_tokens(result["model"]) <= lcm_tools._CROSS_SESSION_METADATA_MAX_TOKENS

    try:
        engine._config.expansion_model = oversized_model
        successful_call = {}

        def synthesize_success(**kwargs):
            successful_call.update(kwargs)
            return "bounded answer"

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize_success)
        success = json.loads(_invoke(engine, _args()))
        assert successful_call["model"] == oversized_model
        assert success["answer"] == "bounded answer"
        assert_safe_model_metadata(success)

        engine._config.expansion_model = ""
        engine._config.summary_model = oversized_model

        def synthesize_failure(**kwargs):
            assert kwargs["model"] == oversized_model
            raise RuntimeError("synthetic synthesis failure")

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize_failure)
        failure = json.loads(_invoke(engine, _args()))
        assert failure["degraded"] is True
        assert failure["error"] == "cross-session expansion synthesis failed"
        assert_safe_model_metadata(failure)
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
        assert captured == {}
        assert result["node_ids"] == []
        assert result["matches"] == []
        assert "DENIED RAW SECRET" not in json.dumps(result)
    finally:
        engine.shutdown()


def test_authorization_denies_summary_derived_from_disallowed_raw_lineage(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    denied_store_id = engine._store.append(
        "denied", {"role": "user", "content": "DENIED DESCENDANT SECRET"}
    )
    allowed_node_id = engine._dag.add_node(SummaryNode(
        session_id="allowed",
        depth=0,
        summary="SUMMARY DISCLOSES DENIED DESCENDANT SECRET",
        token_count=6,
        source_token_count=6,
        source_ids=[denied_store_id],
        source_type="messages",
        created_at=0,
    ))
    synthesized = []
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: synthesized.append(kwargs) or "unsafe",
    )
    try:
        result = json.loads(_invoke(
            engine,
            _args(node_ids=[allowed_node_id], query=""),
            session_ids=["allowed"],
        ))
        assert synthesized == []
        assert result["node_ids"] == []
        assert result["matches"] == []
        assert "DENIED DESCENDANT SECRET" not in json.dumps(result)
    finally:
        engine.shutdown()


def test_rollover_provenance_still_requires_original_raw_session_capability(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    engine._config.new_session_retain_depth = -1
    raw_id = engine._store.append(
        "current", {"role": "user", "content": "ROLLOVER DENIED RAW"}
    )
    node_id = engine._dag.add_node(SummaryNode(
        session_id="current",
        depth=0,
        summary="ROLLOVER SUMMARY REVEALS DENIED RAW",
        token_count=6,
        source_token_count=6,
        source_ids=[raw_id],
        source_type="messages",
        created_at=0,
    ))
    engine.rollover_session(
        "current",
        "carried",
        previous_messages=[],
        platform="test",
    )
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: "unsafe",
    )
    try:
        result = json.loads(_invoke(
            engine,
            _args(node_ids=[node_id], query=""),
            session_ids=["carried"],
        ))
        assert result["node_ids"] == []
        assert "ROLLOVER SUMMARY REVEALS" not in json.dumps(result)
    finally:
        engine.shutdown()


def test_mandatory_redaction_precedes_truncation_and_covers_shared_credentials(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    asia = "ASIAABCDEFGHIJKLMNOP"
    github_pat = "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH"
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "BOUNDARY_PRIVATE_KEY_MATERIAL\n"
        "-----END PRIVATE KEY-----"
    )
    node = engine._dag.get_node(_node(
        engine,
        "archive",
        "archive boundary",
        expand_hint=("x" * 2030) + pem + " " + asia + " " + github_pat,
    ))
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [node])
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: f"answer {pem} {asia} {github_pat}",
    )
    try:
        result = json.loads(_invoke(engine, _args(max_tokens=500)))
        serialized = json.dumps(result)
        for secret in ("BOUNDARY_PRIVATE_KEY_MATERIAL", asia, github_pat):
            assert secret not in serialized
        assert "BEGIN PRIVATE" not in result["matches"][0]["expand_hint"]
        assert "LCM sensitive redaction" in serialized
    finally:
        engine.shutdown()


def test_current_session_expansion_uses_same_bounded_redacted_output_boundary(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    secret = "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH"
    node_id = _node(
        engine,
        "current",
        "current million metadata",
        expand_hint=("h" * 1_000_000) + secret,
    )
    engine._config.expansion_model = ("model " * 200_000) + secret
    captured = {}

    def synthesize(**kwargs):
        captured.update(kwargs)
        return ("answer " * 200_000) + secret

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(lcm_tools.lcm_expand_query(
            {
                "prompt": "What happened?",
                "node_ids": [node_id],
                "max_tokens": 100,
                "context_max_tokens": 200,
            },
            engine=engine,
        ))
        serialized = json.dumps(result)
        assert secret not in serialized
        assert len(serialized) < 50_000
        assert len(json.dumps(captured["context_blocks"])) < 20_000
        assert len(result["model"]) <= lcm_tools._CROSS_SESSION_METADATA_MAX_CHARS
        assert result["answer_truncated"] is True
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


@pytest.mark.parametrize("evidence_kind", ["raw", "child", "root"])
@pytest.mark.parametrize(
    "secret, leaked_marker",
    [
        (
            "-----BEGIN PRIVATE KEY-----\nBOUNDARY-PEM-MATERIAL\n-----END PRIVATE KEY-----",
            "BEGIN PRIVATE KEY",
        ),
        ("ASIAABCDEFGHIJKLMNOP", "ASIA"),
        ("github_pat_11AA22BB33CC44DD55EE66FF77GG88HH", "github_pat_"),
    ],
)
def test_evidence_redaction_precedes_boundary_truncation_for_all_source_levels(
    tmp_path, monkeypatch, evidence_kind, secret, leaked_marker
):
    engine = _engine(tmp_path)
    prefix = "ordinary evidence " * 18
    payload = prefix + secret + " trailing evidence"
    leaf_id = _node(
        engine,
        "current",
        payload if evidence_kind == "child" else "ordinary child summary",
        content=payload if evidence_kind == "raw" else "ordinary raw evidence",
    )
    selected_id = leaf_id
    if evidence_kind in {"child", "root"}:
        selected_id = engine._dag.add_node(SummaryNode(
            session_id="current",
            depth=1,
            summary=payload if evidence_kind == "root" else "ordinary root summary",
            token_count=100,
            source_token_count=200,
            source_ids=[leaf_id],
            source_type="nodes",
            created_at=1,
        ))
    captured = {}

    def synthesize(**kwargs):
        captured["context"] = kwargs["context_blocks"]
        return "safe answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(lcm_tools.lcm_expand_query(
            {
                "prompt": "inspect evidence",
                "node_ids": [selected_id],
                "max_tokens": 100,
                "context_max_tokens": max(1, count_tokens(prefix + secret[:8])),
            },
            engine=engine,
        ))
        serialized_context = json.dumps(captured["context"])
        serialized_result = json.dumps(result)
        assert leaked_marker not in serialized_context
        assert leaked_marker not in serialized_result
        assert len(serialized_context) < 20_000
        assert result["context_truncated"] is True
    finally:
        engine.shutdown()


@pytest.mark.parametrize("outcome", ["no-match", "success", "degraded"])
def test_cross_session_prompt_and_query_are_always_mandatory_redacted_and_bounded(
    tmp_path, monkeypatch, outcome
):
    engine = _engine(tmp_path)
    prompt_secret = "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH"
    query_secret = "ASIAABCDEFGHIJKLMNOP"
    prompt = ("question " * 2_200) + prompt_secret
    query = ("archive " * 240) + query_secret
    node = engine._dag.get_node(_node(engine, "archive", "archive response node"))
    monkeypatch.setattr(
        engine._dag,
        "search",
        (lambda *args, **kwargs: []) if outcome == "no-match" else (lambda *args, **kwargs: [node]),
    )

    def synthesize(**kwargs):
        if outcome == "degraded":
            raise RuntimeError("synthetic degradation")
        return "safe answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(_invoke(engine, _args(prompt=prompt, query=query)))
        serialized = json.dumps(result)
        assert prompt_secret not in serialized
        assert query_secret not in serialized
        assert len(result["prompt"]) <= 20_000
        assert len(result["query"]) <= 2_000
        assert result["prompt_truncated"] is True
        assert result["query_truncated"] is True
    finally:
        engine.shutdown()


@pytest.mark.parametrize("outcome", ["no-match", "success", "degraded"])
def test_current_session_prompt_and_query_are_always_mandatory_redacted_and_bounded(
    tmp_path, monkeypatch, outcome
):
    engine = _engine(tmp_path)
    prompt_secret = "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH"
    query_secret = "ASIAABCDEFGHIJKLMNOP"
    prompt = ("question " * 2_000) + prompt_secret
    query = ("archive " * 220) + query_secret
    node = engine._dag.get_node(_node(engine, "current", "current response node"))
    monkeypatch.setattr(
        engine._dag,
        "search",
        (lambda *args, **kwargs: [])
        if outcome == "no-match"
        else (lambda *args, **kwargs: [node]),
    )
    monkeypatch.setattr(
        engine._store,
        "search",
        lambda *args, **kwargs: [],
    )

    def synthesize(**kwargs):
        if outcome == "degraded":
            raise RuntimeError("synthetic degradation")
        return "safe answer"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        result = json.loads(lcm_tools.lcm_expand_query(
            {"prompt": prompt, "query": query},
            engine=engine,
        ))
        serialized = json.dumps(result)
        assert prompt_secret not in serialized
        assert query_secret not in serialized
        assert len(result["prompt"]) <= 20_000
        assert len(result["query"]) <= 2_000
        assert result["prompt_truncated"] is True
        assert result["query_truncated"] is True
    finally:
        engine.shutdown()


def test_1600_node_adversarial_provenance_is_bounded_deadline_aware_and_fail_closed(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path, max_sessions=10, per_session=20)
    raw_id = engine._store.append(
        "archive", {"role": "user", "content": "adversarial raw evidence"}
    )
    child_id = engine._dag.add_node(SummaryNode(
        session_id="archive",
        depth=0,
        summary="archive adversarial leaf",
        token_count=1,
        source_token_count=1,
        source_ids=[raw_id],
        source_type="messages",
        created_at=1,
    ))
    for index in range(1, 1600):
        child_id = engine._dag.add_node(SummaryNode(
            session_id="archive",
            depth=index,
            summary=f"archive adversarial node {index}",
            token_count=1,
            source_token_count=1,
            source_ids=[child_id],
            source_type="nodes",
            created_at=index + 1,
        ))
    root = engine._dag.get_node(child_id)
    monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [root] * 1600)
    get_node_calls = {"count": 0}
    original_get_node = engine._dag.get_node

    def bounded_get_node(node_id):
        get_node_calls["count"] += 1
        if get_node_calls["count"] > 10_000:
            raise AssertionError("authorization performed unbounded per-candidate graph queries")
        return original_get_node(node_id)

    monkeypatch.setattr(engine._dag, "get_node", bounded_get_node)
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("truncated authorization must not synthesize")),
    )
    started = time.monotonic()
    try:
        result = json.loads(_invoke(
            engine,
            _args(deadline_ms=25, context_max_tokens=100),
            session_ids=["archive"],
        ))
        elapsed = time.monotonic() - started
        assert elapsed < 2.0
        assert result["degraded"] is True
        assert result["authorization_truncated"] or result["authorization_timed_out"]
        assert result["node_ids"] == []
        assert result["matches"] == []
        assert get_node_calls["count"] <= 10_000
    finally:
        engine.shutdown()
