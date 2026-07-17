"""Issue #14 and schema-v8 persisted focus overlay tests."""

from __future__ import annotations

import json
import sqlite3

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.db_bootstrap import SCHEMA_VERSION, get_schema_version, run_versioned_migrations
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _engine(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "focus.db"),
            focus_context_tokens=500,
            focus_output_tokens=100,
            focus_timeout_ms=1_000,
            focus_max_source_nodes=4,
        )
    )
    engine.on_session_start("session", conversation_id="conversation", platform="test")
    return engine


def _node(engine, summary, content):
    store_id = engine._store.append("session", {"role": "user", "content": content})
    node_id = engine._dag.add_node(
        SummaryNode(
            session_id="session",
            depth=0,
            summary=summary,
            token_count=10,
            source_token_count=20,
            source_ids=[store_id],
            source_type="messages",
            created_at=float(store_id),
        )
    )
    return node_id, store_id


def test_focus_persists_immutable_evidence_and_survives_restart(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    node_id, store_id = _node(engine, "project alpha decision", "use the blue deployment")
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: f"Use blue deployment [node {node_id}]. Uncertainty: rollout date unknown.",
    )
    try:
        created = engine.create_focus("project alpha")
        assert created["source_node_ids"] == [node_id]
        assert created["covered_store_id"] == store_id
        focus_id = created["focus_id"]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            engine._focus._conn.execute(
                "UPDATE lcm_focus_briefs SET content = 'mutated' WHERE focus_id = ?",
                (focus_id,),
            )
    finally:
        engine.shutdown()

    restarted = _engine(tmp_path)
    try:
        status = restarted.get_focus_status(preview_chars=0)
        assert status["active"] is True
        assert status["focus_id"] == focus_id
        assert "content" not in status
        assert "preview" not in status
        assert status["source_node_ids"] == [node_id]
    finally:
        restarted.shutdown()


def test_focus_overlay_assembles_without_mutating_canonical_dag_or_fresh_tail(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    node_id, _ = _node(engine, "alpha evidence", "canonical raw evidence")
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: f"ACTIVE ALPHA BRIEF [node {node_id}]",
    )
    try:
        before_nodes = engine._dag.get_session_node_count("session")
        engine.create_focus("alpha")
        tail = [{"role": "user", "content": "new uncovered tail remains"}]
        assembled = engine._assemble_context(None, tail)
        serialized = json.dumps(assembled)
        assert "ACTIVE ALPHA BRIEF" in serialized
        assert "alpha evidence" in serialized
        assert "new uncovered tail remains" in serialized
        assert engine._dag.get_session_node_count("session") == before_nodes
        assert engine._frontier.get_active_frontier("conversation") is None
    finally:
        engine.shutdown()


def test_focus_overlay_never_overruns_assembly_budget(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _node(engine, "alpha evidence", "canonical raw evidence")
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: "oversized-focus " * 200,
    )
    try:
        engine.create_focus("alpha")
        tail = [{"role": "user", "content": "fresh objective"}]
        assembled = engine._assemble_context(
            None,
            tail,
            assembly_cap_override=40,
        )
        assert count_messages_tokens(assembled) <= 40
        assert "fresh objective" in json.dumps(assembled)
        assert "oversized-focus" not in json.dumps(assembled)
    finally:
        engine.shutdown()


def test_refocus_uses_only_post_watermark_dag_delta(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    first_id, _ = _node(engine, "alpha baseline", "old raw evidence")
    calls = []

    def synthesize(**kwargs):
        calls.append(kwargs["context_blocks"])
        return f"brief version {len(calls)} [node {first_id}]"

    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", synthesize)
    try:
        first = engine.create_focus("alpha")
        second_id, second_store_id = _node(engine, "alpha delta", "NEW DELTA EVIDENCE")
        second = engine.create_focus("", refocus=True)

        assert second["refocus"] is True
        assert second["supersedes_focus_id"] == first["focus_id"]
        assert second["covered_store_id"] == second_store_id
        assert second["source_node_ids"] == [first_id, second_id]
        serialized_delta = json.dumps(calls[1])
        assert "NEW DELTA EVIDENCE" in serialized_delta
        assert "old raw evidence" not in serialized_delta
        assert engine._focus.get(first["focus_id"]).active is False
    finally:
        engine.shutdown()


def test_failed_refocus_preserves_previous_active_brief(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _node(engine, "alpha baseline", "old evidence")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "stable brief")
    try:
        first = engine.create_focus("alpha")
        _node(engine, "alpha delta", "new evidence")

        def timeout(**kwargs):
            raise TimeoutError

        monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", timeout)
        result = engine.create_focus("", refocus=True)
        assert result["previous_focus_preserved"] is True
        assert engine.get_focus_status()["focus_id"] == first["focus_id"]
        assert len(engine._focus.history("conversation")) == 1
    finally:
        engine.shutdown()


def test_unfocus_deactivates_overlay_without_deleting_history(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _node(engine, "alpha baseline", "evidence")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "brief")
    try:
        created = engine.create_focus("alpha")
        result = engine.unfocus()
        assert result == {
            "active": False,
            "deactivated_focus_id": created["focus_id"],
            "history_preserved": True,
        }
        assert engine.get_focus_status()["active"] is False
        assert engine._focus.get(created["focus_id"]) is not None
        assert len(engine._focus.history("conversation")) == 1
    finally:
        engine.shutdown()


def test_focus_output_surfaces_redact_sensitive_values(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    assert engine._config.sensitive_patterns_enabled is False
    _node(engine, "alpha credentials", "credential evidence")
    monkeypatch.setattr(
        lcm_tools,
        "_synthesize_expansion_answer",
        lambda **kwargs: "password=supersecret",
    )
    try:
        created = engine.create_focus("alpha password=promptsecret")
        shown = engine.get_focus_status(preview_chars=500)
        serialized = json.dumps({"created": created, "shown": shown})
        assert "supersecret" not in serialized
        assert "promptsecret" not in serialized
        assert "LCM sensitive redaction" in serialized
        persisted = engine._focus.get(created["focus_id"])
        assert persisted is not None
        assert "promptsecret" not in persisted.prompt
        assert "LCM sensitive redaction" in persisted.prompt
    finally:
        engine.shutdown()


def test_failed_refocus_redacts_previous_focus_metadata(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    _node(engine, "alpha credentials", "credential evidence")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "brief")
    try:
        created = engine.create_focus("alpha password=promptsecret")
        monkeypatch.setattr(engine._dag, "search", lambda *args, **kwargs: [])
        monkeypatch.setattr(engine._dag, "get_session_nodes", lambda *args, **kwargs: [])
        result = engine.create_focus("", refocus=True)
        serialized = json.dumps(result)
        assert result["previous_focus_preserved"] is True
        assert created["focus_id"] == result["focus"]["focus_id"]
        assert "promptsecret" not in serialized
        assert "LCM sensitive redaction" in serialized
    finally:
        engine.shutdown()


def test_focus_tool_and_slash_surfaces_are_bounded(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _node(engine, "alpha baseline", "evidence")
    monkeypatch.setattr(lcm_tools, "_synthesize_expansion_answer", lambda **kwargs: "brief content")
    try:
        created = json.loads(
            engine.handle_tool_call("lcm_focus", {"action": "focus", "prompt": "alpha"})
        )
        assert created["active"] is True
        shown = json.loads(engine.handle_tool_call("lcm_focus", {"action": "show"}))
        assert shown["preview"] == "brief content"
        assert "brief content" not in json.dumps(engine.get_status()["focus"])
        assert "brief content" in handle_lcm_command("focus", engine)
        assert "history_preserved: True" in handle_lcm_command("unfocus", engine)
    finally:
        engine.shutdown()


def test_v7_fixture_migrates_to_v8_and_restarts_idempotently(tmp_path):
    path = tmp_path / "v7.db"
    bootstrap = LCMEngine(config=LCMConfig(database_path=str(path)))
    bootstrap.shutdown()
    conn = sqlite3.connect(path)
    # Materialize a complete store, then remove only the v8 objects/column to
    # create a structurally genuine v7 compatibility fixture.
    conn.execute("DROP TABLE lcm_focus_briefs")
    conn.execute(
        "ALTER TABLE lcm_prepared_batches DROP COLUMN resolved_policy_json"
    )
    # Remove the v8-only database guard while reconstructing this v7 fixture.
    conn.execute("DROP TRIGGER lcm_schema_version_monotonic")
    conn.execute(
        "UPDATE metadata SET value = '7' WHERE key = 'schema_version'"
    )
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name = 'v8_focus_and_resolved_policy_metadata'"
    )
    conn.execute(
        """INSERT INTO messages
           (session_id, role, content, timestamp, token_estimate, conversation_id)
           VALUES ('legacy-session', 'user', 'legacy v7 raw evidence', 1.0, 5, 'legacy-conversation')"""
    )
    conn.commit()
    run_versioned_migrations(conn)
    assert get_schema_version(conn) == SCHEMA_VERSION == 10
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_focus_briefs'"
    ).fetchone()
    prepared_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_prepared_batches)")
    }
    assert "resolved_policy_json" in prepared_columns
    assert conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v8_focus_and_resolved_policy_metadata'"
    ).fetchone()
    assert conn.execute(
        "SELECT content FROM messages WHERE session_id = 'legacy-session'"
    ).fetchone() == ("legacy v7 raw evidence",)
    conn.close()

    restarted = sqlite3.connect(path)
    run_versioned_migrations(restarted)
    assert get_schema_version(restarted) == SCHEMA_VERSION
    assert restarted.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    restarted.close()
