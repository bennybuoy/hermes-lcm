"""Issue #11 bounded externalized-payload search regressions."""

from __future__ import annotations

import json
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import externalize_ingest_payload
from hermes_lcm.tools import lcm_grep


def _engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "grep.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_path=str(tmp_path / "payloads"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start(
        "grep-session",
        conversation_id="grep-conversation",
        platform="test",
        context_length=100_000,
    )
    return engine


def _payload(engine, content, *, session_id="grep-session"):
    result = externalize_ingest_payload(
        content,
        role="tool",
        session_id=session_id,
        field_path="content",
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    assert result is not None
    return result["path"].name


def test_default_grep_scope_does_not_search_payload_files(tmp_path):
    engine = _engine(tmp_path)
    try:
        _payload(engine, "payload-only phoenix needle")
        result = json.loads(lcm_grep({"query": "phoenix"}, engine=engine))
        assert result["content_scope"] == "database"
        assert not any(hit.get("type") == "externalized" for hit in result["results"])
    finally:
        engine.shutdown()


def test_externalized_scope_reports_bounded_match_metadata(tmp_path):
    engine = _engine(tmp_path)
    try:
        ref = _payload(engine, "first line\nlarge JSON phoenix value\nlast line")
        result = json.loads(
            lcm_grep(
                {
                    "query": "phoenix",
                    "content_scope": "externalized",
                    "ref": ref,
                    "max_payload_chars": 64,
                },
                engine=engine,
            )
        )

        assert result["total_results"] == 1
        hit = result["results"][0]
        assert hit["type"] == "externalized"
        assert hit["ref"] == ref
        assert hit["session_id"] == "grep-session"
        assert hit["line"] == 2
        assert hit["char_offset"] >= 0
        assert hit["byte_offset"] >= 0
        assert "phoenix" in hit["matched_text"].lower()
        assert result["scan"]["files_scanned"] == 1
        assert result["scan"]["bytes_scanned"] > 0
        assert result["scan"]["max_payload_chars"] == 64
    finally:
        engine.shutdown()


def test_externalized_regex_large_single_line_and_prefix_truncation(tmp_path):
    engine = _engine(tmp_path)
    try:
        _payload(
            engine,
            '{"items":"' + ("x" * 100) + ' TOKEN-4821 ' + ("tail" * 200) + '"}',
        )
        result = json.loads(
            lcm_grep(
                {
                    "query": r"TOKEN-\d+",
                    "regex": True,
                    "content_scope": "externalized",
                    "max_payload_chars": 160,
                },
                engine=engine,
            )
        )
        assert result["total_results"] == 1
        assert result["results"][0]["line"] == 1
        assert result["results"][0]["payload_truncated"] is True
    finally:
        engine.shutdown()


def test_adversarial_regex_is_killed_at_cpu_deadline(tmp_path):
    engine = _engine(tmp_path)
    try:
        ref = _payload(engine, ("a" * 80_000) + "!")
        started = time.monotonic()
        result = json.loads(
            lcm_grep(
                {
                    "query": r"(a+)+$",
                    "regex": True,
                    "content_scope": "externalized",
                    "ref": ref,
                    "max_payload_chars": 100_000,
                },
                engine=engine,
            )
        )
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert result["total_results"] == 0
        assert result["diagnostics"] == [{"ref": ref, "error": "regex_timeout"}]
        assert result["scan"]["regex_timeouts"] == 1
        assert result["scan"]["regex_file_deadline_ms"] == 75
    finally:
        engine.shutdown()


def test_externalized_ref_rejects_traversal_and_missing_is_structured(tmp_path):
    engine = _engine(tmp_path)
    try:
        traversal = json.loads(
            lcm_grep(
                {
                    "query": "x",
                    "content_scope": "externalized",
                    "ref": "../outside.json",
                },
                engine=engine,
            )
        )
        assert traversal["error"] == "invalid externalized ref"

        missing = json.loads(
            lcm_grep(
                {
                    "query": "x",
                    "content_scope": "externalized",
                    "ref": "missing.json",
                },
                engine=engine,
            )
        )
        assert missing["diagnostics"] == [
            {"ref": "missing.json", "error": "missing"}
        ]
    finally:
        engine.shutdown()


def test_cross_session_payload_search_requires_explicit_scope(tmp_path):
    engine = _engine(tmp_path)
    try:
        foreign_ref = _payload(
            engine,
            "foreign nebula evidence",
            session_id="foreign-session",
        )
        current = json.loads(
            lcm_grep(
                {"query": "nebula", "content_scope": "externalized"},
                engine=engine,
            )
        )
        assert current["total_results"] == 0
        assert any(item["ref"] == foreign_ref for item in current["diagnostics"])

        explicit = json.loads(
            lcm_grep(
                {
                    "query": "nebula",
                    "content_scope": "externalized",
                    "session_scope": "session",
                    "session_id": "foreign-session",
                },
                engine=engine,
            )
        )
        assert explicit["total_results"] == 1
        assert explicit["results"][0]["session_id"] == "foreign-session"
    finally:
        engine.shutdown()


def test_externalized_search_redacts_poisoned_legacy_payload_output(tmp_path):
    engine = _engine(tmp_path)
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    ref = "poisoned-legacy.json"
    (payload_dir / ref).write_text(json.dumps({
        "session_id": "grep-session",
        "content": "password=supersecret legacy evidence",
    }))
    try:
        result = json.loads(lcm_grep({
            "query": "supersecret",
            "content_scope": "externalized",
            "ref": ref,
        }, engine=engine))

        assert result["total_results"] == 1
        hit = result["results"][0]
        assert "supersecret" not in hit["snippet"]
        assert "supersecret" not in hit["matched_text"]
        assert "LCM sensitive redaction" in hit["snippet"]
    finally:
        engine.shutdown()
