"""Bounded regressions for issue #23 over-target tool-only tails."""

from __future__ import annotations

from typing import Any

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SUMMARY_SENTINEL = "ISSUE23_CANONICAL_SUMMARY"


def _tool_call(call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "probe", "arguments": "{}"},
        }],
    }


def _tool_result(call_id: str, sentinel: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": sentinel + " " + ("payload " * 320),
    }


def _multi_tool_call(*call_ids: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "probe", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "ISSUE23_SYSTEM_ANCHOR"},
        {"role": "user", "content": "ISSUE23_COVERED_USER choose the safest recovery"},
        _tool_call("call-0"),
        _tool_result("call-0", "ISSUE23_RESULT_0"),
        _tool_call("call-1"),
        _tool_result("call-1", "ISSUE23_RESULT_1"),
        _tool_call("call-2"),
        _tool_result("call-2", "ISSUE23_RESULT_2"),
    ]


def _engine(
    tmp_path,
    *,
    async_enabled: bool = False,
    fresh_tail_count: int = 5,
    database_name: str = "issue-23.db",
) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / database_name),
        fresh_tail_count=fresh_tail_count,
        leaf_chunk_tokens=1,
        context_threshold=0.10,
        max_assembly_tokens=0,
        reserve_tokens_floor=0,
        async_background_compaction_enabled=async_enabled,
        async_background_compaction_worker_enabled=False,
    ))
    engine.update_model(
        model="issue-23-regression",
        provider="test",
        context_length=10_000,
    )
    engine.on_session_start(
        "issue-23-session",
        conversation_id="issue-23-conversation",
        platform="test",
        context_length=10_000,
    )
    return engine


def test_foreground_publication_preserves_summary_and_tool_transaction(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    messages = _messages()
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda chunk, focus_topic=None: (
            list(chunk), 100, SUMMARY_SENTINEL, 1, 1
        ),
    )
    try:
        result = engine.compress(
            messages,
            current_tokens=engine.threshold_tokens + 5,
        )
        combined = "\n".join(str(message.get("content") or "") for message in result)
        assert result
        assert combined.count(SUMMARY_SENTINEL) == 1
        for sentinel in (
            "ISSUE23_RESULT_0",
            "ISSUE23_RESULT_1",
            "ISSUE23_RESULT_2",
        ):
            assert combined.count(sentinel) == 1
        assert engine._ingest_cursor == len(result) > 0
    finally:
        engine.shutdown()


def test_async_summarizer_receives_complete_structured_tool_transaction(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        async_enabled=True,
        fresh_tail_count=1,
        database_name="structured-tool.db",
    )
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        _multi_tool_call("call-a", "call-b"),
        {**_tool_result("call-a", "RESULT_A"), "name": "probe"},
        {**_tool_result("call-b", "RESULT_B"), "name": "probe"},
        {"role": "user", "content": "TAIL"},
    ]
    captured: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda chunk, focus_topic=None: (
            captured.append([dict(message) for message in chunk]) or list(chunk),
            100,
            SUMMARY_SENTINEL,
            1,
            1,
        ),
    )
    try:
        engine.ingest(messages)
        assert engine.prepare_background_compaction_once(messages) is not None
        assistant = next(
            message for message in captured[0] if message.get("role") == "assistant"
        )
        tools = [
            message for message in captured[0] if message.get("role") == "tool"
        ]
        assert assistant.get("tool_calls") == messages[2]["tool_calls"]
        assert [message.get("tool_call_id") for message in tools] == [
            "call-a",
            "call-b",
        ]
        assert [message.get("name") for message in tools] == ["probe", "probe"]
    finally:
        engine.shutdown()
