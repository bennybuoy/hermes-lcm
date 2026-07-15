"""Issue #10 adversarial multi-tool pairing and bounded repair cases."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path):
    return LCMEngine(config=LCMConfig(database_path=str(tmp_path / "pairs.db")))


def _assistant(*call_ids):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "tool", "arguments": "{}"}}
            for call_id in call_ids
        ],
    }


def _tool(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_out_of_order_multi_results_are_reordered_without_losing_valid_pair(tmp_path):
    engine = _engine(tmp_path)
    try:
        result = engine._sanitize_tool_pairs([
            _assistant("a", "b"), _tool("b", "B"), _tool("a", "A"),
            {"role": "user", "content": "after"},
        ])
        assert [(item.get("role"), item.get("tool_call_id")) for item in result] == [
            ("assistant", None), ("tool", "a"), ("tool", "b"), ("user", None),
        ]
        assert [item["content"] for item in result[1:3]] == ["A", "B"]
    finally:
        engine.shutdown()


def test_duplicate_orphan_and_blank_results_are_dropped(tmp_path):
    engine = _engine(tmp_path)
    try:
        result = engine._sanitize_tool_pairs([
            _tool("orphan", "before"),
            _assistant("a"),
            _tool("a", "first"),
            _tool("a", "duplicate"),
            _tool("", "blank"),
        ])
        assert len(result) == 2
        assert result[1] == _tool("a", "first")
    finally:
        engine.shutdown()


def test_missing_result_stub_is_one_per_missing_id_and_bounded(tmp_path):
    engine = _engine(tmp_path)
    try:
        result = engine._sanitize_tool_pairs([_assistant("a", "b"), _tool("b", "B")])
        assert result[1]["tool_call_id"] == "a"
        assert len(result[1]["content"]) < 100
        assert result[2] == _tool("b", "B")

        without_repairs = engine._sanitize_tool_pairs(
            [_assistant("a", "b"), _tool("b", "B")],
            insert_missing_tool_stubs=False,
        )
        assert without_repairs == [_assistant("a", "b"), _tool("b", "B")]
    finally:
        engine.shutdown()

