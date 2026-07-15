from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, *, count: int = 4, token_cap: int = 24) -> LCMEngine:
    return LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "fresh-tail.db"),
        fresh_tail_count=count,
        fresh_tail_max_tokens=token_cap,
    ))


def test_fresh_tail_token_cap_can_select_fewer_messages_than_count(tmp_path):
    engine = _engine(tmp_path, count=4, token_cap=24)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "assistant", "content": "recent compact reply"},
        {"role": "user", "content": "newest user request"},
    ]
    try:
        start = engine._fresh_tail_start(messages)
        assert messages[start:] == messages[-2:]
        diagnostics = engine.get_status()["fresh_tail_selection"]
        assert diagnostics["count_start"] == 1
        assert diagnostics["token_start"] == 3
        assert diagnostics["selected_count"] == 2
        assert diagnostics["boundary_reason"] == "token-cap"
    finally:
        engine.shutdown()


def test_fresh_tail_preserves_one_giant_newest_user_message(tmp_path):
    engine = _engine(tmp_path, count=4, token_cap=8)
    messages = [
        {"role": "assistant", "content": "older reply"},
        {"role": "user", "content": "GIANT-USER " * 200},
    ]
    try:
        start = engine._fresh_tail_start(messages)
        assert start == 1
        diagnostics = engine.get_status()["fresh_tail_selection"]
        assert diagnostics["overflow"] is True
        assert diagnostics["overflow_reason"] == "newest-user-exceeds-token-cap"
    finally:
        engine.shutdown()


def test_fresh_tail_counts_multimodal_placeholders_for_token_boundary(tmp_path):
    engine = _engine(tmp_path, count=3, token_cap=12)
    messages = [
        {"role": "user", "content": "older"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "large multimodal result " * 40},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {"role": "user", "content": "latest"},
    ]
    try:
        start = engine._fresh_tail_start(messages)
        assert start == 2
        assert engine.get_status()["fresh_tail_selection"]["token_start"] == 2
    finally:
        engine.shutdown()


def test_zero_fresh_tail_remains_runtime_compatible(tmp_path):
    engine = _engine(tmp_path, count=0, token_cap=0)
    try:
        engine.update_model("test-model", 100_000, provider="test")
        assert engine._compaction_policy.fresh_tail_count == 0
        assert engine._effective_fresh_tail_count() == 0
    finally:
        engine.shutdown()
