"""Issues #18/#21: cache economics, explicit hot-cache signals, and execution."""

from __future__ import annotations

import time

import hermes_lcm.cache_aware as cache_aware
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _messages(count=8):
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn-{index} " + ("payload " * 30),
        }
        for index in range(count)
    ]


def _engine(tmp_path, *, mode="deferred", economics="discounted", **overrides):
    rule = {
        "name": "route-policy",
        "match": {"provider": "proxy", "model": "model-a", "route": "custom"},
        "overrides": {
            "preparation_threshold": 0.20,
            "cutover_threshold": 0.30,
            "post_compaction_target": 0.10,
            "emergency_threshold": 0.95,
            "cache_economics": economics,
            "compaction_mode": mode,
            "cache_ttl_seconds": 300,
        },
    }
    values = dict(
        database_path=str(tmp_path / f"{mode}-{economics}.db"),
        policy_rules=[rule],
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.50,
        async_background_compaction_enabled=True,
    )
    values.update(overrides)
    engine = LCMEngine(config=LCMConfig(**values))
    engine.on_session_start(
        "cache-session",
        conversation_id="cache-conversation",
        platform="test",
        model="model-a",
        provider="proxy",
        api_mode="custom",
        context_length=1_000,
    )
    return engine


def test_cache_read_telemetry_never_creates_hot_signal_or_economic_discount(tmp_path):
    engine = _engine(tmp_path, economics="none")
    try:
        engine.update_from_response(
            {"prompt_tokens": 1_000, "cache_read_tokens": 900, "cache_write_tokens": 0}
        )

        status = engine.get_status()
        assert status["last_cache_read_tokens"] == 900
        assert status["compaction_policy"]["cache_economics"] == "none"
        assert status["hot_cache_signal"]["state"] == "unknown"
        assert status["hot_cache_signal"]["source"] == "none"
    finally:
        engine.shutdown()


def test_explicit_hot_write_defers_deferred_mode_until_ttl(monkeypatch, tmp_path):
    clock = {"now": 100.0}
    monkeypatch.setattr(cache_aware.time, "monotonic", lambda: clock["now"])
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.record_cache_signal("write", source="host", ttl_seconds=10)

        assert engine.should_compress_preflight(messages) is False
        assert engine.last_compression_noop_reason == "hot-cache-deferred"
        assert engine.get_status()["hot_cache_signal"]["state"] == "hot"

        clock["now"] = 111.0
        assert engine.should_compress_preflight(messages) is True
        assert engine.get_status()["hot_cache_signal"]["state"] == "expired"
    finally:
        engine.shutdown()


def test_inline_mode_ignores_hot_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_aware.time, "monotonic", lambda: 100.0)
    engine = _engine(tmp_path, mode="inline", economics="none")
    try:
        engine.record_cache_signal("write", source="host", ttl_seconds=100)
        assert engine.should_compress_preflight(_messages()) is True
        assert engine.get_status()["compaction_policy"]["selected_strategy"] == "aggressive-inline"
    finally:
        engine.shutdown()


def test_cache_break_and_route_change_invalidate_hot_signal(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_aware.time, "monotonic", lambda: 100.0)
    engine = _engine(tmp_path)
    try:
        engine.record_cache_signal("write", source="host", ttl_seconds=100)
        assert engine.cache_signal_status()["state"] == "hot"
        engine.record_cache_signal("break", source="host")
        assert engine.cache_signal_status()["state"] == "broken"

        engine.record_cache_signal("write", source="host", ttl_seconds=100)
        engine.update_model(
            "model-b", 1_000, provider="proxy", api_mode="custom"
        )
        assert engine.cache_signal_status()["state"] == "route-mismatch"
    finally:
        engine.shutdown()


def test_economics_changes_strategy_and_fingerprint_not_observed_telemetry(tmp_path):
    discounted = _engine(tmp_path / "discounted", economics="discounted")
    no_discount = _engine(tmp_path / "none", economics="none")
    try:
        discounted.update_from_response({"prompt_tokens": 500, "cache_read_tokens": 250})
        no_discount.update_from_response({"prompt_tokens": 500, "cache_read_tokens": 250})
        discounted_policy = discounted.get_status()["compaction_policy"]
        no_discount_policy = no_discount.get_status()["compaction_policy"]

        assert discounted_policy["selected_strategy"] == "cache-aware-deferred"
        assert no_discount_policy["selected_strategy"] == "cache-aware-deferred"
        assert discounted_policy["fingerprint"] != no_discount_policy["fingerprint"]
        assert discounted.last_cache_read_tokens == no_discount.last_cache_read_tokens == 250
    finally:
        discounted.shutdown()
        no_discount.shutdown()


def test_deferred_maintain_prepares_after_hot_ttl(monkeypatch, tmp_path):
    clock = {"now": 100.0}
    monkeypatch.setattr(cache_aware.time, "monotonic", lambda: clock["now"])
    engine = _engine(tmp_path)
    messages = _messages()
    calls = {"count": 0}

    def summarize(chunk, focus_topic=None, timeout_seconds=None):
        calls["count"] += 1
        return list(chunk), 1_000, "prepared after ttl", 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        engine.ingest(messages)
        engine.record_cache_signal("write", source="host", ttl_seconds=10)
        assert engine.maintain(messages, force=True) is None
        assert calls["count"] == 0

        clock["now"] = 111.0
        batch = engine.maintain(messages, force=True)
        assert batch is not None
        assert calls["count"] == 1
    finally:
        engine.shutdown()
