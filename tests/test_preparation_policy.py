"""Issue #19: pressure/utility-aware asynchronous preparation."""

from __future__ import annotations

import json
import threading
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _messages(count: int, prefix: str = "turn") -> list[dict]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"{prefix}-{index} " + ("payload " * 20),
        }
        for index in range(count)
    ]


def _engine(
    tmp_path,
    *,
    session="prep-session",
    database=None,
    runtime_context_length=10_000,
    **overrides,
):
    values = dict(
        database_path=str(database or (tmp_path / "prep.db")),
        async_background_compaction_enabled=True,
        async_preparation_utility_policy_enabled=True,
        async_preparation_min_reduction_tokens=10,
        async_candidate_refresh_min_tokens=100,
        async_max_candidates_per_conversation=2,
        async_max_candidates_per_profile=4,
        async_summary_admission_limit=1,
        async_ready_ttl_seconds=3_600,
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.50,
    )
    values.update(overrides)
    engine = LCMEngine(config=LCMConfig(**values))
    engine.on_session_start(
        session,
        conversation_id=f"conversation-{session}",
        platform="test",
        context_length=runtime_context_length,
    )
    return engine


def _stub_summary(monkeypatch, engine, *, wait=None, entered=None):
    calls = {"count": 0}

    def summarize(messages, focus_topic=None, timeout_seconds=None):
        calls["count"] += 1
        if entered is not None:
            entered.set()
        if wait is not None:
            wait.wait(timeout=5)
        return list(messages), 1_000, "useful compact summary", 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    return calls


def test_small_session_below_preparation_pressure_makes_zero_calls(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(6)
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=400
        )

        assert batch is None
        assert calls["count"] == 0
        status = engine.get_async_compaction_status()
        assert status["prepare_skip_reasons"]["below-preparation-pressure"] == 1
        assert status["last_pressure_signal"] == "host"
        assert status["last_host_prompt_tokens"] == 400
        assert status["last_source_tokens"] > 0
    finally:
        engine.shutdown()


def test_pressure_telemetry_reports_host_source_mismatch(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(8)
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, host_prompt_tokens=9_000)

        status = engine.get_async_compaction_status()
        assert status["last_pressure_signal"] == "host"
        assert status["last_pressure_mismatch_tokens"] == abs(
            status["last_host_prompt_tokens"] - status["last_source_tokens"]
        )
        assert status["total_prepared"] == 1
        persisted_policy = json.loads(batch.resolved_policy_json)
        assert persisted_policy["fingerprint"]
        assert persisted_policy["fresh_tail_count"] == 2
        assert persisted_policy["leaf_chunk_tokens"] == 20
        assert persisted_policy["preparation_threshold"] < persisted_policy["cutover_threshold"]
    finally:
        engine.shutdown()


def test_resolved_policy_survives_restart_without_runtime_route_metadata(
    tmp_path, monkeypatch
):
    database = tmp_path / "metadata-absent.db"
    engine = _engine(
        tmp_path,
        database=database,
        runtime_context_length=0,
    )
    messages = _messages(8)
    _stub_summary(monkeypatch, engine)
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, force=True)
        assert batch is not None
        persisted = batch.resolved_policy()
        assert persisted["fingerprint"]
        assert persisted["selection_reason"] == "global fallback"
        assert persisted["fresh_tail_count"] == 2
    finally:
        engine.shutdown()

    restarted = _engine(
        tmp_path,
        database=database,
        runtime_context_length=0,
    )
    try:
        promoted = restarted.promote_prepared_compaction(batch.batch_id, messages)
        assert promoted.promoted is True
    finally:
        restarted.shutdown()


def test_candidate_refresh_suppresses_same_range_and_small_append(tmp_path, monkeypatch):
    engine = _engine(tmp_path, async_candidate_refresh_min_tokens=500)
    calls = _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(8)
        engine.ingest(messages)
        first = engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=9_000
        )
        assert first is not None
        assert calls["count"] == 1

        assert engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=9_000
        ) is None
        grown = messages + _messages(2, prefix="small-growth")
        engine.ingest(grown)
        assert engine.prepare_background_compaction_once(
            grown, host_prompt_tokens=9_000
        ) is None

        status = engine.get_async_compaction_status()
        assert status["prepare_skip_reasons"]["candidate-already-covers-range"] >= 2
        assert calls["count"] == 1
    finally:
        engine.shutdown()


def test_profile_admission_limits_concurrent_engine_clones(tmp_path, monkeypatch):
    database = tmp_path / "shared-profile.db"
    first = _engine(tmp_path, session="first", database=database)
    second = _engine(tmp_path, session="second", database=database)
    release = threading.Event()
    entered = threading.Event()
    first_calls = _stub_summary(monkeypatch, first, wait=release, entered=entered)
    second_calls = _stub_summary(monkeypatch, second)
    first_messages = _messages(8, "first")
    second_messages = _messages(8, "second")
    first.ingest(first_messages)
    second.ingest(second_messages)
    thread = threading.Thread(
        target=lambda: first.prepare_background_compaction_once(
            first_messages, host_prompt_tokens=9_000
        )
    )
    try:
        thread.start()
        assert entered.wait(timeout=3)

        assert second.prepare_background_compaction_once(
            second_messages, host_prompt_tokens=9_000
        ) is None
        assert second_calls["count"] == 0
        assert second.get_async_compaction_status()["prepare_skip_reasons"][
            "admission-limited"
        ] == 1
    finally:
        release.set()
        thread.join(timeout=5)
        first.shutdown()
        second.shutdown()
    assert first_calls["count"] == 1


def test_ready_ttl_cleanup_supersedes_abandoned_candidate(tmp_path, monkeypatch):
    engine = _engine(tmp_path, async_ready_ttl_seconds=5)
    _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(8)
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=9_000
        )
        engine._frontier.conn.execute(
            "UPDATE lcm_prepared_batches SET updated_at = ? WHERE batch_id = ?",
            (time.time() - 60, batch.batch_id),
        )
        engine._frontier.conn.commit()

        engine.prepare_background_compaction_once(messages, host_prompt_tokens=9_000)

        expired = engine._frontier.get_batch(batch.batch_id)
        assert expired.state == "superseded"
        assert expired.failure_reason == "ttl-expired"
        assert engine.get_async_compaction_status()["cleanup_counts"]["ttl-expired"] >= 1
    finally:
        engine.shutdown()


def test_spend_backoff_skips_before_candidate_or_summary_call(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(8)
        engine.ingest(messages)
        engine._summary_spend_guard._backoff_until = time.monotonic() + 60

        assert engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=9_000
        ) is None
        assert calls["count"] == 0
        assert engine.get_async_compaction_status()["prepare_skip_reasons"][
            "spend-backoff"
        ] == 1
    finally:
        engine.shutdown()


def test_manual_force_bypasses_pressure_but_not_safety_caps(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = _stub_summary(monkeypatch, engine)
    try:
        messages = _messages(6)
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(
            messages, host_prompt_tokens=100, force=True
        )

        assert batch is not None
        assert calls["count"] == 1
    finally:
        engine.shutdown()


def test_candidate_caps_are_atomic_per_conversation_and_profile(tmp_path, monkeypatch):
    database = tmp_path / "candidate-caps.db"
    first = _engine(
        tmp_path,
        session="cap-first",
        database=database,
        async_summary_admission_limit=2,
        async_max_candidates_per_conversation=1,
        async_max_candidates_per_profile=1,
        async_candidate_refresh_min_tokens=1,
    )
    second = _engine(
        tmp_path,
        session="cap-second",
        database=database,
        async_summary_admission_limit=2,
        async_max_candidates_per_conversation=1,
        async_max_candidates_per_profile=1,
        async_candidate_refresh_min_tokens=1,
    )
    _stub_summary(monkeypatch, first)
    second_calls = _stub_summary(monkeypatch, second)
    first_messages = _messages(8, "cap-first")
    second_messages = _messages(8, "cap-second")
    try:
        first.ingest(first_messages)
        second.ingest(second_messages)
        assert first.prepare_background_compaction_once(
            first_messages, host_prompt_tokens=9_000
        ) is not None

        assert second.prepare_background_compaction_once(
            second_messages, host_prompt_tokens=9_000
        ) is None
        assert second_calls["count"] == 0
        assert second.get_async_compaction_status()["prepare_skip_reasons"][
            "profile-candidate-limit"
        ] == 1
    finally:
        first.shutdown()
        second.shutdown()
