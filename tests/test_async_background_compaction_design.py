"""Design-gate tests for opt-in async/background compaction.

These tests lock the public/private contract for prepare → promote CAS
validation, pending-summary invisibility, and status/doctor reporting.
"""

from __future__ import annotations

import json
import threading

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, *, session_id="async-session", conversation_id="async-conversation"):
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
    )
    # Future config fields. They are dynamic here so these RED tests can be
    # written before the dataclass grows the real fields.
    config.async_background_compaction_enabled = True
    config.async_background_compaction_worker_enabled = False
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=1_000,
    )
    return engine


def _messages(count=10, *, prefix="message"):
    messages = [{"role": "system", "content": "system prompt"}]
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(
            {
                "role": role,
                "content": f"{prefix} {idx} " + ("x " * 12),
            }
        )
    return messages


def test_default_disabled_async_compaction_is_inert(tmp_path):
    """Given default config, background prep is disabled and reports zero async debt."""
    config = LCMConfig(
        database_path=str(tmp_path / "disabled.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "disabled-session",
        conversation_id="disabled-conversation",
        platform="test",
        context_length=1_000,
    )
    try:
        messages = _messages()
        engine.ingest(messages)

        result = engine.prepare_background_compaction_once(messages)

        assert result is None or result.state == "disabled"
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["enabled"] is False
        assert status["async_compaction"]["pending_batches"] == 0
        assert status["async_compaction"]["prepared_batches"] == 0
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_pending_summaries_are_invisible_until_atomic_promotion(tmp_path):
    """Given prepared pending leaves, active context/readers ignore them until promotion."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(messages)

        assert batch.state == "ready"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["dag"]["total_nodes"] == 0
        grep = json.loads(engine.handle_tool_call("lcm_grep", {"query": "message"}))
        assert all(result.get("kind") != "pending_summary" for result in grep.get("results", []))
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_stale_source_identity(tmp_path):
    """Given source rows changed after prep, promotion rejects without canonical mutation."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        first_source_id = batch.source_ids[0]
        engine._store._conn.execute(
            "UPDATE messages SET content = content || ' reconciled late' WHERE store_id = ?",
            (first_source_id,),
        )
        engine._store._conn.commit()

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "source_identity_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        assert engine.get_async_compaction_status()["rejected_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_config_change(tmp_path):
    """Given live config changes after prep, live policy wins over stale persisted metadata."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.fresh_tail_count = 6

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_summary_route_change(tmp_path):
    """Given summary model changes after prep, promotion rejects stale route output."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.summary_model = "different-summary-model"

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "summary_route_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_threshold_policy_change(tmp_path):
    """Given threshold changes after prep, live config beats persisted batch policy."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.context_threshold = 0.75

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_foreground_compaction_race_supersedes_pending_batch(tmp_path, monkeypatch):
    """Given foreground compaction lands first, stale pending work is rejected/superseded."""
    engine = _engine(tmp_path)
    try:
        # Force the leaf-compaction race: disable turn-boundary auto-promote so
        # compress() takes the synchronous path and invalidates the ready batch.
        engine._config.async_background_compaction_promote_on_compress = False
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground summary", 0),
        )
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
        assert engine._frontier.get_batch(batch.batch_id).state == "superseded"
        assert engine.get_async_compaction_status()["prepared_batches"] == 0
        result = engine.promote_prepared_compaction(batch.batch_id, compacted)

        assert engine._dag.get_session_node_count(engine.current_session_id) >= 1
        assert result.promoted is False
        assert result.reason == "batch_state_superseded"
        async_status = engine.get_async_compaction_status()
        assert async_status["superseded_batches"] + async_status["rejected_batches"] >= 1
    finally:
        engine.shutdown()


def test_summary_failure_backoff_does_not_wedge_foreground_compaction(tmp_path, monkeypatch):
    """Given background summary failure, backoff is visible but foreground can still compact."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("summary spend backoff open")),
        )
        batch = engine.prepare_background_compaction_once(messages)
        assert batch.state == "failed"
        assert engine.get_async_compaction_status()["failed_batches"] == 1

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground recovery summary", 0),
        )
        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)

        assert compacted != messages
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_restart_recovers_or_discards_pending_batches_safely(tmp_path):
    """Given pending/preparing rows at shutdown, restart never treats them as canonical."""
    db_path = tmp_path / "restart.db"
    config = LCMConfig(database_path=str(db_path), fresh_tail_count=2, leaf_chunk_tokens=20)
    config.async_background_compaction_enabled = True

    engine = LCMEngine(config=config)
    engine.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
    messages = _messages()
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, leave_state="preparing")
        assert batch.state == "preparing"
    finally:
        engine.shutdown()

    restarted = LCMEngine(config=config)
    try:
        restarted.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
        status = restarted.get_async_compaction_status()

        assert restarted._dag.get_session_node_count(restarted.current_session_id) == 0
        assert status["preparing_batches"] == 0
        assert status["pending_batches"] + status["rejected_batches"] + status["failed_batches"] >= 1
    finally:
        restarted.shutdown()


def test_successful_atomic_promotion_is_all_or_nothing(tmp_path):
    """Given a valid ready batch, node insert/frontier advance/batch state commit together."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
        assert engine._dag.get_session_node_count(engine.current_session_id) == batch.expected_leaf_count
        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] > old_frontier
        assert lifecycle["current_frontier_store_id"] == batch.frontier_end_store_id
        assert engine.get_async_compaction_status()["promoted_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rolls_back_partial_publish_failure(tmp_path):
    """Given a mid-promotion failure, no canonical node/frontier/batch half-state remains."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]
        engine._async_compaction_publish_failure_hook = "after_canonical_insert"

        with pytest.raises(RuntimeError, match="injected async promotion failure"):
            engine.promote_prepared_compaction(batch.batch_id, messages)

        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] == old_frontier
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        async_status = engine.get_async_compaction_status()
        assert async_status["promoted_batches"] == 0
        assert async_status["prepared_batches"] == 1
    finally:
        engine.shutdown()


def test_promotion_rolls_back_frontier_when_lifecycle_persist_fails(tmp_path, monkeypatch):
    """A lifecycle write failure must not leave a committed frontier generation."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        before = engine._frontier.get_active_frontier(batch.conversation_id)["generation"]

        def fail_lifecycle(*_args, **_kwargs):
            raise RuntimeError("injected lifecycle persist failure")

        monkeypatch.setattr(engine._lifecycle, "advance_frontier", fail_lifecycle)
        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert engine._frontier.get_active_frontier(batch.conversation_id)["generation"] == before
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        assert engine._frontier.get_batch(batch.batch_id).state == "rejected"
    finally:
        engine.shutdown()


def test_shutdown_keeps_storage_open_until_worker_has_exited(tmp_path):
    """A timed-out worker must never resume against already-closed SQLite handles."""
    engine = _engine(tmp_path)

    class StuckWorker:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            return None

    worker = StuckWorker()
    engine._async_worker_thread = worker
    engine._async_worker_stop = threading.Event()
    engine.shutdown()

    assert engine._frontier._conn is not None
    worker.alive = False
    engine.shutdown()
    assert engine._frontier._conn is None


def test_promotion_rejects_lifecycle_noop_after_frontier_cas(tmp_path, monkeypatch):
    """A stale lifecycle session result must compensate the CAS frontier."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        before = engine._frontier.get_active_frontier(batch.conversation_id)["generation"]
        original = engine._lifecycle.advance_frontier

        def stale_lifecycle(conversation_id, _session_id, frontier_store_id):
            return original(conversation_id, "wrong-session", frontier_store_id)

        monkeypatch.setattr(engine._lifecycle, "advance_frontier", stale_lifecycle)
        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert engine._frontier.get_active_frontier(batch.conversation_id)["generation"] == before
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_rebind_defers_storage_close_while_worker_is_alive(tmp_path):
    """A profile rebind cannot close SQLite while a timed-out worker may resume."""
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))

    class StuckWorker:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            return None

    worker = StuckWorker()
    engine._async_worker_thread = worker
    engine._async_worker_stop = threading.Event()
    original_db = engine._store.db_path
    assert engine._rebind_storage_for_home(str(home_b)) is False
    assert engine._store.db_path == original_db
    assert engine._frontier._conn is not None

    worker.alive = False
    assert engine._rebind_storage_for_home(str(home_b)) is True
    assert engine._store.db_path != original_db
    engine.shutdown()


def test_rebind_does_not_close_storage_against_replacement_worker(tmp_path):
    """Stop/rebind must not close SQLite under a concurrent replacement worker.

    Covers the unlocked gap in a naive stop: a replacement can be installed
    after the snapshotted worker exits and before rebind closes SQLite.
    Ownership must serialize start with the full stop→close→rebind transition
    so storage never closes while any worker could use it.
    """
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))
    engine._config.async_background_compaction_enabled = True
    engine._config.async_background_compaction_worker_enabled = True

    join_entered = threading.Event()
    release_join = threading.Event()

    class ControllableWorker:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            # Appear dead so a concurrent start may race the stop ownership
            # window if the lifecycle transition does not own install rights.
            self._alive = False
            join_entered.set()
            release_join.wait(timeout=5.0)

    original = ControllableWorker()
    engine._async_worker_thread = original
    engine._async_worker_stop = threading.Event()
    original_db = engine._store.db_path
    original_frontier = engine._frontier
    rebind_ok: dict[str, bool] = {}

    def run_rebind() -> None:
        rebind_ok["value"] = engine._rebind_storage_for_home(str(home_b))

    rebind_thread = threading.Thread(target=run_rebind, name="rebind-under-test")
    rebind_thread.start()
    assert join_entered.wait(timeout=5.0)

    # Attempt to install a replacement while stop/rebind claims storage.
    start_thread = threading.Thread(
        target=engine._start_async_worker,
        name="late-start-during-rebind",
    )
    start_thread.start()
    # Give a naive unlocked start a chance to slip through before release.
    start_thread.join(timeout=0.2)
    assert start_thread.is_alive(), "late start must wait on the owner transition"
    # While the transition owns stop→close→bind, no replacement may install
    # and the pre-rebind storage helpers must still be open.
    assert engine._async_worker_thread is original
    assert original_frontier._conn is not None
    assert engine._store.db_path == original_db

    release_join.set()
    rebind_thread.join(timeout=10.0)
    start_thread.join(timeout=10.0)
    assert not rebind_thread.is_alive()
    assert not start_thread.is_alive()

    # Ownership completed a safe bind; late start may then install against the
    # rebound storage (bound lifecycle). That is not a close-under-worker race.
    assert rebind_ok.get("value") is True
    assert engine._store.db_path != original_db
    assert getattr(engine, "_storage_lifetime_state", None) == "bound"
    live_worker = engine._async_worker_thread
    if live_worker is not None and getattr(live_worker, "is_alive", lambda: False)():
        assert live_worker is not original

    engine._config.async_background_compaction_worker_enabled = False
    engine.shutdown()


def test_overlapping_lifecycle_owners_serialize_and_block_late_start(tmp_path):
    """Two overlapping owners plus a late start must not break storage ownership.

    Regression for shared boolean barriers: two transitions can both claim
    protection; the first can finish and clear the barrier while the second is
    still between stop and close, letting a late start install a worker that
    the second then closes beneath.

    Required invariant: no worker becomes live after a transition has claimed
    storage until that transition completes a safe bind or releases ownership;
    no SQLite helper closes while any worker could use it. Successful shutdown
    leaves a terminal closed state that rejects further installs.
    """
    home_a, home_b, home_c = tmp_path / "home-a", tmp_path / "home-b", tmp_path / "home-c"
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))
    engine._config.async_background_compaction_enabled = True
    engine._config.async_background_compaction_worker_enabled = True

    join_entered = threading.Event()
    release_join = threading.Event()

    class ControllableWorker:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            self._alive = False
            join_entered.set()
            release_join.wait(timeout=5.0)

    original = ControllableWorker()
    engine._async_worker_thread = original
    engine._async_worker_stop = threading.Event()
    original_frontier = engine._frontier
    results: dict[str, object] = {}

    def owner_rebind() -> None:
        results["rebind_ok"] = engine._rebind_storage_for_home(str(home_b))

    def owner_shutdown() -> None:
        engine.shutdown()
        results["shutdown_done"] = True

    def late_start() -> None:
        engine._start_async_worker()
        results["late_start_done"] = True
        results["worker_after_late_start"] = engine._async_worker_thread

    rebind_thread = threading.Thread(target=owner_rebind, name="owner-rebind")
    rebind_thread.start()
    assert join_entered.wait(timeout=5.0)

    # Second owner and a late start race while the first still owns stop→close.
    shutdown_thread = threading.Thread(target=owner_shutdown, name="owner-shutdown")
    start_thread = threading.Thread(target=late_start, name="late-start-between-owners")
    shutdown_thread.start()
    start_thread.start()

    # While owner A holds the transition, neither B nor late start may complete,
    # and no replacement worker may become live against storage A is claiming.
    shutdown_thread.join(timeout=0.25)
    start_thread.join(timeout=0.05)
    assert shutdown_thread.is_alive(), "second owner must wait on first transition"
    assert start_thread.is_alive(), "late start must wait on the owner transition"
    assert "shutdown_done" not in results
    assert "late_start_done" not in results
    assert engine._async_worker_thread is original
    assert original_frontier._conn is not None

    release_join.set()
    rebind_thread.join(timeout=10.0)
    shutdown_thread.join(timeout=10.0)
    start_thread.join(timeout=10.0)
    assert not rebind_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert not start_thread.is_alive()

    # Shutdown is a terminal close: storage helpers must be closed and no
    # worker may remain live against them.
    assert results.get("shutdown_done") is True
    assert getattr(engine, "_storage_lifetime_state", None) == "closed"
    assert engine._frontier._conn is None
    live_worker = engine._async_worker_thread
    assert live_worker is None or not getattr(live_worker, "is_alive", lambda: False)()

    # Late starts after terminal close must not resurrect a worker.
    engine._config.async_background_compaction_enabled = True
    engine._config.async_background_compaction_worker_enabled = True
    engine._start_async_worker()
    resurrected = engine._async_worker_thread
    assert resurrected is None or not getattr(resurrected, "is_alive", lambda: False)()

    # A later successful rebind re-opens storage ownership so normal session
    # starts may install a worker again (standard on_session_start path).
    assert engine._rebind_storage_for_home(str(home_c)) is True
    assert getattr(engine, "_storage_lifetime_state", None) == "bound"
    engine.on_session_start(
        "post-rebind-session",
        conversation_id="post-rebind-conversation",
        platform="test",
        hermes_home=str(home_c),
    )
    restarted = engine._async_worker_thread
    assert restarted is not None and restarted.is_alive()

    engine._config.async_background_compaction_worker_enabled = False
    engine.shutdown()


def test_failed_bind_after_close_leaves_storage_non_startable(tmp_path, monkeypatch):
    """Injected _bind_storage failure after close must refuse later worker starts.

    A rebind that closes helpers then fails mid-bind must never retain a
    startable/bound state over closed or half-bound SQLite helpers.
    """
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))
    engine._config.async_background_compaction_enabled = True
    engine._config.async_background_compaction_worker_enabled = True
    original_frontier = engine._frontier
    original_bind = engine._bind_storage

    def fail_bind(db_path, hermes_home=""):
        raise RuntimeError("injected bind failure")

    monkeypatch.setattr(engine, "_bind_storage", fail_bind)
    assert engine._rebind_storage_for_home(str(home_b)) is False
    assert getattr(engine, "_storage_lifetime_state", None) == "unusable"
    # Pre-rebind helpers were closed; must not pretend they are still open/usable.
    assert original_frontier._conn is None

    engine._start_async_worker()
    started = engine._async_worker_thread
    assert started is None or not getattr(started, "is_alive", lambda: False)()

    # Recovery path: a later successful rebind re-opens and allows starts.
    monkeypatch.setattr(engine, "_bind_storage", original_bind)
    assert engine._rebind_storage_for_home(str(home_b)) is True
    assert getattr(engine, "_storage_lifetime_state", None) == "bound"
    engine.on_session_start(
        "recover-session",
        conversation_id="recover-conversation",
        platform="test",
        hermes_home=str(home_b),
    )
    recovered = engine._async_worker_thread
    assert recovered is not None and recovered.is_alive()

    engine._config.async_background_compaction_worker_enabled = False
    engine.shutdown()


def test_session_lifecycle_serializes_with_shutdown_without_closed_conn_errors(tmp_path):
    """Session start/end storage work must serialize with concurrent shutdown.

    Pause on_session_start after a successful rebind and on_session_end before
    final ingest; invoke shutdown while each is mid-critical-section; release
    and assert no AttributeError/ProgrammingError and coherent outcomes
    (session work completes against open storage, or cleanly observes terminal
    closed/unusable state — never half-closed mid-write).
    """
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))
    engine.on_session_start(
        "lifecycle-session",
        conversation_id="lifecycle-conversation",
        platform="test",
        hermes_home=str(home_a),
        context_length=1_000,
    )

    # --- Phase 1: pause on_session_start after rebind, race shutdown ---
    start_entered = threading.Event()
    release_start = threading.Event()
    start_errors: list[BaseException] = []
    start_done = threading.Event()

    original_bind_lifecycle = engine._bind_lifecycle_state

    def pause_after_rebind_bind_lifecycle(*args, **kwargs):
        start_entered.set()
        assert release_start.wait(timeout=5.0)
        return original_bind_lifecycle(*args, **kwargs)

    engine._bind_lifecycle_state = pause_after_rebind_bind_lifecycle  # type: ignore[method-assign]

    def run_session_start() -> None:
        try:
            engine.on_session_start(
                "lifecycle-session-2",
                conversation_id="lifecycle-conversation-2",
                platform="test",
                hermes_home=str(home_b),
                context_length=1_000,
            )
        except BaseException as exc:  # noqa: BLE001 — capture for assertion
            start_errors.append(exc)
        finally:
            start_done.set()

    start_thread = threading.Thread(target=run_session_start, name="paused-session-start")
    start_thread.start()
    assert start_entered.wait(timeout=5.0)

    shutdown_errors: list[BaseException] = []
    shutdown_done = threading.Event()

    def run_shutdown() -> None:
        try:
            engine.shutdown()
        except BaseException as exc:  # noqa: BLE001
            shutdown_errors.append(exc)
        finally:
            shutdown_done.set()

    shutdown_thread = threading.Thread(target=run_shutdown, name="shutdown-vs-start")
    shutdown_thread.start()
    # Shutdown must wait on the session-start lifetime critical section.
    shutdown_thread.join(timeout=0.25)
    assert shutdown_thread.is_alive(), "shutdown must wait for in-flight session start"
    assert not shutdown_done.is_set()

    release_start.set()
    start_thread.join(timeout=10.0)
    shutdown_thread.join(timeout=10.0)
    assert not start_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert start_errors == []
    assert shutdown_errors == []
    assert getattr(engine, "_storage_lifetime_state", None) == "closed"
    assert engine._frontier._conn is None

    # --- Phase 2: fresh engine, pause on_session_end before final ingest ---
    engine2 = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(home_a))
    engine2.on_session_start(
        "end-session",
        conversation_id="end-conversation",
        platform="test",
        hermes_home=str(home_a),
        context_length=1_000,
    )
    engine2.ingest(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]
    )

    end_entered = threading.Event()
    release_end = threading.Event()
    end_errors: list[BaseException] = []
    end_done = threading.Event()
    original_ingest = engine2._ingest_messages

    def pause_before_final_ingest(messages, **kwargs):
        end_entered.set()
        assert release_end.wait(timeout=5.0)
        return original_ingest(messages, **kwargs)

    engine2._ingest_messages = pause_before_final_ingest  # type: ignore[method-assign]

    def run_session_end() -> None:
        try:
            engine2.on_session_end(
                "end-session",
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello world"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "final turn"},
                ],
            )
        except BaseException as exc:  # noqa: BLE001
            end_errors.append(exc)
        finally:
            end_done.set()

    end_thread = threading.Thread(target=run_session_end, name="paused-session-end")
    end_thread.start()
    assert end_entered.wait(timeout=5.0)

    shutdown2_errors: list[BaseException] = []
    shutdown2_done = threading.Event()

    def run_shutdown2() -> None:
        try:
            engine2.shutdown()
        except BaseException as exc:  # noqa: BLE001
            shutdown2_errors.append(exc)
        finally:
            shutdown2_done.set()

    shutdown2_thread = threading.Thread(target=run_shutdown2, name="shutdown-vs-end")
    shutdown2_thread.start()
    shutdown2_thread.join(timeout=0.25)
    assert shutdown2_thread.is_alive(), "shutdown must wait for in-flight session end"
    assert not shutdown2_done.is_set()
    # Storage must remain open while session-end still holds the lifetime lease.
    assert engine2._frontier._conn is not None
    assert engine2._store._conn is not None

    release_end.set()
    end_thread.join(timeout=10.0)
    shutdown2_thread.join(timeout=10.0)
    assert not end_thread.is_alive()
    assert not shutdown2_thread.is_alive()
    assert end_errors == []
    assert shutdown2_errors == []
    assert getattr(engine2, "_storage_lifetime_state", None) == "closed"
    assert engine2._frontier._conn is None


def test_session_reset_serializes_with_shutdown_without_closed_conn_errors(tmp_path):
    """on_session_reset storage work must serialize with concurrent shutdown.

    Pause mid-reset after the lifecycle record write path is entered; invoke
    shutdown while the reset critical section holds storage; release and assert
    no AttributeError/ProgrammingError and coherent closed outcome.
    """
    home = tmp_path / "home-reset"
    engine = LCMEngine(
        config=LCMConfig(database_path="", new_session_retain_depth=0),
        hermes_home=str(home),
    )
    engine.on_session_start(
        "reset-session",
        conversation_id="reset-conversation",
        platform="test",
        hermes_home=str(home),
        context_length=1_000,
    )
    # Ensure DAG has something a retain_depth=0 reset would delete.
    engine.ingest(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "reset me " + ("x " * 20)},
            {"role": "assistant", "content": "ok " + ("y " * 20)},
        ]
    )

    reset_entered = threading.Event()
    release_reset = threading.Event()
    reset_errors: list[BaseException] = []
    original_record_reset = engine._lifecycle.record_reset

    def pause_record_reset(conversation_id):
        reset_entered.set()
        assert release_reset.wait(timeout=5.0)
        return original_record_reset(conversation_id)

    engine._lifecycle.record_reset = pause_record_reset  # type: ignore[method-assign]

    def run_reset() -> None:
        try:
            engine.on_session_reset()
        except BaseException as exc:  # noqa: BLE001
            reset_errors.append(exc)

    reset_thread = threading.Thread(target=run_reset, name="paused-session-reset")
    reset_thread.start()
    assert reset_entered.wait(timeout=5.0)

    shutdown_errors: list[BaseException] = []

    def run_shutdown() -> None:
        try:
            engine.shutdown()
        except BaseException as exc:  # noqa: BLE001
            shutdown_errors.append(exc)

    shutdown_thread = threading.Thread(target=run_shutdown, name="shutdown-vs-reset")
    shutdown_thread.start()
    shutdown_thread.join(timeout=0.25)
    assert shutdown_thread.is_alive(), "shutdown must wait for in-flight session reset"
    # Storage must remain open while reset still holds the lifetime lease.
    assert engine._lifecycle._conn is not None
    assert engine._dag._conn is not None

    release_reset.set()
    reset_thread.join(timeout=10.0)
    shutdown_thread.join(timeout=10.0)
    assert not reset_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert reset_errors == []
    assert shutdown_errors == []
    assert getattr(engine, "_storage_lifetime_state", None) == "closed"
    assert engine._lifecycle._conn is None
    assert engine._dag._conn is None


def test_status_and_doctor_report_async_compaction_counts(tmp_path):
    """Given mixed async states, status and doctor expose pending/prepared/promoted/rejected counts."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        ready = engine.prepare_background_compaction_once(messages)
        engine.reject_prepared_compaction(ready.batch_id, reason="policy_fingerprint_mismatch")
        engine.prepare_background_compaction_once(messages)

        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        doctor = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["async_compaction"]["rejected_batches"] == 1
        async_checks = [check for check in doctor["checks"] if check["check"].startswith("async_compaction")]
        assert async_checks
        assert any("prepared_batches" in check["detail"] for check in async_checks)
    finally:
        engine.shutdown()
