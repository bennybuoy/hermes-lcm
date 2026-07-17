"""Non-vacuous rollover transaction, crash, and final-tail regressions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


OLD = "transactional-rollover-old"
NEW = "transactional-rollover-new"
CONVERSATION = "transactional-rollover-conversation"
TAIL = "FINAL HOST TAIL MUST BE DURABLE"
PRECOMMIT_PHASES = (
    "after_begin",
    "after_revalidation",
    "after_tail_ingest",
    "after_prune",
    "after_reassign",
    "after_frontier",
    "after_lifecycle",
)


def _config(db_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        new_session_retain_depth=2,
        async_background_compaction_worker_enabled=False,
    )


def _seed(db_path: Path) -> tuple[list[dict], int, int]:
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    messages = [{"role": "user", "content": "published old raw"}]
    engine.ingest(messages)
    store_id = engine._get_store_ids_for_messages(messages)[0]
    retained = SummaryNode(
        session_id=OLD,
        depth=2,
        summary="retained transactional rollover summary",
        token_count=4,
        source_token_count=5,
        source_ids=[store_id],
        source_type="messages",
        created_at=1.0,
    )
    published = engine._publish_foreground_leaf(
        node=retained,
        source_end_store_id=store_id,
        covered_source_ids=[store_id],
    )
    assert published["published"] is True
    pruned_id = engine._dag.add_node(
        SummaryNode(
            session_id=OLD,
            depth=0,
            summary="pruned transactional rollover summary",
            token_count=4,
            source_token_count=5,
            source_ids=[store_id],
            source_type="messages",
            created_at=2.0,
        )
    )
    engine.shutdown()
    return [*messages, {"role": "assistant", "content": TAIL}], int(retained.node_id), int(pruned_id)


def _seed_mixed_frontier(db_path: Path) -> tuple[list[dict], int, int]:
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    raw = {"role": "assistant", "content": "PREEXISTING RAW FRONTIER ITEM"}
    engine.ingest([raw])
    raw_id = engine._get_store_ids_for_messages([raw])[0]
    active = engine._frontier.get_active_frontier(CONVERSATION)
    assert active is not None
    generation = engine._frontier.advance_frontier_generation_with_items(
        CONVERSATION,
        OLD,
        raw_id,
        str(active["policy_fingerprint"] or ""),
        str(active["route_fingerprint"] or ""),
        int(active["generation"]),
        [
            {
                "kind": "node",
                "ref_id": retained_id,
                "source_start": int(active["source_end_store_id"]),
                "source_end": int(active["source_end_store_id"]),
            },
            {
                "kind": "message",
                "ref_id": raw_id,
                "source_start": raw_id,
                "source_end": raw_id,
            },
        ],
    )
    assert generation == int(active["generation"]) + 1
    engine.shutdown()
    return [previous_messages[0], raw, previous_messages[-1]], retained_id, raw_id


def _snapshot(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        active = conn.execute(
            """SELECT generation, session_id, source_end_store_id
               FROM lcm_active_frontiers WHERE conversation_id = ?
               ORDER BY generation DESC LIMIT 1""",
            (CONVERSATION,),
        ).fetchone()
        lifecycle = conn.execute(
            """SELECT current_session_id, last_finalized_session_id,
                      current_frontier_store_id, last_finalized_frontier_store_id
               FROM lcm_lifecycle_state WHERE conversation_id = ?""",
            (CONVERSATION,),
        ).fetchone()
        nodes = conn.execute(
            "SELECT node_id, session_id, depth FROM summary_nodes ORDER BY node_id"
        ).fetchall()
        items = conn.execute(
            """SELECT kind, ref_id FROM lcm_frontier_items
               WHERE conversation_id = ? AND generation = ? ORDER BY ordinal""",
            (CONVERSATION, int(active[0])),
        ).fetchall()
        messages = conn.execute(
            "SELECT session_id, role, content FROM messages ORDER BY store_id"
        ).fetchall()
        batches = conn.execute(
            "SELECT state FROM lcm_prepared_batches ORDER BY batch_id"
        ).fetchall()
        return {
            "active": active,
            "lifecycle": lifecycle,
            "nodes": nodes,
            "items": items,
            "messages": messages,
            "batches": batches,
        }
    finally:
        conn.close()


def _canonical_snapshot(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        active = conn.execute(
            """SELECT generation, session_id, source_end_store_id
               FROM lcm_active_frontiers WHERE conversation_id = ?
               ORDER BY generation DESC LIMIT 1""",
            (CONVERSATION,),
        ).fetchone()
        assert active is not None
        items = conn.execute(
            """SELECT kind, ref_id, source_start, source_end
               FROM lcm_frontier_items
               WHERE conversation_id = ? AND generation = ?
               ORDER BY ordinal""",
            (CONVERSATION, int(active[0])),
        ).fetchall()
        messages = conn.execute(
            """SELECT store_id, session_id, role, content
               FROM messages WHERE conversation_id = ? ORDER BY store_id""",
            (CONVERSATION,),
        ).fetchall()
        lifecycle = conn.execute(
            """SELECT current_session_id, last_finalized_session_id,
                      current_frontier_store_id, last_finalized_frontier_store_id
               FROM lcm_lifecycle_state WHERE conversation_id = ?""",
            (CONVERSATION,),
        ).fetchone()
        generations = conn.execute(
            """SELECT generation, session_id, source_end_store_id
               FROM lcm_active_frontiers WHERE conversation_id = ?
               ORDER BY generation""",
            (CONVERSATION,),
        ).fetchall()
        generation_items = conn.execute(
            """SELECT generation, ordinal, kind, ref_id, source_start, source_end
               FROM lcm_frontier_items WHERE conversation_id = ?
               ORDER BY generation, ordinal""",
            (CONVERSATION,),
        ).fetchall()
        return {
            "active": active,
            "items": items,
            "messages": messages,
            "lifecycle": lifecycle,
            "generations": generations,
            "generation_items": generation_items,
        }
    finally:
        conn.close()


def _assert_raw_frontier_coverage(state: dict, contents: set[str]) -> None:
    store_ids = {
        int(store_id)
        for store_id, session_id, _role, content in state["messages"]
        if session_id == OLD and content in contents
    }
    raw_items = {
        int(ref_id)
        for kind, ref_id, source_start, source_end in state["items"]
        if kind == "message"
        and int(ref_id) == int(source_start) == int(source_end)
    }
    assert store_ids
    assert store_ids <= raw_items
    durable_old_ids = {
        int(store_id)
        for store_id, session_id, _role, _content in state["messages"]
        if session_id == OLD and int(store_id) <= int(state["active"][2])
    }
    for store_id in durable_old_ids:
        assert sum(
            int(source_start) <= store_id <= int(source_end)
            for _kind, _ref_id, source_start, source_end in state["items"]
        ) == 1
    assert int(state["active"][2]) == max(raw_items)
    assert state["lifecycle"][2:] == (state["active"][2], state["active"][2])


def _crash(tmp_path: Path, db_path: Path, phase: str, previous_messages: list[dict]):
    package_root = tmp_path / f"rollover-package-{phase}"
    package_root.mkdir(exist_ok=True)
    (package_root / "hermes_lcm").symlink_to(
        Path(__file__).resolve().parents[1], target_is_directory=True
    )
    script = """
import json
import sys
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1],
    new_session_retain_depth=2,
    async_background_compaction_worker_enabled=False,
))
engine.on_session_start(
    "transactional-rollover-old",
    conversation_id="transactional-rollover-conversation",
    platform="test",
)
engine._rollover_publish_crash_hook = sys.argv[2]
engine.rollover_session(
    "transactional-rollover-old",
    "transactional-rollover-new",
    previous_messages=json.loads(sys.argv[3]),
    carry_over_context=True,
    platform="test",
)
raise SystemExit("crash hook did not fire")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(package_root), env.get("PYTHONPATH", "")) if value
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase, json.dumps(previous_messages)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("phase", (*PRECOMMIT_PHASES, "after_commit"))
def test_rollover_crash_exposes_exact_wholly_old_or_wholly_new_state_and_restarts(
    tmp_path, phase
):
    db_path = tmp_path / f"rollover-{phase}.db"
    previous_messages, retained_id, pruned_id = _seed(db_path)
    old = _snapshot(db_path)
    crashed = _crash(tmp_path, db_path, phase, previous_messages)
    assert crashed.returncode == 88, (crashed.stdout, crashed.stderr)
    state = _snapshot(db_path)

    if phase in PRECOMMIT_PHASES:
        assert state == old
    else:
        assert state["active"][1] == NEW
        assert state["lifecycle"][0] == NEW
        assert state["lifecycle"][1] == OLD
        assert state["lifecycle"][2] == state["active"][2]
        assert state["nodes"] == [(retained_id, NEW, 2)]
        assert state["items"] == [
            ("node", retained_id),
            ("message", state["active"][2]),
        ]
        assert any(row[0] == OLD and row[2] == TAIL for row in state["messages"])
        assert all(row[0] != "ready" and row[0] != "preparing" for row in state["batches"])
        assert all(row[0] != pruned_id for row in state["nodes"])

    reopened = LCMEngine(config=_config(db_path))
    try:
        restart_session = OLD if phase in PRECOMMIT_PHASES else NEW
        reopened.on_session_start(
            restart_session,
            conversation_id=CONVERSATION,
            platform="test",
        )
        assert _snapshot(db_path) == state
    finally:
        reopened.shutdown()


def test_final_tail_ingest_lock_error_aborts_rollover_and_keeps_host_tail_visible(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "rollover-tail-lock.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    before = _snapshot(db_path)
    original_messages = json.loads(json.dumps(previous_messages))

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine_module, "protect_messages_for_ingest", locked)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            engine.rollover_session(
                OLD,
                NEW,
                previous_messages=previous_messages,
                carry_over_context=True,
                platform="test",
            )
        assert engine.current_session_id == OLD
        assert previous_messages == original_messages
        assert previous_messages[-1]["content"] == TAIL
        assert _snapshot(db_path) == before
    finally:
        engine.shutdown()


def test_normal_ingest_serializes_across_rollover_snapshot_publication(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "rollover-ingest-race.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    snapshot_captured = threading.Event()
    permit_publication = threading.Event()
    ingest_finished = threading.Event()
    failures: list[BaseException] = []
    original_prepare = engine._prepare_rollover_final_tail

    def pause_after_snapshot(*args, **kwargs):
        tail = original_prepare(*args, **kwargs)
        snapshot_captured.set()
        assert permit_publication.wait(5)
        return tail

    monkeypatch.setattr(engine, "_prepare_rollover_final_tail", pause_after_snapshot)

    def rollover_worker():
        try:
            engine.rollover_session(
                OLD,
                NEW,
                previous_messages=previous_messages,
                carry_over_context=True,
                platform="test",
            )
        except BaseException as exc:  # pragma: no cover - assertion reports below
            failures.append(exc)

    def ingest_worker():
        try:
            engine.ingest([{"role": "user", "content": "CONCURRENT INGEST"}])
        except BaseException as exc:  # pragma: no cover - assertion reports below
            failures.append(exc)
        finally:
            ingest_finished.set()

    rollover_thread = threading.Thread(target=rollover_worker)
    ingest_thread = threading.Thread(target=ingest_worker)
    try:
        rollover_thread.start()
        assert snapshot_captured.wait(5)
        ingest_thread.start()
        assert not ingest_finished.wait(0.25), (
            "normal ingest crossed the captured rollover snapshot"
        )
        permit_publication.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        assert not rollover_thread.is_alive()
        assert not ingest_thread.is_alive()
        assert failures == []

        rows = engine._store._conn.execute(
            "SELECT session_id, content FROM messages ORDER BY store_id"
        ).fetchall()
        assert rows.count((OLD, TAIL)) == 1
        assert rows.count((NEW, "CONCURRENT INGEST")) == 1
    finally:
        permit_publication.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        engine.shutdown()


@pytest.mark.parametrize("ingest_path", ["tool_call", "preflight", "compression"])
def test_every_live_ingest_path_serializes_across_rollover_publication(
    tmp_path, monkeypatch, ingest_path
):
    """No live writer may append against the old binding after tail capture."""
    db_path = tmp_path / f"rollover-{ingest_path}-race.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    snapshot_captured = threading.Event()
    permit_publication = threading.Event()
    ingest_finished = threading.Event()
    failures: list[BaseException] = []
    original_prepare = engine._prepare_rollover_final_tail
    live_messages = [{"role": "user", "content": f"CONCURRENT {ingest_path.upper()} INGEST"}]

    def pause_after_snapshot(*args, **kwargs):
        tail = original_prepare(*args, **kwargs)
        snapshot_captured.set()
        assert permit_publication.wait(5)
        return tail

    monkeypatch.setattr(engine, "_prepare_rollover_final_tail", pause_after_snapshot)

    def rollover_worker():
        try:
            engine.rollover_session(
                OLD,
                NEW,
                previous_messages=previous_messages,
                carry_over_context=True,
                platform="test",
            )
        except BaseException as exc:  # pragma: no cover - assertion reports below
            failures.append(exc)

    def ingest_worker():
        try:
            if ingest_path == "tool_call":
                engine.handle_tool_call("lcm_status", {}, messages=live_messages)
            elif ingest_path == "preflight":
                engine.should_compress_preflight(live_messages)
            else:
                engine.compress(live_messages)
        except BaseException as exc:  # pragma: no cover - assertion reports below
            failures.append(exc)
        finally:
            ingest_finished.set()

    rollover_thread = threading.Thread(target=rollover_worker)
    ingest_thread = threading.Thread(target=ingest_worker)
    try:
        rollover_thread.start()
        assert snapshot_captured.wait(5)
        ingest_thread.start()
        assert not ingest_finished.wait(0.25), (
            f"{ingest_path} ingest crossed the captured rollover snapshot"
        )
        permit_publication.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        assert not rollover_thread.is_alive()
        assert not ingest_thread.is_alive()
        assert failures == []

        rows = engine._store._conn.execute(
            "SELECT session_id, content FROM messages ORDER BY store_id"
        ).fetchall()
        assert rows.count((OLD, TAIL)) == 1
        assert rows.count((NEW, live_messages[0]["content"])) == 1
        assert rows.count((OLD, live_messages[0]["content"])) == 0
    finally:
        permit_publication.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        engine.shutdown()


def test_independent_engine_ingest_between_prepare_and_publish_is_exactly_once(
    tmp_path, monkeypatch
):
    """A second SQLite connection may extend the prefix after tail prepare."""
    db_path = tmp_path / "rollover-independent-ingest-race.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    rollover_engine = LCMEngine(config=_config(db_path))
    ingest_engine = LCMEngine(config=_config(db_path))
    rollover_engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    ingest_engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    initial_generation = int(_snapshot(db_path)["active"][0])
    prepared = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    original_prepare = rollover_engine._prepare_rollover_final_tail

    def pause_after_prepare(*args, **kwargs):
        tail = original_prepare(*args, **kwargs)
        prepared.set()
        assert release.wait(5)
        return tail

    monkeypatch.setattr(
        rollover_engine, "_prepare_rollover_final_tail", pause_after_prepare
    )

    def run_rollover():
        try:
            rollover_engine.rollover_session(
                OLD,
                NEW,
                previous_messages=previous_messages,
                carry_over_context=True,
                platform="test",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=run_rollover)
    try:
        worker.start()
        assert prepared.wait(5)
        ingest_engine.ingest([previous_messages[-1]])
        rows_before_publish = ingest_engine._store._conn.execute(
            "SELECT content FROM messages ORDER BY store_id"
        ).fetchall()
        assert rows_before_publish.count(("published old raw",)) == 1
        assert rows_before_publish.count((TAIL,)) == 1
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert failures == []

        state = _snapshot(db_path)
        assert state["active"][:2] == (initial_generation + 1, NEW)
        assert state["lifecycle"][:2] == (NEW, OLD)
        assert state["messages"].count((OLD, "assistant", TAIL)) == 1
        assert state["nodes"] == [(retained_id, NEW, 2)]
        assert state["items"] == [
            ("node", retained_id),
            ("message", state["active"][2]),
        ]
    finally:
        release.set()
        worker.join(5)
        rollover_engine.shutdown()
        ingest_engine.shutdown()


def test_independent_engine_stale_ingest_cannot_commit_after_rollover(tmp_path):
    """An ingest queued behind publication must recheck durable ownership."""
    db_path = tmp_path / "rollover-independent-stale-ingest.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    rollover_engine = LCMEngine(config=_config(db_path))
    ingest_engine = LCMEngine(config=_config(db_path))
    rollover_engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    ingest_engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    owns_writer = threading.Event()
    release_writer = threading.Event()
    ingest_finished = threading.Event()
    failures: list[BaseException] = []

    def pause_with_writer(phase: str):
        if phase == "after_begin":
            owns_writer.set()
            assert release_writer.wait(5)

    rollover_engine._rollover_publish_failure_hook = pause_with_writer

    def run_rollover():
        try:
            rollover_engine.rollover_session(
                OLD,
                NEW,
                previous_messages=previous_messages,
                carry_over_context=True,
                platform="test",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def run_ingest():
        try:
            ingest_engine.ingest([{"role": "user", "content": "STALE OLD INGEST"}])
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            ingest_finished.set()

    rollover_thread = threading.Thread(target=run_rollover)
    ingest_thread = threading.Thread(target=run_ingest)
    try:
        rollover_thread.start()
        assert owns_writer.wait(5)
        ingest_thread.start()
        assert not ingest_finished.wait(0.25)
        release_writer.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        assert not rollover_thread.is_alive()
        assert not ingest_thread.is_alive()
        assert failures == []

        state = _snapshot(db_path)
        assert state["active"][1] == NEW
        assert state["lifecycle"][:2] == (NEW, OLD)
        assert all(row[2] != "STALE OLD INGEST" for row in state["messages"])
        assert state["messages"].count((OLD, "assistant", TAIL)) == 1
    finally:
        release_writer.set()
        rollover_thread.join(5)
        ingest_thread.join(5)
        rollover_engine.shutdown()
        ingest_engine.shutdown()


@pytest.mark.parametrize("targets", [(NEW, NEW), ("rollover-target-a", "rollover-target-b")])
@pytest.mark.parametrize("divergent", [False, True])
def test_two_engines_past_generation_capture_reconcile_exact_tail_coverage(
    tmp_path, monkeypatch, targets, divergent
):
    db_path = tmp_path / f"rollover-two-engine-{targets[0]}-{targets[1]}-{divergent}.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    initial = _canonical_snapshot(db_path)
    engines = [LCMEngine(config=_config(db_path)), LCMEngine(config=_config(db_path))]
    for engine in engines:
        engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")

    tail_contents = ["DIVERGENT TAIL A", "DIVERGENT TAIL B"] if divergent else [TAIL, TAIL]
    histories = [
        previous_messages + ([{"role": "assistant", "content": tail_contents[index]}] if divergent else [])
        for index in range(2)
    ]
    captured = threading.Barrier(2)
    results: list[tuple[int, str]] = []
    failures: list[BaseException] = []

    for engine in engines:
        original_prepare = engine._prepare_rollover_final_tail

        def pause_after_generation_capture(*args, _prepare=original_prepare, **kwargs):
            prepared = _prepare(*args, **kwargs)
            captured.wait(timeout=5)
            return prepared

        monkeypatch.setattr(engine, "_prepare_rollover_final_tail", pause_after_generation_capture)

    def run(index: int) -> None:
        try:
            moved = engines[index].rollover_session(
                OLD,
                targets[index],
                previous_messages=histories[index],
                carry_over_context=True,
                platform="test",
            )
            results.append((moved, engines[index].current_session_id))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    workers = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        assert all(not worker.is_alive() for worker in workers)
        assert failures == []

        state = _canonical_snapshot(db_path)
        durable_owner = state["active"][1]
        expected_advance = 2 if divergent else 1
        assert state["active"][0] == initial["active"][0] + expected_advance
        assert state["lifecycle"][:2] == (durable_owner, OLD)
        assert {session for _moved, session in results} == {durable_owner}
        assert sorted(moved for moved, _session in results) == [0, 1]
        assert [row[1] for row in state["generations"][-expected_advance:]] == [durable_owner] * expected_advance
        assert [
            row for row in state["generation_items"]
            if int(row[0]) <= int(initial["active"][0])
        ] == initial["generation_items"]
        assert any(item[:2] == ("node", retained_id) for item in state["items"])
        expected_raw = {TAIL, *tail_contents} if divergent else {TAIL}
        _assert_raw_frontier_coverage(state, expected_raw)
        for content in expected_raw:
            assert sum(row[3] == content for row in state["messages"]) == 1
    finally:
        for engine in engines:
            engine.shutdown()


def test_late_old_session_end_extends_winner_once_and_duplicate_replay_is_noop(tmp_path):
    db_path = tmp_path / "rollover-late-session-end.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    late_content = "UNIQUE LATE OLD SESSION END"
    late_messages = previous_messages + [{"role": "assistant", "content": late_content}]
    try:
        engine.rollover_session(
            OLD,
            NEW,
            previous_messages=previous_messages,
            carry_over_context=True,
            platform="test",
        )
        rolled = _canonical_snapshot(db_path)
        engine.on_session_end(OLD, late_messages)
        extended = _canonical_snapshot(db_path)

        assert extended["active"][:2] == (rolled["active"][0] + 1, NEW)
        assert extended["lifecycle"][:2] == (NEW, OLD)
        assert any(item[:2] == ("node", retained_id) for item in extended["items"])
        _assert_raw_frontier_coverage(extended, {TAIL, late_content})
        assert sum(row[3] == late_content for row in extended["messages"]) == 1

        immutable_generations = list(extended["generations"])
        immutable_items = list(extended["items"])
        engine.on_session_end(OLD, late_messages)
        replayed = _canonical_snapshot(db_path)
        assert replayed["generations"] == immutable_generations
        assert replayed["items"] == immutable_items
        assert sum(row[3] == late_content for row in replayed["messages"]) == 1
    finally:
        engine.shutdown()


def test_rollover_preserves_preexisting_mixed_node_and_raw_frontier(tmp_path):
    db_path = tmp_path / "rollover-mixed-frontier.db"
    previous_messages, retained_id, existing_raw_id = _seed_mixed_frontier(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    try:
        engine.rollover_session(
            OLD,
            NEW,
            previous_messages=previous_messages,
            carry_over_context=True,
            platform="test",
        )
        state = _canonical_snapshot(db_path)
        assert state["active"][1] == NEW
        assert ("node", retained_id) in [item[:2] for item in state["items"]]
        assert ("message", existing_raw_id) in [item[:2] for item in state["items"]]
        _assert_raw_frontier_coverage(
            state,
            {"PREEXISTING RAW FRONTIER ITEM", TAIL},
        )
    finally:
        engine.shutdown()


def test_reconverged_suffix_is_matched_and_not_duplicated_by_competing_rollover(tmp_path):
    db_path = tmp_path / "rollover-reconverged-suffix.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    prefix = previous_messages[:1]
    winner_history = prefix + [
        {"role": "assistant", "content": "BRANCH B"},
        {"role": "assistant", "content": "SHARED SUFFIX"},
    ]
    competing_history = prefix + [
        {"role": "assistant", "content": "BRANCH A"},
        {"role": "assistant", "content": "SHARED SUFFIX"},
    ]
    winner = LCMEngine(config=_config(db_path))
    competing = LCMEngine(config=_config(db_path))
    winner.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    competing.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    try:
        winner.rollover_session(OLD, NEW, previous_messages=winner_history, platform="test")
        competing.rollover_session(
            OLD,
            "losing-target",
            previous_messages=competing_history,
            platform="test",
        )
        rows = competing._store._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY store_id",
            (OLD,),
        ).fetchall()
        contents = [row[0] for row in rows]
        assert contents.count("BRANCH B") == 1
        assert contents.count("BRANCH A") == 1
        assert contents.count("SHARED SUFFIX") == 1
    finally:
        winner.shutdown()
        competing.shutdown()


def test_reconverged_duplicate_union_appends_only_host_multiplicity_deficit(tmp_path):
    db_path = tmp_path / "rollover-reconverged-duplicate-union.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    prefix = previous_messages[:1]
    winner_history = prefix + [
        {"role": "assistant", "content": "B"},
        {"role": "assistant", "content": "A"},
    ]
    competitor_history = prefix + [
        {"role": "assistant", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "assistant", "content": "A"},
    ]
    winner = LCMEngine(config=_config(db_path))
    competitor = LCMEngine(config=_config(db_path))
    winner.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    competitor.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    try:
        winner.rollover_session(OLD, NEW, previous_messages=winner_history, platform="test")
        competitor.rollover_session(
            OLD,
            "duplicate-losing-target",
            previous_messages=competitor_history,
            platform="test",
        )
        first = _canonical_snapshot(db_path)
        contents = [row[3] for row in first["messages"]]
        assert contents.count("B") == 1
        assert contents.count("A") == 2

        competitor.on_session_end(OLD, competitor_history)
        assert _canonical_snapshot(db_path) == first
    finally:
        winner.shutdown()
        competitor.shutdown()


def test_ordered_occurrence_reconciliation_preserves_multiplicity_and_scales_linearly():
    reconcile = LCMEngine._ordered_unmatched_retained_indices
    assert reconcile(
        ["P", "A", "B", "A"],
        ["P", "B", "A"],
    ) == [3]
    assert reconcile(
        ["P", "A", "S", "A", "T", "S"],
        ["P", "B", "S", "C", "T", "S"],
    ) == [1, 3]
    assert reconcile(["R", "R", "R"], ["R", "R"]) == [2]
    assert reconcile(
        ["P", "A", "B", "A", "C", "B", "A"],
        ["P", "B", "A", "C", "A"],
    ) == [5, 6]

    size = 100_000
    host = ["adversarial-repeat"] * size
    durable = ["adversarial-repeat"] * (size - 1)
    assert reconcile(host, durable) == [size - 1]


def test_duplicate_late_end_with_ignored_host_row_is_exact_noop(tmp_path, monkeypatch):
    db_path = tmp_path / "rollover-ignored-late-replay.db"
    config = _config(db_path)
    config.ignore_message_patterns = ["^IGNORE THIS ROW$"]
    engine = LCMEngine(config=config)
    monkeypatch.setattr(
        engine,
        "_matches_ignore_message_patterns",
        lambda message, **_kwargs: message.get("content") == "IGNORE THIS ROW",
    )
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    prefix = {"role": "user", "content": "retained prefix"}
    engine.ingest([prefix])
    ignored = {"role": "assistant", "content": "IGNORE THIS ROW"}
    suffix = {"role": "assistant", "content": "retained suffix after ignored row"}
    history = [prefix, ignored, suffix]
    try:
        engine.rollover_session(OLD, NEW, previous_messages=history, platform="test")
        first = _canonical_snapshot(db_path)
        assert sum(row[3] == suffix["content"] for row in first["messages"]) == 1
        assert all(row[3] != ignored["content"] for row in first["messages"])

        engine.on_session_end(OLD, history)
        second = _canonical_snapshot(db_path)
        engine.on_session_end(OLD, history)
        third = _canonical_snapshot(db_path)
        assert third == second
        assert sum(row[3] == suffix["content"] for row in third["messages"]) == 1
        assert all(row[3] != ignored["content"] for row in third["messages"])
    finally:
        engine.shutdown()


def test_no_carry_winner_stores_competing_and_late_old_tails_without_repopulation(tmp_path):
    db_path = tmp_path / "rollover-no-carry-winner.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    winner = LCMEngine(config=_config(db_path))
    competing = LCMEngine(config=_config(db_path))
    winner.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    competing.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    competing_tail = {"role": "assistant", "content": "STORED COMPETING OLD TAIL"}
    late_tail = {"role": "assistant", "content": "STORED LATE OLD TAIL"}
    try:
        winner.rollover_session(
            OLD,
            NEW,
            previous_messages=previous_messages,
            carry_over_context=False,
            platform="test",
        )
        competing.rollover_session(
            OLD,
            "losing-target",
            previous_messages=previous_messages + [competing_tail],
            carry_over_context=True,
            platform="test",
        )
        before_late_end = _canonical_snapshot(db_path)
        assert before_late_end["active"][1:] == (NEW, 0)
        assert before_late_end["items"] == []
        assert sum(row[3] == competing_tail["content"] for row in before_late_end["messages"]) == 1
        competing_boundary = max(row[0] for row in before_late_end["messages"])
        assert before_late_end["lifecycle"][3] == competing_boundary
        conn = sqlite3.connect(str(db_path))
        try:
            policy = conn.execute(
                """SELECT finalized_session_id, current_session_id,
                          finalized_cutoff_store_id, carry_over_context
                   FROM lcm_rollover_policies WHERE conversation_id = ?""",
                (CONVERSATION,),
            ).fetchone()
        finally:
            conn.close()
        assert policy == (OLD, NEW, competing_boundary, 0)

        late_history = previous_messages + [competing_tail, late_tail]
        winner.on_session_end(OLD, late_history)
        after_late_end = _canonical_snapshot(db_path)
        assert after_late_end["active"] == before_late_end["active"]
        assert after_late_end["items"] == []
        assert sum(row[3] == competing_tail["content"] for row in after_late_end["messages"]) == 1
        assert sum(row[3] == late_tail["content"] for row in after_late_end["messages"]) == 1
        late_boundary = max(row[0] for row in after_late_end["messages"])
        assert after_late_end["lifecycle"][3] == late_boundary

        winner.on_session_end(OLD, late_history)
        assert _canonical_snapshot(db_path) == after_late_end
    finally:
        winner.shutdown()
        competing.shutdown()


def test_two_engine_stale_generation_race_obeys_no_carry_winner(tmp_path, monkeypatch):
    db_path = tmp_path / "rollover-no-carry-generation-race.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    winner = LCMEngine(config=_config(db_path))
    stale = LCMEngine(config=_config(db_path))
    winner.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    stale.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    stale_captured = threading.Event()
    release_stale = threading.Event()
    failures: list[BaseException] = []
    original_prepare = stale._prepare_rollover_final_tail

    def pause_stale_after_generation_capture(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        stale_captured.set()
        assert release_stale.wait(5)
        return prepared

    monkeypatch.setattr(stale, "_prepare_rollover_final_tail", pause_stale_after_generation_capture)

    def run_stale():
        try:
            stale.rollover_session(
                OLD,
                "stale-carry-target",
                previous_messages=previous_messages
                + [{"role": "assistant", "content": "STALE RACE TAIL"}],
                carry_over_context=True,
                platform="test",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=run_stale)
    try:
        worker.start()
        assert stale_captured.wait(5)
        winner.rollover_session(
            OLD,
            NEW,
            previous_messages=previous_messages,
            carry_over_context=False,
            platform="test",
        )
        release_stale.set()
        worker.join(5)
        assert not worker.is_alive()
        assert failures == []
        state = _canonical_snapshot(db_path)
        assert state["active"][1:] == (NEW, 0)
        assert state["items"] == []
        assert sum(row[3] == "STALE RACE TAIL" for row in state["messages"]) == 1
    finally:
        release_stale.set()
        worker.join(5)
        winner.shutdown()
        stale.shutdown()


def test_off_current_session_end_begin_immediate_uses_50ms_busy_bound(tmp_path):
    db_path = tmp_path / "rollover-late-end-busy-bound.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    rollover = LCMEngine(config=_config(db_path))
    late_end = LCMEngine(config=_config(db_path))
    rollover.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    rollover.rollover_session(OLD, NEW, previous_messages=previous_messages, platform="test")
    late_end.on_session_start(NEW, conversation_id=CONVERSATION, platform="test")
    late_end._frontier._conn.execute("PRAGMA busy_timeout=700")
    locker = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    locker.execute("PRAGMA journal_mode=WAL")
    locker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            late_end.on_session_end(
                OLD,
                previous_messages + [{"role": "assistant", "content": "LOCKED LATE TAIL"}],
            )
        elapsed = time.monotonic() - started
        assert elapsed < 0.3
        assert late_end._frontier._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 700
    finally:
        locker.execute("ROLLBACK")
        locker.close()
        rollover.shutdown()
        late_end.shutdown()


def test_simultaneous_late_end_and_competing_rollover_is_exact_or_bounded_locked(tmp_path):
    db_path = tmp_path / "rollover-late-end-versus-rollover.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    winner = LCMEngine(config=_config(db_path))
    stale = LCMEngine(config=_config(db_path))
    winner.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    stale.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    late_content = "SIMULTANEOUS LATE END"
    rollover_content = "SIMULTANEOUS ROLLOVER SUFFIX"
    failures: list[BaseException] = []
    start = threading.Barrier(2)
    try:
        winner.rollover_session(
            OLD,
            NEW,
            previous_messages=previous_messages,
            carry_over_context=True,
            platform="test",
        )
        rolled = _canonical_snapshot(db_path)

        def late_end() -> None:
            try:
                start.wait(timeout=5)
                winner.on_session_end(
                    OLD,
                    previous_messages + [{"role": "assistant", "content": late_content}],
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        def competing_rollover() -> None:
            try:
                start.wait(timeout=5)
                stale.rollover_session(
                    OLD,
                    "rejected-competing-target",
                    previous_messages=previous_messages
                    + [{"role": "assistant", "content": rollover_content}],
                    carry_over_context=True,
                    platform="test",
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        workers = [threading.Thread(target=late_end), threading.Thread(target=competing_rollover)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        assert all(not worker.is_alive() for worker in workers)
        assert all(
            isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()
            for exc in failures
        )
        assert len(failures) <= 1

        state = _canonical_snapshot(db_path)
        expected_advance = 1 if failures else 2
        assert state["active"][:2] == (rolled["active"][0] + expected_advance, NEW)
        assert state["lifecycle"][:2] == (NEW, OLD)
        assert any(item[:2] == ("node", retained_id) for item in state["items"])
        expected = {TAIL, rollover_content}
        if not failures:
            expected.add(late_content)
        _assert_raw_frontier_coverage(state, expected)
        for content in expected:
            assert sum(row[3] == content for row in state["messages"]) == 1
        if failures:
            assert all(row[3] != late_content for row in state["messages"])
    finally:
        winner.shutdown()
        stale.shutdown()


def test_late_end_fails_closed_after_old_loses_finalized_ownership(tmp_path):
    db_path = tmp_path / "rollover-late-end-ownership-lost.db"
    previous_messages, _retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    rejected = "REJECTED STALE FINALIZED OWNER"
    try:
        engine.rollover_session(OLD, NEW, previous_messages=previous_messages, platform="test")
        engine.rollover_session(NEW, "newest-owner", previous_messages=[], platform="test")
        resolved = engine._resolve_active_frontier_for_assembly()
        assert resolved is not None
        before = _canonical_snapshot(db_path)

        with pytest.raises(RuntimeError, match="ownership changed"):
            engine.on_session_end(
                OLD,
                previous_messages + [{"role": "assistant", "content": rejected}],
            )

        after = _canonical_snapshot(db_path)
        assert after == before
        assert all(row[3] != rejected for row in after["messages"])
    finally:
        engine.shutdown()


LATE_END_PRECOMMIT_PHASES = (
    "after_begin",
    "after_revalidation",
    "after_tail_ingest",
    "after_frontier",
    "after_lifecycle",
)


def _crash_late_end(tmp_path: Path, db_path: Path, phase: str, messages: list[dict]):
    package_root = tmp_path / f"late-end-package-{phase}"
    package_root.mkdir(exist_ok=True)
    (package_root / "hermes_lcm").symlink_to(
        Path(__file__).resolve().parents[1], target_is_directory=True
    )
    script = """
import json
import sys
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1],
    new_session_retain_depth=2,
    async_background_compaction_worker_enabled=False,
))
engine.on_session_start(
    "transactional-rollover-new",
    conversation_id="transactional-rollover-conversation",
    platform="test",
)
engine._rollover_publish_crash_hook = sys.argv[2]
engine.on_session_end("transactional-rollover-old", json.loads(sys.argv[3]))
raise SystemExit("crash hook did not fire")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(package_root), env.get("PYTHONPATH", "")) if value
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase, json.dumps(messages)],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("phase", (*LATE_END_PRECOMMIT_PHASES, "after_commit"))
def test_late_end_crash_is_wholly_pre_extension_or_post_extension(tmp_path, phase):
    db_path = tmp_path / f"rollover-late-end-crash-{phase}.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    engine.rollover_session(OLD, NEW, previous_messages=previous_messages, platform="test")
    engine.shutdown()
    before = _canonical_snapshot(db_path)
    late_content = f"CRASH-SAFE LATE END {phase}"
    crashed = _crash_late_end(
        tmp_path,
        db_path,
        phase,
        previous_messages + [{"role": "assistant", "content": late_content}],
    )
    assert crashed.returncode == 88, (crashed.stdout, crashed.stderr)
    state = _canonical_snapshot(db_path)

    if phase in LATE_END_PRECOMMIT_PHASES:
        assert state == before
    else:
        assert state["active"][:2] == (before["active"][0] + 1, NEW)
        assert state["lifecycle"][:2] == (NEW, OLD)
        assert any(item[:2] == ("node", retained_id) for item in state["items"])
        _assert_raw_frontier_coverage(state, {TAIL, late_content})
        assert sum(row[3] == late_content for row in state["messages"]) == 1


def _start_rollover_process(
    tmp_path: Path,
    db_path: Path,
    *,
    target: str,
    ready_path: Path,
    go_path: Path,
    previous_messages: list[dict],
) -> subprocess.Popen[str]:
    package_root = tmp_path / f"race-package-{target}"
    package_root.mkdir(exist_ok=True)
    package_link = package_root / "hermes_lcm"
    if not package_link.exists():
        package_link.symlink_to(
            Path(__file__).resolve().parents[1], target_is_directory=True
        )
    script = """
import json
import sys
import time
from pathlib import Path
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1],
    new_session_retain_depth=2,
    async_background_compaction_worker_enabled=False,
))
engine.on_session_start(
    "transactional-rollover-old",
    conversation_id="transactional-rollover-conversation",
    platform="test",
)
original_prepare = engine._prepare_rollover_final_tail
def prepare_after_generation_capture(*args, **kwargs):
    prepared = original_prepare(*args, **kwargs)
    Path(sys.argv[3]).write_text("ready", encoding="utf-8")
    while not Path(sys.argv[4]).exists():
        time.sleep(0.01)
    return prepared
engine._prepare_rollover_final_tail = prepare_after_generation_capture
moved = engine.rollover_session(
    "transactional-rollover-old",
    sys.argv[2],
    previous_messages=json.loads(sys.argv[5]),
    carry_over_context=True,
    platform="test",
)
print(json.dumps({"moved": moved, "session": engine.current_session_id}))
engine.shutdown()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(package_root), env.get("PYTHONPATH", "")) if value
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(db_path),
            target,
            str(ready_path),
            str(go_path),
            json.dumps(previous_messages),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.parametrize("targets", [(NEW, NEW), ("rollover-target-a", "rollover-target-b")])
@pytest.mark.parametrize("divergent", [False, True])
def test_real_process_rollovers_reconcile_after_generation_capture(tmp_path, targets, divergent):
    """Both processes capture one base; divergent suffixes extend its winner."""
    db_path = tmp_path / f"rollover-process-race-{targets[0]}-{targets[1]}-{divergent}.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    initial_generation = int(_snapshot(db_path)["active"][0])
    histories = [
        previous_messages + (
            [{"role": "assistant", "content": f"PROCESS DIVERGENT {index}"}]
            if divergent
            else []
        )
        for index in range(2)
    ]
    go_path = tmp_path / f"go-{targets[0]}-{targets[1]}-{divergent}"
    ready_paths = [
        tmp_path / f"ready-{index}-{targets[index]}-{divergent}" for index in range(2)
    ]
    processes = [
        _start_rollover_process(
            tmp_path,
            db_path,
            target=target,
            ready_path=ready_paths[index],
            go_path=go_path,
            previous_messages=histories[index],
        )
        for index, target in enumerate(targets)
    ]
    try:
        deadline = time.monotonic() + 10
        while not all(path.exists() for path in ready_paths):
            assert time.monotonic() < deadline, "rollover subprocesses did not become ready"
            time.sleep(0.01)
        go_path.write_text("go", encoding="utf-8")
        completed = [process.communicate(timeout=20) for process in processes]
        assert [process.returncode for process in processes] == [0, 0], completed
        results = [json.loads(stdout.strip().splitlines()[-1]) for stdout, _ in completed]

        state = _snapshot(db_path)
        durable_owner = state["active"][1]
        assert durable_owner in set(targets)
        assert state["active"][0] == initial_generation + (2 if divergent else 1)
        assert state["lifecycle"][:2] == (durable_owner, OLD)
        expected_contents = {TAIL}
        if divergent:
            expected_contents.update({"PROCESS DIVERGENT 0", "PROCESS DIVERGENT 1"})
        for content in expected_contents:
            assert state["messages"].count((OLD, "assistant", content)) == 1
        assert state["nodes"] == [(retained_id, durable_owner, 2)]
        canonical = _canonical_snapshot(db_path)
        assert any(item[:2] == ("node", retained_id) for item in canonical["items"])
        _assert_raw_frontier_coverage(canonical, expected_contents)
        assert {result["session"] for result in results} == {durable_owner}
        assert sorted(result["moved"] for result in results) == [0, 1]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
