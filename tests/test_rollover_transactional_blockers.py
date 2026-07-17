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
        assert state["items"] == [("node", retained_id)]
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
        assert state["items"] == [("node", retained_id)]
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
Path(sys.argv[3]).write_text("ready", encoding="utf-8")
while not Path(sys.argv[4]).exists():
    time.sleep(0.01)
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
def test_real_process_rollovers_publish_one_generation_and_one_owner(tmp_path, targets):
    """Same-target retries no-op; different targets converge on one winner."""
    db_path = tmp_path / f"rollover-process-race-{targets[0]}-{targets[1]}.db"
    previous_messages, retained_id, _pruned_id = _seed(db_path)
    initial_generation = int(_snapshot(db_path)["active"][0])
    go_path = tmp_path / f"go-{targets[0]}-{targets[1]}"
    ready_paths = [tmp_path / f"ready-{index}-{targets[index]}" for index in range(2)]
    processes = [
        _start_rollover_process(
            tmp_path,
            db_path,
            target=target,
            ready_path=ready_paths[index],
            go_path=go_path,
            previous_messages=previous_messages,
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
        assert state["active"][0] == initial_generation + 1
        assert state["lifecycle"][:2] == (durable_owner, OLD)
        assert state["messages"].count((OLD, "assistant", TAIL)) == 1
        assert state["nodes"] == [(retained_id, durable_owner, 2)]
        assert state["items"] == [("node", retained_id)]
        assert {result["session"] for result in results} == {durable_owner}
        assert sorted(result["moved"] for result in results) == [0, 1]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
