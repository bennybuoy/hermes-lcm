"""Non-vacuous rollover transaction, crash, and final-tail regressions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
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
