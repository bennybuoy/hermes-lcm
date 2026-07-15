"""Deterministic process-crash coverage for session rollover publication."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


OLD = "rollover-crash-old"
NEW = "rollover-crash-new"
CONVERSATION = "rollover-crash-conversation"
PRECOMMIT_PHASES = (
    "after_begin",
    "after_prune",
    "after_reassign",
    "after_frontier",
    "after_lifecycle",
)
ALL_PHASES = (*PRECOMMIT_PHASES, "after_commit")


def _config(db_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        new_session_retain_depth=2,
        async_background_compaction_worker_enabled=False,
    )


def _seed(db_path: Path) -> tuple[int, int, int]:
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(
        OLD,
        conversation_id=CONVERSATION,
        platform="test",
        context_length=50_000,
    )
    try:
        store_id = engine._store.append(
            OLD,
            {"role": "user", "content": "rollover crash raw lineage"},
            source="test",
            conversation_id=CONVERSATION,
        )
        retained = SummaryNode(
            session_id=OLD,
            depth=2,
            summary="retained rollover crash summary",
            token_count=5,
            source_token_count=6,
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
        pruned_id = engine._dag.add_node(SummaryNode(
            session_id=OLD,
            depth=0,
            summary="pruned rollover crash summary",
            token_count=5,
            source_token_count=6,
            source_ids=[store_id],
            source_type="messages",
            created_at=2.0,
        ))
        return int(retained.node_id), int(pruned_id), int(store_id)
    finally:
        engine.shutdown()


def _crash_rollover(
    tmp_path: Path,
    db_path: Path,
    phase: str,
    carry_over: bool,
) -> subprocess.CompletedProcess[str]:
    package_root = tmp_path / f"package-{phase}-{int(carry_over)}"
    package_root.mkdir(exist_ok=True)
    link = package_root / "hermes_lcm"
    if not link.exists():
        link.symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
    script = """
import sys
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1],
    new_session_retain_depth=2,
    async_background_compaction_worker_enabled=False,
))
engine.on_session_start(
    "rollover-crash-old",
    conversation_id="rollover-crash-conversation",
    platform="test",
    context_length=50000,
)
engine._rollover_publish_crash_hook = sys.argv[2]
engine.rollover_session(
    "rollover-crash-old",
    "rollover-crash-new",
    previous_messages=[],
    carry_over_context=sys.argv[3] == "1",
    platform="test",
    context_length=50000,
)
raise SystemExit("rollover crash hook did not fire")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(package_root), env.get("PYTHONPATH", ""))
        if value
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase, "1" if carry_over else "0"],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _snapshot(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        active = conn.execute(
            """
            SELECT generation, session_id, source_end_store_id
            FROM lcm_active_frontiers
            WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1
            """,
            (CONVERSATION,),
        ).fetchone()
        lifecycle = conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id,
                   current_frontier_store_id
            FROM lcm_lifecycle_state WHERE conversation_id = ?
            """,
            (CONVERSATION,),
        ).fetchone()
        nodes = conn.execute(
            "SELECT node_id, session_id, depth FROM summary_nodes ORDER BY node_id"
        ).fetchall()
        items = conn.execute(
            """
            SELECT kind, ref_id FROM lcm_frontier_items
            WHERE conversation_id = ? AND generation = ? ORDER BY ordinal
            """,
            (CONVERSATION, int(active[0])),
        ).fetchall()
        return {"active": active, "lifecycle": lifecycle, "nodes": nodes, "items": items}
    finally:
        conn.close()


@pytest.mark.parametrize("carry_over", [False, True])
@pytest.mark.parametrize("phase", ALL_PHASES)
def test_rollover_crash_is_wholly_old_or_wholly_new(
    tmp_path, phase, carry_over
):
    db_path = tmp_path / f"rollover-{phase}-{int(carry_over)}.db"
    retained_id, pruned_id, store_id = _seed(db_path)
    old = _snapshot(db_path)

    crashed = _crash_rollover(tmp_path, db_path, phase, carry_over)
    assert crashed.returncode == 88, (crashed.stdout, crashed.stderr)
    state = _snapshot(db_path)

    if phase in PRECOMMIT_PHASES:
        assert state["active"] == old["active"]
        assert state["nodes"] == old["nodes"]
        assert state["items"] == old["items"]
        assert state["lifecycle"][0] in {OLD, None}
        assert state["lifecycle"][1] in {None, OLD}
    elif carry_over:
        assert state["active"][1:] == (NEW, store_id)
        assert state["lifecycle"][0] == NEW
        assert state["lifecycle"][1] == OLD
        assert state["lifecycle"][2] == store_id
        assert state["nodes"] == [(retained_id, NEW, 2)]
        assert state["items"] == [("node", retained_id)]
    else:
        assert state["active"][1:] == (NEW, 0)
        assert state["lifecycle"][0] == NEW
        assert state["lifecycle"][1] == OLD
        assert state["lifecycle"][2] == 0
        assert state["nodes"] == [(retained_id, OLD, 2)]
        assert state["items"] == []
    assert all(node_id != pruned_id for node_id, _session, _depth in state["nodes"]) or phase in PRECOMMIT_PHASES
