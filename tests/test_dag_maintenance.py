"""Issue #16: backup-first immutable-generation DAG maintenance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

import hermes_lcm.maintenance as maintenance_module

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.frontier import FrontierStore
from hermes_lcm.lcm_cli import main as cli_main
from hermes_lcm.maintenance import (
    _MAINTENANCE_MAX_FRONTIER_ITEMS,
    _frontier_rows,
    _source_inventory,
    apply_dag_maintenance,
    create_verified_backup,
    plan_dag_maintenance,
    verify_restore_proof,
)


def _fixture(tmp_path):
    path = tmp_path / "maintenance.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(path)))
    engine.on_session_start("source", conversation_id="source-conv", platform="test")
    first = engine._store.append("source", {"role": "user", "content": "first source"}, conversation_id="source-conv")
    second = engine._store.append("source", {"role": "assistant", "content": "second source"}, conversation_id="source-conv")
    leaf_ids = []
    for store_id, summary in ((first, "leaf one"), (second, "leaf two")):
        leaf_ids.append(engine._dag.add_node(SummaryNode(
            session_id="source", depth=0, summary=summary, token_count=3,
            source_token_count=5, source_ids=[store_id], source_type="messages", created_at=float(store_id),
        )))
    parent = engine._dag.add_node(SummaryNode(
        session_id="source", depth=1, summary="parent condensation", token_count=4,
        source_token_count=10, source_ids=leaf_ids, source_type="nodes", created_at=3.0,
    ))
    engine._frontier.ensure_frontier("source-conv", "source", source_end_store_id=second)
    engine._frontier.set_frontier_items("source-conv", 1, [{
        "kind": "node", "ref_id": parent, "source_start": first, "source_end": second,
    }])

    target_store = engine._store.append(
        "target", {"role": "user", "content": "target anchor"}, conversation_id="target-conv"
    )
    engine._frontier.ensure_frontier("target-conv", "target", source_end_store_id=target_store)
    engine._frontier.set_frontier_items("target-conv", 1, [{
        "kind": "message", "ref_id": target_store,
        "source_start": target_store, "source_end": target_store,
    }])
    engine.shutdown()
    return path, parent, leaf_ids


def _active(path, conversation_id):
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT generation, session_id FROM lcm_active_frontiers WHERE conversation_id=? ORDER BY generation DESC LIMIT 1",
        (conversation_id,),
    ).fetchone()
    items = conn.execute(
        "SELECT kind, ref_id, source_start, source_end FROM lcm_frontier_items WHERE conversation_id=? AND generation=? ORDER BY ordinal",
        (conversation_id, row[0]),
    ).fetchall()
    conn.close()
    return row, items


def _maintenance_state(path):
    conn = sqlite3.connect(path)
    try:
        return {
            "messages": conn.execute(
                """SELECT store_id, session_id, conversation_id, role, content,
                          COALESCE(tool_calls, '') FROM messages ORDER BY store_id"""
            ).fetchall(),
            "nodes": conn.execute(
                """SELECT node_id, session_id, depth, summary, source_ids, source_type
                   FROM summary_nodes ORDER BY node_id"""
            ).fetchall(),
            "frontiers": conn.execute(
                """SELECT conversation_id, generation, session_id, source_end_store_id
                   FROM lcm_active_frontiers ORDER BY conversation_id, generation"""
            ).fetchall(),
            "items": conn.execute(
                """SELECT conversation_id, generation, ordinal, kind, ref_id,
                          source_start, source_end FROM lcm_frontier_items
                   ORDER BY conversation_id, generation, ordinal"""
            ).fetchall(),
            "batches": conn.execute(
                "SELECT batch_id, conversation_id, state FROM lcm_prepared_batches ORDER BY batch_id"
            ).fetchall(),
            "lifecycle": conn.execute(
                """SELECT conversation_id, current_session_id, current_frontier_store_id
                   FROM lcm_lifecycle_state ORDER BY conversation_id"""
            ).fetchall(),
        }
    finally:
        conn.close()


def test_rewrite_requires_confirmation_and_publishes_new_generation(tmp_path):
    path, parent, _ = _fixture(tmp_path)
    plan = plan_dag_maintenance(
        path, operation="rewrite-subtree", conversation_id="source-conv",
        node_id=parent, rewrites={parent: "rewritten parent with evidence"},
    )
    assert plan["dry_run"] is True
    assert plan["new_generation"] == 2
    assert plan["confirmation"] == "APPLY rewrite-subtree"
    assert not (tmp_path / "lcm-maintenance-backups").exists()
    with pytest.raises(ValueError, match="confirmation"):
        apply_dag_maintenance(
            path, operation="rewrite-subtree", conversation_id="source-conv",
            node_id=parent, rewrites={parent: "rewritten"}, confirmation="yes",
        )

    result = apply_dag_maintenance(
        path, operation="rewrite-subtree", conversation_id="source-conv",
        node_id=parent, rewrites={parent: "rewritten parent with evidence"},
        confirmation="APPLY rewrite-subtree",
    )
    row, items = _active(path, "source-conv")
    assert row[0] == 2
    assert items[0][1] != parent
    assert result["backup"]["verified"] is True
    assert result["restore_proof"]["restorable"] is True
    assert result["audit_path"].endswith("lcm-maintenance-audit.jsonl")
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT summary FROM summary_nodes WHERE node_id=?", (parent,)).fetchone()[0] == "parent condensation"
    assert conn.execute("SELECT summary FROM summary_nodes WHERE node_id=?", (items[0][1],)).fetchone()[0] == "rewritten parent with evidence"
    conn.close()


def test_dissolve_restores_children_to_new_ordered_frontier(tmp_path):
    path, parent, leaf_ids = _fixture(tmp_path)
    result = apply_dag_maintenance(
        path, operation="dissolve", conversation_id="source-conv", node_id=parent,
        confirmation="APPLY dissolve",
    )
    row, items = _active(path, "source-conv")
    assert row[0] == 2
    assert [item[1] for item in items] == leaf_ids
    assert [item[2:] for item in items] == [(1, 1), (2, 2)]
    assert result["created_node_ids"] == []


def test_copy_subtree_remaps_messages_nodes_and_publishes_target(tmp_path):
    path, parent, _ = _fixture(tmp_path)
    plan = plan_dag_maintenance(
        path, operation="copy-subtree", conversation_id="source-conv", node_id=parent,
        target_session_id="target", target_conversation_id="target-conv",
    )
    assert plan["rows_added"] == 5  # three nodes and two raw-message rows
    assert plan["node_rows_added"] == 3
    assert plan["message_rows_added"] == 2
    assert plan["frontier_generation_rows_added"] == 1
    assert plan["frontier_item_rows_added"] == 2
    assert plan["total_database_rows_added"] == 8
    assert plan["affected_node_ids"] == sorted(plan["affected_node_ids"])
    result = apply_dag_maintenance(
        path, operation="copy-subtree", conversation_id="source-conv", node_id=parent,
        target_session_id="target", target_conversation_id="target-conv",
        confirmation="APPLY copy-subtree",
    )
    row, items = _active(path, "target-conv")
    assert row == (2, "target")
    copied_root = result["created_node_ids"][-1]
    assert any(item[0] == "node" and item[1] == copied_root for item in items)
    conn = sqlite3.connect(path)
    copied = conn.execute("SELECT session_id, source_ids, source_type FROM summary_nodes WHERE node_id=?", (copied_root,)).fetchone()
    assert copied[0] == "target"
    for child_id in json.loads(copied[1]):
        child = conn.execute("SELECT session_id, source_ids FROM summary_nodes WHERE node_id=?", (child_id,)).fetchone()
        assert child[0] == "target"
        for store_id in json.loads(child[1]):
            assert conn.execute("SELECT session_id, conversation_id FROM messages WHERE store_id=?", (store_id,)).fetchone() == ("target", "target-conv")
    conn.close()

    with pytest.raises(ValueError, match="already active"):
        plan_dag_maintenance(
            path, operation="copy-subtree", conversation_id="source-conv", node_id=parent,
            target_session_id="target", target_conversation_id="target-conv",
        )


def test_backup_restore_proof_and_symlink_refusal(tmp_path):
    path, _, _ = _fixture(tmp_path)
    backup = create_verified_backup(path)
    assert verify_restore_proof(backup["backup_path"])["restorable"] is True
    link = tmp_path / "linked.db"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        create_verified_backup(link)
    assert os.stat(backup["backup_path"]).st_mode & 0o777 == 0o600


def test_maintenance_backup_directory_symlink_is_rejected(tmp_path):
    path, _, _ = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "lcm-maintenance-backups").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        create_verified_backup(path)
    assert list(outside.iterdir()) == []


def test_maintenance_audit_symlink_is_rejected_before_mutation(tmp_path):
    path, parent, _ = _fixture(tmp_path)
    redirected = tmp_path / "redirected-audit"
    redirected.write_text("do not append\n", encoding="utf-8")
    (tmp_path / "lcm-maintenance-audit.jsonl").symlink_to(redirected)

    with pytest.raises(ValueError, match="audit.*symlink"):
        apply_dag_maintenance(
            path,
            operation="dissolve",
            conversation_id="source-conv",
            node_id=parent,
            confirmation="APPLY dissolve",
        )
    assert redirected.read_text(encoding="utf-8") == "do not append\n"
    assert _active(path, "source-conv")[0][0] == 1


def test_operator_tui_once_is_functional_and_read_only(tmp_path, capsys):
    path, _, _ = _fixture(tmp_path)
    before = path.stat().st_mtime_ns
    assert cli_main(["--database", str(path), "tui", "--once"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_only"] is True
    assert "source" in payload["screen"]
    assert "source-conv" in payload["screen"]
    assert path.stat().st_mtime_ns == before


def test_maintenance_atomically_advances_lifecycle_and_settles_base_batch(tmp_path):
    path, parent, _ = _fixture(tmp_path)
    engine = LCMEngine(config=LCMConfig(database_path=str(path)))
    engine.on_session_start("source", conversation_id="source-conv", platform="test")
    batch_id, _ = engine._frontier.create_batch_cas(
        conversation_id="source-conv",
        session_id="source",
        base_generation=1,
        source_end_store_id=2,
        source_identity_hash="prepared-before-maintenance",
        source_ids=[1, 2],
        policy_fingerprint="",
        route_fingerprint="",
    )
    engine.shutdown()
    assert batch_id > 0

    apply_dag_maintenance(
        path,
        operation="dissolve",
        conversation_id="source-conv",
        node_id=parent,
        confirmation="APPLY dissolve",
    )
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT state FROM lcm_prepared_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()[0] == "superseded"
        frontier_end = conn.execute(
            """SELECT source_end_store_id FROM lcm_active_frontiers
               WHERE conversation_id='source-conv' ORDER BY generation DESC LIMIT 1"""
        ).fetchone()[0]
        lifecycle = conn.execute(
            """SELECT current_session_id, current_frontier_store_id
               FROM lcm_lifecycle_state WHERE conversation_id='source-conv'"""
        ).fetchone()
        assert lifecycle == ("source", frontier_end)
    finally:
        conn.close()


def test_concurrent_winner_during_locked_snapshot_serializes_and_rejects(tmp_path):
    path, parent, leaf_ids = _fixture(tmp_path)
    attempting = threading.Event()
    acquired = threading.Event()
    finished = threading.Event()
    outcome: dict[str, str] = {}
    competitors: list[threading.Thread] = []
    competitor_store = FrontierStore(str(path))

    def competing_publisher():
        try:
            attempting.set()
            with competitor_store.publication_transaction() as conn:
                acquired.set()
                generation = competitor_store.publish_generation_state_no_commit(
                    conn,
                    conversation_id="source-conv",
                    session_id="source",
                    source_end_store_id=2,
                    policy_fingerprint="",
                    route_fingerprint="",
                    base_generation=1,
                    items=[{
                        "kind": "node",
                        "ref_id": parent,
                        "source_start": 1,
                        "source_end": 2,
                    }],
                    batch_reason="competing_test_publication",
                )
                outcome["generation"] = str(generation)
                outcome["state"] = "published" if generation else "rejected"
        finally:
            finished.set()

    def start_during_locked_snapshot(phase):
        if phase != "after_snapshot_locked":
            return
        thread = threading.Thread(target=competing_publisher, name="maintenance-competitor")
        competitors.append(thread)
        thread.start()
        assert attempting.wait(timeout=2)
        assert not acquired.wait(timeout=0.05)
        probe = sqlite3.connect(path, timeout=0.0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                probe.execute("BEGIN IMMEDIATE")
        finally:
            probe.close()

    result = apply_dag_maintenance(
        path,
        operation="dissolve",
        conversation_id="source-conv",
        node_id=parent,
        confirmation="APPLY dissolve",
        snapshot_hook=start_during_locked_snapshot,
    )
    assert result["new_generation"] == 2
    assert competitors
    competitors[0].join(timeout=5)
    assert not competitors[0].is_alive()
    assert acquired.is_set() and finished.is_set()
    assert outcome["state"] == "rejected"
    assert outcome["generation"] == "0"
    active, items = _active(path, "source-conv")
    assert active[0] == 2
    assert {item[1] for item in items if item[0] == "node"} == set(leaf_ids)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            """SELECT MAX(generation) FROM lcm_active_frontiers
               WHERE conversation_id='source-conv'"""
        ).fetchone()[0] == 2
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
        competitor_store.close()


def test_maintenance_source_inventory_batches_message_validation_and_rejects_fanout(
    tmp_path, monkeypatch
):
    path, parent, _ = _fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    statements = []
    conn.set_trace_callback(statements.append)
    nodes, messages = _source_inventory(conn, parent)
    assert len(nodes) == 3 and messages == {1, 2}
    message_selects = [
        statement for statement in statements
        if statement.lstrip().upper().startswith("SELECT STORE_ID, SESSION_ID FROM MESSAGES")
    ]
    assert len(message_selects) == 1

    raw = "[" + ",".join(str(index) for index in range(7_000)) + "]"
    conn.execute("UPDATE summary_nodes SET source_ids=? WHERE node_id=?", (raw, parent))
    conn.commit()
    import hermes_lcm.maintenance as maintenance_module
    original_loads = maintenance_module.json.loads

    def guarded_loads(value, *args, **kwargs):
        if value == raw:
            raise AssertionError("oversized source_ids JSON was materialized")
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(maintenance_module.json, "loads", guarded_loads)
    with pytest.raises(ValueError, match="hard cap"):
        _source_inventory(conn, parent)
    conn.close()


def test_maintenance_frontier_rows_are_sql_limited_before_allocation(tmp_path):
    path, _parent, _ = _fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "DELETE FROM lcm_frontier_items WHERE conversation_id='source-conv' AND generation=1"
    )
    conn.executemany(
        """INSERT INTO lcm_frontier_items
           (conversation_id, generation, ordinal, kind, ref_id, source_start, source_end)
           VALUES ('source-conv', 1, ?, 'message', ?, ?, ?)""",
        [
            (index, index + 1, index + 1, index + 1)
            for index in range(_MAINTENANCE_MAX_FRONTIER_ITEMS + 1)
        ],
    )
    conn.commit()
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    with pytest.raises(ValueError, match="frontier-item bound"):
        _frontier_rows(conn, "source-conv", 1)
    assert any(
        f"LIMIT {_MAINTENANCE_MAX_FRONTIER_ITEMS + 1}" in statement
        for statement in statements
    )
    conn.close()


def test_copy_subtree_shared_byte_budget_aborts_before_retaining_source_rows(
    tmp_path, monkeypatch
):
    path, parent, _ = _fixture(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE messages SET content=? WHERE session_id='source' AND store_id=1",
        ("nested payload " + ("x" * 20_000),),
    )
    before = {
        "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "nodes": conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
        "frontiers": conn.execute("SELECT COUNT(*) FROM lcm_active_frontiers").fetchone()[0],
        "items": conn.execute("SELECT COUNT(*) FROM lcm_frontier_items").fetchone()[0],
    }
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        maintenance_module, "_MAINTENANCE_COPY_MAX_BYTES", 1_024, raising=False
    )
    with pytest.raises(ValueError, match="copy.*byte.*budget"):
        apply_dag_maintenance(
            path,
            operation="copy-subtree",
            conversation_id="source-conv",
            node_id=parent,
            target_session_id="target",
            target_conversation_id="target-conv",
            confirmation="APPLY copy-subtree",
        )
    conn = sqlite3.connect(path)
    after = {
        "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "nodes": conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
        "frontiers": conn.execute("SELECT COUNT(*) FROM lcm_active_frontiers").fetchone()[0],
        "items": conn.execute("SELECT COUNT(*) FROM lcm_frontier_items").fetchone()[0],
    }
    conn.close()
    assert after == before


def test_copy_subtree_shared_token_budget_aborts_atomically(tmp_path, monkeypatch):
    path, parent, _ = _fixture(tmp_path)
    monkeypatch.setattr(maintenance_module, "_MAINTENANCE_COPY_MAX_TOKENS", 1)
    with pytest.raises(ValueError, match="copy.*token.*budget"):
        apply_dag_maintenance(
            path,
            operation="copy-subtree",
            conversation_id="source-conv",
            node_id=parent,
            target_session_id="target",
            target_conversation_id="target-conv",
            confirmation="APPLY copy-subtree",
        )
    assert _active(path, "target-conv")[0][0] == 1
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='target'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM summary_nodes WHERE session_id='target'").fetchone()[0] == 0
    conn.close()


def test_copy_subtree_nested_tool_payload_depth_aborts_atomically(
    tmp_path, monkeypatch
):
    path, parent, _ = _fixture(tmp_path)
    nested: object = "leaf"
    for _ in range(8):
        nested = {"content": nested}
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE messages SET tool_calls=? WHERE session_id='source' AND store_id=1",
        (json.dumps(nested),),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(maintenance_module, "_MAINTENANCE_COPY_MAX_NESTED_DEPTH", 3)
    with pytest.raises(ValueError, match="nested-depth"):
        apply_dag_maintenance(
            path,
            operation="copy-subtree",
            conversation_id="source-conv",
            node_id=parent,
            target_session_id="target",
            target_conversation_id="target-conv",
            confirmation="APPLY copy-subtree",
        )
    assert _active(path, "target-conv")[0][0] == 1


@pytest.mark.parametrize("phase", ["after_frontier", "after_commit"])
def test_maintenance_process_death_restarts_at_wholly_old_or_new_state(
    tmp_path, phase
):
    path, parent, _ = _fixture(tmp_path)
    engine = LCMEngine(config=LCMConfig(database_path=str(path)))
    target = engine._frontier.get_active_frontier("target-conv")
    assert target is not None
    batch_id, _ = engine._frontier.create_batch_cas(
        conversation_id="target-conv",
        session_id="target",
        base_generation=int(target["generation"]),
        source_end_store_id=int(target["source_end_store_id"]),
        source_identity_hash="maintenance-crash-batch",
        source_ids=[int(target["source_end_store_id"])],
        policy_fingerprint="",
        route_fingerprint="",
    )
    assert batch_id > 0
    engine.shutdown()
    old_state = _maintenance_state(path)
    script = r'''
import os, sys
from hermes_lcm.maintenance import apply_dag_maintenance
def crash(observed):
    if observed == sys.argv[3]:
        os._exit(91)
apply_dag_maintenance(
    sys.argv[1], operation='copy-subtree', conversation_id='source-conv',
    node_id=int(sys.argv[2]), confirmation='APPLY copy-subtree',
    target_session_id='target', target_conversation_id='target-conv',
    publication_phase_hook=crash,
)
raise SystemExit('crash hook did not fire')
'''
    package_root = tmp_path / f"maintenance-package-{phase}"
    package_root.mkdir()
    (package_root / "hermes_lcm").symlink_to(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(package_root), "/home/ben/hermes-agent-gil-pr"]
    )
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(path), str(parent), phase],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert crashed.returncode == 91, (crashed.stdout, crashed.stderr)
    backups = list((tmp_path / "lcm-maintenance-backups").glob("*.sqlite3"))
    assert len(backups) == 1
    assert _maintenance_state(backups[0]) == old_state

    recovered = _maintenance_state(path)
    if phase == "after_frontier":
        assert recovered == old_state
    else:
        assert len(recovered["messages"]) == len(old_state["messages"]) + 2
        assert len(recovered["nodes"]) == len(old_state["nodes"]) + 3
        assert len(recovered["frontiers"]) == len(old_state["frontiers"]) + 1
        target_frontier = [
            row for row in recovered["frontiers"] if row[0] == "target-conv"
        ][-1]
        assert target_frontier[1] == 2
        target_items = [
            row for row in recovered["items"]
            if row[0] == "target-conv" and row[1] == 2
        ]
        assert {row[3] for row in target_items} == {"message", "node"}
        assert recovered["batches"] == [(batch_id, "target-conv", "superseded")]
        assert ("target-conv", "target", target_frontier[3]) in recovered["lifecycle"]
        conn = sqlite3.connect(path)
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.close()
