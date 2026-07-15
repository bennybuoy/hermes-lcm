"""Issue #16: backup-first immutable-generation DAG maintenance."""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lcm_cli import main as cli_main
from hermes_lcm.maintenance import (
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


def test_operator_tui_once_is_functional_and_read_only(tmp_path, capsys):
    path, _, _ = _fixture(tmp_path)
    before = path.stat().st_mtime_ns
    assert cli_main(["--database", str(path), "tui", "--once"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_only"] is True
    assert "source" in payload["screen"]
    assert "source-conv" in payload["screen"]
    assert path.stat().st_mtime_ns == before
