"""Schema-v13/v14/v15 provenance, receipt, and migration regressions."""

from __future__ import annotations

import sqlite3
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.frontier import FrontierStore
from hermes_lcm.lifecycle_state import LifecycleStateStore


CONVERSATION = "schema-v13-conversation"
A, B, C = "session-a", "session-b", "session-c"
V13_MIGRATION_PHASES = (
    "v13_after_receipt_shadow_table",
    "v13_after_receipt_copy",
    "v13_after_receipt_rebuild",
    "v13_after_node_provenance_table",
    "v13_after_proof_tables",
    "v13_after_ddl",
    "v13_after_provenance_backfill",
    "v13_after_triggers",
    "v13_after_migration_step",
    "v13_after_schema_version",
)
V14_MIGRATION_PHASES = (
    "v14_after_dependency_tables",
    "v14_after_legacy_proof_invalidation",
    "v14_after_active_provenance_backfill",
    "v14_after_provenance_backfill",
    "v14_after_triggers",
    "v14_after_migration_step",
    "v14_after_schema_version",
)
V15_MIGRATION_PHASES = (
    "v15_after_diagnostic_table",
    "v15_after_diagnostic_index",
    "v15_after_ddl",
    "v15_after_provenance_backfill",
    "v15_after_triggers",
    "v15_after_migration_step",
    "v15_after_schema_version",
)


def _config(path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(path),
        async_background_compaction_worker_enabled=False,
    )


def _drop_v13_to_populated_v12(path: Path) -> dict[str, object]:
    engine = LCMEngine(config=_config(path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest([{"role": "user", "content": "v12-node-source"}])
    source_id = int(engine._store._conn.execute(
        "SELECT MAX(store_id) FROM messages WHERE session_id=?", (A,)
    ).fetchone()[0])
    result = engine._publish_foreground_leaf(
        node=SummaryNode(
            session_id=A,
            summary="populated legacy node",
            source_ids=[source_id],
            source_type="messages",
            created_at=time.time(),
        ),
        source_end_store_id=source_id,
        covered_source_ids=[source_id],
    )
    assert result["published"] is True
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    engine._append_off_current_session_end_suffix(
        A,
        [{"role": "assistant", "content": "durable receipt"}],
        prefix_count=0,
        source="test",
        conversation_id=CONVERSATION,
    )
    engine.shutdown()

    conn = sqlite3.connect(path)
    for name in (
        "lcm_protected_item_insert",
        "lcm_protected_item_update",
        "lcm_node_provenance_delete",
        "lcm_node_provenance_lineage_update",
        "lcm_node_provenance_session_update",
        "lcm_node_dependency_update",
        "lcm_node_dependency_delete",
        "lcm_message_dependency_update",
        "lcm_message_dependency_delete",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE IF EXISTS lcm_node_provenance_message_dependencies")
    conn.execute("DROP TABLE IF EXISTS lcm_node_provenance_node_dependencies")
    conn.execute("DROP TABLE lcm_node_provenance_sessions")
    conn.execute("DROP TABLE lcm_node_provenance")
    conn.execute("DROP INDEX IF EXISTS idx_lcm_session_end_receipts_session")
    conn.execute("DROP INDEX IF EXISTS idx_lcm_session_end_receipts_created")
    conn.execute("ALTER TABLE lcm_session_end_receipts RENAME TO lcm_session_end_receipts_v13")
    db_bootstrap.ensure_rollover_v12_tables(conn)
    conn.execute(
        """INSERT INTO lcm_session_end_receipts
           SELECT conversation_id, session_id, payload_fingerprint, rollover_epoch,
                  prefix_count, retained_count, created_at
           FROM lcm_session_end_receipts_v13"""
    )
    conn.execute("DROP TABLE lcm_session_end_receipts_v13")
    conn.execute(
        """DELETE FROM lcm_migration_state
           WHERE step_name='v13_exact_node_provenance_and_durable_receipts'"""
    )
    conn.execute(
        """DELETE FROM lcm_migration_state
           WHERE step_name='v14_exact_provenance_dependencies'"""
    )
    conn.execute("DROP TRIGGER IF EXISTS lcm_schema_version_monotonic")
    conn.execute("UPDATE metadata SET value='12' WHERE key='schema_version'")
    db_bootstrap.ensure_schema_version_monotonic_guard(conn)
    db_bootstrap.ensure_rollover_v12_triggers(conn)
    conn.commit()
    snapshot = {
        table: conn.execute(f'SELECT * FROM "{table}"').fetchall()
        for table in (
            "messages",
            "summary_nodes",
            "lcm_lifecycle_state",
            "lcm_active_frontiers",
            "lcm_frontier_items",
            "lcm_rollover_policies",
            "lcm_protected_sessions",
            "lcm_rollover_heads",
            "lcm_session_end_receipts",
        )
    }
    snapshot["triggers"] = conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='trigger'
           AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%')
           ORDER BY name"""
    ).fetchall()
    snapshot["table_sql"] = conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='table'
           AND name NOT LIKE 'sqlite_%' ORDER BY name"""
    ).fetchall()
    protected_item_sql = " ".join(
        str(sql or "") for name, sql in snapshot["triggers"]
        if str(name).startswith("lcm_protected_item_")
    ).lower()
    assert "with recursive closure" in protected_item_sql
    assert "json_each" in protected_item_sql
    conn.close()
    return snapshot


def test_exact_node_proof_allows_interleaved_ids_but_rejects_actual_a_closure(tmp_path):
    db_path = tmp_path / "exact-proof.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    b_first = engine._store.append(
        B, {"role": "user", "content": "b-before"}, source="test",
        conversation_id=CONVERSATION,
    )
    a_middle = engine._store.append(
        A, {"role": "user", "content": "protected-a"}, source="test",
        conversation_id=CONVERSATION,
    )
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    b_last = engine._store.append(
        B, {"role": "assistant", "content": "b-after"}, source="test",
        conversation_id=CONVERSATION,
    )
    assert [b_first, a_middle, b_last] == [1, 2, 3]

    allowed = engine._publish_foreground_leaf(
        node=SummaryNode(
            session_id=B,
            summary="exact B closure",
            source_ids=[b_first, b_last],
            source_type="messages",
            created_at=time.time(),
        ),
        source_end_store_id=b_last,
        covered_source_ids=[b_first, b_last],
    )
    assert allowed["published"] is True

    conn = engine._store._conn
    bad_node = engine._dag.add_node(SummaryNode(
        session_id=B,
        summary="actual protected A closure",
        source_ids=[a_middle],
        source_type="messages",
        created_at=time.time(),
    ))
    db_bootstrap.materialize_node_provenance_no_commit(
        conn,
        bad_node,
        conversation_id=CONVERSATION,
    )
    conn.commit()
    generation = int(engine._frontier.get_active_frontier(CONVERSATION)["generation"]) + 1
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 1, 1)",
        (CONVERSATION, generation, B, b_last),
    )
    with pytest.raises(sqlite3.IntegrityError, match="provenance"):
        conn.execute(
            "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
            (CONVERSATION, generation, bad_node, a_middle, a_middle),
        )
    conn.rollback()
    engine.shutdown()


def test_old_or_incomplete_node_proof_fails_closed_without_recursive_trigger_sql(tmp_path):
    db_path = tmp_path / "missing-proof.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    source_id = engine._store.append(
        A, {"role": "user", "content": "source"}, source="test",
        conversation_id=CONVERSATION,
    )
    conn = engine._store._conn
    node_id = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'old publisher omitted proof', ?, 'messages', ?)""",
        (A, f"[{source_id}]", time.time()),
    ).lastrowid)
    conn.commit()
    engine._frontier.ensure_frontier(CONVERSATION, A, source_end_store_id=0)
    generation = int(engine._frontier.get_active_frontier(CONVERSATION)["generation"]) + 1
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 1, 1)",
        (CONVERSATION, generation, A, source_id),
    )
    with pytest.raises(sqlite3.IntegrityError, match="proof"):
        conn.execute(
            "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
            (CONVERSATION, generation, node_id, source_id, source_id),
        )
    conn.rollback()
    trigger_sql = " ".join(
        str(row[0] or "")
        for row in conn.execute(
            """SELECT sql FROM sqlite_master WHERE type='trigger'
               AND name LIKE 'lcm_protected_item_%'"""
        )
    ).lower()
    assert "recursive" not in trigger_sql
    assert "json_each" not in trigger_sql
    engine.shutdown()


def test_exact_proof_requires_terminal_messages_and_rejects_cycles_directly(tmp_path):
    db_path = tmp_path / "adversarial-proof-closure.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    source_id = engine._store.append(
        A, {"role": "user", "content": "terminal source"}, source="test",
        conversation_id=CONVERSATION,
    )
    conn = engine._store._conn

    def insert_node(source_type: str, source_ids: str, summary: str) -> int:
        return int(conn.execute(
            """INSERT INTO summary_nodes(
                   session_id, depth, summary, source_ids, source_type, created_at
               ) VALUES(?, 0, ?, ?, ?, 1)""",
            (A, summary, source_ids, source_type),
        ).lastrowid)

    empty_messages = insert_node("messages", "[]", "empty messages")
    empty_nodes = insert_node("nodes", "[]", "empty nodes")
    missing = insert_node("nodes", "[999999999]", "missing dependency")
    malformed = insert_node("nodes", "not-json", "malformed dependency")
    self_cycle = insert_node("nodes", "[]", "self cycle")
    conn.execute(
        "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
        (f"[{self_cycle}]", self_cycle),
    )
    cycle_left = insert_node("nodes", "[]", "cycle left")
    cycle_right = insert_node("nodes", f"[{cycle_left}]", "cycle right")
    conn.execute(
        "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
        (f"[{cycle_right}]", cycle_left),
    )

    invalid_roots = (
        empty_messages,
        empty_nodes,
        missing,
        malformed,
        self_cycle,
        cycle_left,
    )
    for root in invalid_roots:
        assert db_bootstrap.materialize_node_provenance_no_commit(
            conn, root, raise_on_failure=False
        ) is False
        proof = conn.execute(
            """SELECT proof_complete, proof_status, closure_message_count
               FROM lcm_node_provenance WHERE node_id=?""",
            (root,),
        ).fetchone()
        assert proof is not None
        assert proof[0] == 0
        assert proof[1] != "complete"
        assert proof[2] == 0

    leaf = insert_node("messages", f"[{source_id}]", "shared leaf")
    left = insert_node("nodes", f"[{leaf}]", "shared left")
    right = insert_node("nodes", f"[{leaf}]", "shared right")
    shared_root = insert_node("nodes", f"[{left}, {right}]", "shared root")
    assert db_bootstrap.materialize_node_provenance_no_commit(
        conn, shared_root
    ) is True
    assert conn.execute(
        """SELECT proof_complete, proof_status, closure_message_count
           FROM lcm_node_provenance WHERE node_id=?""",
        (shared_root,),
    ).fetchone() == (1, "complete", 1)
    conn.commit()
    engine.shutdown()


def test_materialized_cycle_is_rejected_by_frontier_trigger(tmp_path):
    db_path = tmp_path / "cycle-frontier-rejection.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    source_id = engine._store.append(
        A, {"role": "user", "content": "range anchor"}, source="test",
        conversation_id=CONVERSATION,
    )
    conn = engine._store._conn
    cycle = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'cycle', '[]', 'nodes', 1)""",
        (A,),
    ).lastrowid)
    conn.execute(
        "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
        (f"[{cycle}]", cycle),
    )
    assert db_bootstrap.materialize_node_provenance_no_commit(
        conn, cycle, raise_on_failure=False
    ) is False
    conn.commit()

    engine._frontier.ensure_frontier(CONVERSATION, A, source_end_store_id=0)
    base_generation = int(
        engine._frontier.get_active_frontier(CONVERSATION)["generation"]
    )
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 1, 1)",
        (CONVERSATION, base_generation + 1, A, source_id),
    )
    with pytest.raises(sqlite3.IntegrityError, match="proof"):
        conn.execute(
            "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
            (CONVERSATION, base_generation + 1, cycle, source_id, source_id),
        )
    conn.rollback()
    engine.shutdown()


def test_receipt_key_survives_later_rollovers_restart_and_large_ledger(tmp_path):
    db_path = tmp_path / "stable-receipt.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest([{"role": "user", "content": "initial"}])
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    payload = [{"role": "assistant", "content": "late-once"}]
    assert len(engine._append_off_current_session_end_suffix(
        A, payload, prefix_count=0, source="test", conversation_id=CONVERSATION
    )) == 1
    fingerprint = engine._session_end_receipt_fingerprint(payload, prefix_count=0)
    now = time.time()
    engine._store._conn.executemany(
        """INSERT INTO lcm_session_end_receipts(
               conversation_id, session_id, payload_fingerprint, rollover_epoch,
               prefix_count, retained_count, created_at
           ) VALUES(?, ?, ?, ?, 0, 0, ?)""",
        [
            (CONVERSATION, f"dummy-{index}", f"{index:064x}", 1, now + index)
            for index in range(1100)
        ],
    )
    engine._store._conn.commit()
    engine.rollover_session(B, C, carry_over_context=True, platform="test")
    engine.shutdown()

    restarted = LCMEngine(config=_config(db_path))
    assert restarted._append_off_current_session_end_suffix(
        A, payload, prefix_count=0, source="test", conversation_id=CONVERSATION
    ) == []
    conn = restarted._store._conn
    assert conn.execute(
        """SELECT COUNT(*) FROM lcm_session_end_receipts
           WHERE conversation_id=? AND session_id=? AND payload_fingerprint=?""",
        (CONVERSATION, A, fingerprint),
    ).fetchone() == (1,)
    assert conn.execute(
        "SELECT COUNT(*) FROM lcm_session_end_receipts WHERE conversation_id=?",
        (CONVERSATION,),
    ).fetchone()[0] >= 1101
    restarted.shutdown()


def test_bind_session_holds_writer_before_snapshot_so_rollover_cannot_be_reverted(tmp_path):
    db_path = tmp_path / "bind-barrier.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    engine._store._conn.execute(
        """UPDATE lcm_lifecycle_state SET current_session_id='stale-owner'
           WHERE conversation_id=?""",
        (CONVERSATION,),
    )
    engine._store._conn.commit()
    stale = LifecycleStateStore(db_path)
    snapshot_seen = threading.Event()
    release_bind = threading.Event()
    rollover_done = threading.Event()
    errors: list[BaseException] = []

    def hook(phase: str) -> None:
        if phase == "after_snapshot":
            snapshot_seen.set()
            assert release_bind.wait(5)

    stale._bind_session_phase_hook = hook

    def bind() -> None:
        try:
            stale.bind_session("stale-target", conversation_id=CONVERSATION)
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    def rollover() -> None:
        try:
            engine.rollover_session(
                B, C, previous_messages=[], carry_over_context=False, platform="test"
            )
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)
        finally:
            rollover_done.set()

    bind_thread = threading.Thread(target=bind)
    bind_thread.start()
    assert snapshot_seen.wait(5)
    rollover_thread = threading.Thread(target=rollover)
    rollover_thread.start()
    assert not rollover_done.wait(0.2), "rollover passed bind's writer barrier"
    release_bind.set()
    bind_thread.join(5)
    rollover_thread.join(5)
    assert not errors
    durable = stale.get_by_conversation(CONVERSATION)
    assert durable is not None
    assert durable.current_session_id == C
    assert durable.last_finalized_session_id == B
    assert durable.rollover_carry_over_context is False
    stale.close()
    engine.shutdown()


def test_legacy_null_carry_is_true_and_later_head_keeps_no_carry_history(tmp_path):
    db_path = tmp_path / "legacy-null-carry.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    conn = engine._store._conn
    protected_before = conn.execute(
        "SELECT finalized_session_id FROM lcm_protected_sessions WHERE conversation_id=?",
        (CONVERSATION,),
    ).fetchall()
    active = engine._frontier.get_active_frontier(CONVERSATION)
    next_generation = int(active["generation"]) + 1
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, 0, '', '', 2, 2)",
        (CONVERSATION, next_generation, C),
    )
    conn.execute(
        """UPDATE lcm_lifecycle_state
           SET current_session_id=?, last_finalized_session_id=?,
               rollover_carry_over_context=NULL, updated_at=2
           WHERE conversation_id=?""",
        (C, B, CONVERSATION),
    )
    conn.commit()
    assert conn.execute(
        """SELECT current_session_id, last_finalized_session_id, carry_over_context
           FROM lcm_rollover_heads WHERE conversation_id=?""",
        (CONVERSATION,),
    ).fetchone() == (C, B, 1)
    assert conn.execute(
        "SELECT finalized_session_id FROM lcm_protected_sessions WHERE conversation_id=?",
        (CONVERSATION,),
    ).fetchall() == protected_before
    engine.shutdown()


def test_migration_backfill_of_deep_and_oversized_legacy_lineage_is_bounded_and_closed(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "bounded-backfill.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    leaf = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'leaf', '[1]', 'messages', 1)""",
        (B,),
    ).lastrowid)
    child = leaf
    for depth in range(1, 80):
        child = int(conn.execute(
            """INSERT INTO summary_nodes(
                   session_id, depth, summary, source_ids, source_type, created_at
               ) VALUES(?, ?, 'deep', ?, 'nodes', 1)""",
            (B, depth, f"[{child}]"),
        ).lastrowid)
    deep_root = child
    oversized = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'oversized', ?, 'messages', 1)""",
        (B, "[" + (" " * 130_000) + "1]"),
    ).lastrowid)
    conn.commit()
    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_MIGRATION_ROOTS", 2)
    db_bootstrap.run_versioned_migrations(conn)
    assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
    assert conn.execute(
        """SELECT node_id, proof_complete FROM lcm_node_provenance
           WHERE node_id IN (?, ?) ORDER BY node_id""",
        (deep_root, oversized),
    ).fetchall() == [(deep_root, 0), (oversized, 0)]
    statuses = conn.execute(
        """SELECT proof_status FROM lcm_node_provenance
           WHERE node_id IN (?, ?) ORDER BY node_id""",
        (deep_root, oversized),
    ).fetchall()
    assert all(status and status[0] != "complete" for status in statuses)
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    conn.close()


def test_v14_backfill_prioritizes_all_active_roots_and_records_individual_failures(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "active-first-backfill.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    source_id = int(conn.execute(
        """INSERT INTO messages(
               session_id, source, conversation_id, role, content, timestamp
           ) VALUES(?, 'test', ?, 'user', 'active source', 10)""",
        (B, CONVERSATION),
    ).lastrowid)
    active_root = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'old active root', ?, 'messages', 10)""",
        (B, f"[{source_id}]"),
    ).lastrowid)
    generation = int(conn.execute(
        "SELECT MAX(generation) FROM lcm_active_frontiers WHERE conversation_id=?",
        (CONVERSATION,),
    ).fetchone()[0]) + 1
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 10, 10)",
        (CONVERSATION, generation, B, source_id),
    )
    conn.execute(
        "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
        (CONVERSATION, generation, active_root, source_id, source_id),
    )
    conn.executemany(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, ?, ?, 'messages', 20)""",
        [(B, f"newer optional node {index}", f"[{source_id}]") for index in range(300)],
    )
    oversized = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'oversized active root', ?, 'messages', 30)""",
        (B, "[" + (" " * 130_000) + str(source_id) + "]"),
    ).lastrowid)
    bad_conversation = "malformed-active-conversation"
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, 1, ?, ?, '', '', 30, 30)",
        (bad_conversation, B, source_id),
    )
    conn.execute(
        "INSERT INTO lcm_frontier_items VALUES(?, 1, 0, 'node', ?, ?, ?)",
        (bad_conversation, oversized, source_id, source_id),
    )
    conn.commit()

    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_MIGRATION_ROOTS", 2)
    db_bootstrap.run_versioned_migrations(conn)
    assert conn.execute(
        """SELECT proof_complete, proof_status FROM lcm_node_provenance
           WHERE node_id=?""",
        (active_root,),
    ).fetchone() == (1, "complete")
    bad_status = conn.execute(
        """SELECT proof_complete, proof_status FROM lcm_node_provenance
           WHERE node_id=?""",
        (oversized,),
    ).fetchone()
    assert bad_status[0] == 0
    assert "bound" in bad_status[1]

    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 40, 40)",
        (CONVERSATION, generation + 1, B, source_id),
    )
    conn.execute(
        "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
        (CONVERSATION, generation + 1, active_root, source_id, source_id),
    )
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES(?, 2, ?, ?, '', '', 40, 40)",
        (bad_conversation, B, source_id),
    )
    with pytest.raises(sqlite3.IntegrityError, match="proof"):
        conn.execute(
            "INSERT INTO lcm_frontier_items VALUES(?, 2, 0, 'node', ?, ?, ?)",
            (bad_conversation, oversized, source_id, source_id),
        )
    conn.rollback()
    conn.close()


def test_active_root_backfill_has_one_aggregate_transaction_budget(tmp_path, monkeypatch):
    db_path = tmp_path / "aggregate-active-root-budget.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    source_id = int(conn.execute(
        "SELECT MIN(store_id) FROM messages"
    ).fetchone()[0])
    root_ids = []
    for index in range(1200):
        root_id = int(conn.execute(
            """INSERT INTO summary_nodes(
                   session_id, depth, summary, source_ids, source_type, created_at
               ) VALUES(?, 0, ?, ?, 'messages', 10)""",
            (B, f"active root {index}", f"[{source_id}]"),
        ).lastrowid)
        root_ids.append(root_id)
    conn.executemany(
        "INSERT INTO lcm_active_frontiers VALUES(?, 1, ?, ?, '', '', 10, 10)",
        [(f"budget-conversation-{index}", B, source_id) for index in range(len(root_ids))],
    )
    conn.executemany(
        "INSERT INTO lcm_frontier_items VALUES(?, 1, 0, 'node', ?, ?, ?)",
        [
            (f"budget-conversation-{index}", root_id, source_id, source_id)
            for index, root_id in enumerate(root_ids)
        ],
    )
    conn.commit()

    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_MIGRATION_ROOTS", 4)
    original = db_bootstrap.materialize_node_provenance_no_commit
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(db_bootstrap, "materialize_node_provenance_no_commit", counted)
    started = time.monotonic()
    db_bootstrap.run_versioned_migrations(conn)
    elapsed = time.monotonic() - started
    assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
    assert calls <= 16, "active roots received independent per-root traversal budgets"
    assert elapsed < 5.0
    quiet_root = root_ids[-1]
    quiet_conversation = f"budget-conversation-{len(root_ids) - 1}"
    assert conn.execute(
        """SELECT proof_complete, proof_status FROM lcm_node_provenance
           WHERE node_id=?""",
        (quiet_root,),
    ).fetchone() == (0, "migration_pending")
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    conn.close()

    frontier = FrontierStore(str(db_path))
    generation = frontier.advance_frontier_generation_with_items(
        quiet_conversation,
        B,
        source_id,
        "",
        "",
        1,
        [{
            "kind": "node",
            "ref_id": quiet_root,
            "source_start": source_id,
            "source_end": source_id,
        }],
    )
    assert generation == 2
    assert frontier.conn.execute(
        """SELECT proof_complete, proof_status, proof_version
           FROM lcm_node_provenance WHERE node_id=?""",
        (quiet_root,),
    ).fetchone() == (1, "complete", db_bootstrap.NODE_PROVENANCE_PROOF_VERSION)
    frontier.close()


@pytest.mark.parametrize(
    ("budget_name", "budget_value", "status_fragment"),
    [
        ("NODE_PROVENANCE_MIGRATION_DEPENDENCIES", 1, "dependency_budget"),
        ("NODE_PROVENANCE_MIGRATION_BYTES", 1, "byte_budget"),
    ],
)
def test_active_root_backfill_enforces_aggregate_dependency_and_byte_budgets(
    tmp_path, monkeypatch, budget_name, budget_value, status_fragment
):
    db_path = tmp_path / f"aggregate-{budget_name}.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    source_id = int(conn.execute("SELECT MIN(store_id) FROM messages").fetchone()[0])
    active_root = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'budget-limited active root', ?, 'messages', 10)""",
        (B, f"[{source_id}]"),
    ).lastrowid)
    conn.execute(
        "INSERT INTO lcm_active_frontiers VALUES('budget-limit', 1, ?, ?, '', '', 10, 10)",
        (B, source_id),
    )
    conn.execute(
        "INSERT INTO lcm_frontier_items VALUES('budget-limit', 1, 0, 'node', ?, ?, ?)",
        (active_root, source_id, source_id),
    )
    conn.commit()
    monkeypatch.setattr(db_bootstrap, budget_name, budget_value)

    db_bootstrap.run_versioned_migrations(conn)
    proof = conn.execute(
        """SELECT proof_complete, proof_status
           FROM lcm_node_provenance WHERE node_id=?""",
        (active_root,),
    ).fetchone()
    assert proof is not None and proof[0] == 0
    assert status_fragment in proof[1]
    assert db_bootstrap.get_schema_version(conn) == 15
    conn.close()


def test_active_root_backfill_deadline_exhaustion_is_fail_closed_and_migrates(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "aggregate-deadline.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    monkeypatch.setattr(
        db_bootstrap, "NODE_PROVENANCE_MIGRATION_DEADLINE_SECONDS", 0.0
    )
    started = time.monotonic()

    db_bootstrap.run_versioned_migrations(conn)
    assert time.monotonic() - started < 1.0
    assert db_bootstrap.get_schema_version(conn) == 15
    assert conn.execute(
        """SELECT COUNT(*) FROM lcm_node_provenance
           WHERE proof_complete=1 AND proof_version=?""",
        (db_bootstrap.NODE_PROVENANCE_PROOF_VERSION,),
    ).fetchone() == (0,)
    conn.close()


def test_malformed_active_ref_ids_are_isolated_and_visible_to_cli(tmp_path):
    db_path = tmp_path / "malformed-active-ref-diagnostics.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    source_id = int(conn.execute(
        "SELECT MIN(store_id) FROM messages"
    ).fetchone()[0])
    valid_root = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'valid peer root', ?, 'messages', 10)""",
        (B, f"[{source_id}]"),
    ).lastrowid)
    rows = (
        ("valid-ref-conversation", valid_root),
        ("text-ref-conversation", "bad-ref"),
        ("zero-ref-conversation", 0),
        ("negative-ref-conversation", -7),
    )
    conn.executemany(
        "INSERT INTO lcm_active_frontiers VALUES(?, 1, ?, ?, '', '', 10, 10)",
        [(conversation_id, B, source_id) for conversation_id, _ref_id in rows],
    )
    conn.executemany(
        "INSERT INTO lcm_frontier_items VALUES(?, 1, 0, 'node', ?, ?, ?)",
        [
            (conversation_id, ref_id, source_id, source_id)
            for conversation_id, ref_id in rows
        ],
    )
    conn.commit()

    db_bootstrap.run_versioned_migrations(conn)
    assert conn.execute(
        """SELECT proof_complete, proof_status
           FROM lcm_node_provenance WHERE node_id=?""",
        (valid_root,),
    ).fetchone() == (1, "complete")
    diagnostics = conn.execute(
        """SELECT ref_text, reason
           FROM lcm_provenance_migration_diagnostics
           ORDER BY diagnostic_id"""
    ).fetchall()
    assert set(diagnostics) == {
        ("bad-ref", "non_integer_ref_id"),
        ("0", "non_positive_ref_id"),
        ("-7", "non_positive_ref_id"),
    }
    assert len(diagnostics) <= db_bootstrap.NODE_PROVENANCE_DIAGNOSTIC_MAX_ROWS
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "lcm_cli.py"),
            "--database",
            str(db_path),
            "doctor",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                ["/tmp/hermes-lcm-review-stub", str(Path(__file__).resolve().parents[2])]
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["provenance_migration_diagnostic_count"] == 3
    assert {item["ref_text"] for item in payload["provenance_migration_diagnostics"]} == {
        "bad-ref", "0", "-7"
    }


def test_malformed_active_ref_diagnostics_have_aggregate_row_and_byte_caps(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "bounded-malformed-active-diagnostics.db"
    _drop_v13_to_populated_v12(db_path)
    conn = sqlite3.connect(db_path)
    source_id = int(conn.execute("SELECT MIN(store_id) FROM messages").fetchone()[0])
    invalid_rows = [
        (f"diagnostic-cap-{index}", ("x" * 1000) + str(index))
        for index in range(30)
    ]
    conn.executemany(
        "INSERT INTO lcm_active_frontiers VALUES(?, 1, ?, ?, '', '', 10, 10)",
        [(conversation_id, B, source_id) for conversation_id, _ref in invalid_rows],
    )
    conn.executemany(
        "INSERT INTO lcm_frontier_items VALUES(?, 1, 0, 'node', ?, ?, ?)",
        [
            (conversation_id, ref_id, source_id, source_id)
            for conversation_id, ref_id in invalid_rows
        ],
    )
    conn.commit()
    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_DIAGNOSTIC_MAX_ROWS", 5)
    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_DIAGNOSTIC_MAX_BYTES", 200)

    db_bootstrap.run_versioned_migrations(conn)
    rows = conn.execute(
        """SELECT ref_text, ref_bytes, reason
           FROM lcm_provenance_migration_diagnostics"""
    ).fetchall()
    assert 1 <= len(rows) <= 5
    assert sum(
        len(ref_text.encode("utf-8")) + len(reason.encode("utf-8"))
        for ref_text, _ref_bytes, reason in rows
    ) <= 200
    assert all(ref_bytes > len(ref_text.encode("utf-8")) for ref_text, ref_bytes, _ in rows)
    conn.close()


def test_lazy_carried_node_publication_fails_closed_when_budget_cannot_prove(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "lazy-publication-budget-fail-closed.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    source_id = engine._store.append(
        A,
        {"role": "user", "content": "quiet carried source"},
        source="test",
        conversation_id=CONVERSATION,
    )
    conn = engine._store._conn
    root = int(conn.execute(
        """INSERT INTO summary_nodes(
               session_id, depth, summary, source_ids, source_type, created_at
           ) VALUES(?, 0, 'quiet carried root', ?, 'messages', 10)""",
        (A, f"[{source_id}]"),
    ).lastrowid)
    conn.commit()
    engine._frontier.ensure_frontier(CONVERSATION, A, source_end_store_id=0)
    base_generation = int(
        engine._frontier.get_active_frontier(CONVERSATION)["generation"]
    )
    monkeypatch.setattr(db_bootstrap, "NODE_PROVENANCE_PUBLICATION_ROOTS", 0)

    with pytest.raises(RuntimeError, match="could not prove"):
        engine._frontier.advance_frontier_generation_with_items(
            CONVERSATION,
            A,
            source_id,
            "",
            "",
            base_generation,
            [{
                "kind": "node",
                "ref_id": root,
                "source_start": source_id,
                "source_end": source_id,
            }],
        )
    assert int(engine._frontier.get_active_frontier(CONVERSATION)["generation"]) == base_generation
    assert conn.execute(
        "SELECT 1 FROM lcm_node_provenance WHERE node_id=?", (root,)
    ).fetchone() is None
    engine.shutdown()


def test_v14_dependency_guards_keep_ancestor_proof_exact_across_restart_and_interleaving(
    tmp_path
):
    db_path = tmp_path / "dependency-guards.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(B, conversation_id=CONVERSATION, platform="test")
    first = engine._store.append(B, {"role": "user", "content": "first"})
    interleaved = engine._store.append(A, {"role": "user", "content": "other"})
    last = engine._store.append(B, {"role": "assistant", "content": "last"})
    assert [first, interleaved, last] == [1, 2, 3]
    child = engine._dag.add_node(SummaryNode(
        session_id=B, summary="child", source_ids=[first, last],
        source_type="messages", created_at=1,
    ))
    ancestor = engine._dag.add_node(SummaryNode(
        session_id=B, depth=1, summary="ancestor", source_ids=[child],
        source_type="nodes", created_at=2,
    ))
    conn = engine._store._conn
    assert conn.execute(
        """SELECT dependency_node_id FROM lcm_node_provenance_node_dependencies
           WHERE root_node_id=? ORDER BY dependency_node_id""",
        (ancestor,),
    ).fetchall() == [(child,), (ancestor,)]
    assert conn.execute(
        """SELECT dependency_store_id FROM lcm_node_provenance_message_dependencies
           WHERE root_node_id=? ORDER BY dependency_store_id""",
        (ancestor,),
    ).fetchall() == [(first,), (last,)]

    for statement, params in (
        ("UPDATE messages SET session_id=? WHERE store_id=?", (A, first)),
        ("UPDATE summary_nodes SET source_ids=? WHERE node_id=?", (f"[{interleaved}]", child)),
        ("DELETE FROM summary_nodes WHERE node_id=?", (child,)),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="provenance dependency"):
            conn.execute(statement, params)
        conn.rollback()
    engine.shutdown()

    restarted = LCMEngine(config=_config(db_path))
    proof = restarted._store._conn.execute(
        "SELECT proof_complete, proof_status FROM lcm_node_provenance WHERE node_id=?",
        (ancestor,),
    ).fetchone()
    assert proof == (1, "complete")
    restarted._frontier.ensure_frontier(
        CONVERSATION, B, source_end_store_id=0
    )
    active = restarted._frontier.get_active_frontier(CONVERSATION)
    generation = restarted._frontier.advance_frontier_generation_with_items(
        CONVERSATION,
        B,
        last,
        "",
        "",
        int(active["generation"]),
        [{
            "kind": "node",
            "ref_id": ancestor,
            "source_start": first,
            "source_end": last,
        }],
    )
    assert generation == int(active["generation"]) + 1
    restarted.shutdown()


def test_authorized_message_reassignment_and_node_deletion_invalidate_ancestor_proofs(
    tmp_path
):
    db_path = tmp_path / "authorized-proof-removal.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(B, conversation_id=CONVERSATION, platform="test")
    source = engine._store.append(B, {"role": "user", "content": "source"})
    child = engine._dag.add_node(SummaryNode(
        session_id=B, summary="child", source_ids=[source],
        source_type="messages", created_at=1,
    ))
    ancestor = engine._dag.add_node(SummaryNode(
        session_id=B, depth=1, summary="ancestor", source_ids=[child],
        source_type="nodes", created_at=2,
    ))
    assert engine._store.reassign_session_messages(B, C) == 1
    assert engine._store._conn.execute(
        "SELECT 1 FROM lcm_node_provenance WHERE node_id=?", (ancestor,)
    ).fetchone() is None
    db_bootstrap.materialize_node_provenance_no_commit(engine._store._conn, ancestor)
    engine._store._conn.commit()
    assert engine._dag.delete_node(child) is True
    assert engine._store._conn.execute(
        "SELECT 1 FROM lcm_node_provenance WHERE node_id=?", (ancestor,)
    ).fetchone() is None
    engine.shutdown()


@pytest.mark.parametrize(
    "phase", V13_MIGRATION_PHASES + V14_MIGRATION_PHASES + V15_MIGRATION_PHASES
)
def test_v13_v14_v15_process_kill_restores_all_populated_v12_state_and_triggers(
    tmp_path, phase
):
    db_path = tmp_path / f"kill-{phase}.db"
    snapshot = _drop_v13_to_populated_v12(db_path)
    script = """
import sqlite3, sys
from hermes_lcm import db_bootstrap
db_bootstrap._MIGRATION_CRASH_PHASE = sys.argv[2]
conn = sqlite3.connect(sys.argv[1])
db_bootstrap.run_versioned_migrations(conn)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                ["/tmp/hermes-lcm-review-stub", str(Path(__file__).resolve().parents[2])]
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 88, result.stderr
    conn = sqlite3.connect(db_path)
    assert db_bootstrap.get_schema_version(conn) == 12
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_node_provenance'"
    ).fetchone() is None
    for table, expected in snapshot.items():
        if table in {"triggers", "table_sql"}:
            continue
        assert conn.execute(f'SELECT * FROM "{table}"').fetchall() == expected
    assert conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='table'
           AND name NOT LIKE 'sqlite_%' ORDER BY name"""
    ).fetchall() == snapshot["table_sql"]
    assert conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='trigger'
           AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%')
           ORDER BY name"""
    ).fetchall() == snapshot["triggers"]
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    db_bootstrap.run_versioned_migrations(conn)
    assert db_bootstrap.get_schema_version(conn) == db_bootstrap.SCHEMA_VERSION
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_node_provenance'"
    ).fetchone() == (1,)
    conn.close()
