"""Schema-v13 exact-provenance, durable-receipt, and migration regressions."""

from __future__ import annotations

import sqlite3
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
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
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
    assert db_bootstrap.get_schema_version(conn) == 13
    assert conn.execute(
        "SELECT node_id FROM lcm_node_provenance WHERE node_id IN (?, ?)",
        (deep_root, oversized),
    ).fetchall() == []
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    conn.close()


@pytest.mark.parametrize("phase", V13_MIGRATION_PHASES)
def test_v13_process_kill_restores_all_populated_v12_state_and_triggers(tmp_path, phase):
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
        if table == "triggers":
            continue
        assert conn.execute(f'SELECT * FROM "{table}"').fetchall() == expected
    assert conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='trigger'
           AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%')
           ORDER BY name"""
    ).fetchall() == snapshot["triggers"]
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    db_bootstrap.run_versioned_migrations(conn)
    assert db_bootstrap.get_schema_version(conn) == 13
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_node_provenance'"
    ).fetchone() == (1,)
    conn.close()
