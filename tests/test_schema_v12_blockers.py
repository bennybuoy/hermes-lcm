"""Schema-v12 protected-session, rollover-head, and receipt regressions."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore


CONVERSATION = "schema-v12-conversation"
A, B, C, D = "session-a", "session-b", "session-c", "session-d"
V12_MIGRATION_PHASES = (
    "v12_after_protected_table",
    "v12_after_head_table",
    "v12_after_receipt_table",
    "v12_after_ddl",
    "v12_after_policy_protected_backfill",
    "v12_after_lifecycle_head_backfill",
    "v12_after_policy_head_backfill",
    "v12_after_lifecycle_protected_backfill",
    "v12_after_historical_protected_backfill",
    "v12_after_backfill",
    "v12_after_provenance_triggers",
    "v12_after_legacy_insert_trigger",
    "v12_after_legacy_update_trigger",
    "v12_after_triggers",
    "v12_after_migration_step",
    "v12_after_schema_version",
)


def _config(path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(path),
        async_background_compaction_worker_enabled=False,
    )


def _drop_v12(conn: sqlite3.Connection, *, version: int) -> None:
    trigger_names = [
        row[0]
        for row in conn.execute(
            """SELECT name FROM sqlite_master WHERE type='trigger'
               AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%'
                    OR name LIKE 'lcm_node_provenance%')"""
        )
    ]
    for name in trigger_names:
        conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
    for table in (
        "lcm_session_end_receipts",
        "lcm_rollover_heads",
        "lcm_protected_sessions",
        "lcm_node_provenance_sessions",
        "lcm_node_provenance",
    ):
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(
        """DELETE FROM lcm_migration_state
           WHERE step_name='v12_protected_sessions_heads_and_ingest_receipts'"""
    )
    conn.execute(
        """DELETE FROM lcm_migration_state
           WHERE step_name='v13_exact_node_provenance_and_durable_receipts'"""
    )
    if version == 10:
        conn.execute("DROP TABLE IF EXISTS lcm_rollover_policies")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)")}
        if "binding_generation" in columns:
            conn.execute("ALTER TABLE lcm_lifecycle_state DROP COLUMN binding_generation")
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name='v11_no_carry_frontier_policy'"
        )
    conn.execute("DROP TRIGGER IF EXISTS lcm_schema_version_monotonic")
    conn.execute(
        "UPDATE metadata SET value=? WHERE key='schema_version'", (str(version),)
    )
    db_bootstrap.ensure_schema_version_monotonic_guard(conn)
    if version == 11:
        db_bootstrap.ensure_rollover_policy_triggers(conn)
    conn.commit()


def _populated_legacy_fixture(path: Path, *, version: int) -> dict[str, object]:
    engine = LCMEngine(config=_config(path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest([{"role": "user", "content": "durable-a"}])
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    engine.ingest([{"role": "assistant", "content": "durable-b"}])
    engine.rollover_session(B, C, carry_over_context=False, platform="test")
    engine.shutdown()
    conn = sqlite3.connect(path)
    _drop_v12(conn, version=version)
    snapshot = {
        "lifecycle": conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall(),
        "frontiers": conn.execute("SELECT * FROM lcm_active_frontiers").fetchall(),
        "items": conn.execute("SELECT * FROM lcm_frontier_items").fetchall(),
        "messages": conn.execute(
            "SELECT store_id, session_id, conversation_id, content FROM messages ORDER BY store_id"
        ).fetchall(),
        "policies": (
            conn.execute("SELECT * FROM lcm_rollover_policies").fetchall()
            if version == 11 else []
        ),
        "triggers": conn.execute(
            """SELECT name, sql FROM sqlite_master WHERE type='trigger'
               AND name LIKE 'lcm_no_carry_%' ORDER BY name"""
        ).fetchall(),
    }
    conn.close()
    return snapshot


def test_historical_protection_survives_multiple_rollovers_and_interleaved_ids(tmp_path):
    db_path = tmp_path / "historical.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest([{"role": "user", "content": "protected a"}])
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    engine.ingest([{"role": "user", "content": "new b before late a"}])
    b_before = int(engine._store._conn.execute(
        "SELECT MAX(store_id) FROM messages WHERE session_id=?", (B,)
    ).fetchone()[0])
    late_a = engine._append_off_current_session_end_suffix(
        A,
        [{"role": "assistant", "content": "late protected a"}],
        prefix_count=0,
        source="test",
        conversation_id=CONVERSATION,
    )[0]
    b_after = engine._store.append(
        B,
        {"role": "assistant", "content": "new b after late a"},
        source="test",
        conversation_id=CONVERSATION,
    )
    assert b_before < late_a < b_after

    conn = engine._store._conn
    active = engine._frontier.get_active_frontier(CONVERSATION)
    assert active is not None
    generation = int(active["generation"]) + 1
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 1, 1)""",
        (CONVERSATION, generation, B, b_after),
    )
    conn.execute(
        """INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'message', ?, ?, ?)""",
        (CONVERSATION, generation, b_after, b_after, b_after),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="protected session"):
        conn.execute(
            """INSERT INTO lcm_frontier_items VALUES(?, ?, 1, 'message', ?, ?, ?)""",
            (CONVERSATION, generation, late_a, late_a, late_a),
        )
    conn.rollback()

    engine.rollover_session(B, C, carry_over_context=False, platform="test")
    engine.rollover_session(C, D, carry_over_context=True, platform="test")
    assert conn.execute(
        """SELECT finalized_session_id FROM lcm_protected_sessions
           WHERE conversation_id=? ORDER BY finalized_session_id""",
        (CONVERSATION,),
    ).fetchall() == [(A,), (B,)]
    assert conn.execute(
        """SELECT current_session_id, last_finalized_session_id,
                  carry_over_context, rollover_epoch
           FROM lcm_rollover_heads WHERE conversation_id=?""",
        (CONVERSATION,),
    ).fetchone() == (D, C, 1, 3)

    conn.execute(
        """UPDATE lcm_lifecycle_state SET current_session_id='stale',
                  last_finalized_session_id=NULL,
                  rollover_carry_over_context=NULL
           WHERE conversation_id=?""",
        (CONVERSATION,),
    )
    conn.commit()
    restored = engine._lifecycle.bind_session("stale", conversation_id=CONVERSATION)
    assert (restored.current_session_id, restored.last_finalized_session_id) == (D, C)
    engine.shutdown()


def test_protected_node_owner_and_raw_source_closure_are_rejected(tmp_path):
    db_path = tmp_path / "node-provenance.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    old_id = engine._store.append(
        A, {"role": "user", "content": "old"}, source="test",
        conversation_id=CONVERSATION,
    )
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    owned = engine._dag.add_node(SummaryNode(
        session_id=A, summary="protected owner", source_ids=[old_id],
        source_type="messages", created_at=1.0,
    ))
    closure = engine._dag.add_node(SummaryNode(
        session_id=B, summary="new owner protected closure", source_ids=[old_id],
        source_type="messages", created_at=2.0,
    ))
    conn = engine._store._conn
    base = int(engine._frontier.get_active_frontier(CONVERSATION)["generation"])
    for offset, node_id in enumerate((owned, closure), start=1):
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO lcm_active_frontiers VALUES(?, ?, ?, ?, '', '', 1, 1)",
            (CONVERSATION, base + offset, B, old_id),
        )
        with pytest.raises(sqlite3.IntegrityError, match="provenance"):
            conn.execute(
                "INSERT INTO lcm_frontier_items VALUES(?, ?, 0, 'node', ?, ?, ?)",
                (CONVERSATION, base + offset, node_id, old_id, old_id),
            )
        conn.rollback()
    engine.shutdown()


def test_truncated_receipt_treats_prefix_zero_duplicate_as_fresh_once_across_restart(tmp_path):
    db_path = tmp_path / "receipt.db"
    history = [
        {"role": "user", "content": "older prefix"},
        {"role": "assistant", "content": "identical"},
    ]
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest(history)
    engine.rollover_session(A, B, previous_messages=history,
                            carry_over_context=False, platform="test")
    payload = [{"role": "assistant", "content": "identical"}]
    assert len(engine._append_off_current_session_end_suffix(
        A, payload, prefix_count=0, source="test", conversation_id=CONVERSATION
    )) == 1
    engine.shutdown()

    restarted = LCMEngine(config=_config(db_path))
    assert restarted._append_off_current_session_end_suffix(
        A, payload, prefix_count=0, source="test", conversation_id=CONVERSATION
    ) == []
    assert restarted._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND content='identical'", (A,)
    ).fetchone()[0] == 2
    assert restarted._store._conn.execute(
        "SELECT COUNT(*) FROM lcm_session_end_receipts WHERE conversation_id=?",
        (CONVERSATION,),
    ).fetchone()[0] == 1
    restarted.shutdown()


def test_receipt_fingerprint_scales_to_100k_without_materializing_union():
    payload = [{"role": "assistant", "content": "same"}] * 100_000
    first = LCMEngine._session_end_receipt_fingerprint(payload, prefix_count=0)
    second = LCMEngine._session_end_receipt_fingerprint(payload, prefix_count=0)
    assert first == second
    assert first != LCMEngine._session_end_receipt_fingerprint(payload, prefix_count=1)


@pytest.mark.parametrize("source_version", [10, 11])
def test_v10_lifecycle_and_v11_policy_backfill_atomically(tmp_path, source_version):
    db_path = tmp_path / f"backfill-{source_version}.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(A, conversation_id=CONVERSATION, platform="test")
    engine.ingest([{"role": "user", "content": "backfill"}])
    engine.rollover_session(A, B, carry_over_context=False, platform="test")
    engine.shutdown()
    conn = sqlite3.connect(db_path)
    if source_version == 10:
        conn.execute("DROP TABLE lcm_rollover_policies")
    _drop_v12(conn, version=source_version)
    db_bootstrap.run_versioned_migrations(conn)
    assert db_bootstrap.get_schema_version(conn) == 13
    assert conn.execute(
        """SELECT finalized_session_id FROM lcm_protected_sessions
           WHERE conversation_id=?""", (CONVERSATION,),
    ).fetchone() == (A,)
    assert conn.execute(
        """SELECT current_session_id, last_finalized_session_id, carry_over_context
           FROM lcm_rollover_heads WHERE conversation_id=?""", (CONVERSATION,),
    ).fetchone() == (B, A, 0)
    conn.close()


@pytest.mark.parametrize("source_version", [10, 11])
@pytest.mark.parametrize("phase", V12_MIGRATION_PHASES)
def test_v12_process_kill_restores_populated_authentic_source(
    tmp_path, phase, source_version
):
    db_path = tmp_path / f"kill-v{source_version}-{phase}.db"
    snapshot = _populated_legacy_fixture(db_path, version=source_version)
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
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 88, result.stderr
    conn = sqlite3.connect(db_path)
    assert db_bootstrap.get_schema_version(conn) == source_version
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_heads'"
    ).fetchone() is None
    assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    assert conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall() == snapshot["lifecycle"]
    assert conn.execute("SELECT * FROM lcm_active_frontiers").fetchall() == snapshot["frontiers"]
    assert conn.execute("SELECT * FROM lcm_frontier_items").fetchall() == snapshot["items"]
    assert conn.execute(
        "SELECT store_id, session_id, conversation_id, content FROM messages ORDER BY store_id"
    ).fetchall() == snapshot["messages"]
    assert (
        conn.execute("SELECT * FROM lcm_rollover_policies").fetchall()
        if source_version == 11 else []
    ) == snapshot["policies"]
    assert conn.execute(
        """SELECT name, sql FROM sqlite_master WHERE type='trigger'
           AND name LIKE 'lcm_no_carry_%' ORDER BY name"""
    ).fetchall() == snapshot["triggers"]
    if source_version == 11:
        assert len(snapshot["triggers"]) == 4
    else:
        assert snapshot["triggers"] == []
    db_bootstrap._MIGRATION_CRASH_PHASE = None
    db_bootstrap.run_versioned_migrations(conn)
    assert db_bootstrap.get_schema_version(conn) == 13
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_heads'"
    ).fetchone() == (1,)
    assert conn.execute(
        """SELECT finalized_session_id FROM lcm_protected_sessions
           WHERE conversation_id=? ORDER BY finalized_session_id""",
        (CONVERSATION,),
    ).fetchall() == [(A,), (B,)]
    conn.close()
