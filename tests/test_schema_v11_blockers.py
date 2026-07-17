"""Schema-v11 no-carry policy, binding CAS, and migration crash blockers."""

from __future__ import annotations

import os
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore


CONVERSATION = "schema-v11-conversation"
OLD = "schema-v11-old"
NEW = "schema-v11-new"
MIGRATION_PHASES = (
    "v11_after_column",
    "v11_after_table",
    "v11_after_trigger",
    "v11_after_migration_step",
    "v11_after_schema_version",
)


def _config(db_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        async_background_compaction_worker_enabled=False,
    )


def _publish_no_carry(db_path: Path) -> tuple[LCMEngine, int]:
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    history = [{"role": "user", "content": "frozen old source"}]
    engine.ingest(history)
    engine.rollover_session(
        OLD,
        NEW,
        previous_messages=history,
        carry_over_context=False,
        platform="test",
    )
    row = engine._store._conn.execute(
        """SELECT finalized_cutoff_store_id FROM lcm_rollover_policies
           WHERE conversation_id = ?""",
        (CONVERSATION,),
    ).fetchone()
    assert row is not None
    return engine, int(row[0])


def test_v11_database_policy_blocks_legacy_frontier_sql_but_not_raw_storage(tmp_path):
    db_path = tmp_path / "legacy-publisher.db"
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    history = [{"role": "user", "content": "frozen old source"}]
    engine.ingest(history)
    legacy_sql = r"""
import json
import sqlite3
import sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute('PRAGMA busy_timeout=2000')
print('READY', flush=True)
sys.stdin.readline()
conn.execute('''INSERT INTO messages(
    session_id, source, role, content, timestamp,
    token_estimate, pinned, conversation_id
) VALUES(?, 'legacy-v9', 'assistant', 'late raw survives', 1, 1, 0, ?)''',
    (sys.argv[2], sys.argv[3]))
late_store_id = int(conn.execute('SELECT last_insert_rowid()').fetchone()[0])
blocked = False
try:
    conn.execute('BEGIN IMMEDIATE')
    generation = int(conn.execute('''SELECT MAX(generation)
        FROM lcm_active_frontiers WHERE conversation_id = ?''',
        (sys.argv[3],)).fetchone()[0])
    conn.execute('''INSERT INTO lcm_active_frontiers(
        conversation_id, generation, session_id, source_end_store_id,
        policy_fingerprint, route_fingerprint, created_at, updated_at
    ) VALUES(?, ?, ?, ?, '', '', 1, 1)''',
        (sys.argv[3], generation + 1, sys.argv[2], late_store_id))
    conn.execute('''INSERT INTO lcm_frontier_items(
        conversation_id, generation, ordinal, kind, ref_id,
        source_start, source_end
    ) VALUES(?, ?, 0, 'message', ?, ?, ?)''',
        (sys.argv[3], generation + 1, late_store_id, late_store_id, late_store_id))
    conn.execute('COMMIT')
except sqlite3.IntegrityError as exc:
    blocked = 'no-carry' in str(exc)
    conn.execute('ROLLBACK')
print(json.dumps({'late_store_id': late_store_id, 'blocked': blocked}), flush=True)
conn.close()
"""
    old = subprocess.Popen(
        [sys.executable, "-c", legacy_sql, str(db_path), OLD, CONVERSATION],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert old.stdout is not None
        assert old.stdout.readline().strip() == "READY"
        engine.rollover_session(
            OLD,
            NEW,
            previous_messages=history,
            carry_over_context=False,
            platform="test",
        )
        cutoff = int(engine._store._conn.execute(
            """SELECT finalized_cutoff_store_id FROM lcm_rollover_policies
               WHERE conversation_id = ?""",
            (CONVERSATION,),
        ).fetchone()[0])
        assert old.stdin is not None
        old.stdin.write("publish\n")
        old.stdin.flush()
        stdout, stderr = old.communicate(timeout=10)
        assert old.returncode == 0, stderr
        outcome = json.loads(stdout.strip().splitlines()[-1])
        assert outcome["blocked"] is True
        late_store_id = int(outcome["late_store_id"])
        assert late_store_id > cutoff

        current = engine._frontier.get_active_frontier(CONVERSATION)
        assert current is not None
        assert current["session_id"] == NEW
        assert current["source_end_store_id"] == 0
        assert engine._store._conn.execute(
            "SELECT content FROM messages WHERE store_id = ?", (late_store_id,)
        ).fetchone() == ("late raw survives",)
    finally:
        if old.poll() is None:
            old.kill()
            old.wait(timeout=5)
        engine.shutdown()


def test_v11_policy_rejects_covered_new_session_item_and_survives_lifecycle_rewrite(tmp_path):
    db_path = tmp_path / "policy-survives.db"
    engine, cutoff = _publish_no_carry(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            """UPDATE lcm_lifecycle_state
               SET current_session_id = 'stale-owner',
                   rollover_carry_over_context = NULL
               WHERE conversation_id = ?""",
            (CONVERSATION,),
        )
        assert conn.execute(
            """SELECT finalized_session_id, current_session_id,
                      finalized_cutoff_store_id, carry_over_context
               FROM lcm_rollover_policies WHERE conversation_id = ?""",
            (CONVERSATION,),
        ).fetchone() == (OLD, NEW, cutoff, 0)

        generation = int(conn.execute(
            """SELECT MAX(generation) FROM lcm_active_frontiers
               WHERE conversation_id = ?""",
            (CONVERSATION,),
        ).fetchone()[0])
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO lcm_active_frontiers(
                   conversation_id, generation, session_id,
                   source_end_store_id, policy_fingerprint,
                   route_fingerprint, created_at, updated_at
               ) VALUES(?, ?, ?, ?, '', '', 1, 1)""",
            (CONVERSATION, generation + 1, NEW, cutoff + 1),
        )
        with pytest.raises(sqlite3.IntegrityError, match="no-carry"):
            conn.execute(
                """INSERT INTO lcm_frontier_items(
                       conversation_id, generation, ordinal, kind,
                       ref_id, source_start, source_end
                   ) VALUES(?, ?, 0, 'message', ?, ?, ?)""",
                (CONVERSATION, generation + 1, cutoff, cutoff, cutoff),
            )
        conn.execute("ROLLBACK")
        conn.execute(
            "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (CONVERSATION,),
        )
        restored_store = LifecycleStateStore(db_path)
        try:
            restored = restored_store.bind_session(
                "post-delete-stale-owner", conversation_id=CONVERSATION
            )
            assert restored.current_session_id == NEW
            assert restored.last_finalized_session_id == OLD
            assert restored.rollover_carry_over_context is False
        finally:
            restored_store.close()
    finally:
        conn.close()
        engine.shutdown()


def test_bind_session_stale_owner_generation_cas_cannot_replace_rollover_winner(tmp_path, monkeypatch):
    db_path = tmp_path / "stale-bind.db"
    engine = LCMEngine(config=_config(db_path))
    stale = LifecycleStateStore(db_path)
    engine.on_session_start(OLD, conversation_id=CONVERSATION, platform="test")
    original = stale.get_by_conversation
    observed = []

    def win_after_stale_read(conversation_id):
        state = original(conversation_id)
        if not observed:
            observed.append(state.current_session_id)
            engine.rollover_session(
                OLD,
                NEW,
                previous_messages=[],
                carry_over_context=False,
                platform="test",
            )
        return state

    monkeypatch.setattr(stale, "get_by_conversation", win_after_stale_read)
    try:
        returned = stale.bind_session("stale-target", conversation_id=CONVERSATION)
    finally:
        monkeypatch.setattr(stale, "get_by_conversation", original)

    try:
        durable = stale.get_by_conversation(CONVERSATION)
        assert observed == [OLD]
        assert returned.current_session_id == NEW
        assert durable.current_session_id == NEW
        assert durable.rollover_carry_over_context is False
        policy = stale._conn.execute(
            "SELECT carry_over_context FROM lcm_rollover_policies WHERE conversation_id = ?",
            (CONVERSATION,),
        ).fetchone()
        assert policy is not None and int(policy[0]) == 0
    finally:
        stale.close()
        engine.shutdown()


def _make_v10_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        db_bootstrap.run_versioned_migrations(conn)
        for trigger in (
            "lcm_no_carry_frontier_insert",
            "lcm_no_carry_frontier_update",
            "lcm_no_carry_item_insert",
            "lcm_no_carry_item_update",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        for (trigger,) in conn.execute(
            """SELECT name FROM sqlite_master WHERE type='trigger'
               AND (name LIKE 'lcm_protected_%' OR name LIKE 'lcm_v12_%')"""
        ).fetchall():
            conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        for table in (
            "lcm_session_end_receipts",
            "lcm_rollover_heads",
            "lcm_protected_sessions",
        ):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute("DROP TABLE IF EXISTS lcm_rollover_policies")
        conn.execute(
            "ALTER TABLE lcm_lifecycle_state DROP COLUMN binding_generation"
        )
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name = 'v11_no_carry_frontier_policy'"
        )
        conn.execute(
            """DELETE FROM lcm_migration_state
               WHERE step_name = 'v12_protected_sessions_heads_and_ingest_receipts'"""
        )
        conn.execute("DROP TRIGGER IF EXISTS lcm_schema_version_monotonic")
        conn.execute("UPDATE metadata SET value = '10' WHERE key = 'schema_version'")
        db_bootstrap.ensure_schema_version_monotonic_guard(conn)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("phase", MIGRATION_PHASES)
def test_v11_process_kill_migration_recovers_wholly_v10_or_v11(tmp_path, phase):
    base = tmp_path / "base-v10.db"
    db_path = tmp_path / f"killed-{phase}.db"
    _make_v10_database(base)
    shutil.copy2(base, db_path)
    script = """
import sqlite3
import sys
from hermes_lcm import db_bootstrap
db_bootstrap._MIGRATION_CRASH_PHASE = sys.argv[2]
conn = sqlite3.connect(sys.argv[1])
db_bootstrap.run_versioned_migrations(conn)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(db_path), phase],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 88, result.stderr

    interrupted = sqlite3.connect(db_path)
    try:
        assert db_bootstrap.get_schema_version(interrupted) == 10
        assert interrupted.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_policies'"
        ).fetchone() is None
        assert "binding_generation" not in {
            row[1]
            for row in interrupted.execute("PRAGMA table_info(lcm_lifecycle_state)")
        }
        assert interrupted.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='v11_no_carry_frontier_policy'"
        ).fetchone() is None
        assert interrupted.execute("PRAGMA quick_check").fetchone() == ("ok",)

        db_bootstrap.run_versioned_migrations(interrupted)
        assert db_bootstrap.get_schema_version(interrupted) == 12
        assert interrupted.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_policies'"
        ).fetchone() == (1,)
        assert "binding_generation" in {
            row[1]
            for row in interrupted.execute("PRAGMA table_info(lcm_lifecycle_state)")
        }
        assert interrupted.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='v11_no_carry_frontier_policy'"
        ).fetchone() == (1,)
        triggers = {
            row[0]
            for row in interrupted.execute(
                """SELECT name FROM sqlite_master WHERE type='trigger'
                   AND name LIKE 'lcm_protected_frontier_%'"""
            )
        }
        assert triggers == {
            "lcm_protected_frontier_insert",
            "lcm_protected_frontier_update",
        }
        assert interrupted.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollover_heads'"
        ).fetchone() == (1,)
    finally:
        interrupted.close()
