"""Process-death tests for async canonical publication atomicity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


SESSION_ID = "crash-publication-session"
CONVERSATION_ID = "crash-publication-conversation"
SUMMARY_SENTINEL = "crash atomic publication sentinel"
PRECOMMIT_PHASES = (
    "after_begin",
    "after_canonical_insert",
    "after_frontier_generation",
    "after_frontier_items",
    "after_batch_promoted",
    "after_lifecycle_advanced",
)
ALL_PHASES = (*PRECOMMIT_PHASES, "after_commit")


def _messages(count: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"role": "system", "content": "system prompt"}]
    for index in range(count):
        rows.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"ordinary source message {index} " + ("x " * 20),
            }
        )
    return rows


def _config(db_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=False,
    )


def _open_engine(db_path: Path) -> LCMEngine:
    engine = LCMEngine(config=_config(db_path))
    engine.on_session_start(
        SESSION_ID,
        conversation_id=CONVERSATION_ID,
        platform="test",
        context_length=50_000,
    )
    return engine


def _prepare(db_path: Path, monkeypatch) -> tuple[int, int, int, list[dict[str, Any]]]:
    engine = _open_engine(db_path)
    messages = _messages()

    def fake_summary(initial_chunk, focus_topic=None):
        del focus_topic
        return (list(initial_chunk), 100, SUMMARY_SENTINEL, 1, 1)

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fake_summary)
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        assert batch is not None and batch.state == "ready"
        active = engine._frontier.get_active_frontier(CONVERSATION_ID)
        assert active is not None
        lifecycle = engine._lifecycle.get_by_conversation(CONVERSATION_ID)
        assert lifecycle is not None
        return (
            int(batch.batch_id),
            int(active["generation"]),
            int(lifecycle.current_frontier_store_id),
            messages,
        )
    finally:
        engine.shutdown()


def _crash_promoter(
    tmp_path: Path,
    db_path: Path,
    batch_id: int,
    messages: list[dict[str, Any]],
    phase: str,
) -> subprocess.CompletedProcess[str]:
    package_root = tmp_path / "package-root"
    package_root.mkdir(exist_ok=True)
    package_link = package_root / "hermes_lcm"
    if not package_link.exists():
        package_link.symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
    messages_path = tmp_path / "messages.json"
    messages_path.write_text(json.dumps(messages), encoding="utf-8")
    script = """
import json
import sys
from pathlib import Path
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

db_path = Path(sys.argv[1])
batch_id = int(sys.argv[2])
phase = sys.argv[3]
messages = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
config = LCMConfig(
    database_path=str(db_path),
    fresh_tail_count=2,
    leaf_chunk_tokens=20,
    context_threshold=0.10,
    async_background_compaction_enabled=True,
    async_background_compaction_worker_enabled=False,
)
engine = LCMEngine(config=config)
engine.on_session_start(
    "crash-publication-session",
    conversation_id="crash-publication-conversation",
    platform="test",
    context_length=50_000,
)
engine._async_compaction_publish_crash_hook = phase
engine.promote_prepared_compaction(batch_id, messages)
raise SystemExit("crash hook did not fire")
"""
    env = dict(os.environ)
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(package_root), inherited_pythonpath) if value
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(db_path),
            str(batch_id),
            phase,
            str(messages_path),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_process_crash_exposes_only_wholly_old_or_wholly_new_publication(
    tmp_path,
    monkeypatch,
    phase,
):
    db_path = tmp_path / f"publication-{phase}.db"
    batch_id, old_generation, old_lifecycle_frontier, messages = _prepare(
        db_path, monkeypatch
    )

    crashed = _crash_promoter(
        tmp_path,
        db_path,
        batch_id,
        messages,
        phase,
    )
    assert crashed.returncode == 86, (crashed.stdout, crashed.stderr)

    reopened = _open_engine(db_path)
    try:
        batch = reopened._frontier.get_batch(batch_id)
        active = reopened._frontier.get_active_frontier(CONVERSATION_ID)
        lifecycle = reopened._lifecycle.get_by_conversation(CONVERSATION_ID)
        nodes = reopened._dag.get_session_nodes(SESSION_ID)
        assert batch is not None and active is not None and lifecycle is not None

        status = reopened.get_status()
        async_status = status["async_compaction"]
        doctor = json.loads(reopened.handle_tool_call("lcm_doctor", {}))
        doctor_checks = {check["check"]: check for check in doctor["checks"]}

        if phase in PRECOMMIT_PHASES:
            assert int(active["generation"]) == old_generation
            assert batch.state == "ready"
            assert nodes == []
            assert lifecycle.current_frontier_store_id == old_lifecycle_frontier
            assert async_status["prepared_batches"] == 1
            assert async_status["promoted_batches"] == 0
            grep = json.loads(
                reopened.handle_tool_call("lcm_grep", {"query": SUMMARY_SENTINEL})
            )
            assert grep.get("results", []) == []

            retry = reopened.promote_prepared_compaction(batch_id, messages)
            assert retry.promoted is True
            assert retry.reason != "canonical_source_overlap"
        else:
            assert int(active["generation"]) == old_generation + 1
            assert batch.state == "promoted"
            assert len(nodes) == 1
            assert lifecycle.current_frontier_store_id == batch.frontier_end_store_id
            assert async_status["prepared_batches"] == 0
            assert async_status["promoted_batches"] == 1

            items = reopened._frontier.get_frontier_items(
                CONVERSATION_ID, int(active["generation"])
            )
            assert items
            assert [item["ordinal"] for item in items] == list(range(len(items)))
            node_items = [item for item in items if item["kind"] == "node"]
            assert node_items and int(node_items[-1]["ref_id"]) == int(nodes[-1].node_id)
            retry = reopened.promote_prepared_compaction(batch_id, messages)
            assert retry.promoted is False
            assert retry.reason == "batch_state_promoted"

        assert doctor_checks["orphaned_dag_nodes"]["status"] == "pass"
        doctor_async = doctor_checks["async_compaction_batches"]["detail"]
        assert doctor_async["prepared_batches"] == async_status["prepared_batches"]
        assert doctor_async["promoted_batches"] == async_status["promoted_batches"]

        final_active = reopened._frontier.get_active_frontier(CONVERSATION_ID)
        final_batch = reopened._frontier.get_batch(batch_id)
        final_lifecycle = reopened._lifecycle.get_by_conversation(CONVERSATION_ID)
        final_nodes = reopened._dag.get_session_nodes(SESSION_ID)
        assert final_active is not None and final_batch is not None
        assert final_lifecycle is not None
        assert int(final_active["generation"]) == old_generation + 1
        assert final_batch.state == "promoted"
        assert len(final_nodes) == 1
        final_items = reopened._frontier.get_frontier_items(
            CONVERSATION_ID, int(final_active["generation"])
        )
        assert final_items
        assert final_lifecycle.current_frontier_store_id == final_batch.frontier_end_store_id
        assert [item["ordinal"] for item in final_items] == list(
            range(len(final_items))
        )
        previous_end = 0
        for item in final_items:
            assert int(item["source_start"]) > previous_end
            assert int(item["source_end"]) >= int(item["source_start"])
            previous_end = int(item["source_end"])
        assert {
            int(item["ref_id"])
            for item in final_items
            if item["kind"] == "node"
        } == {int(node.node_id) for node in final_nodes}
        expected_tail_ids = [
            int(row["store_id"])
            for row in reopened._store.get_session_messages_after(
                SESSION_ID,
                after_store_id=int(final_batch.frontier_end_store_id),
            )
        ]
        assert [
            int(item["ref_id"])
            for item in final_items
            if item["kind"] == "message"
        ] == expected_tail_ids
        final_grep = json.loads(
            reopened.handle_tool_call("lcm_grep", {"query": SUMMARY_SENTINEL})
        )
        assert final_grep.get("results")
    finally:
        reopened.shutdown()
