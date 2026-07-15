from __future__ import annotations

from typing import Any

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens


def _engine(tmp_path, *, enabled: bool) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "prompt-aware.db"),
        fresh_tail_count=1,
        max_assembly_tokens=0,
        prompt_aware_eviction_enabled=enabled,
    ))
    engine.on_session_start(
        "prompt-aware-session",
        conversation_id="prompt-aware-conversation",
        platform="test",
        context_length=8_192,
    )
    return engine


def _add_summary(engine: LCMEngine, summary: str, source_id: int) -> int:
    return engine._dag.add_node(SummaryNode(
        session_id=engine.current_session_id,
        depth=0,
        summary=summary,
        token_count=20,
        source_token_count=100,
        source_ids=[source_id],
        source_type="messages",
        expand_hint="details",
    ))


def _part(node_id: int, summary: str) -> str:
    return (
        f"[Recent Summary (d0, node {node_id})]\n"
        f"{summary}\n"
        "[Expand for details: details]"
    )


def _contents(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def test_prompt_aware_eviction_is_opt_in_and_reports_selection(tmp_path):
    engine = _engine(tmp_path, enabled=False)
    old = "garden irrigation mulch pruning schedule " * 4
    relevant = "quantum migration rollback quantum migration rollback " * 4
    old_id = _add_summary(engine, old, 1)
    relevant_id = _add_summary(engine, relevant, 2)
    system = {"role": "system", "content": "system"}
    tail = [{"role": "user", "content": "What is the quantum migration rollback plan?"}]
    summary_role = "user"
    single_summary_budget = max(
        count_message_tokens({"role": summary_role, "content": _part(old_id, old)}),
        count_message_tokens({"role": summary_role, "content": _part(relevant_id, relevant)}),
    )
    cap = (
        count_message_tokens(system)
        + count_message_tokens(tail[0])
        + single_summary_budget
    )
    try:
        chronological = engine._assemble_context(
            system,
            tail,
            assembly_cap_override=cap,
            include_lcm_note=False,
        )
        assert "garden irrigation" in _contents(chronological)
        assert "quantum migration rollback quantum" not in _contents(chronological)
        assert engine.get_status()["assembly_selection"]["mode"] == "chronological"

        engine._config.prompt_aware_eviction_enabled = True
        selected = engine._assemble_context(
            system,
            tail,
            assembly_cap_override=cap,
            include_lcm_note=False,
        )
        selected_text = _contents(selected)
        assert "quantum migration rollback quantum" in selected_text
        assert "garden irrigation" not in selected_text
        selection = engine.get_status()["assembly_selection"]
        assert selection["mode"] == "prompt-aware"
        assert selection["items_considered"] == 2
        assert selection["items_evicted"] == 1
        assert selection["tokens_evicted"] > 0
    finally:
        engine.shutdown()


def test_prompt_aware_selection_restores_chronological_order(tmp_path):
    engine = _engine(tmp_path, enabled=True)
    older = "release database migration evidence " * 3
    unrelated = "garden irrigation pruning notes " * 3
    newer = "database migration database migration release evidence " * 4
    older_id = _add_summary(engine, older, 1)
    unrelated_id = _add_summary(engine, unrelated, 2)
    newer_id = _add_summary(engine, newer, 3)
    system = {"role": "system", "content": "system"}
    tail = [{"role": "user", "content": "Show database migration release evidence"}]
    relevant_pair = "\n\n---\n\n".join([
        _part(older_id, older),
        _part(newer_id, newer),
    ])
    cap = (
        count_message_tokens(system)
        + count_message_tokens(tail[0])
        + count_message_tokens({"role": "user", "content": relevant_pair})
    )
    try:
        assembled = engine._assemble_context(
            system,
            tail,
            assembly_cap_override=cap,
            include_lcm_note=False,
        )
        text = _contents(assembled)
        assert "garden irrigation" not in text
        assert text.index(f"node {older_id}") < text.index(f"node {newer_id}")
        assert engine.get_status()["assembly_selection"]["mode"] == "prompt-aware"
    finally:
        engine.shutdown()


def test_prompt_aware_mode_falls_back_for_prompt_without_terms(tmp_path):
    engine = _engine(tmp_path, enabled=True)
    first = "first historical summary " * 5
    second = "second historical summary " * 5
    first_id = _add_summary(engine, first, 1)
    second_id = _add_summary(engine, second, 2)
    system = {"role": "system", "content": "system"}
    tail = [{"role": "user", "content": "...?!"}]
    cap = (
        count_message_tokens(system)
        + count_message_tokens(tail[0])
        + max(
            count_message_tokens({"role": "user", "content": _part(first_id, first)}),
            count_message_tokens({"role": "user", "content": _part(second_id, second)}),
        )
    )
    try:
        assembled = engine._assemble_context(
            system,
            tail,
            assembly_cap_override=cap,
            include_lcm_note=False,
        )
        text = _contents(assembled)
        assert "first historical summary" in text
        assert "second historical summary" not in text
        assert engine.get_status()["assembly_selection"]["mode"] == "chronological"
    finally:
        engine.shutdown()
