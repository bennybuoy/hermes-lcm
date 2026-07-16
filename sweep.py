"""Bounded generation-scoped full-sweep compaction.

The sweep deliberately separates candidate construction from publication.  All
leaf and condensation summaries are built in memory, then inserted into the DAG
in topological order inside the same writer transaction that performs the
frontier CAS, batch settlement, and lifecycle advance.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .dag import SummaryNode
from .escalation import summarize_with_escalation
from .frontier import compute_source_identity_hash
from .tokens import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

_FULL_SWEEP_QUERY_BATCH = 400
_FULL_SWEEP_MAX_FRONTIER_ROWS = 10_000
_FULL_SWEEP_MAX_MESSAGE_ROWS = 10_000
_FULL_SWEEP_MAX_LOCKED_SERIALIZED_BYTES = 4 * 1024 * 1024
_FULL_SWEEP_MAX_LOCKED_ROWS = 50_000
_FULL_SWEEP_MAX_CANONICAL_ROWS = 10_000
_FULL_SWEEP_MAX_CANONICAL_EDGES = 40_000
_FULL_SWEEP_MAX_CANONICAL_DEPTH = 64


@dataclass(eq=False)
class _StagedNode:
    depth: int
    summary: str
    token_count: int
    source_token_count: int
    raw_source_ids: list[int]
    message_source_ids: list[int] = field(default_factory=list)
    raw_messages_by_id: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    child_sources: list["_StagedNode"] = field(default_factory=list)
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    node_id: int = 0


class FullSweepMixin:
    """Engine mixin implementing issue #8's private-candidate sweep."""

    @staticmethod
    def _charge_full_sweep_locked_read(
        read_budget: dict[str, float | int],
        *,
        rows: int,
        serialized_bytes: int,
        label: str,
    ) -> None:
        if time.monotonic() >= float(read_budget["deadline_at"]):
            raise RuntimeError(f"{label} deadline exceeded")
        next_rows = int(read_budget["rows"]) + int(rows)
        if next_rows > int(read_budget["max_rows"]):
            raise RuntimeError(f"{label} row bound exceeded")
        next_bytes = int(read_budget["bytes"]) + max(0, int(serialized_bytes))
        if next_bytes > int(read_budget["max_bytes"]):
            raise RuntimeError(f"{label} serialized byte bound exceeded")
        read_budget["rows"] = next_rows
        read_budget["bytes"] = next_bytes

    def _full_sweep_status(
        self,
        *,
        reason: str,
        passes: int,
        partial: bool,
        publication_count: int,
        leaf_count: int,
        condensation_count: int,
        used_minimum_fanin: bool,
        started_at: float,
    ) -> dict[str, Any]:
        status = {
            "reason": reason,
            "passes": int(passes),
            "partial": bool(partial),
            "publication_count": int(publication_count),
            "leaf_count": int(leaf_count),
            "condensation_count": int(condensation_count),
            "used_minimum_fanin": bool(used_minimum_fanin),
            "duration_ms": max(0.0, (time.monotonic() - started_at) * 1000.0),
        }
        self._last_full_sweep_status = status
        return status

    @staticmethod
    def _full_sweep_frontier_items(
        roots: list[_StagedNode],
        stored_tail: list[dict[str, Any]],
        carried_items: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [dict(item) for item in (carried_items or [])]
        carried_raw_ids = {
            int(item["ref_id"])
            for item in items
            if item.get("kind") == "message"
            and int(item.get("source_start") or 0) == int(item.get("ref_id") or 0)
            and int(item.get("source_end") or 0) == int(item.get("ref_id") or 0)
        }
        for root in roots:
            source_ids = sorted({int(source_id) for source_id in root.raw_source_ids if source_id})
            if not source_ids or root.node_id <= 0:
                continue
            items.append(
                {
                    "kind": "node",
                    "ref_id": root.node_id,
                    "source_start": source_ids[0],
                    "source_end": source_ids[-1],
                }
            )
        for row in stored_tail:
            store_id = int(row.get("store_id") or 0)
            if store_id <= 0 or store_id in carried_raw_ids:
                continue
            items.append(
                {
                    "kind": "message",
                    "ref_id": store_id,
                    "source_start": store_id,
                    "source_end": store_id,
                }
            )
        items.sort(
            key=lambda item: (
                int(item["source_start"]),
                0 if item["kind"] == "node" else 1,
                int(item["ref_id"]),
            )
        )
        validated: list[dict[str, Any]] = []
        previous_end = 0
        for item in items:
            start = int(item["source_start"])
            end = int(item["source_end"])
            if start <= previous_end or end < start:
                raise RuntimeError(
                    "full sweep candidate range overlap would drop a frontier item"
                )
            validated.append(item)
            previous_end = end
        return validated

    @staticmethod
    def _bounded_full_sweep_frontier_rows_no_commit(
        conn,
        conversation_id: str,
        generation: int,
        *,
        deadline_at: float,
        read_budget: dict[str, float | int],
    ) -> list[tuple[str, int, int, int]]:
        rows: list[tuple[str, int, int, int]] = []
        last_ordinal = -1
        while True:
            if time.monotonic() >= deadline_at:
                raise RuntimeError("full sweep locked frontier deadline exceeded")
            remaining = _FULL_SWEEP_MAX_FRONTIER_ROWS - len(rows)
            page_limit = min(_FULL_SWEEP_QUERY_BATCH, remaining + 1)
            page = conn.execute(
                """SELECT ordinal, kind, ref_id, source_start, source_end
                   FROM lcm_frontier_items
                   WHERE conversation_id=? AND generation=? AND ordinal>?
                   ORDER BY ordinal LIMIT ?""",
                (conversation_id, int(generation), last_ordinal, page_limit),
            ).fetchall()
            if not page:
                return rows
            for ordinal, kind, ref_id, start, end in page:
                if len(rows) >= _FULL_SWEEP_MAX_FRONTIER_ROWS:
                    raise RuntimeError("full sweep locked frontier row bound exceeded")
                charge = len(str(kind).encode("utf-8", errors="replace")) + 48
                FullSweepMixin._charge_full_sweep_locked_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=charge,
                    label="full sweep locked frontier",
                )
                rows.append((str(kind), int(ref_id), int(start or 0), int(end or 0)))
                last_ordinal = int(ordinal)
            if len(page) < page_limit:
                return rows

    @staticmethod
    def _bounded_full_sweep_message_tail_no_commit(
        conn,
        session_id: str,
        source_end: int,
        *,
        deadline_at: float,
        read_budget: dict[str, float | int],
    ) -> list[dict[str, int]]:
        rows: list[dict[str, int]] = []
        last_store_id = int(source_end)
        while True:
            if time.monotonic() >= deadline_at:
                raise RuntimeError("full sweep locked message deadline exceeded")
            remaining = _FULL_SWEEP_MAX_MESSAGE_ROWS - len(rows)
            page_limit = min(_FULL_SWEEP_QUERY_BATCH, remaining + 1)
            page = conn.execute(
                """SELECT store_id FROM messages
                   WHERE session_id=? AND store_id>? ORDER BY store_id LIMIT ?""",
                (session_id, last_store_id, page_limit),
            ).fetchall()
            if not page:
                return rows
            for (raw_store_id,) in page:
                if len(rows) >= _FULL_SWEEP_MAX_MESSAGE_ROWS:
                    raise RuntimeError("full sweep locked message row bound exceeded")
                FullSweepMixin._charge_full_sweep_locked_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=32,
                    label="full sweep locked message",
                )
                store_id = int(raw_store_id)
                rows.append({"store_id": store_id})
                last_store_id = store_id
            if len(page) < page_limit:
                return rows

    def _compress_full_sweep(
        self,
        messages: list[dict[str, Any]],
        *,
        current_tokens: int | None = None,
        focus_topic: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Build a bounded mixed-depth candidate and publish it exactly once."""
        started_at = time.monotonic()
        deadline_seconds = max(
            0.001, float(getattr(self._config, "full_sweep_deadline_seconds", 30.0) or 30.0)
        )
        deadline_at = started_at + deadline_seconds
        max_passes = max(1, int(getattr(self._config, "full_sweep_max_passes", 32) or 32))
        passes = 0
        reason = "complete"
        partial = False
        used_minimum_fanin = False
        staged: list[_StagedNode] = []
        roots: list[_StagedNode] = []
        remaining_messages = list(messages)

        working_messages = self._ingest_messages(messages)
        leading = self._leading_anchor_count(working_messages)
        fresh_start = self._fresh_tail_start(working_messages)
        candidate_raw = list(working_messages[leading:fresh_start])
        protected_tail = list(working_messages[fresh_start:])
        # A prior host replacement places the synthetic LCM summary scaffold
        # ahead of the raw tail. It is provider context, not immutable raw
        # source material, and therefore must never become a new leaf.
        while candidate_raw and self._is_replayed_context_scaffold_message(candidate_raw[0]):
            candidate_raw.pop(0)
        if not candidate_raw:
            self._full_sweep_status(
                reason="no-eligible-raw",
                passes=0,
                partial=False,
                publication_count=0,
                leaf_count=0,
                condensation_count=0,
                used_minimum_fanin=False,
                started_at=started_at,
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "no eligible raw backlog outside fresh tail"
            return messages

        original_message_tokens = count_messages_tokens(messages)
        estimated_active_tokens = int(
            current_tokens
            if current_tokens is not None and current_tokens > 0
            else original_message_tokens
        )
        try:
            post_target = self._post_compaction_target_tokens()
        except Exception:
            post_target = self.threshold_tokens
        post_target = int(post_target or self.threshold_tokens or 1)
        leaf_count = 0
        condensation_count = 0

        while candidate_raw:
            if passes >= max_passes:
                reason, partial = "pass-limit", bool(staged)
                break
            if time.monotonic() >= deadline_at:
                reason, partial = "deadline", bool(staged)
                break
            source_tokens_outside_tail = count_messages_tokens(candidate_raw)
            chunk_tokens = self._working_leaf_chunk_tokens(source_tokens_outside_tail)
            selected = self._select_oldest_leaf_chunk(candidate_raw, chunk_tokens)
            if not selected:
                reason, partial = "no-progress", bool(staged)
                break
            timeout_seconds = max(0.001, deadline_at - time.monotonic())
            compacted, _reported_tokens, summary, _level, _attempts = (
                self._summarize_leaf_chunk_with_rescue(
                    selected,
                    focus_topic=focus_topic,
                    timeout_seconds=timeout_seconds,
                )
            )
            compacted = list(compacted)
            if not compacted or len(compacted) > len(selected):
                reason, partial = "no-progress", bool(staged)
                break
            mapped_source_ids = self._get_store_ids_for_messages(compacted)
            source_ids = sorted(dict.fromkeys(mapped_source_ids))
            actual_source_tokens = count_messages_tokens(compacted)
            summary_tokens = count_tokens(summary)
            if (
                not source_ids
                or len(mapped_source_ids) != len(compacted)
                or len(source_ids) != len(mapped_source_ids)
                or summary_tokens >= actual_source_tokens
            ):
                reason, partial = "no-progress", bool(staged)
                break
            earliest_at, latest_at = self._store.get_time_bounds(source_ids)
            staged_leaf = _StagedNode(
                depth=0,
                summary=summary,
                token_count=summary_tokens,
                source_token_count=actual_source_tokens,
                raw_source_ids=source_ids,
                message_source_ids=source_ids,
                raw_messages_by_id=list(zip(mapped_source_ids, compacted)),
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
            )
            staged.append(staged_leaf)
            roots.append(staged_leaf)
            consumed = len(compacted)
            candidate_raw = candidate_raw[consumed:]
            remaining_messages = (
                list(working_messages[:leading]) + candidate_raw + protected_tail
            )
            passes += 1
            leaf_count += 1
            estimated_active_tokens = max(
                0, estimated_active_tokens - actual_source_tokens + summary_tokens
            )
            if estimated_active_tokens < post_target:
                break

        prefix_target = self._effective_summary_prefix_target_tokens()
        max_depth = self._effective_incremental_max_depth()
        while roots and (max_depth < 0 or any(root.depth < max_depth for root in roots)):
            if passes >= max_passes:
                reason, partial = "pass-limit", True
                break
            if time.monotonic() >= deadline_at:
                reason, partial = "deadline", True
                break
            prefix_tokens = sum(root.token_count for root in roots)
            selected_group: list[_StagedNode] = []
            for depth in sorted({root.depth for root in roots}):
                if max_depth >= 0 and depth >= max_depth:
                    continue
                same_depth = [root for root in roots if root.depth == depth]
                fanin = self._effective_condensation_fanin()
                if len(same_depth) >= fanin:
                    selected_group = same_depth[:fanin]
                    break
                min_fanin = self._effective_condensation_min_fanin()
                under_pressure = prefix_target > 0 and prefix_tokens > prefix_target
                if under_pressure and len(same_depth) >= min_fanin:
                    selected_group = same_depth[:min_fanin]
                    used_minimum_fanin = True
                    break
            if not selected_group:
                break
            combined = "\n\n---\n\n".join(node.summary for node in selected_group)
            source_tokens = sum(node.token_count for node in selected_group)
            timeout_seconds = max(0.001, deadline_at - time.monotonic())
            summary, _level = summarize_with_escalation(
                text=combined,
                source_tokens=source_tokens,
                token_budget=max(1, int(source_tokens * 0.40)),
                depth=selected_group[0].depth + 1,
                model=self._config.summary_model,
                fallback_models=self._config.summary_fallback_models,
                circuit_breaker=self._summary_circuit_breaker,
                spend_guard=self._summary_spend_guard,
                timeout=timeout_seconds,
                l2_budget_ratio=self._config.l2_budget_ratio,
                l3_truncate_tokens=self._config.l3_truncate_tokens,
                focus_topic=focus_topic or "",
                custom_instructions=self._config.custom_instructions,
            )
            summary_tokens = count_tokens(summary)
            if summary_tokens >= source_tokens:
                reason, partial = "no-progress", True
                break
            raw_ids = sorted(
                {source_id for node in selected_group for source_id in node.raw_source_ids}
            )
            raw_pairs = sorted(
                [
                    pair
                    for child in selected_group
                    for pair in child.raw_messages_by_id
                ],
                key=lambda pair: pair[0],
            )
            if len({source_id for source_id, _message in raw_pairs}) != len(raw_pairs):
                raise RuntimeError("full sweep staged lineage overlaps")
            parent = _StagedNode(
                depth=selected_group[0].depth + 1,
                summary=summary,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                raw_source_ids=raw_ids,
                child_sources=list(selected_group),
                raw_messages_by_id=raw_pairs,
                created_at=time.time(),
                earliest_at=min(
                    (node.earliest_at for node in selected_group if node.earliest_at is not None),
                    default=None,
                ),
                latest_at=max(
                    (node.latest_at for node in selected_group if node.latest_at is not None),
                    default=None,
                ),
            )
            staged.append(parent)
            roots = [root for root in roots if root not in selected_group]
            roots.append(parent)
            roots.sort(
                key=lambda node: (
                    min(node.raw_source_ids) if node.raw_source_ids else 0,
                    node.depth,
                    node.created_at,
                )
            )
            passes += 1
            condensation_count += 1

        if not staged or leaf_count == 0:
            self._full_sweep_status(
                reason=reason if reason != "complete" else "no-progress",
                passes=passes,
                partial=False,
                publication_count=0,
                leaf_count=leaf_count,
                condensation_count=condensation_count,
                used_minimum_fanin=used_minimum_fanin,
                started_at=started_at,
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            return messages

        conversation_id = self.current_conversation_id
        session_id = self.current_session_id
        policy_fp = self._async_policy_fingerprint()
        route_fp = self._async_route_fingerprint()
        frontier = self._frontier.get_active_frontier(conversation_id)
        if frontier is None:
            self._frontier.ensure_frontier(
                conversation_id,
                session_id,
                policy_fingerprint=policy_fp,
                route_fingerprint=route_fp,
            )
            frontier = self._frontier.get_active_frontier(conversation_id)
        if frontier is None:
            raise RuntimeError("full sweep could not establish base frontier")

        previous_frontier_store_id = int(self._last_compacted_store_id or 0)
        new_generation = 0
        publication_committed = False
        canonical_winner_observed = False
        covered_source_ids = sorted(
            {source_id for root in roots for source_id in root.raw_source_ids}
        )
        source_end = max(covered_source_ids)
        summarized_pairs = [
            (int(source_id), message)
            for root in roots
            for source_id, message in root.raw_messages_by_id
        ]
        summarized_by_id = dict(summarized_pairs)
        if (
            len(summarized_by_id) != len(summarized_pairs)
            or set(summarized_by_id) != set(covered_source_ids)
        ):
            raise RuntimeError("full sweep summarized source mapping is incomplete")
        summarized_messages = [summarized_by_id[source_id] for source_id in covered_source_ids]
        expected_source_identity = self._foreground_source_identity_for_messages(
            summarized_messages,
            covered_source_ids,
        )
        if not expected_source_identity:
            raise RuntimeError("full sweep could not lock summarized source identity")

        def _publication_boundary(phase: str) -> None:
            crash_hook = getattr(self, "_full_sweep_publish_crash_hook", None)
            if callable(crash_hook):
                crash_hook(phase)
            elif crash_hook == phase:
                os._exit(90)  # noqa: PLW1510 - subprocess crash injection
            failure_hook = getattr(self, "_full_sweep_publish_failure_hook", None)
            if callable(failure_hook):
                failure_hook(phase)
            elif failure_hook == phase:
                raise RuntimeError("injected full-sweep publication failure")

        try:
            with self._frontier.publication_transaction() as publication_conn:
                _publication_boundary("after_begin")
                locked_read_budget: dict[str, float | int] = {
                    "rows": 0,
                    "bytes": 0,
                    "max_rows": int(_FULL_SWEEP_MAX_LOCKED_ROWS),
                    "max_bytes": int(_FULL_SWEEP_MAX_LOCKED_SERIALIZED_BYTES),
                    "deadline_at": float(deadline_at),
                }
                locked_frontier = publication_conn.execute(
                    """SELECT generation, session_id, source_end_store_id
                       FROM lcm_active_frontiers WHERE conversation_id=?
                       ORDER BY generation DESC LIMIT 1""",
                    (conversation_id,),
                ).fetchone()
                if (
                    locked_frontier is None
                    or int(locked_frontier[0]) != int(frontier["generation"])
                    or str(locked_frontier[1] or "") != session_id
                ):
                    canonical_winner_observed = True
                    raise RuntimeError("full sweep frontier CAS mismatch")
                canonical_coverage = self._canonical_message_source_ids_no_commit(
                    publication_conn,
                    session_id,
                    deadline_at=deadline_at,
                    max_rows=_FULL_SWEEP_MAX_CANONICAL_ROWS,
                    max_edges=_FULL_SWEEP_MAX_CANONICAL_EDGES,
                    max_depth=_FULL_SWEEP_MAX_CANONICAL_DEPTH,
                    max_bytes=_FULL_SWEEP_MAX_LOCKED_SERIALIZED_BYTES,
                    read_budget=locked_read_budget,
                )
                if canonical_coverage.intersection(covered_source_ids):
                    canonical_winner_observed = True
                    raise RuntimeError("full sweep canonical source overlap")
                if not self._exact_source_rows_exist_no_commit(
                    publication_conn,
                    session_id,
                    covered_source_ids,
                    deadline_at=deadline_at,
                    read_budget=locked_read_budget,
                ):
                    raise RuntimeError("full sweep source rows changed before publication")
                locked_identity = compute_source_identity_hash(
                    publication_conn,
                    session_id,
                    covered_source_ids,
                    read_budget=locked_read_budget,
                    digest_chars=None,
                    role_default="unknown",
                )
                if locked_identity != expected_source_identity:
                    raise RuntimeError("full sweep source identity mismatch")

                for item in staged:
                    source_ids = (
                        [child.node_id for child in item.child_sources]
                        if item.child_sources
                        else list(item.message_source_ids)
                    )
                    if not source_ids or any(source_id <= 0 for source_id in source_ids):
                        raise RuntimeError("full sweep source closure incomplete before DAG insert")
                    node = SummaryNode(
                        session_id=session_id,
                        depth=item.depth,
                        summary=item.summary,
                        token_count=item.token_count,
                        source_token_count=item.source_token_count,
                        source_ids=source_ids,
                        source_type="nodes" if item.child_sources else "messages",
                        created_at=item.created_at,
                        earliest_at=item.earliest_at,
                        latest_at=item.latest_at,
                        expand_hint=self._extract_expand_hint(item.summary),
                    )
                    item.node_id = self._dag.add_node_no_commit(publication_conn, node)
                _publication_boundary("after_nodes")

                covered_start = min(covered_source_ids)
                active_rows = self._bounded_full_sweep_frontier_rows_no_commit(
                    publication_conn,
                    conversation_id,
                    int(frontier["generation"]),
                    deadline_at=deadline_at,
                    read_budget=locked_read_budget,
                )
                carried_items: list[dict[str, Any]] = []
                for kind, ref_id, start, end in active_rows:
                    start, end = int(start or 0), int(end or 0)
                    if start <= 0 or end < start:
                        raise RuntimeError("full sweep active frontier contains invalid range")
                    if kind not in {"node", "message"}:
                        raise RuntimeError("full sweep active frontier has invalid item kind")
                    if kind == "message":
                        if start != int(ref_id) or end != int(ref_id):
                            raise RuntimeError("full sweep message frontier range is invalid")
                        if int(ref_id) in covered_source_ids:
                            # Exact raw lineage is intentionally replaced by
                            # the newly staged summary nodes.
                            continue
                    if not (end < covered_start or start > source_end):
                        raise RuntimeError(
                            "full sweep candidate declared range overlap would lose canonical coverage"
                        )
                    if kind == "node" and publication_conn.execute(
                        "SELECT 1 FROM summary_nodes WHERE node_id=?", (int(ref_id),)
                    ).fetchone() is None:
                        raise RuntimeError("full sweep carry-forward references missing DAG node")
                    carried_items.append({
                        "kind": str(kind), "ref_id": int(ref_id),
                        "source_start": start, "source_end": end,
                    })
                stored_tail = self._bounded_full_sweep_message_tail_no_commit(
                    publication_conn,
                    session_id,
                    source_end,
                    deadline_at=deadline_at,
                    read_budget=locked_read_budget,
                )
                frontier_items = self._full_sweep_frontier_items(
                    roots, stored_tail, carried_items
                )
                if not frontier_items:
                    raise RuntimeError("full sweep candidate produced an empty frontier")
                new_generation = self._frontier.publish_generation_state_no_commit(
                    publication_conn,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    source_end_store_id=source_end,
                    policy_fingerprint=policy_fp,
                    route_fingerprint=route_fp,
                    base_generation=int(frontier["generation"]),
                    items=frontier_items,
                    batch_reason="full_sweep_generation_published",
                    phase_hook=_publication_boundary,
                )
                if not new_generation:
                    canonical_winner_observed = True
                    raise RuntimeError("full sweep frontier CAS mismatch")
            publication_committed = True
            _publication_boundary("after_commit")
            self._last_compacted_store_id = source_end
        except Exception:
            frontier_rolled_back = not (
                publication_committed or canonical_winner_observed
            )
            if frontier_rolled_back:
                self._last_compacted_store_id = previous_frontier_store_id
                self._ingest_cursor = len(messages)
                self._ingest_cursor_needs_reconcile = False
                self._last_compression_status = "failed"
                rollback_reason = "deadline" if reason == "deadline" else "publication-rolled-back"
                self._last_compression_noop_reason = (
                    "full_sweep_deadline_before_publication"
                    if rollback_reason == "deadline"
                    else "full_sweep_publication_rolled_back"
                )
                self._full_sweep_status(
                    reason=rollback_reason,
                    passes=passes,
                    partial=rollback_reason == "deadline",
                    publication_count=0,
                    leaf_count=leaf_count,
                    condensation_count=condensation_count,
                    used_minimum_fanin=used_minimum_fanin,
                    started_at=started_at,
                )
                return self._sanitize_active_context_messages(messages)

            # Exact rollback can lose a race to a newer authoritative
            # generation. In that case the inserted nodes may already be
            # canonical; never replay the stale host input. Resolve the active
            # frontier and return its lossless replacement with an aligned
            # ingest cursor.
            anchor_leading = self._leading_anchor_count(messages)
            self._pending_context_anchor_messages = list(messages[anchor_leading:])
            try:
                canonical = self._assemble_context(
                    messages[0] if anchor_leading else None,
                    list(messages[anchor_leading:]),
                )
            finally:
                self._pending_context_anchor_messages = None
            canonical = self._sanitize_active_context_messages(canonical)
            self._ingest_cursor = len(canonical)
            self._ingest_cursor_needs_reconcile = False
            self._last_compression_status = "failed"
            self._last_compression_noop_reason = "full_sweep_post_publication_failure"
            self._full_sweep_status(
                reason="post-publication-failure",
                passes=passes,
                partial=True,
                publication_count=1,
                leaf_count=leaf_count,
                condensation_count=condensation_count,
                used_minimum_fanin=used_minimum_fanin,
                started_at=started_at,
            )
            return canonical

        leading = self._leading_anchor_count(remaining_messages)
        anchor_messages = list(working_messages)
        anchor_leading = self._leading_anchor_count(anchor_messages)
        self._pending_context_anchor_messages = anchor_messages[anchor_leading:]
        try:
            result = self._assemble_context(
                remaining_messages[0] if leading else None,
                remaining_messages[leading:],
            )
        finally:
            self._pending_context_anchor_messages = None
        self._ingest_cursor = len(result)
        self._ingest_cursor_needs_reconcile = False
        self.compression_count += 1
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = "" if reason == "complete" else reason
        self._full_sweep_status(
            reason=reason,
            passes=passes,
            partial=partial,
            publication_count=1,
            leaf_count=leaf_count,
            condensation_count=condensation_count,
            used_minimum_fanin=used_minimum_fanin,
            started_at=started_at,
        )
        return self._sanitize_active_context_messages(result)
