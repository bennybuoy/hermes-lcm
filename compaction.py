"""Leaf-compaction pipeline for the LCM engine (WS5 Seam 6).

The ``CompactionMixin`` holds the compaction gate + pipeline: ``should_compress``
/ ``should_compress_preflight`` (public), the leaf-candidate and chunk-selection
helpers, and the main ``compress`` entry point. These methods were lifted
verbatim out of ``LCMEngine`` and continue to run bound to the engine instance
(``self`` is the ``LCMEngine``), so they read and write the engine's runtime
state (``_ingest_cursor``, ``_store``, ``_dag``, ``_lifecycle``, status/telemetry
fields, per-turn caches) and call back into engine helpers (ingest,
reconciliation, placeholder-ledger, the summarize-with-rescue step, assembly,
lifecycle) through normal attribute lookup. ``LCMEngine`` mixes this in ahead of
``ContextEngine`` so the mixin's ``compress`` / ``should_compress`` /
``should_compress_preflight`` override the ContextEngine protocol defaults.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .dag import SummaryNode
from .db_bootstrap import (
    FOREGROUND_COMPRESS_BUSY_TIMEOUT_MS,
    FOREGROUND_COMPRESS_DEADLINE_SECONDS,
)
from .message_content import text_content_for_pattern_matching
from .sanitize import _contains_sensitive_redaction
from .sqlite_util import _is_sqlite_locked_error, _temporary_sqlite_busy_timeout
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)


class CompactionMixin:
    def _maybe_reclassify_late_auxiliary_before_compaction_write(self) -> None:
        maybe_reclassify = getattr(
            self,
            "_maybe_reclassify_current_session_as_auxiliary_before_message_ingest",
            None,
        )
        if callable(maybe_reclassify):
            maybe_reclassify()

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._bypasses_lcm_context_management():
            if self._compression_boundary_cooldown_active():
                return False
            if prompt_tokens is not None:
                tokens = prompt_tokens
            else:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    tokens = self._current_auxiliary_prompt_tokens(auxiliary_session_id)
                else:
                    tokens = self.last_prompt_tokens
            if self._should_force_overflow_recovery(observed_tokens=tokens):
                return True
            if self.threshold_tokens <= 0:
                return False
            return tokens >= self.threshold_tokens
        if self._compression_boundary_cooldown_active():
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            rough = count_messages_tokens(messages)
            if self._compression_boundary_cooldown_active():
                return False
            if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
                return True
            return self.threshold_tokens > 0 and rough >= self.threshold_tokens
        rough = count_messages_tokens(messages)
        pre_ingest_placeholder_ambiguous_noop = False
        pre_ingest_noop_reason = ""
        if (
            self.threshold_tokens > 0
            and rough >= self.threshold_tokens
            and not self._compiled_ignore_message_patterns
            and any(
                self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                for msg in messages
            )
        ):
            eligible, reason = self._leaf_compaction_candidate_status(messages)
            pre_ingest_placeholder_ambiguous_noop = not eligible
            pre_ingest_noop_reason = reason
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages(messages)
                self._record_ingest_success()
            except Exception as e:
                # Fail closed for NORMAL threshold compaction: the store did not
                # accept this turn, so do not compact against a store missing the
                # latest messages - that could rebuild active context without
                # them. But still honor emergency overflow recovery, whose whole
                # job is to keep the prompt under the provider limit; it converges
                # via deterministic L3 truncation without needing the store write.
                self._record_ingest_failure("preflight", e)
                if self._should_force_overflow_recovery(observed_tokens=rough):
                    return True
                return False
        if replay_messages is not None and replay_messages != messages:
            if self._compression_boundary_cooldown_active():
                return False
            replay_rough = count_messages_tokens(replay_messages)
            if self._should_force_overflow_recovery(observed_tokens=replay_rough):
                return self._mark_preflight_compression_requested()
            if self._replay_diff_requests_ingest_cleanup(messages, replay_messages):
                return self._mark_preflight_compression_requested()
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(replay_messages)
            if eligible:
                return self._mark_preflight_compression_requested()
            if self._has_ignored_backlog_outside_fresh_tail(replay_messages):
                return self._mark_preflight_compression_requested()
            if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                if self._should_run_deferred_maintenance(replay_messages, observed_tokens=replay_rough):
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
                logger.info("LCM preflight compression no-op: %s", reason)
                return False
            self._refresh_raw_backlog_debt(replay_messages, observed_tokens=replay_rough)
            if self._should_run_deferred_maintenance(replay_messages, observed_tokens=replay_rough):
                return self._mark_preflight_compression_requested()
            return False
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(messages)
            if eligible:
                return self._mark_preflight_compression_requested()
            if self._has_ignored_backlog_outside_fresh_tail(messages):
                return self._mark_preflight_compression_requested()
            if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
                return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
            return False
        self._refresh_raw_backlog_debt(messages, observed_tokens=rough)
        if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        return False

    def _replay_diff_requests_ingest_cleanup(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
    ) -> bool:
        if len(original_messages) != len(replay_messages):
            return True
        for original_msg, replay_msg in zip(original_messages, replay_messages):
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            if original_text != replay_text:
                if replay_text.startswith("[Externalized LCM ingest payload:"):
                    return True
                if replay_text.startswith("[Externalized payload: kind=raw_payload;"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: assistant output quarantined;"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: message ignored;"):
                    return True
                if "[LCM sensitive redaction:" in replay_text:
                    return True
            if original_msg.get("content") != replay_msg.get("content") and _contains_sensitive_redaction(
                replay_msg.get("content")
            ):
                return True
            if original_msg.get("tool_calls") != replay_msg.get("tool_calls") and _contains_sensitive_redaction(
                replay_msg.get("tool_calls")
            ):
                return True
        return False

    def _has_ignored_backlog_outside_fresh_tail(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._compiled_ignore_message_patterns or not messages:
            return False
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False
        previous_store_id_map = self._current_compress_store_ids_by_message_id
        self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
            messages[leading_anchor_count:fresh_tail_start]
        )
        try:
            return any(
                self._matches_ignore_message_patterns(msg)
                or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                for msg in messages[leading_anchor_count:fresh_tail_start]
            )
        finally:
            self._current_compress_store_ids_by_message_id = previous_store_id_map

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.
        """
        if not messages:
            return False, "empty message list"
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"
        generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
        if self._compiled_ignore_message_patterns or generated_placeholder_hashes:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(candidate_raw)
            try:
                filtered_candidate_raw: list[Dict[str, Any]] = []
                for msg in candidate_raw:
                    content_text = text_content_for_pattern_matching(msg.get("content")) or ""
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    )
                    if (
                        self._matches_ignore_message_patterns(msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                        or self._is_ignored_active_replay_placeholder(msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        continue
                    filtered_candidate_raw.append(msg)
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
            candidate_raw = filtered_candidate_raw
            if not candidate_raw:
                return False, "no eligible raw backlog outside fresh tail"

        if force_overflow:
            return True, "forced overflow recovery"

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
        if raw_tokens_outside_tail < working_leaf_chunk_tokens:
            return False, "raw backlog outside fresh tail is below leaf chunk threshold"
        return True, "eligible raw backlog outside fresh tail"

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail

        Foreground cutover claims ``_foreground_compress_active`` so the
        background preparer cannot enter LLM/SQLite critical sections while
        this path runs. Every DB wait is bounded; on timeout or deadline the
        call returns a clear failure/cooldown state instead of hanging while
        the host holds ``compression_locks``.
        """
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        # Phase timing bookkeeping for operator diagnostics (issue #3).
        phase_timings: dict[str, float] = {}
        deadline_seconds = float(
            getattr(
                self._config,
                "foreground_compress_deadline_seconds",
                FOREGROUND_COMPRESS_DEADLINE_SECONDS,
            )
            or FOREGROUND_COMPRESS_DEADLINE_SECONDS
        )
        busy_timeout_ms = int(
            getattr(
                self._config,
                "foreground_compress_busy_timeout_ms",
                FOREGROUND_COMPRESS_BUSY_TIMEOUT_MS,
            )
            or FOREGROUND_COMPRESS_BUSY_TIMEOUT_MS
        )
        _compress_started = time.perf_counter()
        deadline_at = _compress_started + max(1.0, deadline_seconds)
        self._last_compression_status = "running"
        self._last_compression_noop_reason = ""
        self._last_compress_phase_timings_ms = phase_timings
        # Signal background worker to yield (issue #3 priority).
        fg_event = getattr(self, "_foreground_compress_active", None)
        if fg_event is not None:
            fg_event.set()

        def _phase(name: str, started: float) -> None:
            phase_timings[name] = (time.perf_counter() - started) * 1000.0

        def _deadline_exceeded() -> bool:
            return time.perf_counter() >= deadline_at

        canonical_fallback_state: dict[
            str, Optional[List[Dict[str, Any]]]
        ] = {"messages": None}

        def _fail_timeout(
            reason: str,
            return_messages: Optional[List[Dict[str, Any]]] = None,
        ) -> List[Dict[str, Any]]:
            fallback_messages = (
                return_messages
                if return_messages is not None
                else canonical_fallback_state["messages"]
            )
            self._last_compression_status = "failed"
            self._last_compression_noop_reason = reason
            self._last_compaction_duration_ms = (
                time.perf_counter() - _compress_started
            ) * 1000.0
            # Cooldown so an immediate second pre-API pass does not re-enter
            # a hang loop while the host still owns compression_locks.
            try:
                self._last_boundary_skip_time = time.time()
            except Exception:
                logger.debug("LCM failed to arm compression cooldown", exc_info=True)
            logger.warning(
                "LCM compress aborted: %s (wall=%.1fms phases=%s)",
                reason,
                self._last_compaction_duration_ms,
                phase_timings,
            )
            if fallback_messages is not None:
                # Successful compaction/promotion paths reset this cursor to
                # the host replacement length. Bounded exits after publication
                # must do the same or the next appended raw message can be
                # skipped when the stale cursor is clamped.
                self._ingest_cursor = len(fallback_messages)
                return fallback_messages
            return messages

        try:
            return self._compress_body(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
                phase_timings=phase_timings,
                phase=_phase,
                deadline_exceeded=_deadline_exceeded,
                fail_timeout=_fail_timeout,
                canonical_fallback_state=canonical_fallback_state,
                busy_timeout_ms=busy_timeout_ms,
                compress_started=_compress_started,
            )
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                return _fail_timeout(f"sqlite_locked: {exc}")
            if canonical_fallback_state["messages"] is not None:
                logger.error(
                    "LCM compress failed after async publication; returning "
                    "canonical best-effort replacement",
                    exc_info=True,
                )
                return _fail_timeout(f"compress_error_after_publication: {exc}")
            # Unexpected errors before publication must still release host lock
            # ownership by marking failure, then re-raise for caller visibility.
            self._last_compression_status = "failed"
            self._last_compression_noop_reason = f"compress_error: {exc}"
            self._last_compaction_duration_ms = (
                time.perf_counter() - _compress_started
            ) * 1000.0
            logger.error(
                "LCM compress failed after %.1fms: %s",
                self._last_compaction_duration_ms,
                exc,
                exc_info=True,
            )
            raise
        finally:
            if fg_event is not None:
                fg_event.clear()
            self._last_compress_phase_timings_ms = dict(phase_timings)
            # Never leave status stuck at "running" — that wedges the async
            # worker and confuses host lock owners on force-exit.
            if getattr(self, "_last_compression_status", "") == "running":
                self._last_compression_status = "failed"
                self._last_compression_noop_reason = (
                    self._last_compression_noop_reason
                    or "compress_exited_while_running"
                )

    def _compress_body(
        self,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        phase_timings: dict[str, float],
        phase,
        deadline_exceeded,
        fail_timeout,
        canonical_fallback_state: dict[
            str, Optional[List[Dict[str, Any]]]
        ],
        busy_timeout_ms: int,
        compress_started: float,
    ) -> List[Dict[str, Any]]:
        """Inner compress pipeline (runs under foreground-priority + deadline)."""

        # Once async promotion publishes a canonical leaf, any later bounded
        # failure must return the assembled replacement rather than replaying
        # the original host input that still contains covered raw rows.
        canonical_fallback_messages: Optional[List[Dict[str, Any]]] = None

        # Try to promote a prepared async batch before running foreground
        # compaction. On success, skip the leaf summarization path and
        # reassemble active context from the (now-canonical) DAG, dropping
        # covered raw messages from the returned host context (issue #2).
        promote_hook = getattr(self, "_try_promote_prepared_batch", None)
        if callable(promote_hook):
            try:
                t0 = time.perf_counter()
                if deadline_exceeded():
                    return fail_timeout("deadline_before_promote")
                lock_wait_started = time.perf_counter()
                with _temporary_sqlite_busy_timeout(
                    [
                        getattr(getattr(self, "_store", None), "_conn", None),
                        getattr(getattr(self, "_dag", None), "_conn", None),
                        getattr(getattr(self, "_frontier", None), "_conn", None),
                        getattr(getattr(self, "_lifecycle", None), "_conn", None),
                    ],
                    busy_timeout_ms,
                ):
                    lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
                    if lock_wait_ms > 5.0:
                        logger.info(
                            "LCM compress promote path lock-wait setup %.1fms (busy_timeout=%dms)",
                            lock_wait_ms,
                            busy_timeout_ms,
                        )
                    promoted = promote_hook(messages)
                phase("promotion_lookup", t0)
                if promoted:
                    t_asm = time.perf_counter()
                    covered = list(
                        getattr(self, "_async_last_promoted_source_ids", None) or []
                    )
                    # Host replacement: strip covered raw messages so the
                    # returned prompt is under cutover threshold and does not
                    # immediately re-trigger compression (issue #2).
                    filter_fn = getattr(
                        self, "_filter_messages_excluding_covered_store_ids", None
                    )
                    if callable(filter_fn) and covered:
                        working_messages = filter_fn(list(messages), covered)
                    else:
                        # Fallback: keep only leading anchor + fresh tail.
                        leading = self._leading_anchor_count(messages)
                        tail_start = self._fresh_tail_start(messages)
                        working_messages = list(messages[:leading]) + list(
                            messages[tail_start:]
                        )
                    leading_anchor_count = self._leading_anchor_count(working_messages)
                    # Anchor text comes from the pre-filter host view so the
                    # latest user objective is preserved even when its raw
                    # row was covered by the promoted leaf.
                    anchor_source_messages = list(messages)
                    anchor_leading_count = self._leading_anchor_count(
                        anchor_source_messages
                    )
                    self._pending_context_anchor_messages = anchor_source_messages[
                        anchor_leading_count:
                    ]
                    try:
                        compressed = self._assemble_context(
                            working_messages[0] if leading_anchor_count else None,
                            working_messages[leading_anchor_count:],
                        )
                    finally:
                        self._pending_context_anchor_messages = None
                    phase("assembly", t_asm)
                    canonical_fallback_messages = compressed
                    canonical_fallback_state["messages"] = compressed

                    # Partial promote may still leave the provider prompt above
                    # the post-compaction target. When eligible raw remains,
                    # fall through to foreground leaf compaction in this same
                    # cutover without re-running the already-promoted summary.
                    post_target = None
                    target_fn = getattr(self, "_post_compaction_target_tokens", None)
                    if callable(target_fn):
                        try:
                            post_target = target_fn()
                        except Exception:
                            post_target = None
                    pre_msg_tokens = count_messages_tokens(messages)
                    assembled_tokens = count_messages_tokens(compressed)
                    if current_tokens is not None and current_tokens > 0:
                        overhead = max(0, int(current_tokens) - pre_msg_tokens)
                        estimated_after_promote = overhead + assembled_tokens
                    else:
                        estimated_after_promote = assembled_tokens
                    remaining_leading = self._leading_anchor_count(working_messages)
                    remaining_fresh_start = self._fresh_tail_start(working_messages)
                    remaining_raw = (
                        working_messages[remaining_leading:remaining_fresh_start]
                        if remaining_fresh_start > remaining_leading
                        else []
                    )
                    target_unreached_after_promote = (
                        post_target is not None
                        and post_target > 0
                        and estimated_after_promote > post_target
                        and bool(remaining_raw)
                    )
                    continue_toward_target = (
                        target_unreached_after_promote
                        and not deadline_exceeded()
                    )
                    # Clear covered-id stash so a later compress cannot reuse it.
                    self._async_last_promoted_source_ids = []
                    if continue_toward_target:
                        logger.info(
                            "LCM async promote partial: prompt still above target "
                            "(%d > %d); continuing foreground leaf compaction "
                            "(%d remaining raw msgs, covered=%d)",
                            estimated_after_promote,
                            post_target,
                            len(remaining_raw),
                            len(covered),
                        )
                        # Host context for further leaves is the filtered view;
                        # the promoted leaf is already canonical in the DAG.
                        messages = working_messages
                        current_tokens = estimated_after_promote
                    else:
                        self.compression_count += 1
                        self._last_compaction_duration_ms = (
                            time.perf_counter() - compress_started
                        ) * 1000.0
                        self._ingest_cursor = len(compressed)
                        self._last_compression_status = "compacted"
                        self._last_compression_noop_reason = (
                            "post_compaction_target_not_reached: "
                            "deadline_after_async_promotion"
                            if target_unreached_after_promote and deadline_exceeded()
                            else ""
                        )
                        logger.info(
                            "LCM async promote-on-compress finished in %.1fms "
                            "(%d→%d msgs, covered=%d, phases=%s)",
                            self._last_compaction_duration_ms,
                            len(messages),
                            len(compressed),
                            len(covered),
                            phase_timings,
                        )
                        return compressed
            except Exception as exc:
                if _is_sqlite_locked_error(exc):
                    return fail_timeout(
                        f"sqlite_locked_during_promote: {exc}",
                        canonical_fallback_messages,
                    )
                if canonical_fallback_messages is not None:
                    return fail_timeout(
                        f"post_promotion_error: {exc}",
                        canonical_fallback_messages,
                    )
                logger.debug("LCM async promote hook failed", exc_info=True)

        if deadline_exceeded():
            return fail_timeout(
                "deadline_before_foreground",
                canonical_fallback_messages,
            )

        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            bypass_current_tokens = current_tokens
            if bypass_current_tokens is None or bypass_current_tokens <= 0:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    auxiliary_prompt_tokens = self._current_auxiliary_prompt_tokens(
                        auxiliary_session_id
                    )
                    if auxiliary_prompt_tokens > 0:
                        bypass_current_tokens = auxiliary_prompt_tokens
            return self._compress_lcm_bypassed_session(
                messages,
                current_tokens=bypass_current_tokens,
                focus_topic=focus_topic,
                force=force,
            )

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        # NOTE: deliberately do NOT clear the spend guard on force_overflow.
        # force_overflow is automatic (set every turn the prompt exceeds the
        # assembly cap), which is exactly the sustained-over-cap state a runaway
        # compaction loop produces - clearing it per turn would defeat the guard
        # in the case it exists for. A tripped guard still converges the
        # emergency via deterministic L3 truncation (no LLM spend).
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        # Bound SQLite lock waits so a concurrent writer cannot hang compress
        # while the host holds compression_locks (issue #3).
        t_ingest = time.perf_counter()
        try:
            with _temporary_sqlite_busy_timeout(
                [
                    getattr(getattr(self, "_store", None), "_conn", None),
                    getattr(getattr(self, "_lifecycle", None), "_conn", None),
                ],
                busy_timeout_ms,
            ):
                working_messages = self._ingest_messages(messages)
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                return fail_timeout(
                    f"sqlite_locked_during_ingest: {exc}",
                    canonical_fallback_messages,
                )
            raise
        phase("ingest", t_ingest)
        if deadline_exceeded():
            return fail_timeout(
                "deadline_after_ingest",
                canonical_fallback_messages,
            )
        ingest_cleanup_changed_active_context = working_messages != messages
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        dropped_replayed_scaffold_messages = False
        leaf_passes = 0
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = (
            not force_overflow
            and self._should_run_deferred_maintenance(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
        )
        if deferred_maintenance_active:
            self._lifecycle.record_maintenance_attempt(self._conversation_id)
        base_max_leaf_passes = 4 if self._config.dynamic_leaf_chunk_enabled else 1
        max_leaf_passes = base_max_leaf_passes
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )

        def _resolved_post_compaction_target() -> Optional[int]:
            target_fn = getattr(self, "_post_compaction_target_tokens", None)
            if not callable(target_fn):
                return None
            try:
                value = target_fn()
            except Exception:
                return None
            if value is None:
                return None
            try:
                tokens = int(value)
            except (TypeError, ValueError):
                return None
            return tokens if tokens > 0 else None

        post_compaction_target = _resolved_post_compaction_target()
        # When converging from cutover toward an independent target, allow
        # enough bounded leaf passes to close the gap (still hard-capped).
        # Non-dynamic mode normally consumes the whole backlog in one pass,
        # but adaptive rescue can publish only a prefix; that partial publish
        # must be allowed to continue rather than leaving target convergence
        # to a lossy assembly trim.
        if (
            not force_overflow
            and not deferred_maintenance_active
            and post_compaction_target is not None
            and estimated_active_tokens > post_compaction_target
        ):
            if self._config.dynamic_leaf_chunk_enabled:
                chunk_hint = max(
                    1,
                    int(
                        self._config.dynamic_leaf_chunk_max
                        or self._config.leaf_chunk_tokens
                        or 1
                    ),
                )
            else:
                chunk_hint = max(1, int(self._config.leaf_chunk_tokens or 1))
            gap = int(estimated_active_tokens) - int(post_compaction_target)
            needed = (gap // chunk_hint) + 2
            max_leaf_passes = max(base_max_leaf_passes, min(16, needed))

        explicit_focus_topic = focus_topic is not None

        noop_reason = "no eligible raw backlog outside fresh tail"
        partial_completion_reason = ""
        dependent_reply_message_ids: set[int] = set()
        preexisting_dependent_reply_records = self._load_generated_ignored_dependent_reply_records()

        while leaf_passes < max_leaf_passes:
            if deadline_exceeded():
                if leaf_compacted_this_turn:
                    partial_completion_reason = (
                        "post_compaction_target_not_reached: "
                        "deadline_after_leaf_publication"
                    )
                    logger.warning(
                        "LCM compaction deadline reached after %d published leaf "
                        "pass%s; returning canonical best-effort replacement",
                        leaf_passes,
                        "es" if leaf_passes != 1 else "",
                    )
                    break
                return fail_timeout(
                    "deadline_during_leaf_selection",
                    canonical_fallback_messages,
                )
            n = len(working_messages)
            fresh_tail_start = self._fresh_tail_start(working_messages)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            candidate_start = leading_anchor_count
            while (
                candidate_start < fresh_tail_start
                and self._is_replayed_context_scaffold_message(working_messages[candidate_start])
            ):
                candidate_start += 1
            if candidate_start > leading_anchor_count:
                dropped_replayed_scaffold_messages = True
                working_messages = working_messages[:leading_anchor_count] + working_messages[candidate_start:]
                pressure_messages = pressure_messages[:leading_anchor_count] + pressure_messages[candidate_start:]
                candidate_start = leading_anchor_count
                n = len(working_messages)
                fresh_tail_start = self._fresh_tail_start(working_messages)
                if fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            if candidate_start < fresh_tail_start:
                self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
                    working_messages[leading_anchor_count:]
                )
                compactable_pairs = list(
                    zip(
                        working_messages[candidate_start:fresh_tail_start],
                        pressure_messages[candidate_start:fresh_tail_start],
                    )
                )
                kept_working: list[Dict[str, Any]] = []
                kept_pressure: list[Dict[str, Any]] = []
                dropped_ignored_backlog = False
                drop_dependent_reply = False
                for working_msg, pressure_msg in compactable_pairs:
                    role = str(working_msg.get("role") or "")
                    content_text = text_content_for_pattern_matching(working_msg.get("content")) or ""
                    generated_dependent_reply = self._is_generated_ignored_dependent_reply(
                        working_msg,
                        content_text,
                    )
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(working_msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                    if (
                        self._matches_ignore_message_patterns(working_msg)
                        or self._matches_ignore_message_patterns(pressure_msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(working_msg)
                        or self._is_ignored_active_replay_placeholder(working_msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        dropped_ignored_backlog = True
                        if role in {"user", "system", "tool", "assistant"}:
                            drop_dependent_reply = True
                        continue
                    if generated_dependent_reply:
                        dependent_reply_message_ids.add(id(working_msg))
                        if role in {"assistant", "tool"}:
                            drop_dependent_reply = True
                    if drop_dependent_reply and role in {"assistant", "tool"}:
                        dependent_reply_message_ids.add(id(working_msg))
                        self._remember_generated_ignored_dependent_reply(working_msg, content_text)
                    if role in {"user", "system"}:
                        drop_dependent_reply = False
                    kept_working.append(working_msg)
                    kept_pressure.append(pressure_msg)
                drop_dependent_reply_into_tail = drop_dependent_reply
                if dropped_ignored_backlog:
                    dropped_replayed_scaffold_messages = True
                    working_messages = (
                        working_messages[:candidate_start]
                        + kept_working
                        + working_messages[fresh_tail_start:]
                    )
                    pressure_messages = (
                        pressure_messages[:candidate_start]
                        + kept_pressure
                        + pressure_messages[fresh_tail_start:]
                    )
                    n = len(working_messages)
                    fresh_tail_start = self._fresh_tail_start(working_messages)
                if drop_dependent_reply_into_tail:
                    tail_scan_start = max(fresh_tail_start, leading_anchor_count)
                    pending_tail_dependents: list[tuple[Dict[str, Any], str]] = []
                    saw_tail_boundary = False
                    for tail_msg in working_messages[tail_scan_start:]:
                        if not isinstance(tail_msg, dict):
                            continue
                        tail_role = str(tail_msg.get("role") or "")
                        if tail_role in {"user", "system"}:
                            saw_tail_boundary = True
                            break
                        if tail_role in {"assistant", "tool"}:
                            tail_text = text_content_for_pattern_matching(tail_msg.get("content")) or ""
                            self._remember_generated_ignored_dependent_reply(tail_msg, tail_text)
                            pending_tail_dependents.append((tail_msg, tail_text))
                    if saw_tail_boundary or leading_anchor_count > 0 or kept_working:
                        for tail_msg, _tail_text in pending_tail_dependents:
                            dependent_reply_message_ids.add(id(tail_msg))
                if dropped_ignored_backlog and fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            # Auto-derive focus topic from the post-filter compaction view when
            # not explicitly provided.  The derived focus is summarizer-visible,
            # so it must follow the same ignored-message filtering as the leaf
            # chunk itself.
            if not explicit_focus_topic:
                focus_topic = self._derive_auto_focus_topic(working_messages)

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            # Once cutover has started, keep compacting remaining raw toward
            # the post-compaction target even when the backlog is smaller than
            # the normal leaf chunk threshold (bounded by max_leaf_passes).
            converging_to_target = (
                not force_overflow
                and not deferred_maintenance_active
                and post_compaction_target is not None
                and estimated_active_tokens >= post_compaction_target
            )
            if self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    if not (
                        (deferred_maintenance_active and critical_budget_pressure)
                        or converging_to_target
                    ):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    if not (
                        (deferred_maintenance_active and critical_budget_pressure)
                        or converging_to_target
                    ):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                to_compact = candidate_raw

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            selected_raw_chunk = to_compact
            summary_input_chunk = [
                message for message in selected_raw_chunk if id(message) not in dependent_reply_message_ids
            ]
            if not summary_input_chunk:
                compacted_chunk = selected_raw_chunk
                source_tokens = count_messages_tokens(selected_raw_chunk)
                summary_text = (
                    "Filtered replies derived from ignored messages.\n"
                    "[Expand for details: ignored-dependent reply]"
                )
                _level = 0
                _rescue_attempts = 0
            else:
                # Pre-compaction extraction: best-effort, never blocks compaction.
                # Use the same dependency-filtered view as summarization so ignored
                # turns cannot leak through derived assistant/tool replies.
                if self._config.extraction_enabled:
                    self._run_pre_compaction_extraction(summary_input_chunk)

                if deadline_exceeded():
                    if leaf_compacted_this_turn:
                        partial_completion_reason = (
                            "post_compaction_target_not_reached: "
                            "deadline_before_later_summarization"
                        )
                        logger.warning(
                            "LCM compaction deadline reached before a later "
                            "summary after %d published leaf pass%s; returning "
                            "canonical best-effort replacement",
                            leaf_passes,
                            "es" if leaf_passes != 1 else "",
                        )
                        break
                    return fail_timeout(
                        "deadline_before_summarization",
                        canonical_fallback_messages,
                    )
                t_sum = time.perf_counter()
                compacted_chunk, source_tokens, summary_text, _level, _rescue_attempts = self._summarize_leaf_chunk_with_rescue(
                    summary_input_chunk,
                    focus_topic=focus_topic,
                )
                phase("summarization", t_sum)
            compacted_summary_ids = {id(message) for message in compacted_chunk}
            compacted_positions = [
                idx for idx, message in enumerate(selected_raw_chunk) if id(message) in compacted_summary_ids
            ]
            last_compacted_raw_pos = max(compacted_positions) if compacted_positions else len(compacted_chunk) - 1
            last_consumed_raw_pos = last_compacted_raw_pos
            while (
                last_consumed_raw_pos + 1 < len(selected_raw_chunk)
                and id(selected_raw_chunk[last_consumed_raw_pos + 1]) in dependent_reply_message_ids
            ):
                last_consumed_raw_pos += 1
            source_lookup_chunk = selected_raw_chunk[: last_consumed_raw_pos + 1]
            selected_raw_len = len(source_lookup_chunk)
            remaining_messages = working_messages[leading_anchor_count + selected_raw_len:]
            source_tokens = count_messages_tokens(source_lookup_chunk)

            source_lineage_chunk = [
                message for message in source_lookup_chunk if id(message) not in dependent_reply_message_ids
            ]
            source_store_ids = self._get_store_ids_for_messages(source_lineage_chunk)
            source_store_ids = sorted(dict.fromkeys(source_store_ids))
            consumed_store_ids = self._get_store_ids_for_messages(source_lookup_chunk)
            consumed_store_ids = sorted(dict.fromkeys(consumed_store_ids))
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            t_pub = time.perf_counter()
            published_node_id = 0
            advanced_frontier_generation = 0
            frontier_publication_ready = False
            previous_last_compacted_store_id = self._last_compacted_store_id
            try:
                with _temporary_sqlite_busy_timeout(
                    [
                        getattr(getattr(self, "_dag", None), "_conn", None),
                        getattr(getattr(self, "_frontier", None), "_conn", None),
                        getattr(getattr(self, "_lifecycle", None), "_conn", None),
                        getattr(getattr(self, "_store", None), "_conn", None),
                    ],
                    busy_timeout_ms,
                ):
                    published_node_id = self._dag.add_node(node)
                    self._maybe_gc_compacted_tool_results(compacted_chunk, source_store_ids)
                    self._last_compacted_store_id = max(consumed_store_ids) if consumed_store_ids else 0
                    # Foreground leaf publish wins the async CAS race before the
                    # monotonic lifecycle marker advances. If this atomic
                    # generation+items write fails, the lifecycle checkpoint is
                    # still unchanged and the leaf can be deleted losslessly.
                    note_async = getattr(self, "_note_foreground_async_frontier_advance", None)
                    if callable(note_async):
                        try:
                            frontier_result: Any = note_async(
                                source_end_store_id=self._last_compacted_store_id,
                                node_id=int(published_node_id or 0),
                                covered_source_ids=list(source_store_ids),
                            )
                            frontier_generation = (
                                frontier_result
                                if isinstance(frontier_result, int)
                                else 0
                            )
                            if not frontier_generation:
                                raise RuntimeError(
                                    "foreground async frontier advance failed"
                                )
                            frontier_publication_ready = True
                            if frontier_generation > 0:
                                advanced_frontier_generation = frontier_generation
                        except Exception:
                            logger.debug(
                                "LCM foreground async frontier advance failed",
                                exc_info=True,
                            )
                            raise
                    else:
                        frontier_publication_ready = True
                    self._persist_frontier_marker()
            except Exception as exc:
                if published_node_id:
                    # add_node() commits on its own connection. If a later
                    # store/frontier operation fails, the leaf is canonical and
                    # its covered raw chunk must be consumed before assembly.
                    self._last_compacted_store_id = (
                        max(consumed_store_ids) if consumed_store_ids else 0
                    )
                    # The committed leaf must publish the async frontier before
                    # the monotonic lifecycle marker. Retry only whichever stage
                    # did not already succeed; never create a second generation
                    # after a post-frontier lifecycle error.
                    recovery_note_async = getattr(
                        self,
                        "_note_foreground_async_frontier_advance",
                        None,
                    )
                    recovery_frontier_ok = frontier_publication_ready
                    if not recovery_frontier_ok and callable(recovery_note_async):
                        try:
                            recovery_result: Any = recovery_note_async(
                                source_end_store_id=self._last_compacted_store_id,
                                node_id=int(published_node_id),
                                covered_source_ids=list(source_store_ids),
                            )
                            recovery_generation = (
                                recovery_result
                                if isinstance(recovery_result, int)
                                else 0
                            )
                            recovery_frontier_ok = bool(recovery_generation)
                            if recovery_generation > 0:
                                advanced_frontier_generation = recovery_generation
                        except Exception:
                            recovery_frontier_ok = False
                            logger.debug(
                                "LCM recovery frontier advance failed after "
                                "committed foreground leaf",
                                exc_info=True,
                            )
                    elif not callable(recovery_note_async):
                        recovery_frontier_ok = True

                    recovery_lifecycle_ok = False
                    if recovery_frontier_ok:
                        try:
                            self._persist_frontier_marker()
                        except Exception:
                            logger.debug(
                                "LCM recovery lifecycle advance failed after "
                                "committed foreground leaf",
                                exc_info=True,
                            )
                        try:
                            lifecycle_state = self._lifecycle.get_by_conversation(
                                self._conversation_id
                            )
                            recovery_lifecycle_ok = bool(
                                lifecycle_state is not None
                                and lifecycle_state.current_session_id
                                == self._session_id
                                and int(
                                    lifecycle_state.current_frontier_store_id or 0
                                )
                                >= int(self._last_compacted_store_id or 0)
                            )
                        except Exception:
                            logger.debug(
                                "LCM could not verify lifecycle frontier after "
                                "committed foreground leaf",
                                exc_info=True,
                            )

                    if not (recovery_frontier_ok and recovery_lifecycle_ok):
                        # Publication is incomplete. Roll back an exact frontier
                        # generation before deleting the DAG leaf; if a concurrent
                        # writer has superseded it, keep the leaf and fail closed
                        # rather than create a frontier that references no node.
                        frontier_rolled_back = True
                        if advanced_frontier_generation > 0:
                            try:
                                frontier_rolled_back = bool(
                                    self._frontier.rollback_frontier_generation(
                                        self._conversation_id,
                                        advanced_frontier_generation,
                                    )
                                )
                            except Exception:
                                frontier_rolled_back = False
                                logger.error(
                                    "LCM could not roll back foreground frontier "
                                    "generation %d",
                                    advanced_frontier_generation,
                                    exc_info=True,
                                )
                        if not frontier_rolled_back:
                            raise RuntimeError(
                                "foreground frontier rollback failed after "
                                "incomplete publication"
                            ) from exc

                        rolled_back = False
                        try:
                            rolled_back = bool(
                                self._dag.delete_node(int(published_node_id))
                            )
                        except Exception:
                            logger.error(
                                "LCM could not roll back foreground leaf %d after "
                                "frontier publication failure",
                                published_node_id,
                                exc_info=True,
                            )
                        if not rolled_back:
                            raise RuntimeError(
                                "frontier publication and DAG rollback both failed"
                            ) from exc
                        self._last_compacted_store_id = (
                            previous_last_compacted_store_id
                        )
                        if leaf_compacted_this_turn:
                            partial_completion_reason = (
                                "post_compaction_target_not_reached: "
                                "frontier_advance_failed_leaf_rolled_back"
                            )
                            break
                        return fail_timeout(
                            "frontier_advance_failed_leaf_rolled_back",
                            canonical_fallback_messages,
                        )
                    pressure_remaining_messages = pressure_messages[
                        leading_anchor_count + selected_raw_len:
                    ]
                    working_messages = (
                        working_messages[:leading_anchor_count]
                        + remaining_messages
                    )
                    pressure_messages = (
                        pressure_messages[:leading_anchor_count]
                        + pressure_remaining_messages
                    )
                    leaf_compacted_this_turn = True
                    leaf_passes += 1
                    estimated_active_tokens = max(
                        0,
                        estimated_active_tokens - source_tokens + summary_tokens,
                    )
                    partial_completion_reason = (
                        "post_compaction_target_not_reached: "
                        "publication_followup_error_after_leaf"
                    )
                    logger.warning(
                        "LCM publication follow-up failed after committed leaf "
                        "%d; returning canonical best-effort replacement: %s",
                        published_node_id,
                        exc,
                    )
                    break
                if _is_sqlite_locked_error(exc):
                    if leaf_compacted_this_turn:
                        partial_completion_reason = (
                            "post_compaction_target_not_reached: "
                            "sqlite_lock_after_leaf_publication"
                        )
                        logger.warning(
                            "LCM SQLite lock reached after %d published leaf "
                            "pass%s; returning canonical best-effort replacement",
                            leaf_passes,
                            "es" if leaf_passes != 1 else "",
                        )
                        break
                    return fail_timeout(
                        f"sqlite_locked_during_publication: {exc}",
                        canonical_fallback_messages,
                    )
                raise
            phase("dag_publication", t_pub)

            pressure_remaining_messages = pressure_messages[leading_anchor_count + selected_raw_len:]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)

            if not force_overflow:
                # Ordinary cutover is the sole preflight trigger, but once a
                # cutover starts the leaf loop must converge toward the
                # independent post-compaction target (provider-prompt scale),
                # not stop as soon as estimated tokens fall below cutover.
                post_target = post_compaction_target or _resolved_post_compaction_target()
                stop_tokens = (
                    post_target
                    if post_target is not None and post_target > 0
                    else self.threshold_tokens
                )
                if (
                    (not deferred_maintenance_active)
                    and stop_tokens is not None
                    and stop_tokens > 0
                    and estimated_active_tokens < stop_tokens
                ):
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_start = self._fresh_tail_start(working_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_start
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:remaining_fresh_start
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                still_above_target = (
                    post_target is not None
                    and estimated_active_tokens >= post_target
                )
                if remaining_raw_tokens < remaining_threshold:
                    if not (
                        (deferred_maintenance_active and critical_budget_pressure)
                        or still_above_target
                    ):
                        break

        if not leaf_compacted_this_turn:
            self._refresh_raw_backlog_debt(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
            if force_overflow and len(messages) >= 1:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                compressed = self._assemble_overflow_recovery_context(
                    working_messages[0] if leading_anchor_count else None,
                    working_messages[leading_anchor_count:],
                    assembly_cap_override=recovery_assembly_cap,
                )
                return self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                )
            active_context_messages = self._drop_preexisting_generated_ignored_dependent_eof_replies(
                working_messages,
                preexisting_dependent_reply_records,
            )
            if dropped_replayed_scaffold_messages:
                leading_anchor_count = self._leading_anchor_count(active_context_messages)
                anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
                self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
                try:
                    sanitized_messages = self._assemble_context(
                        active_context_messages[0] if leading_anchor_count else None,
                        active_context_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                    )
                finally:
                    self._pending_context_anchor_messages = None
            else:
                sanitized_messages = self._sanitize_active_context_messages(
                    active_context_messages,
                    insert_missing_tool_stubs=False,
                )
            if sanitized_messages != working_messages or ingest_cleanup_changed_active_context:
                # _ingest_messages() already advanced the cursor to the original
                # active-context length. If the host continues from a sanitized
                # or reassembled context, keeping the old cursor could make the
                # next appended messages look already ingested. This applies to
                # content-only cleanup as well as dropped-message cleanup.
                self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "sanitized"
                self._last_compression_noop_reason = ""
            else:
                if dropped_replayed_scaffold_messages:
                    # The active context changed even though no new leaf node was
                    # written. Keep the cursor aligned with the returned context
                    # so the next appended turn is ingested instead of skipped.
                    self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = noop_reason
                logger.info("LCM compression no-op: %s", noop_reason)
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(sanitized_messages)
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(sanitized_messages)
            )
            return sanitized_messages

        # Step 6: Check if condensation is needed. A deadline that expires
        # after publication cannot roll back the committed leaf; return a
        # canonical best-effort host replacement and avoid starting another
        # LLM call for condensation.
        if deadline_exceeded() and not partial_completion_reason:
            partial_completion_reason = (
                "post_compaction_target_not_reached: "
                "deadline_after_leaf_publication"
            )
        if not partial_completion_reason:
            self._maybe_condense(
                focus_topic=focus_topic,
                leaf_compacted_this_turn=True,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
            )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            compressed = self._assemble_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
            )
        finally:
            self._pending_context_anchor_messages = None
        self.compression_count += 1
        self._last_compaction_duration_ms = (time.perf_counter() - compress_started) * 1000.0
        logger.info(
            "LCM leaf compaction finished in %.1fms phases=%s",
            self._last_compaction_duration_ms,
            phase_timings,
        )
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = partial_completion_reason
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)
        self._ingest_cursor_needs_reconcile = False

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        # ── Active-context cleanup / tool-pair guardrail (same as _assemble_context) ──
        # compress() output is consumed directly by the main loop in some
        # edge cases (e.g. forced overflow recovery bypassing _assemble_context).
        compressed = self._sanitize_active_context_messages(compressed)
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(compressed)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(compressed)
        )

        return compressed
