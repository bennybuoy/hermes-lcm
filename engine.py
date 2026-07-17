"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import copy
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent.context_engine import ContextEngine

from .codex_routing import (
    _codex_oauth_context_cap,
    _is_codex_gpt55_route,
)
from .cache_aware import CacheAwareSignalTracker
from .config import LCMConfig
from .dag import (
    MAX_SOURCE_IDS_JSON_CHARS,
    SummaryDAG,
    SummaryNode,
    decode_source_ids,
)
from .policy import (
    DEFAULT_PREPARATION_RATIO,
    DEFAULT_TARGET_RATIO,
    ModelCompactionPolicy,
    resolve_policy,
)
from .frontier import (
    FrontierStore,
    PREPARED_PAYLOAD_VERSION,
    PreparedBatch,
    PromotionResult,
    compute_source_identity_hash,
    compute_route_fingerprint,
)
from .focus import FocusStore
from .diagnostics import _enforce_state_db_containment
from .engine_registry import (
    _ACTIVE_ENGINE_REGISTRY_LOCK,
    _ACTIVE_ENGINES_BY_CONVERSATION_ID,
    _ACTIVE_ENGINES_BY_SESSION_ID,
    _remove_registry_entries_for_engine,
    resolve_active_lcm_engine,  # noqa: F401  (re-exported: hosts import it from .engine)
)
from .escalation import (
    SummaryCircuitBreaker,
    SummarySpendGuard,
    summarize_with_escalation,
)
from .externalize import (
    build_transcript_gc_placeholder,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    find_externalized_tool_result_content_for_call,
    is_externalized_placeholder,
    load_externalized_payload,
    maybe_externalize_tool_output,
)
from .extraction import (
    extract_before_compaction,
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
    strip_injected_context_blocks,
)
from .ingest_protection import (
    _expected_persisted_output_chars,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _persisted_output_inline_preview_sha256,
    _persisted_output_preview_prefix_digest,
    _persisted_output_saved_path,
    assistant_output_quarantine_reason,
    extract_all_externalized_payload_refs,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_text,
    redact_sensitive_output_text,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
    sensitive_pattern_status,
)
from .runtime_identity import (
    _PLUGIN_ROOT,
    _git_runtime_identity,
    _plugin_metadata,
)
from .schemas import (
    LCM_DESCRIBE,
    LCM_DOCTOR,
    LCM_EXPAND,
    LCM_EXPAND_QUERY,
    LCM_FOCUS,
    LCM_GREP,
    LCM_INSPECT,
    LCM_LOAD_SESSION,
    LCM_STATUS,
)
from .sanitize import (
    _clean_active_assistant_message,
    _should_drop_active_assistant_message,
)
from .session_patterns import (
    build_session_match_keys,
    compile_session_patterns,
    matches_session_pattern,
)
from .message_analysis import (
    _is_synthetic_assistant_noise,
    _matched_tool_call_ids,
    _tool_call_id,
)
from .message_patterns import compile_message_patterns, matches_message_pattern
from .aux_session import AuxiliarySessionMixin
from .placeholder_ledger import PlaceholderLedgerMixin
from .preparation import (
    active_profile_admissions,
    release_profile_admission,
    try_acquire_profile_admission,
)
from .reconcile import ReconcileMixin, _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from .compaction import CompactionMixin
from .sweep import FullSweepMixin
from .reset_state import ResetStateMixin
from .bypass import BypassMixin
from .lifecycle_state import LifecycleStateStore
from .message_content import (
    normalize_content_value,
    stored_text_content_for_pattern_matching,
    text_content_for_pattern_matching,
)
from .sqlite_util import (
    _is_sqlite_locked_error,
    _temporary_sqlite_busy_timeout,
)
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)

_SESSION_END_BUSY_TIMEOUT_MS = 50
_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85

# Auto-focus topic derivation: infer a compact focus hint from the most recent
# real user turns so that summarization can prioritise current user intent.
# Mirrors Hermes upstream fix/compression-auto-focus-topic (#44687 branch).
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700
_PROMPT_AWARE_MAX_TERMS = 64
_PROMPT_AWARE_MAX_SUMMARIES = 512
_PROMPT_AWARE_MAX_PROMPT_CHARS = 20_000
_CANONICAL_LINEAGE_QUERY_BATCH = 400
_CANONICAL_LINEAGE_MAX_ROWS = 10_000
_CANONICAL_LINEAGE_MAX_EDGES = 40_000
_CANONICAL_LINEAGE_MAX_BYTES = 4 * 1024 * 1024
_CANONICAL_LINEAGE_MAX_DEPTH = 64
_PUBLICATION_LOCKED_QUERY_BATCH = 400
_PUBLICATION_LOCKED_MAX_ROWS = 40_000
_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES = 4 * 1024 * 1024
_PUBLICATION_LOCKED_DEADLINE_SECONDS = 2.0
_PUBLICATION_LOCKED_MAX_FIELD_BYTES = 2 * 1024 * 1024
_PUBLICATION_LOCKED_MAX_NESTED_DEPTH = 32
_PUBLICATION_LOCKED_MAX_NESTED_ITEMS = 20_000

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"
_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT = 8


class _CrossSessionCapability:
    """Opaque host-issued authorization for an explicit session allowlist."""

    __slots__ = ("_issuer", "session_ids")

    def __init__(self, issuer: object, session_ids: frozenset[str]):
        self._issuer = issuer
        self.session_ids = session_ids


class LCMEngine(FullSweepMixin, CompactionMixin, ResetStateMixin, ReconcileMixin, AuxiliarySessionMixin, PlaceholderLedgerMixin, BypassMixin, ContextEngine):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_load_session, lcm_describe,
         lcm_expand) to search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        db_path = self._resolve_db_path(hermes_home)
        self._bind_storage(db_path, hermes_home)

        self._session_id: str = ""
        self._session_platform: str = ""
        # Tracks the most recent non-ignored, non-stateless binding so that
        # user-facing tools (lcm_status, lcm_grep default scope, lcm_describe,
        # lcm_expand_query, lcm_doctor) keep showing the foreground session
        # even while a side-channel session (cron, debug) temporarily owns the
        # engine's _session_id binding. Updated alongside _session_id only
        # when _refresh_session_filters classifies the new session as a real
        # foreground (neither ignored nor stateless). Read via the
        # `current_session_id` / `current_session_platform` properties and
        # `current_session_ignored` / `current_session_stateless` /
        # `side_channel_active` companion predicates.
        self._foreground_session_id: str = ""
        self._foreground_session_platform: str = ""
        self._foreground_conversation_id: str = ""
        self._foreground_rebind_session_id: str = ""
        self._foreground_rebind_previous_session_id: str = ""
        self._foreground_rebind_previous_platform: str = ""
        self._foreground_rebind_previous_conversation_id: str = ""
        self._foreground_rebind_parent_session_id: str = ""
        self._conversation_id: str = ""
        self._session_match_keys: list[str] = []
        self._session_ignored = False
        self._session_stateless = False
        self._compiled_ignore_session_patterns = compile_session_patterns(
            self._config.ignore_session_patterns
        )
        self._compiled_stateless_session_patterns = compile_session_patterns(
            self._config.stateless_session_patterns
        )
        self._compiled_ignore_message_patterns = compile_message_patterns(
            self._config.ignore_message_patterns
        )
        self._ignored_message_count: int = 0
        # Raw messages permanently dropped because they matched
        # ignore_message_patterns. These are NOT persisted anywhere, so an
        # over-broad operator pattern silently discards substantive turns from
        # the "lossless" store. Count + log them so the loss is at least
        # visible; full lossless retention (store with ignored=1) is a larger
        # follow-up that touches cursor reconciliation and FTS.
        self._ignore_pattern_dropped_count: int = 0

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        # The cursor is process-local; existing sessions rebound after a
        # gateway restart reconcile it against the durable store on the
        # next ingest.
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation: Dict[str, Any] = {
            "action": "none",
            "reason": "not run",
        }

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.raw_context_length = 0
        self.context_length = 0
        self.effective_context_length_cap: int | None = None
        self.effective_context_length_reason = ""
        self._context_length_source = ""
        self._update_model_pending_session_start = False
        self.threshold_tokens = 0
        self.context_threshold = self._config.context_threshold
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        self._context_threshold_autoraised: dict[str, float] | None = None
        self._compaction_policy: Optional[ModelCompactionPolicy] = None
        self._cross_session_capability_issuer = object()
        self._cache_signal_tracker = CacheAwareSignalTracker()
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self.compression_count = 0
        # Wall-clock of the last leaf compaction (ms); surfaced via telemetry only.
        self._last_compaction_duration_ms = 0.0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        # Host compatibility: LCM treats successful automatic compaction as
        # silent maintenance. Manual /lcm diagnostics and warning/error paths
        # remain explicit.
        self.emit_automatic_compaction_status = False
        self.quiet_mode = True
        self.summary_model = self._config.summary_model
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )
        # Summary spend guard: process-local sliding window so a loop that
        # keeps succeeding cannot burn auxiliary-model budget without bound. When
        # tripped, escalation falls back to deterministic L3 truncation. Set
        # summary_spend_max_calls=0 to disable.
        self._summary_spend_guard = SummarySpendGuard(
            max_calls=int(self._config.summary_spend_max_calls),
            window_seconds=float(self._config.summary_spend_window_seconds),
            backoff_seconds=float(self._config.summary_spend_backoff_seconds),
        )
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        # Ingest-failure tracking. The core promise is that nothing is ever
        # lost, but a swallowed persistence error (disk full, DB locked,
        # corruption) silently breaks it: the turn continues while messages
        # exist only in the volatile host list. Surface it instead of hiding
        # it in a debug log so get_status()/doctor can escalate. Store-scoped,
        # not session-scoped, so it is not cleared on session reset.
        self._ingest_failure_count = 0
        self._consecutive_ingest_failures = 0
        self._last_ingest_error = ""
        self._last_ingest_error_time: float = 0
        # Cooldown timestamp to prevent compression cascade after boundary skip.
        # Set when skip-carry-over path is taken in _continue_compression_boundary.
        self._last_boundary_skip_time: float = 0
        # Temporary source window used only while compress() assembles context.
        # _assemble_context also serves tests and recovery paths directly, so
        # keep anchoring opt-in rather than changing its public behavior.
        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None
        self._current_compress_store_ids_by_message_id: dict[int, int] = {}
        self._current_compress_placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        self._last_active_replay_source_identities: list[tuple[Any, ...]] = []
        self._last_active_replay_messages: list[Dict[str, Any]] = []
        self._generated_ignored_active_replay_placeholder_message_ids: set[int] = set()
        self._logged_filter_config = False
        self._pending_reset_session_id: str = ""
        self._pending_reset_conversation_id: str = ""
        self._pending_reset_frontier_store_id: int = 0
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget: dict[str, int] = {}
        self._compression_boundary_active_placeholder_digest_ordinals: dict[str, set[int]] = {}
        self._compression_boundary_stored_placeholder_digest_counts: dict[str, int] = {}
        self._thread_context = threading.local()
        self._auxiliary_session_ids: set[str] = set()
        self._auxiliary_lineage_session_ids: set[str] = set()
        self._auxiliary_last_prompt_tokens: dict[str, int] = {}
        self._auxiliary_session_generations: dict[str, int] = {}
        self._auxiliary_generation_tokens: dict[int, tuple[Any, int]] = {}
        self._auxiliary_next_generation_token = 0
        self._auxiliary_direct_end_guard_session_ids: set[str] = set()
        self._auxiliary_handoff_parent_session_ids: dict[str, str] = {}
        self._auxiliary_retired_session_generations: dict[str, set[int]] = {}
        self._auxiliary_foreground_reused_session_ids: set[str] = set()
        self._lcm_bypass_lineage_session_ids: set[str] = set()
        self._lcm_bypass_lineage_platforms: dict[str, set[str]] = {}
        self._lcm_non_bypass_platforms: dict[str, set[str]] = {}
        self._lcm_session_last_platform: dict[str, str] = {}
        self._lcm_session_last_normal_platform: dict[str, str] = {}
        self._lcm_session_last_bypassed: dict[str, bool] = {}
        self._lcm_session_last_conversation_id: dict[str, str] = {}
        self._lcm_session_last_normal_conversation_id: dict[str, str] = {}
        self._lcm_bypass_message_prefix_fingerprints: dict[
            str, list[tuple[list[str], bool]]
        ] = {}
        self._lcm_normal_message_prefix_fingerprints: dict[tuple[str, str], list[str]] = {}
        self._lcm_current_start_allows_bypass_lineage = False
        self._auxiliary_session_lock = threading.RLock()
        self._host_fallback_compressor: Any = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False
        # Async background preparation worker (opt-in, daemon thread).
        self._async_worker_thread: Optional[threading.Thread] = None
        self._async_worker_stop: Optional[threading.Event] = None
        self._async_worker_lock = threading.Lock()
        # Storage / worker lifetime ownership.
        #
        # Architecture: a single reentrant lifetime mutex serializes every
        # storage-consuming session lifecycle critical section with
        # stop→close→rebind and stop→close transitions. RLock is intentional so
        # on_session_start → rebind → start (and start → stop when flags are
        # off) re-enter safely. Never introduce a non-reentrant lock that is
        # acquired on a path already holding this one.
        #
        # Lock hierarchy (only this order; never reverse):
        #   1. ``_storage_lifetime_lock`` — session start/end bodies, rebind,
        #      shutdown close, and worker install/stop decisions.
        #   2. ``_async_worker_lock`` — worker thread / stop-event fields only.
        # Join waits for the worker may run while holding (1); they must not
        # attempt to re-acquire a non-reentrant outer lock. The worker loop
        # itself must not ingest messages or otherwise acquire (1), or
        # stop-under-(1) would deadlock. Foreground compression acquires (1)
        # only around its message-store ingestion primitive, never around LLM
        # work or a worker join.
        #
        # ``_storage_lifetime_state``:
        #   "bound"    — helpers usable; worker starts allowed.
        #   "unusable" — rebind closed helpers then failed to bind; starts
        #                refused until a successful rebind recovers.
        #   "closed"   — terminal after successful shutdown close; starts
        #                refused until a successful rebind re-opens.
        self._storage_lifetime_lock = threading.RLock()
        self._storage_lifetime_state: str = "bound"
        # Circuit breaker: consecutive prepare failures → cooldown.
        self._async_worker_consecutive_failures: int = 0
        self._async_worker_cooldown_until: float = 0.0
        # After a cooldown period the next failure re-trips immediately
        # (half-open). Cleared on a successful prepare.
        self._async_worker_half_open: bool = False
        # Telemetry / metrics for get_async_compaction_status().
        self._async_worker_last_tick_at: Optional[float] = None
        self._async_worker_last_tick_duration_ms: Optional[float] = None
        self._async_last_prepare_at: Optional[float] = None
        self._async_last_promote_at: Optional[float] = None
        # Last successful promote telemetry (ms) — validation / publication / wall.
        self._async_last_promote_validation_ms: Optional[float] = None
        self._async_last_promote_publication_ms: Optional[float] = None
        self._async_last_promote_wall_ms: Optional[float] = None
        # Source store_ids covered by the most recent successful promote (for
        # host-side active-context replacement so covered raw rows are dropped).
        self._async_last_promoted_source_ids: list[int] = []
        self._async_last_promoted_node_id: int = 0
        self._async_promotion_ingested_messages: Optional[
            List[Dict[str, Any]]
        ] = None
        # Foreground cutover priority: set while compress() is in the critical
        # path so the background worker must not enter LLM/SQLite work.
        self._foreground_compress_active = threading.Event()
        self._async_total_prepare_attempts: int = 0
        self._async_total_promote_attempts: int = 0
        self._async_total_promote_succeeded: int = 0
        self._async_total_prepared: int = 0
        self._async_prepare_skip_reasons: dict[str, int] = {}
        self._async_cleanup_counts: dict[str, int] = {}
        self._async_last_pressure_signal: str = "none"
        self._async_last_host_prompt_tokens: int = 0
        self._async_last_source_tokens: int = 0
        self._async_last_pressure_mismatch_tokens: int = 0
        self._async_last_expected_reduction_tokens: int = 0
        self._async_last_ready_coverage_tokens: int = 0
        self._async_last_projected_headroom_tokens: int = 0

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM stores mutable
        session binding and ingest cursor state on the engine object itself, so
        sharing one registered instance across agents can let one conversation
        rebind another conversation's raw-message ingest and lifecycle state.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session/cursor/lifecycle runtime state. Runtime
        model and context-window metadata is copied so the clone is immediately
        budget-aware even before a compatible Hermes host calls update_model().
        """
        clone = type(self)(
            config=copy.deepcopy(self._config),
            hermes_home=self._hermes_home,
        )
        clone.model = self.model
        clone.base_url = self.base_url
        clone.api_key = self.api_key
        clone.provider = self.provider
        clone.api_mode = self.api_mode
        if self._context_length_source:
            clone._set_context_length(
                self.raw_context_length,
                source=self._context_length_source,
                model=self.model,
                provider=self.provider,
            )
        elif self.raw_context_length or self.context_length:
            clone._set_context_length(
                self.raw_context_length or self.context_length,
                source="clone_for_agent",
                model=self.model,
                provider=self.provider,
            )
        # ``update_model()`` authority is a per-runtime lifecycle edge, not
        # durable metadata.  Compatible hosts call update_model() on the clone
        # before binding it; hosts that bind only through on_session_start()
        # must still be able to replace the copied prototype route.
        clone._update_model_pending_session_start = False
        clone._lcm_current_start_allows_bypass_lineage = False
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> "LCMEngine":
        """Copy the plugin runtime without pickling SQLite-backed helpers.

        Hermes core may deepcopy plugin context engines while creating isolated
        AIAgent instances. A default object deepcopy walks into MessageStore,
        SummaryDAG, and LifecycleStateStore sqlite3.Connection handles, which
        cannot be pickled. LCM already exposes clone_for_agent() as the safe
        boundary: share durable configuration/database path, but allocate fresh
        per-agent runtime/storage helper objects.
        """
        clone = self.clone_for_agent()
        memo[id(self)] = clone
        return clone

    def _resolve_db_path(self, hermes_home: str = "") -> Path:
        """Resolve the SQLite path for the active Hermes profile/home."""
        if self._config.database_path:
            return Path(self._config.database_path)
        if hermes_home:
            return Path(hermes_home) / "lcm.db"
        return Path.home() / ".hermes" / "lcm.db"

    def _bind_storage(self, db_path: str | Path, hermes_home: str = "") -> None:
        """Bind store/DAG/lifecycle/frontier helpers to one SQLite database."""
        self._store = MessageStore(
            db_path,
            ingest_protection_config=self._config,
            hermes_home=hermes_home,
        )
        self._dag = SummaryDAG(db_path)
        self._lifecycle = LifecycleStateStore(db_path)
        self._frontier = FrontierStore(str(db_path))
        self._focus = FocusStore(str(db_path))

    def _close_storage(self) -> None:
        """Best-effort close of currently bound SQLite helpers."""
        for attr in ("_store", "_dag", "_lifecycle", "_frontier", "_focus"):
            helper = getattr(self, attr, None)
            close = getattr(helper, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LCM failed closing %s during profile rebind", attr, exc_info=True)

    def _reset_profile_runtime_state(self) -> None:
        """Clear process-local session state that cannot cross profile homes."""
        self._unregister_active_engine_binding()
        self._session_id = ""
        self._session_platform = ""
        self._foreground_session_id = ""
        self._foreground_session_platform = ""
        self._foreground_conversation_id = ""
        self._clear_foreground_rebind_candidate()
        self._conversation_id = ""
        self._session_match_keys = []
        self._session_ignored = False
        self._session_stateless = False
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.clear()
            self._auxiliary_lineage_session_ids.clear()
            self._auxiliary_last_prompt_tokens.clear()
            self._auxiliary_session_generations.clear()
            self._auxiliary_generation_tokens.clear()
            self._auxiliary_next_generation_token = 0
            self._auxiliary_direct_end_guard_session_ids.clear()
            self._auxiliary_handoff_parent_session_ids.clear()
            self._auxiliary_retired_session_generations.clear()
            self._auxiliary_foreground_reused_session_ids.clear()
            self._lcm_bypass_lineage_session_ids.clear()
            self._lcm_bypass_lineage_platforms.clear()
            self._lcm_non_bypass_platforms.clear()
            self._lcm_session_last_platform.clear()
            self._lcm_session_last_normal_platform.clear()
            self._lcm_session_last_bypassed.clear()
            self._lcm_session_last_conversation_id.clear()
            self._lcm_session_last_normal_conversation_id.clear()
            self._lcm_bypass_message_prefix_fingerprints.clear()
            self._lcm_normal_message_prefix_fingerprints.clear()
        self._lcm_current_start_allows_bypass_lineage = False
        self._host_fallback_compressor = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False
        self._clear_thread_context_stateless()
        self._reset_session_scoped_runtime_state()

    def _rebind_storage_for_home(self, hermes_home: str = "") -> bool:
        """Switch SQLite-backed state when a reused engine serves another profile.

        Hermes core passes the active ``hermes_home`` on session start.  Older
        Hermes versions may still reuse the same plugin/context-engine object
        after ``HERMES_HOME`` changes, so the plugin must not assume the store
        captured during ``register()`` is still correct.

        Owns the full stop→close→bind transition under ``_storage_lifetime_lock``
        so concurrent session lifecycle, shutdown, and late worker starts cannot
        interleave with closed helpers. A bind failure after close leaves the
        engine ``unusable`` (non-startable) rather than ``bound`` over dead
        connections.
        """
        with self._storage_lifetime_lock:
            if not hermes_home:
                return False
            if self._config.database_path:
                current_home = str(self._hermes_home or "")
                current_store_home = str(
                    getattr(getattr(self, "_store", None), "_hermes_home", "") or ""
                )
                if current_home == str(hermes_home) and current_store_home == str(hermes_home):
                    return False
                self._hermes_home = hermes_home
                store = getattr(self, "_store", None)
                if store is not None:
                    store._hermes_home = hermes_home
                # Configured DB path does not close/reopen SQLite helpers, so it
                # cannot recover ``unusable``/``closed``; leave state unchanged.
                self._reset_profile_runtime_state()
                logger.info(
                    "LCM rebound Hermes home for configured database path %s", hermes_home
                )
                return True

            db_path = self._resolve_db_path(hermes_home)
            current_db = Path(getattr(getattr(self, "_store", None), "db_path", ""))
            if (
                current_db == db_path
                and str(self._hermes_home or "") == str(hermes_home)
                and self._storage_lifetime_state == "bound"
            ):
                return False

            if not self._stop_async_worker():
                logger.warning(
                    "LCM deferred profile storage rebind until async worker exits"
                )
                return False
            self._close_storage()
            try:
                self._hermes_home = hermes_home
                self._bind_storage(db_path, hermes_home)
            except Exception:
                # Close any partial bind so we never retain open handles under
                # a non-startable state (or half-open mixed helper sets).
                try:
                    self._close_storage()
                except Exception:
                    logger.debug(
                        "LCM secondary close after rebind failure failed",
                        exc_info=True,
                    )
                self._storage_lifetime_state = "unusable"
                logger.exception(
                    "LCM storage rebind failed after close; engine left non-startable"
                )
                return False
            self._storage_lifetime_state = "bound"
            self._reset_profile_runtime_state()
            logger.info("LCM rebound storage for Hermes home %s", hermes_home)
            return True

    def _runtime_context_threshold(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[float, str, dict[str, float] | None]:
        configured = float(self._config.context_threshold)
        source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        explicit_lcm_override = source in {
            "env:LCM_CONTEXT_THRESHOLD",
            "config_yaml:lcm.context_threshold",
        }
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        # Per-model threshold overrides take priority over everything else.
        # Longest substring match wins (so "glm-5.2-1M" beats "glm-5.2").
        if self._config.model_thresholds and route_model:
            best_key = ""
            for key in self._config.model_thresholds:
                if key in route_model and len(key) > len(best_key):
                    best_key = key
            if best_key:
                override = float(self._config.model_thresholds[best_key])
                return (
                    override,
                    f"model_thresholds:{best_key}",
                    {"from": configured, "to": override},
                )
        if (
            _is_codex_gpt55_route(route_model, route_provider)
            and self._config.codex_gpt55_autoraise_enabled
            and not explicit_lcm_override
            and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
        ):
            return (
                _CODEX_GPT55_COMPACTION_THRESHOLD,
                "codex_gpt55_autoraise",
                {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
            )
        return configured, source, None

    def _effective_context_length(
        self,
        raw_context_length: int,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[int, int | None, str]:
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        cap = _codex_oauth_context_cap(route_model, route_provider)
        if cap is not None and raw_context_length > cap:
            return (
                cap,
                cap,
                "codex_oauth_context_cap",
            )
        return raw_context_length, None, ""

    def _effective_threshold_tokens(self, context_threshold_tokens: int) -> int:
        """Return the host-visible cutover trigger token count.

        Assembly caps constrain the result of compaction; they are not preflight
        triggers. Keeping these controls independent prevents a deliberately low
        post-compaction target from causing repeated early compaction.
        """
        return context_threshold_tokens

    def _set_context_length(
        self,
        context_length: Any,
        *,
        source: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> bool:
        try:
            parsed_context_length = int(context_length)
        except (TypeError, ValueError):
            logger.debug("LCM ignored invalid %s context_length: %r", source, context_length)
            return False
        if parsed_context_length <= 0:
            logger.debug(
                "LCM cleared non-positive %s context_length: %r",
                source,
                context_length,
            )
            self.raw_context_length = 0
            self.context_length = 0
            self.effective_context_length_cap = None
            self.effective_context_length_reason = ""
            self._context_length_source = source
            self.threshold_tokens = 0
            self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
                self._runtime_context_threshold(model=model, provider=provider)
            )
            self.threshold_percent = self.context_threshold
            return True
        self.raw_context_length = parsed_context_length
        effective_context_length, cap, reason = self._effective_context_length(
            parsed_context_length,
            model=model,
            provider=provider,
        )
        self.context_length = effective_context_length
        self.effective_context_length_cap = cap
        self.effective_context_length_reason = reason
        self._context_length_source = source
        self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
            self._runtime_context_threshold(model=model, provider=provider)
        )
        self.threshold_percent = self.context_threshold
        context_threshold_tokens = int(
            effective_context_length * self.context_threshold
        )
        self.threshold_tokens = self._effective_threshold_tokens(
            context_threshold_tokens
        )
        return True

    def _session_metadata_matches_active_runtime(
        self,
        kwargs: Dict[str, Any],
        *,
        ignore_empty_optional: bool = False,
    ) -> bool:
        if "model" in kwargs and str(kwargs.get("model") or "") != self.model:
            return False
        for key in ("provider", "base_url", "api_key", "api_mode"):
            if key not in kwargs:
                continue
            incoming = str(kwargs.get(key) or "")
            if ignore_empty_optional and not incoming:
                continue
            if incoming != str(getattr(self, key, "") or ""):
                return False
        return True

    @property
    def name(self) -> str:
        return "lcm"

    @property
    def last_compression_status(self) -> str:
        """Public status for the most recent compression/preflight attempt.

        Host runtimes use this to distinguish a real compaction boundary from
        an LCM no-op (for example, when request pressure is high but all
        compactable raw backlog is protected by the fresh tail).
        """
        return self._last_compression_status

    @property
    def last_compression_noop_reason(self) -> str:
        """Human-readable reason for the latest no-op compression decision."""
        return self._last_compression_noop_reason

    @property
    def last_compression_was_noop(self) -> bool:
        """Whether the most recent compression/preflight decision was a no-op."""
        return self._last_compression_status == "noop"

    def _mark_preflight_compression_requested(self) -> bool:
        """Record that preflight found work and clear any stale no-op reason."""
        self._last_compression_status = "pending"
        self._last_compression_noop_reason = ""
        return True

    @property
    def bound_session_id(self) -> str:
        """Session id this engine is actively servicing for ingest/lifecycle.

        This differs from ``current_session_id`` while a side-channel session is
        bound but operator-facing tools should keep showing the foreground
        session. Host lifecycle hooks must compare against this value before
        deciding whether a post-turn ingest needs to rebind the engine.
        """
        return self._session_id

    @property
    def current_session_id(self) -> str:
        """User-facing "current session" id surfaced by LCM tools.

        Returns the most recent foreground binding (the last session id that
        ``_refresh_session_filters`` classified as neither ignored nor
        stateless). Falls back to ``_session_id`` when no foreground has
        ever been bound, so unattended cron-only or stateless-only processes
        remain observable via ``lcm_status``.

        Lifecycle paths (compress, ingest, on_session_end, etc.) must keep
        reading ``_session_id`` directly because those paths must follow the
        binding the engine is actually servicing. Only tool-surface code
        paths that report a "current session" view to operators should read
        this property.
        """
        return self._foreground_session_id or self._session_id

    @property
    def current_session_platform(self) -> str:
        """Platform string paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_session_platform
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        """Conversation id paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_conversation_id
        return self._conversation_id

    @property
    def side_channel_active(self) -> bool:
        """True when an ignored or stateless session has temporarily rebound
        ``_session_id`` while a real foreground binding still exists.

        Operators reading lcm_status during this window see the foreground
        session id and counts (because tools read ``current_session_id``)
        but the engine itself is servicing the side channel. This predicate
        lets diagnostic surfaces (lcm_status, /lcm command) make the
        divergence explicit without recomputing the underlying invariant.
        """
        return bool(self._foreground_session_id) and self._foreground_session_id != self._session_id

    @property
    def current_session_ignored(self) -> bool:
        """``_session_ignored`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-ignored; otherwise this is the bound session's ignore flag.
        """
        if self.side_channel_active:
            return False
        return self._session_ignored

    @property
    def current_session_stateless(self) -> bool:
        """``_session_stateless`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-stateless; otherwise this is the bound session's stateless flag.
        """
        if self.side_channel_active:
            return False
        return self._session_stateless

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._thread_context_stateless():
            auxiliary_session_id = self._thread_context_session_id()
            if auxiliary_session_id:
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                caller_generation = self._in_process_auxiliary_caller_generation(
                    auxiliary_session_id
                )
                with self._auxiliary_session_lock:
                    if auxiliary_session_id not in self._auxiliary_session_ids:
                        return
                    if self._auxiliary_generation_is_retired(
                        auxiliary_session_id,
                        caller_generation,
                    ):
                        return
                    active_generation = self._auxiliary_session_generations.get(
                        auxiliary_session_id
                    )
                    if active_generation is None and caller_generation:
                        expected_parent = self._auxiliary_handoff_parent_session_ids.get(
                            auxiliary_session_id
                        )
                        if expected_parent and self._in_process_parent_session_id(
                            {},
                            session_id=auxiliary_session_id,
                            include_explicit=False,
                        ) != expected_parent:
                            return
                        if auxiliary_session_id in self._auxiliary_last_prompt_tokens:
                            self._auxiliary_direct_end_guard_session_ids.add(
                                auxiliary_session_id
                            )
                        self._auxiliary_last_prompt_tokens.pop(auxiliary_session_id, None)
                        if self._host_fallback_session_id == auxiliary_session_id:
                            self._end_host_fallback_compressor_for_session(
                                auxiliary_session_id,
                                [],
                                current_session_bypasses=True,
                            )
                        self._auxiliary_session_generations[
                            auxiliary_session_id
                        ] = caller_generation
                        active_generation = caller_generation
                    stack = self._thread_context_auxiliary_stack()
                    stack_marks_current_session = bool(
                        active_generation is not None
                        and caller_generation == 0
                        and stack
                        and stack[-1] == auxiliary_session_id
                    )
                    generation_matches = (
                        caller_generation == 0
                        if active_generation is None
                        else caller_generation == active_generation or stack_marks_current_session
                    )
                    if generation_matches:
                        self._auxiliary_last_prompt_tokens[auxiliary_session_id] = prompt_tokens
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        cache_keys = {"cache_read_tokens", "cache_write_tokens"}
        self.cache_metrics_available = any(key in usage for key in cache_keys)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(
            usage.get("output_tokens", self.last_completion_tokens) or 0
        )
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        self.last_reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        self._record_turn_compaction_telemetry()

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def _main_route_fingerprint(self) -> str:
        route = {
            "provider": self.provider or "",
            "model": self.model or "",
            "api_mode": self.api_mode or "",
            "base_url": self.base_url or "",
        }
        return hashlib.sha256(
            json.dumps(route, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    def record_cache_signal(
        self,
        event: str,
        *,
        source: str,
        ttl_seconds: float | None = None,
    ) -> None:
        """Record an explicit host cache write/break signal.

        Usage counters are never consulted. A route change makes the signal
        inapplicable instead of silently transferring cache state.
        """
        policy = getattr(self, "_compaction_policy", None)
        ttl = (
            float(ttl_seconds)
            if ttl_seconds is not None
            else float(
                policy.cache_ttl_seconds
                if policy is not None
                else self._config.cache_ttl_seconds
            )
        )
        self._cache_signal_tracker.record(
            event,
            route_fingerprint=self._main_route_fingerprint(),
            source=source,
            ttl_seconds=ttl,
            conversation_id=self.current_conversation_id,
        )

    def cache_signal_status(self) -> dict[str, object]:
        return self._cache_signal_tracker.status(
            route_fingerprint=self._main_route_fingerprint(),
            conversation_id=self.current_conversation_id,
        )

    def _should_defer_compaction_for_hot_cache(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int,
    ) -> bool:
        policy = getattr(self, "_compaction_policy", None)
        mode = policy.compaction_mode if policy is not None else self._config.compaction_mode
        if mode != "deferred":
            return False
        if self._should_force_overflow_recovery(
            observed_tokens=observed_tokens,
            messages=messages,
        ):
            return False
        signal = self.cache_signal_status()
        if signal.get("state") != "hot":
            return False
        if self._conversation_id:
            try:
                self._lifecycle.record_debt(
                    self._conversation_id,
                    kind="hot_cache_deferred",
                    size_estimate=max(0, int(observed_tokens)),
                )
            except Exception:
                logger.debug("LCM could not persist hot-cache debt", exc_info=True)
        self._last_compression_status = "noop"
        self._last_compression_noop_reason = "hot-cache-deferred"
        return True

    def maintain(
        self,
        messages: List[Dict[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> PreparedBatch | None:
        """Run one deferred maintenance preparation after explicit cache heat ends."""
        active_messages = list(messages or [])
        if self.cache_signal_status().get("state") == "hot":
            return None
        return self.prepare_background_compaction_once(
            active_messages,
            host_prompt_tokens=self.last_prompt_tokens or None,
            force=force,
        )

    def _record_turn_compaction_telemetry(self) -> None:
        """Persist a per-conversation compaction-telemetry snapshot for this turn.

        Best-effort and diagnostic only: any failure is logged at debug and never
        affects the turn. Turns with no token or cache signal are skipped so idle
        turns do not churn the record. The since-compaction accumulators reset off
        the monotonic ``compression_count`` (which also drops to 0 on a session
        reset) rather than instrumenting the compaction hot path.
        """
        conversation_id = self._conversation_id
        if not conversation_id:
            return
        prompt_tokens = self.last_prompt_tokens
        cache_read = self.last_cache_read_tokens
        cache_write = self.last_cache_write_tokens
        if (
            prompt_tokens <= 0
            and cache_read <= 0
            and cache_write <= 0
            and not self.cache_metrics_available
        ):
            return
        try:
            existing = self._store.read_compaction_telemetry(conversation_id) or {}

            if cache_read > 0 or cache_write > 0:
                cache_state = "hot"
            elif self.cache_metrics_available:
                cache_state = "cold"
            else:
                cache_state = "unknown"
            cold_streak = int(existing.get("consecutive_cold_observations", 0) or 0)
            if cache_state == "hot":
                cold_streak = 0
            elif cache_state == "cold":
                cold_streak += 1

            prev_count = int(existing.get("compression_count_at_record", 0) or 0)
            compacted = self.compression_count > prev_count
            rebaselined = self.compression_count != prev_count  # compaction or session reset
            if rebaselined:
                turns_since = 0
                peak_tokens_since = prompt_tokens
            else:
                turns_since = int(existing.get("turns_since_leaf_compaction", 0) or 0) + 1
                peak_tokens_since = max(
                    int(existing.get("peak_prompt_tokens_since_leaf_compaction", 0) or 0),
                    prompt_tokens,
                )
            total_compactions = int(existing.get("total_compactions", 0) or 0)
            if compacted:
                total_compactions += self.compression_count - prev_count
                last_leaf_compaction_at = time.time()
                last_compaction_duration_ms = round(self._last_compaction_duration_ms, 3)
            else:
                last_leaf_compaction_at = existing.get("last_leaf_compaction_at")
                last_compaction_duration_ms = existing.get("last_compaction_duration_ms")

            record = dict(existing)
            record.update({
                "conversation_id": conversation_id,
                "last_observed_prompt_tokens": prompt_tokens,
                "last_observed_cache_read": cache_read,
                "last_observed_cache_write": cache_write,
                "cache_state": cache_state,
                "consecutive_cold_observations": cold_streak,
                "turns_since_leaf_compaction": turns_since,
                "peak_prompt_tokens_since_leaf_compaction": peak_tokens_since,
                # Reserved carry-forward field; no live 'medium'/'high' computation yet.
                "activity_band": existing.get("activity_band", "low"),
                "provider": self.provider or existing.get("provider"),
                "model": self.model or existing.get("model"),
                "last_api_call_at": time.time(),
                "last_leaf_compaction_at": last_leaf_compaction_at,
                "last_compaction_duration_ms": last_compaction_duration_ms,
                "total_compactions": total_compactions,
                "compression_count_at_record": self.compression_count,
            })
            if cache_state == "hot":
                record["last_cache_hit_at"] = time.time()
            self._store.write_compaction_telemetry(conversation_id, record)
        except Exception:
            logger.debug("LCM compaction telemetry update failed", exc_info=True)

    def _compression_boundary_cooldown_active(self) -> bool:
        """Return true while a boundary skip is in its short no-compress window."""
        if self._last_boundary_skip_time <= 0:
            return False
        elapsed = time.time() - self._last_boundary_skip_time
        if elapsed < 60:
            logger.debug(
                "LCM compression cooldown active: %.1f seconds since boundary skip",
                elapsed,
            )
            return True
        self._last_boundary_skip_time = 0
        return False

    def _record_ingest_success(self) -> None:
        self._consecutive_ingest_failures = 0

    def _record_ingest_failure(self, where: str, error: Exception) -> None:
        """Track a swallowed ingest error so it is operator-visible.

        Escalates to error level once failures are consecutive: a single
        transient lock is a warning, but a sustained inability to persist
        means the lossless guarantee is broken and must not stay hidden.
        """
        self._ingest_failure_count += 1
        self._consecutive_ingest_failures += 1
        self._last_ingest_error = f"{type(error).__name__}: {error}"
        self._last_ingest_error_time = time.time()
        message = "LCM ingest failed (%s): %s [consecutive=%d, total=%d]"
        args = (
            where,
            error,
            self._consecutive_ingest_failures,
            self._ingest_failure_count,
        )
        if self._consecutive_ingest_failures >= 3:
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

    def _clear_foreground_rebind_candidate(self) -> None:
        self._foreground_rebind_session_id = ""
        self._foreground_rebind_previous_session_id = ""
        self._foreground_rebind_previous_platform = ""
        self._foreground_rebind_previous_conversation_id = ""
        self._foreground_rebind_parent_session_id = ""

    def _clear_foreground_rebind_candidate_if_bound_session_confirmed(self) -> None:
        if self._foreground_rebind_session_id == self._session_id:
            self._clear_foreground_rebind_candidate()

    def _session_has_foreground_branch_marker(
        self,
        session_id: str,
        parent_session_id: str,
    ) -> bool:
        """Return True when Hermes state.db records ``session_id`` as a real branch.

        An un-ingested foreground branch and a provisional late-marked auxiliary
        child can both expose the same in-process parent id before either has
        durable LCM rows. The canonical host signal for real foreground branches
        is the ``sessions.model_config._branched_from`` marker in state.db; use
        it to decide whether a displaced un-ingested candidate is safe to restore.
        """
        session_id = str(session_id or "")
        parent_session_id = str(parent_session_id or "")
        if not session_id or not parent_session_id:
            return False
        path = self._state_db_path()
        if not path.exists():
            return False
        try:
            uri = path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT parent_session_id, model_config
                    FROM sessions
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            logger.debug("LCM foreground branch marker probe failed", exc_info=True)
            return False
        if not row:
            return False
        row_parent_id = str(row[0] or "")
        if row_parent_id != parent_session_id:
            return False
        try:
            model_config = json.loads(row[1] or "{}")
        except (TypeError, ValueError):
            return False
        if not isinstance(model_config, dict):
            return False
        return str(model_config.get("_branched_from") or "") == parent_session_id

    def _remember_foreground_rebind_candidate(self, session_id: str) -> None:
        """Remember the foreground view displaced by a provisional normal bind.

        Bind-time auxiliary detection can miss background-review children when
        the host seeds its marker late. If that happens, ``on_session_start``
        temporarily classifies the child as a normal foreground and overwrites
        the operator-facing foreground pointer. The first-ingest recheck may
        later prove the child is auxiliary; this snapshot lets that recovery
        restore the parent foreground instead of falling back to the child.
        """
        session_id = str(session_id or "")
        if not session_id:
            self._clear_foreground_rebind_candidate()
            return
        if self._foreground_rebind_session_id == session_id:
            return
        if (
            self._foreground_rebind_session_id
            and self._foreground_rebind_previous_session_id
            and self._foreground_session_id == self._foreground_rebind_session_id
        ):
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
                require_auxiliary_frame=False,
            )
            rebind_candidate_is_foreground_branch = bool(
                self._foreground_rebind_parent_session_id
                and self._session_has_foreground_branch_marker(
                    self._foreground_rebind_session_id,
                    self._foreground_rebind_parent_session_id,
                )
            )
            if (
                (
                    parent_session_id == self._foreground_rebind_session_id
                    and (
                        not self._foreground_rebind_parent_session_id
                        or rebind_candidate_is_foreground_branch
                    )
                )
                or (parent_session_id and not self._foreground_rebind_parent_session_id)
                or (
                    parent_session_id
                    and parent_session_id == self._foreground_rebind_parent_session_id
                    and rebind_candidate_is_foreground_branch
                )
            ):
                self._foreground_rebind_previous_session_id = self._foreground_session_id
                self._foreground_rebind_previous_platform = self._foreground_session_platform
                self._foreground_rebind_previous_conversation_id = self._foreground_conversation_id
            self._foreground_rebind_session_id = session_id
            self._foreground_rebind_parent_session_id = parent_session_id
            return
        if self._foreground_session_id and self._foreground_session_id != session_id:
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
                require_auxiliary_frame=False,
            )
            self._foreground_rebind_session_id = session_id
            self._foreground_rebind_previous_session_id = self._foreground_session_id
            self._foreground_rebind_previous_platform = self._foreground_session_platform
            self._foreground_rebind_previous_conversation_id = self._foreground_conversation_id
            self._foreground_rebind_parent_session_id = parent_session_id
            return
        self._clear_foreground_rebind_candidate()

    def _restore_foreground_after_late_auxiliary_reclassification(self, session_id: str) -> None:
        if self._foreground_session_id != session_id:
            if self._foreground_rebind_session_id == session_id:
                self._clear_foreground_rebind_candidate()
            return
        if (
            self._foreground_rebind_session_id == session_id
            and self._foreground_rebind_previous_session_id
        ):
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            if (
                parent_session_id
                and parent_session_id != session_id
                and parent_session_id == self._foreground_rebind_previous_session_id
                and self._lcm_session_last_normal_conversation_id.get(parent_session_id)
            ):
                self._foreground_session_id = parent_session_id
                self._foreground_session_platform = self._lcm_session_last_normal_platform.get(
                    parent_session_id,
                    self._foreground_rebind_previous_platform,
                )
                self._foreground_conversation_id = self._lcm_session_last_normal_conversation_id[
                    parent_session_id
                ]
            else:
                self._foreground_session_id = self._foreground_rebind_previous_session_id
                self._foreground_session_platform = self._foreground_rebind_previous_platform
                self._foreground_conversation_id = self._foreground_rebind_previous_conversation_id
        else:
            self._foreground_session_id = ""
            self._foreground_session_platform = ""
            self._foreground_conversation_id = ""
        self._clear_foreground_rebind_candidate()

    def _maybe_reclassify_current_session_as_auxiliary_before_message_ingest(self) -> bool:
        """Defense-in-depth for host markers that arrive after session binding.

        Older Hermes Agent background-review forks seed ``_memory_write_origin``
        too late for ``on_session_start`` frame-walk detection. By the first
        message-writing entry point the marker is present on the running agent
        frame, so re-check only while the bound session is still empty. That
        keeps normal foreground session starts/resets writable and avoids
        reclassifying sessions after real data has already been stored.
        """
        session_id = str(self._session_id or "")
        if not session_id:
            return False
        if self._session_ignored or self._session_stateless or self._thread_context_stateless():
            return False
        if self._ingest_cursor > 0:
            return False
        try:
            stored_count = self._store.get_session_count(session_id)
        except Exception:
            logger.debug("LCM first-ingest auxiliary recheck count probe failed", exc_info=True)
            return False
        if stored_count != 0:
            return False
        if not self._in_process_auxiliary_caller_generation(session_id):
            return False

        self._mark_thread_context_stateless(session_id)
        self._restore_foreground_after_late_auxiliary_reclassification(session_id)
        logger.info(
            "LCM reclassified session %s as auxiliary at first ingest after bind-time detection missed",
            session_id,
        )
        return True

    def ingest(self, messages: List[Dict[str, Any]]) -> None:
        """Persist messages to the durable store every turn.

        Called by the post_llm_call plugin hook so messages land in LCM
        regardless of whether compression triggers — short WebUI
        conversations never hit the compression threshold and never
        expire like Telegram sessions do, so without this they'd never
        be ingested.

        Uses the same _ingest_messages cursor as compress(), so if
        compression runs later the same turn, already-ingested messages
        are skipped (no duplicates).
        """
        with self._storage_lifetime_lock:
            self._ingest_locked(messages)

    def _ingest_locked(self, messages: List[Dict[str, Any]]) -> None:
        """Run normal ingest inside the session/rollover lifetime boundary."""
        if self._maybe_reclassify_current_session_as_auxiliary_before_message_ingest():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            return
        if self._bypasses_lcm_context_management():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            return
        if self._session_id and messages:
            try:
                self._remember_lcm_normal_message_prefix(
                    self._session_id,
                    messages,
                    conversation_id=self._conversation_id,
                )
                self._ingest_messages(messages)
                self._record_ingest_success()
                self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
                logger.debug(
                    "Per-turn ingest OK: session=%s msgs=%d cursor=%d",
                    self._session_id, len(messages), self._ingest_cursor,
                )
            except Exception as e:
                self._record_ingest_failure("per-turn ingest()", e)

    def _is_retry_worthy_leaf_summary_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        retry_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "input too long",
            "request too large",
            "timed out",
            "timeout",
        )
        return any(marker in message for marker in retry_markers)

    def _next_leaf_rescue_chunk(
        self,
        current_chunk: List[Dict[str, Any]],
        current_source_tokens: int,
    ) -> List[Dict[str, Any]]:
        if len(current_chunk) <= 1:
            return []

        floor_tokens = self._effective_leaf_chunk_tokens()
        shrink_targets = [
            max(floor_tokens, int(current_source_tokens * 0.75)),
            max(floor_tokens, int(current_source_tokens * 0.50)),
        ]

        for target in shrink_targets:
            if target >= current_source_tokens:
                continue
            smaller = self._select_oldest_leaf_chunk(current_chunk, target)
            if smaller and len(smaller) < len(current_chunk):
                return smaller

        return current_chunk[:-1]

    def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
        timeout_seconds: float | None = None,
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        attempt_number = 0

        while attempt_chunk and attempt_number < max_attempts:
            attempt_number += 1
            source_tokens = count_messages_tokens(attempt_chunk)
            serialized = self._serialize_messages(attempt_chunk)
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)

            try:
                summary_text, level = summarize_with_escalation(
                    text=serialized,
                    source_tokens=source_tokens,
                    token_budget=token_budget,
                    depth=0,
                    model=self._config.summary_model,
                    fallback_models=self._config.summary_fallback_models,
                    circuit_breaker=self._summary_circuit_breaker,
                    spend_guard=self._summary_spend_guard,
                    timeout=(
                        min(self._config.summary_timeout_ms / 1000, timeout_seconds)
                        if timeout_seconds is not None
                        else self._config.summary_timeout_ms / 1000
                    ),
                    l2_budget_ratio=self._config.l2_budget_ratio,
                    l3_truncate_tokens=self._config.l3_truncate_tokens,
                    focus_topic=focus_topic or "",
                    custom_instructions=self._config.custom_instructions,
                )
                return attempt_chunk, source_tokens, summary_text, level, attempt_number
            except Exception as exc:
                if attempt_number >= max_attempts or not self._is_retry_worthy_leaf_summary_error(exc):
                    raise
                smaller_chunk = self._next_leaf_rescue_chunk(attempt_chunk, source_tokens)
                if not smaller_chunk or len(smaller_chunk) >= len(attempt_chunk):
                    raise
                logger.warning(
                    "LCM leaf summarization retrying with smaller oldest chunk after retry-worthy failure: %s (attempt %d/%d, %d→%d messages)",
                    exc,
                    attempt_number,
                    max_attempts,
                    len(attempt_chunk),
                    len(smaller_chunk),
                )
                attempt_chunk = smaller_chunk

        raise RuntimeError("adaptive leaf rescue exhausted without a valid chunk")

    # -- ContextEngine optional methods ------------------------------------

    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._lcm_session_last_conversation_id[session_id] = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id
        self._register_active_engine_binding()
        if not self._session_ignored and not self._session_stateless:
            self._remember_foreground_rebind_candidate(session_id)
            self._lcm_session_last_normal_conversation_id[session_id] = state.conversation_id
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
            self._foreground_conversation_id = state.conversation_id

        # Garbage-collect empty lifecycle rows when the table exceeds threshold.
        # Gateway restarts, ephemeral cron ticks, and crash-loops all create
        # lifecycle rows that never ingest data — prune them here so they
        # don't accumulate forever.
        if (
            self._config.empty_lifecycle_gc_enabled
            and self._lifecycle.row_count() > self._config.empty_lifecycle_gc_threshold
        ):
            protected = {str(self._session_id)} if self._session_id else None
            max_age = self._config.empty_lifecycle_gc_max_age_hours
            try:
                deleted = self._lifecycle.prune_empty_sessions(
                    protected_session_ids=protected,
                    max_age_hours=max_age,
                )
            except Exception:
                deleted = 0
            if deleted:
                logger.info(
                    "LCM pruned %d lifecycle rows with zero stored data "
                    "(table exceeded threshold of %d rows)",
                    deleted,
                    self._config.empty_lifecycle_gc_threshold,
                )

        # Reap any stale "preparing" batches from a crashed/restarted session.
        if self._conversation_id:
            try:
                reaped = self._frontier.reap_stale_preparing(self._conversation_id)
                if reaped:
                    logger.info("LCM reaped %d stale preparing batches on session start", reaped)
            except Exception:
                logger.debug("LCM stale preparing reap failed", exc_info=True)
            # Repair itemless active frontier tips left by older builds (issue #4).
            try:
                repaired = self.reconcile_itemless_frontier_generations(
                    self._conversation_id
                )
                if repaired:
                    logger.info(
                        "LCM reconciled %d itemless frontier generation(s) on session start",
                        repaired,
                    )
            except Exception:
                logger.debug("LCM frontier item reconciliation failed", exc_info=True)

    def _register_active_engine_binding(self) -> None:
        session_id = str(self._session_id or "")
        conversation_id = str(self._conversation_id or "")
        if not session_id:
            return
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(
                self,
                keep_session_id=session_id,
                keep_conversation_id=conversation_id,
            )
            _ACTIVE_ENGINES_BY_SESSION_ID[session_id] = self
            if conversation_id:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID[conversation_id] = self

    def _unregister_active_engine_binding(self) -> None:
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(self)

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _has_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> bool:
        with self._auxiliary_session_lock:
            if session_id not in self._lcm_bypass_lineage_session_ids:
                return False
            if platform is None:
                return True
            platforms = self._lcm_bypass_lineage_platforms.get(session_id) or set()
            return not platforms or platform in platforms

    def _mark_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> None:
        if not session_id:
            return
        platform = self._session_platform if platform is None else str(platform or "")
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.add(session_id)
            self._lcm_bypass_lineage_platforms.setdefault(session_id, set()).add(platform)
            self._lcm_session_last_platform[session_id] = platform
            self._lcm_session_last_bypassed[session_id] = True

    def _unmark_lcm_bypass_lineage_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.discard(session_id)
            self._lcm_bypass_lineage_platforms.pop(session_id, None)

    def _handoff_lcm_bypass_lineage(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        new_platform: str = "",
    ) -> None:
        with self._auxiliary_session_lock:
            if old_session_id:
                self._lcm_bypass_lineage_session_ids.add(old_session_id)
            if new_session_id:
                new_platform = str(new_platform or "")
                self._lcm_bypass_lineage_session_ids.add(new_session_id)
                self._lcm_bypass_lineage_platforms.setdefault(new_session_id, set()).add(new_platform)
                self._lcm_session_last_platform[new_session_id] = new_platform
                self._lcm_session_last_bypassed[new_session_id] = True

    def _compression_boundary_from_lcm_bypassed_session(self, old_session_id: str) -> bool:
        if not old_session_id:
            return False
        if old_session_id in self._lcm_session_last_bypassed:
            return bool(self._lcm_session_last_bypassed.get(old_session_id))
        if old_session_id == self._session_id:
            return bool(
                self._bypasses_lcm_context_management()
                or self._session_id_matches_lcm_bypass_filters(
                    old_session_id,
                    platform=self._session_platform,
                )
            )
        return bool(
            self._has_lcm_bypass_lineage_session(old_session_id)
            or self._session_id_matches_lcm_bypass_filters(old_session_id)
        )

    def _get_allowed_hermes_base(self) -> Path | None:
        """Get the allowed base directory for hermes_home, or None if not restricted."""
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            return Path(env_base).expanduser().resolve()
        return None  # No restriction when env var not set

    def _state_db_path(self, kwargs: Dict[str, Any] | None = None) -> Path:
        kwargs = kwargs or {}
        hermes_home = str(kwargs.get("hermes_home") or self._hermes_home or "")
        if hermes_home:
            return _enforce_state_db_containment(
                Path(hermes_home) / "state.db",
                description=f"hermes_home {hermes_home}",
            )
        db_path = Path(self._store.db_path)
        return _enforce_state_db_containment(
            db_path.parent / "state.db",
            description=f"state database fallback from LCM database {db_path}",
        )

    def _clear_pending_reset_boundary(self) -> None:
        self._pending_reset_session_id = ""
        self._pending_reset_conversation_id = ""
        self._pending_reset_frontier_store_id = 0

    def _finalize_pending_reset_boundary(self, session_id: str) -> None:
        if not self._pending_reset_session_id:
            return
        if self._pending_reset_session_id != session_id:
            self._clear_pending_reset_boundary()
            return
        if not self._pending_reset_conversation_id:
            self._clear_pending_reset_boundary()
            return
        state = self._lifecycle.get_by_conversation(self._pending_reset_conversation_id)
        frontier_store_id = self._pending_reset_frontier_store_id
        if state is not None and state.current_session_id == session_id:
            frontier_store_id = max(
                frontier_store_id,
                int(state.current_frontier_store_id or 0),
            )
        self._lifecycle.finalize_session(
            self._pending_reset_conversation_id,
            self._pending_reset_session_id,
            frontier_store_id=frontier_store_id,
        )
        self._clear_pending_reset_boundary()

    def _effective_fresh_tail_count(self) -> int:
        policy = getattr(self, "_compaction_policy", None)
        if policy is not None:
            return max(0, int(policy.fresh_tail_count))
        return max(0, int(self._config.fresh_tail_count or 0))

    def _effective_fresh_tail_max_tokens(self) -> int:
        policy = getattr(self, "_compaction_policy", None)
        if policy is not None:
            return max(0, int(policy.fresh_tail_max_tokens))
        return max(0, int(self._config.fresh_tail_max_tokens or 0))

    def _effective_policy_value(self, name: str):
        policy = getattr(self, "_compaction_policy", None)
        if policy is not None and hasattr(policy, name):
            return getattr(policy, name)
        return getattr(self._config, name)

    def _effective_leaf_chunk_tokens(self) -> int:
        return max(1, int(self._effective_policy_value("leaf_chunk_tokens") or 1))

    def _effective_dynamic_leaf_chunk_enabled(self) -> bool:
        return bool(self._effective_policy_value("dynamic_leaf_chunk_enabled"))

    def _effective_dynamic_leaf_chunk_max(self) -> int:
        return max(
            self._effective_leaf_chunk_tokens(),
            int(self._effective_policy_value("dynamic_leaf_chunk_max") or 0),
        )

    def _effective_condensation_fanin(self) -> int:
        return max(2, int(self._effective_policy_value("condensation_fanin") or 2))

    def _effective_condensation_min_fanin(self) -> int:
        return max(2, int(self._effective_policy_value("condensation_min_fanin") or 2))

    def _effective_incremental_max_depth(self) -> int:
        return int(self._effective_policy_value("incremental_max_depth"))

    def _effective_cache_friendly_condensation_enabled(self) -> bool:
        return bool(
            self._effective_policy_value("cache_friendly_condensation_enabled")
        )

    def _effective_full_sweep_compaction_enabled(self) -> bool:
        return bool(self._effective_policy_value("full_sweep_compaction_enabled"))

    def _effective_summary_prefix_target_tokens(self) -> int:
        return max(
            0, int(self._effective_policy_value("summary_prefix_target_tokens") or 0)
        )

    def _fresh_tail_start(self, messages: List[Dict[str, Any]]) -> int:
        """Resolve the shared suffix protected by count and token rails."""
        n = len(messages)
        leading = self._leading_anchor_count(messages)
        count_limit = self._effective_fresh_tail_count()
        token_limit = self._effective_fresh_tail_max_tokens()
        count_start = max(leading, n - count_limit) if count_limit > 0 else n

        token_start = leading
        if token_limit > 0:
            token_start = n
            token_total = 0
            for index in range(n - 1, leading - 1, -1):
                message_tokens = count_message_tokens(messages[index])
                if token_total + message_tokens > token_limit:
                    break
                token_total += message_tokens
                token_start = index

        start = max(count_start, token_start)
        overflow_reason = ""
        newest_user_index = next(
            (
                index
                for index in range(n - 1, leading - 1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        tail_enabled = count_limit > 0 or token_limit > 0
        if (
            tail_enabled
            and newest_user_index == n - 1
            and newest_user_index < start
        ):
            start = newest_user_index
            overflow_reason = "newest-user-exceeds-token-cap"

        newest_anchor_index = next(
            (
                index
                for index in range(n - 1, leading - 1, -1)
                if self._is_preserved_todo_context_message(messages[index])
                or bool(self._preserved_objective_context_content(messages[index]))
            ),
            None,
        )
        if (
            tail_enabled
            and newest_anchor_index == n - 1
            and newest_anchor_index < start
        ):
            start = newest_anchor_index
            overflow_reason = "protected-anchor-exceeds-tail-boundary"

        selected = messages[start:]
        selected_tokens = count_messages_tokens(selected)
        overflow = bool(token_limit > 0 and selected_tokens > token_limit)
        if overflow and not overflow_reason:
            overflow_reason = "protected-suffix-exceeds-token-cap"
        boundary_reason = (
            "token-cap"
            if token_limit > 0 and token_start > count_start
            else "count-cap" if count_limit > 0 else "no-count-tail"
        )
        self._last_fresh_tail_selection = {
            "count_limit": count_limit,
            "token_limit": token_limit,
            "count_start": count_start,
            "token_start": token_start,
            "selected_start": start,
            "selected_count": len(selected),
            "selected_tokens": selected_tokens,
            "boundary_reason": boundary_reason,
            "overflow": overflow,
            "overflow_reason": overflow_reason,
            "tool_repair_stub_required": bool(
                selected and selected[0].get("role") == "tool"
            ),
        }
        return start

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return []
        return messages[leading_anchor_count:fresh_tail_start]

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        """Return the number of non-compactable leading messages.

        Only the system prompt is a safe permanent anchor. Hermes gateway
        sessions can begin with a user message when core passes conversation
        history without a system prompt; preserving that first user turn as raw
        active context lets stale requests look current after later compaction.
        """
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

    def _raw_backlog_tokens(self, messages: List[Dict[str, Any]]) -> int:
        backlog = self._raw_backlog_messages(messages)
        if not backlog:
            return 0
        return count_messages_tokens(backlog)

    def _raw_backlog_threshold(self, raw_tokens: int) -> int:
        if self._effective_dynamic_leaf_chunk_enabled():
            return self._working_leaf_chunk_tokens(raw_tokens)
        return self._effective_leaf_chunk_tokens()

    def _has_raw_backlog_debt(self) -> bool:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return False
        state = self._lifecycle.get_by_conversation(self._conversation_id)
        return bool(state and state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0)

    def _budget_pressure_ratio(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> float | None:
        if self.context_length <= 0:
            return None
        token_count: int | None = None
        if observed_tokens is not None and observed_tokens > 0:
            token_count = observed_tokens
        elif messages is not None:
            token_count = count_messages_tokens(messages)
        elif self.last_prompt_tokens > 0:
            token_count = self.last_prompt_tokens
        if token_count is None or token_count <= 0:
            return None
        return token_count / self.context_length

    def _critical_budget_pressure_reached(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> bool:
        threshold = self._config.critical_budget_pressure_ratio
        if threshold <= 0:
            return False
        pressure = self._budget_pressure_ratio(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        return pressure is not None and pressure >= threshold

    def _should_run_deferred_maintenance(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> bool:
        if not self._has_raw_backlog_debt():
            return False
        raw_tokens = self._raw_backlog_tokens(messages)
        if raw_tokens <= 0:
            return False
        if raw_tokens >= self._raw_backlog_threshold(raw_tokens):
            return True
        return self._critical_budget_pressure_reached(
            observed_tokens=observed_tokens,
            messages=messages,
        )

    def _refresh_raw_backlog_debt(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return
        raw_tokens = self._raw_backlog_tokens(messages)
        threshold = self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
        keep_under_critical_pressure = (
            raw_tokens > 0
            and self._has_raw_backlog_debt()
            and self._critical_budget_pressure_reached(
                observed_tokens=observed_tokens,
                messages=messages,
            )
        )
        if raw_tokens > 0 and (raw_tokens >= threshold or keep_under_critical_pressure):
            self._lifecycle.record_debt(
                self._conversation_id,
                kind="raw_backlog",
                size_estimate=raw_tokens,
            )
            return
        if self._has_raw_backlog_debt():
            self._lifecycle.clear_debt(self._conversation_id)

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        self._refresh_session_filters()
        if self.context_length > 0 and self.model:
            self._resolve_live_compaction_policy()
        # Hold the foreground view stable when the new binding is a side
        # channel (cron tick inside the gateway process, debug probe, etc.).
        # Tools that report "current session" to operators must keep pointing
        # at the real foreground rather than the ignored/stateless session
        # that just stole _session_id. Lifecycle paths still read _session_id
        # directly so cron's compress short-circuits correctly via the
        # _session_ignored / _session_stateless gates.
        if not self._session_ignored and not self._session_stateless:
            self._remember_foreground_rebind_candidate(session_id)
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]

        update_model_is_authoritative = (
            self._context_length_source == "update_model"
            and self._update_model_pending_session_start
        )

        # Pick up context_length from kwargs if provided, but do not let stale
        # session metadata undo the authoritative runtime update_model() call.
        # Hermes Agent calls update_model() with the resolver output before it
        # binds a fresh agent/session.  Older or buggy host paths can still pass
        # a context_length copied from the previously bound runtime; treating
        # that as authoritative makes /model switches keep compressing against
        # the old model window.
        if "context_length" in kwargs:
            incoming_context_length = kwargs["context_length"]
            try:
                parsed_context_length = int(incoming_context_length)
            except (TypeError, ValueError):
                logger.debug(
                    "LCM ignored invalid session-start context_length: %r",
                    incoming_context_length,
                )
                self._update_model_pending_session_start = False
                return
            if parsed_context_length <= 0:
                if update_model_is_authoritative:
                    if self._session_metadata_matches_active_runtime(
                        kwargs,
                        ignore_empty_optional=True,
                    ):
                        logger.debug(
                            "LCM ignored missing session-start context_length=%r for model=%s; active update_model context_length=%s",
                            incoming_context_length,
                            self.model or str(kwargs.get("model") or ""),
                            self.context_length,
                        )
                    else:
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                    self._update_model_pending_session_start = False
                    return
                self._set_context_length(parsed_context_length, source="session_start")
                update_model_is_authoritative = False
            else:
                if (
                    update_model_is_authoritative
                    and parsed_context_length not in {self.context_length, self.raw_context_length}
                ):
                    logger.warning(
                        "LCM ignored stale session-start context_length=%s for model=%s; active update_model raw_context_length=%s effective_context_length=%s",
                        parsed_context_length,
                        self.model or str(kwargs.get("model") or ""),
                        self.raw_context_length,
                        self.context_length,
                    )
                    self._update_model_pending_session_start = False
                    return
                if update_model_is_authoritative:
                    if not self._session_metadata_matches_active_runtime(kwargs):
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                        self._update_model_pending_session_start = False
                        return
                else:
                    self._set_context_length(
                        parsed_context_length,
                        source="session_start",
                        model=str(kwargs.get("model") or self.model),
                        provider=str(kwargs.get("provider") or self.provider),
                    )
                    update_model_is_authoritative = False
        if (
            update_model_is_authoritative
            and not self._session_metadata_matches_active_runtime(kwargs)
        ):
            logger.warning(
                "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                str(kwargs.get("model") or ""),
                self.model,
            )
            self._update_model_pending_session_start = False
            return
        if "model" in kwargs:
            self.model = str(kwargs.get("model") or "")
        route_affects_context = "model" in kwargs or "provider" in kwargs
        for key in ("base_url", "api_key", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        if (
            "context_length" not in kwargs
            and route_affects_context
            and (self.raw_context_length or self.context_length)
        ):
            self._set_context_length(
                self.raw_context_length or self.context_length,
                source=self._context_length_source or "session_start",
                model=self.model,
                provider=self.provider,
            )
        if self.context_length > 0 and self.model:
            self._resolve_live_compaction_policy()
        self._update_model_pending_session_start = False

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        requested_conversation_id = kwargs.get("conversation_id")
        session_state = self._lifecycle.get_by_session(old_session_id)
        conversation_state = self._lifecycle.get_by_conversation(old_session_id)

        def _state_conversation_matches(state: Any) -> bool:
            return bool(
                state
                and (
                    not requested_conversation_id
                    or state.conversation_id == requested_conversation_id
                )
            )

        def _has_summary_nodes(candidate_session_id: str | None) -> bool:
            return bool(candidate_session_id and self._dag.get_session_nodes(candidate_session_id))

        def _host_source_from_conversation_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.conversation_id == old_session_id
                and state.current_session_id
                and _has_summary_nodes(state.current_session_id)
            ):
                return state.current_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id
                and _has_summary_nodes(state.last_finalized_session_id)
            ):
                return state.last_finalized_session_id, state
            return "", None

        def _host_source_from_session_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id == old_session_id
                and _has_summary_nodes(old_session_id)
            ):
                return old_session_id, state
            return "", None

        host_source_session_id, host_source_state = _host_source_from_conversation_state(
            conversation_state
        )
        if not host_source_session_id:
            host_source_session_id, host_source_state = _host_source_from_session_state(
                session_state
            )

        source_session_id = host_source_session_id or old_session_id
        source_state = host_source_state or session_state

        if previous_session_id and previous_session_id != old_session_id:
            # Hermes passes the session that actually crossed the compression
            # boundary as old_session_id. A different bound session can be a
            # short-lived subagent/cron/WebUI side channel that ran after the
            # foreground compaction. Prefer the host-authoritative source when
            # durable lifecycle + DAG evidence proves it belongs to LCM, then
            # fall back to the older bound-session recovery path. When the host
            # old_session_id is the durable conversation id, use that row's
            # current/finalized LCM source instead of unrelated auxiliary rows
            # where the id appears only as last_finalized_session_id.
            if host_source_session_id:
                logger.warning(
                    "LCM compression boundary using host old_session_id %s as carry-over source=%s despite bound session drift=%s",
                    old_session_id,
                    host_source_session_id,
                    previous_session_id,
                )
            else:
                bound_state = self._lifecycle.get_by_session(previous_session_id)
                bound_conversation_matches = bool(
                    bound_state
                    and (not self._conversation_id or bound_state.conversation_id == self._conversation_id)
                    and (
                        not requested_conversation_id
                        or bound_state.conversation_id == requested_conversation_id
                    )
                )
                bound_is_active_source = bool(
                    bound_state and bound_state.current_session_id == previous_session_id
                )
                bound_is_finalized_source = bool(
                    bound_state
                    and bound_state.current_session_id is None
                    and bound_state.last_finalized_session_id == previous_session_id
                )
                bound_has_summary_nodes = bool(self._dag.get_session_nodes(previous_session_id))
                if (
                    bound_conversation_matches
                    and (bound_is_active_source or bound_is_finalized_source)
                    and bound_has_summary_nodes
                ):
                    source_session_id = previous_session_id
                    source_state = bound_state
                    logger.warning(
                        "LCM compression boundary using bound session %s as carry-over source; host old_session_id=%s does not match",
                        previous_session_id,
                        old_session_id,
                    )
                else:
                    # Fallback: sibling chain with zero-DAG parent.
                    # When stale old_session_id has no DAG nodes AND the
                    # bound session belongs to a different conversation_id
                    # but shares the same last_finalized_session_id
                    # (parent) — prefer the bound session despite the
                    # conversation_id mismatch. This handles the lifecycle
                    # fork case where two sessions on the same channel
                    # received different conversation_ids.
                    bound_shares_parent_with_host = bool(
                        bound_state
                        and bound_state.last_finalized_session_id == old_session_id
                    )
                    host_has_no_dag = not bool(
                        self._dag.get_session_nodes(old_session_id)
                    )
                    if (
                        bound_shares_parent_with_host
                        and host_has_no_dag
                        and (bound_is_active_source or bound_is_finalized_source)
                        and bound_has_summary_nodes
                    ):
                        source_session_id = previous_session_id
                        source_state = bound_state
                        logger.warning(
                            "LCM compression boundary using bound session %s on sibling chain as carry-over source; host old_session_id=%s has zero DAG, parent=%s matches",
                            previous_session_id,
                            old_session_id,
                            bound_state.last_finalized_session_id,
                        )
                    else:
                        source_session_id = ""
                        source_state = None

        conversation_id = (
            (source_state.conversation_id if source_state else None)
            or kwargs.get("conversation_id")
            or self._conversation_id
            or source_session_id
            or old_session_id
            or session_id
        )
        process_local_frontier = (
            int(self._last_compacted_store_id or 0)
            if source_session_id and previous_session_id == source_session_id
            else 0
        )
        pending_reset_frontier = int(
            self._pending_reset_frontier_store_id
            if self._pending_reset_session_id
            and self._pending_reset_session_id == source_session_id
            else 0
        )
        frontier = max(
            process_local_frontier,
            int(source_state.current_frontier_store_id if source_state else 0),
            int(source_state.last_finalized_frontier_store_id if source_state else 0),
            pending_reset_frontier,
        )
        can_reassign = bool(
            source_session_id
            and session_id
            and source_session_id != session_id
        )
        boundary_placeholder_budget = {}
        boundary_placeholder_ordinals: dict[str, set[int]] = {}
        if can_reassign:
            if previous_session_id == source_session_id:
                boundary_placeholder_budget = self._active_replay_generated_placeholder_digest_budget()
                boundary_placeholder_ordinals = self._generated_placeholder_digest_ordinals_for_active_replay(
                    self._last_active_replay_messages
                )
            if not boundary_placeholder_budget:
                boundary_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_counts",
                        source_session_id,
                    )
                )
            if not boundary_placeholder_ordinals:
                boundary_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_ordinals",
                        source_session_id,
                    )
                )
            for digest, ordinals in boundary_placeholder_ordinals.items():
                boundary_placeholder_budget[digest] = max(
                    boundary_placeholder_budget.get(digest, 0),
                    len(ordinals),
                )
            self._compression_boundary_stored_placeholder_digest_counts = (
                self._stored_active_replay_placeholder_digest_counts(
                    source_session_id,
                    after_store_id=frontier,
                )
            )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                source_session_id,
                frontier_store_id=frontier,
            )
            self._copy_generated_ignore_hashes_to_session(
                source_session_id,
                session_id,
                copy_dependent_content=True,
                source_frontier_store_id=frontier,
            )
            self._write_generated_ignored_placeholder_hash_counts(
                boundary_placeholder_budget,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_counts",
                    session_id,
                ),
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                boundary_placeholder_ordinals,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_ordinals",
                    session_id,
                ),
            )
            # Compression rollover carries derived context forward, but raw
            # messages remain owned by the session that produced them. Moving
            # raw rows here makes session-scoped transcript recovery report the
            # old/child session as missing even though its payload was only
            # reassigned to the next compression segment.
            moved_nodes = self._reassign_canonical_session_state(
                conversation_id,
                source_session_id,
                session_id,
            )
            logger.debug(
                "LCM compression boundary continued %s -> %s: carried %d DAG nodes; preserved raw message ownership",
                source_session_id,
                session_id,
                moved_nodes,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
            self._last_boundary_skip_time = time.time()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            self._schedule_ingest_cursor_reconciliation()
            self._clear_pending_reset_boundary()
            self._log_session_filter_diagnostics()
            return

        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(session_id, conversation_id=conversation_id)
        self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
        if frontier > 0:
            state = self._lifecycle.advance_frontier(
                self._conversation_id,
                session_id,
                frontier,
            )
            if state is not None:
                self._last_compacted_store_id = state.current_frontier_store_id
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = can_reassign
        self._compression_boundary_active_placeholder_digest_budget = boundary_placeholder_budget
        self._compression_boundary_active_placeholder_digest_ordinals = boundary_placeholder_ordinals
        self._log_session_filter_diagnostics()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        # Entire storage-consuming start path holds the lifetime lock so it
        # serializes with rebind/shutdown and cannot observe half-closed helpers.
        with self._storage_lifetime_lock:
            self._on_session_start_locked(session_id, **kwargs)

    def _on_session_start_locked(self, session_id: str, **kwargs) -> None:
        if "hermes_home" in kwargs:
            self._rebind_storage_for_home(str(kwargs.get("hermes_home") or ""))

        if self._storage_lifetime_state != "bound":
            logger.warning(
                "LCM refusing on_session_start while storage is %s",
                self._storage_lifetime_state,
            )
            return

        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        previous_session_id = self._session_id
        self._lcm_current_start_allows_bypass_lineage = False
        requested_platform = str(kwargs.get("platform") or self._session_platform or "")
        pre_reset_preserve_ambiguous_no_frame_old_session = False
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            pre_reset_preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and new_session_auxiliary_parent != old_session_id
            )
        if self._host_fallback_compressor is not None and (
            self._host_fallback_session_id != session_id or requested_platform != self._session_platform
        ) and not (
            pre_reset_preserve_ambiguous_no_frame_old_session
            and self._host_fallback_session_id == old_session_id
        ):
            compressor = self._host_fallback_compressor
            fallback_session_id = self._host_fallback_session_id or previous_session_id
            on_session_end = getattr(compressor, "on_session_end", None)
            if callable(on_session_end) and fallback_session_id:
                try:
                    on_session_end(fallback_session_id, [])
                except Exception:
                    logger.debug("LCM host fallback compressor session-start reset failed", exc_info=True)
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_is_suppressed_foreground = self._auxiliary_lineage_suppressed_as_foreground(
                old_session_id
            )
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_is_auxiliary_continuation = new_session_auxiliary_parent == old_session_id
            preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and not new_session_is_auxiliary_continuation
            )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            if (
                self._has_auxiliary_lineage_session(old_session_id)
                and not old_session_auxiliary_generation_is_stale
                and (
                    old_session_id != self._session_id
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
                and (
                    not old_session_is_suppressed_foreground
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
            ):
                self._handoff_auxiliary_session(
                    old_session_id,
                    session_id,
                    preserve_old_session=(
                        old_session_id == self._session_id
                        or preserve_ambiguous_no_frame_old_session
                    ),
                    preserve_old_foreground_marker=old_session_is_suppressed_foreground,
                )
                logger.info(
                    "LCM auxiliary session %s compressed to %s — keeping boundary stateless",
                    old_session_id,
                    session_id,
                )
                return
            if self._compression_boundary_from_lcm_bypassed_session(old_session_id):
                self._handoff_lcm_bypass_lineage(
                    old_session_id,
                    session_id,
                    new_platform=str(kwargs.get("platform") or ""),
                )
                self._clear_thread_context_stateless()
                if previous_session_id and previous_session_id != session_id:
                    self._finalize_pending_reset_boundary(previous_session_id)
                    self._reset_session_scoped_runtime_state()
                else:
                    self._clear_pending_reset_boundary()
                    self._ingest_cursor = 0
                    self._last_compacted_store_id = 0
                    self._last_overflow_recovery_failed = False
                    self._last_condensation_suppressed_reason = ""
                self._lcm_current_start_allows_bypass_lineage = True
                self._apply_session_start_metadata(session_id, kwargs)
                self._bind_lifecycle_state(
                    session_id,
                    conversation_id=kwargs.get("conversation_id"),
                )
                self._schedule_ingest_cursor_reconciliation()
                self._log_session_filter_diagnostics()
                logger.info(
                    "LCM compression boundary %s -> %s stayed stateless because the source session bypasses LCM storage",
                    old_session_id,
                    session_id,
                )
                return
            self._clear_thread_context_stateless()
            self._continue_compression_boundary(session_id, old_session_id, kwargs)
            return

        if self._is_live_auxiliary_child_session(session_id, previous_session_id, kwargs):
            explicit_parent_id = str(kwargs.get("parent_session_id") or "")
            preserve_foreground_reuse_marker = bool(
                (
                    explicit_parent_id
                    and self._lcm_session_last_bypassed.get(explicit_parent_id)
                )
                or self._lcm_session_last_normal_conversation_id.get(session_id)
            )
            if preserve_foreground_reuse_marker:
                if self._lcm_session_last_normal_conversation_id.get(session_id):
                    with self._auxiliary_session_lock:
                        self._auxiliary_foreground_reused_session_ids.add(session_id)
                self._mark_thread_context_stateless(
                    session_id,
                    preserve_foreground_reuse_marker=True,
                )
            else:
                self._register_auxiliary_session(session_id)
            logger.info(
                "LCM session %s is a live child of bound session %s — treating it as auxiliary/stateless",
                session_id,
                previous_session_id,
            )
            return
        start_platform = str(kwargs.get("platform") or "")
        side_channel_rebind = self._session_id_matches_lcm_bypass_filters(
            session_id,
            platform=start_platform,
        ) or self._has_lcm_bypass_lineage_session(session_id, platform=start_platform)
        self._unmark_thread_context_auxiliary_session(
            session_id,
            suppress_as_foreground_reuse=not side_channel_rebind,
        )
        self._clear_thread_context_stateless()
        if previous_session_id and previous_session_id != session_id:
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
        else:
            self._clear_pending_reset_boundary()
            self._ingest_cursor = 0
            self._last_compacted_store_id = 0
            self._last_overflow_recovery_failed = False
            self._last_condensation_suppressed_reason = ""
        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(
            session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        self._schedule_ingest_cursor_reconciliation()
        self._log_session_filter_diagnostics()
        self._start_async_worker()

    def _session_end_matches_current_store_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        prefix_count = self._session_end_store_prefix_count(session_id, messages)
        return prefix_count is not None and prefix_count > 0

    def _session_end_prefix_compare_value(
        self,
        value: Any,
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: self._session_end_prefix_compare_value(
                    child, session_id=session_id, read_budget=read_budget
                )
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._session_end_prefix_compare_value(
                    child, session_id=session_id, read_budget=read_budget
                )
                for child in value
            ]
        if not isinstance(value, str):
            return value

        text = restore_ingest_payload_placeholders(
            value,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
            read_budget=read_budget,
            budget_label="rollover prefix ingest payload",
            max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
            max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
        )
        stripped = text.strip()
        ingest_refs = extract_ingest_externalized_refs(stripped)
        if (
            len(ingest_refs) == 1
            and stripped.startswith("[Externalized LCM ingest payload:")
            and stripped.endswith("]")
        ):
            payload = load_externalized_payload(
                ingest_refs[0],
                config=self._config,
                hermes_home=self._hermes_home,
                read_budget=read_budget,
                budget_label="rollover prefix",
                max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content

        if is_externalized_placeholder(stripped):
            ref = extract_externalized_ref(stripped)
            payload = load_externalized_payload(
                ref or "",
                config=self._config,
                hermes_home=self._hermes_home,
                read_budget=read_budget,
                budget_label="rollover prefix",
                max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content
        return text

    def _session_end_prefix_compare_content(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> str:
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id,
            read_budget=read_budget,
        )
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> str:
        tool_calls = self._session_end_prefix_compare_value(
            (message or {}).get("tool_calls"),
            session_id=session_id,
            read_budget=read_budget,
        )
        tool_calls = redact_sensitive_value(
            tool_calls,
            self._config,
            parse_json_strings=True,
        )
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        return json.dumps(
            tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _session_end_prefix_compare_identity(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> tuple[str, str, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(
                message, session_id=session_id, read_budget=read_budget
            ),
            str((message or {}).get("tool_call_id") or ""),
            str((message or {}).get("tool_name") or ""),
            self._session_end_prefix_compare_tool_calls(
                message, session_id=session_id, read_budget=read_budget
            ),
        )

    @staticmethod
    def _guard_rollover_nested_representation(
        value: Any,
        *,
        read_budget: dict[str, float | int],
    ) -> None:
        if time.monotonic() >= float(read_budget["deadline_at"]):
            raise RuntimeError("rollover prefix deadline exceeded")
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            depth = 0
            items = 0
            in_string = False
            escaped = False
            for index, character in enumerate(value):
                if index % 4096 == 0 and time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("rollover prefix deadline exceeded")
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                    continue
                if character == '"':
                    in_string = True
                elif character in "[{":
                    depth += 1
                    items += 1
                    if depth > _PUBLICATION_LOCKED_MAX_NESTED_DEPTH:
                        raise RuntimeError("rollover prefix nested-depth bound exceeded")
                elif character in "]}":
                    depth = max(0, depth - 1)
                elif character in ",:":
                    items += 1
                if items > _PUBLICATION_LOCKED_MAX_NESTED_ITEMS:
                    raise RuntimeError("rollover prefix nested-item bound exceeded")
            return
        pending = [(value, 0)]
        items = 0
        while pending:
            current, depth = pending.pop()
            items += 1
            if items > _PUBLICATION_LOCKED_MAX_NESTED_ITEMS:
                raise RuntimeError("rollover prefix nested-item bound exceeded")
            if depth > _PUBLICATION_LOCKED_MAX_NESTED_DEPTH:
                raise RuntimeError("rollover prefix nested-depth bound exceeded")
            if isinstance(current, dict):
                pending.extend((key, depth + 1) for key in current)
                pending.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, (list, tuple)):
                pending.extend((item, depth + 1) for item in current)

    @classmethod
    def _bounded_rollover_active_encoded_bytes(
        cls,
        value: Any,
        *,
        read_budget: dict[str, float | int],
        remaining_bytes: int | None = None,
    ) -> int:
        """Count JSON bytes incrementally before redaction or placeholder IO."""
        cls._guard_rollover_nested_representation(value, read_budget=read_budget)
        remaining = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
        if remaining_bytes is not None:
            remaining = min(remaining, max(0, int(remaining_bytes)))
        encoded_bytes = 0
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        for index, chunk in enumerate(encoder.iterencode(value)):
            if index % 64 == 0 and time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("rollover prefix deadline exceeded")
            encoded_bytes += len(chunk.encode("utf-8", errors="replace"))
            if encoded_bytes > remaining:
                raise RuntimeError("rollover prefix serialized byte bound exceeded")
        return encoded_bytes

    def _session_end_store_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
        raise_on_read_error: bool = False,
        read_budget: dict[str, float | int] | None = None,
        conn: sqlite3.Connection | None = None,
        allow_durable_interleaving: bool = False,
    ) -> Optional[int]:
        if read_budget is None:
            read_budget = self._new_locked_publication_read_budget()
        try:
            compared = 0
            last_store_id = 0
            fields = ("role", "content", "tool_call_id", "tool_name", "tool_calls")
            field_limit = min(
                _PUBLICATION_LOCKED_MAX_FIELD_BYTES,
                int(read_budget["max_bytes"]),
            )
            while True:
                if time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("rollover prefix deadline exceeded")
                remaining_rows = int(read_budget["max_rows"]) - int(read_budget["rows"])
                remaining_bytes = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
                if remaining_rows <= 0:
                    raise RuntimeError("rollover prefix row bound exceeded")
                if remaining_bytes <= 64:
                    raise RuntimeError("rollover prefix serialized byte bound exceeded")
                max_row_bytes = min(
                    int(read_budget["max_bytes"]),
                    (len(fields) * field_limit) + 64,
                )
                byte_sized_rows = max(1, remaining_bytes // max(1, max_row_bytes))
                page_limit = min(
                    _PUBLICATION_LOCKED_QUERY_BATCH,
                    remaining_rows + 1,
                    byte_sized_rows,
                )
                row_payload_cap = max(
                    0, remaining_bytes // max(1, page_limit) - 64
                )
                lengths = [
                    f"COALESCE(length(CAST({field} AS BLOB)), 0)"
                    for field in fields
                ]
                total_length = " + ".join(lengths)
                guard = " AND ".join(
                    [f"({total_length}) <= ?"]
                    + [f"{length} <= ?" for length in lengths]
                )
                bounded = [
                    f"CASE WHEN {guard} THEN "
                    f"CASE WHEN {field} IS NULL THEN NULL ELSE "
                    f"substr(CAST({field} AS TEXT), 1, {field_limit + 1}) END "
                    "ELSE NULL END"
                    for field in fields
                ]
                guard_params = (row_payload_cap, *(field_limit for _ in fields))
                where = ["session_id = ?", "store_id > ?"]
                query_params: list[Any] = [
                    *(guard_params * len(fields)),
                    session_id,
                    last_store_id,
                ]
                if conversation_id:
                    where.append("conversation_id = ?")
                    query_params.append(str(conversation_id).strip())
                query_params.append(page_limit)
                query_conn = conn if conn is not None else self._store._conn
                rows = query_conn.execute(
                    f"""SELECT store_id, {', '.join(bounded)}, {', '.join(lengths)}
                        FROM messages WHERE {' AND '.join(where)}
                        ORDER BY store_id LIMIT ?""",
                    query_params,
                ).fetchall()
                if time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("rollover prefix deadline exceeded")
                if not rows:
                    return compared
                for row in rows:
                    if time.monotonic() >= float(read_budget["deadline_at"]):
                        raise RuntimeError("rollover prefix deadline exceeded")
                    if compared >= len(messages):
                        return compared if allow_durable_interleaving else None
                    row_lengths = [int(value or 0) for value in row[6:11]]
                    row_bytes = sum(row_lengths)
                    if row_bytes > row_payload_cap or any(
                        length > field_limit for length in row_lengths
                    ):
                        raise RuntimeError("rollover prefix serialized byte bound exceeded")
                    self._charge_locked_publication_read(
                        read_budget,
                        rows=1,
                        serialized_bytes=row_bytes + 64,
                        label="rollover prefix",
                    )
                    stored_msg = dict(zip(fields, row[1:6]))
                    msg = messages[compared]
                    # Charge every active field in its original encoded form
                    # before normalization, redaction, or externalized
                    # placeholder expansion can shrink or replace it.
                    active_bytes = 0
                    for field in fields:
                        active_bytes += self._bounded_rollover_active_encoded_bytes(
                            msg.get(field),
                            read_budget=read_budget,
                            remaining_bytes=(
                                int(read_budget["max_bytes"])
                                - int(read_budget["bytes"])
                                - 32
                                - active_bytes
                            ),
                        )
                    self._charge_locked_publication_read(
                        read_budget,
                        rows=0,
                        serialized_bytes=active_bytes + 32,
                        label="rollover prefix",
                    )
                    for candidate in (
                        stored_msg.get("content"), stored_msg.get("tool_calls")
                    ):
                        self._guard_rollover_nested_representation(
                            candidate,
                            read_budget=read_budget,
                        )
                    message_identity = self._session_end_prefix_compare_identity(
                        msg,
                        session_id=session_id,
                        read_budget=read_budget,
                    )
                    stored_identity = self._session_end_prefix_compare_identity(
                        stored_msg,
                        session_id=session_id,
                        read_budget=read_budget,
                    )
                    if time.monotonic() >= float(read_budget["deadline_at"]):
                        raise RuntimeError("rollover prefix deadline exceeded")
                    message_encoded = json.dumps(
                        message_identity,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    stored_encoded = json.dumps(
                        stored_identity,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if max(len(message_encoded), len(stored_encoded)) > row_payload_cap:
                        raise RuntimeError("rollover prefix serialized byte bound exceeded")
                    last_store_id = int(row[0])
                    if hashlib.sha256(message_encoded).digest() != hashlib.sha256(stored_encoded).digest():
                        if allow_durable_interleaving:
                            continue
                        return None
                    compared += 1
                if len(rows) < page_limit:
                    return compared
        except Exception:
            logger.debug("LCM session-end prefix check failed", exc_info=True)
            if raise_on_read_error:
                raise
            return None

    @staticmethod
    def _ordered_unmatched_retained_indices(
        host_identities: Sequence[Any],
        durable_identities: Sequence[Any],
    ) -> list[int]:
        """Return host occurrences not represented by one ordered durable match.

        Durable occurrence positions are queued once, then consumed while the
        host sequence is scanned from left to right.  This recognizes shared
        suffixes after divergent branches, preserves repeated-value
        multiplicity, and stays O(host + durable) in time and space.
        """
        positions: dict[Any, deque[int]] = defaultdict(deque)
        for durable_index, identity in enumerate(durable_identities):
            positions[identity].append(durable_index)

        unmatched: list[int] = []
        last_position = -1
        for host_index, identity in enumerate(host_identities):
            candidates = positions.get(identity)
            if candidates is not None:
                while candidates and candidates[0] <= last_position:
                    candidates.popleft()
            if candidates:
                last_position = candidates.popleft()
            else:
                unmatched.append(host_index)
        return unmatched

    def _session_end_store_retained_identities(
        self,
        session_id: str,
        *,
        conversation_id: str,
        read_budget: dict[str, float | int],
        conn: sqlite3.Connection,
    ) -> list[tuple[str, str, str, str, str]]:
        """Read one bounded, ignore-normalized durable retained sequence."""
        identities: list[tuple[str, str, str, str, str]] = []
        last_store_id = 0
        fields = ("role", "content", "tool_call_id", "tool_name", "tool_calls")
        field_limit = min(
            _PUBLICATION_LOCKED_MAX_FIELD_BYTES,
            int(read_budget["max_bytes"]),
        )
        while True:
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("rollover sequence deadline exceeded")
            remaining_rows = int(read_budget["max_rows"]) - int(read_budget["rows"])
            remaining_bytes = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
            if remaining_rows <= 0:
                raise RuntimeError("rollover sequence row bound exceeded")
            if remaining_bytes <= 64:
                raise RuntimeError("rollover sequence serialized byte bound exceeded")
            max_row_bytes = min(
                int(read_budget["max_bytes"]),
                (len(fields) * field_limit) + 64,
            )
            byte_sized_rows = max(1, remaining_bytes // max(1, max_row_bytes))
            page_limit = min(
                _PUBLICATION_LOCKED_QUERY_BATCH,
                remaining_rows + 1,
                byte_sized_rows,
            )
            row_payload_cap = max(0, remaining_bytes // max(1, page_limit) - 64)
            lengths = [f"COALESCE(length(CAST({field} AS BLOB)), 0)" for field in fields]
            total_length = " + ".join(lengths)
            guard = " AND ".join(
                [f"({total_length}) <= ?"]
                + [f"{length} <= ?" for length in lengths]
            )
            bounded = [
                f"CASE WHEN {guard} THEN "
                f"CASE WHEN {field} IS NULL THEN NULL ELSE "
                f"substr(CAST({field} AS TEXT), 1, {field_limit + 1}) END "
                "ELSE NULL END"
                for field in fields
            ]
            guard_params = (row_payload_cap, *(field_limit for _ in fields))
            rows = conn.execute(
                f"""SELECT store_id, {', '.join(bounded)}, {', '.join(lengths)}
                    FROM messages
                    WHERE session_id = ? AND conversation_id = ? AND store_id > ?
                    ORDER BY store_id LIMIT ?""",
                [
                    *(guard_params * len(fields)),
                    session_id,
                    conversation_id,
                    last_store_id,
                    page_limit,
                ],
            ).fetchall()
            if not rows:
                return identities
            for row in rows:
                row_lengths = [int(value or 0) for value in row[6:11]]
                row_bytes = sum(row_lengths)
                if row_bytes > row_payload_cap or any(
                    length > field_limit for length in row_lengths
                ):
                    raise RuntimeError("rollover sequence serialized byte bound exceeded")
                self._charge_locked_publication_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=row_bytes + 64,
                    label="rollover sequence",
                )
                last_store_id = int(row[0])
                stored_message = dict(zip(fields, row[1:6]))
                if self._matches_ignore_message_patterns(
                    stored_message,
                    stored_row=True,
                    read_budget=read_budget,
                ):
                    continue
                identities.append(
                    self._session_end_prefix_compare_identity(
                        stored_message,
                        session_id=session_id,
                        read_budget=read_budget,
                    )
                )
            if len(rows) < page_limit:
                return identities

    @staticmethod
    def _lcm_bypass_message_fingerprint(message: Dict[str, Any]) -> str:
        tool_calls = message.get("tool_calls")
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        payload = {
            "role": message.get("role"),
            "content": normalize_content_value(message.get("content")),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": tool_calls,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()

    def _remember_lcm_bypass_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            remembered = self._lcm_bypass_message_prefix_fingerprints.setdefault(session_id, [])
            truncated = len(messages) > _LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT
            retained: list[tuple[list[str], bool]] = []
            for existing_fingerprints, existing_truncated in remembered:
                if existing_fingerprints == fingerprints:
                    truncated = truncated or bool(existing_truncated)
                    continue
                retained.append((existing_fingerprints, existing_truncated))
            retained.append((fingerprints, truncated))
            remembered[:] = retained

    def _remember_lcm_normal_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            self._lcm_normal_message_prefix_fingerprints[
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ] = fingerprints

    def _lcm_normal_prefix_key(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        return (
            session_id,
            str(
                conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or ""
            ),
        )

    def _messages_match_fingerprint_prefix(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_fingerprint_prefix_count(fingerprints, messages) > 0

    def _matching_fingerprint_prefix_count(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> int:
        if not fingerprints or not messages:
            return 0
        compare_count = min(len(fingerprints), len(messages))
        if compare_count <= 0:
            return 0
        candidate = [self._lcm_bypass_message_fingerprint(msg) for msg in messages[:compare_count]]
        if candidate == fingerprints[:compare_count]:
            return compare_count
        return 0

    def _messages_match_lcm_bypass_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_lcm_bypass_prefix_count(session_id, messages) > 0

    def _matching_lcm_bypass_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        count, _truncated = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        return count

    def _matching_lcm_bypass_prefix_evidence(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, bool]:
        best_count = 0
        best_truncated = False
        for fingerprints, truncated in self._lcm_bypass_message_prefix_fingerprints.get(session_id, []):
            count = self._matching_fingerprint_prefix_count(fingerprints, messages)
            count_truncated = bool(truncated and count > 0 and count == len(fingerprints))
            if count > best_count:
                best_count = count
                best_truncated = count_truncated
            elif count == best_count:
                best_truncated = best_truncated or count_truncated
        return best_count, best_truncated

    def _messages_match_lcm_normal_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> bool:
        return self._matching_lcm_normal_prefix_count(
            session_id,
            messages,
            conversation_id=conversation_id,
        ) > 0

    def _matching_lcm_normal_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> int:
        return self._matching_fingerprint_prefix_count(
            self._lcm_normal_message_prefix_fingerprints.get(
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ) or [],
            messages,
        )

    def _append_off_current_session_end_suffix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        prefix_count: int,
        source: str,
        conversation_id: str,
    ) -> list[int]:
        if not session_id or not messages:
            return []
        kept: list[tuple[int, Dict[str, Any]]] = []
        # Reconcile the complete retained sequence below.  A host index is not
        # durable evidence because ignored rows legitimately leave gaps.
        for host_index, msg in enumerate(messages):
            if self._matches_ignore_message_patterns(msg):
                self._ignored_message_count += 1
                excerpt = (text_content_for_pattern_matching(msg.get("content")) or "")[:80].replace("\n", " ")
                logger.debug(
                    "LCM ignore_message_patterns dropped late session-end %s message: %r",
                    msg.get("role", "unknown"),
                    excerpt,
                )
                continue
            kept.append((host_index, msg))
        protected_messages = (
            protect_messages_for_ingest(
                [message for _host_index, message in kept],
                session_id=session_id,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if kept
            else []
        )
        prepared_tail = [
            (host_index, protected, count_message_tokens(protected))
            for (host_index, _message), protected in zip(kept, protected_messages)
        ]
        appended_ids: list[int] = []
        with _temporary_sqlite_busy_timeout(
            [getattr(self._frontier, "_conn", None)],
            _SESSION_END_BUSY_TIMEOUT_MS,
        ):
            with self._frontier.publication_transaction() as conn:
                self._rollover_publication_boundary("after_begin")
                read_budget = self._new_locked_publication_read_budget()
                lifecycle = conn.execute(
                    """SELECT current_session_id, last_finalized_session_id,
                              current_frontier_store_id, last_finalized_frontier_store_id,
                              rollover_carry_over_context
                       FROM lcm_lifecycle_state WHERE conversation_id = ?""",
                    (conversation_id,),
                ).fetchone()
                active = conn.execute(
                    """SELECT generation, session_id, source_end_store_id,
                              policy_fingerprint, route_fingerprint
                       FROM lcm_active_frontiers WHERE conversation_id = ?
                       ORDER BY generation DESC LIMIT 1""",
                    (conversation_id,),
                ).fetchone()
                current_session_id = str(lifecycle[0] or "") if lifecycle else ""
                finalized_session_id = str(lifecycle[1] or "") if lifecycle else ""
                active_session_id = str(active[1] or "") if active else ""
                extends_rollover = bool(
                    lifecycle is not None
                    and active is not None
                    and current_session_id
                    and current_session_id != session_id
                    and finalized_session_id == session_id
                    and active_session_id == current_session_id
                )
                finalizes_current = bool(
                    lifecycle is not None and current_session_id == session_id
                )
                extends_finalized = bool(
                    lifecycle is not None
                    and not current_session_id
                    and finalized_session_id == session_id
                )
                if not (extends_rollover or finalizes_current or extends_finalized):
                    raise RuntimeError(
                        "late session-end rejected after durable lifecycle ownership changed"
                    )
                if (
                    extends_rollover
                    and lifecycle[4] is not None
                    and not bool(lifecycle[4])
                ):
                    raise RuntimeError(
                        "late session-end rejected by rollover carry-over policy"
                    )

                appended_ids = self._revalidate_and_append_old_session_tail_no_commit(
                    conn,
                    conversation_id=conversation_id,
                    old_session_id=session_id,
                    previous_messages=messages,
                    prepared_tail=prepared_tail,
                    source=source,
                    read_budget=read_budget,
                )
                self._rollover_publication_boundary("after_revalidation")
                self._rollover_publication_boundary("after_tail_ingest")

                if extends_rollover:
                    self._extend_completed_rollover_no_commit(
                        conn,
                        conversation_id=conversation_id,
                        old_session_id=session_id,
                        winner_session_id=current_session_id,
                        active=active,
                        read_budget=read_budget,
                        batch_reason="late_finalized_session_suffix",
                    )
                elif finalizes_current:
                    frontier_store_id = int(lifecycle[2] or 0)
                    if active is not None and active_session_id == session_id:
                        frontier_store_id = max(frontier_store_id, int(active[2] or 0))
                    self._lifecycle.finalize_session_no_commit(
                        conn,
                        conversation_id,
                        session_id=session_id,
                        frontier_store_id=frontier_store_id,
                    )
                    self._rollover_publication_boundary("after_lifecycle")

        self._rollover_publication_boundary("after_commit")
        return appended_ids

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Entire storage-consuming end path holds the lifetime lock so final
        # ingest/finalize cannot race shutdown/rebind close. Stopping the
        # worker alone does not close storage; full stop→close ownership
        # remains with shutdown / profile rebind under the same lock.
        with self._storage_lifetime_lock:
            self._on_session_end_locked(session_id, messages)

    def _on_session_end_locked(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._storage_lifetime_state != "bound":
            logger.warning(
                "LCM refusing on_session_end storage work while storage is %s",
                self._storage_lifetime_state,
            )
            return
        if session_id == self._session_id:
            self._stop_async_worker()
        ended_generation = self._in_process_auxiliary_caller_generation(session_id)
        active_auxiliary_end = session_id in self._active_auxiliary_session_ids()
        if (
            self._has_auxiliary_lineage_session(session_id)
            and session_id != self._session_id
            and (
                active_auxiliary_end
                or not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                or ended_generation
                or (
                    session_id in self._auxiliary_last_prompt_tokens
                    and not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                )
            )
        ):
            current_thread_session_id = self._thread_context_session_id()
            deactivated = self._deactivate_auxiliary_session(
                session_id,
                generation=ended_generation,
            )
            if deactivated:
                if current_thread_session_id == session_id or active_auxiliary_end:
                    self._remember_lcm_bypass_message_prefix(session_id, messages)
                self._end_host_fallback_compressor_for_session(
                    session_id,
                    messages,
                    current_session_bypasses=False,
                )
                if current_thread_session_id == session_id:
                    self._clear_thread_context_stateless(session_id)
            return
        current_session_bypasses = session_id == self._session_id and self._bypasses_lcm_context_management()
        ended_session_directly_bypasses = self._ended_session_directly_bypasses_lcm(session_id)
        direct_bypass_normal_conversation_id = self._lcm_session_last_normal_conversation_id.get(session_id)
        direct_bypass_normal_prefix_count = None
        if (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and direct_bypass_normal_conversation_id
            and not ended_generation
        ):
            direct_bypass_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=direct_bypass_normal_conversation_id,
            )
        direct_bypass_is_suppressed_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
            and session_id != self._thread_context_session_id()
            and direct_bypass_normal_prefix_count is not None
            and direct_bypass_normal_prefix_count > 0
        )
        if ended_session_directly_bypasses and not direct_bypass_is_suppressed_reused_normal:
            self._remember_lcm_bypass_message_prefix(session_id, messages)
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            if session_id == self._thread_context_session_id():
                self._deactivate_auxiliary_session(session_id, generation=ended_generation)
                self._clear_thread_context_stateless(session_id)
            return
        durable_end_state = self._lifecycle.get_by_session(session_id)
        if (
            durable_end_state is not None
            and durable_end_state.current_session_id
            and durable_end_state.current_session_id != session_id
            and durable_end_state.last_finalized_session_id == session_id
        ):
            durable_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=durable_end_state.conversation_id,
            )
            self._append_off_current_session_end_suffix(
                session_id,
                messages,
                prefix_count=max(0, int(durable_prefix_count or 0)),
                source=(
                    self._lcm_session_last_normal_platform.get(session_id)
                    or self._lcm_session_last_platform.get(session_id, self._session_platform)
                ),
                conversation_id=durable_end_state.conversation_id,
            )
            return
        known_normal_conversation_id = self._lcm_session_last_normal_conversation_id.get(
            session_id
        )
        known_normal_state = self._lifecycle.get_by_conversation(
            known_normal_conversation_id
        ) if known_normal_conversation_id else None
        if (
            known_normal_state is not None
            and known_normal_state.current_session_id != session_id
            and known_normal_state.last_finalized_session_id != session_id
        ):
            raise RuntimeError(
                "late session-end rejected after durable lifecycle ownership changed"
            )
        same_id_has_bypass_lineage = (
            session_id == self._session_id
            and not current_session_bypasses
            and self._has_lcm_bypass_lineage_session(session_id)
        )
        same_id_normal_prefix_count = None
        same_id_recorded_normal_prefix_count = 0
        same_id_bypass_prefix_count = 0
        same_id_bypass_prefix_truncated = False
        if same_id_has_bypass_lineage:
            same_id_conversation_id = (
                self._conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or None
            )
            (
                same_id_bypass_prefix_count,
                same_id_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
            same_id_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
            same_id_recorded_normal_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
        same_id_store_prefix_positive = (
            same_id_normal_prefix_count is not None
            and same_id_normal_prefix_count > 0
        )
        same_id_strongest_normal_prefix_count = max(
            same_id_recorded_normal_prefix_count,
            same_id_normal_prefix_count if same_id_store_prefix_positive else 0,
        )
        same_id_truncated_bypass_prefix_ambiguous = (
            same_id_bypass_prefix_truncated
            and same_id_bypass_prefix_count > 0
            and same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            and len(messages) > same_id_bypass_prefix_count
        )
        same_id_matches_stronger_normal_prefix = (
            same_id_strongest_normal_prefix_count > 0
            and not same_id_truncated_bypass_prefix_ambiguous
            and (
                same_id_bypass_prefix_count <= 0
                or same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            )
        )
        off_current_auxiliary_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
        )
        off_current_lineage = (
            session_id != self._session_id
            and (
                self._has_lcm_bypass_lineage_session(session_id)
                or off_current_auxiliary_reused_normal
            )
        )
        off_current_normal_conversation_id = (
            self._lcm_session_last_normal_conversation_id.get(session_id)
            if off_current_lineage
            else ""
        )
        off_current_store_prefix_count = None
        off_current_recorded_prefix_count = 0
        off_current_bypass_prefix_count = 0
        off_current_bypass_prefix_truncated = False
        if off_current_lineage:
            (
                off_current_bypass_prefix_count,
                off_current_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        if off_current_lineage and off_current_normal_conversation_id:
            off_current_store_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
            off_current_recorded_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
        off_current_prefix_count = None
        off_current_store_prefix_positive = (
            off_current_store_prefix_count is not None
            and off_current_store_prefix_count > 0
        )
        off_current_store_prefix_for_append = int(off_current_store_prefix_count or 0)
        off_current_recorded_prefix_for_append = 0
        if off_current_recorded_prefix_count > 0 and off_current_normal_conversation_id:
            try:
                stored_normal_rows = self._store.get_range(
                    session_id,
                    limit=off_current_recorded_prefix_count + 1,
                    conversation_id=off_current_normal_conversation_id,
                )
            except Exception:
                logger.debug("LCM off-current recorded-prefix row-count probe failed", exc_info=True)
                stored_normal_rows = []
            if len(stored_normal_rows) == off_current_recorded_prefix_count:
                off_current_recorded_prefix_for_append = off_current_recorded_prefix_count
        off_current_strongest_normal_prefix_count = max(
            off_current_store_prefix_for_append if off_current_store_prefix_positive else 0,
            off_current_recorded_prefix_for_append,
        )
        off_current_truncated_bypass_prefix_ambiguous = (
            off_current_bypass_prefix_truncated
            and off_current_bypass_prefix_count > 0
            and off_current_strongest_normal_prefix_count >= off_current_bypass_prefix_count
            and len(messages) > off_current_bypass_prefix_count
        )
        if (
            off_current_store_prefix_positive
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_store_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_store_prefix_for_append
        elif (
            off_current_recorded_prefix_for_append > 0
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_recorded_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_recorded_prefix_for_append
        if (
            off_current_lineage
            and not off_current_auxiliary_reused_normal
            and off_current_normal_conversation_id
            and off_current_store_prefix_count == 0
            and off_current_bypass_prefix_count <= 0
            and self._lcm_session_last_bypassed.get(session_id) is False
        ):
            off_current_prefix_count = 0
        same_id_should_bypass = (
            same_id_has_bypass_lineage
            and same_id_bypass_prefix_count > 0
            and not same_id_matches_stronger_normal_prefix
        )
        off_current_matches_bypass_prefix = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and off_current_bypass_prefix_count > 0
            and off_current_prefix_count is None
        )
        ended_lineage_bypasses = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and bool(self._lcm_session_last_bypassed.get(session_id))
            and not self._session_end_matches_current_store_prefix(session_id, messages)
        )
        off_current_should_bypass = off_current_lineage and off_current_prefix_count is None
        if off_current_prefix_count is not None:
            prefix_count = off_current_prefix_count
            self._append_off_current_session_end_suffix(
                session_id,
                messages,
                prefix_count=prefix_count,
                source=(
                    self._lcm_session_last_normal_platform.get(session_id)
                    or self._lcm_session_last_platform.get(session_id, self._session_platform)
                ),
                conversation_id=off_current_normal_conversation_id,
            )
            return
        if (
            current_session_bypasses
            or same_id_should_bypass
            or off_current_should_bypass
            or off_current_matches_bypass_prefix
            or ended_lineage_bypasses
        ):
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            return
        try:
            with _temporary_sqlite_busy_timeout(
                [
                    getattr(self._store, "_conn", None),
                    getattr(self._lifecycle, "_conn", None),
                    getattr(self._frontier, "_conn", None),
                ],
                _SESSION_END_BUSY_TIMEOUT_MS,
            ):
                try:
                    # Best-effort final flush. Keep this path bounded because
                    # host gateways call session-end hooks from lifecycle paths
                    # that must not wait through SQLite's normal busy timeout.
                    self._ingest_messages(messages)
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end raw-message ingest interrupted; "
                        "final messages may be absent from the plugin-local store"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end raw-message ingest skipped due to SQLite lock after short wait; "
                            "final messages may be absent from the plugin-local store: %s",
                            exc,
                        )
                        return
                    raise

                try:
                    self._lifecycle.finalize_session(
                        self._conversation_id,
                        session_id,
                        frontier_store_id=self._last_compacted_store_id,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end lifecycle finalization interrupted; "
                        "raw messages may be ingested but lifecycle state may be finalized later"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end lifecycle finalization skipped due to SQLite lock after short wait; "
                            "raw messages were ingested but lifecycle state may be finalized later: %s",
                            exc,
                        )
                        return
                    raise
        except KeyboardInterrupt:
            logger.warning("LCM session-end ingest/finalize interrupted before bounded flush completed")
            return
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "LCM session-end ingest/finalize skipped due to SQLite lock before bounded flush: %s",
                    exc,
                )
                return
            raise

    def on_session_reset(self) -> None:
        # Storage-consuming reset (lifecycle record + DAG prune) holds the
        # lifetime lock so concurrent shutdown/rebind cannot close helpers mid-write.
        with self._storage_lifetime_lock:
            self._on_session_reset_locked()

    def _on_session_reset_locked(self, *, persist_storage: bool = True) -> None:
        if self._host_fallback_compressor is not None:
            compressor = self._host_fallback_compressor
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        self._pending_reset_session_id = self._session_id
        self._pending_reset_conversation_id = self._conversation_id
        self._pending_reset_frontier_store_id = self._last_compacted_store_id
        super().on_session_reset()
        # Process-local runtime always clears; storage writes require bound helpers.
        if self._storage_lifetime_state != "bound":
            logger.warning(
                "LCM refusing on_session_reset storage work while storage is %s",
                self._storage_lifetime_state,
            )
            self._reset_session_scoped_runtime_state()
            return
        if not persist_storage:
            self._reset_session_scoped_runtime_state()
            return
        self._lifecycle.record_reset(self._conversation_id)
        self._reset_session_scoped_runtime_state()

        # Retain DAG nodes across sessions based on config.
        #   -1  → keep all nodes
        #    0  → delete everything
        #    N  → keep nodes at depth >= N (e.g. 2 keeps d2+)
        retain = self._config.new_session_retain_depth
        if self._session_id and retain != -1:
            if retain == 0:
                self._dag.delete_session_nodes(self._session_id)
            else:
                self._dag.delete_below_depth(self._session_id, retain)

    def carry_over_new_session_context(self, old_session_id: str, new_session_id: str) -> int:
        """Move retained summaries from the old session into the new one.

        This reassigns session ownership for retained summary nodes, but it does
        not rewrite the nodes' descendant raw-message lineage. Retrieval under
        ``session_scope='current'`` may therefore include a carried-over node in
        the new session, while ``source`` filtering still evaluates against the
        node's original descendant message sources.
        """
        with self._storage_lifetime_lock:
            return self._carry_over_new_session_context_locked(
                old_session_id, new_session_id
            )

    def _reassign_canonical_session_state(
        self,
        conversation_id: str,
        old_session_id: str,
        new_session_id: str,
        *,
        reset_if_no_items: bool = False,
    ) -> int:
        """Carry DAG ownership and its active frontier in one writer transaction."""
        if not conversation_id:
            return self._dag.reassign_session_nodes(old_session_id, new_session_id)

        with self._frontier.publication_transaction() as conn:
            active = conn.execute(
                """
                SELECT generation, session_id, source_end_store_id,
                       policy_fingerprint, route_fingerprint
                FROM lcm_active_frontiers
                WHERE conversation_id = ?
                ORDER BY generation DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            cur = conn.execute(
                "UPDATE summary_nodes SET session_id = ? WHERE session_id = ?",
                (new_session_id, old_session_id),
            )
            moved = int(cur.rowcount or 0)

            if (
                active is None
                or str(active[1] or "") != old_session_id
                or int(active[2] or 0) <= 0
            ):
                return moved

            rows = conn.execute(
                """
                SELECT i.ref_id, i.source_start, i.source_end
                FROM lcm_frontier_items AS i
                JOIN summary_nodes AS n ON n.node_id = i.ref_id
                WHERE i.conversation_id = ? AND i.generation = ?
                  AND i.kind = 'node' AND n.session_id = ?
                ORDER BY i.ordinal
                """,
                (conversation_id, int(active[0]), new_session_id),
            ).fetchall()
            items = [
                {
                    "kind": "node",
                    "ref_id": int(row[0]),
                    "source_start": int(row[1]),
                    "source_end": int(row[2]),
                }
                for row in rows
                if int(row[1] or 0) > 0 and int(row[2] or 0) >= int(row[1] or 0)
            ]
            if not items and not reset_if_no_items:
                raise RuntimeError(
                    "positive canonical frontier has no carried node items"
                )
            next_source_end = int(active[2]) if items else 0
            new_generation = self._frontier.advance_frontier_generation_with_items_no_commit(
                conn,
                conversation_id,
                new_session_id,
                next_source_end,
                str(active[3] or ""),
                str(active[4] or ""),
                int(active[0]),
                items,
            )
            if not new_generation:
                raise RuntimeError("canonical frontier changed during session carry-over")
            return moved

    def _carry_over_new_session_context_locked(
        self, old_session_id: str, new_session_id: str
    ) -> int:
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        if self._session_ignored and new_session_id == self._session_id:
            logger.debug(
                "LCM carry-over skipped for ignored session %s",
                new_session_id,
            )
            return 0
        if self._storage_lifetime_state != "bound":
            logger.warning(
                "LCM refusing carry_over_new_session_context while storage is %s",
                self._storage_lifetime_state,
            )
            return 0
        return self._reassign_canonical_session_state(
            self._conversation_id,
            old_session_id,
            new_session_id,
            reset_if_no_items=True,
        )

    def _rollover_frontier_with_old_raw_tail_no_commit(
        self,
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        generation: int,
        canonical_session_id: str,
        old_session_id: str,
        source_end_store_id: int,
        include_existing: bool,
        read_budget: dict[str, float | int],
    ) -> tuple[list[dict[str, Any]], int, list[int]]:
        """Clone one immutable frontier and close it over durable old raw tail."""
        items: list[dict[str, Any]] = []
        if include_existing and generation > 0:
            rows = self._bounded_frontier_rows_no_commit(
                conn,
                conversation_id,
                generation,
                read_budget=read_budget,
            )
            for _ordinal, kind, ref_id, source_start, source_end, node_session in rows:
                if kind not in {"node", "message"}:
                    raise RuntimeError("rollover frontier contains invalid item kind")
                if source_start <= 0 or source_end < source_start:
                    raise RuntimeError("rollover frontier contains invalid source range")
                if kind == "node" and node_session is None:
                    # Retention pruning removed this canonical summary inside
                    # the same transaction. Its raw descendants are restored
                    # below instead of leaving a source-coverage hole.
                    continue
                if kind == "node" and node_session != canonical_session_id:
                    raise RuntimeError("rollover frontier node was not reassigned to winner")
                if kind == "message" and not (
                    ref_id == source_start == source_end
                ):
                    raise RuntimeError("rollover raw frontier range is invalid")
                items.append(
                    {
                        "kind": kind,
                        "ref_id": ref_id,
                        "source_start": source_start,
                        "source_end": source_end,
                    }
                )

        items.sort(
            key=lambda item: (
                int(item["source_start"]),
                0 if item["kind"] == "node" else 1,
                int(item["ref_id"]),
            )
        )
        previous_end = 0
        for item in items:
            start = int(item["source_start"])
            end = int(item["source_end"])
            if start <= previous_end or end < start:
                raise RuntimeError("rollover frontier source closure overlaps")
            previous_end = end

        # Scan the full bounded old-session sequence, not merely rows after the
        # old source tip.  That preserves existing raw items and restores raw
        # descendants of any summary pruned in this same transaction, so the
        # next source boundary can never jump across an uncovered durable row.
        durable_raw_ids = self._bounded_message_tail_ids_no_commit(
            conn,
            old_session_id,
            0,
            read_budget=read_budget,
            conversation_id=conversation_id,
        )
        added_raw_ids: list[int] = []
        coverage_index = 0
        for store_id in durable_raw_ids:
            while (
                coverage_index < len(items)
                and int(items[coverage_index]["source_end"]) < store_id
            ):
                coverage_index += 1
            if (
                coverage_index < len(items)
                and int(items[coverage_index]["source_start"]) <= store_id
                <= int(items[coverage_index]["source_end"])
            ):
                continue
            items.append(
                {
                    "kind": "message",
                    "ref_id": store_id,
                    "source_start": store_id,
                    "source_end": store_id,
                }
            )
            added_raw_ids.append(store_id)

        ordered = sorted(
            items,
            key=lambda item: (
                int(item["source_start"]),
                0 if item["kind"] == "node" else 1,
                int(item["ref_id"]),
            ),
        )
        previous_end = 0
        for item in ordered:
            start = int(item["source_start"])
            end = int(item["source_end"])
            if start <= previous_end or end < start:
                raise RuntimeError("rollover frontier source closure overlaps")
            previous_end = end
        new_source_end = max(
            (int(item["source_end"]) for item in ordered),
            default=0,
        )
        return ordered, new_source_end, added_raw_ids

    def _revalidate_and_append_old_session_tail_no_commit(
        self,
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        old_session_id: str,
        previous_messages: Sequence[Dict[str, Any]],
        prepared_tail: Sequence[tuple[Any, ...]],
        source: str,
        read_budget: dict[str, float | int],
    ) -> list[int]:
        """Append only retained host occurrences not represented durably."""
        tail_entries: list[tuple[int, Dict[str, Any], int]] = []
        legacy_prefix = len(previous_messages) - len(prepared_tail)
        for offset, entry in enumerate(prepared_tail):
            if len(entry) == 3:
                host_index, message, token_estimate = entry
            elif len(entry) == 2:  # compatibility for direct private callers
                message, token_estimate = entry
                host_index = legacy_prefix + offset
            else:
                raise RuntimeError("invalid prepared old-session tail entry")
            tail_entries.append((int(host_index), message, int(token_estimate)))

        retained_entries = [
            entry
            for entry in tail_entries
            if not self._matches_ignore_message_patterns(entry[1])
        ]
        durable_identities = self._session_end_store_retained_identities(
            old_session_id,
            conversation_id=conversation_id,
            read_budget=read_budget,
            conn=conn,
        )
        host_identities: list[tuple[str, str, str, str, str]] = []
        for _host_index, message, _estimate in retained_entries:
            encoded_bytes = self._bounded_rollover_active_encoded_bytes(
                message,
                read_budget=read_budget,
                remaining_bytes=(
                    int(read_budget["max_bytes"])
                    - int(read_budget["bytes"])
                    - 32
                ),
            )
            self._charge_locked_publication_read(
                read_budget,
                rows=1,
                serialized_bytes=encoded_bytes + 32,
                label="rollover retained host sequence",
            )
            host_identities.append(
                self._session_end_prefix_compare_identity(
                    message,
                    session_id=old_session_id,
                    read_budget=read_budget,
                )
            )
        unmatched = self._ordered_unmatched_retained_indices(
            host_identities,
            durable_identities,
        )
        to_append = [
            (retained_entries[index][1], retained_entries[index][2])
            for index in unmatched
        ]
        if not to_append:
            return []
        return self._store.append_protected_batch_no_commit(
            conn,
            old_session_id,
            [message for message, _estimate in to_append],
            [estimate for _message, estimate in to_append],
            source=source,
            conversation_id=conversation_id,
        )

    def _extend_completed_rollover_no_commit(
        self,
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        old_session_id: str,
        winner_session_id: str,
        active: sqlite3.Row | tuple[Any, ...],
        read_budget: dict[str, float | int],
        batch_reason: str,
    ) -> bool:
        """Close a completed winner over newly durable finalized-session rows."""
        generation = int(active[0])
        items, new_frontier, added_raw_ids = (
            self._rollover_frontier_with_old_raw_tail_no_commit(
                conn,
                conversation_id=conversation_id,
                generation=generation,
                canonical_session_id=winner_session_id,
                old_session_id=old_session_id,
                source_end_store_id=int(active[2] or 0),
                include_existing=True,
                read_budget=read_budget,
            )
        )
        if not added_raw_ids:
            return False
        new_generation = self._frontier.advance_frontier_generation_with_items_no_commit(
            conn,
            conversation_id,
            winner_session_id,
            new_frontier,
            str(active[3] or ""),
            str(active[4] or ""),
            generation,
            items,
        )
        if not new_generation:
            raise RuntimeError("canonical winner changed during finalized suffix adoption")
        self._frontier.supersede_competing_batches_no_commit(
            conn,
            conversation_id,
            generation,
            reason=batch_reason,
        )
        self._rollover_publication_boundary("after_frontier")
        self._lifecycle.extend_finalized_rollover_no_commit(
            conn,
            conversation_id,
            old_session_id=old_session_id,
            current_session_id=winner_session_id,
            frontier_store_id=new_frontier,
        )
        self._rollover_publication_boundary("after_lifecycle")
        return True

    def _rollover_publication_boundary(self, phase: str) -> None:
        crash_hook = getattr(self, "_rollover_publish_crash_hook", None)
        if callable(crash_hook):
            crash_hook(phase)
        elif crash_hook == phase:
            os._exit(88)  # noqa: PLW1510 - deliberate subprocess crash injection
        failure_hook = getattr(self, "_rollover_publish_failure_hook", None)
        if callable(failure_hook):
            failure_hook(phase)
        elif failure_hook == phase:
            raise RuntimeError("injected rollover publication failure")

    def _publish_rollover_state(
        self,
        conversation_id: str,
        old_session_id: str,
        new_session_id: str,
        *,
        carry_over_context: bool,
        final_tail: Sequence[tuple[Any, ...]],
        previous_messages: Sequence[Dict[str, Any]] | None = None,
        expected_generation: int | None = None,
        return_outcome: bool = False,
    ) -> int | tuple[int, str, str]:
        """Publish final tail, DAG/frontier, batches, and lifecycle atomically."""
        moved = 0
        with self._frontier.publication_transaction() as conn:
            self._rollover_publication_boundary("after_begin")
            read_budget = self._new_locked_publication_read_budget()
            active = conn.execute(
                """
                SELECT generation, session_id, source_end_store_id,
                       policy_fingerprint, route_fingerprint
                FROM lcm_active_frontiers
                WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            lifecycle = conn.execute(
                """
                SELECT current_session_id, last_finalized_session_id,
                       rollover_carry_over_context
                FROM lcm_lifecycle_state WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            current_generation = int(active[0]) if active else 0
            active_session_id = str(active[1] or "") if active else ""
            lifecycle_session_id = str(lifecycle[0] or "") if lifecycle else ""
            finalized_session_id = str(lifecycle[1] or "") if lifecycle else ""
            winner_carry_over = (
                True
                if lifecycle is None or lifecycle[2] is None
                else bool(lifecycle[2])
            )
            expected = (
                current_generation
                if expected_generation is None
                else int(expected_generation)
            )

            completed_session_id = ""
            if (
                active is not None
                and active_session_id != old_session_id
                and lifecycle_session_id == active_session_id
                and finalized_session_id == old_session_id
            ):
                completed_session_id = active_session_id
            if completed_session_id:
                if not winner_carry_over:
                    outcome = (
                        "idempotent" if completed_session_id == new_session_id else "competing"
                    )
                    result = (0, outcome, completed_session_id)
                    return result if return_outcome else 0
                self._revalidate_and_append_old_session_tail_no_commit(
                    conn,
                    conversation_id=conversation_id,
                    old_session_id=old_session_id,
                    previous_messages=list(previous_messages or []),
                    prepared_tail=final_tail,
                    source=self._session_platform or "unknown",
                    read_budget=read_budget,
                )
                self._rollover_publication_boundary("after_revalidation")
                self._rollover_publication_boundary("after_tail_ingest")
                self._extend_completed_rollover_no_commit(
                    conn,
                    conversation_id=conversation_id,
                    old_session_id=old_session_id,
                    winner_session_id=completed_session_id,
                    active=active,
                    read_budget=read_budget,
                    batch_reason="rollover_suffix_adopted",
                )
                outcome = (
                    "idempotent" if completed_session_id == new_session_id else "competing"
                )
                result = (0, outcome, completed_session_id)
                return result if return_outcome else 0

            if current_generation != expected:
                raise RuntimeError("canonical frontier generation changed during rollover")
            if active is not None and active_session_id != old_session_id:
                raise RuntimeError("canonical frontier is not owned by rollover old session")
            if lifecycle is None or lifecycle_session_id != old_session_id:
                raise RuntimeError("lifecycle is not owned by rollover old session")

            self._revalidate_and_append_old_session_tail_no_commit(
                conn,
                conversation_id=conversation_id,
                old_session_id=old_session_id,
                previous_messages=list(previous_messages or []),
                prepared_tail=final_tail,
                source=self._session_platform or "unknown",
                read_budget=read_budget,
            )
            self._rollover_publication_boundary("after_revalidation")
            self._rollover_publication_boundary("after_tail_ingest")
            base_generation = int(active[0]) if active else 0
            old_frontier = int(active[2] or 0) if active else 0

            retain = int(self._config.new_session_retain_depth)
            if retain == 0:
                conn.execute(
                    "DELETE FROM summary_nodes WHERE session_id = ?",
                    (old_session_id,),
                )
            elif retain > 0:
                conn.execute(
                    "DELETE FROM summary_nodes WHERE session_id = ? AND depth < ?",
                    (old_session_id, retain),
                )
            self._rollover_publication_boundary("after_prune")

            if carry_over_context:
                cur = conn.execute(
                    "UPDATE summary_nodes SET session_id = ? WHERE session_id = ?",
                    (new_session_id, old_session_id),
                )
                moved = max(0, int(cur.rowcount or 0))
            self._rollover_publication_boundary("after_reassign")

            items: list[dict[str, Any]] = []
            new_frontier = 0
            if carry_over_context and active is not None and old_frontier > 0:
                items, new_frontier, _added_raw_ids = (
                    self._rollover_frontier_with_old_raw_tail_no_commit(
                        conn,
                        conversation_id=conversation_id,
                        generation=base_generation,
                        canonical_session_id=new_session_id,
                        old_session_id=old_session_id,
                        source_end_store_id=old_frontier,
                        include_existing=True,
                        read_budget=read_budget,
                    )
                )
            elif carry_over_context:
                items, new_frontier, _added_raw_ids = (
                    self._rollover_frontier_with_old_raw_tail_no_commit(
                        conn,
                        conversation_id=conversation_id,
                        generation=base_generation,
                        canonical_session_id=new_session_id,
                        old_session_id=old_session_id,
                        source_end_store_id=old_frontier,
                        include_existing=False,
                        read_budget=read_budget,
                    )
                )
            new_generation = self._frontier.advance_frontier_generation_with_items_no_commit(
                conn,
                conversation_id,
                new_session_id,
                new_frontier,
                str(active[3] or "") if active else "",
                str(active[4] or "") if active else "",
                base_generation,
                items,
            )
            if not new_generation:
                raise RuntimeError("canonical frontier changed during rollover")
            self._frontier.supersede_competing_batches_no_commit(
                conn,
                conversation_id,
                base_generation,
                reason="session_rollover_published",
            )
            conn.execute(
                """
                UPDATE lcm_prepared_batches
                SET state = 'superseded',
                    failure_reason = 'session_rollover_published',
                    updated_at = ?
                WHERE conversation_id = ? AND state IN ('preparing', 'ready')
                """,
                (time.time(), conversation_id),
            )
            self._rollover_publication_boundary("after_frontier")

            self._lifecycle.record_rollover_no_commit(
                conn,
                conversation_id,
                old_session_id=old_session_id,
                new_session_id=new_session_id,
                current_frontier_store_id=new_frontier,
                finalized_frontier_store_id=max(old_frontier, new_frontier),
                carry_over_context=carry_over_context,
            )
            self._rollover_publication_boundary("after_lifecycle")

        self._rollover_publication_boundary("after_commit")
        result = (moved, "published", new_session_id)
        return result if return_outcome else moved

    def _prepare_rollover_final_tail(
        self,
        old_session_id: str,
        previous_messages: Sequence[Dict[str, Any]],
        *,
        conversation_id: str,
    ) -> list[tuple[int, Dict[str, Any], int]]:
        """Prepare the bounded, ignore-normalized host retained sequence.

        This phase performs no database writes.  A mismatch is fail-closed: the
        caller keeps its untouched host list and rollover does not prune or bind
        the new session.
        """
        if not previous_messages:
            return []
        read_budget = self._new_locked_publication_read_budget()
        retained = [
            message
            for message in previous_messages
            if not self._matches_ignore_message_patterns(message)
        ]
        if not retained:
            return []
        # The durable-prefix comparison charges only rows it actually
        # compares.  Charge every unmatched host row against that same budget
        # before normalization, redaction, placeholder recovery, tokenization,
        # or the writer transaction can materialize it.
        resolved_suffix: list[Dict[str, Any]] = []
        resolver_cache: dict[tuple[str, str], str] = {}
        for message in retained:
            encoded_bytes = self._bounded_rollover_active_encoded_bytes(
                message,
                read_budget=read_budget,
                remaining_bytes=(
                    int(read_budget["max_bytes"])
                    - int(read_budget["bytes"])
                    - 32
                ),
            )
            self._charge_locked_publication_read(
                read_budget,
                rows=1,
                serialized_bytes=encoded_bytes + 32,
                label="rollover final tail",
            )
            resolved = dict(message)
            content = normalize_content_value(resolved.get("content")) or ""
            stripped = content.strip()
            ref = None
            ingest_refs = extract_ingest_externalized_refs(stripped)
            if (
                len(ingest_refs) == 1
                and stripped.startswith("[Externalized LCM ingest payload:")
                and stripped.endswith("]")
            ):
                ref = ingest_refs[0]
            elif is_externalized_placeholder(stripped):
                ref = extract_externalized_ref(stripped)
            if ref:
                cache_key = ("externalized", ref)
                resolved_content = resolver_cache.get(cache_key)
                if resolved_content is None:
                    payload = load_externalized_payload(
                        ref,
                        config=self._config,
                        hermes_home=self._hermes_home,
                        read_budget=read_budget,
                        budget_label="rollover final tail externalized payload",
                        max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                        max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
                    )
                    if payload is None:
                        raise RuntimeError(
                            "rollover final tail externalized payload could not be resolved"
                        )
                    payload_session_id = str(payload.get("session_id") or "")
                    if payload_session_id and payload_session_id != old_session_id:
                        raise RuntimeError(
                            "rollover final tail externalized payload crosses session boundary"
                        )
                    payload_content = payload.get("content")
                    if not isinstance(payload_content, str):
                        raise RuntimeError(
                            "rollover final tail externalized payload has invalid content"
                        )
                    resolved_content = payload_content
                    resolver_cache[cache_key] = resolved_content
                content = resolved_content
            if (
                str(resolved.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(content)
            ):
                cache_key = ("persisted-output", content)
                recovered = resolver_cache.get(cache_key)
                if recovered is None:
                    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(
                        content,
                        read_budget=read_budget,
                        budget_label="rollover final tail persisted-output file",
                        max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                        max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
                    )
                    if recovered_with_stat is None:
                        raise RuntimeError(
                            "rollover final tail persisted-output file could not be resolved"
                        )
                    recovered = recovered_with_stat[0]
                    resolver_cache[cache_key] = recovered
                content = recovered
            resolved["content"] = content
            resolved_suffix.append(resolved)
        protected = protect_messages_for_ingest(
            resolved_suffix,
            session_id=old_session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return [
            (offset, message, count_message_tokens(message))
            for offset, message in enumerate(protected)
        ]

    def rollover_session(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
        carry_over_context: bool = True,
        **kwargs,
    ) -> int:
        """Complete a Hermes-style `/new` rollover for this engine.

        This is a small helper for host/runtime integrations that need the
        correct lifecycle ordering in one call:
        1. flush old-session messages into the store
        2. prune/reset retained DAG state on the old session
        3. bind the engine to the new session
        4. optionally move retained summaries into the new session

        The full rollover (including nested session end/reset/start/carry-over)
        holds ``_storage_lifetime_lock`` so shutdown/rebind cannot interleave
        between steps or against direct DAG reads in this helper.
        """
        with self._storage_lifetime_lock:
            return self._rollover_session_locked(
                old_session_id,
                new_session_id,
                previous_messages=previous_messages,
                carry_over_context=carry_over_context,
                **kwargs,
            )

    def _rollover_session_locked(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
        carry_over_context: bool = True,
        **kwargs,
    ) -> int:
        previous_messages = previous_messages or []
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        conversation_id = self._conversation_id or old_session_id or new_session_id
        bound_session_id = self._session_id
        can_carry_over = bool(
            old_session_id and bound_session_id and old_session_id == bound_session_id
        )

        if carry_over_context and boundary_reason == "compression" and old_session_id and old_session_id != new_session_id:
            if self._storage_lifetime_state != "bound":
                logger.warning(
                    "LCM refusing compression rollover while storage is %s",
                    self._storage_lifetime_state,
                )
                return 0
            before_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            if can_carry_over:
                self.on_session_end(old_session_id, previous_messages)
            else:
                logger.warning(
                    "LCM compression rollover old_session_id=%s does not match bound session=%s; using boundary handler fallback",
                    old_session_id,
                    bound_session_id,
                )
            self.on_session_start(
                new_session_id,
                old_session_id=old_session_id,
                **kwargs,
            )
            if self._storage_lifetime_state != "bound":
                return 0
            after_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            return len(after_node_ids - before_node_ids)

        published_rollover = False
        moved = 0
        if old_session_id and can_carry_over:
            self._stop_async_worker()
            active = self._frontier.get_active_frontier(conversation_id)
            expected_generation = int(active["generation"]) if active else 0
            final_tail = self._prepare_rollover_final_tail(
                old_session_id,
                previous_messages,
                conversation_id=conversation_id,
            )
            moved, _, durable_session_id = self._publish_rollover_state(
                conversation_id,
                old_session_id,
                new_session_id,
                carry_over_context=carry_over_context,
                final_tail=final_tail,
                previous_messages=previous_messages,
                expected_generation=expected_generation,
                return_outcome=True,
            )
            published_rollover = True
            self._on_session_reset_locked(persist_storage=False)
            self._clear_pending_reset_boundary()
            new_session_id = durable_session_id
        elif old_session_id and not carry_over_context:
            logger.warning(
                "LCM rollover skipped old-session finalization: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )
        elif old_session_id and not can_carry_over:
            logger.warning(
                "LCM carry-over skipped: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )

        self.on_session_start(new_session_id, conversation_id=conversation_id, **kwargs)

        if published_rollover:
            return moved
        if not carry_over_context:
            return 0
        if old_session_id and not can_carry_over:
            return 0
        return self.carry_over_new_session_context(old_session_id, new_session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            LCM_GREP,
            LCM_LOAD_SESSION,
            LCM_DESCRIBE,
            LCM_EXPAND,
            LCM_EXPAND_QUERY,
            LCM_FOCUS,
            LCM_STATUS,
            LCM_INSPECT,
            LCM_DOCTOR,
        ]

    def issue_cross_session_capability(
        self,
        session_ids: Sequence[str],
    ) -> _CrossSessionCapability:
        """Mint a trusted, engine-bound capability outside model arguments.

        Hosts may call this only after their own authorization decision. The
        tool schema cannot construct or request this opaque object.
        """
        if isinstance(session_ids, (str, bytes)):
            raise ValueError("cross-session capability requires a session-id sequence")
        normalized = frozenset(str(value or "").strip() for value in session_ids)
        if not normalized or "" in normalized:
            raise ValueError("cross-session capability requires non-empty session ids")
        if any(len(value) > 256 for value in normalized):
            raise ValueError("cross-session capability session ids are limited to 256 characters")
        if len(normalized) > lcm_tools._CROSS_SESSION_MAX_SESSIONS:
            raise ValueError(
                f"cross-session capability allows at most {lcm_tools._CROSS_SESSION_MAX_SESSIONS} sessions"
            )
        return _CrossSessionCapability(self._cross_session_capability_issuer, normalized)

    def _authorized_cross_session_ids(self, capability: Any) -> frozenset[str] | None:
        if not isinstance(capability, _CrossSessionCapability):
            return None
        if capability._issuer is not self._cross_session_capability_issuer:
            return None
        return capability.session_ids

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")

        if name != "lcm_inspect" and messages and self._session_id:
            if self._maybe_reclassify_current_session_as_auxiliary_before_message_ingest():
                self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            elif not (
                self._session_ignored or self._session_stateless or self._thread_context_stateless()
            ):
                try:
                    self._ingest_messages(messages)
                    self._record_ingest_success()
                    self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
                except Exception as e:
                    self._record_ingest_failure("tool-call ingest", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_load_session": lcm_tools.lcm_load_session,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_focus": lcm_tools.lcm_focus,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_inspect": lcm_tools.lcm_inspect,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            handler_kwargs = {"engine": self}
            if name in {"lcm_grep", "lcm_load_session", "lcm_expand", "lcm_expand_query"} and kwargs.get("cross_session_capability") is not None:
                handler_kwargs["cross_session_capability"] = kwargs["cross_session_capability"]
            return handler(args, **handler_kwargs)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def _database_path_source(self) -> str:
        if self._config.database_path:
            return "config.database_path"
        if self._hermes_home:
            return "hermes_home"
        return "default_home"

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return operator-facing identity for the loaded LCM runtime.

        The public identity follows the same foreground-session view as
        ``lcm_status`` and other tools. When a side-channel session is bound,
        the bound session details are still exposed separately for diagnostics.
        """
        metadata = _plugin_metadata()
        git_identity = _git_runtime_identity(_PLUGIN_ROOT)
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = None
        lifecycle_error = ""
        if conversation_id:
            try:
                lifecycle_state = self._lifecycle.get_by_conversation(conversation_id)
            except Exception as exc:  # pragma: no cover - defensive
                lifecycle_error = str(exc)

        identity: Dict[str, Any] = {
            "engine": self.name,
            "plugin_name": metadata.get("name", "hermes-lcm"),
            "plugin_version": metadata.get("version", "unknown"),
            "plugin_path": str(_PLUGIN_ROOT),
            "module_path": str(Path(__file__).resolve()),
            "hermes_home": str(self._hermes_home or ""),
            "database_path": str(self._store.db_path),
            "database_path_source": self._database_path_source(),
            "session_id": session_id,
            "session_platform": self.current_session_platform,
            "session_bound": bool(session_id),
            "conversation_id": conversation_id,
            "lifecycle_current_session_id": "",
            "lifecycle_last_finalized_session_id": "",
        }
        if self.side_channel_active:
            identity.update({
                "bound_session_id": self._session_id,
                "bound_session_platform": self._session_platform,
                "bound_conversation_id": self._conversation_id,
            })
        identity.update(git_identity)
        if lifecycle_state is not None:
            identity.update({
                "lifecycle_current_session_id": lifecycle_state.current_session_id or "",
                "lifecycle_last_finalized_session_id": lifecycle_state.last_finalized_session_id or "",
            })
        if lifecycle_error:
            identity["lifecycle_error"] = lifecycle_error
        return identity

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "cache_metrics_available": self.cache_metrics_available,
            "cache_read_ratio": round(self.cache_read_ratio, 4),
            "hot_cache_signal": self.cache_signal_status(),
            "raw_context_length": self.raw_context_length,
            "context_length": self.context_length,
            "effective_context_length_cap": self.effective_context_length_cap,
            "effective_context_length_reason": self.effective_context_length_reason,
            "threshold_tokens": self.threshold_tokens,
            "post_compaction_target_tokens": self._post_compaction_target_tokens(),
            "emergency_pressure_ratio": self._config.emergency_pressure_ratio,
            "emergency_threshold_tokens": self._effective_emergency_threshold_tokens(),
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "ingest_failure_count": self._ingest_failure_count,
            "consecutive_ingest_failures": self._consecutive_ingest_failures,
            "last_ingest_error": self._last_ingest_error,
            "last_ingest_error_time": self._last_ingest_error_time,
            "model": self.model,
            "provider": self.provider,
            "context_length_source": self._context_length_source,
            "configured_context_threshold": self._config.context_threshold,
            "context_threshold": self.context_threshold,
            "context_threshold_source": self._context_threshold_source,
            "context_threshold_autoraised": self._context_threshold_autoraised,
            "compaction_policy": (
                self._compaction_policy.to_status_dict(self.context_length)
                if self._compaction_policy is not None
                else None
            ),
            "assembly_selection": dict(getattr(
                self,
                "_last_assembly_selection",
                {
                    "mode": "full-fit",
                    "items_considered": 0,
                    "items_evicted": 0,
                    "tokens_evicted": 0,
                },
            )),
            "fresh_tail_selection": dict(getattr(
                self,
                "_last_fresh_tail_selection",
                {
                    "count_limit": self._effective_fresh_tail_count(),
                    "token_limit": self._effective_fresh_tail_max_tokens(),
                    "selected_count": 0,
                    "selected_tokens": 0,
                    "boundary_reason": "not-evaluated",
                    "overflow": False,
                    "overflow_reason": "",
                },
            )),
            "config_sources": dict(getattr(self._config, "config_sources", {}) or {}),
            "config_source_warnings": list(getattr(self._config, "config_source_warnings", []) or []),
            "ignored_config_yaml_lcm_keys": list(getattr(self._config, "ignored_config_yaml_lcm_keys", []) or []),
            "async_compaction": self.get_async_compaction_status(),
            "full_sweep": dict(getattr(
                self,
                "_last_full_sweep_status",
                {
                    "reason": "not-run",
                    "passes": 0,
                    "partial": False,
                    "publication_count": 0,
                    "leaf_count": 0,
                    "condensation_count": 0,
                    "used_minimum_fanin": False,
                },
            )),
            "focus": self.get_focus_status(preview_chars=0),
        })
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = self._lifecycle.get_by_conversation(conversation_id) if conversation_id else None
        status["engine"] = "lcm"
        status["runtime_identity"] = self.get_runtime_identity()
        status["ingest_protection"] = sensitive_pattern_status(self._config)
        try:
            status["source_lineage"] = self._store.get_source_stats(session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        try:
            status["lifecycle_fragmentation"] = self._lifecycle.get_fragmentation_stats(
                state_db_path=self._state_db_path()
            )
        except Exception as exc:  # pragma: no cover - defensive
            status["lifecycle_fragmentation"] = {"error": str(exc), "read_only": True}
        try:
            rotate_backup_path = self.rotate_backup_path()
            status["rotate_backup_path"] = str(rotate_backup_path)
            # Single stat() to avoid a TOCTOU window where the rolling slot
            # could be atomically replaced between separate mtime and size reads.
            try:
                rotate_stat = rotate_backup_path.stat()
            except FileNotFoundError:
                rotate_stat = None
            if rotate_stat is not None:
                status["last_rotate_at"] = rotate_stat.st_mtime
                status["rotate_backup_size"] = rotate_stat.st_size
            else:
                status["last_rotate_at"] = None
                status["rotate_backup_size"] = 0
        except Exception as exc:  # pragma: no cover - defensive
            status["rotate_backup_path"] = None
            status["last_rotate_at"] = None
            status["rotate_backup_size"] = 0
            status["rotate_backup_error"] = str(exc)
        if session_id:
            status["store_messages"] = self._store.get_session_count(session_id)
            status["dag_nodes"] = self._dag.get_session_node_count(session_id)
            status["session_platform"] = self.current_session_platform
            status["session_ignored"] = self.current_session_ignored
            status["session_stateless"] = self.current_session_stateless
            status["ignore_session_patterns"] = list(self._config.ignore_session_patterns)
            status["stateless_session_patterns"] = list(self._config.stateless_session_patterns)
            status["ignore_message_patterns"] = list(self._config.ignore_message_patterns)
            status["ignore_session_patterns_source"] = self._config.ignore_session_patterns_source
            status["stateless_session_patterns_source"] = self._config.stateless_session_patterns_source
            status["ignore_message_patterns_source"] = self._config.ignore_message_patterns_source
            status["ignored_message_count"] = self._ignored_message_count
            status["ignore_pattern_dropped_count"] = self._ignore_pattern_dropped_count
            status["ingest_reconciliation"] = dict(self._last_ingest_reconciliation)
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = conversation_id
            if lifecycle_state is not None:
                status["lifecycle"] = {
                    "conversation_id": lifecycle_state.conversation_id,
                    "current_session_id": lifecycle_state.current_session_id,
                    "last_finalized_session_id": lifecycle_state.last_finalized_session_id,
                    "current_frontier_store_id": lifecycle_state.current_frontier_store_id,
                    "last_finalized_frontier_store_id": lifecycle_state.last_finalized_frontier_store_id,
                    "debt_kind": lifecycle_state.debt_kind,
                    "debt_size_estimate": lifecycle_state.debt_size_estimate,
                    "current_bound_at": lifecycle_state.current_bound_at,
                    "last_finalized_at": lifecycle_state.last_finalized_at,
                    "debt_updated_at": lifecycle_state.debt_updated_at,
                    "last_maintenance_attempt_at": lifecycle_state.last_maintenance_attempt_at,
                    "last_rollover_at": lifecycle_state.last_rollover_at,
                    "last_reset_at": lifecycle_state.last_reset_at,
                    "rollover_carry_over_context": (
                        lifecycle_state.rollover_carry_over_context
                    ),
                    "updated_at": lifecycle_state.updated_at,
                }
            try:
                telemetry = self._store.read_compaction_telemetry(conversation_id)
            except Exception:
                telemetry = None
            if telemetry:
                status["compaction_telemetry"] = {
                    "cache_state": telemetry.get("cache_state", "unknown"),
                    "consecutive_cold_observations": telemetry.get(
                        "consecutive_cold_observations", 0
                    ),
                    "turns_since_leaf_compaction": telemetry.get(
                        "turns_since_leaf_compaction", 0
                    ),
                    "peak_prompt_tokens_since_leaf_compaction": telemetry.get(
                        "peak_prompt_tokens_since_leaf_compaction", 0
                    ),
                    "last_observed_prompt_tokens": telemetry.get(
                        "last_observed_prompt_tokens", 0
                    ),
                    "last_observed_cache_read": telemetry.get("last_observed_cache_read", 0),
                    "last_observed_cache_write": telemetry.get("last_observed_cache_write", 0),
                    "activity_band": telemetry.get("activity_band", "low"),
                    "total_compactions": telemetry.get("total_compactions", 0),
                    "last_leaf_compaction_at": telemetry.get("last_leaf_compaction_at"),
                    "last_compaction_duration_ms": telemetry.get("last_compaction_duration_ms"),
                    "provider": telemetry.get("provider"),
                    "model": telemetry.get("model"),
                    "last_api_call_at": telemetry.get("last_api_call_at"),
                }
        return status

    def _resolve_live_compaction_policy(self) -> None:
        """Resolve policy from current route/session metadata and apply cutover."""
        self._compaction_policy = resolve_policy(
            model=self.model,
            provider=self.provider,
            route=self.api_mode,
            context_length=self.context_length,
            session_id=self.current_session_id,
            platform=self.current_session_platform,
            policy_rules=self._config.policy_rules,
            model_policies=self._config.model_policies,
            context_threshold=self._config.context_threshold,
            model_thresholds=self._config.model_thresholds,
            emergency_pressure_ratio=self._config.emergency_pressure_ratio,
            max_assembly_tokens=self._config.max_assembly_tokens,
            reserve_tokens_floor=self._config.reserve_tokens_floor,
            summary_model=self._config.summary_model,
            summary_fallback_models=tuple(self._config.summary_fallback_models) if self._config.summary_fallback_models else (),
            fresh_tail_count=self._config.fresh_tail_count,
            fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
            leaf_chunk_tokens=self._config.leaf_chunk_tokens,
            dynamic_leaf_chunk_enabled=self._config.dynamic_leaf_chunk_enabled,
            dynamic_leaf_chunk_max=self._config.dynamic_leaf_chunk_max,
            condensation_fanin=self._config.condensation_fanin,
            condensation_min_fanin=self._config.condensation_min_fanin,
            incremental_max_depth=self._config.incremental_max_depth,
            cache_friendly_condensation_enabled=self._config.cache_friendly_condensation_enabled,
            cache_economics=self._config.cache_economics,
            compaction_mode=self._config.compaction_mode,
            cache_ttl_seconds=self._config.cache_ttl_seconds,
            full_sweep_compaction_enabled=self._config.full_sweep_compaction_enabled,
            summary_prefix_target_tokens=self._config.summary_prefix_target_tokens,
            context_threshold_source=self._context_threshold_source,
            config_sources=getattr(self._config, "config_sources", None),
        )
        # Structured rules are first-class live cutover inputs. Legacy/builtin
        # policy derivation keeps the existing runtime-threshold resolver as the
        # compatibility authority (including Codex auto-raise behavior).
        if self._compaction_policy.source.startswith(
            ("policy_rule:", "model_policies:")
        ):
            self.context_threshold = self._compaction_policy.cutover_threshold
            self.threshold_percent = self.context_threshold
            if self.context_length > 0:
                self.threshold_tokens = self._effective_threshold_tokens(
                    self._compaction_policy.cutover_tokens(self.context_length)
                )

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        parent_session_id = self._in_process_parent_session_id({})
        if parent_session_id:
            logger.debug(
                "LCM model update ignored for auxiliary child of %s",
                parent_session_id,
            )
            return
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        # Resolve typed compaction policy for this model/provider/route/session.
        self._resolve_live_compaction_policy()
        logger.debug(
            "LCM resolved compaction policy: model=%s cutover=%.4f fingerprint=%s",
            self.model,
            self._compaction_policy.cutover_threshold,
            self._compaction_policy.fingerprint,
        )
        self._update_model_pending_session_start = True

    def _refresh_session_filters(self) -> None:
        self._session_match_keys = build_session_match_keys(
            self._session_id,
            platform=self._session_platform,
        )
        self._session_ignored = matches_session_pattern(
            self._session_match_keys,
            self._compiled_ignore_session_patterns,
        )
        self._session_stateless = (
            not self._session_ignored
            and (
                (
                    self._lcm_current_start_allows_bypass_lineage
                    and self._has_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)
                )
                or matches_session_pattern(
                    self._session_match_keys,
                    self._compiled_stateless_session_patterns,
                )
            )
        )
        if self._session_id:
            self._lcm_session_last_platform[self._session_id] = self._session_platform
            self._lcm_session_last_bypassed[self._session_id] = bool(self._session_ignored or self._session_stateless)
            if not self._session_ignored and not self._session_stateless:
                self._lcm_non_bypass_platforms.setdefault(self._session_id, set()).add(self._session_platform)
                self._lcm_session_last_normal_platform[self._session_id] = self._session_platform
        if self._session_ignored or self._session_stateless:
            self._mark_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)

    def _log_session_filter_diagnostics(self) -> None:
        if not self._logged_filter_config:
            if self._config.ignore_session_patterns:
                logger.info(
                    "LCM ignore_session_patterns from %s: %s",
                    self._config.ignore_session_patterns_source,
                    ", ".join(self._config.ignore_session_patterns),
                )
            if self._config.stateless_session_patterns:
                logger.info(
                    "LCM stateless_session_patterns from %s: %s",
                    self._config.stateless_session_patterns_source,
                    ", ".join(self._config.stateless_session_patterns),
                )
            if self._config.ignore_message_patterns:
                logger.info(
                    "LCM ignore_message_patterns from %s: %s",
                    self._config.ignore_message_patterns_source,
                    ", ".join(self._config.ignore_message_patterns),
                )
            self._logged_filter_config = True
        if self._session_ignored:
            logger.info(
                "LCM session %s matched ignore_session_patterns via %s — skipping writes and compaction",
                self._session_id,
                ", ".join(self._session_match_keys),
            )
        elif self._session_stateless:
            logger.info(
                "LCM session %s matched stateless_session_patterns via %s — read-only mode (no LCM writes)",
                self._session_id,
                ", ".join(self._session_match_keys),
            )

    # -- Internal: message ingestion ---------------------------------------

    def _schedule_ingest_cursor_reconciliation(self) -> None:
        """Mark existing-session rebinds for cursor repair on next ingest."""
        self._ingest_cursor_needs_reconcile = False
        if not self._session_id or self._session_ignored or self._session_stateless:
            return
        try:
            self._ingest_cursor_needs_reconcile = self._store.get_session_count(self._session_id) > 0
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation probe failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False

    def _stored_row_externalized_text_parts_for_pattern_matching(
        self,
        msg: Dict[str, Any],
        *,
        read_budget: dict[str, float | int] | None = None,
    ) -> list[str]:
        ref_sources: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            ref_sources.append(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                ref_sources.append(json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                ref_sources.append(str(tool_calls))
        refs: list[str] = []
        for source in ref_sources:
            for ref in extract_all_externalized_payload_refs(source):
                if ref not in refs:
                    refs.append(ref)
        parts: list[str] = []
        session_id = str(msg.get("session_id") or self._session_id or "")
        for ref in refs:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
                read_budget=read_budget,
                budget_label="source reconciliation ignore-pattern payload",
                max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
            )
            if not payload:
                continue
            payload_session_id = str(payload.get("session_id") or "")
            if session_id and payload_session_id and payload_session_id != session_id:
                continue
            payload_content = payload.get("content")
            if isinstance(payload_content, str):
                parts.append(payload_content)
        return parts

    def _stored_row_externalized_text_for_pattern_matching(
        self,
        msg: Dict[str, Any],
        *,
        read_budget: dict[str, float | int] | None = None,
    ) -> str:
        return "\n".join(
            self._stored_row_externalized_text_parts_for_pattern_matching(
                msg, read_budget=read_budget
            )
        )

    def _is_cached_active_replay_message_at_index(self, idx: int, msg: Dict[str, Any]) -> bool:
        if idx < 0 or idx >= len(self._last_active_replay_messages):
            return False
        return self._message_replay_identity(msg) == self._message_replay_identity(
            self._last_active_replay_messages[idx]
        )

    def _matches_ignore_message_patterns(
        self,
        msg: Dict[str, Any],
        *,
        stored_row: bool = False,
        read_budget: dict[str, float | int] | None = None,
    ) -> bool:
        if not self._compiled_ignore_message_patterns:
            return False
        content = msg.get("content")
        text = (
            stored_text_content_for_pattern_matching(content)
            if stored_row
            else text_content_for_pattern_matching(content)
        ) or ""
        if matches_message_pattern(text, self._compiled_ignore_message_patterns):
            return True
        if stored_row:
            externalized_parts = self._stored_row_externalized_text_parts_for_pattern_matching(
                msg, read_budget=read_budget
            )
            for externalized_text in externalized_parts:
                if externalized_text and matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns):
                    return True
            externalized_text = "\n".join(externalized_parts)
            if externalized_text and externalized_text != text:
                return matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns)
        return False

    def _content_has_externalized_placeholder_ref(self, content: str) -> bool:
        return bool(extract_externalized_ref(content) or extract_ingest_externalized_refs(content))

    def _has_prior_raw_externalized_placeholder_row(self, store_id: int, msg: Dict[str, Any]) -> bool:
        if not self._session_id:
            return False
        raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
        after_store_id = 0
        while True:
            rows = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=after_store_id,
                limit=1000,
            )
            if not rows:
                return False
            for row in rows:
                row_store_id = int(row.get("store_id") or 0)
                if row_store_id >= store_id:
                    return False
                if self._raw_externalized_placeholder_replay_identity(row) == raw_identity:
                    return True
                after_store_id = max(after_store_id, row_store_id)

    def _mapped_stored_row_matches_ignore_message_patterns(self, msg: Dict[str, Any]) -> bool:
        store_id = msg.get("store_id")
        content = normalize_content_value(msg.get("content")) or ""
        has_externalized_placeholder = self._content_has_externalized_placeholder_ref(content)
        mapped_from_active_placeholder = False
        if store_id is None:
            store_id = self._current_compress_store_ids_by_message_id.get(id(msg))
            mapped_from_active_placeholder = has_externalized_placeholder and store_id is not None
        if store_id is None:
            return False
        if mapped_from_active_placeholder and self._has_prior_raw_externalized_placeholder_row(int(store_id), msg):
            raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
            if self._current_compress_placeholder_identity_counts.get(raw_identity, 0) <= 1:
                return False
        try:
            stored = self._store.get(int(store_id))
        except Exception:
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        return bool(stored and self._matches_ignore_message_patterns(stored, stored_row=True))

    def _copy_active_replay_messages_preserving_generated_ids(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        copied_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in active_replay_messages:
            copied_message = dict(message)
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(copied_message))
            copied_replay_messages.append(copied_message)
        return copied_replay_messages

    def _remember_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in original_messages
        ]
        self._last_active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
            active_replay_messages
        )
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(active_replay_messages)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(active_replay_messages)
        )
        return active_replay_messages

    def _cached_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        identities = [self._message_replay_identity(message) for message in original_messages]
        if identities == getattr(self, "_last_active_replay_source_identities", None):
            cached = getattr(self, "_last_active_replay_messages", None)
            if cached is not None:
                return self._copy_active_replay_messages_preserving_generated_ids(cached)
        return None

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    def _restore_ingest_payload_placeholders_in_value(
        self,
        value: Any,
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(
                    key, session_id=session_id, read_budget=read_budget
                )
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(
                    val, session_id=session_id, read_budget=read_budget
                )
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [
                self._restore_ingest_payload_placeholders_in_value(
                    item, session_id=session_id, read_budget=read_budget
                )
                for item in value
            ]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
                read_budget=read_budget,
                budget_label="source reconciliation ingest payload",
                max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(
        self,
        content: str,
        *,
        session_id: str,
        read_budget: dict[str, float | int] | None = None,
    ) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
                read_budget=read_budget,
                budget_label="source reconciliation ingest payload",
                max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                    read_budget=read_budget,
                    budget_label="source reconciliation ingest payload",
                    max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
                    max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(
                decoded, session_id=session_id, read_budget=read_budget
            )
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
            read_budget=read_budget,
            budget_label="source reconciliation ingest payload",
            max_nested_depth=_PUBLICATION_LOCKED_MAX_NESTED_DEPTH,
            max_nested_items=_PUBLICATION_LOCKED_MAX_NESTED_ITEMS,
        )

    def _recovered_content_matches_durable_identity(self, recovered_content: str, durable_content: str) -> bool:
        recovered_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                self._config,
                parse_json_strings=False,
            )
        )
        if recovered_identity_content == durable_content:
            return True
        redaction_names = sorted(set(re.findall(r"\[LCM sensitive redaction: name=([^;\]]+)", durable_content)))
        if not redaction_names or bool(getattr(self._config, "sensitive_patterns_enabled", False)):
            return False
        compat_config = copy.copy(self._config)
        compat_config.sensitive_patterns_enabled = True
        compat_config.sensitive_patterns = redaction_names
        compat_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                compat_config,
                parse_json_strings=False,
            )
        )
        return compat_identity_content == durable_content

    @staticmethod
    def _persisted_output_marker_replay_proof(content: str) -> tuple[str | None, bool]:
        inline_preview_sha256 = _persisted_output_inline_preview_sha256(content)
        preview_sha256 = inline_preview_sha256 or _persisted_output_preview_prefix_digest(content)
        if not preview_sha256:
            return None, False
        allow_redacted_preview_match = inline_preview_sha256 is None and not _has_lossy_sensitive_redaction(content)
        return preview_sha256, allow_redacted_preview_match

    def _has_any_durable_persisted_output_payload_for_marker(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if expected_chars is None or not persisted_output_source_path or not persisted_output_preview_sha256:
            return False
        if recover_hermes_persisted_output_with_file_stat(content) is None:
            return False
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return durable_content is not None

    @classmethod
    def _is_active_context_droppable_identity(cls, identity: tuple[str, str, str, str]) -> bool:
        """Return true for durable rows sanitized out of active replay only."""
        role, content, _tool_call_id, tool_calls = identity
        if role != "assistant" or tool_calls:
            return False
        return _should_drop_active_assistant_message({
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        })

    def _ignored_message_is_quarantinable_assistant(self, msg: Dict[str, Any]) -> bool:
        if self._is_volatile_ignored_quarantine_placeholder(
            msg,
            text_content_for_pattern_matching(msg.get("content")) or "",
        ):
            return True
        identity = self._message_replay_identity(msg)
        if self._is_quarantined_assistant_replay_identity(identity):
            return True
        if not self._matches_ignore_message_patterns(msg):
            return False
        if identity[0] != "assistant":
            return False
        content = normalize_content_value(msg.get("content")) or ""
        return assistant_output_quarantine_reason(content) is not None

    def _redact_active_replay_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        redacted_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in messages:
            redacted_message = dict(message)
            if "content" in redacted_message:
                redacted_content = redact_sensitive_value(
                    redacted_message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
                redacted_message["content"] = redacted_content

            if "tool_calls" in redacted_message:
                redacted_message["tool_calls"] = redact_sensitive_value(
                    redacted_message.get("tool_calls"),
                    self._config,
                    parse_json_strings=True,
                )
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(redacted_message))
            redacted_replay_messages.append(redacted_message)
        return redacted_replay_messages

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist messages inside the shared session/rollover boundary.

        This is the single ingestion primitive used by live hooks, tool calls,
        preflight, foreground compression, session finalization, and promotion.
        Keeping the lifetime acquisition here makes a newly added caller safe
        by default. ``RLock`` preserves existing callers that already hold the
        boundary for a larger lifecycle transaction.
        """
        with self._storage_lifetime_lock:
            return self._ingest_messages_locked(messages)

    def _ingest_messages_locked(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.

        Returns a replay-safe copy of ``messages`` with obviously broken
        assistant loops replaced by quarantine placeholders. Existing callers may
        ignore the return value when they only need durable persistence.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return self._redact_active_replay_messages(messages)

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "Ingest skipped for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id,
            )
            return self._redact_active_replay_messages(messages)

        n = len(messages)
        cursor = min(max(self._ingest_cursor, 0), n)
        scan_start = 0 if self._ingest_cursor_needs_reconcile else cursor
        ignored_original_messages = [False] * n
        if self._compiled_ignore_message_patterns:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(messages)
            try:
                for idx in range(scan_start, n):
                    mapped_ignore = self._mapped_stored_row_matches_ignore_message_patterns(messages[idx])
                    ignored_original_messages[idx] = (
                        self._matches_ignore_message_patterns(messages[idx])
                        or mapped_ignore
                    )
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
        externalize_messages = [False] * n
        prefer_existing_externalized = [False] * n
        for idx in range(scan_start, n):
            externalize_messages[idx] = not ignored_original_messages[idx]
        for idx in range(0, scan_start):
            prefer_existing_externalized[idx] = not ignored_original_messages[idx]
        replay_messages = quarantine_suspicious_assistant_messages(
            messages,
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            externalize=externalize_messages,
            prefer_existing_externalized=prefer_existing_externalized,
        )
        replay_messages = self._redact_active_replay_messages(replay_messages)
        replay_messages = self._apply_ignored_active_replay_placeholders(
            messages,
            replay_messages,
            scan_start=scan_start,
            ignored_messages=ignored_original_messages,
        )
        if self._ingest_cursor_needs_reconcile:
            reconcile_messages = [
                original_msg
                if (
                    (
                        str(original_msg.get("role") or "") == "tool"
                        and _is_hermes_persisted_output_marker(
                            normalize_content_value(original_msg.get("content")) or ""
                        )
                        and self._has_any_durable_persisted_output_payload_for_marker(original_msg)
                    )
                    or (
                        self._compiled_ignore_message_patterns
                        and ignored_original_messages[idx]
                    )
                )
                else replay_msg
                for idx, (original_msg, replay_msg) in enumerate(zip(messages, replay_messages))
            ]
            self._ingest_cursor = self._reconcile_ingest_cursor_from_store(reconcile_messages)
            self._ingest_cursor_needs_reconcile = False
        cursor = min(max(self._ingest_cursor, 0), n)
        if cursor > 0:
            cached_source_identities = getattr(self, "_last_active_replay_source_identities", None)
            cached_active_replay_messages = getattr(self, "_last_active_replay_messages", None)
            if (
                cached_source_identities is not None
                and cached_active_replay_messages is not None
                and len(cached_source_identities) >= cursor
                and len(cached_active_replay_messages) >= cursor
            ):
                current_prefix_identities = [
                    self._message_replay_identity(message) for message in messages[:cursor]
                ]
                if current_prefix_identities == cached_source_identities[:cursor]:
                    replay_messages = (
                        self._copy_active_replay_messages_preserving_generated_ids(
                            cached_active_replay_messages[:cursor]
                        )
                        + replay_messages[cursor:]
                    )
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = replay_messages[cursor:] if cursor < n else []
        original_new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            cached_replay = self._cached_active_replay_messages(messages)
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            if cached_replay is not None:
                return cached_replay
            return self._remember_active_replay_messages(messages, replay_messages)

        active_replay_messages = replay_messages
        compression_boundary_ingest_pending = self._compression_boundary_ingest_pending
        empty_session_placeholder_budget: dict[str, int] = {}
        empty_session_placeholder_ordinals: dict[str, set[int]] = {}
        if not compression_boundary_ingest_pending and self._session_id:
            try:
                if self._store.get_session_count(self._session_id) == 0:
                    empty_session_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
                    empty_session_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            except Exception:
                empty_session_placeholder_budget = {}
                empty_session_placeholder_ordinals = {}
        messages_to_store_with_index: list[tuple[int, Dict[str, Any]]] = [
            (cursor + offset, replay_msg)
            for offset, replay_msg in enumerate(new_messages)
        ]
        if messages_to_store_with_index:
            kept: list[tuple[int, Dict[str, Any]]] = []
            boundary_placeholder_seen: dict[str, int] = {}
            boundary_seen_synthetic_summary_before = False
            empty_session_placeholder_seen: dict[str, int] = {}
            if empty_session_placeholder_ordinals and cursor > 0:
                for replay_msg in replay_messages[:cursor]:
                    replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(replay_text)
                    if digest:
                        empty_session_placeholder_seen[digest] = empty_session_placeholder_seen.get(digest, 0) + 1
            boundary_all_placeholder_replay_batch = (
                compression_boundary_ingest_pending
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            if compression_boundary_ingest_pending:
                boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                stored_counts = self._compression_boundary_stored_placeholder_digest_counts
                if boundary_budget and stored_counts:
                    incoming_counts: dict[str, int] = {}
                    relevant_digests = set(boundary_budget) | set(stored_counts)
                    for msg in new_messages:
                        text = text_content_for_pattern_matching(msg.get("content")) or ""
                        digest = self._active_replay_placeholder_digest(text)
                        if digest in relevant_digests:
                            incoming_counts[digest] = incoming_counts.get(digest, 0) + 1
                    adjusted_budget: dict[str, int] = {}
                    for digest, count in boundary_budget.items():
                        parsed_count = max(0, int(count or 0))
                        incoming_count = max(0, int(incoming_counts.get(digest, 0) or 0))
                        stored_count = max(0, int(stored_counts.get(digest, 0) or 0))
                        remaining = min(parsed_count, max(0, incoming_count - stored_count))
                        if remaining > 0:
                            adjusted_budget[digest] = remaining
                    self._compression_boundary_active_placeholder_digest_budget = adjusted_budget
            empty_session_all_placeholder_replay_batch = (
                bool(empty_session_placeholder_ordinals)
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            for offset, (original_msg, replay_msg) in enumerate(zip(original_new_messages, new_messages)):
                absolute_idx = cursor + offset
                replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
                volatile_placeholder = self._is_volatile_ignored_quarantine_placeholder(
                    replay_msg,
                    replay_text,
                )
                volatile_digest = self._active_replay_placeholder_digest(replay_text)
                generated_volatile_placeholder = volatile_placeholder and (
                    original_text != replay_text
                    or (
                        volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                )
                active_replay_placeholder = self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                active_replay_placeholder_digest = self._active_replay_placeholder_digest(replay_text)
                if not active_replay_placeholder:
                    replay_text_stripped = replay_text.strip()
                    if (
                        self._is_context_summary_content(replay_text)
                        or replay_text_stripped.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                        or replay_text_stripped.startswith(_PRESERVED_TODO_CONTEXT_PREFIX)
                    ):
                        boundary_seen_synthetic_summary_before = True
                compression_carried_active_placeholder = False
                metadata_replayed_active_placeholder = False
                if (
                    empty_session_placeholder_budget
                    and empty_session_placeholder_ordinals
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    empty_session_placeholder_seen[active_replay_placeholder_digest] = (
                        empty_session_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    ordinal = empty_session_placeholder_seen[active_replay_placeholder_digest]
                    remaining = empty_session_placeholder_budget.get(active_replay_placeholder_digest, 0)
                    if (
                        remaining > 0
                        and ordinal in empty_session_placeholder_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (ordinal > 1 or empty_session_all_placeholder_replay_batch)
                    ):
                        metadata_replayed_active_placeholder = True
                        if remaining == 1:
                            empty_session_placeholder_budget.pop(active_replay_placeholder_digest, None)
                        else:
                            empty_session_placeholder_budget[active_replay_placeholder_digest] = remaining - 1
                if (
                    compression_boundary_ingest_pending
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    boundary_placeholder_seen[active_replay_placeholder_digest] = (
                        boundary_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    current_placeholder_ordinal = boundary_placeholder_seen[active_replay_placeholder_digest]
                    boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                    boundary_ordinals = self._compression_boundary_active_placeholder_digest_ordinals
                    generated_message_ids = getattr(
                        self,
                        "_generated_ignored_active_replay_placeholder_message_ids",
                        set(),
                    )
                    has_generated_provenance = (
                        id(replay_msg) in generated_message_ids
                        or id(original_msg) in generated_message_ids
                    )
                    ordinal_matches_generated = (
                        current_placeholder_ordinal in boundary_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (
                            current_placeholder_ordinal > 1
                            or boundary_seen_synthetic_summary_before
                            or boundary_all_placeholder_replay_batch
                        )
                    )
                    if boundary_budget and (
                        has_generated_provenance
                        or (not has_generated_provenance and ordinal_matches_generated)
                    ):
                        remaining = boundary_budget.get(active_replay_placeholder_digest, 0)
                        if remaining > 0:
                            compression_carried_active_placeholder = True
                            if remaining == 1:
                                boundary_budget.pop(active_replay_placeholder_digest, None)
                            else:
                                boundary_budget[active_replay_placeholder_digest] = remaining - 1
                replayed_active_placeholder = active_replay_placeholder and (
                    self._is_cached_active_replay_message_at_index(absolute_idx, replay_msg)
                    or compression_carried_active_placeholder
                    or metadata_replayed_active_placeholder
                )
                if (
                    ignored_original_messages[absolute_idx]
                    or generated_volatile_placeholder
                    or replayed_active_placeholder
                ):
                    self._ignored_message_count += 1
                    if generated_volatile_placeholder and volatile_digest is not None:
                        self._remember_generated_ignored_placeholder_hash(volatile_digest)
                    replay_preserves_ignore_decision = (
                        self._is_volatile_ignored_quarantine_placeholder(replay_msg, replay_text)
                        or self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                    )
                    if ignored_original_messages[absolute_idx] and not replay_preserves_ignore_decision:
                        if active_replay_messages is replay_messages:
                            active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                                replay_messages
                            )
                        active_message = dict(active_replay_messages[absolute_idx])
                        active_message["content"] = self._ignored_active_replay_placeholder(original_text)
                        active_replay_messages[absolute_idx] = active_message
                    excerpt = original_text[:80].replace("\n", " ")
                    if ignored_original_messages[absolute_idx]:
                        # A raw message matched ignore_message_patterns and is
                        # discarded here - never persisted anywhere. Count and
                        # log it (INFO) so an over-broad pattern silently eating
                        # substantive turns is at least visible to the operator.
                        self._ignore_pattern_dropped_count += 1
                        logger.info(
                            "LCM ignore_message_patterns dropped %s message "
                            "(not persisted; total dropped=%d): %r",
                            original_msg.get("role", "unknown"),
                            self._ignore_pattern_dropped_count,
                            excerpt,
                        )
                    else:
                        logger.debug(
                            "LCM ignore_message_patterns dropped %s message: %r",
                            original_msg.get("role", "unknown"),
                            excerpt,
                        )
                    continue
                store_msg = replay_msg
                if (
                    str(original_msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(
                        normalize_content_value(original_msg.get("content")) or ""
                    )
                ):
                    store_msg = original_msg
                kept.append((absolute_idx, store_msg))
            messages_to_store_with_index = kept

        if not messages_to_store_with_index:
            self._ingest_cursor = n
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            return self._remember_active_replay_messages(messages, active_replay_messages)

        protected_messages = protect_messages_for_ingest(
            [msg for _idx, msg in messages_to_store_with_index],
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        for (absolute_idx, _replay_msg), protected_msg in zip(
            messages_to_store_with_index,
            protected_messages,
        ):
            if self._protected_message_uses_raw_payload_active_stub(protected_msg):
                if active_replay_messages is replay_messages:
                    active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                        replay_messages
                    )
                active_message = dict(active_replay_messages[absolute_idx])
                active_message["content"] = protected_msg["content"]
                active_replay_messages[absolute_idx] = active_message

        estimates = [count_message_tokens(m) for m in protected_messages]
        ingest_session_id = self._session_id
        ingest_conversation_id = self._conversation_id
        with self._frontier.publication_transaction() as conn:
            lifecycle = conn.execute(
                """
                SELECT current_session_id
                FROM lcm_lifecycle_state WHERE conversation_id = ?
                """,
                (ingest_conversation_id,),
            ).fetchone()
            durable_session_id = str(lifecycle[0] or "") if lifecycle else ""
            if lifecycle is not None and durable_session_id != ingest_session_id:
                raise RuntimeError(
                    "stale-session ingest rejected after durable session ownership changed"
                )
            self._store.append_protected_batch_no_commit(
                conn,
                ingest_session_id,
                protected_messages,
                estimates,
                source=self._session_platform,
                conversation_id=ingest_conversation_id,
            )
        self._ingest_cursor = n
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        logger.debug("Ingested %d messages into LCM store", len(messages_to_store_with_index))
        self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
        # Most ``protected_messages`` changes are storage-only: inline media,
        # tool results, and data/base64 substrings must stay provider-usable in
        # active replay. Whole-message ``raw_payload`` externalization is the
        # exception: it intentionally returns a compact active stub so the host
        # does not replay huge opaque text while SQLite stores only the stub.
        return self._remember_active_replay_messages(messages, active_replay_messages)

    @staticmethod
    def _protected_message_uses_raw_payload_active_stub(message: Dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(
            "[Externalized payload: kind=raw_payload;"
        )

    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        ids_by_message_id = self._get_store_id_map_for_messages(messages)
        return [ids_by_message_id[id(msg)] for msg in messages if id(msg) in ids_by_message_id]

    # -- Internal: summarization -------------------------------------------

    def _run_pre_compaction_extraction(self, messages: List[Dict[str, Any]]) -> None:
        """Best-effort extraction of decisions before compaction."""
        try:
            serialized = self._serialize_messages(messages)
            output_path = self._config.extraction_output_path
            if not output_path:
                base = self._hermes_home or os.path.expanduser("~/.hermes")
                output_path = os.path.join(base, "lcm-extractions")
            extraction_model = self._config.extraction_model or self._config.summary_model
            extract_before_compaction(
                serialized_messages=serialized,
                output_path=output_path,
                session_id=self._session_id or "",
                model=extraction_model,
                timeout=self._config.summary_timeout_ms / 1000,
            )
        except Exception as e:
            logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)

    def _maybe_gc_compacted_tool_results(
        self,
        compacted_chunk: List[Dict[str, Any]],
        source_store_ids: List[int],
    ) -> None:
        if not getattr(self._config, "large_output_transcript_gc_enabled", False):
            return
        if not compacted_chunk or not source_store_ids:
            return

        stored_by_id = self._store.get_batch(source_store_ids)
        for store_id in source_store_ids:
            stored = stored_by_id.get(store_id)
            if not stored or stored.get("session_id") != self._session_id:
                continue
            if stored.get("role") != "tool":
                continue
            content = stored.get("content", "") or ""
            tool_call_id = stored.get("tool_call_id", "") or ""
            if not content:
                continue

            # Only take the fast ref-branch when the ENTIRE row is the
            # externalized placeholder. A ref merely embedded in surrounding
            # text (e.g. a recall-tool result that quotes a placeholder) must
            # fall through to the content-equality lookup below, which tombstones
            # only when the full row content matches the stored payload -
            # otherwise the surrounding, never-externalized text is lost.
            ref = extract_externalized_ref(content) if is_externalized_placeholder(content) else None
            if ref:
                externalized = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None and externalized.get("kind", "tool_result") == "tool_result":
                    placeholder = build_transcript_gc_placeholder(externalized)
                    self._store.gc_externalized_tool_result(store_id, placeholder)
                    continue

            lookup_candidates = []
            sanitized_content = sanitize_pre_compaction_content(content)
            if sanitized_content and sanitized_content != content:
                lookup_candidates.append(sanitized_content)
            lookup_candidates.append(content)

            externalized = None
            for candidate in lookup_candidates:
                externalized = find_externalized_payload_for_message(
                    candidate,
                    tool_call_id=tool_call_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None:
                    break
            if externalized is None:
                continue

            placeholder = build_transcript_gc_placeholder(externalized)
            self._store.gc_externalized_tool_result(store_id, placeholder)

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize messages into labeled text for the summarizer."""
        parts = []
        matched_tool_ids = _matched_tool_call_ids(messages)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = redact_sensitive_value(
                msg.get("content") or "",
                self._config,
                parse_json_strings=False,
            )
            if role == "tool":
                tool_id = str(msg.get("tool_call_id") or "").strip()
                externalized = maybe_externalize_tool_output(
                    content,
                    tool_call_id=tool_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized:
                    content = externalized["placeholder"]
                else:
                    content = sanitize_pre_compaction_content(content)
                    if len(content) > 3000:
                        content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            content = sanitize_pre_compaction_content(content)

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                matched_tool_calls = [
                    tc for tc in tool_calls
                    if not _tool_call_id(tc) or _tool_call_id(tc) in matched_tool_ids
                ]
                if _is_synthetic_assistant_noise(content):
                    if not matched_tool_calls:
                        continue
                    content = ""
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                if matched_tool_calls:
                    tc_parts = []
                    for tc in matched_tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            args = redact_sensitive_value(
                                args,
                                self._config,
                                parse_json_strings=True,
                            )
                            args = sanitize_pre_compaction_tool_arguments(args)
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    # -- Internal: tool-pair sanitization ------------------------------------

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Drop unsafe assistant-only noise, then repair tool sequencing.

        This is intentionally active-context-only: callers pass the selected
        provider replay context, and this helper never mutates stored rows,
        source mappings, or DAG nodes.
        """
        cleaned: list[Dict[str, Any]] = []
        dropped_assistant_messages = 0
        stripped_assistant_messages = 0
        for msg in messages:
            msg = self._sanitize_active_preserved_objective_message(msg)
            if msg.get("role") == "assistant":
                cleaned_msg = _clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped_assistant_messages += 1
                    continue
                if cleaned_msg is not msg:
                    stripped_assistant_messages += 1
                cleaned.append(cleaned_msg)
                continue
            cleaned.append(msg)

        if dropped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: dropped %d assistant message(s) with no visible content",
                dropped_assistant_messages,
            )
        if stripped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: stripped internal content from %d assistant message(s)",
                stripped_assistant_messages,
            )

        return self._sanitize_tool_pairs(
            cleaned,
            insert_missing_tool_stubs=insert_missing_tool_stubs,
        )

    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-safe active-context tool-call/result sequencing.

        Raw store and DAG history remain lossless. This guardrail only sanitizes
        the active context emitted back to providers, where assistant tool calls
        must be followed immediately by their contiguous tool results. Late,
        duplicate, out-of-order, and orphan tool results are dropped; missing
        direct results get synthetic stubs.
        """
        sanitized: List[Dict[str, Any]] = []
        dropped_tool_results = 0
        inserted_stub_results = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "tool":
                dropped_tool_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                    if call_id
                ]
                # Snapshot the entire contiguous result run before matching.
                # Consuming one expected ID at a time used to discard a valid
                # out-of-order result needed by a later call in the same
                # assistant message. Bucket by ID, then emit provider order.
                contiguous_results: dict[str, list[Dict[str, Any]]] = {}
                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    next_msg = messages[i + 1]
                    next_id = str(next_msg.get("tool_call_id") or "").strip()
                    contiguous_results.setdefault(next_id, []).append(next_msg)
                    i += 1

                for expected_id in expected_ids:
                    candidates = contiguous_results.get(expected_id) or []
                    if candidates:
                        sanitized.append(candidates.pop(0))
                    elif insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stub_results += 1
                dropped_tool_results += sum(
                    len(candidates) for candidates in contiguous_results.values()
                )

            i += 1

        if dropped_tool_results:
            logger.info(
                "LCM tool-pair guardrail: dropped %d late/orphan/duplicate tool result(s)",
                dropped_tool_results,
            )
        if inserted_stub_results:
            logger.info(
                "LCM tool-pair guardrail: inserted %d missing tool-result stub(s)",
                inserted_stub_results,
            )

        return sanitized

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
        critical_budget_pressure: bool = False,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._effective_cache_friendly_condensation_enabled():
            return True, ""
        if force_overflow:
            return True, ""
        if critical_budget_pressure:
            return True, ""

        fanin = self._effective_condensation_fanin()
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _maybe_condense(
        self,
        focus_topic: Optional[str] = None,
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
        critical_budget_pressure: bool = False,
    ) -> None:
        """Check if any depth level has enough nodes for condensation."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._effective_incremental_max_depth()
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        condensed_any = False
        suppression_reason = ""

        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(
                self._session_id, depth
            )
            fanin = self._effective_condensation_fanin()
            if len(uncondensed) < fanin:
                continue

            allow_condense, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
            )
            if not allow_condense:
                suppression_reason = reason or suppression_reason
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:fanin]
            combined_text = "\n\n---\n\n".join(n.summary for n in to_condense)
            source_tokens = sum(n.token_count for n in to_condense)
            token_budget = max(1000, int(source_tokens * 0.40))

            summary_text, level = summarize_with_escalation(
                text=combined_text,
                source_tokens=source_tokens,
                token_budget=token_budget,
                depth=depth + 1,
                model=self._config.summary_model,
                fallback_models=self._config.summary_fallback_models,
                circuit_breaker=self._summary_circuit_breaker,
                spend_guard=self._summary_spend_guard,
                timeout=self._config.summary_timeout_ms / 1000,
                l2_budget_ratio=self._config.l2_budget_ratio,
                l3_truncate_tokens=self._config.l3_truncate_tokens,
                focus_topic=focus_topic or "",
                custom_instructions=self._config.custom_instructions,
            )

            earliest_at, latest_at = self._dag.get_source_time_window([n.node_id for n in to_condense])
            node = SummaryNode(
                session_id=self._session_id,
                depth=depth + 1,
                summary=summary_text,
                token_count=count_tokens(summary_text),
                source_token_count=source_tokens,
                source_ids=[n.node_id for n in to_condense],
                source_type="nodes",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            condensed_any = True

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, count_tokens(summary_text),
            )

            if leaf_compacted_this_turn and self._effective_cache_friendly_condensation_enabled():
                break

        if not condensed_any and leaf_compacted_this_turn and self._effective_cache_friendly_condensation_enabled():
            self._last_condensation_suppressed_reason = suppression_reason

    # -- Internal: context assembly ----------------------------------------

    @staticmethod
    def _append_lcm_note_to_content(content: Any) -> Any:
        note = (
            "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
            "and lcm_expand to recover original details from any summary.]"
        )
        if isinstance(content, str):
            return content + note
        note_part = {"type": "text", "text": note.lstrip()}
        if content is None:
            return note.lstrip()
        if isinstance(content, list):
            return list(content) + [note_part]
        normalized = normalize_content_value(content) or ""
        return normalized + note

    @staticmethod
    def _is_preserved_todo_context_message(message: Dict[str, Any]) -> bool:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX)

    @staticmethod
    def _preserved_objective_context_content(message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) else ""

    def _sanitized_preserved_objective_context_content(self, message: Dict[str, Any]) -> str:
        preserved_objective = self._preserved_objective_context_content(message)
        if not preserved_objective:
            return ""
        return self._sanitize_preserved_objective_content(
            preserved_objective,
            role=str(message.get("role") or "user"),
        )

    def _sanitize_active_preserved_objective_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_content = self._sanitized_preserved_objective_context_content(message)
        if not sanitized_content or sanitized_content == message.get("content"):
            return message
        sanitized = dict(message)
        sanitized["content"] = sanitized_content
        return sanitized

    def _sanitize_preserved_objective_content(self, content: str, role: str = "user") -> str:
        content = strip_injected_context_blocks(content)
        content = protect_inline_payloads_in_text(
            content,
            role=role,
            session_id=self._session_id,
            field_path="preserved_objective.content",
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return content

    def _build_preserved_objective_summary_part(self, message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        content = self._sanitize_preserved_objective_content(
            content,
            role=str(message.get("role") or "user"),
        )
        return f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\n{content}"

    def _latest_user_context_anchor(
        self,
        messages: List[Dict[str, Any]],
        selected_tail: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a scaffolded newest real user objective omitted from the tail.

        Tool-heavy turns can push the operative user request outside the fresh
        tail while retaining only assistant/tool traces from that turn.  The
        returned text is active-context scaffolding, not raw conversation: it is
        emitted inside the summary block so restart reconciliation ignores it
        instead of ingesting a duplicate non-contiguous user message.

        Previous preserved-objective scaffolds are derived context, not real
        user turns, so they are not eligible as the next anchor source. Once a
        reverse scan reaches one, older user turns are stale relative to that
        synthetic continuity marker and must not be promoted as current intent.
        """
        selected_tail_messages = [msg for msg in selected_tail if isinstance(msg, dict)]
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content_text = text_content_for_pattern_matching(message.get("content")) or ""
            if (
                self._matches_ignore_message_patterns(message)
                or self._mapped_stored_row_matches_ignore_message_patterns(message)
                or self._is_volatile_ignored_quarantine_placeholder(
                    message,
                    content_text,
                )
                or self._is_ignored_active_replay_placeholder(message, content_text)
            ):
                continue
            if self._preserved_objective_context_content(message):
                return None
            if message.get("role") != "user":
                continue
            if self._is_preserved_todo_context_message(message):
                continue
            if any(message == selected for selected in selected_tail_messages):
                return None
            return self._build_preserved_objective_summary_part(message)
        return None

    def _focus_node_max_store_id(self, node: SummaryNode) -> int:
        """Return a bounded descendant raw watermark for one canonical node."""
        maximum = 0
        stack = [node]
        visited: set[int] = set()
        remaining = max(64, int(self._config.focus_max_source_nodes) * 64)
        while stack and remaining > 0:
            current = stack.pop()
            node_id = int(current.node_id or 0)
            if node_id in visited:
                continue
            visited.add(node_id)
            remaining -= 1
            if current.source_type == "messages":
                for source_id in current.source_ids:
                    try:
                        maximum = max(maximum, int(source_id))
                    except (TypeError, ValueError):
                        continue
                continue
            if current.source_type == "nodes":
                for child_id in current.source_ids:
                    child = self._dag.get_node(int(child_id))
                    if child is not None and child.session_id == current.session_id:
                        stack.append(child)
        return maximum

    def get_focus_status(self, *, preview_chars: int = 0) -> Dict[str, Any]:
        conversation_id = self.current_conversation_id
        if not conversation_id:
            return {"active": False, "reason": "no-bound-conversation"}
        active = self._focus.get_active(conversation_id)
        if active is None:
            return {"active": False, "history_count": len(self._focus.history(conversation_id, limit=200))}
        result = active.metadata(preview_chars=preview_chars)
        result["active"] = True
        result["history_count"] = len(self._focus.history(conversation_id, limit=200))
        return redact_sensitive_value(result, self._config, parse_json_strings=True)

    def create_focus(self, prompt: str, *, refocus: bool = False) -> Dict[str, Any]:
        """Synthesize and atomically publish one immutable focus brief."""
        prompt = str(prompt or "").strip()
        conversation_id = self.current_conversation_id
        session_id = self.current_session_id
        if not conversation_id or not session_id:
            return {"error": "focus requires a bound conversation and session"}
        previous = self._focus.get_active(conversation_id)
        if refocus:
            if previous is None:
                return {"error": "no active focus to refocus"}
            if not prompt:
                prompt = previous.prompt
        elif not prompt:
            return {"error": "focus prompt is required"}
        # Focus is a persisted/output overlay, not lossless ingest. Mandatory
        # high-confidence redaction applies regardless of ingest policy.
        prompt = redact_sensitive_output_text(prompt)

        max_nodes = max(1, int(self._config.focus_max_source_nodes))
        if refocus:
            # Rank matching nodes first, then fill from canonical session nodes;
            # only post-watermark DAG deltas are eligible.
            ranked = self._dag.search(prompt, session_id=session_id, limit=max_nodes * 4)
            ranked_ids = {int(node.node_id) for node in ranked}
            ranked.extend(
                node for node in self._dag.get_session_nodes(session_id, limit=max_nodes * 16)
                if int(node.node_id) not in ranked_ids
            )
            nodes = [
                node for node in ranked
                if self._focus_node_max_store_id(node) > int(previous.covered_store_id)
            ][:max_nodes]
            if not nodes:
                return redact_sensitive_value({
                    "error": "no post-focus DAG delta is available",
                    "previous_focus_preserved": True,
                    "focus": previous.metadata(),
                }, self._config, parse_json_strings=True)
        else:
            nodes = self._dag.search(prompt, session_id=session_id, limit=max_nodes)
            if not nodes:
                return {"error": "no matching canonical LCM summary evidence found"}

        context_budget = max(1, int(self._config.focus_context_tokens))
        context_blocks: list[dict[str, Any]] = []
        budget_used = 0
        if refocus and previous is not None:
            prior_block = {
                "type": "previous_focus_brief",
                "focus_id": previous.focus_id,
                "content": previous.content,
                "source_node_ids": list(previous.source_node_ids),
            }
            prior_tokens = lcm_tools._context_content_token_count([
                {"type": "summary", "summary": previous.content}
            ])
            if prior_tokens <= context_budget:
                context_blocks.append(prior_block)
                budget_used += prior_tokens

        selected_nodes: list[SummaryNode] = []
        truncated = False
        for node in nodes:
            remaining = max(0, context_budget - budget_used)
            if remaining <= 0:
                truncated = True
                break
            blocks = lcm_tools._collect_context_blocks_for_node(
                self,
                node,
                max_tokens=remaining,
                hydrate_externalized_content=False,
                allowed_session_id=session_id,
            )
            for block in blocks:
                block["session_id"] = session_id
            used = lcm_tools._context_content_token_count(blocks)
            context_blocks.extend(blocks)
            budget_used += used
            selected_nodes.append(node)
            truncated = truncated or any(
                block.get("summary_truncated")
                or block.get("pagination", {}).get("has_more")
                for block in blocks
            )

        if not selected_nodes:
            return {
                "error": "no focus evidence fit within the context budget",
                "previous_focus_preserved": bool(previous),
            }
        source_node_ids = list(dict.fromkeys(
            ([*previous.source_node_ids] if refocus and previous is not None else [])
            + [int(node.node_id) for node in selected_nodes]
        ))
        synthesis_prompt = (
            f"Create an immutable focus brief for this objective: {prompt}\n"
            "Cite material claims with [node N]. Mark unsupported or conflicting details as uncertain. "
            "Preserve actionable decisions, constraints, and open work."
        )
        if refocus:
            synthesis_prompt += " Update the previous brief using only the supplied post-focus DAG delta."
        model = self._config.expansion_model or self._config.summary_model or ""
        try:
            content = lcm_tools._synthesize_expansion_answer(
                prompt=synthesis_prompt,
                context_blocks=context_blocks,
                model=model,
                max_tokens=max(1, int(self._config.focus_output_tokens)),
                timeout=max(0.001, float(self._config.focus_timeout_ms) / 1000.0),
            )
        except TimeoutError:
            return {
                "error": "focus synthesis timed out",
                "previous_focus_preserved": bool(previous),
            }
        except Exception:
            logger.warning("LCM focus synthesis failed; preserving the active brief")
            return {
                "error": "focus synthesis failed",
                "previous_focus_preserved": bool(previous),
            }
        content = str(content or "").strip()
        if not content:
            return {
                "error": "focus synthesis returned an empty brief",
                "previous_focus_preserved": bool(previous),
            }
        content = redact_sensitive_output_text(content)
        frontier = self._frontier.get_active_frontier(conversation_id) or {}
        covered_store_id = max(
            [self._focus_node_max_store_id(node) for node in selected_nodes]
            + ([int(previous.covered_store_id)] if previous is not None else [0])
        )
        brief = self._focus.publish(
            conversation_id=conversation_id,
            session_id=session_id,
            prompt=prompt,
            content=content,
            source_node_ids=source_node_ids,
            covered_generation=int(frontier.get("generation") or 0),
            covered_store_id=covered_store_id,
            token_count=count_tokens(content),
            supersedes_focus_id=previous.focus_id if previous is not None else None,
        )
        result = redact_sensitive_value(
            brief.metadata(preview_chars=500),
            self._config,
            parse_json_strings=True,
        )
        result.update({
            "active": True,
            "refocus": refocus,
            "context_tokens_used": budget_used,
            "context_truncated": truncated,
            "model": redact_sensitive_output_text(model),
        })
        return result

    def unfocus(self) -> Dict[str, Any]:
        conversation_id = self.current_conversation_id
        if not conversation_id:
            return {"error": "unfocus requires a bound conversation"}
        brief = self._focus.unfocus(conversation_id)
        if brief is None:
            return {"active": False, "reason": "no-active-focus"}
        return {
            "active": False,
            "deactivated_focus_id": brief.focus_id,
            "history_preserved": True,
        }

    def _assemble_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        include_lcm_note: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [leading anchor, normally system prompt]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []
        self._last_assembly_selection = {
            "mode": "full-fit",
            "items_considered": 0,
            "items_evicted": 0,
            "tokens_evicted": 0,
        }

        # Once a positive active generation has ordered items, that generation
        # is the provider-visible layout. Host messages are only an input seam;
        # they cannot make covered raw rows or unrelated canonical nodes active.
        host_tail_messages = list(tail_messages)
        authoritative_layout = self._resolve_active_frontier_for_assembly()
        authoritative_nodes: Optional[List[SummaryNode]] = None
        if authoritative_layout is not None:
            authoritative_nodes, frontier_messages = authoritative_layout
            frontier_messages = (
                self._drop_preexisting_generated_ignored_dependent_eof_replies(
                    frontier_messages,
                    self._load_generated_ignored_dependent_reply_records(),
                )
            )
            tail_messages = self._merge_unpublished_host_tail(
                frontier_messages,
                host_tail_messages,
            )

        # Leading anchor with optional LCM annotation. Only a true system prompt
        # is a safe permanent anchor; gateway sessions can start directly with
        # user messages, and those user turns must remain compactable.
        leading_msg = system_msg.copy() if system_msg is not None else None
        if leading_msg is not None:
            if (
                leading_msg.get("role") == "system"
                and self.compression_count == 0
                and include_lcm_note
            ):
                leading_msg["content"] = self._append_lcm_note_to_content(
                    leading_msg.get("content", "")
                )
            result.append(leading_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._assembly_token_budget()
        )
        # The typed post-compaction target is a convergence goal, not
        # permission to discard raw messages that have not been published into
        # the DAG. Only an explicit hard rail (or emergency override) may evict
        # tail messages during assembly. The softer policy target still limits
        # how many summaries are inserted around the lossless raw tail.
        tail_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        tail_selected = tail_messages
        anchor_source = getattr(self, "_pending_context_anchor_messages", None)
        if anchor_source is None:
            anchor_source = host_tail_messages
        anchor_part: Optional[str] = None
        summary_budget = None
        used = count_message_tokens(leading_msg) if leading_msg is not None else 0
        if tail_cap is not None:
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            tail_for_selection = self._sanitize_active_context_messages(
                tail_messages,
                insert_missing_tool_stubs=False,
            )
            skipped_tail_gap = False
            for msg in reversed(tail_for_selection):
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > tail_cap:
                    if self._is_budget_droppable_tail_message(msg):
                        skipped_tail_gap = True
                        continue
                    break
                if skipped_tail_gap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
        else:
            tail_token_total = count_messages_tokens(tail_selected)
        if assembly_cap is not None:
            summary_budget = max(0, assembly_cap - used - tail_token_total)
        if anchor_source is not None:
            anchor_part = self._latest_user_context_anchor(anchor_source, tail_selected)

        # Collect the immutable focus overlay plus canonical DAG summaries.
        # Focus is an assembly-only view: it never mutates nodes/frontiers and
        # the ordinary newer summaries and fresh tail remain eligible.
        summary_parts: list[str] = []
        last_role = result[-1].get("role", "system") if result else "system"
        if not result or result[-1].get("role") == "system":
            # The summary becomes the first provider-visible message: either no
            # leading anchor exists (gateway-style assembly) or the system
            # prompt is the only anchor, which Anthropic extracts into a
            # separate field. Either way messages[0] must be role "user"; an
            # assistant summary here is rejected with HTTP 400 after the second
            # compaction.
            summary_role = "user"
        else:
            summary_role = "assistant" if last_role != "assistant" else "user"
        protected_summary_parts: list[str] = []
        if anchor_part is not None:
            anchor_msg = {"role": summary_role, "content": anchor_part}
            if summary_budget is None or count_message_tokens(anchor_msg) <= summary_budget:
                summary_parts.append(anchor_part)
                protected_summary_parts.append(anchor_part)

        active_focus = None
        conversation_id = self.current_conversation_id
        if conversation_id:
            try:
                active_focus = self._focus.get_active(conversation_id)
            except Exception:
                logger.warning("LCM could not load active focus overlay", exc_info=True)
        if active_focus is not None:
            focus_part = (
                f"[Active Focus Brief #{active_focus.focus_id}; immutable evidence nodes: "
                f"{', '.join(str(value) for value in active_focus.source_node_ids) or 'none'}]\n"
                f"{active_focus.content}"
            )
            protected_candidate = "\n\n---\n\n".join(
                [*protected_summary_parts, focus_part]
            )
            focus_msg = {"role": summary_role, "content": protected_candidate}
            if summary_budget is None or count_message_tokens(focus_msg) <= summary_budget:
                summary_parts.append(focus_part)
                protected_summary_parts.append(focus_part)

        all_nodes = (
            authoritative_nodes
            if authoritative_nodes is not None
            else self._dag.get_session_nodes(self._session_id)
        )
        if all_nodes:
            if authoritative_nodes is not None:
                node_groups = [(node.depth, [node]) for node in all_nodes]
            else:
                # Legacy/no-frontier fallback: infer the visible DAG prefix.
                depths = sorted(set(n.depth for n in all_nodes), reverse=True)
                node_groups = [
                    (depth, self._dag.get_uncondensed_at_depth(self._session_id, depth))
                    for depth in depths
                ]
            for d, visible_nodes in node_groups:
                for node in visible_nodes:
                    depth_label = {
                        0: "Recent",
                        1: "Session Arc",
                        2: "Durable",
                    }.get(d, f"Depth-{d}")
                    summary_parts.append(
                        f"[{depth_label} Summary (d{d}, node {node.node_id})]\n"
                        f"{node.summary}\n"
                        f"[Expand for details: {node.expand_hint}]"
                    )

        if summary_parts:
            selected_parts = summary_parts
            if summary_budget is not None:
                anchor_indices = [
                    index
                    for index, part in enumerate(summary_parts)
                    if part in protected_summary_parts
                ]
                candidate_order = [
                    index
                    for index in range(len(summary_parts))
                    if index not in anchor_indices
                ]
                selection_mode = "chronological"
                prompt_terms = self._prompt_aware_search_terms(tail_messages)
                if (
                    self._config.prompt_aware_eviction_enabled
                    and prompt_terms
                ):
                    candidate_order = candidate_order[-_PROMPT_AWARE_MAX_SUMMARIES:]
                    candidate_order.sort(
                        key=lambda index: (
                            -self._prompt_aware_relevance_score(
                                summary_parts[index],
                                prompt_terms,
                            ),
                            index,
                        )
                    )
                    selection_mode = "prompt-aware"
                selected_indices: list[int] = list(anchor_indices)
                for index in candidate_order:
                    part = summary_parts[index]
                    ordered_indices = sorted(selected_indices + [index])
                    candidate = "\n\n---\n\n".join(
                        summary_parts[selected_index]
                        for selected_index in ordered_indices
                    )
                    candidate_msg = {"role": summary_role, "content": candidate}
                    if count_message_tokens(candidate_msg) > summary_budget:
                        continue
                    selected_indices.append(index)
                selected_parts = [
                    summary_parts[index] for index in sorted(selected_indices)
                ]
                if len(selected_parts) == len(summary_parts):
                    selection_mode = "full-fit"
                self._last_assembly_selection = {
                    "mode": selection_mode,
                    "items_considered": len(summary_parts),
                    "items_evicted": len(summary_parts) - len(selected_parts),
                    "tokens_evicted": max(
                        0,
                        count_tokens("\n\n---\n\n".join(summary_parts))
                        - count_tokens("\n\n---\n\n".join(selected_parts)),
                    ),
                }
            else:
                self._last_assembly_selection = {
                    "mode": "full-fit",
                    "items_considered": len(summary_parts),
                    "items_evicted": 0,
                    "tokens_evicted": 0,
                }
            if selected_parts:
                combined = "\n\n---\n\n".join(selected_parts)
                result.append({"role": summary_role, "content": combined})

        # Fresh tail
        result.extend(tail_selected)

        # ── Active-context cleanup / tool-pair guardrail ──
        # Drop assistant turns that carry only blank/internal structured content,
        # then ensure provider-valid tool-call/result sequencing.
        result = self._sanitize_active_context_messages(result)
        if leading_msg is None:
            while result and result[0].get("role") in {"assistant", "tool"}:
                result = result[1:]
        if (
            assembly_cap is not None
            and anchor_part is not None
            and count_messages_tokens(result) > assembly_cap
        ):
            trimmed_result: list[Dict[str, Any]] = []
            for msg in result:
                content = normalize_content_value(msg.get("content")) or ""
                if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX not in content:
                    trimmed_result.append(msg)
                    continue
                parts = [
                    part for part in content.split("\n\n---\n\n")
                    if not part.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                ]
                if parts:
                    trimmed = msg.copy()
                    trimmed["content"] = "\n\n---\n\n".join(parts)
                    trimmed_result.append(trimmed)
            result = self._sanitize_active_context_messages(trimmed_result)

        return result

    @staticmethod
    def _prompt_aware_relevance_score(text: str, terms: list[str]) -> float:
        words = re.findall(r"[\w-]+", (text or "").lower())
        if not words:
            return 0.0
        frequencies: dict[str, int] = {}
        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1
        length_normalizer = 1.0 + (len(words) / 200.0)
        return sum(
            (1.0 + math.log(float(frequencies.get(term, 0)))) / length_normalizer
            for term in terms
            if frequencies.get(term, 0) > 0
        )

    @staticmethod
    def _prompt_aware_search_terms(
        tail_messages: List[Dict[str, Any]],
    ) -> list[str]:
        prompt = ""
        for message in reversed(tail_messages):
            if message.get("role") == "user":
                prompt = normalize_content_value(message.get("content")) or ""
                break
        stop = {"about", "after", "again", "could", "from", "have", "into", "that", "the", "this", "what", "when", "where", "which", "with", "would", "your"}
        terms: list[str] = []
        seen: set[str] = set()
        for term in re.findall(
            r"[\w-]{2,}", prompt[:_PROMPT_AWARE_MAX_PROMPT_CHARS].lower()
        ):
            if term in stop or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= _PROMPT_AWARE_MAX_TERMS:
                break
        return terms

    def _resolve_active_frontier_for_assembly(
        self,
    ) -> Optional[tuple[List[SummaryNode], List[Dict[str, Any]]]]:
        """Resolve one positive active generation into canonical prompt items.

        ``None`` means no authoritative positive generation exists and permits
        the legacy DAG/host fallback. Once a positive generation exists, missing,
        overlapping, cross-session, or unknown refs are invariant violations and
        raise instead of silently replaying the caller's stale host context.
        """
        conversation_id = self.current_conversation_id
        if not conversation_id:
            return None
        frontier = self._frontier.get_active_frontier(conversation_id)
        if frontier is None or int(frontier.get("source_end_store_id") or 0) <= 0:
            return None
        generation = int(frontier["generation"])
        items = self._frontier.get_frontier_items(conversation_id, generation)
        if not items:
            self.reconcile_itemless_frontier_generations(conversation_id)
            items = self._frontier.get_frontier_items(conversation_id, generation)
        if not items:
            raise RuntimeError(
                f"active frontier generation {generation} has no ordered items"
            )

        session_id = str(frontier.get("session_id") or "")
        nodes: list[SummaryNode] = []
        messages: list[Dict[str, Any]] = []
        previous_end = 0
        for item in items:
            start = int(item.get("source_start") or 0)
            end = int(item.get("source_end") or 0)
            ref_id = int(item.get("ref_id") or 0)
            kind = str(item.get("kind") or "")
            if start <= 0 or end < start or start <= previous_end:
                raise RuntimeError(
                    f"invalid frontier item range in generation {generation}"
                )
            previous_end = end
            if kind == "node":
                node = self._dag.get_node(ref_id)
                if node is None or node.session_id != session_id:
                    raise RuntimeError(
                        f"frontier generation {generation} references missing canonical node {ref_id}"
                    )
                nodes.append(node)
                continue
            if kind == "message":
                if ref_id < start or ref_id > end:
                    raise RuntimeError(
                        f"frontier generation {generation} has out-of-range message ref {ref_id}"
                    )
                stored = self._store.get(ref_id)
                stored_conversation_id = str(
                    stored.get("conversation_id") or ""
                ) if stored else ""
                if stored is None or stored_conversation_id != conversation_id:
                    raise RuntimeError(
                        f"frontier generation {generation} references missing raw message {ref_id}"
                    )
                messages.append(self._store.to_openai_msg(stored))
                continue
            raise RuntimeError(
                f"frontier generation {generation} has unknown item kind {kind!r}"
            )
        return nodes, messages

    def _merge_unpublished_host_tail(
        self,
        frontier_messages: List[Dict[str, Any]],
        host_tail_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Append genuinely new host rows without replaying covered history.

        A promotion can win immediately before foreground ingest hits a bounded
        SQLite failure. Those not-yet-stored host rows are newer than the
        generation and must survive the canonical fallback. Durable rows at or
        below the generation boundary, however, are covered and never reappear.
        """
        frontier = self._frontier.get_active_frontier(self.current_conversation_id)
        if frontier is None:
            return list(frontier_messages)
        session_id = str(frontier.get("session_id") or "")
        source_end = int(frontier.get("source_end_store_id") or 0)
        covered_rows: list[Dict[str, Any]] = []
        next_store_id = 0
        while next_store_id <= source_end:
            page = self._store.get_range(
                session_id,
                start_id=next_store_id,
                end_id=source_end,
                limit=1_000,
            )
            if not page:
                break
            covered_rows.extend(page)
            last_store_id = int(page[-1].get("store_id") or 0)
            if last_store_id < next_store_id:
                break
            next_store_id = last_store_id + 1

        def counts(messages, *, stored_row: bool) -> dict[tuple[Any, ...], int]:
            result: dict[tuple[Any, ...], int] = {}
            for message in messages:
                identity = self._message_replay_identity(
                    message,
                    stored_row=stored_row,
                )
                result[identity] = result.get(identity, 0) + 1
            return result

        covered_counts = counts(covered_rows, stored_row=True)
        frontier_counts = counts(frontier_messages, stored_row=False)
        unpublished: list[Dict[str, Any]] = []
        for message in host_tail_messages:
            identity = self._message_replay_identity(message)
            if covered_counts.get(identity, 0) > 0:
                covered_counts[identity] -= 1
                continue
            if frontier_counts.get(identity, 0) > 0:
                frontier_counts[identity] -= 1
                continue
            unpublished.append(message)
        return list(frontier_messages) + unpublished

    def _is_budget_droppable_tail_message(self, message: Dict[str, Any]) -> bool:
        """Return whether an over-budget tail message may be evicted.

        User turns are prompt-bearing context and stop tail selection when they
        cannot fit. Assistant/tool turns are derived context; if one bulky turn
        blocks older prompt material, skip it and keep scanning for budgetable
        user intent or compact status that still fits.
        """
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            return False
        content = normalize_content_value(message.get("content")) or ""
        if _PRESERVED_TODO_CONTEXT_PREFIX in content:
            return False
        if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in content:
            return False
        return True

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages:
            self._last_compression_status = "overflow_recovery"
            self._last_compression_noop_reason = ""
            self._ingest_cursor = len(compressed)
            self._ingest_cursor_needs_reconcile = False
            logger.info(
                "LCM assembly guardrail recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = (
                "forced overflow recovery found no droppable active-context messages"
            )

        effective_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )
        if effective_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > effective_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d; returning best-effort context (%d tokens)",
                    effective_cap,
                    count_messages_tokens(compressed),
                )
        return compressed

    def _effective_emergency_threshold_tokens(self) -> Optional[int]:
        """Resolve emergency provider-window pressure independently of assembly."""
        ratio = float(self._config.emergency_pressure_ratio)
        if self.context_length <= 0 or ratio <= 0:
            return None
        return max(1, math.ceil(self.context_length * ratio))

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Force recovery only at configured provider-window pressure."""
        emergency_tokens = self._effective_emergency_threshold_tokens()
        if emergency_tokens is None:
            return False

        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= emergency_tokens

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        if not candidates:
            return None
        return max(candidates)

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        recovery_cap = self._effective_assembly_token_cap()
        if recovery_cap is None:
            emergency_threshold = self._effective_emergency_threshold_tokens()
            if emergency_threshold is not None:
                recovery_cap = max(1, emergency_threshold - 1)
        if recovery_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return recovery_cap

        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, recovery_cap - overhead_tokens)

    def _post_compaction_target_tokens(self) -> Optional[int]:
        """Return the independent post-compaction target (provider-prompt scale).

        The host-visible cutover (``threshold_tokens``) remains the sole
        ordinary preflight trigger. Once cutover fires, leaf compaction
        converges toward this target.

        Sources (most restrictive wins):
        - typed policy target when a policy is resolved
        - half of the live cutover (``DEFAULT_TARGET_RATIO``) when cutover > 0
        - explicit max_assembly_tokens / reserve_tokens_floor rails

        Zero legacy assembly caps must not hide the half-cutover target.
        """
        caps: list[int] = []

        if self._compaction_policy is not None and self.context_length > 0:
            policy_target = self._compaction_policy.post_compaction_target_tokens(
                self.context_length
            )
            if policy_target is not None:
                caps.append(int(policy_target))

        # Always honour the live host cutover: target is half cutover by
        # default. This stays correct even if policy resolution and
        # threshold resolution temporarily disagree on the cutover ratio.
        if self.threshold_tokens > 0:
            caps.append(max(1, int(self.threshold_tokens * DEFAULT_TARGET_RATIO)))

        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is not None:
            caps.append(assembly_cap)

        if not caps:
            return None
        return max(1, min(caps))

    def _assembly_token_budget(self) -> Optional[int]:
        """Default budget for ``_assemble_context`` (message-token scale).

        Prefer the resolved typed policy target (which already folds explicit
        assembly hard caps / reserve floors). Without a policy, fall back to
        explicit assembly rails only — do not invent a hard trim budget from a
        bare half-cutover threshold alone, so pathological low cutover test
        fixtures and bypassed paths are not over-trimmed. Leaf-loop
        convergence still uses ``_post_compaction_target_tokens``.
        """
        if self._compaction_policy is not None and self.context_length > 0:
            policy_target = self._compaction_policy.post_compaction_target_tokens(
                self.context_length
            )
            if policy_target is not None:
                return int(policy_target)
        return self._effective_assembly_token_cap()

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Return the active assembly hard-cap rails only (not the policy target).

        Two knobs can constrain the assembled active context:
        - max_assembly_tokens: explicit hard cap
        - reserve_tokens_floor: keep headroom inside context_length

        These are safety rails independent of the typed post-compaction
        target. Ordinary assembly defaults to ``_assembly_token_budget``.
        """
        caps: list[int] = []

        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)

        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based assembly cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )

        if (
            self._compaction_policy is not None
            and self.context_length > 0
            and self._compaction_policy.output_reserve > 0
        ):
            output_reserve_cap = self.context_length - self._compaction_policy.output_reserve
            if output_reserve_cap > 0:
                caps.append(output_reserve_cap)
            else:
                logger.warning(
                    "LCM policy output_reserve=%d disables reserve-based assembly cap because context_length=%d",
                    self._compaction_policy.output_reserve,
                    self.context_length,
                )

        if not caps:
            return None

        return max(1, min(caps))

    # -- Async/background compaction ----------------------------------------

    def _async_policy_fingerprint(self) -> str:
        """Fingerprint live config fields that affect async prepare/promote.

        Always re-reads ``self._config`` (and runtime threshold attributes) so
        a live mutation after prepare — e.g. ``fresh_tail_count`` or
        ``context_threshold`` — is visible at promotion time and yields
        ``policy_fingerprint_mismatch``. Includes both configured and runtime
        threshold values plus chunking/tail inputs from the design doc.
        """
        runtime_threshold = float(
            getattr(self, "context_threshold", None)
            if getattr(self, "context_threshold", None) is not None
            else self._config.context_threshold
        )
        fields = {
            "protocol": "async_compaction_protocol_v1",
            "fresh_tail_count": int(self._config.fresh_tail_count),
            "fresh_tail_max_tokens": int(self._config.fresh_tail_max_tokens or 0),
            "leaf_chunk_tokens": int(self._config.leaf_chunk_tokens),
            "configured_context_threshold": float(self._config.context_threshold),
            "runtime_context_threshold": runtime_threshold,
            "dynamic_leaf_chunk_enabled": bool(self._config.dynamic_leaf_chunk_enabled),
            "dynamic_leaf_chunk_max": int(self._config.dynamic_leaf_chunk_max),
            "emergency_pressure_ratio": float(self._config.emergency_pressure_ratio),
            "max_assembly_tokens": int(self._config.max_assembly_tokens or 0),
            "reserve_tokens_floor": int(self._config.reserve_tokens_floor or 0),
            "condensation_fanin": int(self._config.condensation_fanin),
            "incremental_max_depth": int(self._config.incremental_max_depth),
            "cache_friendly_condensation_enabled": bool(
                self._config.cache_friendly_condensation_enabled
            ),
        }
        if self._compaction_policy is not None:
            fields["resolved_policy_fingerprint"] = self._compaction_policy.fingerprint
        raw = json.dumps(fields, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _async_route_fingerprint(self) -> str:
        summary_fingerprint = compute_route_fingerprint(
            self._config.summary_model,
            tuple(self._config.summary_fallback_models)
            if self._config.summary_fallback_models
            else (),
        )
        return hashlib.sha256(
            f"{self._main_route_fingerprint()}:{summary_fingerprint}".encode("utf-8")
        ).hexdigest()[:32]

    @staticmethod
    def _new_locked_publication_read_budget() -> dict[str, float | int]:
        return {
            "rows": 0,
            "bytes": 0,
            "files": 0,
            "max_rows": int(_PUBLICATION_LOCKED_MAX_ROWS),
            "max_bytes": int(_PUBLICATION_LOCKED_MAX_SERIALIZED_BYTES),
            "max_files": int(_PUBLICATION_LOCKED_MAX_ROWS),
            "deadline_at": time.monotonic()
            + max(0.0, float(_PUBLICATION_LOCKED_DEADLINE_SECONDS)),
        }

    @staticmethod
    def _charge_locked_publication_read(
        budget: dict[str, float | int],
        *,
        rows: int,
        serialized_bytes: int,
        label: str,
    ) -> None:
        if time.monotonic() >= float(budget["deadline_at"]):
            raise RuntimeError(f"{label} deadline exceeded")
        next_rows = int(budget["rows"]) + int(rows)
        if next_rows > int(budget["max_rows"]):
            raise RuntimeError(f"{label} row bound exceeded")
        next_bytes = int(budget["bytes"]) + max(0, int(serialized_bytes))
        if next_bytes > int(budget["max_bytes"]):
            raise RuntimeError(f"{label} serialized byte bound exceeded")
        budget["rows"] = next_rows
        budget["bytes"] = next_bytes

    @staticmethod
    def _canonical_message_source_ids_no_commit(
        conn: sqlite3.Connection,
        session_id: str,
        *,
        deadline_at: float,
        max_rows: int,
        max_edges: int,
        max_depth: int,
        max_bytes: int,
        read_budget: dict[str, float | int] | None = None,
    ) -> set[int]:
        """Return bounded canonical raw lineage visible on ``conn``'s transaction.

        ``length`` and bounded ``substr`` are deliberately evaluated by SQLite.
        Python therefore never receives an over-limit lineage blob merely to
        discover that it is unsafe to decode.
        """
        covered: set[int] = set()
        last_node_id = 0
        rows_seen = 0
        edges_seen = 0
        bytes_seen = 0
        while True:
            if time.monotonic() >= deadline_at:
                raise RuntimeError("canonical DAG lineage deadline exceeded")
            remaining = int(max_rows) - rows_seen
            lineage_bytes_remaining = int(max_bytes) - bytes_seen
            shared_bytes_remaining: int | None = None
            if read_budget is not None:
                shared_bytes_remaining = (
                    int(read_budget["max_bytes"]) - int(read_budget["bytes"])
                )
            if lineage_bytes_remaining < 0 or (
                shared_bytes_remaining is not None and shared_bytes_remaining < 0
            ):
                raise RuntimeError("canonical DAG lineage byte bound exceeded")
            local_byte_sized_rows = max(
                1,
                lineage_bytes_remaining // max(1, MAX_SOURCE_IDS_JSON_CHARS),
            )
            shared_byte_sized_rows = _CANONICAL_LINEAGE_QUERY_BATCH
            if shared_bytes_remaining is not None:
                shared_byte_sized_rows = max(
                    1,
                    shared_bytes_remaining // max(1, MAX_SOURCE_IDS_JSON_CHARS + 32),
                )
            page_limit = min(
                _CANONICAL_LINEAGE_QUERY_BATCH,
                remaining + 1,
                local_byte_sized_rows,
                shared_byte_sized_rows,
            )
            local_row_cap = max(
                0, lineage_bytes_remaining // max(1, page_limit)
            )
            shared_row_cap = MAX_SOURCE_IDS_JSON_CHARS
            if shared_bytes_remaining is not None:
                shared_row_cap = max(
                    0,
                    shared_bytes_remaining // max(1, page_limit) - 32,
                )
            row_blob_cap = min(
                MAX_SOURCE_IDS_JSON_CHARS,
                local_row_cap,
                shared_row_cap,
            )
            rows = conn.execute(
                """SELECT node_id, depth,
                          COALESCE(length(CAST(source_ids AS BLOB)), 0),
                          CASE
                            WHEN COALESCE(length(CAST(source_ids AS BLOB)), 0) <= ?
                            THEN substr(CAST(source_ids AS TEXT), 1, ?)
                            ELSE NULL
                          END
                   FROM summary_nodes
                   WHERE session_id = ? AND source_type = 'messages' AND node_id > ?
                   ORDER BY node_id LIMIT ?""",
                (
                    row_blob_cap,
                    row_blob_cap + 1,
                    session_id,
                    last_node_id,
                    page_limit,
                ),
            ).fetchall()
            if not rows:
                return covered
            if time.monotonic() >= deadline_at:
                raise RuntimeError("canonical DAG lineage deadline exceeded")
            for node_id, depth, encoded_bytes, raw_source_ids in rows:
                if time.monotonic() >= deadline_at:
                    raise RuntimeError("canonical DAG lineage deadline exceeded")
                if rows_seen >= int(max_rows):
                    raise RuntimeError("canonical DAG lineage row bound exceeded")
                if int(depth or 0) > int(max_depth):
                    raise RuntimeError("canonical DAG lineage depth bound exceeded")
                encoded_bytes = int(encoded_bytes or 0)
                if encoded_bytes > MAX_SOURCE_IDS_JSON_CHARS:
                    raise RuntimeError("canonical DAG lineage blob bound exceeded")
                if encoded_bytes > row_blob_cap or raw_source_ids is None:
                    raise RuntimeError("canonical DAG lineage byte bound exceeded")
                bytes_seen += encoded_bytes
                if bytes_seen > int(max_bytes):
                    raise RuntimeError("canonical DAG lineage byte bound exceeded")
                if read_budget is not None:
                    LCMEngine._charge_locked_publication_read(
                        read_budget,
                        rows=1,
                        serialized_bytes=encoded_bytes + 32,
                        label="canonical DAG lineage",
                    )
                try:
                    source_values = decode_source_ids(raw_source_ids or "[]")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError("canonical DAG contains unbounded source lineage") from exc
                edges_seen += len(source_values)
                if edges_seen > int(max_edges):
                    raise RuntimeError("canonical DAG lineage edge bound exceeded")
                covered.update(int(source_id) for source_id in source_values)
                rows_seen += 1
                last_node_id = int(node_id)
            if len(rows) < page_limit:
                return covered
        return covered

    @staticmethod
    def _exact_source_rows_exist_no_commit(
        conn: sqlite3.Connection,
        session_id: str,
        source_ids: Sequence[int],
        *,
        deadline_at: float | None = None,
        read_budget: dict[str, float | int] | None = None,
    ) -> bool:
        """Check exact, unique source membership on the locked transaction."""
        normalized = [int(source_id) for source_id in source_ids]
        if not normalized or len(set(normalized)) != len(normalized):
            return False
        found: set[int] = set()
        for offset in range(0, len(normalized), _PUBLICATION_LOCKED_QUERY_BATCH):
            if deadline_at is not None and time.monotonic() >= deadline_at:
                raise RuntimeError("locked exact source membership deadline exceeded")
            page_ids = normalized[offset:offset + _PUBLICATION_LOCKED_QUERY_BATCH]
            placeholders = ",".join("?" for _ in page_ids)
            rows = conn.execute(
                f"""SELECT store_id FROM messages
                    WHERE session_id = ? AND store_id IN ({placeholders})
                    ORDER BY store_id LIMIT ?""",
                (session_id, *page_ids, len(page_ids) + 1),
            ).fetchall()
            if read_budget is not None:
                for _row in rows:
                    LCMEngine._charge_locked_publication_read(
                        read_budget,
                        rows=1,
                        serialized_bytes=32,
                        label="locked exact source membership",
                    )
            found.update(int(row[0]) for row in rows)
        return found == set(normalized)

    @staticmethod
    def _canonical_summarizer_message_identity(
        *,
        role: Any,
        content: Any,
        tool_call_id: Any,
        tool_calls: Any,
    ) -> tuple[str, str, str, str]:
        """Canonicalize all message fields consumed by ``_serialize_messages``."""
        normalized_tool_calls = tool_calls
        if isinstance(normalized_tool_calls, str):
            try:
                normalized_tool_calls = json.loads(normalized_tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if normalized_tool_calls in (None, [], {}):
            tool_calls_text = ""
        elif isinstance(normalized_tool_calls, str):
            tool_calls_text = normalized_tool_calls
        else:
            tool_calls_text = json.dumps(
                normalized_tool_calls,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        return (
            str(role or "unknown"),
            normalize_content_value(content) or "",
            str(tool_call_id or ""),
            tool_calls_text,
        )

    def _foreground_source_identity_for_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        source_ids: Sequence[int],
    ) -> str:
        """Capture DB identity only while it still matches summarized messages."""
        wanted = [int(value) for value in source_ids]
        if len(messages) != len(wanted) or len(set(wanted)) != len(wanted):
            return ""
        # Callers supply the exact summarized IDs in the same stable order as
        # the summarized messages. Re-scanning the session here would allocate
        # complete payload rows before the bounded identity reader runs.
        message_by_store_id = dict(zip(wanted, messages))
        if set(message_by_store_id) != set(wanted):
            return ""
        mismatch = False
        validated_source_ids: set[int] = set()

        def validate_stored_row(source_id: int, stored: dict[str, Any]) -> None:
            nonlocal mismatch
            validated_source_ids.add(source_id)
            active = message_by_store_id[source_id]
            if str(stored.get("session_id") or "") != self._session_id:
                mismatch = True
                return
            active_semantics = self._canonical_summarizer_message_identity(
                role=active.get("role"),
                content=active.get("content"),
                tool_call_id=active.get("tool_call_id"),
                tool_calls=active.get("tool_calls"),
            )
            stored_semantics = self._canonical_summarizer_message_identity(
                role=stored.get("role"),
                content=stored.get("content"),
                tool_call_id=stored.get("tool_call_id"),
                tool_calls=stored.get("tool_calls"),
            )
            if active_semantics == stored_semantics:
                return
            # Externalization/cleanup can intentionally make the durable form
            # differ while retaining an equivalent replay identity.
            active_identity = self._message_replay_identity(active)
            stored_identity = self._message_replay_identity(stored, stored_row=True)
            cleanup_identity = self._active_cleanup_replay_identity(active_identity)
            if stored_identity != active_identity and (
                cleanup_identity is None
                or self._active_cleanup_replay_identity(stored_identity) != cleanup_identity
            ):
                mismatch = True

        # The hash and active-message comparison consume the same bounded,
        # incremental SQL rows.  No unrestricted MessageStore.get_batch()
        # allocation occurs, and there is no second optimistic read that could
        # silently replace the expected identity after validation.
        identity = compute_source_identity_hash(
            self._store._conn,
            self._session_id,
            wanted,
            read_budget=self._new_locked_publication_read_budget(),
            digest_chars=None,
            role_default="unknown",
            row_validator=validate_stored_row,
        )
        return "" if mismatch or validated_source_ids != set(wanted) else identity

    def _publish_foreground_leaf(
        self,
        *,
        node: Optional[SummaryNode],
        source_end_store_id: int,
        covered_source_ids: Sequence[int],
        expected_source_identity_hash: str | None = None,
    ) -> dict[str, Any]:
        """Publish a foreground leaf, frontier, and lifecycle in one transaction.

        Async and foreground publishers contend on the same ``BEGIN IMMEDIATE``
        writer lock. The first canonical coverage wins; the loser returns the
        authoritative generation without inserting a DAG/FTS row.
        """
        conv_id = self._conversation_id
        session_id = self._session_id
        source_ids = [int(value) for value in covered_source_ids]
        source_end = int(source_end_store_id or 0)
        if not session_id:
            raise RuntimeError("foreground publication requires a bound session")
        if not source_ids or source_end <= 0:
            raise RuntimeError("foreground publication requires non-empty raw lineage")

        def _foreground_boundary(phase: str) -> None:
            crash_hook = getattr(self, "_foreground_publish_crash_hook", None)
            if callable(crash_hook):
                crash_hook(phase)
            elif crash_hook == phase:
                os._exit(89)  # noqa: PLW1510 - deliberate subprocess crash injection
            failure_hook = getattr(self, "_foreground_publish_failure_hook", None)
            if callable(failure_hook):
                failure_hook(phase)
            elif failure_hook == phase:
                raise RuntimeError("injected foreground publication failure")

        # Legacy embedders can set only ``_session_id`` and never bind lifecycle
        # conversation state. Async preparation is impossible in that mode, but
        # retain foreground DAG behavior with one writer transaction and the same
        # non-empty/exact-source guards.
        if not conv_id:
            with self._dag._db_lock:
                conn = self._dag._conn
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    read_budget = self._new_locked_publication_read_budget()
                    canonical_coverage = self._canonical_message_source_ids_no_commit(
                        conn,
                        session_id,
                        deadline_at=float(read_budget["deadline_at"]),
                        max_rows=_CANONICAL_LINEAGE_MAX_ROWS,
                        max_edges=_CANONICAL_LINEAGE_MAX_EDGES,
                        max_depth=_CANONICAL_LINEAGE_MAX_DEPTH,
                        max_bytes=_CANONICAL_LINEAGE_MAX_BYTES,
                        read_budget=read_budget,
                    )
                    if canonical_coverage.intersection(source_ids):
                        raise RuntimeError("foreground canonical source overlap")
                    if not self._exact_source_rows_exist_no_commit(
                        conn,
                        session_id,
                        source_ids,
                        deadline_at=float(read_budget["deadline_at"]),
                        read_budget=read_budget,
                    ):
                        raise RuntimeError(
                            "foreground source identity changed before publication"
                        )
                    locked_identity = compute_source_identity_hash(
                        conn,
                        session_id,
                        source_ids,
                        read_budget=read_budget,
                        digest_chars=None,
                        role_default="unknown",
                    )
                    if (
                        expected_source_identity_hash is not None
                        and locked_identity != expected_source_identity_hash
                    ):
                        raise RuntimeError(
                            "foreground source identity changed before publication"
                        )
                    node_id = 0
                    if node is not None:
                        bounds = conn.execute(
                            f"""
                            SELECT MIN(timestamp), MAX(timestamp) FROM messages
                            WHERE session_id = ? AND store_id IN (
                                {','.join('?' for _ in source_ids)}
                            )
                            """,
                            (session_id, *source_ids),
                        ).fetchone()
                        node.earliest_at = bounds[0] if bounds else None
                        node.latest_at = bounds[1] if bounds else None
                        node_id = self._dag.add_node_no_commit(conn, node)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return {
                "published": True,
                "reason": "",
                "node_id": int(node_id),
                "generation": -1,
                "source_end_store_id": source_end,
            }

        policy_fp = self._async_policy_fingerprint()
        route_fp = self._async_route_fingerprint()
        with self._frontier.publication_transaction() as publication_conn:
            _foreground_boundary("after_begin")
            read_budget = self._new_locked_publication_read_budget()
            frontier_row = publication_conn.execute(
                """
                SELECT generation, session_id, source_end_store_id
                FROM lcm_active_frontiers
                WHERE conversation_id = ?
                ORDER BY generation DESC LIMIT 1
                """,
                (conv_id,),
            ).fetchone()
            base_generation = int(frontier_row[0]) if frontier_row else 0
            active_end = int(frontier_row[2] or 0) if frontier_row else 0
            if frontier_row is not None and str(frontier_row[1] or "") != session_id:
                raise RuntimeError("foreground frontier session changed")

            canonical_coverage = self._canonical_message_source_ids_no_commit(
                publication_conn,
                session_id,
                deadline_at=float(read_budget["deadline_at"]),
                max_rows=_CANONICAL_LINEAGE_MAX_ROWS,
                max_edges=_CANONICAL_LINEAGE_MAX_EDGES,
                max_depth=_CANONICAL_LINEAGE_MAX_DEPTH,
                max_bytes=_CANONICAL_LINEAGE_MAX_BYTES,
                read_budget=read_budget,
            )
            if canonical_coverage.intersection(source_ids):
                return {
                    "published": False,
                    "reason": "canonical_source_overlap",
                    "node_id": 0,
                    "generation": base_generation,
                    "source_end_store_id": active_end,
                }
            if not self._exact_source_rows_exist_no_commit(
                publication_conn,
                session_id,
                source_ids,
                deadline_at=float(read_budget["deadline_at"]),
                read_budget=read_budget,
            ):
                raise RuntimeError("foreground source identity changed before publication")

            locked_identity = compute_source_identity_hash(
                publication_conn,
                session_id,
                source_ids,
                read_budget=read_budget,
                digest_chars=None,
                role_default="unknown",
            )
            if (
                expected_source_identity_hash is not None
                and locked_identity != expected_source_identity_hash
            ):
                return {
                    "published": False,
                    "reason": "source_identity_mismatch",
                    "node_id": 0,
                    "generation": base_generation,
                    "source_end_store_id": active_end,
                }

            node_id = 0
            if node is not None:
                bounds = publication_conn.execute(
                    f"""
                    SELECT MIN(timestamp), MAX(timestamp) FROM messages
                    WHERE session_id = ? AND store_id IN ({','.join('?' for _ in source_ids)})
                    """,
                    (session_id, *source_ids),
                ).fetchone()
                node.earliest_at = bounds[0] if bounds else None
                node.latest_at = bounds[1] if bounds else None
                node_id = self._dag.add_node_no_commit(publication_conn, node)
            _foreground_boundary("after_canonical_insert")
            items = self._build_promoted_frontier_items_no_commit(
                publication_conn,
                conversation_id=conv_id,
                session_id=session_id,
                node_id=int(node_id),
                covered_source_ids=source_ids,
                frontier_end_store_id=source_end,
                base_generation=base_generation,
                read_budget=read_budget,
            )
            if not items:
                raise RuntimeError("foreground frontier items are empty")
            new_gen = self._frontier.advance_frontier_generation_with_items(
                conv_id,
                session_id,
                source_end,
                policy_fp,
                route_fp,
                base_generation,
                items,
            )
            if not new_gen:
                raise RuntimeError("foreground frontier changed inside publication")
            _foreground_boundary("after_frontier")
            self._frontier.supersede_competing_batches_no_commit(
                publication_conn,
                conv_id,
                base_generation,
                reason="foreground_generation_published",
            )
            _foreground_boundary("after_batches_superseded")
            with self._lifecycle.publication_connection(publication_conn):
                try:
                    lifecycle_state = self._lifecycle.advance_frontier(
                        conv_id, session_id, source_end
                    )
                except Exception:
                    # A wrapper may raise after applying the no-commit SQL. Verify
                    # on the same transaction: if the marker is present, the
                    # exception is observational and the whole publication can
                    # still commit atomically; otherwise roll everything back.
                    acknowledged = publication_conn.execute(
                        """
                        SELECT current_session_id, current_frontier_store_id
                        FROM lcm_lifecycle_state WHERE conversation_id = ?
                        """,
                        (conv_id,),
                    ).fetchone()
                    if (
                        acknowledged is None
                        or str(acknowledged[0] or "") != session_id
                        or int(acknowledged[1] or 0) < source_end
                    ):
                        raise
                    lifecycle_state = True
            if (
                lifecycle_state is None
                or (
                    lifecycle_state is not True
                    and (
                        lifecycle_state.current_session_id != session_id
                        or int(lifecycle_state.current_frontier_store_id or 0)
                        < source_end
                    )
                )
            ):
                raise RuntimeError("foreground lifecycle frontier not advanced")
            _foreground_boundary("after_lifecycle")
        _foreground_boundary("after_commit")
        return {
            "published": True,
            "reason": "",
            "node_id": int(node_id),
            "generation": int(new_gen),
            "source_end_store_id": source_end,
        }

    def get_async_compaction_status(self) -> dict[str, Any]:
        """Return counts of prepared batches by state for the active conversation."""
        now = time.time()
        cooldown_until = float(getattr(self, "_async_worker_cooldown_until", 0.0) or 0.0)
        in_cooldown = now < cooldown_until
        cooldown_remaining = max(0.0, cooldown_until - now) if in_cooldown else 0.0
        telemetry = {
            "worker_last_tick_at": getattr(self, "_async_worker_last_tick_at", None),
            "worker_last_tick_duration_ms": getattr(
                self, "_async_worker_last_tick_duration_ms", None
            ),
            "worker_consecutive_failures": int(
                getattr(self, "_async_worker_consecutive_failures", 0) or 0
            ),
            "worker_in_cooldown": in_cooldown,
            "worker_cooldown_remaining_seconds": cooldown_remaining,
            "last_prepare_at": getattr(self, "_async_last_prepare_at", None),
            "last_promote_at": getattr(self, "_async_last_promote_at", None),
            "last_promote_validation_ms": getattr(
                self, "_async_last_promote_validation_ms", None
            ),
            "last_promote_publication_ms": getattr(
                self, "_async_last_promote_publication_ms", None
            ),
            "last_promote_wall_ms": getattr(self, "_async_last_promote_wall_ms", None),
            "total_prepare_attempts": int(
                getattr(self, "_async_total_prepare_attempts", 0) or 0
            ),
            "total_promote_attempts": int(
                getattr(self, "_async_total_promote_attempts", 0) or 0
            ),
            "total_promote_succeeded": int(
                getattr(self, "_async_total_promote_succeeded", 0) or 0
            ),
            "total_prepared": int(getattr(self, "_async_total_prepared", 0) or 0),
            "prepare_skip_reasons": dict(
                getattr(self, "_async_prepare_skip_reasons", {}) or {}
            ),
            "cleanup_counts": dict(getattr(self, "_async_cleanup_counts", {}) or {}),
            "last_pressure_signal": getattr(self, "_async_last_pressure_signal", "none"),
            "last_host_prompt_tokens": int(
                getattr(self, "_async_last_host_prompt_tokens", 0) or 0
            ),
            "last_source_tokens": int(
                getattr(self, "_async_last_source_tokens", 0) or 0
            ),
            "last_pressure_mismatch_tokens": int(
                getattr(self, "_async_last_pressure_mismatch_tokens", 0) or 0
            ),
            "last_expected_reduction_tokens": int(
                getattr(self, "_async_last_expected_reduction_tokens", 0) or 0
            ),
            "last_ready_coverage_tokens": int(
                getattr(self, "_async_last_ready_coverage_tokens", 0) or 0
            ),
            "last_projected_headroom_tokens": int(
                getattr(self, "_async_last_projected_headroom_tokens", 0) or 0
            ),
            "profile_active_summary_calls": active_profile_admissions(self._store.db_path),
            "foreground_compress_active": bool(
                getattr(self, "_foreground_compress_active", None)
                and self._foreground_compress_active.is_set()
            ),
        }
        conv_id = self.current_conversation_id
        if not conv_id:
            return {
                "enabled": False,
                "preparing_batches": 0,
                "pending_batches": 0,
                "prepared_batches": 0,
                "promoted_batches": 0,
                "rejected_batches": 0,
                "failed_batches": 0,
                "superseded_batches": 0,
                **telemetry,
            }
        counts = self._frontier.get_batch_counts_by_state(conv_id)
        return {
            "enabled": getattr(self._config, "async_background_compaction_enabled", False),
            "preparing_batches": counts.get("preparing", 0),
            "pending_batches": counts.get("ready", 0),
            "prepared_batches": counts.get("ready", 0),
            "promoted_batches": counts.get("promoted", 0),
            "rejected_batches": counts.get("rejected", 0),
            "failed_batches": counts.get("failed", 0),
            "superseded_batches": counts.get("superseded", 0),
            **telemetry,
        }

    def _record_async_prepare_skip(self, reason: str) -> None:
        counters = getattr(self, "_async_prepare_skip_reasons", None)
        if counters is None:
            counters = {}
            self._async_prepare_skip_reasons = counters
        counters[reason] = int(counters.get(reason, 0)) + 1

    def _async_preparation_threshold_tokens(self) -> int:
        policy = getattr(self, "_compaction_policy", None)
        if policy is not None and self.context_length > 0:
            return int(policy.preparation_tokens(self.context_length))
        if self.context_length > 0:
            return max(
                1,
                int(
                    self.context_length
                    * float(self._config.context_threshold)
                    * DEFAULT_PREPARATION_RATIO
                ),
            )
        return max(1, int(self.threshold_tokens * DEFAULT_PREPARATION_RATIO))

    def prepare_background_compaction_once(
        self,
        messages: List[Dict[str, Any]],
        *,
        leave_state: str = "ready",
        host_prompt_tokens: int | None = None,
        force: bool = False,
    ) -> PreparedBatch | None:
        """Build a prepared compaction batch off-context.

        Snapshots the current source boundary, builds leaf summaries
        privately (without inserting into the canonical DAG), and stores
        a batch in the ``lcm_prepared_batches`` table.

        Returns the ``PreparedBatch`` or ``None`` if disabled or no work.
        """
        if not getattr(self._config, "async_background_compaction_enabled", False):
            return None

        session_id = self.current_session_id
        conv_id = self.current_conversation_id
        if not session_id or not conv_id:
            return None

        self._async_total_prepare_attempts = (
            int(getattr(self, "_async_total_prepare_attempts", 0) or 0) + 1
        )

        # Preparation can be invoked before a compatible host has supplied
        # model metadata. Resolve the global/default policy against the known
        # context/session so the batch still persists a complete policy, not
        # only an opaque fingerprint.
        if self._compaction_policy is None:
            self._resolve_live_compaction_policy()

        policy_fp = self._async_policy_fingerprint()
        route_fp = self._async_route_fingerprint()
        utility_policy_enabled = bool(
            getattr(self._config, "async_preparation_utility_policy_enabled", False)
        )

        # Get the current frontier
        frontier = self._frontier.get_active_frontier(conv_id)
        if frontier is None:
            self._frontier.ensure_frontier(
                conv_id, session_id,
                policy_fingerprint=policy_fp,
                route_fingerprint=route_fp,
            )
            frontier = self._frontier.get_active_frontier(conv_id)

        base_generation = frontier["generation"] if frontier else 1
        source_end = frontier["source_end_store_id"] if frontier else 0

        if utility_policy_enabled:
            cleanup_counts = self._frontier.cleanup_pending_batches(
                conversation_id=conv_id,
                current_generation=int(base_generation),
                policy_fingerprint=policy_fp,
                route_fingerprint=route_fp,
                ttl_seconds=float(self._config.async_ready_ttl_seconds),
            )
            for cleanup_reason, cleanup_count in cleanup_counts.items():
                if cleanup_count:
                    self._async_cleanup_counts[cleanup_reason] = (
                        int(self._async_cleanup_counts.get(cleanup_reason, 0))
                        + int(cleanup_count)
                    )

        # Get the raw messages that are candidates for compaction.
        all_stored = self._store.get_session_messages(session_id)
        if not all_stored:
            return None

        # Determine which messages are candidates (exclude fresh tail)
        candidate_count = self._fresh_tail_start(all_stored)
        if candidate_count <= 0:
            return None

        candidate_start = self._leading_anchor_count(all_stored)
        candidate_stored = [
            row
            for row in all_stored[candidate_start:candidate_count]
            if int(row.get("store_id") or 0) > int(source_end or 0)
        ]
        if not candidate_stored:
            if utility_policy_enabled:
                self._record_async_prepare_skip("candidate-already-covers-range")
            return None
        candidate_store_ids = [m["store_id"] for m in candidate_stored]
        actual_source_end = candidate_store_ids[-1] if candidate_store_ids else source_end

        source_tokens_estimate = sum(
            max(0, int(row.get("token_estimate") or 0)) for row in candidate_stored
        )
        if source_tokens_estimate <= 0:
            source_tokens_estimate = count_messages_tokens(
                [
                    {"role": row.get("role"), "content": row.get("content") or ""}
                    for row in candidate_stored
                ]
            )

        if utility_policy_enabled:
            observed_host_tokens = int(
                host_prompt_tokens
                if host_prompt_tokens is not None
                else (self.last_prompt_tokens or 0)
            )
            if observed_host_tokens > 0:
                pressure_tokens = observed_host_tokens
                pressure_signal = "host"
            else:
                pressure_tokens = source_tokens_estimate
                pressure_signal = "source-estimate"
            self._async_last_pressure_signal = pressure_signal
            self._async_last_host_prompt_tokens = max(0, observed_host_tokens)
            self._async_last_source_tokens = max(0, source_tokens_estimate)
            self._async_last_pressure_mismatch_tokens = (
                abs(observed_host_tokens - source_tokens_estimate)
                if observed_host_tokens > 0
                else 0
            )
            expected_summary_tokens = max(1, int(source_tokens_estimate * 0.20))
            expected_reduction = max(0, source_tokens_estimate - expected_summary_tokens)
            self._async_last_expected_reduction_tokens = expected_reduction
            self._async_last_projected_headroom_tokens = max(
                0,
                int(self.context_length or 0)
                - max(0, pressure_tokens - expected_reduction),
            )
            if not force and pressure_tokens < self._async_preparation_threshold_tokens():
                self._record_async_prepare_skip("below-preparation-pressure")
                return None
            if (
                not force
                and expected_reduction
                < int(self._config.async_preparation_min_reduction_tokens)
            ):
                self._record_async_prepare_skip("insufficient-reduction")
                return None

            pending = [
                batch
                for batch in self._frontier.list_pending_batches(conv_id)
                if int(batch.base_generation) == int(base_generation)
                and batch.policy_fingerprint == policy_fp
                and batch.route_fingerprint == route_fp
            ]
            if pending:
                covered_end = max(int(batch.source_end_store_id) for batch in pending)
                self._async_last_ready_coverage_tokens = sum(
                    max(0, int(row.get("token_estimate") or 0))
                    for row in candidate_stored
                    if int(row.get("store_id") or 0) <= covered_end
                )
                refresh_tokens = sum(
                    max(0, int(row.get("token_estimate") or 0))
                    for row in candidate_stored
                    if int(row.get("store_id") or 0) > covered_end
                )
                if covered_end >= int(actual_source_end) or (
                    not force
                    and refresh_tokens < int(self._config.async_candidate_refresh_min_tokens)
                ):
                    self._record_async_prepare_skip("candidate-already-covers-range")
                    return None
            else:
                self._async_last_ready_coverage_tokens = 0

            model_chain = [
                self._config.summary_model,
                *list(self._config.summary_fallback_models or []),
            ]
            if not model_chain:
                model_chain = [""]
            route_available = any(
                self._summary_circuit_breaker.allows(model) for model in model_chain
            )
            if not self._summary_spend_guard.allows() or not route_available:
                self._record_async_prepare_skip("spend-backoff")
                return None

        # Compute source identity hash for CAS validation
        source_hash = compute_source_identity_hash(
            self._store._conn, session_id, candidate_store_ids,
            read_budget=self._new_locked_publication_read_budget(),
        )

        admission_acquired = False
        resolved_policy_json = json.dumps(
            self._compaction_policy.to_status_dict(self.context_length)
            if self._compaction_policy is not None
            else {
                "fingerprint": policy_fp,
                "selection_reason": "runtime metadata unavailable at prepare",
            },
            sort_keys=True,
        )
        if utility_policy_enabled:
            admission_acquired = try_acquire_profile_admission(
                self._store.db_path,
                int(self._config.async_summary_admission_limit),
            )
            if not admission_acquired:
                self._record_async_prepare_skip("admission-limited")
                return None
            batch_creation_completed = False
            try:
                batch_id, capacity_reason = self._frontier.create_batch_cas(
                    conversation_id=conv_id,
                    session_id=session_id,
                    base_generation=base_generation,
                    source_end_store_id=actual_source_end,
                    source_identity_hash=source_hash,
                    source_ids=candidate_store_ids,
                    policy_fingerprint=policy_fp,
                    route_fingerprint=route_fp,
                    max_conversation_candidates=int(
                        self._config.async_max_candidates_per_conversation
                    ),
                    max_profile_candidates=int(self._config.async_max_candidates_per_profile),
                    resolved_policy_json=resolved_policy_json,
                )
                batch_creation_completed = True
            finally:
                if admission_acquired and not batch_creation_completed:
                    release_profile_admission(self._store.db_path)
                    admission_acquired = False
            if not batch_id:
                release_profile_admission(self._store.db_path)
                admission_acquired = False
                self._record_async_prepare_skip(capacity_reason or "admission-limited")
                return None
        else:
            batch_id, capacity_reason = self._frontier.create_batch_cas(
                conversation_id=conv_id,
                session_id=session_id,
                base_generation=base_generation,
                source_end_store_id=actual_source_end,
                source_identity_hash=source_hash,
                source_ids=candidate_store_ids,
                policy_fingerprint=policy_fp,
                route_fingerprint=route_fp,
                resolved_policy_json=resolved_policy_json,
            )
            if not batch_id:
                self._record_async_prepare_skip(
                    capacity_reason or "generation-superseded"
                )
                return None

        # Build leaf summaries using the existing _summarize_leaf_chunk_with_rescue.
        # Summaries stay private to this path — they are NOT inserted into the
        # canonical DAG (or FTS). Promotion is the only path that publishes.
        # The summary payload IS persisted on the batch so promote is zero-LLM.
        try:
            # Bail out early if shutdown was signaled while we were setting up.
            stop_event = self._async_worker_stop
            if stop_event is not None and stop_event.is_set():
                self._frontier.settle_batch_if_preparing(
                    batch_id, "failed", failure_reason="shutdown_signaled",
                )
                return None
            # Foreground cutover has priority: do not start expensive LLM work
            # once compress() has claimed the critical section.
            if self._foreground_compress_active.is_set():
                self._frontier.settle_batch_if_preparing(
                    batch_id, "failed", failure_reason="foreground_compress_priority",
                )
                return None

            # Convert stored messages to OpenAI format for the summarizer
            candidate_messages = []
            for stored_message in candidate_stored:
                candidate_message = {
                    "role": stored_message["role"],
                    "content": stored_message.get("content") or "",
                }
                if stored_message.get("tool_call_id"):
                    candidate_message["tool_call_id"] = stored_message["tool_call_id"]
                if stored_message.get("tool_calls"):
                    candidate_message["tool_calls"] = stored_message["tool_calls"]
                candidate_messages.append(candidate_message)

            # Use the engine's existing leaf summarization path
            compacted_chunk, source_tokens, summary_text, level, attempts = (
                self._summarize_leaf_chunk_with_rescue(candidate_messages)
            )

            # Adaptive rescue may summarize only an oldest prefix. Publish
            # lineage for exactly that returned prefix; the unsummarized suffix
            # remains raw and eligible for a later batch.
            compacted_count = len(compacted_chunk)
            if compacted_count <= 0 or compacted_count > len(candidate_messages):
                raise RuntimeError("prepared summary returned invalid source coverage")
            for expected, actual in zip(
                candidate_messages[:compacted_count], compacted_chunk
            ):
                if actual is not expected and actual != expected:
                    raise RuntimeError(
                        "prepared summary rescue returned non-prefix source coverage"
                    )
            summarized_stored = candidate_stored[:compacted_count]
            summarized_store_ids = [
                int(row["store_id"]) for row in summarized_stored
            ]
            summarized_source_end = summarized_store_ids[-1]
            summarized_source_hash = compute_source_identity_hash(
                self._store._conn,
                session_id,
                summarized_store_ids,
                read_budget=self._new_locked_publication_read_budget(),
            )

            # Check stop event again after the LLM call — if shutdown
            # was signaled during the API call, abandon the batch.
            if stop_event is not None and stop_event.is_set():
                self._frontier.settle_batch_if_preparing(
                    batch_id, "failed", failure_reason="shutdown_signaled",
                )
                return None
            if self._foreground_compress_active.is_set():
                self._frontier.settle_batch_if_preparing(
                    batch_id, "failed", failure_reason="foreground_compress_priority",
                )
                return None

            leaf_count = 1  # One leaf from one chunk
            leaf_tokens = count_tokens(summary_text)
            # Persist the full publishable payload so promote never re-runs
            # the summarizer (issue #1). payload_version=2 marks a complete
            # private leaf ready for CAS publish.
            summary_payload = json.dumps(
                {
                    "summary_text": summary_text,
                    "source_tokens": int(source_tokens),
                    "token_count": int(leaf_tokens),
                    "level": int(level),
                    "attempts": int(attempts),
                    "source_ids": list(summarized_store_ids),
                    "expand_hint": self._extract_expand_hint(summary_text),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

            finalized, _finalize_reason = self._frontier.finalize_batch_cas(
                batch_id,
                base_generation=int(base_generation),
                state=leave_state,
                expected_leaf_count=leaf_count,
                frontier_end_store_id=summarized_source_end,
                summary_payload=summary_payload,
                payload_version=PREPARED_PAYLOAD_VERSION,
                source_end_store_id=summarized_source_end,
                source_identity_hash=summarized_source_hash,
                source_ids=summarized_store_ids,
            )
            if finalized:
                self._async_last_prepare_at = time.time()
            if finalized and leave_state == "ready":
                self._async_total_prepared = (
                    int(getattr(self, "_async_total_prepared", 0) or 0) + 1
                )
        except Exception as exc:
            logger.warning("LCM background compaction prep failed: %s", exc)
            self._frontier.settle_batch_if_preparing(
                batch_id, "failed", failure_reason=str(exc),
            )
        finally:
            if admission_acquired:
                release_profile_admission(self._store.db_path)

        return self._frontier.get_batch(batch_id)

    def promote_prepared_compaction(
        self,
        batch_id: int,
        messages: List[Dict[str, Any]],
    ) -> PromotionResult:
        """Atomically promote a prepared batch to the canonical DAG.

        CAS validation:
        1. Batch must carry a persisted summary payload (payload_version >= 2)
        2. Source identity hash must match (no raw messages changed)
        3. Policy fingerprint must match (no config change)
        4. Route fingerprint must match (no summary model change)
        5. Base generation must match current frontier (no concurrent promotion)
        6. No canonical DAG node already covers the same source IDs

        On success: insert canonical DAG node from the *persisted* payload
        (ZERO LLM calls), write ordered frontier items, advance frontier
        generation, mark batch as promoted.  On failure: mark batch as
        rejected (except injected mid-publish failures, which roll back and
        re-raise).
        """
        wall_started = time.perf_counter()
        validation_ms = 0.0
        publication_ms = 0.0
        self._async_promotion_ingested_messages = None
        self._async_total_promote_attempts = (
            int(getattr(self, "_async_total_promote_attempts", 0) or 0) + 1
        )

        def _result(
            *,
            promoted: bool,
            reason: str = "",
            node_id: int = 0,
            covered: list[int] | None = None,
        ) -> PromotionResult:
            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            return PromotionResult(
                promoted=promoted,
                reason=reason,
                batch_id=batch_id,
                node_id=node_id,
                covered_source_ids=list(covered or []),
                validation_ms=validation_ms,
                publication_ms=publication_ms,
                wall_ms=wall_ms,
            )

        validation_started = time.perf_counter()
        batch = self._frontier.get_batch(batch_id)
        if batch is None:
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="batch_not_found")

        if batch.state not in ("ready", "preparing"):
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason=f"batch_state_{batch.state}")

        # This is the authoritative promotion boundary. A worker can make a
        # batch ready after compress()'s optimistic lookup, and direct callers
        # can bypass that lookup entirely. Persist the complete current host
        # view before any validation path is allowed to publish the batch.
        promotion_messages = self._ingest_messages(messages)
        self._async_promotion_ingested_messages = promotion_messages

        # 0. Persisted payload required — never re-summarize at promote.
        payload = batch.parsed_summary_payload()
        if payload is None:
            self._frontier.update_batch_state(
                batch_id,
                "superseded",
                failure_reason="legacy_v1_batch_without_payload",
            )
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="legacy_v1_batch_without_payload")

        summary_text = str(payload.get("summary_text") or "")
        source_tokens = int(payload.get("source_tokens") or 0)
        leaf_tokens = int(
            payload.get("token_count")
            or count_tokens(summary_text)
        )
        expand_hint = str(
            payload.get("expand_hint") or self._extract_expand_hint(summary_text)
        )
        covered_source_ids = [int(s) for s in (batch.source_ids or [])]

        # 1. Source identity check
        current_source_hash = compute_source_identity_hash(
            self._store._conn, batch.session_id, batch.source_ids,
            read_budget=self._new_locked_publication_read_budget(),
        )
        if current_source_hash != batch.source_identity_hash:
            self._frontier.update_batch_state(batch_id, "rejected", failure_reason="source_identity_mismatch")
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="source_identity_mismatch")

        # 2. Policy fingerprint check (always recompute from live config).
        # A freshly restarted engine may not yet have seen host route metadata;
        # resolve the deterministic global/default policy before comparing so
        # the same complete policy document prepared pre-restart remains valid.
        if self._compaction_policy is None:
            self._resolve_live_compaction_policy()
        current_policy_fp = self._async_policy_fingerprint()
        if batch.policy_fingerprint != current_policy_fp:
            self._frontier.update_batch_state(batch_id, "rejected", failure_reason="policy_fingerprint_mismatch")
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="policy_fingerprint_mismatch")

        # 3. Route fingerprint check
        current_route_fp = self._async_route_fingerprint()
        if batch.route_fingerprint and current_route_fp and batch.route_fingerprint != current_route_fp:
            self._frontier.update_batch_state(batch_id, "rejected", failure_reason="summary_route_fingerprint_mismatch")
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="summary_route_fingerprint_mismatch")

        # 4. CAS on base generation
        frontier = self._frontier.get_active_frontier(batch.conversation_id)
        current_gen = frontier["generation"] if frontier else 0
        if current_gen != batch.base_generation:
            self._frontier.update_batch_state(batch_id, "rejected", failure_reason="frontier_mismatch")
            validation_ms = (time.perf_counter() - validation_started) * 1000.0
            return _result(promoted=False, reason="frontier_mismatch")

        validation_ms = (time.perf_counter() - validation_started) * 1000.0

        # --- One-connection atomic publication ---
        # Every table below lives in the configured LCM database.  The frontier
        # connection coordinates one BEGIN IMMEDIATE transaction so process
        # death can expose only the old state or the complete new state.  The
        # generation CAS and canonical overlap check are repeated under that
        # write lock; preparation/validation above remains optimistic.
        inserted_node_id = 0

        def _publication_boundary(phase: str) -> None:
            crash_hook = getattr(
                self, "_async_compaction_publish_crash_hook", None
            )
            if callable(crash_hook):
                crash_hook(phase)
            elif crash_hook == phase:
                os._exit(86)  # noqa: PLW1510 - deliberate subprocess crash injection

            failure_hook = getattr(
                self, "_async_compaction_publish_failure_hook", None
            )
            if callable(failure_hook):
                failure_hook(phase)
            elif failure_hook == phase:
                raise RuntimeError("injected async promotion failure")

        try:
            publication_started = time.perf_counter()
            with self._frontier.publication_transaction() as publication_conn:
                _publication_boundary("after_begin")
                read_budget = self._new_locked_publication_read_budget()

                batch_row = publication_conn.execute(
                    "SELECT state FROM lcm_prepared_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if batch_row is None:
                    return _result(promoted=False, reason="batch_not_found")
                if str(batch_row[0]) not in ("ready", "preparing"):
                    return _result(
                        promoted=False,
                        reason=f"batch_state_{batch_row[0]}",
                    )

                generation_row = publication_conn.execute(
                    """
                    SELECT generation FROM lcm_active_frontiers
                    WHERE conversation_id = ?
                    ORDER BY generation DESC LIMIT 1
                    """,
                    (batch.conversation_id,),
                ).fetchone()
                locked_generation = int(generation_row[0]) if generation_row else 0
                if locked_generation != int(batch.base_generation):
                    self._frontier.update_batch_state_no_commit(
                        publication_conn,
                        batch_id,
                        "rejected",
                        failure_reason="frontier_mismatch",
                    )
                    return _result(promoted=False, reason="frontier_mismatch")

                # Optimistic validation can race cleanup, reassignment, or a
                # narrow permitted source rewrite. Recheck exact membership and
                # content identity after BEGIN IMMEDIATE, immediately before the
                # first canonical insert; the writer lock closes the TOCTOU gap.
                exact_sources_exist = self._exact_source_rows_exist_no_commit(
                    publication_conn,
                    batch.session_id,
                    covered_source_ids,
                    deadline_at=float(read_budget["deadline_at"]),
                    read_budget=read_budget,
                )
                locked_source_hash = compute_source_identity_hash(
                    publication_conn,
                    batch.session_id,
                    covered_source_ids,
                    read_budget=read_budget,
                )
                if (
                    not exact_sources_exist
                    or locked_source_hash != batch.source_identity_hash
                ):
                    self._frontier.update_batch_state_no_commit(
                        publication_conn,
                        batch_id,
                        "rejected",
                        failure_reason="source_identity_mismatch",
                    )
                    return _result(
                        promoted=False, reason="source_identity_mismatch"
                    )

                # Both publisher paths use this writer lock and coverage check,
                # so precisely one overlapping canonical leaf can win.
                covered = self._canonical_message_source_ids_no_commit(
                    publication_conn,
                    batch.session_id,
                    deadline_at=float(read_budget["deadline_at"]),
                    max_rows=_CANONICAL_LINEAGE_MAX_ROWS,
                    max_edges=_CANONICAL_LINEAGE_MAX_EDGES,
                    max_depth=_CANONICAL_LINEAGE_MAX_DEPTH,
                    max_bytes=_CANONICAL_LINEAGE_MAX_BYTES,
                    read_budget=read_budget,
                )
                if covered.intersection(covered_source_ids):
                    self._frontier.update_batch_state_no_commit(
                        publication_conn,
                        batch_id,
                        "rejected",
                        failure_reason="canonical_source_overlap",
                    )
                    return _result(
                        promoted=False,
                        reason="canonical_source_overlap",
                    )

                placeholders = ",".join("?" for _ in covered_source_ids)
                bounds = publication_conn.execute(
                    f"""
                    SELECT MIN(timestamp), MAX(timestamp) FROM messages
                    WHERE session_id = ? AND store_id IN ({placeholders})
                    """,
                    (batch.session_id, *covered_source_ids),
                ).fetchone()
                node = SummaryNode(
                    session_id=batch.session_id,
                    depth=0,
                    summary=summary_text,
                    token_count=leaf_tokens,
                    source_token_count=source_tokens,
                    source_ids=list(covered_source_ids),
                    source_type="messages",
                    created_at=time.time(),
                    earliest_at=bounds[0] if bounds else None,
                    latest_at=bounds[1] if bounds else None,
                    expand_hint=expand_hint,
                )
                inserted_node_id = self._dag.add_node_no_commit(
                    publication_conn, node
                )
                _publication_boundary("after_canonical_insert")

                frontier_items = self._build_promoted_frontier_items_no_commit(
                    publication_conn,
                    conversation_id=batch.conversation_id,
                    session_id=batch.session_id,
                    node_id=int(inserted_node_id),
                    covered_source_ids=covered_source_ids,
                    frontier_end_store_id=int(batch.frontier_end_store_id or 0),
                    base_generation=int(batch.base_generation),
                    read_budget=read_budget,
                )
                if not frontier_items:
                    raise RuntimeError("frontier_items_empty_after_promotion")

                self._frontier._publication_phase_hook = _publication_boundary
                try:
                    new_gen = self._frontier.advance_frontier_generation_with_items(
                        batch.conversation_id,
                        batch.session_id,
                        batch.frontier_end_store_id,
                        current_policy_fp,
                        current_route_fp,
                        batch.base_generation,
                        frontier_items,
                    )
                finally:
                    self._frontier._publication_phase_hook = None
                if new_gen == 0:
                    raise RuntimeError("frontier_changed_inside_publication_transaction")

                self._frontier.update_batch_state(batch_id, "promoted")
                self._frontier.supersede_competing_batches_no_commit(
                    publication_conn,
                    batch.conversation_id,
                    batch.base_generation,
                    winner_batch_id=batch_id,
                    reason="async_generation_published",
                )
                _publication_boundary("after_batch_promoted")

                with self._lifecycle.publication_connection(publication_conn):
                    lifecycle_state = self._lifecycle.advance_frontier(
                        batch.conversation_id,
                        batch.session_id,
                        batch.frontier_end_store_id,
                    )
                if (
                    lifecycle_state is None
                    or lifecycle_state.current_session_id != batch.session_id
                    or int(lifecycle_state.current_frontier_store_id or 0)
                    < int(batch.frontier_end_store_id or 0)
                ):
                    raise RuntimeError("lifecycle_frontier_not_advanced")
                _publication_boundary("after_lifecycle_advanced")

            _publication_boundary("after_commit")

            publication_ms = (time.perf_counter() - publication_started) * 1000.0
            wall_ms = (time.perf_counter() - wall_started) * 1000.0
            self._async_last_promote_at = time.time()
            self._async_last_promote_validation_ms = validation_ms
            self._async_last_promote_publication_ms = publication_ms
            self._async_last_promote_wall_ms = wall_ms
            self._async_last_promoted_source_ids = list(covered_source_ids)
            self._async_last_promoted_node_id = int(inserted_node_id or 0)
            self._async_total_promote_succeeded = (
                int(getattr(self, "_async_total_promote_succeeded", 0) or 0) + 1
            )
            logger.info(
                "LCM async promote batch=%s node=%s validation=%.1fms publication=%.1fms wall=%.1fms",
                batch_id,
                inserted_node_id,
                validation_ms,
                publication_ms,
                wall_ms,
            )
            return _result(
                promoted=True,
                node_id=int(inserted_node_id or 0),
                covered=covered_source_ids,
            )

        except Exception as exc:
            # Before commit, the transaction context has already rolled back
            # every publication table and FTS side effect. Injected exception
            # failures deliberately leave the ready batch retryable.
            if (
                isinstance(exc, RuntimeError)
                and "injected async promotion failure" in str(exc)
            ):
                raise

            logger.error("LCM async promotion failed: %s", exc)
            # A failure from ``after_commit`` is observational only: durable
            # state is already wholly new and must never be mislabeled rejected.
            committed_batch = self._frontier.get_batch(batch_id)
            if committed_batch is not None and committed_batch.state == "promoted":
                return _result(
                    promoted=True,
                    reason="promotion_committed_before_observer_error",
                    node_id=int(inserted_node_id or 0),
                    covered=covered_source_ids,
                )
            self._frontier.update_batch_state(
                batch_id,
                "rejected",
                failure_reason=f"promotion_error: {exc}",
            )
            return _result(promoted=False, reason=f"promotion_error: {exc}")

    def _build_promoted_frontier_items_no_commit(
        self,
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        node_id: int,
        covered_source_ids: list[int],
        frontier_end_store_id: int,
        base_generation: int,
        read_budget: dict[str, float | int],
    ) -> list[dict[str, Any]]:
        """Build a frontier layout from the caller's locked SQLite snapshot."""
        covered_set = {int(source_id) for source_id in covered_source_ids}
        if not covered_set or any(source_id <= 0 for source_id in covered_set):
            raise RuntimeError("foreground candidate has invalid source coverage")
        covered_start = min(covered_set)
        covered_end = max(covered_set)
        end_id = int(frontier_end_store_id or 0)
        items: list[dict[str, Any]] = []
        active_raw_ids: set[int] = set()
        active = conn.execute(
            """
            SELECT generation, session_id FROM lcm_active_frontiers
            WHERE conversation_id = ? ORDER BY generation DESC LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if (
            active is not None
            and int(active[0] or 0) == int(base_generation)
            and str(active[1] or "") == session_id
        ):
            rows = self._bounded_frontier_rows_no_commit(
                conn,
                conversation_id,
                int(base_generation),
                read_budget=read_budget,
            )
            for _ordinal, kind, raw_ref_id, raw_start, raw_end, node_session in rows:
                if kind not in {"node", "message"}:
                    raise RuntimeError("foreground frontier contains invalid item kind")
                ref_id = int(raw_ref_id or 0)
                start = int(raw_start or 0)
                end = int(raw_end or 0)
                if start <= 0 or end < start:
                    raise RuntimeError("foreground frontier contains invalid range")
                if kind == "message":
                    if start != ref_id or end != ref_id:
                        raise RuntimeError("foreground message frontier range is invalid")
                    if ref_id in covered_set:
                        # Exact raw-message lineage is intentionally replaced by
                        # the new canonical node.
                        continue
                    if not (
                        end < covered_start
                        or start > covered_end
                    ):
                        raise RuntimeError(
                            "foreground candidate declared range overlap would lose raw coverage"
                        )
                    # Preserve exact disjoint raw lineage. The tail scan below
                    # may rediscover the same row; that exact duplicate is the
                    # only safe case to coalesce.
                    active_raw_ids.add(ref_id)
                    items.append(
                        {
                            "kind": "message",
                            "ref_id": ref_id,
                            "source_start": ref_id,
                            "source_end": ref_id,
                        }
                    )
                    continue
                if not (
                    end < covered_start
                    or start > covered_end
                ):
                    raise RuntimeError(
                        "foreground candidate declared range overlap would lose canonical coverage"
                    )
                if node_session != session_id:
                    raise RuntimeError("foreground frontier node crosses session boundary")
                items.append(
                    {
                        "kind": "node",
                        "ref_id": ref_id,
                        "source_start": start,
                        "source_end": end,
                    }
                )
        if node_id != 0:
            items.append(
                {
                    "kind": "node",
                    "ref_id": int(node_id),
                    "source_start": covered_start,
                    "source_end": covered_end,
                }
            )
        rows = self._bounded_message_tail_ids_no_commit(
            conn,
            session_id,
            end_id,
            read_budget=read_budget,
        )
        for store_id in rows:
            if (
                store_id <= 0
                or store_id in covered_set
                or store_id in active_raw_ids
            ):
                continue
            items.append(
                {
                    "kind": "message",
                    "ref_id": store_id,
                    "source_start": store_id,
                    "source_end": store_id,
                }
            )
        ordered = sorted(
            items,
            key=lambda item: (
                int(item.get("source_start") or 0),
                0 if item.get("kind") == "node" else 1,
                int(item.get("ref_id") or 0),
            ),
        )
        validated: list[dict[str, Any]] = []
        previous_end = 0
        for item in ordered:
            start = int(item.get("source_start") or 0)
            end = int(item.get("source_end") or 0)
            if start <= previous_end or end < start:
                raise RuntimeError(
                    "foreground candidate range overlap would drop a frontier item"
                )
            validated.append(item)
            previous_end = end
        return validated

    @staticmethod
    def _bounded_frontier_rows_no_commit(
        conn: sqlite3.Connection,
        conversation_id: str,
        generation: int,
        *,
        read_budget: dict[str, float | int],
    ) -> list[tuple[int, str, int, int, int, str | None]]:
        rows: list[tuple[int, str, int, int, int, str | None]] = []
        last_ordinal = -1
        while True:
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("locked frontier read deadline exceeded")
            remaining = int(read_budget["max_rows"]) - int(read_budget["rows"])
            page_limit = min(_PUBLICATION_LOCKED_QUERY_BATCH, remaining + 1)
            page = conn.execute(
                """SELECT i.ordinal, i.kind, i.ref_id, i.source_start, i.source_end,
                          n.session_id
                   FROM lcm_frontier_items AS i
                   LEFT JOIN summary_nodes AS n
                     ON i.kind = 'node' AND n.node_id = i.ref_id
                   WHERE i.conversation_id = ? AND i.generation = ?
                     AND i.ordinal > ?
                   ORDER BY i.ordinal LIMIT ?""",
                (conversation_id, int(generation), last_ordinal, page_limit),
            ).fetchall()
            if not page:
                return rows
            for ordinal, kind, ref_id, start, end, node_session in page:
                encoded_bytes = (
                    len(str(kind or "").encode("utf-8", errors="replace"))
                    + len(str(node_session or "").encode("utf-8", errors="replace"))
                    + 48
                )
                LCMEngine._charge_locked_publication_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=encoded_bytes,
                    label="locked frontier read",
                )
                last_ordinal = int(ordinal)
                rows.append((
                    last_ordinal,
                    str(kind or ""),
                    int(ref_id or 0),
                    int(start or 0),
                    int(end or 0),
                    None if node_session is None else str(node_session),
                ))
            if len(page) < page_limit:
                return rows

    @staticmethod
    def _bounded_message_tail_ids_no_commit(
        conn: sqlite3.Connection,
        session_id: str,
        after_store_id: int,
        *,
        read_budget: dict[str, float | int],
        conversation_id: str | None = None,
    ) -> list[int]:
        rows: list[int] = []
        last_store_id = int(after_store_id)
        while True:
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("locked message tail deadline exceeded")
            remaining = int(read_budget["max_rows"]) - int(read_budget["rows"])
            page_limit = min(_PUBLICATION_LOCKED_QUERY_BATCH, remaining + 1)
            if conversation_id:
                page = conn.execute(
                    """SELECT store_id FROM messages
                       WHERE session_id = ? AND conversation_id = ? AND store_id > ?
                       ORDER BY store_id LIMIT ?""",
                    (session_id, conversation_id, last_store_id, page_limit),
                ).fetchall()
            else:
                page = conn.execute(
                    """SELECT store_id FROM messages
                       WHERE session_id = ? AND store_id > ?
                       ORDER BY store_id LIMIT ?""",
                    (session_id, last_store_id, page_limit),
                ).fetchall()
            if not page:
                return rows
            for (raw_store_id,) in page:
                LCMEngine._charge_locked_publication_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=32,
                    label="locked message tail",
                )
                last_store_id = int(raw_store_id)
                rows.append(last_store_id)
            if len(page) < page_limit:
                return rows

    def reconcile_itemless_frontier_generations(
        self,
        conversation_id: str | None = None,
    ) -> int:
        """Repair active generations that have source_end > 0 but no items.

        Rebuilds items from current canonical DAG leaves + uncovered raw
        tail. Returns the number of generations repaired.
        """
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id:
            return 0
        itemless = self._frontier.list_itemless_active_generations(conv_id)
        repaired = 0
        for row in itemless:
            session_id = str(row.get("session_id") or "")
            generation = int(row.get("generation") or 0)
            source_end = int(row.get("source_end_store_id") or 0)
            if not session_id or generation <= 0 or source_end <= 0:
                continue
            try:
                nodes = self._dag.get_session_nodes(session_id)
            except Exception:
                nodes = []
            # Prefer uncondensed depth-0 leaves whose source range ends at or
            # before the generation's source_end.
            node_items: list[dict[str, Any]] = []
            covered: set[int] = set()
            for node in nodes:
                if getattr(node, "depth", 0) != 0:
                    continue
                if getattr(node, "source_type", "") != "messages":
                    continue
                sids = [int(s) for s in (getattr(node, "source_ids", None) or [])]
                if not sids:
                    continue
                if max(sids) > source_end:
                    continue
                covered.update(sids)
                node_items.append(
                    {
                        "kind": "node",
                        "ref_id": int(node.node_id),
                        "source_start": min(sids),
                        "source_end": max(sids),
                    }
                )
            # Sort node items by source_start for monotonic ordering.
            node_items.sort(key=lambda it: (it["source_start"], it["ref_id"]))
            items = list(node_items)
            try:
                stored = self._store.get_session_messages(session_id)
            except Exception:
                stored = []
            for msg in stored:
                try:
                    sid = int(msg.get("store_id") or 0)
                except (TypeError, ValueError):
                    continue
                if sid <= 0 or sid in covered:
                    continue
                if sid <= source_end and any(
                    it["source_start"] <= sid <= it["source_end"] for it in node_items
                ):
                    continue
                if sid <= source_end and node_items:
                    # Covered by the frontier end via raw-only generations:
                    # skip messages already inside a node range.
                    continue
                if sid > source_end:
                    items.append(
                        {
                            "kind": "message",
                            "ref_id": sid,
                            "source_start": sid,
                            "source_end": sid,
                        }
                    )
            if not items and source_end > 0:
                # Last resort: single message range marker so the tip is not
                # itemless (doctor can still flag sparse content later).
                items.append(
                    {
                        "kind": "message",
                        "ref_id": source_end,
                        "source_start": source_end,
                        "source_end": source_end,
                    }
                )
            if items:
                self._frontier.set_frontier_items(conv_id, generation, items)
                repaired += 1
        return repaired

    def reject_prepared_compaction(self, batch_id: int, reason: str = "") -> None:
        """Mark a prepared batch as rejected."""
        self._frontier.update_batch_state(batch_id, "rejected", failure_reason=reason)

    def _try_promote_prepared_batch(self, messages: List[Dict[str, Any]]) -> bool:
        """Promote a ready async batch at the turn boundary if possible.

        Returns True when a batch was promoted so ``compress()`` can skip the
        foreground leaf summarization path and reassemble from the DAG.
        On success, ``_async_last_promoted_source_ids`` holds the covered
        store_ids so the host replacement path can drop those raw rows.
        """
        self._async_promotion_ingested_messages = None
        if not getattr(self._config, "async_background_compaction_enabled", False):
            return False
        if not getattr(self._config, "async_background_compaction_promote_on_compress", True):
            return False
        conv_id = self.current_conversation_id
        if not conv_id:
            return False
        batch = self._frontier.get_ready_batch(conv_id)
        if batch is None:
            return False
        result = self.promote_prepared_compaction(
            batch.batch_id,
            messages,
        )
        if not result.promoted:
            return False
        # Keep in-process frontier marker aligned with the promoted end.
        active = self._frontier.get_active_frontier(batch.conversation_id)
        end_id = int(
            (active or {}).get("source_end_store_id")
            or batch.frontier_end_store_id
            or batch.source_end_store_id
            or 0
        )
        if end_id > 0:
            self._last_compacted_store_id = max(
                int(self._last_compacted_store_id or 0),
                end_id,
            )
            try:
                self._persist_frontier_marker()
            except Exception:
                logger.debug(
                    "LCM persist frontier after async promote failed",
                    exc_info=True,
                )
        # Stash covered IDs for compress() host replacement (issue #2).
        if result.covered_source_ids:
            self._async_last_promoted_source_ids = list(result.covered_source_ids)
        if result.node_id:
            self._async_last_promoted_node_id = int(result.node_id)
        return True

    def _filter_messages_excluding_covered_store_ids(
        self,
        messages: List[Dict[str, Any]],
        covered_store_ids: Sequence[int] | set[int],
    ) -> List[Dict[str, Any]]:
        """Drop host messages whose durable store_id is in *covered_store_ids*.

        Used after async promote so the returned active context does not
        replay raw rows already published into a DAG leaf. Messages without
        a resolvable store_id are kept (e.g. pure system anchors).

        Mapping deliberately ignores ``_last_compacted_store_id``: after
        promote that marker already points at the covered end, so the normal
        compress map would skip every covered row and leave them active.
        """
        if not messages or not covered_store_ids:
            return list(messages)
        covered = {int(s) for s in covered_store_ids}
        # Temporarily clear the compacted frontier so identity mapping can
        # resolve covered rows that were just published.
        previous_marker = int(getattr(self, "_last_compacted_store_id", 0) or 0)
        try:
            self._last_compacted_store_id = 0
            id_map = self._get_store_id_map_for_messages(messages)
        finally:
            self._last_compacted_store_id = previous_marker
        kept: list[Dict[str, Any]] = []
        any_mapped_covered = False
        for msg in messages:
            store_id = id_map.get(id(msg))
            if store_id is not None and int(store_id) in covered:
                any_mapped_covered = True
                continue
            kept.append(msg)
        if any_mapped_covered:
            return kept
        # Fallback when identity mapping fails (e.g. host payload drift): keep
        # leading anchors + the configured fresh tail only.
        leading = self._leading_anchor_count(messages)
        tail_start = self._fresh_tail_start(messages)
        if tail_start >= len(messages):
            return list(messages[:leading]) if leading else []
        return list(messages[:leading]) + list(messages[tail_start:])

    # -- Async background worker -------------------------------------------

    def _start_async_worker(self) -> None:
        """Start the daemon preparer thread when both async flags are enabled.

        Install is refused unless storage lifetime state is ``bound``. The
        lifetime lock serializes install with stop→close→rebind / stop→close
        so a replacement cannot be started against closing or closed helpers.
        """
        with self._storage_lifetime_lock:
            if self._storage_lifetime_state != "bound":
                return
            if not getattr(self._config, "async_background_compaction_enabled", False):
                self._stop_async_worker()
                return
            if not getattr(self._config, "async_background_compaction_worker_enabled", False):
                self._stop_async_worker()
                return
            with self._async_worker_lock:
                if self._async_worker_thread is not None and self._async_worker_thread.is_alive():
                    return
                stop_event = threading.Event()
                self._async_worker_stop = stop_event
                thread = threading.Thread(
                    target=self._async_worker_loop,
                    name="lcm-async-compaction-worker",
                    daemon=True,
                )
                self._async_worker_thread = thread
                thread.start()
                logger.debug("LCM async background compaction worker started")

    def _stop_async_worker(self) -> bool:
        """Signal the preparer thread to stop; return whether it has exited.

        A stop event cannot cancel an in-flight LLM request.  Keeping the
        storage helpers alive until the worker exits prevents its post-request
        state update from racing a closed SQLite connection.

        Runs under ``_storage_lifetime_lock`` so install cannot slip into the
        join gap. Returns True only when no live worker remains in the owned
        slot.
        """
        with self._storage_lifetime_lock:
            with self._async_worker_lock:
                stop_event = self._async_worker_stop
                thread = self._async_worker_thread
            if stop_event is not None:
                stop_event.set()
            if thread is not None and thread.is_alive():
                thread.join(timeout=35.0)
                if thread.is_alive():
                    logger.warning(
                        "LCM async background compaction worker did not stop within 35s; "
                        "deferring storage cleanup until it exits"
                    )
                    return False
            with self._async_worker_lock:
                if self._async_worker_thread is thread:
                    self._async_worker_stop = None
                    self._async_worker_thread = None
                remaining = self._async_worker_thread
                if remaining is not None and remaining.is_alive():
                    return False
            return True

    def _async_worker_loop(self) -> None:
        """Periodic prepare loop. Never raises out of the thread."""
        stop_event = self._async_worker_stop
        if stop_event is None:
            return
        interval = float(
            getattr(
                self._config,
                "async_background_compaction_worker_interval_seconds",
                30.0,
            )
            or 30.0
        )
        if interval <= 0:
            interval = 30.0
        while not stop_event.is_set():
            # Event.wait is cancellable — avoids uninterruptible sleep.
            if stop_event.wait(timeout=interval):
                break
            try:
                # True = prepare succeeded, False = prepare failed,
                # None = intentional skip (no prepare attempted).
                outcome = self._async_worker_tick()
            except Exception:
                logger.warning(
                    "LCM async background compaction worker tick failed",
                    exc_info=True,
                )
                outcome = False
            if outcome is not None:
                self._async_worker_note_tick_outcome(bool(outcome))

    def _async_worker_note_tick_outcome(self, success: bool) -> None:
        """Update consecutive-failure circuit breaker after a prepare attempt."""
        if success:
            self._async_worker_consecutive_failures = 0
            self._async_worker_half_open = False
            return
        self._async_worker_consecutive_failures = (
            int(getattr(self, "_async_worker_consecutive_failures", 0) or 0) + 1
        )
        max_failures = int(
            getattr(
                self._config,
                "async_background_compaction_worker_max_consecutive_failures",
                3,
            )
            or 3
        )
        # Half-open after a prior cooldown: a single failure re-trips.
        threshold = 1 if getattr(self, "_async_worker_half_open", False) else max_failures
        if threshold < 1:
            threshold = 1
        if self._async_worker_consecutive_failures >= threshold:
            cooldown = float(
                getattr(
                    self._config,
                    "async_background_compaction_worker_cooldown_seconds",
                    60.0,
                )
                or 60.0
            )
            self._async_worker_cooldown_until = time.time() + max(0.0, cooldown)
            self._async_worker_consecutive_failures = 0
            self._async_worker_half_open = True
            logger.info(
                "LCM async worker entering cooldown for %.1fs after prepare failures",
                cooldown,
            )

    def _async_worker_tick(self) -> Optional[bool]:
        """One preparation attempt if the session is idle and has backlog.

        Returns:
            True — prepare succeeded
            False — prepare failed (or unrecoverable load error)
            None — intentional skip (cooldown / no work / already ready)
        """
        tick_started = time.perf_counter()
        self._async_worker_last_tick_at = time.time()
        try:
            return self._async_worker_tick_body()
        finally:
            self._async_worker_last_tick_duration_ms = (
                time.perf_counter() - tick_started
            ) * 1000.0

    def _async_worker_tick_body(self) -> Optional[bool]:
        """Core worker tick logic without timing bookkeeping."""
        if time.time() < float(getattr(self, "_async_worker_cooldown_until", 0.0) or 0.0):
            return None
        if not getattr(self._config, "async_background_compaction_enabled", False):
            return None
        if not getattr(self._config, "async_background_compaction_worker_enabled", False):
            return None
        # Foreground cutover has absolute priority (issue #3): never enter
        # prepare LLM / SQLite critical sections while compress() is running.
        if self._foreground_compress_active.is_set():
            return None
        if getattr(self, "_last_compression_status", "") == "running":
            return None
        session_id = self.current_session_id
        conv_id = self.current_conversation_id
        if not session_id or not conv_id:
            return None
        counts = self._frontier.get_batch_counts_by_state(conv_id)
        if counts.get("ready", 0) > 0 or counts.get("preparing", 0) > 0:
            return None
        try:
            stored = self._store.get_session_messages(session_id)
        except Exception:
            logger.warning(
                "LCM async worker failed to load session messages",
                exc_info=True,
            )
            return False
        if not stored or self._fresh_tail_start(stored) <= self._leading_anchor_count(stored):
            return None
        # prepare_background_compaction_once snapshots from the store; the
        # messages argument is only needed for API compatibility.
        batch = self.prepare_background_compaction_once([])
        if batch is None:
            return None
        if batch.state == "failed":
            return False
        return True

    # -- Internal: helpers -------------------------------------------------

    def _assemble_overflow_recovery_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if tail_messages:
            first = tail_messages[0]
            content = first.get("content") or ""
            role = first.get("role") or ""
            if role == "assistant" and self._looks_like_active_summary_blob(content):
                candidate = self._assemble_context(
                    system_msg,
                    tail_messages[1:],
                    assembly_cap_override=assembly_cap_override,
                    include_lcm_note=False,
                )
                if any(
                    (msg.get("content") or "") == content
                    for msg in (candidate[1:] if system_msg is not None else candidate)
                ):
                    return candidate

        candidate = self._assemble_context(
            system_msg,
            tail_messages,
            assembly_cap_override=assembly_cap_override,
            include_lcm_note=False,
        )
        minimum_candidate_len = 1 if system_msg is not None else 0
        if len(candidate) == minimum_candidate_len and tail_messages:
            fallback = ([system_msg] if system_msg is not None else []) + [tail_messages[-1]]
            return self._sanitize_active_context_messages(fallback)
        return candidate

    @staticmethod
    def _looks_like_active_summary_blob(content: str) -> bool:
        if not isinstance(content, str) or not content:
            return False
        block = (
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
            r".*?\n"
            r"\[Expand for details: .*?\]"
        )
        pattern = rf"^{block}(?:\n\n---\n\n{block})*$"
        return re.fullmatch(pattern, content, flags=re.DOTALL) is not None

    def _derive_auto_focus_topic(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns.

        Walks the message list backwards, collecting up to
        ``_AUTO_FOCUS_MAX_TURNS`` user messages (skipping context summaries
        and empty turns).  Returns a brief text block suitable for injection
        into the summarizer prompt as ``focus_topic``.

        IMPORTANT: The ``messages`` parameter must be ``working_messages``
        (output of ``_ingest_messages``), not raw messages.  ``working_messages``
        has already been redacted by ``_redact_active_replay_messages``.

        As an additional safety layer, text extracted by
        ``text_content_for_pattern_matching`` is run through
        ``redact_sensitive_text`` with the active config.  This covers
        sensitive values that ``_redact_active_replay_messages`` misses
        (e.g., dict/JSON token content deserialized into text,
        bearer-style auth text that survived structured-content flattening).

        Mirrors Hermes upstream ``ContextCompressor._derive_auto_focus_topic``
        from ``fix/compression-auto-focus-topic``.
        """
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            # Skip context compaction summaries — they are synthetic, not
            # real user intent.
            if self._is_context_summary_content(content):
                continue
            text = (text_content_for_pattern_matching(content) or "").strip()
            if self._matches_ignore_message_patterns(msg) or self._is_volatile_ignored_quarantine_placeholder(
                msg,
                text,
            ) or self._is_ignored_active_replay_placeholder(msg, text):
                continue
            # Additional redaction safety net: run extracted text through the
            # configured redaction path.  _redact_active_replay_messages uses
            # parse_json_strings=False for content, so structured content
            # (dict/JSON tokens, bearer-style auth text) may not be fully
            # covered.  This extra pass ensures the same redaction rules apply
            # to whatever text is extracted for the focus topic.
            text = redact_sensitive_text(text, self._config)
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        """Check whether message content is a synthetic context summary.

        Only checks string content — LCM/ Hermes compression summaries are
        always stored as plain strings, never as structured multimodal parts.
        """
        if not isinstance(content, str):
            return False
        return (
            "CONTEXT COMPACTION" in content
            or "CONTEXT SUMMARY" in content
            or "Earlier turns have been compacted" in content
            or "Earlier turns were compacted" in content
        )

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Rotate ------------------------------------------------------------

    def backup_dir(self) -> Path:
        """Return the directory where LCM backup snapshots are written.

        Centralized so the timestamped ``/lcm backup`` slot and the rolling
        ``/lcm rotate apply`` slot share the same directory derivation.
        """
        db_path = Path(self._store.db_path)
        backup_root = (
            Path(self._hermes_home).expanduser()
            if getattr(self, "_hermes_home", "")
            else db_path.parent
        )
        return backup_root / "backups" / "lcm"

    def rotate_backup_path(self) -> Path:
        """Return the rolling rotate-latest SQLite backup path for this engine.

        Centralized so command.py (which writes the backup) and get_status()
        (which reads its mtime to surface last_rotate_at) cannot drift.
        """
        db_path = Path(self._store.db_path)
        return self.backup_dir() / f"{db_path.stem}-rotate-latest.sqlite3"

    def rotate_active_session(
        self,
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Compact the active session in-place without changing identity.

        Read-only by default (``apply=False``). Returns a preview describing
        what would change. When ``apply=True``, advances the lifecycle frontier
        marker past the pre-tail raw messages so they are no longer replayed
        into active context on subsequent bootstrap. Raw messages remain in
        the SQLite store and are recoverable through ``lcm_load_session`` and
        ``lcm_expand`` — the lossless raw recovery contract is preserved.

        Refuses on sessions that are unbound, ignored, or stateless.

        Two frontier markers are intentionally kept separate:

        - The **persisted lifecycle frontier**
          (``lifecycle_state.current_frontier_store_id``) is the
          bootstrap signal — on next session start, raw rows at or
          below it are not replayed into the active context. Rotate
          advances this marker.
        - The **in-process source-mapping marker**
          (``self._last_compacted_store_id``) tracks raw rows that the
          *current process* has already moved into summary DAG nodes.
          ``_get_store_ids_for_messages`` uses it to filter candidates
          when mapping in-memory active messages back to ``store_id``.
          Rotate deliberately does NOT advance this marker: pre-tail
          raw messages remain in the in-memory active context until
          the host rebuilds it, so a normal ``compress()`` later in
          the same process can still summarize them with correct
          ``source_ids`` lineage. On next process start,
          ``_bind_lifecycle_state`` reads the persisted frontier into
          the in-process marker — at that point the active context is
          being built from scratch, so the contract holds.

        Refusal/no-op reason codes (returned as ``reason``):

        - ``no_active_session``: engine has no bound session or conversation.
        - ``session_ignored``: foreground session matched
          ``LCM_IGNORE_SESSION_PATTERNS``.
        - ``session_stateless``: foreground session matched
          ``LCM_STATELESS_SESSION_PATTERNS``.
        - ``no_pre_tail_content``: total stored messages do not exceed
          ``fresh_tail_count``; nothing to rotate.
        - ``empty_tail``: tail query returned no rows despite a non-zero
          count (concurrent deletion race); rotate cannot compute a boundary.
        - ``frontier_already_ahead``: lifecycle frontier is already at or
          past the proposed new frontier; rotate is a no-op.
        - ``stale_lifecycle_state``: apply requested but lifecycle's
          ``current_session_id`` did not match this engine's session, so
          ``advance_frontier`` did not persist the change.
        """
        session_id = self._session_id
        conversation_id = self._conversation_id

        if not session_id or not conversation_id:
            return {"ok": False, "reason": "no_active_session"}
        if self._session_ignored:
            return {"ok": False, "reason": "session_ignored", "session_id": session_id}
        if self._session_stateless:
            return {"ok": False, "reason": "session_stateless", "session_id": session_id}

        stored = self._store.get_session_messages(session_id)
        total_count = len(stored)
        fresh_tail_start = self._fresh_tail_start(stored)
        fresh_tail_count = total_count - fresh_tail_start

        state = self._lifecycle.get_by_conversation(conversation_id)
        current_frontier = int(state.current_frontier_store_id) if state else 0

        base = {
            "ok": True,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "total_message_count": total_count,
            "fresh_tail_count": fresh_tail_count,
            "fresh_tail_max_tokens": self._effective_fresh_tail_max_tokens(),
            "current_frontier_store_id": current_frontier,
            "mode": "apply" if apply else "preview",
        }

        if fresh_tail_start <= self._leading_anchor_count(stored):
            return {
                **base,
                "noop": True,
                "reason": "no_pre_tail_content",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        tail = self._store.get_session_tail(session_id, fresh_tail_count)
        if not tail:
            # Concurrent deletion can empty the tail after the count check.
            # Surface the same shape callers expect for any other no-op so
            # downstream formatters can render it without KeyError.
            return {
                **base,
                "noop": True,
                "reason": "empty_tail",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        smallest_tail_store_id = int(tail[0].get("store_id") or 0)
        new_frontier = max(0, smallest_tail_store_id - 1)
        pre_tail_count = max(0, total_count - len(tail))

        is_noop = new_frontier <= current_frontier
        result = {
            **base,
            "pre_tail_message_count": pre_tail_count,
            "new_frontier_store_id": new_frontier,
            "noop": is_noop,
        }
        if is_noop:
            # Set the reason for both preview and apply so downstream
            # formatters can render a stable explanation. Preview previously
            # omitted the reason, which left _rotate_apply_text's preflight
            # check unable to distinguish frontier-already-ahead from other
            # no-ops.
            result["reason"] = "frontier_already_ahead"

        if not apply:
            return result

        if is_noop:
            return result

        new_state = self._lifecycle.advance_frontier(
            conversation_id,
            session_id,
            new_frontier,
        )
        # advance_frontier silently returns the unchanged state when its
        # session_id check fails (lifecycle_state.py:557-559). Detect that
        # by checking whether the persisted frontier actually advanced; only
        # promote the in-process marker on a confirmed persist.
        persisted_frontier = (
            int(new_state.current_frontier_store_id) if new_state else current_frontier
        )
        if persisted_frontier < new_frontier:
            return {
                **{k: v for k, v in result.items() if k != "ok"},
                "ok": False,
                "noop": False,
                "reason": "stale_lifecycle_state",
                "applied_frontier_store_id": persisted_frontier,
            }
        # Deliberately do NOT touch self._last_compacted_store_id here.
        # The in-process source-mapping marker must stay aligned with the
        # in-memory active context the host is still using. Pre-tail raw
        # messages remain in that active context until the host rebuilds
        # it; advancing the marker would make
        # _get_store_ids_for_messages filter out those rows on the next
        # in-process compress(), producing summary nodes whose text
        # covers pre-rotate messages but whose source_ids reference only
        # post-rotate rows. The persisted lifecycle frontier we just
        # advanced is the bootstrap signal for the next process start,
        # where _bind_lifecycle_state will read it into the marker
        # against a freshly-built active context.
        result["applied_frontier_store_id"] = persisted_frontier
        return result

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        # Own the full stop→close transition under the lifetime lock so
        # concurrent session start/end, rebind, and late worker starts cannot
        # install or use helpers against storage that is closing.
        with self._storage_lifetime_lock:
            lcm_tools._cleanup_externalized_runtime_state(self)
            worker_stopped = self._stop_async_worker()
            self._unregister_active_engine_binding()
            if worker_stopped:
                self._close_storage()
                self._storage_lifetime_state = "closed"
            else:
                logger.warning(
                    "LCM deferred SQLite storage cleanup until async worker exits"
                )
