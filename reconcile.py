"""Ingest-cursor reconciliation and replay-identity for the LCM engine (WS5 Seam 4).

The ``ReconcileMixin`` holds the machinery that reconciles the persisted store
tail against the active message list after a process restart, plus the stable
replay-identity primitives it relies on. These methods were lifted verbatim out
of ``LCMEngine`` and continue to run bound to the engine instance (``self`` is
the ``LCMEngine``), so they read the engine's runtime state (``_store``,
``_session_id``, ``_config``, ``_ingest_cursor`` is written by the engine from
the value these return) and call back into engine helpers through normal
attribute lookup. ``LCMEngine`` mixes this in, so no call site and no test
changes.

``_PRESERVED_OBJECTIVE_CONTEXT_PREFIX`` lives here (used by the reconciliation
scan) and is re-exported to ``engine.py``; the two tool-call-identity
staticmethods reference the mixin class directly rather than ``LCMEngine`` to
avoid an import cycle (staticmethod resolution is identical).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    externalized_tool_result_has_persisted_output_marker,
    find_externalized_tool_result_content_for_call,
    load_externalized_payload,
)
from .ingest_protection import (
    _add_inline_persisted_output_generation_metadata,
    _add_inline_persisted_output_identity_metadata,
    _expected_persisted_output_chars,
    _has_inline_persisted_output_generation_metadata,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _json_has_duplicate_object_keys,
    _persisted_output_marker_identity_digest,
    _persisted_output_saved_path,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_value,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .sanitize import _clean_active_assistant_message

import logging

logger = logging.getLogger(__name__)

_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"
_RECONCILIATION_QUERY_BATCH = 400
_RECONCILIATION_MAX_FIELD_BYTES = 2 * 1024 * 1024
_RECONCILIATION_MAX_NESTED_DEPTH = 32
_RECONCILIATION_MAX_NESTED_ITEMS = 20_000


class ReconcileMixin:
    @staticmethod
    def _canonicalize_tool_call_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ReconcileMixin._canonicalize_tool_call_identity_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [ReconcileMixin._canonicalize_tool_call_identity_value(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                if _json_has_duplicate_object_keys(value):
                    return value
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return value
                if isinstance(parsed, (dict, list)):
                    canonical = ReconcileMixin._canonicalize_tool_call_identity_value(parsed)
                    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return value
        return value

    @staticmethod
    def _stable_tool_calls_identity(tool_calls: Any) -> str:
        if not tool_calls:
            return ""
        try:
            canonical = ReconcileMixin._canonicalize_tool_call_identity_value(tool_calls)
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _has_durable_persisted_output_replay_identity(
        self,
        msg: Dict[str, Any],
        *,
        read_budget: dict[str, float | int] | None = None,
    ) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if (
            expected_chars is None
            or not persisted_output_source_path
            or not persisted_output_preview_sha256
        ):
            return False
        recovered_with_stat = recover_hermes_persisted_output_with_file_stat(
            content,
            read_budget=read_budget,
            budget_label="source reconciliation persisted-output file",
            max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
            max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
        )
        if recovered_with_stat is None:
            return False
        require_live_file_freshness = True
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            require_persisted_output_file_not_newer=require_live_file_freshness,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
            read_budget=read_budget,
            budget_label="source reconciliation durable payload",
            max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
            max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
        )
        if durable_content is None:
            return False
        if recovered_with_stat is not None:
            recovered_content, _file_stat = recovered_with_stat
            if not self._recovered_content_matches_durable_identity(recovered_content, durable_content):
                return False
        return True

    def _message_replay_identity(
        self,
        msg: Dict[str, Any],
        *,
        stored_row: bool = False,
        read_budget: dict[str, float | int] | None = None,
    ) -> tuple[str, str, str, str]:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if (
            role == "tool"
            and _is_hermes_persisted_output_marker(content)
            and bool(getattr(self._config, "large_output_externalization_enabled", True))
        ):
            expected_chars = _expected_persisted_output_chars(content)
            persisted_output_source_path = _persisted_output_saved_path(content)
            persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
            durable_content = None
            recovered_with_stat = (
                recover_hermes_persisted_output_with_file_stat(
                    content,
                    read_budget=read_budget,
                    budget_label="source reconciliation persisted-output file",
                    max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                    max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
                )
                if not stored_row
                else None
            )
            recovered_content = recovered_with_stat[0] if recovered_with_stat is not None else None
            recovered_identity_content = None
            if recovered_content is not None:
                recovered_identity_content = normalize_content_value(
                    redact_sensitive_value(
                        recovered_content,
                        self._config,
                        parse_json_strings=False,
                    )
                )
            require_live_file_freshness = recovered_with_stat is not None

            def live_file_generation_identity() -> str:
                try:
                    live_stat = Path(str(persisted_output_source_path)).stat()
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"mtime_ns={live_stat.st_mtime_ns}; "
                        f"chars={expected_chars}]"
                    )
                except OSError:
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"chars={expected_chars}]"
                    )

            if (
                not stored_row
                and expected_chars is not None
                and persisted_output_source_path
                and persisted_output_preview_sha256
                and recovered_with_stat is not None
            ):
                durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    require_persisted_output_file_not_newer=require_live_file_freshness,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                    read_budget=read_budget,
                    budget_label="source reconciliation durable payload",
                    max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                    max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
                )
            if durable_content is not None and (
                recovered_content is None or self._recovered_content_matches_durable_identity(recovered_content, durable_content)
            ):
                content = durable_content
            elif recovered_content is not None:
                stale_durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                    read_budget=read_budget,
                    budget_label="source reconciliation durable payload",
                    max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                    max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
                )
                if (
                    stale_durable_content is not None
                    and self._recovered_content_matches_durable_identity(recovered_content, stale_durable_content)
                    and not _has_lossy_sensitive_redaction(stale_durable_content)
                    and not _has_lossy_sensitive_redaction(recovered_identity_content)
                ):
                    content = stale_durable_content
                elif stale_durable_content is not None:
                    content = live_file_generation_identity()
                elif recovered_with_stat is not None:
                    content = _add_inline_persisted_output_generation_metadata(
                        _add_inline_persisted_output_identity_metadata(
                            content,
                            _persisted_output_marker_identity_digest(content),
                        ),
                        recovered_with_stat[1],
                    )
                elif recovered_identity_content is not None:
                    content = recovered_identity_content
        tool_calls = msg.get("tool_calls")
        if stored_row:
            session_id = str(msg.get("session_id") or self._session_id or "")
            content = self._restore_ingest_payload_placeholders_in_content_identity(
                content,
                session_id=session_id,
                read_budget=read_budget,
            )
            tool_calls = self._restore_ingest_payload_placeholders_in_value(
                tool_calls, session_id=session_id, read_budget=read_budget
            )
        ref = extract_externalized_ref(content)
        if ref and "quarantined_assistant_output" not in content:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
                read_budget=read_budget,
                budget_label="source reconciliation externalized payload",
                max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
            )
            if payload is not None and isinstance(payload.get("content"), str):
                content = payload["content"]
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        return (
            role,
            content,
            str(msg.get("tool_call_id") or ""),
            tool_calls_identity,
        )

    @staticmethod
    def _matches_store_tail_suffix(
        stored_tail: list[tuple[str, str, str, str]],
        candidate_prefix: list[tuple[str, str, str, str]],
    ) -> bool:
        if not candidate_prefix:
            return True
        if len(candidate_prefix) > len(stored_tail):
            return False
        return stored_tail[-len(candidate_prefix) :] == candidate_prefix

    @staticmethod
    def _check_reconciliation_deadline(read_budget: dict[str, float | int]) -> None:
        if time.monotonic() >= float(read_budget["deadline_at"]):
            raise RuntimeError("source reconciliation deadline exceeded")

    @classmethod
    def _store_tail_suffix_matching_prefix_lengths(
        cls,
        stored_tail: list[tuple[str, str, str, str]],
        active_identities: list[tuple[str, str, str, str]],
        *,
        read_budget: dict[str, float | int],
    ) -> set[int]:
        """Return active-prefix lengths equal to a suffix of ``stored_tail``.

        The KMP failure table makes this linear even when every identity is
        repeated.  The old reverse-cursor scan rebuilt and compared each
        prefix independently, making a no-match restart quadratic.
        """
        cls._check_reconciliation_deadline(read_budget)
        if not active_identities:
            return {0}

        failure = [0] * len(active_identities)
        for index in range(1, len(active_identities)):
            cls._check_reconciliation_deadline(read_budget)
            matched = failure[index - 1]
            while matched and active_identities[index] != active_identities[matched]:
                cls._check_reconciliation_deadline(read_budget)
                matched = failure[matched - 1]
            if active_identities[index] == active_identities[matched]:
                matched += 1
            failure[index] = matched

        matched = 0
        for identity in stored_tail:
            cls._check_reconciliation_deadline(read_budget)
            while matched and (
                matched == len(active_identities)
                or identity != active_identities[matched]
            ):
                cls._check_reconciliation_deadline(read_budget)
                matched = failure[matched - 1]
            if matched < len(active_identities) and identity == active_identities[matched]:
                matched += 1

        matching_lengths = {0}
        while matched:
            cls._check_reconciliation_deadline(read_budget)
            matching_lengths.add(matched)
            matched = failure[matched - 1]
        return matching_lengths

    @staticmethod
    def _strip_inline_persisted_output_generation_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, content, tool_call_id, tool_calls = identity
        if role != "tool" or not isinstance(content, str):
            return identity
        stripped = re.sub(
            r"\n?\[LCM persisted-output file generation: "
            r"size=\d+; mtime_ns=\d+; ctime_ns=\d+\]\n?(?=</persisted-output>)",
            "\n",
            content,
        )
        return (role, stripped, tool_call_id, tool_calls)

    def _stored_row_has_durable_persisted_output_marker(
        self,
        row: Dict[str, Any],
        *,
        read_budget: dict[str, float | int] | None = None,
    ) -> bool:
        if str(row.get("role") or "") != "tool":
            return False
        content = normalize_content_value(row.get("content")) or ""
        ref = extract_externalized_ref(content)
        if not ref:
            return False
        return externalized_tool_result_has_persisted_output_marker(
            ref,
            config=self._config,
            hermes_home=self._hermes_home,
            read_budget=read_budget,
            budget_label="source reconciliation persisted-output marker",
            max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
            max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
        )

    @staticmethod
    def _persisted_output_durable_wildcard_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, _content, tool_call_id, tool_calls = identity
        return (role, "[LCM persisted-output durable replay]", tool_call_id, tool_calls)

    def _matches_persisted_output_durable_full_replay(
        self,
        candidate_messages: list[Dict[str, Any]],
        candidate_prefix: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_tail_rows: list[Dict[str, Any]] | None,
        *,
        read_budget: dict[str, float | int] | None = None,
        durable_cache: dict[int, bool] | None = None,
        stored_marker_cache: dict[int, bool] | None = None,
    ) -> bool:
        if not stored_tail_rows or len(candidate_prefix) != len(stored_tail) or len(candidate_messages) != len(candidate_prefix):
            return False
        transformed_candidate: list[tuple[str, str, str, str]] = []
        transformed_stored: list[tuple[str, str, str, str]] = []
        saw_persisted_output = False
        for candidate_msg, candidate_identity, stored_identity, stored_row in zip(
            candidate_messages,
            candidate_prefix,
            stored_tail,
            stored_tail_rows,
        ):
            if read_budget is not None:
                self._check_reconciliation_deadline(read_budget)
            candidate_content = normalize_content_value(candidate_msg.get("content")) or ""
            candidate_is_persisted_marker = (
                str(candidate_msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(candidate_content)
            )
            stored_key = int(stored_row.get("store_id") or id(stored_row))
            if stored_marker_cache is not None and stored_key in stored_marker_cache:
                stored_is_persisted_output = stored_marker_cache[stored_key]
            else:
                stored_is_persisted_output = self._stored_row_has_durable_persisted_output_marker(
                    stored_row, read_budget=read_budget
                )
                if stored_marker_cache is not None:
                    stored_marker_cache[stored_key] = stored_is_persisted_output
            candidate_key = id(candidate_msg)
            if durable_cache is not None and candidate_key in durable_cache:
                candidate_is_durable = durable_cache[candidate_key]
            else:
                candidate_is_durable = self._has_durable_persisted_output_replay_identity(
                    candidate_msg, read_budget=read_budget
                )
                if durable_cache is not None:
                    durable_cache[candidate_key] = candidate_is_durable
            if candidate_is_persisted_marker or stored_is_persisted_output:
                if (
                    not candidate_is_persisted_marker
                    or not stored_is_persisted_output
                    or not candidate_is_durable
                ):
                    return False
                saw_persisted_output = True
                transformed_candidate.append(self._persisted_output_durable_wildcard_identity(candidate_identity))
                transformed_stored.append(self._persisted_output_durable_wildcard_identity(stored_identity))
                continue
            transformed_candidate.append(candidate_identity)
            transformed_stored.append(stored_identity)
        return saw_persisted_output and transformed_candidate == transformed_stored

    @classmethod
    def _identity_content_for_active_cleanup(cls, content: str) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        """
        if not isinstance(content, str):
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return content
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return decoded
        return content

    @classmethod
    def _active_cleanup_replay_identity(
        cls,
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str] | None:
        role, content, tool_call_id, tool_calls = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        }
        if tool_calls:
            try:
                decoded_tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_tool_calls = tool_calls
            msg["tool_calls"] = decoded_tool_calls
        cleaned = _clean_active_assistant_message(msg)
        if cleaned is None:
            return None
        return (
            role,
            normalize_content_value(cleaned.get("content")) or "",
            tool_call_id,
            tool_calls,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls = identity
        if role != "assistant":
            return False
        text = str(content or "").strip()
        return bool(
            re.fullmatch(
                r"\[Externalized LCM ingest payload: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"field=[A-Za-z0-9_.:/<>\[\]-]+; "
                r"chars=\d+; bytes=\d+; "
                r"ref=[^\]\s]+\]",
                text,
            )
            or re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text,
            )
        )

    def _stored_tail_for_sanitized_active_replay(
        self,
        stored_tail: list[tuple[str, str, str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Mirror active-context cleanup for restart replay reconciliation.

        Raw storage remains lossless. This view is used only to reconcile a
        restarted process when the host replays sanitized active context where
        assistant rows may be removed or have internal content stripped.
        """
        sanitized_tail: list[tuple[str, str, str, str]] = []
        for identity in stored_tail:
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            if cleaned_identity is not None:
                sanitized_tail.append(cleaned_identity)
        return sanitized_tail

    def _find_reconciled_cursor_for_store_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_tail: list[tuple[str, str, str, str]],
        *,
        stored_tail_rows: list[Dict[str, Any]] | None = None,
        allow_empty_prefix: bool,
        session_count: int,
        raw_session_count: int,
        read_budget: dict[str, float | int] | None = None,
        active_identities: dict[int, tuple[str, str, str, str]] | None = None,
    ) -> int | None:
        budget = read_budget or self._new_locked_publication_read_budget()
        identity_by_message_id = active_identities or {}
        for msg in messages:
            message_id = id(msg)
            if message_id in identity_by_message_id:
                continue
            self._charge_reconciliation_active_message(msg, read_budget=budget)
            identity_by_message_id[message_id] = self._message_replay_identity(
                msg, read_budget=budget
            )

        def identity_for(msg: Dict[str, Any]) -> tuple[str, str, str, str]:
            return identity_by_message_id[id(msg)]

        persisted_recovery_cache: dict[int, bool] = {}
        durable_replay_cache: dict[int, bool] = {}
        stored_marker_cache: dict[int, bool] = {}

        def persisted_marker_is_recoverable(msg: Dict[str, Any]) -> bool:
            message_id = id(msg)
            if message_id not in persisted_recovery_cache:
                content = normalize_content_value(msg.get("content")) or ""
                persisted_recovery_cache[message_id] = (
                    recover_hermes_persisted_output_with_file_stat(
                        content,
                        read_budget=budget,
                        budget_label="source reconciliation persisted-output proof",
                        max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                        max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
                    )
                    is not None
                )
            return persisted_recovery_cache[message_id]

        def has_durable_replay(msg: Dict[str, Any]) -> bool:
            message_id = id(msg)
            if message_id not in durable_replay_cache:
                durable_replay_cache[message_id] = (
                    self._has_durable_persisted_output_replay_identity(
                        msg, read_budget=budget
                    )
                )
            return durable_replay_cache[message_id]

        sanitized_replay_tail: list[tuple[str, str, str, str]] = []
        stored_cleanup_identities: list[tuple[str, str, str, str] | None] = []
        for identity in stored_tail:
            self._check_reconciliation_deadline(budget)
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            stored_cleanup_identities.append(cleaned_identity)
            if cleaned_identity is not None:
                sanitized_replay_tail.append(cleaned_identity)
        effective_session_count = len(sanitized_replay_tail)
        sanitized_tail_collapsed = len(sanitized_replay_tail) < len(stored_tail)

        raw_suffix_needs_cleanup = [False] * (len(stored_tail) + 1)
        saw_cleanup_difference = False
        for suffix_length in range(1, len(stored_tail) + 1):
            self._check_reconciliation_deadline(budget)
            index = len(stored_tail) - suffix_length
            saw_cleanup_difference = saw_cleanup_difference or (
                stored_cleanup_identities[index] != stored_tail[index]
            )
            raw_suffix_needs_cleanup[suffix_length] = saw_cleanup_difference

        visible_messages: list[Dict[str, Any]] = []
        identity_messages: list[Dict[str, Any]] = []
        visible_identities: list[tuple[str, str, str, str]] = []
        candidate_identities: list[tuple[str, str, str, str]] = []
        visible_count_by_cursor = [0]
        identity_count_by_cursor = [0]
        scaffold_count_by_cursor = [0]
        quarantine_count_by_cursor = [0]
        preserved_objective_count_by_cursor = [0]

        for msg in messages:
            self._check_reconciliation_deadline(budget)
            identity = identity_for(msg)
            is_scaffold = self._is_replayed_context_scaffold_message(msg)
            is_visible = not is_scaffold and not self._matches_ignore_message_patterns(
                msg, read_budget=budget
            )
            text = text_content_for_pattern_matching(msg.get("content")) or ""
            is_quarantined_identity = self._is_quarantined_assistant_replay_identity(identity)
            is_filtered_placeholder = False
            if is_visible:
                is_filtered_placeholder = bool(
                    self._is_volatile_ignored_quarantine_placeholder(msg, text)
                    or self._is_ignored_active_replay_placeholder(msg, text)
                    or (
                        self._compiled_ignore_message_patterns
                        and is_quarantined_identity
                        and self._matches_ignore_message_patterns(
                            msg, stored_row=True, read_budget=budget
                        )
                    )
                )
                visible_messages.append(msg)
                visible_identities.append(identity)
                if not is_filtered_placeholder:
                    identity_messages.append(msg)
                    candidate_identities.append(identity)

            content = normalize_content_value(msg.get("content")) or ""
            has_preserved_objective = (
                str(msg.get("role") or "") != "system"
                and content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
            )
            visible_count_by_cursor.append(len(visible_messages))
            identity_count_by_cursor.append(len(identity_messages))
            scaffold_count_by_cursor.append(
                scaffold_count_by_cursor[-1] + int(is_scaffold)
            )
            quarantine_count_by_cursor.append(
                quarantine_count_by_cursor[-1] + int(is_quarantined_identity)
            )
            preserved_objective_count_by_cursor.append(
                preserved_objective_count_by_cursor[-1] + int(has_preserved_objective)
            )

        raw_matching_lengths = self._store_tail_suffix_matching_prefix_lengths(
            stored_tail,
            candidate_identities,
            read_budget=budget,
        )
        sanitized_matching_lengths = self._store_tail_suffix_matching_prefix_lengths(
            sanitized_replay_tail,
            candidate_identities,
            read_budget=budget,
        )
        visible_raw_matching_lengths = self._store_tail_suffix_matching_prefix_lengths(
            stored_tail,
            visible_identities,
            read_budget=budget,
        )
        visible_sanitized_matching_lengths = self._store_tail_suffix_matching_prefix_lengths(
            sanitized_replay_tail,
            visible_identities,
            read_budget=budget,
        )

        candidate_contents: list[str] = []
        for msg in identity_messages:
            self._check_reconciliation_deadline(budget)
            candidate_contents.append(normalize_content_value(msg.get("content")) or "")
        persisted_marker_flags: list[bool] = []
        for msg, content in zip(identity_messages, candidate_contents):
            self._check_reconciliation_deadline(budget)
            persisted_marker_flags.append(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(content)
            )
        persisted_marker_prefix_counts = [0]
        inline_generation_prefix_counts = [0]
        system_prefix_counts = [0]
        user_prefix_counts = [0]
        for identity, content, is_persisted_marker in zip(
            candidate_identities,
            candidate_contents,
            persisted_marker_flags,
        ):
            self._check_reconciliation_deadline(budget)
            persisted_marker_prefix_counts.append(
                persisted_marker_prefix_counts[-1] + int(is_persisted_marker)
            )
            inline_generation_prefix_counts.append(
                inline_generation_prefix_counts[-1]
                + int(
                    is_persisted_marker
                    and _has_inline_persisted_output_generation_metadata(content)
                )
            )
            system_prefix_counts.append(
                system_prefix_counts[-1] + int(identity[0] == "system")
            )
            user_prefix_counts.append(
                user_prefix_counts[-1] + int(identity[0] == "user")
            )

        first_unrecoverable_marker: int | None = None
        for index, (msg, is_persisted_marker) in enumerate(
            zip(identity_messages, persisted_marker_flags),
            start=1,
        ):
            self._check_reconciliation_deadline(budget)
            if is_persisted_marker and not persisted_marker_is_recoverable(msg):
                first_unrecoverable_marker = index
                break

        durable_full_replay_cache: dict[int, bool] = {}

        def matches_durable_full_replay(prefix_length: int) -> bool:
            if prefix_length not in durable_full_replay_cache:
                if prefix_length != len(stored_tail):
                    durable_full_replay_cache[prefix_length] = False
                else:
                    durable_full_replay_cache[prefix_length] = (
                        self._matches_persisted_output_durable_full_replay(
                            identity_messages[:prefix_length],
                            candidate_identities[:prefix_length],
                            stored_tail,
                            stored_tail_rows,
                            read_budget=budget,
                            durable_cache=durable_replay_cache,
                            stored_marker_cache=stored_marker_cache,
                        )
                    )
            return durable_full_replay_cache[prefix_length]

        active_persisted_marker_flags: list[bool] = []
        for msg in messages:
            self._check_reconciliation_deadline(budget)
            content = normalize_content_value(msg.get("content")) or ""
            active_persisted_marker_flags.append(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(content)
            )

        durable_checked_prefix = 0
        first_durable_marker: int | None = None

        def prefix_has_durable_marker(prefix_length: int) -> bool:
            nonlocal durable_checked_prefix, first_durable_marker
            if first_durable_marker is not None:
                return first_durable_marker <= prefix_length
            while durable_checked_prefix < prefix_length:
                self._check_reconciliation_deadline(budget)
                index = durable_checked_prefix
                durable_checked_prefix += 1
                if active_persisted_marker_flags[index] and has_durable_replay(
                    messages[index]
                ):
                    first_durable_marker = durable_checked_prefix
                    return True
            return False

        generationless_matching_lengths: set[int] | None = None

        def matches_inline_generation_cleanup(prefix_length: int) -> bool:
            nonlocal generationless_matching_lengths
            if generationless_matching_lengths is None:
                generationless_sanitized_tail = []
                for identity in sanitized_replay_tail:
                    self._check_reconciliation_deadline(budget)
                    generationless_sanitized_tail.append(
                        self._strip_inline_persisted_output_generation_identity(identity)
                    )
                generationless_candidate_identities = []
                for identity in candidate_identities:
                    self._check_reconciliation_deadline(budget)
                    generationless_candidate_identities.append(
                        self._strip_inline_persisted_output_generation_identity(identity)
                    )
                generationless_matching_lengths = (
                    self._store_tail_suffix_matching_prefix_lengths(
                        generationless_sanitized_tail,
                        generationless_candidate_identities,
                        read_budget=budget,
                    )
                )
            return prefix_length in generationless_matching_lengths

        dropped_placeholder_count_by_cursor: list[int] | None = None

        def prefix_dropped_quarantine_placeholder(cursor: int) -> bool:
            nonlocal dropped_placeholder_count_by_cursor
            if dropped_placeholder_count_by_cursor is None:
                dropped_placeholder_count_by_cursor = [0]
                for msg in messages:
                    self._check_reconciliation_deadline(budget)
                    text = text_content_for_pattern_matching(msg.get("content")) or ""
                    identity = identity_for(msg)
                    is_dropped = bool(
                        self._is_volatile_ignored_quarantine_placeholder(msg, text)
                        or self._is_ignored_active_replay_placeholder(msg, text)
                        or (
                            self._compiled_ignore_message_patterns
                            and self._is_quarantined_assistant_replay_identity(identity)
                            and self._matches_ignore_message_patterns(
                                msg, stored_row=True, read_budget=budget
                            )
                        )
                    )
                    dropped_placeholder_count_by_cursor.append(
                        dropped_placeholder_count_by_cursor[-1] + int(is_dropped)
                    )
            return dropped_placeholder_count_by_cursor[cursor] > 0

        empty_prefix_cursor: int | None = None
        for cursor in range(len(messages), -1, -1):
            self._check_reconciliation_deadline(budget)
            visible_length = visible_count_by_cursor[cursor]
            prefix_length = identity_count_by_cursor[cursor]
            filtered_candidate_placeholders = prefix_length < visible_length
            candidate_has_scaffold_evidence = scaffold_count_by_cursor[cursor] > 0
            candidate_has_quarantined_replay_evidence = (
                quarantine_count_by_cursor[cursor] > 0
            )
            if prefix_length == 0:
                empty_prefix_cursor = cursor
                if allow_empty_prefix and (
                    not filtered_candidate_placeholders
                    or candidate_has_scaffold_evidence
                    or candidate_has_quarantined_replay_evidence
                ):
                    return cursor
                continue

            matches_sanitized_tail = prefix_length in sanitized_matching_lengths
            matches_raw_tail = prefix_length in raw_matching_lengths
            matches_visible_sanitized_tail = (
                filtered_candidate_placeholders
                and visible_length > 0
                and visible_length in visible_sanitized_matching_lengths
            )
            matches_visible_raw_tail = (
                filtered_candidate_placeholders
                and visible_length > 0
                and visible_length in visible_raw_matching_lengths
            )
            candidate_has_unrecoverable_persisted_marker = (
                first_unrecoverable_marker is not None
                and first_unrecoverable_marker <= prefix_length
            )
            if (
                matches_visible_sanitized_tail or matches_visible_raw_tail
            ) and not candidate_has_unrecoverable_persisted_marker:
                return cursor
            candidate_has_persisted_marker = (
                persisted_marker_prefix_counts[prefix_length] > 0
            )
            matches_durable_persisted_output_full_replay = (
                matches_durable_full_replay(prefix_length)
            )
            matches_inline_generation_cleanup_tail = False
            if candidate_has_unrecoverable_persisted_marker:
                matches_inline_generation_cleanup_tail = (
                    matches_inline_generation_cleanup(prefix_length)
                )
            raw_suffix_needs_cleanup_equivalence = (
                matches_raw_tail
                and raw_suffix_needs_cleanup[prefix_length]
            )
            if (
                not matches_sanitized_tail
                and not matches_raw_tail
                and not matches_inline_generation_cleanup_tail
                and not matches_durable_persisted_output_full_replay
            ):
                continue

            # Matching a stored suffix is not enough evidence by itself.  A
            # gateway restart may provide only newly arrived delta messages; if
            # the first delta happens to repeat the durable tail, treating that
            # row as replay silently loses it.  Only advance the cursor when the
            # incoming prefix proves replay by covering the full durable session.
            # A system prompt is a strong anchor. Older/minimal transcripts can
            # start directly with user/assistant turns, so multi-row full replay
            # is accepted only when active cleanup did not collapse the durable
            # tail; otherwise a fresh delta can repeat the remaining visible
            # suffix and must be preserved.
            candidate_has_system = system_prefix_counts[prefix_length] > 0
            candidate_dropped_quarantine_replay_placeholder = (
                prefix_dropped_quarantine_placeholder(cursor)
            )
            has_quarantined_singleton_replay = (
                matches_sanitized_tail
                and prefix_length == 1
                and effective_session_count == 1
                and self._is_quarantined_assistant_replay_identity(candidate_identities[0])
                and self._is_quarantined_assistant_replay_identity(sanitized_replay_tail[0])
            )
            candidate_singleton_original_content = (
                candidate_contents[0]
                if prefix_length == 1
                else ""
            )
            has_externalized_singleton_replay = (
                matches_raw_tail
                and prefix_length == 1
                and raw_session_count == 1
                and bool(extract_externalized_ref(candidate_singleton_original_content))
                and len(stored_tail) == 1
                and candidate_identities[0] == stored_tail[0]
            )
            has_persisted_marker_singleton_replay = (
                matches_raw_tail
                and not candidate_has_unrecoverable_persisted_marker
                and prefix_length == 1
                and raw_session_count == 1
                and len(stored_tail) == 1
                and candidate_identities[0] == stored_tail[0]
                and candidate_identities[0][0] == "tool"
                and _is_hermes_persisted_output_marker(candidate_singleton_original_content)
            )
            has_durable_persisted_marker_suffix_replay = (
                (matches_sanitized_tail or matches_raw_tail)
                and prefix_has_durable_marker(cursor)
            )
            has_filtered_full_replay = (
                matches_sanitized_tail
                and candidate_dropped_quarantine_replay_placeholder
                and prefix_length >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_generation_cleanup_replay = (
                matches_inline_generation_cleanup_tail
                and candidate_has_unrecoverable_persisted_marker
                and prefix_length >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_persisted_generation_suffix_replay = (
                matches_sanitized_tail
                and inline_generation_prefix_counts[prefix_length] > 0
            )
            if candidate_has_unrecoverable_persisted_marker:
                continue
            has_raw_persisted_marker_exact_replay = (
                candidate_has_persisted_marker
                and not candidate_has_unrecoverable_persisted_marker
                and matches_raw_tail
            )
            has_persisted_marker_specific_replay_evidence = (
                not candidate_has_persisted_marker
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_persisted_marker_singleton_replay
                or has_raw_persisted_marker_exact_replay
            )
            has_effective_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and prefix_length >= effective_session_count
                and (
                    candidate_has_system
                    or (effective_session_count > 1 and not sanitized_tail_collapsed)
                    or has_quarantined_singleton_replay
                    or has_filtered_full_replay
                )
            )

            has_scaffold_evidence = candidate_has_scaffold_evidence
            has_raw_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and not has_scaffold_evidence
                and cursor >= raw_session_count
                and raw_session_count > 1
            )
            has_preserved_objective_scaffold = (
                preserved_objective_count_by_cursor[cursor] > 0
            )
            candidate_suffix_has_user_turn = user_prefix_counts[prefix_length] > 0
            has_scaffold_suffix_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and has_preserved_objective_scaffold
                and not candidate_suffix_has_user_turn
            )
            has_raw_cleanup_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and has_scaffold_evidence
                and cursor < len(messages)
                and prefix_length >= max(1, self._config.fresh_tail_count)
                and raw_suffix_needs_cleanup_equivalence
            )
            if (
                has_effective_full_replay
                or has_externalized_singleton_replay
                or has_persisted_marker_singleton_replay
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_raw_full_replay
                or has_scaffold_suffix_replay
                or has_raw_cleanup_replay
            ):
                return cursor
        return empty_prefix_cursor if allow_empty_prefix else None

    def _record_ingest_reconciliation(
        self,
        *,
        action: str,
        reason: str,
        cursor: int,
        incoming: int,
        session_count: int,
        stored_tail_count: int,
        effective_incoming: int | None = None,
    ) -> None:
        self._last_ingest_reconciliation = {
            "action": action,
            "reason": reason,
            "cursor": cursor,
            "incoming": incoming,
            "session_count": session_count,
            "stored_tail_count": stored_tail_count,
        }
        if effective_incoming is not None:
            self._last_ingest_reconciliation["effective_incoming"] = effective_incoming

    def _effective_replay_identities(
        self,
        messages: List[Dict[str, Any]],
        *,
        active_identities: dict[int, tuple[str, str, str, str]] | None = None,
        read_budget: dict[str, float | int] | None = None,
    ) -> list[tuple[str, str, str, str]]:
        return [
            (
                active_identities[id(msg)]
                if active_identities is not None
                else self._message_replay_identity(msg)
            )
            for msg in messages
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(
                msg, read_budget=read_budget
            )
        ]

    def _is_suspicious_stale_no_overlap_snapshot(
        self,
        incoming_identities: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_head: list[tuple[str, str, str, str]],
    ) -> bool:
        """Return true for short stale snapshots with no durable-tail overlap.

        A restarted gateway can hand LCM a stale, short in-memory snapshot from
        the beginning of a longer session.  When that snapshot has no overlap
        with the durable tail, appending it as a delta creates duplicate rows.
        Fail closed only when the short batch is proven stale by matching the
        contiguous durable-store prefix; singleton no-overlap deltas remain
        ambiguous and are preserved.
        """
        if len(incoming_identities) <= 1:
            return False
        if incoming_identities[0][0] != "system":
            return False
        if not stored_tail or len(incoming_identities) >= len(stored_tail):
            return False
        if set(incoming_identities).intersection(stored_tail):
            return False
        if len(incoming_identities) > len(stored_head):
            return False
        return stored_head[: len(incoming_identities)] == incoming_identities

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
        if not self._session_id or not messages:
            return 0

        try:
            session_count = self._store.get_session_count(self._session_id)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation count failed: %s", exc)
            return 0
        if session_count <= 0:
            placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
            placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            if placeholder_budget and placeholder_ordinals:
                consumed: dict[str, int] = {}
                cursor = 0
                for msg in messages:
                    text = text_content_for_pattern_matching(msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(text)
                    if not digest:
                        break
                    consumed[digest] = consumed.get(digest, 0) + 1
                    ordinal = consumed[digest]
                    remaining = int(placeholder_budget.get(digest, 0) or 0)
                    if remaining <= 0 or ordinal not in placeholder_ordinals.get(digest, set()):
                        break
                    cursor += 1
                if cursor > 0:
                    self._record_ingest_reconciliation(
                        action="advanced cursor",
                        reason="replayed generated placeholders in empty session",
                        cursor=cursor,
                        incoming=len(messages),
                        session_count=session_count,
                        stored_tail_count=0,
                        effective_incoming=cursor,
                    )
                    return cursor
            return 0

        read_budget = self._new_locked_publication_read_budget()
        active_identities: dict[int, tuple[str, str, str, str]] = {}
        for msg in messages:
            self._charge_reconciliation_active_message(
                msg, read_budget=read_budget
            )
            active_identities[id(msg)] = self._message_replay_identity(
                msg, read_budget=read_budget
            )
        stored_identity_cache: dict[int, tuple[str, str, str, str]] = {}

        def stored_identity(row: Dict[str, Any]) -> tuple[str, str, str, str]:
            store_id = int(row.get("store_id") or 0)
            if store_id not in stored_identity_cache:
                stored_identity_cache[store_id] = self._message_replay_identity(
                    row, stored_row=True, read_budget=read_budget
                )
            return stored_identity_cache[store_id]

        tail_limit = min(max(len(messages) * 4, 64), session_count)
        stored_rows = self._bounded_reconciliation_session_rows(
            limit=tail_limit,
            tail=True,
            read_budget=read_budget,
        )
        if not stored_rows:
            return 0
        stored_tail_rows = [
            row
            for row in stored_rows
            if not self._matches_ignore_message_patterns(
                row, stored_row=True, read_budget=read_budget
            )
        ]
        stored_tail = [
            stored_identity(row)
            for row in stored_tail_rows
        ]
        cursor = self._find_reconciled_cursor_for_store_tail(
            messages,
            stored_tail,
            stored_tail_rows=stored_tail_rows,
            allow_empty_prefix=True,
            session_count=len(stored_tail),
            raw_session_count=session_count,
            read_budget=read_budget,
            active_identities=active_identities,
        )
        if cursor is not None and cursor > 0:
            reason = (
                "skipped scaffold-only prefix"
                if not self._effective_replay_identities(
                    messages[:cursor],
                    active_identities=active_identities,
                    read_budget=read_budget,
                )
                else "replayed durable tail"
            )
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason=reason,
                cursor=cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(
                    self._effective_replay_identities(
                        messages,
                        active_identities=active_identities,
                        read_budget=read_budget,
                    )
                ),
            )
            logger.debug(
                "LCM reconciled ingest cursor after existing-session bind: session=%s cursor=%d incoming=%d stored_tail=%d session_count=%d reason=%s",
                self._session_id,
                cursor,
                len(messages),
                len(stored_tail),
                session_count,
                reason,
            )
            return cursor

        incoming_identities = self._effective_replay_identities(
            messages,
            active_identities=active_identities,
            read_budget=read_budget,
        )
        stored_head_rows = self._bounded_reconciliation_session_rows(
            limit=tail_limit,
            tail=False,
            read_budget=read_budget,
        )
        stored_head = [stored_identity(row) for row in stored_head_rows]
        # Stale-snapshot proof uses the raw durable prefix.  Ignore-message
        # filters may suppress noisy rows for tail reconciliation, but filtered
        # history alone must not create replay evidence for skipping a batch.
        incoming_has_unproofed_raw_persisted_marker = any(
            str(msg.get("role") or "") == "tool"
            and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
            and recover_hermes_persisted_output_with_file_stat(
                normalize_content_value(msg.get("content")) or "",
                read_budget=read_budget,
                budget_label="source reconciliation stale-snapshot proof",
                max_nested_depth=_RECONCILIATION_MAX_NESTED_DEPTH,
                max_nested_items=_RECONCILIATION_MAX_NESTED_ITEMS,
            )
            is None
            for msg in messages
        )
        if (
            not incoming_has_unproofed_raw_persisted_marker
            and self._is_suspicious_stale_no_overlap_snapshot(
                incoming_identities,
                stored_tail,
                stored_head,
            )
        ):
            self._record_ingest_reconciliation(
                action="skipped batch",
                reason="skipped stale no-overlap snapshot",
                cursor=len(messages),
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.warning(
                "LCM skipped stale no-overlap snapshot after existing-session bind: session=%s incoming=%d effective_incoming=%d stored_tail=%d session_count=%d",
                self._session_id,
                len(messages),
                len(incoming_identities),
                len(stored_tail),
                session_count,
            )
            return len(messages)

        self._record_ingest_reconciliation(
            action="persisted batch",
            reason="persisted ambiguous delta",
            cursor=0,
            incoming=len(messages),
            session_count=session_count,
            stored_tail_count=len(stored_tail),
            effective_incoming=len(incoming_identities),
        )
        return 0

    def _raw_externalized_placeholder_replay_identity(self, msg: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(msg.get("role") or "unknown"),
            normalize_content_value(msg.get("content")) or "",
            self._stable_tool_calls_identity(msg.get("tool_calls")),
            str(msg.get("tool_call_id") or ""),
        )

    @staticmethod
    def _guard_reconciliation_nested_representation(
        value: Any,
        *,
        read_budget: dict[str, float | int],
    ) -> None:
        """Reject hostile nesting before replay identity can decode a value."""
        if time.monotonic() >= float(read_budget["deadline_at"]):
            raise RuntimeError("source reconciliation deadline exceeded")
        remaining_items = int(read_budget.setdefault("nested_items", 0))
        if isinstance(value, str):
            stripped = value.lstrip()
            if not stripped.startswith(("{", "[")):
                return
            depth = 0
            in_string = False
            escaped = False
            for index, character in enumerate(value):
                if index % 4096 == 0 and time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("source reconciliation deadline exceeded")
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
                    remaining_items += 1
                    if depth > _RECONCILIATION_MAX_NESTED_DEPTH:
                        raise RuntimeError("source reconciliation nested-depth bound exceeded")
                elif character in "]}":
                    depth = max(0, depth - 1)
                elif character in ",:":
                    remaining_items += 1
                if remaining_items > _RECONCILIATION_MAX_NESTED_ITEMS:
                    raise RuntimeError("source reconciliation nested-item bound exceeded")
            read_budget["nested_items"] = remaining_items
            return
        pending = [(value, 0)]
        while pending:
            current, depth = pending.pop()
            if isinstance(current, (dict, list, tuple)):
                remaining_items += 1
                if remaining_items > _RECONCILIATION_MAX_NESTED_ITEMS:
                    raise RuntimeError("source reconciliation nested-item bound exceeded")
                if depth > _RECONCILIATION_MAX_NESTED_DEPTH:
                    raise RuntimeError("source reconciliation nested-depth bound exceeded")
            if isinstance(current, dict):
                pending.extend((key, depth + 1) for key in current)
                pending.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, (list, tuple)):
                pending.extend((item, depth + 1) for item in current)
        read_budget["nested_items"] = remaining_items

    def _charge_reconciliation_active_message(
        self,
        message: Dict[str, Any],
        *,
        read_budget: dict[str, float | int],
    ) -> None:
        """Bound an active row before normalization or externalized recovery."""
        self._guard_reconciliation_nested_representation(
            message, read_budget=read_budget
        )
        remaining = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
        encoded_bytes = 0
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        for index, chunk in enumerate(encoder.iterencode(message)):
            if index % 64 == 0 and time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("source reconciliation deadline exceeded")
            encoded_bytes += len(chunk.encode("utf-8", errors="replace"))
            if encoded_bytes + 32 > remaining:
                raise RuntimeError("source reconciliation serialized byte bound exceeded")
        self._charge_locked_publication_read(
            read_budget,
            rows=1,
            serialized_bytes=encoded_bytes + 32,
            label="source reconciliation",
        )

    def _bounded_reconciliation_session_rows(
        self,
        *,
        limit: int,
        tail: bool,
        read_budget: dict[str, float | int],
    ) -> list[Dict[str, Any]]:
        """Read a durable session head/tail without materializing unchecked fields."""
        if limit <= 0:
            return []
        text_fields = (
            "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "conversation_id",
        )
        field_limit = min(
            _RECONCILIATION_MAX_FIELD_BYTES,
            int(read_budget["max_bytes"]),
        )
        length_exprs = [
            f"COALESCE(length(CAST({field} AS BLOB)), 0)"
            for field in text_fields
        ]
        direction = "DESC" if tail else "ASC"
        cursor = (1 << 63) - 1 if tail else 0
        rows_out: list[Dict[str, Any]] = []
        while len(rows_out) < int(limit):
            self._check_reconciliation_deadline(read_budget)
            remaining_rows = int(read_budget["max_rows"]) - int(read_budget["rows"])
            remaining_bytes = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
            if remaining_rows <= 0:
                raise RuntimeError("source reconciliation row bound exceeded")
            if remaining_bytes <= 128:
                raise RuntimeError("source reconciliation serialized byte bound exceeded")
            page_limit = min(
                _RECONCILIATION_QUERY_BATCH,
                int(limit) - len(rows_out),
                remaining_rows + 1,
            )
            comparison = "<" if tail else ">"
            metadata_rows = self._store._conn.execute(
                f"""SELECT
                           CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                           CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                           CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                           CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                           {', '.join(length_exprs)},
                           {', '.join(f"typeof({field}) IN ('text', 'null')" for field in text_fields)}
                    FROM messages
                    WHERE session_id = ? AND store_id {comparison} ?
                    ORDER BY store_id {direction} LIMIT ?""",
                (self._session_id, cursor, page_limit),
            ).fetchall()
            if not metadata_rows:
                break
            for metadata in metadata_rows:
                self._check_reconciliation_deadline(read_budget)
                if any(value is None for value in metadata[:4]) or not all(
                    bool(value) for value in metadata[12:20]
                ):
                    raise RuntimeError("source reconciliation field type bound exceeded")
                encoded_lengths = [int(value or 0) for value in metadata[4:12]]
                row_bytes = sum(encoded_lengths) + 128
                if any(length > field_limit for length in encoded_lengths):
                    raise RuntimeError("source reconciliation field byte bound exceeded")
                if row_bytes > remaining_bytes:
                    raise RuntimeError("source reconciliation serialized byte bound exceeded")
                self._charge_locked_publication_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=row_bytes,
                    label="source reconciliation",
                )
                guarded_fields = [
                    f"CASE WHEN {field} IS NULL THEN NULL "
                    f"WHEN typeof({field}) = 'text' AND {length_exprs[index]} = ? "
                    f"AND {length_exprs[index]} <= {field_limit} "
                    f"THEN substr(CAST({field} AS TEXT), 1, {field_limit + 1}) END"
                    for index, field in enumerate(text_fields)
                ]
                payload = self._store._conn.execute(
                    f"""SELECT
                               CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                               {', '.join(guarded_fields[:7])},
                               CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                               CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                               CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                               {guarded_fields[7]},
                               {', '.join(length_exprs)},
                               {', '.join(f"typeof({field}) IN ('text', 'null')" for field in text_fields)}
                        FROM messages
                        WHERE session_id = ? AND store_id = ?
                          AND timestamp IS ? AND token_estimate IS ? AND pinned IS ?
                        LIMIT 1""",
                    (
                        *encoded_lengths,
                        self._session_id,
                        int(metadata[0]),
                        metadata[1], metadata[2], metadata[3],
                    ),
                ).fetchone()
                if payload is None or any(payload[index] is None for index in (0, 8, 9, 10)):
                    raise RuntimeError("source reconciliation stored row changed during bounded read")
                if [int(value or 0) for value in payload[12:20]] != encoded_lengths or not all(
                    bool(value) for value in payload[20:28]
                ):
                    raise RuntimeError("source reconciliation stored row changed during bounded read")
                item = dict(zip(
                    (
                        "store_id", "session_id", "source", "role", "content",
                        "tool_call_id", "tool_calls", "tool_name", "timestamp",
                        "token_estimate", "pinned", "conversation_id",
                    ),
                    payload[:12],
                ))
                for value in item.values():
                    self._guard_reconciliation_nested_representation(
                        value, read_budget=read_budget
                    )
                if item.get("tool_calls"):
                    try:
                        item["tool_calls"] = json.loads(item["tool_calls"])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                rows_out.append(item)
                cursor = int(metadata[0])
                remaining_bytes = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
            if len(metadata_rows) < page_limit:
                break
        if tail:
            rows_out.reverse()
        return rows_out

    def _bounded_reconciliation_candidates(
        self,
        *,
        read_budget: dict[str, float | int],
    ) -> tuple[
        list[int],
        list[tuple[Any, ...]],
        list[Optional[tuple[Any, ...]]],
        list[tuple[str, str, str, str]],
        dict[tuple[Any, ...], int],
        dict[tuple[Any, ...], int],
    ]:
        """Incrementally reduce SQL-bounded rows to compact replay identities."""
        fields = ("session_id", "role", "content", "tool_call_id", "tool_calls")
        field_limit = min(
            _RECONCILIATION_MAX_FIELD_BYTES,
            int(read_budget["max_bytes"]),
        )
        store_ids: list[int] = []
        identities: list[tuple[Any, ...]] = []
        cleanup_identities: list[Optional[tuple[Any, ...]]] = []
        raw_identities: list[tuple[str, str, str, str]] = []
        identity_counts: dict[tuple[Any, ...], int] = {}
        cleanup_counts: dict[tuple[Any, ...], int] = {}
        last_store_id = int(self._last_compacted_store_id or 0)
        length_aliases = [f"{field}_bytes" for field in fields]
        length_exprs = [
            f"COALESCE(length(CAST({field} AS BLOB)), 0) AS {alias}"
            for field, alias in zip(fields, length_aliases)
        ]
        row_bytes_expr = " + ".join(length_aliases) + " + 64"
        type_guard = " AND ".join(
            f"typeof(m.{field}) IN ('text', 'null')" for field in fields
        )
        length_guard = " AND ".join(
            f"b.{alias} <= {field_limit}" for alias in length_aliases
        )
        bounded_fields = [
            f"CASE WHEN b.cumulative_bytes <= ? AND {length_guard} AND {type_guard} "
            f"THEN CASE WHEN m.{field} IS NULL THEN NULL ELSE "
            f"substr(CAST(m.{field} AS TEXT), 1, {field_limit + 1}) END END"
            for field in fields
        ]
        while True:
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("source reconciliation deadline exceeded")
            remaining_rows = int(read_budget["max_rows"]) - int(read_budget["rows"])
            remaining_bytes = int(read_budget["max_bytes"]) - int(read_budget["bytes"])
            if remaining_rows <= 0:
                raise RuntimeError("source reconciliation row bound exceeded")
            if remaining_bytes <= 64:
                raise RuntimeError("source reconciliation serialized byte bound exceeded")
            page_limit = min(_RECONCILIATION_QUERY_BATCH, remaining_rows + 1)
            rows = self._store._conn.execute(
                f"""WITH candidate_lengths AS (
                           SELECT store_id, {', '.join(length_exprs)}
                           FROM messages
                           WHERE session_id = ? AND store_id > ?
                           ORDER BY store_id LIMIT ?
                       ), budgeted AS (
                           SELECT *, ({row_bytes_expr}) AS row_bytes,
                                  SUM({row_bytes_expr}) OVER (ORDER BY store_id)
                                      AS cumulative_bytes
                           FROM candidate_lengths
                       )
                       SELECT b.store_id, {', '.join(bounded_fields)},
                              {', '.join(f'b.{alias}' for alias in length_aliases)},
                              b.row_bytes,
                              CASE WHEN b.cumulative_bytes <= ?
                                         AND {length_guard} AND {type_guard}
                                   THEN 1 ELSE 0 END
                       FROM budgeted AS b
                       JOIN messages AS m ON m.store_id = b.store_id
                       ORDER BY b.store_id LIMIT ?""",
                (
                    self._session_id,
                    last_store_id,
                    page_limit,
                    *(remaining_bytes for _ in fields),
                    remaining_bytes,
                    page_limit,
                ),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                if time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("source reconciliation deadline exceeded")
                if not bool(row[-1]):
                    if int(row[-2] or 0) > remaining_bytes:
                        raise RuntimeError(
                            "source reconciliation serialized byte bound exceeded"
                        )
                    raise RuntimeError("source reconciliation field byte bound exceeded")
                row_bytes = int(row[-2] or 0)
                self._charge_locked_publication_read(
                    read_budget,
                    rows=1,
                    serialized_bytes=row_bytes,
                    label="source reconciliation",
                )
                stored = dict(zip(fields, row[1:1 + len(fields)]))
                for value in stored.values():
                    self._guard_reconciliation_nested_representation(
                        value, read_budget=read_budget
                    )
                raw_tool_calls = stored.get("tool_calls")
                if isinstance(raw_tool_calls, str):
                    try:
                        stored["tool_calls"] = json.loads(raw_tool_calls)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        # Preserve legacy malformed text exactly; the replay
                        # identity helper deliberately treats it as a scalar.
                        pass
                identity = self._message_replay_identity(
                    stored, stored_row=True, read_budget=read_budget
                )
                if time.monotonic() >= float(read_budget["deadline_at"]):
                    raise RuntimeError("source reconciliation deadline exceeded")
                cleanup = self._active_cleanup_replay_identity(identity)
                raw_identity = self._raw_externalized_placeholder_replay_identity(stored)
                store_ids.append(int(row[0]))
                identities.append(identity)
                cleanup_identities.append(cleanup)
                raw_identities.append(raw_identity)
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
                if cleanup is not None:
                    cleanup_counts[cleanup] = cleanup_counts.get(cleanup, 0) + 1
                last_store_id = int(row[0])
            if len(rows) < page_limit:
                break
        return (
            store_ids,
            identities,
            cleanup_identities,
            raw_identities,
            identity_counts,
            cleanup_counts,
        )

    def _get_store_id_map_for_messages(self, messages: List[Dict[str, Any]]) -> dict[int, int]:
        """Map current raw message objects back to store_ids in stable order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.  When
        active context has more occurrences of an identical replay identity than
        the store has, the surplus earliest active occurrences are treated as
        synthetic/carry-over and left unmapped so they cannot steal later stored
        literal copies with the same content.
        """
        read_budget = self._new_locked_publication_read_budget()
        active_identity_counts: dict[tuple[Any, ...], int] = {}
        active_identities: dict[int, tuple[Any, ...]] = {}
        active_cleanup_identities: dict[int, tuple[Any, ...] | None] = {}
        active_raw_identities: dict[int, tuple[str, str, str, str] | None] = {}
        for msg in messages:
            self._charge_reconciliation_active_message(msg, read_budget=read_budget)
            identity = self._message_replay_identity(msg, read_budget=read_budget)
            message_id = id(msg)
            active_identities[message_id] = identity
            active_cleanup_identities[message_id] = self._active_cleanup_replay_identity(identity)
            msg_content = normalize_content_value(msg.get("content")) or ""
            raw_identity = None
            if (
                msg.get("store_id") is None
                and self._content_has_externalized_placeholder_ref(msg_content)
            ):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
            active_raw_identities[message_id] = raw_identity
            active_identity_counts[identity] = active_identity_counts.get(identity, 0) + 1
        (
            candidate_store_ids,
            stored_identities,
            stored_cleanup_identities,
            stored_raw_placeholder_identities,
            stored_identity_counts,
            stored_cleanup_identity_counts,
        ) = self._bounded_reconciliation_candidates(read_budget=read_budget)

        active_surplus_skips: dict[tuple[Any, ...], int] = {}
        generated_surplus_skip_message_ids: set[int] = set()
        generated_placeholder_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        generated_by_identity: dict[tuple[Any, ...], list[int]] = {}
        for msg in messages:
            message_id = id(msg)
            if message_id in generated_placeholder_message_ids:
                generated_by_identity.setdefault(
                    active_identities[message_id], []
                ).append(message_id)
        for identity, active_count in active_identity_counts.items():
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("source reconciliation deadline exceeded")
            wanted_cleanup_identity = self._active_cleanup_replay_identity(identity)
            stored_exact = stored_identity_counts.get(identity, 0)
            stored_cleanup = 0
            if wanted_cleanup_identity is not None:
                stored_cleanup = stored_cleanup_identity_counts.get(wanted_cleanup_identity, 0)
            stored_available = max(stored_exact, stored_cleanup)
            if active_count > stored_available:
                surplus_count = active_count - stored_available
                generated_ids = generated_by_identity.get(identity, [])
                generated_surplus_skip_message_ids.update(
                    generated_ids[:surplus_count]
                )
                surplus_count -= min(surplus_count, len(generated_ids))
                if surplus_count > 0:
                    active_surplus_skips[identity] = surplus_count

        placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        for msg in messages:
            raw_identity = active_raw_identities[id(msg)]
            if raw_identity is not None:
                placeholder_identity_counts[raw_identity] = placeholder_identity_counts.get(raw_identity, 0) + 1
        self._current_compress_placeholder_identity_counts = placeholder_identity_counts

        exact_positions: dict[tuple[Any, ...], list[int]] = {}
        cleanup_positions: dict[tuple[Any, ...], list[int]] = {}
        raw_positions: dict[tuple[str, str, str, str], list[int]] = {}
        for index, identity in enumerate(stored_identities):
            exact_positions.setdefault(identity, []).append(index)
            cleanup = stored_cleanup_identities[index]
            if cleanup is not None:
                cleanup_positions.setdefault(cleanup, []).append(index)
            raw_positions.setdefault(
                stored_raw_placeholder_identities[index], []
            ).append(index)

        position_offsets: dict[tuple[str, tuple[Any, ...]], int] = {}

        def first_at_or_after(
            kind: str,
            identity: tuple[Any, ...],
            positions: list[int] | None,
            start: int,
        ) -> int | None:
            if not positions:
                return None
            key = (kind, identity)
            offset = position_offsets.get(key, 0)
            while offset < len(positions) and positions[offset] < start:
                offset += 1
            position_offsets[key] = offset
            return positions[offset] if offset < len(positions) else None

        def find_identity_match_index(message_id: int, start: int) -> int | None:
            identity = active_identities[message_id]
            exact = first_at_or_after(
                "exact", identity, exact_positions.get(identity), start
            )
            cleanup_identity = active_cleanup_identities[message_id]
            cleanup = (
                first_at_or_after(
                    "cleanup",
                    cleanup_identity,
                    cleanup_positions.get(cleanup_identity),
                    start,
                )
                if cleanup_identity is not None
                else None
            )
            candidates = [value for value in (exact, cleanup) if value is not None]
            return min(candidates) if candidates else None

        ids_by_message_id: dict[int, int] = {}
        store_idx = 0
        for msg in messages:
            if time.monotonic() >= float(read_budget["deadline_at"]):
                raise RuntimeError("source reconciliation deadline exceeded")
            message_id = id(msg)
            raw_identity = active_raw_identities[message_id]
            if raw_identity is not None:
                raw_match_idx = first_at_or_after(
                    "raw", raw_identity, raw_positions.get(raw_identity), store_idx
                )
                if raw_match_idx is not None:
                    ids_by_message_id[message_id] = candidate_store_ids[raw_match_idx]
                    store_idx = raw_match_idx + 1
                    continue
            message_identity = active_identities[message_id]
            if message_id in generated_surplus_skip_message_ids:
                continue
            surplus = active_surplus_skips.get(message_identity, 0)
            if surplus > 0:
                active_surplus_skips[message_identity] = surplus - 1
                continue
            match_idx = find_identity_match_index(message_id, store_idx)
            if match_idx is not None:
                ids_by_message_id[message_id] = candidate_store_ids[match_idx]
                store_idx = match_idx + 1

        return ids_by_message_id
