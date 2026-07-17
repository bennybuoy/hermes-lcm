"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import codecs
import bisect
import hashlib
import json
import logging
import math
import multiprocessing
import os
import re
import secrets
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from .externalize import (
    extract_externalized_ref,
    extract_externalized_refs,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    load_externalized_payload,
)
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
)
from .dag import (
    MAX_SOURCE_IDS_JSON_CHARS,
    MAX_SOURCE_IDS_PER_NODE,
    build_nodes_fts_spec,
    decode_source_ids,
)
from .db_bootstrap import check_external_content_fts_integrity, inspect_lcm_schema_health
from .extraction import sanitize_pre_compaction_content
from .ingest_protection import (
    SensitiveOutputSpan,
    externalized_payload_stats,
    extract_ingest_externalized_refs,
    restore_ingest_payload_placeholders,
    redact_sensitive_text,
    redact_sensitive_output_text,
    sensitive_output_spans,
    scan_externalized_payload_integrity,
    scan_sqlite_payload_risks,
    sensitive_pattern_status,
)
from .model_routing import apply_lcm_model_route
from .presets import preset_status_payload
from .search_query import AGE_DECAY_RATE, normalize_search_sort
from .session_patterns import build_session_match_keys, compile_session_pattern
from .store import build_message_fts_spec

if TYPE_CHECKING:
    from .engine import LCMEngine


logger = logging.getLogger(__name__)

_LCM_GREP_EXTERNALIZED_DEFAULT_FILES = 100
_LCM_GREP_EXTERNALIZED_MAX_FILES = 500
_LCM_GREP_EXTERNALIZED_DEFAULT_CHARS = 65_536
_LCM_GREP_EXTERNALIZED_MAX_CHARS = 1_000_000
_LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES = 4_000_000
_LCM_GREP_REGEX_FILE_DEADLINE_SECONDS = 0.075
_LCM_GREP_REGEX_OPERATION_DEADLINE_SECONDS = 1.0
_LCM_GREP_REGEX_MAX_PATTERN_CHARS = 2_000
_LCM_GREP_QUERY_MAX_CHARS = 2_000
_LCM_GREP_QUERY_MAX_TOKENS = 512
_LCM_GREP_SCOPE_MAX_CHARS = 32
_LCM_GREP_SESSION_ID_MAX_CHARS = 512
_LCM_GREP_SOURCE_MAX_CHARS = 512
_LCM_GREP_CONVERSATION_ID_MAX_CHARS = 512
_LCM_GREP_REF_MAX_CHARS = 512
_LCM_GREP_ROLE_MAX_CHARS = 128
_LCM_GREP_SORT_MAX_CHARS = 32
_LCM_GREP_TIMESTAMP_MAX_CHARS = 128
_LCM_GREP_OPERATION_MAX_ROWS = 1_000
_LCM_GREP_OPERATION_MAX_BYTES = 2 * 1024 * 1024
_LCM_GREP_OPERATION_DEADLINE_SECONDS = 1.0
_CROSS_SESSION_EXPANSION_GUARD = threading.Lock()
_ACTIVE_CROSS_SESSION_EXPANSIONS: set[str] = set()
_CROSS_SESSION_MAX_SESSIONS = 10
_CROSS_SESSION_MAX_SUMMARIES_PER_SESSION = 20
_CROSS_SESSION_MAX_RESULTS = 20
_CROSS_SESSION_MAX_ANSWER_TOKENS = 8_192
_CROSS_SESSION_MAX_CONTEXT_TOKENS = 65_536
_CROSS_SESSION_MAX_DEADLINE_MS = 120_000
_CROSS_SESSION_METADATA_MAX_CHARS = 2_048
_CROSS_SESSION_METADATA_MAX_TOKENS = 128
_CROSS_SESSION_CONTEXT_MAX_DEPTH = 8
_CROSS_SESSION_CONTEXT_MAX_ITEMS = 200
_CROSS_SESSION_CONTENT_FIELDS = frozenset({"content", "summary", "transcript_content", "snippet"})
_CROSS_SESSION_AUTH_MAX_CANDIDATES = 256
_CROSS_SESSION_AUTH_MAX_NODES = 1_600
_CROSS_SESSION_AUTH_MAX_MESSAGES = 6_400
_CROSS_SESSION_AUTH_MAX_EDGES = 6_400
_CROSS_SESSION_AUTH_MAX_DEPTH = 64
_CROSS_SESSION_AUTH_QUERY_BATCH = 400
_CROSS_SESSION_AUTH_SESSION_ID_CHARS = 256
_CROSS_SESSION_AUTH_SOURCE_TYPE_CHARS = 32
_CROSS_SESSION_AUTH_MAX_MATERIALIZED_BYTES = 4 * 1024 * 1024
_CURRENT_SESSION_EXPAND_MAX_TOKENS = 65_536
_CURRENT_SESSION_EXPAND_MAX_SOURCES = 200
_CURRENT_SESSION_EXPAND_MAX_CHARS = 100_000
_MANDATORY_REDACTION_LOOKAHEAD_CHARS = 8_192
_MANDATORY_REDACTION_CHARS_PER_TOKEN = 16
_EXPAND_BOUNDARY_SCAN_CHARS = 8_192
_EXPAND_BOUNDARY_SCAN_MAX_CHUNKS = 128
_EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS = 0.25
_expand_scan_now = time.monotonic
_BOUNDARY_PRIVATE_KEY_BEGIN_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE
)
_BOUNDARY_PRIVATE_KEY_END_RE = re.compile(
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE
)
_BOUNDARY_STANDALONE_CREDENTIAL_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]*|"
    r"(?:AKIA|ASIA)[A-Z0-9]*|"
    r"gh[pousr]_[A-Za-z0-9]*|github_pat_[A-Za-z0-9_]*|"
    r"glpat-[A-Za-z0-9_-]*|AIza[A-Za-z0-9_-]*|"
    r"sk_live_[A-Za-z0-9]*|xox[baprs]-[A-Za-z0-9-]*"
    r")\Z",
    re.IGNORECASE,
)
_BOUNDARY_STANDALONE_START_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-|svcacct-)?|"
    r"(?:AKIA|ASIA)|"
    r"gh[pousr]_|github_pat_|glpat-|AIza|sk_live_|xox[baprs]-|"
    r"Bearer\s+"
    r")",
    re.IGNORECASE,
)
_BOUNDARY_STANDALONE_BODY_RE = re.compile(r"[A-Za-z0-9._~+/=-]")
_BOUNDARY_ASSIGNMENT_START_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|"
    r"client[_-]?secret|password|passwd|pwd|passphrase)"
    r"(?![A-Za-z0-9_-])\s*(?:\\?[\"'])?\s*[:=]\s*"
    r"(?:(?P<escaped_quote>\\[\"'])|(?P<quote>[\"'])|"
    r"(?P<unquoted>[^\r\n,;\"'\]}]))",
    re.IGNORECASE,
)

_BOUNDARY_UNQUOTED_TERMINATORS = frozenset("\r\n,;}]")
_BOUNDARY_ESCAPED_QUOTE_TERMINATORS = frozenset("\r\n,}]")


def _is_unquoted_terminator(read_chunk, absolute_offset: int, char: str) -> bool:
    """Recognize only record structure, including the legacy ``::`` fence."""
    if char in _BOUNDARY_UNQUOTED_TERMINATORS:
        return True
    return char == ":" and read_chunk(absolute_offset, 2).startswith("::")


def _escaped_quote_has_safe_terminator(
    read_chunk,
    after_quote_offset: int,
    total_chars: int,
    *,
    allow_eof: bool = False,
) -> bool:
    """Require JSON/record structure after a raw ``\"`` value delimiter."""
    following = read_chunk(after_quote_offset, 128)
    index = 0
    while index < len(following) and following[index] in " \t":
        index += 1
    if index < len(following):
        return following[index] in _BOUNDARY_ESCAPED_QUOTE_TERMINATORS
    return (
        allow_eof
        and after_quote_offset >= total_chars
        and not following
    )


def _first_unescaped_quote(text: str, quote: str | None = None) -> tuple[int, str] | None:
    for index, char in enumerate(text):
        if char not in {'"', "'"} or (quote is not None and char != quote):
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index, char
    return None


def _credential_mode_at_offset(
    read_chunk,
    scan_start: int,
    requested_offset: int,
    *,
    initial_mode: str = "normal",
    initial_quote: str | None = None,
    initial_quote_backslashes: int = 0,
    max_chunks: int = _EXPAND_BOUNDARY_SCAN_MAX_CHUNKS,
    deadline: float = float("inf"),
) -> dict[str, Any]:
    """Stream credential syntax from a proven boundary to one raw offset."""
    mode = initial_mode
    quote = initial_quote
    quote_backslashes = max(0, int(initial_quote_backslashes))
    cursor = scan_start
    chunks_used = 0

    while cursor < requested_offset:
        if chunks_used >= max_chunks or _expand_scan_now() >= deadline:
            return {
                "complete": False,
                "offset": cursor,
                "mode": mode,
                "quote": quote,
                "quote_backslashes": quote_backslashes,
                "chunks_used": chunks_used,
            }
        primary_chars = min(
            _EXPAND_BOUNDARY_SCAN_CHARS, requested_offset - cursor
        )
        # Normal-mode openers can straddle a chunk edge. The bounded lookahead
        # is inspected but never advances the state beyond requested_offset.
        data = read_chunk(cursor, primary_chars + 512)
        chunks_used += 1
        if not data:
            return {
                "complete": False,
                "offset": cursor,
                "mode": mode,
                "quote": quote,
                "quote_backslashes": quote_backslashes,
                "chunks_used": chunks_used,
            }
        primary_chars = min(primary_chars, len(data))
        advance_chars = primary_chars
        index = 0
        while index < primary_chars:
            if mode == "unknown":
                # There is no sound way to infer an opening quote or an
                # unquoted/PEM grammar from arbitrary body bytes. Keep the
                # state unknown; the caller will fail closed.
                index = primary_chars
                continue
            if mode == "normal":
                pem_begin = (
                    _BOUNDARY_PRIVATE_KEY_BEGIN_RE.match(data, index)
                    if data[index] == "-"
                    else None
                )
                lowered_start = data[index].lower()
                assignment = (
                    _BOUNDARY_ASSIGNMENT_START_RE.match(data, index)
                    if lowered_start in {"a", "s", "c", "p"}
                    else None
                )
                standalone = (
                    _BOUNDARY_STANDALONE_START_RE.match(data, index)
                    if lowered_start in {"s", "a", "g", "x", "b"}
                    else None
                )
                if pem_begin is not None:
                    mode = "pem"
                    index = pem_begin.end()
                    advance_chars = max(
                        advance_chars,
                        min(index, requested_offset - cursor),
                    )
                    continue
                if assignment is not None:
                    if assignment.group("escaped_quote") is not None:
                        mode = "escaped_quote"
                        quote = assignment.group("escaped_quote")[-1]
                        quote_backslashes = 0
                        index = assignment.end()
                    elif assignment.group("quote") is not None:
                        mode = "quote"
                        quote = assignment.group("quote")
                        quote_backslashes = 0
                        index = assignment.end()
                    else:
                        mode = "token"
                        index = assignment.start("unquoted")
                    advance_chars = max(
                        advance_chars,
                        min(assignment.end(), requested_offset - cursor),
                    )
                    continue
                if standalone is not None:
                    mode = "standalone"
                    index = standalone.end()
                    advance_chars = max(
                        advance_chars,
                        min(standalone.end(), requested_offset - cursor),
                    )
                    continue
                index += 1
                continue
            if mode == "quote":
                char = data[index]
                if char == "\\":
                    quote_backslashes += 1
                else:
                    if char == quote and quote_backslashes % 2 == 0:
                        mode = "normal"
                        quote = None
                    quote_backslashes = 0
                index += 1
                continue
            if mode == "escaped_quote":
                char = data[index]
                if char == "\\":
                    quote_backslashes += 1
                else:
                    if (
                        char == quote
                        and quote_backslashes == 1
                        and _escaped_quote_has_safe_terminator(
                            read_chunk,
                            cursor + index + 1,
                            requested_offset,
                        )
                    ):
                        mode = "normal"
                        quote = None
                    quote_backslashes = 0
                index += 1
                continue
            if mode == "token":
                if _is_unquoted_terminator(
                    read_chunk,
                    cursor + index,
                    data[index],
                ):
                    mode = "normal"
                    # Reprocess the delimiter: it may begin another expression.
                    continue
                index += 1
                continue
            if mode == "standalone":
                if _BOUNDARY_STANDALONE_BODY_RE.fullmatch(data[index]) is None:
                    mode = "normal"
                    continue
                index += 1
                continue
            if mode == "pem":
                pem_end = (
                    _BOUNDARY_PRIVATE_KEY_END_RE.match(data, index)
                    if data[index] == "-"
                    else None
                )
                if pem_end is not None:
                    if cursor + pem_end.end() <= requested_offset:
                        mode = "normal"
                        index = pem_end.end()
                        advance_chars = max(advance_chars, index)
                    else:
                        # The cursor itself lies inside the terminator. It is
                        # still part of the sensitive PEM span.
                        index = primary_chars
                else:
                    index += 1
                continue
        cursor += advance_chars
        if advance_chars <= 0:
            return {
                "complete": False,
                "offset": cursor,
                "mode": mode,
                "quote": quote,
                "quote_backslashes": quote_backslashes,
                "chunks_used": chunks_used,
            }
    if mode == "token":
        boundary_char = read_chunk(requested_offset, 1)
        if boundary_char and _is_unquoted_terminator(
            read_chunk,
            requested_offset,
            boundary_char[0],
        ):
            # The requested cursor is exactly on the token terminator. It is a
            # safe page boundary, but the benign delimiter still belongs to
            # the caller's lossless output and must not be consumed here.
            mode = "normal"
    elif mode == "standalone":
        boundary_char = read_chunk(requested_offset, 1)
        if boundary_char and _BOUNDARY_STANDALONE_BODY_RE.fullmatch(boundary_char[0]) is None:
            mode = "normal"
    return {
        "complete": True,
        "offset": cursor,
        "mode": mode,
        "quote": quote,
        "quote_backslashes": quote_backslashes,
        "chunks_used": chunks_used,
    }


def _scan_raw_credential_boundary(
    read_chunk,
    total_chars: int,
    requested_offset: int,
    *,
    checkpoint: dict[str, Any] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Classify/skip a credential around one raw cursor using bounded chunks.

    The scanner never accumulates the row. It streams compact syntax state from
    the row's proof boundary, then scans forward from an unsafe cursor for a
    conservative terminator. A credential longer than one operation advances
    only to another fail-closed scan cursor, so benign suffix data is eventually
    reachable without ever exposing an intermediate body fragment.
    """
    total = max(0, int(total_chars))
    offset = min(max(0, int(requested_offset)), total)
    scan_deadline = (
        float(deadline)
        if deadline is not None
        else _expand_scan_now() + _EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS
    )
    checkpoint = checkpoint or {}
    checkpoint_offset = min(
        offset, max(0, int(checkpoint.get("offset", 0) or 0))
    )
    checkpoint_mode = str(checkpoint.get("mode") or "normal")
    if checkpoint_mode not in {
        "normal", "quote", "escaped_quote", "token", "standalone", "pem", "unknown"
    }:
        checkpoint_mode = "unknown"
    checkpoint_quote = checkpoint.get("quote")
    if checkpoint_quote not in {None, '"', "'"}:
        checkpoint_quote = None
        checkpoint_mode = "unknown"
    checkpoint_backslashes = max(
        0, int(checkpoint.get("quote_backslashes", 0) or 0)
    )
    if offset >= total:
        return {
            "safe_content_offset": offset,
            "checkpoint_offset": offset,
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_quote": checkpoint_quote,
            "checkpoint_quote_backslashes": checkpoint_backslashes,
        }

    mode: str | None = checkpoint_mode
    quote: str | None = checkpoint_quote
    quote_backslashes = checkpoint_backslashes
    unsafe = checkpoint_mode != "normal"
    safe_prefix = ""
    forward_start = offset

    # Stream from the only universal proof boundary: the start of the row.
    # State stays bounded to a few scalars while SQLite returns bounded chunks;
    # no whole row is allocated.  Starting at a fixed lookbehind would make a
    # wholly benign deep offset indistinguishable from a credential body.
    chunks_used = 0
    if offset > checkpoint_offset:
        state = _credential_mode_at_offset(
            read_chunk,
            checkpoint_offset,
            offset,
            initial_mode=checkpoint_mode,
            initial_quote=checkpoint_quote,
            initial_quote_backslashes=checkpoint_backslashes,
            max_chunks=_EXPAND_BOUNDARY_SCAN_MAX_CHUNKS,
            deadline=scan_deadline,
        )
        chunks_used = int(state["chunks_used"])
        if not state["complete"]:
            progress = min(total, max(checkpoint_offset, int(state["offset"])))
            return {
                "safe_content_offset": progress,
                "boundary_redacted": True,
                "boundary_pending": True,
                "boundary_safe_prefix": "",
                "checkpoint_offset": progress,
                "checkpoint_mode": state["mode"],
                "checkpoint_quote": state["quote"],
                "checkpoint_quote_backslashes": state["quote_backslashes"],
            }
        mode = str(state["mode"])
        quote = state["quote"]
        quote_backslashes = int(state["quote_backslashes"])
        unsafe = mode != "normal"

    # Probe an opener at the requested cursor only after a deep cursor has been
    # reached from its proven byte/lexer checkpoint. This prevents a nominally
    # bounded probe from forcing a UTF-8 prefix traversal.
    if not unsafe:
        forward_prefix = read_chunk(offset, 512)
        assignment_at_cursor = _BOUNDARY_ASSIGNMENT_START_RE.match(forward_prefix)
        pem_at_cursor = _BOUNDARY_PRIVATE_KEY_BEGIN_RE.match(forward_prefix)
        standalone_at_cursor = _BOUNDARY_STANDALONE_START_RE.match(forward_prefix)
        if assignment_at_cursor is not None:
            unsafe = True
            if assignment_at_cursor.group("escaped_quote") is not None:
                mode = "escaped_quote"
                quote = assignment_at_cursor.group("escaped_quote")[-1]
                forward_start = offset + assignment_at_cursor.end()
                safe_prefix = forward_prefix[:assignment_at_cursor.end()]
            elif assignment_at_cursor.group("quote") is not None:
                mode = "quote"
                quote = assignment_at_cursor.group("quote")
                forward_start = offset + assignment_at_cursor.end()
                safe_prefix = forward_prefix[:assignment_at_cursor.end()]
            else:
                mode = "token"
                forward_start = offset + assignment_at_cursor.start("unquoted")
                safe_prefix = forward_prefix[:assignment_at_cursor.start("unquoted")]
        elif pem_at_cursor is not None:
            unsafe = True
            mode = "pem"
            forward_start = offset + pem_at_cursor.end()
        elif standalone_at_cursor is not None:
            unsafe = True
            mode = "standalone"
            forward_start = offset + standalone_at_cursor.end()

    if not unsafe:
        return {
            "safe_content_offset": offset,
            "checkpoint_offset": offset,
            "checkpoint_mode": "normal",
            "checkpoint_quote": None,
            "checkpoint_quote_backslashes": 0,
        }

    cursor = forward_start
    overlap = ""
    quote_backslashes = locals().get("quote_backslashes", 0)
    remaining_chunks = max(0, _EXPAND_BOUNDARY_SCAN_MAX_CHUNKS - chunks_used)
    for _ in range(remaining_chunks):
        if _expand_scan_now() >= scan_deadline:
            break
        if cursor >= total:
            return {
                "safe_content_offset": total,
                "boundary_redacted": True,
                "boundary_pending": False,
                "boundary_safe_prefix": "",
                "checkpoint_offset": total,
                "checkpoint_mode": "normal",
                "checkpoint_quote": None,
                "checkpoint_quote_backslashes": 0,
            }
        chunk = read_chunk(cursor, _EXPAND_BOUNDARY_SCAN_CHARS)
        if not chunk:
            break
        combined = overlap + chunk
        overlap_chars = len(overlap)
        terminator_end: int | None = None
        if mode == "pem":
            match = _BOUNDARY_PRIVATE_KEY_END_RE.search(combined)
            if match is not None:
                terminator_end = match.end()
        elif mode == "token":
            for index, char in enumerate(chunk):
                if _is_unquoted_terminator(
                    read_chunk,
                    cursor + index,
                    char,
                ):
                    terminator_end = overlap_chars + index
                    break
        elif mode == "standalone":
            for index, char in enumerate(chunk):
                if _BOUNDARY_STANDALONE_BODY_RE.fullmatch(char) is None:
                    terminator_end = overlap_chars + index
                    break
        elif mode == "quote":
            for index, char in enumerate(chunk):
                if char == "\\":
                    quote_backslashes += 1
                    continue
                if char == quote and quote_backslashes % 2 == 0:
                    terminator_end = overlap_chars + index + 1
                    break
                quote_backslashes = 0
        elif mode == "escaped_quote":
            for index, char in enumerate(chunk):
                if char == "\\":
                    quote_backslashes += 1
                    continue
                if (
                    char == quote
                    and quote_backslashes == 1
                    and _escaped_quote_has_safe_terminator(
                        read_chunk,
                        cursor + index + 1,
                        total,
                        allow_eof=True,
                    )
                ):
                    terminator_end = overlap_chars + index + 1
                    break
                quote_backslashes = 0
        elif mode == "unknown":
            # Without a proven assignment start, neither quote type, token
            # whitespace, nor unrelated prose is a sound terminator. Advance
            # only through redacted chunks; EOF is the sole universal boundary.
            terminator_end = None
        else:
            found_quote = _first_unescaped_quote(combined, quote)
            pem_end = _BOUNDARY_PRIVATE_KEY_END_RE.search(combined)
            quote_end = found_quote[0] + 1 if found_quote is not None else None
            pem_end_index = pem_end.end() if pem_end is not None else None
            candidates = [value for value in (quote_end, pem_end_index) if value is not None]
            if candidates:
                terminator_end = min(candidates)
        if terminator_end is not None:
            safe_offset = cursor - overlap_chars + terminator_end
            return {
                "safe_content_offset": min(total, safe_offset),
                "boundary_redacted": True,
                "boundary_pending": False,
                "boundary_safe_prefix": safe_prefix,
                "checkpoint_offset": min(total, safe_offset),
                "checkpoint_mode": "normal",
                "checkpoint_quote": None,
                "checkpoint_quote_backslashes": 0,
            }
        cursor += len(chunk)
        overlap = chunk[-128:]

    return {
        "safe_content_offset": min(total, max(offset + 1, cursor)),
        "boundary_redacted": True,
        "boundary_pending": True,
        # Do not expose even the assignment name until the scanner has proved
        # a terminator in the same bounded operation. A continuation checkpoint
        # deliberately carries lexical state, not potentially sensitive text.
        "boundary_safe_prefix": "",
        "checkpoint_offset": min(total, max(offset + 1, cursor)),
        "checkpoint_mode": mode or "unknown",
        "checkpoint_quote": quote,
        "checkpoint_quote_backslashes": quote_backslashes,
    }


def _combined_result_sort_key(result: dict[str, Any], sort: str) -> tuple:
    sort_timestamp = float(result.get("_sort_ts") or 0.0)
    rank = result.get("_sort_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_sort_directness") or 0.0)
    type_bias = 0 if result.get("type") == "message" else 1
    role = result.get("role")
    if role == "user":
        role_bias = 0
    elif role == "assistant":
        role_bias = 1
    elif role == "tool":
        role_bias = 2
    else:
        role_bias = 1

    effective_directness = directness if result.get("type") == "message" else (directness * 0.8)

    if sort == "relevance":
        return (rank_value, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if sort == "hybrid":
        age_hours = max(0.0, (time.time() - sort_timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        summary_override = int(result.get("_hybrid_summary_override") or 0)
        return (-summary_override, blended, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if result.get("type") == "message":
        return (-sort_timestamp, type_bias, role_bias, rank_value, 0.0, float("inf"))
    return (-sort_timestamp, type_bias, 0, rank_value, 0.0, role_bias)

def _require_engine(kwargs: Dict[str, Any]) -> "LCMEngine | None":
    engine = kwargs.get("engine")
    return engine if engine is not None else None


def _get_session_node(engine: "LCMEngine", node_id: int):
    node = engine._dag.get_node(node_id)
    if node is None or node.session_id != engine.current_session_id:
        return None
    return node


def _get_externalized_payload(
    engine: "LCMEngine",
    ref: str,
    *,
    allowed_session_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    payload = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    if payload is None:
        return None
    payload_session_id = payload.get("session_id") or ""
    allowed = allowed_session_ids or {engine.current_session_id}
    if payload_session_id and payload_session_id not in allowed:
        return None
    return payload


def _decode_json_string_prefix(text: str, start: int, max_chars: int) -> tuple[str, bool]:
    """Decode at most ``max_chars`` from one JSON string starting at quote."""
    if start >= len(text) or text[start] != '"':
        raise ValueError("invalid content string")
    output: list[str] = []
    index = start + 1
    closed = False
    while index < len(text) and len(output) < max_chars:
        char = text[index]
        if char == '"':
            closed = True
            break
        if char != "\\":
            output.append(char)
            index += 1
            continue
        if index + 1 >= len(text):
            break
        escape = text[index + 1]
        if escape == "u":
            if index + 6 > len(text):
                break
            raw_escape = text[index:index + 6]
            index += 6
        else:
            raw_escape = text[index:index + 2]
            index += 2
        try:
            output.append(json.loads(f'"{raw_escape}"'))
        except (ValueError, json.JSONDecodeError):
            output.append("?")
    return "".join(output), closed


def _externalized_literal_match(content: str, query: str) -> re.Match[str] | None:
    terms = [
        phrase or word
        for phrase, word in re.findall(r'"([^"]+)"|(\S+)', query)
        if phrase or word
    ]
    if not terms:
        return None
    lowered = content.lower()
    if not all(term.lower() in lowered for term in terms):
        return None
    return re.search(re.escape(terms[0]), content, flags=re.IGNORECASE)


_EXTERNALIZED_CANONICAL_KEYS = frozenset({
    "kind", "tool_call_id", "role", "session_id", "field_path", "content",
    "content_chars", "content_bytes", "created_at",
    "persisted_output_source_path", "persisted_output_expected_chars",
    "persisted_output_preview_prefix", "persisted_output_preview_sha256",
    "persisted_output_redacted_preview_sha256", "persisted_output_file_size",
    "persisted_output_file_mtime_ns", "persisted_output_file_ctime_ns",
    "persisted_output_markers",
})
_EXTERNALIZED_TRAILING_CANONICAL_KEYS = frozenset({
    "content_chars", "content_bytes", "created_at",
    # ``41b43c8`` wrote the persisted-output block after ``content`` for
    # raw/tool payloads.  Keep this list explicit: persisted_output_* is not a
    # namespace extension point, and unknown lookalikes must remain rejected.
    "persisted_output_source_path", "persisted_output_expected_chars",
    "persisted_output_preview_prefix", "persisted_output_preview_sha256",
    "persisted_output_redacted_preview_sha256", "persisted_output_file_size",
    "persisted_output_file_mtime_ns", "persisted_output_file_ctime_ns",
    "persisted_output_markers",
})
_EXTERNALIZED_SUFFIX_MAX_DEPTH = 3
_EXTERNALIZED_CANONICAL_INTEGER_MAX = (1 << 63) - 1
_EXTERNALIZED_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXTERNALIZED_CANONICAL_STRING_MAX_CHARS = {
    "kind": 128,
    "tool_call_id": 512,
    "role": 128,
    "session_id": 512,
    "field_path": 2_048,
    "persisted_output_source_path": 4_096,
    "persisted_output_preview_prefix": 8_192,
    "persisted_output_preview_sha256": 64,
    "persisted_output_redacted_preview_sha256": 64,
}
_EXTERNALIZED_MARKER_STRING_MAX_CHARS = {
    "source_path": 4_096,
    "preview_prefix": 8_192,
    "preview_sha256": 64,
    "redacted_preview_sha256": 64,
}
# JSON string escapes can consume twelve source characters for one decoded
# non-BMP character (a surrogate pair). These are transport-buffer ceilings,
# not alternate value limits; decoded values are checked against the exact
# per-field maxima above.
_EXTERNALIZED_JSON_STRING_ESCAPE_FACTOR = 12
_EXTERNALIZED_CANONICAL_KEY_MAX_ENCODED_CHARS = 512
_EXTERNALIZED_CANONICAL_NUMBER_MAX_ENCODED_CHARS = 64
_EXTERNALIZED_CANONICAL_MARKER_MAX_ENCODED_CHARS = 192 * 1024
_EXTERNALIZED_PERSISTED_OUTPUT_KEYS = frozenset({
    "persisted_output_source_path", "persisted_output_expected_chars",
    "persisted_output_preview_prefix", "persisted_output_preview_sha256",
    "persisted_output_redacted_preview_sha256", "persisted_output_file_size",
    "persisted_output_file_mtime_ns", "persisted_output_file_ctime_ns",
    "persisted_output_markers",
})
_EXTERNALIZED_PERSISTED_OUTPUT_MARKER_KEYS = frozenset({
    "source_path", "expected_chars", "preview_prefix", "preview_sha256",
    "redacted_preview_sha256", "file_size", "file_mtime_ns", "file_ctime_ns",
})

_EXTERNALIZED_MARKER_STATE_ROOT: Path | None = None
_EXTERNALIZED_MARKER_STATE_TTL_SECONDS = 10 * 60.0
_EXTERNALIZED_MARKER_STATE_MAX_ORPHANS = 64
_EXTERNALIZED_MARKER_STATE_BATCH_ITEMS = 128
_EXTERNALIZED_MARKER_STATE_BATCH_BYTES = 32 * 1024
_EXTERNALIZED_MARKER_STATE_CACHE_BYTES = 64 * 1024
_EXTERNALIZED_MARKER_STATE_GUARD = threading.Lock()
_EXTERNALIZED_PRIVATE_STATE_SCANDIR = os.scandir
_EXTERNALIZED_OWNER_REGISTRY_NAME = ".owner-registry.db"
_EXTERNALIZED_OWNER_REGISTRY_CACHE_BYTES = 64 * 1024
_EXTERNALIZED_OWNER_REAP_MAX_ROWS = 32
_EXTERNALIZED_OWNER_REAP_MAX_ENTRIES = 64
_EXTERNALIZED_OWNER_REAP_DEADLINE_SECONDS = 0.050


class _ExternalizedStateUnavailable(OSError):
    """Private continuation state cannot be created safely on this host."""


class _ExternalizedRegistrySchemaMissing(_ExternalizedStateUnavailable):
    """The registry file exists but first-open schema work is incomplete."""


def _externalized_process_start_identity(pid: int) -> str | None:
    """Return Linux's non-recycled process start tick for ``pid`` when known."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii")
    except (OSError, ValueError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 2:].split()
    # The tail starts at field 3 (state); process starttime is field 22.
    if len(fields) <= 19 or not fields[19].isdigit():
        return None
    return fields[19]


def _externalized_owner_identity(name: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"owner-([0-9]+)-([0-9]+)-([0-9a-f]{32})", name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _externalized_owner_provably_dead(name: str) -> bool:
    identity = _externalized_owner_identity(name)
    if identity is None:
        return False
    pid, recorded_start = identity
    observed_start = _externalized_process_start_identity(pid)
    if observed_start is not None:
        return observed_start != recorded_start
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, PermissionError):
        return False
    return False


def _externalized_try_lock_lease(path: Path) -> int | None:
    """Acquire an orphan lease without following a hostile replacement."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows degrades without reaping
        return None
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        return None
    return descriptor


def _externalized_lock_new_lease(path: Path) -> int:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - no safe lease primitive
        raise _ExternalizedStateUnavailable("file leases are unavailable") from exc
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _externalized_state_root_candidates() -> tuple[Path, ...]:
    override = _EXTERNALIZED_MARKER_STATE_ROOT
    if override is not None:
        return (Path(override),)
    bases: list[Path] = []
    try:
        bases.append(Path(tempfile.gettempdir()))
    except (OSError, RuntimeError, FileNotFoundError):
        pass
    cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if cache_home:
        bases.append(Path(cache_home))
    try:
        bases.append(Path.home() / ".cache")
    except (OSError, RuntimeError):
        pass
    suffix = f"hermes-lcm-private-state-{getattr(os, 'getuid', lambda: 0)()}"
    unique: list[Path] = []
    for base in bases:
        candidate = base / suffix
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _externalized_registry_remaining_ms(deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ExternalizedStateUnavailable("private-state registry deadline expired")
    return max(1, min(50, int(remaining * 1000)))


def _externalized_open_owner_registry_unlocked(
    root: Path,
    *,
    deadline: float,
    allow_schema_create: bool = True,
) -> sqlite3.Connection:
    """Open the private, bounded-cache registry; never the production LCM DB."""
    _externalized_registry_remaining_ms(deadline)
    path = root / _EXTERNALIZED_OWNER_REGISTRY_NAME
    flags = os.O_RDWR | (os.O_CREAT if allow_schema_create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("unsafe private-state owner registry")
        getuid = getattr(os, "getuid", None)
        if callable(getuid) and opened.st_uid != getuid():
            raise PermissionError("private-state registry owner mismatch")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)

    remaining_ms = _externalized_registry_remaining_ms(deadline)
    connection = sqlite3.connect(
        str(path),
        timeout=remaining_ms / 1000.0,
        check_same_thread=False,
        cached_statements=8,
    )
    try:
        connection.execute(
            f"PRAGMA busy_timeout={_externalized_registry_remaining_ms(deadline)}"
        )
        connection.execute("PRAGMA busy_timeout=1")
        for pragma in (
            "PRAGMA temp_store=FILE",
            "PRAGMA mmap_size=0",
            "PRAGMA cache_spill=OFF",
            f"PRAGMA cache_size=-{_EXTERNALIZED_OWNER_REGISTRY_CACHE_BYTES // 1024}",
        ):
            _externalized_registry_remaining_ms(deadline)
            try:
                connection.execute(pragma)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                # These are connection-local memory controls, not schema or
                # correctness state. A contended startup may keep SQLite's
                # defaults for this short-lived registration connection.
                connection.execute("PRAGMA busy_timeout=1")
        _externalized_registry_remaining_ms(deadline)
        connection.execute(
            f"PRAGMA busy_timeout={_externalized_registry_remaining_ms(deadline)}"
        )
        existing_tables = {
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('owner_registry', 'registry_state')"
            )
        }
        if existing_tables != {"owner_registry", "registry_state"}:
            if not allow_schema_create:
                raise _ExternalizedRegistrySchemaMissing(
                    "private-state registry schema is incomplete"
                )
            _externalized_registry_remaining_ms(deadline)
            # No owner intent can exist until the schema is complete. Avoid
            # unbounded per-DDL fsyncs while creating this disposable private
            # registry, then enable FULL durability before returning it to
            # registration callers.
            connection.execute("PRAGMA synchronous=OFF")
            _externalized_registry_remaining_ms(deadline)
            connection.execute("PRAGMA journal_mode=WAL")
            _externalized_registry_remaining_ms(deadline)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS owner_registry ("
                "owner_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "owner_name TEXT NOT NULL UNIQUE, "
                "pid INTEGER NOT NULL, process_start TEXT NOT NULL, "
                "nonce TEXT NOT NULL, phase TEXT NOT NULL, "
                "created_wall REAL NOT NULL)"
            )
            _externalized_registry_remaining_ms(deadline)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS registry_state ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                "reap_cursor INTEGER NOT NULL)"
            )
            _externalized_registry_remaining_ms(deadline)
            connection.execute(
                "INSERT OR IGNORE INTO registry_state(singleton, reap_cursor) "
                "VALUES (1, 0)"
            )
            _externalized_registry_remaining_ms(deadline)
            connection.commit()
            _externalized_registry_remaining_ms(deadline)
            connection.execute("PRAGMA synchronous=FULL")
        else:
            # Legacy registries used reusable integer rowids. They remain safe
            # because reaping now compares every immutable nonce-bearing
            # registration field; newly created registries additionally use
            # AUTOINCREMENT so normal inserts never recycle a registration ID.
            pass
    except BaseException:
        connection.close()
        raise
    return connection


def _externalized_open_owner_registry(
    root: Path,
    *,
    deadline: float | None = None,
) -> sqlite3.Connection:
    """Serialize first-open/schema negotiation across concurrent processes."""
    if deadline is None:
        deadline = time.monotonic() + _EXTERNALIZED_OWNER_REAP_DEADLINE_SECONDS
    _externalized_registry_remaining_ms(deadline)
    lock_path = root / f"{_EXTERNALIZED_OWNER_REGISTRY_NAME}.lock"
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        lock_stat = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise OSError("unsafe private-state registry lock")
        getuid = getattr(os, "getuid", None)
        if callable(getuid) and lock_stat.st_uid != getuid():
            raise PermissionError("private-state registry lock owner mismatch")
        os.fchmod(lock_descriptor, 0o600)
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - no safe schema lock
            raise _ExternalizedStateUnavailable(
                "private-state registry locks are unavailable"
            ) from exc
        while True:
            try:
                fcntl.flock(
                    lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _ExternalizedStateUnavailable(
                        "private-state registry lock deadline expired"
                    ) from exc
                time.sleep(min(0.002, remaining))
        try:
            registry_path = root / _EXTERNALIZED_OWNER_REGISTRY_NAME
            try:
                registry_stat = registry_path.stat()
            except FileNotFoundError:
                registry_stat = None
            if registry_stat is None or registry_stat.st_size == 0:
                return _externalized_open_owner_registry_unlocked(
                    root, deadline=deadline, allow_schema_create=True
                )
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        # Once the non-empty registry exists, the flock protects only schema
        # creation. Release it before connection-local setup so concurrent
        # registrations do not serialize their full SQLite open path.
        return _externalized_open_owner_registry_unlocked(
            root, deadline=deadline, allow_schema_create=False
        )
    finally:
        os.close(lock_descriptor)


def _externalized_bounded_remove_private_tree(
    path: Path,
    *,
    budget: dict[str, int],
    deadline: float,
) -> bool:
    """Make a capped, unsorted cleanup pass over one registered owner tree."""
    if time.monotonic() >= deadline:
        return False
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return True
    if not stat.S_ISDIR(path_stat.st_mode) or path.is_symlink():
        return False
    try:
        with _EXTERNALIZED_PRIVATE_STATE_SCANDIR(path) as entries:
            iterator = iter(entries)
            while budget["entries"] < _EXTERNALIZED_OWNER_REAP_MAX_ENTRIES:
                if time.monotonic() >= deadline:
                    return False
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                budget["entries"] += 1
                child = path / entry.name
                if entry.is_dir(follow_symlinks=False):
                    if not _externalized_bounded_remove_private_tree(
                        child, budget=budget, deadline=deadline
                    ):
                        return False
                else:
                    child.unlink(missing_ok=True)
            else:
                return False
        path.rmdir()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _externalized_reap_dead_owners(root: Path) -> dict[str, Any]:
    """Visit an indexed, persistent-cursor slice of registered owners only."""
    started = time.monotonic()
    deadline = started + _EXTERNALIZED_OWNER_REAP_DEADLINE_SECONDS
    stats: dict[str, Any] = {
        "rows_visited": 0,
        "entries_visited": 0,
        "elapsed": 0.0,
    }
    try:
        connection = _externalized_open_owner_registry(root, deadline=deadline)
    except (OSError, sqlite3.Error):
        stats["elapsed"] = time.monotonic() - started
        return stats
    budget = {"entries": 0}
    try:
        connection.execute(
            f"PRAGMA busy_timeout={_externalized_registry_remaining_ms(deadline)}"
        )
        cursor_row = connection.execute(
            "SELECT reap_cursor FROM registry_state WHERE singleton = 1"
        ).fetchone()
        cursor = int(cursor_row[0]) if cursor_row is not None else 0
        rows = connection.execute(
            "SELECT owner_id, owner_name, pid, process_start, nonce, phase "
            "FROM owner_registry "
            "WHERE owner_id > ? ORDER BY owner_id LIMIT ?",
            (cursor, _EXTERNALIZED_OWNER_REAP_MAX_ROWS),
        ).fetchall()
        if not rows:
            if cursor:
                try:
                    if time.monotonic() >= deadline:
                        return stats
                    remaining_ms = _externalized_registry_remaining_ms(deadline)
                    connection.execute(f"PRAGMA busy_timeout={remaining_ms}")
                    connection.execute(
                        "UPDATE registry_state SET reap_cursor = 0 "
                        "WHERE singleton = 1"
                    )
                    connection.commit()
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower():
                        raise
                    connection.rollback()
            return stats

        completed_cursor = cursor
        delete_owners: list[tuple[int, str, int, str, str, str]] = []
        for (
            raw_owner_id,
            raw_owner_name,
            raw_pid,
            raw_process_start,
            raw_nonce,
            raw_phase,
        ) in rows:
            if time.monotonic() >= deadline:
                break
            owner_id = int(raw_owner_id)
            owner_name = str(raw_owner_name)
            owner_registration = (
                owner_id,
                owner_name,
                int(raw_pid),
                str(raw_process_start),
                str(raw_nonce),
                str(raw_phase),
            )
            stats["rows_visited"] += 1
            identity = _externalized_owner_identity(owner_name)
            if identity is None:
                # App-created owners always have a valid registered identity.
                # Unknown legacy/malformed root entries are isolated: the
                # shared root is never scanned to discover or adopt them.
                delete_owners.append(owner_registration)
                completed_cursor = owner_id
                continue
            if not _externalized_owner_provably_dead(owner_name):
                completed_cursor = owner_id
                continue

            owner_dir = root / owner_name
            lease_path = owner_dir / "owner.lease"
            lease_descriptor = None
            if lease_path.exists():
                lease_descriptor = _externalized_try_lock_lease(lease_path)
                if lease_descriptor is None:
                    completed_cursor = owner_id
                    continue
            try:
                removed = _externalized_bounded_remove_private_tree(
                    owner_dir, budget=budget, deadline=deadline
                )
            finally:
                if lease_descriptor is not None:
                    os.close(lease_descriptor)
            if not removed:
                # Keep the cursor immediately before this row so a large dead
                # owner is drained over repeated bounded initialization passes.
                break
            delete_owners.append(owner_registration)
            completed_cursor = owner_id

        if time.monotonic() >= deadline:
            return stats
        remaining_ms = _externalized_registry_remaining_ms(deadline)
        connection.execute(f"PRAGMA busy_timeout={remaining_ms}")
        try:
            connection.execute("BEGIN IMMEDIATE")
            if delete_owners:
                connection.executemany(
                    "DELETE FROM owner_registry WHERE owner_id = ? "
                    "AND owner_name = ? AND pid = ? AND process_start = ? "
                    "AND nonce = ? AND phase = ?",
                    delete_owners,
                )
            connection.execute(
                "UPDATE registry_state SET reap_cursor = ? WHERE singleton = 1",
                (completed_cursor,),
            )
            connection.commit()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            connection.rollback()
    finally:
        stats["entries_visited"] = budget["entries"]
        stats["elapsed"] = time.monotonic() - started
        connection.close()
    return stats


def _externalized_remove_private_tree(path: Path) -> None:
    """Remove one validated private owner without ambient ``os.scandir``."""
    with _EXTERNALIZED_PRIVATE_STATE_SCANDIR(path) as entries:
        for entry in entries:
            child = path / entry.name
            if entry.is_dir(follow_symlinks=False):
                _externalized_remove_private_tree(child)
            else:
                child.unlink(missing_ok=True)
    path.rmdir()


def _prepare_externalized_marker_state_root() -> Path:
    """Resolve and prepare the private root lazily, never during import."""
    with _EXTERNALIZED_MARKER_STATE_GUARD:
        last_error: BaseException | None = None
        for root in _externalized_state_root_candidates():
            try:
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
                root_stat = root.lstat()
                if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
                    raise OSError("unsafe externalized private-state root")
                getuid = getattr(os, "getuid", None)
                if callable(getuid) and root_stat.st_uid != getuid():
                    raise PermissionError("private-state root owner mismatch")
                root.chmod(0o700)
                _externalized_reap_dead_owners(root)
                return root
            except (OSError, RuntimeError, sqlite3.Error) as exc:
                last_error = exc
        raise _ExternalizedStateUnavailable(
            "no writable private temp/cache state root"
        ) from last_error


class _ExternalizedPrivateRuntimeState:
    """Per-engine private disk owner, shutdown fence, and checkout registry."""

    __slots__ = (
        "lock", "closed", "generation", "active_checkouts", "owner_dir",
        "owner_name", "lease_descriptor", "scheduler_connection",
    )

    def __init__(self):
        self.lock = threading.RLock()
        self.closed = False
        self.generation = 0
        self.active_checkouts: dict[
            int, tuple["_ExternalizedPayloadContinuation", int]
        ] = {}
        self.owner_dir: Path | None = None
        self.owner_name: str | None = None
        self.lease_descriptor: int | None = None
        self.scheduler_connection: sqlite3.Connection | None = None

    def ensure_owner_dir(self) -> Path:
        with self.lock:
            if self.closed:
                raise _ExternalizedStateUnavailable("private state is closed")
            if self.owner_dir is not None:
                return self.owner_dir
            root = _prepare_externalized_marker_state_root()
            # Linux start ticks disambiguate PID reuse. Other platforms retain
            # the random 128-bit nonce plus held lease and conservatively reap
            # only after the PID itself is certainly absent.
            start = _externalized_process_start_identity(os.getpid()) or "0"
            nonce = secrets.token_hex(16)
            owner_name = f"owner-{os.getpid()}-{start}-{nonce}"
            owner_dir = root / owner_name
            registry_open_deadline = time.monotonic() + 1.0
            while True:
                try:
                    registry = _externalized_open_owner_registry(root)
                    break
                except sqlite3.OperationalError as exc:
                    if (
                        "locked" not in str(exc).lower()
                        or time.monotonic() >= registry_open_deadline
                    ):
                        raise
                    time.sleep(0.002)
            registered = False
            lease_descriptor: int | None = None
            try:
                registration_deadline = time.monotonic() + 1.0
                while True:
                    try:
                        registry.execute("PRAGMA busy_timeout=50")
                        registry.execute(
                            "INSERT INTO owner_registry("
                            "owner_name, pid, process_start, nonce, phase, created_wall"
                            ") VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                owner_name,
                                os.getpid(),
                                start,
                                nonce,
                                "intended",
                                time.time(),
                            ),
                        )
                        registry.commit()
                        break
                    except sqlite3.OperationalError as exc:
                        registry.rollback()
                        if (
                            "locked" not in str(exc).lower()
                            or time.monotonic() >= registration_deadline
                        ):
                            raise
                        time.sleep(0.002)
                registered = True
                owner_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
                lease_descriptor = _externalized_lock_new_lease(
                    owner_dir / "owner.lease"
                )
            except BaseException:
                removed = False
                try:
                    _externalized_remove_private_tree(owner_dir)
                    removed = True
                except FileNotFoundError:
                    removed = True
                except OSError:
                    removed = False
                if registered and removed:
                    try:
                        registry.execute(
                            "DELETE FROM owner_registry WHERE owner_name = ?",
                            (owner_name,),
                        )
                        registry.commit()
                    except sqlite3.Error:
                        pass
                if lease_descriptor is not None:
                    try:
                        os.close(lease_descriptor)
                    except OSError:
                        pass
                raise
            finally:
                registry.close()
            self.owner_dir = owner_dir
            self.owner_name = owner_name
            self.lease_descriptor = lease_descriptor
            return owner_dir

    def register_checkout(
        self, continuation: "_ExternalizedPayloadContinuation"
    ) -> int | None:
        with self.lock:
            if self.closed:
                continuation.close()
                return None
            identity = id(continuation)
            current = self.active_checkouts.get(identity)
            count = current[1] + 1 if current is not None else 1
            self.active_checkouts[identity] = (continuation, count)
            return self.generation

    def return_checkout(
        self,
        continuation: "_ExternalizedPayloadContinuation",
        generation: int | None,
    ) -> None:
        close_owner = False
        with self.lock:
            identity = id(continuation)
            current = self.active_checkouts.get(identity)
            if current is not None:
                if current[1] <= 1:
                    self.active_checkouts.pop(identity, None)
                else:
                    self.active_checkouts[identity] = (
                        current[0], current[1] - 1
                    )
            if self.closed or generation != self.generation:
                continuation.close()
            close_owner = self.closed and not self.active_checkouts
        if close_owner:
            self._close_owner()

    def accepts_generation(self, generation: int | None) -> bool:
        with self.lock:
            return not self.closed and generation == self.generation

    def close_for_shutdown(self) -> None:
        scheduler: sqlite3.Connection | None
        close_owner: bool
        with self.lock:
            if not self.closed:
                self.closed = True
                self.generation += 1
            scheduler = self.scheduler_connection
            self.scheduler_connection = None
            close_owner = not self.active_checkouts
        if scheduler is not None:
            scheduler.close()
        if close_owner:
            self._close_owner()

    def _close_owner(self) -> None:
        with self.lock:
            owner_dir = self.owner_dir
            owner_name = self.owner_name
            lease_descriptor = self.lease_descriptor
            self.owner_dir = None
            self.owner_name = None
            self.lease_descriptor = None
        removed = owner_dir is None
        if owner_dir is not None:
            try:
                _externalized_remove_private_tree(owner_dir)
                removed = True
            except FileNotFoundError:
                removed = True
            except OSError:
                logger.debug(
                    "LCM could not remove private-state owner %s",
                    owner_dir,
                    exc_info=True,
                )
        if lease_descriptor is not None:
            try:
                os.close(lease_descriptor)
            except OSError:
                pass
        if removed and owner_dir is not None and owner_name is not None:
            try:
                registry = _externalized_open_owner_registry(owner_dir.parent)
                try:
                    unregister_deadline = time.monotonic() + 1.0
                    while True:
                        try:
                            registry.execute("PRAGMA busy_timeout=50")
                            registry.execute(
                                "DELETE FROM owner_registry WHERE owner_name = ?",
                                (owner_name,),
                            )
                            registry.commit()
                            break
                        except sqlite3.OperationalError as exc:
                            registry.rollback()
                            if (
                                "locked" not in str(exc).lower()
                                or time.monotonic() >= unregister_deadline
                            ):
                                raise
                            time.sleep(0.002)
                finally:
                    registry.close()
            except (OSError, sqlite3.Error):
                logger.debug(
                    "LCM could not unregister private-state owner %s",
                    owner_name,
                    exc_info=True,
                )


def _externalized_runtime_state(engine: "LCMEngine") -> _ExternalizedPrivateRuntimeState:
    with _EXTERNALIZED_CONTINUATION_STATE_GUARD:
        state = getattr(engine, "_externalized_grep_runtime_state", None)
        if not isinstance(state, _ExternalizedPrivateRuntimeState):
            state = _ExternalizedPrivateRuntimeState()
            setattr(engine, "_externalized_grep_runtime_state", state)
    return state


def _unlink_externalized_marker_state(path: Path) -> None:
    """Remove a private marker index and any SQLite sidecars."""
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            logger.debug(
                "LCM could not remove temporary marker state %s",
                candidate,
                exc_info=True,
            )


def _externalized_marker_identity_bytes(identity: tuple[Any, ...]) -> bytes:
    """Return the exact canonical marker identity stored in the UNIQUE index."""
    return json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


class _ExternalizedMarkerIdentityStore:
    """Bounded-heap, disk-backed exact duplicate detector for one marker list."""

    __slots__ = (
        "path", "connection", "pending", "pending_set", "pending_bytes",
        "deadline", "deadline_error", "closed", "stat_identity",
        "runtime_state", "owns_runtime_state",
    )

    def __init__(
        self,
        *,
        stat_identity: tuple[int, ...] = (),
        runtime_state: _ExternalizedPrivateRuntimeState | None = None,
    ):
        self.runtime_state = runtime_state or _ExternalizedPrivateRuntimeState()
        self.owns_runtime_state = runtime_state is None
        root = self.runtime_state.ensure_owner_dir()
        descriptor, raw_path = tempfile.mkstemp(
            prefix="markers-",
            suffix=".sqlite3",
            dir=root,
        )
        os.close(descriptor)
        self.path = Path(raw_path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        self.stat_identity = tuple(int(item) for item in stat_identity)
        self.pending: list[bytes] = []
        self.pending_set: set[bytes] = set()
        self.pending_bytes = 0
        self.deadline: float | None = None
        self.deadline_error = "body_deadline"
        self.closed = False
        self.connection: sqlite3.Connection | None = None
        try:
            self.connection = sqlite3.connect(
                str(self.path),
                timeout=0.05,
                check_same_thread=False,
                cached_statements=4,
            )
            # The index is disposable continuation state, never authoritative
            # data. Avoid durable-journal I/O; process death reaps/culls it.
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA cache_spill=OFF")
            self.connection.execute(
                f"PRAGMA cache_size=-{_EXTERNALIZED_MARKER_STATE_CACHE_BYTES // 1024}"
            )
            self.connection.execute(
                "CREATE TABLE marker_identities "
                "(identity BLOB PRIMARY KEY) WITHOUT ROWID"
            )
            self.connection.execute(
                "CREATE TABLE state_metadata (stat_identity TEXT NOT NULL)"
            )
            self.connection.execute(
                "INSERT INTO state_metadata(stat_identity) VALUES (?)",
                (json.dumps(self.stat_identity, separators=(",", ":")),),
            )
            self.connection.commit()
        except BaseException:
            self.close()
            raise

    def configure_deadline(self, deadline: float | None, error: str) -> None:
        self.deadline = deadline
        self.deadline_error = error

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(self.deadline_error)

    def add(self, identity: tuple[Any, ...]) -> None:
        encoded = _externalized_marker_identity_bytes(identity)
        if self.pending and (
            len(self.pending) >= _EXTERNALIZED_MARKER_STATE_BATCH_ITEMS
            or self.pending_bytes + len(encoded)
            > _EXTERNALIZED_MARKER_STATE_BATCH_BYTES
        ):
            # Flush already-consumed markers before accepting the current one,
            # so a deadline never leaves an unconsumed marker in pending state.
            self.flush()
        if encoded in self.pending_set:
            raise ValueError("invalid_payload")
        self.pending.append(encoded)
        self.pending_set.add(encoded)
        self.pending_bytes += len(encoded)

    def flush(self) -> None:
        if not self.pending:
            self._check_deadline()
            return
        self._check_deadline()
        expected = len(self.pending)
        assert self.connection is not None
        before = self.connection.total_changes
        self.connection.executemany(
            "INSERT OR IGNORE INTO marker_identities(identity) VALUES (?)",
            ((identity,) for identity in self.pending),
        )
        self.connection.commit()
        inserted = self.connection.total_changes - before
        self.pending.clear()
        self.pending_set.clear()
        self.pending_bytes = 0
        if inserted != expected:
            raise ValueError("invalid_payload")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.connection is not None:
                self.connection.close()
        finally:
            self.connection = None
            self.pending.clear()
            self.pending_set.clear()
            self.pending_bytes = 0
            _unlink_externalized_marker_state(self.path)
            if self.owns_runtime_state:
                self.runtime_state.close_for_shutdown()

    def retained_bytes(self) -> int:
        # Native SQLite allocation is verified by subprocess RSS regressions;
        # do not present a pager target as if it were measured native memory.
        return _externalized_python_retained_bytes(self)

    def __del__(self):  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


class _ExternalizedPersistedOutputMarkers:
    """Bounded summary of an incrementally validated historical marker list."""

    __slots__ = ("count", "first_marker")

    def __init__(self, count: int, first_marker: dict[str, Any]):
        self.count = count
        self.first_marker = first_marker


class _ExternalizedSuffixBudgetExceeded(Exception):
    """A canonical-looking suffix exceeded this operation's marker budget."""


class _ExternalizedSuffixOperationBudget:
    """Shared diagnostic/CPU accounting bounded by the operation byte cap."""

    __slots__ = ("max_markers", "markers", "runtime_state")

    def __init__(
        self,
        max_markers: int | None = None,
        *,
        runtime_state: _ExternalizedPrivateRuntimeState | None = None,
    ):
        self.max_markers = (
            None if max_markers is None else max(0, int(max_markers))
        )
        self.markers = 0
        self.runtime_state = runtime_state

    def charge(self, count: int = 1) -> None:
        count = max(0, int(count))
        if (
            self.max_markers is not None
            and self.markers + count > self.max_markers
        ):
            raise _ExternalizedSuffixBudgetExceeded
        self.markers += count


class _ExternalizedByteOperationBudget:
    """Charge transport bytes synchronously, including failed parse attempts."""

    __slots__ = ("max_bytes", "bytes_read", "exhausted")

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_read = 0
        self.exhausted = self.max_bytes == 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_bytes - self.bytes_read)

    def read(self, handle, size: int = -1) -> bytes:
        remaining = self.remaining
        if remaining <= 0:
            self.exhausted = True
            return b""
        requested = remaining if size < 0 else min(max(0, int(size)), remaining)
        if requested <= 0:
            return b""
        try:
            start_offset = handle.tell()
        except (AttributeError, OSError, ValueError):
            start_offset = None
        try:
            raw = handle.read(requested)
        except BaseException:
            # Defensive accounting for file-like implementations that advance
            # before surfacing a transport exception.
            if start_offset is not None:
                try:
                    advanced = max(0, int(handle.tell()) - int(start_offset))
                except (AttributeError, OSError, TypeError, ValueError):
                    advanced = 0
                self.bytes_read += min(remaining, advanced)
                if self.bytes_read >= self.max_bytes:
                    self.exhausted = True
            raise
        # Charge before returning control to any decoder/parser so exceptions
        # cannot bypass the operation ledger.
        self.bytes_read += len(raw)
        if self.bytes_read >= self.max_bytes:
            self.exhausted = True
        return raw


class _ExternalizedBudgetedReader:
    """File-like view whose reads share one operation byte budget."""

    __slots__ = ("_handle", "_budget", "_capture")

    def __init__(
        self,
        handle,
        budget: _ExternalizedByteOperationBudget,
        capture: bytearray | None = None,
    ):
        self._handle = handle
        self._budget = budget
        self._capture = capture

    def read(self, size: int = -1) -> bytes:
        raw = self._budget.read(self._handle, size)
        if self._capture is not None:
            self._capture.extend(raw)
        return raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _ExternalizedPrependReader:
    """Replay bytes already charged/read while continuing from the file."""

    __slots__ = ("_handle", "_prefix", "_index", "_logical_offset")

    def __init__(self, handle, prefix: bytes):
        self._handle = handle
        self._prefix = prefix
        self._index = 0
        self._logical_offset = handle.tell() - len(prefix)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            prefix = self._prefix[self._index:]
            self._index = len(self._prefix)
            raw = prefix + self._handle.read(-1)
        else:
            remaining_prefix = len(self._prefix) - self._index
            prefix_size = min(max(0, size), remaining_prefix)
            prefix = self._prefix[self._index:self._index + prefix_size]
            self._index += prefix_size
            raw = prefix
            if len(raw) < size:
                raw += self._handle.read(size - len(raw))
        self._logical_offset += len(raw)
        return raw

    def tell(self) -> int:
        return self._logical_offset

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _externalized_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one nested JSON object while preserving fail-closed duplicates."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("ambiguous_metadata")
        value[key] = item
    return value


def _externalized_canonical_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _EXTERNALIZED_CANONICAL_INTEGER_MAX
    )


def _validate_externalized_persisted_output_marker(marker: Any) -> tuple[Any, ...]:
    if not isinstance(marker, dict):
        raise ValueError("invalid_payload")
    if not set(marker).issubset(_EXTERNALIZED_PERSISTED_OUTPUT_MARKER_KEYS):
        raise ValueError("ambiguous_metadata")
    if not {"source_path", "expected_chars"}.issubset(marker):
        raise ValueError("invalid_payload")

    source_path = marker["source_path"]
    if (
        not isinstance(source_path, str)
        or not source_path
        or "\x00" in source_path
        or len(source_path) > _EXTERNALIZED_MARKER_STRING_MAX_CHARS["source_path"]
    ):
        raise ValueError("invalid_payload")
    if not _externalized_canonical_integer(marker["expected_chars"]):
        raise ValueError("invalid_payload")
    if "preview_prefix" in marker and (
        not isinstance(marker["preview_prefix"], str)
        or not marker["preview_prefix"]
        or len(marker["preview_prefix"])
        > _EXTERNALIZED_MARKER_STRING_MAX_CHARS["preview_prefix"]
    ):
        raise ValueError("invalid_payload")
    for key in ("preview_sha256", "redacted_preview_sha256"):
        if key in marker and (
            not isinstance(marker[key], str)
            or _EXTERNALIZED_SHA256_RE.fullmatch(marker[key]) is None
            or len(marker[key]) > _EXTERNALIZED_MARKER_STRING_MAX_CHARS[key]
        ):
            raise ValueError("invalid_payload")
    for key in ("file_size", "file_mtime_ns", "file_ctime_ns"):
        if key in marker and not _externalized_canonical_integer(marker[key]):
            raise ValueError("invalid_payload")
    return tuple(
        marker.get(key)
        for key in sorted(_EXTERNALIZED_PERSISTED_OUTPUT_MARKER_KEYS)
    )


def _validate_externalized_metadata_field(key: str, value: Any) -> None:
    if key in {"kind", "tool_call_id", "role", "session_id", "field_path"}:
        if (
            not isinstance(value, str)
            or len(value) > _EXTERNALIZED_CANONICAL_STRING_MAX_CHARS[key]
        ):
            raise ValueError("invalid_payload")
    elif key in {"content_chars", "content_bytes"}:
        if not _externalized_canonical_integer(value):
            raise ValueError("invalid_payload")
    elif key == "created_at":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            or value > _EXTERNALIZED_CANONICAL_INTEGER_MAX
        ):
            raise ValueError("invalid_payload")
    elif key == "persisted_output_source_path":
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or len(value) > _EXTERNALIZED_CANONICAL_STRING_MAX_CHARS[key]
        ):
            raise ValueError("invalid_payload")
    elif key in {
        "persisted_output_expected_chars", "persisted_output_file_size",
        "persisted_output_file_mtime_ns", "persisted_output_file_ctime_ns",
    }:
        if not _externalized_canonical_integer(value):
            raise ValueError("invalid_payload")
    elif key == "persisted_output_preview_prefix":
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _EXTERNALIZED_CANONICAL_STRING_MAX_CHARS[key]
        ):
            raise ValueError("invalid_payload")
    elif key in {
        "persisted_output_preview_sha256",
        "persisted_output_redacted_preview_sha256",
    }:
        if (
            not isinstance(value, str)
            or len(value) > _EXTERNALIZED_CANONICAL_STRING_MAX_CHARS[key]
            or _EXTERNALIZED_SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError("invalid_payload")
    elif key == "persisted_output_markers":
        if isinstance(value, _ExternalizedPersistedOutputMarkers):
            if (
                value.count <= 0
                or not value.first_marker
            ):
                raise ValueError("invalid_payload")
            return
        if (
            not isinstance(value, list)
            or not value
        ):
            raise ValueError("invalid_payload")
        identities = [
            _validate_externalized_persisted_output_marker(marker)
            for marker in value
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("invalid_payload")


def _validate_externalized_metadata(fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        _validate_externalized_metadata_field(key, value)

    present = _EXTERNALIZED_PERSISTED_OUTPUT_KEYS.intersection(fields)
    if not present:
        return
    required = {
        "persisted_output_source_path",
        "persisted_output_expected_chars",
        "persisted_output_markers",
    }
    if not required.issubset(fields):
        raise ValueError("invalid_payload")

    markers = fields["persisted_output_markers"]
    first_marker = (
        markers.first_marker
        if isinstance(markers, _ExternalizedPersistedOutputMarkers)
        else markers[0]
    )
    top_to_marker = {
        "persisted_output_source_path": "source_path",
        "persisted_output_expected_chars": "expected_chars",
        "persisted_output_preview_prefix": "preview_prefix",
        "persisted_output_preview_sha256": "preview_sha256",
        "persisted_output_redacted_preview_sha256": "redacted_preview_sha256",
        "persisted_output_file_size": "file_size",
        "persisted_output_file_mtime_ns": "file_mtime_ns",
        "persisted_output_file_ctime_ns": "file_ctime_ns",
    }
    for top_key, marker_key in top_to_marker.items():
        if top_key in fields and first_marker.get(marker_key) != fields[top_key]:
            raise ValueError("invalid_payload")


def _externalized_prefix_authorization(
    text: str,
) -> tuple[dict[str, Any], set[str], str | None]:
    """Parse pre-content top-level fields and reject duplicate keys."""
    decoder = json.JSONDecoder(
        object_pairs_hook=_externalized_object_without_duplicate_keys
    )
    fields: dict[str, Any] = {}
    seen: set[str] = set()
    index = 0
    length = len(text)

    def whitespace(pos: int) -> int:
        while pos < length and text[pos] in " \t\n\r":
            pos += 1
        return pos

    index = whitespace(index)
    if index >= length or text[index] != "{":
        return fields, seen, "invalid_payload"
    index += 1
    while True:
        index = whitespace(index)
        if index >= length or text[index] != '"':
            return fields, seen, "invalid_payload"
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, seen, "invalid_payload"
        if not isinstance(key, str):
            return fields, seen, "invalid_payload"
        if key not in _EXTERNALIZED_CANONICAL_KEYS:
            return fields, seen, "ambiguous_metadata"
        if key in seen:
            return fields, seen, "ambiguous_metadata"
        seen.add(key)
        index = whitespace(index)
        if index >= length or text[index] != ":":
            return fields, seen, "invalid_payload"
        index = whitespace(index + 1)
        if key == "content":
            if index >= length or text[index] != '"':
                return fields, seen, "invalid_payload"
            return fields, seen, None
        try:
            value, index = decoder.raw_decode(text, index)
            _validate_externalized_metadata_field(key, value)
        except json.JSONDecodeError:
            return fields, seen, "invalid_payload"
        except ValueError as exc:
            error = str(exc)
            return (
                fields,
                seen,
                error
                if error in {"ambiguous_metadata", "payload_truncated"}
                else "invalid_payload",
            )
        fields[key] = value
        index = whitespace(index)
        if index >= length or text[index] != ",":
            return fields, seen, "invalid_payload"
        index += 1


class _ExternalizedSuffixParser:
    """Incrementally validate canonical metadata on either side of content.

    Marker arrays may occur before content (current writer) or after content
    (historical writer). The parser discards every completed marker, retaining
    exact identities in a private disk-backed UNIQUE index and the first marker
    for the canonical top-level consistency check.
    """

    def __init__(
        self,
        *,
        seen: set[str],
        operation_budget: _ExternalizedSuffixOperationBudget | None = None,
        prefix: bool = False,
        stat_identity: tuple[int, ...] = (),
    ):
        self.decoder = json.JSONDecoder(
            object_pairs_hook=_externalized_object_without_duplicate_keys
        )
        self.seen = seen
        self.fields: dict[str, Any] = {}
        self.buffer = ""
        self.index = 0
        self.prefix = prefix
        self.state = "root_start" if prefix else "start"
        self.current_key = ""
        self.marker_count = 0
        self.first_marker: dict[str, Any] | None = None
        self.marker_store: _ExternalizedMarkerIdentityStore | None = None
        self.deadline: float | None = None
        self.deadline_error = "body_deadline"
        self.stat_identity = stat_identity
        self.operation_budget = (
            operation_budget or _ExternalizedSuffixOperationBudget()
        )
        # The outer object is already open when suffix parsing starts.
        self.lexical_depth = 0 if prefix else 1
        self.lexical_in_string = False
        self.lexical_escaped = False

    def _scan_depth(self, text: str) -> None:
        for character in text:
            if self.lexical_in_string:
                if self.lexical_escaped:
                    self.lexical_escaped = False
                elif character == "\\":
                    self.lexical_escaped = True
                elif character == '"':
                    self.lexical_in_string = False
                continue
            if character == '"':
                self.lexical_in_string = True
            elif character in "[{":
                self.lexical_depth += 1
                if self.lexical_depth > _EXTERNALIZED_SUFFIX_MAX_DEPTH:
                    raise ValueError("invalid_payload")
            elif character in "]}":
                self.lexical_depth -= 1
                if self.lexical_depth < 0:
                    raise ValueError("ambiguous_metadata")

    def _whitespace(self, index: int) -> int:
        while index < len(self.buffer) and self.buffer[index] in " \t\n\r":
            index += 1
        return index

    def _raw_decode_with_delimiter(
        self, index: int, delimiters: str
    ) -> tuple[Any, int] | None:
        try:
            value, end = self.decoder.raw_decode(self.buffer, index)
        except json.JSONDecodeError:
            return None
        delimiter_index = self._whitespace(end)
        if delimiter_index >= len(self.buffer):
            # Whitespace proves that the decoded token is complete even when
            # its comma/close delimiter arrives in a later transport call.
            # Objects, strings, and literals are also syntactically complete
            # at their closing token. Only a number ending exactly at the
            # current buffer boundary can still gain digits/exponent bytes.
            if delimiter_index > end or not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return value, end
            return None
        if self.buffer[delimiter_index] not in delimiters:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and self.buffer[delimiter_index] in ".eE+-0123456789"
            ):
                return None
            raise ValueError("invalid_payload")
        return value, end

    def _finish_marker_list(self) -> None:
        if (
            self.marker_count <= 0
            or self.first_marker is None
        ):
            raise ValueError("invalid_payload")
        self.fields["persisted_output_markers"] = (
            _ExternalizedPersistedOutputMarkers(
                self.marker_count,
                self.first_marker,
            )
        )
        if self.marker_store is not None:
            self.marker_store.flush()
            self.marker_store.close()
            self.marker_store = None
        self.state = "value_delimiter"

    def _process(self, *, final: bool) -> None:
        while True:
            self.index = self._whitespace(self.index)
            if self.state == "content":
                break
            if self.state == "done":
                if self.index < len(self.buffer):
                    raise ValueError("ambiguous_metadata")
                break
            if self.index >= len(self.buffer):
                break

            character = self.buffer[self.index]
            if self.state == "root_start":
                if character != "{":
                    raise ValueError("invalid_payload")
                self.index += 1
                self.state = "key"
            elif self.state == "start":
                if character == "}":
                    self.index += 1
                    self.state = "done"
                elif character == ",":
                    self.index += 1
                    self.state = "key"
                else:
                    raise ValueError("ambiguous_metadata")
            elif self.state == "key":
                if character != '"':
                    raise ValueError("invalid_payload")
                try:
                    key, end = self.decoder.raw_decode(self.buffer, self.index)
                except json.JSONDecodeError:
                    break
                if not isinstance(key, str):
                    raise ValueError("invalid_payload")
                allowed_keys = (
                    _EXTERNALIZED_CANONICAL_KEYS
                    if self.prefix
                    else _EXTERNALIZED_TRAILING_CANONICAL_KEYS
                )
                if key in self.seen or key not in allowed_keys:
                    raise ValueError("ambiguous_metadata")
                self.seen.add(key)
                self.current_key = key
                self.index = end
                self.state = "colon"
            elif self.state == "colon":
                if character != ":":
                    raise ValueError("invalid_payload")
                self.index += 1
                if self.current_key == "content":
                    self.state = "content_start"
                elif self.current_key == "persisted_output_markers":
                    self.state = "marker_list_start"
                else:
                    self.state = "value"
            elif self.state == "content_start":
                if character != '"':
                    raise ValueError("invalid_payload")
                self.index += 1
                self.state = "content"
            elif self.state == "value":
                decoded = self._raw_decode_with_delimiter(self.index, ",}")
                if decoded is None:
                    break
                value, end = decoded
                try:
                    _validate_externalized_metadata_field(self.current_key, value)
                except ValueError as exc:
                    if str(exc) == "ambiguous_metadata":
                        raise
                    raise ValueError("invalid_payload") from exc
                self.fields[self.current_key] = value
                self.index = end
                self.state = "value_delimiter"
            elif self.state == "marker_list_start":
                if character != "[":
                    raise ValueError("invalid_payload")
                self.index += 1
                self.state = "marker_or_end"
            elif self.state == "marker_or_end":
                if character == "]":
                    self.index += 1
                    self._finish_marker_list()
                    continue
                if character != "{":
                    raise ValueError("invalid_payload")
                decoded = self._raw_decode_with_delimiter(self.index, ",]")
                if decoded is None:
                    break
                marker, end = decoded
                identity = _validate_externalized_persisted_output_marker(marker)
                self.operation_budget.charge()
                if self.marker_store is None:
                    self.marker_store = _ExternalizedMarkerIdentityStore(
                        stat_identity=self.stat_identity,
                        runtime_state=self.operation_budget.runtime_state,
                    )
                    self.marker_store.configure_deadline(
                        self.deadline, self.deadline_error
                    )
                self.marker_store.add(identity)
                self.marker_count += 1
                if self.first_marker is None:
                    self.first_marker = dict(marker)
                self.index = end
                self.state = "marker_delimiter"
            elif self.state == "marker_delimiter":
                if character == ",":
                    self.index += 1
                    self.state = "marker_or_end"
                elif character == "]":
                    self.index += 1
                    self._finish_marker_list()
                else:
                    raise ValueError("invalid_payload")
            elif self.state == "value_delimiter":
                if character == ",":
                    self.index += 1
                    self.state = "key"
                elif character == "}":
                    self.index += 1
                    self.state = "done"
                else:
                    raise ValueError("invalid_payload")
            else:  # pragma: no cover - internal state invariant
                raise ValueError("invalid_payload")

        if self.index:
            self.buffer = self.buffer[self.index:]
            self.index = 0
        if final and (
            self.state != "done"
            or self.buffer
            or self.lexical_in_string
            or self.lexical_depth != 0
        ):
            raise ValueError("invalid_payload")

    def _pending_encoded_limit(self) -> int | None:
        """Return the fixed transport bound for the current incomplete token."""
        if self.state == "key":
            return _EXTERNALIZED_CANONICAL_KEY_MAX_ENCODED_CHARS
        if self.state == "value":
            max_chars = _EXTERNALIZED_CANONICAL_STRING_MAX_CHARS.get(
                self.current_key
            )
            if max_chars is not None:
                return (
                    _EXTERNALIZED_JSON_STRING_ESCAPE_FACTOR * max_chars
                    + 2
                    + 4_096
                )
            return _EXTERNALIZED_CANONICAL_NUMBER_MAX_ENCODED_CHARS + 4_096
        if self.state == "marker_or_end":
            return _EXTERNALIZED_CANONICAL_MARKER_MAX_ENCODED_CHARS
        if self.state in {
            "root_start", "start", "colon", "content_start",
            "marker_list_start", "marker_delimiter", "value_delimiter", "done",
        }:
            return 4_096
        return None

    def feed(self, text: str, *, final: bool = False) -> None:
        self._scan_depth(text)
        self.buffer += text
        self._process(final=final)
        pending_limit = self._pending_encoded_limit()
        if pending_limit is not None and len(self.buffer) > pending_limit:
            raise ValueError("invalid_payload")

    def configure_deadline(self, deadline: float | None, error: str) -> None:
        self.deadline = deadline
        self.deadline_error = error
        if self.marker_store is not None:
            self.marker_store.configure_deadline(deadline, error)

    def close(self) -> None:
        if self.marker_store is not None:
            self.marker_store.close()
            self.marker_store = None

    @property
    def content_ready(self) -> bool:
        return self.state == "content"


def _externalized_suffix_metadata(
    text: str, *, seen: set[str]
) -> dict[str, Any]:
    """Parse canonical post-content metadata through the streaming validator."""
    parser = _ExternalizedSuffixParser(seen=seen)
    parser.feed(text, final=True)
    return parser.fields


def _stream_externalized_prefix_authorization(
    handle,
    *,
    max_bytes: int,
    deadline: float | None,
    operation_budget: _ExternalizedSuffixOperationBudget,
    allowed_session_ids: frozenset[str],
) -> tuple[dict[str, Any], set[str], bool, bool, bytes]:
    """Validate pre-content metadata without retaining accumulated arrays.

    Returns fields, all seen top-level keys, whether the opening content quote
    was consumed, whether the byte bound interrupted metadata parsing, and any
    already-charged bytes following that quote. Reads remain one byte wide
    until ownership is known; authorized metadata then uses bounded chunks and
    replays any over-read bytes directly into the content stream.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    parser = _ExternalizedSuffixParser(
        seen=set(),
        operation_budget=operation_budget,
        prefix=True,
    )
    bytes_read = 0
    read_limit = max(0, int(max_bytes))
    next_deadline_check = 0
    while bytes_read < read_limit and not parser.content_ready:
        if bytes_read >= next_deadline_check:
            if deadline is not None and _external_metadata_now() >= deadline:
                raise TimeoutError("metadata_deadline")
            next_deadline_check = (
                bytes_read + _LCM_EXTERNAL_METADATA_DEADLINE_CHECK_BYTES
            )
        payload_session_id = parser.fields.get("session_id")
        if isinstance(payload_session_id, str) and payload_session_id:
            if payload_session_id not in allowed_session_ids:
                return parser.fields, parser.seen, False, False, b""
            read_size = min(16 * 1024, read_limit - bytes_read)
        else:
            read_size = 1
        raw = handle.read(read_size)
        if not raw:
            break
        bytes_read += len(raw)
        try:
            decoded = decoder.decode(raw, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            parser.feed(decoded)
    if parser.content_ready:
        pending_bytes, _ = decoder.getstate()
        replay = parser.buffer.encode("utf-8") + pending_bytes
        parser.buffer = ""
        return parser.fields, parser.seen, True, False, replay
    truncated = bytes_read >= read_limit
    if not truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            parser.feed(final_text)
    return parser.fields, parser.seen, False, truncated, b""


def _stream_externalized_json_content(
    handle,
    *,
    max_payload_chars: int,
    max_bytes: int,
    deadline: float | None,
    seen_keys: set[str],
    suffix_operation_budget: _ExternalizedSuffixOperationBudget | None = None,
) -> tuple[str, int, bool, dict[str, Any], int, int]:
    """Stream content and structurally validate a bounded canonical suffix."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pieces: list[str] = []
    kept_chars = 0
    bytes_read = 0
    closed = False
    escaped = False
    unicode_digits = ""
    pending_high_surrogate: int | None = None
    suffix_parser = _ExternalizedSuffixParser(
        seen=seen_keys,
        operation_budget=suffix_operation_budget,
    )
    total_content_chars = 0
    total_content_bytes = 0

    def check_deadline() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("body_deadline")

    def emit(character: str) -> None:
        nonlocal kept_chars, total_content_chars, total_content_bytes
        total_content_chars += 1
        total_content_bytes += len(character.encode("utf-8", errors="surrogatepass"))
        if kept_chars < max_payload_chars:
            pieces.append(character)
            kept_chars += 1

    def emit_codepoint(value: int) -> None:
        nonlocal pending_high_surrogate
        if 0xD800 <= value <= 0xDBFF:
            if pending_high_surrogate is not None:
                emit(chr(pending_high_surrogate))
            pending_high_surrogate = value
        elif 0xDC00 <= value <= 0xDFFF and pending_high_surrogate is not None:
            emit(chr(0x10000 + ((pending_high_surrogate - 0xD800) << 10) + value - 0xDC00))
            pending_high_surrogate = None
        else:
            if pending_high_surrogate is not None:
                emit(chr(pending_high_surrogate))
                pending_high_surrogate = None
            emit(chr(value))

    while bytes_read < max_bytes:
        check_deadline()
        raw = handle.read(min(16 * 1024, max_bytes - bytes_read))
        if not raw:
            break
        bytes_read += len(raw)
        decoded = decoder.decode(raw, final=False)
        suffix_piece: list[str] = []
        for index, character in enumerate(decoded):
            if index % 4096 == 0:
                check_deadline()
            if closed:
                suffix_piece.append(character)
            elif unicode_digits:
                if character not in "0123456789abcdefABCDEF":
                    raise ValueError("invalid_payload")
                unicode_digits += character
                if len(unicode_digits) == 5:
                    emit_codepoint(int(unicode_digits[1:], 16))
                    unicode_digits = ""
                    escaped = False
            elif escaped:
                if character == "u":
                    unicode_digits = "u"
                else:
                    escaped = False
                    mapped = {
                        '"': '"', "\\": "\\", "/": "/", "b": "\b",
                        "f": "\f", "n": "\n", "r": "\r", "t": "\t",
                    }.get(character)
                    if mapped is None:
                        raise ValueError("invalid_payload")
                    emit(mapped)
            elif character == "\\":
                escaped = True
            elif character == '"':
                if pending_high_surrogate is not None:
                    emit(chr(pending_high_surrogate))
                    pending_high_surrogate = None
                closed = True
            elif ord(character) < 0x20:
                raise ValueError("invalid_payload")
            else:
                emit(character)
        if suffix_piece:
            try:
                suffix_parser.feed("".join(suffix_piece))
            except _ExternalizedSuffixBudgetExceeded:
                return (
                    "".join(pieces), bytes_read, False, {},
                    total_content_chars, total_content_bytes,
                )
        check_deadline()

    if bytes_read < max_bytes:
        final = decoder.decode(b"", final=True)
        if final:
            try:
                suffix_parser.feed(final)
            except _ExternalizedSuffixBudgetExceeded:
                return (
                    "".join(pieces), bytes_read, False, {},
                    total_content_chars, total_content_bytes,
                )
    if not closed or escaped or unicode_digits:
        return "".join(pieces), bytes_read, False, {}, total_content_chars, total_content_bytes
    if bytes_read < max_bytes:
        suffix_parser.feed("", final=True)
    elif suffix_parser.state != "done":
        return "".join(pieces), bytes_read, False, {}, total_content_chars, total_content_bytes
    suffix_fields = suffix_parser.fields
    return (
        "".join(pieces),
        bytes_read,
        True,
        suffix_fields,
        total_content_chars,
        total_content_bytes,
    )


def _deadline_checked_literal_span(
    content: str, query: str, *, deadline: float | None
) -> tuple[int, int] | None:
    terms = [
        phrase or word
        for phrase, word in re.findall(r'"([^"]+)"|(\S+)', query)
        if phrase or word
    ]
    if not terms:
        return None
    first_span: tuple[int, int] | None = None
    for term_index, term in enumerate(terms):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        overlap_chars = max(64, len(term) * 2)
        cursor = 0
        overlap = ""
        found: tuple[int, int] | None = None
        while cursor < len(content):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("body_deadline")
            chunk = content[cursor:cursor + 32 * 1024]
            candidate = overlap + chunk
            match = pattern.search(candidate)
            if match is not None and match.end() > len(overlap):
                base = cursor - len(overlap)
                found = (base + match.start(), base + match.end())
                break
            cursor += len(chunk)
            overlap = candidate[-overlap_chars:]
        if found is None:
            return None
        if term_index == 0:
            first_span = found
    return first_span


def _deadline_checked_match_metadata(
    content: str, start: int, end: int, *, deadline: float | None
) -> tuple[int, int, int, int]:
    line = 1
    line_start = 0
    byte_offset = 0
    cursor = 0
    while cursor < start:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("body_deadline")
        piece = content[cursor:min(start, cursor + 4096)]
        line += piece.count("\n")
        newline = piece.rfind("\n")
        if newline >= 0:
            line_start = cursor + newline + 1
        byte_offset += len(piece.encode("utf-8"))
        cursor += len(piece)
    line_end = end
    while line_end < len(content):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("body_deadline")
        block_end = min(len(content), line_end + 4096)
        newline = content.find("\n", line_end, block_end)
        if newline >= 0:
            line_end = newline
            break
        line_end = block_end
    return line, line_start, line_end, byte_offset


def _regex_span_worker(query: str, content: str, connection) -> None:
    try:
        match = re.search(query, content, flags=re.IGNORECASE)
        connection.send(match.span() if match is not None else None)
    except BaseException as exc:
        connection.send({"error": type(exc).__name__})
    finally:
        connection.close()


def _bounded_regex_span(
    query: str,
    content: str,
    *,
    timeout_seconds: float,
) -> tuple[tuple[int, int] | None, str]:
    """Run stdlib regex in a child that can be terminated on CPU deadline."""
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX fallback
        context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_regex_span_worker, args=(query, content, child))
    process.daemon = True
    process.start()
    child.close()
    process.join(max(0.001, float(timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(0.1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(0.1)
        parent.close()
        return None, "regex_timeout"
    try:
        payload = parent.recv() if parent.poll() else None
    except (EOFError, OSError):
        payload = None
    finally:
        parent.close()
        process.close()
    if isinstance(payload, dict):
        return None, f"regex_error:{payload.get('error', 'unknown')}"
    if payload is None:
        return None, ""
    return (int(payload[0]), int(payload[1])), ""


_EXTERNALIZED_CONTINUATION_MAX_FILES = 4
_EXTERNALIZED_CONTINUATION_TTL_SECONDS = 5 * 60.0
_EXTERNALIZED_CONTINUATION_STATE_GUARD = threading.Lock()
_EXTERNALIZED_SCHEDULER_MAX_QUERIES = 8
_EXTERNALIZED_SCHEDULER_TTL_SECONDS = 5 * 60.0
_EXTERNALIZED_SCHEDULER_FAIRNESS_SECONDS = 15.0
_EXTERNALIZED_SCHEDULER_MAX_DISK_QUERIES = 128
_EXTERNALIZED_SCHEDULER_MAX_BYTES = 1024 * 1024


class _ExternalizedDiscoveryScheduler:
    """Bounded deterministic cursor for one root and no-ref search shape."""

    __slots__ = (
        "key", "root_identity", "listing_identity", "refs", "cursor",
        "active_ref", "listing_cookie", "next_listing_cookie",
        "listing_complete", "cached_at", "updated_wall", "blocked",
    )

    def __init__(
        self,
        *,
        key: str = "",
        root_identity: tuple[int, ...],
        refs: tuple[str, ...],
        cached_at: float,
        listing_identity: str = "",
        cursor: int = 0,
        active_ref: str | None = None,
        listing_cookie: int = 0,
        next_listing_cookie: int = 0,
        listing_complete: bool = False,
        updated_wall: float | None = None,
    ):
        self.key = key
        self.root_identity = root_identity
        self.listing_identity = listing_identity
        self.refs = refs
        self.cursor = max(0, min(int(cursor), len(refs)))
        self.active_ref = active_ref if active_ref in refs else None
        self.listing_cookie = max(0, int(listing_cookie))
        self.next_listing_cookie = max(0, int(next_listing_cookie))
        self.listing_complete = bool(listing_complete)
        self.cached_at = cached_at
        self.updated_wall = time.time() if updated_wall is None else updated_wall
        self.blocked = False


def _externalized_scheduler_listing_identity(refs: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for ref in refs:
        digest.update(ref.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def _externalized_scheduler_connection(
    runtime: _ExternalizedPrivateRuntimeState,
) -> sqlite3.Connection | None:
    """Open the disposable scheduler DB; callers decide whether to fail closed."""
    with runtime.lock:
        if runtime.closed:
            return None
        if runtime.scheduler_connection is not None:
            return runtime.scheduler_connection
        try:
            owner_dir = runtime.ensure_owner_dir()
            path = owner_dir / "scheduler.sqlite3"
            connection = sqlite3.connect(
                str(path),
                timeout=0.05,
                check_same_thread=False,
                cached_statements=4,
            )
            connection.execute("PRAGMA page_size=4096")
            connection.execute("PRAGMA auto_vacuum=FULL")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA cache_spill=OFF")
            connection.execute("PRAGMA cache_size=-64")
            connection.execute(
                f"PRAGMA max_page_count={_EXTERNALIZED_SCHEDULER_MAX_BYTES // 4096}"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS scheduler_cursors ("
                "shape_key TEXT PRIMARY KEY, root_identity TEXT NOT NULL, "
                "listing_identity TEXT NOT NULL, cursor INTEGER NOT NULL, "
                "active_ref TEXT, listing_cookie INTEGER NOT NULL, "
                "updated_wall REAL NOT NULL) WITHOUT ROWID"
            )
            columns = {
                str(row[1]) for row in connection.execute(
                    "PRAGMA table_info(scheduler_cursors)"
                )
            }
            if "listing_cookie" not in columns:
                connection.execute(
                    "ALTER TABLE scheduler_cursors ADD COLUMN "
                    "listing_cookie INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute("DELETE FROM scheduler_cursors")
            connection.commit()
        except (OSError, sqlite3.Error, _ExternalizedStateUnavailable):
            try:
                connection.close()  # type: ignore[possibly-undefined]
            except (NameError, sqlite3.Error):
                pass
            logger.debug(
                "LCM scheduler private disk state is unavailable",
                exc_info=True,
            )
            return None
        runtime.scheduler_connection = connection
        return connection


def _externalized_scheduler_reserve_locked(
    runtime: _ExternalizedPrivateRuntimeState,
    *,
    key: str,
    root_identity: tuple[int, ...],
) -> tuple[bool, int]:
    """Durably reserve one no-ref shape before payload candidate discovery."""
    connection = _externalized_scheduler_connection(runtime)
    if connection is None:
        return False, 0
    try:
        _externalized_scheduler_cleanup_disk_locked(
            connection, protected_keys=frozenset({key})
        )
        row = connection.execute(
            "SELECT root_identity, listing_cookie FROM scheduler_cursors "
            "WHERE shape_key = ?", (key,)
        ).fetchone()
        encoded_root = json.dumps(root_identity, separators=(",", ":"))
        if row is None:
            count = int(connection.execute(
                "SELECT COUNT(*) FROM scheduler_cursors"
            ).fetchone()[0])
            if count >= _EXTERNALIZED_SCHEDULER_MAX_DISK_QUERIES:
                return False, 0
            connection.execute(
                "INSERT INTO scheduler_cursors("
                "shape_key, root_identity, listing_identity, cursor, "
                "active_ref, listing_cookie, updated_wall) "
                "VALUES (?, ?, '', 0, NULL, 0, ?)",
                (
                    key,
                    encoded_root,
                    time.time(),
                ),
            )
            listing_cookie = 0
        elif str(row[0]) != encoded_root:
            # Directory mtime/ctime/identity changes invalidate every native
            # cookie and in-slice cursor. Reset before touching the listing.
            connection.execute(
                "UPDATE scheduler_cursors SET root_identity = ?, "
                "listing_identity = '', cursor = 0, active_ref = NULL, "
                "listing_cookie = 0, updated_wall = ? WHERE shape_key = ?",
                (encoded_root, time.time(), key),
            )
            listing_cookie = 0
        else:
            listing_cookie = max(0, int(row[1]))
        connection.commit()
        return True, listing_cookie
    except (TypeError, ValueError, sqlite3.Error):
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        logger.debug("LCM could not reserve scheduler cursor", exc_info=True)
        return False, 0


def _externalized_scheduler_cleanup_disk_locked(
    connection: sqlite3.Connection,
    *,
    protected_keys: frozenset[str],
) -> None:
    cutoff = time.time() - _EXTERNALIZED_SCHEDULER_TTL_SECONDS
    if protected_keys:
        placeholders = ",".join("?" for _ in protected_keys)
        connection.execute(
            "DELETE FROM scheduler_cursors WHERE updated_wall < ? "
            f"AND shape_key NOT IN ({placeholders})",
            (cutoff, *sorted(protected_keys)),
        )
    else:
        connection.execute(
            "DELETE FROM scheduler_cursors WHERE updated_wall < ?", (cutoff,)
        )
    connection.commit()


def _externalized_scheduler_persist_locked(
    runtime: _ExternalizedPrivateRuntimeState,
    state: _ExternalizedDiscoveryScheduler,
    *,
    protected_keys: frozenset[str],
) -> bool:
    connection = _externalized_scheduler_connection(runtime)
    if connection is None:
        return False
    try:
        _externalized_scheduler_cleanup_disk_locked(
            connection, protected_keys=protected_keys
        )
        exists = connection.execute(
            "SELECT 1 FROM scheduler_cursors WHERE shape_key = ?", (state.key,)
        ).fetchone()
        if exists is None:
            count = connection.execute(
                "SELECT COUNT(*) FROM scheduler_cursors"
            ).fetchone()[0]
            # Never evict a non-expired/live shape to make room. A new shape
            # beyond the cap must fail closed; it cannot use memory fairness.
            if int(count) >= _EXTERNALIZED_SCHEDULER_MAX_DISK_QUERIES:
                return False
        state.updated_wall = time.time()
        connection.execute(
            "INSERT INTO scheduler_cursors("
            "shape_key, root_identity, listing_identity, cursor, active_ref, "
            "listing_cookie, updated_wall"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(shape_key) DO UPDATE SET "
            "root_identity=excluded.root_identity, "
            "listing_identity=excluded.listing_identity, cursor=excluded.cursor, "
            "active_ref=excluded.active_ref, "
            "listing_cookie=excluded.listing_cookie, "
            "updated_wall=excluded.updated_wall",
            (
                state.key,
                json.dumps(state.root_identity, separators=(",", ":")),
                state.listing_identity,
                state.cursor,
                state.active_ref,
                state.listing_cookie,
                state.updated_wall,
            ),
        )
        connection.commit()
        return True
    except sqlite3.Error:
        logger.debug("LCM could not persist scheduler cursor", exc_info=True)
        return False


def _externalized_scheduler_load_locked(
    runtime: _ExternalizedPrivateRuntimeState,
    *,
    key: str,
    root_identity: tuple[int, ...],
    listing_identity: str,
    refs: tuple[str, ...],
    listing_cookie: int,
    next_listing_cookie: int,
    listing_complete: bool,
    now: float,
) -> _ExternalizedDiscoveryScheduler | None:
    connection = _externalized_scheduler_connection(runtime)
    if connection is None:
        return None
    try:
        row = connection.execute(
            "SELECT root_identity, listing_identity, cursor, active_ref, "
            "listing_cookie, updated_wall "
            "FROM scheduler_cursors WHERE shape_key = ?",
            (key,),
        ).fetchone()
    except sqlite3.Error:
        logger.debug("LCM could not load scheduler cursor", exc_info=True)
        return None
    if row is None:
        return None
    try:
        stored_root = tuple(int(item) for item in json.loads(row[0]))
        stored_cookie = int(row[4])
        updated_wall = float(row[5])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        stored_root != root_identity
        or stored_cookie != listing_cookie
        or str(row[1]) != listing_identity
        or time.time() - updated_wall > _EXTERNALIZED_SCHEDULER_TTL_SECONDS
    ):
        try:
            connection.execute(
                "DELETE FROM scheduler_cursors WHERE shape_key = ?", (key,)
            )
            connection.commit()
        except sqlite3.Error:
            pass
        return None
    return _ExternalizedDiscoveryScheduler(
        key=key,
        root_identity=root_identity,
        listing_identity=listing_identity,
        refs=refs,
        cursor=int(row[2]),
        active_ref=row[3] if isinstance(row[3], str) else None,
        listing_cookie=listing_cookie,
        next_listing_cookie=next_listing_cookie,
        listing_complete=listing_complete,
        cached_at=now,
        updated_wall=updated_wall,
    )


def _externalized_scheduler_min_live_cursor_locked(
    runtime: _ExternalizedPrivateRuntimeState,
    state: _ExternalizedDiscoveryScheduler,
) -> tuple[int, int]:
    connection = _externalized_scheduler_connection(runtime)
    if connection is None:
        raise _ExternalizedStateUnavailable("scheduler state is unavailable")
    try:
        row = connection.execute(
            "SELECT MIN(cursor), COUNT(*) FROM scheduler_cursors "
            "WHERE root_identity = ? AND listing_identity = ? AND updated_wall >= ?",
            (
                json.dumps(state.root_identity, separators=(",", ":")),
                state.listing_identity,
                time.time() - _EXTERNALIZED_SCHEDULER_FAIRNESS_SECONDS,
            ),
        ).fetchone()
    except sqlite3.Error as exc:
        raise _ExternalizedStateUnavailable(
            "scheduler fairness state is unreadable"
        ) from exc
    if row is None or row[0] is None:
        return state.cursor, 1
    return int(row[0]), int(row[1])


def _externalized_scheduler_ref_still_live_locked(
    engine: "LCMEngine",
    state: _ExternalizedDiscoveryScheduler,
    ref: str,
) -> bool:
    if not state.listing_identity:
        return False
    runtime = _externalized_runtime_state(engine)
    connection = _externalized_scheduler_connection(runtime)
    if connection is None:
        return False
    try:
        ref_index = state.refs.index(ref)
    except ValueError:
        return False
    try:
        row = connection.execute(
            "SELECT 1 FROM scheduler_cursors WHERE root_identity = ? "
            "AND listing_identity = ? AND updated_wall >= ? "
            "AND (cursor <= ? OR active_ref = ?) LIMIT 1",
            (
                json.dumps(state.root_identity, separators=(",", ":")),
                state.listing_identity,
                time.time() - _EXTERNALIZED_SCHEDULER_FAIRNESS_SECONDS,
                ref_index,
                ref,
            ),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _externalized_python_retained_bytes(
    value: Any, seen: set[int] | None = None
) -> int:
    """Recursively account for retained Python containers and slot fields."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _externalized_python_retained_bytes(key, seen)
            + _externalized_python_retained_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(
            _externalized_python_retained_bytes(item, seen) for item in value
        )
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"} or not hasattr(value, slot):
                continue
            size += _externalized_python_retained_bytes(
                getattr(value, slot), seen
            )
    if hasattr(value, "__dict__"):
        size += _externalized_python_retained_bytes(vars(value), seen)
    return size


class _ExternalizedContentContinuation:
    """Bounded, seek-safe state for content and trailing metadata parsing."""

    __slots__ = (
        "decoder", "max_payload_chars", "pieces", "kept_chars",
        "closed", "escaped", "unicode_digits", "pending_high_surrogate",
        "suffix_parser", "total_content_chars", "total_content_bytes",
        "pending_text",
    )

    def __init__(
        self,
        *,
        seen_keys: set[str],
        max_payload_chars: int,
        operation_budget: _ExternalizedSuffixOperationBudget,
        stat_identity: tuple[int, ...] = (),
    ):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.max_payload_chars = max(0, int(max_payload_chars))
        self.pieces: list[str] = []
        self.kept_chars = 0
        self.closed = False
        self.escaped = False
        self.unicode_digits = ""
        self.pending_high_surrogate: int | None = None
        self.suffix_parser = _ExternalizedSuffixParser(
            seen=seen_keys,
            operation_budget=operation_budget,
            stat_identity=stat_identity,
        )
        self.total_content_chars = 0
        self.total_content_bytes = 0
        self.pending_text = ""

    def _emit(self, character: str) -> None:
        self.total_content_chars += 1
        self.total_content_bytes += len(
            character.encode("utf-8", errors="surrogatepass")
        )
        if self.kept_chars < self.max_payload_chars:
            self.pieces.append(character)
            self.kept_chars += 1

    def _emit_codepoint(self, value: int) -> None:
        if 0xD800 <= value <= 0xDBFF:
            if self.pending_high_surrogate is not None:
                self._emit(chr(self.pending_high_surrogate))
            self.pending_high_surrogate = value
        elif (
            0xDC00 <= value <= 0xDFFF
            and self.pending_high_surrogate is not None
        ):
            self._emit(chr(
                0x10000
                + ((self.pending_high_surrogate - 0xD800) << 10)
                + value
                - 0xDC00
            ))
            self.pending_high_surrogate = None
        else:
            if self.pending_high_surrogate is not None:
                self._emit(chr(self.pending_high_surrogate))
                self.pending_high_surrogate = None
            self._emit(chr(value))

    def _feed_segment(self, text: str) -> None:
        suffix_piece: list[str] = []
        for character in text:
            if self.closed:
                suffix_piece.append(character)
            elif self.unicode_digits:
                if character not in "0123456789abcdefABCDEF":
                    raise ValueError("invalid_payload")
                self.unicode_digits += character
                if len(self.unicode_digits) == 5:
                    self._emit_codepoint(int(self.unicode_digits[1:], 16))
                    self.unicode_digits = ""
                    self.escaped = False
            elif self.escaped:
                if character == "u":
                    self.unicode_digits = "u"
                else:
                    self.escaped = False
                    mapped = {
                        '"': '"', "\\": "\\", "/": "/", "b": "\b",
                        "f": "\f", "n": "\n", "r": "\r", "t": "\t",
                    }.get(character)
                    if mapped is None:
                        raise ValueError("invalid_payload")
                    self._emit(mapped)
            elif character == "\\":
                self.escaped = True
            elif character == '"':
                if self.pending_high_surrogate is not None:
                    self._emit(chr(self.pending_high_surrogate))
                    self.pending_high_surrogate = None
                self.closed = True
            elif ord(character) < 0x20:
                raise ValueError("invalid_payload")
            else:
                self._emit(character)
        if suffix_piece:
            try:
                self.suffix_parser.feed("".join(suffix_piece))
            except _ExternalizedSuffixBudgetExceeded as exc:
                raise ValueError("payload_truncated") from exc

    def feed_text(self, text: str, *, deadline: float | None) -> None:
        self.pending_text += text
        while self.pending_text:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("body_deadline")
            segment = self.pending_text[:4096]
            self.pending_text = self.pending_text[len(segment):]
            self._feed_segment(segment)

    def feed_raw(self, raw: bytes, *, deadline: float | None) -> None:
        try:
            decoded = self.decoder.decode(raw, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            self.feed_text(decoded, deadline=deadline)

    def finish(self, *, deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("body_deadline")
        try:
            final_text = self.decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            self.feed_text(final_text, deadline=deadline)
        if (
            not self.closed
            or self.escaped
            or self.unicode_digits
            or self.pending_text
        ):
            raise ValueError("invalid_payload")
        self.suffix_parser.feed("", final=True)

    @property
    def content(self) -> str:
        return "".join(self.pieces)

    def retained_bytes(self) -> int:
        return _externalized_python_retained_bytes(self)

    def close(self) -> None:
        self.suffix_parser.close()


class _ExternalizedPayloadContinuation:
    """Stat-bound parser checkpoint that never retains or replays raw prefixes."""

    __slots__ = (
        "identity", "allowed_session_ids", "max_payload_chars", "offset",
        "phase", "prefix_decoder", "prefix_parser", "content_state",
        "metadata_fields", "completed", "cached_at",
    )

    def __init__(
        self,
        *,
        identity: tuple[int, ...],
        allowed_session_ids: frozenset[str],
        max_payload_chars: int,
        operation_budget: _ExternalizedSuffixOperationBudget,
    ):
        self.identity = identity
        self.allowed_session_ids = allowed_session_ids
        self.max_payload_chars = max(0, int(max_payload_chars))
        self.offset = 0
        self.phase = "metadata"
        self.prefix_decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.prefix_parser = _ExternalizedSuffixParser(
            seen=set(),
            operation_budget=operation_budget,
            prefix=True,
            stat_identity=identity,
        )
        self.content_state: _ExternalizedContentContinuation | None = None
        self.metadata_fields: dict[str, Any] = {}
        self.completed = False
        self.cached_at = 0.0

    def compatible(
        self,
        *,
        identity: tuple[int, ...],
        allowed_session_ids: frozenset[str],
        max_payload_chars: int,
    ) -> bool:
        return (
            self.identity == identity
            and self.allowed_session_ids == allowed_session_ids
            and self.max_payload_chars == max(0, int(max_payload_chars))
        )

    def _authorize_or_transition(
        self,
        *,
        operation_budget: _ExternalizedSuffixOperationBudget,
        deadline: float | None,
    ) -> None:
        payload_session_id = self.prefix_parser.fields.get("session_id")
        if isinstance(payload_session_id, str) and payload_session_id:
            if payload_session_id not in self.allowed_session_ids:
                raise ValueError("session_mismatch")
        if not self.prefix_parser.content_ready:
            return
        if not isinstance(payload_session_id, str) or not payload_session_id:
            raise ValueError("session_metadata_unavailable")
        self.metadata_fields = dict(self.prefix_parser.fields)
        buffered_text = self.prefix_parser.buffer
        self.prefix_parser.buffer = ""
        decoder_state = self.prefix_decoder.getstate()
        content_state = _ExternalizedContentContinuation(
            seen_keys=self.prefix_parser.seen,
            max_payload_chars=self.max_payload_chars,
            operation_budget=operation_budget,
            stat_identity=self.identity,
        )
        content_state.suffix_parser.configure_deadline(deadline, "body_deadline")
        content_state.decoder.setstate(decoder_state)
        self.content_state = content_state
        self.phase = "body"
        if buffered_text:
            content_state.feed_text(buffered_text, deadline=deadline)

    def _feed_metadata(
        self,
        raw: bytes,
        *,
        operation_budget: _ExternalizedSuffixOperationBudget,
        deadline: float | None,
    ) -> None:
        try:
            decoded = self.prefix_decoder.decode(raw, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            self.prefix_parser.feed(decoded)
        self._authorize_or_transition(
            operation_budget=operation_budget,
            deadline=deadline,
        )

    def _finish_metadata(
        self,
        *,
        operation_budget: _ExternalizedSuffixOperationBudget,
        deadline: float | None,
    ) -> None:
        try:
            final_text = self.prefix_decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            self.prefix_parser.feed(final_text)
        self._authorize_or_transition(
            operation_budget=operation_budget,
            deadline=deadline,
        )
        if self.phase != "body":
            payload_session_id = self.prefix_parser.fields.get("session_id")
            if not isinstance(payload_session_id, str) or not payload_session_id:
                raise ValueError("session_metadata_unavailable")
            raise ValueError("content_not_in_prefix")

    def resume(
        self,
        handle,
        *,
        file_size: int,
        byte_budget: _ExternalizedByteOperationBudget,
        operation_budget: _ExternalizedSuffixOperationBudget,
        deadline: float | None,
    ) -> bool:
        # Completion freezes every parser/decoder field. Completed checkpoints
        # are consequently safe for concurrent read-only matching, while every
        # incomplete checkpoint is checked out of the cache before mutation.
        if self.completed:
            return True
        self.prefix_parser.operation_budget = operation_budget
        self.prefix_parser.configure_deadline(deadline, "metadata_deadline")
        if self.content_state is not None:
            self.content_state.suffix_parser.operation_budget = operation_budget
            self.content_state.suffix_parser.configure_deadline(
                deadline, "body_deadline"
            )
        handle.seek(self.offset)
        while self.offset < file_size:
            if deadline is not None:
                now = (
                    _external_metadata_now()
                    if self.phase == "metadata"
                    else time.monotonic()
                )
                if now >= deadline:
                    raise TimeoutError(
                        "metadata_deadline"
                        if self.phase == "metadata"
                        else "body_deadline"
                    )
            if byte_budget.remaining <= 0:
                return False
            read_size = 16 * 1024
            if self.phase == "metadata":
                payload_session_id = self.prefix_parser.fields.get("session_id")
                if not isinstance(payload_session_id, str) or not payload_session_id:
                    read_size = 1
            read_size = min(
                read_size,
                file_size - self.offset,
            )
            raw = byte_budget.read(handle, read_size)
            if not raw:
                return False
            self.offset += len(raw)
            if self.phase == "metadata":
                self._feed_metadata(
                    raw,
                    operation_budget=operation_budget,
                    deadline=deadline,
                )
            else:
                assert self.content_state is not None
                self.content_state.feed_raw(raw, deadline=deadline)

        if self.phase == "metadata":
            self._finish_metadata(
                operation_budget=operation_budget,
                deadline=deadline,
            )
        assert self.content_state is not None
        self.content_state.finish(deadline=deadline)
        self.completed = True
        return True

    def retained_bytes(self) -> int:
        return _externalized_python_retained_bytes(self)

    def close(self) -> None:
        self.prefix_parser.close()
        if self.content_state is not None:
            self.content_state.close()


def _externalized_continuation_cache(
    engine: "LCMEngine",
) -> dict[str, _ExternalizedPayloadContinuation]:
    runtime = _externalized_runtime_state(engine)
    with _EXTERNALIZED_CONTINUATION_STATE_GUARD:
        cache = getattr(engine, "_externalized_grep_continuations", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(engine, "_externalized_grep_continuations", cache)
        setattr(engine, "_externalized_grep_continuations_lock", runtime.lock)
    return cache


def _externalized_continuation_lock(engine: "LCMEngine"):
    _externalized_continuation_cache(engine)
    return engine._externalized_grep_continuations_lock


def _externalized_scheduler_cache(
    engine: "LCMEngine",
) -> dict[str, _ExternalizedDiscoveryScheduler]:
    _externalized_continuation_cache(engine)
    cache = getattr(engine, "_externalized_grep_schedulers", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(engine, "_externalized_grep_schedulers", cache)
    return cache


def _externalized_scheduler_shape_key(
    *,
    root: Path,
    query: str,
    regex_mode: bool,
    allowed_session_ids: frozenset[str],
    limit: int,
    max_files: int,
    max_payload_chars: int,
) -> str:
    shape = json.dumps(
        {
            "root": str(root),
            "query": query,
            "regex": bool(regex_mode),
            "sessions": sorted(allowed_session_ids),
            "limit": int(limit),
            "max_files": int(max_files),
            "max_payload_chars": int(max_payload_chars),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(shape).hexdigest()


def _externalized_bounded_sorted_refs(
    root: Path,
    *,
    listing_cookie: int,
    max_files: int,
    deadline: float | None,
) -> tuple[tuple[str, ...], int, bool, int]:
    """Read one bounded Linux directory-cookie slice without prefix rescans."""
    if sys.platform != "linux":
        raise _ExternalizedStateUnavailable(
            "lossless bounded directory continuation is unavailable"
        )
    try:
        import ctypes
    except ImportError as exc:  # pragma: no cover - CPython always supplies it
        raise _ExternalizedStateUnavailable(
            "native directory continuation is unavailable"
        ) from exc
    if ctypes.sizeof(ctypes.c_long) != 8 or ctypes.sizeof(ctypes.c_ulong) != 8:
        raise _ExternalizedStateUnavailable(
            "native directory cookie ABI is unsupported"
        )

    class _LinuxDirent(ctypes.Structure):
        _fields_ = [
            ("d_ino", ctypes.c_ulong),
            ("d_off", ctypes.c_long),
            ("d_reclen", ctypes.c_ushort),
            ("d_type", ctypes.c_ubyte),
            ("d_name", ctypes.c_char * 256),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    libc.fdopendir.argtypes = [ctypes.c_int]
    libc.fdopendir.restype = ctypes.c_void_p
    libc.readdir.argtypes = [ctypes.c_void_p]
    libc.readdir.restype = ctypes.POINTER(_LinuxDirent)
    libc.telldir.argtypes = [ctypes.c_void_p]
    libc.telldir.restype = ctypes.c_long
    libc.seekdir.argtypes = [ctypes.c_void_p, ctypes.c_long]
    libc.seekdir.restype = None
    libc.closedir.argtypes = [ctypes.c_void_p]
    libc.closedir.restype = ctypes.c_int

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    opened_identity = _externalized_file_identity(os.fstat(descriptor))
    directory = libc.fdopendir(descriptor)
    if not directory:
        error = ctypes.get_errno()
        os.close(descriptor)
        raise OSError(error, os.strerror(error), root)
    # fdopendir owns descriptor after success.
    selected: list[str] = []
    entries_seen = 0
    complete = True
    next_cookie = max(0, int(listing_cookie))
    try:
        if next_cookie:
            ctypes.set_errno(0)
            libc.seekdir(directory, next_cookie)
            if ctypes.get_errno() or int(libc.telldir(directory)) != next_cookie:
                raise _ExternalizedStateUnavailable(
                    "persisted directory cookie cannot be restored"
                )
        while entries_seen < max_files:
            if deadline is not None and time.monotonic() >= deadline:
                complete = False
                break
            ctypes.set_errno(0)
            entry = libc.readdir(directory)
            if not entry:
                error = ctypes.get_errno()
                if error:
                    raise OSError(error, os.strerror(error), root)
                break
            raw_name = bytes(entry.contents.d_name).split(b"\0", 1)[0]
            if raw_name in {b".", b".."}:
                continue
            entries_seen += 1
            name = os.fsdecode(raw_name)
            try:
                name.encode("utf-8")
            except UnicodeEncodeError:
                # Generated payload refs are UTF-8. An undecodable ambient
                # filename consumes the bounded visit budget but is not a ref.
                continue
            if not name.endswith(".json") or len(name) > _LCM_GREP_REF_MAX_CHARS:
                continue
            bisect.insort(selected, name)
        cookie = int(libc.telldir(directory))
        if cookie < 0:
            error = ctypes.get_errno()
            raise OSError(error or 5, os.strerror(error or 5), root)
        next_cookie = cookie
        if entries_seen >= max_files:
            complete = False
        final_identity = _externalized_file_identity(os.fstat(descriptor))
        if final_identity != opened_identity:
            raise _ExternalizedStateUnavailable(
                "directory mutated during bounded listing"
            )
        return tuple(selected), entries_seen, complete, next_cookie
    finally:
        libc.closedir(directory)


def _checkout_externalized_scheduler(
    engine: "LCMEngine",
    *,
    key: str,
    root_identity: tuple[int, ...],
    refs: tuple[str, ...],
    listing_cookie: int,
    next_listing_cookie: int,
    listing_complete: bool,
) -> _ExternalizedDiscoveryScheduler:
    cache = _externalized_scheduler_cache(engine)
    runtime = _externalized_runtime_state(engine)
    with _externalized_continuation_lock(engine):
        now = time.monotonic()
        for stale_key in [
            item_key
            for item_key, state in cache.items()
            if not isinstance(state, _ExternalizedDiscoveryScheduler)
            or now - state.cached_at > _EXTERNALIZED_SCHEDULER_TTL_SECONDS
        ]:
            cache.pop(stale_key, None)
        state = cache.get(key)
        listing_identity = _externalized_scheduler_listing_identity(refs)
        if not (
            isinstance(state, _ExternalizedDiscoveryScheduler)
            and state.root_identity == root_identity
            and state.listing_cookie == listing_cookie
            and state.refs == refs
            and state.listing_identity == listing_identity
        ):
            state = _externalized_scheduler_load_locked(
                runtime,
                key=key,
                root_identity=root_identity,
                listing_identity=listing_identity,
                refs=refs,
                listing_cookie=listing_cookie,
                next_listing_cookie=next_listing_cookie,
                listing_complete=listing_complete,
                now=now,
            )
            if state is None:
                state = _ExternalizedDiscoveryScheduler(
                    key=key,
                    root_identity=root_identity,
                    listing_identity=listing_identity,
                    refs=refs,
                    listing_cookie=listing_cookie,
                    next_listing_cookie=next_listing_cookie,
                    listing_complete=listing_complete,
                    cached_at=now,
                )
        else:
            state.next_listing_cookie = next_listing_cookie
            state.listing_complete = listing_complete
        if not state.refs and state.active_ref is None:
            state.listing_cookie = (
                0 if state.listing_complete else state.next_listing_cookie
            )
            state.listing_identity = ""
            state.cursor = 0
        state.cached_at = now
        if not _externalized_scheduler_persist_locked(
            runtime,
            state,
            protected_keys=frozenset(cache),
        ):
            raise _ExternalizedStateUnavailable(
                "scheduler cursor is not durable"
            )
        cache.pop(key, None)
        cache[key] = state
        minimum_live_cursor, live_shape_count = (
            _externalized_scheduler_min_live_cursor_locked(runtime, state)
        )
        state.blocked = (
            live_shape_count > _EXTERNALIZED_CONTINUATION_MAX_FILES
            and state.cursor > minimum_live_cursor
        )
        while len(cache) > _EXTERNALIZED_SCHEDULER_MAX_QUERIES:
            cache.pop(next(iter(cache)), None)
        return state


def _externalized_scheduler_mark_active(
    engine: "LCMEngine",
    key: str,
    state: _ExternalizedDiscoveryScheduler,
    ref: str,
) -> bool:
    cache = _externalized_scheduler_cache(engine)
    runtime = _externalized_runtime_state(engine)
    with _externalized_continuation_lock(engine):
        if runtime.closed or state.key != key or ref not in state.refs:
            return False
        previous_active_ref = state.active_ref
        state.active_ref = ref
        state.cached_at = time.monotonic()
        persisted = _externalized_scheduler_persist_locked(
            runtime,
            state,
            protected_keys=frozenset(cache) | {key},
        )
        if not persisted:
            state.active_ref = previous_active_ref
        return persisted


def _externalized_scheduler_advance(
    engine: "LCMEngine",
    key: str,
    state: _ExternalizedDiscoveryScheduler,
    ref: str,
) -> bool:
    cache = _externalized_scheduler_cache(engine)
    runtime = _externalized_runtime_state(engine)
    with _externalized_continuation_lock(engine):
        if runtime.closed or state.key != key:
            return False
        try:
            index = state.refs.index(ref, state.cursor)
        except ValueError:
            return False
        previous_cursor = state.cursor
        previous_active_ref = state.active_ref
        previous_listing_identity = state.listing_identity
        previous_listing_cookie = state.listing_cookie
        state.cursor = max(state.cursor, index + 1)
        if state.active_ref == ref:
            state.active_ref = None
        if state.cursor >= len(state.refs) and state.active_ref is None:
            state.listing_cookie = (
                0 if state.listing_complete else state.next_listing_cookie
            )
            state.listing_identity = ""
            state.cursor = 0
        state.cached_at = time.monotonic()
        persisted = _externalized_scheduler_persist_locked(
            runtime,
            state,
            protected_keys=frozenset(cache) | {key},
        )
        if not persisted:
            state.cursor = previous_cursor
            state.active_ref = previous_active_ref
            state.listing_identity = previous_listing_identity
            state.listing_cookie = previous_listing_cookie
        return persisted


def _prune_externalized_continuations_locked(
    cache: dict[str, _ExternalizedPayloadContinuation],
    *,
    now: float,
) -> None:
    expired = [
        key
        for key, continuation in cache.items()
        if not isinstance(continuation, _ExternalizedPayloadContinuation)
        or now - continuation.cached_at > _EXTERNALIZED_CONTINUATION_TTL_SECONDS
    ]
    for key in expired:
        continuation = cache.pop(key, None)
        if isinstance(continuation, _ExternalizedPayloadContinuation):
            continuation.close()


def _checkout_externalized_continuation(
    engine: "LCMEngine",
    key: str,
    *,
    identity: tuple[int, ...],
    allowed_session_ids: frozenset[str],
    max_payload_chars: int,
    file_size: int,
) -> _ExternalizedPayloadContinuation | None:
    cache = _externalized_continuation_cache(engine)
    with _externalized_continuation_lock(engine):
        now = time.monotonic()
        _prune_externalized_continuations_locked(cache, now=now)
        continuation = cache.get(key)
        if not (
            isinstance(continuation, _ExternalizedPayloadContinuation)
            and continuation.compatible(
                identity=identity,
                allowed_session_ids=allowed_session_ids,
                max_payload_chars=max_payload_chars,
            )
            and continuation.offset <= file_size
        ):
            stale = cache.pop(key, None)
            if isinstance(stale, _ExternalizedPayloadContinuation):
                stale.close()
            return None
        continuation.cached_at = now
        if not continuation.completed:
            # Exclusive checkout: no two callers can mutate one parser,
            # decoder, offset, or operation-budget reference.
            cache.pop(key, None)
        return continuation


def _externalized_file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _store_externalized_continuation(
    engine: "LCMEngine",
    key: str,
    continuation: _ExternalizedPayloadContinuation,
    *,
    expected_generation: int | None = None,
) -> bool:
    cache = _externalized_continuation_cache(engine)
    runtime = _externalized_runtime_state(engine)
    with _externalized_continuation_lock(engine):
        if runtime.closed or (
            expected_generation is not None
            and expected_generation != runtime.generation
        ):
            continuation.close()
            return False
        now = time.monotonic()
        _prune_externalized_continuations_locked(cache, now=now)
        existing = cache.get(key)
        if (
            isinstance(existing, _ExternalizedPayloadContinuation)
            and existing.compatible(
                identity=continuation.identity,
                allowed_session_ids=continuation.allowed_session_ids,
                max_payload_chars=continuation.max_payload_chars,
            )
            and (
                existing.completed and not continuation.completed
                or existing.offset > continuation.offset
            )
        ):
            existing.cached_at = now
            if existing is not continuation:
                continuation.close()
        else:
            continuation.cached_at = now
            replaced = cache.pop(key, None)
            if (
                isinstance(replaced, _ExternalizedPayloadContinuation)
                and replaced is not continuation
            ):
                replaced.close()
            cache[key] = continuation
        while len(cache) > _EXTERNALIZED_CONTINUATION_MAX_FILES:
            evicted = cache.pop(next(iter(cache)))
            if isinstance(evicted, _ExternalizedPayloadContinuation):
                evicted.close()
        return True


def _delete_externalized_continuation(
    engine: "LCMEngine",
    key: str,
    continuation: _ExternalizedPayloadContinuation | None = None,
) -> None:
    cache = _externalized_continuation_cache(engine)
    with _externalized_continuation_lock(engine):
        cached = cache.get(key)
        if continuation is not None and cached is not continuation:
            return
        removed = cache.pop(key, None)
        if isinstance(removed, _ExternalizedPayloadContinuation):
            removed.close()


def _touch_externalized_continuation(
    engine: "LCMEngine",
    key: str,
    continuation: _ExternalizedPayloadContinuation,
) -> None:
    """Mark an immutable completion used without deleting or resurrecting it."""
    cache = _externalized_continuation_cache(engine)
    with _externalized_continuation_lock(engine):
        now = time.monotonic()
        _prune_externalized_continuations_locked(cache, now=now)
        if cache.get(key) is continuation:
            continuation.cached_at = now


def _externalized_continuation_memory_bytes(
    cache: dict[str, _ExternalizedPayloadContinuation],
) -> int:
    return sum(
        continuation.retained_bytes()
        for continuation in cache.values()
        if isinstance(continuation, _ExternalizedPayloadContinuation)
    )


def _externalized_continuation_stats(
    engine: "LCMEngine",
    *,
    exclude: tuple[_ExternalizedPayloadContinuation, ...] = (),
) -> tuple[int, int]:
    cache = _externalized_continuation_cache(engine)
    with _externalized_continuation_lock(engine):
        _prune_externalized_continuations_locked(cache, now=time.monotonic())
        excluded_ids = {id(continuation) for continuation in exclude}
        visible = {
            key: continuation
            for key, continuation in cache.items()
            if id(continuation) not in excluded_ids
        }
        return len(visible), _externalized_continuation_memory_bytes(visible)


def _cleanup_externalized_runtime_state(engine: "LCMEngine") -> None:
    """Release private parser files and bounded discovery state on shutdown."""
    runtime = _externalized_runtime_state(engine)
    cache = getattr(engine, "_externalized_grep_continuations", None)
    if not isinstance(cache, dict):
        cache = {}
    lock = getattr(engine, "_externalized_grep_continuations_lock", None)
    if hasattr(lock, "__enter__"):
        context = lock
    else:
        context = _externalized_continuation_lock(engine)
    scheduler_connection: sqlite3.Connection | None = None
    close_owner = False
    with context:
        if not runtime.closed:
            runtime.closed = True
            runtime.generation += 1
        for continuation in tuple(cache.values()):
            if isinstance(continuation, _ExternalizedPayloadContinuation):
                continuation.close()
        cache.clear()
        schedulers = getattr(engine, "_externalized_grep_schedulers", None)
        if isinstance(schedulers, dict):
            schedulers.clear()
        scheduler_connection = runtime.scheduler_connection
        runtime.scheduler_connection = None
        close_owner = not runtime.active_checkouts
    if scheduler_connection is not None:
        scheduler_connection.close()
    if close_owner:
        runtime._close_owner()


class _ExternalizedContinuationCompletion:
    """Outer lcm_grep acknowledgement for one frozen parser checkpoint."""

    __slots__ = (
        "engine", "key", "continuation", "scheduler_key",
        "scheduler", "scheduler_ref",
    )

    def __init__(
        self,
        engine: "LCMEngine",
        key: str,
        continuation: _ExternalizedPayloadContinuation,
        *,
        scheduler_key: str | None = None,
        scheduler: _ExternalizedDiscoveryScheduler | None = None,
        scheduler_ref: str | None = None,
    ):
        self.engine = engine
        self.key = key
        self.continuation = continuation
        self.scheduler_key = scheduler_key
        self.scheduler = scheduler
        self.scheduler_ref = scheduler_ref

    def commit(self) -> None:
        if (
            self.scheduler_key is not None
            and self.scheduler is not None
            and self.scheduler_ref is not None
        ):
            _externalized_scheduler_advance(
                self.engine,
                self.scheduler_key,
                self.scheduler,
                self.scheduler_ref,
            )
            with _externalized_continuation_lock(self.engine):
                if _externalized_scheduler_ref_still_live_locked(
                    self.engine, self.scheduler, self.scheduler_ref
                ):
                    _touch_externalized_continuation(
                        self.engine, self.key, self.continuation
                    )
                else:
                    _delete_externalized_continuation(
                        self.engine, self.key, self.continuation
                    )
        else:
            _touch_externalized_continuation(
                self.engine, self.key, self.continuation
            )

    def preserve(self) -> None:
        _touch_externalized_continuation(
            self.engine, self.key, self.continuation
        )


def _search_externalized_payloads(
    engine: "LCMEngine",
    *,
    query: str,
    regex_mode: bool,
    allowed_session_ids: frozenset[str],
    ref: str,
    limit: int,
    max_files: int,
    max_payload_chars: int,
    max_total_bytes: int = _LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES,
    deadline: float | None = None,
    completion_acknowledgements: list[
        _ExternalizedContinuationCompletion
    ] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    runtime_state = _externalized_runtime_state(engine)
    diagnostics: list[dict[str, str]] = []
    hits: list[dict[str, Any]] = []
    files_scanned = 0
    scan_truncated = False
    effective_total_bytes = min(
        _LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES,
        max(0, int(max_total_bytes)),
    )
    byte_operation_budget = _ExternalizedByteOperationBudget(effective_total_bytes)
    continuation_reused_bytes = 0
    # Every accepted marker consumes transport bytes. Allow at most one
    # carried, previously-read marker in addition to this call's byte budget,
    # so CPU/allocation per call stays bounded without imposing a lifetime
    # marker-count limit on historical writer output.
    suffix_operation_budget = _ExternalizedSuffixOperationBudget(
        max_markers=effective_total_bytes + 1,
        runtime_state=runtime_state,
    )
    try:
        root = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=False,
        ).resolve()
    except (OSError, ValueError):
        return [], [{"ref": ref or "", "error": "storage_root_unavailable"}], {
            "files_scanned": 0,
            "bytes_scanned": 0,
            "matches": 0,
            "scan_truncated": False,
            "max_files": max_files,
            "max_payload_chars": max_payload_chars,
            "max_total_bytes": effective_total_bytes,
            "max_persisted_output_markers": None,
            "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
            "persisted_output_markers_scanned": suffix_operation_budget.markers,
            "byte_budget_exhausted": byte_operation_budget.exhausted,
            "continuation_reused_bytes": 0,
            "continuations_pending": 0,
            "continuation_memory_bytes": 0,
        }
    if not root.exists() or not root.is_dir():
        if ref:
            diagnostics.append({"ref": ref, "error": "missing"})
        return [], diagnostics, {
            "files_scanned": 0,
            "bytes_scanned": 0,
            "matches": 0,
            "scan_truncated": False,
            "max_files": max_files,
            "max_payload_chars": max_payload_chars,
            "max_total_bytes": effective_total_bytes,
            "max_persisted_output_markers": None,
            "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
            "persisted_output_markers_scanned": suffix_operation_budget.markers,
            "byte_budget_exhausted": byte_operation_budget.exhausted,
            "continuation_reused_bytes": 0,
            "continuations_pending": 0,
            "continuation_memory_bytes": 0,
        }
    regex_deadline = min(
        time.monotonic() + _LCM_GREP_REGEX_OPERATION_DEADLINE_SECONDS,
        deadline if deadline is not None else float("inf"),
    )
    regex_timeouts = 0
    candidates_seen = 0
    paths: list[Path] = []
    scheduler_key: str | None = None
    scheduler: _ExternalizedDiscoveryScheduler | None = None
    _externalized_continuation_cache(engine)

    def acknowledge(
        completion: _ExternalizedContinuationCompletion | None,
    ) -> None:
        if completion is None:
            return
        if completion_acknowledgements is None:
            completion.commit()
        else:
            completion_acknowledgements.append(completion)
    if ref:
        candidates_seen = 1
        paths.append(root / ref)
    else:
        scheduler_key = _externalized_scheduler_shape_key(
            root=root,
            query=query,
            regex_mode=regex_mode,
            allowed_session_ids=allowed_session_ids,
            limit=limit,
            max_files=max_files,
            max_payload_chars=max_payload_chars,
        )
        try:
            root_identity = _externalized_file_identity(root.stat())
            with _externalized_continuation_lock(engine):
                reserved, listing_cookie = _externalized_scheduler_reserve_locked(
                    runtime_state,
                    key=scheduler_key,
                    root_identity=root_identity,
                )
            if not reserved:
                raise _ExternalizedStateUnavailable(
                    "scheduler state cannot be reserved"
                )
        except (OSError, sqlite3.Error, _ExternalizedStateUnavailable):
            return [], [{"ref": "", "error": "private_state_unavailable"}], {
                "files_scanned": 0,
                "bytes_scanned": 0,
                "matches": 0,
                "scan_truncated": False,
                "max_files": max_files,
                "max_payload_chars": max_payload_chars,
                "max_total_bytes": effective_total_bytes,
                "max_persisted_output_markers": None,
                "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
                "persisted_output_markers_scanned": suffix_operation_budget.markers,
                "byte_budget_exhausted": byte_operation_budget.exhausted,
                "continuation_reused_bytes": 0,
                "continuations_pending": 0,
                "continuation_memory_bytes": 0,
            }
        try:
            refs, candidates_seen, listing_complete, next_listing_cookie = (
                _externalized_bounded_sorted_refs(
                    root,
                    listing_cookie=listing_cookie,
                    max_files=max_files,
                    deadline=deadline,
                )
            )
            if _externalized_file_identity(root.stat()) != root_identity:
                raise _ExternalizedStateUnavailable(
                    "directory mutated around bounded listing"
                )
        except (OSError, _ExternalizedStateUnavailable):
            return [], [{"ref": "", "error": "private_state_unavailable"}], {
                "files_scanned": 0,
                "entries_scanned": candidates_seen,
                "bytes_scanned": 0,
                "matches": 0,
                "scan_truncated": True,
                "max_files": max_files,
                "max_payload_chars": max_payload_chars,
                "max_total_bytes": effective_total_bytes,
                "max_persisted_output_markers": None,
                "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
                "persisted_output_markers_scanned": suffix_operation_budget.markers,
                "byte_budget_exhausted": byte_operation_budget.exhausted,
                "continuation_reused_bytes": 0,
                "continuations_pending": 0,
                "continuation_memory_bytes": 0,
            }
        scan_truncated = not listing_complete
        try:
            scheduler = _checkout_externalized_scheduler(
                engine,
                key=scheduler_key,
                root_identity=root_identity,
                refs=refs,
                listing_cookie=listing_cookie,
                next_listing_cookie=next_listing_cookie,
                listing_complete=listing_complete,
            )
        except _ExternalizedStateUnavailable:
            return [], [{"ref": "", "error": "private_state_unavailable"}], {
                "files_scanned": 0,
                "bytes_scanned": 0,
                "matches": 0,
                "scan_truncated": scan_truncated,
                "max_files": max_files,
                "max_payload_chars": max_payload_chars,
                "max_total_bytes": effective_total_bytes,
                "max_persisted_output_markers": None,
                "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
                "persisted_output_markers_scanned": suffix_operation_budget.markers,
                "byte_budget_exhausted": byte_operation_budget.exhausted,
                "continuation_reused_bytes": 0,
                "continuations_pending": 0,
                "continuation_memory_bytes": 0,
            }
        start_index = scheduler.cursor
        if scheduler.active_ref is not None:
            try:
                start_index = scheduler.refs.index(scheduler.active_ref)
            except ValueError:
                scheduler.active_ref = None
        if not scheduler.blocked:
            paths.extend(root / item for item in scheduler.refs[start_index:])

    for path_index, path in enumerate(paths):
        if deadline is not None and time.monotonic() >= deadline:
            scan_truncated = True
            break
        if byte_operation_budget.remaining <= 0:
            scan_truncated = True
            break
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except FileNotFoundError:
            diagnostics.append({"ref": path.name, "error": "missing"})
            if scheduler is not None and scheduler_key is not None:
                _externalized_scheduler_advance(
                    engine, scheduler_key, scheduler, path.name
                )
            continue
        except (OSError, ValueError):
            diagnostics.append({"ref": path.name, "error": "path_escape"})
            if scheduler is not None and scheduler_key is not None:
                _externalized_scheduler_advance(
                    engine, scheduler_key, scheduler, path.name
                )
            continue
        if not resolved.is_file():
            diagnostics.append({"ref": path.name, "error": "not_a_file"})
            if scheduler is not None and scheduler_key is not None:
                _externalized_scheduler_advance(
                    engine, scheduler_key, scheduler, path.name
                )
            continue
        continuation_key = str(resolved)
        continuation: _ExternalizedPayloadContinuation | None = None
        completion: _ExternalizedContinuationCompletion | None = None
        checkout_generation: int | None = None
        checkout_registered = False
        try:
            with resolved.open("rb") as raw_handle:
                opened_stat = os.fstat(raw_handle.fileno())
                if not stat.S_ISREG(opened_stat.st_mode):
                    diagnostics.append({"ref": path.name, "error": "not_a_file"})
                    if scheduler is not None and scheduler_key is not None:
                        _externalized_scheduler_advance(
                            engine, scheduler_key, scheduler, path.name
                        )
                    continue
                if scheduler is not None and scheduler_key is not None:
                    if not _externalized_scheduler_mark_active(
                        engine, scheduler_key, scheduler, path.name
                    ):
                        raise _ExternalizedStateUnavailable(
                            "scheduler active cursor is not durable"
                        )
                continuation_identity = _externalized_file_identity(opened_stat)
                continuation = _checkout_externalized_continuation(
                    engine,
                    continuation_key,
                    identity=continuation_identity,
                    allowed_session_ids=allowed_session_ids,
                    max_payload_chars=max_payload_chars,
                    file_size=int(opened_stat.st_size),
                )
                if continuation is not None:
                    continuation_reused_bytes += continuation.offset
                if continuation is None:
                    continuation = _ExternalizedPayloadContinuation(
                        identity=continuation_identity,
                        allowed_session_ids=allowed_session_ids,
                        max_payload_chars=max_payload_chars,
                        operation_budget=suffix_operation_budget,
                    )
                checkout_generation = runtime_state.register_checkout(continuation)
                if checkout_generation is None:
                    raise _ExternalizedStateUnavailable(
                        "externalized runtime state is closed"
                    )
                checkout_registered = True
                completed = continuation.resume(
                    raw_handle,
                    file_size=int(opened_stat.st_size),
                    byte_budget=byte_operation_budget,
                    operation_budget=suffix_operation_budget,
                    deadline=deadline,
                )
                if not completed:
                    scan_truncated = True
                    if continuation.offset < int(opened_stat.st_size):
                        _store_externalized_continuation(
                            engine,
                            continuation_key,
                            continuation,
                            expected_generation=checkout_generation,
                        )
                    payload_session_id = continuation.prefix_parser.fields.get(
                        "session_id"
                    )
                    diagnostics.append({
                        "ref": path.name,
                        "error": (
                            "metadata_prefix_truncated"
                            if not isinstance(payload_session_id, str)
                            or not payload_session_id
                            else "payload_truncated"
                        ),
                    })
                    continue
                if (
                    _externalized_file_identity(os.fstat(raw_handle.fileno()))
                    != continuation_identity
                ):
                    raise ValueError("invalid_payload")
                content_state = continuation.content_state
                assert content_state is not None
                seen_keys = continuation.prefix_parser.seen
                final_metadata = dict(continuation.metadata_fields)
                final_metadata.update(content_state.suffix_parser.fields)
                _validate_externalized_metadata(final_metadata)
                payload_session_id = final_metadata.get("session_id", "")
                if (
                    "session_id" not in seen_keys
                    or not isinstance(payload_session_id, str)
                    or not payload_session_id
                ):
                    diagnostics.append({
                        "ref": path.name,
                        "error": "session_metadata_unavailable",
                    })
                    if scheduler is not None and scheduler_key is not None:
                        _externalized_scheduler_advance(
                            engine, scheduler_key, scheduler, path.name
                        )
                    continue
                if payload_session_id not in allowed_session_ids:
                    diagnostics.append({"ref": path.name, "error": "session_mismatch"})
                    if scheduler is not None and scheduler_key is not None:
                        _externalized_scheduler_advance(
                            engine, scheduler_key, scheduler, path.name
                        )
                    continue
                declared_chars = final_metadata.get("content_chars")
                declared_bytes = final_metadata.get("content_bytes")
                if (
                    declared_chars is not None
                    and (
                        isinstance(declared_chars, bool)
                        or not isinstance(declared_chars, int)
                        or declared_chars != content_state.total_content_chars
                    )
                ) or (
                    declared_bytes is not None
                    and (
                        isinstance(declared_bytes, bool)
                        or not isinstance(declared_bytes, int)
                        or declared_bytes != content_state.total_content_bytes
                    )
                ):
                    diagnostics.append({"ref": path.name, "error": "invalid_payload"})
                    if scheduler is not None and scheduler_key is not None:
                        _externalized_scheduler_advance(
                            engine, scheduler_key, scheduler, path.name
                        )
                    continue
                metadata_fields = final_metadata
                content = content_state.content
                total_content_chars = content_state.total_content_chars
                total_content_bytes = content_state.total_content_bytes
                # Freeze and publish the stat-bound completed checkpoint before
                # any deadline-sensitive matching. Outer lcm_grep removes this
                # exact object only after redaction and response construction.
                _store_externalized_continuation(
                    engine,
                    continuation_key,
                    continuation,
                    expected_generation=checkout_generation,
                )
                completion = _ExternalizedContinuationCompletion(
                    engine,
                    continuation_key,
                    continuation,
                    scheduler_key=scheduler_key,
                    scheduler=scheduler,
                    scheduler_ref=path.name if scheduler is not None else None,
                )
        except TimeoutError:
            scan_truncated = True
            if continuation is not None:
                _store_externalized_continuation(
                    engine,
                    continuation_key,
                    continuation,
                    expected_generation=checkout_generation,
                )
            diagnostics.append({
                "ref": path.name,
                "error": (
                    "metadata_deadline"
                    if continuation is None or continuation.phase == "metadata"
                    else "body_deadline"
                ),
            })
            break
        except ValueError as exc:
            if continuation is not None:
                continuation.close()
            error = str(exc)
            if error == "payload_truncated":
                scan_truncated = True
            diagnostics.append({
                "ref": path.name,
                "error": error
                if error in {
                    "ambiguous_metadata", "invalid_payload", "payload_truncated",
                    "session_mismatch", "session_metadata_unavailable",
                    "content_not_in_prefix",
                }
                else "invalid_payload",
            })
            if (
                error != "payload_truncated"
                and scheduler is not None
                and scheduler_key is not None
            ):
                _externalized_scheduler_advance(
                    engine, scheduler_key, scheduler, path.name
                )
            continue
        except _ExternalizedStateUnavailable:
            if continuation is not None:
                continuation.close()
            if scheduler is None:
                diagnostics.append({"ref": path.name, "error": "unreadable"})
                continue
            hits.clear()
            diagnostics.append({"ref": "", "error": "private_state_unavailable"})
            scan_truncated = True
            break
        except (OSError, UnicodeDecodeError):
            if continuation is not None:
                continuation.close()
            diagnostics.append({"ref": path.name, "error": "unreadable"})
            if scheduler is not None and scheduler_key is not None:
                _externalized_scheduler_advance(
                    engine, scheduler_key, scheduler, path.name
                )
            continue
        finally:
            if checkout_registered and continuation is not None:
                runtime_state.return_checkout(
                    continuation, checkout_generation
                )
        files_scanned += 1
        if regex_mode:
            remaining_regex_time = regex_deadline - time.monotonic()
            if remaining_regex_time <= 0:
                scan_truncated = True
                diagnostics.append({"ref": path.name, "error": "regex_operation_deadline"})
                break
            span, regex_error = _bounded_regex_span(
                query,
                content,
                timeout_seconds=min(
                    _LCM_GREP_REGEX_FILE_DEADLINE_SECONDS,
                    remaining_regex_time,
                ),
            )
            if regex_error:
                diagnostics.append({"ref": path.name, "error": regex_error})
                if regex_error == "regex_timeout":
                    regex_timeouts += 1
                continue
            if span is None:
                acknowledge(completion)
                continue
            start, end = span
        else:
            try:
                span = _deadline_checked_literal_span(
                    content, query, deadline=deadline
                )
            except TimeoutError:
                scan_truncated = True
                diagnostics.append({"ref": path.name, "error": "body_deadline"})
                break
            if span is None:
                acknowledge(completion)
                continue
            start, end = span
        try:
            line, line_start, line_end, byte_offset = _deadline_checked_match_metadata(
                content, start, end, deadline=deadline
            )
        except TimeoutError:
            scan_truncated = True
            diagnostics.append({"ref": path.name, "error": "body_deadline"})
            break
        context_start = max(line_start, start - 120)
        context_end = min(line_end, end + 120)
        try:
            created_at = float(metadata_fields.get("created_at", opened_stat.st_mtime))
        except (OSError, TypeError, ValueError):
            created_at = 0.0
        if deadline is not None and time.monotonic() >= deadline:
            scan_truncated = True
            diagnostics.append({"ref": path.name, "error": "body_deadline"})
            break
        matched_text = content[start:end]
        snippet = content[context_start:context_end]
        safe_snippet = redact_sensitive_text(snippet, engine._config)
        if deadline is not None and time.monotonic() >= deadline:
            scan_truncated = True
            diagnostics.append({"ref": path.name, "error": "body_deadline"})
            break
        safe_matched_text = (
            matched_text
            if matched_text in safe_snippet
            else "[redacted by sensitive-pattern policy]"
        )
        hits.append({
            "type": "externalized",
            "ref": path.name,
            "session_id": payload_session_id,
            "line": line,
            "char_offset": start,
            "byte_offset": byte_offset,
            "matched_text": safe_matched_text,
            "snippet": safe_snippet,
            "payload_truncated": len(content) >= max_payload_chars,
            "content_chars_scanned": len(content),
            "_sort_ts": created_at,
            "_sort_rank": 0.0,
            "_sort_directness": 0.0,
        })
        acknowledge(completion)
        if len(hits) >= limit:
            scan_truncated = scan_truncated or path_index + 1 < len(paths)
            break
    acknowledged_continuations = (
        tuple(
            completion.continuation
            for completion in completion_acknowledgements
        )
        if completion_acknowledgements is not None
        else ()
    )
    continuations_pending, continuation_memory_bytes = (
        _externalized_continuation_stats(
            engine,
            exclude=acknowledged_continuations,
        )
    )
    return hits, diagnostics, {
        "files_scanned": files_scanned,
        "entries_scanned": candidates_seen,
        "bytes_scanned": byte_operation_budget.bytes_read,
        "matches": len(hits),
        "scan_truncated": scan_truncated,
        "max_files": max_files,
        "max_payload_chars": max_payload_chars,
        "max_total_bytes": effective_total_bytes,
        "max_persisted_output_markers": None,
        "max_suffix_depth": _EXTERNALIZED_SUFFIX_MAX_DEPTH,
        "persisted_output_markers_scanned": suffix_operation_budget.markers,
        "byte_budget_exhausted": byte_operation_budget.exhausted,
        "continuation_reused_bytes": continuation_reused_bytes,
        "continuations_pending": continuations_pending,
        "continuation_memory_bytes": continuation_memory_bytes,
        "regex_file_deadline_ms": int(_LCM_GREP_REGEX_FILE_DEADLINE_SECONDS * 1000),
        "regex_operation_deadline_ms": int(_LCM_GREP_REGEX_OPERATION_DEADLINE_SECONDS * 1000),
        "regex_timeouts": regex_timeouts,
    }


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    from .tokens import count_tokens

    if max_tokens <= 0 or not text:
        return "", bool(text)

    if count_tokens(text) <= max_tokens:
        return text, False

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best, True


def _parse_int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_non_negative_int(value: Any, default: int) -> int:
    return max(0, _parse_int_value(value, default))


def _parse_positive_int(value: Any, default: int) -> int:
    return max(1, _parse_int_value(value, default))


def _parse_optional_float(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a number"


def _parse_optional_timestamp(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if isinstance(value, (int, float)):
        try:
            return float(value), None
        except (TypeError, ValueError, OverflowError):
            return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    text = str(value).strip()
    if not text:
        return None, f"{name} must not be empty"
    try:
        return float(text), None
    except (TypeError, ValueError, OverflowError):
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, f"{name} ISO timestamp must include a timezone offset or Z"
    return parsed.timestamp(), None


def _parse_grep_role(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    role = str(value or "").strip()
    valid_roles = {"system", "user", "assistant", "tool", "unknown"}
    if role not in valid_roles:
        return None, "role must be one of: system, user, assistant, tool, unknown"
    return role, None


def _parse_bounded_grep_text(
    value: Any,
    name: str,
    *,
    default: str = "",
    max_chars: int,
) -> tuple[str | None, str | None]:
    if value is None:
        return default, None
    if not isinstance(value, str):
        return None, f"{name} must be a string"
    if len(value) > max_chars:
        return None, f"{name} exceeds the {max_chars} character hard limit"
    return value.strip(), None


def _parse_bounded_grep_int(
    value: Any,
    name: str,
    *,
    default: int,
) -> tuple[int | None, str | None]:
    if value is None:
        return default, None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None, f"{name} must be an integer"
    if isinstance(value, str) and len(value) > 32:
        return None, f"{name} exceeds the numeric character hard limit"
    try:
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


def _parse_strict_int(value: Any, name: str) -> tuple[int | None, str | None]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


_LCM_GREP_VALID_SCOPES = frozenset({"current", "all", "session"})
_LCM_GREP_HARD_LIMIT_CAP = 200
_LCM_GREP_SUMMARY_DEADLINE_SECONDS = 0.25
_LCM_LOAD_SESSION_DEFAULT_LIMIT = 100
_LCM_LOAD_SESSION_HARD_LIMIT_CAP = 200
_LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS = 4000
_LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS = 20_000
_LCM_LOAD_SESSION_MAX_NESTED_DEPTH = 12
_LCM_LOAD_SESSION_MAX_NESTED_ITEMS = 256
_LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES = 64 * 1024
_LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES = 2 * 1024 * 1024
_LCM_LOAD_SESSION_MAX_ROLES = 32
_LCM_LOAD_SESSION_MAX_ROLE_CHARS = 128
_LCM_LOAD_SESSION_MAX_SESSION_ID_CHARS = 512
_LCM_INSPECT_DEFAULT_LIMIT = 20
_LCM_INSPECT_HARD_LIMIT_CAP = 200
_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT = 10_000
_LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES = 16_384
_LCM_EXTERNAL_METADATA_DEADLINE_CHECK_BYTES = 256
_external_metadata_now = time.monotonic
_LCM_INSPECT_LINEAGE_MAX_ROWS = 10_000
_LCM_INSPECT_LINEAGE_MAX_EDGES = 40_000
_LCM_INSPECT_LINEAGE_MAX_BYTES = 4 * 1024 * 1024
_LCM_INSPECT_LINEAGE_MAX_DEPTH = 64
_LCM_INSPECT_LINEAGE_DEADLINE_SECONDS = 0.25


def _slice_content_for_response(
    content: str,
    max_tokens: int,
    content_offset: int = 0,
    *,
    config=None,
) -> dict[str, Any]:
    del config  # Mandatory output redaction is deliberately configuration-free.
    return _slice_redacted_raw_page(
        str(content or ""),
        content_offset=content_offset,
        max_tokens=max_tokens,
        max_chars=_CURRENT_SESSION_EXPAND_MAX_CHARS,
    )


def _redaction_spans_with_incomplete_tail(
    raw_text: str,
    *,
    source_truncated: bool,
) -> tuple[list[SensitiveOutputSpan], int | None]:
    """Return complete spans and the start of an unsafe truncated tail."""
    spans = sensitive_output_spans(raw_text)
    unsafe_tail_start: int | None = None
    if source_truncated:
        if spans and spans[-1].raw_end == len(raw_text):
            # A bounded SQL prefix cannot prove that a credential ending at its
            # right edge really ended there. Resume at the complete expression's
            # start so the full-source expansion can classify it safely.
            unsafe_tail_start = spans[-1].raw_start
        standalone = _BOUNDARY_STANDALONE_CREDENTIAL_RE.search(raw_text)
        pem_begin = _BOUNDARY_PRIVATE_KEY_BEGIN_RE.search(raw_text)
        for match in (standalone, pem_begin):
            if match is not None:
                unsafe_tail_start = (
                    match.start()
                    if unsafe_tail_start is None
                    else min(unsafe_tail_start, match.start())
                )
    return spans, unsafe_tail_start


def _slice_redacted_raw_page(
    raw_text: str,
    *,
    content_offset: int = 0,
    max_tokens: int | None = None,
    max_chars: int | None = None,
    source_content_chars: int | None = None,
    allow_incomplete_tail: bool = False,
    allow_partial_redaction_without_progress: bool = False,
) -> dict[str, Any]:
    """Page mandatory-redacted output while retaining raw-source cursors."""
    total_source_chars = (
        int(source_content_chars)
        if isinstance(source_content_chars, int) and source_content_chars >= 0
        else len(raw_text)
    )
    source_truncated = total_source_chars > len(raw_text)
    requested_offset = min(max(0, int(content_offset)), total_source_chars)
    local_offset = min(requested_offset, len(raw_text))
    spans, unsafe_tail_start = _redaction_spans_with_incomplete_tail(
        raw_text, source_truncated=source_truncated and allow_incomplete_tail
    )

    segments: list[tuple[int, int, str, bool]] = []
    page_raw_end = (
        unsafe_tail_start if unsafe_tail_start is not None else len(raw_text)
    )
    cursor = 0
    for span in spans:
        if span.raw_start >= page_raw_end:
            break
        if cursor < span.raw_start:
            benign_end = min(span.raw_start, page_raw_end)
            segments.append((cursor, benign_end, raw_text[cursor:benign_end], False))
        if span.raw_end > page_raw_end:
            cursor = page_raw_end
            break
        segments.append((span.raw_start, span.raw_end, span.replacement, True))
        cursor = span.raw_end
    if cursor < page_raw_end:
        segments.append((cursor, page_raw_end, raw_text[cursor:page_raw_end], False))

    output = ""
    raw_cursor = local_offset
    redacted = False

    def bounded_candidate(value: str) -> tuple[str, bool]:
        candidate = value
        truncated = False
        if max_chars is not None and len(candidate) > max(0, int(max_chars)):
            candidate = candidate[:max(0, int(max_chars))]
            truncated = True
        if max_tokens is not None:
            candidate, token_truncated = _truncate_text_to_token_budget(
                candidate, max(0, int(max_tokens))
            )
            truncated = truncated or token_truncated
        return candidate, truncated

    for raw_start, raw_end, rendered, sensitive in segments:
        if raw_end <= raw_cursor:
            continue
        if raw_cursor > raw_start:
            if sensitive:
                # An arbitrary caller offset inside a credential is fail-closed.
                raw_cursor = raw_end
                redacted = True
                continue
            rendered = rendered[raw_cursor - raw_start:]
            raw_start = raw_cursor

        candidate, was_truncated = bounded_candidate(output + rendered)
        if not was_truncated:
            output = candidate
            raw_cursor = raw_end
            redacted = redacted or sensitive
            continue

        added = candidate[len(output):] if candidate.startswith(output) else ""
        if sensitive:
            redacted = True
            if not output:
                # A placeholder is atomic in raw coordinates. It may be visibly
                # shortened for a tiny caller budget, but consumes the entire
                # credential so the next raw cursor is safe and strictly advances.
                output = candidate or "[LCM redacted]"
                raw_cursor = raw_end
            elif allow_partial_redaction_without_progress and added:
                output = candidate
                raw_cursor = raw_start
            else:
                raw_cursor = raw_start
            break

        output = candidate
        raw_cursor = raw_start + len(added)
        if raw_cursor == raw_start and raw_start < raw_end:
            # Preserve deterministic progress even when one benign character is
            # larger than a caller's token budget.
            output += rendered[:1]
            raw_cursor += 1
        break

    if unsafe_tail_start is not None and raw_cursor >= page_raw_end:
        raw_cursor = unsafe_tail_start
    raw_cursor = min(raw_cursor, total_source_chars)
    has_more = raw_cursor < total_source_chars
    return {
        "content": output,
        "content_chars": total_source_chars,
        "content_offset": requested_offset,
        "content_returned_chars": len(output),
        "content_truncated": has_more,
        "content_source_truncated": source_truncated,
        "content_output_truncated": has_more or len(output) < len(raw_text),
        "content_redacted": redacted or bool(spans) or unsafe_tail_start is not None,
        "next_content_offset": raw_cursor if has_more else 0,
        "has_more": has_more,
    }


def _query_terms_for_match_window(query: str | None) -> list[str]:
    if not query:
        return []
    terms: list[str] = []
    normalized_query = " ".join(re.findall(r"\w+", query))
    if normalized_query:
        terms.append(normalized_query)

    def add_term(term: str) -> None:
        term = term.strip()
        if not term:
            return
        terms.append(term)
        parts = [part for part in re.split(r"[^\w]+", term) if part]
        if len(parts) > 1:
            terms.append(" ".join(parts))
        terms.extend(part for part in parts if len(part) >= 2)

    for quoted in re.findall(r'"([^"]+)"', query):
        add_term(quoted)
    for token in re.findall(r"[\w][\w:-]*\*?", query):
        token = token.rstrip("*").strip()
        if not token or token.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        if ":" in token:
            token = token.rsplit(":", 1)[-1]
        if len(token) >= 2:
            add_term(token)
    seen: set[str] = set()
    unique: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def _content_offset_for_query_match(content: str, query: str | None) -> int:
    folded = content.casefold()
    for term in _query_terms_for_match_window(query):
        index = folded.find(term.casefold())
        if index >= 0:
            return index
    return 0


def _full_content_slice(content: str, content_offset: int = 0) -> dict[str, Any]:
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    sliced = content[content_offset:]
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": False,
        "next_content_offset": 0,
        "has_more": False,
    }


def _restore_ingest_placeholder_for_lookup(
    content: str,
    ref: str | None,
    payload: dict[str, Any] | None,
    *,
    config,
    hermes_home: str,
    session_id: str,
) -> str | None:
    if not content or not ref or not payload or payload.get("kind") != "ingest_payload":
        return None
    restored = restore_ingest_payload_placeholders(
        content,
        config=config,
        hermes_home=hermes_home,
        session_id=session_id,
    )
    return restored if restored != content else None


def _is_compact_externalized_marker(content: str, ref: str | None) -> bool:
    if not ref or not content:
        return False
    if len(content) > 512:
        return False
    return (
        content.startswith("[Externalized tool output:")
        or content.startswith("[GC'd externalized tool output:")
        or content.startswith("[Externalized payload:")
        or content.startswith("[GC'd externalized payload:")
        or "[Externalized LCM ingest payload:" in content
    )


def _pagination_payload(
    *,
    total_sources: int,
    source_offset: int,
    content_offset: int,
    source_limit: int,
    returned_sources: int,
    next_source_offset: int | None,
    next_content_offset: int,
    has_more: bool,
) -> dict[str, Any]:
    if not has_more:
        next_source_offset = None
        next_content_offset = 0
    remaining_sources = 0
    if has_more and next_source_offset is not None:
        remaining_sources = max(0, total_sources - next_source_offset)
    return {
        "source_offset": source_offset,
        "content_offset": content_offset,
        "source_limit": source_limit,
        "returned_sources": returned_sources,
        "total_sources": total_sources,
        "next_source_offset": next_source_offset,
        "next_content_offset": next_content_offset,
        "has_more": has_more,
        "remaining_sources": remaining_sources,
    }


def _expand_message_sources(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
    content_offset: int = 0,
    hydrate_externalized_content: bool = False,
    allowed_session_id: str | None = None,
    frozen_evidence: dict[str, dict[int, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    content_offset = max(0, content_offset)
    source_ids = node.source_ids[source_offset:source_offset + source_limit]
    stored_by_id = (
        frozen_evidence.get("messages", {})
        if frozen_evidence is not None
        else engine._store.get_batch(source_ids)
    )

    messages: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = source_offset
    next_content_offset = content_offset
    has_more = source_offset < total_sources

    for relative_index, store_id in enumerate(source_ids):
        source_index = source_offset + relative_index
        remaining_tokens = max_tokens - budget_used
        if remaining_tokens <= 0:
            next_source_offset = source_index
            next_content_offset = 0
            has_more = True
            break
        stored = stored_by_id.get(store_id)
        if not stored:
            next_source_offset = source_index + 1
            next_content_offset = 0
            has_more = next_source_offset < total_sources
            continue
        if allowed_session_id is not None and stored.get("session_id", "") != allowed_session_id:
            next_source_offset = source_index + 1
            next_content_offset = 0
            has_more = next_source_offset < total_sources
            continue
        transcript_content = stored.get("content", "")
        content = transcript_content
        content_source = "message"
        externalized = None
        ref_payload = None
        ingest_refs = extract_ingest_externalized_refs(transcript_content)
        ref = ingest_refs[0] if ingest_refs else extract_externalized_ref(transcript_content)
        if ref and frozen_evidence is None:
            payload_allowed_sessions = (
                {allowed_session_id}
                if allowed_session_id is not None
                else {engine.current_session_id, stored.get("session_id", "")}
            )
            ref_payload = _get_externalized_payload(
                engine,
                ref,
                allowed_session_ids=payload_allowed_sessions,
            )
            if ref_payload is not None and ref_payload.get("kind") != "ingest_payload":
                externalized = ref_payload
        if hydrate_externalized_content and externalized is not None:
            content = externalized.get("content", "")
            content_source = "externalized_payload"
        effective_content_offset = content_offset if source_index == source_offset else 0
        if not hydrate_externalized_content and _is_compact_externalized_marker(content, ref):
            sliced = _full_content_slice(content, effective_content_offset)
        elif frozen_evidence is not None:
            sliced = _slice_redacted_raw_page(
                content,
                content_offset=effective_content_offset,
                max_tokens=remaining_tokens,
                max_chars=_CURRENT_SESSION_EXPAND_MAX_CHARS,
                source_content_chars=int(stored.get("content_chars") or len(content)),
                allow_incomplete_tail=True,
            )
        else:
            sliced = _slice_content_for_response(
                content,
                remaining_tokens,
                effective_content_offset,
                config=engine._config,
            )
        expanded = {
            "store_id": stored["store_id"],
            "source_index": source_index,
            "session_id": stored.get("session_id", ""),
            "source": stored.get("source") or "",
            "from_current_session": stored.get("session_id", "") == engine.current_session_id,
            "role": stored["role"],
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "content_source": content_source,
        }
        if content_source == "externalized_payload":
            expanded["transcript_content"] = transcript_content
        if stored.get("role") == "tool" and frozen_evidence is None:
            if externalized is not None:
                externalized_summary = dict(externalized)
                externalized_summary.pop("content", None)
                expanded["externalized"] = externalized_summary
            if "externalized" not in expanded:
                lookup_candidates = [transcript_content]
                restored_ingest_content = _restore_ingest_placeholder_for_lookup(
                    transcript_content,
                    ref,
                    ref_payload,
                    config=engine._config,
                    hermes_home=engine._hermes_home,
                    session_id=stored.get("session_id", ""),
                )
                if restored_ingest_content is not None:
                    lookup_candidates.insert(0, restored_ingest_content)
                    sanitized_restored = sanitize_pre_compaction_content(restored_ingest_content)
                    if sanitized_restored != restored_ingest_content:
                        lookup_candidates.insert(0, sanitized_restored)
                sanitized_content = sanitize_pre_compaction_content(transcript_content)
                if sanitized_content != transcript_content:
                    lookup_candidates.insert(0, sanitized_content)
                for candidate in lookup_candidates:
                    externalized = find_externalized_payload_for_message(
                        candidate,
                        tool_call_id=stored.get("tool_call_id", ""),
                        session_id=stored.get("session_id", ""),
                        config=engine._config,
                        hermes_home=engine._hermes_home,
                    )
                    if externalized is not None:
                        externalized_summary = dict(externalized)
                        externalized_summary.pop("content", None)
                        expanded["externalized"] = externalized_summary
                        break
        messages.append(expanded)
        budget_used += count_tokens(sliced["content"])
        if sliced["has_more"]:
            next_source_offset = source_index
            next_content_offset = sliced["next_content_offset"]
            has_more = True
            break
        next_source_offset = source_index + 1
        next_content_offset = 0
        has_more = next_source_offset < total_sources
    else:
        has_more = (source_offset + source_limit) < total_sources
        next_source_offset = source_offset + source_limit if has_more else None
        next_content_offset = 0

    pagination = _pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=content_offset,
        source_limit=source_limit,
        returned_sources=len(messages),
        next_source_offset=next_source_offset,
        next_content_offset=next_content_offset,
        has_more=has_more,
    )
    return messages, pagination


def _expand_child_nodes(
    engine: "LCMEngine",
    node,
    max_tokens: int | None = None,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
    allowed_session_id: str | None = None,
    frozen_evidence: dict[str, dict[int, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    selected_source_ids = node.source_ids[source_offset:source_offset + source_limit]
    children: list[tuple[int, Any]] = []
    for relative_index, child_id in enumerate(selected_source_ids):
        child = (
            frozen_evidence.get("nodes", {}).get(child_id)
            if frozen_evidence is not None
            else engine._dag.get_node(child_id)
        )
        effective_session_id = allowed_session_id or engine.current_session_id
        if child is None or child.session_id != effective_session_id:
            continue
        children.append((source_offset + relative_index, child))

    expanded: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = None
    has_more = (source_offset + source_limit) < total_sources
    for source_index, child in children:
        remaining_tokens = (
            max_tokens - budget_used
            if max_tokens is not None
            else _CROSS_SESSION_METADATA_MAX_TOKENS
        )
        if remaining_tokens <= 0:
            next_source_offset = source_index
            has_more = True
            break
        summary, summary_truncated = _bounded_cross_session_text(
            child.summary,
            engine._config,
            max_tokens=remaining_tokens,
            max_chars=None if max_tokens is not None else 1000,
        )
        safe_hint, hint_truncated = _bounded_cross_session_text(
            child.expand_hint,
            engine._config,
            max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
            max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
        )
        expanded.append(
            {
                "node_id": child.node_id,
                "source_index": source_index,
                "depth": child.depth,
                "summary": summary,
                "summary_truncated": summary_truncated,
                "token_count": child.token_count,
                "source_token_count": child.source_token_count,
                "expand_hint": safe_hint,
                "expand_hint_truncated": hint_truncated,
            }
        )
        budget_used += count_tokens(summary)
        if summary_truncated:
            next_source_offset = source_index + 1
            has_more = next_source_offset < total_sources
            break
        next_source_offset = source_index + 1

    if has_more and next_source_offset is None:
        next_source_offset = source_offset + source_limit

    return expanded, _pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=0,
        source_limit=source_limit,
        returned_sources=len(expanded),
        next_source_offset=next_source_offset,
        next_content_offset=0,
        has_more=has_more,
    )


def _bounded_source_path_payload(source_path: list[dict[str, int]]) -> dict[str, Any]:
    path_tail = source_path[-8:]
    payload: dict[str, Any] = {
        "source_path": path_tail,
        "source_path_depth": len(source_path),
    }
    if len(path_tail) < len(source_path):
        payload["source_path_truncated"] = True
    return payload


def _collect_descendant_evidence_blocks(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
    visited_node_ids: set[int] | None = None,
    source_path: list[dict[str, int]] | None = None,
    remaining_node_visits: list[int] | None = None,
    allowed_session_id: str | None = None,
    frozen_evidence: dict[str, dict[int, Any]] | None = None,
) -> list[dict[str, Any]]:
    if max_tokens <= 0 or node.source_type != "nodes":
        return []
    if visited_node_ids is None:
        visited_node_ids = set()
    if source_path is None:
        source_path = []
    if remaining_node_visits is None:
        # Budget and cycle detection are the primary limits. Keep a high,
        # budget-scaled guard so corrupt zero-token DAGs cannot make expansion
        # walk an unbounded number of nodes while normal deep summaries still
        # reach their leaf evidence.
        remaining_node_visits = [max(64, int(max_tokens) * 4)]
    if remaining_node_visits[0] <= 0:
        return []

    blocks: list[dict[str, Any]] = []
    budget_used = 0
    root_node_id = int(node.node_id)
    stack: list[tuple[Any, list[dict[str, int]], set[int], int]] = [
        (node, source_path, {*visited_node_ids, root_node_id}, 0)
    ]

    while stack and budget_used < max_tokens and remaining_node_visits[0] > 0:
        current, current_path, current_visited, source_index = stack.pop()
        if source_index >= len(current.source_ids):
            continue

        stack.append((current, current_path, current_visited, source_index + 1))
        child_id = current.source_ids[source_index]
        child = (
            frozen_evidence.get("nodes", {}).get(child_id)
            if frozen_evidence is not None
            else engine._dag.get_node(child_id)
        )
        effective_session_id = allowed_session_id or engine.current_session_id
        if child is None or child.session_id != effective_session_id:
            continue
        child_node_id = int(child.node_id)
        if child_node_id in current_visited:
            continue

        remaining_node_visits[0] -= 1
        child_path = [*current_path, {"node_id": int(current.node_id), "source_index": source_index}]
        remaining_tokens = max(0, max_tokens - budget_used)
        if child.source_type == "messages":
            messages, pagination = _expand_message_sources(
                engine,
                child,
                max_tokens=remaining_tokens,
                hydrate_externalized_content=hydrate_externalized_content,
                allowed_session_id=effective_session_id,
                frozen_evidence=frozen_evidence,
            )
            if messages or pagination.get("has_more"):
                block = {
                    "type": "child_messages",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **_bounded_source_path_payload(child_path),
                    "messages": messages,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += _context_content_token_count([block])
            continue

        if child.source_type == "nodes":
            children, pagination = _expand_child_nodes(
                engine,
                child,
                max_tokens=remaining_tokens,
                allowed_session_id=effective_session_id,
                frozen_evidence=frozen_evidence,
            )
            if children or pagination.get("has_more"):
                block = {
                    "type": "descendant_child_nodes",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **_bounded_source_path_payload(child_path),
                    "children": children,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += _context_content_token_count([block])
            if budget_used < max_tokens and remaining_node_visits[0] > 0:
                stack.append((child, child_path, {*current_visited, child_node_id}, 0))
    return blocks


def _collect_context_blocks_for_node(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
    allowed_session_id: str | None = None,
    frozen_evidence: dict[str, dict[int, Any]] | None = None,
) -> list[dict[str, Any]]:
    from .tokens import count_tokens

    summary, summary_truncated = _bounded_cross_session_text(
        node.summary,
        engine._config,
        max_tokens=max_tokens,
        max_chars=None,
    )
    safe_hint, hint_truncated = _bounded_cross_session_text(
        node.expand_hint,
        engine._config,
        max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
        max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
    )
    blocks: list[dict[str, Any]] = [
        {
            "type": "summary",
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": summary,
            "summary_truncated": summary_truncated,
            "expand_hint": safe_hint,
            "expand_hint_truncated": hint_truncated,
            "token_count": node.token_count,
        }
    ]
    remaining_tokens = max(0, max_tokens - count_tokens(summary))

    if node.source_type == "messages":
        messages, pagination = _expand_message_sources(
            engine,
            node,
            max_tokens=remaining_tokens,
            hydrate_externalized_content=hydrate_externalized_content,
            allowed_session_id=allowed_session_id,
            frozen_evidence=frozen_evidence,
        )
        if messages or pagination.get("has_more"):
            block = {
                "type": "messages",
                "node_id": node.node_id,
                "messages": messages,
                "pagination": pagination,
            }
            blocks.append(block)
    elif node.source_type == "nodes":
        children, pagination = _expand_child_nodes(
            engine,
            node,
            max_tokens=remaining_tokens,
            allowed_session_id=allowed_session_id,
            frozen_evidence=frozen_evidence,
        )
        if children or pagination.get("has_more"):
            blocks.append(
                {
                    "type": "child_nodes",
                    "node_id": node.node_id,
                    "children": children,
                    "pagination": pagination,
                }
            )
        used_tokens = _context_content_token_count(blocks)
        descendant_tokens = max(0, max_tokens - used_tokens)
        if descendant_tokens > 0:
            blocks.extend(
                _collect_descendant_evidence_blocks(
                    engine,
                    node,
                    max_tokens=descendant_tokens,
                    hydrate_externalized_content=hydrate_externalized_content,
                    allowed_session_id=allowed_session_id,
                    frozen_evidence=frozen_evidence,
                )
            )

    return blocks


def _collect_raw_match_context_block(
    engine: "LCMEngine",
    rows: list[dict[str, Any]],
    max_tokens: int,
    *,
    query: str | None = None,
    exclude_store_ids: set[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    from .tokens import count_tokens

    exclude_store_ids = exclude_store_ids or set()
    messages: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    budget_used = 0
    has_more = False
    next_store_id: int | None = None
    for row in rows:
        store_id = row.get("store_id")
        if store_id in exclude_store_ids:
            continue
        remaining_tokens = max(0, max_tokens - budget_used)
        if remaining_tokens <= 0:
            has_more = True
            next_store_id = store_id if isinstance(store_id, int) else None
            break
        content = str(row.get("content") or "")
        match_offset = _content_offset_for_query_match(content, query)
        content_slice = _slice_content_for_response(
            content,
            remaining_tokens,
            content_offset=match_offset,
            config=engine._config,
        )
        content = content_slice["content"]
        item = {
            "store_id": store_id,
            "session_id": row.get("session_id") or "",
            "source": row.get("source") or "",
            "role": row.get("role"),
            "timestamp": row.get("timestamp", 0),
            **content_slice,
            "content_source": "raw_search_hit",
            "search_rank": row.get("search_rank"),
        }
        if row.get("tool_call_id"):
            item["tool_call_id"] = row.get("tool_call_id")
        if match_offset:
            item["match_window_offset"] = match_offset
        if row.get("tool_calls"):
            item["tool_calls_omitted"] = True
        if row.get("tool_name"):
            item["tool_name"] = row.get("tool_name")
        messages.append(item)
        matches.append(
            {
                "store_id": store_id,
                "role": row.get("role"),
                "snippet": row.get("snippet") or content[:300],
                "search_rank": row.get("search_rank"),
            }
        )
        budget_used += count_tokens(content)
        if content_slice["has_more"]:
            has_more = True
            break

    if not messages and not has_more:
        return None, matches
    block = {
        "type": "raw_messages",
        "messages": messages,
        "pagination": {
            "has_more": has_more,
            "returned_sources": len(messages),
            "total_sources": len(rows),
            "next_store_id": next_store_id,
        },
    }
    return block, matches


def _collect_store_ids_from_context_blocks(blocks: list[dict[str, Any]]) -> set[int]:
    store_ids: set[int] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for message in block.get("messages", []) or []:
            store_id = message.get("store_id")
            if isinstance(store_id, int):
                store_ids.add(store_id)
    return store_ids


def _context_content_token_count(blocks: list[dict[str, Any]]) -> int:
    from .tokens import count_tokens

    def count_strings(value: Any) -> int:
        if isinstance(value, str):
            return count_tokens(value)
        if isinstance(value, dict):
            return sum(count_strings(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count_strings(item) for item in value)
        return 0

    total = count_strings(blocks)
    for block in blocks:
        if not isinstance(block, dict) or "source_path" not in block:
            continue
        # Path entries are numeric, but their serialized structure grows with
        # every descendant block and was already part of the context budget.
        # Keep charging it in addition to all free-form string metadata.
        total += count_tokens(
            json.dumps(
                {
                    "source_path": block.get("source_path") or [],
                    "source_path_depth": block.get("source_path_depth"),
                    "source_path_truncated": block.get("source_path_truncated", False),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return total


def _bounded_cross_session_text(
    value: Any,
    config,
    *,
    max_tokens: int,
    max_chars: int | None,
) -> tuple[str, bool]:
    """Redact and bound one archive string before it reaches synthesis/output."""
    raw_text = str(value or "")
    truncated = False
    # Scan a bounded output window plus lookahead before any safe truncation.
    # The lookahead closes credentials that straddle the visible boundary; the
    # explicit tail guards fail closed for a PEM/token longer than the window.
    visible_char_budget = (
        max(0, int(max_chars))
        if max_chars is not None
        else max(1, int(max_tokens)) * _MANDATORY_REDACTION_CHARS_PER_TOKEN
    )
    scan_limit = min(
        len(raw_text),
        visible_char_budget + _MANDATORY_REDACTION_LOOKAHEAD_CHARS,
    )
    text = raw_text[:scan_limit]
    truncated = scan_limit < len(raw_text)
    text = redact_sensitive_output_text(text)
    pem_begin = _BOUNDARY_PRIVATE_KEY_BEGIN_RE.search(text)
    if pem_begin is not None:
        text = (
            text[:pem_begin.start()]
            + "[LCM sensitive redaction: name=private_key; boundary-truncated]"
        )
        truncated = True
    text, boundary_credential_count = _BOUNDARY_STANDALONE_CREDENTIAL_RE.subn(
        "[LCM sensitive redaction: name=standalone_credential; boundary-truncated]",
        text,
    )
    truncated = truncated or bool(boundary_credential_count)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    text, token_truncated = _truncate_text_to_token_budget(text, max_tokens)
    return text, truncated or token_truncated


def _grep_safe_text(value: Any, *, max_chars: int) -> str:
    """Mandatory-redact one grep response field before its output bound."""
    return _bounded_cross_session_text(
        value,
        None,
        max_tokens=max(1, max_chars * 2),
        max_chars=max_chars,
    )[0]


def _grep_safe_snippet(
    value: Any,
    query: str,
    *,
    max_chars: int = 300,
    highlight_matches: bool = True,
    left_boundary_truncated: bool = False,
) -> str:
    """Redact the complete bounded match window before choosing its snippet."""
    raw_window = str(value or "")
    if not raw_window:
        return ""
    if left_boundary_truncated:
        boundary_name = "left_boundary_credential"
        # The outer state is unknown. Assignment-like text in the window may
        # itself be secret body data, so it cannot replace that state or prove
        # a terminator (including same-key decoys). The bounded record/window
        # boundary is the only proof available here; redact through it and add
        # an independently protected query marker for result discoverability.
        boundary_end = len(raw_window)
        entire_window_uncertain = boundary_end >= len(raw_window)
        raw_window = (
            f"[LCM sensitive redaction: name={boundary_name}; boundary-truncated]"
            + raw_window[boundary_end:]
        )
        if entire_window_uncertain:
            # Preserve match discoverability without copying any uncertain body
            # bytes. The query is independently capped and mandatory-redacted.
            safe_query = _grep_safe_text(query, max_chars=min(max_chars // 2, 128))
            if safe_query:
                raw_window += f" >>>{safe_query}<<<"
    protected, _truncated = _bounded_cross_session_text(
        raw_window,
        None,
        max_tokens=max(1, len(raw_window) * 2),
        max_chars=len(raw_window),
    )
    folded = protected.casefold()
    match_start = -1
    match_length = 0
    for term in _query_terms_for_match_window(query):
        index = folded.find(term.casefold())
        if index >= 0 and (match_start < 0 or index < match_start):
            match_start = index
            match_length = len(term)
    if match_start < 0:
        match_start = 0
    left = max(0, match_start - (max_chars // 3))
    right = min(
        len(protected),
        max(left + max_chars, match_start + match_length + (max_chars // 3)),
    )
    if right - left > max_chars:
        right = left + max_chars
    snippet = protected[left:right]
    local_match = match_start - left
    if (
        highlight_matches
        and match_length > 0
        and local_match >= 0
        and local_match + match_length <= len(snippet)
    ):
        snippet = (
            snippet[:local_match]
            + ">>>"
            + snippet[local_match:local_match + match_length]
            + "<<<"
            + snippet[local_match + match_length:]
        )
    if left > 0:
        snippet = "..." + snippet
    if right < len(protected):
        snippet += "..."
    return snippet[:max_chars]


def _mandatory_redact_grep_response(value: Any) -> Any:
    """Fail-safe final boundary: no credential may survive in any grep field."""
    if isinstance(value, str):
        return redact_sensitive_output_text(value)
    if isinstance(value, list):
        return [_mandatory_redact_grep_response(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _mandatory_redact_grep_response(item)
            for key, item in value.items()
        }
    return value


def _authorize_node_provenance_bounded(
    engine: "LCMEngine",
    candidates: list[Any],
    allowed_session_ids: frozenset[str],
    *,
    deadline: float,
) -> tuple[list[Any], dict[str, Any], dict[str, dict[int, Any]]]:
    """Authorize and freeze one bounded closure from one SQLite read snapshot."""
    diagnostics: dict[str, Any] = {
        "authorization_truncated": len(candidates) > _CROSS_SESSION_AUTH_MAX_CANDIDATES,
        "authorization_timed_out": False,
        "authorization_candidates_seen": len(candidates),
        "authorization_candidates_checked": 0,
        "authorization_nodes_checked": 0,
        "authorization_messages_checked": 0,
        "authorization_edges_checked": 0,
        "authorization_limits": {
            "candidates": _CROSS_SESSION_AUTH_MAX_CANDIDATES,
            "nodes": _CROSS_SESSION_AUTH_MAX_NODES,
            "messages": _CROSS_SESSION_AUTH_MAX_MESSAGES,
            "edges": _CROSS_SESSION_AUTH_MAX_EDGES,
            "depth": _CROSS_SESSION_AUTH_MAX_DEPTH,
        },
    }
    selected_ids: list[int] = []
    candidate_rank: dict[int, tuple[Any, Any]] = {}
    for candidate in candidates[:_CROSS_SESSION_AUTH_MAX_CANDIDATES]:
        node_id = int(candidate.node_id)
        if node_id in candidate_rank:
            continue
        selected_ids.append(node_id)
        candidate_rank[node_id] = (
            getattr(candidate, "search_rank", None),
            getattr(candidate, "search_directness", 0.0),
        )

    frozen_nodes: dict[int, Any] = {}
    frozen_messages: dict[int, dict[str, Any]] = {}
    incomplete: set[int] = set()
    message_sessions: dict[int, str] = {}
    materialized_bytes = 0
    savepoint = "lcm_cross_session_evidence_snapshot"
    conn = engine._store._conn

    def charge_row(row: tuple[Any, ...]) -> bool:
        nonlocal materialized_bytes
        row_bytes = sum(
            len(value.encode("utf-8", errors="surrogatepass"))
            for value in row
            if isinstance(value, str)
        )
        if materialized_bytes + row_bytes > _CROSS_SESSION_AUTH_MAX_MATERIALIZED_BYTES:
            diagnostics["authorization_truncated"] = True
            return False
        materialized_bytes += row_bytes
        return True

    with engine._store._write_lock:
        snapshot_data_version = int(conn.execute("PRAGMA data_version").fetchone()[0])
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            queue: list[tuple[int, int]] = [(node_id, 0) for node_id in selected_ids]
            queued = set(selected_ids)
            expanded: set[int] = set()
            message_ids: set[int] = set()
            projection = engine._dag._cross_session_candidate_projection()
            guard = engine._dag._cross_session_candidate_guard()

            while queue and len(frozen_nodes) < _CROSS_SESSION_AUTH_MAX_NODES:
                if time.monotonic() >= deadline:
                    diagnostics["authorization_timed_out"] = True
                    break
                batch_entries = queue[:_CROSS_SESSION_AUTH_QUERY_BATCH]
                del queue[:len(batch_entries)]
                batch_ids = [
                    node_id for node_id, _depth in batch_entries
                    if node_id not in frozen_nodes
                ]
                if batch_ids:
                    remaining_nodes = _CROSS_SESSION_AUTH_MAX_NODES - len(frozen_nodes)
                    batch_ids = batch_ids[:remaining_nodes]
                    placeholders = ",".join("?" for _ in batch_ids)
                    rows = conn.execute(
                        f"SELECT {projection} FROM summary_nodes "
                        f"WHERE node_id IN ({placeholders}) AND {guard} LIMIT ?",
                        [*batch_ids, len(batch_ids)],
                    )
                    found: set[int] = set()
                    while time.monotonic() < deadline:
                        raw_row = rows.fetchone()
                        if raw_row is None:
                            break
                        row = tuple(raw_row)
                        if not charge_row(row):
                            break
                        node = engine._dag._cross_session_row_to_node(row)
                        if node is None:
                            continue
                        found.add(int(node.node_id))
                        frozen_nodes[int(node.node_id)] = node
                    for missing_id in set(batch_ids) - found:
                        incomplete.add(missing_id)
                        diagnostics["authorization_truncated"] = True

                for node_id, depth in batch_entries:
                    if node_id in expanded:
                        continue
                    expanded.add(node_id)
                    node = frozen_nodes.get(node_id)
                    if node is None or node.session_id not in allowed_session_ids:
                        incomplete.add(node_id)
                        continue
                    if depth > _CROSS_SESSION_AUTH_MAX_DEPTH or not node.source_ids:
                        incomplete.add(node_id)
                        diagnostics["authorization_truncated"] = True
                        continue
                    remaining_edges = _CROSS_SESSION_AUTH_MAX_EDGES - int(
                        diagnostics["authorization_edges_checked"]
                    )
                    if len(node.source_ids) > remaining_edges:
                        incomplete.add(node_id)
                        diagnostics["authorization_truncated"] = True
                        continue
                    diagnostics["authorization_edges_checked"] += len(node.source_ids)
                    if node.source_type == "messages":
                        message_ids.update(int(value) for value in node.source_ids)
                    elif node.source_type == "nodes":
                        if depth >= _CROSS_SESSION_AUTH_MAX_DEPTH:
                            incomplete.add(node_id)
                            diagnostics["authorization_truncated"] = True
                            continue
                        for child_id in node.source_ids:
                            child_id = int(child_id)
                            if child_id not in queued:
                                queue.append((child_id, depth + 1))
                                queued.add(child_id)
                    else:
                        incomplete.add(node_id)

            if queue:
                diagnostics["authorization_truncated"] = True
                incomplete.update(node_id for node_id, _depth in queue)
            diagnostics["authorization_nodes_checked"] = len(expanded)

            bounded_message_ids = list(message_ids)[:_CROSS_SESSION_AUTH_MAX_MESSAGES]
            if len(message_ids) > len(bounded_message_ids):
                diagnostics["authorization_truncated"] = True
            # Phase 1: read only bounded ownership metadata. Payload columns
            # are deliberately absent so a foreign provenance edge cannot
            # cause SQLite to touch content/tool_calls before closure auth.
            for offset in range(0, len(bounded_message_ids), _CROSS_SESSION_AUTH_QUERY_BATCH):
                if time.monotonic() >= deadline:
                    diagnostics["authorization_timed_out"] = True
                    break
                batch = bounded_message_ids[offset:offset + _CROSS_SESSION_AUTH_QUERY_BATCH]
                placeholders = ",".join("?" for _ in batch)
                cursor = conn.execute(
                    f"""SELECT
                               CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                               CASE WHEN typeof(session_id) = 'text'
                                          AND length(CAST(session_id AS BLOB)) <= {_CROSS_SESSION_AUTH_SESSION_ID_CHARS * 4}
                                          AND length(CAST(session_id AS TEXT)) <= {_CROSS_SESSION_AUTH_SESSION_ID_CHARS}
                                    THEN substr(CAST(session_id AS TEXT), 1, {_CROSS_SESSION_AUTH_SESSION_ID_CHARS}) END,
                               typeof(session_id)
                        FROM messages WHERE store_id IN ({placeholders}) LIMIT ?""",
                    [*batch, len(batch)],
                )
                while time.monotonic() < deadline:
                    raw_row = cursor.fetchone()
                    if raw_row is None:
                        break
                    row = tuple(raw_row)
                    if not charge_row(row):
                        break
                    if (
                        row[0] is None
                        or not isinstance(row[1], str)
                        or row[2] != "text"
                    ):
                        diagnostics["authorization_truncated"] = True
                        continue
                    store_id = int(row[0])
                    message_sessions[store_id] = row[1]
            diagnostics["authorization_messages_checked"] = len(message_sessions)

            memo: dict[int, bool] = {}
            visiting: set[int] = set()

            def authorized(node_id: int, depth: int = 0) -> bool:
                if depth > _CROSS_SESSION_AUTH_MAX_DEPTH or node_id in incomplete:
                    return False
                if node_id in memo:
                    return memo[node_id]
                if node_id in visiting:
                    memo[node_id] = False
                    return False
                node = frozen_nodes.get(node_id)
                if (
                    node is None
                    or node.session_id not in allowed_session_ids
                    or not node.source_ids
                ):
                    memo[node_id] = False
                    return False
                visiting.add(node_id)
                if node.source_type == "messages":
                    result = all(
                        message_sessions.get(int(source_id)) in allowed_session_ids
                        for source_id in node.source_ids
                    )
                elif node.source_type == "nodes":
                    result = all(
                        authorized(int(child_id), depth + 1)
                        for child_id in node.source_ids
                    )
                else:
                    result = False
                visiting.discard(node_id)
                memo[node_id] = result
                return result

            authorized_candidates: list[Any] = []
            for node_id in selected_ids:
                if not authorized(node_id):
                    continue
                node = frozen_nodes[node_id]
                node.search_rank, node.search_directness = candidate_rank[node_id]
                authorized_candidates.append(node)
            diagnostics["authorization_candidates_checked"] = len(selected_ids)

            # Phase 2: only after complete closure authorization, derive the
            # exact authorized message ID set and materialize bounded payload
            # for those IDs from the same SQLite read snapshot.
            authorized_message_ids: set[int] = set()
            collected_nodes: set[int] = set()

            def collect_authorized_messages(node_id: int) -> None:
                if node_id in collected_nodes:
                    return
                collected_nodes.add(node_id)
                node = frozen_nodes[node_id]
                if node.source_type == "messages":
                    authorized_message_ids.update(
                        int(source_id) for source_id in node.source_ids
                    )
                else:
                    for child_id in node.source_ids:
                        collect_authorized_messages(int(child_id))

            for node in authorized_candidates:
                collect_authorized_messages(int(node.node_id))

            bounded_payload_ids = sorted(authorized_message_ids)
            for offset in range(0, len(bounded_payload_ids), _CROSS_SESSION_AUTH_QUERY_BATCH):
                if time.monotonic() >= deadline:
                    diagnostics["authorization_timed_out"] = True
                    break
                batch = bounded_payload_ids[
                    offset:offset + _CROSS_SESSION_AUTH_QUERY_BATCH
                ]
                placeholders = ",".join("?" for _ in batch)
                cursor = conn.execute(
                    f"""SELECT
                               CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                               substr(CAST(session_id AS TEXT), 1, {_CROSS_SESSION_AUTH_SESSION_ID_CHARS}),
                               substr(CAST(source AS TEXT), 1, 512),
                               substr(CAST(role AS TEXT), 1, 128),
                               CASE WHEN typeof(content) = 'text'
                                    THEN substr(CAST(content AS TEXT), 1, {_CURRENT_SESSION_EXPAND_MAX_CHARS}) END,
                               substr(CAST(tool_call_id AS TEXT), 1, 512),
                               substr(CAST(tool_name AS TEXT), 1, 512),
                               CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                               CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                               CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                               substr(CAST(conversation_id AS TEXT), 1, 512),
                               COALESCE(length(CAST(content AS TEXT)), 0),
                               typeof(content)
                        FROM messages WHERE store_id IN ({placeholders}) LIMIT ?""",
                    [*batch, len(batch)],
                )
                while time.monotonic() < deadline:
                    raw_row = cursor.fetchone()
                    if raw_row is None:
                        break
                    row = tuple(raw_row)
                    if not charge_row(row):
                        break
                    store_id = int(row[0]) if row[0] is not None else -1
                    if (
                        store_id not in authorized_message_ids
                        or row[1] != message_sessions.get(store_id)
                        or row[12] not in {"text", "null"}
                    ):
                        diagnostics["authorization_truncated"] = True
                        continue
                    frozen_messages[store_id] = {
                        "store_id": store_id,
                        "session_id": row[1],
                        "source": row[2] or "",
                        "role": row[3] or "unknown",
                        "content": row[4] or "",
                        "tool_call_id": row[5] or "",
                        "tool_calls": None,
                        "tool_name": row[6] or "",
                        "timestamp": row[7] or 0,
                        "token_estimate": row[8] or 0,
                        "pinned": row[9] or 0,
                        "conversation_id": row[10] or "",
                        "content_chars": int(row[11] or 0),
                    }

            if set(frozen_messages) != authorized_message_ids:
                diagnostics["authorization_truncated"] = True
                authorized_candidates = []
                frozen_messages = {}
            if diagnostics["authorization_timed_out"]:
                authorized_candidates = []
                frozen_nodes = {}
                frozen_messages = {}
            conn.execute(f"RELEASE {savepoint}")
            if int(conn.execute("PRAGMA data_version").fetchone()[0]) != snapshot_data_version:
                # A writer committed while this connection held its read view.
                # The frozen view is internally consistent, but deny this turn
                # so authorization cannot race a concurrent ownership change.
                diagnostics["authorization_truncated"] = True
                diagnostics["authorization_concurrent_mutation"] = True
                authorized_candidates = []
                frozen_nodes = {}
                frozen_messages = {}
            return authorized_candidates, diagnostics, {
                "nodes": frozen_nodes,
                "messages": frozen_messages,
            }
        except Exception:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
            raise


def _bound_cross_session_context(
    value: Any,
    *,
    max_tokens: int,
    config,
) -> tuple[Any, bool]:
    """Recursively fit all archive context strings into one shared token budget."""
    from .tokens import count_tokens

    state = {"remaining": max(0, int(max_tokens)), "truncated": False}

    def bound(item: Any, *, field_name: str = "", depth: int = 0) -> Any:
        if depth > _CROSS_SESSION_CONTEXT_MAX_DEPTH:
            state["truncated"] = True
            return None
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            entries = list(item.items())[:_CROSS_SESSION_CONTEXT_MAX_ITEMS]
            if len(item) > len(entries):
                state["truncated"] = True
            for raw_key, child in entries:
                safe_key, key_truncated = _bounded_cross_session_text(
                    raw_key,
                    config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=256,
                )
                state["truncated"] = state["truncated"] or key_truncated
                result[safe_key] = bound(
                    child,
                    field_name=safe_key,
                    depth=depth + 1,
                )
            return result
        if isinstance(item, (list, tuple)):
            entries = list(item)[:_CROSS_SESSION_CONTEXT_MAX_ITEMS]
            if len(item) > len(entries):
                state["truncated"] = True
            return [bound(child, field_name=field_name, depth=depth + 1) for child in entries]
        if isinstance(item, str):
            is_content = field_name in _CROSS_SESSION_CONTENT_FIELDS
            field_tokens = state["remaining"]
            if not is_content:
                field_tokens = min(field_tokens, _CROSS_SESSION_METADATA_MAX_TOKENS)
            protected, truncated = _bounded_cross_session_text(
                item,
                config,
                max_tokens=field_tokens,
                max_chars=None if is_content else _CROSS_SESSION_METADATA_MAX_CHARS,
            )
            used = count_tokens(protected)
            state["remaining"] = max(0, state["remaining"] - used)
            state["truncated"] = state["truncated"] or truncated
            return protected
        return item

    bounded = bound(value)
    if isinstance(bounded, list):
        # Numeric descendant paths have a serialized structural cost in
        # _context_content_token_count. The recursive string allocator cannot
        # reserve that cost before it sees the complete block, so discard
        # trailing evidence blocks until the fully-accounted payload fits.
        while bounded and _context_content_token_count(bounded) > max_tokens:
            bounded.pop()
            state["truncated"] = True
    return bounded, bool(state["truncated"])


def _bound_current_session_value(
    value: Any,
    *,
    config,
    content_max_chars: int | None = None,
) -> tuple[Any, bool]:
    """Redact/bound current-session fields without changing pagination shape.

    Current-session collectors already allocate the caller's evidence budget and
    deliberately preserve zero/one-token pagination sentinels. This pass bounds
    hostile metadata and every emitted string, but does not spend structural
    labels ahead of evidence or discard blocks as the cross-session shared-budget
    allocator must.
    """
    state = {"truncated": False}

    def bound(item: Any, *, field_name: str = "", depth: int = 0) -> Any:
        if depth > _CROSS_SESSION_CONTEXT_MAX_DEPTH:
            state["truncated"] = True
            return None
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            entries = list(item.items())[:_CROSS_SESSION_CONTEXT_MAX_ITEMS]
            state["truncated"] = state["truncated"] or len(entries) < len(item)
            for raw_key, child in entries:
                safe_key, key_truncated = _bounded_cross_session_text(
                    raw_key,
                    config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=256,
                )
                state["truncated"] = state["truncated"] or key_truncated
                result[safe_key] = bound(
                    child, field_name=safe_key, depth=depth + 1
                )
            return result
        if isinstance(item, (list, tuple)):
            entries = list(item)[:_CROSS_SESSION_CONTEXT_MAX_ITEMS]
            state["truncated"] = state["truncated"] or len(entries) < len(item)
            return [
                bound(child, field_name=field_name, depth=depth + 1)
                for child in entries
            ]
        if isinstance(item, str):
            is_content = field_name in _CROSS_SESSION_CONTENT_FIELDS
            if is_content:
                # The current-session collectors have already applied their
                # shared token budget and chosen cursor-safe one-character
                # sentinels where necessary. Redact the complete collected
                # value, then apply only the optional hard character ceiling;
                # a second token pass here can erase those sentinels and break
                # deterministic pagination under custom tokenizers.
                protected = redact_sensitive_output_text(item)
                truncated = False
                if content_max_chars is not None and len(protected) > content_max_chars:
                    protected = protected[:content_max_chars]
                    truncated = True
            else:
                protected, truncated = _bounded_cross_session_text(
                    item,
                    config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
                )
            state["truncated"] = state["truncated"] or truncated
            return protected
        return item

    return bound(value), bool(state["truncated"])


def _synthesize_expansion_answer(
    *,
    prompt: str,
    context_blocks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    timeout: float,
) -> str:
    from agent.auxiliary_client import call_llm

    system_prompt = (
        "You answer questions using expanded LCM retrieval context. "
        "Be concise, factual, and grounded in the provided context. "
        "If the context is insufficient, say so plainly."
    )
    user_prompt = (
        f"QUESTION:\n{prompt}\n\n"
        "EXPANDED CONTEXT:\n"
        f"{json.dumps(context_blocks, ensure_ascii=False, indent=2)}"
    )
    call_kwargs = {
        "task": "compression",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    apply_lcm_model_route(call_kwargs, model)
    response = call_llm(**call_kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        content = str(content) if content else ""
    from .escalation import _strip_reasoning_blocks
    return _strip_reasoning_blocks(content).strip()


def _cross_session_expand_query(
    engine: "LCMEngine",
    args: Dict[str, Any],
    *,
    prompt: str,
    query: str,
    raw_node_ids: list[Any],
    max_tokens: int,
    context_max_tokens: int,
    max_results: int,
    allowed_session_ids: frozenset[str],
) -> str:
    """Run one trusted-capability-authorized, profile-bounded archive synthesis."""
    if not engine._config.cross_session_expansion_enabled:
        return json.dumps({"error": "cross-session expansion is disabled for this profile"})
    scope = "capability_allowlist"

    def _bounded_positive_int(name: str, default: int, hard_max: int) -> tuple[int | None, str | None]:
        try:
            value = int(args.get(name, default))
        except (TypeError, ValueError):
            return None, f"{name} must be an integer"
        if value < 1:
            return None, f"{name} must be positive"
        return min(value, hard_max), None

    max_sessions, error = _bounded_positive_int(
        "max_sessions",
        engine._config.cross_session_max_sessions,
        min(engine._config.cross_session_max_sessions, _CROSS_SESSION_MAX_SESSIONS),
    )
    if error:
        return json.dumps({"error": error})
    per_session_limit = min(
        max_results,
        engine._config.cross_session_max_summaries_per_session,
        _CROSS_SESSION_MAX_SUMMARIES_PER_SESSION,
    )
    configured_deadline_ms = min(
        int(engine._config.cross_session_expansion_deadline_ms),
        int(engine._config.expansion_timeout_ms),
        _CROSS_SESSION_MAX_DEADLINE_MS,
    )
    deadline_ms, error = _bounded_positive_int(
        "deadline_ms",
        configured_deadline_ms,
        configured_deadline_ms,
    )
    if error:
        return json.dumps({"error": error})
    safe_prompt, prompt_truncated = _bounded_cross_session_text(
        prompt,
        engine._config,
        max_tokens=2_000,
        max_chars=20_000,
    )
    safe_query, query_truncated = _bounded_cross_session_text(
        query,
        engine._config,
        max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
        max_chars=2_000,
    )

    guard_key = str(Path(engine._config.database_path or "<default>").resolve())
    with _CROSS_SESSION_EXPANSION_GUARD:
        if guard_key in _ACTIVE_CROSS_SESSION_EXPANSIONS:
            return json.dumps({
                "error": "cross-session expansion already active for this profile",
                "reentry_blocked": True,
            })
        _ACTIVE_CROSS_SESSION_EXPANSIONS.add(guard_key)

    started = time.monotonic()
    deadline = started + (float(deadline_ms) / 1000.0)
    try:
        candidates = []
        candidate_load_truncated = False
        if raw_node_ids:
            for raw_node_id in raw_node_ids:
                if time.monotonic() >= deadline:
                    candidate_load_truncated = True
                    break
                try:
                    node_id = int(raw_node_id)
                except (TypeError, ValueError):
                    return json.dumps({"error": "node_ids must contain only integers"})
                node = engine._dag.get_cross_session_candidate(
                    node_id, deadline=deadline
                )
                if node is not None:
                    candidates.append(node)
                else:
                    # Missing, malformed, oversized, and deadline-exhausted
                    # candidates all fail closed.  Do not issue a second,
                    # potentially unbounded getter merely to distinguish them.
                    candidate_load_truncated = True
        elif query:
            discovery_limit = max_sessions * per_session_limit * 8
            candidates = engine._dag.search_cross_session_candidates(
                query,
                limit=min(discovery_limit, _CROSS_SESSION_AUTH_MAX_CANDIDATES),
                deadline=deadline,
            )
            candidate_load_truncated = time.monotonic() >= deadline
        else:
            return json.dumps({"error": "Provide either query or node_ids"})

        owner_scoped_candidates = [
            node for node in candidates if node.session_id in allowed_session_ids
        ]
        candidates, authorization, frozen_evidence = _authorize_node_provenance_bounded(
            engine,
            owner_scoped_candidates,
            allowed_session_ids,
            deadline=deadline,
        )
        authorization["authorization_truncated"] = bool(
            authorization["authorization_truncated"] or candidate_load_truncated
        )

        # Search order is relevance order. First bucket occurrence therefore
        # ranks sessions before any source expansion occurs.
        buckets: dict[str, list[Any]] = {}
        for node in candidates:
            bucket = buckets.setdefault(node.session_id, [])
            if len(bucket) < per_session_limit:
                bucket.append(node)

        ranked_session_ids = list(buckets)
        selected_session_ids = ranked_session_ids[:max_sessions]
        skipped_buckets = [
            {
                "session_id": _bounded_cross_session_text(
                    session_id,
                    engine._config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
                )[0],
                "reason": "max-sessions",
            }
            for session_id in ranked_session_ids[max_sessions:]
        ]
        if not selected_session_ids:
            payload = {
                "prompt": safe_prompt,
                "prompt_truncated": prompt_truncated,
                "query": safe_query,
                "query_truncated": query_truncated,
                "answer": "No authorized matching summary nodes found in the selected LCM sessions.",
                "cross_session": True,
                "session_scope": scope,
                "contributing_session_ids": [],
                "node_ids": [],
                "matches": [],
                "session_status": [],
                "skipped_buckets": skipped_buckets,
                "context_truncated": False,
                "externalized_refs": "metadata-only",
                **authorization,
            }
            if (
                authorization["authorization_truncated"]
                or authorization["authorization_timed_out"]
            ):
                payload.update({
                    "degraded": True,
                    "timed_out": bool(
                        authorization["authorization_timed_out"]
                    ),
                    "context_truncated": True,
                })
            return json.dumps(payload)

        context_blocks: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        session_status: list[dict[str, Any]] = []
        context_used = 0
        timed_out = False
        for session_id in selected_session_ids:
            safe_session_id = _bounded_cross_session_text(
                session_id,
                engine._config,
                max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
            )[0]
            bucket_blocks_before = len(context_blocks)
            expanded_nodes = 0
            bucket_truncated = False
            bucket_state = "complete"
            for node in buckets[session_id]:
                if time.monotonic() >= deadline:
                    timed_out = True
                    bucket_state = "timed-out"
                    bucket_truncated = True
                    break
                remaining = max(0, context_max_tokens - context_used)
                if remaining <= 0:
                    bucket_state = "budget-exhausted"
                    bucket_truncated = True
                    break
                blocks = _collect_context_blocks_for_node(
                    engine,
                    node,
                    max_tokens=remaining,
                    hydrate_externalized_content=False,
                    allowed_session_id=session_id,
                    frozen_evidence=frozen_evidence,
                )
                for block in blocks:
                    block["session_id"] = safe_session_id
                blocks, metadata_truncated = _bound_cross_session_context(
                    blocks,
                    max_tokens=remaining,
                    config=engine._config,
                )
                context_blocks.extend(blocks)
                context_used += _context_content_token_count(blocks)
                expanded_nodes += 1
                safe_summary, summary_truncated = _bounded_cross_session_text(
                    node.summary,
                    engine._config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=300,
                )
                safe_hint, hint_truncated = _bounded_cross_session_text(
                    node.expand_hint,
                    engine._config,
                    max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                    max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
                )
                matches.append({
                    "session_id": safe_session_id,
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "summary": safe_summary,
                    "summary_truncated": summary_truncated,
                    "expand_hint": safe_hint,
                    "expand_hint_truncated": hint_truncated,
                })
                if metadata_truncated or any(
                    block.get("summary_truncated")
                    or block.get("pagination", {}).get("has_more")
                    for block in blocks
                ):
                    bucket_truncated = True
                if time.monotonic() >= deadline:
                    timed_out = True
                    bucket_state = "timed-out"
                    bucket_truncated = True
                    break
            session_status.append({
                "session_id": safe_session_id,
                "status": bucket_state,
                "candidate_nodes": len(buckets[session_id]),
                "expanded_nodes": expanded_nodes,
                "context_blocks": len(context_blocks) - bucket_blocks_before,
                "truncated": bucket_truncated,
            })
            if timed_out:
                for skipped_session_id in selected_session_ids[len(session_status):]:
                    safe_skipped_session_id = _bounded_cross_session_text(
                        skipped_session_id,
                        engine._config,
                        max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
                        max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
                    )[0]
                    skipped_buckets.append({
                        "session_id": safe_skipped_session_id,
                        "reason": "deadline",
                    })
                break

        contributing_session_ids = [
            item["session_id"] for item in session_status if item["context_blocks"] > 0
        ]
        node_ids = [match["node_id"] for match in matches]
        context_truncated = timed_out or context_used >= context_max_tokens or any(
            item["truncated"] for item in session_status
        ) or bool(skipped_buckets)
        base_payload: dict[str, Any] = {
            "prompt": safe_prompt,
            "prompt_truncated": prompt_truncated,
            "query": safe_query,
            "query_truncated": query_truncated,
            "cross_session": True,
            "session_scope": scope,
            "max_sessions": max_sessions,
            "max_summaries_per_session": per_session_limit,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_tokens_used": context_used,
            "deadline_ms": deadline_ms,
            "timed_out": timed_out,
            "context_truncated": context_truncated,
            "externalized_refs": "metadata-only",
            "contributing_session_ids": contributing_session_ids,
            "node_ids": node_ids,
            "matches": matches,
            "session_status": session_status,
            "skipped_buckets": skipped_buckets,
            **authorization,
        }
        if not context_blocks:
            base_payload.update({
                "answer": "No evidence fit within the shared cross-session expansion bounds.",
                "degraded": True,
            })
            return json.dumps(base_payload)

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            base_payload.update({
                "error": "cross-session expansion deadline exhausted before synthesis",
                "degraded": True,
            })
            return json.dumps(base_payload)
        model = engine._config.expansion_model or engine._config.summary_model or ""
        safe_model, model_truncated = _bounded_cross_session_text(
            model,
            engine._config,
            max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
            max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
        )
        try:
            answer = _synthesize_expansion_answer(
                prompt=prompt,
                context_blocks=context_blocks,
                model=model,
                max_tokens=max_tokens,
                timeout=remaining_seconds,
            )
        except TimeoutError:
            base_payload.update({
                "error": "cross-session expansion synthesis timed out",
                "degraded": True,
                "timed_out": True,
                "context_truncated": True,
                "model": safe_model,
                "model_truncated": model_truncated,
            })
            return json.dumps(base_payload)
        except Exception:
            logger.warning(
                "LCM cross-session synthesis failed after bounded evidence collection"
            )
            base_payload.update({
                "error": "cross-session expansion synthesis failed",
                "degraded": True,
                "model": safe_model,
                "model_truncated": model_truncated,
            })
            return json.dumps(base_payload)
        answer = str(answer or "").strip()
        if not answer:
            base_payload.update({
                "error": "cross-session expansion synthesis returned an empty answer",
                "degraded": True,
                "model": safe_model,
                "model_truncated": model_truncated,
            })
            return json.dumps(base_payload)
        bounded_answer, answer_truncated = _bounded_cross_session_text(
            answer,
            engine._config,
            max_tokens=max_tokens,
            max_chars=None,
        )
        base_payload.update({
            "answer": bounded_answer,
            "answer_truncated": answer_truncated,
            "model": safe_model,
            "model_truncated": model_truncated,
        })
        return json.dumps(base_payload)
    finally:
        with _CROSS_SESSION_EXPANSION_GUARD:
            _ACTIVE_CROSS_SESSION_EXPANSIONS.discard(guard_key)


def _parse_load_session_roles(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], "roles must be an array of strings"
    if len(value) > _LCM_LOAD_SESSION_MAX_ROLES:
        return [], "roles count exceeds hard cap"
    roles: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return [], "roles must contain only strings"
        role = item.strip()
        if not role:
            return [], "roles must contain only non-empty strings"
        if len(role) > _LCM_LOAD_SESSION_MAX_ROLE_CHARS:
            return [], "role string exceeds hard character cap"
        if role not in seen:
            roles.append(role)
            seen.add(role)
            if len(roles) > _LCM_LOAD_SESSION_MAX_ROLES:
                return [], "roles count exceeds hard cap"
    return roles, None


def _slice_loaded_content(
    content: Any,
    max_content_chars: int,
    *,
    source_content_chars: int | None = None,
) -> dict[str, Any]:
    raw_text = str(content or "")
    total_source_chars = (
        int(source_content_chars)
        if isinstance(source_content_chars, int) and source_content_chars >= 0
        else len(raw_text)
    )
    return _slice_redacted_raw_page(
        raw_text,
        content_offset=0,
        max_tokens=max(1, max_content_chars * 2),
        max_chars=min(max(0, int(max_content_chars)), total_source_chars),
        source_content_chars=total_source_chars,
        allow_incomplete_tail=True,
        allow_partial_redaction_without_progress=False,
    )


def _bounded_loaded_nested_value(
    value: Any,
    *,
    char_budget: list[int],
    item_budget: list[int],
    depth: int = 0,
) -> tuple[Any, bool]:
    """Mandatory-redact and bound every nested value in a loaded row."""
    if depth > _LCM_LOAD_SESSION_MAX_NESTED_DEPTH or item_budget[0] <= 0:
        return "[LCM nested value truncated]", True
    item_budget[0] -= 1
    if isinstance(value, str):
        allowed = max(0, char_budget[0])
        if allowed <= 0:
            return "[LCM string truncated]", True
        bounded, truncated = _bounded_cross_session_text(
            value,
            None,
            max_tokens=max(1, allowed * 2),
            max_chars=allowed,
        )
        char_budget[0] = max(0, char_budget[0] - len(bounded))
        return bounded, truncated
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        truncated = False
        for index, (key, item) in enumerate(value.items()):
            if index >= _LCM_LOAD_SESSION_MAX_NESTED_ITEMS or item_budget[0] <= 0:
                truncated = True
                break
            safe_key, key_truncated = _bounded_loaded_nested_value(
                key if isinstance(key, str) else str(key),
                char_budget=char_budget,
                item_budget=item_budget,
                depth=depth + 1,
            )
            safe_item, item_truncated = _bounded_loaded_nested_value(
                item,
                char_budget=char_budget,
                item_budget=item_budget,
                depth=depth + 1,
            )
            result[str(safe_key)] = safe_item
            truncated = truncated or key_truncated or item_truncated
        return result, truncated
    if isinstance(value, (list, tuple)):
        result = []
        truncated = False
        for index, item in enumerate(value):
            if index >= _LCM_LOAD_SESSION_MAX_NESTED_ITEMS or item_budget[0] <= 0:
                truncated = True
                break
            safe_item, item_truncated = _bounded_loaded_nested_value(
                item,
                char_budget=char_budget,
                item_budget=item_budget,
                depth=depth + 1,
            )
            result.append(safe_item)
            truncated = truncated or item_truncated
        return result, truncated
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    return _bounded_loaded_nested_value(
        str(value), char_budget=char_budget, item_budget=item_budget, depth=depth + 1
    )


def _serialize_loaded_message(engine: "LCMEngine", row: dict[str, Any], max_content_chars: int) -> dict[str, Any]:
    stored_session_id = row.get("session_id", "")
    stored_content_chars = row.get("content_chars")
    content_slice = _slice_loaded_content(
        row.get("content", "") or "",
        max_content_chars,
        source_content_chars=(
            stored_content_chars
            if isinstance(stored_content_chars, int) and stored_content_chars >= 0
            else None
        ),
    )
    safe_session_id, _ = _bounded_cross_session_text(
        stored_session_id, None, max_tokens=512, max_chars=512
    )
    safe_source, _ = _bounded_cross_session_text(
        row.get("source") or "", None, max_tokens=512, max_chars=512
    )
    safe_role, _ = _bounded_cross_session_text(
        row.get("role") or "", None, max_tokens=128, max_chars=128
    )
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": safe_session_id,
        "source": safe_source,
        "role": safe_role,
        "timestamp": row.get("timestamp", 0),
        "content": content_slice["content"],
        "content_chars": content_slice["content_chars"],
        "content_returned_chars": content_slice["content_returned_chars"],
        "content_truncated": content_slice["content_truncated"],
        "content_source_truncated": content_slice["content_source_truncated"],
        "content_output_truncated": content_slice["content_output_truncated"],
        "content_redacted": content_slice["content_redacted"],
        "next_content_offset": content_slice["next_content_offset"],
        "from_current_session": bool(engine.current_session_id) and stored_session_id == engine.current_session_id,
    }
    nested_truncated = False
    char_budget = [max(1, _LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES // 2)]
    item_budget = [_LCM_LOAD_SESSION_MAX_NESTED_ITEMS]
    if row.get("tool_call_id"):
        item["tool_call_id"], changed = _bounded_loaded_nested_value(
            row.get("tool_call_id"), char_budget=char_budget, item_budget=item_budget
        )
        nested_truncated = nested_truncated or changed
    if row.get("tool_calls"):
        item["tool_calls"], changed = _bounded_loaded_nested_value(
            row.get("tool_calls"), char_budget=char_budget, item_budget=item_budget
        )
        nested_truncated = nested_truncated or changed
    elif row.get("tool_calls_encoded_too_large"):
        item["tool_calls_truncated"] = True
        item["tool_calls_encoded_bytes"] = int(
            row.get("tool_calls_encoded_bytes") or 0
        )
        nested_truncated = True
    if row.get("tool_name"):
        item["tool_name"], changed = _bounded_loaded_nested_value(
            row.get("tool_name"), char_budget=char_budget, item_budget=item_budget
        )
        nested_truncated = nested_truncated or changed
    encoded_size = len(json.dumps(item, ensure_ascii=False, default=str).encode("utf-8"))
    if encoded_size > _LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES:
        item.pop("tool_calls", None)
        item["tool_calls_truncated"] = True
        nested_truncated = True
        encoded_size = len(json.dumps(item, ensure_ascii=False, default=str).encode("utf-8"))
    item["serialized_truncated"] = bool(
        nested_truncated or content_slice["content_truncated"]
    )
    item["serialized_bytes"] = encoded_size
    return item


def lcm_load_session(args: Dict[str, Any], **kwargs) -> str:
    """Load an ordered, bounded raw-message page for one explicit session_id."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = str(args.get("session_id") or "").strip()
    if not session_id:
        return json.dumps({"error": "session_id is required"})
    if len(session_id) > _LCM_LOAD_SESSION_MAX_SESSION_ID_CHARS:
        return json.dumps({"error": "session_id exceeds hard character cap"})
    if session_id != engine.current_session_id:
        allowed_session_ids = engine._authorized_cross_session_ids(
            kwargs.get("cross_session_capability")
        )
        if allowed_session_ids is None or session_id not in allowed_session_ids:
            return json.dumps({
                "error": "cross-session load requires a trusted host capability"
            })

    raw_limit_arg = args.get("limit", _LCM_LOAD_SESSION_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_LOAD_SESSION_HARD_LIMIT_CAP)

    raw_max_content_chars = args.get("max_content_chars", _LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS)
    max_content_chars, max_content_error = _parse_strict_int(raw_max_content_chars, "max_content_chars")
    if max_content_error:
        return json.dumps({"error": max_content_error})
    if max_content_chars is None or max_content_chars <= 0:
        return json.dumps({"error": "max_content_chars must be a positive integer"})
    requested_max_content_chars = max_content_chars
    max_content_chars = min(max_content_chars, _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS)

    after_store_id, cursor_error = _parse_strict_int(args.get("after_store_id", 0), "after_store_id")
    if cursor_error:
        return json.dumps({"error": cursor_error})
    if after_store_id is None or after_store_id < 0:
        return json.dumps({"error": "after_store_id must be a non-negative integer"})

    roles, roles_error = _parse_load_session_roles(args.get("roles"))
    if roles_error:
        return json.dumps({"error": roles_error})

    time_from, time_from_error = _parse_optional_float(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_float(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})

    try:
        rows = engine._store.load_session_page(
            session_id,
            after_store_id=after_store_id,
            limit=limit + 1,
            roles=roles or None,
            time_from=time_from,
            time_to=time_to,
            max_content_chars=max_content_chars,
            content_lookahead_chars=_MANDATORY_REDACTION_LOOKAHEAD_CHARS,
            max_serialized_bytes=_LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES,
            max_row_serialized_bytes=_LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    total_messages = int(getattr(rows, "total_messages", 0))
    page_rows = rows[:limit]
    storage_budget_exhausted = bool(getattr(rows, "budget_exhausted", False))
    has_more = len(rows) > limit or storage_budget_exhausted
    next_cursor = page_rows[-1]["store_id"] if has_more and page_rows else None

    serialized_messages: list[dict[str, Any]] = []
    shared_messages_bytes = 0
    shared_budget_exhausted = storage_budget_exhausted
    response_reserve = min(
        16_384, max(1_024, _LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES // 4)
    )
    message_byte_limit = max(
        0, _LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES - response_reserve
    )
    for row in page_rows:
        item = _serialize_loaded_message(engine, row, max_content_chars)
        item_bytes = len(json.dumps(item, ensure_ascii=False, default=str).encode("utf-8"))
        if shared_messages_bytes + item_bytes > message_byte_limit:
            shared_budget_exhausted = True
            has_more = True
            break
        serialized_messages.append(item)
        shared_messages_bytes += item_bytes

    if has_more:
        next_cursor = (
            serialized_messages[-1]["store_id"] if serialized_messages else after_store_id
        )

    response: dict[str, Any] = {
        "session_id": session_id,
        "limit": limit,
        "max_content_chars": max_content_chars,
        "after_store_id": after_store_id,
        "total_messages": total_messages,
        "returned_messages": len(serialized_messages),
        "messages": serialized_messages,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "serialized_byte_limit": _LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES,
        "serialized_budget_exhausted": shared_budget_exhausted,
    }
    if roles:
        response["roles"] = roles
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if requested_limit > _LCM_LOAD_SESSION_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_max_content_chars > _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS:
        response["max_content_chars_clamped_from"] = requested_max_content_chars
    response["serialized_bytes"] = 0
    while True:
        for _ in range(8):
            encoded = json.dumps(
                response, ensure_ascii=False, default=str, separators=(",", ":")
            )
            serialized_bytes = len(encoded.encode("utf-8"))
            if serialized_bytes == response["serialized_bytes"]:
                break
            response["serialized_bytes"] = serialized_bytes
        encoded = json.dumps(
            response, ensure_ascii=False, default=str, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) <= _LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES:
            return encoded
        if response["messages"]:
            response["messages"].pop()
            response["returned_messages"] = len(response["messages"])
            response["has_more"] = True
            response["serialized_budget_exhausted"] = True
            response["next_cursor"] = (
                response["messages"][-1]["store_id"]
                if response["messages"] else after_store_id
            )
            response["serialized_bytes"] = 0
            continue
        return json.dumps(
            {
                "error": "load-session response metadata exceeds serialized hard cap",
                "serialized_byte_limit": _LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES,
            },
            separators=(",", ":"),
        )


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search raw messages + summaries with optional cross-session scoping.

    Default scope is the current session, preserving historical behavior and returning
    both raw-message and summary-node hits. Callers may explicitly request
    ``session_scope='all'`` (every session in the local LCM database) or
    ``session_scope='session'`` (a single ``session_id``); broader scopes return
    raw-message hits only and exist for bounded archive recovery over rows already
    present in ``lcm.db``. ``limit`` is clamped to ``_LCM_GREP_HARD_LIMIT_CAP``
    regardless of input.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    operation_started = time.monotonic()
    operation_deadline = operation_started + _LCM_GREP_OPERATION_DEADLINE_SECONDS
    raw_query = args.get("query", "")
    if not isinstance(raw_query, str):
        return json.dumps({"error": "query must be a string"})
    if len(raw_query) > _LCM_GREP_QUERY_MAX_CHARS:
        return json.dumps({
            "error": (
                f"query exceeds the {_LCM_GREP_QUERY_MAX_CHARS} character hard limit"
            ),
        })
    query = raw_query.strip()
    if not query:
        return json.dumps({"error": "No query provided"})
    from .tokens import count_tokens
    if count_tokens(query) > _LCM_GREP_QUERY_MAX_TOKENS:
        return json.dumps({
            "error": (
                f"query exceeds the {_LCM_GREP_QUERY_MAX_TOKENS} token hard limit"
            ),
        })

    content_scope, metadata_error = _parse_bounded_grep_text(
        args.get("content_scope"),
        "content_scope",
        default="database",
        max_chars=_LCM_GREP_SCOPE_MAX_CHARS,
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    content_scope = str(content_scope or "database").lower()
    if content_scope == "files":
        content_scope = "externalized"
    if content_scope not in {"database", "externalized", "all"}:
        return json.dumps({
            "error": "content_scope must be database, externalized, or all",
        })
    raw_regex_mode = args.get("regex", False)
    if not isinstance(raw_regex_mode, bool):
        return json.dumps({"error": "regex must be a boolean"})
    regex_mode = raw_regex_mode
    if regex_mode:
        if len(query) > _LCM_GREP_REGEX_MAX_PATTERN_CHARS:
            return json.dumps({"error": "regex pattern exceeds 2000 character hard limit"})
        try:
            re.compile(query)
        except re.error as exc:
            return json.dumps({"error": f"invalid regex: {exc}"})
    ref, metadata_error = _parse_bounded_grep_text(
        args.get("ref"), "ref", max_chars=_LCM_GREP_REF_MAX_CHARS
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    ref = ref or ""
    if ref and (Path(ref).name != ref or "/" in ref or "\\" in ref):
        return json.dumps({"error": "invalid externalized ref"})
    if ref and content_scope == "database":
        return json.dumps({"error": "ref requires externalized content_scope"})

    requested_max_files, numeric_error = _parse_bounded_grep_int(
        args.get("max_files"),
        "max_files",
        default=_LCM_GREP_EXTERNALIZED_DEFAULT_FILES,
    )
    if numeric_error:
        return json.dumps({"error": numeric_error})
    requested_max_payload_chars, numeric_error = _parse_bounded_grep_int(
        args.get("max_payload_chars"),
        "max_payload_chars",
        default=_LCM_GREP_EXTERNALIZED_DEFAULT_CHARS,
    )
    if numeric_error:
        return json.dumps({"error": numeric_error})
    assert requested_max_files is not None
    assert requested_max_payload_chars is not None
    if requested_max_files <= 0 or requested_max_payload_chars <= 0:
        return json.dumps({"error": "externalized scan bounds must be positive"})
    max_files = min(requested_max_files, _LCM_GREP_EXTERNALIZED_MAX_FILES)
    max_payload_chars = min(
        requested_max_payload_chars,
        _LCM_GREP_EXTERNALIZED_MAX_CHARS,
    )

    raw_limit_arg = args.get("limit", 10)
    parsed_limit, numeric_error = _parse_bounded_grep_int(
        raw_limit_arg, "limit", default=10
    )
    if numeric_error:
        return json.dumps({"error": numeric_error})
    assert parsed_limit is not None
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    raw_sort, metadata_error = _parse_bounded_grep_text(
        args.get("sort"),
        "sort",
        default="recency",
        max_chars=_LCM_GREP_SORT_MAX_CHARS,
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    sort = normalize_search_sort(raw_sort)
    source_limit = max(limit * 4, limit, 20)

    requested_session_scope, metadata_error = _parse_bounded_grep_text(
        args.get("session_scope"),
        "session_scope",
        default="current",
        max_chars=_LCM_GREP_SCOPE_MAX_CHARS,
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    requested_session_scope = str(requested_session_scope or "current").lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id, metadata_error = _parse_bounded_grep_text(
        raw_session_id_arg,
        "session_id",
        max_chars=_LCM_GREP_SESSION_ID_MAX_CHARS,
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    explicit_session_id = explicit_session_id or ""
    source, metadata_error = _parse_bounded_grep_text(
        args.get("source"), "source", max_chars=_LCM_GREP_SOURCE_MAX_CHARS
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    source = source or None
    conversation_id, metadata_error = _parse_bounded_grep_text(
        args.get("conversation_id"),
        "conversation_id",
        max_chars=_LCM_GREP_CONVERSATION_ID_MAX_CHARS,
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    conversation_id = conversation_id or None
    raw_role, metadata_error = _parse_bounded_grep_text(
        args.get("role"), "role", max_chars=_LCM_GREP_ROLE_MAX_CHARS
    )
    if metadata_error:
        return json.dumps({"error": metadata_error})
    role, role_error = _parse_grep_role(raw_role or None)
    if role_error:
        return json.dumps({"error": role_error})
    for timestamp_name in ("time_from", "time_to"):
        timestamp_value = args.get(timestamp_name)
        if isinstance(timestamp_value, str) and len(timestamp_value) > _LCM_GREP_TIMESTAMP_MAX_CHARS:
            return json.dumps({
                "error": (
                    f"{timestamp_name} exceeds the "
                    f"{_LCM_GREP_TIMESTAMP_MAX_CHARS} character hard limit"
                )
            })
        if timestamp_value is not None and not isinstance(
            timestamp_value, (str, int, float)
        ):
            return json.dumps({
                "error": f"{timestamp_name} must be a Unix timestamp or timezone-aware ISO 8601 string"
            })
    time_from, time_from_error = _parse_optional_timestamp(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_timestamp(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})
    raw_message_filter_active = (
        role is not None
        or time_from is not None
        or time_to is not None
        or conversation_id is not None
    )

    allowed_cross_session_ids: frozenset[str] | None = None
    if requested_session_scope == "current":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is only valid with session_scope=session",
            })
        # MessageStore.search and SummaryDAG.search treat session_id="" as a
        # literal scoped filter, so an unbound engine searching scope=current
        # returns zero results rather than leaking cross-session matches.
        # Read current_session_id (the foreground view) so a cron-style side
        # channel that briefly owns engine._session_id does not redirect the
        # default search scope away from the operator's real conversation.
        search_session_id: str | None = engine.current_session_id
        session_scope = "current"
    elif requested_session_scope == "all":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is not used with session_scope=all",
            })
        allowed_cross_session_ids = engine._authorized_cross_session_ids(
            kwargs.get("cross_session_capability")
        )
        if not allowed_cross_session_ids:
            return json.dumps({
                "error": "cross-session grep requires a trusted host capability",
            })
        search_session_id = None
        session_scope = "all"
    elif requested_session_scope == "session":
        if not explicit_session_id:
            return json.dumps({
                "error": "session_scope=session requires session_id",
            })
        allowed_cross_session_ids = engine._authorized_cross_session_ids(
            kwargs.get("cross_session_capability")
        )
        if not allowed_cross_session_ids:
            return json.dumps({
                "error": "cross-session grep requires a trusted host capability",
            })
        if explicit_session_id not in allowed_cross_session_ids:
            return json.dumps({
                "error": "session is not authorized by the trusted host capability",
            })
        search_session_id = explicit_session_id
        session_scope = "session"
    else:
        # Preserve historical behavior for unknown scopes: route through the
        # current-session path and report. The data-layer empty-string scoping
        # contract keeps an unbound engine from leaking cross-session matches
        # here too.
        search_session_id = engine.current_session_id
        session_scope = "current"
        logger.warning(
            "Ignoring unsupported session_scope=%s for lcm_grep",
            requested_session_scope,
        )

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    results: list[Dict[str, Any]] = []
    externalized_diagnostics: list[dict[str, str]] = []
    externalized_scan: dict[str, Any] | None = None
    externalized_completions: list[
        _ExternalizedContinuationCompletion
    ] = []
    operation_rows_reserved = 0
    operation_rows_materialized = 0
    charged_input_text = (
        query,
        content_scope,
        ref,
        raw_sort or "",
        requested_session_scope,
        explicit_session_id,
        source or "",
        conversation_id or "",
        role or "",
        str(args.get("time_from") or ""),
        str(args.get("time_to") or ""),
    )
    operation_bytes_materialized = sum(
        len(value.encode("utf-8", errors="surrogatepass"))
        for value in charged_input_text
    )
    operation_budget_exhausted = False

    def reserve_discovery_rows(requested: int) -> int:
        nonlocal operation_rows_reserved, operation_budget_exhausted
        if time.monotonic() >= operation_deadline:
            operation_budget_exhausted = True
            return 0
        remaining_rows = _LCM_GREP_OPERATION_MAX_ROWS - operation_rows_reserved
        allowed = min(
            max(0, int(requested)),
            max(0, remaining_rows),
        )
        if allowed <= 0:
            operation_budget_exhausted = True
            return 0
        operation_rows_reserved += allowed
        if allowed < max(0, int(requested)) or (
            operation_rows_reserved >= _LCM_GREP_OPERATION_MAX_ROWS
        ):
            operation_budget_exhausted = True
        return allowed

    def charge_materialized(value: Any) -> bool:
        nonlocal operation_rows_materialized, operation_bytes_materialized
        nonlocal operation_budget_exhausted
        if time.monotonic() >= operation_deadline:
            operation_budget_exhausted = True
            return False
        if operation_rows_materialized >= _LCM_GREP_OPERATION_MAX_ROWS:
            operation_budget_exhausted = True
            return False
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8", errors="surrogatepass")
        except (TypeError, ValueError, OverflowError):
            operation_budget_exhausted = True
            return False
        if (
            operation_bytes_materialized + len(encoded)
            > _LCM_GREP_OPERATION_MAX_BYTES
        ):
            operation_budget_exhausted = True
            return False
        operation_rows_materialized += 1
        operation_bytes_materialized += len(encoded)
        if operation_rows_materialized >= _LCM_GREP_OPERATION_MAX_ROWS:
            operation_budget_exhausted = True
        return True

    # External hits are materialized candidates too. Reserve their maximum
    # cardinality before database discovery so mixed searches cannot spend the
    # complete operation row budget and then append an unreserved file batch.
    external_rows_reserved = 0
    if content_scope in {"externalized", "all"}:
        external_rows_reserved = reserve_discovery_rows(1 if ref else max_files)
        if operation_rows_reserved >= _LCM_GREP_OPERATION_MAX_ROWS:
            operation_budget_exhausted = True

    if content_scope in {"database", "all"} and not regex_mode:
      try:
        message_search_sessions = (
            sorted(allowed_cross_session_ids)
            if session_scope == "all" and allowed_cross_session_ids is not None
            else [search_session_id]
        )
        # Filters that suppress summaries, and raw-only all-session recovery,
        # may use the full row budget. Mixed current/session searches reserve a
        # larger share for summary discovery so source filtering and
        # directness ranking can page past newer unrelated nodes.
        raw_candidate_pool = min(
            (
                _LCM_GREP_OPERATION_MAX_ROWS
                if raw_message_filter_active or session_scope == "all"
                else 400
            ),
            _LCM_GREP_OPERATION_MAX_ROWS,
        )
        raw_candidates_reserved = 0
        for session_index, authorized_session_id in enumerate(message_search_sessions):
            sessions_remaining = len(message_search_sessions) - session_index
            raw_candidates_remaining = raw_candidate_pool - raw_candidates_reserved
            candidate_allowance = min(
                raw_candidates_remaining,
                max(
                    1,
                    (raw_candidates_remaining + sessions_remaining - 1)
                    // max(1, sessions_remaining),
                ),
            )
            candidate_limit = reserve_discovery_rows(candidate_allowance)
            if candidate_limit <= 0:
                break
            raw_candidates_reserved += candidate_limit
            call_limit = min(source_limit, candidate_limit)
            msg_hits = engine._store.search(
                query,
                session_id=authorized_session_id,
                limit=call_limit,
                sort=sort,
                source=source,
                conversation_id=conversation_id,
                role=role,
                time_from=time_from,
                time_to=time_to,
                bounded_output=True,
                max_candidate_rows=candidate_limit,
                deadline=operation_deadline,
                max_materialized_bytes=max(
                    0,
                    _LCM_GREP_OPERATION_MAX_BYTES - operation_bytes_materialized,
                ),
            )
            for hit in msg_hits:
                if not charge_materialized(hit):
                    break
                if (
                    allowed_cross_session_ids is not None
                    and str(hit.get("session_id") or "") not in allowed_cross_session_ids
                ):
                    continue
                timestamp_value = hit.get("timestamp", 0) or 0
                results.append(
                    {
                        "type": "message",
                        "depth": "raw",
                        "store_id": hit["store_id"],
                        "session_id": _grep_safe_text(
                            hit.get("session_id") or "", max_chars=512
                        ),
                        "source": _grep_safe_text(
                            hit.get("source") or "", max_chars=512
                        ),
                        "conversation_id": _grep_safe_text(
                            hit.get("conversation_id") or "", max_chars=512
                        ),
                        "role": _grep_safe_text(hit.get("role") or "", max_chars=128),
                        "timestamp": timestamp_value,
                        "snippet": _grep_safe_snippet(
                            hit.get("snippet", hit.get("content", "")),
                            query,
                            max_chars=300,
                            highlight_matches=bool(
                                hit.get("_grep_highlight_matches", True)
                            ),
                            left_boundary_truncated=(
                                int(hit.get("_grep_window_start") or 1) > 1
                            ),
                        ),
                        "from_current_session": has_current_session and hit["session_id"] == current_session_id,
                        "_sort_ts": timestamp_value,
                        "_sort_rank": hit.get("search_rank"),
                        "_sort_directness": hit.get("_directness_score") or 0.0,
                    }
                )
            if operation_budget_exhausted:
                break
      except Exception as exc:
        logger.warning("Message search failed: %s", exc)

    # Summary search uses an explicit SQL prefix projection for every scope.
    # Cross-session rows are constrained by the already-validated host
    # capability, just like raw-message rows.
    if (
        content_scope in {"database", "all"}
        and not regex_mode
        and not raw_message_filter_active
        and session_scope != "all"
    ):
        try:
            summary_search_sessions = (
                sorted(allowed_cross_session_ids)
                if session_scope == "all" and allowed_cross_session_ids is not None
                else [str(search_session_id or "")]
            )
            summary_candidate_limit = reserve_discovery_rows(
                _LCM_GREP_OPERATION_MAX_ROWS - operation_rows_reserved
            )
            node_hits = (
                engine._dag.search_bounded_summary_candidates(
                    query,
                    session_ids=summary_search_sessions,
                    limit=source_limit,
                    sort=sort,
                    source=source,
                    deadline=min(
                        operation_deadline,
                        time.monotonic() + _LCM_GREP_SUMMARY_DEADLINE_SECONDS,
                    ),
                    max_materialized_bytes=max(
                        0,
                        _LCM_GREP_OPERATION_MAX_BYTES - operation_bytes_materialized,
                    ),
                    max_candidate_rows=summary_candidate_limit,
                )
                if summary_candidate_limit > 0
                else []
            )
            for node in node_hits:
                if not charge_materialized({
                    "node_id": node.node_id,
                    "session_id": node.session_id,
                    "summary": node.summary,
                    "expand_hint": node.expand_hint,
                    "source_ids": node.source_ids,
                    "source_type": node.source_type,
                }):
                    break
                if (
                    allowed_cross_session_ids is not None
                    and node.session_id not in allowed_cross_session_ids
                ):
                    continue
                results.append(
                    {
                        "type": "summary",
                        "depth": f"d{node.depth}",
                        "node_id": node.node_id,
                        "session_id": _grep_safe_text(
                            node.session_id, max_chars=512
                        ),
                        "snippet": _grep_safe_text(node.summary, max_chars=300),
                        "token_count": node.token_count,
                        "expand_hint": _grep_safe_text(
                            node.expand_hint, max_chars=2_048
                        ),
                        "earliest_at": node.earliest_at,
                        "latest_at": node.latest_at,
                        "from_current_session": (
                            has_current_session
                            and node.session_id == current_session_id
                        ),
                        "_sort_ts": node.latest_at or node.created_at,
                        "_sort_rank": node.search_rank,
                        "_sort_directness": node.search_directness or 0.0,
                    }
                )
        except Exception as exc:
            logger.warning("Node search failed: %s", exc)

    if (
        content_scope in {"externalized", "all"}
        and external_rows_reserved > 0
        and operation_rows_materialized < _LCM_GREP_OPERATION_MAX_ROWS
        and time.monotonic() < operation_deadline
        and operation_bytes_materialized < _LCM_GREP_OPERATION_MAX_BYTES
    ):
        external_hits, externalized_diagnostics, externalized_scan = (
            _search_externalized_payloads(
                engine,
                query=query,
                regex_mode=regex_mode,
                allowed_session_ids=(
                    allowed_cross_session_ids
                    if allowed_cross_session_ids is not None
                    else frozenset({str(search_session_id or "")})
                ),
                ref=ref,
                limit=min(limit, external_rows_reserved),
                max_files=min(max_files, external_rows_reserved),
                max_payload_chars=max_payload_chars,
                max_total_bytes=max(
                    0,
                    _LCM_GREP_OPERATION_MAX_BYTES - operation_bytes_materialized,
                ),
                deadline=operation_deadline,
                completion_acknowledgements=externalized_completions,
            )
        )
        for hit in external_hits:
            if not charge_materialized(hit):
                break
            if (
                allowed_cross_session_ids is not None
                and str(hit.get("session_id") or "") not in allowed_cross_session_ids
            ):
                continue
            hit["from_current_session"] = bool(
                current_session_id
                and hit.get("session_id") == current_session_id
            )
            results.append(hit)

    # Discovery, ranking, redaction, and response construction all share the
    # same deadline.  Once it expires, fail closed by dropping discovered
    # payload rows; bounded/redacted metadata may still report the exhaustion.
    if time.monotonic() >= operation_deadline:
        operation_budget_exhausted = True
        results = []

    if sort == "hybrid":
        max_message_directness = max(
            (float(result.get("_sort_directness") or 0.0) for result in results if result.get("type") == "message"),
            default=0.0,
        )
        for result in results:
            if result.get("type") == "summary":
                result["_hybrid_summary_override"] = 1 if float(result.get("_sort_directness") or 0.0) >= (max_message_directness + 8.0) else 0

    results.sort(key=lambda result: _combined_result_sort_key(result, sort))
    if time.monotonic() >= operation_deadline:
        operation_budget_exhausted = True
        results = []
    for result in results:
        result.pop("_sort_ts", None)
        result.pop("_sort_rank", None)
        result.pop("_sort_directness", None)
        result.pop("_hybrid_summary_override", None)

    response: Dict[str, Any] = {
        "query": query,
        "sort": sort,
        "content_scope": content_scope,
        "regex": regex_mode,
        "session_scope": session_scope,
        "source": source,
        "conversation_id": conversation_id,
        "limit": limit,
        "total_results": len(results),
        "results": results[:limit],
        "operation_budget": {
            "rows_limit": _LCM_GREP_OPERATION_MAX_ROWS,
            "rows_reserved": operation_rows_reserved,
            "rows_materialized": operation_rows_materialized,
            "bytes_limit": _LCM_GREP_OPERATION_MAX_BYTES,
            "bytes_materialized": operation_bytes_materialized,
            "deadline_ms": int(_LCM_GREP_OPERATION_DEADLINE_SECONDS * 1_000),
            "exhausted": operation_budget_exhausted,
        },
    }
    if externalized_scan is not None:
        response["scan"] = externalized_scan
        response["diagnostics"] = externalized_diagnostics
        if requested_max_files > max_files:
            response["max_files_clamped_from"] = requested_max_files
        if requested_max_payload_chars > max_payload_chars:
            response["max_payload_chars_clamped_from"] = requested_max_payload_chars
    if role is not None:
        response["role"] = role
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if raw_message_filter_active:
        response["summary_results_omitted"] = True
    if session_scope == "session":
        response["session_id"] = explicit_session_id
    if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = (
            "Unsupported session_scope; stayed on current. "
            "Valid values: current, all, session."
        )
    # Redact response metadata first, then payload rows one at a time while the
    # operation deadline remains.  Never return a row whose mandatory boundary
    # redaction completed after the deadline.
    response_results = response.pop("results")
    protected_response = _mandatory_redact_grep_response(response)
    protected_response["results"] = []
    for response_result in response_results:
        if time.monotonic() >= operation_deadline:
            protected_response["operation_budget"]["exhausted"] = True
            protected_response["operation_budget"]["deadline_exhausted"] = True
            break
        protected_result = _mandatory_redact_grep_response(response_result)
        if time.monotonic() >= operation_deadline:
            protected_response["operation_budget"]["exhausted"] = True
            protected_response["operation_budget"]["deadline_exhausted"] = True
            break
        protected_response["results"].append(protected_result)
    serialized = json.dumps(protected_response)
    while (
        len(serialized.encode("utf-8", errors="surrogatepass"))
        > _LCM_GREP_OPERATION_MAX_BYTES
        and protected_response.get("results")
    ):
        protected_response["results"].pop()
        protected_response["operation_budget"]["exhausted"] = True
        protected_response["operation_budget"]["response_truncated"] = True
        serialized = json.dumps(protected_response)
    if time.monotonic() >= operation_deadline:
        protected_response["operation_budget"]["exhausted"] = True
        protected_response["operation_budget"]["deadline_exhausted"] = True
        protected_response["results"] = []
        serialized = json.dumps(protected_response)
    if len(serialized.encode("utf-8", errors="surrogatepass")) > _LCM_GREP_OPERATION_MAX_BYTES:
        for completion in externalized_completions:
            completion.preserve()
        return json.dumps({
            "error": "grep response metadata exceeds operation byte budget",
            "operation_budget": {
                "bytes_limit": _LCM_GREP_OPERATION_MAX_BYTES,
                "exhausted": True,
            },
        })
    completed_successfully = not protected_response["operation_budget"].get(
        "exhausted", False
    )
    for completion in externalized_completions:
        if completed_successfully:
            completion.commit()
        else:
            completion.preserve()
    if not completed_successfully and externalized_scan is not None:
        pending_count, pending_memory = _externalized_continuation_stats(engine)
        protected_response["scan"]["continuations_pending"] = pending_count
        protected_response["scan"]["continuation_memory_bytes"] = pending_memory
        serialized = json.dumps(protected_response)
        if (
            len(serialized.encode("utf-8", errors="surrogatepass"))
            > _LCM_GREP_OPERATION_MAX_BYTES
        ):
            return json.dumps({
                "error": "grep response metadata exceeds operation byte budget",
                "operation_budget": {
                    "bytes_limit": _LCM_GREP_OPERATION_MAX_BYTES,
                    "exhausted": True,
                },
            })
    return serialized


def lcm_describe(args: Dict[str, Any], **kwargs) -> str:
    """Inspect a summary node's subtree or get session DAG overview."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    if externalized_ref:
        payload = _get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return json.dumps({"error": f"Externalized payload {externalized_ref} not found in current session"})
        return json.dumps(
            {
                "externalized_ref": externalized_ref,
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
                "content_chars": payload.get("content_chars", 0),
                "content_bytes": payload.get("content_bytes", 0),
                "created_at": payload.get("created_at"),
                "content_preview": (payload.get("content") or "")[:500],
            }
        )

    node_id = args.get("node_id")
    session_id = engine.current_session_id

    if node_id is not None:
        node = _get_session_node(engine, node_id)
        if node is None:
            return json.dumps({"error": f"Node {node_id} not found in current session"})
        info = engine._dag.describe_subtree(node_id)
        return json.dumps(info)

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    depth_samples = engine._dag.get_session_depth_samples(
        session_id,
        per_depth_limit=20,
        depths=list(depth_stats),
    )
    overview = {
        "session_id": session_id,
        "store_message_count": engine._store.get_session_count(session_id),
        "depths": {},
    }

    for depth, stats in sorted(depth_stats.items()):
        nodes = depth_samples.get(depth, [])
        overview["depths"][f"d{depth}"] = {
            "count": stats["count"],
            "total_tokens": stats["tokens"],
            "total_source_tokens": stats["source_tokens"],
            "nodes": [
                {
                    "node_id": node.node_id,
                    "token_count": node.token_count,
                    "expand_hint": node.expand_hint,
                }
                for node in nodes
            ],
        }

    return json.dumps(overview)


def lcm_expand(args: Dict[str, Any], **kwargs) -> str:
    """Expand a summary node, externalized payload, or raw message to its content.

    Mode selection (exactly one is required):
    - ``externalized_ref``: open a stored externalized payload by ref filename (current session only)
    - ``store_id``: fetch a single raw message by store_id; works across sessions
    - ``node_id``: expand a summary node to its source content (current session only)

    Only ``store_id`` mode accepts an arbitrary cross-session target. ``node_id``
    stays current-session scoped, but carried-over current-session nodes may
    reference raw source rows that still belong to the previous session.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    raw_store_id_arg = args.get("store_id")
    raw_node_id_arg = args.get("node_id")

    modes_provided: list[str] = []
    if externalized_ref:
        modes_provided.append("externalized_ref")
    if raw_store_id_arg is not None:
        modes_provided.append("store_id")
    if raw_node_id_arg is not None:
        modes_provided.append("node_id")

    if len(modes_provided) > 1:
        return json.dumps({
            "error": (
                "Provide only one of node_id, externalized_ref, store_id "
                f"(got {', '.join(modes_provided)})"
            ),
        })
    if not modes_provided:
        return json.dumps({
            "error": "node_id, externalized_ref, or store_id is required",
        })

    max_tokens = _parse_positive_int(args.get("max_tokens", 4000), 4000)
    source_offset = _parse_non_negative_int(args.get("source_offset", 0), 0)
    source_limit_arg = args.get("source_limit")
    source_limit = _parse_positive_int(source_limit_arg, 0) if source_limit_arg is not None else None
    content_offset = _parse_non_negative_int(args.get("content_offset", 0), 0)
    if max_tokens > _CURRENT_SESSION_EXPAND_MAX_TOKENS:
        return json.dumps({"error": "max_tokens exceeds the 65536 hard cap"})
    if source_limit is not None and source_limit > _CURRENT_SESSION_EXPAND_MAX_SOURCES:
        return json.dumps({"error": "source_limit exceeds the 200 hard cap"})

    if externalized_ref:
        payload = _get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return json.dumps({"error": f"Externalized payload {externalized_ref} not found in current session"})
        content = payload.get("content", "")
        sliced = _slice_content_for_response(
            content,
            max_tokens,
            content_offset,
            config=engine._config,
        )
        return json.dumps(
            {
                "externalized_ref": externalized_ref,
                "source_type": "externalized_payload",
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
                "content_chars": payload.get("content_chars", len(content)),
                "content_bytes": payload.get("content_bytes", 0),
                "content": sliced["content"],
                "content_offset": sliced["content_offset"],
                "content_returned_chars": sliced["content_returned_chars"],
                "content_truncated": sliced["content_truncated"],
                "next_content_offset": sliced["next_content_offset"],
                "has_more": sliced["has_more"],
            }
        )

    if raw_store_id_arg is not None:
        try:
            store_id = int(raw_store_id_arg)
        except (TypeError, ValueError, OverflowError):
            return json.dumps({"error": "store_id must be an integer"})
        engine_session_id = engine.current_session_id
        allowed_session_ids = engine._authorized_cross_session_ids(
            kwargs.get("cross_session_capability")
        )
        authorization_failure = ""

        def authorize_owner(stored_session_id: str) -> bool:
            nonlocal authorization_failure
            if stored_session_id == engine_session_id:
                return True
            if not allowed_session_ids:
                authorization_failure = "missing_capability"
                return False
            if stored_session_id not in allowed_session_ids:
                authorization_failure = "session_not_allowed"
                return False
            return True

        loaded = engine._store.get_for_expansion(
            store_id,
            authorize_session=authorize_owner,
            content_offset=content_offset,
            max_content_chars=_CURRENT_SESSION_EXPAND_MAX_CHARS,
            content_lookahead_chars=_MANDATORY_REDACTION_LOOKAHEAD_CHARS,
            boundary_scanner=_scan_raw_credential_boundary,
            boundary_scan_deadline=(
                _expand_scan_now() + _EXPAND_BOUNDARY_SCAN_DEADLINE_SECONDS
            ),
        )
        status = loaded.get("status")
        if status == "not_found":
            return json.dumps({"error": f"Message store_id {store_id} not found"})
        if status == "unauthorized":
            if authorization_failure == "missing_capability":
                return json.dumps({
                    "error": "cross-session store_id expansion requires a trusted host capability",
                })
            if authorization_failure == "session_not_allowed":
                return json.dumps({
                    "error": "store_id session is not authorized by the trusted host capability",
                })
            return json.dumps({"error": "store_id expansion authorization failed closed"})
        if status == "scan_pending":
            return json.dumps({
                "store_id": store_id,
                "source_type": "raw_message",
                "session_id": loaded.get("session_id") or "",
                "content_scan_pending": True,
                "content_chars_scanned": int(
                    loaded.get("content_chars_scanned") or 0
                ),
                "content_bytes": int(loaded.get("content_bytes") or 0),
                "content_scan_byte_offset": int(
                    loaded.get("content_scan_byte_offset") or 0
                ),
                "retryable": True,
            })
        if status != "ok":
            return json.dumps({
                "error": f"Message store_id {store_id} has invalid or changed bounded metadata",
            })
        stored = loaded["message"]
        transcript_content = stored.get("content", "") or ""
        total_content_chars = max(0, int(stored.get("content_chars") or 0))
        window_offset = max(0, int(stored.get("content_window_offset") or 0))
        if stored.get("boundary_redacted"):
            boundary_cursor = min(
                total_content_chars,
                max(0, int(stored.get("boundary_next_content_offset") or 0)),
            )
            boundary_pending = bool(stored.get("boundary_pending"))
            boundary_placeholder = (
                str(stored.get("boundary_safe_prefix") or "")
                + "[LCM sensitive redaction: name=bounded_credential; "
                "boundary-truncated]"
            )
            sliced = {
                "content": boundary_placeholder,
                "content_chars": total_content_chars,
                "content_offset": min(content_offset, total_content_chars),
                "content_returned_chars": len(boundary_placeholder),
                "content_truncated": True,
                "next_content_offset": (
                    boundary_cursor if boundary_cursor < total_content_chars else 0
                ),
                "has_more": boundary_cursor < total_content_chars,
                "content_boundary_scan_pending": boundary_pending,
            }
        elif window_offset >= total_content_chars:
            bounded_requested_offset = min(content_offset, total_content_chars)
            sliced = {
                "content": "",
                "content_chars": total_content_chars,
                "content_offset": bounded_requested_offset,
                "content_returned_chars": 0,
                "content_truncated": False,
                "next_content_offset": None,
                "has_more": False,
            }
        else:
            local_offset = max(0, content_offset - window_offset)
            sliced = _slice_redacted_raw_page(
                transcript_content,
                content_offset=local_offset,
                max_tokens=max_tokens,
                max_chars=_CURRENT_SESSION_EXPAND_MAX_CHARS,
                source_content_chars=total_content_chars - window_offset,
                allow_incomplete_tail=True,
            )
            sliced["content_chars"] = total_content_chars
            sliced["content_offset"] = min(
                total_content_chars,
                window_offset + int(sliced["content_offset"]),
            )
            local_has_more = bool(sliced["has_more"])
            if local_has_more:
                sliced["next_content_offset"] = min(
                    total_content_chars,
                    window_offset + int(sliced["next_content_offset"]),
                )
            else:
                # The raw paging contract uses zero (not None) as its terminal
                # cursor.  Do not reinterpret that sentinel after translating
                # window-local coordinates back to source coordinates.
                sliced["next_content_offset"] = 0
            sliced["has_more"] = bool(
                local_has_more
                and sliced["next_content_offset"] < total_content_chars
            )
        stored_session_id = stored.get("session_id", "")
        result: Dict[str, Any] = {
            "store_id": store_id,
            "source_type": "raw_message",
            "session_id": stored_session_id,
            "source": stored.get("source") or "",
            "conversation_id": stored.get("conversation_id") or "",
            "role": stored.get("role"),
            "timestamp": stored.get("timestamp", 0),
            "tool_call_id": stored.get("tool_call_id") or "",
            "from_current_session": bool(engine_session_id) and stored_session_id == engine_session_id,
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "has_more": sliced["has_more"],
        }
        if sliced.get("content_boundary_scan_pending"):
            result["content_boundary_scan_pending"] = True
            result["content_scan_checkpoint_offset"] = int(
                stored.get("boundary_checkpoint_offset") or 0
            )
        # Surface externalized-payload metadata when the row references one. Content
        # is not hydrated by default, mirroring the existing _expand_message_sources
        # default. Externalized lookup remains session-scoped (per the existing
        # _get_externalized_payload contract); cross-session rows surface only the
        # ref string, with a hint pointing at the same-session expansion path.
        ref_values = [transcript_content]
        if stored.get("tool_calls"):
            try:
                ref_values.append(json.dumps(stored.get("tool_calls"), ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError):
                ref_values.append(str(stored.get("tool_calls")))
        refs: list[str] = []
        for value in ref_values:
            if not isinstance(value, str):
                continue
            for found_ref in extract_ingest_externalized_refs(value):
                if found_ref not in refs:
                    refs.append(found_ref)
            legacy_ref = extract_externalized_ref(value)
            if legacy_ref and legacy_ref not in refs:
                refs.append(legacy_ref)
        if refs:
            result["externalized_refs"] = refs
            result["externalized_ref"] = refs[0]
            if bool(engine_session_id) and stored_session_id == engine_session_id:
                payload_summaries = []
                for ref in refs:
                    payload = _get_externalized_payload(engine, ref)
                    if payload is None:
                        continue
                    payload_summary = dict(payload)
                    payload_summary.pop("content", None)
                    payload_summaries.append(payload_summary)
                if payload_summaries:
                    result["externalized_payloads"] = payload_summaries
                    result["externalized"] = payload_summaries[0]
            else:
                result["externalized_note"] = (
                    "Externalized payload metadata is session-scoped; "
                    "cross-session ref is surfaced for traceability only and cannot be expanded in this version."
                )
        return json.dumps(result)

    node_id = raw_node_id_arg

    node = _get_session_node(engine, node_id)
    if node is None:
        return json.dumps({"error": f"Node {node_id} not found in current session"})

    if node.source_type == "messages":
        messages, pagination = _expand_message_sources(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
            content_offset=content_offset,
        )
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "messages",
                "expanded": messages,
                "pagination": pagination,
            }
        )

    if node.source_type == "nodes":
        children, pagination = _expand_child_nodes(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
        )
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "nodes",
                "expanded": children,
                "pagination": pagination,
            }
        )

    return json.dumps({"error": f"Unknown source_type: {node.source_type}"})


def lcm_expand_query(args: Dict[str, Any], **kwargs) -> str:
    """Answer a question by expanding matching summaries or explicit node ids."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return json.dumps({"error": "prompt is required"})
    if len(prompt) > 20_000:
        return json.dumps({"error": "prompt exceeds 20000 characters"})

    def _parse_int_arg(name: str, default: int) -> tuple[int | None, str | None]:
        raw_value = args.get(name, default)
        try:
            return int(raw_value), None
        except (TypeError, ValueError):
            return None, f"{name} must be an integer"

    max_tokens, max_tokens_error = _parse_int_arg("max_tokens", 2000)
    if max_tokens_error:
        return json.dumps({"error": max_tokens_error})
    max_tokens = min(max(1, max_tokens), _CROSS_SESSION_MAX_ANSWER_TOKENS)
    context_default = max(max_tokens, int(getattr(engine._config, "expansion_context_tokens", 32_000) or 32_000))
    context_max_tokens, context_max_tokens_error = _parse_int_arg("context_max_tokens", context_default)
    if context_max_tokens_error:
        return json.dumps({"error": context_max_tokens_error})
    context_max_tokens = min(
        max(1, context_max_tokens),
        _CROSS_SESSION_MAX_CONTEXT_TOKENS,
    )

    max_results, max_results_error = _parse_int_arg("max_results", 5)
    if max_results_error:
        return json.dumps({"error": max_results_error})
    max_results = min(
        max(1, int(max_results or 5)),
        _CROSS_SESSION_MAX_RESULTS,
    )

    query = str(args.get("query") or "").strip()
    if len(query) > 2_000:
        return json.dumps({"error": "query exceeds 2000 characters"})
    raw_node_ids = args.get("node_ids") or []
    if not isinstance(raw_node_ids, list):
        return json.dumps({"error": "node_ids must be an array"})
    if len(raw_node_ids) > _CROSS_SESSION_MAX_RESULTS:
        return json.dumps({"error": "node_ids exceeds the 20 item hard cap"})

    if args.get("cross_session") is True:
        if len(prompt) > 20_000:
            return json.dumps({"error": "cross-session prompt exceeds 20000 characters"})
        if len(query) > 2_000:
            return json.dumps({"error": "cross-session query exceeds 2000 characters"})
        raw_node_ids = raw_node_ids[:_CROSS_SESSION_MAX_RESULTS]
        allowed_session_ids = engine._authorized_cross_session_ids(
            kwargs.get("cross_session_capability")
        )
        if not allowed_session_ids:
            return json.dumps({
                "error": "cross-session expansion requires a trusted host capability",
            })
        return _cross_session_expand_query(
            engine,
            args,
            prompt=prompt,
            query=query,
            raw_node_ids=raw_node_ids,
            max_tokens=max_tokens,
            context_max_tokens=context_max_tokens,
            max_results=max_results,
            allowed_session_ids=allowed_session_ids,
        )

    safe_prompt, prompt_truncated = _bounded_cross_session_text(
        prompt,
        engine._config,
        max_tokens=2_000,
        max_chars=20_000,
    )
    safe_query, query_truncated = _bounded_cross_session_text(
        query,
        engine._config,
        max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
        max_chars=2_000,
    )

    nodes = []
    raw_results: list[dict[str, Any]] = []
    if raw_node_ids:
        for node_id in raw_node_ids:
            try:
                parsed_node_id = int(node_id)
            except (TypeError, ValueError):
                return json.dumps({"error": "node_ids must contain only integers"})
            node = _get_session_node(engine, parsed_node_id)
            if node is not None:
                nodes.append(node)
    elif query:
        nodes = engine._dag.search(query, session_id=engine.current_session_id, limit=max_results)
        raw_results = engine._store.search(query, session_id=engine.current_session_id, limit=max_results)
    else:
        return json.dumps({"error": "Provide either query or node_ids"})

    if not nodes and not raw_results:
        return json.dumps(
            {
                "prompt": safe_prompt,
                "prompt_truncated": prompt_truncated,
                "query": safe_query,
                "query_truncated": query_truncated,
                "answer": "No matching summaries or raw messages found in the current session.",
                "node_ids": [],
                "matches": [],
                "raw_matches": [],
            }
        )

    context_blocks = []
    context_budget_used = 0
    for node in nodes[:max_results]:
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)
        node_blocks = _collect_context_blocks_for_node(
            engine,
            node,
            max_tokens=remaining_context_tokens,
            hydrate_externalized_content=True,
        )
        context_blocks.extend(node_blocks)
        context_budget_used += _context_content_token_count(node_blocks)

    raw_matches: list[dict[str, Any]] = []
    if raw_results:
        seen_store_ids = _collect_store_ids_from_context_blocks(context_blocks)
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)
        raw_block, raw_matches = _collect_raw_match_context_block(
            engine,
            raw_results,
            max_tokens=remaining_context_tokens,
            query=query,
            exclude_store_ids=seen_store_ids,
        )
        if raw_block is not None:
            context_blocks.append(raw_block)
            context_budget_used += _context_content_token_count([raw_block])

    context_blocks, output_context_truncated = _bound_current_session_value(
        context_blocks,
        config=engine._config,
    )
    context_budget_used = _context_content_token_count(context_blocks)

    context_pagination = []
    for block in context_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "summary" and block.get("summary_truncated"):
            context_pagination.append(
                {
                    "node_id": block.get("node_id"),
                    "type": "summary",
                    "summary_truncated": True,
                    "expand_args": {"node_id": block.get("node_id")},
                }
            )
            continue

        if block_type in {"child_nodes", "descendant_child_nodes"}:
            for child in block.get("children", []):
                if child.get("summary_truncated"):
                    child_node_id = child.get("node_id")
                    context_pagination.append(
                        {
                            "node_id": block.get("node_id"),
                            "type": "child_summary" if block_type == "child_nodes" else "descendant_child_summary",
                            "child_node_id": child_node_id,
                            "source_index": child.get("source_index"),
                            "summary_truncated": True,
                            "expand_args": {"node_id": child_node_id},
                        }
                    )

        pagination = block.get("pagination")
        if not pagination or not pagination.get("has_more"):
            continue

        item = {
            "node_id": block.get("node_id"),
            "type": block_type,
            "pagination": pagination,
        }
        if block_type in {"messages", "child_messages"}:
            truncated_message = next(
                (message for message in block.get("messages", []) if message.get("content_truncated")),
                None,
            )
            if truncated_message:
                item["source_index"] = truncated_message.get("source_index")
                item["content_source"] = truncated_message.get("content_source")
                externalized = truncated_message.get("externalized") or {}
                externalized_ref = externalized.get("ref")
                if externalized_ref:
                    item["externalized_ref"] = externalized_ref
                    item["tool_call_id"] = externalized.get("tool_call_id")
                if truncated_message.get("content_source") == "externalized_payload" and externalized_ref:
                    item["expand_args"] = {
                        "externalized_ref": externalized_ref,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
                else:
                    item["expand_args"] = {
                        "node_id": block.get("node_id"),
                        "source_offset": pagination.get("next_source_offset") or 0,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
            else:
                item["expand_args"] = {
                    "node_id": block.get("node_id"),
                    "source_offset": pagination.get("next_source_offset") or 0,
                    "content_offset": pagination.get("next_content_offset") or 0,
                }
        elif block_type == "raw_messages":
            truncated_message = next(
                (message for message in block.get("messages", []) if message.get("content_truncated")),
                None,
            )
            if truncated_message:
                item["store_id"] = truncated_message.get("store_id")
                item["content_source"] = truncated_message.get("content_source")
                item["expand_args"] = {
                    "store_id": truncated_message.get("store_id"),
                    "content_offset": truncated_message.get("next_content_offset") or 0,
                }
            elif pagination.get("next_store_id"):
                item["store_id"] = pagination.get("next_store_id")
                item["expand_args"] = {"store_id": pagination.get("next_store_id")}
        elif block_type in {"child_nodes", "descendant_child_nodes"}:
            item["expand_args"] = {
                "node_id": block.get("node_id"),
                "source_offset": pagination.get("next_source_offset") or 0,
            }
        context_pagination.append(item)

    context_truncated = output_context_truncated or any(
        bool(item.get("summary_truncated")) or bool(item.get("pagination", {}).get("has_more"))
        for item in context_pagination
    )

    selected_nodes = nodes[:max_results]
    matches = []
    for node in selected_nodes:
        safe_summary, summary_truncated = _bounded_cross_session_text(
            node.summary,
            engine._config,
            max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
            max_chars=300,
        )
        safe_hint, hint_truncated = _bounded_cross_session_text(
            node.expand_hint,
            engine._config,
            max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
            max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
        )
        matches.append({
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": safe_summary,
            "summary_truncated": summary_truncated,
            "expand_hint": safe_hint,
            "expand_hint_truncated": hint_truncated,
        })
    raw_matches, raw_matches_truncated = _bound_current_session_value(
        raw_matches,
        config=engine._config,
        content_max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
    )
    context_truncated = context_truncated or raw_matches_truncated
    node_ids = [node.node_id for node in selected_nodes]

    model = engine._config.expansion_model or engine._config.summary_model or ""
    safe_model, model_truncated = _bounded_cross_session_text(
        model,
        engine._config,
        max_tokens=_CROSS_SESSION_METADATA_MAX_TOKENS,
        max_chars=_CROSS_SESSION_METADATA_MAX_CHARS,
    )

    def _degraded_payload(reason: str, *, include_timeout: bool = False) -> str:
        payload: Dict[str, Any] = {
            "prompt": safe_prompt,
            "prompt_truncated": prompt_truncated,
            "query": safe_query,
            "query_truncated": query_truncated,
            "error": reason,
            "degraded": True,
            "model": safe_model,
            "model_truncated": model_truncated,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
            "raw_matches": raw_matches,
        }
        if include_timeout:
            payload["timeout_seconds"] = timeout
        return json.dumps(payload)

    timeout = engine._config.expansion_timeout_ms / 1000
    try:
        answer = _synthesize_expansion_answer(
            prompt=prompt,
            context_blocks=context_blocks,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning("LCM expand_query synthesis timed out after %.3fs", timeout)
        return _degraded_payload(
            f"lcm_expand_query synthesis timed out after {timeout:.3g}s",
            include_timeout=True,
        )
    except Exception:
        logger.warning(
            "LCM expand_query synthesis failed after bounded evidence collection"
        )
        raise

    answer = str(answer).strip() if answer is not None else ""
    if not answer:
        logger.warning("LCM expand_query synthesis returned an empty answer")
        return _degraded_payload("lcm_expand_query synthesis returned an empty answer")

    bounded_answer, answer_truncated = _bounded_cross_session_text(
        answer,
        engine._config,
        max_tokens=max_tokens,
        max_chars=None,
    )
    return json.dumps(
        {
            "prompt": safe_prompt,
            "prompt_truncated": prompt_truncated,
            "query": safe_query,
            "query_truncated": query_truncated,
            "answer": bounded_answer,
            "answer_truncated": answer_truncated,
            "model": safe_model,
            "model_truncated": model_truncated,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
            "raw_matches": raw_matches,
        }
    )


def lcm_focus(args: Dict[str, Any], **kwargs) -> str:
    """Create, inspect, refresh, or deactivate a persisted focus overlay."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    action = str(args.get("action") or "show").strip().lower()
    if action == "show":
        return json.dumps(engine.get_focus_status(preview_chars=500))
    if action == "focus":
        return json.dumps(engine.create_focus(str(args.get("prompt") or "")))
    if action == "refocus":
        return json.dumps(engine.create_focus(str(args.get("prompt") or ""), refocus=True))
    if action == "unfocus":
        return json.dumps(engine.unfocus())
    return json.dumps({"error": "action must be show, focus, refocus, or unfocus"})


def _leaf_health_stats(engine: "LCMEngine") -> dict[str, Any]:
    """Return database-wide read-only depth-0 leaf size diagnostics."""
    conn = engine._dag.connection
    if conn is None:
        raise RuntimeError("LCM DAG connection is not initialized")

    configured_leaf_chunk_tokens = max(1, int(engine._config.leaf_chunk_tokens))
    dynamic_leaf_chunk_enabled = bool(engine._config.dynamic_leaf_chunk_enabled)
    configured_dynamic_leaf_chunk_max = max(
        1, int(engine._config.dynamic_leaf_chunk_max)
    )
    effective_max = (
        max(configured_leaf_chunk_tokens, configured_dynamic_leaf_chunk_max)
        if dynamic_leaf_chunk_enabled
        else configured_leaf_chunk_tokens
    )
    totals = conn.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(source_token_count), 0),
            COALESCE(AVG(source_token_count), 0),
            COALESCE(MAX(source_token_count), 0),
            COALESCE(SUM(token_count), 0),
            SUM(CASE WHEN source_token_count <= 8000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > 8000 AND source_token_count <= 20000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > 20000 AND source_token_count <= 40000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > 40000 AND source_token_count <= 80000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > 80000 AND source_token_count <= 160000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > 160000 THEN 1 ELSE 0 END),
            SUM(CASE WHEN source_token_count > ? THEN 1 ELSE 0 END),
            COUNT(DISTINCT CASE WHEN source_token_count > ? THEN session_id END)
        FROM summary_nodes
        WHERE depth = 0
        """,
        (effective_max, effective_max),
    ).fetchone()
    worst_rows = conn.execute(
        """
        SELECT node_id, session_id, source_token_count, token_count
        FROM summary_nodes
        WHERE depth = 0 AND source_token_count > ?
        ORDER BY source_token_count DESC, node_id ASC
        LIMIT 10
        """,
        (effective_max,),
    ).fetchall()
    high_raw_threshold = max(100_000, effective_max * 2)
    high_raw_rows = conn.execute(
        """
        WITH raw_sessions AS (
            SELECT
                session_id,
                COALESCE(SUM(token_estimate), 0) AS raw_message_tokens,
                COUNT(*) AS raw_message_count
            FROM messages
            GROUP BY session_id
        ),
        depth0_sessions AS (
            SELECT session_id, COUNT(*) AS depth0_node_count
            FROM summary_nodes
            WHERE depth = 0
            GROUP BY session_id
        )
        SELECT
            raw.session_id,
            raw.raw_message_tokens,
            raw.raw_message_count,
            COALESCE(leaves.depth0_node_count, 0),
            COUNT(*) OVER () AS qualifying_session_count
        FROM raw_sessions AS raw
        LEFT JOIN depth0_sessions AS leaves ON leaves.session_id = raw.session_id
        WHERE raw.raw_message_tokens >= ?
          AND COALESCE(leaves.depth0_node_count, 0) <= 1
        ORDER BY raw.raw_message_tokens DESC, raw.session_id ASC
        LIMIT 20
        """,
        (high_raw_threshold,),
    ).fetchall()

    total_nodes = int(totals[0] or 0)
    total_source_tokens = int(totals[1] or 0)
    total_summary_tokens = int(totals[4] or 0)
    return {
        "scope": "database",
        "total_depth0_nodes": total_nodes,
        "total_source_tokens": total_source_tokens,
        "average_source_tokens": round(float(totals[2] or 0), 1),
        "max_source_tokens": int(totals[3] or 0),
        "total_summary_tokens": total_summary_tokens,
        "overall_compression_ratio": (
            round(total_source_tokens / total_summary_tokens, 1)
            if total_summary_tokens > 0
            else 0.0
        ),
        "source_token_buckets": {
            "up_to_8k": int(totals[5] or 0),
            "8k_to_20k": int(totals[6] or 0),
            "20k_to_40k": int(totals[7] or 0),
            "40k_to_80k": int(totals[8] or 0),
            "80k_to_160k": int(totals[9] or 0),
            "over_160k": int(totals[10] or 0),
        },
        "configured_leaf_chunk_tokens": configured_leaf_chunk_tokens,
        "dynamic_leaf_chunk_enabled": dynamic_leaf_chunk_enabled,
        "configured_dynamic_leaf_chunk_max": configured_dynamic_leaf_chunk_max,
        "effective_max_source_tokens": effective_max,
        "oversized_depth0_nodes": int(totals[11] or 0),
        "oversized_sessions": int(totals[12] or 0),
        "high_raw_token_threshold": high_raw_threshold,
        "high_raw_low_node_session_count": (
            int(high_raw_rows[0][4]) if high_raw_rows else 0
        ),
        "high_raw_low_node_sessions": [
            {
                "session_id": session_id,
                "raw_message_tokens": int(raw_message_tokens or 0),
                "raw_message_count": int(raw_message_count or 0),
                "depth0_node_count": int(depth0_node_count or 0),
            }
            for (
                session_id,
                raw_message_tokens,
                raw_message_count,
                depth0_node_count,
                _qualifying_session_count,
            ) in high_raw_rows
        ],
        "worst_oversized_nodes": [
            {
                "node_id": int(node_id),
                "session_id": session_id,
                "source_token_count": int(source_token_count or 0),
                "token_count": int(token_count or 0),
                "compression_ratio": (
                    round(float(source_token_count) / float(token_count), 1)
                    if token_count and token_count > 0
                    else None
                ),
            }
            for node_id, session_id, source_token_count, token_count in worst_rows
        ],
    }


def _summary_quality_stats(engine: "LCMEngine", session_id: str) -> dict[str, Any]:
    """Return read-only summary compression quality diagnostics for one session."""
    conn = engine._dag.connection
    if conn is None:
        raise RuntimeError("LCM DAG connection is not initialized")
    rows = conn.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count
        FROM summary_nodes
        WHERE session_id = ? AND source_token_count > 0
        ORDER BY
            CASE WHEN token_count <= 0 THEN 1 ELSE 0 END DESC,
            CASE WHEN token_count > 0
                 THEN CAST(source_token_count AS REAL) / token_count
                 ELSE source_token_count
            END DESC
        LIMIT 5
        """,
        (session_id,),
    ).fetchall()
    totals = conn.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(source_token_count), 0),
            COALESCE(SUM(token_count), 0),
            SUM(CASE WHEN source_token_count >= 100000
                      AND token_count < 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN token_count > 0
                      AND CAST(source_token_count AS REAL) / token_count >= 400
                     THEN 1 ELSE 0 END)
        FROM summary_nodes
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    total_nodes = int(totals[0] or 0)
    total_source_tokens = int(totals[1] or 0)
    total_summary_tokens = int(totals[2] or 0)
    tiny_large_source_nodes = int(totals[3] or 0)
    extreme_ratio_nodes = int(totals[4] or 0)
    overall_ratio = (
        round(total_source_tokens / total_summary_tokens, 1)
        if total_summary_tokens > 0
        else 0.0
    )
    worst_nodes = []
    for node_id, session_id, depth, token_count, source_token_count in rows:
        ratio = (
            round(float(source_token_count) / float(token_count), 1)
            if token_count and token_count > 0
            else None
        )
        worst_nodes.append({
            "node_id": int(node_id),
            "session_id": session_id,
            "depth": int(depth),
            "source_token_count": int(source_token_count or 0),
            "token_count": int(token_count or 0),
            "compression_ratio": ratio,
        })
    return {
        "total_nodes": total_nodes,
        "session_id": session_id,
        "total_source_tokens": total_source_tokens,
        "total_summary_tokens": total_summary_tokens,
        "overall_compression_ratio": overall_ratio,
        "extreme_ratio_threshold": 400,
        "tiny_large_source_threshold": {
            "source_token_count_min": 100000,
            "token_count_max": 500,
        },
        "extreme_ratio_nodes": extreme_ratio_nodes,
        "tiny_large_source_nodes": tiny_large_source_nodes,
        "worst_nodes": worst_nodes,
        "recommendation": (
            "Inspect worst_nodes with lcm_expand; tiny summaries for very large sources often indicate degraded fallback summarization."
            if extreme_ratio_nodes or tiny_large_source_nodes
            else "summary compression ratios are within the diagnostic thresholds"
        ),
    }


def _matched_session_patterns(session_keys: list[str], patterns: list[str]) -> list[str]:
    """Return configured session glob patterns that match the supplied keys."""
    matched: list[str] = []
    for pattern in patterns:
        try:
            compiled = compile_session_pattern(pattern)
        except re.error:
            continue
        if any(compiled.match(key) for key in session_keys if key):
            matched.append(pattern)
    return matched


def _inspect_externalized_refs_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sources = [value]
    else:
        try:
            sources = [json.dumps(value, ensure_ascii=False)]
        except (TypeError, ValueError):
            sources = [str(value)]

    refs: list[str] = []
    for source in sources:
        for ref in extract_ingest_externalized_refs(source) + extract_externalized_refs(source):
            if ref not in refs:
                refs.append(ref)
    return refs


def _inspect_message_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return row metadata only; never include raw message content."""
    content = row.get("content") or ""
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": row.get("session_id") or "",
        "source": row.get("source") or "",
        "conversation_id": row.get("conversation_id") or "",
        "role": row.get("role") or "unknown",
        "timestamp": row.get("timestamp", 0),
        "token_estimate": row.get("token_estimate", 0),
        "content_chars": len(content),
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")

    refs: list[str] = []
    for value in (row.get("content"), row.get("tool_calls")):
        for ref in _inspect_externalized_refs_from_value(value):
            if ref not in refs:
                refs.append(ref)
    if refs:
        item["externalized_refs"] = refs
    return item


def _inspect_lifecycle_state(engine: "LCMEngine", session_id: str, conversation_id: str) -> dict[str, Any] | None:
    state = None
    if conversation_id:
        state = engine._lifecycle.get_by_conversation(conversation_id)
    if state is None and session_id:
        state = engine._lifecycle.get_by_session(session_id)
    if state is None:
        return None
    return {
        "conversation_id": state.conversation_id,
        "current_session_id": state.current_session_id,
        "last_finalized_session_id": state.last_finalized_session_id,
        "current_frontier_store_id": state.current_frontier_store_id,
        "last_finalized_frontier_store_id": state.last_finalized_frontier_store_id,
        "debt_kind": state.debt_kind,
        "debt_size_estimate": state.debt_size_estimate,
        "current_bound_at": state.current_bound_at,
        "last_finalized_at": state.last_finalized_at,
        "debt_updated_at": state.debt_updated_at,
        "last_maintenance_attempt_at": state.last_maintenance_attempt_at,
        "last_rollover_at": state.last_rollover_at,
        "last_reset_at": state.last_reset_at,
        "updated_at": state.updated_at,
    }


def _inspect_highest_compacted_source_store_id(engine: "LCMEngine", session_id: str) -> int:
    highest = 0
    edges = 0
    encoded_bytes = 0
    deadline = time.monotonic() + _LCM_INSPECT_LINEAGE_DEADLINE_SECONDS
    rows_seen = 0
    last_node_id = 0
    while rows_seen < _LCM_INSPECT_LINEAGE_MAX_ROWS:
        if time.monotonic() >= deadline:
            break
        page_limit = min(400, _LCM_INSPECT_LINEAGE_MAX_ROWS - rows_seen)
        rows = engine._dag.connection.execute(
            """SELECT node_id, depth,
                      COALESCE(length(CAST(source_ids AS BLOB)), 0),
                      CASE
                        WHEN COALESCE(length(CAST(source_ids AS BLOB)), 0) <= ?
                        THEN CAST(source_ids AS TEXT) ELSE NULL
                      END
               FROM summary_nodes
               WHERE session_id = ? AND source_type = 'messages' AND node_id > ?
               ORDER BY node_id LIMIT ?""",
            (
                min(MAX_SOURCE_IDS_JSON_CHARS, _LCM_INSPECT_LINEAGE_MAX_BYTES),
                session_id,
                last_node_id,
                page_limit,
            ),
        ).fetchall()
        if not rows:
            break
        for node_id, depth, raw_bytes, raw_source_ids in rows:
            if time.monotonic() >= deadline:
                return highest
            rows_seen += 1
            last_node_id = int(node_id)
            if int(depth or 0) > _LCM_INSPECT_LINEAGE_MAX_DEPTH:
                return highest
            raw_bytes = int(raw_bytes or 0)
            encoded_bytes += raw_bytes
            if (
                raw_source_ids is None
                or raw_bytes > MAX_SOURCE_IDS_JSON_CHARS
                or encoded_bytes > _LCM_INSPECT_LINEAGE_MAX_BYTES
            ):
                return highest
            try:
                source_ids = decode_source_ids(raw_source_ids or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if edges + len(source_ids) > _LCM_INSPECT_LINEAGE_MAX_EDGES:
                return highest
            edges += len(source_ids)
            for source_id in source_ids:
                try:
                    highest = max(highest, int(source_id))
                except (TypeError, ValueError, OverflowError):
                    continue
        if len(rows) < page_limit:
            break
    return highest


def _inspect_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, str], bool]:
    decoder = json.JSONDecoder()
    fields: dict[str, str] = {}
    length = len(text)
    index = 0

    def skip_json_whitespace(pos: int) -> int:
        while pos < length and text[pos] in " \t\n\r":
            pos += 1
        return pos

    index = skip_json_whitespace(index)
    if index >= length or text[index] != "{":
        return fields, False
    index += 1

    while True:
        index = skip_json_whitespace(index)
        if index >= length or text[index] == "}":
            return fields, False
        if text[index] != '"':
            return fields, False
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, False
        if not isinstance(key, str):
            return fields, False
        index = skip_json_whitespace(index)
        if index >= length or text[index] != ":":
            return fields, False
        index += 1
        index = skip_json_whitespace(index)
        if key == "content":
            return fields, index < length and text[index] == '"'
        if index >= length:
            return fields, False
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, False
        if isinstance(value, str):
            fields[key] = value
        elif key == "session_id":
            fields.pop(key, None)
        index = skip_json_whitespace(index)
        if index >= length:
            return fields, False
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return fields, False
        return fields, False


def _read_externalized_payload_metadata_prefix_from_handle(
    handle,
    *,
    max_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
    deadline: float | None = None,
) -> tuple[str, bool, bool]:
    """Lex metadata once, stopping before the first content body byte.

    A one-byte transport read is intentional here: authorization depends on a
    field inside the JSON envelope, and reading a larger block could pull an
    unauthorized ``content`` body into Python before that decision. Lexical
    work is nevertheless O(n): each decoded character advances one compact
    state machine exactly once, with no repeated JSON-prefix reparsing.
    """
    read_limit = min(
        _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
        max(0, int(max_bytes)),
    )
    if read_limit <= 0:
        return "", False, True
    if deadline is not None and _external_metadata_now() >= deadline:
        raise TimeoutError("metadata_deadline")
    prefix = bytearray()
    text_parts: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    phase = "start"
    key_raw: list[str] = []
    current_key = ""
    escaped = False
    container_stack: list[str] = []
    complex_in_string = False
    complex_escaped = False
    content_key_seen = False

    def consume(char: str) -> bool:
        nonlocal phase, current_key, escaped
        nonlocal complex_in_string, complex_escaped, content_key_seen
        if phase == "start":
            if char in " \t\n\r":
                return False
            phase = "key_or_end" if char == "{" else "invalid"
            return False
        if phase == "key_or_end":
            if char in " \t\n\r":
                return False
            if char == '"':
                key_raw.clear()
                escaped = False
                phase = "key_string"
            elif char == "}":
                phase = "done"
            else:
                phase = "invalid"
            return False
        if phase == "key_string":
            if escaped:
                key_raw.append(char)
                escaped = False
            elif char == "\\":
                key_raw.append(char)
                escaped = True
            elif char == '"':
                try:
                    current_key = json.loads('"' + "".join(key_raw) + '"')
                except (TypeError, ValueError, json.JSONDecodeError):
                    phase = "invalid"
                    return False
                phase = "colon"
            else:
                key_raw.append(char)
            return False
        if phase == "colon":
            if char in " \t\n\r":
                return False
            phase = "value_start" if char == ":" else "invalid"
            return False
        if phase == "value_start":
            if char in " \t\n\r":
                return False
            if current_key == "content":
                content_key_seen = char == '"'
                phase = "content" if content_key_seen else "invalid"
                return content_key_seen
            if char == '"':
                escaped = False
                phase = "value_string"
            elif char in "[{":
                container_stack[:] = [char]
                if len(container_stack) + 1 > _EXTERNALIZED_SUFFIX_MAX_DEPTH:
                    raise ValueError("invalid_payload")
                complex_in_string = False
                complex_escaped = False
                phase = "value_complex"
            elif char in "-0123456789tfn":
                phase = "value_scalar"
            else:
                phase = "invalid"
            return False
        if phase == "value_string":
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                phase = "after_value"
            return False
        if phase == "value_scalar":
            if char == ",":
                phase = "key_or_end"
            elif char == "}":
                phase = "done"
            elif char in " \t\n\r":
                phase = "after_value"
            return False
        if phase == "value_complex":
            if complex_in_string:
                if complex_escaped:
                    complex_escaped = False
                elif char == "\\":
                    complex_escaped = True
                elif char == '"':
                    complex_in_string = False
                return False
            if char == '"':
                complex_in_string = True
            elif char in "[{":
                container_stack.append(char)
                if len(container_stack) + 1 > _EXTERNALIZED_SUFFIX_MAX_DEPTH:
                    raise ValueError("invalid_payload")
            elif char in "]}":
                if not container_stack:
                    phase = "invalid"
                else:
                    opener = container_stack.pop()
                    if (opener, char) not in {("[", "]"), ("{", "}")}:
                        phase = "invalid"
                    elif not container_stack:
                        phase = "after_value"
            return False
        if phase == "after_value":
            if char in " \t\n\r":
                return False
            if char == ",":
                phase = "key_or_end"
            elif char == "}":
                phase = "done"
            else:
                phase = "invalid"
            return False
        return False

    next_deadline_check = 0
    while len(prefix) < read_limit:
        if len(prefix) >= next_deadline_check:
            if deadline is not None and _external_metadata_now() >= deadline:
                raise TimeoutError("metadata_deadline")
            next_deadline_check = (
                len(prefix) + _LCM_EXTERNAL_METADATA_DEADLINE_CHECK_BYTES
            )
        byte = handle.read(1)
        if not byte:
            break
        prefix.extend(byte)
        try:
            decoded = decoder.decode(byte, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            text_parts.append(decoded)
            for char in decoded:
                if consume(char):
                    return "".join(text_parts), True, False
    # The bound itself is not exceeded merely to distinguish exact EOF. A
    # prefix that fills it without reaching content is conservatively treated
    # as truncated.
    prefix_truncated = len(prefix) >= read_limit
    if not prefix_truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            text_parts.append(final_text)
    return "".join(text_parts), content_key_seen, prefix_truncated


def _read_externalized_payload_metadata_prefix(
    path: Path,
    *,
    max_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
) -> tuple[str, bool, bool]:
    """Read bounded JSON metadata before the externalized payload body.

    Returns ``(prefix_text, content_string_seen, prefix_truncated)``. The content
    string body is intentionally not consumed; ``lcm_inspect`` reports bounded
    metadata only and leaves full JSON/body validation to explicit expansion.
    """
    with path.open("rb") as handle:
        return _read_externalized_payload_metadata_prefix_from_handle(
            handle,
            max_bytes=max_bytes,
        )


def _inspect_externalized_payload_metadata(engine: "LCMEngine", ref: str, session_id: str) -> dict[str, Any]:
    if not ref or Path(ref).name != ref:
        return {"readable": False, "error": "invalid_ref"}
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=False,
        )
        path = storage_dir / ref
        if not path.exists():
            return {"readable": False, "error": "missing"}
        if not path.is_file():
            return {"readable": False, "error": "not_a_file"}
        metadata_prefix_text, content_key_seen, prefix_truncated = _read_externalized_payload_metadata_prefix(path)
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except (OSError, ValueError) as exc:
        return {"readable": False, "error": str(exc)}

    metadata_fields, _content_key_seen = _inspect_top_level_json_string_fields_before_content(metadata_prefix_text)
    payload_session_id = metadata_fields.get("session_id", "")
    if payload_session_id and payload_session_id != session_id:
        return {"readable": False, "error": "session_mismatch"}
    if not payload_session_id:
        return {"readable": False, "error": "session_metadata_unavailable"}
    if not content_key_seen:
        error = "metadata_prefix_truncated" if prefix_truncated else "invalid_payload"
        return {"readable": False, "error": error}

    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except OSError as exc:
        return {"readable": False, "error": str(exc)}

    metadata: dict[str, Any] = {
        "readable": True,
        "file_size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "payload_validation": "metadata_prefix",
    }
    if payload_session_id:
        metadata["payload_session_id"] = payload_session_id
    return metadata


def _inspect_externalized_refs(engine: "LCMEngine", session_id: str, limit: int) -> dict[str, Any]:
    message_total = engine._store.get_session_count(session_id)
    rows = engine._store.load_session_page(session_id, limit=_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT)
    scan_truncated = message_total > len(rows)
    items: list[dict[str, Any]] = []
    total_known = 0
    seen: set[tuple[int, str]] = set()
    for row in rows:
        refs: list[str] = []
        for value in (row.get("content"), row.get("tool_calls")):
            for ref in _inspect_externalized_refs_from_value(value):
                if ref not in refs:
                    refs.append(ref)
        for ref in refs:
            key = (int(row.get("store_id") or 0), ref)
            if key in seen:
                continue
            seen.add(key)
            total_known += 1
            if len(items) >= limit:
                continue
            metadata = _inspect_externalized_payload_metadata(engine, ref, session_id)
            item: dict[str, Any] = {
                "externalized_ref": ref,
                "store_id": row.get("store_id"),
                "session_id": row.get("session_id") or "",
                "source": row.get("source") or "",
                "conversation_id": row.get("conversation_id") or "",
                "role": row.get("role") or "unknown",
                "timestamp": row.get("timestamp", 0),
                "readable": metadata.get("readable") is True,
            }
            if row.get("tool_call_id"):
                item["tool_call_id"] = row.get("tool_call_id")
            item.update(metadata)
            items.append(item)

    return {
        "total_known": total_known,
        "total_known_exact": not scan_truncated,
        "scanned_messages": len(rows),
        "scan_truncated": scan_truncated,
        "returned": len(items),
        "has_more": total_known > len(items) or scan_truncated,
        "items": items,
    }


def lcm_inspect(args: Dict[str, Any], **kwargs) -> str:
    """Return a read-only metadata inventory of the current LCM session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    raw_limit_arg = args.get("limit", _LCM_INSPECT_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_INSPECT_HARD_LIMIT_CAP)

    session_id = engine.current_session_id
    conversation_id = engine.current_conversation_id
    if not session_id:
        full_status = engine.get_status()
        return json.dumps({
            "error": "No active session",
            "read_only": True,
            "runtime_identity": full_status.get("runtime_identity") or engine.get_runtime_identity(),
            "ingest_protection": full_status.get("ingest_protection"),
        })

    full_status = engine.get_status()
    runtime_identity = full_status.get("runtime_identity") or engine.get_runtime_identity()
    lifecycle = _inspect_lifecycle_state(engine, session_id, conversation_id)

    store_totals_row = engine._store.connection.execute(
        """
        SELECT COUNT(*), MIN(store_id), MAX(store_id), COALESCE(SUM(token_estimate), 0)
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    message_total = int(store_totals_row[0] or 0) if store_totals_row else 0
    min_store_id = store_totals_row[1] if store_totals_row else None
    max_store_id = store_totals_row[2] if store_totals_row else None
    estimated_tokens = int(store_totals_row[3] or 0) if store_totals_row else 0
    stored_messages = engine._store.get_session_messages(session_id)
    fresh_tail_start = engine._fresh_tail_start(stored_messages)
    selected_tail = stored_messages[fresh_tail_start:]
    fresh_tail_count = len(selected_tail)
    fresh_tail_limit = min(limit, fresh_tail_count) if fresh_tail_count > 0 else 0
    fresh_tail_items = [
        _inspect_message_metadata(row)
        for row in (
            selected_tail[-fresh_tail_limit:] if fresh_tail_limit > 0 else []
        )
    ]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    total_dag_nodes = sum(info["count"] for info in depth_stats.values())
    total_dag_tokens = sum(info["tokens"] for info in depth_stats.values())
    total_dag_source_tokens = sum(info["source_tokens"] for info in depth_stats.values())
    latest_node_rows = engine._dag.connection.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count,
               source_type, created_at, earliest_at, latest_at, expand_hint
        FROM summary_nodes
        WHERE session_id = ?
        ORDER BY created_at DESC, node_id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    latest_nodes = [
        {
            "node_id": int(row[0]),
            "session_id": row[1],
            "depth": int(row[2]),
            "token_count": int(row[3] or 0),
            "source_token_count": int(row[4] or 0),
            "source_type": row[5],
            "created_at": row[6],
            "earliest_at": row[7],
            "latest_at": row[8],
            "expand_hint_available": bool(row[9]),
            "expand_hint_chars": len(row[9] or ""),
        }
        for row in latest_node_rows
    ]

    highest_compacted_source_store_id = _inspect_highest_compacted_source_store_id(engine, session_id)
    lifecycle_current_frontier = int((lifecycle or {}).get("current_frontier_store_id") or 0)
    lifecycle_finalized_frontier = int((lifecycle or {}).get("last_finalized_frontier_store_id") or 0)
    runtime_last_compacted = int(getattr(engine, "_last_compacted_store_id", 0) or 0)

    platform = engine.current_session_platform
    session_keys = build_session_match_keys(session_id, platform=platform)
    ignore_patterns = list(engine._config.ignore_session_patterns or [])
    stateless_patterns = list(engine._config.stateless_session_patterns or [])

    response: dict[str, Any] = {
        "read_only": True,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "limit": limit,
        "runtime_identity": runtime_identity,
        "lineage": {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "session_platform": platform,
            "side_channel_active": engine.side_channel_active,
            "bound_session_id": getattr(engine, "_session_id", ""),
            "bound_conversation_id": getattr(engine, "_conversation_id", ""),
            "lifecycle": lifecycle,
            "source_lineage": full_status.get("source_lineage"),
        },
        "messages": {
            "total": message_total,
            "estimated_tokens": estimated_tokens,
            "min_store_id": min_store_id,
            "max_store_id": max_store_id,
            "fresh_tail_count": fresh_tail_count,
            "fresh_tail_count_limit": engine._effective_fresh_tail_count(),
            "fresh_tail_max_tokens": engine._effective_fresh_tail_max_tokens(),
            "fresh_tail_selection": full_status.get("fresh_tail_selection"),
            "pre_tail_message_count": fresh_tail_start,
            "fresh_tail": {
                "returned": len(fresh_tail_items),
                "items": fresh_tail_items,
            },
        },
        "compaction": {
            "last": {
                "status": full_status.get("last_compression_status", "idle"),
                "noop_reason": full_status.get("last_compression_noop_reason", ""),
                "condensation_suppressed_reason": full_status.get("condensation_suppressed_reason", ""),
                "compression_count": engine.compression_count,
                "last_prompt_tokens": engine.last_prompt_tokens,
                "threshold_tokens": engine.threshold_tokens,
            },
            "frontier": {
                "runtime_last_compacted_store_id": runtime_last_compacted,
                "highest_compacted_source_store_id": highest_compacted_source_store_id,
                "lifecycle_current_frontier_store_id": lifecycle_current_frontier,
                "lifecycle_last_finalized_frontier_store_id": lifecycle_finalized_frontier,
            },
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "total_source_tokens": total_dag_source_tokens,
            "depths": {f"d{depth}": info for depth, info in sorted(depth_stats.items())},
            "latest_nodes": latest_nodes,
        },
        "externalized_refs": _inspect_externalized_refs(engine, session_id, limit),
        "ingest_protection": full_status.get("ingest_protection"),
        "filters": {
            "session_keys": session_keys,
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": ignore_patterns,
            "stateless_session_patterns": stateless_patterns,
            "matched_ignore_session_patterns": _matched_session_patterns(session_keys, ignore_patterns),
            "matched_stateless_session_patterns": _matched_session_patterns(session_keys, stateless_patterns),
            "ignore_message_patterns": list(engine._config.ignore_message_patterns or []),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
        },
    }
    if requested_limit > _LCM_INSPECT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return json.dumps(response)


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    # Read the foreground view so a side-channel session that briefly owns
    # engine._session_id (cron tick inside the gateway process, debug probe,
    # etc.) does not divert lcm_status away from the operator's real
    # conversation. Falls back to the bound id when no foreground has ever
    # been bound, so cron-only or stateless-only deployments still report
    # something usable.
    session_id = engine.current_session_id
    if not session_id:
        return json.dumps({
            "error": "No active session",
            "runtime_identity": engine.get_runtime_identity(),
        })

    # Store stats
    store_messages = engine._store.get_session_count(session_id)
    store_tokens = engine._store.get_session_token_total(session_id)

    # DAG stats by depth
    depths = engine._dag.get_session_depth_stats(session_id)

    total_dag_tokens = sum(d["tokens"] for d in depths.values())
    total_source_tokens = sum(d["source_tokens"] for d in depths.values())
    total_dag_nodes = sum(d["count"] for d in depths.values())
    compression_ratio = round(total_source_tokens / total_dag_tokens, 1) if total_dag_tokens > 0 else 0
    full_status = engine.get_status()
    lifecycle = full_status.get("lifecycle")
    lifecycle_fragmentation = full_status.get("lifecycle_fragmentation")
    source_lineage = full_status.get("source_lineage")
    runtime_identity = full_status.get("runtime_identity")
    ingest_reconciliation = full_status.get("ingest_reconciliation")
    config_sources = full_status.get("config_sources") or {}
    config_source_warnings = full_status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = full_status.get("ignored_config_yaml_lcm_keys") or []
    try:
        leaf_health = _leaf_health_stats(engine)
    except Exception as exc:
        leaf_health = {"scope": "database", "status": "unavailable", "error": str(exc)}

    # Filter classification for the session lcm_status is reporting on.
    # The engine encapsulates the foreground vs bound divergence; this tool
    # just reads the property contract.
    side_channel_active = engine.side_channel_active

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "last_compression_status": full_status.get("last_compression_status", "idle"),
        "last_compression_noop_reason": full_status.get("last_compression_noop_reason", ""),
        "model": full_status.get("model", ""),
        "provider": full_status.get("provider", ""),
        "raw_context_length": full_status.get("raw_context_length", engine.context_length),
        "context_length": engine.context_length,
        "effective_context_length_cap": full_status.get("effective_context_length_cap"),
        "effective_context_length_reason": full_status.get("effective_context_length_reason", ""),
        "context_length_source": full_status.get("context_length_source", ""),
        "configured_context_threshold": full_status.get("configured_context_threshold", engine._config.context_threshold),
        "context_threshold": full_status.get("context_threshold", engine._config.context_threshold),
        "context_threshold_source": full_status.get("context_threshold_source", ""),
        "context_threshold_autoraised": full_status.get("context_threshold_autoraised"),
        "threshold_tokens": engine.threshold_tokens,
        "last_prompt_tokens": engine.last_prompt_tokens,
        "last_input_tokens": engine.last_input_tokens,
        "last_output_tokens": engine.last_output_tokens,
        "last_cache_read_tokens": engine.last_cache_read_tokens,
        "last_cache_write_tokens": engine.last_cache_write_tokens,
        "last_reasoning_tokens": engine.last_reasoning_tokens,
        "cache_metrics_available": engine.cache_metrics_available,
        "cache_read_ratio": round(engine.cache_read_ratio, 4),
        "store": {
            "messages": store_messages,
            "estimated_tokens": store_tokens,
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "compression_ratio": f"{compression_ratio}:1",
            "depths": {
                f"d{depth}": info for depth, info in sorted(depths.items())
            },
        },
        "leaf_health": leaf_health,
        "config": {
            "fresh_tail_count": engine._config.fresh_tail_count,
            "fresh_tail_max_tokens": engine._config.fresh_tail_max_tokens,
            "leaf_chunk_tokens": engine._config.leaf_chunk_tokens,
            "dynamic_leaf_chunk_enabled": engine._config.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": engine._config.dynamic_leaf_chunk_max,
            "cache_friendly_condensation_enabled": engine._config.cache_friendly_condensation_enabled,
            "cache_friendly_min_debt_groups": engine._config.cache_friendly_min_debt_groups,
            "deferred_maintenance_enabled": engine._config.deferred_maintenance_enabled,
            "deferred_maintenance_max_passes": engine._config.deferred_maintenance_max_passes,
            "critical_budget_pressure_ratio": engine._config.critical_budget_pressure_ratio,
            "context_threshold": engine._config.context_threshold,
            "max_depth": engine._config.incremental_max_depth,
            "condensation_fanin": engine._config.condensation_fanin,
            "summary_model": engine._config.summary_model or "(auxiliary)",
            "summary_timeout_ms": engine._config.summary_timeout_ms,
            "summary_spend_max_calls": engine._config.summary_spend_max_calls,
            "summary_spend_window_seconds": engine._config.summary_spend_window_seconds,
            "summary_spend_backoff_seconds": engine._config.summary_spend_backoff_seconds,
            "expansion_model": engine._config.expansion_model or "(summary model)",
        },
        "config_sources": config_sources,
        "config_source_warnings": config_source_warnings,
        "ignored_config_yaml_lcm_keys": ignored_config_yaml_lcm_keys,
        "session_filters": {
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": full_status.get("ignore_session_patterns", []),
            "ignore_session_patterns_source": full_status.get("ignore_session_patterns_source", "default"),
            "stateless_session_patterns": full_status.get("stateless_session_patterns", []),
            "stateless_session_patterns_source": full_status.get("stateless_session_patterns_source", "default"),
            "ignore_message_patterns": full_status.get("ignore_message_patterns", []),
            "ignore_message_patterns_source": full_status.get("ignore_message_patterns_source", "default"),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
            "side_channel_active": side_channel_active,
            **(
                {"side_channel_session_id": engine._session_id}
                if side_channel_active
                else {}
            ),
        },
        "source_lineage": source_lineage,
        "ingest_protection": full_status.get("ingest_protection", sensitive_pattern_status(engine._config)),
        "preset_suggestion": preset_status_payload(engine),
        "ingest_reconciliation": ingest_reconciliation,
        "runtime_identity": runtime_identity,
        "lifecycle": lifecycle,
        "lifecycle_fragmentation": lifecycle_fragmentation,
        "async_compaction": full_status.get("async_compaction") or engine.get_async_compaction_status(),
    })


def lcm_doctor(args: Dict[str, Any], **kwargs) -> str:
    """Run diagnostics on the LCM database and configuration."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    checks: list[dict] = []
    # Diagnose the foreground session, not whatever side-channel session
    # currently owns engine._session_id. Falls back to the bound id when no
    # foreground has ever been bound.
    session_id = engine.current_session_id

    # 1. Database integrity
    try:
        result = engine._store.connection.execute("PRAGMA integrity_check").fetchone()
        ok = result and result[0] == "ok"
        checks.append({
            "check": "database_integrity",
            "status": "pass" if ok else "fail",
            "detail": result[0] if result else "no response",
        })
    except Exception as e:
        checks.append({
            "check": "database_integrity",
            "status": "fail",
            "detail": str(e),
        })

    # Ingest health: a swallowed persistence error means turns were not
    # durably stored, silently breaking the lossless guarantee. Surface it.
    ingest_failures = int(getattr(engine, "_ingest_failure_count", 0) or 0)
    consecutive_failures = int(getattr(engine, "_consecutive_ingest_failures", 0) or 0)
    if consecutive_failures > 0:
        ingest_status = "fail"
    elif ingest_failures > 0:
        ingest_status = "warn"
    else:
        ingest_status = "pass"
    checks.append({
        "check": "ingest_health",
        "status": ingest_status,
        "detail": {
            "total_failures": ingest_failures,
            "consecutive_failures": consecutive_failures,
            "last_error": getattr(engine, "_last_ingest_error", "") or "",
            "last_error_time": getattr(engine, "_last_ingest_error_time", 0) or 0,
        } if ingest_failures else "no ingest failures recorded",
    })

    # ignore_message_patterns drops discard raw content that is never persisted.
    # A non-zero count is worth surfacing so an over-broad pattern is noticed.
    dropped = int(getattr(engine, "_ignore_pattern_dropped_count", 0) or 0)
    checks.append({
        "check": "ignore_pattern_drops",
        "status": "warn" if dropped else "pass",
        "detail": (
            f"{dropped} message(s) dropped by ignore_message_patterns and not "
            "persisted; verify the pattern is not matching substantive turns"
            if dropped
            else "no messages dropped by ignore_message_patterns"
        ),
    })

    try:
        conn = engine._store.connection
        if conn is None:
            raise RuntimeError("LCM store connection is not initialized")
        schema_health = inspect_lcm_schema_health(
            conn,
            database_path=str(engine._store.db_path),
        )
        missing_tables = schema_health.get("missing_tables")
        has_missing = isinstance(missing_tables, list) and bool(missing_tables)
        checks.append({
            "check": "schema_core_tables",
            "status": "fail" if has_missing or schema_health.get("error") else "pass",
            "detail": schema_health,
        })
    except Exception as e:
        checks.append({
            "check": "schema_core_tables",
            "status": "fail",
            "detail": str(e),
        })

    # 1b. FTS5 integrity, separated from generic SQLite integrity so malformed
    # inverted indexes point at the exact table and repair path.
    for check_name, conn, spec in (
        ("messages_fts_integrity", engine._store.connection, build_message_fts_spec()),
        ("nodes_fts_integrity", engine._dag.connection, build_nodes_fts_spec()),
    ):
        try:
            fts_integrity = check_external_content_fts_integrity(conn, spec)
            status = fts_integrity["status"]
            checks.append({
                "check": check_name,
                "status": "warn" if status == "unchecked" else status,
                "detail": fts_integrity if status == "unchecked" else fts_integrity["detail"],
            })
        except Exception as e:
            checks.append({
                "check": check_name,
                "status": "fail",
                "detail": str(e),
            })

    # 2. SQLite storage posture and payload diagnostics
    try:
        journal_mode_row = engine._store.connection.execute("PRAGMA journal_mode").fetchone()
        quick_check_row = engine._store.connection.execute("PRAGMA quick_check").fetchone()
        db_path = Path(engine._store.db_path)
        wal_path = Path(str(db_path) + "-wal")
        checks.append({
            "check": "sqlite_storage",
            "status": "pass" if quick_check_row and quick_check_row[0] == "ok" else "fail",
            "detail": {
                "database_path": str(db_path),
                "database_exists": db_path.exists(),
                "journal_mode": journal_mode_row[0] if journal_mode_row else "unknown",
                "quick_check": quick_check_row[0] if quick_check_row else "unknown",
                "database_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            },
        })
        payload_risks = scan_sqlite_payload_risks(engine._store.connection)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            engine._store.connection,
            engine._config,
            hermes_home=engine._hermes_home,
        )
        suspicious_count = (
            len(payload_risks["suspicious_data_uri_content_rows"])
            + len(payload_risks["suspicious_data_uri_tool_calls_rows"])
            + len(payload_risks["suspicious_base64_like_rows"])
            + len(payload_risks["suspicious_repetitive_assistant_rows"])
            + len(payload_risks["heartbeat_noise_rows"])
        )
        missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
        checks.append({
            "check": "payload_storage",
            "status": "warn" if suspicious_count or missing_externalized_refs else "pass",
            "detail": {
                **payload_risks,
                **externalized_stats,
                **externalized_integrity,
            },
        })
    except Exception as e:
        checks.append({
            "check": "payload_storage",
            "status": "fail",
            "detail": str(e),
        })

    try:
        protection = sensitive_pattern_status(engine._config)
        protection_status = "pass"
        if protection["enabled"] and not protection["active_patterns"]:
            protection_status = "warn"
        elif protection["unknown_patterns"]:
            protection_status = "warn"
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": protection_status,
            "detail": protection,
        })
    except Exception as e:
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": "fail",
            "detail": str(e),
        })

    # 3. FTS index sync
    try:
        msg_count = engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        fts_count = engine._store.connection.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts
            JOIN messages ON messages_fts.rowid = messages.store_id
            WHERE messages.session_id = ?
            """,
            (session_id,),
        ).fetchone()[0]
        checks.append({
            "check": "fts_index_sync",
            "status": "pass" if fts_count >= msg_count else "warn",
            "detail": f"{fts_count} session FTS rows, {msg_count} session messages",
        })
    except Exception as e:
        checks.append({
            "check": "fts_index_sync",
            "status": "fail",
            "detail": str(e),
        })

    # 3. Orphaned DAG nodes (nodes referencing store_ids that don't exist)
    try:
        all_nodes = engine._dag.get_session_nodes(session_id)
        orphaned = 0
        for node in all_nodes:
            if node.source_type == "messages":
                for sid in node.source_ids:
                    stored = engine._store.get(sid)
                    if stored is None:
                        orphaned += 1
                        break
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "pass" if orphaned == 0 else "warn",
            "detail": f"{orphaned} nodes reference missing store messages" if orphaned else "all nodes have valid sources",
        })
    except Exception as e:
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "fail",
            "detail": str(e),
        })

    try:
        leaf_health = _leaf_health_stats(engine)
        checks.append({
            "check": "leaf_health",
            "status": "warn" if (
                leaf_health["oversized_depth0_nodes"]
                or leaf_health["high_raw_low_node_session_count"]
            ) else "pass",
            "detail": leaf_health,
        })
    except Exception as e:
        checks.append({
            "check": "leaf_health",
            "status": "fail",
            "detail": str(e),
        })

    try:
        summary_quality = _summary_quality_stats(engine, session_id)
        degraded_count = (
            summary_quality.get("extreme_ratio_nodes", 0)
            + summary_quality.get("tiny_large_source_nodes", 0)
        )
        checks.append({
            "check": "summary_quality",
            "status": "warn" if degraded_count else "pass",
            "detail": summary_quality,
        })
    except Exception as e:
        checks.append({
            "check": "summary_quality",
            "status": "fail",
            "detail": str(e),
        })

    # 4. Configuration validation
    config_warnings = []
    c = engine._config
    if c.fresh_tail_count < 2:
        config_warnings.append("fresh_tail_count < 2 may cause aggressive compaction")
    runtime_context_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
    if runtime_context_threshold > 0.95:
        config_warnings.append("runtime context_threshold > 0.95 leaves very little headroom")
    if runtime_context_threshold < 0.3:
        config_warnings.append("runtime context_threshold < 0.3 triggers compaction very early")
    if c.condensation_fanin < 2:
        config_warnings.append("condensation_fanin < 2 creates excessive depth growth")
    if c.incremental_max_depth == 0:
        config_warnings.append("incremental_max_depth=0 disables condensation entirely")
    for warning in getattr(c, "config_source_warnings", []) or []:
        config_warnings.append(warning)
    for key in getattr(c, "ignored_config_yaml_lcm_keys", []) or []:
        config_warnings.append(
            f"config.yaml lcm.{key} is not a supported LCM config.yaml key and was ignored; use the matching LCM_* env var if this setting is intentional"
        )

    checks.append({
        "check": "config_validation",
        "status": "pass" if not config_warnings else "warn",
        "detail": config_warnings if config_warnings else "all settings within normal ranges",
    })

    # 5. Source-lineage hygiene
    try:
        source_stats = engine._store.get_source_stats()
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "pass",
            "detail": {
                **source_stats,
                "normalization_mode": "backcompat-normalization",
            },
        })
    except Exception as e:
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "fail",
            "detail": str(e),
        })

    # 6. Lifecycle/session fragmentation
    try:
        lifecycle_fragmentation = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "warn" if _has_lifecycle_fragmentation(lifecycle_fragmentation) else "pass",
            "detail": lifecycle_fragmentation,
        })
    except Exception as e:
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "fail",
            "detail": str(e),
        })

    # 7. Context pressure
    if engine.context_length > 0:
        usage_pct = round(engine.last_prompt_tokens / engine.context_length * 100, 1) if engine.context_length else 0
        runtime_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
        threshold_pct = round(runtime_threshold * 100, 1)
        checks.append({
            "check": "context_pressure",
            "status": "pass" if usage_pct < threshold_pct else "warn",
            "detail": f"{usage_pct}% used, compaction triggers at {threshold_pct}%",
        })

    # 8. Async/background compaction batch hygiene
    try:
        async_status = engine.get_async_compaction_status()
        prepared = int(async_status.get("prepared_batches", 0) or 0)
        preparing = int(async_status.get("preparing_batches", 0) or 0)
        rejected = int(async_status.get("rejected_batches", 0) or 0)
        failed = int(async_status.get("failed_batches", 0) or 0)
        promoted = int(async_status.get("promoted_batches", 0) or 0)
        superseded = int(async_status.get("superseded_batches", 0) or 0)
        pending = int(async_status.get("pending_batches", 0) or 0)
        enabled = bool(async_status.get("enabled", False))
        # Stale preparing rows indicate a crash mid-prep; surface as warn.
        async_status_level = "pass"
        if preparing > 0:
            async_status_level = "warn"
        elif enabled and (failed > 0 or rejected > 0):
            async_status_level = "pass"  # counts are informational
        checks.append({
            "check": "async_compaction_batches",
            "status": async_status_level,
            "detail": {
                "enabled": enabled,
                "prepared_batches": prepared,
                "pending_batches": pending,
                "preparing_batches": preparing,
                "promoted_batches": promoted,
                "rejected_batches": rejected,
                "failed_batches": failed,
                "superseded_batches": superseded,
            },
        })
        checks.append({
            "check": "async_compaction_enabled",
            "status": "pass",
            "detail": (
                f"async background compaction enabled={enabled}; "
                f"prepared_batches={prepared} rejected_batches={rejected} "
                f"promoted_batches={promoted}"
            ),
        })
    except Exception as e:
        checks.append({
            "check": "async_compaction_batches",
            "status": "fail",
            "detail": str(e),
        })

    overall = "healthy"
    if any(ch["status"] == "fail" for ch in checks):
        overall = "unhealthy"
    elif any(ch["status"] == "warn" for ch in checks):
        overall = "warnings"

    return json.dumps({
        "overall": overall,
        "runtime_identity": engine.get_runtime_identity(),
        "checks": checks,
        "guidance": doctor_guidance_for_checks(checks),
    })
