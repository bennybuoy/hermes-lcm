"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import codecs
import json
import logging
import multiprocessing
import os
import re
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
    externalized_payload_stats,
    extract_ingest_externalized_refs,
    restore_ingest_payload_placeholders,
    redact_sensitive_text,
    redact_sensitive_output_text,
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
_CURRENT_SESSION_EXPAND_MAX_TOKENS = 65_536
_CURRENT_SESSION_EXPAND_MAX_SOURCES = 200
_CURRENT_SESSION_EXPAND_MAX_CHARS = 100_000
_MANDATORY_REDACTION_LOOKAHEAD_CHARS = 8_192
_MANDATORY_REDACTION_CHARS_PER_TOKEN = 16
_BOUNDARY_PRIVATE_KEY_BEGIN_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE
)
_BOUNDARY_STANDALONE_CREDENTIAL_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:github_pat_[A-Za-z0-9_]*|(?:AKIA|ASIA)[A-Z0-9]*)\Z"
)


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


def _json_prefix_field(text: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*("(?:\\.|[^"\\])*")', text)
    if not match:
        return ""
    try:
        value = json.loads(match.group(1))
    except (ValueError, json.JSONDecodeError):
        return ""
    return value if isinstance(value, str) else ""


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


def _search_externalized_payloads(
    engine: "LCMEngine",
    *,
    query: str,
    regex_mode: bool,
    session_id: str | None,
    ref: str,
    limit: int,
    max_files: int,
    max_payload_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    diagnostics: list[dict[str, str]] = []
    hits: list[dict[str, Any]] = []
    files_scanned = 0
    bytes_scanned = 0
    scan_truncated = False
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
        }
    regex_deadline = time.monotonic() + _LCM_GREP_REGEX_OPERATION_DEADLINE_SECONDS
    regex_timeouts = 0
    candidates_seen = 0
    paths: list[Path] = []
    if ref:
        candidates_seen = 1
        paths.append(root / ref)
    else:
        exhausted = False
        with os.scandir(root) as entries:
            iterator = iter(entries)
            while candidates_seen < max_files:
                try:
                    entry = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                candidates_seen += 1
                if entry.name.endswith(".json"):
                    paths.append(root / entry.name)
        if not exhausted and candidates_seen >= max_files:
            scan_truncated = True

    for path_index, path in enumerate(paths):
        if bytes_scanned >= _LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES:
            scan_truncated = True
            break
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except FileNotFoundError:
            diagnostics.append({"ref": path.name, "error": "missing"})
            continue
        except (OSError, ValueError):
            diagnostics.append({"ref": path.name, "error": "path_escape"})
            continue
        if not resolved.is_file():
            diagnostics.append({"ref": path.name, "error": "not_a_file"})
            continue
        remaining = _LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES - bytes_scanned
        read_limit = min(remaining, max(4_096, max_payload_chars * 6 + 4_096))
        try:
            with resolved.open("rb") as handle:
                raw = handle.read(read_limit + 1)
        except OSError:
            diagnostics.append({"ref": path.name, "error": "unreadable"})
            continue
        files_scanned += 1
        bytes_scanned += min(len(raw), read_limit)
        raw_truncated = len(raw) > read_limit
        prefix = raw[:read_limit].decode("utf-8", errors="replace")
        payload_session_id = _json_prefix_field(prefix, "session_id")
        if session_id is not None and payload_session_id != session_id:
            diagnostics.append({"ref": path.name, "error": "session_mismatch"})
            continue
        content_key = re.search(r'"content"\s*:\s*', prefix)
        if content_key is None:
            diagnostics.append({"ref": path.name, "error": "content_not_in_prefix"})
            continue
        try:
            content, content_closed = _decode_json_string_prefix(
                prefix,
                content_key.end(),
                max_payload_chars,
            )
        except ValueError:
            diagnostics.append({"ref": path.name, "error": "invalid_payload"})
            continue
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
                continue
            start, end = span
        else:
            match = _externalized_literal_match(content, query)
            if match is None:
                continue
            start, end = match.span()
        line = content.count("\n", 0, start) + 1
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", end)
        if line_end < 0:
            line_end = len(content)
        context_start = max(line_start, start - 120)
        context_end = min(line_end, end + 120)
        try:
            created_match = re.search(r'"created_at"\s*:\s*([0-9.]+)', prefix)
            created_at = float(created_match.group(1)) if created_match else resolved.stat().st_mtime
        except (OSError, ValueError):
            created_at = 0.0
        matched_text = content[start:end]
        snippet = content[context_start:context_end]
        safe_snippet = redact_sensitive_text(snippet, engine._config)
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
            "byte_offset": len(content[:start].encode("utf-8")),
            "matched_text": safe_matched_text,
            "snippet": safe_snippet,
            "payload_truncated": bool(raw_truncated or not content_closed),
            "content_chars_scanned": len(content),
            "_sort_ts": created_at,
            "_sort_rank": 0.0,
            "_sort_directness": 0.0,
        })
        if len(hits) >= limit:
            scan_truncated = scan_truncated or path_index + 1 < len(paths)
            break
    return hits, diagnostics, {
        "files_scanned": files_scanned,
        "entries_scanned": candidates_seen,
        "bytes_scanned": bytes_scanned,
        "matches": len(hits),
        "scan_truncated": scan_truncated,
        "max_files": max_files,
        "max_payload_chars": max_payload_chars,
        "max_total_bytes": _LCM_GREP_EXTERNALIZED_MAX_TOTAL_BYTES,
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


def _parse_strict_int(value: Any, name: str) -> tuple[int | None, str | None]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


_LCM_GREP_VALID_SCOPES = frozenset({"current", "all", "session"})
_LCM_GREP_HARD_LIMIT_CAP = 200
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
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    source = content[content_offset:]
    if config is None:
        sliced, window_truncated = _truncate_text_to_token_budget(source, max_tokens)
    else:
        sliced, window_truncated = _bounded_cross_session_text(
            source,
            config,
            max_tokens=max_tokens,
            max_chars=_CURRENT_SESSION_EXPAND_MAX_CHARS,
        )
    redaction_changed = False
    protected_probe = source
    if config is not None:
        probe_limit = min(
            len(source),
            max(1, int(max_tokens)) * _MANDATORY_REDACTION_CHARS_PER_TOKEN
            + _MANDATORY_REDACTION_LOOKAHEAD_CHARS,
        )
        raw_probe = source[:probe_limit]
        protected_probe = redact_sensitive_output_text(raw_probe)
        redaction_changed = protected_probe != raw_probe
    if not sliced and content_offset < len(content):
        # A tiny token budget can fail to fit even the next character. Return one
        # character anyway so callers make deterministic, lossless cursor progress
        # instead of receiving has_more=true with the same content_offset forever.
        sliced = "[LCM redacted]" if redaction_changed else source[:1]
    # Offsets are raw-source cursors. Redacted pages skip the complete inspected
    # probe; ordinary pages advance only by visible raw text. Non-truncated pages
    # consume the complete source.
    if not window_truncated:
        raw_consumed = len(source)
    elif redaction_changed:
        # Skip the complete bounded probe that produced the placeholder.  This
        # prevents a subsequent cursor from entering the middle of a secret.
        raw_consumed = max(1, min(len(source), probe_limit))
    else:
        # For ordinary text output length and raw-source length are identical.
        # Advance only by the visible page so tiny budgets remain lossless.
        raw_consumed = max(1, min(len(source), len(sliced)))
    next_content_offset = content_offset + raw_consumed
    has_more = window_truncated or next_content_offset < len(content)
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": next_content_offset if has_more else 0,
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
    stored_by_id = engine._store.get_batch(source_ids)

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
        if ref:
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
        if stored.get("role") == "tool":
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
        child = engine._dag.get_node(child_id)
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
        child = engine._dag.get_node(child_id)
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


def _authorize_node_provenance_bounded(
    engine: "LCMEngine",
    candidates: list[Any],
    allowed_session_ids: frozenset[str],
    *,
    deadline: float,
) -> tuple[list[Any], dict[str, Any]]:
    """Authorize candidates with shared hard caps, bulk reads, and fail-closed gaps."""
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
    selected: list[Any] = []
    selected_ids: set[int] = set()
    for candidate in candidates[:_CROSS_SESSION_AUTH_MAX_CANDIDATES]:
        node_id = int(candidate.node_id)
        if node_id in selected_ids:
            continue
        selected_ids.add(node_id)
        selected.append(candidate)

    graph: dict[int, tuple[str, str, list[int]] | None] = {}
    incomplete: set[int] = set()
    queued: set[int] = set()
    queue: list[tuple[int, int]] = []
    for candidate in selected:
        node_id = int(candidate.node_id)
        graph[node_id] = (
            str(candidate.session_id or ""),
            str(candidate.source_type or ""),
            [int(source_id) for source_id in candidate.source_ids],
        )
        queue.append((node_id, 0))
        queued.add(node_id)

    message_ids: set[int] = set()
    expanded_nodes: set[int] = set()
    while queue:
        if time.monotonic() >= deadline:
            diagnostics["authorization_timed_out"] = True
            break
        node_id, depth = queue.pop(0)
        if node_id in expanded_nodes:
            continue
        if depth > _CROSS_SESSION_AUTH_MAX_DEPTH:
            diagnostics["authorization_truncated"] = True
            incomplete.add(node_id)
            continue
        missing_ids = [node_id] if node_id not in graph else []
        if missing_ids:
            missing_set = {node_id}
            for queued_id, _queued_depth in queue:
                if len(missing_ids) >= _CROSS_SESSION_AUTH_QUERY_BATCH:
                    break
                if queued_id in graph or queued_id in missing_set:
                    continue
                missing_ids.append(queued_id)
                missing_set.add(queued_id)
        if missing_ids:
            remaining_nodes = _CROSS_SESSION_AUTH_MAX_NODES - len(graph)
            if remaining_nodes <= 0:
                diagnostics["authorization_truncated"] = True
                incomplete.add(node_id)
                break
            missing_ids = missing_ids[:remaining_nodes]
            placeholders = ",".join("?" for _ in missing_ids)
            rows = engine._dag._conn.execute(
                f"""SELECT node_id, session_id, source_ids, source_type
                    FROM summary_nodes WHERE node_id IN ({placeholders})""",
                missing_ids,
            ).fetchall()
            found_ids: set[int] = set()
            for row in rows:
                found_id = int(row[0])
                found_ids.add(found_id)
                raw_source_ids = row[2] or "[]"
                encoded_too_large = (
                    not isinstance(raw_source_ids, str)
                    or len(raw_source_ids) > MAX_SOURCE_IDS_JSON_CHARS
                )
                edge_count = (
                    raw_source_ids.strip().count(",") + 1
                    if isinstance(raw_source_ids, str)
                    and raw_source_ids.strip() not in {"", "[]"}
                    else 0
                )
                remaining_edges = _CROSS_SESSION_AUTH_MAX_EDGES - int(
                    diagnostics["authorization_edges_checked"]
                )
                if (
                    encoded_too_large
                    or edge_count > MAX_SOURCE_IDS_PER_NODE
                    or edge_count > remaining_edges
                ):
                    source_ids = []
                    incomplete.add(found_id)
                    diagnostics["authorization_truncated"] = True
                else:
                    try:
                        decoded = json.loads(raw_source_ids)
                        if not isinstance(decoded, list):
                            raise ValueError("source_ids is not a list")
                        source_ids = [int(value) for value in decoded]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        source_ids = []
                        incomplete.add(found_id)
                graph[found_id] = (
                    str(row[1] or ""), str(row[3] or ""), source_ids
                )
            for missing_id in set(missing_ids) - found_ids:
                graph[missing_id] = None
        record = graph.get(node_id)
        expanded_nodes.add(node_id)
        if record is None:
            continue
        session_id, source_type, source_ids = record
        if not source_ids or session_id not in allowed_session_ids:
            continue
        remaining_edges = _CROSS_SESSION_AUTH_MAX_EDGES - int(
            diagnostics["authorization_edges_checked"]
        )
        if len(source_ids) > remaining_edges:
            diagnostics["authorization_truncated"] = True
            incomplete.add(node_id)
            source_ids = source_ids[:max(0, remaining_edges)]
        diagnostics["authorization_edges_checked"] += len(source_ids)
        if source_type == "messages":
            message_ids.update(source_ids)
        elif source_type == "nodes":
            if depth >= _CROSS_SESSION_AUTH_MAX_DEPTH and source_ids:
                diagnostics["authorization_truncated"] = True
                incomplete.add(node_id)
                continue
            for child_id in source_ids:
                if child_id not in queued:
                    queue.append((child_id, depth + 1))
                    queued.add(child_id)
        else:
            incomplete.add(node_id)

    diagnostics["authorization_nodes_checked"] = len(expanded_nodes)
    if len(message_ids) > _CROSS_SESSION_AUTH_MAX_MESSAGES:
        diagnostics["authorization_truncated"] = True
    bounded_message_ids = list(message_ids)[:_CROSS_SESSION_AUTH_MAX_MESSAGES]
    message_sessions: dict[int, str] = {}
    for offset in range(0, len(bounded_message_ids), _CROSS_SESSION_AUTH_QUERY_BATCH):
        if time.monotonic() >= deadline:
            diagnostics["authorization_timed_out"] = True
            break
        batch = bounded_message_ids[offset:offset + _CROSS_SESSION_AUTH_QUERY_BATCH]
        placeholders = ",".join("?" for _ in batch)
        rows = engine._store._conn.execute(
            f"SELECT store_id, session_id FROM messages WHERE store_id IN ({placeholders})",
            batch,
        ).fetchall()
        message_sessions.update({int(row[0]): str(row[1] or "") for row in rows})
    diagnostics["authorization_messages_checked"] = len(message_sessions)

    if diagnostics["authorization_timed_out"]:
        return [], diagnostics

    memo: dict[int, bool] = {}
    visiting: set[int] = set()

    def authorized(node_id: int, depth: int = 0) -> bool:
        if time.monotonic() >= deadline:
            diagnostics["authorization_timed_out"] = True
            return False
        if depth > _CROSS_SESSION_AUTH_MAX_DEPTH or node_id in incomplete:
            diagnostics["authorization_truncated"] = True
            return False
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            memo[node_id] = False
            return False
        record = graph.get(node_id)
        if record is None:
            memo[node_id] = False
            return False
        session_id, source_type, source_ids = record
        if session_id not in allowed_session_ids or not source_ids:
            memo[node_id] = False
            return False
        visiting.add(node_id)
        if source_type == "messages":
            result = all(
                message_sessions.get(source_id) in allowed_session_ids
                for source_id in source_ids
            )
        elif source_type == "nodes":
            result = all(authorized(child_id, depth + 1) for child_id in source_ids)
        else:
            result = False
        visiting.discard(node_id)
        memo[node_id] = result
        return result

    authorized_candidates = [
        candidate for candidate in selected if authorized(int(candidate.node_id))
    ]
    diagnostics["authorization_candidates_checked"] = len(selected)
    if diagnostics["authorization_timed_out"]:
        return [], diagnostics
    return authorized_candidates, diagnostics


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
        if raw_node_ids:
            for raw_node_id in raw_node_ids:
                try:
                    node_id = int(raw_node_id)
                except (TypeError, ValueError):
                    return json.dumps({"error": "node_ids must contain only integers"})
                try:
                    node = engine._dag.get_node(node_id)
                except ValueError:
                    return json.dumps({
                        "error": "node source_ids exceed authorization bounds",
                        "authorization_truncated": True,
                    })
                if node is not None:
                    candidates.append(node)
        elif query:
            discovery_limit = max_sessions * per_session_limit * 8
            try:
                candidates = engine._dag.search(
                    query, session_id=None, limit=discovery_limit
                )
            except ValueError:
                return json.dumps({
                    "error": "candidate source_ids exceed authorization bounds",
                    "authorization_truncated": True,
                })
        else:
            return json.dumps({"error": "Provide either query or node_ids"})

        owner_scoped_candidates = [
            node for node in candidates if node.session_id in allowed_session_ids
        ]
        candidates, authorization = _authorize_node_provenance_bounded(
            engine,
            owner_scoped_candidates,
            allowed_session_ids,
            deadline=deadline,
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


def _slice_loaded_content(content: Any, max_content_chars: int) -> dict[str, Any]:
    text = str(content or "")
    sliced, has_more = _bounded_cross_session_text(
        text,
        None,
        max_tokens=max(1, max_content_chars * 2),
        max_chars=max_content_chars,
    )
    return {
        "content": sliced,
        "content_chars": len(text),
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": len(sliced) if has_more else 0,
    }


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
    content_slice = _slice_loaded_content(row.get("content", "") or "", max_content_chars)
    stored_content_chars = row.get("content_chars")
    if isinstance(stored_content_chars, int) and stored_content_chars >= 0:
        returned_chars = len(content_slice["content"])
        content_slice["content_chars"] = stored_content_chars
        content_slice["content_truncated"] = stored_content_chars > returned_chars
        content_slice["next_content_offset"] = (
            returned_chars if stored_content_chars > returned_chars else 0
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

    total_messages = engine._store.count_session_load_messages(
        session_id,
        roles=roles or None,
        time_from=time_from,
        time_to=time_to,
    )
    try:
        rows = engine._store.load_session_page(
            session_id,
            after_store_id=after_store_id,
            limit=limit + 1,
            roles=roles or None,
            time_from=time_from,
            time_to=time_to,
            max_content_chars=max_content_chars,
            max_serialized_bytes=_LCM_LOAD_SESSION_MAX_SERIALIZED_BYTES,
            max_row_serialized_bytes=_LCM_LOAD_SESSION_MAX_ROW_SERIALIZED_BYTES,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
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

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    content_scope = str(args.get("content_scope") or "database").strip().lower()
    if content_scope == "files":
        content_scope = "externalized"
    if content_scope not in {"database", "externalized", "all"}:
        return json.dumps({
            "error": "content_scope must be database, externalized, or all",
        })
    regex_mode = bool(args.get("regex", False))
    if regex_mode:
        if len(query) > _LCM_GREP_REGEX_MAX_PATTERN_CHARS:
            return json.dumps({"error": "regex pattern exceeds 2000 character limit"})
        try:
            re.compile(query)
        except re.error as exc:
            return json.dumps({"error": f"invalid regex: {exc}"})
    ref = str(args.get("ref") or "").strip()
    if ref and (Path(ref).name != ref or "/" in ref or "\\" in ref):
        return json.dumps({"error": "invalid externalized ref"})
    if ref and content_scope == "database":
        return json.dumps({"error": "ref requires externalized content_scope"})

    requested_max_files = _parse_int_value(
        args.get("max_files"),
        _LCM_GREP_EXTERNALIZED_DEFAULT_FILES,
    )
    requested_max_payload_chars = _parse_int_value(
        args.get("max_payload_chars"),
        _LCM_GREP_EXTERNALIZED_DEFAULT_CHARS,
    )
    if requested_max_files <= 0 or requested_max_payload_chars <= 0:
        return json.dumps({"error": "externalized scan bounds must be positive"})
    max_files = min(requested_max_files, _LCM_GREP_EXTERNALIZED_MAX_FILES)
    max_payload_chars = min(
        requested_max_payload_chars,
        _LCM_GREP_EXTERNALIZED_MAX_CHARS,
    )

    raw_limit_arg = args.get("limit", 10)
    parsed_limit = _parse_int_value(raw_limit_arg, 10)
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)

    requested_session_scope = str(args.get("session_scope", "current")).lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id = (
        str(raw_session_id_arg).strip() if raw_session_id_arg is not None else ""
    )
    source = str(args.get("source") or "").strip() or None
    conversation_id = str(args.get("conversation_id") or "").strip() or None
    role, role_error = _parse_grep_role(args.get("role"))
    if role_error:
        return json.dumps({"error": role_error})
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

    if content_scope in {"database", "all"} and not regex_mode:
      try:
        message_search_sessions = (
            sorted(allowed_cross_session_ids)
            if session_scope == "all" and allowed_cross_session_ids is not None
            else [search_session_id]
        )
        msg_hits: list[dict[str, Any]] = []
        for authorized_session_id in message_search_sessions:
            msg_hits.extend(engine._store.search(
                query,
                session_id=authorized_session_id,
                limit=source_limit,
                sort=sort,
                source=source,
                conversation_id=conversation_id,
                role=role,
                time_from=time_from,
                time_to=time_to,
            ))
        for hit in msg_hits:
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
                    "session_id": hit["session_id"],
                    "source": hit.get("source") or "",
                    "conversation_id": hit.get("conversation_id") or "",
                    "role": hit["role"],
                    "timestamp": timestamp_value,
                    "snippet": hit.get("snippet", hit.get("content", "")[:200]),
                    "from_current_session": has_current_session and hit["session_id"] == current_session_id,
                    "_sort_ts": timestamp_value,
                    "_sort_rank": hit.get("search_rank"),
                    "_sort_directness": hit.get("_directness_score") or 0.0,
                }
            )
      except Exception as exc:
        logger.warning("Message search failed: %s", exc)

    # Summary-node search is intentionally current-session only. Cross-session
    # DAG expansion is deferred; returning summary hits without an expansion
    # contract would push this tool toward a memory-system shape rather than
    # a plugin-local archive search. Raw-message hits remain expandable across
    # sessions via lcm_expand(store_id=...).
    if (
        content_scope in {"database", "all"}
        and not regex_mode
        and session_scope == "current"
        and not raw_message_filter_active
    ):
        try:
            node_hits = engine._dag.search(
                query,
                session_id=search_session_id,
                limit=source_limit,
                sort=sort,
                source=source,
            )
            for node in node_hits:
                results.append(
                    {
                        "type": "summary",
                        "depth": f"d{node.depth}",
                        "node_id": node.node_id,
                        "session_id": node.session_id,
                        "snippet": node.summary[:300],
                        "token_count": node.token_count,
                        "expand_hint": node.expand_hint,
                        "earliest_at": node.earliest_at,
                        "latest_at": node.latest_at,
                        "from_current_session": True,
                        "_sort_ts": node.latest_at or node.created_at,
                        "_sort_rank": node.search_rank,
                        "_sort_directness": node.search_directness or 0.0,
                    }
                )
        except Exception as exc:
            logger.warning("Node search failed: %s", exc)

    if content_scope in {"externalized", "all"}:
        external_hits, externalized_diagnostics, externalized_scan = (
            _search_externalized_payloads(
                engine,
                query=query,
                regex_mode=regex_mode,
                session_id=search_session_id,
                ref=ref,
                limit=limit,
                max_files=max_files,
                max_payload_chars=max_payload_chars,
            )
        )
        for hit in external_hits:
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

    if sort == "hybrid":
        max_message_directness = max(
            (float(result.get("_sort_directness") or 0.0) for result in results if result.get("type") == "message"),
            default=0.0,
        )
        for result in results:
            if result.get("type") == "summary":
                result["_hybrid_summary_override"] = 1 if float(result.get("_sort_directness") or 0.0) >= (max_message_directness + 8.0) else 0

    results.sort(key=lambda result: _combined_result_sort_key(result, sort))
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
    return json.dumps(response)


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
        stored = engine._store.get(store_id)
        if stored is None:
            return json.dumps({"error": f"Message store_id {store_id} not found"})
        transcript_content = stored.get("content", "") or ""
        sliced = _slice_content_for_response(
            transcript_content,
            max_tokens,
            content_offset,
            config=engine._config,
        )
        engine_session_id = engine.current_session_id
        stored_session_id = stored.get("session_id", "")
        if stored_session_id != engine_session_id:
            allowed_session_ids = engine._authorized_cross_session_ids(
                kwargs.get("cross_session_capability")
            )
            if not allowed_session_ids:
                return json.dumps({
                    "error": "cross-session store_id expansion requires a trusted host capability",
                })
            if stored_session_id not in allowed_session_ids:
                return json.dumps({
                    "error": "store_id session is not authorized by the trusted host capability",
                })
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


def _read_externalized_payload_metadata_prefix(path: Path) -> tuple[str, bool, bool]:
    """Read bounded JSON metadata before the externalized payload body.

    Returns ``(prefix_text, content_string_seen, prefix_truncated)``. The content
    string body is intentionally not consumed; ``lcm_inspect`` reports bounded
    metadata only and leaves full JSON/body validation to explicit expansion.
    """
    prefix = bytearray()
    text_parts: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    prefix_truncated = False
    with path.open("rb") as handle:
        while len(prefix) < _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES:
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
            prefix_text = "".join(text_parts)
            _, content_key_seen = _inspect_top_level_json_string_fields_before_content(prefix_text)
            if content_key_seen:
                return prefix_text, True, False
        prefix_truncated = len(prefix) >= _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES and bool(handle.read(1))
    if not prefix_truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            text_parts.append(final_text)
    return "".join(text_parts), False, prefix_truncated


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
