from __future__ import annotations

"""Immutable-first message store — the source of truth.

Every message is persisted durably in SQLite. The normal model is append-only,
with one narrow opt-in exception: already-externalized summarized tool-result
rows may be rewritten to compact GC tombstones while preserving the original
row identity (`store_id`) for DAG/source lookup.
"""


import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .db_bootstrap import (
    ExternalContentFtsSpec,
    add_column_if_missing,
    configure_connection,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .config import LCMConfig
from .ingest_protection import protect_message_for_ingest, protect_messages_for_ingest
from .search_query import (
    build_snippet,
    compute_search_candidate_cap,
    compute_directness_rank_bonus_upper_bound,
    compute_directness_score,
    compute_like_fallback_fetch_limit,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    requires_like_fallback,
    sanitize_fts5_query,
    AGE_DECAY_RATE,
    should_apply_directness_rank_adjustment,
)
from .message_content import normalize_content_value as _normalize_content_value
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)


_MESSAGE_ROLE_BIAS_SQL = "CASE m.role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"
_MESSAGE_SELECT_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned, conversation_id"
)
_UNKNOWN_SOURCE = "unknown"
_LOAD_SESSION_MAX_TOOL_CALLS_ENCODED_BYTES = 64 * 1024
_LOAD_SESSION_MAX_SCALAR_TEXT_BYTES = 8 * 1024
_LOAD_SESSION_MAX_SESSION_TEXT_BYTES = 2 * 1024
_LOAD_SESSION_MAX_ROLE_TEXT_BYTES = 512
_LOAD_SESSION_MAX_ROW_MATERIALIZED_BYTES = 96 * 1024
_LOAD_SESSION_MAX_PAGE_MATERIALIZED_BYTES = 2 * 1024 * 1024
_GREP_SEARCH_VISIBLE_CHARS = 300
_GREP_SEARCH_BOUNDARY_CHARS = 8_192
_GREP_SEARCH_WINDOW_CHARS = (
    _GREP_SEARCH_VISIBLE_CHARS + (2 * _GREP_SEARCH_BOUNDARY_CHARS)
)
_EXPAND_SESSION_ID_MAX_CHARS = 512
_EXPAND_SCALAR_MAX_CHARS = 8 * 1024
_EXPAND_TOOL_CALLS_MAX_CHARS = 64 * 1024
_EXPAND_CONTENT_LOOKBEHIND_CHARS = 8 * 1024
_EXPAND_CONTENT_BLOB_READ_BYTES = 16 * 1024
_EXPAND_LEGACY_REVISION_READ_BYTES = 1024 * 1024
_EXPAND_SQL_PROGRESS_OPCODES = 1_000
_EXPAND_SQL_PROGRESS_CALLBACK_CAP = 20_000


class _LCMSQLiteConnection(sqlite3.Connection):
    """Connection that remembers the Python progress handler for safe nesting."""

    _lcm_progress_handler: Callable[[], int] | None = None
    _lcm_progress_handler_n: int = 0

    def set_progress_handler(self, progress_handler, n):
        result = super().set_progress_handler(progress_handler, n)
        self._lcm_progress_handler = progress_handler
        self._lcm_progress_handler_n = int(n)
        return result


@contextmanager
def _temporary_sqlite_progress_budget(
    conn: sqlite3.Connection,
    *,
    deadline: float | None,
) -> Iterator[None]:
    """Bound SQLite VM work and restore the caller's handler on every exit."""
    previous = getattr(conn, "_lcm_progress_handler", None)
    previous_n = max(0, int(getattr(conn, "_lcm_progress_handler_n", 0) or 0))
    callbacks = 0
    previous_opcode_accumulator = 0

    def progress() -> int:
        nonlocal callbacks, previous_opcode_accumulator
        callbacks += 1
        if callbacks >= _EXPAND_SQL_PROGRESS_CALLBACK_CAP:
            return 1
        if deadline is not None and time.monotonic() >= deadline:
            return 1
        if previous is not None and previous_n > 0:
            previous_opcode_accumulator += _EXPAND_SQL_PROGRESS_OPCODES
            while previous_opcode_accumulator >= previous_n:
                previous_opcode_accumulator -= previous_n
                if previous():
                    return 1
        return 0

    conn.set_progress_handler(progress, _EXPAND_SQL_PROGRESS_OPCODES)
    try:
        yield
    finally:
        conn.set_progress_handler(previous, previous_n if previous is not None else 0)


@contextmanager
def _suspended_sqlite_progress_handler(conn: sqlite3.Connection) -> Iterator[None]:
    """Permit one bounded bookkeeping statement after a work deadline expires."""
    previous = getattr(conn, "_lcm_progress_handler", None)
    previous_n = max(0, int(getattr(conn, "_lcm_progress_handler_n", 0) or 0))
    conn.set_progress_handler(None, 0)
    try:
        yield
    finally:
        conn.set_progress_handler(previous, previous_n if previous is not None else 0)


class _ContentBlobReader:
    """Bounded UTF-8 character reads over SQLite's O(1)-seek incremental BLOB."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        store_id: int,
        content_fingerprint: str,
        total_chars: int,
        total_bytes: int,
        deadline: float | None,
    ) -> None:
        self._conn = conn
        self._store_id = int(store_id)
        self._fingerprint = str(content_fingerprint)
        self._total_chars = max(0, int(total_chars))
        self._total_bytes = max(0, int(total_bytes))
        self._deadline = deadline
        self._blob = conn.blobopen("messages", "content", self._store_id, readonly=True)
        self._positions: dict[int, int] = {0: 0, self._total_chars: self._total_bytes}
        self._decoded_spans: list[tuple[int, int, str]] = []
        self._expired_read_allowance = 0

    def close(self) -> None:
        self._blob.close()

    def _check_deadline(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            if self._expired_read_allowance > 0:
                self._expired_read_allowance -= 1
                return
            raise TimeoutError("content_scan_deadline")

    def allow_one_expired_bounded_read(self) -> None:
        """Guarantee restart progress by permitting one already-bounded read."""
        self._expired_read_allowance = max(self._expired_read_allowance, 1)

    def remember_byte_offset(self, char_offset: int, byte_offset: int) -> None:
        self._positions[max(0, int(char_offset))] = max(0, int(byte_offset))

    def _nearest_position(self, char_offset: int) -> tuple[int, int]:
        if char_offset in self._positions:
            return char_offset, self._positions[char_offset]
        candidates = [offset for offset in self._positions if offset <= char_offset]
        best_char = max(candidates) if candidates else 0
        best_byte = self._positions.get(best_char, 0)
        row = self._conn.execute(
            """SELECT char_offset, byte_offset
               FROM lcm_content_scan_checkpoints
               WHERE store_id = ? AND content_fingerprint = ?
                 AND char_offset <= ? AND byte_offset >= 0
               ORDER BY char_offset DESC LIMIT 1""",
            (self._store_id, self._fingerprint, char_offset),
        ).fetchone()
        if row is not None and int(row[0]) > best_char:
            best_char, best_byte = int(row[0]), int(row[1])
            self._positions[best_char] = best_byte
        return best_char, best_byte

    @staticmethod
    def _decode_complete_prefix(data: bytes, *, final: bool) -> tuple[str, bytes]:
        try:
            return data.decode("utf-8"), b""
        except UnicodeDecodeError as exc:
            if not final and exc.reason == "unexpected end of data" and exc.end == len(data):
                return data[:exc.start].decode("utf-8"), data[exc.start:]
            raise

    def read_chars(self, offset: int, chars: int) -> str:
        requested = min(max(0, int(offset)), self._total_chars)
        wanted = min(max(0, int(chars)), 128 * 1024)
        target_end = min(self._total_chars, requested + wanted)
        known_requested_byte = self.known_byte_offset(requested)
        if known_requested_byte is not None:
            cursor_char, cursor_byte = requested, known_requested_byte
        else:
            cursor_char, cursor_byte = self._nearest_position(requested)
        self._blob.seek(cursor_byte)
        carry = b""
        pieces: list[str] = []
        while cursor_char < target_end:
            self._check_deadline()
            remaining_bytes = self._total_bytes - self._blob.tell()
            if remaining_bytes <= 0:
                break
            raw = self._blob.read(min(_EXPAND_CONTENT_BLOB_READ_BYTES, remaining_bytes))
            if not raw:
                break
            read_end = self._blob.tell()
            decoded, carry = self._decode_complete_prefix(
                carry + raw, final=read_end >= self._total_bytes
            )
            decoded_start_char = cursor_char
            decoded_start_byte = read_end - len(raw) - (0 if not carry else 0)
            # ``cursor_byte`` is always a proven UTF-8 boundary.  Account for
            # any incomplete suffix retained from the preceding transport read.
            decoded_start_byte = cursor_byte
            decoded_end_byte = read_end - len(carry)
            if decoded:
                self._decoded_spans.append(
                    (decoded_start_char, decoded_start_byte, decoded)
                )
                del self._decoded_spans[:-8]
                decoded_end_char = decoded_start_char + len(decoded)
                for exact_char in (requested, target_end):
                    if decoded_start_char <= exact_char <= decoded_end_char:
                        self._positions[exact_char] = decoded_start_byte + len(
                            decoded[: exact_char - decoded_start_char].encode("utf-8")
                        )
                take_start = max(0, requested - decoded_start_char)
                take_end = min(len(decoded), target_end - decoded_start_char)
                if take_end > take_start:
                    pieces.append(decoded[take_start:take_end])
                if decoded_start_char <= requested <= decoded_start_char + len(decoded):
                    self._positions[requested] = decoded_start_byte + len(
                        decoded[: requested - decoded_start_char].encode("utf-8")
                    )
                cursor_char += len(decoded)
                cursor_byte = decoded_end_byte
                self._positions[cursor_char] = cursor_byte
            elif carry:
                cursor_byte = decoded_end_byte
            # Once one bounded transport read has been decoded, return its
            # proven cursor even if the deadline crossed during that chunk.
            # The caller can then persist monotonic lexer progress instead of
            # discarding the completed read and reporting the old checkpoint.
            if cursor_char < target_end:
                self._check_deadline()
        if cursor_char >= target_end and target_end not in self._positions:
            # The target can fall within the final decoded block; derive its
            # byte boundary from the bounded returned suffix rather than from a
            # prefix traversal of the SQLite TEXT value.
            nearest_char, nearest_byte = self._nearest_position(target_end)
            if nearest_char == target_end:
                self._positions[target_end] = nearest_byte
        return "".join(pieces)

    def known_byte_offset(self, char_offset: int) -> int | None:
        """Return an offset already proved while decoding bounded BLOB chunks."""
        target = min(max(0, int(char_offset)), self._total_chars)
        exact = self._positions.get(target)
        if exact is not None:
            return exact
        for start_char, start_byte, decoded in reversed(self._decoded_spans):
            if start_char <= target <= start_char + len(decoded):
                exact = start_byte + len(
                    decoded[: target - start_char].encode("utf-8")
                )
                self._positions[target] = exact
                return exact
        return None

    def byte_offset(self, char_offset: int) -> int:
        target = min(max(0, int(char_offset)), self._total_chars)
        if target not in self._positions:
            self.read_chars(target, 1 if target < self._total_chars else 0)
        if target in self._positions:
            return self._positions[target]
        # ``read_chars(target, 0)`` has no reason to advance. Read from the
        # nearest checkpoint in bounded chunks until the exact boundary exists.
        nearest_char, _ = self._nearest_position(target)
        while nearest_char < target:
            self.read_chars(nearest_char, min(16 * 1024, target - nearest_char))
            nearest_char, _ = self._nearest_position(target)
        return self._positions.get(target, self._total_bytes)


def _advance_legacy_content_revision(
    conn: sqlite3.Connection,
    *,
    store_id: int,
    revision_row: tuple | None,
    deadline: float | None,
    allow_one_expired_chunk: bool,
) -> dict[str, Any]:
    """Incrementally publish legacy content length metadata from a BLOB handle.

    ``len(sqlite3.Blob)`` delegates to SQLite's native blob-size metadata and is
    O(1). Character counting then advances in bounded byte chunks, persisting a
    UTF-8 boundary after every call so restarts make monotonic progress.
    """
    with _suspended_sqlite_progress_handler(conn):
        content_type = conn.execute(
            "SELECT typeof(content) FROM messages WHERE store_id=?", (store_id,)
        ).fetchone()
    if content_type is None:
        return {"status": "changed"}
    if content_type[0] == "null":
        with _suspended_sqlite_progress_handler(conn):
            conn.execute(
                """INSERT OR REPLACE INTO lcm_content_revisions
                       (store_id, content_fingerprint, content_chars, content_bytes,
                        storage_version, scan_byte_offset)
                   VALUES (?, lower(hex(randomblob(16))), 0, 0, 2, 0)""",
                (store_id,),
            )
        return {
            "status": "complete",
            "content_chars": 0,
            "content_bytes": 0,
            "scan_byte_offset": 0,
            "storage_version": 2,
        }
    if content_type[0] not in {"text", "blob"}:
        return {"status": "invalid_payload"}

    blob = conn.blobopen("messages", "content", int(store_id), readonly=True)
    try:
        total_bytes = len(blob)
        if revision_row is None:
            with _suspended_sqlite_progress_handler(conn):
                fingerprint = conn.execute(
                    "SELECT lower(hex(randomblob(16)))"
                ).fetchone()[0]
            content_chars = 0
            scan_byte_offset = 0
            with _suspended_sqlite_progress_handler(conn):
                conn.execute(
                    "DELETE FROM lcm_content_scan_checkpoints WHERE store_id=?",
                    (store_id,),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO lcm_content_revisions
                           (store_id, content_fingerprint, content_chars, content_bytes,
                            storage_version, scan_byte_offset)
                       VALUES (?, ?, 0, ?, 1, 0)""",
                    (store_id, fingerprint, total_bytes),
                )
        else:
            fingerprint = str(revision_row[0])
            content_chars = max(0, int(revision_row[1] or 0))
            recorded_bytes = max(0, int(revision_row[2] or 0))
            scan_byte_offset = max(0, int(revision_row[4] or 0))
            if (
                recorded_bytes != total_bytes
                or scan_byte_offset > total_bytes
                or (scan_byte_offset == 0 and content_chars != 0)
            ):
                with _suspended_sqlite_progress_handler(conn):
                    fingerprint = conn.execute(
                        "SELECT lower(hex(randomblob(16)))"
                    ).fetchone()[0]
                content_chars = 0
                scan_byte_offset = 0
                with _suspended_sqlite_progress_handler(conn):
                    conn.execute(
                        "DELETE FROM lcm_content_scan_checkpoints WHERE store_id=?",
                        (store_id,),
                    )

        blob.seek(scan_byte_offset)
        durable_byte_offset = scan_byte_offset
        added_chars = 0
        bytes_read = 0
        carry = b""
        while (
            durable_byte_offset < total_bytes
            and bytes_read < _EXPAND_LEGACY_REVISION_READ_BYTES
        ):
            if (
                deadline is not None
                and time.monotonic() >= deadline
                and (bytes_read > 0 or not allow_one_expired_chunk)
            ):
                break
            remaining = total_bytes - blob.tell()
            if remaining <= 0:
                break
            raw = blob.read(
                min(
                    _EXPAND_CONTENT_BLOB_READ_BYTES,
                    remaining,
                    _EXPAND_LEGACY_REVISION_READ_BYTES - bytes_read,
                )
            )
            if not raw:
                break
            bytes_read += len(raw)
            candidate = carry + raw
            final = blob.tell() >= total_bytes
            decoded, carry = _ContentBlobReader._decode_complete_prefix(
                candidate, final=final
            )
            durable_byte_offset += len(candidate) - len(carry)
            added_chars += len(decoded)
            if deadline is not None and time.monotonic() >= deadline:
                break

        complete = durable_byte_offset >= total_bytes and not carry
        content_chars += added_chars
        storage_version = 2 if complete else 1
        with _suspended_sqlite_progress_handler(conn):
            conn.execute(
                """INSERT OR REPLACE INTO lcm_content_revisions
                       (store_id, content_fingerprint, content_chars, content_bytes,
                        storage_version, scan_byte_offset)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    store_id,
                    fingerprint,
                    content_chars,
                    total_bytes,
                    storage_version,
                    durable_byte_offset,
                ),
            )
        return {
            "status": "complete" if complete else "scan_pending",
            "content_fingerprint": fingerprint,
            "content_chars": content_chars,
            "content_bytes": total_bytes,
            "scan_byte_offset": durable_byte_offset,
            "storage_version": storage_version,
        }
    finally:
        blob.close()


def _grep_bounded_search_projection(
    terms: list[str], *, alias: str = ""
) -> tuple[str, str, list[str]]:
    """Return an explicit grep projection and its SELECT-clause bind args."""
    prefix = f"{alias}." if alias else ""
    content = f"{prefix}content"
    tool_calls = f"{prefix}tool_calls"
    first_term = terms[0] if terms else ""
    if first_term:
        position = f"instr(lower(CAST({content} AS TEXT)), lower(?))"
        window_start = (
            f"CASE WHEN {position} > {_GREP_SEARCH_BOUNDARY_CHARS} "
            f"THEN {position} - {_GREP_SEARCH_BOUNDARY_CHARS} ELSE 1 END"
        )
        # The position expression appears in both the bounded payload and its
        # raw-start metadata, so each occurrence has its own bound parameters.
        select_args = [first_term, first_term, first_term, first_term]
    else:
        window_start = "1"
        select_args = []
    bounded_content = (
        f"CASE WHEN typeof({content}) = 'text' THEN "
        f"substr(CAST({content} AS TEXT), {window_start}, {_GREP_SEARCH_WINDOW_CHARS}) END"
    )
    columns = ", ".join(
        [
            f"CASE WHEN typeof({prefix}store_id) = 'integer' THEN {prefix}store_id END",
            f"substr(CAST({prefix}session_id AS TEXT), 1, 512)",
            f"substr(CAST({prefix}source AS TEXT), 1, 512)",
            f"substr(CAST({prefix}role AS TEXT), 1, 128)",
            bounded_content,
            f"substr(CAST({prefix}tool_call_id AS TEXT), 1, 512)",
            "NULL",
            f"substr(CAST({prefix}tool_name AS TEXT), 1, 512)",
            f"CASE WHEN typeof({prefix}timestamp) IN ('integer', 'real') THEN {prefix}timestamp END",
            f"CASE WHEN typeof({prefix}token_estimate) = 'integer' THEN {prefix}token_estimate END",
            f"CASE WHEN typeof({prefix}pinned) = 'integer' THEN {prefix}pinned END",
            f"substr(CAST({prefix}conversation_id AS TEXT), 1, 512)",
        ]
    )
    metadata = ", ".join(
        [
            f"COALESCE(length(CAST({content} AS BLOB)), 0)",
            f"COALESCE(length(CAST({content} AS TEXT)), 0)",
            f"COALESCE(length(CAST({tool_calls} AS BLOB)), 0)",
            window_start,
            f"typeof({content})",
            f"typeof({tool_calls})",
        ]
    )
    return columns, metadata, select_args


class SessionLoadPage(list[dict[str, Any]]):
    """List-compatible bounded page with aggregate exhaustion metadata."""

    budget_exhausted: bool = False
    total_messages: int = 0


def _legacy_blank_source_clause(column: str) -> str:
    # SQLite TRIM() only strips spaces unless given an explicit character set.
    # Match Python's write-time `str.strip()` behavior for common ASCII whitespace
    # so legacy tabs/newlines do not become a fake attributed source bucket.
    whitespace_chars = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)"
    return f"({column} IS NULL OR TRIM({column}, {whitespace_chars}) = '')"


def _normalize_source_value(source: str | None) -> str:
    normalized = (source or "").strip()
    return normalized or _UNKNOWN_SOURCE


def _normalize_conversation_id_value(conversation_id: str | None) -> str:
    return (conversation_id or "").strip()


def _source_filter_clause(column: str, source: str | None) -> tuple[str | None, list[str]]:
    normalized = _normalize_source_value(source) if source is not None else ""
    if not normalized:
        return None, []
    if normalized == _UNKNOWN_SOURCE:
        return f"({column} = ? OR {_legacy_blank_source_clause(column)})", [_UNKNOWN_SOURCE]
    return f"{column} = ?", [normalized]


def _conversation_filter_clause(column: str, conversation_id: str | None) -> tuple[str | None, list[str]]:
    normalized = _normalize_conversation_id_value(conversation_id)
    if not normalized:
        return None, []
    return f"{column} = ?", [normalized]


def _message_role_bias(role: str | None) -> float:
    if role == "user":
        return 0.0
    if role == "assistant":
        return 1.0
    if role == "tool":
        return 2.0
    return 1.0


def _message_directness_score(role: str | None, content: str | None, terms: List[str], phrases: List[str] | None = None) -> float:
    score = compute_directness_score(content or "", terms, phrases)
    if role == "tool":
        stripped = (content or "").lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            score -= 4.0
    return score


def _build_search_order_by(
    sort: str | None,
    timestamp_expr: str,
    role_penalty_expr: str | None = None,
) -> str:
    normalized = normalize_search_sort(sort)
    order_parts: list[str] = []
    if normalized == "relevance":
        if role_penalty_expr:
            order_parts.extend(["rank ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend(["rank ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    if normalized == "hybrid":
        blended = f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {timestamp_expr}) / 3600.0)) * {AGE_DECAY_RATE})))"
        if role_penalty_expr:
            order_parts.extend([f"{blended} ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend([f"{blended} ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    order_parts.append(f"{timestamp_expr} DESC")
    if role_penalty_expr:
        order_parts.append(f"{role_penalty_expr} ASC")
    order_parts.append("rank ASC")
    return ", ".join(order_parts)


def _fallback_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(result.get("_fallback_score") or 0.0)
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (-score, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, -score, -directness)


def _fts_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (rank_value, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        return (blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, rank_value, 0.0)


def _fts_primary_value(result: Dict[str, Any], sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        timestamp = float(result.get("timestamp") or 0.0)
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        return rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
    return rank_value


def build_message_fts_spec() -> ExternalContentFtsSpec:
    return ExternalContentFtsSpec(
        table_name="messages_fts",
        content_table="messages",
        content_rowid="store_id",
        indexed_column="content",
        trigger_sqls=(
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_insert
                AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                    VALUES (new.store_id, new.content);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_delete
                AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.store_id, old.content);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS msg_fts_update
                AFTER UPDATE OF content ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.store_id, old.content);
                INSERT INTO messages_fts(rowid, content)
                    VALUES (new.store_id, new.content);
            END;
            """,
        ),
    )


class MessageStoreStartupError(RuntimeError):
    """Raised when SQLite cannot complete MessageStore initialization."""


class MessageStore:
    """SQLite-backed immutable message store."""

    def __init__(self, db_path: str | Path, *, ingest_protection_config=None, hermes_home: str = ""):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ingest_protection_config = ingest_protection_config or LCMConfig(database_path=str(self.db_path))
        self._hermes_home = hermes_home or str(self.db_path.parent)
        self._conn: Optional[sqlite3.Connection] = None
        # ``self._conn`` is shared across threads (the connection is opened with
        # ``check_same_thread=False``). SQLite's own C-level mutex serializes
        # statements at the engine layer, but the Python ``sqlite3`` module
        # releases the GIL while the C call runs. Under heavy thread contention
        # with concurrent HTTPS clients in the same process, downstream
        # operators have observed on-disk corruption that is consistent with
        # external bytes landing inside SQLite's write path (e.g. the first
        # 28 bytes of the database file replaced with a TLS record header +
        # ciphertext while the "SQLit" magic remains intact).
        #
        # This re-entrant lock is defense-in-depth: it forces all write call
        # sites that use ``self._conn`` to be serialized at the Python layer,
        # eliminating any window where Python-side buffer reuse or memory
        # aliasing could intersect SQLite's flush of a write. It does not
        # change semantics for single-threaded callers and adds only a single
        # uncontended ``RLock.acquire``/``release`` pair per operation.
        self._write_lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            factory=_LCMSQLiteConnection,
        )
        try:
            refuse_schema_version_too_new(self._conn)
            configure_connection(self._conn)
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source TEXT DEFAULT '',
                    conversation_id TEXT DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls TEXT,
                    tool_name TEXT,
                    timestamp REAL NOT NULL,
                    token_estimate INTEGER DEFAULT 0,
                    pinned INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session
                    ON messages(session_id, store_id);
                CREATE INDEX IF NOT EXISTS idx_msg_session_ts
                    ON messages(session_id, timestamp);

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            run_versioned_migrations(
                self._conn,
                fts_specs=(build_message_fts_spec(),),
            )
            self._ensure_source_column()
            self._ensure_conversation_id_column()
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            self._conn.close()
            self._conn = None
            raise MessageStoreStartupError(
                f"Could not initialize MessageStore at {self.db_path}: {exc}"
            ) from exc
        except Exception:
            self._conn.close()
            self._conn = None
            raise

    def _ensure_source_column(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        add_column_if_missing(
            self._conn, columns, "source",
            "ALTER TABLE messages ADD COLUMN source TEXT DEFAULT ''",
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_source_session ON messages(source, session_id, store_id)"
        )

    def _ensure_conversation_id_column(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        add_column_if_missing(
            self._conn, columns, "conversation_id",
            "ALTER TABLE messages ADD COLUMN conversation_id TEXT DEFAULT ''",
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_conversation_session ON messages(conversation_id, session_id, store_id)"
        )

    # -- Write operations ---------------------------------------------------

    def append(self, session_id: str, msg: Dict[str, Any],
               token_estimate: int = 0, source: str = "",
               conversation_id: str = "") -> int:
        """Persist a message and return its store_id."""
        msg = protect_message_for_ingest(
            msg,
            config=self._ingest_protection_config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )
        tool_calls = msg.get("tool_calls")
        tc_json = json.dumps(tool_calls) if tool_calls else None

        with self._write_lock:
            cur = self._conn.execute(
                """INSERT INTO messages
                   (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
                    tool_name, timestamp, token_estimate, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    _normalize_source_value(source),
                    _normalize_conversation_id_value(conversation_id),
                    msg.get("role", "unknown"),
                    _normalize_content_value(msg.get("content")),
                    msg.get("tool_call_id"),
                    tc_json,
                    msg.get("tool_name"),
                    time.time(),
                    token_estimate,
                    0,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def append_batch(self, session_id: str,
                     messages: List[Dict[str, Any]],
                     token_estimates: List[int] | None = None,
                     source: str = "",
                     conversation_id: str = "") -> List[int]:
        """Persist multiple messages in one transaction. Returns store_ids."""
        protected_messages = protect_messages_for_ingest(
            messages,
            config=self._ingest_protection_config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )
        return self._append_protected_batch(
            session_id,
            protected_messages,
            token_estimates,
            source=source,
            conversation_id=conversation_id,
        )

    def _append_protected_batch(self, session_id: str,
                                messages: List[Dict[str, Any]],
                                token_estimates: List[int] | None = None,
                                source: str = "",
                                conversation_id: str = "") -> List[int]:
        """Persist messages that already passed ingest protection.

        This is an internal fast path for callers that need the protected form
        before storage, for example to update active replay with raw-payload
        stubs. Direct callers should use ``append_batch`` so storage-boundary
        payload protection cannot be bypassed accidentally.
        """
        if token_estimates is None:
            token_estimates = [0] * len(messages)

        ids = []
        with self._write_lock, self._conn:
            for msg, est in zip(messages, token_estimates):
                tc = msg.get("tool_calls")
                tc_json = json.dumps(tc) if tc else None
                ts = time.time()
                cur = self._conn.execute(
                    """INSERT INTO messages
                       (session_id, source, conversation_id, role, content, tool_call_id, tool_calls,
                        tool_name, timestamp, token_estimate, pinned)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        _normalize_source_value(source),
                        _normalize_conversation_id_value(conversation_id),
                        msg.get("role", "unknown"),
                        _normalize_content_value(msg.get("content")),
                        msg.get("tool_call_id"),
                        tc_json,
                        msg.get("tool_name"),
                        ts,
                        est,
                        0,
                    ),
                )
                ids.append(cur.lastrowid)
        return ids

    def reassign_session_messages(self, old_session_id: str, new_session_id: str) -> int:
        """Move all persisted messages from one session_id to another."""
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        with self._write_lock:
            cur = self._conn.execute(
                "UPDATE messages SET session_id = ? WHERE session_id = ?",
                (new_session_id, old_session_id),
            )
            self._conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        with self._write_lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            deleted = cur.rowcount if cur.rowcount is not None else 0
            return deleted

    def gc_externalized_tool_result(self, store_id: int, placeholder: str) -> bool:
        """Rewrite one unpinned tool-result row to a compact GC placeholder."""
        with self._write_lock:
            row = self._conn.execute(
                "SELECT role, pinned, content, tool_call_id FROM messages WHERE store_id = ?",
                (store_id,),
            ).fetchone()
            if row is None:
                return False
            role, pinned, current_content, tool_call_id = row
            if role != "tool" or bool(pinned) or current_content == placeholder:
                return False
            placeholder_tokens = count_message_tokens(
                {
                    "role": "tool",
                    "content": placeholder,
                    "tool_call_id": tool_call_id,
                }
            )
            self._conn.execute(
                "UPDATE messages SET content = ?, token_estimate = ? WHERE store_id = ?",
                (placeholder, placeholder_tokens, store_id),
            )
            self._conn.commit()
            return True

    def pin(self, store_id: int) -> None:

        """Mark a message as pinned (protected from pruning)."""
        with self._write_lock:
            self._conn.execute(
                "UPDATE messages SET pinned = 1 WHERE store_id = ?", (store_id,)
            )
            self._conn.commit()

    def unpin(self, store_id: int) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE messages SET pinned = 0 WHERE store_id = ?", (store_id,)
            )
            self._conn.commit()

    # -- Read operations ----------------------------------------------------

    def get(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single message by store_id."""
        row = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id = ?", (store_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_for_expansion(
        self,
        store_id: int,
        *,
        authorize_session: Callable[[str], bool],
        content_offset: int,
        max_content_chars: int,
        content_lookahead_chars: int,
        boundary_scanner: Callable[
            ..., Dict[str, Any]
        ] | None = None,
        boundary_scan_deadline: float | None = None,
    ) -> Dict[str, Any]:
        """Authorize bounded ownership metadata before reading payload columns.

        Both reads share one SQLite savepoint/read snapshot.  The first query
        projects only the row identity and a strictly guarded session owner;
        an unauthorized row therefore never causes ``content`` or
        ``tool_calls`` to be read by SQLite or materialized in Python.  After
        authorization, the second query projects a bounded content window and
        bounded scalar metadata from that same immutable snapshot.
        """
        savepoint = "lcm_expand_message_snapshot"
        bounded_offset = max(0, int(content_offset))
        visible_chars = max(1, int(max_content_chars))
        lookahead_chars = max(0, int(content_lookahead_chars))
        deadline_was_future = (
            boundary_scan_deadline is None
            or time.monotonic() < boundary_scan_deadline
        )
        with self._write_lock:
            progress_budget = _temporary_sqlite_progress_budget(
                self._conn, deadline=boundary_scan_deadline
            )
            progress_budget.__enter__()
            content_reader: _ContentBlobReader | None = None
            savepoint_started = False
            try:
                self._conn.execute(f"SAVEPOINT {savepoint}")
                savepoint_started = True
                metadata = self._conn.execute(
                    """SELECT
                               CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                               COALESCE(length(CAST(session_id AS BLOB)), 0),
                               COALESCE(length(CAST(session_id AS TEXT)), 0),
                               CASE WHEN typeof(session_id) = 'text'
                                          AND COALESCE(length(CAST(session_id AS BLOB)), 0) <= ?
                                          AND COALESCE(length(CAST(session_id AS TEXT)), 0) <= ?
                                    THEN substr(CAST(session_id AS TEXT), 1, ?) END,
                               typeof(session_id)
                        FROM messages WHERE store_id = ? LIMIT 1""",
                    (
                        _EXPAND_SESSION_ID_MAX_CHARS * 4,
                        _EXPAND_SESSION_ID_MAX_CHARS,
                        _EXPAND_SESSION_ID_MAX_CHARS,
                        store_id,
                    ),
                ).fetchone()
                if metadata is None:
                    result = {"status": "not_found"}
                elif (
                    metadata[0] is None
                    or metadata[4] != "text"
                    or int(metadata[1] or 0) > _EXPAND_SESSION_ID_MAX_CHARS * 4
                    or int(metadata[2] or 0) > _EXPAND_SESSION_ID_MAX_CHARS
                    or not isinstance(metadata[3], str)
                ):
                    result = {"status": "invalid_metadata"}
                else:
                    session_id = metadata[3]
                    if not authorize_session(session_id):
                        result = {
                            "status": "unauthorized",
                            "session_id": session_id,
                        }
                    else:
                        scalar_columns = (
                            "source", "role", "tool_call_id", "tool_name",
                            "conversation_id",
                        )
                        scalar_projection = ", ".join(
                            f"CASE WHEN {column} IS NULL THEN NULL "
                            f"WHEN typeof({column}) = 'text' "
                            f"AND COALESCE(length(CAST({column} AS TEXT)), 0) <= {_EXPAND_SCALAR_MAX_CHARS} "
                            f"THEN substr(CAST({column} AS TEXT), 1, {_EXPAND_SCALAR_MAX_CHARS}) END"
                            for column in scalar_columns
                        )
                        revision_row = self._conn.execute(
                            """SELECT content_fingerprint, content_chars,
                                      content_bytes, storage_version, scan_byte_offset
                               FROM lcm_content_revisions WHERE store_id = ?""",
                            (store_id,),
                        ).fetchone()
                        if revision_row is None or int(revision_row[3] or 0) < 2:
                            revision_progress = _advance_legacy_content_revision(
                                self._conn,
                                store_id=store_id,
                                revision_row=revision_row,
                                deadline=boundary_scan_deadline,
                                allow_one_expired_chunk=deadline_was_future,
                            )
                            if revision_progress["status"] in {
                                "changed", "invalid_payload"
                            }:
                                result = {"status": revision_progress["status"]}
                                with _suspended_sqlite_progress_handler(self._conn):
                                    self._conn.execute(f"RELEASE {savepoint}")
                                return result
                            if (
                                revision_progress["status"] == "scan_pending"
                                or (
                                    boundary_scan_deadline is not None
                                    and time.monotonic() >= boundary_scan_deadline
                                )
                            ):
                                result = {
                                    "status": "scan_pending",
                                    "session_id": session_id,
                                    "content_chars_scanned": int(
                                        revision_progress["content_chars"]
                                    ),
                                    "content_bytes": int(
                                        revision_progress["content_bytes"]
                                    ),
                                    "content_scan_byte_offset": int(
                                        revision_progress["scan_byte_offset"]
                                    ),
                                }
                                with _suspended_sqlite_progress_handler(self._conn):
                                    self._conn.execute(f"RELEASE {savepoint}")
                                return result
                        content_metadata = self._conn.execute(
                            """SELECT r.content_chars, typeof(m.content),
                                      r.content_fingerprint, r.content_bytes
                               FROM messages AS m
                               JOIN lcm_content_revisions AS r
                                 ON r.store_id = m.store_id
                               WHERE m.store_id = ?
                                 AND m.session_id = ?
                                 AND COALESCE(length(CAST(m.session_id AS BLOB)), 0) = ?
                                 AND COALESCE(length(CAST(m.session_id AS TEXT)), 0) = ?
                               LIMIT 1""",
                            (
                                store_id,
                                session_id,
                                int(metadata[1]),
                                int(metadata[2]),
                            ),
                        ).fetchone()
                        if content_metadata is None:
                            result = {"status": "changed"}
                            self._conn.execute(f"RELEASE {savepoint}")
                            return result
                        total_content_chars = int(content_metadata[0] or 0)
                        if content_metadata[1] not in {"text", "null"}:
                            result = {"status": "invalid_payload"}
                            self._conn.execute(f"RELEASE {savepoint}")
                            return result
                        total_content_bytes = int(content_metadata[3] or 0)
                        if content_metadata[1] == "text" and total_content_chars:
                            content_reader = _ContentBlobReader(
                                self._conn,
                                store_id=store_id,
                                content_fingerprint=content_metadata[2],
                                total_chars=total_content_chars,
                                total_bytes=total_content_bytes,
                                deadline=boundary_scan_deadline,
                            )

                        boundary_scan: Dict[str, Any] = {}
                        if boundary_scanner is not None and total_content_chars:
                            checkpoint_row = self._conn.execute(
                                """SELECT char_offset, byte_offset, mode, quote, quote_backslashes
                                   FROM lcm_content_scan_checkpoints
                                   WHERE store_id = ?
                                     AND content_fingerprint = ?
                                     AND char_offset <= ?
                                   ORDER BY char_offset DESC
                                   LIMIT 1""",
                                (
                                    store_id,
                                    content_metadata[2],
                                    bounded_offset,
                                ),
                            ).fetchone()
                            checkpoint = {
                                "offset": int(checkpoint_row[0]),
                                "byte_offset": int(checkpoint_row[1] or 0),
                                "mode": checkpoint_row[2],
                                "quote": checkpoint_row[3],
                                "quote_backslashes": int(checkpoint_row[4] or 0),
                            } if checkpoint_row is not None else {
                                "offset": 0,
                                "byte_offset": 0,
                                "mode": "normal",
                                "quote": None,
                                "quote_backslashes": 0,
                            }
                            if (
                                checkpoint_row is not None
                                and int(checkpoint["offset"]) < bounded_offset
                                and content_reader is not None
                            ):
                                content_reader.remember_byte_offset(
                                    int(checkpoint["offset"]),
                                    int(checkpoint["byte_offset"]),
                                )
                                content_reader.allow_one_expired_bounded_read()

                            def read_content_chunk(offset: int, chars: int) -> str:
                                bounded_chunk_offset = min(
                                    max(0, int(offset)), total_content_chars
                                )
                                bounded_chunk_chars = min(
                                    max(0, int(chars)), 16 * 1024
                                )
                                if bounded_chunk_chars <= 0:
                                    return ""
                                if content_reader is None:
                                    return ""
                                try:
                                    return content_reader.read_chars(
                                        bounded_chunk_offset, bounded_chunk_chars
                                    )
                                except (TimeoutError, sqlite3.OperationalError) as exc:
                                    if (
                                        isinstance(exc, sqlite3.OperationalError)
                                        and "interrupted" not in str(exc).lower()
                                    ):
                                        raise
                                    return ""

                            effective_scan_deadline = boundary_scan_deadline
                            if (
                                boundary_scan_deadline is not None
                                and deadline_was_future
                                and time.monotonic() >= boundary_scan_deadline
                            ):
                                # Authorization/metadata work consumed the
                                # caller's positive budget. Permit only a tiny
                                # recovery slice so the byte-native cursor can
                                # advance instead of livelocking across restarts.
                                effective_scan_deadline = time.monotonic() + 0.005
                                if content_reader is not None:
                                    content_reader.allow_one_expired_bounded_read()
                            boundary_scan = boundary_scanner(
                                read_content_chunk,
                                total_content_chars,
                                bounded_offset,
                                checkpoint=checkpoint,
                                deadline=effective_scan_deadline,
                            )
                            checkpoint_offset = min(
                                total_content_chars,
                                max(0, int(boundary_scan.get(
                                    "checkpoint_offset", checkpoint["offset"]
                                ))),
                            )
                            checkpoint_mode = str(
                                boundary_scan.get("checkpoint_mode") or "unknown"
                            )
                            checkpoint_quote = boundary_scan.get("checkpoint_quote")
                            checkpoint_backslashes = max(0, int(
                                boundary_scan.get(
                                    "checkpoint_quote_backslashes", 0
                                ) or 0
                            ))
                            try:
                                known_byte_offset = (
                                    content_reader.known_byte_offset(checkpoint_offset)
                                    if content_reader is not None
                                    else 0
                                )
                                checkpoint_byte_offset = (
                                    known_byte_offset
                                    if known_byte_offset is not None
                                    else content_reader.byte_offset(checkpoint_offset)
                                )
                            except (TimeoutError, sqlite3.OperationalError) as exc:
                                if (
                                    isinstance(exc, sqlite3.OperationalError)
                                    and "interrupted" not in str(exc).lower()
                                ):
                                    raise
                                checkpoint_byte_offset = int(
                                    checkpoint.get("byte_offset", 0) or 0
                                )
                                checkpoint_offset = int(checkpoint["offset"])
                                checkpoint_mode = str(checkpoint["mode"])
                                checkpoint_quote = checkpoint["quote"]
                                checkpoint_backslashes = int(
                                    checkpoint["quote_backslashes"]
                                )
                            try:
                                with _suspended_sqlite_progress_handler(self._conn):
                                    self._conn.execute(
                                        """INSERT OR REPLACE INTO lcm_content_scan_checkpoints
                                               (store_id, content_fingerprint, char_offset, byte_offset,
                                                mode, quote, quote_backslashes)
                                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                        (
                                            store_id,
                                            content_metadata[2],
                                            checkpoint_offset,
                                            checkpoint_byte_offset,
                                            checkpoint_mode,
                                            checkpoint_quote,
                                            checkpoint_backslashes,
                                        ),
                                    )
                            except sqlite3.OperationalError as exc:
                                # A sibling writer can advance the row after the
                                # authorization snapshot. The old checkpoint is
                                # then unnecessary and must never turn a safe
                                # read into a failed expansion.
                                if not any(
                                    marker in str(exc).lower()
                                    for marker in ("locked", "interrupted")
                                ):
                                    raise
                                checkpoint_offset = int(checkpoint["offset"])
                                checkpoint_byte_offset = int(
                                    checkpoint.get("byte_offset", 0) or 0
                                )
                                checkpoint_mode = str(checkpoint["mode"])
                                checkpoint_quote = checkpoint["quote"]
                                checkpoint_backslashes = int(
                                    checkpoint["quote_backslashes"]
                                )
                            boundary_scan["checkpoint_offset"] = checkpoint_offset

                        scan_offset = min(
                            max(
                                0,
                                int(boundary_scan.get("safe_content_offset", bounded_offset)),
                            ),
                            total_content_chars,
                        )
                        window_start = max(
                            0, scan_offset - _EXPAND_CONTENT_LOOKBEHIND_CHARS
                        )
                        window_chars = (
                            (scan_offset - window_start)
                            + visible_chars
                            + lookahead_chars
                        )
                        if boundary_scan.get("boundary_pending"):
                            # The returned cursor remains inside a conservatively
                            # sensitive span.  Do not materialize body bytes; the
                            # next request resumes the bounded chunk scanner.
                            window_chars = 0

                        content_window = ""
                        if window_chars and content_reader is not None:
                            try:
                                content_window = content_reader.read_chars(
                                    window_start, window_chars
                                )
                            except (TimeoutError, sqlite3.OperationalError) as exc:
                                if (
                                    isinstance(exc, sqlite3.OperationalError)
                                    and "interrupted" not in str(exc).lower()
                                ):
                                    raise
                                boundary_scan = {
                                    "boundary_redacted": True,
                                    "boundary_pending": True,
                                    "safe_content_offset": checkpoint_offset,
                                    "checkpoint_offset": checkpoint_offset,
                                }
                                scan_offset = checkpoint_offset
                                window_start = scan_offset

                        payload = self._conn.execute(
                            f"""SELECT
                                       CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                                       substr(CAST(session_id AS TEXT), 1, {_EXPAND_SESSION_ID_MAX_CHARS}),
                                       {scalar_projection},
                                       ?,
                                       CASE WHEN tool_calls IS NULL THEN NULL
                                            WHEN typeof(tool_calls) = 'text'
                                             AND COALESCE(length(CAST(tool_calls AS TEXT)), 0) <= {_EXPAND_TOOL_CALLS_MAX_CHARS}
                                            THEN substr(CAST(tool_calls AS TEXT), 1, {_EXPAND_TOOL_CALLS_MAX_CHARS}) END,
                                       CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                                       CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                                       CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                                       ?,
                                       COALESCE(length(CAST(tool_calls AS TEXT)), 0),
                                       typeof(content), typeof(tool_calls)
                                FROM messages
                                WHERE store_id = ?
                                  AND session_id = ?
                                  AND COALESCE(length(CAST(session_id AS BLOB)), 0) = ?
                                  AND COALESCE(length(CAST(session_id AS TEXT)), 0) = ?
                                LIMIT 1""",
                            (
                                content_window,
                                total_content_chars,
                                store_id,
                                session_id,
                                int(metadata[1]),
                                int(metadata[2]),
                            ),
                        ).fetchone()
                        if payload is None:
                            result = {"status": "changed"}
                        else:
                            # Reorder the explicit bounded projection into the
                            # store's standard message shape without ever
                            # constructing an unbounded row.
                            standard_row = (
                                payload[0], payload[1], payload[2], payload[3],
                                payload[7], payload[4], payload[8], payload[5],
                                payload[9], payload[10], payload[11], payload[6],
                            )
                            item = self._row_to_dict(standard_row)
                            item["content_chars"] = int(payload[12] or 0)
                            item["content_window_offset"] = window_start
                            item["requested_content_offset"] = bounded_offset
                            if boundary_scan.get("boundary_redacted"):
                                item["boundary_redacted"] = True
                                item["boundary_pending"] = bool(
                                    boundary_scan.get("boundary_pending")
                                )
                                item["boundary_next_content_offset"] = scan_offset
                                item["boundary_safe_prefix"] = str(
                                    boundary_scan.get("boundary_safe_prefix") or ""
                                )[:512]
                                item["boundary_checkpoint_offset"] = int(
                                    checkpoint_offset
                                )
                            item["tool_calls_chars"] = int(payload[13] or 0)
                            item["tool_calls_omitted"] = (
                                int(payload[13] or 0) > _EXPAND_TOOL_CALLS_MAX_CHARS
                            )
                            if payload[14] not in {"text", "null"} or payload[15] not in {"text", "null"}:
                                result = {"status": "invalid_payload"}
                            else:
                                result = {"status": "ok", "message": item}
                self._conn.execute(f"RELEASE {savepoint}")
                return result
            except Exception:
                if savepoint_started:
                    # Cleanup must not itself be aborted by the expired work
                    # budget; the context finally restores the caller handler.
                    self._conn.set_progress_handler(None, 0)
                    self._conn.execute(f"ROLLBACK TO {savepoint}")
                    self._conn.execute(f"RELEASE {savepoint}")
                raise
            finally:
                try:
                    if content_reader is not None:
                        content_reader.close()
                finally:
                    progress_budget.__exit__(None, None, None)

    def get_batch(self, store_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve multiple messages by store_id in a single query.

        Returns a dict mapping store_id → message dict.
        """
        if not store_ids:
            return {}
        placeholders = ",".join("?" for _ in store_ids)
        rows = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchall()
        return {row[0]: self._row_to_dict(row) for row in rows}

    def get_range(self, session_id: str, start_id: int = 0,
                  end_id: int | None = None,
                  limit: int = 1000,
                  conversation_id: str | None = None) -> List[Dict[str, Any]]:
        """Get messages in a store_id range for a session."""
        where = ["session_id = ?", "store_id >= ?"]
        args: list[Any] = [session_id, start_id]
        conversation_clause, conversation_args = _conversation_filter_clause("conversation_id", conversation_id)
        if conversation_clause:
            where.append(conversation_clause)
            args.extend(conversation_args)
        if end_id is not None:
            where.append("store_id <= ?")
            args.append(end_id)
        args.append(limit)
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE {' AND '.join(where)}
               ORDER BY store_id LIMIT ?""",
            args,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _session_load_where(
        self,
        session_id: str,
        *,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> tuple[list[str], list[Any]]:
        where = ["session_id = ?"]
        args: list[Any] = [session_id]
        if roles:
            placeholders = ",".join("?" for _ in roles)
            where.append(f"role IN ({placeholders})")
            args.extend(roles)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        return where, args

    def count_session_load_messages(
        self,
        session_id: str,
        *,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> int:
        """Count messages matching the lcm_load_session filter contract."""
        where, args = self._session_load_where(
            session_id,
            roles=roles,
            time_from=time_from,
            time_to=time_to,
        )
        return int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE {' AND '.join(where)}",
                args,
            ).fetchone()[0]
        )

    def load_session_page(
        self,
        session_id: str,
        *,
        after_store_id: int = 0,
        limit: int = 100,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
        max_content_chars: int = 20_000,
        content_lookahead_chars: int = 0,
        max_serialized_bytes: int = _LOAD_SESSION_MAX_PAGE_MATERIALIZED_BYTES,
        max_row_serialized_bytes: int = _LOAD_SESSION_MAX_ROW_MATERIALIZED_BYTES,
    ) -> List[Dict[str, Any]]:
        """Load a page from one immutable read snapshot.

        The metadata preflight and bounded payload reads intentionally share a
        SQLite read transaction.  The payload query also reapplies every
        caller filter, so a concurrent reassignment or equal-length rewrite
        can neither change the authorized snapshot nor bypass its predicates.
        """
        savepoint = "lcm_load_session_snapshot"
        with self._write_lock:
            self._conn.execute(f"SAVEPOINT {savepoint}")
            try:
                count_where, count_args = self._session_load_where(
                    session_id,
                    roles=roles,
                    time_from=time_from,
                    time_to=time_to,
                )
                total_messages = int(
                    self._conn.execute(
                        f"SELECT COUNT(*) FROM messages WHERE {' AND '.join(count_where)}",
                        count_args,
                    ).fetchone()[0]
                )
                result = self._load_session_page_locked(
                    session_id,
                    after_store_id=after_store_id,
                    limit=limit,
                    roles=roles,
                    time_from=time_from,
                    time_to=time_to,
                    max_content_chars=max_content_chars,
                    content_lookahead_chars=content_lookahead_chars,
                    max_serialized_bytes=max_serialized_bytes,
                    max_row_serialized_bytes=max_row_serialized_bytes,
                )
                result.total_messages = total_messages
                self._conn.execute(f"RELEASE {savepoint}")
                return result
            except Exception:
                self._conn.execute(f"ROLLBACK TO {savepoint}")
                self._conn.execute(f"RELEASE {savepoint}")
                raise

    def _load_session_page_locked(
        self,
        session_id: str,
        *,
        after_store_id: int = 0,
        limit: int = 100,
        roles: list[str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
        max_content_chars: int = 20_000,
        content_lookahead_chars: int = 0,
        max_serialized_bytes: int = _LOAD_SESSION_MAX_PAGE_MATERIALIZED_BYTES,
        max_row_serialized_bytes: int = _LOAD_SESSION_MAX_ROW_MATERIALIZED_BYTES,
    ) -> List[Dict[str, Any]]:
        """Load one ordered raw-message page for a session.

        ``after_store_id`` is exclusive so callers can use the previous page's
        ``next_cursor`` without duplicating the cursor row.
        """
        where, args = self._session_load_where(
            session_id,
            roles=roles,
            time_from=time_from,
            time_to=time_to,
        )
        where.append("store_id > ?")
        args.append(after_store_id)
        result = SessionLoadPage()
        materialized_bytes = 0
        last_store_id = int(after_store_id)
        text_columns = (
            "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "conversation_id",
        )
        content_scan_chars = max(1, int(max_content_chars)) + max(
            0, int(content_lookahead_chars)
        )
        text_limits = (
            _LOAD_SESSION_MAX_SESSION_TEXT_BYTES,
            _LOAD_SESSION_MAX_SCALAR_TEXT_BYTES,
            _LOAD_SESSION_MAX_ROLE_TEXT_BYTES,
            content_scan_chars * 4,
            _LOAD_SESSION_MAX_SCALAR_TEXT_BYTES,
            _LOAD_SESSION_MAX_TOOL_CALLS_ENCODED_BYTES,
            _LOAD_SESSION_MAX_SCALAR_TEXT_BYTES,
            _LOAD_SESSION_MAX_SESSION_TEXT_BYTES,
        )
        length_exprs = [
            f"COALESCE(length(CAST({column} AS BLOB)), 0)" for column in text_columns
        ]
        char_length_exprs = [
            f"COALESCE(length(CAST({column} AS TEXT)), 0)" for column in text_columns
        ]
        while len(result) < int(limit):
            metadata_where = [*where[:-1], "store_id > ?"]
            metadata_args = [*args[:-1], last_store_id]
            metadata = self._conn.execute(
                f"""SELECT
                           CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                           CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                           CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                           CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                           {', '.join(length_exprs)},
                           {', '.join(char_length_exprs)},
                           {', '.join(f"typeof({column}) IN ('text', 'null')" for column in text_columns)}
                    FROM messages WHERE {' AND '.join(metadata_where)}
                    ORDER BY store_id LIMIT 1""",
                metadata_args,
            ).fetchone()
            if metadata is None:
                break
            if any(value is None for value in metadata[:4]) or not all(
                bool(value) for value in metadata[20:28]
            ):
                raise ValueError("invalid stored message scalar type")
            store_id = int(metadata[0])
            encoded_lengths = [int(value or 0) for value in metadata[4:12]]
            char_lengths = [int(value or 0) for value in metadata[12:20]]
            invalid_scalar_text = any(
                length > limit_value
                for index, (length, limit_value) in enumerate(zip(encoded_lengths, text_limits))
                if text_columns[index] not in {"content", "tool_calls"}
            )
            if invalid_scalar_text:
                raise ValueError("invalid stored message scalar text bound")
            bounded_lengths = [
                min(length, limit_value)
                for length, limit_value in zip(encoded_lengths, text_limits)
            ]
            tool_calls_omitted = (
                encoded_lengths[5] > _LOAD_SESSION_MAX_TOOL_CALLS_ENCODED_BYTES
            )
            if tool_calls_omitted:
                bounded_lengths[5] = 0
            row_upper_bound = sum(bounded_lengths) + 128
            if row_upper_bound > int(max_row_serialized_bytes):
                # A JSON tool_calls value cannot be safely substring-truncated.
                # Omit it before reserving the rest of the per-row budget.
                fixed_without_content = row_upper_bound - bounded_lengths[3]
                if (
                    bounded_lengths[5]
                    and fixed_without_content > int(max_row_serialized_bytes)
                ):
                    row_upper_bound -= bounded_lengths[5]
                    bounded_lengths[5] = 0
                    tool_calls_omitted = True
                # Content is intentionally sliceable; reserve the remaining
                # row cap for it after all non-content scalar fields.
                fixed = row_upper_bound - bounded_lengths[3]
                bounded_lengths[3] = max(0, int(max_row_serialized_bytes) - fixed)
                row_upper_bound = fixed + bounded_lengths[3]
            if row_upper_bound > int(max_row_serialized_bytes):
                result.budget_exhausted = True
                break
            if materialized_bytes + row_upper_bound > int(max_serialized_bytes):
                result.budget_exhausted = True
                break
            content_char_cap = min(
                content_scan_chars,
                (
                    char_lengths[3]
                    if bounded_lengths[3] >= encoded_lengths[3]
                    else max(0, bounded_lengths[3] // 4)
                ),
            )
            bounded_sql: dict[str, str] = {}
            for index, column in enumerate(text_columns):
                if column == "tool_calls" and tool_calls_omitted:
                    bounded_sql[column] = "NULL"
                    continue
                char_cap = content_char_cap if column == "content" else char_lengths[index]
                length_guard = ""
                if column != "content":
                    length_guard = f" AND {length_exprs[index]} <= {text_limits[index]}"
                bounded_sql[column] = (
                    f"CASE WHEN {column} IS NULL THEN NULL ELSE "
                    f"CASE WHEN typeof({column}) = 'text'{length_guard} THEN "
                    f"substr(CAST({column} AS TEXT), 1, {int(char_cap)}) "
                    "ELSE NULL END END"
                )
            payload_where = [*where[:-1], "store_id = ?"]
            payload_args = [*args[:-1], store_id]
            payload_where.extend(
                [
                    "timestamp IS ?",
                    "token_estimate IS ?",
                    "pinned IS ?",
                    *(f"{expression} = ?" for expression in length_exprs),
                    *(f"{expression} = ?" for expression in char_length_exprs),
                    *(f"typeof({column}) IN ('text', 'null')" for column in text_columns),
                ]
            )
            payload_args.extend(
                [
                    metadata[1],
                    metadata[2],
                    metadata[3],
                    *encoded_lengths,
                    *char_lengths,
                ]
            )
            row = self._conn.execute(
                f"""SELECT
                           CASE WHEN typeof(store_id) = 'integer' THEN store_id END,
                           {bounded_sql['session_id']}, {bounded_sql['source']},
                           {bounded_sql['role']}, {bounded_sql['content']},
                           {bounded_sql['tool_call_id']}, {bounded_sql['tool_calls']},
                           {bounded_sql['tool_name']},
                           CASE WHEN typeof(timestamp) IN ('integer', 'real') THEN timestamp END,
                           CASE WHEN typeof(token_estimate) = 'integer' THEN token_estimate END,
                           CASE WHEN typeof(pinned) = 'integer' THEN pinned END,
                           {bounded_sql['conversation_id']},
                           {', '.join(length_exprs)},
                           {', '.join(f"typeof({column}) IN ('text', 'null')" for column in text_columns)}
                    FROM messages WHERE {' AND '.join(payload_where)} LIMIT 1""",
                payload_args,
            ).fetchone()
            if row is None or any(row[index] is None for index in (0, 8, 9, 10)):
                raise ValueError("invalid stored message scalar type")
            if [int(value or 0) for value in row[12:20]] != encoded_lengths or not all(
                bool(value) for value in row[20:28]
            ):
                raise ValueError("stored message changed during bounded load")
            item = self._row_to_dict(row[:12])
            item["content_chars"] = char_lengths[3]
            encoded_bytes = encoded_lengths[5]
            item["tool_calls_encoded_bytes"] = encoded_bytes
            item["tool_calls_encoded_too_large"] = (
                tool_calls_omitted
            )
            result.append(item)
            materialized_bytes += row_upper_bound
            last_store_id = store_id
        return result

    def get_session_messages(self, session_id: str,
                             limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by store_id."""
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE session_id = ?
               ORDER BY store_id LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_messages_after(self, session_id: str,
                                   after_store_id: int = 0,
                                   limit: int = 10000) -> List[Dict[str, Any]]:
        """Get session messages after a store_id, ordered by store_id."""
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
               WHERE session_id = ? AND store_id > ?
               ORDER BY store_id LIMIT ?""",
            (session_id, after_store_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_tail(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get the latest messages for a session, returned in store order."""
        if limit <= 0:
            return []
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS}
               FROM (
                   SELECT {_MESSAGE_SELECT_COLUMNS}
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY store_id DESC
                   LIMIT ?
               )
               ORDER BY store_id""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_count(self, session_id: str) -> int:
        """Count messages in a session."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_token_total(self, session_id: str) -> int:
        """Sum of token estimates for a session."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_source_stats(self, session_id: str | None = None) -> Dict[str, int]:
        """Return raw source-bucket counts for diagnostics."""
        where = ""
        args: list[Any] = []
        if session_id is not None:
            where = "WHERE session_id = ?"
            args.append(session_id)

        legacy_blank_clause = _legacy_blank_source_clause("source")
        query = f"""
            SELECT COUNT(*) AS messages_total,
                   COALESCE(SUM(CASE WHEN source = ? THEN 1 ELSE 0 END), 0) AS normalized_unknown_messages,
                   COALESCE(SUM(CASE WHEN {legacy_blank_clause} THEN 1 ELSE 0 END), 0) AS legacy_blank_source_messages,
                   COALESCE(SUM(CASE WHEN NOT {legacy_blank_clause} AND source != ? THEN 1 ELSE 0 END), 0) AS attributed_messages
            FROM messages
            {where}
            """
        query_args: list[Any] = [_UNKNOWN_SOURCE, _UNKNOWN_SOURCE, *args]
        row = self._conn.execute(query, query_args).fetchone()

        messages_total = int(row[0] or 0) if row else 0
        normalized_unknown = int(row[1] or 0) if row else 0
        legacy_blank = int(row[2] or 0) if row else 0
        attributed = int(row[3] or 0) if row else 0
        return {
            "messages_total": messages_total,
            "attributed_messages": attributed,
            "normalized_unknown_messages": normalized_unknown,
            "legacy_blank_source_messages": legacy_blank,
            "effective_unknown_messages": normalized_unknown + legacy_blank,
        }

    def scan_session_cleanup_stats(self) -> List[tuple]:
        """Per-session ``(session_id, message_count, token_total, node_count)``
        rows across messages and summary nodes, for ``/lcm doctor clean``
        candidate scanning. Callers own the pattern/protection policy."""
        return self._conn.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM messages
                UNION
                SELECT session_id FROM summary_nodes
            ),
            message_stats AS (
                SELECT session_id,
                       COUNT(*) AS message_count,
                       COALESCE(SUM(token_estimate), 0) AS token_total
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id, COUNT(*) AS node_count
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            ORDER BY s.session_id
            """
        ).fetchall()

    def scan_session_retention_stats(self, session_id: str) -> List[tuple]:
        """Per-session activity/token stats for one session (messages + summary
        nodes), for ``/lcm doctor retention`` scanning. Callers own the
        staleness/protection policy."""
        return self._conn.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM messages
                UNION
                SELECT session_id FROM summary_nodes
            ),
            message_stats AS (
                SELECT session_id,
                       COUNT(*) AS message_count,
                       COALESCE(SUM(token_estimate), 0) AS token_total,
                       MIN(timestamp) AS first_message_at,
                       MAX(timestamp) AS last_message_at
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id,
                       COUNT(*) AS node_count,
                       COALESCE(SUM(token_count), 0) AS node_token_total,
                       MIN(COALESCE(earliest_at, created_at)) AS first_node_at,
                       MAX(COALESCE(latest_at, created_at)) AS last_node_at
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count,
                   COALESCE(n.node_token_total, 0) AS node_token_total,
                   m.first_message_at,
                   m.last_message_at,
                   n.first_node_at,
                   n.last_node_at
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            WHERE s.session_id = ?
            ORDER BY s.session_id
            """,
            (session_id,),
        ).fetchall()

    def get_source_normalization_plan(self) -> Dict[str, Any]:
        """Return a dry-run plan for normalizing legacy blank source values."""
        stats_before = self.get_source_stats()
        blank_clause = _legacy_blank_source_clause("source")
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS would_update_messages,
                   COUNT(DISTINCT session_id) AS affected_sessions
            FROM messages
            WHERE {blank_clause}
            """
        ).fetchone()
        would_update = int(row[0] or 0) if row else 0
        affected_sessions = int(row[1] or 0) if row else 0
        return {
            "target_source": _UNKNOWN_SOURCE,
            "would_update_messages": would_update,
            "affected_sessions": affected_sessions,
            "stats_before": stats_before,
        }

    def normalize_legacy_blank_sources(self) -> Dict[str, Any]:
        """Normalize legacy NULL/blank source rows to the explicit unknown bucket."""
        stats_before = self.get_source_stats()
        blank_clause = _legacy_blank_source_clause("source")
        with self._write_lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE messages SET source = ? WHERE {blank_clause}",
                (_UNKNOWN_SOURCE,),
            )
        updated = cur.rowcount if cur.rowcount is not None else 0
        stats_after = self.get_source_stats()
        return {
            "target_source": _UNKNOWN_SOURCE,
            "updated_messages": int(updated),
            "stats_before": stats_before,
            "stats_after": stats_after,
        }

    def get_time_bounds(self, store_ids: List[int]) -> tuple[float | None, float | None]:
        if not store_ids:
            return None, None
        placeholders = ",".join("?" * len(store_ids))
        row = self._conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    # -- Metadata key/value JSON --------------------------------------------

    def read_metadata_json(self, key: str) -> Any:
        """Return the JSON-decoded value stored under ``key`` in the metadata table.

        Returns ``None`` when the connection is closed, the key is absent, or the
        stored value is empty. JSON decoding is deliberately *not* wrapped: a
        malformed value raises, so callers keep the ``try``/``except`` scoping
        that decides whether one bad key aborts a multi-key load or is skipped.
        Reads are unlocked, matching the store's other read paths (``_write_lock``
        guards writes only).
        """
        conn = self._conn
        if conn is None:
            return None
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(str(row[0]))

    def write_metadata_json(
        self,
        keys: list[str],
        serialized: str,
        *,
        skip_unchanged: bool = False,
    ) -> bool:
        """Write the pre-serialized JSON string ``serialized`` to every key in ``keys``.

        Serialization stays with the caller so it keeps control of ``sort_keys``
        and payload shape. Runs under the store write lock and issues at most one
        commit. With ``skip_unchanged=True`` a key already holding ``serialized``
        is left untouched and the commit is skipped entirely when nothing changed
        -- the ingest-hot-path optimization used by the placeholder count/ordinal
        writers. Returns ``True`` if any key was written.
        """
        conn = self._conn
        if conn is None:
            return False
        wrote = False
        with self._write_lock:
            for key in keys:
                if skip_unchanged:
                    existing = conn.execute(
                        "SELECT value FROM metadata WHERE key = ?", (key,)
                    ).fetchone()
                    if existing is not None and existing[0] == serialized:
                        continue
                conn.execute(
                    """
                    INSERT INTO metadata(key, value)
                    VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, serialized),
                )
                wrote = True
            if wrote:
                conn.commit()
        return wrote

    # -- Compaction telemetry ------------------------------------------------

    @staticmethod
    def _compaction_telemetry_key(conversation_id: str) -> str:
        return f"compaction_telemetry:{conversation_id}"

    def read_compaction_telemetry(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Return the persisted per-conversation compaction-telemetry record, or None.

        Best-effort: a closed connection, missing/empty row, or malformed JSON all
        yield None. Telemetry is diagnostic and must never block a turn. Reads are
        unlocked, matching the store's other read paths.
        """
        if not conversation_id:
            return None
        try:
            data = self.read_metadata_json(self._compaction_telemetry_key(conversation_id))
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def write_compaction_telemetry(self, conversation_id: str, record: Dict[str, Any]) -> None:
        """Upsert the per-conversation compaction-telemetry record.

        Stored as a single JSON row in the existing metadata table (no dedicated
        schema, no version bump) under the store write lock. The write -- and its
        commit -- is skipped when the serialized payload is unchanged so idle
        turns do not churn the row.
        """
        if not conversation_id:
            return
        serialized = json.dumps(record, sort_keys=True)
        key = self._compaction_telemetry_key(conversation_id)
        self.write_metadata_json([key], serialized, skip_unchanged=True)

    # -- Search -------------------------------------------------------------

    def _grep_query_rows(
        self,
        sql: str,
        args: list[Any],
        *,
        row_limit: int,
        deadline: float | None,
        materialization_budget: dict[str, int] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Fetch a hard-bounded grep batch with an optional SQLite deadline."""
        if row_limit <= 0 or (deadline is not None and time.monotonic() >= deadline):
            return []

        def interrupt_after_deadline() -> int:
            return int(deadline is not None and time.monotonic() >= deadline)

        rows: list[tuple[Any, ...]] = []
        with self._write_lock:
            if deadline is not None:
                self._conn.set_progress_handler(interrupt_after_deadline, 1_000)
            try:
                cursor = self._conn.execute(sql, args)
                while len(rows) < row_limit:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    row = cursor.fetchone()
                    if row is None:
                        break
                    row_bytes = sum(
                        len(value.encode("utf-8", errors="surrogatepass"))
                        for value in row
                        if isinstance(value, str)
                    )
                    if materialization_budget is not None:
                        if row_bytes > int(materialization_budget.get("remaining", 0)):
                            materialization_budget["exhausted"] = 1
                            break
                        materialization_budget["remaining"] = (
                            int(materialization_budget.get("remaining", 0)) - row_bytes
                        )
                    rows.append(tuple(row))
            except sqlite3.OperationalError as exc:
                if deadline is None or "interrupted" not in str(exc).lower():
                    raise
            finally:
                if deadline is not None:
                    self._conn.set_progress_handler(None, 0)
        return rows

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None,
               conversation_id: str | None = None,
               role: str | None = None,
               time_from: float | None = None,
               time_to: float | None = None,
               bounded_output: bool = False,
               max_candidate_rows: int | None = None,
               deadline: float | None = None,
               max_materialized_bytes: int | None = None) -> List[Dict[str, Any]]:
        """FTS5 search across raw messages.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` limits which raw rows inside those sessions are eligible
        - ``source='unknown'`` means the explicit unknown-source bucket, with
          legacy blank-source rows treated as equivalent for back-compat
        - ``conversation_id`` limits rows to one gateway conversation/session key
        """
        safe_query = sanitize_fts5_query(query)
        materialization_budget = (
            {"remaining": max(0, int(max_materialized_bytes)), "exhausted": 0}
            if max_materialized_bytes is not None
            else None
        )
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if requires_like_fallback(query):
            return self._search_like(
                query,
                session_id=session_id,
                limit=limit,
                sort=sort,
                source=source,
                conversation_id=conversation_id,
                role=role,
                time_from=time_from,
                time_to=time_to,
                bounded_output=bounded_output,
                max_candidate_rows=max_candidate_rows,
                deadline=deadline,
                max_materialized_bytes=max_materialized_bytes,
            )

        order_by = _build_search_order_by(
            sort,
            "m.timestamp",
            _MESSAGE_ROLE_BIAS_SQL,
        )
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        if max_candidate_rows is not None:
            candidate_cap = min(candidate_cap, max(0, int(max_candidate_rows)))
        fetch_limit = min(fetch_limit, candidate_cap)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 3e-7
        source_clause, source_args = _source_filter_clause("m.source", source)
        conversation_clause, conversation_args = _conversation_filter_clause("m.conversation_id", conversation_id)
        offset = 0
        scanned_rows = 0
        results: list[Dict[str, Any]] = []
        while True:
            try:
                where = ["messages_fts MATCH ?"]
                args: list[Any] = [safe_query]
                if session_id is not None:
                    where.append("m.session_id = ?")
                    args.append(session_id)
                if source_clause:
                    where.append(source_clause)
                    args.extend(source_args)
                if conversation_clause:
                    where.append(conversation_clause)
                    args.extend(conversation_args)
                if role is not None:
                    where.append("m.role = ?")
                    args.append(role)
                if time_from is not None:
                    where.append("m.timestamp >= ?")
                    args.append(time_from)
                if time_to is not None:
                    where.append("m.timestamp <= ?")
                    args.append(time_to)
                args.extend([fetch_limit, offset])
                if bounded_output:
                    select_columns, metadata_columns, select_args = (
                        _grep_bounded_search_projection(terms, alias="m")
                    )
                    select_sql = (
                        f"{select_columns}, rank AS search_rank, '' AS snippet, "
                        f"{metadata_columns}"
                    )
                else:
                    select_args = []
                    select_sql = (
                        "m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id, "
                        "m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned, m.conversation_id, "
                        "rank AS search_rank, "
                        "snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet"
                    )
                rows = self._grep_query_rows(
                    f"""SELECT {select_sql}
                       FROM messages_fts fts
                       JOIN messages m ON m.store_id = fts.rowid
                       WHERE {' AND '.join(where)}
                       ORDER BY {order_by} LIMIT ? OFFSET ?""",
                    [*select_args, *args],
                    row_limit=min(fetch_limit, max(0, candidate_cap - scanned_rows)),
                    deadline=deadline,
                    materialization_budget=materialization_budget,
                )
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS message search failed, falling back to LIKE: %s", exc)
                return self._search_like(
                    query,
                    session_id=session_id,
                    limit=limit,
                    sort=sort,
                    source=source,
                    conversation_id=conversation_id,
                    role=role,
                    time_from=time_from,
                    time_to=time_to,
                    bounded_output=bounded_output,
                    max_candidate_rows=max_candidate_rows,
                    deadline=deadline,
                    max_materialized_bytes=(
                        int(materialization_budget.get("remaining", 0))
                        if materialization_budget is not None
                        else None
                    ),
                )

            raw_primary_values: list[float] = []
            for r in rows:
                d = self._row_to_dict(r)
                base_columns = 12
                d["search_rank"] = r[base_columns] if len(r) > base_columns else None
                d["snippet"] = r[base_columns + 1] if len(r) > (base_columns + 1) else ""
                if bounded_output:
                    metadata_offset = base_columns + 2
                    d["content_bytes"] = int(r[metadata_offset] or 0)
                    d["content_chars"] = int(r[metadata_offset + 1] or 0)
                    d["tool_calls_bytes"] = int(r[metadata_offset + 2] or 0)
                    d["_grep_window_start"] = int(r[metadata_offset + 3] or 1)
                    d["_grep_bounded"] = True
                    d["_grep_highlight_matches"] = True
                    if r[metadata_offset + 4] != "text" or r[metadata_offset + 5] not in {"text", "null"}:
                        continue
                    d["snippet"] = d.get("content") or ""
                d["_directness_score"] = _message_directness_score(d.get("role"), d.get("content"), terms, phrases)
                if apply_directness_adjustment and d["search_rank"] is not None:
                    rank_adjustment = max(float(d["_directness_score"]), 0.0)
                    d["search_rank"] = float(d["search_rank"]) - (rank_adjustment * 3e-7)
                raw_primary_values.append(_fts_primary_value(d, sort))
                results.append(d)
            results.sort(key=lambda result: _fts_result_sort_key(result, sort))

            if not apply_directness_adjustment or len(rows) < fetch_limit or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = raw_primary_values[-1]
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            if scanned_rows >= candidate_cap:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None,
                     conversation_id: str | None = None,
                     role: str | None = None,
                     time_from: float | None = None,
                     time_to: float | None = None,
                     bounded_output: bool = False,
                     max_candidate_rows: int | None = None,
                     deadline: float | None = None,
                     max_materialized_bytes: int | None = None) -> List[Dict[str, Any]]:
        safe_query = sanitize_fts5_query(query)
        materialization_budget = (
            {"remaining": max(0, int(max_materialized_bytes)), "exhausted": 0}
            if max_materialized_bytes is not None
            else None
        )
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["content IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        source_clause, source_args = _source_filter_clause("source", source)
        if source_clause:
            where.append(source_clause)
            args.extend(source_args)
        conversation_clause, conversation_args = _conversation_filter_clause("conversation_id", conversation_id)
        if conversation_clause:
            where.append(conversation_clause)
            args.extend(conversation_args)
        if role is not None:
            where.append("role = ?")
            args.append(role)
        if time_from is not None:
            where.append("timestamp >= ?")
            args.append(time_from)
        if time_to is not None:
            where.append("timestamp <= ?")
            args.append(time_to)
        like_clauses = []
        for term in terms:
            like_clauses.append("content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        normalized_sort = normalize_search_sort(sort)
        results: List[Dict[str, Any]] = []
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        order_by = ""
        order_args: list[Any] = []
        role_bias = "CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"

        def count_expr(term: str) -> tuple[str, list[Any]]:
            return (
                "((LENGTH(LOWER(content)) - LENGTH(REPLACE(LOWER(content), LOWER(?), ''))) "
                "/ NULLIF(LENGTH(?), 0))",
                [term, term],
            )

        if normalized_sort == "recency":
            score_exprs: list[str] = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"

            def build_unique_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(f"CASE WHEN ({expr}) > 0 THEN 1 ELSE 0 END")
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            def build_total_exprs(selected_terms: list[str]) -> tuple[str, list[Any]]:
                parts: list[str] = []
                expr_args: list[Any] = []
                for selected_term in selected_terms:
                    expr, args_for_expr = count_expr(selected_term)
                    parts.append(expr)
                    expr_args.extend(args_for_expr)
                return (" + ".join(parts) if parts else "0", expr_args)

            directness_args: list[Any] = []
            unique_score_expr, expr_args = build_unique_exprs(terms)
            directness_args.extend(expr_args)
            normalized_phrases = {(phrase or "").strip().lower() for phrase in phrases if (phrase or "").strip()}
            if phrases:
                phrase_hit_exprs: list[str] = []
                for phrase in phrases:
                    phrase_hit_exprs.append("CASE WHEN INSTR(LOWER(content), LOWER(?)) > 0 THEN 1 ELSE 0 END")
                    directness_args.append(phrase)
                phrase_hit_expr = " + ".join(phrase_hit_exprs) if phrase_hit_exprs else "0"
                non_phrase_terms = [term for term in terms if term.strip().lower() not in normalized_phrases]
                non_phrase_total_expr, expr_args = build_total_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                non_phrase_unique_expr, expr_args = build_unique_exprs(non_phrase_terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({non_phrase_total_expr}) - ({non_phrase_unique_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) + (({phrase_hit_expr}) * 8.0) - MIN(({repetition_expr}), 6)"
            else:
                total_repetition_expr, expr_args = build_total_exprs(terms)
                directness_args.extend(expr_args)
                unique_repetition_expr, expr_args = build_unique_exprs(terms)
                directness_args.extend(expr_args)
                repetition_expr = f"MAX(({total_repetition_expr}) - ({unique_repetition_expr}), 0)"
                directness_expr = f"(({unique_score_expr}) * 5.0) - MIN(({repetition_expr}), 6)"
            order_args.extend(directness_args)
            order_by = (
                f"ORDER BY timestamp DESC, {role_bias} ASC, ({score_expr}) DESC, "
                f"({directness_expr}) DESC, store_id DESC"
            )

        def add_rows(rows: list[sqlite3.Row]) -> None:
            for row in rows:
                result = self._row_to_dict(row)
                if bounded_output:
                    result["content_bytes"] = int(row[12] or 0)
                    result["content_chars"] = int(row[13] or 0)
                    result["tool_calls_bytes"] = int(row[14] or 0)
                    result["_grep_window_start"] = int(row[15] or 1)
                    result["_grep_bounded"] = True
                    result["_grep_highlight_matches"] = False
                    if row[16] != "text" or row[17] not in {"text", "null"}:
                        continue
                content = result.get("content") or ""
                score = sum(
                    min(count_term_matches(content, term), 1) if collapse_risky_repeats else count_term_matches(content, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                result["search_rank"] = -float(score)
                result["snippet"] = content if bounded_output else build_snippet(content, terms)
                result["_fallback_score"] = float(score)
                result["_directness_score"] = _message_directness_score(result.get("role"), content, terms, phrases)
                results.append(result)

        if normalized_sort == "recency":
            if bounded_output:
                select_columns, metadata_columns, bounded_select_args = (
                    _grep_bounded_search_projection(terms)
                )
                message_select = f"{select_columns}, {metadata_columns}"
            else:
                bounded_select_args = []
                message_select = _MESSAGE_SELECT_COLUMNS
            candidate_cap = compute_search_candidate_cap(limit)
            hard_candidate_cap = candidate_cap
            if max_candidate_rows is not None:
                hard_candidate_cap = max(0, int(max_candidate_rows))
                candidate_cap = min(candidate_cap, hard_candidate_cap)
            offset = 0
            scanned_rows = 0
            while True:
                batch_limit = min(fetch_limit, candidate_cap - scanned_rows)
                if batch_limit <= 0:
                    break
                rows = self._grep_query_rows(
                    f"""SELECT {message_select}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*bounded_select_args, *base_args, *order_args, batch_limit, offset],
                    row_limit=batch_limit,
                    deadline=deadline,
                    materialization_budget=materialization_budget,
                )
                scanned_rows += len(rows)
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break
                if scanned_rows >= candidate_cap:
                    boundary_timestamp = rows[-1][8]
                    boundary_role_bias = _message_role_bias(rows[-1][3])
                    while True:
                        # ``candidate_cap`` is the ordinary ranking window. A
                        # recency/role tie may extend beyond it so Python can
                        # apply JSON/directness penalties, but never beyond the
                        # caller's operation-wide hard row reservation.
                        tie_budget = hard_candidate_cap - scanned_rows
                        if tie_budget <= 0:
                            break
                        tie_rows = self._grep_query_rows(
                            f"""SELECT {message_select}
                                FROM messages
                                WHERE {' AND '.join(where)}
                                {order_by}
                                LIMIT ? OFFSET ?""",
                            [*bounded_select_args, *base_args, *order_args, min(fetch_limit, tie_budget), offset],
                            row_limit=min(fetch_limit, tie_budget),
                            deadline=deadline,
                            materialization_budget=materialization_budget,
                        )
                        if not tie_rows:
                            break
                        scanned_rows += len(tie_rows)
                        matching_tie_rows = []
                        reached_next_primary_group = False
                        for tie_row in tie_rows:
                            if tie_row[8] == boundary_timestamp and _message_role_bias(tie_row[3]) == boundary_role_bias:
                                matching_tie_rows.append(tie_row)
                            else:
                                reached_next_primary_group = True
                                break
                        add_rows(matching_tie_rows)
                        if reached_next_primary_group or len(tie_rows) < fetch_limit:
                            break
                        offset += len(tie_rows)
                    break
        else:
            # Deterministic relevance/hybrid candidate scan for LIKE fallback.
            # Apply the same coarse score/directness ordering before the hard
            # candidate cap that Python uses below; otherwise a recent-biased
            # window can exclude older but materially better relevance matches.
            score_exprs: list[str] = []
            order_args = []
            for term in terms:
                if collapse_risky_repeats:
                    score_exprs.append("CASE WHEN content LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END")
                    order_args.append(f"%{escape_like(term)}%")
                else:
                    expr, expr_args = count_expr(term)
                    score_exprs.append(expr)
                    order_args.extend(expr_args)
            score_expr = " + ".join(score_exprs) if score_exprs else "0"
            exact_query = (query or "").strip()
            exact_expr = "CASE WHEN LOWER(content) = LOWER(?) THEN 1 ELSE 0 END" if exact_query else "0"
            exact_args: list[Any] = [exact_query] if exact_query else []
            directness_expr = "0.0 + 0"

            if normalized_sort == "hybrid":
                primary_expr = (
                    f"(({score_expr}) / (1 + (MAX(0.0, "
                    f"((strftime('%s','now') - timestamp) / 3600.0)) * {AGE_DECAY_RATE})))"
                )
            else:
                primary_expr = f"({score_expr})"

            order_by = (
                f"ORDER BY {primary_expr} DESC, ({exact_expr}) DESC, ({directness_expr}) DESC, "
                f"{role_bias} ASC, timestamp DESC, store_id DESC"
            )
            candidate_cap = compute_search_candidate_cap(limit)
            if max_candidate_rows is not None:
                candidate_cap = min(candidate_cap, max(0, int(max_candidate_rows)))
            if bounded_output:
                select_columns, metadata_columns, bounded_select_args = (
                    _grep_bounded_search_projection(terms)
                )
                message_select = f"{select_columns}, {metadata_columns}"
            else:
                bounded_select_args = []
                message_select = _MESSAGE_SELECT_COLUMNS
            offset = 0
            while offset < candidate_cap:
                batch_limit = min(fetch_limit, candidate_cap - offset)
                rows = self._grep_query_rows(
                    f"""SELECT {message_select}
                        FROM messages
                        WHERE {' AND '.join(where)}
                        {order_by}
                        LIMIT ? OFFSET ?""",
                    [*bounded_select_args, *base_args, *order_args, *exact_args, batch_limit, offset],
                    row_limit=batch_limit,
                    deadline=deadline,
                    materialization_budget=materialization_budget,
                )
                if not rows:
                    break
                add_rows(rows)
                offset += len(rows)
                if len(rows) < batch_limit:
                    break

        results.sort(key=lambda result: _fallback_result_sort_key(result, sort))
        for result in results:
            result.pop("_fallback_score", None)
        return results[:limit]

    # -- Helpers ------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a sqlite3 row to a dict."""
        if row is None:
            return {}
        cols = [
            "store_id", "session_id", "source", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned", "conversation_id",
        ]
        d = dict(zip(cols, row[:len(cols)]))
        d["source"] = _normalize_source_value(d.get("source"))
        d["conversation_id"] = _normalize_conversation_id_value(d.get("conversation_id"))
        # Deserialize tool_calls JSON
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def to_openai_msg(self, stored: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a stored message back to OpenAI format."""
        msg: Dict[str, Any] = {"role": stored["role"]}
        if stored.get("content") is not None:
            msg["content"] = stored["content"]
        if stored.get("tool_calls"):
            msg["tool_calls"] = stored["tool_calls"]
        if stored.get("tool_call_id"):
            msg["tool_call_id"] = stored["tool_call_id"]
        if stored.get("tool_name"):
            msg["name"] = stored["tool_name"]
        return msg

    # -- Connection access --------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live SQLite connection, or ``None`` once :meth:`close` has run.

        Exposed for read-oriented diagnostics and inspection -- integrity /
        quick checks, FTS sync counts, schema health -- that need ad-hoc
        queries the store does not wrap in a purpose-built method. Callers must
        treat it as read-only and tolerate ``None``; writes still go through the
        store's own methods so the ``_write_lock`` contract stays in one place.
        """
        return self._conn

    def commit(self) -> None:
        """Commit pending writes on the store connection.

        Used by the backup path's cross-connection flush so callers do not reach
        the private connection. Requires a live connection: a closed store
        raises, matching direct ``_conn.commit()`` use.
        """
        with self._write_lock:
            self._conn.commit()

    def backup(self, dest: sqlite3.Connection) -> None:
        """Copy the store's database into the already-open ``dest`` connection.

        Thin wrapper over ``sqlite3.Connection.backup`` so callers snapshot the
        store without reaching its private connection. Requires a live
        connection, matching direct ``_conn.backup(dest)`` use.
        """
        with self._write_lock:
            self._conn.backup(dest)

    # -- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn:
            # Graceful shutdown hygiene: checkpoint committed WAL frames before
            # releasing the connection.  This does not run on crash/kill, and
            # PASSIVE can leave frames behind when another reader is active.
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass  # best-effort only; don't let this mask the real close()
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
