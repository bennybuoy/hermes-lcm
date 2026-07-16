"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import time
from typing import Any

from .db_bootstrap import configure_connection, read_existing_schema_version, SCHEMA_VERSION
from .dag import MAX_SOURCE_IDS_JSON_CHARS, MAX_SOURCE_IDS_PER_NODE
from .frontier import finalize_generation_winner_no_commit
from .tokens import count_tokens


_MAINTENANCE_MAX_NODES = 10_000
_MAINTENANCE_MAX_EDGES = 40_000
_MAINTENANCE_MAX_MESSAGES = 40_000
_MAINTENANCE_MAX_FRONTIER_ITEMS = 10_000
_MAINTENANCE_QUERY_BATCH = 400
_MAINTENANCE_COPY_MAX_BYTES = 16 * 1024 * 1024
_MAINTENANCE_COPY_MAX_TOKENS = 1_000_000
_MAINTENANCE_COPY_MAX_FIELD_BYTES = 2 * 1024 * 1024
_MAINTENANCE_COPY_MAX_NESTED_DEPTH = 32
_MAINTENANCE_COPY_MAX_NESTED_ITEMS = 20_000
_MAINTENANCE_LINEAGE_MAX_BYTES = 4 * 1024 * 1024
_MAINTENANCE_LINEAGE_DEADLINE_SECONDS = 2.0
_MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES = 8 * 1024

_MAINTENANCE_NODE_COLUMNS = (
    "node_id", "session_id", "depth", "summary", "token_count",
    "source_token_count", "source_ids", "source_type", "created_at",
    "earliest_at", "latest_at", "expand_hint",
)
_MAINTENANCE_NODE_TEXT_COLUMNS = (
    "session_id", "summary", "source_ids", "source_type", "expand_hint",
)
_MAINTENANCE_NODE_TEXT_LIMITS = (
    _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
    _MAINTENANCE_COPY_MAX_FIELD_BYTES,
    MAX_SOURCE_IDS_JSON_CHARS,
    _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
    _MAINTENANCE_COPY_MAX_FIELD_BYTES,
)
_MAINTENANCE_MESSAGE_COLUMNS = (
    "store_id", "session_id", "source", "role", "content", "tool_call_id",
    "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned",
    "conversation_id",
)
_MAINTENANCE_MESSAGE_TEXT_COLUMNS = {
    "session_id", "source", "role", "content", "tool_call_id", "tool_calls",
    "tool_name", "conversation_id",
}


def _new_maintenance_copy_budget() -> dict[str, float | int]:
    return {
        "bytes": 0,
        "tokens": 0,
        "deadline_at": time.monotonic()
        + max(0.0, float(_MAINTENANCE_LINEAGE_DEADLINE_SECONDS)),
    }


def _check_copy_deadline(budget: dict[str, float | int]) -> None:
    if time.monotonic() >= float(budget["deadline_at"]):
        raise ValueError("maintenance copy deadline exceeded")


def _charge_copy_budget(
    budget: dict[str, float | int], *, encoded_bytes: int, tokens: int = 0
) -> None:
    _check_copy_deadline(budget)
    next_bytes = int(budget["bytes"]) + max(0, int(encoded_bytes))
    if next_bytes > _MAINTENANCE_COPY_MAX_BYTES:
        raise ValueError("maintenance copy shared byte budget exceeded")
    next_tokens = int(budget["tokens"]) + max(0, int(tokens))
    if next_tokens > _MAINTENANCE_COPY_MAX_TOKENS:
        raise ValueError("maintenance copy shared token budget exceeded")
    budget["bytes"] = next_bytes
    budget["tokens"] = next_tokens


def _preflight_copy_token_budget(
    budget: dict[str, float | int], *, encoded_bytes: int
) -> None:
    """Reject before fetching text when even its byte count cannot fit tokens.

    A tokenizer cannot emit more tokens than the number of UTF-8 bytes fed to
    it.  Using that as a conservative upper bound lets maintenance prove the
    shared token cap before SQLite returns summaries, hints, IDs, or messages.
    The exact token count is still charged after the bounded fetch.
    """
    _check_copy_deadline(budget)
    if int(budget["tokens"]) + max(0, int(encoded_bytes)) > _MAINTENANCE_COPY_MAX_TOKENS:
        raise ValueError("maintenance copy shared token budget exceeded")


def _validate_copy_nested_value(value: Any) -> None:
    pending = [(value, 0)]
    items = 0
    while pending:
        current, depth = pending.pop()
        items += 1
        if items > _MAINTENANCE_COPY_MAX_NESTED_ITEMS:
            raise ValueError("maintenance copy nested-item bound exceeded")
        if depth > _MAINTENANCE_COPY_MAX_NESTED_DEPTH:
            raise ValueError("maintenance copy nested-depth bound exceeded")
        if isinstance(current, dict):
            pending.extend((key, depth + 1) for key in current)
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth + 1) for item in current)


def _bounded_copy_message_row(
    conn: sqlite3.Connection,
    store_id: int,
    message_columns: list[str],
    budget: dict[str, float | int],
) -> sqlite3.Row:
    _check_copy_deadline(budget)
    if tuple(message_columns) != _MAINTENANCE_MESSAGE_COLUMNS:
        raise ValueError("maintenance copy encountered unsupported message schema")
    quoted = [f'"{column.replace(chr(34), chr(34) * 2)}"' for column in message_columns]
    text_lengths = {
        column: f"COALESCE(length(CAST(\"{column}\" AS BLOB)), 0)"
        for column in _MAINTENANCE_MESSAGE_TEXT_COLUMNS
    }
    size_sql = ", ".join(text_lengths[column] for column in message_columns
                         if column in text_lengths)
    sizes = conn.execute(
        f"SELECT {size_sql} FROM messages WHERE store_id=?", (int(store_id),)
    ).fetchone()
    if sizes is None:
        raise ValueError("copy source message disappeared from locked snapshot")
    encoded_sizes = [int(value or 0) for value in sizes]
    if any(value > _MAINTENANCE_COPY_MAX_FIELD_BYTES for value in encoded_sizes):
        raise ValueError("maintenance copy field byte bound exceeded")
    row_bytes = sum(encoded_sizes)
    _preflight_copy_token_budget(budget, encoded_bytes=row_bytes)
    _charge_copy_budget(budget, encoded_bytes=row_bytes)
    bounded_columns: list[str] = []
    for column in message_columns:
        quoted_column = f'"{column}"'
        if column in _MAINTENANCE_MESSAGE_TEXT_COLUMNS:
            bounded_columns.append(
                f"CASE WHEN {quoted_column} IS NULL THEN NULL ELSE "
                f"substr(CAST({quoted_column} AS TEXT), "
                f"1, {_MAINTENANCE_COPY_MAX_FIELD_BYTES + 1}) END"
            )
        elif column in {"store_id", "token_estimate", "pinned"}:
            bounded_columns.append(
                f"CASE WHEN typeof({quoted_column}) = 'integer' THEN {quoted_column} END"
            )
        elif column == "timestamp":
            bounded_columns.append(
                f"CASE WHEN typeof({quoted_column}) IN ('integer', 'real') THEN {quoted_column} END"
            )
        else:  # pragma: no cover - schema tuple above is exhaustive
            raise ValueError("maintenance copy encountered unsupported message column")
    message = conn.execute(
        f"SELECT {','.join(bounded_columns)} FROM messages WHERE store_id=?",
        (int(store_id),),
    ).fetchone()
    if message is None:
        raise ValueError("copy source message disappeared from locked snapshot")
    if any(
        message[index] is None
        for index, column in enumerate(message_columns)
        if column not in _MAINTENANCE_MESSAGE_TEXT_COLUMNS
    ):
        raise ValueError("maintenance copy invalid message scalar type")
    actual_text_sizes = [
        len(str(message[index] or "").encode("utf-8", errors="replace"))
        for index, column in enumerate(message_columns)
        if column in _MAINTENANCE_MESSAGE_TEXT_COLUMNS
    ]
    if actual_text_sizes != encoded_sizes:
        raise ValueError("maintenance copy field byte bound exceeded")
    row_tokens = 0
    for column, value in zip(message_columns, message):
        if not isinstance(value, str):
            continue
        row_tokens += count_tokens(value)
        if column in {"content", "tool_calls"} and value.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                nested = None
            if nested is not None:
                _validate_copy_nested_value(nested)
    _charge_copy_budget(budget, encoded_bytes=0, tokens=row_tokens)
    return message


def _safe_database_path(db_path: str | Path) -> Path:
    raw = Path(db_path).expanduser()
    if raw.is_symlink():
        raise ValueError("maintenance refuses symlink database paths")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("maintenance database path is not a file")
    return resolved


def flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Shared by ``backup_database`` (timestamped backup) and
    ``rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place.
    """
    # Each connection is shared across threads. Commit only while holding the
    # lock that owns that connection; in particular, never commit the frontier
    # coordinator while another thread is midway through canonical publication.
    stores = (
        (getattr(engine, "_store", None), "_write_lock"),
        (getattr(engine, "_dag", None), "_db_lock"),
        (getattr(engine, "_lifecycle", None), "_lock"),
        (getattr(engine, "_frontier", None), "_lock"),
        (getattr(engine, "_focus", None), "_lock"),
    )
    for store, lock_name in stores:
        if store is None:
            continue
        conn = getattr(store, "_conn", None)
        lock = getattr(store, lock_name, None)
        if conn is None or lock is None:
            continue
        # Locks are acquired one at a time, so maintenance introduces no
        # cross-store lock order and cannot participate in an inversion.
        with lock:
            conn.commit()


def backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    directory_fd = -1
    backup_fd = -1
    dest = None
    try:
        if not backup_dir.parent.exists():
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
        directory_fd = _open_directory_nofollow(backup_dir, create=True)
        flush_engine_connections(engine)

        backup_fd = _open_private_exclusive_file(directory_fd, backup_path.name)
        dest = _sqlite_connection_for_fd(backup_fd)
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
            dest = None
        os.fsync(backup_fd)
    except (OSError, sqlite3.Error, ValueError) as exc:
        if backup_fd >= 0:
            try:
                backup_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }
    finally:
        if dest is not None:
            dest.close()
        if backup_fd >= 0:
            os.close(backup_fd)
        if directory_fd >= 0:
            os.close(directory_fd)

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def rotate_backup_database(engine) -> dict[str, Any]:
    """Write a rolling rotate-latest SQLite snapshot of the LCM store.

    Atomic via tmp-then-rename so the slot is never half-written. Unlike
    ``backup_database`` which produces timestamped files, this overwrites a
    single rolling slot so disk usage stays bounded across repeated rotates.
    """
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    tmp_path = backup_path.with_name(
        f"{backup_path.name}.{time.time_ns():x}.tmp"
    )

    directory_fd = -1
    tmp_fd = -1
    dest = None
    try:
        if not backup_dir.parent.exists():
            backup_dir.parent.mkdir(parents=True, exist_ok=True)
        directory_fd = _open_directory_nofollow(backup_dir, create=True)
        flush_engine_connections(engine)

        tmp_fd = _open_private_exclusive_file(directory_fd, tmp_path.name)
        dest = _sqlite_connection_for_fd(tmp_fd)
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
            dest = None
        os.fsync(tmp_fd)
        # Atomic replace so the rolling slot is never half-written.
        os.replace(
            tmp_path.name,
            backup_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except (OSError, sqlite3.Error, ValueError) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        if tmp_fd >= 0:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }
    finally:
        if dest is not None:
            dest.close()
        if tmp_fd >= 0:
            os.close(tmp_fd)
        if directory_fd >= 0:
            os.close(directory_fd)

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def _open_directory_nofollow(path: Path, *, create: bool) -> int:
    if path.is_symlink():
        raise ValueError(f"maintenance refuses symlink directory: {path}")
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISDIR(file_stat.st_mode):
            raise ValueError(f"maintenance artifact directory is not a directory: {path}")
        os.fchmod(fd, 0o700)
        return fd
    except Exception:
        os.close(fd)
        raise


def _open_private_exclusive_file(directory_fd: int, name: str) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError("maintenance artifact is not a private regular file")
        os.fchmod(fd, 0o600)
        return fd
    except Exception:
        os.close(fd)
        raise


def _sqlite_connection_for_fd(fd: int) -> sqlite3.Connection:
    descriptor_path = Path(f"/proc/self/fd/{fd}")
    if not descriptor_path.exists():  # pragma: no cover - non-Linux POSIX fallback
        descriptor_path = Path(f"/dev/fd/{fd}")
    return sqlite3.connect(str(descriptor_path))


def _maintenance_backup_path(db_path: Path) -> tuple[Path, int]:
    directory = db_path.parent / "lcm-maintenance-backups"
    directory_fd = _open_directory_nofollow(directory, create=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{db_path.stem}-{stamp}.sqlite3", directory_fd


def _validate_audit_path(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("maintenance audit path is a symlink")
    if path.exists() and not path.is_file():
        raise ValueError("maintenance audit path is not a regular file")


def create_verified_backup(db_path: str | Path) -> dict[str, Any]:
    """Create and read back a consistent SQLite backup for an offline apply."""
    source_path = _safe_database_path(db_path)
    backup_path, directory_fd = _maintenance_backup_path(source_path)
    backup_fd = -1
    destination = None
    source = None
    try:
        source = sqlite3.connect(str(source_path), timeout=5.0)
        backup_fd = _open_private_exclusive_file(directory_fd, backup_path.name)
        destination = _sqlite_connection_for_fd(backup_fd)
        source.backup(destination)
        destination.commit()
        quick = destination.execute("PRAGMA quick_check").fetchone()[0]
        if quick != "ok":
            raise sqlite3.DatabaseError(f"backup quick_check failed: {quick}")
        source_version = read_existing_schema_version(source)
        backup_version = read_existing_schema_version(destination)
        if source_version != backup_version:
            raise sqlite3.DatabaseError("backup schema version mismatch")
        source_counts = {
            table: int(source.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in ("messages", "summary_nodes", "lcm_active_frontiers", "lcm_frontier_items")
            if source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
        }
        backup_counts = {
            table: int(destination.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in source_counts
        }
        if source_counts != backup_counts:
            raise sqlite3.DatabaseError("backup row-count proof failed")
    except Exception:
        created_backup = backup_fd >= 0
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        if backup_fd >= 0:
            os.close(backup_fd)
            backup_fd = -1
        os.close(directory_fd)
        if created_backup:
            try:
                backup_path.unlink()
            except OSError:
                pass
        raise
    destination.close()
    os.fsync(backup_fd)
    os.close(backup_fd)
    os.close(directory_fd)
    source.close()
    digest_state = hashlib.sha256()
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    read_fd = os.open(backup_path, read_flags)
    with os.fdopen(read_fd, "rb") as backup_file:
        for chunk in iter(lambda: backup_file.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    return {
        "backup_path": str(backup_path),
        "backup_size": backup_path.stat().st_size,
        "sha256": digest,
        "schema_version": source_version,
        "row_counts": source_counts,
        "verified": True,
    }


def verify_restore_proof(backup_path: str | Path) -> dict[str, Any]:
    """Restore a backup into a temporary DB and prove it opens cleanly."""
    raw_backup = Path(backup_path).expanduser()
    if raw_backup.is_symlink():
        raise ValueError("maintenance refuses symlink backup paths")
    backup = raw_backup.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="lcm-restore-proof-") as directory:
        restored_path = Path(directory) / "restored.sqlite3"
        source = sqlite3.connect(str(backup))
        restored = sqlite3.connect(str(restored_path))
        try:
            source.backup(restored)
            restored.commit()
            quick = restored.execute("PRAGMA quick_check").fetchone()[0]
            version = read_existing_schema_version(restored)
        finally:
            restored.close()
            source.close()
    return {
        "backup_path": str(backup),
        "quick_check": quick,
        "schema_version": version,
        "restorable": quick == "ok" and version <= SCHEMA_VERSION,
    }


def _node_row(
    conn: sqlite3.Connection,
    node_id: int,
    *,
    budget: dict[str, float | int] | None = None,
) -> sqlite3.Row:
    """Read one node through explicit, byte-capped columns.

    Copy operations pass one shared budget for every node and message. Other
    maintenance operations still use a bounded one-row budget rather than
    falling back to SELECT *.
    """
    active_budget = budget if budget is not None else _new_maintenance_copy_budget()
    _check_copy_deadline(active_budget)
    length_exprs = [
        f"COALESCE(length(CAST({column} AS BLOB)), 0)"
        for column in _MAINTENANCE_NODE_TEXT_COLUMNS
    ]
    metadata = conn.execute(
        f"""SELECT
                   CASE WHEN typeof(node_id) = 'integer' THEN node_id END,
                   CASE WHEN typeof(depth) = 'integer' THEN depth END,
                   CASE WHEN typeof(token_count) = 'integer' THEN token_count END,
                   CASE WHEN typeof(source_token_count) = 'integer' THEN source_token_count END,
                   CASE WHEN typeof(created_at) IN ('integer', 'real') THEN created_at END,
                   CASE WHEN typeof(earliest_at) IN ('integer', 'real', 'null') THEN earliest_at ELSE 'invalid' END,
                   CASE WHEN typeof(latest_at) IN ('integer', 'real', 'null') THEN latest_at ELSE 'invalid' END,
                   {', '.join(length_exprs)}
            FROM summary_nodes WHERE node_id=? LIMIT 1""",
        (int(node_id),),
    ).fetchone()
    if metadata is None:
        raise ValueError(f"summary node {node_id} not found")
    if any(value is None for value in metadata[:5]) or any(
        value == "invalid" for value in metadata[5:7]
    ):
        raise ValueError("maintenance copy invalid node scalar type")
    encoded_sizes = [int(value or 0) for value in metadata[7:12]]
    if any(
        size > limit
        for size, limit in zip(encoded_sizes, _MAINTENANCE_NODE_TEXT_LIMITS)
    ):
        raise ValueError("maintenance copy field byte bound exceeded")
    _preflight_copy_token_budget(
        active_budget,
        encoded_bytes=sum(
            encoded_sizes[_MAINTENANCE_NODE_TEXT_COLUMNS.index(column)]
            for column in ("summary", "source_ids", "expand_hint")
        ),
    )
    _charge_copy_budget(active_budget, encoded_bytes=sum(encoded_sizes))
    bounded_text = {
        column: (
            f"CASE WHEN {column} IS NULL THEN NULL ELSE "
            f"substr(CAST({column} AS TEXT), 1, {limit + 1}) END"
        )
        for column, limit in zip(
            _MAINTENANCE_NODE_TEXT_COLUMNS, _MAINTENANCE_NODE_TEXT_LIMITS
        )
    }
    select_columns: list[str] = []
    for column in _MAINTENANCE_NODE_COLUMNS:
        if column in bounded_text:
            select_columns.append(bounded_text[column])
        elif column in {"node_id", "depth", "token_count", "source_token_count"}:
            select_columns.append(
                f"CASE WHEN typeof({column}) = 'integer' THEN {column} END"
            )
        elif column in {"created_at", "earliest_at", "latest_at"}:
            select_columns.append(
                f"CASE WHEN typeof({column}) IN ('integer', 'real', 'null') THEN {column} END"
            )
    row = conn.execute(
        f"SELECT {', '.join(select_columns)} FROM summary_nodes WHERE node_id=? LIMIT 1",
        (int(node_id),),
    ).fetchone()
    if row is None:
        raise ValueError(f"summary node {node_id} not found")
    actual_sizes = [
        len(str(row[_MAINTENANCE_NODE_COLUMNS.index(column)] or "").encode(
            "utf-8", errors="replace"
        ))
        for column in _MAINTENANCE_NODE_TEXT_COLUMNS
    ]
    if actual_sizes != encoded_sizes:
        raise ValueError("maintenance copy field byte bound exceeded")
    row_tokens = sum(
        count_tokens(str(row[_MAINTENANCE_NODE_COLUMNS.index(column)] or ""))
        for column in ("summary", "source_ids", "expand_hint")
    )
    _charge_copy_budget(active_budget, encoded_bytes=0, tokens=row_tokens)
    return row


def _node_session_id(conn: sqlite3.Connection, node_id: int) -> str:
    length_expr = "COALESCE(length(CAST(session_id AS BLOB)), 0)"
    row = conn.execute(
        f"""SELECT CASE
                      WHEN typeof(session_id) = 'text' AND {length_expr} <= ?
                      THEN substr(session_id, 1, ?)
                      ELSE NULL
                   END,
                   {length_expr}
            FROM summary_nodes WHERE node_id=? LIMIT 1""",
        (
            _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
            _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES + 1,
            int(node_id),
        ),
    ).fetchone()
    if row is None:
        raise ValueError(f"summary node {node_id} not found")
    if row[0] is None or int(row[1] or 0) > _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES:
        raise ValueError("DAG lineage byte bound exceeded")
    return str(row[0])


def _json_ids(raw: Any) -> list[int]:
    if not isinstance(raw, str):
        raise ValueError("source_ids must be encoded JSON text")
    if len(raw) > MAX_SOURCE_IDS_JSON_CHARS:
        raise ValueError("source_ids encoded-size hard cap exceeded")
    stripped = raw.strip()
    encoded_count = (
        stripped.count(",") + 1 if stripped not in {"", "[]"} else 0
    )
    if encoded_count > MAX_SOURCE_IDS_PER_NODE:
        raise ValueError("source_ids cardinality hard cap exceeded")
    try:
        value = json.loads(stripped or "[]")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid source_ids JSON") from exc
    if not isinstance(value, list):
        raise ValueError("source_ids must be a list")
    return [int(item) for item in value]


def _source_bounds(conn: sqlite3.Connection, node_id: int, *, limit: int = _MAINTENANCE_MAX_NODES) -> tuple[int, int]:
    _nodes, source_ids = _source_inventory(conn, node_id, limit=limit)
    if not source_ids:
        raise ValueError(f"node {node_id} has no raw source closure")
    return min(source_ids), max(source_ids)


def _source_inventory(
    conn: sqlite3.Connection,
    node_id: int,
    *,
    limit: int = 10_000,
) -> tuple[set[int], set[int]]:
    """Return the exact node/message closure beneath ``node_id``."""
    root_session_id = _node_session_id(conn, node_id)
    pending = [int(node_id)]
    nodes: set[int] = set()
    messages: set[int] = set()
    adjacency: dict[int, list[int]] = {}
    edges = 0
    lineage_bytes = 0
    deadline_at = time.monotonic() + max(
        0.0, float(_MAINTENANCE_LINEAGE_DEADLINE_SECONDS)
    )
    while pending:
        if time.monotonic() >= deadline_at:
            raise ValueError("DAG lineage deadline exceeded")
        remaining_bytes = _MAINTENANCE_LINEAGE_MAX_BYTES - lineage_bytes
        if remaining_bytes < 0:
            raise ValueError("DAG lineage byte bound exceeded")
        max_row_bytes = (
            MAX_SOURCE_IDS_JSON_CHARS
            + (2 * _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES)
            + 32
        )
        byte_sized_rows = max(1, remaining_bytes // max(1, max_row_bytes))
        page_limit = min(_MAINTENANCE_QUERY_BATCH, byte_sized_rows)
        row_byte_cap = min(
            max_row_bytes,
            max(0, remaining_bytes // max(1, page_limit)),
        )
        batch: list[int] = []
        while pending and len(batch) < page_limit:
            candidate = pending.pop()
            if candidate not in nodes and candidate not in batch:
                batch.append(candidate)
        if not batch:
            continue
        if len(nodes) + len(batch) > limit:
            raise ValueError("DAG source-closure bound exceeded")
        placeholders = ",".join("?" for _ in batch)
        session_length = "COALESCE(length(CAST(session_id AS BLOB)), 0)"
        source_length = "COALESCE(length(CAST(source_ids AS BLOB)), 0)"
        type_length = "COALESCE(length(CAST(source_type AS BLOB)), 0)"
        total_length = f"({session_length} + {source_length} + {type_length})"
        guard = (
            f"{total_length} <= ? AND {session_length} <= ? "
            f"AND {source_length} <= ? AND {type_length} <= ?"
        )
        rows = conn.execute(
            f"""SELECT node_id,
                       CASE WHEN {guard} THEN
                            substr(CAST(session_id AS TEXT), 1, ?)
                            ELSE NULL END,
                       NULL, NULL, NULL, NULL,
                       CASE WHEN {guard} THEN
                            substr(CAST(source_ids AS TEXT), 1, ?)
                            ELSE NULL END,
                       CASE WHEN {guard} THEN
                            substr(CAST(source_type AS TEXT), 1, ?)
                            ELSE NULL END,
                       {session_length}, {source_length}, {type_length}
                FROM summary_nodes WHERE node_id IN ({placeholders})""",
            (
                row_byte_cap,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                MAX_SOURCE_IDS_JSON_CHARS,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES + 1,
                row_byte_cap,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                MAX_SOURCE_IDS_JSON_CHARS,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                MAX_SOURCE_IDS_JSON_CHARS + 1,
                row_byte_cap,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                MAX_SOURCE_IDS_JSON_CHARS,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES + 1,
                *batch,
            ),
        ).fetchall()
        if time.monotonic() >= deadline_at:
            raise ValueError("DAG lineage deadline exceeded")
        by_id = {int(row[0]): row for row in rows}
        if set(batch) != set(by_id):
            raise ValueError("DAG source closure references a missing node")
        for current_id in batch:
            row = by_id[current_id]
            if time.monotonic() >= deadline_at:
                raise ValueError("DAG lineage deadline exceeded")
            row_bytes = sum(int(value or 0) for value in row[8:11])
            if row_bytes > row_byte_cap or any(
                row[index] is None for index in (1, 6, 7)
            ):
                raise ValueError("DAG lineage byte bound exceeded")
            lineage_bytes += row_bytes
            if lineage_bytes > _MAINTENANCE_LINEAGE_MAX_BYTES:
                raise ValueError("DAG lineage byte bound exceeded")
            if str(row[1]) != root_session_id:
                raise ValueError(f"node {current_id} crosses session boundary")
            nodes.add(current_id)
            ids = _json_ids(row[6])
            edges += len(ids)
            if edges > _MAINTENANCE_MAX_EDGES:
                raise ValueError("DAG source-edge bound exceeded")
            if row[7] == "messages":
                messages.update(ids)
                if len(messages) > _MAINTENANCE_MAX_MESSAGES:
                    raise ValueError("DAG source-message bound exceeded")
            elif row[7] == "nodes":
                adjacency[current_id] = ids
                pending.extend(ids)
            else:
                raise ValueError(f"node {current_id} has unknown source_type")
    indegree = {current_id: 0 for current_id in nodes}
    for child_ids in adjacency.values():
        for child_id in child_ids:
            indegree[child_id] += 1
    ready = [current_id for current_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current_id = ready.pop()
        visited += 1
        for child_id in adjacency.get(current_id, []):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    if visited != len(nodes):
        raise ValueError("cycle detected in DAG source closure")

    message_ids = list(messages)
    for offset in range(0, len(message_ids), _MAINTENANCE_QUERY_BATCH):
        batch = message_ids[offset:offset + _MAINTENANCE_QUERY_BATCH]
        placeholders = ",".join("?" for _ in batch)
        session_length = "COALESCE(length(CAST(session_id AS BLOB)), 0)"
        rows = conn.execute(
            f"""SELECT store_id,
                       CASE WHEN typeof(session_id) = 'text' AND {session_length} <= ?
                            THEN substr(session_id, 1, ?) ELSE NULL END,
                       {session_length}
                FROM messages WHERE store_id IN ({placeholders})""",
            (
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES,
                _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES + 1,
                *batch,
            ),
        ).fetchall()
        if {int(row[0]) for row in rows} != set(batch) or any(
            row[1] is None
            or int(row[2] or 0) > _MAINTENANCE_LINEAGE_SCALAR_MAX_BYTES
            or str(row[1]) != root_session_id
            for row in rows
        ):
            raise ValueError("DAG has missing/cross-session message sources")
    return nodes, messages


def _frontier_rows(
    conn: sqlite3.Connection, conversation_id: str, generation: int
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """SELECT kind, ref_id, source_start, source_end
           FROM lcm_frontier_items WHERE conversation_id=? AND generation=?
           ORDER BY ordinal LIMIT ?""",
        (conversation_id, generation, _MAINTENANCE_MAX_FRONTIER_ITEMS + 1),
    ).fetchall()
    if len(rows) > _MAINTENANCE_MAX_FRONTIER_ITEMS:
        raise ValueError("maintenance frontier-item bound exceeded")
    return rows


def _active_frontier(conn: sqlite3.Connection, conversation_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM lcm_active_frontiers WHERE conversation_id=?
        ORDER BY generation DESC LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    if row is None:
        raise ValueError("active frontier not found")
    return row


def plan_dag_maintenance(
    db_path: str | Path,
    *,
    operation: str,
    conversation_id: str,
    node_id: int,
    rewrites: dict[int, str] | None = None,
    target_session_id: str = "",
    target_conversation_id: str = "",
) -> dict[str, Any]:
    """Return an exact bounded dry-run without opening the DB writable."""
    path = _safe_database_path(db_path)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        node_budget = (
            _new_maintenance_copy_budget() if operation == "copy-subtree" else None
        )
        row = _node_row(conn, node_id, budget=node_budget)
        source_start, source_end = _source_bounds(conn, node_id)
        frontier = _active_frontier(conn, conversation_id)
        if str(row[1]) != str(frontier[2]):
            raise ValueError("maintenance root does not belong to the active conversation session")
        items = _frontier_rows(conn, conversation_id, int(frontier[1]))
        if not items:
            raise ValueError("positive frontier generation is itemless")
        root_is_active = any(
            item["kind"] == "node" and int(item["ref_id"]) == int(node_id)
            for item in items
        )
        affected = [int(node_id)]
        node_rows_added = 0
        message_rows_added = 0
        frontier_item_rows_added = len(items)
        token_delta = 0
        storage_token_delta = 0
        target_base_generation = 0
        if operation == "rewrite-subtree":
            if not root_is_active:
                raise ValueError("rewrite root is not in the active frontier")
            mapping = {int(key): str(value) for key, value in (rewrites or {}).items()}
            if int(node_id) not in mapping:
                raise ValueError("rewrite-subtree requires a replacement for the root node")
            closure_nodes, _ = _source_inventory(conn, node_id)
            if not set(mapping).issubset(closure_nodes):
                raise ValueError("rewrite set is outside the selected subtree")
            parents: dict[int, set[int]] = {}
            for closure_id in closure_nodes:
                closure_row = _node_row(conn, closure_id)
                if closure_row[7] == "nodes":
                    for child_id in _json_ids(closure_row[6]):
                        parents.setdefault(child_id, set()).add(closure_id)
            pending = [rewrite_id for rewrite_id in mapping if rewrite_id != int(node_id)]
            seen_rewrite_paths: set[int] = set()
            while pending:
                rewrite_id = pending.pop()
                if rewrite_id in seen_rewrite_paths:
                    continue
                seen_rewrite_paths.add(rewrite_id)
                for parent_id in parents.get(rewrite_id, set()):
                    if parent_id not in mapping:
                        raise ValueError(
                            "rewrite set must include every ancestor through the active root"
                        )
                    if parent_id != int(node_id):
                        pending.append(parent_id)
            for rewrite_id, summary in mapping.items():
                rewrite_row = _node_row(conn, rewrite_id)
                if str(rewrite_row[1]) != str(row[1]):
                    raise ValueError("rewrite set crosses session boundary")
                storage_token_delta += count_tokens(summary) - int(rewrite_row[4] or 0)
            affected = sorted(mapping)
            node_rows_added = len(mapping)
            token_delta = count_tokens(mapping[int(node_id)]) - int(row[4] or 0)
        elif operation == "dissolve":
            if not root_is_active:
                raise ValueError("dissolve root is not in the active frontier")
            if row[7] != "nodes":
                raise ValueError("dissolve requires a condensation node")
            child_ids = _json_ids(row[6])
            affected.extend(child_ids)
            frontier_item_rows_added = len(items) - 1 + len(child_ids)
            token_delta = (
                sum(int(_node_row(conn, child_id)[4] or 0) for child_id in child_ids)
                - int(row[4] or 0)
            )
        elif operation == "copy-subtree":
            if not target_session_id or not target_conversation_id:
                raise ValueError("copy-subtree requires target session and conversation")
            target_frontier = _active_frontier(conn, target_conversation_id)
            target_base_generation = int(target_frontier[1])
            if str(target_frontier[2]) != str(target_session_id):
                raise ValueError("target session does not own the target conversation frontier")
            target_items = _frontier_rows(
                conn, target_conversation_id, int(target_frontier[1])
            )
            for target_item in target_items:
                if target_item["kind"] != "node":
                    continue
                target_root = _node_row(
                    conn, int(target_item["ref_id"]), budget=node_budget
                )
                if (
                    int(target_root[2]) == int(row[2])
                    and str(target_root[3]) == str(row[3])
                    and str(target_root[7]) == str(row[7])
                ):
                    raise ValueError("equivalent copied subtree is already active in the target")
            closure_nodes, closure_messages = _source_inventory(conn, node_id)
            affected = sorted(closure_nodes)
            node_rows_added = len(closure_nodes)
            message_rows_added = len(closure_messages)
            frontier_item_rows_added = len(target_items) + 1
            storage_token_delta = sum(
                int(
                    (
                        row
                        if int(closure_id) == int(node_id)
                        else _node_row(conn, closure_id, budget=node_budget)
                    )[4]
                    or 0
                )
                for closure_id in closure_nodes
            )
            token_delta = int(row[4] or 0)
        else:
            raise ValueError("operation must be rewrite-subtree, dissolve, or copy-subtree")
        rows_added = node_rows_added + message_rows_added
        return {
            "dry_run": True,
            "operation": operation,
            "database_path": str(path),
            "conversation_id": conversation_id,
            "base_generation": int(frontier[1]),
            "new_generation": int(frontier[1]) + 1,
            "root_node_id": int(node_id),
            "source_start": source_start,
            "source_end": source_end,
            "affected_node_ids": affected,
            "rows_added": rows_added,
            # Kept as an alias for older machine consumers. The value is now
            # exact for every supported operation.
            "rows_added_estimate": rows_added,
            "node_rows_added": node_rows_added,
            "message_rows_added": message_rows_added,
            "frontier_generation_rows_added": 1,
            "frontier_item_rows_added": frontier_item_rows_added,
            "total_database_rows_added": (
                rows_added + 1 + frontier_item_rows_added
            ),
            "frontier_rows_republished": frontier_item_rows_added,
            "token_delta": token_delta,
            "storage_token_delta": storage_token_delta,
            "target_session_id": target_session_id,
            "target_conversation_id": target_conversation_id,
            "target_base_generation": target_base_generation,
            "confirmation": f"APPLY {operation}",
        }
    finally:
        conn.close()


def _insert_node_copy(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    session_id: str,
    source_ids: list[int],
    summary: str | None = None,
) -> int:
    text = str(row[3] if summary is None else summary)
    cur = conn.execute(
        """
        INSERT INTO summary_nodes
            (session_id, depth, summary, token_count, source_token_count,
             source_ids, source_type, created_at, expand_hint, earliest_at, latest_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            int(row[2]),
            text,
            count_tokens(text),
            int(row[5] or 0),
            json.dumps(source_ids),
            str(row[7]),
            time.time(),
            str(row[11] or ""),
            float(row[9] or 0),
            float(row[10] or 0),
        ),
    )
    return int(cur.lastrowid)


def _publish_items(
    conn: sqlite3.Connection,
    *,
    frontier: sqlite3.Row,
    conversation_id: str,
    session_id: str,
    items: list[dict[str, int | str]],
    phase_hook=None,
) -> int:
    base_generation = int(frontier[1])
    latest = _active_frontier(conn, conversation_id)
    if int(latest[1]) != base_generation:
        raise ValueError("frontier changed during maintenance")
    if not items or int(frontier[3] or 0) > 0 and not items:
        raise ValueError("positive frontier generation cannot be itemless")
    previous_end = 0
    for item in items:
        start, end = int(item["source_start"]), int(item["source_end"])
        if start <= previous_end or end < start:
            raise ValueError("frontier item ranges are missing, overlapping, or unordered")
        previous_end = end
    generation = base_generation + 1
    now = time.time()
    conn.execute(
        """
        INSERT INTO lcm_active_frontiers
            (conversation_id, generation, session_id, source_end_store_id,
             policy_fingerprint, route_fingerprint, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            generation,
            session_id,
            max(int(item["source_end"]) for item in items),
            str(frontier[4] or ""),
            str(frontier[5] or ""),
            now,
            now,
        ),
    )
    conn.executemany(
        """
        INSERT INTO lcm_frontier_items
            (conversation_id, generation, ordinal, kind, ref_id, source_start, source_end)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                conversation_id,
                generation,
                index,
                str(item["kind"]),
                int(item["ref_id"]),
                int(item["source_start"]),
                int(item["source_end"]),
            )
            for index, item in enumerate(items)
        ],
    )
    finalize_generation_winner_no_commit(
        conn,
        conversation_id=conversation_id,
        session_id=session_id,
        source_end_store_id=max(int(item["source_end"]) for item in items),
        base_generation=base_generation,
        batch_reason="dag_maintenance_generation_published",
        phase_hook=phase_hook,
    )
    return generation


def apply_dag_maintenance(
    db_path: str | Path,
    *,
    operation: str,
    conversation_id: str,
    node_id: int,
    confirmation: str,
    rewrites: dict[int, str] | None = None,
    target_session_id: str = "",
    target_conversation_id: str = "",
    publication_phase_hook=None,
    snapshot_hook=None,
) -> dict[str, Any]:
    """Backup, mutate privately, validate, then publish one new generation."""
    if confirmation != f"APPLY {operation}":
        raise ValueError(f"explicit confirmation must equal APPLY {operation}")
    path = _safe_database_path(db_path)
    audit_path = path.parent / "lcm-maintenance-audit.jsonl"
    _validate_audit_path(audit_path)
    plan = plan_dag_maintenance(
        db_path,
        operation=operation,
        conversation_id=conversation_id,
        node_id=node_id,
        rewrites=rewrites,
        target_session_id=target_session_id,
        target_conversation_id=target_conversation_id,
    )
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    configure_connection(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    created_node_ids: list[int] = []
    backup: dict[str, Any] | None = None
    try:
        if callable(snapshot_hook):
            snapshot_hook("before_begin")
        conn.execute("BEGIN IMMEDIATE")
        copy_budget = (
            _new_maintenance_copy_budget() if operation == "copy-subtree" else None
        )
        root = _node_row(conn, node_id, budget=copy_budget)
        frontier = _active_frontier(conn, conversation_id)
        if int(frontier[1]) != int(plan["base_generation"]):
            raise ValueError("frontier changed after dry-run")
        if callable(snapshot_hook):
            snapshot_hook("after_snapshot_locked")
        # The writer lock closes the dry-run/backup/mutation gap.  A sibling
        # read connection can take a consistent SQLite backup while no writer
        # can change the mutation snapshot validated above.
        backup = create_verified_backup(db_path)
        old_items = [
            dict(row) for row in _frontier_rows(
                conn, conversation_id, int(frontier[1])
            )
        ]
        if operation == "rewrite-subtree":
            mapping = {int(key): str(value) for key, value in (rewrites or {}).items()}
            remap: dict[int, int] = {}
            rewrite_rows = sorted(
                (_node_row(conn, rewrite_id) for rewrite_id in mapping),
                key=lambda row: int(row[2]),
            )
            for row in rewrite_rows:
                sources = [remap.get(source_id, source_id) for source_id in _json_ids(row[6])]
                new_id = _insert_node_copy(
                    conn,
                    row,
                    session_id=str(row[1]),
                    source_ids=sources,
                    summary=mapping[int(row[0])],
                )
                remap[int(row[0])] = new_id
                created_node_ids.append(new_id)
            if int(node_id) not in remap:
                raise ValueError("root rewrite missing")
            new_items = [
                {**item, "ref_id": remap.get(int(item["ref_id"]), int(item["ref_id"]))}
                for item in old_items
            ]
            generation = _publish_items(
                conn,
                frontier=frontier,
                conversation_id=conversation_id,
                session_id=str(frontier[2]),
                items=new_items,
                phase_hook=publication_phase_hook,
            )
        elif operation == "dissolve":
            child_ids = _json_ids(root[6])
            replacement = []
            found = False
            for item in old_items:
                if item["kind"] == "node" and int(item["ref_id"]) == int(node_id):
                    found = True
                    for child_id in child_ids:
                        child_start, child_end = _source_bounds(conn, child_id)
                        replacement.append({
                            "kind": "node",
                            "ref_id": child_id,
                            "source_start": child_start,
                            "source_end": child_end,
                        })
                else:
                    replacement.append(item)
            if not found:
                raise ValueError("dissolve root is not in the active frontier")
            generation = _publish_items(
                conn,
                frontier=frontier,
                conversation_id=conversation_id,
                session_id=str(frontier[2]),
                items=replacement,
                phase_hook=publication_phase_hook,
            )
        elif operation == "copy-subtree":
            message_remap: dict[int, int] = {}
            node_remap: dict[int, int] = {}
            _closure_nodes, closure_message_ids = _source_inventory(conn, int(node_id))
            message_columns = list(_MAINTENANCE_MESSAGE_COLUMNS)
            assert copy_budget is not None
            preloaded_rows = {int(node_id): root}

            def copy_node(source_node_id: int) -> int:
                if source_node_id in node_remap:
                    return node_remap[source_node_id]
                row = preloaded_rows.pop(source_node_id, None)
                if row is None:
                    row = _node_row(conn, source_node_id, budget=copy_budget)
                sources = _json_ids(row[6])
                if row[7] == "messages":
                    copied_sources = []
                    for store_id in sources:
                        if store_id not in message_remap:
                            message = _bounded_copy_message_row(
                                conn, store_id, message_columns, copy_budget
                            )
                            values = dict(zip(message_columns, message))
                            values.pop("store_id", None)
                            values["session_id"] = target_session_id
                            values["conversation_id"] = target_conversation_id
                            keys = list(values)
                            cur = conn.execute(
                                f"INSERT INTO messages ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                                [values[key] for key in keys],
                            )
                            message_remap[store_id] = int(cur.lastrowid)
                        copied_sources.append(message_remap[store_id])
                else:
                    copied_sources = [copy_node(child_id) for child_id in sources]
                copied_id = _insert_node_copy(
                    conn,
                    row,
                    session_id=target_session_id,
                    source_ids=copied_sources,
                )
                node_remap[source_node_id] = copied_id
                created_node_ids.append(copied_id)
                return copied_id

            copied_root = copy_node(int(node_id))
            target_frontier = _active_frontier(conn, target_conversation_id)
            if int(target_frontier[1]) != int(plan["target_base_generation"]):
                raise ValueError("target frontier changed after dry-run")
            target_items = [
                dict(row) for row in _frontier_rows(
                    conn, target_conversation_id, int(target_frontier[1])
                )
            ]
            start, end = _source_bounds(conn, copied_root)
            target_items.append({"kind": "node", "ref_id": copied_root, "source_start": start, "source_end": end})
            target_items.sort(key=lambda item: (int(item["source_start"]), int(item["source_end"])))
            generation = _publish_items(
                conn,
                frontier=target_frontier,
                conversation_id=target_conversation_id,
                session_id=target_session_id,
                items=target_items,
                phase_hook=publication_phase_hook,
            )
        else:  # pragma: no cover - plan already validates
            raise ValueError("unknown maintenance operation")

        for created_id in created_node_ids:
            _source_bounds(conn, created_id)
        foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
        quick = conn.execute("PRAGMA quick_check").fetchone()[0]
        if foreign or quick != "ok":
            raise sqlite3.DatabaseError("post-mutation integrity proof failed")
        conn.commit()
        if callable(publication_phase_hook):
            publication_phase_hook("after_commit")
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    assert backup is not None
    restore = verify_restore_proof(backup["backup_path"])
    audit_record = {
        "timestamp": time.time(),
        "operation": operation,
        "conversation_id": conversation_id,
        "root_node_id": int(node_id),
        "new_generation": generation,
        "created_node_ids": created_node_ids,
        "backup_sha256": backup["sha256"],
    }
    audit_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        audit_flags |= os.O_NOFOLLOW
    fd = os.open(audit_path, audit_flags, 0o600)
    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        os.close(fd)
        raise ValueError("maintenance audit artifact is not a private regular file")
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as audit:
        audit.write(json.dumps(audit_record, sort_keys=True) + "\n")
        audit.flush()
        os.fsync(audit.fileno())
    return {
        **plan,
        "dry_run": False,
        "applied": True,
        "new_generation": generation,
        "created_node_ids": created_node_ids,
        "backup": backup,
        "restore_proof": restore,
        "audit_path": str(audit_path),
    }
