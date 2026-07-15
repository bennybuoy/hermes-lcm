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
import tempfile
import time
from typing import Any

from .db_bootstrap import configure_connection, read_existing_schema_version, SCHEMA_VERSION
from .tokens import count_tokens


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
    engine._store.commit()
    engine._dag._conn.commit()
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        lifecycle_conn.commit()
    frontier_conn = getattr(getattr(engine, "_frontier", None), "_conn", None)
    if frontier_conn is not None:
        frontier_conn.commit()
    focus_conn = getattr(getattr(engine, "_focus", None), "_conn", None)
    if focus_conn is not None:
        focus_conn.commit()


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

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

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
    tmp_path = backup_path.with_name(backup_path.name + ".tmp")

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        if tmp_path.exists():
            tmp_path.unlink()
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
    except (OSError, sqlite3.Error) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def _maintenance_backup_path(db_path: Path) -> Path:
    directory = db_path.parent / "lcm-maintenance-backups"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{db_path.stem}-{stamp}.sqlite3"


def create_verified_backup(db_path: str | Path) -> dict[str, Any]:
    """Create and read back a consistent SQLite backup for an offline apply."""
    source_path = _safe_database_path(db_path)
    backup_path = _maintenance_backup_path(source_path)
    source = sqlite3.connect(str(source_path), timeout=5.0)
    destination = sqlite3.connect(str(backup_path))
    try:
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
        destination.close()
        source.close()
        try:
            backup_path.unlink()
        except OSError:
            pass
        raise
    destination.close()
    source.close()
    digest_state = hashlib.sha256()
    with backup_path.open("rb") as backup_file:
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
    backup = Path(backup_path).expanduser().resolve(strict=True)
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


def _node_row(conn: sqlite3.Connection, node_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM summary_nodes WHERE node_id=?", (int(node_id),)).fetchone()
    if row is None:
        raise ValueError(f"summary node {node_id} not found")
    return row


def _json_ids(raw: Any) -> list[int]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid source_ids JSON") from exc
    if not isinstance(value, list):
        raise ValueError("source_ids must be a list")
    return [int(item) for item in value]


def _source_bounds(conn: sqlite3.Connection, node_id: int, *, limit: int = 10_000) -> tuple[int, int]:
    pending = [int(node_id)]
    seen: set[int] = set()
    source_ids: list[int] = []
    while pending:
        current_id = pending.pop()
        if current_id in seen:
            raise ValueError("cycle detected in DAG source closure")
        seen.add(current_id)
        if len(seen) > limit:
            raise ValueError("DAG source-closure bound exceeded")
        row = _node_row(conn, current_id)
        ids = _json_ids(row[6])
        if row[7] == "messages":
            for store_id in ids:
                message = conn.execute(
                    "SELECT session_id FROM messages WHERE store_id=?", (store_id,)
                ).fetchone()
                if message is None or str(message[0]) != str(row[1]):
                    raise ValueError(f"node {current_id} has missing/cross-session message source {store_id}")
            source_ids.extend(ids)
        elif row[7] == "nodes":
            for child_id in ids:
                child = _node_row(conn, child_id)
                if str(child[1]) != str(row[1]):
                    raise ValueError(f"node {current_id} has cross-session child {child_id}")
            pending.extend(ids)
        else:
            raise ValueError(f"node {current_id} has unknown source_type")
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
    pending = [int(node_id)]
    nodes: set[int] = set()
    messages: set[int] = set()
    while pending:
        current_id = pending.pop()
        if current_id in nodes:
            continue
        nodes.add(current_id)
        if len(nodes) > limit:
            raise ValueError("DAG source-closure bound exceeded")
        row = _node_row(conn, current_id)
        ids = _json_ids(row[6])
        if row[7] == "messages":
            messages.update(ids)
        elif row[7] == "nodes":
            pending.extend(ids)
        else:
            raise ValueError(f"node {current_id} has unknown source_type")
    return nodes, messages


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
        row = _node_row(conn, node_id)
        source_start, source_end = _source_bounds(conn, node_id)
        frontier = _active_frontier(conn, conversation_id)
        if str(row[1]) != str(frontier[2]):
            raise ValueError("maintenance root does not belong to the active conversation session")
        items = conn.execute(
            """SELECT ordinal, kind, ref_id, source_start, source_end
               FROM lcm_frontier_items WHERE conversation_id=? AND generation=?
               ORDER BY ordinal""",
            (conversation_id, frontier[1]),
        ).fetchall()
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
            target_items = conn.execute(
                """SELECT kind, ref_id, source_start, source_end
                   FROM lcm_frontier_items WHERE conversation_id=? AND generation=?
                   ORDER BY ordinal""",
                (target_conversation_id, target_frontier[1]),
            ).fetchall()
            for target_item in target_items:
                if target_item["kind"] != "node":
                    continue
                target_root = _node_row(conn, int(target_item["ref_id"]))
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
                int(_node_row(conn, closure_id)[4] or 0)
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
) -> dict[str, Any]:
    """Backup, mutate privately, validate, then publish one new generation."""
    if confirmation != f"APPLY {operation}":
        raise ValueError(f"explicit confirmation must equal APPLY {operation}")
    plan = plan_dag_maintenance(
        db_path,
        operation=operation,
        conversation_id=conversation_id,
        node_id=node_id,
        rewrites=rewrites,
        target_session_id=target_session_id,
        target_conversation_id=target_conversation_id,
    )
    backup = create_verified_backup(db_path)
    path = _safe_database_path(db_path)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    configure_connection(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    created_node_ids: list[int] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        root = _node_row(conn, node_id)
        frontier = _active_frontier(conn, conversation_id)
        if int(frontier[1]) != int(plan["base_generation"]):
            raise ValueError("frontier changed after dry-run")
        old_items = [dict(row) for row in conn.execute(
            """SELECT kind, ref_id, source_start, source_end FROM lcm_frontier_items
               WHERE conversation_id=? AND generation=? ORDER BY ordinal""",
            (conversation_id, frontier[1]),
        )]
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
            )
        elif operation == "dissolve":
            child_ids = _json_ids(root[6])
            replacement = []
            found = False
            for item in old_items:
                if item["kind"] == "node" and int(item["ref_id"]) == int(node_id):
                    found = True
                    replacement.extend(
                        {
                            "kind": "node",
                            "ref_id": child_id,
                            "source_start": _source_bounds(conn, child_id)[0],
                            "source_end": _source_bounds(conn, child_id)[1],
                        }
                        for child_id in child_ids
                    )
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
            )
        elif operation == "copy-subtree":
            message_remap: dict[int, int] = {}
            node_remap: dict[int, int] = {}

            def copy_node(source_node_id: int) -> int:
                if source_node_id in node_remap:
                    return node_remap[source_node_id]
                row = _node_row(conn, source_node_id)
                sources = _json_ids(row[6])
                if row[7] == "messages":
                    copied_sources = []
                    for store_id in sources:
                        if store_id not in message_remap:
                            message = conn.execute("SELECT * FROM messages WHERE store_id=?", (store_id,)).fetchone()
                            columns = [description[1] for description in conn.execute("PRAGMA table_info(messages)")]
                            values = dict(zip(columns, message))
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
            target_items = [dict(row) for row in conn.execute(
                """SELECT kind, ref_id, source_start, source_end FROM lcm_frontier_items
                   WHERE conversation_id=? AND generation=? ORDER BY ordinal""",
                (target_conversation_id, target_frontier[1]),
            )]
            start, end = _source_bounds(conn, copied_root)
            target_items.append({"kind": "node", "ref_id": copied_root, "source_start": start, "source_end": end})
            target_items.sort(key=lambda item: (int(item["source_start"]), int(item["source_end"])))
            generation = _publish_items(
                conn,
                frontier=target_frontier,
                conversation_id=target_conversation_id,
                session_id=target_session_id,
                items=target_items,
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
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    restore = verify_restore_proof(backup["backup_path"])
    audit_path = path.parent / "lcm-maintenance-audit.jsonl"
    audit_record = {
        "timestamp": time.time(),
        "operation": operation,
        "conversation_id": conversation_id,
        "root_node_id": int(node_id),
        "new_generation": generation,
        "created_node_ids": created_node_ids,
        "backup_sha256": backup["sha256"],
    }
    fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as audit:
        audit.write(json.dumps(audit_record, sort_keys=True) + "\n")
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
