"""JSON-first, read-only diagnostics CLI for hermes-lcm."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_NOT_FOUND = 3
EXIT_CONFIG = 4
EXIT_DATABASE = 5
SCHEMA_VERSION = 7


class CliError(RuntimeError):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


def _database_path(args: argparse.Namespace) -> Path:
    if args.database:
        return Path(args.database).expanduser().resolve()
    env_path = os.environ.get("LCM_DATABASE_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    home = Path(args.hermes_home or os.environ.get("HERMES_HOME") or "~/.hermes").expanduser().resolve()
    profile = args.profile or os.environ.get("HERMES_PROFILE", "").strip()
    if profile and profile != "default":
        return home / "profiles" / profile / "lcm.db"
    return home / "lcm.db"


def _config_path(args: argparse.Namespace) -> Path:
    home = Path(args.hermes_home or os.environ.get("HERMES_HOME") or "~/.hermes").expanduser().resolve()
    profile = args.profile or os.environ.get("HERMES_PROFILE", "").strip()
    if profile and profile != "default":
        return home / "profiles" / profile / "config.yaml"
    return home / "config.yaml"


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.exists() or not path.is_file():
        raise CliError(f"database not found: {path}", EXIT_NOT_FOUND)
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=2000")
        return conn
    except sqlite3.Error as exc:
        raise CliError(f"database open failed: {exc}", EXIT_DATABASE) from exc


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (table,),
    ).fetchone() is not None


def _require_table(conn: sqlite3.Connection, table: str) -> None:
    if not _table_exists(conn, table):
        raise CliError(f"database missing required table: {table}", EXIT_DATABASE)


def _bounded_limit(value: int) -> int:
    if value <= 0 or value > 200:
        raise CliError("limit must be between 1 and 200", EXIT_INVALID)
    return value


def _preview(value: Any, chars: int, full: bool) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if full or len(text) <= chars:
        return text, False
    return text[:chars], True


def _schema_version(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "metadata"):
        return 0
    row = conn.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()
    try:
        return int(row[0]) if row else 0
    except (TypeError, ValueError):
        return 0


def _count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _status(conn: sqlite3.Connection, path: Path, _args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "messages")
    return {
        "database_path": str(path),
        "read_only": True,
        "schema_version": _schema_version(conn),
        "supported_schema_version": SCHEMA_VERSION,
        "counts": {
            "messages": _count(conn, "messages"),
            "summaries": _count(conn, "summary_nodes"),
            "frontier_generations": _count(conn, "lcm_active_frontiers"),
            "frontier_items": _count(conn, "lcm_frontier_items"),
            "prepared_batches": _count(conn, "lcm_prepared_batches"),
        },
    }


def _sessions(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "messages")
    if args.action == "show":
        row = conn.execute(
            """
            SELECT session_id, COUNT(*) AS messages, MIN(store_id) AS first_store_id,
                   MAX(store_id) AS last_store_id, MIN(timestamp) AS first_at,
                   MAX(timestamp) AS last_at
            FROM messages WHERE session_id=? GROUP BY session_id
            """,
            (args.session_id,),
        ).fetchone()
        if row is None:
            raise CliError("not found", EXIT_NOT_FOUND)
        payload = dict(row)
        payload["summaries"] = int(conn.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id=?",
            (args.session_id,),
        ).fetchone()[0]) if _table_exists(conn, "summary_nodes") else 0
        return payload
    limit = _bounded_limit(args.limit)
    rows = conn.execute(
        """
        SELECT session_id, COUNT(*) AS messages, MIN(store_id) AS first_store_id,
               MAX(store_id) AS last_store_id, MAX(timestamp) AS last_at
        FROM messages WHERE session_id > ? GROUP BY session_id
        ORDER BY session_id LIMIT ?
        """,
        (args.after_session_id, limit),
    ).fetchall()
    items = [dict(row) for row in rows]
    return {
        "items": items,
        "limit": limit,
        "next_cursor": items[-1]["session_id"] if len(items) == limit else None,
    }


def _messages(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "messages")
    limit = _bounded_limit(args.limit)
    preview_chars = args.preview_chars
    if preview_chars <= 0 or preview_chars > 20_000:
        raise CliError("preview-chars must be between 1 and 20000", EXIT_INVALID)
    where = ["store_id > ?"]
    values: list[Any] = [args.after_store_id]
    if args.session_id:
        where.append("session_id = ?")
        values.append(args.session_id)
    order = "DESC" if args.action == "tail" else "ASC"
    values.append(limit)
    rows = conn.execute(
        f"""
        SELECT store_id, session_id, source, conversation_id, role, content,
               tool_call_id, tool_name, timestamp, token_estimate, pinned
        FROM messages WHERE {' AND '.join(where)}
        ORDER BY store_id {order} LIMIT ?
        """,
        values,
    ).fetchall()
    if args.action == "tail":
        rows = list(reversed(rows))
    items = []
    for row in rows:
        item = dict(row)
        item["content"], item["content_truncated"] = _preview(
            item.get("content"), preview_chars, args.full
        )
        items.append(item)
    return {
        "items": items,
        "limit": limit,
        "preview_only": not args.full,
        "next_cursor": items[-1]["store_id"] if len(items) == limit else None,
    }


def _summaries(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "summary_nodes")
    if args.action == "show":
        row = conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id=?",
            (args.node_id,),
        ).fetchone()
        if row is None:
            raise CliError("not found", EXIT_NOT_FOUND)
        item = dict(row)
        item["summary"], item["summary_truncated"] = _preview(
            item.get("summary"), args.preview_chars, args.full
        )
        return item
    limit = _bounded_limit(args.limit)
    where = ["node_id > ?"]
    values: list[Any] = [args.after_node_id]
    if args.session_id:
        where.append("session_id=?")
        values.append(args.session_id)
    values.append(limit)
    rows = conn.execute(
        f"""
        SELECT node_id, session_id, depth, summary, token_count,
               source_token_count, source_type, created_at, expand_hint
        FROM summary_nodes WHERE {' AND '.join(where)}
        ORDER BY node_id LIMIT ?
        """,
        values,
    ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["summary"], item["summary_truncated"] = _preview(
            item.get("summary"), args.preview_chars, args.full
        )
        items.append(item)
    return {
        "items": items,
        "limit": limit,
        "preview_only": not args.full,
        "next_cursor": items[-1]["node_id"] if len(items) == limit else None,
    }


def _frontier(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "lcm_active_frontiers")
    if args.conversation_id:
        row = conn.execute(
            """
            SELECT * FROM lcm_active_frontiers WHERE conversation_id=?
            ORDER BY generation DESC LIMIT 1
            """,
            (args.conversation_id,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM lcm_active_frontiers ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise CliError("not found", EXIT_NOT_FOUND)
    payload = dict(row)
    payload["items"] = [dict(item) for item in conn.execute(
        """
        SELECT ordinal, kind, ref_id, source_start, source_end
        FROM lcm_frontier_items WHERE conversation_id=? AND generation=?
        ORDER BY ordinal
        """,
        (payload["conversation_id"], payload["generation"]),
    ).fetchall()]
    return payload


def _prepared(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "lcm_prepared_batches")
    if args.action == "show":
        row = conn.execute(
            "SELECT * FROM lcm_prepared_batches WHERE batch_id=?",
            (args.batch_id,),
        ).fetchone()
        if row is None:
            raise CliError("not found", EXIT_NOT_FOUND)
        item = dict(row)
        if not args.full and "summary_payload" in item:
            item["summary_payload"], item["summary_payload_truncated"] = _preview(
                item["summary_payload"], args.preview_chars, False
            )
        return item
    limit = _bounded_limit(args.limit)
    rows = conn.execute(
        """
        SELECT batch_id, conversation_id, session_id, base_generation,
               source_end_store_id, state, expected_leaf_count,
               frontier_end_store_id, created_at, updated_at, failure_reason,
               payload_version
        FROM lcm_prepared_batches WHERE batch_id > ?
        ORDER BY batch_id LIMIT ?
        """,
        (args.after_batch_id, limit),
    ).fetchall()
    items = [dict(row) for row in rows]
    return {
        "items": items,
        "limit": limit,
        "next_cursor": items[-1]["batch_id"] if len(items) == limit else None,
    }


def _doctor(conn: sqlite3.Connection, path: Path, _args: argparse.Namespace) -> dict[str, Any]:
    quick = conn.execute("PRAGMA quick_check").fetchall()
    foreign = conn.execute("PRAGMA foreign_key_check").fetchmany(100)
    itemless = 0
    missing_nodes = 0
    if _table_exists(conn, "lcm_active_frontiers") and _table_exists(conn, "lcm_frontier_items"):
        itemless = int(conn.execute(
            """
            SELECT COUNT(*) FROM lcm_active_frontiers f
            WHERE f.source_end_store_id > 0 AND NOT EXISTS (
              SELECT 1 FROM lcm_frontier_items i
              WHERE i.conversation_id=f.conversation_id AND i.generation=f.generation
            )
            """
        ).fetchone()[0])
        if _table_exists(conn, "summary_nodes"):
            missing_nodes = int(conn.execute(
                """
                SELECT COUNT(*) FROM lcm_frontier_items i
                WHERE i.kind='node' AND NOT EXISTS (
                  SELECT 1 FROM summary_nodes n WHERE n.node_id=i.ref_id
                )
                """
            ).fetchone()[0])
    ok = quick == [("ok",)] or [row[0] for row in quick] == ["ok"]
    return {
        "database_path": str(path),
        "read_only": True,
        "quick_check": [row[0] for row in quick],
        "foreign_key_violations": [list(row) for row in foreign],
        "itemless_positive_frontiers": itemless,
        "missing_frontier_nodes": missing_nodes,
        "status": "pass" if ok and not foreign and not itemless and not missing_nodes else "fail",
    }


def _load_lcm_config(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise CliError(f"config not found: {path}", EXIT_CONFIG)
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise CliError(f"config read failed: {exc}", EXIT_CONFIG) from exc
    if not isinstance(loaded, dict):
        raise CliError("config root must be an object", EXIT_CONFIG)
    lcm = loaded.get("lcm") or {}
    if not isinstance(lcm, dict):
        raise CliError("lcm config must be an object", EXIT_CONFIG)
    return dict(lcm)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    path = _config_path(args)
    lcm = _load_lcm_config(path)
    if args.action == "get":
        if args.key not in lcm:
            raise CliError("not found", EXIT_NOT_FOUND)
        return {"config_path": str(path), "key": args.key, "value": lcm[args.key], "read_only": True}
    return {"config_path": str(path), "lcm": lcm, "read_only": True}


def _render(payload: Any, args: argparse.Namespace) -> str:
    if not args.table:
        return json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=args.pretty)
    items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(items, list) and items:
        columns = list(items[0])
        lines = ["\t".join(columns)]
        lines.extend("\t".join(str(item.get(column, "")) for column in columns) for item in items)
        return "\n".join(lines)
    if isinstance(payload, dict):
        return "\n".join(f"{key}\t{value}" for key, value in payload.items())
    return str(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-lcm")
    parser.add_argument("--database")
    parser.add_argument("--hermes-home")
    parser.add_argument("--profile")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--table", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("doctor")

    sessions = sub.add_parser("sessions").add_subparsers(dest="action", required=True)
    sessions_list = sessions.add_parser("list")
    sessions_list.add_argument("--limit", type=int, default=100)
    sessions_list.add_argument("--after-session-id", default="")
    sessions_show = sessions.add_parser("show")
    sessions_show.add_argument("session_id")

    messages = sub.add_parser("messages").add_subparsers(dest="action", required=True)
    for name in ("list", "tail"):
        message_parser = messages.add_parser(name)
        message_parser.add_argument("--session-id", default="")
        message_parser.add_argument("--limit", type=int, default=100)
        message_parser.add_argument("--after-store-id", type=int, default=0)
        message_parser.add_argument("--preview-chars", type=int, default=500)
        message_parser.add_argument("--full", action="store_true")

    summaries = sub.add_parser("summaries").add_subparsers(dest="action", required=True)
    summaries_list = summaries.add_parser("list")
    summaries_list.add_argument("--session-id", default="")
    summaries_list.add_argument("--limit", type=int, default=100)
    summaries_list.add_argument("--after-node-id", type=int, default=0)
    summaries_list.add_argument("--preview-chars", type=int, default=500)
    summaries_list.add_argument("--full", action="store_true")
    summaries_show = summaries.add_parser("show")
    summaries_show.add_argument("node_id", type=int)
    summaries_show.add_argument("--preview-chars", type=int, default=500)
    summaries_show.add_argument("--full", action="store_true")

    frontier = sub.add_parser("frontier").add_subparsers(dest="action", required=True)
    frontier_show = frontier.add_parser("show")
    frontier_show.add_argument("--conversation-id", default="")

    batches = sub.add_parser("prepared-batches").add_subparsers(dest="action", required=True)
    batches_list = batches.add_parser("list")
    batches_list.add_argument("--limit", type=int, default=100)
    batches_list.add_argument("--after-batch-id", type=int, default=0)
    batches_show = batches.add_parser("show")
    batches_show.add_argument("batch_id", type=int)
    batches_show.add_argument("--preview-chars", type=int, default=500)
    batches_show.add_argument("--full", action="store_true")

    config = sub.add_parser("config").add_subparsers(dest="action", required=True)
    config.add_parser("show")
    config_get = config.add_parser("get")
    config_get.add_argument("key")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.pretty and args.table:
            raise CliError("--pretty and --table are mutually exclusive", EXIT_INVALID)
        if args.command == "config":
            payload = _config(args)
        else:
            path = _database_path(args)
            conn = _open_read_only(path)
            try:
                if args.command == "status":
                    payload = _status(conn, path, args)
                elif args.command == "sessions":
                    payload = _sessions(conn, args)
                elif args.command == "messages":
                    payload = _messages(conn, args)
                elif args.command == "summaries":
                    payload = _summaries(conn, args)
                elif args.command == "frontier":
                    payload = _frontier(conn, args)
                elif args.command == "prepared-batches":
                    payload = _prepared(conn, args)
                else:
                    payload = _doctor(conn, path, args)
            finally:
                conn.close()
        print(_render(payload, args))
        return EXIT_OK
    except CliError as exc:
        print(json.dumps({"error": str(exc)}))
        return exc.exit_code
    except sqlite3.Error as exc:
        print(json.dumps({"error": f"database failure: {exc}"}))
        return EXIT_DATABASE


if __name__ == "__main__":
    raise SystemExit(main())
