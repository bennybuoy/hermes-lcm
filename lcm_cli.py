"""JSON-first, read-only diagnostics CLI for hermes-lcm."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_NOT_FOUND = 3
EXIT_CONFIG = 4
EXIT_DATABASE = 5
SCHEMA_VERSION = 8
_CLI_MAX_PREVIEW_CHARS = 20_000
_CLI_MAX_OUTPUT_CHARS = 100_000
_CLI_MAX_OUTPUT_NODES = 2_000
_CLI_MAX_CONTAINER_ITEMS = 200
_CLI_MAX_DEPTH = 8
_FRONTIER_ITEMS_LIMIT = 1_000
_CLI_TRUNCATED = "[TRUNCATED]"
_CLI_REDACTED = "[REDACTED by hermes-lcm CLI]"
# Redaction happens before serialization truncation. Bound every regex input,
# while retaining enough lookahead to catch a normal PEM block or standalone
# credential that begins immediately before the emitted preview boundary.
_CLI_REDACTION_LOOKAHEAD_CHARS = 20_000
_CLI_REDACTION_SCAN_MAX_CHARS = _CLI_MAX_PREVIEW_CHARS + _CLI_REDACTION_LOOKAHEAD_CHARS
_CLI_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|bearer[_-]?token|access[_-]?token|password|passwd|pwd|passphrase|client[_-]?secret|private[_-]?key|credential|secret|\btoken\b)",
    re.IGNORECASE,
)
_CLI_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]+=*")
_CLI_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|password|passwd|pwd|passphrase|client[_-]?secret|authorization|access[_-]?token|secret|token)\b\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_CLI_PRIVATE_KEY_BEGIN_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE
)
_CLI_PRIVATE_KEY_END_RE = re.compile(
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE
)
_CLI_STANDALONE_CREDENTIAL_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    (?:
        sk-(?:proj-)?[A-Za-z0-9_-]{32,}
        |(?:AKIA|ASIA)[A-Z0-9]{16}
        |gh[pousr]_[A-Za-z0-9]{36}
        |github_pat_[A-Za-z0-9_]{20,}
        |glpat-[A-Za-z0-9_-]{20,}
        |AIza[A-Za-z0-9_-]{35}
        |sk_live_[A-Za-z0-9]{16,}
        |xox[baprs]-[A-Za-z0-9-]{20,}
    )
    (?![A-Za-z0-9_-])
    """,
    re.VERBOSE,
)


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


def _bounded_preview_chars(value: int) -> int:
    if value <= 0 or value > _CLI_MAX_PREVIEW_CHARS:
        raise CliError(
            f"preview-chars must be between 1 and {_CLI_MAX_PREVIEW_CHARS}",
            EXIT_INVALID,
        )
    return value


def _truncate_cli_text(value: str, limit: int, *, force: bool = False) -> str:
    if not force and len(value) <= limit:
        return value
    if limit <= 0:
        return ""
    if limit < len(_CLI_TRUNCATED):
        return value[:limit]
    prefix_limit = limit - len(_CLI_TRUNCATED)
    return value[:prefix_limit] + _CLI_TRUNCATED


def _redact_cli_text(value: str, *, max_chars: int = _CLI_MAX_PREVIEW_CHARS) -> str:
    output_limit = min(max(0, int(max_chars)), _CLI_MAX_PREVIEW_CHARS)
    if output_limit == 0:
        return ""
    scan_limit = min(
        _CLI_REDACTION_SCAN_MAX_CHARS,
        output_limit + _CLI_REDACTION_LOOKAHEAD_CHARS,
    )
    input_truncated = len(value) > scan_limit
    bounded = value[:scan_limit]
    protected = _redact_cli_private_keys(bounded)
    protected = _CLI_BEARER_RE.sub(_CLI_REDACTED, protected)
    protected = _CLI_STANDALONE_CREDENTIAL_RE.sub(_CLI_REDACTED, protected)
    protected = _CLI_ASSIGNMENT_RE.sub(
        lambda match: match.group(1) + _CLI_REDACTED,
        protected,
    )
    return _truncate_cli_text(protected, output_limit, force=input_truncated)


def _redact_cli_private_keys(value: str) -> str:
    """Replace complete PEM private-key blocks with one linear scan."""
    if "private key-----" not in value.lower():
        return value
    chunks: list[str] = []
    cursor = 0
    changed = False
    while True:
        begin = _CLI_PRIVATE_KEY_BEGIN_RE.search(value, cursor)
        if begin is None:
            chunks.append(value[cursor:])
            break
        end = _CLI_PRIVATE_KEY_END_RE.search(value, begin.end())
        if end is None:
            chunks.append(value[cursor:])
            break
        chunks.append(value[cursor:begin.start()])
        chunks.append(_CLI_REDACTED)
        cursor = end.end()
        changed = True
    return "".join(chunks) if changed else value


def _sanitize_output(value: Any) -> Any:
    """Recursively redact and bound every CLI output surface."""
    state = {"chars": _CLI_MAX_OUTPUT_CHARS, "nodes": _CLI_MAX_OUTPUT_NODES}

    def sanitize(item: Any, *, depth: int = 0, sensitive_key: bool = False) -> Any:
        if state["nodes"] <= 0:
            return _CLI_TRUNCATED
        state["nodes"] -= 1
        if depth > _CLI_MAX_DEPTH:
            return _CLI_TRUNCATED
        if sensitive_key and item not in (None, ""):
            state["chars"] = max(0, state["chars"] - len(_CLI_REDACTED))
            return _CLI_REDACTED
        if isinstance(item, dict):
            result = {}
            entries = list(item.items())[:_CLI_MAX_CONTAINER_ITEMS]
            for key, child in entries:
                raw_key = str(key)
                safe_key = _redact_cli_text(raw_key, max_chars=256)
                result[safe_key] = sanitize(
                    child,
                    depth=depth + 1,
                    sensitive_key=bool(
                        _CLI_SENSITIVE_KEY_RE.search(
                            raw_key[:_CLI_REDACTION_SCAN_MAX_CHARS]
                        )
                    ),
                )
            if len(item) > len(entries):
                result["output_truncated"] = True
            return result
        if isinstance(item, (list, tuple)):
            entries = list(item)[:_CLI_MAX_CONTAINER_ITEMS]
            result = [sanitize(child, depth=depth + 1) for child in entries]
            if len(item) > len(entries):
                result.append(_CLI_TRUNCATED)
            return result
        if isinstance(item, str):
            limit = min(_CLI_MAX_PREVIEW_CHARS, max(0, state["chars"]))
            protected = _redact_cli_text(item, max_chars=limit)
            state["chars"] = max(0, state["chars"] - len(protected))
            return protected
        return item

    return sanitize(value)


def _preview(value: Any, chars: int, full: bool) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    limit = _CLI_MAX_PREVIEW_CHARS if full else _bounded_preview_chars(chars)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


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
    preview_chars = _bounded_preview_chars(args.preview_chars)
    if args.action == "tail":
        where = ["store_id < ?"] if args.after_store_id > 0 else ["1=1"]
        values: list[Any] = [args.after_store_id] if args.after_store_id > 0 else []
    else:
        where = ["store_id > ?"]
        values = [args.after_store_id]
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
        "next_cursor": (
            min(item["store_id"] for item in items)
            if args.action == "tail" and len(items) == limit
            else items[-1]["store_id"] if len(items) == limit
            else None
        ),
    }


def _summaries(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "summary_nodes")
    preview_chars = _bounded_preview_chars(args.preview_chars)
    if args.action == "show":
        row = conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id=?",
            (args.node_id,),
        ).fetchone()
        if row is None:
            raise CliError("not found", EXIT_NOT_FOUND)
        item = dict(row)
        item["summary"], item["summary_truncated"] = _preview(
            item.get("summary"), preview_chars, args.full
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
            item.get("summary"), preview_chars, args.full
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
    item_rows = conn.execute(
        """
        SELECT ordinal, kind, ref_id, source_start, source_end
        FROM lcm_frontier_items WHERE conversation_id=? AND generation=?
        ORDER BY ordinal LIMIT ?
        """,
        (payload["conversation_id"], payload["generation"], _FRONTIER_ITEMS_LIMIT + 1),
    ).fetchall()
    payload["items_truncated"] = len(item_rows) > _FRONTIER_ITEMS_LIMIT
    payload["items"] = [dict(item) for item in item_rows[:_FRONTIER_ITEMS_LIMIT]]
    return payload


def _prepared(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    _require_table(conn, "lcm_prepared_batches")
    preview_chars = _bounded_preview_chars(getattr(args, "preview_chars", 500))
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
                item["summary_payload"], preview_chars, False
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


def _maintenance(args: argparse.Namespace) -> dict[str, Any]:
    from .maintenance import apply_dag_maintenance, plan_dag_maintenance

    rewrites: dict[int, str] = {}
    for value in args.rewrite or []:
        if "=" not in value:
            raise CliError("--rewrite must use NODE_ID=SUMMARY", EXIT_INVALID)
        raw_id, summary = value.split("=", 1)
        try:
            rewrite_id = int(raw_id)
        except ValueError as exc:
            raise CliError("--rewrite node id must be an integer", EXIT_INVALID) from exc
        if not summary.strip():
            raise CliError("--rewrite summary cannot be empty", EXIT_INVALID)
        rewrites[rewrite_id] = summary
    kwargs = {
        "operation": args.operation,
        "conversation_id": args.conversation_id,
        "node_id": args.node_id,
        "rewrites": rewrites or None,
        "target_session_id": args.target_session_id,
        "target_conversation_id": args.target_conversation_id,
    }
    try:
        if args.action == "plan":
            return plan_dag_maintenance(_database_path(args), **kwargs)
        return apply_dag_maintenance(
            _database_path(args), confirmation=args.confirm, **kwargs
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise CliError(f"maintenance failed: {exc}", EXIT_DATABASE) from exc


def _tui_snapshot(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
    sessions = [dict(row) for row in conn.execute(
        """
        SELECT session_id, COUNT(*) AS messages, MAX(store_id) AS latest_store_id
        FROM messages GROUP BY session_id ORDER BY latest_store_id DESC LIMIT 20
        """
    ).fetchall()]
    frontiers = [dict(row) for row in conn.execute(
        """
        SELECT conversation_id, generation, session_id, source_end_store_id, updated_at
        FROM lcm_active_frontiers ORDER BY updated_at DESC LIMIT 20
        """
    ).fetchall()] if _table_exists(conn, "lcm_active_frontiers") else []
    lines = [f"hermes-lcm operator browser — {path}", "", "sessions"]
    lines.extend(
        f"  {row['session_id']}  messages={row['messages']} latest={row['latest_store_id']}"
        for row in sessions
    )
    lines.extend(["", "active frontiers"])
    lines.extend(
        f"  {row['conversation_id']} gen={row['generation']} session={row['session_id']} end={row['source_end_store_id']}"
        for row in frontiers
    )
    return {
        "database_path": str(path),
        "read_only": True,
        "sessions": sessions,
        "frontiers": frontiers,
        "screen": "\n".join(lines),
        "commands": ["r refresh", "q quit"],
    }


def _tui(args: argparse.Namespace) -> dict[str, Any]:
    path = _database_path(args)
    conn = _open_read_only(path)
    try:
        snapshot = _tui_snapshot(conn, path)
        if args.once or not sys.stdin.isatty():
            return snapshot
        while True:
            print("\033[2J\033[H" + snapshot["screen"])
            command = input("[r]efresh [q]uit > ").strip().lower()
            if command in {"q", "quit"}:
                break
            snapshot = _tui_snapshot(conn, path)
        return {**snapshot, "closed": True}
    finally:
        conn.close()


def _activation_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Prove plugin discovery/registration/binding in this fresh CLI process."""
    import hermes_lcm

    phases: list[dict[str, Any]] = []
    started = time.perf_counter()

    def phase(name: str, phase_started: float, **details: Any) -> None:
        phases.append({
            "phase": name,
            "duration_ms": round((time.perf_counter() - phase_started) * 1000.0, 3),
            **details,
        })

    root = Path(hermes_lcm.__file__).resolve().parent
    manifest = root / "plugin.yaml"
    if not manifest.is_file():
        raise CliError("plugin discovery failed: plugin.yaml missing", EXIT_CONFIG)
    phase("discovery", started, plugin_path=str(root))

    class Context:
        preflight_only = True

        def __init__(self):
            self.engines = []

        def register_context_engine(self, engine):
            self.engines.append(engine)

        def register_tool(self, **kwargs):
            return None

    with tempfile.TemporaryDirectory(prefix="hermes-lcm-activation-") as directory:
        prior_database = os.environ.get("LCM_DATABASE_PATH")
        prior_home = os.environ.get("HERMES_HOME")
        os.environ["LCM_DATABASE_PATH"] = str(Path(directory) / "preflight.db")
        os.environ["HERMES_HOME"] = directory
        context = Context()
        activation_started = time.perf_counter()
        try:
            hermes_lcm.register(context)
        finally:
            if prior_database is None:
                os.environ.pop("LCM_DATABASE_PATH", None)
            else:
                os.environ["LCM_DATABASE_PATH"] = prior_database
            if prior_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = prior_home
        phase("activation", activation_started)
        if len(context.engines) != 1:
            raise CliError(
                f"registration failed: expected one context engine, got {len(context.engines)}",
                EXIT_CONFIG,
            )
        engine = context.engines[0]
        phase_started = time.perf_counter()
        if getattr(engine, "name", "") != args.expected_engine:
            engine.shutdown()
            raise CliError(
                f"engine resolution failed: expected {args.expected_engine}", EXIT_CONFIG
            )
        phase("registration", phase_started, engine_name=engine.name)
        phase_started = time.perf_counter()
        engine.on_session_start(
            "activation-preflight-session",
            conversation_id="activation-preflight-conversation",
            platform="preflight",
        )
        identity = engine.get_runtime_identity()
        tool_names = [schema["name"] for schema in engine.get_tool_schemas()]
        phase("session-binding", phase_started, session_id=identity["session_id"])
        engine.shutdown()

    return {
        "status": "pass",
        "fresh_process_pid": os.getpid(),
        "expected_engine": args.expected_engine,
        "effective_engine": "lcm",
        "plugin_name": identity["plugin_name"],
        "plugin_version": identity["plugin_version"],
        "plugin_path": identity["plugin_path"],
        "tool_names": tool_names,
        "phases": phases,
        "total_duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "host_activation_ordering_verified": False,
        "host_contract": "docs/host-activation-contract.md",
        "note": "This proves plugin-side registration in a fresh process; Hermes host startup ordering remains host-owned.",
    }


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
    tui = sub.add_parser("tui")
    tui.add_argument("--once", action="store_true")
    activation = sub.add_parser("activation-preflight")
    activation.add_argument("--expected-engine", default="lcm")

    maintenance = sub.add_parser("maintenance").add_subparsers(dest="action", required=True)
    for action in ("plan", "apply"):
        mutation = maintenance.add_parser(action)
        mutation.add_argument("operation", choices=("rewrite-subtree", "dissolve", "copy-subtree"))
        mutation.add_argument("--conversation-id", required=True)
        mutation.add_argument("--node-id", type=int, required=True)
        mutation.add_argument("--rewrite", action="append", default=[])
        mutation.add_argument("--target-session-id", default="")
        mutation.add_argument("--target-conversation-id", default="")
        if action == "apply":
            mutation.add_argument("--confirm", required=True)

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
        elif args.command == "maintenance":
            payload = _maintenance(args)
        elif args.command == "tui":
            payload = _tui(args)
        elif args.command == "activation-preflight":
            payload = _activation_preflight(args)
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
        print(_render(_sanitize_output(payload), args))
        return EXIT_OK
    except CliError as exc:
        print(json.dumps(_sanitize_output({"error": str(exc)})))
        return exc.exit_code
    except sqlite3.Error as exc:
        print(json.dumps(_sanitize_output({"error": f"database failure: {exc}"})))
        return EXIT_DATABASE


if __name__ == "__main__":
    raise SystemExit(main())
