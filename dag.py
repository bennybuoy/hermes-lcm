"""Summary DAG — hierarchical compaction graph.

Each node is a summary of source material (raw messages or lower-depth
summaries). Nodes form a directed acyclic graph where edges point from
a summary to its sources.

Depth semantics:
  D0 — leaf summaries of raw messages (minutes timescale)
  D1 — condensation of D0 nodes (hours)
  D2 — condensation of D1 nodes (days)
  D3+ — further condensation (weeks/months)
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .db_bootstrap import (
    ExternalContentFtsSpec,
    add_column_if_missing,
    configure_connection,
    ensure_external_content_fts,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .search_query import (
    AGE_DECAY_RATE,
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
    should_apply_directness_rank_adjustment,
)

MAX_SOURCE_IDS_PER_NODE = 6_400
MAX_SOURCE_IDS_JSON_CHARS = 128_000
_CROSS_SESSION_SESSION_ID_MAX_BYTES = 2 * 1024
_CROSS_SESSION_SESSION_ID_PREFIX_CHARS = 512
_CROSS_SESSION_SUMMARY_PREFIX_CHARS = 16 * 1024
_CROSS_SESSION_EXPAND_HINT_PREFIX_CHARS = 4 * 1024
_CROSS_SESSION_SOURCE_TYPE_MAX_BYTES = 128
_CROSS_SESSION_SOURCE_TYPE_PREFIX_CHARS = 64
_CROSS_SESSION_ROW_MAX_BYTES = 384 * 1024
_CROSS_SESSION_QUERY_MAX_MATERIALIZED_BYTES = 8 * 1024 * 1024


def decode_source_ids(raw: Any) -> list[int]:
    """Decode lineage only after cheap encoded-size/cardinality checks."""
    if raw in (None, ""):
        return []
    if not isinstance(raw, str):
        raise ValueError("source_ids must be encoded JSON text")
    if len(raw) > MAX_SOURCE_IDS_JSON_CHARS:
        raise ValueError("source_ids encoded-size hard cap exceeded")
    stripped = raw.strip()
    if stripped not in {"", "[]"} and stripped.count(",") + 1 > MAX_SOURCE_IDS_PER_NODE:
        raise ValueError("source_ids cardinality hard cap exceeded")
    value = json.loads(stripped or "[]")
    if not isinstance(value, list) or len(value) > MAX_SOURCE_IDS_PER_NODE:
        raise ValueError("source_ids must be a bounded list")
    return [int(item) for item in value]
from .store import _normalize_source_value, _UNKNOWN_SOURCE, _legacy_blank_source_clause


logger = logging.getLogger(__name__)


def _build_search_order_by(sort: str | None, recency_expr: str) -> str:
    normalized = normalize_search_sort(sort)
    if normalized == "relevance":
        return f"rank ASC, {recency_expr} DESC"
    if normalized == "hybrid":
        return (
            f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {recency_expr}) / 3600.0)) * {AGE_DECAY_RATE}))) ASC, "
            f"{recency_expr} DESC"
        )
    return f"{recency_expr} DESC"


def _fallback_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(node.search_rank or 0.0) * -1.0
    recency = float(node.latest_at or node.created_at or 0.0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (-score, -directness, -recency)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, -recency)
    return (-recency, -score, -directness)


def _fts_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    recency = float(node.latest_at or node.created_at or 0.0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (rank_value, -directness, -recency)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return (-blended_strength, -directness, -recency)
    return (-recency, rank_value, 0.0)


def _fts_primary_value(node: "SummaryNode", sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        recency = float(node.latest_at or node.created_at or 0.0)
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return -blended_strength
    return rank_value


def build_nodes_fts_spec() -> ExternalContentFtsSpec:
    return ExternalContentFtsSpec(
        table_name="nodes_fts",
        content_table="summary_nodes",
        content_rowid="node_id",
        indexed_column="summary",
        trigger_sqls=(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                AFTER DELETE ON summary_nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, summary)
                    VALUES('delete', old.node_id, old.summary);
            END;
            """,
        ),
    )


@dataclass
class SummaryNode:
    """A single node in the summary DAG."""
    node_id: int = 0
    session_id: str = ""
    depth: int = 0
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0  # total tokens of source material
    source_ids: List[int] = field(default_factory=list)  # store_ids or node_ids
    source_type: str = "messages"  # "messages" or "nodes"
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    expand_hint: str = ""  # "Expand for details about: ..."
    search_rank: float | None = None
    search_directness: float = 0.0


class SummaryDAG:
    """SQLite-backed DAG of summary nodes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.RLock()
        self._init_db()

    @property
    def connection(self) -> Optional[sqlite3.Connection]:
        """The live SQLite connection, or ``None`` once :meth:`close` has run.

        Exposed for read-oriented diagnostics and inspection -- FTS sync counts,
        integrity checks, latest-node lookups -- that need ad-hoc queries the DAG
        does not wrap in a purpose-built method. Callers must treat it as
        read-only and tolerate ``None``; writes still go through the DAG's own
        methods so the ``_db_lock`` contract stays in one place.
        """
        return self._conn

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                earliest_at REAL,
                latest_at REAL,
                expand_hint TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_session_depth
                ON summary_nodes(session_id, depth, created_at);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        ensure_external_content_fts(
            self._conn,
            build_nodes_fts_spec(),
        )
        run_versioned_migrations(self._conn)
        self._ensure_source_window_columns()
        self._conn.commit()

    def _ensure_source_window_columns(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(summary_nodes)").fetchall()
        }
        add_column_if_missing(
            self._conn, columns, "earliest_at",
            "ALTER TABLE summary_nodes ADD COLUMN earliest_at REAL",
        )
        add_column_if_missing(
            self._conn, columns, "latest_at",
            "ALTER TABLE summary_nodes ADD COLUMN latest_at REAL",
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at)"
        )

    # -- Write --------------------------------------------------------------

    @staticmethod
    def add_node_no_commit(
        conn: sqlite3.Connection,
        node: SummaryNode,
    ) -> int:
        """Insert ``node`` on an existing transaction without committing.

        Cross-table publication uses the frontier coordinator's connection so
        the canonical row, FTS trigger, frontier, batch, and lifecycle changes
        share one SQLite commit.  The caller owns the connection lock and the
        transaction lifetime.
        """
        cur = conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, earliest_at, latest_at, expand_hint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.session_id,
                node.depth,
                node.summary,
                node.token_count,
                node.source_token_count,
                json.dumps(node.source_ids),
                node.source_type,
                node.created_at or time.time(),
                node.earliest_at,
                node.latest_at,
                node.expand_hint,
            ),
        )
        node.node_id = int(cur.lastrowid)
        return node.node_id

    def add_node(self, node: SummaryNode) -> int:
        """Insert a summary node and return its node_id."""
        with self._db_lock:
            node_id = self.add_node_no_commit(self._conn, node)
            self._conn.commit()
            return node_id

    def delete_node(self, node_id: int) -> bool:
        """Delete a single summary node by id. Returns True if a row was removed.

        Used to roll back a partial async-promotion publish when the frontier
        CAS fails after a canonical insert, or when a test injects a mid-publish
        failure hook.
        """
        if not node_id:
            return False
        with self._db_lock:
            cur = self._conn.execute(
                "DELETE FROM summary_nodes WHERE node_id = ?",
                (int(node_id),),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_below_depth(self, session_id: str, min_depth: int) -> int:
        """Delete all nodes for a session with depth < min_depth.

        Returns the number of deleted nodes. Used during session reset
        to retain only high-level summaries across sessions.
        """
        with self._db_lock:
            cur = self._conn.execute(
                """DELETE FROM summary_nodes
                   WHERE session_id = ? AND depth < ?""",
                (session_id, min_depth),
            )
            deleted = cur.rowcount
            self._conn.commit()
        return deleted

    def delete_session_nodes(self, session_id: str) -> int:
        """Delete all nodes for a session. Returns count deleted."""
        with self._db_lock:
            cur = self._conn.execute(
                "DELETE FROM summary_nodes WHERE session_id = ?",
                (session_id,),
            )
            deleted = cur.rowcount
            self._conn.commit()
        return deleted

    def reassign_session_nodes(self, old_session_id: str, new_session_id: str) -> int:
        """Move all nodes from one session_id to another.

        Used for /new carry-over where retained summaries should become part of
        the fresh session while preserving node IDs and node-to-node links.
        """
        with self._db_lock:
            cur = self._conn.execute(
                "UPDATE summary_nodes SET session_id = ? WHERE session_id = ?",
                (new_session_id, old_session_id),
            )
            moved = cur.rowcount
            self._conn.commit()
        return moved

    # -- Read ---------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[SummaryNode]:
        row = self._conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_node_for_cross_session_authorization(
        self, node_id: int
    ) -> Optional[SummaryNode]:
        """Load an explicit archive candidate with lineage guarded in SQLite."""
        row = self._conn.execute(
            f"""SELECT node_id, session_id, depth, summary, token_count,
                       source_token_count,
                       COALESCE(length(CAST(source_ids AS BLOB)), 0),
                       CASE
                         WHEN typeof(source_ids) = 'text'
                          AND COALESCE(length(CAST(source_ids AS BLOB)), 0) <= ?
                         THEN substr(CAST(source_ids AS TEXT), 1, ?)
                       END,
                       source_type, created_at, earliest_at, latest_at, expand_hint
                FROM summary_nodes WHERE node_id = ?""",
            (
                MAX_SOURCE_IDS_JSON_CHARS,
                MAX_SOURCE_IDS_JSON_CHARS + 1,
                node_id,
            ),
        ).fetchone()
        if row is None:
            return None
        raw_bytes = int(row[6] or 0)
        raw_source_ids = row[7]
        if raw_bytes > MAX_SOURCE_IDS_JSON_CHARS or raw_source_ids is None:
            raise ValueError("source_ids encoded-size hard cap exceeded")
        bounded_row = (
            row[0], row[1], row[2], row[3], row[4], row[5], raw_source_ids,
            row[8], row[9], row[10], row[11], row[12],
        )
        return self._row_to_node(bounded_row)

    @staticmethod
    def _cross_session_candidate_projection(alias: str = "") -> str:
        """Explicit, prefix-capped archive candidate projection.

        The corresponding WHERE guard rejects over-limit source rows before
        SQLite returns any text to Python.  Length and typeof metadata remain
        in the result so the Python boundary can verify the SQL contract before
        decoding lineage or constructing a SummaryNode.
        """
        prefix = f"{alias}." if alias else ""

        def bounded(column: str, chars: int) -> str:
            qualified = f"{prefix}{column}"
            return (
                f"CASE WHEN typeof({qualified}) = 'text' "
                f"THEN substr(CAST({qualified} AS TEXT), 1, {chars}) END"
            )

        text_columns = ("session_id", "summary", "source_ids", "source_type", "expand_hint")
        return ", ".join(
            [
                f"CASE WHEN typeof({prefix}node_id) = 'integer' THEN {prefix}node_id END",
                bounded("session_id", _CROSS_SESSION_SESSION_ID_PREFIX_CHARS),
                f"COALESCE(length(CAST({prefix}session_id AS BLOB)), 0)",
                f"COALESCE(length(CAST({prefix}session_id AS TEXT)), 0)",
                f"CASE WHEN typeof({prefix}depth) = 'integer' THEN {prefix}depth END",
                bounded("summary", _CROSS_SESSION_SUMMARY_PREFIX_CHARS),
                f"COALESCE(length(CAST({prefix}summary AS BLOB)), 0)",
                f"COALESCE(length(CAST({prefix}summary AS TEXT)), 0)",
                f"CASE WHEN typeof({prefix}token_count) = 'integer' THEN {prefix}token_count END",
                f"CASE WHEN typeof({prefix}source_token_count) = 'integer' THEN {prefix}source_token_count END",
                bounded("source_ids", MAX_SOURCE_IDS_JSON_CHARS + 1),
                f"COALESCE(length(CAST({prefix}source_ids AS BLOB)), 0)",
                f"COALESCE(length(CAST({prefix}source_ids AS TEXT)), 0)",
                bounded("source_type", _CROSS_SESSION_SOURCE_TYPE_PREFIX_CHARS),
                f"COALESCE(length(CAST({prefix}source_type AS BLOB)), 0)",
                f"COALESCE(length(CAST({prefix}source_type AS TEXT)), 0)",
                f"CASE WHEN typeof({prefix}created_at) IN ('integer', 'real') THEN {prefix}created_at END",
                f"CASE WHEN {prefix}earliest_at IS NULL OR typeof({prefix}earliest_at) IN ('integer', 'real') THEN {prefix}earliest_at END",
                f"CASE WHEN {prefix}latest_at IS NULL OR typeof({prefix}latest_at) IN ('integer', 'real') THEN {prefix}latest_at END",
                bounded("expand_hint", _CROSS_SESSION_EXPAND_HINT_PREFIX_CHARS),
                f"COALESCE(length(CAST({prefix}expand_hint AS BLOB)), 0)",
                f"COALESCE(length(CAST({prefix}expand_hint AS TEXT)), 0)",
                *(f"typeof({prefix}{column})" for column in text_columns),
            ]
        )

    @staticmethod
    def _cross_session_candidate_guard(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        lengths = {
            column: f"COALESCE(length(CAST({prefix}{column} AS BLOB)), 0)"
            for column in ("session_id", "summary", "source_ids", "source_type", "expand_hint")
        }
        return " AND ".join(
            [
                f"typeof({prefix}node_id) = 'integer'",
                f"typeof({prefix}depth) = 'integer'",
                f"typeof({prefix}token_count) = 'integer'",
                f"typeof({prefix}source_token_count) = 'integer'",
                f"typeof({prefix}created_at) IN ('integer', 'real')",
                f"({prefix}earliest_at IS NULL OR typeof({prefix}earliest_at) IN ('integer', 'real'))",
                f"({prefix}latest_at IS NULL OR typeof({prefix}latest_at) IN ('integer', 'real'))",
                *(f"typeof({prefix}{column}) = 'text'" for column in lengths),
                f"{lengths['session_id']} <= {_CROSS_SESSION_SESSION_ID_MAX_BYTES}",
                f"COALESCE(length(CAST({prefix}session_id AS TEXT)), 0) <= {_CROSS_SESSION_SESSION_ID_PREFIX_CHARS}",
                f"{lengths['source_ids']} <= {MAX_SOURCE_IDS_JSON_CHARS}",
                f"COALESCE(length(CAST({prefix}source_ids AS TEXT)), 0) <= {MAX_SOURCE_IDS_JSON_CHARS}",
                f"{lengths['source_type']} <= {_CROSS_SESSION_SOURCE_TYPE_MAX_BYTES}",
                f"COALESCE(length(CAST({prefix}source_type AS TEXT)), 0) <= {_CROSS_SESSION_SOURCE_TYPE_PREFIX_CHARS}",
            ]
        )

    def _bounded_cross_session_rows(
        self,
        sql: str,
        args: list[Any] | tuple[Any, ...],
        *,
        deadline: float,
        max_materialized_bytes: int = _CROSS_SESSION_QUERY_MAX_MATERIALIZED_BYTES,
    ) -> list[tuple[Any, ...]]:
        """Execute one bounded candidate query with a SQLite CPU deadline."""
        if time.monotonic() >= deadline:
            return []
        rows: list[tuple[Any, ...]] = []
        materialized_bytes = 0

        def interrupt_after_deadline() -> int:
            return 1 if time.monotonic() >= deadline else 0

        with self._db_lock:
            self._conn.set_progress_handler(interrupt_after_deadline, 1_000)
            try:
                cursor = self._conn.execute(sql, args)
                while time.monotonic() < deadline:
                    row = cursor.fetchone()
                    if row is None:
                        break
                    row_bytes = sum(
                        len(value.encode("utf-8", errors="surrogatepass"))
                        for value in row
                        if isinstance(value, str)
                    )
                    if row_bytes > _CROSS_SESSION_ROW_MAX_BYTES:
                        continue
                    if materialized_bytes + row_bytes > min(
                        _CROSS_SESSION_QUERY_MAX_MATERIALIZED_BYTES,
                        max(0, int(max_materialized_bytes)),
                    ):
                        break
                    materialized_bytes += row_bytes
                    rows.append(tuple(row))
            except sqlite3.OperationalError as exc:
                if "interrupted" not in str(exc).lower():
                    raise
            finally:
                self._conn.set_progress_handler(None, 0)
        return rows

    @staticmethod
    def _cross_session_row_to_node(row: tuple[Any, ...], *, rank_index: int | None = None) -> Optional[SummaryNode]:
        """Validate bounded metadata, then decode lineage and build a node."""
        if len(row) < 27 or any(row[index] is None for index in (0, 1, 4, 5, 8, 9, 10, 13, 16, 19)):
            return None
        if tuple(row[22:27]) != ("text", "text", "text", "text", "text"):
            return None
        # Full source lengths are diagnostic metadata. Summary and hint may be
        # arbitrarily large in legacy databases, but only their SQL-capped
        # prefixes are eligible to cross the Python boundary. Session identity,
        # source type, and lineage remain exact because authorization depends on
        # them and the SQL guard rejects their oversized forms.
        projected_bytes = sum(
            len(str(row[index]).encode("utf-8", errors="surrogatepass"))
            for index in (1, 5, 10, 13, 19)
        )
        if projected_bytes > _CROSS_SESSION_ROW_MAX_BYTES:
            return None
        try:
            source_ids = decode_source_ids(row[10])
            node = SummaryNode(
                node_id=int(row[0]),
                session_id=str(row[1]),
                depth=int(row[4]),
                summary=str(row[5]),
                token_count=int(row[8]),
                source_token_count=int(row[9]),
                source_ids=source_ids,
                source_type=str(row[13]),
                created_at=float(row[16]),
                earliest_at=float(row[17]) if row[17] is not None else None,
                latest_at=float(row[18]) if row[18] is not None else None,
                expand_hint=str(row[19]),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if rank_index is not None and len(row) > rank_index and row[rank_index] is not None:
            node.search_rank = float(row[rank_index])
        return node

    def get_cross_session_candidate(
        self, node_id: int, *, deadline: float
    ) -> Optional[SummaryNode]:
        """Load one explicit archive candidate without an unbounded text column."""
        projection = self._cross_session_candidate_projection()
        guard = self._cross_session_candidate_guard()
        rows = self._bounded_cross_session_rows(
            f"SELECT {projection} FROM summary_nodes WHERE node_id = ? AND {guard} LIMIT 1",
            (int(node_id),),
            deadline=deadline,
        )
        return self._cross_session_row_to_node(rows[0]) if rows else None

    def search_cross_session_candidates(
        self,
        query: str,
        *,
        limit: int,
        deadline: float,
    ) -> List[SummaryNode]:
        """Discover archive candidates through bounded FTS/LIKE SQL projections."""
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms or time.monotonic() >= deadline:
            return []
        bounded_limit = max(1, int(limit))
        candidate_cap = min(compute_search_candidate_cap(bounded_limit), 5_000)
        projection = self._cross_session_candidate_projection("n")
        guard = self._cross_session_candidate_guard("n")

        if not requires_like_fallback(query):
            order_by = _build_search_order_by(None, "COALESCE(n.latest_at, n.created_at)")
            try:
                rows = self._bounded_cross_session_rows(
                    f"""SELECT {projection}, rank AS search_rank
                        FROM nodes_fts fts
                        JOIN summary_nodes n ON n.node_id = fts.rowid
                        WHERE nodes_fts MATCH ? AND {guard}
                        ORDER BY {order_by} LIMIT ?""",
                    (safe_query, candidate_cap),
                    deadline=deadline,
                )
                nodes = [
                    node
                    for row in rows
                    if (node := self._cross_session_row_to_node(row, rank_index=27)) is not None
                ]
                for node in nodes:
                    node.search_directness = compute_directness_score(node.summary, terms, phrases)
                nodes.sort(key=lambda node: _fts_result_sort_key(node, None))
                return nodes[:bounded_limit]
            except sqlite3.Error as exc:
                if time.monotonic() >= deadline:
                    return []
                logger.warning("Bounded FTS node discovery failed, falling back to LIKE: %s", exc)

        like_clauses = ["n.summary LIKE ? ESCAPE '\\'" for _term in terms]
        like_args = [f"%{escape_like(term)}%" for term in terms]
        rows = self._bounded_cross_session_rows(
            f"""SELECT {projection}
                FROM summary_nodes n
                WHERE ({' OR '.join(like_clauses)}) AND {guard}
                LIMIT ?""",
            [*like_args, candidate_cap],
            deadline=deadline,
        )
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        nodes: list[SummaryNode] = []
        for row in rows:
            node = self._cross_session_row_to_node(row)
            if node is None:
                continue
            score = sum(
                min(count_term_matches(node.summary, term), 1)
                if collapse_risky_repeats
                else count_term_matches(node.summary, term)
                for term in terms
            )
            if score <= 0:
                continue
            node.search_rank = -float(score)
            node.search_directness = compute_directness_score(node.summary, terms, phrases)
            nodes.append(node)
        nodes.sort(key=lambda node: _fallback_result_sort_key(node, None))
        return nodes[:bounded_limit]

    def search_bounded_summary_candidates(
        self,
        query: str,
        *,
        session_ids: Sequence[str],
        limit: int,
        sort: str | None,
        source: str | None,
        deadline: float,
        max_materialized_bytes: int = _CROSS_SESSION_QUERY_MAX_MATERIALIZED_BYTES,
        max_candidate_rows: int | None = None,
    ) -> List[SummaryNode]:
        """Search grep summaries without materializing an unbounded text field."""
        bounded_sessions = [str(value) for value in session_ids if str(value)]
        if not bounded_sessions or time.monotonic() >= deadline:
            return []
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []

        bounded_limit = max(1, int(limit))
        candidate_cap = min(compute_search_candidate_cap(bounded_limit), 5_000)
        if max_candidate_rows is not None:
            candidate_cap = min(candidate_cap, max(0, int(max_candidate_rows)))
        if candidate_cap <= 0:
            return []
        projection = self._cross_session_candidate_projection("n")
        guard = self._cross_session_candidate_guard("n")
        placeholders = ",".join("?" for _ in bounded_sessions)
        session_clause = f"n.session_id IN ({placeholders})"
        source_match_cache: dict[int, bool] = {}
        apply_directness_adjustment = should_apply_directness_rank_adjustment(
            terms, phrases
        )

        if not requires_like_fallback(query):
            order_by = _build_search_order_by(
                sort, "COALESCE(n.latest_at, n.created_at)"
            )
            try:
                rows = self._bounded_cross_session_rows(
                    f"""SELECT {projection}, rank AS search_rank
                        FROM nodes_fts fts
                        JOIN summary_nodes n ON n.node_id = fts.rowid
                        WHERE nodes_fts MATCH ? AND {session_clause} AND {guard}
                        ORDER BY {order_by} LIMIT ?""",
                    [safe_query, *bounded_sessions, candidate_cap],
                    deadline=deadline,
                    max_materialized_bytes=max_materialized_bytes,
                )
                nodes: list[SummaryNode] = []
                for row in rows:
                    node = self._cross_session_row_to_node(row, rank_index=27)
                    if node is None:
                        continue
                    if source and not self._node_matches_source(
                        node.node_id,
                        source,
                        cache=source_match_cache,
                        deadline=deadline,
                    ):
                        continue
                    node.search_directness = compute_directness_score(
                        node.summary, terms, phrases
                    )
                    if apply_directness_adjustment and node.search_rank is not None:
                        node.search_rank = float(node.search_rank) - (
                            max(float(node.search_directness or 0.0), 0.0) * 3e-7
                        )
                    nodes.append(node)
                nodes.sort(key=lambda node: _fts_result_sort_key(node, sort))
                return nodes[:bounded_limit]
            except sqlite3.Error as exc:
                if time.monotonic() >= deadline:
                    return []
                logger.warning(
                    "Bounded FTS summary grep failed, falling back to LIKE: %s",
                    exc,
                )

        like_clauses = ["n.summary LIKE ? ESCAPE '\\'" for _term in terms]
        like_args = [f"%{escape_like(term)}%" for term in terms]
        rows = self._bounded_cross_session_rows(
            f"""SELECT {projection}
                FROM summary_nodes n
                WHERE ({' OR '.join(like_clauses)})
                  AND {session_clause} AND {guard}
                LIMIT ?""",
            [*like_args, *bounded_sessions, candidate_cap],
            deadline=deadline,
            max_materialized_bytes=max_materialized_bytes,
        )
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        nodes = []
        for row in rows:
            node = self._cross_session_row_to_node(row)
            if node is None:
                continue
            if source and not self._node_matches_source(
                node.node_id,
                source,
                cache=source_match_cache,
                deadline=deadline,
            ):
                continue
            score = sum(
                min(count_term_matches(node.summary, term), 1)
                if collapse_risky_repeats
                else count_term_matches(node.summary, term)
                for term in terms
            )
            if score <= 0:
                continue
            node.search_rank = -float(score)
            node.search_directness = compute_directness_score(
                node.summary, terms, phrases
            )
            nodes.append(node)
        nodes.sort(key=lambda node: _fallback_result_sort_key(node, sort))
        return nodes[:bounded_limit]

    def get_session_nodes(self, session_id: str,
                          depth: int | None = None,
                          limit: int = 1000) -> List[SummaryNode]:
        """Get nodes for a session, optionally filtered by depth."""
        with self._db_lock:
            if depth is not None:
                rows = self._conn.execute(
                    """SELECT * FROM summary_nodes
                       WHERE session_id = ? AND depth = ?
                       ORDER BY created_at LIMIT ?""",
                    (session_id, depth, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM summary_nodes
                       WHERE session_id = ?
                       ORDER BY depth, created_at LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_at_depth(self, session_id: str, depth: int) -> int:
        """Count nodes at a specific depth for a session."""
        with self._db_lock:
            row = self._conn.execute(
                """SELECT COUNT(*) FROM summary_nodes
                   WHERE session_id = ? AND depth = ?""",
                (session_id, depth),
            ).fetchone()
        return row[0] if row else 0

    def get_session_node_count(self, session_id: str) -> int:
        """Count summary nodes for a session without loading node rows."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0] if row else 0)

    def get_session_depth_stats(self, session_id: str) -> Dict[int, Dict[str, int]]:
        """Aggregate per-depth node/token stats for a session."""
        rows = self._conn.execute(
            """SELECT depth,
                      COUNT(*) AS count,
                      COALESCE(SUM(token_count), 0) AS tokens,
                      COALESCE(SUM(source_token_count), 0) AS source_tokens
               FROM summary_nodes
               WHERE session_id = ?
               GROUP BY depth
               ORDER BY depth""",
            (session_id,),
        ).fetchall()
        return {
            int(row[0]): {
                "count": int(row[1] or 0),
                "tokens": int(row[2] or 0),
                "source_tokens": int(row[3] or 0),
            }
            for row in rows
        }

    def get_session_depth_samples(
        self,
        session_id: str,
        *,
        per_depth_limit: int = 20,
        depths: List[int] | None = None,
    ) -> Dict[int, List[SummaryNode]]:
        """Return a bounded ordered sample of nodes per depth."""
        if per_depth_limit <= 0:
            return {}
        if depths is None:
            depth_rows = self._conn.execute(
                """SELECT DISTINCT depth FROM summary_nodes
                   WHERE session_id = ?
                   ORDER BY depth""",
                (session_id,),
            ).fetchall()
            depths = [int(row[0]) for row in depth_rows]

        samples: Dict[int, List[SummaryNode]] = {}
        for depth in depths:
            rows = self._conn.execute(
                """SELECT * FROM summary_nodes
                   WHERE session_id = ? AND depth = ?
                   ORDER BY created_at LIMIT ?""",
                (session_id, depth, per_depth_limit),
            ).fetchall()
            samples[int(depth)] = [self._row_to_node(row) for row in rows]
        return samples

    def get_uncondensed_at_depth(self, session_id: str, depth: int,
                                  limit: int = 100) -> List[SummaryNode]:
        """Get nodes at a depth that haven't been condensed yet.

        A node is 'uncondensed' if it's not referenced as a source by
        any higher-depth node.
        """
        with self._db_lock:
            rows = self._conn.execute(
                """SELECT n.* FROM summary_nodes n
                   WHERE n.session_id = ? AND n.depth = ?
                   AND n.node_id NOT IN (
                       SELECT json_each.value FROM summary_nodes p,
                       json_each(p.source_ids)
                       WHERE p.session_id = ? AND p.depth > ? AND p.source_type = 'nodes'
                   )
                   ORDER BY n.created_at LIMIT ?""",
                (session_id, depth, session_id, depth, limit),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None) -> List[SummaryNode]:
        """FTS5 search across summary nodes.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` filters summaries by descendant raw-message lineage, not by
          session-level source presence
        - mixed-source nodes may match more than one ``source`` filter
        """
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if requires_like_fallback(query):
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

        order_by = _build_search_order_by(sort, "COALESCE(n.latest_at, n.created_at)")
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 2e-7
        offset = 0
        scanned_rows = 0
        results: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            try:
                with self._db_lock:
                    if session_id is not None:
                        rows = self._conn.execute(
                            f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                               JOIN summary_nodes n ON n.node_id = fts.rowid
                               WHERE nodes_fts MATCH ? AND n.session_id = ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (safe_query, session_id, fetch_limit, offset),
                        ).fetchall()
                    else:
                        rows = self._conn.execute(
                            f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                               JOIN summary_nodes n ON n.node_id = fts.rowid
                               WHERE nodes_fts MATCH ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (safe_query, fetch_limit, offset),
                        ).fetchall()
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS node search failed, falling back to LIKE: %s", exc)
                return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

            raw_nodes = [self._row_to_node(r) for r in rows]
            for node in raw_nodes:
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                if apply_directness_adjustment and node.search_rank is not None:
                    rank_adjustment = max(float(node.search_directness), 0.0)
                    node.search_rank = float(node.search_rank) - (rank_adjustment * 2e-7)
                results.append(node)
            results.sort(key=lambda node: _fts_result_sort_key(node, sort))

            exhausted = len(rows) < fetch_limit or scanned_rows >= candidate_cap
            if source and not exhausted:
                offset += len(rows)
                remaining = candidate_cap - scanned_rows
                if remaining <= 0:
                    return results[:limit]
                fetch_limit = min(fetch_limit * 2, remaining)
                continue

            if exhausted or not apply_directness_adjustment or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = _fts_primary_value(raw_nodes[-1], sort)
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None) -> List[SummaryNode]:
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["summary IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        like_clauses = []
        for term in terms:
            like_clauses.append("summary LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        candidate_cap = compute_search_candidate_cap(limit)
        offset = 0
        scanned_rows = 0
        nodes: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            with self._db_lock:
                rows = self._conn.execute(
                    f"""SELECT * FROM summary_nodes
                        WHERE {' AND '.join(where)}
                        LIMIT ? OFFSET ?""",
                    [*base_args, fetch_limit, offset],
                ).fetchall()
            scanned_rows += len(rows)
            for row in rows:
                node = self._row_to_node(row)
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                score = sum(
                    min(count_term_matches(node.summary, term), 1) if collapse_risky_repeats else count_term_matches(node.summary, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                node.search_rank = -float(score)
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                nodes.append(node)

            nodes.sort(key=lambda node: _fallback_result_sort_key(node, sort))
            if not source or len(rows) < fetch_limit or scanned_rows >= candidate_cap:
                return nodes[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return nodes[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    # -- DAG traversal ------------------------------------------------------

    def get_source_nodes(self, node: SummaryNode) -> List[SummaryNode]:
        """Get the immediate child nodes of a summary node."""
        if node.source_type != "nodes" or not node.source_ids:
            return []
        placeholders = ",".join("?" * len(node.source_ids))
        rows = self._conn.execute(
            f"""SELECT * FROM summary_nodes
                WHERE node_id IN ({placeholders})
                ORDER BY created_at""",
            node.source_ids,
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _node_matches_source(
        self,
        node_id: int,
        source: str,
        *,
        cache: dict[int, bool] | None = None,
        deadline: float | None = None,
    ) -> bool:
        if not source:
            return True
        normalized_source = _normalize_source_value(source)
        if cache is not None and node_id in cache:
            return cache[node_id]
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        if deadline is not None and time.monotonic() >= deadline:
            return False

        def interrupt_after_deadline() -> int:
            return int(deadline is not None and time.monotonic() >= deadline)

        with self._db_lock:
            if deadline is not None:
                self._conn.set_progress_handler(interrupt_after_deadline, 1_000)
            try:
                row = self._conn.execute(
                    f"""
                    WITH RECURSIVE source_walk(source_type, source_id) AS (
                        SELECT n.source_type, CAST(j.value AS INTEGER)
                        FROM summary_nodes n, json_each(n.source_ids) j
                        WHERE n.node_id = ?

                        UNION ALL

                        SELECT child.source_type, CAST(j.value AS INTEGER)
                        FROM summary_nodes child
                        JOIN source_walk walk
                          ON walk.source_type = 'nodes'
                         AND child.node_id = walk.source_id
                        JOIN json_each(child.source_ids) j
                    )
                    SELECT 1
                    FROM source_walk walk
                    JOIN messages m
                      ON walk.source_type = 'messages'
                     AND m.store_id = walk.source_id
                    WHERE CASE
                            WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                            ELSE m.source = ?
                          END
                    LIMIT 1
                    """,
                    (
                        node_id,
                        normalized_source,
                        _UNKNOWN_SOURCE,
                        normalized_source,
                        normalized_source,
                    ),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if deadline is None or "interrupted" not in str(exc).lower():
                    raise
                row = None
            finally:
                if deadline is not None:
                    self._conn.set_progress_handler(None, 0)
        matched = row is not None
        if cache is not None:
            cache[node_id] = matched
        return matched

    def get_source_time_window(self, node_ids: List[int]) -> tuple[float | None, float | None]:
        if not node_ids:
            return None, None
        placeholders = ",".join("?" * len(node_ids))
        with self._db_lock:
            row = self._conn.execute(
                f"""SELECT
                        MIN(COALESCE(earliest_at, created_at)),
                        MAX(COALESCE(latest_at, created_at))
                    FROM summary_nodes
                    WHERE node_id IN ({placeholders})""",
                node_ids,
            ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def describe_subtree(self, node_id: int) -> Dict[str, Any]:
        """Return metadata about a node's subtree without loading content."""
        node = self.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        children = []
        if node.source_type == "nodes":
            for child_node in self.get_source_nodes(node):
                children.append({
                    "node_id": child_node.node_id,
                    "depth": child_node.depth,
                    "token_count": child_node.token_count,
                    "source_token_count": child_node.source_token_count,
                    "expand_hint": child_node.expand_hint,
                })

        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "token_count": node.token_count,
            "source_token_count": node.source_token_count,
            "source_type": node.source_type,
            "num_sources": len(node.source_ids),
            "earliest_at": node.earliest_at,
            "latest_at": node.latest_at,
            "expand_hint": node.expand_hint,
            "children": children,
        }

    # -- Helpers ------------------------------------------------------------

    def _row_to_node(self, row) -> SummaryNode:
        return SummaryNode(
            node_id=row[0],
            session_id=row[1],
            depth=row[2],
            summary=row[3],
            token_count=row[4],
            source_token_count=row[5],
            source_ids=decode_source_ids(row[6]),
            source_type=row[7],
            created_at=row[8],
            earliest_at=row[9],
            latest_at=row[10],
            expand_hint=row[11] or "",
            search_rank=row[12] if len(row) > 12 else None,
        )

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
