"""Tool schemas for LCM — what the LLM sees."""

LCM_GREP = {
    "name": "lcm_grep",
    "description": (
        "Search the plugin-local LCM database for past conversation content. "
        "Default scope is the active session and returns both raw messages and summary nodes across all depths. "
        "Broader scopes ('all' or 'session') must be requested explicitly and exist for bounded archive recovery "
        "over rows already present in lcm.db, including externally backfilled rows that may carry source strings "
        "such as openclaw-lcm:* . In broader scopes only raw-message hits are returned; cross-session summary "
        "node expansion is intentionally deferred. Use lcm_expand(store_id=...) on a cross-session message hit "
        "to drill into its full content. For Hermes-tracked session history outside the LCM database, use session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (FTS5 syntax: keywords, phrases, OR/NOT). "
                    "FTS5 defaults to AND matching, so prefer 1-3 distinctive terms or one quoted multi-word phrase. "
                    "Wrap exact phrases in quotes. Short CJK fragments and emoji-heavy queries may use substring fallback instead of plain FTS token matching."
                ),
            },
            "content_scope": {
                "type": "string",
                "enum": ["database", "externalized", "files", "all"],
                "description": (
                    "Content sources to search. 'database' is the safe default and preserves existing behavior; "
                    "'externalized'/'files' searches bounded prefixes of plugin-managed payload files; "
                    "'all' combines both sources."
                ),
                "default": "database",
            },
            "regex": {
                "type": "boolean",
                "description": "Use a regular expression for externalized-payload search. Database search is omitted in regex mode.",
                "default": False,
            },
            "ref": {
                "type": "string",
                "description": "Optional basename-only externalized ref filter; requires externalized/files/all content_scope.",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum payload files scanned (default 100, hard cap 500).",
                "default": 100,
            },
            "max_payload_chars": {
                "type": "integer",
                "description": "Maximum decoded content prefix searched per payload (default 65536, hard cap 1000000).",
                "default": 65536,
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Max results to return (default 10, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from in the response."
                ),
                "default": 10,
            },
            "sort": {
                "type": "string",
                "enum": ["recency", "relevance", "hybrid"],
                "description": (
                    "How to order matches. 'recency' favors newer hits, 'relevance' favors strongest FTS matches, "
                    "and 'hybrid' keeps strong older matches competitive while still boosting newer context."
                ),
                "default": "recency",
            },
            "session_scope": {
                "type": "string",
                "enum": ["current", "all", "session"],
                "description": (
                    "Scope of the search across the plugin-local LCM database. "
                    "'current' (default) restricts to the active session and preserves historical behavior. "
                    "'all' searches host-authorized sessions in the local LCM database. "
                    "'session' restricts to the authorized session_id supplied via the session_id parameter. "
                    "Both broader scopes require an opaque trusted-host capability that tool arguments cannot create. "
                    "Cross-session search returns snippets and message store_ids; cross-session summary node expansion is deferred. "
                    "For Hermes-tracked session history outside the LCM database, use session_search."
                ),
                "default": "current",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "When session_scope='session', the explicit session id to restrict the search to. "
                    "Must not be supplied with session_scope='current' or session_scope='all'."
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source/platform filter (for example cli, discord, telegram). "
                    "Applies directly to raw messages and to summaries via descendant source lineage. "
                    "Use 'unknown' for explicit unknown-source content."
                ),
            },
            "conversation_id": {
                "type": "string",
                "description": (
                    "Optional gateway conversation/session key filter for lane-scoped retrieval. "
                    "Use this to restrict Discord searches to one channel/thread/forum topic lane when rows carry metadata."
                ),
            },
            "role": {
                "type": "string",
                "enum": ["system", "user", "assistant", "tool", "unknown"],
                "description": "Optional raw-message role filter. When supplied, lcm_grep returns raw message hits only.",
            },
            "time_from": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive minimum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
            "time_to": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive maximum raw-message timestamp. Accepts Unix seconds or timezone-aware ISO 8601; "
                    "naive ISO timestamps are rejected. When supplied, lcm_grep returns raw message hits only."
                ),
            },
        },
        "required": ["query"],
    },
}

LCM_LOAD_SESSION = {
    "name": "lcm_load_session",
    "description": (
        "Load an ordered raw-message transcript page for one explicit session_id from the plugin-local LCM database. "
        "This is enumeration, not search: it does not require a query, returns raw message content rather than snippets, "
        "and orders rows chronologically by store_id. Use this after session_search or lcm_grep has identified a session_id "
        "that already exists in lcm.db. Output is bounded by limit, per-row content is bounded by max_content_chars, "
        "and row pagination uses after_store_id/next_cursor. "
        "It returns raw rows only; cross-session summary/DAG expansion remains out of scope."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Explicit LCM session id to load. Required; no implicit current/all fallback is applied.",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum raw messages to return (default 100, hard upper bound 200). "
                    "Values above the cap are clamped and reported via limit_clamped_from."
                ),
                "default": 100,
            },
            "max_content_chars": {
                "type": "integer",
                "description": (
                    "Maximum content characters to include per message (default 4000, hard upper bound 20000). "
                    "Longer rows include content_truncated=true and can be recovered fully with lcm_expand(store_id=...)."
                ),
                "default": 4000,
            },
            "after_store_id": {
                "type": "integer",
                "description": (
                    "Exclusive cursor for pagination. Pass the previous response's next_cursor "
                    "to continue with rows whose store_id is greater than this value."
                ),
                "default": 0,
            },
            "roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional role filter, for example ['user', 'assistant', 'tool', 'system'].",
            },
            "time_from": {
                "type": "number",
                "description": "Optional inclusive minimum message timestamp (Unix seconds).",
            },
            "time_to": {
                "type": "number",
                "description": "Optional inclusive maximum message timestamp (Unix seconds).",
            },
        },
        "required": ["session_id"],
    },
}

LCM_DESCRIBE = {
    "name": "lcm_describe",
    "description": (
        "Inspect a current-session summary node's subtree metadata WITHOUT loading full "
        "content, or inspect an externalized payload ref without opening the "
        "full payload. Returns token counts, child manifest, expand hints, "
        "or externalized payload metadata/preview. Use this to plan retrieval "
        "strategy before spending tokens on lcm_expand inside the active conversation. "
        "For cross-session recall, use session_search first. If called with no "
        "node_id or externalized_ref, returns the top-level DAG overview for "
        "the current session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": "Summary node ID to inspect. Omit for session overview.",
            },
            "externalized_ref": {
                "type": "string",
                "description": "Optional externalized payload ref filename to inspect instead of a summary node.",
            },
        },
        "required": [],
    },
}

LCM_EXPAND = {
    "name": "lcm_expand",
    "description": (
        "Recover the original detail behind a summary node, externalized payload, or raw message. "
        "Mode selection (exactly one): node_id (current session only) returns the source messages "
        "or lower-depth summaries that were compacted into a summary node; externalized_ref "
        "(current session only) returns a stored externalized payload's content; store_id returns "
        "a single raw message by store_id and works across sessions, suitable for drilling into "
        "cross-session lcm_grep results. Output is bounded by max_tokens; raw recovery is pageable "
        "via content_offset (and source_offset/source_limit for node_id mode). For Hermes-tracked "
        "session history outside the LCM database, prefer session_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "integer",
                "description": (
                    "Summary node ID to expand. Current-session only — cross-session DAG expansion "
                    "is not supported in this version."
                ),
            },
            "externalized_ref": {
                "type": "string",
                "description": "Externalized payload ref filename to expand instead of a summary node. Current-session only.",
            },
            "store_id": {
                "type": "integer",
                "description": (
                    "Raw message store_id to fetch. Works across trusted-host-authorized sessions, so a store_id surfaced by "
                    "a cross-session lcm_grep result can be expanded directly. Returns the message's "
                    "content paged by content_offset. If the row references an externalized payload, "
                    "the ref is surfaced via 'externalized_ref'; payload metadata and content are "
                    "session-scoped, so a cross-session row also includes 'externalized_note' "
                    "explaining that the ref is for traceability only and cannot be expanded in this version."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "description": "Token budget for returned content (default 4000, hard cap 65536)",
                "default": 4000,
            },
            "source_offset": {
                "type": "integer",
                "description": "Zero-based pagination offset into the node's immediate source list (node_id mode only).",
                "default": 0,
            },
            "source_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Maximum number of immediate sources to return from source_offset (node_id mode only, hard cap 200). Output still respects max_tokens.",
            },
            "content_offset": {
                "type": "integer",
                "description": "Character offset used to continue an oversized raw message, externalized payload, or store_id-mode message. Use next_content_offset from the previous response.",
                "default": 0,
            },
        },
        "required": [],
    },
}

LCM_STATUS = {
    "name": "lcm_status",
    "description": (
        "Get a quick health overview of the LCM engine for the current session. "
        "Shows compression count, store size, DAG depth distribution, context usage, "
        "active configuration, session/message filter state, database-wide depth-0 "
        "leaf-health aggregates (including bounded cross-session IDs), and rotate snapshot "
        "state (last_rotate_at, rotate_backup_path, rotate_backup_size when a "
        "/lcm rotate apply has been run). Use this to understand how much history "
        "has been compacted, how the engine is performing, whether the current "
        "session is matched by ignore or stateless session patterns, which message "
        "noise-suppression patterns are loaded, and when the rolling rotate "
        "backup was last written."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_FOCUS = {
    "name": "lcm_focus",
    "description": (
        "Manage a persisted, non-destructive focus overlay for the current conversation. "
        "show returns bounded metadata/preview; focus synthesizes from explicit canonical "
        "LCM node evidence; refocus uses post-focus DAG deltas; unfocus preserves history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["show", "focus", "refocus", "unfocus"],
                "default": "show",
            },
            "prompt": {
                "type": "string",
                "maxLength": 20000,
                "description": "Required for focus; optional replacement objective for refocus",
            },
        },
        "required": [],
    },
}

LCM_INSPECT = {
    "name": "lcm_inspect",
    "description": (
        "Inspect read-only LCM metadata for the current session: session/conversation "
        "lineage, message frontier and fresh tail, DAG compaction frontier, latest "
        "compaction skip/no-op reason, externalized payload refs and readability, "
        "and matched ignore/stateless patterns. This is an operator inventory tool; "
        "use lcm_grep/lcm_load_session/lcm_expand when you need actual content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of rows/items to return for bounded sections. Defaults to 20 and is capped at 200.",
                "default": 20,
            },
        },
        "required": [],
    },
}

LCM_DOCTOR = {
    "name": "lcm_doctor",
    "description": (
        "Run diagnostics on the LCM database and configuration. Checks database "
        "integrity, detects orphaned DAG nodes, validates configuration, and reports "
        "database-wide depth-0 leaf health with bounded cross-session identifiers. "
        "Use this to troubleshoot problems or verify a healthy setup."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LCM_EXPAND_QUERY = {
    "name": "lcm_expand_query",
    "description": (
        "Answer a natural-language question using expanded LCM context from the current session by default. Provide a prompt, and either "
        "query matching summaries/raw messages to expand or explicit node_ids to inspect. Uses the expansion path "
        "instead of the summarization path so retrieval/synthesis can use a different model or timeout. "
        "When expanding parent summary nodes, it recursively descends the DAG under the context budget to include leaf evidence where possible. "
        "Prefer this for questions about the active conversation after compaction; for host history, use session_search first. "
        "An administrator-enabled archive mode can search LCM DAGs across explicitly authorized sessions with shared bounds; it never activates implicitly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "maxLength": 20000,
                "description": "The question or task to answer from expanded LCM context",
            },
            "query": {
                "type": "string",
                "maxLength": 2000,
                "description": "Optional search query used to find candidate summaries before expansion",
            },
            "node_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 20,
                "description": "Optional explicit summary node IDs to expand instead of searching",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Max candidate summaries to expand when using query (default 5)",
                "default": 5,
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8192,
                "description": "Max answer tokens for bounded synthesis returned to the main agent (default 2000)",
                "default": 2000,
            },
            "context_max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "description": "Expanded serialized summary/raw/child-source/externalized fresh context budget for the auxiliary LLM before it returns the bounded answer (default max(answer max_tokens, 32000 or LCM_EXPANSION_CONTEXT_TOKENS))",
                "default": 32000,
            },
            "cross_session": {
                "type": "boolean",
                "description": "Request profile-gated cross-session LCM DAG expansion. The trusted host must separately supply a session-scoped capability.",
                "default": False,
            },
            "max_sessions": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Requested session-bucket cap, clamped to the profile maximum",
            },
            "deadline_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 120000,
                "description": "Requested operation-wide deadline, clamped to the profile and expansion timeout",
            },
        },
        "required": ["prompt"],
    },
}
