"""Typed compaction policy: per-model resolution of independent token boundaries.

Replaces the case-sensitive model-substring ``model_thresholds`` dict with a
structured ``ModelCompactionPolicy`` dataclass.  The policy resolver selects
on normalized provider + model + route, with aliases and longest-match
compatibility, and produces a stable fingerprint.

Invariant (validated at construction)::

    post_compaction_target < preparation_threshold <= cutover_threshold < emergency_threshold

The old ``context_threshold`` (scalar) and ``model_thresholds`` (dict[str, float])
remain as backward-compatible cutover inputs — they are folded into the
resolver as legacy cutover overrides with lowest priority.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from fnmatch import fnmatchcase
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default per-model cutover ratios
# ---------------------------------------------------------------------------

# Wishlist 2026-07-11 table. These are temporary cutover values until full
# per-model policy presets are benchmarked.
_DEFAULT_CUTOVER_OVERRIDES: dict[str, float] = {
    # MiniMax M3 — stay below 512K 2x pricing cliff (480K of 1M)
    "minimax-m3": 0.48,
    # GPT-5.6 Sol/Terra via Codex — preserve cached prefix, leave output/host reserve
    "gpt-5.6-sol": 0.85,
    "gpt-5.6-terra": 0.85,
    "gpt-5.5": 0.85,
    # GLM-5.2 through LlamaHerd — route cap, no verified price cliff
    "glm-5.2": 0.85,
    # DeepSeek V4 Flash — cached-context benefit, no verified cliff
    "deepseek-v4-flash": 0.85,
    # DeepSeek V4 Pro — hosted cache unknown, Extra High Usage
    "deepseek-v4-pro": 0.80,
    # Nemotron 3 Ultra — use route cap, not native 1M
    "nemotron-3-ultra": 0.85,
    # Kimi K2.6 — cache-capable route hypothesis
    "kimi-k2.6": 0.85,
    # Qwen 3.5 — cache-capable route hypothesis
    "qwen3.5": 0.85,
    # Qwen 3.5 alternate naming
    "qwen-3.5": 0.85,
}

DEFAULT_UNKNOWN_CUTOVER = 0.75

# Default post-compaction target as a fraction of cutover.  When no explicit
# max_assembly_tokens / reserve_tokens_floor is configured, the target is
# derived as cutover_tokens * DEFAULT_TARGET_RATIO.
DEFAULT_TARGET_RATIO = 0.50

# Default emergency pressure ratio (fraction of context_length).
DEFAULT_EMERGENCY_RATIO = 0.95

# Default preparation threshold as a fraction of cutover.  Candidates are
# built off-context when prompt pressure reaches this level.
DEFAULT_PREPARATION_RATIO = 0.80


# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------

def _normalize_model_name(model: str) -> str:
    """Lowercase and strip common provider prefixes/suffixes for matching."""
    return (model or "").strip().lower()


def _normalize_provider(provider: str) -> str:
    """Lowercase and strip provider."""
    return (provider or "").strip().lower()


# Alias map: alternate model names that map to the same canonical key.
_MODEL_ALIASES: dict[str, str] = {
    "minimax-m3": "minimax-m3",
    "m3": "minimax-m3",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.5": "gpt-5.5",
    "glm-5.2": "glm-5.2",
    "glm5.2": "glm-5.2",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "nemotron-3-ultra": "nemotron-3-ultra",
    "kimi-k2.6": "kimi-k2.6",
    "kimi-k2-6": "kimi-k2.6",
    "qwen3.5": "qwen3.5",
    "qwen-3.5": "qwen3.5",
}


def _resolve_alias(normalized: str) -> str:
    return _MODEL_ALIASES.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCompactionPolicy:
    """Typed compaction policy resolved per model/provider/route.

    All ratios are fractions of the effective context window (0.0–1.0).
    The invariant ``post_compaction_target < preparation_threshold <=
    cutover_threshold < emergency_threshold`` is validated at construction.
    """

    # The four token boundaries (as ratios of context_length):
    preparation_threshold: float  # build candidates off-context
    cutover_threshold: float      # atomically promote; this is host threshold_tokens
    post_compaction_target: float  # continue compaction until this is reached
    emergency_threshold: float     # force bounded convergence before provider failure

    # Assembly guardrails (absolute tokens, 0 = disabled):
    assembly_hard_cap: int = 0     # max_assembly_tokens
    assembly_reserve_floor: int = 0  # reserve_tokens_floor

    # Summary routing:
    summary_model: str = ""
    summary_fallback_models: tuple[str, ...] = field(default_factory=tuple)

    # Leaf sizing:
    fresh_tail_count: int = 32
    fresh_tail_max_tokens: int = 0
    leaf_chunk_tokens: int = 20_000
    dynamic_leaf_chunk_enabled: bool = False
    dynamic_leaf_chunk_max: int = 40_000

    # Condensation:
    condensation_fanin: int = 4
    condensation_min_fanin: int = 2
    incremental_max_depth: int = 3
    cache_friendly_condensation_enabled: bool = False

    # Output reserve (tokens to reserve for model output, 0 = disabled):
    output_reserve: int = 0

    # Route economics/strategy. Cache telemetry is intentionally separate:
    # observed cache reads never imply a billing discount.
    cache_economics: str = "unknown"  # discounted | none | unknown
    compaction_mode: str = "inline"  # inline | deferred
    cache_ttl_seconds: int = 0
    full_sweep_compaction_enabled: bool = False
    summary_prefix_target_tokens: int = 0

    # Source reporting:
    source: str = "default"
    model_selector: str = ""
    provider_selector: str = ""
    route_selector: str = ""
    session_selector: str = ""
    selection_reason: str = "global fallback"

    # Stable fingerprint of the resolved policy (excluding runtime context_length):
    fingerprint: str = ""

    def __post_init__(self):
        # Validate the invariant
        if not (0 < self.post_compaction_target < self.preparation_threshold):
            raise ValueError(
                f"Policy invariant violated: post_compaction_target ({self.post_compaction_target}) "
                f"must be > 0 and < preparation_threshold ({self.preparation_threshold})"
            )
        if not (self.preparation_threshold <= self.cutover_threshold):
            raise ValueError(
                f"Policy invariant violated: preparation_threshold ({self.preparation_threshold}) "
                f"must be <= cutover_threshold ({self.cutover_threshold})"
            )
        if not (self.cutover_threshold < self.emergency_threshold):
            raise ValueError(
                f"Policy invariant violated: cutover_threshold ({self.cutover_threshold}) "
                f"must be < emergency_threshold ({self.emergency_threshold})"
            )
        if not (0 < self.emergency_threshold <= 1.0):
            raise ValueError(
                f"Policy invariant violated: emergency_threshold ({self.emergency_threshold}) "
                f"must be in (0, 1.0]"
            )
        if self.fresh_tail_count < 0:
            raise ValueError("fresh_tail_count must be non-negative")
        if self.fresh_tail_max_tokens < 0:
            raise ValueError("fresh_tail_max_tokens must be non-negative")
        if self.leaf_chunk_tokens < 1:
            raise ValueError("leaf_chunk_tokens must be positive")
        if self.dynamic_leaf_chunk_max < 1:
            raise ValueError("dynamic_leaf_chunk_max must be positive")
        if self.condensation_fanin < 2:
            raise ValueError("condensation_fanin must be at least 2")
        if not 2 <= self.condensation_min_fanin <= self.condensation_fanin:
            raise ValueError(
                "condensation_min_fanin must be between 2 and condensation_fanin"
            )
        if self.incremental_max_depth < -1:
            raise ValueError("incremental_max_depth must be at least -1")
        if self.output_reserve < 0:
            raise ValueError("output_reserve must be non-negative")
        for field_name in (
            "dynamic_leaf_chunk_enabled",
            "cache_friendly_condensation_enabled",
            "full_sweep_compaction_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if self.cache_economics not in {"discounted", "none", "unknown"}:
            raise ValueError(
                "cache_economics must be discounted, none, or unknown"
            )
        if self.compaction_mode not in {"inline", "deferred"}:
            raise ValueError("compaction_mode must be inline or deferred")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")
        if self.summary_prefix_target_tokens < 0:
            raise ValueError("summary_prefix_target_tokens must be non-negative")

    # -- Token boundary helpers -------------------------------------------

    def cutover_tokens(self, context_length: int) -> int:
        """Host-visible preflight cutover trigger."""
        return max(1, int(context_length * self.cutover_threshold))

    def preparation_tokens(self, context_length: int) -> int:
        """Off-context candidate preparation trigger."""
        return max(1, int(context_length * self.preparation_threshold))

    def post_compaction_target_tokens(self, context_length: int) -> Optional[int]:
        """Independent post-compaction target.

        If assembly_hard_cap or assembly_reserve_floor are configured, the
        target is the minimum of the derived target and the assembly cap.
        """
        derived = max(1, int(context_length * self.post_compaction_target))
        caps: list[int] = []
        if self.assembly_hard_cap > 0:
            caps.append(self.assembly_hard_cap)
        if context_length > 0 and self.assembly_reserve_floor > 0:
            reserve_cap = context_length - self.assembly_reserve_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
        if context_length > 0 and self.output_reserve > 0:
            output_cap = context_length - self.output_reserve
            if output_cap > 0:
                caps.append(output_cap)
        if caps:
            return max(1, min(derived, min(caps)))
        return derived

    def emergency_tokens(self, context_length: int) -> int:
        """Provider-window pressure threshold."""
        import math
        return max(1, math.ceil(context_length * self.emergency_threshold))

    def to_status_dict(self, context_length: int) -> dict[str, Any]:
        """Return a JSON-serializable status dict for lcm_status."""
        if self.compaction_mode == "deferred":
            selected_strategy = "cache-aware-deferred"
        elif self.cache_economics == "none":
            selected_strategy = "aggressive-inline"
        elif self.cache_economics == "unknown":
            selected_strategy = "conservative-inline"
        else:
            selected_strategy = "inline"
        return {
            "fingerprint": self.fingerprint,
            "source": self.source,
            "model_selector": self.model_selector,
            "provider_selector": self.provider_selector,
            "route_selector": self.route_selector,
            "session_selector": self.session_selector,
            "selection_reason": self.selection_reason,
            "preparation_threshold": round(self.preparation_threshold, 6),
            "cutover_threshold": round(self.cutover_threshold, 6),
            "post_compaction_target": round(self.post_compaction_target, 6),
            "emergency_threshold": round(self.emergency_threshold, 6),
            "assembly_hard_cap": self.assembly_hard_cap,
            "assembly_reserve_floor": self.assembly_reserve_floor,
            "preparation_tokens": self.preparation_tokens(context_length),
            "cutover_tokens": self.cutover_tokens(context_length),
            "post_compaction_target_tokens": self.post_compaction_target_tokens(context_length),
            "emergency_tokens": self.emergency_tokens(context_length),
            "summary_model": self.summary_model,
            "summary_fallback_models": list(self.summary_fallback_models),
            "fresh_tail_count": self.fresh_tail_count,
            "fresh_tail_max_tokens": self.fresh_tail_max_tokens,
            "leaf_chunk_tokens": self.leaf_chunk_tokens,
            "dynamic_leaf_chunk_enabled": self.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": self.dynamic_leaf_chunk_max,
            "condensation_fanin": self.condensation_fanin,
            "condensation_min_fanin": self.condensation_min_fanin,
            "incremental_max_depth": self.incremental_max_depth,
            "cache_friendly_condensation_enabled": self.cache_friendly_condensation_enabled,
            "output_reserve": self.output_reserve,
            "cache_economics": self.cache_economics,
            "compaction_mode": self.compaction_mode,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "full_sweep_compaction_enabled": self.full_sweep_compaction_enabled,
            "summary_prefix_target_tokens": self.summary_prefix_target_tokens,
            "selected_strategy": selected_strategy,
            "observed_cache_telemetry_is_economic_classification": False,
        }


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _compute_fingerprint(policy_fields: dict[str, Any]) -> str:
    """Compute a stable SHA-256 fingerprint of the policy's semantic fields."""
    # Only include fields that affect behavior, not source/model selectors
    # (which are reporting metadata).
    semantic_keys = [
        "preparation_threshold",
        "cutover_threshold",
        "post_compaction_target",
        "emergency_threshold",
        "assembly_hard_cap",
        "assembly_reserve_floor",
        "summary_model",
        "summary_fallback_models",
        "fresh_tail_count",
        "fresh_tail_max_tokens",
        "leaf_chunk_tokens",
        "dynamic_leaf_chunk_enabled",
        "dynamic_leaf_chunk_max",
        "condensation_fanin",
        "condensation_min_fanin",
        "incremental_max_depth",
        "cache_friendly_condensation_enabled",
        "output_reserve",
        "cache_economics",
        "compaction_mode",
        "cache_ttl_seconds",
        "full_sweep_compaction_enabled",
        "summary_prefix_target_tokens",
    ]
    semantic = {k: policy_fields.get(k) for k in semantic_keys}
    raw = json.dumps(semantic, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def _longest_match(model: str, overrides: dict[str, float]) -> tuple[str, float] | None:
    """Find the longest substring match in *overrides* for *model*.

    Returns (key, value) or None.  Matching is case-insensitive.
    """
    if not overrides or not model:
        return None
    normalized_model = _normalize_model_name(model)
    best_key = ""
    best_value = 0.0
    for key, value in overrides.items():
        nk = _normalize_model_name(key)
        if nk in normalized_model and len(nk) > len(best_key):
            best_key = nk
            best_value = float(value)
    if best_key:
        return (best_key, best_value)
    return None


_RULE_MATCH_FIELDS = frozenset({
    "provider",
    "model",
    "route",
    "min_context_window",
    "max_context_window",
    "session",
    "platform",
})
_RULE_OVERRIDE_FIELDS = frozenset({
    "preparation_threshold",
    "cutover_threshold",
    "post_compaction_target",
    "emergency_threshold",
    "fresh_tail_count",
    "fresh_tail_max_tokens",
    "leaf_chunk_tokens",
    "dynamic_leaf_chunk_enabled",
    "dynamic_leaf_chunk_max",
    "condensation_fanin",
    "condensation_min_fanin",
    "incremental_max_depth",
    "output_reserve",
    "cache_friendly_condensation_enabled",
    "cache_friendly_condensation",
    "cache_economics",
    "compaction_mode",
    "cache_ttl_seconds",
    "full_sweep_compaction_enabled",
    "summary_prefix_target_tokens",
})

_RATIO_OVERRIDE_FIELDS = frozenset({
    "preparation_threshold",
    "cutover_threshold",
    "post_compaction_target",
    "emergency_threshold",
})
_BOOLEAN_OVERRIDE_FIELDS = frozenset({
    "dynamic_leaf_chunk_enabled",
    "cache_friendly_condensation_enabled",
    "cache_friendly_condensation",
    "full_sweep_compaction_enabled",
})
_INTEGER_OVERRIDE_MINIMUMS = {
    "fresh_tail_count": 0,
    "fresh_tail_max_tokens": 0,
    "leaf_chunk_tokens": 1,
    "dynamic_leaf_chunk_max": 1,
    "condensation_fanin": 2,
    "condensation_min_fanin": 2,
    "incremental_max_depth": -1,
    "output_reserve": 0,
    "cache_ttl_seconds": 0,
    "summary_prefix_target_tokens": 0,
}
_ENUM_OVERRIDE_VALUES = {
    "cache_economics": frozenset({"discounted", "none", "unknown"}),
    "compaction_mode": frozenset({"inline", "deferred"}),
}


def _coerce_policy_boolean(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} must be a boolean or 'true'/'false'")


def _validate_policy_overrides(overrides: dict[str, Any], *, prefix: str) -> None:
    unknown_override = set(overrides) - _RULE_OVERRIDE_FIELDS
    if unknown_override:
        raise ValueError(f"unknown override field: {sorted(unknown_override)[0]}")
    for field_name, value in overrides.items():
        if field_name in _RATIO_OVERRIDE_FIELDS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{prefix} {field_name} must be a number")
            if not math.isfinite(float(value)) or not 0 < float(value) <= 1.0:
                raise ValueError(f"{prefix} {field_name} must be in (0, 1]")
        elif field_name in _BOOLEAN_OVERRIDE_FIELDS:
            _coerce_policy_boolean(field_name, value)
        elif field_name in _INTEGER_OVERRIDE_MINIMUMS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{prefix} {field_name} must be an integer")
            minimum = _INTEGER_OVERRIDE_MINIMUMS[field_name]
            if value < minimum:
                raise ValueError(f"{prefix} {field_name} must be at least {minimum}")
        elif field_name in _ENUM_OVERRIDE_VALUES:
            if not isinstance(value, str) or value.strip().lower() not in _ENUM_OVERRIDE_VALUES[field_name]:
                allowed = ", ".join(sorted(_ENUM_OVERRIDE_VALUES[field_name]))
                raise ValueError(f"{prefix} {field_name} must be one of: {allowed}")


def _normalize_policy_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    for field_name in _BOOLEAN_OVERRIDE_FIELDS & set(normalized):
        normalized[field_name] = _coerce_policy_boolean(field_name, normalized[field_name])
    if "cache_friendly_condensation" in normalized:
        alias_value = normalized.pop("cache_friendly_condensation")
        normalized.setdefault("cache_friendly_condensation_enabled", alias_value)
    for field_name in _ENUM_OVERRIDE_VALUES.keys() & normalized.keys():
        normalized[field_name] = normalized[field_name].strip().lower()
    return normalized


def validate_policy_rules(rules: Optional[list[dict[str, Any]]]) -> None:
    """Validate structured rules without depending on runtime metadata."""
    if rules is None:
        return
    if not isinstance(rules, list):
        raise ValueError("policy_rules must be a list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"policy rule {index} must be an object")
        unknown_top = set(rule) - {"name", "match", "overrides"}
        if unknown_top:
            raise ValueError(
                f"policy rule {index} has unknown field: {sorted(unknown_top)[0]}"
            )
        if "name" in rule and not isinstance(rule["name"], str):
            raise ValueError(f"policy rule {index} name must be a string")
        match = rule.get("match", {})
        overrides = rule.get("overrides", {})
        if not isinstance(match, dict) or not isinstance(overrides, dict):
            raise ValueError(f"policy rule {index} match/overrides must be objects")
        unknown_match = set(match) - _RULE_MATCH_FIELDS
        if unknown_match:
            raise ValueError(
                f"unknown match field: {sorted(unknown_match)[0]}"
            )
        for field_name, value in match.items():
            if field_name in {"min_context_window", "max_context_window"}:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"policy rule {index} {field_name} must be a positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"policy rule {index} {field_name} must be a non-empty string")
        minimum = match.get("min_context_window")
        maximum = match.get("max_context_window")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"policy rule {index} min_context_window exceeds max_context_window")
        _validate_policy_overrides(overrides, prefix=f"policy rule {index}")


def _pattern_match(value: str, pattern: Any) -> tuple[bool, bool]:
    normalized_value = (value or "").strip().lower()
    normalized_pattern = str(pattern or "").strip().lower()
    if not normalized_pattern:
        return False, False
    if any(char in normalized_pattern for char in "*?["):
        return fnmatchcase(normalized_value, normalized_pattern), False
    return normalized_value == normalized_pattern, normalized_value == normalized_pattern


def _matching_policy_rule(
    rules: Optional[list[dict[str, Any]]],
    *,
    provider: str,
    model: str,
    route: str,
    context_length: int,
    session_id: str,
    platform: str,
) -> tuple[dict[str, Any], str, str] | None:
    validate_policy_rules(rules)
    candidates: list[tuple[int, int, dict[str, Any], str, str]] = []
    for index, rule in enumerate(rules or []):
        match = rule.get("match") or {}
        score = 0
        selectors: list[str] = []
        matched = True
        for field_name, runtime_value in (
            ("provider", provider),
            ("model", model),
            ("route", route),
        ):
            if field_name not in match:
                continue
            field_matched, exact = _pattern_match(runtime_value, match[field_name])
            if not field_matched:
                matched = False
                break
            score += 1_000 if exact else 700
            selectors.append(f"{field_name}={match[field_name]}")
        if not matched:
            continue
        for field_name, runtime_value in (
            ("session", session_id),
            ("platform", platform),
        ):
            if field_name not in match:
                continue
            field_matched, exact = _pattern_match(runtime_value, match[field_name])
            if not field_matched:
                matched = False
                break
            score += 400 if exact else 300
            selectors.append(f"{field_name}={match[field_name]}")
        if not matched:
            continue
        minimum = match.get("min_context_window")
        maximum = match.get("max_context_window")
        if minimum is not None and context_length < int(minimum):
            continue
        if maximum is not None and context_length > int(maximum):
            continue
        if minimum is not None or maximum is not None:
            score += 100
            selectors.append(f"context={minimum or '*'}..{maximum or '*'}")
        name = str(rule.get("name") or f"rule-{index + 1}")
        candidates.append((score, -index, rule, name, ", ".join(selectors) or "match-all"))
    if not candidates:
        return None
    _score, _order, selected, name, reason = max(candidates, key=lambda item: (item[0], item[1]))
    return selected, name, reason


def resolve_policy(
    *,
    model: str,
    provider: str = "",
    route: str = "",
    context_length: int = 0,
    session_id: str = "",
    platform: str = "",
    policy_rules: Optional[list[dict[str, Any]]] = None,
    model_policies: Optional[dict[str, dict[str, Any]]] = None,
    # Legacy backward-compatible inputs:
    context_threshold: float = 0.35,
    model_thresholds: Optional[dict[str, float]] = None,
    emergency_pressure_ratio: float = DEFAULT_EMERGENCY_RATIO,
    max_assembly_tokens: int = 0,
    reserve_tokens_floor: int = 0,
    summary_model: str = "",
    summary_fallback_models: Optional[tuple[str, ...]] = None,
    fresh_tail_count: int = 32,
    fresh_tail_max_tokens: int = 0,
    leaf_chunk_tokens: int = 20_000,
    dynamic_leaf_chunk_enabled: bool = False,
    dynamic_leaf_chunk_max: int = 40_000,
    condensation_fanin: int = 4,
    condensation_min_fanin: int = 2,
    incremental_max_depth: int = 3,
    cache_friendly_condensation_enabled: bool = False,
    output_reserve: int = 0,
    cache_economics: str = "unknown",
    compaction_mode: str = "inline",
    cache_ttl_seconds: int = 0,
    full_sweep_compaction_enabled: bool = False,
    summary_prefix_target_tokens: int = 0,
    # Config source tracking:
    context_threshold_source: str = "manual_or_default",
    config_sources: Optional[dict[str, str]] = None,
) -> ModelCompactionPolicy:
    """Resolve a ``ModelCompactionPolicy`` from model, provider, and config.

    Resolution priority (highest first):

    1. Explicit ``model_thresholds`` longest match (legacy per-model override)
    2. Built-in ``_DEFAULT_CUTOVER_OVERRIDES`` longest match
    3. Fallback to ``context_threshold`` (scalar)

    Emergency ratio always comes from ``emergency_pressure_ratio`` (global
    config), not from per-model overrides — this preserves the four-boundary
    independence established in commit 0442900.

    Post-compaction target is derived from the cutover ratio when no assembly
    cap is configured::

        target = cutover_ratio * DEFAULT_TARGET_RATIO

    When an assembly cap IS configured, the target is the minimum of the
    derived target and the assembly cap.

    Preparation threshold is derived as::

        preparation = cutover * DEFAULT_PREPARATION_RATIO
    """
    n_model = _normalize_model_name(model)
    n_provider = _normalize_provider(provider)
    alias = _resolve_alias(n_model)

    # --- Cutover resolution -----------------------------------------------
    cutover_ratio = float(context_threshold)
    cutover_source = context_threshold_source
    matched_explicit_model_threshold = False

    # Priority 1: explicit model_thresholds (legacy dict). Wins over builtins
    # even when the engine already stamped context_threshold_source as
    # model_thresholds:* (update_model resolves threshold before policy).
    if model_thresholds:
        match = _longest_match(model, model_thresholds)
        if match:
            cutover_ratio = match[1]
            cutover_source = f"model_thresholds:{match[0]}"
            matched_explicit_model_threshold = True

    # Priority 2: built-in default overrides — only when no explicit
    # model_thresholds entry selected the cutover.
    if not matched_explicit_model_threshold and (
        cutover_source == context_threshold_source
        or cutover_source == "manual_or_default"
    ):
        match = _longest_match(alias, _DEFAULT_CUTOVER_OVERRIDES)
        if match:
            cutover_ratio = match[1]
            cutover_source = f"builtin:{match[0]}"
        elif cutover_ratio == 0.35:
            # No match and still at default — use conservative unknown default
            cutover_ratio = DEFAULT_UNKNOWN_CUTOVER
            cutover_source = "builtin:unknown_default"

    # --- Derive the other three boundaries --------------------------------
    preparation_ratio = cutover_ratio * DEFAULT_PREPARATION_RATIO
    target_ratio = cutover_ratio * DEFAULT_TARGET_RATIO
    emergency_ratio = float(emergency_pressure_ratio)

    # Structured rules outrank legacy model thresholds and built-ins. Explicit
    # model_policies are compatibility sugar for highest-priority model rules.
    selected_rule = _matching_policy_rule(
        policy_rules,
        provider=provider,
        model=model,
        route=route,
        context_length=context_length,
        session_id=session_id,
        platform=platform,
    )
    selected_model_policy: tuple[str, dict[str, Any]] | None = None
    if model_policies:
        best_key = ""
        normalized_model = _normalize_model_name(model)
        for key, value in model_policies.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("model_policies keys must be non-empty strings")
            if not isinstance(value, dict):
                raise ValueError(f"model policy {key} must be an object")
            _validate_policy_overrides(value, prefix=f"model policy {key}")
            normalized_key = _normalize_model_name(str(key))
            if (
                normalized_key
                and normalized_key in normalized_model
                and len(normalized_key) > len(best_key)
                and isinstance(value, dict)
            ):
                best_key = normalized_key
                selected_model_policy = (str(key), value)

    selected_overrides: dict[str, Any] = {}
    selection_reason = "global fallback"
    route_selector = ""
    session_selector = ""
    if selected_rule is not None:
        rule, rule_name, reason = selected_rule
        selected_overrides = _normalize_policy_overrides(rule.get("overrides") or {})
        cutover_source = f"policy_rule:{rule_name}"
        selection_reason = reason
        match = rule.get("match") or {}
        route_selector = str(match.get("route") or "")
        session_selector = str(match.get("session") or match.get("platform") or "")
    if selected_model_policy is not None:
        selector_name, model_overrides = selected_model_policy
        selected_overrides.update(_normalize_policy_overrides(model_overrides))
        cutover_source = f"model_policies:{selector_name}"
        selection_reason = f"longest model match {selector_name}"

    if "cutover_threshold" in selected_overrides:
        cutover_ratio = float(selected_overrides["cutover_threshold"])
        if "preparation_threshold" not in selected_overrides:
            preparation_ratio = cutover_ratio * DEFAULT_PREPARATION_RATIO
        if "post_compaction_target" not in selected_overrides:
            target_ratio = cutover_ratio * DEFAULT_TARGET_RATIO
    if "preparation_threshold" in selected_overrides:
        preparation_ratio = float(selected_overrides["preparation_threshold"])
    if "post_compaction_target" in selected_overrides:
        target_ratio = float(selected_overrides["post_compaction_target"])
    if "emergency_threshold" in selected_overrides:
        emergency_ratio = float(selected_overrides["emergency_threshold"])
    if selected_overrides and not (
        0 < target_ratio < preparation_ratio <= cutover_ratio < emergency_ratio <= 1.0
    ):
        raise ValueError(
            "Policy invariant violated: expected post_compaction_target < "
            "preparation_threshold <= cutover_threshold < emergency_threshold <= 1.0"
        )

    # Adjust target if assembly caps are configured — but never let the
    # target reach or exceed cutover (the invariant).
    if max_assembly_tokens > 0 and context_length > 0:
        cap_ratio = max_assembly_tokens / context_length
        if cap_ratio < target_ratio:
            target_ratio = max(0.01, cap_ratio * 0.95)  # stay slightly below cap
    if reserve_tokens_floor > 0 and context_length > 0:
        reserve_ratio = (context_length - reserve_tokens_floor) / context_length
        if reserve_ratio < target_ratio:
            target_ratio = max(0.01, reserve_ratio * 0.95)

    # Clamp to invariant: target < preparation
    if target_ratio >= preparation_ratio:
        target_ratio = preparation_ratio * 0.5

    # Clamp to invariant: preparation <= cutover
    if preparation_ratio > cutover_ratio:
        preparation_ratio = cutover_ratio

    # Clamp to invariant: cutover < emergency
    if cutover_ratio >= emergency_ratio:
        # This shouldn't happen with sane defaults, but handle it
        emergency_ratio = min(1.0, cutover_ratio + 0.05)

    # --- Build the policy -------------------------------------------------
    fallbacks = summary_fallback_models or ()
    selector = alias or n_model

    fields = {
        "preparation_threshold": round(preparation_ratio, 6),
        "cutover_threshold": round(cutover_ratio, 6),
        "post_compaction_target": round(target_ratio, 6),
        "emergency_threshold": round(emergency_ratio, 6),
        "assembly_hard_cap": max_assembly_tokens,
        "assembly_reserve_floor": reserve_tokens_floor,
        "summary_model": summary_model,
        "summary_fallback_models": fallbacks,
        "fresh_tail_count": int(selected_overrides.get("fresh_tail_count", fresh_tail_count)),
        "fresh_tail_max_tokens": int(selected_overrides.get("fresh_tail_max_tokens", fresh_tail_max_tokens)),
        "leaf_chunk_tokens": int(selected_overrides.get("leaf_chunk_tokens", leaf_chunk_tokens)),
        "dynamic_leaf_chunk_enabled": selected_overrides.get("dynamic_leaf_chunk_enabled", dynamic_leaf_chunk_enabled),
        "dynamic_leaf_chunk_max": int(selected_overrides.get("dynamic_leaf_chunk_max", dynamic_leaf_chunk_max)),
        "condensation_fanin": int(selected_overrides.get("condensation_fanin", condensation_fanin)),
        "condensation_min_fanin": int(selected_overrides.get("condensation_min_fanin", condensation_min_fanin)),
        "incremental_max_depth": int(selected_overrides.get("incremental_max_depth", incremental_max_depth)),
        "cache_friendly_condensation_enabled": selected_overrides.get("cache_friendly_condensation_enabled", cache_friendly_condensation_enabled),
        "output_reserve": int(selected_overrides.get("output_reserve", output_reserve)),
        "cache_economics": str(selected_overrides.get("cache_economics", cache_economics)).lower(),
        "compaction_mode": str(selected_overrides.get("compaction_mode", compaction_mode)).lower(),
        "cache_ttl_seconds": int(selected_overrides.get("cache_ttl_seconds", cache_ttl_seconds)),
        "full_sweep_compaction_enabled": selected_overrides.get("full_sweep_compaction_enabled", full_sweep_compaction_enabled),
        "summary_prefix_target_tokens": int(selected_overrides.get("summary_prefix_target_tokens", summary_prefix_target_tokens)),
    }

    fingerprint = _compute_fingerprint(fields)

    policy = ModelCompactionPolicy(
        **fields,
        source=cutover_source,
        model_selector=selector,
        provider_selector=n_provider,
        route_selector=route_selector,
        session_selector=session_selector,
        selection_reason=selection_reason,
        fingerprint=fingerprint,
    )
    return policy
