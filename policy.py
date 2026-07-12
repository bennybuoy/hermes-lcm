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
    leaf_chunk_tokens: int = 20_000
    dynamic_leaf_chunk_enabled: bool = False
    dynamic_leaf_chunk_max: int = 40_000

    # Condensation:
    condensation_fanin: int = 4
    incremental_max_depth: int = 3
    cache_friendly_condensation_enabled: bool = False

    # Output reserve (tokens to reserve for model output, 0 = disabled):
    output_reserve: int = 0

    # Source reporting:
    source: str = "default"
    model_selector: str = ""
    provider_selector: str = ""

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
        if caps:
            return max(1, min(derived, min(caps)))
        return derived

    def emergency_tokens(self, context_length: int) -> int:
        """Provider-window pressure threshold."""
        import math
        return max(1, math.ceil(context_length * self.emergency_threshold))

    def to_status_dict(self, context_length: int) -> dict[str, Any]:
        """Return a JSON-serializable status dict for lcm_status."""
        return {
            "fingerprint": self.fingerprint,
            "source": self.source,
            "model_selector": self.model_selector,
            "provider_selector": self.provider_selector,
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
            "leaf_chunk_tokens": self.leaf_chunk_tokens,
            "condensation_fanin": self.condensation_fanin,
            "incremental_max_depth": self.incremental_max_depth,
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
        "leaf_chunk_tokens",
        "dynamic_leaf_chunk_enabled",
        "dynamic_leaf_chunk_max",
        "condensation_fanin",
        "incremental_max_depth",
        "cache_friendly_condensation_enabled",
        "output_reserve",
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


def resolve_policy(
    *,
    model: str,
    provider: str = "",
    context_length: int = 0,
    # Legacy backward-compatible inputs:
    context_threshold: float = 0.35,
    model_thresholds: Optional[dict[str, float]] = None,
    emergency_pressure_ratio: float = DEFAULT_EMERGENCY_RATIO,
    max_assembly_tokens: int = 0,
    reserve_tokens_floor: int = 0,
    summary_model: str = "",
    summary_fallback_models: Optional[tuple[str, ...]] = None,
    leaf_chunk_tokens: int = 20_000,
    dynamic_leaf_chunk_enabled: bool = False,
    dynamic_leaf_chunk_max: int = 40_000,
    condensation_fanin: int = 4,
    incremental_max_depth: int = 3,
    cache_friendly_condensation_enabled: bool = False,
    output_reserve: int = 0,
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

    # Priority 1: explicit model_thresholds (legacy dict)
    if model_thresholds:
        match = _longest_match(model, model_thresholds)
        if match:
            cutover_ratio = match[1]
            cutover_source = f"model_thresholds:{match[0]}"

    # Priority 2: built-in default overrides
    if cutover_source == context_threshold_source or cutover_source == "manual_or_default":
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
        "leaf_chunk_tokens": leaf_chunk_tokens,
        "dynamic_leaf_chunk_enabled": dynamic_leaf_chunk_enabled,
        "dynamic_leaf_chunk_max": dynamic_leaf_chunk_max,
        "condensation_fanin": condensation_fanin,
        "incremental_max_depth": incremental_max_depth,
        "cache_friendly_condensation_enabled": cache_friendly_condensation_enabled,
        "output_reserve": output_reserve,
    }

    fingerprint = _compute_fingerprint(fields)

    policy = ModelCompactionPolicy(
        **fields,
        source=cutover_source,
        model_selector=selector,
        provider_selector=n_provider,
        fingerprint=fingerprint,
    )
    return policy