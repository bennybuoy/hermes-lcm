"""Reproducible trace metrics for compaction-strategy comparisons.

The harness aggregates measurements present in a trace. It never converts
cache-read counters into economic discounts and never fabricates missing
provider billing or retrieval-quality results.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any


def _optional_sum(turns: list[dict[str, Any]], field: str) -> int | None:
    values = [turn[field] for turn in turns if turn.get(field) is not None]
    if len(values) != len(turns):
        return None
    return int(sum(int(value) for value in values))


def _optional_mean(turns: list[dict[str, Any]], field: str) -> float | None:
    values = [float(turn[field]) for turn in turns if turn.get(field) is not None]
    if len(values) != len(turns) or not values:
        return None
    return round(fmean(values), 6)


def summarize_strategy_trace(turns: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(max(0, int(turn.get("input_tokens") or 0)) for turn in turns)
    cache_read_tokens = sum(
        max(0, int(turn.get("cache_read_tokens") or 0)) for turn in turns
    )
    prefixes = [str(turn.get("prefix_fingerprint") or "") for turn in turns]
    prefix_churn = sum(
        1
        for previous, current in zip(prefixes, prefixes[1:])
        if previous and current and previous != current
    )
    cutovers = sum(1 for turn in turns if bool(turn.get("cutover")))
    return {
        "turns": len(turns),
        "total_input_tokens": input_tokens,
        "observed_cache_read_tokens": cache_read_tokens,
        "observed_cache_reuse_ratio": round(
            cache_read_tokens / input_tokens, 6
        ) if input_tokens else 0.0,
        "total_billable_input_tokens": _optional_sum(
            turns, "billable_input_tokens"
        ),
        "billable_input_source": (
            "trace-observed"
            if turns and all(turn.get("billable_input_tokens") is not None for turn in turns)
            else "unavailable"
        ),
        "summary_input_tokens": sum(
            max(0, int(turn.get("summary_input_tokens") or 0)) for turn in turns
        ),
        "cutovers": cutovers,
        "cutover_frequency": round(cutovers / len(turns), 6) if turns else 0.0,
        "prefix_churn_events": prefix_churn,
        "prefix_churn_rate": round(prefix_churn / max(1, len(turns) - 1), 6),
        "mean_retrieval_score": _optional_mean(turns, "retrieval_score"),
        "mean_recall_at_k": _optional_mean(turns, "recall_at_k"),
        "mean_assembly_latency_ms": _optional_mean(turns, "assembly_latency_ms"),
    }


def build_policy_benchmark_report(payload: dict[str, Any]) -> dict[str, Any]:
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("benchmark payload requires a non-empty strategies object")
    report = {
        "schema_version": 1,
        "fixture_kind": str(payload.get("fixture_kind") or "unspecified"),
        "provider_results_claimed": False,
        "economic_contract": (
            "billable input is aggregated only when every trace turn supplies "
            "billable_input_tokens; cache_read_tokens remain observed telemetry"
        ),
        "strategies": {},
    }
    for name in sorted(strategies):
        turns = strategies[name]
        if not isinstance(turns, list) or not all(isinstance(turn, dict) for turn in turns):
            raise ValueError(f"strategy {name!r} must be a list of turn objects")
        report["strategies"][str(name)] = summarize_strategy_trace(turns)
    return report
