"""Regressions for issues #9, #18, and #21 structured route policies."""

from __future__ import annotations

import pytest

from hermes_lcm.config import LCMConfig, _parse_model_policies_env
from hermes_lcm.engine import LCMEngine
from hermes_lcm.policy import resolve_policy


def _resolve(rules, **overrides):
    kwargs = {
        "model": "route-model",
        "provider": "proxy",
        "route": "responses",
        "context_length": 100_000,
        "session_id": "discord:project-alpha",
        "platform": "discord",
        "context_threshold": 0.70,
        "policy_rules": rules,
        "fresh_tail_count": 32,
    }
    kwargs.update(overrides)
    return resolve_policy(**kwargs)


def test_exact_model_and_route_rule_outranks_session_and_context_rules():
    policy = _resolve(
        [
            {
                "name": "broad-window",
                "match": {"min_context_window": 50_000},
                "overrides": {"cutover_threshold": 0.65},
            },
            {
                "name": "session-rule",
                "match": {"session": "discord:*"},
                "overrides": {"cutover_threshold": 0.55},
            },
            {
                "name": "exact-route",
                "match": {
                    "provider": "proxy",
                    "model": "route-model",
                    "route": "responses",
                },
                "overrides": {
                    "preparation_threshold": 0.30,
                    "cutover_threshold": 0.40,
                    "post_compaction_target": 0.18,
                    "emergency_threshold": 0.90,
                    "fresh_tail_count": 12,
                    "fresh_tail_max_tokens": 8_000,
                    "leaf_chunk_tokens": 9_000,
                    "condensation_fanin": 3,
                    "condensation_min_fanin": 2,
                    "output_reserve": 4_096,
                    "cache_economics": "none",
                    "compaction_mode": "inline",
                },
            },
        ]
    )

    assert policy.source == "policy_rule:exact-route"
    assert policy.cutover_threshold == 0.40
    assert policy.preparation_threshold == 0.30
    assert policy.post_compaction_target == 0.18
    assert policy.fresh_tail_count == 12
    assert policy.fresh_tail_max_tokens == 8_000
    assert policy.condensation_min_fanin == 2
    assert policy.cache_economics == "none"
    assert policy.compaction_mode == "inline"


def test_session_rule_outranks_context_window_rule_and_ties_use_order():
    policy = _resolve(
        [
            {
                "name": "first-session",
                "match": {"session": "discord:*"},
                "overrides": {"cutover_threshold": 0.51},
            },
            {
                "name": "second-session",
                "match": {"session": "discord:*"},
                "overrides": {"cutover_threshold": 0.52},
            },
            {
                "name": "window",
                "match": {"min_context_window": 50_000},
                "overrides": {"cutover_threshold": 0.61},
            },
        ]
    )

    assert policy.source == "policy_rule:first-session"
    assert policy.cutover_threshold == 0.51


def test_cache_economics_changes_fingerprint_without_cache_telemetry():
    discounted = _resolve(
        [{
            "name": "discounted",
            "match": {"route": "responses"},
            "overrides": {"cache_economics": "discounted", "compaction_mode": "deferred"},
        }]
    )
    no_discount = _resolve(
        [{
            "name": "none",
            "match": {"route": "responses"},
            "overrides": {"cache_economics": "none", "compaction_mode": "inline"},
        }]
    )

    assert discounted.fingerprint != no_discount.fingerprint
    assert discounted.cache_economics == "discounted"
    assert no_discount.cache_economics == "none"


@pytest.mark.parametrize(
    "rule,match",
    [
        (
            {"name": "bad", "match": {"unknown": "x"}, "overrides": {}},
            "unknown match field",
        ),
        (
            {"name": "bad", "match": {}, "overrides": {"mystery": 1}},
            "unknown override field",
        ),
        (
            {
                "name": "bad-order",
                "match": {},
                "overrides": {
                    "post_compaction_target": 0.6,
                    "preparation_threshold": 0.5,
                },
            },
            "Policy invariant",
        ),
    ],
)
def test_invalid_policy_rules_fail_clearly(rule, match):
    with pytest.raises(ValueError, match=match):
        _resolve([rule])


def test_model_policy_env_parser_supports_issue_21_format():
    parsed = _parse_model_policies_env(
        "glm-5.2:inline:0.35:false,claude-sonnet-4:deferred:0.75:true:300"
    )
    assert parsed == {
        "glm-5.2": {
            "compaction_mode": "inline",
            "cutover_threshold": 0.35,
            "cache_friendly_condensation_enabled": False,
            "cache_ttl_seconds": 0,
        },
        "claude-sonnet-4": {
            "compaction_mode": "deferred",
            "cutover_threshold": 0.75,
            "cache_friendly_condensation_enabled": True,
            "cache_ttl_seconds": 300,
        },
    }


def test_engine_uses_resolved_rule_for_live_cutover_and_status(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "structured-policy.db"),
        policy_rules=[
            {
                "name": "proxy-no-cache",
                "match": {"provider": "proxy", "route": "responses"},
                "overrides": {
                    "preparation_threshold": 0.25,
                    "cutover_threshold": 0.35,
                    "post_compaction_target": 0.15,
                    "emergency_threshold": 0.95,
                    "cache_economics": "none",
                    "compaction_mode": "inline",
                },
            }
        ],
    )
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            "route-model",
            100_000,
            provider="proxy",
            api_mode="responses",
        )
        status = engine.get_status()["compaction_policy"]

        assert engine.threshold_tokens == 35_000
        assert status["source"] == "policy_rule:proxy-no-cache"
        assert status["cache_economics"] == "none"
        assert status["compaction_mode"] == "inline"
        assert status["cutover_tokens"] == 35_000
        assert status["post_compaction_target_tokens"] == 15_000
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "rule,match",
    [
        ({"match": {"min_context_window": "large"}, "overrides": {}}, "min_context_window"),
        ({"match": {}, "overrides": {"cutover_threshold": "0.5"}}, "cutover_threshold"),
        ({"match": {}, "overrides": {"fresh_tail_count": -1}}, "fresh_tail_count"),
        ({"match": {}, "overrides": {"condensation_fanin": True}}, "condensation_fanin"),
        ({"match": {}, "overrides": {"cache_economics": "cheap"}}, "cache_economics"),
        ({"match": {}, "overrides": {"compaction_mode": "later"}}, "compaction_mode"),
        ({"match": {}, "overrides": {"dynamic_leaf_chunk_enabled": "sometimes"}}, "dynamic_leaf_chunk_enabled"),
    ],
)
def test_policy_rules_reject_wrong_types_and_ranges(rule, match):
    with pytest.raises(ValueError, match=match):
        _resolve([rule])


def test_quoted_false_boolean_is_parsed_without_enabling_strategy():
    policy = _resolve([{
        "match": {"route": "responses"},
        "overrides": {
            "dynamic_leaf_chunk_enabled": "false",
            "cache_friendly_condensation_enabled": "false",
            "full_sweep_compaction_enabled": "false",
        },
    }])

    assert policy.dynamic_leaf_chunk_enabled is False
    assert policy.cache_friendly_condensation_enabled is False
    assert policy.full_sweep_compaction_enabled is False


def test_legacy_model_policy_layers_over_structured_rule_without_erasing_it():
    policy = _resolve(
        [{
            "name": "route-base",
            "match": {"route": "responses"},
            "overrides": {
                "fresh_tail_count": 7,
                "dynamic_leaf_chunk_enabled": True,
            },
        }],
        model_policies={
            "route-model": {
                "cutover_threshold": 0.60,
                "cache_friendly_condensation_enabled": False,
            },
        },
    )

    assert policy.cutover_threshold == 0.60
    assert policy.fresh_tail_count == 7
    assert policy.dynamic_leaf_chunk_enabled is True
    assert policy.cache_friendly_condensation_enabled is False


def test_output_reserve_constrains_live_runtime_assembly_cap(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "output-reserve.db"),
        policy_rules=[{
            "match": {"model": "reserve-model"},
                "overrides": {"output_reserve": 90_000},
        }],
    )
    engine = LCMEngine(config=config)
    try:
        engine.update_model("reserve-model", 100_000, provider="test")
        assert engine._compaction_policy.output_reserve == 90_000
        assert engine._effective_assembly_token_cap() == 10_000
        assert engine._assembly_token_budget() == 10_000
    finally:
        engine.shutdown()
