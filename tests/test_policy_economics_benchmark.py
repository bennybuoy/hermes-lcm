from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hermes_lcm.benchmarking.policy_economics import build_policy_benchmark_report


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_policy_benchmark_artifact_is_reproducible():
    fixture = json.loads(
        (ROOT / "benchmarks/fixtures/policy_strategy_synthetic.json").read_text()
    )
    expected = json.loads(
        (ROOT / "benchmarks/artifacts/policy_strategy_synthetic_report.json").read_text()
    )
    assert build_policy_benchmark_report(fixture) == expected


def test_repository_benchmark_script_runs_without_installing_package():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/lcm_policy_benchmark.py"),
            str(ROOT / "benchmarks/fixtures/policy_strategy_synthetic.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = json.loads(
        (ROOT / "benchmarks/artifacts/policy_strategy_synthetic_report.json").read_text()
    )
    assert json.loads(completed.stdout) == expected


def test_cache_reads_never_fill_missing_billable_input():
    report = build_policy_benchmark_report(
        {
            "fixture_kind": "test",
            "strategies": {
                "route": [
                    {"input_tokens": 100, "cache_read_tokens": 99},
                    {"input_tokens": 100, "cache_read_tokens": 99},
                ]
            },
        }
    )
    route = report["strategies"]["route"]
    assert route["observed_cache_read_tokens"] == 198
    assert route["total_billable_input_tokens"] is None
    assert route["billable_input_source"] == "unavailable"
