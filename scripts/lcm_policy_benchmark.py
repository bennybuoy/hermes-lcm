#!/usr/bin/env python3
"""Aggregate measured compaction-policy traces without pricing assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.policy_economics import build_policy_benchmark_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    report = build_policy_benchmark_report(payload)
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
