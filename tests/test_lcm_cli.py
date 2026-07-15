"""Issue #13 read-only diagnostics CLI regressions."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT_ROOT = Path("/home/ben/hermes-agent-gil-pr")


def _seed(tmp_path):
    db = tmp_path / "cli.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db)))
    engine.on_session_start(
        "cli-session",
        conversation_id="cli-conversation",
        platform="test",
        context_length=100_000,
    )
    engine.ingest([
        {"role": "user", "content": "first CLI diagnostic message"},
        {"role": "assistant", "content": "second CLI diagnostic message"},
    ])
    engine.shutdown()
    return db


def _run(db, *args):
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "lcm_cli.py"),
            "--database",
            str(db),
            *args,
        ],
        cwd=os.fspath(db.parent),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(REPO_ROOT),
                    "/home/ben/hermes-agent-gil-pr",
                ]
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_status_is_json_first_and_works_without_gateway(tmp_path):
    db = _seed(tmp_path)
    result = _run(db, "status")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["database_path"] == str(db)
    assert payload["schema_version"] == 8
    assert payload["counts"]["messages"] == 2
    assert payload["read_only"] is True


def test_cli_message_list_is_preview_only_and_keyset_paginated(tmp_path):
    db = _seed(tmp_path)
    first = _run(
        db,
        "messages",
        "list",
        "--session-id",
        "cli-session",
        "--limit",
        "1",
        "--preview-chars",
        "8",
    )
    assert first.returncode == 0
    payload = json.loads(first.stdout)
    assert len(payload["items"]) == 1
    assert payload["items"][0]["content"] == "first CL"
    assert payload["items"][0]["content_truncated"] is True
    cursor = payload["next_cursor"]

    second = _run(
        db,
        "messages",
        "list",
        "--session-id",
        "cli-session",
        "--limit",
        "1",
        "--after-store-id",
        str(cursor),
    )
    assert second.returncode == 0
    second_payload = json.loads(second.stdout)
    assert second_payload["items"][0]["store_id"] > cursor


def test_cli_does_not_migrate_or_create_tables(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    result = _run(db, "status")
    assert result.returncode == 5
    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == []
    finally:
        conn.close()


def test_cli_config_show_exposes_only_lcm_section(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "provider:\n  api_key: secret\nlcm:\n  context_threshold: 0.4\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "lcm_cli.py"),
            "--hermes-home",
            str(hermes_home),
            "config",
            "show",
        ],
        cwd=os.fspath(tmp_path),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "config_path": str(hermes_home / "config.yaml"),
        "lcm": {"context_threshold": 0.4},
        "read_only": True,
    }
    assert "secret" not in result.stdout


def test_cli_exit_codes_distinguish_not_found_and_invalid_input(tmp_path):
    db = _seed(tmp_path)
    missing = _run(db, "summaries", "show", "999999")
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["error"] == "not found"

    invalid = _run(db, "messages", "list", "--limit", "0")
    assert invalid.returncode == 2


def test_packaged_console_script_runs_without_gateway(tmp_path):
    db = _seed(tmp_path)
    target = tmp_path / "site"
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", "build", "*.egg-info", "__pycache__", "CODEX_REPORT.md"
        ),
    )
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_lcm.lcm_cli",
            "--database",
            str(db),
            "status",
        ],
        env={**os.environ, "PYTHONPATH": str(target)},
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["read_only"] is True

    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_lcm.lcm_cli",
            "--pretty",
            "activation-preflight",
        ],
        env={
            **os.environ,
            "HOME": str(tmp_path / "preflight-home"),
            "PYTHONPATH": os.pathsep.join([str(target), str(HERMES_AGENT_ROOT)]),
        },
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    preflight_payload = json.loads(preflight.stdout)
    assert preflight_payload["status"] == "pass"
    assert preflight_payload["host_activation_ordering_verified"] is False
    assert set(preflight_payload["tool_names"]) == {
        "lcm_grep",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_focus",
        "lcm_status",
        "lcm_inspect",
        "lcm_doctor",
    }
    assert (target / "hermes_lcm" / "plugin.yaml").is_file()
    assert (
        target / "hermes_lcm" / "docs" / "host-activation-contract.md"
    ).is_file()
