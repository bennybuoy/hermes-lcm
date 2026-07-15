"""Issue #13 read-only diagnostics CLI regressions."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


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
            "/tmp/hermes-lcm-codex-all-issues/lcm_cli.py",
            "--database",
            str(db),
            *args,
        ],
        cwd=os.fspath(db.parent),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    "/tmp/hermes-lcm-codex-all-issues",
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
    assert payload["schema_version"] == 7
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
            "/tmp/hermes-lcm-codex-all-issues/lcm_cli.py",
            "--hermes-home",
            str(hermes_home),
            "config",
            "show",
        ],
        cwd=os.fspath(tmp_path),
        env={**os.environ, "PYTHONPATH": "/tmp/hermes-lcm-codex-all-issues"},
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
        "/tmp/hermes-lcm-codex-all-issues",
        source,
        ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"),
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
