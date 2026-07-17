"""Issue #13 read-only diagnostics CLI regressions."""

from __future__ import annotations

import json
from email.parser import Parser
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

import hermes_lcm.lcm_cli as lcm_cli
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.db_bootstrap import SCHEMA_VERSION
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
    assert payload["schema_version"] == SCHEMA_VERSION == 10
    assert payload["supported_schema_version"] == SCHEMA_VERSION
    assert payload["counts"]["messages"] == 2
    assert payload["read_only"] is True


def test_cli_reads_legacy_v9_without_migrating_it(tmp_path):
    db = _seed(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER lcm_schema_version_monotonic")
    conn.execute(
        "ALTER TABLE lcm_lifecycle_state DROP COLUMN rollover_carry_over_context"
    )
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name = 'v10_rollover_carry_policy'"
    )
    conn.execute(
        "UPDATE metadata SET value = '9' WHERE key = 'schema_version'"
    )
    conn.commit()
    conn.close()

    result = _run(db, "status")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 9
    assert payload["supported_schema_version"] == SCHEMA_VERSION == 10
    check = sqlite3.connect(db)
    try:
        assert check.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("9",)
        assert "rollover_carry_over_context" not in {
            row[1]
            for row in check.execute(
                "PRAGMA table_info(lcm_lifecycle_state)"
            ).fetchall()
        }
        assert check.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = 'v10_rollover_carry_policy'"
        ).fetchone() is None
    finally:
        check.close()


def test_cli_refuses_schema_newer_than_authoritative_version_without_writes(tmp_path):
    db = tmp_path / "future-cli.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    result = _run(db, "status")

    assert result.returncode == lcm_cli.EXIT_DATABASE
    error = json.loads(result.stdout)["error"]
    assert f"schema version {SCHEMA_VERSION + 1}" in error
    assert f"supports (v{SCHEMA_VERSION})" in error
    check = sqlite3.connect(db)
    try:
        assert check.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == (str(SCHEMA_VERSION + 1),)
        assert check.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall() == [("metadata",)]
    finally:
        check.close()


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


def test_cli_message_tail_cursor_pages_toward_older_rows(tmp_path):
    db = _seed(tmp_path)
    conn = sqlite3.connect(db)
    conn.executemany(
        """INSERT INTO messages
           (session_id, source, role, content, timestamp, token_estimate, pinned)
           VALUES ('cli-session', 'test', 'user', ?, ?, 1, 0)""",
        [(f"tail-{index}", float(index)) for index in range(6)],
    )
    conn.commit()
    conn.close()

    first = json.loads(_run(db, "messages", "tail", "--limit", "3").stdout)
    first_ids = [item["store_id"] for item in first["items"]]
    assert first_ids == sorted(first_ids)
    assert first["next_cursor"] == min(first_ids)

    second = json.loads(_run(
        db,
        "messages",
        "tail",
        "--limit",
        "3",
        "--after-store-id",
        str(first["next_cursor"]),
    ).stdout)
    second_ids = [item["store_id"] for item in second["items"]]
    assert second_ids
    assert max(second_ids) < min(first_ids)
    assert not set(first_ids).intersection(second_ids)


def test_cli_frontier_items_are_sql_limited_before_materialization(tmp_path):
    db = _seed(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO lcm_active_frontiers
           (conversation_id, generation, session_id, source_end_store_id,
            policy_fingerprint, route_fingerprint, created_at, updated_at)
           VALUES ('wide-frontier', 1, 'cli-session', 2000, '', '', 1, 1)"""
    )
    conn.executemany(
        """INSERT INTO lcm_frontier_items
           (conversation_id, generation, ordinal, kind, ref_id, source_start, source_end)
           VALUES ('wide-frontier', 1, ?, 'message', ?, ?, ?)""",
        [(index, index + 1, index + 1, index + 1) for index in range(2000)],
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    from argparse import Namespace
    payload = lcm_cli._frontier(
        conn,
        Namespace(conversation_id="wide-frontier"),
    )
    conn.close()
    assert len(payload["items"]) == lcm_cli._FRONTIER_ITEMS_LIMIT
    assert payload["items_truncated"] is True


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


def test_cli_full_output_is_recursively_redacted_and_hard_bounded(tmp_path):
    db = _seed(tmp_path)
    secret = "sk-proj-cli-super-secret"
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE messages SET content=? WHERE store_id=(SELECT MIN(store_id) FROM messages)",
        (f'api_key: {secret}\n' + ("x" * 250_000),),
    )
    conn.commit()
    conn.close()

    result = _run(db, "messages", "list", "--full", "--limit", "200")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert "[REDACTED" in serialized
    assert payload["items"][0]["content_truncated"] is True
    assert len(result.stdout) < 150_000


def test_cli_config_output_redacts_nested_credentials_and_bounds_values(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    secret = "config-super-secret"
    (hermes_home / "config.yaml").write_text(
        "lcm:\n"
        "  custom_instructions: 'password: " + secret + " " + ("z" * 250_000) + "'\n"
        "  nested:\n"
        "    api_key: 'nested-secret'\n",
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
    assert secret not in result.stdout
    assert "nested-secret" not in result.stdout
    assert "[REDACTED" in result.stdout
    assert len(result.stdout) < 150_000


def test_cli_config_show_and_get_redact_unlabeled_standalone_credentials(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    credentials = {
        "openai": "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "github_classic": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
        "github_fine_grained": "github_pat_11AA0_example0123456789abcdef0123456789",
    }
    config = {
        "lcm": {
            "custom_instructions": (
                "Use these directly: "
                + credentials["openai"]
                + " and "
                + credentials["github_classic"]
            ),
            "nested": {
                "provider_values": [credentials["aws"], {"raw": credentials["github_fine_grained"]}],
            },
        }
    }
    (hermes_home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    commands = [
        ("config", "show"),
        ("config", "get", "custom_instructions"),
        ("config", "get", "nested"),
    ]
    for command in commands:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "lcm_cli.py"),
                "--hermes-home",
                str(hermes_home),
                *command,
            ],
            cwd=os.fspath(tmp_path),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        for credential in credentials.values():
            assert credential not in result.stdout
        assert "[REDACTED" in result.stdout


def test_cli_standalone_credential_patterns_preserve_non_secret_lookalikes(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    lookalikes = [
        "sk-short",
        "sk-proj-documentation-placeholder",
        "AKIA prefix documentation",
        "ghp_short",
        "github_pat_example",
    ]
    (hermes_home / "config.yaml").write_text(
        json.dumps({"lcm": {"custom_instructions": " | ".join(lookalikes)}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "lcm_cli.py"),
            "--hermes-home",
            str(hermes_home),
            "config",
            "get",
            "custom_instructions",
        ],
        cwd=os.fspath(tmp_path),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for lookalike in lookalikes:
        assert lookalike in result.stdout
    assert "[REDACTED" not in result.stdout


def test_cli_redaction_bounds_pathological_private_key_scan_and_keeps_detection():
    header = "-----BEGIN PRIVATE KEY-----\n"
    pathological = (header * ((300 * 1024 // len(header)) + 1))[:300 * 1024]
    private_key_body = "LEGITIMATE_PRIVATE_KEY_BODY_0123456789"
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        + private_key_body
        + "\n-----END PRIVATE KEY-----"
    )
    standalone = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"

    started = time.monotonic()
    protected = lcm_cli._sanitize_output({
        "pathological": pathological,
        "private_key": private_key,
        "standalone": standalone,
    })
    elapsed = time.monotonic() - started
    serialized = json.dumps(protected)

    assert elapsed < 2.0
    assert private_key_body not in serialized
    assert standalone not in serialized
    assert serialized.count("[REDACTED by hermes-lcm CLI]") >= 2
    assert len(protected["pathological"]) <= lcm_cli._CLI_MAX_PREVIEW_CHARS


def test_cli_200_adversarial_private_key_keys_are_linear_and_still_redact():
    header = "-----BEGIN PRIVATE KEY-----\n"
    adversarial_key = (header * 800)[:20_000]
    payload = {
        "legitimate": (
            "-----BEGIN PRIVATE KEY-----\n"
            "REAL_PRIVATE_KEY_MATERIAL\n"
            "-----END PRIVATE KEY-----"
        )
    }
    payload.update({
        f"{index:03d}-{adversarial_key}": "ordinary"
        for index in range(199)
    })

    started = time.monotonic()
    protected = lcm_cli._sanitize_output(payload)
    elapsed = time.monotonic() - started
    serialized = json.dumps(protected)

    assert elapsed < 2.0
    assert "REAL_PRIVATE_KEY_MATERIAL" not in serialized
    assert "[REDACTED by hermes-lcm CLI]" in serialized


def test_cli_preview_bounds_apply_to_summary_and_prepared_show(tmp_path):
    db = _seed(tmp_path)
    summary = _run(db, "summaries", "show", "1", "--preview-chars", "20001")
    prepared = _run(db, "prepared-batches", "show", "1", "--preview-chars", "20001")
    assert summary.returncode == 2
    assert prepared.returncode == 2


def test_cli_exit_codes_distinguish_not_found_and_invalid_input(tmp_path):
    db = _seed(tmp_path)
    missing = _run(db, "summaries", "show", "999999")
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["error"] == "not found"

    invalid = _run(db, "messages", "list", "--limit", "0")
    assert invalid.returncode == 2


def test_packaged_console_script_runs_without_gateway(tmp_path):
    db = _seed(tmp_path)
    engine = LCMEngine(config=LCMConfig(database_path=str(db)))
    engine.on_session_start(
        "maintenance-session",
        conversation_id="maintenance-conversation",
        platform="test",
    )
    store_id = engine._store.append(
        "maintenance-session",
        {"role": "user", "content": "maintenance source"},
        conversation_id="maintenance-conversation",
    )
    leaf_id = engine._dag.add_node(SummaryNode(
        session_id="maintenance-session",
        depth=0,
        summary="maintenance leaf",
        token_count=3,
        source_token_count=3,
        source_ids=[store_id],
        source_type="messages",
        created_at=1.0,
    ))
    parent_id = engine._dag.add_node(SummaryNode(
        session_id="maintenance-session",
        depth=1,
        summary="maintenance parent",
        token_count=3,
        source_token_count=3,
        source_ids=[leaf_id],
        source_type="nodes",
        created_at=2.0,
    ))
    engine._frontier.ensure_frontier(
        "maintenance-conversation",
        "maintenance-session",
        source_end_store_id=store_id,
    )
    engine._frontier.set_frontier_items("maintenance-conversation", 1, [{
        "kind": "node",
        "ref_id": parent_id,
        "source_start": store_id,
        "source_end": store_id,
    }])
    engine.shutdown()
    target = tmp_path / "venv"
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", "build", "*.egg-info", "__pycache__", "CODEX_REPORT.md"
        ),
    )
    wheelhouse = tmp_path / "wheelhouse"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    create_venv = subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert create_venv.returncode == 0, create_venv.stderr
    wheel = next(wheelhouse.glob("hermes_lcm-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
    requirements = Parser().parsestr(metadata).get_all("Requires-Dist", [])
    assert any(requirement.replace(" ", "") == "PyYAML>=6.0" for requirement in requirements)
    install = subprocess.run(
        [str(target / "bin" / "python"), "-m", "pip", "install", "--no-deps", str(wheel)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    result = subprocess.run(
        [
            str(target / "bin" / "hermes-lcm"),
            "--database",
            str(db),
            "status",
        ],
        env={**os.environ},
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["read_only"] is True

    # Populate the otherwise isolated venv with the installed runtime's actual
    # PyYAML package, then exercise the wheel-only console script's config path.
    purelib = subprocess.run(
        [str(target / "bin" / "python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    shutil.copytree(Path(yaml.__file__).parent, Path(purelib) / "yaml")
    hermes_home = tmp_path / "clean-hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "lcm:\n  context_threshold: 0.42\n", encoding="utf-8"
    )
    config_result = subprocess.run(
        [
            str(target / "bin" / "hermes-lcm"),
            "--hermes-home",
            str(hermes_home),
            "config",
            "get",
            "context_threshold",
        ],
        env={**os.environ},
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert config_result.returncode == 0, config_result.stderr
    assert json.loads(config_result.stdout)["value"] == 0.42

    maintenance = subprocess.run(
        [
            str(target / "bin" / "hermes-lcm"),
            "--database",
            str(db),
            "maintenance",
            "plan",
            "dissolve",
            "--conversation-id",
            "maintenance-conversation",
            "--node-id",
            str(parent_id),
        ],
        env={**os.environ},
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert maintenance.returncode == 0, maintenance.stderr
    maintenance_payload = json.loads(maintenance.stdout)
    assert maintenance_payload["dry_run"] is True
    assert maintenance_payload["operation"] == "dissolve"

    preflight = subprocess.run(
        [
            str(target / "bin" / "hermes-lcm"),
            "--pretty",
            "activation-preflight",
        ],
        env={
            **os.environ,
            "HOME": str(tmp_path / "preflight-home"),
            "PYTHONPATH": os.pathsep.join([
                str(HERMES_AGENT_ROOT),
                *[
                    entry for entry in sys.path
                    if entry and ("site-packages" in entry or "dist-packages" in entry)
                ],
            ]),
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
    site_packages = next((target / "lib").glob("python*/site-packages"))
    assert (site_packages / "hermes_lcm" / "plugin.yaml").is_file()
    assert (
        site_packages / "hermes_lcm" / "docs" / "host-activation-contract.md"
    ).is_file()
