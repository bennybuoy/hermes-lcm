"""Profile-scoped config path: under a multiplexed gateway the context-local
home override must win over the frozen launch-home HERMES_HOME env var.

Regression: _hermes_config_path() used to read os.environ["HERMES_HOME"],
which stays frozen at the launch profile under multiplex. Every profile's
LCM policy (context_threshold, model_thresholds) then silently resolved from
the default profile's config.yaml — the exact "unbound read is a silent
default-profile leak" bug class (root AGENTS.md: profile scope).
"""

import os
from pathlib import Path

import pytest

import config as lcm_config  # conftest registers the plugin package modules


def test_context_override_wins_over_launch_env(tmp_path, monkeypatch):
    """A→B→A: the context-local home must win while it is set, and the
    launch-home env var must win again after the override resets."""
    launch_home = tmp_path / "launch-home"
    profile_home = tmp_path / "profile-home"
    for home, marker in ((launch_home, "launch"), (profile_home, "profile")):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(f"marker: {marker}\n")

    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    # Baseline (A): without an override, the env var wins.
    assert lcm_config._hermes_config_path() == launch_home / "config.yaml"

    # Scoped (B): context-local override wins over the frozen env var.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(profile_home)
    try:
        assert lcm_config._hermes_config_path() == profile_home / "config.yaml"
    finally:
        reset_hermes_home_override(token)

    # Back to A: override reset restores the launch-home env resolution.
    assert lcm_config._hermes_config_path() == launch_home / "config.yaml"


def test_env_used_when_host_helpers_unavailable(tmp_path, monkeypatch):
    """No host import (or import failure) must fall back to HERMES_HOME env,
    never to a wrong profile — standalone (non-multiplex) deployments and
    plugin-only test runs keep their existing behaviour."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name)),
    )
    path = lcm_config._hermes_config_path()
    assert path == tmp_path / "config.yaml"