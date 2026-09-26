# Packaging and distribution posture

## Current decision

`hermes-lcm` intentionally remains a clone-or-symlink Hermes user plugin for now. The supported install path is:

```bash
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/plugins/hermes-lcm
```

For profile-specific installs, clone under `~/.hermes/profiles/<profile>/plugins/hermes-lcm`. For development checkouts, `scripts/install.sh` creates a profile-aware symlink into the active Hermes plugin directory and refuses to overwrite an existing checkout or unrelated symlink.

## Why not pip-style packaging yet?

The repository is a Hermes plugin, not a standalone Python application. Runtime discovery currently depends on:

- `plugin.yaml` declaring the plugin name and registered tools
- the repo root containing `__init__.py` for Hermes plugin registration
- the operator placing or symlinking the checkout into Hermes' plugin search path
- no plugin-resolved third-party runtime dependencies beyond the host APIs, with optional features such as `tiktoken`, `regex`, NumPy, and FastEmbed supplied by the host environment

There is no packaging today, and that is deliberate until Hermes plugin
packaging/discovery has a stable target for pip-installed plugins. The
minimal `[project]` metadata in `pyproject.toml` is not a packaging story:
it exists so Hermes' package manager can snapshot the plugin into a uv
workspace (a name-only `[project]` table is rejected by uv >= 0.12).
Adding generic Python packaging before the host install contract is clear
would create a second install story without making first-run activation
simpler.

## Host-owned dependency assurance

[`dependency-contract.json`](../dependency-contract.json) is the authoritative,
versioned host-owned dependency boundary. It records supported Hermes Agent and
Python versions, the update owner, and every external import observed in shipped
plugin, operator-script, and benchmarking sources. Validate it with:

```bash
python scripts/validate_dependency_contract.py --report-environment
```

The validator fails when runtime imports and the contract drift. Its local
availability/version report is not a lock, SBOM, or vulnerability scan; those
remain the responsibility of the resolved Hermes Agent host environment. See
[Dependency assurance](dependency-assurance.md) for update and scanner evidence
requirements.

## Next packaging step

Make packaging a separate implementation lane only when one of these is true:

1. Hermes Agent documents a stable pip/distribution entrypoint for plugins.
2. Users need version-pinned installs without direct git checkouts.
3. Release automation needs packaged artifacts beyond GitHub tags/releases.

The narrow next step would be a `[build-system]` plus tests that prove a packaged install still exposes `hermes-lcm`, context engine `lcm`, and all 15 LCM tools through `hermes plugins`. The minimal `[project]` table added for PM workspace snapshots is not that step: it carries no `[build-system]`, so the plugin stays a virtual workspace member. Until packaging proper lands, clone/symlink remains the documented path.

## Current install and update references

- Quickstart: [README](../README.md)
- Detailed install/update/verify: [Operator guide](operator-guide.md)
- Standalone install script contract: [`tests/test_packaging_install.py`](../tests/test_packaging_install.py)
