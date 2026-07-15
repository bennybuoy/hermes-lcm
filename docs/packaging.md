# Packaging and distribution posture

## Current decision

`hermes-lcm` remains a clone-or-symlink Hermes user plugin for host discovery,
and also ships Python packaging for its standalone diagnostics executable. The
supported host install path is:

```bash
git clone https://github.com/stephenschoettler/hermes-lcm \
  ~/.hermes/plugins/hermes-lcm
```

For profile-specific installs, clone under `~/.hermes/profiles/<profile>/plugins/hermes-lcm`. For development checkouts, `scripts/install.sh` creates a profile-aware symlink into the active Hermes plugin directory and refuses to overwrite an existing checkout or unrelated symlink.

## Standalone diagnostics package

The repository is a Hermes plugin, not a standalone Python application. Runtime discovery currently depends on:

- `plugin.yaml` declaring the plugin name and registered tools
- the repo root containing `__init__.py` for Hermes plugin registration
- the operator placing or symlinking the checkout into Hermes' plugin search path
- no required third-party runtime dependencies beyond Python 3.11+ and optional accelerators such as `tiktoken` and `regex`

`pyproject.toml` installs the `hermes-lcm` JSON-first CLI and the package modules
needed by it. This does not make pip an implicit Hermes plugin discovery path;
the manifest checkout/symlink remains required by the current host contract.
Packaging tests install from a copied clean tree and exercise the executable
without a gateway.

## Host packaging boundary

Make packaging a separate implementation lane only when one of these is true:

1. Hermes Agent documents a stable pip/distribution entrypoint for plugins.
2. Users need version-pinned installs without direct git checkouts.
3. Release automation needs packaged artifacts beyond GitHub tags/releases.

The wheel includes `plugin.yaml` and copied-clean-tree tests prove the
standalone `hermes-lcm` executable and manifest tool inventory. Hermes does not
currently define pip packages as a plugin-discovery source, so clone/symlink
remains the documented host path. The plugin-side fresh-process activation
preflight is diagnostic evidence; it does not claim that the host selected the
context engine before tool discovery.

## Current install and update references

- Quickstart: [README](../README.md)
- Detailed install/update/verify: [Operator guide](operator-guide.md)
- Standalone install script contract: [`tests/test_packaging_install.py`](../tests/test_packaging_install.py)
