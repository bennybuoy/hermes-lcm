# Read-only diagnostics CLI

The installed `hermes-lcm` command is a JSON-first interface for inspecting an
LCM database without starting Hermes or running plugin migrations.

```bash
hermes-lcm --database ~/.hermes/lcm.db status
hermes-lcm --profile work sessions list --limit 25
hermes-lcm messages tail --session-id SESSION --limit 20
hermes-lcm frontier show --conversation-id CONVERSATION
hermes-lcm doctor --pretty
```

Database connections use SQLite read-only mode plus `PRAGMA query_only=ON`.
List commands use bounded keyset pagination. Message and summary content is a
bounded preview unless `--full` is explicitly supplied. JSON is the default;
`--pretty` and `--table` provide human-oriented rendering.

`status` reports both the database's `schema_version` and this build's
`supported_schema_version` (currently 10). The CLI can inspect older databases
without migrating them, reads schema v10, and exits with database failure code
`5` before running a command when the database schema is newer than the build.

Path precedence is `--database`, `LCM_DATABASE_PATH`, a named `--profile`, then
the default Hermes home. `config show/get` exposes only the `lcm` mapping and
never prints unrelated Hermes configuration or credentials.

Exit codes are `0` success, `2` invalid input, `3` not found, `4` configuration
failure, and `5` database failure. The initial CLI is deliberately read-only;
future configuration or DAG mutations require a separately designed
backup-first apply boundary.
