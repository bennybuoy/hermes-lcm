# Hermes host activation contract

Issue #20 cannot be closed by plugin code alone. `hermes-lcm` can prove its
manifest, registration entry point, identity, schemas, and session binding in a
fresh process with:

```bash
hermes-lcm activation-preflight --expected-engine lcm --pretty
```

The command uses only a temporary profile database. Its
`host_activation_ordering_verified` field is deliberately `false`: it does not
claim that a Hermes gateway awaited plugin activation before resolving a real
session.

The host implementation must provide these ordering and observability rules:

1. Discover profile-local plugins and record stable plugin name/version/path.
2. Start and await every startup-activation plugin before resolving
   `context.engine`.
3. Reject duplicate engine names deterministically with both plugin identities.
4. Resolve the configured engine exactly once per session. Missing or failed
   engines fail closed unless the profile explicitly selects a fallback mode.
5. Persist the effective engine, plugin identity, activation duration, and any
   explicit fallback reason in session metadata.
6. Never switch an already-bound session because a late registration arrives.
7. Run discovery independently per profile; a process-global registration from
   another profile is not proof of availability.

Host tests must cover slow activation, activation failure, duplicates, disabled
plugins, explicit fallback, CLI startup, gateway startup, and multiple profiles.
Logs should expose separate discovery, activation-start, registration,
resolution, and session-binding events with stable identifiers and durations.
No plugin wrapper or site-packages modification is an acceptable substitute.

Release verification should run the plugin preflight in a new OS process and a
host-owned fresh-process startup test. Only the latter can set an equivalent of
`host_activation_ordering_verified=true`.
