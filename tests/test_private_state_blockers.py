"""Non-vacuous regressions for the private externalized-grep state subsystem."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

import hermes_lcm.tools as tools_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_package_env(tmp_path: Path) -> dict[str, str]:
    package_root = tmp_path / "subprocess-package"
    package_root.mkdir(exist_ok=True)
    package_link = package_root / "hermes_lcm"
    if not package_link.exists():
        package_link.symlink_to(REPO_ROOT, target_is_directory=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(package_root), env.get("PYTHONPATH", ""))
        if value
    )
    return env


def _engine(tmp_path: Path) -> LCMEngine:
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir(exist_ok=True)
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "private-state.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_path=str(payload_dir),
        async_background_compaction_worker_enabled=False,
    ), hermes_home=str(tmp_path))
    engine.on_session_start(
        "current",
        conversation_id="private-state",
        platform="test",
        context_length=100_000,
    )
    return engine


def _engine_with_payload_dir(tmp_path: Path, payload_dir: Path) -> LCMEngine:
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "private-state.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_path=str(payload_dir),
        async_background_compaction_worker_enabled=False,
    ), hermes_home=str(tmp_path))
    engine.on_session_start(
        "current",
        conversation_id="private-state",
        platform="test",
        context_length=100_000,
    )
    return engine


def _marker_payload(marker_count: int, needle: str) -> str:
    markers = ",".join(
        '{{"source_path":"/private/{0}","expected_chars":{0}}}'.format(index)
        for index in range(marker_count)
    )
    return (
        '{"session_id":"current","persisted_output_source_path":"/private/0",'
        '"persisted_output_expected_chars":0,"persisted_output_markers":['
        + markers
        + '],"content":'
        + json.dumps(needle)
        + "}"
    )


def test_tools_import_and_engine_startup_do_not_resolve_temp_root(tmp_path):
    script = r'''
import sys
import types
agent = types.ModuleType("agent")
agent.__path__ = []
context = types.ModuleType("agent.context_engine")
class ContextEngine: pass
context.ContextEngine = ContextEngine
agent.context_engine = context
sys.modules["agent"] = agent
sys.modules["agent.context_engine"] = context
import tempfile
tempfile.gettempdir = lambda: (_ for _ in ()).throw(FileNotFoundError("no temp root"))
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.tools
engine = LCMEngine(config=LCMConfig(
    database_path=sys.argv[1],
    async_background_compaction_worker_enabled=False,
))
engine.shutdown()
print("startup-ok")
'''
    env = _subprocess_package_env(tmp_path)
    unusable = tmp_path / "not-a-directory"
    unusable.write_text("blocked", encoding="utf-8")
    env.update({"TMPDIR": str(unusable), "TMP": str(unusable), "TEMP": str(unusable)})
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "startup.db")],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stdout.strip() == "startup-ok"


def test_externalized_marker_scan_fails_closed_when_private_root_is_unusable(
    tmp_path, monkeypatch
):
    payload_path = tmp_path / "payloads" / "no-private-state.json"
    payload_path.parent.mkdir()
    payload_path.write_text(
        _marker_payload(10, "MUST-NOT-LEAK-WITHOUT-STATE"), encoding="utf-8"
    )
    unusable = tmp_path / "state-root-is-a-file"
    unusable.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", unusable)
    engine = _engine(tmp_path)
    try:
        result = json.loads(tools_module.lcm_grep(
            {
                "query": "MUST-NOT-LEAK-WITHOUT-STATE",
                "content_scope": "externalized",
                "ref": payload_path.name,
            },
            engine=engine,
        ))
        assert result["results"] == []
        assert result["diagnostics"] == [
            {"ref": payload_path.name, "error": "unreadable"}
        ]
        assert not (tmp_path / "private-state.db").with_suffix(".db-journal").exists()
    finally:
        engine.shutdown()


def test_no_ref_search_fails_closed_before_listing_for_nine_alternating_shapes(
    tmp_path, monkeypatch
):
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    payload = payload_dir / "explicit-still-works.json"
    payload.write_text(
        json.dumps({
            "session_id": "current",
            "content": "EXPLICIT-WORKS " + " ".join(
                f"NO-REF-SHAPE-{index}" for index in range(9)
            ),
        }),
        encoding="utf-8",
    )
    unusable = tmp_path / "state-root-is-a-file"
    unusable.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", unusable)
    real_scandir = tools_module.os.scandir
    payload_listings = 0

    def counted_scandir(path):
        nonlocal payload_listings
        if Path(path) == payload_dir:
            payload_listings += 1
        return real_scandir(path)

    monkeypatch.setattr(tools_module.os, "scandir", counted_scandir)
    engine = _engine(tmp_path)
    try:
        for round_index in range(2):
            for shape_index in range(9):
                result = json.loads(tools_module.lcm_grep(
                    {
                        "query": f"NO-REF-SHAPE-{shape_index}",
                        "content_scope": "externalized",
                        "max_files": 1 + ((shape_index + round_index) % 2),
                        "limit": 1,
                    },
                    engine=engine,
                ))
                assert result["results"] == []
                assert result["diagnostics"] == [
                    {"ref": "", "error": "private_state_unavailable"}
                ]
                assert result["scan"]["files_scanned"] == 0
        assert payload_listings == 0

        explicit = json.loads(tools_module.lcm_grep(
            {
                "query": "EXPLICIT-WORKS",
                "content_scope": "externalized",
                "ref": payload.name,
            },
            engine=engine,
        ))
        assert explicit["total_results"] == 1
        assert explicit["results"][0]["ref"] == payload.name
    finally:
        engine.shutdown()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires proc owner identity")
@pytest.mark.parametrize(
    "crash_phase",
    ["before_mkdir", "after_mkdir", "before_lease", "after_lease"],
)
def test_registered_owner_intent_recovers_crash_at_every_creation_phase(
    tmp_path, monkeypatch, crash_phase
):
    state_root = tmp_path / f"crash-{crash_phase}"
    script = r'''
import os
from pathlib import Path
import sys
import hermes_lcm.tools as tools

root = Path(sys.argv[1])
phase = sys.argv[2]
tools._EXTERNALIZED_MARKER_STATE_ROOT = root
original_mkdir = Path.mkdir
original_lease = tools._externalized_lock_new_lease

def crash_mkdir(path, *args, **kwargs):
    if path.name.startswith("owner-") and phase == "before_mkdir":
        os._exit(81)
    result = original_mkdir(path, *args, **kwargs)
    if path.name.startswith("owner-") and phase == "after_mkdir":
        os._exit(82)
    return result

def crash_lease(path):
    if phase == "before_lease":
        os._exit(83)
    descriptor = original_lease(path)
    if phase == "after_lease":
        os._exit(84)
    return descriptor

Path.mkdir = crash_mkdir
tools._externalized_lock_new_lease = crash_lease
tools._ExternalizedPrivateRuntimeState().ensure_owner_dir()
raise AssertionError("crash hook did not fire")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(state_root), crash_phase],
        env=_subprocess_package_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == {
        "before_mkdir": 81,
        "after_mkdir": 82,
        "before_lease": 83,
        "after_lease": 84,
    }[crash_phase], (completed.stdout, completed.stderr)

    registry_path = state_root / ".owner-registry.db"
    assert registry_path.is_file(), "intent must be durable before owner mkdir"
    with sqlite3.connect(registry_path) as connection:
        registered = connection.execute(
            "SELECT owner_name, pid, process_start, nonce FROM owner_registry"
        ).fetchall()
    assert len(registered) == 1
    owner_name, pid, process_start, nonce = registered[0]
    assert owner_name == f"owner-{pid}-{process_start}-{nonce}"

    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    local = tools_module._ExternalizedPrivateRuntimeState()
    local.ensure_owner_dir()
    local.close_for_shutdown()
    assert not (state_root / owner_name).exists()
    with sqlite3.connect(registry_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM owner_registry").fetchone()[0] == 0


@pytest.mark.parametrize(
    "crash_phase",
    [
        "after_file_open",
        "after_wal",
        "after_owner_registry_ddl",
        "after_registry_state_ddl",
        "after_scheduler_cursors_ddl",
        "after_scheduler_ttl_index_ddl",
        "after_scheduler_fairness_index_ddl",
        "after_schema_marker",
        "before_commit",
    ],
)
def test_registry_schema_bootstrap_recovers_every_crash_boundary(
    tmp_path, monkeypatch, crash_phase
):
    state_root = tmp_path / f"registry-schema-{crash_phase}"
    script = r'''
import os
from pathlib import Path
import sys
import hermes_lcm.tools as tools

root = Path(sys.argv[1])
root.mkdir(mode=0o700)
tools._EXTERNALIZED_REGISTRY_SCHEMA_CRASH_PHASE = sys.argv[2]
tools._externalized_open_owner_registry(root)
raise AssertionError("schema crash hook did not fire")
'''
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(state_root), crash_phase],
        env=_subprocess_package_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert crashed.returncode == 91, (crashed.stdout, crashed.stderr)
    registry_path = state_root / ".owner-registry.db"
    assert registry_path.is_file()

    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    runtime = tools_module._ExternalizedPrivateRuntimeState()
    runtime.ensure_owner_dir()
    runtime.close_for_shutdown()
    with sqlite3.connect(registry_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        objects = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }
        assert {
            "owner_registry",
            "registry_state",
            "scheduler_cursors",
            "scheduler_cursors_updated_idx",
            "scheduler_cursors_fairness_idx",
        } <= objects


@pytest.mark.parametrize(
    "slow_phase", ["after_registry_state_ddl", "before_commit"]
)
def test_slow_registry_schema_recovery_has_separate_cleanup_and_owner_budgets(
    tmp_path, monkeypatch, slow_phase
):
    state_root = tmp_path / f"slow-registry-schema-{slow_phase}"
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    delay = tools_module._EXTERNALIZED_OWNER_REAP_DEADLINE_SECONDS + 0.020

    def slow_schema(phase: str) -> None:
        if phase == slow_phase:
            time.sleep(delay)

    monkeypatch.setattr(
        tools_module, "_externalized_registry_schema_crash_point", slow_schema
    )
    runtime = tools_module._ExternalizedPrivateRuntimeState()
    started = time.monotonic()
    owner_dir = runtime.ensure_owner_dir()
    elapsed = time.monotonic() - started
    try:
        assert owner_dir.parent == state_root
        assert elapsed < 0.50
        with sqlite3.connect(state_root / ".owner-registry.db") as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            assert connection.execute(
                "SELECT COUNT(*) FROM owner_registry"
            ).fetchone()[0] == 1
    finally:
        runtime.close_for_shutdown()


def test_registry_schema_repairs_missing_v1_objects_and_rejects_newer_version(
    tmp_path
):
    state_root = tmp_path / "registry-versioning"
    state_root.mkdir()
    registry_path = state_root / ".owner-registry.db"
    with sqlite3.connect(registry_path) as connection:
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE owner_registry ("
            "owner_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "owner_name TEXT NOT NULL UNIQUE, pid INTEGER NOT NULL, "
            "process_start TEXT NOT NULL, nonce TEXT NOT NULL, "
            "phase TEXT NOT NULL, created_wall REAL NOT NULL)"
        )
        connection.commit()
    repaired = tools_module._externalized_open_owner_registry(state_root)
    repaired.close()
    with sqlite3.connect(registry_path) as connection:
        objects = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }
        assert {"registry_state", "scheduler_cursors"} <= objects
        connection.execute("PRAGMA user_version=2")
        connection.commit()
    with pytest.raises(tools_module._ExternalizedRegistrySchemaMissing):
        tools_module._externalized_open_owner_registry(state_root)


def test_owner_registry_supports_concurrent_process_start_and_shutdown(tmp_path):
    state_root = tmp_path / "concurrent-owner-registry"
    script = r'''
from pathlib import Path
import sys
import hermes_lcm.tools as tools
tools._EXTERNALIZED_MARKER_STATE_ROOT = Path(sys.argv[1])
runtime = tools._ExternalizedPrivateRuntimeState()
owner_dir = runtime.ensure_owner_dir()
print(owner_dir.name, flush=True)
sys.stdin.read(1)
runtime.close_for_shutdown()
'''
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(state_root)],
            env=_subprocess_package_env(tmp_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(12)
    ]
    owner_names = []
    try:
        for process in processes:
            assert process.stdout is not None
            owner_name = process.stdout.readline().strip()
            assert owner_name, (
                process.stderr.read() if process.stderr is not None else ""
            )
            owner_names.append(owner_name)
        assert len(set(owner_names)) == len(processes)
        with sqlite3.connect(state_root / ".owner-registry.db") as registry:
            assert registry.execute(
                "SELECT COUNT(*) FROM owner_registry"
            ).fetchone()[0] == len(processes)
        assert all((state_root / name / "owner.lease").is_file() for name in owner_names)
    finally:
        for process in processes:
            if process.stdin is not None:
                process.stdin.write("x")
                process.stdin.flush()
        for process in processes:
            process.wait(timeout=20)
            assert process.returncode == 0, (
                process.stderr.read() if process.stderr is not None else ""
            )
    with sqlite3.connect(state_root / ".owner-registry.db") as registry:
        assert registry.execute("SELECT COUNT(*) FROM owner_registry").fetchone()[0] == 0
    assert list(state_root.glob("owner-*")) == []


def _start_marker_owner(
    tmp_path: Path,
    state_root: Path,
    *,
    crash: bool,
) -> tuple[subprocess.Popen[str], Path]:
    script = r'''
import os
from pathlib import Path
import sys
import hermes_lcm.tools as tools
tools._EXTERNALIZED_MARKER_STATE_ROOT = Path(sys.argv[1])
store = tools._ExternalizedMarkerIdentityStore()
store.add((sys.argv[2], 1))
store.flush()
print(store.path, flush=True)
if sys.argv[2] == "crash":
    os._exit(91)
sys.stdin.read(1)
store.close()
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(state_root), "crash" if crash else "live"],
        env=_subprocess_package_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    raw_path = process.stdout.readline().strip()
    assert raw_path, process.stderr.read() if process.stderr is not None else ""
    return process, Path(raw_path)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires proc owner identity")
def test_reaper_preserves_two_live_subprocess_owners_and_reaps_real_crash(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "marker-state"
    first, first_path = _start_marker_owner(tmp_path, state_root, crash=False)
    second, second_path = _start_marker_owner(tmp_path, state_root, crash=False)
    crashed, crashed_path = _start_marker_owner(tmp_path, state_root, crash=True)
    assert crashed.wait(timeout=10) == 91
    try:
        assert first_path.exists() and second_path.exists() and crashed_path.exists()
        assert first_path.parent != second_path.parent
        assert (first_path.parent / "owner.lease").is_file()
        assert (second_path.parent / "owner.lease").is_file()
        monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
        local = tools_module._ExternalizedMarkerIdentityStore()
        try:
            assert first_path.exists()
            assert second_path.exists()
            assert not crashed_path.exists()
        finally:
            local.close()
    finally:
        for process in (first, second):
            if process.stdin is not None:
                process.stdin.write("x")
                process.stdin.flush()
            process.wait(timeout=10)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires proc owner identity")
def test_indexed_registry_reaping_is_capped_and_progresses_with_100k_rows(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "large-owner-registry"
    state_root.mkdir(mode=0o700)
    connection = tools_module._externalized_open_owner_registry(state_root)
    current_pid = os.getpid()
    current_start = tools_module._externalized_process_start_identity(current_pid)
    assert current_start is not None
    initial_dead: set[str] = set()
    live_names: set[str] = set()
    rows = []
    now = time.time()
    for index in range(100_000):
        if index < 96 and index % 3 == 0:
            nonce = f"{index:032x}"
            name = f"owner-99999999-1-{nonce}"
            initial_dead.add(name)
            owner_dir = state_root / name
            owner_dir.mkdir()
            (owner_dir / "owner.lease").write_text("", encoding="ascii")
            rows.append((name, 99999999, "1", nonce, "leased", now))
        elif index < 96 and index % 3 == 1:
            nonce = f"{index:032x}"
            name = f"owner-{current_pid}-{current_start}-{nonce}"
            live_names.add(name)
            owner_dir = state_root / name
            owner_dir.mkdir()
            (owner_dir / "owner.lease").write_text("", encoding="ascii")
            rows.append((name, current_pid, current_start, nonce, "leased", now))
        else:
            rows.append((
                f"malformed-registry-row-{index}",
                -1,
                "bad",
                "bad",
                "malformed",
                now,
            ))
    connection.executemany(
        "INSERT INTO owner_registry("
        "owner_name, pid, process_start, nonce, phase, created_wall"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()

    real_scandir = tools_module._EXTERNALIZED_PRIVATE_STATE_SCANDIR
    root_scans = 0

    def reject_root_scan(path):
        nonlocal root_scans
        if Path(path) == state_root:
            root_scans += 1
            raise AssertionError("shared root must never be scanned")
        return real_scandir(path)

    monkeypatch.setattr(
        tools_module, "_EXTERNALIZED_PRIVATE_STATE_SCANDIR", reject_root_scan
    )
    remaining_counts = []
    for _ in range(12):
        stats = tools_module._externalized_reap_dead_owners(state_root)
        assert stats["rows_visited"] <= tools_module._EXTERNALIZED_OWNER_REAP_MAX_ROWS
        assert stats["entries_visited"] <= tools_module._EXTERNALIZED_OWNER_REAP_MAX_ENTRIES
        assert stats["elapsed"] <= tools_module._EXTERNALIZED_OWNER_REAP_DEADLINE_SECONDS + 0.5
        remaining_counts.append(sum(
            (state_root / owner_name).exists() for owner_name in initial_dead
        ))
    assert root_scans == 0
    assert remaining_counts == sorted(remaining_counts, reverse=True)
    assert remaining_counts[-1] == 0
    assert any(
        later < earlier
        for earlier, later in zip(remaining_counts, remaining_counts[1:])
    )
    assert all((state_root / owner_name).is_dir() for owner_name in live_names)

    with sqlite3.connect(state_root / ".owner-registry.db") as registry:
        registered_names = {
            row[0] for row in registry.execute(
                "SELECT owner_name FROM owner_registry WHERE owner_name IN ("
                + ",".join("?" for _ in live_names)
                + ")",
                tuple(live_names),
            )
        }
        assert registered_names == live_names
        cursor = registry.execute(
            "SELECT reap_cursor FROM registry_state WHERE singleton = 1"
        ).fetchone()[0]
        assert cursor > 0
    registry = tools_module._externalized_open_owner_registry(state_root)
    try:
        assert registry.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert registry.execute("PRAGMA mmap_size").fetchone()[0] == 0
        cache_pages = registry.execute("PRAGMA cache_size").fetchone()[0]
        assert -64 <= cache_pages < 0
    finally:
        registry.close()

    with sqlite3.connect(tmp_path / "private-state.db") as production_db:
        assert production_db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'owner_registry'"
        ).fetchone() is None


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux flock")
def test_reaper_deadline_includes_contended_registry_acquisition(tmp_path):
    import fcntl

    state_root = tmp_path / "contended-owner-registry"
    state_root.mkdir(mode=0o700)
    lock_path = state_root / ".owner-registry.db.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def delayed_release() -> None:
        time.sleep(0.30)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    release = threading.Thread(target=delayed_release)
    release.start()
    started = time.monotonic()
    stats = tools_module._externalized_reap_dead_owners(state_root)
    elapsed = time.monotonic() - started
    release.join(timeout=2)
    assert not release.is_alive()
    assert elapsed < 0.20
    assert stats["elapsed"] < 0.20
    assert stats["rows_visited"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux flock")
def test_held_cleanup_lock_does_not_make_valid_root_unwritable(
    tmp_path, monkeypatch
):
    import fcntl

    state_root = tmp_path / "held-cleanup-lock"
    state_root.mkdir(mode=0o700)
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    lock_path = state_root / ".owner-registry.db.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        started = time.monotonic()
        assert tools_module._prepare_externalized_marker_state_root() == state_root
        elapsed = time.monotonic() - started
        assert elapsed < 0.20
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    runtime = tools_module._ExternalizedPrivateRuntimeState()
    owner_dir = runtime.ensure_owner_dir()
    try:
        assert owner_dir.parent == state_root
    finally:
        runtime.close_for_shutdown()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="requires proc owner identity")
def test_stale_reaper_cannot_delete_same_key_replacement(tmp_path, monkeypatch):
    state_root = tmp_path / "owner-registry-aba"
    state_root.mkdir(mode=0o700)
    registry = tools_module._externalized_open_owner_registry(state_root)
    old_nonce = "1" * 32
    old_name = f"owner-99999999-1-{old_nonce}"
    registry.execute(
        "INSERT INTO owner_registry("
        "owner_name, pid, process_start, nonce, phase, created_wall"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        (old_name, 99999999, "1", old_nonce, "leased", time.time()),
    )
    old_id = int(registry.execute(
        "SELECT owner_id FROM owner_registry WHERE owner_name = ?", (old_name,)
    ).fetchone()[0])
    registry.commit()
    registry.close()
    old_dir = state_root / old_name
    old_dir.mkdir()
    (old_dir / "owner.lease").write_text("", encoding="ascii")

    selected = threading.Event()
    release = threading.Event()

    def barrier_remove(*args, **kwargs):
        selected.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(
        tools_module, "_externalized_bounded_remove_private_tree", barrier_remove
    )
    outcomes: list[object] = []

    def stale_reap() -> None:
        try:
            outcomes.append(tools_module._externalized_reap_dead_owners(state_root))
        except BaseException as exc:  # noqa: BLE001 - asserted in parent
            outcomes.append(exc)

    thread = threading.Thread(target=stale_reap)
    thread.start()
    assert selected.wait(timeout=5)

    current_pid = os.getpid()
    current_start = tools_module._externalized_process_start_identity(current_pid)
    assert current_start is not None
    replacement_nonce = "2" * 32
    replacement_name = f"owner-{current_pid}-{current_start}-{replacement_nonce}"
    with sqlite3.connect(state_root / ".owner-registry.db") as competitor:
        competitor.execute("DELETE FROM owner_registry WHERE owner_id = ?", (old_id,))
        competitor.execute(
            "INSERT INTO owner_registry("
            "owner_id, owner_name, pid, process_start, nonce, phase, created_wall"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                old_id,
                replacement_name,
                current_pid,
                current_start,
                replacement_nonce,
                "intended",
                time.time(),
            ),
        )
        competitor.commit()
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcomes and not isinstance(outcomes[0], BaseException)

    with sqlite3.connect(state_root / ".owner-registry.db") as check:
        replacement = check.execute(
            "SELECT owner_name, pid, process_start, nonce, phase "
            "FROM owner_registry WHERE owner_id = ?",
            (old_id,),
        ).fetchone()
        assert replacement == (
            replacement_name,
            current_pid,
            current_start,
            replacement_nonce,
            "intended",
        )
        check.execute("DELETE FROM owner_registry WHERE owner_id = ?", (old_id,))
        check.execute(
            "INSERT INTO owner_registry("
            "owner_name, pid, process_start, nonce, phase, created_wall"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            ("malformed-next", -1, "bad", "3" * 32, "malformed", time.time()),
        )
        next_id = int(check.execute(
            "SELECT owner_id FROM owner_registry WHERE owner_name = 'malformed-next'"
        ).fetchone()[0])
        assert next_id > old_id


def test_shutdown_fences_checked_out_marker_continuation(tmp_path, monkeypatch):
    state_root = tmp_path / "marker-state"
    payload_path = tmp_path / "payloads" / "shutdown-race.json"
    payload_path.parent.mkdir()
    payload_path.write_text(
        _marker_payload(8_000, "SHUTDOWN-RACE-NEEDLE"), encoding="utf-8"
    )
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 64 * 1024)
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0)
    engine = _engine(tmp_path)
    first = json.loads(tools_module.lcm_grep(
        {
            "query": "SHUTDOWN-RACE-NEEDLE",
            "content_scope": "externalized",
            "ref": payload_path.name,
        },
        engine=engine,
    ))
    assert first["scan"]["continuations_pending"] == 1
    assert list(state_root.rglob("*.sqlite3"))

    entered = threading.Event()
    release = threading.Event()
    original_resume = tools_module._ExternalizedPayloadContinuation.resume

    def blocked_resume(continuation, *args, **kwargs):
        entered.set()
        assert release.wait(timeout=10)
        return original_resume(continuation, *args, **kwargs)

    monkeypatch.setattr(
        tools_module._ExternalizedPayloadContinuation, "resume", blocked_resume
    )
    outcome: list[object] = []

    def retry() -> None:
        try:
            outcome.append(json.loads(tools_module.lcm_grep(
                {
                    "query": "SHUTDOWN-RACE-NEEDLE",
                    "content_scope": "externalized",
                    "ref": payload_path.name,
                },
                engine=engine,
            )))
        except BaseException as exc:  # noqa: BLE001 - asserted in parent
            outcome.append(exc)

    thread = threading.Thread(target=retry)
    thread.start()
    assert entered.wait(timeout=10)
    engine.shutdown()
    release.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert outcome and not isinstance(outcome[0], BaseException)
    assert getattr(engine, "_externalized_grep_runtime_state").closed is True
    assert getattr(engine, "_externalized_grep_continuations", {}) == {}
    assert list(state_root.rglob("*.sqlite3")) == []
    assert list(state_root.glob("owner-*")) == []


@pytest.mark.parametrize("shape_count", [9, 50])
def test_no_ref_scheduler_resumes_evicted_query_shapes_from_private_disk(
    tmp_path, monkeypatch, shape_count
):
    state_root = tmp_path / "marker-state"
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    needles = [f"PRIVATE-SHAPE-{index:03d}-LAST" for index in range(shape_count)]
    for index in range(5):
        content = "x" * (24 * 1024)
        if index == 4:
            content = " ".join(needles) + content
        (payload_dir / f"shape-{index}.json").write_text(
            json.dumps({"session_id": "current", "content": content}, separators=(",", ":")),
            encoding="utf-8",
        )
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_MAX_BYTES", 12 * 1024)
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0)
    engine = _engine(tmp_path)
    remaining = set(needles)
    try:
        for _round in range(40):
            for needle in needles:
                if needle not in remaining:
                    continue
                result = json.loads(tools_module.lcm_grep(
                    {
                        "query": needle,
                        "content_scope": "externalized",
                        "max_files": 5,
                        "limit": 5,
                    },
                    engine=engine,
                ))
                if result["total_results"]:
                    assert result["results"][0]["ref"] == "shape-4.json"
                    remaining.remove(needle)
            if not remaining:
                break
        assert not remaining
        assert len(engine._externalized_grep_schedulers) <= (
            tools_module._EXTERNALIZED_SCHEDULER_MAX_QUERIES
        )
        scheduler_db = state_root / ".owner-registry.db"
        import sqlite3
        with sqlite3.connect(scheduler_db) as connection:
            rows = connection.execute("SELECT COUNT(*) FROM scheduler_cursors").fetchone()[0]
            scheduler_bytes = connection.execute(
                "SELECT COALESCE(SUM(state_bytes), 0) FROM scheduler_cursors"
            ).fetchone()[0]
            assert scheduler_bytes <= tools_module._EXTERNALIZED_SCHEDULER_MAX_BYTES
            indexes = {
                row[1] for row in connection.execute(
                    "PRAGMA index_list(scheduler_cursors)"
                )
            }
            assert {
                "scheduler_cursors_updated_idx",
                "scheduler_cursors_fairness_idx",
            } <= indexes
            live_keys = {
                row[0] for row in connection.execute(
                    "SELECT shape_key FROM scheduler_cursors"
                )
            }
            stale_candidates = live_keys - set(engine._externalized_grep_schedulers)
            assert stale_candidates
            stale_key = next(iter(stale_candidates))
            connection.execute(
                "UPDATE scheduler_cursors SET updated_wall = ? WHERE shape_key = ?",
                (
                    time.time() - tools_module._EXTERNALIZED_SCHEDULER_TTL_SECONDS - 1,
                    stale_key,
                ),
            )
            connection.commit()
        assert shape_count <= rows <= tools_module._EXTERNALIZED_SCHEDULER_MAX_DISK_QUERIES
        json.loads(tools_module.lcm_grep(
            {
                "query": "PRIVATE-SCHEDULER-CLEANUP-SHAPE",
                "content_scope": "externalized",
                "max_files": 5,
                "limit": 5,
            },
            engine=engine,
        ))
        with sqlite3.connect(scheduler_db) as connection:
            after_cleanup = {
                row[0] for row in connection.execute(
                    "SELECT shape_key FROM scheduler_cursors"
                )
            }
        assert stale_key not in after_cleanup
        assert live_keys - {stale_key} <= after_cleanup
        with sqlite3.connect(tmp_path / "private-state.db") as production_db:
            assert production_db.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'scheduler_cursors'"
            ).fetchone() is None
    finally:
        engine.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="requires durable directory cookies")
def test_no_ref_listing_cookie_survives_shutdown_and_process_restart(tmp_path):
    state_root = tmp_path / "listing-cookie-state"
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    for index in range(1_500):
        (payload_dir / f"cookie-{index:04d}.json").write_text(
            '{"session_id":"current","content":"ordinary haystack"}',
            encoding="utf-8",
        )
    native_order = [entry.name for entry in os.scandir(payload_dir)]
    target = native_order[-1]
    (payload_dir / target).write_text(
        '{"session_id":"current","content":"COOKIE-LAST-NEEDLE"}',
        encoding="utf-8",
    )
    process_a = r'''
import json
from pathlib import Path
import sqlite3
import sys
from hermes_lcm.benchmarking.standalone import ensure_agent_context_engine_importable
ensure_agent_context_engine_importable()
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.tools as tools

state_root, payload_dir, work_dir = map(Path, sys.argv[1:4])
tools._EXTERNALIZED_MARKER_STATE_ROOT = state_root
tools._LCM_GREP_OPERATION_DEADLINE_SECONDS = 10.0
work_dir.mkdir()
engine = LCMEngine(config=LCMConfig(
    database_path=str(work_dir / "engine.db"),
    large_output_externalization_enabled=True,
    large_output_externalization_path=str(payload_dir),
    async_background_compaction_worker_enabled=False,
), hermes_home=str(work_dir))
engine.on_session_start("current", conversation_id="restart-a", platform="test",
                        context_length=100_000)
entries = 0
for _ in range(6):
    result = json.loads(tools.lcm_grep({
        "query": "COOKIE-LAST-NEEDLE", "content_scope": "externalized",
        "max_files": 50, "limit": 1,
    }, engine=engine))
    assert result["total_results"] == 0
    entries += result["scan"]["entries_scanned"]
with sqlite3.connect(state_root / ".owner-registry.db") as connection:
    cookie, version = connection.execute(
        "SELECT listing_cookie, version FROM scheduler_cursors"
    ).fetchone()
engine.shutdown()
print(json.dumps({"entries": entries, "cookie": cookie, "version": version}),
      flush=True)
'''
    first = subprocess.run(
        [
            sys.executable,
            "-c",
            process_a,
            str(state_root),
            str(payload_dir),
            str(tmp_path / "process-a"),
        ],
        env=_subprocess_package_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert first.returncode == 0, (first.stdout, first.stderr)
    first_state = json.loads(first.stdout)
    assert first_state["entries"] == 300
    assert first_state["cookie"] > 0
    assert list(state_root.glob("owner-*")) == []

    process_b = r'''
import json
from pathlib import Path
import sqlite3
import sys
from hermes_lcm.benchmarking.standalone import ensure_agent_context_engine_importable
ensure_agent_context_engine_importable()
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.tools as tools

state_root, payload_dir, work_dir = map(Path, sys.argv[1:4])
tools._EXTERNALIZED_MARKER_STATE_ROOT = state_root
tools._LCM_GREP_OPERATION_DEADLINE_SECONDS = 10.0
work_dir.mkdir()
observed_cookies = []
real_listing = tools._externalized_bounded_sorted_refs
def observed_listing(root, *, listing_cookie, max_files, deadline):
    observed_cookies.append(listing_cookie)
    return real_listing(root, listing_cookie=listing_cookie, max_files=max_files,
                        deadline=deadline)
tools._externalized_bounded_sorted_refs = observed_listing
engine = LCMEngine(config=LCMConfig(
    database_path=str(work_dir / "engine.db"),
    large_output_externalization_enabled=True,
    large_output_externalization_path=str(payload_dir),
    async_background_compaction_worker_enabled=False,
), hermes_home=str(work_dir))
engine.on_session_start("current", conversation_id="restart-b", platform="test",
                        context_length=100_000)
entries = 0
result = None
for _ in range(30):
    result = json.loads(tools.lcm_grep({
        "query": "COOKIE-LAST-NEEDLE", "content_scope": "externalized",
        "max_files": 50, "limit": 1,
    }, engine=engine))
    entries += result["scan"]["entries_scanned"]
    if result["total_results"]:
        break
assert result is not None and result["total_results"] == 1
with sqlite3.connect(state_root / ".owner-registry.db") as connection:
    cookie, version = connection.execute(
        "SELECT listing_cookie, version FROM scheduler_cursors"
    ).fetchone()
engine.shutdown()
print(json.dumps({"entries": entries, "first_cookie": observed_cookies[0],
                  "cookie": cookie, "version": version,
                  "ref": result["results"][0]["ref"]}), flush=True)
'''
    second = subprocess.run(
        [
            sys.executable,
            "-c",
            process_b,
            str(state_root),
            str(payload_dir),
            str(tmp_path / "process-b"),
        ],
        env=_subprocess_package_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert second.returncode == 0, (second.stdout, second.stderr)
    resumed = json.loads(second.stdout)
    assert resumed["first_cookie"] == first_state["cookie"]
    assert resumed["entries"] <= 1_500 - first_state["entries"]
    assert resumed["version"] > first_state["version"]
    assert resumed["ref"] == target


@pytest.mark.skipif(sys.platform != "linux", reason="requires durable directory cookies")
def test_same_shape_stale_writer_cannot_rollback_cookie_or_version(
    tmp_path, monkeypatch
):
    state_root = tmp_path / "same-shape-state"
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    for index in range(240):
        (payload_dir / f"contention-{index:04d}.json").write_text(
            '{"session_id":"current","content":"ordinary haystack"}',
            encoding="utf-8",
        )
    monkeypatch.setattr(tools_module, "_EXTERNALIZED_MARKER_STATE_ROOT", state_root)
    monkeypatch.setattr(tools_module, "_LCM_GREP_OPERATION_DEADLINE_SECONDS", 10.0)
    stale_engine = _engine_with_payload_dir(tmp_path / "stale", payload_dir)
    winner_engine = _engine_with_payload_dir(tmp_path / "winner", payload_dir)
    real_listing = tools_module._externalized_bounded_sorted_refs
    call_lock = threading.Lock()
    stale_entered = threading.Event()
    release_stale = threading.Event()
    make_first_stale = False

    def barrier_listing(root, *, listing_cookie, max_files, deadline):
        nonlocal make_first_stale
        with call_lock:
            stale_call = make_first_stale
            if stale_call:
                make_first_stale = False
        if stale_call:
            stale_entered.set()
            assert release_stale.wait(timeout=10)
            return (), 0, False, listing_cookie
        return real_listing(
            root,
            listing_cookie=listing_cookie,
            max_files=max_files,
            deadline=deadline,
        )

    monkeypatch.setattr(
        tools_module, "_externalized_bounded_sorted_refs", barrier_listing
    )
    args = {
        "query": "NEVER-PRESENT-CONTENTION-NEEDLE",
        "content_scope": "externalized",
        "max_files": 20,
        "limit": 1,
    }
    previous_cookie = -1
    previous_version = -1
    try:
        for _round in range(6):
            stale_entered.clear()
            release_stale.clear()
            with call_lock:
                make_first_stale = True
            stale_results: list[dict] = []
            stale_thread = threading.Thread(
                target=lambda: stale_results.append(
                    json.loads(tools_module.lcm_grep(args, engine=stale_engine))
                )
            )
            stale_thread.start()
            assert stale_entered.wait(timeout=10)
            winner = json.loads(
                tools_module.lcm_grep(args, engine=winner_engine)
            )
            assert winner["diagnostics"] == []
            registry_path = state_root / ".owner-registry.db"
            with sqlite3.connect(registry_path) as connection:
                won_cookie, won_version = connection.execute(
                    "SELECT listing_cookie, version FROM scheduler_cursors"
                ).fetchone()
            assert won_cookie > previous_cookie
            assert won_version > previous_version
            release_stale.set()
            stale_thread.join(timeout=15)
            assert not stale_thread.is_alive()
            assert stale_results[0]["diagnostics"] == [
                {"ref": "", "error": "private_state_unavailable"}
            ]
            with sqlite3.connect(registry_path) as connection:
                final_cookie, final_version = connection.execute(
                    "SELECT listing_cookie, version FROM scheduler_cursors"
                ).fetchone()
            assert (final_cookie, final_version) == (won_cookie, won_version)
            previous_cookie, previous_version = final_cookie, final_version
    finally:
        release_stale.set()
        stale_engine.shutdown()
        winner_engine.shutdown()


def _measure_marker_rss(tmp_path: Path, marker_count: int, stores: int) -> dict[str, int]:
    script = r'''
import gc
import json
from pathlib import Path
import sys
import hermes_lcm.tools as tools
tools._EXTERNALIZED_MARKER_STATE_ROOT = Path(sys.argv[1])

def memory_value(name):
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith(name + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(name)

gc.collect()
baseline = memory_value("VmRSS")
baseline_hwm = memory_value("VmHWM")
count = int(sys.argv[2])
store_count = int(sys.argv[3])
active = [tools._ExternalizedMarkerIdentityStore() for _ in range(store_count)]
for index in range(count):
    for store_index, store in enumerate(active):
        store.add((store_index, index, "/marker", index))
for store in active:
    store.flush()
current = memory_value("VmRSS")
high_water = memory_value("VmHWM")
print(json.dumps({
    "baseline": baseline,
    "baseline_hwm": baseline_hwm,
    "current": current,
    "high_water": high_water,
    "delta": max(current - baseline, high_water - baseline_hwm),
}))
for store in active:
    store.close()
'''
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / f"rss-state-{marker_count}-{stores}"),
            str(marker_count),
            str(stores),
        ],
        env=_subprocess_package_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout)


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="requires Linux RSS accounting")
def test_marker_sqlite_native_rss_is_low_and_scales_by_checkpoints_not_rows(tmp_path):
    one_100k = _measure_marker_rss(tmp_path, 100_000, 1)
    one_500k = _measure_marker_rss(tmp_path, 500_000, 1)
    four_100k = _measure_marker_rss(tmp_path, 100_000, 4)

    # A row-proportional in-memory identity set grows by tens of MiB between
    # 100k and 500k. The disk index may retain allocator/page noise, but its RSS
    # slope must stay under 12 MiB and four live checkpoints under 12 MiB each.
    assert one_500k["delta"] <= one_100k["delta"] + (12 * 1024 * 1024)
    assert four_100k["delta"] <= one_100k["delta"] + (36 * 1024 * 1024)
    assert one_500k["delta"] < 48 * 1024 * 1024
    assert four_100k["delta"] < 64 * 1024 * 1024

    store = tools_module._ExternalizedMarkerIdentityStore()
    try:
        assert store.connection is not None
        assert store.connection.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert store.connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
        cache_pages = store.connection.execute("PRAGMA cache_size").fetchone()[0]
        assert -64 <= cache_pages < 0
    finally:
        store.close()
