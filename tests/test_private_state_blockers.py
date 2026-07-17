"""Non-vacuous regressions for the private externalized-grep state subsystem."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
        scheduler_db = next(state_root.rglob("scheduler.sqlite3"))
        assert scheduler_db.stat().st_size <= tools_module._EXTERNALIZED_SCHEDULER_MAX_BYTES
        import sqlite3
        with sqlite3.connect(scheduler_db) as connection:
            rows = connection.execute("SELECT COUNT(*) FROM scheduler_cursors").fetchone()[0]
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
