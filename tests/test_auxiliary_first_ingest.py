from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _FakeAgent:
    def __init__(self, engine: LCMEngine, session_id: str, parent_session_id: str = ""):
        self.engine = engine
        self.session_id = session_id
        self._parent_session_id = parent_session_id
        self._memory_write_origin = "assistant_tool"
        self._memory_write_context = "foreground"
        self.log_prefix = ""
        self.enabled_toolsets = {"terminal", "file", "web"}

    def start_session(self) -> None:
        self.engine.on_session_start(
            self.session_id,
            platform="cli",
            conversation_id=f"conversation:{self.session_id}",
        )

    def ingest_history_after_background_marker(self, history):
        self._memory_write_origin = "background_review"
        self._memory_write_context = "background_review"
        self.engine.ingest(history)

    def ingest_history_as_foreground(self, history):
        self.engine.ingest(history)


class _ForegroundAgent(_FakeAgent):
    pass


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def test_first_ingest_rechecks_background_review_marker_after_missed_bind_time(tmp_path):
    engine = _engine(tmp_path)
    agent = _FakeAgent(engine, "review-session", parent_session_id="foreground-session")
    history = [
        {"role": "user", "content": "foreground question"},
        {"role": "assistant", "content": "foreground answer"},
    ]

    try:
        # Host bug shape: on_session_start fires while the agent still exposes
        # the foreground default, so bind-time auxiliary detection misses.
        agent.start_session()
        assert engine._store.get_session_count("review-session") == 0

        # By first post-LLM ingest the host has set the background-review marker.
        # LCM must re-check before writing the replay snapshot as a new session.
        agent.ingest_history_after_background_marker(history)

        assert engine._store.get_session_count("review-session") == 0
        assert engine._thread_context_has_auxiliary_session("review-session")
    finally:
        engine.shutdown()


def test_first_ingest_recheck_does_not_flag_foreground_session(tmp_path):
    engine = _engine(tmp_path)
    agent = _ForegroundAgent(engine, "foreground-session")
    history = [{"role": "user", "content": "real foreground turn"}]

    try:
        agent.start_session()
        agent.ingest_history_as_foreground(history)

        assert engine._store.get_session_count("foreground-session") == 1
        assert not engine._thread_context_has_auxiliary_session("foreground-session")
    finally:
        engine.shutdown()


def test_first_ingest_recheck_does_not_poison_session_reset_foreground(tmp_path):
    engine = _engine(tmp_path)
    first = _ForegroundAgent(engine, "foreground-a")
    second = _ForegroundAgent(engine, "foreground-b")

    try:
        first.start_session()
        first.ingest_history_as_foreground([{"role": "user", "content": "before reset"}])
        engine.on_session_reset()

        second.start_session()
        second.ingest_history_as_foreground([{"role": "user", "content": "after reset"}])

        assert engine._store.get_session_count("foreground-a") == 1
        assert engine._store.get_session_count("foreground-b") == 1
        assert not engine._thread_context_has_auxiliary_session("foreground-b")
    finally:
        engine.shutdown()
