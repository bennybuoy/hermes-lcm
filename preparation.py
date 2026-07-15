"""Process-wide admission control for speculative summary preparation."""

from __future__ import annotations

import threading
from pathlib import Path


_LOCK = threading.Lock()
_ACTIVE_BY_PROFILE: dict[str, int] = {}


def profile_admission_key(database_path: str | Path) -> str:
    return str(Path(database_path).expanduser().resolve(strict=False))


def try_acquire_profile_admission(database_path: str | Path, limit: int) -> bool:
    key = profile_admission_key(database_path)
    hard_limit = max(1, int(limit or 1))
    with _LOCK:
        active = int(_ACTIVE_BY_PROFILE.get(key, 0))
        if active >= hard_limit:
            return False
        _ACTIVE_BY_PROFILE[key] = active + 1
        return True


def release_profile_admission(database_path: str | Path) -> None:
    key = profile_admission_key(database_path)
    with _LOCK:
        active = int(_ACTIVE_BY_PROFILE.get(key, 0))
        if active <= 1:
            _ACTIVE_BY_PROFILE.pop(key, None)
        else:
            _ACTIVE_BY_PROFILE[key] = active - 1


def active_profile_admissions(database_path: str | Path) -> int:
    key = profile_admission_key(database_path)
    with _LOCK:
        return int(_ACTIVE_BY_PROFILE.get(key, 0))
