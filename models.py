from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, asdict, field
from typing import Any


def app_state_dir() -> str:
    if platform.system() == "Windows":
        base = os.getenv("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "anigui_app")
    elif platform.system() == "Darwin":
        d = os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "anigui_app",
        )
    else:
        base = os.getenv("XDG_STATE_HOME") or os.path.join(
            os.path.expanduser("~"),
            ".local",
            "state",
        )
        d = os.path.join(base, "anigui_app")

    os.makedirs(d, exist_ok=True)
    return d


HISTORY_PATH = os.path.join(app_state_dir(), "history.json")
SEARCH_HISTORY_PATH = os.path.join(app_state_dir(), "search_history.json")
OFFLINE_COVERS_MAP_PATH = os.path.join(app_state_dir(), "offline_covers.json")
SETTINGS_PATH = os.path.join(app_state_dir(), "settings.json")
FAVORITES_PATH = os.path.join(app_state_dir(), "favorites.json")
METADATA_CACHE_PATH = os.path.join(app_state_dir(), "metadata_cache.json")


@dataclass
class HistoryEntry:
    provider: str
    identifier: str
    name: str
    lang: str  # "SUB" / "DUB"
    last_ep: float
    updated_at: float
    cover_url: str | None = None
    last_pos: float = 0.0
    last_duration: float = 0.0
    last_percent: float = 0.0
    completed: bool = False
    watched_eps: list[float] = field(default_factory=list)
    episode_progress: dict[str, dict[str, Any]] = field(default_factory=dict)


class HistoryStore:
    def __init__(self, path: str):
        self.path = path
        self._data: list[HistoryEntry] = []
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._data = [HistoryEntry(**x) for x in raw]
        except FileNotFoundError:
            self._data = []
        except Exception:
            try:
                if os.path.exists(self.path):
                    os.replace(self.path, self.path + ".broken")
            except Exception:
                pass
            self._data = []

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(x) for x in self._data],
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    def list(self) -> list[HistoryEntry]:
        return sorted(self._data, key=lambda x: x.updated_at, reverse=True)

    def upsert(self, entry: HistoryEntry) -> None:
        for i, e in enumerate(self._data):
            if (
                e.provider == entry.provider
                and e.identifier == entry.identifier
                and e.lang == entry.lang
            ):
                self._data[i] = entry
                self.save()
                return
        self._data.append(entry)
        self.save()

    def delete(self, provider: str, identifier: str, lang: str) -> None:
        self._data = [
            e
            for e in self._data
            if not (
                e.provider == provider
                and e.identifier == identifier
                and e.lang == lang
            )
        ]
        self.save()

