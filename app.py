from __future__ import annotations

import json
import os
import sys
import time
import shutil
import warnings
import socket
import tempfile
import platform
import subprocess
import urllib.request
import urllib.error
from urllib.parse import quote_plus, parse_qs, urlsplit
import hashlib
import importlib
import re
import zipfile
from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Callable
import errno
import threading
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEvent, QSize, QByteArray, QBuffer, QPropertyAnimation, QUrl, QProcess
from PySide6.QtGui import QShortcut, QKeySequence, QPixmap, QIcon, QColor, QPainter, QCursor, QImage, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QStackedWidget,
    QLabel,
    QSplitter,
    QMessageBox,
    QComboBox,
    QTabWidget,
    QSlider,
    QStyle,
    QGroupBox,
    QListWidgetItem,
    QListView,
    QFrame,
    QProgressBar,
    QGraphicsOpacityEffect,
    QAbstractItemView,
    QFileDialog,
    QSizePolicy,
)
from urllib3.exceptions import InsecureRequestWarning

from anipy_api.provider import get_provider, LanguageTypeEnum
from anipy_api.anime import Anime

# Suppress known noisy warning from upstream provider calls that use unverified TLS
# for api.allanime.day. Keep other TLS warnings untouched.
warnings.filterwarnings(
    "ignore",
    message=r"Unverified HTTPS request is being made to host 'api\.allanime\.day'.*",
    category=InsecureRequestWarning,
)


# ---------------------------
# Utils: paths + history store
# ---------------------------
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
APP_VERSION = "0.1.0"
UPDATE_MANIFEST_URL = os.getenv(
    "ANIGUI_UPDATE_MANIFEST_URL",
    "https://raw.githubusercontent.com/Zeyzers/anigui/main/update_manifest.json",
)


def debug_log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    ms = int((time.time() % 1) * 1000)
    thread_name = threading.current_thread().name
    print(f"[anigui {ts}.{ms:03d}] [{thread_name}] {msg}", flush=True)


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
            # in case of corruption: keep a backup and reset
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
        # newest first
        return sorted(self._data, key=lambda x: x.updated_at, reverse=True)

    def upsert(self, entry: HistoryEntry) -> None:
        # unique key = provider + identifier + lang
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


# ---------------------------
# Worker threads (network calls)
# ---------------------------
@dataclass
class SearchItem:
    name: str
    identifier: str
    languages: set[Any]
    cover_url: str | None = None
    source: str = "allanime"
    raw: Any = None
    year: int | None = None
    season: str | None = None
    studio: str | None = None
    rating: float | None = None

@dataclass
class OfflineAnimeItem:
    name: str
    folder: str
    cover_url: str | None = None

@dataclass
class FavoriteEntry:
    name: str
    identifier: str
    source: str
    cover_url: str | None = None
    added_at: float = 0.0

@dataclass
class StreamResult:
    url: str
    referrer: str | None = None
    sub_file: str | None = None


class MpvCmdWorker(QThread):
    done = Signal()
    fail = Signal(str)

    def __init__(self, fn: Callable, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            self.fn(*self.args)
            self.done.emit()
        except Exception as ex:
            debug_log(f"MpvCmdWorker fail: {ex}")
            self.fail.emit(str(ex))


class Worker(QThread):
    ok = Signal(object)
    err = Signal(str)

    def __init__(self, fn: Callable, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            res = self.fn(*self.args)
            self.ok.emit(res)
        except Exception as ex:
            debug_log(f"Worker err: fn={getattr(self.fn, '__name__', str(self.fn))} err={ex}")
            self.err.emit(str(ex))


@dataclass
class DownloadTask:
    task_id: str
    anime_name: str
    episode: float | int
    provider: str
    lang: Any
    quality: Any
    anime_obj: Any
    out_path: str
    status: str = "queued"  # queued/resolving/downloading/completed/failed/cancelled
    progress: float = 0.0
    downloaded: int = 0
    total: int | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 1


class DownloadWorker(QThread):
    progress = Signal(str, int, int, int)  # task_id, pct, downloaded, total(-1 unknown)
    done = Signal(str, str)  # task_id, out_path
    fail = Signal(str, str)  # task_id, error
    cancelled = Signal(str)  # task_id

    def __init__(self, task_id: str, url: str, out_path: str, referrer: str | None = None):
        super().__init__()
        self.task_id = task_id
        self.url = url
        self.out_path = out_path
        self.referrer = referrer
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        tmp_path = self.out_path + ".part"
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            req = urllib.request.Request(
                self.url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    **({"Referer": self.referrer} if self.referrer else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=30.0) as r, open(tmp_path, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                downloaded = 0
                while True:
                    if self._cancel_requested:
                        try:
                            f.close()
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        self.cancelled.emit(self.task_id)
                        return
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = int((downloaded / total) * 100) if total > 0 else -1
                    self.progress.emit(self.task_id, pct, downloaded, total)

            os.replace(tmp_path, self.out_path)
            self.done.emit(self.task_id, self.out_path)
        except Exception as ex:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            self.fail.emit(self.task_id, str(ex))


# ---------------------------
# Player Abstraction
# ---------------------------
class PlayerBase(QWidget):
    """Embedded player widget with basic controls."""

    def load(
        self,
        url: str,
        referrer: str | None = None,
        sub_file: str | None = None,
    ) -> None:
        raise NotImplementedError

    def play(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def toggle_pause(self) -> None:
        raise NotImplementedError

    def seek(self, seconds: float) -> None:
        raise NotImplementedError

    def seek_ratio(self, ratio: float) -> None:
        raise NotImplementedError

    def set_volume(self, vol: int) -> None:
        raise NotImplementedError

    def get_time_pos(self) -> float | None:
        raise NotImplementedError

    def get_duration(self) -> float | None:
        raise NotImplementedError

    def get_pause(self) -> bool | None:
        raise NotImplementedError

    def get_percent_pos(self) -> float | None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class FullscreenPlayerWindow(QWidget):
    exit_requested = Signal()
    activity = Signal()

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)

    def mouseMoveEvent(self, e):
        self.activity.emit()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        self.activity.emit()
        super().mousePressEvent(e)

    def resizeEvent(self, e):
        self.activity.emit()
        super().resizeEvent(e)

    def closeEvent(self, e):
        self.exit_requested.emit()
        e.ignore()


class MiniPlayerWindow(QWidget):
    exit_requested = Signal()

    def closeEvent(self, e):
        self.exit_requested.emit()
        e.ignore()


class LibMpvPlayer(PlayerBase):
    """Uses python-mpv (requires libmpv). Great if available."""

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)

        import mpv  # requires libmpv present

        self.mpv = mpv.MPV(
            wid=str(int(self.winId())),
            vo="gpu",
            hwdec="auto",
            input_default_bindings=True,
            input_vo_keyboard=True,
        )

    def load(
        self,
        url: str,
        referrer: str | None = None,
        sub_file: str | None = None,
    ) -> None:
        if referrer:
            self.mpv["referrer"] = referrer
        if sub_file:
            self.mpv.command("sub-add", sub_file, "select")
        self.mpv.play(url)

    def play(self) -> None:
        if platform.system() == "Windows" and hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            self._pending_pause = False
            return
        self._set_prop("pause", False)

    def pause(self) -> None:
        if platform.system() == "Windows" and hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            self._pending_pause = True
            return
        self._set_prop("pause", True)

    def toggle_pause(self) -> None:
        if platform.system() == "Windows" and hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            # durante load: inverti pending se c'è, altrimenti metti "pause"
            self._pending_pause = not bool(self._pending_pause) if self._pending_pause is not None else True
            return
        cur = self.get_pause()
        if cur is None:
            return
        self._set_prop("pause", not cur)


    def seek(self, seconds: float) -> None:
        if platform.system() == "Windows" and hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            self._pending_seek = float(seconds)
            return
        self._set_prop("time-pos", float(seconds))

    def seek_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        try:
            self.mpv.command("seek", ratio * 100.0, "absolute-percent")
        except Exception:
            self._set_prop("percent-pos", ratio * 100.0)


    def set_volume(self, vol: int) -> None:
        if platform.system() == "Windows" and hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
            self._pending_volume = int(vol)
            return
        self._set_prop("volume", int(vol))


    def get_time_pos(self) -> float | None:
        for name in ("time-pos", "time-pos/full", "playback-time"):
            try:
                v = self.mpv.get_property(name)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def get_duration(self) -> float | None:
        for name in ("duration", "duration/full"):
            try:
                v = self.mpv.get_property(name)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def get_pause(self) -> bool | None:
        try:
            v = self.mpv.get_property("pause")
            return bool(v) if v is not None else None
        except Exception:
            return None

    def get_percent_pos(self) -> float | None:
        try:
            v = self.mpv.get_property("percent-pos")
            return float(v) if v is not None else None
        except Exception:
            return None

    def stop(self) -> None:
        try:
            self.mpv.command("stop")
        except Exception:
            pass

    def closeEvent(self, e):
        try:
            self.mpv.terminate()
        except Exception:
            pass
        super().closeEvent(e)

    def _set_prop(self, name: str, value):
        try:
            self.mpv.set_property(name, value)
        except Exception:
            try:
                self.mpv[name] = value
            except Exception:
                pass


class MpvIpcWidPlayer(PlayerBase):
    """
    mpv subprocess embedded with --wid + IPC (single instance).
    Fixes: episode switching reliability + no background mpv.
    """

    def __init__(self, mpv_path: str):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self._pending_volume: int | None = None
        self._pending_seek: float | None = None
        self._pending_pause: bool | None = None
        self.mpv_path = mpv_path
        self.proc: subprocess.Popen | None = None
        self.ipc_path: str | None = None
        self._sock: socket.socket | None = None


    def _mk_ipc_path(self) -> str:
        if platform.system() == "Windows":
            return (
                r"\\.\pipe\anigui_mpv_"
                + str(os.getpid())
                + "_"
                + str(int(time.time() * 1000))
            )
        d = tempfile.mkdtemp(prefix="anigui_mpv_")
        return os.path.join(d, "ipc.sock")

    def _ensure_started(self) -> None:
        if self.proc and self.proc.poll() is None:
            debug_log("mpv process already running")
            return

        self.ipc_path = self._mk_ipc_path()
        wid = str(int(self.winId()))
        debug_log(f"Starting mpv process, wid={wid}, ipc={self.ipc_path}")

        cmd = [
            self.mpv_path,
            f"--wid={wid}",
            "--force-window=yes",
            "--idle=yes",  # keep mpv alive
            "--keep-open=yes",
            "--hwdec=auto",
            "--vo=gpu",
            "--osc=no",
            "--input-default-bindings=no",
            "--osd-level=0",
            "--no-terminal",
            f"--input-ipc-server={self.ipc_path}",
        ]

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        debug_log("mpv process started")

        # reset socket connection
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

        # wait a bit for IPC to appear
        time.sleep(0.2)
        debug_log("Initial wait for mpv IPC complete")

    def _connect_ipc(self, timeout_s: float = 3.0) -> None:
        if not self.ipc_path:
            raise RuntimeError("IPC path not set")

        start = time.time()
        while time.time() - start < timeout_s:
            try:
                if platform.system() == "Windows":
                    # On Windows we open the pipe per request; nothing to connect persistently.
                    return
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.ipc_path)
                self._sock = s
                return
            except Exception:
                time.sleep(0.05)

        raise RuntimeError("Could not connect to mpv IPC")

    def _ipc_send(self, payload: dict) -> dict | None:
        if not self.ipc_path:
            debug_log("IPC send skipped: ipc_path not set")
            return None

        if platform.system() == "Windows":
            data = (json.dumps(payload) + "\n").encode("utf-8")
            cmd_name = payload.get("command", ["?"])[0] if isinstance(payload.get("command"), list) else "?"

            # Retry veloce: evita freeze se la pipe non è pronta
            for _ in range(40):  # 40 * 0.05s = ~2 secondi max
                try:
                    with open(self.ipc_path, "r+b", buffering=0) as f:
                        f.write(data)
                        line = f.readline()

                    if not line:
                        debug_log(f"IPC empty response: command={cmd_name}")
                        return None

                    #debug_log(f"IPC response received: command={cmd_name}")
                    return json.loads(line.decode("utf-8", errors="ignore"))

                except OSError as e:
                    # pipe non pronta / busy / broken
                    winerr = getattr(e, "winerror", None)

                    # 2 = file not found (pipe non creata ancora)
                    # 231 = all pipe instances are busy
                    # 232 = the pipe is being closed / broken pipe
                    if winerr in (2, 231, 232) or e.errno == errno.ENOENT:
                        time.sleep(0.05)
                        continue

                    debug_log(f"IPC OSError non-retry: command={cmd_name} err={e}")
                    return None

                except Exception:
                    debug_log(f"IPC exception: command={cmd_name}")
                    return None

            debug_log(f"IPC timeout/retries exhausted: command={cmd_name}")
            return None

        try:
            if self._sock is None:
                self._connect_ipc()
            assert self._sock is not None
            self._sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            if not data:
                return None
            return json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return None

    def _get_prop(self, name: str):
        resp = self._ipc_send({"command": ["get_property", name]})
        return None if not resp else resp.get("data")

    def _set_prop(self, name: str, value):
        self._ipc_send({"command": ["set_property", name, value]})

    # ---- PlayerBase impl ----
    def load(self, url: str, referrer: str | None = None, sub_file: str | None = None) -> None:
        # su Windows NON fare IPC nel thread UI
        debug_log(
            f"Player load requested: url={url[:120]}{'...' if len(url) > 120 else ''}, "
            f"referrer={'yes' if referrer else 'no'}, sub={'yes' if sub_file else 'no'}"
        )
        if platform.system() == "Windows":
            if hasattr(self, "_load_thread") and self._load_thread and self._load_thread.isRunning():
                # se stai già caricando qualcosa, lascia perdere (oppure potresti terminare il thread)
                debug_log("Skipped load: previous load thread still running")
                return

            self._load_thread = MpvCmdWorker(self._load_blocking, url, referrer, sub_file)
            debug_log("Starting background mpv load thread")
            self._load_thread.start()
            return

        # su Linux/mac ok farlo sync
        self._load_blocking(url, referrer, sub_file)


    def _load_blocking(self, url: str, referrer: str | None, sub_file: str | None) -> None:
        t0 = time.perf_counter()
        debug_log("Load blocking start")
        self._ensure_started()
        debug_log(f"_ensure_started done in {time.perf_counter() - t0:.3f}s")

        # attende che la pipe risponda davvero (senza bloccare UI perché siamo in thread)
        start = time.time()
        while time.time() - start < 3.0:
            resp = self._ipc_send({"command": ["get_property", "mpv-version"]})
            if resp is not None:
                debug_log("IPC ready (mpv-version responded)")
                break
            time.sleep(0.05)
        else:
            debug_log("IPC readiness wait timed out (3s)")

        # referrer
        if referrer:
            self._set_prop("referrer", referrer)
        else:
            self._set_prop("referrer", "")
        debug_log("Referrer configured")

        # play
        self._ipc_send({"command": ["loadfile", url, "replace"]})
        debug_log("loadfile command sent")

        # subs
        if sub_file:
            self._ipc_send({"command": ["sub-add", sub_file, "select"]})
            debug_log("Subtitle command sent")
        # apply pending controls (if user moved sliders while loading)
        if self._pending_volume is not None:
            self._set_prop("volume", int(self._pending_volume))
            self._pending_volume = None

        if self._pending_pause is not None:
            self._set_prop("pause", bool(self._pending_pause))
            self._pending_pause = None

        if self._pending_seek is not None:
            self._set_prop("time-pos", float(self._pending_seek))
            self._pending_seek = None
        debug_log(f"Load blocking complete in {time.perf_counter() - t0:.3f}s")



    def play(self) -> None:
        self._set_prop("pause", False)

    def pause(self) -> None:
        self._set_prop("pause", True)

    def toggle_pause(self) -> None:
        cur = self.get_pause()
        if cur is None:
            return
        self._set_prop("pause", not cur)

    def seek(self, seconds: float) -> None:
        self._set_prop("time-pos", float(seconds))

    def seek_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        self._ipc_send({"command": ["seek", ratio * 100.0, "absolute-percent"]})

    def set_volume(self, vol: int) -> None:
        self._set_prop("volume", int(vol))

    def get_time_pos(self) -> float | None:
        for name in ("time-pos", "playback-time", "time-pos/full"):
            v = self._get_prop(name)
            try:
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def get_duration(self) -> float | None:
        for name in ("duration", "duration/full"):
            v = self._get_prop(name)
            try:
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    def get_pause(self) -> bool | None:
        v = self._get_prop("pause")
        if v is None:
            return None
        return bool(v)

    def get_percent_pos(self) -> float | None:
        v = self._get_prop("percent-pos")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def stop(self) -> None:
        # Stop playback but keep mpv alive (so next load works instantly)
        try:
            self._ipc_send({"command": ["stop"]})
        except Exception:
            pass

    def closeEvent(self, e):
        # terminate mpv process when closing app
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None
        super().closeEvent(e)


def _runtime_roots() -> list[str]:
    roots: list[str] = []
    try:
        if getattr(sys, "frozen", False):
            roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    except Exception:
        pass
    try:
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(os.path.abspath(str(meipass)))
    except Exception:
        pass
    roots.append(os.path.dirname(os.path.abspath(__file__)))

    out: list[str] = []
    seen: set[str] = set()
    for p in roots:
        n = os.path.normcase(os.path.normpath(p))
        if n in seen:
            continue
        seen.add(n)
        out.append(p)
    return out


def _resolve_mpv_path() -> str | None:
    exe_names = ["mpv.exe"] if platform.system() == "Windows" else ["mpv"]
    rel_candidates = []
    for name in exe_names:
        rel_candidates.append(name)
        rel_candidates.append(os.path.join("mpv", name))
        rel_candidates.append(os.path.join("bin", name))

    for root in _runtime_roots():
        for rel in rel_candidates:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p

    # On Windows prefer the real executable over mpv.com shim.
    if platform.system() == "Windows":
        return shutil.which("mpv.exe") or shutil.which("mpv")
    return shutil.which("mpv")


def create_player_widget() -> PlayerBase:
    prefer_libmpv = os.getenv("ANIGUI_PLAYER", "").strip().lower() == "libmpv"

    # default: mpv subprocess + IPC (piu affidabile per timeline/seek su stream web)
    mpv_path = _resolve_mpv_path()
    if mpv_path and not prefer_libmpv:
        debug_log(f"Using MpvIpcWidPlayer with mpv_path={mpv_path}")
        return MpvIpcWidPlayer(mpv_path)

    # fallback/opt-in: libmpv
    try:
        if platform.system() == "Windows":
            cur_path = os.environ.get("PATH", "")
            add_parts: list[str] = []
            for root in _runtime_roots():
                add_parts.append(root)
                add_parts.append(os.path.join(root, "mpv"))
                add_parts.append(os.path.join(root, "bin"))
            merged = os.pathsep.join([p for p in add_parts if p]) + os.pathsep + cur_path
            os.environ["PATH"] = merged
        debug_log("Trying LibMpvPlayer")
        return LibMpvPlayer()
    except Exception:
        debug_log("LibMpvPlayer unavailable, fallback to mpv IPC")

    # hard fallback: IPC
    if not mpv_path:
        raise RuntimeError(
            "mpv non trovato. Nella build standalone deve essere incluso in "
            "'mpv\\mpv.exe' accanto all'eseguibile."
        )
    debug_log(f"Using MpvIpcWidPlayer with mpv_path={mpv_path}")
    return MpvIpcWidPlayer(mpv_path)


# ---------------------------
# Main App
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anigui")
        self.resize(1300, 780)

        self._settings = self._load_settings()
        self.provider_name = str(self._settings.get("default_provider", "allanime"))
        if self.provider_name not in ("allanime", "aw_animeworld", "aw_animeunity"):
            self.provider_name = "allanime"
        lang_s = str(self._settings.get("default_lang", "SUB")).upper()
        self.lang = LanguageTypeEnum.DUB if lang_s == "DUB" else LanguageTypeEnum.SUB
        self.quality = str(self._settings.get("default_quality", "best"))
        if self.quality not in ("best", "worst", "360", "480", "720", "1080"):
            self.quality = "best"
        try:
            self._startup_parallel_downloads = int(self._settings.get("parallel_downloads", 2))
        except Exception:
            self._startup_parallel_downloads = 2
        self._startup_parallel_downloads = max(1, min(4, self._startup_parallel_downloads))
        self._scheduler_enabled = bool(self._settings.get("scheduler_enabled", False))
        self._scheduler_start = str(self._settings.get("scheduler_start", "00:00"))
        self._scheduler_end = str(self._settings.get("scheduler_end", "23:59"))
        try:
            self._integrity_min_mb = float(self._settings.get("integrity_min_mb", 2.0))
        except Exception:
            self._integrity_min_mb = 2.0
        self._integrity_min_mb = max(0.0, self._integrity_min_mb)
        try:
            self._integrity_retry_count = int(self._settings.get("integrity_retry_count", 1))
        except Exception:
            self._integrity_retry_count = 1
        self._integrity_retry_count = max(0, min(5, self._integrity_retry_count))
        self._anilist_enabled = bool(self._settings.get("anilist_enabled", False))
        self._anilist_token = str(self._settings.get("anilist_token", "")).strip()
        self.app_language = str(self._settings.get("app_language", "it")).strip().lower()
        if self.app_language not in ("it", "en"):
            self.app_language = "it"
        self._incognito_enabled = False

        self.selected_anime: Optional[Any] = None
        self.selected_search_item: Optional[SearchItem] = None
        self.episodes_list: list[float | int] = []
        self.current_ep: float | int | None = None
        self._aw_provider_instances: dict[str, Any] = {}
        self._aw_anime_cls: Any = None
        self._aw_config_ready = False
        self._aw_runtime_ready = False
        self._aw_cover_url_cache: dict[str, str | None] = {}
        self._aw_cover_miss_until: dict[str, float] = {}

        self.history = HistoryStore(HISTORY_PATH)
        self.cover_cache_dir = os.path.join(app_state_dir(), "covers")
        os.makedirs(self.cover_cache_dir, exist_ok=True)
        dl_dir = str(self._settings.get("download_dir", "")).strip()
        if dl_dir:
            self.download_dir = os.path.abspath(os.path.expanduser(dl_dir))
        else:
            self.download_dir = os.path.join(app_state_dir(), "downloads")
        os.makedirs(self.download_dir, exist_ok=True)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        QShortcut(QKeySequence("F"), self, activated=self.toggle_fullscreen)
        QShortcut(
            QKeySequence("Escape"),
            self,
            activated=lambda: self._exit_video_fullscreen() if self._is_video_fullscreen() else None,
        )

        # top bar
        top = QHBoxLayout()
        top.setSpacing(8)
        outer.addLayout(top)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cerca anime…")
        self.search_input.returnPressed.connect(self.on_search)
        self.search_input.textChanged.connect(self.on_search_text_changed)
        top.addWidget(self.search_input, 1)

        self.btn_search = QPushButton("🔎 Cerca")
        self.btn_search.clicked.connect(self.on_search)
        self.btn_search.setToolTip("Avvia ricerca anime")
        top.addWidget(self.btn_search)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("AllAnime", "allanime")
        self.provider_combo.addItem("AnimeWorld ITA", "aw_animeworld")
        self.provider_combo.addItem("AnimeUnity ITA", "aw_animeunity")
        self.provider_combo.currentIndexChanged.connect(self.on_provider_change)
        top.addWidget(self.provider_combo)

        self.lang_combo = QComboBox()
        self.lang_combo.addItem("SUB", LanguageTypeEnum.SUB)
        self.lang_combo.addItem("DUB", LanguageTypeEnum.DUB)
        self.lang_combo.currentIndexChanged.connect(self.on_lang_change)
        top.addWidget(self.lang_combo)

        self.quality_combo = QComboBox()
        for q in ["best", "worst", "360", "480", "720", "1080"]:
            self.quality_combo.addItem(q)
        self.quality_combo.currentIndexChanged.connect(self.on_quality_change)
        top.addWidget(self.quality_combo)

        self.provider_combo.blockSignals(True)
        self.lang_combo.blockSignals(True)
        self.quality_combo.blockSignals(True)
        pidx = self.provider_combo.findData(self.provider_name)
        if pidx >= 0:
            self.provider_combo.setCurrentIndex(pidx)
        self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.quality_combo.findText(self.quality)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)
        self.provider_combo.blockSignals(False)
        self.lang_combo.blockSignals(False)
        self.quality_combo.blockSignals(False)
        self._apply_provider_ui_state()

        self.lbl_spinner = QLabel("")
        self.lbl_spinner.setFixedWidth(24)
        top.addWidget(self.lbl_spinner)

        self.btn_incognito = QPushButton("Incognito OFF")
        self.btn_incognito.setCheckable(True)
        self.btn_incognito.toggled.connect(self.on_toggle_incognito)
        self.btn_incognito.setToolTip("Disabilita salvataggi locali e cache")
        top.addWidget(self.btn_incognito)
        self.lbl_incognito_badge = QLabel("INCOGNITO")
        self.lbl_incognito_badge.setObjectName("incognitoBadge")
        self.lbl_incognito_badge.setVisible(False)
        top.addWidget(self.lbl_incognito_badge)

        # tabs: Search / History
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        # ---- Search tab ----
        self.tab_search = QWidget()
        self.tabs.addTab(self.tab_search, "Search")
        search_layout = QVBoxLayout(self.tab_search)

        self.suggestions_panel = QWidget()
        self.suggestions_panel.setObjectName("suggestionsPanel")
        self.suggestions_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sugg_layout = QVBoxLayout(self.suggestions_panel)
        sugg_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_suggestions = QLabel("Suggerimenti")
        self.lbl_suggestions.setObjectName("sectionTitle")
        sugg_layout.addWidget(self.lbl_suggestions)

        sugg_row = QHBoxLayout()
        self.list_recent_queries = QListWidget()
        self.list_recent_queries.setObjectName("suggestionsList")
        self.list_recent_queries.itemClicked.connect(self.on_pick_recent_query)
        self.list_recent_queries.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list_recent_queries.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list_recent_queries.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sugg_row.addWidget(self.list_recent_queries, 1)

        sugg_actions = QHBoxLayout()
        self.btn_clear_search_history = QPushButton("🧹 Clear search history")
        self.btn_clear_search_history.clicked.connect(self.on_clear_search_history)
        sugg_actions.addStretch(1)
        sugg_actions.addWidget(self.btn_clear_search_history)
        sugg_layout.addLayout(sugg_actions)

        sugg_layout.addLayout(sugg_row, 1)
        search_layout.addWidget(self.suggestions_panel)

        self.search_stack = QStackedWidget()
        search_layout.addWidget(self.search_stack, 1)

        # catalog page (full tab)
        self.page_catalog = QWidget()
        cat_layout = QVBoxLayout(self.page_catalog)
        cat_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_search_meta = QLabel("")
        self.lbl_search_meta.setObjectName("sectionTitle")
        self.lbl_search_meta.setVisible(False)
        cat_layout.addWidget(self.lbl_search_meta)
        self.lbl_search_state = QLabel("")
        self.lbl_search_state.setObjectName("mutedLabel")
        self.lbl_search_state.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_search_state.setVisible(False)
        cat_layout.addWidget(self.lbl_search_state)

        filters_row = QHBoxLayout()
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Rilevanza", "Titolo A→Z", "Titolo Z→A"])
        self.combo_sort.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_sort)

        self.combo_season = QComboBox()
        self.combo_season.addItems(["Stagione: Qualsiasi", "Winter", "Spring", "Summer", "Fall"])
        self.combo_season.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_season)

        self.combo_year = QComboBox()
        self.combo_year.addItem("Anno: Qualsiasi")
        current_year = time.localtime().tm_year
        for y in range(current_year, current_year - 25, -1):
            self.combo_year.addItem(str(y))
        self.combo_year.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_year)

        self.input_studio = QLineEdit()
        self.input_studio.setPlaceholderText("Studio")
        self.input_studio.textChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.input_studio)

        self.combo_rating = QComboBox()
        self.combo_rating.addItems(["Rating: Qualsiasi", ">= 9", ">= 8", ">= 7", ">= 6"])
        self.combo_rating.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_rating)
        self._search_filter_widgets = [
            self.combo_sort,
            self.combo_season,
            self.combo_year,
            self.input_studio,
            self.combo_rating,
        ]

        cat_layout.addLayout(filters_row)

        self.results = QListWidget()
        self.results.itemClicked.connect(self.on_pick_result)
        self.results.setViewMode(QListView.ViewMode.IconMode)
        self.results.setResizeMode(QListView.ResizeMode.Adjust)
        self.results.setMovement(QListView.Movement.Static)
        self.results.setWrapping(True)
        self.results.setSpacing(14)
        self.results.setIconSize(QSize(150, 220))
        self.results.setWordWrap(True)
        cat_layout.addWidget(self.results, 1)

        self.search_stack.addWidget(self.page_catalog)

        # anime page (episodes list only)
        self.page_anime = QWidget()
        anime_layout = QVBoxLayout(self.page_anime)
        anime_layout.setContentsMargins(0, 0, 0, 0)

        anime_header = QHBoxLayout()
        self.btn_back_catalog = QPushButton("← Back to Search")
        self.btn_back_catalog.clicked.connect(self.on_back_to_catalog)
        anime_header.addWidget(self.btn_back_catalog, 0)
        self.lbl_anime_title = QLabel("Anime")
        self.lbl_anime_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        anime_header.addWidget(self.lbl_anime_title, 1)
        self.btn_queue_selected_eps = QPushButton("⬇ Queue selected")
        self.btn_queue_selected_eps.clicked.connect(self.on_download_add_selected_episodes)
        anime_header.addWidget(self.btn_queue_selected_eps, 0)
        self.btn_fav_anime = QPushButton("❤ Favorite")
        self.btn_fav_anime.clicked.connect(self.on_favorite_add_current)
        anime_header.addWidget(self.btn_fav_anime, 0)
        anime_layout.addLayout(anime_header)

        self.episodes = QListWidget()
        self.episodes.itemDoubleClicked.connect(self.on_play_selected_episode)
        self.episodes.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.episodes.setSpacing(8)
        anime_layout.addWidget(QLabel("Episodi"), 0)
        anime_layout.addWidget(self.episodes, 1)
        self.search_stack.addWidget(self.page_anime)

        # player page (full area)
        self.page_player = QWidget()
        player_page_layout = QVBoxLayout(self.page_player)
        player_page_layout.setContentsMargins(0, 0, 0, 0)
        player_page_layout.setSpacing(8)

        player_header = QHBoxLayout()
        self.btn_back_episodes = QPushButton("← Back to Episodes")
        self.btn_back_episodes.clicked.connect(self.on_back_to_episodes)
        player_header.addWidget(self.btn_back_episodes, 0)
        self.lbl_player_title = QLabel("Player")
        self.lbl_player_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        player_header.addWidget(self.lbl_player_title, 1)
        player_page_layout.addLayout(player_header)

        self.player_section = QWidget()
        player_section_layout = QVBoxLayout(self.player_section)
        player_section_layout.setContentsMargins(0, 0, 0, 0)
        player_section_layout.setSpacing(8)

        self.player = create_player_widget()
        self.player_host = QWidget()
        self.player_host_layout = QVBoxLayout(self.player_host)
        self.player_host_layout.setContentsMargins(0, 0, 0, 0)
        self.player_host_layout.setSpacing(0)
        self.player_host_layout.addWidget(self.player, 1)
        player_section_layout.addWidget(self.player_host, 1)

        # controls group
        controls = QGroupBox("Controls")
        c = QVBoxLayout(controls)

        row1 = QHBoxLayout()

        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.clicked.connect(self.on_prev_episode)
        row1.addWidget(self.btn_prev)

        self.btn_playpause = QPushButton("⏯ Play/Pause")
        self.btn_playpause.clicked.connect(self.on_toggle_pause)
        row1.addWidget(self.btn_playpause)

        self.btn_next = QPushButton("⏭ Next")
        self.btn_next.clicked.connect(self.on_next_episode)
        row1.addWidget(self.btn_next)

        self.btn_fs = QPushButton("⛶ Fullscreen")
        self.btn_fs.clicked.connect(self.toggle_fullscreen)
        row1.addWidget(self.btn_fs)

        self.btn_mini = QPushButton("▣ Mini")
        self.btn_mini.clicked.connect(self.toggle_mini_player)
        row1.addWidget(self.btn_mini)

        c.addLayout(row1)

        # seek slider + labels
        row2 = QHBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        row2.addWidget(self.lbl_time)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        row2.addWidget(self.seek_slider, 1)
        self._resume_marker = QFrame(self.seek_slider)
        self._resume_marker.setFixedWidth(2)
        self._resume_marker.setStyleSheet("background: #e50914;")
        self._resume_marker.hide()
        self._resume_marker_pos: float | None = None
        self.seek_slider.installEventFilter(self)
        c.addLayout(row2)

        # volume
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Volume"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 130)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self.on_volume_changed)
        row3.addWidget(self.vol_slider, 1)
        c.addLayout(row3)

        player_section_layout.addWidget(controls, 0)
        player_page_layout.addWidget(self.player_section, 1)
        self.search_stack.addWidget(self.page_player)
        self.search_stack.setCurrentWidget(self.page_catalog)

        # ---- Recommended tab ----
        self.tab_recommended = QWidget()
        self.tabs.addTab(self.tab_recommended, "Recommended")
        rec_layout = QVBoxLayout(self.tab_recommended)

        rec_top = QHBoxLayout()
        self.lbl_recommended = QLabel("Recent Releases + Recommended")
        rec_top.addWidget(self.lbl_recommended, 1)
        self.btn_refresh_recommended = QPushButton("↻ Aggiorna")
        self.btn_refresh_recommended.clicked.connect(self.on_refresh_recommended)
        rec_top.addWidget(self.btn_refresh_recommended, 0)
        rec_layout.addLayout(rec_top)

        rec_split = QSplitter(Qt.Orientation.Horizontal)
        rec_layout.addWidget(rec_split, 1)

        recent_col = QWidget()
        recent_col_layout = QVBoxLayout(recent_col)
        recent_col_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_recent_header = QLabel("Recent Releases")
        self.lbl_recent_header.setObjectName("recentHeader")
        recent_col_layout.addWidget(self.lbl_recent_header)

        self.recent = QListWidget()
        self.recent.itemClicked.connect(self.on_pick_recent)
        self.recent.setViewMode(QListView.ViewMode.IconMode)
        self.recent.setResizeMode(QListView.ResizeMode.Adjust)
        self.recent.setMovement(QListView.Movement.Static)
        self.recent.setWrapping(True)
        self.recent.setSpacing(14)
        self.recent.setIconSize(QSize(150, 220))
        self.recent.setWordWrap(True)
        recent_col_layout.addWidget(self.recent, 1)
        rec_split.addWidget(recent_col)

        recommended_col = QWidget()
        recommended_col_layout = QVBoxLayout(recommended_col)
        recommended_col_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_recommended_header = QLabel("Recommended")
        self.lbl_recommended_header.setObjectName("recommendedHeader")
        recommended_col_layout.addWidget(self.lbl_recommended_header)

        self.recommended = QListWidget()
        self.recommended.itemClicked.connect(self.on_pick_recommended)
        self.recommended.setViewMode(QListView.ViewMode.IconMode)
        self.recommended.setResizeMode(QListView.ResizeMode.Adjust)
        self.recommended.setMovement(QListView.Movement.Static)
        self.recommended.setWrapping(True)
        self.recommended.setSpacing(14)
        self.recommended.setIconSize(QSize(150, 220))
        self.recommended.setWordWrap(True)
        recommended_col_layout.addWidget(self.recommended, 1)
        rec_split.addWidget(recommended_col)
        rec_split.setStretchFactor(0, 1)
        rec_split.setStretchFactor(1, 1)

        # ---- Downloads tab ----
        self.tab_downloads = QWidget()
        self.tabs.addTab(self.tab_downloads, "Downloads")
        dl_layout = QVBoxLayout(self.tab_downloads)

        dl_top = QHBoxLayout()
        self.btn_dl_add_current = QPushButton("＋ Aggiungi episodio corrente")
        self.btn_dl_add_current.clicked.connect(self.on_download_add_current)
        dl_top.addWidget(self.btn_dl_add_current)

        self.btn_dl_start = QPushButton("▶ Avvia coda")
        self.btn_dl_start.clicked.connect(self.on_download_start_queue)
        self.btn_dl_start.setToolTip("Avvia i download in coda")
        dl_top.addWidget(self.btn_dl_start)

        self.combo_dl_parallel = QComboBox()
        self.combo_dl_parallel.addItem("Parallel x1", 1)
        self.combo_dl_parallel.addItem("Parallel x2", 2)
        self.combo_dl_parallel.addItem("Parallel x3", 3)
        self.combo_dl_parallel.addItem("Parallel x4", 4)
        self.combo_dl_parallel.setCurrentIndex(self._startup_parallel_downloads - 1)
        self.combo_dl_parallel.currentIndexChanged.connect(self.on_download_parallel_change)
        dl_top.addWidget(self.combo_dl_parallel)

        self.btn_dl_cancel = QPushButton("✖ Annulla selezionato")
        self.btn_dl_cancel.clicked.connect(self.on_download_cancel_selected)
        dl_top.addWidget(self.btn_dl_cancel)

        self.btn_dl_clear = QPushButton("🧹 Pulisci completati")
        self.btn_dl_clear.clicked.connect(self.on_download_clear_completed)
        dl_top.addWidget(self.btn_dl_clear)

        self.btn_dl_open = QPushButton("📂 Open folder")
        self.btn_dl_open.clicked.connect(self.on_download_open_folder)
        dl_top.addWidget(self.btn_dl_open)
        dl_layout.addLayout(dl_top)

        self.downloads_list = QListWidget()
        dl_layout.addWidget(self.downloads_list, 1)

        # ---- Offline tab ----
        self.tab_offline = QWidget()
        self.tabs.addTab(self.tab_offline, "Offline")
        off_layout = QVBoxLayout(self.tab_offline)

        off_top = QHBoxLayout()
        self.lbl_offline = QLabel("Anime scaricati (stream locale)")
        off_top.addWidget(self.lbl_offline, 1)
        self.btn_offline_refresh = QPushButton("↻ Aggiorna")
        self.btn_offline_refresh.clicked.connect(self.refresh_offline_library)
        off_top.addWidget(self.btn_offline_refresh, 0)
        self.btn_offline_open = QPushButton("📂 Open downloads")
        self.btn_offline_open.clicked.connect(self.on_download_open_folder)
        off_top.addWidget(self.btn_offline_open, 0)
        off_layout.addLayout(off_top)

        self.offline_stack = QStackedWidget()
        off_layout.addWidget(self.offline_stack, 1)

        self.page_offline_catalog = QWidget()
        off_cat_layout = QVBoxLayout(self.page_offline_catalog)
        off_cat_layout.setContentsMargins(0, 0, 0, 0)
        off_cat_layout.addWidget(QLabel("Catalogo Offline"))
        self.offline_results = QListWidget()
        self.offline_results.itemClicked.connect(self.on_pick_offline_anime)
        self.offline_results.setViewMode(QListView.ViewMode.IconMode)
        self.offline_results.setResizeMode(QListView.ResizeMode.Adjust)
        self.offline_results.setMovement(QListView.Movement.Static)
        self.offline_results.setWrapping(True)
        self.offline_results.setSpacing(14)
        self.offline_results.setIconSize(QSize(150, 220))
        self.offline_results.setWordWrap(True)
        self.offline_results.setObjectName("resultsList")
        off_cat_layout.addWidget(self.offline_results, 1)
        self.offline_stack.addWidget(self.page_offline_catalog)

        self.page_offline_anime = QWidget()
        off_anime_layout = QVBoxLayout(self.page_offline_anime)
        off_anime_layout.setContentsMargins(0, 0, 0, 0)
        off_header = QHBoxLayout()
        self.btn_offline_back = QPushButton("← Back to Offline Catalog")
        self.btn_offline_back.clicked.connect(self.on_offline_back_to_catalog)
        off_header.addWidget(self.btn_offline_back, 0)
        self.lbl_offline_anime_title = QLabel("Anime")
        off_header.addWidget(self.lbl_offline_anime_title, 1)
        off_anime_layout.addLayout(off_header)
        self.offline_episodes = QListWidget()
        self.offline_episodes.itemDoubleClicked.connect(self.on_play_offline_episode)
        self.offline_episodes.itemClicked.connect(self.on_play_offline_episode)
        self.offline_episodes.setObjectName("episodesList")
        off_anime_layout.addWidget(QLabel("Episodi / File"), 0)
        off_anime_layout.addWidget(self.offline_episodes, 1)
        self.offline_stack.addWidget(self.page_offline_anime)
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)

        # ---- Favorites tab ----
        self.tab_favorites = QWidget()
        self.tabs.addTab(self.tab_favorites, "Favorites")
        fav_layout = QVBoxLayout(self.tab_favorites)
        fav_top = QHBoxLayout()
        fav_top.addWidget(QLabel("Watchlist / Favorites"), 1)
        self.btn_fav_add_current = QPushButton("❤ Add current anime")
        self.btn_fav_add_current.clicked.connect(self.on_favorite_add_current)
        fav_top.addWidget(self.btn_fav_add_current, 0)
        self.btn_fav_remove = QPushButton("🗑 Remove selected")
        self.btn_fav_remove.clicked.connect(self.on_favorite_remove_selected)
        fav_top.addWidget(self.btn_fav_remove, 0)
        fav_layout.addLayout(fav_top)
        self.favorites_list = QListWidget()
        self.favorites_list.itemClicked.connect(self.on_pick_favorite)
        self.favorites_list.setViewMode(QListView.ViewMode.IconMode)
        self.favorites_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.favorites_list.setMovement(QListView.Movement.Static)
        self.favorites_list.setWrapping(True)
        self.favorites_list.setSpacing(14)
        self.favorites_list.setIconSize(QSize(150, 220))
        self.favorites_list.setWordWrap(True)
        self.favorites_list.setObjectName("resultsList")
        fav_layout.addWidget(self.favorites_list, 1)

        # ---- Settings tab ----
        self.tab_settings = QWidget()
        self.tabs.addTab(self.tab_settings, "Settings")
        settings_layout = QVBoxLayout(self.tab_settings)

        row_dl = QHBoxLayout()
        self.lbl_settings_download_dir = QLabel("Download directory")
        row_dl.addWidget(self.lbl_settings_download_dir, 0)
        self.input_settings_download_dir = QLineEdit(self.download_dir)
        row_dl.addWidget(self.input_settings_download_dir, 1)
        self.btn_settings_browse_download_dir = QPushButton("Sfoglia")
        self.btn_settings_browse_download_dir.clicked.connect(self.on_settings_browse_download_dir)
        row_dl.addWidget(self.btn_settings_browse_download_dir, 0)
        settings_layout.addLayout(row_dl)

        row_defaults = QHBoxLayout()
        self.lbl_settings_default_provider = QLabel("Default provider")
        row_defaults.addWidget(self.lbl_settings_default_provider, 0)
        self.combo_settings_provider = QComboBox()
        self.combo_settings_provider.addItem("AllAnime", "allanime")
        self.combo_settings_provider.addItem("AnimeWorld ITA", "aw_animeworld")
        self.combo_settings_provider.addItem("AnimeUnity ITA", "aw_animeunity")
        row_defaults.addWidget(self.combo_settings_provider, 1)
        self.lbl_settings_default_lang = QLabel("Default language")
        row_defaults.addWidget(self.lbl_settings_default_lang, 0)
        self.combo_settings_lang = QComboBox()
        self.combo_settings_lang.addItem("SUB", "SUB")
        self.combo_settings_lang.addItem("DUB", "DUB")
        row_defaults.addWidget(self.combo_settings_lang, 1)
        settings_layout.addLayout(row_defaults)

        row_app_lang = QHBoxLayout()
        self.lbl_settings_app_language = QLabel("App language")
        row_app_lang.addWidget(self.lbl_settings_app_language, 0)
        self.combo_settings_app_language = QComboBox()
        self.combo_settings_app_language.addItem("Italiano", "it")
        self.combo_settings_app_language.addItem("English", "en")
        row_app_lang.addWidget(self.combo_settings_app_language, 1)
        row_app_lang.addStretch(1)
        settings_layout.addLayout(row_app_lang)

        row_quality = QHBoxLayout()
        self.lbl_settings_default_quality = QLabel("Default quality")
        row_quality.addWidget(self.lbl_settings_default_quality, 0)
        self.combo_settings_quality = QComboBox()
        for q in ["best", "worst", "360", "480", "720", "1080"]:
            self.combo_settings_quality.addItem(q, q)
        row_quality.addWidget(self.combo_settings_quality, 1)
        self.lbl_settings_parallel_downloads = QLabel("Parallel downloads")
        row_quality.addWidget(self.lbl_settings_parallel_downloads, 0)
        self.combo_settings_parallel = QComboBox()
        self.combo_settings_parallel.addItem("1", 1)
        self.combo_settings_parallel.addItem("2", 2)
        self.combo_settings_parallel.addItem("3", 3)
        self.combo_settings_parallel.addItem("4", 4)
        row_quality.addWidget(self.combo_settings_parallel, 1)
        settings_layout.addLayout(row_quality)

        row_sched = QHBoxLayout()
        self.lbl_settings_scheduler = QLabel("Scheduler")
        row_sched.addWidget(self.lbl_settings_scheduler, 0)
        self.combo_settings_scheduler_enabled = QComboBox()
        self.combo_settings_scheduler_enabled.addItem("Disabled", False)
        self.combo_settings_scheduler_enabled.addItem("Enabled", True)
        row_sched.addWidget(self.combo_settings_scheduler_enabled, 1)
        self.lbl_settings_scheduler_start = QLabel("Start HH:MM")
        row_sched.addWidget(self.lbl_settings_scheduler_start, 0)
        self.input_settings_scheduler_start = QLineEdit(self._scheduler_start)
        self.input_settings_scheduler_start.setMaxLength(5)
        row_sched.addWidget(self.input_settings_scheduler_start, 1)
        self.lbl_settings_scheduler_end = QLabel("End HH:MM")
        row_sched.addWidget(self.lbl_settings_scheduler_end, 0)
        self.input_settings_scheduler_end = QLineEdit(self._scheduler_end)
        self.input_settings_scheduler_end.setMaxLength(5)
        row_sched.addWidget(self.input_settings_scheduler_end, 1)
        settings_layout.addLayout(row_sched)

        row_integrity = QHBoxLayout()
        self.lbl_settings_integrity_min_mb = QLabel("Integrity min MB")
        row_integrity.addWidget(self.lbl_settings_integrity_min_mb, 0)
        self.input_settings_integrity_min_mb = QLineEdit(str(self._integrity_min_mb))
        row_integrity.addWidget(self.input_settings_integrity_min_mb, 1)
        self.lbl_settings_retry_count = QLabel("Retry count")
        row_integrity.addWidget(self.lbl_settings_retry_count, 0)
        self.combo_settings_integrity_retries = QComboBox()
        for i in range(0, 6):
            self.combo_settings_integrity_retries.addItem(str(i), i)
        row_integrity.addWidget(self.combo_settings_integrity_retries, 1)
        settings_layout.addLayout(row_integrity)

        row_anilist = QHBoxLayout()
        self.lbl_settings_anilist_sync = QLabel("AniList sync")
        row_anilist.addWidget(self.lbl_settings_anilist_sync, 0)
        self.combo_settings_anilist_enabled = QComboBox()
        self.combo_settings_anilist_enabled.addItem("Disabled", False)
        self.combo_settings_anilist_enabled.addItem("Enabled", True)
        row_anilist.addWidget(self.combo_settings_anilist_enabled, 1)
        self.lbl_settings_anilist_token = QLabel("AniList token")
        row_anilist.addWidget(self.lbl_settings_anilist_token, 0)
        self.input_settings_anilist_token = QLineEdit(self._anilist_token)
        self.input_settings_anilist_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_settings_anilist_token.setPlaceholderText("Personal Access Token")
        row_anilist.addWidget(self.input_settings_anilist_token, 2)
        settings_layout.addLayout(row_anilist)

        row_actions = QHBoxLayout()
        self.btn_settings_save = QPushButton("Salva impostazioni")
        self.btn_settings_save.clicked.connect(self.on_settings_save)
        row_actions.addWidget(self.btn_settings_save, 0)
        self.btn_settings_reset = QPushButton("Reset default")
        self.btn_settings_reset.clicked.connect(self.on_settings_reset)
        row_actions.addWidget(self.btn_settings_reset, 0)
        self.btn_settings_backup = QPushButton("Backup")
        self.btn_settings_backup.clicked.connect(self.on_settings_backup)
        row_actions.addWidget(self.btn_settings_backup, 0)
        self.btn_settings_restore = QPushButton("Restore")
        self.btn_settings_restore.clicked.connect(self.on_settings_restore)
        row_actions.addWidget(self.btn_settings_restore, 0)
        self.btn_settings_anilist_test = QPushButton("Test connessione AniList")
        self.btn_settings_anilist_test.clicked.connect(self.on_anilist_test_connection)
        row_actions.addWidget(self.btn_settings_anilist_test, 0)
        self.btn_settings_anilist_sync_now = QPushButton("Sync now")
        self.btn_settings_anilist_sync_now.clicked.connect(self.on_anilist_sync_now)
        self.btn_settings_anilist_sync_now.setToolTip("Invia subito il progresso corrente ad AniList")
        row_actions.addWidget(self.btn_settings_anilist_sync_now, 0)
        self.btn_settings_anilist_pull = QPushButton("Importa da AniList")
        self.btn_settings_anilist_pull.clicked.connect(self.on_anilist_pull_from_remote)
        self.btn_settings_anilist_pull.setToolTip("Importa watchlist/progressi da AniList")
        row_actions.addWidget(self.btn_settings_anilist_pull, 0)
        self.btn_settings_update_check = QPushButton("Check updates")
        self.btn_settings_update_check.clicked.connect(self.on_update_check_clicked)
        row_actions.addWidget(self.btn_settings_update_check, 0)
        self.btn_settings_update_apply = QPushButton("Apply update")
        self.btn_settings_update_apply.clicked.connect(self.on_update_apply_clicked)
        self.btn_settings_update_apply.setEnabled(False)
        row_actions.addWidget(self.btn_settings_update_apply, 0)
        row_actions.addStretch(1)
        settings_layout.addLayout(row_actions)
        settings_layout.addStretch(1)

        # ---- History tab ----
        self.tab_history = QWidget()
        self.tabs.addTab(self.tab_history, "Watchlist")
        hist_layout = QVBoxLayout(self.tab_history)

        hist_top = QHBoxLayout()
        self.lbl_history_header = QLabel("Watchlist: Planned · Watching · Completed")
        hist_top.addWidget(self.lbl_history_header, 1)
        self.lbl_history_filter = QLabel("Filter")
        hist_top.addWidget(self.lbl_history_filter, 0)
        self.combo_history_filter = QComboBox()
        self.combo_history_filter.addItem("All", "All")
        self.combo_history_filter.addItem("Completed", "Completed")
        self.combo_history_filter.addItem("Watching", "Watching")
        self.combo_history_filter.addItem("Planned", "Planned")
        self.combo_history_filter.currentIndexChanged.connect(self.on_history_filter_changed)
        hist_top.addWidget(self.combo_history_filter, 0)
        hist_layout.addLayout(hist_top)

        self.hist_list = QListWidget()
        self.hist_list.itemClicked.connect(self.on_pick_history)
        self.hist_list.itemDoubleClicked.connect(self.on_history_open_details)
        self.hist_list.setViewMode(QListView.ViewMode.IconMode)
        self.hist_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.hist_list.setMovement(QListView.Movement.Static)
        self.hist_list.setWrapping(True)
        self.hist_list.setFlow(QListView.Flow.LeftToRight)
        self.hist_list.setSpacing(8)
        self.hist_list.setWordWrap(False)
        self.hist_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hist_list.setObjectName("continueList")
        self.hist_list.verticalScrollBar().valueChanged.connect(self._position_history_section_overlays)
        hist_layout.addWidget(self.hist_list, 1)

        hist_buttons = QHBoxLayout()

        self.btn_hist_resume = QPushButton("▶ Resume selected")
        self.btn_hist_resume.clicked.connect(self.on_history_resume)
        hist_buttons.addWidget(self.btn_hist_resume)


        self.btn_hist_resume_next = QPushButton("⏭ Resume next")
        self.btn_hist_resume_next.clicked.connect(self.on_history_resume_next)
        hist_buttons.addWidget(self.btn_hist_resume_next)

        self.btn_hist_mark_seen = QPushButton("✅ Mark as seen")
        self.btn_hist_mark_seen.clicked.connect(self.on_history_mark_seen)
        hist_buttons.addWidget(self.btn_hist_mark_seen)

        self.btn_hist_delete = QPushButton("🗑 Remove")
        self.btn_hist_delete.clicked.connect(self.on_history_delete)
        hist_buttons.addWidget(self.btn_hist_delete)

        hist_layout.addLayout(hist_buttons)

        hist_clear = QHBoxLayout()
        self.btn_clear_watch_history = QPushButton("🧹 Clear watch history")
        self.btn_clear_watch_history.clicked.connect(self.on_clear_watch_history)
        hist_clear.addWidget(self.btn_clear_watch_history)
        hist_layout.addLayout(hist_clear)

        # status
        self.status = QLabel("Pronto.")
        self.status.setObjectName("statusBar")
        outer.addWidget(self.status)

        # internal
        self._workers: list[Worker] = []
        self._result_items: list[SearchItem] = []
        self._history_items: list[HistoryEntry] = []
        self._history_filter = "All"
        self._history_seen_eps_for_ui: set[float] | None = None
        self._history_episode_progress_for_ui: dict[float, dict[str, Any]] | None = None
        self._history_cover_labels: dict[int, QLabel] = {}
        self._history_section_overlays: dict[int, QWidget] = {}
        self._update_info: dict[str, Any] | None = None
        self._update_download_path: str | None = None
        self._search_cards: list[QListWidgetItem] = []
        self._recent_items: list[SearchItem] = []
        self._recent_cards: list[QListWidgetItem] = []
        self._recommended_items: list[SearchItem] = []
        self._recommended_cards: list[QListWidgetItem] = []
        self._favorite_items: list[FavoriteEntry] = []
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._download_tasks: dict[str, DownloadTask] = {}
        self._download_order: list[str] = []
        self._active_download_workers: dict[str, DownloadWorker] = {}
        self._active_download_resolve_ids: set[str] = set()
        self._max_parallel_downloads = self._startup_parallel_downloads
        self._search_nonce = 0
        self._recent_nonce = 0
        self._recommended_nonce = 0
        self._recommended_loaded = False
        self._current_search_view = "catalog"
        self._recent_queries: list[str] = []
        self._offline_covers_map: dict[str, str] = {}
        self._last_search_query: str | None = None
        self._pending_search_query: str | None = None
        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.setInterval(350)
        self._search_debounce_timer.timeout.connect(self._run_debounced_search)
        self._filters_pending = False
        self._enrich_nonce = 0
        self._filtering_nonce = 0
        self._result_items_raw: list[SearchItem] = []
        self._requests_in_flight = 0
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._spin_tick)
        self._skeleton_phase = 0
        self._skeleton_timer = QTimer(self)
        self._skeleton_timer.setInterval(120)
        self._skeleton_timer.timeout.connect(self._update_skeletons)
        self._last_timeline_diag_at = 0.0
        self._timeline_missing_logged = False
        self._last_progress_save_at = 0.0
        self._last_progress_pos = -1.0
        self._last_progress_percent = -1.0
        self._pending_resume_seek: float | None = None
        self._pending_resume_seek_ratio: float | None = None
        self._pending_resume_seek_attempts = 0
        self._local_media_active = False
        self._offline_items: list[OfflineAnimeItem] = []
        self._offline_episode_files: list[str] = []
        self._offline_current_anime_dir: str | None = None
        self._offline_current_episode_index: int | None = None
        self._offline_nonce = 0
        self._anilist_media_id_cache: dict[str, int] = {}
        self._anilist_last_sync_ts: dict[str, float] = {}
        self._anilist_last_synced_progress: dict[str, int] = {}
        self._mini_player_window: QWidget | None = None
        self._video_fs_window: FullscreenPlayerWindow | None = None
        self._video_fs_shortcuts: list[QShortcut] = []
        self._closing_video_fs = False
        self._fs_overlay: QWidget | None = None
        self._fs_overlay_effect: QGraphicsOpacityEffect | None = None
        self._fs_overlay_anim: QPropertyAnimation | None = None
        self._fs_seek_slider: QSlider | None = None
        self._fs_vol_slider: QSlider | None = None
        self._fs_lbl_time: QLabel | None = None
        self._fs_btn_playpause: QPushButton | None = None
        self._fs_btn_prev: QPushButton | None = None
        self._fs_btn_next: QPushButton | None = None
        self._fs_btn_exit: QPushButton | None = None
        self._fs_help_label: QLabel | None = None
        self._fs_seeking = False
        self._fs_hide_delay_ms = 2200
        self._fs_autohide_timer = QTimer(self)
        self._fs_autohide_timer.setSingleShot(True)
        self._fs_autohide_timer.setInterval(self._fs_hide_delay_ms)
        self._fs_autohide_timer.timeout.connect(self._fs_apply_hidden)
        self._fs_last_cursor_pos = None
        self._fs_mouse_poll_timer = QTimer(self)
        self._fs_mouse_poll_timer.setInterval(120)
        self._fs_mouse_poll_timer.timeout.connect(self._fs_poll_cursor)

        # seek tracking
        self._seeking = False

        # timers for UI update
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.poll_player)
        self.poll_timer.start()
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.setInterval(30000)
        self.scheduler_timer.timeout.connect(self._scheduler_tick)
        self.scheduler_timer.start()

        self._apply_netflix_theme()
        self._apply_app_language_ui()
        self.recent.setObjectName("resultsList")
        self.recommended.setObjectName("resultsList")
        self.refresh_history_ui()
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self._load_search_history()
        self._load_offline_covers_map()
        self._load_favorites()
        self._load_metadata_cache()
        self._sync_settings_ui_from_state()
        self._refresh_recent_queries_ui()
        self._update_suggestions_visibility()
        self._refresh_search_layout()
        self.refresh_offline_library()
        self.refresh_favorites_ui()
        QTimer.singleShot(1200, self.check_updates_silent)

    # ---------------- UI helpers ----------------
    def do_resolve_stream(
        self,
        anime: Any,
        ep: float | int,
        lang,
        preferred_quality,
        provider_name: str | None = None,
    ) -> StreamResult:
        provider_name = provider_name or self.provider_name
        t0 = time.perf_counter()
        debug_log(
            f"Resolving stream start: anime={anime.name}, ep={ep}, "
            f"lang={'SUB' if lang == LanguageTypeEnum.SUB else 'DUB'}, quality={preferred_quality}"
        )
        if self._is_aw_provider_name(provider_name):
            provider = self._aw_provider(provider_name)
            ep_key = str(int(ep)) if isinstance(ep, float) and ep.is_integer() else str(ep)
            if hasattr(anime, "has_episode") and anime.has_episode(ep_key):
                aw_ep = anime.episode(ep_key)
            else:
                aw_ep = anime.episode(str(ep))
            url = provider.episode_link(anime, aw_ep)
            debug_log(f"Resolving stream done in {time.perf_counter() - t0:.3f}s")
            return StreamResult(url=url, referrer=None, sub_file=None)

        stream = anime.get_video(
            episode=ep,
            lang=lang,
            preferred_quality=preferred_quality,
        )
        url = getattr(stream, "url", None) or str(stream)
        ref = getattr(stream, "referrer", None)
        debug_log(f"Resolving stream done in {time.perf_counter() - t0:.3f}s")
        return StreamResult(url=url, referrer=ref, sub_file=None)

    def notify_err(self, msg: str):
        title = self._tr("Errore", "Error")
        QMessageBox.critical(self, title, self._localize_runtime_text(str(msg)))

    def set_status(self, msg: str):
        self.status.setText(self._localize_runtime_text(str(msg)))

    def _tr(self, it: str, en: str) -> str:
        return en if self.app_language == "en" else it

    def _translate_literal(self, text: str) -> str:
        if not text:
            return text
        pairs = [
            ("Search", "Cerca"),
            ("Recommended", "Consigliati"),
            ("Downloads", "Download"),
            ("Favorites", "Preferiti"),
            ("Settings", "Impostazioni"),
            ("Download directory", "Cartella download"),
            ("Default provider", "Provider predefinito"),
            ("Default language", "Lingua predefinita"),
            ("App language", "Lingua app"),
            ("Default quality", "Qualita predefinita"),
            ("Parallel downloads", "Download paralleli"),
            ("Start HH:MM", "Inizio HH:MM"),
            ("End HH:MM", "Fine HH:MM"),
            ("Retry count", "Numero retry"),
            ("Browse", "Sfoglia"),
            ("Save settings", "Salva impostazioni"),
            ("Reset defaults", "Reset predefiniti"),
            ("Restore", "Ripristina"),
            ("Test AniList connection", "Test connessione AniList"),
            ("Import from AniList", "Importa da AniList"),
            ("Filter", "Filtro"),
            ("All", "Tutti"),
            ("Completed", "Completati"),
            ("Watching", "In corso"),
            ("Planned", "Da iniziare"),
            ("Open folder", "Apri cartella"),
            ("Open downloads", "Apri download"),
            ("Back to Search", "Torna alla ricerca"),
            ("Back to Episodes", "Torna agli episodi"),
            ("Back to Offline Catalog", "Torna al catalogo offline"),
            ("Queue selected", "Metti in coda selezionati"),
            ("Favorite", "Preferito"),
            ("Prev", "Precedente"),
            ("Next", "Successivo"),
            ("Fullscreen", "Schermo intero"),
            ("Mini", "Mini Player"),
            ("Recent Releases", "Uscite recenti"),
            ("Suggestions", "Suggerimenti"),
            ("Volume", "Volume"),
            ("Controls", "Controlli"),
            ("Watchlist", "Lista visione"),
            ("Watchlist / Favorites", "Lista visione / Preferiti"),
            ("Episodes / Files", "Episodi / File"),
            ("Offline Catalog", "Catalogo Offline"),
            ("Episodes", "Episodi"),
            ("Typing...", "Scrittura in corso..."),
            ("Typing…", "Scrittura in corso..."),
            ("Relevance", "Rilevanza"),
            ("Title A→Z", "Titolo A→Z"),
            ("Title Z→A", "Titolo Z→A"),
            ("Season: Any", "Stagione: Qualsiasi"),
            ("Year: Any", "Anno: Qualsiasi"),
            ("Rating: Any", "Rating: Qualsiasi"),
            ("No results.", "Nessun risultato."),
            ("Filtering...", "Filtraggio..."),
            ("Filtering…", "Filtraggio..."),
            ("Start queue", "Avvia coda"),
            ("Cancel selected", "Annulla selezionato"),
            ("Clear completed", "Pulisci completati"),
            ("Refresh", "Aggiorna"),
            ("Add current anime", "Aggiungi anime corrente"),
            ("Remove selected", "Rimuovi selezionato"),
            ("Add current episode", "Aggiungi episodio corrente"),
            ("Exit FS", "Esci FS"),
            ("Exit Mini", "Esci mini"),
            ("Play", "Play"),
            ("Pause", "Pausa"),
            ("Sync now", "Sincronizza ora"),
            ("Backup", "Backup"),
        ]
        if self.app_language == "en":
            for en, it in pairs:
                if text == it:
                    return en
        else:
            for en, it in pairs:
                if text == en:
                    return it
        return text

    def _localize_runtime_text(self, text: str) -> str:
        s = self._translate_literal(text)

        def _rx(pat: str, it_fmt: str, en_fmt: str):
            nonlocal s
            m = re.fullmatch(pat, s)
            if not m:
                return
            fmt = en_fmt if self.app_language == "en" else it_fmt
            s = fmt.format(*m.groups())

        _rx(r"Episodi: (\d+)", "Episodi: {}", "Episodes: {}")
        _rx(r"Searching…", "Ricerca in corso…", "Searching...")
        _rx(r"Carico recent \+ recommended…", "Carico recenti + consigliati…", "Loading recent + recommended...")
        _rx(r"Trovati (\d+) risultati\.", "Trovati {} risultati.", "Found {} results.")
        _rx(r"Selezionato: (.+) — carico episodi…", "Selezionato: {} — carico episodi…", "Selected: {} — loading episodes...")
        _rx(r"Selezionato in cronologia: (.+)", "Selezionato in cronologia: {}", "Selected in history: {}")
        _rx(r"Carico episodi per resume: (.+)…", "Carico episodi per ripresa: {}…", "Loading episodes for resume: {}...")
        _rx(r"Apro dettagli anime: (.+)…", "Apro dettagli anime: {}…", "Opening anime details: {}...")
        _rx(r"Risolvo stream ep (.+)…", "Risolvo stream ep {}…", "Resolving stream ep {}...")
        _rx(r"Risolvo stream per download: (.+) ep (.+)", "Risolvo stream per download: {} ep {}", "Resolving stream for download: {} ep {}")
        _rx(r"Aggiunto download: (.+) ep (.+)", "Aggiunto download: {} ep {}", "Added download: {} ep {}")
        _rx(r"Aggiunti (\d+) episodi alla coda download\.", "Aggiunti {} episodi alla coda download.", "Added {} episodes to download queue.")
        _rx(r"Gia presente: (.+)", "Gia presente: {}", "Already present: {}")
        _rx(r"Integrity check fallita, retry: (.+) ep (.+)", "Integrity check fallita, retry: {} ep {}", "Integrity check failed, retry: {} ep {}")
        _rx(r"▶ (.+) — ep (.+)", "▶ {} — ep {}", "▶ {} — ep {}")
        _rx(r"▶ Offline: (.+)", "▶ Offline: {}", "▶ Offline: {}")
        _rx(r"Segnato come visto: (.+) ep (.+)", "Segnato come visto: {} ep {}", "Marked as seen: {} ep {}")
        _rx(r"Rimosso dalla cronologia\.", "Rimosso dalla cronologia.", "Removed from history.")
        _rx(r"Backup creato: (.+)", "Backup creato: {}", "Backup created: {}")
        _rx(r"Pull AniList -> Anigui in corso…", "Pull AniList -> Anigui in corso…", "AniList pull -> Anigui in progress...")
        _rx(r"Pull AniList fallito\.", "Pull AniList fallito.", "AniList pull failed.")
        _rx(r"Restore completato\.", "Ripristino completato.", "Restore completed.")
        _rx(r"Catalogo offline\.", "Catalogo offline.", "Offline catalog.")
        _rx(r"Ricerca anime\.", "Ricerca anime.", "Anime search.")
        _rx(r"Lista episodi\.", "Lista episodi.", "Episode list.")
        _rx(r"Aggiunto ai preferiti\.", "Aggiunto ai preferiti.", "Added to favorites.")
        _rx(r"Download fallito\.", "Download fallito.", "Download failed.")
        _rx(r"Download fallito \(integrity\)\.", "Download fallito (integrity).", "Download failed (integrity).")
        _rx(r"Download annullato\.", "Download annullato.", "Download cancelled.")
        _rx(r"Download completato: (.+)", "Download completato: {}", "Download completed: {}")
        _rx(r"Errore resolve download\.", "Errore risoluzione download.", "Download resolve error.")
        _rx(r"Scheduler attivo: fuori finestra oraria, coda in pausa\.", "Scheduler attivo: fuori finestra oraria, coda in pausa.", "Scheduler active: outside time window, queue paused.")
        _rx(r"Incognito attivo: sync esterno disabilitato\.", "Incognito attivo: sync esterno disabilitato.", "Incognito active: external sync disabled.")
        _rx(r"Incognito attivo: salvataggio settings bloccato\.", "Incognito attivo: salvataggio impostazioni bloccato.", "Incognito active: settings save blocked.")
        _rx(r"Incognito attivo: restore bloccato\.", "Incognito attivo: ripristino bloccato.", "Incognito active: restore blocked.")
        _rx(r"Incognito attivo: modifica cronologia bloccata\.", "Incognito attivo: modifica cronologia bloccata.", "Incognito active: history changes blocked.")
        _rx(r"Modalita Incognito attiva: cache/salvataggi disabilitati\.", "Modalita Incognito attiva: cache/salvataggi disabilitati.", "Incognito mode enabled: cache/saves disabled.")
        _rx(r"Modalita Incognito disattivata\.", "Modalita Incognito disattivata.", "Incognito mode disabled.")
        _rx(r"Sincronizza ora: nessun anime/episodio attivo da sincronizzare\.", "Sincronizza ora: nessun anime/episodio attivo da sincronizzare.", "Sync now: no active anime/episode to sync.")
        _rx(r"Sync now: nessun anime/episodio attivo da sincronizzare\.", "Sincronizza ora: nessun anime/episodio attivo da sincronizzare.", "Sync now: no active anime/episode to sync.")
        _rx(r"Sincronizza ora: episodio non completato, sync non inviato\.", "Sincronizza ora: episodio non completato, sync non inviato.", "Sync now: episode not completed, sync not sent.")
        _rx(r"Sync now: episodio non completato, sync non inviato\.", "Sincronizza ora: episodio non completato, sync non inviato.", "Sync now: episode not completed, sync not sent.")
        _rx(r"Sincronizzazione AniList: (.+) ep (.+)…", "Sincronizzazione AniList: {} ep {}…", "AniList sync: {} ep {}...")
        _rx(r"Sync AniList: (.+) ep (.+)…", "Sincronizzazione AniList: {} ep {}…", "AniList sync: {} ep {}...")
        _rx(r"Sincronizzazione AniList ok: (.+) ep (.+)", "Sincronizzazione AniList ok: {} ep {}", "AniList sync ok: {} ep {}")
        _rx(r"AniList sync ok: (.+) ep (.+)", "Sincronizzazione AniList ok: {} ep {}", "AniList sync ok: {} ep {}")
        _rx(r"Sincronizzazione AniList fallita\.", "Sincronizzazione AniList fallita.", "AniList sync failed.")
        _rx(r"AniList sync fallito\.", "Sincronizzazione AniList fallita.", "AniList sync failed.")
        _rx(r"Test AniList in corso…", "Test AniList in corso…", "Testing AniList...")
        _rx(r"AniList connesso: (.+)", "AniList connesso: {}", "AniList connected: {}")
        _rx(r"AniList token mancante\.", "Token AniList mancante.", "AniList token missing.")
        _rx(r"AniList non raggiungibile/token non valido\.", "AniList non raggiungibile/token non valido.", "AniList unavailable/invalid token.")
        _rx(r"Impostazioni salvate\.", "Impostazioni salvate.", "Settings saved.")
        _rx(r"Impostazioni ripristinate ai default \(premi Salva\)\.", "Impostazioni ripristinate ai default (premi Salva).", "Settings reset to defaults (press Save).")
        _rx(r"Nessun nuovo episodio aggiunto \(gia in coda/attivi\)\.", "Nessun nuovo episodio aggiunto (gia in coda/attivi).", "No new episodes added (already queued/active).")
        _rx(r"Inserisci prima il token AniList\.", "Inserisci prima il token AniList.", "Enter AniList token first.")
        _rx(r"AniList test fallito: (.+)", "AniList test fallito: {}", "AniList test failed: {}")
        _rx(r"AniList sync fallito: (.+)", "AniList sync fallito: {}", "AniList sync failed: {}")
        _rx(r"Pull AniList fallito: (.+)", "Pull AniList fallito: {}", "AniList pull failed: {}")
        _rx(r"Seleziona o avvia un episodio prima di aggiungerlo alla coda download\.", "Seleziona o avvia un episodio prima di aggiungerlo alla coda download.", "Select or start an episode before adding it to download queue.")
        _rx(r"Apri prima un anime\.", "Apri prima un anime.", "Open an anime first.")
        _rx(r"Seleziona uno o piu episodi dalla lista\.", "Seleziona uno o piu episodi dalla lista.", "Select one or more episodes from the list.")
        _rx(r"Su Linux al momento e supportato solo Nautilus\.\nInstalla Nautilus oppure apri manualmente il percorso:\n(.+)", "Su Linux al momento e supportato solo Nautilus.\nInstalla Nautilus oppure apri manualmente il percorso:\n{}", "On Linux only Nautilus is currently supported.\nInstall Nautilus or open this path manually:\n{}")
        _rx(r"Impossibile avviare Nautilus automaticamente\.\nPercorso: (.+)", "Impossibile avviare Nautilus automaticamente.\nPercorso: {}", "Could not launch Nautilus automatically.\nPath: {}")
        _rx(r"File locale non trovato\.", "File locale non trovato.", "Local file not found.")
        _rx(r"Download directory non valido\.", "Download directory non valido.", "Invalid download directory.")
        _rx(r"Impossibile creare la cartella download: (.+)", "Impossibile creare la cartella download: {}", "Unable to create download directory: {}")
        _rx(r"Formato scheduler non valido\. Usa HH:MM \(es\. 23:30\)\.", "Formato scheduler non valido. Usa HH:MM (es. 23:30).", "Invalid scheduler format. Use HH:MM (e.g. 23:30).")
        _rx(r"Inserisci un AniList token oppure disattiva AniList sync\.", "Inserisci un AniList token oppure disattiva AniList sync.", "Enter an AniList token or disable AniList sync.")
        _rx(r"Errore salvataggio settings: (.+)", "Errore salvataggio settings: {}", "Settings save error: {}")
        _rx(r"Backup fallito: (.+)", "Backup fallito: {}", "Backup failed: {}")
        _rx(r"Restore fallito: (.+)", "Restore fallito: {}", "Restore failed: {}")
        _rx(r"Pronto\.", "Pronto.", "Ready.")
        _rx(r"Errore\.", "Errore.", "Error.")
        return s

    def _translate_widget_tree(self):
        for lbl in self.findChildren(QLabel):
            txt = lbl.text()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                lbl.setText(new_txt)
        for btn in self.findChildren(QPushButton):
            txt = btn.text()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                btn.setText(new_txt)
        for box in self.findChildren(QGroupBox):
            txt = box.title()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                box.setTitle(new_txt)
        for edit in self.findChildren(QLineEdit):
            txt = edit.placeholderText()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                edit.setPlaceholderText(new_txt)

    @staticmethod
    def _combo_set_items(
        combo: QComboBox,
        items: list[tuple[str, Any]],
        keep_data: Any = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        if keep_data is not None:
            idx = combo.findData(keep_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _apply_app_language_ui(self):
        self.setWindowTitle("Anigui")

        # Tabs
        self.tabs.setTabText(self.tabs.indexOf(self.tab_search), self._tr("Cerca", "Search"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_recommended), self._tr("Consigliati", "Recommended"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_downloads), self._tr("Download", "Downloads"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_offline), "Offline")
        self.tabs.setTabText(self.tabs.indexOf(self.tab_favorites), self._tr("Preferiti", "Favorites"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_settings), self._tr("Impostazioni", "Settings"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_history), self._tr("Lista visione", "Watchlist"))

        # Top/search/player
        self.search_input.setPlaceholderText(self._tr("Cerca anime…", "Search anime..."))
        self.btn_search.setText(self._tr("🔎 Cerca", "🔎 Search"))
        self.btn_search.setToolTip(self._tr("Avvia ricerca anime", "Start anime search"))
        self.btn_incognito.setToolTip(self._tr("Disabilita salvataggi locali e cache", "Disable local saves and cache"))
        self.btn_incognito.setText(self._tr("Incognito ON", "Incognito ON") if self._incognito_enabled else self._tr("Incognito OFF", "Incognito OFF"))
        self.btn_clear_search_history.setText(self._tr("🧹 Pulisci cronologia ricerche", "🧹 Clear search history"))
        self.btn_back_catalog.setText(self._tr("← Torna alla ricerca", "← Back to Search"))
        self.btn_queue_selected_eps.setText(self._tr("⬇ Metti in coda selezionati", "⬇ Queue selected"))
        self.btn_fav_anime.setText(self._tr("❤ Preferito", "❤ Favorite"))
        self.btn_back_episodes.setText(self._tr("← Torna agli episodi", "← Back to Episodes"))
        self.btn_prev.setText(self._tr("⏮ Precedente", "⏮ Prev"))
        self.btn_playpause.setText(self._tr("⏯ Play/Pause", "⏯ Play/Pause"))
        self.btn_next.setText(self._tr("⏭ Successivo", "⏭ Next"))
        self.btn_fs.setText(self._tr("⛶ Schermo intero", "⛶ Fullscreen"))
        self.btn_mini.setText(self._tr("▣ Mini Player", "▣ Mini"))
        self.lbl_player_title.setText(self._tr("Player", "Player"))

        # Recommended
        self.lbl_suggestions.setText(self._tr("Suggerimenti", "Suggestions"))
        self.lbl_recommended.setText(self._tr("Uscite recenti + Consigliati", "Recent Releases + Recommended"))
        self.lbl_recent_header.setText(self._tr("Uscite recenti", "Recent Releases"))
        self.lbl_recommended_header.setText(self._tr("Consigliati", "Recommended"))

        # Downloads / offline / favorites
        self.btn_dl_open.setText(self._tr("📂 Apri cartella", "📂 Open folder"))
        self.lbl_offline.setText(self._tr("Anime scaricati (stream locale)", "Downloaded anime (local stream)"))
        self.btn_offline_open.setText(self._tr("📂 Apri download", "📂 Open downloads"))
        self.btn_offline_back.setText(self._tr("← Torna al catalogo offline", "← Back to Offline Catalog"))
        self.btn_fav_add_current.setText(self._tr("❤ Aggiungi anime corrente", "❤ Add current anime"))
        self.btn_fav_remove.setText(self._tr("🗑 Rimuovi selezionato", "🗑 Remove selected"))
        self.btn_dl_add_current.setText(self._tr("＋ Aggiungi episodio corrente", "＋ Add current episode"))
        self.btn_dl_start.setText(self._tr("▶ Avvia coda", "▶ Start queue"))
        self.btn_dl_start.setToolTip(self._tr("Avvia i download in coda", "Start queued downloads"))
        self.btn_dl_cancel.setText(self._tr("✖ Annulla selezionato", "✖ Cancel selected"))
        self.btn_dl_clear.setText(self._tr("🧹 Pulisci completati", "🧹 Clear completed"))
        self.btn_refresh_recommended.setText(self._tr("↻ Aggiorna", "↻ Refresh"))
        self.btn_offline_refresh.setText(self._tr("↻ Aggiorna", "↻ Refresh"))
        self.lbl_anime_title.setText(self._tr("Anime", "Anime"))
        self.lbl_offline_anime_title.setText(self._tr("Anime", "Anime"))

        # Settings labels/buttons
        self.lbl_settings_download_dir.setText(self._tr("Cartella download", "Download directory"))
        self.lbl_settings_default_provider.setText(self._tr("Provider predefinito", "Default provider"))
        self.lbl_settings_default_lang.setText(self._tr("Lingua predefinita", "Default language"))
        self.lbl_settings_app_language.setText(self._tr("Lingua app", "App language"))
        self.lbl_settings_default_quality.setText(self._tr("Qualita predefinita", "Default quality"))
        self.lbl_settings_parallel_downloads.setText(self._tr("Download paralleli", "Parallel downloads"))
        self.lbl_settings_scheduler.setText("Scheduler")
        self.lbl_settings_scheduler_start.setText(self._tr("Inizio HH:MM", "Start HH:MM"))
        self.lbl_settings_scheduler_end.setText(self._tr("Fine HH:MM", "End HH:MM"))
        self.lbl_settings_integrity_min_mb.setText("Integrity min MB")
        self.lbl_settings_retry_count.setText(self._tr("Numero retry", "Retry count"))
        self.lbl_settings_anilist_sync.setText("AniList sync")
        self.lbl_settings_anilist_token.setText("AniList token")
        self.btn_settings_browse_download_dir.setText(self._tr("Sfoglia", "Browse"))
        self.btn_settings_save.setText(self._tr("Salva impostazioni", "Save settings"))
        self.btn_settings_reset.setText(self._tr("Reset predefiniti", "Reset defaults"))
        self.btn_settings_backup.setText(self._tr("Backup", "Backup"))
        self.btn_settings_restore.setText(self._tr("Ripristina", "Restore"))
        self.btn_settings_anilist_test.setText(self._tr("Test connessione AniList", "Test AniList connection"))
        self.btn_settings_anilist_sync_now.setText(self._tr("Sincronizza ora", "Sync now"))
        self.btn_settings_anilist_pull.setText(self._tr("Importa da AniList", "Import from AniList"))
        self.btn_settings_update_check.setText(self._tr("Controlla aggiornamenti", "Check updates"))
        if platform.system() == "Windows" and getattr(sys, "frozen", False):
            self.btn_settings_update_apply.setText(self._tr("Aggiorna ora", "Update now"))
        else:
            self.btn_settings_update_apply.setText(self._tr("Apri pagina update", "Open update page"))
        self.btn_settings_anilist_sync_now.setToolTip(
            self._tr("Invia subito il progresso corrente ad AniList", "Send current progress to AniList now")
        )
        self.btn_settings_anilist_pull.setToolTip(
            self._tr("Importa watchlist/progressi da AniList", "Import watchlist/progress from AniList")
        )

        # Search filters
        sort_idx = self.combo_sort.currentIndex()
        self.combo_sort.blockSignals(True)
        self.combo_sort.clear()
        self.combo_sort.addItems(
            [
                self._tr("Rilevanza", "Relevance"),
                self._tr("Titolo A→Z", "Title A→Z"),
                self._tr("Titolo Z→A", "Title Z→A"),
            ]
        )
        self.combo_sort.setCurrentIndex(max(0, sort_idx))
        self.combo_sort.blockSignals(False)

        season_idx = self.combo_season.currentIndex()
        self.combo_season.blockSignals(True)
        self.combo_season.clear()
        self.combo_season.addItems(
            [
                self._tr("Stagione: Qualsiasi", "Season: Any"),
                "Winter",
                "Spring",
                "Summer",
                "Fall",
            ]
        )
        self.combo_season.setCurrentIndex(max(0, season_idx))
        self.combo_season.blockSignals(False)

        if self.combo_year.count() > 0:
            self.combo_year.setItemText(0, self._tr("Anno: Qualsiasi", "Year: Any"))

        rating_idx = self.combo_rating.currentIndex()
        self.combo_rating.blockSignals(True)
        self.combo_rating.clear()
        self.combo_rating.addItems(
            [
                self._tr("Rating: Qualsiasi", "Rating: Any"),
                ">= 9",
                ">= 8",
                ">= 7",
                ">= 6",
            ]
        )
        self.combo_rating.setCurrentIndex(max(0, rating_idx))
        self.combo_rating.blockSignals(False)

        # Parallel download combos
        dl_parallel_data = self.combo_dl_parallel.currentData()
        self._combo_set_items(
            self.combo_dl_parallel,
            [
                (self._tr("Paralleli x1", "Parallel x1"), 1),
                (self._tr("Paralleli x2", "Parallel x2"), 2),
                (self._tr("Paralleli x3", "Parallel x3"), 3),
                (self._tr("Paralleli x4", "Parallel x4"), 4),
            ],
            keep_data=dl_parallel_data,
        )

        # Settings combos with translated labels
        self._combo_set_items(
            self.combo_settings_scheduler_enabled,
            [
                (self._tr("Disabilitato", "Disabled"), False),
                (self._tr("Abilitato", "Enabled"), True),
            ],
            keep_data=self.combo_settings_scheduler_enabled.currentData(),
        )
        self._combo_set_items(
            self.combo_settings_anilist_enabled,
            [
                (self._tr("Disabilitato", "Disabled"), False),
                (self._tr("Abilitato", "Enabled"), True),
            ],
            keep_data=self.combo_settings_anilist_enabled.currentData(),
        )

        # History/watchlist
        self.lbl_history_header.setText(
            self._tr(
                "Lista visione: Da iniziare · In corso · Completati",
                "Watchlist: Planned · Watching · Completed",
            )
        )
        self.lbl_history_filter.setText(self._tr("Filtro", "Filter"))
        self._combo_set_items(
            self.combo_history_filter,
            [
                (self._tr("Tutti", "All"), "All"),
                (self._tr("Completati", "Completed"), "Completed"),
                (self._tr("In corso", "Watching"), "Watching"),
                (self._tr("Da iniziare", "Planned"), "Planned"),
            ],
            keep_data=self._history_filter,
        )
        self.btn_hist_resume.setText(self._tr("▶ Riprendi selezionato", "▶ Resume selected"))
        self.btn_hist_resume_next.setText(self._tr("⏭ Riprendi prossimo", "⏭ Resume next"))
        self.btn_hist_mark_seen.setText(self._tr("✅ Segna come visto", "✅ Mark as seen"))
        self.btn_hist_delete.setText(self._tr("🗑 Rimuovi", "🗑 Remove"))
        self.btn_clear_watch_history.setText(self._tr("🧹 Pulisci cronologia visione", "🧹 Clear watch history"))

        # Anonymous labels/buttons created inline.
        self._translate_widget_tree()
        # Refresh lists that contain language-dependent placeholder/status labels.
        self._refresh_downloads_ui()
        self._refresh_recent_queries_ui()

    def on_toggle_incognito(self, enabled: bool):
        self._incognito_enabled = bool(enabled)
        if self._incognito_enabled:
            self.btn_incognito.setText(self._tr("Incognito ON", "Incognito ON"))
            self.lbl_incognito_badge.setVisible(True)
            self.set_status("Modalita Incognito attiva: cache/salvataggi disabilitati.")
        else:
            self.btn_incognito.setText(self._tr("Incognito OFF", "Incognito OFF"))
            self.lbl_incognito_badge.setVisible(False)
            self.set_status("Modalita Incognito disattivata.")

    def _anilist_headers(self) -> dict[str, str]:
        return self._anilist_headers_for_token(self._anilist_token)

    def _anilist_headers_for_token(self, token: str | None) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Anigui/1.0 (+https://anilist.co)",
        }
        t = self._normalize_anilist_token(token)
        if t:
            h["Authorization"] = f"Bearer {t}"
        return h

    @staticmethod
    def _normalize_anilist_token(token: str | None) -> str:
        t = (token or "").strip()
        if not t:
            return ""
        if t.lower().startswith("bearer "):
            t = t[7:].strip()
        # If user pasted full redirect URL or fragment, extract access_token.
        if "access_token=" in t:
            try:
                frag = urlsplit(t).fragment or ""
                if frag:
                    q = parse_qs(frag)
                    val = q.get("access_token", [])
                    if val and val[0]:
                        return str(val[0]).strip()
                # fallback: parse from raw string
                q = parse_qs(t.replace("#", "&").replace("?", "&"))
                val = q.get("access_token", [])
                if val and val[0]:
                    return str(val[0]).strip()
            except Exception:
                pass
        return t

    def _anilist_graphql(self, query: str, variables: dict[str, Any], token: str | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            "https://graphql.anilist.co",
            data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
            headers=self._anilist_headers_for_token(token if token is not None else self._anilist_token),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=12.0) as r:
                raw = r.read()
        except urllib.error.HTTPError as ex:
            body = ""
            try:
                body = ex.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                body = ""
            msg = f"HTTP {ex.code} {ex.reason}"
            if body:
                msg = f"{msg} - {body[:400]}"
            raise RuntimeError(msg) from ex
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
        if isinstance(payload, dict) and payload.get("errors"):
            raise RuntimeError(str(payload["errors"]))
        if not isinstance(payload, dict):
            raise RuntimeError("AniList response non valida.")
        return payload

    @staticmethod
    def _anilist_status_for_entry(entry: HistoryEntry) -> str:
        # Episode-complete sync should not mark the whole series as completed.
        return "CURRENT"

    def _anilist_sync_key(self, entry: HistoryEntry) -> str:
        return f"{entry.provider}:{entry.identifier}:{entry.lang}"

    def _anilist_sync_progress_async(self, entry: HistoryEntry, force: bool = False):
        if self._incognito_enabled:
            return
        if not self._anilist_enabled:
            return
        if not (self._anilist_token or "").strip():
            return
        if not bool(entry.completed):
            return
        progress = int(max(1, float(entry.last_ep)))
        key = self._anilist_sync_key(entry)
        prev = int(self._anilist_last_synced_progress.get(key, 0))
        now = time.time()
        if not force:
            if progress <= prev:
                return
            if now - float(self._anilist_last_sync_ts.get(key, 0.0)) < 20.0:
                return

        w = Worker(
            self._anilist_sync_worker,
            entry.name,
            progress,
            self._anilist_status_for_entry(entry),
            self._anilist_token,
        )

        def on_ok(_res: object, k=key, p=progress):
            self._anilist_last_synced_progress[k] = max(
                p, int(self._anilist_last_synced_progress.get(k, 0))
            )
            self._anilist_last_sync_ts[k] = time.time()

        def on_err(msg: str):
            debug_log(f"AniList sync failed: {msg}")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()

    def _anilist_sync_worker(
        self,
        anime_name: str,
        progress: int,
        status: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        cache_key = anime_name.strip().lower()
        media_id = self._anilist_media_id_cache.get(cache_key)
        if media_id is None:
            q = """
            query ($search: String) {
              Media(search: $search, type: ANIME) { id }
            }
            """
            payload = self._anilist_graphql(q, {"search": anime_name}, token=token)
            media = (
                payload.get("data", {}).get("Media")
                if isinstance(payload.get("data"), dict)
                else None
            )
            if not isinstance(media, dict) or not media.get("id"):
                raise RuntimeError(f"AniList match non trovato: {anime_name}")
            media_id = int(media["id"])
            self._anilist_media_id_cache[cache_key] = media_id

        m = """
        mutation ($mediaId: Int, $status: MediaListStatus, $progress: Int) {
          SaveMediaListEntry(mediaId: $mediaId, status: $status, progress: $progress) {
            id
            status
            progress
          }
        }
        """
        payload = self._anilist_graphql(
            m,
            {
                "mediaId": int(media_id),
                "status": status,
                "progress": int(max(1, progress)),
            },
            token=token,
        )
        return payload.get("data", {})

    def _anilist_active_token_from_ui(self) -> str:
        tok = ""
        if hasattr(self, "input_settings_anilist_token"):
            tok = self.input_settings_anilist_token.text().strip()
        if not tok:
            tok = (self._anilist_token or "").strip()
        return self._normalize_anilist_token(tok)

    @staticmethod
    def _norm_title(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _planned_identifier_from_titles(
        titles: list[str],
        media_id: int | None = None,
        entry_id: int | None = None,
    ) -> str:
        if media_id is not None and int(media_id) > 0:
            return f"planned:anilist:{int(media_id)}"
        if entry_id is not None and int(entry_id) > 0:
            return f"planned:anilist-entry:{int(entry_id)}"
        base = " | ".join([t.strip() for t in titles if str(t).strip()]) or "planned"
        h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"planned:{h}"

    def _search_items_for_provider(
        self,
        query: str,
        provider_name: str,
        lang: Any,
        strict_lang: bool = True,
    ) -> list[SearchItem]:
        if not query.strip():
            return []
        if self._is_aw_provider_name(provider_name):
            provider = self._aw_provider(provider_name)
            res = provider.search(query) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                if strict_lang:
                    if lang == LanguageTypeEnum.DUB and not is_dub:
                        continue
                    if lang == LanguageTypeEnum.SUB and is_dub:
                        continue
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=provider_name,
                        raw=r,
                    )
                )
            return out
        provider = get_provider(provider_name)
        res = provider.get_search(query)
        return [
            SearchItem(
                r.name,
                r.identifier,
                r.languages,
                cover_url=self._extract_cover_url(r),
                source=provider_name,
                raw=r,
            )
            for r in res
        ]

    def _pick_best_item_for_titles(
        self,
        titles: list[str],
        items: list[SearchItem],
    ) -> SearchItem | None:
        if not items:
            return None
        norm_titles = [self._norm_title(t) for t in titles if t and self._norm_title(t)]
        if not norm_titles:
            return items[0]
        for t in norm_titles:
            for it in items:
                if self._norm_title(it.name) == t:
                    return it
        for t in norm_titles:
            for it in items:
                n = self._norm_title(it.name)
                if t in n or n in t:
                    return it
        return items[0]

    def _anilist_fetch_progress_entries(self, token: str) -> list[dict[str, Any]]:
        q = """
        query {
          Viewer {
            id
            name
            mediaListOptions { scoreFormat }
          }
          MediaListCollection(
            userId: 0,
            type: ANIME,
            status_in: [CURRENT, REPEATING, COMPLETED, PLANNING]
          ) {
            lists {
              status
              entries {
                id
                progress
                media {
                  id
                  title { romaji english native }
                }
              }
            }
          }
        }
        """
        # AniList does not accept userId:0, resolve viewer first.
        v = self._anilist_graphql("query { Viewer { id name } }", {}, token=token)
        viewer = v.get("data", {}).get("Viewer") if isinstance(v.get("data"), dict) else None
        if not isinstance(viewer, dict) or not viewer.get("id"):
            raise RuntimeError("Impossibile leggere Viewer AniList.")
        user_id = int(viewer["id"])
        q2 = """
        query ($userId: Int) {
          MediaListCollection(
            userId: $userId,
            type: ANIME,
            status_in: [CURRENT, REPEATING, COMPLETED, PLANNING]
          ) {
            lists {
              status
              entries {
                id
                progress
                media {
                  id
                  title { romaji english native }
                }
              }
            }
          }
        }
        """
        payload = self._anilist_graphql(q2, {"userId": user_id}, token=token)
        coll = payload.get("data", {}).get("MediaListCollection") if isinstance(payload.get("data"), dict) else None
        lists = coll.get("lists", []) if isinstance(coll, dict) else []
        out: list[dict[str, Any]] = []
        for lst in lists:
            if not isinstance(lst, dict):
                continue
            status = str(lst.get("status", "") or "")
            for e in lst.get("entries", []) or []:
                if not isinstance(e, dict):
                    continue
                entry_id = int(e.get("id") or 0) if isinstance(e.get("id"), (int, float, str)) else 0
                prog = int(e.get("progress") or 0)
                media = e.get("media") if isinstance(e.get("media"), dict) else {}
                media_id = int(media.get("id") or 0) if isinstance(media.get("id"), (int, float, str)) else 0
                title = media.get("title") if isinstance(media.get("title"), dict) else {}
                titles = [
                    str(title.get("english") or "").strip(),
                    str(title.get("romaji") or "").strip(),
                    str(title.get("native") or "").strip(),
                ]
                titles = [t for t in titles if t]
                if not titles and media_id > 0:
                    titles = [f"AniList #{media_id}"]
                if not titles and entry_id > 0:
                    titles = [f"AniList entry #{entry_id}"]
                if not titles:
                    titles = ["Planned anime"]
                out.append(
                    {
                        "status": status,
                        "progress": prog,
                        "titles": titles,
                        "media_id": media_id,
                        "entry_id": entry_id,
                    }
                )
        return out

    def _anilist_pull_worker(
        self,
        token: str,
        provider_name: str,
        lang_name: str,
    ) -> dict[str, Any]:
        lang = LanguageTypeEnum.DUB if lang_name == "DUB" else LanguageTypeEnum.SUB
        rows = self._anilist_fetch_progress_entries(token)
        seen_ident: set[str] = set()
        built: list[HistoryEntry] = []
        skipped = 0
        remote_planned = 0
        imported_planned = 0
        planned_all_keys: list[str] = []
        planned_all_seen: set[str] = set()
        planned_imported_keys: set[str] = set()
        for row in rows:
            titles = list(row.get("titles", []))
            progress = int(row.get("progress") or 0)
            media_id = int(row.get("media_id") or 0)
            entry_id = int(row.get("entry_id") or 0)
            remote_status = str(row.get("status") or "").upper()
            planned_key = ""
            if remote_status == "PLANNING":
                if media_id > 0:
                    planned_key = f"m:{media_id}"
                elif entry_id > 0:
                    planned_key = f"e:{entry_id}"
                else:
                    planned_key = f"t:{self._planned_identifier_from_titles(titles)}"
            if remote_status == "PLANNING":
                remote_planned += 1
                if planned_key and planned_key not in planned_all_seen:
                    planned_all_seen.add(planned_key)
                    planned_all_keys.append(planned_key)
            found: SearchItem | None = None
            for t in titles:
                items = self._search_items_for_provider(t, provider_name, lang)
                found = self._pick_best_item_for_titles(titles, items)
                if found is not None:
                    break
            if found is None and remote_status == "PLANNING":
                for t in titles:
                    items = self._search_items_for_provider(
                        t,
                        provider_name,
                        lang,
                        strict_lang=False,
                    )
                    found = self._pick_best_item_for_titles(titles, items)
                    if found is not None:
                        break
            if found is None:
                if remote_status == "PLANNING":
                    placeholder_id = self._planned_identifier_from_titles(
                        titles,
                        media_id=media_id,
                        entry_id=entry_id,
                    )
                    uniq = f"{provider_name}:{placeholder_id}:{lang_name}"
                    if uniq in seen_ident:
                        if planned_key:
                            planned_imported_keys.add(planned_key)
                        continue
                    seen_ident.add(uniq)
                    fallback_name = next((t for t in titles if t.strip()), f"AniList #{media_id}" if media_id > 0 else "Planned anime")
                    built.append(
                        HistoryEntry(
                            provider=provider_name,
                            identifier=placeholder_id,
                            name=fallback_name,
                            lang=lang_name,
                            last_ep=0.0,
                            updated_at=time.time(),
                            cover_url=None,
                            last_pos=0.0,
                            last_duration=0.0,
                            last_percent=0.0,
                            completed=False,
                            watched_eps=[],
                            episode_progress={},
                        )
                    )
                    imported_planned += 1
                    if planned_key:
                        planned_imported_keys.add(planned_key)
                    continue
                skipped += 1
                continue
            uniq = f"{provider_name}:{found.identifier}:{lang_name}"
            if uniq in seen_ident:
                if remote_status == "PLANNING":
                    # Provider dedupe collision: still keep this AniList planning entry via placeholder.
                    placeholder_id = self._planned_identifier_from_titles(
                        titles,
                        media_id=media_id,
                        entry_id=entry_id,
                    )
                    puniq = f"{provider_name}:{placeholder_id}:{lang_name}"
                    if puniq not in seen_ident:
                        seen_ident.add(puniq)
                        fallback_name = next((t for t in titles if t.strip()), f"AniList #{media_id}" if media_id > 0 else "Planned anime")
                        built.append(
                            HistoryEntry(
                                provider=provider_name,
                                identifier=placeholder_id,
                                name=fallback_name,
                                lang=lang_name,
                                last_ep=0.0,
                                updated_at=time.time(),
                                cover_url=None,
                                last_pos=0.0,
                                last_duration=0.0,
                                last_percent=0.0,
                                completed=False,
                                watched_eps=[],
                                episode_progress={},
                            )
                        )
                    imported_planned += 1
                    if planned_key:
                        planned_imported_keys.add(planned_key)
                continue
            seen_ident.add(uniq)

            progress = max(0, int(progress))
            prog = float(progress)
            watched = [float(i) for i in range(1, int(prog) + 1)] if progress <= 400 else [prog]
            if progress <= 0:
                watched = []
            ep_prog: dict[str, dict[str, Any]] = {}
            for epf in watched[:400]:
                ep_key = str(int(epf)) if float(epf).is_integer() else str(epf)
                ep_prog[ep_key] = {
                    "pos": 0.0,
                    "dur": 0.0,
                    "percent": 100.0,
                    "completed": True,
                    "updated_at": time.time(),
                }
            built.append(
                HistoryEntry(
                    provider=provider_name,
                    identifier=found.identifier,
                    name=found.name,
                    lang=lang_name,
                    last_ep=prog,
                    updated_at=time.time(),
                    cover_url=found.cover_url,
                    last_pos=0.0,
                    last_duration=0.0,
                    last_percent=100.0 if progress > 0 else 0.0,
                    completed=(remote_status == "COMPLETED"),
                    watched_eps=watched,
                    episode_progress=ep_prog,
                )
            )
            if remote_status == "PLANNING":
                imported_planned += 1
                if planned_key:
                    planned_imported_keys.add(planned_key)
        planned_missing_keys = [k for k in planned_all_keys if k not in planned_imported_keys]
        return {
            "total_remote": len(rows),
            "built": built,
            "skipped": skipped,
            "remote_planned": remote_planned,
            "imported_planned": imported_planned,
            "planned_missing_keys": planned_missing_keys,
        }

    def _anilist_test_connection_worker(self, token: str) -> str:
        q = """
        query {
          Viewer { id name }
        }
        """
        payload = self._anilist_graphql(q, {}, token=token)
        viewer = payload.get("data", {}).get("Viewer") if isinstance(payload.get("data"), dict) else None
        if not isinstance(viewer, dict) or not viewer.get("name"):
            raise RuntimeError("AniList viewer non disponibile.")
        return str(viewer.get("name"))

    def on_anilist_test_connection(self):
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            self.set_status("AniList token mancante.")
            return
        self.set_status("Test AniList in corso…")
        self._begin_request()
        w = Worker(self._anilist_test_connection_worker, token)

        def on_ok(name: str):
            self._end_request()
            self.set_status(f"AniList connesso: {name}")

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"AniList test fallito: {msg}")
            self.set_status("AniList non raggiungibile/token non valido.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()

    def on_anilist_sync_now(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: sync esterno disabilitato.")
            return
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            return

        anime_name: str | None = None
        progress = 0
        completed = False
        source = "none"

        if self.selected_anime is not None and self.current_ep is not None:
            anime_name = str(getattr(self.selected_anime, "name", "")).strip() or None
            progress = int(max(1, float(self.current_ep)))
            try:
                pos = float(self.player.get_time_pos() or 0.0)
                dur = float(self.player.get_duration() or 0.0)
                completed = self._is_episode_completed(pos, dur)
                source = "active-player"
            except Exception:
                completed = False
                source = "active-player-exc"
            if not completed:
                hist_active = self._current_history_entry()
                if hist_active is not None:
                    hp, hc = self._history_entry_best_progress_and_completed_for_sync(hist_active)
                    if hp > 0 and hc:
                        progress = hp
                        completed = True
                        source = "active-history-fallback"
        else:
            e = self._history_entry_for_sync_now()
            if e is not None:
                anime_name = e.name
                progress, completed = self._history_entry_best_progress_and_completed_for_sync(e)
                source = "history"

        debug_log(
            "sync-now candidate: "
            f"source={source} anime={anime_name!r} progress={progress} completed={completed}"
        )

        fail_reason = ""
        if not anime_name:
            fail_reason = "no-anime"
        elif int(progress) <= 0:
            fail_reason = "no-progress"
        elif not bool(completed):
            fail_reason = "not-completed"

        if fail_reason:
            debug_log(
                "sync-now blocked: "
                f"reason={fail_reason} source={source} anime={anime_name!r} "
                f"progress={progress} completed={completed}"
            )
            if fail_reason in ("no-anime", "no-progress"):
                self.set_status("Sincronizza ora: nessun anime/episodio attivo da sincronizzare.")
            else:
                self.set_status("Sincronizza ora: episodio non completato, sync non inviato.")
            return

        status = "CURRENT"
        self.set_status(f"Sincronizzazione AniList: {anime_name} ep {progress}…")
        self._begin_request()
        w = Worker(self._anilist_sync_worker, anime_name, progress, status, token)

        def on_ok(_data: object, name=anime_name, ep=progress):
            self._end_request()
            self.set_status(f"Sincronizzazione AniList ok: {name} ep {ep}")

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"AniList sync fallito: {msg}")
            self.set_status("Sincronizzazione AniList fallita.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()

    def _history_entry_for_sync_now(self) -> HistoryEntry | None:
        if getattr(self, "_selected_history", None):
            return self._selected_history
        try:
            cur = self._history_entry_from_current_item()
            if cur is not None:
                self._selected_history = cur
                return self._selected_history
        except Exception:
            pass
        items = self.history.list()
        if items:
            self._selected_history = items[0]
            return self._selected_history
        return None

    def _history_entry_best_progress_for_sync(self, entry: HistoryEntry) -> int:
        best, _completed = self._history_entry_best_progress_and_completed_for_sync(entry)
        return best

    def _history_entry_best_progress_and_completed_for_sync(self, entry: HistoryEntry) -> tuple[int, bool]:
        vals: list[float] = []
        completed_vals: list[float] = []
        try:
            w = [float(x) for x in (entry.watched_eps or [])]
            vals.extend(w)
            completed_vals.extend(w)
        except Exception:
            pass
        ep_map = entry.episode_progress if isinstance(entry.episode_progress, dict) else {}
        for k, v in ep_map.items():
            try:
                epf = float(k)
            except Exception:
                continue
            if isinstance(v, dict):
                comp = bool(v.get("completed", False))
                vals.append(epf)
                if comp:
                    completed_vals.append(epf)
        if not vals:
            return 0, False
        best = max(vals)
        best_completed = max(completed_vals) if completed_vals else 0.0
        if best_completed > 0:
            return int(max(1, best_completed)), True
        return int(max(1, best)), False

    def on_anilist_pull_from_remote(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: sync esterno disabilitato.")
            return
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            return
        provider_name = str(self.provider_name or "allanime")
        lang_name = "DUB" if self.lang == LanguageTypeEnum.DUB else "SUB"
        self.set_status("Pull AniList -> Anigui in corso…")
        self._begin_request()
        w = Worker(self._anilist_pull_worker, token, provider_name, lang_name)

        def on_ok(res: dict[str, Any]):
            self._end_request()
            built = list(res.get("built") or [])
            imported = 0
            skipped_local = 0
            for incoming in built:
                existing: HistoryEntry | None = None
                for e in self.history._data:
                    if (
                        e.provider == incoming.provider
                        and e.identifier == incoming.identifier
                        and e.lang == incoming.lang
                    ):
                        existing = e
                        break
                if existing is None:
                    self.history.upsert(incoming)
                    imported += 1
                    continue

                # Never downgrade local progress.
                if float(existing.last_ep) > float(incoming.last_ep):
                    skipped_local += 1
                    continue

                merged = HistoryEntry(
                    provider=existing.provider,
                    identifier=existing.identifier,
                    name=existing.name or incoming.name,
                    lang=existing.lang,
                    last_ep=max(float(existing.last_ep), float(incoming.last_ep)),
                    updated_at=time.time(),
                    cover_url=existing.cover_url or incoming.cover_url,
                    last_pos=existing.last_pos,
                    last_duration=existing.last_duration,
                    last_percent=max(float(existing.last_percent), float(incoming.last_percent)),
                    completed=bool(existing.completed or incoming.completed),
                    watched_eps=sorted(
                        set(float(x) for x in (existing.watched_eps or []))
                        | set(float(x) for x in (incoming.watched_eps or []))
                    ),
                    episode_progress=dict(existing.episode_progress or {}),
                )
                for k, v in (incoming.episode_progress or {}).items():
                    if k not in merged.episode_progress:
                        merged.episode_progress[k] = v
                self.history.upsert(merged)
                imported += 1

            self.refresh_history_ui()
            planned_missing = list(res.get("planned_missing_keys") or [])
            planned_all = int(res.get("remote_planned", 0))
            planned_imp = int(res.get("imported_planned", 0))
            debug_log(
                "AniList pull planned diag: "
                f"imported={planned_imp}/{planned_all} "
                f"missing={planned_missing}"
            )
            diag = ""
            if planned_missing:
                preview = ", ".join(str(x) for x in planned_missing[:8])
                if len(planned_missing) > 8:
                    preview += ", ..."
                diag = self._tr(
                    f" chiavi pianificate mancanti: {preview}",
                    f" missing planned keys: {preview}",
                )
            self.set_status(
                self._tr(
                    "Pull AniList completato: "
                    f"{imported} importati, "
                    f"{int(res.get('skipped', 0))} non matchati, "
                    f"{skipped_local} ignorati (progress locale migliore), "
                    f"pianificati {int(res.get('imported_planned', 0))}/{int(res.get('remote_planned', 0))}."
                    f"{diag}",
                    "AniList pull completed: "
                    f"{imported} imported, "
                    f"{int(res.get('skipped', 0))} unmatched, "
                    f"{skipped_local} skipped (better local progress), "
                    f"planned {int(res.get('imported_planned', 0))}/{int(res.get('remote_planned', 0))}."
                    f"{diag}",
                )
            )

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"Pull AniList fallito: {msg}")
            self.set_status("Pull AniList fallito.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()

    def _timeline_diag(self, msg: str, force: bool = False):
        now = time.time()
        if not force and (now - self._last_timeline_diag_at) < 2.0:
            return
        self._last_timeline_diag_at = now
        debug_log(f"timeline: {msg}")

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
        return cleaned or "anime"

    @staticmethod
    def _compact_title(title: str, max_len: int = 34) -> str:
        t = (title or "").strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 1].rstrip() + "…"

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(max(0, n))
        i = 0
        while v >= 1024.0 and i < len(units) - 1:
            v /= 1024.0
            i += 1
        return f"{v:.1f}{units[i]}"

    def _check_download_integrity(self, out_path: str) -> str | None:
        try:
            sz = os.path.getsize(out_path)
        except Exception:
            return "file mancante"
        min_bytes = int(max(0.0, self._integrity_min_mb) * 1024 * 1024)
        if sz < min_bytes:
            return f"file troppo piccolo ({self._fmt_bytes(sz)} < {self._fmt_bytes(min_bytes)})"
        return None

    def _download_label(self, t: DownloadTask) -> str:
        ep = t.episode
        st = t.status
        tag_done = "[DONE]" if self.app_language == "en" else "[FATTO]"
        tag_fail = "[FAIL]" if self.app_language == "en" else "[ERRORE]"
        tag_stop = "[STOP]" if self.app_language == "en" else "[STOP]"
        tag_resolve = "[RESOLVE]" if self.app_language == "en" else "[RISOLVI]"
        tag_queue = "[QUEUE]" if self.app_language == "en" else "[CODA]"
        if st == "downloading":
            if t.total and t.total > 0:
                pct = int(max(0.0, min(100.0, t.progress)))
                return (
                    f"[{pct:3d}%] {t.anime_name} · ep {ep} · "
                    f"{self._fmt_bytes(t.downloaded)}/{self._fmt_bytes(t.total)}"
                )
            return f"[..] {t.anime_name} · ep {ep} · {self._fmt_bytes(t.downloaded)}"
        if st == "completed":
            return f"{tag_done} {t.anime_name} · ep {ep}"
        if st == "failed":
            return f"{tag_fail} {t.anime_name} · ep {ep} · {t.error or self._tr('errore', 'error')}"
        if st == "cancelled":
            return f"{tag_stop} {t.anime_name} · ep {ep}"
        if st == "resolving":
            return f"{tag_resolve} {t.anime_name} · ep {ep}"
        return f"{tag_queue} {t.anime_name} · ep {ep}"

    def _refresh_downloads_ui(self):
        self.downloads_list.clear()
        for task_id in self._download_order:
            t = self._download_tasks.get(task_id)
            if not t:
                continue
            item = QListWidgetItem(self._download_label(t))
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self.downloads_list.addItem(item)

    def _queue_download(self, anime_obj: Any, anime_name: str, ep: float | int):
        self._remember_offline_cover_url(
            anime_name,
            getattr(self.selected_search_item, "cover_url", None) if self.selected_search_item else None,
        )
        ep_token = str(int(ep)) if isinstance(ep, float) and ep.is_integer() else str(ep)
        task_id = f"{self.provider_name}:{getattr(anime_obj, 'name', anime_name)}:{ep_token}:{int(time.time()*1000)}"
        lang_tag = "dub" if self.lang == LanguageTypeEnum.DUB else "sub"
        base_name = self._safe_name(anime_name)
        anime_dir = os.path.join(self.download_dir, base_name)
        os.makedirs(anime_dir, exist_ok=True)
        ext = ".mp4"

        # If this episode already exists locally, mark task as done and skip re-download.
        existing_pat = re.compile(
            rf"^{re.escape(base_name)} - ep {re.escape(ep_token)} \[{re.escape(lang_tag)}\](?: \(\d+\))?{re.escape(ext)}$",
            re.IGNORECASE,
        )
        existing_path = None
        try:
            for fn in os.listdir(anime_dir):
                if existing_pat.match(fn):
                    p = os.path.join(anime_dir, fn)
                    if os.path.isfile(p):
                        existing_path = p
                        break
        except Exception:
            existing_path = None

        if existing_path is not None:
            task = DownloadTask(
                task_id=task_id,
                anime_name=anime_name,
                episode=ep,
                provider=self.provider_name,
                lang=self.lang,
                quality=self.quality,
                anime_obj=anime_obj,
                out_path=existing_path,
                status="completed",
                progress=100.0,
                downloaded=os.path.getsize(existing_path),
                total=os.path.getsize(existing_path),
                error=None,
                max_retries=self._integrity_retry_count,
            )
            self._download_tasks[task_id] = task
            self._download_order.append(task_id)
            self._refresh_downloads_ui()
            self.set_status(f"Gia presente: {os.path.basename(existing_path)}")
            self.refresh_offline_library()
            return

        out_name = f"{base_name} - ep {ep_token} [{lang_tag}]{ext}"
        out_path = os.path.join(anime_dir, out_name)
        suffix = 1
        while os.path.exists(out_path):
            out_name = f"{base_name} - ep {ep_token} [{lang_tag}] ({suffix}){ext}"
            out_path = os.path.join(anime_dir, out_name)
            suffix += 1

        task = DownloadTask(
            task_id=task_id,
            anime_name=anime_name,
            episode=ep,
            provider=self.provider_name,
            lang=self.lang,
            quality=self.quality,
            anime_obj=anime_obj,
            out_path=out_path,
            max_retries=self._integrity_retry_count,
        )
        self._download_tasks[task_id] = task
        self._download_order.append(task_id)
        self._refresh_downloads_ui()

    def on_download_add_current(self):
        anime = self.selected_anime
        if anime is None:
            return
        ep = self.current_ep
        if ep is None and self.episodes_list:
            row = self.episodes.currentRow()
            if row >= 0:
                ep = self.episodes_list[row]
        if ep is None:
            self.notify_err("Seleziona o avvia un episodio prima di aggiungerlo alla coda download.")
            return
        name = getattr(anime, "name", None) or getattr(self.selected_search_item, "name", "anime")
        self._queue_download(anime, str(name), ep)
        self.set_status(f"Aggiunto download: {name} ep {ep}")

    def on_download_add_selected_episodes(self):
        anime = self.selected_anime
        if anime is None:
            self.notify_err("Apri prima un anime.")
            return
        selected = self.episodes.selectedIndexes()
        if not selected:
            self.notify_err("Seleziona uno o piu episodi dalla lista.")
            return
        name = getattr(anime, "name", None) or getattr(self.selected_search_item, "name", "anime")
        added = 0
        in_flight = {
            (t.provider, t.anime_name, str(t.episode))
            for t in self._download_tasks.values()
            if t.status in ("queued", "resolving", "downloading")
        }
        for idx in selected:
            row = idx.row()
            if row < 0 or row >= len(self.episodes_list):
                continue
            ep = self.episodes_list[row]
            dedupe_key = (self.provider_name, str(name), str(ep))
            if dedupe_key in in_flight:
                continue
            self._queue_download(anime, str(name), ep)
            in_flight.add(dedupe_key)
            added += 1
        if added > 0:
            self.set_status(f"Aggiunti {added} episodi alla coda download.")
        else:
            self.set_status("Nessun nuovo episodio aggiunto (gia in coda/attivi).")

    def on_download_start_queue(self):
        self._start_next_downloads()

    def on_download_parallel_change(self):
        self._max_parallel_downloads = int(self.combo_dl_parallel.currentData() or 1)
        self._start_next_downloads()

    def _start_next_downloads(self):
        if not self._is_scheduler_window_open():
            self.set_status("Scheduler attivo: fuori finestra oraria, coda in pausa.")
            return
        while (len(self._active_download_workers) + len(self._active_download_resolve_ids)) < self._max_parallel_downloads:
            next_task = None
            for task_id in self._download_order:
                t = self._download_tasks.get(task_id)
                if t and t.status == "queued":
                    next_task = t
                    break
            if next_task is None:
                return

            next_task.status = "resolving"
            self._active_download_resolve_ids.add(next_task.task_id)
            self._refresh_downloads_ui()
            self.set_status(f"Risolvo stream per download: {next_task.anime_name} ep {next_task.episode}")

            w = Worker(
                self.do_resolve_stream,
                next_task.anime_obj,
                next_task.episode,
                next_task.lang,
                next_task.quality if next_task.provider == "allanime" else "best",
                next_task.provider,
            )

            def on_ok(res: StreamResult, tid=next_task.task_id):
                self._active_download_resolve_ids.discard(tid)
                task = self._download_tasks.get(tid)
                if not task or task.status != "resolving":
                    self._start_next_downloads()
                    return
                task.status = "downloading"
                self._refresh_downloads_ui()
                dw = DownloadWorker(tid, res.url, task.out_path, referrer=res.referrer)
                self._active_download_workers[tid] = dw
                dw.progress.connect(self._on_download_progress)
                dw.done.connect(self._on_download_done)
                dw.fail.connect(self._on_download_fail)
                dw.cancelled.connect(self._on_download_cancelled)
                dw.start()
                self._start_next_downloads()

            def on_err(msg: str, tid=next_task.task_id):
                self._active_download_resolve_ids.discard(tid)
                task = self._download_tasks.get(tid)
                if task:
                    task.status = "failed"
                    task.error = msg
                self._refresh_downloads_ui()
                self.set_status("Errore resolve download.")
                self._start_next_downloads()

            w.ok.connect(on_ok)
            w.err.connect(on_err)
            self._workers.append(w)
            w.start()

    def _on_download_progress(self, task_id: str, pct: int, downloaded: int, total: int):
        task = self._download_tasks.get(task_id)
        if not task:
            return
        task.downloaded = downloaded
        task.total = total if total > 0 else None
        task.progress = float(max(0, pct)) if pct >= 0 else task.progress
        task.status = "downloading"
        self._refresh_downloads_ui()

    def _on_download_done(self, task_id: str, out_path: str):
        task = self._download_tasks.get(task_id)
        self._active_download_workers.pop(task_id, None)
        if task:
            err = self._check_download_integrity(out_path)
            if err is not None:
                task.retry_count += 1
                if task.retry_count <= task.max_retries:
                    task.status = "queued"
                    task.error = f"Integrity fail, retry {task.retry_count}/{task.max_retries}: {err}"
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass
                    self._refresh_downloads_ui()
                    self.set_status(f"Integrity check fallita, retry: {task.anime_name} ep {task.episode}")
                    self._start_next_downloads()
                    return
                task.status = "failed"
                task.error = f"Integrity fail: {err}"
            else:
                task.status = "completed"
                task.progress = 100.0
                task.error = None
        self._refresh_downloads_ui()
        self.refresh_offline_library()
        if task and task.status == "completed":
            self.set_status(f"Download completato: {os.path.basename(out_path)}")
        else:
            self.set_status("Download fallito (integrity).")
        self._start_next_downloads()

    def _on_download_fail(self, task_id: str, msg: str):
        task = self._download_tasks.get(task_id)
        if task:
            task.status = "failed"
            task.error = msg
        self._active_download_workers.pop(task_id, None)
        self._refresh_downloads_ui()
        self.set_status("Download fallito.")
        self._start_next_downloads()

    def _on_download_cancelled(self, task_id: str):
        task = self._download_tasks.get(task_id)
        if task:
            task.status = "cancelled"
        self._active_download_workers.pop(task_id, None)
        self._refresh_downloads_ui()
        self.set_status("Download annullato.")
        self._start_next_downloads()

    def on_download_cancel_selected(self):
        item = self.downloads_list.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        task = self._download_tasks.get(task_id)
        if not task:
            return
        worker = self._active_download_workers.get(task_id)
        if worker is not None:
            worker.request_cancel()
            return
        if task.status in ("queued", "resolving"):
            task.status = "cancelled"
            self._active_download_resolve_ids.discard(task_id)
            self._refresh_downloads_ui()
            self._start_next_downloads()

    def on_download_clear_completed(self):
        keep: list[str] = []
        for task_id in self._download_order:
            t = self._download_tasks.get(task_id)
            if not t:
                continue
            if t.status == "completed":
                self._download_tasks.pop(task_id, None)
                continue
            keep.append(task_id)
        self._download_order = keep
        self._refresh_downloads_ui()

    def on_download_open_folder(self):
        path = self.download_dir
        debug_log(f"open-folder: requested path={path}")
        try:
            os.makedirs(path, exist_ok=True)
            debug_log("open-folder: ensured directory exists")
            system = platform.system()
            if system == "Linux":
                nautilus = shutil.which("nautilus")
                if not nautilus:
                    self.notify_err(
                        "Su Linux al momento e supportato solo Nautilus.\n"
                        "Installa Nautilus oppure apri manualmente il percorso:\n"
                        f"{path}"
                    )
                    return
                debug_log(f"open-folder: linux try {nautilus} {path}")
                detached = QProcess.startDetached(nautilus, [path])
                debug_log(f"open-folder: linux detached={'yes' if detached else 'no'} {nautilus} {path}")
                if not detached:
                    self.notify_err(
                        "Impossibile avviare Nautilus automaticamente.\n"
                        f"Percorso: {path}"
                    )
                return

            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            debug_log(f"open-folder: QDesktopServices.openUrl -> {opened}")
            if opened:
                return
            if system == "Windows":
                debug_log("open-folder: fallback os.startfile")
                os.startfile(path)  # type: ignore[attr-defined]
            elif system == "Darwin":
                debug_log("open-folder: fallback open")
                subprocess.run(["open", path], check=True)
            else:
                debug_log("open-folder: fallback xdg-open")
                subprocess.run(["xdg-open", path], check=True)
        except Exception as ex:
            debug_log(f"open-folder: error={ex}")
            self.notify_err(str(ex))

    @staticmethod
    def _is_local_media_file(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    @staticmethod
    def _natural_sort_key(text: str):
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]

    def refresh_offline_library(self):
        self._offline_nonce += 1
        nonce = self._offline_nonce
        self.offline_results.clear()
        self.offline_episodes.clear()
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)
        self._offline_items = []
        self._offline_episode_files = []
        self._offline_current_anime_dir = None
        self._offline_current_episode_index = None

        try:
            os.makedirs(self.download_dir, exist_ok=True)
        except Exception:
            pass

        dirs: list[str] = []
        try:
            for name in os.listdir(self.download_dir):
                full = os.path.join(self.download_dir, name)
                if not os.path.isdir(full):
                    continue
                try:
                    has_media = any(
                        self._is_local_media_file(os.path.join(full, f))
                        for f in os.listdir(full)
                    )
                except Exception:
                    has_media = False
                if has_media:
                    dirs.append(full)
        except Exception:
            dirs = []

        dirs.sort(key=lambda p: self._natural_sort_key(os.path.basename(p)))
        self._offline_items = [
            OfflineAnimeItem(
                name=os.path.basename(d),
                folder=d,
                cover_url=self._offline_cover_url_for_name(os.path.basename(d)),
            )
            for d in dirs
        ]

        if not self._offline_items:
            self.offline_results.addItem(self._tr("(nessun anime scaricato)", "(no downloaded anime)"))
            self.offline_results.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return

        placeholder = self._make_cover_placeholder()
        for row, it in enumerate(self._offline_items):
            item = QListWidgetItem(placeholder, it.name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            item.setSizeHint(QSize(170, 290))
            self.offline_results.addItem(item)
            if it.cover_url:
                w = Worker(self._offline_cover_cached_only, it.cover_url)
                w.ok.connect(lambda data, r=row, n=nonce: self._on_offline_cover_loaded(r, n, data[0], data[1]))
                w.err.connect(lambda _msg: None)
                self._workers.append(w)
                w.start()

    def on_pick_offline_anime(self):
        row = self.offline_results.currentRow()
        if row < 0 or row >= len(self._offline_items):
            return
        picked = self._offline_items[row]
        anime_dir = picked.folder
        if not anime_dir or not os.path.isdir(anime_dir):
            return

        self.offline_episodes.clear()
        files: list[str] = []
        try:
            for name in os.listdir(anime_dir):
                full = os.path.join(anime_dir, name)
                if os.path.isfile(full) and self._is_local_media_file(full):
                    files.append(full)
        except Exception:
            files = []

        files.sort(key=lambda p: self._natural_sort_key(os.path.basename(p)))
        self._offline_current_anime_dir = anime_dir
        self._offline_episode_files = files
        self._offline_current_episode_index = None
        self.lbl_offline_anime_title.setText(picked.name)
        self.offline_stack.setCurrentWidget(self.page_offline_anime)

        if not files:
            self.offline_episodes.addItem(self._tr("(nessun episodio/file)", "(no episodes/files)"))
            self.offline_episodes.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return

        for fp in files:
            it = QListWidgetItem(os.path.basename(fp))
            it.setData(Qt.ItemDataRole.UserRole, fp)
            self.offline_episodes.addItem(it)

    def on_play_offline_episode(self, *_args):
        item = self.offline_episodes.currentItem()
        if item is None:
            return
        fp = item.data(Qt.ItemDataRole.UserRole)
        if not fp or not os.path.isfile(fp):
            return
        try:
            idx = self._offline_episode_files.index(fp)
        except Exception:
            idx = None
        self._offline_current_episode_index = idx
        self._play_local_file(fp)

    def on_offline_back_to_catalog(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)
        self.set_status("Catalogo offline.")

    def _offline_cover_url_for_name(self, name: str) -> str | None:
        target = self._safe_name(name or "").lower()
        if not target:
            return None
        mapped = self._offline_covers_map.get(target)
        if mapped:
            return mapped
        for e in self.history.list():
            hist_name = self._safe_name(e.name or "").lower()
            if hist_name == target and e.cover_url:
                self._remember_offline_cover_url(name, e.cover_url)
                return e.cover_url
        return None

    def _offline_cover_cached_only(self, cover_ref: str) -> tuple[bytes | None, bool]:
        # cover_ref can be:
        # 1) local cache path (*.img) [new format]
        # 2) original URL [legacy format]
        ref = (cover_ref or "").strip()
        if not ref:
            return None, False
        if ref.lower().endswith(".img") and os.path.exists(ref):
            cache_path = ref
        else:
            cache_path = self._cover_cache_path(ref)
        if not cache_path or not os.path.exists(cache_path):
            return None, False
        try:
            with open(cache_path, "rb") as f:
                data = f.read()
            return (data if data else None), bool(data)
        except Exception:
            return None, False

    def _on_offline_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._offline_nonce:
            return
        if data is None:
            return
        if row < 0 or row >= self.offline_results.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.offline_results.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.offline_results.item(row).setIcon(QIcon(scaled))

    def _play_local_file(self, path: str):
        if not os.path.isfile(path):
            self.notify_err("File locale non trovato.")
            return

        self._save_current_progress(force=True)
        self._pending_resume_seek = None
        self._pending_resume_seek_attempts = 0
        self._resume_marker_pos = None
        self._local_media_active = True
        self.current_ep = None
        self.selected_anime = None
        self.selected_search_item = None

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_player)
        self._current_search_view = "player"
        self._refresh_search_layout()
        base = os.path.basename(path)
        self.lbl_player_title.setText(f"Offline — {base}")
        self.set_status(f"▶ Offline: {base}")
        self._set_transport_enabled(False)
        try:
            self.player.load(path, referrer=None, sub_file=None)
        finally:
            self._set_transport_enabled(True)

    def on_back_to_catalog(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.search_stack.setCurrentWidget(self.page_catalog)
        self._current_search_view = "catalog"
        self._refresh_search_layout()
        self.set_status("Ricerca anime.")

    def on_back_to_episodes(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status("Lista episodi.")

    @staticmethod
    def _is_episode_completed(pos: float, dur: float) -> bool:
        if dur <= 0:
            return False
        remaining = max(0.0, dur - pos)
        return (pos / dur) >= 0.97 or remaining <= 90.0

    @staticmethod
    def _is_episode_completed_by_percent(percent: float | None) -> bool:
        if percent is None:
            return False
        try:
            p = float(percent)
        except Exception:
            return False
        return p >= 97.0

    def _save_current_progress(self, force: bool = False):
        if self._incognito_enabled:
            return
        if self._local_media_active:
            return
        if not self.selected_anime or self.current_ep is None:
            return
        try:
            pos_raw = self.player.get_time_pos()
        except Exception:
            pos_raw = None
        try:
            dur_raw = self.player.get_duration()
        except Exception:
            dur_raw = None
        try:
            percent_raw = self.player.get_percent_pos()
        except Exception:
            percent_raw = None

        now = time.time()

        identifier = (
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        ident_s = str(identifier)
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"

        prev_entry: HistoryEntry | None = None
        for e in self.history._data:
            if e.provider == self.provider_name and e.identifier == ident_s and e.lang == lang_s:
                prev_entry = e
                break

        dur: float | None
        if dur_raw is not None:
            dur = max(0.0, float(dur_raw))
        elif prev_entry is not None:
            dur = max(0.0, float(prev_entry.last_duration))
        else:
            dur = None

        percent: float | None = None
        if percent_raw is not None:
            try:
                percent = max(0.0, min(100.0, float(percent_raw)))
            except Exception:
                percent = None
        elif prev_entry is not None and float(getattr(prev_entry, "last_percent", 0.0) or 0.0) > 0.0:
            percent = max(0.0, min(100.0, float(prev_entry.last_percent)))

        pos: float | None = None
        if pos_raw is not None:
            pos = max(0.0, float(pos_raw))
        elif dur is not None and dur > 0 and percent_raw is not None:
            try:
                pos = max(0.0, min(dur, dur * (float(percent_raw) / 100.0)))
            except Exception:
                pos = None
        elif prev_entry is not None:
            pos = max(0.0, float(prev_entry.last_pos))

        # if we still cannot resolve a timeline position and no percent exists, skip save
        if pos is None and percent is None:
            return
        if pos is None:
            pos = max(0.0, float(prev_entry.last_pos)) if prev_entry is not None else 0.0
        if dur is None:
            dur = 0.0

        if not force:
            pos_delta_small = abs(pos - self._last_progress_pos) < 5.0
            pct_now = float(percent) if percent is not None else -1.0
            pct_delta_small = abs(pct_now - self._last_progress_percent) < 1.0
            if now - self._last_progress_save_at < 5.0 and pos_delta_small and pct_delta_small:
                return

        watched_eps: list[float] = list(prev_entry.watched_eps) if prev_entry is not None else []
        episode_progress: dict[str, dict[str, Any]] = (
            dict(prev_entry.episode_progress)
            if prev_entry is not None and isinstance(prev_entry.episode_progress, dict)
            else {}
        )
        cover_url = (
            getattr(self.selected_search_item, "cover_url", None)
            if self.selected_search_item
            else (prev_entry.cover_url if prev_entry is not None else None)
        )
        episode_completed_now = self._is_episode_completed(pos, dur) or self._is_episode_completed_by_percent(percent)
        completed_now = episode_completed_now and self._is_last_available_episode(
            self.current_ep,
            self.episodes_list,
        )

        entry = HistoryEntry(
            provider=self.provider_name,
            identifier=ident_s,
            name=self.selected_anime.name,
            lang=lang_s,
            cover_url=cover_url,
            last_ep=float(self.current_ep),
            updated_at=now,
            last_pos=pos,
            last_duration=dur,
            last_percent=float(percent) if percent is not None else 0.0,
            completed=completed_now,
            watched_eps=watched_eps,
            episode_progress=episode_progress,
        )
        self._entry_set_episode_progress(
            entry,
            self.current_ep,
            pos=pos,
            dur=dur,
            percent=float(percent) if percent is not None else 0.0,
            completed=episode_completed_now,
        )
        if episode_completed_now:
            self._entry_add_watched_ep(entry, self.current_ep)
        self.history.upsert(entry)
        self._anilist_sync_progress_async(entry, force=force)
        self._last_progress_save_at = now
        self._last_progress_pos = pos
        self._last_progress_percent = float(percent) if percent is not None else -1.0
        if force:
            self.refresh_history_ui()

    def _try_apply_resume_seek_ratio(self):
        ratio = self._pending_resume_seek_ratio
        if ratio is None:
            return
        try:
            self.player.seek_ratio(float(ratio))
        except Exception:
            pass
        self._pending_resume_seek_ratio = None

    def _try_apply_resume_seek(self):
        target = self._pending_resume_seek
        if target is None:
            return
        self._pending_resume_seek_attempts += 1
        try:
            self.player.seek(target)
        except Exception:
            pass

        cur = self.player.get_time_pos()
        if cur is not None and abs(float(cur) - target) <= 8.0:
            self._pending_resume_seek = None
            return

        if self._pending_resume_seek_attempts < 8:
            QTimer.singleShot(350, self._try_apply_resume_seek)
        else:
            self._pending_resume_seek = None
            if self._pending_resume_seek_ratio is not None:
                QTimer.singleShot(0, self._try_apply_resume_seek_ratio)

    @staticmethod
    def fmt_time(sec: float | None) -> str:
        if sec is None or sec != sec or sec < 0:
            return "00:00"
        s = int(sec)
        m = s // 60
        s = s % 60
        h = m // 60
        m = m % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _fmt_relative_time(self, ts: float | None) -> str:
        if not ts:
            return self._tr("proprio adesso", "just now")
        delta = max(0, int(time.time() - float(ts)))
        if delta < 60:
            return self._tr("proprio adesso", "just now")
        if delta < 3600:
            n = delta // 60
            return self._tr(
                f"{n} minuto{'i' if n != 1 else ''} fa",
                f"{n} minute{'s' if n != 1 else ''} ago",
            )
        if delta < 86400:
            n = delta // 3600
            return self._tr(
                f"{n} ora{'e' if n != 1 else ''} fa",
                f"{n} hour{'s' if n != 1 else ''} ago",
            )
        if delta < 86400 * 30:
            n = delta // 86400
            return self._tr(
                f"{n} giorno{'i' if n != 1 else ''} fa",
                f"{n} day{'s' if n != 1 else ''} ago",
            )
        if delta < 86400 * 365:
            n = delta // (86400 * 30)
            return self._tr(
                f"{n} mese{'i' if n != 1 else ''} fa",
                f"{n} month{'s' if n != 1 else ''} ago",
            )
        n = delta // (86400 * 365)
        return self._tr(
            f"{n} anno{'i' if n != 1 else ''} fa",
            f"{n} year{'s' if n != 1 else ''} ago",
        )

    @staticmethod
    def _extract_cover_url(obj: Any) -> str | None:
        for key in ("cover", "cover_url", "image", "img", "poster", "thumbnail"):
            v = getattr(obj, key, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _episode_label(self, ep: float | int) -> str:
        return self._tr(f"Episodio {ep}", f"Episode {ep}")

    @staticmethod
    def _to_ep_float(ep: float | int | str | None) -> float | None:
        try:
            return float(ep) if ep is not None else None
        except Exception:
            return None

    def _entry_watched_eps_set(self, entry: HistoryEntry) -> set[float]:
        out: set[float] = set()
        for v in (entry.watched_eps or []):
            f = self._to_ep_float(v)
            if f is not None:
                out.add(f)
        # Legacy fallback: consider only current episode as seen if the entry is completed.
        if not out and bool(entry.completed):
            last = self._to_ep_float(entry.last_ep)
            if last is not None and last > 0:
                out.add(float(last))
        return out

    def _entry_add_watched_ep(self, entry: HistoryEntry, ep: float | int | str | None) -> None:
        f = self._to_ep_float(ep)
        if f is None or f <= 0:
            return
        eps = self._entry_watched_eps_set(entry)
        eps.add(float(f))
        entry.watched_eps = sorted(eps)

    @staticmethod
    def _ep_key(ep: float | int | str | None) -> str:
        try:
            f = float(ep)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except Exception:
            return str(ep or "")

    def _entry_episode_progress_map(self, entry: HistoryEntry) -> dict[float, dict[str, Any]]:
        out: dict[float, dict[str, Any]] = {}
        raw = entry.episode_progress if isinstance(entry.episode_progress, dict) else {}
        for k, v in raw.items():
            try:
                epf = float(k)
            except Exception:
                continue
            if isinstance(v, dict):
                out[epf] = dict(v)
        return out

    def _entry_get_episode_progress(
        self,
        entry: HistoryEntry,
        ep: float | int | str | None,
    ) -> dict[str, Any] | None:
        key = self._ep_key(ep)
        raw = entry.episode_progress if isinstance(entry.episode_progress, dict) else {}
        val = raw.get(key)
        return dict(val) if isinstance(val, dict) else None

    def _entry_set_episode_progress(
        self,
        entry: HistoryEntry,
        ep: float | int | str | None,
        pos: float,
        dur: float,
        percent: float,
        completed: bool,
    ) -> None:
        raw = dict(entry.episode_progress) if isinstance(entry.episode_progress, dict) else {}
        raw[self._ep_key(ep)] = {
            "pos": float(max(0.0, pos)),
            "dur": float(max(0.0, dur)),
            "percent": float(max(0.0, min(100.0, percent))),
            "completed": bool(completed),
            "updated_at": time.time(),
        }
        entry.episode_progress = raw

    def _is_last_available_episode(
        self,
        ep: float | int | str | None,
        available_eps: list[float | int] | None,
    ) -> bool:
        if not available_eps:
            return False
        epf = self._to_ep_float(ep)
        if epf is None:
            return False
        vals: list[float] = []
        for x in available_eps:
            xf = self._to_ep_float(x)
            if xf is not None:
                vals.append(xf)
        if not vals:
            return False
        max_ep = max(vals)
        return abs(epf - max_ep) < 1e-6

    def _entry_is_series_completed(
        self,
        entry: HistoryEntry,
        available_eps: list[float | int] | None,
    ) -> bool:
        if not available_eps:
            return False
        vals: list[float] = []
        for x in available_eps:
            xf = self._to_ep_float(x)
            if xf is not None:
                vals.append(xf)
        if not vals:
            return False
        max_ep = max(vals)
        watched = self._entry_watched_eps_set(entry)
        if any(abs(max_ep - w) < 1e-6 for w in watched):
            return True
        p = self._entry_get_episode_progress(entry, max_ep)
        return bool(p and p.get("completed", False))

    def _is_aw_provider_name(self, name: str) -> bool:
        return name in ("aw_animeworld", "aw_animeunity")

    def _ensure_aw_runtime(self):
        if self._aw_runtime_ready:
            return
        if platform.system() == "Windows":
            shim_dir = os.path.join(app_state_dir(), "aw_shims")
            os.makedirs(shim_dir, exist_ok=True)
            uname_bat = os.path.join(shim_dir, "uname.bat")
            if not os.path.exists(uname_bat):
                with open(uname_bat, "w", encoding="utf-8") as f:
                    f.write("@echo off\n")
                    f.write("echo Windows_NT anigui\n")
            cur_path = os.environ.get("PATH", "")
            parts = cur_path.split(os.pathsep) if cur_path else []
            if shim_dir not in parts:
                os.environ["PATH"] = shim_dir + os.pathsep + cur_path
        self._aw_runtime_ready = True

    def _aw_provider_class(self, name: str):
        self._ensure_aw_runtime()
        try:
            mod = importlib.import_module("aw_cli.providers")
        except Exception as ex:
            raise RuntimeError(
                "Provider ITA non disponibile: controlla aw-cli su questo sistema "
                f"(errore import: {ex})."
            ) from ex

        if name == "aw_animeworld":
            return getattr(mod, "Animeworld")
        if name == "aw_animeunity":
            return getattr(mod, "Animeunity")
        raise RuntimeError(f"Provider ITA sconosciuto: {name}")

    def _aw_provider(self, name: str | None = None):
        key = name or self.provider_name
        if not self._is_aw_provider_name(key):
            raise RuntimeError("Provider non-ITA richiesto come aw-cli.")
        self._ensure_aw_config()
        if key not in self._aw_provider_instances:
            cls = self._aw_provider_class(key)
            self._aw_provider_instances[key] = cls()
        return self._aw_provider_instances[key]

    def _aw_anime_class(self):
        if self._aw_anime_cls is not None:
            return self._aw_anime_cls
        self._ensure_aw_runtime()
        try:
            mod = importlib.import_module("aw_cli.anime")
        except Exception as ex:
            raise RuntimeError(
                "Provider ITA non disponibile: controlla aw-cli su questo sistema "
                f"(errore import: {ex})."
            ) from ex
        self._aw_anime_cls = getattr(mod, "Anime")
        return self._aw_anime_cls

    def _aw_cover_cache_key(self, item: SearchItem) -> str:
        return f"{item.source}:{item.identifier}"

    def _aw_resolve_cover_url(self, item: SearchItem) -> str | None:
        key = self._aw_cover_cache_key(item)
        miss_until = self._aw_cover_miss_until.get(key)
        if miss_until is not None and time.time() < miss_until:
            return None
        if key in self._aw_cover_url_cache:
            return self._aw_cover_url_cache[key]

        url: str | None = None
        try:
            provider = self._aw_provider(item.source)
            if item.source == "aw_animeunity":
                ident = str(item.identifier)
                if ident.isdigit():
                    r = provider.Client.get(f"{provider.BASE_URL}/info_api/{ident}/")
                    r.raise_for_status()
                    data = r.json()
                    for k in ("imageurl", "image_url", "image", "cover", "poster"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            url = v.strip()
                            break
            elif item.source == "aw_animeworld":
                ref = str(item.identifier)
                if ref and not ref.startswith("http"):
                    ref = provider.BASE_URL.rstrip("/") + "/" + ref.lstrip("/")
                if ref:
                    if hasattr(provider, "_get_html"):
                        html = provider._get_html(ref)
                    else:
                        r = provider.Client.get(ref)
                        r.raise_for_status()
                        html = r.text
                    m = re.search(
                        r"<meta[^>]+property=['\"]og:image['\"][^>]+content=['\"]([^'\"]+)['\"]",
                        html,
                        re.IGNORECASE,
                    )
                    if not m:
                        m = re.search(
                            r"<meta[^>]+content=['\"]([^'\"]+)['\"][^>]+property=['\"]og:image['\"]",
                            html,
                            re.IGNORECASE,
                        )
                    if not m:
                        m = re.search(
                            r'<img[^>]+class="[^"]*poster[^"]*"[^>]+(?:src|data-src)="([^"]+)"',
                            html,
                            re.IGNORECASE,
                        )
                    if m:
                        url = m.group(1).strip()
                        if url.startswith("/"):
                            url = provider.BASE_URL.rstrip("/") + url
                    if not url:
                        # fallback permissivo: prima immagine assoluta nel markup
                        m = re.search(
                            r"(https?:\\/\\/[^'\"\\s>]+\\.(?:jpg|jpeg|png|webp))",
                            html,
                            re.IGNORECASE,
                        )
                        if m:
                            url = m.group(1).replace("\\/", "/")
                    if not url:
                        # fallback: cerca in pagina search per nome anime
                        q = quote_plus(item.name)
                        s_html = provider._get_html(f"{provider.BASE_URL}/search?keyword={q}")
                        href = str(item.identifier)
                        href_rel = href.replace(provider.BASE_URL, "")
                        block_pat = re.compile(
                            r'<div class="inner">(?P<block>.*?)</div>',
                            re.IGNORECASE | re.DOTALL,
                        )
                        for bm in block_pat.finditer(s_html):
                            block = bm.group("block")
                            if href not in block and href_rel not in block:
                                continue
                            im = re.search(
                                r'(?:src|data-src)=["\']([^"\']+\.(?:jpg|jpeg|png|webp)[^"\']*)["\']',
                                block,
                                re.IGNORECASE,
                            )
                            if im:
                                url = im.group(1).strip()
                                if url.startswith("//"):
                                    url = "https:" + url
                                elif url.startswith("/"):
                                    url = provider.BASE_URL.rstrip("/") + url
                                break
        except Exception:
            url = None

        if url:
            self._aw_cover_url_cache[key] = url
            self._aw_cover_miss_until.pop(key, None)
        else:
            self._aw_cover_miss_until[key] = time.time() + 600.0
            debug_log(f"cover resolve miss: source={item.source} id={item.identifier}")
        return url

    def _ensure_aw_config(self):
        if self._aw_config_ready:
            return
        self._ensure_aw_runtime()
        try:
            ut_mod = importlib.import_module("aw_cli.utilities")
        except Exception:
            return
        cfg = getattr(ut_mod, "configData", None)
        if cfg is None:
            self._aw_config_ready = True
            return
        general = cfg.setdefault("general", {})
        general.setdefault("specials", False)
        self._aw_config_ready = True

    def _make_cover_placeholder(self, title: str | None = None) -> QIcon:
        pix = QPixmap(300, 440)
        pix.fill(QColor("#2b2b2b"))
        painter = QPainter(pix)
        painter.fillRect(0, 0, 300, 100, QColor(229, 9, 20, 90))
        if title:
            painter.setPen(QColor("#e9edf1"))
            painter.drawText(
                16,
                280,
                268,
                130,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                title,
            )
        painter.end()
        return QIcon(pix)

    def _make_skeleton_pixmap(self, size: QSize, phase: int) -> QPixmap:
        w = max(1, size.width())
        h = max(1, size.height())
        pix = QPixmap(w, h)
        pix.fill(QColor("#1e1e1e"))
        painter = QPainter(pix)
        band_w = max(30, w // 4)
        x = (phase % (w + band_w)) - band_w
        grad = QColor(255, 255, 255, 30)
        painter.fillRect(x, 0, band_w, h, grad)
        painter.end()
        return pix

    def _start_skeletons(self):
        if not self._skeleton_timer.isActive():
            self._skeleton_timer.start()

    def _update_skeletons(self):
        self._skeleton_phase += 8
        any_loading = False
        for lw in (self.results, self.recent, self.recommended):
            for i in range(lw.count()):
                item = lw.item(i)
                if item.data(Qt.ItemDataRole.UserRole) != "loading":
                    continue
                any_loading = True
                pix = self._make_skeleton_pixmap(lw.iconSize(), self._skeleton_phase)
                item.setIcon(QIcon(pix))
        if not any_loading:
            self._skeleton_timer.stop()

    def _apply_cached_badge(self, pix: QPixmap) -> QPixmap:
        out = QPixmap(pix)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        radius = max(8, out.width() // 14)
        painter.setBrush(QColor(46, 184, 92, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(out.width() - radius * 2 - 6, 6, radius * 2, radius * 2)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(out.width() - radius * 2 - 6, 6, radius * 2, radius * 2, Qt.AlignmentFlag.AlignCenter, "C")
        painter.end()
        return out

    def _begin_request(self):
        self._requests_in_flight += 1
        if not self._spinner_timer.isActive():
            self._spinner_timer.start()

    def _end_request(self):
        self._requests_in_flight = max(0, self._requests_in_flight - 1)
        if self._requests_in_flight == 0:
            self._spinner_timer.stop()
            self.lbl_spinner.setText("")

    def _spin_tick(self):
        if self._requests_in_flight <= 0:
            self.lbl_spinner.setText("")
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.lbl_spinner.setText(self._spinner_frames[self._spinner_idx])

    @staticmethod
    def _default_settings_dict() -> dict[str, Any]:
        return {
            "download_dir": os.path.join(app_state_dir(), "downloads"),
            "default_provider": "allanime",
            "default_lang": "SUB",
            "default_quality": "best",
            "parallel_downloads": 2,
            "scheduler_enabled": False,
            "scheduler_start": "00:00",
            "scheduler_end": "23:59",
            "integrity_min_mb": 2.0,
            "integrity_retry_count": 1,
            "anilist_enabled": False,
            "anilist_token": "",
            "app_language": "it",
        }

    def _load_settings(self) -> dict[str, Any]:
        base = self._default_settings_dict()
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                base.update(raw)
        except Exception:
            pass
        return base

    def _save_settings(self, settings: dict[str, Any]):
        if self._incognito_enabled:
            return
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        os.replace(tmp, SETTINGS_PATH)
        self._settings = dict(settings)

    def _apply_runtime_from_settings(self):
        dl_dir = str(self._settings.get("download_dir", "")).strip()
        if dl_dir:
            self.download_dir = os.path.abspath(os.path.expanduser(dl_dir))
        else:
            self.download_dir = os.path.join(app_state_dir(), "downloads")
        os.makedirs(self.download_dir, exist_ok=True)
        self.provider_name = str(self._settings.get("default_provider", "allanime"))
        if self.provider_name not in ("allanime", "aw_animeworld", "aw_animeunity"):
            self.provider_name = "allanime"
        lang_s = str(self._settings.get("default_lang", "SUB")).upper()
        self.lang = LanguageTypeEnum.DUB if lang_s == "DUB" else LanguageTypeEnum.SUB
        self.quality = str(self._settings.get("default_quality", "best"))
        if self.quality not in ("best", "worst", "360", "480", "720", "1080"):
            self.quality = "best"
        try:
            self._max_parallel_downloads = int(self._settings.get("parallel_downloads", 2))
        except Exception:
            self._max_parallel_downloads = 2
        self._max_parallel_downloads = max(1, min(4, self._max_parallel_downloads))
        self._scheduler_enabled = bool(self._settings.get("scheduler_enabled", False))
        self._scheduler_start = str(self._settings.get("scheduler_start", "00:00"))
        self._scheduler_end = str(self._settings.get("scheduler_end", "23:59"))
        try:
            self._integrity_min_mb = float(self._settings.get("integrity_min_mb", 2.0))
        except Exception:
            self._integrity_min_mb = 2.0
        self._integrity_min_mb = max(0.0, self._integrity_min_mb)
        try:
            self._integrity_retry_count = int(self._settings.get("integrity_retry_count", 1))
        except Exception:
            self._integrity_retry_count = 1
        self._integrity_retry_count = max(0, min(5, self._integrity_retry_count))
        self._anilist_enabled = bool(self._settings.get("anilist_enabled", False))
        self._anilist_token = str(self._settings.get("anilist_token", "")).strip()
        self.app_language = str(self._settings.get("app_language", "it")).strip().lower()
        if self.app_language not in ("it", "en"):
            self.app_language = "it"

        pidx = self.provider_combo.findData(self.provider_name)
        if pidx >= 0:
            self.provider_combo.setCurrentIndex(pidx)
        self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.quality_combo.findText(self.quality)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)
        self.combo_dl_parallel.setCurrentIndex(self._max_parallel_downloads - 1)
        self._apply_provider_ui_state()
        self._apply_app_language_ui()

    def _sync_settings_ui_from_state(self):
        if not hasattr(self, "input_settings_download_dir"):
            return
        self.input_settings_download_dir.setText(self.download_dir)
        pidx = self.combo_settings_provider.findData(self.provider_name)
        if pidx >= 0:
            self.combo_settings_provider.setCurrentIndex(pidx)
        self.combo_settings_lang.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.combo_settings_quality.findData(self.quality)
        if qidx >= 0:
            self.combo_settings_quality.setCurrentIndex(qidx)
        didx = self.combo_settings_parallel.findData(self._max_parallel_downloads)
        if didx >= 0:
            self.combo_settings_parallel.setCurrentIndex(didx)
        sidx = self.combo_settings_scheduler_enabled.findData(self._scheduler_enabled)
        if sidx >= 0:
            self.combo_settings_scheduler_enabled.setCurrentIndex(sidx)
        self.input_settings_scheduler_start.setText(self._scheduler_start)
        self.input_settings_scheduler_end.setText(self._scheduler_end)
        self.input_settings_integrity_min_mb.setText(str(self._integrity_min_mb))
        ridx = self.combo_settings_integrity_retries.findData(self._integrity_retry_count)
        if ridx >= 0:
            self.combo_settings_integrity_retries.setCurrentIndex(ridx)
        aidx = self.combo_settings_anilist_enabled.findData(self._anilist_enabled)
        if aidx >= 0:
            self.combo_settings_anilist_enabled.setCurrentIndex(aidx)
        lidx = self.combo_settings_app_language.findData(self.app_language)
        if lidx >= 0:
            self.combo_settings_app_language.setCurrentIndex(lidx)
        self.input_settings_anilist_token.setText(self._anilist_token)

    def on_settings_browse_download_dir(self):
        start = self.input_settings_download_dir.text().strip() or self.download_dir
        picked = QFileDialog.getExistingDirectory(
            self,
            self._tr("Seleziona cartella download", "Select download directory"),
            start,
        )
        if picked:
            self.input_settings_download_dir.setText(os.path.abspath(picked))

    def on_settings_save(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: salvataggio settings bloccato.")
            return
        download_dir = self.input_settings_download_dir.text().strip()
        if not download_dir:
            self.notify_err("Download directory non valido.")
            return
        download_dir = os.path.abspath(os.path.expanduser(download_dir))
        try:
            os.makedirs(download_dir, exist_ok=True)
        except Exception as ex:
            self.notify_err(f"Impossibile creare la cartella download: {ex}")
            return

        provider = str(self.combo_settings_provider.currentData() or "allanime")
        lang_s = str(self.combo_settings_lang.currentData() or "SUB").upper()
        quality = str(self.combo_settings_quality.currentData() or "best")
        parallel = int(self.combo_settings_parallel.currentData() or 2)
        parallel = max(1, min(4, parallel))
        scheduler_enabled = bool(self.combo_settings_scheduler_enabled.currentData())
        scheduler_start = self.input_settings_scheduler_start.text().strip()
        scheduler_end = self.input_settings_scheduler_end.text().strip()
        if scheduler_enabled:
            if self._parse_hhmm(scheduler_start) is None or self._parse_hhmm(scheduler_end) is None:
                self.notify_err("Formato scheduler non valido. Usa HH:MM (es. 23:30).")
                return
        try:
            integrity_min_mb = float(self.input_settings_integrity_min_mb.text().strip())
        except Exception:
            integrity_min_mb = self._integrity_min_mb
        integrity_min_mb = max(0.0, integrity_min_mb)
        integrity_retry_count = int(self.combo_settings_integrity_retries.currentData() or 1)
        integrity_retry_count = max(0, min(5, integrity_retry_count))
        anilist_enabled = bool(self.combo_settings_anilist_enabled.currentData())
        anilist_token = self.input_settings_anilist_token.text().strip()
        app_language = str(self.combo_settings_app_language.currentData() or "it").strip().lower()
        if app_language not in ("it", "en"):
            app_language = "it"
        if anilist_enabled and not anilist_token:
            self.notify_err("Inserisci un AniList token oppure disattiva AniList sync.")
            return

        settings = {
            "download_dir": download_dir,
            "default_provider": provider,
            "default_lang": lang_s,
            "default_quality": quality,
            "parallel_downloads": parallel,
            "scheduler_enabled": scheduler_enabled,
            "scheduler_start": scheduler_start,
            "scheduler_end": scheduler_end,
            "integrity_min_mb": integrity_min_mb,
            "integrity_retry_count": integrity_retry_count,
            "anilist_enabled": anilist_enabled,
            "anilist_token": anilist_token,
            "app_language": app_language,
        }
        try:
            self._save_settings(settings)
        except Exception as ex:
            self.notify_err(f"Errore salvataggio settings: {ex}")
            return

        self.download_dir = download_dir
        self.provider_name = provider
        self.lang = LanguageTypeEnum.DUB if lang_s == "DUB" else LanguageTypeEnum.SUB
        self.quality = quality
        self._max_parallel_downloads = parallel
        self._scheduler_enabled = scheduler_enabled
        self._scheduler_start = scheduler_start
        self._scheduler_end = scheduler_end
        self._integrity_min_mb = integrity_min_mb
        self._integrity_retry_count = integrity_retry_count
        self._anilist_enabled = anilist_enabled
        self._anilist_token = anilist_token
        self.app_language = app_language

        pidx = self.provider_combo.findData(self.provider_name)
        if pidx >= 0:
            self.provider_combo.setCurrentIndex(pidx)
        self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.quality_combo.findText(self.quality)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)
        self.combo_dl_parallel.setCurrentIndex(self._max_parallel_downloads - 1)
        self._apply_provider_ui_state()
        self._apply_app_language_ui()
        self.refresh_history_ui()
        self.refresh_offline_library()
        self.set_status(self._tr("Impostazioni salvate.", "Settings saved."))

    def on_settings_reset(self):
        defaults = self._default_settings_dict()
        self.input_settings_download_dir.setText(str(defaults["download_dir"]))
        pidx = self.combo_settings_provider.findData(defaults["default_provider"])
        if pidx >= 0:
            self.combo_settings_provider.setCurrentIndex(pidx)
        self.combo_settings_lang.setCurrentIndex(0)
        qidx = self.combo_settings_quality.findData(defaults["default_quality"])
        if qidx >= 0:
            self.combo_settings_quality.setCurrentIndex(qidx)
        didx = self.combo_settings_parallel.findData(defaults["parallel_downloads"])
        if didx >= 0:
            self.combo_settings_parallel.setCurrentIndex(didx)
        sidx = self.combo_settings_scheduler_enabled.findData(defaults["scheduler_enabled"])
        if sidx >= 0:
            self.combo_settings_scheduler_enabled.setCurrentIndex(sidx)
        self.input_settings_scheduler_start.setText(str(defaults["scheduler_start"]))
        self.input_settings_scheduler_end.setText(str(defaults["scheduler_end"]))
        self.input_settings_integrity_min_mb.setText(str(defaults["integrity_min_mb"]))
        ridx = self.combo_settings_integrity_retries.findData(defaults["integrity_retry_count"])
        if ridx >= 0:
            self.combo_settings_integrity_retries.setCurrentIndex(ridx)
        aidx = self.combo_settings_anilist_enabled.findData(defaults["anilist_enabled"])
        if aidx >= 0:
            self.combo_settings_anilist_enabled.setCurrentIndex(aidx)
        lidx = self.combo_settings_app_language.findData(defaults["app_language"])
        if lidx >= 0:
            self.combo_settings_app_language.setCurrentIndex(lidx)
        self.input_settings_anilist_token.setText(str(defaults["anilist_token"]))
        self.set_status(self._tr("Impostazioni ripristinate ai default (premi Salva).", "Settings reset to defaults (press Save)."))

    @staticmethod
    def _parse_hhmm(v: str) -> tuple[int, int] | None:
        m = re.fullmatch(r"(\d{2}):(\d{2})", v.strip())
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return hh, mm

    def _is_scheduler_window_open(self) -> bool:
        if not self._scheduler_enabled:
            return True
        s = self._parse_hhmm(self._scheduler_start)
        e = self._parse_hhmm(self._scheduler_end)
        if s is None or e is None:
            return True
        now = time.localtime()
        cur = now.tm_hour * 60 + now.tm_min
        start = s[0] * 60 + s[1]
        end = e[0] * 60 + e[1]
        if start <= end:
            return start <= cur <= end
        return cur >= start or cur <= end

    def _scheduler_tick(self):
        if self._scheduler_enabled and self._is_scheduler_window_open():
            self._start_next_downloads()

    def on_settings_backup(self):
        default_name = f"anigui_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Crea backup", "Create backup"),
            os.path.join(self.download_dir, default_name),
            self._tr("File zip (*.zip)", "Zip files (*.zip)"),
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".zip"):
            out_path += ".zip"
        try:
            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                base = app_state_dir()
                files = [
                    ("history.json", HISTORY_PATH),
                    ("search_history.json", SEARCH_HISTORY_PATH),
                    ("offline_covers.json", OFFLINE_COVERS_MAP_PATH),
                    ("settings.json", SETTINGS_PATH),
                    ("favorites.json", FAVORITES_PATH),
                    ("metadata_cache.json", METADATA_CACHE_PATH),
                ]
                for arc, src in files:
                    if os.path.exists(src):
                        zf.write(src, arcname=arc)
                covers_dir = os.path.join(base, "covers")
                if os.path.isdir(covers_dir):
                    for fn in os.listdir(covers_dir):
                        src = os.path.join(covers_dir, fn)
                        if os.path.isfile(src):
                            zf.write(src, arcname=os.path.join("covers", fn))
            self.set_status(f"Backup creato: {out_path}")
        except Exception as ex:
            self.notify_err(f"Backup fallito: {ex}")

    def on_settings_restore(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: restore bloccato.")
            return
        in_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Ripristina backup", "Restore backup"),
            self.download_dir,
            self._tr("File zip (*.zip)", "Zip files (*.zip)"),
        )
        if not in_path:
            return
        try:
            with zipfile.ZipFile(in_path, "r") as zf:
                members = set(zf.namelist())
                base = app_state_dir()
                mapping = {
                    "history.json": HISTORY_PATH,
                    "search_history.json": SEARCH_HISTORY_PATH,
                    "offline_covers.json": OFFLINE_COVERS_MAP_PATH,
                    "settings.json": SETTINGS_PATH,
                    "favorites.json": FAVORITES_PATH,
                    "metadata_cache.json": METADATA_CACHE_PATH,
                }
                for arc, dst in mapping.items():
                    if arc in members:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with zf.open(arc, "r") as src, open(dst, "wb") as out:
                            out.write(src.read())
                for m in members:
                    if not m.startswith("covers/") or m.endswith("/"):
                        continue
                    rel = m[len("covers/"):]
                    dst = os.path.join(base, "covers", os.path.basename(rel))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with zf.open(m, "r") as src, open(dst, "wb") as out:
                        out.write(src.read())

            # reload in-memory data from restored files
            self.history.load()
            self.refresh_history_ui()
            self._settings = self._load_settings()
            self._apply_runtime_from_settings()
            self._load_search_history()
            self._load_offline_covers_map()
            self._load_favorites()
            self._load_metadata_cache()
            self.refresh_favorites_ui()
            self._sync_settings_ui_from_state()
            self.refresh_offline_library()
            self.set_status("Restore completato.")
        except Exception as ex:
            self.notify_err(f"Restore fallito: {ex}")

    @staticmethod
    def _version_key(v: str) -> tuple[int, ...]:
        nums = [int(x) for x in re.findall(r"\d+", str(v or "").strip())]
        if not nums:
            return (0,)
        return tuple(nums)

    @classmethod
    def _is_version_newer(cls, remote: str, local: str) -> bool:
        a = list(cls._version_key(remote))
        b = list(cls._version_key(local))
        n = max(len(a), len(b))
        a.extend([0] * (n - len(a)))
        b.extend([0] * (n - len(b)))
        return tuple(a) > tuple(b)

    def _update_fetch_manifest_worker(self) -> dict[str, Any]:
        req = urllib.request.Request(
            UPDATE_MANIFEST_URL,
            headers={"User-Agent": "AniPyApp/1.0"},
        )
        with urllib.request.urlopen(req, timeout=12.0) as r:
            raw = r.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Manifest update non valido.")
        latest = str(data.get("version", "")).strip()
        if not latest:
            raise ValueError("Manifest update senza campo 'version'.")

        is_win = platform.system() == "Windows"
        plat = "windows" if is_win else "linux"
        sect = data.get(plat, {})
        if not isinstance(sect, dict):
            sect = {}

        url = str(
            sect.get("url")
            or data.get(f"{plat}_url")
            or data.get("url")
            or ""
        ).strip()
        sha256 = str(
            sect.get("sha256")
            or data.get(f"{plat}_sha256")
            or data.get("sha256")
            or ""
        ).strip().lower()
        page_url = str(
            data.get("page_url")
            or data.get("release_url")
            or url
            or ""
        ).strip()
        notes = str(data.get("notes") or "").strip()
        available = self._is_version_newer(latest, APP_VERSION)
        return {
            "current_version": APP_VERSION,
            "latest_version": latest,
            "available": bool(available),
            "url": url,
            "sha256": sha256,
            "page_url": page_url,
            "notes": notes,
        }

    def _start_update_check(self, silent: bool):
        self.btn_settings_update_check.setEnabled(False)
        w = Worker(self._update_fetch_manifest_worker)
        w.ok.connect(lambda info, sl=silent: self._on_update_check_ok(info, sl))
        w.err.connect(lambda msg, sl=silent: self._on_update_check_err(msg, sl))
        self._workers.append(w)
        w.start()

    def check_updates_silent(self):
        self._start_update_check(True)

    def on_update_check_clicked(self):
        self.set_status(self._tr("Controllo aggiornamenti in corso...", "Checking for updates..."))
        self._start_update_check(False)

    def _on_update_check_ok(self, info: dict[str, Any], silent: bool):
        self.btn_settings_update_check.setEnabled(True)
        self._update_info = info if isinstance(info, dict) else None
        available = bool((info or {}).get("available"))
        latest = str((info or {}).get("latest_version", APP_VERSION))
        if not available:
            self.btn_settings_update_apply.setEnabled(False)
            self._update_download_path = None
            if not silent:
                QMessageBox.information(
                    self,
                    self._tr("Aggiornamenti", "Updates"),
                    self._tr(
                        f"Sei gia all'ultima versione ({APP_VERSION}).",
                        f"You are already on the latest version ({APP_VERSION}).",
                    ),
                )
                self.set_status(self._tr("Nessun aggiornamento disponibile.", "No updates available."))
            return

        can_apply = bool((info or {}).get("url") or (info or {}).get("page_url"))
        self.btn_settings_update_apply.setEnabled(can_apply)
        self._update_download_path = None
        self.set_status(
            self._tr(
                f"Aggiornamento disponibile: {APP_VERSION} -> {latest}",
                f"Update available: {APP_VERSION} -> {latest}",
            )
        )
        if not silent:
            QMessageBox.information(
                self,
                self._tr("Aggiornamenti", "Updates"),
                self._tr(
                    f"Nuova versione disponibile: {latest}",
                    f"New version available: {latest}",
                ),
            )

    def _on_update_check_err(self, msg: str, silent: bool):
        self.btn_settings_update_check.setEnabled(True)
        self.btn_settings_update_apply.setEnabled(False)
        if not silent:
            self.notify_err(self._tr(f"Check update fallito: {msg}", f"Update check failed: {msg}"))
            self.set_status(self._tr("Check aggiornamenti fallito.", "Update check failed."))

    def _download_update_worker(self, info: dict[str, Any]) -> str:
        url = str(info.get("url") or "").strip()
        if not url:
            raise ValueError("URL update mancante nel manifest.")
        expected_sha = str(info.get("sha256") or "").strip().lower()
        req = urllib.request.Request(url, headers={"User-Agent": "AniPyApp/1.0"})
        suffix = ".exe" if platform.system() == "Windows" else ".bin"
        fd, tmp_path = tempfile.mkstemp(prefix="anigui_update_", suffix=suffix)
        os.close(fd)
        hasher = hashlib.sha256()
        try:
            with urllib.request.urlopen(req, timeout=20.0) as r, open(tmp_path, "wb") as f:
                while True:
                    chunk = r.read(512 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
            if expected_sha:
                actual = hasher.hexdigest().lower()
                if actual != expected_sha:
                    raise ValueError("Checksum update non valida (SHA256 mismatch).")
            return tmp_path
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def _open_update_page(self):
        info = self._update_info or {}
        page_url = str(info.get("page_url") or info.get("url") or "").strip()
        if not page_url:
            self.notify_err(self._tr("URL update non disponibile.", "Update URL not available."))
            return
        QDesktopServices.openUrl(QUrl(page_url))

    def on_update_apply_clicked(self):
        info = self._update_info or {}
        if not bool(info.get("available")):
            self.notify_err(self._tr("Nessun aggiornamento disponibile.", "No updates available."))
            return

        is_win_frozen = platform.system() == "Windows" and bool(getattr(sys, "frozen", False))
        if not is_win_frozen:
            self._open_update_page()
            self.set_status(self._tr("Aperta pagina aggiornamento.", "Update page opened."))
            return

        if self._update_download_path and os.path.exists(self._update_download_path):
            self._apply_windows_update_and_restart(self._update_download_path)
            return

        self.btn_settings_update_apply.setEnabled(False)
        self.set_status(self._tr("Download aggiornamento in corso...", "Downloading update..."))
        w = Worker(self._download_update_worker, dict(info))
        w.ok.connect(self._on_update_download_ok)
        w.err.connect(self._on_update_download_err)
        self._workers.append(w)
        w.start()

    def _on_update_download_ok(self, path: str):
        self._update_download_path = str(path or "").strip()
        if not self._update_download_path:
            self._on_update_download_err("download-path-empty")
            return
        self._apply_windows_update_and_restart(self._update_download_path)

    def _on_update_download_err(self, msg: str):
        self.btn_settings_update_apply.setEnabled(True)
        self.notify_err(self._tr(f"Download update fallito: {msg}", f"Update download failed: {msg}"))
        self.set_status(self._tr("Download aggiornamento fallito.", "Update download failed."))

    def _apply_windows_update_and_restart(self, downloaded_exe: str):
        if not (platform.system() == "Windows" and bool(getattr(sys, "frozen", False))):
            self._open_update_page()
            return
        current_exe = os.path.abspath(sys.executable)
        src = os.path.abspath(downloaded_exe)
        if not os.path.exists(src):
            self.notify_err(self._tr("File update mancante.", "Update file missing."))
            return
        if not src.lower().endswith(".exe"):
            self.notify_err(self._tr("Update non valido: file .exe atteso.", "Invalid update: .exe file expected."))
            return
        cur_pid = os.getpid()
        bat_path = os.path.join(tempfile.gettempdir(), f"anigui_updater_{int(time.time())}.bat")
        bat = (
            "@echo off\n"
            "setlocal\n"
            f"set \"SRC={src}\"\n"
            f"set \"DST={current_exe}\"\n"
            f"set \"PID={cur_pid}\"\n"
            ":waitclose\n"
            "tasklist /FI \"PID eq %PID%\" | findstr /I \"%PID%\" > nul\n"
            "if not errorlevel 1 (\n"
            "  timeout /t 1 /nobreak > nul\n"
            "  goto waitclose\n"
            ")\n"
            "copy /Y \"%SRC%\" \"%DST%\" > nul\n"
            "if errorlevel 1 (\n"
            "  copy /Y \"%SRC%\" \"%DST%\" > nul\n"
            ")\n"
            "if errorlevel 1 (\n"
            "  start \"\" \"%SRC%\"\n"
            ") else (\n"
            "  start \"\" \"%DST%\"\n"
            ")\n"
            "del \"%SRC%\" > nul 2>&1\n"
            "del \"%~f0\" > nul 2>&1\n"
        )
        with open(bat_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(bat)
        self.set_status(self._tr("Riavvio per applicare update...", "Restarting to apply update..."))
        try:
            subprocess.Popen(["cmd", "/c", bat_path], cwd=os.path.dirname(current_exe))
        except Exception as ex:
            self.notify_err(self._tr(f"Avvio updater fallito: {ex}", f"Updater launch failed: {ex}"))
            return
        QApplication.instance().quit()

    def _load_favorites(self):
        try:
            with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                self._favorite_items = [FavoriteEntry(**x) for x in raw if isinstance(x, dict)]
            else:
                self._favorite_items = []
        except Exception:
            self._favorite_items = []

    def _save_favorites(self):
        if self._incognito_enabled:
            return
        tmp = FAVORITES_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([asdict(x) for x in self._favorite_items], f, ensure_ascii=False, indent=2)
            os.replace(tmp, FAVORITES_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _favorite_exists(self, source: str, identifier: str) -> bool:
        for x in self._favorite_items:
            if x.source == source and x.identifier == identifier:
                return True
        return False

    def _add_favorite_from_item(self, item: SearchItem):
        if self._favorite_exists(item.source, item.identifier):
            return
        self._favorite_items.append(
            FavoriteEntry(
                name=item.name,
                identifier=item.identifier,
                source=item.source,
                cover_url=item.cover_url,
                added_at=time.time(),
            )
        )
        self._save_favorites()
        self.refresh_favorites_ui()

    def refresh_favorites_ui(self):
        self.favorites_list.clear()
        if not self._favorite_items:
            self.favorites_list.addItem(self._tr("(nessun preferito)", "(no favorites)"))
            self.favorites_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return
        ordered = sorted(self._favorite_items, key=lambda x: x.added_at, reverse=True)
        self._favorite_items = ordered
        placeholder = self._make_cover_placeholder()
        for row, fav in enumerate(self._favorite_items):
            it = QListWidgetItem(placeholder, fav.name)
            it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            it.setSizeHint(QSize(170, 290))
            self.favorites_list.addItem(it)
            if fav.cover_url:
                w = Worker(self.do_fetch_cover, fav.cover_url)
                w.ok.connect(lambda data, r=row: self._on_favorite_cover_loaded(r, data[0], data[1]))
                w.err.connect(lambda _msg: None)
                self._workers.append(w)
                w.start()

    def _on_favorite_cover_loaded(self, row: int, data: bytes | None, from_cache: bool):
        if data is None or row < 0 or row >= self.favorites_list.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.favorites_list.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.favorites_list.item(row).setIcon(QIcon(scaled))

    def on_favorite_add_current(self):
        if self.selected_search_item is not None:
            self._add_favorite_from_item(self.selected_search_item)
            self.set_status("Aggiunto ai preferiti.")
            return
        if self.selected_anime is not None:
            ident = str(
                getattr(self.selected_anime, "identifier", None)
                or getattr(self.selected_anime, "_identifier", None)
                or getattr(self.selected_anime, "ref", "")
            )
            item = SearchItem(
                name=getattr(self.selected_anime, "name", "anime"),
                identifier=ident,
                languages={LanguageTypeEnum.SUB, LanguageTypeEnum.DUB},
                cover_url=getattr(self.selected_search_item, "cover_url", None) if self.selected_search_item else None,
                source=self.provider_name,
            )
            self._add_favorite_from_item(item)
            self.set_status("Aggiunto ai preferiti.")

    def on_favorite_remove_selected(self):
        row = self.favorites_list.currentRow()
        if row < 0 or row >= len(self._favorite_items):
            return
        self._favorite_items.pop(row)
        self._save_favorites()
        self.refresh_favorites_ui()

    def on_pick_favorite(self, *_args):
        row = self.favorites_list.currentRow()
        if row < 0 or row >= len(self._favorite_items):
            return
        fav = self._favorite_items[row]
        item = SearchItem(
            name=fav.name,
            identifier=fav.identifier,
            languages={LanguageTypeEnum.SUB, LanguageTypeEnum.DUB},
            cover_url=fav.cover_url,
            source=fav.source,
        )
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.fetch_episodes()

    def _metadata_cache_ttl_seconds(self) -> float:
        return 7 * 24 * 3600.0

    def _metadata_key(self, source: str, identifier: str) -> str:
        return f"{source}:{identifier}"

    def _load_metadata_cache(self):
        try:
            with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._metadata_cache = raw if isinstance(raw, dict) else {}
        except Exception:
            self._metadata_cache = {}

    def _save_metadata_cache(self):
        if self._incognito_enabled:
            return
        tmp = METADATA_CACHE_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._metadata_cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp, METADATA_CACHE_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _apply_metadata_cache_to_item(self, it: SearchItem):
        key = self._metadata_key(it.source, it.identifier)
        obj = self._metadata_cache.get(key)
        if not isinstance(obj, dict):
            return
        ts = float(obj.get("updated_at", 0.0) or 0.0)
        if (time.time() - ts) > self._metadata_cache_ttl_seconds():
            return
        if it.year is None:
            it.year = obj.get("year")
        if not it.season:
            it.season = obj.get("season")
        if not it.studio:
            it.studio = obj.get("studio")
        if it.rating is None:
            it.rating = obj.get("rating")

    def _store_metadata_cache_for_item(self, it: SearchItem):
        if self._incognito_enabled:
            return
        key = self._metadata_key(it.source, it.identifier)
        self._metadata_cache[key] = {
            "updated_at": time.time(),
            "year": it.year,
            "season": it.season,
            "studio": it.studio,
            "rating": it.rating,
        }
        self._save_metadata_cache()

    def _load_search_history(self):
        try:
            with open(SEARCH_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._recent_queries = [str(x) for x in data if str(x).strip()]
        except FileNotFoundError:
            self._recent_queries = []
        except Exception:
            self._recent_queries = []

    def _save_search_history(self):
        if self._incognito_enabled:
            return
        tmp = SEARCH_HISTORY_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._recent_queries, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SEARCH_HISTORY_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _load_offline_covers_map(self):
        try:
            with open(OFFLINE_COVERS_MAP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                loaded = {
                    str(k): str(v)
                    for k, v in data.items()
                    if str(k).strip() and str(v).strip()
                }
                changed = False
                for k, v in list(loaded.items()):
                    # migrate legacy URL entries to local *.img cache path
                    if not v.lower().endswith(".img"):
                        p = self._cover_cache_path(v)
                        if p and os.path.exists(p):
                            loaded[k] = p
                            changed = True
                self._offline_covers_map = loaded
                if changed:
                    self._save_offline_covers_map()
            else:
                self._offline_covers_map = {}
                self._save_offline_covers_map()
        except FileNotFoundError:
            self._offline_covers_map = {}
            self._save_offline_covers_map()
        except Exception:
            self._offline_covers_map = {}
            self._save_offline_covers_map()

    def _save_offline_covers_map(self):
        if self._incognito_enabled:
            return
        tmp = OFFLINE_COVERS_MAP_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._offline_covers_map, f, ensure_ascii=False, indent=2)
            os.replace(tmp, OFFLINE_COVERS_MAP_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _remember_offline_cover_url(self, anime_name: str, cover_url: str | None):
        if self._incognito_enabled:
            return
        if not cover_url:
            return
        cache_path = self._cover_cache_path(cover_url)
        if not cache_path or not os.path.exists(cache_path):
            return
        key = self._safe_name(anime_name).lower()
        if not key:
            return
        if self._offline_covers_map.get(key) == cache_path:
            return
        self._offline_covers_map[key] = cache_path
        self._save_offline_covers_map()

    def _add_recent_query(self, q: str):
        q = q.strip()
        if not q:
            return
        self._recent_queries = [x for x in self._recent_queries if x.lower() != q.lower()]
        self._recent_queries.insert(0, q)
        if not self._incognito_enabled:
            self._save_search_history()
        self._refresh_recent_queries_ui()

    def on_clear_search_history(self):
        self._recent_queries = []
        if not self._incognito_enabled:
            try:
                if os.path.exists(SEARCH_HISTORY_PATH):
                    os.remove(SEARCH_HISTORY_PATH)
            except Exception:
                pass
        self._refresh_recent_queries_ui()

    def _refresh_recent_queries_ui(self):
        self.list_recent_queries.clear()
        if not self._recent_queries:
            self.list_recent_queries.addItem(self._tr("(nessuna ricerca recente)", "(no recent searches)"))
            self.list_recent_queries.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return
        for q in self._recent_queries:
            self.list_recent_queries.addItem(q)

    def _update_suggestions_visibility(self):
        empty = not self.search_input.text().strip()
        self.suggestions_panel.setVisible(self._current_search_view == "catalog" and empty)

    def _set_search_results_area_visible(self, visible: bool):
        for w in getattr(self, "_search_filter_widgets", []):
            w.setVisible(visible)
        self.results.setVisible(visible)

    def _refresh_search_layout(self):
        self._update_suggestions_visibility()
        has_query = len(self.search_input.text().strip()) >= 3
        in_catalog = self._current_search_view == "catalog"
        self._set_search_results_area_visible(in_catalog and has_query)
        self.search_stack.setVisible((not in_catalog) or has_query)

    def _apply_netflix_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0d1418, stop:1 #0b0f12);
                color: #e9edf1;
                font-family: "Fira Sans", "IBM Plex Sans", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QListWidget, QGroupBox {
                background-color: #111a1f;
                border: 1px solid #1f2a33;
                border-radius: 10px;
                padding: 7px 10px;
                selection-background-color: #12b5a5;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #12b5a5;
            }
            QPushButton {
                background-color: #162129;
                border: 1px solid #22303a;
                border-radius: 10px;
                padding: 7px 12px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #1c2a33; }
            QPushButton:pressed { background-color: #10181e; }
            QPushButton:disabled {
                background-color: #101820;
                color: #70818f;
                border-color: #1a252e;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #0f171c;
                border: 1px solid #1f2a33;
                selection-background-color: #12313c;
            }
            QTabWidget::pane { border: 1px solid #1f2a33; border-radius: 12px; }
            QTabBar::tab {
                background: #0f171c;
                border: 1px solid #1f2a33;
                padding: 8px 14px;
                margin-right: 6px;
                border-radius: 10px;
            }
            QTabBar::tab:selected {
                background: #12b5a5;
                color: #0b0f12;
                border-color: #12b5a5;
                font-weight: 600;
            }
            QWidget#suggestionsPanel {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 8px;
            }
            QLabel#sectionTitle { font-size: 15px; font-weight: 600; color: #e9edf1; }
            QLabel#mutedLabel { color: #9fb0bb; }
            QListWidget#resultsList::item {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 6px;
                margin: 6px;
            }
            QListWidget#resultsList::item:selected {
                border: 1px solid #12b5a5;
                background: #132029;
            }
            QListWidget#resultsList::item:hover {
                border: 1px solid #2f4957;
                background: #14212a;
            }
            QListWidget#continueList {
                background: #061227;
                border: 1px solid #17314d;
                border-radius: 12px;
                padding: 6px;
            }
            QListWidget#continueList::item {
                border: none;
                margin: 3px 2px;
                padding: 0;
            }
            QWidget#continueCard {
                background: #0c1d33;
                border: 1px solid #17314d;
                border-radius: 8px;
            }
            QWidget#continueCard:hover {
                border: 1px solid #2d5f98;
                background: #102642;
            }
            QLabel#continueTitle {
                color: #b9d8ff;
                font-size: 14px;
                font-weight: 500;
            }
            QLabel#continueSub {
                color: #a6bfdc;
                font-size: 13px;
            }
            QLabel#continueMeta {
                color: #7ea8d5;
                font-size: 12px;
            }
            QLabel#continueTime {
                color: #7fa3cf;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#continueIcons {
                color: #9db5d1;
                font-size: 13px;
            }
            QWidget#historySectionRow {
                background: transparent;
            }
            QFrame#historySectionLine {
                background: #1a3553;
                border: none;
                min-height: 1px;
                max-height: 1px;
            }
            QLabel#historySectionPill {
                border-radius: 11px;
                padding: 4px 12px;
                border: 1px solid #2e5f8f;
                background: #0f2947;
                color: #b9d8ff;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#historySectionPill[status="watching"] {
                border-color: #2f8e88;
                background: #113737;
                color: #8ff2d9;
            }
            QLabel#historySectionPill[status="planned"] {
                border-color: #8a7440;
                background: #3a301a;
                color: #f4d27a;
            }
            QLabel#historySectionPill[status="completed"] {
                border-color: #4d7aa3;
                background: #1b3046;
                color: #9fceff;
            }
            QListWidget#episodesList::item {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 10px;
                margin: 2px 4px;
            }
            QListWidget#episodesList::item:selected {
                border: 1px solid #12b5a5;
                background: #132029;
            }
            QListWidget#suggestionsList::item {
                background: #101a20;
                border: 1px solid #1f2a33;
                border-radius: 10px;
                padding: 6px;
                margin: 3px;
            }
            QLabel#recentHeader {
                color: #3ddc84;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#recommendedHeader {
                color: #4db8ff;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#incognitoBadge {
                background: #c62026;
                color: #ffffff;
                border: 1px solid #ff5a5f;
                border-radius: 10px;
                font-weight: 700;
                padding: 4px 10px;
            }
            QLabel#statusBar {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 8px;
                padding: 6px 10px;
                color: #c0d0db;
            }
            QScrollBar:vertical {
                background: #0b0f12;
                width: 10px;
                margin: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1f2a33;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #2b3b46; }
            """
        )
        self.results.setObjectName("resultsList")
        self.episodes.setObjectName("episodesList")

    def do_fetch_cover(self, url: str) -> tuple[bytes | None, bool]:
        cache_path = self._cover_cache_path(url)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = f.read()
                if data:
                    return data, True
            except Exception:
                pass
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=8.0) as r:
            data = r.read()
        if data:
            try:
                scaled = self._downscale_cover_bytes(data)
                if scaled:
                    data = scaled
            except Exception:
                pass
        if data and cache_path and not self._incognito_enabled:
            tmp = cache_path + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, cache_path)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        return (data if data else None), False

    def _downscale_cover_bytes(self, data: bytes) -> bytes | None:
        img = QImage.fromData(data)
        if img.isNull():
            return None
        new_w = max(1, int(img.width() * 0.7))
        new_h = max(1, int(img.height() * 0.7))
        if new_w == img.width() and new_h == img.height():
            return None
        scaled = img.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        buf = QBuffer()
        arr = QByteArray()
        buf.setBuffer(arr)
        if not buf.open(QBuffer.OpenModeFlag.WriteOnly):
            return None
        fmt = b"PNG" if scaled.hasAlphaChannel() else b"JPG"
        ok = scaled.save(buf, fmt)
        buf.close()
        if not ok:
            return None
        return bytes(arr)

    def _cover_cache_path(self, url: str) -> str | None:
        if not url:
            return None
        h = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()
        return os.path.join(self.cover_cache_dir, f"{h}.img")

    def do_fetch_cover_for_item(self, item: SearchItem) -> tuple[bytes | None, bool]:
        url = item.cover_url
        if not url and self._is_aw_provider_name(item.source):
            url = self._aw_resolve_cover_url(item)
            if url:
                item.cover_url = url
        if not url:
            try:
                anime = self.build_anime_from_item(item)
                info = anime.get_info()
                url = getattr(info, "image", None)
                if isinstance(url, str) and url.strip():
                    item.cover_url = url.strip()
                    url = item.cover_url
            except Exception:
                return None, False
        if not url:
            return None, False
        return self.do_fetch_cover(url)

    def do_fetch_cover_for_history_entry(
        self,
        entry: HistoryEntry,
    ) -> tuple[bytes | None, bool, str | None]:
        url = entry.cover_url
        if not url:
            try:
                if self._is_aw_provider_name(entry.provider):
                    aw_item = SearchItem(
                        name=entry.name,
                        identifier=entry.identifier,
                        languages=set(),
                        cover_url=None,
                        source=entry.provider,
                    )
                    url = self._aw_resolve_cover_url(aw_item)
                if not url:
                    anime = self.build_anime_from_history(entry)
                    info = anime.get_info()
                    cand = self._extract_cover_url(info)
                    if isinstance(cand, str) and cand.strip():
                        url = cand.strip()
            except Exception:
                url = None
        if not url:
            return None, False, None
        data, from_cache = self.do_fetch_cover(url)
        return data, from_cache, url

    def _on_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._search_nonce:
            return
        if data is None:
            if 0 <= row < self.results.count():
                self.results.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.results.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.results.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._result_items):
            it = self._result_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.results.item(row).setIcon(QIcon(scaled))
        self.results.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    def _on_recommended_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._recommended_nonce:
            return
        if data is None:
            if 0 <= row < self.recommended.count():
                self.recommended.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.recommended.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.recommended.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._recommended_items):
            it = self._recommended_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.recommended.item(row).setIcon(QIcon(scaled))
        self.recommended.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    def _on_recent_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._recent_nonce:
            return
        if data is None:
            if 0 <= row < self.recent.count():
                self.recent.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.recent.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.recent.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._recent_items):
            it = self._recent_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.recent.item(row).setIcon(QIcon(scaled))
        self.recent.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    # ---------------- anipy operations ----------------
    def do_search(self, query: str) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            res = provider.search(query) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                if self.lang == LanguageTypeEnum.DUB and not is_dub:
                    continue
                if self.lang == LanguageTypeEnum.SUB and is_dub:
                    continue
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=self.provider_name,
                        raw=r,
                    )
                )
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        res = provider.get_search(query)
        out = [
            SearchItem(
                r.name,
                r.identifier,
                r.languages,
                cover_url=self._extract_cover_url(r),
                source=self.provider_name,
                raw=r,
            )
            for r in res
        ]
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_recent_releases(self) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            filter_code = "d" if self.lang == LanguageTypeEnum.DUB else "s"
            res = provider.latest(filter_code) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=self.provider_name,
                        raw=r,
                    )
                )
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        seen: set[str] = set()
        out: list[SearchItem] = []
        now = time.localtime()
        year = now.tm_year
        month = now.tm_mon
        season = "winter" if month <= 3 else "spring" if month <= 6 else "summer" if month <= 9 else "fall"
        queries = [
            f"{season} {year}",
            f"{year}",
            "ongoing",
            "new",
        ]

        for q in queries:
            try:
                res = provider.get_search(q)
            except Exception:
                continue
            for r in res:
                ident = getattr(r, "identifier", None)
                if not ident or ident in seen:
                    continue
                seen.add(ident)
                out.append(
                    SearchItem(
                        r.name,
                        r.identifier,
                        r.languages,
                        cover_url=self._extract_cover_url(r),
                        source=self.provider_name,
                        raw=r,
                    )
                )
                if len(out) >= 60:
                    for it in out:
                        self._apply_metadata_cache_to_item(it)
                    return out
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_recommended(self) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            seen: set[str] = set()
            out: list[SearchItem] = []
            for q in ("popular", "top", "best", "shonen"):
                res = provider.search(q) or []
                for r in res:
                    ident = str(getattr(r, "ref", ""))
                    if not ident or ident in seen:
                        continue
                    seen.add(ident)
                    is_dub = bool(getattr(r, "dub", False))
                    if self.lang == LanguageTypeEnum.DUB and not is_dub:
                        continue
                    if self.lang == LanguageTypeEnum.SUB and is_dub:
                        continue
                    out.append(
                        SearchItem(
                            name=r.name,
                            identifier=ident,
                            languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                            cover_url=None,
                            source=self.provider_name,
                            raw=r,
                        )
                    )
                    if len(out) >= 60:
                        for it in out:
                            self._apply_metadata_cache_to_item(it)
                        return out
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        seen: set[str] = set()
        out: list[SearchItem] = []
        queries = ["popular", "top", "best", "trending"]

        for q in queries:
            try:
                res = provider.get_search(q)
            except Exception:
                continue
            for r in res:
                ident = getattr(r, "identifier", None)
                if not ident or ident in seen:
                    continue
                seen.add(ident)
                out.append(
                    SearchItem(
                        r.name,
                        r.identifier,
                        r.languages,
                        cover_url=self._extract_cover_url(r),
                        source=self.provider_name,
                        raw=r,
                    )
                )
                if len(out) >= 60:
                    for it in out:
                        self._apply_metadata_cache_to_item(it)
                    return out
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_episodes(self, anime: Any) -> list[float | int]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            provider.episodes(anime)
            out: list[float | int] = []
            for ep in anime.episodes():
                try:
                    f = float(ep)
                    out.append(int(f) if f.is_integer() else f)
                except Exception:
                    continue
            return out
        return anime.get_episodes(lang=self.lang)

    def build_anime_from_item(self, item: SearchItem) -> Any:
        if self._is_aw_provider_name(item.source):
            if item.raw is not None:
                return item.raw
            anime_cls = self._aw_anime_class()
            anime = anime_cls(item.name, item.identifier)
            try:
                self._aw_provider(item.source).info_anime(anime)
            except Exception:
                pass
            return anime
        provider = get_provider(item.source)
        return Anime(provider, item.name, item.identifier, item.languages)

    def build_anime_from_history(self, entry: HistoryEntry) -> Any:
        if self._is_aw_provider_name(entry.provider):
            anime_cls = self._aw_anime_class()
            anime = anime_cls(entry.name, entry.identifier)
            try:
                self._aw_provider(entry.provider).info_anime(anime)
            except Exception:
                pass
            return anime
        provider = get_provider(entry.provider)
        languages = {LanguageTypeEnum.SUB, LanguageTypeEnum.DUB}
        return Anime(provider, entry.name, entry.identifier, languages)

    def _current_history_entry(self) -> HistoryEntry | None:
        if not self.selected_anime:
            return None
        ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        for e in self.history._data:
            if e.provider == self.provider_name and e.identifier == ident and e.lang == lang_s:
                return e
        return None

    def _episodes_list_for_history_entry(self, entry: HistoryEntry) -> list[float | int] | None:
        if not self.episodes_list:
            return None
        if self.provider_name != entry.provider:
            return None
        cur_lang = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        if cur_lang != entry.lang:
            return None
        if not self.selected_anime:
            return None
        cur_ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        if cur_ident != str(entry.identifier):
            return None
        return self.episodes_list

    def _history_context_for_selected_anime(
        self,
    ) -> tuple[set[float] | None, dict[float, dict[str, Any]] | None]:
        entry = self._current_history_entry()
        if entry is None:
            return None, None
        return self._entry_watched_eps_set(entry), self._entry_episode_progress_map(entry)

    def _load_items_with_covers(
        self,
        items: list[SearchItem],
        list_widget: QListWidget,
        nonce: int,
        on_cover: Callable[[int, int, bytes | None, bool], None],
    ):
        list_widget.clear()
        for row, it in enumerate(items):
            item = QListWidgetItem(self._make_cover_placeholder(it.name), it.name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            item.setSizeHint(QSize(170, 290))
            item.setData(Qt.ItemDataRole.UserRole, "loading")
            list_widget.addItem(item)

            w = Worker(self.do_fetch_cover_for_item, it)
            w.ok.connect(lambda data, r=row, n=nonce: on_cover(r, n, data[0], data[1]))
            w.err.connect(lambda _msg: None)
            self._workers.append(w)
            w.start()

        self._start_skeletons()

    # ---------------- topbar handlers ----------------
    def _apply_provider_ui_state(self):
        is_aw = self.provider_name.startswith("aw_")
        self.quality_combo.setEnabled(not is_aw)
        if is_aw:
            self.quality_combo.setToolTip("Qualita stream non disponibile su provider ITA (aw-cli).")
        else:
            self.quality_combo.setToolTip("")

    def _set_provider(self, name: str):
        self.provider_name = name
        idx = self.provider_combo.findData(name)
        if idx >= 0 and self.provider_combo.currentIndex() != idx:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(idx)
            self.provider_combo.blockSignals(False)
        self._apply_provider_ui_state()

    def on_provider_change(self):
        self.provider_name = self.provider_combo.currentData()
        self._apply_provider_ui_state()
        q = self.search_input.text().strip()
        if len(q) >= 3:
            self.on_search(query=q)

    def on_lang_change(self):
        self.lang = self.lang_combo.currentData()
        if self.selected_anime:
            self.fetch_episodes()

    def on_quality_change(self):
        self.quality = self.quality_combo.currentText()

    # ---------------- Search flow ----------------
    def on_pick_recent_query(self, item: QListWidgetItem):
        q = item.text().strip()
        if not q or q.startswith("("):
            return
        self.search_input.setText(q)
        self.on_search(query=q)

    def on_filters_changed(self, *_args):
        if not self._result_items_raw:
            return
        nonce = self._search_nonce
        items = self._apply_filters(self._result_items_raw, nonce)
        self._result_items = items
        self._render_search_results(items, nonce)

    def on_search_text_changed(self, text: str):
        q = text.strip()
        self._pending_search_query = q
        if not q:
            self._last_search_query = None
            if self._search_debounce_timer.isActive():
                self._search_debounce_timer.stop()
            self.results.clear()
            self._search_cards = []
            self._result_items = []
            self.lbl_search_meta.setVisible(False)
            self.lbl_search_state.setVisible(False)
            self.set_status("Pronto.")
            self._update_suggestions_visibility()
            self._refresh_search_layout()
            return
        if len(q) < 3:
            self.lbl_search_state.setText(self._tr("Scrittura in corso...", "Typing..."))
            self.lbl_search_state.setVisible(False)
            self.results.clear()
            self._search_cards = []
            self._result_items = []
            self.lbl_search_meta.setVisible(False)
            self._update_suggestions_visibility()
            self._refresh_search_layout()
            return
        self.lbl_search_state.setVisible(False)
        self._update_suggestions_visibility()
        self._refresh_search_layout()
        self._search_debounce_timer.start()

    def _run_debounced_search(self):
        if not self._pending_search_query:
            return
        if len(self._pending_search_query) < 3:
            return
        self.on_search(query=self._pending_search_query)

    def on_search(self, query: str | None = None):
        q = (query if query is not None else self.search_input.text()).strip()
        if not q:
            return
        if q == self._last_search_query:
            return
        self._last_search_query = q
        if self._search_debounce_timer.isActive():
            self._search_debounce_timer.stop()

        self._search_nonce += 1
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_catalog)
        self._current_search_view = "catalog"
        self._refresh_search_layout()
        self._save_current_progress(force=True)
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.set_status(self._tr("Ricerca in corso…", "Searching..."))
        self._begin_request()
        self.results.clear()
        self._search_cards = []
        self._result_items_raw = []
        self.lbl_search_meta.setText(self._tr(f"Risultati per: {q}", f"Results for: {q}"))
        self.lbl_search_meta.setVisible(True)
        self.lbl_search_state.setText(self._tr("Cercando…", "Searching..."))
        self.lbl_search_state.setVisible(True)
        self.episodes.clear()
        self.lbl_anime_title.setText(self._tr("Anime", "Anime"))
        self.lbl_player_title.setText(self._tr("Player", "Player"))
        self.selected_anime = None
        self.episodes_list = []
        self.current_ep = None

        nonce = self._search_nonce
        w = Worker(self.do_search, q)
        w.ok.connect(lambda items, n=nonce, q=q: self.on_search_results(n, q, items))
        w.err.connect(self.on_worker_error)
        self._workers.append(w)
        w.start()

    def on_refresh_recommended(self):
        self._recommended_loaded = True
        self._begin_request()
        self._recent_nonce += 1
        self._recommended_nonce += 1
        recent_nonce = self._recent_nonce
        nonce = self._recommended_nonce
        self.set_status("Carico recent + recommended…")
        self.recent.clear()
        self._recent_items = []
        self._recent_cards = []
        self.recommended.clear()
        self._recommended_items = []
        self._recommended_cards = []

        w_recent = Worker(self.do_recent_releases)
        w = Worker(self.do_recommended)

        def on_recent_ok(items: list[SearchItem]):
            if recent_nonce != self._recent_nonce:
                self._end_request()
                return
            self._recent_items = items
            self._load_items_with_covers(
                items,
                self.recent,
                recent_nonce,
                self._on_recent_cover_loaded,
            )
            self.set_status(
                f"Recent: {len(self._recent_items)} | Recommended: {len(self._recommended_items)}"
            )
            self._end_request()

        def on_ok(items: list[SearchItem]):
            if nonce != self._recommended_nonce:
                self._end_request()
                return
            self._recommended_items = items
            self._load_items_with_covers(
                items,
                self.recommended,
                nonce,
                self._on_recommended_cover_loaded,
            )
            self.set_status(
                f"Recent: {len(self._recent_items)} | Recommended: {len(self._recommended_items)}"
            )
            self._end_request()

        w_recent.ok.connect(on_recent_ok)
        w_recent.err.connect(self.on_worker_error)
        w.ok.connect(on_ok)
        w.err.connect(self.on_worker_error)
        self._workers.append(w_recent)
        self._workers.append(w)
        w_recent.start()
        w.start()

    def on_tab_changed(self, idx: int):
        if self.tabs.widget(idx) is self.tab_recommended and not self._recommended_loaded:
            self.on_refresh_recommended()
        if self.tabs.widget(idx) is self.tab_offline:
            self.refresh_offline_library()
        if self.tabs.widget(idx) is self.tab_favorites:
            self.refresh_favorites_ui()
        if self.tabs.widget(idx) is self.tab_history:
            self._save_current_progress(force=True)
            self.refresh_history_ui()
            QTimer.singleShot(0, self._relayout_history_cards)
            QTimer.singleShot(80, self._relayout_history_cards)

    def on_search_results(self, nonce: int, query: str, items: list[SearchItem]):
        self._end_request()
        if nonce != self._search_nonce:
            return
        current = self.search_input.text().strip()
        if current == query and len(query) >= 3:
            self._add_recent_query(query)
        self._result_items_raw = items
        items = self._apply_filters(items, nonce)
        self._result_items = items
        self._render_search_results(items, nonce)
        self.set_status(f"Trovati {len(items)} risultati.")

    def _render_search_results(self, items: list[SearchItem], nonce: int):
        self._search_cards = []
        if not items:
            self.lbl_search_state.setText(self._tr("Nessun risultato.", "No results."))
            self.lbl_search_state.setVisible(True)
        else:
            self.lbl_search_state.setVisible(False)
        self._load_items_with_covers(
            items,
            self.results,
            nonce,
            self._on_cover_loaded,
        )

    def _filters_active(self) -> bool:
        if self.combo_season.currentIndex() > 0:
            return True
        if self.combo_year.currentIndex() > 0:
            return True
        if self.input_studio.text().strip():
            return True
        if self.combo_rating.currentIndex() > 0:
            return True
        return False

    def _apply_filters(self, items: list[SearchItem], nonce: int) -> list[SearchItem]:
        items = list(items)
        allow_missing = False
        if self._filters_active():
            if self._items_need_enrichment(items):
                self.lbl_search_state.setText(self._tr("Filtraggio…", "Filtering..."))
                self.lbl_search_state.setVisible(True)
                self._start_enrich_items(items, nonce)
                allow_missing = True
            items = [it for it in items if self._item_matches_filters(it, allow_missing)]
        items = self._sort_items(items)
        return items

    def _sort_items(self, items: list[SearchItem]) -> list[SearchItem]:
        idx = self.combo_sort.currentIndex()
        if idx == 1:
            return sorted(items, key=lambda x: x.name.lower())
        if idx == 2:
            return sorted(items, key=lambda x: x.name.lower(), reverse=True)
        return items

    def _items_need_enrichment(self, items: list[SearchItem]) -> bool:
        if not self._filters_active():
            return False
        for it in items:
            if self._item_missing_required_fields(it):
                return True
        return False

    def _item_missing_required_fields(self, it: SearchItem) -> bool:
        if self.combo_season.currentIndex() > 0 and not it.season:
            return True
        if self.combo_year.currentIndex() > 0 and not it.year:
            return True
        if self.input_studio.text().strip() and not it.studio:
            return True
        if self.combo_rating.currentIndex() > 0 and it.rating is None:
            return True
        return False

    def _item_matches_filters(self, it: SearchItem, allow_missing: bool) -> bool:
        if self.combo_season.currentIndex() > 0:
            season = self.combo_season.currentText().lower()
            if not it.season:
                return allow_missing
            if it.season.lower() != season:
                return False
        if self.combo_year.currentIndex() > 0:
            try:
                year = int(self.combo_year.currentText())
            except Exception:
                year = None
            if year and it.year is None:
                return allow_missing
            if year and it.year != year:
                return False
        studio = self.input_studio.text().strip().lower()
        if studio:
            if not it.studio:
                return allow_missing
            if studio not in it.studio.lower():
                return False
        if self.combo_rating.currentIndex() > 0:
            threshold = {1: 9, 2: 8, 3: 7, 4: 6}.get(self.combo_rating.currentIndex(), 0)
            if it.rating is None:
                return allow_missing
            if it.rating < threshold:
                return False
        return True

    def _start_enrich_items(self, items: list[SearchItem], nonce: int):
        if self._filters_pending:
            return
        self._filters_pending = True
        self._enrich_nonce += 1
        enrich_nonce = self._enrich_nonce
        self._begin_request()
        w = Worker(self._enrich_items_worker, items)
        w.ok.connect(lambda enriched, n=enrich_nonce, sn=nonce: self._on_items_enriched(n, sn, enriched))
        w.err.connect(lambda _msg: None)
        self._workers.append(w)
        w.start()

    def _enrich_items_worker(self, items: list[SearchItem]) -> list[SearchItem]:
        out: list[SearchItem] = []
        for it in items:
            self._apply_metadata_cache_to_item(it)
            if not self._item_missing_required_fields(it):
                out.append(it)
                continue
            try:
                anime = self.build_anime_from_item(it)
                info = anime.get_info()
                it.year = getattr(info, "year", None) or getattr(info, "release_year", None)
                it.season = getattr(info, "season", None)
                it.studio = getattr(info, "studio", None) or getattr(info, "studios", None)
                it.rating = getattr(info, "rating", None) or getattr(info, "score", None)
                if isinstance(it.studio, list):
                    it.studio = ", ".join([str(x) for x in it.studio if x])
                self._store_metadata_cache_for_item(it)
            except Exception:
                pass
            out.append(it)
        return out

    def _on_items_enriched(self, enrich_nonce: int, search_nonce: int, items: list[SearchItem]):
        self._end_request()
        self._filters_pending = False
        if enrich_nonce != self._enrich_nonce:
            return
        if search_nonce != self._search_nonce:
            return
        self._result_items_raw = items
        filtered = self._apply_filters(items, search_nonce)
        self._result_items = filtered
        self._render_search_results(filtered, search_nonce)

    def _on_enrich_error(self, _msg: str):
        self._end_request()
        self._filters_pending = False

    def on_pick_result(self, *_args):
        row = self.results.currentRow()
        if row < 0:
            return
        item = self._result_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_pick_recommended(self):
        row = self.recommended.currentRow()
        if row < 0 or row >= len(self._recommended_items):
            return
        item = self._recommended_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_pick_recent(self):
        row = self.recent.currentRow()
        if row < 0 or row >= len(self._recent_items):
            return
        item = self._recent_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def fetch_episodes(
        self,
        seen_eps: set[float] | None = None,
        episode_progress: dict[float, dict[str, Any]] | None = None,
    ):
        if not self.selected_anime:
            return
        if seen_eps is None or episode_progress is None:
            hist_seen, hist_prog = self._history_context_for_selected_anime()
            if seen_eps is None:
                seen_eps = hist_seen
            if episode_progress is None:
                episode_progress = hist_prog
        self._history_seen_eps_for_ui = seen_eps
        self._history_episode_progress_for_ui = episode_progress
        self.episodes.clear()
        w = Worker(self.do_episodes, self.selected_anime)
        w.ok.connect(self.on_episodes_ready)
        w.err.connect(self.on_worker_error)
        self._workers.append(w)
        w.start()

    def on_episodes_ready(self, eps: list[float | int]):
        self.episodes_list = eps
        hist_entry = self._current_history_entry()
        if hist_entry is not None and not self._incognito_enabled:
            new_completed = self._entry_is_series_completed(hist_entry, eps)
            if bool(hist_entry.completed) != bool(new_completed):
                hist_entry.completed = bool(new_completed)
                try:
                    self.history.upsert(hist_entry)
                except Exception:
                    pass
        self.episodes.clear()
        play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        seen_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        resume_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        seen_eps = self._history_seen_eps_for_ui
        ep_prog = self._history_episode_progress_for_ui or {}
        for ep in eps:
            icon = play_icon
            label = self._episode_label(ep)
            ef = self._to_ep_float(ep)
            p = None if ef is None else ep_prog.get(ef)
            if seen_eps is not None:
                if ef is not None and any(abs(ef - s) < 1e-6 for s in seen_eps):
                    icon = seen_icon
                    label = f"{label}  ✓ seen"
                elif isinstance(p, dict):
                    pos = float(p.get("pos", 0.0) or 0.0)
                    pct = float(p.get("percent", 0.0) or 0.0)
                    if pos > 0.0:
                        icon = resume_icon
                        label = f"{label}  ▶ {self.fmt_time(pos)}"
                    elif pct > 0.0:
                        icon = resume_icon
                        label = f"{label}  ▶ {int(max(0.0, min(100.0, pct)))}%"
            item = QListWidgetItem(icon, label)
            item.setToolTip(self._tr(f"Doppio click per riprodurre episodio {ep}", f"Double click to play episode {ep}"))
            item.setSizeHint(QSize(0, 44))
            self.episodes.addItem(item)
        if self.selected_anime:
            self.search_stack.setCurrentWidget(self.page_anime)
            self._current_search_view = "anime"
            self._refresh_search_layout()
        self.set_status(f"Episodi: {len(eps)}")
        self._history_seen_eps_for_ui = None
        self._history_episode_progress_for_ui = None

    def on_play_selected_episode(self, *_args):
        row = self.episodes.currentRow()
        if row < 0 or not self.selected_anime:
            return
        ep = self.episodes_list[row]
        hist = self._current_history_entry()
        if hist is not None:
            prog = self._entry_get_episode_progress(hist, ep)
            if isinstance(prog, dict):
                pos = float(prog.get("pos", 0.0) or 0.0)
                pct = float(prog.get("percent", 0.0) or 0.0)
                self._pending_resume_seek = None
                self._pending_resume_seek_ratio = None
                self._pending_resume_seek_attempts = 0
                if pos > 0.0:
                    self._pending_resume_seek = max(0.0, pos - 5.0)
                elif pct > 0.0:
                    self._pending_resume_seek_ratio = max(0.0, min(1.0, (pct / 100.0) - 0.01))
        self.play_episode(ep)

    # ---------------- History tab ----------------
    def refresh_history_ui(self):
        self._clear_history_section_overlays()
        self.hist_list.clear()
        self._history_cover_labels = {}
        all_items = self.history.list()
        self._history_items = []
        active_filter = str(getattr(self, "_history_filter", "All") or "All")
        ordered_rows: list[tuple[HistoryEntry, str, bool, int, dict[str, Any] | None]] = []
        marker_rank = {"Watching": 0, "Planned": 1, "Completed": 2}
        for e in all_items:
            marker, in_progress, seen_count, last_prog = self._history_entry_status(e)
            if active_filter != "All" and marker != active_filter:
                continue
            ordered_rows.append((e, marker, in_progress, seen_count, last_prog))

        ordered_rows.sort(
            key=lambda row: (
                marker_rank.get(row[1], 99),
                str(getattr(row[0], "name", "") or "").strip().casefold(),
            )
        )
        self._history_items = [row[0] for row in ordered_rows]
        prev_marker: str | None = None
        grid_slot = 0
        for e, marker, in_progress, seen_count, last_prog in ordered_rows:
            if active_filter == "All" and marker != prev_marker:
                # In 2-column mode, force each section header to start on a new row.
                if grid_slot % 2 != 0:
                    hpre = QListWidgetItem("")
                    hpre.setFlags(Qt.ItemFlag.NoItemFlags)
                    hpre.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header_prepad")
                    hpre.setSizeHint(QSize(420, 34))
                    self.hist_list.addItem(hpre)
                    grid_slot += 1
                section_text = {
                    "Watching": self._tr("In corso", "Watching"),
                    "Planned": self._tr("Da iniziare", "Planned"),
                    "Completed": self._tr("Completati", "Completed"),
                }.get(marker, marker)
                hitem = QListWidgetItem("")
                hitem.setFlags(Qt.ItemFlag.NoItemFlags)
                hitem.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header")
                hitem.setData(Qt.ItemDataRole.UserRole + 3, section_text)
                hitem.setData(Qt.ItemDataRole.UserRole + 4, marker)
                hitem.setSizeHint(QSize(420, 34))
                self.hist_list.addItem(hitem)
                grid_slot += 1
                hpad = QListWidgetItem("")
                hpad.setFlags(Qt.ItemFlag.NoItemFlags)
                hpad.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header_pad")
                hpad.setSizeHint(QSize(420, 34))
                self.hist_list.addItem(hpad)
                grid_slot += 1
                prev_marker = marker

            rel_time = self._fmt_relative_time(e.updated_at)
            marker_label = {
                "Completed": self._tr("Completato", "Completed"),
                "Watching": self._tr("In corso", "Watching"),
                "Planned": self._tr("Da iniziare", "Planned"),
            }.get(marker, marker)
            if marker == "Completed":
                title_line = f"{self._tr('Completato', 'Completed')} {e.name}"
                progress_line = f"EP {e.last_ep} · {e.lang}"
            elif in_progress:
                title_line = self._tr(f"In visione episodio {e.last_ep} di", f"Watching episode {e.last_ep} of")
                progress_line = e.name
            elif marker == "Planned":
                title_line = f"{self._tr('Da iniziare', 'Planned')} {e.name}"
                progress_line = f"EP {e.last_ep} · {e.lang}"
            else:
                title_line = self._tr(f"Visto episodio {e.last_ep} di", f"Watched episode {e.last_ep} of")
                progress_line = e.name

            detail_line = ""
            if in_progress:
                if isinstance(last_prog, dict):
                    pos_v = float(last_prog.get("pos", 0.0) or 0.0)
                    dur_v = float(last_prog.get("dur", 0.0) or 0.0)
                    pct_v = float(last_prog.get("percent", 0.0) or 0.0)
                else:
                    pos_v = float(e.last_pos or 0.0)
                    dur_v = float(e.last_duration or 0.0)
                    pct_v = float(getattr(e, "last_percent", 0.0) or 0.0)
                if dur_v > 0:
                    pct = int(max(0.0, min(1.0, pos_v / dur_v)) * 100)
                    detail_line = self._tr(
                        f"Riprendi {self.fmt_time(pos_v)} · {pct}%",
                        f"Resume {self.fmt_time(pos_v)} · {pct}%",
                    )
                elif pct_v > 0.0:
                    pct = int(max(0.0, min(100.0, pct_v)))
                    detail_line = self._tr(f"Riprendi {pct}%", f"Resume {pct}%")
                else:
                    detail_line = self._tr(f"Riprendi {self.fmt_time(pos_v)}", f"Resume {self.fmt_time(pos_v)}")
            elif marker == "Watching" and seen_count > 0:
                detail_line = self._tr(f"Episodi visti: {seen_count}", f"Seen episodes: {seen_count}")
            elif marker == "Planned":
                detail_line = self._tr("Non iniziato", "Not started")

            item = QListWidgetItem("")
            item.setSizeHint(QSize(420, 98))
            item.setData(Qt.ItemDataRole.UserRole, e)
            item.setData(Qt.ItemDataRole.UserRole + 1, "loading")
            tip = f"{e.name}\n{marker_label} · EP {e.last_ep} · {e.lang}\n{rel_time}"
            if detail_line:
                tip += f"\n{detail_line}"
            item.setToolTip(tip)
            self.hist_list.addItem(item)
            grid_slot += 1
            list_row = self.hist_list.count() - 1

            card = QWidget()
            card.setObjectName("continueCard")
            lay = QHBoxLayout(card)
            lay.setContentsMargins(10, 8, 10, 8)
            lay.setSpacing(10)

            cover = QLabel()
            cover.setFixedSize(52, 74)
            cover.setObjectName("continueCover")
            ph = self._make_cover_placeholder(e.name).pixmap(52, 74)
            cover.setPixmap(ph)
            cover.setScaledContents(True)
            lay.addWidget(cover, 0)

            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(2)

            lbl_title = QLabel(self._compact_title(title_line, max_len=46))
            lbl_title.setObjectName("continueTitle")
            lbl_title.setWordWrap(False)
            text_col.addWidget(lbl_title, 0)

            lbl_sub = QLabel(self._compact_title(progress_line, max_len=46))
            lbl_sub.setObjectName("continueSub")
            lbl_sub.setWordWrap(False)
            text_col.addWidget(lbl_sub, 0)

            if detail_line:
                lbl_detail = QLabel(detail_line)
                lbl_detail.setObjectName("continueMeta")
                text_col.addWidget(lbl_detail, 0)

            lay.addLayout(text_col, 1)

            right_col = QVBoxLayout()
            right_col.setContentsMargins(0, 0, 0, 0)
            right_col.setSpacing(6)

            lbl_time = QLabel(rel_time)
            lbl_time.setObjectName("continueTime")
            lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            right_col.addWidget(lbl_time, 0)

            right_col.addStretch(1)

            lbl_icons = QLabel("💬  ❤")
            lbl_icons.setObjectName("continueIcons")
            lbl_icons.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            right_col.addWidget(lbl_icons, 0)

            lay.addLayout(right_col, 0)
            self.hist_list.setItemWidget(item, card)
            self._history_cover_labels[list_row] = cover

            if e.cover_url:
                w_cover = Worker(self.do_fetch_cover, e.cover_url)
                w_cover.ok.connect(lambda data, r=list_row: self._on_history_cover_loaded(r, data[0], data[1]))
                w_cover.err.connect(lambda _msg: None)
                self._workers.append(w_cover)
                w_cover.start()
            else:
                w_cover = Worker(self.do_fetch_cover_for_history_entry, e)
                w_cover.ok.connect(
                    lambda data, r=list_row, h=e: self._on_history_cover_resolved(
                        r,
                        h,
                        data[0],
                        data[1],
                        data[2],
                    )
                )
                w_cover.err.connect(lambda _msg: None)
                self._workers.append(w_cover)
                w_cover.start()
        self._relayout_history_cards()
        QTimer.singleShot(0, self._relayout_history_cards)
        QTimer.singleShot(80, self._relayout_history_cards)
        if not self._history_items:
            self.hist_list.addItem(self._tr("(vuoto)", "(empty)"))

    def _make_history_section_header_widget(self, section_text: str, marker: str) -> QWidget:
        row = QWidget(self.hist_list.viewport())
        row.setObjectName("historySectionRow")
        row.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(8)

        left_line = QFrame(row)
        left_line.setObjectName("historySectionLine")
        left_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(left_line, 1)

        pill = QLabel(str(section_text or "").upper(), row)
        pill.setObjectName("historySectionPill")
        pill.setProperty("status", str(marker or "").strip().lower())
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pill, 0)

        right_line = QFrame(row)
        right_line.setObjectName("historySectionLine")
        right_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(right_line, 1)
        return row

    def _clear_history_section_overlays(self):
        overlays = getattr(self, "_history_section_overlays", {})
        for w in overlays.values():
            try:
                w.deleteLater()
            except Exception:
                pass
        self._history_section_overlays = {}

    def _position_history_section_overlays(self):
        if not hasattr(self, "hist_list"):
            return
        if not hasattr(self, "_history_section_overlays"):
            self._history_section_overlays = {}
        to_keep: set[int] = set()
        n = self.hist_list.count()
        for i in range(n):
            item = self.hist_list.item(i)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole + 2) != "history_section_header":
                continue
            rect = self.hist_list.visualItemRect(item)
            if rect.height() <= 0:
                continue
            if i + 1 < n:
                next_item = self.hist_list.item(i + 1)
                if (
                    next_item is not None
                    and next_item.data(Qt.ItemDataRole.UserRole + 2) == "history_section_header_pad"
                    and not next_item.isHidden()
                ):
                    rect = rect.united(self.hist_list.visualItemRect(next_item))

            overlay = self._history_section_overlays.get(i)
            if overlay is None:
                section_text = str(item.data(Qt.ItemDataRole.UserRole + 3) or "")
                marker = str(item.data(Qt.ItemDataRole.UserRole + 4) or "")
                overlay = self._make_history_section_header_widget(section_text, marker)
                self._history_section_overlays[i] = overlay
            overlay.setGeometry(rect)
            overlay.show()
            overlay.raise_()
            to_keep.add(i)

        stale = [k for k in self._history_section_overlays.keys() if k not in to_keep]
        for k in stale:
            try:
                self._history_section_overlays[k].deleteLater()
            except Exception:
                pass
            self._history_section_overlays.pop(k, None)

    def _history_entry_status(
        self,
        entry: HistoryEntry,
    ) -> tuple[str, bool, int, dict[str, Any] | None]:
        ep_map = self._entry_episode_progress_map(entry)
        last_epf = self._to_ep_float(entry.last_ep)
        last_prog = self._entry_get_episode_progress(entry, entry.last_ep)
        if last_prog is None and last_epf is not None:
            last_prog = ep_map.get(last_epf)
        in_progress = False
        if isinstance(last_prog, dict):
            lp_pos = float(last_prog.get("pos", 0.0) or 0.0)
            lp_pct = float(last_prog.get("percent", 0.0) or 0.0)
            lp_done = bool(last_prog.get("completed", False))
            in_progress = (not lp_done) and (lp_pos > 0.0 or lp_pct > 0.0)
        seen_count = len(self._entry_watched_eps_set(entry))
        if bool(entry.completed) and not in_progress:
            return "Completed", in_progress, seen_count, last_prog
        if in_progress or seen_count > 0:
            return "Watching", in_progress, seen_count, last_prog
        return "Planned", in_progress, seen_count, last_prog

    def on_history_filter_changed(self):
        self._history_filter = str(self.combo_history_filter.currentData() or "All")
        self._selected_history = None
        self.refresh_history_ui()

    def _on_history_cover_resolved(
        self,
        row: int,
        entry: HistoryEntry,
        data: bytes | None,
        from_cache: bool,
        url: str | None,
    ):
        if url:
            entry.cover_url = url
            self._remember_offline_cover_url(entry.name, url)
            if not self._incognito_enabled:
                try:
                    self.history.upsert(entry)
                except Exception:
                    pass
        self._on_history_cover_loaded(row, data, from_cache)

    def _relayout_history_cards(self):
        if not hasattr(self, "hist_list"):
            return
        if self.hist_list.count() <= 0:
            self._clear_history_section_overlays()
            return
        vw = self.hist_list.viewport().width()
        # Keep a small safety margin to avoid 1-column fallback from rounding/borders.
        vw = max(0, vw - 12)
        spacing = max(0, self.hist_list.spacing())
        min_card_w = 250
        cols = max(1, min(2, (vw + spacing) // max(1, (min_card_w + spacing))))
        avail = max(min_card_w, vw - 8)
        card_w = (avail - spacing * (cols - 1)) // cols
        card_w = max(min_card_w, card_w)
        self.hist_list.setGridSize(QSize(card_w, 106))
        for i in range(self.hist_list.count()):
            item = self.hist_list.item(i)
            if item is None:
                continue
            item_kind = item.data(Qt.ItemDataRole.UserRole + 2)
            if item_kind == "history_section_header":
                item.setHidden(False)
                item.setSizeHint(QSize(card_w, 34))
                continue
            if item_kind in {"history_section_header_prepad", "history_section_header_pad"}:
                item.setHidden(cols < 2)
                item.setSizeHint(QSize(card_w, 34))
                continue
            if item.flags() == Qt.ItemFlag.NoItemFlags:
                continue
            item.setSizeHint(QSize(card_w, 98))
        self._position_history_section_overlays()

    def _on_history_cover_loaded(self, row: int, data: bytes | None, from_cache: bool):
        if data is None:
            return
        if row < 0 or row >= self.hist_list.count():
            return
        label = self._history_cover_labels.get(row)
        if label is None:
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        self.hist_list.item(row).setData(Qt.ItemDataRole.UserRole + 1, "loaded")

    def _history_entry_from_current_item(self) -> HistoryEntry | None:
        item = self.hist_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, HistoryEntry) else None

    def on_pick_history(self):
        entry = self._history_entry_from_current_item()
        if entry is None:
            return
        self._selected_history = entry
        self.set_status(f"Selezionato in cronologia: {entry.name}")

    def on_history_delete(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        self.history.delete(e.provider, e.identifier, e.lang)
        self.refresh_history_ui()
        self.set_status("Rimosso dalla cronologia.")

    def on_history_resume(self):
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history

        self._set_provider(e.provider if e.provider else "allanime")
        self.lang = LanguageTypeEnum.SUB if e.lang == "SUB" else LanguageTypeEnum.DUB
        self.lang_combo.setCurrentIndex(0 if self.lang == LanguageTypeEnum.SUB else 1)

        try:
            self.selected_anime = self.build_anime_from_history(e)
        except Exception as ex:
            self.notify_err(str(ex))
            return

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.lbl_anime_title.setText(e.name)
        self.lbl_player_title.setText(e.name)
        self.set_status(f"Carico episodi per resume: {e.name}…")

        def after_eps(eps: list[float | int]):
            self.episodes_list = eps
            self.episodes.clear()
            play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            for ep in eps:
                item = QListWidgetItem(play_icon, self._episode_label(ep))
                item.setToolTip(self._tr(f"Doppio click per riprodurre episodio {ep}", f"Double click to play episode {ep}"))
                item.setSizeHint(QSize(0, 44))
                self.episodes.addItem(item)

            target_ep = eps[0] if eps else e.last_ep
            should_seek_resume = not e.completed

            # map history episode (float) to the exact episode object returned by provider
            # to avoid provider edge cases where 1.0 != "1"/1 internally
            matched_idx = None
            for i, ep in enumerate(eps):
                try:
                    if abs(float(ep) - float(e.last_ep)) < 1e-6:
                        matched_idx = i
                        break
                except Exception:
                    continue

            if matched_idx is not None:
                if e.completed and matched_idx + 1 < len(eps):
                    target_ep = eps[matched_idx + 1]
                else:
                    target_ep = eps[matched_idx]
            elif e.completed:
                for ep in eps:
                    try:
                        if float(ep) > float(e.last_ep):
                            target_ep = ep
                            break
                    except Exception:
                        continue

            self._pending_resume_seek = None
            self._pending_resume_seek_ratio = None
            self._pending_resume_seek_attempts = 0
            if should_seek_resume:
                if float(e.last_pos) > 0.0:
                    self._pending_resume_seek = max(0.0, float(e.last_pos) - 5.0)
                if float(getattr(e, "last_percent", 0.0) or 0.0) > 0.0:
                    self._pending_resume_seek_ratio = max(
                        0.0, min(1.0, (float(e.last_percent) / 100.0) - 0.01)
                    )
                self._resume_marker_pos = max(0.0, float(e.last_pos))
            else:
                self._resume_marker_pos = None

            self.play_episode(target_ep)

        w = Worker(self.do_episodes, self.selected_anime)
        w.ok.connect(after_eps)
        w.err.connect(self.on_worker_error)
        self._workers.append(w)
        w.start()

    def on_history_resume_next(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        cur_prog = self._entry_get_episode_progress(e, e.last_ep) or {}
        cur_dur = float(cur_prog.get("dur", e.last_duration) or 0.0)
        self._entry_add_watched_ep(e, e.last_ep)
        self._entry_set_episode_progress(
            e,
            e.last_ep,
            pos=cur_dur if cur_dur > 0 else float(cur_prog.get("pos", e.last_pos) or 0.0),
            dur=cur_dur,
            percent=100.0,
            completed=True,
        )
        e.last_percent = 100.0
        e.updated_at = time.time()
        avail_eps = self._episodes_list_for_history_entry(e)
        if avail_eps is not None:
            e.completed = self._entry_is_series_completed(e, avail_eps)
        self.history.upsert(e)
        self.refresh_history_ui()
        self.on_history_resume()

    def on_history_mark_seen(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        if e.last_duration > 0:
            e.last_pos = e.last_duration
        else:
            e.last_pos = max(0.0, float(e.last_pos))
        self._entry_add_watched_ep(e, e.last_ep)
        self._entry_set_episode_progress(
            e,
            e.last_ep,
            pos=float(e.last_pos),
            dur=float(e.last_duration),
            percent=100.0,
            completed=True,
        )
        e.last_percent = 100.0
        e.updated_at = time.time()
        avail_eps = self._episodes_list_for_history_entry(e)
        if avail_eps is not None:
            e.completed = self._entry_is_series_completed(e, avail_eps)
        self.history.upsert(e)
        self.refresh_history_ui()
        self.set_status(f"Segnato come visto: {e.name} ep {e.last_ep}")

    def on_history_open_details(self, *_args):
        if not getattr(self, "_selected_history", None):
            cur = self._history_entry_from_current_item()
            if cur is None:
                return
            self._selected_history = cur
        e: HistoryEntry = self._selected_history

        self._set_provider(e.provider if e.provider else "allanime")
        self.lang = LanguageTypeEnum.SUB if e.lang == "SUB" else LanguageTypeEnum.DUB
        self.lang_combo.setCurrentIndex(0 if self.lang == LanguageTypeEnum.SUB else 1)

        try:
            self.selected_anime = self.build_anime_from_history(e)
        except Exception as ex:
            self.notify_err(str(ex))
            return

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.lbl_anime_title.setText(e.name)
        self.lbl_player_title.setText(e.name)
        self.set_status(f"Apro dettagli anime: {e.name}…")
        self.fetch_episodes(
            seen_eps=self._entry_watched_eps_set(e),
            episode_progress=self._entry_episode_progress_map(e),
        )

    def on_clear_watch_history(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not self.history.list():
            return
        self.history._data = []
        try:
            if os.path.exists(self.history.path):
                os.remove(self.history.path)
        except Exception:
            pass
        self.refresh_history_ui()

    # ---------------- Playback ----------------
    def play_episode(self, ep: float | int):
        if not self.selected_anime:
            return

        t0 = time.perf_counter()
        debug_log(f"Play episode requested: ep={ep}, anime={self.selected_anime.name}")
        self.lbl_player_title.setText(f"{self.selected_anime.name} — Ep {ep}")
        self.search_stack.setCurrentWidget(self.page_player)
        self._current_search_view = "player"
        self._refresh_search_layout()
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = ep
        self.set_status(f"Risolvo stream ep {ep}…")
        self._set_transport_enabled(False)

        preferred_quality = (
            self.quality if self.quality in ("best", "worst") else int(self.quality)
        )

        anime = self.selected_anime
        lang = self.lang

        w = Worker(self.do_resolve_stream, anime, ep, lang, preferred_quality, self.provider_name)

        def on_ok(res: StreamResult):
            try:
                debug_log(f"Stream resolved, loading player for ep={ep}")
                self.player.load(res.url, referrer=res.referrer, sub_file=res.sub_file)

                self._last_progress_save_at = 0.0
                self._last_progress_pos = -1.0
                self._last_progress_percent = -1.0
                if self._pending_resume_seek is not None:
                    self._pending_resume_seek_attempts = 0
                    QTimer.singleShot(500, self._try_apply_resume_seek)
                elif self._pending_resume_seek_ratio is not None:
                    QTimer.singleShot(500, self._try_apply_resume_seek_ratio)

                self.set_status(f"▶ {anime.name} — ep {ep}")
                debug_log(f"Episode ready: ep={ep}, total flow {time.perf_counter() - t0:.3f}s")
            finally:
                self._set_transport_enabled(True)

        def on_err(msg: str):
            debug_log(f"Play episode error: ep={ep} err={msg}")
            self.notify_err(msg)
            self.set_status("Errore.")
            self._set_transport_enabled(True)

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()


    def _is_video_fullscreen(self) -> bool:
        fs = getattr(self, "_video_fs_window", None)
        return fs is not None and fs.isVisible()

    def _set_transport_enabled(self, enabled: bool):
        self.btn_playpause.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        if self._fs_btn_playpause:
            self._fs_btn_playpause.setEnabled(enabled)
        if self._fs_btn_prev:
            self._fs_btn_prev.setEnabled(enabled)
        if self._fs_btn_next:
            self._fs_btn_next.setEnabled(enabled)

    def _set_volume_controls(self, v: int, source: str):
        if source != "main" and self.vol_slider.value() != v:
            self.vol_slider.blockSignals(True)
            self.vol_slider.setValue(v)
            self.vol_slider.blockSignals(False)

        if source != "fs" and self._fs_vol_slider and self._fs_vol_slider.value() != v:
            self._fs_vol_slider.blockSignals(True)
            self._fs_vol_slider.setValue(v)
            self._fs_vol_slider.blockSignals(False)

    def _build_fs_overlay(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        panel.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 rgba(0,0,0,0), stop:1 rgba(0,0,0,210));"
            "border-top: 1px solid rgba(255,255,255,45);"
            "color: #f2f2f2;"
        )
        self._fs_overlay_effect = QGraphicsOpacityEffect(panel)
        self._fs_overlay_effect.setOpacity(1.0)
        panel.setGraphicsEffect(self._fs_overlay_effect)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(18, 24, 18, 12)
        pl.setSpacing(10)

        fs_seek = QSlider(Qt.Orientation.Horizontal)
        fs_seek.setRange(0, 1000)
        fs_seek.sliderPressed.connect(self._on_fs_seek_pressed)
        fs_seek.sliderReleased.connect(self._on_fs_seek_released)
        pl.addWidget(fs_seek)

        row = QHBoxLayout()
        row.setSpacing(10)

        fs_prev = QPushButton(self._tr("⏮ Precedente", "⏮ Prev"))
        fs_prev.clicked.connect(self.on_prev_episode)
        row.addWidget(fs_prev)

        fs_playpause = QPushButton(self._tr("⏯ Play/Pause", "⏯ Play/Pause"))
        fs_playpause.clicked.connect(self.on_toggle_pause)
        row.addWidget(fs_playpause)

        fs_next = QPushButton(self._tr("⏭ Successivo", "⏭ Next"))
        fs_next.clicked.connect(self.on_next_episode)
        row.addWidget(fs_next)

        fs_time = QLabel("00:00 / 00:00")
        row.addWidget(fs_time)

        row.addStretch(1)
        row.addWidget(QLabel(self._tr("Volume", "Volume")))

        fs_vol = QSlider(Qt.Orientation.Horizontal)
        fs_vol.setRange(0, 130)
        fs_vol.setFixedWidth(200)
        fs_vol.setValue(self.vol_slider.value())
        fs_vol.valueChanged.connect(self._on_fs_volume_changed)
        row.addWidget(fs_vol)

        fs_exit = QPushButton(self._tr("Esci FS", "Exit FS"))
        fs_exit.clicked.connect(self.toggle_fullscreen)
        row.addWidget(fs_exit)

        pl.addLayout(row)

        self._fs_seek_slider = fs_seek
        self._fs_vol_slider = fs_vol
        self._fs_lbl_time = fs_time
        self._fs_btn_playpause = fs_playpause
        self._fs_btn_prev = fs_prev
        self._fs_btn_next = fs_next
        self._fs_btn_exit = fs_exit
        self._set_transport_enabled(self.btn_playpause.isEnabled())
        return panel

    def _position_fs_overlay(self):
        if not self._video_fs_window or not self._fs_overlay:
            return
        margin = 20
        h = self._fs_overlay.sizeHint().height()
        w = max(480, self._video_fs_window.width() - (margin * 2))
        self._fs_overlay.setGeometry(
            margin,
            max(0, self._video_fs_window.height() - h - 14),
            w,
            h,
        )
        self._fs_overlay.raise_()
        if self._fs_help_label:
            self._fs_help_label.adjustSize()
            self._fs_help_label.move(
                max(12, (self._video_fs_window.width() - self._fs_help_label.width()) // 2),
                14,
            )

    def _fs_mark_active(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if self._fs_overlay:
            self._fs_overlay.show()
            self._fs_fade_overlay(1.0)
            self._position_fs_overlay()
        self._video_fs_window.unsetCursor()
        self._fs_autohide_timer.start()

    def _fs_fade_overlay(self, target: float):
        if not self._fs_overlay_effect:
            return
        if self._fs_overlay_anim:
            self._fs_overlay_anim.stop()
        self._fs_overlay_anim = QPropertyAnimation(self._fs_overlay_effect, b"opacity")
        self._fs_overlay_anim.setDuration(180)
        self._fs_overlay_anim.setStartValue(self._fs_overlay_effect.opacity())
        self._fs_overlay_anim.setEndValue(target)
        if target <= 0.0 and self._fs_overlay:
            self._fs_overlay_anim.finished.connect(self._fs_overlay.hide)
        self._fs_overlay_anim.start()

    def _fs_poll_cursor(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        pos = QCursor.pos()
        if self._fs_last_cursor_pos is None:
            self._fs_last_cursor_pos = pos
            return
        if pos != self._fs_last_cursor_pos:
            self._fs_last_cursor_pos = pos
            self._fs_mark_active()

    def _fs_apply_hidden(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if self._fs_seeking:
            self._fs_autohide_timer.start()
            return
        if self._fs_overlay:
            self._fs_fade_overlay(0.0)
        self._video_fs_window.setCursor(Qt.CursorShape.BlankCursor)

    def _fs_show_help(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if not self._fs_help_label:
            return
        self._fs_help_label.show()
        self._position_fs_overlay()
        QTimer.singleShot(2500, self._fs_help_label.hide)

    def _enter_video_fullscreen(self):
        if self._is_video_fullscreen():
            return
        if self._mini_player_window is not None:
            self._exit_mini_player()

        debug_log("Entering video-only fullscreen")
        fs = FullscreenPlayerWindow()
        fs.setWindowTitle(self._tr("Anigui Player", "Anigui Player"))
        fs.setWindowFlag(Qt.WindowType.Window, True)
        fs.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        fs.exit_requested.connect(self._exit_video_fullscreen)
        fs.activity.connect(self._fs_mark_active)

        fs_layout = QVBoxLayout(fs)
        fs_layout.setContentsMargins(0, 0, 0, 0)
        fs_layout.setSpacing(0)
        self.player.setParent(fs)
        fs_layout.addWidget(self.player, 1)

        self._fs_overlay = self._build_fs_overlay(fs)

        fs_esc = QShortcut(QKeySequence("Escape"), fs, activated=self._exit_video_fullscreen)
        fs_f = QShortcut(QKeySequence("F"), fs, activated=self.toggle_fullscreen)
        fs_help = QShortcut(QKeySequence("?"), fs, activated=self._fs_show_help)
        self._video_fs_shortcuts = [fs_esc, fs_f, fs_help]

        self._fs_help_label = QLabel(
            self._tr(
                "Scorciatoie: F schermo intero, Esc esci, Space play/pausa, ←/→ seek, ↑/↓ volume, ? aiuto",
                "Shortcuts: F fullscreen, Esc exit, Space play/pause, ←/→ seek, ↑/↓ volume, ? help",
            ),
            fs,
        )
        self._fs_help_label.setStyleSheet(
            "background: rgba(0,0,0,180); color: #ffffff; padding: 8px 12px; border-radius: 8px;"
        )
        self._fs_help_label.setVisible(False)

        self._video_fs_window = fs
        fs.showFullScreen()
        self._fs_last_cursor_pos = QCursor.pos()
        self._fs_mouse_poll_timer.start()
        fs.installEventFilter(self)
        self.player.installEventFilter(self)
        if self._fs_overlay:
            self._fs_overlay.installEventFilter(self)
            for w in self._fs_overlay.findChildren(QWidget):
                w.installEventFilter(self)
        self._position_fs_overlay()
        self._fs_mark_active()
        self.btn_fs.setText(self._tr("Esci FS", "Exit FS"))

    def _exit_video_fullscreen(self):
        if self._closing_video_fs:
            return
        fs = self._video_fs_window
        if fs is None:
            return

        self._closing_video_fs = True
        try:
            debug_log("Exiting video-only fullscreen")
            self.player.setParent(self.player_host)
            self.player_host_layout.addWidget(self.player, 1)

            for sc in self._video_fs_shortcuts:
                try:
                    sc.setParent(None)
                except Exception:
                    pass
            self._video_fs_shortcuts = []

            if self._fs_overlay:
                for w in self._fs_overlay.findChildren(QWidget):
                    w.removeEventFilter(self)
                self._fs_overlay.removeEventFilter(self)
                self._fs_overlay.hide()
                self._fs_overlay.deleteLater()
            if self._fs_help_label:
                self._fs_help_label.hide()
                self._fs_help_label.deleteLater()
            self.player.removeEventFilter(self)
            fs.removeEventFilter(self)
            self._fs_autohide_timer.stop()
            self._fs_mouse_poll_timer.stop()
            fs.unsetCursor()
            self._fs_overlay = None
            self._fs_overlay_effect = None
            self._fs_overlay_anim = None
            self._fs_seek_slider = None
            self._fs_vol_slider = None
            self._fs_lbl_time = None
            self._fs_btn_playpause = None
            self._fs_btn_prev = None
            self._fs_btn_next = None
            self._fs_btn_exit = None
            self._fs_help_label = None
            self._fs_seeking = False
            self._fs_last_cursor_pos = None

            fs.hide()
            fs.deleteLater()
            self._video_fs_window = None
            self.btn_fs.setText(self._tr("⛶ Schermo intero", "⛶ Fullscreen"))
        finally:
            self._closing_video_fs = False

    def toggle_fullscreen(self):
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        else:
            self._enter_video_fullscreen()

    def _enter_mini_player(self):
        if self._mini_player_window is not None:
            return
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        mini = MiniPlayerWindow()
        mini.setWindowTitle(self._tr("Anigui Mini Player", "Anigui Mini Player"))
        mini.setWindowFlag(Qt.WindowType.Window, True)
        mini.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        mini.exit_requested.connect(self._exit_mini_player)
        lay = QVBoxLayout(mini)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.player.setParent(mini)
        lay.addWidget(self.player, 1)
        mini.resize(520, 300)
        mini.show()
        self._mini_player_window = mini
        self.btn_mini.setText(self._tr("Esci mini", "Exit Mini"))

    def _exit_mini_player(self):
        mini = self._mini_player_window
        if mini is None:
            return
        self.player.setParent(self.player_host)
        self.player_host_layout.addWidget(self.player, 1)
        mini.hide()
        mini.deleteLater()
        self._mini_player_window = None
        self.btn_mini.setText(self._tr("▣ Mini Player", "▣ Mini"))

    def toggle_mini_player(self):
        if self._mini_player_window is None:
            self._enter_mini_player()
        else:
            self._exit_mini_player()

    def on_prev_episode(self):
        if self._local_media_active and self._offline_episode_files:
            if self._offline_current_episode_index is None:
                return
            idx = self._offline_current_episode_index
            if idx <= 0:
                return
            self._offline_current_episode_index = idx - 1
            self._play_local_file(self._offline_episode_files[self._offline_current_episode_index])
            return
        if self.current_ep is None or not self.episodes_list:
            return
        try:
            idx = self.episodes_list.index(self.current_ep)
        except ValueError:
            return
        if idx <= 0:
            return
        self.play_episode(self.episodes_list[idx - 1])

    def on_next_episode(self):
        if self._local_media_active and self._offline_episode_files:
            if self._offline_current_episode_index is None:
                return
            idx = self._offline_current_episode_index
            if idx >= len(self._offline_episode_files) - 1:
                return
            self._offline_current_episode_index = idx + 1
            self._play_local_file(self._offline_episode_files[self._offline_current_episode_index])
            return
        if self.current_ep is None or not self.episodes_list:
            return
        try:
            idx = self.episodes_list.index(self.current_ep)
        except ValueError:
            return
        if idx >= len(self.episodes_list) - 1:
            return
        self.play_episode(self.episodes_list[idx + 1])

    def on_toggle_pause(self):
        try:
            self.player.toggle_pause()
        except Exception:
            pass

    def on_volume_changed(self, v: int):
        try:
            self.player.set_volume(v)
            self._set_volume_controls(v, source="main")
        except Exception:
            pass

    def _on_fs_volume_changed(self, v: int):
        try:
            self.player.set_volume(v)
            self._set_volume_controls(v, source="fs")
            self._fs_mark_active()
        except Exception:
            pass

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_released(self):
        self._seeking = False
        ratio = self.seek_slider.value() / 1000.0
        dur = self.player.get_duration()
        self._timeline_diag(f"seek(main) ratio={ratio:.3f} dur={dur}", force=True)
        try:
            if dur is None or dur <= 0:
                self.player.seek_ratio(ratio)
            else:
                self.player.seek(ratio * dur)
        except Exception:
            pass

    def _position_resume_marker(self):
        if self._resume_marker_pos is None:
            self._resume_marker.hide()
            return
        dur = self.player.get_duration()
        if dur is None or dur <= 0:
            self._resume_marker.hide()
            return
        ratio = max(0.0, min(1.0, self._resume_marker_pos / dur))
        w = self.seek_slider.width()
        x = int(ratio * (w - self._resume_marker.width()))
        self._resume_marker.setGeometry(x, 2, self._resume_marker.width(), self.seek_slider.height() - 4)
        self._resume_marker.show()

    def _on_fs_seek_pressed(self):
        self._fs_seeking = True
        self._fs_mark_active()

    def _on_fs_seek_released(self):
        self._fs_seeking = False
        if self._fs_seek_slider is None:
            return
        ratio = self._fs_seek_slider.value() / 1000.0
        dur = self.player.get_duration()
        self._timeline_diag(f"seek(fs) ratio={ratio:.3f} dur={dur}", force=True)
        try:
            if dur is None or dur <= 0:
                self.player.seek_ratio(ratio)
            else:
                self.player.seek(ratio * dur)
            self._fs_mark_active()
        except Exception:
            pass

    def poll_player(self):
        if not self._local_media_active and self.current_ep is None:
            return
        if hasattr(self.player, "_load_thread") and getattr(self.player, "_load_thread", None) and self.player._load_thread.isRunning():
            return
        
        dur = self.player.get_duration()
        pos = self.player.get_time_pos()
        percent = self.player.get_percent_pos()
        paused = self.player.get_pause()
        if pos is None and dur is not None and dur > 0 and percent is not None:
            pos = max(0.0, min(dur, dur * (percent / 100.0)))
        if pos is None:
            self._timeline_missing_logged = True
            self._timeline_diag(
                f"no-pos dur={dur} percent={percent} paused={paused} player={self.player.__class__.__name__}"
            )
            return
        if self._timeline_missing_logged:
            self._timeline_diag(
                f"recovered pos={pos:.2f} dur={dur} percent={percent} player={self.player.__class__.__name__}",
                force=True,
            )
            self._timeline_missing_logged = False
        if dur is None or dur <= 0:
            self.lbl_time.setText(f"{self.fmt_time(pos)} / --:--")
        else:
            self.lbl_time.setText(f"{self.fmt_time(pos)} / {self.fmt_time(dur)}")
        if self._fs_lbl_time is not None:
            if dur is None or dur <= 0:
                self._fs_lbl_time.setText(f"{self.fmt_time(pos)} / --:--")
            else:
                self._fs_lbl_time.setText(f"{self.fmt_time(pos)} / {self.fmt_time(dur)}")

        if not self._seeking and dur is not None and dur > 0:
            ratio = max(0.0, min(1.0, pos / dur))
            if percent is not None:
                ratio = max(0.0, min(1.0, percent / 100.0))
            v = int(ratio * 1000)
            self.seek_slider.setValue(v)
            if self._fs_seek_slider is not None and not self._fs_seeking:
                self._fs_seek_slider.setValue(v)
        self._position_resume_marker()

        if paused is True:
            self.btn_playpause.setText(self._tr("▶ Play", "▶ Play"))
            if self._fs_btn_playpause is not None:
                self._fs_btn_playpause.setText(self._tr("▶ Play", "▶ Play"))
        elif paused is False:
            self.btn_playpause.setText(self._tr("⏸ Pausa", "⏸ Pause"))
            if self._fs_btn_playpause is not None:
                self._fs_btn_playpause.setText(self._tr("⏸ Pausa", "⏸ Pause"))

        self._save_current_progress(force=False)

    # ---------------- Worker errors ----------------
    def on_worker_error(self, msg: str):
        self._end_request()
        self.notify_err(msg)
        self.set_status("Errore.")

    def eventFilter(self, obj, event):
        if not hasattr(self, "_video_fs_window"):
            return super().eventFilter(obj, event)
        if obj is self.seek_slider and event.type() == QEvent.Type.Resize:
            self._position_resume_marker()
        if self._is_video_fullscreen() and self._video_fs_window is not None:
            if isinstance(obj, QWidget) and obj.window() is self._video_fs_window:
                t = event.type()
                if t in (
                    QEvent.Type.MouseMove,
                    QEvent.Type.MouseButtonPress,
                    QEvent.Type.MouseButtonRelease,
                    QEvent.Type.Wheel,
                    QEvent.Type.KeyPress,
                    QEvent.Type.KeyRelease,
                ):
                    self._fs_mark_active()
        return super().eventFilter(obj, event)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "hist_list"):
            QTimer.singleShot(0, self._relayout_history_cards)

    def closeEvent(self, e):
        self._save_current_progress(force=True)
        for worker in list(self._active_download_workers.values()):
            try:
                worker.request_cancel()
            except Exception:
                pass
        for worker in list(self._active_download_workers.values()):
            try:
                worker.wait(1200)
            except Exception:
                pass
        try:
            self.player.stop()
        except Exception:
            pass
        if self._mini_player_window is not None:
            self._exit_mini_player()
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        super().closeEvent(e)


def main():
    debug_log("Application start")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    debug_log("Application UI shown, entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
