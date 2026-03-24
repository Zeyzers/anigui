from __future__ import annotations

import json
import os
import sys
import time
import shutil
import socket
import tempfile
import platform
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
import errno
import threading
from crash_logging import log_debug
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QWidget


def debug_log(msg: str) -> None:
    log_debug(msg)


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

