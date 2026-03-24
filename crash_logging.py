from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication


LOGGER_NAME = "anigui"
_APP_VERSION = "unknown"
_LAST_ACTION = ""
_CONFIGURED_LOG_PATH: str | None = None


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def set_last_action(action: str | None) -> None:
    global _LAST_ACTION
    text = str(action or "").strip()
    if text:
        _LAST_ACTION = text


def last_action() -> str:
    return _LAST_ACTION or "(unknown)"


def _metadata_line() -> str:
    return (
        f"platform={platform.platform()} "
        f"python={platform.python_version()} "
        f"app_version={_APP_VERSION} "
        f"last_action={last_action()}"
    )


def setup_logging(
    *,
    log_dir: str = "logs",
    app_version: str | None = None,
    stream: TextIO | None = None,
    force: bool = False,
) -> str:
    global _APP_VERSION, _CONFIGURED_LOG_PATH

    if app_version:
        _APP_VERSION = str(app_version)

    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = str((target_dir / "app.log").resolve())

    logger = _logger()
    if logger.handlers and _CONFIGURED_LOG_PATH == log_path and not force:
        return log_path

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(stream or sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)

    _CONFIGURED_LOG_PATH = log_path
    logger.debug("Logging initialized | %s | log_file=%s", _metadata_line(), log_path)
    return log_path


def log_debug(message: str) -> None:
    set_last_action(message)
    _logger().debug("%s", str(message))


def log_exception(
    context: str,
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback,
) -> None:
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    _logger().critical(
        "Unhandled exception in %s | %s\nException type: %s\nException message: %s\nTraceback:\n%s",
        context,
        _metadata_line(),
        getattr(exc_type, "__name__", str(exc_type)),
        str(exc_value),
        tb_text.rstrip(),
    )


def install_exception_hooks() -> None:
    def _sys_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log_exception("sys.excepthook", exc_type, exc_value, exc_traceback)

    def _thread_hook(args: threading.ExceptHookArgs):
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        thread_name = getattr(args.thread, "name", "unknown-thread")
        log_exception(f"threading.excepthook[{thread_name}]", args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


class LoggingApplication(QApplication):
    def notify(self, receiver, event):  # type: ignore[override]
        try:
            return super().notify(receiver, event)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type is not None and exc_value is not None:
                receiver_name = type(receiver).__name__ if receiver is not None else "None"
                event_name = type(event).__name__ if event is not None else "None"
                log_exception(f"Qt notify receiver={receiver_name} event={event_name}", exc_type, exc_value, exc_traceback)
            return False


class UiFreezeWatchdog(QObject):
    def __init__(self, *, threshold_s: float = 2.5, heartbeat_ms: int = 500):
        super().__init__()
        self.threshold_s = max(1.0, float(threshold_s))
        self.heartbeat_ms = max(100, int(heartbeat_ms))
        self._last_heartbeat = time.monotonic()
        self._warned = False
        self._stop = threading.Event()
        self._timer = QTimer(self)
        self._timer.setInterval(self.heartbeat_ms)
        self._timer.timeout.connect(self._heartbeat)
        self._timer.start()
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="UIFreezeWatchdog",
            daemon=True,
        )
        self._thread.start()

    def _heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()
        if self._warned:
            _logger().warning("UI thread heartbeat recovered | %s", _metadata_line())
            self._warned = False

    def _watchdog_loop(self) -> None:
        sleep_s = min(0.5, self.threshold_s / 2.0)
        while not self._stop.wait(sleep_s):
            lag = time.monotonic() - self._last_heartbeat
            if lag > self.threshold_s and not self._warned:
                self._warned = True
                _logger().warning(
                    "UI thread heartbeat stalled for %.2fs | %s",
                    lag,
                    _metadata_line(),
                )

    def stop(self) -> None:
        self._stop.set()
        try:
            self._timer.stop()
        except Exception:
            pass

