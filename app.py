from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys

from crash_logging import install_exception_hooks, setup_logging
from services import APP_VERSION


def _has_xcb_cursor_lib() -> bool:
    probe_names = ["xcb-cursor", "xcb_cursor"]
    for name in probe_names:
        lib = ctypes.util.find_library(name)
        if not lib:
            continue
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            pass

    # Common runtime sonames on Debian/Ubuntu/Mint.
    for soname in ("libxcb-cursor.so.0", "libxcb-cursor.so"):
        try:
            ctypes.CDLL(soname)
            return True
        except OSError:
            pass
    return False


def _linux_qt_runtime_guard() -> int:
    if platform.system() != "Linux":
        return 0
    if os.getenv("QT_QPA_PLATFORM"):
        return 0

    # Prefer native Wayland if the session is Wayland and no platform is pinned.
    if os.getenv("XDG_SESSION_TYPE", "").strip().lower() == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return 0

    # On X11 Qt xcb plugin needs libxcb-cursor at runtime.
    if not _has_xcb_cursor_lib():
        print(
            "[anigui] Missing Linux dependency: libxcb-cursor0.\n"
            "Install it and retry.\n"
            "Debian/Ubuntu/Linux Mint:\n"
            "  sudo apt update && sudo apt install -y libxcb-cursor0",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    setup_logging(app_version=APP_VERSION)
    install_exception_hooks()
    code = _linux_qt_runtime_guard()
    if code != 0:
        raise SystemExit(code)

    from main_window import main

    main()
