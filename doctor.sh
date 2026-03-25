#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; }

status=0

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    ok "$name found: $(command -v "$name")"
  else
    fail "$name not found"
    status=1
  fi
}

check_xcb_cursor() {
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is required for xcb-cursor runtime check"
    status=1
    return
  fi

  if python3 - <<'PY'
import ctypes
import ctypes.util

names = ["xcb-cursor", "xcb_cursor", "libxcb-cursor.so.0", "libxcb-cursor.so"]
for n in names:
    lib = ctypes.util.find_library(n) or n
    try:
        ctypes.CDLL(lib)
        raise SystemExit(0)
    except OSError:
        pass
raise SystemExit(1)
PY
  then
    ok "libxcb-cursor runtime is available"
  else
    fail "libxcb-cursor runtime is missing (Qt xcb plugin may crash)"
    echo "      Debian/Ubuntu/Mint fix: sudo apt update && sudo apt install -y libxcb-cursor0"
    status=1
  fi
}

check_requirements() {
  if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    ok ".venv found"
  else
    warn ".venv not found (optional but recommended)"
  fi

  if python3 -m pip show PySide6 >/dev/null 2>&1; then
    ok "PySide6 installed"
  else
    warn "PySide6 not installed in current environment"
    status=1
  fi
}

echo "[doctor] Running Linux runtime checks..."
check_cmd python3
check_cmd mpv
check_xcb_cursor
check_requirements

if [[ "$status" -eq 0 ]]; then
  echo "[doctor] All checks passed."
else
  echo "[doctor] One or more checks failed."
fi
exit "$status"

