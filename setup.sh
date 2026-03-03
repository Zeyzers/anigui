#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_SYSTEM=0
SKIP_VENV=0

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Options:
  --skip-system   Skip OS package installation
  --skip-venv     Install with current Python, do not create/use .venv
  --python PATH   Python binary to use (default: python3)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-system)
      SKIP_SYSTEM=1
      shift
      ;;
    --skip-venv)
      SKIP_VENV=1
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

install_system_deps() {
  if [[ "$SKIP_SYSTEM" -eq 1 ]]; then
    echo "[setup] Skipping system package installation."
    return
  fi

  if [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
  else
    SUDO=""
  fi

  if command -v apt-get >/dev/null 2>&1; then
    echo "[setup] Installing dependencies via apt..."
    $SUDO apt-get update
    $SUDO apt-get install -y \
      python3 python3-venv python3-pip \
      mpv libxcb-cursor0
    return
  fi

  if command -v dnf >/dev/null 2>&1; then
    echo "[setup] Installing dependencies via dnf..."
    $SUDO dnf install -y \
      python3 python3-pip \
      mpv libxcb-cursor
    return
  fi

  if command -v pacman >/dev/null 2>&1; then
    echo "[setup] Installing dependencies via pacman..."
    $SUDO pacman -Sy --noconfirm \
      python python-pip \
      mpv xcb-util-cursor
    return
  fi

  if command -v zypper >/dev/null 2>&1; then
    echo "[setup] Installing dependencies via zypper..."
    $SUDO zypper --non-interactive install \
      python3 python3-pip python3-venv \
      mpv libxcb-cursor0
    return
  fi

  echo "[setup] Unsupported package manager. Install manually:" >&2
  echo "  - Python 3.10+" >&2
  echo "  - mpv" >&2
  echo "  - libxcb-cursor runtime package (name varies by distro)" >&2
}

setup_python_env() {
  if [[ "$SKIP_VENV" -eq 1 ]]; then
    echo "[setup] Installing Python dependencies in current environment..."
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
    return
  fi

  if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo "[setup] Creating virtual environment in .venv..."
    "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
  fi

  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  echo "[setup] Installing Python dependencies in .venv..."
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT_DIR/requirements.txt"
}

main() {
  cd "$ROOT_DIR"
  install_system_deps
  setup_python_env
  echo "[setup] Done."
  echo "[setup] Run app with:"
  if [[ "$SKIP_VENV" -eq 1 ]]; then
    echo "  $PYTHON_BIN app.py"
  else
    echo "  source .venv/bin/activate && python app.py"
  fi
}

main "$@"

