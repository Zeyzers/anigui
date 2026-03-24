from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import urllib.request
from urllib.parse import urlparse
from typing import Any


APP_NAME = "Anigui"
APP_VERSION = "1.1.8"
UPDATE_MANIFEST_URL = os.getenv(
    "ANIGUI_UPDATE_MANIFEST_URL",
    "https://raw.githubusercontent.com/Zeyzers/anigui/main/update_manifest.json",
)


class UpdateService:
    _SAFE_URL_SCHEMES = {"http", "https"}

    def __init__(self, manifest_url: str = UPDATE_MANIFEST_URL, current_version: str = APP_VERSION):
        self.manifest_url = manifest_url
        self.current_version = current_version

    @classmethod
    def _user_agent(cls, system_name: str | None = None) -> str:
        ua = f"{APP_NAME}/{APP_VERSION}"
        sys_name = str(system_name or "").strip()
        if sys_name:
            return f"{ua} ({sys_name})"
        return ua

    @classmethod
    def _url_origin(cls, url: str) -> str:
        candidate = str(url or "").strip()
        if not candidate:
            return ""
        parsed = urlparse(candidate)
        host = parsed.hostname or ""
        if not host:
            return ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme.lower()}://{host.lower()}{port}"

    @classmethod
    def _is_safe_http_url(cls, url: str) -> bool:
        candidate = str(url or "").strip()
        if not candidate or any(ch in candidate for ch in ("\r", "\n", "\t")):
            return False
        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in cls._SAFE_URL_SCHEMES:
            return False
        if not cls._url_origin(candidate):
            return False
        if parsed.username or parsed.password:
            return False
        return True

    @classmethod
    def _validated_http_url(cls, url: str, *, field_name: str) -> str:
        candidate = str(url or "").strip()
        if not cls._is_safe_http_url(candidate):
            raise ValueError(f"URL {field_name} non valida o non sicura.")
        return candidate

    @staticmethod
    def _validated_sha256(value: str) -> str:
        candidate = str(value or "").strip().lower()
        if candidate and not re.fullmatch(r"[0-9a-f]{64}", candidate):
            raise ValueError("Checksum update non valida nel manifest.")
        return candidate

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

    def fetch_manifest(self, system_name: str | None = None) -> dict[str, Any]:
        manifest_url = self._validated_http_url(self.manifest_url, field_name="del manifest")
        req = urllib.request.Request(
            manifest_url,
            headers={"User-Agent": self._user_agent(system_name)},
        )
        with urllib.request.urlopen(req, timeout=12.0) as r:
            raw = r.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Manifest update non valido.")
        latest = str(data.get("version", "")).strip()
        if not latest:
            raise ValueError("Manifest update senza campo 'version'.")

        sys_name = (system_name or platform.system()).lower()
        is_win = sys_name.startswith("win")
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
        url = self._validated_http_url(url, field_name="aggiornamento")
        sha256 = self._validated_sha256(
            str(
                sect.get("sha256")
                or data.get(f"{plat}_sha256")
                or data.get("sha256")
                or ""
            )
        )
        page_url = str(
            data.get("page_url")
            or data.get("release_url")
            or url
            or ""
        ).strip()
        if not self._is_safe_http_url(page_url):
            page_url = url
        notes = str(data.get("notes") or "").strip()
        available = self._is_version_newer(latest, self.current_version)
        return {
            "current_version": self.current_version,
            "latest_version": latest,
            "available": bool(available),
            "url": url,
            "sha256": sha256,
            "page_url": page_url,
            "notes": notes,
        }

    def download_update(self, info: dict[str, Any], system_name: str | None = None) -> str:
        url = self._validated_http_url(info.get("url") or "", field_name="update")
        expected_sha = str(info.get("sha256") or "").strip().lower()
        if expected_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("Checksum update non valida nel manifest.")
        req = urllib.request.Request(url, headers={"User-Agent": self._user_agent(system_name)})
        sys_name = (system_name or platform.system()).lower()
        suffix = ".exe" if sys_name.startswith("win") else ".bin"
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
