from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from models import SETTINGS_PATH, app_state_dir


@dataclass(frozen=True)
class RuntimeSettings:
    download_dir: str
    provider_name: str
    lang_name: str
    quality: str
    parallel_downloads: int
    scheduler_enabled: bool
    scheduler_start: str
    scheduler_end: str
    integrity_min_mb: float
    integrity_retry_count: int
    anilist_enabled: bool
    anilist_token: str
    app_language: str


@dataclass(frozen=True)
class PreparedSettings:
    settings: dict[str, Any]
    runtime: RuntimeSettings


class SettingsService:
    _VALID_PROVIDERS = {"allanime", "aw_animeworld", "aw_animeunity"}
    _VALID_LANGS = {"SUB", "DUB"}
    _VALID_QUALITIES = {"best", "worst", "360", "480", "720", "1080"}
    _VALID_LANGUAGES = {"it", "en"}
    _MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "ï")

    @classmethod
    def repair_mojibake_text(cls, value: Any) -> str:
        text = str(value or "")
        if not text or not any(marker in text for marker in cls._MOJIBAKE_MARKERS):
            return text
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except Exception:
            return text
        return repaired if repaired else text

    @classmethod
    def default_settings(cls, *, state_dir: str | None = None) -> dict[str, Any]:
        base_dir = state_dir or app_state_dir()
        return {
            "download_dir": os.path.join(base_dir, "downloads"),
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

    @classmethod
    def normalize_settings(
        cls,
        settings: dict[str, Any] | None,
        *,
        state_dir: str | None = None,
        normalize_token: Callable[[str | None], str] | None = None,
    ) -> dict[str, Any]:
        normalized = cls.default_settings(state_dir=state_dir)
        if isinstance(settings, dict):
            normalized.update(settings)

        download_dir = cls.repair_mojibake_text(normalized.get("download_dir", "")).strip()
        if download_dir:
            normalized["download_dir"] = os.path.abspath(os.path.expanduser(download_dir))
        else:
            normalized["download_dir"] = cls.default_settings(state_dir=state_dir)["download_dir"]

        provider = str(normalized.get("default_provider", "allanime") or "").strip()
        normalized["default_provider"] = provider if provider in cls._VALID_PROVIDERS else "allanime"

        lang_name = str(normalized.get("default_lang", "SUB") or "").strip().upper()
        normalized["default_lang"] = lang_name if lang_name in cls._VALID_LANGS else "SUB"

        quality = str(normalized.get("default_quality", "best") or "").strip()
        normalized["default_quality"] = quality if quality in cls._VALID_QUALITIES else "best"

        try:
            parallel = int(normalized.get("parallel_downloads", 2))
        except Exception:
            parallel = 2
        normalized["parallel_downloads"] = max(1, min(4, parallel))

        normalized["scheduler_enabled"] = bool(normalized.get("scheduler_enabled", False))
        normalized["scheduler_start"] = str(normalized.get("scheduler_start", "00:00") or "00:00").strip()
        normalized["scheduler_end"] = str(normalized.get("scheduler_end", "23:59") or "23:59").strip()

        try:
            integrity_min_mb = float(normalized.get("integrity_min_mb", 2.0))
        except Exception:
            integrity_min_mb = 2.0
        normalized["integrity_min_mb"] = max(0.0, integrity_min_mb)

        try:
            integrity_retry_count = int(normalized.get("integrity_retry_count", 1))
        except Exception:
            integrity_retry_count = 1
        normalized["integrity_retry_count"] = max(0, min(5, integrity_retry_count))

        normalized["anilist_enabled"] = bool(normalized.get("anilist_enabled", False))
        token = normalized.get("anilist_token", "")
        normalized["anilist_token"] = (
            normalize_token(token) if normalize_token is not None else str(token or "").strip()
        )

        app_language = str(normalized.get("app_language", "it") or "").strip().lower()
        normalized["app_language"] = app_language if app_language in cls._VALID_LANGUAGES else "it"
        return normalized

    @classmethod
    def load(
        cls,
        *,
        path: str = SETTINGS_PATH,
        state_dir: str | None = None,
        normalize_token: Callable[[str | None], str] | None = None,
    ) -> dict[str, Any]:
        raw: dict[str, Any] | None = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw = payload
        except Exception:
            raw = None
        return cls.normalize_settings(raw, state_dir=state_dir, normalize_token=normalize_token)

    @classmethod
    def save(
        cls,
        settings: dict[str, Any],
        *,
        path: str = SETTINGS_PATH,
        state_dir: str | None = None,
        normalize_token: Callable[[str | None], str] | None = None,
    ) -> dict[str, Any]:
        normalized = cls.normalize_settings(
            settings,
            state_dir=state_dir,
            normalize_token=normalize_token,
        )
        target_dir = os.path.dirname(path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return normalized

    @classmethod
    def runtime_state(
        cls,
        settings: dict[str, Any],
        *,
        state_dir: str | None = None,
        normalize_token: Callable[[str | None], str] | None = None,
    ) -> RuntimeSettings:
        normalized = cls.normalize_settings(
            settings,
            state_dir=state_dir,
            normalize_token=normalize_token,
        )
        return RuntimeSettings(
            download_dir=str(normalized["download_dir"]),
            provider_name=str(normalized["default_provider"]),
            lang_name=str(normalized["default_lang"]),
            quality=str(normalized["default_quality"]),
            parallel_downloads=int(normalized["parallel_downloads"]),
            scheduler_enabled=bool(normalized["scheduler_enabled"]),
            scheduler_start=str(normalized["scheduler_start"]),
            scheduler_end=str(normalized["scheduler_end"]),
            integrity_min_mb=float(normalized["integrity_min_mb"]),
            integrity_retry_count=int(normalized["integrity_retry_count"]),
            anilist_enabled=bool(normalized["anilist_enabled"]),
            anilist_token=str(normalized["anilist_token"]),
            app_language=str(normalized["app_language"]),
        )

    @staticmethod
    def parse_hhmm(value: str) -> tuple[int, int] | None:
        raw = str(value or "").strip()
        if len(raw) != 5 or raw[2] != ":":
            return None
        try:
            hh = int(raw[:2])
            mm = int(raw[3:])
        except Exception:
            return None
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return hh, mm

    @classmethod
    def scheduler_window_open(
        cls,
        *,
        scheduler_enabled: bool,
        scheduler_start: str,
        scheduler_end: str,
        current_minutes: int,
    ) -> bool:
        if not scheduler_enabled:
            return True
        start = cls.parse_hhmm(scheduler_start)
        end = cls.parse_hhmm(scheduler_end)
        if start is None or end is None:
            return True
        start_minutes = start[0] * 60 + start[1]
        end_minutes = end[0] * 60 + end[1]
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes
        return current_minutes >= start_minutes or current_minutes <= end_minutes

    @classmethod
    def prepare_for_save(
        cls,
        *,
        download_dir: str,
        provider_name: str,
        lang_name: str,
        quality: str,
        parallel_downloads: int,
        scheduler_enabled: bool,
        scheduler_start: str,
        scheduler_end: str,
        integrity_min_mb: float,
        integrity_retry_count: int,
        anilist_enabled: bool,
        anilist_token: str | None,
        app_language: str,
        state_dir: str | None = None,
        normalize_token: Callable[[str | None], str] | None = None,
    ) -> PreparedSettings:
        normalized_download_dir = os.path.abspath(os.path.expanduser(str(download_dir or "").strip()))
        if not normalized_download_dir:
            raise ValueError("invalid_download_dir")
        scheduler_start = str(scheduler_start or "").strip()
        scheduler_end = str(scheduler_end or "").strip()
        if scheduler_enabled:
            if cls.parse_hhmm(scheduler_start) is None or cls.parse_hhmm(scheduler_end) is None:
                raise ValueError("invalid_scheduler")
        normalized_token = normalize_token(anilist_token) if normalize_token is not None else str(anilist_token or "").strip()
        if bool(anilist_enabled) and not normalized_token:
            raise ValueError("missing_anilist_token")
        settings = {
            "download_dir": normalized_download_dir,
            "default_provider": str(provider_name or "allanime"),
            "default_lang": str(lang_name or "SUB").upper(),
            "default_quality": str(quality or "best"),
            "parallel_downloads": int(parallel_downloads),
            "scheduler_enabled": bool(scheduler_enabled),
            "scheduler_start": scheduler_start,
            "scheduler_end": scheduler_end,
            "integrity_min_mb": float(integrity_min_mb),
            "integrity_retry_count": int(integrity_retry_count),
            "anilist_enabled": bool(anilist_enabled),
            "anilist_token": normalized_token,
            "app_language": str(app_language or "it").strip().lower(),
        }
        normalized = cls.normalize_settings(
            settings,
            state_dir=state_dir,
            normalize_token=normalize_token,
        )
        return PreparedSettings(
            settings=normalized,
            runtime=cls.runtime_state(
                normalized,
                state_dir=state_dir,
                normalize_token=normalize_token,
            ),
        )
