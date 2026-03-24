from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Callable

from models import (
    FAVORITES_PATH,
    HISTORY_PATH,
    METADATA_CACHE_PATH,
    OFFLINE_COVERS_MAP_PATH,
    SEARCH_HISTORY_PATH,
    SETTINGS_PATH,
    app_state_dir,
)


def backup_file_mapping(path_overrides: dict[str, str] | None = None) -> list[tuple[str, str]]:
    mapping = [
        ("history.json", HISTORY_PATH),
        ("search_history.json", SEARCH_HISTORY_PATH),
        ("offline_covers.json", OFFLINE_COVERS_MAP_PATH),
        ("settings.json", SETTINGS_PATH),
        ("favorites.json", FAVORITES_PATH),
        ("metadata_cache.json", METADATA_CACHE_PATH),
    ]
    if path_overrides is None:
        return mapping
    ordered: list[tuple[str, str]] = [
        (name, path_overrides[name])
        for name, _path in mapping
        if name in path_overrides
    ]
    extras = sorted(
        (name, path)
        for name, path in path_overrides.items()
        if name not in {key for key, _value in mapping}
    )
    return ordered + extras


def atomic_json_save(path: str, payload: Any) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def load_metadata_cache(path: str = METADATA_CACHE_PATH) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_metadata_cache(cache: dict[str, Any], path: str = METADATA_CACHE_PATH) -> None:
    atomic_json_save(path, cache)


def metadata_cache_key(source: str, identifier: str) -> str:
    return f"{source}:{identifier}"


def metadata_cache_entry(
    *,
    updated_at: float,
    year: Any,
    season: Any,
    studio: Any,
    rating: Any,
) -> dict[str, Any]:
    return {
        "updated_at": float(updated_at),
        "year": year,
        "season": season,
        "studio": studio,
        "rating": rating,
    }


def load_search_history(path: str = SEARCH_HISTORY_PATH) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return []


def save_search_history(queries: list[str], path: str = SEARCH_HISTORY_PATH) -> None:
    atomic_json_save(path, queries)


def clear_search_history(path: str = SEARCH_HISTORY_PATH) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def load_offline_covers_map(
    *,
    path: str = OFFLINE_COVERS_MAP_PATH,
    resolve_legacy_cover_path: Callable[[str], str | None],
) -> tuple[dict[str, str], bool]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}, True
        loaded = {
            str(k): str(v)
            for k, v in data.items()
            if str(k).strip() and str(v).strip()
        }
        changed = False
        for key, value in list(loaded.items()):
            if value.lower().endswith(".img"):
                continue
            legacy_path = resolve_legacy_cover_path(value)
            if legacy_path and os.path.exists(legacy_path):
                loaded[key] = legacy_path
                changed = True
        return loaded, changed
    except FileNotFoundError:
        return {}, True
    except Exception:
        return {}, True


def save_offline_covers_map(mapping: dict[str, str], path: str = OFFLINE_COVERS_MAP_PATH) -> None:
    atomic_json_save(path, mapping)


def remember_offline_cover(
    mapping: dict[str, str],
    *,
    anime_name: str,
    cover_url: str | None,
    safe_name: Callable[[str], str],
    cover_cache_path: Callable[[str], str | None],
) -> bool:
    if not cover_url:
        return False
    cache_path = cover_cache_path(cover_url)
    if not cache_path or not os.path.exists(cache_path):
        return False
    key = safe_name(anime_name).lower()
    if not key:
        return False
    if mapping.get(key) == cache_path:
        return False
    mapping[key] = cache_path
    return True


def write_backup_archive(
    out_path: str,
    *,
    state_dir: str | None = None,
    path_overrides: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    written_files: list[str] = []
    written_covers: list[str] = []
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in backup_file_mapping(path_overrides):
            if os.path.exists(src):
                zf.write(src, arcname=arcname)
                written_files.append(arcname)
        covers_dir = os.path.join(state_dir or app_state_dir(), "covers")
        if os.path.isdir(covers_dir):
            for fn in sorted(os.listdir(covers_dir)):
                src = os.path.join(covers_dir, fn)
                if os.path.isfile(src):
                    zf.write(src, arcname=os.path.join("covers", fn))
                    written_covers.append(fn)
    return tuple(written_files), tuple(written_covers)


def restore_backup_archive(
    in_path: str,
    *,
    state_dir: str | None = None,
    path_overrides: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    restored_files: list[str] = []
    restored_covers: list[str] = []
    with zipfile.ZipFile(in_path, "r") as zf:
        members = set(zf.namelist())
        mapping = dict(backup_file_mapping(path_overrides))
        for arcname, dst in mapping.items():
            if arcname not in members:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(arcname, "r") as src, open(dst, "wb") as out:
                out.write(src.read())
            restored_files.append(arcname)
        base = state_dir or app_state_dir()
        for member in sorted(members):
            if not member.startswith("covers/") or member.endswith("/"):
                continue
            rel = member[len("covers/"):]
            dst = os.path.join(base, "covers", os.path.basename(rel))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(member, "r") as src, open(dst, "wb") as out:
                out.write(src.read())
            restored_covers.append(os.path.basename(rel))
    return tuple(restored_files), tuple(restored_covers)


@dataclass(frozen=True)
class BackupResult:
    archive_path: str
    written_files: tuple[str, ...]
    written_covers: tuple[str, ...]


@dataclass(frozen=True)
class RestoreResult:
    archive_path: str
    restored_files: tuple[str, ...]
    restored_covers: tuple[str, ...]


class AppStateService:
    @staticmethod
    def create_backup_archive(
        out_path: str,
        *,
        state_dir: str | None = None,
        path_overrides: dict[str, str] | None = None,
    ) -> BackupResult:
        target = os.path.abspath(os.path.expanduser(str(out_path or "").strip()))
        if not target:
            raise ValueError("invalid_backup_path")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        written_files, written_covers = write_backup_archive(
            target,
            state_dir=state_dir,
            path_overrides=path_overrides,
        )
        return BackupResult(
            archive_path=target,
            written_files=written_files,
            written_covers=written_covers,
        )

    @staticmethod
    def restore_backup_archive(
        in_path: str,
        *,
        state_dir: str | None = None,
        path_overrides: dict[str, str] | None = None,
    ) -> RestoreResult:
        source = os.path.abspath(os.path.expanduser(str(in_path or "").strip()))
        if not source:
            raise ValueError("invalid_backup_path")
        restored_files, restored_covers = restore_backup_archive(
            source,
            state_dir=state_dir,
            path_overrides=path_overrides,
        )
        return RestoreResult(
            archive_path=source,
            restored_files=restored_files,
            restored_covers=restored_covers,
        )


    @staticmethod
    def backup_file_mapping(path_overrides: dict[str, str] | None = None) -> list[tuple[str, str]]:
        return backup_file_mapping(path_overrides)

    @staticmethod
    def load_metadata_cache(path: str = METADATA_CACHE_PATH) -> dict[str, Any]:
        return load_metadata_cache(path)

    @staticmethod
    def save_metadata_cache(cache: dict[str, Any], path: str = METADATA_CACHE_PATH) -> None:
        save_metadata_cache(cache, path)

    @staticmethod
    def metadata_cache_key(source: str, identifier: str) -> str:
        return metadata_cache_key(source, identifier)

    @staticmethod
    def metadata_cache_entry(
        *,
        updated_at: float,
        year: Any,
        season: Any,
        studio: Any,
        rating: Any,
    ) -> dict[str, Any]:
        return metadata_cache_entry(
            updated_at=updated_at,
            year=year,
            season=season,
            studio=studio,
            rating=rating,
        )

    @staticmethod
    def load_search_history(path: str = SEARCH_HISTORY_PATH) -> list[str]:
        return load_search_history(path)

    @staticmethod
    def save_search_history(queries: list[str], path: str = SEARCH_HISTORY_PATH) -> None:
        save_search_history(queries, path)

    @staticmethod
    def clear_search_history(path: str = SEARCH_HISTORY_PATH) -> None:
        clear_search_history(path)

    @staticmethod
    def load_offline_covers_map(
        *,
        path: str = OFFLINE_COVERS_MAP_PATH,
        resolve_legacy_cover_path: Callable[[str], str | None],
    ) -> tuple[dict[str, str], bool]:
        return load_offline_covers_map(
            path=path,
            resolve_legacy_cover_path=resolve_legacy_cover_path,
        )

    @staticmethod
    def save_offline_covers_map(mapping: dict[str, str], path: str = OFFLINE_COVERS_MAP_PATH) -> None:
        save_offline_covers_map(mapping, path)

    @staticmethod
    def remember_offline_cover(
        mapping: dict[str, str],
        *,
        anime_name: str,
        cover_url: str | None,
        safe_name: Callable[[str], str],
        cover_cache_path: Callable[[str], str | None],
    ) -> bool:
        return remember_offline_cover(
            mapping,
            anime_name=anime_name,
            cover_url=cover_url,
            safe_name=safe_name,
            cover_cache_path=cover_cache_path,
        )
