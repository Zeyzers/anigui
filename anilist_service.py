from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Protocol
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

from components import SearchItem
from models import HistoryEntry


__all__ = [
    "AniListRemoteRow",
    "AniListService",
    "anilist_headers_for_token",
    "anilist_status_for_entry",
    "build_history_entry",
    "build_completed_history_entry",
    "anilist_unmatched_identifier",
    "best_item_for_titles",
    "build_imported_history_entry",
    "build_merged_history_entry",
    "build_planned_history_entry",
    "build_placeholder_history_entry",
    "dedupe_history_entries_prefer_cover",
    "entry_best_progress_and_completed_for_sync",
    "entry_add_watched_ep",
    "entry_apply_episode_progress_update",
    "entry_episode_is_completed",
    "entry_episode_progress_map",
    "entry_get_episode_progress",
    "entry_has_progress",
    "entry_is_series_completed",
    "entry_reconcile_series_state",
    "entry_set_episode_progress",
    "entry_status",
    "watch_status_for_completion",
    "entry_watched_eps_set",
    "episode_is_completed",
    "extract_season_number",
    "history_entry_sync_key",
    "merge_anilist_history_entries",
    "norm_title",
    "normalize_anilist_token",
    "normalize_history_entry",
    "planned_identifier_from_titles",
    "title_search_candidates",
    "watch_status_from_anilist",
]


def normalize_anilist_token(token: str | None) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    if "access_token=" in t:
        try:
            frag = urlsplit(t).fragment or ""
            if frag:
                q = parse_qs(frag)
                val = q.get("access_token", [])
                if val and val[0]:
                    return str(val[0]).strip()
            q = parse_qs(t.replace("#", "&").replace("?", "&"))
            val = q.get("access_token", [])
            if val and val[0]:
                return str(val[0]).strip()
        except Exception:
            pass
    return t


def anilist_headers_for_token(token: str | None) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Anigui/1.0 (+https://anilist.co)",
    }
    t = normalize_anilist_token(token)
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h


def anilist_graphql(
    query: str,
    variables: dict[str, Any],
    *,
    token: str | None = None,
    url: str = "https://graphql.anilist.co",
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers=anilist_headers_for_token(token),
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


def anilist_viewer(token: str) -> tuple[int, str]:
    payload = anilist_graphql(
        """
        query {
          Viewer { id name }
        }
        """,
        {},
        token=token,
    )
    viewer = payload.get("data", {}).get("Viewer") if isinstance(payload.get("data"), dict) else None
    if not isinstance(viewer, dict) or not viewer.get("id"):
        raise RuntimeError("AniList viewer non disponibile.")
    return int(viewer.get("id")), str(viewer.get("name") or "")


def anilist_viewer_name(token: str) -> str:
    _viewer_id, viewer_name = anilist_viewer(token)
    if not viewer_name:
        raise RuntimeError("AniList viewer non disponibile.")
    return viewer_name


def watch_status_from_anilist(remote_status: str) -> str:
    rs = str(remote_status or "").strip().upper()
    if rs == "PLANNING":
        return "Planned"
    if rs == "PAUSED":
        return "Paused"
    if rs == "DROPPED":
        return "Dropped"
    if rs == "COMPLETED":
        return "Completed"
    return "Watching"


def anilist_status_for_entry(entry: HistoryEntry) -> str:
    local_status, in_progress, _seen_count, _last_prog = entry_status(entry)
    if local_status == "Completed" and not in_progress:
        return "COMPLETED"
    if local_status == "Paused":
        return "PAUSED"
    if local_status == "Dropped":
        return "DROPPED"
    if local_status == "Watching":
        return "CURRENT"
    return "PLANNING"


def history_entry_sync_key(entry: HistoryEntry) -> str:
    return f"{entry.provider}:{entry.identifier}:{entry.lang}"


def norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_TITLE_NOISE_WORDS = {
    "a",
    "an",
    "and",
    "da",
    "de",
    "del",
    "di",
    "end",
    "ga",
    "il",
    "la",
    "le",
    "no",
    "of",
    "s",
    "the",
    "to",
}


def title_alias_keys(title: str) -> set[str]:
    raw = str(title or "").strip()
    if not raw:
        return set()
    keys: set[str] = set()
    for candidate in title_search_candidates([raw]) or [raw]:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        normalized = norm_title(candidate)
        if normalized:
            keys.add(normalized)

        for part in re.split(r"[:|/]+", candidate):
            normalized_part = norm_title(part)
            if normalized_part:
                keys.add(normalized_part)

        core_words = [w for w in normalized.split() if w and w not in _TITLE_NOISE_WORDS]
        if 2 <= len(core_words) <= 3:
            tail = core_words[-1]
            if len(tail) >= 6:
                keys.add(tail)
    return keys


def titles_match(title_a: str, title_b: str) -> bool:
    norm_a = norm_title(title_a)
    norm_b = norm_title(title_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    season_a = extract_season_number(title_a)
    season_b = extract_season_number(title_b)
    if season_a is not None and season_b is not None and season_a != season_b:
        return False
    return bool(title_alias_keys(title_a) & title_alias_keys(title_b))


def extract_season_number(title: str) -> int | None:
    s = str(title or "").strip().lower()
    if not s:
        return None
    patterns = [
        r"\b(\d+)(?:st|nd|rd|th)\s+season\b",
        r"\bseason\s*(\d+)\b",
        r"\bs(\d+)\b",
        r"\bpart\s*(\d+)\b",
        r"第\s*(\d+)\s*期",
        r"\b(\d+)\s*기\b",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                if n > 0:
                    return n
            except Exception:
                continue
    return None


def title_search_candidates(titles: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in titles:
        s = str(t or "").strip()
        if not s:
            continue
        variants = [s]
        cleaned = re.sub(r"\b\d+(st|nd|rd|th)\s+season\b", "", s, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(season|stagione)\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bs\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(part|cour)\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:|")
        if cleaned and cleaned.casefold() != s.casefold():
            variants.append(cleaned)
        for v in variants:
            key = v.casefold()
            if key and key not in seen:
                seen.add(key)
                out.append(v)
    return out


def planned_identifier_from_titles(
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


def anilist_unmatched_identifier(
    titles: list[str],
    media_id: int | None = None,
    entry_id: int | None = None,
) -> str:
    if media_id is not None and int(media_id) > 0:
        return f"anilist-only:{int(media_id)}"
    if entry_id is not None and int(entry_id) > 0:
        return f"anilist-only-entry:{int(entry_id)}"
    base = " | ".join([t.strip() for t in titles if str(t).strip()]) or "anilist-only"
    h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"anilist-only:{h}"


def _to_ep_float(ep: float | int | str | None) -> float | None:
    try:
        return float(ep) if ep is not None else None
    except Exception:
        return None


def entry_watched_eps_set(entry: HistoryEntry) -> set[float]:
    out: set[float] = set()
    for v in (entry.watched_eps or []):
        f = _to_ep_float(v)
        if f is not None:
            out.add(f)
    if not out and bool(entry.completed):
        last = _to_ep_float(entry.last_ep)
        if last is not None and last > 0:
            out.add(float(last))
    return out


def _ep_key(ep: float | int | str | None) -> str:
    try:
        f = float(ep)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(ep or "")


def entry_episode_progress_map(entry: HistoryEntry) -> dict[float, dict[str, Any]]:
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


def entry_get_episode_progress(
    entry: HistoryEntry,
    ep: float | int | str | None,
) -> dict[str, Any] | None:
    key = _ep_key(ep)
    raw = entry.episode_progress if isinstance(entry.episode_progress, dict) else {}
    val = raw.get(key)
    return dict(val) if isinstance(val, dict) else None


def entry_add_watched_ep(entry: HistoryEntry, ep: float | int | str | None) -> None:
    f = _to_ep_float(ep)
    if f is None or f <= 0:
        return
    eps = entry_watched_eps_set(entry)
    eps.add(float(f))
    entry.watched_eps = sorted(eps)


def entry_set_episode_progress(
    entry: HistoryEntry,
    ep: float | int | str | None,
    *,
    pos: float,
    dur: float,
    percent: float,
    completed: bool,
    now_ts: float | None = None,
) -> None:
    raw = dict(entry.episode_progress) if isinstance(entry.episode_progress, dict) else {}
    raw[_ep_key(ep)] = {
        "pos": float(max(0.0, pos)),
        "dur": float(max(0.0, dur)),
        "percent": float(max(0.0, min(100.0, percent))),
        "completed": bool(completed),
        "updated_at": float(time() if now_ts is None else now_ts),
    }
    entry.episode_progress = raw
    if bool(completed):
        entry_add_watched_ep(entry, ep)


def entry_apply_episode_progress_update(
    entry: HistoryEntry,
    ep: float | int | str | None,
    *,
    pos: float,
    dur: float,
    percent: float,
    completed: bool,
    now_ts: float | None = None,
) -> None:
    now = float(time() if now_ts is None else now_ts)
    epf = _to_ep_float(ep)
    if epf is not None and epf > 0:
        entry.last_ep = float(epf)
    entry.last_pos = float(max(0.0, pos))
    entry.last_duration = float(max(0.0, dur))
    entry.last_percent = float(max(0.0, min(100.0, percent)))
    entry.updated_at = now
    entry_set_episode_progress(
        entry,
        ep,
        pos=pos,
        dur=dur,
        percent=percent,
        completed=completed,
        now_ts=now,
    )
    entry.watch_status = watch_status_for_completion(entry, bool(getattr(entry, "completed", False)))


def entry_reconcile_series_state(
    entry: HistoryEntry,
    available_eps: list[float | int] | None,
    *,
    now_ts: float | None = None,
) -> tuple[bool, str]:
    completed = entry_is_series_completed(entry, available_eps)
    entry.completed = bool(completed)
    entry.watch_status = watch_status_for_completion(entry, bool(completed))
    if now_ts is not None:
        entry.updated_at = float(now_ts)
    return bool(entry.completed), str(entry.watch_status or "")


def entry_episode_is_completed(
    entry: HistoryEntry,
    ep: float | int | str | None,
) -> bool:
    prog = entry_get_episode_progress(entry, ep)
    if isinstance(prog, dict) and bool(prog.get("completed", False)):
        return True
    epf = _to_ep_float(ep)
    if epf is None:
        return False
    return any(abs(epf - w) < 1e-6 for w in entry_watched_eps_set(entry))


def entry_has_progress(
    entry: HistoryEntry,
    *,
    last_prog: dict[str, Any] | None = None,
) -> bool:
    if entry_watched_eps_set(entry):
        return True
    ep_map = entry_episode_progress_map(entry)
    if ep_map:
        return True
    if isinstance(last_prog, dict):
        lp_pos = float(last_prog.get("pos", 0.0) or 0.0)
        lp_pct = float(last_prog.get("percent", 0.0) or 0.0)
        if lp_pos > 0.0 or lp_pct > 0.0:
            return True
    last_epf = _to_ep_float(entry.last_ep)
    return bool(last_epf is not None and last_epf > 0.0)


def entry_best_progress_and_completed_for_sync(entry: HistoryEntry) -> tuple[int, bool]:
    vals: list[float] = []
    completed_vals: list[float] = []
    try:
        w = sorted(entry_watched_eps_set(entry))
        vals.extend(w)
        completed_vals.extend(w)
    except Exception:
        pass
    last_epf = _to_ep_float(entry.last_ep)
    if last_epf is not None and last_epf > 0.0:
        vals.append(last_epf)
        if bool(entry.completed):
            completed_vals.append(last_epf)
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


def episode_is_completed(
    pos: float | None,
    dur: float | None,
    percent: float | None = None,
) -> bool:
    try:
        if percent is not None and float(percent) >= 97.0:
            return True
    except Exception:
        pass
    try:
        pos_v = float(pos or 0.0)
        dur_v = float(dur or 0.0)
    except Exception:
        return False
    if dur_v <= 0:
        return False
    remaining = max(0.0, dur_v - pos_v)
    return (pos_v / dur_v) >= 0.97 or remaining <= 90.0


def entry_is_series_completed(
    entry: HistoryEntry,
    available_eps: list[float | int] | None,
) -> bool:
    if not available_eps:
        return False
    vals: list[float] = []
    for x in available_eps:
        xf = _to_ep_float(x)
        if xf is not None:
            vals.append(xf)
    if not vals:
        return False
    max_ep = max(vals)
    return entry_episode_is_completed(entry, max_ep)


def entry_status(
    entry: HistoryEntry,
    *,
    last_prog: dict[str, Any] | None = None,
) -> tuple[str, bool, int, dict[str, Any] | None]:
    ep_map = entry_episode_progress_map(entry)
    last_epf = _to_ep_float(entry.last_ep)
    if last_prog is None and last_epf is not None:
        last_prog = ep_map.get(last_epf)
    in_progress = False
    if isinstance(last_prog, dict):
        lp_pos = float(last_prog.get("pos", 0.0) or 0.0)
        lp_pct = float(last_prog.get("percent", 0.0) or 0.0)
        lp_done = bool(last_prog.get("completed", False))
        in_progress = (not lp_done) and (lp_pos > 0.0 or lp_pct > 0.0)
    has_progress = entry_has_progress(entry, last_prog=last_prog)
    seen_count = len(entry_watched_eps_set(entry))
    status_raw = str(getattr(entry, "watch_status", "") or "").strip().casefold()
    status_map = {
        "current": "Watching",
        "watching": "Watching",
        "repeating": "Watching",
        "planning": "Planned",
        "planned": "Planned",
        "paused": "Paused",
        "dropped": "Dropped",
        "completed": "Completed",
    }
    explicit = status_map.get(status_raw, "")
    if bool(entry.completed) and not in_progress:
        return "Completed", in_progress, seen_count, last_prog
    if explicit in {"Paused", "Dropped"}:
        return explicit, in_progress, seen_count, last_prog
    if in_progress or has_progress:
        return "Watching", in_progress, seen_count, last_prog
    if explicit == "Watching":
        return "Watching", in_progress, seen_count, last_prog
    if explicit == "Planned":
        return "Planned", in_progress, seen_count, last_prog
    return "Planned", in_progress, seen_count, last_prog


def watch_status_for_completion(entry: HistoryEntry, completed: bool) -> str:
    if bool(completed):
        return "Completed"
    explicit = str(getattr(entry, "watch_status", "") or "").strip().casefold()
    if explicit == "paused":
        return "Paused"
    if explicit == "dropped":
        return "Dropped"
    if entry_has_progress(entry):
        return "Watching"
    return "Planned"


def _episode_progress_key(epf: float) -> str:
    return str(int(epf)) if float(epf).is_integer() else str(epf)


def _seed_completed_progress(
    progress: int | float,
    *,
    now_ts: float,
) -> tuple[float, list[float], dict[str, dict[str, Any]], float]:
    try:
        progress_i = max(0, int(progress))
    except Exception:
        progress_i = 0
    prog = float(progress_i)
    watched_eps: list[float] = []
    episode_progress: dict[str, dict[str, Any]] = {}
    if progress_i > 0:
        watched_eps = [float(i) for i in range(1, progress_i + 1)] if progress_i <= 400 else [prog]
        for epf in watched_eps[:400]:
            episode_progress[_episode_progress_key(epf)] = {
                "pos": 0.0,
                "dur": 0.0,
                "percent": 100.0,
                "completed": True,
                "updated_at": now_ts,
            }
    return prog, watched_eps, episode_progress, (100.0 if progress_i > 0 else 0.0)


def normalize_history_entry(
    entry: HistoryEntry,
    *,
    fallback_provider: str = "",
    fallback_identifier: str = "",
    fallback_name: str = "",
    fallback_lang: str = "SUB",
    fallback_cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    now = float(time() if now_ts is None else now_ts)
    provider = str(getattr(entry, "provider", "") or fallback_provider or "")
    identifier = str(getattr(entry, "identifier", "") or fallback_identifier or "")
    name = str(getattr(entry, "name", "") or fallback_name or identifier or "Unknown anime")
    lang = str(getattr(entry, "lang", "") or fallback_lang or "SUB")
    cover_url = getattr(entry, "cover_url", None) or fallback_cover_url

    watched_eps = sorted(entry_watched_eps_set(entry))
    ep_map_raw = entry_episode_progress_map(entry)
    episode_progress: dict[str, dict[str, Any]] = {}
    max_ep = 0.0
    for epf, raw_prog in ep_map_raw.items():
        try:
            epf = float(epf)
        except Exception:
            continue
        max_ep = max(max_ep, epf)
        prog = raw_prog if isinstance(raw_prog, dict) else {}
        completed = bool(prog.get("completed", False))
        percent_default = 100.0 if completed else 0.0
        episode_progress[_episode_progress_key(epf)] = {
            "pos": float(prog.get("pos", 0.0) or 0.0),
            "dur": float(prog.get("dur", 0.0) or 0.0),
            "percent": float(prog.get("percent", percent_default) or 0.0),
            "completed": completed,
            "updated_at": float(prog.get("updated_at", now) or now),
        }

    for epf in watched_eps:
        try:
            epf = float(epf)
        except Exception:
            continue
        max_ep = max(max_ep, epf)
        key = _episode_progress_key(epf)
        if key not in episode_progress:
            episode_progress[key] = {
                "pos": 0.0,
                "dur": 0.0,
                "percent": 100.0,
                "completed": True,
                "updated_at": now,
            }

    try:
        last_ep = float(getattr(entry, "last_ep", 0.0) or 0.0)
    except Exception:
        last_ep = 0.0
    if max_ep > 0.0:
        last_ep = max(last_ep, max_ep)
    last_pos = float(getattr(entry, "last_pos", 0.0) or 0.0)
    last_duration = float(getattr(entry, "last_duration", 0.0) or 0.0)
    last_percent = float(getattr(entry, "last_percent", 0.0) or 0.0)
    completed = bool(getattr(entry, "completed", False))

    normalized = HistoryEntry(
        provider=provider,
        identifier=identifier,
        name=name,
        lang=lang,
        last_ep=last_ep,
        updated_at=float(getattr(entry, "updated_at", now) or now),
        cover_url=cover_url,
        last_pos=last_pos,
        last_duration=last_duration,
        last_percent=last_percent,
        completed=completed,
        watch_status=str(getattr(entry, "watch_status", "") or ""),
        watched_eps=watched_eps,
        episode_progress=episode_progress,
    )
    normalized.watch_status = watch_status_for_completion(normalized, completed)
    return normalized


def build_history_entry(
    *,
    provider_name: str,
    identifier: str,
    name: str,
    lang_name: str,
    progress: int = 0,
    last_ep: float | None = None,
    last_pos: float = 0.0,
    last_duration: float = 0.0,
    last_percent: float | None = None,
    completed: bool = False,
    watch_status: str = "",
    watched_eps: list[float] | list[int] | None = None,
    episode_progress: dict[str, dict[str, Any]] | None = None,
    cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    now = float(time() if now_ts is None else now_ts)
    prog, seeded_watched_eps, seeded_episode_progress, seeded_last_percent = _seed_completed_progress(progress, now_ts=now)
    entry = HistoryEntry(
        provider=provider_name,
        identifier=identifier,
        name=name,
        lang=lang_name,
        last_ep=float(prog if last_ep is None else last_ep),
        updated_at=now,
        cover_url=cover_url,
        last_pos=float(last_pos or 0.0),
        last_duration=float(last_duration or 0.0),
        last_percent=float(seeded_last_percent if last_percent is None else last_percent),
        completed=bool(completed),
        watch_status=watch_status,
        watched_eps=list(seeded_watched_eps if watched_eps is None else watched_eps),
        episode_progress=dict(seeded_episode_progress if episode_progress is None else episode_progress),
    )
    return normalize_history_entry(
        entry,
        fallback_provider=provider_name,
        fallback_identifier=identifier,
        fallback_name=name,
        fallback_lang=lang_name,
        fallback_cover_url=cover_url,
        now_ts=now,
    )


def build_completed_history_entry(
    *,
    provider_name: str,
    identifier: str,
    name: str,
    lang_name: str,
    watched_eps: list[float] | list[int],
    cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    watched_set: set[float] = set()
    for raw_ep in watched_eps:
        try:
            ep = float(raw_ep)
        except Exception:
            continue
        if ep > 0:
            watched_set.add(ep)
    watched = sorted(watched_set)
    last_ep = max(watched) if watched else 0.0
    return build_history_entry(
        provider_name=provider_name,
        identifier=identifier,
        name=name,
        lang_name=lang_name,
        last_ep=last_ep,
        last_percent=100.0 if watched else 0.0,
        completed=bool(watched),
        watched_eps=watched,
        cover_url=cover_url,
        now_ts=now_ts,
    )


def build_placeholder_history_entry(
    *,
    provider_name: str,
    lang_name: str,
    titles: list[str],
    media_id: int = 0,
    entry_id: int = 0,
    progress: int = 0,
    remote_status: str = "PLANNING",
    local_status: str | None = None,
    cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    fallback_name = next((t for t in titles if str(t).strip()), f"AniList #{media_id}" if media_id > 0 else "Planned anime")
    placeholder_id = planned_identifier_from_titles(titles, media_id=media_id, entry_id=entry_id)
    return build_history_entry(
        provider_name=provider_name,
        identifier=placeholder_id,
        name=fallback_name,
        lang_name=lang_name,
        progress=progress,
        completed=(remote_status == "COMPLETED"),
        watch_status=local_status or watch_status_from_anilist(remote_status),
        cover_url=cover_url,
        now_ts=now_ts,
    )


def build_planned_history_entry(
    *,
    provider_name: str,
    lang_name: str,
    titles: list[str],
    media_id: int = 0,
    entry_id: int = 0,
    cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    return build_placeholder_history_entry(
        provider_name=provider_name,
        lang_name=lang_name,
        titles=titles,
        media_id=media_id,
        entry_id=entry_id,
        progress=0,
        remote_status="PLANNING",
        local_status="Planned",
        cover_url=cover_url,
        now_ts=now_ts,
    )


def build_imported_history_entry(
    *,
    provider_name: str,
    lang_name: str,
    found: SearchItem,
    titles: list[str],
    progress: int,
    remote_status: str,
    local_status: str | None = None,
    cover_url: str | None = None,
    now_ts: float | None = None,
) -> HistoryEntry:
    fallback_name = found.name or next((t for t in titles if str(t).strip()), "Imported anime")
    return build_history_entry(
        provider_name=provider_name,
        identifier=found.identifier,
        name=fallback_name,
        lang_name=lang_name,
        progress=progress,
        completed=(remote_status == "COMPLETED"),
        watch_status=local_status or watch_status_from_anilist(remote_status),
        cover_url=found.cover_url or cover_url,
        now_ts=now_ts,
    )


def build_merged_history_entry(
    existing: HistoryEntry,
    incoming: HistoryEntry,
    *,
    now_ts: float | None = None,
) -> HistoryEntry:
    now = float(time() if now_ts is None else now_ts)
    try:
        existing_ep = float(existing.last_ep)
    except Exception:
        existing_ep = 0.0
    try:
        incoming_ep = float(incoming.last_ep)
    except Exception:
        incoming_ep = 0.0
    episode_progress = dict(getattr(existing, "episode_progress", {}) or {})
    for k, v in (getattr(incoming, "episode_progress", {}) or {}).items():
        if k not in episode_progress:
            episode_progress[k] = v
    return build_history_entry(
        provider_name=existing.provider or incoming.provider,
        identifier=existing.identifier or incoming.identifier,
        name=existing.name or incoming.name,
        lang_name=existing.lang or incoming.lang,
        last_ep=max(existing_ep, incoming_ep),
        last_pos=float(getattr(existing, "last_pos", 0.0) or 0.0),
        last_duration=float(getattr(existing, "last_duration", 0.0) or 0.0),
        last_percent=max(
            float(getattr(existing, "last_percent", 0.0) or 0.0),
            float(getattr(incoming, "last_percent", 0.0) or 0.0),
        ),
        completed=bool(existing.completed or incoming.completed),
        watch_status=str(
            getattr(incoming, "watch_status", "")
            or getattr(existing, "watch_status", "")
            or ""
        ),
        watched_eps=sorted(entry_watched_eps_set(existing) | entry_watched_eps_set(incoming)),
        episode_progress=episode_progress,
        cover_url=existing.cover_url or incoming.cover_url,
        now_ts=now,
    )


def best_item_for_titles(
    titles: list[str],
    items: list[SearchItem],
) -> SearchItem | None:
    if not items:
        return None
    norm_titles = [norm_title(t) for t in titles if t and norm_title(t)]
    if not norm_titles:
        return items[0]
    target_season: int | None = None
    for t in titles:
        sn = extract_season_number(t)
        if sn is not None:
            target_season = sn
            break

    best: SearchItem | None = None
    best_score = -10**9
    for it in items:
        name_norm = norm_title(it.name)
        if not name_norm:
            continue
        score = 0
        for t in norm_titles:
            if name_norm == t:
                score = max(score, 100)
            elif t in name_norm or name_norm in t:
                score = max(score, 65)
            else:
                tset = set(t.split())
                nset = set(name_norm.split())
                if tset and nset:
                    overlap = len(tset & nset)
                    if overlap > 0:
                        ratio = overlap / max(len(tset), len(nset))
                        score = max(score, int(ratio * 70))

        if target_season is not None and target_season > 1:
            cand_season = extract_season_number(it.name)
            if cand_season is None:
                score -= 35
            elif cand_season != target_season:
                score -= 90

        if score > best_score:
            best_score = score
            best = it

    if best is None:
        return None
    if best_score < 45:
        return None
    return best


def _status_rank(e: HistoryEntry) -> int:
    s = str(getattr(e, "watch_status", "") or "").strip().casefold()
    if bool(e.completed) or s == "completed":
        return 5
    if s == "watching":
        return 4
    if s == "paused":
        return 3
    if s == "planned":
        return 2
    if s == "dropped":
        return 1
    return 0


def _progress_rank(e: HistoryEntry) -> tuple[float, int]:
    try:
        last_ep = float(getattr(e, "last_ep", 0.0) or 0.0)
    except Exception:
        last_ep = 0.0
    seen = len(getattr(e, "watched_eps", []) or [])
    return last_ep, seen


def _engagement_rank(e: HistoryEntry) -> tuple[int, float, int]:
    pr = _progress_rank(e)
    return _status_rank(e), pr[0], pr[1]


def dedupe_history_entries_prefer_cover(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    grouped: dict[tuple[str, str, str], HistoryEntry] = {}
    passthrough: list[HistoryEntry] = []
    for e in entries:
        provider = str(getattr(e, "provider", "") or "")
        lang = str(getattr(e, "lang", "") or "")
        name_norm = norm_title(str(getattr(e, "name", "") or ""))
        if not name_norm:
            passthrough.append(e)
            continue
        key = (provider, lang, name_norm)

        prev = grouped.get(key)
        if prev is None:
            grouped[key] = e
            continue

        prev_has_cover = bool(str(getattr(prev, "cover_url", "") or "").strip())
        cur_has_cover = bool(str(getattr(e, "cover_url", "") or "").strip())
        choose_cur = False
        prev_eng = _engagement_rank(prev)
        cur_eng = _engagement_rank(e)
        if cur_eng != prev_eng:
            choose_cur = cur_eng > prev_eng
        elif cur_has_cover != prev_has_cover:
            choose_cur = cur_has_cover
        else:
            choose_cur = float(getattr(e, "updated_at", 0.0) or 0.0) > float(
                getattr(prev, "updated_at", 0.0) or 0.0
            )
        if choose_cur:
            grouped[key] = e

    out = list(grouped.values()) + passthrough
    out.sort(key=lambda x: float(getattr(x, "updated_at", 0.0) or 0.0), reverse=True)
    return out


@dataclass(frozen=True)
class AniListRemoteRow:
    status: str
    progress: int
    titles: list[str]
    media_id: int = 0
    entry_id: int = 0


class ProviderSearch(Protocol):
    def __call__(self, query: str, strict_lang: bool = True) -> list[SearchItem]: ...


def _search_languages(search_lang: Any) -> set[Any]:
    if isinstance(search_lang, (set, frozenset, list, tuple)):
        return set(search_lang)
    return {search_lang}


def _row_planned_key(row: AniListRemoteRow) -> str:
    if row.media_id > 0:
        return f"m:{row.media_id}"
    if row.entry_id > 0:
        return f"e:{row.entry_id}"
    return f"t:{planned_identifier_from_titles(row.titles)}"


def merge_anilist_history_entries(
    rows: list[AniListRemoteRow],
    local_history: list[HistoryEntry],
    *,
    provider_name: str,
    lang_name: str,
    search_provider: ProviderSearch,
    search_lang: Any,
    now_ts: float | None = None,
) -> dict[str, Any]:
    now = float(time() if now_ts is None else now_ts)
    history_name_idx: dict[str, HistoryEntry] = {}
    for e in local_history:
        if e.provider != provider_name or e.lang != lang_name:
            continue
        n = norm_title(e.name)
        if not n:
            continue
        prev = history_name_idx.get(n)
        if prev is None or float(getattr(prev, "updated_at", 0.0) or 0.0) < float(getattr(e, "updated_at", 0.0) or 0.0):
            history_name_idx[n] = e

    seen_ident: set[str] = set()
    built: list[HistoryEntry] = []
    cleanup_placeholder_keys: set[tuple[str, str, str]] = set()
    skipped = 0
    remote_planned = 0
    imported_planned = 0
    planned_all_keys: list[str] = []
    planned_all_seen: set[str] = set()
    planned_imported_keys: set[str] = set()

    search_cache: dict[tuple[str, bool], list[SearchItem]] = {}

    def provider_search_cached(query: str, strict_lang: bool = True) -> list[SearchItem]:
        key = ((query or "").strip().casefold(), bool(strict_lang))
        if not key[0]:
            return []
        if key in search_cache:
            return search_cache[key]
        try:
            items = search_provider(query, strict_lang=strict_lang)
        except Exception:
            items = []
        search_cache[key] = items
        return items

    for row in rows:
        titles = list(row.titles or [])
        search_queries = title_search_candidates(titles)
        progress = int(row.progress or 0)
        media_id = int(row.media_id or 0)
        entry_id = int(row.entry_id or 0)
        remote_status = str(row.status or "").upper()
        local_status = watch_status_from_anilist(remote_status)
        planned_key = ""
        if remote_status == "PLANNING":
            planned_key = _row_planned_key(row)
            remote_planned += 1
            if planned_key and planned_key not in planned_all_seen:
                planned_all_seen.add(planned_key)
                planned_all_keys.append(planned_key)

        found: SearchItem | None = None
        for t in titles:
            nt = norm_title(t)
            if not nt:
                continue
            existing_local = history_name_idx.get(nt)
            if (
                existing_local is not None
                and not str(existing_local.identifier).startswith("planned:")
                and not str(existing_local.identifier).startswith("anilist-only:")
            ):
                found = SearchItem(
                    name=existing_local.name,
                    identifier=existing_local.identifier,
                    languages=_search_languages(search_lang),
                    cover_url=existing_local.cover_url,
                    source=provider_name,
                    raw=None,
                )
                break
        for t in search_queries:
            if found is not None:
                break
            items = provider_search_cached(t, strict_lang=True)
            found = best_item_for_titles(titles, items)
        if found is None:
            for t in search_queries[:1]:
                if found is not None:
                    break
                items = provider_search_cached(t, strict_lang=False)
                found = best_item_for_titles(titles, items)

        if found is None:
            placeholder = build_placeholder_history_entry(
                provider_name=provider_name,
                lang_name=lang_name,
                titles=titles,
                media_id=media_id,
                entry_id=entry_id,
                progress=progress,
                remote_status=remote_status,
                local_status=local_status,
                now_ts=now,
            )
            built.append(placeholder)
            skipped += 1
            continue

        uniq = f"{provider_name}:{found.identifier}:{lang_name}"
        if uniq in seen_ident:
            if remote_status == "PLANNING":
                placeholder = build_placeholder_history_entry(
                    provider_name=provider_name,
                    lang_name=lang_name,
                    titles=titles,
                    media_id=media_id,
                    entry_id=entry_id,
                    progress=0,
                    remote_status=remote_status,
                    local_status=local_status,
                    now_ts=now,
                )
                puniq = f"{provider_name}:{placeholder.identifier}:{lang_name}"
                if puniq not in seen_ident:
                    seen_ident.add(puniq)
                    built.append(placeholder)
                imported_planned += 1
                if planned_key:
                    planned_imported_keys.add(planned_key)
            continue

        seen_ident.add(uniq)
        placeholder_ids: list[str] = []
        if media_id > 0:
            placeholder_ids.append(f"planned:anilist:{media_id}")
            placeholder_ids.append(f"anilist-only:{media_id}")
        if entry_id > 0:
            placeholder_ids.append(f"planned:anilist-entry:{entry_id}")
            placeholder_ids.append(f"anilist-only-entry:{entry_id}")
        placeholder_ids.append(planned_identifier_from_titles(titles))
        placeholder_ids.append(anilist_unmatched_identifier(titles))
        for pid in placeholder_ids:
            if pid:
                cleanup_placeholder_keys.add((provider_name, pid, lang_name))

        built.append(
            build_imported_history_entry(
                provider_name=provider_name,
                lang_name=lang_name,
                found=found,
                titles=titles,
                progress=progress,
                remote_status=remote_status,
                local_status=local_status,
                now_ts=now,
            )
        )
        if remote_status == "PLANNING":
            imported_planned += 1
            if planned_key:
                planned_imported_keys.add(planned_key)

    planned_missing_keys = [k for k in planned_all_keys if k not in planned_imported_keys]
    existing_map: dict[tuple[str, str, str], HistoryEntry] = {
        (e.provider, e.identifier, e.lang): e for e in local_history
    }
    skipped_local = 0
    imported = 0

    for raw_key in cleanup_placeholder_keys:
        try:
            p, i, l = raw_key
            existing_map.pop((str(p), str(i), str(l)), None)
        except Exception:
            continue

    for incoming in built:
        key = (incoming.provider, incoming.identifier, incoming.lang)
        existing = existing_map.get(key)
        if existing is None:
            existing_map[key] = incoming
            imported += 1
            continue

        try:
            existing_ep = float(existing.last_ep)
        except Exception:
            existing_ep = 0.0
        try:
            incoming_ep = float(incoming.last_ep)
        except Exception:
            incoming_ep = 0.0
        if existing_ep > incoming_ep:
            skipped_local += 1
            continue

        existing_map[key] = build_merged_history_entry(existing, incoming, now_ts=now)
        imported += 1

    merged_history = dedupe_history_entries_prefer_cover(list(existing_map.values()))
    return {
        "built": built,
        "cleanup_placeholder_keys": list(cleanup_placeholder_keys),
        "merged_history": merged_history,
        "imported": imported,
        "skipped_local": skipped_local,
        "remote_planned": remote_planned,
        "imported_planned": imported_planned,
        "planned_missing_keys": planned_missing_keys,
        "skipped": skipped,
    }


def fetch_progress_entries(token: str) -> list[AniListRemoteRow]:
    user_id, _viewer_name = anilist_viewer(token)
    payload = anilist_graphql(
        """
        query ($userId: Int) {
          MediaListCollection(
            userId: $userId,
            type: ANIME,
            status_in: [CURRENT, REPEATING, COMPLETED, PAUSED, DROPPED, PLANNING]
          ) {
            lists {
              status
              entries {
                id
                progress
                media {
                  id
                  synonyms
                  title { romaji english native }
                }
              }
            }
          }
        }
        """,
        {"userId": user_id},
        token=token,
    )
    coll = payload.get("data", {}).get("MediaListCollection") if isinstance(payload.get("data"), dict) else None
    lists = coll.get("lists", []) if isinstance(coll, dict) else []
    out: list[AniListRemoteRow] = []
    for lst in lists:
        if not isinstance(lst, dict):
            continue
        status = str(lst.get("status", "") or "")
        for entry in lst.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            entry_id = int(entry.get("id") or 0) if isinstance(entry.get("id"), (int, float, str)) else 0
            progress = int(entry.get("progress") or 0)
            media = entry.get("media") if isinstance(entry.get("media"), dict) else {}
            media_id = int(media.get("id") or 0) if isinstance(media.get("id"), (int, float, str)) else 0
            title = media.get("title") if isinstance(media.get("title"), dict) else {}
            titles = [
                str(title.get("english") or "").strip(),
                str(title.get("romaji") or "").strip(),
                str(title.get("native") or "").strip(),
            ]
            for syn in (media.get("synonyms") or []):
                s = str(syn or "").strip()
                if s:
                    titles.append(s)
            titles = [t for t in titles if t]
            if not titles and media_id > 0:
                titles = [f"AniList #{media_id}"]
            if not titles and entry_id > 0:
                titles = [f"AniList entry #{entry_id}"]
            if not titles:
                titles = ["Planned anime"]
            out.append(
                AniListRemoteRow(
                    status=status,
                    progress=progress,
                    titles=titles,
                    media_id=media_id,
                    entry_id=entry_id,
                )
            )
    return out


def save_media_list_entry(
    *,
    anime_name: str,
    progress: int,
    status: str,
    token: str | None = None,
    media_id_cache: dict[str, int] | None = None,
) -> dict[str, Any]:
    cache = media_id_cache if media_id_cache is not None else {}
    cache_key = norm_title(anime_name)
    media_id = cache.get(cache_key)
    if media_id is None:
        payload = anilist_graphql(
            """
            query ($search: String) {
              Media(search: $search, type: ANIME) { id }
            }
            """,
            {"search": anime_name},
            token=token,
        )
        media = payload.get("data", {}).get("Media") if isinstance(payload.get("data"), dict) else None
        if not isinstance(media, dict) or not media.get("id"):
            raise RuntimeError(f"AniList match non trovato: {anime_name}")
        media_id = int(media["id"])
        cache[cache_key] = media_id

    payload = anilist_graphql(
        """
        mutation ($mediaId: Int, $status: MediaListStatus, $progress: Int) {
          SaveMediaListEntry(mediaId: $mediaId, status: $status, progress: $progress) {
            id
            status
            progress
          }
        }
        """,
        {
            "mediaId": int(media_id),
            "status": status,
            "progress": int(max(0 if str(status).upper() in {"PLANNING", "PAUSED", "DROPPED"} else 1, progress)),
        },
        token=token,
    )
    return payload.get("data", {})


class AniListService:
    @staticmethod
    def normalize_token(token: str | None) -> str:
        return normalize_anilist_token(token)

    @staticmethod
    def headers_for_token(token: str | None) -> dict[str, str]:
        return anilist_headers_for_token(token)

    @staticmethod
    def graphql(query: str, variables: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        return anilist_graphql(query, variables, token=token)

    @staticmethod
    def viewer(token: str) -> tuple[int, str]:
        return anilist_viewer(token)

    @staticmethod
    def viewer_name(token: str) -> str:
        return anilist_viewer_name(token)

    @staticmethod
    def watch_status_from_remote(remote_status: str) -> str:
        return watch_status_from_anilist(remote_status)

    @staticmethod
    def status_for_entry(entry: HistoryEntry) -> str:
        return anilist_status_for_entry(entry)

    @staticmethod
    def sync_key(entry: HistoryEntry) -> str:
        return history_entry_sync_key(entry)

    @staticmethod
    def normalize_title(title: str) -> str:
        return norm_title(title)

    @staticmethod
    def extract_season_number(title: str) -> int | None:
        return extract_season_number(title)

    @staticmethod
    def title_search_candidates(titles: list[str]) -> list[str]:
        return title_search_candidates(titles)

    @staticmethod
    def title_matches(title_a: str, title_b: str) -> bool:
        return titles_match(title_a, title_b)

    @staticmethod
    def pick_best_item_for_titles(titles: list[str], items: list[SearchItem]) -> SearchItem | None:
        return best_item_for_titles(titles, items)

    @staticmethod
    def planned_identifier(titles: list[str], media_id: int | None = None, entry_id: int | None = None) -> str:
        return planned_identifier_from_titles(titles, media_id=media_id, entry_id=entry_id)

    @staticmethod
    def unmatched_identifier(titles: list[str], media_id: int | None = None, entry_id: int | None = None) -> str:
        return anilist_unmatched_identifier(titles, media_id=media_id, entry_id=entry_id)

    @staticmethod
    def watched_eps_set(entry: HistoryEntry) -> set[float]:
        return entry_watched_eps_set(entry)

    @staticmethod
    def episode_progress_map(entry: HistoryEntry) -> dict[float, dict[str, Any]]:
        return entry_episode_progress_map(entry)

    @staticmethod
    def episode_progress(entry: HistoryEntry, ep: float | int | str | None) -> dict[str, Any] | None:
        return entry_get_episode_progress(entry, ep)

    @staticmethod
    def add_watched_ep(entry: HistoryEntry, ep: float | int | str | None) -> None:
        entry_add_watched_ep(entry, ep)

    @staticmethod
    def set_episode_progress(
        entry: HistoryEntry,
        ep: float | int | str | None,
        *,
        pos: float,
        dur: float,
        percent: float,
        completed: bool,
        now_ts: float | None = None,
    ) -> None:
        entry_set_episode_progress(
            entry,
            ep,
            pos=pos,
            dur=dur,
            percent=percent,
            completed=completed,
            now_ts=now_ts,
        )

    @staticmethod
    def apply_episode_progress_update(
        entry: HistoryEntry,
        ep: float | int | str | None,
        *,
        pos: float,
        dur: float,
        percent: float,
        completed: bool,
        now_ts: float | None = None,
    ) -> None:
        entry_apply_episode_progress_update(
            entry,
            ep,
            pos=pos,
            dur=dur,
            percent=percent,
            completed=completed,
            now_ts=now_ts,
        )

    @staticmethod
    def episode_completed(entry: HistoryEntry, ep: float | int | str | None) -> bool:
        return entry_episode_is_completed(entry, ep)

    @staticmethod
    def has_progress(entry: HistoryEntry, *, last_prog: dict[str, Any] | None = None) -> bool:
        return entry_has_progress(entry, last_prog=last_prog)

    @staticmethod
    def best_progress_and_completed(entry: HistoryEntry) -> tuple[int, bool]:
        return entry_best_progress_and_completed_for_sync(entry)

    @staticmethod
    def episode_is_completed(pos: float | None, dur: float | None, percent: float | None = None) -> bool:
        return episode_is_completed(pos, dur, percent)

    @staticmethod
    def series_completed(entry: HistoryEntry, available_eps: list[float | int] | None) -> bool:
        return entry_is_series_completed(entry, available_eps)

    @staticmethod
    def reconcile_series_state(
        entry: HistoryEntry,
        available_eps: list[float | int] | None,
        *,
        now_ts: float | None = None,
    ) -> tuple[bool, str]:
        return entry_reconcile_series_state(entry, available_eps, now_ts=now_ts)

    @staticmethod
    def entry_status(entry: HistoryEntry, *, last_prog: dict[str, Any] | None = None) -> tuple[str, bool, int, dict[str, Any] | None]:
        return entry_status(entry, last_prog=last_prog)

    @staticmethod
    def watch_status_for_completion(entry: HistoryEntry, completed: bool) -> str:
        return watch_status_for_completion(entry, completed)

    @staticmethod
    def normalize_entry(
        entry: HistoryEntry,
        *,
        fallback_provider: str = "",
        fallback_identifier: str = "",
        fallback_name: str = "",
        fallback_lang: str = "SUB",
        fallback_cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return normalize_history_entry(
            entry,
            fallback_provider=fallback_provider,
            fallback_identifier=fallback_identifier,
            fallback_name=fallback_name,
            fallback_lang=fallback_lang,
            fallback_cover_url=fallback_cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_history_entry(
        *,
        provider_name: str,
        identifier: str,
        name: str,
        lang_name: str,
        progress: int = 0,
        last_ep: float | None = None,
        last_pos: float = 0.0,
        last_duration: float = 0.0,
        last_percent: float | None = None,
        completed: bool = False,
        watch_status: str = "",
        watched_eps: list[float] | list[int] | None = None,
        episode_progress: dict[str, dict[str, Any]] | None = None,
        cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return build_history_entry(
            provider_name=provider_name,
            identifier=identifier,
            name=name,
            lang_name=lang_name,
            progress=progress,
            last_ep=last_ep,
            last_pos=last_pos,
            last_duration=last_duration,
            last_percent=last_percent,
            completed=completed,
            watch_status=watch_status,
            watched_eps=watched_eps,
            episode_progress=episode_progress,
            cover_url=cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_completed_entry(
        *,
        provider_name: str,
        identifier: str,
        name: str,
        lang_name: str,
        watched_eps: list[float] | list[int],
        cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return build_completed_history_entry(
            provider_name=provider_name,
            identifier=identifier,
            name=name,
            lang_name=lang_name,
            watched_eps=watched_eps,
            cover_url=cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_placeholder_entry(
        *,
        provider_name: str,
        lang_name: str,
        titles: list[str],
        media_id: int = 0,
        entry_id: int = 0,
        progress: int = 0,
        remote_status: str = "PLANNING",
        local_status: str | None = None,
        cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return build_placeholder_history_entry(
            provider_name=provider_name,
            lang_name=lang_name,
            titles=titles,
            media_id=media_id,
            entry_id=entry_id,
            progress=progress,
            remote_status=remote_status,
            local_status=local_status,
            cover_url=cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_planned_entry(
        *,
        provider_name: str,
        lang_name: str,
        titles: list[str],
        media_id: int = 0,
        entry_id: int = 0,
        cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return build_planned_history_entry(
            provider_name=provider_name,
            lang_name=lang_name,
            titles=titles,
            media_id=media_id,
            entry_id=entry_id,
            cover_url=cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_imported_entry(
        *,
        provider_name: str,
        lang_name: str,
        found: SearchItem,
        titles: list[str],
        progress: int,
        remote_status: str,
        local_status: str | None = None,
        cover_url: str | None = None,
        now_ts: float | None = None,
    ) -> HistoryEntry:
        return build_imported_history_entry(
            provider_name=provider_name,
            lang_name=lang_name,
            found=found,
            titles=titles,
            progress=progress,
            remote_status=remote_status,
            local_status=local_status,
            cover_url=cover_url,
            now_ts=now_ts,
        )

    @staticmethod
    def build_merged_entry(existing: HistoryEntry, incoming: HistoryEntry, *, now_ts: float | None = None) -> HistoryEntry:
        return build_merged_history_entry(existing, incoming, now_ts=now_ts)

    @staticmethod
    def dedupe_history(entries: list[HistoryEntry]) -> list[HistoryEntry]:
        return dedupe_history_entries_prefer_cover(entries)

    @staticmethod
    def merge_history_entries(
        rows: list[AniListRemoteRow],
        local_history: list[HistoryEntry],
        *,
        provider_name: str,
        lang_name: str,
        search_provider: ProviderSearch,
        search_lang: Any,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        return merge_anilist_history_entries(
            rows,
            local_history,
            provider_name=provider_name,
            lang_name=lang_name,
            search_provider=search_provider,
            search_lang=search_lang,
            now_ts=now_ts,
        )

    @staticmethod
    def fetch_progress_entries(token: str) -> list[AniListRemoteRow]:
        return fetch_progress_entries(token)

    @staticmethod
    def save_media_list_entry(
        *,
        anime_name: str,
        progress: int,
        status: str,
        token: str | None = None,
        media_id_cache: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return save_media_list_entry(
            anime_name=anime_name,
            progress=progress,
            status=status,
            token=token,
            media_id_cache=media_id_cache,
        )
