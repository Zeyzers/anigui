from __future__ import annotations

from dataclasses import dataclass

from anilist_service import (
    AniListService,
    entry_episode_is_completed,
)
from models import HistoryEntry


@dataclass(frozen=True)
class ProgressSaveResult:
    entry: HistoryEntry
    pos: float
    dur: float
    percent: float


@dataclass(frozen=True)
class ResumePlan:
    target_ep: float | int | str | None
    should_seek_resume: bool
    resume_marker_pos: float | None
    seek_pos: float | None
    seek_ratio: float | None


@dataclass(frozen=True)
class ReconcileResult:
    entry: HistoryEntry
    completed: bool
    watch_status: str


def _to_ep_float(ep: float | int | str | None) -> float | None:
    try:
        return float(ep) if ep is not None else None
    except Exception:
        return None


def find_history_entry(
    entries: list[HistoryEntry],
    *,
    provider: str,
    identifier: str,
    lang: str,
    name: str | None = None,
) -> HistoryEntry | None:
    for entry in entries:
        if entry.provider == provider and entry.identifier == identifier and entry.lang == lang:
            return entry
    lookup_name = str(name or "").strip()
    if not lookup_name:
        return None
    for entry in entries:
        if entry.lang != lang:
            continue
        if AniListService.title_matches(entry.name, lookup_name):
            return entry
    return None


def build_saved_progress_entry(
    *,
    provider_name: str,
    identifier: str,
    name: str,
    lang_name: str,
    current_ep: float | int,
    pos_raw: Any,
    dur_raw: Any,
    percent_raw: Any,
    prev_entry: HistoryEntry | None,
    cover_url: str | None,
    now_ts: float,
) -> ProgressSaveResult | None:
    if dur_raw is not None:
        dur = max(0.0, float(dur_raw))
    elif prev_entry is not None:
        dur = max(0.0, float(getattr(prev_entry, "last_duration", 0.0) or 0.0))
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
        pos = max(0.0, float(getattr(prev_entry, "last_pos", 0.0) or 0.0))

    if pos is None and percent is None:
        return None
    if pos is None:
        pos = max(0.0, float(getattr(prev_entry, "last_pos", 0.0) or 0.0)) if prev_entry is not None else 0.0
    if dur is None:
        dur = 0.0

    completed_now = bool(getattr(prev_entry, "completed", False)) if prev_entry is not None else False
    entry = AniListService.build_history_entry(
        provider_name=provider_name,
        identifier=identifier,
        name=name,
        lang_name=lang_name,
        last_ep=float(current_ep),
        last_pos=pos,
        last_duration=dur,
        last_percent=float(percent) if percent is not None else 0.0,
        completed=completed_now,
        watch_status=str(getattr(prev_entry, "watch_status", "") or ""),
        watched_eps=list(getattr(prev_entry, "watched_eps", []) or []) if prev_entry is not None else [],
        episode_progress=(
            dict(prev_entry.episode_progress)
            if prev_entry is not None and isinstance(prev_entry.episode_progress, dict)
            else {}
        ),
        cover_url=cover_url,
        now_ts=now_ts,
    )
    episode_completed_now = AniListService.episode_is_completed(pos, dur, percent)
    AniListService.apply_episode_progress_update(
        entry,
        current_ep,
        pos=pos,
        dur=dur,
        percent=float(percent) if percent is not None else 0.0,
        completed=episode_completed_now,
        now_ts=now_ts,
    )
    return ProgressSaveResult(
        entry=entry,
        pos=pos,
        dur=dur,
        percent=float(percent) if percent is not None else 0.0,
    )


def apply_mark_episode_completed(
    entry: HistoryEntry,
    ep: float | int | str | None,
    *,
    fallback_pos: float | None = None,
    fallback_dur: float | None = None,
    now_ts: float | None = None,
) -> None:
    cur_prog = AniListService.episode_progress(entry, ep) or {}
    pos_seed = float(getattr(entry, "last_pos", 0.0) or 0.0) if fallback_pos is None else float(fallback_pos)
    dur_seed = float(getattr(entry, "last_duration", 0.0) or 0.0) if fallback_dur is None else float(fallback_dur)
    cur_dur = float(cur_prog.get("dur", dur_seed) or 0.0)
    pos = cur_dur if cur_dur > 0 else float(cur_prog.get("pos", pos_seed) or 0.0)
    AniListService.apply_episode_progress_update(
        entry,
        ep,
        pos=pos,
        dur=cur_dur,
        percent=100.0,
        completed=True,
        now_ts=now_ts,
    )


def reconcile_history_entry(
    entry: HistoryEntry,
    eps: list[float | int],
    *,
    now_ts: float | None = None,
) -> ReconcileResult:
    completed, watch_status = AniListService.reconcile_series_state(
        entry,
        eps,
        now_ts=now_ts,
    )
    return ReconcileResult(
        entry=entry,
        completed=bool(completed),
        watch_status=str(watch_status or ""),
    )


def build_resume_plan(
    entry: HistoryEntry,
    eps: list[float | int],
) -> ResumePlan:
    target_ep: float | int | str | None = eps[0] if eps else entry.last_ep
    current_completed = bool(getattr(entry, "completed", False)) or entry_episode_is_completed(entry, entry.last_ep)
    matched_idx = None

    for i, ep in enumerate(eps):
        try:
            if abs(float(ep) - float(entry.last_ep)) < 1e-6:
                matched_idx = i
                break
        except Exception:
            continue

    if matched_idx is not None:
        if current_completed and matched_idx + 1 < len(eps):
            target_ep = eps[matched_idx + 1]
        else:
            target_ep = eps[matched_idx]
    elif current_completed:
        for ep in eps:
            try:
                if float(ep) > float(entry.last_ep):
                    target_ep = ep
                    break
            except Exception:
                continue

    should_seek_resume = not current_completed
    seek_pos = None
    seek_ratio = None
    resume_marker_pos = None
    if should_seek_resume:
        if float(getattr(entry, "last_pos", 0.0) or 0.0) > 0.0:
            seek_pos = max(0.0, float(entry.last_pos) - 5.0)
        if float(getattr(entry, "last_percent", 0.0) or 0.0) > 0.0:
            seek_ratio = max(0.0, min(1.0, (float(entry.last_percent) / 100.0) - 0.01))
        resume_marker_pos = max(0.0, float(getattr(entry, "last_pos", 0.0) or 0.0))

    return ResumePlan(
        target_ep=target_ep,
        should_seek_resume=should_seek_resume,
        resume_marker_pos=resume_marker_pos,
        seek_pos=seek_pos,
        seek_ratio=seek_ratio,
    )
