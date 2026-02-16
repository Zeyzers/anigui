from __future__ import annotations

import os
import time
from typing import Any

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon, QPixmap, QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStyle,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from anipy_api.provider import LanguageTypeEnum

from components import SearchItem, Worker
from models import HistoryEntry


class HistoryMixin:

    def refresh_history_ui(self):
        self._clear_history_section_overlays()
        self.hist_list.clear()
        self._history_cover_labels = {}
        all_items = self.history.list()
        self._history_items = []
        active_filter = str(getattr(self, "_history_filter", "All") or "All")
        ordered_rows: list[tuple[HistoryEntry, str, bool, int, dict[str, Any] | None]] = []
        marker_rank = {"Watching": 0, "Planned": 1, "Completed": 2}
        for e in all_items:
            marker, in_progress, seen_count, last_prog = self._history_entry_status(e)
            if active_filter != "All" and marker != active_filter:
                continue
            ordered_rows.append((e, marker, in_progress, seen_count, last_prog))

        ordered_rows.sort(
            key=lambda row: (
                marker_rank.get(row[1], 99),
                str(getattr(row[0], "name", "") or "").strip().casefold(),
            )
        )
        self._history_items = [row[0] for row in ordered_rows]
        prev_marker: str | None = None
        grid_slot = 0
        for e, marker, in_progress, seen_count, last_prog in ordered_rows:
            if active_filter == "All" and marker != prev_marker:
                # In 2-column mode, force each section header to start on a new row.
                if grid_slot % 2 != 0:
                    hpre = QListWidgetItem("")
                    hpre.setFlags(Qt.ItemFlag.NoItemFlags)
                    hpre.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header_prepad")
                    hpre.setSizeHint(QSize(420, 34))
                    self.hist_list.addItem(hpre)
                    grid_slot += 1
                section_text = {
                    "Watching": self._tr("In corso", "Watching"),
                    "Planned": self._tr("Da iniziare", "Planned"),
                    "Completed": self._tr("Completati", "Completed"),
                }.get(marker, marker)
                hitem = QListWidgetItem("")
                hitem.setFlags(Qt.ItemFlag.NoItemFlags)
                hitem.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header")
                hitem.setData(Qt.ItemDataRole.UserRole + 3, section_text)
                hitem.setData(Qt.ItemDataRole.UserRole + 4, marker)
                hitem.setSizeHint(QSize(420, 34))
                self.hist_list.addItem(hitem)
                grid_slot += 1
                hpad = QListWidgetItem("")
                hpad.setFlags(Qt.ItemFlag.NoItemFlags)
                hpad.setData(Qt.ItemDataRole.UserRole + 2, "history_section_header_pad")
                hpad.setSizeHint(QSize(420, 34))
                self.hist_list.addItem(hpad)
                grid_slot += 1
                prev_marker = marker

            rel_time = self._fmt_relative_time(e.updated_at)
            marker_label = {
                "Completed": self._tr("Completato", "Completed"),
                "Watching": self._tr("In corso", "Watching"),
                "Planned": self._tr("Da iniziare", "Planned"),
            }.get(marker, marker)
            if marker == "Completed":
                title_line = f"{self._tr('Completato', 'Completed')} {e.name}"
                progress_line = f"EP {e.last_ep} · {e.lang}"
            elif in_progress:
                title_line = self._tr(f"In visione episodio {e.last_ep} di", f"Watching episode {e.last_ep} of")
                progress_line = e.name
            elif marker == "Planned":
                title_line = f"{self._tr('Da iniziare', 'Planned')} {e.name}"
                progress_line = f"EP {e.last_ep} · {e.lang}"
            else:
                title_line = self._tr(f"Visto episodio {e.last_ep} di", f"Watched episode {e.last_ep} of")
                progress_line = e.name

            detail_line = ""
            if in_progress:
                if isinstance(last_prog, dict):
                    pos_v = float(last_prog.get("pos", 0.0) or 0.0)
                    dur_v = float(last_prog.get("dur", 0.0) or 0.0)
                    pct_v = float(last_prog.get("percent", 0.0) or 0.0)
                else:
                    pos_v = float(e.last_pos or 0.0)
                    dur_v = float(e.last_duration or 0.0)
                    pct_v = float(getattr(e, "last_percent", 0.0) or 0.0)
                if dur_v > 0:
                    pct = int(max(0.0, min(1.0, pos_v / dur_v)) * 100)
                    detail_line = self._tr(
                        f"Riprendi {self.fmt_time(pos_v)} · {pct}%",
                        f"Resume {self.fmt_time(pos_v)} · {pct}%",
                    )
                elif pct_v > 0.0:
                    pct = int(max(0.0, min(100.0, pct_v)))
                    detail_line = self._tr(f"Riprendi {pct}%", f"Resume {pct}%")
                else:
                    detail_line = self._tr(f"Riprendi {self.fmt_time(pos_v)}", f"Resume {self.fmt_time(pos_v)}")
            elif marker == "Watching" and seen_count > 0:
                detail_line = self._tr(f"Episodi visti: {seen_count}", f"Seen episodes: {seen_count}")
            elif marker == "Planned":
                detail_line = self._tr("Non iniziato", "Not started")

            item = QListWidgetItem("")
            item.setSizeHint(QSize(420, 98))
            item.setData(Qt.ItemDataRole.UserRole, e)
            item.setData(Qt.ItemDataRole.UserRole + 1, "loading")
            tip = f"{e.name}\n{marker_label} · EP {e.last_ep} · {e.lang}\n{rel_time}"
            if detail_line:
                tip += f"\n{detail_line}"
            item.setToolTip(tip)
            self.hist_list.addItem(item)
            grid_slot += 1
            list_row = self.hist_list.count() - 1

            card = QWidget()
            card.setObjectName("continueCard")
            lay = QHBoxLayout(card)
            lay.setContentsMargins(10, 8, 10, 8)
            lay.setSpacing(10)

            cover = QLabel()
            cover.setFixedSize(52, 74)
            cover.setObjectName("continueCover")
            ph = self._make_cover_placeholder(e.name).pixmap(52, 74)
            cover.setPixmap(ph)
            cover.setScaledContents(True)
            lay.addWidget(cover, 0)

            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(2)

            lbl_title = QLabel(self._compact_title(title_line, max_len=46))
            lbl_title.setObjectName("continueTitle")
            lbl_title.setWordWrap(False)
            text_col.addWidget(lbl_title, 0)

            lbl_sub = QLabel(self._compact_title(progress_line, max_len=46))
            lbl_sub.setObjectName("continueSub")
            lbl_sub.setWordWrap(False)
            text_col.addWidget(lbl_sub, 0)

            if detail_line:
                lbl_detail = QLabel(detail_line)
                lbl_detail.setObjectName("continueMeta")
                text_col.addWidget(lbl_detail, 0)

            lay.addLayout(text_col, 1)

            right_col = QVBoxLayout()
            right_col.setContentsMargins(0, 0, 0, 0)
            right_col.setSpacing(6)

            lbl_time = QLabel(rel_time)
            lbl_time.setObjectName("continueTime")
            lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            right_col.addWidget(lbl_time, 0)

            right_col.addStretch(1)

            lbl_icons = QLabel("💬  ❤")
            lbl_icons.setObjectName("continueIcons")
            lbl_icons.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            right_col.addWidget(lbl_icons, 0)

            lay.addLayout(right_col, 0)
            self.hist_list.setItemWidget(item, card)
            self._history_cover_labels[list_row] = cover

            if e.cover_url:
                w_cover = Worker(self.do_fetch_cover, e.cover_url)
                w_cover.ok.connect(lambda data, r=list_row: self._on_history_cover_loaded(r, data[0], data[1]))
                w_cover.err.connect(lambda _msg: None)
                self._workers.append(w_cover)
                w_cover.start()
            else:
                w_cover = Worker(self.do_fetch_cover_for_history_entry, e)
                w_cover.ok.connect(
                    lambda data, r=list_row, h=e: self._on_history_cover_resolved(
                        r,
                        h,
                        data[0],
                        data[1],
                        data[2],
                    )
                )
                w_cover.err.connect(lambda _msg: None)
                self._workers.append(w_cover)
                w_cover.start()
        self._relayout_history_cards()
        QTimer.singleShot(0, self._relayout_history_cards)
        QTimer.singleShot(80, self._relayout_history_cards)
        if not self._history_items:
            self.hist_list.addItem(self._tr("(vuoto)", "(empty)"))

    def _make_history_section_header_widget(self, section_text: str, marker: str) -> QWidget:
        row = QWidget(self.hist_list.viewport())
        row.setObjectName("historySectionRow")
        row.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(8)

        left_line = QFrame(row)
        left_line.setObjectName("historySectionLine")
        left_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(left_line, 1)

        pill = QLabel(str(section_text or "").upper(), row)
        pill.setObjectName("historySectionPill")
        pill.setProperty("status", str(marker or "").strip().lower())
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(pill, 0)

        right_line = QFrame(row)
        right_line.setObjectName("historySectionLine")
        right_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(right_line, 1)
        return row

    def _clear_history_section_overlays(self):
        overlays = getattr(self, "_history_section_overlays", {})
        for w in overlays.values():
            try:
                w.deleteLater()
            except Exception:
                pass
        self._history_section_overlays = {}

    def _position_history_section_overlays(self):
        if not hasattr(self, "hist_list"):
            return
        if not hasattr(self, "_history_section_overlays"):
            self._history_section_overlays = {}
        to_keep: set[int] = set()
        n = self.hist_list.count()
        for i in range(n):
            item = self.hist_list.item(i)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole + 2) != "history_section_header":
                continue
            rect = self.hist_list.visualItemRect(item)
            if rect.height() <= 0:
                continue
            if i + 1 < n:
                next_item = self.hist_list.item(i + 1)
                if (
                    next_item is not None
                    and next_item.data(Qt.ItemDataRole.UserRole + 2) == "history_section_header_pad"
                    and not next_item.isHidden()
                ):
                    rect = rect.united(self.hist_list.visualItemRect(next_item))

            overlay = self._history_section_overlays.get(i)
            if overlay is None:
                section_text = str(item.data(Qt.ItemDataRole.UserRole + 3) or "")
                marker = str(item.data(Qt.ItemDataRole.UserRole + 4) or "")
                overlay = self._make_history_section_header_widget(section_text, marker)
                self._history_section_overlays[i] = overlay
            overlay.setGeometry(rect)
            overlay.show()
            overlay.raise_()
            to_keep.add(i)

        stale = [k for k in self._history_section_overlays.keys() if k not in to_keep]
        for k in stale:
            try:
                self._history_section_overlays[k].deleteLater()
            except Exception:
                pass
            self._history_section_overlays.pop(k, None)

    def _history_entry_status(
        self,
        entry: HistoryEntry,
    ) -> tuple[str, bool, int, dict[str, Any] | None]:
        ep_map = self._entry_episode_progress_map(entry)
        last_epf = self._to_ep_float(entry.last_ep)
        last_prog = self._entry_get_episode_progress(entry, entry.last_ep)
        if last_prog is None and last_epf is not None:
            last_prog = ep_map.get(last_epf)
        in_progress = False
        if isinstance(last_prog, dict):
            lp_pos = float(last_prog.get("pos", 0.0) or 0.0)
            lp_pct = float(last_prog.get("percent", 0.0) or 0.0)
            lp_done = bool(last_prog.get("completed", False))
            in_progress = (not lp_done) and (lp_pos > 0.0 or lp_pct > 0.0)
        seen_count = len(self._entry_watched_eps_set(entry))
        if bool(entry.completed) and not in_progress:
            return "Completed", in_progress, seen_count, last_prog
        if in_progress or seen_count > 0:
            return "Watching", in_progress, seen_count, last_prog
        return "Planned", in_progress, seen_count, last_prog

    def on_history_filter_changed(self):
        self._history_filter = str(self.combo_history_filter.currentData() or "All")
        self._selected_history = None
        self.refresh_history_ui()

    def _on_history_cover_resolved(
        self,
        row: int,
        entry: HistoryEntry,
        data: bytes | None,
        from_cache: bool,
        url: str | None,
    ):
        if url:
            entry.cover_url = url
            self._remember_offline_cover_url(entry.name, url)
            if not self._incognito_enabled:
                try:
                    self.history.upsert(entry)
                except Exception:
                    pass
        self._on_history_cover_loaded(row, data, from_cache)

    def _relayout_history_cards(self):
        if not hasattr(self, "hist_list"):
            return
        if self.hist_list.count() <= 0:
            self._clear_history_section_overlays()
            return
        vw = self.hist_list.viewport().width()
        # Keep a small safety margin to avoid 1-column fallback from rounding/borders.
        vw = max(0, vw - 12)
        spacing = max(0, self.hist_list.spacing())
        min_card_w = 250
        cols = max(1, min(2, (vw + spacing) // max(1, (min_card_w + spacing))))
        avail = max(min_card_w, vw - 8)
        card_w = (avail - spacing * (cols - 1)) // cols
        card_w = max(min_card_w, card_w)
        self.hist_list.setGridSize(QSize(card_w, 106))
        for i in range(self.hist_list.count()):
            item = self.hist_list.item(i)
            if item is None:
                continue
            item_kind = item.data(Qt.ItemDataRole.UserRole + 2)
            if item_kind == "history_section_header":
                item.setHidden(False)
                item.setSizeHint(QSize(card_w, 34))
                continue
            if item_kind in {"history_section_header_prepad", "history_section_header_pad"}:
                item.setHidden(cols < 2)
                item.setSizeHint(QSize(card_w, 34))
                continue
            if item.flags() == Qt.ItemFlag.NoItemFlags:
                continue
            item.setSizeHint(QSize(card_w, 98))
        self._position_history_section_overlays()

    def _on_history_cover_loaded(self, row: int, data: bytes | None, from_cache: bool):
        if data is None:
            return
        if row < 0 or row >= self.hist_list.count():
            return
        label = self._history_cover_labels.get(row)
        if label is None:
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        self.hist_list.item(row).setData(Qt.ItemDataRole.UserRole + 1, "loaded")

    def _history_entry_from_current_item(self) -> HistoryEntry | None:
        item = self.hist_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, HistoryEntry) else None

    def on_pick_history(self):
        entry = self._history_entry_from_current_item()
        if entry is None:
            return
        self._selected_history = entry
        self.set_status(f"Selezionato in cronologia: {entry.name}")

    def on_history_delete(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        self.history.delete(e.provider, e.identifier, e.lang)
        self.refresh_history_ui()
        self.set_status("Rimosso dalla cronologia.")

    def on_history_resume(self):
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history

        self._set_provider(e.provider if e.provider else "allanime")
        self.lang = LanguageTypeEnum.SUB if e.lang == "SUB" else LanguageTypeEnum.DUB
        self.lang_combo.setCurrentIndex(0 if self.lang == LanguageTypeEnum.SUB else 1)

        try:
            self.selected_anime = self.build_anime_from_history(e)
        except Exception as ex:
            self.notify_err(str(ex))
            return

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.lbl_anime_title.setText(e.name)
        self.lbl_player_title.setText(e.name)
        self.set_status(f"Carico episodi per resume: {e.name}…")

        def after_eps(eps: list[float | int]):
            self.episodes_list = eps
            self.episodes.clear()
            play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            for ep in eps:
                item = QListWidgetItem(play_icon, self._episode_label(ep))
                item.setToolTip(self._tr(f"Doppio click per riprodurre episodio {ep}", f"Double click to play episode {ep}"))
                item.setSizeHint(QSize(0, 44))
                self.episodes.addItem(item)

            target_ep = eps[0] if eps else e.last_ep
            should_seek_resume = not e.completed

            # map history episode (float) to the exact episode object returned by provider
            # to avoid provider edge cases where 1.0 != "1"/1 internally
            matched_idx = None
            for i, ep in enumerate(eps):
                try:
                    if abs(float(ep) - float(e.last_ep)) < 1e-6:
                        matched_idx = i
                        break
                except Exception:
                    continue

            if matched_idx is not None:
                if e.completed and matched_idx + 1 < len(eps):
                    target_ep = eps[matched_idx + 1]
                else:
                    target_ep = eps[matched_idx]
            elif e.completed:
                for ep in eps:
                    try:
                        if float(ep) > float(e.last_ep):
                            target_ep = ep
                            break
                    except Exception:
                        continue

            self._pending_resume_seek = None
            self._pending_resume_seek_ratio = None
            self._pending_resume_seek_attempts = 0
            if should_seek_resume:
                if float(e.last_pos) > 0.0:
                    self._pending_resume_seek = max(0.0, float(e.last_pos) - 5.0)
                if float(getattr(e, "last_percent", 0.0) or 0.0) > 0.0:
                    self._pending_resume_seek_ratio = max(
                        0.0, min(1.0, (float(e.last_percent) / 100.0) - 0.01)
                    )
                self._resume_marker_pos = max(0.0, float(e.last_pos))
            else:
                self._resume_marker_pos = None

            self.play_episode(target_ep)

        w = Worker(self.do_episodes, self.selected_anime)
        w.ok.connect(after_eps)
        w.err.connect(self.on_worker_error)
        self._workers.append(w)
        w.start()

    def on_history_resume_next(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        cur_prog = self._entry_get_episode_progress(e, e.last_ep) or {}
        cur_dur = float(cur_prog.get("dur", e.last_duration) or 0.0)
        self._entry_add_watched_ep(e, e.last_ep)
        self._entry_set_episode_progress(
            e,
            e.last_ep,
            pos=cur_dur if cur_dur > 0 else float(cur_prog.get("pos", e.last_pos) or 0.0),
            dur=cur_dur,
            percent=100.0,
            completed=True,
        )
        e.last_percent = 100.0
        e.updated_at = time.time()
        avail_eps = self._episodes_list_for_history_entry(e)
        if avail_eps is not None:
            e.completed = self._entry_is_series_completed(e, avail_eps)
        self.history.upsert(e)
        self.refresh_history_ui()
        self.on_history_resume()

    def on_history_mark_seen(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not getattr(self, "_selected_history", None):
            return
        e: HistoryEntry = self._selected_history
        if e.last_duration > 0:
            e.last_pos = e.last_duration
        else:
            e.last_pos = max(0.0, float(e.last_pos))
        self._entry_add_watched_ep(e, e.last_ep)
        self._entry_set_episode_progress(
            e,
            e.last_ep,
            pos=float(e.last_pos),
            dur=float(e.last_duration),
            percent=100.0,
            completed=True,
        )
        e.last_percent = 100.0
        e.updated_at = time.time()
        avail_eps = self._episodes_list_for_history_entry(e)
        if avail_eps is not None:
            e.completed = self._entry_is_series_completed(e, avail_eps)
        self.history.upsert(e)
        self.refresh_history_ui()
        self.set_status(f"Segnato come visto: {e.name} ep {e.last_ep}")

    def on_history_open_details(self, *_args):
        if not getattr(self, "_selected_history", None):
            cur = self._history_entry_from_current_item()
            if cur is None:
                return
            self._selected_history = cur
        e: HistoryEntry = self._selected_history

        self._set_provider(e.provider if e.provider else "allanime")
        self.lang = LanguageTypeEnum.SUB if e.lang == "SUB" else LanguageTypeEnum.DUB
        self.lang_combo.setCurrentIndex(0 if self.lang == LanguageTypeEnum.SUB else 1)

        try:
            self.selected_anime = self.build_anime_from_history(e)
        except Exception as ex:
            self.notify_err(str(ex))
            return

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.lbl_anime_title.setText(e.name)
        self.lbl_player_title.setText(e.name)
        self.set_status(f"Apro dettagli anime: {e.name}…")
        self.fetch_episodes(
            seen_eps=self._entry_watched_eps_set(e),
            episode_progress=self._entry_episode_progress_map(e),
        )

    def on_clear_watch_history(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: modifica cronologia bloccata.")
            return
        if not self.history.list():
            return
        self.history._data = []
        try:
            if os.path.exists(self.history.path):
                os.remove(self.history.path)
        except Exception:
            pass
        self.refresh_history_ui()
