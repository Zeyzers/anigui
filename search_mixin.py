from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlsplit

from PySide6.QtCore import Qt, QByteArray, QBuffer, QSize, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QStyle

from anipy_api.anime import Anime
from anipy_api.provider import LanguageTypeEnum, get_provider
from anilist_service import AniListService
from components import SearchItem, Worker
from history_service import find_history_entry, reconcile_history_entry
from models import HistoryEntry


class SearchMixin:

    def _apply_netflix_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0d1418, stop:1 #0b0f12);
                color: #e9edf1;
                font-family: "Fira Sans", "IBM Plex Sans", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QListWidget, QGroupBox {
                background-color: #111a1f;
                border: 1px solid #1f2a33;
                border-radius: 10px;
                padding: 7px 10px;
                selection-background-color: #12b5a5;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #12b5a5;
            }
            QPushButton {
                background-color: #162129;
                border: 1px solid #22303a;
                border-radius: 10px;
                padding: 7px 12px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #1c2a33; }
            QPushButton:pressed { background-color: #10181e; }
            QPushButton:disabled {
                background-color: #101820;
                color: #70818f;
                border-color: #1a252e;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #0f171c;
                border: 1px solid #1f2a33;
                selection-background-color: #12313c;
            }
            QTabWidget::pane { border: 1px solid #1f2a33; border-radius: 12px; }
            QTabBar::tab {
                background: #0f171c;
                border: 1px solid #1f2a33;
                padding: 8px 14px;
                margin-right: 6px;
                border-radius: 10px;
            }
            QTabBar::tab:selected {
                background: #12b5a5;
                color: #0b0f12;
                border-color: #12b5a5;
                font-weight: 600;
            }
            QWidget#suggestionsPanel {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 8px;
            }
            QLabel#sectionTitle { font-size: 15px; font-weight: 600; color: #e9edf1; }
            QLabel#mutedLabel { color: #9fb0bb; }
            QListWidget#resultsList::item {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 6px;
                margin: 6px;
            }
            QListWidget#resultsList::item:selected {
                border: 1px solid #12b5a5;
                background: #132029;
            }
            QListWidget#resultsList::item:hover {
                border: 1px solid #2f4957;
                background: #14212a;
            }
            QListWidget#continueList {
                background: #061227;
                border: 1px solid #17314d;
                border-radius: 12px;
                padding: 6px;
            }
            QListWidget#continueList::item {
                border: none;
                margin: 3px 2px;
                padding: 0;
            }
            QWidget#continueCard {
                background: #0c1d33;
                border: 1px solid #17314d;
                border-radius: 8px;
            }
            QWidget#continueCard:hover {
                border: 1px solid #2d5f98;
                background: #102642;
            }
            QLabel#continueTitle {
                color: #b9d8ff;
                font-size: 14px;
                font-weight: 500;
            }
            QLabel#continueSub {
                color: #a6bfdc;
                font-size: 13px;
            }
            QLabel#continueMeta {
                color: #7ea8d5;
                font-size: 12px;
            }
            QLabel#continueTime {
                color: #7fa3cf;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#continueIcons {
                color: #9db5d1;
                font-size: 13px;
            }
            QWidget#historySectionRow {
                background: transparent;
            }
            QFrame#historySectionLine {
                background: #1a3553;
                border: none;
                min-height: 1px;
                max-height: 1px;
            }
            QLabel#historySectionPill {
                border-radius: 11px;
                padding: 4px 12px;
                border: 1px solid #2e5f8f;
                background: #0f2947;
                color: #b9d8ff;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#historySectionPill[status="watching"] {
                border-color: #2f8e88;
                background: #113737;
                color: #8ff2d9;
            }
            QLabel#historySectionPill[status="planned"] {
                border-color: #8a7440;
                background: #3a301a;
                color: #f4d27a;
            }
            QLabel#historySectionPill[status="completed"] {
                border-color: #4d7aa3;
                background: #1b3046;
                color: #9fceff;
            }
            QListWidget#episodesList::item {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 12px;
                padding: 10px;
                margin: 2px 4px;
            }
            QListWidget#episodesList::item:selected {
                border: 1px solid #12b5a5;
                background: #132029;
            }
            QListWidget#suggestionsList::item {
                background: #101a20;
                border: 1px solid #1f2a33;
                border-radius: 10px;
                padding: 6px;
                margin: 3px;
            }
            QLabel#recentHeader {
                color: #3ddc84;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#recommendedHeader {
                color: #4db8ff;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#featuredHeader {
                color: #f4b55f;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#incognitoBadge {
                background: #c62026;
                color: #ffffff;
                border: 1px solid #ff5a5f;
                border-radius: 10px;
                font-weight: 700;
                padding: 4px 10px;
            }
            QLabel#statusBar {
                background: #0f171c;
                border: 1px solid #1f2a33;
                border-radius: 8px;
                padding: 6px 10px;
                color: #c0d0db;
            }
            QScrollBar:vertical {
                background: #0b0f12;
                width: 10px;
                margin: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1f2a33;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #2b3b46; }
            """
        )
        self.results.setObjectName("resultsList")
        self.episodes.setObjectName("episodesList")

    def do_fetch_cover(self, url: str) -> tuple[bytes | None, bool]:
        cache_path = self._cover_cache_path(url)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = f.read()
                if data:
                    return data, True
            except Exception:
                pass
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=8.0) as r:
            data = r.read()
        if data:
            try:
                scaled = self._downscale_cover_bytes(data)
                if scaled:
                    data = scaled
            except Exception:
                pass
        if data and cache_path and not self._incognito_enabled:
            tmp = cache_path + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, cache_path)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        return (data if data else None), False

    def _downscale_cover_bytes(self, data: bytes) -> bytes | None:
        img = QImage.fromData(data)
        if img.isNull():
            return None
        new_w = max(1, int(img.width() * 0.7))
        new_h = max(1, int(img.height() * 0.7))
        if new_w == img.width() and new_h == img.height():
            return None
        scaled = img.scaled(
            new_w,
            new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        buf = QBuffer()
        arr = QByteArray()
        buf.setBuffer(arr)
        if not buf.open(QBuffer.OpenModeFlag.WriteOnly):
            return None
        fmt = b"PNG" if scaled.hasAlphaChannel() else b"JPG"
        ok = scaled.save(buf, fmt)
        buf.close()
        if not ok:
            return None
        return bytes(arr)

    def _cover_cache_path(self, url: str) -> str | None:
        if not url:
            return None
        h = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()
        return os.path.join(self.cover_cache_dir, f"{h}.img")

    def _provider_cover_base_url(self, provider_name: str | None) -> str:
        provider_s = str(provider_name or "").strip()
        if not provider_s:
            return ""
        try:
            if self._is_aw_provider_name(provider_s):
                return str(self._aw_provider(provider_s).BASE_URL or "").rstrip("/")
            provider = get_provider(provider_s)
            return str(getattr(provider, "BASE_URL", "") or "").rstrip("/")
        except Exception:
            return ""

    def _normalize_cover_url(self, url: str | None, provider_name: str | None = None) -> str | None:
        raw = str(url or "").strip()
        if not raw:
            return None
        if raw.startswith("//"):
            return "https:" + raw
        parts = urlsplit(raw)
        if parts.scheme in {"http", "https"} and parts.netloc:
            return raw
        base_url = self._provider_cover_base_url(provider_name)
        if base_url:
            if raw.startswith("/"):
                return base_url + raw
            if not parts.scheme and not parts.netloc:
                return base_url + "/" + raw.lstrip("/")
        return None

    def _extract_normalized_cover_url(self, obj: Any, provider_name: str | None = None) -> str | None:
        return self._normalize_cover_url(self._extract_cover_url(obj), provider_name)

    def do_fetch_cover_for_item(self, item: SearchItem) -> tuple[bytes | None, bool]:
        url = self._normalize_cover_url(item.cover_url, item.source)
        if url and item.cover_url != url:
            item.cover_url = url
        if not url and self._is_aw_provider_name(item.source):
            url = self._aw_resolve_cover_url(item)
            if url:
                item.cover_url = self._normalize_cover_url(url, item.source) or url
                url = item.cover_url
        if not url:
            try:
                anime = self.build_anime_from_item(item)
                info = anime.get_info()
                url = getattr(info, "image", None)
                if isinstance(url, str) and url.strip():
                    item.cover_url = self._normalize_cover_url(url.strip(), item.source) or url.strip()
                    url = item.cover_url
            except Exception:
                return None, False
        if not url:
            return None, False
        return self.do_fetch_cover(url)

    def do_fetch_cover_for_history_entry(
        self,
        entry: HistoryEntry,
    ) -> tuple[bytes | None, bool, str | None]:
        url = self._normalize_cover_url(entry.cover_url, entry.provider)
        if not url:
            try:
                if self._is_aw_provider_name(entry.provider):
                    aw_item = SearchItem(
                        name=entry.name,
                        identifier=entry.identifier,
                        languages=set(),
                        cover_url=None,
                        source=entry.provider,
                    )
                    url = self._aw_resolve_cover_url(aw_item)
                if not url:
                    anime = self.build_anime_from_history(entry)
                    info = anime.get_info()
                    cand = self._extract_cover_url(info)
                    if isinstance(cand, str) and cand.strip():
                        url = cand.strip()
            except Exception:
                url = None
        url = self._normalize_cover_url(url, entry.provider)
        if not url:
            return None, False, None
        data, from_cache = self.do_fetch_cover(url)
        return data, from_cache, url

    def _on_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._search_nonce:
            return
        if data is None:
            if 0 <= row < self.results.count():
                self.results.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.results.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.results.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._result_items):
            it = self._result_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.results.item(row).setIcon(QIcon(scaled))
        self.results.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    def _on_recommended_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._recommended_nonce:
            return
        if data is None:
            if 0 <= row < self.recommended.count():
                self.recommended.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.recommended.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.recommended.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._recommended_items):
            it = self._recommended_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.recommended.item(row).setIcon(QIcon(scaled))
        self.recommended.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    def _on_recent_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._recent_nonce:
            return
        if data is None:
            if 0 <= row < self.recent.count():
                self.recent.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.recent.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.recent.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._recent_items):
            it = self._recent_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.recent.item(row).setIcon(QIcon(scaled))
        self.recent.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    def _on_featured_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._featured_nonce:
            return
        if data is None:
            if 0 <= row < self.featured.count():
                self.featured.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")
            return
        if row < 0 or row >= self.featured.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.featured.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if 0 <= row < len(self._featured_items):
            it = self._featured_items[row]
            self._remember_offline_cover_url(it.name, it.cover_url)
        self.featured.item(row).setIcon(QIcon(scaled))
        self.featured.item(row).setData(Qt.ItemDataRole.UserRole, "loaded")

    # ---------------- anipy operations ----------------
    def do_search(self, query: str) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            res = provider.search(query) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                if self.lang == LanguageTypeEnum.DUB and not is_dub:
                    continue
                if self.lang == LanguageTypeEnum.SUB and is_dub:
                    continue
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=self.provider_name,
                        raw=r,
                    )
                )
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        res = provider.get_search(query)
        out = [
            SearchItem(
                r.name,
                r.identifier,
                r.languages,
                cover_url=self._extract_normalized_cover_url(r, self.provider_name),
                source=self.provider_name,
                raw=r,
            )
            for r in res
        ]
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_recent_releases(self) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            filter_code = "d" if self.lang == LanguageTypeEnum.DUB else "s"
            res = provider.latest(filter_code) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=self.provider_name,
                        raw=r,
                    )
                )
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        seen: set[str] = set()
        out: list[SearchItem] = []
        now = time.localtime()
        year = now.tm_year
        month = now.tm_mon
        season = "winter" if month <= 3 else "spring" if month <= 6 else "summer" if month <= 9 else "fall"
        queries = [
            f"{season} {year}",
            f"{year}",
            "ongoing",
            "new",
        ]

        for q in queries:
            try:
                res = provider.get_search(q)
            except Exception:
                continue
            for r in res:
                ident = getattr(r, "identifier", None)
                if not ident or ident in seen:
                    continue
                seen.add(ident)
                out.append(
                    SearchItem(
                        r.name,
                        r.identifier,
                        r.languages,
                        cover_url=self._extract_normalized_cover_url(r, self.provider_name),
                        source=self.provider_name,
                        raw=r,
                    )
                )
                if len(out) >= 60:
                    for it in out:
                        self._apply_metadata_cache_to_item(it)
                    return out
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_recommended(self) -> list[SearchItem]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            seen: set[str] = set()
            out: list[SearchItem] = []
            for q in ("popular", "top", "best", "shonen"):
                res = provider.search(q) or []
                for r in res:
                    ident = str(getattr(r, "ref", ""))
                    if not ident or ident in seen:
                        continue
                    seen.add(ident)
                    is_dub = bool(getattr(r, "dub", False))
                    if self.lang == LanguageTypeEnum.DUB and not is_dub:
                        continue
                    if self.lang == LanguageTypeEnum.SUB and is_dub:
                        continue
                    out.append(
                        SearchItem(
                            name=r.name,
                            identifier=ident,
                            languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                            cover_url=None,
                            source=self.provider_name,
                            raw=r,
                        )
                    )
                    if len(out) >= 60:
                        for it in out:
                            self._apply_metadata_cache_to_item(it)
                        return out
            for it in out:
                self._apply_metadata_cache_to_item(it)
            return out

        provider = get_provider(self.provider_name)
        seen: set[str] = set()
        out: list[SearchItem] = []
        queries = ["popular", "top", "best", "trending"]

        for q in queries:
            try:
                res = provider.get_search(q)
            except Exception:
                continue
            for r in res:
                ident = getattr(r, "identifier", None)
                if not ident or ident in seen:
                    continue
                seen.add(ident)
                out.append(
                    SearchItem(
                        r.name,
                        r.identifier,
                        r.languages,
                        cover_url=self._extract_normalized_cover_url(r, self.provider_name),
                        source=self.provider_name,
                        raw=r,
                    )
                )
                if len(out) >= 60:
                    for it in out:
                        self._apply_metadata_cache_to_item(it)
                    return out
        for it in out:
            self._apply_metadata_cache_to_item(it)
        return out

    def do_episodes(self, anime: Any) -> list[float | int]:
        if self._is_aw_provider_name(self.provider_name):
            provider = self._aw_provider()
            provider.episodes(anime)
            out: list[float | int] = []
            for ep in anime.episodes():
                try:
                    f = float(ep)
                    out.append(int(f) if f.is_integer() else f)
                except Exception:
                    continue
            return out
        return anime.get_episodes(lang=self.lang)

    def build_anime_from_item(self, item: SearchItem) -> Any:
        if self._is_aw_provider_name(item.source):
            if item.raw is not None:
                return item.raw
            anime_cls = self._aw_anime_class()
            anime = anime_cls(item.name, item.identifier)
            try:
                self._aw_provider(item.source).info_anime(anime)
            except Exception:
                pass
            return anime
        provider = get_provider(item.source)
        return Anime(provider, item.name, item.identifier, item.languages)

    def build_anime_from_history(self, entry: HistoryEntry) -> Any:
        if self._is_aw_provider_name(entry.provider):
            anime_cls = self._aw_anime_class()
            anime = anime_cls(entry.name, entry.identifier)
            try:
                self._aw_provider(entry.provider).info_anime(anime)
            except Exception:
                pass
            return anime
        provider = get_provider(entry.provider)
        languages = {LanguageTypeEnum.SUB, LanguageTypeEnum.DUB}
        return Anime(provider, entry.name, entry.identifier, languages)

    def _current_history_entry(self) -> HistoryEntry | None:
        if not self.selected_anime:
            return None
        ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        anime_name = str(getattr(self.selected_anime, "name", "") or "").strip()
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        return find_history_entry(
            self.history.read(),
            provider=self.provider_name,
            identifier=ident,
            lang=lang_s,
            name=anime_name,
        )

    def _episodes_list_for_history_entry(self, entry: HistoryEntry) -> list[float | int] | None:
        if not self.episodes_list:
            return None
        if self.provider_name != entry.provider:
            return None
        cur_lang = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        if cur_lang != entry.lang:
            return None
        if not self.selected_anime:
            return None
        cur_ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        if cur_ident != str(entry.identifier):
            return None
        return self.episodes_list

    def _history_context_for_selected_anime(
        self,
    ) -> tuple[set[float] | None, dict[float, dict[str, Any]] | None]:
        entry = self._current_history_entry()
        if entry is None:
            return None, None
        return self._entry_watched_eps_set(entry), self._entry_episode_progress_map(entry)

    def _load_items_with_covers(
        self,
        items: list[SearchItem],
        list_widget: QListWidget,
        nonce: int,
        on_cover: Callable[[int, int, bytes | None, bool], None],
    ):
        list_widget.clear()
        for row, it in enumerate(items):
            item = QListWidgetItem(self._make_cover_placeholder(it.name), it.name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            item.setSizeHint(QSize(170, 290))
            item.setData(Qt.ItemDataRole.UserRole, "loading")
            list_widget.addItem(item)

            w = Worker(self.do_fetch_cover_for_item, it)
            w.ok.connect(lambda data, r=row, n=nonce: on_cover(r, n, data[0], data[1]))
            w.err.connect(lambda _msg: None)
            self._track_worker(w)
            w.start()

        self._start_skeletons()

    # ---------------- topbar handlers ----------------
    def _apply_provider_ui_state(self):
        is_aw = self.provider_name.startswith("aw_")
        self.quality_combo.setEnabled(not is_aw)
        if is_aw:
            self.quality_combo.setToolTip("Qualita stream non disponibile su provider ITA (aw-cli).")
        else:
            self.quality_combo.setToolTip("")

    def _set_provider(self, name: str):
        self.provider_name = name
        idx = self.provider_combo.findData(name)
        if idx >= 0 and self.provider_combo.currentIndex() != idx:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(idx)
            self.provider_combo.blockSignals(False)
        self._apply_provider_ui_state()

    def on_provider_change(self):
        self.provider_name = self.provider_combo.currentData()
        self._apply_provider_ui_state()
        q = self.search_input.text().strip()
        if len(q) >= 3:
            self.on_search(query=q)

    def on_lang_change(self):
        self.lang = self.lang_combo.currentData()
        if self.selected_anime:
            self.fetch_episodes()

    def on_quality_change(self):
        self.quality = self.quality_combo.currentText()

    # ---------------- Search flow ----------------
    def on_pick_recent_query(self, item: QListWidgetItem):
        q = item.text().strip()
        if not q or q.startswith("("):
            return
        self.search_input.setText(q)
        self.on_search(query=q)

    def on_filters_changed(self, *_args):
        if not self._result_items_raw:
            return
        nonce = self._search_nonce
        items = self._apply_filters(self._result_items_raw, nonce)
        self._result_items = items
        self._render_search_results(items, nonce)

    def on_search_text_changed(self, text: str):
        q = text.strip()
        self._pending_search_query = q
        if not q:
            self._last_search_query = None
            self._last_search_provider = None
            if self._search_debounce_timer.isActive():
                self._search_debounce_timer.stop()
            self.results.clear()
            self._search_cards = []
            self._result_items = []
            self.lbl_search_meta.setVisible(False)
            self.lbl_search_state.setVisible(False)
            self.set_status("Pronto.")
            self._update_suggestions_visibility()
            self._refresh_search_layout()
            return
        if len(q) < 3:
            self.lbl_search_state.setText(self._tr("Scrittura in corso...", "Typing..."))
            self.lbl_search_state.setVisible(False)
            self.results.clear()
            self._search_cards = []
            self._result_items = []
            self.lbl_search_meta.setVisible(False)
            self._update_suggestions_visibility()
            self._refresh_search_layout()
            return
        self.lbl_search_state.setVisible(False)
        self._update_suggestions_visibility()
        self._refresh_search_layout()
        self._search_debounce_timer.start()

    def _run_debounced_search(self):
        if not self._pending_search_query:
            return
        if len(self._pending_search_query) < 3:
            return
        self.on_search(query=self._pending_search_query)

    def on_search(self, query: str | None = None):
        q = (query if query is not None else self.search_input.text()).strip()
        if not q:
            return
        last_search_provider = getattr(self, "_last_search_provider", None)
        if q == self._last_search_query and self.provider_name == last_search_provider:
            return
        self._last_search_query = q
        self._last_search_provider = self.provider_name
        if self._search_debounce_timer.isActive():
            self._search_debounce_timer.stop()

        self._search_nonce += 1
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_catalog)
        self._current_search_view = "catalog"
        self._refresh_search_layout()
        self._save_current_progress(force=True)
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.set_status(self._tr("Ricerca in corso…", "Searching..."))
        self._begin_request()
        self.results.clear()
        self._search_cards = []
        self._result_items_raw = []
        self.lbl_search_meta.setText(self._tr(f"Risultati per: {q}", f"Results for: {q}"))
        self.lbl_search_meta.setVisible(True)
        self.lbl_search_state.setText(self._tr("Cercando…", "Searching..."))
        self.lbl_search_state.setVisible(True)
        self.episodes.clear()
        self.lbl_anime_title.setText(self._tr("Anime", "Anime"))
        self.lbl_player_title.setText(self._tr("Player", "Player"))
        self.selected_anime = None
        self.episodes_list = []
        self.current_ep = None

        nonce = self._search_nonce
        w = Worker(self.do_search, q)
        w.ok.connect(lambda items, n=nonce, q=q: self.on_search_results(n, q, items))
        w.err.connect(self.on_worker_error)
        self._track_worker(w)
        w.start()

    def on_refresh_recommended(self):
        self._recommended_loaded = True
        self._recent_nonce += 1
        self._recommended_nonce += 1
        recent_nonce = self._recent_nonce
        nonce = self._recommended_nonce
        self.set_status(self._tr("Carico recenti + consigliati…", "Loading recent + recommended..."))
        self.recent.clear()
        self._recent_items = []
        self._recent_cards = []
        self.recommended.clear()
        self._recommended_items = []
        self._recommended_cards = []

        w_recent = Worker(self.do_recent_releases)
        w = Worker(self.do_recommended)

        def on_recent_ok(items: list[SearchItem]):
            if recent_nonce != self._recent_nonce:
                self._end_request()
                return
            self._recent_items = items
            self._load_items_with_covers(
                items,
                self.recent,
                recent_nonce,
                self._on_recent_cover_loaded,
            )
            self.set_status(
                f"Recent: {len(self._recent_items)} | Recommended: {len(self._recommended_items)}"
            )
            self._end_request()

        def on_ok(items: list[SearchItem]):
            if nonce != self._recommended_nonce:
                self._end_request()
                return
            self._recommended_items = items
            self._load_items_with_covers(
                items,
                self.recommended,
                nonce,
                self._on_recommended_cover_loaded,
            )
            self.set_status(
                f"Recent: {len(self._recent_items)} | Recommended: {len(self._recommended_items)}"
            )
            self._end_request()

        w_recent.ok.connect(on_recent_ok)
        w_recent.err.connect(self.on_worker_error)
        w.ok.connect(on_ok)
        w.err.connect(self.on_worker_error)
        self._begin_request()
        self._track_worker(w_recent)
        self._begin_request()
        self._track_worker(w)
        w_recent.start()
        w.start()

    def on_refresh_featured(self, force: bool = False):
        self._featured_loaded = True
        self._featured_nonce += 1
        nonce = self._featured_nonce
        self.featured.clear()
        self._featured_items = []
        self._featured_cards = []
        if not self._is_featured_tab_enabled():
            self.featured.addItem(self._tr("(attiva AniList per vedere questa tab)", "(enable AniList to use this tab)"))
            self.featured.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return
        token = self._anilist_active_token_from_ui()
        provider_name = str(self.provider_name or "allanime")
        lang_name = "DUB" if self.lang == LanguageTypeEnum.DUB else "SUB"
        refresh_salt = int(time.time() * 1000) if force else 0
        token_hash = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:12]
        cache_key = f"{provider_name}|{lang_name}|{token_hash}"
        cache_ttl = 15 * 60
        if not force:
            cached = self._featured_cache.get(cache_key)
            if isinstance(cached, dict):
                ts = float(cached.get("ts", 0.0) or 0.0)
                if (time.time() - ts) <= cache_ttl:
                    cached_items = list(cached.get("items") or [])
                    if cached_items:
                        self._featured_items = cached_items
                        self._load_items_with_covers(
                            self._featured_items,
                            self.featured,
                            nonce,
                            self._on_featured_cover_loaded,
                        )
                        for i in range(self.featured.count()):
                            it = self.featured.item(i)
                            if it is not None:
                                it.setToolTip(self._featured_algorithm_tooltip())
                        self.set_status(f"Featured: {len(self._featured_items)}")
                        return
        self.set_status(self._tr("Calcolo featured in corso…", "Computing featured picks..."))
        self._begin_request()
        w = Worker(
            self._anilist_featured_recommendations_worker,
            token,
            provider_name,
            lang_name,
            refresh_salt,
        )

        def on_ok(items: list[SearchItem]):
            if nonce != self._featured_nonce:
                self._end_request()
                return
            self._featured_items = list(items or [])
            if not self._featured_items:
                self.featured.addItem(self._tr("(nessun featured trovato)", "(no featured picks found)"))
                self.featured.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
                self.set_status("Featured: 0")
                self._end_request()
                return
            self._load_items_with_covers(
                self._featured_items,
                self.featured,
                nonce,
                self._on_featured_cover_loaded,
            )
            self._featured_cache[cache_key] = {
                "ts": time.time(),
                "items": list(self._featured_items),
            }
            for i in range(self.featured.count()):
                it = self.featured.item(i)
                if it is not None:
                    it.setToolTip(self._featured_algorithm_tooltip())
            self.set_status(f"Featured: {len(self._featured_items)}")
            self._end_request()

        def on_err(msg: str):
            if nonce != self._featured_nonce:
                self._end_request()
                return
            self.featured.clear()
            self._featured_items = []
            self.featured.addItem(self._tr("(featured non disponibile)", "(featured unavailable)"))
            self.featured.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            self.set_status(self._tr(f"Featured non disponibile: {msg}", f"Featured unavailable: {msg}"))
            self._end_request()

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()

    def on_tab_changed(self, idx: int):
        if self.tabs.widget(idx) is self.tab_recommended and not self._recommended_loaded:
            self.on_refresh_recommended()
        if self.tabs.widget(idx) is self.tab_featured and not self._featured_loaded:
            self.on_refresh_featured()
        if self.tabs.widget(idx) is self.tab_offline:
            self.refresh_offline_library()
        if self.tabs.widget(idx) is self.tab_favorites:
            self.refresh_favorites_ui()
        if self.tabs.widget(idx) is self.tab_history:
            self._save_current_progress(force=True)
            self._queue_history_refresh()
            QTimer.singleShot(0, self._relayout_history_cards)
            QTimer.singleShot(80, self._relayout_history_cards)

    def on_search_results(self, nonce: int, query: str, items: list[SearchItem]):
        self._end_request()
        if nonce != self._search_nonce:
            return
        current = self.search_input.text().strip()
        if current == query and len(query) >= 3:
            self._add_recent_query(query)
        self._result_items_raw = items
        items = self._apply_filters(items, nonce)
        self._result_items = items
        self._render_search_results(items, nonce)
        self.set_status(f"Trovati {len(items)} risultati.")

    def _render_search_results(self, items: list[SearchItem], nonce: int):
        self._search_cards = []
        if not items:
            self.lbl_search_state.setText(self._tr("Nessun risultato.", "No results."))
            self.lbl_search_state.setVisible(True)
        else:
            self.lbl_search_state.setVisible(False)
        self._load_items_with_covers(
            items,
            self.results,
            nonce,
            self._on_cover_loaded,
        )

    def _filters_active(self) -> bool:
        if self.combo_season.currentIndex() > 0:
            return True
        if self.combo_year.currentIndex() > 0:
            return True
        if self.input_studio.text().strip():
            return True
        if self.combo_rating.currentIndex() > 0:
            return True
        return False

    def _apply_filters(self, items: list[SearchItem], nonce: int) -> list[SearchItem]:
        items = list(items)
        allow_missing = False
        if self._filters_active():
            if self._items_need_enrichment(items):
                self.lbl_search_state.setText(self._tr("Filtraggio…", "Filtering..."))
                self.lbl_search_state.setVisible(True)
                self._start_enrich_items(items, nonce)
                allow_missing = True
            items = [it for it in items if self._item_matches_filters(it, allow_missing)]
        items = self._sort_items(items)
        return items

    def _sort_items(self, items: list[SearchItem]) -> list[SearchItem]:
        idx = self.combo_sort.currentIndex()
        if idx == 1:
            return sorted(items, key=lambda x: x.name.lower())
        if idx == 2:
            return sorted(items, key=lambda x: x.name.lower(), reverse=True)
        return items

    def _items_need_enrichment(self, items: list[SearchItem]) -> bool:
        if not self._filters_active():
            return False
        for it in items:
            if self._item_missing_required_fields(it):
                return True
        return False

    def _item_missing_required_fields(self, it: SearchItem) -> bool:
        if self.combo_season.currentIndex() > 0 and not it.season:
            return True
        if self.combo_year.currentIndex() > 0 and not it.year:
            return True
        if self.input_studio.text().strip() and not it.studio:
            return True
        if self.combo_rating.currentIndex() > 0 and it.rating is None:
            return True
        return False

    def _item_matches_filters(self, it: SearchItem, allow_missing: bool) -> bool:
        if self.combo_season.currentIndex() > 0:
            season = self.combo_season.currentText().lower()
            if not it.season:
                return allow_missing
            if it.season.lower() != season:
                return False
        if self.combo_year.currentIndex() > 0:
            try:
                year = int(self.combo_year.currentText())
            except Exception:
                year = None
            if year and it.year is None:
                return allow_missing
            if year and it.year != year:
                return False
        studio = self.input_studio.text().strip().lower()
        if studio:
            if not it.studio:
                return allow_missing
            if studio not in it.studio.lower():
                return False
        if self.combo_rating.currentIndex() > 0:
            threshold = {1: 9, 2: 8, 3: 7, 4: 6}.get(self.combo_rating.currentIndex(), 0)
            if it.rating is None:
                return allow_missing
            if it.rating < threshold:
                return False
        return True

    def _start_enrich_items(self, items: list[SearchItem], nonce: int):
        if self._filters_pending:
            return
        self._filters_pending = True
        self._enrich_nonce += 1
        enrich_nonce = self._enrich_nonce
        self._begin_request()
        w = Worker(self._enrich_items_worker, items)
        w.ok.connect(lambda enriched, n=enrich_nonce, sn=nonce: self._on_items_enriched(n, sn, enriched))
        w.err.connect(self._on_enrich_error)
        self._track_worker(w)
        w.start()

    def _enrich_items_worker(self, items: list[SearchItem]) -> list[SearchItem]:
        out: list[SearchItem] = []
        for it in items:
            self._apply_metadata_cache_to_item(it)
            if not self._item_missing_required_fields(it):
                out.append(it)
                continue
            try:
                anime = self.build_anime_from_item(it)
                info = anime.get_info()
                it.year = getattr(info, "year", None) or getattr(info, "release_year", None)
                it.season = getattr(info, "season", None)
                it.studio = getattr(info, "studio", None) or getattr(info, "studios", None)
                it.rating = getattr(info, "rating", None) or getattr(info, "score", None)
                if isinstance(it.studio, list):
                    it.studio = ", ".join([str(x) for x in it.studio if x])
                self._store_metadata_cache_for_item(it)
            except Exception:
                pass
            out.append(it)
        return out

    def _on_items_enriched(self, enrich_nonce: int, search_nonce: int, items: list[SearchItem]):
        self._end_request()
        self._filters_pending = False
        if enrich_nonce != self._enrich_nonce:
            return
        if search_nonce != self._search_nonce:
            return
        self._result_items_raw = items
        filtered = self._apply_filters(items, search_nonce)
        self._result_items = filtered
        self._render_search_results(filtered, search_nonce)

    def _on_enrich_error(self, _msg: str):
        self._end_request()
        self._filters_pending = False
        self.notify_err(_msg)
        self.set_status("Errore.")

    def on_pick_result(self, *_args):
        row = self.results.currentRow()
        if row < 0:
            return
        item = self._result_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_pick_recommended(self):
        row = self.recommended.currentRow()
        if row < 0 or row >= len(self._recommended_items):
            return
        item = self._recommended_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_pick_featured(self):
        row = self.featured.currentRow()
        if row < 0 or row >= len(self._featured_items):
            return
        item = self._featured_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_pick_recent(self):
        row = self.recent.currentRow()
        if row < 0 or row >= len(self._recent_items):
            return
        item = self._recent_items[row]
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status(f"Selezionato: {item.name} — carico episodi…")
        self.fetch_episodes()

    def on_mark_anime_planned(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: salvataggio cronologia bloccato.")
            return
        if not self.selected_anime:
            self.notify_err(self._tr("Apri prima un anime.", "Open an anime first."))
            return
        ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        ).strip()
        if not ident:
            self.notify_err(self._tr("Impossibile identificare questo anime.", "Unable to identify this anime."))
            return
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        now = time.time()
        existing = self._current_history_entry()
        name = str(getattr(self.selected_anime, "name", "") or getattr(self.selected_search_item, "name", "anime")).strip()
        cover_url = (
            getattr(self.selected_search_item, "cover_url", None)
            if self.selected_search_item
            else (existing.cover_url if existing is not None else None)
        )
        entry = AniListService.build_history_entry(
            provider_name=(existing.provider if existing is not None else self.provider_name),
            identifier=(existing.identifier if existing is not None else ident),
            name=(existing.name if existing is not None and str(existing.name or "").strip() else (name or "anime")),
            lang_name=lang_s,
            cover_url=cover_url,
            now_ts=now,
        )
        self.history.upsert(entry)
        self._anilist_sync_progress_async(entry, force=True)
        self.refresh_history_ui()
        if self.selected_anime is not None:
            self.fetch_episodes()
        self.set_status(self._tr(f"Aggiunto a Da iniziare: {entry.name}", f"Added to Planned: {entry.name}"))

    def on_mark_anime_completed(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: salvataggio cronologia bloccato.")
            return
        if not self.selected_anime:
            self.notify_err(self._tr("Apri prima un anime.", "Open an anime first."))
            return
        ident = str(
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        ).strip()
        if not ident:
            self.notify_err(self._tr("Impossibile identificare questo anime.", "Unable to identify this anime."))
            return
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"
        now = time.time()
        existing = self._current_history_entry()
        name = str(getattr(self.selected_anime, "name", "") or getattr(self.selected_search_item, "name", "anime")).strip()
        cover_url = (
            getattr(self.selected_search_item, "cover_url", None)
            if self.selected_search_item
            else (existing.cover_url if existing is not None else None)
        )

        eps_values: list[float] = []
        for ep in (self.episodes_list or []):
            f = self._to_ep_float(ep)
            if f is not None and f > 0:
                eps_values.append(float(f))
        eps_values = sorted(set(eps_values))
        if not eps_values and existing is not None:
            eps_values = sorted(self._entry_watched_eps_set(existing))
        if not eps_values and self.current_ep is not None:
            cur = self._to_ep_float(self.current_ep)
            if cur is not None and cur > 0:
                eps_values = [float(cur)]
        if not eps_values:
            self.notify_err(
                self._tr(
                    "Attendi il caricamento episodi prima di segnare completato.",
                    "Wait for episode list to load before marking completed.",
                )
            )
            return

        entry = AniListService.build_completed_entry(
            provider_name=(existing.provider if existing is not None else self.provider_name),
            identifier=(existing.identifier if existing is not None else ident),
            name=(existing.name if existing is not None and str(existing.name or "").strip() else (name or "anime")),
            lang_name=lang_s,
            watched_eps=list(eps_values),
            cover_url=cover_url,
            now_ts=now,
        )
        self.history.upsert(entry)
        self._anilist_sync_progress_async(entry, force=True)
        self.refresh_history_ui()
        if self.selected_anime is not None:
            self.fetch_episodes()
        self.set_status(self._tr(f"Segnato come completato: {entry.name}", f"Marked as completed: {entry.name}"))

    def fetch_episodes(
        self,
        seen_eps: set[float] | None = None,
        episode_progress: dict[float, dict[str, Any]] | None = None,
    ):
        if not self.selected_anime:
            return
        if seen_eps is None or episode_progress is None:
            hist_seen, hist_prog = self._history_context_for_selected_anime()
            if seen_eps is None:
                seen_eps = hist_seen
            if episode_progress is None:
                episode_progress = hist_prog
        self._history_seen_eps_for_ui = seen_eps
        self._history_episode_progress_for_ui = episode_progress
        self.episodes.clear()
        w = Worker(self.do_episodes, self.selected_anime)
        w.ok.connect(self.on_episodes_ready)
        w.err.connect(self.on_worker_error)
        self._track_worker(w)
        w.start()

    def on_episodes_ready(self, eps: list[float | int]):
        self.episodes_list = eps
        hist_entry = self._current_history_entry()
        if hist_entry is not None and not self._incognito_enabled:
            prev_completed = bool(hist_entry.completed)
            prev_watch_status = str(getattr(hist_entry, "watch_status", "") or "")
            reconcile = reconcile_history_entry(
                hist_entry,
                eps,
            )
            new_completed = reconcile.completed
            new_watch_status = reconcile.watch_status
            if (
                prev_completed != bool(new_completed)
                or prev_watch_status != new_watch_status
            ):
                try:
                    self.history.upsert(hist_entry)
                except Exception:
                    pass
        self.episodes.clear()
        play_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        seen_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        resume_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        seen_eps = self._history_seen_eps_for_ui
        ep_prog = self._history_episode_progress_for_ui or {}
        for ep in eps:
            icon = play_icon
            label = self._episode_label(ep)
            ef = self._to_ep_float(ep)
            p = None if ef is None else ep_prog.get(ef)
            if seen_eps is not None:
                if ef is not None and any(abs(ef - s) < 1e-6 for s in seen_eps):
                    icon = seen_icon
                    label = f"{label}  ✓ seen"
                elif isinstance(p, dict):
                    pos = float(p.get("pos", 0.0) or 0.0)
                    pct = float(p.get("percent", 0.0) or 0.0)
                    if pos > 0.0:
                        icon = resume_icon
                        label = f"{label}  ▶ {self.fmt_time(pos)}"
                    elif pct > 0.0:
                        icon = resume_icon
                        label = f"{label}  ▶ {int(max(0.0, min(100.0, pct)))}%"
            item = QListWidgetItem(icon, label)
            item.setToolTip(self._tr(f"Doppio click per riprodurre episodio {ep}", f"Double click to play episode {ep}"))
            item.setSizeHint(QSize(0, 44))
            self.episodes.addItem(item)
        if self.selected_anime:
            self.search_stack.setCurrentWidget(self.page_anime)
            self._current_search_view = "anime"
            self._refresh_search_layout()
        self.set_status(f"Episodi: {len(eps)}")
        self._history_seen_eps_for_ui = None
        self._history_episode_progress_for_ui = None

    def on_play_selected_episode(self, *_args):
        row = self.episodes.currentRow()
        if row < 0 or not self.selected_anime:
            return
        ep = self.episodes_list[row]
        hist = self._current_history_entry()
        if hist is not None:
            prog = self._entry_get_episode_progress(hist, ep)
            if isinstance(prog, dict):
                pos = float(prog.get("pos", 0.0) or 0.0)
                pct = float(prog.get("percent", 0.0) or 0.0)
                self._pending_resume_seek = None
                self._pending_resume_seek_ratio = None
                self._pending_resume_seek_attempts = 0
                if pos > 0.0:
                    self._pending_resume_seek = max(0.0, pos - 5.0)
                elif pct > 0.0:
                    self._pending_resume_seek_ratio = max(0.0, min(1.0, (pct / 100.0) - 0.01))
        self.play_episode(ep)

    # ---------------- History tab ----------------
