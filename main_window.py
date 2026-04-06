from __future__ import annotations

import json
import os
import sys
import time
import warnings
import socket
import tempfile
import platform
import subprocess
import urllib.request
import urllib.error
from urllib.parse import quote_plus, parse_qs, urlsplit
import hashlib
import importlib
import re
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Optional, Any, Callable
import errno
import threading
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEvent, QSize, QByteArray, QBuffer, QPropertyAnimation, QUrl, QProcess
from PySide6.QtGui import QShortcut, QKeySequence, QPixmap, QIcon, QColor, QPainter, QCursor, QImage, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QStackedWidget,
    QLabel,
    QSplitter,
    QMessageBox,
    QComboBox,
    QTabWidget,
    QSlider,
    QStyle,
    QGroupBox,
    QListWidgetItem,
    QListView,
    QFrame,
    QProgressBar,
    QGraphicsOpacityEffect,
    QAbstractItemView,
    QFileDialog,
    QSizePolicy,
)
from urllib3.exceptions import InsecureRequestWarning

from anipy_api.provider import get_provider, LanguageTypeEnum
from anipy_api.anime import Anime
from anilist_service import AniListService
from components import (
    SearchItem,
    OfflineAnimeItem,
    FavoriteEntry,
    StreamResult,
    Worker,
    DownloadTask,
    DownloadWorker,
    PlayerBase,
    FullscreenPlayerWindow,
    MiniPlayerWindow,
    create_player_widget,
)
from models import (
    app_state_dir,
    HISTORY_PATH,
    SEARCH_HISTORY_PATH,
    OFFLINE_COVERS_MAP_PATH,
    SETTINGS_PATH,
    FAVORITES_PATH,
    METADATA_CACHE_PATH,
    HistoryEntry,
    HistoryStore,
)
from history_service import build_saved_progress_entry, find_history_entry
from settings_service import RuntimeSettings, SettingsService
from app_state_service import AppStateService
from crash_logging import LoggingApplication, UiFreezeWatchdog, log_debug
from services import APP_VERSION, UPDATE_MANIFEST_URL, UpdateService
from download_mixin import DownloadMixin
from update_mixin import UpdateMixin
from history_mixin import HistoryMixin
from search_mixin import SearchMixin
from player_mixin import PlayerMixin

# Suppress known noisy warning from upstream provider calls that use unverified TLS
# for api.allanime.day. Keep other TLS warnings untouched.
warnings.filterwarnings(
    "ignore",
    message=r"Unverified HTTPS request is being made to host 'api\.allanime\.day'.*",
    category=InsecureRequestWarning,
)


def debug_log(msg: str) -> None:
    log_debug(msg)

# ---------------------------
# Main App
# ---------------------------
class MainWindow(PlayerMixin, SearchMixin, DownloadMixin, UpdateMixin, HistoryMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anigui" + APP_VERSION)
        self.resize(1300, 780)

        self._init_settings_and_runtime()
        self._build_search_and_player_ui()
        self._build_recommended_tab()
        self._build_featured_tab()
        self._build_downloads_tab()
        self._build_offline_tab()
        self._build_favorites_tab()
        self._build_settings_tab()
        self._build_history_tab()
        self._build_watchparty_tab()
        self._build_status_bar()
        self._init_runtime_fields()
        self._init_timers()
        self._finish_startup()

    def _init_settings_and_runtime(self):
        self._settings = self._load_settings()
        runtime = self._runtime_settings()
        self._assign_runtime_settings(runtime, startup=True)
        self._incognito_enabled = False

        self.selected_anime: Optional[Any] = None
        self.selected_search_item: Optional[SearchItem] = None
        self.episodes_list: list[float | int] = []
        self.current_ep: float | int | None = None
        self._aw_provider_instances: dict[str, Any] = {}
        self._aw_anime_cls: Any = None
        self._aw_config_ready = False
        self._aw_runtime_ready = False
        self._aw_cover_url_cache: dict[str, str | None] = {}
        self._aw_cover_miss_until: dict[str, float] = {}

        self.history = HistoryStore(HISTORY_PATH)
        self.cover_cache_dir = os.path.join(app_state_dir(), "covers")
        os.makedirs(self.cover_cache_dir, exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)

    def _runtime_settings(self) -> RuntimeSettings:
        return SettingsService.runtime_state(
            self._settings,
            state_dir=app_state_dir(),
            normalize_token=self._normalize_anilist_token,
        )

    def _assign_runtime_settings(self, runtime: RuntimeSettings, *, startup: bool = False) -> None:
        self.download_dir = runtime.download_dir
        self.provider_name = runtime.provider_name
        self.lang = LanguageTypeEnum.DUB if runtime.lang_name == "DUB" else LanguageTypeEnum.SUB
        self.quality = runtime.quality
        if startup:
            self._startup_parallel_downloads = runtime.parallel_downloads
        self._max_parallel_downloads = runtime.parallel_downloads
        self._scheduler_enabled = runtime.scheduler_enabled
        self._scheduler_start = runtime.scheduler_start
        self._scheduler_end = runtime.scheduler_end
        self._integrity_min_mb = runtime.integrity_min_mb
        self._integrity_retry_count = runtime.integrity_retry_count
        self._anilist_enabled = runtime.anilist_enabled
        self._anilist_token = runtime.anilist_token
        self.app_language = runtime.app_language

    def _build_search_and_player_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self._outer_layout = outer
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        QShortcut(QKeySequence("F"), self, activated=self.toggle_fullscreen)
        QShortcut(
            QKeySequence("Escape"),
            self,
            activated=lambda: self._exit_video_fullscreen() if self._is_video_fullscreen() else None,
        )

        # top bar
        top = QHBoxLayout()
        top.setSpacing(8)
        outer.addLayout(top)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cerca anime…")
        self.search_input.returnPressed.connect(self.on_search)
        self.search_input.textChanged.connect(self.on_search_text_changed)
        top.addWidget(self.search_input, 1)

        self.btn_search = QPushButton("🔎 Cerca")
        self.btn_search.clicked.connect(self.on_search)
        self.btn_search.setToolTip("Avvia ricerca anime")
        top.addWidget(self.btn_search)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("AllAnime", "allanime")
        self.provider_combo.addItem("AnimeWorld ITA", "aw_animeworld")
        self.provider_combo.addItem("AnimeUnity ITA", "aw_animeunity")
        self.provider_combo.currentIndexChanged.connect(self.on_provider_change)
        top.addWidget(self.provider_combo)

        self.lang_combo = QComboBox()
        self.lang_combo.addItem("SUB", LanguageTypeEnum.SUB)
        self.lang_combo.addItem("DUB", LanguageTypeEnum.DUB)
        self.lang_combo.currentIndexChanged.connect(self.on_lang_change)
        top.addWidget(self.lang_combo)

        self.quality_combo = QComboBox()
        for q in ["best", "worst", "360", "480", "720", "1080"]:
            self.quality_combo.addItem(q)
        self.quality_combo.currentIndexChanged.connect(self.on_quality_change)
        top.addWidget(self.quality_combo)

        self.provider_combo.blockSignals(True)
        self.lang_combo.blockSignals(True)
        self.quality_combo.blockSignals(True)
        pidx = self.provider_combo.findData(self.provider_name)
        if pidx >= 0:
            self.provider_combo.setCurrentIndex(pidx)
        self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.quality_combo.findText(self.quality)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)
        self.provider_combo.blockSignals(False)
        self.lang_combo.blockSignals(False)
        self.quality_combo.blockSignals(False)
        self._apply_provider_ui_state()

        self.lbl_spinner = QLabel("")
        self.lbl_spinner.setFixedWidth(24)
        top.addWidget(self.lbl_spinner)

        self.btn_incognito = QPushButton("Incognito OFF")
        self.btn_incognito.setCheckable(True)
        self.btn_incognito.toggled.connect(self.on_toggle_incognito)
        self.btn_incognito.setToolTip("Disabilita salvataggi locali e cache")
        top.addWidget(self.btn_incognito)
        self.lbl_incognito_badge = QLabel("INCOGNITO")
        self.lbl_incognito_badge.setObjectName("incognitoBadge")
        self.lbl_incognito_badge.setVisible(False)
        top.addWidget(self.lbl_incognito_badge)

        # tabs: Search / History
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        # ---- Search tab ----
        self.tab_search = QWidget()
        self.tabs.addTab(self.tab_search, "Search")
        search_layout = QVBoxLayout(self.tab_search)

        self.suggestions_panel = QWidget()
        self.suggestions_panel.setObjectName("suggestionsPanel")
        self.suggestions_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sugg_layout = QVBoxLayout(self.suggestions_panel)
        sugg_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_suggestions = QLabel("Suggerimenti")
        self.lbl_suggestions.setObjectName("sectionTitle")
        sugg_layout.addWidget(self.lbl_suggestions)

        sugg_row = QHBoxLayout()
        self.list_recent_queries = QListWidget()
        self.list_recent_queries.setObjectName("suggestionsList")
        self.list_recent_queries.itemClicked.connect(self.on_pick_recent_query)
        self.list_recent_queries.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list_recent_queries.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list_recent_queries.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        sugg_row.addWidget(self.list_recent_queries, 1)

        sugg_actions = QHBoxLayout()
        self.btn_clear_search_history = QPushButton("🧹 Clear search history")
        self.btn_clear_search_history.clicked.connect(self.on_clear_search_history)
        sugg_actions.addStretch(1)
        sugg_actions.addWidget(self.btn_clear_search_history)
        sugg_layout.addLayout(sugg_actions)

        sugg_layout.addLayout(sugg_row, 1)
        search_layout.addWidget(self.suggestions_panel)

        self.search_stack = QStackedWidget()
        search_layout.addWidget(self.search_stack, 1)

        # catalog page (full tab)
        self.page_catalog = QWidget()
        cat_layout = QVBoxLayout(self.page_catalog)
        cat_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_search_meta = QLabel("")
        self.lbl_search_meta.setObjectName("sectionTitle")
        self.lbl_search_meta.setVisible(False)
        cat_layout.addWidget(self.lbl_search_meta)
        self.lbl_search_state = QLabel("")
        self.lbl_search_state.setObjectName("mutedLabel")
        self.lbl_search_state.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.lbl_search_state.setVisible(False)
        cat_layout.addWidget(self.lbl_search_state)

        filters_row = QHBoxLayout()
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Rilevanza", "Titolo A→Z", "Titolo Z→A"])
        self.combo_sort.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_sort)

        self.combo_season = QComboBox()
        self.combo_season.addItems(["Stagione: Qualsiasi", "Winter", "Spring", "Summer", "Fall"])
        self.combo_season.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_season)

        self.combo_year = QComboBox()
        self.combo_year.addItem("Anno: Qualsiasi")
        current_year = time.localtime().tm_year
        for y in range(current_year, current_year - 25, -1):
            self.combo_year.addItem(str(y))
        self.combo_year.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_year)

        self.input_studio = QLineEdit()
        self.input_studio.setPlaceholderText("Studio")
        self.input_studio.textChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.input_studio)

        self.combo_rating = QComboBox()
        self.combo_rating.addItems(["Rating: Qualsiasi", ">= 9", ">= 8", ">= 7", ">= 6"])
        self.combo_rating.currentIndexChanged.connect(self.on_filters_changed)
        filters_row.addWidget(self.combo_rating)
        self._search_filter_widgets = [
            self.combo_sort,
            self.combo_season,
            self.combo_year,
            self.input_studio,
            self.combo_rating,
        ]

        cat_layout.addLayout(filters_row)

        self.results = QListWidget()
        self.results.itemClicked.connect(self.on_pick_result)
        self.results.setViewMode(QListView.ViewMode.IconMode)
        self.results.setResizeMode(QListView.ResizeMode.Adjust)
        self.results.setMovement(QListView.Movement.Static)
        self.results.setWrapping(True)
        self.results.setSpacing(14)
        self.results.setIconSize(QSize(150, 220))
        self.results.setWordWrap(True)
        cat_layout.addWidget(self.results, 1)

        self.search_stack.addWidget(self.page_catalog)

        # anime page (episodes list only)
        self.page_anime = QWidget()
        anime_layout = QVBoxLayout(self.page_anime)
        anime_layout.setContentsMargins(0, 0, 0, 0)

        anime_header = QHBoxLayout()
        self.btn_back_catalog = QPushButton("← Back to Search")
        self.btn_back_catalog.clicked.connect(self.on_back_to_catalog)
        anime_header.addWidget(self.btn_back_catalog, 0)
        self.lbl_anime_title = QLabel("Anime")
        self.lbl_anime_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        anime_header.addWidget(self.lbl_anime_title, 1)
        self.btn_queue_selected_eps = QPushButton("⬇ Queue selected")
        self.btn_queue_selected_eps.clicked.connect(self.on_download_add_selected_episodes)
        anime_header.addWidget(self.btn_queue_selected_eps, 0)
        self.btn_fav_anime = QPushButton("❤ Favorite")
        self.btn_fav_anime.clicked.connect(self.on_favorite_add_current)
        anime_header.addWidget(self.btn_fav_anime, 0)
        self.btn_mark_planned_anime = QPushButton("📝 Planned")
        self.btn_mark_planned_anime.clicked.connect(self.on_mark_anime_planned)
        anime_header.addWidget(self.btn_mark_planned_anime, 0)
        self.btn_mark_completed_anime = QPushButton("✅ Completed")
        self.btn_mark_completed_anime.clicked.connect(self.on_mark_anime_completed)
        anime_header.addWidget(self.btn_mark_completed_anime, 0)
        anime_layout.addLayout(anime_header)

        self.episodes = QListWidget()
        self.episodes.itemDoubleClicked.connect(self.on_play_selected_episode)
        self.episodes.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.episodes.setSpacing(8)
        anime_layout.addWidget(QLabel("Episodi"), 0)
        anime_layout.addWidget(self.episodes, 1)
        self.search_stack.addWidget(self.page_anime)

        # player page (full area)
        self.page_player = QWidget()
        player_page_layout = QVBoxLayout(self.page_player)
        player_page_layout.setContentsMargins(0, 0, 0, 0)
        player_page_layout.setSpacing(8)

        player_header = QHBoxLayout()
        self.btn_back_episodes = QPushButton("← Back to Episodes")
        self.btn_back_episodes.clicked.connect(self.on_back_to_episodes)
        player_header.addWidget(self.btn_back_episodes, 0)
        self.lbl_player_title = QLabel("Player")
        self.lbl_player_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        player_header.addWidget(self.lbl_player_title, 1)
        player_page_layout.addLayout(player_header)

        self.player_section = QWidget()
        player_section_layout = QVBoxLayout(self.player_section)
        player_section_layout.setContentsMargins(0, 0, 0, 0)
        player_section_layout.setSpacing(8)

        self.player = create_player_widget()
        self.player_host = QWidget()
        self.player_host_layout = QVBoxLayout(self.player_host)
        self.player_host_layout.setContentsMargins(0, 0, 0, 0)
        self.player_host_layout.setSpacing(0)
        self.player_host_layout.addWidget(self.player, 1)
        player_section_layout.addWidget(self.player_host, 1)

        # controls group
        controls = QGroupBox("Controls")
        c = QVBoxLayout(controls)

        row1 = QHBoxLayout()

        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.clicked.connect(self.on_prev_episode)
        row1.addWidget(self.btn_prev)

        self.btn_playpause = QPushButton("⏯ Play/Pause")
        self.btn_playpause.clicked.connect(self.on_toggle_pause)
        row1.addWidget(self.btn_playpause)

        self.btn_next = QPushButton("⏭ Next")
        self.btn_next.clicked.connect(self.on_next_episode)
        row1.addWidget(self.btn_next)

        self.btn_fs = QPushButton("⛶ Fullscreen")
        self.btn_fs.clicked.connect(self.toggle_fullscreen)
        row1.addWidget(self.btn_fs)

        self.btn_mini = QPushButton("▣ Mini")
        self.btn_mini.clicked.connect(self.toggle_mini_player)
        row1.addWidget(self.btn_mini)

        c.addLayout(row1)

        # seek slider + labels
        row2 = QHBoxLayout()
        self.lbl_time = QLabel("00:00 / 00:00")
        row2.addWidget(self.lbl_time)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        row2.addWidget(self.seek_slider, 1)
        self._resume_marker = QFrame(self.seek_slider)
        self._resume_marker.setFixedWidth(2)
        self._resume_marker.setStyleSheet("background: #e50914;")
        self._resume_marker.hide()
        self._resume_marker_pos: float | None = None
        self.seek_slider.installEventFilter(self)
        c.addLayout(row2)

        # volume
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Volume"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 130)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self.on_volume_changed)
        row3.addWidget(self.vol_slider, 1)
        c.addLayout(row3)

        player_section_layout.addWidget(controls, 0)
        player_page_layout.addWidget(self.player_section, 1)
        self.search_stack.addWidget(self.page_player)
        self.search_stack.setCurrentWidget(self.page_catalog)

    def _build_recommended_tab(self):
        # ---- Recommended tab ----
        self.tab_recommended = QWidget()
        self.tabs.addTab(self.tab_recommended, "Recommended")
        rec_layout = QVBoxLayout(self.tab_recommended)

        rec_top = QHBoxLayout()
        self.lbl_recommended = QLabel("Recent Releases + Recommended")
        rec_top.addWidget(self.lbl_recommended, 1)
        self.btn_refresh_recommended = QPushButton("↻ Aggiorna")
        self.btn_refresh_recommended.clicked.connect(self.on_refresh_recommended)
        rec_top.addWidget(self.btn_refresh_recommended, 0)
        rec_layout.addLayout(rec_top)

        rec_split = QSplitter(Qt.Orientation.Horizontal)
        rec_layout.addWidget(rec_split, 1)

        recent_col = QWidget()
        recent_col_layout = QVBoxLayout(recent_col)
        recent_col_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_recent_header = QLabel("Recent Releases")
        self.lbl_recent_header.setObjectName("recentHeader")
        recent_col_layout.addWidget(self.lbl_recent_header)

        self.recent = QListWidget()
        self.recent.itemClicked.connect(self.on_pick_recent)
        self.recent.setViewMode(QListView.ViewMode.IconMode)
        self.recent.setResizeMode(QListView.ResizeMode.Adjust)
        self.recent.setMovement(QListView.Movement.Static)
        self.recent.setWrapping(True)
        self.recent.setSpacing(14)
        self.recent.setIconSize(QSize(150, 220))
        self.recent.setWordWrap(True)
        recent_col_layout.addWidget(self.recent, 1)
        rec_split.addWidget(recent_col)

        recommended_col = QWidget()
        recommended_col_layout = QVBoxLayout(recommended_col)
        recommended_col_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_recommended_header = QLabel("Recommended")
        self.lbl_recommended_header.setObjectName("recommendedHeader")
        recommended_col_layout.addWidget(self.lbl_recommended_header)

        self.recommended = QListWidget()
        self.recommended.itemClicked.connect(self.on_pick_recommended)
        self.recommended.setViewMode(QListView.ViewMode.IconMode)
        self.recommended.setResizeMode(QListView.ResizeMode.Adjust)
        self.recommended.setMovement(QListView.Movement.Static)
        self.recommended.setWrapping(True)
        self.recommended.setSpacing(14)
        self.recommended.setIconSize(QSize(150, 220))
        self.recommended.setWordWrap(True)
        recommended_col_layout.addWidget(self.recommended, 1)
        rec_split.addWidget(recommended_col)
        rec_split.setStretchFactor(0, 1)
        rec_split.setStretchFactor(1, 1)

    def _build_featured_tab(self):
        self.tab_featured = QWidget()
        self.tabs.addTab(self.tab_featured, "Featured")
        feat_layout = QVBoxLayout(self.tab_featured)

        feat_top = QHBoxLayout()
        self.lbl_featured_header = QLabel("Featured for You")
        self.lbl_featured_header.setObjectName("featuredHeader")
        self.lbl_featured_header.setToolTip(self._featured_algorithm_tooltip())
        feat_top.addWidget(self.lbl_featured_header, 1)
        self.btn_refresh_featured = QPushButton("↻ Aggiorna")
        self.btn_refresh_featured.clicked.connect(lambda: self.on_refresh_featured(force=True))
        self.btn_refresh_featured.setToolTip(self._featured_algorithm_tooltip())
        feat_top.addWidget(self.btn_refresh_featured, 0)
        feat_layout.addLayout(feat_top)

        self.featured = QListWidget()
        self.featured.itemClicked.connect(self.on_pick_featured)
        self.featured.setViewMode(QListView.ViewMode.IconMode)
        self.featured.setResizeMode(QListView.ResizeMode.Adjust)
        self.featured.setMovement(QListView.Movement.Static)
        self.featured.setWrapping(True)
        self.featured.setSpacing(14)
        self.featured.setIconSize(QSize(150, 220))
        self.featured.setWordWrap(True)
        self.featured.setToolTip(self._featured_algorithm_tooltip())
        feat_layout.addWidget(self.featured, 1)

    def _build_downloads_tab(self):
        # ---- Downloads tab ----
        self.tab_downloads = QWidget()
        self.tabs.addTab(self.tab_downloads, "Downloads")
        dl_layout = QVBoxLayout(self.tab_downloads)

        dl_top = QHBoxLayout()
        self.btn_dl_add_current = QPushButton("＋ Aggiungi episodio corrente")
        self.btn_dl_add_current.clicked.connect(self.on_download_add_current)
        dl_top.addWidget(self.btn_dl_add_current)

        self.btn_dl_start = QPushButton("▶ Avvia coda")
        self.btn_dl_start.clicked.connect(self.on_download_start_queue)
        self.btn_dl_start.setToolTip("Avvia i download in coda")
        dl_top.addWidget(self.btn_dl_start)

        self.combo_dl_parallel = QComboBox()
        self.combo_dl_parallel.addItem("Parallel x1", 1)
        self.combo_dl_parallel.addItem("Parallel x2", 2)
        self.combo_dl_parallel.addItem("Parallel x3", 3)
        self.combo_dl_parallel.addItem("Parallel x4", 4)
        self.combo_dl_parallel.setCurrentIndex(self._startup_parallel_downloads - 1)
        self.combo_dl_parallel.currentIndexChanged.connect(self.on_download_parallel_change)
        dl_top.addWidget(self.combo_dl_parallel)

        self.btn_dl_cancel = QPushButton("✖ Annulla selezionato")
        self.btn_dl_cancel.clicked.connect(self.on_download_cancel_selected)
        dl_top.addWidget(self.btn_dl_cancel)

        self.btn_dl_clear = QPushButton("🧹 Pulisci completati")
        self.btn_dl_clear.clicked.connect(self.on_download_clear_completed)
        dl_top.addWidget(self.btn_dl_clear)

        self.btn_dl_open = QPushButton("📂 Open folder")
        self.btn_dl_open.clicked.connect(self.on_download_open_folder)
        dl_top.addWidget(self.btn_dl_open)
        dl_layout.addLayout(dl_top)

        self.downloads_list = QListWidget()
        dl_layout.addWidget(self.downloads_list, 1)

    def _build_offline_tab(self):
        # ---- Offline tab ----
        self.tab_offline = QWidget()
        self.tabs.addTab(self.tab_offline, "Offline")
        off_layout = QVBoxLayout(self.tab_offline)

        off_top = QHBoxLayout()
        self.lbl_offline = QLabel("Anime scaricati (stream locale)")
        off_top.addWidget(self.lbl_offline, 1)
        self.btn_offline_refresh = QPushButton("↻ Aggiorna")
        self.btn_offline_refresh.clicked.connect(self.refresh_offline_library)
        off_top.addWidget(self.btn_offline_refresh, 0)
        self.btn_offline_open = QPushButton("📂 Open downloads")
        self.btn_offline_open.clicked.connect(self.on_download_open_folder)
        off_top.addWidget(self.btn_offline_open, 0)
        off_layout.addLayout(off_top)

        self.offline_stack = QStackedWidget()
        off_layout.addWidget(self.offline_stack, 1)

        self.page_offline_catalog = QWidget()
        off_cat_layout = QVBoxLayout(self.page_offline_catalog)
        off_cat_layout.setContentsMargins(0, 0, 0, 0)
        off_cat_layout.addWidget(QLabel("Catalogo Offline"))
        self.offline_results = QListWidget()
        self.offline_results.itemClicked.connect(self.on_pick_offline_anime)
        self.offline_results.setViewMode(QListView.ViewMode.IconMode)
        self.offline_results.setResizeMode(QListView.ResizeMode.Adjust)
        self.offline_results.setMovement(QListView.Movement.Static)
        self.offline_results.setWrapping(True)
        self.offline_results.setSpacing(14)
        self.offline_results.setIconSize(QSize(150, 220))
        self.offline_results.setWordWrap(True)
        self.offline_results.setObjectName("resultsList")
        off_cat_layout.addWidget(self.offline_results, 1)
        self.offline_stack.addWidget(self.page_offline_catalog)

        self.page_offline_anime = QWidget()
        off_anime_layout = QVBoxLayout(self.page_offline_anime)
        off_anime_layout.setContentsMargins(0, 0, 0, 0)
        off_header = QHBoxLayout()
        self.btn_offline_back = QPushButton("← Back to Offline Catalog")
        self.btn_offline_back.clicked.connect(self.on_offline_back_to_catalog)
        off_header.addWidget(self.btn_offline_back, 0)
        self.lbl_offline_anime_title = QLabel("Anime")
        off_header.addWidget(self.lbl_offline_anime_title, 1)
        off_anime_layout.addLayout(off_header)
        self.offline_episodes = QListWidget()
        self.offline_episodes.itemDoubleClicked.connect(self.on_play_offline_episode)
        self.offline_episodes.itemClicked.connect(self.on_play_offline_episode)
        self.offline_episodes.setObjectName("episodesList")
        off_anime_layout.addWidget(QLabel("Episodi / File"), 0)
        off_anime_layout.addWidget(self.offline_episodes, 1)
        self.offline_stack.addWidget(self.page_offline_anime)
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)

    def _build_favorites_tab(self):
        # ---- Favorites tab ----
        self.tab_favorites = QWidget()
        self.tabs.addTab(self.tab_favorites, "Favorites")
        fav_layout = QVBoxLayout(self.tab_favorites)
        fav_top = QHBoxLayout()
        fav_top.addWidget(QLabel("Watchlist / Favorites"), 1)
        self.btn_fav_add_current = QPushButton("❤ Add current anime")
        self.btn_fav_add_current.clicked.connect(self.on_favorite_add_current)
        fav_top.addWidget(self.btn_fav_add_current, 0)
        self.btn_fav_remove = QPushButton("🗑 Remove selected")
        self.btn_fav_remove.clicked.connect(self.on_favorite_remove_selected)
        fav_top.addWidget(self.btn_fav_remove, 0)
        fav_layout.addLayout(fav_top)
        self.favorites_list = QListWidget()
        self.favorites_list.itemClicked.connect(self.on_pick_favorite)
        self.favorites_list.setViewMode(QListView.ViewMode.IconMode)
        self.favorites_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.favorites_list.setMovement(QListView.Movement.Static)
        self.favorites_list.setWrapping(True)
        self.favorites_list.setSpacing(14)
        self.favorites_list.setIconSize(QSize(150, 220))
        self.favorites_list.setWordWrap(True)
        self.favorites_list.setObjectName("resultsList")
        fav_layout.addWidget(self.favorites_list, 1)

    def _build_settings_tab(self):
        # ---- Settings tab ----
        self.tab_settings = QWidget()
        self.tabs.addTab(self.tab_settings, "Settings")
        settings_layout = QVBoxLayout(self.tab_settings)

        row_dl = QHBoxLayout()
        self.lbl_settings_download_dir = QLabel("Download directory")
        row_dl.addWidget(self.lbl_settings_download_dir, 0)
        self.input_settings_download_dir = QLineEdit(self.download_dir)
        row_dl.addWidget(self.input_settings_download_dir, 1)
        self.btn_settings_browse_download_dir = QPushButton("Sfoglia")
        self.btn_settings_browse_download_dir.clicked.connect(self.on_settings_browse_download_dir)
        row_dl.addWidget(self.btn_settings_browse_download_dir, 0)
        settings_layout.addLayout(row_dl)

        row_defaults = QHBoxLayout()
        self.lbl_settings_default_provider = QLabel("Default provider")
        row_defaults.addWidget(self.lbl_settings_default_provider, 0)
        self.combo_settings_provider = QComboBox()
        self.combo_settings_provider.addItem("AllAnime", "allanime")
        self.combo_settings_provider.addItem("AnimeWorld ITA", "aw_animeworld")
        self.combo_settings_provider.addItem("AnimeUnity ITA", "aw_animeunity")
        row_defaults.addWidget(self.combo_settings_provider, 1)
        self.lbl_settings_default_lang = QLabel("Default language")
        row_defaults.addWidget(self.lbl_settings_default_lang, 0)
        self.combo_settings_lang = QComboBox()
        self.combo_settings_lang.addItem("SUB", "SUB")
        self.combo_settings_lang.addItem("DUB", "DUB")
        row_defaults.addWidget(self.combo_settings_lang, 1)
        settings_layout.addLayout(row_defaults)

        row_app_lang = QHBoxLayout()
        self.lbl_settings_app_language = QLabel("App language")
        row_app_lang.addWidget(self.lbl_settings_app_language, 0)
        self.combo_settings_app_language = QComboBox()
        self.combo_settings_app_language.addItem("Italiano", "it")
        self.combo_settings_app_language.addItem("English", "en")
        row_app_lang.addWidget(self.combo_settings_app_language, 1)
        row_app_lang.addStretch(1)
        settings_layout.addLayout(row_app_lang)

        row_quality = QHBoxLayout()
        self.lbl_settings_default_quality = QLabel("Default quality")
        row_quality.addWidget(self.lbl_settings_default_quality, 0)
        self.combo_settings_quality = QComboBox()
        for q in ["best", "worst", "360", "480", "720", "1080"]:
            self.combo_settings_quality.addItem(q, q)
        row_quality.addWidget(self.combo_settings_quality, 1)
        self.lbl_settings_parallel_downloads = QLabel("Parallel downloads")
        row_quality.addWidget(self.lbl_settings_parallel_downloads, 0)
        self.combo_settings_parallel = QComboBox()
        self.combo_settings_parallel.addItem("1", 1)
        self.combo_settings_parallel.addItem("2", 2)
        self.combo_settings_parallel.addItem("3", 3)
        self.combo_settings_parallel.addItem("4", 4)
        row_quality.addWidget(self.combo_settings_parallel, 1)
        settings_layout.addLayout(row_quality)

        row_sched = QHBoxLayout()
        self.lbl_settings_scheduler = QLabel("Scheduler")
        row_sched.addWidget(self.lbl_settings_scheduler, 0)
        self.combo_settings_scheduler_enabled = QComboBox()
        self.combo_settings_scheduler_enabled.addItem("Disabled", False)
        self.combo_settings_scheduler_enabled.addItem("Enabled", True)
        row_sched.addWidget(self.combo_settings_scheduler_enabled, 1)
        self.lbl_settings_scheduler_start = QLabel("Start HH:MM")
        row_sched.addWidget(self.lbl_settings_scheduler_start, 0)
        self.input_settings_scheduler_start = QLineEdit(self._scheduler_start)
        self.input_settings_scheduler_start.setMaxLength(5)
        row_sched.addWidget(self.input_settings_scheduler_start, 1)
        self.lbl_settings_scheduler_end = QLabel("End HH:MM")
        row_sched.addWidget(self.lbl_settings_scheduler_end, 0)
        self.input_settings_scheduler_end = QLineEdit(self._scheduler_end)
        self.input_settings_scheduler_end.setMaxLength(5)
        row_sched.addWidget(self.input_settings_scheduler_end, 1)
        settings_layout.addLayout(row_sched)

        row_integrity = QHBoxLayout()
        self.lbl_settings_integrity_min_mb = QLabel("Integrity min MB")
        row_integrity.addWidget(self.lbl_settings_integrity_min_mb, 0)
        self.input_settings_integrity_min_mb = QLineEdit(str(self._integrity_min_mb))
        row_integrity.addWidget(self.input_settings_integrity_min_mb, 1)
        self.lbl_settings_retry_count = QLabel("Retry count")
        row_integrity.addWidget(self.lbl_settings_retry_count, 0)
        self.combo_settings_integrity_retries = QComboBox()
        for i in range(0, 6):
            self.combo_settings_integrity_retries.addItem(str(i), i)
        row_integrity.addWidget(self.combo_settings_integrity_retries, 1)
        settings_layout.addLayout(row_integrity)

        row_anilist = QHBoxLayout()
        self.lbl_settings_anilist_sync = QLabel("AniList sync")
        row_anilist.addWidget(self.lbl_settings_anilist_sync, 0)
        self.combo_settings_anilist_enabled = QComboBox()
        self.combo_settings_anilist_enabled.addItem("Disabled", False)
        self.combo_settings_anilist_enabled.addItem("Enabled", True)
        row_anilist.addWidget(self.combo_settings_anilist_enabled, 1)
        self.lbl_settings_anilist_token = QLabel("AniList token")
        row_anilist.addWidget(self.lbl_settings_anilist_token, 0)
        self.input_settings_anilist_token = QLineEdit(self._anilist_token)
        self.input_settings_anilist_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_settings_anilist_token.setPlaceholderText("Personal Access Token")
        row_anilist.addWidget(self.input_settings_anilist_token, 2)
        settings_layout.addLayout(row_anilist)

        row_actions = QHBoxLayout()
        self.btn_settings_save = QPushButton("Salva impostazioni")
        self.btn_settings_save.clicked.connect(self.on_settings_save)
        row_actions.addWidget(self.btn_settings_save, 0)
        self.btn_settings_reset = QPushButton("Reset default")
        self.btn_settings_reset.clicked.connect(self.on_settings_reset)
        row_actions.addWidget(self.btn_settings_reset, 0)
        self.btn_settings_backup = QPushButton("Backup")
        self.btn_settings_backup.clicked.connect(self.on_settings_backup)
        row_actions.addWidget(self.btn_settings_backup, 0)
        self.btn_settings_restore = QPushButton("Restore")
        self.btn_settings_restore.clicked.connect(self.on_settings_restore)
        row_actions.addWidget(self.btn_settings_restore, 0)
        self.btn_settings_anilist_test = QPushButton("Test connessione AniList")
        self.btn_settings_anilist_test.clicked.connect(self.on_anilist_test_connection)
        row_actions.addWidget(self.btn_settings_anilist_test, 0)
        self.btn_settings_anilist_sync_now = QPushButton("Sync now")
        self.btn_settings_anilist_sync_now.clicked.connect(self.on_anilist_sync_now)
        self.btn_settings_anilist_sync_now.setToolTip("Invia subito il progresso corrente ad AniList")
        row_actions.addWidget(self.btn_settings_anilist_sync_now, 0)
        self.btn_settings_anilist_pull = QPushButton("Importa da AniList")
        self.btn_settings_anilist_pull.clicked.connect(self.on_anilist_pull_from_remote)
        self.btn_settings_anilist_pull.setToolTip("Importa watchlist/progressi da AniList")
        row_actions.addWidget(self.btn_settings_anilist_pull, 0)
        self.btn_settings_update_check = QPushButton("Check updates")
        self.btn_settings_update_check.clicked.connect(self.on_update_check_clicked)
        row_actions.addWidget(self.btn_settings_update_check, 0)
        self.btn_settings_update_apply = QPushButton("Apply update")
        self.btn_settings_update_apply.clicked.connect(self.on_update_apply_clicked)
        self.btn_settings_update_apply.setEnabled(False)
        row_actions.addWidget(self.btn_settings_update_apply, 0)
        row_actions.addStretch(1)
        settings_layout.addLayout(row_actions)
        settings_layout.addStretch(1)

    def _build_history_tab(self):
        # existing history tab code continues as is
        # ---- History tab ----
        self.tab_history = QWidget()
        self.tabs.addTab(self.tab_history, "Watchlist")
        hist_layout = QVBoxLayout(self.tab_history)

        hist_top = QHBoxLayout()
        self.lbl_history_header = QLabel("Watchlist: Planned · Watching · Paused · Dropped · Completed")
        hist_top.addWidget(self.lbl_history_header, 1)
        self.lbl_history_filter = QLabel("Filter")
        hist_top.addWidget(self.lbl_history_filter, 0)
        self.combo_history_filter = QComboBox()
        self.combo_history_filter.addItem("All", "All")
        self.combo_history_filter.addItem("Completed", "Completed")
        self.combo_history_filter.addItem("Watching", "Watching")
        self.combo_history_filter.addItem("Paused", "Paused")
        self.combo_history_filter.addItem("Planned", "Planned")
        self.combo_history_filter.addItem("Dropped", "Dropped")
        self.combo_history_filter.currentIndexChanged.connect(self.on_history_filter_changed)
        hist_top.addWidget(self.combo_history_filter, 0)
        hist_layout.addLayout(hist_top)

        self.hist_list = QListWidget()
        self.hist_list.itemClicked.connect(self.on_pick_history)
        self.hist_list.itemDoubleClicked.connect(self.on_history_open_details)
        self.hist_list.setViewMode(QListView.ViewMode.IconMode)
        self.hist_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.hist_list.setMovement(QListView.Movement.Static)
        self.hist_list.setWrapping(True)
        self.hist_list.setFlow(QListView.Flow.LeftToRight)
        self.hist_list.setSpacing(8)
        self.hist_list.setWordWrap(False)
        self.hist_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hist_list.setObjectName("continueList")
        self.hist_list.verticalScrollBar().valueChanged.connect(self._position_history_section_overlays)
        hist_layout.addWidget(self.hist_list, 1)

        hist_buttons = QHBoxLayout()

        self.btn_hist_resume = QPushButton("▶ Resume selected")
        self.btn_hist_resume.clicked.connect(self.on_history_resume)
        hist_buttons.addWidget(self.btn_hist_resume)


        self.btn_hist_resume_next = QPushButton("⏭ Resume next")
        self.btn_hist_resume_next.clicked.connect(self.on_history_resume_next)
        hist_buttons.addWidget(self.btn_hist_resume_next)

        self.btn_hist_mark_seen = QPushButton("✅ Mark as seen")
        self.btn_hist_mark_seen.clicked.connect(self.on_history_mark_seen)
        hist_buttons.addWidget(self.btn_hist_mark_seen)

        self.btn_hist_delete = QPushButton("🗑 Remove")
        self.btn_hist_delete.clicked.connect(self.on_history_delete)
        hist_buttons.addWidget(self.btn_hist_delete)

        hist_layout.addLayout(hist_buttons)

        hist_clear = QHBoxLayout()
        self.btn_clear_watch_history = QPushButton("🧹 Clear watch history")
        self.btn_clear_watch_history.clicked.connect(self.on_clear_watch_history)
        hist_clear.addWidget(self.btn_clear_watch_history)
        hist_layout.addLayout(hist_clear)
    def _build_watchparty_tab(self):
        """Create the Watch‑Party tab and its UI."""
        from watchparty.watchparty_manager import WatchPartySession
        from watchparty.watchparty_ui import WatchPartyTab
        import uuid
        self.tab_watchparty = QWidget()
        self.tabs.addTab(self.tab_watchparty, "Watch‑Party")
        layout = QVBoxLayout(self.tab_watchparty)
        self.watchparty_session = WatchPartySession(session_id=str(uuid.uuid4()))
        self.watchparty_session.get_host_state_snapshot = self._watchparty_current_state_snapshot
        self.watchparty_ui = WatchPartyTab(self.watchparty_session)
        self.watchparty_ui.playback_received.connect(self._apply_watchparty_playback)
        self.watchparty_ui.media_change_received.connect(self._apply_watchparty_media_change)
        layout.addWidget(self.watchparty_ui)

    def _build_status_bar(self):
        # status
        self.status = QLabel("Pronto.")
        self.status.setObjectName("statusBar")
        self._outer_layout.addWidget(self.status)

    def _init_runtime_fields(self):
        # internal
        self._workers: list[Worker] = []
        self._result_items: list[SearchItem] = []
        self._history_items: list[HistoryEntry] = []
        self._history_filter = "All"
        self._history_seen_eps_for_ui: set[float] | None = None
        self._history_episode_progress_for_ui: dict[float, dict[str, Any]] | None = None
        self._history_cover_labels: dict[int, QLabel] = {}
        self._history_section_overlays: dict[int, QWidget] = {}
        self._update_info: dict[str, Any] | None = None
        self._update_download_path: str | None = None
        self._update_service = UpdateService(UPDATE_MANIFEST_URL, APP_VERSION)
        self._search_cards: list[QListWidgetItem] = []
        self._recent_items: list[SearchItem] = []
        self._recent_cards: list[QListWidgetItem] = []
        self._recommended_items: list[SearchItem] = []
        self._recommended_cards: list[QListWidgetItem] = []
        self._featured_items: list[SearchItem] = []
        self._featured_cards: list[QListWidgetItem] = []
        self._featured_cache: dict[str, dict[str, Any]] = {}
        self._featured_provider_search_cache: dict[str, tuple[float, list[SearchItem]]] = {}
        self._favorite_items: list[FavoriteEntry] = []
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._download_tasks: dict[str, DownloadTask] = {}
        self._download_order: list[str] = []
        self._active_download_workers: dict[str, DownloadWorker] = {}
        self._active_download_resolve_ids: set[str] = set()
        self._max_parallel_downloads = self._startup_parallel_downloads
        self._search_nonce = 0
        self._recent_nonce = 0
        self._recommended_nonce = 0
        self._featured_nonce = 0
        self._recommended_loaded = False
        self._featured_loaded = False
        self._current_search_view = "catalog"
        self._recent_queries: list[str] = []
        self._offline_covers_map: dict[str, str] = {}
        self._last_search_query: str | None = None
        self._pending_search_query: str | None = None
        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.setInterval(350)
        self._search_debounce_timer.timeout.connect(self._run_debounced_search)
        self._filters_pending = False
        self._enrich_nonce = 0
        self._filtering_nonce = 0
        self._result_items_raw: list[SearchItem] = []
        self._requests_in_flight = 0
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._spin_tick)
        self._skeleton_phase = 0
        self._skeleton_timer = QTimer(self)
        self._skeleton_timer.setInterval(120)
        self._skeleton_timer.timeout.connect(self._update_skeletons)
        self._last_timeline_diag_at = 0.0
        self._timeline_missing_logged = False
        self._last_progress_save_at = 0.0
        self._last_progress_pos = -1.0
        self._last_progress_percent = -1.0
        self._pending_resume_seek: float | None = None
        self._pending_resume_seek_ratio: float | None = None
        self._pending_resume_seek_attempts = 0
        self._watchparty_applying_remote_control = False
        self._watchparty_pending_media_position: float | None = None
        self._watchparty_pending_media_playing: bool | None = None
        self._local_media_active = False
        self._offline_items: list[OfflineAnimeItem] = []
        self._offline_episode_files: list[str] = []
        self._offline_current_anime_dir: str | None = None
        self._offline_current_episode_index: int | None = None
        self._offline_nonce = 0
        self._anilist_media_id_cache: dict[str, int] = {}
        self._anilist_last_sync_ts: dict[str, float] = {}
        self._anilist_last_synced_progress: dict[str, int] = {}
        self._mini_player_window: QWidget | None = None
        self._video_fs_window: FullscreenPlayerWindow | None = None
        self._video_fs_shortcuts: list[QShortcut] = []
        self._closing_video_fs = False
        self._fs_overlay: QWidget | None = None
        self._fs_overlay_effect: QGraphicsOpacityEffect | None = None
        self._fs_overlay_anim: QPropertyAnimation | None = None
        self._fs_seek_slider: QSlider | None = None
        self._fs_vol_slider: QSlider | None = None
        self._fs_lbl_time: QLabel | None = None
        self._fs_btn_playpause: QPushButton | None = None
        self._fs_btn_prev: QPushButton | None = None
        self._fs_btn_next: QPushButton | None = None
        self._fs_btn_exit: QPushButton | None = None
        self._fs_help_label: QLabel | None = None
        self._fs_seeking = False
        self._fs_hide_delay_ms = 2200
        self._fs_autohide_timer = QTimer(self)
        self._fs_autohide_timer.setSingleShot(True)
        self._fs_autohide_timer.setInterval(self._fs_hide_delay_ms)
        self._fs_autohide_timer.timeout.connect(self._fs_apply_hidden)
        self._fs_last_cursor_pos = None
        self._fs_mouse_poll_timer = QTimer(self)
        self._fs_mouse_poll_timer.setInterval(120)
        self._fs_mouse_poll_timer.timeout.connect(self._fs_poll_cursor)

    def _init_timers(self):
        # seek tracking
        self._seeking = False

        # timers for UI update
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self.poll_player)
        self.poll_timer.start()
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.setInterval(30000)
        self.scheduler_timer.timeout.connect(self._scheduler_tick)
        self.scheduler_timer.start()

    def _finish_startup(self):
        self._apply_netflix_theme()
        self._apply_app_language_ui()
        self.recent.setObjectName("resultsList")
        self.recommended.setObjectName("resultsList")
        self.featured.setObjectName("resultsList")
        self._run_history_maintenance()
        self.refresh_history_ui()
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self._load_search_history()
        self._load_offline_covers_map()
        self._load_favorites()
        self._load_metadata_cache()
        self._sync_settings_ui_from_state()
        self._refresh_recent_queries_ui()
        self._update_suggestions_visibility()
        self._refresh_search_layout()
        self._update_featured_tab_visibility()
        self.refresh_offline_library()
        self.refresh_favorites_ui()
        QTimer.singleShot(300, self._debug_log_anilist_viewer_startup)
        QTimer.singleShot(1200, self.check_updates_silent)

    def _apply_watchparty_playback(self, payload: object):
        if not isinstance(payload, dict):
            return
        session = getattr(self, "watchparty_session", None)
        if session is None or session.role != "guest":
            return
        action = payload.get("action")
        position = payload.get("position")
        ratio = payload.get("ratio")
        self._watchparty_applying_remote_control = True
        try:
            if position is not None:
                self.player.seek(float(position))
            elif ratio is not None:
                self.player.seek_ratio(float(ratio))
            if action == "play":
                self.player.play()
            elif action == "pause":
                self.player.pause()
        except Exception:
            pass
        finally:
            self._watchparty_applying_remote_control = False

    def _apply_watchparty_media_change(self, payload: object):
        if not isinstance(payload, dict):
            return
        session = getattr(self, "watchparty_session", None)
        if session is None or session.role != "guest":
            return
        provider = str(payload.get("provider") or "").strip()
        identifier = str(payload.get("identifier") or "").strip()
        name = str(payload.get("name") or "").strip() or "anime"
        lang_name = str(payload.get("lang") or "SUB").upper()
        episode = payload.get("episode")
        if not provider or not identifier or episode is None:
            return
        self._watchparty_applying_remote_control = True
        try:
            self.tabs.setCurrentWidget(self.tab_search)
            self.provider_name = provider
            self.lang = LanguageTypeEnum.DUB if lang_name == "DUB" else LanguageTypeEnum.SUB
            try:
                pidx = self.provider_combo.findData(self.provider_name)
                if pidx >= 0:
                    self.provider_combo.blockSignals(True)
                    self.provider_combo.setCurrentIndex(pidx)
                    self.provider_combo.blockSignals(False)
            except Exception:
                pass
            try:
                self.lang_combo.blockSignals(True)
                self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
                self.lang_combo.blockSignals(False)
            except Exception:
                pass
            item = SearchItem(
                name=name,
                identifier=identifier,
                languages={self.lang},
                source=provider,
            )
            self.selected_search_item = item
            self.selected_anime = self.build_anime_from_item(item)
            self.lbl_anime_title.setText(name)
            self.lbl_player_title.setText(name)
            pos = payload.get("position")
            self._watchparty_pending_media_position = float(pos) if pos is not None else None
            playing = payload.get("playing")
            self._watchparty_pending_media_playing = None if playing is None else bool(playing)
            self.play_episode(episode)
        except Exception:
            self._watchparty_pending_media_position = None
            self._watchparty_pending_media_playing = None
            self._watchparty_applying_remote_control = False
            raise

    # ---------------- UI helpers ----------------
    def do_resolve_stream(
        self,
        anime: Any,
        ep: float | int,
        lang,
        preferred_quality,
        provider_name: str | None = None,
    ) -> StreamResult:
        provider_name = provider_name or self.provider_name
        t0 = time.perf_counter()
        debug_log(
            f"Resolving stream start: anime={anime.name}, ep={ep}, "
            f"lang={'SUB' if lang == LanguageTypeEnum.SUB else 'DUB'}, quality={preferred_quality}"
        )
        if self._is_aw_provider_name(provider_name):
            provider = self._aw_provider(provider_name)
            ep_key = str(int(ep)) if isinstance(ep, float) and ep.is_integer() else str(ep)
            if hasattr(anime, "has_episode") and anime.has_episode(ep_key):
                aw_ep = anime.episode(ep_key)
            else:
                aw_ep = anime.episode(str(ep))
            url = provider.episode_link(anime, aw_ep)
            debug_log(f"Resolving stream done in {time.perf_counter() - t0:.3f}s")
            return StreamResult(url=url, referrer=None, sub_file=None)

        stream = anime.get_video(
            episode=ep,
            lang=lang,
            preferred_quality=preferred_quality,
        )
        url = getattr(stream, "url", None) or str(stream)
        ref = getattr(stream, "referrer", None)
        debug_log(f"Resolving stream done in {time.perf_counter() - t0:.3f}s")
        return StreamResult(url=url, referrer=ref, sub_file=None)

    def notify_err(self, msg: str):
        title = self._tr("Errore", "Error")
        QMessageBox.critical(self, title, self._localize_runtime_text(str(msg)))

    def set_status(self, msg: str):
        self.status.setText(self._localize_runtime_text(str(msg)))

    def _track_worker(self, w: QThread):
        self._workers.append(w)

        def _cleanup_worker():
            try:
                self._workers.remove(w)
            except ValueError:
                pass
            try:
                w.deleteLater()
            except Exception:
                pass

        w.finished.connect(_cleanup_worker)

    def _tr(self, it: str, en: str) -> str:
        return en if self.app_language == "en" else it

    def _featured_algorithm_tooltip(self) -> str:
        return self._tr(
            "Questa lista usa AniList: calcola i tuoi generi piu guardati (pesando stato, progresso e score), "
            "cerca titoli affini, esclude quelli gia visti e ordina i risultati per match generi/popolarita/valutazione.",
            "This list uses AniList: it derives your most watched genres (weighted by status, progress and score), "
            "finds similar titles, excludes already watched ones, and ranks results by genre match/popularity/score.",
        )

    def _is_featured_tab_enabled(self) -> bool:
        return bool(self._anilist_enabled and self._normalize_anilist_token(self._anilist_token))

    def _update_featured_tab_visibility(self):
        if not hasattr(self, "tabs") or not hasattr(self, "tab_featured"):
            return
        idx = self.tabs.indexOf(self.tab_featured)
        if idx < 0:
            return
        visible = self._is_featured_tab_enabled()
        self.tabs.setTabVisible(idx, visible)
        if not visible and self.tabs.currentWidget() is self.tab_featured:
            self.tabs.setCurrentWidget(self.tab_recommended)

    def _translate_literal(self, text: str) -> str:
        if not text:
            return text
        pairs = [
            ("Search", "Cerca"),
            ("Recommended", "Consigliati"),
            ("Downloads", "Download"),
            ("Favorites", "Preferiti"),
            ("Settings", "Impostazioni"),
            ("Download directory", "Cartella download"),
            ("Default provider", "Provider predefinito"),
            ("Default language", "Lingua predefinita"),
            ("App language", "Lingua app"),
            ("Default quality", "Qualita predefinita"),
            ("Parallel downloads", "Download paralleli"),
            ("Start HH:MM", "Inizio HH:MM"),
            ("End HH:MM", "Fine HH:MM"),
            ("Retry count", "Numero retry"),
            ("Browse", "Sfoglia"),
            ("Save settings", "Salva impostazioni"),
            ("Reset defaults", "Reset predefiniti"),
            ("Restore", "Ripristina"),
            ("Test AniList connection", "Test connessione AniList"),
            ("Import from AniList", "Importa da AniList"),
            ("Filter", "Filtro"),
            ("All", "Tutti"),
            ("Completed", "Completati"),
            ("Watching", "In corso"),
            ("Paused", "In pausa"),
            ("Planned", "Da iniziare"),
            ("Dropped", "Interrotti"),
            ("Open folder", "Apri cartella"),
            ("Open downloads", "Apri download"),
            ("Back to Search", "Torna alla ricerca"),
            ("Back to Episodes", "Torna agli episodi"),
            ("Back to Offline Catalog", "Torna al catalogo offline"),
            ("Queue selected", "Metti in coda selezionati"),
            ("Favorite", "Preferito"),
            ("Prev", "Precedente"),
            ("Next", "Successivo"),
            ("Fullscreen", "Schermo intero"),
            ("Mini", "Mini Player"),
            ("Recent Releases", "Uscite recenti"),
            ("Suggestions", "Suggerimenti"),
            ("Volume", "Volume"),
            ("Controls", "Controlli"),
            ("Watchlist", "Lista visione"),
            ("Watchlist / Favorites", "Lista visione / Preferiti"),
            ("Episodes / Files", "Episodi / File"),
            ("Offline Catalog", "Catalogo Offline"),
            ("Episodes", "Episodi"),
            ("Typing...", "Scrittura in corso..."),
            ("Typing…", "Scrittura in corso..."),
            ("Relevance", "Rilevanza"),
            ("Title A→Z", "Titolo A→Z"),
            ("Title Z→A", "Titolo Z→A"),
            ("Season: Any", "Stagione: Qualsiasi"),
            ("Year: Any", "Anno: Qualsiasi"),
            ("Rating: Any", "Rating: Qualsiasi"),
            ("No results.", "Nessun risultato."),
            ("Filtering...", "Filtraggio..."),
            ("Filtering…", "Filtraggio..."),
            ("Start queue", "Avvia coda"),
            ("Cancel selected", "Annulla selezionato"),
            ("Clear completed", "Pulisci completati"),
            ("Refresh", "Aggiorna"),
            ("Add current anime", "Aggiungi anime corrente"),
            ("Remove selected", "Rimuovi selezionato"),
            ("Add current episode", "Aggiungi episodio corrente"),
            ("Exit FS", "Esci FS"),
            ("Exit Mini", "Esci mini"),
            ("Play", "Play"),
            ("Pause", "Pausa"),
            ("Sync now", "Sincronizza ora"),
            ("Backup", "Backup"),
        ]
        if self.app_language == "en":
            for en, it in pairs:
                if text == it:
                    return en
        else:
            for en, it in pairs:
                if text == en:
                    return it
        return text

    def _localize_runtime_text(self, text: str) -> str:
        s = self._translate_literal(text)

        def _rx(pat: str, it_fmt: str, en_fmt: str):
            nonlocal s
            m = re.fullmatch(pat, s)
            if not m:
                return
            fmt = en_fmt if self.app_language == "en" else it_fmt
            s = fmt.format(*m.groups())

        _rx(r"Episodi: (\d+)", "Episodi: {}", "Episodes: {}")
        _rx(r"Searching…", "Ricerca in corso…", "Searching...")
        _rx(r"Carico recent \+ recommended…", "Carico recenti + consigliati…", "Loading recent + recommended...")
        _rx(r"Trovati (\d+) risultati\.", "Trovati {} risultati.", "Found {} results.")
        _rx(r"Selezionato: (.+) — carico episodi…", "Selezionato: {} — carico episodi…", "Selected: {} — loading episodes...")
        _rx(r"Selezionato in cronologia: (.+)", "Selezionato in cronologia: {}", "Selected in history: {}")
        _rx(r"Carico episodi per resume: (.+)…", "Carico episodi per ripresa: {}…", "Loading episodes for resume: {}...")
        _rx(r"Apro dettagli anime: (.+)…", "Apro dettagli anime: {}…", "Opening anime details: {}...")
        _rx(r"Risolvo stream ep (.+)…", "Risolvo stream ep {}…", "Resolving stream ep {}...")
        _rx(r"Risolvo stream per download: (.+) ep (.+)", "Risolvo stream per download: {} ep {}", "Resolving stream for download: {} ep {}")
        _rx(r"Aggiunto download: (.+) ep (.+)", "Aggiunto download: {} ep {}", "Added download: {} ep {}")
        _rx(r"Aggiunti (\d+) episodi alla coda download\.", "Aggiunti {} episodi alla coda download.", "Added {} episodes to download queue.")
        _rx(r"Gia presente: (.+)", "Gia presente: {}", "Already present: {}")
        _rx(r"Integrity check fallita, retry: (.+) ep (.+)", "Integrity check fallita, retry: {} ep {}", "Integrity check failed, retry: {} ep {}")
        _rx(r"▶ (.+) — ep (.+)", "▶ {} — ep {}", "▶ {} — ep {}")
        _rx(r"▶ Offline: (.+)", "▶ Offline: {}", "▶ Offline: {}")
        _rx(r"Segnato come visto: (.+) ep (.+)", "Segnato come visto: {} ep {}", "Marked as seen: {} ep {}")
        _rx(r"Rimosso dalla cronologia\.", "Rimosso dalla cronologia.", "Removed from history.")
        _rx(r"Backup creato: (.+)", "Backup creato: {}", "Backup created: {}")
        _rx(r"Pull AniList -> Anigui in corso…", "Pull AniList -> Anigui in corso…", "AniList pull -> Anigui in progress...")
        _rx(r"Pull AniList fallito\.", "Pull AniList fallito.", "AniList pull failed.")
        _rx(r"Restore completato\.", "Ripristino completato.", "Restore completed.")
        _rx(r"Catalogo offline\.", "Catalogo offline.", "Offline catalog.")
        _rx(r"Ricerca anime\.", "Ricerca anime.", "Anime search.")
        _rx(r"Lista episodi\.", "Lista episodi.", "Episode list.")
        _rx(r"Aggiunto ai preferiti\.", "Aggiunto ai preferiti.", "Added to favorites.")
        _rx(r"Download fallito\.", "Download fallito.", "Download failed.")
        _rx(r"Download fallito \(integrity\)\.", "Download fallito (integrity).", "Download failed (integrity).")
        _rx(r"Download annullato\.", "Download annullato.", "Download cancelled.")
        _rx(r"Download completato: (.+)", "Download completato: {}", "Download completed: {}")
        _rx(r"Errore resolve download\.", "Errore risoluzione download.", "Download resolve error.")
        _rx(r"Scheduler attivo: fuori finestra oraria, coda in pausa\.", "Scheduler attivo: fuori finestra oraria, coda in pausa.", "Scheduler active: outside time window, queue paused.")
        _rx(r"Incognito attivo: sync esterno disabilitato\.", "Incognito attivo: sync esterno disabilitato.", "Incognito active: external sync disabled.")
        _rx(r"Incognito attivo: salvataggio settings bloccato\.", "Incognito attivo: salvataggio impostazioni bloccato.", "Incognito active: settings save blocked.")
        _rx(r"Incognito attivo: restore bloccato\.", "Incognito attivo: ripristino bloccato.", "Incognito active: restore blocked.")
        _rx(r"Incognito attivo: modifica cronologia bloccata\.", "Incognito attivo: modifica cronologia bloccata.", "Incognito active: history changes blocked.")
        _rx(r"Modalita Incognito attiva: cache/salvataggi disabilitati\.", "Modalita Incognito attiva: cache/salvataggi disabilitati.", "Incognito mode enabled: cache/saves disabled.")
        _rx(r"Modalita Incognito disattivata\.", "Modalita Incognito disattivata.", "Incognito mode disabled.")
        _rx(r"Sincronizza ora: nessun anime/episodio attivo da sincronizzare\.", "Sincronizza ora: nessun anime/episodio attivo da sincronizzare.", "Sync now: no active anime/episode to sync.")
        _rx(r"Sync now: nessun anime/episodio attivo da sincronizzare\.", "Sincronizza ora: nessun anime/episodio attivo da sincronizzare.", "Sync now: no active anime/episode to sync.")
        _rx(r"Sincronizza ora: episodio non completato, sync non inviato\.", "Sincronizza ora: episodio non completato, sync non inviato.", "Sync now: episode not completed, sync not sent.")
        _rx(r"Sync now: episodio non completato, sync non inviato\.", "Sincronizza ora: episodio non completato, sync non inviato.", "Sync now: episode not completed, sync not sent.")
        _rx(r"Sincronizzazione AniList: (.+) ep (.+)…", "Sincronizzazione AniList: {} ep {}…", "AniList sync: {} ep {}...")
        _rx(r"Sync AniList: (.+) ep (.+)…", "Sincronizzazione AniList: {} ep {}…", "AniList sync: {} ep {}...")
        _rx(r"Sincronizzazione AniList ok: (.+) ep (.+)", "Sincronizzazione AniList ok: {} ep {}", "AniList sync ok: {} ep {}")
        _rx(r"AniList sync ok: (.+) ep (.+)", "Sincronizzazione AniList ok: {} ep {}", "AniList sync ok: {} ep {}")
        _rx(r"Sincronizzazione AniList fallita\.", "Sincronizzazione AniList fallita.", "AniList sync failed.")
        _rx(r"AniList sync fallito\.", "Sincronizzazione AniList fallita.", "AniList sync failed.")
        _rx(r"Test AniList in corso…", "Test AniList in corso…", "Testing AniList...")
        _rx(r"AniList connesso: (.+)", "AniList connesso: {}", "AniList connected: {}")
        _rx(r"AniList token mancante\.", "Token AniList mancante.", "AniList token missing.")
        _rx(r"AniList non raggiungibile/token non valido\.", "AniList non raggiungibile/token non valido.", "AniList unavailable/invalid token.")
        _rx(r"Impostazioni salvate\.", "Impostazioni salvate.", "Settings saved.")
        _rx(r"Impostazioni ripristinate ai default \(premi Salva\)\.", "Impostazioni ripristinate ai default (premi Salva).", "Settings reset to defaults (press Save).")
        _rx(r"Nessun nuovo episodio aggiunto \(gia in coda/attivi\)\.", "Nessun nuovo episodio aggiunto (gia in coda/attivi).", "No new episodes added (already queued/active).")
        _rx(r"Inserisci prima il token AniList\.", "Inserisci prima il token AniList.", "Enter AniList token first.")
        _rx(r"AniList test fallito: (.+)", "AniList test fallito: {}", "AniList test failed: {}")
        _rx(r"AniList sync fallito: (.+)", "AniList sync fallito: {}", "AniList sync failed: {}")
        _rx(r"Pull AniList fallito: (.+)", "Pull AniList fallito: {}", "AniList pull failed: {}")
        _rx(r"Seleziona o avvia un episodio prima di aggiungerlo alla coda download\.", "Seleziona o avvia un episodio prima di aggiungerlo alla coda download.", "Select or start an episode before adding it to download queue.")
        _rx(r"Apri prima un anime\.", "Apri prima un anime.", "Open an anime first.")
        _rx(r"Seleziona uno o piu episodi dalla lista\.", "Seleziona uno o piu episodi dalla lista.", "Select one or more episodes from the list.")
        _rx(r"Su Linux al momento e supportato solo Nautilus\.\nInstalla Nautilus oppure apri manualmente il percorso:\n(.+)", "Su Linux al momento e supportato solo Nautilus.\nInstalla Nautilus oppure apri manualmente il percorso:\n{}", "On Linux only Nautilus is currently supported.\nInstall Nautilus or open this path manually:\n{}")
        _rx(r"Impossibile avviare Nautilus automaticamente\.\nPercorso: (.+)", "Impossibile avviare Nautilus automaticamente.\nPercorso: {}", "Could not launch Nautilus automatically.\nPath: {}")
        _rx(r"File locale non trovato\.", "File locale non trovato.", "Local file not found.")
        _rx(r"Download directory non valido\.", "Download directory non valido.", "Invalid download directory.")
        _rx(r"Impossibile creare la cartella download: (.+)", "Impossibile creare la cartella download: {}", "Unable to create download directory: {}")
        _rx(r"Formato scheduler non valido\. Usa HH:MM \(es\. 23:30\)\.", "Formato scheduler non valido. Usa HH:MM (es. 23:30).", "Invalid scheduler format. Use HH:MM (e.g. 23:30).")
        _rx(r"Inserisci un AniList token oppure disattiva AniList sync\.", "Inserisci un AniList token oppure disattiva AniList sync.", "Enter an AniList token or disable AniList sync.")
        _rx(r"Errore salvataggio settings: (.+)", "Errore salvataggio settings: {}", "Settings save error: {}")
        _rx(r"Backup fallito: (.+)", "Backup fallito: {}", "Backup failed: {}")
        _rx(r"Restore fallito: (.+)", "Restore fallito: {}", "Restore failed: {}")
        _rx(r"Pronto\.", "Pronto.", "Ready.")
        _rx(r"Errore\.", "Errore.", "Error.")
        return s

    def _translate_widget_tree(self):
        for lbl in self.findChildren(QLabel):
            txt = lbl.text()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                lbl.setText(new_txt)
        for btn in self.findChildren(QPushButton):
            txt = btn.text()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                btn.setText(new_txt)
        for box in self.findChildren(QGroupBox):
            txt = box.title()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                box.setTitle(new_txt)
        for edit in self.findChildren(QLineEdit):
            txt = edit.placeholderText()
            new_txt = self._translate_literal(txt)
            if new_txt != txt:
                edit.setPlaceholderText(new_txt)

    @staticmethod
    def _combo_set_items(
        combo: QComboBox,
        items: list[tuple[str, Any]],
        keep_data: Any = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        if keep_data is not None:
            idx = combo.findData(keep_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _apply_app_language_ui(self):
        self.setWindowTitle("Anigui")

        # Tabs
        self.tabs.setTabText(self.tabs.indexOf(self.tab_search), self._tr("Cerca", "Search"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_recommended), self._tr("Consigliati", "Recommended"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_featured), self._tr("In primo piano", "Featured"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_downloads), self._tr("Download", "Downloads"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_offline), "Offline")
        self.tabs.setTabText(self.tabs.indexOf(self.tab_favorites), self._tr("Preferiti", "Favorites"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_settings), self._tr("Impostazioni", "Settings"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_history), self._tr("Lista visione", "Watchlist"))

        # Top/search/player
        self.search_input.setPlaceholderText(self._tr("Cerca anime…", "Search anime..."))
        self.btn_search.setText(self._tr("🔎 Cerca", "🔎 Search"))
        self.btn_search.setToolTip(self._tr("Avvia ricerca anime", "Start anime search"))
        self.btn_incognito.setToolTip(self._tr("Disabilita salvataggi locali e cache", "Disable local saves and cache"))
        self.btn_incognito.setText(self._tr("Incognito ON", "Incognito ON") if self._incognito_enabled else self._tr("Incognito OFF", "Incognito OFF"))
        self.btn_clear_search_history.setText(self._tr("🧹 Pulisci cronologia ricerche", "🧹 Clear search history"))
        self.btn_back_catalog.setText(self._tr("← Torna alla ricerca", "← Back to Search"))
        self.btn_queue_selected_eps.setText(self._tr("⬇ Metti in coda selezionati", "⬇ Queue selected"))
        self.btn_fav_anime.setText(self._tr("❤ Preferito", "❤ Favorite"))
        self.btn_mark_planned_anime.setText(self._tr("📝 Da iniziare", "📝 Planned"))
        self.btn_mark_completed_anime.setText(self._tr("✅ Completato", "✅ Completed"))
        self.btn_back_episodes.setText(self._tr("← Torna agli episodi", "← Back to Episodes"))
        self.btn_prev.setText(self._tr("⏮ Precedente", "⏮ Prev"))
        self.btn_playpause.setText(self._tr("⏯ Play/Pause", "⏯ Play/Pause"))
        self.btn_next.setText(self._tr("⏭ Successivo", "⏭ Next"))
        self.btn_fs.setText(self._tr("⛶ Schermo intero", "⛶ Fullscreen"))
        self.btn_mini.setText(self._tr("▣ Mini Player", "▣ Mini"))
        self.lbl_player_title.setText(self._tr("Player", "Player"))

        # Recommended
        self.lbl_suggestions.setText(self._tr("Suggerimenti", "Suggestions"))
        self.lbl_recommended.setText(self._tr("Uscite recenti + Consigliati", "Recent Releases + Recommended"))
        self.lbl_recent_header.setText(self._tr("Uscite recenti", "Recent Releases"))
        self.lbl_recommended_header.setText(self._tr("Consigliati", "Recommended"))
        self.lbl_featured_header.setText(self._tr("In primo piano per te", "Featured for You"))
        self.lbl_featured_header.setToolTip(self._featured_algorithm_tooltip())
        self.btn_refresh_featured.setText(self._tr("↻ Aggiorna", "↻ Refresh"))
        self.btn_refresh_featured.setToolTip(self._featured_algorithm_tooltip())
        self.featured.setToolTip(self._featured_algorithm_tooltip())

        # Downloads / offline / favorites
        self.btn_dl_open.setText(self._tr("📂 Apri cartella", "📂 Open folder"))
        self.lbl_offline.setText(self._tr("Anime scaricati (stream locale)", "Downloaded anime (local stream)"))
        self.btn_offline_open.setText(self._tr("📂 Apri download", "📂 Open downloads"))
        self.btn_offline_back.setText(self._tr("← Torna al catalogo offline", "← Back to Offline Catalog"))
        self.btn_fav_add_current.setText(self._tr("❤ Aggiungi anime corrente", "❤ Add current anime"))
        self.btn_fav_remove.setText(self._tr("🗑 Rimuovi selezionato", "🗑 Remove selected"))
        self.btn_dl_add_current.setText(self._tr("＋ Aggiungi episodio corrente", "＋ Add current episode"))
        self.btn_dl_start.setText(self._tr("▶ Avvia coda", "▶ Start queue"))
        self.btn_dl_start.setToolTip(self._tr("Avvia i download in coda", "Start queued downloads"))
        self.btn_dl_cancel.setText(self._tr("✖ Annulla selezionato", "✖ Cancel selected"))
        self.btn_dl_clear.setText(self._tr("🧹 Pulisci completati", "🧹 Clear completed"))
        self.btn_refresh_recommended.setText(self._tr("↻ Aggiorna", "↻ Refresh"))
        self.btn_offline_refresh.setText(self._tr("↻ Aggiorna", "↻ Refresh"))
        self.lbl_anime_title.setText(self._tr("Anime", "Anime"))
        self.lbl_offline_anime_title.setText(self._tr("Anime", "Anime"))

        # Settings labels/buttons
        self.lbl_settings_download_dir.setText(self._tr("Cartella download", "Download directory"))
        self.lbl_settings_default_provider.setText(self._tr("Provider predefinito", "Default provider"))
        self.lbl_settings_default_lang.setText(self._tr("Lingua predefinita", "Default language"))
        self.lbl_settings_app_language.setText(self._tr("Lingua app", "App language"))
        self.lbl_settings_default_quality.setText(self._tr("Qualita predefinita", "Default quality"))
        self.lbl_settings_parallel_downloads.setText(self._tr("Download paralleli", "Parallel downloads"))
        self.lbl_settings_scheduler.setText("Scheduler")
        self.lbl_settings_scheduler_start.setText(self._tr("Inizio HH:MM", "Start HH:MM"))
        self.lbl_settings_scheduler_end.setText(self._tr("Fine HH:MM", "End HH:MM"))
        self.lbl_settings_integrity_min_mb.setText("Integrity min MB")
        self.lbl_settings_retry_count.setText(self._tr("Numero retry", "Retry count"))
        self.lbl_settings_anilist_sync.setText("AniList sync")
        self.lbl_settings_anilist_token.setText("AniList token")
        self.btn_settings_browse_download_dir.setText(self._tr("Sfoglia", "Browse"))
        self.btn_settings_save.setText(self._tr("Salva impostazioni", "Save settings"))
        self.btn_settings_reset.setText(self._tr("Reset predefiniti", "Reset defaults"))
        self.btn_settings_backup.setText(self._tr("Backup", "Backup"))
        self.btn_settings_restore.setText(self._tr("Ripristina", "Restore"))
        self.btn_settings_anilist_test.setText(self._tr("Test connessione AniList", "Test AniList connection"))
        self.btn_settings_anilist_sync_now.setText(self._tr("Sincronizza ora", "Sync now"))
        self.btn_settings_anilist_pull.setText(self._tr("Importa da AniList", "Import from AniList"))
        self.btn_settings_update_check.setText(self._tr("Controlla aggiornamenti", "Check updates"))
        if platform.system() == "Windows" and getattr(sys, "frozen", False):
            self.btn_settings_update_apply.setText(self._tr("Aggiorna ora", "Update now"))
        else:
            self.btn_settings_update_apply.setText(self._tr("Apri pagina update", "Open update page"))
        self.btn_settings_anilist_sync_now.setToolTip(
            self._tr("Invia subito il progresso corrente ad AniList", "Send current progress to AniList now")
        )
        self.btn_settings_anilist_pull.setToolTip(
            self._tr("Importa watchlist/progressi da AniList", "Import watchlist/progress from AniList")
        )

        # Search filters
        sort_idx = self.combo_sort.currentIndex()
        self.combo_sort.blockSignals(True)
        self.combo_sort.clear()
        self.combo_sort.addItems(
            [
                self._tr("Rilevanza", "Relevance"),
                self._tr("Titolo A→Z", "Title A→Z"),
                self._tr("Titolo Z→A", "Title Z→A"),
            ]
        )
        self.combo_sort.setCurrentIndex(max(0, sort_idx))
        self.combo_sort.blockSignals(False)

        season_idx = self.combo_season.currentIndex()
        self.combo_season.blockSignals(True)
        self.combo_season.clear()
        self.combo_season.addItems(
            [
                self._tr("Stagione: Qualsiasi", "Season: Any"),
                "Winter",
                "Spring",
                "Summer",
                "Fall",
            ]
        )
        self.combo_season.setCurrentIndex(max(0, season_idx))
        self.combo_season.blockSignals(False)

        if self.combo_year.count() > 0:
            self.combo_year.setItemText(0, self._tr("Anno: Qualsiasi", "Year: Any"))

        rating_idx = self.combo_rating.currentIndex()
        self.combo_rating.blockSignals(True)
        self.combo_rating.clear()
        self.combo_rating.addItems(
            [
                self._tr("Rating: Qualsiasi", "Rating: Any"),
                ">= 9",
                ">= 8",
                ">= 7",
                ">= 6",
            ]
        )
        self.combo_rating.setCurrentIndex(max(0, rating_idx))
        self.combo_rating.blockSignals(False)

        # Parallel download combos
        dl_parallel_data = self.combo_dl_parallel.currentData()
        self._combo_set_items(
            self.combo_dl_parallel,
            [
                (self._tr("Paralleli x1", "Parallel x1"), 1),
                (self._tr("Paralleli x2", "Parallel x2"), 2),
                (self._tr("Paralleli x3", "Parallel x3"), 3),
                (self._tr("Paralleli x4", "Parallel x4"), 4),
            ],
            keep_data=dl_parallel_data,
        )

        # Settings combos with translated labels
        self._combo_set_items(
            self.combo_settings_scheduler_enabled,
            [
                (self._tr("Disabilitato", "Disabled"), False),
                (self._tr("Abilitato", "Enabled"), True),
            ],
            keep_data=self.combo_settings_scheduler_enabled.currentData(),
        )
        self._combo_set_items(
            self.combo_settings_anilist_enabled,
            [
                (self._tr("Disabilitato", "Disabled"), False),
                (self._tr("Abilitato", "Enabled"), True),
            ],
            keep_data=self.combo_settings_anilist_enabled.currentData(),
        )

        # History/watchlist
        self.lbl_history_header.setText(
            self._tr(
                "Lista visione: Da iniziare · In corso · In pausa · Interrotti · Completati",
                "Watchlist: Planned · Watching · Paused · Dropped · Completed",
            )
        )
        self.lbl_history_filter.setText(self._tr("Filtro", "Filter"))
        self._combo_set_items(
            self.combo_history_filter,
            [
                (self._tr("Tutti", "All"), "All"),
                (self._tr("Completati", "Completed"), "Completed"),
                (self._tr("In corso", "Watching"), "Watching"),
                (self._tr("In pausa", "Paused"), "Paused"),
                (self._tr("Da iniziare", "Planned"), "Planned"),
                (self._tr("Interrotti", "Dropped"), "Dropped"),
            ],
            keep_data=self._history_filter,
        )
        self.btn_hist_resume.setText(self._tr("▶ Riprendi selezionato", "▶ Resume selected"))
        self.btn_hist_resume_next.setText(self._tr("⏭ Riprendi prossimo", "⏭ Resume next"))
        self.btn_hist_mark_seen.setText(self._tr("✅ Segna come visto", "✅ Mark as seen"))
        self.btn_hist_delete.setText(self._tr("🗑 Rimuovi", "🗑 Remove"))
        self.btn_clear_watch_history.setText(self._tr("🧹 Pulisci cronologia visione", "🧹 Clear watch history"))

        # Anonymous labels/buttons created inline.
        self._translate_widget_tree()
        # Refresh lists that contain language-dependent placeholder/status labels.
        self._refresh_downloads_ui()
        self._refresh_recent_queries_ui()

    def on_toggle_incognito(self, enabled: bool):
        self._incognito_enabled = bool(enabled)
        if self._incognito_enabled:
            self.btn_incognito.setText(self._tr("Incognito ON", "Incognito ON"))
            self.lbl_incognito_badge.setVisible(True)
            self.set_status("Modalita Incognito attiva: cache/salvataggi disabilitati.")
        else:
            self.btn_incognito.setText(self._tr("Incognito OFF", "Incognito OFF"))
            self.lbl_incognito_badge.setVisible(False)
            self.set_status("Modalita Incognito disattivata.")

    def _anilist_headers_for_token(self, token: str | None) -> dict[str, str]:
        return AniListService.headers_for_token(token)

    @staticmethod
    def _normalize_anilist_token(token: str | None) -> str:
        return AniListService.normalize_token(token)

    def _anilist_graphql(self, query: str, variables: dict[str, Any], token: str | None = None) -> dict[str, Any]:
        return AniListService.graphql(
            query,
            variables,
            token=token if token is not None else self._anilist_token,
        )

    @staticmethod
    def _watch_status_from_anilist(remote_status: str) -> str:
        return AniListService.watch_status_from_remote(remote_status)

    @staticmethod
    def _anilist_status_for_entry(entry: HistoryEntry) -> str:
        return AniListService.status_for_entry(entry)

    def _anilist_sync_key(self, entry: HistoryEntry) -> str:
        return AniListService.sync_key(entry)

    def _anilist_sync_progress_async(self, entry: HistoryEntry, force: bool = False):
        if self._incognito_enabled:
            return
        if not self._anilist_enabled:
            return
        if not (self._anilist_token or "").strip():
            return
        status = self._anilist_status_for_entry(entry)
        try:
            progress = int(max(0.0, float(entry.last_ep)))
        except Exception:
            progress = 0
        if status == "CURRENT" and progress <= 0:
            progress = 1
        key = self._anilist_sync_key(entry)
        prev = int(self._anilist_last_synced_progress.get(key, 0))
        now = time.time()
        if not force:
            if status == "CURRENT" and progress <= prev:
                return
            if now - float(self._anilist_last_sync_ts.get(key, 0.0)) < 20.0:
                return

        w = Worker(
            self._anilist_sync_worker,
            entry.name,
            progress,
            status,
            self._anilist_token,
        )

        def on_ok(_res: object, k=key, p=progress, st=status):
            if st == "CURRENT":
                self._anilist_last_synced_progress[k] = max(
                    p, int(self._anilist_last_synced_progress.get(k, 0))
                )
            else:
                self._anilist_last_synced_progress[k] = 0
            self._anilist_last_sync_ts[k] = time.time()

        def on_err(msg: str):
            debug_log(f"AniList sync failed: {msg}")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()

    def _anilist_sync_worker(
        self,
        anime_name: str,
        progress: int,
        status: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        return AniListService.save_media_list_entry(
            anime_name=anime_name,
            progress=progress,
            status=status,
            token=token,
            media_id_cache=self._anilist_media_id_cache,
        )

    def _anilist_active_token_from_ui(self) -> str:
        tok = ""
        if hasattr(self, "input_settings_anilist_token"):
            tok = self.input_settings_anilist_token.text().strip()
        if not tok:
            tok = (self._anilist_token or "").strip()
        return self._normalize_anilist_token(tok)

    @staticmethod
    def _norm_title(s: str) -> str:
        return AniListService.normalize_title(s)

    @staticmethod
    def _extract_season_number(title: str) -> int | None:
        return AniListService.extract_season_number(title)

    @staticmethod
    def _planned_identifier_from_titles(
        titles: list[str],
        media_id: int | None = None,
        entry_id: int | None = None,
    ) -> str:
        return AniListService.planned_identifier(titles, media_id=media_id, entry_id=entry_id)

    @staticmethod
    def _anilist_unmatched_identifier(
        titles: list[str],
        media_id: int | None = None,
        entry_id: int | None = None,
    ) -> str:
        return AniListService.unmatched_identifier(titles, media_id=media_id, entry_id=entry_id)

    def _title_search_candidates(self, titles: list[str]) -> list[str]:
        return AniListService.title_search_candidates(titles)

    def _search_items_for_provider(
        self,
        query: str,
        provider_name: str,
        lang: Any,
        strict_lang: bool = True,
    ) -> list[SearchItem]:
        if not query.strip():
            return []
        if self._is_aw_provider_name(provider_name):
            provider = self._aw_provider(provider_name)
            res = provider.search(query) or []
            out: list[SearchItem] = []
            for r in res:
                is_dub = bool(getattr(r, "dub", False))
                if strict_lang:
                    if lang == LanguageTypeEnum.DUB and not is_dub:
                        continue
                    if lang == LanguageTypeEnum.SUB and is_dub:
                        continue
                out.append(
                    SearchItem(
                        name=r.name,
                        identifier=str(getattr(r, "ref", "")),
                        languages={LanguageTypeEnum.DUB} if is_dub else {LanguageTypeEnum.SUB},
                        cover_url=None,
                        source=provider_name,
                        raw=r,
                    )
                )
            return out
        provider = get_provider(provider_name)
        res = provider.get_search(query)
        return [
            SearchItem(
                r.name,
                r.identifier,
                r.languages,
                cover_url=self._extract_normalized_cover_url(r, provider_name),
                source=provider_name,
                raw=r,
            )
            for r in res
        ]

    def _pick_best_item_for_titles(
        self,
        titles: list[str],
        items: list[SearchItem],
    ) -> SearchItem | None:
        return AniListService.pick_best_item_for_titles(titles, items)

    def _anilist_fetch_progress_entries(self, token: str) -> list[Any]:
        return AniListService.fetch_progress_entries(token)

    def _anilist_featured_recommendations_worker(
        self,
        token: str,
        provider_name: str,
        lang_name: str,
        refresh_salt: int = 0,
    ) -> list[SearchItem]:
        lang = LanguageTypeEnum.DUB if lang_name == "DUB" else LanguageTypeEnum.SUB
        q = """
        query {
          Viewer { id }
        }
        """
        payload_viewer = self._anilist_graphql(q, {}, token=token)
        viewer = payload_viewer.get("data", {}).get("Viewer") if isinstance(payload_viewer.get("data"), dict) else None
        if not isinstance(viewer, dict) or not viewer.get("id"):
            raise RuntimeError("Impossibile leggere Viewer AniList.")
        user_id = int(viewer["id"])

        q_lists = """
        query ($userId: Int) {
          MediaListCollection(
            userId: $userId,
            type: ANIME,
            status_in: [CURRENT, REPEATING, COMPLETED, PAUSED, DROPPED, PLANNING]
          ) {
            lists {
              status
              entries {
                progress
                score
                media {
                  id
                  genres
                  title { romaji english native }
                }
              }
            }
          }
        }
        """
        payload_lists = self._anilist_graphql(q_lists, {"userId": user_id}, token=token)
        coll = (
            payload_lists.get("data", {}).get("MediaListCollection")
            if isinstance(payload_lists.get("data"), dict)
            else None
        )
        lists = coll.get("lists", []) if isinstance(coll, dict) else []
        excluded_media_ids: set[int] = set()
        genre_weights: dict[str, float] = defaultdict(float)
        status_weight = {
            "COMPLETED": 1.8,
            "REPEATING": 1.6,
            "CURRENT": 1.4,
            "PAUSED": 0.8,
            "DROPPED": 0.3,
            "PLANNING": 0.0,
        }
        local_excluded_titles: set[str] = set()
        for e in self.history.read():
            n = self._norm_title(getattr(e, "name", ""))
            if n:
                # Exclude both already watched and already planned local entries.
                local_excluded_titles.add(n)

        for lst in lists:
            if not isinstance(lst, dict):
                continue
            remote_status = str(lst.get("status") or "").upper()
            base_w = float(status_weight.get(remote_status, 0.0))
            for entry in (lst.get("entries") or []):
                if not isinstance(entry, dict):
                    continue
                media = entry.get("media") if isinstance(entry.get("media"), dict) else {}
                media_id = int(media.get("id") or 0) if isinstance(media.get("id"), (int, float, str)) else 0
                if media_id > 0 and remote_status in ("CURRENT", "REPEATING", "COMPLETED", "PAUSED", "DROPPED", "PLANNING"):
                    excluded_media_ids.add(media_id)
                progress = int(entry.get("progress") or 0)
                progress_w = min(24.0, max(0.0, float(progress))) / 24.0
                score_val = float(entry.get("score") or 0.0)
                score_w = 0.0
                if score_val > 0:
                    if score_val > 10.0:
                        score_w = min(1.0, score_val / 100.0)
                    else:
                        score_w = min(1.0, score_val / 10.0)
                total_w = base_w + (progress_w * 0.8) + (score_w * 0.6)
                for g in (media.get("genres") or []):
                    g_s = str(g or "").strip()
                    if g_s and total_w > 0.0:
                        genre_weights[g_s] += total_w

        top_genres = sorted(
            [(g, w) for g, w in genre_weights.items() if w > 0.0],
            key=lambda x: x[1],
            reverse=True,
        )[:4]
        if not top_genres:
            return []
        top_genre_weights = {g: w for g, w in top_genres}

        q_media = """
        query ($genres: [String], $page: Int, $perPage: Int) {
          Page(page: $page, perPage: $perPage) {
            media(
              type: ANIME,
              genre_in: $genres,
              isAdult: false,
              sort: [POPULARITY_DESC, SCORE_DESC]
            ) {
              id
              popularity
              averageScore
              genres
              title { romaji english native }
              coverImage { large medium }
            }
          }
        }
        """
        candidates: dict[int, dict[str, Any]] = {}
        genres_param = [g for g, _w in top_genres]
        for page in (1, 2, 3):
            payload_page = self._anilist_graphql(
                q_media,
                {"genres": genres_param, "page": page, "perPage": 32},
                token=token,
            )
            page_node = payload_page.get("data", {}).get("Page") if isinstance(payload_page.get("data"), dict) else None
            media_list = page_node.get("media", []) if isinstance(page_node, dict) else []
            for media in media_list:
                if not isinstance(media, dict):
                    continue
                mid = int(media.get("id") or 0) if isinstance(media.get("id"), (int, float, str)) else 0
                if mid <= 0 or mid in excluded_media_ids:
                    continue
                title = media.get("title") if isinstance(media.get("title"), dict) else {}
                titles = [
                    str(title.get("english") or "").strip(),
                    str(title.get("romaji") or "").strip(),
                    str(title.get("native") or "").strip(),
                ]
                titles = [t for t in titles if t]
                if not titles:
                    continue
                norm_titles = [self._norm_title(t) for t in titles if self._norm_title(t)]
                if any(t in local_excluded_titles for t in norm_titles):
                    continue
                media_genres = [str(x).strip() for x in (media.get("genres") or []) if str(x).strip()]
                genre_match_score = sum(top_genre_weights.get(g, 0.0) for g in media_genres)
                if genre_match_score <= 0.0:
                    continue
                match_count = sum(1 for g in media_genres if g in top_genre_weights)
                popularity = float(media.get("popularity") or 0.0)
                avg_score = float(media.get("averageScore") or 0.0)
                rank_score = genre_match_score
                rank_score += min(1.0, popularity / 300000.0) * 0.9
                rank_score += min(1.0, avg_score / 100.0) * 1.1
                if match_count >= 2:
                    rank_score += 0.7
                cover = media.get("coverImage") if isinstance(media.get("coverImage"), dict) else {}
                cover_url = str(cover.get("large") or cover.get("medium") or "").strip() or None
                prev = candidates.get(mid)
                if prev is None or float(prev.get("rank_score", 0.0)) < rank_score:
                    candidates[mid] = {
                        "titles": titles,
                        "rank_score": rank_score,
                        "cover_url": cover_url,
                        "media_id": mid,
                    }
                # warm cache for fast AniList sync later
                for t in titles:
                    nt = self._norm_title(t)
                    if nt and nt not in self._anilist_media_id_cache:
                        self._anilist_media_id_cache[nt] = mid

        ranked = sorted(candidates.values(), key=lambda x: float(x.get("rank_score", 0.0)), reverse=True)[:220]
        if int(refresh_salt) != 0:
            rng = random.Random(int(refresh_salt))
            diversified: list[dict[str, Any]] = []
            for c in ranked:
                base = float(c.get("rank_score", 0.0))
                jitter = (rng.random() - 0.5) * 0.55
                x = dict(c)
                x["rank_score"] = base + jitter
                diversified.append(x)
            ranked = sorted(diversified, key=lambda x: float(x.get("rank_score", 0.0)), reverse=True)

        out: list[SearchItem] = []
        seen_ident: set[str] = set()
        target_count = 40
        provider_lookup_budget = 180
        cache_ttl = 30 * 60
        now_ts = time.time()

        def provider_search_cached(query: str, strict_lang: bool = True) -> list[SearchItem]:
            qn = self._norm_title(query)
            if not qn:
                return []
            key = f"{provider_name}|{lang_name}|{1 if strict_lang else 0}|{qn}"
            cached = self._featured_provider_search_cache.get(key)
            if cached is not None:
                ts, items = cached
                if now_ts - float(ts) <= cache_ttl:
                    return items
            try:
                items = self._search_items_for_provider(query, provider_name, lang, strict_lang=strict_lang)
            except Exception:
                items = []
            if len(self._featured_provider_search_cache) > 2500:
                self._featured_provider_search_cache.clear()
            self._featured_provider_search_cache[key] = (now_ts, items)
            return items

        for cand in ranked:
            if len(out) >= target_count or provider_lookup_budget <= 0:
                break
            titles = list(cand.get("titles") or [])
            if not titles:
                continue
            # prefer english/romaji and avoid duplicate provider hits
            ordered_titles = []
            seen_t: set[str] = set()
            for t in titles:
                nt = self._norm_title(t)
                if nt and nt not in seen_t:
                    seen_t.add(nt)
                    ordered_titles.append(t)
            found: SearchItem | None = None
            # try strict on first 2 titles max
            for t in ordered_titles[:2]:
                if provider_lookup_budget <= 0:
                    break
                provider_lookup_budget -= 1
                items = provider_search_cached(t, strict_lang=True)
                found = self._pick_best_item_for_titles(titles, items)
                if found is not None:
                    break
            if found is None:
                for t in ordered_titles[:1]:
                    if provider_lookup_budget <= 0:
                        break
                    provider_lookup_budget -= 1
                    items = provider_search_cached(t, strict_lang=False)
                    found = self._pick_best_item_for_titles(titles, items)
                    if found is not None:
                        break
            if found is None:
                continue
            n_found = self._norm_title(found.name)
            if n_found and n_found in local_excluded_titles:
                continue
            uniq = f"{found.source}:{found.identifier}"
            if uniq in seen_ident:
                continue
            seen_ident.add(uniq)
            if not found.cover_url:
                found.cover_url = cand.get("cover_url")
            out.append(found)
        return out

    def _anilist_pull_worker(
        self,
        token: str,
        provider_name: str,
        lang_name: str,
    ) -> dict[str, Any]:
        return {"rows": self._anilist_fetch_progress_entries(token)}

    @staticmethod
    def _save_history_entries_to_path(path: str, entries: list[HistoryEntry]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(x) for x in entries],
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, path)

    def _anilist_pull_and_merge_worker(
        self,
        token: str,
        provider_name: str,
        lang_name: str,
        local_history: list[HistoryEntry],
    ) -> dict[str, Any]:
        rows = list(AniListService.fetch_progress_entries(token) or [])
        lang = LanguageTypeEnum.DUB if lang_name == "DUB" else LanguageTypeEnum.SUB
        merged = AniListService.merge_history_entries(
            rows,
            local_history,
            provider_name=provider_name,
            lang_name=lang_name,
            search_provider=lambda query, strict_lang=True: self._search_items_for_provider(
                query,
                provider_name,
                lang,
                strict_lang=strict_lang,
            ),
            search_lang=lang,
        )
        self._save_history_entries_to_path(HISTORY_PATH, merged.get("merged_history") or [])
        return merged

    def _dedupe_history_entries_prefer_cover(
        self,
        entries: list[HistoryEntry],
    ) -> list[HistoryEntry]:
        return AniListService.dedupe_history(entries)

    def _run_history_maintenance(self) -> None:
        current = self.history.read()
        deduped = self._dedupe_history_entries_prefer_cover(current)
        if len(deduped) != len(current):
            try:
                self.history.replace(deduped)
            except Exception:
                pass

    def _anilist_test_connection_worker(self, token: str) -> str:
        return AniListService.viewer_name(token)

    def _anilist_viewer_worker(self, token: str) -> tuple[int, str]:
        return AniListService.viewer(token)

    def _debug_log_anilist_viewer_startup(self):
        token = self._anilist_active_token_from_ui()
        if not token:
            debug_log("AniList debug startup: token missing, skip Viewer lookup.")
            return
        w = Worker(self._anilist_viewer_worker, token)

        def on_ok(data: object):
            try:
                uid, uname = data
                _ = int(uid)
                debug_log(f"AniList debug startup: Viewer.name={str(uname)}")
            except Exception:
                debug_log(f"AniList debug startup: unexpected viewer payload={data!r}")

        def on_err(msg: str):
            debug_log(f"AniList debug startup: Viewer lookup failed: {msg}")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()

    def on_anilist_test_connection(self):
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            self.set_status("AniList token mancante.")
            return
        self.set_status("Test AniList in corso…")
        self._begin_request()
        w = Worker(self._anilist_test_connection_worker, token)

        def on_ok(name: str):
            self._end_request()
            self.set_status(f"AniList connesso: {name}")

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"AniList test fallito: {msg}")
            self.set_status("AniList non raggiungibile/token non valido.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()

    def on_anilist_sync_now(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: sync esterno disabilitato.")
            return
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            return

        anime_name: str | None = None
        progress = 0
        completed = False
        status = "CURRENT"
        source = "none"

        if self.selected_anime is not None and self.current_ep is not None:
            anime_name = str(getattr(self.selected_anime, "name", "")).strip() or None
            progress = int(max(1, float(self.current_ep)))
            try:
                pos = float(self.player.get_time_pos() or 0.0)
                dur = float(self.player.get_duration() or 0.0)
                percent = float(self.player.get_percent_pos() or 0.0)
                completed = AniListService.episode_is_completed(pos, dur, percent)
                source = "active-player"
            except Exception:
                completed = False
                source = "active-player-exc"
            if not completed:
                hist_active = self._current_history_entry()
                if hist_active is not None:
                    hp, hc = self._history_entry_best_progress_and_completed_for_sync(hist_active)
                    if hp > 0 and hc:
                        progress = hp
                        completed = True
                        source = "active-history-fallback"
        else:
            e = self._history_entry_for_sync_now()
            if e is not None:
                anime_name = e.name
                progress, completed = self._history_entry_best_progress_and_completed_for_sync(e)
                status = self._anilist_status_for_entry(e)
                source = "history"

        debug_log(
            "sync-now candidate: "
            f"source={source} anime={anime_name!r} progress={progress} completed={completed}"
        )

        fail_reason = ""
        if not anime_name:
            fail_reason = "no-anime"
        elif status in {"CURRENT", "COMPLETED"} and int(progress) <= 0:
            fail_reason = "no-progress"
        elif status == "CURRENT" and not bool(completed):
            fail_reason = "not-completed"

        if fail_reason:
            debug_log(
                "sync-now blocked: "
                f"reason={fail_reason} source={source} anime={anime_name!r} "
                f"progress={progress} completed={completed}"
            )
            if fail_reason in ("no-anime", "no-progress"):
                self.set_status("Sincronizza ora: nessun anime/episodio attivo da sincronizzare.")
            else:
                self.set_status("Sincronizza ora: episodio non completato, sync non inviato.")
            return

        if status == "PLANNING":
            self.set_status(f"Sincronizzazione AniList: {anime_name} -> Planned…")
        elif status == "PAUSED":
            self.set_status(f"Sincronizzazione AniList: {anime_name} -> Paused…")
        elif status == "DROPPED":
            self.set_status(f"Sincronizzazione AniList: {anime_name} -> Dropped…")
        elif status == "COMPLETED":
            self.set_status(f"Sincronizzazione AniList: {anime_name} -> Completed ({progress})…")
        else:
            self.set_status(f"Sincronizzazione AniList: {anime_name} ep {progress}…")
        self._begin_request()
        w = Worker(self._anilist_sync_worker, anime_name, progress, status, token)

        def on_ok(_data: object, name=anime_name, ep=progress, st=status):
            self._end_request()
            if st == "PLANNING":
                self.set_status(f"Sincronizzazione AniList ok: {name} (Planned)")
            elif st == "PAUSED":
                self.set_status(f"Sincronizzazione AniList ok: {name} (Paused)")
            elif st == "DROPPED":
                self.set_status(f"Sincronizzazione AniList ok: {name} (Dropped)")
            elif st == "COMPLETED":
                self.set_status(f"Sincronizzazione AniList ok: {name} (Completed {ep})")
            else:
                self.set_status(f"Sincronizzazione AniList ok: {name} ep {ep}")

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"AniList sync fallito: {msg}")
            self.set_status("Sincronizzazione AniList fallita.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()

    def _history_entry_for_sync_now(self) -> HistoryEntry | None:
        if getattr(self, "_selected_history", None):
            return self._selected_history
        try:
            cur = self._history_entry_from_current_item()
            if cur is not None:
                self._selected_history = cur
                return self._selected_history
        except Exception:
            pass
        items = self.history.list()
        if items:
            self._selected_history = items[0]
            return self._selected_history
        return None

    def _history_entry_best_progress_for_sync(self, entry: HistoryEntry) -> int:
        best, _completed = self._history_entry_best_progress_and_completed_for_sync(entry)
        return best

    def _history_entry_best_progress_and_completed_for_sync(self, entry: HistoryEntry) -> tuple[int, bool]:
        return AniListService.best_progress_and_completed(entry)

    def on_anilist_pull_from_remote(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: sync esterno disabilitato.")
            return
        token = self._anilist_active_token_from_ui()
        if not token:
            self.notify_err("Inserisci prima il token AniList.")
            return
        provider_name = str(self.provider_name or "allanime")
        lang_name = "DUB" if self.lang == LanguageTypeEnum.DUB else "SUB"
        self.set_status("Pull AniList -> Anigui in corso…")
        self._begin_request()
        local_snapshot = self.history.read()
        w = Worker(
            self._anilist_pull_and_merge_worker,
            token,
            provider_name,
            lang_name,
            local_snapshot,
        )

        def on_ok(res: dict[str, Any]):
            self._end_request()
            self.history.replace(res.get("merged_history") or [])
            imported = int(res.get("imported", 0))
            skipped_local = int(res.get("skipped_local", 0))

            self.refresh_history_ui()
            planned_missing = list(res.get("planned_missing_keys") or [])
            planned_all = int(res.get("remote_planned", 0))
            planned_imp = int(res.get("imported_planned", 0))
            debug_log(
                "AniList pull planned diag: "
                f"imported={planned_imp}/{planned_all} "
                f"missing={planned_missing}"
            )
            diag = ""
            if planned_missing:
                preview = ", ".join(str(x) for x in planned_missing[:8])
                if len(planned_missing) > 8:
                    preview += ", ..."
                diag = self._tr(
                    f" chiavi pianificate mancanti: {preview}",
                    f" missing planned keys: {preview}",
                )
            self.set_status(
                self._tr(
                    "Pull AniList completato: "
                    f"{imported} importati, "
                    f"{int(res.get('skipped', 0))} non matchati, "
                    f"{skipped_local} ignorati (progress locale migliore), "
                    f"pianificati {int(res.get('imported_planned', 0))}/{int(res.get('remote_planned', 0))}."
                    f"{diag}",
                    "AniList pull completed: "
                    f"{imported} imported, "
                    f"{int(res.get('skipped', 0))} unmatched, "
                    f"{skipped_local} skipped (better local progress), "
                    f"planned {int(res.get('imported_planned', 0))}/{int(res.get('remote_planned', 0))}."
                    f"{diag}",
                )
            )

        def on_err(msg: str):
            self._end_request()
            self.notify_err(f"Pull AniList fallito: {msg}")
            self.set_status("Pull AniList fallito.")

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._track_worker(w)
        w.start()


    @staticmethod
    def _is_local_media_file(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    @staticmethod
    def _natural_sort_key(text: str):
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]

    def refresh_offline_library(self):
        self._offline_nonce += 1
        nonce = self._offline_nonce
        self.offline_results.clear()
        self.offline_episodes.clear()
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)
        self._offline_items = []
        self._offline_episode_files = []
        self._offline_current_anime_dir = None
        self._offline_current_episode_index = None

        try:
            os.makedirs(self.download_dir, exist_ok=True)
        except Exception:
            pass

        dirs: list[str] = []
        try:
            for name in os.listdir(self.download_dir):
                full = os.path.join(self.download_dir, name)
                if not os.path.isdir(full):
                    continue
                try:
                    has_media = any(
                        self._is_local_media_file(os.path.join(full, f))
                        for f in os.listdir(full)
                    )
                except Exception:
                    has_media = False
                if has_media:
                    dirs.append(full)
        except Exception:
            dirs = []

        dirs.sort(key=lambda p: self._natural_sort_key(os.path.basename(p)))
        self._offline_items = [
            OfflineAnimeItem(
                name=os.path.basename(d),
                folder=d,
                cover_url=self._offline_cover_url_for_name(os.path.basename(d)),
            )
            for d in dirs
        ]

        if not self._offline_items:
            self.offline_results.addItem(self._tr("(nessun anime scaricato)", "(no downloaded anime)"))
            self.offline_results.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return

        placeholder = self._make_cover_placeholder()
        for row, it in enumerate(self._offline_items):
            item = QListWidgetItem(placeholder, it.name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            item.setSizeHint(QSize(170, 290))
            self.offline_results.addItem(item)
            if it.cover_url:
                w = Worker(self._offline_cover_cached_only, it.cover_url)
                w.ok.connect(lambda data, r=row, n=nonce: self._on_offline_cover_loaded(r, n, data[0], data[1]))
                w.err.connect(lambda _msg: None)
                self._track_worker(w)
                w.start()

    def on_pick_offline_anime(self):
        row = self.offline_results.currentRow()
        if row < 0 or row >= len(self._offline_items):
            return
        picked = self._offline_items[row]
        anime_dir = picked.folder
        if not anime_dir or not os.path.isdir(anime_dir):
            return

        self.offline_episodes.clear()
        files: list[str] = []
        try:
            for name in os.listdir(anime_dir):
                full = os.path.join(anime_dir, name)
                if os.path.isfile(full) and self._is_local_media_file(full):
                    files.append(full)
        except Exception:
            files = []

        files.sort(key=lambda p: self._natural_sort_key(os.path.basename(p)))
        self._offline_current_anime_dir = anime_dir
        self._offline_episode_files = files
        self._offline_current_episode_index = None
        self.lbl_offline_anime_title.setText(picked.name)
        self.offline_stack.setCurrentWidget(self.page_offline_anime)

        if not files:
            self.offline_episodes.addItem(self._tr("(nessun episodio/file)", "(no episodes/files)"))
            self.offline_episodes.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return

        for fp in files:
            it = QListWidgetItem(os.path.basename(fp))
            it.setData(Qt.ItemDataRole.UserRole, fp)
            self.offline_episodes.addItem(it)

    def on_play_offline_episode(self, *_args):
        item = self.offline_episodes.currentItem()
        if item is None:
            return
        fp = item.data(Qt.ItemDataRole.UserRole)
        if not fp or not os.path.isfile(fp):
            return
        try:
            idx = self._offline_episode_files.index(fp)
        except Exception:
            idx = None
        self._offline_current_episode_index = idx
        self._play_local_file(fp)

    def on_offline_back_to_catalog(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.offline_stack.setCurrentWidget(self.page_offline_catalog)
        self.set_status("Catalogo offline.")

    def _offline_cover_url_for_name(self, name: str) -> str | None:
        target = self._safe_name(name or "").lower()
        if not target:
            return None
        mapped = self._offline_covers_map.get(target)
        if mapped:
            return mapped
        for e in self.history.list():
            hist_name = self._safe_name(e.name or "").lower()
            if hist_name == target and e.cover_url:
                self._remember_offline_cover_url(name, e.cover_url)
                return e.cover_url
        return None

    def _offline_cover_cached_only(self, cover_ref: str) -> tuple[bytes | None, bool]:
        # cover_ref can be:
        # 1) local cache path (*.img) [new format]
        # 2) original URL [legacy format]
        ref = (cover_ref or "").strip()
        if not ref:
            return None, False
        if ref.lower().endswith(".img") and os.path.exists(ref):
            cache_path = ref
        else:
            cache_path = self._cover_cache_path(ref)
        if not cache_path or not os.path.exists(cache_path):
            return None, False
        try:
            with open(cache_path, "rb") as f:
                data = f.read()
            return (data if data else None), bool(data)
        except Exception:
            return None, False

    def _on_offline_cover_loaded(self, row: int, nonce: int, data: bytes | None, from_cache: bool):
        if nonce != self._offline_nonce:
            return
        if data is None:
            return
        if row < 0 or row >= self.offline_results.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.offline_results.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.offline_results.item(row).setIcon(QIcon(scaled))

    def _play_local_file(self, path: str):
        if not os.path.isfile(path):
            self.notify_err("File locale non trovato.")
            return

        self._save_current_progress(force=True)
        self._pending_resume_seek = None
        self._pending_resume_seek_attempts = 0
        self._resume_marker_pos = None
        self._local_media_active = True
        self.current_ep = None
        self.selected_anime = None
        self.selected_search_item = None

        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_player)
        self._current_search_view = "player"
        self._refresh_search_layout()
        base = os.path.basename(path)
        self.lbl_player_title.setText(f"Offline — {base}")
        self.set_status(f"▶ Offline: {base}")
        self._set_transport_enabled(False)
        try:
            self.player.load(path, referrer=None, sub_file=None)
        finally:
            self._set_transport_enabled(True)

    def on_back_to_catalog(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.search_stack.setCurrentWidget(self.page_catalog)
        self._current_search_view = "catalog"
        self._refresh_search_layout()
        self.set_status("Ricerca anime.")

    def on_back_to_episodes(self):
        self._save_current_progress(force=True)
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        try:
            self.player.stop()
        except Exception:
            pass
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = None
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.set_status("Lista episodi.")

    @staticmethod
    def _is_episode_completed(pos: float, dur: float) -> bool:
        return AniListService.episode_is_completed(pos, dur)

    @staticmethod
    def _is_episode_completed_by_percent(percent: float | None) -> bool:
        return AniListService.episode_is_completed(None, None, percent)

    def _save_current_progress(self, force: bool = False):
        if self._incognito_enabled:
            return
        if self._local_media_active:
            return
        if not self.selected_anime or self.current_ep is None:
            return
        try:
            pos_raw = self.player.get_time_pos()
        except Exception:
            pos_raw = None
        try:
            dur_raw = self.player.get_duration()
        except Exception:
            dur_raw = None
        try:
            percent_raw = self.player.get_percent_pos()
        except Exception:
            percent_raw = None

        now = time.time()

        identifier = (
            getattr(self.selected_anime, "identifier", None)
            or getattr(self.selected_anime, "_identifier", None)
            or getattr(self.selected_anime, "ref", "")
        )
        ident_s = str(identifier)
        lang_s = "SUB" if self.lang == LanguageTypeEnum.SUB else "DUB"

        prev_entry = find_history_entry(
            self.history.read(),
            provider=self.provider_name,
            identifier=ident_s,
            lang=lang_s,
            name=self.selected_anime.name,
        )

        if not force:
            pos_guess = (
                max(0.0, float(pos_raw))
                if pos_raw is not None
                else max(0.0, float(getattr(prev_entry, "last_pos", 0.0) or 0.0))
                if prev_entry is not None
                else 0.0
            )
            pct_guess = (
                max(0.0, min(100.0, float(percent_raw)))
                if percent_raw is not None
                else max(0.0, min(100.0, float(getattr(prev_entry, "last_percent", 0.0) or 0.0)))
                if prev_entry is not None and float(getattr(prev_entry, "last_percent", 0.0) or 0.0) > 0.0
                else -1.0
            )
            pos_delta_small = abs(pos_guess - self._last_progress_pos) < 5.0
            pct_delta_small = abs(pct_guess - self._last_progress_percent) < 1.0
            if now - self._last_progress_save_at < 5.0 and pos_delta_small and pct_delta_small:
                return

        cover_url = (
            getattr(self.selected_search_item, "cover_url", None)
            if self.selected_search_item
            else (prev_entry.cover_url if prev_entry is not None else None)
        )
        result = build_saved_progress_entry(
            provider_name=(prev_entry.provider if prev_entry is not None else self.provider_name),
            identifier=(prev_entry.identifier if prev_entry is not None else ident_s),
            name=(prev_entry.name if prev_entry is not None and str(prev_entry.name or "").strip() else self.selected_anime.name),
            lang_name=lang_s,
            current_ep=self.current_ep,
            pos_raw=pos_raw,
            dur_raw=dur_raw,
            percent_raw=percent_raw,
            prev_entry=prev_entry,
            cover_url=cover_url,
            now_ts=now,
        )
        if result is None:
            return

        entry = result.entry
        self.history.upsert(entry)
        self._anilist_sync_progress_async(entry, force=force)
        self._last_progress_save_at = now
        self._last_progress_pos = result.pos
        self._last_progress_percent = result.percent
        if force:
            self.refresh_history_ui()

    def _try_apply_resume_seek_ratio(self):
        ratio = self._pending_resume_seek_ratio
        if ratio is None:
            return
        try:
            self.player.seek_ratio(float(ratio))
        except Exception:
            pass
        self._pending_resume_seek_ratio = None

    def _try_apply_resume_seek(self):
        target = self._pending_resume_seek
        if target is None:
            return
        self._pending_resume_seek_attempts += 1
        try:
            self.player.seek(target)
        except Exception:
            pass

        cur = self.player.get_time_pos()
        if cur is not None and abs(float(cur) - target) <= 8.0:
            self._pending_resume_seek = None
            return

        if self._pending_resume_seek_attempts < 8:
            QTimer.singleShot(350, self._try_apply_resume_seek)
        else:
            self._pending_resume_seek = None
            if self._pending_resume_seek_ratio is not None:
                QTimer.singleShot(0, self._try_apply_resume_seek_ratio)

    @staticmethod
    def fmt_time(sec: float | None) -> str:
        if sec is None or sec != sec or sec < 0:
            return "00:00"
        s = int(sec)
        m = s // 60
        s = s % 60
        h = m // 60
        m = m % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _fmt_relative_time(self, ts: float | None) -> str:
        if not ts:
            return self._tr("proprio adesso", "just now")
        delta = max(0, int(time.time() - float(ts)))
        if delta < 60:
            return self._tr("proprio adesso", "just now")
        if delta < 3600:
            n = delta // 60
            return self._tr(
                f"{n} minuto{'i' if n != 1 else ''} fa",
                f"{n} minute{'s' if n != 1 else ''} ago",
            )
        if delta < 86400:
            n = delta // 3600
            return self._tr(
                f"{n} ora{'e' if n != 1 else ''} fa",
                f"{n} hour{'s' if n != 1 else ''} ago",
            )
        if delta < 86400 * 30:
            n = delta // 86400
            return self._tr(
                f"{n} giorno{'i' if n != 1 else ''} fa",
                f"{n} day{'s' if n != 1 else ''} ago",
            )
        if delta < 86400 * 365:
            n = delta // (86400 * 30)
            return self._tr(
                f"{n} mese{'i' if n != 1 else ''} fa",
                f"{n} month{'s' if n != 1 else ''} ago",
            )
        n = delta // (86400 * 365)
        return self._tr(
            f"{n} anno{'i' if n != 1 else ''} fa",
            f"{n} year{'s' if n != 1 else ''} ago",
        )

    @staticmethod
    def _extract_cover_url(obj: Any) -> str | None:
        for key in ("cover", "cover_url", "image", "img", "poster", "thumbnail"):
            v = getattr(obj, key, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _episode_label(self, ep: float | int) -> str:
        return self._tr(f"Episodio {ep}", f"Episode {ep}")

    @staticmethod
    def _to_ep_float(ep: float | int | str | None) -> float | None:
        try:
            return float(ep) if ep is not None else None
        except Exception:
            return None

    def _entry_watched_eps_set(self, entry: HistoryEntry) -> set[float]:
        return AniListService.watched_eps_set(entry)

    def _entry_add_watched_ep(self, entry: HistoryEntry, ep: float | int | str | None) -> None:
        AniListService.add_watched_ep(entry, ep)

    @staticmethod
    def _ep_key(ep: float | int | str | None) -> str:
        try:
            f = float(ep)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except Exception:
            return str(ep or "")

    def _entry_episode_progress_map(self, entry: HistoryEntry) -> dict[float, dict[str, Any]]:
        return AniListService.episode_progress_map(entry)

    def _entry_get_episode_progress(
        self,
        entry: HistoryEntry,
        ep: float | int | str | None,
    ) -> dict[str, Any] | None:
        return AniListService.episode_progress(entry, ep)

    def _entry_set_episode_progress(
        self,
        entry: HistoryEntry,
        ep: float | int | str | None,
        pos: float,
        dur: float,
        percent: float,
        completed: bool,
    ) -> None:
        AniListService.set_episode_progress(
            entry,
            ep,
            pos=pos,
            dur=dur,
            percent=percent,
            completed=completed,
        )

    def _is_last_available_episode(
        self,
        ep: float | int | str | None,
        available_eps: list[float | int] | None,
    ) -> bool:
        if not available_eps:
            return False
        epf = self._to_ep_float(ep)
        if epf is None:
            return False
        vals: list[float] = []
        for x in available_eps:
            xf = self._to_ep_float(x)
            if xf is not None:
                vals.append(xf)
        if not vals:
            return False
        max_ep = max(vals)
        return abs(epf - max_ep) < 1e-6

    def _entry_is_series_completed(
        self,
        entry: HistoryEntry,
        available_eps: list[float | int] | None,
    ) -> bool:
        return AniListService.series_completed(entry, available_eps)

    def _is_aw_provider_name(self, name: str) -> bool:
        return name in ("aw_animeworld", "aw_animeunity")

    def _ensure_aw_runtime(self):
        if self._aw_runtime_ready:
            return
        if platform.system() == "Windows":
            shim_dir = os.path.join(app_state_dir(), "aw_shims")
            os.makedirs(shim_dir, exist_ok=True)
            uname_bat = os.path.join(shim_dir, "uname.bat")
            if not os.path.exists(uname_bat):
                with open(uname_bat, "w", encoding="utf-8") as f:
                    f.write("@echo off\n")
                    f.write("echo Windows_NT anigui\n")
            cur_path = os.environ.get("PATH", "")
            parts = cur_path.split(os.pathsep) if cur_path else []
            if shim_dir not in parts:
                os.environ["PATH"] = shim_dir + os.pathsep + cur_path
        self._aw_runtime_ready = True

    def _aw_provider_class(self, name: str):
        self._ensure_aw_runtime()
        try:
            mod = importlib.import_module("aw_cli.providers")
        except Exception as ex:
            raise RuntimeError(
                "Provider ITA non disponibile: controlla aw-cli su questo sistema "
                f"(errore import: {ex})."
            ) from ex

        if name == "aw_animeworld":
            return getattr(mod, "Animeworld")
        if name == "aw_animeunity":
            return getattr(mod, "Animeunity")
        raise RuntimeError(f"Provider ITA sconosciuto: {name}")

    def _aw_provider(self, name: str | None = None):
        key = name or self.provider_name
        if not self._is_aw_provider_name(key):
            raise RuntimeError("Provider non-ITA richiesto come aw-cli.")
        self._ensure_aw_config()
        if key not in self._aw_provider_instances:
            cls = self._aw_provider_class(key)
            self._aw_provider_instances[key] = cls()
        return self._aw_provider_instances[key]

    def _aw_anime_class(self):
        if self._aw_anime_cls is not None:
            return self._aw_anime_cls
        self._ensure_aw_runtime()
        try:
            mod = importlib.import_module("aw_cli.anime")
        except Exception as ex:
            raise RuntimeError(
                "Provider ITA non disponibile: controlla aw-cli su questo sistema "
                f"(errore import: {ex})."
            ) from ex
        self._aw_anime_cls = getattr(mod, "Anime")
        return self._aw_anime_cls

    def _aw_cover_cache_key(self, item: SearchItem) -> str:
        return f"{item.source}:{item.identifier}"

    def _aw_resolve_cover_url(self, item: SearchItem) -> str | None:
        key = self._aw_cover_cache_key(item)
        miss_until = self._aw_cover_miss_until.get(key)
        if miss_until is not None and time.time() < miss_until:
            return None
        if key in self._aw_cover_url_cache:
            return self._aw_cover_url_cache[key]

        url: str | None = None
        try:
            provider = self._aw_provider(item.source)
            if item.source == "aw_animeunity":
                ident = str(item.identifier)
                if ident.isdigit():
                    r = provider.Client.get(f"{provider.BASE_URL}/info_api/{ident}/")
                    r.raise_for_status()
                    data = r.json()
                    for k in ("imageurl", "image_url", "image", "cover", "poster"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            url = v.strip()
                            break
            elif item.source == "aw_animeworld":
                ref = str(item.identifier)
                if ref and not ref.startswith("http"):
                    ref = provider.BASE_URL.rstrip("/") + "/" + ref.lstrip("/")
                if ref:
                    if hasattr(provider, "_get_html"):
                        html = provider._get_html(ref)
                    else:
                        r = provider.Client.get(ref)
                        r.raise_for_status()
                        html = r.text
                    m = re.search(
                        r"<meta[^>]+property=['\"]og:image['\"][^>]+content=['\"]([^'\"]+)['\"]",
                        html,
                        re.IGNORECASE,
                    )
                    if not m:
                        m = re.search(
                            r"<meta[^>]+content=['\"]([^'\"]+)['\"][^>]+property=['\"]og:image['\"]",
                            html,
                            re.IGNORECASE,
                        )
                    if not m:
                        m = re.search(
                            r'<img[^>]+class="[^"]*poster[^"]*"[^>]+(?:src|data-src)="([^"]+)"',
                            html,
                            re.IGNORECASE,
                        )
                    if m:
                        url = m.group(1).strip()
                        if url.startswith("/"):
                            url = provider.BASE_URL.rstrip("/") + url
                    if not url:
                        # fallback permissivo: prima immagine assoluta nel markup
                        m = re.search(
                            r"(https?:\\/\\/[^'\"\\s>]+\\.(?:jpg|jpeg|png|webp))",
                            html,
                            re.IGNORECASE,
                        )
                        if m:
                            url = m.group(1).replace("\\/", "/")
                    if not url:
                        # fallback: cerca in pagina search per nome anime
                        q = quote_plus(item.name)
                        s_html = provider._get_html(f"{provider.BASE_URL}/search?keyword={q}")
                        href = str(item.identifier)
                        href_rel = href.replace(provider.BASE_URL, "")
                        block_pat = re.compile(
                            r'<div class="inner">(?P<block>.*?)</div>',
                            re.IGNORECASE | re.DOTALL,
                        )
                        for bm in block_pat.finditer(s_html):
                            block = bm.group("block")
                            if href not in block and href_rel not in block:
                                continue
                            im = re.search(
                                r'(?:src|data-src)=["\']([^"\']+\.(?:jpg|jpeg|png|webp)[^"\']*)["\']',
                                block,
                                re.IGNORECASE,
                            )
                            if im:
                                url = im.group(1).strip()
                                if url.startswith("//"):
                                    url = "https:" + url
                                elif url.startswith("/"):
                                    url = provider.BASE_URL.rstrip("/") + url
                                break
        except Exception:
            url = None

        if url:
            self._aw_cover_url_cache[key] = url
            self._aw_cover_miss_until.pop(key, None)
        else:
            self._aw_cover_miss_until[key] = time.time() + 600.0
            debug_log(f"cover resolve miss: source={item.source} id={item.identifier}")
        return url

    def _ensure_aw_config(self):
        if self._aw_config_ready:
            return
        self._ensure_aw_runtime()
        try:
            ut_mod = importlib.import_module("aw_cli.utilities")
        except Exception:
            return
        cfg = getattr(ut_mod, "configData", None)
        if cfg is None:
            self._aw_config_ready = True
            return
        general = cfg.setdefault("general", {})
        general.setdefault("specials", False)
        self._aw_config_ready = True

    def _make_cover_placeholder(self, title: str | None = None) -> QIcon:
        pix = QPixmap(300, 440)
        pix.fill(QColor("#2b2b2b"))
        painter = QPainter(pix)
        painter.fillRect(0, 0, 300, 100, QColor(229, 9, 20, 90))
        if title:
            painter.setPen(QColor("#e9edf1"))
            painter.drawText(
                16,
                280,
                268,
                130,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                title,
            )
        painter.end()
        return QIcon(pix)

    def _make_skeleton_pixmap(self, size: QSize, phase: int) -> QPixmap:
        w = max(1, size.width())
        h = max(1, size.height())
        pix = QPixmap(w, h)
        pix.fill(QColor("#1e1e1e"))
        painter = QPainter(pix)
        band_w = max(30, w // 4)
        x = (phase % (w + band_w)) - band_w
        grad = QColor(255, 255, 255, 30)
        painter.fillRect(x, 0, band_w, h, grad)
        painter.end()
        return pix

    def _start_skeletons(self):
        if not self._skeleton_timer.isActive():
            self._skeleton_timer.start()

    def _update_skeletons(self):
        self._skeleton_phase += 8
        any_loading = False
        for lw in (self.results, self.recent, self.recommended, self.featured):
            for i in range(lw.count()):
                item = lw.item(i)
                if item.data(Qt.ItemDataRole.UserRole) != "loading":
                    continue
                any_loading = True
                pix = self._make_skeleton_pixmap(lw.iconSize(), self._skeleton_phase)
                item.setIcon(QIcon(pix))
        if not any_loading:
            self._skeleton_timer.stop()

    def _apply_cached_badge(self, pix: QPixmap) -> QPixmap:
        out = QPixmap(pix)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        radius = max(8, out.width() // 14)
        painter.setBrush(QColor(46, 184, 92, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(out.width() - radius * 2 - 6, 6, radius * 2, radius * 2)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(out.width() - radius * 2 - 6, 6, radius * 2, radius * 2, Qt.AlignmentFlag.AlignCenter, "C")
        painter.end()
        return out

    def _begin_request(self):
        self._requests_in_flight += 1
        if not self._spinner_timer.isActive():
            self._spinner_timer.start()

    def _end_request(self):
        self._requests_in_flight = max(0, self._requests_in_flight - 1)
        if self._requests_in_flight == 0:
            self._spinner_timer.stop()
            self.lbl_spinner.setText("")

    def _spin_tick(self):
        if self._requests_in_flight <= 0:
            self.lbl_spinner.setText("")
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.lbl_spinner.setText(self._spinner_frames[self._spinner_idx])

    @staticmethod
    def _default_settings_dict() -> dict[str, Any]:
        return SettingsService.default_settings(state_dir=app_state_dir())

    def _load_settings(self) -> dict[str, Any]:
        return SettingsService.load(
            path=SETTINGS_PATH,
            state_dir=app_state_dir(),
            normalize_token=self._normalize_anilist_token,
        )

    def _save_settings(self, settings: dict[str, Any]):
        if self._incognito_enabled:
            return
        self._settings = SettingsService.save(
            settings,
            path=SETTINGS_PATH,
            state_dir=app_state_dir(),
            normalize_token=self._normalize_anilist_token,
        )

    def _apply_runtime_from_settings(self):
        runtime = self._runtime_settings()
        MainWindow._assign_runtime_settings(self, runtime)
        os.makedirs(self.download_dir, exist_ok=True)

        pidx = self.provider_combo.findData(self.provider_name)
        if pidx >= 0:
            self.provider_combo.setCurrentIndex(pidx)
        self.lang_combo.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.quality_combo.findText(self.quality)
        if qidx >= 0:
            self.quality_combo.setCurrentIndex(qidx)
        self.combo_dl_parallel.setCurrentIndex(self._max_parallel_downloads - 1)
        self._apply_provider_ui_state()
        self._apply_app_language_ui()
        self._update_featured_tab_visibility()
        self._featured_loaded = False

    def _sync_settings_ui_from_state(self):
        if not hasattr(self, "input_settings_download_dir"):
            return
        self.input_settings_download_dir.setText(self.download_dir)
        pidx = self.combo_settings_provider.findData(self.provider_name)
        if pidx >= 0:
            self.combo_settings_provider.setCurrentIndex(pidx)
        self.combo_settings_lang.setCurrentIndex(1 if self.lang == LanguageTypeEnum.DUB else 0)
        qidx = self.combo_settings_quality.findData(self.quality)
        if qidx >= 0:
            self.combo_settings_quality.setCurrentIndex(qidx)
        didx = self.combo_settings_parallel.findData(self._max_parallel_downloads)
        if didx >= 0:
            self.combo_settings_parallel.setCurrentIndex(didx)
        sidx = self.combo_settings_scheduler_enabled.findData(self._scheduler_enabled)
        if sidx >= 0:
            self.combo_settings_scheduler_enabled.setCurrentIndex(sidx)
        self.input_settings_scheduler_start.setText(self._scheduler_start)
        self.input_settings_scheduler_end.setText(self._scheduler_end)
        self.input_settings_integrity_min_mb.setText(str(self._integrity_min_mb))
        ridx = self.combo_settings_integrity_retries.findData(self._integrity_retry_count)
        if ridx >= 0:
            self.combo_settings_integrity_retries.setCurrentIndex(ridx)
        aidx = self.combo_settings_anilist_enabled.findData(self._anilist_enabled)
        if aidx >= 0:
            self.combo_settings_anilist_enabled.setCurrentIndex(aidx)
        lidx = self.combo_settings_app_language.findData(self.app_language)
        if lidx >= 0:
            self.combo_settings_app_language.setCurrentIndex(lidx)
        self.input_settings_anilist_token.setText(self._anilist_token)

    def on_settings_browse_download_dir(self):
        start = self.input_settings_download_dir.text().strip() or self.download_dir
        picked = QFileDialog.getExistingDirectory(
            self,
            self._tr("Seleziona cartella download", "Select download directory"),
            start,
        )
        if picked:
            self.input_settings_download_dir.setText(os.path.abspath(picked))

    def on_settings_save(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: salvataggio settings bloccato.")
            return
        download_dir = self.input_settings_download_dir.text().strip()
        if not download_dir:
            self.notify_err("Download directory non valido.")
            return
        try:
            integrity_min_mb = float(self.input_settings_integrity_min_mb.text().strip())
        except Exception:
            integrity_min_mb = self._integrity_min_mb
        try:
            prepared = SettingsService.prepare_for_save(
                download_dir=download_dir,
                provider_name=str(self.combo_settings_provider.currentData() or "allanime"),
                lang_name=str(self.combo_settings_lang.currentData() or "SUB").upper(),
                quality=str(self.combo_settings_quality.currentData() or "best"),
                parallel_downloads=int(self.combo_settings_parallel.currentData() or 2),
                scheduler_enabled=bool(self.combo_settings_scheduler_enabled.currentData()),
                scheduler_start=self.input_settings_scheduler_start.text().strip(),
                scheduler_end=self.input_settings_scheduler_end.text().strip(),
                integrity_min_mb=integrity_min_mb,
                integrity_retry_count=int(self.combo_settings_integrity_retries.currentData() or 1),
                anilist_enabled=bool(self.combo_settings_anilist_enabled.currentData()),
                anilist_token=self.input_settings_anilist_token.text(),
                app_language=str(self.combo_settings_app_language.currentData() or "it").strip().lower(),
                normalize_token=self._normalize_anilist_token,
            )
        except ValueError as ex:
            if str(ex) == "invalid_scheduler":
                self.notify_err("Formato scheduler non valido. Usa HH:MM (es. 23:30).")
            elif str(ex) == "missing_anilist_token":
                self.notify_err("Inserisci un AniList token oppure disattiva AniList sync.")
            else:
                self.notify_err("Download directory non valido.")
            return

        try:
            os.makedirs(prepared.runtime.download_dir, exist_ok=True)
        except Exception as ex:
            self.notify_err(f"Impossibile creare la cartella download: {ex}")
            return

        try:
            self._save_settings(prepared.settings)
        except Exception as ex:
            self.notify_err(f"Errore salvataggio settings: {ex}")
            return

        self._apply_runtime_from_settings()
        self.refresh_history_ui()
        self.refresh_offline_library()
        self.set_status(self._tr("Impostazioni salvate.", "Settings saved."))

    def on_settings_reset(self):
        defaults = self._default_settings_dict()
        self.input_settings_download_dir.setText(str(defaults["download_dir"]))
        pidx = self.combo_settings_provider.findData(defaults["default_provider"])
        if pidx >= 0:
            self.combo_settings_provider.setCurrentIndex(pidx)
        self.combo_settings_lang.setCurrentIndex(0)
        qidx = self.combo_settings_quality.findData(defaults["default_quality"])
        if qidx >= 0:
            self.combo_settings_quality.setCurrentIndex(qidx)
        didx = self.combo_settings_parallel.findData(defaults["parallel_downloads"])
        if didx >= 0:
            self.combo_settings_parallel.setCurrentIndex(didx)
        sidx = self.combo_settings_scheduler_enabled.findData(defaults["scheduler_enabled"])
        if sidx >= 0:
            self.combo_settings_scheduler_enabled.setCurrentIndex(sidx)
        self.input_settings_scheduler_start.setText(str(defaults["scheduler_start"]))
        self.input_settings_scheduler_end.setText(str(defaults["scheduler_end"]))
        self.input_settings_integrity_min_mb.setText(str(defaults["integrity_min_mb"]))
        ridx = self.combo_settings_integrity_retries.findData(defaults["integrity_retry_count"])
        if ridx >= 0:
            self.combo_settings_integrity_retries.setCurrentIndex(ridx)
        aidx = self.combo_settings_anilist_enabled.findData(defaults["anilist_enabled"])
        if aidx >= 0:
            self.combo_settings_anilist_enabled.setCurrentIndex(aidx)
        lidx = self.combo_settings_app_language.findData(defaults["app_language"])
        if lidx >= 0:
            self.combo_settings_app_language.setCurrentIndex(lidx)
        self.input_settings_anilist_token.setText(str(defaults["anilist_token"]))
        self.set_status(self._tr("Impostazioni ripristinate ai default (premi Salva).", "Settings reset to defaults (press Save)."))

    @staticmethod
    def _parse_hhmm(v: str) -> tuple[int, int] | None:
        return SettingsService.parse_hhmm(v)

    def _is_scheduler_window_open(self) -> bool:
        now = time.localtime()
        return SettingsService.scheduler_window_open(
            scheduler_enabled=self._scheduler_enabled,
            scheduler_start=self._scheduler_start,
            scheduler_end=self._scheduler_end,
            current_minutes=(now.tm_hour * 60 + now.tm_min),
        )

    def _scheduler_tick(self):
        if self._scheduler_enabled and self._is_scheduler_window_open():
            self._start_next_downloads()

    def on_settings_backup(self):
        default_name = f"anigui_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("Crea backup", "Create backup"),
            os.path.join(self.download_dir, default_name),
            self._tr("File zip (*.zip)", "Zip files (*.zip)"),
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".zip"):
            out_path += ".zip"
        try:
            AppStateService.create_backup_archive(
                out_path,
                state_dir=app_state_dir(),
                path_overrides={
                    "history.json": HISTORY_PATH,
                    "search_history.json": SEARCH_HISTORY_PATH,
                    "offline_covers.json": OFFLINE_COVERS_MAP_PATH,
                    "settings.json": SETTINGS_PATH,
                    "favorites.json": FAVORITES_PATH,
                    "metadata_cache.json": METADATA_CACHE_PATH,
                },
            )
            self.set_status(f"Backup creato: {out_path}")
        except Exception as ex:
            self.notify_err(f"Backup fallito: {ex}")

    def on_settings_restore(self):
        if self._incognito_enabled:
            self.set_status("Incognito attivo: restore bloccato.")
            return
        in_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("Ripristina backup", "Restore backup"),
            self.download_dir,
            self._tr("File zip (*.zip)", "Zip files (*.zip)"),
        )
        if not in_path:
            return
        try:
            AppStateService.restore_backup_archive(
                in_path,
                state_dir=app_state_dir(),
                path_overrides={
                    "history.json": HISTORY_PATH,
                    "search_history.json": SEARCH_HISTORY_PATH,
                    "offline_covers.json": OFFLINE_COVERS_MAP_PATH,
                    "settings.json": SETTINGS_PATH,
                    "favorites.json": FAVORITES_PATH,
                    "metadata_cache.json": METADATA_CACHE_PATH,
                },
            )

            # reload in-memory data from restored files
            self.history.load()
            self._run_history_maintenance()
            self.refresh_history_ui()
            self._settings = self._load_settings()
            self._apply_runtime_from_settings()
            self._load_search_history()
            self._load_offline_covers_map()
            self._load_favorites()
            self._load_metadata_cache()
            self.refresh_favorites_ui()
            self._sync_settings_ui_from_state()
            self.refresh_offline_library()
            self.set_status("Restore completato.")
        except Exception as ex:
            self.notify_err(f"Restore fallito: {ex}")


    def _load_favorites(self):
        try:
            with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                self._favorite_items = [FavoriteEntry(**x) for x in raw if isinstance(x, dict)]
            else:
                self._favorite_items = []
        except Exception:
            self._favorite_items = []

    def _save_favorites(self):
        if self._incognito_enabled:
            return
        tmp = FAVORITES_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([asdict(x) for x in self._favorite_items], f, ensure_ascii=False, indent=2)
            os.replace(tmp, FAVORITES_PATH)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _favorite_exists(self, source: str, identifier: str) -> bool:
        for x in self._favorite_items:
            if x.source == source and x.identifier == identifier:
                return True
        return False

    def _add_favorite_from_item(self, item: SearchItem):
        if self._favorite_exists(item.source, item.identifier):
            return
        self._favorite_items.append(
            FavoriteEntry(
                name=item.name,
                identifier=item.identifier,
                source=item.source,
                cover_url=item.cover_url,
                added_at=time.time(),
            )
        )
        self._save_favorites()
        self.refresh_favorites_ui()

    def refresh_favorites_ui(self):
        self.favorites_list.clear()
        if not self._favorite_items:
            self.favorites_list.addItem(self._tr("(nessun preferito)", "(no favorites)"))
            self.favorites_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return
        ordered = sorted(self._favorite_items, key=lambda x: x.added_at, reverse=True)
        self._favorite_items = ordered
        placeholder = self._make_cover_placeholder()
        for row, fav in enumerate(self._favorite_items):
            it = QListWidgetItem(placeholder, fav.name)
            it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            it.setSizeHint(QSize(170, 290))
            self.favorites_list.addItem(it)
            if fav.cover_url:
                w = Worker(self.do_fetch_cover, fav.cover_url)
                w.ok.connect(lambda data, r=row: self._on_favorite_cover_loaded(r, data[0], data[1]))
                w.err.connect(lambda _msg: None)
                self._track_worker(w)
                w.start()

    def _on_favorite_cover_loaded(self, row: int, data: bytes | None, from_cache: bool):
        if data is None or row < 0 or row >= self.favorites_list.count():
            return
        pix = QPixmap()
        if not pix.loadFromData(data):
            return
        if from_cache:
            pix = self._apply_cached_badge(pix)
        scaled = pix.scaled(
            self.favorites_list.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.favorites_list.item(row).setIcon(QIcon(scaled))

    def on_favorite_add_current(self):
        if self.selected_search_item is not None:
            self._add_favorite_from_item(self.selected_search_item)
            self.set_status("Aggiunto ai preferiti.")
            return
        if self.selected_anime is not None:
            ident = str(
                getattr(self.selected_anime, "identifier", None)
                or getattr(self.selected_anime, "_identifier", None)
                or getattr(self.selected_anime, "ref", "")
            )
            item = SearchItem(
                name=getattr(self.selected_anime, "name", "anime"),
                identifier=ident,
                languages={LanguageTypeEnum.SUB, LanguageTypeEnum.DUB},
                cover_url=getattr(self.selected_search_item, "cover_url", None) if self.selected_search_item else None,
                source=self.provider_name,
            )
            self._add_favorite_from_item(item)
            self.set_status("Aggiunto ai preferiti.")

    def on_favorite_remove_selected(self):
        row = self.favorites_list.currentRow()
        if row < 0 or row >= len(self._favorite_items):
            return
        self._favorite_items.pop(row)
        self._save_favorites()
        self.refresh_favorites_ui()

    def on_pick_favorite(self, *_args):
        row = self.favorites_list.currentRow()
        if row < 0 or row >= len(self._favorite_items):
            return
        fav = self._favorite_items[row]
        item = SearchItem(
            name=fav.name,
            identifier=fav.identifier,
            languages={LanguageTypeEnum.SUB, LanguageTypeEnum.DUB},
            cover_url=fav.cover_url,
            source=fav.source,
        )
        self._set_provider(item.source)
        self.selected_search_item = item
        self.selected_anime = self.build_anime_from_item(item)
        self.lbl_anime_title.setText(item.name)
        self.lbl_player_title.setText(item.name)
        self.tabs.setCurrentWidget(self.tab_search)
        self.search_stack.setCurrentWidget(self.page_anime)
        self._current_search_view = "anime"
        self._refresh_search_layout()
        self.fetch_episodes()

    def _metadata_cache_ttl_seconds(self) -> float:
        return 7 * 24 * 3600.0

    def _metadata_key(self, source: str, identifier: str) -> str:
        return AppStateService.metadata_cache_key(source, identifier)

    def _load_metadata_cache(self):
        self._metadata_cache = AppStateService.load_metadata_cache(METADATA_CACHE_PATH)

    def _save_metadata_cache(self):
        if self._incognito_enabled:
            return
        try:
            AppStateService.save_metadata_cache(self._metadata_cache, METADATA_CACHE_PATH)
        except Exception:
            pass

    def _apply_metadata_cache_to_item(self, it: SearchItem):
        key = self._metadata_key(it.source, it.identifier)
        obj = self._metadata_cache.get(key)
        if not isinstance(obj, dict):
            return
        ts = float(obj.get("updated_at", 0.0) or 0.0)
        if (time.time() - ts) > self._metadata_cache_ttl_seconds():
            return
        if it.year is None:
            it.year = obj.get("year")
        if not it.season:
            it.season = obj.get("season")
        if not it.studio:
            it.studio = obj.get("studio")
        if it.rating is None:
            it.rating = obj.get("rating")

    def _store_metadata_cache_for_item(self, it: SearchItem):
        if self._incognito_enabled:
            return
        key = self._metadata_key(it.source, it.identifier)
        self._metadata_cache[key] = AppStateService.metadata_cache_entry(
            updated_at=time.time(),
            year=it.year,
            season=it.season,
            studio=it.studio,
            rating=it.rating,
        )
        self._save_metadata_cache()

    def _load_search_history(self):
        self._recent_queries = AppStateService.load_search_history(SEARCH_HISTORY_PATH)

    def _save_search_history(self):
        if self._incognito_enabled:
            return
        try:
            AppStateService.save_search_history(self._recent_queries, SEARCH_HISTORY_PATH)
        except Exception:
            pass

    def _load_offline_covers_map(self):
        loaded, changed = AppStateService.load_offline_covers_map(
            path=OFFLINE_COVERS_MAP_PATH,
            resolve_legacy_cover_path=self._cover_cache_path,
        )
        self._offline_covers_map = loaded
        if changed:
            self._save_offline_covers_map()

    def _save_offline_covers_map(self):
        if self._incognito_enabled:
            return
        try:
            AppStateService.save_offline_covers_map(self._offline_covers_map, OFFLINE_COVERS_MAP_PATH)
        except Exception:
            pass

    def _remember_offline_cover_url(self, anime_name: str, cover_url: str | None):
        if self._incognito_enabled:
            return
        changed = AppStateService.remember_offline_cover(
            self._offline_covers_map,
            anime_name=anime_name,
            cover_url=cover_url,
            safe_name=self._safe_name,
            cover_cache_path=self._cover_cache_path,
        )
        if changed:
            self._save_offline_covers_map()

    def _add_recent_query(self, q: str):
        q = q.strip()
        if not q:
            return
        self._recent_queries = [x for x in self._recent_queries if x.lower() != q.lower()]
        self._recent_queries.insert(0, q)
        if not self._incognito_enabled:
            self._save_search_history()
        self._refresh_recent_queries_ui()

    def on_clear_search_history(self):
        self._recent_queries = []
        if not self._incognito_enabled:
            AppStateService.clear_search_history(SEARCH_HISTORY_PATH)
        self._refresh_recent_queries_ui()

    def _refresh_recent_queries_ui(self):
        self.list_recent_queries.clear()
        if not self._recent_queries:
            self.list_recent_queries.addItem(self._tr("(nessuna ricerca recente)", "(no recent searches)"))
            self.list_recent_queries.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            return
        for q in self._recent_queries:
            self.list_recent_queries.addItem(q)

    def _update_suggestions_visibility(self):
        empty = not self.search_input.text().strip()
        self.suggestions_panel.setVisible(self._current_search_view == "catalog" and empty)

    def _set_search_results_area_visible(self, visible: bool):
        for w in getattr(self, "_search_filter_widgets", []):
            w.setVisible(visible)
        self.results.setVisible(visible)

    def _refresh_search_layout(self):
        self._update_suggestions_visibility()
        has_query = len(self.search_input.text().strip()) >= 3
        in_catalog = self._current_search_view == "catalog"
        self._set_search_results_area_visible(in_catalog and has_query)
        self.search_stack.setVisible((not in_catalog) or has_query)



    def on_worker_error(self, msg: str):
        self._end_request()
        self.notify_err(msg)
        self.set_status("Errore.")

    def eventFilter(self, obj, event):
        if not hasattr(self, "_video_fs_window"):
            return super().eventFilter(obj, event)
        if obj is self.seek_slider and event.type() == QEvent.Type.Resize:
            self._position_resume_marker()
        if self._is_video_fullscreen() and self._video_fs_window is not None:
            if isinstance(obj, QWidget) and obj.window() is self._video_fs_window:
                t = event.type()
                if t in (
                    QEvent.Type.MouseMove,
                    QEvent.Type.MouseButtonPress,
                    QEvent.Type.MouseButtonRelease,
                    QEvent.Type.Wheel,
                    QEvent.Type.KeyPress,
                    QEvent.Type.KeyRelease,
                ):
                    self._fs_mark_active()
        return super().eventFilter(obj, event)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "hist_list"):
            QTimer.singleShot(0, self._relayout_history_cards)

    def closeEvent(self, e):
        self._save_current_progress(force=True)
        for worker in list(self._active_download_workers.values()):
            try:
                worker.request_cancel()
            except Exception:
                pass
        for worker in list(self._active_download_workers.values()):
            try:
                worker.wait(1200)
            except Exception:
                pass
        for worker in list(self._workers):
            try:
                if hasattr(worker, "request_cancel"):
                    worker.request_cancel()
            except Exception:
                pass
        for worker in list(self._workers):
            try:
                if worker.isRunning():
                    worker.wait(1200)
            except Exception:
                pass
        self._workers.clear()
        try:
            self.player.stop()
        except Exception:
            pass
        if self._mini_player_window is not None:
            self._exit_mini_player()
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        super().closeEvent(e)


def main():
    debug_log("Application start")
    app = LoggingApplication(sys.argv)
    app._ui_freeze_watchdog = UiFreezeWatchdog()
    w = MainWindow()
    w.show()
    debug_log("Application UI shown, entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
