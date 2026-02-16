from __future__ import annotations

import threading
import time

from PySide6.QtCore import QEvent, QPropertyAnimation, Qt, QTimer, QSize
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from components import StreamResult, Worker


def debug_log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    ms = int((time.time() % 1) * 1000)
    thread_name = threading.current_thread().name
    print(f"[anigui {ts}.{ms:03d}] [{thread_name}] {msg}", flush=True)


class PlayerMixin:

    def play_episode(self, ep: float | int):
        if not self.selected_anime:
            return

        t0 = time.perf_counter()
        debug_log(f"Play episode requested: ep={ep}, anime={self.selected_anime.name}")
        self.lbl_player_title.setText(f"{self.selected_anime.name} — Ep {ep}")
        self.search_stack.setCurrentWidget(self.page_player)
        self._current_search_view = "player"
        self._refresh_search_layout()
        self._local_media_active = False
        self._offline_current_episode_index = None
        self.current_ep = ep
        self.set_status(f"Risolvo stream ep {ep}…")
        self._set_transport_enabled(False)

        preferred_quality = (
            self.quality if self.quality in ("best", "worst") else int(self.quality)
        )

        anime = self.selected_anime
        lang = self.lang

        w = Worker(self.do_resolve_stream, anime, ep, lang, preferred_quality, self.provider_name)

        def on_ok(res: StreamResult):
            try:
                debug_log(f"Stream resolved, loading player for ep={ep}")
                self.player.load(res.url, referrer=res.referrer, sub_file=res.sub_file)

                self._last_progress_save_at = 0.0
                self._last_progress_pos = -1.0
                self._last_progress_percent = -1.0
                if self._pending_resume_seek is not None:
                    self._pending_resume_seek_attempts = 0
                    QTimer.singleShot(500, self._try_apply_resume_seek)
                elif self._pending_resume_seek_ratio is not None:
                    QTimer.singleShot(500, self._try_apply_resume_seek_ratio)

                self.set_status(f"▶ {anime.name} — ep {ep}")
                debug_log(f"Episode ready: ep={ep}, total flow {time.perf_counter() - t0:.3f}s")
            finally:
                self._set_transport_enabled(True)

        def on_err(msg: str):
            debug_log(f"Play episode error: ep={ep} err={msg}")
            self.notify_err(msg)
            self.set_status("Errore.")
            self._set_transport_enabled(True)

        w.ok.connect(on_ok)
        w.err.connect(on_err)
        self._workers.append(w)
        w.start()


    def _is_video_fullscreen(self) -> bool:
        fs = getattr(self, "_video_fs_window", None)
        return fs is not None and fs.isVisible()

    def _set_transport_enabled(self, enabled: bool):
        self.btn_playpause.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        if self._fs_btn_playpause:
            self._fs_btn_playpause.setEnabled(enabled)
        if self._fs_btn_prev:
            self._fs_btn_prev.setEnabled(enabled)
        if self._fs_btn_next:
            self._fs_btn_next.setEnabled(enabled)

    def _set_volume_controls(self, v: int, source: str):
        if source != "main" and self.vol_slider.value() != v:
            self.vol_slider.blockSignals(True)
            self.vol_slider.setValue(v)
            self.vol_slider.blockSignals(False)

        if source != "fs" and self._fs_vol_slider and self._fs_vol_slider.value() != v:
            self._fs_vol_slider.blockSignals(True)
            self._fs_vol_slider.setValue(v)
            self._fs_vol_slider.blockSignals(False)

    def _build_fs_overlay(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        panel.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 rgba(0,0,0,0), stop:1 rgba(0,0,0,210));"
            "border-top: 1px solid rgba(255,255,255,45);"
            "color: #f2f2f2;"
        )
        self._fs_overlay_effect = QGraphicsOpacityEffect(panel)
        self._fs_overlay_effect.setOpacity(1.0)
        panel.setGraphicsEffect(self._fs_overlay_effect)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(18, 24, 18, 12)
        pl.setSpacing(10)

        fs_seek = QSlider(Qt.Orientation.Horizontal)
        fs_seek.setRange(0, 1000)
        fs_seek.sliderPressed.connect(self._on_fs_seek_pressed)
        fs_seek.sliderReleased.connect(self._on_fs_seek_released)
        pl.addWidget(fs_seek)

        row = QHBoxLayout()
        row.setSpacing(10)

        fs_prev = QPushButton(self._tr("⏮ Precedente", "⏮ Prev"))
        fs_prev.clicked.connect(self.on_prev_episode)
        row.addWidget(fs_prev)

        fs_playpause = QPushButton(self._tr("⏯ Play/Pause", "⏯ Play/Pause"))
        fs_playpause.clicked.connect(self.on_toggle_pause)
        row.addWidget(fs_playpause)

        fs_next = QPushButton(self._tr("⏭ Successivo", "⏭ Next"))
        fs_next.clicked.connect(self.on_next_episode)
        row.addWidget(fs_next)

        fs_time = QLabel("00:00 / 00:00")
        row.addWidget(fs_time)

        row.addStretch(1)
        row.addWidget(QLabel(self._tr("Volume", "Volume")))

        fs_vol = QSlider(Qt.Orientation.Horizontal)
        fs_vol.setRange(0, 130)
        fs_vol.setFixedWidth(200)
        fs_vol.setValue(self.vol_slider.value())
        fs_vol.valueChanged.connect(self._on_fs_volume_changed)
        row.addWidget(fs_vol)

        fs_exit = QPushButton(self._tr("Esci FS", "Exit FS"))
        fs_exit.clicked.connect(self.toggle_fullscreen)
        row.addWidget(fs_exit)

        pl.addLayout(row)

        self._fs_seek_slider = fs_seek
        self._fs_vol_slider = fs_vol
        self._fs_lbl_time = fs_time
        self._fs_btn_playpause = fs_playpause
        self._fs_btn_prev = fs_prev
        self._fs_btn_next = fs_next
        self._fs_btn_exit = fs_exit
        self._set_transport_enabled(self.btn_playpause.isEnabled())
        return panel

    def _position_fs_overlay(self):
        if not self._video_fs_window or not self._fs_overlay:
            return
        margin = 20
        h = self._fs_overlay.sizeHint().height()
        w = max(480, self._video_fs_window.width() - (margin * 2))
        self._fs_overlay.setGeometry(
            margin,
            max(0, self._video_fs_window.height() - h - 14),
            w,
            h,
        )
        self._fs_overlay.raise_()
        if self._fs_help_label:
            self._fs_help_label.adjustSize()
            self._fs_help_label.move(
                max(12, (self._video_fs_window.width() - self._fs_help_label.width()) // 2),
                14,
            )

    def _fs_mark_active(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if self._fs_overlay:
            self._fs_overlay.show()
            self._fs_fade_overlay(1.0)
            self._position_fs_overlay()
        self._video_fs_window.unsetCursor()
        self._fs_autohide_timer.start()

    def _fs_fade_overlay(self, target: float):
        if not self._fs_overlay_effect:
            return
        if self._fs_overlay_anim:
            self._fs_overlay_anim.stop()
        self._fs_overlay_anim = QPropertyAnimation(self._fs_overlay_effect, b"opacity")
        self._fs_overlay_anim.setDuration(180)
        self._fs_overlay_anim.setStartValue(self._fs_overlay_effect.opacity())
        self._fs_overlay_anim.setEndValue(target)
        if target <= 0.0 and self._fs_overlay:
            self._fs_overlay_anim.finished.connect(self._fs_overlay.hide)
        self._fs_overlay_anim.start()

    def _fs_poll_cursor(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        pos = QCursor.pos()
        if self._fs_last_cursor_pos is None:
            self._fs_last_cursor_pos = pos
            return
        if pos != self._fs_last_cursor_pos:
            self._fs_last_cursor_pos = pos
            self._fs_mark_active()

    def _fs_apply_hidden(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if self._fs_seeking:
            self._fs_autohide_timer.start()
            return
        if self._fs_overlay:
            self._fs_fade_overlay(0.0)
        self._video_fs_window.setCursor(Qt.CursorShape.BlankCursor)

    def _fs_show_help(self):
        if not self._is_video_fullscreen() or not self._video_fs_window:
            return
        if not self._fs_help_label:
            return
        self._fs_help_label.show()
        self._position_fs_overlay()
        QTimer.singleShot(2500, self._fs_help_label.hide)

    def _enter_video_fullscreen(self):
        if self._is_video_fullscreen():
            return
        if self._mini_player_window is not None:
            self._exit_mini_player()

        debug_log("Entering video-only fullscreen")
        fs = FullscreenPlayerWindow()
        fs.setWindowTitle(self._tr("Anigui Player", "Anigui Player"))
        fs.setWindowFlag(Qt.WindowType.Window, True)
        fs.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        fs.exit_requested.connect(self._exit_video_fullscreen)
        fs.activity.connect(self._fs_mark_active)

        fs_layout = QVBoxLayout(fs)
        fs_layout.setContentsMargins(0, 0, 0, 0)
        fs_layout.setSpacing(0)
        self.player.setParent(fs)
        fs_layout.addWidget(self.player, 1)

        self._fs_overlay = self._build_fs_overlay(fs)

        fs_esc = QShortcut(QKeySequence("Escape"), fs, activated=self._exit_video_fullscreen)
        fs_f = QShortcut(QKeySequence("F"), fs, activated=self.toggle_fullscreen)
        fs_help = QShortcut(QKeySequence("?"), fs, activated=self._fs_show_help)
        self._video_fs_shortcuts = [fs_esc, fs_f, fs_help]

        self._fs_help_label = QLabel(
            self._tr(
                "Scorciatoie: F schermo intero, Esc esci, Space play/pausa, ←/→ seek, ↑/↓ volume, ? aiuto",
                "Shortcuts: F fullscreen, Esc exit, Space play/pause, ←/→ seek, ↑/↓ volume, ? help",
            ),
            fs,
        )
        self._fs_help_label.setStyleSheet(
            "background: rgba(0,0,0,180); color: #ffffff; padding: 8px 12px; border-radius: 8px;"
        )
        self._fs_help_label.setVisible(False)

        self._video_fs_window = fs
        fs.showFullScreen()
        self._fs_last_cursor_pos = QCursor.pos()
        self._fs_mouse_poll_timer.start()
        fs.installEventFilter(self)
        self.player.installEventFilter(self)
        if self._fs_overlay:
            self._fs_overlay.installEventFilter(self)
            for w in self._fs_overlay.findChildren(QWidget):
                w.installEventFilter(self)
        self._position_fs_overlay()
        self._fs_mark_active()
        self.btn_fs.setText(self._tr("Esci FS", "Exit FS"))

    def _exit_video_fullscreen(self):
        if self._closing_video_fs:
            return
        fs = self._video_fs_window
        if fs is None:
            return

        self._closing_video_fs = True
        try:
            debug_log("Exiting video-only fullscreen")
            self.player.setParent(self.player_host)
            self.player_host_layout.addWidget(self.player, 1)

            for sc in self._video_fs_shortcuts:
                try:
                    sc.setParent(None)
                except Exception:
                    pass
            self._video_fs_shortcuts = []

            if self._fs_overlay:
                for w in self._fs_overlay.findChildren(QWidget):
                    w.removeEventFilter(self)
                self._fs_overlay.removeEventFilter(self)
                self._fs_overlay.hide()
                self._fs_overlay.deleteLater()
            if self._fs_help_label:
                self._fs_help_label.hide()
                self._fs_help_label.deleteLater()
            self.player.removeEventFilter(self)
            fs.removeEventFilter(self)
            self._fs_autohide_timer.stop()
            self._fs_mouse_poll_timer.stop()
            fs.unsetCursor()
            self._fs_overlay = None
            self._fs_overlay_effect = None
            self._fs_overlay_anim = None
            self._fs_seek_slider = None
            self._fs_vol_slider = None
            self._fs_lbl_time = None
            self._fs_btn_playpause = None
            self._fs_btn_prev = None
            self._fs_btn_next = None
            self._fs_btn_exit = None
            self._fs_help_label = None
            self._fs_seeking = False
            self._fs_last_cursor_pos = None

            fs.hide()
            fs.deleteLater()
            self._video_fs_window = None
            self.btn_fs.setText(self._tr("⛶ Schermo intero", "⛶ Fullscreen"))
        finally:
            self._closing_video_fs = False

    def toggle_fullscreen(self):
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        else:
            self._enter_video_fullscreen()

    def _enter_mini_player(self):
        if self._mini_player_window is not None:
            return
        if self._is_video_fullscreen():
            self._exit_video_fullscreen()
        mini = MiniPlayerWindow()
        mini.setWindowTitle(self._tr("Anigui Mini Player", "Anigui Mini Player"))
        mini.setWindowFlag(Qt.WindowType.Window, True)
        mini.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        mini.exit_requested.connect(self._exit_mini_player)
        lay = QVBoxLayout(mini)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.player.setParent(mini)
        lay.addWidget(self.player, 1)
        mini.resize(520, 300)
        mini.show()
        self._mini_player_window = mini
        self.btn_mini.setText(self._tr("Esci mini", "Exit Mini"))

    def _exit_mini_player(self):
        mini = self._mini_player_window
        if mini is None:
            return
        self.player.setParent(self.player_host)
        self.player_host_layout.addWidget(self.player, 1)
        mini.hide()
        mini.deleteLater()
        self._mini_player_window = None
        self.btn_mini.setText(self._tr("▣ Mini Player", "▣ Mini"))

    def toggle_mini_player(self):
        if self._mini_player_window is None:
            self._enter_mini_player()
        else:
            self._exit_mini_player()

    def on_prev_episode(self):
        if self._local_media_active and self._offline_episode_files:
            if self._offline_current_episode_index is None:
                return
            idx = self._offline_current_episode_index
            if idx <= 0:
                return
            self._offline_current_episode_index = idx - 1
            self._play_local_file(self._offline_episode_files[self._offline_current_episode_index])
            return
        if self.current_ep is None or not self.episodes_list:
            return
        try:
            idx = self.episodes_list.index(self.current_ep)
        except ValueError:
            return
        if idx <= 0:
            return
        self.play_episode(self.episodes_list[idx - 1])

    def on_next_episode(self):
        if self._local_media_active and self._offline_episode_files:
            if self._offline_current_episode_index is None:
                return
            idx = self._offline_current_episode_index
            if idx >= len(self._offline_episode_files) - 1:
                return
            self._offline_current_episode_index = idx + 1
            self._play_local_file(self._offline_episode_files[self._offline_current_episode_index])
            return
        if self.current_ep is None or not self.episodes_list:
            return
        try:
            idx = self.episodes_list.index(self.current_ep)
        except ValueError:
            return
        if idx >= len(self.episodes_list) - 1:
            return
        self.play_episode(self.episodes_list[idx + 1])

    def on_toggle_pause(self):
        try:
            self.player.toggle_pause()
        except Exception:
            pass

    def on_volume_changed(self, v: int):
        try:
            self.player.set_volume(v)
            self._set_volume_controls(v, source="main")
        except Exception:
            pass

    def _on_fs_volume_changed(self, v: int):
        try:
            self.player.set_volume(v)
            self._set_volume_controls(v, source="fs")
            self._fs_mark_active()
        except Exception:
            pass

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_released(self):
        self._seeking = False
        ratio = self.seek_slider.value() / 1000.0
        dur = self.player.get_duration()
        self._timeline_diag(f"seek(main) ratio={ratio:.3f} dur={dur}", force=True)
        try:
            if dur is None or dur <= 0:
                self.player.seek_ratio(ratio)
            else:
                self.player.seek(ratio * dur)
        except Exception:
            pass

    def _position_resume_marker(self):
        if self._resume_marker_pos is None:
            self._resume_marker.hide()
            return
        dur = self.player.get_duration()
        if dur is None or dur <= 0:
            self._resume_marker.hide()
            return
        ratio = max(0.0, min(1.0, self._resume_marker_pos / dur))
        w = self.seek_slider.width()
        x = int(ratio * (w - self._resume_marker.width()))
        self._resume_marker.setGeometry(x, 2, self._resume_marker.width(), self.seek_slider.height() - 4)
        self._resume_marker.show()

    def _on_fs_seek_pressed(self):
        self._fs_seeking = True
        self._fs_mark_active()

    def _on_fs_seek_released(self):
        self._fs_seeking = False
        if self._fs_seek_slider is None:
            return
        ratio = self._fs_seek_slider.value() / 1000.0
        dur = self.player.get_duration()
        self._timeline_diag(f"seek(fs) ratio={ratio:.3f} dur={dur}", force=True)
        try:
            if dur is None or dur <= 0:
                self.player.seek_ratio(ratio)
            else:
                self.player.seek(ratio * dur)
            self._fs_mark_active()
        except Exception:
            pass

    def poll_player(self):
        if not self._local_media_active and self.current_ep is None:
            return
        if hasattr(self.player, "_load_thread") and getattr(self.player, "_load_thread", None) and self.player._load_thread.isRunning():
            return
        
        dur = self.player.get_duration()
        pos = self.player.get_time_pos()
        percent = self.player.get_percent_pos()
        paused = self.player.get_pause()
        if pos is None and dur is not None and dur > 0 and percent is not None:
            pos = max(0.0, min(dur, dur * (percent / 100.0)))
        if pos is None:
            self._timeline_missing_logged = True
            self._timeline_diag(
                f"no-pos dur={dur} percent={percent} paused={paused} player={self.player.__class__.__name__}"
            )
            return
        if self._timeline_missing_logged:
            self._timeline_diag(
                f"recovered pos={pos:.2f} dur={dur} percent={percent} player={self.player.__class__.__name__}",
                force=True,
            )
            self._timeline_missing_logged = False
        if dur is None or dur <= 0:
            self.lbl_time.setText(f"{self.fmt_time(pos)} / --:--")
        else:
            self.lbl_time.setText(f"{self.fmt_time(pos)} / {self.fmt_time(dur)}")
        if self._fs_lbl_time is not None:
            if dur is None or dur <= 0:
                self._fs_lbl_time.setText(f"{self.fmt_time(pos)} / --:--")
            else:
                self._fs_lbl_time.setText(f"{self.fmt_time(pos)} / {self.fmt_time(dur)}")

        if not self._seeking and dur is not None and dur > 0:
            ratio = max(0.0, min(1.0, pos / dur))
            if percent is not None:
                ratio = max(0.0, min(1.0, percent / 100.0))
            v = int(ratio * 1000)
            self.seek_slider.setValue(v)
            if self._fs_seek_slider is not None and not self._fs_seeking:
                self._fs_seek_slider.setValue(v)
        self._position_resume_marker()

        if paused is True:
            self.btn_playpause.setText(self._tr("▶ Play", "▶ Play"))
            if self._fs_btn_playpause is not None:
                self._fs_btn_playpause.setText(self._tr("▶ Play", "▶ Play"))
        elif paused is False:
            self.btn_playpause.setText(self._tr("⏸ Pausa", "⏸ Pause"))
            if self._fs_btn_playpause is not None:
                self._fs_btn_playpause.setText(self._tr("⏸ Pausa", "⏸ Pause"))

        self._save_current_progress(force=False)

    # ---------------- Worker errors ----------------
