from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox

from components import Worker
from services import APP_VERSION


class UpdateMixin:

    def _update_fetch_manifest_worker(self) -> dict[str, Any]:
        return self._update_service.fetch_manifest(platform.system())

    def _start_update_check(self, silent: bool):
        self.btn_settings_update_check.setEnabled(False)
        w = Worker(self._update_fetch_manifest_worker)
        w.ok.connect(lambda info, sl=silent: self._on_update_check_ok(info, sl))
        w.err.connect(lambda msg, sl=silent: self._on_update_check_err(msg, sl))
        self._track_worker(w)
        w.start()

    def check_updates_silent(self):
        self._start_update_check(True)

    def on_update_check_clicked(self):
        self.set_status(self._tr("Controllo aggiornamenti in corso...", "Checking for updates..."))
        self._start_update_check(False)

    def _on_update_check_ok(self, info: dict[str, Any], silent: bool):
        self.btn_settings_update_check.setEnabled(True)
        self._update_info = info if isinstance(info, dict) else None
        available = bool((info or {}).get("available"))
        latest = str((info or {}).get("latest_version", APP_VERSION))
        if not available:
            self.btn_settings_update_apply.setEnabled(False)
            self._update_download_path = None
            if not silent:
                QMessageBox.information(
                    self,
                    self._tr("Aggiornamenti", "Updates"),
                    self._tr(
                        f"Sei gia all'ultima versione ({APP_VERSION}).",
                        f"You are already on the latest version ({APP_VERSION}).",
                    ),
                )
                self.set_status(self._tr("Nessun aggiornamento disponibile.", "No updates available."))
            return

        can_apply = bool((info or {}).get("url") or (info or {}).get("page_url"))
        self.btn_settings_update_apply.setEnabled(can_apply)
        self._update_download_path = None
        self.set_status(
            self._tr(
                f"Aggiornamento disponibile: {APP_VERSION} -> {latest}",
                f"Update available: {APP_VERSION} -> {latest}",
            )
        )
        if not silent:
            QMessageBox.information(
                self,
                self._tr("Aggiornamenti", "Updates"),
                self._tr(
                    f"Nuova versione disponibile: {latest}",
                    f"New version available: {latest}",
                ),
            )

    def _on_update_check_err(self, msg: str, silent: bool):
        self.btn_settings_update_check.setEnabled(True)
        self.btn_settings_update_apply.setEnabled(False)
        if not silent:
            self.notify_err(self._tr(f"Check update fallito: {msg}", f"Update check failed: {msg}"))
            self.set_status(self._tr("Check aggiornamenti fallito.", "Update check failed."))

    def _download_update_worker(self, info: dict[str, Any]) -> str:
        return self._update_service.download_update(info, platform.system())

    def _open_update_page(self):
        info = self._update_info or {}
        page_url = str(info.get("page_url") or info.get("url") or "").strip()
        if not page_url:
            self.notify_err(self._tr("URL update non disponibile.", "Update URL not available."))
            return
        QDesktopServices.openUrl(QUrl(page_url))

    def on_update_apply_clicked(self):
        info = self._update_info or {}
        if not bool(info.get("available")):
            self.notify_err(self._tr("Nessun aggiornamento disponibile.", "No updates available."))
            return

        is_win_frozen = platform.system() == "Windows" and bool(getattr(sys, "frozen", False))
        if not is_win_frozen:
            self._open_update_page()
            self.set_status(self._tr("Aperta pagina aggiornamento.", "Update page opened."))
            return

        if self._update_download_path and os.path.exists(self._update_download_path):
            self._apply_windows_update_and_restart(self._update_download_path)
            return

        self.btn_settings_update_apply.setEnabled(False)
        self.set_status(self._tr("Download aggiornamento in corso...", "Downloading update..."))
        w = Worker(self._download_update_worker, dict(info))
        w.ok.connect(self._on_update_download_ok)
        w.err.connect(self._on_update_download_err)
        self._track_worker(w)
        w.start()

    def _on_update_download_ok(self, path: str):
        self._update_download_path = str(path or "").strip()
        if not self._update_download_path:
            self._on_update_download_err("download-path-empty")
            return
        self._apply_windows_update_and_restart(self._update_download_path)

    def _on_update_download_err(self, msg: str):
        self.btn_settings_update_apply.setEnabled(True)
        self.notify_err(self._tr(f"Download update fallito: {msg}", f"Update download failed: {msg}"))
        self.set_status(self._tr("Download aggiornamento fallito.", "Update download failed."))

    def _apply_windows_update_and_restart(self, downloaded_exe: str):
        if not (platform.system() == "Windows" and bool(getattr(sys, "frozen", False))):
            self._open_update_page()
            return
        current_exe = os.path.abspath(sys.executable)
        src = os.path.abspath(downloaded_exe)
        if not os.path.exists(src):
            self.notify_err(self._tr("File update mancante.", "Update file missing."))
            return
        if not src.lower().endswith(".exe"):
            self.notify_err(self._tr("Update non valido: file .exe atteso.", "Invalid update: .exe file expected."))
            return
        cur_pid = os.getpid()
        bat_path = os.path.join(tempfile.gettempdir(), f"anigui_updater_{int(time.time())}.bat")
        log_path = os.path.join(tempfile.gettempdir(), "anigui_updater.log")
        bat = (
            "@echo off\n"
            "setlocal\n"
            f"set \"SRC={src}\"\n"
            f"set \"DST={current_exe}\"\n"
            f"set \"PID={cur_pid}\"\n"
            "set \"MAX_TRIES=60\"\n"
            "set \"TRY=0\"\n"
            f"set \"LOG={log_path}\"\n"
            "echo ==== anigui updater start %date% %time% ==== > \"%LOG%\"\n"
            "echo SRC=%SRC%>>\"%LOG%\"\n"
            "echo DST=%DST%>>\"%LOG%\"\n"
            "echo PID=%PID%>>\"%LOG%\"\n"
            ":waitclose\n"
            "tasklist /FI \"PID eq %PID%\" | findstr /I \"%PID%\" > nul\n"
            "if not errorlevel 1 (\n"
            "  timeout /t 1 /nobreak > nul\n"
            "  goto waitclose\n"
            ")\n"
            "echo Process closed, starting copy retries...>>\"%LOG%\"\n"
            ":copyretry\n"
            "set /a TRY=%TRY%+1\n"
            "copy /Y \"%SRC%\" \"%DST%\" > nul 2> nul\n"
            "if errorlevel 1 (\n"
            "  echo Copy failed attempt %TRY%>>\"%LOG%\"\n"
            "  if %TRY% GEQ %MAX_TRIES% goto copyfail\n"
            "  timeout /t 1 /nobreak > nul\n"
            "  goto copyretry\n"
            ")\n"
            "echo Copy ok after %TRY% attempts>>\"%LOG%\"\n"
            "set \"_MEIPASS2=\"\n"
            "set \"_PYI_APPLICATION_HOME_DIR=\"\n"
            "set \"_PYI_ARCHIVE_FILE=\"\n"
            "set \"_PYI_PARENT_PROCESS_LEVEL=\"\n"
            "set \"_PYI_SPLASH_IPC=\"\n"
            "set \"PYTHONHOME=\"\n"
            "set \"PYTHONPATH=\"\n"
            "if errorlevel 1 (\n"
            "  start \"\" \"%SRC%\"\n"
            ") else (\n"
            "  start \"\" \"%DST%\"\n"
            ")\n"
            "del \"%SRC%\" > nul 2>&1\n"
            "echo updater done>>\"%LOG%\"\n"
            "del \"%~f0\" > nul 2>&1\n"
            "exit /b 0\n"
            ":copyfail\n"
            "echo Copy failed permanently, launching downloaded exe>>\"%LOG%\"\n"
            "set \"_MEIPASS2=\"\n"
            "set \"_PYI_APPLICATION_HOME_DIR=\"\n"
            "set \"_PYI_ARCHIVE_FILE=\"\n"
            "set \"_PYI_PARENT_PROCESS_LEVEL=\"\n"
            "set \"_PYI_SPLASH_IPC=\"\n"
            "set \"PYTHONHOME=\"\n"
            "set \"PYTHONPATH=\"\n"
            "start \"\" \"%SRC%\"\n"
        )
        with open(bat_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(bat)
        self.set_status(self._tr("Riavvio per applicare update...", "Restarting to apply update..."))
        try:
            subprocess.Popen(["cmd", "/c", bat_path], cwd=os.path.dirname(current_exe))
        except Exception as ex:
            self.notify_err(self._tr(f"Avvio updater fallito: {ex}", f"Updater launch failed: {ex}"))
            return
        QApplication.instance().quit()
