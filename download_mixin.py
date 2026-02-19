from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Any

from PySide6.QtCore import Qt, QUrl, QProcess
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QListWidgetItem

from anipy_api.provider import LanguageTypeEnum
from components import DownloadTask, DownloadWorker, StreamResult, Worker


def debug_log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    ms = int((time.time() % 1) * 1000)
    thread_name = threading.current_thread().name
    print(f"[anigui {ts}.{ms:03d}] [{thread_name}] {msg}", flush=True)


class DownloadMixin:

    def _timeline_diag(self, msg: str, force: bool = False):
        now = time.time()
        if not force and (now - self._last_timeline_diag_at) < 2.0:
            return
        self._last_timeline_diag_at = now
        debug_log(f"timeline: {msg}")

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
        return cleaned or "anime"

    @staticmethod
    def _compact_title(title: str, max_len: int = 34) -> str:
        t = (title or "").strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 1].rstrip() + "…"

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(max(0, n))
        i = 0
        while v >= 1024.0 and i < len(units) - 1:
            v /= 1024.0
            i += 1
        return f"{v:.1f}{units[i]}"

    def _check_download_integrity(self, out_path: str) -> str | None:
        try:
            sz = os.path.getsize(out_path)
        except Exception:
            return "file mancante"
        min_bytes = int(max(0.0, self._integrity_min_mb) * 1024 * 1024)
        if sz < min_bytes:
            return f"file troppo piccolo ({self._fmt_bytes(sz)} < {self._fmt_bytes(min_bytes)})"
        return None

    def _download_label(self, t: DownloadTask) -> str:
        ep = t.episode
        st = t.status
        tag_done = "[DONE]" if self.app_language == "en" else "[FATTO]"
        tag_fail = "[FAIL]" if self.app_language == "en" else "[ERRORE]"
        tag_stop = "[STOP]" if self.app_language == "en" else "[STOP]"
        tag_resolve = "[RESOLVE]" if self.app_language == "en" else "[RISOLVI]"
        tag_queue = "[QUEUE]" if self.app_language == "en" else "[CODA]"
        if st == "downloading":
            if t.total and t.total > 0:
                pct = int(max(0.0, min(100.0, t.progress)))
                return (
                    f"[{pct:3d}%] {t.anime_name} · ep {ep} · "
                    f"{self._fmt_bytes(t.downloaded)}/{self._fmt_bytes(t.total)}"
                )
            return f"[..] {t.anime_name} · ep {ep} · {self._fmt_bytes(t.downloaded)}"
        if st == "completed":
            return f"{tag_done} {t.anime_name} · ep {ep}"
        if st == "failed":
            return f"{tag_fail} {t.anime_name} · ep {ep} · {t.error or self._tr('errore', 'error')}"
        if st == "cancelled":
            return f"{tag_stop} {t.anime_name} · ep {ep}"
        if st == "resolving":
            return f"{tag_resolve} {t.anime_name} · ep {ep}"
        return f"{tag_queue} {t.anime_name} · ep {ep}"

    def _refresh_downloads_ui(self):
        self.downloads_list.clear()
        for task_id in self._download_order:
            t = self._download_tasks.get(task_id)
            if not t:
                continue
            item = QListWidgetItem(self._download_label(t))
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self.downloads_list.addItem(item)

    def _queue_download(self, anime_obj: Any, anime_name: str, ep: float | int):
        self._remember_offline_cover_url(
            anime_name,
            getattr(self.selected_search_item, "cover_url", None) if self.selected_search_item else None,
        )
        ep_token = str(int(ep)) if isinstance(ep, float) and ep.is_integer() else str(ep)
        task_id = f"{self.provider_name}:{getattr(anime_obj, 'name', anime_name)}:{ep_token}:{int(time.time()*1000)}"
        lang_tag = "dub" if self.lang == LanguageTypeEnum.DUB else "sub"
        base_name = self._safe_name(anime_name)
        anime_dir = os.path.join(self.download_dir, base_name)
        os.makedirs(anime_dir, exist_ok=True)
        ext = ".mp4"

        # If this episode already exists locally, mark task as done and skip re-download.
        existing_pat = re.compile(
            rf"^{re.escape(base_name)} - ep {re.escape(ep_token)} \[{re.escape(lang_tag)}\](?: \(\d+\))?{re.escape(ext)}$",
            re.IGNORECASE,
        )
        existing_path = None
        try:
            for fn in os.listdir(anime_dir):
                if existing_pat.match(fn):
                    p = os.path.join(anime_dir, fn)
                    if os.path.isfile(p):
                        existing_path = p
                        break
        except Exception:
            existing_path = None

        if existing_path is not None:
            task = DownloadTask(
                task_id=task_id,
                anime_name=anime_name,
                episode=ep,
                provider=self.provider_name,
                lang=self.lang,
                quality=self.quality,
                anime_obj=anime_obj,
                out_path=existing_path,
                status="completed",
                progress=100.0,
                downloaded=os.path.getsize(existing_path),
                total=os.path.getsize(existing_path),
                error=None,
                max_retries=self._integrity_retry_count,
            )
            self._download_tasks[task_id] = task
            self._download_order.append(task_id)
            self._refresh_downloads_ui()
            self.set_status(f"Gia presente: {os.path.basename(existing_path)}")
            self.refresh_offline_library()
            return

        out_name = f"{base_name} - ep {ep_token} [{lang_tag}]{ext}"
        out_path = os.path.join(anime_dir, out_name)
        suffix = 1
        while os.path.exists(out_path):
            out_name = f"{base_name} - ep {ep_token} [{lang_tag}] ({suffix}){ext}"
            out_path = os.path.join(anime_dir, out_name)
            suffix += 1

        task = DownloadTask(
            task_id=task_id,
            anime_name=anime_name,
            episode=ep,
            provider=self.provider_name,
            lang=self.lang,
            quality=self.quality,
            anime_obj=anime_obj,
            out_path=out_path,
            max_retries=self._integrity_retry_count,
        )
        self._download_tasks[task_id] = task
        self._download_order.append(task_id)
        self._refresh_downloads_ui()

    def on_download_add_current(self):
        anime = self.selected_anime
        if anime is None:
            return
        ep = self.current_ep
        if ep is None and self.episodes_list:
            row = self.episodes.currentRow()
            if row >= 0:
                ep = self.episodes_list[row]
        if ep is None:
            self.notify_err("Seleziona o avvia un episodio prima di aggiungerlo alla coda download.")
            return
        name = getattr(anime, "name", None) or getattr(self.selected_search_item, "name", "anime")
        self._queue_download(anime, str(name), ep)
        self.set_status(f"Aggiunto download: {name} ep {ep}")

    def on_download_add_selected_episodes(self):
        anime = self.selected_anime
        if anime is None:
            self.notify_err("Apri prima un anime.")
            return
        selected = self.episodes.selectedIndexes()
        if not selected:
            self.notify_err("Seleziona uno o piu episodi dalla lista.")
            return
        name = getattr(anime, "name", None) or getattr(self.selected_search_item, "name", "anime")
        added = 0
        in_flight = {
            (t.provider, t.anime_name, str(t.episode))
            for t in self._download_tasks.values()
            if t.status in ("queued", "resolving", "downloading")
        }
        for idx in selected:
            row = idx.row()
            if row < 0 or row >= len(self.episodes_list):
                continue
            ep = self.episodes_list[row]
            dedupe_key = (self.provider_name, str(name), str(ep))
            if dedupe_key in in_flight:
                continue
            self._queue_download(anime, str(name), ep)
            in_flight.add(dedupe_key)
            added += 1
        if added > 0:
            self.set_status(f"Aggiunti {added} episodi alla coda download.")
        else:
            self.set_status("Nessun nuovo episodio aggiunto (gia in coda/attivi).")

    def on_download_start_queue(self):
        self._start_next_downloads()

    def on_download_parallel_change(self):
        self._max_parallel_downloads = int(self.combo_dl_parallel.currentData() or 1)
        self._start_next_downloads()

    def _start_next_downloads(self):
        if not self._is_scheduler_window_open():
            self.set_status("Scheduler attivo: fuori finestra oraria, coda in pausa.")
            return
        while (len(self._active_download_workers) + len(self._active_download_resolve_ids)) < self._max_parallel_downloads:
            next_task = None
            for task_id in self._download_order:
                t = self._download_tasks.get(task_id)
                if t and t.status == "queued":
                    next_task = t
                    break
            if next_task is None:
                return

            next_task.status = "resolving"
            self._active_download_resolve_ids.add(next_task.task_id)
            self._refresh_downloads_ui()
            self.set_status(f"Risolvo stream per download: {next_task.anime_name} ep {next_task.episode}")

            w = Worker(
                self.do_resolve_stream,
                next_task.anime_obj,
                next_task.episode,
                next_task.lang,
                next_task.quality if next_task.provider == "allanime" else "best",
                next_task.provider,
            )

            def on_ok(res: StreamResult, tid=next_task.task_id):
                self._active_download_resolve_ids.discard(tid)
                task = self._download_tasks.get(tid)
                if not task or task.status != "resolving":
                    self._start_next_downloads()
                    return
                task.status = "downloading"
                self._refresh_downloads_ui()
                dw = DownloadWorker(tid, res.url, task.out_path, referrer=res.referrer)
                self._active_download_workers[tid] = dw
                dw.progress.connect(self._on_download_progress)
                dw.done.connect(self._on_download_done)
                dw.fail.connect(self._on_download_fail)
                dw.cancelled.connect(self._on_download_cancelled)
                dw.start()
                self._start_next_downloads()

            def on_err(msg: str, tid=next_task.task_id):
                self._active_download_resolve_ids.discard(tid)
                task = self._download_tasks.get(tid)
                if task:
                    task.status = "failed"
                    task.error = msg
                self._refresh_downloads_ui()
                self.set_status("Errore resolve download.")
                self._start_next_downloads()

            w.ok.connect(on_ok)
            w.err.connect(on_err)
            self._track_worker(w)
            w.start()

    def _on_download_progress(self, task_id: str, pct: int, downloaded: int, total: int):
        task = self._download_tasks.get(task_id)
        if not task:
            return
        task.downloaded = downloaded
        task.total = total if total > 0 else None
        task.progress = float(max(0, pct)) if pct >= 0 else task.progress
        task.status = "downloading"
        self._refresh_downloads_ui()

    def _on_download_done(self, task_id: str, out_path: str):
        task = self._download_tasks.get(task_id)
        self._active_download_workers.pop(task_id, None)
        if task:
            err = self._check_download_integrity(out_path)
            if err is not None:
                task.retry_count += 1
                if task.retry_count <= task.max_retries:
                    task.status = "queued"
                    task.error = f"Integrity fail, retry {task.retry_count}/{task.max_retries}: {err}"
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass
                    self._refresh_downloads_ui()
                    self.set_status(f"Integrity check fallita, retry: {task.anime_name} ep {task.episode}")
                    self._start_next_downloads()
                    return
                task.status = "failed"
                task.error = f"Integrity fail: {err}"
            else:
                task.status = "completed"
                task.progress = 100.0
                task.error = None
        self._refresh_downloads_ui()
        self.refresh_offline_library()
        if task and task.status == "completed":
            self.set_status(f"Download completato: {os.path.basename(out_path)}")
        else:
            self.set_status("Download fallito (integrity).")
        self._start_next_downloads()

    def _on_download_fail(self, task_id: str, msg: str):
        task = self._download_tasks.get(task_id)
        if task:
            task.status = "failed"
            task.error = msg
        self._active_download_workers.pop(task_id, None)
        self._refresh_downloads_ui()
        self.set_status("Download fallito.")
        self._start_next_downloads()

    def _on_download_cancelled(self, task_id: str):
        task = self._download_tasks.get(task_id)
        if task:
            task.status = "cancelled"
        self._active_download_workers.pop(task_id, None)
        self._refresh_downloads_ui()
        self.set_status("Download annullato.")
        self._start_next_downloads()

    def on_download_cancel_selected(self):
        item = self.downloads_list.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        task = self._download_tasks.get(task_id)
        if not task:
            return
        worker = self._active_download_workers.get(task_id)
        if worker is not None:
            worker.request_cancel()
            return
        if task.status in ("queued", "resolving"):
            task.status = "cancelled"
            self._active_download_resolve_ids.discard(task_id)
            self._refresh_downloads_ui()
            self._start_next_downloads()

    def on_download_clear_completed(self):
        keep: list[str] = []
        for task_id in self._download_order:
            t = self._download_tasks.get(task_id)
            if not t:
                continue
            if t.status == "completed":
                self._download_tasks.pop(task_id, None)
                continue
            keep.append(task_id)
        self._download_order = keep
        self._refresh_downloads_ui()

    def on_download_open_folder(self):
        path = self.download_dir
        debug_log(f"open-folder: requested path={path}")
        try:
            os.makedirs(path, exist_ok=True)
            debug_log("open-folder: ensured directory exists")
            system = platform.system()
            if system == "Linux":
                nautilus = shutil.which("nautilus")
                if not nautilus:
                    self.notify_err(
                        "Su Linux al momento e supportato solo Nautilus.\n"
                        "Installa Nautilus oppure apri manualmente il percorso:\n"
                        f"{path}"
                    )
                    return
                debug_log(f"open-folder: linux try {nautilus} {path}")
                detached = QProcess.startDetached(nautilus, [path])
                debug_log(f"open-folder: linux detached={'yes' if detached else 'no'} {nautilus} {path}")
                if not detached:
                    self.notify_err(
                        "Impossibile avviare Nautilus automaticamente.\n"
                        f"Percorso: {path}"
                    )
                return

            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            debug_log(f"open-folder: QDesktopServices.openUrl -> {opened}")
            if opened:
                return
            if system == "Windows":
                debug_log("open-folder: fallback os.startfile")
                os.startfile(path)  # type: ignore[attr-defined]
            elif system == "Darwin":
                debug_log("open-folder: fallback open")
                subprocess.run(["open", path], check=True)
            else:
                debug_log("open-folder: fallback xdg-open")
                subprocess.run(["xdg-open", path], check=True)
        except Exception as ex:
            debug_log(f"open-folder: error={ex}")
            self.notify_err(str(ex))
