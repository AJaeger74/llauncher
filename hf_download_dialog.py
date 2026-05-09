#!/usr/bin/env python3
"""
hf_download_dialog – Qt6 Dialog zum Herunterladen von Modellen von Hugging Face Hub.

Zeigt ein Dialog-Fenster mit:
- Eingabefeld für Model-URL oder owner/repo
- QComboBox zur Dateiauswahl (wird aus HF Tree API geladen)
- QProgressBar für Download-Fortschritt
- Download läuft im Hintergrund (QThread)

Unterstützt:
- Kurze Form: "owner/repo" → listet Dateien auf, User wählt eine
|- Vollständige URL: "https://huggingface.co/owner/repo/blob/main/file.gguf" (oder …/resolve/main/…) → Download sofort
"""

import json
import math
import os
import re
import sys
import ctypes
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QProgressBar, QFileDialog, QMessageBox,
)

# Import i18n gettext function
try:
    from i18n import I18nManager
    gettext = I18nManager.get_instance().gettext
except ImportError:
    def gettext(key):
        return key

# ===================================================================
# Constants
# ===================================================================

HUB_BASE = "https://huggingface.co"
REQUEST_TIMEOUT = 30  # seconds for HTTP requests
CHUNK_SIZE = 1024 * 1024  # 1 MB download chunks

# ===================================================================
# Helpers – URL parsing (same logic as hf_download.py)
# ===================================================================

def parse_hf_url(raw: str):
    """
    Normalise a Hugging Face reference into (owner/repo, file_path | None).

    Returns:
        short_id  – always "owner/repo"
        file_path – the file inside the repo if a full URL was given, else None.
    """
    raw = raw.strip()

    # --- Full URL ---------------------------------------------------
    if raw.startswith("http"):
        parsed = urlparse(raw)
        parts = [p for p in parsed.path.split("/") if p]

        # Expected: owner / repo / blob / main / file  OR  owner / repo / raw / main / file
        idx_blob = None
        for i, p in enumerate(parts):
            if p in ("blob", "raw", "resolve"):
                idx_blob = i
                break

        if idx_blob is not None and len(parts) > idx_blob + 2:
            short_id = f"{parts[idx_blob - 2]}/{parts[idx_blob - 1]}"
            file_path = "/".join(parts[idx_blob + 2:])
            return short_id, file_path

        # Fallback: owner / repo only (no blob/raw in path)
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}", None

        raise ValueError(f"Unrecognised Hugging Face URL: {raw}")

    # --- Short form -------------------------------------------------
    short_id = raw
    if "/" not in short_id:
        raise ValueError(
            f"'{raw}' does not look like a valid owner/repo reference. "
            f"Use 'owner/repo' or a full URL."
        )

    return short_id, None

# ===================================================================
# Helpers – size formatting
# ===================================================================

def human_size(nbytes: int) -> str:
    """Format bytes as a human-readable string (e.g. 1.5 GiB)."""
    nbytes = abs(nbytes)
    if nbytes <= 0:
        return "0 B"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    exp = min(int(math.log(nbytes, 1024)), len(units) - 1)
    val = nbytes / (1024 ** exp)
    return f"{val:.2f} {units[exp]}"

# ===================================================================
# Repo listing via HF Tree API
# ===================================================================

def list_repo_files(short_id: str) -> list[dict]:
    """
    Return a list of file info dicts for the given repo using the Tree API.
    Uses only stdlib urllib.
    Only non-hidden files (no leading '.') are returned.
    """
    tree_url = f"{HUB_BASE}/api/models/{short_id}/tree/main"

    try:
        with urlopen(tree_url, timeout=REQUEST_TIMEOUT) as resp:
            tree_data = json.loads(resp.read().decode())
    except Exception as exc:
        return []

    result = []
    for item in tree_data:
        path = item.get("path", "")
        if not path.startswith("."):
            result.append({
                "filename": path,
                "size_bytes": item.get("size", 0),
            })

    # Sort by size descending – largest first (most likely to be what user wants)
    result.sort(key=lambda f: f["size_bytes"], reverse=True)
    return result

# ===================================================================
# Worker thread – handles downloads in the background
# ===================================================================


def _async_raise(tid, exc_type):
    """Raise an exception in a running Python thread via ctypes.
    This is the only reliable way to interrupt a blocking socket read."""
    if not tid:
        return
    try:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid),
            ctypes.py_object(exc_type)
        )
    except Exception:
        pass


class HfDownloadWorker(QThread):
    """Worker thread for downloading a file from Hugging Face."""

    size_changed = pyqtSignal(str, object)  # filename, current_bytes_on_disk (object to avoid 32-bit PyQt int truncation)
    progress_percent = pyqtSignal(int)  # 0-100
    finished_signal = pyqtSignal(bool, str)  # success, result_message

    def __init__(self, short_id: str, file_path: str, target_dir: str, file_name: str):
        super().__init__()
        self.short_id = short_id
        self.file_path = file_path
        self.target_dir = target_dir
        self.file_name = file_name
        self._cancelled = False

    def cancel(self) -> None:
        """Signal the worker to stop as soon as possible.
        This method can be called from any thread."""
        # 1. Set the cancellation flag – the download loop checks this each iteration.
        self._cancelled = True

        # 2. If a response object is still open, close its socket to unblock urlopen.
        if hasattr(self, "_current_response") and self._current_response:
            try:
                self._current_response.fp.close()
            except Exception:
                pass

        # 3. Raise KeyboardInterrupt in the worker's own thread to break out of
        #    any blocking read. The injected exception is caught by run().
        try:
            tid = self.threadId()
            if tid != -1:
                _async_raise(tid, KeyboardInterrupt)
        except Exception:
            pass

    def run(self):
        try:
            self._download()
        except KeyboardInterrupt:
            # Cancellation injected via ctypes – _download should have already
            # emitted the cancel signal before we got here. If it didn't
            # (e.g. exception arrived mid-connect before we checked _cancelled),
            # emit it now to be safe.
            if not getattr(self, "_signal_emitted", False):
                self.finished_signal.emit(False, gettext("msg_download_cancelled"))
            return
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))

    def _download(self):
        """Execute the actual HTTP download with progress reporting."""
        import os
        import sys

        url = f"{HUB_BASE}/{self.short_id}/resolve/main/{self.file_path}"
        target = Path(self.target_dir)
        target.mkdir(parents=True, exist_ok=True)

        dst = target / self.file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        partial_path = Path(str(dst) + ".partial")

        max_retries = 5
        attempt = 0

        while True:
            # --- Check for resume candidate -----------------------------
            start_pos = 0
            if partial_path.exists():
                start_pos = partial_path.stat().st_size

            # --- Open stream and download -------------------------------
            headers: dict[str, str] = {}
            if start_pos > 0:
                headers["Range"] = f"bytes={start_pos}-"
            

            current = start_pos  # always defined for progress reporting on retry

            try:
                req = Request(url, headers=headers)
                
                with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    self._current_response = resp  # For cancellation from closeEvent
                    status_code = resp.status
                    content_length = resp.headers.get('Content-Length')
                    
                    # Compute total file size for progress bar
                    # For fresh download: total = content_length
                    # For resumed (206): total = start_pos + remaining (content_length)
                    self._total_size = None
                    if content_length is not None:
                        cl = int(content_length)
                        if start_pos > 0 and status_code == 206:
                            self._total_size = start_pos + cl
                        elif start_pos == 0:
                            self._total_size = cl
                    
                    mode = "ab" if start_pos > 0 else "wb"
                    first_chunk = True
                    download_aborted = False

                    with open(partial_path, mode) as fh:
                        bytes_written_this_session = 0
                        while True:
                            chunk = resp.read(CHUNK_SIZE)
                            
                            if not chunk:
                                # Empty chunk = end of stream. Read real size
                                # from disk to handle any last partial writes.
                                current = os.path.getsize(partial_path)

                                if start_pos == 0 and first_chunk:
                                    # Fresh download – empty first read means
                                    # connection dropped before any data arrived.
                                    raise ConnectionError(
                                        gettext("msg_download_empty").format()
                                    )
                                # Resume case: empty first read after a valid
                                # Range header means "no more data to send".
                                # Server considers the file complete.
                                if current < start_pos:
                                    # Data was lost during download – retry
                                    raise ConnectionError(
                                        gettext("msg_download_incomplete").format()
                                    )

                                break  # genuine end of stream

                            first_chunk = False
                            fh.write(chunk)
                            fh.flush()
                            os.fsync(fh.fileno())
                            bytes_written_this_session += len(chunk)

                            # Check for cancellation every chunk
                            if self._cancelled:
                                download_aborted = True
                                break

                            # Read real on-disk size after fsync for accurate
                            # progress reporting.
                            sz = os.path.getsize(partial_path)
                            self.size_changed.emit(
                                self.file_path,
                                sz,
                            )

                            # Emit progress percentage if total is known
                            if self._total_size is not None and self._total_size > 0:
                                pct = min(100, int(sz * 100 // self._total_size))
                                self.progress_percent.emit(pct)

                # --- Handle completion or abortion --------------------------
                if download_aborted:
                    # User cancelled — keep partial for resume, emit cancel signal
                    self._signal_emitted = True
                    self.finished_signal.emit(False, gettext("msg_download_cancelled"))
                    return

                # Atomic rename (success)
                if partial_path.exists():
                    partial_path.replace(dst)

                self._signal_emitted = True
                self.finished_signal.emit(
                    True, gettext("msg_download_complete").format(path=str(dst))
                )
                return  # done, no more retries needed

            except Exception as exc:
                attempt += 1
                error_msg = str(exc)

                if attempt >= max_retries:
                    # Final failure – keep partial for manual inspection
                    self._signal_emitted = True
                    self.finished_signal.emit(
                        False, gettext("msg_download_error").format(error=error_msg)
                    )
                    return

                # Not final – wait with exponential backoff and retry
                import time

                wait_time = min(2 ** attempt, 30)  # cap at 30s
                # Read on-disk size for accurate progress during retry wait
                current = os.path.getsize(partial_path) if partial_path.exists() else start_pos
                self.size_changed.emit(
                    self.file_path,
                    current,
                )
                # Also emit progress percentage if total is known
                if self._total_size is not None and self._total_size > 0:
                    pct = min(100, int(current * 100 // self._total_size))
                    self.progress_percent.emit(pct)
                time.sleep(wait_time)

# ===================================================================
# Worker thread – fetches file listings in the background
# ===================================================================

class HfFilesWorker(QThread):
    """Worker thread that fetches repo file listings from HF Tree API."""

    files_ready = pyqtSignal(list)  # list of file dicts
    error_occurred = pyqtSignal(str)

    def __init__(self, short_id: str, request_id: int):
        super().__init__()
        self.short_id = short_id
        self.request_id = request_id

    def run(self):
        try:
            files = list_repo_files(self.short_id)
            self.files_ready.emit(files)
        except Exception as exc:
            self.error_occurred.emit(gettext("msg_repo_loading_failed").format(error=str(exc)))

# ===================================================================
# Dialog
# ===================================================================

class HfDownloadDialog(QDialog):
    """Dialog for downloading model files from Hugging Face Hub."""

    def __init__(self, parent=None, current_light_theme: bool = False):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.worker = None  # HfDownloadWorker instance
        self._file_list = []  # Current file list from HF API
        self._current_short_id = None
        self._request_counter = 0  # Monotonically increasing request ID

        self.setup_ui()
        self.apply_theme(current_light_theme)
        self.setWindowTitle(gettext("hf_dl_dialog_title"))
        self.resize(520, 380)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- URL / owner-repo input row ---
        top_layout = QHBoxLayout()
        url_label = QLabel(gettext("lbl_hf_url_or_repo"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(gettext("placeholder_hf_url_repo"))
        self.url_edit.setMinimumHeight(32)

        # Debounce timer: wait 500ms after user stops typing before firing request
        self._url_debounce_timer = None

        # Connect textChanged — only load files when user finishes typing
        self.url_edit.textChanged.connect(self._on_url_changed)

        top_layout.addWidget(url_label)
        top_layout.addWidget(self.url_edit, stretch=1)
        layout.addLayout(top_layout)

        # --- File selection row ---
        file_label = QLabel(gettext("lbl_hf_files"))
        self.file_combo = QComboBox()
        self.file_combo.setMinimumHeight(32)
        self.file_combo.setEnabled(False)  # Disabled until files are loaded
        self.file_combo.setEditable(False)  # Proper dropdown, not editable
        self.file_combo.setMaxVisibleItems(20)

        layout.addWidget(file_label)
        layout.addWidget(self.file_combo)

        self.file_combo.currentIndexChanged.connect(self._on_file_selected)

        # --- Target directory display ---
        target_dir_label = QLabel(gettext("lbl_target_dir"))
        self.target_dir_label = QLabel("")
        self.target_dir_label.setStyleSheet("color: #4fc3f7; font-size: 9pt;")
        self.target_dir_label.setWordWrap(True)
        layout.addWidget(target_dir_label)
        layout.addWidget(self.target_dir_label)

        # --- Progress row ---
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setVisible(False)  # Hidden for now
        
        # File size label - shows real on-disk size
        self.size_label = QLabel("0 B")
        self.size_label.setStyleSheet("color: #4fc3f7; font-size: 10pt; font-weight: bold;")
        self.size_label.setVisible(False)

        progress_layout.addWidget(self.progress_bar, stretch=1)
        progress_layout.addWidget(self.size_label)
        layout.addLayout(progress_layout)

        # --- Status label ---
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Buttons row ---
        btn_layout = QHBoxLayout()
        self.download_btn = QPushButton(gettext("btn_download_file"))
        self.download_btn.setEnabled(False)  # Disabled until a file is selected
        self.download_btn.setMinimumHeight(36)
        self.download_btn.setStyleSheet("""
            QPushButton {
                font-size: 13px;
                padding: 6px 16px;
            }
            QPushButton:disabled {
                color: #888;
            }
        """)
        self.download_btn.clicked.connect(self._start_download)

        close_btn = QPushButton(gettext("btn_cancel"))
        close_btn.setMinimumHeight(36)
        close_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.download_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # --- Loading indicator ---
        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #999; font-style: italic; font-size: 9pt;")
        self.loading_label.setVisible(False)
        layout.addWidget(self.loading_label)

    def _on_url_changed(self):
        """Called on every keystroke — sets up a debounced timer."""
        text = self.url_edit.text().strip()

        # Clear previous debounce timer
        if hasattr(self, '_url_debounce_timer') and self._url_debounce_timer is not None:
            self._url_debounce_timer.stop()

        if not text:
            # Nothing typed — clear everything
            self.file_combo.clear()
            self.file_combo.setEnabled(False)
            self.download_btn.setEnabled(False)
            self.status_label.setText("")
            self.loading_label.setVisible(False)
            self.target_dir_label.setText("")
            return

        # Check if it's a full file URL (blob, raw, or resolve)
        if "blob" in text.lower() or "raw" in text.lower() or "resolve" in text.lower():
            # Full file URL – skip file list, go straight to download
            try:
                short_id, file_path = parse_hf_url(text)
                # Strip query string (e.g. ?download=true) from displayed URL
                parsed_url = urlparse(text)
                clean_url = urlunparse((
                    parsed_url.scheme,
                    parsed_url.netloc,
                    parsed_url.path,
                    "", "", "",
                ))
                self.url_edit.setText(clean_url)
                # Show target directory for full URL too
                self._update_target_dir_label(short_id)
                # Pre-populate combo so user can see the resolved filename
                self.file_combo.clear()
                self.file_combo.addItem(file_path, file_path)
                self.file_combo.setEnabled(False)
                self.download_btn.setEnabled(True)
                self.status_label.setText(f"{short_id}/{file_path}")
            except ValueError as exc:
                self.status_label.setText(str(exc))
                self.download_btn.setEnabled(False)
        else:
            # Short form (owner/repo) — debounce before firing the API call
            try:
                short_id, _ = parse_hf_url(text)
                self._current_short_id = short_id
                self.status_label.setText("")
                self.loading_label.setVisible(True)

                # Show target directory early, so user knows where the file will go
                self._update_target_dir_label(short_id)

                # Start a one-shot timer that fires after 500ms of inactivity
                from PyQt6.QtCore import QTimer
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(
                    lambda sid=short_id: self._load_files(sid)
                )
                timer.start(500)
                self._url_debounce_timer = timer
            except ValueError as exc:
                self.status_label.setText(str(exc))
                self.file_combo.setEnabled(False)
                self.download_btn.setEnabled(False)

    def _load_files(self, short_id: str):
        """Load file list from HF Tree API using a background worker thread."""
        # Only cancel the previous request if it's still running (don't kill mid-flight)
        if hasattr(self, '_files_worker') and self._files_worker is not None:
            if self._files_worker.isRunning():
                self._files_worker.terminate()
                self._files_worker.wait(1000)

        self.loading_label.setVisible(True)
        self.status_label.setText("")

        self.file_combo.clear()
        self._file_list = []

        self._request_counter += 1
        current_request_id = self._request_counter

        self._files_worker = HfFilesWorker(short_id, current_request_id)
        self._files_worker.files_ready.connect(self._on_files_loaded)
        self._files_worker.error_occurred.connect(self._on_files_error)
        self._files_worker.start()

    def _on_files_loaded(self, files: list[dict]):
        """Handle file list received from the worker thread."""
        self.loading_label.setVisible(False)

        # Ignore stale responses from older workers
        if not hasattr(self, '_files_worker') or self._files_worker.request_id != self._request_counter:
            return

        # Filter to only .gguf files
        gguf_files = [f for f in files if f["filename"].lower().endswith(".gguf")]
        if not gguf_files:
            self.status_label.setText(gettext("msg_no_files_found"))
            self.file_combo.setEnabled(False)
            self.download_btn.setEnabled(False)
            return

        # Populate combo box with "filename (size)" entries — sorted by size desc
        self._file_list = gguf_files
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for f in gguf_files:
            size_str = human_size(f["size_bytes"])
            entry = f"{f['filename']} ({size_str})"
            self.file_combo.addItem(entry, f["filename"])
        self.file_combo.blockSignals(False)

        self.file_combo.setEnabled(True)
        self.download_btn.setEnabled(False)

        # Auto-select the largest .gguf file
        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(0)
        self.file_combo.blockSignals(False)
        self._on_file_selected()

    def _on_files_error(self, error_msg: str):
        """Handle error from the file fetch worker."""
        self.loading_label.setVisible(False)

        # Ignore stale responses from older workers
        if not hasattr(self, '_files_worker') or self._files_worker.request_id != self._request_counter:
            return

        self.status_label.setText(error_msg)
        self.file_combo.setEnabled(False)
        self.download_btn.setEnabled(False)

    def _on_file_selected(self):
        """Enable download button when a file is selected."""
        self.download_btn.setEnabled(True)

    def _update_target_dir_label(self, short_id: str):
        """Compute and display the target directory for a given short_id.

        Called whenever the short_id becomes known (URL entered, repo loaded).
        Shows the absolute path where the file will be downloaded.
        """
        try:
            model_dir = self._get_model_directory()
            repo_subdir = short_id.replace("/", "_")
            target_dir = os.path.join(model_dir, repo_subdir)
            self.target_dir_label.setText(target_dir)
        except Exception:
            self.target_dir_label.setText("")

    def _get_model_directory(self) -> str:
        """Get the model directory from config or default to ~/.models."""
        try:
            config_path = Path.home() / ".llauncher" / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
                return config.get("model_directory", str(Path.home() / "models"))
        except Exception:
            pass

        # Fallback to default
        return str(Path.home() / "models")

    def _start_download(self):
        """Start the background download."""
        
        self.download_btn.setEnabled(False)
        # Reset progress bar at start (will be updated by worker signals)
        self.progress_bar.setValue(0)
        
        # Show download in progress in UI label
        self.status_label.setText(gettext("msg_downloading"))

        text = self.url_edit.text().strip()
        if not text:
            QMessageBox.warning(self, gettext("hf_dl_dialog_title"),
                                gettext("msg_no_url_entered"))
            return

        try:
            short_id, file_path = parse_hf_url(text)
        except ValueError as exc:
            QMessageBox.warning(self, gettext("hf_dl_dialog_title"), str(exc))
            return

        # If no specific file was in URL (short form), get from combo box
        if not file_path and self.file_combo.count() > 0:
            file_path = str(self.file_combo.currentData()) or ""
            if not file_path:
                QMessageBox.warning(self, gettext("hf_dl_dialog_title"),
                                    gettext("msg_no_files_found"))
                return

        if not file_path:
            QMessageBox.warning(self, gettext("hf_dl_dialog_title"),
                                gettext("msg_no_files_found"))
            return

        model_dir = self._get_model_directory()
        repo_subdir = short_id.replace("/", "_")

        # Avoid double-adding subdir if model_dir already ends with it
        model_dir_stripped = model_dir.rstrip(os.sep)
        if model_dir_stripped.endswith(os.sep + repo_subdir) or model_dir_stripped == repo_subdir:
            target_dir = model_dir_stripped
        else:
            target_dir = os.path.join(model_dir, repo_subdir)

        file_name = os.path.basename(file_path)  # Extract just the filename
        dst_path = Path(target_dir) / file_name

        # Show target directory in the dialog before download starts
        self.target_dir_label.setText(target_dir)

        # Check if file already exists — ask for overwrite
        if dst_path.exists():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(gettext("hf_dl_dialog_title"))
            msg_box.setText(gettext("msg_file_exists").format(path=str(dst_path)))
            yes_btn = msg_box.addButton(
                gettext("msg_yes"), QMessageBox.ButtonRole.YesRole
            )
            no_btn = msg_box.addButton(
                gettext("msg_no"), QMessageBox.ButtonRole.NoRole
            )
            msg_box.setDefaultButton(no_btn)
            msg_box.exec()
            if msg_box.clickedButton() != yes_btn:
                # User declined overwrite — close dialog
                self.reject()
                return

        # Compute partial path the same way the worker does, so we can show
        # the size of an existing partial file *before* the download starts.
        dst_path_full = Path(target_dir) / file_name
        partial_path_str = str(dst_path_full) + ".partial"
        if os.path.exists(partial_path_str):
            real_size = os.path.getsize(partial_path_str)
            self._on_size_changed(file_name, real_size)

        self.status_label.setText(gettext("msg_downloading"))

        # --- Create and start the worker ----------------------------------
        
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
        
        self.worker = HfDownloadWorker(short_id, file_path, target_dir, file_name)
        self.worker.size_changed.connect(self._on_size_changed)
        self.worker.progress_percent.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._on_download_finished)
        # Show progress bar during download
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.worker.start()

    def _on_size_changed(self, filename: str, current_bytes: int):
        """Update file size label from filesystem."""
        self.size_label.setVisible(True)

        # Ignore zero — means worker hasn't written anything yet.
        if current_bytes <= 0:
            return

        # Debug: log raw value before processing

        # Simple approach: display the raw on-disk size directly.
        # The worker already calls os.path.getsize() which is the source of truth.
        new_display = human_size(current_bytes)
        self.size_label.setText(new_display)

    def _on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        # Set final progress bar state
        self.progress_bar.setValue(100 if success else 0)
        # Ensure size label is visible
        self.size_label.setVisible(True)
        self.download_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, gettext("hf_dl_dialog_title"), message)
            # Close the dialog after user dismisses the success message
            self.reject()
        else:
            QMessageBox.critical(self, gettext("hf_dl_dialog_title"), message)
            # Leave dialog open on error so user can try again
        self.status_label.setText(message)

    def apply_theme(self, use_light: bool):
        """Apply light or dark theme to the dialog."""
        if use_light:
            self.setStyleSheet("""
                QDialog {
                    background-color: #f5f5f5;
                }
                QLabel {
                    color: #333;
                }
                QLineEdit, QComboBox, QProgressBar {
                    background-color: #fff;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    padding: 4px;
                }
                QProgressBar::chunk {
                    background-color: #0078d7;
                    border-radius: 2px;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #bbb;
                    border-radius: 3px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:disabled {
                    color: #999;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #1e1e1e;
                }
                QLabel {
                    color: #ddd;
                }
                QLineEdit, QComboBox, QProgressBar {
                    background-color: #2a2a2a;
                    border: 1px solid #444;
                    border-radius: 3px;
                    padding: 4px;
                    color: #ddd;
                }
                QProgressBar::chunk {
                    background-color: #0078d7;
                    border-radius: 2px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QPushButton {
                    background-color: #3a3a3a;
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 6px 12px;
                    color: #ddd;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:disabled {
                    color: #777;
                }
            """
)

    def reject(self):
        """Called when user clicks Cancel or closes the dialog via X button.
        Cancels any running download before proceeding."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)  # graceful termination window
        super().reject()

    def closeEvent(self, event):
        """Cancel any running download when dialog is closed (e.g. via X button)."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)  # give more time for cleanup via closeEvent path
        event.accept()
