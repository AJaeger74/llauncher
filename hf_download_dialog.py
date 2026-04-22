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
- Vollständige URL: "https://huggingface.co/owner/repo/blob/main/file.gguf" → Download sofort
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse
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
            if p in ("blob", "raw"):
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

class HfDownloadWorker(QThread):
    """Worker thread for downloading a file from Hugging Face."""

    size_changed = pyqtSignal(str, int)  # filename, current_bytes_on_disk
    finished_signal = pyqtSignal(bool, str)  # success, result_message

    def __init__(self, short_id: str, file_path: str, target_dir: str, file_name: str):
        super().__init__()
        self.short_id = short_id
        self.file_path = file_path
        self.target_dir = target_dir
        self.file_name = file_name

    def run(self):
        try:
            self._download()
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))

    def _download(self):
        """Execute the actual HTTP download with progress reporting."""
        import os

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
                    mode = "ab" if start_pos > 0 else "wb"
                    first_chunk = True

                    with open(partial_path, mode) as fh:
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
                            # Real on-disk size (includes resume offset + bytes
                            # written this session). Always authoritative.
                            current = os.path.getsize(partial_path)
                            self.size_changed.emit(
                                self.file_path,
                                os.path.getsize(partial_path),
                            )

                # --- Atomic rename (success) ----------------------------
                if partial_path.exists():
                    partial_path.replace(dst)
                self.finished_signal.emit(
                    True, gettext("msg_download_complete").format(path=str(dst))
                )
                return  # done, no more retries needed

            except Exception as exc:
                attempt += 1
                error_msg = str(exc)
                if attempt >= max_retries:
                    # Final failure – keep partial for manual inspection
                    self.finished_signal.emit(
                        False, gettext("msg_download_error").format(error=error_msg)
                    )
                    return

               # Not final – wait with exponential backoff and retry
                import time

                wait_time = min(2 ** attempt, 30)  # cap at 30s
                # Always read real on-disk size for accurate progress during wait
                current = os.path.getsize(partial_path) if partial_path.exists() else start_pos
                self.size_changed.emit(
                    self.file_path,
                    os.path.getsize(partial_path),
                )
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

        # --- Progress row ---
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
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
            return

        # Check if it's a full URL (contains blob or raw)
        if "blob" in text.lower() or "raw" in text.lower():
            # Full file URL – skip file list, go straight to download
            try:
                short_id, file_path = parse_hf_url(text)
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
        self.file_combo.clear()
        for f in gguf_files:
            size_str = human_size(f["size_bytes"])
            entry = f"{f['filename']} ({size_str})"
            self.file_combo.addItem(entry, f["filename"])

        self.file_combo.setEnabled(True)
        self.download_btn.setEnabled(False)

        # Auto-select the largest .gguf file
        self.file_combo.setCurrentIndex(0)

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
        self.progress_bar.setValue(0)
        
        # Ensure size label is visible from the start
        self.size_label.setVisible(True)

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

        # Check if file already exists — ask for overwrite
        if dst_path.exists():
            reply = QMessageBox.question(
                self,
                gettext("hf_dl_dialog_title"),
                gettext("msg_file_exists").format(path=str(dst_path)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                # User declined overwrite — close dialog
                self.reject()
                return

        # Compute partial path the same way the worker does, so we can show
        # the size of an existing partial file *before* the download starts.
        dst_path_full = Path(target_dir) / file_name
        partial_path = str(dst_path_full) + ".partial"

        # Start worker thread
        self.worker = HfDownloadWorker(short_id, file_path, target_dir, file_name)
        self.worker.size_changed.connect(self._on_size_changed)
        self.worker.finished_signal.connect(self._on_download_finished)
        self.worker.start()

        # If a partial file already exists on disk, show its size immediately.
        # Store it so we can protect it from being overwritten by the worker's
        # first signal (which may read start_pos=0 if fsync hasn't completed).
        if os.path.exists(partial_path):
            real_size = os.path.getsize(partial_path)
            self._initial_partial_size = real_size
            self._on_size_changed(file_name, real_size)
        else:
            self._initial_partial_size = None

        self.status_label.setText(gettext("msg_downloading"))

    def _on_size_changed(self, filename: str, current_bytes: int):
        """Update file size label from filesystem."""
        self.size_label.setVisible(True)
        
        # Protect initial partial size from being overwritten by worker's
        # first signal (which may read start_pos=0 if fsync hasn't completed).
        if (self._initial_partial_size is not None 
                and current_bytes == 0 
                and self.size_label.text() != "0 B"):
            return
        
        # Only update if value changed to prevent unnecessary UI repaints
        current_text = self.size_label.text()
        new_text = f"{human_size(current_bytes)}"
        if current_text != new_text:
            self.size_label.setText(new_text)

    def _on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        self.download_btn.setEnabled(True)
        # Ensure size label reflects final state before closing
        if success:
            # Force one last update to the size label from the UI thread
            if hasattr(self.worker, 'size_changed'):
                # Get the last emitted size if available, or rely on the label
                pass
            self.size_label.setVisible(True)
            QMessageBox.information(self, gettext("hf_dl_dialog_title"), message)
            # Close the dialog after user dismisses the success message
            self.reject()
        else:
            QMessageBox.critical(self, gettext("hf_dl_dialog_title"), message)
            # Leave dialog open on error so user can try again
        self.status_label.setText(message)
        self.progress_bar.setValue(100 if success else 0)

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
            """)

    def closeEvent(self, event):
        """Cancel any running download when dialog is closed."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()
