#!/usr/bin/env python3
"""
fork_manager – Dialog for cloning a llama.cpp repository fork.

Opens a dialog where the user selects a target directory and enters
a repo URL. On confirmation, runs `git clone <url> <target_dir>` in
a background thread (non-blocking) and dumps all output to the debug
area when complete.
"""

import subprocess
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox
)

# Import i18n gettext function
try:
    from i18n import I18nManager
    gettext = I18nManager.get_instance().gettext
except ImportError:
    def gettext(key):
        return key


DARK_THEME = """
QDialog { background-color: #2d2d2d; }
QLabel { color: #cccccc; }
QPushButton { background-color: #0078d7; color: white; padding: 6px 12px; border-radius: 3px; }
QPushButton:hover { background-color: #006cc1; }
QLineEdit { padding: 5px; border-radius: 3px; background-color: #444; color: white; border: 1px solid #555; }
"""

LIGHT_THEME = """
QDialog { background-color: #ffffff; }
QLabel { color: #333333; }
QPushButton { background-color: #0078d7; color: white; padding: 6px 12px; border-radius: 3px; }
QPushButton:hover { background-color: #006cc1; }
QLineEdit { padding: 5px; border-radius: 3px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
"""


class GitCloneWorker(QThread):
    """Background thread that runs git clone and returns all output."""

    output_signal = pyqtSignal(str)   # "running..." status line
    finished_signal = pyqtSignal(int, str)  # returncode, combined stdout+stderr

    def __init__(self, url: str, target_path: str):
        super().__init__()
        self.url = url
        self.target_path = target_path

    def run(self):
        """Run git clone in background (non-blocking)."""
        try:
            self.output_signal.emit("running")

            result = subprocess.run(
                ["git", "clone", self.url, self.target_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300  # 5 minute timeout
            )

            combined = (result.stdout + "\n" + result.stderr).strip()
            self.finished_signal.emit(result.returncode, combined)

        except subprocess.TimeoutExpired:
            self.finished_signal.emit(-1, "Clone timed out (5 min)")
        except FileNotFoundError:
            self.finished_signal.emit(-1, "'git' command not found")
        except Exception as e:
            self.finished_signal.emit(-1, str(e))


class ForkManagerDialog(QDialog):
    """Dialog for cloning a llama.cpp fork."""

    def __init__(self, parent=None, current_light_theme: bool = False):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.target_dir = None
        self.repo_url = None
        self.clone_thread = None  # GitCloneWorker instance
        self.setup_ui()
        self.apply_theme(current_light_theme)
        self.setWindowTitle(gettext("fork_dialog_title"))

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Target Directory row ---
        dir_layout = QHBoxLayout()
        dir_label = QLabel(gettext("lbl_fork_directory"))
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setReadOnly(True)
        self.dir_path_edit.setPlaceholderText(gettext("lbl_no_file_selected"))
        self.dir_path_edit.setMinimumWidth(300)
        browse_dir_btn = QPushButton(gettext("btn_browse_dir"))

        browse_dir_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_path_edit, 1)
        dir_layout.addWidget(browse_dir_btn)
        layout.addLayout(dir_layout)

        # --- Helper text ---
        help_label = QLabel(gettext("lbl_fork_clonedir"))
        help_label.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(help_label)

        # --- Repository URL row ---
        url_layout = QHBoxLayout()
        url_label = QLabel(gettext("lbl_fork_url"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://github.com/...")
        self.url_edit.setMinimumWidth(300)
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_edit, 1)
        layout.addLayout(url_layout)

        # Stretch to push buttons down
        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        clone_btn = QPushButton(gettext("btn_clone_repo"))
        clone_btn.setObjectName("btn_clone_repo")
        cancel_btn = QPushButton(gettext("btn_cancel"))
        cancel_btn.setObjectName("btn_cancel")

        clone_btn.clicked.connect(self._clone_repo)
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(clone_btn)
        btn_layout.setSpacing(10)

        layout.addLayout(btn_layout)

    def apply_theme(self, use_light: bool):
        """Apply theme to dialog."""
        theme = LIGHT_THEME if use_light else DARK_THEME
        self.setStyleSheet(theme)

    def _browse_directory(self):
        """Open directory selection dialog."""
        path = QFileDialog.getExistingDirectory(
            self,
            gettext("btn_browse_dir"),
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            self.target_dir = Path(path)
            self.dir_path_edit.setText(str(self.target_dir))

    def _get_debug_text(self):
        """Lazily get the parent window's debug text widget."""
        if self.parent() and hasattr(self.parent(), 'debug_text'):
            return self.parent().debug_text
        return None

    def _dump_to_debug(self, message: str):
        """Write a line to the main window's debug output area."""
        debug = self._get_debug_text()
        if debug:
            debug.append(message)

    def _clone_repo(self):
        """Validate inputs and start git clone in background thread."""
        # Prevent double-clicks
        clone_btn = self.findChild(QPushButton, "btn_clone_repo")

        # Validate directory
        if not self.target_dir or not self.target_dir.exists():
            QMessageBox.warning(
                self,
                gettext("fork_result_title"),
                gettext("msg_no_directory_selected")
            )
            return

        # Validate URL
        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(
                self,
                gettext("fork_result_title"),
                gettext("msg_no_url_entered")
            )
            return

        self.repo_url = url

        # Extract repo name from URL for display
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target_path = self.target_dir / repo_name

        # Confirm before cloning
        reply = QMessageBox.question(
            self,
            gettext("fork_result_title"),
            f"Clone '{url}'\ninto:\n{target_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Log clone start to debug area
        self._dump_to_debug(f"[FORK] git clone {url} {target_path}")

        # Disable clone button during operation
        clone_btn.setEnabled(False)
        try:
            from i18n import I18nManager
            loading_text = I18nManager.get_instance().gettext("lbl_loading")
        except Exception:
            loading_text = "Loading..."
        clone_btn.setText(loading_text)

        # Create and configure the worker thread
        self.clone_thread = GitCloneWorker(url, str(target_path))
        self.clone_thread.finished_signal.connect(self._on_clone_finished)
        self.clone_thread.start()

    def _on_clone_finished(self, returncode: int, combined_output: str):
        """Handle clone completion."""
        # Re-enable the button
        clone_btn = self.findChild(QPushButton, "btn_clone_repo")
        if clone_btn:
            clone_btn.setEnabled(True)
            clone_btn.setText(gettext("btn_clone_repo"))

        # Extract repo name from URL for display
        repo_name = self.repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target_path = self.target_dir / repo_name

        if returncode == 0:
            # Dump git output to debug area
            if combined_output.strip():
                self._dump_to_debug(f"[FORK] {combined_output}")
            else:
                self._dump_to_debug("[FORK] (no output)")
            self._dump_to_debug(
                f"[FORK] ✓ Cloned '{repo_name}' → {target_path}"
            )
            QMessageBox.information(
                self,
                gettext("fork_result_title"),
                gettext("fork_result_success").format(repo=repo_name, path=str(target_path))
            )
            self.accept()
        else:
            error_msg = combined_output.strip() if combined_output else "unknown error"
            self._dump_to_debug(f"[FORK] ✗ Clone failed (exit {returncode})")
            self._dump_to_debug(f"[FORK] {error_msg}")
            QMessageBox.critical(
                self,
                gettext("fork_result_title"),
                gettext("fork_result_error").format(error=error_msg)
            )
