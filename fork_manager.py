#!/usr/bin/env python3
"""
fork_manager – Dialog for cloning a llama.cpp repository fork.

Opens a dialog where the user selects a target directory and enters
a repo URL. On confirmation, runs `git clone <url> <target_dir>` and
shows a result message.
"""

import subprocess
from pathlib import Path
from PyQt6.QtCore import Qt
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


class ForkManagerDialog(QDialog):
    """Dialog for cloning a llama.cpp fork."""

    def __init__(self, parent=None, current_light_theme: bool = False):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.target_dir = None
        self.repo_url = None
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
        cancel_btn = QPushButton(gettext("btn_cancel"))

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

    def _clone_repo(self):
        """Validate inputs and run git clone."""
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

        # Run git clone
        try:
            result = subprocess.run(
                ["git", "clone", url, str(target_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                QMessageBox.information(
                    self,
                    gettext("fork_result_title"),
                    gettext("fork_result_success").format(repo=repo_name, path=str(target_path))
                )
                self.accept()
            else:
                error_msg = result.stderr.strip() if result.stderr else "unknown error"
                QMessageBox.critical(
                    self,
                    gettext("fork_result_title"),
                    gettext("fork_result_error").format(error=error_msg)
                )
        except subprocess.TimeoutExpired:
            QMessageBox.critical(
                self,
                gettext("fork_result_title"),
                gettext("fork_result_error").format(error="Clone timed out (5 min)")
            )
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                gettext("fork_result_title"),
                gettext("fork_result_error").format(error="'git' command not found")
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                gettext("fork_result_title"),
                gettext("fork_result_error").format(error=str(e))
            )
