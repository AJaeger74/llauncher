#!/usr/bin/env python3
"""
fork_manager - Dialog for cloning, building, and switching llama.cpp repository forks.

Opens a dialog where the user selects a target directory and enters
a repo URL. On confirmation, runs `git clone <url> <target_dir>` in
a background thread, then optionally builds the fork, then optionally
switches the main app to use this fork's binary.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QComboBox,
    QTextEdit, QGroupBox
)

try:
    from i18n import I18nManager
    gettext = I18nManager.get_instance().gettext
except ImportError:
    def gettext(key):
        return key


LLAMA_JSON_PATH = Path.home() / ".llauncher" / "llama.json"


def _load_fork_entries():
    """Load fork entries from ~/.llauncher/llama.json.

    Returns dict: {fork_name: {"repo": str, "branch": str, "build": str}} or {} on failure.
    Handles malformed JSON (backslash line-continuations in build fields).
    """
    try:
        with open(LLAMA_JSON_PATH, "r") as f:
            raw = f.read()
        import re as _re
        raw = _re.sub(r"\s*\\\s*\n", r" ", raw)
        raw = raw.replace("\n", " ")
        data = json.loads(raw)
        return data
    except Exception:
        return {}


def _save_fork_entry(fork_name, url, branch, build_command):
    """Add or update a fork entry in ~/.llauncher/llama.json."""
    try:
        entries = _load_fork_entries()
        entries[fork_name] = {
            "repo": url,
            "branch": branch,
            "build": build_command,
        }
        with open(LLAMA_JSON_PATH, "w") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass  # Silent fail, build will still run


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

    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)

    def __init__(self, url: str, target_path: str, branch: str = ""):
        super().__init__()
        self.url = url
        self.target_path = target_path
        self.branch = branch

    def run(self):
        try:
            self.output_signal.emit("running")
            cmd = ["git", "clone"]
            if self.branch:
                cmd.extend(["-b", self.branch])
            cmd.extend([self.url, self.target_path])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300
            )

            combined = (result.stdout + "\n" + result.stderr).strip()
            self.finished_signal.emit(result.returncode, combined)
        except subprocess.TimeoutExpired:
            self.finished_signal.emit(-1, "Clone timed out (5 min)")
        except FileNotFoundError:
            self.finished_signal.emit(-1, "'git' command not found")
        except Exception as e:
            self.finished_signal.emit(-1, str(e))


class GitPullWorker(QThread):
    """Background thread that runs git pull and streams output."""

    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)
    cancelled_signal = pyqtSignal()

    def __init__(self, workdir: str):
        super().__init__()
        self.workdir = workdir
        self._process = None
        self._cancelled = False

    def run(self):
        try:
            self._process = subprocess.Popen(
                ["git", "pull"],
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            while self._process.poll() is None:
                if self._cancelled:
                    self._process.terminate()
                    self.cancelled_signal.emit()
                    break
                line = self._process.stdout.readline()
                if line:
                    self.output_signal.emit(line.strip())
            returncode = self._process.wait() if self._process else 1
            self.finished_signal.emit(returncode)
        except Exception as e:
            self.finished_signal.emit(-1)

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.terminate()


class BuildWorker(QThread):
    """Background thread that runs a build command and streams output."""

    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)
    cancelled_signal = pyqtSignal()

    def __init__(self, command: str, workdir: str):
        super().__init__()
        self.command = command
        self.workdir = workdir
        self._process = None
        self._cancelled = False

    def run(self):
        import sys
        try:
            print(f"[FORK-DEBUG] BUILD_WORKER: pwd={self.workdir}", file=sys.stderr)
            print(f"[FORK-DEBUG] BUILD_WORKER: command={self.command}", file=sys.stderr)
            self._process = subprocess.Popen(
                ["bash", "-c", self.command],
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            while self._process.poll() is None:
                if self._cancelled:
                    self._process.terminate()
                    self.cancelled_signal.emit()
                    break
                line = self._process.stdout.readline()
                if line:
                    self.output_signal.emit(line.strip())
            returncode = self._process.wait() if self._process else 1
            sys.stderr.write(f"[FORK] BUILD_WORKER: process exited with returncode={returncode}\n")
            sys.stderr.flush()
            if returncode == 0:
                summary = gettext("msg_build_complete")
            else:
                summary = gettext("msg_build_failed").format(code=returncode)
            sys.stderr.write(f"[FORK] BUILD_WORKER: emitting finished_signal({returncode}, {summary!r})\n")
            sys.stderr.flush()
            self.finished_signal.emit(returncode, summary)
        except Exception as e:
            sys.stderr.write(f"[FORK] BUILD_WORKER: exception {e!r}\n")
            sys.stderr.flush()
            self.finished_signal.emit(-1, str(e))

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.terminate()


def _is_git_repo(directory: str) -> bool:
    """Check if a directory is a git repository."""
    try:
        subprocess.run(
            ["git", "-C", directory, "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


def _extract_fork_name(url: str, entries: dict) -> str:
    """Look up fork name from llama.json entries using the URL.

    Returns the fork name (dict key) or the repo name from the URL.
    """
    clean_url = url.rstrip("/").replace(".git", "")
    for name, entry in entries.items():
        entry_url = entry.get("repo", "").rstrip("/").replace(".git", "")
        if clean_url == entry_url:
            return name
    # Fallback: extract from URL
    repo_name = clean_url.split("/")[-1]
    return repo_name


def _ask_question(parent, title, msg):
    """Ask a Yes/No question with translated button labels."""
    d = QMessageBox(parent)
    d.setWindowTitle(title)
    d.setText(msg)
    yes_btn = d.addButton(gettext("msg_yes"), QMessageBox.ButtonRole.YesRole)
    d.addButton(gettext("msg_no"), QMessageBox.ButtonRole.NoRole)
    d.exec()
    return d.clickedButton() == yes_btn


class ForkManagerDialog(QDialog):
    """Dialog for cloning, building, and switching llama.cpp forks."""

    def __init__(self, parent=None, current_light_theme: bool = False):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.target_dir = None
        self.repo_url = None
        self.clone_thread = None
        self.build_thread = None
        self._pull_thread = None      # keep reference so it survives _do_pull

        # State carried through the workflow
        self._fork_dir = None         # full path to cloned fork directory
        self._fork_name = ""          # from llama.json key
        self._build_command = ""      # the build command string
        self._build_cancelled = False

        self.setup_ui()
        self.apply_theme(current_light_theme)
        self.setWindowTitle(gettext("fork_dialog_title"))
        self.setMinimumHeight(520)

    # ---- UI Setup ----

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

        # --- Fork name label (read from llama.json) ---
        self.fork_name_label = QLabel(gettext("lbl_fork_name") + " --")
        self.fork_name_label.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(self.fork_name_label)

        # --- Helper text ---
        help_label = QLabel(gettext("lbl_fork_clonedir"))
        help_label.setStyleSheet("color: #888888; font-size: 9pt;")
        layout.addWidget(help_label)

        # --- Repository URL row ---
        url_layout = QHBoxLayout()
        url_label = QLabel(gettext("lbl_fork_url"))
        self.url_combo = QComboBox()
        self.url_combo.setEditable(True)
        self.url_combo.setPlaceholderText("https://github.com/...")
        self.url_combo.setMinimumWidth(300)
        for entry in _load_fork_entries().values():
            self.url_combo.addItem(entry.get("repo", ""), entry.get("repo", ""))
        self.url_combo.setEditText("")
        self.url_combo.currentTextChanged.connect(self._on_url_changed)
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_combo, 1)
        layout.addLayout(url_layout)

        # --- Branch row (optional) ---
        branch_layout = QHBoxLayout()
        branch_label = QLabel(gettext("lbl_fork_branch"))
        self.branch_edit = QLineEdit()
        self.branch_edit.setPlaceholderText(gettext("placeholder_branch_default"))
        self.branch_edit.setMinimumWidth(300)
        branch_hint = QLabel(gettext("lbl_fork_branch_hint"))
        branch_hint.setStyleSheet("color: #888888; font-size: 9pt;")
        branch_layout.addWidget(branch_label)
        branch_layout.addWidget(self.branch_edit, 1)
        branch_layout.addWidget(branch_hint)
        layout.addLayout(branch_layout)

        layout.addStretch()

        # --- Clone buttons ---
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

        # --- Build section (hidden initially) ---
        self._build_group = QGroupBox()
        build_layout = QVBoxLayout(self._build_group)
        build_layout.setSpacing(8)

        # Build output (QTextEdit for multi-line)
        self.build_text = QTextEdit()
        self.build_text.setReadOnly(True)
        mono_font = self.build_text.font()
        mono_font.setFamily("Monospace")
        mono_font.setPointSize(9)
        self.build_text.setFont(mono_font)
        self.build_text.setMaximumHeight(200)
        self.build_text.setPlaceholderText(gettext("msg_build_running"))
        build_layout.addWidget(self.build_text)

        # Build status label
        self.build_status_label = QLabel("")
        self.build_status_label.setStyleSheet("color: #888888; font-size: 9pt;")
        build_layout.addWidget(self.build_status_label)

        # Build buttons
        build_btn_layout = QHBoxLayout()
        self.build_btn = QPushButton(gettext("btn_build"))
        self.build_cancel_btn = QPushButton(gettext("btn_cancel_build"))
        self.build_cancel_btn.setEnabled(False)
        self.build_btn.clicked.connect(self._start_build)
        self.build_cancel_btn.clicked.connect(self._cancel_build)
        build_btn_layout.addWidget(self.build_cancel_btn)
        build_btn_layout.addWidget(self.build_btn)
        build_btn_layout.setSpacing(10)
        build_layout.addLayout(build_btn_layout)

        build_layout.addStretch()
        self._build_group.setVisible(False)
        layout.addWidget(self._build_group)

        layout.addStretch()

    def apply_theme(self, use_light: bool):
        self.setStyleSheet(LIGHT_THEME if use_light else DARK_THEME)

    # ---- Directory/URL interactions ----

    def _browse_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, gettext("btn_browse_dir"),
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            self.target_dir = Path(path)
            self.dir_path_edit.setText(str(self.target_dir))
            self._update_fork_name_label()

    def _on_url_changed(self, text: str):
        """When user selects or types a URL, update fork name label."""
        self._update_fork_name_label()

    def _update_fork_name_label(self):
        """Look up the fork name from llama.json based on selected URL."""
        url = self.url_combo.currentText().strip()
        if not url:
            self.fork_name_label.setText(gettext("lbl_fork_name") + " --")
            return

        entries = _load_fork_entries()
        fork_name = _extract_fork_name(url, entries)
        if fork_name and fork_name != "":
            self.fork_name_label.setText(f"{gettext('lbl_fork_name')} {fork_name}")
        else:
            self.fork_name_label.setText(f"{gettext('lbl_fork_name')} --")

    def _get_debug_text(self):
        if self.parent() and hasattr(self.parent(), 'debug_text'):
            return self.parent().debug_text
        return None

    def _dump_to_debug(self, message: str):
        debug = self._get_debug_text()
        if debug:
            debug.append(message)

    # ---- Clone logic ----

    def _clone_repo(self):
        if not self.target_dir or not self.target_dir.exists():
            QMessageBox.warning(self, gettext("fork_result_title"),
                                gettext("msg_no_directory_selected"))
            return

        url = self.url_combo.currentText().strip()
        if not url:
            QMessageBox.warning(self, gettext("fork_result_title"),
                                gettext("msg_no_url_entered"))
            return

        self.repo_url = url
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        self._fork_dir = str(self.target_dir / repo_name)
        branch = self.branch_edit.text().strip()

        clone_btn = self.findChild(QPushButton, "btn_clone_repo")
        if clone_btn:
            clone_btn.setEnabled(False)
            clone_btn.setText(gettext("lbl_loading"))

        if os.path.isdir(self._fork_dir):
            # Check if it's a git repo
            if _is_git_repo(self._fork_dir):
                reply = _ask_question(
                    self, gettext("fork_result_title"),
                    gettext("msg_dir_exists_git")
                )
                if reply:
                    self._do_pull()
                else:
                    # Delete and re-clone
                    try:
                        shutil.rmtree(self._fork_dir)
                        self._do_clone(url, self._fork_dir, branch)
                    except Exception as e:
                        QMessageBox.critical(self, gettext("fork_result_title"),
                                             str(e))
                # Cancel = skip
            else:
                reply = _ask_question(
                    self, gettext("fork_result_title"),
                    gettext("msg_dir_exists_no_git")
                )
                if reply:
                    try:
                        shutil.rmtree(self._fork_dir)
                        self._do_clone(url, self._fork_dir, branch)
                    except Exception as e:
                        QMessageBox.critical(self, gettext("fork_result_title"),
                                             str(e))
        else:
            self._do_clone(url, self._fork_dir, branch)

    def _do_clone(self, url, target_path, branch):
        # Clean up any previous clone thread
        if self.clone_thread and self.clone_thread.isRunning():
            self.clone_thread.terminate()
            self.clone_thread.wait(3000)

        self.clone_thread = GitCloneWorker(url, target_path, branch)
        self.clone_thread.output_signal.connect(lambda line: self._dump_to_debug(f"[FORK] {line}"))
        self.clone_thread.finished_signal.connect(self._on_clone_finished)
        self.clone_thread.start()

    def _do_pull(self):
        # Clean up any previous pull thread
        if self._pull_thread and self._pull_thread.isRunning():
            self._pull_thread.cancel()
            self._pull_thread.terminate()
            self._pull_thread.wait(3000)

        self._dump_to_debug("[FORK] " + gettext("msg_pulling"))
        self.build_btn.setEnabled(False)
        self._pull_thread = GitPullWorker(self._fork_dir)
        self._pull_thread.output_signal.connect(
            lambda line: self._dump_to_debug(f"[PULL] {line}")
        )
        self._pull_thread.finished_signal.connect(self._on_pull_finished)
        self._pull_thread.start()

    def closeEvent(self, event):
        """Clean up background threads before closing the dialog."""
        import sys
        sys.stderr.write(f"[FORK] CLOSE: _pull_thread={self._pull_thread}, clone_thread={self.clone_thread}, build_thread={self.build_thread}\n")
        sys.stderr.flush()
        # Terminate pull thread if still running
        if self._pull_thread and self._pull_thread.isRunning():
            sys.stderr.write("[FORK] CLOSE: terminating _pull_thread\n")
            sys.stderr.flush()
            self._pull_thread.cancel()
            self._pull_thread.terminate()
            self._pull_thread.wait(3000)
        self._pull_thread = None
        # Terminate clone thread if still running
        if self.clone_thread and self.clone_thread.isRunning():
            sys.stderr.write("[FORK] CLOSE: terminating clone_thread\n")
            sys.stderr.flush()
            self.clone_thread.terminate()
            self.clone_thread.wait(3000)
        self.clone_thread = None
        # Terminate build thread if still running
        if self.build_thread and self.build_thread.isRunning():
            sys.stderr.write("[FORK] CLOSE: terminating build_thread\n")
            sys.stderr.flush()
            self._build_cancelled = True
            self.build_thread.cancel()
            self.build_thread.terminate()
            self.build_thread.wait(3000)
            # Clean up build dir
            if self._fork_dir:
                build_path = Path(self._fork_dir) / "build"
                if build_path.exists():
                    try:
                        shutil.rmtree(build_path)
                    except Exception:
                        pass
        self.build_thread = None
        sys.stderr.write("[FORK] CLOSE: done\n")
        sys.stderr.flush()
        event.accept()

    def _on_pull_finished(self, returncode):
        import sys
        sys.stderr.write(f"[FORK] PULL EXIT: returncode={returncode}\n")
        sys.stderr.flush()
        # Wait for thread to fully stop before destroying the object
        if self._pull_thread:
            self._pull_thread.wait(5000)
            self._pull_thread = None
        self.build_btn.setEnabled(True)
        # Set fork_name from URL if not already set (e.g. after app restart)
        if not self._fork_name and self.repo_url:
            entries = _load_fork_entries()
            self._fork_name = _extract_fork_name(self.repo_url, entries)
            sys.stderr.write(f"[FORK] PULL: resolved fork_name='{self._fork_name}' from repo_url\n")
            sys.stderr.flush()
        if returncode == 0:
            self._dump_to_debug("[FORK] " + gettext("msg_pull_complete"))
            self._ask_build()
        else:
            self._dump_to_debug("[FORK] Pull failed (exit {})".format(returncode))
            self._ask_build()

    def _on_clone_finished(self, returncode: int, combined_output: str):
        import sys
        sys.stderr.write(f"[FORK] CLONE EXIT: returncode={returncode}\n")
        sys.stderr.flush()
        # Wait for thread to fully stop before destroying the object
        if self.clone_thread:
            self.clone_thread.wait(5000)
            self.clone_thread = None
        clone_btn = self.findChild(QPushButton, "btn_clone_repo")
        if clone_btn:
            clone_btn.setEnabled(True)
            clone_btn.setText(gettext("btn_clone_repo"))

        if returncode == 0:
            if combined_output.strip():
                self._dump_to_debug("[FORK] " + combined_output)

            # Determine fork name for label
            entries = _load_fork_entries()
            self._fork_name = _extract_fork_name(self.repo_url, entries)

            path_display = self._fork_dir
            self.fork_name_label.setText(
                f"{gettext('lbl_fork_name')} {self._fork_name}"
            )

            self._dump_to_debug(
                "[FORK] " + gettext("msg_clone_complete").format(
                    fork_name=self._fork_name, path=path_display
                )
            )
            self._ask_build()
        else:
            error_msg = combined_output.strip() if combined_output else gettext("msg_build_failed").format(code=returncode)
            self._dump_to_debug("[FORK] " + error_msg)
            QMessageBox.critical(self, gettext("fork_result_title"),
                                 gettext("fork_result_error").format(error=error_msg))

    # ---- Build logic ----

    def _ask_build(self):
        import sys
        sys.stderr.write(f"[FORK] _ask_build called: _fork_name='{self._fork_name}', repo_url='{self.repo_url}', _fork_dir='{self._fork_dir}'\n")
        sys.stderr.flush()
        reply = _ask_question(
            self, gettext("fork_result_title"),
            gettext("msg_build_now")
        )
        if reply:
            self._show_build_group()
            self._build_command = self._get_or_ask_build_command()
            if self._build_command:
                self._start_build()
            else:
                self._show_build_group()
        else:
            # No build - just finish
            QMessageBox.information(
                self, gettext("fork_result_title"),
                gettext("msg_clone_complete").format(
                    fork_name=self._fork_name, path=self._fork_dir
                )
            )

    def _get_or_ask_build_command(self) -> str:
        entries = _load_fork_entries()
        saved_cmd = ""
        if self._fork_name in entries and entries[self._fork_name].get("build"):
            saved_cmd = entries[self._fork_name]["build"]
            import sys
            sys.stderr.write(f"[FORK] Loaded build command for '{self._fork_name}' from ~/.llauncher/llama.json\n")
            sys.stderr.flush()
        else:
            import sys
            sys.stderr.write(f"[FORK] could not load command for '{self._fork_name}' from ~/.llauncher/llama.json\n")
            sys.stderr.flush()

        default_cmd = gettext("msg_default_build_cmd")
        dialog = QDialog(self)
        dialog.setWindowTitle(gettext("lbl_building"))
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(280)

        layout = QVBoxLayout(dialog)

        hint = QLabel(gettext("msg_enter_build_command"))
        layout.addWidget(hint)

        text_edit = QTextEdit()
        text_edit.setFont(QTextEdit().font())
        mono = text_edit.font()
        mono.setFamily("Monospace")
        mono.setPointSize(9)
        text_edit.setFont(mono)
        text_edit.setPlainText(saved_cmd if saved_cmd else default_cmd)
        layout.addWidget(text_edit)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(gettext("btn_ok"))
        cancel_btn2 = QPushButton(gettext("btn_cancel"))
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn2.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn2)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            cmd = text_edit.toPlainText().strip()
            return cmd if cmd else default_cmd
        return ""

    def _save_build_command(self, cmd):
        entries = _load_fork_entries()
        entries[self._fork_name] = {
            "repo": self.repo_url,
            "branch": self.branch_edit.text().strip() or "master",
            "build": cmd,
        }
        with open(LLAMA_JSON_PATH, "w") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def _show_build_group(self):
        self._build_group.setVisible(True)
        self.build_btn.setEnabled(True)
        self.build_cancel_btn.setEnabled(False)
        self.build_status_label.setText("")
        self.build_text.clear()

    def _start_build(self):
        if not self._fork_dir or not os.path.isdir(self._fork_dir):
            QMessageBox.warning(self, gettext("fork_result_title"),
                                gettext("msg_no_directory_selected"))
            return

        # Clean up any previous build thread before starting a new one
        if self.build_thread and self.build_thread.isRunning():
            self._build_cancelled = True
            self.build_thread.cancel()
            self.build_thread.terminate()
            self.build_thread.wait(3000)

        self._build_cancelled = False
        self._build_start_time = time.time()
        self.build_btn.setEnabled(False)
        self.build_cancel_btn.setEnabled(True)
        self.build_status_label.setText(gettext("lbl_building"))
        self.build_text.clear()

        self.build_thread = BuildWorker(self._build_command, self._fork_dir)
        self.build_thread.output_signal.connect(self._on_build_output)
        self.build_thread.finished_signal.connect(self._on_build_finished)
        self.build_thread.start()

    def _on_build_output(self, line: str):
        """Stream build output to the dialog QTextEdit and parent debug area."""
        error_patterns = ["error", "fatal", "failed", "undefined", "cannot"]
        is_error = any(p in line.lower() for p in error_patterns)

        # Dialog QTextEdit with HTML color for errors
        if is_error:
            self.build_text.append(f'<span style="color: #ff6b6b;">{line}</span>')
        else:
            self.build_text.append(line)

        # Scroll dialog text to bottom
        self.build_text.verticalScrollBar().setValue(
            self.build_text.verticalScrollBar().maximum()
        )

        # Also pipe to parent's debug area
        self._dump_to_debug(f"[BUILD] {line}")

    def _on_build_finished(self, returncode: int, summary: str):
        import sys
        # Ensure thread has fully stopped before dropping reference
        if self.build_thread:
            self.build_thread.wait(5000)
        sys.stderr.write(f"[FORK] BUILD EXIT: returncode={returncode} summary={summary!r}\n")
        sys.stderr.flush()
        self.build_thread = None
        sys.stderr.write(f"[FORK] BUILD: build_thread set to None\n")
        sys.stderr.flush()
        self.build_btn.setEnabled(True)
        self.build_cancel_btn.setEnabled(False)
        self.build_status_label.setText(summary)

        elapsed = ""
        if hasattr(self, '_build_start_time'):
            elapsed = self._format_elapsed(time.time() - self._build_start_time)

        # Clean up build/ directory only on failure or cancellation
        build_path = Path(self._fork_dir) / "build"
        if returncode != 0 and build_path.exists():
            try:
                shutil.rmtree(build_path)
            except Exception:
                pass

        if returncode == 0:
            # Ask about switching to this fork
            reply = _ask_question(
                self, gettext("fork_result_title"),
                gettext("msg_switch_fork")
            )
            if reply:
                self._switch_to_fork()
                sys.stderr.write(f"[FORK] BUILD: about to close() - build_thread={self.build_thread}\n")
                sys.stderr.flush()
                self.close()
                self._show_build_completed(elapsed)
            else:
                self._dump_to_debug("[FORK] " + gettext("msg_build_complete"))
                sys.stderr.write(f"[FORK] BUILD: about to close() - build_thread={self.build_thread}\n")
                sys.stderr.flush()
                self.close()
                self._show_build_completed(elapsed)
        else:
            self._dump_to_debug("[FORK] " + summary)
            reply = _ask_question(
                self, gettext("fork_result_title"),
                summary + "\n\n" + gettext("msg_build_retry")
            )
            if reply:
                self._start_build()

    def _cancel_build(self):
        self._build_cancelled = True
        if self.build_thread:
            self.build_thread.cancel()
        # Clean up build directory
        build_path = Path(self._fork_dir) / "build"
        if build_path.exists():
            try:
                shutil.rmtree(build_path)
            except Exception:
                pass
        self.build_status_label.setText(gettext("status_cancelled"))
        self.build_btn.setEnabled(True)
        self.build_cancel_btn.setEnabled(False)

    # ---- Elapsed time helpers ----

    def _format_elapsed(self, seconds: float) -> str:
        """Format seconds as hh:mm:ss."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _show_build_completed(self, elapsed: str):
        """Show build completion dialog with elapsed time and close the fork dialog."""
        msg = gettext("msg_build_completed").format(elapsed=elapsed)
        QMessageBox.information(
            self, gettext("fork_result_title"),
            msg
        )

    # ---- Switch to fork ----

    def _switch_to_fork(self):
        """Update the parent window to use this fork's binary."""
        import sys
        print(f"[FORK-DEBUG] === SWITCH START === fork_dir={self._fork_dir}", file=sys.stderr)
        # Update llama_cpp_path to point to the fork directory
        if self.parent() and hasattr(self.parent(), 'llama_cpp_path'):
            self.parent().llama_cpp_path = self._fork_dir
            print(f"[FORK-DEBUG] llama_cpp_path -> {self.parent().llama_cpp_path}", file=sys.stderr)

        # Update the parent's llama.cpp directory field
        if self.parent() and hasattr(self.parent(), 'exe_line'):
            self.parent().exe_line.setText(self._fork_dir)
            print(f"[FORK-DEBUG] exe_line nach Zeile 832 -> {self.parent().exe_line.text()}", file=sys.stderr)

        # Refresh executables (now checks build/bin/)
        if self.parent() and hasattr(self.parent(), 'find_executables'):
            self.parent().find_executables()
            print(f"[FORK-DEBUG] exe_line nach find_executables -> {self.parent().exe_line.text()}", file=sys.stderr)

        # Auto-select llama-server if available
        if self.parent() and hasattr(self.parent(), 'exe_combo'):
            combo = self.parent().exe_combo
            print(f"[FORK-DEBUG] exe_combo count vor clear: {combo.count()}", file=sys.stderr)
            combo.clear()
            exe_dir = Path(self._fork_dir)
            for d in [exe_dir, exe_dir / "build", exe_dir / "build" / "bin"]:
                if d.exists():
                    try:
                        items = sorted(list(d.iterdir()))
                        print(f"[FORK-DEBUG] Verzeichnis {d}: {len(items)} Einträge", file=sys.stderr)
                        for f in items:
                            if f.is_file() and f.name == "llama-server":
                                combo.addItem(f.name)
                                print(f"[FORK-DEBUG] Hinzugefügt: {f.name} aus {d}", file=sys.stderr)
                    except Exception as e:
                        print(f"[FORK-DEBUG] Fehler bei {d}: {type(e).__name__}: {e}", file=sys.stderr)

            # Select build/bin/llama-server (last occurrence = build/bin, not root)
            idx = -1
            for i in range(combo.count() - 1, -1, -1):
                if combo.itemText(i) == "llama-server":
                    idx = i
                    break
            print(f"[FORK-DEBUG] Gefundener Index für llama-server: {idx}", file=sys.stderr)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                print(f"[FORK-DEBUG] setCurrentIndex({idx}) aufgerufen", file=sys.stderr)

            # Update cache-type options for the new binary
            if self.parent() and hasattr(self.parent(), 'update_cache_type_options'):
                exe_name = combo.currentText()
                if exe_name:
                    binary_path = str(Path(self._fork_dir) / "build" / "bin" / exe_name)
                    if not Path(binary_path).exists():
                        binary_path = str(Path(self._fork_dir) / "build" / exe_name)
                    if not Path(binary_path).exists():
                        binary_path = str(Path(self._fork_dir) / exe_name)
                    self.parent().update_cache_type_options(binary_path)

            # Save config
            if self.parent() and hasattr(self.parent(), 'exe_combo'):
                selected = self.parent().exe_combo.currentText()
                if selected:
                    exe_full_path = str(Path(self._fork_dir) / "build" / "bin" / selected)
                    if not Path(exe_full_path).exists():
                        exe_full_path = str(Path(self._fork_dir) / "build" / selected)
                    if not Path(exe_full_path).exists():
                        exe_full_path = str(Path(self._fork_dir) / selected)
                    from storage import save_config
                    save_config({
                        "llama_cpp_path": self._fork_dir,
                        "selected_executable": exe_full_path,
                    })
                    self.parent().exe_line.setText(exe_full_path)
                    print(f"[FORK-DEBUG] exe_line nach save_config -> {self.parent().exe_line.text()}", file=sys.stderr)
                    print(f"[FORK-DEBUG] selected_executable in config -> {exe_full_path}", file=sys.stderr)

        self._dump_to_debug(
            f"[FORK] Switched to {self._fork_name} -> {self._fork_dir}"
        )
        QMessageBox.information(
            self, gettext("fork_result_title"),
            gettext("msg_build_complete")
        )
