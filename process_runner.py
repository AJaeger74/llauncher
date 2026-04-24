#!/usr/bin/env python3
"""Process management and inspection for llauncher"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import shlex
from PyQt6.QtCore import QThread, pyqtSignal, Qt


class ProcessRunner(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, args: list[str], workdir: str):
        super().__init__()
        self.args = args
        self.workdir = workdir
        self._process = None

    def start(self):
        """Start the process. Delegates to QThread's native threading."""
        super().start()

    def run(self) -> None:
        """Run the actual process in QThread's execution thread."""
        try:
            self._process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.workdir,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
        except Exception as e:
            self.output_signal.emit(f"ERROR: {e}")
            self.finished_signal.emit(-1)
            return

        try:
            while True:
                if self._process is None or self._process.poll() is not None:
                    break
                line = self._process.stdout.readline()
                if not line:
                    break
                if line:
                    self.output_signal.emit(line.strip())

            if self._process:
                returncode = self._process.wait()
                self.finished_signal.emit(returncode)
            else:
                # Detached via force_exit() — don't emit an error, just finish
                self.finished_signal.emit(-2)
        except Exception as e:
            self.output_signal.emit(f"ERROR: {e}")
            self.finished_signal.emit(-1)

    def get_pid(self):
        """PID des laufenden Prozesses zurueckgeben"""
        if self._process and self._process.pid:
            return self._process.pid
        return None

    def get_args_from_proc(self) -> list[str]:
        """Parameter aus /proc/[pid]/cmdline lesen (Linux-spezifisch)"""
        pid = self.get_pid()
        if not pid:
            return []

        try:
            cmdline_path = f'/proc/{pid}/cmdline'
            with open(cmdline_path, 'r') as f:
                content = f.read()
            args = [arg for arg in content.split('\x00') if arg]
            return args[1:]
        except (FileNotFoundError, PermissionError, IOError):
            return []

    @staticmethod
    def terminate_by_pid(pid: int, timeout_sec: float = 3.0) -> bool:
        """Terminiere externen Prozess via PID (SIGINT -> SIGTERM -> SIGKILL)."""
        if pid is None:
            return True

        for attempt in range(2):
            try:
                os.kill(pid, signal.SIGINT)
                start = time.time()
                while os.path.exists(f'/proc/{pid}') and (time.time() - start) < timeout_sec / 2:
                    time.sleep(0.1)
                if not os.path.exists(f'/proc/{pid}'):
                    return True
            except ProcessLookupError:
                return True

        try:
            os.kill(pid, signal.SIGTERM)
            start = time.time()
            while os.path.exists(f'/proc/{pid}') and (time.time() - start) < timeout_sec / 2:
                time.sleep(0.1)
            if not os.path.exists(f'/proc/{pid}'):
                return True
        except ProcessLookupError:
            return True

        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            return True
        except ProcessLookupError:
            pass

        print(f"Failed to kill process {pid}")
        return False

    def force_exit(self):
        """Detach from the process — don't kill it, just close our pipes."""
        if self._process and self._process.poll() is None:
            # Close stdout to unblock readline() — need to close the file descriptor
            try:
                if self._process.stdout:
                    # Close the underlying fd to force EOF on next read
                    fd = self._process.stdout.fileno()
                    os.close(fd)
                    self._process.stdout = None
            except Exception:
                pass
            
            # Clear reference so run() loop sees None immediately
            self._process = None

    def terminate_process(self) -> bool:
        if not self._process or self._process.poll() is not None:
            return True
        pid = self._process.pid
        print(f"Stopping process {pid}...")
        return self.terminate_by_pid(pid)


def find_llama_processes():
    """Findet alle laufenden llama-server / llama-cli Prozesse."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "llama"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pids = [int(pid) for pid in out.strip().split('\n') if pid]
        return pids
    except subprocess.CalledProcessError:
        return []


def check_running_processes():
    """Prueft ob ein llama-server Prozess laeuft und zeigt Details an."""
    pids = find_llama_processes()
    if not pids:
        return "Kein llama-server Prozess gefunden."

    result = []
    for pid in pids:
        try:
            cmdline_path = f'/proc/{pid}/cmdline'
            with open(cmdline_path, 'r') as f:
                cmdline = f.read()
            args = [arg for arg in cmdline.split('\x00') if arg]
            result.append(f"PID {pid}: {' '.join(args)}")
        except (FileNotFoundError, PermissionError):
            result.append(f"PID {pid}: Zugriff verweigert")

    return '\n'.join(result)


def read_running_llama_args() -> tuple[dict, str, str, int]:
    """Liest Parameter aus einem laufenden llama-server Prozess."""
    pids = find_llama_processes()
    if not pids:
        return None, None, None, False

    for pid in pids:
        try:
            cmdline_path = f'/proc/{pid}/cmdline'
            with open(cmdline_path, 'r') as f:
                cmdline = f.read()
            args = [arg for arg in cmdline.split('\x00') if arg]

            if len(args) < 2:
                continue

            exe_path = args[0] if args[0].startswith('/') else f'/proc/{pid}/exe'
            model_path = None
            param_dict = {}

            i = 1
            while i < len(args):
                arg = args[i]
                if arg == '-m' and i + 1 < len(args):
                    model_path = args[i + 1]
                    param_dict['-m'] = args[i + 1]
                    i += 2
                elif arg == '--model' and i + 1 < len(args):
                    model_path = args[i + 1]
                    param_dict['--model'] = args[i + 1]
                    i += 2
                elif arg == '--mmproj' and i + 1 < len(args):
                    param_dict['--mmproj'] = args[i + 1]
                    i += 2
                elif arg.startswith('-') or arg.startswith('--'):
                    if i + 1 < len(args):
                        next_arg = args[i + 1]

                        # Lange Flags (--foo) sind immer Boolean-Flags ohne Wert
                        if next_arg.startswith('--'):
                            param_dict[arg] = True
                            i += 1

                        # Negative Zahlen (-1, -2) SIND Werte, keine Flags
                        elif next_arg.startswith('-') and next_arg.lstrip('-').isdigit():
                            param_dict[arg] = next_arg
                            i += 2

                        # Andere Parameter mit '-' (z.B. -c) die keine Zahlen sind -> Boolean
                        elif next_arg.startswith('-'):
                            param_dict[arg] = True
                            i += 1

                        # Normaler Wert (keine Flag, keine negative Zahl)
                        else:
                            param_dict[arg] = next_arg
                            i += 2
                    else:
                        param_dict[arg] = True
                        i += 1
                else:
                    i += 1

            return param_dict, model_path, exe_path, True
        except (FileNotFoundError, PermissionError, IOError):
            continue

    return None, None, None, False


def show_external_args_dialog(external_args, model_path, parent_window):
    """Zeigt einen Dialog mit externen Parametern."""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt

    # i18n Schluessel holen
    def t(key):
        from storage import load_config
        config = load_config()
        locale = config.get('locale', 'de')
        try:
            with open(f'locales/{locale}.json', 'r') as f:
                import json
                translations = json.load(f)
                return translations.get(key, key)
        except:
            return key

    dialog = QDialog(parent_window)
    dialog.setWindowTitle(t('dialog_external_args_title') or "Externe Parameter")
    layout = QVBoxLayout(dialog)

    # Title label
    title_label = QLabel(t('msg_external_args') or "Externe Parameter (nicht in APP verwaltet):")
    title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
    layout.addWidget(title_label)

    text = QTextEdit()
    text.setReadOnly(True)

    # Monospace font fuer bessere Lesbarkeit
    mono_font = QFont()
    mono_font.setFamily("Monospace")
    mono_font.setPointSize(9)
    text.setFont(mono_font)

    for key, value in external_args.items():
        # Boolean Flags ohne "= True" anzeigen, negative Zahlen mit =
        if value is True:
            text.append(f"  {key}")
        elif isinstance(value, str) and value.startswith('-') and value.lstrip('-').isdigit():
            # Negative Zahl als Wert (z.B. -n = -1)
            text.append(f"  {key} = {value}")
        else:
            text.append(f"  {key} = {value}")
    layout.addWidget(text)

    close_btn = QPushButton(t('btn_close') or "Schliessen")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)

    # Dialog-groesse verdoppeln (mindestens 500x400)
    dialog.setMinimumSize(500, 400)
    dialog.resize(600, 500)

    dialog.exec()


def read_and_apply_running_args(window, ui_components=None, param_keys=None):
    """
    Liest laufende llama-server Prozesse und wendet Parameter auf UI an.

    Args:
        window: llauncher Hauptfenster mit Slider- und Edit-Widgets
        ui_components: Optionaler ui_components dict ( fuer Kompatibilitaet)
        param_keys: Optionaler param_keys set ( fuer Kompatibilitaet)

    Returns:
        tuple: (external_args, model_path, exe_path, pid_found)
               external_args: Nur Parameter, die nicht in PARAM_DEFINitions sind
    """
    param_definitions = getattr(window, 'PARAM_DEFINITIONS', {})

    PARAM_ALIAS_MAP = {
        '--ctx-size': '-c',
        '--batch-size': '-b',
        '--parallel': '-np',
        '--ubatch-size': None,  # External, unmanaged parameter
    }

    # Reverse map: short aliases -> long form (for normalization)
    ALIAS_TO_LONG_MAP = {
        '-c': '--ctx-size',
        '-b': '--batch-size',
        '-np': '--parallel',
        '-m': '--model',
        '-n': '--predict-prev',  # -n for predict-prev context
        '--mmproj': '--mmproj',  # mmproj stays as-is
    }

    external_args, model_path, exe_path, pid_found = read_running_llama_args()

    if external_args is None:
        return [], None, None, False

    if exe_path and hasattr(window, 'exe_line'):
        window.exe_line.setText(exe_path)

    if model_path:
        if hasattr(window, 'model_line'):
            model_dir = os.path.dirname(model_path)
            if model_dir:
                window.model_line.setText(model_dir)

        if hasattr(window, 'model_combo'):
            model_name = os.path.basename(model_path)
            # Suche nach model_name im UserRole (nicht im Display-Text mit Groessenangabe!)
            found_index = -1
            for i in range(window.model_combo.count()):
                user_data = window.model_combo.itemData(i, role=Qt.ItemDataRole.UserRole)
                if user_data and user_data == model_name:
                    found_index = i
                    break

            if found_index >= 0:
                window.model_combo.setCurrentIndex(found_index)
            else:
                window.model_combo.setCurrentText(model_name)

    managed_args = {}
    normalized_args = {}

    # First, apply ALL parameters to their respective widgets (managed + unmanaged)
    for key, value in external_args.items():
        # Skip cont-batching - this is an internal llama.cpp flag
        if key == '--cont-batching':
            continue

        # Debug logging for batch-size handling
        if key == '--batch-size' or key == '-b':
            print(f"[DEBUG] Processing batch-size: key={key}, value={value}")

        # First try standard mapping (--param -> -param)
        mapped_key = PARAM_ALIAS_MAP.get(key, key)

        # If mapped_key still has --, check reverse map (-param -> --param)
        if mapped_key.startswith('--'):
            mapped_key = ALIAS_TO_LONG_MAP.get(mapped_key, mapped_key)

        # Handle special parameters that are managed but NOT in PARAM_DEFINITIONS
        if key == '-m' or key == '--model' or key == '--mmproj':
            if key in ('-m', '--model'):
                # model handling (skip setting widgets, already handled earlier)
                continue
            elif key == '--mmproj':
                # mmproj is managed via mmproj_line but not in PARAM_DEFINITIONS
                if hasattr(window, 'mmproj_line'):
                    window.mmproj_line.setText(value)
                managed_args[key] = value
                continue

        # Determine the actual key to use in param_definitions (for slider/combo params)
        actual_key = key if key in param_definitions else mapped_key if mapped_key in param_definitions else None

        print(f"[DEBUG] Processing param: key={key}, value={value}, mapped={mapped_key}, actual={actual_key}")

        if actual_key:
            if actual_key in ('-c', '-n', '-t', '-b', '-ngl', '-np') or key in ('-c', '-n', '-t', '-b', '-ngl', '-np'):
                print(f"[DEBUG] Trying to set {actual_key} with value={value}")

                # Use param_sliders dict instead of direct attributes (widget attributes don't exist)
                param_sliders_dict = getattr(window, 'param_sliders', {}).get(actual_key, {})
                slider = param_sliders_dict.get('slider') if isinstance(param_sliders_dict, dict) else None
                edit = param_sliders_dict.get('edit') if isinstance(param_sliders_dict, dict) else None

                print(f"[DEBUG] Found slider={slider is not None}, edit={edit is not None}")

                if slider and edit:
                    # Special handling for -ngl "all"
                    if actual_key == '-ngl' and value == 'all':
                        window.ngl_all_checkbox.setChecked(True)
                        slider.setValue(0)
                        edit.setText('all')
                        print(f"[DEBUG] Set -ngl to 'all', checkbox checked")
                    else:
                        try:
                            slider.setValue(int(value))
                            edit.setText(value)
                            print(f"[DEBUG] Successfully set {actual_key} to {value}")
                        except ValueError as e:
                            print(f"[DEBUG] ValueError setting {actual_key}: {e}")
                            pass  # Skip invalid int values
                else:
                    print(f"[DEBUG] WARNING: Could not find slider/edit for {actual_key}, param_sliders={list(getattr(window, 'param_sliders', {}).keys())}")
                managed_args[actual_key] = value
                # Also add to normalized_args with long form for external display
                if key.startswith('--'):
                    normalized_args[key] = value
                continue

            elif actual_key and actual_key.startswith('--'):
                param_sliders = getattr(window, 'param_sliders', {})
                slider_data = param_sliders.get(actual_key, {})

                if slider_data:
                    combo = slider_data.get('combo')
                    if combo:
                        idx = combo.findText(value)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                        else:
                            combo.setCurrentText(value)
                        managed_args[actual_key] = value
                        continue

                    edit = slider_data.get('edit')
                    if edit and not slider_data.get('slider'):
                        edit.setText(value)
                        managed_args[actual_key] = value
                        continue
                    else:
                        slider = slider_data.get('slider')

                        if slider and edit:
                            try:
                                float_val = float(value)
                                slider.setValue(int(float_val * 100))
                                edit.setText(value)
                            except ValueError:
                                pass  # Skip invalid float values
                            managed_args[actual_key] = value
                            continue

    # If we reach here, parameter is not managed - keep in normalized_args
    if mapped_key:  # Only add if mapped_key is not None
        normalized_args[mapped_key] = value

    external_args = normalized_args

    # Filter out managed parameters AND special parameters (mmproj, model)
    # Note: --mmproj is managed via mmproj_line but not in PARAM_DEFINITIONS
    excluded_keys = ('-m', '--model', '--mmproj')  # Always exclude these from external display
    external_only = {k: v for k, v in external_args.items()
                     if k not in managed_args and k not in excluded_keys}

    return external_only, model_path, exe_path, pid_found
