#!/usr/bin/env python3
"""Process management and inspection for llauncher"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import shlex
from PyQt6.QtCore import QThread, pyqtSignal

class ProcessRunner(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)
    
    # Class variable to store the actual daemon thread
    _daemon_thread = None
    _daemon_lock = threading.Lock()
    
    def __init__(self, args: list[str], workdir: str):
        super().__init__()
        self.args = args
        self.workdir = workdir
        self._process = None
    
    def start(self):
        """Override start() to use a daemon thread instead of QThread's native threading."""
        with self._daemon_lock:
            if self._daemon_thread is None or not self._daemon_thread.is_alive():
                self._daemon_thread = threading.Thread(
                    target=self._run_daemon,
                    args=(self.args, self.workdir),
                    daemon=True
                )
                self._daemon_thread.start()
        super().start()
    
    def _run_daemon(self, args, workdir):
        """Run the actual process monitoring in a daemon thread."""
        self._run(args, workdir)
    
    def _run(self, args, workdir):
        """Run the actual process monitoring logic."""
        try:
            self._process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=workdir,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
            while True:
                if self._process is None:
                    break
                if self._process.poll() is not None:
                    break
                line = self._process.stdout.readline()
                if not line:
                    break
                if line:
                    self.output_signal.emit(line.strip())
            
            if self._process and self._process.poll() is None:
                returncode = self._process.wait()
            else:
                returncode = 0
            
            self.finished_signal.emit(returncode)
        except Exception as e:
            self.output_signal.emit(f"ERROR: {e}")
            self.finished_signal.emit(-1)
    
    def run(self) -> None:
        """QThread's run method - just yield control, actual work is in daemon thread."""
        pass
    
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
        """Force the thread to exit by clearing the process reference and closing pipes."""
        if self._process and self._process.poll() is None:
            if self._process.stdout:
                self._process.stdout.close()
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
                    if i + 1 < len(args) and not args[i + 1].startswith('-'):
                        param_dict[arg] = args[i + 1]
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
    from PyQt6.QtCore import Qt
    
    dialog = QDialog(parent_window)
    dialog.setWindowTitle("Externe Parameter")
    layout = QVBoxLayout(dialog)
    
    text = QTextEdit()
    text.setReadOnly(True)
    text.setPlainText("Externe Parameter (nicht in APP verwaltet):\n\n")
    for key, value in external_args.items():
        text.append(f"  {key} = {value}")
    layout.addWidget(text)
    
    close_btn = QPushButton("Schließen")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)
    
    dialog.exec()


def read_and_apply_running_args(window, ui_components=None, param_keys=None):
    """
    Liest laufende llama-server Prozesse und wendet Parameter auf UI an.
    
    Args:
        window: llauncher Hauptfenster mit Slider- und Edit-Widgets
        ui_components: Optionaler ui_components dict (fuer Kompatibilitaet)
        param_keys: Optionaler param_keys set (fuer Kompatibilitaet)
        
    Returns:
        tuple: (external_args, model_path, exe_path, pid_found)
               external_args: Nur Parameter, die nicht in PARAM_DEFINitions sind
    """
    param_definitions = getattr(window, 'PARAM_DEFINITIONS', {})
    
    PARAM_ALIAS_MAP = {
        '--ctx-size': '-c',
        '--batch-size': '-b',
        '--ubatch-size': None,
        '--image-min-tokens': None,
        '--cont-batching': None,
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
            idx = window.model_combo.findText(model_name)
            if idx >= 0:
                window.model_combo.setCurrentIndex(idx)
            else:
                window.model_combo.setCurrentText(model_name)
    
    managed_args = {}
    
    normalized_args = {}
    for key, value in external_args.items():
        if key == '--cont-batching':
            continue
        
        mapped_key = PARAM_ALIAS_MAP.get(key, key)
        
        if mapped_key is None:
            normalized_args[key] = value
        else:
            normalized_args[mapped_key] = value
    
    external_args = normalized_args
    
    for key, value in external_args.items():
        if key in ('-m', '--model'):
            continue
        
        if key in ('-m', '--mmproj'):
            if hasattr(window, 'mmproj_line'):
                window.mmproj_line.setText(value)
            managed_args[key] = value
            continue
        
        if key in ('-c', '-n', '-t', '-b', '-ngl'):
            slider = getattr(window, f'{key}_slider', None)
            edit = getattr(window, f'{key}_edit', None)
            if slider and edit:
                slider.setValue(int(value))
                edit.setText(value)
            managed_args[key] = value
        
        elif key.startswith('--'):
            param_sliders = getattr(window, 'param_sliders', {})
            slider_data = param_sliders.get(key, {})
            
            if slider_data:
                combo = slider_data.get('combo')
                if combo:
                    idx = combo.findText(value)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    else:
                        combo.setCurrentText(value)
                    managed_args[key] = value
                else:
                    edit = slider_data.get('edit')
                    if edit and not slider_data.get('slider'):
                        edit.setText(value)
                        managed_args[key] = value
                    else:
                        slider = slider_data.get('slider')
                        
                        if slider and edit:
                            try:
                                float_val = float(value)
                                slider.setValue(int(float_val * 100))
                                edit.setText(value)
                            except ValueError:
                                pass
                            managed_args[key] = value
    
    external_only = {k: v for k, v in external_args.items() 
                     if k not in managed_args and k not in ('-m', '--model')}
    
    return external_only, model_path, exe_path, pid_found
