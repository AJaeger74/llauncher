#!/usr/bin/env python3
"""
llauncher – GUI für llama.cpp
Ein Mischpult-Style Launcher zur Steuerung von llama.cpp mit Presets, Benchmarking und GPU-Monitoring.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Importiere GGUF-Utilities aus separater Datei
from gguf_utils import get_cpu_count, read_gguf_context_length
from help_parser import parse_cache_type_options
from storage import (
    load_config, save_config, load_presets, save_presets,
    load_benchmarks, save_benchmarks, apply_preset
)
from gpu_monitor import GPUMonitor, update_gpu_display
from process_runner import ProcessRunner
from float_slider_sync import DirectClickSlider
from preset_manager import (
    show_preset_save_dialog,
    show_preset_load_dialog,
    show_preset_args as preset_show_args,
    ask_quality_and_save_benchmark,
)
from ui_builder import build_llauncher_ui, setup_timers_and_load

# Import i18n for lazy gettext() loading
from i18n import I18nManager

from PyQt6.QtCore import Qt, pyqtSignal, QDate, QTime, QEvent, QTimer


def gettext(key: str) -> str:
    """Lazy-loaded gettext function - waits for i18n initialization."""
    try:
        return I18nManager.get_instance().gettext(key)
    except Exception:
        return key


def translatable(key: str, **kwargs) -> str:
    """Get translated string with optional format arguments.
    
    Usage:
        translatable("msg_preset_saved", name=name)
        translatable("status_gpu_stats", gpu=50, vram=4096, total=8192, temp=65)
    
    Falls Translation fehlt, wird der Key zurückgegeben (mit format args falls vorhanden).
    """
    try:
        translated = I18nManager.get_instance().gettext(key)
        if kwargs:
            return translated.format(**kwargs)
        return translated
    except Exception:
        # Fallback: return key itself, try to format if kwargs provided
        if kwargs:
            return key.format(**kwargs)
        return key


def t(key: str, **kwargs) -> str:
    """Shortcut for translatable()."""
    return translatable(key, **kwargs)


from http_benchmark_thread import HTTPBenchmarkRunner
from model_info_fetcher import fetch_running_model_info
from PyQt6.QtGui import QFont, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QPlainTextEdit,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QHeaderView,
    QSplitter,
)


class llauncher(QMainWindow):
    VERSION = "1.0"

    CONFIG_DIR = Path.home() / ".llauncher"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    PRESETS_FILE = CONFIG_DIR / "presets.json"
    BENCHMARKS_FILE = CONFIG_DIR / "benchmarks.json"

    PARAM_DEFINITIONS_BASE = {
   "-c": {
        "label_key": "param_context_size",
        "type": "slider",
        "min": 2048,
        "max": 8192,
        "default": 4096,
        "tooltip_key": "tooltip_context_size",
        "tooltip_key": "tooltip_context_size",
    },
        "--cache-type-k": {
            "label_key": "help_parser_k_type",  # Wird später mit gettext aufgelöst
            "type": "combo",
            "default": "f16",
            "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            "tooltip_key": "tooltip_cache_type_k",
            "tooltip_key": "tooltip_cache_type_k",
        },
        "--cache-type-v": {
            "label_key": "help_parser_v_type",  # Wird später mit gettext aufgelöst
            "type": "combo",
            "default": "f16",
            "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
            "tooltip_key": "tooltip_cache_type_v",
        },
        "-n": {
            "label_key": "param_max_tokens",  # Wird später mit gettext aufgelöst
            "type": "slider",
            "min": -1,
            "max": 8192,
           "default": 4096,
            "tooltip_key": "tooltip_max_tokens",
        },
        "-np": {
            "label_key": "param_parallel_slots",  # Wird später mit gettext aufgelöst
            "type": "slider",
            "min": -1,
            "max": 8,
            "default": -1,
            "tooltip_key": "tooltip_np",
        },
"-t": {
            "label_key": "param_cpu_threads",  # Wird später mit gettext aufgelöst
            "type": "slider",
            "min": 1,
            "max": 32,
            "default": 8,
            "tooltip_key": "tooltip_threads",
        },
"-b": {
            "label_key": "param_batch_size",  # Wird später mit gettext aufgelöst
            "type": "slider",
            "min": 1,
            "max": 8192,
           "default": 2048,
            "tooltip_key": "tooltip_batch_size",
        },
        "-ngl": {
            "label_key": "param_gpu_layers",  # Wird später mit gettext aufgelöst
            "type": "slider",
            "min": 0,
            "max": "{{GPU_LAYERS}}",  # Wird dynamisch ersetzt (optional)
            "default": 35,
            "tooltip_key": "tooltip_gpu_layers",
        },
        "--temp": {
            "label_key": "param_temperature",  # Muss noch in JSONs hinzugefügt werden
            "type": "float_slider",
            "min": 0.1,
            "max": 2.0,
            "default": 0.8,
            "step": 0.1,
            "tooltip_key": "tooltip_temperature",
        },
       "--top-p": {
            "label_key": "param_top_p",  # Muss noch in JSONs hinzugefügt werden
            "type": "float_slider",
            "min": 0.1,
            "max": 1.0,
            "default": 0.95,
            "step": 0.05,
            "tooltip_key": "tooltip_top_p",
        },
        "--top-k": {
            "label_key": "param_top_k",
            "type": "slider",
            "min": 0,
            "max": 1000,
            "default": 40,
            "tooltip_key": "tooltip_top_k",
        },
        "--min-p": {
            "label_key": "param_min_p",
            "type": "float_slider",
            "min": 0.0,
            "max": 0.5,
            "default": 0.05,
            "step": 0.01,
            "tooltip_key": "tooltip_min_p",
        },
        "--repeat-penalty": {
            "label_key": "param_repeat_penalty",  # Muss noch in JSONs hinzugefügt werden
            "type": "float_slider",
            "min": 0.5,
            "max": 2.5,
            "default": 1.0,
            "step": 0.05,
            "tooltip_key": "tooltip_repeat_penalty",
        },
        "--flash-attn": {
            "label_key": "param_flash_attn",  # Muss noch in JSONs hinzugefügt werden
            "type": "combo",
            "default": "off",
            "options": ["off", "on"],
            "tooltip_key": "tooltip_flash_attn",
        },

        "--host": {
            "label_key": "param_host",  # Muss noch in JSONs hinzugefügt werden
            "type": "text_input",
            "default": "localhost",
            "tooltip_key": "tooltip_host",
        },
       "--slot-save-path": {
            "label_key": "param_slot_save_path",
            "type": "path_input",
            "default": "/dev/shm/llama-slots",
            "tooltip_key": "tooltip_slot_save_path",
        },
        "benchmark_file_path": {
            "label_key": "param_benchmark_file",
            "type": "file_input",
            "default": "",
            "tooltip_key": "tooltip_benchmark_file",
        },
    }


    # Dynamische Parameter definieren (ersetzt Platzhalter)
    @classmethod
    def get_param_definitions(cls):
        """PARAM_DEFINITIONS mit dynamischen Werten erstellen"""
        import copy
        definitions = copy.deepcopy(cls.PARAM_DEFINITIONS_BASE)
        
        # CPU Count ersetzen
        cpu_count = get_cpu_count()
        for key, value in definitions.items():
            if isinstance(value, dict) and "max" in value:
                max_val = value["max"]
                if isinstance(max_val, str):
                    if "{{CPU_COUNT}}" in max_val:
                        value["max"] = cpu_count
                    elif "{{GPU_LAYERS}}" in max_val:
                        # GPU Layers als 100 lassen (oder später dynamisch)
                        value["max"] = 100
        
        return definitions

    def __init__(self):
        super().__init__()
        self.llama_cpp_path = str(Path.home() / "llama.cpp")
        self.model_directory = str(Path.home() / "models")
        self.selected_model: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.runner: Optional[ProcessRunner] = None
        self.bench_thread: Optional[QThread] = None  # dedicated thread for benchmarks
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.external_args: Optional[dict] = None  # Externe Parameter vom laufenden Prozess
        self.external_runner_args: Optional[list] = None  # Echte Prozess-Args (für build_full_command)
        self.external_runner_pid: Optional[int] = None
        
        # Theme
        self.light_theme = False
        
        # Dynamische Parameter definieren
        self.PARAM_DEFINITIONS = self.get_param_definitions()

        
 
        build_llauncher_ui(self)
        setup_timers_and_load(self)
        
        # Fenster-Geometrie und Splitter-State laden (nach UI-Setup)
        self.restore_geometry()
        self.process_check_timer.setInterval(1000)  # 1 Sekunde
        self.process_check_timer.timeout.connect(self.check_existing_process)
        self.process_check_timer.start()
        
       # Prüfen ob bereits ein llama-server läuft (vom User gestartet)
        self.check_existing_process()
    
    def apply_theme(self, use_light: bool):
        """Apply theme stylesheet based on use_light flag."""
        from ui_builder import DARK_THEME, LIGHT_THEME
        self.light_theme = use_light
        stylesheet = LIGHT_THEME if use_light else DARK_THEME
        self.setStyleSheet(stylesheet)
        
        # Update params_widget background to match theme
        bg_color = "#ffffff" if use_light else "#1e1e1e"
        color = "#333333" if use_light else "#cccccc"
        border = "border: 1px solid #cccccc;" if use_light else "border: none;"
        
        # Update params_widget (the inner widget with all parameters)
        if hasattr(self, 'params_widget'):
            self.params_widget.setStyleSheet(f"QWidget {{ background-color: {bg_color}; }}")
        
        # Apply to all slider edit fields
        for key, slider_dict in getattr(self, 'param_sliders', {}).items():
            if isinstance(slider_dict, dict) and "edit" in slider_dict:
                edit_widget = slider_dict["edit"]
                edit_widget.setStyleSheet(f"QLineEdit {{ padding: 5px; border-radius: 3px; background-color: {bg_color}; color: {color}; {border} }}")
            elif isinstance(slider_dict, dict) and "combo" in slider_dict:
                combo_widget = slider_dict["combo"]
                combo_widget.setStyleSheet(f"QComboBox {{ padding: 5px; border-radius: 3px; background-color: {bg_color}; color: {color}; {border} }}")
        
        # Update stats label colors to match theme
        if hasattr(self, 'stats_label'):
            if use_light:
                self.stats_label.setStyleSheet("color: #333333; padding: 10px;")
            else:
                self.stats_label.setStyleSheet("color: #cccccc; padding: 10px;")
    
    def on_theme_toggled(self, state):
        """Handle theme checkbox toggle."""
        use_light = state == 2  # Qt.CheckState.Checked.value
        self.apply_theme(use_light)
        
        # Save theme preference to config
        try:
            with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                import json
                config = json.load(f)
            config["theme"] = "light" if use_light else "dark"
            with open(Path.home() / ".llauncher" / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass  # Non-fatal
    
    def on_check_process_click(self):
        """Ruft check_running_processes() auf und zeigt Output im Debug-Fenster."""
        from process_runner import check_running_processes
        
        try:
            output = check_running_processes()
            self.debug_text.append(output)
        except Exception as e:
            self.debug_text.append(f"Fehler beim Prozess-Check: {e}")

    def _load_running_process_args_silent(self):
        """Lädt Parameter ohne Dialog – für automatischen Start-Lauf."""
        return self.load_running_process_args(show_dialogs=False)

    def load_running_process_args(self, show_dialogs: bool = True):
        """Liest Parameter des laufenden llama-server und lädt sie in die App.
        
        Args:
            show_dialogs: Ob Warnungen/Dialoge angezeigt werden sollen (default: True)
        """
        from process_runner import read_and_apply_running_args
        
        # Prüfen ob UI-Komponenten initialisiert sind (Schutz vor Race Condition)
        if not hasattr(self, 'param_sliders') or not self.param_sliders:
            return False
        
        # Zuerst check_existing_process() aufrufen, um external_runner_args zu setzen
        self.check_existing_process()
        
        # UI Components Dict zusammenstellen – greift auf param_sliders zu
        ui_components = {
            'model_line': self.model_line if hasattr(self, 'model_line') else None,
            'mmproj_line': self.mmproj_line if hasattr(self, 'mmproj_line') else None,
            # Integer-Slider (param_sliders[key] = {"slider": DirectClickSlider, "edit": QLineEdit})
            'ctx_slider': self.param_sliders.get('-c', {}).get('slider'),
            'ctx_edit': self.param_sliders.get('-c', {}).get('edit'),
            'batch_slider': self.param_sliders.get('-b', {}).get('slider'),
            'batch_edit': self.param_sliders.get('-b', {}).get('edit'),
            'threads_slider': self.param_sliders.get('-t', {}).get('slider'),
            'threads_edit': self.param_sliders.get('-t', {}).get('edit'),
            'gpu_layers_slider': self.param_sliders.get('-ngl', {}).get('slider'),
            'gpu_layers_edit': self.param_sliders.get('-ngl', {}).get('edit'),
            # -ngl "all" Checkbox (wenn vorhanden)
            'ngl_all_checkbox': getattr(self, 'ngl_all_checkbox', None),
            'parallel_slider': self.param_sliders.get('-np', {}).get('slider'),
            'parallel_edit': self.param_sliders.get('-np', {}).get('edit'),
            # Float-Slider (gleiche Struktur)
            'temp_slider': self.param_sliders.get('--temp', {}).get('slider'),
            'temp_edit': self.param_sliders.get('--temp', {}).get('edit'),
            'top_p_slider': self.param_sliders.get('--top-p', {}).get('slider'),
            'top_p_edit': self.param_sliders.get('--top-p', {}).get('edit'),
            'repeat_penalty_slider': self.param_sliders.get('--repeat-penalty', {}).get('slider'),
            'repeat_penalty_edit': self.param_sliders.get('--repeat-penalty', {}).get('edit'),
            # ComboBox (param_sliders[key] = {"combo": QComboBox})
            'cache_type_k_combo': self.param_sliders.get('--cache-type-k', {}).get('combo'),
            'cache_type_v_combo': self.param_sliders.get('--cache-type-v', {}).get('combo'),
            'flash_attn_combo': self.param_sliders.get('--flash-attn', {}).get('combo'),
            # Text Input (param_sliders[key] = {"edit": QLineEdit})
            'host_edit': self.param_sliders.get('--host', {}).get('edit'),
            'save_path_edit': self.param_sliders.get('--slot-save-path', {}).get('edit'),
        }
        
         # Flag setzen um zu verhindern dass on_model_selected() den Slider überschreibt
        self.loading_running_args = True
        
        external_args, model_path, exe_path, pid_found = read_and_apply_running_args(
            self
        )
        
        if not pid_found:
            self.loading_running_args = False  # Reset auch im Fehlerfall
            if show_dialogs:
                QMessageBox.warning(self, translatable("msg_no_running_process_title"),
                                  translatable("msg_no_llama_server_found"))
            return False
        
         # Modell-Pfad aufteilen in Verzeichnis + Dateiname
        if model_path:
            import os
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path)
            
            # Setze "Modelle"-Feld (Verzeichnis)
            if hasattr(self, 'model_line'):
                self.model_line.setText(model_dir)
            
            # Setze "Modell (.gguf)"-ComboBox auf den Dateinamen
            if hasattr(self, 'model_combo'):
                idx = self.model_combo.findText(model_name)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                else:
                    # Falls Datei nicht in ComboBox ist, direkt setzen
                    self.model_combo.setCurrentText(model_name)
        
         # Exe-Pfad setzen (falls vorhanden) - nur Verzeichnis ohne Filename
        if exe_path and hasattr(self, 'exe_line'):
            import os
            exe_dir = os.path.dirname(exe_path)
            self.exe_line.setText(exe_dir)
        
     # Externe Parameter anzeigen (nicht in APP verwaltet) – nur wenn es welche gibt
        if external_args and len(external_args) > 0 and show_dialogs:
            self._show_external_args_dialog(external_args, model_path)
        
        # Externe Parameter speichern für Debug-Ausgabe und Kommandozeilen-Generierung
        self.external_args = external_args
        
        # Debug-Bereich aktualisieren mit vollständiger Kommandozeile (inkl. externer Args)
        try:
            command = self.build_full_command()
            self.debug_text.setText(command)
        except Exception as e:
            self.debug_text.append(f"⚠️ Konnte Debug-Ausgabe nicht aktualisieren: {e}")
        
        if external_args is not None:
            self.debug_text.append(translatable("msg_loaded_params", pid=pid_found, count=len(external_args)))
        
        # Flag zurücksetzen – jetzt darf on_model_selected() wieder normal arbeiten
        self.loading_running_args = False

    def _show_external_args_dialog(self, external_args, model_path=None):
        """Zeigt einen Dialog mit externen Parametern (nicht in APP verwaltet)."""
        from process_runner import show_external_args_dialog
        
        show_external_args_dialog(external_args, model_path, self)

    def browse_llama_dir(self):
        dialog = QFileDialog(self, "llama.cpp Verzeichnis wählen", self.llama_cpp_path)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                path = selected_paths[0]
                self.exe_line.setText(path)
                self.llama_cpp_path = path
                save_config({"llama_cpp_path": path})
                self.find_executables()

    def browse_model_dir(self):
        dialog = QFileDialog(self, "Model-Verzeichnis wählen", self.model_directory)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                path = selected_paths[0]
                self.model_line.setText(path)
                self.model_directory = path
                save_config({"model_directory": path})
                self.update_model_dropdown()

    def find_executables(self):
        exe_dir = Path(self.llama_cpp_path)
        if not exe_dir.exists():
            self.exe_combo.clear()
            self.exe_combo.addItem("llama.cpp nicht gefunden")
            return

        executables = []
        for f in exe_dir.iterdir():
            if f.is_file() and (f.name == "main" or f.name == "llama-server"):
                executables.append(f.name)

        self.exe_combo.clear()
        for name in sorted(set(executables)):
            self.exe_combo.addItem(name)

    def on_exe_changed(self, name: str):
        # Guard: Signal kann mit leerem String feuern beim ersten Mal
        if not name or name == "":
            return
            
        # Volle Pfad speichern
        exe_full_path = Path(self.llama_cpp_path) / name
        
        save_config({
            "llama_cpp_path": self.llama_cpp_path,
            "selected_executable": str(exe_full_path),  # Key muss mit load_config() übereinstimmen
        })
        
        # Debug: Zeige was wir haben
        self.debug_text.append(f"=== Binary-Auswahl ===")
        self.debug_text.append(f"llama_cpp_path: {self.llama_cpp_path}")
        self.debug_text.append(f"name (aus ComboBox): {name!r}")
        self.debug_text.append(f"exe_full_path: {str(exe_full_path)}")
        
        # Dynamisch cache-type-k/v Optionen aus --help extrahieren
        # Nur wenn name nicht leer und kein Verzeichnis-Pfad (keine '/' im Namen)
        if name and '/' not in name and '\\' not in name:  # Name darf keine Pfadtrenner enthalten
            self.update_cache_type_options(str(exe_full_path))
        else:
            self.debug_text.append(f"⚠️ Skipping --help (name leer oder ungültig)")
    
    def update_cache_type_options(self, binary_path: str):
        """
        Extrahiert dynamisch die allowed values für cache-type-k und cache-type-v
        aus llama-server --help und aktualisiert die ComboBoxen.
        
        Args:
            binary_path: Pfad zum llama-server Binary
        """
        # Guard: Prüfen ob Path existiert (temporär auskommentiert für Debug)
        if not Path(binary_path).exists():
            self.debug_text.append(f"⚠️ Binary nicht gefunden: {binary_path}")
            return
        
        try:
            options = parse_cache_type_options(binary_path)
            
            # Cache-Type K ComboBox aktualisieren
            if '--cache-type-k' in self.param_sliders and 'combo' in self.param_sliders['--cache-type-k']:
                combo_k = self.param_sliders['--cache-type-k']['combo']  # War 'widget', ist aber 'combo'
                current_text = combo_k.currentText()
                
                # ComboBox leeren und mit neuen Optionen füllen
                combo_k.clear()
                for opt in options.get('k', ['f16']):  # Fallback auf f16 wenn nichts gefunden
                    combo_k.addItem(opt)
                
                # Alte Auswahl wiederherstellen falls noch vorhanden
                if current_text in options.get('k', []):
                    idx = combo_k.findText(current_text)
                    if idx >= 0:
                        combo_k.setCurrentIndex(idx)
            
            # Cache-Type V ComboBox aktualisieren
            if '--cache-type-v' in self.param_sliders and 'combo' in self.param_sliders['--cache-type-v']:
                combo_v = self.param_sliders['--cache-type-v']['combo']  # War 'widget', ist aber 'combo'
                current_text = combo_v.currentText()
                
                # ComboBox leeren und mit neuen Optionen füllen
                combo_v.clear()
                for opt in options.get('v', ['f16']):  # Fallback auf f16 wenn nichts gefunden
                    combo_v.addItem(opt)
                
                # Alte Auswahl wiederherstellen falls noch vorhanden
                if current_text in options.get('v', []):
                    idx = combo_v.findText(current_text)
                    if idx >= 0:
                        combo_v.setCurrentIndex(idx)
                        
        except Exception as e:
            self.debug_text.append(f"Warnung: Konnte cache-type Optionen nicht laden ({e})")
    
    def browse_path(self, line_edit: QLineEdit, start_dir: str):
        dialog = QFileDialog(self, "Pfad wählen", start_dir)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                line_edit.setText(selected_paths[0])
    
    def on_select_benchmark_file(self, line_edit: QLineEdit):
        """Dateidialog für Benchmark File öffnen"""
        from PyQt6.QtWidgets import QFileDialog
        
        dialog = QFileDialog(self, "Benchmark Datei wählen", str(Path.home()))
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(["Alle Dateien (*)"])
        
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                filepath = selected_files[0]
                line_edit.setText(filepath)
                
                # Speichere Dateipfad in Config
                try:
                    config_path = Path.home() / ".llauncher" / "config.json"
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if "benchmark" not in config:
                        config["benchmark"] = {}
                    config["benchmark"]["benchmark_file_path"] = filepath
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                except Exception as e:
                    self.debug_text.append(f"⚠️ Konnte Config nicht speichern: {e}")
                
                # Debug-Output mit Dateiinfo
                try:
                    import subprocess
                    import os
                    
                    file_size = os.path.getsize(filepath)
                    size_str = self._format_file_size(file_size)
                    
                    # file command ausführen
                    try:
                        file_result = subprocess.run(['file', '-b', filepath], capture_output=True, text=True, timeout=2)
                        file_desc = file_result.stdout.strip() if file_result.returncode == 0 else "Unbekannt"
                    except:
                        file_desc = "Nicht verfügbar"
                    
                    # MIME Type
                    try:
                        mime_result = subprocess.run(['file', '-b', '--mime-type', filepath], capture_output=True, text=True, timeout=2)
                        mime_type = mime_result.stdout.strip() if mime_result.returncode == 0 else "Unbekannt"
                    except:
                        mime_type = "Nicht verfügbar"
                    
                    self.debug_text.append(f"✓ Benchmark File: {filepath}")
                    self.debug_text.append(f"  → Size: {size_str}")
                    self.debug_text.append(f"  → file: {file_desc}")
                    self.debug_text.append(f"  → MIME: {mime_type}")
                    self.debug_text.append(f"  → Saved to config.json[benchmark][benchmark_file_path]")
                    
                except Exception as e:
                    self.debug_text.append(f"⚠️ Konnte Dateiinfo nicht lesen: {e}")
    
    def on_clear_benchmark_file(self, line_edit: QLineEdit):
        """Benchmark File Field leeren"""
        line_edit.setText("")
        
        # Lösche Dateipfad aus Config
        try:
            config_path = Path.home() / ".llauncher" / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            if "benchmark" in config and "benchmark_file_path" in config["benchmark"]:
                del config["benchmark"]["benchmark_file_path"]
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            self.debug_text.append(f"⚠️ Konnte Config nicht aktualisieren: {e}")
        
        self.debug_text.append("ℹ Benchmark File: (none)")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Größe in lesbarem Format ausgeben"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}" if unit != 'B' else f"{size_bytes} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def find_models(self) -> list[Path]:
        models_dir = Path(self.model_directory)
        if not models_dir.exists():
            return []
        return [f for f in models_dir.iterdir() if f.suffix == ".gguf"]

    def update_model_dropdown(self):
        self.model_combo.clear()
        models = self.find_models()
        for model in models:
            self.model_combo.addItem(model.name)

    def on_model_selected(self, name: str):
        model_path = (Path(self.model_directory) / name).resolve()
        self.selected_model = str(model_path)
        
        # ctx_size Slider Maximum aus GGUF-Datei lesen
        if model_path.exists() and model_path.is_file():
            ctx_length = read_gguf_context_length(str(model_path))
            if ctx_length and ctx_length > 0:
                slider_data = self.param_sliders["-c"]
                slider = slider_data["slider"]
                edit = slider_data["edit"]
                
                # Slider Maximum setzen (kein Cap, aber realistisch begrenzen)
                slider.setMaximum(ctx_length)
                # Default auf den gelesenen Wert setzen – ABER nur wenn wir nicht gerade
                # externe Parameter von einem laufenden Prozess laden!
                if not getattr(self, 'loading_running_args', False):
                    slider.setValue(ctx_length)
                
                # Edit-Widget Breite aktualisieren für neue maximale Zahl
                max_width = len(str(ctx_length)) * 9 + 15
                edit.setMinimumWidth(max_width)
                edit.setMaximumWidth(max_width)
        
        save_config({
            "model_directory": self.model_directory,
            "selected_model": str(model_path),
        })

    def on_slider_changed(self, param_key: str, value: int):
        pass

    def on_float_slider_changed(self, param_key: str, value: int):
        # Wert vom Slider in Float umwandeln
        float_value = value / 10.0
        if param_key in self.param_sliders and "edit" in self.param_sliders[param_key]:
            self.param_sliders[param_key]["edit"].setText(f"{float_value:.2f}")

    def on_float_edit_changed(self, param_key: str, text: str):
        # Wert vom Edit-Widget zum Slider umwandeln
        if text and param_key in self.param_sliders and "slider" in self.param_sliders[param_key]:
            try:
                float_value = float(text)
                int_value = int(float_value * 10)
                self.param_sliders[param_key]["slider"].setValue(int_value)
            except ValueError:
                pass

    def get_current_args(self) -> list[str]:
        exe_name = self.exe_combo.currentText()
        if exe_name == "llama.cpp nicht gefunden":
            return []

        args = [str(Path(self.llama_cpp_path) / exe_name)]

        # Modell-Pfad (nur einmal!)
        if self.selected_model:
            args.extend(["-m", self.selected_model])

        # mmproj für Vision-Modelle
        mmproj_text = self.mmproj_line.text().strip()
        if mmproj_text:
            mmproj_path = Path(mmproj_text)
            if not mmproj_path.is_absolute():
                mmproj_path = Path(self.model_directory) / mmproj_path
            if mmproj_path.exists():
                args.extend(["--mmproj", str(mmproj_path)])

        # Parameter aus Slidern (nur wenn vom Default abweichen)
        for param_key, config in self.PARAM_DEFINITIONS.items():
            if param_key not in self.param_sliders:
                continue
            slider = self.param_sliders[param_key]
            
            if config.get("type") == "float_slider":
                # Float-Slider: Wert aus Edit-Widget lesen
                value_edit = slider["edit"]
                try:
                    value = float(value_edit.text())
                except ValueError:
                    continue
                
                if abs(value - config["default"]) > 0.01:
                    args.append(param_key)
                    args.append(f"{value:.2f}")
            elif config.get("type") == "combo":
                # ComboBox – Wert als String lesen
                combo = slider["combo"]
                value = combo.currentText()
                # cache-type-k/v immer explizit setzen (nicht nur bei Abweichung vom Default)
                if param_key in ("--cache-type-k", "--cache-type-v"):
                    args.append(param_key)
                    args.append(value)
                elif value != config["default"]:
                    args.append(param_key)
                    args.append(value)
            
            elif config.get("type") in ("text_input", "path_input", "file_input"):
                # Textfeld, Pfad oder Datei-Eingabe – Wert als String lesen
                # WICHTIG: benchmark_file_path NICHT in Command Line (nur für Benchmark)
                if param_key == "benchmark_file_path":
                    continue
                
                text_edit = slider["edit"]
                value = text_edit.text()
                if value and value != config["default"]:
                    args.append(param_key)
                    args.append(value)
            
            else:
                # Integer-Slider
                if isinstance(slider, dict):
                    # Priorität: Wert aus dem Edit-Feld lesen, falls vorhanden
                    edit_widget = slider.get("edit")
                    slider_widget = slider.get("slider")
                    
                    if edit_widget:
                        try:
                            value = int(edit_widget.text())
                        except ValueError:
                            # Fallback auf Slider-Widget, falls Edit leer/ungültig
                            value = slider_widget.value() if slider_widget else 0
                    elif slider_widget:
                        value = slider_widget.value()
                    else:
                        value = 0
                else:
                    value = slider.value()
                
                # Sonderfall: -ngl mit "all" Checkbox
                if param_key == "-ngl":
                    if hasattr(self, "ngl_all_checkbox") and self.ngl_all_checkbox.isChecked():
                        args.append(param_key)
                        args.append("all")
                    elif value != config["default"]:
                        args.append(param_key)
                        args.append(str(value))
                elif value != config["default"]:
                    args.append(param_key)
                    args.append(str(value))

        return args
    
    def build_full_command(self) -> str:
        """Vollständige Kommandozeile als String bauen (1:1 wie ausgeführt)
        
        Priorität:
        1. Echte Kommandozeile vom externen Runner (falls vorhanden)
        2. Echte Kommandozeile vom ProcessRunner (falls vorhanden)
        3. UI-Werte zusammengesetzt (Fallback)
        """
        # 1. Versuche externe Runner args (für externe Prozesse)
        if hasattr(self, 'external_runner_args') and self.external_runner_args:
            return " ".join(shlex.quote(arg) for arg in self.external_runner_args)
        
        # 2. Versuche ProcessRunner (für interne Prozesse)
        if hasattr(self, 'process_runner') and self.process_runner:
            try:
                real_args = self.process_runner.get_args_from_proc()
                if real_args:
                    return " ".join(shlex.quote(arg) for arg in real_args)
            except Exception:
                pass
        
        # 3. Fallback: UI-Werte zusammengesetzt
        exe_name = self.exe_combo.currentText()
        if exe_name == "llama.cpp nicht gefunden":
            return "# llama.cpp nicht gefunden"
        
        args = [str(Path(self.llama_cpp_path) / exe_name)]
        
        # Modell-Pfad (nur einmal!)
        if self.selected_model:
            args.extend(["-m", self.selected_model])
        
        # mmproj für Vision-Modelle
        mmproj_text = self.mmproj_line.text().strip()
        if mmproj_text:
            mmproj_path = Path(mmproj_text)
            if not mmproj_path.is_absolute():
                mmproj_path = Path(self.model_directory) / mmproj_path
            if mmproj_path.exists():
                args.extend(["--mmproj", str(mmproj_path)])
        
        # Parameter aus Slidern (alle Werte, nicht nur abweichende!)
        for param_key, config in self.PARAM_DEFINITIONS.items():
            if param_key not in self.param_sliders:
                continue
            
            slider = self.param_sliders[param_key]
            
            try:
                if config.get("type") == "float_slider":
                    # Float-Slider: Wert aus Edit-Widget lesen
                    value_edit = slider["edit"]
                    try:
                        value = float(value_edit.text())
                    except ValueError:
                        continue
                    
                    args.append(param_key)
                    args.append(f"{value:.2f}")
                
                elif config.get("type") == "combo":
                    # ComboBox – Wert als String lesen
                    combo = slider["combo"]
                    value = combo.currentText()
                    # cache-type-k/v immer explizit setzen (nicht nur bei Abweichung vom Default)
                    if param_key in ("--cache-type-k", "--cache-type-v"):
                        args.append(param_key)
                        args.append(value)
                    elif value != config["default"]:
                        args.append(param_key)
                        args.append(value)
                
                elif config.get("type") in ("text_input", "path_input", "file_input"):
                    # Textfeld, Pfad oder Datei-Eingabe – Wert als String lesen
                    # WICHTIG: benchmark_file_path NICHT in Command Line (nur für Benchmark)
                    if param_key == "benchmark_file_path":
                        continue
                    
                    text_edit = slider["edit"]
                    value = text_edit.text()
                    if value:
                        args.append(param_key)
                        args.append(value)
                
                else:
                    # Integer-Slider
                    if isinstance(slider, dict):
                        # Priorität: Wert aus dem Edit-Feld lesen, falls vorhanden
                        edit_widget = slider.get("edit")
                        slider_widget = slider.get("slider")
                        
                        if edit_widget:
                            try:
                                value = int(edit_widget.text())
                            except ValueError:
                                # Fallback auf Slider-Widget, falls Edit leer/ungültig
                                value = slider_widget.value() if slider_widget else 0
                        elif slider_widget:
                            value = slider_widget.value()
                        else:
                            value = 0
                    else:
                        value = slider.value()
                    
                    # Sonderfall: -ngl mit "all" Checkbox
                    if param_key == "-ngl":
                        if hasattr(self, "ngl_all_checkbox") and self.ngl_all_checkbox.isChecked():
                            args.append(param_key)
                            args.append("all")
                        else:
                            args.append(param_key)
                            args.append(str(value))
                    else:
                        args.append(param_key)
                        args.append(str(value))
            except (KeyError, AttributeError, TypeError):
                # Slider noch nicht initialisiert oder ungültig - überspringen
                continue
        
        # Externe Parameter hinzufügen (nicht in APP verwaltbar)
        if self.external_args:
            for key, value in self.external_args.items():
                args.append(key)
                if isinstance(value, bool):
                    if value:  # Boolean flags nur anzeigen wenn True
                        pass  # Kein Wert nach dem Flag
                else:
                    args.append(str(value))
        
        return " ".join(shlex.quote(arg) for arg in args)

    def on_param_changed(self):
        """Debug-Output live aktualisieren wenn sich ein Parameter ändert"""
        try:
            # Prüfen ob param_sliders initialisiert ist (kann None sein während init_ui)
            if not hasattr(self, 'param_sliders') or self.param_sliders is None:
                return
            
            # Alle Sliders müssen existieren und initialisiert sein
            for param_key in self.PARAM_DEFINITIONS.keys():
                if param_key not in self.param_sliders:
                    continue
                slider = self.param_sliders[param_key]
                config = self.PARAM_DEFINITIONS[param_key]
                
                try:
                    if config.get("type") == "float_slider":
                        if "edit" not in slider or not slider["edit"]:
                            return
                    elif config.get("type") in ("text_input", "path_input"):
                        if "edit" not in slider or not slider["edit"]:
                            return
                    else:
                        if isinstance(slider, dict) and "slider" not in slider:
                            return
                except (KeyError, AttributeError):
                    return
            
            command = self.build_full_command()
            self.debug_text.setText(command)
        except Exception as e:
            self.debug_text.setText(f"Fehler beim Aktualisieren: {e}")

    def on_benchmark_finished(self, tps, token_count):
        """Handle benchmark completion."""
        self.debug_text.append(f"\n[BENCHMARK FINISHED] TPS: {tps:.2f}, Tokens: {token_count}")
        
        # Disable cancel button
        if hasattr(self, 'cancel_bench_btn'):
            self.cancel_bench_btn.setEnabled(False)
        
        # Reset status
        self.status_label.setText(gettext("status_ready"))
        
         # Save benchmark result using preset_manager helper
        ask_quality_and_save_benchmark(
            self,
            self.debug_text,
            self.status_label,
            tps,
            token_count,
            self._last_benchmark_command
        )
        # Progress bar stays visible - will be reset by check_existing_process()
    
    def on_benchmark_token_update(self, token_count: int):
        """Update progress bar with current token count."""
        if hasattr(self, 'bench_progress_bar'):
            # Use a reasonable max value based on typical benchmark token counts
            # Set max to 1024 tokens, clamp display to avoid overly long bars
            max_tokens = 1024
            display_value = min(token_count, max_tokens)
            self.bench_progress_bar.setValue(display_value)
            # Show token count as tooltip
            self.bench_progress_bar.setToolTip(f"Tokens: {token_count}")
    
    def toggle_process(self):
        if hasattr(self, 'external_runner_pid') and self.external_runner_pid:
            # Externer Prozess stoppen über terminate_by_pid (SIGINT×2 → SIGTERM → SIGKILL)
            self.start_stop_btn.setText(gettext("btn_stop"))
            self.status_label.setText(gettext("status_stopping"))
            
            stopped = ProcessRunner.terminate_by_pid(self.external_runner_pid)

            if not stopped:

                self.status_label.setText(gettext("status_failed"))
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                self.start_stop_btn.setText(gettext("btn_start"))
                self.start_stop_btn.setObjectName("")
            else:
                self.external_runner_pid = None
                self.external_runner_args = None
                
                self.status_label.setText(gettext("status_stopped"))
                self.status_label.setStyleSheet("")
                self.start_stop_btn.setText(gettext("btn_start"))
                self.start_stop_btn.setObjectName("")
            # Reset progress bar to 0% after stopping external process (SIGINT×2 → SIGTERM → SIGKILL)
            if hasattr(self, 'bench_progress_bar'):
                self.bench_progress_bar.setValue(0)
                self.bench_progress_bar.setToolTip("Stopped")
        elif self.runner and self.runner.isRunning():
            # Eigener Prozess stoppen - Button auf "Stoppe..." setzen, Status auch
            self.start_stop_btn.setText(gettext("btn_stop"))
            self.status_label.setText(gettext("status_stopping"))

            # Prozess stoppen (SIGINT → SIGINT → SIGTERM → SIGKILL)
            stopped = self.runner.terminate_process()
            
            # QThread warten bis er fertig ist (max 3 Sekunden für alle Signals)
            start_time = time.time()
            while self.runner.isRunning() and (time.time() - start_time) < 3:
                time.sleep(0.1)
            
            # Runner auf None setzen damit nächstes Starten korrekt funktioniert
            self.runner = None
            
            # Status nach erfolgreichem Stopp auf "Gestoppt" setzen, Button zurücksetzen
            self.status_label.setText(gettext("status_stopped"))
            self.status_label.setStyleSheet("")  # CSS reset
            self.start_stop_btn.setText(gettext("btn_start"))
            self.start_stop_btn.setObjectName("")
            # Reset progress bar to 0% after stopping internal process
            if hasattr(self, 'bench_progress_bar'):
                self.bench_progress_bar.setValue(0)
                self.bench_progress_bar.setToolTip("Stopped")
        else:
            # Starten - Status zuerst auf "Lade Modell..." setzen
            print(f"[DEBUG toggle_process] Starting process...")
            self.start_stop_btn.setText(gettext("btn_stop"))
            self.start_stop_btn.setObjectName("StartButton")
            self.status_label.setText(gettext("status_loading_model"))
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")

            args = self.get_current_args()
            print(f"[DEBUG toggle_process] Args: {args}")
            
            if not args or "-m" not in args:
                print(f"[DEBUG toggle_process] ERROR: No model selected!")
                QMessageBox.warning(self, gettext("msg_no_model_selected"))
                self.start_stop_btn.setText(gettext("btn_start"))
                self.start_stop_btn.setObjectName("StopButton")
                self.status_label.setText("")
                return
            
            # Status auf "Fehlgeschlagen" setzen wenn Prozess fehlschlägt
            def on_process_finished(exit_code):
                print(f"[DEBUG on_process_finished] Exit code: {exit_code}")
                if exit_code != 0:
                    self.status_label.setText(gettext("status_failed"))
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")
                    # Button zurücksetzen auf "Start"
                    self.start_stop_btn.setText(gettext("btn_start"))
                    self.start_stop_btn.setObjectName("")
                    # Wichtig: PID auf None setzen damit nächster Klick neu starten kann
                    if hasattr(self, 'external_runner_pid') and self.external_runner_pid:
                        self.external_runner_pid = None
                    if hasattr(self, 'runner') and self.runner:
                        self.runner = None
            
            # Output überwachen für "all slots are idle" Signal
           # Output überwachen für "all slots are idle" Signal
            
            # Initialize idle state flag
            if not hasattr(self, '_was_idle'):
                self._was_idle = False
            
            def on_output(line):
                # Parse progress from llama.cpp output: "prompt processing progress, n_tokens = 52544, batch.n_tokens = 32, progress = 0.999125"
                progress_match = re.search(r'progress\s*=\s*([0-9.]+)', line)
                if progress_match and hasattr(window, 'bench_progress_bar'):
                    try:
                        progress = float(progress_match.group(1))
                        # Update progress bar (0-1 scale)
                        window.bench_progress_bar.setValue(int(progress * 100))
                        window.bench_progress_bar.setToolTip(f"Progress: {progress:.2%}")
                    except (ValueError, TypeError):
                        pass
                
                if "all slots are idle" in line and not getattr(self, 'benchmark_running', False):
                    self.status_label.setText(gettext("status_idle"))
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    self._was_idle = True  # Mark that we're in idle state
                    # Set progress bar to 100% when all slots are idle
                    if hasattr(self, 'bench_progress_bar'):
                        self.bench_progress_bar.setValue(100)
                        self.bench_progress_bar.setToolTip("")
                
                # Any output after idle means we're active again
                elif line.strip() and getattr(self, '_was_idle', False):
                    self.status_label.setText(gettext("status_running"))
                    self.status_label.setStyleSheet("color: green; font-weight: bold;")
                    self._was_idle = False
                
                self.debug_text.append(line)
            
            # Prozess starten
            workdir = str(Path(self.llama_cpp_path))
            self.runner = ProcessRunner(args, workdir)
            self.runner.output_signal.connect(on_output)
            self.runner.finished_signal.connect(on_process_finished)
            self.runner.start()

    def on_benchmark_output(self, text: str):
        """Handles benchmark output to prevent line-break issues in streaming mode."""
        if not text:
            return
        # Remove Think blocks and generic XML tags for clean streaming output
        cleaned_text = re.sub(r'</think>', ' ', text)
        cleaned_text = re.sub(r'<\w+>', ' ', cleaned_text)
        # Accumulate text without adding newlines between chunks
        # The benchmark thread already provides properly-formatted output
        self.debug_text.moveCursor(self.debug_text.textCursor().MoveOperation.End)
        self.debug_text.insertPlainText(cleaned_text)
        # Ensure scroll to bottom
        self.debug_text.ensureCursorVisible()

    def _get_free_gpu_memory(self) -> int:
        """Return free GPU memory in MB using nvidia-smi."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            free_mb = int(out.strip().splitlines()[0])
            return max(free_mb, 0)
        except Exception:
            # Fallback: assume a safe amount
            return 2048

    def copy_debug(self):
        self.debug_text.selectAll()
        self.debug_text.copy()

    def check_existing_process(self):
        """Prüft ob bereits ein llama-server läuft und passt UI entsprechend an."""
        import shlex
        
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                # Kein Prozess läuft - UI zurücksetzen, aber Progress Bar auf 0% lassen
                self.status_label.setText(gettext("status_ready"))
                self.status_label.setStyleSheet("")
                self.start_stop_btn.setText(gettext("btn_start"))
                self.start_stop_btn.setObjectName("StartButton")
                # Progress bar bleibt bei 0% - wird durch toggle_process() nach Stop gesetzt
                return
            
            pids = [int(pid) for pid in result.stdout.strip().split() if pid.isdigit()]
            
            # Prüfen ob einer der Prozesse vom User gestartet wurde (eigener Prozess)
            for pid in pids:
                cmdline_path = f'/proc/{pid}/cmdline'
                try:
                    with open(cmdline_path, 'r') as f:
                        content = f.read()
                    
                    args = [arg for arg in content.split('\x00') if arg]
                    if not args or 'llama-server' not in args[0]:
                        continue
                    
                    # Kommandozeile zusammenbauen
                    full_cmd = " ".join(shlex.quote(arg) for arg in args)
                    
                    # UI anpassen: Button auf "Stop" setzen
                    # Aber nicht wenn gerade ein Benchmark läuft!
                    if not getattr(self, 'benchmark_running', False):
                        # Nur Button-Status aktualisieren, nicht den Label-Text
                        # (on_output() verwaltet Idle/Running Status korrekt)
                        self.start_stop_btn.setText(gettext("btn_stop"))
                        self.start_stop_btn.setObjectName("StopButton")
                    
                    # Runner als "externer" Prozess markieren
                    # Wir speichern PID und args, können aber nicht über QThread steuern
                    self.external_runner_pid = pid
                    self.external_runner_args = args
                    
                    return  # Nur erster laufender Prozess relevant
                
                except (FileNotFoundError, PermissionError, IOError):
                    continue
        
        except Exception as e:
            pass
    
    def show_bench_context_menu(self, pos):
        """Zeigt Kontextmenü für Benchmark-Tabelle mit 'Kopieren' Option."""
        from PyQt6.QtGui import QClipboard
        
        index = self.bench_table.indexAt(pos)
        if not index.isValid():
            return
        
        row = index.row()
        col = index.column()
        
        # Nur 4. Spalte (Kommandozeile) ist kopierbar
        if col != 3:
            return
        
        command_item = self.bench_table.item(row, 3)
        command = command_item.text()
        
        # Ins Clipboard kopieren
        clipboard = QApplication.clipboard()
        clipboard.setText(command)
        
        # Visuelles Feedback: Zelle kurz weiß umranden
        from PyQt6.QtGui import QBrush, QColor
        
        original_brush = command_item.background()
        
        # Temporären Style für die Zelle setzen (weißer Hintergrund)
        brush = QBrush(QColor(255, 255, 255))
        command_item.setBackground(brush)
        
        # Feedback nach 300ms zurücksetzen
        QTimer.singleShot(300, lambda: command_item.setBackground(original_brush))

    # ----------------------------------------------------------------------
    # Helper: is there a running llama process that can be reused?
    # ----------------------------------------------------------------------
    def _model_is_already_loaded(self) -> bool:
        """Returns True if the ProcessRunner thread is alive and its command line contains the same executable we would use for a fresh run. This tells us whether we can skip the "-m" flag in the benchmark."""
        runner = getattr(self, "runner", None)
        if runner is None or not runner.isRunning():
            return False
        cmd = getattr(runner, "cmd", "")
        exe_path = str(Path(self.llama_cpp_path) / self.exe_combo.currentText())
       # Simple check - if the executable part matches we assume it's the same run.
        return exe_path in cmd
    
    def _append_text_inline(self, text: str):
        """Append text to debug_text - handles both inline and multi-line content."""
        # Debug: Log all received signals
        import sys
  
        
        # Use append for reliability with QTextEdit
        self.debug_text.append(text)

    def run_benchmark_streaming(self):
        """Run HTTP-based benchmark in streaming mode for live display."""
        
        # Build and store command for benchmark completion handler
        self._last_benchmark_command = self.build_full_command()
        
        # Enable cancel button during benchmark
        if hasattr(self, 'cancel_bench_btn'):
            self.cancel_bench_btn.setEnabled(True)
        
        # Ensure we don't have a stale benchmark thread
        bench_thread = getattr(self, 'bench_thread', None)
        if bench_thread and bench_thread.isRunning():
            QMessageBox.warning(self, "Fehler", "Ein Benchmark läuft bereits.")
            return
        
        self.debug_text.clear()
        self.status_label.setText("Benchmark (Live) läuft...")
        
       # Start GPU monitoring during benchmark (if not already running)
        if not hasattr(self, 'gpu_monitor') or not self.gpu_monitor.isRunning():
            self.gpu_monitor = GPUMonitor()
            self.gpu_monitor.gpu_update.connect(self.update_gpu_display)
            self.gpu_monitor.start()
        
     # Get max_tokens from -n slider (default 64)
        n_slider_data = self.param_sliders.get("-n")
        if n_slider_data and isinstance(n_slider_data, dict) and "slider" in n_slider_data:
            max_tokens = n_slider_data["slider"].value()
        else:
            max_tokens = 64
        
        # Import benchmark runner and create thread
        from http_benchmark_thread import HTTPBenchmarkRunner
        
        self.bench_thread = HTTPBenchmarkRunner(
            max_tokens=max_tokens,
            server_pid=self.external_runner_pid,
            streaming=True,
            model_path=self.selected_model
        )
        self.bench_thread.output_signal.connect(self.on_benchmark_output)
        self.bench_thread.status_signal.connect(self.status_label.setText)
        self.bench_thread.finished_signal.connect(self.on_benchmark_finished)
        self.bench_thread.token_update_signal.connect(self.on_benchmark_token_update)
        self.bench_thread.start()

    def cancel_benchmark(self):
        """Cancel the currently running benchmark."""
        print("[DEBUG] cancel_benchmark() called!")  # Terminal output for debugging
        
        bench_thread = getattr(self, 'bench_thread', None)
        if not bench_thread:
            self.debug_text.append("ERROR: No benchmark thread found to cancel!")
            return
        
        # Signal cancellation to the thread - call cancel() method directly
        self.debug_text.append(f"Cancelling benchmark... (thread={bench_thread})")
        
        if hasattr(bench_thread, '_cancelled') and hasattr(bench_thread, 'cancel'):
            bench_thread._cancelled = True
            # Call cancel() which writes to pipe AND closes socket
            bench_thread.cancel()
            self.debug_text.append("Cancel signal sent!")
        else:
            self.debug_text.append(f"WARNING: Thread missing _cancelled or cancel method (has: {dir(bench_thread)})")
        
        if hasattr(self, 'cancel_bench_btn'):
            self.cancel_bench_btn.setEnabled(False)

    def edit_prompt_dialog(self):
        """Show dialog to edit benchmark prompt."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox
        from PyQt6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Benchmark Prompt")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        info_label = QLabel("Edit the prompt for benchmarks. This will be saved to config.json:")
        layout.addWidget(info_label)
        
        config_path = Path.home() / ".llauncher" / "config.json"
        current_prompt = ""
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                current_prompt = config.get("benchmark", {}).get("prompt", "")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not read config: {e}")
        
        prompt_edit = QTextEdit()
        prompt_edit.setPlainText(current_prompt)
        prompt_edit.setPlaceholderText("Enter benchmark prompt here...")
        layout.addWidget(prompt_edit)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        
        def on_save():
            new_prompt = prompt_edit.toPlainText()
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if "benchmark" not in config:
                    config["benchmark"] = {}
                config["benchmark"]["prompt"] = new_prompt
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save config: {e}")
        
        save_btn.clicked.connect(on_save)
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()

    def run_benchmark(self):
        """Run HTTP-based benchmark in standard mode."""
        
        # Build and store command for benchmark completion handler
        self._last_benchmark_command = self.build_full_command()
        
        # Enable cancel button during benchmark
        if hasattr(self, 'cancel_bench_btn'):
            self.cancel_bench_btn.setEnabled(True)
        
     # Get max_tokens from -n slider (default 64)
        n_slider_data = self.param_sliders.get("-n")
        if n_slider_data and isinstance(n_slider_data, dict) and "slider" in n_slider_data:
            max_tokens = n_slider_data["slider"].value()
        else:
            max_tokens = 64
        
        # Import benchmark runner and create thread
        from http_benchmark_thread import HTTPBenchmarkRunner
        
        self.bench_thread = HTTPBenchmarkRunner(
            max_tokens=max_tokens,
            server_pid=self.external_runner_pid,
            streaming=False,
            model_path=self.selected_model
         )
        self.bench_thread.output_signal.connect(self.on_benchmark_output)
        self.bench_thread.status_signal.connect(self.status_label.setText)
        self.bench_thread.finished_signal.connect(self.on_benchmark_finished)
        self.bench_thread.token_update_signal.connect(self.on_benchmark_token_update)
        self.bench_thread.token_update_signal.connect(self.on_benchmark_token_update)
        self.bench_thread.start()

    def _get_model_info(self) -> str:
        """Ermittelt Modellname + Parameter für Benchmark-Tabelle."""
        # 1. Prüfe ob ProcessRunner existiert (neuer Prozess gestartet)
        if self.runner and hasattr(self.runner, 'args') and self.runner.args:
            args = self.runner.args
            model_path = None
            params = []
            
            for i, arg in enumerate(args):
                if arg == "-m" and i + 1 < len(args):
                    model_path = Path(args[i + 1]).name
                elif arg.startswith("-") or arg.startswith("--"):
                    param_name = arg
                    param_value = None
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        param_value = args[i + 1]
                    
                    if param_value:
                        params.append(f"{param_name} {param_value}")
                    else:
                        params.append(param_name)
            
            if model_path:
                return f"{model_path} {' '.join(params)}"
        
        # 2. Prüfe ob Server bereits läuft (HTTP-Ping)
        server_info = fetch_running_model_info()
        if server_info and server_info.get("model_name"):
            params = server_info["params"]
            param_strs = []
            for key, value in params.items():
                param_strs.append(f"--{key} {value}")
            
            return f"{server_info['model_name']} {' '.join(param_strs)}"
        
        # 3. Fallback: UI-Werte (noch nicht geladen)
        model_name = (
            self.model_combo.currentText()
            if self.model_combo.currentText()
            else "Manuell"
        )
        
        args = self.get_current_args()
        params = []
        for i, arg in enumerate(args):
            if arg.startswith("-") or arg.startswith("--"):
                param_name = arg
                param_value = None
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    param_value = args[i + 1]
                
                if param_value:
                    params.append(f"{param_name} {param_value}")
                else:
                    params.append(param_name)
        
        return f"{model_name} {' '.join(params)}"

    def _get_running_server_command(self) -> Optional[str]:
        """Liest Kommandozeile von laufendem llama-server Prozess aus /proc."""
        import shlex
        
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                return None
            
            pids = [int(pid) for pid in result.stdout.strip().split() if pid.isdigit()]
            
            for pid in pids:
                cmdline_path = f'/proc/{pid}/cmdline'
                try:
                    with open(cmdline_path, 'r') as f:
                        content = f.read()
                    
                    args = [arg for arg in content.split('\x00') if arg]
                    if not args or 'llama-server' not in args[0]:
                        continue
                    
                    # Vollständige Kommandozeile zusammenbauen
                    full_cmd = " ".join(shlex.quote(arg) for arg in args)
                    return full_cmd
                
                except (FileNotFoundError, PermissionError, IOError):
                    continue
            
            return None
        
        except Exception:
            return None
    
    def _finalize_benchmark(self, tps: float = 0.0, tokens: int = 0):
        self.status_label.setText(gettext("status_ready"))
        
        # Zuerst prüfen ob ein Server läuft und dessen Kommandozeile nehmen
        full_command = self._get_running_server_command()
        
        if not full_command:
            # Fallback auf UI-Werte wenn kein Server läuft
            full_command = self.build_full_command()
        
        ask_quality_and_save_benchmark(
            self,
            self.debug_text,
            self.status_label,
            tps,   # actual TPS from benchmark
            str(tokens), # actual token count
            full_command,
        )

    # save_benchmark jetzt in preset_manager.py ausgelagert
    
    def save_preset(self):
        result = show_preset_save_dialog(
            self,
            self.param_sliders,
            self.PARAM_DEFINITIONS,
            self.llama_cpp_path,
            self.model_directory,
            self.selected_model,
            self.mmproj_line,
        )
        
        if not result:
            return
            
        name, preset = result
        
        if not name or not preset:
            return
        
        # Speichern und Bestätigung
        presets = load_presets()
        presets[name] = preset
        save_presets(presets)

        QMessageBox.information(self, gettext("msg_preset_saved_title"), gettext("msg_preset_saved").format(name=name))

    def load_preset_dialog(self):
        name, preset = show_preset_load_dialog(
            self,
            self.param_sliders,
            self.PARAM_DEFINITIONS,
            self.llama_cpp_path,
            self.model_directory,
            self.selected_model,
            self.mmproj_line,
            self.exe_combo,
        )
        
        if not name:
            return
        
        # Preset anwenden und Kommandozeile anzeigen
        apply_preset(self, preset)
        preset_show_args(
            self,
            self.debug_text,
            name,
            preset,
            self.param_sliders,
            self.PARAM_DEFINITIONS,
            self.llama_cpp_path,
            self.model_directory,
            self.selected_model,
            self.mmproj_line,
            self.exe_combo,
        )

    # show_preset_args jetzt in preset_manager.py ausgelagert

    def load_config(self):
        config = load_config()
        if config:
            exe_path = config.get("llama_cpp_path")
            model_dir = config.get("model_directory")
            selected_model = config.get("selected_model")
            # Backward Compatibility: sowohl alte als auch neue Key unterstützen
            selected_exec = config.get("selected_executable") or config.get("selected_exe")
            
            # Theme loading
            self.light_theme = config.get("theme") == "light"
            if hasattr(self, 'light_theme_checkbox'):
                self.light_theme_checkbox.setChecked(self.light_theme)
            
            if exe_path and Path(exe_path).exists():
                self.llama_cpp_path = exe_path
                self.exe_line.setText(exe_path)

            if model_dir and Path(model_dir).exists():
                self.model_directory = model_dir
                self.model_line.setText(model_dir)
            if selected_exec:
                idx = self.exe_combo.findText(selected_exec)
                if idx >= 0:
                    self.exe_combo.setCurrentIndex(idx)
                    # Cache-Type Optionen direkt aktualisieren (Signal vielleicht nicht ausgelöst)
                    exe_full_path = Path(self.llama_cpp_path) / selected_exec
                    self.debug_text.append(f"=== Config-Load: Binary {selected_exec!r} ===")
                    self.debug_text.append(f"Pfad: {str(exe_full_path)}")
                    if '/' not in selected_exec and '\\' not in selected_exec:
                        self.update_cache_type_options(str(exe_full_path))
                    else:
                        self.debug_text.append("⚠️ Skipping --help (ungültiger Name)")

            if selected_model:
                self.selected_model = selected_model

    def apply_presets(self):
        presets = load_presets()
        if presets:
            last_preset = list(presets.values())[-1]
            apply_preset(self, last_preset)
            
            # Cache-Type Optionen nach Preset-Anwendung aktualisieren
            selected_exec = last_preset.get("selected_executable") or last_preset.get("selected_exe")
            if selected_exec and '/' not in str(selected_exec) and '\\\\' not in str(selected_exec):
                exe_full_path = Path(self.llama_cpp_path) / selected_exec
                self.debug_text.append(f"=== Preset-Apply: Binary {selected_exec!r} ===")
                self.update_cache_type_options(str(exe_full_path))

    def restore_geometry(self):
        """Fenster-Position, Größe und Splitter-State laden"""
        config = load_config()
        
        # Fenster-Position & Größe laden (explizit als Integer)
        x = config.get('window_x')
        y = config.get('window_y')
        width = config.get('window_width')
        height = config.get('window_height')
        
        if all(v is not None for v in [x, y, width, height]):
            try:
       
                self.move(x, y)
                self.resize(width, height)
          
            except Exception as e:
                pass
        else:
          
            # Fallback auf alte Methode wenn keine expliziten Werte da sind
            geom_data = config.get('window_geometry')
            if geom_data:
                try:
                    self.restoreGeometry(bytes(geom_data, 'ascii'))
                
                except Exception as e:
                    pass
        
# Splitter-Position laden (als Integer-Liste, nicht Binary-State)
        if hasattr(self, 'splitter'):
            sizes_data = config.get('splitter_sizes')
            if sizes_data and isinstance(sizes_data, list):
                try:
                    self.splitter.setSizes(sizes_data)
                except Exception as e:
                
                  
                    self.splitter.setSizes([self.width() * 0.6, self.width() * 0.4])

               

    def resizeEvent(self, event):
        """Speichert Fenster-Geometrie bei jeder Größenänderung"""
        super().resizeEvent(event)
        
        # Nur Geometrie speichern (Splitter-State zu aggressiv für resizeEvent)
        config = load_config()
        config['window_x'] = self.x()
        config['window_y'] = self.y()
        config['window_width'] = self.width()
        config['window_height'] = self.height()
        
        save_config(config)

    def closeEvent(self, event):
        """Fenster-Geometrie und Splitter-State speichern + Timer stoppen"""
        config = load_config()
        
        # Explizit Breite, Höhe, x, y speichern (robuster als saveGeometry)
        config['window_x'] = self.x()
        config['window_y'] = self.y()
        config['window_width'] = self.width()
        config['window_height'] = self.height()
        
        # Splitter-Position speichern (als Liste von Ints, nicht Binary-State)
        if hasattr(self, 'splitter'):
            try:
                sizes_before = self.splitter.sizes()
                config['splitter_sizes'] = list(sizes_before)
            except Exception as e:
                pass
        else:
            pass
        
        # QThread detachten, damit Prozess weiterlaufen kann
        if hasattr(self, 'runner') and self.runner:
            if self.runner.isRunning():
                # Force thread to exit by clearing process reference
                self.runner.force_exit()
                # Clear reference so Qt can destroy the QThread object
                self.runner = None
        
        save_config(config)
        
        # Nur Timer stoppen
        if hasattr(self, 'process_check_timer'):
            self.process_check_timer.stop()
        
        event.accept()


# storage-Imports nur noch für load_config und apply_presets (in Methoden inline)


if __name__ == "__main__":
    # Initialize i18n before creating any widgets
    from i18n import I18nManager
    
    i18n = I18nManager.get_instance()
    
   # Auto-detect language if missing and write it to config
    from i18n_util import ensure_language, DEFAULT_LANG
    lang_code = ensure_language()
    print(f"[i18n] Config has 'language': {lang_code}")

    if not i18n.load_language(lang_code):
        i18n.load_language(DEFAULT_LANG)
        lang_code = DEFAULT_LANG
    
    print(f"[i18n] Active language: {lang_code}")
    print(f"[i18n] Available languages: {i18n.get_available_languages()}")
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = llauncher()
    window.show()

    sys.exit(app.exec())
