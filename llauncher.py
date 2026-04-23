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
from gguf_utils import get_cpu_count, read_gguf_context_length, get_model_info, format_size, read_gguf_tensor_count
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
from settings_dialog import SettingsDialog
from fork_manager import ForkManagerDialog
from hf_download_dialog import HfDownloadDialog

# Import i18n for lazy gettext() loading
from i18n import I18nManager

from command_builder import get_current_args, build_full_command, on_param_changed
from ui_builder import build_llauncher_ui, setup_timers_and_load
from model_inspector import on_model_selected
from process_signals import start_gpu_monitor, get_free_gpu_memory
from status_manager import update_status, handle_process_error, reset_progress_bar
from ui_helpers import append_text_to_widget, _append_text_inline, _format_file_size



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


from params import get_param_definitions
from PyQt6.QtCore import Qt, pyqtSignal, QDate, QTime, QEvent, QTimer
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
        self.PARAM_DEFINITIONS = get_param_definitions()

        
 
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
        
        # Update progress bar colors to match theme
        if hasattr(self, 'bench_progress_bar'):
            bar_bg = "#e0e0e0" if use_light else "#2d2d2d"
            chunk_start = "#4CAF50" if use_light else "#66bb6a"
            chunk_end = "#8BC34A" if use_light else "#81c784"
            self.bench_progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #999999;
                    border-radius: 4px;
                    text-align: center;
                    background: {bar_bg};
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,
                                                stop:0 {chunk_start}, stop:1 {chunk_end});
                    border-radius: 3px;
                }}
            """)
    
    def show_settings_dialog(self):
        """Show settings dialog for theme and language."""
        # Get current config
        try:
            with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                import json
                config = json.load(f)
            use_light = config.get("theme") == "light"
            lang = config.get("language", "en")
        except Exception:
            use_light = False
            lang = "en"
        
        def reload_language(new_lang):
                """Callback to reload language immediately."""
                from i18n import I18nManager, gettext
                I18nManager.get_instance().reload(new_lang)
                self.status_label.setText(gettext("status_ready"))
        
        dialog = SettingsDialog(self, use_light, lang)
        if dialog.exec() == 1:  # QDialog.DialogCode.Accepted
            new_light, new_lang = dialog.get_settings()
            
            # Save settings to config
            try:
                with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                    config = json.load(f)
                config["theme"] = "light" if new_light else "dark"
                config["language"] = new_lang
                with open(Path.home() / ".llauncher" / "config.json", 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception:
                pass  # Config-Saving-Fehler ignoriert
            
            # Restart app if language changed (to load new language from config)
            if dialog.restart_on_language_change:
                self.close()
                import os
                import sys
                os.execlp(sys.executable, sys.executable, *sys.argv[1:])
            
            # Apply theme
            self.apply_theme(new_light)
            
            # Save settings
            try:
                with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                    config = json.load(f)
                config["theme"] = "light" if new_light else "dark"
                config["language"] = new_lang
                with open(Path.home() / ".llauncher" / "config.json", 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception:
                pass


    def show_fork_dialog(self):
        """Show fork manager dialog for cloning llama.cpp repos."""
        # Read current theme from config (same pattern as show_settings_dialog)
        try:
            with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                import json as _json
                config = _json.load(f)
            use_light = config.get("theme") == "light"
        except Exception:
            use_light = False

        dialog = ForkManagerDialog(parent=self, current_light_theme=use_light)
        dialog.exec()

    def show_hf_download_dialog(self):
        """Show Hugging Face download dialog."""
        try:
            with open(Path.home() / ".llauncher" / "config.json", 'r') as f:
                import json as _json
                config = _json.load(f)
            use_light = config.get("theme") == "light"
        except Exception:
            use_light = False

        dialog = HfDownloadDialog(parent=self, current_light_theme=use_light)
        dialog.exec()

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
                # Suche nach model_name im UserRole (nicht im Display-Text mit Größenangabe!)
                found_index = -1
                for i in range(self.model_combo.count()):
                    user_data = self.model_combo.itemData(i, role=Qt.ItemDataRole.UserRole)
                    if user_data and user_data == model_name:
                        found_index = i
                        break
                
                if found_index >= 0:
                    self.model_combo.setCurrentIndex(found_index)
                else:
                    # Falls Datei nicht in ComboBox ist, direkt den Dateinamen setzen (ohne Größe)
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
                self.debug_text.append(f"ℹ model_directory geändert: {path}")
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
                    size_str = _format_file_size(file_size)
                    
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
        """Repopulate the model dropdown without triggering on_model_selected()."""
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            models = self.find_models()
            for model in models:
                # Get file size via stat (cheaper than full GGUF parse)
                try:
                    file_size = model.stat().st_size
                    size_str = f" ({format_size(file_size)})"
                except Exception:
                    size_str = ""
                # Display name + size, but only store the base name for presets
                display_text = f"{model.name}{size_str}"
                self.model_combo.addItem(display_text)
                # Store clean filename as user data (for preset saving/loading)
                self.model_combo.setItemData(self.model_combo.count() - 1, model.name, role=Qt.ItemDataRole.UserRole)
        finally:
            self.model_combo.blockSignals(False)

    def on_model_selected_from_index(self, idx: int):
        """Handle model selection by index to get clean filename from UserRole."""
        if idx < 0:
            return
        # Get clean filename from UserRole (not display text with size)
        model_name = self.model_combo.itemData(idx, role=Qt.ItemDataRole.UserRole)
        if not model_name:
            model_name = self.model_combo.currentText()
            # Fallback: strip size suffix if present
            if '(' in model_name:
                model_name = model_name.split('(')[0].strip()
        
        on_model_selected(self, model_name)

    def on_model_selected(self, name: str):
        model_path = (Path(self.model_directory) / name).resolve()
        self.selected_model = str(model_path)
        
         # Debug Output: Modell-Informationen anzeigen
        if model_path.exists() and model_path.is_file():
            try:
                info = get_model_info(str(model_path))
                
                # Strip leading/trailing whitespace/NULs from strings
                name = (info.get('name') or '').strip('\x00 \n\r\t')
                arch = (info.get('arch') or 'unknown').strip('\x00 ')
                
                 # Debug-Separator
                self.debug_text.append("─" * 60)
                self.debug_text.append(f"📦 {t('msg_model_selected')}: {info['filename']}")
                self.debug_text.append("─" * 60)
                self.debug_text.append(f"  {t('debug_model_name')}             {name or t('msg_unavailable')}")
                self.debug_text.append(f"  {t('debug_model_architecture')}      {arch}")
                
                if info.get('tags'):
                    tags_str = ", ".join(str(t) for t in info['tags'][:5])
                    if len(info['tags']) > 5:
                        tags_str += f" (+{len(info['tags']) - 5} more)"
                    self.debug_text.append(f"  {t('debug_model_tags')}             {tags_str}")
                
                if info.get('url'):
                    short_url = info['url'][:50] + "..." if len(info['url']) > 50 else info['url']
                    self.debug_text.append(f"  {t('debug_model_url')}              {short_url}")
                
                self.debug_text.append(f"  {t('debug_model_size')}       {format_size(info['file_size'])}")
                self.debug_text.append(f"  {t('debug_gguf_version')}     v{info['version']}")
                self.debug_text.append(f"  {t('debug_tensor_count')}     {info['tensor_count']:,}")
                self.debug_text.append(f"  {t('debug_context_length')}   {info['context_length'] or t('msg_not_found')}")
                
                if info.get('embedding_length'):
                    self.debug_text.append(f"  {t('debug_embedding_length')}  {info['embedding_length']}")
                if info.get('block_count'):
                    self.debug_text.append(f"  {t('debug_block_count')}      {info['block_count']}")
                
                self.debug_text.append("─" * 60)
            except Exception as e:
                self.debug_text.append(f"⚠️ {t('msg_model_info_read_error', error=e)}")
        
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
    
    def on_param_changed(self):
        """Wrapper for command_builder.on_param_changed"""
        from command_builder import on_param_changed as cb_on_param_changed
        cb_on_param_changed(self)
    
    def get_current_args(self) -> list:
        """Wrapper for command_builder.get_current_args"""
        from command_builder import get_current_args as cb_get_current_args
        return cb_get_current_args(self)
    
    def build_full_command(self, external_args: dict = None) -> str:
        """Wrapper for command_builder.build_full_command"""
        from command_builder import build_full_command as cb_build_full_command
        return cb_build_full_command(self, external_args)
    
    def on_benchmark_finished(self, tps, token_count):
        """Handle benchmark completion."""
        import re
        
        # Get last benchmark command from benchmark manager
        if hasattr(self, 'benchmark_manager'):
            full_command = self.benchmark_manager.get_last_benchmark_command()
        else:
            full_command = self.build_full_command()
        
        # Collect metrics from bench_thread if available (from JSON usage field)
        json_metrics = {}
        if hasattr(self, 'bench_thread') and hasattr(self.bench_thread, '_metrics'):
            json_metrics = self.bench_thread._metrics.copy()  # Copy to avoid stale references
        
        # Parse server logs from debug text for accurate timing
        server_log_metrics = {}
        try:
            debug_text = self.debug_text.toPlainText()
            
            # Parse prompt eval time first (must be before 'eval time' to avoid matching partial)
            match = re.search(r'prompt\s+eval\s+time\s*[:=]\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', debug_text, re.IGNORECASE)
            if match:
                server_log_metrics['prompt_eval_time_ms'] = float(match.group(1))
                server_log_metrics['prefill_tokens'] = int(match.group(2))
            
            # Parse eval time (generation) - check each line to avoid matching "prompt eval"
            lines = debug_text.split('\n')
            for line in lines:
                if 'eval time' in line and not 'prompt eval time' in line:
                    # Skip estimated values - prefer JSON metrics
                    if '[estimated]' in line:
                        continue
                    match = re.search(r'eval\s+time\s*[:=]\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', line, re.IGNORECASE)
                    if match:
                        server_log_metrics['eval_time_ms'] = float(match.group(1))
                        server_log_metrics['gen_tokens'] = int(match.group(2))
                    break
            
            # Parse total time - skip estimated values
            match = re.search(r'total\s+time\s*[:=]\s*([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens', debug_text, re.IGNORECASE)
            if match and '[estimated]' not in match.group(0):
                server_log_metrics['total_time_ms'] = float(match.group(1))
                server_log_metrics['total_tokens'] = int(match.group(2))
        except Exception as e:
            self.debug_text.append(f"Warning: Failed to parse server logs: {e}\n")
        
        # Debug: show what was parsed
        if server_log_metrics:
            self.debug_text.append(f"[SERVER LOG METRICS PARSED] {server_log_metrics}")
        
        # Use JSON metrics from benchmark thread (includes HTTP timing measurement)
        json_metrics = {}
        if hasattr(self, 'bench_thread') and hasattr(self.bench_thread, '_metrics'):
            json_metrics = self.bench_thread._metrics.copy()
        
        # Initialize details list
        details_lines = []
        
        # Priority: JSON metrics > server log metrics
        # JSON gives accurate HTTP timing; server logs are only used as fallback
        display_metrics = {}
        
        # Use JSON eval_time if available, otherwise server log
        json_eval_time = json_metrics.get('eval_time')
        if json_eval_time is not None and json_eval_time > 0:
            display_metrics['eval_time'] = json_eval_time
        elif server_log_metrics.get('eval_time_ms'):
            display_metrics['eval_time'] = server_log_metrics['eval_time_ms'] / 1000
        
        # Use JSON prompt_eval_time if available, otherwise server log
        json_prompt_eval_time = json_metrics.get('prompt_eval_time')
        if json_prompt_eval_time is not None and json_prompt_eval_time > 0:
            display_metrics['prompt_eval_time'] = json_prompt_eval_time
        elif server_log_metrics.get('prompt_eval_time_ms'):
            display_metrics['prompt_eval_time'] = server_log_metrics['prompt_eval_time_ms'] / 1000
        
        # Use JSON prefill_tokens if available, otherwise server log
        json_prefill_tokens = json_metrics.get('prefill_tokens')
        if json_prefill_tokens is not None and json_prefill_tokens > 0:
            display_metrics['prefill_tokens'] = json_prefill_tokens
        elif server_log_metrics.get('prefill_tokens'):
            display_metrics['prefill_tokens'] = server_log_metrics['prefill_tokens']
        
        # Use JSON total_time if available, otherwise server log
        json_total_time = json_metrics.get('total_time')
        if json_total_time is not None and json_total_time > 0:
            display_metrics['total_time'] = json_total_time
        elif server_log_metrics.get('total_time_ms'):
            display_metrics['total_time'] = server_log_metrics['total_time_ms'] / 1000
        
        # Use JSON total_tokens if available, otherwise server log
        json_total_tokens = json_metrics.get('total_tokens')
        if json_total_tokens is not None and json_total_tokens > 0:
            display_metrics['total_tokens'] = json_total_tokens
        elif server_log_metrics.get('total_tokens'):
            display_metrics['total_tokens'] = server_log_metrics['total_tokens']
        
           # Prompt eval (prefill)
        if display_metrics.get('prompt_eval_time'):
            pe_t = display_metrics['prompt_eval_time'] * 1000  # Convert to ms
            pe_tokens = display_metrics.get('prefill_tokens', 0)
            # Calculate ms/token and TPS from available data
            if display_metrics.get('prompt_eval_time') and pe_tokens > 0:
                pe_per_token_ms = (pe_t / pe_tokens)
                pe_tps = (pe_tokens / display_metrics['prompt_eval_time'])
            else:
                pe_per_token_ms = (pe_t / pe_tokens if pe_tokens > 0 else 0) * 1000
                pe_tps = (pe_tokens / pe_t if pe_t > 0 else 0)
            details_lines.append(f"✓ Prompt eval time:   {pe_t/1000:.3f}s / {pe_tokens} tokens ({pe_per_token_ms:.2f} ms/token, {pe_tps:.2f} TPS)")
        
        # Generation time
        if display_metrics.get('eval_time'):
            gen_t = display_metrics['eval_time'] * 1000  # Convert to ms
            # Use JSON completion_tokens or fallback to token_count
            gen_tokens = json_metrics.get('completion_tokens') or server_log_metrics.get('gen_tokens') or token_count
            # Calculate from available data
            if display_metrics.get('eval_time') and gen_tokens > 0:
                gen_per_token_ms = (gen_t / gen_tokens)
                gen_tps = (gen_tokens / display_metrics['eval_time'])
            else:
                gen_per_token_ms = (gen_t / gen_tokens if gen_tokens > 0 else 0) * 1000
                gen_tps = (gen_tokens / gen_t if gen_t > 0 else 0)
            details_lines.append(f"✓ Generation time:    {gen_t/1000:.3f}s / {gen_tokens} tokens ({gen_per_token_ms:.2f} ms/token, {gen_tps:.2f} TPS)")
        
        # Total - prefer JSON total_time, fallback to server log
        display_total_tokens = display_metrics.get('total_tokens') or (server_log_metrics.get('total_tokens') or token_count)
        if display_metrics.get('total_time'):
            total_t = display_metrics['total_time']
            # Calculate from available data
            if display_metrics.get('total_time') and display_total_tokens > 0:
                total_per_token_ms = (total_t * 1000) / display_total_tokens
                total_tps = display_total_tokens / total_t
            else:
                total_per_token_ms = (total_t * 1000) / display_total_tokens if display_total_tokens > 0 else 0
                total_tps = display_total_tokens / total_t if total_t > 0 else 0
            source = "json" if json_metrics.get('total_time', 0) > 0 else "server log" if server_log_metrics.get('total_time_ms') else "measured"
            details_lines.append(f"✓ Total time:         {total_t:.3f}s / {display_total_tokens} tokens ({total_per_token_ms:.2f} ms/token, {total_tps:.2f} TPS) [{source}]\n")
        
        # Join lines before using in debug output
        details = "\n".join(details_lines) if details_lines else "No metrics available"
        
        # Debug output for troubleshooting
        self.debug_text.append(f"\n[DEBUG DIALOG DETAILS]\n{details}\n\n[DEBUG JSON METRICS]\n{json_metrics}")
        
        # Use generation TPS from available data (JSON preferred, then server log, then calculated)
        json_eval_time = json_metrics.get('eval_time')
        json_completion_tokens = json_metrics.get('completion_tokens')
        if json_eval_time is not None and json_eval_time > 0 and json_completion_tokens is not None and json_completion_tokens > 0:
            display_tps = json_completion_tokens / json_eval_time
        elif server_log_metrics.get('eval_time_ms') and server_log_metrics.get('gen_tokens'):
            # Server log metrics are more accurate than JSON or HTTP fallback
            display_tps = server_log_metrics['gen_tokens'] / (server_log_metrics['eval_time_ms'] / 1000)
        elif 'gen_tps' in locals() and gen_tps > 0:
            display_tps = gen_tps
        else:
            display_tps = tps
        
        # Pass benchmark data to the dialog with preload/gen metrics
        ask_quality_and_save_benchmark(
            self,
            self.debug_text,
            self.status_label,
            display_tps,
            token_count,
            full_command,
            preload_time=display_metrics.get('prompt_eval_time'),
            preload_tokens=display_metrics.get('prefill_tokens'),
            preload_tps=pe_tps if 'pe_tps' in locals() else None,
            gen_time=display_metrics.get('eval_time'),
            gen_tokens=gen_tokens if 'gen_tokens' in locals() else None,
            gen_tps=gen_tps if 'gen_tps' in locals() else None,
        )
        
        # Disable cancel button and reset status after dialog closes
        if hasattr(self, 'cancel_bench_btn'):
            self.cancel_bench_btn.setEnabled(False)
        self.status_label.setText(gettext("status_ready"))
        
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
        from process_inspector import check_existing_process as pi_check
        pi_check(self)
    
    def show_bench_context_menu(self, pos):
        """Zeigt Kontextmenü für Benchmark-Tabelle mit 'Kopieren' Option."""
        from PyQt6.QtGui import QClipboard
        
        index = self.bench_table.indexAt(pos)
        if not index.isValid():
            return
        
        row = index.row()
        col = index.column()
        
        # Nur 9. Spalte (Kommandozeile) ist kopierbar
        if col != 8:
            return
        
        command_item = self.bench_table.item(row, 8)
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
        if not hasattr(self, 'benchmark_manager'):
            from benchmark_manager import BenchmarkManager
            self.benchmark_manager = BenchmarkManager(self)
        self.benchmark_manager.run_benchmark_streaming()

    def cancel_benchmark(self):
        """Cancel the currently running benchmark."""
        if not hasattr(self, 'benchmark_manager'):
            from benchmark_manager import BenchmarkManager
            self.benchmark_manager = BenchmarkManager(self)
        self.benchmark_manager.cancel_benchmark()

    def edit_prompt_dialog(self):
        """Show dialog to edit benchmark prompt."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox
        from PyQt6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Benchmark Prompt")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        info_label = QLabel(gettext("benchmark_prompt_info"))
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
        save_btn = QPushButton(gettext("save"))
        cancel_btn = QPushButton(gettext("cancel"))
        
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
        if not hasattr(self, 'benchmark_manager'):
            from benchmark_manager import BenchmarkManager
            self.benchmark_manager = BenchmarkManager(self)
        self.benchmark_manager.run_benchmark()

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
        from process_inspector import get_running_server_command as pi_get_cmd
        return pi_get_cmd(self)
    
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
            
            if exe_path and Path(exe_path).exists():
                self.llama_cpp_path = exe_path
                self.exe_line.setText(exe_path)

            if model_dir and Path(model_dir).exists():
                self.model_directory = model_dir
                self.debug_text.append(f"ℹ model_directory aus config.json geladen: {model_dir}")
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
        from ui_persistence import restore_geometry as up_restore
        up_restore(self)

    def resizeEvent(self, event):
        """Speichert Fenster-Geometrie bei jeder Größenänderung"""
        from ui_persistence import save_window_geometry as up_save_geom
        up_save_geom(self)

    def closeEvent(self, event):
        """Fenster-Geometrie und Splitter-State speichern + Timer stoppen"""
        from ui_persistence import save_window_state as up_save_state
        up_save_state(self)
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
