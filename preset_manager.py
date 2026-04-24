#!/usr/bin/env python3
"""
Preset Manager – Dialoge für Preset-Management und Benchmarking
Ausgelagert aus llauncher.py zur Reduktion der Komplexität.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QInputDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QListWidget,
    QLineEdit,
    QDialog,
    QGridLayout,
    QGroupBox,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QDate, QTime, Qt
from i18n import I18nManager
gettext = I18nManager.get_instance().gettext

from storage import load_presets, save_benchmarks, load_benchmarks, save_presets


def save_active_preset(window):
    """Save current config as preset to ~/.llauncher/presets.json"""
    from storage import CONFIG_DIR
    
    config_path = CONFIG_DIR / "config.json"
    with open(config_path, "r") as f:
        import json
        config = json.load(f)
    
    # Extract relevant preset fields (exclude internal/private fields)
    preset_data = {k: v for k, v in config.items() if not k.startswith("_")}
    
    name = config.get("preset_name", "Unnamed")
    preset = {"name": name, **preset_data}
    
    # Custom Commands aus UI-Textfeld holen (falls vorhanden)
    if hasattr(window, 'custom_cmd_edit'):
        custom_text = window.custom_cmd_edit.toPlainText().strip()
        if custom_text:
            preset["custom_commands"] = custom_text
    
    # Use storage layer for consistent format (dict: {name: {...}, ...})
    presets = load_presets()
    presets[name] = preset
    save_presets(presets)


def show_preset_save_dialog(window, param_sliders, PARAM_DEFINITIONS, 
                            llama_cpp_path, model_directory, selected_model,
                            mmproj_line):
    """Dialog zum Speichern eines Presets mit Liste existierender Presets."""
    
    # Bestehende Presets laden
    presets = load_presets()
    
    # Dialog-Fenster erstellen
    dialog = QDialog(window)
    dialog.setWindowTitle(gettext("dialog_save_preset_title"))
    layout = QVBoxLayout(dialog)
    
    # Label für bestehende Presets
    info_label = QLabel("Vorhandene Presets:")
    layout.addWidget(info_label)
    
    # Liste der existierenden Presets
    preset_list = QListWidget()
    if presets:
        for name in sorted(presets.keys()):
            preset_list.addItem(name)
    else:
        preset_list.addItem("(keine vorhanden)")
        preset_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
    
    # Doppelklick auf bestehenden Eintrag kopiert Namen ins Eingabefeld
    def on_preset_double_click(index):
        item = preset_list.item(index.row())
        name_edit.setText(item.text())
    
    preset_list.doubleClicked.connect(on_preset_double_click)
    preset_list.setMinimumWidth(320)
    layout.addWidget(preset_list)
    
    # Eingabefeld für neuen Namen
    name_edit = QLineEdit()
    name_label = QLabel("Name des Presets:")
    layout.addWidget(name_label)
    layout.addWidget(name_edit)
    
    # Buttons
    btn_layout = QVBoxLayout()
    save_btn = QPushButton("Speichern")
    cancel_btn = QPushButton("Abbrechen")
    btn_layout.addWidget(save_btn)
    btn_layout.addWidget(cancel_btn)
    layout.addLayout(btn_layout)
    
    dialog.setLayout(layout)
    dialog.setMinimumSize(380, 350)
    
    # Speichern-Button Aktion
    def handle_save():
        name = name_edit.text()
        if not name:
            QMessageBox.warning(dialog, "Kein Name", "Bitte geben Sie einen Namen ein.")
            return None
        
        # Prüfen ob Preset existiert
        if name in presets:
            reply = QMessageBox.question(
                dialog, "Preset existiert bereits",
                f"Das Preset '{name}' existiert bereits. Überschreiben?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None
        
        # mmproj vollen Pfad speichern
        mmproj_text = mmproj_line.text().strip()
        mmproj_path = ""
        if mmproj_text:
            mmproj_path_obj = Path(mmproj_text)
            if not mmproj_path_obj.is_absolute():
                mmproj_path = str(Path(model_directory) / mmproj_path_obj)
            else:
                mmproj_path = mmproj_text
        
        preset = {
            "llama_cpp_path": llama_cpp_path,
            "model_directory": model_directory,
            "selected_model": selected_model or "",
            "mmproj_path": mmproj_path,
            "params": {},
        }
        
        # Parameter aus Slidern sammeln
        for param_key, config in PARAM_DEFINITIONS.items():
            if param_key not in param_sliders:
                continue
            
            slider = param_sliders[param_key]
            
            try:
                if config.get("type") == "float_slider":
                    value_edit = slider["edit"]
                    preset["params"][param_key] = float(value_edit.text())
                
                elif config.get("type") == "combo":
                    preset["params"][param_key] = slider["combo"].currentText()
                
                elif config.get("type") in ("text_input", "path_input"):
                    text_edit = slider["edit"]
                    preset["params"][param_key] = text_edit.text()
                
                else:  # Integer-Slider
                    if isinstance(slider, dict):
                        # Sonderfall: -ngl mit "all" Checkbox → String "all" speichern wenn aktiviert
                        if param_key == "-ngl" and hasattr(window, "ngl_all_checkbox") and window.ngl_all_checkbox.isChecked():
                            preset["params"][param_key] = "all"
                        else:
                            preset["params"][param_key] = slider["slider"].value()
                    else:
                        preset["params"][param_key] = slider.value()
            except (KeyError, AttributeError, TypeError, ValueError):
                continue
        
              # Custom Commands aus UI-Textfeld holen
        if hasattr(window, 'custom_cmd_edit'):
            custom_text = window.custom_cmd_edit.toPlainText().strip()
            if custom_text:
                preset["custom_commands"] = custom_text
        
        return name, preset
    
    save_result = [None]  # Mutable container to store result from button click
    
    def on_save_click():
        result = handle_save()
        if result:
            save_result[0] = result
            dialog.accept()
    
    save_btn.clicked.connect(on_save_click)
    cancel_btn.clicked.connect(dialog.reject)
    
    dialog.exec()
    return save_result[0]


def show_preset_load_dialog(window, param_sliders, PARAM_DEFINITIONS,
                            llama_cpp_path, model_directory, selected_model,
                            mmproj_line, exe_combo):
    """Dialog zum Laden eines Presets. Gibt (Preset-Name, Preset-Dict) zurück oder (None, None)."""
    
    presets = load_presets()
    if not presets:
        QMessageBox.information(window, "Keine Presets", 
                                "Keine gespeicherten Presets gefunden.")
        return None, None
    
    name_list = list(presets.keys())
    preset_name, ok = QInputDialog.getItem(
        window, "Preset laden", "Wähle ein Preset:", 
        name_list, editable=False
    )
    if not ok or not preset_name:
        return None, None
    
    return preset_name, presets[preset_name]


def show_preset_args(window, debug_text, preset_name: str, preset: dict,
                     param_sliders, PARAM_DEFINITIONS, llama_cpp_path,
                     model_directory, selected_model, mmproj_line, exe_combo):
    """Zeigt die vollständige Kommandozeile im Debug-Bereich."""
    
    args = []
    
    # Executable (voller Pfad aus Preset oder Config)
    selected_exe = preset.get("selected_exe", "")
    if selected_exe and Path(selected_exe).exists():
        exe_path = selected_exe
    else:
        exe_name = exe_combo.currentText()
        if exe_name != "llama.cpp nicht gefunden":
            exe_path = str(Path(llama_cpp_path) / exe_name)
        else:
            exe_path = ""
    
    if exe_path:
        args.append(exe_path)
    
    # Modell-Pfad (voller Pfad aus Preset oder Config)
    model_from_preset = preset.get("selected_model", "")
    if not Path(model_from_preset).exists() and model_from_preset == "":
        current_model_name = window.model_combo.currentText() if hasattr(window, 'model_combo') else ""
        if current_model_name:
            selected_model = str(Path(model_directory) / current_model_name)
    
    if selected_model or model_from_preset:
        model_path = selected_model or model_from_preset
        args.extend(["-m", model_path])
    
    # mmproj (voller Pfad aus Preset oder Config)
    mmproj_path = preset.get("mmproj_path", "")
    
    if not mmproj_path or not Path(mmproj_path).exists():
        if hasattr(window, 'mmproj_line'):
            mmproj_text = window.mmproj_line.text().strip()
            if mmproj_text:
                mmproj_path = mmproj_text
    
    if mmproj_path:
        args.extend(["--mmproj", mmproj_path])
    
    # Parameter aus Preset ODER UI - PRIORITÄT: UI bei "Manuell"
    preset_params = preset.get("params", {})
    
    for param_key, config in PARAM_DEFINITIONS.items():
        value = None
        
        if preset_name != "Manuell" and param_key in preset_params:
            value = preset_params[param_key]
        elif param_key in param_sliders:
            slider_data = param_sliders[param_key]
            
            try:
                if config.get("type") == "float_slider":
                    value = float(slider_data["edit"].text())
                
                elif config.get("type") == "combo":
                    combo = slider_data["combo"]
                    value = combo.currentText()
                
                elif config.get("type") in ("text_input", "path_input"):
                    text_edit = slider_data["edit"]
                    value = text_edit.text()
                
                else:  # Integer-Slider
                    if isinstance(slider_data, dict):
                        value = slider_data["slider"].value()
                    else:
                        value = slider_data.value()
            except (KeyError, AttributeError, TypeError, ValueError):
                continue
        
        if value is not None:
            if config.get("type") == "combo":
                if value != config["default"]:
                    args.append(param_key)
                    args.append(value)
            
            elif config.get("type") in ("text_input", "path_input"):
                if value and value != config["default"]:
                    args.append(param_key)
                    args.append(value)
            
            else:  # Integer/Float-Slider
                # Sonderfall: -ngl mit String "all" → direkt als String verwenden
                if param_key == "-ngl" and isinstance(value, str) and value == "all":
                    args.append(param_key)
                    args.append(value)
                    continue
                
                default_val = config["default"]
                if isinstance(default_val, float):
                    diff = abs(float(value) - default_val)
                else:
                    diff = abs(int(value) - int(default_val))
                
                if diff > 0.01:
                    args.append(param_key)
                    if config.get("type") == "float_slider":
                        args.append(f"{value:.2f}")
                    else:
                        args.append(str(value))
    
    # Kommandozeile formatieren (mit Escape für Leerzeichen/Special-Chars)
    import shlex
    cmd_line = " ".join(shlex.quote(arg) for arg in args if arg)

    debug_text.clear()
    preset_header = f"=== Preset: {preset_name} ===\n\nKommandozeile:\n{cmd_line}\n"
    debug_text.setPlainText(preset_header)


class BenchmarkRatingDialog(QDialog):
    """Custom dialog showing detailed benchmark metrics with quality input."""
    
    def __init__(self, window, tps, token_count, full_command,
                 preload_time=None, preload_tokens=None, preload_tps=None,
                 gen_time=None, gen_tokens=None, gen_tps=None):
        super().__init__(window)
        self.setWindowTitle(gettext("dialog_rating_title"))
        self.setModal(True)
        self.setMinimumSize(480, 380)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        
        # Title
        title_label = QLabel(gettext("msg_benchmark_complete_title"))
        title_label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        grid = QGridLayout()
        grid.setSpacing(8)
        
        row = 0
        mono = QFont("Monospace")
        
        if preload_time is not None:
            # Preload section header
            h = QLabel(gettext("lbl_preload_section"))
            h.setStyleSheet("font-weight: bold; font-size: 10pt; color: #5dade2;")
            grid.addWidget(h, row, 0, 1, 4)
            row += 1
            
            # Time + Tokens on same row
            lbl = QLabel(f"{gettext('lbl_preload_time')}:  ")
            grid.addWidget(lbl, row, 0)
            v = QLabel(self._format_time(preload_time))
            v.setFont(mono)
            grid.addWidget(v, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
            lbl2 = QLabel(f"{gettext('lbl_preload_tokens')}:  ")
            grid.addWidget(lbl2, row, 2)
            v2 = QLabel(str(preload_tokens))
            v2.setFont(mono)
            grid.addWidget(v2, row, 3, alignment=Qt.AlignmentFlag.AlignRight)
            row += 1
            
            # TPS on its own row
            lbl3 = QLabel(f"{gettext('lbl_preload_tps')}:  ")
            grid.addWidget(lbl3, row, 0)
            v3 = QLabel(f"{preload_tps:.2f}")
            v3.setFont(mono)
            grid.addWidget(v3, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
            # colspan for empty space
            row += 1
        
        if gen_time is not None:
            h = QLabel(gettext("lbl_gen_section"))
            h.setStyleSheet("font-weight: bold; font-size: 10pt; color: #5dade2;")
            grid.addWidget(h, row, 0, 1, 4)
            row += 1
            
            lbl4 = QLabel(f"{gettext('lbl_gen_time')}:  ")
            grid.addWidget(lbl4, row, 0)
            v4 = QLabel(self._format_time(gen_time))
            v4.setFont(mono)
            grid.addWidget(v4, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
            lbl5 = QLabel(f"{gettext('lbl_gen_tokens')}:  ")
            grid.addWidget(lbl5, row, 2)
            v5 = QLabel(str(gen_tokens))
            v5.setFont(mono)
            grid.addWidget(v5, row, 3, alignment=Qt.AlignmentFlag.AlignRight)
            row += 1
            
            lbl6 = QLabel(f"{gettext('lbl_gen_tps')}:  ")
            grid.addWidget(lbl6, row, 0)
            v6 = QLabel(f"{gen_tps:.2f}")
            v6.setFont(mono)
            grid.addWidget(v6, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
            row += 1
        
        # Summary
        lbl7 = QLabel(f"{gettext('lbl_overall_tps')}:  ")
        grid.addWidget(lbl7, row, 0)
        item = QLabel(f"{tps:.2f} TPS")
        item.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        grid.addWidget(item, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
        lbl8 = QLabel(f"{gettext('lbl_gen_count')}:  ")
        grid.addWidget(lbl8, row, 2)
        v7 = QLabel(str(token_count))
        v7.setFont(mono)
        grid.addWidget(v7, row, 3, alignment=Qt.AlignmentFlag.AlignRight)
        
        main_layout.addLayout(grid)
        
        # Quality input section
        quality_layout = QVBoxLayout()
        quality_label = QLabel(gettext("lbl_quality_input"))
        quality_label.setFont(QFont("Monospace", 10))
        quality_layout.addWidget(quality_label)
        
        self.quality_edit = QLineEdit()
        self.quality_edit.setPlaceholderText(gettext("dlg_quality_placeholder"))
        self.quality_edit.setFont(QFont("Monospace", 11))
        quality_layout.addWidget(self.quality_edit)
        main_layout.addLayout(quality_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton(gettext("btn_save_rating"))
        save_btn.setStyleSheet("""
            QPushButton { 
                background-color: #27ae60; color: white; padding: 8px 20px; 
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #219a52; }
        """)
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton(gettext("btn_cancel"))
        cancel_btn.setStyleSheet("""
            QPushButton { 
                background-color: #7f8c8d; color: white; padding: 8px 20px; 
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #6c7a7b; }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(btn_layout)
    
    def _format_time(self, seconds):
        """Format seconds as ms or s."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        return f"{seconds:.2f}s"
    
    def get_quality(self):
        return self.quality_edit.text().strip()


def ask_quality_and_save_benchmark(window, debug_text, status_label, 
                                   tps, token_count, full_command,
                                   preload_time=None, preload_tokens=None, preload_tps=None,
                                   gen_time=None, gen_tokens=None, gen_tps=None):
    """Fragt Qualitätsbewertung ab und speichert Benchmark.
    
    full_command: Vollständige Kommandozeile (z.B. "/home/user/llama.cpp/llama-server -m /home/user/models/model.gguf -c 2048 ...")
    preload_time: Prefill/Preload Zeit in Sekunden (float)
    preload_tokens: Anzahl Prefill-Tokens (int)
    preload_tps: Prefill/Preload Tokens per Second (float)
    gen_time: Generation Zeit in Sekunden (float)
    gen_tokens: Anzahl generierter Tokens (int)
    gen_tps: Generation Tokens per Second (float)
    """
    
    # Custom rating dialog with all metrics
    dialog = BenchmarkRatingDialog(
        window, tps, token_count, full_command,
        preload_time=preload_time,
        preload_tokens=preload_tokens,
        preload_tps=preload_tps,
        gen_time=gen_time,
        gen_tokens=gen_tokens,
        gen_tps=gen_tps,
    )
    
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    
    quality = dialog.get_quality()
    if not quality:
        return None
    
    # Benchmark speichern
    timestamp = QDate.currentDate().toString("yyyy-MM-dd") + " " + \
                QTime.currentTime().toString("HH:mm:ss")
    
    benchmark_entry = {
        "timestamp": timestamp,
        "full_command": full_command,
        "tps": round(tps, 2),
        "quality": quality,
    }
    # Neue Felder hinzufügen falls vorhanden
    if preload_time is not None:
        benchmark_entry["preload_time"] = round(preload_time, 3)
    if preload_tokens is not None:
        benchmark_entry["preload_tokens"] = preload_tokens
    if preload_tps is not None:
        benchmark_entry["preload_tps"] = round(preload_tps, 2)
    if gen_time is not None:
        benchmark_entry["gen_time"] = round(gen_time, 3)
    if gen_tokens is not None:
        benchmark_entry["gen_tokens"] = gen_tokens
    if gen_tps is not None:
        benchmark_entry["gen_tps"] = round(gen_tps, 2)
    
    save_benchmarks([benchmark_entry])
    
    # Tabelle aktualisieren
    row = window.bench_table.rowCount()
    window.bench_table.insertRow(row)
    
    # Datum/Zeit: read-only
    date_item = QTableWidgetItem(timestamp)
    date_item.setFlags(date_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 0, date_item)
    
    # Preload Time: read-only (col 1)
    if preload_time is not None:
        preload_str = f"{preload_time*1000:.0f}ms" if preload_time < 1 else f"{preload_time:.2f}s"
    else:
        preload_str = "-"
    preload_time_item = QTableWidgetItem(preload_str)
    preload_time_item.setFlags(preload_time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 1, preload_time_item)
    
    # Preload Tokens: read-only (col 2)
    if preload_tokens is not None:
        preload_tok_str = str(preload_tokens)
    else:
        preload_tok_str = "-"
    preload_tok_item = QTableWidgetItem(preload_tok_str)
    preload_tok_item.setFlags(preload_tok_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 2, preload_tok_item)
    
    # Preload TPS: read-only (col 3)
    if preload_tps is not None:
        preload_tps_str = f"{preload_tps:.2f}"
    else:
        preload_tps_str = "-"
    preload_tps_item = QTableWidgetItem(preload_tps_str)
    preload_tps_item.setFlags(preload_tps_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 3, preload_tps_item)
    
    # Generation Time: read-only (col 4)
    if gen_time is not None:
        gen_str = f"{gen_time*1000:.0f}ms" if gen_time < 1 else f"{gen_time:.2f}s"
    else:
        gen_str = "-"
    gen_time_item = QTableWidgetItem(gen_str)
    gen_time_item.setFlags(gen_time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 4, gen_time_item)
    
    # Generation Tokens: read-only (col 5)
    if gen_tokens is not None:
        gen_tok_str = str(gen_tokens)
    else:
        gen_tok_str = "-"
    gen_tok_item = QTableWidgetItem(gen_tok_str)
    gen_tok_item.setFlags(gen_tok_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 5, gen_tok_item)
    
    # Generation TPS: read-only (col 6)
    if gen_tps is not None:
        gen_tps_str = f"{gen_tps:.2f}"
    else:
        gen_tps_str = "-"
    gen_tps_item = QTableWidgetItem(gen_tps_str)
    gen_tps_item.setFlags(gen_tps_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 6, gen_tps_item)
    
    # Qualität: editierbar (col 7)
    quality_item = QTableWidgetItem(quality)
    quality_item.setFlags(quality_item.flags() | Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 7, quality_item)
    
    # Kommandozeile: read-only (col 8)
    cmd_item = QTableWidgetItem(full_command)
    cmd_item.setFlags(cmd_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 8, cmd_item)
    
    return benchmark_entry