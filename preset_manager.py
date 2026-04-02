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
    QLabel,
    QPushButton,
    QTextEdit,
    QListWidget,
    QLineEdit,
    QDialog,
)
from PyQt6.QtCore import QDate, QTime, Qt
from i18n import I18nManager
gettext = I18nManager.get_instance().gettext

from storage import load_presets, save_benchmarks, load_benchmarks


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


def ask_quality_and_save_benchmark(window, debug_text, status_label, 
                                   tps, token_count, full_command):
    """Fragt Qualitätsbewertung ab und speichert Benchmark.
    
    full_command: Vollständige Kommandozeile (z.B. "/home/user/llama.cpp/llama-server -m /home/user/models/model.gguf -c 2048 ...")
    """
    
    quality, ok = QInputDialog.getText(
        window, gettext("dialog_quality_title"),
        gettext("msg_benchmark_complete").format(tps=tps, token_count=token_count) + "\n\n" + gettext("lbl_quality_input")
    )
    if not ok or not quality:
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
    save_benchmarks([benchmark_entry])
    
    # Tabelle aktualisieren
    row = window.bench_table.rowCount()
    window.bench_table.insertRow(row)
    
    # Datum/Zeit: read-only (standardmäßig ItemIsEditable entfernen)
    date_item = QTableWidgetItem(timestamp)
    date_item.setFlags(date_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 0, date_item)
    
    # TPS: read-only
    tps_item = QTableWidgetItem(f"{tps:.2f}")
    tps_item.setFlags(tps_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 1, tps_item)
    
    # Qualität: editierbar (ItemIsEditable setzen)
    quality_item = QTableWidgetItem(quality)
    quality_item.setFlags(quality_item.flags() | Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 2, quality_item)
    
    # Kommandozeile: read-only
    cmd_item = QTableWidgetItem(full_command)
    cmd_item.setFlags(cmd_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    window.bench_table.setItem(row, 3, cmd_item)
    
    return benchmark_entry