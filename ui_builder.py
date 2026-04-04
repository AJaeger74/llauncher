#!/usr/bin/env python3
"""
ui_builder – UI-Setup für llauncher

Extrahiert init_ui() aus llauncher.py zur besseren Wartbarkeit.
Baute alle Layouts, Sliders, Buttons und Labels auf.
"""

from pathlib import Path
from typing import Optional

# Import i18n gettext function
try:
    from i18n import I18nManager
    gettext = I18nManager.get_instance().gettext
except ImportError:
    def gettext(key):
        return key

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QPlainTextEdit, QSlider,
    QTextEdit, QVBoxLayout, QWidget, QTableWidget, QHeaderView, QSplitter,
    QCheckBox,
)

from gguf_utils import get_cpu_count
from storage import load_config, apply_preset, load_benchmarks
from gpu_monitor import GPUMonitor, update_gpu_display
from float_slider_sync import create_float_slider, create_int_slider


def build_llauncher_ui(window):
    """
    Baut das komplette UI für llauncher.
    
    Args:
        window: llauncher QMainWindow Instance (wird modifiziert)
    """
    # Grund-Setup
    window.setWindowTitle(f"llauncher v{window.VERSION}")
    window.setMinimumSize(1000, 800)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(10, 10, 10, 10)

    # Hauptbereich: Linke Spalte (Pfade + Parameter) und Rechte Spalte (Debug Output)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    
    # Splitter als Attribut speichern für späteren Zugriff
    window.splitter = splitter
    
    # ========== LINKE SPALTE ==========
    left_col = QWidget()
    left_layout = QVBoxLayout(left_col)
    left_layout.setSpacing(10)

    # Pfade Section
    paths_frame = QFrame()
    paths_frame.setObjectName("PathsFrame")
    paths_layout = QFormLayout(paths_frame)

    window.exe_line = QLineEdit(window.llama_cpp_path)
    window.exe_line.setReadOnly(True)
    browse_exe_btn = QPushButton(gettext("btn_browse_exe"))
    browse_exe_btn.clicked.connect(window.browse_llama_dir)

    exe_row = QWidget()
    exe_row_layout = QHBoxLayout(exe_row)
    exe_row_layout.addWidget(window.exe_line)
    exe_row_layout.addWidget(browse_exe_btn)

    window.model_line = QLineEdit(window.model_directory)
    window.model_line.setReadOnly(True)
    browse_model_btn = QPushButton(gettext("btn_browse_model"))
    browse_model_btn.clicked.connect(window.browse_model_dir)

    model_row = QWidget()
    model_row_layout = QHBoxLayout(model_row)
    model_row_layout.addWidget(window.model_line)
    model_row_layout.addWidget(browse_model_btn)

    paths_layout.addRow(gettext("lbl_exe_label"), exe_row)
    paths_layout.addRow(gettext("lbl_models_label"), model_row)

    window.exe_combo = QComboBox()
    # find_executables() wird später bei apply_presets() aufgerufen
    window.exe_combo.currentTextChanged.connect(window.on_exe_changed)

    window.model_combo = QComboBox()
    window.update_model_dropdown()
    window.model_combo.currentTextChanged.connect(window.on_model_selected)

    window.mmproj_line = QLineEdit()
    window.mmproj_line.setPlaceholderText("Optional: mmproj für Vision-Modelle")

    paths_layout.addRow("Executable:", window.exe_combo)
    paths_layout.addRow(gettext("lbl_model_select"), window.model_combo)
    paths_layout.addRow("mmproj (Vision):", window.mmproj_line)

    left_layout.addWidget(paths_frame)

    # Parameter Sliders Section
    params_scroll = QScrollArea()
    params_scroll.setWidgetResizable(True)
    params_scroll.setStyleSheet("QScrollArea { background-color: #1e1e1e; border: none; }")

    params_widget = QWidget()
    params_layout = QFormLayout(params_widget)
    params_layout.setSpacing(8)
    params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    params_layout.setHorizontalSpacing(12)

    window.param_sliders = {}
    for param_key, config in window.PARAM_DEFINITIONS.items():
        if config.get("type") == "float_slider":
            # Float-Slider: Ausgelagert nach float_slider_sync.py
            row_widget, slider_dict = create_float_slider(param_key, config)
            
            # Sync mit Debug-Output
            slider_dict["slider"].valueChanged.connect(window.on_param_changed)
            slider_dict["edit"].textChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            # Tooltip via tooltip_key übersetzen
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))
            
            params_layout.addRow(label, row_widget)
            window.param_sliders[param_key] = slider_dict
            
        elif config.get("type") == "combo":
            # ComboBox für String-Optionen (cache-type-k/v, flash-attn)
            combo = QComboBox()
            for opt in config["options"]:
                combo.addItem(opt)
            # Default-Wert finden und auswählen
            default_idx = combo.findText(config["default"])
            if default_idx >= 0:
                combo.setCurrentIndex(default_idx)
            combo.setFixedHeight(30)
            
            # Debug-Output live aktualisieren wenn Auswahl geändert wird
            combo.currentTextChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            # Tooltip via tooltip_key übersetzen
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))

            params_layout.addRow(label, combo)
            window.param_sliders[param_key] = {"combo": combo}
        
        elif config.get("type") == "text_input":
            # Einfaches Textfeld (z.B. --host)
            text_edit = QLineEdit()
            text_edit.setText(config["default"])
            text_edit.setFixedHeight(30)
            
            # Debug-Output live aktualisieren wenn Text geändert wird
            text_edit.textChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            # Tooltip via tooltip_key übersetzen
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))

            params_layout.addRow(label, text_edit)
            window.param_sliders[param_key] = {"edit": text_edit}
        
        elif config.get("type") == "path_input":
            # Pfad-Eingabe mit Browse-Button (z.B. --slot-save-path)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            
            path_edit = QLineEdit()
            path_edit.setText(config["default"])
            path_edit.setFixedHeight(30)
            
            browse_btn = QPushButton(gettext("btn_browse_model"))
            # Default-Pfad als Startverzeichnis für Dialog setzen
            default_dir = str(Path(config["default"]).parent)
            browse_btn.clicked.connect(lambda p=path_edit, d=default_dir: window.browse_path(p, d))
            
            row_layout.addWidget(path_edit, stretch=1)
            row_layout.addWidget(browse_btn)
            
            # Debug-Output live aktualisieren wenn Text geändert wird
            path_edit.textChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            # Tooltip via tooltip_key übersetzen
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))

            params_layout.addRow(label, row_widget)
            window.param_sliders[param_key] = {"edit": path_edit}
        
        elif config.get("type") == "file_input":
            # Datei-Eingabe mit Select/Löschen-Buttons (z.B. benchmark_file_path)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            
            file_edit = QLineEdit()
            file_edit.setText(config["default"])
            file_edit.setPlaceholderText(gettext("lbl_no_file_selected"))
            file_edit.setReadOnly(True)
            file_edit.setFixedHeight(30)
            
            select_btn = QPushButton(gettext("btn_select_file"))
            select_btn.setFixedWidth(80)
            select_btn.clicked.connect(lambda: window.on_select_benchmark_file(file_edit))
            
            clear_btn = QPushButton("X")
            clear_btn.setFixedWidth(30)
            clear_btn.clicked.connect(lambda: window.on_clear_benchmark_file(file_edit))
            
            row_layout.addWidget(file_edit, stretch=1)
            row_layout.addWidget(select_btn)
            row_layout.addWidget(clear_btn)
            
            # Debug-Output live aktualisieren wenn Text geändert wird
            file_edit.textChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))
            
            params_layout.addRow(label, row_widget)
            window.param_sliders[param_key] = {"edit": file_edit}
        
        elif config.get("type") == "slider":
            # Integer-Slider: Ausgelagert nach float_slider_sync.py
            row_widget, slider_dict = create_int_slider(param_key, config)
            
            # Sync mit Debug-Output
            slider_dict["slider"].valueChanged.connect(window.on_param_changed)
            slider_dict["edit"].textChanged.connect(window.on_param_changed)
            
            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
            label_text = gettext(config.get("label_key", config.get("label", param_key)))
            label = QLabel(f"{label_text} ({param_key})")
            # Tooltip via tooltip_key übersetzen
            tooltip_key = config.get("tooltip_key")
            if tooltip_key:
                label.setToolTip(gettext(tooltip_key))
            
            # Sonderfall: -ngl (GPU layers) mit "all" Checkbox
            if param_key == "-ngl":
                ngl_all_checkbox = QCheckBox("all")
                ngl_all_checkbox.setToolTip("Alle GPU-Layer laden (so viele wie möglich)")
                ngl_all_checkbox.setChecked(False)
                
                # Sync: Checkbox → Edit-Feld auf "all" setzen
                def on_ngl_checkbox_toggled(checked, edit=slider_dict["edit"]):
                    if checked:
                        edit.setText("all")
                    
                ngl_all_checkbox.toggled.connect(on_ngl_checkbox_toggled)
                
                # Sync: Edit-Feld → Checkbox aktivieren wenn "all" im Feld steht
                def on_ngl_edit_changed(text, checkbox=ngl_all_checkbox):
                    if text.lower() == "all":
                        checkbox.setChecked(True)
                    
                slider_dict["edit"].textChanged.connect(on_ngl_edit_changed)
                
                # Sync: Slider-Value-Änderung deaktiviert Checkbox automatisch
                def on_ngl_slider_changed(value, checkbox=ngl_all_checkbox):
                    checkbox.setChecked(False)
                
                slider_dict["slider"].valueChanged.connect(on_ngl_slider_changed)
                
                # Initial: Wenn Default "all" ist, Checkbox aktivieren und Edit-Feld setzen
                if str(config.get("default", "")).lower() == "all":
                    ngl_all_checkbox.setChecked(True)
                    slider_dict["edit"].setText("all")
                
                # Layout für Label + Checkbox in einem QWidget verpacken
                label_widget = QWidget()
                label_layout = QHBoxLayout(label_widget)
                label_layout.setContentsMargins(0, 0, 0, 0)
                label_layout.addWidget(label)
                label_layout.addWidget(ngl_all_checkbox)
                label_layout.addStretch()
                
                params_layout.addRow(label_widget, row_widget)
                window.ngl_all_checkbox = ngl_all_checkbox  # Für get_current_args verfügbar
            else:
                params_layout.addRow(label, row_widget)
            
            window.param_sliders[param_key] = slider_dict

    params_scroll.setWidget(params_widget)
    left_layout.addWidget(params_scroll)
    
    splitter.addWidget(left_col)
    
    # ========== RECHTE SPALTE: Debug Output ==========
    debug_frame = QFrame()
    debug_layout = QVBoxLayout(debug_frame)

    debug_label = QLabel(gettext("lbl_debug_output"))
    debug_label.setStyleSheet("font-weight: bold;")
    
    window.debug_text = QTextEdit()
    window.debug_text.setReadOnly(True)
    window.debug_text.setFont(QFont("Monospace", 9))
    window.debug_text.setMinimumWidth(500)
    # Prevent cropping of long debug output
    try:
        window.debug_text.setMaximumBlockCount(15000)
        from PyQt6.QtWidgets import QTextEdit as QPlainTextEdit
        window.debug_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    except AttributeError:
        pass  # Mindestbreite erhöhen

    copy_btn = QPushButton(gettext("btn_copy"))
    copy_btn.clicked.connect(window.copy_debug)
    
    debug_layout.addWidget(debug_label)
    debug_layout.addWidget(window.debug_text, stretch=1)
    debug_layout.addWidget(copy_btn)
    
    splitter.addWidget(debug_frame)
    
    # Splitter-Position setzen (60% links, 40% rechts)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)

    main_layout.addWidget(splitter)

    # ========== STATUS + START/STOP BUTTON ==========
    control_row = QHBoxLayout()

    window.status_label = QLabel(gettext("status_ready"))
    window.status_label.setObjectName("StatusLabel")
    window.status_label.setMinimumWidth(200)

    window.start_stop_btn = QPushButton(gettext("btn_start"))
    window.start_stop_btn.clicked.connect(window.toggle_process)
    window.start_stop_btn.setMinimumHeight(40)
    window.start_stop_btn.setStyleSheet("font-size: 16px; padding: 10px;")

    control_row.addWidget(window.status_label)
    control_row.addWidget(window.start_stop_btn, stretch=1)

    main_layout.addLayout(control_row)

    # ========== STATISTIKEN + PRESET BUTTONS ==========
    stats_presets_row = QHBoxLayout()

    window.stats_label = QLabel(gettext("stats_label"))

    presets_frame = QFrame()
    presets_layout = QHBoxLayout(presets_frame)

    save_btn = QPushButton(gettext("btn_save_preset"))
    save_btn.clicked.connect(window.save_preset)

    load_btn = QPushButton(gettext("btn_load_preset"))
    load_btn.clicked.connect(window.load_preset_dialog)

      # Benchmark Buttons (Standard + Streaming)
    run_bench_btn = QPushButton(gettext("btn_run_benchmark"))
    run_bench_btn.clicked.connect(window.run_benchmark)
    
    run_bench_streaming_btn = QPushButton(gettext("btn_run_benchmark_live"))
    run_bench_streaming_btn.clicked.connect(window.run_benchmark_streaming)
    
    # Edit prompt button
    edit_prompt_btn = QPushButton("✏️")
    edit_prompt_btn.setToolTip(gettext("tooltip_edit_prompt"))
    edit_prompt_btn.setFixedSize(40, 30)
    edit_prompt_btn.setStyleSheet("font-size: 14px;")
    edit_prompt_btn.clicked.connect(window.edit_prompt_dialog)
    
    # Cancel button for running benchmarks (hidden by default)
    window.cancel_bench_btn = QPushButton(gettext("btn_cancel"))
    window.cancel_bench_btn.setEnabled(False)  # Disabled when no benchmark is running
    window.cancel_bench_btn.setStyleSheet("color: red; font-weight: bold;")
    window.cancel_bench_btn.clicked.connect(window.cancel_benchmark)

    check_proc_btn = QPushButton(gettext("btn_check_process"))
    check_proc_btn.clicked.connect(window.on_check_process_click)

    load_args_btn = QPushButton(gettext("btn_load_process_args"))
    load_args_btn.setToolTip("Parameter des laufenden llama-server in die App laden\n"
                             "Unverwaltete Parameter werden separat angezeigt")
    # Explizit show_dialogs=True, damit Dialog immer erscheint wenn externe Args gefunden
    def on_load_args():
        window.load_running_process_args(show_dialogs=True)
    load_args_btn.clicked.connect(on_load_args)

    presets_layout.addWidget(save_btn)
    presets_layout.addWidget(load_btn)
    presets_layout.addWidget(run_bench_btn)
    presets_layout.addWidget(run_bench_streaming_btn)
    presets_layout.addWidget(edit_prompt_btn)
    presets_layout.addWidget(window.cancel_bench_btn)
    presets_layout.addWidget(check_proc_btn)
    presets_layout.addWidget(load_args_btn)
    presets_layout.addStretch()

    stats_presets_row.addWidget(window.stats_label)
    stats_presets_row.addWidget(presets_frame)

    main_layout.addLayout(stats_presets_row)

    # ========== BENCHMARK ERGEBNISSE (volle Breite, ganz unten) ==========
    bench_frame = QFrame()
    bench_layout = QVBoxLayout(bench_frame)

    bench_label = QLabel(gettext("label_bench_results"))
    bench_label.setFont(QFont("Monospace", 9))

    window.bench_table = QTableWidget()
    window.bench_table.setColumnCount(4)
    window.bench_table.setHorizontalHeaderLabels([gettext("bench_table_date"), gettext("bench_table_tps"), gettext("bench_table_quality"), gettext("lbl_command_line")])
    
    # Nur Doppelklick auf Qualitätsspalte erlaubt zum Editieren
    window.bench_table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
    
    # Fixe Spaltenbreiten: Datum/Zeit (150px), TPS und Qualität schmal, Kommandozeile streckt sich
    window.bench_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
    window.bench_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
    window.bench_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
    window.bench_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    window.bench_table.setColumnWidth(0, 150)  # Datum/Zeit fix
    window.bench_table.setColumnWidth(1, 80)   # TPS fix (5 Stellen vor + Komma + 2 nach = max 8 Zeichen)
    window.bench_table.setColumnWidth(2, 90)   # Qualität fix (Sehr gut / Gut / Mittel / Schlecht)
    
    # Kontextmenü für Rechtsklick aktivieren
    window.bench_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    window.bench_table.customContextMenuRequested.connect(window.show_bench_context_menu)

    # ========== EXPORT BUTTON ==========
    export_btn = QPushButton("Export")
    export_btn.setMaximumWidth(80)
    
    def on_export():
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            window, "Benchmark exportieren",
            str(Path.home() / "Downloads" / "benchmarks.csv"),
            "CSV-Dateien (*.csv);;JSON-Dateien (*.json);;Alle Dateien (*)"
        )
        
        if not file_path:
            return
        
        benchmarks = load_benchmarks()
        
        if file_path.endswith('.csv'):
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Datum/Zeit", "TPS", "Qualität", "Kommandozeile"])
                for b in benchmarks:
                    writer.writerow([b["timestamp"], b["tps"], b["quality"], b["full_command"]])
        
        elif file_path.endswith('.json'):
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(benchmarks, f, ensure_ascii=False, indent=2)
        
        QMessageBox.information(window, "Export erfolgreich", 
                               f"Benchmark exportiert nach:\n{file_path}")

    export_btn.clicked.connect(on_export)
    
    bench_layout.addWidget(export_btn, stretch=0)
    bench_layout.addWidget(bench_label)
    bench_layout.addWidget(window.bench_table, stretch=1)
    main_layout.addWidget(bench_frame)

    # ========== STYLING (KDE Plasma Dark Theme) ==========
    window.setStyleSheet("""
        QMainWindow { background-color: #2d2d2d; }
        QFrame#PathsFrame { background-color: #3a3a3a; padding: 10px; border-radius: 5px; }
        QLabel#StatusLabel { color: #0078d7; font-weight: bold; padding: 10px; min-width: 200px; }
        QLabel#StatsLabel { color: #cccccc; padding: 10px; }
        QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
        QPushButton:hover { background-color: #006cc1; }
        QPushButton#StopButton { background-color: #ff6600; }
        QPushButton#StopButton:hover { background-color: #e65c00; }
        QComboBox, QLineEdit { padding: 5px; border-radius: 3px; background-color: #444; color: white; }
        QTextEdit { background-color: #1a1a1a; color: #cccccc; }
    """)

    # ========== GPU-MONITOR SOFORT STARTEN ==========
    window.gpu_monitor = GPUMonitor()
    window.gpu_monitor.gpu_update.connect(lambda data: update_gpu_display(window.stats_label, data))
    window.gpu_monitor.start()


def setup_timers_and_load(window):
    """
    Setzt up die Timer für Prozess-Check und lädt Config/Presets.
    
    Args:
        window: llauncher QMainWindow Instance (wird modifiziert)
    """
    # Timer für sekundliche Prüfung auf externe Prozesse
    window.process_check_timer = QTimer()
    window.process_check_timer.setInterval(1000)  # 1 Sekunde
    window.process_check_timer.timeout.connect(window.check_existing_process)
    window.process_check_timer.start()
    
    # Konfiguration und Dropdowns laden
    window.load_config()
    window.update_model_dropdown()
    
    # Erst Prozess prüfen, dann Parameter laden (statt umgekehrt)
    # WICHTIG: Nur wenn kein Preset geladen werden soll - sonst UI zerstören
    if not window._load_running_process_args_silent():
        # Kein externer Prozess gefunden - Presets laden als Fallback
        window.apply_presets()
