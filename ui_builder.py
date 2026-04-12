     1|#!/usr/bin/env python3
     2|"""
     3|ui_builder – UI-Setup für llauncher
     4|
     5|Extrahiert init_ui() aus llauncher.py zur besseren Wartbarkeit.
     6|Baute alle Layouts, Sliders, Buttons und Labels auf.
     7|"""
     8|
     9|import json
    10|from pathlib import Path
    11|from typing import Optional
    12|
    13|# ========== THEME STYLES ==========\
    14|DARK_THEME = """
    15|QMainWindow { background-color: #2d2d2d; }
    16|QFrame#PathsFrame { background-color: #3a3a3a; padding: 10px; border-radius: 5px; }
    17|QLabel#StatusLabel { color: #0078d7; font-weight: bold; padding: 10px; min-width: 200px; }
    18|QLabel#StatsLabel { color: #cccccc; padding: 10px; }
    19|QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
    20|QPushButton:hover { background-color: #006cc1; }
    21|QPushButton#StopButton { background-color: #ff6600; }
    22|QPushButton#StopButton:hover { background-color: #e65c00; }
    23|QComboBox, QLineEdit { padding: 5px; border-radius: 3px; background-color: #444; color: white; }
    24|QTextEdit { background-color: #1a1a1a; color: #cccccc; }
    25|QScrollArea { background-color: #1e1e1e; border: none; }
    26|"""
    27|
    28|LIGHT_THEME = """
    29|QMainWindow { background-color: #f5f5f5; }
    30|QFrame#PathsFrame { background-color: #ffffff; padding: 10px; border-radius: 5px; }
    31|QLabel#StatusLabel { color: #0078d7; font-weight: bold; padding: 10px; min-width: 200px; }
    32|QLabel#StatsLabel { color: #333333; padding: 10px; }
    33|QLabel { color: #333333; }
    34|QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
    35|QPushButton:hover { background-color: #006cc1; }
    36|QPushButton#StopButton { background-color: #cc5200; }
    37|QPushButton#StopButton:hover { background-color: #b34700; }
    38|QComboBox, QLineEdit { padding: 5px; border-radius: 3px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
    39|QTextEdit { background-color: #ffffff; color: #333333; }
    40|QScrollArea { background-color: #ffffff; border: none; }
    41|"""
    42|
    43|# Import i18n gettext function
    44|try:
    45|    from i18n import I18nManager
    46|    gettext = I18nManager.get_instance().gettext
    47|except ImportError:
    48|    def gettext(key):
    49|        return key
    50|
    51|from PyQt6.QtCore import Qt, QTimer
    52|from PyQt6.QtGui import QFont
    53|from PyQt6.QtWidgets import (
    54|    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    55|    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    56|    QMessageBox, QPushButton, QProgressBar, QScrollArea, QPlainTextEdit, QSlider,
    57|    QTextEdit, QVBoxLayout, QWidget, QTableWidget, QHeaderView, QSplitter,
    58|    QCheckBox,
    59|)
    60|
    61|from gguf_utils import get_cpu_count
    62|from storage import load_config, apply_preset, load_benchmarks
    63|from gpu_monitor import GPUMonitor, update_gpu_display
    64|from float_slider_sync import create_float_slider, create_int_slider
    65|
    66|
    67|def build_llauncher_ui(window):
    68|    """
    69|    Baut das komplette UI für llauncher.
    70|    
    71|    Args:
    72|        window: llauncher QMainWindow Instance (wird modifiziert)
    73|    """
    74|    # Grund-Setup
    75|    window.setWindowTitle(f"llauncher v{window.VERSION}")
    76|    window.setMinimumSize(1000, 800)
    77|
    78|    central_widget = QWidget()
    79|    window.setCentralWidget(central_widget)
    80|    main_layout = QVBoxLayout(central_widget)
    81|    main_layout.setSpacing(10)
    82|    main_layout.setContentsMargins(10, 10, 10, 10)
    83|
    84|    # Hauptbereich: Linke Spalte (Pfade + Parameter) und Rechte Spalte (Debug Output)
    85|    splitter = QSplitter(Qt.Orientation.Horizontal)
    86|    
    87|    # Splitter als Attribut speichern für späteren Zugriff
    88|    window.splitter = splitter
    89|    
    90|    # ========== LINKE SPALTE ==========
    91|    left_col = QWidget()
    92|    left_layout = QVBoxLayout(left_col)
    93|    left_layout.setSpacing(10)
    94|
    95|  # Pfade Section
    96|    paths_frame = QFrame()
    97|    paths_frame.setObjectName("PathsFrame")
    98|    paths_layout = QFormLayout(paths_frame)
    99|    
   100|    window.exe_line = QLineEdit(window.llama_cpp_path)
   101|    window.exe_line.setReadOnly(True)
   102|    browse_exe_btn = QPushButton(gettext("btn_browse_exe"))
   103|    browse_exe_btn.clicked.connect(window.browse_llama_dir)
   104|    
   105|    exe_row = QWidget()
   106|    exe_row_layout = QHBoxLayout(exe_row)
   107|    exe_row_layout.addWidget(window.exe_line)
   108|    exe_row_layout.addWidget(browse_exe_btn)
   109|    
   110|    window.model_line = QLineEdit(window.model_directory)
   111|    window.model_line.setReadOnly(True)
   112|    browse_model_btn = QPushButton(gettext("btn_browse_model"))
   113|    browse_model_btn.clicked.connect(window.browse_model_dir)
   114|    
   115|    model_row = QWidget()
   116|    model_row_layout = QHBoxLayout(model_row)
   117|    model_row_layout.addWidget(window.model_line)
   118|    model_row_layout.addWidget(browse_model_btn)
   119|    
   120|    exe_label = QLabel(gettext("lbl_exe_label"))
   121|    exe_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   122|    exe_label.setStyleSheet("margin-top: 8px;")
   123|    
   124|    model_label = QLabel(gettext("lbl_models_label"))
   125|    model_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   126|    model_label.setStyleSheet("margin-top: 8px;")
   127|    
   128|    paths_layout.addRow(exe_label, exe_row)
   129|    paths_layout.addRow(model_label, model_row)
   130|    
   131|    window.exe_combo = QComboBox()
   132|    # find_executables() wird später bei apply_presets() aufgerufen
   133|    window.exe_combo.currentTextChanged.connect(window.on_exe_changed)
   134|    
   135|    window.model_combo = QComboBox()
   136|    window.update_model_dropdown()
   137|    window.model_combo.currentTextChanged.connect(window.on_model_selected)
   138|    
   139|    window.mmproj_line = QLineEdit()
   140|    window.mmproj_line.setPlaceholderText("Optional: mmproj für Vision-Modelle")
   141|    
   142|    paths_layout.addRow(gettext("lbl_exe_label"), window.exe_combo)
   143|    paths_layout.addRow(gettext("lbl_model_select"), window.model_combo)
   144|    paths_layout.addRow(gettext("lbl_mmproj_vision"), window.mmproj_line)
   145|    
   146|    left_layout.addWidget(paths_frame)
   147|    
   148|    # ========== THEME TOGGLE ==========\
   149|    theme_frame = QFrame()
   150|    theme_frame.setFixedHeight(40)
   151|    theme_layout = QHBoxLayout(theme_frame)
   152|    theme_layout.setContentsMargins(0, 0, 0, 0)
   153|    theme_layout.setSpacing(10)
   154|    
   155|    window.light_theme_checkbox = QCheckBox(gettext("lbl_light_theme"))
   156|    window.light_theme_checkbox.setChecked(False)  # Default: dark theme
   157|    window.light_theme_checkbox.stateChanged.connect(lambda state: window.on_theme_toggled(state))
   158|    
   159|    theme_layout.addStretch()
   160|    theme_layout.addWidget(window.light_theme_checkbox)
   161|    
   162|    left_layout.addWidget(theme_frame)
   163|
   164|  # Parameter Sliders Section
   165|    params_scroll = QScrollArea()
   166|    params_scroll.setWidgetResizable(True)
   167|    # Background set dynamically in apply_theme() - hardcoded here removed
   168|    
   169|    params_widget = QWidget()
   170|    window.params_widget = params_widget  # Save reference for theme updates
   171|    params_layout = QFormLayout(params_widget)
   172|    params_layout.setSpacing(8)
   173|    params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
   174|    params_layout.setHorizontalSpacing(12)
   175|
   176|    window.param_sliders = {}
   177|    for param_key, config in window.PARAM_DEFINITIONS.items():
   178|        if config.get("type") == "float_slider":
   179|            # Float-Slider: Ausgelagert nach float_slider_sync.py
   180|            row_widget, slider_dict = create_float_slider(param_key, config)
   181|            
   182|            # Sync mit Debug-Output
   183|            slider_dict["slider"].valueChanged.connect(window.on_param_changed)
   184|            slider_dict["edit"].textChanged.connect(window.on_param_changed)
   185|            
   186|            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
   187|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   188|            label = QLabel(f"{label_text} ({param_key})")
   189|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   190|            label.setStyleSheet("margin-top: 4px;")
   191|            # Tooltip via tooltip_key übersetzen
   192|            tooltip_key = config.get("tooltip_key")
   193|            if tooltip_key:
   194|                label.setToolTip(gettext(tooltip_key))
   195|            
   196|            params_layout.addRow(label, row_widget)
   197|            window.param_sliders[param_key] = slider_dict
   198|            
   199|        elif config.get("type") == "combo":
   200|            # ComboBox für String-Optionen (cache-type-k/v, flash-attn)
   201|            combo = QComboBox()
   202|            for opt in config["options"]:
   203|                combo.addItem(opt)
   204|            # Default-Wert finden und auswählen
   205|            default_idx = combo.findText(config["default"])
   206|            if default_idx >= 0:
   207|                combo.setCurrentIndex(default_idx)
   208|            combo.setFixedHeight(30)
   209|            
   210|           # Debug-Output live aktualisieren wenn Auswahl geändert wird
   211|            combo.currentTextChanged.connect(window.on_param_changed)
   212|            
   213|            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
   214|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   215|            label = QLabel(f"{label_text} ({param_key})")
   216|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   217|            # Tooltip via tooltip_key übersetzen
   218|            tooltip_key = config.get("tooltip_key")
   219|            if tooltip_key:
   220|                label.setToolTip(gettext(tooltip_key))
   221|
   222|            params_layout.addRow(label, combo)
   223|            window.param_sliders[param_key] = {"combo": combo}
   224|        
   225|        elif config.get("type") == "text_input":
   226|            # Einfaches Textfeld (z.B. --host)
   227|            text_edit = QLineEdit()
   228|            text_edit.setText(config["default"])
   229|            text_edit.setFixedHeight(30)
   230|            
   231|           # Debug-Output live aktualisieren wenn Text geändert wird
   232|            text_edit.textChanged.connect(window.on_param_changed)
   233|            
   234|            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
   235|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   236|            label = QLabel(f"{label_text} ({param_key})")
   237|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   238|            # Tooltip via tooltip_key übersetzen
   239|            tooltip_key = config.get("tooltip_key")
   240|            if tooltip_key:
   241|                label.setToolTip(gettext(tooltip_key))
   242|
   243|            params_layout.addRow(label, text_edit)
   244|            window.param_sliders[param_key] = {"edit": text_edit}
   245|        
   246|        elif config.get("type") == "path_input":
   247|            # Pfad-Eingabe mit Browse-Button (z.B. --slot-save-path)
   248|            row_widget = QWidget()
   249|            row_layout = QHBoxLayout(row_widget)
   250|            
   251|            path_edit = QLineEdit()
   252|            path_edit.setText(config["default"])
   253|            path_edit.setFixedHeight(30)
   254|            
   255|            browse_btn = QPushButton(gettext("btn_browse_model"))
   256|            # Default-Pfad als Startverzeichnis für Dialog setzen
   257|            default_dir = str(Path(config["default"]).parent)
   258|            browse_btn.clicked.connect(lambda p=path_edit, d=default_dir: window.browse_path(p, d))
   259|            browse_btn.setStyleSheet("background-color: #0078d7; color: white; padding: 5px; border-radius: 3px;")
   260|            
   261|            row_layout.addWidget(path_edit, stretch=1)
   262|            row_layout.addWidget(browse_btn)
   263|            
   264|            # Debug-Output live aktualisieren wenn Text geändert wird
   265|            path_edit.textChanged.connect(window.on_param_changed)
   266|            
   267|           # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
   268|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   269|            label = QLabel(f"{label_text} ({param_key})")
   270|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   271|            label.setStyleSheet("margin-top: 9px;")
   272|            
   273|            # Tooltip via tooltip_key übersetzen
   274|            tooltip_key = config.get("tooltip_key")
   275|            if tooltip_key:
   276|                label.setToolTip(gettext(tooltip_key))
   277|
   278|            params_layout.addRow(label, row_widget)
   279|            window.param_sliders[param_key] = {"edit": path_edit}
   280|        
   281|        elif config.get("type") == "file_input":
   282|            # Datei-Eingabe mit Select/Löschen-Buttons (z.B. benchmark_file_path)
   283|            row_widget = QWidget()
   284|            row_layout = QHBoxLayout(row_widget)
   285|            
   286|            file_edit = QLineEdit()
   287|            file_edit.setText(config["default"])
   288|            file_edit.setPlaceholderText(gettext("lbl_no_file_selected"))
   289|            file_edit.setReadOnly(True)
   290|            file_edit.setFixedHeight(30)
   291|            
   292|            select_btn = QPushButton(gettext("btn_select_file"))
   293|            select_btn.setFixedWidth(80)
   294|            select_btn.clicked.connect(lambda: window.on_select_benchmark_file(file_edit))
   295|            select_btn.setStyleSheet("background-color: #0078d7; color: white; padding: 5px; border-radius: 3px;")
   296|            
   297|            clear_btn = QPushButton(gettext("btn_clear_file"))
   298|            clear_btn.setFixedWidth(30)
   299|            clear_btn.clicked.connect(lambda: window.on_clear_benchmark_file(file_edit))
   300|            clear_btn.setStyleSheet("background-color: #0078d7; color: white; padding: 5px; border-radius: 3px;")
   301|            
   302|            row_layout.addWidget(file_edit, stretch=1)
   303|            row_layout.addWidget(select_btn)
   304|            row_layout.addWidget(clear_btn)
   305|            
   306|            # Debug-Output live aktualisieren wenn Text geändert wird
   307|            file_edit.textChanged.connect(window.on_param_changed)
   308|            
   309|             # Übersetztes Label verwenden
   310|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   311|            label = QLabel(f"{label_text} ({param_key})")
   312|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   313|            label.setStyleSheet("margin-top: 9px;")
   314|            tooltip_key = config.get("tooltip_key")
   315|            if tooltip_key:
   316|                label.setToolTip(gettext(tooltip_key))
   317|            
   318|            params_layout.addRow(label, row_widget)
   319|            window.param_sliders[param_key] = {"edit": file_edit}
   320|        
   321|        elif config.get("type") == "slider":
   322|            # Integer-Slider: Ausgelagert nach float_slider_sync.py
   323|            row_widget, slider_dict = create_int_slider(param_key, config)
   324|            
   325|            # Sync mit Debug-Output
   326|            slider_dict["slider"].valueChanged.connect(window.on_param_changed)
   327|            slider_dict["edit"].textChanged.connect(window.on_param_changed)
   328|            
   329|            # Übersetztes Label verwenden (label_key aus JSON resolve via gettext)
   330|            label_text = gettext(config.get("label_key", config.get("label", param_key)))
   331|            label = QLabel(f"{label_text} ({param_key})")
   332|            label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
   333|            label.setStyleSheet("margin-top: 4px;")
   334|            # Tooltip via tooltip_key übersetzen
   335|            tooltip_key = config.get("tooltip_key")
   336|            if tooltip_key:
   337|                label.setToolTip(gettext(tooltip_key))
   338|            
   339|            # Sonderfall: -ngl (GPU layers) mit "all" Checkbox
   340|            if param_key == "-ngl":
   341|                ngl_all_checkbox = QCheckBox(gettext("lbl_all"))
   342|                ngl_all_checkbox.setToolTip(gettext("tooltip_all_layers"))
   343|                ngl_all_checkbox.setChecked(False)
   344|                
   345|                # Sync: Checkbox → Edit-Feld auf "all" setzen
   346|                def on_ngl_checkbox_toggled(checked, edit=slider_dict["edit"]):
   347|                    if checked:
   348|                        edit.setText("all")
   349|                    
   350|                ngl_all_checkbox.toggled.connect(on_ngl_checkbox_toggled)
   351|                
   352|                # Sync: Edit-Feld → Checkbox aktivieren wenn "all" im Feld steht
   353|                def on_ngl_edit_changed(text, checkbox=ngl_all_checkbox):
   354|                    if text.lower() == "all":
   355|                        checkbox.setChecked(True)
   356|                    
   357|                slider_dict["edit"].textChanged.connect(on_ngl_edit_changed)
   358|                
   359|                # Sync: Slider-Value-Änderung deaktiviert Checkbox automatisch
   360|                def on_ngl_slider_changed(value, checkbox=ngl_all_checkbox):
   361|                    checkbox.setChecked(False)
   362|                
   363|                slider_dict["slider"].valueChanged.connect(on_ngl_slider_changed)
   364|                
   365|                # Initial: Wenn Default "all" ist, Checkbox aktivieren und Edit-Feld setzen
   366|                if str(config.get("default", "")).lower() == "all":
   367|                    ngl_all_checkbox.setChecked(True)
   368|                    slider_dict["edit"].setText("all")
   369|                
   370|                # Layout für Label + Checkbox in einem QWidget verpacken
   371|                label_widget = QWidget()
   372|                label_layout = QHBoxLayout(label_widget)
   373|                label_layout.setContentsMargins(0, 0, 0, 0)
   374|                label_layout.addWidget(label)
   375|                label_layout.addWidget(ngl_all_checkbox)
   376|                label_layout.addStretch()
   377|                
   378|                params_layout.addRow(label_widget, row_widget)
   379|                window.ngl_all_checkbox = ngl_all_checkbox  # Für get_current_args verfügbar
   380|            else:
   381|                params_layout.addRow(label, row_widget)
   382|            
   383|            window.param_sliders[param_key] = slider_dict
   384|
   385|    params_scroll.setWidget(params_widget)
   386|    left_layout.addWidget(params_scroll)
   387|    
   388|    splitter.addWidget(left_col)
   389|    
   390|   # ========== RECHTE SPALTE: Debug Output ==========
   391|    debug_frame = QFrame()
   392|    debug_layout = QVBoxLayout(debug_frame)
   393|    
   394|    # Progress Bar for token count
   395|    window.bench_progress_bar = QProgressBar()
   396|    window.bench_progress_bar.setRange(0, 100)
   397|    window.bench_progress_bar.setValue(100)  # Start at 100% when idle
   398|    window.bench_progress_bar.setVisible(True)
   399|    window.bench_progress_bar.setStyleSheet("""
   400|        QProgressBar {
   401|            border: 1px solid gray;
   402|            border-radius: 4px;
   403|            text-align: center;
   404|            background: #1a1a1a;
   405|        }
   406|        QProgressBar::chunk {
   407|            background: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5,
   408|                                        stop:0 #4CAF50, stop:1 #8BC34A);
   409|            border-radius: 3px;
   410|        }
   411|    """)
   412|    debug_layout.addWidget(window.bench_progress_bar)
   413|    
   414|    debug_label = QLabel(gettext("lbl_debug_output"))
   415|    debug_label.setStyleSheet("font-weight: bold;")
   416|    
   417|    window.debug_text = QTextEdit()
   418|    window.debug_text.setReadOnly(True)
   419|    window.debug_text.setFont(QFont("Monospace", 9))
   420|    window.debug_text.setMinimumWidth(500)
   421|    # Prevent cropping of long debug output
   422|    try:
   423|        window.debug_text.setMaximumBlockCount(15000)
   424|        from PyQt6.QtWidgets import QTextEdit as QPlainTextEdit
   425|        window.debug_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
   426|    except AttributeError:
   427|        pass  # Mindestbreite erhöhen
   428|
   429|    copy_btn = QPushButton(gettext("btn_copy"))
   430|    copy_btn.clicked.connect(window.copy_debug)
   431|    
   432|    debug_layout.addWidget(debug_label)
   433|    debug_layout.addWidget(window.debug_text, stretch=1)
   434|    debug_layout.addWidget(copy_btn)
   435|    
   436|    splitter.addWidget(debug_frame)
   437|    
   438|    # Splitter-Position setzen (60% links, 40% rechts)
   439|    splitter.setStretchFactor(0, 3)
   440|    splitter.setStretchFactor(1, 2)
   441|
   442|    main_layout.addWidget(splitter)
   443|
   444|    # ========== STATUS + START/STOP BUTTON ==========
   445|    control_row = QHBoxLayout()
   446|
   447|    window.status_label = QLabel(gettext("status_ready"))
   448|    window.status_label.setObjectName("StatusLabel")
   449|    window.status_label.setMinimumWidth(200)
   450|
   451|    window.start_stop_btn = QPushButton(gettext("btn_start"))
   452|    window.start_stop_btn.clicked.connect(window.toggle_process)
   453|    window.start_stop_btn.setMinimumHeight(40)
   454|    window.start_stop_btn.setStyleSheet("font-size: 16px; padding: 10px;")
   455|
   456|    control_row.addWidget(window.status_label)
   457|    control_row.addWidget(window.start_stop_btn, stretch=1)
   458|
   459|    main_layout.addLayout(control_row)
   460|
   461|    # ========== STATISTIKEN + PRESET BUTTONS ==========
   462|    stats_presets_row = QHBoxLayout()
   463|
   464|    window.stats_label = QLabel(gettext("stats_label"))
   465|
   466|    presets_frame = QFrame()
   467|    presets_layout = QHBoxLayout(presets_frame)
   468|
   469|    save_btn = QPushButton(gettext("btn_save_preset"))
   470|    save_btn.clicked.connect(window.save_preset)
   471|
   472|    load_btn = QPushButton(gettext("btn_load_preset"))
   473|    load_btn.clicked.connect(window.load_preset_dialog)
   474|
   475|      # Benchmark Buttons (Standard + Streaming)
   476|    run_bench_btn = QPushButton(gettext("btn_run_benchmark"))
   477|    run_bench_btn.clicked.connect(window.run_benchmark)
   478|    
   479|    run_bench_streaming_btn = QPushButton(gettext("btn_run_benchmark_live"))
   480|    run_bench_streaming_btn.clicked.connect(window.run_benchmark_streaming)
   481|    
   482|    # Edit prompt button
   483|    edit_prompt_btn = QPushButton("✏️")
   484|    edit_prompt_btn.setToolTip(gettext("tooltip_edit_prompt"))
   485|    edit_prompt_btn.setFixedSize(40, 30)
   486|    edit_prompt_btn.setStyleSheet("font-size: 14px;")
   487|    edit_prompt_btn.clicked.connect(window.edit_prompt_dialog)
   488|    
   489|    # Cancel button for running benchmarks (hidden by default)
   490|    window.cancel_bench_btn = QPushButton(gettext("btn_cancel"))
   491|    window.cancel_bench_btn.setEnabled(False)  # Disabled when no benchmark is running
   492|    window.cancel_bench_btn.setStyleSheet("color: red; font-weight: bold;")
   493|    window.cancel_bench_btn.clicked.connect(window.cancel_benchmark)
   494|
   495|    check_proc_btn = QPushButton(gettext("btn_check_process"))
   496|    check_proc_btn.setToolTip(gettext("tooltip_check_process"))
   497|    check_proc_btn.clicked.connect(window.on_check_process_click)
   498|
   499|    load_args_btn = QPushButton(gettext("btn_load_process_args"))
   500|    load_args_btn.setToolTip(gettext("tooltip_load_process_args"))
   501|