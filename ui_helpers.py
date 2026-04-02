#!/usr/bin/env python3
"""
UI-Helfer für llauncher
Erstellt Parameter-Widgets basierend auf PARAM_DEFINITIONS.
Modularisierung von init_ui() in llauncher.py.
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

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QLabel, QLineEdit, QPushButton, QSlider, QWidget, QHBoxLayout,
    QFileDialog
)

from direct_slider import DirectClickSlider


def create_integer_slider(param_key: str, config: dict):
    """Erstellt Integer-Slider mit editierbarem Wert-Display
    
    Args:
        param_key: Parameter-Key (z.B. "-c", "-t")
        config: Parametereinstellungen aus PARAM_DEFINITIONS
        
    Returns:
        Dict mit {"slider": DirectClickSlider, "edit": QLineEdit}
    """
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    
    # Slider erstellen
    slider = DirectClickSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(config["min"])
    slider.setMaximum(config["max"])
    slider.setValue(config["default"])
    slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    slider.setFixedHeight(30)
    slider.setTickPosition(QSlider.TickPosition.NoTicks)
    slider.setTickInterval(0)
    
    # Wert-Display rechts (editierbar)
    max_width = len(str(config["max"])) * 9 + 15
    value_edit = QLineEdit()
    value_edit.setText(f"{config['default']}")
    value_edit.setMinimumWidth(max_width)
    value_edit.setMaximumWidth(max_width)
    value_edit.setReadOnly(False)
    value_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
    
    # Sync: Slider -> Edit (bei Bewegung)
    def sync_and_update(v, p=param_key, edit=value_edit):
        edit.setText(str(v))
    
    slider.valueChanged.connect(sync_and_update)
    
    # Sync: Edit -> Slider (bei Eingabe + Enter/Tab)
    def make_slider_sync(slider=slider, key=param_key):
        def handler(text):
            try:
                val = int(text)
                slider.setValue(val)
            except (ValueError, TypeError):
                pass  # Leeres Feld oder ungültig – Slider nicht ändern
        return handler
    
    value_edit.textChanged.connect(make_slider_sync())
    
    row_layout.addWidget(slider, stretch=1)
    row_layout.addWidget(value_edit)
    
    label = QLabel(f"{config['label']} ({param_key})")
    # Tooltip via tooltip_key übersetzen
    tooltip_key = config.get("tooltip_key")
    if tooltip_key:
        label.setToolTip(gettext(tooltip_key))
    
    # Label zum Layout hinzufügen (wird von caller gemacht)
    return {"slider": slider, "edit": value_edit}, row_widget, label


def create_float_slider(param_key: str, config: dict):
    """Erstellt Float-Slider mit editierbarem Wert-Display
    
    Args:
        param_key: Parameter-Key (z.B. "--temp", "--top-p")
        config: Parametereinstellungen aus PARAM_DEFINITIONS
        
    Returns:
        Dict mit {"slider": DirectClickSlider, "edit": QLineEdit}
    """
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    
    multiplier = 10  # Skalierung für Dezimalstellen (0.1 → 1, 2.0 → 20)
    slider = DirectClickSlider(Qt.Orientation.Horizontal, multiplier=multiplier)
    slider.setMinimum(int(config["min"] * multiplier))
    slider.setMaximum(int(config["max"] * multiplier))
    slider.setValue(int(config["default"] * multiplier))
    slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    slider.setFixedHeight(30)
    
    # Wert-Display: Breite passend zum maximalen Wert (2 Stellen + Punkt = 4 Zeichen)
    value_edit = QLineEdit()
    value_edit.setValidator(QDoubleValidator(0.1, config["max"], 2))
    value_edit.setText(f"{config['default']:.2f}")
    max_width = len(f"{config['max']:.2f}") * 9 + 10  # Zeichenbreite × Font-Width + Padding
    value_edit.setMinimumWidth(max_width)
    value_edit.setMaximumWidth(max_width)
    
    # Sync: Slider -> Edit (bei Bewegung)
    def sync_from_slider(v, p=param_key, target=value_edit):
        float_val = v / 10.0
        target.setText(f"{float_val:.2f}")
    
    slider.valueChanged.connect(sync_from_slider)
    
    # Sync: Edit -> Slider (bei Eingabe + Enter/Tab)
    def make_sync_handler(slider=slider, edit=value_edit, key=param_key, mult=multiplier):
        def handler(text):
            try:
                val = float(text)
                int_val = int(val * mult)
                slider.setValue(int_val)
                # Text direkt setzen (nicht über Handler)
                edit.setText(f"{val:.2f}")
            except (ValueError, TypeError):
                pass
        return handler
    
    value_edit.textChanged.connect(make_sync_handler())
    
    row_layout.addWidget(slider, stretch=1)
    row_layout.addWidget(value_edit)
    
    label = QLabel(f"{config['label']} ({param_key})")
    # Tooltip via tooltip_key übersetzen
    tooltip_key = config.get("tooltip_key")
    if tooltip_key:
        label.setToolTip(gettext(tooltip_key))
    
    return {"slider": slider, "edit": value_edit}, row_widget, label


def create_combo_slider(param_key: str, config: dict):
    """Erstellt ComboBox für String-Optionen (cache-type-k/v, flash-attn)
    
    Args:
        param_key: Parameter-Key
        config: Parametereinstellungen aus PARAM_DEFINITIONS
        
    Returns:
        Dict mit {"combo": QComboBox}
    """
    combo = QComboBox()
    for opt in config["options"]:
        combo.addItem(opt)
    # Default-Wert finden und auswählen
    default_idx = combo.findText(config["default"])
    if default_idx >= 0:
        combo.setCurrentIndex(default_idx)
    combo.setFixedHeight(30)
    
    label = QLabel(f"{config['label']} ({param_key})")
    # Tooltip via tooltip_key übersetzen
    tooltip_key = config.get("tooltip_key")
    if tooltip_key:
        label.setToolTip(gettext(tooltip_key))
    
    return {"combo": combo}, combo, label


def create_text_input(param_key: str, config: dict):
    """Erstellt einfaches QLineEdit für String-Parameter (--host)
    
    Args:
        param_key: Parameter-Key
        config: Parametereinstellungen aus PARAM_DEFINITIONS
        
    Returns:
        Dict mit {"edit": QLineEdit}
    """
    text_edit = QLineEdit()
    text_edit.setText(config["default"])
    text_edit.setFixedHeight(30)
    
    label = QLabel(f"{config['label']} ({param_key})")
    # Tooltip via tooltip_key übersetzen
    tooltip_key = config.get("tooltip_key")
    if tooltip_key:
        label.setToolTip(gettext(tooltip_key))
    
    return {"edit": text_edit}, text_edit, label


def create_path_input(param_key: str, config: dict):
    """Erstellt Path-Input mit QLineEdit + Browse Button (--slot-save-path)
    
    Args:
        param_key: Parameter-Key
        config: Parametereinstellungen aus PARAM_DEFINITIONS
        
    Returns:
        Dict mit {"edit": QLineEdit}
    """
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    
    path_edit = QLineEdit()
    path_edit.setText(config["default"])
    path_edit.setFixedHeight(30)
    
    browse_btn = QPushButton("...")
    browse_btn.setFixedWidth(30)
    
    # Closure mit Default-Argument für param_key und path_edit
    def make_browse_handler(pkey=param_key, edit=path_edit):
        def browse():
            dialog = QFileDialog(None, "Pfad wählen", config["default"])
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            if dialog.exec():
                selected = dialog.selectedFiles()
                if selected:
                    path_edit.setText(selected[0])
        return browse
    
    browse_btn.clicked.connect(make_browse_handler())
    
    row_layout.addWidget(path_edit, stretch=1)
    row_layout.addWidget(browse_btn)
    
    label = QLabel(f"{config['label']} ({param_key})")
    # Tooltip via tooltip_key übersetzen
    tooltip_key = config.get("tooltip_key")
    if tooltip_key:
        label.setToolTip(gettext(tooltip_key))
    
    return {"edit": path_edit}, row_widget, label
