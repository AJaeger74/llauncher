#!/usr/bin/env python3
"""
settings_dialog – Settings window for llauncher

Dialog zum Ändern von Theme (light/dark) und Sprache.
"""

from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QWidget, QFrame
)

# Import i18n gettext function
try:
    from i18n import I18nManager
    gettext = I18nManager.get_instance().gettext
except ImportError:
    def gettext(key):
        return key


DARK_THEME = """
QDialog { background-color: #2d2d2d; }
QLabel { color: #cccccc; }
QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
QPushButton:hover { background-color: #006cc1; }
QComboBox { padding: 5px; border-radius: 3px; background-color: #444; color: white; }
"""

LIGHT_THEME = """
QDialog { background-color: #ffffff; }
QLabel { color: #333333; }
QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
QPushButton:hover { background-color: #006cc1; }
QComboBox { padding: 5px; border-radius: 3px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
"""


class SettingsDialog(QDialog):
    """Settings dialog for theme and language preferences."""
    
    settings_changed = pyqtSignal(bool, str)  # light_theme, language
    
    def __init__(self, parent=None, current_light_theme: bool = False, current_language: str = 'en'):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.current_language = current_language
        
        self.setup_ui()
        self.apply_theme(self.current_light_theme)
        self.setWindowTitle(gettext("settings_title"))
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Theme section
        theme_label = QLabel(gettext("lbl_theme_setting"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(gettext("theme_dark"), "dark")
        self.theme_combo.addItem(gettext("theme_light"), "light")
        
        # Set current selection
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == ("dark" if self.current_light_theme else "light"):
                self.theme_combo.setCurrentIndex(i)
                break
        
        theme_row = QHBoxLayout()
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        
        layout.addLayout(theme_row)
        
        # Language section
        lang_label = QLabel(gettext("lbl_language_setting"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English (US)", "en")
        self.lang_combo.addItem("Deutsch (DE)", "de")
        
        # Set current selection
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == self.current_language:
                self.lang_combo.setCurrentIndex(i)
                break
        
        lang_row = QHBoxLayout()
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        
        layout.addLayout(lang_row)
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton(gettext("btn_save_settings"))
        cancel_btn = QPushButton(gettext("btn_cancel"))
        
        save_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.setSpacing(10)
        
        layout.addLayout(btn_layout)
    
    def apply_theme(self, use_light: bool):
        """Apply theme to dialog."""
        theme = LIGHT_THEME if use_light else DARK_THEME
        self.setStyleSheet(theme)
    
    def get_settings(self):
        """Return current settings from dialog."""
        selected_theme = self.theme_combo.currentData()
        light_mode = (selected_theme == "light")
        selected_lang = self.lang_combo.currentData()
        
        return light_mode, selected_lang
