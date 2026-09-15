#!/usr/bin/env python3
"""
settings_dialog – Settings window for llauncher

Dialog zum Ändern von Theme (light/dark) und Sprache.
"""

from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QLineEdit, QWidget, QFrame
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
QLineEdit { padding: 5px; border-radius: 3px; background-color: #444; color: white; border: 1px solid #555; font-family: monospace; }
"""

LIGHT_THEME = """
QDialog { background-color: #ffffff; }
QLabel { color: #333333; }
QPushButton { background-color: #0078d7; color: white; padding: 10px; border-radius: 3px; }
QPushButton:hover { background-color: #006cc1; }
QComboBox { padding: 5px; border-radius: 3px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; }
QLineEdit { padding: 5px; border-radius: 3px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; font-family: monospace; }
"""


class SettingsDialog(QDialog):
    """Settings dialog for theme and language preferences."""
    
    settings_changed = pyqtSignal(bool, str)  # light_theme, language
    
    def __init__(self, parent=None, current_light_theme: bool = False, current_language: str = 'en', lang_reload_callback=None,
                 launch_cmd_start: str = "", launch_cmd_finish: str = ""):
        super().__init__(parent)
        self.current_light_theme = current_light_theme
        self.current_language = current_language
        self.lang_reload_callback = lang_reload_callback
        self._launch_cmd_start = launch_cmd_start or ""
        self._launch_cmd_finish = launch_cmd_finish or ""
        
        self.setup_ui()
        self.apply_theme(self.current_light_theme)
        self.setWindowTitle(gettext("settings_title"))
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Theme section
        theme_label = QLabel(gettext("lbl_theme_setting"))
        theme_label.setObjectName("theme_label")
        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("theme_combo")
        self.theme_combo.addItem(gettext("theme_dark"), "dark")
        self.theme_combo.addItem(gettext("theme_light"), "light")
        
        # Set current selection
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == ("light" if self.current_light_theme else "dark"):
                self.theme_combo.setCurrentIndex(i)
                break
        
        theme_row = QHBoxLayout()
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        
        layout.addLayout(theme_row)
        
        # Language section
        lang_label = QLabel(gettext("lbl_language_setting"))
        lang_label.setObjectName("lang_label")
        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("lang_combo")
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

        # Launch commands section (executed during manual model start)
        cmd_group = QFrame()
        cmd_group.setObjectName("cmd_group")
        cmd_layout = QVBoxLayout(cmd_group)
        cmd_layout.setContentsMargins(10, 10, 10, 10)

        cmd_section_label = QLabel(gettext("lbl_launch_commands"))
        cmd_section_label.setObjectName("cmd_section_label")
        cmd_section_label.setStyleSheet("font-weight: bold;")
        cmd_layout.addWidget(cmd_section_label)

        cmd_start_label = QLabel(gettext("lbl_launch_cmd_start"))
        cmd_start_label.setObjectName("cmd_start_label")
        self.cmd_start_edit = QLineEdit()
        self.cmd_start_edit.setObjectName("cmd_start_edit")
        self.cmd_start_edit.setPlaceholderText(gettext("placeholder_launch_cmd_start"))
        self.cmd_start_edit.setToolTip(gettext("tooltip_launch_cmd_start"))
        self.cmd_start_edit.setText(self._launch_cmd_start)

        cmd_finish_label = QLabel(gettext("lbl_launch_cmd_finish"))
        cmd_finish_label.setObjectName("cmd_finish_label")
        self.cmd_finish_edit = QLineEdit()
        self.cmd_finish_edit.setObjectName("cmd_finish_edit")
        self.cmd_finish_edit.setPlaceholderText(gettext("placeholder_launch_cmd_finish"))
        self.cmd_finish_edit.setToolTip(gettext("tooltip_launch_cmd_finish"))
        self.cmd_finish_edit.setText(self._launch_cmd_finish)

        cmd_layout.addWidget(cmd_start_label)
        cmd_layout.addWidget(self.cmd_start_edit)
        cmd_layout.addWidget(cmd_finish_label)
        cmd_layout.addWidget(self.cmd_finish_edit)
        layout.addWidget(cmd_group)
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton(gettext("btn_save_settings"))
        save_btn.setObjectName("save_btn")
        cancel_btn = QPushButton(gettext("btn_cancel"))
        cancel_btn.setObjectName("cancel_btn")
        
        save_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.setSpacing(10)
        
        layout.addLayout(btn_layout)
    
    def update_ui_text(self):
        """Update all UI text after language reload."""
        from i18n import gettext
        self.setWindowTitle(gettext("settings_title"))
        theme_label = self.findChild(QLabel, "theme_label")
        if theme_label:
            theme_label.setText(gettext("lbl_theme_setting"))
        theme_combo = self.findChild(QComboBox, "theme_combo")
        if theme_combo:
            theme_combo.setItemText(0, gettext("theme_dark"))
            theme_combo.setItemText(1, gettext("theme_light"))
        lang_label = self.findChild(QLabel, "lang_label")
        if lang_label:
            lang_label.setText(gettext("lbl_language_setting"))
        save_btn = self.findChild(QPushButton, "save_btn")
        cancel_btn = self.findChild(QPushButton, "cancel_btn")
        if save_btn:
            save_btn.setText(gettext("btn_save_settings"))
        if cancel_btn:
            cancel_btn.setText(gettext("btn_cancel"))
        cmd_section_label = self.findChild(QLabel, "cmd_section_label")
        if cmd_section_label:
            cmd_section_label.setText(gettext("lbl_launch_commands"))
        cmd_start_label = self.findChild(QLabel, "cmd_start_label")
        if cmd_start_label:
            cmd_start_label.setText(gettext("lbl_launch_cmd_start"))
        cmd_finish_label = self.findChild(QLabel, "cmd_finish_label")
        if cmd_finish_label:
            cmd_finish_label.setText(gettext("lbl_launch_cmd_finish"))
        cmd_start_edit = self.findChild(QLineEdit, "cmd_start_edit")
        if cmd_start_edit:
            cmd_start_edit.setPlaceholderText(gettext("placeholder_launch_cmd_start"))
            cmd_start_edit.setToolTip(gettext("tooltip_launch_cmd_start"))
        cmd_finish_edit = self.findChild(QLineEdit, "cmd_finish_edit")
        if cmd_finish_edit:
            cmd_finish_edit.setPlaceholderText(gettext("placeholder_launch_cmd_finish"))
            cmd_finish_edit.setToolTip(gettext("tooltip_launch_cmd_finish"))
    
    def apply_theme(self, use_light: bool):
        """Apply theme to dialog."""
        theme = LIGHT_THEME if use_light else DARK_THEME
        self.setStyleSheet(theme)
    
    def accept(self):
        """Handle save button click - return flag for app restart if language changed."""
        _, new_lang, _, _ = self.get_settings()
        self.restart_on_language_change = (new_lang != self.current_language)
        super().accept()
    
    def get_settings(self):
        """Return current settings from dialog."""
        selected_theme = self.theme_combo.currentData()
        light_mode = (selected_theme == "light")
        selected_lang = self.lang_combo.currentData()

        launch_cmd_start = self.cmd_start_edit.text().strip() if hasattr(self, 'cmd_start_edit') else ""
        launch_cmd_finish = self.cmd_finish_edit.text().strip() if hasattr(self, 'cmd_finish_edit') else ""

        return light_mode, selected_lang, launch_cmd_start, launch_cmd_finish
