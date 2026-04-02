#!/usr/bin/env python3
"""
Float-Slider Sync – Helper für Slider mit Edit-Feldern.

Enthält:
- DirectClickSlider (für Integer und Float via multiplier)
- create_float_slider() – generiert Float-Slider + Edit-Kombination
- create_int_slider() – generiert Integer-Slider + Edit-Kombination
- Closure-Fixes über Default-Argumente
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (QLineEdit, QHBoxLayout, QWidget, QSlider,
                             QStyleOptionSlider)
from PyQt6.QtWidgets import QStyle


class DirectClickSlider(QSlider):
    """QSlider der direkt zur Klickposition springt (nicht nur Drag-only)."""
    
    def __init__(self, orientation=Qt.Orientation.Horizontal, multiplier=1.0):
        super().__init__(orientation)
        self.multiplier = multiplier  # Skalierung für Float-Werte (1.0 = Integer, 10/100 = Float)
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        """Prüft ob Klick innerhalb des Slider-Bereichs war -> Drag oder Springen."""
        # Klick auf Slider-Fläche gilt als "auf Handle" wenn innerhalb von ±10px um Handle-Mitte
        if not self._is_click_on_handle(event):
            # Klick außerhalb: Springen zur Position
            self._set_value_at_position(event)
        else:
            # Auf Handle geklickt -> Qt muss Drag starten (super() Aufruf nötig!)
            super().mousePressEvent(event)
    
    def _is_click_on_handle(self, event):
        """Prüft ob Klick innerhalb des Slider-Knopfs war oder ±10px drumherum."""
        if self.orientation() == Qt.Orientation.Horizontal:
            click_x = int(event.position().x())
            style_option = QStyleOptionSlider()
            self.initStyleOption(style_option)
            
            handle_rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, style_option,
                QStyle.SubControl.SC_SliderHandle, self
            )
            
            # Erweiterten Bereich prüfen: Handle-Mitte ±10px
            handle_center_x = (handle_rect.left() + handle_rect.right()) // 2
            tolerance = 10  # Pixel Toleranz um das Handle
            return abs(click_x - handle_center_x) <= tolerance
        else:
            click_y = int(event.position().y())
            style_option = QStyleOptionSlider()
            self.initStyleOption(style_option)
            
            handle_rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, style_option,
                QStyle.SubControl.SC_SliderHandle, self
            )
            
            # Erweiterten Bereich prüfen: Handle-Mitte ±10px
            handle_center_y = (handle_rect.top() + handle_rect.bottom()) // 2
            tolerance = 10  # Pixel Toleranz um das Handle
            return abs(click_y - handle_center_y) <= tolerance
    
    def _set_value_at_position(self, event):
        """Springt zur Klickposition und setzt Wert."""
        if self.orientation() == Qt.Orientation.Horizontal:
            click_x = int(event.position().x())
            slider_width = self.width()
            
            if slider_width <= 0:
                return
            
            ratio = click_x / slider_width
            min_val = self.minimum()
            max_val = self.maximum()
            range_size = max_val - min_val
            
            raw_value = int(min_val + ratio * range_size)
            value = max(self.minimum(), min(self.maximum(), raw_value))
            
            self.setValue(value)
        else:
            click_y = int(event.position().y())
            slider_height = self.height()
            
            if slider_height <= 0:
                return
            
            ratio = 1.0 - (click_y / slider_height)
            min_val = self.minimum()
            max_val = self.maximum()
            range_size = max_val - min_val
            
            raw_value = int(min_val + ratio * range_size)
            value = max(self.minimum(), min(self.maximum(), raw_value))
            
            self.setValue(value)


def create_float_slider(param_key, config):
    """
    Erstellt Float-Slider mit Edit-Feld und bidirektionalem Sync.
    
    Args:
        param_key: Parameter-Key (z.B. "--temp")
        config: Dict mit "min", "max", "default", "step"
        
    Returns:
        Tuple: (slider_widget, param_sliders_dict)
            - slider_widget: QWidget mit QHBoxLayout (Slider + Edit)
            - param_sliders_dict: {"slider": DirectClickSlider, "edit": QLineEdit}
    """
    multiplier = 10.0  # Standard für eine Dezimalstelle
    
    return _create_slider_row(param_key, config, multiplier)


def create_int_slider(param_key, config):
    """
    Erstellt Integer-Slider mit Edit-Feld und bidirektionalem Sync.
    
    Args:
        param_key: Parameter-Key (z.B. "-c")
        config: Dict mit "min", "max", "default"
        
    Returns:
        Tuple: (slider_widget, param_sliders_dict)
            - slider_widget: QWidget mit QHBoxLayout (Slider + Edit)
            - param_sliders_dict: {"slider": DirectClickSlider, "edit": QLineEdit}
    """
    return _create_slider_row(param_key, config, multiplier=1.0)


def _create_slider_row(param_key, config, multiplier):
    """
    Interner Helper zum Erstellen von Slider+Edit-Row.
    
    Args:
        param_key: Parameter-Key (z.B. "-c", "--temp")
        config: Dict mit "min", "max", "default" (+ optional "step")
        multiplier: Skalierung (1.0 = Integer, 10/100 = Float)
        
    Returns:
        Tuple: (row_widget, param_sliders_dict)
    """
    # Widget-Layout erstellen
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    
    # Slider mit Int-Skalierung
    slider = DirectClickSlider(Qt.Orientation.Horizontal, multiplier=multiplier)
    slider.setMinimum(int(config["min"] * multiplier))
    slider.setMaximum(int(config["max"] * multiplier))
    slider.setValue(int(config["default"] * multiplier))
    slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    slider.setFixedHeight(30)
    
    # Tick Marks entfernen für sauberes DirectClick-Verhalten
    slider.setTickPosition(QSlider.TickPosition.NoTicks)
    slider.setTickInterval(0)
    
    # Edit-Feld mit QDoubleValidator (funktioniert auch für Integer)
    value_edit = QLineEdit()
    if multiplier > 1.0:
        # Float-Slider: QDoubleValidator mit zwei Dezimalstellen
        value_edit.setValidator(QDoubleValidator(0.1, config["max"], 2))
        max_width = len(f"{config['max']:.2f}") * 9 + 10
    else:
        # Integer-Slider: Breite basierend auf max-Wert (z.B. "65536")
        value_edit.setValidator(None)  # Keine Validator-Beschränkung für Integer
        max_width = len(str(config["max"])) * 9 + 15
    
    value_edit.setText(f"{config['default']:.2f}" if multiplier > 1.0 else f"{config['default']}")
    value_edit.setMinimumWidth(max_width)
    value_edit.setMaximumWidth(max_width)
    
    if multiplier == 1.0:
        # Integer-Slider: Rechtsbündig
        value_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
    
     # Sync von Slider → Edit (mit Closure-Fix via Default-Argumente)
    def sync_from_slider(v, p=param_key, target=value_edit):
        float_val = v / multiplier
        if multiplier > 1.0:
            target.setText(f"{float_val:.2f}")
        else:
            target.setText(str(int(float_val)))
    
    slider.valueChanged.connect(sync_from_slider)
    
    # Sync von Edit → Slider (mit Closure-Fix via Default-Argumente)
    def make_sync_handler(slider=slider, edit=value_edit, key=param_key, mult=multiplier):
        def handler(text):
            try:
                val = float(text)
                int_val = int(val * mult)
                slider.setValue(int_val)
                # Text direkt setzen (nicht über Handler)
                if mult > 1.0:
                    edit.setText(f"{val:.2f}")
                else:
                    edit.setText(str(int(val)))
            except (ValueError, TypeError):
                pass
        return handler
    
    value_edit.textChanged.connect(make_sync_handler())
    
    row_layout.addWidget(slider, stretch=1)
    row_layout.addWidget(value_edit)
    
    param_sliders_dict = {"slider": slider, "edit": value_edit}
    
    return row_widget, param_sliders_dict