#!/usr/bin/env python3
"""UI persistence for llauncher - saves/restores window geometry and state."""

from pathlib import Path


def restore_geometry(window):
    """Fenster-Position, Größe und Splitter-State laden.
    
    Args:
        window: Main llauncher window instance
    """
    config = load_config_from_window(window)
    
    # Fenster-Position & Größe laden (explizit als Integer)
    x = config.get('window_x')
    y = config.get('window_y')
    width = config.get('window_width')
    height = config.get('window_height')
    
    if all(v is not None for v in [x, y, width, height]):
        try:
            window.move(x, y)
            window.resize(width, height)
        
        except Exception:
            pass
    else:
        # Fallback auf alte Methode wenn keine expliziten Werte da sind
        geom_data = config.get('window_geometry')
        if geom_data:
            try:
                window.restoreGeometry(bytes(geom_data, 'ascii'))
            
            except Exception:
                pass
    
    # Splitter-Position laden (als Integer-Liste, nicht Binary-State)
    if hasattr(window, 'splitter'):
        sizes_data = config.get('splitter_sizes')
        if sizes_data and isinstance(sizes_data, list):
            try:
                window.splitter.setSizes(sizes_data)
            except Exception:
                window.splitter.setSizes([window.width() * 0.6, window.width() * 0.4])


def save_window_geometry(window):
    """Speichert Fenster-Geometrie bei jeder Größenänderung.
    
    Args:
        window: Main llauncher window instance
        event: PyQt6 QResizeEvent (optional, passed by Qt)
    """
    config = load_config_from_window(window)
    
    # Nur Geometrie speichern (Splitter-State zu aggressiv für resizeEvent)
    config['window_x'] = window.x()
    config['window_y'] = window.y()
    config['window_width'] = window.width()
    config['window_height'] = window.height()
    
    save_config_to_window(window, config)


def save_window_state(window):
    """Fenster-Geometrie und Splitter-State speichern + Timer stoppen.
    
    Wird aufgerufen aus closeEvent.
    
    Args:
        window: Main llauncher window instance
        event: PyQt6 QCloseEvent (optional, passed by Qt)
    """
    config = load_config_from_window(window)
    
    # Explizit Breite, Höhe, x, y speichern (robuster als saveGeometry)
    config['window_x'] = window.x()
    config['window_y'] = window.y()
    config['window_width'] = window.width()
    config['window_height'] = window.height()
    
    # Splitter-Position speichern (als Liste von Ints, nicht Binary-State)
    if hasattr(window, 'splitter'):
        try:
            sizes_before = window.splitter.sizes()
            config['splitter_sizes'] = list(sizes_before)
        except Exception:
            pass
    
    # QThread detachten, damit Prozess weiterlaufen kann
    if hasattr(window, 'runner') and window.runner:
        if window.runner.isRunning():
            # Force thread to exit by clearing process reference
            window.runner.force_exit()
            # Clear reference so Qt can destroy the QThread object
            window.runner = None
    
    save_config_to_window(window, config)
    
    # Nur Timer stoppen
    if hasattr(window, 'process_check_timer'):
        window.process_check_timer.stop()


def load_config_from_window(window):
    """Load config from window's config storage."""
    from storage import load_config as storage_load_config
    return storage_load_config()


def save_config_to_window(window, config):
    """Save config to window's config storage."""
    from storage import save_config as storage_save_config
    storage_save_config(config)
