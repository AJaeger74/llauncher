#!/usr/bin/env python3
"""Process inspection for llauncher - detects and inspects running llama.cpp processes."""

import subprocess
import shlex


def check_existing_process(window):
    """Prüft ob bereits ein llama-server läuft und passt UI entsprechend an.
    
    Args:
        window: Main llauncher window instance (provides UI access)
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 or not result.stdout.strip():
            # Kein Prozess läuft - UI zurücksetzen, aber Progress Bar auf 0% lassen
            window.status_label.setText(window.t("status_ready"))
            window.status_label.setStyleSheet("")
            window.start_stop_btn.setText(window.t("btn_start"))
            window.start_stop_btn.setObjectName("StartButton")
            # Progress bar bleibt bei 0% - wird durch toggle_process() nach Stop gesetzt
            return
        
        pids = [int(pid) for pid in result.stdout.strip().split() if pid.isdigit()]
        
        # Prüfen ob einer der Prozesse vom User gestartet wurde (eigener Prozess)
        for pid in pids:
            cmdline_path = f'/proc/{pid}/cmdline'
            try:
                with open(cmdline_path, 'r') as f:
                    content = f.read()
                
                args = [arg for arg in content.split('\x00') if arg]
                if not args or 'llama-server' not in args[0]:
                    continue
                
                # Kommandozeile zusammenbauen
                full_cmd = " ".join(shlex.quote(arg) for arg in args)
                
                # UI anpassen: Button auf "Stop" setzen
                # Aber nicht wenn gerade ein Benchmark läuft!
                if not getattr(window, 'benchmark_running', False):
                    # Nur Button-Status aktualisieren, nicht den Label-Text
                    # (on_output() verwaltet Idle/Running Status korrekt)
                    window.start_stop_btn.setText(window.t("btn_stop"))
                    window.start_stop_btn.setObjectName("StopButton")
                
                # Runner als "externer" Prozess markieren
                # Wir speichern PID und args, können aber nicht über QThread steuern
                window.external_runner_pid = pid
                window.external_runner_args = args
                
                return  # Nur erster laufender Prozess relevant
            
            except (FileNotFoundError, PermissionError, IOError):
                continue
    
    except Exception:
        pass


def get_running_server_command(window):
    """Liest Kommandozeile von laufendem llama-server Prozess aus /proc.
    
    Args:
        window: Main llauncher window instance
        
    Returns:
        str: Full command line of running llama-server, or None if not found
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        pids = [int(pid) for pid in result.stdout.strip().split() if pid.isdigit()]
        
        for pid in pids:
            cmdline_path = f'/proc/{pid}/cmdline'
            try:
                with open(cmdline_path, 'r') as f:
                    content = f.read()
                
                args = [arg for arg in content.split('\x00') if arg]
                if not args or 'llama-server' not in args[0]:
                    continue
                
                # Vollständige Kommandozeile zusammenbauen
                full_cmd = " ".join(shlex.quote(arg) for arg in args)
                return full_cmd
            
            except (FileNotFoundError, PermissionError, IOError):
                continue
        
        return None
    
    except Exception:
        return None
