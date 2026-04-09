#!/usr/bin/env python3
"""
GPU-Monitoring für llauncher
Live-Daten von nvidia-smi via QThread.
"""

import subprocess
from PyQt6.QtCore import QThread, pyqtSignal


class GPUMonitor(QThread):
    """Live-GPU-Monitoring via nvidia-smi."""

    gpu_update = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                # Query power draw (in watts)
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.total,memory.used,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    values = [v.strip() for v in result.stdout.strip().split(",")]
                    # First 5 values are ints, last one (power.draw) may be float like "45.5"
                    parsed_values = []
                    for i, v in enumerate(values[:5]):
                        try:
                            parsed_values.append(int(v))
                        except ValueError:
                            parsed_values.append(0)
                    # Power draw is decimal string like "45.5W", parse just the number
                    try:
                        power_str = values[5].split()[0]  # Remove "W" suffix
                        power_draw = float(power_str)
                    except (IndexError, ValueError):
                        power_draw = 0.0
                    
                    self.gpu_update.emit({
                        "gpu_usage": parsed_values[0],
                        "mem_usage": parsed_values[1],
                        "temp": parsed_values[2],
                        "total_mb": parsed_values[3],
                        "used_mb": parsed_values[4],
                        "power_draw": power_draw,
                    })
            except Exception:
                pass  # Nichts anzeigen wenn GPU nicht verfügbar
            self.msleep(2000)

    def stop(self):
        """Thread sauber beenden."""
        self._running = False
        self.wait()


def update_gpu_display(label, data: dict):
    """Aktualisiert ein QLabel mit GPU-Statistiken."""
    if "gpu_usage" in data:
        power_str = f"{data.get('power_draw', 0.0):.1f}W" if data.get("power_draw") else "--W"
        stats = f"GPU: {data['gpu_usage']}% | VRAM: {data['used_mb']}/{data['total_mb']}MB | Temp: {data['temp']}°C | Power: {power_str}"
        label.setText(stats)
