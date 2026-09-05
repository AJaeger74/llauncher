#!/usr/bin/env python3
"""
GPU-Monitoring für llauncher
Live-Daten von nvidia-smi via QThread.
"""

import subprocess
import json
from PyQt6.QtCore import QThread, pyqtSignal


class GPUMonitor(QThread):
    """Live-GPU-Monitoring via nvidia-smi + llama.cpp /slots API."""

    gpu_update = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._running = False
        self._window = None  # set by start_gpu_monitor() for dynamic host resolution
        self.slots_port = "8080"
        self.slots_url = "http://localhost:8080/slots"  # legacy fallback

    def _get_slots_url(self):
        """Build slots URL from the window's --host field or fallback to default."""
        from ui_helpers import sanitize_host, host_port

        if hasattr(self, "_window") and self._window is not None:
            host_edit = getattr(self._window, "param_sliders", {}).get("--host", {}).get("edit")
            if host_edit:
                host_text = host_edit.text().strip()
                if host_text:
                    return f"http://{sanitize_host(host_text)}:{host_port(host_text, self.slots_port)}/slots"
        return f"http://localhost:{self.slots_port}/slots"

    def _query_slots(self):
        """Query /slots API for live context usage — always returns data if server is up."""
        try:
            url = self._get_slots_url()
            result = subprocess.run(
                ["curl", "-s", "--max-time", "3", url],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            slots = json.loads(result.stdout)
            if not slots or not isinstance(slots, list):
                return None
            # Use first slot (even if idle) — always show context info when server is up
            slot = slots[0]
            ctx = slot.get("n_ctx", 0)
            prompt = slot.get("n_prompt_tokens", 0)
            decoded = slot.get("next_token", [{}])[0].get("n_decoded", 0)
            kv_bpv = slot.get("kv_bpv", 0)
            spec = slot.get("speculative", False)
            is_processing = slot.get("is_processing", False)
            params = slot.get("params", {})
            spec_types_str = params.get("speculative.types", "")
            # Extract the active spec type (non-"none" type from comma-separated list)
            spec_type = ""
            if spec and spec_types_str:
                types = [t.strip() for t in spec_types_str.split(",")]
                active = [t for t in types if t != "none"]
                if active:
                    spec_type = active[0]
            return {
                "ctx": ctx,
                "prompt_tokens": prompt,
                "decoded_tokens": decoded,
                "used_tokens": prompt + decoded,
                "kv_bpv": kv_bpv,
                "shed_offer": slot.get("cotenancy", {}).get("shed_offer", 0),
                "speculative": spec,
                "spec_type": spec_type,
                "is_processing": is_processing,
            }
        except Exception:
            return None

    def run(self):
        self._running = True
        while self._running:
            gpu_data = {}
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

                    gpu_data = {
                        "gpu_usage": parsed_values[0],
                        "mem_usage": parsed_values[1],
                        "temp": parsed_values[2],
                        "total_mb": parsed_values[3],
                        "used_mb": parsed_values[4],
                        "power_draw": power_draw,
                    }
            except Exception:
                pass  # Nichts anzeigen wenn GPU nicht verfügbar

            slots = self._query_slots()
            if slots:
                gpu_data["slots"] = slots

            if gpu_data:
                self.gpu_update.emit(gpu_data)
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
