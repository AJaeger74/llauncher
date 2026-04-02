#!/usr/bin/env python3
"""Fetch running model info from llama.cpp server via HTTP."""

import json
import urllib.request
import urllib.error


def fetch_running_model_info(server_url: str = "http://127.0.0.1:8080") -> dict | None:
    """Prüft ob Server läuft und ermitteln geladenes Modell + Parameter.
    
    Returns dict mit keys:
      - model_name: Name der GGUF-Datei (aus /v1/models)
      - params: Dict von Parametern aus /metrics oder /status
    
    Returns None wenn Server nicht erreichbar.
    """
    try:
        # Prüfe ob Server läuft via /v1/models
        models_url = f"{server_url}/v1/models"
        req = urllib.request.Request(models_url)
        
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.load(resp)
        
        if not body.get("data"):
            return None
        
        # Erster Eintrag ist das geladene Modell
        model_data = body["data"][0]
        model_name = model_data.get("id", "unknown")
        
        # Versuche Parameter aus /metrics zu holen (falls verfügbar)
        params = {}
        try:
            metrics_url = f"{server_url}/metrics"
            req = urllib.request.Request(metrics_url)
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                metrics_text = resp.read().decode("utf-8")
            
            # Parse Prometheus-format metrics
            for line in metrics_text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith("llama_"):
                    metric_name = parts[0]
                    value = float(parts[1])
                    
                    # Extrahiere relevante Parameter aus Metric-Namen
                    if "ctx_size" in metric_name:
                        params["ctx"] = int(value)
                    elif "n_batch" in metric_name:
                        params["batch"] = int(value)
                    elif "n_threads" in metric_name:
                        params["threads"] = int(value)
                    elif "n_gpu_layers" in metric_name:
                        params["ngl"] = int(value)
        
        except Exception:
            # Metrics nicht verfügbar – ignoriere
            pass
        
        return {
            "model_name": model_name,
            "params": params,
        }
    
    except urllib.error.URLError:
        return None
    except Exception:
        return None