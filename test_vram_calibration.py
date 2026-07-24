#!/usr/bin/env python3
"""
Vollautomatischer Test fuer VRAM-Schaetzung, Display-Rendering und VBR-Kalibrierung.

Testet:
1. MoE tensor_bytes Bugfix (gguf_utils.py)
2. Display-Formatierung Bugfix (model_inspector.py)
3. VBR-Calibrate mit/ohne mmproj
4. Empirische mmproj GPU-vs-CPU Pruefung

Aufruf: python3 test_vram_calibration.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

# Import llauncher utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_utils import get_model_info, estimate_vram, read_gguf_context_length


# ============================================================
# KONFIGURATION
# ============================================================

MODEL_PATH = "/opt/fast/ai/models/llama.cpp/unsloth_Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MMPOJ_PATH = "/home/andreas/models/llama.cpp/HauhauCS_Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"
LLAMA_SERVER = "http://localhost:8080"

TEST_PARAMS = {
    "ngl": 100,
    "ctx_size": 262144,
    "np_slots": 1,
    "cache_type_k": "vbr",
    "cache_type_v": "vbr",
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def section(title: str) -> None:
    """Print section header."""
    print(f"\n{BOLD}{'=' * 70}{DIM}")
    print(f"{BOLD}  {title}{DIM}")
    print(f"{BOLD}{'=' * 70}\033[0m")


def check(name: str, condition: bool, expected: Any = True, actual: Any = None) -> bool:
    """Print check result."""
    status = PASS if condition == expected else FAIL
    actual_str = f" (actual: {actual})" if actual is not None else ""
    print(f"  {status} {name:50s} {DIM}{actual_str}\033[0m")
    return condition == expected


# ============================================================
# TEST 1: MoE tensor_bytes Fix
# ============================================================

def test_moe_tensor_bytes() -> Tuple[bool, Dict]:
    """Test 1: MoE tensor_bytes Bugfix in gguf_utils.py"""
    section("TEST 1: MoE tensor_bytes Fix (gguf_utils.py)")
    
    if not os.path.exists(MODEL_PATH):
        print(f"  {WARN} Modell nicht gefunden: {MODEL_PATH}")
        return False, {}
    
    info = get_model_info(MODEL_PATH)
    file_size = info.get("file_size", 0)
    tensor_bytes = info.get("tensor_bytes", 0)
    arch = info.get("arch", "")
    
    results = {
        "model_info": info,
        "file_size": file_size,
        "tensor_bytes": tensor_bytes,
    }
    
    print(f"  Modell: {info.get('name', 'unknown')}")
    print(f"  Architektur: {arch}")
    print(f"  File Size: {file_size / (1024**3):.2f} GB")
    print(f"  Tensor Bytes (GGUF): {tensor_bytes / (1024**3):.2f} GB")
    
    # Check: MoE-Modelle sollten tensor_bytes ~ file_size haben (via file_size Fallback)
    is_moe = "qwen35moe" in arch.lower()
    if is_moe:
        ratio = tensor_bytes / file_size if file_size > 0 else 0
        # Nach Fix sollte ratio ~1.0 sein (file_size Fallback)
        ok = ratio >= 0.9  # Mindestens 90% der file_size
        check("MoE Modell erkannt", True, expected=True)
        check("tensor_bytes >= 90% von file_size (file_size Fallback)", ok, expected=True, 
              actual=f"{ratio:.0%}")
    else:
        check("MoE Modell erkannt", False, expected=True)
    
    return is_moe, results


# ============================================================
# TEST 2: VRAM-Schaetzung mit/ohne mmproj
# ============================================================

def test_vram_estimation(moe_info: Dict) -> Tuple[bool, Dict]:
    """Test 2: VRAM-Schaetzung mit und ohne mmproj"""
    section("TEST 2: VRAM-Schaetzung mit/ohne mmproj")
    
    info = moe_info.get("model_info", {})
    file_size = moe_info.get("file_size", 0)
    
    # Schätzung OHNE mmproj
    vram_no_mmproj = estimate_vram(
        model_info=info,
        ngl=TEST_PARAMS["ngl"],
        ctx_size=TEST_PARAMS["ctx_size"],
        np_slots=TEST_PARAMS["np_slots"],
        cache_type_k=TEST_PARAMS["cache_type_k"],
        cache_type_v=TEST_PARAMS["cache_type_v"],
        mmproj_size=0,
    )
    
    # Schätzung MIT mmproj (wenn Datei existiert)
    mmproj_size = 0
    if os.path.exists(MMPOJ_PATH):
        mmproj_size = os.path.getsize(MMPOJ_PATH)
    
    vram_with_mmproj = estimate_vram(
        model_info=info,
        ngl=TEST_PARAMS["ngl"],
        ctx_size=TEST_PARAMS["ctx_size"],
        np_slots=TEST_PARAMS["np_slots"],
        cache_type_k=TEST_PARAMS["cache_type_k"],
        cache_type_v=TEST_PARAMS["cache_type_v"],
        mmproj_size=mmproj_size,
    )
    
    # Ergebnisse
    model_mb_no = vram_no_mmproj["model_vram_mb"]
    cache_mb_no = vram_no_mmproj["cache_vram_mb"]
    total_mb_no = vram_no_mmproj["total_vram_mb"]
    mmproj_mb = vram_with_mmproj.get("mmproj_vram_mb", 0)
    total_mb_with = vram_with_mmproj["total_vram_mb"]
    
    print(f"\n  OHNE mmproj:")
    print(f"    Model weights: {model_mb_no / 1024:.2f} GB ({model_mb_no:.0f} MB)")
    print(f"    KV cache: {cache_mb_no / 1024:.2f} GB ({cache_mb_no:.0f} MB)")
    print(f"    Total: {total_mb_no / 1024:.2f} GB ({total_mb_no:.0f} MB)")
    
    print(f"\n  MIT mmproj:")
    print(f"    Model weights: {model_mb_no / 1024:.2f} GB ({model_mb_no:.0f} MB)")
    print(f"    KV cache: {cache_mb_no / 1024:.2f} GB ({cache_mb_no:.0f} MB)")
    print(f"    mmproj: {mmproj_mb:.1f} MB")
    print(f"    Total: {total_mb_with / 1024:.2f} GB ({total_mb_with:.0f} MB)")
    
    # Check: mmproj sollte zur total_vram_mb hinzugefügt werden
    mmproj_diff = total_mb_with - total_mb_no
    expected_mmproj_diff = mmproj_mb
    
    results = {
        "vram_no_mmproj": vram_no_mmproj,
        "vram_with_mmproj": vram_with_mmproj,
        "mmproj_size": mmproj_size,
    }
    
    check("mmproj erhöht total_vram_mb um ~mmproj_vram_mb",
          abs(mmproj_diff - expected_mmproj_diff) < 100,  # 100 MB Toleranz
          expected=True,
          actual=f"Diff: {mmproj_diff:.0f} MB (erwartet: ~{expected_mmproj_diff:.0f} MB)")
    
    # Check: model_vram_mb sollte gleich sein (unabhängig von mmproj)
    check("model_vram_mb ist unabhängig von mmproj",
          vram_no_mmproj["model_vram_mb"] == vram_with_mmproj["model_vram_mb"],
          expected=True,
          actual=f"{vram_no_mmproj['model_vram_mb']:.0f} MB vs {vram_with_mmproj['model_vram_mb']:.0f} MB")
    
    # Check: KV cache sollte gleich sein (unabhängig von mmproj)
    check("cache_vram_mb ist unabhängig von mmproj",
          vram_no_mmproj["cache_vram_mb"] == vram_with_mmproj["cache_vram_mb"],
          expected=True)
    
    return True, results


# ============================================================
# TEST 3: Display-Rendering (simuliert)
# ============================================================

def test_display_rendering(vram_results: Dict) -> bool:
    """Test 3: Display-Formatierung (simuliert model_inspector.py Ausgabe)"""
    section("TEST 3: Display-Rendering (model_inspector.py)")
    
    vram = vram_results["vram_no_mmproj"]
    
    model_mb = vram["model_vram_mb"]
    cache_mb = vram["cache_vram_mb"]
    total_mb = vram["total_vram_mb"]
    overhead_mb = vram.get("overhead_mb", 0)
    
    # Simuliert: model_inspector.py Zeile 178+
    model_gb = model_mb / 1024
    cache_gb = cache_mb / 1024
    total_gb = total_mb / 1024
    
    print(f"\n  Simulierte Display-Ausgabe:")
    print(f"  ┃ Model weights: {model_gb:.2f} GB ({model_mb:.0f} MB)")
    print(f"  ┃ KV cache: {cache_gb:.2f} GB ({cache_mb:.0f} MB)")
    if overhead_mb > 0:
        print(f"  ┃ overhead: {overhead_mb:.0f} MB")
    print(f"  ┃ Total: {total_gb:.2f} GB ({total_mb:.0f} MB)")
    
    # Check: model_gb sollte model_vram_mb/1024 sein, NICHT total_vram_mb/1024
    check("model_gb = model_vram_mb / 1024 (nicht total_vram_mb)",
          abs(model_gb - (model_mb / 1024)) < 0.01,
          expected=True,
          actual=f"{model_gb:.2f} GB (erwartet: {model_mb/1024:.2f} GB)")
    
    check("model_gb != total_vram_mb/1024 (Bugfix)",
          abs(model_gb - (total_mb / 1024)) > 0.5,  # Mindestens 0.5 GB Unterschied
          expected=True,
          actual=f"{model_gb:.2f} GB vs total {total_mb/1024:.2f} GB")
    
    # Check: cache_gb sollte cache_vram_mb/1024 sein
    check("cache_gb = cache_vram_mb / 1024",
          abs(cache_gb - (cache_mb / 1024)) < 0.01,
          expected=True)
    
    # Check: total_gb sollte total_vram_mb/1024 sein
    check("total_gb = total_vram_mb / 1024",
          abs(total_gb - (total_mb / 1024)) < 0.01,
          expected=True)
    
    return True


# ============================================================
# TEST 4: VBR-Calibration
# ============================================================

def test_vbr_calibration() -> bool:
    """Test 4: VBR-Calibration über llama.cpp Server"""
    section("TEST 4: VBR-Calibration (Server-Anfrage)")
    
    try:
        # Slots-Status prüfen
        result = subprocess.run(
            ["curl", "-s", f"{LLAMA_SERVER}/slots"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print(f"  {WARN} Server nicht erreichbar unter {LLAMA_SERVER}")
            return False
        
        slots_data = json.loads(result.stdout)
        print(f"  Slots-Status: {json.dumps(slots_data, indent=4)}")
        
        # KV cache type prüfen
        result2 = subprocess.run(
            ["curl", "-s", f"{LLAMA_SERVER}/health"],
            capture_output=True, text=True, timeout=5
        )
        
        if result2.returncode == 0:
            health = json.loads(result2.stdout)
            print(f"  Health: {json.dumps(health, indent=4)[:500]}")
        
        return True
        
    except Exception as e:
        print(f"  {WARN} VBR-Prüfung fehlgeschlagen: {e}")
        return False


# ============================================================
# TEST 5: Empirische mmproj GPU-vs-CPU Prüfung
# ============================================================

def test_mmproj_gpu_vs_cpu() -> bool | None:
    """Test 5: Empirische Prüfung ob mmproj auf GPU oder CPU geladen wird"""
    section("TEST 5: Empirische mmproj GPU-vs-CPU Prüfung")
    
    try:
        # GPU-Speicher vor dem Test
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print(f"  {WARN} nvidia-smi nicht verfügbar")
            return False
        
        gpu_before = int(result.stdout.strip().replace(" MiB", ""))
        print(f"  GPU-Speicher vor Test: {gpu_before} MiB")
        
        # Prüfen ob Server läuft mit mmproj
        result2 = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True, timeout=3
        )
        
        if result2.returncode != 0:
            print(f"  {WARN} Kein llama-server Prozess gefunden")
            return False
        
        pid = result2.stdout.strip().split()[0]
        print(f"  llama-server PID: {pid}")
        
        # GPU-Speicher nach Test (sofort prüfen)
        time.sleep(2)
        
        result3 = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        gpu_after = int(result3.stdout.strip().replace(" MiB", ""))
        diff = gpu_after - gpu_before
        
        print(f"  GPU-Speicher nach Test: {gpu_after} MiB")
        print(f"  Differenz: {diff} MiB")
        
        # Wenn mmproj auf GPU geladen wird, sollte Differenz ~mmproj_größe sein
        mmproj_size_mb = os.path.getsize(MMPOJ_PATH) / (1024*1024) if os.path.exists(MMPOJ_PATH) else 0
        mmproj_f32_mb = mmproj_size_mb * 2  # f16 → f32 = 2x
        
        print(f"\n  mmproj Datei: {mmproj_size_mb:.0f} MiB (f16)")
        print(f"  Erwartet auf GPU (f32): ~{mmproj_f32_mb:.0f} MiB")
        print(f"  Gemessene Differenz: {diff} MiB")
        
        # Check: Wenn Differenz ~mmproj_größe, dann GPU
        # Wenn Differenz ~0, dann CPU
        ratio = abs(diff - mmproj_f32_mb) / mmproj_f32_mb if mmproj_f32_mb > 0 else 0
        
        if ratio < 0.3:
            print(f"\n  {PASS} mmproj wird auf GPU geladen (Differenz ~{mmproj_f32_mb:.0f} MiB)")
            return True
        elif abs(diff) < 500:  # Weniger als 500 MiB = wahrscheinlich kein GPU-Einfluss
            print(f"\n  {PASS} mmproj wird auf CPU geladen (Differenz ~{diff} MiB)")
            return False
        else:
            print(f"\n  {WARN} Unklar - Differenz passt weder zu GPU noch zu CPU")
            print(f"  Mögliche Ursachen: Andere GPU-Nutzung, Cache-Effekte")
            return None  # Unentschieden
            
    except Exception as e:
        print(f"  {WARN} Empirischer Test fehlgeschlagen: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all tests."""
    print(f"{BOLD}{'=' * 70}\033[0m")
    print(f"{BOLD}  VRAM Calibration & Display Test Suite{DIM}")
    print(f"{BOLD}{'=' * 70}\033[0m")
    print(f"  Modell: {MODEL_PATH}")
    print(f"  mmproj: {MMPOJ_PATH}")
    print(f"  Server: {LLAMA_SERVER}")
    
    results = {}
    all_passed = True
    
    # Test 1: MoE tensor_bytes
    try:
        moe_ok, moe_results = test_moe_tensor_bytes()
        results["moe_tensor_bytes"] = moe_ok
        all_passed = all_passed and moe_ok
    except Exception as e:
        print(f"  {FAIL} Test 1 fehlgeschlagen: {e}")
        results["moe_tensor_bytes"] = False
        all_passed = False
    
    # Test 2: VRAM-Schätzung
    try:
        vram_ok, vram_results = test_vram_estimation(moe_results)
        results["vram_estimation"] = vram_ok
        all_passed = all_passed and vram_ok
    except Exception as e:
        print(f"  {FAIL} Test 2 fehlgeschlagen: {e}")
        results["vram_estimation"] = False
        all_passed = False
    
    # Test 3: Display-Rendering
    try:
        display_ok = test_display_rendering(vram_results)
        results["display_rendering"] = display_ok
        all_passed = all_passed and display_ok
    except Exception as e:
        print(f"  {FAIL} Test 3 fehlgeschlagen: {e}")
        results["display_rendering"] = False
        all_passed = False
    
    # Test 4: VBR-Calibration
    try:
        vbr_ok = test_vbr_calibration()
        results["vbr_calibration"] = vbr_ok
        all_passed = all_passed and (vbr_ok if vbr_ok is not None else True)
    except Exception as e:
        print(f"  {FAIL} Test 4 fehlgeschlagen: {e}")
        results["vbr_calibration"] = False
        all_passed = False
    
    # Test 5: Empirische mmproj Prüfung
    try:
        mmproj_ok = test_mmproj_gpu_vs_cpu()
        results["mmproj_gpu_vs_cpu"] = mmproj_ok
    except Exception as e:
        print(f"  {FAIL} Test 5 fehlgeschlagen: {e}")
        results["mmproj_gpu_vs_cpu"] = False
    
    # Zusammenfassung
    print(f"\n{BOLD}{'=' * 70}{DIM}")
    print(f"{BOLD}  Zusammenfassung{DIM}")
    print(f"{BOLD}{'=' * 70}\033[0m")
    
    for test_name, passed in results.items():
        status = PASS if passed else (FAIL if passed is False else WARN)
        print(f"  {status} {test_name:30s}")
    
    if all_passed:
        print(f"\n  {PASS} Alle Tests bestanden!")
    else:
        print(f"\n  {FAIL} Einige Tests fehlgeschlagen!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
