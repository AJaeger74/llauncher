#!/usr/bin/env python3
"""
Parser für llama-server --help Output.
Extrahiert dynamisch die 'allowed values' für cache-type-k und cache-type-v.

Beispiel Output:
  -ctk,  --cache-type-k TYPE              KV cache data type for K
                                          allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1,
                                          turbo2, turbo3, turbo4
                                          (default: f16)

Funktion: parse_cache_type_options(binary_path)
→ Führt llama-server --help aus
→ Sucht nach --cache-type-k und --cache-type-v
→ Extrahiert alle 'allowed values' (auch über mehrere Zeilen)
→ Gibt Dict zurück: { 'k': [...], 'v': [...] }
"""

import subprocess
import re
from pathlib import Path
from typing import Optional

# Standard-Werte die in fast allen llama.cpp Builds unterstützt werden
FALLBACK_CACHE_TYPES = ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl']


def parse_cache_type_options(binary_path: str) -> dict[str, list[str]]:
    """
    Extrahiert die allowed values für --cache-type-k und --cache-type-v
    aus llama-server --help Output.

    Args:
        binary_path: Pfad zum llama-server Binary
        
    Returns:
        Dict mit keys 'k' und 'v', jeweils Liste der erlaubten Cache-Typen
        Beispiel: {'k': ['f32', 'f16', 'bf16', ...], 'v': ['f32', 'f16', ...]}
    """
    
    # Debug-Ausgabe in UI (wenn window übergeben wird)
    debug_path = str(binary_path)
    
    # ik_llama.cpp hat kein "allowed values:" Label im Help-Output
    # → direkt Fallback-Werte verwenden, kein Parsing-Versuch
    detected_ik = binary_path and any(x in str(binary_path).lower() for x in ["ik_llama.cpp"])
    
    if detected_ik:
        return {'k': list(FALLBACK_CACHE_TYPES), 'v': list(FALLBACK_CACHE_TYPES)}
    
    try:
        # --help ausführen (mit Timeout und Fehlerbehandlung)
        result = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,  # Das ist äquivalent zu stdout=PIPE, stderr=PIPE
            text=True,
            timeout=10
        )
        
        # Exit-Code 0 oder 1 sind OK (1 bedeutet oft "Help ausgegeben")
        if result.returncode not in (0, 1):
            return {'k': [], 'v': []}
            
        help_text = result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        return {'k': [], 'v': []}
    except FileNotFoundError as e:
        return {'k': [], 'v': []}
    except PermissionError as e:
        # Fallback: Versuche, --help trotzdem zu lesen (manchmal funktioniert es)
        try:
            result = subprocess.run(
                ["bash", "-c", f"{binary_path} --help 2>&1"],
                capture_output=True,
                text=True,
                timeout=10
            )
            help_text = result.stdout + result.stderr
            
            if result.returncode in (0, 1) and len(help_text.strip()) > 0:
                pass
            else:
                return {'k': [], 'v': []}
        except Exception as fallback_error:
            return {'k': [], 'v': []}
    except Exception as e:
        return {'k': [], 'v': []}
    
 # Parsing-Logik
    lines = help_text.split('\n')
    cache_options = {'k': [], 'v': []}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Prüfe auf --cache-type-k oder --cache-type-v Linie (mit -ctk/-ctv Abkürzung)
        if '--cache-type-k TYPE' in line or '-ctk,  --cache-type-k TYPE' in line:
            cache_options['k'] = _extract_allowed_values(lines, i)
            
        elif '--cache-type-v TYPE' in line or '-ctv,  --cache-type-v TYPE' in line:
            cache_options['v'] = _extract_allowed_values(lines, i)
        
        i += 1
    
    return cache_options


def _extract_allowed_values(lines: list[str], start_idx: int) -> list[str]:
    """
    Extrahiert alle 'allowed values' ab der Start-Zeile.
    Sammelt den Text zwischen 'allowed values:' und '(default:' (oder Ende),
    dann kommaseparierte Werte aufsplittern.

    Args:
        lines: Alle --help Output Zeilen
        start_idx: Index der Zeile mit dem Parameter-Name

    Returns:
        Liste der erlaubten Cache-Typen
    """
    # 1) Den zusammengehörenden Textblock suchen (max 50 Zeilen, nicht leerer Abschnitt)
    block_lines = []
    for j in range(start_idx + 1, min(start_idx + 51, len(lines))):
        stripped = lines[j].strip()
        if not stripped:
            break  # Leerzeile → Ende des Blocks
        block_lines.append(stripped)

    full_text = ' '.join(block_lines)

    # 2) Text nach "allowed values:" filtern (case-insensitive)
    match_av = re.search(r'allowed\s+values\s*:\s*', full_text, re.IGNORECASE)
    if not match_av:
        # Kein "allowed values:" Label im Help-Output (z.B. ik_llama.cpp)
        # → Fallback auf Standard-Werte
        return list(FALLBACK_CACHE_TYPES)
    
    full_text = full_text[match_av.end():]

    # 3) Text bis "(default:" kürzen
    idx_default = full_text.lower().find('(default:')
    if idx_default != -1:
        full_text = full_text[:idx_default]

    # 4) Kommaseparierte Werte extrahieren
    values = []
    for token in full_text.replace(',', ' ').split():
        token = token.strip().lower()
        if token and not token.startswith('(') and token not in values:
            values.append(token)

    if not values:
        return list(FALLBACK_CACHE_TYPES)

    return values


if __name__ == "__main__":
    # Test mit lokalem llama-server (Pfad anpassen!)
    import sys
    if len(sys.argv) > 1:
        binary = sys.argv[1]
    else:
        binary = "./llama-server"
    
    options = parse_cache_type_options(binary)
    
    print(f"\nCache Type Options:")
    print(f"  K-Type: {options['k']}")
    print(f"  V-Type: {options['v']}")
