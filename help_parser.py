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
    Sammelt Werte über mehrere Zeilen (wenn sie umgebrochen sind).
    
    Args:
        lines: Alle --help Output Zeilen
        start_idx: Index der Zeile mit dem Parameter-Name
        
    Returns:
        Liste der erlaubten Cache-Typen
    """
    allowed_values = []
    
    # Ab Zeile nach dem Start suchen
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        
        # Leerzeile oder neuer Abschnitt beendet die Suche
        if not line.strip() or (line.startswith('-') and not line.startswith(' ')):
            break
        
        # "allowed values:" finden und Werte extrahieren
        if 'allowed values:' in line.lower():
            # Nach dem ":" alles nehmen
            values_part = line.split(':', 1)[1].strip()
            
            # Kommata entfernen und aufschlüsseln
            for val in values_part.replace(',', ' ').split():
                val = val.strip().lower()
                if val and not val.startswith('(') and val not in allowed_values:
                    allowed_values.append(val)
        
        # Fortgesetzte Werte auf nachfolgenden Zeilen (wenn umbrochen ohne Label)
        elif line.strip() and not line.startswith('(') and not allowed_values:
            for val in line.replace(',', ' ').split():
                val = val.strip().lower()
                if val and not val.startswith('(') and val not in allowed_values:
                    allowed_values.append(val)
    
   # Wenn keine Werte gefunden wurden (z.B. il_llama.cpp zeigt nur default),
    # verwende bekannte Standard-Werte
    if not allowed_values:
        return list(FALLBACK_CACHE_TYPES)
    
    return allowed_values


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
