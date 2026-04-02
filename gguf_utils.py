#!/usr/bin/env python3
"""
llauncher – GGUF Utilities
Statische Helper-Funktionen für GGUF-Dateien und System-Ermittlung.
Unabhängig von PyQt6, kann unit-getestet werden.
"""

import os
import struct
from typing import Optional


def get_cpu_count() -> int:
    """Anzahl der logischen CPUs im System ermitteln."""
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 8


def read_gguf_context_length(path: str) -> Optional[int]:
    """
    context_length aus einer GGUF-Datei durch direkte Binary-Suche lesen.
    
    Liest nur die ersten 1MB (Metadata ist am Anfang), sucht nach dem Key
    "context_length" und extrahiert den Wert basierend auf seinem Type.
    
    Args:
        path: Pfad zur GGUF-Datei
        
    Returns:
        context_length als Integer oder None bei Fehlern
    """
    try:
        with open(path, "rb") as f:
            # Nur erste 1MB lesen (Metadata ist am Anfang)
            data = f.read(1024 * 1024)
        
        if len(data) < 8 or data[0:4] != b"GGUF":
            raise ValueError("Not a GGUF file")
        
        version, = struct.unpack("<I", data[4:8])
        
        # Suche nach "context_length" Key im File
        idx = data.find(b"context_length")
        if idx == -1:
            return None
        
        key_end = idx + len(b"context_length")
        
        # value_type (uint32) direkt hinter dem Key
        val_type, = struct.unpack("<I", data[key_end:key_end+4])
        
        if val_type == 2:  # uint32
            ctx_length, = struct.unpack("<I", data[key_end+4:key_end+8])
        elif val_type in (3, 6):  # int64 oder uint64
            ctx_length, = struct.unpack("<Q" if val_type == 6 else "<q", data[key_end+4:key_end+12])
        elif val_type == 4:  # float32 (manche Qwen-Modelle)
            ctx_length, = struct.unpack("<I", data[key_end+4:key_end+8])
        else:
            return None
        
        return ctx_length
    except Exception as e:
        print(f"Konnte context_length nicht lesen von {path}: {e}")
        return None
