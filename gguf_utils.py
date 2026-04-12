#!/usr/bin/env python3
"""
llauncher – GGUF Utilities
"""

import os
import struct
from pathlib import Path
from typing import Optional, Dict, Any


def get_cpu_count() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 8


def _read_string_content(data: bytes, start: int, max_len: int = 256) -> tuple[str, int]:
    """Read string content until padding or next entry."""
    # Skip leading NUL bytes (padding after uint32 value)
    while start < len(data) and data[start] == 0:
        start += 1
    
    end = start
    while end < min(start + max_len, len(data)):
        if data[end] == 0:
            # Check for padding (4+ consecutive zeros)
            if end + 4 <= len(data) and data[end:end+4] == b'\x00\x00\x00\x00':
                break
        elif end + 8 <= len(data):
            # Check for next entry's key_len
            potential = struct.unpack('<Q', data[end:end+8])[0]
            if 1 <= potential <= 50:
                candidate = data[start:end]
                if len(candidate) > 0 and sum(1 for b in candidate if b != 0) / len(candidate) > 0.7:
                    break
        end += 1
    
    try:
        return data[start:end].decode('utf-8'), end
    except:
        return data[start:end].decode('latin-1'), end


def read_gguf_context_length(path: str) -> Optional[int]:
    """Find context_length using direct search with architecture-specific handling."""
    try:
        with open(path, "rb") as f:
            data = f.read(50 * 1024)
        
        if len(data) < 8 or data[0:4] != b"GGUF":
            return None
        
        # Try multiple key patterns for different architectures
        patterns = [
            b"context_length",           # General
            b"general.context_length",   # Standard location
            b"nemotron_h_moe.context_length",  # Nemotron models
            b"seq_length",               # Alternative name
        ]
        
        for key in patterns:
            idx = data.find(key)
            if idx == -1:
                continue
            
            key_end = idx + len(key)
            
            # Try uint64 first (most common in GGUF v3)
            if key_end + 12 <= len(data):
                val_u64 = struct.unpack('<Q', data[key_end+4:key_end+12])[0]
                if 1024 <= val_u64 <= 10000000:
                    return val_u64
            
            # Try uint32
            if key_end + 8 <= len(data):
                val_u32 = struct.unpack('<I', data[key_end+4:key_end+8])[0]
                if 1024 <= val_u32 <= 10000000:
                    return val_u32
                
                # Try int32
                val_i32 = struct.unpack('<i', data[key_end+4:key_end+8])[0]
                if 1024 <= val_i32 <= 10000000:
                    return val_i32
        
        return None
    except Exception:
        return None


def read_gguf_string_value(path: str, key_name: str) -> Optional[str]:
    """Read a string metadata value by key name.
    
    Handles both standard GGUF v3 format and non-standard variants
    (e.g., Nemotron models with extra padding).
    """
    try:
        with open(path, "rb") as f:
            data = f.read(50 * 1024)
        
        if len(data) < 8 or data[0:4] != b"GGUF":
            return None
        
        idx = data.find(key_name.encode('utf-8'))
        if idx == -1:
            return None
        
        key_end = idx + len(key_name)
        type_byte = data[key_end:key_end+1][0]  # Single byte type indicator
        
        # Standard GGUF v3 string format (type 5)
        if type_byte == 5:
            str_len = struct.unpack('<Q', data[key_end+4:key_end+12])[0]
            return data[key_end+12:key_end+12+str_len].decode('utf-8')
        
        # Nemotron-style format (type 8): uint32 length + padding + string
        elif type_byte == 8:
            str_len = struct.unpack('<I', data[key_end+4:key_end+8])[0]
            
            # Try multiple offsets to find actual string content
            for offset in range(8, min(20, key_end + 8 + str_len + 1)):
                candidate_start = key_end + offset
                if candidate_start + str_len > len(data):
                    continue
                
                try:
                    candidate = data[candidate_start:candidate_start+str_len]
                    s = candidate.decode('utf-8')
                    # Validate: should start with printable characters, not nulls
                    if s and (s[0].isalnum() or s[0] in '-_'):
                        return s
                except:
                    continue
            
            return None
        
        elif type_byte == 0 or type_byte > 10:
            # Handle corrupted/missing type byte - scan for valid string length
            for offset in range(4, 20):
                if key_end + offset + 4 > len(data):
                    continue
                
                potential_len = struct.unpack('<I', data[key_end+offset:key_end+offset+4])[0]
                if 1 <= potential_len <= 256:
                    for str_offset in range(8, min(24, key_end + offset + 4 + potential_len)):
                        candidate_start = key_end + offset + str_offset
                        if candidate_start + potential_len > len(data):
                            continue
                        
                        try:
                            candidate = data[candidate_start:candidate_start+potential_len]
                            s = candidate.decode('utf-8')
                            if s and (s[0].isalnum() or s[0] in '-_'):
                                return s
                        except:
                            continue
                    break  # First valid length found, don't try other offsets
            
            return None
        
        return None
    except Exception:
        return None


def format_size(bytes_size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_size) < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_model_info(path: str) -> Dict[str, Any]:
    """Extract model info from GGUF file."""
    try:
        stat = os.stat(path)
    except Exception:
        return {"filename": Path(path).name}
    
    # Read values using direct search (most reliable)
    name = read_gguf_string_value(path, "general.name")
    arch = read_gguf_string_value(path, "general.architecture") or "unknown"
    ctx_len = read_gguf_context_length(path)
    
    return {
        "filename": Path(path).name,
        "arch": arch,
        "name": name,
        "context_length": ctx_len,
        "file_size": stat.st_size,
        "version": 3,
        "tensor_count": 0,
    }
