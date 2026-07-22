#!/usr/bin/env python3
"""
llauncher – GGUF Utilities
"""

import os
import struct
import subprocess
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


def read_gguf_integer(path: str, key_name: str, min_val: int = 0, max_val: int = 10000000) -> Optional[int]:
    """Read an integer value from GGUF by key name.
    
    Supports uint32, uint64, int32. Validates the parsed value is within [min_val, max_val].
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
        
        # Try uint64 (type 3 in GGUF)
        if key_end + 12 <= len(data):
            type_byte = data[key_end:key_end+1][0]
            if type_byte == 3:
                val = struct.unpack('<Q', data[key_end+4:key_end+12])[0]
                if min_val <= val <= max_val:
                    return val
            else:
                # Try raw uint64 without type check
                val = struct.unpack('<Q', data[key_end+4:key_end+12])[0]
                if min_val <= val <= max_val:
                    return val
        
        # Try uint32 (type 2)
        if key_end + 8 <= len(data):
            val = struct.unpack('<I', data[key_end+4:key_end+8])[0]
            if min_val <= val <= max_val:
                return val
            
            # Try int32 (type 6)
            val = struct.unpack('<i', data[key_end+4:key_end+8])[0]
            if min_val <= val <= max_val:
                return val
        
        return None
    except Exception:
        return None


def read_gguf_float(path: str, key_name: str) -> Optional[float]:
    """Read a float32 value from GGUF by key name."""
    try:
        with open(path, "rb") as f:
            data = f.read(50 * 1024)
        
        if len(data) < 8 or data[0:4] != b"GGUF":
            return None
        
        idx = data.find(key_name.encode('utf-8'))
        if idx == -1:
            return None
        
        key_end = idx + len(key_name)
        
        # Try float32 (type 1)
        if key_end + 8 <= len(data):
            type_byte = data[key_end:key_end+1][0]
            if type_byte == 1:
                return struct.unpack('<f', data[key_end+4:key_end+8])[0]
            else:
                # Try raw float32
                val = struct.unpack('<f', data[key_end+4:key_end+8])[0]
                if 0 < val < 1e10:
                    return val
        
        return None
    except Exception:
        return None


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


def read_gguf_tensor_count(path: str) -> int:
    """Read tensor count from GGUF header."""
    try:
        with open(path, "rb") as f:
            data = f.read(16)
        
        if len(data) < 16 or data[0:4] != b"GGUF":
            return 0
        
        version = struct.unpack('<I', data[4:8])[0]
        
        if version == 3:
            # GGUF v3: tensor_count is uint64 at offset 8
            return struct.unpack('<Q', data[8:16])[0]
        elif version == 2:
            # GGUF v2: tensor_count is uint64 at offset 8
            return struct.unpack('<Q', data[8:16])[0]
        
        return 0
    except Exception:
        return 0


# Architekturen die von llama.cpp unterstützt werden (llama.cpp/src/llama-arch.cpp)
KNOWN_LLAMA_ARCHITECTURES = {
    "llama", "mamba", "gpt-neox", "stablelm", "stablelm2", "phi", "phi-3", "phi3",
    "gpt-2", "bert", "qwen2", "qwen2_moe", "qwen2_vl", "gptj", "starcoder2",
    "command-r", "command-r-plus", "internlm2", "internlm", "minicpm", "minicpm3",
    "chatglm", "dbrx", "deepseek-v2", "deepseek-v3", "xverse", "falcon", "falcon2",
    "smollm", "olmo", "olmo2", "arctic", "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
    "jamba", "jetmoe", "bloom", "mpt", "persimmon", "exaone3", "granite",
    "granite-dbrx", "grok-1", "grok1", "olmoe", "openelm", "owltow", "platypus2",
    "telechat", "textgen", "whisper", "t5", "bart", "bart2",
    "nemotron", "nemotron_h_moe", "nemotron_hpu_moe", "pwm",
    "ernie", "ernievision", "deepseek-vl", "molmo", "pangu",
    "baichuan", "qwen", "qwen1_5", "qwen35", "qwen35moe", "xglm", "refact", "smaug",
    "griffin", "baidu",
}


def check_model_architecture(arch: str) -> str | None:
    """Prüft ob eine GGUF-Architektur von llama.cpp unterstützt wird.

    Args:
        arch: Architektur-String aus GGUF-Metadaten (z.B. 'llama', 'gemma2')

    Returns:
        Fehlermeldung als String wenn Architektur nicht unterstützt, None wenn OK.
    """
    if not arch or arch == "unknown":
        return None

    if arch in KNOWN_LLAMA_ARCHITECTURES:
        return None

    return arch


def get_model_info(path: str) -> Dict[str, Any]:
    """Extract model info from GGUF file."""
    try:
        stat = os.stat(path)
    except Exception:
        return {"filename": Path(path).name}
    
    # Read header info
    with open(path, "rb") as f:
        data = f.read(16)
    
    version = 3
    if len(data) >= 8:
        version = struct.unpack('<I', data[4:8])[0]
    
    # Read values using direct search (most reliable)
    name = read_gguf_string_value(path, "general.name")
    arch = read_gguf_string_value(path, "general.architecture") or "unknown"
    ctx_len = read_gguf_context_length(path)
    tensor_count = read_gguf_tensor_count(path)
    
    # Architecture-specific prefix for metadata keys
    # Try both the arch-specific prefix and generic fallbacks
    block_count = (
        read_gguf_integer(path, f"{arch}.block_count", min_val=1, max_val=500) or
        read_gguf_integer(path, "block_count", min_val=1, max_val=500)
    )
    embedding_length = (
        read_gguf_integer(path, f"{arch}.embedding_length", min_val=1, max_val=3000000) or
        read_gguf_integer(path, "embedding_length", min_val=1, max_val=3000000)
    )
    
    # Attention parameters for VRAM calculation
    head_count = (
        read_gguf_integer(path, f"{arch}.attention.head_count", min_val=1, max_val=500) or
        read_gguf_integer(path, "attention.head_count", min_val=1, max_val=500)
    )
    key_head_count = (
        read_gguf_integer(path, f"{arch}.attention.key_head_count", min_val=1, max_val=500) or
        read_gguf_integer(path, "attention.key_head_count", min_val=1, max_val=500)
    )
    # For GQA: key_head_count < head_count; for MHA: key_head_count == head_count
    # Fallback: assume MHA (key_head_count == head_count), default to 1 if unknown
    if key_head_count is None:
        key_head_count = head_count if head_count is not None else 1
    
    # Feed-forward dimension (for expert/MoE models)
    ff_length = (
        read_gguf_integer(path, f"{arch}.feed_forward_length", min_val=1, max_val=10000000) or
        read_gguf_integer(path, "feed_forward_length", min_val=1, max_val=10000000)
    )
    
    result = {
        "filename": Path(path).name,
        "arch": arch,
        "name": name,
        "context_length": ctx_len,
        "file_size": stat.st_size,
        "version": version,
        "tensor_count": tensor_count,
        "block_count": block_count,
        "embedding_length": embedding_length,
        "head_count": head_count,
        "key_head_count": key_head_count,
        "ff_length": ff_length,
    }
    
    # Remove None values for cleaner output
    return {k: v for k, v in result.items() if v is not None}


def read_gpu_vram() -> Optional[Dict[str, int]]:
    """Query nvidia-smi for GPU VRAM info.
    
    Returns dict with total_mb, used_mb, free_mb or None if GPU not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        
        values = [v.strip() for v in result.stdout.strip().split(",")]
        total_mb = int(values[0])
        used_mb = int(values[1])
        return {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": total_mb - used_mb,
        }
    except Exception:
        return None


# KV cache type sizes in bytes per element
KV_CACHE_TYPE_SIZES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "q8_0": 1,
    "q4_0": 0.5,
    "q4_1": 0.5,
    "iq4_nl": 0.5,
    "q5_0": 0.625,
    "q5_1": 0.625,
}


def estimate_vram(
    model_info: Dict[str, Any],
    ngl: int = 0,
    ctx_size: int = 4096,
    np_slots: int = 1,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
) -> Dict[str, Any]:
    """Estimate VRAM usage for a model given the current parameters.
    
    Args:
        model_info: Output of get_model_info() — needs file_size, block_count,
                    embedding_length, head_count, key_head_count.
        ngl: Number of GPU layers (0 = CPU-only, -1 = all layers).
        ctx_size: Context size in tokens.
        np_slots: Number of parallel slots (default 1, -1 treated as 1).
        cache_type_k: KV cache type for keys (e.g. "f16", "q4_0").
        cache_type_v: KV cache type for values.
    
    Returns dict with:
        model_vram_mb: Estimated model weight VRAM in MB
        cache_vram_mb: Estimated KV cache VRAM in MB
        total_vram_mb: Total estimated VRAM in MB
        fit_status: "fit", "partial", or "nofit"
    """
    file_size = model_info.get("file_size", 0)
    block_count = model_info.get("block_count")
    embedding_length = model_info.get("embedding_length")
    head_count = model_info.get("head_count")
    key_head_count = model_info.get("key_head_count") or head_count
    
    if np_slots < 0 or np_slots == 0:
        np_slots = 1
    
    # Model weights VRAM estimation:
    # The file size on disk is the quantized size. When loaded into VRAM,
    # llama.cpp keeps the quantized format — so file_size is a good lower bound.
    # However, the actual GPU memory includes the quantized data plus workspace.
    # For a rough estimate: file_size * 1.05 (5% overhead for quant format metadata)
    model_vram_bytes = file_size * 1.05 if file_size > 0 else 0
    
    # If ngl == 0, model stays on CPU — no model weights in VRAM
    if ngl <= 0:
        model_vram_bytes = 0
    elif ngl < 0:
        # ngl == -1 means "all layers" — treat as full model
        pass  # model_vram_bytes stays as calculated
    
    # KV Cache estimation:
    # Each token in context needs: n_kv_heads * (head_dim) * 2 (K + V) * dtype_size bytes
    # head_dim = embedding_length / head_count
    if embedding_length and head_count:
        head_dim = embedding_length // head_count
        bytes_per_token_per_head = head_dim
        
        cache_bytes_k = KV_CACHE_TYPE_SIZES.get(cache_type_k, 2)  # default f16 = 2
        cache_bytes_v = KV_CACHE_TYPE_SIZES.get(cache_type_v, 2)
        
        # KV cache per slot: ctx * key_heads * head_dim * (bytes_k + bytes_v)
        kv_cache_per_slot = ctx_size * key_head_count * bytes_per_token_per_head * (cache_bytes_k + cache_bytes_v)
        
        # Also account for RoPE embeddings (usually small, ~ctx * embedding_length * 1 byte)
        rope_per_slot = ctx_size * embedding_length * 1
        
        total_cache_bytes = (kv_cache_per_slot + rope_per_slot) * np_slots
    else:
        # Fallback: estimate ~2 bytes per token per dimension (f16)
        # Very rough: file_size / (block_count + 1) gives ~bytes per layer
        # Cache ≈ ctx * embedding_length * 2 (K+V) * 2 (f16) * slots
        if embedding_length:
            total_cache_bytes = ctx_size * embedding_length * 4 * np_slots
        else:
            total_cache_bytes = 0
    
    model_vram_mb = model_vram_bytes / (1024 * 1024)
    cache_vram_mb = total_cache_bytes / (1024 * 1024)
    total_vram_mb = model_vram_mb + cache_vram_mb
    
    return {
        "model_vram_mb": round(model_vram_mb, 1),
        "cache_vram_mb": round(cache_vram_mb, 1),
        "total_vram_mb": round(total_vram_mb, 1),
    }
