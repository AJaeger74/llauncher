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


# GGML type → (block_size, type_size) for byte calculation
# Covers all known llama.cpp quantization types.
# block_size = number of elements per block
# type_size  = number of bytes per block
GGML_TYPE_SIZES = {
    0:  (1, 4),   # GGML_TYPE_F32
    1:  (1, 2),   # GGML_TYPE_F16
    2:  (32, 20), # GGML_TYPE_Q4_0
    3:  (32, 22), # GGML_TYPE_Q4_1
    6:  (32, 22), # GGML_TYPE_Q5_0
    7:  (32, 24), # GGML_TYPE_Q5_1
    8:  (32, 34), # GGML_TYPE_Q8_0
    9:  (32, 36), # GGML_TYPE_Q8_1
    10: (16, 13), # GGML_TYPE_Q2_K
    11: (32, 22), # GGML_TYPE_Q3_K
    12: (32, 24), # GGML_TYPE_Q4_K
    13: (32, 28), # GGML_TYPE_Q5_K
    14: (32, 36), # GGML_TYPE_Q6_K
    15: (32, 36), # GGML_TYPE_Q8_K
    16: (256, 56),# GGML_TYPE_IQ2_XXS
    17: (256, 76),# GGML_TYPE_IQ2_XS
    18: (256, 56),# GGML_TYPE_IQ3_XXS
    19: (256, 36),# GGML_TYPE_IQ1_S
    20: (32, 20), # GGML_TYPE_IQ4_NL
    21: (256, 68),# GGML_TYPE_IQ3_S
    22: (256, 76),# GGML_TYPE_IQ2_S
    23: (256, 68),# GGML_TYPE_IQ4_XS
    24: (256, 76),# GGML_TYPE_IQ2_M
    25: (1, 1),   # GGML_TYPE_I8
    26: (1, 2),   # GGML_TYPE_I16
    27: (1, 4),   # GGML_TYPE_I32
    28: (1, 8),   # GGML_TYPE_I64
    29: (1, 8),   # GGML_TYPE_F64
    30: (256, 36),# GGML_TYPE_IQ1_M
    31: (1, 2),   # GGML_TYPE_BF16
    32: (256, 96),# GGML_TYPE_Q4_0_4_4
    33: (256, 96),# GGML_TYPE_Q4_0_4_8
    34: (256, 96),# GGML_TYPE_Q4_0_8_8
    35: (256, 26),# GGML_TYPE_TQ1_0
    36: (256, 42),# GGML_TYPE_TQ2_0
}


def _read_gguf_header(data: bytes) -> Optional[Dict[str, int]]:
    """Parse GGUF header fields: version, metadata_kv_count, tensor_count.

    Returns dict or None if not a valid GGUF file.
    GGUF v3 layout (all little-endian):
        [0:4]  magic  b"GGUF"
        [4:8]  version  uint32
        [8:16] metadata_kv_count  uint64
        [16:24] tensor_count  uint64
    """
    if len(data) < 24 or data[0:4] != b"GGUF":
        return None

    version = struct.unpack('<I', data[4:8])[0]
    metadata_kv_count = struct.unpack('<Q', data[8:16])[0]
    tensor_count = struct.unpack('<Q', data[16:24])[0]

    # Sanity checks
    if version not in (2, 3):
        return None
    if metadata_kv_count > 10000 or tensor_count > 100000:
        return None

    return {
        "version": version,
        "metadata_kv_count": metadata_kv_count,
        "tensor_count": tensor_count,
    }


def _skip_gguf_kv_entry(data: bytes, offset: int) -> int:
    """Skip a single GGUF metadata KV entry and return the next offset.

    GGUF v3 KV entry layout:
        key_len  uint32
        type     uint32
        value    depends on type (aligned to 8 bytes)
    Returns the offset after this entry.
    """
    if offset + 8 > len(data):
        return len(data)  # malformed — bail

    key_len = struct.unpack('<I', data[offset:offset + 4])[0]
    # type is at offset+4, but we only need to skip the value
    type_byte = data[offset + 4]  # single byte is enough for type dispatch

    val_offset = offset + 8
    if val_offset > len(data):
        return len(data)

    # Type→size mapping (GGUF v3 types)
    # 0: bool, 1: float32, 2: uint32, 3: uint64, 4: int32, 5: string,
    # 6: int64, 7: array (special)
    if type_byte == 5:  # string
        if val_offset + 8 > len(data):
            return len(data)
        str_len = struct.unpack('<Q', data[val_offset:val_offset + 8])[0]
        next_off = val_offset + 8 + str_len
    elif type_byte in (0, 1, 2, 4):  # bool, float32, uint32, int32
        next_off = val_offset + 4
    elif type_byte in (3, 6):  # uint64, int64
        next_off = val_offset + 8
    elif type_byte == 7:  # array
        # array: type(4) + n_items(8) + items (each parsed recursively)
        if val_offset + 12 > len(data):
            return len(data)
        arr_type = data[val_offset]
        n_items = struct.unpack('<Q', data[val_offset + 4:val_offset + 12])[0]
        item_off = val_offset + 12
        # Skip each item in the array (they are primitive types or strings)
        for _ in range(n_items):
            if arr_type == 5:  # array of strings
                if item_off + 8 > len(data):
                    return len(data)
                item_len = struct.unpack('<Q', data[item_off:item_off + 8])[0]
                item_off += 8 + item_len
            elif arr_type in (0, 1, 2, 4):
                item_off += 4
            elif arr_type in (3, 6):
                item_off += 8
            else:
                break  # unknown nested type — bail
        next_off = item_off
    else:
        return len(data)  # unknown type

    # Align to 8 bytes
    next_off = ((next_off + 7) // 8) * 8
    return next_off


def read_gguf_tensor_bytes(path: str) -> Optional[int]:
    """Parse the GGUF tensor table and return the total bytes of all tensors.

    Reads enough of the file to cover the header, metadata, and tensor table.
    Uses GGML block sizes to correctly calculate quantized tensor sizes.

    Returns total bytes or None if parsing fails.
    """
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return None

    # Estimate how much to read: header + metadata + tensor table
    # Each tensor entry ≈ 32-64 bytes + name length
    # Read up to 500KB to be safe for models with many tensors
    read_size = min(file_size, 500 * 1024)
    # If file is small, read it all
    if read_size >= file_size:
        read_size = file_size

    try:
        with open(path, "rb") as f:
            data = f.read(read_size)
    except OSError:
        return None

    header = _read_gguf_header(data)
    if header is None:
        return None

    tensor_count = header["tensor_count"]
    if tensor_count == 0:
        return 0

    # Skip metadata KV entries to reach the tensor table
    offset = 24  # after header
    for _ in range(header["metadata_kv_count"]):
        offset = _skip_gguf_kv_entry(data, offset)
        if offset >= len(data):
            return None  # truncated data — not enough read

    # Now iterate through tensor entries
    total_tensor_bytes = 0
    for _ in range(tensor_count):
        if offset + 12 > len(data):
            break  # truncated

        # name_len (uint64)
        name_len = struct.unpack('<Q', data[offset:offset + 8])[0]
        offset += 8

        if offset + name_len > len(data):
            break  # truncated
        # Skip tensor name
        offset += name_len

        if offset + 12 > len(data):
            break

        # n_dims (uint32)
        n_dims = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        if offset + (n_dims * 8) + 12 > len(data):
            break

        # dims (n_dims × uint64) — calculate total elements
        total_elements = 1
        for _ in range(n_dims):
            dim = struct.unpack('<Q', data[offset:offset + 8])[0]
            total_elements *= dim
            offset += 8

        # dtype (uint32) — GGML type
        dtype = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        # tensor offset (uint64) — skip it
        offset += 8

        # Calculate tensor size from dtype + elements
        block_info = GGML_TYPE_SIZES.get(dtype)
        if block_info:
            block_size, type_size = block_info
            # Number of blocks (rounded up)
            num_blocks = (total_elements + block_size - 1) // block_size
            tensor_bytes = num_blocks * type_size
            total_tensor_bytes += tensor_bytes

        # Align to 8 bytes
        offset = ((offset + 7) // 8) * 8

    return total_tensor_bytes


def read_gguf_tensor_count(path: str) -> int:
    """Read tensor count from GGUF header."""
    try:
        with open(path, "rb") as f:
            data = f.read(24)

        if len(data) < 24 or data[0:4] != b"GGUF":
            return 0

        return struct.unpack('<Q', data[16:24])[0]
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
    
    # Parse tensor bytes from GGUF header (accurate, excludes metadata)
    tensor_bytes = read_gguf_tensor_bytes(path)
    
    result = {
        "filename": Path(path).name,
        "arch": arch,
        "name": name,
        "context_length": ctx_len,
        "file_size": stat.st_size,
        "version": version,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
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
    mmproj_size: int = 0,
) -> Dict[str, Any]:
    """Estimate VRAM usage for a model given the current parameters.

    Three components:
        VRAM_model   = tensor_bytes × GPU fraction × 1.05 (alignment padding)
        VRAM_context = KV cache + RoPE per slot
        VRAM_overhead = 1024 MB fixed buffer (CUDA runtime + driver)

    Args:
        model_info: Output of get_model_info() — needs tensor_bytes, file_size,
                    block_count, embedding_length, head_count, key_head_count.
        ngl: Number of GPU layers (0 = CPU-only, -1 = all layers).
        ctx_size: Context size in tokens.
        np_slots: Number of parallel slots (default 1, -1 treated as 1).
        cache_type_k: KV cache type for keys (e.g. "f16", "q4_0").
        cache_type_v: KV cache type for values.
        mmproj_size: Size of mmproj file in bytes (for vision models).

    Returns dict with:
        model_vram_mb: Estimated model weight VRAM in MB
        cache_vram_mb: Estimated KV cache VRAM in MB
        mmproj_vram_mb: Estimated mmproj VRAM in MB
        overhead_mb: Fixed CUDA/driver overhead (1024 MB)
        total_vram_mb: Total estimated VRAM in MB
    """
    # Prefer tensor_bytes (from GGUF header) over file_size.
    # tensor_bytes excludes metadata/tokenizer/vocab — only GPU-relevant weights.
    total_tensor_bytes = model_info.get("tensor_bytes") or model_info.get("file_size", 0)
    block_count = model_info.get("block_count")
    embedding_length = model_info.get("embedding_length")
    head_count = model_info.get("head_count")
    key_head_count = model_info.get("key_head_count") or head_count or 1

    if np_slots < 0 or np_slots == 0:
        np_slots = 1

    # ── Model weights VRAM ──────────────────────────────────────────
    # GGUF layout: tok_embd + N blocks + output projection
    # Each block ≈ total_tensor_bytes / (block_count + 2)
    # Embedding + output always go to GPU when ngl > 0.

    if ngl == 0:
        # CPU-only mode — no weights on GPU
        model_vram_bytes = 0
    elif ngl < 0:
        # -ngl all — full model on GPU
        model_vram_bytes = total_tensor_bytes
    elif block_count and block_count > 0:
        bytes_per_unit = total_tensor_bytes / (block_count + 2)
        gpu_units = min(ngl, block_count) + 2  # ngl blocks + embedding + output
        model_vram_bytes = bytes_per_unit * gpu_units
    else:
        # Fallback: no block info, assume full model
        model_vram_bytes = total_tensor_bytes if total_tensor_bytes > 0 else 0

    # 5% safety margin for CUDA alignment and page padding in VRAM
    if model_vram_bytes > 0:
        model_vram_bytes = model_vram_bytes * 1.05

    # ── KV Cache VRAM ───────────────────────────────────────────────
    # bytes_per_token = key_heads × head_dim × (cache_k_bytes + cache_v_bytes)
    # head_dim = embedding_length / head_count
    if embedding_length and head_count:
        head_dim = embedding_length // head_count

        cache_bytes_k = KV_CACHE_TYPE_SIZES.get(cache_type_k, 2)  # default f16
        cache_bytes_v = KV_CACHE_TYPE_SIZES.get(cache_type_v, 2)

        kv_cache_per_slot = ctx_size * key_head_count * head_dim * (cache_bytes_k + cache_bytes_v)

        # RoPE embeddings (small: ctx × embedding_length × 1 byte)
        rope_per_slot = ctx_size * embedding_length

        total_cache_bytes = (kv_cache_per_slot + rope_per_slot) * np_slots
    else:
        # Fallback: rough f16 estimate
        total_cache_bytes = (
            ctx_size * (embedding_length or 4096) * 4 * np_slots
        )

    # ── mmproj VRAM ─────────────────────────────────────────────────
    # llama.cpp loads mmproj as float32 (quantized file → f32 expansion)
    mmproj_mb = (mmproj_size * 2 / (1024 * 1024)) if mmproj_size > 0 else 0

    # ── Fixed overhead ──────────────────────────────────────────────
    overhead_mb = 1024  # CUDA runtime + driver + desktop compositor

    model_vram_mb = model_vram_bytes / (1024 * 1024)
    cache_vram_mb = total_cache_bytes / (1024 * 1024)

    total_vram_mb = model_vram_mb + cache_vram_mb + mmproj_mb + overhead_mb

    return {
        "model_vram_mb": round(model_vram_mb, 1),
        "cache_vram_mb": round(cache_vram_mb, 1),
        "mmproj_vram_mb": round(mmproj_mb, 1),
        "overhead_mb": overhead_mb,
        "total_vram_mb": round(total_vram_mb, 1),
    }


def suggest_ngl(
    model_info: Dict[str, Any],
    ctx_size: int = 4096,
    np_slots: int = 1,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    mmproj_size: int = 0,
    free_vram_mb: int = 0,
) -> int:
    """Calculate the maximum safe -ngl value given available VRAM.

    Uses binary-free math:
        available = free_vram_mb - cache_mb - overhead_mb - mmproj_mb
        bytes_per_layer = total_tensor_bytes / (block_count + 2)
        ngl = available / bytes_per_layer × 0.95 (safety margin)

    Returns:
        -1 if the full model fits
        n  if partial offload is optimal (1..block_count)
        0  if even 1 layer + context exceeds free VRAM
    """
    block_count = model_info.get("block_count")
    if not block_count or block_count <= 0:
        return -1  # can't determine, assume all

    total_tensor_bytes = model_info.get("tensor_bytes") or model_info.get("file_size", 0)
    if not total_tensor_bytes:
        return -1

    # Estimate cache + overhead + mmproj (these are ngl-independent)
    cache_vram = estimate_vram(
        model_info=model_info,
        ngl=0,  # model weights stay on CPU
        ctx_size=ctx_size,
        np_slots=np_slots,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        mmproj_size=mmproj_size,
    )
    non_model_mb = (
        cache_vram["cache_vram_mb"]
        + cache_vram["overhead_mb"]
        + cache_vram["mmproj_vram_mb"]
    )

    # Bytes per layer (each block ≈ tensor_bytes / (block_count + 2))
    bytes_per_unit = total_tensor_bytes / (block_count + 2)
    bytes_per_layer_mb = bytes_per_unit / (1024 * 1024)

    # Available for model weights (with 5% alignment padding)
    available_for_model = free_vram_mb - non_model_mb
    if available_for_model <= 0:
        return 0  # not even cache + overhead fits

    # Full model check: embedding + output + all blocks
    full_model_mb = (bytes_per_unit * (block_count + 2) * 1.05) / (1024 * 1024)
    if full_model_mb <= available_for_model:
        return -1  # full model fits — use -ngl all

    # Calculate max layers that fit
    # embedding + output always go to GPU, so subtract them first
    embd_output_mb = (bytes_per_unit * 2 * 1.05) / (1024 * 1024)
    remaining_mb = available_for_model - embd_output_mb

    if remaining_mb <= 0:
        return 0  # not enough for even embedding + output + 1 layer

    max_layers = int(remaining_mb / bytes_per_layer_mb)
    max_layers = max(0, min(max_layers, block_count))

    return max_layers
