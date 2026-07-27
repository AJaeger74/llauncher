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


def _gguf_scalar_size(entry_type: int) -> int:
    """Return the byte size of a scalar GGUF v3 value type.

    Type enum (from GGUF spec):
        0:UINT8(1B)  1:INT8(1B)  2:UINT16(2B)  3:INT16(2B)
        4:UINT32(4B)  5:INT32(4B)  6:FLOAT32(4B)  7:BOOL(1B)
        10:UINT64(8B)  11:INT64(8B)  12:FLOAT64(8B)
    """
    if entry_type in (0, 1, 7):
        return 1
    if entry_type in (2, 3):
        return 2
    if entry_type in (4, 5, 6):
        return 4
    if entry_type in (10, 11, 12):
        return 8
    return 0


def _skip_gguf_kv_entry(data: bytes, offset: int) -> int:
    """Skip a single GGUF metadata KV entry and return the next offset.

    GGUF v3 KV entry layout (verified from hex dump):
        key_len   uint32
        padding   4 bytes (to align key to 8-byte boundary from entry start)
        key       char[key_len]
        type      uint32  (right after key, no alignment)
        value     depends on type  (right after type, no alignment)

    Entries are packed immediately — no alignment between entries.
    GGUF type enum: 8=STRING, 9=ARRAY, others are scalars (see _gguf_scalar_size).
    """
    if offset + 8 > len(data):
        return len(data)

    key_len = struct.unpack('<I', data[offset:offset + 4])[0]
    if key_len > 10000:
        return len(data)  # malformed — bail

    # Key starts at offset + 8 (key_len uint32 + 4 bytes padding)
    key_start = offset + 8
    if key_start + key_len > len(data):
        return len(data)

    # Type is right after the key (no alignment)
    type_off = key_start + key_len
    if type_off + 4 > len(data):
        return len(data)
    entry_type = struct.unpack('<I', data[type_off:type_off + 4])[0]

    # Value starts right after type (no alignment)
    val_off = type_off + 4
    if val_off > len(data):
        return len(data)

    if entry_type == 8:  # STRING
        if val_off + 8 > len(data):
            return len(data)
        str_len = struct.unpack('<Q', data[val_off:val_off + 8])[0]
        if str_len > 1000000:
            return len(data)  # sanity check
        return val_off + 8 + str_len
    elif entry_type == 9:  # ARRAY
        if val_off + 12 > len(data):
            return len(data)
        # arr_type is uint32, NOT uint8
        arr_type = struct.unpack('<I', data[val_off:val_off + 4])[0]
        n_items = struct.unpack('<Q', data[val_off + 4:val_off + 12])[0]
        if n_items > 1000000:
            return len(data)  # sanity check
        item_off = val_off + 12
        for _ in range(n_items):
            if arr_type == 8:  # array of strings
                if item_off + 8 > len(data):
                    return len(data)
                il = struct.unpack('<Q', data[item_off:item_off + 8])[0]
                if il > 1000000:
                    return len(data)
                item_off += 8 + il
            else:
                sz = _gguf_scalar_size(arr_type)
                if sz == 0:
                    return len(data)
                item_off += sz
        return item_off
    else:
        # Scalar value
        sz = _gguf_scalar_size(entry_type)
        if sz == 0:
            return len(data)  # unknown scalar type
        return val_off + sz


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
    # Metadata can be large (tokenizer/vocab) — large MoE models may have
    # 50+ MB of tokenizer metadata. Tensor table is small (~5 KB).
    # Read up to 100 MB to be safe; for smaller files read everything.
    read_size = min(file_size, 100 * 1024 * 1024)

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

    # Skip metadata KV entries to reach the tensor table.
    # The header's metadata_kv_count can over-count (some files include
    # tokenizer array items in the count). Detect the tensor table by
    # checking that the next entry's key is valid ASCII — tensor table
    # entries start with a uint64 name_len, not a printable key.
    offset = 24  # after header
    for _ in range(header["metadata_kv_count"]):
        # Peek at the next entry's key_len to detect tensor table
        if offset + 12 > len(data):
            return None
        next_key_len = struct.unpack('<I', data[offset:offset + 4])[0]
        # If key_len is zero, we may be at a valid empty-key entry or
        # at the tensor table boundary. Check the key for validity.
        next_key_start = offset + 8
        if next_key_len > 0 and next_key_start + next_key_len > len(data):
            return None
        # Valid metadata keys are printable ASCII — NUL bytes or absurdly
        # large key_len mean we're in tensor table data being misread as a key
        if next_key_len == 0 or next_key_len > 1000:
            # Try to read past this entry; if it fails we're at the tensor table
            test_off = _skip_gguf_kv_entry(data, offset)
            if test_off >= len(data) or test_off <= offset:
                break
            # Entry was valid (empty key is rare but possible), continue
            offset = test_off
            continue
        next_key = data[next_key_start:next_key_start + next_key_len]
        if b'\x00' in next_key:
            break

        offset = _skip_gguf_kv_entry(data, offset)
        if offset >= len(data):
            return None  # truncated data — not enough read

    # After metadata ends, scan forward for the tensor table.
    # The table starts with the embedding token tensor — look for
    # known first-tensor names to avoid false positives in metadata
    # padding or boundary regions.
    first_tensor_names = [
        b"token_embd.weight",   # Standard naming (llama, qwen, etc.)
        b"tok_embd.weight",     # Alternative naming
        b"tok_embeddings.weight",  # GPT-NeoX style
    ]
    tensor_table_offset: int | None = None
    for search_name in first_tensor_names:
        idx = data.find(search_name, offset)
        if idx > 0 and idx >= 8:
            name_len = struct.unpack('<Q', data[idx - 8:idx])[0]
            if name_len == len(search_name):
                tensor_table_offset = idx - 8
                break

    # Fallback: if we didn't find a known first tensor, scan for any
    # valid tensor entry (name_len + printable name with a dot).
    if tensor_table_offset is None:
        scan = offset
        while scan < len(data) - 8:
            name_len = struct.unpack('<Q', data[scan:scan + 8])[0]
            if 1 <= name_len <= 200 and scan + 8 + name_len <= len(data):
                candidate_name = data[scan + 8:scan + 8 + name_len]
                if all(32 <= b < 127 for b in candidate_name) and b'.' in candidate_name:
                    tensor_table_offset = scan
                    break
            scan += 1

    if tensor_table_offset is None:
        return None  # couldn't find tensor table

    offset = tensor_table_offset

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
        read_gguf_integer(path, "attention.key_head_count", min_val=1, max_val=500) or
        read_gguf_integer(path, f"{arch}.attention.head_count_kv", min_val=1, max_val=500) or
        read_gguf_integer(path, "attention.head_count_kv", min_val=1, max_val=500)
    )
    # For GQA: key_head_count < head_count; for MHA: key_head_count == head_count
    # Fallback: assume MHA (key_head_count == head_count), default to 1 if unknown
    if key_head_count is None:
        key_head_count = head_count if head_count is not None else 1

    # KV head dimension — for most models this equals embedding_length // head_count,
    # but MTP models (Qwen3.6, etc.) store a separate attention.key_length that
    # differs from the Q-head dimension. Prefer the explicit GGUF key_length.
    kv_head_dim = (
        read_gguf_integer(path, f"{arch}.attention.key_length", min_val=1, max_val=4096) or
        read_gguf_integer(path, "attention.key_length", min_val=1, max_val=4096)
    )
    if kv_head_dim is None and embedding_length and head_count:
        kv_head_dim = embedding_length // head_count
    
    # Feed-forward dimension (for expert/MoE models)
    ff_length = (
        read_gguf_integer(path, f"{arch}.feed_forward_length", min_val=1, max_val=10000000) or
        read_gguf_integer(path, "feed_forward_length", min_val=1, max_val=10000000)
    )
    
    # Parse tensor bytes from GGUF header (accurate, excludes metadata)
    tensor_bytes = read_gguf_tensor_bytes(path)
    file_size = stat.st_size
    
    # Sanity check: the parser only handles standard llama/gpt-neox layouts.
    # For non-standard architectures (MTP, MoE, SSM) it may return garbage
    # — e.g. Qwen3.6-27B MTP returns only ~13% of file_size.  If the parser
    # result is less than 70% of file_size, treat it as a failure.
    if tensor_bytes is not None and file_size > 0:
        if tensor_bytes < file_size * 0.7:
            tensor_bytes = None  # treat as parse failure
    
    # Fallback: when the parser fails, use a fraction of file_size.
    # GGUF metadata (tokenizer/vocab) is not loaded on GPU.  For these models
    # the parser returns ~10-13% of file_size (structurally incomplete), so we
    # fall back to a calibrated factor.  82% works for both MoE and Dense
    # Qwen3.6 models — validated against nvidia-smi.
    if tensor_bytes is None:
        tensor_bytes = int(file_size * 0.82)
    
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
        "kv_head_dim": kv_head_dim,
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
# Standard: exact bytes/value from GGML block layout (block_size=32)
# Turbo:   bytes/value from block_turbo{2,3,4,8}_0 structs (block_size varies)
# TCQ:     bytes/value from block_turbo{1,2,3}_tcq structs (block_size=128)
# Source:  ggml/src/ggml-common.h (buun-llama-cpp master), ggml/src/ggml.c (type_traits)
#
# VBR (Variable Bit Rate) is a dynamic mode that degrades layer-by-layer.
# When VBR is active, estimate_vram() uses --vbr-vram budget if set,
# otherwise falls back to the floor tier for estimation.
KV_CACHE_TYPE_SIZES = {
    # Full precision
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    # Quantized (block_size=32)
    "q8_0": 1.0625,   # 34 / 32 = 8.5 bits/value
    "q4_0": 0.625,    # 20 / 32 = 5 bits/value
    "q4_1": 0.6875,   # 22 / 32 = 5.5 bits/value
    "iq4_nl": 0.625,  # 20 / 32 = 5 bits/value
    "q5_0": 0.6875,   # 22 / 32 = 5.5 bits/value
    "q5_1": 0.75,     # 24 / 32 = 6 bits/value
    # TurboQuant — PolarQuant (from ggml-common.h block_turbo*_0)
    # turbo2: block_size=32, 10 bytes total
    "turbo2": 0.3125,       # 10 / 32 = 2.5 bits/value
    "turbo2_0": 0.3125,
    # turbo3: block_size=128, 50 bytes total (norm 2 + qs 32 + signs 16)
    "turbo3": 0.390625,     # 50 / 128 = 3.125 bits/value
    "turbo3_0": 0.390625,
    # turbo4: block_size=128, 66 bytes total (norm 2 + qs 64)
    "turbo4": 0.515625,     # 66 / 128 = 4.125 bits/value
    "turbo4_0": 0.515625,
    # turbo8: block_size=128, 130 bytes total (norm 2 + qs 128)
    "turbo8": 1.015625,     # 130 / 128 = 8.125 bits/value
    "turbo8_0": 1.015625,
    # TurboQuant — TCQ (Trellis-Coded Quantization) (from ggml-common.h block_turbo*_tcq)
    # turbo3_tcq: block_size=128, 52 bytes total (norm 2 + qs 49 + pad 1)
    "turbo3_tcq": 0.40625,  # 52 / 128 = 3.25 bits/value
    # turbo2_tcq: block_size=128, 36 bytes total (norm 2 + qs 33 + pad 1)
    "turbo2_tcq": 0.28125,  # 36 / 128 = 2.25 bits/value
    # turbo1_tcq: block_size=128, 20 bytes total (norm 2 + qs 17 + pad 1)
    "turbo1_tcq": 0.15625,  # 20 / 128 = 1.25 bits/value
    # TQ3_0 (unixsysdev/llama-turboquant, block_size=32)
    "tq3_0": 0.4375,  # 14 / 32 = 3.5 bits/value (qs 8 + qr 4 + gamma 2)
    # VBR (Variable Bit Rate) — dynamic, starts at turbo8 tier and degrades.
    # Empirical value: 0.38 B/v (~3.0 bpv), derived from live GPU measurement
    # at 117K tokens where 52/65 layers were degraded (avg 0.36 B/v on degraded).
    # At max context (262K), all layers are degraded — 0.38 is the safe average.
    # The old 0.72 was measured at 120K with only partial degradation.
    "vbr": 0.38,  # ~3.0 bits/value (empirical from live GPU at full degradation)
}


def estimate_vram(
    model_info: Dict[str, Any],
    ngl: int = 0,
    ctx_size: int = 4096,
    np_slots: int = 1,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    mmproj_size: int = 0,
    vbr_vram_mb: int | None = None,
    vbr_calibrated_v_bytes: float | None = None,
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
        cache_type_k: KV cache type for keys (e.g. "f16", "q4_0", "vbr").
        cache_type_v: KV cache type for values.
        mmproj_size: Size of mmproj file in bytes (for vision models).
        vbr_vram_mb: VBR VRAM budget in MB (--vbr-vram). When set and cache_type
                     is "vbr" on either side, this overrides the computed cache size.
        vbr_calibrated_v_bytes: When VBR is active, use the live-calibrated V-cache
                     bytes/value from /slots API instead of the static type map.
                     Set by VBR-Calibrate button (e.g. 0.72 for ~5.8 bpv).

    Returns dict with:
        model_vram_mb: Estimated model weight VRAM in MB
        cache_vram_mb: Estimated KV cache VRAM in MB
        mmproj_vram_mb: Estimated mmproj VRAM in MB
        overhead_mb: Fixed CUDA/driver overhead (1024 MB)
        total_vram_mb: Total estimated VRAM in MB
    """
    # Prefer tensor_bytes (from GGUF header) over file_size.
    # tensor_bytes excludes metadata/tokenizer/vocab — only GPU-relevant weights.
    # llama.cpp keeps tokenizer/metadata on CPU — only tensor weights go to VRAM.
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

    # For MoE models the alignment overhead is much smaller (more compact
    # tensor packing, fewer small tensors).
    if model_vram_bytes > 0:
        model_vram_bytes *= 1.05

    # ── KV Cache VRAM ───────────────────────────────────────────────
    # Per llama.cpp llama-kv-cache:
    #   K tensor: ne[0]=head_dim, ne[1]=n_head_k, ne[2]=n_ctx
    #   V tensor: ne[0]=head_dim, ne[1]=1 (transposed), ne[2]=n_ctx
    # Per-layer: (key_heads + 1) × head_dim × row_bytes × ctx
    # Total: n_layers × kv_per_layer × n_slots
    use_vbr = cache_type_k.lower() == "vbr" or cache_type_v.lower() == "vbr"
    
    if use_vbr and vbr_vram_mb is not None and vbr_vram_mb > 0:
        # VBR with explicit budget — use the budget directly as cache size.
        # RoPE is computed on-the-fly in llama.cpp (not stored in KV cache).
        total_cache_bytes = int(vbr_vram_mb * 1024 * 1024)
    elif embedding_length and head_count:
        # Prefer explicit KV head dimension (e.g. from attention.key_length)
        # over the naive embedding_length // head_count derivation.
        # MTP/Qwen3.6 models store a separate key_length that differs from
        # the Q-head dimension, so embedding/head_count gives wrong head_dim.
        head_dim = model_info.get("kv_head_dim") or (embedding_length // head_count)

        cache_bytes_k = KV_CACHE_TYPE_SIZES.get(cache_type_k, 2)  # default f16
        cache_bytes_v = KV_CACHE_TYPE_SIZES.get(cache_type_v, 2)

        # Use calibrated V bytes when VBR is active on the V side and the
        # calibrated value shows real degradation (below the static type map).
        # At low token counts VBR stays at F16 (2.0 B/v) — using that for
        # estimation would massively overestimate VRAM.
        if cache_type_v.lower() == "vbr" and vbr_calibrated_v_bytes is not None:
            if vbr_calibrated_v_bytes < cache_bytes_v:
                cache_bytes_v = vbr_calibrated_v_bytes
            # else: calibrated shows F16 at current context → use type map
            #       (VBR will degrade at target context size)

        # K and V each store: key_heads × head_dim × ctx elements.
        # V is transposed (ne[1]=1) for access but still holds the same
        # number of elements as K — transposition is a layout choice,
        # not a reduction in storage.
        kv_k_per_slot = ctx_size * key_head_count * head_dim * cache_bytes_k
        kv_v_per_slot = ctx_size * key_head_count * head_dim * cache_bytes_v
        kv_per_slot = kv_k_per_slot + kv_v_per_slot

        # Multiply by layers (each layer has its own KV cache) and slots.
        # RoPE embeddings are computed on-the-fly in llama.cpp, NOT stored.
        n_layers = model_info.get("block_count") or 1
        total_cache_bytes = kv_per_slot * np_slots * n_layers
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
