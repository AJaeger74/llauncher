#!/usr/bin/env python3
"""
VBR (Variable Bit Rate) KV-Cache VRAM Calculator

Standalone script or importable module for:
- VBR degradation simulation
- Runtime degradation analysis from /slots API
- Effective tier detection and VRAM savings

Usage as script:
    python vbr_calc.py

Usage as module:
    from vbr_calc import simulate_vbr, effective_degradation, fetch_runtime_info
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------
# VBR Ladder
# ------------------------------------------------------------

@dataclass
class Tier:
    name: str
    bpv: float

VBR_LADDER = [
    Tier("f16",        16.0),
    Tier("turbo8",      8.125),
    Tier("turbo4",      4.125),
    Tier("turbo3_tcq",  3.25),
    Tier("turbo2_tcq",  2.25),
    Tier("turbo1_tcq",  1.25),
]


def tier_index(name: str) -> int:
    for i, t in enumerate(VBR_LADDER):
        if t.name == name:
            return i
    raise ValueError(f"Unknown tier: {name}")


# ------------------------------------------------------------
# De-Quant Factor Estimation
# ------------------------------------------------------------

def _dequant_factor(info: dict) -> float:
    """Estimate dequantization multiplier from GGUF metadata or model name.

    Quantized models store weights at lower precision on disk but expand
    to FP16/FP32 in VRAM during inference. This factor bridges that gap.

    Args:
        info: Output of gguf_utils.get_model_info()

    Returns:
        Multiplication factor (1.35–1.55 for typical quantizations)
    """
    # 1. Explicit quantization tag in GGUF header
    qlevel = info.get("general.quantization_level")
    if qlevel:
        low = {"q4_0", "q5_0", "q8_0"}
        mid = {"q5_1", "q6_k", "q6_z"}
        high = {"q7_k", "q8_1"}
        if qlevel in low:
            return 1.35
        if qlevel in mid:
            return 1.45
        if qlevel in high:
            return 1.55
        return 1.5

    # 2. Name-based mapping for models without header tag
    name = info.get("general.name", "").lower()
    if "4bit" in name or "q4" in name:
        return 1.35
    if "5bit" in name or "q5" in name:
        return 1.45
    # Default: good for most APEX/quality variants
    return 1.5


# ------------------------------------------------------------
# Effective Weight Bytes (raw × dequant factor)
# ------------------------------------------------------------

def _effective_weights_bytes(info: dict) -> int:
    """Calculate VRAM-relevant weight size (raw bytes × dequant factor).

    For nemotron_h_moe models, llama.cpp keeps weights in quantized form on
    VRAM and only dequantizes per-tile during compute. So raw tensor_bytes
    ≈ actual VRAM usage — no dequant multiplier needed.

    For all other models, returns raw tensor_bytes unchanged.

    Args:
        info: Output of gguf_utils.get_model_info()

    Returns:
        Estimated VRAM usage of model weights in bytes
    """
    raw = info.get("tensor_bytes", 0)
    if raw == 0:
        return 0

    # Nemotron H-MoE: weights stay quantized on GPU
    if info.get("arch") == "nemotron_h_moe":
        return raw
    else:
        # All other models: unchanged behavior
        return raw


# ------------------------------------------------------------
# Nemotron-specific VBR Simulation Wrapper
# ------------------------------------------------------------

def simulate_vbr_nemotron(
    total_vram_bytes: int,
    info: dict,
    ctx: int,
    layers: int,
    heads: int,
    head_dim: int,
    floor: str = "turbo1_tcq",
) -> tuple:
    """VBR simulation wrapper for nemotron_h_moe models.

    Uses effective weight bytes (dequantization-aware) and actual
    context size from the slot instead of hardcoded maximums.

    Args:
        total_vram_bytes: Total GPU VRAM in bytes
        info: Output of gguf_utils.get_model_info()
        ctx: Actual context size from active slot
        layers: Number of model layers (block_count)
        heads: KV head count (key_head_count)
        head_dim: Key/value head dimension (kv_head_dim)
        floor: Minimum VBR tier (default: turbo1_tcq)

    Returns:
        (kv_actual_bytes, kv_budget_bytes, tier_list_per_segment)
    """
    effective_weights_bytes = _effective_weights_bytes(info)

    return simulate_vbr(
        total_vram_bytes=total_vram_bytes,
        weights_bytes=effective_weights_bytes,
        ctx=ctx,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        floor=floor,
    )


# ------------------------------------------------------------
# KV Cache Berechnung
# ------------------------------------------------------------

def kv_bytes_static(ctx: int, layers: int, heads: int, head_dim: int, tier: Tier) -> int:
    """Calculate KV cache bytes for a single tier across all layers."""
    values = ctx * layers * 2 * heads * head_dim
    return int(values * tier.bpv / 8)


def kv_bytes_segments(ctx: int, layers: int, heads: int, head_dim: int, bpv_list: list) -> int:
    """Calculate KV cache bytes from per-segment bpv values."""
    values_per_segment = ctx * heads * head_dim
    bits = sum(values_per_segment * bpv for bpv in bpv_list)
    return int(bits / 8)


# ------------------------------------------------------------
# VBR Simulation
# ------------------------------------------------------------

def simulate_vbr(
    total_vram_bytes: int,
    weights_bytes: int,
    ctx: int,
    layers: int,
    heads: int,
    head_dim: int,
    floor: str = "turbo1_tcq",
) -> tuple:
    """Simulate VBR degradation given VRAM budget and model parameters.

    Returns:
        (kv_actual_bytes, kv_budget_bytes, tier_list_per_segment)
    """
    kv_budget = total_vram_bytes - weights_bytes
    if kv_budget <= 0:
        raise RuntimeError("Modell passt nicht ins VRAM.")

    n_segments = layers * 2
    seg_tier_idx = [0] * n_segments
    floor_idx = tier_index(floor)

    def current_kv():
        bpv = [VBR_LADDER[i].bpv for i in seg_tier_idx]
        return kv_bytes_segments(ctx, layers, heads, head_dim, bpv)

    kv = current_kv()
    if kv <= kv_budget:
        return kv, kv_budget, [VBR_LADDER[i] for i in seg_tier_idx]

    changed = True
    while kv > kv_budget and changed:
        changed = False
        for s in range(n_segments):
            if seg_tier_idx[s] < floor_idx:
                seg_tier_idx[s] += 1
                changed = True
                kv = current_kv()
                if kv <= kv_budget:
                    break

    return kv, kv_budget, [VBR_LADDER[i] for i in seg_tier_idx]


# ------------------------------------------------------------
# Runtime Infos via /slots API
# ------------------------------------------------------------

def fetch_runtime_info(server_url: str = "http://localhost:8080/slots") -> Optional[dict]:
    """Fetch runtime info from llama.cpp /slots API.

    Returns dict with ctx and kv_bpv, or None if server unreachable.
    """
    try:
        req = urllib.request.Request(server_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        slot = data[0]
        return {
            "ctx": slot["n_ctx"],
            "kv_bpv": slot.get("kv_bpv", 16.0),
            "n_tokens": slot.get("index", {}).get("n_tokens", 0),
        }
    except (urllib.error.URLError, IndexError, KeyError):
        return None


# ------------------------------------------------------------
# Effective Degradation Analysis
# ------------------------------------------------------------

def effective_degradation(
    kv_bpv_eff: float,
    layers: int,
    heads: int,
    head_dim: int,
    ctx: int,
) -> dict:
    """Calculate effective KV degradation from server-reported kv_bpv.

    Args:
        kv_bpv_eff: Effective bits-per-value from /slots API
        layers: Number of model layers (block_count)
        heads: KV head count (key_head_count)
        head_dim: Key/value head dimension (kv_head_dim)
        ctx: Context size in tokens

    Returns dict with:
        kv_f16_bytes: Theoretical F16 cache size
        kv_eff_bytes: Effective cache size at current bpv
        saved_bytes: VRAM saved vs F16
        eff_segments: Equivalent number of fully degraded segments
        eff_tier: Name of closest VBR tier
    """
    bpv_f16 = 16.0
    bpv_floor = 1.25  # turbo1_tcq

    total_values = ctx * layers * 2 * heads * head_dim

    kv_f16 = total_values * (bpv_f16 / 8)
    kv_eff = total_values * (kv_bpv_eff / 8)
    saved = kv_f16 - kv_eff

    total_segments = layers * 2
    eff_segments = (bpv_f16 - kv_bpv_eff) / (bpv_f16 - bpv_floor) * total_segments

    tier_sorted = sorted(VBR_LADDER, key=lambda t: abs(t.bpv - kv_bpv_eff))
    eff_tier = tier_sorted[0].name

    return {
        "kv_f16_bytes": kv_f16,
        "kv_eff_bytes": kv_eff,
        "saved_bytes": saved,
        "eff_segments": eff_segments,
        "eff_tier": eff_tier,
    }


# ------------------------------------------------------------
# Model Info Integration
# ------------------------------------------------------------

def get_model_arch(model_info: dict) -> dict:
    """Extract VBR-relevant architecture from get_model_info() output.

    Args:
        model_info: Output of gguf_utils.get_model_info()

    Returns dict with:
        layers, heads, head_dim, n_embd, weights_bytes
    """
    return {
        "layers": model_info.get("block_count", 0),
        "heads": model_info.get("key_head_count", 1),
        "head_dim": model_info.get("kv_head_dim", 256),
        "n_embd": model_info.get("embedding_length", 0),
        "weights_bytes": model_info.get("tensor_bytes", 0),
    }


# ------------------------------------------------------------
# Main (standalone execution)
# ------------------------------------------------------------

if __name__ == "__main__":
    from gguf_utils import get_model_info, read_gpu_vram, format_size

    GGUF_PATH = "/opt/fast/ai/models/llama.cpp/unsloth_Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q6_K.gguf"

    # Model info from project's native GGUF reader
    model_info = get_model_info(GGUF_PATH)
    arch = get_model_arch(model_info)

    # Runtime info from server
    runtime = fetch_runtime_info()

    print("=== Modell ===")
    print(f"  Layers:       {arch['layers']}")
    print(f"  KV Heads:     {arch['heads']}")
    print(f"  Head Dim:     {arch['head_dim']}")
    print(f"  n_embd:       {arch['n_embd']}")
    print(f"  Weights:      {format_size(arch['weights_bytes'])}")

    if runtime is None:
        print("\nKein Server erreichbar unter http://localhost:8080/slots")
        import sys
        sys.exit(1)

    print(f"\n=== Runtime ===")
    print(f"  Context:      {runtime['ctx']}")
    print(f"  kv_bpv:       {runtime['kv_bpv']}")
    print(f"  n_tokens:     {runtime['n_tokens']}")

    # GPU VRAM
    gpu = read_gpu_vram()
    if gpu:
        total_vram_bytes = gpu["total_mb"] * 1024 * 1024
        print(f"\n=== GPU ===")
        print(f"  Total:      {gpu['total_mb']} MiB")
        print(f"  Used:       {gpu['used_mb']} MiB")
        print(f"  Free:       {gpu['free_mb']} MiB")
    else:
        # Fallback: hardcoded 32 GiB
        total_vram_bytes = int(32.0 * (1024 ** 3))
        print(f"\n=== GPU ===")
        print("  (nvidia-smi nicht verfuegbar, verwende 32 GiB)")

    # VBR Simulation
    try:
        kv, kv_budget, tiers = simulate_vbr(
            total_vram_bytes,
            arch["weights_bytes"],
            runtime["ctx"],
            arch["layers"],
            arch["heads"],
            arch["head_dim"],
        )

        print(f"\n=== VBR Simulation ===")
        print(f"  KV Budget:  {format_size(kv_budget)}")
        print(f"  KV Cache:   {format_size(kv)}")
        print(f"  Total:      {format_size(arch['weights_bytes'] + kv)}")

        # Count tiers
        tier_counts = {}
        for t in tiers:
            tier_counts[t.name] = tier_counts.get(t.name, 0) + 1
        tier_summary = ", ".join(f"{k}: {v}" for k, v in sorted(tier_counts.items()))
        print(f"  Tiers:      {tier_summary}")

    except RuntimeError as e:
        print(f"\nSimulation fehlgeschlagen: {e}")

    # Effective degradation
    eff = effective_degradation(
        runtime["kv_bpv"],
        arch["layers"],
        arch["heads"],
        arch["head_dim"],
        runtime["ctx"],
    )

    print(f"\n=== Effektive KV-Degradation ===")
    print(f"  Tier:           {eff['eff_tier']}")
    print(f"  KV @ F16:       {format_size(eff['kv_f16_bytes'])}")
    print(f"  KV effektiv:    {format_size(eff['kv_eff_bytes'])}")
    print(f"  Ersparnis:      {format_size(eff['saved_bytes'])}")
    total_segs = arch["layers"] * 2
    print(f"  Degradierte:    {eff['eff_segments']:.0f} / {total_segs} Segmente")
