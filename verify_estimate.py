#!/usr/bin/env python3
from gguf_utils import get_model_info, estimate_vram, KV_CACHE_TYPE_SIZES

model = '/opt/fast/ai/models/llama.cpp/unsloth_Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q6_K.gguf'
info = get_model_info(model)

print('KV_CACHE_TYPE_SIZES:')
for k, v in sorted(KV_CACHE_TYPE_SIZES.items()):
    print(f'  {k}: {v} B/v')

print()
vram = estimate_vram(model_info=info, ngl=-1, ctx_size=262144, np_slots=1,
                     cache_type_k='vbr', cache_type_v='vbr')
print(f'VBR+VBR estimate: {vram["total_vram_mb"]/1024:.1f} GB')
print(f'  model: {vram["model_vram_mb"]/1024:.1f} GB')
print(f'  cache: {vram["cache_vram_mb"]/1024:.1f} GB')
print(f'  overhead: {vram["overhead_mb"]/1024:.1f} GB')
print()

# Actual GPU usage: ~27 GB, model is 3 GB, so cache ~24 GB
# kv_bpv=16.0 at 41 tokens (F16, no degradation yet)
# At full context, VBR would degrade. Let's find the right VBR value.

actual_cache = 24e9  # ~24 GB actual cache
ctx = 262144
k_heads = 4
head_dim = 256
layers = 65

# cache = 2 * ctx * k_heads * head_dim * vbr_bytes * layers
vbr_actual = actual_cache / (2 * ctx * k_heads * head_dim * layers)
print(f'Actual VBR per side needed: {vbr_actual:.3f} B/v')
print(f'  (from 24 GB actual cache at 262K context)')
print()

# Also check: what if we use overhead=2?
vram2 = estimate_vram(model_info=info, ngl=-1, ctx_size=262144, np_slots=1,
                      cache_type_k='vbr', cache_type_v='vbr')
print(f'With current VBR=0.72, cache={vram["cache_vram_mb"]/1024:.1f} GB')
# Adjust to match actual
print(f'  -> need VBR ~{vbr_actual:.2f} instead of 0.72')
