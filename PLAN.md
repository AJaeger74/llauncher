# VBR VRAM Estimation — Findings & Improvement Plan

Date: 2026-09-05
Scope: `gguf_utils.py` (`estimate_vram`, `estimate_vram_from_preset`, `get_vbr_degradation_info`),
`vbr_calc.py`, `llauncher.py` (`_calibrate_vbr`, `_display_vram_estimate`), `get_model_info()`.

Verified against code AND a live VBR server:
Qwen3.8-Flash-Next UD-IQ3_XXS (shard 1/3), `-c 262144 --cache-type-k vbr --cache-type-v vbr`,
~60K tokens in cache, GPU 30.4/31.8 GiB used, `/slots` `kv_bpv` observed at 9.109 → 8.781.

## Findings

### 1. The core assumption behind the current design is outdated — `kv_bpv` IS live (highest impact)

Code comments and skill notes treat `/slots` `kv_bpv` as "the entry type
(always 16.0), not current state". That forced the fragile workaround:
back-calculate cache from nvidia-smi usage − weights − overhead, plus
`shed_offer` to count F16 layers.

Live data disproves this: `kv_bpv: 9.109` then `8.781` minutes later — it
tracks degradation exactly as `docs/vbr-architecture.md` describes
(`kv_bpv_accum()` sums actual bytes over all K+V layer tensors).

Primary estimation path can become simple and robust:

    cache_bytes ≈ kv_bpv / 8 × head_dim × key_heads × layers × 2 × cells

No nvidia-smi dependency, no contamination from other GPU processes.
The nvidia-smi / shed_offer machinery gets demoted to a cross-check.
(Side check: `shed_offer / (F16→turbo8 delta)` ≈ 1 F16 layer on the live
server — that math is correct, just unnecessary as the primary path.)

### 2. Capacity vs. used-tokens ambiguity

Cross-check on live numbers (`layers=48, kvh=2, hd=256`):

    kv_bpv × capacity (262144 cells) = 13.2 GB
    kv_bpv × used tokens (60130)     =  3.0 GB

Docs say `cells = kv_size` (capacity), i.e. the pool is accounted at
allocated capacity, not used rows — and 13.2 GB is consistent with actual
GPU usage once weights are added. Current `estimate_vram_from_preset()`
substitutes used-token counts as ctx_size for nemotron → under-counts what
the server actually holds. Action: confirm with one more before/after
measurement on the live server, then switch the formula to
capacity × kv_bpv.

### 3. `vbr_vram_mb` is dead code

`estimate_vram()` has a `--vbr-vram` budget branch (gguf_utils.py ~line 1119)
but nothing passes the parameter — GUI and preset pipeline never read the
flag. Either wire it (parse `--vbr-vram` from custom commands) or delete the
branch.

### 4. VBR logic is nemotron-only

The `/slots` introspection in `estimate_vram_from_preset()` only runs for
`arch == "nemotron_h_moe"`. The server currently running is `qwen4exp` with
vbr/vbr and gets none of that treatment. Introspection should key off
"VBR active" (cache-type-k/v == vbr), not architecture.

### 5. `VBR_BASE_OVERHEAD_MB = 3600` is a hardcoded Nemotron measurement

Applied to any model in lazy mode. Should scale with model dims
(layers × key_heads × head_dim) or disappear entirely once #1 lands
(kv_bpv-at-startup formula covers it).

### 6. Multi-file GGUF breaks weights entirely (adjacent bug, hits VRAM estimates hard)

Live example: `get_model_info()` reports tensor_bytes = 0.008 GB,
file_size = 0.010 GB — it read shard 1 of 3 only. Estimated model VRAM:
9 MB for a ~15+ GB model. The `file_size × 0.82` fallback doesn't trigger
because the sanity check compares shard-1 against shard-1.
Split GGUFs (`-00001-of-0000N`) must have all shard sizes summed before
parsing/fallback in `get_model_info()`.

### 7. Minor issues

- Hardcoded port 8080 in `_calibrate_vbr()` (host is read from UI, port isn't).
- Defaults 65/4/256 in `get_vbr_degradation_info()` silently used if
  `model_info` lacks fields — should fail loudly or return the empty result.
- `_effective_weights_bytes()` in `vbr_calc.py` documents a dequant factor
  but returns raw `tensor_bytes` for every arch (dead branch; also
  duplicated/contradictory docstring in the skill).
- 2 of 5 tests in `test_vram_calibration.py` currently error out
  (`test_vram_estimation`, `test_display_rendering`).

## Status (updated 2026-09-05, second pass — floor re-fit)

DONE: #1 (live kv_bpv as primary signal), #2 (used-tokens, hybrid layer count),
#3 (lazy floor — re-fit below), #4 (multi-file GGUF size summation + --vbr-vram
dead branch removed), #5 (tests for all of the above).

### #3 RESOLVED: lazy floor scales with WEIGHT SIZE, not KV-layer count

The `measure_floor.py` batch ran (Andreas stopped the 27B assistant, ran the
one command, restarted). Three CLEAN zero-request floors landed in
`/tmp/floor_results.json`:

| model                    | weights (MiB) | KV layers | arch             | non-weight footprint (MiB) |
|--------------------------|---------------|-----------|------------------|----------------------------|
| Qwen3-1.7B-Q4_K_M        | 866           | 28        | qwen3 (dense)    | **830**                    |
| Qwen3.6-27B-Q4_K_M       | 13151         | **16**    | qwen35 (hybrid)  | **3007**                   |
| Nemotron-3-Nano-30B-A3B  | 17964         | 52        | nemotron (MoE)   | **4284**                   |

Two surprises vs. the plan's assumptions:
- **Qwen3.6-27B is a hybrid, not dense.** `full_attention_interval=4` →
  `kv_layer_count=16` of 64 blocks (48 linear-attention layers). It was
  supposed to be the "decisive high-KV dense point" but is actually the
  LOW-KV / high-constant-state point.
- **The old model is dead.** A 16-KV-layer model (3007 MiB) holds a LARGER
  non-weight footprint than a 28-KV-layer one (830 MiB). The plan's
  linear-in-KV-layers fit (`3600 × kv_layers/52`) is contradicted by the data.
  Even the anchor is off: re-measuring Nemotron gives a 4284 MiB footprint,
  not the 3600 MB the original provenance-undocumented point claimed.

The one predictor that fits all three is **weight size** (R²=0.99):

    footprint ≈ 605 + 0.197 × weights_mib        (R²=0.990, ≤7% per point)

This makes physical sense: at zero tokens the VBR pool has mapped ~nothing
(lazy), so the residual is compute buffers + constant linear-attention state
(the 27B's 48 non-KV layers) + CUDA runtime — all of which track model size,
not KV-layer count. `_vbr_lazy_base_mb()` now implements exactly this line
(`_VBR_IDLE_FLOOR_MB=605.0`, `_VBR_IDLE_PER_WEIGHT_MB=0.197`). The live
`kv_bpv` path is untouched; this only changes the pre-request / no-server
estimate. TEST 3c now validates against the three real measured floors
(3–7% error each) plus a weight-vs-layer regression guard.

All tests pass except the pre-existing `moe_tensor_bytes` failure (missing
model asset on disk, not code).

**Extrapolation caveat:** the fit's data range is 0.866–17.9 GB of weights.
At 78 GB (Qwen3.8-Flash-Next, the live-verification model) the line yields a
~15.6 GB lazy floor — ~20% of weight size, in line with the ~20–24% ratio the
points show, but unmeasured. It is only the pre-request / no-server estimate;
the live kv_bpv path takes over once the server has tokens.

## Status (first pass — kept for history)

DONE: #1 (live kv_bpv as primary signal), #2 (used-tokens, hybrid layer count),
#3 (lazy base generalized to KV-layer scaling), #4 (multi-file GGUF size
summation + --vbr-vram dead branch removed), #5 (tests for all of the above).

### What changed in `gguf_utils.py`

1. `get_model_info()` now reads `full_attention_interval` and derives
   `kv_layer_count` (token-scaling KV layers). Qwen "Flash-Next" qwen4exp:
   interval=4 → 12 of 48 layers, so the old block_count=48 formula overestimated
   the token-scaling cache 4x. Dense models: kv_layer_count == block_count.

2. `estimate_vram_from_preset()` VBR introspection is now keyed off
   `use_vbr` (any arch) instead of `arch == "nemotron_h_moe"`. It picks an
   active slot, cross-checks n_ctx against preset -c (different-server guard),
   and captures the LIVE `kv_bpv` + used-token count. The non-VBR nemotron
   token override is preserved in an `elif` branch (behavior unchanged).

3. `estimate_vram()` gained `vbr_live_bpv`. New PRIMARY path:
   `cache = 2 * ctx_size * key_heads * head_dim * (kv_bpv/8) * kv_layers * n_slots`.
   Static type map / calibrated bytes / vbr_lazy(3600 MB) / vbr_vram_mb are all
   retained as fallbacks (used at startup, before any /slots data). The static
   path now multiplies by `kv_layer_count` (hybrid-aware) instead of block_count.

### Live verification (qwen4exp Qwen3.8-Flash-Next, -c 262144, vbr/vbr)

Ramp test filled context in cached requests, sampled /slots + nvidia-smi:
- kv_bpv is LIVE: moved 16.0 → 9.76 → 6.1 as context grew (disproves old note).
- Cache is LAZY: F16-region GPU rose 1:1 with used tokens (not n_ctx-reserved).
- F16 slope = 24.17 MiB/1024 tok == predicted 12-layer value (24.0); 48-layer
  would be 96.0 → confirms kv_layer_count=12.
- estimate_vram_from_preset cache term = 2602.8 MB, exactly matching
  2 * used_tokens * kvh * hd * (kv_bpv/8) * kv_layers computed independently.

Note: model_vram_mb previously read only GGUF shard 1/3 (9 MB for a ~76 GB
model). Fixed in the #4a change below — `_split_gguf_sizes()` now sums the whole
shard family, so model_vram_mb reports ~80 GB.

## Plan (in order)

1. [DONE] Make `kv_bpv` from `/slots` the primary live signal for VBR cache
   estimation; demote nvidia-smi/shed_offer to validation-only. Update skill +
   memory note claiming kv_bpv is static.
2. [DONE] Used-tokens (lazy) confirmed; fixed `estimate_vram_from_preset()` and
   added hybrid `kv_layer_count`.
3. [DONE] Generalized lazy-mode base: `_vbr_lazy_base_mb()` anchors to the one
   measured floor (Nemotron, 52 KV layers, 3600 MB) and scales by the model's
   kv_layer_count; 512 MB floor for very small models. Verified: 52→3600,
   12→830, 4→512 MB.
4. [DONE] `--vbr-vram`: confirmed the buun build exposes no `--vbr*` CLI flags
   and no caller passes `vbr_vram_mb`, and the branch was unreachable (VBR
   resolves to live-bpv or lazy first). Deleted the dead branch, param, and
   docstring entry; static `vbr`→0.38 B/v path handles it. Multi-file GGUF:
   `_split_gguf_sizes()` sums the whole shard family for `file_size` and the
   tensor-bearing shards for `tensor_bytes` (quantized weights stay packed on
   VRAM, so no metadata haircut). Verified on the 3-shard Qwen3.8-Flash-Next:
   file_size 0.01→76.33 GB, model VRAM ~9 MB→80 GB; single-file path unchanged.
5. [DONE] Test suite: added TEST 3c (lazy floor scaling) and TEST 3d (split
   GGUF) to `test_vram_calibration.py`. All pass except the pre-existing
   moe_tensor_bytes failure (missing model asset on disk, not code).

## Caveat for review (2026-09-05: RESOLVED — see second-pass Status above)

#3's lazy floor (`_vbr_lazy_base_mb()`) was anchored to a single data point
(Nemotron 3600 MB / 52 KV layers) and scaled LINEARLY by kv_layer_count.
The `measure_floor.py` batch ran and the linear assumption is DISPROVEN: the
decisive "high-KV dense" point (Qwen3.6-27B) turned out to be a 16-KV-LAYER
hybrid (48 linear-attention layers), and its 3007 MiB non-weight footprint
exceeds the 28-KV-layer Qwen3-1.7B's 830 MiB. Weight size is the real
predictor (R²=0.99). `_vbr_lazy_base_mb()` now implements
`605 + 0.197 × weights_mib`. The live `kv_bpv` path was never affected —
this only changes the pre-request / no-server estimate.

## RUN-THIS batch (for Andreas — COMPLETED 2026-09-05)

The two remaining points (Nemotron 52 KV, Qwen3.6-27B 16 KV) were ~16-23 GB
each and did not fit alongside the running 27B. Stopped the assistant, ran
`measure_floor.py`, restarted it. Both landed in `/tmp/floor_results.json`.

    cd /home/hermes/hermes/dev/llauncher
    # 1) STOP the 27B (the assistant). Llama Launcher GUI Stop button, or:
    #    kill <pid>
    # 2) Confirm VRAM is free:
    #    nvidia-smi --query-gpu=memory.free --format=csv,noheader
    # 3) Self-driving measurement (appends to /tmp/floor_results.json):
    #    python3 measure_floor.py <nemotron.gguf> <qwen3.6-27b.gguf> --port 8090
    # 4) RESTART the assistant from the Llama Launcher GUI. Then: "read PLAN.md"

Small models (≤ ~3 GB) can be measured WITHOUT stopping the assistant.

## How the floor is computed (measure_floor.py)

Clean floor = resident process VRAM a VBR server holds BEFORE any request,
minus its weight bytes. VBR allocates lazily, so at zero requests the KV pool
has mapped ~nothing; what remains is compute buffers + any constant
linear-attention state + CUDA runtime. The script launches ONE model (no
--spec-type, no --ctx-checkpoints, no mmproj, no restored context), waits for
/health + 5 s, sends NO completion, samples
`nvidia-smi --query-compute-apps=pid,used_memory` 3× (2 s apart), and reports:

    pool_floor = stable_total_process_VRAM - weights(MiB)

That is a NON-WEIGHT process footprint (compute + constant state + CUDA), not
a pure KV-cache number. `_vbr_lazy_base_mb()` now models exactly this
footprint as a function of weight size. Results append to /tmp/floor_results.json
as {kv_layers, arch, weights_mib, total_mib, pool_mib, ctx, ts}.

Target points (ALL DONE — see second-pass Status):
- 28 KV: Qwen3-1.7B-Q4_K_M — **DONE, pool ≈ 830 MiB** (weights 866 MiB).
- 52 KV: Nemotron-3-Nano-30B-A3B — **DONE, pool ≈ 4284 MiB** (weights 17964).
- 16 KV: Qwen3.6-27B-Q4_K_M — **DONE, pool ≈ 3007 MiB** (weights 13151).
  NOTE: this is a HYBRID (full_attention_interval=4 → 16 KV of 64 layers),
  not the dense 64-KV model originally assumed.

The fit (see second-pass Status): `pool ≈ 605 + 0.197 × weights_mib`,
R²=0.99. The predictor is WEIGHT SIZE, not kv_layer_count. `_vbr_lazy_base_mb()`
and TEST 3c are updated accordingly. Do NOT push — user pushes.

Models that do NOT load standalone (skip as floor points): the
Qwen3.8-Flash-Next MTP sidecars (`borrow_shared_tensor: token_embd.weight`)
and any draft/spec model.
