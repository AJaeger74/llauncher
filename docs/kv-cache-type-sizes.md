# KV Cache Type Sizes — Exakte Werte aus ggml-common.h

## Quelle

`ggml/src/ggml-common.h` aus buun-llama-cpp (https://github.com/spiritbuun/buun-llama-cpp).
Alle Werte aus `static_assert(sizeof(block_...) == ...)`.

---

## Standard-Typen

| Typ | Block Size | Block Bytes | Bytes/Value | Bits/Value |
|---|---|---|---|---|
| f32 | 1 | 4 | 4.0 | 32 |
| f16 | 1 | 2 | 2.0 | 16 |
| bf16 | 1 | 2 | 2.0 | 16 |
| q8_0 | 32 | 34 | 1.0625 | 8.5 |
| q4_0 | 32 | 20 | 0.625 | 5.0 |
| q4_1 | 32 | 22 | 0.6875 | 5.5 |
| iq4_nl | 32 | 20 | 0.625 | 5.0 |
| q5_0 | 32 | 22 | 0.6875 | 5.5 |
| q5_1 | 32 | 24 | 0.75 | 6.0 |

---

## TurboQuant-Typen (korrigiert)

**WICHTIG:** Die alten Werte in diesem Dokument (Block Size 128, 10/14/68 Bytes)
waren **falsch**. Korrektur aus `static_assert` in ggml-common.h:

| Typ | QK | Block Bytes | Bytes/Value | Bits/Value | Block Struktur |
|---|---|---|---|---|---|
| turbo8_0 | 128 | 130 | 1.015625 | 8.125 | norm(2) + qs[128] = 130 |
| turbo4_0 | 128 | 66 | 0.515625 | 4.125 | norm(2) + qs[64] = 66 |
| turbo3_tcq | 128 | 52 | 0.40625 | 3.25 | norm(2) + qs[49] + pad(1) = 52 |
| turbo2_tcq | 128 | 36 | 0.28125 | 2.25 | norm(2) + qs[33] + pad(1) = 36 |
| turbo2_0 | 32 | 10 | 0.3125 | 2.5 | norm(2) + qs[8] = 10 |
| turbo3_0 | 32 | 14 | 0.4375 | 3.5 | norm(2) + qs[8] + signs[4] = 14 |
| tq3_0 | 32 | 14 | 0.4375 | 3.5 | qs[8] + qr[4] + gamma(2) = 14 |
| turbo1_tcq | 128 | 20 | 0.15625 | 1.25 | norm(2) + qs[17] + pad(1) = 20 |
| turbo1_cq | 128 | 18 | 0.140625 | 1.125 | scale(2) + qs[16] = 18 |
| turbo1_nsn | 128 | 20 | 0.15625 | 1.25 | s1(2)+s2(2)+signs[16] = 20 |
| turbo1 | 128 | 18 | 0.140625 | 1.125 | scale(2)+signs[16] = 18 |

### Block-Strukturen im Detail

**turbo8_0 (8.125 bpw):**
- `norm` (fp16): 2 Byte — L2 norm
- `qs[QK_TURBO8]` (int8): 128 Byte — 8-bit codebook indices
- Total: 130 bytes per 128 values = 1.015625 B/V

**turbo4_0 (4.125 bpw):**
- `norm` (fp16): 2 Byte — L2 norm
- `qs[QK_TURBO4 / 2]` (uint8): 64 Byte — 4-bit indices (low nibble first)
- Total: 66 bytes per 128 values = 0.515625 B/V

**turbo3_tcq (3.25 bpw, TCQ):**
- `norm` (fp16): 2 Byte — corrected group L2 norm
- `qs[49]` (uint8): 49 Byte — 390-bit trellis bitstream (2 padding bits)
- `pad`: 1 Byte — alignment padding
- Total: 52 bytes per 128 values = 0.40625 B/V

**turbo2_tcq (2.25 bpw, TCQ):**
- `norm` (fp16): 2 Byte — corrected group L2 norm
- `qs[33]` (uint8): 33 Byte — 262-bit trellis bitstream (2 padding bits)
- `pad`: 1 Byte — alignment padding
- Total: 36 bytes per 128 values = 0.28125 B/V

**turbo2_0 (2.5 bpw, original TQ):**
- `norm` (fp16): 2 Byte — corrected vector L2 norm
- `qs[QK_TURBO2 / 4]` (uint8): 8 Byte — 2-bit indices (4 per byte)
- Total: 10 bytes per 32 values = 0.3125 B/V

**turbo3_0 (3.5 bpw, original TQ):**
- `norm` (fp16): 2 Byte — vector L2 norm
- `qs[QK_TURBO3 / 4]` (uint8): 8 Byte — lower 2-bit indices
- `signs[QK_TURBO3 / 8]` (uint8): 4 Byte — upper 1-bit of 3-bit index
- Total: 14 bytes per 32 values = 0.4375 B/V

**tq3_0 (3.5 bpw, TQ3_0):**
- `qs[8]` (uint8): 8 Byte
- `qr[4]` (uint8): 4 Byte
- `gamma` (fp16): 2 Byte
- Total: 14 bytes per 32 values = 0.4375 B/V

### turbo1 variants (RESERVED — codec entfernt 2026-07-05)

Die structs existieren noch in ggml-common.h für enum-stability, aber
**keine Encode/Decode-Pfade** mehr:
- turbo1: 18 bytes (scale + 128 sign bits)
- turbo1_nsn: 20 bytes (double-normalize + 128 sign bits)
- turbo1_cq: 18 bytes (16 codebook indices per 128 values)
- turbo1_tcq: 20 bytes (1-bit Trellis-Coded Quantization)

---

## Praktische Werte für VRAM-Berechnung

| GGML-Typ | Bytes/Value | VBR-Tier |
|---|---|---|
| GGML_TYPE_F16 | 2.0 | entry tier |
| GGML_TYPE_TURBO8_0 | 1.015625 | VBR_TIER_T8 |
| GGML_TYPE_TURBO4_0 | 0.515625 | VBR_TIER_T4 |
| GGML_TYPE_TURBO3_TCQ | 0.40625 | VBR_TIER_T3_TCQ |
| GGML_TYPE_TURBO2_TCQ | 0.28125 | VBR_TIER_T2_TCQ |
| GGML_TYPE_TURBO1_TCQ | 0.15625 | VBR_TIER_T1_TCQ (RESERVED) |

---

## Bemerkungen

- turbo8_0 und turbo4_0 haben Block Size 128 (QK_TURBO8/4 = 128)
- turbo2_tcq, turbo3_tcq haben Block Size 128 (TCQ über 128-Element Rotation Groups)
- turbo2_0, turbo3_0, tq3_0 haben Block Size 32 (original TurboQuant)
- turbo1 variants: Block Size 128, aber codec entfernt 2026-07-05
- turbo3_tcq = TRELLIS_CODED, nicht gleich turbo3_0! Andere Block-Struktur
- turbo2_tcq = TRELLIS_CODED, nicht gleich turbo2_0! Andere Block-Struktur
- Bei head_dim != multiple of QK: Zero-Padding auf nächste Block-Grenze → +0-25% Overhead

## Referenzen

- ggml-common.h: https://github.com/spiritbuun/buun-llama-cpp/blob/master/ggml/src/ggml-common.h
- TurboQuant Paper (ICLR 2026): https://arxiv.org/abs/2504.19874
- VBR-Doku: ./vbr-architecture.md
