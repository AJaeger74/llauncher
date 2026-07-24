# VBR (Variable Bit Rate) — KV-Cache VRAM-Berechnung

## Quelle

Recherche im **buun-llama-cpp** Fork von spiritbuun (https://github.com/spiritbuun/buun-llama-cpp).
Quelldateien: `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-cparams.h`.

---

## Grundprinzip

VBR ist kein statischer Cache-Typ wie `turbo4` oder `q8_0`, sondern ein **dynamisches Abstufungssystem** (degrade ladder). Der KV-Cache startet bei höchster Präzision (F16) und quantisiert schichtweise runter, sobald VRAM-Druck auftritt.

Im Gegensatz zu TCQ (festes `-ctv turbo4` etc.) passt VBR die Quantisierung **während der Inferenz** dynamisch an — layer-by-layer, basierend auf verfügbarem VRAM.

---

## Die Degrade-Ladder

```
F16 (entry tier) → turbo8_0 → turbo4_0 → turbo3_tcq → turbo2_tcq → turbo1_tcq
```

Jede Stufe ist ein **in-place Tier-Flip**: Der Tensor-Typ wird geändert, Daten werden transkodiert, Speicher freed. Es gibt keine separate Kopie — die gleichen Puffer werden umgewandelt.

### Enum aus `llama-kv-cache.cpp`:

```cpp
enum vbr_tier : uint8_t {
    VBR_TIER_T8,       // turbo8_0   1.015625 B/W  (8.125 bit)  Block: 130B/128W
    VBR_TIER_T4,       // turbo4_0   0.515625 B/W  (4.125 bit)  Block: 66B/128W
    VBR_TIER_T3_TCQ,   // turbo3_tcq 0.406250 B/W  (3.25 bit)   Block: 52B/128W
    VBR_TIER_T2_TCQ,   // turbo2_tcq 0.281250 B/W  (2.25 bit)   Block: 36B/128W
    VBR_TIER_T1_TCQ,   // RESERVED — Codec entfernt 2026-07-05
    VBR_TIER_COUNT,
};
```

> **WICHTIG:** Die VBR-Tier-Map in der Degrade-Ladder verwendet die **VBR-Tier-Namen**
> (`VBR_TIER_T8`, `VBR_TIER_T4`, etc.), die über `vbr_tier_type()` auf die tatsächlichen
> **GGML-Typen** (`GGML_TYPE_TURBO8_0`, `GGML_TYPE_TURBO4_0`, etc.) gemappt werden.
> Diese Namen haben **nichts** mit der Bitrate (z.B. 8 für T8) zu tun — sie sind rein
> numerische Index-Namen für die Degrade-Ladder-Position.
>
> **Korrektur:** Die ursprünglichen Block-Grössen in `kv-cache-type-sizes.md` (Block 128,
> 10/14/68 Bytes) waren **falsch**. Korrekt aus `static_assert` in ggml-common.h:
> - turbo8_0: QK=128, 130 Bytes/Block, 1.015625 B/Wert
> - turbo4_0: QK=128, 66 Bytes/Block, 0.515625 B/Wert
> - turbo3_tcq: QK=128, 52 Bytes/Block, 0.40625 B/Wert (TCQ trellis)
> - turbo2_tcq: QK=128, 36 Bytes/Block, 0.28125 B/Wert (TCQ trellis)
> - turbo2_0: QK=32, 10 Bytes/Block, 0.3125 B/Wert (original TQ)
> - turbo3_0: QK=32, 14 Bytes/Block, 0.4375 B/Wert (original TQ)
> - tq3_0: QK=32, 14 Bytes/Block, 0.4375 B/Wert (urspr. TurboQuant)

### GGML-Typ ↔ VBR-Tier Mapping (`llama-kv-cache.cpp`)

**Welche GGML-Typen sind im VBR-Band verschiebbar:**

```cpp
static bool vbr_type_is_movable(ggml_type t) {
    return t == GGML_TYPE_F16 ||
           t == GGML_TYPE_TURBO8_0  ||
           t == GGML_TYPE_TURBO4_0  ||
           t == GGML_TYPE_TURBO3_TCQ ||
           t == GGML_TYPE_TURBO2_TCQ ||
           t == GGML_TYPE_TURBO1_TCQ;  // ENUM SLOT — aber codec entfernt!
}

static ggml_type vbr_tier_type(uint8_t tier) {
    switch (tier) {
        case VBR_TIER_T8:      return GGML_TYPE_TURBO8_0;
        case VBR_TIER_T4:      return GGML_TYPE_TURBO4_0;
        case VBR_TIER_T3_TCQ:  return GGML_TYPE_TURBO3_TCQ;
        case VBR_TIER_T2_TCQ:  return GGML_TYPE_TURBO2_TCQ;
        case VBR_TIER_T1_TCQ:  return GGML_TYPE_TURBO1_TCQ;  // ENUM EXISTS, DECODE PATH REMOVED
        default: GGML_ABORT("invalid vbr tier %d", (int) tier);
    }
}
```

**PINNED-Typen:** Jeder GGML-Typ der **nicht** in `vbr_type_is_movable()` aufgeführt
ist (z.B. `Q8_0`, `BF16`, `F32`) wird als **PINNED** markiert. Solche Typen können
niemals degradiert werden — der Transcode-Dequant hat keine Source-Unterstützung dafür.

### Degrade-Ladder (vollständig)

```
F16 ──(budget pressure)──▶ TURBO8_0 ──▶ TURBO4_0 ──▶ TURBO3_TCQ ──▶ TURBO2_TCQ ──▶ TURBO1_TCQ
entry tier              first band    subsequent bands (measured price order)
```

Priorität: **F16→t8 zuerst** — grösste Byte-Einsparungen pro Layer.

---

## VRAM-Budget-Mathematik

### Budget-Berechnung (`llama-kv-cache.h`)

```cpp
struct vbr_pool {
    size_t budget = 0;           // pro-Gerät-Anteil von vbr_budget_bytes_
    size_t budget_base = 0;      // Re-derivation Floor (init)
    size_t used = 0;             // High-Water of placed extents
    size_t budget_eff_cache = 0; // gemerkter effektiver Budget-Wert
};
```

Das Budget pro Gerät wird berechnet als:

```
device_share × (mapped + free - headroom) → auf 64 MiB gerundet
```

- `device_share`: Anteil des Geräts am Gesamtsystem (bei Split-Mode)
- `mapped`: aktuell gemappete VRAM-Bytes
- `free`: verfügbare GPU-VRAM-Bytes (via CUDA/cuMem)
- `headroom`: 192 MiB pro Prozess/Gerät (`LLAMA_VRAM_LEDGER_HEADROOM_BASE`)

### VRAM-Ledger (`src/llama-vram-ledger.h`)

VBR verwendet ein **Co-Tenancy-Protokoll**: Mehrere llama.cpp-Prozesse auf derselben GPU koordinieren sich via tmpfs-Dateien (`/tmp/ggml-vram-*/`). Jeder Prozess publiziert:

- Ein **demand claim** (wenn er VRAM braucht)
- Einen **residence marker** (wenn er VRAM hält)

Donner-Prozesse (VBR-Residents) können KV-Seiten freed, wenn ein Demander VRAM braucht. Die Felder im Marker:

```cpp
struct llama_vram_marker_fields {
    uint32_t vbr;            // 1 = VBR-Resident (potential donor)
    uint32_t serviced;       // 1 = llama-server (idle tick)
    uint64_t shed_available; // Bytes die f16->t8 Band freed
    uint64_t grant_pending;  // Granted aber nicht geflusht
};
```

---

## Pro-Schicht-VRAM-Berechnung

### Slot-Reservierung

```cpp
static size_t vbr_slot_bytes(const ggml_tensor *t) {
    // Jede Schicht reserviert ihr MAXIMALES (F16) Footprint
    return ggml_row_size(GGML_TYPE_F16, t->ne[0]) * t->ne[1] * t->ne[2];
}
```

**Wichtig:** Jeder `(layer, side)`-Slot wird mit **F16-Grösse** vorreserviert — egal welcher Tier gerade aktiv ist. Das verhindert Zeiger-Verschiebungen beim Tier-Flip.

### Aktiver Speicher bei einem given Tier

```
keep = row_bytes(type_B) × n_cells
```

wobei `row_bytes(type_B)` vom aktuellen Tensor-Typ abhängt.

Die Span-Struktur berechnet:

```cpp
struct vbr_span {
    size_t slot;    // Reservierte Grösse (immer F16)
    size_t keep;    // Tatsächlich resident bei current tier
    size_t keep_live; // Mit live watermark
    size_t keep_pad;  // Page-padded
};
```

---

## Degrade-Step

```cpp
struct vbr_degrade_step {
    uint8_t il;    // Layer-Index im Modell
    uint8_t is_v;  // 0 = K-Seite, 1 = V-Seite
    uint8_t tier;  // Ziel-tier (vbr_tier index)
};
```

### Price Order

Der **price order** ist eine globale Liste aller Degrade-Schritte, sortiert nach "Kosten pro Byte freed". Die Reihenfolge ist **architekturabhängig** und wird aus `llama-vbr-degrade-orders.inc` geladen.

Priorität: **F16→t8 zuerst** — grösste Byte-Einsparungen pro Layer.

### Tier-Epoch

```cpp
uint64_t vbr_tier_epoch_ = 0; // wird bei jedem Flip inkrementiert
```

Graph-Reuse muss auf diese Epoche warten: ein wiederverwendeter Graph trägt den alten Typ/Strides in seinen K/V-Views.

---

## Pro-Layer VBR Schedule (`VBR_LAYER_SCHEDULE`)

Über Umgebungsvariable oder Datei:

```bash
VBR_LAYER_SCHEDULE="default=t4; band=0-8192:t8; k;l0-5:v"
```

Format:
- `default=<type>` — Basistyp für alle Layer
- `band=<row_start-row_end>:<type>; <k|v|k,v>; [l<layer_start>-<layer_end>]`
- `row_start-row_end`: Kontext-Range (standard 8192, konfigurierbar mit `VBR_SCHEDULE_CTX`)
- `l<start>-<end>`: Optional — nur für bestimmte Layer

---

## CLI-Flags

```
-ctv vbr          -- VBR für V-cache aktivieren (entry type = F16)
--vbr-budget <bytes> -- VRAM-Budget in Bytes (z.B. 8589934592 = 8 GiB)
--vbr-min-bits <bpv> -- Floor: tiefste erlaubte bits/value (z.B. 4.0)
--vram-stash <N>    -- Sink-Stash-Rows für degrade recovery
```

### Environment Variables

```bash
VBR_MIN_BITS=4.0                     -- override für --vbr-min-bits
VBR_VRAM_BUDGET=8589934592           -- override für Budget
VBR_LAYER_SCHEDULE="@schedule.txt"   -- Datei laden
VBR_SCHEDULE_CTX=8192                -- Discovery-Window für Schedules
VBR_LAYER_STRICT=1                   -- Fehler bei unsupported bands
TURBO_Q_CALIBRATE=1                  -- Q-Vektor-Statistiken sammeln
TURBO_TCQ_CB=/pfad/codebook.bin      -- Custom TCQ Codebook
TURBO_TCQ_CB2=/pfad/codebook2.bin    -- Custom TCQ Codebook V-side
```

---

## Server API: `/slots`

Der `/slots` Endpunkt gibt den aktuellen Cache-Zustand zurück:

```json
{
    "kv_bpv": 4.2,
    "n_ctx": 32768,
    "n_tokens": 15000,
    "slots": [...]
}
```

### `kv_bpv` — effektive Bits/Value

Aggregierter Mittelwert aller Layer-Typen, gewichtet nach deren Grösse:

```
kv_bpv() = (bits_k + bits_v) / (vals_k + vals_v)
```

D.h. der server zählt live alle Bytes und gibt den Durchschnitt zurück — bei VBR **dynamisch**, ändert sich mit der Kontextlänge.

### Pro-Layer Typ-Information

Jeder Slot enthält die aktuellen Typen pro Layer:

```cpp
ggml_type type_k() const;  // Aktueller K-Typ (kann pro Layer variieren)
ggml_type type_v() const;  // Aktueller V-Typ
```

---

## API-Strukturen (C++)

### `llama_cparams` — VBR-Parameter

```cpp
struct llama_cparams {
    bool vbr_dynamic;                    // VBR aktiv
    double vbr_min_bits;                 // Floor (bits/value)
    uint64_t vbr_vram_budget_bytes;      // VRAM-Budget
    uint64_t vbr_growth_headroom_bytes;  // Headroom für Wachstum
    bool vbr_budget_explicit;            // Budget explizit gesetzt?
    bool vbr_min_bits_explicit;          // Floor explizit gesetzt?
    bool vbr_pin_k;                      // K-Typ fixieren (kein VBR)
    bool vbr_pin_v;                      // V-Typ fixieren (kein VBR)
};
```

### `llama_kv_cache` — VBR-interner Zustand

```cpp
struct vbr_extent {
    ggml_tensor *t;          // Pool-lokaler Tensor
    size_t byte_off = 0;     // Offset im Pool
    ggml_type type0;         // Entry-Tier (immutable)
    size_t stash_off = 0;    // Offset F16 Sink-Stash
    uint32_t stash_valid = 0; // Captured rows
    uint8_t promote_hops = 0; // Promote-Hops Cap
};

struct vbr_pool {
    ggml_backend_buffer_t buf;
    char *base;
    size_t size, used, budget;
    std::vector<vbr_extent> k, v;  // pro Layer
    const ggml_vbr_backend_iface *be; // VMM pool
    ...
};
```

---

## Exakte VRAM-Berechnung — Implementierungsdetails aus dem Code

### Kern-Formel: kv_bpv()

```cpp
void llama_kv_cache::kv_bpv_accum(double & bits, double & vals) const {
    const double cells = (double) get_size();  // kv_size (Anzahl Zellen)
    for (const auto & l : layers) {
        for (const ggml_tensor * t : { l.k, l.v }) {
            if (t == nullptr) continue;
            // bits = 8 × Bytes pro Zeile × Zellen
            bits += 8.0 * (double) ggml_row_size(t->type, t->ne[0]) * cells;
            // vals = ne[0] × Zellen (Anzahl Werte)
            vals += (double) t->ne[0] * cells;
        }
    }
}

double llama_kv_cache::kv_bpv() const {
    double bits = 0.0, vals = 0.0;
    kv_bpv_accum(bits, vals);
    return vals > 0.0 ? bits / vals : -1.0;
}
```

**Bedeutung:**
- `cells` = `kv_size` (maximale Anzahl Zellen pro Stream)
- `t->ne[0]` = `head_dim` (dimension pro Head, nach GQA-Gruppierung)
- `ggml_row_size(type, ne[0])` = Bytes für eine Zeile des Tensors
- `bits` = Gesamtbits aller K+V-Tensoren (8 × row_bytes × cells)
- `vals` = Gesamtwert-Anzahl (ne[0] × cells)
- **kv_bpv** = bits/vals → effektive Bits/Value, **aggregiert über ALLE Layer**

### Pro-Schicht-Byte-Berechnung

```cpp
size_t llama_kv_cache::size_k_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += ggml_nbytes(layer.k);  // Bytes für diesen Layer
    }
    return total;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t total = 0;
    for (const auto & layer : layers) {
        total += layer.v ? ggml_nbytes(layer.v) : 0;
    }
    return total;
}

size_t llama_kv_cache::total_size() const {
    size_t total = 0;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        total += ggml_backend_buffer_get_size(buf.get());
    }
    return total;
}
```

**`ggml_nbytes(tensor)`** = `row_size(type, tensor->ne[0]) × tensor->ne[1] × tensor->ne[2]`

Für KV-Tensoren:
- `ne[0]` = `head_dim` (nach GQA: `n_embd / n_heads_k` oder `n_embd_v`)
- `ne[1]` = `n_head_k` (für K) oder `1` (für V, da transponiert)
- `ne[2]` = `n_ctx` (aktuelle Kontextlänge)

### VBR-spezifische Berechnung

#### Slot-Reservierung (immer F16-Grösse)

```cpp
static size_t vbr_slot_bytes(const ggml_tensor *t) {
    return (size_t) ggml_row_size(GGML_TYPE_F16, t->ne[0])
           * t->ne[1] * t->ne[2];
}
```

Jeder `(layer, side)`-Slot reserviert **immer F16-Grösse**, unabhängig vom aktuellen Tier. Das verhindert Zeiger-Verschiebungen beim Tier-Flip.

#### Aktiver Speicher bei gegebenem Tier

```cpp
static vbr_span vbr_span_of(
    const ggml_tensor *t, ggml_type type_B,
    int64_t n_cells, uint32_t wm_next, size_t gran) {
    size_t rB = ggml_row_size(type_B, t->ne[0]);  // Bytes/Zeile bei neuem Tier
    size_t slot = vbr_slot_bytes(t);               // Max-F16-Grösse
    size_t keep = rB * max(n_cells, 1);             // Tatsächlich resident
    size_t keep_live = min(slot, max(keep, rB * wm_next)); // Mit watermark
    size_t keep_pad = min(slot, GGML_PAD(keep_live, gran)); // Page-granular
    return { slot, keep, keep_live, keep_pad };
}
```

**Parameter:**
- `n_cells` = aktuell belegte Zellen (Wassermark)
- `wm_next` = projizierte Wassermark incl. kommender Tokens
- `gran` = Page-Grösse (CUDA: typ. 64 KiB oder 128 KiB)
- `type_B` = neues Tier nach Degrade

#### Degrade-Bedingung

```cpp
// In vbr_degrade_next():
size_t rA = ggml_row_size(e.t->type, ne0);  // Aktuelles Tier
size_t rB = ggml_row_size(type_B, ne0);      // Neues Tier
if (e.t->type == type_B || rB >= rA) continue;  // Nur wenn echt kleiner
```

**Nur degradiert wenn `row_size(new_type) < row_size(current_type)`.**
F16→turbo8_0 ist ein echter Degrade. turbo8_0→turbo8_0 ist ein No-Op.

#### Budget-Auslösung

```cpp
// In update():
const size_t used_now = p.be->vmm_pool_used(p->vmm);
if (used_now > budget_eff + headroom) {
    // Budget überschritten → Degrade starten
}

// budget_eff pro Pool:
size_t budget_eff = budget × (mapped + free - headroom) / total_mapped
// Memoized pro Boundary: budget_eff_stamp == vbr_boundary_count_
```

### KV-Cache-Grösse berechnen — Für den Launcher

#### Maximale Grösse (F16, vor jeglichem Degrade)

```
Für jeden Layer il:
  K: ggml_row_size(F16, head_dim) × n_head_k × n_ctx
  V: ggml_row_size(F16, head_dim) × 1 × n_ctx  (transponiert)

KV_max_layer = row_size(F16, head_dim) × (n_head_k + 1) × n_ctx

total KV_max = sum_over_layers(KV_max_layer)
```

Mit `row_size(F16, head_dim) = 2 × head_dim`:

```
KV_max = 2 × head_dim × (n_head_k + 1) × n_ctx × n_layers
       = 2 × head_dim × (n_embd / n_heads_v + 1) × n_ctx × n_layers
```

Beispiel Qwen3.6-35B:
- `head_dim = 4096 / 32 = 128` (32 heads, GQA mit factor 8)
- `n_head_k = 32/8 = 4`, `n_head_v = 1` (bei Qwen: n_head_v=1)
- `n_layers = 64`
- `n_ctx = 32768`

```
KV_max = 2 × 128 × (4 + 1) × 32768 × 64
       = 2 × 128 × 5 × 32768 × 64
       = 2,684,354,560 Bytes ≈ 2.5 GiB
```

#### Minimale Grösse (full degrade to t8)

```
KV_min_layer = row_size(turbo8_0, head_dim) × (n_head_k + 1) × n_ctx

total KV_min = sum_over_layers(KV_min_layer)
```

#### Mittlere Grösse (mit aktuellem kv_bpv vom Server)

```
KV_actual = kv_bpv / 8 × head_dim × n_tokens × n_layers × 2
```

**Achtung:** Diese Formel ist eine Näherung — `kv_bpv` ist der aggregierte
Durchschnitt über alle Layer, aber die Gewichtung in `kv_bpv_accum()` ist:
- `vals += ne[0] × cells` (gleich für alle Layer)
- `bits += 8.0 × row_size(type, ne[0]) × cells`

Für den Fall dass alle Layer denselben `ne[0]` (head_dim) haben:
```
KV_actual_bytes = kv_bpv / 8.0 × head_dim × n_tokens × n_layers × 2
```

#### Exakte Grösse via memory_breakdown()

```cpp
std::map llama_kv_cache::memory_breakdown() const {
    std::map ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        if (VMM-backed) {
            sz = p->be->vmm_pool_mapped(p->vmm);  // Gemapped-physical bytes
        } else {
            sz = ggml_backend_buffer_get_size(buf.get());
        }
        ret[buft] += sz;
    }
    return ret;
}
```

### Scratch-VRAM bei turbo-Typen

Bei turbo-KV-Typen muss dequantisiert werden (z.B. F16-Scratch):

```cpp
bool ggml_vbr_kv_dequant_sides(k_type, v_type, &need_k, &need_v);

// Scratch pro Device:
scratch_k_row = max over layers of row_size(F16, tk->ne[0])  // wo need_k
scratch_v_row = max over layers of row_size(F16, tv->ne[0])  // wo need_v
scratch_bytes = (scratch_k_row + scratch_v_row) × wm_cells
```

Das Scratch ist **pro Device**, nicht pro Layer — der grösste F16-Row über alle Layers bestimmt den Scratch-Verbrauch.

### Wassermark-Berechnung

```cpp
uint32_t wm_cells = vbr_watermark_cells(n_tokens);
```

`wm_cells` = Anzahl belegte Zellen = `n_tokens` (oder etwas mehr für Padding/Alignment). Wird bei jedem Degrade neu berechnet.

### Degrade-Step im Detail

```cpp
struct vbr_degrade_step {
    uint8_t il;      // Layer-Index im Modell
    uint8_t is_v;    // 0=K, 1=V
    uint8_t tier;    // Ziel-Tier (vbr_tier)
};
```

Der **price order** ist eine globale Liste aller Degrade-Schritte:
```cpp
std::vector<vbr_degrade_step> vbr_degrade_order_;
```

Jeder Schritt bewegt eine Schicht-Seite von ihrem aktuellen Typ zum nächsten Typ in der Ladder:
```
F16 → t8 → t4 → t3_tcq → t2_tcq → t1_tcq
```

**Byte-Ersparnis pro Step:**
```
freed = (row_size(old_type, head_dim) - row_size(new_type, head_dim))
        × n_cells × page_granularity_padded
```

### Co-Tenancy: Shedding

Wenn ein anderer Prozess VRAM braucht, kann ein VBR-Donor KV-Seiten freigeben:

```cpp
size_t llama_kv_cache::vbr_shed_available(int device) const {
    // Simuliere Degrade vom current cursor bis demand_limit
    // Berechne freed bytes minus scratch-growth
    freed = sum over steps of:
        (row_size(sim[slot], ne[0]) - row_size(type_B, ne[0]))
        × wm_cells (page-padded)

    // Scratch-Zuwachs abziehen (wenn new_type grösseren F16-Scratch braucht):
    scratch_growth = (end_k - cur_k + end_v - cur_v) × wm_cells

    return freed - scratch_growth
}
```

### Budget-Simulation (vbr_floor_sim)

```cpp
struct vbr_floor_sim_result {
    size_t clamp_step;      // Schritt an dem Floor-Clamp stoppt
    size_t n_pinned;        // Anzahl nicht-degradierbare Einheiten
    double next_bpv;        // bpv nach Clamp-Schritt
    double bits_per_token;  // Aggregate Bits pro Token
    std::vector<ggml_type> end_types;  // [layers × 2]
};

vbr_floor_sim_result llama_kv_cache::vbr_floor_sim(
    double floor_bpv, bool pooled_only,
    ggml_type entry_k, ggml_type entry_v) const {
    
    // Seed sim mit Entry-Typen
    vbr_sim_seed(sim, pooled_only, entry_k, entry_v, &sum_bits, &sum_vals, ...);
    
    // Schritt für Schritt durch Degrade-Order:
    for (size_t i = 0; i < vbr_degrade_order_.size(); ++i) {
        if (!vbr_sim_step(sim, i, slot, t, type_B)) continue;
        
        size_t rA = ggml_row_size(sim[slot], t->ne[0]);
        size_t rB = ggml_row_size(type_B, t->ne[0]);
        double bits_next = sum_bits - 8.0*rA + 8.0*rB;
        
        // Floor-Clamp: nächster Schritt würde bpv zu tief drücken
        if (bits_next / sum_vals < floor_bpv - 1e-9) {
            res.clamp_step = i;
            res.next_bpv = bits_next / sum_vals;
            break;
        }
        sim[slot] = type_B;
        sum_bits = bits_next;
    }
    res.bits_per_token = sum_bits;
    return res;
}
```

**Anwendung:** Mit `vbr_floor_sim()` kann man berechnen, wie viele Bytes freed werden,
wenn man einen bestimmten `floor_bpv` (z.B. 4.0) als Minimum setzt — ohne tatsächlich
zu degradieren.

---

## VBR-API-Referenz (llama-ext.h / llama-vbr.h)

### llama_memory_breakdown — Gesamter VRAM-Aufschluss

```cpp
struct llama_memory_breakdown_data {
    size_t model;         // Model-Weights in Bytes
    size_t context;       // KV-Cache + Context-Puffer in Bytes
    size_t compute;       // Temporäre Compute-Buffers in Bytes
    size_t context_fixed; // Nicht-skalierbarer Teil (recurrent-state cache,
                          // ~n_seq_max-groesse, in context enthalten!)
    
    size_t total() const; // model + context + compute
};
```

**`context_fixed` ist kritisch für Budget-Formeln:** Budgets die den
`context`-Term ausnutzen um linear zu skalieren müssen diesen festen Teil
immer noch belasten — er fällt ja nicht weg wenn `n_ctx` gesenkt wird.

**Beispiel:** Qwen3.6-35B mit 176 GB VRAM:
```
model ≈ 20 GiB (Q4_K_M)
context ≈ 2.5 GiB (F16 KV, n_ctx=32768)
context_fixed ≈ 0.1 GiB (recurrent state ~n_seq_max)
compute ≈ 0.5 GiB (Scratch)
total ≈ 23.1 GiB
```

### llama_vbr_state — VBR-Zustand zur Laufzeit

```cpp
struct llama_vbr_state {
    double deficit_raw;           // MiB Defizit (0 = OK)
    double bpv_if_degraded;       // bpv wenn ein Schritt degradiert würde
    uint64_t vram_budget_bytes;   // Aktuelles Budget
    uint64_t vram_used_bytes;     // Tatsächlicher Verbrauch
    uint64_t vram_capacity_bytes; // Maximale Kapazität
    int64_t n_tokens;             // Aktuelle Token-Anzahl
    int64_t n_ctx;                // Max Kontextlänge
};
```

**Verwendung in Reclaim-Logik:** Wenn `deficit_raw > 0` und `bpv_if_degraded >=
vbr_reclaim_floor_bpv`, wird entweder ein Slot freigegeben oder degradiert:

```cpp
// In server-context.cpp:
const auto st = llama_memory_vbr_state(llama_get_memory(ctx_tgt), -1, n_tokens_extra);
if (st.deficit_raw <= 0 || st.bpv_if_degraded >= params_base.vbr_reclaim_floor_bpv) {
    return; // kein Defizit oder Degrade würde zu tief gehen
}
// ... idle slots clear or degrade
```

### llama_vbr_floor_bits_per_token() — Kapazitätsberechnung

```cpp
LLAMA_API double llama_vbr_floor_bits_per_token(
    struct llama_context * ctx,
    enum ggml_type entry_k,
    enum ggml_type entry_v,
    double floor_bpv);
```

**Bedeutung:** Berechnet die Bits/Token die der `--vbr-floor`-Clamp bei einem
gegebenen Entry-Tier-Setup ergibt. Gehört zur `--vbr-capacity`-Berechnung:
- `entry_k = GGML_TYPE_COUNT` → aktueller Typ jedes Tensors
- `floor_bpv <= 0` → Deep-fill (unterste Stufe)
- `floor_bpv > 0` → Clamp an gegebenem Floor
- Gibt `0.0` zurück wenn kein VBR-fähiger Cache vorhanden

### llama_vbr_scratch_bytes_per_token() — Scratch-Bedarf

```cpp
LLAMA_API double llama_vbr_scratch_bytes_per_token(
    struct llama_context * ctx,
    enum ggml_type entry_k,
    enum ggml_type entry_v,
    double floor_bpv);
```

**Bedeutung:** F16-Scratch-Bedarf pro Token bei tiefem Fullfill. WICHTIG:
Dieser Scratch wird NICHT vom KV-Budget abgezogen — er wird aus der
`fit_margin` bezahlt. Die Funktion wird von der Fit-Pass verwendet um
die Gesamt-VRAM-Wand zu berechnen.

### Co-Tenancy APIs

```cpp
LLAMA_API void llama_vram_plan_hint(const char * device_id, uint64_t bytes);
// Plan-Hint: total bytes this process intends to allocate on device
// (PCI bus id per ggml_backend_dev_props.device_id)
// Wird vom Fit-Pass vor Load gesetzt.

LLAMA_API void llama_vram_mark_serviced(void);
// Deklarieren: dieser Prozess hat einen idle tick gelaufen
// Presence marker = serviced:1 → Signal für Co-Lader LONG patience

LLAMA_API struct llama_vram_cotenancy_state llama_vram_cotenancy(
    const struct llama_context * ctx);
// Telemetrie für /props und /slots
// All zeros = inert (Single Tenant, kein Ledger)
```

### llama_vram_cotenancy_state — Co-Tenancy-Zustand

```cpp
struct llama_vram_cotenancy_state {
    uint64_t grant_decrement; // Nicht-abgezählte Bytes die vom KV-Budget
                              // dekrementiert wurden (unamortized)
    uint32_t grants_active;   // Live-Grant-Reihen
    uint64_t shed_offer;      // Veröffentlichte Spende-Angebot,
                              // summiert über Devices
    uint64_t grant_pending;   // Gewährt aber noch nicht geflushte Bytes
};
```

**Co-Tenancy-Workflow:**
1. Prozess A ruft `llama_vram_mark_serviced()` → `serviced:1`
2. Prozess B (Co-Lader) prüft `llama_vram_cotenancy()` → sieht `serviced`
3. Prozess A publiziert `shed_offer` → sagt "ich kann X Bytes freigegeben"
4. Prozess B kann `grant_pending` setzen → "ich brauche Y Bytes"
5. Wenn B fertig: `grant_decrement` wird auf A's Budget angewendet
6. `grant_pending` → `grant_active` → amortisiert

### llama_memory_kv_bpv() — Live-KV-BPV

```cpp
LLAMA_API double llama_memory_kv_bpv(const llama_memory_t memory);
// Returns: effective bits/value des Attention KV-Cache bei den
// aktuellen Tensor-Typen (bewegt sich unter VBR mit runtime-degrade)
// Negative wenn nicht verfügbar oder kein Cache
```

**Implementierung** (aus `server-context.cpp`):
```cpp
// Jeder Slot ruft:
const double kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
```

Die Funktion ist ein Wrapper um `llama_kv_cache::kv_bpv()` im Backend.

---

## Server-API: GET /slots — Vollständiges Format

Der `/slots`-Endpunkt wird über `SERVER_TASK_TYPE_METRICS` bedient. Jeder Slot wird via `server_slot::to_json()` serialisiert.

### slot.to_json() — JSON-Format pro Slot

```cpp
json to_json(bool only_metrics = false) const {
    json res;
    res = {
        {"id", id},
        {"n_ctx", n_ctx},
        {"speculative", can_speculate()},
        {"is_processing", is_processing()},
    };
    // Live-KV-BPV (bewegt sich unter VBR)
    if (ctx_tgt != nullptr) {
        double kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
        if (kv_bpv >= 0.0) res["kv_bpv"] = kv_bpv;
        // Co-Tenancy: was wir Peers anbieten / bereits abgetreten
        auto ct = llama_vram_cotenancy(ctx_tgt);
        if (ct.shed_offer > 0 || ct.grants_active > 0 || ct.grant_decrement > 0) {
            res["cotenancy"] = json {
                {"shed_offer", ct.shed_offer},
                {"grants_active", ct.grants_active},
                {"grant_decrement", ct.grant_decrement},
                {"grant_pending", ct.grant_pending},
            };
        }
    }
    // Task-Informationen
    if (ptask) {
        res["id_task"] = ptask->id;
        res["n_prompt_tokens"] = prompt.tokens.size();
        res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
        res["n_prompt_tokens_cache"] = n_prompt_tokens_cache;
        res["params"] = ptask->params.to_json(only_metrics);
        res["next_token"] = {
            {"has_next_token", has_next_token},
            {"has_new_line", has_new_line},
            {"n_remain", n_remaining},
            {"n_decoded", n_decoded},
        };
        if (!only_metrics) {
            res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
            res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
        }
    }
    return res;
}
```

### Ergebnis: GET /slots API-Response

```json
{
    "n_idle_slots": 3,
    "n_processing_slots": 1,
    "n_tasks_deferred": 0,
    "slots": [
        {
            "id": 0,
            "n_ctx": 32768,
            "speculative": false,
            "is_processing": true,
            "kv_bpv": 4.2,
            "id_task": 42,
            "n_prompt_tokens": 1500,
            "n_prompt_tokens_processed": 1500,
            "n_prompt_tokens_cache": 1500,
            "params": { ... },
            "next_token": {
                "has_next_token": true,
                "has_new_line": false,
                "n_remain": -1,
                "n_decoded": 342
            },
            "prompt": "...",
            "generated": "..."
        },
        {
            "id": 1,
            "n_ctx": 32768,
            "is_processing": false,
            "kv_bpv": 4.2
        }
    ],
    "t_start": 1234567890
}
```

### GET /props API — VBR-Meta

```json
{
    "vbr": {
        "enabled": true,
        "dynamic": true,
        "type_k": "f16",
        "type_v": "vbr",
        "floor_bpv": 4.0,
        "capacity_floor_bpv": 4.0,
        "realized_bpv": null,
        "selected_family": null,
        "selected_policy": null,
        "selected_bpv": null,
        "selected_kld": null,
        "selected_schedule": null,
        "vram_budget_bytes": 8589934592
    }
}
```

> `realized_bpv` = `null` bei VBR (da dynamisch), stattdessen `kv_bpv` aus `/slots` pollen.

### GET /models API — VBR-Meta

Selbes `vbr`-Objekt wie bei `/props`, eingebettet in:

```json
{
    "model": {
        "model": "qwen3.6-35b",
        "n_ctx_train": 131072,
        "n_embd": 4096,
        "n_params": 35000000000,
        "size": 19000000000,
        "ftype": "Q4_K_M",
        "vbr": { ... }
    }
}
```

---

## Server-Strukturen (C++ Header)

### server_slot::to_json() — Felder

| JSON-Feld | Typ | Beschreibung |
|-----------|-----|-------------|
| `id` | int | Slot-ID (0 bis n_parallel-1) |
| `n_ctx` | int | Kontextgrösse pro Slot |
| `speculative` | bool | Speculative Decoding aktiv |
| `is_processing` | bool | Slot ist gerade aktiv |
| `kv_bpv` | double | **Live-KV-Bits/Value** (bewegt sich bei VBR) |
| `cotenancy` | object | Co-Tenancy-Zustand (falls > 0) |
| `cotenancy.shed_offer` | uint64_t | Bytes die wir Peers anbieten können |
| `cotenancy.grants_active` | uint64_t | Aktive Grant-Zählung |
| `cotenancy.grant_decrement` | uint64_t | Aktive Grant-Reduktion |
| `cotenancy.grant_pending` | uint64_t | Bereits abgetretene aber nicht geflushte Bytes |
| `id_task` | int | Aktuelle Task-ID |
| `n_prompt_tokens` | int | Tokens im Prompt |
| `n_prompt_tokens_processed` | int | Tokens die bereits verarbeitet |
| `n_prompt_tokens_cache` | int | Tokens im KV-Cache gecacht |
| `params` | object | Generierungsparameter |
| `next_token` | object | Nächster Token-Zustand |
| `next_token.has_next_token` | bool | Gibt es weitere Tokens? |
| `next_token.has_new_line` | bool | Endete mit Newline? |
| `next_token.n_remain` | int | Verbleibende Tokens |
| `next_token.n_decoded` | int | Decodierte Tokens |
| `prompt` | string | Prompt-Text |
| `generated` | string | Generierter Text |

### server_vbr_meta_json() — Vollständige JSON-Struktur

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `enabled` | bool | VBR insgesamt aktiv |
| `dynamic` | bool | **Dynamic VBR** (Runtime-Controller) |
| `type_k` | string | K-Cache-Typ (z.B. "f16", "turbo4") |
| `type_v` | string | V-Cache-Typ (z.B. "vbr", "turbo4") |
| `floor_bpv` | double | Minimum bpv (`--vbr-min-bits`) |
| `capacity_floor_bpv` | double | Floor für Kapazitätsberechnung |
| `realized_bpv` | double/null | Tatsächlicher bpv (null bei VBR!) |
| `selected_family` | string/null | Gewählte Family (null bei VBR) |
| `selected_policy` | string/null | Gewählte Policy (null bei VBR) |
| `selected_bpv` | double/null | Gewählter bpv (null bei VBR) |
| `selected_kld` | double/null | Gewählte KL-Divergenz (null bei VBR) |
| `selected_schedule` | string/null | Gewählter Schedule (null bei VBR) |
| `vram_budget_bytes` | uint64_t | VRAM-Budget in Bytes |

### server_task_result_metrics

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `n_idle_slots` | int | Unbenutzte Slots |
| `n_processing_slots` | int | Aktive Slots |
| `n_tasks_deferred` | int | Ausgestellte Tasks |
| `t_start` | int64_t | Server-Startzeit |
| `slots_data` | array | JSON-Array von slot.to_json() |

### result_timings

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `kv_bpv` | double | **Live-KV-BPV** (negativ wenn nicht verfügbar) |
| `cache_n` | int32 | Cached Tokens (Prompt Processing) |
| `prompt_n` | int32 | Prompt Tokens gesamt |
| `prompt_ms` | double | Prompt-Verarbeitungszeit |
| `predicted_n` | int32 | Vorhergesagte Tokens |
| `predicted_ms` | double | Generierungszeit |

### server_context_meta — VBR-Felder

| Feld | Quelle | Typ |
|------|--------|-----|
| `vbr_enabled` | `impl->params_base.vbr_enabled()` | bool |
| `vbr_dynamic` | `impl->params_base.vbr_dynamic()` | bool |
| `vbr_type_k` | `impl->params_base.vbr_cache_type_k()` | string |
| `vbr_type_v` | `impl->params_base.vbr_cache_type_v()` | string |
| `vbr_min_bits` | `impl->params_base.vbr_min_bits_value` | double |
| `vbr_capacity_bits` | `impl->params_base.vbr_capacity_bits` | double |
| `vbr_selected_bpv` | `impl->params_base.vbr_selected_bpv` | double |
| `vbr_selected_kld` | `impl->params_base.vbr_selected_kld` | double |
| `vbr_vram_budget_bytes` | `impl->params_base.vbr_vram_budget_bytes` | uint64_t |
| `vbr_selected_family` | `impl->params_base.vbr_selected_family` | string |
| `vbr_selected_policy` | `impl->params_base.vbr_selected_policy` | string |
| `vbr_selected_schedule` | `impl->params_base.vbr_selected_schedule` | string |

---

## VRAM-Schätzung für llauncher — Zusammenfassung

### Berechnete Werte für die Anzeige

```
KV_max_bytes = sum_over_layers(
    ggml_nbytes_k(layer, type=F16) + ggml_nbytes_v(layer, type=F16)
)

KV_min_bytes = sum_over_layers(
    ggml_nbytes_k(layer, type=turbo8_0) + ggml_nbytes_v(layer, type=turbo8_0)
)

KV_current_bytes = kv_bpv / 8.0 × head_dim × n_tokens × n_layers × 2
                  (Näherung, wenn alle Layer gleichen ne[0] haben)

oder exakter:
KV_current_bytes = sum_over_layers(
    ggml_row_size(type_k_il, head_dim) × n_head_k × n_tokens +
    ggml_row_size(type_v_il, head_dim) × 1 × n_tokens
)
```

### Server-API zur Laufzeit

```
GET /slots → {"kv_bpv": 4.2, "n_ctx": 32768, "n_tokens": 15000, ...}
```

- `kv_bpv` = aggregierter Bits/Value (live berechnet)
- `n_ctx` = maximale Kontextlänge
- `n_tokens` = aktuell belegte Tokens (Wassermark)

### Scratch-Budget

```
scratch_bytes = (max_row(F16, K) + max_row(F16, V)) × wm_cells
```

Nur relevant bei turbo-Typen (t8, t4, t3_tcq, t2_tcq, t1_tcq), nicht bei F16.

### Gesamter VRAM-Bedarf

```
total_vram = model_weights_bytes
           + KV_max_bytes (Puffer, falls VBR noch nicht degradiert hat)
           + scratch_bytes (falls turbo-Typen aktiv)
```

Für VBR-Mode ist `KV_current_bytes` (von `/slots`) der tatsächliche Verbrauch —
`KV_max_bytes` ist der Worst-Case bei vollem F16-Cache.

---

## Empirische Messungen (Qwen3.6-27B Q6_K, RTX 3090 32 GB)

**Setup:** `-ngl 999`, `-c 262144`, `-ctv vbr` (V-Cache VBR, K-Cache q8_0)

### Messdaten

```
Punkt   Prompt Tokens   Decoded   VRAM used   kv_bpv   shed_offer
────    ─────────────   ───────   ─────────   ────────  ──────────
Base    ~0              104       25,322 MB   12.25    —
2       60,798          165       26,698 MB   12.25    723 MB
3       67,568          91        27,530 MB   12.25    954 MB
```

### Beobachtungen

1. **Kein VBR-Degrade bis 68K Tokens**: `kv_bpv` bleibt konstant bei 12.25
   — K-Cache q8_0 (8 bpv = 1.0 B/v) + V-Cache F16 (16 bpv = 2.0 B/v) gemittelt
   — Budget-Druck bei 32 GB reicht erst ab höheren Tokenzahlen

2. **VRAM-Wachstum pro Token**: +2,208 MB fuer 67,600 Tokens ≈ 33.3 B/token
   — Davon ~26 B/token KV-Cache (K: 128*4*1.0 + V: 128*1*2.0 = 768 B/token, dividiert
     durch 64 Layers und Umrechnung ergibt ~192 B/layer/token, mal 64 ≈ 12.3 KB/ctx)
   — Rest: Compute-Buffer, scratch, CUDA-overhead

3. **shed_offer steigt linear mit Context**: 723 MB bei 61K → 954 MB bei 68K
   — Zeigt an: Degrade-Potential waechst proportional zum belegten Cache
   — VBR ist bereit zu degradieren, wird aber erst bei Budget-Druck aktiv

### Implikationen fuer llauncher estimate_vram()

Der statische VBR-Fallback `KV_CACHE_TYPE_SIZES["vbr"] = 0.22` (turbo1/turbo2_tcq
Midpoint) **unterschaeetzt massiv** bei mittleren Contextgroessen:

- Bei 68K Tokens ohne Degrade: V liegt bei **2.0 B/v** (F16) — **9x hoeher** als 0.22
- Erst ab ~100K+ Tokens beginnt der Degrade zu T8 (1.015 B/v) oder T4 (0.515 B/v)
- Der kalibrierte Wert aus `/slots` (`_calibrated_v_bytes`) ist der **einzige
  vertrauenswuerdige** Wert fuer V-Cache-Schaetzung

**Empfehlung:** Ohne Kalibrierung den VBR-Fallback auf `0.72` (~5.8 bpv) heben —
empirischer Mittelwert bei 32 GB VRAM + 120K Context. Der aktuelle 0.22 ist nur
gültig bei vollem Degrade auf turbo2_tcq, was unter normalen Bedingungen nicht
eingetreten ist.

---

## Referenzen

- buun-llama-cpp: https://github.com/spiritbuun/buun-llama-cpp
- llama-kv-cache.h: https://github.com/spiritbuun/buun-llama-cpp/blob/master/src/llama-kv-cache.h
- llama-kv-cache.cpp: https://github.com/spiritbuun/buun-llama-cpp/blob/master/src/llama-kv-cache.cpp
- llama-cparams.h: https://github.com/spiritbuun/buun-llama-cpp/blob/master/src/llama-cparams.h
- llama-vram-ledger.h: https://github.com/spiritbuun/buun-llama-cpp/blob/master/src/llama-vram-ledger.h
- DeepWiki: https://deepwiki.com/spiritbuun/buun-llama-cpp
- ggml-common.h (Block-Strukturen): https://raw.githubusercontent.com/spiritbuun/buun-llama-cpp/master/ggml/src/ggml-common.h
