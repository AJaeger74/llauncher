#!/usr/bin/env python3
"""
params.py - Parameter definitions for llauncher

Zentralisierte Definition aller verwaltbaren llama.cpp Parameter
mit Slidern, Comboboxen, Text-Feldern.

Wird von llauncher.py importiert und dynamisch erweitert über get_param_definitions().
"""

# Dynamische Parameter definieren
PARAM_DEFINITIONS_BASE = {
    "-c": {
        "label_key": "param_context_size",
        "type": "slider",
        "min": 2048,
        "max": 8192,
        "default": 4096,
        "tooltip_key": "tooltip_context_size",
    },
    "--cache-type-k": {
        "label_key": "help_parser_k_type",
        "type": "combo",
        "default": "f16",
        "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "tooltip_key": "tooltip_cache_type_k",
    },
    "--cache-type-v": {
        "label_key": "help_parser_v_type",
        "type": "combo",
        "default": "f16",
        "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "tooltip_key": "tooltip_cache_type_v",
    },
    "-n": {
        "label_key": "param_max_tokens",
        "type": "slider",
        "min": -1,
        "max": 8192,
        "default": 4096,
        "tooltip_key": "tooltip_max_tokens",
    },
    "-np": {
        "label_key": "param_parallel_slots",
        "type": "slider",
        "min": -1,
        "max": 8,
        "default": -1,
        "tooltip_key": "tooltip_np",
    },
    "-t": {
        "label_key": "param_cpu_threads",
        "type": "slider",
        "min": 1,
        "max": 32,
        "default": 8,
        "tooltip_key": "tooltip_threads",
    },
    "-b": {
        "label_key": "param_batch_size",
        "type": "slider",
        "min": 1,
        "max": 8192,
        "default": 2048,
        "tooltip_key": "tooltip_batch_size",
    },
    "-ngl": {
        "label_key": "param_gpu_layers",
        "type": "slider",
        "min": 0,
        "max": "{{GPU_LAYERS}}",
        "default": 35,
        "tooltip_key": "tooltip_gpu_layers",
    },
    "--temp": {
        "label_key": "param_temperature",
        "type": "float_slider",
        "min": 0.1,
        "max": 2.0,
        "default": 0.8,
        "step": 0.1,
        "tooltip_key": "tooltip_temperature",
    },
    "--top-p": {
        "label_key": "param_top_p",
        "type": "float_slider",
        "min": 0.1,
        "max": 1.0,
        "default": 0.95,
        "step": 0.05,
        "tooltip_key": "tooltip_top_p",
    },
    "--top-k": {
        "label_key": "param_top_k",
        "type": "slider",
        "min": 0,
        "max": 1000,
        "default": 40,
        "tooltip_key": "tooltip_top_k",
    },
    "--min-p": {
        "label_key": "param_min_p",
        "type": "float_slider",
        "min": 0.0,
        "max": 0.5,
        "default": 0.05,
        "step": 0.01,
        "tooltip_key": "tooltip_min_p",
    },
    "--repeat-penalty": {
        "label_key": "param_repeat_penalty",
        "type": "float_slider",
        "min": 0.5,
        "max": 2.5,
        "default": 1.0,
        "step": 0.05,
        "tooltip_key": "tooltip_repeat_penalty",
    },
    "--flash-attn": {
        "label_key": "param_flash_attn",
        "type": "combo",
        "default": "off",
        "options": ["off", "on"],
        "tooltip_key": "tooltip_flash_attn",
    },
    "--host": {
        "label_key": "param_host",
        "type": "text_input",
        "default": "localhost",
        "tooltip_key": "tooltip_host",
    },
    "--slot-save-path": {
        "label_key": "param_slot_save_path",
        "type": "path_input",
        "default": "/dev/shm/llama-slots",
        "tooltip_key": "tooltip_slot_save_path",
    },
    "benchmark_file_path": {
        "label_key": "param_benchmark_file",
        "type": "file_input",
        "default": "",
        "tooltip_key": "tooltip_benchmark_file",
    },
}


def get_param_definitions():
    """PARAM_DEFINITIONS mit dynamischen Werten erstellen.
    
    Ersetzt Platzhalter wie {{CPU_COUNT}} und {{GPU_LAYERS}} durch
    tatsächliche Werte aus System-Info.
    
    Returns:
        dict: Vollständige Parameter-Definitionen
    """
    import copy
    from gguf_utils import get_cpu_count
    
    definitions = copy.deepcopy(PARAM_DEFINITIONS_BASE)
    
    # CPU Count ersetzen
    cpu_count = get_cpu_count()
    for key, value in definitions.items():
        if isinstance(value, dict) and "max" in value:
            max_val = value["max"]
            if isinstance(max_val, str):
                if "{{CPU_COUNT}}" in max_val:
                    value["max"] = cpu_count
                elif "{{GPU_LAYERS}}" in max_val:
                    # GPU Layers als 100 lassen (oder später dynamisch)
                    value["max"] = 100
    
    return definitions
