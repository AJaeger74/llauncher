# llauncher – GUI for llama.cpp

A mixer-style launcher for controlling llama.cpp with presets, benchmarking, GPU monitoring, and multi-fork management.
![Llaunchy](llauncher.png)

## Features

◆ **Hugging Face Model Download**
Download GGUF models directly from Hugging Face Hub:
- Use full URL or author/reponame syntax for interactive selection
- Browse and search model repositories
- Progress bar with percentage display
- Cancel mid-download (keeping partial file for resuming later)
- Overwrite confirmation for existing local files
- Target directory preview before download starts

◆ **Fork Manager**
Clone, build, and switch between different llama.cpp forks without leaving the GUI:
- Enter a repo URL, pick a target directory, and start the clone
- Optionally build the fork after cloning (make/cmake)
- Switch the main app to use the new fork's binary
- Manage multiple forks side by side

◆ **Parameter Control** like on a mixing console
Each parameter has a slider (integer) or float slider with edit field:
- `-c` Context Size (dynamic maximum from GGUF metadata)
- `-n` Max Tokens
- `-t` CPU Threads
- `-b` Batch Size
- `-ngl` GPU Layers with "all" checkbox
- `--temp`, `--top-p`, `--repeat-penalty` (float sliders)
- `--cache-type-k` / `--cache-type-v` (combo boxes, dynamically parsed from `--help`)
- `--flash-attn` (combo: on/off)
- `--host` (text input)

◆ **Save & Load Presets**
Save current configuration via dialog, restore via dialog.
Saved to: `~/.llauncher/presets.json`

◆ **Benchmarking**
Run test inference and save results:
- **Standard mode**: full benchmark run with TPS (tokens per second) calculation
- **Streaming mode**: live token output while benchmark runs, with cancel support
- Free-text field for quality rating (1-5 or custom text)
- Saved to: `~/.llauncher/benchmarks.json`
- Automatic chat template application per model family (Gemma, Qwen, Llama, Mistral, Nemotron, Hermes, Phi) to prevent generation collapse

◆ **Model Selection + Path Configuration**
- llama.cpp directory + executable selection
- Dropdown with all `.gguf` files from model directory
- mmproj path for vision models (saved with presets)
- GGUF metadata inspection: context length, tensor count, model info via HTTP API
- Paths saved in `~/.llauncher/config.json`

◆ **Custom Commands**
Free-text field for external/unmanaged CLI parameters not covered by the built-in sliders:
- Enter arguments in `key value` format (one per line) or `key=value` format
- Bare flags (no value) are also supported
- Lines starting with `#` are treated as comments and ignored
- Loaded automatically when "Load Process Args" detects non-managed parameters
- Saved and restored with presets, with backward-compatibility for old presets
- Auto-strips rich text formatting on paste
- Live-updates the debug command line preview

◆ **Settings Dialog**
- Theme toggle (light / dark)
- Language selection (German / English)
- Model directory and llama.cpp path configuration

◆ **Internationalization (i18n)**
Full UI translation for German (`de`) and English (`en`).
Translation files in `locales/` (JSON-based, gettext-style).

◆ **Debug Output**
Full command line (1:1 as executed) and live output during runtime

◆ **GPU Monitoring (nvidia-smi)**
Live data in status area:
- GPU utilization (%)
- VRAM usage (MB)
- Temperature (°C)

◆ **Window State Persistence**
Window geometry, splitter positions, and UI state are saved across sessions.

◆ **KDE Plasma Look**
Dark theme with `#0078d7` accent color, adapts to system settings

◆ **Start/Stop Button**
- `Ready to go` (green)
- `Loading model...` (orange)
- `Loaded` (green)
- `Stopped` (red)
- `Failed` (red)

## Installation

Prerequisites:
```bash
pip install PyQt6 psutil PyMuPDF
```

For PDF benchmarking, PyMuPDF is required. Alternative: `pip install pdfplumber`

Arch Linux: `pacman -S python-pyqt6 python-psutil`

## Starting

```bash
python3 llauncher.py
```

## File Structure

```
~/llama.cpp/          # llama.cpp build directory
~/models/             # GGUF models
~/.llauncher/         # Configuration and presets
├── config.json       # Paths and last settings
├── presets.json      # Saved presets
└── benchmarks.json   # Benchmark results

./                      # Project directory
├── llauncher.py          # Main window (~1750 lines)
├── ui_builder.py         # Declarative UI construction
├── ui_helpers.py         # Text append, formatting utilities
├── ui_persistence.py     # Window geometry + splitter state
├── command_builder.py    # CLI arg assembly + param change signals
├── params.py             # Parameter definitions and types
├── gguf_utils.py         # GGUF parsing, CPU detection, model info
├── storage.py            # JSON I/O for config/presets/benchmarks
├── preset_manager.py     # Preset dialogs (save/load/benchmark rating)
├── gpu_monitor.py        # GPUMonitor QThread with nvidia-smi polling
├── process_runner.py     # ProcessRunner + terminate_by_pid()
├── process_inspector.py  # Runtime process arg introspection
├── process_signals.py    # GPU monitor startup + free VRAM queries
├── status_manager.py     # Status label updates + error handling
├── float_slider_sync.py  # DirectClickSlider + Float/Integer Slider Creation
├── help_parser.py        # Dynamic parameter extraction from llama-server --help
├── fork_manager.py       # Clone/build/switch llama.cpp forks
├── settings_dialog.py    # Theme, language, path settings
├── chat_templates.py     # Model-specific chat templates for benchmarks
├── http_benchmark_thread.py  # HTTP streaming benchmark QThread
├── benchmark_manager.py  # Benchmark lifecycle orchestration
├── hf_download_dialog.py # Hugging Face model download dialog
├── model_inspector.py    # GGUF metadata on model selection
├── model_info_fetcher.py # Running model info via HTTP API
├── i18n.py               # I18nManager + gettext system
├── i18n_util.py          # Language helpers (auto-detect, defaults)
├── locales/              # Translation JSON files (de.json, en.json)
└── README.md
```

## Example Configuration

For fast inference on consumer hardware:
- Threads: 8 (matches physical cores)
- GPU Layers: as many as VRAM allows (usually 30-45)
- Context Size: 2048 (or higher for long contexts)
- Max Tokens: 512 (standard), -1 for unlimited

For vision models:
1. Select model (.gguf with ViT support)
2. Enter mmproj path (e.g., `~/models/vision/mmproj-model-f16.gguf`)
3. Start

## Technical Details

- **GUI Framework**: PyQt6
- **GPU Monitoring**: nvidia-smi (NVIDIA-only)
- **Process Management**: QThread with multi-signal shutdown (SIGINT→SIGTERM→SIGKILL)
- **Styling**: Qt Style Sheet (QSS)
- **Modularized**: ~9200 lines total across 26 modules; `llauncher.py` acts as orchestrator (~1750 lines)
- **Signal Handling**: Custom `object` type signals to avoid PyQt6 32-bit int truncation on large value emissions (progress sizes)
- **i18n**: JSON-based translation system with lazy loading and runtime language switching
- **Chat Templates**: Automatic template formatting per model family to prevent generation collapse during benchmarks

## Extension Possibilities

- AMD GPU support (radeontop/virtuoso)
- Batch inference (multiple models in parallel)
- WebUI version with QtWebEngine
- More llama.cpp parameters (dynamically extracted from `--help`)
