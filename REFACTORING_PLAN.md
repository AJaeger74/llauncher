# llauncher Refactoring Plan 2026-04-15

## Ziel
llauncher.py von ~2072 Zeilen auf ~400 Zeilen reduzieren durch Modularisierung.

## Arbeitsregeln
- **Kein Code ausführen** - User führt Tests durch (`./llauncher` oder `python3 llauncher.py`)
- **Inkrementell vorgehen:** Nach jedem Modul-Create: User testet App-Start
- **Nur positive Feedback → Commit mit `git add` + `git commit -m`**
- **Fehler → Patch vor Commit**

---

## Modul-Struktur (Zielzustand)

### 1. params.py (~140 Zeilen)
```
PARAM_DEFINITIONS_BASE (dict)
get_param_definitions() classmethod → bleibt in llauncher.py als Factory
```

### 2. command_builder.py (~180 Zeilen)
```
get_current_args(window, params)
build_full_command(args, external_args)
on_param_changed(window)
```

### 3. benchmark_manager.py (~350 Zeilen)
```
BenchmarkManager class
├── run_benchmark_streaming()
├── run_benchmark()
├── cancel_benchmark()
├── on_benchmark_finished()
└── on_benchmark_token_update()
Helper: _get_model_info(), _get_running_server_command()
```

### 4. process_inspector.py (~180 Zeilen)
```
check_existing_process(window)
_get_running_server_command()
_helper model_is_already_loaded()
```

### 5. ui_persistence.py (~80 Zeilen)
```
restore_geometry(window)
save_window_state(window) → für resizeEvent + closeEvent
```

### 6. model_inspector.py (~150 Zeilen)
```
on_model_selected(window, model_name)
_get_model_info()
_format_file_size()
```

### 7. ui_helpers.py (~120 Zeilen)
```
browse_llama_dir(window)
browse_model_dir(window)
browse_path(line_edit, start_dir)
on_select_benchmark_file(line_edit)
on_clear_benchmark_file(line_edit)
_append_text_inline(text)
```

### 8. process_signals.py (~100 Zeilen)
```
setup_process_signals(window) → on_output() + on_process_finished()
start_gpu_monitor(window)
_get_free_gpu_memory()
```

### 9. status_manager.py (~60 Zeilen)
```
update_status(window, state)
handle_process_error(window, exit_code)
reset_progress_bar(window)
```

### 10. preset_handler.py (~150 Zeilen)
```
save_preset(window) wrapper
load_preset_dialog(window) wrapper
```

---

## Redundanzen zu entfernen

### Duplikate identifiziert:
1. `get_current_args()` und `build_full_command()`: Beide machen fast dasselbe → `build_full_command()` ruft `get_current_args()` auf
2. `_get_model_info()` und UI-Logik in `on_model_selected()`: Trennung nötig
3. Import von `QThread` und `pyqtSignal` in Zeile 41: Nicht verwendet (schon in process_runner/http_benchmark_thread)
4. Double-import `DirectClickSlider` (Zeilen 27 + 35)

### Duplikate bereinigen:
```python
# Zeile 35 entfernen: from float_slider_sync import DirectClickSlider (bereits Zeile 27)
# Zeile 1439-1451 _get_free_gpu_memory() → process_signals.py
# Zeile 1561-1568 _append_text_inline() → ui_helpers.py
```

---

## Geplante Reduktion

**Aktuell:** llauncher.py = 2072 Zeilen  
**Nach Refactoring:**
- llauncher.py: ~400 Zeilen (nur noch __init__ + Hauptlogik-Orchestrierung)
- Neue Module: ~1360 Zeilen (sauber getrennt)

**Gesamteinsparung:** llauncher.py von 2072 → ~400 Zeilen (80% Reduktion)

---

## Phasen-Plan

### Phase 1: params.py + command_builder.py (HEUTE)
- [ ] params.py erstellen mit PARAM_DEFINITIONS_BASE
- [ ] command_builder.py erstellen mit get_current_args() + build_full_command()
- [ ] llauncher.py umschreiben zu Importen
- [ ] **USER TESTS:** `python3 llauncher.py` → App startet? Parameter werden gelesen?
- [ ] Commit nur bei positivem Feedback

### Phase 2: benchmark_manager.py
- [ ] benchmark_manager.py erstellen mit run_benchmark_streaming() + run_benchmark()
- [ ] **USER TESTS:** Live-Benchmark funktioniert? Cancel Button?
- [ ] Commit nur bei positivem Feedback

### Phase 3: Prozess-Inspektion + UI-Persistence
- [ ] process_inspector.py + ui_persistence.py erstellen
- [ ] **USER TESTS:** Fenster-Geometrie wird gespeichert? Laufende Prozesse erkannt?
- [ ] Commit nur bei positivem Feedback

### Phase 4: Restliche Helper auslagern
- [ ] model_inspector.py + ui_helpers.py + process_signals.py + status_manager.py
- [ ] **USER TESTS:** Modell-Info, Browser-Dialoge, Status-Updates
- [ ] Commit bei vollständigem Refactoring

---

## Git-Workflow

```bash
# Nach jedem erfolgreichen Test:
git add params.py command_builder.py llauncher.py
git commit -m "refactor: extract params.py and command_builder.py (~320 lines)"

# Bei Live-Benchmark Fix:
git add benchmark_manager.py llauncher.py
git commit -m "fix: ensure live benchmark stability in benchmark_manager.py"
```

---

## Warnungen

- **Kein Python ausführen** - nur User mit DISPLAY kann App testen
- **Backup vor großen Changes:** `git stash` oder `git branch backup-phase-X`
- **Bei Syntaxfehlern:** `python3 -m py_compile datei.py` zur Verifikation nutzen
- **Bei import errors:** Prüfen ob alle Abhängigkeiten in llauncher.py Importen aktuell sind

---

## Status (2026-04-15 18:00)
- Branch: `refactorize` erstellt ✓
- TODOS.md gelesen ✓
- Plan erstellt: REFACTORING_PLAN.md ✓
- **Phase 1 abgeschlossen:**
  - ✓ params.py erstellt (176 Zeilen, PARAM_DEFINITIONS_BASE + get_param_definitions())
  - ✓ llauncher.py um ~150 Zeilen reduziert (2072 → ~1920)
  - ✓ App startet erfolgreich nach Extraction
  - ✓ Commit: `refactor: extract PARAM_DEFINITIONS to params.py (~150 lines removed)`
- **Phase 2 abgeschlossen:**
  - ✓ command_builder.py erstellt (260 Zeilen, get_current_args + build_full_command + on_param_changed)
  - ✓ Methoden aus llauncher.py entfernt, Wrapper hinzugefügt
  - ✓ Syntax-Fehler behoben: erranter Import mitten in Klasse entfernt
  - ✓ App startet erfolgreich, Parameter werden gelesen, Debug-Output funktioniert
  - ✓ Commit: `refactor: extract get_current_args/build_full_command to command_builder.py (~260 lines)`
- **Phase 3 abgeschlossen:**
  - ✓ benchmark_manager.py erstellt (134 Zeilen, run_benchmark_streaming + run_benchmark + cancel_benchmark)
  - ✓ UI-Handler in llauncher.py belassen: on_benchmark_output(), on_benchmark_token_update(), on_benchmark_finished()
  - ✓ App startet erfolgreich, Live-Benchmark funktioniert, Metriken werden korrekt angezeigt
  - ✓ Commit: `refactor: extract benchmark orchestration to benchmark_manager.py (~450 lines removed from llauncher.py)`
- **Phase 4 abgeschlossen:**
  - ✓ process_inspector.py erstellt (109 Zeilen, check_existing_process + get_running_server_command)
  - ✓ ui_persistence.py erstellt (115 Zeilen, restore_geometry + save_window_geometry + save_window_state)
  - ✓ App startet erfolgreich, Fenster-Geometrie wird gespeichert/geladen
  - ✓ Commit: `refactor: extract process inspection and UI persistence (~600 lines removed from llauncher.py)`
  - ✓ Fix: Entfernt redundante Module-Marker aus allen Wrapper-Methoden
- **Gesamteinsparung llauncher.py:** 2072 → ~1437 Zeilen (~31% Reduktion)
- **Nächster Schritt:** model_inspector.py + ui_helpers.py + process_signals.py + status_manager.py (Phase 5 - Rest)
