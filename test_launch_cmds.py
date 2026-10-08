#!/usr/bin/env python3
"""Smoke test: launch_cmd_start/finish sequentielle Ausführung (offscreen, echtes Start-Fluss).

Übt den REALEN toggle_process()-Pfad aus:
  config.json -> launch_cmd_start (vorher, blocking) -> get_full_args -> ProcessRunner
  -> on_output "listening on http://" -> launch_cmd_finish -> fertig

Fälle:
  T2  Normalfall: Reihenfolge PRE → (server läuft) → POST, server lief.
  T3  pre mit exit 1 → Warnung, server startet trotzdem, POST läuft danach.
  T4  manual=False (Crash-Restart) → keine Launch-Commands.
  T5  Preset mit skip_launch_cmds → keine Launch-Commands (manueller Start).

Kein echtes Modell/Binary nötig: Dummy-Binary + monkeygepatchte Arch-Checks.
"""
import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))

WORK = Path(tempfile.mkdtemp(prefix="llauncher_test_"))
MARKER = WORK / "order.txt"


def mark(name):
    with open(MARKER, "a") as f:
        f.write(f"{name}\n")


def reset_marker():
    MARKER.write_text("")


def read_marker():
    return MARKER.read_text().splitlines()


def main():
    # ── Setup: Dummy-Binary am exakten Ort, den get_current_args erwartet ──
    bin_dir = WORK / "llama.cpp" / "build" / "bin"
    bin_dir.mkdir(parents=True)
    dummy = bin_dir / "llama-server"
    dummy.write_text(textwrap.dedent("""\
        #!/bin/sh
        echo "llama.cpp version 0.0.0"
        echo "load_model: loading weights from dummy..."
        echo "compute_type: CPU"
        echo "load_model: model loaded in 0.5s"
        echo "srv  listening on http://127.0.0.1:8080"
        # auf SIGINT/SIGTERM warten, dann clean exit 0
        trap 'exit 0' INT TERM
        while :; do sleep 0.2; done
    """))
    dummy.chmod(0o755)

    PRE = WORK / "pre.sh"; POST = WORK / "post.sh"; PRE_FAIL = WORK / "pre_fail.sh"
    # Marker werden BEI AUFAHRT des Scripts geschrieben (nicht bei Anlage!)
    PRE.write_text(f"#!/bin/sh\necho pre-running\necho PRE >> {MARKER}\nexit 0\n")
    POST.write_text(f"#!/bin/sh\necho post-running\necho POST >> {MARKER}\nexit 0\n")
    PRE_FAIL.write_text(f"#!/bin/sh\necho pre-failing\necho PRE_FAIL >> {MARKER}\nexit 1\n")
    for s in (PRE, POST, PRE_FAIL):
        s.chmod(0o755)

    cfg_file = Path.home() / ".llauncher" / "config.json"
    backup = cfg_file.read_text() if cfg_file.exists() else None

    def restore_config():
        """config.json auf den Zustand vor dem Test zurücksetzen."""
        try:
            if backup is not None:
                cfg_file.write_text(backup)
            elif cfg_file.exists():
                c = json.loads(cfg_file.read_text())
                for k in ("launch_cmd_start", "launch_cmd_finish"):
                    c.pop(k, None)
                cfg_file.write_text(json.dumps(c, indent=2))
        except Exception:
            pass

    # SIGINT/SIGTERM während eines modalen Dialogs → finally läuft nicht,
    # der Handler stellt die Config trotzdem wieder her.
    def _sig_restore(signum, frame):
        restore_config()
        sys.exit(128 + signum)
    import signal
    signal.signal(signal.SIGINT, _sig_restore)
    signal.signal(signal.SIGTERM, _sig_restore)

    def set_cfg(start, finish):
        cfg = json.loads(backup) if backup else {}
        cfg["launch_cmd_start"] = start
        cfg["launch_cmd_finish"] = finish
        cfg_file.write_text(json.dumps(cfg, indent=2))

    from PyQt6.QtWidgets import QApplication, QDialog
    app = QApplication(sys.argv)

    # i18n sauber laden (Test-Artefakt sonst: leeres Dict → "missing in de" für alle Keys)
    from i18n import I18nManager
    I18nManager.get_instance().load_language("en")

    # ── T1: SettingsDialog Sektion + 4-Tupel ─────────────────────────────
    from settings_dialog import SettingsDialog
    dlg = SettingsDialog(None, False, "en",
                         launch_cmd_start="a b", launch_cmd_finish="c d")
    assert dlg.cmd_start_edit.text() == "a b"
    assert dlg.cmd_finish_edit.text() == "c d"
    light, lang, cs, cf = dlg.get_settings()
    assert (cs, cf) == ("a b", "c d"), (cs, cf)
    print("T1 OK: SettingsDialog Sektion + get_settings 4-Tupel")

    # ── Window (echter Build) + Arch-Checks wegpatchen ───────────────────
    import llauncher as llmod
    win = llmod.llauncher()
    win.show()

    import gguf_utils
    gguf_utils.get_model_info = lambda *a, **k: {"arch": "llama"}
    gguf_utils.check_model_architecture = lambda *a, **k: None
    # (Dummy-Binary exitet auf SIGINT mit 0 → kein Crash-Dialog beim Stoppen.)

    win.llama_cpp_path = str(WORK / "llama.cpp")
    win.exe_line.setText(win.llama_cpp_path)
    win.exe_combo.blockSignals(True)
    win.exe_combo.addItem("llama-server")
    win.exe_combo.setCurrentText("llama-server")  # Placeholder ("...nicht gefunden") überspringen
    win.exe_combo.blockSignals(False)
    win.selected_model = "dummy-llama.gguf"

    # Frischen Start-Zustand erzwingen: beim Window-Init kann
    # check_existing_process() einen realen llama-server adoptiert haben
    # (external_runner_pid). Der Test will IMMER den internen Start-Pfad.
    win.runner = None
    win.external_runner_pid = None
    win.external_runner_args = None

    def wait_runner_idle(timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.05)
            if not (win.runner and win.runner.isRunning()):
                break

    def stop_and_cleanup():
        if win.runner and win.runner.isRunning():
            win.runner.terminate_process()
            t0 = time.time()
            while win.runner.isRunning() and time.time() - t0 < 5:
                app.processEvents(); time.sleep(0.05)
        win.runner = None
        app.processEvents()

    def fresh_start():
        """Frischen Start-Zustand erzwingen, OHNE dass der 1s-Timer
        dazwischen einen Prozess neu adoptieren kann."""
        win.process_check_timer.stop()
        app.processEvents()
        win.runner = None
        win.external_runner_pid = None
        win.external_runner_args = None

    try:
        # ── T2: Normalfall — PRE vor Prozess, POST nach load ─────────────
        fresh_start()
        reset_marker()
        set_cfg(str(PRE), str(POST))
        win.toggle_process(manual=True)
        # finish feuert asynchron im Runner-Thread; warten bis POST da ist
        t0 = time.time()
        while "POST" not in read_marker() and time.time() - t0 < 15:
            app.processEvents(); time.sleep(0.05)
        seq = read_marker()
        assert seq == ["PRE", "POST"], f"Reihenfolge falsch: {seq}"
        # finish darf erst NACH dem "listening on"-Output erscheinen (Bug-Regression)
        dt = win.debug_text.toPlainText()
        pos_listen = dt.find("listening on http://")
        pos_finish = dt.find("[launch_cmd:finish]")
        assert pos_listen != -1 and pos_finish != -1, (pos_listen, pos_finish)
        assert pos_finish > pos_listen, f"finish vor load-Signal: listen@{pos_listen}, finish@{pos_finish}"
        print("T2 OK:", seq, "(finish nach load-Signal)")
        stop_and_cleanup()

        # ── T3: pre exit 1 → Warnung, Modell startet trotzdem, finish läuft ─
        fresh_start()
        reset_marker()
        set_cfg(str(PRE_FAIL), str(POST))
        win.toggle_process(manual=True)
        t0 = time.time()
        while "POST" not in read_marker() and time.time() - t0 < 15:
            app.processEvents(); time.sleep(0.05)
        seq = read_marker()
        assert seq == ["PRE_FAIL", "POST"], f"Reihenfolge falsch: {seq}"
        debug_text = win.debug_text.toPlainText()
        assert "launch_cmd_start" in debug_text and "1" in debug_text
        print(f"T3 OK: pre exit 1 → Warnung, Start trotzdem: {seq}")
        stop_and_cleanup()

        # ── T4: manual=False → keine Launch-Commands ─────────────────────
        fresh_start()
        reset_marker()
        set_cfg(str(PRE), str(POST))
        win.toggle_process(manual=False)
        t0 = time.time()
        while win.runner and win.runner.isRunning() and time.time() - t0 < 15:
            app.processEvents(); time.sleep(0.05)
        app.processEvents()
        seq = read_marker()
        assert seq == [], f"Skripte liefen trotz manual=False: {seq}"
        print("T4 OK: manual=False → keine Launch-Commands")
        stop_and_cleanup()

        # ── T5: Preset mit skip_launch_cmds → keine Launch-Commands ───────
        fresh_start()
        reset_marker()
        set_cfg(str(PRE), str(POST))
        win._current_preset = {"skip_launch_cmds": True}
        win.toggle_process(manual=True)
        # Warten bis das Load-Signal im Debug-Output steht (finish hätte
        # asynchron danach feuern können) + kurze Nachlaufzeit.
        t0 = time.time()
        while "listening on http://" not in win.debug_text.toPlainText() \
                and time.time() - t0 < 15:
            app.processEvents(); time.sleep(0.05)
        for _ in range(40):
            app.processEvents(); time.sleep(0.05)
        seq = read_marker()
        assert seq == [], f"Skripte liefen trotz skip_launch_cmds: {seq}"
        dt = win.debug_text.toPlainText()
        assert "skip_launch_cmds" in dt, "Skip-Info im Debug-Output fehlt"
        print("T5 OK: skip_launch_cmds → keine Launch-Commands, Info geloggt")
        win._current_preset = None
        stop_and_cleanup()

        # Intentional stop must not open the crash/restart dialog after SIGKILL.
        dummy.write_text("#!/bin/sh\ntrap '' INT TERM\necho 'srv listening on http://127.0.0.1:8080'\nwhile :; do sleep 0.2; done\n")
        fresh_start()
        set_cfg("", "")
        win.toggle_process(manual=True)
        deadline = time.monotonic() + 5
        while not win.runner.get_pid() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.05)
        time.sleep(0.2)  # Allow the child's signal traps to be installed.
        from unittest.mock import patch
        with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected) as dialog_exec:
            win.toggle_process()
            app.processEvents()
            assert win.runner is None, "Stop did not release the finished runner"
            dialog_exec.assert_not_called()
        print("T6 OK: intentional forced stop does not restart")

        print("\nALLE TESTS OK")
    finally:
        restore_config()
        win.close()
        app.processEvents()


if __name__ == "__main__":
    main()
