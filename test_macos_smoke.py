import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys, tempfile, time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    with tempfile.TemporaryDirectory(prefix='llauncher-smoke-') as tmp, patch.object(Path, 'home', return_value=Path(tmp)):
        from PyQt6.QtWidgets import QApplication
        from i18n import I18nManager
        I18nManager.get_instance().load_language('de')
        from llauncher import llauncher
        from command_builder import get_full_args
        from process_runner import ProcessRunner
        app = QApplication([])
        window = llauncher()
        window.show()
        app.processEvents()
        args = get_full_args(window)
        import shutil
        assert args[0] == shutil.which('llama-server'), args
        assert '/dev/shm' not in ' '.join(args), args
        print('GUI constructed; native server resolved:', args[0])
        output, codes = [], []
        runner = ProcessRunner([args[0], '--version'], window.llama_cpp_path)
        runner.output_signal.connect(output.append)
        runner.finished_signal.connect(codes.append)
        runner.start()
        deadline = time.monotonic()+15
        while runner.isRunning() and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.02)
        assert runner.wait(1000)
        app.processEvents()
        assert codes == [0], (codes, output)
        print('Native ProcessRunner OK:', output)
        window.update_gpu_display({'slots': []})
        window.close()
        app.processEvents()
        print('GUI smoke OK')

if __name__ == "__main__":
    main()
