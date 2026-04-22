"""`python -m phoneme` entry point."""

from __future__ import annotations

import logging
import sys


def _probe_microphone() -> None:
    """Briefly open an input stream so macOS/TCC surfaces the mic permission
    prompt at launch, rather than silently when the pipeline later tries to
    record. On Linux/Windows this is a harmless no-op open+close."""
    try:
        import sounddevice as sd

        with sd.InputStream(channels=1, samplerate=16000, blocksize=256):
            pass
    except Exception as e:
        logging.getLogger("phoneme").warning(
            "mic permission probe failed (%s). The app will retry when the "
            "pipeline starts; if no prompt appears, grant Microphone access "
            "to your terminal/IDE in System Settings → Privacy & Security.",
            e,
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from .runtime import describe

    logging.getLogger("phoneme").info("runtime: %s", describe())

    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Phoneme")
    app.setOrganizationName("Phoneme")

    # Ask for mic permission before showing the UI so the OS dialog appears
    # immediately on first launch.
    _probe_microphone()

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
