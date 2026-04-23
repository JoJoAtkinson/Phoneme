"""`python -m phoneme` entry point."""

from __future__ import annotations

import logging
import sys


def _probe_microphone() -> None:
    """Force macOS/TCC to surface the mic permission prompt at launch.

    Just opening an InputStream isn't enough — PortAudio's Audio Unit
    doesn't request samples from CoreAudio until something consumes data,
    so TCC stays quiet until the real pipeline's callback fires later.
    We install a real callback and actively read one buffer, which is what
    actually triggers the prompt.

    Runs in a daemon thread so a denial or delayed user response doesn't
    block the Qt startup path. No-op / harmless open+close on other OSes."""
    import threading

    def _probe():
        try:
            import sounddevice as sd

            stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                blocksize=256,
                dtype="float32",
                callback=lambda *_args: None,  # real callback → real input I/O
            )
            stream.start()
            # Hold the stream open briefly so CoreAudio actually delivers
            # a sample buffer (which is what fires the TCC prompt).
            import time

            time.sleep(0.2)
            stream.stop()
            stream.close()
        except Exception as e:
            logging.getLogger("phoneme").warning(
                "mic permission probe failed (%s). If no prompt appears, "
                "grant Microphone access to your terminal/IDE in System "
                "Settings → Privacy & Security.",
                e,
            )

    threading.Thread(target=_probe, name="mic-probe", daemon=True).start()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    from .runtime import describe

    logging.getLogger("phoneme").info("runtime: %s", describe())

    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    # Kick off the mic probe before QApplication so CoreAudio has a head
    # start — by the time the window paints, TCC should already be asking.
    _probe_microphone()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Phoneme")
    app.setOrganizationName("Phoneme")

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
