"""Build a distributable Phoneme.app bundle on macOS via py2app.

Model weights are *not* bundled — the first launch lazily downloads to
``~/Library/Application Support/Phoneme/models/``. Bundling them would push
the .app past 2 GB which is painful to share. The first-run download fetches
roughly 800 MB (phoneme model + whisper-small + kokoro).

Usage:
    # from repo root, on Mac:
    uv sync --extra mac --extra package
    uv run python scripts/build_mac_app.py py2app

The .app lands in ./dist/Phoneme.app — share as a zip/dmg. Users will need to
right-click → Open the first time (Gatekeeper), or you codesign + notarize
with an Apple Developer cert.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup  # type: ignore

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "src"

if __name__ != "__main__":
    raise RuntimeError("run as a script")

if sys.platform != "darwin":
    raise SystemExit("py2app only runs on macOS. Use `uv build` for wheels on other platforms.")

sys.path.insert(0, str(SRC))

APP = [str(SRC / "phoneme" / "__main__.py")]

OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "phoneme",
        "PySide6",
        "shiboken6",
        "huggingface_hub",
        "transformers",
        "tokenizers",
        "sounddevice",
        "soundfile",
        "numpy",
        "scipy",
        "torch",
        "silero_vad",
        "onnxruntime",
        "faster_whisper",
        "kokoro_onnx",
    ],
    "includes": [
        "phoneme.ui.main_window",
        "phoneme.pipeline.streaming",
        "phoneme.pipeline.utterance",
        "phoneme.pipeline.aligned",
        "phoneme.pipeline.compress",
    ],
    "excludes": [
        "PyQt5", "PyQt6", "PySide2",
        "tkinter", "matplotlib.tests", "pytest",
    ],
    "plist": {
        "CFBundleName": "Phoneme",
        "CFBundleDisplayName": "Phoneme",
        "CFBundleIdentifier": "com.phoneme.app",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
        "NSMicrophoneUsageDescription": (
            "Phoneme uses the microphone to hear what you say and show the IPA "
            "symbols for each sound."
        ),
        "NSHighResolutionCapable": True,
    },
    "arch": "universal2",
}

setup(
    app=APP,
    name="Phoneme",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
