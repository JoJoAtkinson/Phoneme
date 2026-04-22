"""Voice-clone TTS (optional, not wired into the UI).

Scaffold for two potential backends:
  * ``OpenVoiceV2`` — MIT license, voice cloning from a short reference clip.
  * ``XTTSv2``     — Coqui CPML license (non-commercial), via the `TTS` pkg.

Coqui's ``TTS`` transitively pins ``numpy<2`` (via gruut==2.2.3), which is
incompatible with this project's core deps (``kokoro-onnx`` needs numpy>=2)
and with the Mac MLX stack (``parakeet-mlx`` needs numpy>=2.2.5). So ``TTS``
is NOT declared as an extra on this project.

To experiment with XTTS without breaking the main env, install it in an
isolated tool environment, e.g.::

    uv tool install TTS
    # or: pipx install TTS

and have the main app shell out to the CLI, or run this module from a
separate venv. ``get_voiceclone()`` returns None when TTS isn't importable,
and the app falls back to Kokoro.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class XTTSV2:
    """Coqui XTTS-v2 voice cloning. CPML (non-commercial)."""

    def __init__(self, reference_wav: Path, language: str = "en"):
        self.reference_wav = Path(reference_wav)
        self.language = language
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from TTS.api import TTS as CoquiTTS  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Coqui TTS not installed in this environment. "
                "Install it in an isolated tool env (e.g. `uv tool install TTS` "
                "or `pipx install TTS`); it cannot coexist with the main app's "
                "numpy>=2 stack."
            ) from e

        self._model = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.load()
        assert self._model is not None
        samples = self._model.tts(
            text=text,
            speaker_wav=str(self.reference_wav),
            language=self.language,
        )
        arr = np.asarray(samples, dtype=np.float32)
        return arr, 24000


def get_voiceclone(reference_wav: Path | None) -> XTTSV2 | None:
    """Return a cloning TTS engine if reference audio + deps are available."""
    if reference_wav is None or not reference_wav.exists():
        return None
    try:
        return XTTSV2(reference_wav)
    except Exception as e:
        log.warning("voice clone unavailable: %s", e)
        return None
