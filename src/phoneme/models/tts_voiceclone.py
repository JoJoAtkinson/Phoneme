"""Voice-clone TTS (optional).

We keep this as a thin interface with two backends users can opt into:
  * ``OpenVoiceV2`` — MIT license, voice cloning from a short reference clip.
  * ``XTTSv2``     — Coqui CPML license (non-commercial), via the `TTS` pkg.

Neither is a hard dependency. Install ``phoneme[voice-clone]`` to enable XTTS;
OpenVoice requires a manual install per their README. When these aren't
installed, ``get_voiceclone()`` returns None and the app falls back to Kokoro.
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
                "Coqui TTS not installed. Run: uv pip install 'phoneme[voice-clone]'"
            ) from e

        self._model = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.load()
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
