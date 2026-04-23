"""Kokoro TTS via kokoro-onnx — fast, Apache 2.0, no voice cloning.

For voice-cloned echo, see ``tts_voiceclone.py`` (optional).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..paths import models_dir

log = logging.getLogger(__name__)


class KokoroTTS:
    def __init__(self, voice: str = "af_sky", speed: float = 0.9):
        self.voice = voice
        self.speed = speed
        self._engine = None

    # kokoro-onnx expects a single bundled voices file; upstream publishes it
    # only as a GitHub release asset (the HuggingFace ONNX repo ships per-voice
    # .bin files instead, which aren't the format kokoro-onnx consumes).
    VOICES_URL = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin"
    )

    def load(self) -> None:
        if self._engine is not None:
            return
        import urllib.request

        from huggingface_hub import hf_hub_download
        from kokoro_onnx import Kokoro

        from ..runtime import onnx_providers

        cache = models_dir() / "kokoro-onnx"
        cache.mkdir(parents=True, exist_ok=True)
        model_path = Path(
            hf_hub_download(
                repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
                filename="onnx/model.onnx",
                local_dir=str(cache),
            )
        )
        voices_path = cache / "voices-v1.0.bin"
        if not voices_path.exists():
            log.info("downloading Kokoro voices bundle → %s", voices_path)
            urllib.request.urlretrieve(self.VOICES_URL, voices_path)

        # Prefer CoreML on Apple Silicon. kokoro-onnx passes extra kwargs
        # through to onnxruntime.InferenceSession, so 'providers=' works.
        providers = onnx_providers()
        log.info(
            "loading Kokoro: model=%s voices=%s providers=%s",
            model_path.name,
            voices_path.name,
            [p if isinstance(p, str) else p[0] for p in providers],
        )
        try:
            self._engine = Kokoro(str(model_path), str(voices_path), providers=providers)
        except TypeError:
            # Older kokoro-onnx without providers kwarg: fall back to CPU default.
            log.warning("kokoro-onnx does not accept providers=; using default EP")
            self._engine = Kokoro(str(model_path), str(voices_path))

        # kokoro-onnx 0.5 hardcodes speed=int32 on the "newer export" branch,
        # but the `onnx-community/Kokoro-82M-v1.0-ONNX` model declares speed
        # as float32 — so every call errors with InvalidArgument. Wrap the
        # session's run() to coerce the speed input on its way through.
        sess = self._engine.sess
        _raw_run = sess.run
        speed_dtype = None
        for inp in sess.get_inputs():
            if inp.name == "speed":
                speed_dtype = np.float32 if "float" in inp.type else np.int32
                break

        def _patched_run(output_names, input_feed, run_options=None):
            if speed_dtype is not None and "speed" in input_feed:
                arr = np.asarray(input_feed["speed"])
                if arr.dtype != speed_dtype:
                    input_feed = {**input_feed, "speed": arr.astype(speed_dtype)}
            return _raw_run(output_names, input_feed, run_options)

        sess.run = _patched_run

    def synthesize(self, text: str, language: str = "en-us") -> tuple[np.ndarray, int]:
        """Return (audio_float32_mono, sample_rate)."""
        self.load()
        assert self._engine is not None
        samples, sr = self._engine.create(
            text, voice=self.voice, speed=self.speed, lang=language
        )
        return np.asarray(samples, dtype=np.float32), int(sr)

    def list_voices(self) -> list[str]:
        self.load()
        assert self._engine is not None
        return sorted(self._engine.get_voices())
