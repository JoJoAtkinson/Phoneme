"""MLX-Whisper backend — Metal-native Whisper on Apple Silicon.

Typically 2-3× faster than faster-whisper (which is CPU-only via CTranslate2)
for the same model size. Only works on Apple Silicon. Falls back gracefully
when not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..runtime import has_mlx

log = logging.getLogger(__name__)


@dataclass
class WordToken:
    text: str
    start_s: float
    end_s: float
    confidence: float


class MLXWhisperASR:
    # Map generic Whisper sizes to mlx-community repos.
    SIZE_TO_REPO = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._loaded_repo: str | None = None

    def available(self) -> bool:
        if not has_mlx():
            return False
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        if self._loaded_repo is not None:
            return
        if not self.available():
            raise RuntimeError(
                "mlx-whisper is only supported on Apple Silicon. "
                "Install with: uv sync --extra mac"
            )
        self._loaded_repo = self.SIZE_TO_REPO.get(
            self.model_size, self.SIZE_TO_REPO["small"]
        )
        log.info("mlx-whisper will use %s", self._loaded_repo)

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en"
    ) -> list[WordToken]:
        self.load()
        assert self._loaded_repo is not None
        import mlx_whisper

        # mlx-whisper works on float32 @ 16 kHz.
        if sample_rate != 16000:
            raise ValueError("mlx-whisper requires 16 kHz audio")
        result = mlx_whisper.transcribe(
            audio.astype(np.float32, copy=False),
            path_or_hf_repo=self._loaded_repo,
            language=language,
            word_timestamps=True,
            verbose=False,
        )

        out: list[WordToken] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                out.append(
                    WordToken(
                        text=str(w.get("word", "")).strip(),
                        start_s=float(w.get("start", 0.0) or 0.0),
                        end_s=float(w.get("end", 0.0) or 0.0),
                        confidence=float(w.get("probability", 1.0) or 1.0),
                    )
                )
        return out
