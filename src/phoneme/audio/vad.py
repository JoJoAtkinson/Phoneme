"""Silero VAD streaming wrapper.

Feed 16 kHz float32 audio in 512-sample chunks (32 ms). Get speech probability.
Fires ``on_speech_start`` and ``on_speech_end`` callbacks with utterance audio.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)


class StreamingVAD:
    CHUNK = 512  # Silero expects 512 samples @ 16 kHz

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_ms: int = 600,
        min_speech_ms: int = 80,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[np.ndarray], None] | None = None,
        on_prob: Callable[[float], None] | None = None,
    ):
        if sample_rate != 16000:
            raise ValueError("Silero VAD requires 16 kHz audio")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_prob = on_prob

        self._model = None
        self._in_speech = False
        self._silence_run = 0
        self._speech_run = 0
        self._utterance_buffer: list[np.ndarray] = []
        self._pending: np.ndarray = np.zeros(0, dtype=np.float32)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        from ..runtime import onnx_providers

        # silero-vad >= 5 exposes `load_silero_vad`. We patch the ONNX session
        # afterwards to use CoreML on Apple Silicon (gives roughly 3-4x
        # speedup over CPU EP and keeps the VAD loop under 1 ms per chunk).
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=True)

        providers = onnx_providers()
        if providers != ["CPUExecutionProvider"]:
            # Silero wraps an onnxruntime.InferenceSession. Reach in, swap
            # providers, fall back silently if the silero internals changed.
            try:
                import onnxruntime as ort

                path = None
                for attr in ("model", "session", "_ort_session", "ort_session"):
                    obj = getattr(self._model, attr, None)
                    if obj is not None:
                        path = getattr(obj, "_model_path", None) or getattr(
                            obj, "model_path", None
                        )
                        if path:
                            break
                if path:
                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    new_sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
                    # Replace silero's internal session. The attribute name
                    # varies across versions, so swap any that exist.
                    for attr in ("session", "_ort_session", "ort_session", "model"):
                        if hasattr(self._model, attr):
                            setattr(self._model, attr, new_sess)
                            break
                    log.info("Silero VAD accelerated via providers=%s", providers)
            except Exception as e:
                log.debug("CoreML acceleration for VAD unavailable: %s", e)
        log.info("Silero VAD loaded (onnx)")

    def reset(self) -> None:
        self._in_speech = False
        self._silence_run = 0
        self._speech_run = 0
        self._utterance_buffer = []
        self._pending = np.zeros(0, dtype=np.float32)
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def feed(self, samples: np.ndarray) -> None:
        """Feed an arbitrary-length slice of float32 audio @ 16 kHz."""
        self._lazy_load()
        assert self._model is not None
        import torch

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._pending = np.concatenate([self._pending, samples])

        while len(self._pending) >= self.CHUNK:
            chunk = self._pending[: self.CHUNK]
            self._pending = self._pending[self.CHUNK :]
            with torch.no_grad():
                prob = float(self._model(torch.from_numpy(chunk), self.sample_rate).item())
            if self.on_prob is not None:
                self.on_prob(prob)
            self._handle_chunk(chunk, prob)

    def _handle_chunk(self, chunk: np.ndarray, prob: float) -> None:
        is_speech = prob >= self.threshold

        if self._in_speech:
            self._utterance_buffer.append(chunk)
            if is_speech:
                self._silence_run = 0
                self._speech_run += self.CHUNK
            else:
                self._silence_run += self.CHUNK
                if self._silence_run >= self.min_silence_samples:
                    if self._speech_run >= self.min_speech_samples:
                        full = np.concatenate(self._utterance_buffer)
                        # Trim trailing silence a bit, keep 100 ms padding
                        tail = min(len(full), self._silence_run - int(self.sample_rate * 0.1))
                        if tail > 0:
                            full = full[:-tail] if tail < len(full) else full
                        if self.on_speech_end is not None:
                            self.on_speech_end(full)
                    self._in_speech = False
                    self._utterance_buffer = []
                    self._silence_run = 0
                    self._speech_run = 0
        else:
            if is_speech:
                self._in_speech = True
                self._utterance_buffer = [chunk]
                self._silence_run = 0
                self._speech_run = self.CHUNK
                if self.on_speech_start is not None:
                    self.on_speech_start()

    def force_end(self) -> None:
        """Manually end the current utterance (e.g. user released hold-to-talk)."""
        if self._in_speech and self._utterance_buffer:
            full = np.concatenate(self._utterance_buffer)
            if len(full) >= self.min_speech_samples and self.on_speech_end is not None:
                self.on_speech_end(full)
        self.reset()
