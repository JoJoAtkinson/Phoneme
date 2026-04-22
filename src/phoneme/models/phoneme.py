"""Wav2Vec2 phoneme recognizer (IPA output).

Supports two usage patterns:
  * ``transcribe(audio)`` — one-shot full utterance
  * ``stream_step(audio_window)`` — streaming, returns newly stable phonemes
    using CTC argmax on the latest window and comparing against last call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..runtime import torch_device
from .downloader import ensure_downloaded

log = logging.getLogger(__name__)


@dataclass
class PhonemeToken:
    ipa: str
    start_s: float
    end_s: float
    confidence: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Wav2Vec2PhonemeRecognizer:
    def __init__(
        self,
        model_key: str = "phoneme/wav2vec2-espeak",
        device: str | None = None,
    ):
        self.model_key = model_key
        self.device = device
        self._model = None
        self._processor = None
        self._frame_hop_s: float = 0.02  # Wav2Vec2 stride is 20 ms
        self._last_emitted: list[PhonemeToken] = []

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        local = ensure_downloaded(self.model_key)
        log.info("loading phoneme model from %s", local)
        self._processor = AutoProcessor.from_pretrained(local)
        self._model = AutoModelForCTC.from_pretrained(local)

        if self.device is None:
            self.device = torch_device()
        # MPS inference on CTC logits works fine in float32; half precision
        # saves a little memory but current PyTorch MPS has flaky softmax
        # reductions in fp16, so we stay fp32.
        self._model.to(self.device)
        self._model.eval()
        # Warm the compute graph with a tiny dummy pass to hide first-frame
        # latency (MPS compiles kernels on first execution).
        try:
            with torch.no_grad():
                warm = torch.zeros(1, 1600, device=self.device)
                self._model(warm)
        except Exception as e:
            log.debug("phoneme warmup skipped: %s", e)
        log.info("phoneme model on %s", self.device)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_confidence: float = 0.35,
        merge_repeats: bool = True,
    ) -> list[PhonemeToken]:
        """Return IPA phoneme tokens with per-token timestamps."""
        import torch

        self.load()
        if audio.ndim > 1:
            audio = audio[:, 0]
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self._model(input_values).logits[0]  # [T, V]
        probs = torch.softmax(logits, dim=-1)
        conf, ids = probs.max(dim=-1)
        ids = ids.cpu().numpy()
        conf = conf.cpu().numpy()

        blank = self._model.config.pad_token_id
        vocab = self._processor.tokenizer.get_vocab()
        id_to_tok = {v: k for k, v in vocab.items()}

        tokens: list[PhonemeToken] = []
        cur_id = None
        cur_start = 0
        cur_confs: list[float] = []
        for t, tok_id in enumerate(ids):
            if tok_id != cur_id:
                if cur_id is not None and cur_id != blank:
                    ipa = id_to_tok.get(int(cur_id), "")
                    if ipa and ipa not in {"|", "<pad>", "<s>", "</s>", "<unk>"}:
                        start = cur_start * self._frame_hop_s
                        end = t * self._frame_hop_s
                        c = float(np.mean(cur_confs)) if cur_confs else 0.0
                        if c >= min_confidence:
                            tokens.append(PhonemeToken(ipa, start, end, c))
                cur_id = int(tok_id)
                cur_start = t
                cur_confs = [float(conf[t])]
            else:
                cur_confs.append(float(conf[t]))
        # flush
        if cur_id is not None and cur_id != blank:
            ipa = id_to_tok.get(int(cur_id), "")
            if ipa and ipa not in {"|", "<pad>", "<s>", "</s>", "<unk>"}:
                start = cur_start * self._frame_hop_s
                end = len(ids) * self._frame_hop_s
                c = float(np.mean(cur_confs)) if cur_confs else 0.0
                if c >= min_confidence:
                    tokens.append(PhonemeToken(ipa, start, end, c))

        if merge_repeats:
            tokens = _merge_repeats(tokens)
        return tokens

    def stream_step(
        self,
        window_audio: np.ndarray,
        sample_rate: int = 16000,
        min_confidence: float = 0.35,
        stable_tail_s: float = 0.12,
    ) -> list[PhonemeToken]:
        """Transcribe a sliding window; return only newly-stable tokens.

        We treat phonemes whose end is older than ``stable_tail_s`` as stable.
        Duplicates against the last emission are filtered.
        """
        all_tokens = self.transcribe(
            window_audio, sample_rate, min_confidence, merge_repeats=True
        )
        window_len_s = len(window_audio) / sample_rate
        stable = [t for t in all_tokens if (window_len_s - t.end_s) >= stable_tail_s]

        # Emit only tokens we haven't emitted on this window cycle.
        # Caller is responsible for resetting state between utterances via ``reset``.
        new = stable[len(self._last_emitted) :]
        self._last_emitted = stable
        return new

    def reset(self) -> None:
        self._last_emitted = []


def _merge_repeats(tokens: list[PhonemeToken]) -> list[PhonemeToken]:
    """Collapse immediate repeats into one token with averaged confidence."""
    out: list[PhonemeToken] = []
    for t in tokens:
        if out and out[-1].ipa == t.ipa:
            prev = out[-1]
            new_conf = (prev.confidence + t.confidence) / 2
            out[-1] = PhonemeToken(prev.ipa, prev.start_s, t.end_s, new_conf)
        else:
            out.append(t)
    return out
