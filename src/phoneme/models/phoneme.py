"""Wav2Vec2 phoneme recognizer (IPA output).

Supports two usage patterns:
  * ``transcribe(audio)`` — one-shot full utterance
  * ``stream_step(audio_window)`` — streaming, returns newly stable phonemes
    using CTC argmax on the latest window and comparing against last call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from ..runtime import torch_device
from .downloader import ensure_downloaded

log = logging.getLogger(__name__)

# Non-phoneme vocab entries across the wav2vec2 phoneme models we support.
# Different checkpoints use different conventions (HF-style <pad>/<unk>,
# BERT-style [PAD]/[UNK], "|" for word boundary, prosody marks from espeak-
# derived tokenizers). We drop these so the chip strip only shows actual
# phonemes — stress markers aren't sounds, they're metadata on the next vowel.
_NON_PHONEME_TOKENS = {
    "|",
    "<pad>", "<s>", "</s>", "<unk>",
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "ˈ", "ˌ",  # primary / secondary stress
}


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
        self._feature_extractor = None
        self._id_to_tok: dict[int, str] = {}
        self._frame_hop_s: float = 0.02  # Wav2Vec2 stride is 20 ms
        self._last_emitted: list[PhonemeToken] = []

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

        local = ensure_downloaded(self.model_key)
        if local is None:
            raise RuntimeError(
                f"phoneme model '{self.model_key}' unavailable on this platform"
            )
        log.info("loading phoneme model from %s", local)
        # Deliberately skip AutoProcessor: Wav2Vec2PhonemeCTCTokenizer's
        # __init__ calls into phonemizer/espeak (a system binary we don't
        # need at inference time — only for training-time G2P). We read
        # the vocab directly and feed audio through the feature extractor.
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(local)
        with (local / "vocab.json").open() as f:
            vocab = json.load(f)
        self._id_to_tok = {int(v): k for k, v in vocab.items()}
        self._model = AutoModelForCTC.from_pretrained(local)

        if self.device is None:
            self.device = torch_device()
        # MPS inference on CTC logits works fine in float32; half precision
        # saves a little memory but current PyTorch MPS has flaky softmax
        # reductions in fp16, so we stay fp32.
        self._model.to(self.device)
        self._model.eval()
        # Warm the compute graph with a dummy pass at the real streaming
        # window size (720 ms @ 16 kHz = 11520 samples). MPS compiles a new
        # Metal kernel per distinct input shape, so warming at a mismatched
        # size would leave first-real-frame latency elevated.
        try:
            with torch.no_grad():
                warm = torch.zeros(1, 11520, device=self.device)
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
        assert self._model is not None and self._feature_extractor is not None
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Pad short utterances with silence on both sides so the model has
        # acoustic context around the word. Wav2Vec2 was trained on clips
        # that had leading/trailing silence; feeding a tight 400 ms word with
        # no padding destabilizes the CTC decoder (it sees a transient
        # onset/offset the training data never had). 200 ms each side.
        min_padded = int(sample_rate * 0.2)
        pad = np.zeros(min_padded, dtype=np.float32)
        audio = np.concatenate([pad, audio.astype(np.float32, copy=False), pad])
        inputs = self._feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self._model(input_values).logits[0]  # [T, V]
        probs = torch.softmax(logits, dim=-1)
        conf, ids = probs.max(dim=-1)
        ids = ids.cpu().numpy()
        conf = conf.cpu().numpy()

        blank = self._model.config.pad_token_id
        id_to_tok = self._id_to_tok

        # For diagnostics: record every segment the model predicted — before
        # the confidence filter — so we can see which phonemes got dropped.
        all_segments: list[tuple[str, float]] = []  # (ipa, avg_confidence)
        tokens: list[PhonemeToken] = []
        cur_id = None
        cur_start = 0
        cur_confs: list[float] = []
        from ..data.ipa import normalize_to_english

        def _emit(cur_id_: int, cur_start_: int, end_t: int, cur_confs_: list[float]) -> None:
            raw = id_to_tok.get(int(cur_id_), "")
            if not raw or raw in _NON_PHONEME_TOKENS:
                return
            ipa = normalize_to_english(raw)
            if not ipa or ipa in _NON_PHONEME_TOKENS:
                return
            start = cur_start_ * self._frame_hop_s
            end = end_t * self._frame_hop_s
            c = float(np.mean(cur_confs_)) if cur_confs_ else 0.0
            all_segments.append((ipa, c))
            if c >= min_confidence:
                tokens.append(PhonemeToken(ipa, start, end, c))

        for t, tok_id in enumerate(ids):
            if tok_id != cur_id:
                if cur_id is not None and cur_id != blank:
                    _emit(cur_id, cur_start, t, cur_confs)
                cur_id = int(tok_id)
                cur_start = t
                cur_confs = [float(conf[t])]
            else:
                cur_confs.append(float(conf[t]))
        # flush
        if cur_id is not None and cur_id != blank:
            _emit(cur_id, cur_start, len(ids), cur_confs)

        if merge_repeats:
            tokens = _merge_repeats(tokens)

        # Diagnostic: log every segment the model predicted plus the subset
        # that cleared the confidence filter. If "kept" is much shorter than
        # "all" on your real-mic holds, the filter is what's dropping chips —
        # lower phoneme_min_confidence in settings. If "all" itself is tiny,
        # the model isn't picking up the acoustic content (mic gain, noise).
        log.info(
            "[PHONEME] all(%d)=%s kept(%d)=%s",
            len(all_segments),
            [(s, round(c, 2)) for s, c in all_segments],
            len(tokens),
            [(t.ipa, round(t.confidence, 2)) for t in tokens],
        )
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
