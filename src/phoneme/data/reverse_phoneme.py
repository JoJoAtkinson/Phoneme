"""Phoneme → word lookup using CMU Pronouncing Dictionary.

Given a sequence of IPA-ish phonemes, find the English word(s) whose
pronunciation best matches. Used by the "phoneme compression" mode: user says
/k/ ... /æ/ ... /t/ one at a time, we glue them and look up "cat".
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence

log = logging.getLogger(__name__)

# Coarse ARPAbet → espeak IPA map. CMUdict uses ARPAbet without stress-less
# variants, so we normalize on both sides.
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ",
    "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜː",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ",
    "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ",
    "UH": "ʊ", "UW": "uː", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


def _strip_stress(p: str) -> str:
    # CMU appends "0", "1", "2" to vowels
    return "".join(ch for ch in p if not ch.isdigit())


@functools.lru_cache(maxsize=1)
def _ipa_index() -> dict[tuple[str, ...], list[str]]:
    """Build { (ipa1, ipa2, ...) : [word, ...] } from CMU dict."""
    import cmudict

    d = cmudict.dict()
    index: dict[tuple[str, ...], list[str]] = {}
    for word, prons in d.items():
        for pron in prons:
            ipa = tuple(ARPABET_TO_IPA.get(_strip_stress(p), "") for p in pron)
            ipa = tuple(x for x in ipa if x)
            if not ipa:
                continue
            index.setdefault(ipa, []).append(word)
    log.info("indexed %d pronunciations across %d entries", sum(map(len, index.values())), len(index))
    return index


def normalize_phoneme(p: str) -> str:
    """Collapse a user-produced phoneme to our canonical IPA form."""
    p = p.replace("ˈ", "").replace("ˌ", "")
    # squash long variant into base for matching (we keep a loose-match pass)
    return p


def lookup(phonemes: Sequence[str]) -> list[str]:
    """Exact match first; then loose match (ignore vowel length)."""
    key = tuple(normalize_phoneme(p) for p in phonemes)
    idx = _ipa_index()
    if key in idx:
        return idx[key]
    # Loose match: strip ː
    loose = tuple(p.rstrip("ː") for p in key)
    loose_hits: list[str] = []
    for k, words in idx.items():
        if tuple(x.rstrip("ː") for x in k) == loose:
            loose_hits.extend(words)
    if loose_hits:
        return loose_hits
    # Nearest neighbor by edit distance, top 5
    return _nearest(key, idx, k=5)


def _nearest(key: tuple[str, ...], idx: dict, k: int = 5) -> list[str]:
    best: list[tuple[int, str]] = []
    for kk, words in idx.items():
        d = _edit_distance(key, kk)
        for w in words:
            if len(best) < k:
                best.append((d, w))
                best.sort()
            elif d < best[-1][0]:
                best[-1] = (d, w)
                best.sort()
    return [w for _, w in best]


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]
