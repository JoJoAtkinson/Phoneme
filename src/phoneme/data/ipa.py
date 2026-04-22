"""IPA helpers: category coloring, long/short vowel detection, articulation hints."""

from __future__ import annotations

# Category → fill color (QSS-friendly hex)
CATEGORY_COLOR = {
    "vowel_long": "#f6a623",
    "vowel_short": "#f5d742",
    "vowel_diphthong": "#f78f3e",
    "nasal": "#8e6bd9",
    "stop_voiced": "#4c8bf5",
    "stop_unvoiced": "#8fa7c7",
    "fricative_voiced": "#2aa6a0",
    "fricative_unvoiced": "#7dc3be",
    "affricate": "#d36bd0",
    "approximant": "#62c478",
    "other": "#bbbbbb",
}

# Strip diacritics like ː, ˈ, ˌ before classifying
_STRIP = str.maketrans("", "", "ˈˌ")


# Canonical espeak/IPA vowel sets
LONG_VOWELS = {"iː", "uː", "ɑː", "ɔː", "ɜː", "eː", "oː", "aː", "yː"}
SHORT_VOWELS = {"ɪ", "ʊ", "ʌ", "ə", "æ", "ɛ", "ɒ", "ɐ", "ɨ", "ʉ"}
DIPHTHONGS = {"aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "ɪə", "eə", "ʊə", "əʊ"}

VOICED_STOPS = {"b", "d", "ɡ", "g"}
UNVOICED_STOPS = {"p", "t", "k", "ʔ"}
NASALS = {"m", "n", "ŋ", "ɲ"}
VOICED_FRICATIVES = {"v", "z", "ʒ", "ð", "ɣ", "ʁ", "h"}
UNVOICED_FRICATIVES = {"f", "s", "ʃ", "θ", "x", "ç"}
AFFRICATES = {"tʃ", "dʒ", "ts", "dz"}
APPROXIMANTS = {"j", "w", "l", "ɹ", "r", "ɻ"}


def classify(ipa: str) -> str:
    t = ipa.translate(_STRIP)
    if t in LONG_VOWELS or t.endswith("ː"):
        return "vowel_long"
    if t in DIPHTHONGS:
        return "vowel_diphthong"
    if t in SHORT_VOWELS:
        return "vowel_short"
    if t in VOICED_STOPS:
        return "stop_voiced"
    if t in UNVOICED_STOPS:
        return "stop_unvoiced"
    if t in NASALS:
        return "nasal"
    if t in VOICED_FRICATIVES:
        return "fricative_voiced"
    if t in UNVOICED_FRICATIVES:
        return "fricative_unvoiced"
    if t in AFFRICATES:
        return "affricate"
    if t in APPROXIMANTS:
        return "approximant"
    return "other"


def color_for(ipa: str) -> str:
    return CATEGORY_COLOR[classify(ipa)]


def is_vowel(ipa: str) -> bool:
    cat = classify(ipa)
    return cat.startswith("vowel_")


# Long/short marker for pedagogical display. We mark:
#   * IPA-native long vowels (contain ː) → macron: ā
#   * Duration-observed long vowels (held > threshold) → macron: ¯
#   * Short vowels → breve: ̆
# For non-vowels we return None.
LONG_DURATION_S = 0.18


def vowel_length_marker(ipa: str, duration_s: float) -> str | None:
    if not is_vowel(ipa):
        return None
    if "ː" in ipa or ipa in LONG_VOWELS or duration_s >= LONG_DURATION_S:
        return "¯"
    return "˘"


# Place-of-articulation hint for the mouth diagram
ARTICULATION = {
    "p": "bilabial stop",
    "b": "bilabial stop (voiced)",
    "t": "alveolar stop",
    "d": "alveolar stop (voiced)",
    "k": "velar stop",
    "g": "velar stop (voiced)",
    "ɡ": "velar stop (voiced)",
    "m": "bilabial nasal",
    "n": "alveolar nasal",
    "ŋ": "velar nasal",
    "f": "labiodental fricative",
    "v": "labiodental fricative (voiced)",
    "θ": "dental fricative",
    "ð": "dental fricative (voiced)",
    "s": "alveolar fricative",
    "z": "alveolar fricative (voiced)",
    "ʃ": "postalveolar fricative",
    "ʒ": "postalveolar fricative (voiced)",
    "h": "glottal fricative",
    "tʃ": "postalveolar affricate",
    "dʒ": "postalveolar affricate (voiced)",
    "j": "palatal approximant",
    "w": "labial-velar approximant",
    "l": "alveolar lateral",
    "ɹ": "alveolar approximant",
    "r": "alveolar trill",
}


def articulation_hint(ipa: str) -> str:
    base = ipa.translate(_STRIP).rstrip("ː")
    return ARTICULATION.get(base, "")
