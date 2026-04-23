"""Lazy model downloader using huggingface_hub.

Weights live under ``models_dir()``. By default the ``.app`` we ship is small
and downloads only what's needed at runtime. When running locally you can
pre-fetch any profile with the CLI:

    phoneme-download                 # default stack (~800 MB)
    phoneme-download --profile best  # highest-quality stack (~4 GB)
    phoneme-download --profile full  # every model in the catalog (~7 GB)
    phoneme-download --profile mac   # everything plus Apple Silicon extras
    phoneme-download --list          # show the catalog with size estimates
    phoneme-download <key1> <key2>   # download specific keys by hand
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import models_dir

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    repo_id: str
    kind: str  # "phoneme" | "asr" | "tts"
    approx_mb: int = 0
    mac_only: bool = False
    allow_patterns: list[str] | None = None
    notes: str = ""


# Curated catalog of every model the app can switch between.
CATALOG: dict[str, ModelSpec] = {
    # ---- Phoneme recognizers (IPA output) --------------------------------
    "phoneme/wav2vec2-espeak": ModelSpec(
        repo_id="facebook/wav2vec2-lv-60-espeak-cv-ft",
        kind="phoneme",
        approx_mb=315,
        notes="IPA phoneme recognizer (multilingual). App default.",
    ),
    "phoneme/wav2vec2-gruut-en": ModelSpec(
        repo_id="bookbot/wav2vec2-ljspeech-gruut",
        kind="phoneme",
        approx_mb=360,
        notes="English-only gruut phonemes, finer-grained for US English.",
    ),
    "phoneme/wav2vec2-speech31-en": ModelSpec(
        repo_id="speech31/wav2vec2-large-english-phoneme-v2",
        kind="phoneme",
        approx_mb=1260,
        notes=(
            "English-only, 315M params, trained on varied English speech. "
            "More robust to real-world mic audio than the LJSpeech-trained "
            "gruut model — recommended when running with a laptop/Bluetooth mic."
        ),
    ),
    # ---- ASR -------------------------------------------------------------
    # English-only (.en) variants are smaller and more accurate on English
    # speech than the same-size multilingual models. The app defaults to
    # these; multilingual entries below are available for users who need them.
    "asr/faster-whisper-tiny.en": ModelSpec(
        repo_id="Systran/faster-whisper-tiny.en",
        kind="asr",
        approx_mb=75,
        notes="English-only tiny Whisper. Fastest.",
    ),
    "asr/faster-whisper-base.en": ModelSpec(
        repo_id="Systran/faster-whisper-base.en",
        kind="asr",
        approx_mb=145,
        notes="English-only base Whisper.",
    ),
    "asr/faster-whisper-small.en": ModelSpec(
        repo_id="Systran/faster-whisper-small.en",
        kind="asr",
        approx_mb=460,
        notes="App default ASR; English-only, balanced speed/accuracy.",
    ),
    "asr/faster-whisper-medium.en": ModelSpec(
        repo_id="Systran/faster-whisper-medium.en",
        kind="asr",
        approx_mb=1500,
        notes="English-only medium Whisper.",
    ),
    # Multilingual fallbacks (no .en variant exists for large-v3).
    "asr/faster-whisper-small": ModelSpec(
        repo_id="Systran/faster-whisper-small",
        kind="asr",
        approx_mb=460,
        notes="Multilingual small Whisper.",
    ),
    "asr/faster-whisper-large-v3": ModelSpec(
        repo_id="Systran/faster-whisper-large-v3",
        kind="asr",
        approx_mb=3000,
        notes="Best Whisper accuracy; multilingual only at this size.",
    ),
    "asr/parakeet-mlx": ModelSpec(
        repo_id="mlx-community/parakeet-tdt-0.6b-v2",
        kind="asr",
        approx_mb=1200,
        mac_only=True,
        notes="Apple Silicon only; best-in-class ASR quality.",
    ),
    # ---- TTS -------------------------------------------------------------
    "tts/kokoro-onnx": ModelSpec(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
        kind="tts",
        approx_mb=330,
        allow_patterns=["*.onnx", "*.bin", "*.json", "voices/*"],
        notes="Apache 2.0, fast, no voice cloning. App default TTS.",
    ),
}


# Download presets. A profile is just a list of catalog keys.
@dataclass
class Profile:
    description: str
    keys: list[str] = field(default_factory=list)


PROFILES: dict[str, Profile] = {
    "minimal": Profile(
        description="Smallest stack that boots the app (~400 MB, English-only).",
        keys=[
            "phoneme/wav2vec2-espeak",
            "asr/faster-whisper-tiny.en",
            "tts/kokoro-onnx",
        ],
    ),
    "default": Profile(
        description="Balanced defaults used at first launch (~1.1 GB, English-only).",
        keys=[
            "phoneme/wav2vec2-espeak",
            "asr/faster-whisper-small.en",
            "tts/kokoro-onnx",
        ],
    ),
    "best": Profile(
        description="Highest-quality cross-platform stack (~3.6 GB).",
        keys=[
            "phoneme/wav2vec2-espeak",
            "asr/faster-whisper-medium.en",
            "asr/faster-whisper-large-v3",  # multilingual fallback
            "tts/kokoro-onnx",
        ],
    ),
    "mac": Profile(
        description="Best quality on Apple Silicon (adds Parakeet-MLX, ~4.8 GB, English-only).",
        keys=[
            "phoneme/wav2vec2-espeak",
            "asr/faster-whisper-small.en",
            "asr/parakeet-mlx",  # English-only by design
            "tts/kokoro-onnx",
        ],
    ),
    "full": Profile(
        description="Every model in the catalog (everything, ~7 GB).",
        keys=list(CATALOG.keys()),
    ),
}


def hf_token() -> str | None:
    """Return HF token from env, preferring HUGGING_FACE_HUB_TOKEN."""
    return os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def local_dir_for(spec: ModelSpec) -> Path:
    safe = spec.repo_id.replace("/", "__")
    p = models_dir() / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def ensure_downloaded(
    key: str,
    progress: "callable | None" = None,
    skip_mac_only: bool = True,
) -> Path | None:
    """Download (or verify) a model by catalog key, return its local dir."""
    if key not in CATALOG:
        raise KeyError(f"Unknown model key '{key}'. Known: {list(CATALOG)}")
    spec = CATALOG[key]
    if skip_mac_only and spec.mac_only and not is_apple_silicon():
        log.warning("skipping %s (Apple Silicon only)", key)
        return None
    target = local_dir_for(spec)

    from huggingface_hub import snapshot_download

    log.info("ensuring %s → %s", spec.repo_id, target)
    path = snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(target),
        token=hf_token(),
        allow_patterns=spec.allow_patterns,
    )
    if progress is not None:
        progress(1.0)
    return Path(path)


def list_models() -> list[tuple[str, ModelSpec]]:
    return sorted(CATALOG.items())


def estimate_mb(keys: list[str]) -> int:
    return sum(CATALOG[k].approx_mb for k in keys if k in CATALOG)


def main() -> int:
    """CLI: ``phoneme-download [--profile NAME] [keys...]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Download Phoneme models. Run with no args for the default stack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"  {name:8s} {p.description}" for name, p in PROFILES.items()]
        ),
    )
    parser.add_argument(
        "keys",
        nargs="*",
        help="catalog keys, e.g. phoneme/wav2vec2-espeak asr/parakeet-mlx",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        help="Named profile to download. 'full' grabs every model in the catalog.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alias for '--profile full' (every model in the catalog).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the catalog and profiles, then exit.",
    )
    parser.add_argument(
        "--include-mac-only",
        action="store_true",
        help="Include Apple-Silicon-only models even on non-Mac hosts (will fail at load time).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded and the approx size; do not fetch.",
    )
    args = parser.parse_args()

    if args.list:
        print("Catalog:")
        for key, spec in list_models():
            tag = " [mac-only]" if spec.mac_only else ""
            size = f"{spec.approx_mb:>5} MB" if spec.approx_mb else "       ?"
            print(f"  {key:32s} {size}{tag}  {spec.notes}")
        print("\nProfiles:")
        for name, p in PROFILES.items():
            mb = estimate_mb(p.keys)
            print(f"  {name:8s} ~{mb} MB  {p.description}")
        return 0

    if args.all:
        keys = PROFILES["full"].keys
    elif args.profile:
        keys = PROFILES[args.profile].keys
    elif args.keys:
        keys = args.keys
    else:
        keys = PROFILES["default"].keys

    # Filter mac-only unless asked for them
    skip_mac_only = not (args.include_mac_only or is_apple_silicon())
    runnable = [k for k in keys if not (skip_mac_only and CATALOG[k].mac_only)]
    skipped = [k for k in keys if k not in runnable]

    total_mb = estimate_mb(runnable)
    print(f"Will download {len(runnable)} model(s), ~{total_mb} MB:")
    for k in runnable:
        spec = CATALOG[k]
        print(f"  {k:32s} ~{spec.approx_mb} MB   {spec.repo_id}")
    if skipped:
        print(f"Skipping {len(skipped)} Apple-Silicon-only model(s) on this host: {skipped}")
    if args.dry_run:
        print("(dry run — nothing fetched)")
        return 0

    failures: list[tuple[str, str]] = []
    for key in runnable:
        try:
            path = ensure_downloaded(key, skip_mac_only=skip_mac_only)
            if path is not None:
                print(f"OK {key} → {path}")
        except Exception as e:
            failures.append((key, str(e)))
            print(f"FAIL {key}: {e}")
    if failures:
        print(f"\n{len(failures)} model(s) failed:")
        for k, msg in failures:
            print(f"  - {k}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
