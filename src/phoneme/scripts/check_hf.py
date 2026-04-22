"""Smoke-test HF connectivity + tiny download.

Uses ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) from the environment, hits
the whoami endpoint, then downloads the phoneme model config.json as a
minimal download check. Safe to run anywhere — does not require audio
hardware or GPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="Repo to test-download one file from",
    )
    parser.add_argument(
        "--file", default="config.json", help="File to download from the repo"
    )
    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        print(f"HF token present ({len(token)} chars)")
    else:
        print("WARN: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN in env (public repos still work)")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        print(f"FAIL: huggingface_hub not installed: {e}")
        return 2

    # whoami
    if token:
        try:
            who = HfApi().whoami(token=token)
            print(f"whoami: {who.get('name') or who}")
        except Exception as e:
            print(f"WARN: whoami failed: {e}")

    # tiny file download
    try:
        path = hf_hub_download(repo_id=args.repo, filename=args.file, token=token)
        size = os.path.getsize(path)
        print(f"OK downloaded {args.repo}/{args.file} ({size} bytes) → {path}")
    except Exception as e:
        print(f"FAIL: download of {args.repo}/{args.file} failed: {e}")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
