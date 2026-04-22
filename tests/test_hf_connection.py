"""Light network test — downloads a tiny file from a public HF repo.

Skipped automatically when HF_TOKEN is absent AND offline. Safe to run in CI.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("PHONEME_SKIP_NET") == "1",
    reason="network tests disabled",
)
def test_can_download_small_file():
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id="facebook/wav2vec2-lv-60-espeak-cv-ft",
        filename="config.json",
        token=token,
    )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


@pytest.mark.skipif(
    os.environ.get("PHONEME_SKIP_NET") == "1" or not (
        os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    ),
    reason="HF token not set; skipping whoami",
)
def test_whoami_if_token_present():
    from huggingface_hub import HfApi

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    info = HfApi().whoami(token=token)
    assert info.get("name") or info.get("email") or info
