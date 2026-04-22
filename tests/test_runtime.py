"""Runtime helper tests — no heavy deps required."""

from __future__ import annotations

import os

from phoneme.runtime import describe, is_apple_silicon, onnx_providers, torch_device


def test_torch_device_returns_a_string():
    # When PyTorch isn't installed, we fall back to 'cpu'. When it is,
    # whatever torch reports should be valid.
    d = torch_device()
    assert isinstance(d, str)
    assert d in {"cpu", "mps", "cuda"}


def test_onnx_providers_never_empty_and_ends_with_cpu():
    # Even without onnxruntime installed, we return a sensible default.
    providers = onnx_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1
    last = providers[-1]
    name = last if isinstance(last, str) else last[0]
    assert name == "CPUExecutionProvider"


def test_describe_has_expected_keys():
    info = describe()
    for key in ("platform", "machine", "torch_device", "onnx_providers", "mlx"):
        assert key in info


def test_device_env_override(monkeypatch):
    # Clear the lru_cache first so the override takes effect.
    from phoneme import runtime

    runtime.torch_device.cache_clear()
    monkeypatch.setenv("PHONEME_DEVICE", "cpu")
    assert runtime.torch_device() == "cpu"
    runtime.torch_device.cache_clear()
    monkeypatch.delenv("PHONEME_DEVICE", raising=False)


def test_apple_silicon_only_true_on_mac_arm64():
    # Tautological but ensures the helper doesn't crash.
    result = is_apple_silicon()
    assert isinstance(result, bool)
