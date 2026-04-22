"""Hardware acceleration helpers.

Single source of truth for "what device / providers should this backend run
on?". Every model loader funnels through here so behavior is consistent.

Supported accelerators:
  * Apple Silicon:
      - PyTorch: MPS (Metal Performance Shaders)
      - ONNX Runtime: CoreMLExecutionProvider (uses Neural Engine + GPU)
      - MLX: native Metal (parakeet-mlx, mlx-whisper)
      - faster-whisper (CTranslate2): CPU with int8 quantization — CT2 has
        no Metal backend yet, but int8 CPU on M-series is fast enough for
        small/medium models in real time. Prefer mlx-whisper for best speed.
  * NVIDIA: PyTorch CUDA, ONNX CUDA EP.
  * CPU fallback everywhere.

Environment overrides (handy for debugging):
  PHONEME_DEVICE=cpu|mps|cuda        force torch device
  PHONEME_DISABLE_COREML=1           skip CoreML EP even when available
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from functools import lru_cache

log = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


@lru_cache(maxsize=1)
def torch_device() -> str:
    """Best available PyTorch device. 'mps' on Apple Silicon, else cuda/cpu."""
    override = os.environ.get("PHONEME_DEVICE")
    if override:
        return override
    try:
        import torch  # noqa: F401
    except ImportError:
        return "cpu"
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def onnx_providers() -> list:
    """Provider list to pass to onnxruntime.InferenceSession.

    On Apple Silicon we lead with CoreMLExecutionProvider (which uses the
    Neural Engine / Metal), then fall back to CPU. Order matters — ORT tries
    them left-to-right and uses the first one that can host each op.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    providers: list = []

    if is_apple_silicon() and "CoreMLExecutionProvider" in available:
        if os.environ.get("PHONEME_DISABLE_COREML") != "1":
            # CoreML EP options: allow GPU + ANE + CPU, use FP16 acceleration.
            providers.append(
                (
                    "CoreMLExecutionProvider",
                    {
                        # 0 = ALL, 1 = CPU_AND_GPU, 2 = ALL + FP16 + sub-graph partition
                        # mlas 2 = ANE + GPU + CPU, FP16 where safe.
                        "MLComputeUnits": "ALL",
                        "AllowLowPrecisionAccumulationOnGPU": "1",
                        "ModelFormat": "MLProgram",
                    },
                )
            )

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


@lru_cache(maxsize=1)
def has_mlx() -> bool:
    """Apple MLX framework available (required for mlx-whisper, parakeet-mlx)."""
    if not is_apple_silicon():
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def describe() -> dict:
    """One-line summary of detected accelerators, for the About dialog / logs."""
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "torch_device": torch_device(),
        "onnx_providers": [p if isinstance(p, str) else p[0] for p in onnx_providers()],
        "mlx": has_mlx(),
    }
