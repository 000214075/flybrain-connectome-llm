"""Device selection and ROCm tuning.

The AMD Windows ROCm build exposes the HIP backend through the `torch.cuda` API,
so the same code paths are used as on an NVIDIA machine. What differs is the
environment: hipBLASLt is a large win for RDNA3 matmuls, and the caching
allocator needs expandable segments when activations are spiky.
"""

from __future__ import annotations

import os


def tune_rocm(verbose: bool = False, *, experimental_attention: bool = True) -> dict:
    """Set the ROCm environment variables that matter for training throughput.

    Returns the settings applied. Must be called before the first HIP call for
    the runtime options to take effect.

    Measured on an RX 7900 XTX (gfx1100, ROCm 7.10 / torch 2.9.1):
      * bf16 matmul reaches ~77 TFLOPS while fp32 reaches ~3 TFLOPS, so every
        training path here runs in bf16 autocast.
      * the flash and memory-efficient attention kernels are built but gated
        behind an experimental flag; enabling it is the difference between the
        MATH fallback and a fused kernel.
    """
    settings = {
        # RDNA3 ships hipBLASLt kernels that beat the legacy rocBLAS path.
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        # Unlock the AOTriton flash/mem-efficient attention kernels.
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1" if experimental_attention else "0",
        # Allow hipBLASLt to use TF32-equivalent paths where the input is fp32.
        "HIPBLASLT_ALLOW_TF32": "1",
        # Avoid re-selecting MIOpen kernels for every new shape.
        "MIOPEN_FIND_MODE": "FAST",
    }
    applied = {}
    for key, value in settings.items():
        if key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]
    if verbose:
        for key, value in applied.items():
            print(f"  {key}={value}")
    return applied


def pick_device(prefer: str | None = None):
    """Return the best available torch device, bias-corrected for ROCm naming."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> dict:
    """Everything worth knowing about the accelerator, for logs and the report."""
    import torch

    info: dict = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "is_rocm": bool(getattr(torch.version, "hip", None)),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": props.name,
                "gcn_arch": getattr(props, "gcnArchName", None),
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "multi_processor_count": props.multi_processor_count,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "sdpa_backends": _sdpa_backends(torch),
            }
        )
    return info


def _sdpa_backends(torch) -> dict:
    """Which scaled-dot-product-attention kernels this build actually has.

    Each probe is run inside a warning filter because torch explains at length
    why a kernel is unavailable, and those messages are noise in a log.
    """
    import warnings

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        names = {}
        for backend in (
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.CUDNN_ATTENTION,
        ):
            try:
                with warnings.catch_warnings(), sdpa_kernel(backend):
                    warnings.simplefilter("ignore")
                    q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
                    torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
                names[backend.name] = True
            except Exception:
                names[backend.name] = False
        return names
    except Exception as exc:  # pragma: no cover - only on very old torch
        return {"error": str(exc)}


def describe_memory() -> str:
    import torch

    if not torch.cuda.is_available():
        return "no accelerator"
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return (
        f"vram free {free / 1024**3:.2f} GiB / total {total / 1024**3:.2f} GiB, "
        f"torch allocated {allocated / 1024**3:.2f} GiB"
    )
