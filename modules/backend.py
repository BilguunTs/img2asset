"""
backend.py — Detect available compute backend and select pipeline path.

CUDA  → Wonder3D + InstantMesh  (full quality)
MPS   → TripoSR                 (Apple Silicon fallback)
CPU   → TripoSR                 (slow, emergency fallback)
"""

from loguru import logger


def detect() -> str:
    """Returns 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def select_pipeline(backend: str) -> str:
    """
    Returns pipeline mode string:
      'full'    — Wonder3D → InstantMesh      (CUDA)
      'zero123' — Zero123++ → CRM            (MPS / CUDA, high quality)
      'triposr' — TripoSR only               (MPS / CPU fallback)
    """
    if backend == "cuda":
        return "full"
    if backend == "mps":
        return "zero123"
    return "triposr"


def log_backend_info(backend: str, mode: str):
    if mode == "full":
        logger.info(f"Backend: {backend.upper()} — using full pipeline (Wonder3D → InstantMesh)")
    elif mode == "zero123":
        logger.info(
            f"Backend: {backend.upper()} — using Zero123++ → CRM pipeline "
            f"(multi-view, high quality, MPS-accelerated)"
        )
    else:
        logger.warning(
            f"Backend: {backend.upper()} — using TripoSR fallback "
            f"(single-image, lower quality)."
        )
        if backend == "cpu":
            logger.warning("CPU only — reconstruction will be slow (5–15 min per image).")
