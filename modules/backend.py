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
      'full'    — Wonder3D → InstantMesh  (CUDA)
      'triposr' — TripoSR only            (MPS / CPU)
    """
    if backend == "cuda":
        return "full"
    return "triposr"


def log_backend_info(backend: str, mode: str):
    if mode == "full":
        logger.info(f"Backend: {backend.upper()} — using full pipeline (Wonder3D → InstantMesh)")
    else:
        logger.warning(
            f"Backend: {backend.upper()} — CUDA not available. "
            f"Using TripoSR fallback (lower quality, single-image reconstruction)."
        )
        if backend == "mps":
            logger.info("Apple Silicon MPS detected — TripoSR will use Metal acceleration.")
        else:
            logger.warning("CPU only — reconstruction will be slow (5–15 min per image).")
