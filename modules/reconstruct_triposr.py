"""
reconstruct_triposr.py — TripoSR single-image 3D reconstruction

Fallback path for non-CUDA systems (Apple Silicon MPS, CPU).
Skips the multi-view generation stage entirely.

Input:  single reference image (preprocessed)
Output: raw .glb mesh

Repo: https://github.com/VAST-AI-Research/TripoSR
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class TripoSRConfig:
    mesh_format: str = "glb"
    chunk_size: int = 8192          # lower = less VRAM, slower
    mc_resolution: int = 256        # marching cubes resolution (128=fast, 256=quality)
    remove_background: bool = True
    image_size: int = 512
    output_dir: str = "outputs/meshes"


class TripoSRReconstructor:
    """
    Runs TripoSR directly on a single image.
    Works on Apple Silicon (MPS) and CPU — no CUDA required.
    """

    def __init__(self, config: TripoSRConfig):
        self.config = config
        self._model = None
        self._device = None

    def _detect_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("No GPU found — running on CPU. Will be slow.")
        return "cpu"

    def _load_model(self):
        if self._model is not None:
            return

        self._device = self._detect_device()
        logger.info(f"Loading TripoSR on {self._device}...")

        # Suppress xatlas import errors — we skip it intentionally on Apple Silicon
        import sys, types
        if "xatlas" not in sys.modules:
            sys.modules["xatlas"] = types.ModuleType("xatlas")  # stub

        try:
            from tsr.system import TSR

            self._model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self._model.renderer.set_chunk_size(self.config.chunk_size)
            self._model.to(self._device)
            logger.success(f"TripoSR loaded on {self._device}")
        except ImportError:
            raise RuntimeError(
                "TripoSR not installed. Run setup.sh or:\n"
                "  git clone https://github.com/VAST-AI-Research/TripoSR deps/TripoSR\n"
                "  pip install -r deps/TripoSR/requirements.txt\n"
                "  export PYTHONPATH=$PYTHONPATH:deps/TripoSR"
            )

    def _preprocess(self, image_path: str):
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGBA")

        if self.config.remove_background:
            try:
                from rembg import remove
                logger.info("Removing background...")
                img = remove(img)
            except ImportError:
                logger.warning("rembg not installed — skipping background removal")

        # Composite onto white, resize
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB").resize(
            (self.config.image_size, self.config.image_size), Image.LANCZOS
        )
        return img

    def reconstruct(self, image_path: str, name: str = None) -> Path:
        """
        Run TripoSR on a single image.

        Args:
            image_path: path to reference image
            name:       output filename stem

        Returns:
            Path to the exported mesh file
        """
        self._load_model()

        import torch

        image_path = Path(image_path)
        name = name or image_path.stem
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.{self.config.mesh_format}"

        logger.info(f"Running TripoSR on: {image_path.name}")
        img = self._preprocess(str(image_path))

        with torch.inference_mode():
            scene_codes = self._model([img], device=self._device)
            meshes = self._model.extract_mesh(
                scene_codes,
                resolution=self.config.mc_resolution,
            )

        mesh = meshes[0]
        mesh.export(str(output_path))

        logger.success(f"TripoSR mesh saved: {output_path}")
        return output_path
