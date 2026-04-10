"""
multiview_zero123.py — Zero123++ multi-view generation

Generates 6 consistent views from a single image using Zero123++.
Works on Apple Silicon MPS and CUDA — no CUDA-specific ops.

Model:  sudo-ai/zero123plus-v1.2
Output: 6 views arranged as a 2×3 grid at fixed camera positions:
    (az=30°, el=20°)  (az=90°, el=-10°)  (az=150°, el=20°)
    (az=210°, el=-10°) (az=270°, el=20°) (az=330°, el=-10°)
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


# Fixed camera poses Zero123++ uses (azimuth, elevation) in degrees
ZERO123_CAMERA_POSES = [
    (30,   20),
    (90,  -10),
    (150,  20),
    (210, -10),
    (270,  20),
    (330, -10),
]


@dataclass
class Zero123Config:
    num_inference_steps: int = 75
    guidance_scale: float = 4.0
    image_size: int = 512
    remove_background: bool = True
    output_dir: str = "outputs/views"


class Zero123MultiviewGenerator:
    """Generates 6 consistent views using Zero123++. Runs on MPS and CUDA."""

    def __init__(self, config: Zero123Config):
        self.config = config
        self._pipeline = None
        self._device = None

    def _detect_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("No GPU found — Zero123++ will run on CPU (very slow).")
        return "cpu"

    def _load_model(self):
        if self._pipeline is not None:
            return

        import torch
        from diffusers import DiffusionPipeline

        self._device = self._detect_device()
        logger.info(f"Loading Zero123++ on {self._device}...")

        # float16 only works reliably on CUDA; use float32 on MPS/CPU
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        # Use local pipeline.py from cloned repo (sudo-ai/zero123plus is private on HF)
        pipeline_file = Path(__file__).parent.parent / "deps" / "zero123plus" / "diffusers-support" / "pipeline.py"
        if not pipeline_file.exists():
            raise RuntimeError(
                "Zero123++ pipeline not found. Run:\n"
                "  git clone https://github.com/SUDO-AI-3D/zero123plus deps/zero123plus"
            )

        self._pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline=str(pipeline_file),
            torch_dtype=dtype,
        )
        self._pipeline.to(self._device)
        logger.success(f"Zero123++ loaded on {self._device}")

    def _preprocess(self, image_path: str):
        """Remove background and paste onto white for Zero123++ input."""
        from PIL import Image
        import rembg

        img = Image.open(image_path).convert("RGBA")

        if self.config.remove_background:
            logger.info("Removing background...")
            session = rembg.new_session()
            img = rembg.remove(img, session=session)

        # White background — Zero123++ expects clean white bg
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert("RGB").resize(
            (self.config.image_size, self.config.image_size), Image.LANCZOS
        )

    def generate(self, image_path: str) -> dict:
        """
        Generate 6 views from a single image using Zero123++.

        Returns:
            {
                "color_views": [Path, ...],    # 6 individual view images
                "camera_poses": list,           # (azimuth, elevation) per view
                "output_dir": Path,
            }
        """
        self._load_model()

        import torch

        image_path = Path(image_path)
        output_dir = Path(self.config.output_dir) / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating 6 views for: {image_path.name}")
        img = self._preprocess(str(image_path))

        with torch.inference_mode():
            result = self._pipeline(
                img,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
            ).images[0]

        # Zero123++ outputs a single 960×640 image — a 2×3 grid of 320×320 views
        color_views = []
        view_w, view_h = 320, 320

        for row in range(2):
            for col in range(3):
                idx = row * 3 + col
                view = result.crop((
                    col * view_w, row * view_h,
                    col * view_w + view_w, row * view_h + view_h,
                ))
                path = output_dir / f"view_{idx:02d}.png"
                view.save(path)
                color_views.append(path)
                az, el = ZERO123_CAMERA_POSES[idx]
                logger.debug(f"  View {idx}: az={az}° el={el}°")

        logger.success(f"Saved {len(color_views)} views to {output_dir}")
        return {
            "color_views": color_views,
            "camera_poses": ZERO123_CAMERA_POSES,
            "output_dir": output_dir,
        }
