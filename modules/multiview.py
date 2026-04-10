"""
multiview.py — Wonder3D multi-view generation module

Input:  single reference image (any resolution)
Output: 6 consistent views (color + normal maps) saved to output_dir
"""

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class MultiviewConfig:
    num_views: int = 6
    image_size: int = 256
    remove_background: bool = True
    output_dir: str = "outputs/views"


class MultiviewGenerator:
    """Wraps Wonder3D to generate multi-view consistent images from a single input."""

    def __init__(self, config: MultiviewConfig):
        self.config = config
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return

        logger.info("Loading Wonder3D pipeline...")
        try:
            # Wonder3D uses a custom diffusion pipeline
            # Repo: https://github.com/xxlong0/Wonder3D
            from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
            import torch

            self._pipeline = MVDiffusionImagePipeline.from_pretrained(
                "flamehaze1115/wonder3d-v1.0",
                torch_dtype=torch.float16,
            )
            self._pipeline.to("cuda")
            logger.success("Wonder3D loaded.")
        except ImportError:
            raise RuntimeError(
                "Wonder3D not installed. Clone https://github.com/xxlong0/Wonder3D "
                "and install its dependencies first."
            )

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background using rembg."""
        try:
            from rembg import remove
            return remove(image)
        except ImportError:
            logger.warning("rembg not installed — skipping background removal. "
                           "Run: pip install rembg")
            return image

    def _preprocess(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGBA")

        if self.config.remove_background:
            logger.info("Removing background...")
            img = self._remove_background(img)

        # Paste onto white background, resize
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background.convert("RGB")
        img = img.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)
        return img

    def generate(self, image_path: str) -> dict:
        """
        Run Wonder3D on a single image.

        Returns:
            {
                "color_views": [Path, ...],   # 6 color images
                "normal_views": [Path, ...],  # 6 normal maps
                "output_dir": Path
            }
        """
        self._load_model()

        image_path = Path(image_path)
        output_dir = Path(self.config.output_dir) / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {self.config.num_views} views for: {image_path.name}")
        img = self._preprocess(str(image_path))

        import torch
        with torch.inference_mode():
            output = self._pipeline(
                image=img,
                num_views=self.config.num_views,
                guidance_scale=3.0,
                num_inference_steps=50,
            )

        color_views = []
        normal_views = []

        for i, (color, normal) in enumerate(zip(output.images, output.normals)):
            color_path = output_dir / f"color_{i:02d}.png"
            normal_path = output_dir / f"normal_{i:02d}.png"

            color.save(color_path)
            normal.save(normal_path)

            color_views.append(color_path)
            normal_views.append(normal_path)

        logger.success(f"Saved {len(color_views)} views to {output_dir}")

        return {
            "color_views": color_views,
            "normal_views": normal_views,
            "output_dir": output_dir,
        }
