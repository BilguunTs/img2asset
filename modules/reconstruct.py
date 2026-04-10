"""
reconstruct.py — InstantMesh 3D reconstruction module

Input:  multi-view color + normal images (from multiview.py)
Output: raw .glb or .obj mesh file
"""

import os
from pathlib import Path
from dataclasses import dataclass

from loguru import logger


@dataclass
class ReconstructConfig:
    mesh_format: str = "glb"      # "glb" or "obj"
    output_dir: str = "outputs/meshes"


# Standard camera elevations Wonder3D uses for its 6 views
# (azimuth in degrees, elevation in degrees)
WONDER3D_CAMERA_POSES = [
    (0,   0),    # front
    (60,  0),    # front-right
    (120, 0),    # back-right
    (180, 0),    # back
    (240, 0),    # back-left
    (300, 0),    # front-left
]


class MeshReconstructor:
    """Wraps InstantMesh to reconstruct a 3D mesh from multi-view images."""

    def __init__(self, config: ReconstructConfig):
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        logger.info("Loading InstantMesh model...")
        try:
            import torch
            from huggingface_hub import hf_hub_download

            # InstantMesh repo: https://github.com/TencentARC/InstantMesh
            # Pull the model config + checkpoint
            model_ckpt = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="instantmesh-large.ckpt",
            )

            from src.models.instantmesh import InstantMesh  # from cloned repo

            self._model = InstantMesh.load_from_checkpoint(model_ckpt)
            self._model.eval().cuda()
            logger.success("InstantMesh loaded.")
        except ImportError:
            raise RuntimeError(
                "InstantMesh not installed. Clone https://github.com/TencentARC/InstantMesh "
                "and add it to your PYTHONPATH."
            )

    def _build_camera_params(self):
        """Build camera parameter tensors matching Wonder3D's 6-view layout."""
        import torch
        import math

        params = []
        for az, el in WONDER3D_CAMERA_POSES:
            az_rad = math.radians(az)
            el_rad = math.radians(el)
            params.append([az_rad, el_rad, 1.5])  # az, el, radius

        return torch.tensor(params, dtype=torch.float32).cuda()

    def reconstruct(self, multiview_result: dict, name: str = "mesh") -> Path:
        """
        Run InstantMesh on multi-view output from Wonder3D.

        Args:
            multiview_result: dict returned by MultiviewGenerator.generate()
            name:             stem name for the output file

        Returns:
            Path to the exported mesh file
        """
        self._load_model()

        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.{self.config.mesh_format}"

        color_views = multiview_result["color_views"]
        normal_views = multiview_result["normal_views"]

        logger.info(f"Reconstructing mesh from {len(color_views)} views...")

        # Load and stack images into tensors [N, C, H, W]
        color_tensors = torch.stack([
            TF.to_tensor(Image.open(p).convert("RGB"))
            for p in color_views
        ]).cuda()

        normal_tensors = torch.stack([
            TF.to_tensor(Image.open(p).convert("RGB"))
            for p in normal_views
        ]).cuda()

        cameras = self._build_camera_params()

        with torch.inference_mode():
            mesh = self._model(
                color_images=color_tensors.unsqueeze(0),    # [1, N, C, H, W]
                normal_images=normal_tensors.unsqueeze(0),
                cameras=cameras.unsqueeze(0),               # [1, N, 3]
            )

        # Export
        if self.config.mesh_format == "glb":
            mesh.export(str(output_path))
        else:
            mesh.export(str(output_path))

        logger.success(f"Mesh saved: {output_path}")
        return output_path
