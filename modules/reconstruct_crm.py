"""
reconstruct_crm.py — Multi-view TripoSR fusion reconstruction

CRM requires nvdiffrast (CUDA-only) and can't run on MPS.
Instead: run TripoSR on 3 evenly-spaced views from Zero123++,
then fuse the meshes via voxel union. Each view covers a different
side of the object; the union gives complete 3D coverage.

Works entirely on MPS and CPU — no CUDA required.
"""

from dataclasses import dataclass
from pathlib import Path
from functools import reduce

from loguru import logger


@dataclass
class CRMConfig:
    mc_resolution: int = 256
    voxel_pitch: float = 0.015     # voxel grid cell size for fusion (smaller = more detail)
    mesh_format: str = "glb"
    output_dir: str = "outputs/meshes"


# Indices of the 3 views from Zero123++ that give best coverage
# Zero123++ poses: 30°, 90°, 150°, 210°, 270°, 330°
# Views 0, 2, 4 are spaced ~120° apart — good for 3-way fusion
FUSION_VIEW_INDICES = [0, 2, 4]


class CRMReconstructor:
    """
    Multi-view TripoSR fusion. Reconstructs from 3 views via voxel union.
    Runs on MPS and CUDA.
    """

    def __init__(self, config: CRMConfig):
        self.config = config
        self._triposr = None

    def _get_triposr(self):
        """Lazily load the TripoSR model (reuses any already-loaded instance)."""
        if self._triposr is not None:
            return self._triposr

        from modules.reconstruct_triposr import TripoSRReconstructor, TripoSRConfig
        cfg = TripoSRConfig(
            mc_resolution=self.config.mc_resolution,
            remove_background=False,   # Zero123++ views already have clean bg
        )
        self._triposr = TripoSRReconstructor(cfg)
        self._triposr._load_model()
        return self._triposr

    def _reconstruct_view(self, image_path: Path):
        """Run TripoSR on a single view, return a trimesh.Trimesh."""
        import torch
        from PIL import Image

        tsr = self._get_triposr()
        img = tsr._preprocess(str(image_path))

        with torch.inference_mode():
            scene_codes = tsr._model([img], device=tsr._device)
            meshes = tsr._model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=self.config.mc_resolution,
                threshold=tsr.config.mc_threshold,
            )
        return meshes[0]

    def reconstruct(self, multiview_result: dict, name: str = "mesh") -> Path:
        """
        Fuse 3 TripoSR reconstructions (from views 0, 2, 4) via voxel union.

        Args:
            multiview_result: dict from Zero123MultiviewGenerator.generate()
            name:             output filename stem

        Returns:
            Path to the exported mesh file
        """
        color_views = multiview_result["color_views"]
        camera_poses = multiview_result["camera_poses"]

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.{self.config.mesh_format}"

        import trimesh
        import numpy as np

        meshes = []
        for idx in FUSION_VIEW_INDICES:
            az, el = camera_poses[idx]
            logger.info(f"Reconstructing view {idx} (az={az}°, el={el}°)...")
            mesh = self._reconstruct_view(color_views[idx])
            meshes.append(mesh)

        logger.info("Fusing meshes via voxel union...")
        voxel_grids = [m.voxelized(pitch=self.config.voxel_pitch) for m in meshes]
        fused = reduce(lambda a, b: a | b, voxel_grids)
        result = fused.marching_cubes   # trimesh.Trimesh from the fused volume

        # Smooth out voxelization artefacts
        trimesh.smoothing.filter_laplacian(result, iterations=3)

        result.export(str(output_path))
        logger.success(f"Fused mesh saved: {output_path}")
        return output_path
