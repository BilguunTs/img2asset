"""
validate.py — Mesh quality validation

Checks the reconstructed mesh for common issues and logs a quality report.
Optionally renders a turntable preview.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ValidationConfig:
    enabled: bool = True
    check_manifold: bool = True
    check_holes: bool = True
    max_poly_warn: int = 15000
    render_preview: bool = True
    preview_frames: int = 8


@dataclass
class ValidationReport:
    poly_count: int
    is_manifold: bool
    has_holes: bool
    warnings: list
    passed: bool
    preview_dir: Path = None

    def print(self):
        status = "PASS" if self.passed else "WARN"
        logger.info(f"  [{status}] Poly count:  {self.poly_count}")
        logger.info(f"  [{status}] Manifold:    {self.is_manifold}")
        logger.info(f"  [{status}] Holes:       {'yes' if self.has_holes else 'no'}")
        for w in self.warnings:
            logger.warning(f"  ! {w}")
        if self.preview_dir:
            logger.info(f"  Preview frames: {self.preview_dir}")


class MeshValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(self, mesh_path: Path) -> ValidationReport:
        if not self.config.enabled:
            return ValidationReport(
                poly_count=0, is_manifold=True,
                has_holes=False, warnings=[], passed=True
            )

        try:
            import trimesh
        except ImportError:
            raise RuntimeError("trimesh not installed. Run: pip install trimesh")

        logger.info(f"Validating mesh: {mesh_path.name}")
        mesh = trimesh.load(str(mesh_path), force="mesh")

        poly_count = len(mesh.faces)
        is_manifold = mesh.is_watertight  # watertight = manifold + no holes
        has_holes = not mesh.is_watertight

        warnings = []
        if poly_count > self.config.max_poly_warn:
            warnings.append(f"Poly count {poly_count} exceeds warning threshold {self.config.max_poly_warn}")
        if not is_manifold and self.config.check_manifold:
            warnings.append("Mesh is not manifold — may cause issues in game engines")
        if has_holes and self.config.check_holes:
            warnings.append("Mesh has open holes — consider re-running cleanup with fill_holes=true")

        preview_dir = None
        if self.config.render_preview:
            preview_dir = self._render_turntable(mesh, mesh_path)

        report = ValidationReport(
            poly_count=poly_count,
            is_manifold=is_manifold,
            has_holes=has_holes,
            warnings=warnings,
            passed=len(warnings) == 0,
            preview_dir=preview_dir,
        )
        report.print()
        return report

    def _render_turntable(self, mesh, mesh_path: Path) -> Path:
        """Render N frames rotating around the mesh and save as PNGs."""
        import math
        import trimesh
        import numpy as np

        preview_dir = mesh_path.parent / f"{mesh_path.stem}_preview"
        preview_dir.mkdir(exist_ok=True)

        try:
            scene = trimesh.Scene(mesh)
            n = self.config.preview_frames

            for i in range(n):
                angle = (2 * math.pi / n) * i
                scene.set_camera(
                    angles=[0, angle, 0],
                    distance=mesh.scale * 2.5,
                )
                png = scene.save_image(resolution=(512, 512))
                frame_path = preview_dir / f"frame_{i:02d}.png"
                frame_path.write_bytes(png)

            logger.info(f"  Turntable preview: {n} frames saved to {preview_dir}")
        except Exception as e:
            logger.warning(f"  Preview render failed: {e}")

        return preview_dir
