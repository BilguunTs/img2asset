"""
cleanup.py — Blender MCP agent caller

Invokes blender/cleanup_script.py headlessly as a subprocess.
Input:  raw mesh path (.glb or .obj)
Output: game-ready mesh path
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class CleanupConfig:
    blender_bin: str = "blender"
    target_poly_count: int = 8000
    smooth_iterations: int = 2
    fill_holes: bool = True
    unwrap_uvs: bool = True
    bake_texture: bool = True
    texture_size: int = 1024
    output_dir: str = "outputs/final"


CLEANUP_SCRIPT = Path(__file__).parent.parent / "blender" / "cleanup_script.py"


class BlenderCleanup:
    """Runs the Blender cleanup script headlessly via subprocess."""

    def __init__(self, config: CleanupConfig):
        self.config = config

    def _check_blender(self):
        result = subprocess.run(
            [self.config.blender_bin, "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender not found at '{self.config.blender_bin}'. "
                "Install Blender and make sure it's in PATH or set blender_bin in config."
            )
        version_line = result.stdout.splitlines()[0]
        logger.info(f"Using {version_line}")

    def clean(self, mesh_path: Path, name: str = None) -> Path:
        """
        Run Blender cleanup on a raw mesh.

        Args:
            mesh_path: path to raw .glb / .obj from InstantMesh
            name:      output filename stem (defaults to input stem + "_clean")

        Returns:
            Path to the cleaned game-ready mesh
        """
        self._check_blender()

        mesh_path = Path(mesh_path)
        name = name or f"{mesh_path.stem}_clean"
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.glb"

        cmd = [
            self.config.blender_bin,
            "--background",
            "--python", str(CLEANUP_SCRIPT),
            "--",
            "--input",             str(mesh_path),
            "--output",            str(output_path),
            "--poly-target",       str(self.config.target_poly_count),
            "--smooth-iterations", str(self.config.smooth_iterations),
            "--texture-size",      str(self.config.texture_size),
        ]
        if self.config.fill_holes:
            cmd.append("--fill-holes")
        if self.config.unwrap_uvs:
            cmd.append("--unwrap-uvs")
        if self.config.bake_texture:
            cmd.append("--bake-texture")

        logger.info(f"Running Blender cleanup on: {mesh_path.name}")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Forward Blender stdout for visibility
        for line in result.stdout.splitlines():
            if line.startswith("[img2asset]") or "ERROR" in line:
                logger.info(line)

        if result.returncode != 0:
            logger.error(result.stderr[-2000:])  # last 2k chars of stderr
            raise RuntimeError(f"Blender cleanup failed (exit {result.returncode})")

        logger.success(f"Clean mesh saved: {output_path}")
        return output_path
