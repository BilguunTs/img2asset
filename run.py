"""
run.py — img2asset pipeline entrypoint

Usage:
    # Single image
    python run.py --image ref.png

    # Single image with custom config
    python run.py --image ref.png --config config/custom.yaml

    # Batch (folder of images)
    python run.py --batch ./reference_images/

    # Override specific settings
    python run.py --image ref.png --poly-target 5000 --no-bake

Pipeline stages:
    1. Wonder3D  → multi-view consistent images
    2. InstantMesh → raw 3D mesh
    3. Blender   → game-ready mesh (decimated, UVs, texture)
    4. Validate  → quality report + preview
"""

import sys
from pathlib import Path

import click
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from modules.multiview import MultiviewGenerator, MultiviewConfig
from modules.reconstruct import MeshReconstructor, ReconstructConfig
from modules.cleanup import BlenderCleanup, CleanupConfig
from modules.validate import MeshValidator, ValidationConfig

console = Console()

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ─── Config loading ───────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_configs(cfg: dict, overrides: dict):
    mv_cfg = cfg.get("multiview", {})
    re_cfg = cfg.get("reconstruct", {})
    cl_cfg = cfg.get("cleanup", {})
    va_cfg = cfg.get("validation", {})

    # Apply CLI overrides
    if overrides.get("poly_target"):
        cl_cfg["target_poly_count"] = overrides["poly_target"]
    if overrides.get("no_bake"):
        cl_cfg["bake_texture"] = False

    return (
        MultiviewConfig(**mv_cfg),
        ReconstructConfig(**re_cfg),
        CleanupConfig(**cl_cfg),
        ValidationConfig(**va_cfg),
    )


# ─── Single image pipeline ────────────────────────────────────────────────────

def run_pipeline(image_path: Path, mv_cfg, re_cfg, cl_cfg, va_cfg) -> dict:
    name = image_path.stem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Stage 1 — Multi-view generation
        task = progress.add_task(f"[cyan]Stage 1/4  Generating views ({name})...", total=None)
        generator = MultiviewGenerator(mv_cfg)
        multiview_result = generator.generate(str(image_path))
        progress.update(task, description=f"[green]Stage 1/4  Views generated ({len(multiview_result['color_views'])} views)")

        # Stage 2 — 3D reconstruction
        task = progress.add_task("[cyan]Stage 2/4  Reconstructing mesh...", total=None)
        reconstructor = MeshReconstructor(re_cfg)
        raw_mesh_path = reconstructor.reconstruct(multiview_result, name=name)
        progress.update(task, description="[green]Stage 2/4  Mesh reconstructed")

        # Stage 3 — Blender cleanup
        task = progress.add_task("[cyan]Stage 3/4  Running Blender cleanup...", total=None)
        cleanup = BlenderCleanup(cl_cfg)
        clean_mesh_path = cleanup.clean(raw_mesh_path, name=f"{name}_final")
        progress.update(task, description="[green]Stage 3/4  Cleanup done")

        # Stage 4 — Validation
        task = progress.add_task("[cyan]Stage 4/4  Validating output...", total=None)
        validator = MeshValidator(va_cfg)
        report = validator.validate(clean_mesh_path)
        status = "passed" if report.passed else f"passed with {len(report.warnings)} warning(s)"
        progress.update(task, description=f"[green]Stage 4/4  Validation {status}")

    return {
        "name": name,
        "input": image_path,
        "views": multiview_result["output_dir"],
        "raw_mesh": raw_mesh_path,
        "final_mesh": clean_mesh_path,
        "report": report,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--image",        required=True,  type=click.Path(exists=True), help="Path to reference image")
@click.option("--config",       default="config/default.yaml", show_default=True, help="Config file")
@click.option("--poly-target",  type=int,       default=None,  help="Override poly count target")
@click.option("--no-bake",      is_flag=True,   default=False, help="Skip texture baking")
def single(image, config, poly_target, no_bake):
    """Run the pipeline on a single image."""
    cfg = load_config(config)
    mv_cfg, re_cfg, cl_cfg, va_cfg = build_configs(cfg, {"poly_target": poly_target, "no_bake": no_bake})

    console.print(Panel.fit(
        f"[bold]img2asset[/bold] — single image\n"
        f"Input:  {image}\n"
        f"Config: {config}",
        border_style="blue"
    ))

    result = run_pipeline(Path(image), mv_cfg, re_cfg, cl_cfg, va_cfg)

    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n"
        f"Final asset: [cyan]{result['final_mesh']}[/cyan]\n"
        f"Poly count:  {result['report'].poly_count}\n"
        f"Manifold:    {result['report'].is_manifold}",
        border_style="green"
    ))


@cli.command()
@click.option("--batch",        required=True,  type=click.Path(exists=True), help="Folder of reference images")
@click.option("--config",       default="config/default.yaml", show_default=True)
@click.option("--poly-target",  type=int,       default=None)
@click.option("--no-bake",      is_flag=True,   default=False)
@click.option("--fail-fast",    is_flag=True,   default=False, help="Stop on first failure")
def batch(batch, config, poly_target, no_bake, fail_fast):
    """Run the pipeline on a folder of images."""
    cfg = load_config(config)
    batch_cfg = cfg.get("batch", {})
    skip_existing = batch_cfg.get("skip_existing", True)

    mv_cfg, re_cfg, cl_cfg, va_cfg = build_configs(cfg, {"poly_target": poly_target, "no_bake": no_bake})

    images = [
        p for p in Path(batch).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        logger.error(f"No supported images found in {batch}")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold]img2asset[/bold] — batch\n"
        f"Input folder: {batch}\n"
        f"Images found: {len(images)}",
        border_style="blue"
    ))

    results = []
    failed = []

    for i, image_path in enumerate(sorted(images), 1):
        console.rule(f"[{i}/{len(images)}] {image_path.name}")

        # Skip if final output already exists
        if skip_existing:
            expected = Path(cl_cfg.output_dir) / f"{image_path.stem}_final.glb"
            if expected.exists():
                logger.info(f"Skipping {image_path.name} (output exists)")
                continue

        try:
            result = run_pipeline(image_path, mv_cfg, re_cfg, cl_cfg, va_cfg)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {image_path.name} — {e}")
            failed.append((image_path, str(e)))
            if fail_fast:
                sys.exit(1)

    # Summary
    console.print(Panel.fit(
        f"[bold]Batch complete[/bold]\n"
        f"Processed: {len(results)}/{len(images)}\n"
        f"Failed:    {len(failed)}",
        border_style="green" if not failed else "yellow"
    ))

    if failed:
        console.print("[bold red]Failures:[/bold red]")
        for path, err in failed:
            console.print(f"  {path.name}: {err}")


if __name__ == "__main__":
    cli()
