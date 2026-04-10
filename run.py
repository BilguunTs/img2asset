"""
run.py — img2asset pipeline entrypoint

Automatically selects pipeline based on available hardware:
  CUDA  → Wonder3D → InstantMesh → Blender → Validate  (full quality)
  MPS   → TripoSR  → Blender → Validate                (Apple Silicon)
  CPU   → TripoSR  → Blender → Validate                (slow fallback)

Usage:
    python run.py single --image ref.png
    python run.py single --image ref.png --backend triposr   # force TripoSR
    python run.py batch  --batch ./references/
    python run.py info                                        # show detected backend
"""

import sys
from pathlib import Path

import click
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from modules.backend import detect, select_pipeline, log_backend_info
from modules.multiview import MultiviewGenerator, MultiviewConfig
from modules.reconstruct import MeshReconstructor, ReconstructConfig
from modules.reconstruct_triposr import TripoSRReconstructor, TripoSRConfig
from modules.cleanup import BlenderCleanup, CleanupConfig
from modules.validate import MeshValidator, ValidationConfig

console = Console()

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_configs(cfg: dict, overrides: dict):
    mv_cfg  = cfg.get("multiview",  {})
    re_cfg  = cfg.get("reconstruct", {})
    tsr_cfg = cfg.get("triposr",    {})
    cl_cfg  = cfg.get("cleanup",    {})
    va_cfg  = cfg.get("validation", {})

    if overrides.get("poly_target"):
        cl_cfg["target_poly_count"] = overrides["poly_target"]
    if overrides.get("no_bake"):
        cl_cfg["bake_texture"] = False

    return (
        MultiviewConfig(**mv_cfg),
        ReconstructConfig(**re_cfg),
        TripoSRConfig(**tsr_cfg),
        CleanupConfig(**cl_cfg),
        ValidationConfig(**va_cfg),
    )


# ─── Pipeline runners ─────────────────────────────────────────────────────────

def run_full_pipeline(image_path: Path, mv_cfg, re_cfg, cl_cfg, va_cfg) -> dict:
    """CUDA path: Wonder3D → InstantMesh → Blender → Validate"""
    name = image_path.stem

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:

        t = p.add_task(f"[cyan]1/4  Generating views ({name})...", total=None)
        multiview_result = MultiviewGenerator(mv_cfg).generate(str(image_path))
        p.update(t, description=f"[green]1/4  {len(multiview_result['color_views'])} views generated")

        t = p.add_task("[cyan]2/4  Reconstructing mesh (InstantMesh)...", total=None)
        raw_mesh = MeshReconstructor(re_cfg).reconstruct(multiview_result, name=name)
        p.update(t, description="[green]2/4  Mesh reconstructed")

        t = p.add_task("[cyan]3/4  Blender cleanup...", total=None)
        clean_mesh = BlenderCleanup(cl_cfg).clean(raw_mesh, name=f"{name}_final")
        p.update(t, description="[green]3/4  Cleanup done")

        t = p.add_task("[cyan]4/4  Validating...", total=None)
        report = MeshValidator(va_cfg).validate(clean_mesh)
        p.update(t, description=f"[green]4/4  Validation {'passed' if report.passed else 'done (warnings)'}")

    return {"name": name, "input": image_path, "views": multiview_result["output_dir"],
            "raw_mesh": raw_mesh, "final_mesh": clean_mesh, "report": report}


def run_triposr_pipeline(image_path: Path, tsr_cfg, cl_cfg, va_cfg) -> dict:
    """MPS/CPU path: TripoSR → Blender → Validate"""
    name = image_path.stem

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as p:

        t = p.add_task(f"[cyan]1/3  Reconstructing with TripoSR ({name})...", total=None)
        raw_mesh = TripoSRReconstructor(tsr_cfg).reconstruct(str(image_path), name=name)
        p.update(t, description="[green]1/3  TripoSR mesh done")

        t = p.add_task("[cyan]2/3  Blender cleanup...", total=None)
        clean_mesh = BlenderCleanup(cl_cfg).clean(raw_mesh, name=f"{name}_final")
        p.update(t, description="[green]2/3  Cleanup done")

        t = p.add_task("[cyan]3/3  Validating...", total=None)
        report = MeshValidator(va_cfg).validate(clean_mesh)
        p.update(t, description=f"[green]3/3  Validation {'passed' if report.passed else 'done (warnings)'}")

    return {"name": name, "input": image_path, "views": None,
            "raw_mesh": raw_mesh, "final_mesh": clean_mesh, "report": report}


def run_pipeline(image_path: Path, mode: str, mv_cfg, re_cfg, tsr_cfg, cl_cfg, va_cfg) -> dict:
    if mode == "full":
        return run_full_pipeline(image_path, mv_cfg, re_cfg, cl_cfg, va_cfg)
    return run_triposr_pipeline(image_path, tsr_cfg, cl_cfg, va_cfg)


def print_result(result: dict):
    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n"
        f"Final asset: [cyan]{result['final_mesh']}[/cyan]\n"
        f"Poly count:  {result['report'].poly_count}\n"
        f"Manifold:    {result['report'].is_manifold}",
        border_style="green"
    ))


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def info():
    """Show detected backend and pipeline mode."""
    backend = detect()
    mode = select_pipeline(backend)
    log_backend_info(backend, mode)
    console.print(Panel.fit(
        f"Backend:  [bold]{backend.upper()}[/bold]\n"
        f"Pipeline: [bold]{'Full (Wonder3D → InstantMesh)' if mode == 'full' else 'TripoSR fallback'}[/bold]",
        border_style="blue"
    ))


@cli.command()
@click.option("--image",       required=True, type=click.Path(exists=True))
@click.option("--config",      default="config/default.yaml", show_default=True)
@click.option("--poly-target", type=int,      default=None,  help="Override poly count target")
@click.option("--no-bake",     is_flag=True,  default=False, help="Skip texture baking")
@click.option("--backend",     type=click.Choice(["auto", "full", "triposr"]), default="auto",
              show_default=True, help="Force pipeline backend")
def single(image, config, poly_target, no_bake, backend):
    """Run the pipeline on a single image."""
    cfg = load_config(config)
    mv_cfg, re_cfg, tsr_cfg, cl_cfg, va_cfg = build_configs(
        cfg, {"poly_target": poly_target, "no_bake": no_bake}
    )

    detected = detect()
    mode = backend if backend != "auto" else select_pipeline(detected)
    log_backend_info(detected, mode)

    console.print(Panel.fit(
        f"[bold]img2asset[/bold] — single image\n"
        f"Input:    {image}\n"
        f"Backend:  {detected.upper()}  →  {'Full pipeline' if mode == 'full' else 'TripoSR'}",
        border_style="blue"
    ))

    result = run_pipeline(Path(image), mode, mv_cfg, re_cfg, tsr_cfg, cl_cfg, va_cfg)
    print_result(result)


@cli.command()
@click.option("--batch",       required=True, type=click.Path(exists=True))
@click.option("--config",      default="config/default.yaml", show_default=True)
@click.option("--poly-target", type=int,      default=None)
@click.option("--no-bake",     is_flag=True,  default=False)
@click.option("--fail-fast",   is_flag=True,  default=False)
@click.option("--backend",     type=click.Choice(["auto", "full", "triposr"]), default="auto",
              show_default=True)
def batch(batch, config, poly_target, no_bake, fail_fast, backend):
    """Run the pipeline on a folder of images."""
    cfg = load_config(config)
    batch_cfg = cfg.get("batch", {})
    skip_existing = batch_cfg.get("skip_existing", True)
    mv_cfg, re_cfg, tsr_cfg, cl_cfg, va_cfg = build_configs(
        cfg, {"poly_target": poly_target, "no_bake": no_bake}
    )

    detected = detect()
    mode = backend if backend != "auto" else select_pipeline(detected)
    log_backend_info(detected, mode)

    images = sorted(
        p for p in Path(batch).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        logger.error(f"No supported images in {batch}")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold]img2asset[/bold] — batch\n"
        f"Folder:  {batch}\n"
        f"Images:  {len(images)}\n"
        f"Backend: {detected.upper()}  →  {'Full pipeline' if mode == 'full' else 'TripoSR'}",
        border_style="blue"
    ))

    results, failed = [], []

    for i, image_path in enumerate(images, 1):
        console.rule(f"[{i}/{len(images)}] {image_path.name}")

        if skip_existing:
            expected = Path(cl_cfg.output_dir) / f"{image_path.stem}_final.glb"
            if expected.exists():
                logger.info(f"Skipping {image_path.name} (output exists)")
                continue

        try:
            result = run_pipeline(image_path, mode, mv_cfg, re_cfg, tsr_cfg, cl_cfg, va_cfg)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {image_path.name} — {e}")
            failed.append((image_path, str(e)))
            if fail_fast:
                sys.exit(1)

    console.print(Panel.fit(
        f"[bold]Batch complete[/bold]\n"
        f"Processed: {len(results)}/{len(images)}\n"
        f"Failed:    {len(failed)}",
        border_style="green" if not failed else "yellow"
    ))
    for path, err in failed:
        console.print(f"  [red]{path.name}[/red]: {err}")


if __name__ == "__main__":
    cli()
