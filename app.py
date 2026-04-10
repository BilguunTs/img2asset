"""
app.py — img2asset web UI

Run:
    python app.py
    python app.py --share      # public shareable link (useful for remote GPU)
    python app.py --port 7860
"""

import argparse
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import yaml
from loguru import logger

from modules.backend import detect, select_pipeline, log_backend_info
from modules.reconstruct_triposr import TripoSRReconstructor, TripoSRConfig
from modules.multiview import MultiviewGenerator, MultiviewConfig
from modules.multiview_zero123 import Zero123MultiviewGenerator, Zero123Config
from modules.reconstruct import MeshReconstructor, ReconstructConfig
from modules.reconstruct_crm import CRMReconstructor, CRMConfig
from modules.cleanup import BlenderCleanup, CleanupConfig
from modules.validate import MeshValidator, ValidationConfig

# ─── Load default config ──────────────────────────────────────────────────────

with open("config/default.yaml") as f:
    DEFAULT_CFG = yaml.safe_load(f)

BACKEND  = detect()
MODE     = select_pipeline(BACKEND)
log_backend_info(BACKEND, MODE)

BACKEND_LABEL = {
    "cuda": "CUDA (Full pipeline — Wonder3D → InstantMesh)",
    "mps":  "Apple Silicon MPS (Zero123++ → CRM)",
    "cpu":  "CPU (TripoSR — slow)",
}.get(BACKEND, BACKEND)


# ─── Core pipeline function ───────────────────────────────────────────────────

def run(
    image,
    force_backend,
    poly_target,
    smooth_iterations,
    bake_texture,
    texture_size,
    mc_resolution,
    remove_background,
):
    if image is None:
        raise gr.Error("Please upload a reference image.")

    mode = force_backend if force_backend != "auto" else MODE

    # Build configs directly from UI values — no YAML layering
    cl_cfg = CleanupConfig(
        blender_bin=DEFAULT_CFG.get("cleanup", {}).get("blender_bin", "blender"),
        target_poly_count=int(poly_target),
        smooth_iterations=int(smooth_iterations),
        fill_holes=True,
        unwrap_uvs=True,
        bake_texture=bake_texture,
        texture_size=int(texture_size),
        output_dir=DEFAULT_CFG.get("cleanup", {}).get("output_dir", "outputs/final"),
    )
    va_cfg = ValidationConfig(
        enabled=True,
        check_manifold=True,
        check_holes=True,
        max_poly_warn=15000,
        render_preview=False,   # skip turntable in web UI for speed
    )
    tsr_cfg = TripoSRConfig(
        mc_resolution=int(mc_resolution),
        remove_background=remove_background,
        output_dir=DEFAULT_CFG.get("triposr", {}).get("output_dir", "outputs/meshes"),
    )
    mv_cfg = MultiviewConfig(
        remove_background=remove_background,
        output_dir=DEFAULT_CFG.get("multiview", {}).get("output_dir", "outputs/views"),
    )
    z123_cfg = Zero123Config(
        remove_background=remove_background,
        output_dir=DEFAULT_CFG.get("multiview", {}).get("output_dir", "outputs/views"),
    )
    crm_cfg = CRMConfig(
        mc_resolution=int(mc_resolution),
        output_dir=DEFAULT_CFG.get("reconstruct", {}).get("output_dir", "outputs/meshes"),
    )

    # Save uploaded image to a temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        from PIL import Image as PILImage
        PILImage.fromarray(image).save(tmp.name)
        image_path = Path(tmp.name)

    name = "output"
    logs = []

    try:
        if mode == "triposr":
            yield None, None, "Running TripoSR reconstruction...", ""

            raw_mesh = TripoSRReconstructor(tsr_cfg).reconstruct(str(image_path), name=name)
            logs.append("TripoSR reconstruction done.")

        elif mode == "zero123":
            yield None, None, "Generating multi-view images (Zero123++)...", ""

            multiview_result = Zero123MultiviewGenerator(z123_cfg).generate(str(image_path))
            logs.append(f"Generated {len(multiview_result['color_views'])} views with Zero123++.")

            yield None, None, "Reconstructing mesh (CRM)...", "\n".join(logs)

            raw_mesh = CRMReconstructor(crm_cfg).reconstruct(multiview_result, name=name)
            logs.append("CRM reconstruction done.")

        else:
            yield None, None, "Generating multi-view images (Wonder3D)...", ""

            multiview_result = MultiviewGenerator(mv_cfg).generate(str(image_path))
            logs.append(f"Generated {len(multiview_result['color_views'])} views.")

            yield None, None, "Reconstructing mesh (InstantMesh)...", "\n".join(logs)

            re_cfg = ReconstructConfig(
                output_dir=DEFAULT_CFG.get("reconstruct", {}).get("output_dir", "outputs/meshes")
            )
            raw_mesh = MeshReconstructor(re_cfg).reconstruct(multiview_result, name=name)
            logs.append("InstantMesh reconstruction done.")

        yield None, None, "Running Blender cleanup...", "\n".join(logs)

        clean_mesh = BlenderCleanup(cl_cfg).clean(raw_mesh, name=f"{name}_final")
        logs.append("Blender cleanup done.")

        yield None, None, "Validating mesh...", "\n".join(logs)

        report = MeshValidator(va_cfg).validate(clean_mesh)
        logs.append(
            f"Validation: {report.poly_count} polys | "
            f"manifold={report.is_manifold} | holes={report.has_holes}"
        )
        for w in report.warnings:
            logs.append(f"Warning: {w}")

        summary = (
            f"Poly count:   {report.poly_count}\n"
            f"Manifold:     {report.is_manifold}\n"
            f"Has holes:    {report.has_holes}\n"
            f"Warnings:     {len(report.warnings)}"
        )

        yield str(clean_mesh), str(clean_mesh), summary, "\n".join(logs)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(tb)
        raise gr.Error(f"{type(e).__name__}: {e}")


# ─── UI ───────────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 0.95em; }
#backend-badge { font-size: 0.85em; padding: 6px 12px; border-radius: 8px; background: #1e1e2e; color: #cba6f7; text-align: center; margin-bottom: 16px; }
#run-btn { width: 100%; }
.log-box textarea { font-family: monospace; font-size: 0.82em; }
"""

with gr.Blocks(title="img2asset") as demo:

    gr.Markdown("# img2asset", elem_id="title")
    gr.Markdown("Image → Game-Ready 3D Asset", elem_id="subtitle")
    gr.Markdown(f"**Backend:** {BACKEND_LABEL}", elem_id="backend-badge")

    with gr.Row():

        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Reference Image",
                type="numpy",
                image_mode="RGB",
                height=280,
            )

            with gr.Accordion("Settings", open=False):
                force_backend = gr.Radio(
                    choices=["auto", "triposr", "zero123", "full"],
                    value="auto",
                    label="Pipeline",
                    info="'auto' picks best for your GPU. 'zero123' = Zero123++→CRM (MPS). 'full' requires CUDA.",
                )
                remove_background = gr.Checkbox(
                    value=True,
                    label="Remove background (rembg)",
                )

                gr.Markdown("**Reconstruction**")
                mc_resolution = gr.Slider(
                    minimum=64, maximum=512, step=64, value=256,
                    label="Marching cubes resolution",
                    info="Higher = more detail, more VRAM. TripoSR only.",
                )

                gr.Markdown("**Cleanup**")
                poly_target = gr.Slider(
                    minimum=1000, maximum=50000, step=500, value=8000,
                    label="Target poly count",
                )
                smooth_iterations = gr.Slider(
                    minimum=0, maximum=10, step=1, value=2,
                    label="Smooth iterations",
                )
                bake_texture = gr.Checkbox(value=True, label="Bake texture")
                texture_size = gr.Radio(
                    choices=[512, 1024, 2048],
                    value=1024,
                    label="Texture size (px)",
                )

            run_btn = gr.Button("Generate 3D Asset", variant="primary", elem_id="run-btn")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=1):
            model_viewer = gr.Model3D(
                label="3D Preview",
                height=380,
                camera_position=(0, 70, 4),
            )
            download_btn = gr.File(label="Download .glb", visible=True)

            with gr.Accordion("Quality Report", open=True):
                summary_box = gr.Textbox(
                    label="", lines=5, interactive=False
                )

            with gr.Accordion("Logs", open=False):
                log_box = gr.Textbox(
                    label="", lines=8, interactive=False,
                    elem_classes=["log-box"]
                )

    status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    # ── Wire up ───────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run,
        inputs=[
            image_input,
            force_backend,
            poly_target,
            smooth_iterations,
            bake_texture,
            texture_size,
            mc_resolution,
            remove_background,
        ],
        outputs=[model_viewer, download_btn, summary_box, log_box],
        show_progress="minimal",
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   type=str, default="127.0.0.1")
    args = parser.parse_args()

    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
