# img2asset

Convert a single reference image into a game-ready 3D asset — fully automated.

```
reference image → Wonder3D → InstantMesh → Blender → game-ready .glb
```

---

## Pipeline

```
Stage 1  multiview.py    Wonder3D generates 6 consistent views
Stage 2  reconstruct.py  InstantMesh builds a raw 3D mesh from those views
Stage 3  cleanup.py      Blender decimates, smooths, unwraps UVs, bakes texture
Stage 4  validate.py     Checks mesh quality and renders a turntable preview
```

---

## Hardware requirements

| Level       | GPU              | RAM   | Storage |
|-------------|------------------|-------|---------|
| Minimum     | RTX 3080 12GB    | 32GB  | ~50GB   |
| Comfortable | RTX 4090         | 64GB  | ~100GB  |

> Wonder3D is the most VRAM-heavy stage (~10GB peak).

---

## Setup

### 1. Clone external dependencies

```bash
# Wonder3D
git clone https://github.com/xxlong0/Wonder3D
cd Wonder3D && pip install -r requirements.txt && cd ..

# InstantMesh
git clone https://github.com/TencentARC/InstantMesh
cd InstantMesh && pip install -r requirements.txt && cd ..

# Add both to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:./Wonder3D:./InstantMesh/src"
```

### 2. Install pipeline dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Blender

Download from https://www.blender.org/download/ (4.x recommended).
Make sure `blender` is in your PATH, or set `blender_bin` in `config/default.yaml`.

### 4. (Optional) Background removal

```bash
pip install rembg
```

---

## Usage

### Single image

```bash
python run.py single --image path/to/reference.png
```

### Single image with overrides

```bash
python run.py single \
  --image ref.png \
  --poly-target 5000 \
  --no-bake
```

### Batch (folder of images)

```bash
python run.py batch --batch ./references/
```

---

## Configuration

Edit `config/default.yaml` to tune each stage:

```yaml
multiview:
  num_views: 6          # views generated around the object
  remove_background: true

reconstruct:
  mesh_format: glb

cleanup:
  target_poly_count: 8000   # game-ready target
  bake_texture: true
  texture_size: 1024

validation:
  render_preview: true      # turntable preview frames
```

---

## Output structure

```
outputs/
├── views/
│   └── my_ref/
│       ├── color_00.png ... color_05.png
│       └── normal_00.png ... normal_05.png
├── meshes/
│   └── my_ref.glb          ← raw InstantMesh output
└── final/
    ├── my_ref_final.glb    ← game-ready asset
    └── my_ref_final_preview/
        └── frame_00.png ... frame_07.png
```

---

## Known limitations

- **Multi-view consistency** — still not perfect for complex shapes. Wonder3D handles it well for most objects but may struggle with thin/transparent parts.
- **Topology** — AI-generated meshes need Blender cleanup. The pipeline handles this automatically but very complex shapes may still need manual retopo.
- **Texture quality** — baked textures capture diffuse color only. PBR materials (roughness, metallic, normals) require additional passes not yet implemented.

---

## Roadmap

- [ ] PBR texture baking (roughness, metallic, normal maps)
- [ ] Retopology via instant-meshes or QuadRemesher
- [ ] Web UI for non-technical users
- [ ] Game engine export presets (Unity, Unreal, Godot)
