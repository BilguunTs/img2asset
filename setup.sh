#!/usr/bin/env bash
# setup.sh — img2asset environment setup
# Run once before using the pipeline.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[setup]${NC} $1"; }
success() { echo -e "${GREEN}[done]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $1"; }
error()   { echo -e "${RED}[error]${NC} $1"; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$ROOT/deps"
mkdir -p "$DEPS_DIR"

echo ""
echo "  img2asset — setup"
echo "  ─────────────────────────────────────"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────

info "Checking Python..."
PYTHON=$(command -v python3 || command -v python || error "Python not found. Install Python 3.10+")
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  error "Python 3.10+ required (found $PY_VERSION)"
fi
success "Python $PY_VERSION"

# ── 2. CUDA check ─────────────────────────────────────────────────────────────

info "Checking CUDA..."
if command -v nvidia-smi &>/dev/null; then
  CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  success "NVIDIA GPU detected (driver $CUDA_VERSION)"
else
  warn "nvidia-smi not found — GPU acceleration may not work."
  warn "Pipeline will be very slow on CPU only."
fi

# ── 3. Blender check ──────────────────────────────────────────────────────────

info "Checking Blender..."
if command -v blender &>/dev/null; then
  BLENDER_VERSION=$(blender --version 2>/dev/null | head -1)
  success "$BLENDER_VERSION"
else
  warn "Blender not found in PATH."
  warn "Download from https://www.blender.org/download/ and add to PATH,"
  warn "or set 'blender_bin' in config/default.yaml to the full path."
fi

# ── 4. Pipeline Python dependencies ───────────────────────────────────────────

info "Installing pipeline dependencies..."
"$PYTHON" -m pip install --upgrade pip -q
"$PYTHON" -m pip install -r "$ROOT/requirements.txt" -q
success "Pipeline dependencies installed"

# ── 5. Optional: rembg for background removal ─────────────────────────────────

info "Installing rembg (background removal)..."
"$PYTHON" -m pip install rembg -q && success "rembg installed" || warn "rembg failed to install — background removal will be skipped"

# ── 6. Detect backend ─────────────────────────────────────────────────────────

info "Detecting compute backend..."
BACKEND=$("$PYTHON" - <<'EOF'
import torch
if torch.cuda.is_available():
    print("cuda")
elif torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
EOF
)
success "Backend: $BACKEND"

# ── 7. TripoSR (always installed — works on MPS/CPU/CUDA) ─────────────────────

info "Setting up TripoSR..."
if [ -d "$DEPS_DIR/TripoSR" ]; then
  info "TripoSR already cloned, pulling latest..."
  git -C "$DEPS_DIR/TripoSR" pull -q
else
  git clone https://github.com/VAST-AI-Research/TripoSR "$DEPS_DIR/TripoSR" -q
fi
"$PYTHON" -m pip install -r "$DEPS_DIR/TripoSR/requirements.txt" -q
success "TripoSR ready"

info "Downloading TripoSR weights..."
"$PYTHON" - <<'EOF'
from huggingface_hub import snapshot_download
import os
cache = os.path.join(os.environ.get("ROOT", "."), "deps", "weights", "triposr")
os.makedirs(cache, exist_ok=True)
snapshot_download(repo_id="stabilityai/TripoSR", local_dir=cache)
print("  TripoSR weights downloaded.")
EOF
success "TripoSR weights ready"

# ── 8. CUDA-only: Wonder3D + InstantMesh ──────────────────────────────────────

if [ "$BACKEND" = "cuda" ]; then
  info "CUDA detected — installing full pipeline (Wonder3D + InstantMesh)..."

  if [ -d "$DEPS_DIR/Wonder3D" ]; then
    git -C "$DEPS_DIR/Wonder3D" pull -q
  else
    git clone https://github.com/xxlong0/Wonder3D "$DEPS_DIR/Wonder3D" -q
  fi
  "$PYTHON" -m pip install -r "$DEPS_DIR/Wonder3D/requirements.txt" -q
  success "Wonder3D ready"

  if [ -d "$DEPS_DIR/InstantMesh" ]; then
    git -C "$DEPS_DIR/InstantMesh" pull -q
  else
    git clone https://github.com/TencentARC/InstantMesh "$DEPS_DIR/InstantMesh" -q
  fi
  "$PYTHON" -m pip install -r "$DEPS_DIR/InstantMesh/requirements.txt" -q
  success "InstantMesh ready"

  info "Downloading Wonder3D weights..."
  "$PYTHON" - <<'EOF'
from huggingface_hub import snapshot_download
import os
cache = os.path.join(os.environ.get("ROOT", "."), "deps", "weights", "wonder3d")
os.makedirs(cache, exist_ok=True)
snapshot_download(repo_id="flamehaze1115/wonder3d-v1.0", local_dir=cache,
                  ignore_patterns=["*.bin.index.json"])
print("  Wonder3D weights downloaded.")
EOF
  success "Wonder3D weights ready"

  info "Downloading InstantMesh weights..."
  "$PYTHON" - <<'EOF'
from huggingface_hub import hf_hub_download
import os
cache = os.path.join(os.environ.get("ROOT", "."), "deps", "weights", "instantmesh")
os.makedirs(cache, exist_ok=True)
hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instantmesh-large.ckpt",
                local_dir=cache)
print("  InstantMesh weights downloaded.")
EOF
  success "InstantMesh weights ready"

else
  warn "Skipping Wonder3D + InstantMesh (CUDA required)."
  warn "Pipeline will use TripoSR automatically on $BACKEND."
fi

# ── 9. Write .env with PYTHONPATH ─────────────────────────────────────────────

info "Writing .env..."
PYTHONPATH_EXTRA="$DEPS_DIR/TripoSR"
if [ "$BACKEND" = "cuda" ]; then
  PYTHONPATH_EXTRA="$DEPS_DIR/Wonder3D:$DEPS_DIR/InstantMesh/src:$PYTHONPATH_EXTRA"
fi

cat > "$ROOT/.env" <<EOF
# Auto-generated by setup.sh — backend: $BACKEND
export PYTHONPATH="$PYTHONPATH_EXTRA:\$PYTHONPATH"
export TRIPOSR_MODEL_PATH="$DEPS_DIR/weights/triposr"
export WONDER3D_MODEL_PATH="$DEPS_DIR/weights/wonder3d"
export INSTANTMESH_MODEL_PATH="$DEPS_DIR/weights/instantmesh"
export ROOT="$ROOT"
EOF
success ".env written"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}  Setup complete!${NC}"
echo ""
if [ "$BACKEND" = "cuda" ]; then
  echo "  Full pipeline available (Wonder3D → InstantMesh → Blender)"
else
  echo "  TripoSR pipeline ready (single-image → Blender)"
  echo "  Note: Multi-view quality requires a CUDA GPU."
fi
echo ""
echo "  To use:"
echo "    source .env"
echo "    python run.py info                          # confirm backend"
echo "    python run.py single --image path/to/ref.png"
echo "    python run.py batch  --batch ./references/"
echo ""
