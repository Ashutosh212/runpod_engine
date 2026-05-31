#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mandara — one-shot setup
# Works wherever the repo is cloned and on any CUDA server.
# Run once:  bash setup.sh
# Start:     bash start.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Project root: $PROJ"

# ── 1. Find or install conda ─────────────────────────────────────────────────
CONDA_BASE=""

if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

if [ -z "$CONDA_BASE" ]; then
    for candidate in /opt/miniconda /sfs/miniconda /workspace/miniconda \
                     "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$candidate"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ]; then
    for candidate in /sfs/miniconda /workspace/miniconda "$HOME/miniconda3"; do
        PARENT="$(dirname "$candidate")"
        if [ -d "$PARENT" ] && [ -w "$PARENT" ]; then
            INSTALL_TARGET="$candidate"
            break
        fi
    done
    INSTALL_TARGET="${INSTALL_TARGET:-$HOME/miniconda3}"
    echo "==> Installing Miniconda to $INSTALL_TARGET"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$INSTALL_TARGET"
    CONDA_BASE="$INSTALL_TARGET"
fi

echo "==> Using conda at: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 2. Create / update conda env ─────────────────────────────────────────────
# Stored under /workspace so envs survive pod resets
AI4RS_ENV="/workspace/envs/ai4rs_infer"
mkdir -p /workspace/envs

if [ -d "$AI4RS_ENV" ]; then
    echo "==> Updating existing env: $AI4RS_ENV"
    conda env update --prefix "$AI4RS_ENV" -f "$PROJ/environment.yml" --prune
else
    echo "==> Creating env: $AI4RS_ENV"
    conda env create -f "$PROJ/environment.yml" --prefix "$AI4RS_ENV"
fi

conda activate "$AI4RS_ENV"

# ── 3. Detect CUDA and install torch ─────────────────────────────────────────
CUDA_MAJOR=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
fi
echo "==> CUDA major version: ${CUDA_MAJOR:-none (CPU only)}"

if [ "${CUDA_MAJOR}" = "12" ]; then
    # cu124 torch is required for CUDA 12.x — cu121 has a prod() kernel bug on L40S/Ada Lovelace (sm_89)
    # mmcv has no cu124 build so we still use the cu121 mmcv wheel (CUDA backward compatible)
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html"
elif [ "${CUDA_MAJOR}" = "11" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html"
fi

pip install setuptools wheel
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url "$TORCH_INDEX"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"

# ── 4. Install mmengine, mmcv, mmdet, mmseg ───────────────────────────────────
pip install -U openmim
mim install mmengine

# Use pip directly for mmcv — mim generates wrong URL (torch2.4.0 vs torch2.4)
# --no-build-isolation avoids isolated-env pkg_resources missing error
pip install mmcv==2.2.0 -f "$MMCV_INDEX" --no-build-isolation

mim install 'mmdet>3.0.0rc6,<3.4.0'
mim install "mmsegmentation>=1.2.2"

# ── 5. Patch version caps (mmcv 2.2.0 is newer than these packages expect) ───
# Use pip show to locate files — do NOT import mmdet/mmseg, they raise
# AssertionError on import when mmcv 2.2.0 is present (before the patch).
python3 - <<'PYEOF'
import pathlib, subprocess, sys

def patch(path, old, new):
    if not path.exists():
        print(f"  Not found: {path}")
        return
    txt = path.read_text()
    if old in txt:
        path.write_text(txt.replace(old, new))
        print(f"  Patched: {path}")
    else:
        print(f"  Already OK: {path}")

def site_location(pkg):
    r = subprocess.run([sys.executable, '-m', 'pip', 'show', pkg],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('Location:'):
            return line.split(':', 1)[1].strip()
    return None

loc = site_location('mmdet')
if loc:
    patch(pathlib.Path(loc) / 'mmdet' / '__init__.py',
          "mmcv_maximum_version = '2.2.0'", "mmcv_maximum_version = '2.3.0'")
else:
    print("  mmdet not found via pip show, skipping")

loc = site_location('mmsegmentation')
if loc:
    patch(pathlib.Path(loc) / 'mmseg' / '__init__.py',
          "MMCV_MAX = '2.2.0'", "MMCV_MAX = '2.3.0'")
else:
    print("  mmseg not found via pip show, skipping")
PYEOF

# ── 6. Clone and install mmrotate 1.x ────────────────────────────────────────
# Persist alongside the project so it survives pod resets
MMROTATE_DIR="$(dirname "$PROJ")/mmrotate_runpod"
if [ ! -d "$MMROTATE_DIR/.git" ]; then
    echo "==> Cloning mmrotate 1.x to $MMROTATE_DIR"
    git clone -b 1.x https://github.com/open-mmlab/mmrotate.git "$MMROTATE_DIR"
else
    echo "==> mmrotate already cloned at $MMROTATE_DIR"
fi

# --no-deps because mmrotate requires mmcv-full (old name) — we have mmcv 2.x already
pip install "$MMROTATE_DIR" --no-build-isolation --no-deps

# Patch mmrotate version caps (same issue — import raises AssertionError before patch)
python3 - <<'PYEOF'
import pathlib, subprocess, sys

def site_location(pkg):
    r = subprocess.run([sys.executable, '-m', 'pip', 'show', pkg],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('Location:'):
            return line.split(':', 1)[1].strip()
    return None

loc = site_location('mmrotate')
if loc:
    p = pathlib.Path(loc) / 'mmrotate' / '__init__.py'
    if p.exists():
        txt = p.read_text()
        txt = txt.replace("mmcv_maximum_version = '2.1.0'", "mmcv_maximum_version = '2.3.0'")
        txt = txt.replace("mmdet_maximum_version = '3.1.0'", "mmdet_maximum_version = '3.4.0'")
        p.write_text(txt)
        print(f"  Patched: {p}")
        # Clear stale .pyc so Python recompiles from the patched source
        import shutil
        pycache = p.parent / '__pycache__'
        if pycache.exists():
            shutil.rmtree(pycache)
            print(f"  Cleared pycache: {pycache}")
    else:
        print(f"  mmrotate __init__.py not found at {p}")
else:
    print("  mmrotate not found via pip show")
PYEOF

# ── 7. Create missing __init__.py so mmengine can import custom modules ───────
touch "$PROJ/projects/__init__.py"
touch "$PROJ/projects/OrientedFormer/__init__.py"
echo "==> Created projects/__init__.py and projects/OrientedFormer/__init__.py"

# ── 8. Install FastAPI server dependencies ───────────────────────────────────
pip install "fastapi==0.111.0" "uvicorn[standard]==0.30.0" \
    pydantic pillow numpy python-multipart

# ── 9. Set up DEIMv2 conda env (for the comparison model) ────────────────────
# All DEIMv2 assets are bundled inside this repo — no external dependency.
DEIMV2_SRC="$PROJ/deim_src"
DEIMV2_CKPT="$PROJ/checkpoints/deim/best_stg1.pth"

if [ ! -d "$DEIMV2_SRC/engine" ]; then
    echo "==> WARNING: $DEIMV2_SRC/engine not found. DEIMv2 inference will not work."
    echo "    Ensure deim_src/engine/ is present in the repo."
elif [ ! -f "$DEIMV2_CKPT" ]; then
    echo "==> WARNING: DEIMv2 checkpoint not found at $DEIMV2_CKPT"
    echo "    Copy best_stg1.pth to checkpoints/deim/ (it is .gitignored)."
else
    echo "==> DEIMv2 source: $DEIMV2_SRC"
    echo "==> DEIMv2 checkpoint: $DEIMV2_CKPT"

    DEIMV2_ENV="/workspace/envs/deimv2"

    if [ -d "$DEIMV2_ENV" ]; then
        echo "==> DEIMv2 conda env already exists at $DEIMV2_ENV"
    else
        echo "==> Creating DEIMv2 conda env (Python 3.11) at $DEIMV2_ENV"
        conda create --prefix "$DEIMV2_ENV" python=3.11 -y
        conda run --prefix "$DEIMV2_ENV" pip install setuptools wheel

        if [ "${CUDA_MAJOR}" = "12" ]; then
            DEIM_TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        elif [ "${CUDA_MAJOR}" = "11" ]; then
            DEIM_TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        else
            DEIM_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        fi

        conda run --prefix "$DEIMV2_ENV" pip install torch torchvision --index-url "$DEIM_TORCH_INDEX"
        # Install all DEIMv2 runtime deps from bundled requirements.txt
        # (excludes torch/torchvision which are already installed above)
        conda run --prefix "$DEIMV2_ENV" pip install \
            pyyaml pillow numpy scipy tensorboard \
            faster-coco-eval calflops transformers
    fi
    echo "==> DEIMv2 env ready"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                             ║"
echo "║                                                              ║"
printf "║  Start server:  bash %-39s║\n" "$PROJ/start.sh"
echo "╚══════════════════════════════════════════════════════════════╝"
