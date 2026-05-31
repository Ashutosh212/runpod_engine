"""DEIMv2 tiled inference — called from app.py.

Finds the deimv2 conda env and DEIMv2 project automatically,
tiles the image using the same 1280×720 / 200px-overlap grid as patcher.py,
runs deim_infer_helper.py in one subprocess call (model loaded once),
and returns stitched DOTA-format predictions.
"""
import json
import os
import shutil
import subprocess
import tempfile

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))

PATCH_W = 1280
PATCH_H = 720
OVERLAP = 200
STEP_W  = PATCH_W - OVERLAP   # 1080
STEP_H  = PATCH_H - OVERLAP   # 520

# ── Paths — all internal to this folder ──────────────────────────────────────

def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# Source code: always use the bundled copy inside this repo
DEIMV2_ROOT = os.path.join(_HERE, 'deim_src')

# Config and checkpoint: bundled inside this repo
DEIMV2_CONFIG     = os.path.join(_HERE, 'configs', 'deim', 'deimv2_dinov3_s_vehicle.yml')
DEIMV2_CHECKPOINT = os.path.join(_HERE, 'checkpoints', 'deim', 'best_stg1.pth')

def _find_deimv2_python() -> str | None:
    """Find the deimv2 conda env's python3 interpreter.

    Tries a static list of common conda env locations first (fast path),
    then falls back to asking conda directly so it works on any server
    regardless of where conda chose to install the env.
    """
    # Fast path: common install locations
    static = _first_existing(
        '/sfs/envs/deimv2/bin/python3',
        '/opt/miniconda/envs/deimv2/bin/python3',
        '/sfs/miniconda3/envs/deimv2/bin/python3',
        '/workspace/envs/deimv2/bin/python3',
        os.path.expanduser('~/miniconda3/envs/deimv2/bin/python3'),
        os.path.expanduser('~/miniconda/envs/deimv2/bin/python3'),
        os.path.expanduser('~/envs/deimv2/bin/python3'),
    )
    if static:
        return static

    # Fallback: ask conda where the env lives
    conda_exe = shutil.which('conda')
    if not conda_exe:
        # Search common conda binary locations
        conda_exe = _first_existing(
            '/opt/miniconda/bin/conda',
            '/sfs/miniconda3/bin/conda',
            '/workspace/miniconda/bin/conda',
            os.path.expanduser('~/miniconda3/bin/conda'),
            os.path.expanduser('~/miniconda/bin/conda'),
        )
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, 'run', '-n', 'deimv2', 'which', 'python3'],
                capture_output=True, text=True, timeout=15,
            )
            p = result.stdout.strip()
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    return None


DEIMV2_PYTHON = _find_deimv2_python()

HELPER_SCRIPT = os.path.join(_HERE, 'deim_infer_helper.py')


def is_available() -> bool:
    return bool(
        DEIMV2_PYTHON and os.path.isfile(DEIMV2_PYTHON)
        and DEIMV2_ROOT and os.path.isdir(DEIMV2_ROOT)
        and DEIMV2_CONFIG and os.path.isfile(DEIMV2_CONFIG)
        and DEIMV2_CHECKPOINT and os.path.isfile(DEIMV2_CHECKPOINT)
    )


# ── Tiling ───────────────────────────────────────────────────────────────────

def _tile_origins(img_w: int, img_h: int) -> list[tuple[int, int]]:
    def axis(size, step, crop):
        if size <= crop:
            return [0]
        coords = list(range(0, size - crop, step))
        last = size - crop
        if not coords or coords[-1] < last:
            coords.append(last)
        return sorted(set(max(0, c) for c in coords))

    xs = axis(img_w, STEP_W, PATCH_W)
    ys = axis(img_h, STEP_H, PATCH_H)
    return [(x, y) for y in ys for x in xs]


# ── Main inference call ───────────────────────────────────────────────────────

def predict_deim_patched(image_path: str, score_thr: float = 0.05) -> tuple[str, list, tuple]:
    """Tile the image, run DEIMv2 on all tiles in one subprocess, stitch.

    Returns:
        stitched_dota  — full-image DOTA prediction string
        origins        — [[ox, oy], ...] tile top-left corners
        (img_w, img_h) — original image dimensions
    """
    if not is_available():
        raise RuntimeError(
            'DEIMv2 not available. '
            f'Python: {DEIMV2_PYTHON}, Root: {DEIMV2_ROOT}, '
            f'Config: {DEIMV2_CONFIG}, Checkpoint: {DEIMV2_CHECKPOINT}'
        )

    img    = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size
    origins = _tile_origins(img_w, img_h)

    tmpdir = tempfile.mkdtemp(prefix='deim_tiles_')
    try:
        # Save all tiles
        tile_entries = []
        for i, (ox, oy) in enumerate(origins):
            x1 = min(ox + PATCH_W, img_w)
            y1 = min(oy + PATCH_H, img_h)
            tile = img.crop((ox, oy, x1, y1))
            tile_path = os.path.join(tmpdir, f'tile_{i:04d}.png')
            tile.save(tile_path)
            tile_entries.append({'path': tile_path, 'ox': ox, 'oy': oy})

        # Write manifest
        manifest = {
            'config':     DEIMV2_CONFIG,
            'checkpoint': DEIMV2_CHECKPOINT,
            'proj_root':  DEIMV2_ROOT,
            'threshold':  score_thr,
            'tiles':      tile_entries,
        }
        manifest_path = os.path.join(tmpdir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Run helper — model loaded once, all tiles processed
        cmd = [DEIMV2_PYTHON, HELPER_SCRIPT, '--manifest', manifest_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            raise RuntimeError(f'DEIMv2 helper exited {proc.returncode}: {proc.stderr[-600:]}')

        # Parse result
        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError(f'DEIMv2 helper produced no output. stderr: {proc.stderr[-300:]}')

        data = json.loads(stdout)
        if data.get('error'):
            raise RuntimeError(f'DEIMv2 error: {data["error"][:500]}')

        predictions = data.get('predictions', '')

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return predictions, [[ox, oy] for ox, oy in origins], (img_w, img_h)
