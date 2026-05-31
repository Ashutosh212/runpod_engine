# Mandara — Satellite Vehicle Detection Server

## What This Is

A self-contained FastAPI inference server that runs **two models side-by-side** on satellite imagery and shows results in a browser GUI. Upload an image, click Run — both models infer in parallel and results appear as overlaid bounding boxes.

- **Mandara (left panel)**: OrientedFormer oriented bounding box detector, trained 9 epochs
- **DEIMv2 (right panel)**: DINO-based fine-tuned detector, trained 72 epochs
- Both detect: `vehicle`, `smallvehicle`

**The folder is fully self-contained.** No external paths, no symlinks. Move it to any server and `bash setup.sh` + `bash start.sh` just works.

---

## Directory Layout

```
runpod_engine/
├── app.py                        # FastAPI server — all HTTP endpoints
├── inference.py                  # Mandara model load + single-image predict
├── patcher.py                    # Tile-and-stitch pipeline (1280×720 / 200px overlap)
├── deim_inference.py             # DEIMv2 orchestrator — tiles image, calls helper subprocess
├── deim_infer_helper.py          # Runs inside deimv2 env; loads model once, infers all tiles
├── gui.html                      # Browser GUI (served at GET /)
├── setup.sh                      # One-shot install — run once per new environment
├── start.sh                      # Start the server
├── environment.yml               # Conda env spec for ai4rs_infer (Mandara)
├── progress.md                   # Debugging log — read this before changing anything
│
├── checkpoints/
│   ├── epoch_9.pth               # Mandara checkpoint (540 MB, .gitignored)
│   └── deim/
│       └── best_stg1.pth         # DEIMv2 checkpoint (150 MB, .gitignored)
│
├── configs/
│   ├── _base_/                   # mmengine base configs for OrientedFormer
│   ├── deim/
│   │   └── deimv2_dinov3_s_vehicle.yml   # DEIMv2 vehicle config
│   ├── dataset/                  # DEIMv2 dataset config (included by deim config)
│   ├── base/                     # DEIMv2 base optimizer/dataloader configs
│   └── runtime.yml               # DEIMv2 runtime config
│
├── deim_src/
│   └── engine/                   # DEIMv2 Python source (1.3 MB, imported via sys.path)
│
├── projects/
│   └── OrientedFormer/
│       ├── __init__.py           # Required — mmengine needs this to traverse packages
│       ├── configs/              # OrientedFormer model config (.py)
│       └── orientedformer/       # Custom mmrotate modules (11 .py files)
│
├── pred/                         # Saved prediction .txt files (output)
└── sample_data/                  # Sample images served by the GUI
```

---

## Two Conda Environments

### `ai4rs_infer` — Mandara / OrientedFormer
- **Python**: 3.10
- **Torch**: 2.4.0 + CUDA 12.1
- **Key packages**: mmengine 0.10.7, mmcv 2.2.0, mmdet 3.3.0, mmrotate 1.0.0rc1 (1.x branch), mmsegmentation ≥1.2.2, FastAPI 0.111, uvicorn 0.30
- **Used for**: running the FastAPI server and Mandara inference

### `deimv2` — DEIMv2 DINO
- **Python**: 3.11
- **Torch**: latest stable + CUDA 12.4
- **Key packages**: pyyaml, pillow, numpy, scipy
- **Used for**: subprocess-isolated DEIMv2 inference (incompatible torch/Python versions with ai4rs_infer)
- **Does NOT need pip install**: the `engine/` module is loaded via `sys.path.insert(0, deim_src/)`

### Backup environments (if recreation fails)
```
/sfs/env_backups/ai4rs_infer/     # Full clone, directly activatable
/sfs/env_backups/deimv2/          # Full clone, directly activatable
/sfs/env_backups/ai4rs_infer_spec.yml   # Exact package list for diffing
/sfs/env_backups/deimv2_spec.yml        # Exact package list for diffing
```
To diff a freshly built env: `conda env export -n ai4rs_infer | diff - /sfs/env_backups/ai4rs_infer_spec.yml`

---

## Setup and Start

```bash
# First time on a new server (takes ~15-20 min):
bash setup.sh

# Every time to start the server:
bash start.sh
# → Server at http://localhost:8000
# → GUI at http://localhost:8000
```

`setup.sh` is **idempotent** — safe to re-run. It skips steps already done (existing env, existing mmrotate clone).

### What `setup.sh` does (in order):
1. Auto-detects conda (checks PATH, common install locations); installs Miniconda if missing
2. Creates `ai4rs_infer` env from `environment.yml`
3. Detects CUDA version → picks correct torch/mmcv index URL
4. Installs torch 2.4.0 + torchvision
5. Installs mmengine via mim; installs mmcv **via pip directly** (not mim — mim generates wrong URL)
6. Installs mmdet + mmsegmentation via mim
7. Patches mmdet, mmseg, mmrotate version caps (they reject mmcv 2.2.0 without patches)
8. Clones mmrotate 1.x to `$(dirname $PROJ)/mmrotate_runpod`; installs with `--no-deps`
9. Creates `projects/__init__.py` and `projects/OrientedFormer/__init__.py`
10. Installs FastAPI, uvicorn, pydantic, pillow, numpy, python-multipart
11. Creates `deimv2` env (Python 3.11) + installs torch + scipy/pyyaml/pillow/numpy

### What `start.sh` does:
- Sources conda, activates `ai4rs_infer`
- Sets `PYTHONPATH` to include the project root
- Starts uvicorn on port 8000
- Mandara model loads on startup (`inference.py:load_model()`)

---

## Inference Architecture

### Mandara (OrientedFormer)
```
upload → patcher.py → tiles (1280×720, 200px overlap)
       → inference.py:predict_image() per tile (in ai4rs_infer env)
       → patcher.py:_stitch() → shift coords by tile origin
       → DOTA format output: "x1 y1 x2 y2 x3 y3 x4 y4 class score" per line
       → saved as Mandara_model_NNNN_stem.txt
```

### DEIMv2
```
upload → deim_inference.py → same tile grid as patcher.py (identical params)
       → writes manifest JSON (tile paths + origins + config + checkpoint)
       → subprocess: deimv2/bin/python3 deim_infer_helper.py --manifest ...
           (inside helper: stdout redirected to stderr before model import
            to prevent debug prints corrupting JSON output)
       → helper outputs single JSON line to stdout
       → deim_inference.py parses JSON → stitches AABB coords → converts to 4-point quad
       → saved as dino_trained_model_NNNN_stem.txt
```

### Why subprocess isolation?
DEIMv2 needs Python 3.11 + torch 2.5; Mandara needs Python 3.10 + torch 2.4. They cannot coexist in one process. DEIMv2 runs as a subprocess using its own conda env's interpreter.

### Tiling (identical for both models)
- Tile size: 1280 × 720 px
- Overlap: 200 px (step: 1080 horizontal, 520 vertical)
- Edge handling: last tile snapped to `size - crop` (no padding)
- Both models see exactly the same tiles at exactly the same pixel coordinates

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Browser GUI (gui.html) |
| GET | `/health` | `{"status":"ok","model_ready":true}` |
| GET | `/samples` | List sample images |
| GET | `/sample_data/{filename}` | Serve a sample image |
| POST | `/predict` | Single image → DOTA text (no save) |
| POST | `/predict/save` | Upload → infer → save pred_NNNN.txt |
| POST | `/predict/patched` | Tile → infer → stitch → save |
| POST | `/predict/patched/stream` | Same with SSE progress events |
| POST | `/predict/compare/stream` | **Main endpoint** — runs both models, SSE stream |
| POST | `/predict/batch` | Server-side folder batch inference |

### `/predict/compare/stream` SSE events (in order):
```
mandara_start       → n_patches, patch_w, patch_h
mandara_tile_done   → tile, total  (one per tile)
mandara_stitching
mandara_done        → predictions (DOTA str), filename, detections, n_patches, img_w, img_h
deim_start          → n_patches
deim_done           → predictions (DOTA str), filename, detections
compare_done        → seq
error               → message  (on any failure)
```

---

## GUI Features

- **Side-by-side canvases**: Mandara left, DEIMv2 right
- **Overlay / Original toggle** per panel
- **Score threshold slider**: redraws both canvases live
- **Shared zoom**: 0.5× to 4× applies to both canvases
- **Tooltip**: hover over a box to see class + confidence
- **Full Image / Patch View · Top 10** toggle:
  - Ranks all tiles by sum of Mandara calibrated scores
  - Shows the 10 highest-density tiles
  - Navigate with `[←] [→]` buttons or keyboard arrow keys
  - Both canvases show the same tile simultaneously
- **Download .txt**: downloads the prediction file client-side
- **Show raw**: shows DOTA-format prediction text
- **Sample images**: loads from `sample_data/`
- **History**: lists all runs this session

### Score calibration
Raw scores are gamma-calibrated before display: `displayed_score = raw_score ^ 0.4`
Minimum raw score for display: 0.1 (filtered in `parseCalibrated()` in gui.html).

### Output file naming
```
pred/Mandara_model_NNNN_stem.txt      — Mandara predictions
pred/dino_trained_model_NNNN_stem.txt — DEIMv2 predictions
```
`NNNN` is a sequential counter shared across all output files.

---

## DOTA Format

Each line:
```
x1 y1 x2 y2 x3 y3 x4 y4 class score
```
- For Mandara: true oriented bounding boxes (4-corner polygon, may be rotated)
- For DEIMv2: axis-aligned boxes expressed as 4-corner quads (`x0 y0 x1 y0 x1 y1 x0 y1`)
- Coordinates are in full-image pixel space (tile origin already added during stitching)

---

## Non-Obvious Things (Read Before Changing)

1. **`projects/__init__.py` must exist** — mmengine's custom module loader requires it. `setup.sh` creates it with `touch`. If it disappears, the model fails to load with a cryptic import error.

2. **mmcv must be installed via pip, not mim** — mim generates URL `torch2.4.0` but CDN uses `torch2.4`. Always use: `pip install mmcv==2.2.0 -f <cdnurl> --no-build-isolation`

3. **mmrotate must be 1.x branch** — default clone is 0.3.x (old API). Always `git clone -b 1.x`.

4. **Three version cap patches are needed** — mmdet, mmseg, and mmrotate all ship with upper bounds that reject mmcv 2.2.0. `setup.sh` patches all three after install.

5. **DEIMv2 helper stdout must be silenced** — model prints debug lines to stdout before JSON. `deim_infer_helper.py` does `sys.stdout = sys.stderr` before any model import, restores only for the final `print(json.dumps(...))`.

6. **DEIMv2 Python interpreter detection** — `deim_inference.py:_find_deimv2_python()` first checks a static list of common conda env paths, then falls back to `conda run -n deimv2 which python3` so it works on any server regardless of conda install location.

7. **Checkpoints are .gitignored** — `checkpoints/epoch_9.pth` (540 MB) and `checkpoints/deim/best_stg1.pth` (150 MB) must be copied manually to a new server. Code and configs travel via git; models do not.

8. **All DEIMv2 assets are internal** — `deim_src/engine/`, `configs/deim/`, and `configs/dataset|base|runtime.yml` are all inside this repo. No external DEIMv2 project directory is needed.
