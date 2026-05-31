# Mandara — Satellite Vehicle Detection Server

A self-contained FastAPI inference server that runs two detection models side-by-side on satellite imagery and displays results in a browser GUI. Upload an image, click **Run Inference** — both models infer and results appear as overlaid bounding boxes on dual canvases.

| Model | Panel | Training | Classes |
|-------|-------|----------|---------|
| **Mandara** (OrientedFormer) | Left | Epoch 9 | vehicle · smallvehicle |
| **DEIMv2** (DINO fine-tuned) | Right | Epoch 72 | vehicle · smallvehicle |

---

## Quickstart

### Step 1 — Clone the repo

```bash
git clone <your-repo-url>
cd runpod_engine
```

### Step 2 — Add the model checkpoints (not in git — too large)

```bash
# Create directories
mkdir -p checkpoints/deim

# Copy checkpoints from wherever you have them:
cp /path/to/epoch_9.pth          checkpoints/epoch_9.pth        # 540 MB — Mandara
cp /path/to/best_stg1.pth        checkpoints/deim/best_stg1.pth # 150 MB — DEIMv2
```

Both files are `.gitignored`. They must be copied manually to every new server.

### Step 3 — Run setup (once per new environment)

```bash
bash setup.sh
```

This takes **15–20 minutes** on first run. It is **idempotent** — safe to re-run after partial failures. When finished you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║  Setup complete!                                             ║
║  Start server:  bash /path/to/runpod_engine/start.sh        ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 4 — Start the server

```bash
bash start.sh
```

### Step 5 — Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","model_ready":true}
```

Open `http://localhost:8000` in your browser.

---

## What `setup.sh` Does

`setup.sh` is fully automatic. It detects your environment and installs everything:

1. **Finds or installs conda** — checks PATH and common locations (`/opt/miniconda`, `/sfs/miniconda3`, `~/miniconda3`); installs Miniconda if none found
2. **Creates `ai4rs_infer` conda env** (Python 3.10) from `environment.yml`
3. **Detects CUDA version** → selects correct torch/mmcv index URL (CUDA 11 → cu118, CUDA 12 → cu121, no GPU → cpu)
4. **Installs PyTorch 2.4.0** + torchvision
5. **Installs mmengine** via mim; installs **mmcv 2.2.0 via pip directly** (not mim — mim generates a wrong CDN URL)
6. **Installs mmdet + mmsegmentation** via mim
7. **Patches version caps** — mmdet, mmseg, and mmrotate all ship with upper bounds that reject mmcv 2.2.0; setup patches all three `__init__.py` files automatically
8. **Clones mmrotate 1.x** to `../mmrotate_runpod` (alongside this repo) and installs with `--no-deps`
9. **Creates `projects/__init__.py`** files — required for mmengine's custom module loader
10. **Installs FastAPI, uvicorn, pydantic, pillow, numpy**
11. **Creates `deimv2` conda env** (Python 3.11) + installs torch + scipy/pyyaml/pillow/numpy

---

## Two Conda Environments

The server requires **two separate environments** because Mandara (Python 3.10 + torch 2.4) and DEIMv2 (Python 3.11 + torch 2.5) cannot coexist in one process.

| Env | Python | Purpose | Key packages |
|-----|--------|---------|--------------|
| `ai4rs_infer` | 3.10 | Mandara inference + FastAPI server | torch 2.4, mmcv 2.2.0, mmdet 3.3.0, mmrotate 1.0.0rc1 |
| `deimv2` | 3.11 | DEIMv2 inference (subprocess) | torch latest, pyyaml, pillow, scipy |

DEIMv2 runs as a **subprocess** — each inference request spawns `deimv2/bin/python3 deim_infer_helper.py`. This is intentional to isolate the incompatible environments.

---

## Self-Contained Folder

Everything needed to run is inside `runpod_engine/`. No external paths required.

```
runpod_engine/
├── checkpoints/
│   ├── epoch_9.pth                 ← Mandara checkpoint (540 MB, .gitignored)
│   └── deim/
│       └── best_stg1.pth           ← DEIMv2 checkpoint (150 MB, .gitignored)
│
├── configs/
│   ├── deim/
│   │   └── deimv2_dinov3_s_vehicle.yml   ← DEIMv2 inference config
│   ├── dataset/                    ← DEIMv2 config includes
│   ├── base/                       ← DEIMv2 config includes
│   └── runtime.yml                 ← DEIMv2 config includes
│
├── deim_src/
│   └── engine/                     ← DEIMv2 Python source (1.3 MB, in git)
│
├── projects/
│   └── OrientedFormer/
│       ├── __init__.py             ← Required (empty, created by setup.sh)
│       ├── configs/                ← OrientedFormer model config
│       └── orientedformer/         ← Custom mmrotate modules
│
├── app.py                          ← FastAPI server
├── inference.py                    ← Mandara model loading + prediction
├── patcher.py                      ← Tile-and-stitch pipeline
├── deim_inference.py               ← DEIMv2 orchestrator
├── deim_infer_helper.py            ← DEIMv2 subprocess worker
├── gui.html                        ← Browser GUI
├── setup.sh                        ← One-shot install script
├── start.sh                        ← Server start script
└── environment.yml                 ← Conda env spec for ai4rs_infer
```

---

## GUI Features

- **Side-by-side comparison** — Mandara (left) and DEIMv2 (right) on the same image
- **Overlay / Original toggle** — per panel
- **Score threshold slider** — live redraw on both canvases
- **Shared zoom** — 0.5× to 4×, applies to both canvases simultaneously
- **Hover tooltips** — class name + confidence on any bounding box
- **Patch View · Top 10** — shows the 10 tiles with the highest detection density; navigate with `[←] [→]` buttons or keyboard arrow keys; both canvases always show the same tile
- **Download .txt** — saves prediction file client-side
- **Sample images** — loaded from `sample_data/` (not in git)
- **Run history** — lists all inference runs this session

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser GUI |
| `GET` | `/health` | `{"status":"ok","model_ready":true/false}` |
| `GET` | `/samples` | List sample images |
| `GET` | `/sample_data/{filename}` | Serve a sample image |
| `POST` | `/predict` | Upload → DOTA text (no save) |
| `POST` | `/predict/save` | Upload → infer → save `pred_NNNN.txt` |
| `POST` | `/predict/patched` | Tile → stitch → save |
| `POST` | `/predict/patched/stream` | Same with live SSE progress |
| `POST` | `/predict/compare/stream` | **Main endpoint** — both models, SSE stream |
| `POST` | `/predict/batch` | Server-side folder batch inference |

---

## Output Format (DOTA)

Predictions are saved as:
```
pred/Mandara_model_NNNN_stem.txt
pred/dino_trained_model_NNNN_stem.txt
```

Each line is one detection:
```
x1 y1 x2 y2 x3 y3 x4 y4 class score
```

- Mandara: true oriented bounding boxes (rotated quads)
- DEIMv2: axis-aligned boxes expressed as 4-corner quads
- Coordinates are in full-image pixel space

---

## Tiling

Both models use **identical tiling parameters**:

| Parameter | Value |
|-----------|-------|
| Tile size | 1280 × 720 px |
| Overlap | 200 px |
| Horizontal step | 1080 px |
| Vertical step | 520 px |

---

## What Must Be Copied Manually (Not in Git)

| File | Size | Why not in git |
|------|------|----------------|
| `checkpoints/epoch_9.pth` | 540 MB | Too large for GitHub |
| `checkpoints/deim/best_stg1.pth` | 150 MB | Too large for GitHub |
| `sample_data/` | ~43 MB | Optional demo images |

Transfer with rsync:
```bash
rsync -avP checkpoints/ user@newserver:/path/to/runpod_engine/checkpoints/
rsync -avP sample_data/ user@newserver:/path/to/runpod_engine/sample_data/
```

---

## Troubleshooting

See [`progress.md`](progress.md) for a full log of every non-obvious fix — 11 documented issues covering mmcv/mmrotate/mmdet version conflicts, missing `__init__.py` files, DEIMv2 subprocess stdout pollution, and more.

See [`CLAUDE.md`](CLAUDE.md) for architecture details, inference flow, and design decisions.

### Quick checks if something breaks

```bash
# Is the server up?
curl http://localhost:8000/health

# Are both envs installed?
conda env list | grep -E "ai4rs_infer|deimv2"

# Are checkpoints present?
ls -lh checkpoints/epoch_9.pth checkpoints/deim/best_stg1.pth

# Test DEIMv2 standalone
/path/to/envs/deimv2/bin/python3 deim_infer_helper.py --manifest /tmp/test.json
```

### Compare a freshly built env against the known-good backup

```bash
conda env export -n ai4rs_infer | diff - /sfs/env_backups/ai4rs_infer_spec.yml
conda env export -n deimv2      | diff - /sfs/env_backups/deimv2_spec.yml
```
