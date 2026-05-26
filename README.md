# Mandara — Satellite Vehicle Detection

Mandara is a FastAPI inference server for detecting vehicles in satellite imagery using an OrientedFormer 8-class oriented bounding box (OBB) detector.

---

## Features

- **Patch & stitch pipeline** — large images are automatically tiled into 1280×720 patches (200 px overlap), inferred tile-by-tile, and stitched back to full-image coordinates
- **Real-time progress** — live step-by-step status streamed to the browser via Server-Sent Events
- **Browser GUI** — drag-and-drop upload, interactive canvas overlay, Overlay / Original toggle, zoom controls, hover tooltips with confidence scores
- **Vehicle-focused display** — overlay shows `vehicle` and `smallvehicle` classes only
- **Sequential prediction saving** — results saved as `pred/pred_NNNN_<stem>.txt` in DOTA format
- **REST API** — programmatic access via `/predict`, `/predict/save`, `/predict/patched`, and `/predict/batch`

---

## Quickstart

### 1. Environment

```bash
conda env create -f environment.yml
conda activate ai4rs
```

### 2. Checkpoint

Place (or symlink) your trained checkpoint at:

```
checkpoints/epoch_9.pth
```

### 3. Start the server

```bash
bash setup.sh
# or directly:
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Browser GUI |
| `GET`  | `/health` | Model status |
| `POST` | `/predict` | Single image → DOTA text |
| `POST` | `/predict/save` | Single image → save to `pred/` |
| `POST` | `/predict/patched` | Patch & stitch → save |
| `POST` | `/predict/patched/stream` | Patch & stitch with live SSE progress |
| `POST` | `/predict/batch` | Folder of images → DOTA `.txt` files |

---

## Output Format

Predictions are saved in DOTA format:

```
x1 y1 x2 y2 x3 y3 x4 y4 class score
```

Each row is one oriented bounding box defined by its four corner points.

---

## Detected Classes

`arty` · `camo` · `logistic` · `missile` · `radar` · `smallvehicle` · `tank` · `vehicle`
