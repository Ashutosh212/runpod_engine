# OrientedFormer Inference Endpoint

## Goal
This directory is a self-contained inference service for an 8-class oriented object detector
(OrientedFormer) trained on satellite imagery. The service exposes a REST API on port 8000
that accepts image uploads and returns rotated bounding-box predictions in DOTA format.

## Model
- **Architecture**: OrientedFormer (DDQ-RCNN variant with OrientedAttention)
- **Classes (index 0–7)**: arty, camo, logistic, missile, radar, smallvehicle, tank, vehicle
- **Angle convention**: le90 (radians, range [-π/2, π/2])
- **Input resolution**: trained at 1280×720, accepts any size
- **Checkpoint**: `checkpoints/epoch_9.pth` (symlink; update to final epoch when training finishes)

## Directory Layout
```
inference_endpoint/
├── app.py                  # FastAPI server (port 8000)
├── inference.py            # Model loading + prediction logic
├── environment.yml         # Conda env (Python 3.10, torch 2.4, CUDA 12.4)
├── setup.sh                # One-shot pod setup script
├── CLAUDE.md               # This file
├── checkpoints/
│   └── epoch_9.pth         # Symlink to /sfs/work_dirs/.../epoch_9.pth
├── configs/
│   └── _base_/             # mmengine base configs (copied from ai4rs)
└── projects/
    └── OrientedFormer/
        ├── configs/        # Main model config
        └── orientedformer/ # Custom module (11 .py files)
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model status |
| POST | `/predict` | Upload image → DOTA text response |
| POST | `/predict/batch` | Process server-side folder, save .txt files |

### POST /predict
- Body: `multipart/form-data` with `image` field (PNG/JPG)
- Query param: `score_thr` (float, default 0.05)
- Response: plain text, one detection per line:
  `x1 y1 x2 y2 x3 y3 x4 y4 class score`

### POST /predict/batch
```json
{ "img_dir": "/path/to/images", "out_dir": "/path/to/save", "score_thr": 0.05 }
```

## Starting the Server
```bash
conda activate ai4rs_infer
cd /path/to/inference_endpoint
python app.py
# → listening on 0.0.0.0:8000
```

## Key Files to Edit
- **`inference.py`**: Change `CHECKPOINT` constant to point to a newer epoch after training finishes.
- **`app.py`**: Add auth, rate limiting, or CORS headers if exposing publicly.
- **`configs/`**: If re-training with a different class set, update base dataset config and CLASSES tuple in `inference.py`.

## Deployment Pod (RunPod A5000)
- Expose **HTTP port 8000** in pod settings (Edit Pod → Expose HTTP Ports → 8000)
- Optionally expose **TCP port 22** for SSH
- After start, access via: `https://<pod-id>-8000.proxy.runpod.net`

## Dependencies
- Python 3.10, PyTorch 2.4, CUDA 12.4
- mmengine 0.10.7, mmcv 2.2.0, mmdet 3.3.0, mmrotate 1.0.0rc1
- FastAPI 0.111, uvicorn 0.30

## Training Context
Model was trained on a merged 3-source dataset (27,566 images total):
- `/sfs/data/custom/trainval` — 5,541 real images
- `/sfs/syn_all` — 13,189 synthetic images
- `/sfs/up42_train_2ndstage_may_1/patches` — 8,836 real images

This is stage-2 fine-tuning (36 epochs, LR=2e-5) from a stage-1 checkpoint at epoch 36.
Training work dir: `/sfs/work_dirs/orientedformer_custom_8cls_may_w3/`
