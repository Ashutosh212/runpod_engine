"""FastAPI inference server for OrientedFormer 8-class detector.

Endpoints:
  GET  /                      — browser GUI
  GET  /health                — liveness check + model status
  GET  /samples               — list sample images available for demo
  GET  /sample_data/{filename}— serve a sample image file
  POST /predict               — single image upload → DOTA-format text
  POST /predict/save          — upload + save to pred/ with sequential naming
  POST /predict/patched       — patch → infer each tile → stitch → save
  POST /predict/batch         — process a server-side folder → save DOTA .txt files
"""
import glob
import json
import os
import re
import tempfile

from fastapi import FastAPI, File, Form, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="OrientedFormer Inference API", version="1.0")

_model_ready = False

PRED_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

# ── Upload size limit ─────────────────────────────────────────────────────────
# Derived from GSD ≈ 0.30 m/px, 3-band uint16 (worst case), 30 km² ceiling:
#   30 km² × 11.2M px/km² × 3 bands × 2 bytes = ~2.0 GB
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_UPLOAD_LABEL = "2 GB (≈ 30 km²)"

_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _validate_upload(image: UploadFile, content: bytes) -> None:
    """Raise HTTPException if the upload is too large or wrong format."""
    ext = os.path.splitext(image.filename or "")[-1].lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. Accepted: PNG, JPG, TIFF (3-band RGB)."
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(content) / 1e9:.2f} GB). "
            f"Maximum allowed: {MAX_UPLOAD_LABEL}."
        )


def _next_seq(out_dir: str) -> int:
    """Return next sequential index based on existing pred_NNNN_*.txt files."""
    existing = glob.glob(os.path.join(out_dir, "pred_*.txt"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.match(r"pred_(\d+)_", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


@app.on_event("startup")
async def startup():
    global _model_ready
    try:
        from inference import load_model
        load_model()
        _model_ready = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")


@app.get("/", response_class=HTMLResponse)
def gui():
    return HTMLResponse(content=open(
        os.path.join(os.path.dirname(__file__), "gui.html")
    ).read())


@app.get("/health")
def health():
    return {"status": "ok" if _model_ready else "model_not_loaded",
            "model_ready": _model_ready}


@app.get("/samples")
def list_samples():
    """Return metadata for all sample images available for demo."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    files = sorted(
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return [{"filename": f, "url": f"/sample_data/{f}"} for f in files]


@app.get("/sample_data/{filename}")
def serve_sample(filename: str):
    """Serve a sample image file by name."""
    safe = os.path.basename(filename)
    path = os.path.join(SAMPLE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Sample not found: {safe}")
    return FileResponse(path, media_type="image/png")


@app.post("/predict", response_class=PlainTextResponse)
async def predict(
    image: UploadFile = File(...),
    score_thr: float = Query(0.05, ge=0.0, le=1.0),
):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")

    content = await image.read()
    _validate_upload(image, content)

    suffix = os.path.splitext(image.filename)[-1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from inference import predict_image
        result = predict_image(tmp_path, score_thr=score_thr)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/predict/save")
async def predict_save(
    image: UploadFile = File(...),
    score_thr: float = Form(0.05),
    out_dir: str = Form(PRED_DIR),
):
    """Upload image → run inference → save DOTA .txt with sequential name.

    Saved as: {out_dir}/pred_{NNNN}_{original_stem}.txt
    Returns JSON with predictions text, saved filename, and detection count.
    """
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")

    os.makedirs(out_dir, exist_ok=True)

    content = await image.read()
    _validate_upload(image, content)

    suffix = os.path.splitext(image.filename)[-1] or ".png"
    stem = os.path.splitext(image.filename)[0] if image.filename else "image"
    # strip unsafe path chars from the original stem
    stem = re.sub(r"[^\w\-]", "_", stem)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from inference import predict_image
        result = predict_image(tmp_path, score_thr=score_thr)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    seq = _next_seq(out_dir)
    filename = f"pred_{seq:04d}_{stem}.txt"
    save_path = os.path.join(out_dir, filename)
    with open(save_path, "w") as f:
        f.write(result)

    det_count = len([l for l in result.splitlines() if l.strip()])
    return JSONResponse({
        "filename": filename,
        "save_path": save_path,
        "detections": det_count,
        "predictions": result,
    })


@app.post("/predict/patched")
async def predict_patched(
    image: UploadFile = File(...),
    score_thr: float = Form(0.05),
    out_dir: str = Form(PRED_DIR),
):
    """Tile the image → infer each tile → stitch → save with sequential name.

    Returns JSON with stitched predictions, saved filename, patch metadata.
    """
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")

    os.makedirs(out_dir, exist_ok=True)

    content = await image.read()
    _validate_upload(image, content)

    suffix = os.path.splitext(image.filename)[-1] or ".png"
    stem   = re.sub(r"[^\w\-]", "_", os.path.splitext(image.filename)[0] if image.filename else "image")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from patcher import patch_and_predict, PATCH_W, PATCH_H
        stitched, origins, (img_w, img_h) = patch_and_predict(tmp_path, score_thr=score_thr)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    seq      = _next_seq(out_dir)
    filename = f"pred_{seq:04d}_{stem}.txt"
    with open(os.path.join(out_dir, filename), "w") as f:
        f.write(stitched)

    det_count = len([l for l in stitched.splitlines() if l.strip()])
    return JSONResponse({
        "filename":    filename,
        "save_path":   os.path.join(out_dir, filename),
        "detections":  det_count,
        "predictions": stitched,
        "n_patches":   len(origins),
        "patch_origins": origins,
        "patch_w":     PATCH_W,
        "patch_h":     PATCH_H,
        "img_w":       img_w,
        "img_h":       img_h,
    })


@app.post("/predict/patched/stream")
async def predict_patched_stream(
    image: UploadFile = File(...),
    score_thr: float = Form(0.05),
    out_dir: str = Form(PRED_DIR),
):
    """SSE stream: patch → infer each tile (with live progress) → stitch → save.

    Client reads data lines; each is a JSON object with a 'type' field.
    Final event type=='done' carries predictions, filename, counts.
    """
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")

    content = await image.read()
    _validate_upload(image, content)

    suffix  = os.path.splitext(image.filename)[-1] or ".png"
    stem    = re.sub(r"[^\w\-]", "_",
                     os.path.splitext(image.filename)[0] if image.filename else "image")

    async def generate():
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with open(tmp_path, 'wb') as f:
                f.write(content)

            from patcher import patch_and_predict_stream
            async for event in patch_and_predict_stream(tmp_path, score_thr):
                if event['type'] == 'done':
                    stitched = event.pop('stitched')
                    os.makedirs(out_dir, exist_ok=True)
                    seq      = _next_seq(out_dir)
                    filename = f"pred_{seq:04d}_{stem}.txt"
                    with open(os.path.join(out_dir, filename), 'w') as f:
                        f.write(stitched)
                    det_count = len([l for l in stitched.splitlines() if l.strip()])
                    event.update({
                        'filename':    filename,
                        'save_path':   os.path.join(out_dir, filename),
                        'detections':  det_count,
                        'predictions': stitched,
                    })
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class BatchRequest(BaseModel):
    img_dir: str
    out_dir: str
    score_thr: float = 0.05


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not _model_ready:
        raise HTTPException(503, "Model not loaded yet")
    if not os.path.isdir(req.img_dir):
        raise HTTPException(400, f"img_dir not found: {req.img_dir}")

    from inference import predict_batch as _batch
    _batch(req.img_dir, req.out_dir, score_thr=req.score_thr)

    saved = len([f for f in os.listdir(req.out_dir) if f.endswith(".txt")])
    return {"status": "done", "out_dir": req.out_dir, "files_saved": saved}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
