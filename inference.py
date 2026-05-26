"""Core inference logic for OrientedFormer 8-class detector."""
import math
import os
import sys
import numpy as np

# Ensure the orientedformer module is importable as projects.OrientedFormer.orientedformer
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CONFIG = os.path.join(
    _ROOT,
    "projects/OrientedFormer/configs/"
    "orientedformer_le90_r50_q300_layer2_head64_point32_1x_custom_8cls_stage2.py",
)
CHECKPOINT = os.path.join(_ROOT, "checkpoints/epoch_9.pth")

CLASSES = ("arty", "camo", "logistic", "missile", "radar",
           "smallvehicle", "tank", "vehicle")

_model = None


def load_model(config=CONFIG, checkpoint=CHECKPOINT, device="cuda:0"):
    global _model
    if _model is not None:
        return _model
    from mmdet.apis import init_detector
    _model = init_detector(config, checkpoint, device=device)
    return _model


def rbox_to_poly(cx, cy, w, h, angle):
    """(cx,cy,w,h,angle_rad) → [x1,y1,x2,y2,x3,y3,x4,y4]"""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    corners = [(hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)]
    pts = []
    for dx, dy in corners:
        pts.append(dx * cos_a - dy * sin_a + cx)
        pts.append(dx * sin_a + dy * cos_a + cy)
    return pts


def predict_image(image_path: str, score_thr: float = 0.05) -> str:
    """Run inference on one image. Returns DOTA-format string."""
    from mmdet.apis import inference_detector

    model = load_model()
    result = inference_detector(model, image_path)
    pred = result.pred_instances

    bboxes = pred.bboxes.cpu().numpy()  # (N, 5): cx cy w h angle
    scores = pred.scores.cpu().numpy()  # (N,)
    labels = pred.labels.cpu().numpy()  # (N,)

    lines = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < score_thr:
            continue
        cx, cy, w, h, angle = (float(bboxes[i][j]) for j in range(5))
        cls = CLASSES[int(labels[i])]
        pts = rbox_to_poly(cx, cy, w, h, angle)
        coords = " ".join(f"{v:.2f}" for v in pts)
        lines.append(f"{coords} {cls} {score:.6f}")

    return "\n".join(lines)


def predict_batch(img_dir: str, out_dir: str, score_thr: float = 0.05):
    """Run inference on all .png images in img_dir, save DOTA .txt to out_dir."""
    import glob
    os.makedirs(out_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not images:
        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        dota_str = predict_image(img_path, score_thr=score_thr)
        with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
            f.write(dota_str)
    print(f"Done: {len(images)} images → {out_dir}")
