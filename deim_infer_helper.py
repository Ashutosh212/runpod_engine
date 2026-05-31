"""
DEIMv2 batch inference helper.

Run by the deimv2 conda env via subprocess.
Receives a manifest JSON file, loads the model ONCE, processes all tiles,
and writes JSON result to stdout.

Manifest format (JSON file):
{
  "config":     "/path/to/config.yml",
  "checkpoint": "/path/to/best_stg1.pth",
  "proj_root":  "/path/to/DEIMv2",
  "threshold":  0.05,
  "tiles": [
    {"path": "/tmp/tile_0000.png", "ox": 0, "oy": 0},
    ...
  ]
}

Output (stdout):
{
  "predictions": "DOTA-format string (full-image coords, stitched)",
  "n_detections": 42,
  "error": null
}
"""
import argparse
import json
import os
import sys


CLASS_NAMES = {0: 'vehicle', 1: 'smallvehicle'}


def run(manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)

    sys.path.insert(0, manifest['proj_root'])

    # Redirect stdout → stderr so model debug prints don't corrupt our JSON output
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from engine.core import YAMLConfig

    cfg = YAMLConfig(manifest['config'], resume=manifest['checkpoint'])
    ckpt = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False)
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    cfg.model.load_state_dict(state)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy().cuda()
            self.postprocessor = cfg.postprocessor.deploy().cuda()
        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = _Model().eval()
    img_size     = cfg.yaml_cfg['eval_spatial_size']
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)
    threshold    = float(manifest.get('threshold', 0.05))

    transforms = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if vit_backbone else T.Lambda(lambda x: x),
    ])

    all_lines = []
    for tile in manifest['tiles']:
        ox = tile['ox']
        oy = tile['oy']
        im = Image.open(tile['path']).convert('RGB')
        w, h = im.size
        orig_size = torch.tensor([[w, h]]).cuda()
        im_data   = transforms(im).unsqueeze(0).cuda()

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        l, b, s = labels[0], boxes[0], scores[0]
        mask = s > threshold
        l, b, s = l[mask], b[mask], s[mask]

        for label, box, score in zip(l, b, s):
            cls = CLASS_NAMES.get(label.item(), str(label.item()))
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            # Shift to full-image coords and convert AABB → 4-point DOTA quad
            ax0, ay0 = x0 + ox, y0 + oy
            ax1, ay1 = x1 + ox, y1 + oy
            coords = (f"{ax0:.2f} {ay0:.2f} {ax1:.2f} {ay0:.2f} "
                      f"{ax1:.2f} {ay1:.2f} {ax0:.2f} {ay1:.2f}")
            all_lines.append(f"{coords} {cls} {score.item():.6f}")

    result = '\n'.join(all_lines)

    # Restore stdout and write clean JSON
    sys.stdout = _real_stdout
    print(json.dumps({
        'predictions': result,
        'n_detections': len(all_lines),
        'error': None,
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to manifest JSON file')
    args = parser.parse_args()
    try:
        run(args.manifest)
    except Exception as e:
        import traceback
        sys.stdout = sys.__stdout__
        print(json.dumps({'predictions': '', 'n_detections': 0,
                          'error': str(e) + '\n' + traceback.format_exc()}))
