"""Patch-and-stitch pipeline for full-image inference.

Slides a 1280×720 window (200px overlap) over the uploaded image,
runs inference on each tile, then shifts predicted coordinates back
to full-image space by adding the tile's origin offset.
"""
import os
import tempfile
from PIL import Image

PATCH_W = 1280
PATCH_H = 720
OVERLAP = 200
STEP_W  = PATCH_W - OVERLAP   # 1080
STEP_H  = PATCH_H - OVERLAP   # 520


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


async def patch_and_predict_stream(img_path: str, score_thr: float = 0.05):
    """Async generator — yields progress dicts as tiles are inferred.

    Event types:
      patch_start  — image split, includes n_patches / img_w / img_h
      tile_done    — one tile finished, includes tile / total
      stitching    — all tiles done, stitching now
      done         — finished, includes stitched DOTA string + counts
      error        — something went wrong, includes message
    """
    import asyncio
    from inference import predict_image as _infer

    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    origins = _tile_origins(img_w, img_h)
    n = len(origins)

    yield {'type': 'patch_start', 'n_patches': n,
           'img_w': img_w, 'img_h': img_h,
           'patch_w': PATCH_W, 'patch_h': PATCH_H, 'overlap': OVERLAP}

    loop = asyncio.get_running_loop()
    patch_dota = []

    for i, (x0, y0) in enumerate(origins):
        x1 = min(x0 + PATCH_W, img_w)
        y1 = min(y0 + PATCH_H, img_h)
        crop = img.crop((x0, y0, x1, y1))

        fd, tmp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            crop.save(tmp_path)
            dota_str = await loop.run_in_executor(None, _infer, tmp_path, score_thr)
        except Exception as exc:
            yield {'type': 'error', 'message': str(exc)}
            return
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        patch_dota.append(dota_str)
        yield {'type': 'tile_done', 'tile': i + 1, 'total': n}

    yield {'type': 'stitching'}
    stitched = _stitch(patch_dota, origins)
    yield {'type': 'done', 'stitched': stitched,
           'n_patches': n, 'img_w': img_w, 'img_h': img_h}


def patch_and_predict(img_path: str, score_thr: float = 0.05):
    """Tile the image, infer each tile, stitch coordinates back.

    Returns:
        stitched_dota : str            — full-image DOTA predictions
        origins       : list[[x0, y0]] — top-left of every tile
        img_size      : (W, H)
    """
    from inference import predict_image

    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    origins = _tile_origins(img_w, img_h)

    patch_dota = []
    for x0, y0 in origins:
        x1 = min(x0 + PATCH_W, img_w)
        y1 = min(y0 + PATCH_H, img_h)
        crop = img.crop((x0, y0, x1, y1))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            crop.save(tmp.name)
            tmp_path = tmp.name
        try:
            patch_dota.append(predict_image(tmp_path, score_thr=score_thr))
        finally:
            os.unlink(tmp_path)

    stitched = _stitch(patch_dota, origins)
    return stitched, [[x, y] for x, y in origins], (img_w, img_h)


def _stitch(patch_results: list[str], origins: list[tuple[int, int]]) -> str:
    lines = []
    for dota_str, (x0, y0) in zip(patch_results, origins):
        for line in dota_str.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            coords = list(map(float, parts[:8]))
            for i in range(4):
                coords[i * 2]     += x0
                coords[i * 2 + 1] += y0
            cls, score = parts[8], parts[9]
            lines.append(' '.join(f'{v:.2f}' for v in coords) + f' {cls} {score}')
    return '\n'.join(lines)
