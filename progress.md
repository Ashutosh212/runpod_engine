# Mandara — Setup Fixes & Changes Log

Every non-obvious fix, gotcha, and design decision is documented here.
Read this before changing `setup.sh`, `deim_inference.py`, or the model loading code.

---

## Fix 1 — `pkg_resources` missing when building mmcv

**Symptom:** `ModuleNotFoundError: No module named 'pkg_resources'` during `pip install mmcv`.

**Root cause:** `mim install mmcv` constructs the wheel index URL as `torch2.4.0` but the
OpenMMLab CDN uses `torch2.4` (no `.0`). pip finds no pre-built wheel, falls back to the
source tarball from PyPI, and tries to build it inside an isolated build environment that
has a broken setuptools (missing `pkg_resources`).

**Fix in `setup.sh`:** Bypass mim for mmcv entirely. Use pip directly with the correct URL
and `--no-build-isolation`:
```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html --no-build-isolation
```

---

## Fix 2 — mmdet rejects mmcv 2.2.0

**Symptom:** `AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0`

**Root cause:** mmdet ships with `mmcv_maximum_version = '2.2.0'` which is a strict less-than
check, so mmcv 2.2.0 itself is rejected.

**Fix in `setup.sh`:** Patch `mmdet/__init__.py` after install:
```
mmcv_maximum_version = '2.2.0'  →  mmcv_maximum_version = '2.3.0'
```

---

## Fix 3 — mmseg rejects mmcv 2.2.0

**Symptom:** `AssertionError: MMCV==2.2.0 is used but incompatible.`

**Root cause:** Same strict version cap in mmsegmentation.

**Fix in `setup.sh`:** Patch `mmseg/__init__.py` after install:
```
MMCV_MAX = '2.2.0'  →  MMCV_MAX = '2.3.0'
```

---

## Fix 4 — mmrotate 0.x installed instead of 1.x

**Symptom:** `ModuleNotFoundError: No module named 'mmrotate.registry'`
(also `mmrotate.structures.bbox.RotatedBoxes` not found).

**Root cause:** Default `git clone https://github.com/open-mmlab/mmrotate.git` gives the 0.3.x
branch which uses the old mmcv 1.x API. OrientedFormer imports from the new 1.x API.

**Fix in `setup.sh`:**
```bash
git clone -b 1.x https://github.com/open-mmlab/mmrotate.git
```

---

## Fix 5 — mmrotate install fails due to `mmcv-full` dependency

**Symptom:** `ERROR: Encountered error while generating package metadata. mmcv-full`

**Root cause:** mmrotate 1.x's `setup.py` lists `mmcv-full` as a dependency (the old package
name). We already have `mmcv` 2.x installed, so this conflicts.

**Fix in `setup.sh`:** Install with `--no-deps`:
```bash
pip install "$MMROTATE_DIR" --no-build-isolation --no-deps
```

---

## Fix 6 — mmrotate rejects mmcv 2.2.0 and mmdet 3.3.0

**Symptom:**
- `AssertionError: MMCV 2.2.0 is incompatible. Please use MMCV >= 2.0.0rc4, <= 2.1.0`
- `AssertionError: MMDetection 3.3.0 is incompatible. Please use MMDetection >= 3.0.0rc6, < 3.1.0`

**Root cause:** mmrotate 1.0.0rc1 ships with tight version caps that predate mmcv 2.2.0 and mmdet 3.3.0.

**Fix in `setup.sh`:** Patch `mmrotate/__init__.py` after install:
```
mmcv_maximum_version = '2.1.0'   →  mmcv_maximum_version = '2.3.0'
mmdet_maximum_version = '3.1.0'  →  mmdet_maximum_version = '3.4.0'
```

---

## Fix 7 — Missing `__init__.py` files in `projects/`

**Symptom:** `Model load failed: Failed to import custom modules from
{'imports': ['projects.OrientedFormer.orientedformer']}`

**Root cause:** `projects/` and `projects/OrientedFormer/` had no `__init__.py`.
mmengine's custom module loader cannot traverse a package hierarchy without these files.

**Fix in `setup.sh`:**
```bash
touch "$PROJ/projects/__init__.py"
touch "$PROJ/projects/OrientedFormer/__init__.py"
```
These files must exist even if empty.

---

## Fix 8 — mmrotate cloned to `/tmp` (lost on pod reset)

**Root cause:** `/tmp` is ephemeral — wiped on every pod restart. Cloning mmrotate there
meant re-cloning on every boot.

**Fix in `setup.sh`:** Clone alongside the project repo to a persistent path:
```bash
MMROTATE_DIR="$(dirname "$PROJ")/mmrotate_runpod"
```
On `/sfs`-based servers this resolves to `/sfs/mmrotate_runpod`.

---

## Fix 9 — DEIMv2 subprocess stdout pollution (JSON parse failure)

**Symptom:** GUI showed "Stream ended without complete results from both models."
Server logs: `JSONDecodeError` when parsing DEIMv2 subprocess output.

**Root cause:** DEIMv2 model code prints debug lines to stdout before the JSON result:
```
Training ViT-Tiny from scratch...
Using Lite Spatial Prior Module...
```
These lines are emitted during `import engine` and model construction, before any inference
code runs. `json.loads()` in `deim_inference.py` failed because stdout was not pure JSON.

**Fix in `deim_infer_helper.py`:** Redirect stdout to stderr before any model import,
restore only for the final JSON print:
```python
_real_stdout = sys.stdout
sys.stdout = sys.stderr          # swallow all model debug prints
# ... all imports, model loading, inference ...
sys.stdout = _real_stdout
print(json.dumps({'predictions': result, 'n_detections': len(all_lines), 'error': None}))
```
Error path also restores stdout:
```python
except Exception as e:
    sys.stdout = sys.__stdout__
    print(json.dumps({'predictions': '', 'n_detections': 0, 'error': str(e) + '\n' + traceback.format_exc()}))
```

---

## Fix 10 — `epoch_9.pth` was a symlink (breaks on server move)

**Symptom:** Moving `runpod_engine/` to a new server — `checkpoints/epoch_9.pth` was a
512-byte symlink pointing to `/sfs/work_dirs/orientedformer_custom_8cls_stage2/.../epoch_9.pth`.
On a new server that path doesn't exist.

**Fix:** Replace symlink with the actual 540 MB file:
```bash
rm /sfs/runpod_engine/checkpoints/epoch_9.pth
cp /sfs/work_dirs/orientedformer_custom_8cls_stage2/orientedformer_custom_8cls_may_w3/epoch_9.pth \
   /sfs/runpod_engine/checkpoints/epoch_9.pth
```
File is `.gitignored` so it doesn't go to GitHub but stays with the folder on disk.

---

## Fix 11 — DEIMv2 Python interpreter not found on unfamiliar servers

**Symptom:** On a new server, `is_available()` returns False because the deimv2 env's
python is not at any of the hardcoded static paths.

**Root cause:** `conda create -n deimv2` places the env in the conda installation's `envs/`
directory, which varies: `/opt/miniconda/envs/`, `/sfs/miniconda3/envs/`, `~/miniconda3/envs/`, etc.

**Fix in `deim_inference.py`:** Added `_find_deimv2_python()` which:
1. Checks a static list of common paths (fast path)
2. Falls back to `conda run -n deimv2 which python3` (works anywhere conda is installed)

```python
def _find_deimv2_python():
    static = _first_existing('/sfs/envs/deimv2/bin/python3', ...)
    if static:
        return static
    conda_exe = shutil.which('conda') or _first_existing('/opt/miniconda/bin/conda', ...)
    if conda_exe:
        result = subprocess.run([conda_exe, 'run', '-n', 'deimv2', 'which', 'python3'], ...)
        p = result.stdout.strip()
        if p and os.path.isfile(p):
            return p
    return None
```

---

## Fix 12 — `torch.prod()` CUDA kernel crash on L40S / Ada Lovelace GPUs

**Symptom:** Model loads successfully (`Model loaded successfully.` in server.log) but every
inference request returns "Stream ended without complete results from both models." No error
appears in the server log — the exception is silently swallowed inside the SSE generator.

Direct test reveals:
```
RuntimeError: CUDA driver error: invalid argument
```
at this line inside `oriented_adamixer_ddq.py`:
```python
z = (wh).prod(-1, keepdim=True).sqrt().log2()
```

**Root cause:** PyTorch 2.4.0+**cu121** has a broken `torch.prod()` CUDA kernel on GPUs with
compute capability 8.9 (Ada Lovelace architecture: L40, L40S, RTX 4090, etc.).
The cu121 build does not include the correct kernel variant for sm_89.
The cu121 build works fine on older GPUs (A100 sm_80, A5000 sm_86, etc.).

**How to confirm it's this issue:**
```bash
conda run -n ai4rs_infer python3 -c "
import torch
x = torch.abs(torch.randn(2, 100, 2)).cuda()
p = x.prod(-1, keepdim=True)   # fails here with cu121 on L40S
print('OK:', p.shape)
"
```
If it crashes → cu121 kernel bug confirmed.

**Fix:** Reinstall torch with the **cu124** build. mmcv has no cu124 wheel so keep mmcv as-is
(CUDA is backward compatible — mmcv cu121 wheel works fine with torch cu124 runtime):
```bash
source /sfs/miniconda3/etc/profile.d/conda.sh   # or /sfs/miniconda/etc/...
conda activate ai4rs_infer
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124
```
Then restart: `bash /sfs/runpod_engine/start.sh`

**Fixed in `setup.sh`:** CUDA 12 installs now always use `cu124` torch index.
`MMCV_INDEX` still points to the cu121 CDN (only available build for mmcv 2.2.0).

**Affected GPUs:** Any Ada Lovelace GPU (sm_89): L40, L40S, RTX 4090, RTX 4080, etc.
**Safe GPUs (cu121 works fine):** A100 (sm_80), A5000/A6000 (sm_86), V100 (sm_70).

---

## Change — Self-contained folder (all assets internal)

**What changed:** The folder no longer depends on any external path.

| Asset | Was | Now |
|-------|-----|-----|
| Mandara checkpoint | Symlink → `/sfs/work_dirs/...` | Real file at `checkpoints/epoch_9.pth` |
| DEIMv2 checkpoint | `/sfs/DEIMv2/outputs/.../best_stg1.pth` | `checkpoints/deim/best_stg1.pth` |
| DEIMv2 config | `/sfs/DEIMv2/configs/deimv2/...yml` | `configs/deim/deimv2_dinov3_s_vehicle.yml` |
| DEIMv2 config includes | `/sfs/DEIMv2/configs/dataset/`, `base/`, `runtime.yml` | `configs/dataset/`, `configs/base/`, `configs/runtime.yml` |
| DEIMv2 source code | `/sfs/DEIMv2/engine/` (external project) | `deim_src/engine/` (1.3 MB, bundled) |

**Key detail about DEIMv2 config includes:** `deimv2_dinov3_s_vehicle.yml` uses relative
paths: `__include__: ['../dataset/custom_detection.yml', '../runtime.yml', ...]`
These resolve relative to the config file's location, so the config must live at
`configs/deim/` with siblings `configs/dataset/`, `configs/base/`, `configs/runtime.yml`.

**DEIMv2 source import:** `deim_infer_helper.py` does:
```python
sys.path.insert(0, manifest['proj_root'])  # points to deim_src/
from engine.core import YAMLConfig          # imports deim_src/engine/core/__init__.py
```
No `pip install` of the DEIMv2 package is needed — just the `engine/` directory present.

---

## Change — DEIMv2 comparison feature added

**What was added:**
- `deim_inference.py` — tiles image identically to Mandara, calls `deim_infer_helper.py` subprocess
- `deim_infer_helper.py` — loads DEIMv2 model once, infers all tiles, outputs JSON
- `app.py:/predict/compare/stream` — SSE endpoint running Mandara then DEIMv2, saves two .txt files
- `gui.html` — side-by-side canvas comparison, per-model stats, download buttons

**Output filenames:**
```
pred/Mandara_model_NNNN_stem.txt
pred/dino_trained_model_NNNN_stem.txt
```

---

## Change — Patch View feature added to GUI

**What it does:** After inference, a "Patch View · Top 10" toggle appears below the canvases.
Shows the 10 highest-detection-density tiles (ranked by sum of Mandara calibrated scores)
with both models' boxes overlaid on the actual image crop.

**Ranking:** Uses `mandaraRaw` (all detections above 0.1 score, before display threshold)
so the top-10 tiles are stable and don't change when the user moves the score slider.

**Navigation:** `[←] [→]` buttons or keyboard arrow keys (left/right).

**Both canvases always show the same tile.** Each model's detections are independently
filtered by centroid position to only show boxes whose center falls inside the current tile.

---

## Fix 13 — DEIMv2 env missing packages (tensorboard, faster-coco-eval, calflops, transformers)

**Symptom:** GUI shows "Stream ended without complete results from both models."
Server-side error in DEIMv2 subprocess: `No module named 'tensorboard'` then `No module named 'faster_coco_eval'`.

**Root cause:** `setup.sh` only installed `pyyaml pillow numpy scipy` in the `deimv2` env.
The `engine/optim/ema.py` imports tensorboard; other engine modules need `faster-coco-eval`,
`calflops`, and `transformers`. These are in `DEIMv2/requirements.txt` but we weren't
reading that file during setup.

**Fix in `setup.sh`:** Install the full set of runtime deps:
```bash
conda run -n deimv2 pip install \
    pyyaml pillow numpy scipy tensorboard \
    faster-coco-eval calflops transformers
```
Also copied `DEIMv2/requirements.txt` → `deim_src/requirements.txt` for reference.

Note: `torch` and `torchvision` are installed separately with the correct CUDA index URL
before this step, so they are intentionally excluded here.

---

## Fix 12 — Patch script crashes on `import mmdet` / `import mmrotate`

**Symptom:** `setup.sh` step 5 exits with:
```
AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.
```
The script exits here and never reaches the mmrotate clone or deimv2 env setup.

**Root cause:** The patch scripts used `import mmdet; pathlib.Path(mmdet.__file__)` to
locate the `__init__.py`. But importing mmdet triggers the version assertion before
the patch can be applied — catch-22.

**Fix in `setup.sh`:** Use `pip show <pkg>` to find the install location without importing:
```python
def site_location(pkg):
    r = subprocess.run([sys.executable, '-m', 'pip', 'show', pkg],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('Location:'):
            return line.split(':', 1)[1].strip()
    return None

loc = site_location('mmdet')
p = pathlib.Path(loc) / 'mmdet' / '__init__.py'
# patch p directly — never import mmdet
```
Same fix applied to mmseg and mmrotate patches.

---

## Backup Environments

Full clones of both working environments saved at:
```
/sfs/env_backups/ai4rs_infer/          # Runnable backup (6 GB)
/sfs/env_backups/deimv2/               # Runnable backup (5.8 GB)
/sfs/env_backups/ai4rs_infer_spec.yml  # Full package list for diffing
/sfs/env_backups/deimv2_spec.yml       # Full package list for diffing
```

To compare a freshly built env against the backup:
```bash
conda env export -n ai4rs_infer | diff - /sfs/env_backups/ai4rs_infer_spec.yml
conda env export -n deimv2      | diff - /sfs/env_backups/deimv2_spec.yml
```

To activate backup directly (if rebuild fails):
```bash
conda activate /sfs/env_backups/ai4rs_infer
```

---

## Package Versions Known to Work

### ai4rs_infer (Mandara)
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | |
| torch | 2.4.0+cu121 | cu121 index works on CUDA 11 and 12 |
| torchvision | 0.19.0 | |
| mmengine | 0.10.7 | |
| mmcv | 2.2.0 | Pre-built wheel from openmmlab CDN, NOT via mim |
| mmdet | 3.3.0 | Patched max mmcv cap → 2.3.0 |
| mmsegmentation | ≥1.2.2 | Patched max mmcv cap → 2.3.0 |
| mmrotate | 1.0.0rc1 | 1.x branch; patched mmcv+mmdet caps |
| fastapi | 0.111.0 | |
| uvicorn | 0.30.0 | |

### deimv2 (DEIMv2)
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | |
| torch | latest stable | cu124 for CUDA 12, cu118 for CUDA 11 |
| torchvision | latest stable | |
| pyyaml | any | |
| pillow | any | |
| numpy | any | |
| scipy | any | |

---

## Server Startup Checklist

1. `checkpoints/epoch_9.pth` exists and is a real file (not a symlink) — 540 MB
2. `checkpoints/deim/best_stg1.pth` exists — 150 MB
3. `deim_src/engine/` exists — 1.3 MB
4. `configs/deim/deimv2_dinov3_s_vehicle.yml` exists
5. Run `bash setup.sh` (first time only, or after deleting envs)
6. Run `bash start.sh`
7. Check `curl http://localhost:8000/health` returns `{"model_ready":true}`
