#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mandara inference server — start
# Works wherever the repo is cloned and on any server.
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Project root: $PROJ"

# ── 1. Find conda ─────────────────────────────────────────────────────────────
CONDA_BASE=""

if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

if [ -z "$CONDA_BASE" ]; then
    for candidate in /opt/miniconda /sfs/miniconda /workspace/miniconda \
                     "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$candidate"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found. Run:  bash $PROJ/setup.sh"
    exit 1
fi

echo "==> Using conda at: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 2. Activate env ───────────────────────────────────────────────────────────
AI4RS_ENV="/workspace/envs/ai4rs_infer"

if [ ! -d "$AI4RS_ENV" ]; then
    echo "ERROR: env not found at $AI4RS_ENV — run:  bash $PROJ/setup.sh"
    exit 1
fi

conda activate "$AI4RS_ENV"

# ── 3. Start inference server ─────────────────────────────────────────────────
export PYTHONPATH="$PROJ:$PYTHONPATH"
cd "$PROJ"

echo "==> Stopping any running server..."
pkill -f "python app.py" 2>/dev/null || true
sleep 1

echo "==> Starting server..."
nohup python app.py > "$PROJ/server.log" 2>&1 &
SERVER_PID=$!
echo "    PID: $SERVER_PID  |  Log: $PROJ/server.log"

# ── 4. Wait for model to load ─────────────────────────────────────────────────
echo "==> Waiting for model to load (up to 60s)..."
LOADED=false
for i in $(seq 1 30); do
    sleep 2
    STATUS=$(curl -s http://localhost:8000/health 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_ready',''))" \
        2>/dev/null) || true
    if [ "$STATUS" = "True" ]; then
        echo "    Model ready in $((i*2))s"
        LOADED=true
        break
    fi
done

if [ "$LOADED" = "false" ]; then
    echo "    WARNING: Model not loaded yet — check $PROJ/server.log"
fi

# ── 5. Optional Cloudflare tunnel ────────────────────────────────────────────
PUBLIC_URL=""
if command -v cloudflared &>/dev/null; then
    echo "==> Starting Cloudflare tunnel..."
    pkill cloudflared 2>/dev/null || true
    sleep 1
    nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
        > "$PROJ/tunnel.log" 2>&1 &
    sleep 6
    PUBLIC_URL=$(grep -o 'https://[a-z0-9\-]*\.trycloudflare\.com' "$PROJ/tunnel.log" | head -1)
fi

RUNPOD_URL=""
if [ -n "$RUNPOD_POD_ID" ]; then
    RUNPOD_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Mandara is live!                                            ║"
echo "║                                                              ║"
echo "║  Local:   http://localhost:8000                              ║"
[ -n "$RUNPOD_URL" ] && printf "║  RunPod:  %-50s║\n" "$RUNPOD_URL"
[ -n "$PUBLIC_URL" ] && printf "║  Tunnel:  %-50s║\n" "$PUBLIC_URL"
echo "╚══════════════════════════════════════════════════════════════╝"
