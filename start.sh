#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mandara inference server — start after pod reset
# Usage:  bash /workspace/runpod_engine/start.sh
#
# Everything (conda, packages) lives in /workspace/miniconda — persistent.
# No reinstallation needed after pod resets.
# ─────────────────────────────────────────────────────────────────────────────
set -e
PROJ=/workspace/runpod_engine
CONDA_HOME=/workspace/miniconda

# Ensure conda paths point to /workspace/miniconda (survives pod moves)
if grep -q '/opt/miniconda' $CONDA_HOME/etc/profile.d/conda.sh 2>/dev/null; then
    sed -i 's|/opt/miniconda|/workspace/miniconda|g' $CONDA_HOME/etc/profile.d/conda.sh
    grep -rl '#!/opt/miniconda' $CONDA_HOME/bin/ 2>/dev/null | xargs sed -i 's|#!/opt/miniconda|#!/workspace/miniconda|g' 2>/dev/null || true
fi

echo "=== Activating conda from persistent storage ==="
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ai4rs

echo "=== Starting inference server ==="
cd $PROJ
export PYTHONPATH=$PROJ:$PYTHONPATH

# Kill any old server instance
pkill -f "python app.py" 2>/dev/null || true
sleep 1

nohup python app.py > $PROJ/server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for model to load (up to 60s)
echo "Waiting for model to load..."
for i in $(seq 1 30); do
    sleep 2
    STATUS=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('model_ready',''))" 2>/dev/null) || true
    if [ "$STATUS" = "True" ]; then
        echo "Model loaded! ($((i*2))s)"
        break
    fi
done

echo ""
echo "=== Starting public tunnel ==="
pkill cloudflared 2>/dev/null || true
sleep 1
nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
    > $PROJ/tunnel.log 2>&1 &
sleep 6
PUBLIC_URL=$(grep -o 'https://[a-z0-9\-]*\.trycloudflare\.com' $PROJ/tunnel.log | head -1)

RUNPOD_URL=""
if [ -n "$RUNPOD_POD_ID" ]; then
    RUNPOD_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Mandara is live!                                            ║"
echo "║                                                              ║"
echo "║  Local:  http://localhost:8000                               ║"
if [ -n "$RUNPOD_URL" ]; then
printf  "║  RunPod: %-52s║\n" "$RUNPOD_URL"
fi
if [ -n "$PUBLIC_URL" ]; then
printf  "║  Tunnel: %-52s║\n" "$PUBLIC_URL"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
