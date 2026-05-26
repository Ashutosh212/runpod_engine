#!/bin/bash
# One-shot setup for OrientedFormer inference pod (CUDA 12.4, A5000)
set -e

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    source /opt/miniconda/etc/profile.d/conda.sh
    conda init bash
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 2. Create conda env from yml
conda env create -f environment.yml -n ai4rs_infer || conda env update -f environment.yml -n ai4rs_infer

conda activate ai4rs_infer

# 3. Install mmcv wheel (pre-built for cu124 + torch 2.4)
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html

# 4. Install mmrotate from source (no PyPI wheel for 1.0.0rc1)
if [ ! -d "/tmp/mmrotate" ]; then
    git clone https://github.com/open-mmlab/mmrotate.git /tmp/mmrotate
fi
pip install -e /tmp/mmrotate --no-build-isolation

echo "Setup complete. Start server with:"
echo "  conda activate ai4rs_infer && cd /path/to/inference_endpoint && python app.py"
