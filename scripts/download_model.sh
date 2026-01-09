#!/bin/bash
# Install modelscope if not exists
pip install modelscope

# Download Qwen3-8B to Data Disk (autodl-tmp)
# This uses ModelScope which is faster in China
mkdir -p /root/autodl-tmp/models

python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', cache_dir='/root/autodl-tmp/models')"

echo "Model downloaded to /root/autodl-tmp/models/Qwen/Qwen3-8B"

