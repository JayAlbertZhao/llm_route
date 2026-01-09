#!/bin/bash
# Install modelscope if not exists
pip install modelscope

# Download Qwen3-8B to a local directory
# This uses ModelScope which is faster in China
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', cache_dir='./models')"

echo "Model downloaded to ./models/Qwen/Qwen3-8B"

