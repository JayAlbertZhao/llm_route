#!/bin/bash
# Start a single vLLM instance in foreground
# Usage: bash scripts/run_vllm_node.sh --id <0-3>

# Fix for "ImportError: libcudart.so.12: cannot open shared object file"
# Add torch lib path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')

# Default args
NODE_ID=0

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --id) NODE_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Configuration Map
# ID -> GPU_ID, PORT
PORT=$((8081 + NODE_ID))
GPU_ID=$NODE_ID

MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-8B"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at $MODEL_PATH"
    exit 1
fi

echo "🚀 Starting vLLM Node $NODE_ID"
echo "   GPU: $GPU_ID"
echo "   Port: $PORT"
echo "   Model: $MODEL_PATH"
echo "-----------------------------------"

# Run vLLM in foreground
# Added --trust-remote-code for Qwen
CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "qwen-8b" \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --trust-remote-code

