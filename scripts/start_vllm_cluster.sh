#!/bin/bash
# Start 4 vLLM instances on 4 GPUs
# Usage: bash scripts/start_vllm_cluster.sh

# Update dependencies if needed (Qwen3 requires vllm>=0.8.5)
# pip install --upgrade vllm transformers

MODEL_PATH="./models/Qwen/Qwen3-8B"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH. Please run scripts/download_model.sh first."
    exit 1
fi

echo "Starting vLLM Cluster..."

# GPU 0 -> Port 8081
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "qwen-8b" \
    --port 8081 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    > logs/vllm_8081.log 2>&1 &
echo "Started vLLM on GPU 0 (Port 8081)"

# GPU 1 -> Port 8082
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "qwen-8b" \
    --port 8082 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    > logs/vllm_8082.log 2>&1 &
echo "Started vLLM on GPU 1 (Port 8082)"

# GPU 2 -> Port 8083
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "qwen-8b" \
    --port 8083 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    > logs/vllm_8083.log 2>&1 &
echo "Started vLLM on GPU 2 (Port 8083)"

# GPU 3 -> Port 8084
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "qwen-8b" \
    --port 8084 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    > logs/vllm_8084.log 2>&1 &
echo "Started vLLM on GPU 3 (Port 8084)"

echo "All vLLM instances started. Check logs/vllm_*.log for status."

