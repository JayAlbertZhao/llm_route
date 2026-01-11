import json
import time
import logging
import asyncio
import aiohttp
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from src.router.strategies.round_robin import RoundRobinStrategy
from src.router.strategies.hierarchical import HierarchicalStrategy
from src.backend.vllm_client import VLLMClient
from src.utils.logger import setup_logger

# Setup logging
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "default_experiment")
logger, _ = setup_logger("Router", experiment_id=EXPERIMENT_ID)

app = FastAPI(title="vLLM Router")

# Load Configuration
BACKENDS = [
    "http://localhost:8081",
    "http://localhost:8082",
    "http://localhost:8083",
    "http://localhost:8084"
]
try:
    import yaml
    with open("config/secrets.yaml", "r") as f:
        conf = yaml.safe_load(f)
        if "backends" in conf:
            BACKENDS = conf["backends"]
except Exception:
    pass 

# Switch to Hierarchical Strategy
strategy_name = os.environ.get("ROUTER_STRATEGY", "hierarchical").lower()
if strategy_name == "rr" or strategy_name == "round_robin":
    STRATEGY = RoundRobinStrategy()
else:
    STRATEGY = HierarchicalStrategy()
    
logger.info(f"Initialized Strategy: {type(STRATEGY).__name__}")
CLIENT = VLLMClient()

# System State (Grey-box)
# Structure: {backend_url: {"active_requests": int, "num_running": int, "num_waiting": int, "last_update": float}}
SYSTEM_STATE = {
    "backends": {b: {"active_requests": 0, "num_running": 0, "num_waiting": 0} for b in BACKENDS}
}

async def poll_metrics():
    """Background task to poll /metrics from all backends."""
    logger.info("Starting Metrics Poller...")
    while True:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
                tasks = []
                for url in BACKENDS:
                    tasks.append(fetch_metrics(session, url))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, res in zip(BACKENDS, results):
                    if isinstance(res, dict):
                        # Update shared state
                        # Note: active_requests is maintained by Router itself (client-side count)
                        # We only update server-side metrics here
                        SYSTEM_STATE["backends"][url]["num_running"] = res.get("num_running", 0)
                        SYSTEM_STATE["backends"][url]["num_waiting"] = res.get("num_waiting", 0)
                        SYSTEM_STATE["backends"][url]["gpu_cache_usage"] = res.get("gpu_cache_usage", 0)
                        SYSTEM_STATE["backends"][url]["last_update"] = time.time()
                    else:
                        # logger.warning(f"Failed to poll {url}: {res}")
                        pass
        except Exception as e:
            logger.error(f"Poller loop error: {e}")
            
        await asyncio.sleep(0.1) # 10Hz polling

async def fetch_metrics(session, base_url):
    try:
        url = f"{base_url}/metrics"
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                return parse_metrics(text)
    except Exception as e:
        return e
    return None

def parse_metrics(text):
    data = {}
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            try: data["num_running"] = float(line.split()[-1])
            except: pass
        elif line.startswith("vllm:num_requests_waiting"):
            try: data["num_waiting"] = float(line.split()[-1])
            except: pass
        elif line.startswith("vllm:gpu_cache_usage_perc"):
            try: data["gpu_cache_usage"] = float(line.split()[-1])
            except: pass
    return data

@app.on_event("startup")
async def startup_event():
    logger.info(f"Router started with backends: {BACKENDS}")
    logger.info(f"Using Strategy: {type(STRATEGY).__name__}")
    # Start poller
    asyncio.create_task(poll_metrics())

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. Select Backend
    try:
        backend_url = await STRATEGY.select_backend(payload, BACKENDS, SYSTEM_STATE)
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail="Routing failed")

    # Update State (Client-side tracking)
    SYSTEM_STATE["backends"][backend_url]["active_requests"] += 1
    req_id = payload.get("id", "unknown")
    
    # logger.info(f"Routing request {req_id} to {backend_url}")

    # 2. Forward and Stream
    async def response_generator():
        start_time = time.time()
        ttft = None
        token_count = 0
        
        try:
            async for msg in CLIENT.stream_chat(backend_url, payload):
                if "error" in msg:
                    logger.error(f"Backend error: {msg['error']}")
                    yield f"data: {json.dumps({'error': msg['error']})}\n\n"
                    break
                
                if msg.get("type") == "chunk":
                    chunk = msg["data"]
                    
                    # Track TTFT
                    if ttft is None:
                        ttft = time.time() - start_time
                        # Log less frequently to reduce noise in high throughput
                        # logger.info(f"Request {req_id} TTFT: {ttft:.4f}s")
                    
                    token_count += 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                elif msg.get("type") == "usage":
                    pass
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            # Update State
            SYSTEM_STATE["backends"][backend_url]["active_requests"] -= 1
            duration = time.time() - start_time
            # logger.info(f"Request {req_id} completed. Tokens: {token_count}, Duration: {duration:.4f}s")
            yield "data: [DONE]\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "backends": len(BACKENDS), "strategy": type(STRATEGY).__name__}

if __name__ == "__main__":
    import uvicorn
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--exp_id", type=str, default="default_experiment")
    args = parser.parse_args()
    
    if args.exp_id != "default_experiment":
        logger, _ = setup_logger("Router", experiment_id=args.exp_id)
        os.environ["EXPERIMENT_ID"] = args.exp_id

    logger.info(f"Starting Router on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
