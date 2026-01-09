import json
import time
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from src.router.strategies.round_robin import RoundRobinStrategy
from src.backend.vllm_client import VLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Router")

app = FastAPI(title="vLLM Router")

# Global Configuration (TODO: Move to config file)
# AutoDL/SeetaCloud Deployment:
# Router listens on 6006 (mapped to public 8443)
# vLLM instances on 8081-8084
BACKENDS = [
    "http://localhost:8081",
    "http://localhost:8082",
    "http://localhost:8083",
    "http://localhost:8084"
]
STRATEGY = RoundRobinStrategy()
CLIENT = VLLMClient()

# System State (Mock)
SYSTEM_STATE = {
    "backends": {b: {"active_requests": 0} for b in BACKENDS}
}

@app.on_event("startup")
async def startup_event():
    logger.info(f"Router started with backends: {BACKENDS}")

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

    # Update State
    SYSTEM_STATE["backends"][backend_url]["active_requests"] += 1
    req_id = payload.get("id", "unknown")
    
    logger.info(f"Routing request {req_id} to {backend_url}")

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
                        logger.info(f"Request {req_id} TTFT: {ttft:.4f}s")
                    
                    token_count += 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                elif msg.get("type") == "usage":
                    # End of stream
                    pass
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            # Update State
            SYSTEM_STATE["backends"][backend_url]["active_requests"] -= 1
            duration = time.time() - start_time
            logger.info(f"Request {req_id} completed. Tokens: {token_count}, Duration: {duration:.4f}s")
            yield "data: [DONE]\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "backends": len(BACKENDS)}

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    
    logger.info(f"Starting Router on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


