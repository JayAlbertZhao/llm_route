import asyncio
import time
import random
import uuid
import json
from typing import Optional, List, Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import argparse

app = FastAPI()

# Configuration for simulation
class SimulationConfig:
    TTFT_MEAN = 0.1  # seconds
    TTFT_STD = 0.02
    TPOT_MEAN = 0.02 # seconds (50 tok/s)
    TPOT_STD = 0.005
    MAX_TOKENS_DEFAULT = 16

config = SimulationConfig()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

async def simulate_processing_delay(input_len: int):
    # Simulate some load-dependent delay? 
    # For now just random TTFT
    delay = max(0.01, random.normalvariate(config.TTFT_MEAN, config.TTFT_STD))
    await asyncio.sleep(delay)

async def stream_generator(request_id: str, model: str, output_len: int):
    created = int(time.time())
    
    # 1. First token (TTFT already handled before calling this, or handle here)
    # We'll simulate token-by-token generation
    for i in range(output_len):
        # Simulate decode time
        step_delay = max(0.005, random.normalvariate(config.TPOT_MEAN, config.TPOT_STD))
        await asyncio.sleep(step_delay)
        
        token_text = f" tok_{i}"
        
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # End of stream
    chunk_end = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(chunk_end)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    # Estimate input length (rough char count / 4)
    full_prompt = "".join([m.content for m in request.messages])
    input_len = len(full_prompt) // 4
    
    # TTFT Simulation
    await simulate_processing_delay(input_len)
    
    output_len = request.max_tokens if request.max_tokens else config.MAX_TOKENS_DEFAULT
    
    if request.stream:
        return StreamingResponse(stream_generator(request_id, request.model, output_len), media_type="text/event-stream")
    else:
        # Non-streaming: wait for all tokens
        total_gen_time = output_len * config.TPOT_MEAN
        await asyncio.sleep(total_gen_time)
        
        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Simulated response of {output_len} tokens."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": input_len + output_len
            }
        })

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Similar to chat completions but for raw prompts
    request_id = f"cmpl-{uuid.uuid4()}"
    prompt_str = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    input_len = len(prompt_str) // 4
    
    await simulate_processing_delay(input_len)
    output_len = request.max_tokens if request.max_tokens else config.MAX_TOKENS_DEFAULT
    
    if request.stream:
        # Reuse generator but adjust format if needed (vLLM usually unifies chunk format often)
        # Standard completion chunk has 'text' instead of 'delta'
        async def completion_generator():
            created = int(time.time())
            for i in range(output_len):
                await asyncio.sleep(config.TPOT_MEAN)
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [{"text": f" tok_{i}", "index": 0, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(completion_generator(), media_type="text/event-stream")
    else:
        await asyncio.sleep(output_len * config.TPOT_MEAN)
        return JSONResponse({
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"text": f"Simulated completion", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": output_len, "total_tokens": input_len + output_len}
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)


