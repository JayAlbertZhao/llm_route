import time
import json
import aiohttp
import asyncio
import logging
from typing import Dict, Any, AsyncGenerator

logger = logging.getLogger(__name__)

class VLLMClient:
    def __init__(self, timeout=300):
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def stream_chat(self, backend_url: str, payload: Dict[str, Any]) -> AsyncGenerator[Dict, None]:
        """
        Forwards request to backend and yields chunks. 
        Injects TTFT metric into the stream or returns it.
        """
        endpoint = f"{backend_url}/v1/chat/completions"
        # Ensure stream is True to measure TTFT
        payload["stream"] = True 
        
        start_time = time.time()
        ttft = None
        first_token_received = False
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(endpoint, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Backend {backend_url} error {response.status}: {error_text}")
                        yield {"error": f"HTTP {response.status}", "details": error_text}
                        return

                    async for line in response.content:
                        if not line:
                            continue
                        
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                yield {"type": "usage", "data": "[DONE]"} # custom marker or just forward
                                break
                            
                            try:
                                chunk = json.loads(data_str)
                                
                                # Check if this chunk actually has content (sometimes first chunk is empty role)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if not first_token_received and content:
                                    ttft = time.time() - start_time
                                    first_token_received = True
                                    # We can log here or yield a special internal metric packet
                                    # For simplicity, let's yield the chunk and handle metrics in the router wrapper
                                    # But to be precise, the router loop controlling this generator captures the time
                                
                                yield {"type": "chunk", "data": chunk, "timestamp": time.time()}
                                
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {data_str}")
                                continue
        except Exception as e:
            logger.error(f"Request to {backend_url} failed: {e}")
            yield {"error": str(e)}

    async def health_check(self, backend_url: str) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{backend_url}/health") as response:
                    return response.status == 200
        except:
            return False


