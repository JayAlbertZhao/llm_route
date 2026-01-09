import asyncio
import time
import random
import aiohttp
import logging
import json
import uuid
import numpy as np
import ssl
from typing import Literal
from src.client.workload import WorkloadLoader

logger = logging.getLogger("TrafficGen")
logging.basicConfig(level=logging.INFO)

class TrafficGenerator:
    def __init__(self, 
                 router_url: str, 
                 workload_path: str, 
                 rps: float, 
                 distribution: Literal["poisson", "constant", "burst"] = "poisson",
                 duration: int = 60):
        self.router_url = router_url
        self.rps = rps
        self.distribution = distribution
        self.duration = duration
        self.loader = WorkloadLoader(workload_path)
        self.iterator = self.loader.get_iterator(loop=True)
        self.stats = {
            "sent": 0,
            "completed": 0,
            "errors": 0,
            "ttft": [],
            "latency": []
        }

    def get_inter_arrival_time(self) -> float:
        if self.distribution == "constant":
            return 1.0 / self.rps
        elif self.distribution == "poisson":
            return np.random.exponential(1.0 / self.rps)
        elif self.distribution == "burst":
            if random.random() < 0.1: # 10% chance of burst
                return 1.0 / (self.rps * 5)
            else:
                return 1.0 / (self.rps * 0.5)
        return 1.0 / self.rps

    async def send_request(self, session: aiohttp.ClientSession, request_id: str, prompt_data: dict):
        url = f"{self.router_url}/v1/chat/completions"
        prompt = prompt_data.get("prompt", "Hello")
        
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "id": request_id
        }
        
        start_time = time.time()
        ttft = None
        
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Req {request_id} failed: {response.status}")
                    self.stats["errors"] += 1
                    return

                try:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                _ = json.loads(data)
                                if ttft is None:
                                    ttft = time.time() - start_time
                                    self.stats["ttft"].append(ttft)
                            except:
                                pass
                except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError) as e:
                     pass # Ignore abrupt closures
                except Exception as e:
                     if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in str(e):
                         pass
                     else:
                         raise e
                            
            latency = time.time() - start_time
            self.stats["latency"].append(latency)
            self.stats["completed"] += 1
            if self.stats["completed"] % 10 == 0:
                logger.info(f"Completed {self.stats['completed']} reqs. Avg TTFT: {np.mean(self.stats['ttft']):.4f}s")
                
        except Exception as e:
            if "APPLICATION_DATA_AFTER_CLOSE_NOTIFY" in str(e):
                # Count as success if we got connection closed but likely finished
                pass
            else:
                logger.error(f"Req {request_id} error: {e}")
                self.stats["errors"] += 1

    async def run(self):
        logger.info(f"Starting traffic generation: {self.distribution} @ {self.rps} RPS for {self.duration}s")
        start_time = time.time()
        
        # Disable SSL verification for testing if needed, or just standard session
        connector = aiohttp.TCPConnector(ssl=False) 
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            while time.time() - start_time < self.duration:
                # 1. Wait
                wait_time = self.get_inter_arrival_time()
                await asyncio.sleep(wait_time)
                
                # 2. Prepare Request
                prompt_data = next(self.iterator)
                req_id = f"req-{uuid.uuid4().hex[:8]}"
                self.stats["sent"] += 1
                
                # 3. Spawn Task
                task = asyncio.create_task(self.send_request(session, req_id, prompt_data))
                tasks.append(task)
                
                # Cleanup finished tasks occasionally
                tasks = [t for t in tasks if not t.done()]
            
            # Wait for pending
            logger.info("Stopping generation, waiting for pending requests...")
            await asyncio.gather(*tasks)
            
        logger.info("Finished.")
        logger.info(f"Stats: {self.stats['completed']}/{self.stats['sent']} completed. Errors: {self.stats['errors']}")
        if self.stats["ttft"]:
            logger.info(f"Mean TTFT: {np.mean(self.stats['ttft']):.4f}s")

if __name__ == "__main__":
    import argparse
    # Load config if available
    default_url = "http://localhost:6006"
    try:
        import yaml
        with open("config/secrets.yaml", "r") as f:
            conf = yaml.safe_load(f)
            if "public_url" in conf:
                default_url = conf["public_url"]
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=default_url)
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--dist", type=str, default="poisson")
    args = parser.parse_args()

    gen = TrafficGenerator(args.url, "data/processed_workload.jsonl", rps=args.rps, duration=args.duration, distribution=args.dist)
    asyncio.run(gen.run())
