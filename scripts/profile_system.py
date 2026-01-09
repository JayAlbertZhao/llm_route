import asyncio
import aiohttp
import json
import time
import pandas as pd
import numpy as np
import argparse
import random
from src.client.workload import WorkloadLoader

# Profiling Config
# Default URL (can be overridden by args)
TARGET_URL = "http://localhost:8081/v1/chat/completions" 
OUTPUT_FILE = "data/profiling_data.csv"
MODEL_NAME = "qwen-8b"

class Profiler:
    def __init__(self, workload_path, target_url):
        self.loader = WorkloadLoader(workload_path)
        self.loader.load()
        # Flatten buckets into a single list for random sampling
        self.all_data = [item for bucket in self.loader.data_buckets.values() for item in bucket]
        self.results = []
        self.target_url = target_url
        
    async def send_request(self, session, prompt, current_rps, active_reqs):
        # ... (same as before)
        
        try:
            # Use self.target_url instead of global constant
            async with session.post(self.target_url, json=payload) as response:
                if response.status != 200:
                    return None
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if ttft is None:
                                ttft = time.time() - start_time
                            token_count += 1
                        except:
                            pass
                            
            duration = time.time() - start_time
            tbt = (duration - ttft) / (token_count - 1) if token_count > 1 and ttft else 0
            
            return {
                "input_len": len(prompt) // 4, # Approx token len
                "system_load": current_rps,    # We use RPS as a proxy for load intensity in this simplified profile
                "active_reqs": active_reqs,    # Concurrency
                "ttft": ttft,
                "tbt": tbt,
                "total_time": duration
            }
        except Exception as e:
            # print(f"Error: {e}")
            return None

    async def run_phase(self, rps, duration_sec=30):
        print(f"Profiling RPS={rps} for {duration_sec}s...")
        start_time = time.time()
        tasks = []
        active_reqs = 0
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_sec:
                # Poisson arrival
                wait_time = np.random.exponential(1.0 / rps)
                await asyncio.sleep(wait_time)
                
                # Sample prompt
                item = random.choice(self.all_data)
                prompt = item["prompt"]
                
                # Track concurrency
                # Note: this is client-side view of concurrency, roughly approximates server load
                active_reqs += 1
                
                # Launch task
                task = asyncio.create_task(self.send_request(session, prompt, rps, active_reqs))
                
                def callback(fut):
                    nonlocal active_reqs
                    active_reqs -= 1
                    res = fut.result()
                    if res:
                        self.results.append(res)

                task.add_done_callback(callback)
                tasks.append(task)
                
                # Cleanup
                tasks = [t for t in tasks if not t.done()]
            
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self):
        # Sweep RPS from low to high
        # Adjust range based on your GPU capacity. For 8B model, single card might handle 5-20 RPS depending on prompt len.
        rps_levels = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
        
        for rps in rps_levels:
            await self.run_phase(rps)
            # Cooldown
            print("Cooldown 5s...")
            await asyncio.sleep(5)
            
        # Save
        df = pd.DataFrame(self.results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    import argparse
    
    # Load default from secrets if available
    default_url = "http://localhost:8081/v1/chat/completions"
    try:
        import yaml
        with open("config/secrets.yaml", "r") as f:
            conf = yaml.safe_load(f)
            if "public_url" in conf:
                default_url = conf["public_url"] + "/v1/chat/completions"
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=default_url, help="Target URL (e.g., http://localhost:8081/v1/chat/completions)")
    parser.add_argument("--workload", type=str, default="data/processed_workload.jsonl")
    args = parser.parse_args()

    profiler = Profiler(args.workload, args.url)
    asyncio.run(profiler.run())

