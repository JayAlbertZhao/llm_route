import asyncio
import aiohttp
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import os
from src.client.workload import WorkloadLoader

# Configuration
OUTPUT_DIR = "logs/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ExperimentRunner:
    def __init__(self, router_url, workload_path, rps, duration, scenario="random"):
        self.router_url = router_url
        self.rps = rps
        self.duration = duration
        self.scenario = scenario
        
        # Load Data
        self.loader = WorkloadLoader(workload_path)
        self.loader.load()
        # Ensure we have buckets loaded
        # WorkloadLoader currently just loads a flat list, but let's re-bucket them in memory for sampling
        self.buckets = {
            "short": [],   # < 64
            "medium": [],  # 64 - 512
            "long": [],    # 512 - 2048
            "extra_long": [] # > 2048
        }
        self._bucket_data()
        
    def _bucket_data(self):
        print("Bucketing data for scenario sampling...")
        for item in self.loader.data:
            length = item.get("token_len", 0)
            if length < 64: self.buckets["short"].append(item)
            elif length < 512: self.buckets["medium"].append(item)
            elif length < 2048: self.buckets["long"].append(item)
            else: self.buckets["extra_long"].append(item)
            
        print(f"Buckets: Short={len(self.buckets['short'])}, Med={len(self.buckets['medium'])}, Long={len(self.buckets['long'])}, XLong={len(self.buckets['extra_long'])}")

    def get_next_prompt(self, request_idx):
        """
        Custom Sampling Logic based on Scenario.
        """
        # Scenario 1: Random (Natural Distribution)
        if self.scenario == "random":
            return random.choice(self.loader.data)["prompt"]
            
        # Scenario 2: Worst Case for Round Robin (3 Short, 1 Long Cyclic)
        # This creates a convoy effect where Short requests get stuck behind Long ones
        elif self.scenario == "rr_worst_case":
            if request_idx % 4 == 0:
                # Every 4th request is Long
                return random.choice(self.buckets["long"] + self.buckets["extra_long"])["prompt"]
            else:
                # Others are Short
                return random.choice(self.buckets["short"])["prompt"]
                
        # Scenario 3: Long Heavy (Stress Test)
        elif self.scenario == "long_heavy":
            if random.random() < 0.5:
                return random.choice(self.buckets["long"])["prompt"]
            else:
                return random.choice(self.buckets["medium"])["prompt"]
                
        # Default
        return random.choice(self.loader.data)["prompt"]

    async def send_request(self, session, idx, prompt):
        payload = {
            "model": "qwen-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "stream": True,
            "id": f"req_{idx}"
        }
        
        start_time = time.time()
        ttft = None
        
        try:
            async with session.post(self.router_url, json=payload) as response:
                if response.status != 200:
                    return {"error": response.status, "ts": start_time}
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        if ttft is None:
                            ttft = time.time() - start_time
                        if line == "data: [DONE]":
                            break
        except Exception as e:
            return {"error": str(e), "ts": start_time}
            
        duration = time.time() - start_time
        return {
            "req_id": idx,
            "start_ts": start_time,
            "ttft": ttft,
            "duration": duration,
            "prompt_len": len(prompt) // 4 # Approx
        }

    async def run(self):
        print(f"Starting Experiment: RPS={self.rps}, Duration={self.duration}s, Scenario={self.scenario}")
        results = []
        start_exp = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            req_idx = 0
            
            while time.time() - start_exp < self.duration:
                # Poisson arrival
                wait_time = np.random.exponential(1.0 / self.rps)
                await asyncio.sleep(wait_time)
                
                prompt = self.get_next_prompt(req_idx)
                
                task = asyncio.create_task(self.send_request(session, req_idx, prompt))
                tasks.append(task)
                req_idx += 1
                
                # Cleanup finished tasks
                done = [t for t in tasks if t.done()]
                tasks = [t for t in tasks if not t.done()]
                for t in done:
                    res = t.result()
                    if res: results.append(res)
            
            # Wait for pending
            print("Waiting for pending requests...")
            if tasks:
                done_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in done_results:
                    if isinstance(res, dict): results.append(res)
                    
        # Analyze
        df = pd.DataFrame(results)
        df = df.dropna(subset=["ttft"])
        
        print("\n--- Results ---")
        print(f"Total Requests: {len(df)}")
        if len(df) > 0:
            print(f"Mean TTFT: {df['ttft'].mean():.4f}s")
            print(f"P50 TTFT: {df['ttft'].quantile(0.5):.4f}s")
            print(f"P95 TTFT: {df['ttft'].quantile(0.95):.4f}s")
            print(f"P99 TTFT: {df['ttft'].quantile(0.99):.4f}s")
            print(f"Max TTFT: {df['ttft'].max():.4f}s")
            
            # Save
            filename = f"exp_rps{self.rps}_{self.scenario}_{int(time.time())}.csv"
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"Saved details to {path}")
            
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--scenario", type=str, default="random", choices=["random", "rr_worst_case", "long_heavy"])
    parser.add_argument("--url", type=str, default="http://localhost:6006/v1/chat/completions")
    parser.add_argument("--workload", type=str, default="data/processed_workload.jsonl")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.url, args.workload, args.rps, args.duration, args.scenario)
    asyncio.run(runner.run())

