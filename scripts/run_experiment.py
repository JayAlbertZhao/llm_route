import asyncio
import aiohttp
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import os
import csv
from src.client.workload import WorkloadLoader

# Configuration
OUTPUT_DIR = "logs/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RealTimeCSVWriter:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = open(filepath, "w", newline="", buffering=1) # Line buffering
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row):
        try:
            self.writer.writerow(row)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def close(self):
        self.file.close()

class ExperimentRunner:
    def __init__(self, router_url, workload_path, rps, duration, scenario="random", arrival="poisson"):
        self.router_url = router_url
        self.rps = rps
        self.duration = duration
        self.scenario = scenario
        self.arrival = arrival
        
        # Load Data
        self.loader = WorkloadLoader(workload_path)
        self.loader.load()
        self.buckets = {
            "short": [],   # < 64
            "medium": [],  # 64 - 512
            "long": [],    # 512 - 2048
            "extra_long": [] # > 2048
        }
        self._bucket_data()
        
        # Setup CSV Writer
        self.filename = f"exp_{self.arrival}_{self.scenario}_rps{self.rps}_{int(time.time())}.csv"
        self.filepath = os.path.join(OUTPUT_DIR, self.filename)
        self.fieldnames = ["req_id", "start_ts", "ttft", "duration", "prompt_len", "error", "rps", "scenario", "arrival"]
        self.writer = RealTimeCSVWriter(self.filepath, self.fieldnames)
        
    def _bucket_data(self):
        for item in self.loader.data:
            length = item.get("token_len", 0)
            if length < 64: self.buckets["short"].append(item)
            elif length < 512: self.buckets["medium"].append(item)
            elif length < 2048: self.buckets["long"].append(item)
            else: self.buckets["extra_long"].append(item)

    def get_next_prompt(self, request_idx):
        if self.scenario == "random":
            return random.choice(self.loader.data)["prompt"]
        elif self.scenario == "rr_worst_case":
            if request_idx % 4 == 0:
                return random.choice(self.buckets["long"] + self.buckets["extra_long"])["prompt"]
            else:
                return random.choice(self.buckets["short"])["prompt"]
        elif self.scenario == "long_heavy":
            if random.random() < 0.7: 
                return random.choice(self.buckets["long"] + self.buckets["extra_long"])["prompt"]
            else:
                return random.choice(self.buckets["medium"])["prompt"]
        elif self.scenario == "short_only":
            return random.choice(self.buckets["short"])["prompt"]
        elif self.scenario == "bimodal":
            if random.random() < 0.5:
                return random.choice(self.buckets["short"])["prompt"]
            else:
                return random.choice(self.buckets["extra_long"])["prompt"]
        return random.choice(self.loader.data)["prompt"]

    def get_inter_arrival_time(self):
        target_interval = 1.0 / self.rps
        if self.arrival == "poisson":
            return np.random.exponential(target_interval)
        elif self.arrival == "constant":
            return target_interval
        elif self.arrival == "burst":
            return np.random.gamma(shape=0.5, scale=2.0 * target_interval)
        return target_interval

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
        error = None
        duration = 0
        
        try:
            async with session.post(self.router_url, json=payload, timeout=90) as response: # Increased request timeout
                if response.status != 200:
                    error = f"HTTP {response.status}"
                else:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            if ttft is None:
                                ttft = time.time() - start_time
                            if line == "data: [DONE]":
                                break
        except Exception as e:
            error = str(e)
            
        duration = time.time() - start_time
        
        # Real-time Write
        row = {
            "req_id": idx,
            "start_ts": start_time,
            "ttft": ttft,
            "duration": duration,
            "prompt_len": len(prompt) // 4,
            "error": error,
            "rps": self.rps,
            "scenario": self.scenario,
            "arrival": self.arrival
        }
        self.writer.write(row)
        return row

    async def run(self):
        print(f"Starting Experiment: RPS={self.rps}, Arrival={self.arrival}, Scenario={self.scenario}")
        results = []
        start_exp = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            req_idx = 0
            
            while time.time() - start_exp < self.duration:
                wait_time = self.get_inter_arrival_time()
                await asyncio.sleep(wait_time)
                
                prompt = self.get_next_prompt(req_idx)
                task = asyncio.create_task(self.send_request(session, req_idx, prompt))
                tasks.append(task)
                req_idx += 1
                
                if req_idx % 100 == 0:
                     tasks = [t for t in tasks if not t.done()]
            
            # Wait for pending (with timeout)
            print(f"Waiting for {len([t for t in tasks if not t.done()])} pending requests...")
            if tasks:
                try:
                    # Timeout set to 20s to drain pending
                    done_results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=20)
                    for res in done_results:
                        if isinstance(res, dict): results.append(res)
                except asyncio.TimeoutError:
                    print("Timeout waiting for pending requests. Cancelling...")
                    for t in tasks: t.cancel()
        
        self.writer.close()
        print(f"Saved details to {self.filepath}")

        # Post-analysis for Summary
        # We read back the file to ensure we analyze exactly what was saved
        try:
            df = pd.read_csv(self.filepath)
            success = df[df["ttft"].notna()]
            errors = df[df["error"].notna()]
            
            print("\n--- Results ---")
            print(f"Total Sent: {len(df)}")
            print(f"Success: {len(success)}")
            print(f"Errors: {len(errors)}")
            
            if len(success) > 0:
                print(f"Mean TTFT: {success['ttft'].mean():.4f}s")
                print(f"P95 TTFT: {success['ttft'].quantile(0.95):.4f}s")
                print(f"P99 TTFT: {success['ttft'].quantile(0.99):.4f}s")
        except Exception as e:
            print(f"Analysis failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--scenario", type=str, default="random", 
                      choices=["random", "rr_worst_case", "long_heavy", "short_only", "bimodal"])
    parser.add_argument("--arrival", type=str, default="poisson",
                      choices=["poisson", "constant", "burst"])
    parser.add_argument("--url", type=str, default="http://localhost:6006/v1/chat/completions")
    parser.add_argument("--workload", type=str, default="data/processed_workload.jsonl")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.url, args.workload, args.rps, args.duration, args.scenario, args.arrival)
    asyncio.run(runner.run())
