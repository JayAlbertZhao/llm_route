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
    # Target URL for inference requests
    TARGET_URL = "http://localhost:8081/v1/chat/completions"
    # Metrics URL for scraping system state
    METRICS_URL = "http://localhost:8081/metrics"
    OUTPUT_FILE = "data/profiling_data.csv"
    MODEL_NAME = "qwen-8b"

    class MetricsCollector:
        def __init__(self, metrics_url, interval=0.1):
            self.url = metrics_url
            self.interval = interval
            self.running = False
            self.logs = [] # List of (timestamp, running, waiting)
            self._lock = asyncio.Lock()

        async def start(self):
            self.running = True
            while self.running:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.url) as response:
                            if response.status == 200:
                                text = await response.text()
                                self._parse_and_store(text)
                except Exception:
                    pass
                await asyncio.sleep(self.interval)

        def stop(self):
            self.running = False

        def _parse_and_store(self, text):
            # Simple parsing for speed
            # vllm:num_requests_running{...} 0.0
            num_running = 0
            num_waiting = 0
            
            for line in text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    try:
                        num_running = float(line.split()[-1])
                    except: pass
                elif line.startswith("vllm:num_requests_waiting"):
                    try:
                        num_waiting = float(line.split()[-1])
                    except: pass
            
            # Store with high precision timestamp
            self.logs.append({
                "ts": time.time(),
                "running": num_running,
                "waiting": num_waiting
            })

        def get_state_at(self, timestamp):
            # Find the log entry closest to the given timestamp (before or slightly after)
            # Since logs are sorted by time, we can use binary search or simple iteration if list is small
            # For simplicity in this script, we'll just search backwards
            if not self.logs:
                return 0, 0
                
            # Iterate backwards to find the state just before the request started
            for entry in reversed(self.logs):
                if entry["ts"] <= timestamp:
                    return entry["running"], entry["waiting"]
            
            # Fallback to earliest
            return self.logs[0]["running"], self.logs[0]["waiting"]

    class Profiler:
        def __init__(self, workload_path, target_url, metrics_url):
            self.loader = WorkloadLoader(workload_path)
            self.loader.load()
            self.all_data = self.loader.data
            self.results = []
            self.target_url = target_url
            self.metrics_collector = MetricsCollector(metrics_url)
            self.tokenizer = None # Load lazily

        async def send_request(self, session, prompt, current_rps, active_reqs_client):
            # ... existing send_request code ...
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            # Remove max_tokens limit or set very high to stress TPOT
            "max_tokens": 1024, 
            "stream": True
        }
        
        start_time = time.time()
        ttft = None
        completion_text = []
        token_count = 0 # Count chunks, not real tokens yet
        
        try:
            # Capture state *before* sending
            # Note: This is imperfect as network latency is involved, but better than client-side count
            # We use the collector's history to look up the state at exactly start_time
            pass 
            
            # Use self.target_url instead of global constant
            async with session.post(self.target_url, json=payload) as response:
                if response.status != 200:
                    print(f"Request failed: HTTP {response.status}")
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
                            
                            # Collect text for post-processing
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                completion_text.append(delta)
                            token_count += 1
                        except:
                            pass
                            
            duration = time.time() - start_time
            # Raw TBT estimate (will refine later)
            tbt = (duration - ttft) / (token_count - 1) if token_count > 1 and ttft else 0
            
            # Lookup real vLLM state at start_time
            real_running, real_waiting = self.metrics_collector.get_state_at(start_time)
            
            return {
                "prompt_text": prompt,             # Store raw text
                "completion_text": "".join(completion_text), # Store raw text
                "system_load": current_rps,
                "active_reqs_client": active_reqs_client,
                "num_running": real_running,      # New!
                "num_waiting": real_waiting,      # New!
                "ttft": ttft,
                "tbt": tbt,
                "total_time": duration
            }
        except Exception as e:
            # print(f"Error: {e}")
            return None

    def post_process_tokens(self):
        print("Post-processing: Calculating exact token counts...")
        from transformers import AutoTokenizer
        # Use a small fast tokenizer for calculation (e.g. gpt2 or qwen)
        # Ideally match the server model, but any standard tokenizer is fine for estimating load
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2") 
        except:
            print("Warning: Could not load gpt2 tokenizer. Using simple length heuristic.")
            self.tokenizer = None

        processed_results = []
        for res in self.results:
            if self.tokenizer:
                input_len = len(self.tokenizer.encode(res["prompt_text"]))
                output_len = len(self.tokenizer.encode(res["completion_text"]))
            else:
                input_len = len(res["prompt_text"]) // 4
                output_len = len(res["completion_text"]) // 4
            
            # Refine TBT with exact output length
            if output_len > 1 and res["ttft"]:
                res["tbt"] = (res["total_time"] - res["ttft"]) / (output_len - 1)
            
            res["input_len"] = input_len
            res["output_len"] = output_len
            
            # Remove heavy text fields before saving
            del res["prompt_text"]
            del res["completion_text"]
            processed_results.append(res)
            
        self.results = processed_results

    async def run_phase(self, rps, duration_sec=30):
        print(f"Profiling RPS={rps} for {duration_sec}s...")
        start_time = time.time()
        tasks = []
        active_reqs = 0
        
        # Disable SSL verification for testing
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
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
        # Suppress SSL errors
        def handle_exception(loop, context):
            msg = context.get("exception", context.get("message"))
            if "SSL" in str(msg) or "ClientConnectionError" in str(msg) or "application data after close notify" in str(msg):
                return
            loop.default_exception_handler(context)

        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handle_exception)

        # Start metrics collector
        collector_task = asyncio.create_task(self.metrics_collector.start())

        # Sweep RPS from low to high
        # Start from meaningful load, push to extreme
        rps_levels = [16, 32, 64, 128]
        
        for rps in rps_levels:
            await self.run_phase(rps)
            # Cooldown
            print("Cooldown 5s...")
            await asyncio.sleep(5)
            
        # Stop collector
        self.metrics_collector.stop()
        try:
            await collector_task
        except: pass
            
        # Post-process
        self.post_process_tokens()
        
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
    parser.add_argument("--metrics_url", type=str, default=default_url.replace("/v1/chat/completions", "/metrics"), help="Metrics URL")
    parser.add_argument("--workload", type=str, default="data/processed_workload.jsonl")
    args = parser.parse_args()

    profiler = Profiler(args.workload, args.url, args.metrics_url)
    asyncio.run(profiler.run())

