import aiohttp
import asyncio
import argparse

async def fetch_metrics(url):
    print(f"Fetching metrics from {url}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    print("\n--- Metrics Response Sample ---")
                    # Print first 20 lines to avoid spam
                    print("\n".join(text.split("\n")[:20]))
                    print("...\n")
                    
                    # Parse looking for specific keys we discussed
                    print("--- Key Metrics Check ---")
                    keys_to_check = [
                        "vllm:num_requests_running",
                        "vllm:num_requests_waiting", 
                        "vllm:num_requests_swapped",
                        "vllm:gpu_cache_usage_perc"
                    ]
                    
                    for line in text.split("\n"):
                        for key in keys_to_check:
                            if key in line and "#" not in line: # Skip comments
                                print(line)
                else:
                    print(f"Failed: HTTP {response.status}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8081/metrics")
    args = parser.parse_args()
    
    asyncio.run(fetch_metrics(args.url))

