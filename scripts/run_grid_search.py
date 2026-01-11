import os
import subprocess
import time
import pandas as pd
import requests
import signal
import sys
import glob

# Grid Configuration
STRATEGIES = ["rr", "hierarchical"]
SCENARIOS = ["random", "rr_worst_case", "long_heavy", "bimodal"]
ARRIVALS = ["poisson", "burst"]
RPS_LEVELS = [10, 20, 40, 80]
DURATION = 45 # Seconds per run

OUTPUT_FILE = "data/model_comparison.csv"
ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 6006
ROUTER_URL = f"http://localhost:{ROUTER_PORT}/v1/chat/completions"

def check_router_health():
    url = f"http://localhost:{ROUTER_PORT}/health"
    try:
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except:
        return False

def start_router(strategy):
    print(f"\n[Manager] Starting Router with Strategy={strategy}...")
    env = os.environ.copy()
    env["ROUTER_STRATEGY"] = strategy
    
    # Start process
    # Use unbuffered stdout/stderr to capture output immediately
    # Fix: Set PYTHONPATH to current directory so 'src' can be imported
    env["PYTHONPATH"] = os.getcwd()
    cmd = [sys.executable, "-u", "src/router/gateway.py", "--host", ROUTER_HOST, "--port", str(ROUTER_PORT)]
    
    # Pipe stderr to stdout so we can capture everything
    # But for debugging, let's redirect to a log file instead of /dev/null
    log_file = open(f"logs/router_{strategy}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Wait for health check
    for i in range(20):
        if check_router_health():
            print("[Manager] Router is UP.")
            return proc
        time.sleep(1)
        # Check if process died
        if proc.poll() is not None:
            print(f"[Manager] Router process died early with code {proc.returncode}")
            break
    
    print("[Manager] Router failed to start! Check logs/router_*.log")
    # Clean up file handle
    log_file.close()
    
    # Print the last few lines of the log for immediate feedback
    try:
        with open(f"logs/router_{strategy}.log", "r") as f:
            print("--- Router Log Tail ---")
            print("".join(f.readlines()[-10:]))
            print("-----------------------")
    except:
        pass

    if proc.poll() is None:
        proc.kill()
    return None

def run_experiment(strategy, scenario, arrival, rps):
    print(f"\n>>> Running: Strat={strategy}, Scen={scenario}, Arr={arrival}, RPS={rps}")
    
    # Run the experiment script (it saves detailed CSVs to logs/experiments/)
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--rps", str(rps),
        "--duration", str(DURATION),
        "--scenario", scenario,
        "--arrival", arrival,
        "--url", ROUTER_URL
    ]
    
    try:
        # Capture output to find the saved filename
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DURATION + 30)
        output = result.stdout
        print(output)
        
        # Parse metrics from stdout for quick summary
        metrics = {"strategy": strategy, "scenario": scenario, "arrival": arrival, "rps": rps}
        
        # Simple parsing of the stdout block
        for line in output.splitlines():
            if "Mean TTFT:" in line: metrics["mean_ttft"] = float(line.split(":")[1].strip().replace("s", ""))
            if "P99 TTFT:" in line: metrics["p99_ttft"] = float(line.split(":")[1].strip().replace("s", ""))
            if "Total Sent:" in line: metrics["sent"] = int(line.split(":")[1].strip())
            if "Success:" in line: metrics["success"] = int(line.split(":")[1].strip())
            if "Errors:" in line: metrics["errors"] = int(line.split(":")[1].strip())
            
        return metrics
        
    except Exception as e:
        print(f"[Manager] Experiment failed: {e}")
        return None

def main():
    results = []
    
    # Ensure no old router is running
    os.system(f"fuser -k {ROUTER_PORT}/tcp > /dev/null 2>&1")
    
    total_runs = len(STRATEGIES) * len(SCENARIOS) * len(ARRIVALS) * len(RPS_LEVELS)
    curr_run = 0
    
    for strategy in STRATEGIES:
        # Start router once per strategy to save startup time? 
        # Ideally yes, but state accumulation might affect results. 
        # Safest is restart. But for speed, let's restart per strategy or per scenario.
        # Let's restart per strategy to clear state.
        
        router_proc = start_router(strategy)
        if not router_proc:
            print("Skipping strategy due to startup failure.")
            continue
            
        for scenario in SCENARIOS:
            for arrival in ARRIVALS:
                for rps in RPS_LEVELS:
                    curr_run += 1
                    print(f"\n=== Progress: {curr_run}/{total_runs} ===")
                    
                    metrics = run_experiment(strategy, scenario, arrival, rps)
                    if metrics:
                        results.append(metrics)
                        
                    # Save intermediate results
                    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                    
                    # Short cooldown
                    time.sleep(2)
        
        print(f"[Manager] Killing Router for strategy {strategy}...")
        router_proc.terminate()
        try:
            router_proc.wait(timeout=5)
        except:
            router_proc.kill()
        
        # Force kill port just in case
        os.system(f"fuser -k {ROUTER_PORT}/tcp > /dev/null 2>&1")
        time.sleep(2)

    print(f"\n[Manager] Grid Search Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

