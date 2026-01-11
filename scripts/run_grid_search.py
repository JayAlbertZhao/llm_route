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
SCENARIOS = ["random", "rr_worst_case", "long_heavy", "bimodal", "short_only"]
ARRIVALS = ["poisson", "burst"]

DURATION = 45 # Seconds per run
TIMEOUT_BUFFER = 90 # Extra time for draining pending requests

OUTPUT_FILE = "data/model_comparison.csv"
ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 6006
ROUTER_URL = f"http://localhost:{ROUTER_PORT}/v1/chat/completions"

def get_rps_levels(scenario):
    """Dynamic RPS Configuration based on Scenario Difficulty"""
    if scenario == "long_heavy":
        return [5, 10, 20] # Lower RPS for heavy tasks
    elif scenario == "short_only":
        return [10, 20, 40, 80] # Higher RPS for light tasks
    else:
        return [10, 20, 40, 80] # Default

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
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [sys.executable, "-u", "src/router/gateway.py", "--host", ROUTER_HOST, "--port", str(ROUTER_PORT)]
    
    log_file = open(f"logs/router_{strategy}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    
    for i in range(20):
        if check_router_health():
            print("[Manager] Router is UP.")
            return proc
        time.sleep(1)
        if proc.poll() is not None:
            print(f"[Manager] Router process died early with code {proc.returncode}")
            break
    
    print("[Manager] Router failed to start! Check logs/router_*.log")
    log_file.close()
    
    try:
        with open(f"logs/router_{strategy}.log", "r") as f:
            print("--- Router Log Tail ---")
            print("".join(f.readlines()[-10:]))
            print("-----------------------")
    except: pass

    if proc.poll() is None: proc.kill()
    return None

def run_experiment(strategy, scenario, arrival, rps):
    print(f"\n>>> Running: Strat={strategy}, Scen={scenario}, Arr={arrival}, RPS={rps}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--rps", str(rps),
        "--duration", str(DURATION),
        "--scenario", scenario,
        "--arrival", arrival,
        "--url", ROUTER_URL
    ]
    
    try:
        # Increase timeout to ensure we don't kill experiments that are just finishing up
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=DURATION + TIMEOUT_BUFFER, env=env)
        output = result.stdout
        print(output)
        
        metrics = {"strategy": strategy, "scenario": scenario, "arrival": arrival, "rps": rps}
        
        for line in output.splitlines():
            if "Mean TTFT:" in line: metrics["mean_ttft"] = float(line.split(":")[1].strip().replace("s", ""))
            if "P99 TTFT:" in line: metrics["p99_ttft"] = float(line.split(":")[1].strip().replace("s", ""))
            if "Total Sent:" in line: metrics["sent"] = int(line.split(":")[1].strip())
            if "Success:" in line: metrics["success"] = int(line.split(":")[1].strip())
            if "Errors:" in line: metrics["errors"] = int(line.split(":")[1].strip())
            
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"[Manager] Experiment Timed Out (Max {DURATION + TIMEOUT_BUFFER}s).")
        # In case of timeout, we still want to record that it failed or record partial data if possible
        # But run_experiment.py now handles timeout internally by cancelling tasks, so subprocess shouldn't timeout unless it hangs.
        return {"strategy": strategy, "scenario": scenario, "arrival": arrival, "rps": rps, "errors": -1} # Mark as timeout
    except Exception as e:
        print(f"[Manager] Experiment failed: {e}")
        return None

def load_existing_results():
    if os.path.exists(OUTPUT_FILE):
        try:
            return pd.read_csv(OUTPUT_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def is_done(df, strategy, scenario, arrival, rps):
    if df.empty: return False
    # Check if a row exists with these exact parameters
    mask = (df["strategy"] == strategy) & \
           (df["scenario"] == scenario) & \
           (df["arrival"] == arrival) & \
           (df["rps"] == rps)
    return not df[mask].empty

def main():
    # Ensure no old router is running
    os.system(f"fuser -k {ROUTER_PORT}/tcp > /dev/null 2>&1")
    
    # Load existing results for checkpointing
    existing_df = load_existing_results()
    print(f"[Manager] Loaded {len(existing_df)} existing records.")
    
    # Initialize results list with existing data to preserve it
    # Actually, we will just append new rows to file, but keep list for current session
    results = [] 
    
    total_combinations = 0
    # Calculate total runs
    for scen in SCENARIOS:
        total_combinations += len(STRATEGIES) * len(ARRIVALS) * len(get_rps_levels(scen))
        
    curr_run = 0
    
    for strategy in STRATEGIES:
        router_proc = start_router(strategy)
        if not router_proc:
            print("Skipping strategy due to startup failure.")
            continue
            
        for scenario in SCENARIOS:
            rps_list = get_rps_levels(scenario)
            
            for arrival in ARRIVALS:
                for rps in rps_list:
                    curr_run += 1
                    
                    # Checkpoint Check
                    if is_done(existing_df, strategy, scenario, arrival, rps):
                        print(f"--- Skipping {curr_run}/{total_combinations} (Already Done): {strategy}/{scenario}/{arrival}/{rps}")
                        continue

                    print(f"\n=== Progress: {curr_run}/{total_combinations} ===")
                    
                    metrics = run_experiment(strategy, scenario, arrival, rps)
                    if metrics:
                        # Append to CSV immediately
                        df_new = pd.DataFrame([metrics])
                        header = not os.path.exists(OUTPUT_FILE)
                        df_new.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
                        
                        # Update in-memory df for subsequent checks (though not strictly needed if we don't repeat)
                        existing_df = pd.concat([existing_df, df_new], ignore_index=True)
                        
                    time.sleep(2)
        
        print(f"[Manager] Killing Router for strategy {strategy}...")
        router_proc.terminate()
        try:
            router_proc.wait(timeout=5)
        except:
            router_proc.kill()
        
        os.system(f"fuser -k {ROUTER_PORT}/tcp > /dev/null 2>&1")
        time.sleep(2)

    print(f"\n[Manager] Grid Search Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
