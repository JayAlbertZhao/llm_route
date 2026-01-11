import os
import subprocess
import time
import sys
import requests
import signal

# Test Configuration
STRATEGY = "hierarchical"
SCENARIO = "random"
ARRIVAL = "poisson"
RPS = 5
DURATION = 10

ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 6006
ROUTER_URL = f"http://localhost:{ROUTER_PORT}/v1/chat/completions"

def check_router_health():
    url = f"http://localhost:{ROUTER_PORT}/health"
    try:
        resp = requests.get(url, timeout=2)
        print(f"[Health] Status: {resp.status_code}, Body: {resp.text}")
        return resp.status_code == 200
    except Exception as e:
        print(f"[Health] Check failed: {e}")
        return False

def main():
    print("=== STARTING SINGLE CASE TEST ===")
    
    # 1. Kill old router
    print("1. Cleaning up old processes...")
    os.system(f"fuser -k {ROUTER_PORT}/tcp")
    time.sleep(1)
    
    # 2. Start Router
    print(f"2. Starting Router (Strategy={STRATEGY})...")
    env = os.environ.copy()
    env["ROUTER_STRATEGY"] = STRATEGY
    env["PYTHONPATH"] = os.getcwd()
    
    # Use unbuffered output and pipe to stdout so we see it immediately
    cmd = [sys.executable, "-u", "src/router/gateway.py", "--host", ROUTER_HOST, "--port", str(ROUTER_PORT)]
    
    router_proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    
    # Wait for startup
    started = False
    for i in range(10):
        print(f"   Waiting for router... ({i+1}/10)")
        if check_router_health():
            started = True
            break
        time.sleep(1)
        
    if not started:
        print("!!! Router Failed to Start !!!")
        if router_proc.poll() is not None:
             print(f"Process exit code: {router_proc.returncode}")
        router_proc.terminate()
        return

    print(">>> Router Started Successfully. <<<")
    
    # 3. Run Experiment
    print(f"\n3. Running Experiment (RPS={RPS}, Dur={DURATION}s)...")
    exp_cmd = [
        sys.executable, "scripts/run_experiment.py",
        "--rps", str(RPS),
        "--duration", str(DURATION),
        "--scenario", SCENARIO,
        "--arrival", ARRIVAL,
        "--url", ROUTER_URL
    ]
    
    # Run synchronously and show output
    try:
        subprocess.run(exp_cmd, env=env, check=True)
        print("\n>>> Experiment Script Finished Successfully. <<<")
    except subprocess.CalledProcessError as e:
        print(f"!!! Experiment Script Failed with code {e.returncode} !!!")
    
    # 4. Cleanup
    print("\n4. Terminating Router...")
    router_proc.terminate()
    try:
        router_proc.wait(timeout=2)
    except:
        router_proc.kill()
        
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()

