# Detailed Implementation Plan: vLLM Routing Experiment

## 1. Project Structure & Environment
**Environment**: `conda activate route`
**Project Layout**:
```
vllm_routing/
├── data/                   # Dataset storage and processed buckets
├── logs/                   # Experiment logs and metrics
├── models/                 # Trained routing models (Linear, DT, MLP)
├── src/
│   ├── client/             # Traffic generation and request sending
│   │   ├── generator.py    # Poisson, Burst, Constant patterns
│   │   └── workload.py     # Data loader and tokenizer
│   ├── router/             # Routing logic
│   │   ├── strategies/     # RR, LeastLoad, PredictionModels
│   │   ├── predictor.py    # Model inference wrappers
│   │   └── gateway.py      # Main router application (FastAPI/Async)
│   ├── backend/            # vLLM interaction (if needed wrappers)
│   │   └── vllm_client.py  # Async wrapper for vLLM API
│   ├── training/           # Offline model training scripts
│   └── utils/              # Logging, metrics calculation, config
├── scripts/                # Execution scripts
│   ├── profile_system.py   # Phase 2: Profiling
│   ├── train_models.py     # Phase 3: Training
│   └── run_experiment.py   # Phase 5: Grid Search Runner
├── config/                 # Configuration files (YAML/JSON)
└── requirements.txt
```

## 2. Phase-wise Implementation Strategy

### Phase 1: Infrastructure & Base Components
*Goal: Establish communication between Client, Router, and vLLM Backend.*
1.  **Backend Setup**:
    *   Assume vLLM is running on specific ports (e.g., `http://localhost:8000`, `8001`).
    *   Create `VLLMClient` class: Async HTTP client to send `v1/completions` or `v1/chat/completions` requests.
2.  **Traffic Generator**:
    *   Implement `WorkloadGenerator` class:
        *   Input: Distribution type (Poisson, Burst, Constant), RPS (Requests Per Second).
        *   Output: Async stream of request events.
3.  **Basic Router**:
    *   Implement `Router` class with `RoundRobin` strategy initially.
    *   Metric logging: Request ID, Arrival Time, Backend Send Time, Response Time, TTFT (from vLLM stream), Token Count.

### Phase 2: Data Preparation & Profiling
*Goal: Prepare simulation data and collect training data for predictors.*
1.  **Data Processing**:
    *   Script to download `allenai/WildChat-1M`.
    *   Filter/Clean data.
    *   **Bucketing**: Group prompts by token length (e.g., <128, 128-512, 512-1024, >1024).
    *   Save as `processed_workload.jsonl`.
2.  **Profiling Run**:
    *   Script `profile_system.py`:
        *   Target single vLLM instance.
        *   Sweep through different RPS (Low to Overload) and Prompt Lengths.
        *   Record inputs (length, complexity features) and outputs (TTFT, TBT, Total Generation Time).
    *   **Output**: `profiling_data.csv` (Features: `input_len`, `system_load`; Labels: `ttft`, `tbt`).

### Phase 3: Offline Training (Model Development)
*Goal: Train predictors for the routing strategies.*
1.  **Feature Engineering**:
    *   Inputs: Current System Load (Queue depth / Active requests), Input Token Length.
    *   Targets: TTFT, TBT.
2.  **Model Implementation** (using `scikit-learn` / `torch`):
    *   `LinearRegressor`: Simple baseline.
    *   `DecisionTree`: CART implementation.
    *   `MLP`: Small neural net for non-linear relationships.
    *   `Distill`: Train a small transformer-based model and distill it into decisiontree.
3.  **Training Script**:
    *   `train_models.py`: Loads `profiling_data.csv`, trains models, saves weights to `models/`.

### Phase 4: Advanced Router & Strategies
*Goal: Implement the "Smart" routing logic.*
1.  **Strategy Interface**:
    *   `RouteStrategy` abstract class: `select_backend(request_features, system_state) -> backend_id`.
2.  **Implement Strategies**:
    *   `RoundRobin`: Cyclic selection.
    *   `LeastLoad`: Select backend with fewest active requests.
    *   `PredictiveRouter`:
        *   Load trained models (Linear, DT, MLP).
        *   Predict cost (e.g., predicted TTFT) for each backend.
        *   Select minimum cost.
    *   `GlobalDynamicLinear`:
        *   Online learning / Weight update mechanism based on feedback (Actual vs Predicted).
3.  **System State Manager**:
    *   Track active requests per backend in real-time.

### Phase 5: Experiment Execution & Grid Search
*Goal: Run the full matrix of experiments.*
1.  **Configuration Manager**:
    *   Define experiment matrix in `config/experiments.yaml`.
    *   Variables: Strategy, Traffic Pattern, RPS, Backend Count.
2.  **Automated Runner** (`run_experiment.py`):
    *   Loop through configurations.
    *   Spin up Router with specific strategy.
    *   Start Traffic Generator.
    *   Wait for completion.
    *   Save logs with unique Experiment ID.

### Phase 6: Analysis & Visualization
1.  **Metrics Parser**: Calculate Aggregated TTFT, Goodput (req/s under SLO), TBT.
2.  **Plotting**: Compare strategies under different loads.

## 3. Communication & Data Flow
*   **Protocol**: HTTP/REST (FastAPI for Router, aiohttp for Client).
*   **Payload**:
    *   Client -> Router: `{"prompt": "...", "model": "...", "timestamp": ...}`
    *   Router -> Backend: Forward standard OpenAI API format.
    *   Backend -> Router -> Client: Streamed response (SSE) to capture TTFT.

## 4. Immediate Next Steps (Day 1)
1.  **Skeleton Code**: Set up directory structure.
2.  **Data Loader**: Download and inspect WildChat dataset.
3.  **Dummy Backend**: Create a mock vLLM server if GPUs are not immediately available for dev, to test the loop.

