# Essential Files Guide for vLLM Routing Project

This document lists the critical files needed to understand the project architecture, experiments, and current findings. Use this as a map when starting a new session.

## 1. Documentation & Reports (Must Read)
*   **`Experiment_Report.md`**: The living document of our findings. Contains the **Phase 2 Analysis** (K=4 Clustering), **Hierarchical Routing Strategy** design, and discovery logs.
*   **`Detailed_Implementation_Plan.md`**: The original master plan outlining the 6 phases of the project.
*   **`Implement_plan.txt`**: Incremental logs and TODOs.

## 2. Core Code (Source)
*   **`src/router/gateway.py`**: The FastAPI router entry point. Currently implements a basic RoundRobin loop, but destined to host the new Hierarchical Strategy.
*   **`src/router/strategies/`**: Where routing logic lives.
    *   `base.py`: Abstract base class.
    *   `round_robin.py`: Baseline implementation.
*   **`src/backend/vllm_client.py`**: Async client for vLLM interaction. Handled TTFT measurement and stream parsing.
*   **`src/client/workload.py`**: Data loader for WildChat dataset. Handles tokenization and bucketing.

## 3. Analysis & Scripts (The "Brain")
*   **`scripts/profile_system.py`**: **Critical**. The data collector.
    *   Features: Concurrent async load generation, **vLLM Metrics Scraping** (grey-box), Fine-grained RPS sweeping.
    *   Usage: `python scripts/profile_system.py` (Run on GPU server).
*   **`scripts/analyze_clusters.py`**: The "unsupervised learning" script.
    *   Performs K-Means (K=4), PCA, t-SNE, and Feature Importance analysis.
    *   Run this to reproduce the "4 Clusters" finding.
*   **`scripts/train_models.py`**: (To be updated) Script to train the Router Models. Currently trains simple regressors, needs update for Classifier + Regressor architecture.

## 4. Key Data & Visualizations (Evidence)
*   **`data/profiling_data.csv`**: The massive dataset (13k+ rows) covering Low -> Med -> High load. (Note: ignored in git, pull from server or re-run profiler).
*   **`data/cluster_tsne.png`**: **The "Map"**. Shows the continuous manifold of system states and the 4 clusters.
*   **`data/cluster_boxplots.png`**: Shows the physical meaning of clusters (TTFT/ActiveReqs distributions).
*   **`data/optimal_k.png`**: Justification for choosing K=4.

## 5. Environment
*   **`scripts/run_vllm_node.sh`**: Startup script for the backend vLLM instance.
*   **`requirements.txt`**: Python dependencies.

---
**Quick Start for New Session:**
1.  Read `Experiment_Report.md` to understand the "Cluster-Aware Hierarchical Routing" strategy.
2.  Check `scripts/analyze_clusters.py` output to verify data assumptions.
3.  Next Task: Implement the Hierarchical Strategy in `gateway.py` and train models in `train_models.py`.

