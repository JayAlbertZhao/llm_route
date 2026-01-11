import logging
import joblib
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .base import RoutingStrategy

logger = logging.getLogger("Router")

class HierarchicalStrategy(RoutingStrategy):
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.classifier = None
        self.regressor = None
        self.cluster_meta = {}
        self.load_models()

    def load_models(self):
        try:
            # Load Classifier
            clf_path = os.path.join(self.models_dir, "routing_classifier.pkl")
            if os.path.exists(clf_path):
                self.classifier = joblib.load(clf_path)
                logger.info(f"Loaded Classifier from {clf_path}")
            else:
                logger.warning(f"Classifier not found at {clf_path}")

            # Load Regressor
            reg_path = os.path.join(self.models_dir, "medium_load_regressor.pkl")
            if os.path.exists(reg_path):
                self.regressor = joblib.load(reg_path)
                logger.info(f"Loaded Regressor from {reg_path}")
            else:
                logger.warning(f"Regressor not found at {reg_path}")

            # Load Cluster Stats/Meta
            meta_path = os.path.join(self.models_dir, "cluster_stats.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.cluster_meta = json.load(f)
                logger.info(f"Loaded Cluster Meta: {self.cluster_meta.get('mapping')}")
            else:
                logger.warning("Cluster stats not found, using default heuristics")
                # Fallback default
                self.cluster_meta = {
                    "mapping": {"low": 1, "medium": 2, "high": 0, "long_context": 3}
                }

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    async def select_backend(self, request: Dict[str, Any], backends: List[str], system_state: Dict[str, Any]) -> str:
        """
        Hierarchical Routing Logic:
        1. Extract Features (Client Reqs, Server Running, Input Len)
        2. Predict Cluster (Low, Med, High) for EACH backend candidates
        3. Apply Logic based on Cluster
        """
        if not backends:
            raise ValueError("No backends available")

        # 0. Feature Extraction
        # We need to predict the state for EACH backend if we were to route there.
        # But actually, the state is mostly independent of the current request (except +1).
        # We evaluate the CURRENT state of each backend.
        
        # Get input length (approximate)
        prompt = request.get("messages", [{}])[-1].get("content", "")
        # Simple char heuristic if no tokenizer (4 chars ~ 1 token)
        input_len = len(prompt) // 4 
        
        candidates = []
        
        for backend in backends:
            state = system_state.get("backends", {}).get(backend, {})
            
            # Features: [active_reqs_client, num_running, input_len]
            # Note: active_reqs is from Client view, num_running is from Server view (grey-box)
            active_reqs = state.get("active_requests", 0)
            num_running = state.get("num_running", 0)
            
            # Create feature vector (DataFrame for sklearn to match training columns)
            features = pd.DataFrame([{
                "active_reqs_client": active_reqs,
                "num_running": num_running,
                "input_len": input_len
            }])
            
            candidates.append({
                "url": backend,
                "features": features,
                "active_reqs": active_reqs,
                "num_running": num_running
            })

        # --- Layer 1: Classification ---
        # We want to filter out 'High Load' nodes first
        valid_candidates = []
        
        mapping = self.cluster_meta.get("mapping", {})
        high_id = mapping.get("high", 0)
        low_id = mapping.get("low", 1)
        medium_id = mapping.get("medium", 2)
        
        if self.classifier:
            for cand in candidates:
                try:
                    cluster_id = self.classifier.predict(cand["features"])[0]
                    cand["cluster"] = cluster_id
                    
                    # Circuit Breaker: Skip High Load nodes
                    if cluster_id == high_id:
                        logger.debug(f"Backend {cand['url']} is High Load (Cluster {cluster_id}). Skipping.")
                        continue
                        
                    valid_candidates.append(cand)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    valid_candidates.append(cand) # Fallback: keep it
        else:
            valid_candidates = candidates # No model, use all
            
        if not valid_candidates:
            # All nodes are overloaded? Fallback to Least Load among all
            logger.warning("All backends are overloaded! Falling back to Least Load.")
            return min(candidates, key=lambda x: x["active_reqs"])["url"]

        # --- Layer 2: Strategy Selection ---
        
        # Check if we have any "Low Load" candidates
        low_load_candidates = [c for c in valid_candidates if c.get("cluster") == low_id]
        
        if low_load_candidates:
            # Strategy: Least Load (Fastest, no inference needed)
            # In Low Load zone, simple counting is sufficient and robust
            best = min(low_load_candidates, key=lambda x: x["active_reqs"])
            logger.debug(f"Routing to Low Load node: {best['url']}")
            return best["url"]
            
        # Check for Medium Load candidates
        medium_load_candidates = [c for c in valid_candidates if c.get("cluster") == medium_id]
        
        if medium_load_candidates and self.regressor:
            # Strategy: Predictive (Regressor)
            # Use model to predict TTFT
            best_score = float('inf')
            best_url = None
            
            for cand in medium_load_candidates:
                try:
                    pred_ttft = self.regressor.predict(cand["features"])[0]
                    cand["pred_ttft"] = pred_ttft
                    if pred_ttft < best_score:
                        best_score = pred_ttft
                        best_url = cand["url"]
                except Exception as e:
                    logger.error(f"Regression error: {e}")
            
            if best_url:
                logger.debug(f"Routing to Medium Load node (Predicted TTFT {best_score:.2f}s): {best_url}")
                return best_url
        
        # Fallback (e.g. if only Long Context nodes are left, or Regressor failed)
        # Just pick the one with least active reqs
        best = min(valid_candidates, key=lambda x: x["active_reqs"])
        logger.debug(f"Fallback routing to: {best['url']}")
        return best["url"]

