import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Configuration
DATA_PATH = "data/profiling_data.csv"
MODEL_DIR = "models"
SEED = 42

def load_and_label_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Please run profile_system.py first.")
        
    df = pd.read_csv(filepath)
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    
    # --- Step 1: Reproduce Clustering (Ground Truth Generation) ---
    # Features used for clustering (System State)
    cluster_cols = ["active_reqs_client", "num_running", "input_len"]
    X_cluster = df[cluster_cols]
    
    # Normalize for KMeans (Distance-based)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K=4 based on analysis
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters
    
    # Analyze cluster mapping to ensure consistency (e.g., which cluster is High Load?)
    print("\nCluster Stats:")
    stats = df.groupby("cluster")["ttft"].agg(["mean", "count", "min", "max"]).sort_values("mean")
    print(stats)
    
    return df

def train_classifier(df):
    print("\n--- Training Classifier ---")
    
    # Features for the Classifier
    feature_cols = ["active_reqs_client", "num_running", "input_len"]
    X = df[feature_cols]
    y = df["cluster"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # 1. Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=SEED)
    dt.fit(X_train, y_train)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")
    
    # Save the best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "routing_classifier.pkl")
    joblib.dump(dt, model_path)
    print(f"Saved Decision Tree model to {model_path}")
    
    # Save cluster stats
    cluster_stats = df.groupby("cluster")["ttft"].mean().to_dict()
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1])
    
    # Heuristic mapping based on sorted mean TTFT
    medium_cluster_id = sorted_clusters[1][0]
    print(f"\nIdentified Medium Load Cluster ID: {medium_cluster_id} (Mean TTFT: {sorted_clusters[1][1]:.2f}s)")
    
    stats_path = os.path.join(MODEL_DIR, "cluster_stats.json")
    import json
    with open(stats_path, "w") as f:
        meta = {
            "stats": cluster_stats,
            "mapping": {
                "low": sorted_clusters[0][0],
                "medium": sorted_clusters[1][0],
                "long_context": sorted_clusters[2][0] if len(sorted_clusters) > 2 else -1,
                "high": sorted_clusters[-1][0]
            }
        }
        json.dump(meta, f, indent=2)
    print(f"Saved cluster stats to {stats_path}")
    
    return medium_cluster_id

def train_regressor(df, target_cluster_id):
    print(f"\n--- Training Regressor for Cluster {target_cluster_id} ---")
    
    # Filter data for just this cluster
    cluster_df = df[df["cluster"] == target_cluster_id]
    print(f"Training samples: {len(cluster_df)}")
    
    # Features for Regression
    reg_features = ["num_running", "input_len", "active_reqs_client"]
    X = cluster_df[reg_features]
    y = cluster_df["ttft"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # 1. Gradient Boosting
    print("Training Gradient Boosting Regressor...")
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED)
    gbr.fit(X_train, y_train)
    
    score = gbr.score(X_test, y_test)
    print(f"GBR R^2 Score: {score:.4f}")
    
    model_path = os.path.join(MODEL_DIR, "medium_load_regressor.pkl")
    joblib.dump(gbr, model_path)
    print(f"Saved Regressor model to {model_path}")

if __name__ == "__main__":
    try:
        df = load_and_label_data(DATA_PATH)
        medium_id = train_classifier(df)
        train_regressor(df, medium_id)
    except Exception as e:
        print(f"Error: {e}")
