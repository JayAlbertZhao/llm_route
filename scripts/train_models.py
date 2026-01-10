import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
DATA_PATH = "data/profiling_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Data file not found!")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Feature Engineering
    # X: [input_len, active_reqs]
    # y: [ttft]
    X = df[["input_len", "active_reqs"]]
    y = df["ttft"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    results = {}

    # 1. Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, "LinearRegression", results)
    joblib.dump(lr, os.path.join(MODEL_DIR, "linear_predictor.pkl"))

    # 2. Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    evaluate_model(dt, X_test, y_test, "DecisionTree", results)
    joblib.dump(dt, os.path.join(MODEL_DIR, "dt_predictor.pkl"))

    # 3. MLP
    print("\nTraining MLP (Neural Network)...")
    # Scaling is important for MLP
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Save scaler for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "mlp_scaler.pkl"))

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    
    # Eval MLP separately because of scaling
    y_pred = mlp.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results["MLP"] = {"MAE": mae, "MSE": mse, "R2": r2}
    print(f"MLP - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    
    joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_predictor.pkl"))

    # Summary
    print("\n--- Model Comparison ---")
    res_df = pd.DataFrame(results).T
    print(res_df)
    res_df.to_csv("data/model_comparison.csv")
    
    # Plot Logic (Optional)
    # visualize_predictions(dt, X_test, y_test)

def evaluate_model(model, X_test, y_test, name, results):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
    print(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train_models()

