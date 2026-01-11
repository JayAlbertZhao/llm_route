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
    
    # --- Advanced Feature Engineering ---
    print("Generating advanced features...")
    # 1. Interaction Features
    df["len_x_reqs"] = df["input_len"] * df["active_reqs"]
    df["len_sq"] = df["input_len"] ** 2

    # 2. Queueing Theory Features (New)
    # Models queueing delay which grows super-linearly with load
    df["reqs_sq"] = df["active_reqs"] ** 2
    df["len_x_reqs_sq"] = df["input_len"] * (df["active_reqs"] ** 2)
    
    # 3. Rolling/Lag Features (Simulating system state awareness)
    # We assume data is time-ordered roughly.
    # Rolling mean of LAST 5 completed requests' TTFT (Target Encoding Leakage? No, we use *past* data to predict *future*)
    # In practice, Router knows the TTFT of requests that just finished.
    df["last_5_avg_ttft"] = df["ttft"].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
    df["last_5_avg_tbt"] = df["tbt"].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
    
    # Drop rows with NaNs from shifting
    df = df.dropna()

    # Features to use
    features = ["input_len", "active_reqs", "len_x_reqs", "len_sq", "reqs_sq", "len_x_reqs_sq", "last_5_avg_ttft", "last_5_avg_tbt"]
    print(f"Features used: {features}")
    
    X = df[features]
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

    # 3. MLP (Deep Learning)
    print("\nTraining MLP (Deep Learning)...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "mlp_scaler.pkl"))

    # Deeper and wider network
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), 
        activation='relu',
        solver='adam',
        max_iter=1000, 
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42
    )
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

    # Additional: Check R2 on high load subset of test data (if possible)
    # We need to reconstruct the dataframe or pass it in to do this properly, 
    # but for now let's just inspect the overall improvement.

if __name__ == "__main__":
    train_models()

