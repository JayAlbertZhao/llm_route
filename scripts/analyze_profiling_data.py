import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_profiling():
    df = pd.read_csv("data/profiling_data.csv")
    
    plt.figure(figsize=(15, 5))
    
    # 1. Input Len vs TTFT
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x="input_len", y="ttft", hue="system_load", alpha=0.5)
    plt.title("Input Len vs TTFT")
    plt.xlabel("Input Tokens")
    plt.ylabel("TTFT (s)")
    
    # 2. System Load (RPS) vs TTFT
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x="system_load", y="ttft")
    plt.title("System Load (RPS) vs TTFT")
    
    # 3. Active Reqs vs TTFT
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x="active_reqs", y="ttft")
    plt.title("Active Reqs vs TTFT")
    
    plt.tight_layout()
    plt.savefig("data/profiling_analysis.png")
    print("Saved analysis plot to data/profiling_analysis.png")
    
    # Check correlation
    print("\n--- Correlation Matrix ---")
    print(df[["input_len", "system_load", "active_reqs", "ttft", "tbt"]].corr())

if __name__ == "__main__":
    analyze_profiling()

