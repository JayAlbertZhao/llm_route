import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configuration
# Try to find the latest comparison file
POSSIBLE_FILES = [
    "data/model_comparison (1).csv",
    "data/model_comparison.csv"
]
DATA_FILE = None
for f in POSSIBLE_FILES:
    if os.path.exists(f):
        DATA_FILE = f
        break

OUTPUT_DIR = "data/analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_data():
    if not DATA_FILE:
        print("No data file found!")
        return None
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    return df

def plot_rps_vs_metric(df, metric="mean_ttft", ylabel="Mean TTFT (s)"):
    """
    Generate line plots for each scenario: Metric vs RPS
    Comparing RR vs Hierarchical
    """
    scenarios = df["scenario"].unique()
    arrivals = df["arrival"].unique()
    
    for arrival in arrivals:
        # Create a figure with subplots for scenarios
        # We handle 5 scenarios, so maybe 2x3 grid
        n_scenarios = len(scenarios)
        cols = 3
        rows = (n_scenarios + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
        axes = axes.flatten()
        
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            subset = df[(df["scenario"] == scenario) & (df["arrival"] == arrival)]
            
            if subset.empty:
                continue
                
            sns.lineplot(
                data=subset, x="rps", y=metric,
                hue="strategy", style="strategy",
                markers=True, dashes=False, linewidth=2.5, markersize=9,
                ax=ax, palette=["#e74c3c", "#2ecc71"] # Red for RR, Green for Hierarchical
            )
            
            ax.set_title(f"Scenario: {scenario.upper()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("RPS")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Improve legend
            if i == 0:
                ax.legend(title="Strategy")
            else:
                ax.get_legend().remove()
                
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle(f"{ylabel} vs Load (Arrival: {arrival.upper()})", fontsize=16, y=1.02)
        plt.tight_layout()
        filename = os.path.join(OUTPUT_DIR, f"{metric}_vs_rps_{arrival}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved {filename}")

def plot_improvement_bar(df):
    """
    Bar chart showing % improvement of Hierarchical over RR at Low/Medium Load (10, 20 RPS)
    """
    # Filter for interesting loads
    subset = df[df["rps"].isin([10, 20])].copy()
    
    # Pivot to calculate improvement
    pivot = subset.pivot_table(
        index=["scenario", "arrival", "rps"],
        columns="strategy",
        values="mean_ttft"
    ).reset_index()
    
    if "rr" not in pivot.columns or "hierarchical" not in pivot.columns:
        return

    # Calculate Improvement %
    pivot["improvement"] = (pivot["rr"] - pivot["hierarchical"]) / pivot["rr"] * 100
    
    # Plot
    g = sns.catplot(
        data=pivot, kind="bar",
        x="scenario", y="improvement", hue="rps", col="arrival",
        palette="viridis", height=6, aspect=1.2
    )
    
    g.despine(left=True)
    g.set_axis_labels("", "Latency Reduction (%)")
    g.fig.suptitle("Performance Gain at Low/Medium Load (Higher is Better)", y=1.05, fontsize=16)
    
    # Add value labels
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1f}%", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)

    filename = os.path.join(OUTPUT_DIR, "improvement_bar.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved {filename}")

def plot_tail_latency(df):
    """
    Compare P99 Latency for Worst Case Scenario
    """
    scenario = "rr_worst_case"
    subset = df[df["scenario"] == scenario]
    
    if subset.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=subset, x="rps", y="p99_ttft", hue="strategy",
        palette=["#e74c3c", "#2ecc71"]
    )
    
    plt.title(f"Tail Latency (P99) in '{scenario}' Scenario", fontsize=16)
    plt.ylabel("P99 TTFT (s)")
    plt.xlabel("RPS")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    filename = os.path.join(OUTPUT_DIR, "tail_latency_worst_case.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved {filename}")

def main():
    df = load_data()
    if df is None: return

    # 1. Mean TTFT Analysis
    plot_rps_vs_metric(df, "mean_ttft", "Mean TTFT (s)")
    
    # 2. Tail Latency Analysis
    plot_rps_vs_metric(df, "p99_ttft", "P99 TTFT (s)")
    
    # 3. Specific Improvement Analysis
    plot_improvement_bar(df)
    
    # 4. Tail Latency Deep Dive
    plot_tail_latency(df)

if __name__ == "__main__":
    main()

