import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

CLEAN_DATA_DIR = PROJECT_ROOT / "clean_data"
RAW_DIR = PROJECT_ROOT / "raw_result" / "advanced_workflow"
VIS_DIR = PROJECT_ROOT / "visualization_data"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# List of baselines to compare
# METHODS = [
#     "flat_prompt", 
#     "static_first", 
#     "fixed_semantic", 
#     "pla_token_hash"
# ]

METHODS = [
    "pla_token_hash",
    "pla_character"
]

# Better display names for plots
# METHOD_LABELS = {
#     "flat_prompt": "Flat Prompt",
#     "static_first": "Static-First",
#     "fixed_semantic": "Fixed Semantic",
#     "pla_character": "PLA (Character)"
# }

METHOD_LABELS = {
    "pla_token_hash": "PLA (Token Hash)",
    "pla_character": "PLA (Character)"
}

# Set global Seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==============================================================================
# 1. LOAD & AGGREGATE TASK METRICS (Averaging across 3 turns)
# ==============================================================================
def load_and_aggregate_task_metrics() -> pd.DataFrame:
    """
    Loads clean JSONL data, extracts metrics, and averages them across 3 turns 
    for each specific task_id to remove random system variance.
    """
    all_records = []
    
    for method in METHODS:
        norm_path = CLEAN_DATA_DIR / method / f"{method}_cleaned_results.jsonl"
        if not norm_path.exists():
            print(f"[Warning] Cleaned data not found for {method}: {norm_path}")
            continue
            
        with open(norm_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                total_prompt = 0
                total_cached = 0
                total_ttft = 0.0
                
                for agent in data.get("agent_traces", []):
                    metrics = agent.get("metrics", {})
                    p_tokens = metrics.get("prompt_tokens", 0)
                    c_tokens = metrics.get("cached_tokens", 0)
                    
                    total_prompt += p_tokens
                    total_cached += c_tokens
                    total_ttft += metrics.get("ttft", 0.0)
                
                overall_cache_ratio = (total_cached / total_prompt * 100) if total_prompt > 0 else 0.0
                uncached_prefill_tokens = total_prompt - total_cached
                
                all_records.append({
                    "method": METHOD_LABELS.get(method, method),
                    "task_id": data.get("task_id"),
                    "workflow_latency": data.get("workflow_latency", 0.0),
                    "total_ttft": total_ttft,
                    "overall_cache_ratio": overall_cache_ratio,
                    "uncached_prefill_tokens": uncached_prefill_tokens
                })
                
    raw_df = pd.DataFrame(all_records)
    
    if raw_df.empty:
        return raw_df
        
    # Average across turns for each task_id & method
    agg_df = raw_df.groupby(['method', 'task_id']).mean().reset_index()
    return agg_df

# ==============================================================================
# 2. PLOT: MEAN COMPARISON (BAR CHARTS)
# ==============================================================================
def plot_mean_comparisons(df: pd.DataFrame):
    """Plots bar charts for average latency, TTFT, Cache Ratio, and Uncached Prefill."""
    metrics = [
        ("workflow_latency", "End-to-End Workflow Latency (s)", "Lower is better"),
        ("total_ttft", "Total Time-To-First-Token (s)", "Lower is better"),
        ("overall_cache_ratio", "Overall Cache Hit Ratio (%)", "Higher is better"),
        ("uncached_prefill_tokens", "Uncached Prefill Tokens", "Lower is better")
    ]
    
    # Updated to 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Performance Comparison (Averaged across 3 turns)", fontsize=16, fontweight='bold', y=1.05)
    
    palette = sns.color_palette("Set2", len(METHODS))
    
    for i, (col, title, subtitle) in enumerate(metrics):
        # Calculate means
        means = df.groupby('method')[col].mean().sort_values(ascending=False if "Ratio" in title else True)
        order = means.index
        
        ax = axes[i]
        # Removed errorbar=('ci', 95) to hide the error bars (whiskers)
        sns.barplot(data=df, x="method", y=col, ax=ax, order=order, palette=palette, errorbar=None)
        
        ax.set_title(f"{title}\n({subtitle})", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=15)
        
        # Add value labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_task_metrics_mean_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 3. PLOT: VARIANCE & DISTRIBUTION (BOXPLOTS)
# ==============================================================================
def plot_variance_distributions(df: pd.DataFrame):
    """Plots boxplots to show the stability and variance across 50 tasks."""
    metrics = [
        ("workflow_latency", "Workflow Latency Distribution"),
        ("total_ttft", "Total TTFT Distribution"),
        ("overall_cache_ratio", "Cache Hit Ratio Distribution"),
        ("uncached_prefill_tokens", "Uncached Tokens Distribution")
    ]
    
    # Updated to 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Stability & Variance Across 50 Tasks", fontsize=16, fontweight='bold', y=1.05)
    
    palette = sns.color_palette("pastel", len(METHODS))
    
    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        # Draw Boxplot
        sns.boxplot(data=df, x="method", y=col, ax=ax, palette=palette, width=0.5, showfliers=False)
        # Overlay Strip plot for data points
        sns.stripplot(data=df, x="method", y=col, ax=ax, color="black", alpha=0.4, jitter=True, size=3)
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=15)
        
    plt.tight_layout()
    plt.savefig(VIS_DIR / "02_task_metrics_variance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 4. LOAD & AGGREGATE SYSTEM METRICS (Time-Aligned Averaging)
# ==============================================================================
def process_and_plot_system_metrics():
    """
    Reads the 3 CSV files (turn 1, 2, 3) for each method.
    Averages the values at each relative_time_s to create a 'Representative Turn'.
    Plots the smoothed time-series data.
    """
    representative_dfs = []
    
    for method in METHODS:
        method_dir = RAW_DIR / method
        turn_dfs = []
        
        # Load all available turns for this method
        for turn_idx in range(1, 4):
            csv_path = method_dir / f"system_metrics_turn_{turn_idx}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['relative_time_s'] = df['relative_time_s'].astype(int)
                # Group by exact second in case of multiple readings per second
                df = df.groupby('relative_time_s').mean().reset_index()
                turn_dfs.append(df)
                
        if not turn_dfs:
            print(f"[Warning] No system metrics CSVs found for {method}")
            continue
            
        # Time-Aligned Averaging
        merged_turns = pd.concat(turn_dfs)
        avg_turn_df = merged_turns.groupby('relative_time_s').mean().reset_index()
        
        # Calculate Occupancy %
        total_slots = avg_turn_df['kv_used'] + avg_turn_df['kv_evictable'] + avg_turn_df['kv_available']
        avg_turn_df['kv_occupancy_pct'] = np.where(total_slots > 0, ((avg_turn_df['kv_used'] + avg_turn_df['kv_evictable']) / total_slots) * 100, 0.0)
        
        avg_turn_df['method'] = METHOD_LABELS.get(method, method)
        representative_dfs.append(avg_turn_df)
        
    if not representative_dfs:
        print("[Warning] Could not process system metrics. Skipping plots.")
        return
        
    final_sys_df = pd.concat(representative_dfs, ignore_index=True)
    
    # Smooth data for plotting (Exponential Moving Average)
    for method in final_sys_df['method'].unique():
        mask = final_sys_df['method'] == method
        final_sys_df.loc[mask, 'throughput_smoothed'] = final_sys_df.loc[mask, 'gen_throughput'].ewm(span=5, adjust=False).mean()
        final_sys_df.loc[mask, 'hit_rate_smoothed'] = final_sys_df.loc[mask, 'global_hit_rate'].ewm(span=5, adjust=False).mean()

    # ---- Plotting ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    fig.suptitle("System Performance Over Time\n(Averaged Representation of 3 Turns)", fontsize=16, fontweight='bold', y=0.92)
    
    colors = sns.color_palette("tab10", len(METHODS))
    
    for idx, method_label in enumerate(METHOD_LABELS.values()):
        method_df = final_sys_df[final_sys_df['method'] == method_label]
        if method_df.empty: continue
        
        axes[0].plot(method_df['relative_time_s'], method_df['kv_occupancy_pct'], label=method_label, color=colors[idx], linewidth=2)
        axes[1].plot(method_df['relative_time_s'], method_df['throughput_smoothed'], label=method_label, color=colors[idx], linewidth=2)
        axes[2].plot(method_df['relative_time_s'], method_df['hit_rate_smoothed'] * 100, label=method_label, color=colors[idx], linewidth=2)

    # Styling Axes
    axes[0].set_title("KV Cache Occupancy (%)", fontsize=13)
    axes[0].set_ylabel("Occupancy (%)")
    axes[0].legend(loc='lower right')
    
    axes[1].set_title("Generation Throughput (Tokens/sec) - EMA Smoothed", fontsize=13)
    axes[1].set_ylabel("Tokens / sec")
    axes[1].legend(loc='lower right')
    
    axes[2].set_title("Global Cache Hit Rate (%) - EMA Smoothed", fontsize=13)
    axes[2].set_xlabel("Experiment Timeline (Seconds)")
    axes[2].set_ylabel("Hit Rate (%)")
    axes[2].legend(loc='lower right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(VIS_DIR / "03_system_time_series_representative.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("--- Starting Data Visualization Pipeline ---")
    
    print("1. Aggregating Task Metrics across turns...")
    task_df = load_and_aggregate_task_metrics()
    
    if not task_df.empty:
        print("2. Generating Bar Charts (Means)...")
        plot_mean_comparisons(task_df)
        
        print("3. Generating Boxplots (Variance)...")
        plot_variance_distributions(task_df)
    else:
        print("[Error] Task DataFrame is empty. Check clean_data directory.")
        
    print("4. Processing Time-Aligned System Metrics...")
    process_and_plot_system_metrics()
    
    print(f"--- Pipeline Completed! Charts saved to '{VIS_DIR.name}/' ---")

if __name__ == "__main__":
    main()