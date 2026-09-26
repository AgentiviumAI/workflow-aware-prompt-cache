# visualization/plot_correlation_matrix.py
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
RAW_DIR = PROJECT_ROOT / "raw_result"
VIS_DIR = PROJECT_ROOT / "visualization_data"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Select Workflow Mode Here: "simple_workflow" or "advanced_workflow"
WORKFLOW_MODE = "advanced_workflow"  

# The 3 foundational baselines to explore dependencies
BASELINES = [
    f"flat_prompt_{WORKFLOW_MODE}",
    f"static_first_{WORKFLOW_MODE}",
    f"fixed_semantic_{WORKFLOW_MODE}"
]

# Better display names for axes
METRIC_LABELS = {
    "workflow_latency": "Workflow Latency",
    "overall_cache_ratio": "Cache Hit Ratio",
    "uncached_prefill_tokens": "Uncached Prefill Tokens",
    "total_ttft": "Total TTFT"
}

# ==============================================================================
# 1. DATA EXTRACTION
# ==============================================================================
def load_and_extract_metrics() -> pd.DataFrame:
    """
    Reads the raw results.jsonl for the foundational baselines.
    Extracts only the specified metrics to build the correlation matrix.
    """
    records = []
    
    for method in BASELINES:
        # Assuming the raw data structure is: raw_result / <workflow> / <method> / results.jsonl
        # If your structure is flat (raw_result/advanced_workflow/flat_prompt/results.jsonl)
        raw_file = RAW_DIR / WORKFLOW_MODE / method.replace(f"_{WORKFLOW_MODE}", "") / "results.jsonl"
        
        # Fallback if the folder name includes the workflow mode
        if not raw_file.exists():
            raw_file = RAW_DIR / WORKFLOW_MODE / method / "results.jsonl"
            
        if not raw_file.exists():
            print(f"[Warning] Raw data not found: {raw_file}")
            continue
            
        with open(raw_file, 'r', encoding='utf-8') as f:
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
                
                # Calculate required metrics
                overall_cache_ratio = (total_cached / total_prompt * 100) if total_prompt > 0 else 0.0
                uncached_prefill_tokens = total_prompt - total_cached
                workflow_latency = data.get("workflow_latency", 0.0)
                
                records.append({
                    "workflow_latency": workflow_latency,
                    "overall_cache_ratio": overall_cache_ratio,
                    "uncached_prefill_tokens": uncached_prefill_tokens,
                    "total_ttft": total_ttft
                })
                
    return pd.DataFrame(records)

# ==============================================================================
# 2. VISUALIZATION
# ==============================================================================
def plot_correlation_matrix(df: pd.DataFrame):
    if df.empty:
        print("[Error] DataFrame is empty. Check raw data paths.")
        return
        
    # Calculate the Pearson correlation matrix
    corr_matrix = df.corr(method='pearson')
    
    # Rename columns and indices for a cleaner plot
    corr_matrix.rename(columns=METRIC_LABELS, index=METRIC_LABELS, inplace=True)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(9, 8))
    
    # Generate a custom diverging colormap similar to the reference image
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr_matrix, 
        annot=True,          # Show the correlation values
        fmt=".2f",           # Format to 2 decimal places
        annot_kws={"size": 18},
        cmap=cmap,           # Color palette
        vmax=1.0,            # Max value for color bar
        vmin=-1.0,           # Min value for color bar
        center=0,            # Center color bar at 0
        square=True,         # Make cells square
        linewidths=.5,       # Add lines between cells
        cbar_kws={"shrink": .5} # Shrink the color bar slightly
    )
    
    # Rotate the labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Add a title corresponding to the selected workflow
    title_suffix = "Simple Workflow" if "simple" in WORKFLOW_MODE else "Advanced Workflow"
    plt.title(f"Correlation Matrix\n(Foundational Baselines - {title_suffix})", fontsize=15, pad=20)
    
    # Save the plot
    output_filename = f"correlation_matrix_{WORKFLOW_MODE}.png"
    plt.tight_layout()
    plt.savefig(VIS_DIR / output_filename, dpi=300)
    plt.close()
    
    print(f"Correlation matrix saved to {VIS_DIR / output_filename}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"--- Starting Correlation Analysis for {WORKFLOW_MODE.upper()} ---")
    df_metrics = load_and_extract_metrics()
    plot_correlation_matrix(df_metrics)