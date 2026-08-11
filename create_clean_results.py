import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DIR = "raw_result"
NORM_DIR = "norm_result"
VIS_DIR = "visualizations"
STATS_DIR = "statistics"

# METHODS = ["flat_prompt_simple_workflow_simple", "static_first_simple_workflow_simple", "fixed_semantic_prompt_simple_workflow_simple", "prefix_lineage_aware_simple_workflow_simple"]
METHODS = ["flat_prompt_advanced_workflow", "static_first_advanced_workflow", "fixed_semantic_prompt_advanced_workflow", "prefix_lineage_aware_advanced_workflow"]

for directory in [RAW_DIR, NORM_DIR, VIS_DIR, STATS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==========================================
# 1. DATA NORMALIZATION
# ==========================================
def clean_and_normalize_data(method_name: str):
    raw_path = os.path.join(RAW_DIR, f"{method_name}.jsonl")
    norm_path = os.path.join(NORM_DIR, f"{method_name}_cleaned.jsonl")
    
    if not os.path.exists(raw_path):
        print(f"[Warning] Raw data file not found: {raw_path}")
        return False

    cleaned_records = []
    with open(raw_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            clean_data = {
                "task_id": data.get("task_id"),
                "topic": data.get("topic"),
                "workflow_latency": data.get("workflow_latency"),
                "llm_score": data.get("llm_score"),
                "format_score": data.get("format_score", 0.0),
                "citation_score": data.get("citation_score", 0.0),
                "agents": []
            }
            for trace in data.get("agent_traces", []):
                clean_data["agents"].append({
                    "agent": trace.get("agent"),
                    "metrics": trace.get("metrics", {})
                })
            cleaned_records.append(clean_data)
            
    with open(norm_path, 'w', encoding='utf-8') as outfile:
        for record in cleaned_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True

# ==========================================
# 2. FEATURE EXTRACTION & AGGREGATION
# ==========================================
def extract_task_metrics(method_name: str) -> pd.DataFrame:
    norm_path = os.path.join(NORM_DIR, f"{method_name}_cleaned.jsonl")
    rows = []
    with open(norm_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            total_prompt = 0
            total_cached = 0
            uncached_prefill = 0
            ttft_list, prefill_list = [], []
            
            for agent in data.get("agents", []):
                metrics = agent.get("metrics", {})
                p_tokens = metrics.get("prompt_tokens", 0)
                c_tokens = metrics.get("cached_tokens", 0)
                
                total_prompt += p_tokens
                total_cached += c_tokens
                uncached_prefill += (p_tokens - c_tokens)
                
                if metrics.get("ttft") is not None:
                    ttft_list.append(metrics.get("ttft"))
                if metrics.get("prefill_latency_approx") is not None:
                    prefill_list.append(metrics.get("prefill_latency_approx"))
            
            rows.append({
                "method": method_name,
                "task_id": data.get("task_id"),
                "llm_score": max(0.0, data.get("llm_score", 0.0)),
                "format_score": data.get("format_score", 0.0),
                "citation_score": data.get("citation_score", 0.0),
                "workflow_latency": data.get("workflow_latency", 0.0),
                "total_cached_tokens": total_cached,
                "uncached_prefill_tokens": uncached_prefill,
                "overall_cache_ratio": (total_cached / total_prompt) if total_prompt > 0 else 0.0,
                "avg_prefill_latency": np.mean(prefill_list) if prefill_list else 0.0,
                "avg_ttft": np.mean(ttft_list) if ttft_list else 0.0
            })
    return pd.DataFrame(rows)

# ==========================================
# 3. STATISTICAL CALCULATION
# ==========================================
def calculate_and_save_statistics(df_all: pd.DataFrame):
    metrics = ["llm_score", "format_score", "citation_score", "workflow_latency", 
               "total_cached_tokens", "uncached_prefill_tokens", "overall_cache_ratio", "avg_ttft"]
    stats_df = df_all.groupby("method")[metrics].agg(['mean', 'median', 'var', 'std'])
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    
    csv_path = os.path.join(STATS_DIR, "descriptive_statistics.csv")
    stats_df.to_csv(csv_path)
    return stats_df

# ==========================================
# 4. VISUALIZATION (REQUEST METRICS)
# ==========================================
def plot_mean_comparisons(df_all: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    
    # [NEW] Đã thêm format_score và citation_score vào danh sách metrics
    metrics = ["llm_score", "format_score", "citation_score", "workflow_latency", 
               "total_cached_tokens", "overall_cache_ratio", "uncached_prefill_tokens", "avg_ttft"]
    
    # [NEW] Nâng cấp lưới subplot từ 2x3 lên 2x4 để vừa đủ 8 đồ thị
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Mean Comparison Across Methods (Client-Side Metrics)", fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        sns.barplot(data=df_all, x="method", y=metric, ax=axes[i], hue="method", palette="viridis", legend=False, errorbar=None)
        axes[i].set_title(f"Mean: {metric}", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=15) # Nghiêng nhãn X 15 độ cho khỏi đè nhau
        
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "01_mean_comparison.png"), dpi=300)
    plt.close()

def plot_variance_distributions(df_all: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    
    # [NEW] Tương tự, thêm 2 metric mới vào đồ thị phân phối
    metrics = ["llm_score", "format_score", "citation_score", "workflow_latency", 
               "total_cached_tokens", "overall_cache_ratio", "uncached_prefill_tokens", "avg_ttft"]
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Variance & Distribution (Spread)", fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df_all, x="method", y=metric, ax=axes[i], hue="method", palette="pastel", legend=False, width=0.5)
        sns.stripplot(data=df_all, x="method", y=metric, ax=axes[i], color="black", alpha=0.5, jitter=True, size=4)
        axes[i].set_title(f"Distribution: {metric}", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis='x', rotation=15)
        
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "02_variance_distribution.png"), dpi=300)
    plt.close()

# ==========================================
# 5. TIME-SERIES VISUALIZATION (SYSTEM METRICS)
# ==========================================
def process_and_plot_system_metrics():
    sys_dfs = {}
    max_time_global = 0

    for method in METHODS:
        csv_path = os.path.join(RAW_DIR, f"system_metrics_{method}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['relative_time_s'] = df['relative_time_s'].astype(int) 
            df = df.groupby('relative_time_s').mean().reset_index()

            total_slots = df['kv_used'] + df['kv_evictable'] + df['kv_available']
            df['kv_occupancy_pct'] = np.where(total_slots > 0, ((df['kv_used'] + df['kv_evictable']) / total_slots) * 100, 0.0)

            sys_dfs[method] = df
            if df['relative_time_s'].max() > max_time_global:
                max_time_global = df['relative_time_s'].max()

    if not sys_dfs:
        print("[Warning] No system_metrics.csv found. Skipping time-series charts.")
        return

    common_time_index = pd.DataFrame({'relative_time_s': np.arange(0, max_time_global + 1)})
    normalized_dfs = []

    for method, df in sys_dfs.items():
        merged = pd.merge(common_time_index, df, on='relative_time_s', how='left')
        merged['kv_occupancy_pct'] = merged['kv_occupancy_pct'].ffill().fillna(0)
        merged['gen_throughput'] = merged['gen_throughput'].fillna(0)
        merged['global_hit_rate'] = merged['global_hit_rate'].fillna(0)
        merged['throughput_smoothed'] = merged['gen_throughput'].ewm(span=10, adjust=False).mean()
        merged['hit_rate_smoothed'] = merged['global_hit_rate'].ewm(span=10, adjust=False).mean()
        merged['method'] = method
        normalized_dfs.append(merged)

    final_norm_df = pd.concat(normalized_dfs, ignore_index=True)
    norm_csv_path = os.path.join(NORM_DIR, "normalized_system_metrics.csv")
    final_norm_df.to_csv(norm_csv_path, index=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("System-Level Performance Over Time (Smoothed & Aligned)", fontsize=18, fontweight='bold')
    
    colors = sns.color_palette("tab10", len(METHODS))

    for idx, method in enumerate(METHODS):
        method_df = final_norm_df[final_norm_df['method'] == method]
        if method_df.empty: continue

        axes[0].plot(method_df['relative_time_s'], method_df['kv_occupancy_pct'], label=method, color=colors[idx], linewidth=2)
        axes[1].plot(method_df['relative_time_s'], method_df['throughput_smoothed'], label=method, color=colors[idx], linewidth=2)
        axes[2].plot(method_df['relative_time_s'], method_df['hit_rate_smoothed'] * 100, label=method, color=colors[idx], linewidth=2)

    axes[0].set_title("KV Cache Occupancy (%) - Forward Filled", fontsize=14)
    axes[0].set_ylabel("Occupancy (%)")
    axes[0].legend()

    axes[1].set_title("Generation Throughput (Tokens/sec) - Smoothed (EMA)", fontsize=14)
    axes[1].set_ylabel("Tokens / sec")
    axes[1].legend()

    axes[2].set_title("Global Cache Hit Rate (%) - Smoothed (EMA)", fontsize=14)
    axes[2].set_xlabel("Experiment Time (seconds)")
    axes[2].set_ylabel("Hit Rate (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "03_system_time_series_smoothed.png"), dpi=300)
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Starting Data Processing Pipeline ---")
    
    all_dfs = []
    for method in METHODS:
        if clean_and_normalize_data(method):
            df = extract_task_metrics(method)
            all_dfs.append(df)
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        calculate_and_save_statistics(combined_df)
        plot_mean_comparisons(combined_df)
        plot_variance_distributions(combined_df)
    
    process_and_plot_system_metrics()
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()