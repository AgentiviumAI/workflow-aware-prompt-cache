import os
import json
import numpy as np
from scipy import stats
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "clean_data"
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"

BASELINE_METHOD = "fixed_semantic"
OPTIMIZED_METHOD = "pla_character"

def load_latency_data(method_name: str) -> dict:
    file_path = CLEAN_DATA_DIR / method_name / f"{method_name}_cleaned_results.jsonl"
    data_map = {}
    
    if not file_path.exists():
        return data_map
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            turn_idx = record.get("turn_index")
            key = (task_id, turn_idx)
            
            workflow_latency = record.get("workflow_latency", 0.0)
            
            total_ttft = 0.0
            for trace in record.get("agent_traces", []):
                total_ttft += trace.get("metrics", {}).get("ttft", 0.0)
                
            data_map[key] = {
                "workflow_latency": workflow_latency,
                "total_ttft": total_ttft
            }
            
    return data_map

def run_paired_ttest(baseline_data: dict, optimized_data: dict, metric_key: str) -> str:
    common_keys = set(baseline_data.keys()).intersection(set(optimized_data.keys()))
    
    if not common_keys:
        return f"  [WARNING] No common pairs found for metric: {metric_key}\n"
        
    base_values = []
    opt_values = []
    
    for k in common_keys:
        base_values.append(baseline_data[k][metric_key])
        opt_values.append(optimized_data[k][metric_key])
        
    base_arr = np.array(base_values)
    opt_arr = np.array(opt_values)
    
    # 1. Basic Stats
    diffs = base_arr - opt_arr
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1) # Sample standard deviation
    n = len(diffs)
    
    # 2. Paired T-test
    t_stat, p_value = stats.ttest_rel(base_arr, opt_arr)
    
    # 3. Effect Size (Cohen's d for paired samples)
    # Formula: Mean difference / Standard deviation of the differences
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # 4. 95% Confidence Interval
    # Standard Error of the Mean
    sem = std_diff / np.sqrt(n)
    # T-critical value for 95% CI (two-tailed)
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin_of_error = t_crit * sem
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error
    
    # --- Format Report ---
    report = f"\n--- Statistical Evaluation for: {metric_key.upper()} ---\n"
    report += f"Sample Size (n):     {n} pairs (Across 3 turns)\n"
    report += f"Baseline Mean:       {np.mean(base_arr):.4f}s (SD: {np.std(base_arr, ddof=1):.4f})\n"
    report += f"Optimized Mean:      {np.mean(opt_arr):.4f}s (SD: {np.std(opt_arr, ddof=1):.4f})\n"
    report += f"Mean Difference (Δ): {mean_diff:.4f}s\n"
    report += "-" * 40 + "\n"
    report += f"T-Statistic:         {t_stat:.4f}\n"
    report += f"P-Value:             {p_value:.2e}\n"
    report += f"Effect Size (d):     {cohens_d:.4f} "
    
    # Interpret Cohen's d
    if abs(cohens_d) < 0.2: report += "(Negligible Effect)\n"
    elif abs(cohens_d) < 0.5: report += "(Small Effect)\n"
    elif abs(cohens_d) < 0.8: report += "(Medium Effect)\n"
    else: report += "(Large Effect)\n"
    
    report += f"95% CI of Diff:      [{ci_lower:.4f}s, {ci_upper:.4f}s]\n"
    
    # Conclusion
    if p_value < 0.05:
        report += "\n=> CONCLUSION: The reduction in latency is statistically significant (p < 0.05). "
        report += f"With 95% confidence, the true mean time saved lies between {ci_lower:.2f}s and {ci_upper:.2f}s. "
        report += f"The large effect size (d={cohens_d:.2f}) confirms the practical magnitude of the optimization.\n"
    else:
        report += "\n=> CONCLUSION: No statistical significance (p >= 0.05).\n"
        
    return report

def main():
    VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path = VALIDATION_DATA_DIR / "ttest_latency_report.txt"
    
    report_content = "==================================================\n"
    report_content += f" REPEATED RUNS PAIRED T-TEST: {BASELINE_METHOD} vs {OPTIMIZED_METHOD}\n"
    report_content += "==================================================\n"
    
    baseline_data = load_latency_data(BASELINE_METHOD)
    optimized_data = load_latency_data(OPTIMIZED_METHOD)
    
    report_content += run_paired_ttest(baseline_data, optimized_data, "workflow_latency")
    report_content += run_paired_ttest(baseline_data, optimized_data, "total_ttft")
    
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Paired T-Test with Effect Size and 95% CI completed. Report saved to: {report_file_path}")

if __name__ == "__main__":
    main()