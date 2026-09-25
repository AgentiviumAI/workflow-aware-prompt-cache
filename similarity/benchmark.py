import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add PROJECT_ROOT to sys.path so we can import 'utility.config'
sys.path.append(str(PROJECT_ROOT))

CLEAN_DATA_DIR = PROJECT_ROOT / "clean_data"
SIMILARITY_DIR = PROJECT_ROOT / "similarity_data"

# [UPDATED] Lùi ra một cấp khỏi PROJECT_ROOT để vào thư mục deep_research_bench-main
PROMPT_DATA_PATH = PROJECT_ROOT.parent / "deep_research_bench-main" / "data" / "prompt_data" / "query.jsonl"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GROQ_API_KEYS = [
    k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") 
    if k.strip() and not k.startswith("YOUR_")
]
GROQ_MODELS = [
    m.strip() for m in os.getenv("GROQ_MODELS", "llama-3.3-70b-versatile,llama3-70b-8192").split(",") 
    if m.strip()
]

# --- PREVIOUS EVALUATORS (Commented out as requested) ---
# from bert_score_eval import BERTEvaluator
# from llm_score import LLMEvaluator

# --- NEW QUALITY EVALUATOR ---
from llm_quality import LLMQualityJudge

# Set Target Methods
BASELINE_METHOD = "fixed_semantic"
OPTIMIZED_METHOD = "pla_character"

def load_queries() -> dict:
    """Loads original prompts from query.jsonl mapped by task_id."""
    queries = {}
    if not PROMPT_DATA_PATH.exists():
        print(f"[ERROR] Query file not found: {PROMPT_DATA_PATH}")
        return queries
        
    with open(PROMPT_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_id = int(data.get("id"))
                queries[task_id] = data.get("prompt", "")
    return queries

def load_cleaned_data_mapped(method_name: str) -> dict:
    """Loads clean data and maps it by a tuple (task_id, turn_index)."""
    file_path = CLEAN_DATA_DIR / method_name / f"{method_name}_cleaned_results.jsonl"
    records = {}
    
    if not file_path.exists():
        print(f"[ERROR] Data file not found: {file_path}")
        return records
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                key = (data["task_id"], data["turn_index"])
                records[key] = data
    return records

def load_cleaned_data(method_name: str) -> list[dict]:
    """Loads the consolidated cleaned JSONL file for a given method."""
    file_path = CLEAN_DATA_DIR / method_name / f"{method_name}_cleaned_results.jsonl"
    
    records = []
    if not file_path.exists():
        print(f"[ERROR] Data file not found: {file_path}")
        return records
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    return records

async def main():
    print("==================================================")
    print(f" STARTING QUALITY BENCHMARK")
    print(f" Baseline (Method A):  {BASELINE_METHOD}")
    print(f" Optimized (Method B): {OPTIMIZED_METHOD}")
    print("==================================================\n")
    
    # 1. Load Data
    queries = load_queries()
    data_a = load_cleaned_data_mapped(BASELINE_METHOD)
    data_b = load_cleaned_data_mapped(OPTIMIZED_METHOD)
    
    # Find matching pairs by (task_id, turn_index)
    common_keys = set(data_a.keys()).intersection(set(data_b.keys()))
    if not common_keys:
        print("[ERROR] No common data points found between the two methods.")
        return
        
    # # ---------------------------------------------------------
    # # (COMMENTED OUT) Phase 1 - 2: BERTScore & Equivalence Evaluation
    # # ---------------------------------------------------------
    # from bert_score_eval import BERTEvaluator
    # from llm_score import LLMEvaluator

    # print("==================================================")
    # print(f" STARTING EQUIVALENCE BENCHMARK")
    # print(f" Baseline:  {BASELINE_METHOD}")
    # print(f" Optimized: {OPTIMIZED_METHOD}")
    # print("==================================================\n")
    
    # # 1. Load Data
    # baseline_records = load_cleaned_data(BASELINE_METHOD)
    # optimized_records = load_cleaned_data(OPTIMIZED_METHOD)
    
    # if len(baseline_records) != len(optimized_records):
    #     print(f"[WARNING] Data length mismatch! Baseline: {len(baseline_records)}, Optimized: {len(optimized_records)}")
    #     print("Ensure both methods completed all turns successfully.")
    #     return
        
    # if not baseline_records:
    #     return
        
    # # Prepare text lists for evaluation
    # questions = []
    # baseline_texts = []
    # optimized_texts = []
    
    # for base_rec, opt_rec in zip(baseline_records, optimized_records):
    #     # Using topic as a proxy for the question context
    #     questions.append(f"Generate a comprehensive synthesis regarding: {base_rec.get('topic', 'Unknown')}")
    #     baseline_texts.append(base_rec.get("final_output", ""))
    #     optimized_texts.append(opt_rec.get("final_output", ""))
        
    # # Initialize unified results list based on optimized records structure
    # final_results = []
    # for opt_rec in optimized_records:
    #     new_rec = json.loads(json.dumps(opt_rec)) 
    #     final_results.append(new_rec)
        
    # # ---------------------------------------------------------
    # # 2. RUN BERTSCORE
    # # ---------------------------------------------------------
    # print("\n--- Phase 1: BERTScore Evaluation ---")
    # bert_evaluator = BERTEvaluator()
    # bert_scores = bert_evaluator.evaluate_batch(baseline_texts, optimized_texts)
    
    # for i, score in enumerate(bert_scores):
    #     final_results[i]["bert_score"] = round(float(score), 4)
        
    # # ---------------------------------------------------------
    # # 3. RUN LLM-AS-A-JUDGE
    # # ---------------------------------------------------------
    # print("\n--- Phase 2: LLM Equivalence Evaluation ---")
    # llm_evaluator = LLMEvaluator(api_keys=GROQ_API_KEYS, models=GROQ_MODELS)
    # llm_results = await llm_evaluator.evaluate_batch(questions, baseline_texts, optimized_texts)
    
    # for i, res in enumerate(llm_results):
    #     final_results[i]["llm_is_equivalent"] = res.get("is_equivalent", False)
    #     final_results[i]["llm_reason"] = res.get("reason", "")
        
    # equivalence_pct = llm_evaluator.calculate_equivalence_percentage(llm_results)
    
    # # ---------------------------------------------------------
    # # 4. SAVE RESULTS
    # # ---------------------------------------------------------
    # output_dir = SIMILARITY_DIR / OPTIMIZED_METHOD
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # output_file = output_dir / f"similarity_vs_{BASELINE_METHOD}.json"
    
    # final_package = {
    #     "baseline_method": BASELINE_METHOD,
    #     "optimized_method": OPTIMIZED_METHOD,
    #     "total_records": len(final_results),
    #     "llm_equivalence_percentage": round(equivalence_pct, 2),
    #     "average_bert_score": round(sum(bert_scores) / len(bert_scores), 4) if bert_scores else 0.0,
    #     "data": final_results
    # }
    
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(final_package, f, ensure_ascii=False, indent=4)
        
    # print(f"\n==================================================")
    # print(f" EVALUATION COMPLETED")
    # print(f" Average BERTScore (F1): {final_package['average_bert_score']}")
    # print(f" LLM Equivalence Match:  {final_package['llm_equivalence_percentage']}%")
    # print(f" Output saved to:        {output_file.relative_to(PROJECT_ROOT)}")
    # print(f"==================================================")

    # ---------------------------------------------------------
    # Phase 3: LLM Quality Judge
    # ---------------------------------------------------------
    print("\n--- Phase 3: LLM Quality Evaluation ---")
    evaluation_tasks = []
    
    for key in common_keys:
        task_id, turn_idx = key
        if task_id not in queries:
            continue
            
        evaluation_tasks.append({
            "task_id": task_id,
            "turn_index": turn_idx,
            "prompt": queries[task_id],
            "topic": data_a[key].get("topic", "Unknown"),
            "output_a": data_a[key].get("final_output", ""),
            "output_b": data_b[key].get("final_output", "")
        })

    judge = LLMQualityJudge(api_keys=GROQ_API_KEYS, models=GROQ_MODELS)
    results = await judge.evaluate_batch(evaluation_tasks)
    
    # Calculate Scores
    total_score_a = sum(res["score_a"] for res in results)
    total_score_b = sum(res["score_b"] for res in results)
    total_items = len(results)
    
    # Save Results
    SIMILARITY_DIR.mkdir(parents=True, exist_ok=True)
    output_file = SIMILARITY_DIR / OPTIMIZED_METHOD / f"quality_comparison_{BASELINE_METHOD}_vs_{OPTIMIZED_METHOD}.json"
    
    final_package = {
        "method_a": BASELINE_METHOD,
        "method_b": OPTIMIZED_METHOD,
        "total_evaluated": total_items,
        "method_a_total_score": total_score_a,
        "method_b_total_score": total_score_b,
        "method_a_quality_rate": round((total_score_a / total_items) * 100, 2) if total_items else 0,
        "method_b_quality_rate": round((total_score_b / total_items) * 100, 2) if total_items else 0,
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_package, f, ensure_ascii=False, indent=4)
        
    print(f"\n==================================================")
    print(f" QUALITY EVALUATION COMPLETED")
    print(f" Total Pairs Evaluated: {total_items}")
    print(f" {BASELINE_METHOD} Quality Rate: {final_package['method_a_quality_rate']}% ({total_score_a}/{total_items})")
    print(f" {OPTIMIZED_METHOD} Quality Rate: {final_package['method_b_quality_rate']}% ({total_score_b}/{total_items})")
    print(f" Output saved to: {output_file.relative_to(PROJECT_ROOT)}")
    print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(main())