import os
import json
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# Current script is located in: <PROJECT_ROOT>/clean_data_utils/
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

INPUT_BASE_DIR = PROJECT_ROOT / "raw_result" / "advanced_workflow"
OUTPUT_BASE_DIR = PROJECT_ROOT / "deepresearchbench_data"
QUERY_FILE_PATH = PROJECT_ROOT.parent / "deep_research_bench-main" / "data" / "prompt_data" / "query.jsonl"

def load_original_queries(query_path: Path) -> dict:
    """
    Loads original prompts into a dictionary mapped by their task ID.
    """
    queries = {}
    if not query_path.exists():
        print(f"[WARNING] Query file not found at: {query_path}")
        return queries
        
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data.get("id")
            
            try:
                task_id = int(task_id)
            except (ValueError, TypeError):
                pass
                
            queries[task_id] = data.get("prompt", "")
            
    return queries

def process_and_split_results(input_file: Path, output_method_dir: Path, query_dict: dict, method_name: str):
    """
    Reads a multi-turn JSONL file, extracts the final verified output,
    matches it with the original prompt, and saves per-turn files with baseline name.
    """
    turn_data = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 1. Extract Task ID
                task_id = data.get("task_id", data.get("id"))
                try:
                    task_id = int(task_id)
                except (ValueError, TypeError):
                    pass
                
                # 2. Extract Turn Index
                turn_index = data.get("turn_index", 1)
                
                # 3. Extract the final answer produced by MAS
                final_answer = data.get("final_output", "")
                
                # Fallback: Extract from traces if final_output is omitted
                if not final_answer and "agent_traces" in data:
                    verifier_trace = next((t for t in data["agent_traces"] if t.get("agent") == "Verifier"), None)
                    if verifier_trace:
                        final_answer = verifier_trace.get("output", "")
                
                # 4. Construct the standard DeepResearchBench schema
                drb_record = {
                    "id": task_id,
                    "prompt": query_dict.get(task_id, ""),
                    "article": final_answer
                }
                
                if turn_index not in turn_data:
                    turn_data[turn_index] = []
                turn_data[turn_index].append(drb_record)
                
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Failed to parse JSON line in {input_file.name}: {e}")

    # 5. Write split JSONL files with baseline name prefix
    for turn_idx, records in turn_data.items():
        output_file_name = f"{method_name}_turn_{turn_idx}_results.jsonl"
        output_file_path = output_method_dir / output_file_name
        
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"  -> Saved {len(records)} tasks to {output_file_path.relative_to(PROJECT_ROOT)}")

def main():
    print("=== STARTING DEEPRERESEARCHBENCH DATA FORMATTER ===")
    print(f"Root Directory:       {PROJECT_ROOT}")
    print(f"Input Directory:      {INPUT_BASE_DIR}")
    print(f"Output Directory:     {OUTPUT_BASE_DIR}")
    print(f"Query File:           {QUERY_FILE_PATH}\n")
    
    if not INPUT_BASE_DIR.exists():
        print(f"[ERROR] Input directory '{INPUT_BASE_DIR}' not found. Ensure raw benchmark runs are finished.")
        return
        
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-load the original queries for matching
    query_dict = load_original_queries(QUERY_FILE_PATH)
    if not query_dict:
        print("[WARNING] Proceeding without original prompts because query dictionary is empty.")
    
    # Iterate across each method folder in raw_result/advanced_workflow/
    method_folders = [p for p in INPUT_BASE_DIR.iterdir() if p.is_dir()]
    
    if not method_folders:
        print(f"[WARNING] No method subdirectories found in '{INPUT_BASE_DIR}'.")
        return
    
    for method_dir in method_folders:
        method_name = method_dir.name
        print(f"Processing Method: {method_name}")
        
        input_file = method_dir / "results.jsonl"
        if not input_file.exists():
            print(f"  [SKIP] '{input_file.name}' not found in {method_dir}")
            continue
            
        output_method_dir = OUTPUT_BASE_DIR / method_name
        output_method_dir.mkdir(parents=True, exist_ok=True)
        
        # Pass the method_name into the processing function
        process_and_split_results(input_file, output_method_dir, query_dict, method_name)
        
    print("\n=== FORMATTING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()