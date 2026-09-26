import os
import json
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# Current script is located in: <PROJECT_ROOT>/clean_data_utils/
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

INPUT_BASE_DIR = PROJECT_ROOT / "raw_result" / "simple_workflow"
OUTPUT_BASE_DIR = PROJECT_ROOT / "clean_data" / "simple_workflow"

def clean_and_consolidate_results(input_file: Path, output_method_dir: Path, method_name: str):
    """
    Reads a multi-turn JSONL file, keeps the top-level metrics, and sanitizes 
    the 'agent_traces' by removing bulky 'prompt' and 'output' fields.
    Saves all turns into a single consolidated JSONL file.
    """
    cleaned_records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Create a new clean dictionary
                clean_data = {
                    "task_id": data.get("task_id", data.get("id")),
                    "turn_index": data.get("turn_index", 1),
                    "topic": data.get("topic", ""),
                    "workflow_latency": data.get("workflow_latency", 0.0),
                    "format_score": data.get("format_score", 0.0),
                    "citation_score": data.get("citation_score", 1.0),
                    "final_output": data.get("final_output", ""),
                    "agent_traces": []
                }
                
                # Process agent_traces: keep metrics, discard prompt/output
                if "agent_traces" in data:
                    for trace in data["agent_traces"]:
                        clean_trace = {
                            "agent": trace.get("agent", "Unknown"),
                            "metrics": trace.get("metrics", {})
                        }
                        
                        # Optionally keep evidence URLs if needed for citation verification later
                        if "evidence" in trace:
                            clean_trace["evidence_length"] = len(trace["evidence"])
                        if "evidence_used" in trace:
                            clean_trace["evidence_used_length"] = len(trace["evidence_used"])
                            
                        clean_data["agent_traces"].append(clean_trace)
                        
                cleaned_records.append(clean_data)
                
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Failed to parse JSON line in {input_file.name}: {e}")

    # Write the consolidated JSONL file for the baseline
    output_file_name = f"{method_name}_cleaned_results.jsonl"
    output_file_path = output_method_dir / output_file_name
    
    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        for record in cleaned_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"  -> Saved {len(cleaned_records)} consolidated tasks to {output_file_path.relative_to(PROJECT_ROOT)}")

def main():
    print("=== STARTING DATA CLEANING PROCESS ===")
    print(f"Root Directory:       {PROJECT_ROOT}")
    print(f"Input Directory:      {INPUT_BASE_DIR}")
    print(f"Output Directory:     {OUTPUT_BASE_DIR}\n")
    
    if not INPUT_BASE_DIR.exists():
        print(f"[ERROR] Input directory '{INPUT_BASE_DIR}' not found. Ensure raw benchmark runs are finished.")
        return
        
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
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
        
        # Clean and consolidate
        clean_and_consolidate_results(input_file, output_method_dir, method_name)
        
    print("\n=== DATA CLEANING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()