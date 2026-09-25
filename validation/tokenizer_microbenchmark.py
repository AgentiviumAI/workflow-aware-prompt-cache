import os
import sys
import random
from pathlib import Path

# ==============================================================================
# ENVIRONMENT SWITCHER
# ==============================================================================
venv_python = os.path.expanduser("~/sglang_env/bin/python")
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"[System] Switching interpreter to {venv_python} for Tokenizer access...")
    os.execv(venv_python, [venv_python] + sys.argv)

from transformers import AutoTokenizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
NUM_TRIALS = 10000

BLOCKS = {
    "POLICY": "<POLICY>\nYou are a helpful AI.\n</POLICY>\n\n",
    "ROLE": "<ROLE>\nResearcher\n</ROLE>\n\n",
    "TASK": "<CURRENT_TASK>\nFind info on AI.\n</CURRENT_TASK>\n\n",
    "EVIDENCE": "<EVIDENCE>\nSource: URL\nContent: Data points.\n</EVIDENCE>\n\n",
    "INSTRUCTION": "<INSTRUCTION>\nSummarize the text.\n</INSTRUCTION>\n\n"
}

def generate_random_prefix_and_suffix():
    keys = list(BLOCKS.keys())
    random.shuffle(keys)
    
    split_idx = random.randint(1, len(keys) - 1)
    prefix_keys = keys[:split_idx]
    suffix_keys = keys[split_idx:]
    
    prefix_str = "".join([BLOCKS[k] for k in prefix_keys])
    suffix_str = "".join([BLOCKS[k] for k in suffix_keys])
    
    return prefix_str, suffix_str

def main():
    VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path = VALIDATION_DATA_DIR / "tokenizer_microbenchmark_report.txt"
    
    report_content = "==================================================\n"
    report_content += " TOKENIZER MICROBENCHMARK (Boundary Bleed Test)\n"
    report_content += f" Model: {MODEL_NAME}\n"
    report_content += f" Trials: {NUM_TRIALS}\n"
    report_content += "==================================================\n\n"
    
    print("Loading Tokenizer and running simulation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    disagreement_count = 0
    total_tokens_lost = 0
    
    for _ in range(NUM_TRIALS):
        prefix_str, suffix_str = generate_random_prefix_and_suffix()
        full_str = prefix_str + suffix_str
        
        prefix_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)
        full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
        
        token_overlap_len = 0
        min_len = min(len(prefix_tokens), len(full_tokens))
        for j in range(min_len):
            if prefix_tokens[j] == full_tokens[j]:
                token_overlap_len += 1
            else:
                break
                
        if token_overlap_len < len(prefix_tokens):
            disagreement_count += 1
            tokens_lost = len(prefix_tokens) - token_overlap_len
            total_tokens_lost += tokens_lost
            
    disagreement_rate = (disagreement_count / NUM_TRIALS) * 100
    
    report_content += "--- RESULTS ---\n"
    report_content += f"Total Trials:                {NUM_TRIALS}\n"
    report_content += f"Boundary Disagreements:      {disagreement_count}\n"
    report_content += f"Disagreement Rate:           {disagreement_rate:.2f}%\n"
    
    if disagreement_count > 0:
        avg_loss = total_tokens_lost / disagreement_count
        report_content += f"Avg Tokens Lost per Bleed:   {avg_loss:.2f} tokens\n"
    
    report_content += "\n--- CONCLUSION ---\n"
    if disagreement_rate < 1.0:
        report_content += "=> SUCCESS: The Character-level proxy is highly accurate. Boundary bleed is negligible.\n"
    else:
        report_content += f"=> DISCOVERY: High boundary bleed detected ({disagreement_rate:.2f}%).\n"
        report_content += "   This proves Character-level matching is a flawed heuristic for KV Cache,\n"
        report_content += "   validating the necessity of Token-level optimization algorithms.\n"

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Tokenizer Microbenchmark completed. Report saved to: {report_file_path}")

if __name__ == "__main__":
    main()