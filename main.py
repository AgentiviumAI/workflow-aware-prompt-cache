import asyncio
import json
import time
import re
import httpx
import os
import subprocess
from dotenv import load_dotenv
from utility.data_loader import load_tasks
from utility.agent import run_planner, run_researcher, run_compressor, run_verifier
from utility.utils import check_format_validity, check_citation_hallucination
from utility.config import NUM_TASKS, NUM_REPEATED_RUNS, TESTING_MODE, TOPIC_TO_POLICY, TOPIC_TO_FORMAT, DEFAULT_POLICY, DEFAULT_FORMAT

# 1. Import PLA Optimizers
from utility_baseline.pla_optimizer import (
    PrefixLineageAwareOptimizer_Token,
    PrefixLineageAwareOptimizer_TokenHash,
    PrefixLineageAwareOptimizer_Character
)

# 2. Import Baseline Optimizers (Class Adapters)
from utility_baseline.prompt_builder import (
    FlatPromptOptimizer,
    StaticFirstOptimizer,
    FixedSemanticOptimizer
)

load_dotenv()
SIMPLE_WORKFLOW = os.getenv("SIMPLE_WORKFLOW", "false").lower() == "true"

# def restart_sglang_server():
#     """Restarts the SGLang backend to clear the KV Cache."""
#     print("\n--- Restarting SGLang Server to clear KV Cache ---")
#     os.system("pkill -f sglang") 
#     time.sleep(5) 
#     if os.path.exists("sglang.log"):
#         os.remove("sglang.log")
        
#     venv_activate = os.path.expanduser("~/sglang_env/bin/activate")
#     start_cmd = (
#         f"source {venv_activate} && "
#         "sglang serve --model-path Qwen/Qwen2.5-7B-Instruct-AWQ "
#         "--port 30000 --host 0.0.0.0 --disable-cuda-graph "
#         "--chunked-prefill-size 8192 --max-prefill-tokens 16384 "
#         "--mem-fraction-static 0.87 --enable-metrics > sglang.log 2>&1 &"
#     )
#     os.system(f"bash -c '{start_cmd}'")
    
#     print("Waiting for SGLang server to initialize (reading sglang.log)...")
#     success_msg = "The server is fired up and ready to roll!"
#     max_wait_seconds = 180
#     start_wait = time.time()
    
#     while time.time() - start_wait < max_wait_seconds:
#         if os.path.exists("sglang.log"):
#             with open("sglang.log", "r", encoding="utf-8") as f:
#                 content = f.read()
#                 if success_msg in content:
#                     print("=> SGLang server restarted successfully and is ready for requests!\n")
#                     time.sleep(2) 
#                     return
#         time.sleep(2)
#     print("\n[WARNING] Timeout waiting for SGLang server to start.")

def restart_sglang_server():
    """Restarts the SGLang backend to clear the KV Cache."""
    print("\n--- Restarting SGLang Server to clear KV Cache ---")
    
    # 1. Hard Kill
    os.system("fuser -k 30000/tcp >/dev/null 2>&1")
    os.system("pkill -9 -f 'sglang.launch_server'")
    os.system("pkill -9 -f 'Qwen2.5-7B-Instruct-AWQ'") 
    
    time.sleep(8) 
    
    if os.path.exists("sglang.log"):
        os.remove("sglang.log")
        
    venv_activate = os.path.expanduser("~/sglang_env/bin/activate")
    
    start_cmd = (
        f"source {venv_activate} && "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        "sglang serve --model-path Qwen/Qwen2.5-7B-Instruct-AWQ "
        "--port 30000 --host 0.0.0.0 --disable-cuda-graph "
        "--chunked-prefill-size 8192 --max-prefill-tokens 16384 "
        "--mem-fraction-static 0.82 --enable-metrics > sglang.log 2>&1 &"
    )
    
    os.system(f"bash -c '{start_cmd}'")
    
    print("Waiting for SGLang server to initialize (reading sglang.log)...")
    success_msg = "The server is fired up and ready to roll!"
    max_wait_seconds = 180
    start_wait = time.time()
    
    while time.time() - start_wait < max_wait_seconds:
        if os.path.exists("sglang.log"):
            with open("sglang.log", "r", encoding="utf-8") as f:
                content = f.read()
                if success_msg in content:
                    print("=> SGLang server restarted successfully and is ready for requests!\n")
                    time.sleep(3) 
                    return
        time.sleep(2)
    print("\n[WARNING] Timeout waiting for SGLang server to start.")

async def warmup_sglang():
    """Sends a dummy request to initialize CUDA contexts and allocate memory."""
    print("  -> Executing warm-up request to initialize CUDA context...")
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                "http://localhost:30000/generate", 
                json={"text": "Hello", "sampling_params": {"max_new_tokens": 1}}, 
                timeout=10.0
            )
        except Exception:
            pass

async def monitor_sglang_metrics(metrics_file: str, stop_event: asyncio.Event, interval: float = 1.0):
    """Monitors and saves system metrics to the specified file path."""
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("relative_time_s,kv_used,kv_evictable,kv_available,gen_throughput,global_hit_rate\n")
            
    start_time = time.time()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while not stop_event.is_set():
            try:
                resp = await client.get("http://localhost:30000/metrics")
                if resp.status_code == 200:
                    text = resp.text
                    def extract_metric(name):
                        pattern = name + r'(?:\{[^}]*\})?\s+([0-9\.eE+-]+)'
                        match = re.search(pattern, text)
                        return float(match.group(1)) if match else 0.0

                    kv_used = extract_metric(r'sglang:kv_used_tokens')
                    kv_evictable = extract_metric(r'sglang:kv_evictable_tokens')
                    kv_available = extract_metric(r'sglang:kv_available_tokens')
                    gen_throughput = extract_metric(r'sglang:gen_throughput')
                    global_hit_rate = extract_metric(r'sglang:cache_hit_rate')
                    
                    rel_time = round(time.time() - start_time, 2)
                    with open(metrics_file, "a", encoding="utf-8") as f:
                        f.write(f"{rel_time},{kv_used},{kv_evictable},{kv_available},{gen_throughput},{global_hit_rate}\n")
            except Exception:
                pass 
            await asyncio.sleep(interval)

async def execute_workflow(task_data: dict, semaphore: asyncio.Semaphore, turn_index: int, optimizer: object) -> dict:
    """Executes the prompt optimization workflow using injected optimizer."""
    task_prompt = task_data["prompt"]
    topic = task_data.get("topic", "Unknown")
    
    policy = TOPIC_TO_POLICY.get(topic, DEFAULT_POLICY)
    format_req = TOPIC_TO_FORMAT.get(topic, DEFAULT_FORMAT)
    
    workflow_start = time.time()
    
    async with semaphore:
        planner_res = await run_planner(task_prompt, policy, optimizer)
        planner_trace = planner_res["prompt"] + planner_res["output"]
        
        raw_planner_output = planner_res.get("output", "")
        queries = []
        try:
            match = re.search(r'\[(.*?)\]', raw_planner_output, re.DOTALL)
            if match:
                queries = json.loads(f"[{match.group(1)}]")
            else:
                queries = json.loads(raw_planner_output)
        except json.JSONDecodeError:
            queries = [f"{topic} core concepts", f"{topic} recent advancements"]
            
        if not isinstance(queries, list) or len(queries) == 0:
            queries = [f"{topic} core concepts", f"{topic} recent advancements"]
            
        queries = [str(q) for q in queries][:2]
        while len(queries) < 2: 
            queries.append(f"{topic} additional details")
        
        flat_research_traces = []
        evidence_texts = [] 
        
        if SIMPLE_WORKFLOW:
            target_query = queries[0] if queries else f"{topic} research"
            researcher_traces = await run_researcher(task_prompt, target_query, 1, policy, planner_trace, optimizer)
            
            flat_research_traces.extend(researcher_traces)
            for trace in researcher_traces:
                if "evidence" in trace: evidence_texts.append(trace["evidence"])
                if "evidence_used" in trace: evidence_texts.append(trace["evidence_used"])
            
            last_loop_trace = researcher_traces[-1].get("full_trace", "")
            researcher_output = researcher_traces[-1]["output"]
            
            verifier_res = await run_verifier(task_prompt, researcher_output, policy, format_req, last_loop_trace, optimizer)
            trace = [planner_res] + flat_research_traces + [verifier_res]
            
        else:
            research_tasks = [run_researcher(task_prompt, q, idx + 1, policy, planner_trace, optimizer) for idx, q in enumerate(queries)]
            research_results = await asyncio.gather(*research_tasks)
            
            researcher_full_traces = {}
            for idx, researcher_traces in enumerate(research_results):
                flat_research_traces.extend(researcher_traces)
                last_loop_trace = researcher_traces[-1].get("full_trace", "")
                researcher_full_traces[f"Researcher_{idx+1}"] = last_loop_trace
                
                for tr in researcher_traces:
                    if "evidence" in tr: evidence_texts.append(tr["evidence"])
                    if "evidence_used" in tr: evidence_texts.append(tr["evidence_used"])
            
            compressor_res = await run_compressor(task_prompt, researcher_full_traces, policy, format_req, optimizer)
            compressor_trace = compressor_res["prompt"] + compressor_res["output"]
            
            verifier_res = await run_verifier(task_prompt, compressor_res["output"], policy, format_req, compressor_trace, optimizer)
            trace = [planner_res] + flat_research_traces + [compressor_res, verifier_res]
    
    workflow_latency = time.time() - workflow_start
    final_output = verifier_res["output"]
    
    format_score = check_format_validity(final_output, format_req)
    citation_score = check_citation_hallucination(final_output, evidence_texts)
    
    record = {
        "task_id": task_data["id"],
        "turn_index": turn_index,
        "topic": topic,
        "workflow_latency": round(workflow_latency, 4),
        "format_score": format_score,
        "citation_score": citation_score,
        "final_output": final_output
    }
    
    if TESTING_MODE:
        record["agent_traces"] = trace
        
    return record

async def main():
    OPTIMIZERS_TO_TEST = {
        "flat_prompt": FlatPromptOptimizer,
        "static_first": StaticFirstOptimizer,
        "fixed_semantic": FixedSemanticOptimizer,
        "pla_token": PrefixLineageAwareOptimizer_Token,
        "pla_token_hash": PrefixLineageAwareOptimizer_TokenHash,
        "pla_character": PrefixLineageAwareOptimizer_Character
    }
    
    workflow_type = "simple_workflow" if SIMPLE_WORKFLOW else "advanced_workflow"
    base_output_dir = os.path.join("raw_result", workflow_type)
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" STARTING MULTI-ALGORITHM BENCHMARK | Mode: {workflow_type.upper()}")
    print(f" Tasks per Turn: {NUM_TASKS} | Total Turns: {NUM_REPEATED_RUNS}")
    print(f" Data will be saved to: {base_output_dir}/<method>/")
    print(f"==================================================\n")
    
    global_start_time = time.time()
    
    for opt_name, opt_class in OPTIMIZERS_TO_TEST.items():
        
        # Create subfolder for the specific optimizer
        method_dir = os.path.join(base_output_dir, opt_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # File paths inside the method subfolder
        output_file = os.path.join(method_dir, "results.jsonl")
        
        print(f"\n[INIT] Instantiating optimizer: {opt_name}...")
        if opt_name == "pla_token" or opt_name == "pla_token_hash":
            optimizer_instance = opt_class("Qwen/Qwen2.5-7B-Instruct-AWQ")
        else:
            optimizer_instance = opt_class()
            
        # Clear previous results file for this algorithm
        open(output_file, 'w').close() 
        
        print(f"\n--- RUNNING ALGORITHM: {opt_name.upper()} ---")
        
        for current_turn in range(1, NUM_REPEATED_RUNS + 1):
            print(f"  -> Turn {current_turn}/{NUM_REPEATED_RUNS}")
            
            restart_sglang_server()
            await warmup_sglang()
            
            random_seed_for_turn = 42 + current_turn
            tasks = load_tasks("../deep_research_bench-main/data/prompt_data/query.jsonl", 
                               num_tasks=NUM_TASKS, seed=random_seed_for_turn, 
                               min_task_id=51, max_task_id=100)
            
            semaphore = asyncio.Semaphore(10)
            stop_event = asyncio.Event()
            
            # Save metrics explicitly to the method folder, separated by turn
            metrics_file_path = os.path.join(method_dir, f"system_metrics_turn_{current_turn}.csv")
            monitor_task = asyncio.create_task(monitor_sglang_metrics(metrics_file_path, stop_event, interval=1.0))
            
            turn_start_time = time.time()
            
            coros = [execute_workflow(task, semaphore, current_turn, optimizer_instance) for task in tasks]
            results = await asyncio.gather(*coros)
            
            turn_duration = time.time() - turn_start_time
            
            stop_event.set()
            await monitor_task
            
            with open(output_file, "a", encoding="utf-8") as f:
                for result_record in results:
                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    
            print(f"  -> Completed in {round(turn_duration, 2)} seconds.")
            
        print(f"--- FINISHED ALGORITHM: {opt_name.upper()} ---")

    global_end_time = time.time()
    print(f"\n==================================================")
    print(f" ALL BENCHMARKS FINISHED!")
    print(f" Total time elapsed: {round((global_end_time - global_start_time) / 60, 2)} minutes.")
    print(f"==================================================")

if __name__ == "__main__":
    asyncio.run(main())