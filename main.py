import asyncio
import json
import time
import re
import httpx
from data_loader import load_tasks
from agent import run_planner, run_researcher, run_compressor, run_verifier
from utils import evaluate_quality_robust, check_format_validity, check_citation_hallucination
from config import NUM_TASKS, TESTING_MODE, TOPIC_TO_POLICY, TOPIC_TO_FORMAT, DEFAULT_POLICY, DEFAULT_FORMAT
import os

# ==========================================
# SYSTEM METRICS MONITOR (Background Task)
# ==========================================
async def monitor_sglang_metrics(experiment_name: str, stop_event: asyncio.Event, interval: float = 1.0):
    """
    Pings the SGLang /metrics endpoint continuously and logs System Metrics.
    """
    os.makedirs("raw_result", exist_ok=True)
    metrics_file = f"raw_result/system_metrics_{experiment_name}.csv"
    with open(metrics_file, "w", encoding="utf-8") as f:
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

# ==========================================
# WORKFLOW EXECUTION
# ==========================================
async def execute_workflow(task_data: dict, semaphore: asyncio.Semaphore) -> dict:
    task_prompt = task_data["prompt"]
    topic = task_data.get("topic", "Unknown")
    
    # Extract dynamic policy and format based on the topic
    policy = TOPIC_TO_POLICY.get(topic, DEFAULT_POLICY)
    format_req = TOPIC_TO_FORMAT.get(topic, DEFAULT_FORMAT)
    
    workflow_start = time.time()
    
    # ==========================================
    # 1. SGLANG COMPUTE-BOUND BLOCK (Có Semaphore)
    # ==========================================
    async with semaphore:
        # Node 1: Planner
        planner_res = await run_planner(task_prompt, policy)
        
        # [NEW] Trích xuất Trace của Planner để truyền cho Researcher
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
        
        # Node 2: Fan-out Researchers (Nhận planner_trace)
        research_tasks = [run_researcher(task_prompt, q, idx + 1, policy, planner_trace) for idx, q in enumerate(queries)]
        research_results = await asyncio.gather(*research_tasks)
        
        flat_research_traces = []
        evidence_texts = [] 
        
        # [NEW] Dictionary lưu Trace hoàn chỉnh của từng Researcher cho Compressor
        researcher_full_traces = {}
        
        for idx, researcher_traces in enumerate(research_results):
            flat_research_traces.extend(researcher_traces)
            
            # Lấy Trace của Loop cuối cùng (đầy đủ nhất)
            last_loop_trace = researcher_traces[-1].get("full_trace", "")
            researcher_full_traces[f"Researcher_{idx+1}"] = last_loop_trace
            
            for trace in researcher_traces:
                if "evidence" in trace:
                    evidence_texts.append(trace["evidence"])
                if "evidence_used" in trace:
                    evidence_texts.append(trace["evidence_used"])
        
        # Node 3: Fan-in Compressor (Kích hoạt Trace Embedding của PLA)
        compressor_res = await run_compressor(task_prompt, researcher_full_traces, policy, format_req)
        
        # [NEW] Trace of Compressor for Verifier
        compressor_trace = compressor_res["prompt"] + compressor_res["output"]
        
        # Node 4: Verifier (get compressor_trace)
        verifier_res = await run_verifier(task_prompt, compressor_res["output"], policy, format_req, compressor_trace)
    
    workflow_latency = time.time() - workflow_start
    final_output = verifier_res["output"]
    
    # ==========================================
    # 2. I/O-BOUND EVALUATION BLOCK
    # ==========================================
    judge_eval = await evaluate_quality_robust(task_prompt, final_output)
    format_score = check_format_validity(final_output, format_req)
    citation_score = check_citation_hallucination(final_output, evidence_texts)
    
    trace = [planner_res] + flat_research_traces + [compressor_res, verifier_res]
    
    record = {
        "task_id": task_data["id"],
        "topic": topic,
        "workflow_latency": round(workflow_latency, 4),
        "llm_score": judge_eval.get("score", -1.0),
        "format_score": format_score,
        "citation_score": citation_score,
        "llm_reasoning": judge_eval.get("reasoning", ""),
        "final_output": final_output
    }
    
    if TESTING_MODE:
        record["agent_traces"] = trace
        
    return record

# ==========================================
# MAIN
# ==========================================
async def main():
    experiment_name = "prefix_lineage_aware"
    output_file = f"raw_result/{experiment_name}.jsonl"
    
    # Load tasks
    tasks = load_tasks("../deep_research_bench-main/data/prompt_data/query.jsonl", num_tasks=NUM_TASKS, seed=42, min_task_id=51, max_task_id=100)
    
    print(f"Starting experiment: {experiment_name} with {len(tasks)} tasks.")
    
    semaphore = asyncio.Semaphore(10)
    
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_sglang_metrics(experiment_name, stop_event, interval=1.0))
    
    experiment_start_time = time.time()
    
    coros = [execute_workflow(task, semaphore) for task in tasks]
    results = await asyncio.gather(*coros)
    
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    stop_event.set()
    await monitor_task
    
    total_tokens_generated = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for result_record in results:
            f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            if TESTING_MODE and "agent_traces" in result_record:
                for trace in result_record["agent_traces"]:
                    metrics = trace.get("metrics", {})
                    total_tokens_generated += metrics.get("completion_tokens", 0)
    
    system_throughput = total_tokens_generated / total_experiment_time if total_experiment_time > 0 else 0
    
    print(f"Experiment completed. Results saved to {output_file}.")
    print(f"System Level Metrics:")
    print(f" - Total Time: {round(total_experiment_time, 2)} seconds")
    print(f" - Total Completion Tokens Generated: {total_tokens_generated}")
    print(f" - System Throughput: {round(system_throughput, 2)} tokens/sec")
    print(f" - System Memory/Eviction log saved to raw_result/system_metrics_{experiment_name}.csv")

if __name__ == "__main__":
    asyncio.run(main())

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# OLD ALGORITHM (COMMENTED)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=

# import asyncio
# import json
# import time
# import re
# import httpx
# from data_loader import load_tasks
# from agent import run_planner, run_researcher, run_compressor, run_verifier
# from utils import evaluate_quality_robust, check_format_validity, check_citation_hallucination
# from config import NUM_TASKS, TESTING_MODE, TOPIC_TO_POLICY, TOPIC_TO_FORMAT, DEFAULT_POLICY, DEFAULT_FORMAT
# import os

# # ==========================================
# # SYSTEM METRICS MONITOR (Background Task)
# # ==========================================
# async def monitor_sglang_metrics(experiment_name: str, stop_event: asyncio.Event, interval: float = 1.0):
#     """
#     Pings the SGLang /metrics endpoint continuously and logs System Metrics.
#     """
 
#     os.makedirs("raw_result", exist_ok=True)
#     metrics_file = f"raw_result/system_metrics_{experiment_name}.csv"
#     with open(metrics_file, "w", encoding="utf-8") as f:
#         f.write("relative_time_s,kv_used,kv_evictable,kv_available,gen_throughput,global_hit_rate\n")
        
#     start_time = time.time()
    
#     async with httpx.AsyncClient(timeout=5.0) as client:
#         while not stop_event.is_set():
#             try:
#                 resp = await client.get("http://localhost:30000/metrics")
#                 if resp.status_code == 200:
#                     text = resp.text
                    
#                     def extract_metric(name):
                     
#                         pattern = name + r'(?:\{[^}]*\})?\s+([0-9\.eE+-]+)'
#                         match = re.search(pattern, text)
#                         return float(match.group(1)) if match else 0.0

#                     kv_used = extract_metric(r'sglang:kv_used_tokens')
#                     kv_evictable = extract_metric(r'sglang:kv_evictable_tokens')
#                     kv_available = extract_metric(r'sglang:kv_available_tokens')
#                     gen_throughput = extract_metric(r'sglang:gen_throughput')
#                     global_hit_rate = extract_metric(r'sglang:cache_hit_rate')
                    
#                     rel_time = round(time.time() - start_time, 2)
                    
#                     with open(metrics_file, "a", encoding="utf-8") as f:
#                         f.write(f"{rel_time},{kv_used},{kv_evictable},{kv_available},{gen_throughput},{global_hit_rate}\n")
#             except Exception:
#                 pass 
                
#             await asyncio.sleep(interval)

# # ==========================================
# # WORKFLOW EXECUTION
# # ==========================================
# async def execute_workflow(task_data: dict, semaphore: asyncio.Semaphore) -> dict:
#     task_prompt = task_data["prompt"]
#     topic = task_data.get("topic", "Unknown")
    
#     # Extract dynamic policy and format based on the topic
#     policy = TOPIC_TO_POLICY.get(topic, DEFAULT_POLICY)
#     format_req = TOPIC_TO_FORMAT.get(topic, DEFAULT_FORMAT)
    
#     workflow_start = time.time()
    
#     # ==========================================
#     # 1. SGLANG COMPUTE-BOUND BLOCK (Có Semaphore)
#     # ==========================================
#     async with semaphore:
#         # Node 1: Planner
#         planner_res = await run_planner(task_prompt, policy)
        
#         raw_planner_output = planner_res.get("output", "")
#         queries = []
#         try:
#             match = re.search(r'\[(.*?)\]', raw_planner_output, re.DOTALL)
#             if match:
#                 queries = json.loads(f"[{match.group(1)}]")
#             else:
#                 queries = json.loads(raw_planner_output)
#         except json.JSONDecodeError:
#             queries = [f"{topic} core concepts", f"{topic} recent advancements"]
            
#         if not isinstance(queries, list) or len(queries) == 0:
#             queries = [f"{topic} core concepts", f"{topic} recent advancements"]
            
#         queries = [str(q) for q in queries][:2]
#         while len(queries) < 2: 
#             queries.append(f"{topic} additional details")
        
#         # Node 2: Fan-out Researchers
#         research_tasks = [run_researcher(task_prompt, q, idx + 1, policy) for idx, q in enumerate(queries)]
#         research_results = await asyncio.gather(*research_tasks)
        
#         flat_research_traces = []
#         research_texts = []
#         evidence_texts = [] 
        
#         for researcher_traces in research_results:
#             flat_research_traces.extend(researcher_traces)
#             final_loop_output = researcher_traces[-1]["output"]
#             research_texts.append(final_loop_output)
            
#             for trace in researcher_traces:
#                 if "evidence" in trace:
#                     evidence_texts.append(trace["evidence"])
#                 if "evidence_used" in trace:
#                     evidence_texts.append(trace["evidence_used"])
        
#         # Node 3: Fan-in Compressor
#         compressor_res = await run_compressor(task_prompt, research_texts, policy, format_req)
        
#         # Node 4: Verifier
#         verifier_res = await run_verifier(task_prompt, compressor_res["output"], policy, format_req)
    
#     workflow_latency = time.time() - workflow_start
#     final_output = verifier_res["output"]
    
#     # ==========================================
#     # 2. I/O-BOUND EVALUATION BLOCK
#     # ==========================================
#     judge_eval = await evaluate_quality_robust(task_prompt, final_output)
#     format_score = check_format_validity(final_output, format_req)
#     citation_score = check_citation_hallucination(final_output, evidence_texts)
    
#     trace = [planner_res] + flat_research_traces + [compressor_res, verifier_res]
    
#     record = {
#         "task_id": task_data["id"],
#         "topic": topic,
#         "workflow_latency": round(workflow_latency, 4),
#         "llm_score": judge_eval.get("score", -1.0),
#         "format_score": format_score,
#         "citation_score": citation_score,
#         "llm_reasoning": judge_eval.get("reasoning", ""),
#         "final_output": final_output
#     }
    
#     if TESTING_MODE:
#         record["agent_traces"] = trace
        
#     return record

# # ==========================================
# # MAIN
# # ==========================================
# async def main():
#     experiment_name = "fixed_semantic_prompt"
#     output_file = f"raw_result/{experiment_name}.jsonl"
    
#     # Load tasks
#     tasks = load_tasks("../deep_research_bench-main/data/prompt_data/query.jsonl", num_tasks=NUM_TASKS, seed=42, min_task_id=51, max_task_id=100)
    
#     print(f"Starting experiment: {experiment_name} with {len(tasks)} tasks.")
    
#     semaphore = asyncio.Semaphore(10)
    
#     stop_event = asyncio.Event()
#     monitor_task = asyncio.create_task(monitor_sglang_metrics(experiment_name, stop_event, interval=1.0))
    
#     experiment_start_time = time.time()
    
#     coros = [execute_workflow(task, semaphore) for task in tasks]
#     results = await asyncio.gather(*coros)
    
#     experiment_end_time = time.time()
#     total_experiment_time = experiment_end_time - experiment_start_time
    
#     stop_event.set()
#     await monitor_task
    
#     total_tokens_generated = 0
#     with open(output_file, "w", encoding="utf-8") as f:
#         for result_record in results:
#             f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
#             if TESTING_MODE and "agent_traces" in result_record:
#                 for trace in result_record["agent_traces"]:
#                     metrics = trace.get("metrics", {})
#                     total_tokens_generated += metrics.get("completion_tokens", 0)
    
#     system_throughput = total_tokens_generated / total_experiment_time if total_experiment_time > 0 else 0
    
#     print(f"Experiment completed. Results saved to {output_file}.")
#     print(f"System Level Metrics:")
#     print(f" - Total Time: {round(total_experiment_time, 2)} seconds")
#     print(f" - Total Completion Tokens Generated: {total_tokens_generated}")
#     print(f" - System Throughput: {round(system_throughput, 2)} tokens/sec")
#     print(f" - System Memory/Eviction log saved to system_metrics.csv")

# if __name__ == "__main__":
#     asyncio.run(main())