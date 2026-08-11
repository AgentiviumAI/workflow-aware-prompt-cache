# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# NEW ALGORITHM: PREFIX-LINEAGE-AWARE (ACTIVE)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
import asyncio
from sglang_client import generate_response
from tools import search_web
from pla_optimizer import PrefixLineageAwareOptimizer_Token, PrefixLineageAwareOptimizer_TokenHash, PrefixLineageAwareOptimizer

# PLA Algorithm
# pla_optimizer = PrefixLineageAwareOptimizer("Qwen/Qwen2.5-7B-Instruct-AWQ")
# pla_optimizer = PrefixLineageAwareOptimizer_Hash("Qwen/Qwen2.5-7B-Instruct-AWQ")
pla_optimizer = PrefixLineageAwareOptimizer()
def create_blocks(policy: str, role: str, instruction: str, task: str, domain: str, tools: str, evidence: str = "", fmt: str = "") -> dict:
    """Tạo các mảnh ghép (Blocks) với thẻ XML chuẩn hóa cho Optimizer."""
    blocks = {
        "POLICY": f"<POLICY>\n{policy}\n</POLICY>\n\n",
        "ROLE": f"<ROLE>\n{role}\n</ROLE>\n\n",
        "CURRENT_TASK": f"<CURRENT_TASK>\n{task}\n</CURRENT_TASK>\n\n",
        "DOMAIN_CONTEXT": f"<DOMAIN_CONTEXT>\n{domain}\n</DOMAIN_CONTEXT>\n\n",
        "TOOLS_AVAILABLE": f"<TOOLS_AVAILABLE>\n{tools}\n</TOOLS_AVAILABLE>\n\n",
        "INSTRUCTION": f"<INSTRUCTION>\n{instruction}\n</INSTRUCTION>\n\n"
    }
    if evidence:
        blocks["EVIDENCE"] = f"<EVIDENCE>\n{evidence}\n</EVIDENCE>\n\n"
    if fmt:
        blocks["FORMAT_REQUIREMENT"] = f"<FORMAT_REQUIREMENT>\n{fmt}\n</FORMAT_REQUIREMENT>\n\n"
    return blocks

async def run_planner(task: str, policy: str) -> dict:
    role = "Planner"
    instruction = "Break down the research task into exactly 2 specific search queries. Output ONLY a valid JSON list of strings representing the queries. Example: [\"query 1\", \"query 2\"]"
    
    blocks = create_blocks(policy, role, instruction, task, "None", "None")
    
    # Planner Node
    opt_res = pla_optimizer.optimize_prompt(blocks, predecessor_traces=[])
    prompt = opt_res["prompt_text"]
    
    res = await generate_response(prompt, max_tokens=150)
    
    return {"agent": "Planner", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}

async def run_researcher(task: str, initial_query: str, researcher_id: int, policy: str, planner_trace: str) -> list:
    role = "Researcher"
    instruction = "Analyze the evidence provided and extract key facts related to the task."
    traces = []
    
    # ==========================
    # LOOP 1
    # ==========================
    evidence_1 = await search_web(initial_query)
    blocks_1 = create_blocks(policy, role, instruction, task, "Web Search Data", "Web Search", evidence_1)
    
    # PLA gets planner_trace
    opt_res_1 = pla_optimizer.optimize_prompt(blocks_1, predecessor_traces=[planner_trace])
    prompt_1 = opt_res_1["prompt_text"]
    
    res_1 = await generate_response(prompt_1, max_tokens=300)
    full_trace_1 = prompt_1 + res_1["text"]
    
    traces.append({
        "agent": f"Researcher {researcher_id} Loop 1",
        "prompt": prompt_1,
        "output": res_1["text"],
        "metrics": res_1.get("metrics", {}),
        "evidence": evidence_1,
        "full_trace": full_trace_1
    })
    
    # ==========================
    # LOOP 2
    # ==========================
    follow_up_query = f"{initial_query} detailed analysis"
    evidence_2 = await search_web(follow_up_query)
    combined_evidence = f"--- First Search ---\n{evidence_1}\n\n--- Second Search ---\n{evidence_2}\n\n--- Previous Thoughts ---\n{res_1['text']}"
    
    blocks_2 = create_blocks(policy, role, instruction, task, "Web Search Data", "Web Search", combined_evidence)
    
    # PLA get trace of Loop 1 to make it Prior Node
    opt_res_2 = pla_optimizer.optimize_prompt(blocks_2, predecessor_traces=[full_trace_1])
    prompt_2 = opt_res_2["prompt_text"]
    
    res_2 = await generate_response(prompt_2, max_tokens=300)
    full_trace_2 = prompt_2 + res_2["text"]
    
    traces.append({
        "agent": f"Researcher {researcher_id} Loop 2",
        "prompt": prompt_2,
        "output": res_2["text"],
        "metrics": res_2.get("metrics", {}),
        "evidence_used": combined_evidence,
        "full_trace": full_trace_2
    })
    
    return traces

async def run_compressor(task: str, researcher_full_traces: dict, policy: str, format_req: str) -> dict:
    role = "Compressor"
    instruction = "Synthesize the research data into a concise summary of approximately 300 to 400 words. Focus on accuracy and comprehensively answering the task."
    
    # Fallback evidence if Trace Embedding doesn't exist
    fallback_evidence = "\n\n".join([f"--- {name} ---\n{trace}" for name, trace in researcher_full_traces.items()])
    blocks = create_blocks(policy, role, instruction, task, "Synthesis Engine", "None", fallback_evidence, format_req)
    
    # TRACE EMBEDDING: ADD all traces of all Researchers to PLA
    predecessor_list = list(researcher_full_traces.values())
    
    opt_res = pla_optimizer.optimize_prompt(
        blocks=blocks, 
        predecessor_traces=predecessor_list,
        embedded_traces=researcher_full_traces  # Plan B (Fan-in)
    )
    prompt = opt_res["prompt_text"]
    
    res = await generate_response(prompt, max_tokens=1024)
    
    return {"agent": "Compressor", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}

async def run_verifier(task: str, summary: str, policy: str, format_req: str, compressor_trace: str) -> dict:
    role = "Verifier"
    instruction = "Check the summary against the original task for completeness and output the final formatted result. Ensure no missing constraints."
    
    blocks = create_blocks(policy, role, instruction, task, "Format Checker", "None", summary, format_req)
    
    # PLA gets trace of Compressor (hay Trace của Researcher nếu chạy Simple Workflow)
    opt_res = pla_optimizer.optimize_prompt(blocks, predecessor_traces=[compressor_trace])
    prompt = opt_res["prompt_text"]
    
    res = await generate_response(prompt, max_tokens=1024)
    
    return {"agent": "Verifier", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}


# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# # OLD ALGORITHM (COMMENTED)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
# import asyncio
# from prompt_builder import build_flat_prompt, build_static_first_prompt, build_fixed_semantic_prompt
# from sglang_client import generate_response
# from tools import search_web

# async def run_planner(task: str, policy: str) -> dict:
#     role = "Planner"
#     instruction = "Break down the research task into exactly 2 specific search queries. Output ONLY a valid JSON list of strings representing the queries. Example: [\"query 1\", \"query 2\"]"
    
#     # prompt = build_flat_prompt(policy, role, instruction, task, "None", "None")
#     # prompt = build_static_first_prompt(policy, role, instruction, task, "None", "None")
#     prompt = build_fixed_semantic_prompt(policy, role, instruction, task, "None", "None")
#     res = await generate_response(prompt, max_tokens=150)
    
#     return {"agent": "Planner", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}

# async def run_verifier(task: str, summary: str, policy: str, format_req: str) -> dict:
#     role = "Verifier"
#     instruction = "Check the summary against the original task for completeness and output the final formatted result. Ensure no missing constraints."
    
#     # prompt = build_flat_prompt(policy, role, instruction, task, "Format Checker", "None", summary, format_req)
#     # prompt = build_static_first_prompt(policy, role, instruction, task, "Format Checker", "None", summary, format_req)
#     prompt = build_fixed_semantic_prompt(policy, role, instruction, task, "Format Checker", "None", summary, format_req)
#     res = await generate_response(prompt, max_tokens=1024)
    
#     return {"agent": "Verifier", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}

# async def run_researcher(task: str, initial_query: str, researcher_id: int, policy: str) -> list:
#     role = "Researcher"
#     instruction = "Analyze the evidence provided and extract key facts related to the task."
#     traces = []
    
#     evidence_1 = await search_web(initial_query)
#     # prompt_1 = build_flat_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", evidence_1)
#     # prompt_1 = build_static_first_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", evidence_1)
#     prompt_1 = build_fixed_semantic_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", evidence_1)
#     res_1 = await generate_response(prompt_1, max_tokens=300)
    
#     traces.append({
#         "agent": f"Researcher {researcher_id} Loop 1",
#         "prompt": prompt_1,
#         "output": res_1["text"],
#         "metrics": res_1.get("metrics", {}),
#         "evidence": evidence_1
#     })
    
#     follow_up_query = f"{initial_query} detailed analysis"
#     evidence_2 = await search_web(follow_up_query)
#     combined_evidence = f"--- First Search ---\n{evidence_1}\n\n--- Second Search ---\n{evidence_2}\n\n--- Previous Thoughts ---\n{res_1['text']}"
    
#     # prompt_2 = build_flat_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", combined_evidence)
#     # prompt_2 = build_static_first_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", combined_evidence)
#     prompt_2 = build_fixed_semantic_prompt(policy, role, instruction, task, "Web Search Data", "Web Search", combined_evidence)
#     res_2 = await generate_response(prompt_2, max_tokens=300)
    
#     traces.append({
#         "agent": f"Researcher {researcher_id} Loop 2",
#         "prompt": prompt_2,
#         "output": res_2["text"],
#         "metrics": res_2.get("metrics", {}),
#         "evidence_used": combined_evidence
#     })
    
#     return traces

# async def run_compressor(task: str, research_data: list, policy: str, format_req: str) -> dict:
#     role = "Compressor"
#     instruction = "Synthesize the research data into a concise summary of approximately 300 to 400 words. Focus on accuracy and comprehensively answering the task."
    
#     evidence = "\n\n".join(research_data)
#     # prompt = build_flat_prompt(policy, role, instruction, task, "Synthesis Engine", "None", evidence, format_req)
#     # prompt = build_static_first_prompt(policy, role, instruction, task, "Synthesis Engine", "None", evidence, format_req)
#     prompt = build_fixed_semantic_prompt(policy, role, instruction, task, "Synthesis Engine", "None", evidence, format_req)
#     res = await generate_response(prompt, max_tokens=1024)
    
#     return {"agent": "Compressor", "prompt": prompt, "output": res["text"], "metrics": res.get("metrics", {})}