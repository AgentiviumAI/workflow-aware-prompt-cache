import asyncio
from sglang_client import generate_response
from tools import search_web

def create_blocks(policy: str, role: str, instruction: str, task: str, domain: str, tools: str, evidence: str = "", fmt: str = "") -> dict:
    """Construct prompt blocks wrapped in XML tags for the optimizer."""
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

async def run_planner(task: str, policy: str, optimizer: object) -> dict:
    role = "Planner"
    instruction = "Break down the research task into exactly 2 specific search queries. Output ONLY a valid JSON list of strings representing the queries. Example: [\"query 1\", \"query 2\"]"
    blocks = create_blocks(policy, role, instruction, task, "None", "None")
    
    opt_res = optimizer.optimize_prompt(blocks, predecessor_traces=[])
    prompt = opt_res["prompt_text"]
    res = await generate_response(prompt, max_tokens=150)
    
    output_text = res.get("text", "Error: Backend failed to generate response.")
    return {"agent": "Planner", "prompt": prompt, "output": output_text, "metrics": res.get("metrics", {})}

async def run_researcher(task: str, initial_query: str, researcher_id: int, policy: str, planner_trace: str, optimizer: object) -> list:
    role = "Researcher"
    instruction = "Analyze the evidence provided and extract key facts related to the task."
    traces = []
    
    # Loop 1
    evidence_1 = await search_web(initial_query)
    blocks_1 = create_blocks(policy, role, instruction, task, "Web Search Data", "Web Search", evidence_1)
    
    opt_res_1 = optimizer.optimize_prompt(blocks_1, predecessor_traces=[planner_trace])
    prompt_1 = opt_res_1["prompt_text"]
    res_1 = await generate_response(prompt_1, max_tokens=300)
    
    output_text_1 = res_1.get("text", "Error: Backend failed to generate response.")
    full_trace_1 = prompt_1 + output_text_1
    
    traces.append({
        "agent": f"Researcher {researcher_id} Loop 1",
        "prompt": prompt_1,
        "output": output_text_1,
        "metrics": res_1.get("metrics", {}),
        "evidence": evidence_1,
        "full_trace": full_trace_1
    })
    
    # Loop 2
    follow_up_query = f"{initial_query} detailed analysis"
    evidence_2 = await search_web(follow_up_query)
    combined_evidence = f"--- First Search ---\n{evidence_1}\n\n--- Second Search ---\n{evidence_2}\n\n--- Previous Thoughts ---\n{output_text_1}"
    
    blocks_2 = create_blocks(policy, role, instruction, task, "Web Search Data", "Web Search", combined_evidence)
    
    opt_res_2 = optimizer.optimize_prompt(blocks_2, predecessor_traces=[full_trace_1])
    prompt_2 = opt_res_2["prompt_text"]
    res_2 = await generate_response(prompt_2, max_tokens=300)
    
    output_text_2 = res_2.get("text", "Error: Backend failed to generate response.")
    full_trace_2 = prompt_2 + output_text_2
    
    traces.append({
        "agent": f"Researcher {researcher_id} Loop 2",
        "prompt": prompt_2,
        "output": output_text_2,
        "metrics": res_2.get("metrics", {}),
        "evidence_used": combined_evidence,
        "full_trace": full_trace_2
    })
    
    return traces

async def run_compressor(task: str, researcher_full_traces: dict, policy: str, format_req: str, optimizer: object) -> dict:
    role = "Compressor"
    instruction = "Synthesize the research data into a concise summary of approximately 300 to 400 words. Focus on accuracy and comprehensively answering the task."
    
    fallback_evidence = "\n\n".join([f"--- {name} ---\n{trace}" for name, trace in researcher_full_traces.items()])
    blocks = create_blocks(policy, role, instruction, task, "Synthesis Engine", "None", fallback_evidence, format_req)
    
    predecessor_list = list(researcher_full_traces.values())
    opt_res = optimizer.optimize_prompt(
        blocks=blocks, 
        predecessor_traces=predecessor_list,
        embedded_traces=researcher_full_traces 
    )
    prompt = opt_res["prompt_text"]
    res = await generate_response(prompt, max_tokens=1024)
    
    output_text = res.get("text", "Error: Backend failed to generate response.")
    return {"agent": "Compressor", "prompt": prompt, "output": output_text, "metrics": res.get("metrics", {})}

async def run_verifier(task: str, summary: str, policy: str, format_req: str, compressor_trace: str, optimizer: object) -> dict:
    role = "Verifier"
    instruction = "Check the summary against the original task for completeness and output the final formatted result. Ensure no missing constraints."
    
    blocks = create_blocks(policy, role, instruction, task, "Format Checker", "None", summary, format_req)
    
    opt_res = optimizer.optimize_prompt(blocks, predecessor_traces=[compressor_trace])
    prompt = opt_res["prompt_text"]
    res = await generate_response(prompt, max_tokens=1024)
    
    output_text = res.get("text", "Error: Backend failed to generate response.")
    return {"agent": "Verifier", "prompt": prompt, "output": output_text, "metrics": res.get("metrics", {})}