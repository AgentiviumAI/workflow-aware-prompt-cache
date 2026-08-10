def build_flat_prompt(policy: str, role: str, instruction: str, task: str, domain_context: str, tools_context: str, dynamic_evidence: str = "", format_req: str = "") -> str:
    """
    Constructs a Flat Prompt by concatenating all context linearly.
    """
    prompt = f"System Policy: {policy}\n\n"
    prompt += f"Role: {role}\n\n"
    prompt += f"Domain Context: {domain_context}\n\n"
    prompt += f"Tools Available: {tools_context}\n\n"
    
    if dynamic_evidence:
        prompt += f"Evidence: {dynamic_evidence}\n\n"
        
    prompt += f"Current Task: {task}\n\n"
    prompt += f"Instruction: {instruction}\n"
    
    if format_req:
        prompt += f"Output Format Requirement: {format_req}\n"
        
    prompt += "\nProvide your response below:\n"
    
    return prompt

def build_static_first_prompt(policy: str, role: str, instruction: str, task: str, domain_context: str, tools_context: str, dynamic_evidence: str = "", format_req: str = "") -> str:
    """
    Constructs a Static-First Prompt using XML tags.
    Blocks are ordered from most static (Policy, Tools, Task) to most dynamic (Evidence, Instruction).
    """
    # ==========================================
    # 1. GLOBAL STATIC BLOCKS
    # ==========================================
    prompt = f"<POLICY>\n{policy}\n</POLICY>\n\n"

    prompt += f"<CURRENT_TASK>\n{task}\n</CURRENT_TASK>\n\n"
    

        
    # ==========================================
    # 2. WORKFLOW STATIC BLOCK (Intra-task)
    # ==========================================
    if format_req:
        prompt += f"<FORMAT_REQUIREMENT>\n{format_req}\n</FORMAT_REQUIREMENT>\n\n"

    prompt += f"<TOOLS_AVAILABLE>\n{tools_context}\n</TOOLS_AVAILABLE>\n\n"
    
    # ==========================================
    # 3. AGENT DYNAMIC BLOCKS
    # ==========================================
    prompt += f"<DOMAIN_CONTEXT>\n{domain_context}\n</DOMAIN_CONTEXT>\n\n"
    
    prompt += f"<ROLE>\n{role}\n</ROLE>\n\n"
    
    # ==========================================
    # 4. HIGHLY DYNAMIC BLOCK
    # ==========================================
    if dynamic_evidence:
        prompt += f"<EVIDENCE>\n{dynamic_evidence}\n</EVIDENCE>\n\n"
        
    # Instruction is dynamic per agent and placed at the very end for strict adherence (Recency Bias)
    prompt += f"<INSTRUCTION>\n{instruction}\n</INSTRUCTION>\n\n"
    
    prompt += "Provide your response below:\n"
    
    return prompt

def build_fixed_semantic_prompt(policy: str, role: str, instruction: str, task: str, domain_context: str, tools_context: str, dynamic_evidence: str = "", format_req: str = "") -> str:
    """
    Constructs a Fixed Semantic Order Prompt.
    Follows natural narrative flow, placing Evidence BEFORE the Current Task.
    """
    prompt = f"<ROLE>\n{role}\n</ROLE>\n\n"
    prompt += f"<POLICY>\n{policy}\n</POLICY>\n\n"
    prompt += f"<DOMAIN_CONTEXT>\n{domain_context}\n</DOMAIN_CONTEXT>\n\n"
    prompt += f"<TOOLS_AVAILABLE>\n{tools_context}\n</TOOLS_AVAILABLE>\n\n"
    
    # Highly dynamic block breaks the prefix lineage here
    if dynamic_evidence:
        prompt += f"<EVIDENCE>\n{dynamic_evidence}\n</EVIDENCE>\n\n"
        
    # Intra-task static block is forced to recompute
    prompt += f"<CURRENT_TASK>\n{task}\n</CURRENT_TASK>\n\n"
    
    prompt += f"<INSTRUCTION>\n{instruction}\n</INSTRUCTION>\n\n"
    
    if format_req:
        prompt += f"<FORMAT_REQUIREMENT>\n{format_req}\n</FORMAT_REQUIREMENT>\n\n"
        
    prompt += "Provide your response below:\n"
    return prompt