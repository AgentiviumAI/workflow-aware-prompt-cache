class FlatPromptOptimizer:
    """
    Constructs a Flat Prompt by concatenating all context linearly without XML boundaries.
    """
    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str] = None, embedded_traces: dict = None) -> dict:
        # Fan-in fallback: Concat all embedded traces if present
        evidence_content = blocks.get("EVIDENCE", "")
        if embedded_traces:
            evidence_content = "\n\n".join([f"--- {name} ---\n{trace}" for name, trace in embedded_traces.items()])
            
        policy = blocks.get("POLICY", "").replace("<POLICY>\n", "").replace("\n</POLICY>\n\n", "")
        role = blocks.get("ROLE", "").replace("<ROLE>\n", "").replace("\n</ROLE>\n\n", "")
        domain = blocks.get("DOMAIN_CONTEXT", "").replace("<DOMAIN_CONTEXT>\n", "").replace("\n</DOMAIN_CONTEXT>\n\n", "")
        tools = blocks.get("TOOLS_AVAILABLE", "").replace("<TOOLS_AVAILABLE>\n", "").replace("\n</TOOLS_AVAILABLE>\n\n", "")
        task = blocks.get("CURRENT_TASK", "").replace("<CURRENT_TASK>\n", "").replace("\n</CURRENT_TASK>\n\n", "")
        instruction = blocks.get("INSTRUCTION", "").replace("<INSTRUCTION>\n", "").replace("\n</INSTRUCTION>\n\n", "")
        fmt = blocks.get("FORMAT_REQUIREMENT", "").replace("<FORMAT_REQUIREMENT>\n", "").replace("\n</FORMAT_REQUIREMENT>\n\n", "")

        prompt = f"System Policy: {policy}\n\n"
        prompt += f"Role: {role}\n\n"
        prompt += f"Domain Context: {domain}\n\n"
        prompt += f"Tools Available: {tools}\n\n"
        
        if evidence_content:
            prompt += f"Evidence: {evidence_content}\n\n"
            
        prompt += f"Current Task: {task}\n\n"
        prompt += f"Instruction: {instruction}\n"
        
        if fmt:
            prompt += f"Output Format Requirement: {fmt}\n"
            
        prompt += "\nProvide your response below:\n"
        
        return {
            "prompt_text": prompt,
            "layout_names": ["FLAT_CONCATENATION"],
            "strategy": "Static_Baseline"
        }


class StaticFirstOptimizer:
    """
    Constructs a Static-First Prompt using XML tags.
    Blocks are ordered from most static (Policy, Tools, Task) to most dynamic (Evidence, Instruction).
    """
    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str] = None, embedded_traces: dict = None) -> dict:
        prompt = ""
        
        # 1. GLOBAL STATIC BLOCKS
        if "POLICY" in blocks: prompt += blocks["POLICY"]
        if "CURRENT_TASK" in blocks: prompt += blocks["CURRENT_TASK"]
        
        # 2. WORKFLOW STATIC BLOCK
        if "FORMAT_REQUIREMENT" in blocks: prompt += blocks["FORMAT_REQUIREMENT"]
        if "TOOLS_AVAILABLE" in blocks: prompt += blocks["TOOLS_AVAILABLE"]
        
        # 3. AGENT DYNAMIC BLOCKS
        if "DOMAIN_CONTEXT" in blocks: prompt += blocks["DOMAIN_CONTEXT"]
        if "ROLE" in blocks: prompt += blocks["ROLE"]
        
        # 4. HIGHLY DYNAMIC BLOCK
        if embedded_traces:
            combined_evidence = "\n\n".join([f"--- {name} ---\n{trace}" for name, trace in embedded_traces.items()])
            prompt += f"<EVIDENCE>\n{combined_evidence}\n</EVIDENCE>\n\n"
        elif "EVIDENCE" in blocks:
            prompt += blocks["EVIDENCE"]
            
        # 5. TAIL
        if "INSTRUCTION" in blocks: prompt += blocks["INSTRUCTION"]
        prompt += "Provide your response below:\n"
        
        return {
            "prompt_text": prompt,
            "layout_names": ["POLICY", "CURRENT_TASK", "FORMAT_REQUIREMENT", "TOOLS_AVAILABLE", "DOMAIN_CONTEXT", "ROLE", "EVIDENCE", "INSTRUCTION"],
            "strategy": "Static_Baseline"
        }


class FixedSemanticOptimizer:
    """
    Constructs a Fixed Semantic Order Prompt.
    Follows natural narrative flow, placing Evidence BEFORE the Current Task.
    """
    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str] = None, embedded_traces: dict = None) -> dict:
        prompt = ""
        
        if "ROLE" in blocks: prompt += blocks["ROLE"]
        if "POLICY" in blocks: prompt += blocks["POLICY"]
        if "DOMAIN_CONTEXT" in blocks: prompt += blocks["DOMAIN_CONTEXT"]
        if "TOOLS_AVAILABLE" in blocks: prompt += blocks["TOOLS_AVAILABLE"]
        
        # Highly dynamic block breaks the prefix lineage here
        if embedded_traces:
            combined_evidence = "\n\n".join([f"--- {name} ---\n{trace}" for name, trace in embedded_traces.items()])
            prompt += f"<EVIDENCE>\n{combined_evidence}\n</EVIDENCE>\n\n"
        elif "EVIDENCE" in blocks:
            prompt += blocks["EVIDENCE"]
            
        # Intra-task static block is forced to recompute
        if "CURRENT_TASK" in blocks: prompt += blocks["CURRENT_TASK"]
        if "INSTRUCTION" in blocks: prompt += blocks["INSTRUCTION"]
        if "FORMAT_REQUIREMENT" in blocks: prompt += blocks["FORMAT_REQUIREMENT"]
            
        prompt += "Provide your response below:\n"
        
        return {
            "prompt_text": prompt,
            "layout_names": ["ROLE", "POLICY", "DOMAIN_CONTEXT", "TOOLS_AVAILABLE", "EVIDENCE", "CURRENT_TASK", "INSTRUCTION", "FORMAT_REQUIREMENT"],
            "strategy": "Static_Baseline"
        }