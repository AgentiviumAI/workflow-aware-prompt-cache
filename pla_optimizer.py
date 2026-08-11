import itertools
from transformers import AutoTokenizer

class PrefixLineageAwareOptimizer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
        # Init Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # STEP 3: Semantic Constraints  
        # Tuple (A, B) -> Block A MUST be prior to Block B
        # Prevent Structure Hallucation
        self.semantic_constraints = [
            ("POLICY", "CURRENT_TASK"),  # Policy (Global Rule) must be known before knowing the current task
            ("ROLE", "INSTRUCTION"),     # The agent must knows its role before doing as follow
        ]
        
        # STEP 6: Instruction AT TAIL tok take at advantage of Recency Bias
        self.mandatory_tail = ["INSTRUCTION", "FORMAT_REQUIREMENT"]

    def _get_valid_layouts(self, block_keys: list) -> list:
        """Create all kind of order but must follow the Semantic Constraints."""
        floating_blocks = [b for b in block_keys if b not in self.mandatory_tail]
        valid_layouts = []
        
        for perm in itertools.permutations(floating_blocks):
            # Concat the body and the tail
            layout = list(perm) + [b for b in self.mandatory_tail if b in block_keys]
            
            is_valid = True
            for a, b in self.semantic_constraints:
                if a in layout and b in layout:
                    if layout.index(a) > layout.index(b):
                        is_valid = False
                        break
            
            if is_valid:
                valid_layouts.append(layout)
                        
        return valid_layouts

    def _measure_exact_prefix_overlap(self, candidate_str: str, trace_str: str) -> int:
        """STEP 4: ESTIMATE exact-prefix overlap based on TOKEN IDs."""
        if not trace_str:
            return 0
            
        # Encode character into list of Token IDs
        candidate_tokens = self.tokenizer.encode(candidate_str, add_special_tokens=False)
        trace_tokens = self.tokenizer.encode(trace_str, add_special_tokens=False)
            
        m = min(len(candidate_tokens), len(trace_tokens))
        for i in range(m):
            if candidate_tokens[i] != trace_tokens[i]:
                return i
        return m

    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str], embedded_traces: dict = None) -> dict:
        """
        The core of the Algorithm PLA.
        - blocks: Dict is composed of ingredient for Prompt (VD: {"POLICY": "...", "ROLE": "..."})
        - predecessor_traces: List of Prompts + Outputs that has existed from Prior Node (EX: Planner Trace).
        - embedded_traces: (Optional Fan-in) Input Trace of Prior Node to act as Block.
        """
        candidates = []
        
        # Plan A: Standard Orders Block (Fan-out / Sequential Node )
        valid_layouts = self._get_valid_layouts(list(blocks.keys()))
        for layout in valid_layouts:
            prompt_text = "".join([blocks[k] for k in layout])
            candidates.append({
                "prompt_text": prompt_text,
                "layout_names": layout,
                "strategy": "Standard_Composition"
            })

        # Plan B: Trace Embedding ( Bottleneck Fan-in)
        if embedded_traces:
            for prefix_trace_name, prefix_trace_content in embedded_traces.items():

                layout = [f"EMBEDDED_{prefix_trace_name}"]
                prompt_text = prefix_trace_content + "\n\n" 
                           
                prompt_text += "<OTHER_BRANCHES_DATA>\n"
                for other_name, other_content in embedded_traces.items():
                    if other_name != prefix_trace_name:
                        layout.append(f"CONTEXT_{other_name}")
                        prompt_text += f"--- Data from {other_name} ---\n{other_content}\n\n"
                prompt_text += "</OTHER_BRANCHES_DATA>\n\n"
                
                layout.append("ROLE_SWITCH")
                prompt_text += "<ROLE_SWITCH>\nYour role is now COMPRESSOR. Stop researching and synthesize the data.\n</ROLE_SWITCH>\n\n"
                
                for tail in self.mandatory_tail:
                    if tail in blocks:
                        layout.append(tail)
                        prompt_text += blocks[tail]
                
                candidates.append({
                    "prompt_text": prompt_text,
                    "layout_names": layout,
                    "strategy": "Trace_Embedding"
                })

        # STEP 5: SCORE and select the most overlaped Layout
        best_candidate = None
        max_overlap = -1
        
        for candidate in candidates:
            current_max_overlap = 0
            
            # Compare with each Prior Node in the Graph DAG
            if predecessor_traces:
                for trace in predecessor_traces:
                    overlap = self._measure_exact_prefix_overlap(candidate["prompt_text"], trace)
                    if overlap > current_max_overlap:
                        current_max_overlap = overlap
            else:
                # Planner is the first node, overlap=0
                current_max_overlap = 0
                
            candidate["estimated_overlap_tokens"] = current_max_overlap
            
            if current_max_overlap > max_overlap:
                max_overlap = current_max_overlap
                best_candidate = candidate
                
        # Fallback for Safety: If overlap = 0, choose the first valid order
        if best_candidate is None:
            best_candidate = candidates[0]
            
        return best_candidate