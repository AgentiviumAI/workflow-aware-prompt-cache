import itertools
from transformers import AutoTokenizer

class PrefixLineageAwareOptimizer_Character:
    def __init__(self):
        self.semantic_constraints = [
            ("POLICY", "CURRENT_TASK"),
            ("ROLE", "INSTRUCTION"),
        ]
        self.mandatory_tail = ["INSTRUCTION", "FORMAT_REQUIREMENT"]

    def _get_valid_layouts(self, block_keys: list) -> list:
        floating_blocks = [b for b in block_keys if b not in self.mandatory_tail]
        valid_layouts = []
        
        for perm in itertools.permutations(floating_blocks):
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
        if not trace_str:
            return 0
            
        m = min(len(candidate_str), len(trace_str))
        for i in range(m):
            if candidate_str[i] != trace_str[i]:
                return i
        return m

    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str], embedded_traces: dict = None) -> dict:
        candidates = []
        
        valid_layouts = self._get_valid_layouts(list(blocks.keys()))
        for layout in valid_layouts:
            prompt_text = "".join([blocks[k] for k in layout])
            candidates.append({
                "prompt_text": prompt_text,
                "layout_names": layout,
                "strategy": "Standard_Composition"
            })

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

        best_candidate = None
        max_overlap = -1
        
        for candidate in candidates:
            current_max_overlap = 0
            if predecessor_traces:
                for trace in predecessor_traces:
                    overlap = self._measure_exact_prefix_overlap(candidate["prompt_text"], trace)
                    if overlap > current_max_overlap:
                        current_max_overlap = overlap
            else:
                current_max_overlap = 0
                
            candidate["estimated_overlap_chars"] = current_max_overlap
            
            if current_max_overlap > max_overlap:
                max_overlap = current_max_overlap
                best_candidate = candidate
                
        if best_candidate is None:
            best_candidate = candidates[0]
            
        return best_candidate


class PrefixLineageAwareOptimizer_Token:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.semantic_constraints = [
            ("POLICY", "CURRENT_TASK"),
            ("ROLE", "INSTRUCTION"),
        ]
        self.mandatory_tail = ["INSTRUCTION", "FORMAT_REQUIREMENT"]

    def _get_valid_layouts(self, block_keys: list) -> list:
        floating_blocks = [b for b in block_keys if b not in self.mandatory_tail]
        valid_layouts = []
        for perm in itertools.permutations(floating_blocks):
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
        if not trace_str:
            return 0
        candidate_tokens = self.tokenizer.encode(candidate_str, add_special_tokens=False)
        trace_tokens = self.tokenizer.encode(trace_str, add_special_tokens=False)
            
        m = min(len(candidate_tokens), len(trace_tokens))
        for i in range(m):
            if candidate_tokens[i] != trace_tokens[i]:
                return i
        return m

    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str], embedded_traces: dict = None) -> dict:
        candidates = []
        valid_layouts = self._get_valid_layouts(list(blocks.keys()))
        for layout in valid_layouts:
            prompt_text = "".join([blocks[k] for k in layout])
            candidates.append({
                "prompt_text": prompt_text,
                "layout_names": layout,
                "strategy": "Standard_Composition"
            })

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

        best_candidate = None
        max_overlap = -1
        
        for candidate in candidates:
            current_max_overlap = 0
            if predecessor_traces:
                for trace in predecessor_traces:
                    overlap = self._measure_exact_prefix_overlap(candidate["prompt_text"], trace)
                    if overlap > current_max_overlap:
                        current_max_overlap = overlap
            else:
                current_max_overlap = 0
                
            candidate["estimated_overlap_tokens"] = current_max_overlap
            
            if current_max_overlap > max_overlap:
                max_overlap = current_max_overlap
                best_candidate = candidate
                
        if best_candidate is None:
            best_candidate = candidates[0]
            
        return best_candidate


class PrefixLineageAwareOptimizer_TokenHash:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._token_cache = {} # Memoization Cache
        self.semantic_constraints = [
            ("POLICY", "CURRENT_TASK"),  
            ("ROLE", "INSTRUCTION"),     
        ]
        self.mandatory_tail = ["INSTRUCTION", "FORMAT_REQUIREMENT"]

    def _get_valid_layouts(self, block_keys: list) -> list:
        floating_blocks = [b for b in block_keys if b not in self.mandatory_tail]
        valid_layouts = []
        for perm in itertools.permutations(floating_blocks):
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

    def _measure_exact_prefix_overlap(self, candidate_tokens: list, trace_tokens: list) -> int:
        if not trace_tokens:
            return 0
        m = min(len(candidate_tokens), len(trace_tokens))
        for i in range(m):
            if candidate_tokens[i] != trace_tokens[i]:
                return i
        return m

    def optimize_prompt(self, blocks: dict, predecessor_traces: list[str], embedded_traces: dict = None) -> dict:
        candidates = []
        valid_layouts = self._get_valid_layouts(list(blocks.keys()))
        for layout in valid_layouts:
            prompt_text = "".join([blocks[k] for k in layout])
            candidates.append({
                "prompt_text": prompt_text,
                "layout_names": layout,
                "strategy": "Standard_Composition"
            })

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

        # Batch Encoding Optimization
        strings_to_tokenize = []
        for candidate in candidates:
            if candidate["prompt_text"] not in self._token_cache:
                strings_to_tokenize.append(candidate["prompt_text"])
                
        if predecessor_traces:
            for trace in predecessor_traces:
                if trace not in self._token_cache:
                    strings_to_tokenize.append(trace)

        if strings_to_tokenize:
            batch_results = self.tokenizer(strings_to_tokenize, add_special_tokens=False)["input_ids"]
            for text, tokens in zip(strings_to_tokenize, batch_results):
                self._token_cache[text] = tokens

        best_candidate = None
        max_overlap = -1
        
        for candidate in candidates:
            current_max_overlap = 0
            candidate_tokens = self._token_cache[candidate["prompt_text"]]
            
            if predecessor_traces:
                for trace in predecessor_traces:
                    trace_tokens = self._token_cache[trace]
                    overlap = self._measure_exact_prefix_overlap(candidate_tokens, trace_tokens)
                    if overlap > current_max_overlap:
                        current_max_overlap = overlap
            else:
                current_max_overlap = 0
                
            candidate["estimated_overlap_tokens"] = current_max_overlap
            
            if current_max_overlap > max_overlap:
                max_overlap = current_max_overlap
                best_candidate = candidate
                
        if best_candidate is None:
            best_candidate = candidates[0]
            
        return best_candidate