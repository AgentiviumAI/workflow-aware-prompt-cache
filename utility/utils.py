import json
import re
from config import FORMAT_JSON, FORMAT_TABLE, FORMAT_BULLETS, FORMAT_REPORT

def check_format_validity(final_output: str, format_req: str) -> float:
    """
    Rule-based check to verify if the output adheres to the requested format.
    Returns 1.0 (Valid) or 0.0 (Invalid).
    """
    if format_req == FORMAT_JSON:
        try:
            clean_str = final_output.strip().removeprefix("```json").removesuffix("```").strip()
            json.loads(clean_str)
            return 1.0
        except json.JSONDecodeError:
            return 0.0
            
    elif format_req == FORMAT_TABLE:
        if re.search(r'\|[\s\-]+\|', final_output):
            return 1.0
        return 0.0
        
    elif format_req == FORMAT_BULLETS:
        if "\n-" in final_output or "\n*" in final_output:
            return 1.0
        return 0.0
        
    elif format_req == FORMAT_REPORT:
        if "Introduction" in final_output or "Conclusion" in final_output or "Key Findings" in final_output:
            return 1.0
        return 0.0
        
    return 1.0  

def check_citation_hallucination(final_output: str, evidence_texts: list[str]) -> float:
    """
    Cross-checks URLs in the final output against URLs found in the evidence.
    If the LLM hallucinates a fake URL that was not in the search data, returns 0.0.
    Otherwise, returns 1.0.
    """
    url_pattern = r'(https?://[^\s]+)'
    
    output_urls = set(re.findall(url_pattern, final_output))
    if not output_urls:
        return 1.0
        
    evidence_urls = []
    for ev in evidence_texts:
        evidence_urls.extend(re.findall(url_pattern, ev))
    evidence_urls = set(evidence_urls)
    
    for url in output_urls:
        clean_url = url.strip(").,;\"'")
        if not any(clean_url in ev_url for ev_url in evidence_urls):
            return 0.0
            
    return 1.0