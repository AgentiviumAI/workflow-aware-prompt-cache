import time
import json
import httpx
from config import SGLANG_GENERATE_ENDPOINT, STOP_TOKENS

async def generate_response(prompt: str, max_tokens: int = 1024) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "repetition_penalty": 1.15,
            "stop": STOP_TOKENS
        },
        "stream": True,
        "return_usage": True
    }
    
    start_time = time.time()
    ttft = None
    generated_text = ""
    meta_info = {}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream("POST", SGLANG_GENERATE_ENDPOINT, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        decoded_line = line.strip()
                        
                   
                        if decoded_line.startswith("data: "):
                            decoded_line = decoded_line[6:]
                            
                        if decoded_line == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(decoded_line)
                            
                
                            if ttft is None:
                                ttft = time.time() - start_time
                                
                            if "text" in chunk:
                                generated_text = chunk["text"]
                                
                    
                            if "meta_info" in chunk:
                                meta_info = chunk["meta_info"]
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"SGLang API Error: {e}")
            return {"error": str(e), "metrics": {}}

    end_time = time.time()
    total_latency = end_time - start_time
    
  
    prompt_tokens = meta_info.get("prompt_tokens", 0)
    cached_tokens = meta_info.get("cached_tokens", 0)
    

    completion_tokens = meta_info.get("completion_tokens", 0) 
    

    cached_token_ratio = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
    
    metrics = {
        "ttft": round(ttft if ttft else total_latency, 4),
        "total_latency": round(total_latency, 4),
        "prefill_latency_approx": round(ttft if ttft else 0.0, 4),
        "input_tokens": prompt_tokens,         
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cached_token_ratio": round(cached_token_ratio, 4)
    }
    
    return {
        "text": generated_text,
        "metrics": metrics
    }