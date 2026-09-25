import json
import asyncio
from openai import AsyncOpenAI

class LLMEvaluator:
    def __init__(self, api_keys: list[str], models: list[str]):
        """
        Initialize the Evaluator with a list of API keys for Rate Limit fallback
        and a target model from Groq.
        """
        if not api_keys:
            raise ValueError("[ERROR] No GROQ API Keys provided.")
            
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.model_name = models[0] if models else "llama-3.3-70b-versatile"
        
        # Initialize client with the first key
        self.client = self._get_client()

    def _get_client(self) -> AsyncOpenAI:
        """Returns a new AsyncOpenAI client configured for Groq with the current active key."""
        current_key = self.api_keys[self.current_key_idx]
        return AsyncOpenAI(api_key=current_key, base_url="https://api.groq.com/openai/v1")

    def _rotate_key(self):
        """Rotates to the next API key in the list."""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"  [LLM Judge] Switching to API Key index {self.current_key_idx}")
        self.client = self._get_client()

    async def evaluate_pair(self, question: str, baseline_answer: str, optimized_answer: str, max_retries=3) -> dict:
        """
        Evaluates a single pair of answers using Groq to determine semantic equivalence.
        Includes automatic retry and key rotation on Rate Limits (HTTP 429).
        """
        prompt = f"""
You are an expert evaluator. I will provide you with a Question, a Baseline Answer, and an Optimized Answer.
Your task is to determine if the Optimized Answer preserves the core information, logic, and correctness of the Baseline Answer. Minor wording differences are acceptable.

Question: {question}

Baseline Answer: 
{baseline_answer}

Optimized Answer: 
{optimized_answer}

Output JSON format: {{"is_equivalent": true/false, "reason": "short explanation"}}
"""
        system_prompt = "You are an expert evaluator. Output strictly in JSON format."

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Ensure the expected keys exist
                if "is_equivalent" not in result or "reason" not in result:
                    raise ValueError("JSON missing required keys")
                    
                return result

            except Exception as e:
                err_str = str(e)
                # Check for rate limit or quota issues
                if "429" in err_str or "rate_limit" in err_str.lower():
                    print(f"  [LLM Judge Warning] Rate limit hit on Key {self.current_key_idx}. Rotating key...")
                    self._rotate_key()
                    await asyncio.sleep(1) # Small buffer before retry
                    continue
                else:
                    print(f"  [LLM Judge Error]: {err_str[:100]}...")
                    if attempt == max_retries - 1:
                        return {"is_equivalent": False, "reason": f"Evaluation Failed: {err_str[:50]}"}
                    await asyncio.sleep(2)

        return {"is_equivalent": False, "reason": "Max retries exceeded"}

    async def evaluate_batch(self, questions: list[str], baseline_answers: list[str], optimized_answers: list[str]) -> list[dict]:
        """
        Evaluates a batch of pairs concurrently. 
        Concurrency is slightly limited to avoid instantly nuking all Groq rate limits.
        """
        print(f"  [LLM Judge] Evaluating {len(questions)} pairs with {self.model_name}...")
        
        # Semaphore to limit concurrent API requests (Groq has strict limits per minute)
        sem = asyncio.Semaphore(15) 
        
        async def bounded_eval(q, b, o):
            async with sem:
                return await self.evaluate_pair(q, b, o)
                
        tasks = [bounded_eval(q, b, o) for q, b, o in zip(questions, baseline_answers, optimized_answers)]
        results = await asyncio.gather(*tasks)
        return results

    def calculate_equivalence_percentage(self, llm_results: list[dict]) -> float:
        """Helper function to calculate the percentage of True values."""
        if not llm_results:
            return 0.0
            
        true_count = sum(1 for res in llm_results if res.get("is_equivalent") is True)
        return (true_count / len(llm_results)) * 100