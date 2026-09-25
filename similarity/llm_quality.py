import json
import asyncio
from openai import AsyncOpenAI
from utility.config import TOPIC_TO_POLICY, DEFAULT_POLICY, TOPIC_TO_FORMAT, DEFAULT_FORMAT

class LLMQualityJudge:
    def __init__(self, api_keys: list[str], models: list[str]):
        if not api_keys:
            raise ValueError("[ERROR] No GROQ API Keys provided.")
        self.api_keys = api_keys
        self.current_key_idx = 0
        # Use the provided model or fallback to 70b
        self.model_name = models[0] if models else "llama-3.3-70b-versatile"
        self.client = self._get_client()

    def _get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_keys[self.current_key_idx], base_url="https://api.groq.com/openai/v1")

    def _rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"  [LLM Quality] Switching to API Key index {self.current_key_idx}")
        self.client = self._get_client()

    async def evaluate_quality(self, task_id: int, turn_idx: int, prompt: str, topic: str, output_a: str, output_b: str, max_total_retries=10) -> dict:
        policy = TOPIC_TO_POLICY.get(topic, DEFAULT_POLICY)
        expected_format = TOPIC_TO_FORMAT.get(topic, DEFAULT_FORMAT)
        
        eval_prompt = f"""
You are an expert, impartial AI judge evaluating the quality of two AI-generated outputs based on a specific prompt, policy, and format requirement.

**Original Prompt:** 
{prompt}

**Required Policy for this Topic ({topic}):** 
{policy}

**Required Format:** 
{expected_format}

**Evaluation Criteria:**
1. Accuracy & Relevance: Does the output directly answer the prompt? Is the information factual?
2. Policy Adherence: Does the tone and content align with the required policy?
3. Format Compliance: Is the output strictly in the requested format?
4. Comprehensiveness: Is the information deep and complete?

Score each method independently. A method gets 1 point if it generally satisfies all criteria well. It gets 0 points if it fails significantly on format, policy, or correctness. Both can be 1, both can be 0, or they can be different.

**Method A Output:**
{output_a}

**Method B Output:**
{output_b}

Output strictly in JSON format using this structure:
{{"score_a": 0 or 1, "score_b": 0 or 1, "reason": "Short explanation justifying both scores"}}
"""
        system_prompt = "You are a strict, objective AI evaluator. Output ONLY valid JSON."

        retries_on_current_key = 0
        
        for attempt in range(max_total_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": eval_prompt}
                    ],
                    temperature=0.0
                )
                
                result = json.loads(response.choices[0].message.content)
                if "score_a" not in result or "score_b" not in result:
                    raise ValueError("JSON missing required score keys")
                
                return {
                    "task_id": task_id,
                    "turn_index": turn_idx,
                    "score_a": int(result["score_a"]),
                    "score_b": int(result["score_b"]),
                    "reason": result.get("reason", "")
                }

            except Exception as e:
                err_str = str(e)
                # Handle Rate Limit (HTTP 429) specifically
                if "429" in err_str or "rate_limit" in err_str.lower():
                    retries_on_current_key += 1
                    if retries_on_current_key <= 1:
                        # First rate limit hit on this key: Wait 5s and try again (Based on 30 RPM logic)
                        # print(f"  [Warning] Rate limit on Key {self.current_key_idx}. Waiting 5s...")
                        await asyncio.sleep(5)
                    else:
                        # Second hit: The key is exhausted for now. Rotate and reset counter.
                        self._rotate_key()
                        retries_on_current_key = 0
                        await asyncio.sleep(1)
                else:
                    # Non-429 error (e.g., connection issue)
                    # print(f"  [Error] {err_str[:50]}")
                    await asyncio.sleep(2)

        return {"task_id": task_id, "turn_index": turn_idx, "score_a": 0, "score_b": 0, "reason": "Max retries exceeded"}

    async def evaluate_batch(self, tasks_data: list) -> list[dict]:
        """
        Evaluates a batch of pairs concurrently. 
        Semaphore of 6 ensures we don't bombard the API and respect the overall 6-key pool limits.
        """
        print(f"  [LLM Quality] Evaluating {len(tasks_data)} items with {self.model_name}...")
        sem = asyncio.Semaphore(6) 
        
        async def bounded_eval(data):
            async with sem:
                return await self.evaluate_quality(
                    data["task_id"], data["turn_index"], data["prompt"], 
                    data["topic"], data["output_a"], data["output_b"]
                )
                
        tasks = [bounded_eval(d) for d in tasks_data]
        return await asyncio.gather(*tasks)