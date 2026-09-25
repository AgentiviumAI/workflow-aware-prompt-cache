import os
import sys

# ==============================================================================
# ENVIRONMENT SWITCHER
# Because we already have torch and bert_score installed in ~/sglang_env, 
# we must ensure the script is running inside that specific virtual environment.
# Instead of using a bash 'source' command (which only affects sub-shells),
# os.execv dynamically replaces the current running process with the venv's Python.
# ==============================================================================
venv_python = os.path.expanduser("~/sglang_env/bin/python")

# If the current Python interpreter is not the one from sglang_env, switch it.
if sys.executable != venv_python and os.path.exists(venv_python):
    print(f"  [System] Switching interpreter to {venv_python} to access torch & bert_score...")
    # Re-execute the script using the virtual environment's python
    os.execv(venv_python, [venv_python] + sys.argv)

# Now it is safe to import heavy libraries
import torch
from bert_score import BERTScorer

class BERTEvaluator:
    def __init__(self, model_type="roberta-large", lang="en"):
        """
        Initializes the BERTScorer. 
        Using roberta-large as it's the standard for accurate English semantic similarity.
        """
        print(f"  [BERTScore] Loading model: {model_type}...")
        # Check if CUDA is available for faster computation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = BERTScorer(model_type=model_type, lang=lang, rescale_with_baseline=True, device=device)

    def evaluate_batch(self, baseline_answers: list[str], optimized_answers: list[str]) -> list[float]:
        """
        Computes BERTScore (F1) for a batch of text pairs.
        Returns a list of F1 scores.
        """
        if not baseline_answers or not optimized_answers:
            return []
            
        print(f"  [BERTScore] Computing similarity for {len(baseline_answers)} pairs...")
        
        # P, R, F1 tensors are returned
        P, R, F1 = self.scorer.score(optimized_answers, baseline_answers)
        
        # Convert tensor to list of floats
        return F1.tolist()