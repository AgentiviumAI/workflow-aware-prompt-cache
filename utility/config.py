import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ==========================================
# SGLANG CONFIGURATION
# ==========================================
SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000")
SGLANG_GENERATE_ENDPOINT = f"{SGLANG_BASE_URL}/generate"

MAX_SEARCH_CONTENT_CHARS = int(os.getenv("MAX_SEARCH_CONTENT_CHARS", "800"))
NUM_TASKS = int(os.getenv("NUM_TASKS", "50")) # Update to 50 for final evaluation
TESTING_MODE = os.getenv("TESTING_MODE", "True").lower() == "true"
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
NUM_REPEATED_RUNS = int(os.getenv("NUM_REPEATED_RUNS", "3")) # Number of benchmark turns

# ==========================================
# PROMPT POLICIES & FORMATS
# ==========================================
POLICY_STRICT_ANALYTICAL = "Maintain strict neutrality. Rely exclusively on factual data and verifiable metrics. Avoid speculative language."
POLICY_EDUCATIONAL = "Explain concepts clearly and comprehensively. Use analogies if helpful, but maintain historical and factual accuracy."
POLICY_CREATIVE_DESCRIPTIVE = "Provide a rich, descriptive, and engaging overview. Highlight cultural impact, aesthetic values, and trends."
DEFAULT_POLICY = "You are a helpful AI assistant. Always adhere to strict formatting and truthful answers."

TOPIC_TO_POLICY = {
    "Finance & Business": POLICY_STRICT_ANALYTICAL,
    "Science & Technology": POLICY_STRICT_ANALYTICAL,
    "Software Development": POLICY_STRICT_ANALYTICAL,
    "Software": POLICY_STRICT_ANALYTICAL,
    "Hardware": POLICY_STRICT_ANALYTICAL,
    "Industrial": POLICY_STRICT_ANALYTICAL,
    "Crime & Law": POLICY_STRICT_ANALYTICAL,
    "Transportation": POLICY_STRICT_ANALYTICAL,
    "Education & Jobs": POLICY_EDUCATIONAL,
    "Health": POLICY_EDUCATIONAL,
    "History": POLICY_EDUCATIONAL,
    "Religion": POLICY_EDUCATIONAL,
    "Social Life": POLICY_EDUCATIONAL,
    "Literature": POLICY_CREATIVE_DESCRIPTIVE,
    "Art & Design": POLICY_CREATIVE_DESCRIPTIVE,
    "Games": POLICY_CREATIVE_DESCRIPTIVE,
    "Entertainment": POLICY_CREATIVE_DESCRIPTIVE,
    "Sports & Fitness": POLICY_CREATIVE_DESCRIPTIVE,
    "Home & Hobbies": POLICY_CREATIVE_DESCRIPTIVE,
    "Travel": POLICY_CREATIVE_DESCRIPTIVE,
    "Food & Dining": POLICY_CREATIVE_DESCRIPTIVE,
    "Fashion & Beauty": POLICY_CREATIVE_DESCRIPTIVE,
}

FORMAT_TABLE = "Provide the final synthesis strictly as a Markdown table comparing key entities."
FORMAT_JSON = "Output the final answer strictly in JSON format with appropriate logical keys."
FORMAT_BULLETS = "Provide an Executive Summary using concise bullet points and bold headers."
FORMAT_REPORT = "Write a structured analytical report with 'Introduction', 'Key Findings', and 'Conclusion' sections."
DEFAULT_FORMAT = "Provide a clear and well-structured final answer."

TOPIC_TO_FORMAT = {
    "Finance & Business": FORMAT_TABLE,
    "Hardware": FORMAT_TABLE,
    "Software Development": FORMAT_JSON,
    "Software": FORMAT_JSON,
    "History": FORMAT_REPORT,
    "Health": FORMAT_BULLETS,
}