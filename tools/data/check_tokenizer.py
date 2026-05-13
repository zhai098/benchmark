from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from transformers import AutoTokenizer
from benchmark_core.config import Config

model_name = Config["reasoning_model"]
print(f"Loading tokenizer for {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a mathematician."},
    {"role": "user", "content": "Problem:\n1+1=?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt end repr: {repr(prompt[-20:])}")
print(f"Prompt end: {prompt[-20:]}")

if prompt and not prompt[-1].isspace():
    print("Prompt does not end with space.")
else:
    print("Prompt ends with space.")
