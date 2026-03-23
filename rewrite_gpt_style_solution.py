import argparse
import json
import os
import re
from config import Config
from runner import DEEPSEEK_API_runner, VLLMRunner, TransformersLogProbRunner
from prompt import On_Policy_Prompt, On_Policy_1_Prompt


def extract_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<<<ANSWER>>>\s*(.*?)\s*<<<END>>>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite gpt_style_solution.txt using On_Policy_Prompt from prompt.py."
    )
    parser.add_argument(
        "--input",
        default="gpt_style_solution.txt",
        help="Input text file to rewrite.",
    )
    parser.add_argument(
        "--output",
        default="gpt_style_solution_2_onpolicy.txt",
        help="Output file for rewritten text.",
    )
    
    args = parser.parse_args()

    input_text = read_text(args.input)
    problem = "Does there exist positive reals $a_0, a_1,\\ldots ,a_{19}$, such that the polynomial $P(x)=x^{20}+a_{19}x^{19}+\\ldots +a_1x+a_0$ does not have any real roots, yet all polynomials formed from swapping any two coefficients $a_i,a_j$ has at least one real root?"
    runner = VLLMRunner(
        model=Config["judge_model"],
        vllm_config=Config["judge_model_params"],
        sampling_config=Config["judge_sampling_params"],
        gpus=Config["judge_model_gpus"]
    )
    logprob_runner = TransformersLogProbRunner(
        model=Config["judge_model"],
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    onpolicy = On_Policy_1_Prompt(runner)
    result = onpolicy.run(input_text)
    print(result)
    #rewritten = extract_answer(result)
    rewritten = str(result)
    score = logprob_runner.score(problem=problem, solution=rewritten)
    score_org = logprob_runner.score(problem=problem, solution=input_text)
    print(f"Original Score: {score_org}, Rewritten Score: {score}")
    payload = {
        "input": input_text,
        "rewritten": rewritten,
        "score": score,
    }

    write_text(args.output, json.dumps(payload, ensure_ascii=False, indent=2))
    

    print(f"Saved rewritten text to {args.output}")


if __name__ == "__main__":
    main()
