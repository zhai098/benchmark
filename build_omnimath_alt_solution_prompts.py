#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量生成 Omni-MATH 的“多解法改写”请求消息（system + user）。

输入：jsonl（每行至少包含 problem / solution / answer）
输出：jsonl（每行只包含一个 message 单元：messages）

示例：
python build_omnimath_alt_solution_prompts.py \
  --input Omni_MATH/Omni_MATH_Human_Segmented_100_1.jsonl \
  --output Omni_MATH/alt_solution_messages_100_1.jsonl \
  --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_MESSAGE = """You are an expert mathematical writing assistant.

You will be given:
- a math problem,
- an existing correct solution (for style reference),
- and the final answer.

Your task is to produce multiple alternative, mathematically correct solution processes that end with exactly the same final answer.

Requirements:
1) Ensure that the reasoning behind the answer is sequential, following a forward-moving logical progression; whenever possible, avoid the practice of working backward—deriving the process from the final answer.
2) Use genuinely different reasoning strategies across alternatives (not paraphrases).
3) Ensure each alternative includes a brief verification step right before the final answer.
4) Keep the final answer at the end of each alternative and match it exactly.
5) Do not change the original problem or add extra assumptions.
"""


USER_TEMPLATE = """You are given a math problem and an existing correct solution (including its final answer). Your job is to generate OTHER valid solution processes that also reach the same final answer.

INPUT

1. Problem:
   << {problem} >>

Original solution (for reference on formatting, notation, and level of detail):
<< {solution} >>

1. Final answer (must match exactly):
   << {answer} >>

TASK

* Produce {n} alternative solution processes (default N=3) that are each mathematically correct and end with the exact same final answer.
* Each alternative must use meaningfully different reasoning, strategy, or intermediate steps (e.g., different algebraic approach, different theorem, different substitution, different viewpoint). Do NOT simply rephrase the original.
* Keep the formatting, tone, notation style, and level of detail as close as possible to the original solution (headings, bulleting, equation style, etc.).
* Present steps sequentially and coherently (no missing leaps).
* The final answer must appear only at the end of each solution, in the same format as the provided final answer.
* Do not change the problem, introduce new constraints, or assume extra information. If an assumption is unavoidable, state it explicitly and ensure it is consistent with the original problem.

QUALITY CHECK (required)
For each alternative solution:

* Include a brief verification step right before the final answer (e.g., substitution/checking, sanity check, or confirming conditions), without moving the final answer away from the end.
* Ensure the final answer matches the provided one exactly.

OUTPUT FORMAT
Alternative Solution 1:

<verification...>
Final Answer: 

Alternative Solution 2:

<verification...>
Final Answer: 

Alternative Solution 3:

<verification...>
Final Answer:
"""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at line {line_no}: {exc}") from exc


def build_user_message(problem: str, solution: str, answer: str, n: int) -> str:
    return USER_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution.strip(),
        answer=answer.strip(),
        n=n,
    )


def build_record(obj: Dict[str, Any], idx: int, n: int) -> Dict[str, Any]:
    problem = str(obj.get("problem", "")).strip()
    solution = str(obj.get("solution", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    if not problem or not solution or not answer:
        raise ValueError(
            f"Record {idx} missing required fields (problem/solution/answer)."
        )

    user_message = build_user_message(problem, solution, answer, n)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate system+user prompt messages for Omni-MATH alternative-solution task."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Omni_MATH/Omni_MATH_Human_Segmented_100_1.jsonl",
        help="Input Omni-MATH JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alt_solution_messages.jsonl",
        help="Output JSONL file with full chat messages.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of alternative solutions required per question.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows: List[Dict[str, Any]] = []
    for idx, obj in enumerate(iter_jsonl(input_path), 1):
        rows.append(build_record(obj, idx, args.n))
        if args.max_cases is not None and len(rows) >= args.max_cases:
            break

    write_jsonl(output_path, rows)
    print(f"[DONE] generated {len(rows)} message records -> {output_path}")


if __name__ == "__main__":
    main()
