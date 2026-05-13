#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import re
from pathlib import Path
from typing import Any

from benchmark_core.paths import ALT_SOLUTIONS_DIR, HIGH_DIFFICULTY_DATA_DIR, STRICT_DIFFICULTY_DIR


STRICT_INPUT_DEFAULT = str(STRICT_DIFFICULTY_DIR / "high_difficulty_quality_ge4_clear_sound_hard_plus_strict_difficulty_api_requests.jsonl")
RAW_INPUT_DEFAULT = str(HIGH_DIFFICULTY_DATA_DIR / "high_difficulty_quality_ge4_clear_sound_hard_plus.jsonl")
OUTPUT_DEFAULT = str(ALT_SOLUTIONS_DIR / "high_difficulty_quality_ge4_clear_sound_hard_plus_multisample_solution_prompts.jsonl")


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

* Produce {n} alternative solution processes that are each mathematically correct and end with the exact same final answer.
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


STRICT_PROMPT_PATTERN = re.compile(
    r"Problem\n(.*?)\n\nReference solution\n(.*?)\n\nFinal answer\n(.*)",
    re.S,
)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at line {line_no} in {path}: {exc}") from exc
    return rows


def extract_problem_from_strict_prompt(user_message: str) -> str:
    match = STRICT_PROMPT_PATTERN.search(user_message)
    if not match:
        raise ValueError("Unable to parse strict-difficulty prompt fields.")
    return match.group(1).strip()


def build_user_message(problem: str, solution: str, answer: str, n: int) -> str:
    return USER_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution.strip(),
        answer=answer.strip(),
        n=n,
    )


def build_record(
    strict_row: dict[str, Any],
    raw_row: dict[str, Any],
    *,
    n: int,
) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": build_user_message(
                    problem=str(raw_row["problem"]),
                    solution=str(raw_row["solution"]),
                    answer=str(raw_row["answer"]),
                    n=n,
                ),
            },
        ],
        "meta": {
            **(strict_row.get("meta") or {}),
            "source": raw_row.get("source"),
            "domain": raw_row.get("domain"),
            "original_difficulty": raw_row.get("difficulty"),
            "question_accuracy": raw_row.get("question_accuracy"),
            "accuracy_correct_count": raw_row.get("accuracy_correct_count"),
            "accuracy_num_responses": raw_row.get("accuracy_num_responses"),
            "accuracy_index": raw_row.get("accuracy_index"),
            "requested_num_samples": n,
        },
    }
    return record


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-sample solution prompts for the strict-difficulty subset."
    )
    parser.add_argument("--strict-input", default=STRICT_INPUT_DEFAULT)
    parser.add_argument("--raw-input", default=RAW_INPUT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--n", type=int, default=6, help="Number of alternative solutions to request.")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer.")

    strict_rows = iter_jsonl(Path(args.strict_input))
    raw_rows = iter_jsonl(Path(args.raw_input))
    raw_by_problem = {str(row["problem"]).strip(): row for row in raw_rows}

    output_rows: list[dict[str, Any]] = []
    for idx, strict_row in enumerate(strict_rows, 1):
        messages = strict_row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Strict row {idx} is missing system/user messages.")
        problem = extract_problem_from_strict_prompt(str(messages[1].get("content", "")))
        raw_row = raw_by_problem.get(problem)
        if raw_row is None:
            raise KeyError(f"Could not match strict row {idx} to raw problem.")
        output_rows.append(build_record(strict_row, raw_row, n=args.n))

    write_jsonl(Path(args.output), output_rows)
    print(f"[DONE] generated {len(output_rows)} multi-sample prompt records -> {args.output}")


if __name__ == "__main__":
    main()
