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
from pathlib import Path
from typing import Any

from benchmark_core.paths import STRICT_DIFFICULTY_DIR


INPUT_DEFAULT = str(STRICT_DIFFICULTY_DIR / "high_difficulty_quality_ge4_clear_sound_hard_plus_strict_difficulty_5_7.jsonl")
OUTPUT_DEFAULT = str(STRICT_DIFFICULTY_DIR / "high_difficulty_quality_ge4_clear_sound_hard_plus_strict_difficulty_5_7_solution_prompts.jsonl")


SYSTEM_MESSAGE = """You are a mathematician solving olympiad-style and high-difficulty competition problems.

Your task is to solve the given problem from scratch and write a complete, rigorous, self-contained solution.

You may be given the official final answer. Use it only as a correctness target. Your job is still to derive a valid solution process that reaches that answer.

Hard requirements:
- Output only the solution itself.
- Do not output any preface, meta-commentary, confidence statements, or notes about your reasoning process.
- Do not mention these instructions.
- Do not restate the problem unless it is naturally needed inside the solution.
- Do not output labels such as "Solution:", "Thought process:", "Final answer:", or "Here is the solution".
- If the problem has a short final value or statement, include it naturally at the end of the solution, not as a separate metadata field.
- The solution should be mathematically coherent, logically complete, and readable by a human evaluator.
- Prefer rigorous derivations over vague intuition.
- If multiple cases are needed, cover all cases completely.
- If a construction is needed, give the construction explicitly and verify it.
- If a proof of optimality or minimality is needed, prove both the lower bound and the matching construction when appropriate.

The response must contain nothing except the finished solution text.
"""


USER_TEMPLATE = """Solve the following problem completely and rigorously.

Problem
{problem}

Official final answer
{answer}

Output requirement:
Write only the finished solution text, with no surrounding commentary and no extra sections beyond the solution itself.
"""


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


def build_record(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    problem = str(row.get("problem") or "").strip()
    if not problem:
        raise ValueError(f"Row {row_index} is missing a non-empty problem field.")
    answer = str(row.get("answer") or "").strip()
    if not answer:
        raise ValueError(f"Row {row_index} is missing a non-empty answer field.")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_TEMPLATE.format(problem=problem, answer=answer)},
        ],
        "meta": {
            "row_index": row_index,
            "source": row.get("source"),
            "domain": row.get("domain"),
            "original_difficulty": row.get("difficulty"),
            "answer": answer,
            "question_accuracy": row.get("question_accuracy"),
            "accuracy_correct_count": row.get("accuracy_correct_count"),
            "accuracy_num_responses": row.get("accuracy_num_responses"),
            "accuracy_index": row.get("accuracy_index"),
            "strict_subset": "strict_difficulty_score_5_7",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build direct-solution prompts for the strict-difficulty 5/6/7 subset."
    )
    parser.add_argument("--input", default=INPUT_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    input_rows = iter_jsonl(Path(args.input))
    output_rows = [build_record(row, idx) for idx, row in enumerate(input_rows)]
    write_jsonl(Path(args.output), output_rows)
    print(f"[DONE] generated {len(output_rows)} solution prompt records -> {args.output}")


if __name__ == "__main__":
    main()
