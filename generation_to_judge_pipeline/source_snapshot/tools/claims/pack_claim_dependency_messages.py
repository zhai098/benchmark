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
from typing import Any, Dict, List

from benchmark_core.paths import CLAIMS_WORKFLOW_DIR
from benchmark_core.prompt import Claim_Dependency_Prompt


def _sorted_case_files(claims_dir: Path) -> List[Path]:
    files = [p for p in claims_dir.glob("*.json") if p.is_file()]

    def key_func(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(files, key=key_func)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_prior_claims(steps: List[Dict[str, Any]], current_step_index: int) -> List[Dict[str, Any]]:
    priors: List[Dict[str, Any]] = []
    for step in steps:
        step_idx = int(step.get("step_index", 0))
        if step_idx >= current_step_index:
            continue
        for claim in (step.get("claims", []) or []):
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            priors.append(
                {
                    "step_index": step_idx,
                    "claim_id": claim.get("id"),
                    "text": text,
                }
            )
    return priors


def build_all_requests(case_obj: Dict[str, Any], dep_prompt: Claim_Dependency_Prompt) -> List[Dict[str, Any]]:
    """
    按 step 批次打包请求：
    - step 1 不评
    - step t (t>=2): 当前 step 的每条 claim 与全部前序 step claims 逐个比较（A->B）
    - 输出一条请求 = 一行 JSONL
    """
    requests: List[Dict[str, Any]] = []

    case_id = case_obj.get("id")
    steps = case_obj.get("steps", []) or []

    for step in steps:
        step_index = int(step.get("step_index", 0))
        if step_index <= 1:
            continue

        current_claims = step.get("claims", []) or []
        prior_claims = _flatten_prior_claims(steps, step_index)
        if not current_claims or not prior_claims:
            continue

        pair_in_step = 0
        for claim_a in current_claims:
            claim_a_id = claim_a.get("id")
            claim_a_text = str(claim_a.get("text", "")).strip()
            if not claim_a_text:
                continue

            for prior in prior_claims:
                pair_in_step += 1
                dep_prompt.build_user(claim_a_text, prior["text"])
                prompt_obj = dep_prompt.return_prompt()  # {"messages": [...]} 

                requests.append(
                    {
                        "case_id": case_id,
                        "step_index": step_index,
                        "batch_id": f"case_{case_id}_step_{step_index}",
                        "pair_in_step": pair_in_step,
                        "current_claim_id": claim_a_id,
                        "current_claim_text": claim_a_text,
                        "prior_step_index": prior["step_index"],
                        "prior_claim_id": prior["claim_id"],
                        "prior_claim_text": prior["text"],
                        "messages": prompt_obj["messages"],
                    }
                )

    return requests


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Claim_Dependency_Prompt messages from claims/*.json into one JSONL file."
    )
    parser.add_argument("--claims-dir", type=str, default=str(CLAIMS_WORKFLOW_DIR / "claims"), help="Input claims folder")
    parser.add_argument(
        "--output",
        type=str,
        default=str(CLAIMS_WORKFLOW_DIR / "claim_dependency_messages.jsonl"),
        help="Output packed JSONL file",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional max number of case files")
    args = parser.parse_args()

    claims_dir = Path(args.claims_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    case_files = _sorted_case_files(claims_dir)
    if args.max_cases is not None:
        case_files = case_files[: args.max_cases]

    # 只用于构建 prompt，不调用 run()，因此 model 可为 None
    dep_prompt = Claim_Dependency_Prompt(model=None)

    all_rows: List[Dict[str, Any]] = []
    for case_path in case_files:
        case_obj = _read_json(case_path)
        all_rows.extend(build_all_requests(case_obj, dep_prompt))

    write_jsonl(output, all_rows)
    print(f"[DONE] packed {len(all_rows)} requests -> {output}")


if __name__ == "__main__":
    main()
