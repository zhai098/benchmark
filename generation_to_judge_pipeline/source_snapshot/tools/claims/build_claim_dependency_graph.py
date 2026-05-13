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
from typing import Any, Dict, List, Tuple

from benchmark_core.data_process import safe_json_loads
from benchmark_core.prompt import Claim_Dependency_Prompt
from runner import DEEPSEEK_API_runner


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
    """
    收集所有前序 step 的 claims。
    current_step_index 为 1-based step 序号。
    """
    priors: List[Dict[str, Any]] = []
    for step in steps:
        step_idx = int(step.get("step_index", 0))
        if step_idx >= current_step_index:
            continue
        for claim in step.get("claims", []) or []:
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


def _normalize_result(raw_output: Any) -> Dict[str, str]:
    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, list):
        candidate = raw_output[0] if raw_output else ""
        if isinstance(candidate, dict):
            parsed = candidate
        else:
            try:
                parsed = safe_json_loads(str(candidate))
            except Exception:
                parsed = None
    else:
        try:
            parsed = safe_json_loads(str(raw_output))
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        conclusion = str(parsed.get("conclusion", "uncertain")).strip().lower()
        if conclusion not in {"yes", "no", "uncertain"}:
            conclusion = "uncertain"
        explanation = str(parsed.get("explanation", "")).strip()
        return {"conclusion": conclusion, "explanation": explanation}

    return {"conclusion": "uncertain", "explanation": "unable to parse model output"}


def process_case(case_obj: Dict[str, Any], dep_prompt: Claim_Dependency_Prompt, model) -> Dict[str, Any]:
    steps = case_obj.get("steps", []) or []

    output_steps: List[Dict[str, Any]] = []

    for step in steps:
        step_index = int(step.get("step_index", 0))
        step_text = step.get("step_text", "")
        current_claims = step.get("claims", []) or []

        # 第一步不评
        if step_index <= 1:
            out_claims = []
            for claim in current_claims:
                out_claims.append(
                    {
                        "id": claim.get("id"),
                        "text": claim.get("text", ""),
                        "dependency_claims": [],
                        "num_dependency_claims": 0,
                    }
                )

            output_steps.append(
                {
                    "step_index": step_index,
                    "step_text": step_text,
                    "batch_evaluated": False,
                    "reason": "step_1_skipped",
                    "claims": out_claims,
                }
            )
            continue

        prior_claims = _flatten_prior_claims(steps, step_index)

        # 当前 step 与前序都无可比 claim
        if not current_claims or not prior_claims:
            out_claims = []
            for claim in current_claims:
                out_claims.append(
                    {
                        "id": claim.get("id"),
                        "text": claim.get("text", ""),
                        "dependency_claims": [],
                        "num_dependency_claims": 0,
                    }
                )

            output_steps.append(
                {
                    "step_index": step_index,
                    "step_text": step_text,
                    "batch_evaluated": True,
                    "num_pairs": 0,
                    "claims": out_claims,
                }
            )
            continue

        # 该 step 一个批次：构造所有 A->B 对比请求
        pair_meta: List[Tuple[int, int]] = []
        prompts: List[Any] = []

        for current_idx, claim_a in enumerate(current_claims):
            claim_a_text = str(claim_a.get("text", "")).strip()
            if not claim_a_text:
                continue
            for prior_idx, prior in enumerate(prior_claims):
                dep_prompt.build_user(claim_a_text, prior["text"])
                prompts.append(dep_prompt.return_prompt())
                pair_meta.append((current_idx, prior_idx))

        reasonings, raw_outputs = model.generate(prompts, dep_prompt.output_schema) if prompts else ([], [])

        # 初始化每个 claim 的依赖子集
        dependency_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(current_claims))}
        comparisons_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(current_claims))}

        for i, (current_idx, prior_idx) in enumerate(pair_meta):
            raw_text = raw_outputs[i] if i < len(raw_outputs) else ""
            reasoning_text = reasonings[i] if i < len(reasonings) else ""
            result = _normalize_result(raw_text)

            prior = prior_claims[prior_idx]
            comparison = {
                "prior_step_index": prior["step_index"],
                "prior_claim_id": prior["claim_id"],
                "prior_claim_text": prior["text"],
                "conclusion": result["conclusion"],
                "explanation": result["explanation"],
                "raw_output": raw_text,
                "reasoning_output": reasoning_text,
            }
            comparisons_map[current_idx].append(comparison)

            if result["conclusion"] == "yes":
                dependency_map[current_idx].append(
                    {
                        "step_index": prior["step_index"],
                        "claim_id": prior["claim_id"],
                        "text": prior["text"],
                    }
                )

        out_claims = []
        for current_idx, claim_a in enumerate(current_claims):
            deps = dependency_map.get(current_idx, [])
            out_claims.append(
                {
                    "id": claim_a.get("id"),
                    "text": claim_a.get("text", ""),
                    "dependency_claims": deps,
                    "num_dependency_claims": len(deps),
                    "comparisons": comparisons_map.get(current_idx, []),
                }
            )

        output_steps.append(
            {
                "step_index": step_index,
                "step_text": step_text,
                "batch_evaluated": True,
                "num_pairs": len(pair_meta),
                "claims": out_claims,
            }
        )

    return {
        "id": case_obj.get("id"),
        "difficulty": case_obj.get("difficulty"),
        "problem": case_obj.get("problem"),
        "answer": case_obj.get("answer"),
        "num_steps": case_obj.get("num_steps", len(steps)),
        "steps": output_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-claim dependency subsets using Claim_Dependency_Prompt (A->B only)."
    )
    parser.add_argument("--claims-dir", type=str, default="claims", help="Input claims folder")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="claim_dependencies",
        help="Output folder for dependency results",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional max cases")
    args = parser.parse_args()

    claims_dir = Path(args.claims_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DEEPSEEK_API_runner()
    dep_prompt = Claim_Dependency_Prompt(model)

    case_files = _sorted_case_files(claims_dir)
    processed = 0

    for case_path in case_files:
        if args.max_cases is not None and processed >= args.max_cases:
            break

        case_obj = _read_json(case_path)
        result_obj = process_case(case_obj, dep_prompt, model)

        out_path = out_dir / case_path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result_obj, f, ensure_ascii=False, indent=2)

        processed += 1
        print(f"[INFO] processed case file: {case_path.name} -> {out_path}")

    print(f"[DONE] processed {processed} cases. output dir: {out_dir}")


if __name__ == "__main__":
    main()
