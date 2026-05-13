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

from benchmark_core.prompt import Claim_Segment_Prompt
from runner import DEEPSEEK_API_runner


def build_judge_model():
    """Build the model used by Claim_Segment_Prompt."""
    return DEEPSEEK_API_runner()



def _safe_case_filename(case_id: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in str(case_id)).strip().replace(" ", "_")


def _iter_input_records(input_path: Path):
    """Iterate JSON/JSONL records from input path."""
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    yield rec
        elif isinstance(obj, dict):
            yield obj
        return

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _extract_step_texts(rec: Dict[str, Any], step_field: str) -> List[str]:
    """Extract list[str] steps from record by field name."""
    raw = rec.get(step_field)
    if not isinstance(raw, list):
        return []

    texts: List[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("content", "")).strip()
        else:
            text = ""
        texts.append(text)
    return texts


def _payloads_from_generate_output(out: Any) -> List[Any]:
    """Normalize runner.generate output to a list payload for batch processing."""
    payload = out
    if isinstance(out, tuple) and len(out) >= 2:
        payload = out[1]
    if isinstance(payload, list):
        return payload
    return [payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment each case step into claims via Claim_Segment_Prompt")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file path")
    parser.add_argument(
        "--step-field",
        default="ref_steps",
        help="Field containing per-step solution list (default: ref_steps)",
    )
    parser.add_argument(
        "--out-dir",
        default="claims",
        help="Output folder for per-case claim JSON files (default: claims)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional max number of cases to process",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_judge_model()
    claim_segmenter = Claim_Segment_Prompt(model)

    processed = 0
    for idx, rec in enumerate(_iter_input_records(input_path), start=1):
        if args.max_cases is not None and processed >= args.max_cases:
            break

        case_id = rec.get("id", idx)
        safe_id = _safe_case_filename(case_id) or f"case_{idx}"

        step_texts = _extract_step_texts(rec, args.step_field)
        step_claims: List[Dict[str, Any]] = []

        prompt_tasks: List[tuple[int, str, Any]] = []

        for step_idx, step_text in enumerate(step_texts, start=1):
            if not step_text:
                step_claims.append(
                    {
                        "step_index": step_idx,
                        "step_text": step_text,
                        "claims": [],
                        "error": "empty_step",
                    }
                )
                continue

            claim_segmenter.build_user(step_text)
            prompt_tasks.append((step_idx, step_text, claim_segmenter.return_prompt()))

        batched_payloads: List[Any] = []
        if prompt_tasks:
            prompts = [item[2] for item in prompt_tasks]
            batched_out = model.generate(prompts, claim_segmenter.output_schema)
            
            batched_payloads = _payloads_from_generate_output(batched_out)

        for i, (step_idx, step_text, _) in enumerate(prompt_tasks):
            try:
                payload = batched_payloads[i] if i < len(batched_payloads) else "{}"
                if isinstance(payload, dict):
                    result = payload
                else:
                    result = json.loads(payload)

                claims = result.get("segments", []) if isinstance(result, dict) else []
                if not isinstance(claims, list):
                    claims = []
                    print(f"[WARN] claims is not a list for case {case_id} step {step_idx}")
                step_claims.append(
                    {
                        "step_index": step_idx,
                        "step_text": step_text,
                        "claims": claims,
                    }
                )
            except Exception as e:
                step_claims.append(
                    {
                        "step_index": step_idx,
                        "step_text": step_text,
                        "claims": [],
                        "error": str(e),
                    }
                )

        case_output = {
            "id": case_id,
            "difficulty": rec.get("difficulty"),
            "problem": rec.get("problem"),
            "answer": rec.get("answer"),
            "step_field": args.step_field,
            "num_steps": len(step_texts),
            "steps": step_claims,
        }

        case_path = out_dir / f"{safe_id}.json"
        with case_path.open("w", encoding="utf-8") as f:
            json.dump(case_output, f, ensure_ascii=False, indent=2)

        processed += 1
        print(f"[INFO] wrote claims for case {case_id} -> {case_path}")

    print(f"[DONE] processed {processed} cases, output dir: {out_dir}")


if __name__ == "__main__":
    main()
