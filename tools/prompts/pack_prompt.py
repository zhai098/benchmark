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
import os
import re
from typing import Any, Dict, List, Tuple

from benchmark_core.config import Config
from benchmark_core.data_process import Processor, _normalize_generation_input
from benchmark_core.log_reference import claims_for_step, dependency_claims_for_step, step_id_at_index
from benchmark_core.prompt import Holistic_Prompt, Pairwise_Prompt, SelfJudge_Prompt


class _NullModel:
    model_name = "deepseek-reasoner"


processor = Processor()


def _safe_case_id(rec: Dict[str, Any], fallback_i: int) -> str:
    for key in ("annotation_uid", "id", "uid", "qid", "uuid", "case_id"):
        value = rec.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return f"{fallback_i:06d}"


def _build_gen_prefix(gen_output_item: str) -> str:
    current_output = _normalize_generation_input(gen_output_item)
    gen_sents_all = processor.sentence_split_en(current_output)
    k = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
    return " ".join(gen_sents_all[:k]).strip()


def _iter_scored_prefixes(rec: Dict[str, Any]) -> List[Tuple[int, str]]:
    """Mirror judge.py's scoring window and prefer stored generation prefixes."""
    gen_output: List[str] = rec.get("gen_output") or []
    stored_prefixes: List[str] = rec.get("gen_prefix") or []
    upper = max(0, len(stored_prefixes) - 2) if stored_prefixes else max(0, len(gen_output) - 2)
    upper = min(upper, len(gen_output))

    scored: List[Tuple[int, str]] = []
    for idx in range(upper):
        prefix = ""
        if idx < len(stored_prefixes):
            prefix = str(stored_prefixes[idx] or "").strip()
        if not prefix:
            prefix = _build_gen_prefix(gen_output[idx])
        if prefix:
            scored.append((idx, prefix))
    return scored


def _reference_step_text(record: Dict[str, Any], idx: int) -> str:
    steps = record.get("steps", [])
    if idx < len(steps) and isinstance(steps[idx], dict):
        return str(steps[idx].get("text", "")).strip()
    return ""


def _prior_reference_text(record: Dict[str, Any], idx: int) -> str:
    steps = []
    for prior_idx in range(idx + 1):
        text = _reference_step_text(record, prior_idx)
        if text:
            steps.append(text)
    return "\n".join(steps).strip()


def _pairwise_requests(
    rec: Dict[str, Any],
    idx: int,
    gen_prefix: str,
    pairwise: Pairwise_Prompt,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    prior_ref = _prior_reference_text(rec, idx)
    dependency_claims = dependency_claims_for_step(rec, idx)
    step_label = step_id_at_index(rec, idx) or f"s{idx + 1}"

    for dep_idx, claim in enumerate(dependency_claims):
        pairwise.build_user(gen_prefix, claim["text"], prefix=prior_ref)
        requests.append(
            {
                "request_id": f"pairwise_i{idx:04d}_d{dep_idx:04d}",
                "route": "pairwise",
                "idx": idx,
                "ref_idx": dep_idx,
                "prompt": pairwise.return_prompt(),
                "schema": pairwise.output_schema,
                "meta": {
                    "current_step_id": step_label,
                    "dependency_claim_id": claim["id"],
                    "dependency_claim_text": claim["text"],
                    "prior_ref_len_chars": len(prior_ref),
                    "gen_prefix_len_chars": len(gen_prefix),
                },
            }
        )
    return requests


def _holistic_request(
    rec: Dict[str, Any],
    idx: int,
    gen_prefix: str,
    holistic: Holistic_Prompt,
) -> Dict[str, Any]:
    prior_ref = _prior_reference_text(rec, idx)
    step_label = step_id_at_index(rec, idx) or f"s{idx + 1}"
    holistic.build_user(gen_prefix, prior_ref)
    return {
        "request_id": f"holistic_i{idx:04d}",
        "route": "holistic",
        "idx": idx,
        "ref_idx": None,
        "prompt": holistic.return_prompt(),
        "schema": holistic.output_schema,
        "meta": {
            "step_id": step_label,
            "prior_ref_len_chars": len(prior_ref),
            "gen_prefix_len_chars": len(gen_prefix),
        },
    }


def _selfjudge_without_reference_request(
    rec: Dict[str, Any],
    idx: int,
    gen_prefix: str,
    selfjudge: SelfJudge_Prompt,
) -> Dict[str, Any]:
    step_label = step_id_at_index(rec, idx) or f"s{idx + 1}"
    selfjudge.build_user_without_reference(gen_prefix)
    return {
        "request_id": f"selfjudge_without_reference_i{idx:04d}",
        "route": "selfjudge_without_reference",
        "idx": idx,
        "ref_idx": None,
        "prompt": selfjudge.return_prompt(),
        "schema": selfjudge.output_schema,
        "meta": {"current_step_id": step_label, "gen_prefix_len_chars": len(gen_prefix)},
    }


def _selfjudge_with_reference_requests(
    rec: Dict[str, Any],
    idx: int,
    gen_prefix: str,
    selfjudge: SelfJudge_Prompt,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    step_claims = claims_for_step(rec, idx)
    step_label = step_id_at_index(rec, idx) or f"s{idx + 1}"
    for claim_idx, claim in enumerate(step_claims):
        selfjudge.build_user_with_reference(gen_prefix, claim["text"], step_label=step_label)
        requests.append(
            {
                "request_id": f"selfjudge_with_reference_i{idx:04d}_c{claim_idx:04d}",
                "route": "selfjudge_with_reference",
                "idx": idx,
                "ref_idx": claim_idx,
                "prompt": selfjudge.return_prompt(),
                "schema": selfjudge.output_schema,
                "meta": {
                    "current_step_id": step_label,
                    "current_step_claim_id": claim["id"],
                    "current_step_claim_text": claim["text"],
                    "gen_prefix_len_chars": len(gen_prefix),
                },
            }
        )
    return requests


def _iter_case_requests_cache_optimal(
    rec: Dict[str, Any],
    pairwise: Pairwise_Prompt,
    holistic: Holistic_Prompt,
    selfjudge: SelfJudge_Prompt,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    scored_prefixes = _iter_scored_prefixes(rec)

    for idx, gen_prefix in scored_prefixes:
        requests.extend(_pairwise_requests(rec, idx, gen_prefix, pairwise))

    for idx, gen_prefix in scored_prefixes:
        requests.append(_holistic_request(rec, idx, gen_prefix, holistic))

    for idx, gen_prefix in scored_prefixes:
        requests.append(_selfjudge_without_reference_request(rec, idx, gen_prefix, selfjudge))

    for idx, gen_prefix in scored_prefixes:
        requests.extend(_selfjudge_with_reference_requests(rec, idx, gen_prefix, selfjudge))

    return requests


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True, help="stage1 generation jsonl path")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_cases", type=int, default=None, help="Only package the first N cases")
    parser.add_argument("--write_all", action="store_true", help="Also write a concatenated ALL_cache.jsonl")
    args = parser.parse_args()

    gen_file = os.path.abspath(args.gen_file)
    out_dir = (
        os.path.join(args.out_dir, "cache_prompts")
        if args.out_dir
        else os.path.join(os.path.dirname(gen_file), "cache_prompts")
    )
    os.makedirs(out_dir, exist_ok=True)

    dummy = _NullModel()
    pairwise = Pairwise_Prompt(dummy)
    holistic = Holistic_Prompt(dummy)
    selfjudge = SelfJudge_Prompt(dummy)

    cases: List[Tuple[str, Dict[str, Any]]] = []
    with open(gen_file, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = _safe_case_id(record, i)
            cases.append((case_id, record))
            if args.max_cases is not None and len(cases) >= args.max_cases:
                break

    all_rows: List[Dict[str, Any]] = []
    for idx, (case_id, record) in enumerate(cases, start=1):
        requests = _iter_case_requests_cache_optimal(record, pairwise, holistic, selfjudge)
        case_path = os.path.join(out_dir, f"case_{case_id}_cache.jsonl")
        _write_jsonl(case_path, requests)

        if args.write_all:
            for row in requests:
                merged = dict(row)
                merged["case_id"] = case_id
                all_rows.append(merged)

        if idx % 50 == 0:
            print(f"[INFO] wrote {idx} cases...")

    if args.write_all:
        all_path = os.path.join(out_dir, "ALL_cache.jsonl")
        _write_jsonl(all_path, all_rows)

    print(f"[DONE] wrote {len(cases)} case files into: {out_dir}")
    if args.write_all:
        print(f"[DONE] wrote concatenated file: {os.path.join(out_dir, 'ALL_cache.jsonl')}")


if __name__ == "__main__":
    main()
