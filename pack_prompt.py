#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-case prompt files in a *cache-optimal* request order (DeepSeek Context Caching friendly).

Input:  stage1 gen jsonl (same schema judge.py consumes), each line like:
  {
    "ref_steps": [...],
    "gen_output": [...],
    ... optional fields ...
  }

Output:
  <out_dir>/case_<case_id>.cache.jsonl     # one file per case, requests already ordered for cache
  <out_dir>/ALL.cache.jsonl                # (optional) concatenation of all case requests in cache-optimal order

Cache-optimal ordering strategy (prefix cache friendly):
    1) Within each case:
         - Emit ALL pairwise requests first, grouped by idx (step index), then by ref_idx.
         - Then emit all holistic requests, grouped by idx.
         - Then emit all selfjudge requests, grouped by idx.
  2) Across cases: keep input order (you can optionally sort; see --sort_cases).

Why this helps:
  - Pairwise prompts share the same long system prompt and a growing GLOBAL_PREFIX (REF prefix),
    so sending them in monotone-prefix order maximizes cache hits.
  - Holistic has a different system prompt, so it is best separated from pairwise.

NOTE: This script only writes prompts. To exploit cache fully at runtime, execute requests
      sequentially (or low concurrency) in the order written.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from data_process import Processor, _normalize_generation_input
from prompt import Pairwise_Prompt, Holistic_Prompt, SelfJudge_Prompt


class _NullModel:
    """Just enough to satisfy prompt classes' __init__ signatures. No generation happens here."""
    model_name = "deepseek-reasoner"


processor = Processor()


def _safe_case_id(rec: Dict[str, Any], fallback_i: int) -> str:
    for k in ("case_id", "id", "uid", "qid", "uuid"):
        v = rec.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return f"{fallback_i:06d}"


def _build_gen_prefix(gen_output_item: str) -> str:
    """Exactly match judge.py's prefix extraction behavior."""
    current_output = _normalize_generation_input(gen_output_item)
    gen_sents_all = processor.sentence_split_en(current_output)
    # judge.py: K = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
    K = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
    gen_sents = gen_sents_all[:K]
    return " ".join(gen_sents).strip()


def _iter_case_requests_cache_optimal(
    rec: Dict[str, Any],
    pairwise: Pairwise_Prompt,
    holistic: Holistic_Prompt,
    selfjudge: SelfJudge_Prompt,
) -> List[Dict[str, Any]]:
    """
    Return ordered request objects for a single case.
    Each entry:
            {
                "request_id": str,
                "route": "pairwise"|"holistic"|"selfjudge",
        "idx": int,
        "ref_idx": int|None,
        "prompt": {"messages":[...]},
        "schema": { ...json schema... },
        "meta": {...}
      }
    """
    ref_steps: List[str] = rec["ref_steps"]
    gen_output: List[str] = rec["gen_output"]

    requests: List[Dict[str, Any]] = []

    # --- Pairwise first (maximizes reuse of pairwise system prompt + growing prefix)
    for idx in range(len(gen_output)):
        gen_prefix = _build_gen_prefix(gen_output[idx])
        if not gen_prefix:
            continue

        ref_slice = ref_steps[: idx + 1]
        prior_ref = "\n".join(ref_slice) if ref_slice else ""

        # Build each REF_STEP prompt in increasing order (ref_idx)
        for ref_idx, ref_step in enumerate(ref_slice):
            pairwise.build_user(gen_prefix, ref_step, prefix=prior_ref)
            p = pairwise.return_prompt()
            requests.append(
                {
                    "request_id": f"pairwise_i{idx:04d}_r{ref_idx:04d}",
                    "route": "pairwise",
                    "idx": idx,
                    "ref_idx": ref_idx,
                    "prompt": p,
                    "schema": pairwise.output_schema,
                    "meta": {
                        "gen_prefix_len_chars": len(gen_prefix),
                        "prior_ref_len_chars": len(prior_ref),
                    },
                }
            )

    # --- Then Holistic (different system prompt; keep grouped so cache isn't thrashed)
    for idx in range(len(gen_output)):
        gen_prefix = _build_gen_prefix(gen_output[idx])
        if not gen_prefix:
            continue

        prior_ref = "\n".join(ref_steps[: idx + 1]) if ref_steps else ""
        holistic.build_user(gen_prefix, prior_ref)
        p = holistic.return_prompt()
        requests.append(
            {
                "request_id": f"holistic_i{idx:04d}",
                "route": "holistic",
                "idx": idx,
                "ref_idx": None,
                "prompt": p,
                "schema": holistic.output_schema,
                "meta": {
                    "gen_prefix_len_chars": len(gen_prefix),
                    "prior_ref_len_chars": len(prior_ref),
                },
            }
        )

    # --- Then SelfJudge (reference-free, distinct system prompt)
    for idx in range(len(gen_output)):
        gen_prefix = _build_gen_prefix(gen_output[idx])
        if not gen_prefix:
            continue

        selfjudge.build_user(gen_prefix)
        p = selfjudge.return_prompt()
        requests.append(
            {
                "request_id": f"selfjudge_i{idx:04d}",
                "route": "selfjudge",
                "idx": idx,
                "ref_idx": None,
                "prompt": p,
                "schema": selfjudge.output_schema,
                "meta": {
                    "gen_prefix_len_chars": len(gen_prefix),
                },
            }
        )

    return requests


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_file", type=str, required=True, help="stage1 生成的 gen_only.jsonl 路径（judge.py 同款输入）")
    ap.add_argument("--out_dir", type=str, default=None, help="输出目录（默认: <gen_file_dir>/cache_prompts）")
    ap.add_argument("--max_cases", type=int, default=None, help="只处理前 N 个 case（调试用）")
    ap.add_argument("--write_all", action="store_true", help="额外写一个 ALL_cache.jsonl（把所有 case 的请求拼起来）")
    args = ap.parse_args()

    gen_file = os.path.abspath(args.gen_file)
    out_dir = os.path.join(args.out_dir, "cache_prompts") if args.out_dir else os.path.join(os.path.dirname(gen_file), "cache_prompts")
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate prompt builders (no actual generation in this script)
    dummy = _NullModel()
    pairwise = Pairwise_Prompt(dummy)
    holistic = Holistic_Prompt(dummy)
    selfjudge = SelfJudge_Prompt(dummy)

    cases: List[Tuple[str, Dict[str, Any]]] = []
    with open(gen_file, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            rec = json.loads(line)
            case_id = _safe_case_id(rec, i)
            cases.append((case_id, rec))
            if args.max_cases is not None and len(cases) >= args.max_cases:
                break


    all_rows: List[Dict[str, Any]] = []

    for i, (case_id, rec) in enumerate(cases):
        reqs = _iter_case_requests_cache_optimal(rec, pairwise, holistic, selfjudge)

        # per-case file
        case_path = os.path.join(out_dir, f"case_{case_id}_cache.jsonl")
        _write_jsonl(case_path, reqs)

        if args.write_all:
            # add case_id into each row for the global file
            for r in reqs:
                r2 = dict(r)
                r2["case_id"] = case_id
                all_rows.append(r2)

        if (i + 1) % 50 == 0:
            print(f"[INFO] wrote {i+1} cases...")

    if args.write_all:
        all_path = os.path.join(out_dir, "ALL_cache.jsonl")
        _write_jsonl(all_path, all_rows)

    print(f"[DONE] wrote {len(cases)} case files into: {out_dir}")
    if args.write_all:
        print(f"[DONE] wrote concatenated file: {os.path.join(out_dir, 'ALL_cache.jsonl')}")


if __name__ == "__main__":
    main()
