#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 judge runner (ORDERED) that reads *prebuilt prompt files* and sends requests
STRICTLY in file order for scheduling.

Changes vs earlier "parallel-grouped-by-route" script:
- No grouping by route/idx for scheduling (we only use route/idx for scoring aggregation)
- No "prime_first" and no per-first-request generate_one
- Still uses parallelism via your DEEPSEEK_API_runner.generate(), but only in
  **contiguous batches with identical schema** (because your runner.generate
  accepts one extra_params/schema per batch).
- Preserves request order end-to-end: requests are *batched in the same order*
  they appear in the prompt file.

Important note (DeepSeek JSON Output):
If schema is used, your prompt MUST include the word "json" and an example format,
otherwise the API may appear "stuck" due to whitespace generation. (See DeepSeek docs.)
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from runner import DEEPSEEK_API_runner, DOUBAO_deepseek_API_runner


def _safe_json_extract(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    blocks = re.findall(r"\{[\s\S]*\}", s)
    for cand in reversed(blocks):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _pairwise_aggregate(scores: List[float]) -> float:
    """mean(lowest 2)"""
    if not scores:
        return 0.0
    s = sorted(float(x) for x in scores)
    k = min(2, len(s))
    return float(_mean(s[:k]))


def _safe_case_filename(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(case_id))


def _load_gen_meta(gen_file: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not gen_file:
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    with open(gen_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("id", "")).strip()
            if cid:
                meta[cid] = rec
    return meta


def _parse_case_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("case_") and base.endswith(".cache.jsonl"):
        return base[len("case_"):-len(".cache.jsonl")]
    if base.startswith("case_") and base.endswith("_cache.jsonl"):
        return base[len("case_"):-len("_cache.jsonl")]
    return os.path.splitext(os.path.splitext(base)[0])[0]


def _iter_case_files(prompt_dir: str) -> List[str]:
    # support both:
    # - case_<id>.cache.jsonl
    # - case_<id>_cache.jsonl
    a = sorted(glob.glob(os.path.join(prompt_dir, "case_*.cache.jsonl")))
    b = sorted(glob.glob(os.path.join(prompt_dir, "case_*_cache.jsonl")))
    seen = set()
    out = []
    for p in a + b:
        if p not in seen:
            seen.add(p)
            out.append(p)

    def _case_sort_key(path: str) -> tuple:
        cid = _parse_case_id_from_filename(path)
        if str(cid).isdigit():
            return (0, int(cid))
        return (1, str(cid))

    return sorted(out, key=_case_sort_key)


def _load_requests(path: str) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reqs.append(json.loads(line))
    return reqs


def _schema_key(schema: Any) -> str:
    """Stable key for batching by identical schema."""
    if schema is None:
        return "NONE"
    try:
        return "JSON:" + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    except Exception:
        # fallback
        return f"OBJ:{type(schema).__name__}:{repr(schema)[:200]}"


def _flush_batch(
    runner: DEEPSEEK_API_runner,
    batch: List[Dict[str, Any]],
    *,
    max_workers: int,
    f_req,
    # accumulators
    pair_scores_by_idx: Dict[int, List[float]],
    hol_score_by_idx: Dict[int, float],
    self_score_by_idx: Dict[int, float],
    pair_raw_by_idx: Dict[int, List[str]],
    pair_reason_by_idx: Dict[int, List[str]],
    hol_raw_by_idx: Dict[int, List[str]],
    hol_reason_by_idx: Dict[int, List[str]],
    self_raw_by_idx: Dict[int, List[str]],
    self_reason_by_idx: Dict[int, List[str]],
    case_id: str,
):
    if not batch:
        return 0

    prompts = [r.get("prompt", "") for r in batch]
    schema = batch[0].get("schema", None)

    # Use your runner's parallel generate
    print(f"[INFO] generating batch of {len(batch)} requests (max_workers={max_workers})...")
    reasonings, contents = runner.generate(prompts, extra_params=schema, max_workers=max_workers)

    n_ok = 0
    for r, reasoning, raw in zip(batch, reasonings, contents):
        rid = r.get("request_id", "")
        route = r.get("route")
        idx = int(r.get("idx", -1))
        ref_idx = r.get("ref_idx", None)
        
        score = None
        parse_err = None
        print(f"[DEBUG] Raw output for request_id={rid}:\n{raw}\n---")
        if schema is not None:
            if isinstance(raw, str) and raw.startswith("<Error:"):
                parse_err = "runner_error"
            elif raw.startswith("<Error:"):
                parse_err = "error"
            else:
                j = _safe_json_extract(raw)
                if j is None:
                    parse_err = "json_parse_failed"
                elif "score" not in j:
                    parse_err = "missing_score"
                else:
                    try:
                        score = float(j["score"])
                    except Exception:
                        parse_err = "score_not_number"

        # Collect outputs (always keep raw/reasoning)
        if route == "pairwise":
            pair_raw_by_idx.setdefault(idx, []).append(raw or "")
            pair_reason_by_idx.setdefault(idx, []).append(reasoning or "")
        elif route == "holistic":
            hol_raw_by_idx.setdefault(idx, []).append(raw or "")
            hol_reason_by_idx.setdefault(idx, []).append(reasoning or "")
        elif route == "selfjudge":
            self_raw_by_idx.setdefault(idx, []).append(raw or "")
            self_reason_by_idx.setdefault(idx, []).append(reasoning or "")

        # Aggregate scores
        if score is not None:
            n_ok += 1
            if route == "pairwise":
                pair_scores_by_idx.setdefault(idx, []).append(score)
            elif route == "holistic":
                hol_score_by_idx[idx] = score
            elif route == "selfjudge":
                self_score_by_idx[idx] = score

        f_req.write(json.dumps({
            "case_id": case_id,
            "request_id": rid,
            "route": route,
            "idx": idx,
            "ref_idx": ref_idx,
            "parsed_score": score,
            "parse_error": parse_err,
            "mode": f"batch(max_workers={max_workers})",
        }, ensure_ascii=False) + "\n")

    return len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_dir", type=str, required=True)
    ap.add_argument("--gen_file", type=str, default=None)
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)

    ap.add_argument("--max_workers", type=int, default=32, help="传给 runner.generate 的并行线程数")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="每个 batch 最多多少条请求（仍保持文件顺序；batch 内并行）")

    ap.add_argument("--max_tokens", type=int, default=0,
                    help="覆盖 runner.default_params['max_tokens']（0 表示不改，推荐 256~1024）")
    args = ap.parse_args()

    prompt_dir = os.path.abspath(args.prompt_dir)
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.path.join(prompt_dir, "run_from_cache_ordered")
    os.makedirs(run_dir, exist_ok=True)
    per_case_dir = os.path.join(run_dir, "cases")
    os.makedirs(per_case_dir, exist_ok=True)

    out_summary = os.path.join(run_dir, "summary.json")
    out_cases = os.path.join(run_dir, "case_results.jsonl")
    out_req_log = os.path.join(run_dir, "requests_log.jsonl")

    gen_meta = _load_gen_meta(os.path.abspath(args.gen_file)) if args.gen_file else {}

    runner = DEEPSEEK_API_runner()
    if args.max_tokens and args.max_tokens > 0:
        runner.default_params["max_tokens"] = int(args.max_tokens)

    case_files = _iter_case_files(prompt_dir)
    if not case_files:
        raise RuntimeError(f"No case_*.cache.jsonl found under {prompt_dir}")

    scores: List[float] = []
    total_req = 0
    t0_all = time.time()

    with open(out_cases, "w", encoding="utf-8", buffering=1) as f_cases, \
         open(out_req_log, "w", encoding="utf-8", buffering=1) as f_req:

        for ci, cf in enumerate(case_files):
            if args.max_cases is not None and ci >= args.max_cases:
                break

            case_id = _parse_case_id_from_filename(cf)
            safe_id = _safe_case_filename(case_id)
            reqs = _load_requests(cf)

            # Aggregation storage
            pair_scores_by_idx: Dict[int, List[float]] = {}
            hol_score_by_idx: Dict[int, float] = {}
            self_score_by_idx: Dict[int, float] = {}
            pair_raw_by_idx: Dict[int, List[str]] = {}
            pair_reason_by_idx: Dict[int, List[str]] = {}
            hol_raw_by_idx: Dict[int, List[str]] = {}
            hol_reason_by_idx: Dict[int, List[str]] = {}
            self_raw_by_idx: Dict[int, List[str]] = {}
            self_reason_by_idx: Dict[int, List[str]] = {}

            meta_rec = gen_meta.get(case_id)
            if meta_rec and isinstance(meta_rec.get("ref_steps"), list):
                N = len(meta_rec["ref_steps"])
            else:
                idxs = [int(r.get("idx", -1)) for r in reqs if str(r.get("idx", "")).lstrip("-").isdigit()]
                N = (max(idxs) + 1) if idxs else 1

            t_case0 = time.time()

            # ---- scheduling: STRICT FILE ORDER ----
            batch: List[Dict[str, Any]] = []
            cur_key: Optional[str] = None

            for r in reqs:
                k = _schema_key(r.get("schema", None))
                if cur_key is None:
                    cur_key = k

                # Flush when schema changes or batch reaches size
                if k != cur_key or (args.batch_size and len(batch) >= args.batch_size):
                    total_req += _flush_batch(
                        runner, batch,
                        max_workers=args.max_workers,
                        f_req=f_req,
                        pair_scores_by_idx=pair_scores_by_idx,
                        hol_score_by_idx=hol_score_by_idx,
                        self_score_by_idx=self_score_by_idx,
                        pair_raw_by_idx=pair_raw_by_idx,
                        pair_reason_by_idx=pair_reason_by_idx,
                        hol_raw_by_idx=hol_raw_by_idx,
                        hol_reason_by_idx=hol_reason_by_idx,
                        self_raw_by_idx=self_raw_by_idx,
                        self_reason_by_idx=self_reason_by_idx,
                        case_id=case_id,
                    )
                    batch = []
                    cur_key = k
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                batch.append(r)

            # Flush remaining 
            if batch:
                total_req += _flush_batch(
                    runner, batch,
                    max_workers=args.max_workers,
                    f_req=f_req,
                    pair_scores_by_idx=pair_scores_by_idx,
                    hol_score_by_idx=hol_score_by_idx,
                    self_score_by_idx=self_score_by_idx,
                    pair_raw_by_idx=pair_raw_by_idx,
                    pair_reason_by_idx=pair_reason_by_idx,
                    hol_raw_by_idx=hol_raw_by_idx,
                    hol_reason_by_idx=hol_reason_by_idx,
                    self_raw_by_idx=self_raw_by_idx,
                    self_reason_by_idx=self_reason_by_idx,
                    case_id=case_id,
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)

            # ---- aggregate per-step ----
            steps = []
            total_score = 0.0
            all_idxs = sorted(set(
                list(pair_scores_by_idx.keys())
                + list(hol_score_by_idx.keys())
                + list(self_score_by_idx.keys())
            ))
            for idx in all_idxs:
                pair_list = pair_scores_by_idx.get(idx, [])
                hol_s = float(hol_score_by_idx.get(idx, 0.0))
                self_s = float(self_score_by_idx.get(idx, 0.0))
                pair_agg = _pairwise_aggregate(pair_list)
                step_score = float((pair_agg + hol_s + self_s) / 3.0)
                contrib = step_score / max(1, N) * 20.0
                total_score += contrib
                steps.append({
                    "index": idx + 1,
                    "score": step_score,
                    "routes": {"pairwise": pair_agg, "holistic": hol_s, "selfjudge": self_s},
                    "pairwise_scores": pair_list,
                    "judge_detail": {
                        "pairwise": {
                            "scores": pair_list,
                            "raw_outputs": pair_raw_by_idx.get(idx, []),
                            "reasoning_outputs": pair_reason_by_idx.get(idx, []),
                        },
                        "holistic": {
                            "score": hol_s,
                            "raw_output": hol_raw_by_idx.get(idx, []),
                            "reasoning_output": hol_reason_by_idx.get(idx, []),
                        },
                        "selfjudge": {
                            "score": self_s,
                            "raw_output": self_raw_by_idx.get(idx, []),
                            "reasoning_output": self_reason_by_idx.get(idx, []),
                        },
                    },
                    "contrib": contrib,
                })

            scores.append(float(total_score))

            case_record = {
                "id": case_id,
                "difficulty": float(meta_rec.get("difficulty", 0.0)) if meta_rec else 0.0,
                "score": float(total_score),
                "num_steps": int(N),
                "steps": steps,
                "problem": meta_rec.get("problem", "") if meta_rec else "",
                "answer": meta_rec.get("answer", "") if meta_rec else "",
                "request_file": cf,
                "max_workers": int(args.max_workers),
                "batch_size": int(args.batch_size),
            }

            with open(os.path.join(per_case_dir, f"{safe_id}.json"), "w", encoding="utf-8") as fcase:
                json.dump(case_record, fcase, ensure_ascii=False, indent=2)

            f_cases.write(json.dumps({
                "id": case_id,
                "score": float(total_score),
                "num_steps": int(N),
                "difficulty": float(meta_rec.get("difficulty", 0.0)) if meta_rec else 0.0,
            }, ensure_ascii=False) + "\n")

            t_case1 = time.time()
            print(f"[INFO] scored case {case_id}: score={total_score:.4f}, steps={len(steps)}, time={t_case1 - t_case0:.2f}s")

    avg = float(sum(scores) / max(1, len(scores)))
    t1_all = time.time()

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "num": int(len(scores)),
            "avg_score": avg,
            "requests": int(total_req),
            "wall_time_s": float(t1_all - t0_all),
            "max_workers": int(args.max_workers),
            "batch_size": int(args.batch_size),
            "max_tokens_override": int(args.max_tokens) if args.max_tokens else None,
        }, f, ensure_ascii=False, indent=2)

    print(f"[RESULT] Processed {len(scores)} cases")
    print(f"[RESULT] Final model score ≈ {avg:.2f}")
    print(f"[OUT] run_dir = {run_dir}")


if __name__ == "__main__":
    main()
