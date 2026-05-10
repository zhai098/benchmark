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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

from benchmark_core.config import Config
from benchmark_core.data_process import Processor, _normalize_generation_input
from benchmark_core.log_reference import claims_for_step, dependency_claims_for_step, step_id_at_index
from benchmark_core.prompt import Holistic_Prompt, Pairwise_Prompt, SelfJudge_Prompt


class _NullModel:
    model_name = "deepseek-reasoner"


processor = Processor()


DEFAULT_GENPREFIX_TOKENIZER = "/data/pretrain/Qwen/Qwen3-32B"
DEFAULT_MAX_GENPREFIX_TOKENS = 900


@dataclass(frozen=True)
class StepWindow:
    """One generated prefix aligned to one reference-step position."""

    idx: int
    step_id: str
    gen_prefix: str
    prior_ref: str


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


def _load_tokenizer(tokenizer_name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers 未安装，无法按 token 截断 gen_prefix。"
            "请在包含 transformers 的环境中运行（例如 conda run -n vllm-qwen36 ...）。"
        ) from exc

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )


def _truncate_by_tokens(text: str, *, tokenizer: Any, max_tokens: int) -> str:
    if not text:
        return ""
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        return ""

    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)

    if len(ids) <= max_tokens:
        return text

    truncated_ids = ids[:max_tokens]
    # decode back to text to keep downstream prompt templates unchanged
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def _iter_scored_prefixes(
    rec: Dict[str, Any],
    *,
    exclude_last_positions: int = 2,
) -> List[Tuple[int, str]]:
    """Select evaluated generated-prefix positions and prefer stored prefixes."""
    gen_output: List[str] = rec.get("gen_output") or []
    stored_prefixes: List[str] = rec.get("gen_prefix") or []
    excluded = max(0, int(exclude_last_positions))
    upper = max(0, len(stored_prefixes) - excluded) if stored_prefixes else max(0, len(gen_output) - excluded)
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


def _scored_windows(rec: Dict[str, Any], *, exclude_last_positions: int) -> List[StepWindow]:
    windows: List[StepWindow] = []
    tokenizer = rec.get("__genprefix_tokenizer")
    max_tokens = rec.get("__max_genprefix_tokens")
    for idx, gen_prefix in _iter_scored_prefixes(
        rec,
        exclude_last_positions=exclude_last_positions,
    ):
        if tokenizer is not None and max_tokens is not None:
            gen_prefix = _truncate_by_tokens(gen_prefix, tokenizer=tokenizer, max_tokens=int(max_tokens))
        windows.append(
            StepWindow(
                idx=idx,
                step_id=step_id_at_index(rec, idx) or f"s{idx + 1}",
                gen_prefix=gen_prefix,
                prior_ref=_prior_reference_text(rec, idx),
            )
        )
    return windows


def _request(
    *,
    request_id: str,
    route: str,
    window: StepWindow,
    prompt_builder: Any,
    ref_idx: int | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "route": route,
        "idx": window.idx,
        "ref_idx": ref_idx,
        "prompt": prompt_builder.return_prompt(),
        "schema": prompt_builder.output_schema,
        "meta": meta or {},
    }


def _pairwise_requests(
    rec: Dict[str, Any],
    window: StepWindow,
    pairwise: Pairwise_Prompt,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for dep_idx, claim in enumerate(dependency_claims_for_step(rec, window.idx)):
        pairwise.build_user(window.gen_prefix, claim["text"], prefix=window.prior_ref)
        requests.append(
            _request(
                request_id=f"pairwise_i{window.idx:04d}_d{dep_idx:04d}",
                route="pairwise",
                window=window,
                prompt_builder=pairwise,
                ref_idx=dep_idx,
                meta={
                    "current_step_id": window.step_id,
                    "dependency_claim_id": claim["id"],
                    "dependency_claim_text": claim["text"],
                    "prior_ref_len_chars": len(window.prior_ref),
                    "gen_prefix_len_chars": len(window.gen_prefix),
                },
            )
        )
    return requests


def _holistic_request(
    _rec: Dict[str, Any],
    window: StepWindow,
    holistic: Holistic_Prompt,
) -> Dict[str, Any]:
    holistic.build_user(window.gen_prefix, window.prior_ref)
    return _request(
        request_id=f"holistic_i{window.idx:04d}",
        route="holistic",
        window=window,
        prompt_builder=holistic,
        meta={
            "step_id": window.step_id,
            "prior_ref_len_chars": len(window.prior_ref),
            "gen_prefix_len_chars": len(window.gen_prefix),
        },
    )


def _selfjudge_without_reference_request(
    _rec: Dict[str, Any],
    window: StepWindow,
    selfjudge: SelfJudge_Prompt,
) -> Dict[str, Any]:
    selfjudge.build_user_without_reference(window.gen_prefix)
    return _request(
        request_id=f"selfjudge_without_reference_i{window.idx:04d}",
        route="selfjudge_without_reference",
        window=window,
        prompt_builder=selfjudge,
        meta={"current_step_id": window.step_id, "gen_prefix_len_chars": len(window.gen_prefix)},
    )


def _selfjudge_with_reference_requests(
    rec: Dict[str, Any],
    window: StepWindow,
    selfjudge: SelfJudge_Prompt,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for claim_idx, claim in enumerate(claims_for_step(rec, window.idx)):
        selfjudge.build_user_with_reference(
            window.gen_prefix,
            claim["text"],
            step_label=window.step_id,
        )
        requests.append(
            _request(
                request_id=f"selfjudge_with_reference_i{window.idx:04d}_c{claim_idx:04d}",
                route="selfjudge_with_reference",
                window=window,
                prompt_builder=selfjudge,
                ref_idx=claim_idx,
                meta={
                    "current_step_id": window.step_id,
                    "current_step_claim_id": claim["id"],
                    "current_step_claim_text": claim["text"],
                    "gen_prefix_len_chars": len(window.gen_prefix),
                },
            )
        )
    return requests


def _extend_requests(
    output: List[Dict[str, Any]],
    builder: Callable[[StepWindow], Dict[str, Any] | Iterable[Dict[str, Any]]],
    windows: List[StepWindow],
) -> None:
    for window in windows:
        built = builder(window)
        if isinstance(built, dict):
            output.append(built)
        else:
            output.extend(built)


def _case_requests(
    rec: Dict[str, Any],
    pairwise: Pairwise_Prompt,
    holistic: Holistic_Prompt,
    selfjudge: SelfJudge_Prompt,
    *,
    exclude_last_positions: int,
) -> List[Dict[str, Any]]:
    windows = _scored_windows(rec, exclude_last_positions=exclude_last_positions)
    requests: List[Dict[str, Any]] = []
    # Keep the old output order: all pairwise, then holistic, then self-judge.
    _extend_requests(requests, lambda w: _pairwise_requests(rec, w, pairwise), windows)
    _extend_requests(requests, lambda w: _holistic_request(rec, w, holistic), windows)
    _extend_requests(requests, lambda w: _selfjudge_without_reference_request(rec, w, selfjudge), windows)
    _extend_requests(requests, lambda w: _selfjudge_with_reference_requests(rec, w, selfjudge), windows)
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
    parser.add_argument(
        "--exclude_last_positions",
        type=int,
        default=2,
        help="Exclude this many final generated-prefix positions from judge-prompt construction",
    )
    parser.add_argument(
        "--genprefix_tokenizer",
        type=str,
        default=DEFAULT_GENPREFIX_TOKENIZER,
        help="固定 tokenizer（HF name/path），用于对 gen_prefix 做 token 截断",
    )
    parser.add_argument(
        "--max_genprefix_tokens",
        type=int,
        default=DEFAULT_MAX_GENPREFIX_TOKENS,
        help="若 gen_prefix token 数超过该值，则截断到该 token 数（默认 900）",
    )
    parser.add_argument("--write_all", action="store_true", help="Also write a concatenated ALL_cache.jsonl")
    args = parser.parse_args()

    genprefix_tokenizer = _load_tokenizer(args.genprefix_tokenizer)
    max_genprefix_tokens = int(args.max_genprefix_tokens)

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
            # pass truncation settings to downstream helpers without changing their public signatures
            record["__genprefix_tokenizer"] = genprefix_tokenizer
            record["__max_genprefix_tokens"] = max_genprefix_tokens
            case_id = _safe_case_id(record, i)
            cases.append((case_id, record))
            if args.max_cases is not None and len(cases) >= args.max_cases:
                break

    all_rows: List[Dict[str, Any]] = []
    manifest_cases: List[Dict[str, Any]] = []
    for idx, (case_id, record) in enumerate(cases, start=1):
        requests = _case_requests(
            record,
            pairwise,
            holistic,
            selfjudge,
            exclude_last_positions=args.exclude_last_positions,
        )
        case_path = os.path.join(out_dir, f"case_{case_id}_cache.jsonl")
        _write_jsonl(case_path, requests)
        scored_indices = sorted({int(row["idx"]) for row in requests if "idx" in row})
        manifest_cases.append(
            {
                "case_id": case_id,
                "request_file": os.path.basename(case_path),
                "num_requests": len(requests),
                "scored_indices": scored_indices,
                "num_scored_positions": len(scored_indices),
            }
        )

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

    manifest_path = os.path.join(out_dir, "pack_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "gen_file": gen_file,
                "out_dir": out_dir,
                "num_cases": len(cases),
                "exclude_last_positions": int(args.exclude_last_positions),
                "max_prefix_num": int(Config["max prefix_num"]),
                "write_all": bool(args.write_all),
                "cases": manifest_cases,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] wrote {len(cases)} case files into: {out_dir}")
    print(f"[DONE] wrote manifest: {manifest_path}")
    if args.write_all:
        print(f"[DONE] wrote concatenated file: {os.path.join(out_dir, 'ALL_cache.jsonl')}")


if __name__ == "__main__":
    main()
