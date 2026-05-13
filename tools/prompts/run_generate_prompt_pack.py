#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_core.config import Config
from benchmark_core.data_process import Processor, _normalize_generation_input


processor = Processor()


CONTINUATION_GUARD = (
    "IMPORTANT CONTINUATION CONTRACT: the final assistant message is an unfinished "
    "solution prefix. Continue it directly. Output only the next mathematical prose "
    "after that exact prefix. Do not restart the solution, do not summarize the "
    "problem, and do not write meta-analysis such as “The user wants me to solve”, "
    "“Let me understand”, “I need to solve”, “I need to find”, or “First, let me parse”."
)

RESTART_OUTPUT_PATTERNS = (
    "the user wants me",
    "the user asks",
    "let me understand",
    "let me parse",
    "let me analyze the problem",
    "first, let me understand",
    "first, let me parse",
    "i need to solve",
    "i will solve",
    "we need solve",
    "we need to solve",
    "the problem asks",
)

STYLE_PREAMBLE_PATTERNS = (
    "i need to find",
    "i need to determine",
    "i need to compute",
    "let me set up",
    "let me compute",
    "let me verify",
    "let me work",
    "let me use",
)


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            yield obj


def _chunks(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _split_generate_response(response: Any) -> Tuple[List[str], List[str]]:
    def as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item or "") for item in value]

    if isinstance(response, tuple) and len(response) == 2:
        reasonings, generations = response
        generations_list = as_list(generations)
        reasonings_list = as_list(reasonings)
        if len(reasonings_list) < len(generations_list):
            reasonings_list.extend([""] * (len(generations_list) - len(reasonings_list)))
        return reasonings_list[: len(generations_list)], generations_list
    generations = as_list(response)
    return [""] * len(generations), generations


def _looks_like_restart_output(value: Any) -> bool:
    text = re.sub(r"\s+", " ", _normalize_generation_input(value)).strip().lower()
    if not text:
        return False
    head = text[:800]
    return any(pattern in head for pattern in RESTART_OUTPUT_PATTERNS)


def _looks_like_style_preamble(value: Any) -> bool:
    text = re.sub(r"\s+", " ", _normalize_generation_input(value)).strip().lower()
    if not text:
        return False
    head = text[:800]
    return any(pattern in head for pattern in STYLE_PREAMBLE_PATTERNS)


def _is_generation_error(value: Any) -> bool:
    text = _normalize_generation_input(value).strip()
    return text.startswith("<Error:") or text.startswith("Error:")


def _with_continuation_guard(prompt: Any) -> Any:
    """Add a strict continuation contract without changing the task semantics."""

    def guard_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        guarded = [dict(message) for message in messages]
        for message in guarded:
            if message.get("role") == "system":
                content = str(message.get("content") or "")
                if CONTINUATION_GUARD not in content:
                    message["content"] = f"{content}\n\n{CONTINUATION_GUARD}".strip()
                return guarded
        return [{"role": "system", "content": CONTINUATION_GUARD}] + guarded

    if isinstance(prompt, dict) and "messages" in prompt:
        updated = dict(prompt)
        updated["messages"] = guard_messages(prompt["messages"])
        return updated
    if (
        isinstance(prompt, list)
        and all(isinstance(message, dict) and "role" in message for message in prompt)
    ):
        return guard_messages(prompt)
    return prompt


def _call_model(model: Any, prompts: Sequence[Any]) -> Any:
    return model.generate(list(prompts), None)


def _generate_with_empty_retries(
    model: Any,
    prompts: Sequence[Any],
    *,
    max_empty_retries: int,
    max_restart_retries: int,
    enforce_continuation_guard: bool,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    active_prompts = [
        _with_continuation_guard(prompt) if enforce_continuation_guard else prompt
        for prompt in prompts
    ]
    restart_retry_counts = [0] * len(active_prompts)
    reasonings, generations = _split_generate_response(_call_model(model, active_prompts))
    if len(generations) < len(prompts):
        generations.extend([""] * (len(prompts) - len(generations)))
    if len(reasonings) < len(prompts):
        reasonings.extend([""] * (len(prompts) - len(reasonings)))

    for attempt in range(1, max_empty_retries + 1):
        empty_indices = [
            idx
            for idx, text in enumerate(generations)
            if not _normalize_generation_input(text).strip() or _is_generation_error(text)
        ]
        if not empty_indices:
            break
        print(f"[WARN] empty outputs at local indices {empty_indices}; retry {attempt}/{max_empty_retries}")
        retry_prompts = [active_prompts[idx] for idx in empty_indices]
        retry_reasonings, retry_generations = _split_generate_response(
            _call_model(model, retry_prompts)
        )
        for local_idx, original_idx in enumerate(empty_indices):
            retry_text = retry_generations[local_idx] if local_idx < len(retry_generations) else ""
            if _normalize_generation_input(retry_text).strip() and not _is_generation_error(retry_text):
                generations[original_idx] = retry_text
                if local_idx < len(retry_reasonings):
                    reasonings[original_idx] = retry_reasonings[local_idx]

    for attempt in range(1, max_restart_retries + 1):
        restart_indices = [
            idx
            for idx, text in enumerate(generations)
            if _looks_like_restart_output(text)
        ]
        if not restart_indices:
            break
        print(
            f"[WARN] restart/meta outputs at local indices {restart_indices}; "
            f"continuation retry {attempt}/{max_restart_retries}"
        )
        retry_prompts = [_with_continuation_guard(active_prompts[idx]) for idx in restart_indices]
        retry_reasonings, retry_generations = _split_generate_response(
            _call_model(model, retry_prompts)
        )
        for local_idx, original_idx in enumerate(restart_indices):
            retry_text = retry_generations[local_idx] if local_idx < len(retry_generations) else ""
            if _normalize_generation_input(retry_text).strip() and not _is_generation_error(retry_text):
                generations[original_idx] = retry_text
                restart_retry_counts[original_idx] += 1
                if local_idx < len(retry_reasonings):
                    reasonings[original_idx] = retry_reasonings[local_idx]

    final_empty = [
        idx
        for idx, text in enumerate(generations)
        if not _normalize_generation_input(text).strip() or _is_generation_error(text)
    ]
    for idx, text in enumerate(generations[: len(prompts)]):
        if _looks_like_restart_output(text):
            print(f"[WARN] output at local index {idx} still looks like a restarted solution.")
    return reasonings[: len(prompts)], generations[: len(prompts)], final_empty, restart_retry_counts[: len(prompts)]


SPECIAL_GENERATION_TOKENS = (
    "<|tool_call_end|>",
    "<|tool_call_start|>",
    "<|tool▁call▁end|>",
    "<|tool▁call▁start|>",
)


def _clean_generation_output(value: Any) -> str:
    text = _normalize_generation_input(value)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "")
    for token in SPECIAL_GENERATION_TOKENS:
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def _build_gen_prefix(output: str) -> str:
    sentences = processor.sentence_split_en(output)
    k = max(1, min(Config["max prefix_num"], len(sentences)))
    return " ".join(sentences[:k]).strip()


def _extract_prompt(row: Dict[str, Any]) -> Any:
    if "prompt" in row:
        return row["prompt"]
    if "messages" in row:
        return row["messages"]
    raise ValueError(f"Prompt row {row.get('prompt_id') or row.get('id')} has no prompt/messages field.")


def _build_runner(args: argparse.Namespace) -> Any:
    from runner import VLLMRunner

    if args.backend != "vllm":
        raise ValueError(f"Generation prompt packs only support local vLLM backend, got: {args.backend}")
    return VLLMRunner(
        model=args.model or Config["reasoning_model"],
        vllm_config=Config["reasoning_model_params"],
        sampling_config=Config["reasoning_sampling_params"],
        gpus=args.gpus or Config["reasoning_model_gpus"],
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _aggregate_gen_only(prompt_rows: List[Dict[str, Any]], output_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    outputs_by_prompt = {row["prompt_id"]: row for row in output_rows}

    for row in prompt_rows:
        case_id = str(row["case_id"])
        item = by_case.get(case_id)
        if item is None:
            item = {
                "id": case_id,
                "case_id": case_id,
                "question": row.get("question") or row.get("problem", ""),
                "problem": row.get("problem") or row.get("question", ""),
                "answer": row.get("answer", ""),
                "standard_solution": row.get("standard_solution", ""),
                "difficulty": (row.get("meta") or {}).get("difficulty"),
                "prompts": [],
                "ref_steps": [],
                "steps": row.get("reference_steps") or [],
                "gen_output": [],
                "gen_prefix": [],
                "reasoning_content": [],
                "empty_generation_indices": [],
                "empty_generation_count": 0,
                "restart_generation_indices": [],
                "restart_generation_count": 0,
                "style_preamble_generation_indices": [],
                "style_preamble_generation_count": 0,
                "restart_retry_counts": [],
                "prompt_pack_schema": "generated_from_prompt_pack",
            }
            by_case[case_id] = item

        output_row = outputs_by_prompt.get(row["prompt_id"])
        generation = output_row.get("generation", "") if output_row else ""
        reasoning = output_row.get("reasoning", "") if output_row else ""
        cleaned = _clean_generation_output(generation)
        item["prompts"].append(row.get("prompt"))
        next_step = row.get("next_reference_step") or {}
        item["ref_steps"].append(next_step.get("text") or "")
        item["gen_output"].append(cleaned)
        item["gen_prefix"].append(_build_gen_prefix(cleaned) if cleaned else "")
        item["reasoning_content"].append(reasoning)
        if not cleaned:
            item["empty_generation_indices"].append(len(item["gen_output"]) - 1)
        if output_row and output_row.get("looks_like_restart"):
            item["restart_generation_indices"].append(len(item["gen_output"]) - 1)
        if output_row and output_row.get("looks_like_style_preamble"):
            item["style_preamble_generation_indices"].append(len(item["gen_output"]) - 1)
        item["restart_retry_counts"].append(int(output_row.get("restart_retry_count") or 0) if output_row else 0)

    for item in by_case.values():
        item["empty_generation_count"] = len(item["empty_generation_indices"])
        item["restart_generation_count"] = len(item["restart_generation_indices"])
        item["style_preamble_generation_count"] = len(item["style_preamble_generation_indices"])
    return list(by_case.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation over a packed generation prompt JSONL.")
    parser.add_argument("--prompt-pack", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-empty-retries", type=int, default=2)
    parser.add_argument(
        "--max-restart-retries",
        type=int,
        default=1,
        help="Retry outputs that look like the model restarted the solution instead of continuing the assistant prefix.",
    )
    parser.add_argument(
        "--enforce-continuation-guard",
        action="store_true",
        help="Inject a strict continuation contract into system messages at call time.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize the pack without calling a model.")
    args = parser.parse_args()

    prompt_pack = Path(args.prompt_pack).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prompt_rows = list(_jsonl_rows(prompt_pack))
    if args.max_prompts is not None:
        prompt_rows = prompt_rows[: args.max_prompts]

    print(f"[INFO] loaded prompts={len(prompt_rows)} from {prompt_pack}")
    if args.dry_run:
        case_count = len({row.get("case_id") for row in prompt_rows})
        formats = sorted({str(row.get("prompt_format")) for row in prompt_rows})
        print(f"[DRY-RUN] cases={case_count}, prompt_formats={formats}")
        return

    model = _build_runner(args)
    output_rows: List[Dict[str, Any]] = []
    start = time.time()
    processed = 0
    for batch in _chunks(prompt_rows, max(1, args.batch_size)):
        prompts = [_extract_prompt(row) for row in batch]
        reasonings, generations, empty_local, restart_retry_counts = _generate_with_empty_retries(
            model,
            prompts,
            max_empty_retries=args.max_empty_retries,
            max_restart_retries=args.max_restart_retries,
            enforce_continuation_guard=args.enforce_continuation_guard,
        )
        for idx, row in enumerate(batch):
            generation = generations[idx] if idx < len(generations) else ""
            output_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "case_id": row["case_id"],
                    "step_index": row["step_index"],
                    "step_id": row["step_id"],
                    "prompt_format": row.get("prompt_format"),
                    "model_name": args.model or row.get("model_name"),
                    "reasoning": reasonings[idx] if idx < len(reasonings) else "",
                    "generation": generation,
                    "is_empty": not _normalize_generation_input(generation).strip(),
                    "is_error": _is_generation_error(generation),
                    "looks_like_restart": _looks_like_restart_output(generation),
                    "looks_like_style_preamble": _looks_like_style_preamble(generation),
                    "restart_retry_count": restart_retry_counts[idx] if idx < len(restart_retry_counts) else 0,
                }
            )
        processed += len(batch)
        restart_local = [
            idx
            for idx, text in enumerate(generations)
            if _looks_like_restart_output(text)
        ]
        print(
            f"[INFO] processed {processed}/{len(prompt_rows)} prompts; "
            f"empty_local={empty_local}; restart_like_local={restart_local}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_outputs_path = out_dir / "prompt_outputs.jsonl"
    gen_only_path = out_dir / "gen_only_from_prompt_pack.jsonl"
    run_info_path = out_dir / "run_info.json"

    _write_jsonl(prompt_outputs_path, output_rows)
    _write_jsonl(gen_only_path, _aggregate_gen_only(prompt_rows, output_rows))
    run_info_path.write_text(
        json.dumps(
            {
                "prompt_pack": str(prompt_pack),
                "out_dir": str(out_dir),
                "backend": args.backend,
                "model": args.model,
                "prompt_count": len(prompt_rows),
                "empty_count": sum(1 for row in output_rows if row["is_empty"]),
                "error_count": sum(1 for row in output_rows if row.get("is_error")),
                "restart_like_count": sum(1 for row in output_rows if row.get("looks_like_restart")),
                "style_preamble_like_count": sum(1 for row in output_rows if row.get("looks_like_style_preamble")),
                "restart_retry_total": sum(int(row.get("restart_retry_count") or 0) for row in output_rows),
                "enforce_continuation_guard": args.enforce_continuation_guard,
                "max_restart_retries": args.max_restart_retries,
                "elapsed_sec": time.time() - start,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] prompt_outputs={prompt_outputs_path}")
    print(f"[DONE] gen_only={gen_only_path}")
    print(f"[DONE] run_info={run_info_path}")


if __name__ == "__main__":
    main()
