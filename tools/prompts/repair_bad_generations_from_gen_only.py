#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_core.config import Config
from benchmark_core.data_process import _normalize_generation_input
from benchmark_core.prompt import Generate_Prompt
from runner import VLLMRunner

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


MOJIBAKE_RE = re.compile(
    r"�|ï¿½|(?:Ã[\x80-\xBF])|(?:Â[\x80-\xBF])|(?:â[\x80-\xBF]{1,2})|[\x00-\x08\x0b\x0c\x0e-\x1f]"
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


DEFAULT_RUNAWAY_MIN_CHARS = 30000
MAX_EMPTY_RETRIES = 2


class DummyModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def excerpt(value: Any, limit: int = 360) -> str:
    text = normalize(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def repeat_metrics(text: str) -> dict[str, Any]:
    chunks = [normalize(part) for part in SENT_SPLIT_RE.split(text) if len(normalize(part)) >= 20]
    counts = Counter(chunks)
    max_sentence_repeat = max(counts.values()) if counts else 0
    repeated_sentence_total = sum(count for count in counts.values() if count >= 3)

    dense = compact(text)
    max_window_repeat = 0
    if len(dense) > 200:
        windows = [dense[i : i + 40] for i in range(0, len(dense) - 39, 10)]
        max_window_repeat = Counter(windows).most_common(1)[0][1]

    max_same_char_run = 0
    current = 0
    previous = None
    for char in text:
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        max_same_char_run = max(max_same_char_run, current)

    large_repeat = (
        max_sentence_repeat >= 4
        or repeated_sentence_total >= 8
        or max_window_repeat >= 8
        or max_same_char_run >= 80
    )
    return {
        "large_repeat": large_repeat,
        "max_sentence_repeat": max_sentence_repeat,
        "repeated_sentence_total": repeated_sentence_total,
        "max_window_repeat": max_window_repeat,
        "max_same_char_run": max_same_char_run,
    }


def step_texts(row: dict[str, Any]) -> list[str]:
    raw_steps = row.get("steps") or row.get("reference_steps") or []
    texts: list[str] = []
    for step in raw_steps:
        if isinstance(step, dict):
            text = step.get("text") or step.get("content") or ""
        else:
            text = str(step or "")
        text = str(text or "").strip()
        if text:
            texts.append(text)
    if texts:
        return texts
    return [str(item or "").strip() for item in (row.get("ref_steps") or []) if str(item or "").strip()]


def build_prompt_like_generate_py(row: dict[str, Any], prompt_index: int, model_name: str) -> Any:
    problem = row.get("question") or row.get("problem") or ""
    builder = Generate_Prompt(DummyModel(model_name), query=problem)
    for text in step_texts(row)[: prompt_index + 1]:
        builder.add_step(text)
    if hasattr(builder, "return_prompt_ids"):
        return builder.return_prompt_ids()
    return builder.return_prompt()


def signature_accepts_kwargs(callable_obj: Any) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def strip_prefix_flags(prompt: Any) -> list[dict[str, Any]] | None:
    if isinstance(prompt, list) and all(isinstance(message, dict) and "role" in message for message in prompt):
        return [{"role": message["role"], "content": message.get("content", "")} for message in prompt]
    return None


def render_with_tokenizer(prompt: Any, model_name: str) -> tuple[Any, list[int] | None, str, str | None]:
    messages = strip_prefix_flags(prompt)
    if messages is None:
        return prompt, None, "already_rendered", None
    if AutoTokenizer is None:
        return prompt, None, "messages_unrendered_no_transformers", "transformers is not installed"
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        sig = inspect.signature(tokenizer.apply_chat_template)
        accepts_kwargs = signature_accepts_kwargs(tokenizer.apply_chat_template)
        kwargs: dict[str, Any] = {"tokenize": False}
        if "add_generation_prompt" in sig.parameters or accepts_kwargs:
            kwargs["add_generation_prompt"] = False
        if "continue_final_message" in sig.parameters or accepts_kwargs:
            kwargs["continue_final_message"] = True
        if "enable_thinking" in sig.parameters or accepts_kwargs:
            kwargs.setdefault("enable_thinking", True)
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "reasoning_effort" in chat_template and ("reasoning_effort" in sig.parameters or accepts_kwargs):
            kwargs.setdefault("reasoning_effort", "high")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if rendered and not token_ids:
            raise ValueError("tokenizer produced an empty token id list for a non-empty rendered prompt")
        return rendered, token_ids, "tokenizer_apply_chat_template", None
    except Exception as exc:
        return prompt, None, "messages_unrendered_tokenizer_error", f"{type(exc).__name__}: {exc}"


def detect_issue_types(output: str, prompt_index: int, total_prefixes: int, runaway_min_chars: int) -> tuple[list[str], dict[str, Any]]:
    stripped = output.strip()
    metrics = repeat_metrics(output)
    issues: list[str] = []
    if prompt_index < total_prefixes - 1 and not stripped:
        issues.append("nonfinal_empty")
    if stripped and MOJIBAKE_RE.search(output):
        issues.append("mojibake")
    if stripped and metrics["large_repeat"]:
        issues.append("repeat")
    if stripped and len(output) >= runaway_min_chars and metrics["large_repeat"]:
        issues.append("runaway")
    return issues, metrics


def jsonl_rows(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "error": str(exc), "line_prefix": line[:180]})
                continue
            if isinstance(obj, dict):
                rows.append((line_no, obj))
    return rows, errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_generation(value: Any) -> str:
    text = _normalize_generation_input(value)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "")
    for token in ("<|tool_call_end|>", "<|tool_call_start|>", "<|tool▁call▁end|>", "<|tool▁call▁start|>"):
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def resolve_gen_only(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "gen_only.jsonl"
    if not resolved.exists():
        raise FileNotFoundError(f"gen_only file not found: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def build_runner(model: str) -> VLLMRunner:
    return VLLMRunner(
        model=model,
        vllm_config=dict(Config["reasoning_model_params"]),
        sampling_config=dict(Config["reasoning_sampling_params"]),
        gpus=str(Config["reasoning_model_gpus"]),
    )


def build_prompt_rows(gen_file: Path, model: str, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed_rows, parse_errors = jsonl_rows(gen_file)
    source_model_dir = gen_file.parent.name
    prompt_rows: list[dict[str, Any]] = []

    for source_line, row in parsed_rows:
        outputs = [str(item or "") for item in (row.get("gen_output") or [])]
        total_prefixes = len(outputs)
        old_prompts = row.get("prompts") or []
        for idx, output in enumerate(outputs):
            issue_types, metrics = detect_issue_types(
                output,
                idx,
                total_prefixes,
                DEFAULT_RUNAWAY_MIN_CHARS,
            )
            if not issue_types:
                continue

            repair_id = f"{source_model_dir}__line_{source_line:06d}__idx_{idx:04d}"
            rebuilt_prompt = build_prompt_like_generate_py(row, idx, model)
            old_prompt = old_prompts[idx] if idx < len(old_prompts) else None

            prompt_for_pack, prompt_token_ids, prompt_render_mode, prompt_render_error = render_with_tokenizer(
                rebuilt_prompt,
                model,
            )
            if prompt_render_error and isinstance(old_prompt, str) and old_prompt:
                prompt_for_pack = old_prompt
                prompt_token_ids = None
                prompt_render_mode = "fallback_old_rendered_prompt"

            prompt_rows.append(
                {
                    "repair_id": repair_id,
                    "prompt_id": repair_id,
                    "source_model_dir": source_model_dir,
                    "source_gen_file": str(gen_file),
                    "source_line": source_line,
                    "row_id": row.get("id"),
                    "case_id": row.get("case_id"),
                    "annotation_uid": row.get("annotation_uid"),
                    "sample_idx": row.get("sample_idx"),
                    "prompt_index": idx,
                    "step_index": idx,
                    "step_id": f"s{idx + 1}",
                    "total_prefixes": total_prefixes,
                    "is_final_prefix": idx == total_prefixes - 1,
                    "issue_types": issue_types,
                    "issue_metrics": metrics,
                    "old_output_sha256": sha256_text(output),
                    "old_output_len": len(output),
                    "old_output_excerpt": excerpt(output),
                    "old_prompt_sha256": sha256_obj(old_prompt) if old_prompt is not None else "",
                    "rebuilt_prompt_sha256": sha256_obj(rebuilt_prompt),
                    "pack_prompt_sha256": sha256_obj(prompt_for_pack),
                    "model_name": model,
                    "prompt_format": prompt_render_mode,
                    "prompt_render_error": prompt_render_error,
                    "prompt_messages": rebuilt_prompt,
                    "prompt": prompt_for_pack,
                    "prompt_token_ids": prompt_token_ids,
                    "question": row.get("question") or row.get("problem") or "",
                    "problem": row.get("problem") or row.get("question") or "",
                    "answer": row.get("answer") or "",
                    "standard_solution": row.get("standard_solution") or "",
                    "meta": {
                        "difficulty": row.get("difficulty"),
                        "old_prompt_matches_rebuilt": sha256_obj(old_prompt) == sha256_obj(rebuilt_prompt)
                        if old_prompt is not None
                        else None,
                        "old_prompt_matches_pack_prompt": sha256_obj(old_prompt) == sha256_obj(prompt_for_pack)
                        if old_prompt is not None
                        else None,
                    },
                }
            )

    write_jsonl(out_dir / "bad_generation_steps.jsonl", prompt_rows)
    write_jsonl(out_dir / "prompt_pack.jsonl", prompt_rows)
    return prompt_rows, parse_errors


def generate_batch(model: VLLMRunner, rows: list[dict[str, Any]]) -> list[str]:
    prompts = [row["prompt"] for row in rows]
    generations = model.generate(prompts, None)
    if len(generations) < len(prompts):
        generations.extend([""] * (len(prompts) - len(generations)))
    generations = generations[: len(prompts)]
    for attempt in range(1, MAX_EMPTY_RETRIES + 1):
        empty_indices = [
            idx
            for idx, text in enumerate(generations)
            if not _normalize_generation_input(text).strip()
        ]
        if not empty_indices:
            break
        print(f"[WARN] empty outputs at local indices {empty_indices}; retry {attempt}/{MAX_EMPTY_RETRIES}")
        retry_prompts = [prompts[idx] for idx in empty_indices]
        retry_generations = model.generate(retry_prompts, None)
        for local_idx, original_idx in enumerate(empty_indices):
            retry_text = retry_generations[local_idx] if local_idx < len(retry_generations) else ""
            if _normalize_generation_input(retry_text).strip():
                generations[original_idx] = retry_text
    return generations


def run_regeneration(prompt_rows: list[dict[str, Any]], model_path: str, out_dir: Path) -> list[dict[str, Any]]:
    batch_size = int(Config["reasoning_model_params"].get("max_num_seqs") or 16)
    batch_size = max(1, batch_size)
    runner = build_runner(model_path)
    output_rows: list[dict[str, Any]] = []
    processed = 0

    for batch in chunks(prompt_rows, batch_size):
        generations = generate_batch(runner, batch)
        for idx, row in enumerate(batch):
            raw_generation = generations[idx] if idx < len(generations) else ""
            cleaned = clean_generation(raw_generation)
            repeat = repeat_metrics(cleaned)
            output_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "repair_id": row["repair_id"],
                    "source_model_dir": row["source_model_dir"],
                    "source_gen_file": row["source_gen_file"],
                    "source_line": row["source_line"],
                    "row_id": row.get("row_id"),
                    "case_id": row.get("case_id"),
                    "sample_idx": row.get("sample_idx"),
                    "prompt_index": row["prompt_index"],
                    "total_prefixes": row["total_prefixes"],
                    "issue_types": row["issue_types"],
                    "old_output_sha256": row["old_output_sha256"],
                    "old_output_len": row["old_output_len"],
                    "model": model_path,
                    "raw_generation": raw_generation,
                    "generation": cleaned,
                    "is_empty": not _normalize_generation_input(cleaned).strip(),
                    "mojibake": bool(MOJIBAKE_RE.search(cleaned)),
                    "large_repeat": repeat["large_repeat"],
                    "repeat_metrics": repeat,
                    "generated_len": len(cleaned),
                }
            )
        processed += len(batch)
        print(f"[INFO] regenerated {processed}/{len(prompt_rows)}")

    write_jsonl(out_dir / "prompt_outputs.jsonl", output_rows)
    return output_rows


def issue_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues = Counter()
    combinations = Counter()
    formats = Counter()
    for row in rows:
        issue_types = tuple(sorted(row.get("issue_types") or []))
        combinations["+".join(issue_types)] += 1
        formats[row.get("prompt_format") or ""] += 1
        for issue in issue_types:
            issues[issue] += 1
    return {
        "total_bad_steps": len(rows),
        "issues_nonexclusive": dict(issues),
        "issue_combinations": dict(combinations),
        "prompt_formats": dict(formats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect bad prefixes in one gen_only.jsonl, rebuild regeneration prompts, "
            "and run them with local VLLMRunner. vLLM and sampling parameters come from benchmark_core/config.py."
        )
    )
    parser.add_argument("--model", required=True, help="Local model path or model id to pass to VLLMRunner.")
    parser.add_argument("--gen-only", required=True, type=Path, help="Path to gen_only.jsonl, or a directory containing it.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for prompt pack and regeneration outputs.")
    parser.add_argument("--max-prompts", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    started = time.time()
    gen_file = resolve_gen_only(args.gen_only)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] gen_only={gen_file}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] gpus={Config['reasoning_model_gpus']}")

    detected_prompt_rows, parse_errors = build_prompt_rows(gen_file, args.model, out_dir)
    prompt_rows = detected_prompt_rows
    if args.max_prompts is not None:
        prompt_rows = detected_prompt_rows[: args.max_prompts]
        write_jsonl(out_dir / "prompt_pack.jsonl", prompt_rows)
    print(f"[INFO] bad_prefix_prompts={len(prompt_rows)}")

    if prompt_rows:
        output_rows = run_regeneration(prompt_rows, args.model, out_dir)
    else:
        output_rows = []
        write_jsonl(out_dir / "prompt_outputs.jsonl", output_rows)
    summary = {
        "gen_only": str(gen_file),
        "out_dir": str(out_dir),
        "model": args.model,
        "gpus": str(Config["reasoning_model_gpus"]),
        "config_reasoning_model_params": Config["reasoning_model_params"],
        "config_reasoning_sampling_params": Config["reasoning_sampling_params"],
        "runaway_min_chars": DEFAULT_RUNAWAY_MIN_CHARS,
        "max_empty_retries": MAX_EMPTY_RETRIES,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "detected_prompt_summary": issue_summary(detected_prompt_rows),
        "run_prompt_summary": issue_summary(prompt_rows),
        "output_count": len(output_rows),
        "empty_count": sum(1 for row in output_rows if row["is_empty"]),
        "mojibake_count": sum(1 for row in output_rows if row["mojibake"]),
        "large_repeat_count": sum(1 for row in output_rows if row["large_repeat"]),
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "run_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] prompt_pack={out_dir / 'prompt_pack.jsonl'}")
    print(f"[DONE] prompt_outputs={out_dir / 'prompt_outputs.jsonl'}")
    print(f"[DONE] run_info={out_dir / 'run_info.json'}")


if __name__ == "__main__":
    main()
