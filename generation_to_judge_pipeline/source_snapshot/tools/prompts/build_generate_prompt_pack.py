#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - vllm-messages can still be built
    AutoTokenizer = None

from benchmark_core.config import Config


SYSTEM_MESSAGE = (
    "You are a mathematician. Solve the problem."
    " You are continuing an already-started assistant solution."
    " The final assistant message is a literal prefix of the solution; your output must be only the text that immediately follows that prefix."
    "## Style preferences (keep them light; do not change your underlying approach):"
    "- Treat `current_solution`/`ref` as correct established premises and build directly on them."
    "- Start immediately with the next logical derivation. Do not restate the problem or re-summarize what has already been established."
    "- Write as continuous mathematical prose (no section headers, no “Step 1/2/3”)."
    "- Avoid repeating the same conditions. If you must reference a prior premise, do it minimally (e.g., “from the previous inequality …”)."
    "- Never write meta-analysis or first-person planning such as “The user wants me to solve”, “Let me understand”, “I need to solve”, “I need to find”, or “First, let me parse”."
)


PROMPT_FORMATS = {
    "auto",
    "vllm-messages",
    "chat-template",
}


def _jsonl_rows(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
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
            yield line_no, obj


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _safe_id(text: Any, fallback: str) -> str:
    raw = _first_nonempty(text, fallback) or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or fallback


def _clean_reference_step_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("content") or ""
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}") and "text" in stripped:
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    value = parsed.get("text") or parsed.get("content") or ""
            except (SyntaxError, ValueError):
                pass
    return str(value or "").strip()


def _reference_step_records(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_steps = (
        obj.get("reference_steps")
        or obj.get("steps")
        or obj.get("segments")
        or []
    )
    records: List[Dict[str, Any]] = []
    for idx, step in enumerate(raw_steps):
        text = _clean_reference_step_text(step)
        if not text:
            continue
        step_type = "text"
        source_id = idx
        if isinstance(step, dict):
            step_type = str(step.get("type") or "text")
            source_id = step.get("id", idx)
        records.append(
            {
                "step_id": f"s{len(records) + 1}",
                "text": text,
                "type": step_type,
                "source_step_id": source_id,
                "source_index": idx,
            }
        )
    return records


def _base_messages(problem: str, prefix_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"Solve the Problem:\n{problem}"},
    ]
    messages.append({"role": "assistant", "content": prefix_text, "prefix": True})
    return messages


def _load_tokenizer(model_name: str) -> Any:
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not installed; cannot build chat-template prompt format.")
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)


def _apply_chat_template(messages: List[Dict[str, Any]], model_name: str) -> str:
    tokenizer = _load_tokenizer(model_name)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(f"Tokenizer for {model_name} does not provide apply_chat_template().")

    tokenizer_messages = [
        {"role": message["role"], "content": message.get("content", "")}
        for message in messages
    ]
    sig = inspect.signature(tokenizer.apply_chat_template)
    kwargs: Dict[str, Any] = {"tokenize": False}
    if "add_generation_prompt" in sig.parameters:
        kwargs["add_generation_prompt"] = False
    if "continue_final_message" not in sig.parameters:
        raise RuntimeError(
            f"Tokenizer for {model_name} does not support continue_final_message; "
            "choose --prompt-format vllm-messages and let VLLMRunner handle rendering."
        )
    kwargs["continue_final_message"] = True
    if "enable_thinking" in sig.parameters:
        kwargs["enable_thinking"] = True
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "reasoning_effort" in chat_template:
        kwargs.setdefault("reasoning_effort", "high")
    return tokenizer.apply_chat_template(tokenizer_messages, **kwargs)


def _resolve_prompt_format(requested: str, model_name: str) -> str:
    if requested not in PROMPT_FORMATS:
        raise ValueError(f"Unsupported prompt format: {requested}")
    if requested != "auto":
        return requested
    return "vllm-messages"


def _format_prompt(problem: str, prefix_text: str, model_name: str, prompt_format: str) -> Tuple[Any, str]:
    actual_format = _resolve_prompt_format(prompt_format, model_name)
    messages = _base_messages(problem, prefix_text)
    if actual_format == "chat-template":
        return _apply_chat_template(messages, model_name), actual_format
    return messages, actual_format


def _build_rows_for_case(
    obj: Dict[str, Any],
    *,
    source_line: int,
    case_ordinal: int,
    model_name: str,
    prompt_format: str,
    include_final_prefix: bool,
) -> List[Dict[str, Any]]:
    problem = _first_nonempty(obj.get("question"), obj.get("problem"))
    if not problem:
        raise ValueError(f"Input line {source_line} is missing question/problem.")

    steps = _reference_step_records(obj)
    upper = len(steps) if include_final_prefix else max(0, len(steps) - 1)
    case_id = _safe_id(
        _first_nonempty(obj.get("id"), obj.get("case_id"), obj.get("original_case_id")),
        f"case_{case_ordinal:06d}",
    )
    answer = _first_nonempty(obj.get("answer"), obj.get("reference_answer")) or ""
    standard_solution = _first_nonempty(obj.get("standard_solution"), obj.get("solution")) or ""

    rows: List[Dict[str, Any]] = []
    prefix_parts: List[str] = []
    actual_format: Optional[str] = None
    for step_index, step in enumerate(steps[:upper]):
        prefix_parts.append(step["text"])
        prefix_text = "\n".join(prefix_parts).strip()
        prompt, actual_format = _format_prompt(problem, prefix_text, model_name, prompt_format)
        prompt_id = f"{case_id}__gen_step_{step_index + 1:04d}"
        rows.append(
            {
                "prompt_id": prompt_id,
                "case_id": case_id,
                "case_row_index": case_ordinal - 1,
                "source_line": source_line,
                "step_index": step_index,
                "step_id": step["step_id"],
                "prompt_kind": "generate_continuation",
                "model_name": model_name,
                "prompt_format": actual_format,
                "continuation_mode": (
                    "tokenizer_continue_final_message"
                    if actual_format == "chat-template"
                    else "vllm_runner_model_specific"
                ),
                "prompt": prompt,
                "prefix_text": prefix_text,
                "prefix_step_ids": [s["step_id"] for s in steps[: step_index + 1]],
                "next_reference_step": steps[step_index + 1] if step_index + 1 < len(steps) else None,
                "problem": problem,
                "question": problem,
                "answer": answer,
                "standard_solution": standard_solution,
                "reference_steps": steps,
                "meta": {
                    "domain": obj.get("domain"),
                    "difficulty": obj.get("difficulty"),
                    "source": obj.get("source"),
                    "segment_num": obj.get("segment_num"),
                    "input_id": obj.get("id"),
                    "include_final_prefix": include_final_prefix,
                },
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build JSONL prompt packs for generation continuation over reference steps."
    )
    parser.add_argument("--input-path", default=Config["Input_path"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=Config["reasoning_model"])
    parser.add_argument(
        "--prompt-format",
        default="auto",
        choices=sorted(PROMPT_FORMATS),
        help=(
            "auto: write vllm message prompts and let VLLMRunner apply each model's supported "
            "assistant-continuation rule. chat-template pre-renders with tokenizer "
            "continue_final_message for local debugging only."
        ),
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--include-final-prefix",
        action="store_true",
        help="Also build a prompt after the final reference step. Normally this is not useful.",
    )
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    all_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    case_count = 0

    for source_line, obj in _jsonl_rows(input_path):
        if args.max_cases is not None and case_count >= args.max_cases:
            break
        case_count += 1
        try:
            rows = _build_rows_for_case(
                obj,
                source_line=source_line,
                case_ordinal=case_count,
                model_name=args.model,
                prompt_format=args.prompt_format,
                include_final_prefix=args.include_final_prefix,
            )
        except Exception as exc:
            skipped.append({"source_line": source_line, "error": str(exc)})
            continue
        if not rows:
            skipped.append({"source_line": source_line, "error": "no reference continuation prompts"})
            continue
        all_rows.extend(rows)

    prompt_count = _write_jsonl(output_path, all_rows)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_path.with_suffix(".manifest.json")
    manifest = {
        "input_path": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "requested_prompt_format": args.prompt_format,
        "case_count_seen": case_count,
        "prompt_count": prompt_count,
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "include_final_prefix": args.include_final_prefix,
        "schema": "one_jsonl_row_per_generation_prompt",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] cases_seen={case_count}, prompts={prompt_count}, skipped={len(skipped)}")
    print(f"[DONE] prompt_pack={output_path}")
    print(f"[DONE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
