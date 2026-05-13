#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_core.config import Config
from benchmark_core.data_process import _normalize_generation_input
from benchmark_core.prompt import Generate_Prompt
from generate_ds import (
    DEEPSEEK_CONFIG,
    DeepSeekContinuationRunner,
    _api_sampling_params_from_config,
    _ensure_deepseek_prefix_messages,
    _looks_like_json_schema,
    _non_retryable_error,
)


MOJIBAKE_RE = re.compile(
    r"�|ï¿½|(?:Ã[\x80-\xBF])|(?:Â[\x80-\xBF])|(?:â[\x80-\xBF]{1,2})|[\x00-\x08\x0b\x0c\x0e-\x1f]"
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
SPECIAL_GENERATION_TOKENS = (
    "<|tool_call_end|>",
    "<|tool_call_start|>",
    "<|tool▁call▁end|>",
    "<|tool▁call▁start|>",
)

DEFAULT_RUNAWAY_MIN_CHARS = 30000
MAX_EMPTY_RETRIES = 2
FALLBACK_MAX_TOKENS = 2048


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


def clean_generation(value: Any) -> str:
    text = _normalize_generation_input(value)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "")
    for token in SPECIAL_GENERATION_TOKENS:
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


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


def empty_repeat_metrics() -> dict[str, Any]:
    return {
        "large_repeat": False,
        "max_sentence_repeat": 0,
        "repeated_sentence_total": 0,
        "max_window_repeat": 0,
        "max_same_char_run": 0,
    }


def detect_issue_types(output: str, prompt_index: int, total_prefixes: int, runaway_min_chars: int) -> tuple[list[str], dict[str, Any]]:
    del runaway_min_chars
    issues: list[str] = []
    if prompt_index < total_prefixes - 1 and not output.strip():
        issues.append("nonfinal_empty")
    return issues, empty_repeat_metrics()


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


def resolve_gen_only(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "gen_only.jsonl"
    if not resolved.exists():
        raise FileNotFoundError(f"gen_only file not found: {resolved}")
    return resolved


def chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def api_sampling_params(model_name: str, max_tokens_override: int | None = None) -> dict[str, Any]:
    extra_sampling: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p", "stop"):
        if DEEPSEEK_CONFIG.get(key) is not None:
            extra_sampling[key] = DEEPSEEK_CONFIG[key]
    if isinstance(DEEPSEEK_CONFIG.get("extra_params"), dict):
        extra_sampling.update(DEEPSEEK_CONFIG["extra_params"])

    params = _api_sampling_params_from_config(model_name, extra_sampling)
    # In DeepSeek beta prefix-continuation repair, the generic stop list can
    # terminate otherwise valid continuations immediately for some prefixes.
    params.pop("stop", None)
    if max_tokens_override is not None:
        params["max_tokens"] = max(1, int(max_tokens_override))
    min_tokens = int((Config.get("reasoning_sampling_params") or {}).get("min_tokens") or 1)
    params["min_tokens"] = max(1, min_tokens)
    return params


class DeepSeekRepairRunner(DeepSeekContinuationRunner):
    def _get_client(self, max_workers: int) -> OpenAI:
        if getattr(self._tls, "client", None) is None:
            limits = httpx.Limits(
                max_connections=max(32, max_workers * 4),
                max_keepalive_connections=max(16, max_workers * 2),
                keepalive_expiry=30,
            )
            http_client = httpx.Client(
                timeout=httpx.Timeout(600.0, connect=10.0),
                limits=limits,
                http2=False,
            )
            self._tls.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
            )
        return self._tls.client

    def generate_one(
        self,
        prompt: Any,
        extra_params: dict | None = None,
        *,
        max_workers_hint: int = 8,
    ) -> dict[str, str]:
        params = dict(self.default_params)
        min_tokens = params.pop("min_tokens", None)
        schema = None
        if extra_params:
            if _looks_like_json_schema(extra_params):
                schema = extra_params
            else:
                params.update(extra_params)
        if schema is not None:
            params.setdefault("response_format", {"type": "json_object"})

        messages = _ensure_deepseek_prefix_messages(prompt)
        client = self._get_client(max_workers_hint)

        extra_body: dict[str, Any] = {}
        if self.thinking_type and self.thinking_type != "none":
            extra_body["thinking"] = {"type": self.thinking_type}
        if min_tokens is not None:
            extra_body["min_tokens"] = max(1, int(min_tokens))

        last_error: Exception | None = None
        for attempt in range(self.request_max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    **params,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                reasoning = getattr(msg, "reasoning_content", "") or ""
                content = getattr(msg, "content", "") or ""
                return {"reasoning": reasoning, "content": content}
            except Exception as exc:
                last_error = exc
                if _non_retryable_error(exc) or attempt >= self.request_max_retries:
                    break
                sleep_s = self.retry_sleep_seconds * (2 ** attempt)
                print(
                    f"[WARN][DEEPSEEK] request failed at attempt "
                    f"{attempt + 1}/{self.request_max_retries + 1}: {exc}; "
                    f"retry in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)

        assert last_error is not None
        return {"reasoning": "", "content": f"<Error: {last_error}" + ">"}


def build_deepseek_runner(model_name: str, *, max_tokens_override: int | None = None) -> DeepSeekContinuationRunner:
    return DeepSeekRepairRunner(
        model_name=model_name,
        api_key=str(DEEPSEEK_CONFIG["api_key"]),
        base_url=str(DEEPSEEK_CONFIG["base_url"]),
        max_workers_default=int(DEEPSEEK_CONFIG["max_workers"]),
        default_params=api_sampling_params(model_name, max_tokens_override=max_tokens_override),
        thinking_type=str(DEEPSEEK_CONFIG["thinking_type"]),
        request_max_retries=int(DEEPSEEK_CONFIG["request_max_retries"]),
        retry_sleep_seconds=float(DEEPSEEK_CONFIG["retry_sleep_seconds"]),
    )


def is_api_error(text: Any) -> bool:
    stripped = str(text or "").strip()
    return stripped.startswith("<Error:") or stripped.startswith("Error code:")


def prompt_with_assistant_trailing_space(prompt: Any) -> Any:
    copied = json.loads(json.dumps(prompt, ensure_ascii=False))
    if isinstance(copied, list) and copied and isinstance(copied[-1], dict):
        copied[-1]["content"] = str(copied[-1].get("content") or "") + " "
    return copied


def build_prompt_rows(gen_file: Path, model_name: str, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parsed_rows, parse_errors = jsonl_rows(gen_file)
    source_model_dir = gen_file.parent.name
    prompt_rows: list[dict[str, Any]] = []

    for source_line, row in parsed_rows:
        outputs = [str(item or "") for item in (row.get("gen_output") or [])]
        total_prefixes = len(outputs)
        old_prompts = row.get("prompts") or []
        for idx, output in enumerate(outputs):
            issue_types, metrics = detect_issue_types(output, idx, total_prefixes, DEFAULT_RUNAWAY_MIN_CHARS)
            if not issue_types:
                continue

            repair_id = f"{source_model_dir}__line_{source_line:06d}__idx_{idx:04d}"
            rebuilt_prompt = build_prompt_like_generate_py(row, idx, model_name)
            old_prompt = old_prompts[idx] if idx < len(old_prompts) else None
            prompt_for_pack = rebuilt_prompt
            prompt_format = "generate_py_messages"
            prompt_render_error = None
            if not prompt_for_pack and old_prompt:
                prompt_for_pack = old_prompt
                prompt_format = "fallback_old_prompt"
                prompt_render_error = "rebuilt prompt was empty"

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
                    "model_name": model_name,
                    "prompt_format": prompt_format,
                    "prompt_render_error": prompt_render_error,
                    "prompt": prompt_for_pack,
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


def generate_batch(runner: DeepSeekContinuationRunner, rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    prompts = [row["prompt"] for row in rows]
    prompt_variants = ["original"] * len(prompts)
    reasonings, generations = runner.generate(prompts, None)
    if len(generations) < len(prompts):
        generations.extend([""] * (len(prompts) - len(generations)))
    if len(reasonings) < len(prompts):
        reasonings.extend([""] * (len(prompts) - len(reasonings)))
    generations = generations[: len(prompts)]
    reasonings = reasonings[: len(prompts)]

    for attempt in range(1, MAX_EMPTY_RETRIES + 1):
        retry_indices = [
            idx
            for idx, text in enumerate(generations)
            if not _normalize_generation_input(text).strip() or is_api_error(text)
        ]
        if not retry_indices:
            break
        print(f"[WARN] failed API outputs at local indices {retry_indices}; retry {attempt}/{MAX_EMPTY_RETRIES}")
        retry_prompts = [prompts[idx] for idx in retry_indices]
        retry_reasonings, retry_generations = runner.generate(retry_prompts, None)
        for local_idx, original_idx in enumerate(retry_indices):
            retry_text = retry_generations[local_idx] if local_idx < len(retry_generations) else ""
            if _normalize_generation_input(retry_text).strip() and not is_api_error(retry_text):
                generations[original_idx] = retry_text
                reasonings[original_idx] = retry_reasonings[local_idx] if local_idx < len(retry_reasonings) else ""
    retry_indices = [
        idx
        for idx, text in enumerate(generations)
        if not _normalize_generation_input(text).strip() or is_api_error(text)
    ]
    if retry_indices:
        print(f"[WARN] using trailing-space fallback for local indices {retry_indices}")
        fallback_runner = build_deepseek_runner(runner.model_name, max_tokens_override=FALLBACK_MAX_TOKENS)
        fallback_prompts = [prompt_with_assistant_trailing_space(prompts[idx]) for idx in retry_indices]
        fallback_reasonings, fallback_generations = fallback_runner.generate(
            fallback_prompts,
            None,
            max_workers=min(len(fallback_prompts), int(DEEPSEEK_CONFIG["max_workers"])),
        )
        for local_idx, original_idx in enumerate(retry_indices):
            fallback_text = fallback_generations[local_idx] if local_idx < len(fallback_generations) else ""
            if _normalize_generation_input(fallback_text).strip() and not is_api_error(fallback_text):
                generations[original_idx] = fallback_text
                reasonings[original_idx] = (
                    fallback_reasonings[local_idx] if local_idx < len(fallback_reasonings) else ""
                )
                prompt_variants[original_idx] = "assistant_trailing_space_max_tokens_2048"
    return reasonings, generations, prompt_variants


def run_regeneration(prompt_rows: list[dict[str, Any]], model_name: str, out_dir: Path) -> list[dict[str, Any]]:
    batch_size = max(1, int(DEEPSEEK_CONFIG["max_workers"]))
    runner = build_deepseek_runner(model_name)
    output_rows: list[dict[str, Any]] = []
    processed = 0

    for batch in chunks(prompt_rows, batch_size):
        reasonings, generations, prompt_variants = generate_batch(runner, batch)
        for idx, row in enumerate(batch):
            raw_generation = generations[idx] if idx < len(generations) else ""
            reasoning = reasonings[idx] if idx < len(reasonings) else ""
            cleaned = clean_generation(raw_generation)
            repeat = repeat_metrics(cleaned)
            api_error = is_api_error(raw_generation) or is_api_error(cleaned)
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
                    "model": model_name,
                    "raw_reasoning": reasoning,
                    "raw_generation": raw_generation,
                    "generation": cleaned,
                    "repair_prompt_variant": prompt_variants[idx] if idx < len(prompt_variants) else "original",
                    "api_error": api_error,
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
            "Detect non-final empty prefixes in a DeepSeek API gen_only.jsonl, rebuild prompts, "
            "and regenerate them through generate_ds.py's DeepSeek prefix API runner."
        )
    )
    parser.add_argument("--model", default=str(DEEPSEEK_CONFIG["model"]))
    parser.add_argument("--gen-only", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-prompts", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    started = time.time()
    gen_file = resolve_gen_only(args.gen_only)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] gen_only={gen_file}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] deepseek_min_tokens={api_sampling_params(args.model).get('min_tokens')}")

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
        "base_url": DEEPSEEK_CONFIG["base_url"],
        "max_workers": DEEPSEEK_CONFIG["max_workers"],
        "thinking_type": DEEPSEEK_CONFIG["thinking_type"],
        "api_sampling_params": api_sampling_params(args.model),
        "detection_policy": "nonfinal_empty_only",
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
        "api_error_count": sum(1 for row in output_rows if row.get("api_error")),
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "run_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] prompt_pack={out_dir / 'prompt_pack.jsonl'}")
    print(f"[DONE] prompt_outputs={out_dir / 'prompt_outputs.jsonl'}")
    print(f"[DONE] run_info={out_dir / 'run_info.json'}")


if __name__ == "__main__":
    main()
