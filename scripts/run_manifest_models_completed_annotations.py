#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run completed-annotation generation for every local model in model/download_manifest.tsv.

Workflow per model:
1. Resolve local model path from manifest, with basename fallback under /data/pretrain.
2. Run a 10-row smoke generation and prompt pack.
3. Validate gen_only.jsonl and packed prompts.
4. If smoke passes, run full 317 rows. Use sharded parallelism when the model
   can fit with tp=1; otherwise use tp=2 or tp=4.
5. Validate each finished artifact and write a machine-readable summary.

This script is intentionally conservative for very large models: directories
larger than the available 4x80GB H100 memory budget are recorded as blocked
instead of launching a predictable OOM run.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO = Path("/home/zhaipengxiang/benchmark")
PY = Path(os.environ.get("COMPLETED_MANIFEST_PY", "/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12"))
MANIFEST = REPO / "model/download_manifest.tsv"
TEST_INPUT = REPO / "workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl"
FULL_INPUT = REPO / "workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl"
OUT_ROOT = Path(os.environ.get("COMPLETED_MANIFEST_OUT_ROOT", str(REPO / "artifacts/model_outputs/completed_annotations_manifest_models")))
LOG_ROOT = Path(os.environ.get("COMPLETED_MANIFEST_LOG_ROOT", str(REPO / "logs/completed_annotations_manifest_models")))
STATUS_PATH = LOG_ROOT / "summary.json"
SUMMARY_JSONL = LOG_ROOT / "model_results.jsonl"
CONTINUATION_ISSUES_MD = LOG_ROOT / "continuation_issues.md"
CONTINUATION_COMPAT_MD = LOG_ROOT / "continuation_compatibility_report.md"
WRAPPER = REPO / "scripts/run_completed_annotations_generate_and_pack.py"
PROBE = REPO / "scripts/probe_manifest_model_continuation.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "model"


def run(cmd: Sequence[str], *, log: Path, env: Optional[Dict[str, str]] = None, cwd: Path = REPO) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env.pop("PYTHONSTARTUP", None)
    clean_env.pop("PYTHON_BASIC_REPL", None)
    if env:
        clean_env.update(env)
    with log.open("ab") as f:
        f.write(("\n\n===== START %s =====\n%s\n" % (now(), " ".join(cmd))).encode())
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=clean_env)
        return proc.wait()


def run_many(commands: List[Tuple[Sequence[str], Path]]) -> List[int]:
    procs = []
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONSTARTUP", None)
    env.pop("PYTHON_BASIC_REPL", None)
    for cmd, log in commands:
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("ab")
        f.write(("\n\n===== START %s =====\n%s\n" % (now(), " ".join(cmd))).encode())
        f.flush()
        procs.append((subprocess.Popen(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=env), f))
    codes = []
    for proc, f in procs:
        codes.append(proc.wait())
        f.close()
    return codes


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def dir_size_gib(path: Path) -> float:
    proc = subprocess.run(["du", "-s", str(path)], text=True, capture_output=True, check=True)
    kib = int(proc.stdout.split()[0])
    return kib / 1024 / 1024


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(data: Dict[str, Any]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at_utc"] = now()
    STATUS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_result(row: Dict[str, Any]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["updated_at_utc"] = now()
    with SUMMARY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_continuation_issue(result: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return a human-facing continuation issue classification, if present."""
    text = json.dumps(result, ensure_ascii=False)
    lower = text.lower()
    probe = result.get("continuation_probe")
    if isinstance(probe, dict) and not probe.get("ok"):
        return {
            "issue_type": str(probe.get("issue_type") or "preflight_continuation_failed"),
            "likely_cause": str(probe.get("likely_cause") or "tokenizer_or_chat_template_capability"),
            "analysis": (
                "预检阶段无法稳定渲染该模型当前 runner 支持的 assistant-prefix "
                "续写 prompt，因此按本轮规则跳过模型加载和生成；不尝试语义 cue/fallback。"
            ),
        }
    if "does not support continue_final_message" in text:
        return {
            "issue_type": "tokenizer_no_supported_assistant_continuation",
            "likely_cause": "tokenizer_or_chat_template_capability",
            "analysis": (
                "该模型在当前本地 HF/vLLM runner 下没有可用的 assistant 续写渲染方式。"
                "本地 HF/vLLM 路径优先使用 tokenizer 的 continue_final_message；"
                "API 路径可使用各自原生 partial/prefill 机制。"
            ),
        }
    for key in ("smoke_validation", "full_validation"):
        validation = result.get(key) or {}
        count = validation.get("scored_empty_output_count") or 0
        if count:
            return {
                "issue_type": "empty_generation_under_continuation",
                "likely_cause": "model_behavior_or_chat_template_mismatch",
                "analysis": (
                    f"{key} 中有 {count} 个处在 judge 评分窗口内的续写输出为空。"
                    "这说明该模型支持的续写渲染链路技术上可执行，但模型可能把前缀视为完整回答并直接 EOS，"
                    "也可能是 chat template 与 assistant prefix 续写任务不匹配。按当前要求，不改用语义 cue/fallback。"
                ),
            }
    if "continue_final_message" in lower and ("error" in lower or "exception" in lower or "traceback" in lower):
        return {
            "issue_type": "continuation_render_or_runtime_error",
            "likely_cause": "code_or_tokenizer_runtime",
            "analysis": (
                "日志中出现本地 HF/vLLM 续写渲染相关运行错误。需要优先检查 tokenizer/chat_template 签名、"
                "transformers 版本以及 runner 的 prompt 渲染分支；API 模型则检查对应 partial/prefill 参数。"
            ),
        }
    return None


def refresh_continuation_issue_doc() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if SUMMARY_JSONL.exists():
        for line in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            issue = classify_continuation_issue(row)
            if issue:
                rows.append({**row, "continuation_issue": issue})

    lines = [
        "# assistant 续写问题记录",
        "",
        f"更新时间：{now()}",
        "",
        "规则：如果某模型有原生 assistant 续写/partial/prefill 机制，就使用该模型支持的机制。"
        "本地 HF/vLLM 模型使用 tokenizer `continue_final_message`；Kimi/Moonshot API 使用 assistant `partial=true`。"
        "若原生续写机制不可用或不稳定，本轮不使用 cue/fallback 改写 prompt 语义绕过，只记录问题、证据和归因。",
        "",
        "归因口径：",
        "- `tokenizer_or_chat_template_capability`：当前 runner 找不到该模型可用的原生续写渲染方式。",
        "- `model_behavior_or_chat_template_mismatch`：渲染可执行，但模型在评分窗口内直接返回空输出/EOS。",
        "- `code_or_tokenizer_runtime`：runner 或 tokenizer 渲染分支出现运行时错误，需要修代码或版本适配。",
        "",
    ]
    if not rows:
        lines += ["当前尚无 assistant 续写异常。", ""]
    else:
        for row in rows:
            issue = row["continuation_issue"]
            model = row.get("model_name")
            lines += [
                f"## {model}",
                "",
                f"- 状态：`{row.get('status')}`",
                f"- resolved_path：`{row.get('resolved_path')}`",
                f"- 问题类型：`{issue['issue_type']}`",
                f"- 初步归因：`{issue['likely_cause']}`",
                f"- 分析：{issue['analysis']}",
            ]
            for key in ("smoke_validation", "full_validation", "prior_validation_failed"):
                validation = row.get(key)
                if isinstance(validation, dict):
                    evidence = {
                        "rows": validation.get("rows"),
                        "expected_rows": validation.get("expected_rows"),
                        "scored_empty_output_count": validation.get("scored_empty_output_count"),
                        "scored_empty_output_examples": validation.get("scored_empty_output_examples"),
                        "suspicious_output_count": validation.get("suspicious_output_count"),
                        "error": validation.get("error"),
                    }
                    evidence = {k: v for k, v in evidence.items() if v not in (None, [], "")}
                    if evidence:
                        lines += [f"- 证据 `{key}`：", "```json", json.dumps(evidence, ensure_ascii=False, indent=2), "```"]
            status = row.get("smoke_validation", {}).get("status") if isinstance(row.get("smoke_validation"), dict) else None
            if status:
                lines += ["- 状态片段：", "```json", json.dumps(status, ensure_ascii=False, indent=2)[:4000], "```"]
            lines.append("")
    CONTINUATION_ISSUES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_result_and_refresh(row: Dict[str, Any]) -> None:
    append_result(row)
    refresh_continuation_issue_doc()


def resolve_model_paths() -> List[Dict[str, Any]]:
    index: Dict[str, List[Path]] = {}
    pretrain = Path("/data/pretrain")
    if pretrain.exists():
        for p in pretrain.rglob("*"):
            if p.is_dir() and not p.name.startswith("."):
                index.setdefault(p.name, []).append(p)

    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            local_dir = rec["local_dir"]
            candidates = []
            for p in [REPO / local_dir, Path("/") / local_dir]:
                if p.exists():
                    candidates.append(p)
            if not candidates:
                candidates.extend(index.get(Path(local_dir).name, []))
            filtered = []
            for p in candidates:
                if any((p / name).exists() for name in ("config.json", "params.json", "tokenizer.json", "tokenizer_config.json")):
                    filtered.append(p)
            if filtered:
                candidates = filtered
            rec["resolved_path"] = str(candidates[0]) if candidates else None
            rec["exists"] = bool(candidates)
            rows.append(rec)
    return rows


@dataclass
class Plan:
    mode: str
    tp: int
    shard_count: int
    gpu_groups: List[str]
    max_model_len: int = 8192
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.35
    config_format: Optional[str] = None
    load_format: Optional[str] = None
    tokenizer_mode: Optional[str] = None


def choose_plan(rec: Dict[str, Any], size_gib: float) -> Tuple[Optional[Plan], Optional[str]]:
    name = rec["model_name"]
    track = rec.get("vllm_track") or ""

    if size_gib > 360:
        return None, f"blocked_insufficient_gpu_memory: model directory is {size_gib:.1f} GiB, above 4xH100 practical budget"

    if "special_mistral" in track:
        return Plan(
            mode="tp2",
            tp=2,
            shard_count=2,
            gpu_groups=["0,1", "2,3"],
            max_model_len=8192,
            max_num_seqs=16,
            gpu_memory_utilization=0.55,
            config_format="mistral",
            load_format="mistral",
            tokenizer_mode="mistral",
        ), None

    if size_gib >= 120 or "MiniMax" in name or "DeepSeek" in name or "Hunyuan" in name:
        return Plan(
            mode="tp4",
            tp=4,
            shard_count=1,
            gpu_groups=["0,1,2,3"],
            max_model_len=8192,
            max_num_seqs=16,
            gpu_memory_utilization=0.70,
        ), None

    if size_gib >= 70:
        return Plan(
            mode="tp2",
            tp=2,
            shard_count=2,
            gpu_groups=["0,1", "2,3"],
            max_model_len=8192,
            max_num_seqs=16,
            gpu_memory_utilization=0.55,
        ), None

    if size_gib >= 40:
        return Plan(
            mode="shard4_highmem",
            tp=1,
            shard_count=4,
            gpu_groups=["0", "1", "2", "3"],
            max_model_len=8192,
            max_num_seqs=16,
            gpu_memory_utilization=0.85,
        ), None

    return Plan(
        mode="shard4",
        tp=1,
        shard_count=4,
        gpu_groups=["0", "1", "2", "3"],
        max_model_len=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.35,
    ), None


def chat_template_kwargs_for_model(rec: Dict[str, Any]) -> Dict[str, Any]:
    if "special_mistral" in (rec.get("vllm_track") or ""):
        return {}
    note = (rec.get("continue_note") or "").lower()
    kwargs: Dict[str, Any] = {}
    if "enable_thinking=false" in note:
        kwargs["enable_thinking"] = False
    if "enable_thinking=true" in note:
        kwargs["enable_thinking"] = True
    if "thinking=false" in note or "think_tags_off" in note:
        kwargs["thinking"] = False
        kwargs.setdefault("enable_thinking", False)
    if "reasoning_effort=none" in note or "non_reasoning_mode" in note:
        kwargs["reasoning_effort"] = "none"
        kwargs.setdefault("enable_thinking", False)
        kwargs.setdefault("thinking", False)
    return kwargs


def no_system_role_for_model(rec: Dict[str, Any]) -> bool:
    return (rec.get("model_name") or "").lower() == "gemma-2-27b-it"


def chat_template_content_overrides_for_model(rec: Dict[str, Any]) -> Dict[str, str]:
    name = rec.get("model_name") or ""
    if name == "NVIDIA-Nemotron-Nano-9B-v2":
        # Native Nemotron-H control: its chat template scans system/user content
        # for /no_think and disables the model's thinking channel.
        return {"system_suffix": "/no_think", "first_user_prefix": ""}
    return {"system_suffix": "", "first_user_prefix": ""}


def sampling_overrides_for_model(rec: Dict[str, Any]) -> Dict[str, Any]:
    name = rec.get("model_name") or ""
    overrides: Dict[str, Any] = {}
    if "Nemotron" in name:
        # Replacement-character generations are invalid for downstream judge
        # prompts; banning the literal token is a model-local decoding fix.
        overrides["bad_words"] = ["\ufffd"]
    if name == "NVIDIA-Nemotron-Nano-9B-v2":
        # This model can emit EOS immediately under assistant prefill. Keep the
        # native prefill prompt but ignore EOS during generation.
        overrides["ignore_eos"] = True
    return overrides


def run_continuation_probe(
    model_name: str,
    model_path: Path,
    chat_template_kwargs: Dict[str, Any],
    *,
    no_system_role: bool,
    system_suffix: str,
    first_user_prefix: str,
) -> Dict[str, Any]:
    out_json = LOG_ROOT / "continuation_probes" / f"{safe_slug(model_name)}.json"
    cmd = [
        str(PY),
        str(PROBE),
        "--model-path",
        str(model_path),
        "--model-name",
        model_name,
        "--chat-template-kwargs-json",
        json.dumps(chat_template_kwargs, ensure_ascii=False),
    ]
    if no_system_role:
        cmd.append("--no-system-role")
    if system_suffix:
        cmd += ["--system-suffix", system_suffix]
    if first_user_prefix:
        cmd += ["--first-user-prefix", first_user_prefix]
    cmd += ["--out-json", str(out_json)]
    code = run(cmd, log=LOG_ROOT / "continuation_probes" / f"{safe_slug(model_name)}.log")
    probe = load_json(out_json)
    if not probe:
        probe = {"ok": False, "exit_code": code, "error": "probe produced no json"}
    probe["exit_code"] = code
    return probe


def refresh_continuation_compatibility_doc() -> None:
    probe_dir = LOG_ROOT / "continuation_probes"
    lines = [
        "# tokenizer 续写兼容性预检",
        "",
        f"更新时间：{now()}",
        "",
        "本表只检查 tokenizer/chat_template 是否能渲染 assistant-prefix 续写；通过后仍需跑 10 条小样本验证模型是否会正常续写而不是直接 EOS。",
        "",
    ]
    probe_paths = sorted(probe_dir.glob("*.json")) if probe_dir.exists() else []
    if not probe_paths:
        lines += ["尚未产生预检结果。", ""]
    for path in probe_paths:
        probe = load_json(path)
        model = probe.get("model_name") or path.stem
        lines += [
            f"## {model}",
            "",
            f"- 预检结论：`{'通过' if probe.get('ok') else '不通过'}`",
            f"- tokenizer：`{probe.get('tokenizer_class')}`",
            f"- continuation_mode：`{probe.get('continuation_mode')}`",
            f"- supports_assistant_continuation：`{probe.get('supports_assistant_continuation')}`",
            f"- supports_continue_final_message：`{probe.get('supports_continue_final_message')}`",
            f"- template kwargs：`{json.dumps(probe.get('chat_template_kwargs_used') or probe.get('chat_template_kwargs_requested') or {}, ensure_ascii=False)}`",
            f"- no_system_role：`{probe.get('no_system_role')}`",
        ]
        if not probe.get("ok"):
            lines += [
                f"- 问题类型：`{probe.get('issue_type')}`",
                f"- 初步归因：`{probe.get('likely_cause')}`",
                f"- 错误：{str(probe.get('error', ''))[:1000]}",
            ]
        else:
            lines += [
                f"- rendered_length：`{probe.get('rendered_length')}`",
                f"- assistant_prefix_present：`{probe.get('assistant_prefix_present')}`",
            ]
        lines.append("")
    CONTINUATION_COMPAT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_wrapper_cmd(
    *,
    input_path: Path,
    tag: str,
    model_path: Path,
    gpus: str,
    plan: Plan,
    chat_template_kwargs: Dict[str, Any],
    no_system_role: bool,
    content_overrides: Dict[str, str],
    sampling_overrides: Dict[str, Any],
    max_cases: int,
    status_path: Path,
) -> List[str]:
    # Avoid starting beside unrelated GPU jobs. vLLM can OOM during weight
    # loading even when 30-40 GiB is free, so require essentially idle cards.
    wait_gpu_free_mib = 76000
    wait_gpu_max_util = 20
    cmd = [
        str(PY),
        str(WRAPPER),
        "--input-path",
        str(input_path.relative_to(REPO) if input_path.is_relative_to(REPO) else input_path),
        "--out-root",
        str(OUT_ROOT),
        "--tag",
        tag,
        "--model-path",
        str(model_path),
        "--gpus",
        gpus,
        "--tensor-parallel-size",
        str(plan.tp),
        "--max-model-len",
        str(plan.max_model_len),
        "--max-num-seqs",
        str(plan.max_num_seqs),
        "--max-num-batched-tokens",
        str(plan.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(plan.gpu_memory_utilization),
        "--max-cases",
        str(max_cases),
        "--wait-gpu-free-mib",
        str(wait_gpu_free_mib),
        "--wait-gpu-max-util",
        str(wait_gpu_max_util),
        "--wait-poll-seconds",
        "60",
        "--status-path",
        str(status_path.relative_to(REPO) if status_path.is_relative_to(REPO) else status_path),
        "--write-all-prompts",
    ]
    if plan.config_format:
        cmd += ["--config-format", plan.config_format]
    if plan.load_format:
        cmd += ["--load-format", plan.load_format]
    if plan.tokenizer_mode:
        cmd += ["--tokenizer-mode", plan.tokenizer_mode]
    if chat_template_kwargs:
        cmd += ["--chat-template-kwargs-json", json.dumps(chat_template_kwargs, ensure_ascii=False)]
    if no_system_role:
        cmd.append("--chat-template-no-system-role")
    if content_overrides.get("system_suffix"):
        cmd += ["--chat-template-system-suffix", content_overrides["system_suffix"]]
    if content_overrides.get("first_user_prefix"):
        cmd += ["--chat-template-first-user-prefix", content_overrides["first_user_prefix"]]
    if sampling_overrides.get("bad_words"):
        cmd += ["--sampling-bad-words-json", json.dumps(sampling_overrides["bad_words"], ensure_ascii=False)]
    if sampling_overrides.get("stop_token_ids"):
        cmd += ["--sampling-stop-token-ids-json", json.dumps(sampling_overrides["stop_token_ids"])]
    if sampling_overrides.get("ignore_eos"):
        cmd.append("--sampling-ignore-eos")
    return cmd


def run_dir(model_path: Path, tag: str) -> Path:
    return OUT_ROOT / f"{model_path.name}_{tag}"


ERROR_OR_RUNTIME_MARKERS = (
    "Traceback",
    "CUDA out of memory",
    "ImportError",
    "Exception:",
    "Error:",
)

CHAT_TEMPLATE_LEAK_MARKERS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "[INST]",
    "[/INST]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "<s>",
    "</s>",
)

MOJIBAKE_MARKERS = (
    "\ufffd",
    "Ã",
    "Â",
    "â€™",
    "â€œ",
    "â€",
    "å",
)


def output_format_issues(text: str) -> List[str]:
    issues: List[str] = []
    if any(marker in text for marker in ERROR_OR_RUNTIME_MARKERS):
        issues.append("runtime_error_text")
    if any(marker in text for marker in CHAT_TEMPLATE_LEAK_MARKERS):
        issues.append("chat_template_token_leak")
    if re.search(r"(?im)^\s*(assistant|user|system)\s*[:：]", text):
        issues.append("chat_role_marker_leak")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        issues.append("mojibake_or_replacement_char")
    control_chars = [
        ch for ch in text
        if (ord(ch) < 32 and ch not in "\n\r\t")
    ]
    if control_chars:
        issues.append("control_characters")
    return issues


def validate_artifact(run_dir_path: Path, input_path: Path, expected_rows: int) -> Dict[str, Any]:
    gen_file = run_dir_path / "gen_only.jsonl"
    cache_dir = run_dir_path / "packed_prompts/cache_prompts"
    all_cache = cache_dir / "ALL_cache.jsonl"
    result: Dict[str, Any] = {
        "run_dir": str(run_dir_path),
        "gen_file": str(gen_file),
        "gen_exists": gen_file.exists(),
        "expected_rows": expected_rows,
    }
    if not gen_file.exists():
        result["ok"] = False
        result["error"] = "missing gen_only.jsonl"
        return result

    input_rows = read_jsonl(input_path)
    input_ids = [(r.get("id") or r.get("annotation_uid") or r.get("case_id")) for r in input_rows]
    rows = []
    json_errors = []
    with gen_file.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                json_errors.append({"line": i, "error": str(exc)})

    ids = [(r.get("id") or r.get("annotation_uid") or r.get("case_id")) for r in rows]
    empty_outputs = []
    scored_empty_outputs = []
    suspicious_outputs = []
    format_issues = []
    prompt_mismatch = []
    total_output_count = 0
    for idx, row in enumerate(rows, 1):
        outputs = row.get("gen_output")
        prompts = row.get("prompts")
        prefixes = row.get("gen_prefix") or []
        upper = max(0, len(prefixes) - 2) if prefixes else max(0, len(outputs or []) - 2)
        if not isinstance(outputs, list):
            empty_outputs.append(row.get("id") or idx)
            scored_empty_outputs.append(row.get("id") or idx)
            outputs = []
        total_output_count += len(outputs)
        for out_idx, output in enumerate(outputs):
            output_text = str(output or "")
            if not output_text.strip():
                item = {"id": row.get("id") or idx, "output_idx": out_idx, "scored": out_idx < upper}
                empty_outputs.append(item)
                if out_idx < upper:
                    scored_empty_outputs.append(item)
                continue
            issues = output_format_issues(output_text)
            if issues:
                format_issues.append(
                    {
                        "id": row.get("id") or idx,
                        "output_idx": out_idx,
                        "issues": issues,
                        "preview": output_text[:240],
                    }
                )
        text = "\n".join(str(x) for x in outputs or [])
        if any(marker in text for marker in ERROR_OR_RUNTIME_MARKERS):
            suspicious_outputs.append(row.get("id") or idx)
        if isinstance(outputs, list) and isinstance(prompts, list) and len(outputs) != len(prompts):
            prompt_mismatch.append(row.get("id") or idx)

    case_files = list(cache_dir.glob("case_*_cache.jsonl")) if cache_dir.exists() else []
    all_cache_lines = line_count(all_cache)
    empty_output_ratio = (len(empty_outputs) / total_output_count) if total_output_count else 1.0
    empty_output_threshold = max(5, int(total_output_count * 0.15))
    excessive_empty_outputs = len(empty_outputs) > empty_output_threshold
    result.update(
        {
            "rows": len(rows),
            "json_error_count": len(json_errors),
            "unique_ids": len(set(ids)),
            "input_output_id_sets_equal": set(input_ids) == set(ids),
            "duplicate_id_count": len(ids) - len(set(ids)),
            "total_output_count": total_output_count,
            "empty_output_count": len(empty_outputs),
            "empty_output_ratio": round(empty_output_ratio, 4),
            "empty_output_threshold": empty_output_threshold,
            "excessive_empty_output_count": excessive_empty_outputs,
            "scored_empty_output_count": len(scored_empty_outputs),
            "suspicious_output_count": len(suspicious_outputs),
            "format_issue_count": len(format_issues),
            "prompt_mismatch_count": len(prompt_mismatch),
            "empty_output_examples": empty_outputs[:10],
            "scored_empty_output_examples": scored_empty_outputs[:10],
            "suspicious_output_examples": suspicious_outputs[:10],
            "format_issue_examples": format_issues[:10],
            "prompt_mismatch_examples": prompt_mismatch[:10],
            "all_cache_exists": all_cache.exists(),
            "all_cache_lines": all_cache_lines,
            "case_cache_files": len(case_files),
            "cache_total_files": len(list(cache_dir.glob("*.jsonl"))) if cache_dir.exists() else 0,
        }
    )
    result["ok"] = (
        len(rows) == expected_rows
        and len(rows) == len(input_rows)
        and len(set(ids)) == expected_rows
        and set(input_ids) == set(ids)
        and not json_errors
        and not scored_empty_outputs
        and not excessive_empty_outputs
        and not suspicious_outputs
        and not format_issues
        and not prompt_mismatch
        and result["all_cache_exists"]
        and len(case_files) == expected_rows
        and all_cache_lines > 0
    )
    return result


def split_input(input_path: Path, model_slug: str, shard_count: int) -> List[Path]:
    shard_dir = LOG_ROOT / "shards" / model_slug
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = input_path.read_text(encoding="utf-8").splitlines()
    shard_paths = [shard_dir / f"shard_{i:02d}.jsonl" for i in range(shard_count)]
    handles = [p.open("w", encoding="utf-8") for p in shard_paths]
    try:
        for idx, line in enumerate(rows):
            if line.strip():
                handles[idx % shard_count].write(line + "\n")
    finally:
        for h in handles:
            h.close()
    return shard_paths


def merge_shards(model_path: Path, model_slug: str, full_tag: str, shard_tags: List[str], input_path: Path) -> Tuple[Path, Dict[str, Any]]:
    combined_tag = f"{full_tag}_combined"
    combined_dir = run_dir(model_path, combined_tag)
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_gen = combined_dir / "gen_only.jsonl"
    with combined_gen.open("w", encoding="utf-8") as out:
        for tag in shard_tags:
            shard_gen = run_dir(model_path, tag) / "gen_only.jsonl"
            out.write(shard_gen.read_text(encoding="utf-8"))

    validation = validate_gen_only_only(combined_gen, input_path, line_count(input_path))
    (combined_dir / "merge_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not validation["ok"]:
        return combined_dir, validation

    pack_cmd = [
        str(PY),
        str(REPO / "tools/prompts/pack_prompt.py"),
        "--gen_file",
        str(combined_gen),
        "--out_dir",
        str(combined_dir / "packed_prompts"),
        "--write_all",
    ]
    code = run(pack_cmd, log=LOG_ROOT / f"{model_slug}_combined_pack.log")
    if code != 0:
        validation["ok"] = False
        validation["pack_error"] = f"pack_prompt exited {code}"
    return combined_dir, validation


def validate_gen_only_only(gen_file: Path, input_path: Path, expected_rows: int) -> Dict[str, Any]:
    input_rows = read_jsonl(input_path)
    input_ids = [(r.get("id") or r.get("annotation_uid") or r.get("case_id")) for r in input_rows]
    rows = read_jsonl(gen_file)
    ids = [(r.get("id") or r.get("annotation_uid") or r.get("case_id")) for r in rows]
    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_ids": len(set(ids)),
        "input_output_id_sets_equal": set(input_ids) == set(ids),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "ok": len(rows) == expected_rows and len(set(ids)) == expected_rows and set(input_ids) == set(ids),
    }


def run_model(rec: Dict[str, Any]) -> Dict[str, Any]:
    model_name = rec["model_name"]
    model_slug = safe_slug(model_name)
    result: Dict[str, Any] = {
        "model_name": model_name,
        "family": rec.get("family"),
        "tier": rec.get("tier"),
        "manifest_local_dir": rec.get("local_dir"),
        "resolved_path": rec.get("resolved_path"),
        "started_at_utc": now(),
    }
    write_status({"phase": "running_model", "model": model_name, "result_so_far": result})

    if not rec.get("exists"):
        result.update({"status": "missing_local_model", "ok": False})
        append_result_and_refresh(result)
        return result

    model_path = Path(str(rec["resolved_path"]))
    size_gib = dir_size_gib(model_path)
    result["size_gib"] = round(size_gib, 2)
    plan, blocked = choose_plan(rec, size_gib)
    if blocked or plan is None:
        result.update({"status": "blocked", "blocked_reason": blocked, "ok": False})
        append_result_and_refresh(result)
        return result
    result["plan"] = asdict(plan)
    chat_template_kwargs = chat_template_kwargs_for_model(rec)
    no_system_role = no_system_role_for_model(rec)
    content_overrides = chat_template_content_overrides_for_model(rec)
    sampling_overrides = sampling_overrides_for_model(rec)
    result["chat_template_kwargs"] = chat_template_kwargs
    result["chat_template_no_system_role"] = no_system_role
    result["chat_template_content_overrides"] = content_overrides
    result["sampling_overrides"] = sampling_overrides

    continuation_probe = run_continuation_probe(
        model_name,
        model_path,
        chat_template_kwargs,
        no_system_role=no_system_role,
        system_suffix=content_overrides.get("system_suffix", ""),
        first_user_prefix=content_overrides.get("first_user_prefix", ""),
    )
    refresh_continuation_compatibility_doc()
    result["continuation_probe"] = continuation_probe
    if not continuation_probe.get("ok"):
        result.update({"status": "preflight_continuation_failed", "ok": False})
        append_result_and_refresh(result)
        return result

    # Skip rerunning Granite if the previous full audited artifact exists.
    if model_name == "granite-4.1-8b":
        prior = REPO / "artifacts/model_outputs/completed_annotations/granite-4.1-8b_completed_annotations_full_granite_4_1_8b_4shard_combined"
        if prior.exists():
            val = validate_artifact(prior, FULL_INPUT, 317)
            if val.get("ok"):
                result.update({"status": "already_completed_prior", "full_validation": val, "ok": True})
                append_result_and_refresh(result)
                return result
            result["prior_validation_failed"] = val

    smoke_tag = f"manifest_smoke_{model_slug}"
    smoke_status = LOG_ROOT / f"{model_slug}_smoke_status.json"
    smoke_cmd = make_wrapper_cmd(
        input_path=TEST_INPUT,
        tag=smoke_tag,
        model_path=model_path,
        gpus=plan.gpu_groups[0],
        plan=Plan(**{**asdict(plan), "shard_count": 1, "gpu_groups": [plan.gpu_groups[0]]}),
        chat_template_kwargs=chat_template_kwargs,
        no_system_role=no_system_role,
        content_overrides=content_overrides,
        sampling_overrides=sampling_overrides,
        max_cases=10,
        status_path=smoke_status,
    )
    smoke_code = run(smoke_cmd, log=LOG_ROOT / f"{model_slug}_smoke.log")
    result["smoke_exit_code"] = smoke_code
    smoke_dir = run_dir(model_path, smoke_tag)
    smoke_validation = validate_artifact(smoke_dir, TEST_INPUT, 10) if smoke_code == 0 else {"ok": False, "status": load_json(smoke_status)}
    result["smoke_validation"] = smoke_validation
    if smoke_code != 0 or not smoke_validation.get("ok"):
        result.update({"status": "smoke_failed", "ok": False})
        append_result_and_refresh(result)
        return result

    full_tag = f"manifest_full_{model_slug}"
    expected_full = line_count(FULL_INPUT)
    if plan.shard_count == 1:
        status_path = LOG_ROOT / f"{model_slug}_full_status.json"
        full_cmd = make_wrapper_cmd(
            input_path=FULL_INPUT,
            tag=full_tag,
            model_path=model_path,
            gpus=plan.gpu_groups[0],
            plan=plan,
            chat_template_kwargs=chat_template_kwargs,
            no_system_role=no_system_role,
            content_overrides=content_overrides,
            sampling_overrides=sampling_overrides,
            max_cases=100000,
            status_path=status_path,
        )
        full_code = run(full_cmd, log=LOG_ROOT / f"{model_slug}_full.log")
        full_dir = run_dir(model_path, full_tag)
    else:
        shard_inputs = split_input(FULL_INPUT, model_slug, plan.shard_count)
        commands: List[Tuple[Sequence[str], Path]] = []
        shard_tags = []
        for i, shard_input in enumerate(shard_inputs):
            shard_tag = f"{full_tag}_shard{i:02d}"
            shard_tags.append(shard_tag)
            shard_plan = Plan(**{**asdict(plan), "shard_count": 1, "gpu_groups": [plan.gpu_groups[i]]})
            commands.append(
                (
                    make_wrapper_cmd(
                        input_path=shard_input,
                        tag=shard_tag,
                        model_path=model_path,
                        gpus=plan.gpu_groups[i],
                        plan=shard_plan,
                        chat_template_kwargs=chat_template_kwargs,
                        no_system_role=no_system_role,
                        content_overrides=content_overrides,
                        sampling_overrides=sampling_overrides,
                        max_cases=100000,
                        status_path=LOG_ROOT / f"{model_slug}_full_shard{i:02d}_status.json",
                    ),
                    LOG_ROOT / f"{model_slug}_full_shard{i:02d}.log",
                )
            )
        codes = run_many(commands)
        result["full_shard_exit_codes"] = codes
        if any(code != 0 for code in codes):
            result.update({"status": "full_shard_failed", "ok": False})
            append_result_and_refresh(result)
            return result
        full_dir, merge_validation = merge_shards(model_path, model_slug, full_tag, shard_tags, FULL_INPUT)
        result["merge_validation"] = merge_validation
        full_code = 0 if merge_validation.get("ok") else 1

    result["full_exit_code"] = full_code
    full_validation = validate_artifact(full_dir, FULL_INPUT, expected_full) if full_code == 0 else {"ok": False}
    result["full_validation"] = full_validation
    result.update({"status": "completed" if full_validation.get("ok") else "full_validation_failed", "ok": bool(full_validation.get("ok"))})
    append_result_and_refresh(result)
    return result


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = resolve_model_paths()
    allowlist_raw = os.environ.get("COMPLETED_MANIFEST_MODEL_ALLOWLIST", "").strip()
    allowlist = {
        item.strip()
        for item in re.split(r"[,\n]", allowlist_raw)
        if item.strip()
    }
    unknown_allowlist = sorted(allowlist - {m.get("model_name") for m in models})
    if allowlist:
        models = [m for m in models if m.get("model_name") in allowlist]
    available = [m for m in models if m.get("exists")]

    # Prioritize small and known-working models, then larger/special models.
    size_cache: Dict[str, float] = {}
    for m in available:
        try:
            size_cache[m["model_name"]] = dir_size_gib(Path(str(m["resolved_path"])))
        except Exception:
            size_cache[m["model_name"]] = 10**9
    ordered = sorted(available, key=lambda m: (size_cache.get(m["model_name"], 10**9), m["tier"] != "main", m["model_name"]))
    missing = [m for m in models if not m.get("exists")]

    plan_summary = {
        "phase": "starting",
        "total_manifest_models": len(models),
        "available_models": len(available),
        "missing_models": len(missing),
        "ordered_models": [m["model_name"] for m in ordered],
        "missing_model_names": [m["model_name"] for m in missing],
        "model_allowlist": sorted(allowlist),
        "unknown_allowlist": unknown_allowlist,
        "log_root": str(LOG_ROOT),
        "out_root": str(OUT_ROOT),
    }
    write_status(plan_summary)
    (LOG_ROOT / "resolved_manifest_models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOG_ROOT / "run_order.json").write_text(json.dumps(plan_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not SUMMARY_JSONL.exists():
        SUMMARY_JSONL.write_text("", encoding="utf-8")

    completed_names = set()
    if SUMMARY_JSONL.exists():
        for line in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") in {"completed", "already_completed_prior", "blocked", "missing_local_model", "preflight_continuation_failed", "smoke_failed", "full_validation_failed", "full_shard_failed"}:
                completed_names.add(row.get("model_name"))

    for m in missing:
        if m["model_name"] not in completed_names:
            append_result_and_refresh({"model_name": m["model_name"], "family": m.get("family"), "tier": m.get("tier"), "status": "missing_local_model", "ok": False, "manifest_local_dir": m.get("local_dir")})

    for idx, rec in enumerate(ordered, 1):
        if rec["model_name"] in completed_names:
            continue
        write_status({"phase": "running", "index": idx, "available_count": len(ordered), "current_model": rec["model_name"]})
        try:
            run_model(rec)
        except Exception as exc:
            row = {"model_name": rec.get("model_name"), "status": "orchestrator_exception", "ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            append_result_and_refresh(row)

    rows = [json.loads(line) for line in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = {
        "phase": "completed",
        "total_results": len(rows),
        "completed_ok": sum(1 for r in rows if r.get("ok") is True),
        "failed_or_blocked": sum(1 for r in rows if r.get("ok") is not True),
        "summary_jsonl": str(SUMMARY_JSONL),
    }
    refresh_continuation_compatibility_doc()
    write_status(final)


if __name__ == "__main__":
    main()
