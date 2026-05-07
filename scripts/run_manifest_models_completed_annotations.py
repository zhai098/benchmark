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
PY = Path("/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12")
MANIFEST = REPO / "model/download_manifest.tsv"
TEST_INPUT = REPO / "workflow_data/annotation_exports/completed_annotation_records_test_subset/purified_cases.jsonl"
FULL_INPUT = REPO / "workflow_data/annotation_exports/completed_annotation_records/purified_cases.jsonl"
OUT_ROOT = REPO / "artifacts/model_outputs/completed_annotations_manifest_models"
LOG_ROOT = REPO / "logs/completed_annotations_manifest_models"
STATUS_PATH = LOG_ROOT / "summary.json"
SUMMARY_JSONL = LOG_ROOT / "model_results.jsonl"
WRAPPER = REPO / "scripts/run_completed_annotations_generate_and_pack.py"


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

    return Plan(
        mode="shard4",
        tp=1,
        shard_count=4,
        gpu_groups=["0", "1", "2", "3"],
        max_model_len=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.35,
    ), None


def make_wrapper_cmd(
    *,
    input_path: Path,
    tag: str,
    model_path: Path,
    gpus: str,
    plan: Plan,
    max_cases: int,
    status_path: Path,
) -> List[str]:
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
        "30000",
        "--wait-gpu-max-util",
        "100",
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
    return cmd


def run_dir(model_path: Path, tag: str) -> Path:
    return OUT_ROOT / f"{model_path.name}_{tag}"


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
    prompt_mismatch = []
    for idx, row in enumerate(rows, 1):
        outputs = row.get("gen_output")
        prompts = row.get("prompts")
        prefixes = row.get("gen_prefix") or []
        upper = max(0, len(prefixes) - 2) if prefixes else max(0, len(outputs or []) - 2)
        if not isinstance(outputs, list):
            empty_outputs.append(row.get("id") or idx)
            scored_empty_outputs.append(row.get("id") or idx)
            outputs = []
        for out_idx, output in enumerate(outputs):
            if not str(output or "").strip():
                item = {"id": row.get("id") or idx, "output_idx": out_idx, "scored": out_idx < upper}
                empty_outputs.append(item)
                if out_idx < upper:
                    scored_empty_outputs.append(item)
        text = "\n".join(str(x) for x in outputs or [])
        if any(marker in text for marker in ("Traceback", "CUDA out of memory", "ImportError", "Exception:", "Error:")):
            suspicious_outputs.append(row.get("id") or idx)
        if isinstance(outputs, list) and isinstance(prompts, list) and len(outputs) != len(prompts):
            prompt_mismatch.append(row.get("id") or idx)

    case_files = list(cache_dir.glob("case_*_cache.jsonl")) if cache_dir.exists() else []
    all_cache_lines = line_count(all_cache)
    result.update(
        {
            "rows": len(rows),
            "json_error_count": len(json_errors),
            "unique_ids": len(set(ids)),
            "input_output_id_sets_equal": set(input_ids) == set(ids),
            "duplicate_id_count": len(ids) - len(set(ids)),
            "empty_output_count": len(empty_outputs),
            "scored_empty_output_count": len(scored_empty_outputs),
            "suspicious_output_count": len(suspicious_outputs),
            "prompt_mismatch_count": len(prompt_mismatch),
            "empty_output_examples": empty_outputs[:10],
            "scored_empty_output_examples": scored_empty_outputs[:10],
            "suspicious_output_examples": suspicious_outputs[:10],
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
        and not suspicious_outputs
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
        append_result(result)
        return result

    model_path = Path(str(rec["resolved_path"]))
    size_gib = dir_size_gib(model_path)
    result["size_gib"] = round(size_gib, 2)
    plan, blocked = choose_plan(rec, size_gib)
    if blocked or plan is None:
        result.update({"status": "blocked", "blocked_reason": blocked, "ok": False})
        append_result(result)
        return result
    result["plan"] = asdict(plan)

    # Skip rerunning Granite if the previous full audited artifact exists.
    if model_name == "granite-4.1-8b":
        prior = REPO / "artifacts/model_outputs/completed_annotations/granite-4.1-8b_completed_annotations_full_granite_4_1_8b_4shard_combined"
        if prior.exists():
            val = validate_artifact(prior, FULL_INPUT, 317)
            if val.get("ok"):
                result.update({"status": "already_completed_prior", "full_validation": val, "ok": True})
                append_result(result)
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
        append_result(result)
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
            append_result(result)
            return result
        full_dir, merge_validation = merge_shards(model_path, model_slug, full_tag, shard_tags, FULL_INPUT)
        result["merge_validation"] = merge_validation
        full_code = 0 if merge_validation.get("ok") else 1

    result["full_exit_code"] = full_code
    full_validation = validate_artifact(full_dir, FULL_INPUT, expected_full) if full_code == 0 else {"ok": False}
    result["full_validation"] = full_validation
    result.update({"status": "completed" if full_validation.get("ok") else "full_validation_failed", "ok": bool(full_validation.get("ok"))})
    append_result(result)
    return result


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = resolve_model_paths()
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
            if row.get("status") in {"completed", "already_completed_prior", "blocked", "missing_local_model", "smoke_failed", "full_validation_failed", "full_shard_failed"}:
                completed_names.add(row.get("model_name"))

    for m in missing:
        if m["model_name"] not in completed_names:
            append_result({"model_name": m["model_name"], "family": m.get("family"), "tier": m.get("tier"), "status": "missing_local_model", "ok": False, "manifest_local_dir": m.get("local_dir")})

    for idx, rec in enumerate(ordered, 1):
        if rec["model_name"] in completed_names:
            continue
        write_status({"phase": "running", "index": idx, "available_count": len(ordered), "current_model": rec["model_name"]})
        try:
            run_model(rec)
        except Exception as exc:
            row = {"model_name": rec.get("model_name"), "status": "orchestrator_exception", "ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            append_result(row)

    rows = [json.loads(line) for line in SUMMARY_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = {
        "phase": "completed",
        "total_results": len(rows),
        "completed_ok": sum(1 for r in rows if r.get("ok") is True),
        "failed_or_blocked": sum(1 for r in rows if r.get("ok") is not True),
        "summary_jsonl": str(SUMMARY_JSONL),
    }
    write_status(final)


if __name__ == "__main__":
    main()
