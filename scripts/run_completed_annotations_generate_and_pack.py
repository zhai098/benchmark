#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run generate.py on completed annotation records and pack judge prompts.

This wrapper keeps benchmark_core/config.py unchanged. It mutates the imported
Config object at runtime, waits for selected GPUs to become available, calls
generate.main(), then calls tools/prompts/pack_prompt.py on the produced
gen_only.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(kwargs)
    existing["updated_at_utc"] = now_iso()
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gpu_snapshot() -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=True)
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_total_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "utilization_gpu": int(parts[3]),
            }
        )
    return rows


def wait_for_gpus(gpus: str, min_free_mib: int, max_util: int, poll_seconds: int, status_path: Path) -> None:
    selected = {int(item) for item in gpus.split(",") if item.strip()}
    while True:
        snapshot = gpu_snapshot()
        selected_rows = [row for row in snapshot if row["index"] in selected]
        ready = (
            len(selected_rows) == len(selected)
            and all(row["memory_free_mib"] >= min_free_mib and row["utilization_gpu"] <= max_util for row in selected_rows)
        )
        write_status(
            status_path,
            phase="waiting_for_gpus" if not ready else "gpus_ready",
            selected_gpus=sorted(selected),
            gpu_snapshot=snapshot,
            min_free_mib=min_free_mib,
            max_util=max_util,
        )
        if ready:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run completed annotation generation and prompt packing.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--config-format", default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument("--tokenizer-mode", default=None)
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument("--chat-template-no-system-role", action="store_true")
    parser.add_argument("--max-cases", type=int, default=100000)
    parser.add_argument("--wait-gpu-free-mib", type=int, default=50000)
    parser.add_argument("--wait-gpu-max-util", type=int, default=10)
    parser.add_argument("--wait-poll-seconds", type=int, default=120)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--write-all-prompts", action="store_true")
    args = parser.parse_args()

    status_path = Path(args.status_path).resolve()
    try:
        write_status(status_path, phase="starting", input_path=args.input_path, out_root=args.out_root, tag=args.tag)
        wait_for_gpus(args.gpus, args.wait_gpu_free_mib, args.wait_gpu_max_util, args.wait_poll_seconds, status_path)

        from benchmark_core.config import Config

        vllm_config: Dict[str, Any] = {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": False,
        }
        if args.config_format:
            vllm_config["config_format"] = args.config_format
        if args.load_format:
            vllm_config["load_format"] = args.load_format
        if args.tokenizer_mode:
            vllm_config["tokenizer_mode"] = args.tokenizer_mode

        Config["reasoning_model"] = args.model_path
        Config["reasoning_model_params"] = vllm_config
        Config["reasoning_model_gpus"] = args.gpus
        Config["tag"] = args.tag
        try:
            chat_template_kwargs = json.loads(args.chat_template_kwargs_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --chat-template-kwargs-json: {exc}") from exc
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("--chat-template-kwargs-json must decode to an object")
        Config["generation_chat_template_kwargs"] = chat_template_kwargs
        Config["generation_chat_template_no_system_role"] = bool(args.chat_template_no_system_role)

        write_status(
            status_path,
            phase="generating",
            model_path=args.model_path,
            vllm_config=vllm_config,
            chat_template_kwargs=chat_template_kwargs,
            chat_template_no_system_role=bool(args.chat_template_no_system_role),
        )
        import generate

        sys.argv = [
            "generate.py",
            "--input_path",
            args.input_path,
            "--out_root",
            args.out_root,
            "--tag",
            args.tag,
            "--max_cases",
            str(args.max_cases),
            "--use_vllm_local",
        ]
        generate.main()

        run_dir = Path(args.out_root).resolve() / f"{Path(args.model_path).name}_{args.tag}"
        gen_file = run_dir / "gen_only.jsonl"
        if not gen_file.exists():
            raise FileNotFoundError(f"expected generation file not found: {gen_file}")

        write_status(status_path, phase="packing_prompts", gen_file=str(gen_file))
        pack_dir = run_dir / "packed_prompts"
        pack_command = [
            sys.executable,
            str(REPO_ROOT / "tools/prompts/pack_prompt.py"),
            "--gen_file",
            str(gen_file),
            "--out_dir",
            str(pack_dir),
        ]
        if args.write_all_prompts:
            pack_command.append("--write_all")
        subprocess.run(pack_command, cwd=str(REPO_ROOT), check=True)

        all_cache = pack_dir / "cache_prompts/ALL_cache.jsonl"
        case_files = list((pack_dir / "cache_prompts").glob("case_*_cache.jsonl"))
        write_status(
            status_path,
            phase="completed",
            gen_file=str(gen_file),
            pack_dir=str(pack_dir),
            all_cache=str(all_cache) if all_cache.exists() else None,
            packed_case_files=len(case_files),
        )
    except Exception as exc:
        write_status(status_path, phase="failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
