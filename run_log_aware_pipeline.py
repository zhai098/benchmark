from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_core.config import Config
from benchmark_core.log_reference import iter_purified_rows, load_benchmark_cases, purify_annotations_folder


def _run(command: list[str]) -> None:
    print("[PIPELINE] running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the log-aware purification -> generate -> pack workflow.")
    parser.add_argument("--logs_dir", required=True, help="Annotation log folder")
    parser.add_argument("--benchmark_input", default=Config["Input_path"])
    parser.add_argument("--work_dir", default=None, help="Output working directory")
    parser.add_argument("--gen_file", default=None, help="Use an existing generation jsonl instead of calling generate.py")
    parser.add_argument("--skip_generate", action="store_true", help="Do not call generate.py")
    parser.add_argument("--use_vllm_local", action="store_true", help="Pass through to generate.py")
    parser.add_argument("--max_cases", type=int, default=100)
    parser.add_argument("--write_all_prompts", action="store_true")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else logs_dir / "workflow_outputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cases = load_benchmark_cases(args.benchmark_input)
    records, summary = purify_annotations_folder(logs_dir, benchmark_cases=benchmark_cases)

    purified_dir = work_dir / "purified"
    purified_dir.mkdir(parents=True, exist_ok=True)
    purified_cases_path = purified_dir / "purified_cases.jsonl"
    purified_summary_path = purified_dir / "purified_summary.json"
    purified_cases_path.write_text("\n".join(iter_purified_rows(records)) + ("\n" if records else ""), encoding="utf-8")
    purified_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[PIPELINE] total={summary['total_files']}, "
        f"with_correct_sample={summary['with_correct_sample']}, "
        f"without_correct_sample={summary['without_correct_sample']}"
    )
    print(f"[PIPELINE] purified rows: {purified_cases_path}")

    gen_file = Path(args.gen_file).resolve() if args.gen_file else None
    if not args.skip_generate and gen_file is None:
        gen_out_dir = work_dir / "gen_output"
        command = [
            sys.executable,
            "generate.py",
            "--input_path",
            str(purified_cases_path),
            "--out_root",
            str(gen_out_dir),
            "--tag",
            "log_aware",
            "--max_cases",
            str(args.max_cases),
        ]
        if args.use_vllm_local:
            command.append("--use_vllm_local")
        _run(command)
        run_dir = gen_out_dir / f"{Config['reasoning_model']}_log_aware"
        gen_file = run_dir / "gen_only.jsonl"

    if gen_file is None:
        raise ValueError("No generation file available. Provide --gen_file or omit --skip_generate.")

    pack_dir = work_dir / "packed_prompts"
    command = [
        sys.executable,
        "pack_prompt.py",
        "--gen_file",
        str(gen_file),
        "--out_dir",
        str(pack_dir),
        "--max_cases",
        str(args.max_cases),
    ]
    if args.write_all_prompts:
        command.append("--write_all")
    _run(command)

    print(f"[PIPELINE] completed. prompt cache dir: {pack_dir / 'cache_prompts'}")


if __name__ == "__main__":
    main()
