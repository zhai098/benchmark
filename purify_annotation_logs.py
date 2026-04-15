from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import Config
from log_reference import iter_purified_rows, load_benchmark_cases, purify_annotations_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract usable reference records from annotation logs.")
    parser.add_argument("--logs_dir", required=True, help="Annotation log folder to inspect.")
    parser.add_argument(
        "--benchmark_input",
        default=Config["Input_path"],
        help="Benchmark JSONL used only to attach question / standard solution / segments by q-<n>.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <logs_dir>/purified.",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else logs_dir / "purified"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cases = load_benchmark_cases(args.benchmark_input)
    records, summary = purify_annotations_folder(logs_dir, benchmark_cases=benchmark_cases)

    rows_path = out_dir / "purified_cases.jsonl"
    summary_path = out_dir / "purified_summary.json"

    rows_path.write_text("\n".join(iter_purified_rows(records)) + ("\n" if records else ""), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[PURIFY] logs_dir={logs_dir}")
    print(f"[PURIFY] total={summary['total_files']}")
    print(f"[PURIFY] with_correct_sample={summary['with_correct_sample']}")
    print(f"[PURIFY] without_correct_sample={summary['without_correct_sample']}")
    print(f"[PURIFY] with_structured_annotation={summary['with_structured_annotation']}")
    print(f"[PURIFY] rows={rows_path}")
    print(f"[PURIFY] summary={summary_path}")


if __name__ == "__main__":
    main()
