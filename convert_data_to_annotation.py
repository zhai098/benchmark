#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def transform_record(record: dict, index: int) -> dict:
    record_id = record.get("id")
    if record_id is None or str(record_id).strip() == "":
        record_id = f"q-{index}"

    question = record.get("question")
    if question is None or str(question).strip() == "":
        question = record.get("problem", "")

    solution = record.get("solution", "")

    known_solutions = record.get("known_solutions")
    if not isinstance(known_solutions, list):
        known_solutions = []

    return {
        "id": record_id,
        "question": question,
        "reference_answer": solution,
        "known_solutions": known_solutions,
        "samples": [{"solution": solution}],
    }


def convert_jsonl(input_path: Path, output_path: Path) -> None:
    total = 0
    converted = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            total += 1
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"输入文件第 {line_no} 行不是合法 JSON: {exc}") from exc

            out_record = transform_record(record, total)
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"转换完成：{converted} 条记录 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Omni_MATH 风格 JSONL 转为 annotation_app README 需要的 JSONL 格式")
    parser.add_argument("input", type=Path, help="输入 JSONL 路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 路径")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
