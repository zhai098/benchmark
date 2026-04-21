#!/usr/bin/env python3
"""
find_negative_scores.py

遍历指定的 `cases` 目录（递归），查找所有值为 -1 的评分项，并将结果输出为可读的文本报告和一个 JSON 文件。

用法示例：
  python find_negative_scores.py --cases-dir /path/to/run/cases --out report_dir

识别的分数字段包括（非穷尽）：
- 顶层 case 的 `score`
- `steps` 列表中每步的 `score`
- `steps[*].routes` 中的 `pairwise` / `holistic` 等
- `steps[*].judge_detail` 中嵌套的 `scores` 列表

输出：
- `negative_scores_report.txt`：人类可读的标注列表
- `negative_scores_report.json`：结构化的发现记录

"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def find_negatives_in_obj(obj: Any, path_prefix: str = "") -> List[Dict[str, Any]]:
    """递归检查 Python 对象中值为 -1 的位置，返回列表项包含路径和值的字典"""
    found: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path_prefix}.{k}" if path_prefix else k
            # 常见直接分数字段
            if isinstance(v, (int, float)) and float(v) == -1.0:
                found.append({"path": p, "value": v})
            else:
                found.extend(find_negatives_in_obj(v, p))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            p = f"{path_prefix}[{i}]"
            if isinstance(item, (int, float)) and float(item) == -1.0:
                found.append({"path": p, "value": item})
            else:
                found.extend(find_negatives_in_obj(item, p))

    # 其它类型（str等）忽略
    return found


def scan_cases_dir(cases_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in sorted(cases_dir.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        negs = find_negatives_in_obj(data)
        if negs:
            records.append({"file": str(p), "type": "json", "negatives": negs, "id": data.get("id")})

    for p in sorted(cases_dir.rglob("*.jsonl")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = [l for l in text.splitlines() if l.strip()]
        perfile: List[Dict[str, Any]] = []
        for i, line in enumerate(lines, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            negs = find_negatives_in_obj(obj)
            if negs:
                perfile.append({"line": i, "id": obj.get("id"), "negatives": negs})
        if perfile:
            records.append({"file": str(p), "type": "jsonl", "entries": perfile})

    return records


def write_reports(records: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "negative_scores_report.txt"
    json_path = out_dir / "negative_scores_report.json"

    with txt_path.open("w", encoding="utf-8") as f:
        if not records:
            f.write("No negative (-1) scores found.\n")
        for rec in records:
            f.write(f"File: {rec.get('file')}\n")
            if rec.get("type") == "json":
                f.write(f"  id: {rec.get('id')}\n")
                for neg in rec.get("negatives", []):
                    f.write(f"    - {neg['path']}: {neg['value']}\n")
            else:
                for entry in rec.get("entries", []):
                    f.write(f"  Line {entry['line']} id={entry.get('id')}\n")
                    for neg in entry.get("negatives", []):
                        f.write(f"    - {neg['path']}: {neg['value']}\n")
            f.write("\n")

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {txt_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Find -1 scores in cases folder and write report")
    parser.add_argument("--cases-dir", dest="cases_dir", default=None, help="Path to cases directory (defaults to ./cases)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory for reports (defaults to cases parent 'negative_reports')")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir) if args.cases_dir else Path.cwd() / "cases"
    if not cases_dir.exists() or not cases_dir.is_dir():
        raise SystemExit(f"cases dir not found: {cases_dir}")

    records = scan_cases_dir(cases_dir)

    default_out = cases_dir.parent / "negative_reports"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    write_reports(records, out_dir)


if __name__ == "__main__":
    main()
