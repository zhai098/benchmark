#!/usr/bin/env python3
"""
summarize_cases.py

遍历给定的 `cases` 文件夹，提取每个样本的 difficulty、num_steps、score 以及每一步的 score，
并将简洁的实验结果汇总为一个 JSON 文件写入 `cases` 上级目录（默认文件名 `cases_summary.json`）。

用法:
    python summarize_cases.py /path/to/run_folder/cases
或者直接指定 cases 文件夹:
    python summarize_cases.py --cases-dir /path/to/run_folder/cases

脚本会在 cases 的上级目录生成 `cases_summary.json`。
"""
import argparse
from pathlib import Path
import json
from typing import Any, Dict, List


def extract_summary_from_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    # 尝试从常见字段读取 difficulty, num_steps, score, per-step scores
    difficulty = obj.get("difficulty")

    # num_steps 可来自多种字段
    num_steps = obj.get("num_steps")
    if num_steps is None:
        if isinstance(obj.get("steps"), list):
            num_steps = len(obj.get("steps"))
        elif isinstance(obj.get("segments"), list):
            num_steps = len(obj.get("segments"))

    # score 优先常用字段
    score = None
    for fk in ("score", "final_score", "model_score", "evaluation_score"):
        if fk in obj:
            score = obj[fk]
            break

    # 提取每一步的 score，兼容多种结构
    step_scores: List[Any] = []
    if isinstance(obj.get("steps"), list):
        for s in obj.get("steps"):
            if isinstance(s, dict) and "score" in s:
                step_scores.append(s.get("score"))
            # 兼容 judge_detail 等嵌套场景
            elif isinstance(s, dict) and "routes" in s and isinstance(s.get("routes"), dict):
                # 尝试 pairwise/holistic 或直接 score
                if "pairwise" in s.get("routes") and isinstance(s.get("routes").get("pairwise"), (int, float)):
                    step_scores.append(s.get("routes").get("pairwise"))
                elif "holistic" in s.get("routes") and isinstance(s.get("routes").get("holistic"), (int, float)):
                    step_scores.append(s.get("routes").get("holistic"))
                else:
                    step_scores.append(None)
            else:
                step_scores.append(None)
    else:
        # 如果没有 steps 字段，但存在 per_step_scores 或 step_scores
        for fk in ("step_scores", "per_step_scores", "scores"):
            if fk in obj and isinstance(obj.get(fk), list):
                step_scores = obj.get(fk)
                break

    return {
        "id": obj.get("id") if isinstance(obj, dict) and "id" in obj else None,
        "difficulty": difficulty,
        "num_steps": num_steps,
        "score": score,
        "step_scores": step_scores,
    }


def summarize_cases(cases_dir: Path) -> List[Dict[str, Any]]:
    if not cases_dir.exists() or not cases_dir.is_dir():
        raise FileNotFoundError(f"cases dir not found: {cases_dir}")

    summaries: List[Dict[str, Any]] = []
    files = sorted([p for p in cases_dir.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}], key=lambda p: p.name)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            # 如果是 jsonl，逐行解析并取第一个对象（多数 cases 是单个 json）
            if f.suffix == ".jsonl":
                lines = [l for l in text.splitlines() if l.strip()]
                if not lines:
                    continue
                # 处理每行：若多行，尝试逐条添加（以防多样本文件）
                for i, line in enumerate(lines):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    summary = extract_summary_from_obj(obj)
                    # 记录来源文件与行号
                    summary["_source"] = f.name + (f":{i+1}" if len(lines) > 1 else "")
                    summaries.append(summary)
            else:
                obj = json.loads(text)
                summary = extract_summary_from_obj(obj)
                summary["_source"] = f.name
                summaries.append(summary)
        except Exception as e:
            # 忽略无法解析的文件，但在输出中记录警告条目
            summaries.append({"_source": f.name, "error": str(e)})

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Summarize cases folder into a compact JSON report")
    parser.add_argument("path", nargs="?", default=None, help="Path to a 'cases' folder or its parent run folder")
    parser.add_argument("--cases-dir", dest="cases_dir", help="Explicit path to the cases folder")
    parser.add_argument("--out", dest="out_name", default="case_summary.json", help="Output file name or path (default: cases_summary.json in parent folder)")
    args = parser.parse_args()

    if args.cases_dir:
        cases = Path(args.cases_dir)
    elif args.path:
        p = Path(args.path)
        # 如果用户传入的是 run_folder/cases 或者 run_folder
        if p.name == "cases":
            cases = p
        else:
            maybe = p / "cases"
            if maybe.exists() and maybe.is_dir():
                cases = maybe
            else:
                # 最后一种可能：用户直接传入文件
                raise SystemExit(f"Cannot find a 'cases' dir at {p} or {p / 'cases'}; you can pass --cases-dir explicitly")
    else:
        # 默认为当前目录下的 cases
        maybe = Path.cwd() / "cases"
        if maybe.exists() and maybe.is_dir():
            cases = maybe
        else:
            raise SystemExit("No cases directory found. Provide path or use --cases-dir")

    summaries = summarize_cases(cases)

    out_path = Path(args.out_name)
    if not out_path.is_absolute() and len(out_path.parts) == 1:
        out_path = cases.parent / out_path
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "case_summary.json"
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary for {len(summaries)} items to: {out_path}")


if __name__ == "__main__":
    main()
