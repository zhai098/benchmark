#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SENT_END_CHARS = set("。！？.!?\n")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Streaming JSONL reader.

    - Skips empty lines
    - Skips markdown code fences like ```jsonl / ```
    - Expects each remaining line to be a JSON object
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip("\ufeff\n\r")
            if not line.strip():
                continue
            if line.strip().startswith("```"):
                # common when a JSONL is pasted into markdown
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path} 第 {line_no} 行: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"JSON 类型错误: {path} 第 {line_no} 行不是 object，而是 {type(obj).__name__}")
            yield line_no, obj


def get_by_path(obj: Dict[str, Any], key_path: str) -> Any:
    cur: Any = obj
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key_path)
        cur = cur[part]
    return cur


def set_by_path(obj: Dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cur: Any = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def find_first_existing_key(obj: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        try:
            _ = get_by_path(obj, k)
            return k
        except KeyError:
            continue
    return None


def first_n_sentences(text: str, n: int = 10) -> Tuple[str, List[str]]:
    """Cut the first n sentences from text.

    Sentence boundary heuristic:
    - Ends on one of: 。！？.!? or newline
    - Keeps end punctuation
    """
    if n <= 0:
        return "", []

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    sentences: List[str] = []
    buf: List[str] = []

    def flush_sentence() -> None:
        nonlocal buf
        sent = "".join(buf).strip()
        buf = []
        if sent:
            sentences.append(sent)

    for ch in s:
        buf.append(ch)
        if ch in SENT_END_CHARS:
            flush_sentence()
            if len(sentences) >= n:
                break

    if len(sentences) < n and buf:
        flush_sentence()

    cut_text = "".join(sentences)
    return cut_text, sentences


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract first N sentences from each gen field in a JSONL and write to a new JSONL."
    )
    ap.add_argument("-i", "--input", required=True, help="input jsonl path")
    ap.add_argument("-o", "--output", required=True, help="output jsonl path")
    ap.add_argument("--n", type=int, default=10, help="number of sentences to keep (default: 10)")
    ap.add_argument(
        "--gen-key",
        default="",
        help=(
            "key path for gen text (supports dot path). "
            "If empty, will auto-detect from common keys."
        ),
    )
    ap.add_argument(
        "--out-key",
        default="gen_first10",
        help="where to write the cut text (dot path supported). default: gen_first10",
    )
    ap.add_argument(
        "--out-sentences-key",
        default="gen_first10_sentences",
        help="where to write the sentence list (dot path supported). default: gen_first10_sentences",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite gen field itself (writes cut text back to --gen-key)",
    )
    ap.add_argument("--ensure-ascii", action="store_true", help="escape non-ASCII in output")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    common_candidates = [
        "gen",
        "generation",
        "output",
        "text",
        "response",
        "content",
        "answer",
        "model_output",
        "result",
        "result.content",
        "choices.0.message.content",
    ]

    n_in = 0
    n_out = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for line_no, obj in iter_jsonl(in_path):
            n_in += 1

            gen_key = args.gen_key.strip()
            if not gen_key:
                gen_key = find_first_existing_key(obj, common_candidates) or ""

            if not gen_key:
                raise ValueError(
                    f"无法自动找到 gen 字段: {in_path} 第 {line_no} 行。"
                    f"请手动指定 --gen-key"
                )

            try:
                gen_val = get_by_path(obj, gen_key)
            except KeyError:
                raise ValueError(f"缺少 gen 字段 {gen_key!r}: {in_path} 第 {line_no} 行")

            if gen_val is None:
                gen_text = ""
            elif isinstance(gen_val, str):
                gen_text = gen_val
            else:
                # be forgiving
                gen_text = str(gen_val)

            cut_text, sentences = first_n_sentences(gen_text, n=args.n)

            if args.overwrite:
                set_by_path(obj, gen_key, cut_text)
            else:
                set_by_path(obj, args.out_key, cut_text)

            if args.out_sentences_key:
                set_by_path(obj, args.out_sentences_key, sentences)

            fout.write(json.dumps(obj, ensure_ascii=args.ensure_ascii) + "\n")
            n_out += 1

    print(f"Done. Read {n_in} lines, wrote {n_out} lines to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
