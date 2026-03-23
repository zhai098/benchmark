#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path


def advance_pos(text: str, line: int, col: int):
    """根据 text 的内容推进 (line, col)，列从 1 开始计数。"""
    nl = text.count("\n")
    if nl == 0:
        return line, col + len(text)
    else:
        line += nl
        last = text.rsplit("\n", 1)[-1]
        col = 1 + len(last)
        return line, col


def lstrip_with_pos(s: str, line: int, col: int):
    """对 s 做 lstrip，同时正确更新 (line, col)。"""
    i = 0
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    if i:
        line, col = advance_pos(s[:i], line, col)
    return s[i:], line, col


def print_decode_error(buf: str, base_line: int, base_col: int, e: json.JSONDecodeError):
    # 绝对行列
    abs_line = base_line + e.lineno - 1
    if e.lineno == 1:
        abs_col = base_col + e.colno - 1
    else:
        abs_col = e.colno

    # 取出出错所在行（在当前 buf 内）
    pos = e.pos
    line_start = buf.rfind("\n", 0, pos) + 1
    line_end = buf.find("\n", pos)
    if line_end == -1:
        line_end = len(buf)
    err_line_text = buf[line_start:line_end]

    caret_pos = max(0, pos - line_start)

    print(f"[JSON 解析失败] 第 {abs_line} 行，第 {abs_col} 列：{e.msg}", file=sys.stderr)
    print("出错行内容：", file=sys.stderr)
    print(err_line_text, file=sys.stderr)
    print(" " * caret_pos + "^", file=sys.stderr)


def iter_json_values_stream(fin, chunk_size=1 << 20):
    """
    从文本流中解析一串 JSON 值（对象/数组/字符串/数字都行），允许值之间由任意空白分隔。
    """
    decoder = json.JSONDecoder()
    buf = ""
    base_line, base_col = 1, 1
    eof = False

    while True:
        if not eof:
            chunk = fin.read(chunk_size)
            if chunk == "":
                eof = True
            else:
                buf += chunk

        progressed = False

        while True:
            buf, base_line, base_col = lstrip_with_pos(buf, base_line, base_col)
            if not buf:
                break

            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError as e:
                # 如果还没 EOF，并且错误位置接近末尾，可能只是“还没读够”，继续读
                if not eof and e.pos >= max(0, len(buf) - 2):
                    break

                # 真坏了：打印错误并抛出
                print_decode_error(buf, base_line, base_col, e)
                raise

            # 成功解析一个值
            progressed = True
            parsed_text = buf[:idx]
            base_line, base_col = advance_pos(parsed_text, base_line, base_col)
            buf = buf[idx:]
            yield obj

        if eof:
            # EOF 后仍有残留非空白：也算错误
            if buf.strip():
                try:
                    decoder.raw_decode(buf.lstrip())
                except json.JSONDecodeError as e:
                    buf2, l2, c2 = lstrip_with_pos(buf, base_line, base_col)
                    print_decode_error(buf2, l2, c2, e)
                    raise
                raise ValueError("EOF 后残留内容不是合法 JSON。")
            break

        # 没推进且没 EOF：继续读更多
        if not progressed and not eof:
            continue


def main():
    ap = argparse.ArgumentParser(description="Convert multiline JSON stream into compact JSONL; print errors to stderr.")
    ap.add_argument("-i", "--input", required=True, help="input file path")
    ap.add_argument("-o", "--output", required=True, help="output jsonl file path")
    ap.add_argument("--ensure-ascii", action="store_true", help="escape non-ASCII characters (default: keep UTF-8)")
    ap.add_argument("--only-object", action="store_true",
                    help="only accept JSON objects (dict). Non-objects will be reported as error and stop.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    n = 0
    try:
        with in_path.open("r", encoding="utf-8-sig", errors="replace") as fin, \
             out_path.open("w", encoding="utf-8") as fout:
            for obj in iter_json_values_stream(fin):
                if args.only_object and not isinstance(obj, dict):
                    print(f"[类型错误] 解析到的第 {n+1} 个 JSON 值不是 object，而是 {type(obj).__name__}", file=sys.stderr)
                    raise ValueError("Non-object JSON value encountered.")
                fout.write(json.dumps(obj, ensure_ascii=args.ensure_ascii) + "\n")
                n += 1
    except Exception as e:
        print(f"[终止] 已成功写出 {n} 个对象到 {out_path}", file=sys.stderr)
        print(f"[异常] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote {n} JSON objects to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
