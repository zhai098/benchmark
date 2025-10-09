from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

import json, re, ast, os

class Processor():
    def __init__(self):
       
        self._ABBREVIATIONS = [
            # common English abbreviations
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Co.", "Inc.", "Ltd.",
            "vs.", "etc.", "e.g.", "i.e.", "cf.", "Fig.", "Eq.", "Sec.", "No.", "pp.", "Ch.",
            # academic/Latin
            "et al.", "al.",
            # locales
            "U.S.", "U.K.", "U.N."
        ]
        # map each abbreviation to a unique placeholder (no dots inside)
        self._ABBREV_MAP = {abbr: f"⟪ABBR{idx}⟫" for idx, abbr in enumerate(self._ABBREVIATIONS)}
        self._EN_SENT_END = re.compile(
            r'(?:(?<!\d)\.(?!\d)|[!?;])(?=[)"\'\]\}]*\s+|$)'
        )
    
    
    def sentence_split_en(self, text: str) -> List[str]:
        """
        English sentence splitter with abbreviation and number protection.
        Steps:
          1) Protect abbreviations by placeholder substitution.
          2) Split by sentence enders (. ! ? ;) with heuristics.
          3) Restore abbreviations.
        """
        if not text:
            return []

        # 1) protect abbreviations (longer first to avoid partial overlaps)
        tmp = text
        for abbr in sorted(self._ABBREV_MAP.keys(), key=len, reverse=True):
            tmp = tmp.replace(abbr, self._ABBREV_MAP[abbr])

        sentences: List[str] = []
        start = 0
        for m in self._EN_SENT_END.finditer(tmp):
            end = m.end()
            seg = tmp[start:end].strip()
            if seg:
                sentences.append(seg)
            start = end
        if start < len(tmp):
            rest = tmp[start:].strip()
            if rest:
                sentences.append(rest)

        # 3) restore abbreviations and clean spacing
        restored: List[str] = []
        for s in sentences:
            for abbr, placeholder in self._ABBREV_MAP.items():
                s = s.replace(placeholder, abbr)
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                restored.append(s)

        return restored

    def jaccard(a: str, b: str) -> float:
        A = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
        B = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    import json, re


# 允许的 JSON 空白范围内做规整：\t \n \r space；并清理常见“非标准空白/分隔”
_JSON_WS_FIX = {
    "\uFEFF": "",     # BOM
    "\u200B": "",     # ZERO WIDTH SPACE
    "\u200C": "", "\u200D": "", "\u2060": "",
    "\u00A0": " ",    # NO-BREAK SPACE
    "\u2007": " ", "\u202F": " ",
    "\u2028": "\n", "\u2029": "\n",
}

_SMART_QUOTES = {
    "“": "\"", "”": "\"", "„": "\"", "‟": "\"",
    "‘": "'",  "’": "'",  "‚": "'",  "‛": "'",
}

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)

def extract_floats(s):
    # 使用正则表达式提取浮点数
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    
    # 将匹配到的字符串转换为 float 类型
    return [float(num) for num in floats]

def _normalize_ws_and_quotes(s: str) -> str:
    s = s or ""
    for k, v in _JSON_WS_FIX.items():
        s = s.replace(k, v)
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    # 去除 Markdown 代码围栏
    s = _CODE_FENCE_RE.sub("", s)
    # 规整多余空行
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _extract_first_json_chunk(s: str) -> str | None:
    """从文本中提取首个完整 JSON 块（支持对象{}或数组[]），基于括号计数。"""
    if not s:
        return None
    # 找最早出现的 { 或 [
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [x for x in [start_obj, start_arr] if x >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = s[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None  # 未闭合

def _strip_trailing_commas(s: str) -> str:
    # 去掉 } 或 ] 前面多余逗号（尾逗号）
    return re.sub(r",\s*([}\]])", r"\1", s)

def _fix_unquoted_keys(s: str) -> str:
    # 将 { a: 1, b_c: 2 } 这类键名补双引号（尽量保守：仅匹配安全的标识符键）
    return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', s)

def _single_to_double_quotes(s: str) -> str:
    # 仅在“明显是 JSON 上下文”中把字符串的单引号换成双引号（尽量保守）
    # 先处理键名：'key': → "key":
    s = re.sub(r"([{\[,]\s*)'([^'\"\\]+?)'\s*:", r'\1"\2":', s)
    # 再处理字符串值：: 'value' → : "value"
    s = re.sub(r':\s*\'([^\'"\\]*?)\'(\s*[,\}\]])', r': "\1"\2', s)
    return s

def _replace_py_literals(s: str) -> str:
    # 把 Python 字面量替换为 JSON：True/False/None → true/false/null
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s

def _try_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def safe_json_loads(text: str) -> dict | list:
    """
    尽力把模型输出解析为 JSON（对象或数组）：
      1) 标准化空白/引号/去代码块
      2) 直接 json.loads
      3) 抽取首个 JSON 块再 loads
      4) 序列修复：尾逗号 → 单引号/无引号键 → Py 字面量 → 再抽取 → 再 loads
      5) 兜底：尝试 ast.literal_eval → 再转 JSON
    失败会抛出 ValueError
    """
    raw = _normalize_ws_and_quotes(text)

    # 直接尝试
    obj = _try_json_loads(raw)
    if obj is not None:
        return obj

    # 从混合文本中抽取首个 JSON 块再试
    chunk = _extract_first_json_chunk(raw)
    if chunk:
        obj = _try_json_loads(chunk)
        if obj is not None:
            return obj

    # 依次做可逆修复（每步都尝试解析）
    candidates = []
    s = raw

    s1 = _strip_trailing_commas(s)
    candidates.append(s1)

    s2 = _single_to_double_quotes(s1)
    candidates.append(s2)

    s3 = _fix_unquoted_keys(s2)
    candidates.append(s3)

    s4 = _replace_py_literals(s3)
    candidates.append(s4)

    for c in candidates:
        # 再抽取一次（以防修复后结构闭合）
        chunk2 = _extract_first_json_chunk(c) or c
        obj = _try_json_loads(chunk2)
        if obj is not None:
            return obj

    # 最后的兜底：literal_eval（宽松，但仅在安全上下文里使用）
    try:
        # 尽量转换成 Python 字面量可接受的形式
        s5 = _replace_py_literals(_single_to_double_quotes(_strip_trailing_commas(raw)))
        lit = ast.literal_eval(_extract_first_json_chunk(s5) or s5)
        # 再转成 JSON 兼容结构
        return json.loads(json.dumps(lit))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON after repairs: {e}")

def _to_str_atom(x: Any) -> str:
    """Make a single item a clean string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (dict, list, tuple)):
        # dict 用 JSON 形式，list/tuple 由外层处理
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()

def flatten_to_string(gen: Any, sep: str = " ") -> str:
    """
    Convert str / list / tuple / nested lists to a single string.
    - Trims whitespace
    - Flattens nested arrays
    - Skips empty atoms
    """
    out: list[str] = []

    def _walk(v: Any):
        if isinstance(v, (list, tuple)):
            for e in v:
                _walk(e)
        else:
            s = _to_str_atom(v)
            if s:
                out.append(s)

    _walk(gen)
    return sep.join(out)

def _write_jsonl_line(handle, payload: Dict[str, Any]) -> None:
    """写入单行 JSON 并立即落盘."""
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_pretty_json(handle, payload: Dict[str, Any]) -> None:
    """写入缩进 JSON，记录之间用空行分隔，方便人工浏览."""
    handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
    handle.write("\n\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_case_text_log(
    handle,
    *,
    case_record: Dict[str, Any],
    case_genlog: Dict[str, Any],
) -> None:
    """写入便于人工检视的纯文本摘要."""

    def _fmt_step(step: Dict[str, Any]) -> str:
        idx = step.get("index", "?")
        score = step.get("score", 0.0)
        step_score = step.get("step_score", score)
        hallucination = step.get("hallucination", 0)
        return (
            f"  - Step {idx}: score={float(score):.4f}, "
            f"weighted={float(step_score):.4f}, hallucination={int(hallucination)}"
        )

    def _fmt_generation(gen_output: List[Any]) -> List[str]:
        lines: List[str] = []
        for idx, item in enumerate(gen_output, 1):
            text = flatten_to_string(item, sep=" ") if item else ""
            if not text:
                text = "<empty>"
            lines.append(f"  [{idx}] {text}")
        return lines or ["  <no generation recorded>"]

    lines = [
        f"Case #{case_record.get('id', '?')}",
        f"Difficulty: {case_record.get('difficulty', '?')}",
        f"Weighted Score: {float(case_record.get('score', 0.0)):.4f}",
        f"Total Steps: {case_record.get('num_steps', 0)}",
        "Problem:",
        _to_str_atom(case_record.get("problem", "")),
        "Answer:",
        _to_str_atom(case_record.get("answer", "")),
        "Step Scores:",
    ]

    steps = case_record.get("steps") or []
    if steps:
        lines.extend(_fmt_step(step) for step in steps)
    else:
        lines.append("  <no step scores>")

    lines.append("Generated Outputs:")
    lines.extend(_fmt_generation(case_genlog.get("gen_output") or []))
    lines.append("-" * 80)

    handle.write("\n".join(lines) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
