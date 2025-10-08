from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
import json
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

# 允许的 JSON 空白： \t \n \r space；清理常见“非标准空白”
_JSON_WS_FIX = {
    "\u00A0": " ",   # NO-BREAK SPACE
    "\u2007": " ", "\u202F": " ",
    "\u2028": "\n", "\u2029": "\n"
}

def _normalize_ws(s: str) -> str:
    for k, v in _JSON_WS_FIX.items():
        s = s.replace(k, v)
    # 规整多余空行（可选）
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def _extract_first_json_obj(s: str) -> str:
    # 从第一个 '{' 开始做括号计数，拿到首个完整对象
    start = s.find("{")
    if start < 0:
        raise ValueError("No '{' found in model output.")
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    raise ValueError("Unclosed JSON object in model output.")

def safe_json_loads(s: str) -> dict:
    s = _normalize_ws(s or "")
    try:
        return json.loads(s)
    except Exception:
        obj = _extract_first_json_obj(s)
        return json.loads(obj)
