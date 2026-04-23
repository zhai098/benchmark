from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List


class Processor:
    def __init__(self) -> None:
        abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Co.", "Inc.", "Ltd.",
            "vs.", "etc.", "e.g.", "i.e.", "cf.", "Fig.", "Eq.", "Sec.", "No.", "pp.", "Ch.",
            "et al.", "al.",
            "U.S.", "U.K.", "U.N.",
        ]
        self._abbrev_map = {
            abbr: f"__ABBR_{idx}__"
            for idx, abbr in enumerate(sorted(abbreviations, key=len, reverse=True))
        }
        self._sentence_end_re = re.compile(
            r'(?:(?<!\d)\.(?!\d)|(?<=\d)\.(?=[)"\'\]\}]*\s+[A-Z])|[!?;])(?=[)"\'\]\}]*\s+|$)'
        )

    def sentence_split_en(self, text: Any) -> List[str]:
        """
        Split English text into sentences while protecting common abbreviations
        and decimal numbers. Returns a best-effort list and never raises.
        """
        normalized = _normalize_generation_input(text)
        if not normalized:
            return []

        tmp = normalized
        for abbr, placeholder in self._abbrev_map.items():
            tmp = tmp.replace(abbr, placeholder)

        sentences: List[str] = []
        start = 0
        for match in self._sentence_end_re.finditer(tmp):
            end = match.end()
            segment = tmp[start:end].strip()
            if segment:
                sentences.append(segment)
            start = end

        if start < len(tmp):
            tail = tmp[start:].strip()
            if tail:
                sentences.append(tail)

        restored: List[str] = []
        for sentence in sentences:
            for abbr, placeholder in self._abbrev_map.items():
                sentence = sentence.replace(placeholder, abbr)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if sentence:
                restored.append(sentence)
        return restored


_JSON_WS_FIX = {
    "\uFEFF": "",
    "\u200B": "",
    "\u200C": "",
    "\u200D": "",
    "\u2060": "",
    "\u00A0": " ",
    "\u2007": " ",
    "\u202F": " ",
    "\u2028": "\n",
    "\u2029": "\n",
}

_SMART_QUOTES = {
    "“": "\"",
    "”": "\"",
    "„": "\"",
    "‟": "\"",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
}

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')


def _normalize_ws_and_quotes(text: Any) -> str:
    raw = _normalize_generation_input(text)
    for src, dst in _JSON_WS_FIX.items():
        raw = raw.replace(src, dst)
    for src, dst in _SMART_QUOTES.items():
        raw = raw.replace(src, dst)
    raw = _CODE_FENCE_RE.sub("", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _extract_first_json_chunk(text: str) -> str | None:
    """Extract the first balanced JSON object/array while respecting strings."""
    if not text:
        return None

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None

    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False
    quote_char = ""

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_unquoted_keys(text: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', text)


def _single_to_double_quotes(text: str) -> str:
    text = re.sub(r"([{\[,]\s*)'([^'\"\\]+?)'\s*:", r'\1"\2":', text)
    text = re.sub(r':\s*\'([^\'"\\]*?)\'(\s*[,\}\]])', r': "\1"\2', text)
    return text


def _replace_py_literals(text: str) -> str:
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def safe_json_loads(text: Any) -> dict | list:
    """
    Best-effort JSON parser for noisy model output.

    Accepts strings, bytes, dicts, and lists. Dict/list inputs are returned
    directly. Raises ValueError only after all repair strategies fail.
    """
    if isinstance(text, (dict, list)):
        return text
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")

    raw = _normalize_ws_and_quotes(text)
    if not raw:
        raise ValueError("Failed to parse JSON after repairs: empty input")

    parsed = _try_json_loads(raw)
    if parsed is not None:
        return parsed

    chunk = _extract_first_json_chunk(raw)
    if chunk:
        parsed = _try_json_loads(chunk)
        if parsed is not None:
            return parsed

    candidates: List[str] = []
    current = raw
    for fixer in (_strip_trailing_commas, _single_to_double_quotes, _fix_unquoted_keys, _replace_py_literals):
        current = fixer(current)
        candidates.append(current)

    for candidate in candidates:
        parsed = _try_json_loads(candidate)
        if parsed is not None:
            return parsed
        chunk = _extract_first_json_chunk(candidate)
        if chunk:
            parsed = _try_json_loads(chunk)
            if parsed is not None:
                return parsed

    try:
        fallback = _replace_py_literals(_single_to_double_quotes(_strip_trailing_commas(raw)))
        literal = ast.literal_eval(_extract_first_json_chunk(fallback) or fallback)
        normalized = json.loads(json.dumps(literal, ensure_ascii=False))
        if isinstance(normalized, (dict, list)):
            return normalized
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON after repairs: {exc}") from exc

    raise ValueError("Failed to parse JSON after repairs: unsupported literal result")


def _to_str_atom(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return ""
    return str(value).strip()


def flatten_to_string(value: Any, sep: str = " ") -> str:
    """Flatten nested lists/tuples into a single string, skipping empty atoms."""
    parts: List[str] = []

    def _walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                _walk(child)
            return
        atom = _to_str_atom(item)
        if atom:
            parts.append(atom)

    _walk(value)
    return sep.join(parts)


def _flush_handle(handle: Any) -> None:
    try:
        handle.flush()
    except Exception:
        return

    fileno = getattr(handle, "fileno", None)
    if fileno is None:
        return
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError, ValueError):
        pass


def _write_jsonl_line(handle: Any, payload: Dict[str, Any]) -> None:
    """Write one JSONL row and flush when possible."""
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _flush_handle(handle)


def _write_pretty_json(handle: Any, payload: Dict[str, Any]) -> None:
    """Write indented JSON blocks separated by blank lines."""
    handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
    handle.write("\n\n")
    _flush_handle(handle)


def _write_case_text_log(
    handle: Any,
    *,
    case_record: Dict[str, Any],
    case_genlog: Dict[str, Any],
) -> None:
    """Write a human-readable text summary for manual inspection."""

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _fmt_step(step: Dict[str, Any]) -> str:
        idx = step.get("index", "?")
        score = _safe_float(step.get("score", 0.0))
        step_score = _safe_float(step.get("step_score", score))
        hallucination = _safe_int(step.get("hallucination", 0))
        return (
            f"  - Step {idx}: score={score:.4f}, "
            f"weighted={step_score:.4f}, hallucination={hallucination}"
        )

    def _fmt_generation(gen_output: List[Any]) -> List[str]:
        lines: List[str] = []
        for idx, item in enumerate(gen_output, start=1):
            text = flatten_to_string(item, sep=" ")
            lines.append(f"  [{idx}] {text or '<empty>'}")
        return lines or ["  <no generation recorded>"]

    lines = [
        f"Case #{case_record.get('id', '?')}",
        f"Difficulty: {case_record.get('difficulty', '?')}",
        f"Weighted Score: {_safe_float(case_record.get('score', 0.0)):.4f}",
        f"Total Steps: {_safe_int(case_record.get('num_steps', 0))}",
        "Problem:",
        _to_str_atom(case_record.get("problem", "")),
        "Answer:",
        _to_str_atom(case_record.get("answer", "")),
        "Step Scores:",
    ]

    steps = case_record.get("steps") or []
    if steps:
        lines.extend(_fmt_step(step) for step in steps if isinstance(step, dict))
    else:
        lines.append("  <no step scores>")

    lines.append("Generated Outputs:")
    lines.extend(_fmt_generation(case_genlog.get("gen_output") or []))
    lines.append("-" * 80)

    handle.write("\n".join(lines) + "\n")
    _flush_handle(handle)


def _normalize_generation_input(value: Any) -> str:
    """Convert arbitrary user/model outputs to stable plain text without flattening line structure."""

    def _normalize_text(raw: str) -> str:
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = raw.replace("\t", " ").replace("\f", " ").replace("\v", " ")
        raw = re.sub(r"[^\S\n]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        return raw.strip()

    if value is None:
        return ""
    if isinstance(value, bytes):
        return _normalize_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _normalize_text(value)
    try:
        return _normalize_text(flatten_to_string(value, sep=" "))
    except Exception:
        return _normalize_text(str(value))


def extract_last_score_part(text: Any) -> float:
    """
    Extract the last numeric `score` field from noisy text.

    Returns -1.0 when no score can be recovered.
    """
    raw = _normalize_generation_input(text)
    if not raw:
        return -1.0

    matches = _SCORE_RE.findall(raw)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return -1.0

    try:
        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
    except Exception:
        pass
    return -1.0


def extract_prefix(text: Any) -> str | None:
    """
    Extract a `prefix` field from raw model output such as:
      assistantfinal{"prefix":"..."}
    Returns None when parsing fails or the field is absent.
    """
    raw = _normalize_generation_input(text)
    if not raw:
        return None

    brace_idx = raw.find("{")
    candidates = [raw]
    if brace_idx >= 0:
        candidates.insert(0, raw[brace_idx:])

    for candidate in candidates:
        try:
            parsed = safe_json_loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            prefix = parsed.get("prefix")
            if prefix is None:
                continue
            normalized = _normalize_generation_input(prefix)
            return normalized or None
    return None
