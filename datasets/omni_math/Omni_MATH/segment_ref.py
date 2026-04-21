# -*- coding: utf-8 -*-
"""
Omni-Math solution processing utilities (English corpus)
-------------------------------------------------------
This module provides a class `OmniMathSolutionProcessor` to:
1) Filter long solutions first (dataset-level).
2) Then finely segment only those long solutions, with paragraph-first strategy.

Key requirements satisfied:
- Coarse segmentation = paragraph split (paragraphs separated by a single blank line; also tolerant to multiple).
- Text is English-oriented (sentence splitter & role hints in English).
- All helper methods have full docstrings and indicate where they are used.

Author: (copy & use directly)
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Iterable, Optional
import json
from datasets import load_dataset


class Dataset_Processor:
    """
    Processor for Omni-Math 'solution' texts:
    - Filter "long" solutions by multiple criteria (chars / tokens / sentences / paragraphs).
    - Finely segment long solutions into many small, well-ordered units, preserving display math blocks.

    Typical workflow:
        proc = OmniMathSolutionProcessor(solution_key="solution")
        long_items = proc.filter_long_solutions(dataset, min_paragraphs=3, min_chars=800)
        segmented = proc.process_dataset(
            dataset,
            filter_kwargs={"min_paragraphs": 3, "min_chars": 800},
            segment_kwargs={"force_max_chars": 220, "min_units": 14, "keep_math_block": True}
        )
    """

    # --------------------------
    # Configuration / constants
    # --------------------------

    # English abbreviations to protect in sentence splitting
    _ABBREV_MAP = {
        "e.g.": "e<dot>g<dot>",
        "i.e.": "i<dot>e<dot>",
        "etc.": "etc<dot>",
        "Mr.": "Mr<dot>",
        "Mrs.": "Mrs<dot>",
        "Ms.": "Ms<dot>",
        "Dr.": "Dr<dot>",
        "Prof.": "Prof<dot>",
        "vs.": "vs<dot>",
        "Fig.": "Fig<dot>",
        "Eq.": "Eq<dot>",
        "Prop.": "Prop<dot>",
        "Thm.": "Thm<dot>",
        "Cor.": "Cor<dot>",
        "No.": "No<dot>",
    }

    # Lightweight role hints (English), used by tag_role()
    _ROLE_HINTS = {
        "LowerBound": ["lower bound", "at least", "necessary", "must be ≥", "cannot be less"],
        "UpperBound/Construction": ["upper bound", "construction", "we can achieve", "it suffices to construct"],
        "Conclusion/Answer": ["therefore", "thus", "hence", "in conclusion", "we conclude", "\\boxed"],
        "CaseSplit": ["case", "otherwise", "when", "if", "suppose that"],
        "Claim": ["claim", "lemma", "proposition"],
        "Observation": ["observe that", "note that", "we note"],
        "Verification": ["verify", "check", "it remains to show", "it remains to verify"],
        "Inequality/Algebra": ["inequality", "rearranging", "algebraically", "by AM-GM", "by Cauchy"],
    }

    # Compiled regexes
    _WS_RE = re.compile(r"[ \t\r\f\v]+")
    _MULTI_BLANKLINES_RE = re.compile(r"\n\s*\n+")          # for paragraph splitting
    _EN_SENT_END = re.compile(r"([.!?;]+)(\s+|$)")           # English sentence boundaries
    _LATEX_ENV_BEGIN_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
    _LATEX_ENV_END_RE = re.compile(r"\\end\{([a-zA-Z*]+)\}")

    def __init__(self, *, solution_key: str = "solution") -> None:
        """
        Initialize the processor.

        Args:
            solution_key: the key name where solution text is stored inside dataset items.
        """
        self.solution_key = solution_key

    # --------------------------
    # Basic utilities
    # --------------------------

    def normalize_text(self, text: str) -> str:
        """
        Normalize whitespace without damaging LaTeX content.
        - Converts CRLF/CR to LF.
        - Collapses multiple horizontal spaces.
        - Trims leading/trailing blank lines.
        Used by: filter_long_solutions(), fine_grained_segment().
        """
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = self._WS_RE.sub(" ", t)
        # preserve double newlines for paragraphs; trim excessive blank lines at ends
        t = t.strip("\n ")
        return t

    def split_paragraphs(self, text: str) -> List[str]:
        """
        Coarse segmentation by paragraphs.
        A paragraph boundary is a single blank line; we also tolerate multiple blank lines.
        Returns a list of paragraphs with original order preserved.

        Used by: filter_long_solutions() (to count paragraphs),
                 fine_grained_segment() (as the first step of segmentation).
        """
        if not text:
            return []
        # Split on at least one blank line, keep only non-empty segments
        parts = self._MULTI_BLANKLINES_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def approx_token_count(self, text: str) -> int:
        """
        Rough token count for English text.
        Splits on whitespace and treats punctuation tokens as separate where possible.
        Used by: filter_long_solutions().
        """
        if not text:
            return 0
        # split on whitespace
        rough = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
        return len(rough)

    def split_display_math_blocks(self, text: str) -> List[tuple[str, str]]:
        """
        Split a paragraph into ("math"|"text", content) blocks.
        Treats the following as display-math blocks:
        - $$ ... $$
        - \[ ... \]
        - \\begin{env} ... \\end{env}  (env = common LaTeX environments, greedy per env)

        The order of blocks is preserved. This function does NOT split into sentences;
        it only separates math from surrounding text so later steps can keep math as standalone units.

        Used by: fine_grained_segment() (inside each paragraph).
        """
        if not text:
            return []

        # Normalize \[...\] -> $$...$$ (non-greedy), for uniform handling
        norm = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)

        # Find \begin{...}...\end{...} spans and replace with placeholders
        env_spans = []
        for m in self._LATEX_ENV_BEGIN_RE.finditer(norm):
            env_name = m.group(1)
            end_pat = re.compile(r"\\end\{" + re.escape(env_name) + r"\}")
            end_m = end_pat.search(norm, m.end())
            if end_m:
                env_spans.append((m.start(), end_m.end()))
        env_spans.sort()

        merged = []
        for s, e in env_spans:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        placeholders, out, cursor = [], [], 0
        for (s, e) in merged:
            out.append(norm[cursor:s])
            ph = f"@@LATEXENV{len(placeholders)}@@"
            out.append(ph)
            placeholders.append(norm[s:e])
            cursor = e
        out.append(norm[cursor:])
        stage = "".join(out)

        # Now split $$...$$
        tokens: List[tuple[str, str]] = []
        pattern = re.compile(r"\$\$(.+?)\$\$", flags=re.S)
        last = 0
        for m in pattern.finditer(stage):
            if m.start() > last:
                tokens.append(("text", stage[last:m.start()]))
            tokens.append(("math", m.group(0)))
            last = m.end()
        if last < len(stage):
            tokens.append(("text", stage[last:]))

        # Restore placeholders
        def putback(match):
            idx = int(match.group(1))
            return placeholders[idx]

        restored: List[tuple[str, str]] = []
        for typ, content in tokens:
            if typ == "text":
                content = re.sub(r"@@LATEXENV(\d+)@@", putback, content)
                if content:
                    restored.append((typ, content))
            else:
                restored.append((typ, content))
        # Trim
        cleaned = [(t, c.strip()) for (t, c) in restored if c and c.strip()]
        return cleaned

    def sentence_split_en(self, text: str) -> List[str]:
        """
        English sentence splitter with abbreviation protection.
        - Temporarily protects common abbreviations (Mr., e.g., etc., ...).
        - Splits by [. ! ? ;] (followed by space or EOS).
        - Restores abbreviations.
        Used by: fine_grained_segment() (inside each text block).
        """
        if not text:
            return []
        tmp = text
        for k, v in self._ABBREV_MAP.items():
            tmp = tmp.replace(k, v)

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

        restored = []
        for s in sentences:
            for k, v in self._ABBREV_MAP.items():
                s = s.replace(v, k)
            s = s.strip()
            if s:
                restored.append(s)
        return restored

    def soft_wrap_long_sentence(self, s: str, max_chars: int = 220) -> List[str]:
        """
        Soft-wrap a single long English sentence to increase unit count.
        Splits by commas/semicolons/colons and common coordinators (and/or/then/which),
        while trying to keep each piece under `max_chars`.

        Used by: fine_grained_segment() (to further split long sentences).
        """
        s = s.strip()
        if len(s) <= max_chars:
            return [s]

        # Split by preferred separators (keep separators attached to preceding chunk)
        # We split but keep semantic cohesion by re-joining up to max_chars
        parts = re.split(r"(,|;|:|\band\b|\bor\b|\bthen\b|\bwhich\b)", s)
        # Re-assemble with a greedy line-length cap
        out, cur = [], ""
        for p in parts:
            if p is None:
                continue
            candidate = (cur + p).strip()
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                if cur:
                    out.append(cur)
                cur = p.strip()
        if cur:
            out.append(cur)
        # Final clean
        out = [seg.strip(" ,;:") for seg in out if seg.strip(" ,;:")]
        return out or [s]


    # --------------------------
    # Stage 1: filter long solutions
    # --------------------------

    def filter_long_solutions(
        self,
        dataset: list[Dict[str, Any]],
        *,
        min_chars: Optional[int] = 800,
        min_tokens: Optional[int] = None,
        min_sentences: Optional[int] = None,
        min_paragraphs: Optional[int] = 3,
        count_math_as_sentence: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Select a subset of "long" solutions BEFORE fine segmentation.
        A sample is kept if it meets ANY of the provided thresholds.

        Args:
            dataset: iterable of dict-like items containing a solution field.
            min_chars: minimal character count (after normalization).
            min_tokens: minimal approximate token count (English).
            min_sentences: minimal sentence count (English). If `count_math_as_sentence=False`,
                           sentences are computed only on text parts (display math ignored).
            min_paragraphs: minimal paragraph count (paragraphs are separated by a blank line).
            count_math_as_sentence: whether to count display-math blocks as one sentence.

        Returns:
            A list of original dataset items that satisfy at least one "long" criterion.

        Used by: process_dataset() as the first stage.
        """
        kept: List[Dict[str, Any]] = []
        for item in dataset:
            sol = item.get(self.solution_key, "")
            if not isinstance(sol, str) or not sol.strip():
                continue

            sol_norm = self.normalize_text(sol)
            ok = False

            # Paragraphs
            """if not ok and min_paragraphs is not None:
                paras = self.split_paragraphs(sol_norm)
                if len(paras) >= min_paragraphs:
                    ok = True
            """
            # Characters
            if not ok and min_chars is not None:
                if len(sol_norm) >= min_chars:
                    ok = True

            # Tokens
            if not ok and min_tokens is not None:
                if self.approx_token_count(sol_norm) >= min_tokens:
                    ok = True

            # Sentences (English)
            if not ok and min_sentences is not None:
                # Count sentences on text-only or include math as 1 sentence each
                sentence_count = 0
                for para in self.split_paragraphs(sol_norm):
                    blocks = self.split_display_math_blocks(para)
                    for typ, content in blocks:
                        if typ == "text":
                            sentence_count += len(self.sentence_split_en(content))
                        elif count_math_as_sentence:
                            sentence_count += 1
                if sentence_count >= min_sentences:
                    ok = True

            if ok:
                kept.append(item)
        return kept

    # --------------------------
    # Stage 2: fine-grained segmentation (for long samples)
    # --------------------------

    def merge_short_units(
        self,
        units: List[Dict[str, Any]],
        *,
        min_unit_chars: int = 60,      # 阈值：小于此长度的 text 段必须合并
    ) -> List[Dict[str, Any]]:
        """
        Merge short TEXT segments within the same paragraph.
        - Only merges 'text' with adjacent 'text'.
        - 'math' units are never merged (hard boundary).
        - No cross-paragraph merging.
        - If a text segment is shorter than min_unit_chars, it must merge with a neighbor.
        - Reassigns ids after merging.
        """
        if not units:
            return []

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(units):
                cur = units[i]
                if cur["type"] != "text" or len(cur.get("content", "")) >= min_unit_chars:
                    i += 1
                    continue

                pid = cur["paragraph_index"]
                left = units[i - 1] if i - 1 >= 0 and units[i - 1]["paragraph_index"] == pid else None
                right = units[i + 1] if i + 1 < len(units) and units[i + 1]["paragraph_index"] == pid else None

                # 仅考虑 text 邻居
                left = left if left and left["type"] == "text" else None
                right = right if right and right["type"] == "text" else None

                if not left and not right:
                    # 没有可合并的 text 邻居，保留
                    i += 1
                    continue

                # 选择更短的邻居，若相等或缺失则优先右侧
                def seg_len(x): return len(x.get("content", "")) if x else 10**9
                if left and right:
                    pick_right = seg_len(right) <= seg_len(left)
                else:
                    pick_right = bool(right)

                neighbor = right if pick_right else left

                if pick_right:
                    cur["content"] = (cur["content"] + " " + neighbor["content"]).strip()
                    del units[i + 1]
                    changed = True
                else:
                    units[i - 1]["content"] = (neighbor["content"] + " " + cur["content"]).strip()
                    del units[i]
                    changed = True
                    i -= 1  # 回到合并后的位置
                # 不加 i++，因为要重新检查合并后的段

        # 重新编号 id
        for k, seg in enumerate(units):
            seg["id"] = k
        return units


    def fine_grained_segment(
        self,
        text: str,
        *,
        force_max_chars: int = 220,
        keep_math_block: bool = True,
        min_unit_chars: int = 60,              # 新增：最小片段长度阈值
   ) -> List[Dict[str, Any]]:
        """
        Finely segment a SINGLE solution into small, ordered units:
        1) Coarse split by paragraphs (blank line between paragraphs).
        2) Within each paragraph:
           - Separate display math blocks as standalone units (if keep_math_block=True).
           - English sentence splitting on text blocks.
           - Soft-wrap overlong sentences by commas/coordinators to boost unit count.

        Args:
            text: the raw solution string.
            force_max_chars: soft cap for sentence/segment length.
            keep_math_block: if True, display-math blocks stay as "math" units; otherwise treated as text.
            min_units: minimal desired number of segments; method will try to reach it.

        Returns:
            List of segments. Each segment is a dict:
            {
                "type": "text" | "math",
                "content": "...",
                "paragraph_index": int               # 0-based paragraph index
            }

        Used by: process_dataset() on the subset returned by filter_long_solutions().
        """
        text = self.normalize_text(text)
        paragraphs = self.split_paragraphs(text)

        units: List[Dict[str, Any]] = []
        for p_idx, para in enumerate(paragraphs):
            blocks = self.split_display_math_blocks(para)
            for typ, content in blocks:
                if typ == "math" and keep_math_block:
                    units.append({
                        "id": id,
                        "type": "math",
                        "content": content,
                        "paragraph_index": p_idx
                    })
                else:
                    # Treat math as text if keep_math_block=False
                    sentences = self.sentence_split_en(content)
                    for sent in sentences:
                        pieces = self.soft_wrap_long_sentence(sent, max_chars=force_max_chars)
                        for seg in pieces:
                            if seg:
                                units.append({
                                    "id": id,
                                    "type": "text",
                                    "content": seg,
                                    "paragraph_index": p_idx
                                })

        units = self.merge_short_units(
        units,
        min_unit_chars=min_unit_chars,
    )

        return len(units), units

    # --------------------------
    # Pipeline across a dataset
    # --------------------------

    def process_dataset(
        self,
        dataset: List[Dict[str, Any]],
        *,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        segment_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline:
        - Stage 1: filter long solutions with `filter_long_solutions(**filter_kwargs)`.
        - Stage 2: for those long items, run `fine_grained_segment(**segment_kwargs)`.

        Args:
            dataset: iterable of dict items containing a solution.
            filter_kwargs: arguments forwarded to filter_long_solutions().
            segment_kwargs: arguments forwarded to fine_grained_segment().

        Returns:
            A new list of items where each item includes:
            {
                **original_fields,
                "is_long": bool,
                "segments": [ ... ]     # only for long items; empty for others
            }
        """
        filter_kwargs = filter_kwargs or {}
        segment_kwargs = segment_kwargs or {}

        long_subset = self.filter_long_solutions(dataset, **filter_kwargs)
        long_ids = set(id(obj) for obj in long_subset)

        results: List[Dict[str, Any]] = []
        for item in dataset:
            sol = item.get(self.solution_key, "")
            if not isinstance(sol, str) or not sol.strip():
                continue
            out = dict(item)
            if id(item) in long_ids:
                out["segment_num"], out["segments"] = self.fine_grained_segment(sol, **segment_kwargs)
                results.append(out)
        return results


# --------------------------
# Example usage (commented)
# --------------------------
if __name__ == "__main__":
    
    dataset = load_dataset("KbsdJames/Omni-MATH",split="test")
    print(len(dataset))  
    dataset = dataset.to_list()
    proc = Dataset_Processor(solution_key="solution")

    # Stage 1 only: filter long solutions (≥2 paragraphs OR ≥700 chars)
    long_items = proc.filter_long_solutions(
        dataset,
        min_paragraphs=2,
        min_chars=700,
        min_tokens=None,
        min_sentences=None
    )
    print("Long items:", len(long_items))

    # Full pipeline: filter -> fine segmentation
    results = proc.process_dataset(
        dataset,
        filter_kwargs={"min_paragraphs": 2, "min_chars": 700},
        segment_kwargs={"force_max_chars": 200, "keep_math_block": True}
    )
    
    # 保存为JSONL文件
    with open('Omni_MATH_Long_Segmented_r.jsonl', 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
    
    with open('Omni_MATH_Long_Segmented.jsonl', 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')