#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_process import safe_json_loads


def _read_jsonl(path: Path) -> List[Any]:
    rows: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_raw": line, "_line_no": line_no})
    return rows


def _extract_response_text(row: Any) -> str:
    if isinstance(row, str):
        return row

    if not isinstance(row, dict):
        return str(row)

    # Common simple keys
    for key in ["response", "content", "output", "text", "generated", "raw_output"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # OpenAI-like / chat-completions-like
    choices = row.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0]
        if isinstance(c0, dict):
            msg = c0.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
            content = c0.get("content")
            if isinstance(content, str) and content.strip():
                return content

    # tuple/list wrapper-like
    if isinstance(row.get("data"), str):
        return row["data"]

    return json.dumps(row, ensure_ascii=False)


def _normalize_judgment(text: str) -> Dict[str, str]:
    conclusion = "uncertain"
    explanation = "unable to parse response as required JSON"
    try:
        parsed = safe_json_loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        c = str(parsed.get("conclusion", "uncertain")).strip().lower()
        if c in {"yes", "no", "uncertain"}:
            conclusion = c
        e = str(parsed.get("explanation", "")).strip()
        if e:
            explanation = e

    return {"conclusion": conclusion, "explanation": explanation}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_case_steps(claims_dir: Path, case_id: Any) -> List[Dict[str, Any]]:
    case_path = claims_dir / f"{case_id}.json"
    if not case_path.exists():
        return []
    with case_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("steps", []) or []


def build_case_level_rows(pair_rows: List[Dict[str, Any]], claims_dir: Path) -> List[Dict[str, Any]]:
    by_case: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in pair_rows:
        by_case[r.get("case_id")].append(r)

    case_rows: List[Dict[str, Any]] = []

    for case_id, rows in sorted(by_case.items(), key=lambda x: (str(x[0]))):
        steps_src = _read_case_steps(claims_dir, case_id)

        # index pair comparisons by (step, claim_id, claim_text)
        comp_map: Dict[Tuple[int, Any, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            key = (
                int(r.get("step_index", 0)),
                r.get("current_claim_id"),
                str(r.get("current_claim_text", "")),
            )
            comp_map[key].append(
                {
                    "prior_step_index": r.get("prior_step_index"),
                    "prior_claim_id": r.get("prior_claim_id"),
                    "prior_claim_text": r.get("prior_claim_text"),
                    "conclusion": r.get("conclusion"),
                    "explanation": r.get("explanation"),
                    "response_text": r.get("response_text"),
                    "response_row_index": r.get("response_row_index"),
                }
            )

        out_steps: List[Dict[str, Any]] = []
        for step in steps_src:
            step_index = int(step.get("step_index", 0))
            step_text = step.get("step_text", "")
            src_claims = step.get("claims", []) or []

            if step_index <= 1:
                out_claims = [
                    {
                        "id": c.get("id"),
                        "text": c.get("text", ""),
                        "dependency_claims": [],
                        "num_dependency_claims": 0,
                    }
                    for c in src_claims
                ]
                out_steps.append(
                    {
                        "step_index": step_index,
                        "step_text": step_text,
                        "batch_evaluated": False,
                        "reason": "step_1_skipped",
                        "claims": out_claims,
                    }
                )
                continue

            out_claims = []
            total_pairs = 0
            for c in src_claims:
                key = (step_index, c.get("id"), str(c.get("text", "")))
                comps = comp_map.get(key, [])
                total_pairs += len(comps)

                deps = [
                    {
                        "step_index": cp.get("prior_step_index"),
                        "claim_id": cp.get("prior_claim_id"),
                        "text": cp.get("prior_claim_text"),
                    }
                    for cp in comps
                    if cp.get("conclusion") == "yes"
                ]

                out_claims.append(
                    {
                        "id": c.get("id"),
                        "text": c.get("text", ""),
                        "dependency_claims": deps,
                        "num_dependency_claims": len(deps),
                        "comparisons": comps,
                    }
                )

            out_steps.append(
                {
                    "step_index": step_index,
                    "step_text": step_text,
                    "batch_evaluated": True,
                    "num_pairs": total_pairs,
                    "claims": out_claims,
                }
            )

        case_rows.append(
            {
                "case_id": case_id,
                "num_steps": len(steps_src),
                "steps": out_steps,
            }
        )

    return case_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge packed claim-dependency messages with model responses into enriched JSONL."
    )
    parser.add_argument(
        "--messages",
        type=str,
        default="claim_dependency_messages.jsonl",
        help="Packed request JSONL created by pack_claim_dependency_messages.py",
    )
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Model response JSONL (line-by-line aligned with messages)",
    )
    parser.add_argument(
        "--output-pairs",
        type=str,
        default="claim_dependency_pairs_merged.jsonl",
        help="Merged pair-level JSONL output",
    )
    parser.add_argument(
        "--output-cases",
        type=str,
        default="claim_dependency_cases_merged.jsonl",
        help="Optional case-level aggregated JSONL output",
    )
    parser.add_argument(
        "--claims-dir",
        type=str,
        default="claims",
        help="Claims dir used to recover full step skeleton for case-level aggregation",
    )
    args = parser.parse_args()

    msg_rows = _read_jsonl(Path(args.messages).expanduser().resolve())
    rsp_rows = _read_jsonl(Path(args.responses).expanduser().resolve())

    n_msg = len(msg_rows)
    n_rsp = len(rsp_rows)
    n_use = min(n_msg, n_rsp)

    if n_msg != n_rsp:
        print(f"[WARN] messages={n_msg}, responses={n_rsp}, using first {n_use} aligned rows.")

    merged_pairs: List[Dict[str, Any]] = []
    for i in range(n_use):
        m = msg_rows[i] if isinstance(msg_rows[i], dict) else {}
        response_text = _extract_response_text(rsp_rows[i])
        judgment = _normalize_judgment(response_text)

        merged_pairs.append(
            {
                "row_index": i,
                "response_row_index": i,
                "case_id": m.get("case_id"),
                "step_index": m.get("step_index"),
                "batch_id": m.get("batch_id"),
                "pair_in_step": m.get("pair_in_step"),
                "current_claim_id": m.get("current_claim_id"),
                "current_claim_text": m.get("current_claim_text"),
                "prior_step_index": m.get("prior_step_index"),
                "prior_claim_id": m.get("prior_claim_id"),
                "prior_claim_text": m.get("prior_claim_text"),
                "conclusion": judgment["conclusion"],
                "explanation": judgment["explanation"],
                "response_text": response_text,
            }
        )

    write_jsonl(Path(args.output_pairs).expanduser().resolve(), merged_pairs)
    print(f"[DONE] wrote pair-level merged JSONL: {args.output_pairs} ({len(merged_pairs)} rows)")

    case_rows = build_case_level_rows(
        merged_pairs,
        claims_dir=Path(args.claims_dir).expanduser().resolve(),
    )
    write_jsonl(Path(args.output_cases).expanduser().resolve(), case_rows)
    print(f"[DONE] wrote case-level merged JSONL: {args.output_cases} ({len(case_rows)} cases)")


if __name__ == "__main__":
    main()
