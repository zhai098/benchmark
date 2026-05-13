from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def case_id_to_index(case_id: str) -> int | None:
    if case_id.startswith("q-") and case_id[2:].isdigit():
        return int(case_id[2:])
    return None


def load_benchmark_cases(input_path: str | Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records[f"q-{idx}"] = payload
    return records


def _truthy_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _decision_is_correct(decision: Any) -> bool:
    return isinstance(decision, dict) and bool(decision.get("is_correct"))


def _normalize_steps(raw_steps: Any) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not isinstance(raw_steps, list):
        return steps
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, dict):
            text = _truthy_text(step.get("text"))
            step_id = _truthy_text(step.get("step_id")) or _truthy_text(step.get("id")) or f"s{idx}"
        else:
            text = _truthy_text(step)
            step_id = f"s{idx}"
        if not text:
            continue
        steps.append({"step_id": step_id, "text": text})
    return steps


def _normalize_claims(raw_claims: Any) -> List[Dict[str, Any]]:
    claims_by_step: List[Dict[str, Any]] = []
    if not isinstance(raw_claims, list):
        return claims_by_step

    for step_idx, step_entry in enumerate(raw_claims, start=1):
        if not isinstance(step_entry, dict):
            continue
        step_id = _truthy_text(step_entry.get("step_id")) or f"s{step_idx}"
        normalized_claims: List[Dict[str, str]] = []
        for claim_idx, claim_text in enumerate(step_entry.get("claims", []), start=1):
            text = _truthy_text(claim_text)
            if not text:
                continue
            normalized_claims.append({"id": f"{step_id}c{claim_idx}", "text": text})
        if normalized_claims:
            claims_by_step.append({"step_id": step_id, "claims": normalized_claims})
    return claims_by_step


def _normalize_dependencies(raw_dependencies: Any) -> Dict[str, List[str]]:
    if not isinstance(raw_dependencies, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for step_id, dependency_ids in raw_dependencies.items():
        if not isinstance(dependency_ids, list):
            continue
        normalized[str(step_id)] = [str(dep_id) for dep_id in dependency_ids if str(dep_id).strip()]
    return normalized


def _annotation_payload(
    annotation: Dict[str, Any] | None,
    *,
    sample_idx: int | None,
    solution_text: str = "",
) -> Dict[str, Any]:
    annotation = annotation or {}
    normalized_solution = _truthy_text(solution_text) or _truthy_text(annotation.get("selected_solution_text"))
    steps = _normalize_steps(annotation.get("steps"))
    claims_by_step = _normalize_claims(annotation.get("claims"))
    step_dependencies = _normalize_dependencies(annotation.get("step_dependencies"))
    return {
        "has_correct_sample": sample_idx is not None,
        "correct_sample_idx": sample_idx,
        "correct_sample_solution": normalized_solution,
        "steps": steps,
        "claims_by_step": claims_by_step,
        "step_dependencies": step_dependencies,
    }


def _empty_annotation_payload() -> Dict[str, Any]:
    return {
        "has_correct_sample": False,
        "correct_sample_idx": None,
        "correct_sample_solution": "",
        "steps": [],
        "claims_by_step": [],
        "step_dependencies": {},
    }


def _active_correct_sample(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    current_state = payload.get("current_workflow_state") or {}
    active_idx = current_state.get("active_sample_idx")
    sample_decisions = payload.get("sample_decisions") or []
    current_annotations = payload.get("current_annotations")
    if not isinstance(active_idx, int) or not isinstance(current_annotations, dict):
        return None
    if active_idx >= len(sample_decisions) or not _decision_is_correct(sample_decisions[active_idx]):
        return None
    if not (
        _truthy_text(current_annotations.get("selected_solution_text"))
        or current_annotations.get("steps")
        or current_annotations.get("claims")
    ):
        return None
    return _annotation_payload(
        current_annotations,
        sample_idx=active_idx,
    )


def _completed_correct_sample(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    current_annotations = payload.get("current_annotations") if isinstance(payload.get("current_annotations"), dict) else {}
    sample_annotations = current_annotations.get("sample_annotations") if isinstance(current_annotations, dict) else {}
    for item in payload.get("correct_solutions") or []:
        if not isinstance(item, dict):
            continue
        sample_idx = item.get("sample_idx")
        if not isinstance(sample_idx, int):
            continue
        annotation = sample_annotations.get(str(sample_idx)) if isinstance(sample_annotations, dict) else None
        return _annotation_payload(
            annotation if isinstance(annotation, dict) else None,
            sample_idx=sample_idx,
            solution_text=_truthy_text(item.get("solution")),
        )
    return None


def _other_correct_sample(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    current_annotations = payload.get("current_annotations") if isinstance(payload.get("current_annotations"), dict) else {}
    sample_annotations = current_annotations.get("sample_annotations") if isinstance(current_annotations, dict) else {}
    sample_decisions = payload.get("sample_decisions") or []
    if not isinstance(sample_annotations, dict):
        return None

    for sample_idx in sorted(sample_annotations.keys(), key=lambda value: int(value)):
        idx = int(sample_idx)
        if idx >= len(sample_decisions) or not _decision_is_correct(sample_decisions[idx]):
            continue
        annotation = sample_annotations.get(sample_idx)
        if not isinstance(annotation, dict):
            continue
        if not (
            _truthy_text(annotation.get("selected_solution_text"))
            or annotation.get("steps")
            or annotation.get("claims")
        ):
            continue
        return _annotation_payload(
            annotation,
            sample_idx=idx,
        )
    return None


def extract_annotation_reference(payload: Dict[str, Any]) -> Dict[str, Any]:
    for extractor in (_active_correct_sample, _completed_correct_sample, _other_correct_sample):
        result = extractor(payload)
        if result is not None:
            return result
    return _empty_annotation_payload()


def purify_annotations_folder(
    logs_dir: str | Path,
    *,
    benchmark_cases: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = Path(logs_dir)
    records: List[Dict[str, Any]] = []

    for path in sorted(base.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or path.stem)
        benchmark_row = (benchmark_cases or {}).get(case_id, {})
        annotation_reference = extract_annotation_reference(payload)
        record = {
            "case_id": case_id,
            "question": benchmark_row.get("problem", ""),
            "standard_solution": benchmark_row.get("solution", ""),
            "segments": benchmark_row.get("segments", []),
            **annotation_reference,
        }
        records.append(record)

    summary = {
        "logs_dir": str(base),
        "total_files": len(records),
        "with_correct_sample": sum(1 for item in records if item["has_correct_sample"]),
        "without_correct_sample": sum(1 for item in records if not item["has_correct_sample"]),
        "with_structured_annotation": sum(1 for item in records if item["steps"] and item["claims_by_step"]),
    }
    return records, summary


def step_id_at_index(record: Dict[str, Any], step_index: int) -> str | None:
    steps = record.get("steps") or []
    if step_index < len(steps) and isinstance(steps[step_index], dict):
        step_id = steps[step_index].get("step_id")
        if isinstance(step_id, str) and step_id:
            return step_id
    claims_by_step = record.get("claims_by_step") or []
    if step_index < len(claims_by_step) and isinstance(claims_by_step[step_index], dict):
        step_id = claims_by_step[step_index].get("step_id")
        if isinstance(step_id, str) and step_id:
            return step_id
    return None


def _claim_lookup(record: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for step_entry in record.get("claims_by_step", []) or []:
        for claim in step_entry.get("claims", []) or []:
            claim_id = str(claim.get("id") or "")
            if claim_id:
                lookup[claim_id] = claim
    return lookup


def claims_for_step(record: Dict[str, Any], step_index: int) -> List[Dict[str, str]]:
    target_step_id = step_id_at_index(record, step_index)
    if not target_step_id:
        return []
    for step_entry in record.get("claims_by_step", []) or []:
        if step_entry.get("step_id") == target_step_id:
            return list(step_entry.get("claims", []))
    return []


def dependency_claims_for_step(record: Dict[str, Any], step_index: int) -> List[Dict[str, str]]:
    target_step_id = step_id_at_index(record, step_index)
    if not target_step_id:
        return []
    dependency_ids = record.get("step_dependencies", {}).get(target_step_id, [])
    if not dependency_ids:
        return []
    lookup = _claim_lookup(record)
    return [lookup[claim_id] for claim_id in dependency_ids if claim_id in lookup]


def iter_purified_rows(records: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for record in records:
        yield json.dumps(record, ensure_ascii=False)
