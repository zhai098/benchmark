import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark_core.log_reference import (
    claims_for_step,
    dependency_claims_for_step,
    load_benchmark_cases,
    purify_annotations_folder,
    step_id_at_index,
)
from benchmark_core.paths import OMNI_MATH_DIR


LOGS_DIR = Path("annotation_app/data/annotations/___/dev-1775126623662-xze9d4")
WORKFLOW_OUTPUTS_DIR = LOGS_DIR / "workflow_outputs"
PURIFIED_CASES_PATH = WORKFLOW_OUTPUTS_DIR / "purified" / "purified_cases.jsonl"


def _records():
    benchmark_cases = load_benchmark_cases(str(OMNI_MATH_DIR / "Omni_MATH_Human_Segmented_100_1.jsonl"))
    return purify_annotations_folder(LOGS_DIR, benchmark_cases=benchmark_cases)


def _workflow_records():
    rows = []
    for line in PURIFIED_CASES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_purify_annotations_folder_keeps_all_questions():
    records, summary = _records()

    assert summary["total_files"] == 14
    assert len(records) == 14
    assert summary["with_correct_sample"] == 10
    assert summary["without_correct_sample"] == 4
    assert summary["with_structured_annotation"] >= 7

    record = next(item for item in records if item["case_id"] == "q-23")
    assert record["has_correct_sample"] is False
    assert record["correct_sample_solution"] == ""
    assert record["steps"] == []
    assert record["claims_by_step"] == []
    assert record["step_dependencies"] == {}
    assert record["question"]
    assert record["standard_solution"]


def test_active_correct_sample_can_supply_structure():
    records, _ = _records()
    record = next(item for item in records if item["case_id"] == "q-26")

    assert record["has_correct_sample"] is True
    assert record["correct_sample_idx"] == 2
    assert len(record["steps"]) == 14
    assert len(record["claims_by_step"]) == 14

    assert step_id_at_index(record, 2) == "s3"
    step_three_claims = claims_for_step(record, 2)
    assert step_three_claims[0]["id"] == "s3c1"

    dependency_claims = dependency_claims_for_step(record, 2)
    dependency_ids = [claim["id"] for claim in dependency_claims]
    assert dependency_ids == ["s1c1", "s2c3", "s2c1"]


def test_completed_correct_solution_prefers_completed_sample_annotation():
    records, _ = _records()
    record = next(item for item in records if item["case_id"] == "q-27")

    assert record["has_correct_sample"] is True
    assert record["correct_sample_idx"] == 1
    assert record["correct_sample_solution"]
    assert len(record["steps"]) == 16
    assert step_id_at_index(record, 0) == "s1"


def test_workflow_outputs_preserve_structured_step_claim_dependency_extraction_for_q21():
    raw = json.loads((LOGS_DIR / "q-21.json").read_text(encoding="utf-8"))
    workflow_record = next(item for item in _workflow_records() if item["case_id"] == "q-21")

    sample_annotation = raw["current_annotations"]["sample_annotations"]["0"]
    expected_steps = [
        {"step_id": step["id"], "text": step["text"]}
        for step in sample_annotation["steps"]
        if step.get("text")
    ]
    expected_claims = []
    for step_entry in sample_annotation["claims"]:
        normalized = []
        for claim_idx, claim_text in enumerate(step_entry.get("claims", []), start=1):
            if claim_text:
                normalized.append({"id": f"{step_entry['step_id']}c{claim_idx}", "text": claim_text})
        if normalized:
            expected_claims.append({"step_id": step_entry["step_id"], "claims": normalized})

    assert workflow_record["reference_quality"] == "structured"
    assert workflow_record["accepted_sample_idx"] == 0
    assert workflow_record["reference_steps"] == expected_steps
    assert workflow_record["reference_claims_by_step"] == expected_claims
    assert workflow_record["reference_step_dependencies"] == sample_annotation["step_dependencies"]


def test_workflow_outputs_keep_raw_fallback_for_cases_without_structured_steps():
    raw = json.loads((LOGS_DIR / "q-1.json").read_text(encoding="utf-8"))
    workflow_record = next(item for item in _workflow_records() if item["case_id"] == "q-1")

    assert raw["current_annotations"]["steps"] == []
    assert raw["current_annotations"]["sample_annotations"]["0"]["steps"] == []
    assert workflow_record["reference_quality"] == "raw_fallback"
    assert workflow_record["reference_steps"] == []
    assert workflow_record["reference_claims_by_step"] == []
    assert workflow_record["reference_step_dependencies"] == {}
