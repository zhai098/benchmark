from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from log_reference import (
    claims_for_step,
    dependency_claims_for_step,
    load_benchmark_cases,
    purify_annotations_folder,
    step_id_at_index,
)


LOGS_DIR = Path("annotation_app/data/annotations/___/dev-1775126623662-xze9d4")


def _records():
    benchmark_cases = load_benchmark_cases("Omni_MATH/Omni_MATH_Human_Segmented_100_1.jsonl")
    return purify_annotations_folder(LOGS_DIR, benchmark_cases=benchmark_cases)


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
