from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

from benchmark_core.paths import ANNOTATION_EXPORTS_DIR

BASE_PATH = ANNOTATION_EXPORTS_DIR / "annotation_data_with_claims_from_0413_gpt5.4rtn_high_difficulty_samples_acc_le_1_8.jsonl"
VERYHARD_PATH = ANNOTATION_EXPORTS_DIR / "annotation_data_reference_claims_quality_ge4_clear_sound_veryhard_plus.jsonl"
OUTPUT_PATH = ANNOTATION_EXPORTS_DIR / "annotation_data_veryhard_plus_with_7_samples.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_solution(text: str) -> str:
    return " ".join(str(text).split())


def main() -> None:
    base_rows = read_jsonl(BASE_PATH)
    veryhard_rows = read_jsonl(VERYHARD_PATH)
    base_by_id = {row["id"]: row for row in base_rows}

    merged_rows: list[dict] = []
    seen_ids: set[str] = set()

    for row in veryhard_rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"duplicate id in veryhard file: {row_id}")
        seen_ids.add(row_id)

        if row_id not in base_by_id:
            raise KeyError(f"missing base row for id={row_id}")

        base_row = base_by_id[row_id]
        if base_row["question"] != row["question"]:
            raise ValueError(f"question mismatch for id={row_id}")

        base_samples = list(base_row.get("samples", []))
        extra_samples = list(row.get("samples", []))

        if len(base_samples) != 6:
            raise ValueError(f"expected 6 base samples for id={row_id}, got {len(base_samples)}")
        if len(extra_samples) != 1:
            raise ValueError(f"expected 1 extra sample for id={row_id}, got {len(extra_samples)}")

        existing_solutions = {normalize_solution(sample.get("solution", "")) for sample in base_samples}
        extra_solution = normalize_solution(extra_samples[0].get("solution", ""))
        if extra_solution in existing_solutions:
            raise ValueError(f"duplicate solution detected for id={row_id}")

        merged = {
            "id": row_id,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "known_solutions": row.get("known_solutions", []),
            "samples": base_samples + extra_samples,
        }

        for optional_key in ("source", "difficulty", "domain", "question_accuracy"):
            if optional_key in row:
                merged[optional_key] = row[optional_key]

        if len(merged["samples"]) != 7:
            raise ValueError(f"expected 7 merged samples for id={row_id}, got {len(merged['samples'])}")

        for idx, sample in enumerate(merged["samples"]):
            if "solution" not in sample or "claims" not in sample:
                raise ValueError(f"sample {idx} for id={row_id} is missing solution/claims")

        merged_rows.append(merged)

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(merged_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
