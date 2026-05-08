from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = REPO_ROOT / "datasets"
WORKFLOW_DATA_DIR = REPO_ROOT / "workflow_data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

OMNI_MATH_DIR = DATASETS_DIR / "omni_math" / "Omni_MATH"
HIGH_DIFFICULTY_DATA_DIR = DATASETS_DIR / "high_difficulty"
SAMPLED_CASES_DIR = DATASETS_DIR / "sampled_cases"

CLAIMS_WORKFLOW_DIR = WORKFLOW_DATA_DIR / "claims"
QUESTION_QUALITY_DIR = WORKFLOW_DATA_DIR / "question_quality"
STRICT_DIFFICULTY_DIR = WORKFLOW_DATA_DIR / "strict_difficulty"
ANNOTATION_EXPORTS_DIR = WORKFLOW_DATA_DIR / "annotation_exports"
ANNOTATION_EXPORT_SPLITS_DIR = ANNOTATION_EXPORTS_DIR / "splits"
ALT_SOLUTIONS_DIR = WORKFLOW_DATA_DIR / "alt_solutions"
MANUAL_REVIEW_DIR = WORKFLOW_DATA_DIR / "manual_review"
PREFIX_WORKFLOW_DIR = WORKFLOW_DATA_DIR / "prefix"

ANALYSIS_ARTIFACTS_DIR = ARTIFACTS_DIR / "analysis"
MODEL_OUTPUTS_DIR = ARTIFACTS_DIR / "model_outputs"
PREVIEWS_DIR = ARTIFACTS_DIR / "previews"
ARCHIVES_DIR = ARTIFACTS_DIR / "archives"


def repo_rel(*parts: str) -> Path:
    """Convenience helper for repo-root-relative paths."""

    return REPO_ROOT.joinpath(*parts)
