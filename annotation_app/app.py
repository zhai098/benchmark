from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from flask import Flask, abort, g, jsonify, render_template, request, send_file, session

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANNOTATION_APP_DATA_DIR", str(BASE_DIR / "data"))).resolve()
RECORDS_DIR = DATA_DIR / "records"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
GUIDE_PATH = DATA_DIR / "guideline.md"
FRONTEND_OUT_DIR = BASE_DIR.parent / "frontend" / "out"

SCHEMA_VERSION = 2
WORKFLOW_STATES = {
    "sample_selected",
    "steps_segmented",
    "claims_assigned",
    "claims_checked",
    "dependencies_labeled",
    "completed",
}
PIPELINE_STATUSES = {"not_started", "ready", "in_progress", "completed", "discarded"}
CASE_STATUSES = {"in_progress", "completed"}
SCREENING_DECISIONS = {"pass", "reject"}
WORKFLOW_STATE_ORDER = {
    "sample_selected": 0,
    "steps_segmented": 1,
    "claims_assigned": 2,
    "claims_checked": 3,
    "dependencies_labeled": 4,
    "completed": 5,
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "annotation-app-dev-secret"


class ProgressLoadError(RuntimeError):
    pass


def logs_dir() -> Path:
    return DATA_DIR / "logs"


def configure_app_logging() -> None:
    if app.config.get("_logging_configured"):
        return

    log_dir = logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    app.config["app_log_path"] = log_dir / "app.log"
    app.config["access_log_path"] = log_dir / "access.log"
    app.config["_logging_configured"] = True


def write_json_log(logger_name: str, event: str, **fields: Any) -> None:
    configure_app_logging()
    payload = {"ts_utc": now_utc_iso(), "event": event, **fields}
    config_key = {"app_logger": "app_log_path", "access_logger": "access_log_path"}[logger_name]
    path = app.config[config_key]
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or ""


def current_role() -> str:
    role = str(session.get("role") or "annotator")
    return role if role in {"annotator", "reviewer"} else "annotator"


def require_reviewer() -> None:
    if current_role() != "reviewer":
        abort(403, description="reviewer access required")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    if not GUIDE_PATH.exists():
        GUIDE_PATH.write_text(
            "# 标注指南\n\n"
            "1. 先做多采样验证，再做 step/claim/依赖标注。\n"
            "2. 只切分，不修改原始文本。\n"
            "3. 依赖只能选前序 claim（同 step 前序 + 前面 steps）。\n"
            "4. 每个问题完成后再提交。\n",
            encoding="utf-8",
        )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_request_json() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        raw = request.get_data(cache=False, as_text=True)
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        return {}
    return payload


def safe_name(value: str, default: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", (value or "").strip())
    return sanitized or default


def progress_dir(annotator_id: str, device_id: str) -> Path:
    annotator_safe = safe_name(annotator_id, "unknown")
    device_safe = safe_name(device_id, "device")
    dir_path = ANNOTATIONS_DIR / annotator_safe / device_safe
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def progress_base_path(annotator_id: str, device_id: str, case_id: str) -> Path:
    case_safe = safe_name(case_id, "case")
    return progress_dir(annotator_id, device_id) / case_safe


def progress_path(annotator_id: str, device_id: str, case_id: str) -> Path:
    return progress_base_path(annotator_id, device_id, case_id).with_suffix(".json")


def progress_summary_path(annotator_id: str, device_id: str, case_id: str) -> Path:
    return progress_base_path(annotator_id, device_id, case_id).with_suffix(".summary.json")


def progress_detail_path(annotator_id: str, device_id: str, case_id: str) -> Path:
    return progress_base_path(annotator_id, device_id, case_id).with_suffix(".detail.json")


def _json_dumps_pretty(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _content_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_content_hash(data: dict[str, Any], *, skip_keys: set[str]) -> str:
    def _strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _strip(v) for k, v in value.items() if k not in skip_keys}
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    payload = _strip(data)
    return _content_hash(payload)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        payload = _json_dumps_pretty(data).encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ProgressLoadError(f"record at {path} is not a JSON object")
    return data


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_int(value: Any, default: int = 0, *, minimum: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out >= minimum else default


def _normalize_iso(value: Any, default: str) -> str:
    text = _normalize_string(value, default)
    return text or default


def _normalize_case_status(value: Any, default: str) -> str:
    text = _normalize_string(value, default)
    return text if text in CASE_STATUSES else default


def _normalize_workflow_state(value: Any, default: str = "sample_selected") -> str:
    text = _normalize_string(value, default)
    return text if text in WORKFLOW_STATES else default


def _normalize_pipeline_status(value: Any, default: str = "not_started") -> str:
    text = _normalize_string(value, default)
    return text if text in PIPELINE_STATUSES else default


def _normalize_screening(raw: Any) -> dict[str, Any]:
    obj = raw if isinstance(raw, dict) else {}
    decision_raw = obj.get("decision")
    decision = decision_raw if decision_raw in SCREENING_DECISIONS else None
    return {
        "decision": decision,
        "reason": _normalize_string(obj.get("reason"), ""),
        "other_text": _normalize_string(obj.get("other_text"), ""),
        "rejected_at": _normalize_string(obj.get("rejected_at"), ""),
    }


def _normalize_current_workflow_state(raw: Any) -> dict[str, Any]:
    obj = raw if isinstance(raw, dict) else {}
    active_sample_idx = obj.get("active_sample_idx")
    if active_sample_idx is None:
        active = None
    else:
        active = _normalize_int(active_sample_idx, default=0, minimum=0)
    return {
        "active_sample_idx": active,
        "sample_cursor": _normalize_int(obj.get("sample_cursor"), default=0, minimum=0),
        "workflow_state": _normalize_workflow_state(obj.get("workflow_state"), "sample_selected"),
        "problem_quality_screening": _normalize_screening(obj.get("problem_quality_screening")),
    }


def _normalize_claim_groups(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            step_id = _normalize_string(item.get("step_id"), f"s{idx + 1}")
            claims_raw = item.get("claims")
            if isinstance(claims_raw, list):
                claims = [_normalize_string(claim, "") for claim in claims_raw]
                claims = [claim for claim in claims if claim]
            else:
                claims = []
        elif isinstance(item, str):
            step_id = f"s{idx + 1}"
            claims = [_normalize_string(item, "")]
            claims = [claim for claim in claims if claim]
        else:
            continue
        out.append({"step_id": step_id, "claims": claims})
    return out


def _normalize_optional_step_idx(value: Any, fallback_step_id: Any = None) -> int | None:
    if value is not None:
        try:
            out = int(value)
        except (TypeError, ValueError):
            out = None
        if out is not None and out >= 0:
            return out
    step_text = _normalize_string(fallback_step_id, "")
    if not step_text:
        return None
    digits = re.sub(r"[^\d]", "", step_text)
    if not digits:
        return None
    try:
        parsed = int(digits)
    except ValueError:
        return None
    return parsed - 1 if parsed > 0 else None


def _parse_maybe_serialized_claim(value: str) -> Any | None:
    text = _normalize_string(value, "")
    if not text or text[:1] not in "{[":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _normalize_presegmented_claims(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []

    def _append_item(item: Any, default_id: str) -> None:
        if isinstance(item, str):
            parsed = _parse_maybe_serialized_claim(item)
            if parsed is not None:
                if isinstance(parsed, list):
                    for parsed_idx, parsed_item in enumerate(parsed):
                        _append_item(parsed_item, f"{default_id}_{parsed_idx + 1}")
                else:
                    _append_item(parsed, default_id)
                return
            text = _normalize_string(item, "")
            if text:
                out.append({"id": default_id, "text": text, "step_idx": None})
            return

        if not isinstance(item, dict):
            return

        if isinstance(item.get("claims"), list):
            step_idx = _normalize_optional_step_idx(item.get("step_index"), item.get("step_id"))
            for claim_idx, claim in enumerate(item.get("claims", [])):
                text = _normalize_string(claim, "")
                if text:
                    out.append(
                        {
                            "id": f"{default_id}_{claim_idx + 1}",
                            "text": text,
                            "step_idx": step_idx,
                        }
                    )
            return

        text = _normalize_string(item.get("text") or item.get("claim"), "")
        if not text:
            return
        out.append(
            {
                "id": _normalize_string(item.get("id"), default_id),
                "text": text,
                "step_idx": _normalize_optional_step_idx(
                    item.get("step_idx") if "step_idx" in item else item.get("step_index"),
                    item.get("step_id"),
                ),
            }
        )

    for idx, item in enumerate(raw):
        _append_item(item, f"p{idx + 1}")

    return out


def _normalize_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [text for text in (_normalize_string(x, "") for x in raw) if text]


def _normalize_mapping_of_str_lists(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        norm_key = _normalize_string(key, "")
        if not norm_key:
            continue
        norm_vals = _normalize_str_list(value)
        if norm_vals:
            out[norm_key] = norm_vals
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _normalize_claim_checks(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        norm_key = _normalize_string(key, "")
        norm_value = _normalize_string(value, "")
        if not norm_key or not norm_value:
            continue
        out[norm_key] = norm_value
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _annotation_has_meaningful_content(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    fields = (
        raw.get("selected_solution_text"),
        raw.get("cut_points"),
        raw.get("steps"),
        raw.get("presegmented_claims"),
        raw.get("claims"),
        raw.get("claim_checks"),
        raw.get("dependencies"),
        raw.get("step_dependencies"),
    )
    return any(bool(field) for field in fields)


def _normalize_sample_annotation(raw: Any, *, default_workflow_state: str = "sample_selected", now: str) -> dict[str, Any]:
    obj = raw if isinstance(raw, dict) else {}
    cut_points: list[int] = []
    raw_cut_points = obj.get("cut_points", [])
    if isinstance(raw_cut_points, list):
        for value in raw_cut_points:
            norm = _normalize_int(value, default=-1, minimum=0)
            if norm >= 0:
                cut_points.append(norm)
    return {
        "selected_solution_text": _normalize_string(obj.get("selected_solution_text"), ""),
        "cut_points": sorted(set(cut_points)),
        "steps": _normalize_str_list(obj.get("steps")),
        "presegmented_claims": _normalize_presegmented_claims(obj.get("presegmented_claims")),
        "claims": _normalize_claim_groups(obj.get("claims")),
        "claim_checks": _normalize_claim_checks(obj.get("claim_checks")),
        "dependencies": _normalize_mapping_of_str_lists(obj.get("dependencies")),
        "step_dependencies": _normalize_mapping_of_str_lists(obj.get("step_dependencies")),
        "workflow_state": _normalize_workflow_state(obj.get("workflow_state"), default_workflow_state),
        "updated_at_utc": _normalize_iso(obj.get("updated_at_utc"), now),
    }


def _merge_legacy_active_annotation(
    sample_annotations: dict[str, dict[str, Any]],
    raw_current_annotations: dict[str, Any],
    current_workflow_state: dict[str, Any],
    now: str,
) -> dict[str, dict[str, Any]]:
    if not _annotation_has_meaningful_content(raw_current_annotations):
        return sample_annotations

    target_idx = current_workflow_state.get("active_sample_idx")
    if target_idx is None:
        target_idx = current_workflow_state.get("sample_cursor")
    if target_idx is None:
        return sample_annotations

    target_key = str(_normalize_int(target_idx, default=0, minimum=0))
    if target_key in sample_annotations:
        return sample_annotations

    sample_annotations[target_key] = _normalize_sample_annotation(
        raw_current_annotations,
        default_workflow_state=current_workflow_state.get("workflow_state", "sample_selected"),
        now=now,
    )
    return sample_annotations


def _normalize_sample_annotations(raw_current_annotations: dict[str, Any], current_workflow_state: dict[str, Any], now: str) -> dict[str, dict[str, Any]]:
    raw = raw_current_annotations.get("sample_annotations")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            idx = _normalize_int(key, default=0, minimum=0)
            out[str(idx)] = _normalize_sample_annotation(
                value,
                default_workflow_state=current_workflow_state.get("workflow_state", "sample_selected"),
                now=now,
            )
    out = dict(sorted(out.items(), key=lambda kv: int(kv[0])))
    return _merge_legacy_active_annotation(out, raw_current_annotations, current_workflow_state, now)


def _normalize_correct_solutions(raw: Any, sample_annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    dedup: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        sample_idx = _normalize_int(item.get("sample_idx"), default=0, minimum=0)
        solution = _normalize_string(item.get("solution"), "")
        completed_at = _normalize_string(item.get("completed_at"), "")
        sample_ann = sample_annotations.get(str(sample_idx), {})
        workflow_state_at_accept = _normalize_workflow_state(
            item.get("workflow_state_at_accept"),
            sample_ann.get("workflow_state", "completed"),
        )
        content_hash = _normalize_string(item.get("content_hash"), "")
        if not content_hash:
            hash_source = solution or sample_ann.get("selected_solution_text", "")
            content_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest() if hash_source else ""
        dedup[sample_idx] = {
            "sample_idx": sample_idx,
            "solution": solution,
            "completed_at": completed_at,
            "workflow_state_at_accept": workflow_state_at_accept,
            "content_hash": content_hash,
        }
    return [dedup[idx] for idx in sorted(dedup)]


def _normalize_sample_decisions(
    raw: Any,
    *,
    sample_annotations: dict[str, dict[str, Any]],
    current_workflow_state: dict[str, Any],
    correct_solutions: list[dict[str, Any]],
    now: str,
) -> list[dict[str, Any]]:
    raw_list = raw if isinstance(raw, list) else []
    max_idx = -1
    if raw_list:
        max_idx = max(max_idx, len(raw_list) - 1)
    if sample_annotations:
        max_idx = max(max_idx, max(int(k) for k in sample_annotations))
    if correct_solutions:
        max_idx = max(max_idx, max(item["sample_idx"] for item in correct_solutions))
    active_idx = current_workflow_state.get("active_sample_idx")
    if active_idx is not None:
        max_idx = max(max_idx, active_idx)
    out: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        raw_item = raw_list[idx] if idx < len(raw_list) and isinstance(raw_list[idx], dict) else {}
        is_correct = raw_item.get("is_correct")
        if not isinstance(is_correct, bool):
            is_correct = None
        pipeline_status = _normalize_pipeline_status(raw_item.get("pipeline_status"), "not_started")
        if str(idx) in sample_annotations and is_correct is True and pipeline_status == "not_started":
            pipeline_status = "ready"
        if str(idx) in sample_annotations and current_workflow_state.get("active_sample_idx") == idx:
            pipeline_status = "in_progress"
        if any(item["sample_idx"] == idx for item in correct_solutions):
            pipeline_status = "completed"

        out.append(
            {
                "is_correct": is_correct,
                "class_name": _normalize_string(raw_item.get("class_name"), ""),
                "is_new_class": bool(raw_item.get("is_new_class", False)),
                "summary": _normalize_string(raw_item.get("summary"), ""),
                "pipeline_status": pipeline_status,
                "decided_at_utc": _normalize_string(raw_item.get("decided_at_utc"), ""),
                "updated_at_utc": _normalize_iso(raw_item.get("updated_at_utc"), now),
            }
        )
    return out


def _count_claims(claim_groups: Any) -> int:
    if not isinstance(claim_groups, list):
        return 0
    total = 0
    for item in claim_groups:
        if isinstance(item, dict):
            claims = item.get("claims")
            if isinstance(claims, list):
                total += len(claims)
        elif isinstance(item, str):
            total += 1
    return total


def _count_dependencies(dependencies: Any) -> int:
    if not isinstance(dependencies, dict):
        return 0
    total = 0
    for value in dependencies.values():
        if isinstance(value, list):
            total += len(value)
    return total


def _summarize_samples(
    sample_annotations: dict[str, dict[str, Any]],
    sample_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    sample_rows: list[dict[str, Any]] = []
    total_steps = 0
    total_claims = 0
    total_dependencies = 0
    sample_stats = {
        "total_samples": len(sample_decisions),
        "correct_samples": 0,
        "completed_samples": 0,
        "discarded_samples": 0,
        "ready_samples": 0,
        "in_progress_samples": 0,
    }

    for idx, decision in enumerate(sample_decisions):
        ann = sample_annotations.get(str(idx), {})
        step_count = len(ann.get("steps", [])) if isinstance(ann, dict) else 0
        claim_count = _count_claims(ann.get("claims", [])) if isinstance(ann, dict) else 0
        dependency_count = _count_dependencies(ann.get("dependencies", {})) if isinstance(ann, dict) else 0
        total_steps += step_count
        total_claims += claim_count
        total_dependencies += dependency_count

        if decision.get("is_correct") is True:
            sample_stats["correct_samples"] += 1
        status = decision.get("pipeline_status")
        if status == "completed":
            sample_stats["completed_samples"] += 1
        elif status == "discarded":
            sample_stats["discarded_samples"] += 1
        elif status == "ready":
            sample_stats["ready_samples"] += 1
        elif status == "in_progress":
            sample_stats["in_progress_samples"] += 1

        sample_rows.append(
            {
                "sample_idx": idx,
                "is_correct": decision.get("is_correct"),
                "pipeline_status": _normalize_pipeline_status(status, "not_started"),
                "workflow_state": _normalize_workflow_state(
                    ann.get("workflow_state") if isinstance(ann, dict) else None,
                    "sample_selected",
                ),
                "class_name": decision.get("class_name", ""),
                "is_new_class": bool(decision.get("is_new_class", False)),
                "summary": decision.get("summary", ""),
                "step_count": step_count,
                "claim_count": claim_count,
                "dependency_count": dependency_count,
                "updated_at_utc": (
                    ann.get("updated_at_utc")
                    if isinstance(ann, dict) and ann.get("updated_at_utc")
                    else decision.get("updated_at_utc", "")
                ),
            }
        )

    annotation_stats = {
        "total_steps": total_steps,
        "total_claims": total_claims,
        "total_dependencies": total_dependencies,
    }
    return sample_rows, sample_stats, annotation_stats


def _derive_case_workflow_state(
    current_workflow_state: dict[str, Any],
    sample_annotations: dict[str, dict[str, Any]],
    sample_decisions: list[dict[str, Any]],
) -> str:
    states = [current_workflow_state.get("workflow_state", "sample_selected")]
    for ann in sample_annotations.values():
        if isinstance(ann, dict):
            states.append(_normalize_workflow_state(ann.get("workflow_state"), "sample_selected"))
    if sample_decisions and all(
        _normalize_pipeline_status(item.get("pipeline_status"), "not_started") in {"completed", "discarded"}
        for item in sample_decisions
    ):
        states.append("completed")
    return max(states, key=lambda state: WORKFLOW_STATE_ORDER.get(state, -1))


def _build_summary_payload(detail_payload: dict[str, Any]) -> dict[str, Any]:
    sample_rows, sample_stats, annotation_stats = _summarize_samples(
        detail_payload.get("sample_annotations", {}),
        detail_payload.get("sample_decisions", []),
    )
    current_workflow_state = detail_payload.get("current_workflow_state", {})
    latest_workflow_state = _derive_case_workflow_state(
        current_workflow_state,
        detail_payload.get("sample_annotations", {}),
        detail_payload.get("sample_decisions", []),
    )
    screening = current_workflow_state.get("problem_quality_screening", {})
    correct_solutions = []
    for item in detail_payload.get("correct_solutions", []):
        correct_solutions.append(
            {
                "sample_idx": item.get("sample_idx"),
                "completed_at": item.get("completed_at", ""),
                "content_hash": item.get("content_hash", ""),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "annotator_id": detail_payload["annotator_id"],
        "device_id": detail_payload["device_id"],
        "case_id": detail_payload["case_id"],
        "client_revision": detail_payload.get("client_revision", 0),
        "status": detail_payload["status"],
        "created_at_utc": detail_payload["created_at_utc"],
        "updated_at_utc": detail_payload["updated_at_utc"],
        "problem_quality_screening": screening,
        "active_sample_idx": current_workflow_state.get("active_sample_idx"),
        "sample_cursor": current_workflow_state.get("sample_cursor", 0),
        "latest_workflow_state": latest_workflow_state,
        "sample_stats": sample_stats,
        "annotation_stats": annotation_stats,
        "correct_solutions": correct_solutions,
        "samples": sample_rows,
        "summary_content_hash": "",
    }


def _normalize_progress_payload(
    payload: dict[str, Any],
    *,
    default_status: str,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    now = now_utc_iso()
    annotator_id = _normalize_string(payload.get("annotator_id") or payload.get("annotator"), "unknown")
    device_id = _normalize_string(payload.get("device_id"), "device")
    case_id = _normalize_string(payload.get("case_id"), "case")
    current_workflow_state = _normalize_current_workflow_state(payload.get("current_workflow_state"))
    raw_current_annotations = payload.get("current_annotations")
    raw_current_annotations = raw_current_annotations if isinstance(raw_current_annotations, dict) else {}
    if isinstance(payload.get("sample_annotations"), dict):
        raw_current_annotations = {**raw_current_annotations, "sample_annotations": payload.get("sample_annotations")}
    sample_annotations = _normalize_sample_annotations(raw_current_annotations, current_workflow_state, now)
    correct_solutions = _normalize_correct_solutions(payload.get("correct_solutions"), sample_annotations)
    sample_decisions = _normalize_sample_decisions(
        payload.get("sample_decisions"),
        sample_annotations=sample_annotations,
        current_workflow_state=current_workflow_state,
        correct_solutions=correct_solutions,
        now=now,
    )

    detail_payload = {
        "schema_version": SCHEMA_VERSION,
        "annotator_id": annotator_id,
        "device_id": device_id,
        "case_id": case_id,
        "client_revision": _normalize_int(payload.get("client_revision"), default=0, minimum=0),
        "status": _normalize_case_status(payload.get("status"), default_status),
        "current_step": _normalize_int(payload.get("current_step"), default=1, minimum=0),
        "current_workflow_state": current_workflow_state,
        "sample_decisions": sample_decisions,
        "correct_solutions": correct_solutions,
        "sample_annotations": sample_annotations,
        "created_at_utc": created_at or now,
        "updated_at_utc": updated_at or now,
        "detail_content_hash": "",
    }
    detail_payload["detail_content_hash"] = _record_content_hash(
        detail_payload,
        skip_keys={"detail_content_hash", "created_at_utc", "updated_at_utc"},
    )
    return detail_payload


def _detail_to_compat_progress(detail_payload: dict[str, Any], summary_payload: dict[str, Any]) -> dict[str, Any]:
    current_workflow_state = detail_payload.get("current_workflow_state", {})
    active_idx = current_workflow_state.get("active_sample_idx")
    if active_idx is None:
        active_idx = current_workflow_state.get("sample_cursor")
    active_ann = {}
    if active_idx is not None:
        active_ann = detail_payload.get("sample_annotations", {}).get(str(active_idx), {})

    current_annotations = {
        "selected_solution_text": active_ann.get("selected_solution_text", ""),
        "cut_points": active_ann.get("cut_points", []),
        "steps": active_ann.get("steps", []),
        "presegmented_claims": active_ann.get("presegmented_claims", []),
        "claims": active_ann.get("claims", []),
        "claim_checks": active_ann.get("claim_checks", {}),
        "dependencies": active_ann.get("dependencies", {}),
        "step_dependencies": active_ann.get("step_dependencies", {}),
        "sample_annotations": detail_payload.get("sample_annotations", {}),
    }

    return {
        "schema_version": detail_payload.get("schema_version", SCHEMA_VERSION),
        "annotator_id": detail_payload.get("annotator_id", ""),
        "device_id": detail_payload.get("device_id", ""),
        "case_id": detail_payload.get("case_id", ""),
        "client_revision": detail_payload.get("client_revision", 0),
        "status": detail_payload.get("status", "in_progress"),
        "current_step": detail_payload.get("current_step", 1),
        "current_workflow_state": current_workflow_state,
        "current_annotations": current_annotations,
        "sample_decisions": detail_payload.get("sample_decisions", []),
        "correct_solutions": detail_payload.get("correct_solutions", []),
        "created_at_utc": detail_payload.get("created_at_utc", ""),
        "updated_at_utc": detail_payload.get("updated_at_utc", ""),
        "content_hash": detail_payload.get("detail_content_hash", ""),
        "case_summary": {
            "sample_stats": summary_payload.get("sample_stats", {}),
            "annotation_stats": summary_payload.get("annotation_stats", {}),
            "latest_workflow_state": summary_payload.get("latest_workflow_state", "sample_selected"),
        },
    }


def _maybe_upgrade_legacy_record(data: dict[str, Any]) -> bool:
    return "sample_annotations" not in data and "current_annotations" in data


def _load_existing_created_at(annotator_id: str, device_id: str, case_id: str) -> str | None:
    for path in _iter_case_record_paths(annotator_id, case_id, preferred_device_id=device_id):
        if not path.exists():
            continue
        try:
            data = _read_json_file(path)
        except Exception:
            continue
        created_at = data.get("created_at_utc")
        if isinstance(created_at, str) and created_at:
            return created_at
    return None


def _validate_progress_payload(payload: dict[str, Any], *, minimal: bool = False) -> None:
    annotator_id = _normalize_string(payload.get("annotator_id") or payload.get("annotator"), "")
    case_id = _normalize_string(payload.get("case_id"), "")
    if not annotator_id:
        raise ValueError("annotator_id 不能为空")
    if not case_id:
        raise ValueError("case_id 不能为空")
    if minimal:
        return
    meaningful = any(
        key in payload
        for key in ("current_workflow_state", "current_annotations", "sample_annotations", "sample_decisions", "correct_solutions", "current_step")
    )
    if not meaningful:
        raise ValueError("保存负载缺少进度字段")


def _iter_case_bases(annotator_id: str, case_id: str) -> list[Path]:
    annotator_dir = ANNOTATIONS_DIR / safe_name(annotator_id, "unknown")
    if not annotator_dir.exists():
        return []
    case_safe = safe_name(case_id, "case")
    bases: list[Path] = []
    for device_dir in annotator_dir.iterdir():
        if not device_dir.is_dir():
            continue
        base = device_dir / case_safe
        if any(path.exists() for path in (base.with_suffix(".detail.json"), base.with_suffix(".summary.json"), base.with_suffix(".json"))):
            bases.append(base)
    return bases


def _record_sort_key(path: Path) -> tuple[float, str]:
    try:
        data = _read_json_file(path)
    except Exception:
        return (0.0, str(path))
    updated_at = _normalize_string(data.get("updated_at_utc"), "")
    try:
        ts = datetime.fromisoformat(updated_at).timestamp() if updated_at else 0.0
    except ValueError:
        ts = 0.0
    return (ts, str(path))


def _iter_case_record_paths(annotator_id: str, case_id: str, *, preferred_device_id: str | None = None) -> list[Path]:
    bases = _iter_case_bases(annotator_id, case_id)
    preferred_base = None
    if preferred_device_id:
        preferred_base = progress_base_path(annotator_id, preferred_device_id, case_id)
    ordered: list[Path] = []
    seen: set[Path] = set()

    def add_base(base: Path) -> None:
        for path in (base.with_suffix(".detail.json"), base.with_suffix(".summary.json"), base.with_suffix(".json")):
            if path.exists() and path not in seen:
                ordered.append(path)
                seen.add(path)

    if preferred_base and preferred_base in bases:
        add_base(preferred_base)

    remaining_bases = [base for base in bases if base != preferred_base]
    remaining_paths: list[Path] = []
    for base in remaining_bases:
        remaining_paths.extend([p for p in (base.with_suffix(".detail.json"), base.with_suffix(".summary.json"), base.with_suffix(".json")) if p.exists()])
    for path in sorted(remaining_paths, key=_record_sort_key, reverse=True):
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _latest_case_revision(annotator_id: str, case_id: str) -> tuple[int, str | None]:
    for path in _iter_case_record_paths(annotator_id, case_id):
        try:
            data = _read_json_file(path)
        except Exception:
            continue
        revision = _normalize_int(data.get("client_revision"), default=0, minimum=0)
        content_hash = _normalize_string(data.get("detail_content_hash") or data.get("content_hash"), "")
        return revision, (content_hash or None)
    return 0, None


def persist_progress_payload(payload: dict[str, Any], default_status: str = "in_progress") -> dict[str, Any]:
    annotator_id = _normalize_string(payload.get("annotator_id") or payload.get("annotator"), "unknown")
    device_id = _normalize_string(payload.get("device_id"), "device")
    case_id = _normalize_string(payload.get("case_id"), "case")
    created_at = _load_existing_created_at(annotator_id, device_id, case_id)
    latest_revision, latest_content_hash = _latest_case_revision(annotator_id, case_id)

    detail_payload = _normalize_progress_payload(payload, default_status=default_status, created_at=created_at)
    summary_payload = _build_summary_payload(detail_payload)
    summary_payload["summary_content_hash"] = _record_content_hash(
        summary_payload,
        skip_keys={"summary_content_hash", "created_at_utc", "updated_at_utc"},
    )

    incoming_revision = detail_payload["client_revision"]
    if incoming_revision and latest_revision and incoming_revision < latest_revision:
        write_json_log(
            "app_logger",
            "progress.stale_write_ignored",
            annotator_id=annotator_id,
            device_id=device_id,
            case_id=case_id,
            incoming_revision=incoming_revision,
            latest_revision=latest_revision,
        )
        return {
            "ok": True,
            "summary_path": str(progress_summary_path(annotator_id, device_id, case_id)),
            "detail_path": str(progress_detail_path(annotator_id, device_id, case_id)),
            "path": str(progress_detail_path(annotator_id, device_id, case_id)),
            "updated_at_utc": detail_payload["updated_at_utc"],
            "unchanged": True,
            "ignored_stale": True,
            "content_hash": latest_content_hash or detail_payload["detail_content_hash"],
            "client_revision": latest_revision,
        }

    summary_path = progress_summary_path(annotator_id, device_id, case_id)
    detail_path = progress_detail_path(annotator_id, device_id, case_id)

    existing_detail_hash = None
    existing_summary_hash = None
    if detail_path.exists():
        try:
            existing_detail_hash = _read_json_file(detail_path).get("detail_content_hash")
        except Exception:
            existing_detail_hash = None
    if summary_path.exists():
        try:
            existing_summary_hash = _read_json_file(summary_path).get("summary_content_hash")
        except Exception:
            existing_summary_hash = None

    unchanged = (
        existing_detail_hash == detail_payload["detail_content_hash"]
        and existing_summary_hash == summary_payload["summary_content_hash"]
    )
    if unchanged:
        return {
            "ok": True,
            "summary_path": str(summary_path),
            "detail_path": str(detail_path),
            "path": str(detail_path),
            "updated_at_utc": detail_payload["updated_at_utc"],
            "unchanged": True,
            "content_hash": detail_payload["detail_content_hash"],
            "client_revision": detail_payload["client_revision"],
        }

    try:
        _atomic_write_json(detail_path, detail_payload)
        _atomic_write_json(summary_path, summary_payload)
    except Exception as exc:
        write_json_log(
            "app_logger",
            "progress.persist_failed",
            annotator_id=annotator_id,
            device_id=device_id,
            case_id=case_id,
            error_type=type(exc).__name__,
            error=str(exc),
            summary_path=str(summary_path),
            detail_path=str(detail_path),
        )
        raise

    return {
        "ok": True,
        "summary_path": str(summary_path),
        "detail_path": str(detail_path),
        "path": str(detail_path),
        "updated_at_utc": detail_payload["updated_at_utc"],
        "unchanged": False,
        "content_hash": detail_payload["detail_content_hash"],
        "client_revision": detail_payload["client_revision"],
    }

def _load_progress_pair_from_base(base_path: Path, *, annotator_id: str, device_id: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary_path = base_path.with_suffix(".summary.json")
    detail_path = base_path.with_suffix(".detail.json")
    legacy_path = base_path.with_suffix(".json")
    summary_data = None
    detail_data = None

    if detail_path.exists():
        try:
            detail_data = _read_json_file(detail_path)
        except Exception as exc:
            write_json_log(
                "app_logger",
                "progress.detail_read_failed",
                annotator_id=annotator_id,
                device_id=device_id,
                case_id=case_id,
                path=str(detail_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise ProgressLoadError(f"detail record unreadable: {exc}") from exc

    if summary_path.exists():
        try:
            summary_data = _read_json_file(summary_path)
        except Exception as exc:
            write_json_log(
                "app_logger",
                "progress.summary_read_failed",
                annotator_id=annotator_id,
                device_id=device_id,
                case_id=case_id,
                path=str(summary_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            summary_data = None

    if detail_data is None and legacy_path.exists():
        try:
            legacy_data = _read_json_file(legacy_path)
        except Exception as exc:
            raise ProgressLoadError(f"legacy record unreadable: {exc}") from exc
        detail_data = _normalize_progress_payload(
            legacy_data,
            default_status=_normalize_case_status(legacy_data.get("status"), "in_progress"),
            created_at=_normalize_string(legacy_data.get("created_at_utc"), now_utc_iso()),
            updated_at=_normalize_string(legacy_data.get("updated_at_utc"), now_utc_iso()),
        )
        summary_data = _build_summary_payload(detail_data)
        summary_data["summary_content_hash"] = _record_content_hash(
            summary_data,
            skip_keys={"summary_content_hash", "created_at_utc", "updated_at_utc"},
        )
        write_json_log(
            "app_logger",
            "progress.legacy_fallback_used",
            annotator_id=annotator_id,
            device_id=device_id,
            case_id=case_id,
            path=str(legacy_path),
        )
        return summary_data, detail_data, "legacy"

    if detail_data is None:
        if summary_path.exists():
            raise ProgressLoadError("summary exists but detail record is missing")
        raise FileNotFoundError

    detail_data = _normalize_progress_payload(
        detail_data,
        default_status=_normalize_case_status(detail_data.get("status"), "in_progress"),
        created_at=_normalize_string(detail_data.get("created_at_utc"), now_utc_iso()),
        updated_at=_normalize_string(detail_data.get("updated_at_utc"), now_utc_iso()),
    )
    if summary_data is None:
        summary_data = _build_summary_payload(detail_data)
        summary_data["summary_content_hash"] = _record_content_hash(
            summary_data,
            skip_keys={"summary_content_hash", "created_at_utc", "updated_at_utc"},
        )
        write_json_log(
            "app_logger",
            "progress.summary_fallback_used",
            annotator_id=annotator_id,
            device_id=device_id,
            case_id=case_id,
            path=str(detail_path),
        )
    return summary_data, detail_data, "pair"


def _load_progress_pair(annotator_id: str, device_id: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    exact_base = progress_base_path(annotator_id, device_id, case_id)
    bases = _iter_case_bases(annotator_id, case_id)
    if not bases:
        raise FileNotFoundError

    if exact_base in bases:
        try:
            summary_data, detail_data, source = _load_progress_pair_from_base(
                exact_base,
                annotator_id=annotator_id,
                device_id=device_id,
                case_id=case_id,
            )
            return summary_data, detail_data, f"exact_device:{source}"
        except ProgressLoadError:
            bases = [base for base in bases if base != exact_base]
            if not bases:
                raise

    fallback_base = max(
        bases,
        key=lambda base: max(
            (_record_sort_key(path) for path in (base.with_suffix(".detail.json"), base.with_suffix(".summary.json"), base.with_suffix(".json")) if path.exists()),
            default=(0.0, str(base)),
        ),
    )
    fallback_device_id = fallback_base.parent.name
    summary_data, detail_data, source = _load_progress_pair_from_base(
        fallback_base,
        annotator_id=annotator_id,
        device_id=fallback_device_id,
        case_id=case_id,
    )
    return summary_data, detail_data, f"annotator_fallback:{source}"


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj.setdefault("id", f"case-{i+1}")
        obj.setdefault("question", "")
        obj.setdefault("reference_answer", "")
        obj.setdefault("known_solutions", [])
        obj.setdefault("samples", [])
        items.append(obj)
    return items


def split_by_cut_points(text: str, cut_points: list[int]) -> list[str]:
    points = sorted(set([p for p in cut_points if isinstance(p, int) and 0 < p < len(text)]))
    out: list[str] = []
    prev = 0
    for p in points:
        seg = text[prev:p].strip()
        if seg:
            out.append(seg)
        prev = p
    last = text[prev:].strip()
    if last:
        out.append(last)
    return out


def _build_review_record_from_summary(
    *,
    file_label: str,
    summary_data: dict[str, Any],
    detail_data: dict[str, Any] | None,
    raw_payload: dict[str, Any],
    load_error: bool = False,
    error_type: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    sample_stats = summary_data.get("sample_stats", {}) if isinstance(summary_data, dict) else {}
    annotation_stats = summary_data.get("annotation_stats", {}) if isinstance(summary_data, dict) else {}
    return {
        "file": file_label,
        "annotator": summary_data.get("annotator_id", ""),
        "case_id": summary_data.get("case_id", ""),
        "status": summary_data.get("status", ""),
        "saved_at_utc": summary_data.get("updated_at_utc", ""),
        "sample_valid_count": sample_stats.get("correct_samples", 0),
        "total_samples": sample_stats.get("total_samples", 0),
        "completed_samples": sample_stats.get("completed_samples", 0),
        "discarded_samples": sample_stats.get("discarded_samples", 0),
        "step_count": annotation_stats.get("total_steps", 0),
        "claim_count": annotation_stats.get("total_claims", 0),
        "dependency_count": annotation_stats.get("total_dependencies", 0),
        "active_sample_idx": summary_data.get("active_sample_idx"),
        "latest_workflow_state": summary_data.get("latest_workflow_state", ""),
        "load_error": load_error,
        "error_type": error_type,
        "error_message": error_message,
        "raw": raw_payload,
    }


def _build_legacy_review_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    sample_validation = data.get("sample_validation", []) if isinstance(data.get("sample_validation"), list) else []
    return {
        "file": f"legacy/{path.name}",
        "annotator": data.get("annotator", ""),
        "case_id": data.get("case_id", ""),
        "status": data.get("status", ""),
        "saved_at_utc": data.get("saved_at_utc", ""),
        "sample_valid_count": sum(1 for s in sample_validation if isinstance(s, dict) and s.get("is_correct") is True),
        "total_samples": len(sample_validation),
        "completed_samples": sum(1 for s in sample_validation if isinstance(s, dict) and s.get("pipeline_status") == "completed"),
        "discarded_samples": sum(1 for s in sample_validation if isinstance(s, dict) and s.get("pipeline_status") == "discarded"),
        "step_count": len(data.get("steps", [])),
        "claim_count": sum(len(x.get("claims", [])) for x in data.get("claims", []) if isinstance(x, dict)),
        "dependency_count": sum(len(v) for v in data.get("dependencies", {}).values() if isinstance(v, list)),
        "active_sample_idx": None,
        "latest_workflow_state": "",
        "load_error": False,
        "error_type": "",
        "error_message": "",
        "raw": data,
    }


@app.before_request
def start_request_timer() -> None:
    g.request_started_at = time.perf_counter()


@app.after_request
def log_access(response):
    duration_ms = round((time.perf_counter() - getattr(g, "request_started_at", time.perf_counter())) * 1000, 2)
    write_json_log(
        "access_logger",
        "request.completed",
        method=request.method,
        path=request.path,
        query=request.query_string.decode("utf-8", errors="ignore"),
        status_code=response.status_code,
        duration_ms=duration_ms,
        client_ip=client_ip(),
        annotator_id=request.args.get("annotator_id") or "",
        user_agent=request.headers.get("User-Agent", ""),
    )
    return response


@app.teardown_request
def log_request_exception(exc: BaseException | None) -> None:
    if exc is None:
        return
    write_json_log(
        "app_logger",
        "request.exception",
        path=request.path,
        method=request.method,
        client_ip=client_ip(),
        error_type=type(exc).__name__,
        error=str(exc),
    )


@app.get("/")
def landing_page():
    if current_role() != "annotator":
        session["role"] = "annotator"
    return render_template("annotator.html")


@app.get("/annotator")
def annotator_page():
    if current_role() != "annotator":
        session["role"] = "annotator"
    return render_template("annotator.html")


@app.get("/review")
def review_page():
    require_reviewer()
    return render_template("review.html")


@app.get("/<path:asset_path>")
def frontend_asset(asset_path: str):
    if asset_path.startswith(("api/", "review", "annotator", "static/")):
        abort(404)
    resolved = (FRONTEND_OUT_DIR / asset_path).resolve()
    try:
        resolved.relative_to(FRONTEND_OUT_DIR.resolve())
    except ValueError:
        abort(404)

    if resolved.is_dir():
        candidate = resolved / "index.html"
    else:
        candidate = resolved
    if candidate.exists() and candidate.is_file():
        return send_file(candidate)
    abort(404)


@app.get("/api/guideline")
def get_guideline():
    ensure_dirs()
    return jsonify({"content": GUIDE_PATH.read_text(encoding="utf-8")})


@app.put("/api/guideline")
def update_guideline():
    require_reviewer()
    ensure_dirs()
    payload = parse_request_json()
    content = str(payload.get("content") or "").strip()
    if not content:
        return jsonify({"error": "内容不能为空"}), 400
    GUIDE_PATH.write_text(content + "\n", encoding="utf-8")
    return jsonify({"ok": True, "updated_at_utc": now_utc_iso()})


@app.post("/api/session/role")
def set_role():
    payload = parse_request_json()
    role = str(payload.get("role") or "annotator")
    if role not in {"annotator", "reviewer"}:
        return jsonify({"error": "invalid role"}), 400

    if role == "reviewer":
        access_key = str(payload.get("access_key") or "")
        expected = (
            str((DATA_DIR / ".review_key").read_text(encoding="utf-8").strip())
            if (DATA_DIR / ".review_key").exists()
            else "reviewer"
        )
        if access_key != expected:
            return jsonify({"error": "reviewer key 无效"}), 403
    session["role"] = role
    return jsonify({"ok": True, "role": role})


@app.get("/api/session")
def get_session():
    return jsonify({"role": current_role()})


@app.post("/api/load_jsonl")
def load_jsonl():
    ensure_dirs()
    data = request.get_json(force=True)
    path_str = data.get("path", "").strip()
    if not path_str:
        return jsonify({"error": "请提供 JSONL 文件路径"}), 400

    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return jsonify({"error": f"文件不存在: {path_str}"}), 404

    try:
        items = parse_jsonl(p)
    except Exception as exc:
        return jsonify({"error": f"JSONL 解析失败: {exc}"}), 400

    return jsonify({"items": items})


@app.post("/api/translate")
def translate_api():
    payload = parse_request_json()
    text = (payload.get("text") or "").strip()
    target = (payload.get("target") or "zh-CN").strip()
    if not text:
        return jsonify({"error": "text 不能为空"}), 400

    try:
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx"
            f"&sl=auto&tl={quote(target)}&dt=t&q={quote(text)}"
        )
        with urlopen(url, timeout=10) as resp:  # nosec - user-triggered translation helper
            data = json.loads(resp.read().decode("utf-8"))
        translated = "".join(piece[0] for piece in data[0] if piece and piece[0])
        return jsonify({"translated": translated, "provider": "google_gtx"})
    except Exception as exc:
        return jsonify({"error": f"翻译服务暂不可用: {exc}"}), 502


@app.route("/api/save_progress", methods=["PUT", "POST"])
def save_progress():
    ensure_dirs()
    payload = parse_request_json()
    try:
        _validate_progress_payload(payload, minimal=False)
        result = persist_progress_payload(payload, default_status="in_progress")
    except ValueError as exc:
        write_json_log(
            "app_logger",
            "progress.validation_failed",
            annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
            device_id=str(payload.get("device_id") or "device"),
            case_id=str(payload.get("case_id") or "unknown"),
            error=str(exc),
        )
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        write_json_log(
            "app_logger",
            "progress.saved_failed",
            annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
            device_id=str(payload.get("device_id") or "device"),
            case_id=str(payload.get("case_id") or "unknown"),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return jsonify({"error": f"保存失败: {exc}"}), 500
    write_json_log(
        "app_logger",
        "progress.saved",
        annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
        device_id=str(payload.get("device_id") or "device"),
        case_id=str(payload.get("case_id") or "unknown"),
        ok=bool(result.get("ok")),
        unchanged=bool(result.get("unchanged", False)),
        content_hash=str(result.get("content_hash") or ""),
    )
    return jsonify(result)


@app.get("/api/load_progress")
def load_progress():
    ensure_dirs()
    annotator_id = request.args.get("annotator_id", "").strip()
    device_id = request.args.get("device_id", "device")
    case_id = request.args.get("case_id", "").strip()

    if not annotator_id or not case_id:
        return jsonify({"error": "annotator_id 和 case_id 不能为空"}), 400

    try:
        summary_data, detail_data, source = _load_progress_pair(annotator_id, device_id, case_id)
    except FileNotFoundError:
        return jsonify({"found": False})
    except ProgressLoadError as exc:
        return jsonify({"found": False, "error": str(exc)}), 500

    progress = _detail_to_compat_progress(detail_data, summary_data)
    return jsonify(
        {
            "found": True,
            "progress": progress,
            "path": str(progress_detail_path(annotator_id, device_id, case_id)),
            "summary_path": str(progress_summary_path(annotator_id, device_id, case_id)),
            "source": source,
            "client_revision": progress.get("client_revision", 0),
        }
    )


@app.post("/api/save_record")
def save_record():
    payload = parse_request_json()
    payload.setdefault("status", "completed")
    try:
        _validate_progress_payload(payload, minimal=True)
        result = persist_progress_payload(payload, default_status="completed")
    except ValueError as exc:
        write_json_log(
            "app_logger",
            "record.validation_failed",
            annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
            device_id=str(payload.get("device_id") or "device"),
            case_id=str(payload.get("case_id") or "unknown"),
            error=str(exc),
        )
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        write_json_log(
            "app_logger",
            "record.saved_failed",
            annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
            device_id=str(payload.get("device_id") or "device"),
            case_id=str(payload.get("case_id") or "unknown"),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return jsonify({"error": f"保存失败: {exc}"}), 500
    write_json_log(
        "app_logger",
        "record.saved",
        annotator_id=str(payload.get("annotator_id") or payload.get("annotator") or "unknown"),
        device_id=str(payload.get("device_id") or "device"),
        case_id=str(payload.get("case_id") or "unknown"),
        ok=bool(result.get("ok")),
        unchanged=bool(result.get("unchanged", False)),
        content_hash=str(result.get("content_hash") or ""),
    )
    return jsonify(result)


@app.get("/api/review_records")
def review_records_api():
    require_reviewer()
    ensure_dirs()
    records = []

    summary_files = {path.with_suffix("").with_suffix(""): path for path in ANNOTATIONS_DIR.glob("*/*/*.summary.json")}
    detail_files = {path.with_suffix("").with_suffix(""): path for path in ANNOTATIONS_DIR.glob("*/*/*.detail.json")}
    paired_keys = sorted(set(summary_files) | set(detail_files))
    paired_annotation_keys = {
        (
            str(base_key.parent.relative_to(ANNOTATIONS_DIR)),
            base_key.name,
        )
        for base_key in paired_keys
    }

    for base_key in paired_keys:
        summary_path = summary_files.get(base_key)
        detail_path = detail_files.get(base_key)
        file_label = str((summary_path or detail_path).relative_to(ANNOTATIONS_DIR))
        summary_data = None
        detail_data = None
        errors: list[str] = []

        if summary_path is not None:
            try:
                summary_data = _read_json_file(summary_path)
            except Exception as exc:
                errors.append(f"summary:{type(exc).__name__}:{exc}")
                write_json_log(
                    "app_logger",
                    "review.summary_read_failed",
                    path=str(summary_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    fallback_used=bool(detail_path),
                )

        if detail_path is not None:
            try:
                detail_data = _read_json_file(detail_path)
                detail_data = _normalize_progress_payload(
                    detail_data,
                    default_status=_normalize_case_status(detail_data.get("status"), "in_progress"),
                    created_at=_normalize_string(detail_data.get("created_at_utc"), now_utc_iso()),
                )
            except Exception as exc:
                errors.append(f"detail:{type(exc).__name__}:{exc}")
                write_json_log(
                    "app_logger",
                    "review.detail_read_failed",
                    path=str(detail_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    fallback_used=bool(summary_path),
                )

        if summary_data is None and detail_data is not None:
            summary_data = _build_summary_payload(detail_data)
            summary_data["summary_content_hash"] = _record_content_hash(
                summary_data,
                skip_keys={"summary_content_hash", "created_at_utc", "updated_at_utc"},
            )

        if summary_data is None and detail_data is None:
            records.append(
                {
                    "file": file_label,
                    "annotator": "",
                    "case_id": base_key.name,
                    "status": "",
                    "saved_at_utc": "",
                    "sample_valid_count": 0,
                    "total_samples": 0,
                    "completed_samples": 0,
                    "discarded_samples": 0,
                    "step_count": 0,
                    "claim_count": 0,
                    "dependency_count": 0,
                    "active_sample_idx": None,
                    "latest_workflow_state": "",
                    "load_error": True,
                    "error_type": "ReadError",
                    "error_message": "; ".join(errors),
                    "raw": {"summary": None, "detail": None},
                }
            )
            continue

        raw_payload = {"summary": summary_data, "detail": detail_data}
        records.append(
            _build_review_record_from_summary(
                file_label=file_label,
                summary_data=summary_data or {},
                detail_data=detail_data,
                raw_payload=raw_payload,
                load_error=bool(errors),
                error_type="ReadError" if errors else "",
                error_message="; ".join(errors),
            )
        )

    for path in sorted(ANNOTATIONS_DIR.glob("*/*/*.json")):
        if path.name.endswith(".summary.json") or path.name.endswith(".detail.json"):
            continue
        legacy_key = (
            str(path.parent.relative_to(ANNOTATIONS_DIR)),
            path.stem,
        )
        if legacy_key in paired_annotation_keys:
            continue
        try:
            data = _read_json_file(path)
            if _maybe_upgrade_legacy_record(data):
                detail_data = _normalize_progress_payload(
                    data,
                    default_status=_normalize_case_status(data.get("status"), "in_progress"),
                    created_at=_normalize_string(data.get("created_at_utc"), now_utc_iso()),
                    updated_at=_normalize_string(data.get("updated_at_utc"), now_utc_iso()),
                )
                summary_data = _build_summary_payload(detail_data)
                summary_data["summary_content_hash"] = _record_content_hash(
                    summary_data,
                    skip_keys={"summary_content_hash", "created_at_utc", "updated_at_utc"},
                )
                records.append(
                    _build_review_record_from_summary(
                        file_label=str(path.relative_to(ANNOTATIONS_DIR)),
                        summary_data=summary_data,
                        detail_data=detail_data,
                        raw_payload={"summary": summary_data, "detail": detail_data, "legacy": data},
                    )
                )
            else:
                records.append(_build_legacy_review_record(path, data))
        except Exception as exc:
            records.append(
                {
                    "file": str(path.relative_to(ANNOTATIONS_DIR)),
                    "annotator": "",
                    "case_id": path.stem,
                    "status": "",
                    "saved_at_utc": "",
                    "sample_valid_count": 0,
                    "total_samples": 0,
                    "completed_samples": 0,
                    "discarded_samples": 0,
                    "step_count": 0,
                    "claim_count": 0,
                    "dependency_count": 0,
                    "active_sample_idx": None,
                    "latest_workflow_state": "",
                    "load_error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw": None,
                }
            )

    for path in sorted(RECORDS_DIR.glob("*.json")):
        try:
            data = _read_json_file(path)
            records.append(_build_legacy_review_record(path, data))
        except Exception as exc:
            records.append(
                {
                    "file": f"records/{path.name}",
                    "annotator": "",
                    "case_id": path.stem,
                    "status": "",
                    "saved_at_utc": "",
                    "sample_valid_count": 0,
                    "total_samples": 0,
                    "completed_samples": 0,
                    "discarded_samples": 0,
                    "step_count": 0,
                    "claim_count": 0,
                    "dependency_count": 0,
                    "active_sample_idx": None,
                    "latest_workflow_state": "",
                    "load_error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw": None,
                }
            )

    return jsonify({"records": records})


@app.post("/api/split_steps")
def split_steps_api():
    payload = parse_request_json()
    solution = payload.get("solution", "")
    cut_points = payload.get("cut_points", [])
    return jsonify({"steps": split_by_cut_points(solution, cut_points)})


if __name__ == "__main__":
    ensure_dirs()
    configure_app_logging()
    host = os.environ.get("ANNOTATION_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("ANNOTATION_APP_PORT", "5000"))
    debug = os.environ.get("ANNOTATION_APP_DEBUG", "").strip() == "1"
    write_json_log("app_logger", "server.start", host=host, port=port, debug=debug)
    app.run(host=host, port=port, debug=debug)
