from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from flask import Flask, abort, jsonify, render_template, request, send_file, session

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
GUIDE_PATH = DATA_DIR / "guideline.md"
FRONTEND_OUT_DIR = BASE_DIR.parent / "frontend" / "out"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "annotation-app-dev-secret"




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


def persist_progress_payload(payload: dict[str, Any], default_status: str = "in_progress") -> dict[str, Any]:
    annotator_id = str(payload.get("annotator_id") or payload.get("annotator") or "unknown")
    device_id = str(payload.get("device_id") or "device")
    case_id = str(payload.get("case_id") or "unknown")

    path = progress_path(annotator_id, device_id, case_id)
    now = now_utc_iso()
    content = {
        "annotator_id": annotator_id,
        "device_id": device_id,
        "case_id": case_id,
        "status": payload.get("status", default_status),
        "current_step": payload.get("current_step", 1),
        "current_workflow_state": payload.get("current_workflow_state", {}),
        "current_annotations": payload.get("current_annotations", {}),
        "sample_decisions": payload.get("sample_decisions", []),
        "correct_solutions": payload.get("correct_solutions", []),
    }
    content_hash = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_hash") == content_hash:
            return {
                "ok": True,
                "path": str(path),
                "updated_at_utc": existing.get("updated_at_utc"),
                "unchanged": True,
                "content_hash": content_hash,
            }
        created_at = existing.get("created_at_utc", now)
    else:
        created_at = now

    to_store = {
        **content,
        "updated_at_utc": now,
        "created_at_utc": created_at,
        "content_hash": content_hash,
    }
    path.write_text(json.dumps(to_store, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path), "updated_at_utc": now, "unchanged": False, "content_hash": content_hash}


def safe_name(value: str, default: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", (value or "").strip())
    return sanitized or default


def progress_path(annotator_id: str, device_id: str, case_id: str) -> Path:
    annotator_safe = safe_name(annotator_id, "unknown")
    device_safe = safe_name(device_id, "device")
    case_safe = safe_name(case_id, "case")
    dir_path = ANNOTATIONS_DIR / annotator_safe / device_safe
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{case_safe}.json"


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


@app.get("/")
def landing_page():
    return render_template("home.html", role=current_role())


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
        expected = str((DATA_DIR / ".review_key").read_text(encoding="utf-8").strip()) if (DATA_DIR / ".review_key").exists() else "reviewer"
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
    result = persist_progress_payload(payload, default_status="in_progress")
    return jsonify(result)


@app.get("/api/load_progress")
def load_progress():
    ensure_dirs()
    annotator_id = request.args.get("annotator_id", "unknown")
    device_id = request.args.get("device_id", "device")
    case_id = request.args.get("case_id", "unknown")

    path = progress_path(annotator_id, device_id, case_id)
    if not path.exists():
        return jsonify({"found": False})

    data = json.loads(path.read_text(encoding="utf-8"))
    return jsonify({"found": True, "progress": data, "path": str(path)})


@app.post("/api/save_record")
def save_record():
    # backward-compatible alias for explicit final save
    payload = parse_request_json()
    payload.setdefault("status", "completed")
    result = persist_progress_payload(payload, default_status="completed")
    return jsonify(result)


@app.get("/api/review_records")
def review_records_api():
    require_reviewer()
    ensure_dirs()
    records = []

    for path in sorted(ANNOTATIONS_DIR.glob("*/*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        current_annotations = data.get("current_annotations", {})
        sample_decisions = data.get("sample_decisions", [])
        records.append(
            {
                "file": str(path.relative_to(ANNOTATIONS_DIR)),
                "annotator": data.get("annotator_id", ""),
                "case_id": data.get("case_id", ""),
                "saved_at_utc": data.get("updated_at_utc", ""),
                "sample_valid_count": sum(1 for s in sample_decisions if s.get("is_correct") is True),
                "step_count": len(current_annotations.get("steps", [])),
                "claim_count": sum(len(x.get("claims", [])) for x in current_annotations.get("claims", [])),
                "dependency_count": sum(len(v) for v in current_annotations.get("dependencies", {}).values()),
                "raw": data,
            }
        )

    for path in sorted(RECORDS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "file": f"legacy/{path.name}",
                "annotator": data.get("annotator", ""),
                "case_id": data.get("case_id", ""),
                "saved_at_utc": data.get("saved_at_utc", ""),
                "sample_valid_count": sum(
                    1
                    for s in data.get("sample_validation", [])
                    if s.get("is_correct") is True
                ),
                "step_count": len(data.get("steps", [])),
                "claim_count": sum(len(x.get("claims", [])) for x in data.get("claims", [])),
                "dependency_count": sum(len(v) for v in data.get("dependencies", {}).values()),
                "raw": data,
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
    app.run(host="0.0.0.0", port=5000, debug=True)
