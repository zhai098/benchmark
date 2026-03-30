from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
GUIDE_PATH = DATA_DIR / "guideline.md"

app = Flask(__name__, template_folder="templates", static_folder="static")


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
def annotator_page():
    return render_template("annotator.html")


@app.get("/review")
def review_page():
    return render_template("review.html")


@app.get("/api/guideline")
def get_guideline():
    ensure_dirs()
    return jsonify({"content": GUIDE_PATH.read_text(encoding="utf-8")})


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
    payload = request.get_json(force=True)
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
    payload = request.get_json(force=True)

    annotator_id = str(payload.get("annotator_id") or payload.get("annotator") or "unknown")
    device_id = str(payload.get("device_id") or "device")
    case_id = str(payload.get("case_id") or "unknown")

    path = progress_path(annotator_id, device_id, case_id)
    now = now_utc_iso()

    to_store = {
        "annotator_id": annotator_id,
        "device_id": device_id,
        "case_id": case_id,
        "status": payload.get("status", "in_progress"),
        "current_step": payload.get("current_step", 1),
        "current_workflow_state": payload.get("current_workflow_state", {}),
        "current_annotations": payload.get("current_annotations", {}),
        "sample_decisions": payload.get("sample_decisions", []),
        "correct_solutions": payload.get("correct_solutions", []),
        "updated_at_utc": now,
        "created_at_utc": now,
    }

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        to_store["created_at_utc"] = existing.get("created_at_utc", now)

    path.write_text(json.dumps(to_store, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "path": str(path), "updated_at_utc": now})


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
    payload = request.get_json(force=True)
    payload.setdefault("status", "completed")
    return save_progress()


@app.get("/api/review_records")
def review_records_api():
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
    payload = request.get_json(force=True)
    solution = payload.get("solution", "")
    cut_points = payload.get("cut_points", [])
    return jsonify({"steps": split_by_cut_points(solution, cut_points)})


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
