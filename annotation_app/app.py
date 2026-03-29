from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
GUIDE_PATH = DATA_DIR / "guideline.md"

app = Flask(__name__, template_folder="templates", static_folder="static")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    if not GUIDE_PATH.exists():
        GUIDE_PATH.write_text(
            "# 标注指南\n\n"
            "1. 先做多采样验证，再做 step/claim/依赖标注。\n"
            "2. 只切分，不修改原始文本。\n"
            "3. 依赖只能选前序 claim（同 step 前序 + 前面 steps）。\n"
            "4. 每个问题完成后再提交。\n",
            encoding="utf-8",
        )


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


def annotator_record_path(annotator: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (annotator or "unknown"))
    return RECORDS_DIR / f"{safe}.json"


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


@app.post("/api/save_record")
def save_record():
    ensure_dirs()
    payload = request.get_json(force=True)
    annotator = payload.get("annotator", "unknown")
    case_id = str(payload.get("case_id", "unknown"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = annotator_record_path(annotator)

    store = {"annotator": annotator, "updated_at_utc": ts, "cases": {}}
    if out_path.exists():
        try:
            store = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(store.get("cases"), dict):
                store["cases"] = {}
        except Exception:
            store = {"annotator": annotator, "updated_at_utc": ts, "cases": {}}

    payload["saved_at_utc"] = ts
    store["annotator"] = annotator
    store["updated_at_utc"] = ts
    store["cases"][case_id] = payload
    out_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "path": str(out_path), "saved_at_utc": ts})


@app.get("/api/load_progress")
def load_progress():
    ensure_dirs()
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"error": "annotator 不能为空"}), 400
    out_path = annotator_record_path(annotator)
    if not out_path.exists():
        return jsonify({"annotator": annotator, "cases": {}, "updated_at_utc": None})
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return jsonify(
        {
            "annotator": annotator,
            "cases": data.get("cases", {}),
            "updated_at_utc": data.get("updated_at_utc"),
            "path": str(out_path),
        }
    )


@app.get("/api/review_records")
def review_records_api():
    ensure_dirs()
    records = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for case_id, case_data in (data.get("cases") or {}).items():
            sample_pipelines = case_data.get("sample_pipelines") or {}
            records.append(
                {
                    "file": path.name,
                    "annotator": data.get("annotator", ""),
                    "case_id": case_id,
                    "saved_at_utc": case_data.get("saved_at_utc", ""),
                    "sample_valid_count": sum(
                        1
                        for s in case_data.get("sample_validation", [])
                        if s.get("is_correct") is True
                    ),
                    "step_count": sum(len(v.get("steps", [])) for v in sample_pipelines.values()),
                    "claim_count": sum(
                        sum(len(x.get("claims", [])) for x in v.get("claims", []))
                        for v in sample_pipelines.values()
                    ),
                    "dependency_count": sum(
                        sum(len(v2) for v2 in (v.get("dependencies") or {}).values())
                        for v in sample_pipelines.values()
                    ),
                    "raw": case_data,
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
