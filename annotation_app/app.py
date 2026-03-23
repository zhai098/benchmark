from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

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


def fallback_segment_claims(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[。；;\n]+", text) if p.strip()]
    return parts or [text.strip()] if text.strip() else []


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


def parse_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def extract_json_text(raw: str) -> str:
    content = (raw or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def generate_claims_with_openai(
    steps: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")

    client = OpenAI(api_key=api_key, base_url=(base_url or None))

    step_payload = [
        {
            "step_id": s.get("id", f"s{i+1}"),
            "text": s.get("text", ""),
        }
        for i, s in enumerate(steps)
    ]
    system_prompt = (
        "你是严谨的数学/推理标注助手。"
        "请把每个 step 切分为最小可核验 claim。"
        "保持原意，不要补充事实，不要改写结论。"
        "输出必须是严格 JSON，不要任何额外说明。"
    )
    user_prompt = (
        "请对以下 steps 做 claim 切分，并仅返回 JSON 数组：\n"
        '[{"step_id":"...","claims":["..."]}]\n'
        "要求：\n"
        "1) step_id 与输入一致；\n"
        "2) claims 为非空字符串数组；\n"
        "3) 不要遗漏 step；\n"
        f"输入 steps: {json.dumps(step_payload, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    parsed = json.loads(extract_json_text(content or "[]"))
    if not isinstance(parsed, list):
        raise RuntimeError("模型返回格式错误：不是列表")

    by_id = {str(s.get("id", f"s{i+1}")): s for i, s in enumerate(steps)}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(parsed):
        if not isinstance(row, dict):
            raise RuntimeError(f"模型返回第 {i+1} 项不是对象")
        step_id = str(row.get("step_id", "")).strip()
        if not step_id:
            raise RuntimeError(f"模型返回第 {i+1} 项缺少 step_id")
        claims = row.get("claims", [])
        if not isinstance(claims, list):
            raise RuntimeError(f"step_id={step_id} 的 claims 不是数组")
        clean_claims = [str(c).strip() for c in claims if str(c).strip()]
        out.append({"step_id": step_id, "claims": clean_claims})

    # 确保每个输入 step 都有返回，缺失则补空数组，避免前端对齐错误
    seen = {x["step_id"] for x in out}
    for i, s in enumerate(steps):
        sid = str(s.get("id", f"s{i+1}"))
        if sid not in seen:
            out.append({"step_id": sid, "claims": []})

    # 按输入顺序排序
    order = {str(s.get("id", f"s{i+1}")): i for i, s in enumerate(steps)}
    out.sort(key=lambda x: order.get(x["step_id"], 10**9))
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


@app.post("/api/generate_claims")
def generate_claims_api():
    payload = request.get_json(force=True)
    steps: list[dict[str, Any]] = payload.get("steps", [])
    model = (payload.get("model") or os.getenv("OPENAI_CLAIM_MODEL") or "gpt-4o-mini").strip()
    base_url = (payload.get("base_url") or os.getenv("OPENAI_BASE_URL") or "").strip()
    allow_fallback = parse_bool(payload.get("allow_fallback"), default=True)
    try:
        temperature = float(payload.get("temperature", 0))
    except Exception:
        temperature = 0
    try:
        max_tokens = int(payload.get("max_tokens", 1500))
    except Exception:
        max_tokens = 1500

    try:
        claims_by_step = generate_claims_with_openai(
            steps,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        return jsonify({"claims_by_step": claims_by_step, "source": "openai_sdk", "model": model})
    except Exception as exc:
        if not allow_fallback:
            return jsonify({"error": f"OpenAI claim 生成失败: {exc}"}), 502

    output = []
    for s in steps:
        claims = fallback_segment_claims(s.get("text", ""))
        output.append({"step_id": s.get("id"), "claims": claims})
    return jsonify({"claims_by_step": output, "source": "fallback"})


@app.post("/api/save_record")
def save_record():
    ensure_dirs()
    payload = request.get_json(force=True)
    annotator = payload.get("annotator", "unknown")
    case_id = payload.get("case_id", "unknown")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{case_id}__{annotator}__{ts}.json"
    out_path = RECORDS_DIR / filename

    payload["saved_at_utc"] = ts
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "path": str(out_path)})


@app.get("/api/review_records")
def review_records_api():
    ensure_dirs()
    records = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "file": path.name,
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
