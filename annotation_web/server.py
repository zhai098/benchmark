#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
ANNOTATION_DIR = DATA_DIR / "annotations"
DATASET_CACHE = DATA_DIR / "dataset_cache.json"
GUIDELINE_FILE = BASE_DIR / "guideline.md"


for path in (DATA_DIR, ANNOTATION_DIR):
    path.mkdir(parents=True, exist_ok=True)


if not GUIDELINE_FILE.exists():
    GUIDELINE_FILE.write_text(
        "# 标注指南（简版）\n\n"
        "1. 先判断多采样解是否正确。\n"
        "2. 正确样本需分类到方法类别；新方法需写概要。\n"
        "3. Step 切分只允许切分，不允许改写原文。\n"
        "4. Claim 切分以语义最小可核验单元为主。\n"
        "5. 依赖关系仅可指向当前 claim 之前出现的 claim。\n",
        encoding="utf-8",
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_steps_preserve(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    segments = re.split(r"(?<=[。！？.!?])\s+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def split_claims(step_text: str) -> List[str]:
    parts = re.split(r"(?<=[，,；;。.!?])\s*", step_text)
    claims = [normalize_text(p) for p in parts if normalize_text(p)]
    return claims or [normalize_text(step_text)]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data.get("question") or data.get("problem") or data.get("prompt") or ""
            standard_answer = data.get("standard_answer") or data.get("answer") or data.get("reference") or ""

            samples = []
            raw_samples = data.get("samples") or data.get("multi_samples") or data.get("solutions")
            if isinstance(raw_samples, list):
                for sidx, sample in enumerate(raw_samples):
                    if isinstance(sample, dict):
                        solution = sample.get("solution") or sample.get("text") or sample.get("response") or ""
                    else:
                        solution = str(sample)
                    samples.append({"sample_id": sidx + 1, "solution": solution})

            if not samples:
                single_solution = data.get("solution") or data.get("response") or ""
                if single_solution:
                    samples = [{"sample_id": 1, "solution": single_solution}]

            records.append(
                {
                    "task_id": idx,
                    "question": question,
                    "standard_answer": standard_answer,
                    "samples": samples,
                    "source": path,
                }
            )
    return records


def dataset_key(path: str) -> str:
    return os.path.abspath(path)


def ensure_dataset_cached(path: str) -> List[Dict[str, Any]]:
    cache = read_json(DATASET_CACHE, {})
    key = dataset_key(path)
    if key not in cache:
        cache[key] = load_jsonl(path)
        write_json(DATASET_CACHE, cache)
    return cache[key]


def annotation_file(annotator: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", annotator) or "anonymous"
    return ANNOTATION_DIR / f"{safe}.json"


def default_task_state(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "status": "not_started",
        "sample_reviews": [],
        "method_categories": ["与已知解同方法", "等价变体"],
        "selected_primary_sample": None,
        "step_segments": [],
        "claims": [],
        "dependencies": {},
        "saved_at": None,
        "submitted_at": None,
    }


def load_annotation(annotator: str) -> Dict[str, Any]:
    path = annotation_file(annotator)
    return read_json(
        path,
        {
            "annotator": annotator,
            "dataset_path": "",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "tasks": {},
        },
    )


def save_annotation(annotator: str, data: Dict[str, Any]) -> None:
    data["updated_at"] = now_iso()
    write_json(annotation_file(annotator), data)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        if route == "/":
            return self._send_file(STATIC_DIR / "client.html", "text/html; charset=utf-8")
        if route == "/review":
            return self._send_file(STATIC_DIR / "reviewer.html", "text/html; charset=utf-8")
        if route == "/style.css":
            return self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        if route == "/client.js":
            return self._send_file(STATIC_DIR / "client.js", "application/javascript; charset=utf-8")
        if route == "/reviewer.js":
            return self._send_file(STATIC_DIR / "reviewer.js", "application/javascript; charset=utf-8")

        if route == "/api/guideline":
            return self._send_json({"content": GUIDELINE_FILE.read_text(encoding="utf-8")})

        if route == "/api/session":
            annotator = (qs.get("annotator") or [""])[0]
            if not annotator:
                return self._send_json({"error": "annotator is required"}, HTTPStatus.BAD_REQUEST)
            ann = load_annotation(annotator)
            dataset_path = ann.get("dataset_path")
            tasks = ensure_dataset_cached(dataset_path) if dataset_path else []
            states = ann.get("tasks", {})
            items = []
            for t in tasks:
                tid = str(t["task_id"])
                state = states.get(tid, default_task_state(t))
                items.append({"task_id": t["task_id"], "status": state["status"], "saved_at": state.get("saved_at")})
            return self._send_json({"annotator": annotator, "dataset_path": dataset_path, "tasks": items})

        if route == "/api/task":
            annotator = (qs.get("annotator") or [""])[0]
            task_id = (qs.get("task_id") or [""])[0]
            if not annotator or not task_id:
                return self._send_json({"error": "annotator and task_id are required"}, HTTPStatus.BAD_REQUEST)
            ann = load_annotation(annotator)
            dataset_path = ann.get("dataset_path")
            if not dataset_path:
                return self._send_json({"error": "session not initialized"}, HTTPStatus.BAD_REQUEST)
            tasks = ensure_dataset_cached(dataset_path)
            task = next((t for t in tasks if str(t["task_id"]) == str(task_id)), None)
            if not task:
                return self._send_json({"error": "task not found"}, HTTPStatus.NOT_FOUND)
            state = ann["tasks"].get(str(task_id), default_task_state(task))
            return self._send_json({"task": task, "state": state})

        if route == "/api/review/summary":
            rows = []
            for fp in sorted(ANNOTATION_DIR.glob("*.json")):
                ann = read_json(fp, {})
                tasks = ann.get("tasks", {})
                submitted = sum(1 for t in tasks.values() if t.get("status") == "submitted")
                rows.append(
                    {
                        "annotator": ann.get("annotator"),
                        "dataset_path": ann.get("dataset_path"),
                        "total_tasks": len(tasks),
                        "submitted_tasks": submitted,
                        "updated_at": ann.get("updated_at"),
                    }
                )
            return self._send_json({"annotators": rows})

        if route == "/api/review/task":
            annotator = (qs.get("annotator") or [""])[0]
            task_id = (qs.get("task_id") or [""])[0]
            ann = load_annotation(annotator)
            dataset_path = ann.get("dataset_path")
            tasks = ensure_dataset_cached(dataset_path) if dataset_path else []
            task = next((t for t in tasks if str(t["task_id"]) == str(task_id)), None)
            state = ann.get("tasks", {}).get(str(task_id))
            if not task or not state:
                return self._send_json({"error": "task or annotation not found"}, HTTPStatus.NOT_FOUND)
            return self._send_json({"task": task, "state": state, "annotator": annotator})

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        data = self._read_json()

        if route == "/api/session/init":
            annotator = (data.get("annotator") or "").strip()
            dataset_path = (data.get("dataset_path") or "").strip()
            if not annotator or not dataset_path:
                return self._send_json({"error": "annotator and dataset_path required"}, HTTPStatus.BAD_REQUEST)
            if not os.path.exists(dataset_path):
                return self._send_json({"error": "dataset file not found"}, HTTPStatus.BAD_REQUEST)
            tasks = ensure_dataset_cached(dataset_path)
            ann = load_annotation(annotator)
            ann["dataset_path"] = dataset_path
            for t in tasks:
                ann["tasks"].setdefault(str(t["task_id"]), default_task_state(t))
            save_annotation(annotator, ann)
            return self._send_json({"ok": True, "task_count": len(tasks)})

        if route in {"/api/task/save", "/api/task/submit", "/api/task/generate_claims"}:
            annotator = data.get("annotator", "")
            task_id = str(data.get("task_id", ""))
            ann = load_annotation(annotator)
            dataset_path = ann.get("dataset_path")
            tasks = ensure_dataset_cached(dataset_path) if dataset_path else []
            task = next((t for t in tasks if str(t["task_id"]) == task_id), None)
            if not task:
                return self._send_json({"error": "task not found"}, HTTPStatus.NOT_FOUND)
            state = ann["tasks"].setdefault(task_id, default_task_state(task))

            if route == "/api/task/generate_claims":
                step_segments = data.get("step_segments") or []
                claims = []
                for sidx, step in enumerate(step_segments):
                    for cidx, claim in enumerate(split_claims(step), start=1):
                        claims.append(
                            {
                                "claim_id": f"S{sidx+1}-C{cidx}",
                                "step_index": sidx,
                                "text": claim,
                            }
                        )
                state["step_segments"] = step_segments
                state["claims"] = claims
                save_annotation(annotator, ann)
                return self._send_json({"claims": claims})

            state["sample_reviews"] = data.get("sample_reviews", state.get("sample_reviews", []))
            state["method_categories"] = data.get("method_categories", state.get("method_categories", []))
            state["selected_primary_sample"] = data.get("selected_primary_sample")
            state["step_segments"] = data.get("step_segments", state.get("step_segments", []))
            state["claims"] = data.get("claims", state.get("claims", []))
            state["dependencies"] = data.get("dependencies", state.get("dependencies", {}))
            state["status"] = "in_progress" if route == "/api/task/save" else "submitted"
            state["saved_at"] = now_iso()
            if route == "/api/task/submit":
                state["submitted_at"] = now_iso()
            ann["tasks"][task_id] = state
            save_annotation(annotator, ann)
            return self._send_json({"ok": True, "status": state["status"], "saved_at": state["saved_at"]})

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
