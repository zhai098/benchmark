#!/usr/bin/env python3
"""Guard monitor for completed-annotation model generation runs.

Polls active run status files and partially-written gen_only.jsonl files. If a
run shows systemic non-accidental failures, terminates only the affected wrapper
process, preserving logs and artifacts.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO = Path("/home/zhaipengxiang/benchmark")
LOGS_ROOT = REPO / "logs"
STATE_PATH = LOGS_ROOT / "completed_annotation_guard_state.json"
EVENTS_PATH = LOGS_ROOT / "completed_annotation_guard_events.jsonl"
GUARD_LOG = LOGS_ROOT / "completed_annotation_guard_monitor.log"
POLL_SECONDS = int(os.environ.get("COMPLETED_GUARD_POLL_SECONDS", "60"))
NO_PROGRESS_SECONDS = int(os.environ.get("COMPLETED_GUARD_NO_PROGRESS_SECONDS", "1800"))
SCORED_EMPTY_LIMIT = int(os.environ.get("COMPLETED_GUARD_SCORED_EMPTY_LIMIT", "2"))
MOJIBAKE_LIMIT = int(os.environ.get("COMPLETED_GUARD_MOJIBAKE_LIMIT", "0"))
TEMPLATE_LEAK_LIMIT = int(os.environ.get("COMPLETED_GUARD_TEMPLATE_LEAK_LIMIT", "1"))

ERROR_OR_RUNTIME_MARKERS = (
    "Traceback",
    "CUDA out of memory",
    "ImportError",
    "Exception:",
    "Error:",
)
CHAT_TEMPLATE_LEAK_MARKERS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "[INST]",
    "[/INST]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "<s>",
    "</s>",
)
MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€™", "â€œ", "â€", "å")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {message}\n"
    with GUARD_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def append_event(event: Dict[str, Any]) -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event.setdefault("created_at_utc", now())
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    log(f"EVENT {json.dumps(event, ensure_ascii=False)}")


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def gpu_busy() -> bool:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return False
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                if int(parts[0]) > 10000 and int(parts[1]) > 50:
                    return True
            except ValueError:
                pass
    return False


def active_status_files() -> List[Path]:
    files: List[Path] = []
    for root in LOGS_ROOT.glob("completed_annotations_manifest_models*"):
        # Historical archives keep old status files and can reuse the same
        # tags as current runs. Never let those archived files control live
        # processes.
        if root.is_dir() and "aborted" not in root.name:
            files.extend(root.glob("*_status.json"))
    return sorted(files)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def output_issues(text: str) -> List[str]:
    issues: List[str] = []
    if any(marker in text for marker in ERROR_OR_RUNTIME_MARKERS):
        issues.append("runtime_error_text")
    if any(marker in text for marker in CHAT_TEMPLATE_LEAK_MARKERS):
        issues.append("chat_template_token_leak")
    if re.search(r"(?im)^\s*(assistant|user|system)\s*[:：]", text):
        issues.append("chat_role_marker_leak")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        issues.append("mojibake_or_replacement_char")
    return issues


def scan_gen_file(path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "rows": 0,
        "scored_empty": 0,
        "mojibake": 0,
        "template_leak": 0,
        "runtime_text": 0,
        "examples": [],
    }
    if not path.exists():
        return stats
    with path.open(encoding="utf-8", errors="replace") as f:
        for row_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                stats["runtime_text"] += 1
                stats["examples"].append({"line": row_idx, "issue": "json_error", "error": str(exc)[:200]})
                continue
            stats["rows"] += 1
            outputs = row.get("gen_output") or []
            prefixes = row.get("gen_prefix") or []
            upper = max(0, len(prefixes) - 2) if prefixes else max(0, len(outputs) - 2)
            if not isinstance(outputs, list):
                stats["scored_empty"] += 1
                continue
            for out_idx, output in enumerate(outputs):
                text = str(output or "")
                if not text.strip() and out_idx < upper:
                    stats["scored_empty"] += 1
                    if len(stats["examples"]) < 8:
                        stats["examples"].append({"id": row.get("id"), "output_idx": out_idx, "issue": "scored_empty"})
                    continue
                issues = output_issues(text)
                if "mojibake_or_replacement_char" in issues:
                    stats["mojibake"] += 1
                if "chat_template_token_leak" in issues or "chat_role_marker_leak" in issues:
                    stats["template_leak"] += 1
                if "runtime_error_text" in issues:
                    stats["runtime_text"] += 1
                if issues and len(stats["examples"]) < 8:
                    stats["examples"].append({"id": row.get("id"), "output_idx": out_idx, "issues": issues, "preview": text[:220]})
    return stats


def ps_rows() -> List[Tuple[int, int, str]]:
    proc = subprocess.run(["ps", "-eo", "pid=,ppid=,cmd="], text=True, capture_output=True, check=True)
    rows: List[Tuple[int, int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
    return rows


def descendants(pid: int, rows: List[Tuple[int, int, str]]) -> List[int]:
    children: Dict[int, List[int]] = {}
    for child, parent, _cmd in rows:
        children.setdefault(parent, []).append(child)
    result: List[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            result.append(child)
            stack.append(child)
    return result


def terminate_tag(tag: str, reason: str, evidence: Dict[str, Any], *, expected_out_root: str = "") -> None:
    rows = ps_rows()
    matched = []
    for pid, _ppid, cmd in rows:
        if f"--tag {tag}" not in cmd:
            continue
        if expected_out_root and expected_out_root not in cmd:
            continue
        matched.append((pid, cmd))
    pids = sorted({pid for pid, _cmd in matched for pid in [pid, *descendants(pid, rows)]})
    if not pids:
        append_event({"event": "stop_requested_but_no_pid", "tag": tag, "reason": reason, "expected_out_root": expected_out_root, "evidence": evidence})
        return
    append_event({"event": "terminating_run", "tag": tag, "reason": reason, "pids": pids, "expected_out_root": expected_out_root, "evidence": evidence})
    for sig, delay in ((signal.SIGTERM, 10), (signal.SIGKILL, 0)):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                append_event({"event": "kill_permission_error", "tag": tag, "pid": pid, "signal": sig.name, "error": str(exc)})
        if delay:
            time.sleep(delay)


def evaluate_status(path: Path, state: Dict[str, Any]) -> None:
    status = load_json(path)
    phase = status.get("phase")
    tag = status.get("tag")
    out_root = status.get("out_root")
    model_path = status.get("model_path")
    if not tag or phase not in {"generating", "packing_prompts"}:
        return
    key = str(path)
    if state.get(key, {}).get("stopped"):
        return
    run_dir = Path(out_root or "") / f"{Path(str(model_path or '')).name}_{tag}"
    gen_file = run_dir / "gen_only.jsonl"
    stats = scan_gen_file(gen_file)

    stop_reason = None
    if stats["scored_empty"] > SCORED_EMPTY_LIMIT:
        stop_reason = "systemic_scored_empty_outputs"
    elif stats["mojibake"] > MOJIBAKE_LIMIT:
        stop_reason = "mojibake_or_replacement_char_output"
    elif stats["template_leak"] >= TEMPLATE_LEAK_LIMIT:
        stop_reason = "chat_template_or_role_marker_leak"
    elif stats["runtime_text"] > 0:
        stop_reason = "runtime_error_text_in_outputs"

    now_ts = time.time()
    count = line_count(gen_file)
    rec = state.get(key, {})
    if rec.get("line_count") != count:
        rec = {"line_count": count, "last_progress_ts": now_ts, "tag": tag, "gen_file": str(gen_file)}
        state[key] = rec
    elif phase == "generating" and gpu_busy() and now_ts - float(rec.get("last_progress_ts", now_ts)) > NO_PROGRESS_SECONDS:
        stop_reason = "no_progress_while_gpu_busy"

    if stop_reason:
        terminate_tag(
            tag,
            stop_reason,
            {"status_path": str(path), "gen_file": str(gen_file), "partial_stats": stats},
            expected_out_root=str(out_root or ""),
        )
        state[key]["stopped"] = True
        state[key]["stop_reason"] = stop_reason


def main() -> None:
    log("guard monitor started")
    while True:
        state = load_state()
        try:
            for path in active_status_files():
                evaluate_status(path, state)
            save_state(state)
        except Exception as exc:
            append_event({"event": "guard_exception", "error": repr(exc)})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
