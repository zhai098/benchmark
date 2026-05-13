#!/usr/bin/env python3
"""Download every model listed in model/download_manifest.tsv.

This script is intentionally resumable. Existing target directories are skipped,
and existing directories elsewhere under /data/pretrain with the same basename
are linked into the manifest target path to avoid duplicating multi-hundred-GB
models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


REPO = Path("/home/zhaipengxiang/benchmark")
DEFAULT_MANIFEST = REPO / "model/download_manifest.tsv"
DEFAULT_PRETRAIN_ROOT = Path("/data/pretrain")
SKIP_DIR_MARKERS = {".cache", "refs", "blobs", "snapshots"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def dir_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink():
        return path.exists()
    if not path.is_dir():
        return False
    for child in path.rglob("*"):
        if child.is_file() and ".cache" not in child.parts:
            return True
    return False


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True, stderr=subprocess.DEVNULL)
        return int(out.split()[0])
    except Exception:
        return 0


def human_size(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def manifest_target(local_dir: str, pretrain_root: Path) -> Path:
    prefix = "data/pretrain/"
    if local_dir.startswith(prefix):
        return pretrain_root / local_dir[len(prefix) :]
    path = Path(local_dir)
    if path.is_absolute():
        return path
    return REPO / path


def load_manifest(path: Path, pretrain_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = dict(row)
            row["target"] = str(manifest_target(row["local_dir"], pretrain_root))
            rows.append(row)
    return rows


def index_pretrain_dirs(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if any(part in SKIP_DIR_MARKERS for part in path.parts):
            continue
        index.setdefault(path.name, []).append(path)
    return index


def choose_existing_alternate(target: Path, index: dict[str, list[Path]]) -> Path | None:
    for candidate in sorted(index.get(target.name, []), key=lambda p: len(p.parts)):
        if candidate == target:
            continue
        if dir_has_files(candidate):
            return candidate
    return None


def link_existing(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    os.symlink(source, target)


def status_key(row: dict[str, Any]) -> str:
    return str(row["model_name"])


def update_status(status_path: Path, lock_path: Path, update: dict[str, Any]) -> None:
    # Atomic-enough for the low write rate here. Avoid extra dependencies.
    for _ in range(200):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise RuntimeError(f"Could not acquire status lock {lock_path}")
    try:
        if status_path.exists():
            data = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            data = {"created_at_utc": utc_now(), "models": {}}
        data["updated_at_utc"] = utc_now()
        model = update.pop("model_name")
        previous = data["models"].get(model, {})
        previous.update(update)
        previous["model_name"] = model
        previous["updated_at_utc"] = utc_now()
        data["models"][model] = previous
        write_json(status_path, data)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def download_one(
    row: dict[str, Any],
    *,
    status_path: Path,
    events_path: Path,
    lock_path: Path,
    pretrain_root: Path,
    per_model_workers: int,
    verify_existing: bool,
    token: bool | None,
) -> dict[str, Any]:
    model = row["model_name"]
    target = Path(row["target"])
    repo_id = row["hf_source"]
    include_pattern = row.get("include_pattern") or ""
    target_for_download = target.parent if include_pattern else target
    allow_patterns = [include_pattern] if include_pattern else None
    start = time.time()

    base_event = {
        "model_name": model,
        "repo_id": repo_id,
        "target": str(target),
        "include_pattern": include_pattern,
        "gated": row.get("gated"),
    }
    update_status(status_path, lock_path, {**base_event, "status": "running", "started_at_utc": utc_now()})
    append_jsonl(events_path, {**base_event, "event": "start", "created_at_utc": utc_now()})

    try:
        if dir_has_files(target) and not verify_existing:
            result = {
                **base_event,
                "status": "already_present",
                "size_bytes": du_bytes(target),
                "size": human_size(du_bytes(target)),
                "duration_sec": round(time.time() - start, 1),
            }
            update_status(status_path, lock_path, result.copy())
            append_jsonl(events_path, {**result, "event": "done", "created_at_utc": utc_now()})
            return result

        target.parent.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id,
            repo_type="model",
            local_dir=target_for_download,
            allow_patterns=allow_patterns,
            max_workers=per_model_workers,
            token=token,
            etag_timeout=60,
        )
        size_target = target if include_pattern else Path(local_dir)
        result = {
            **base_event,
            "status": "downloaded",
            "snapshot_local_dir": str(local_dir),
            "size_bytes": du_bytes(size_target),
            "size": human_size(du_bytes(size_target)),
            "duration_sec": round(time.time() - start, 1),
        }
        update_status(status_path, lock_path, result.copy())
        append_jsonl(events_path, {**result, "event": "done", "created_at_utc": utc_now()})
        return result
    except Exception as exc:
        result = {
            **base_event,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback": traceback.format_exc()[-5000:],
            "duration_sec": round(time.time() - start, 1),
        }
        update_status(status_path, lock_path, result.copy())
        append_jsonl(events_path, {**result, "event": "failed", "created_at_utc": utc_now()})
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--parallel-models", type=int, default=3)
    parser.add_argument("--per-model-workers", type=int, default=8)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--token", choices=["auto", "none"], default="auto")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "download_status.json"
    events_path = args.log_dir / "download_events.jsonl"
    lock_path = args.log_dir / ".status.lock"
    rows = load_manifest(args.manifest, args.pretrain_root)
    index = index_pretrain_dirs(args.pretrain_root)

    planned: list[dict[str, Any]] = []
    linked: list[dict[str, Any]] = []
    for row in rows:
        target = Path(row["target"])
        if not dir_has_files(target):
            alt = choose_existing_alternate(target, index)
            if alt:
                link_existing(target, alt)
                item = {
                    "model_name": row["model_name"],
                    "repo_id": row["hf_source"],
                    "target": str(target),
                    "status": "linked_existing",
                    "linked_from": str(alt),
                    "size_bytes": du_bytes(target),
                    "size": human_size(du_bytes(target)),
                }
                linked.append(item)
                update_status(status_path, lock_path, item.copy())
                append_jsonl(events_path, {**item, "event": "linked_existing", "created_at_utc": utc_now()})
        planned.append(row)

    summary = {
        "created_at_utc": utc_now(),
        "manifest": str(args.manifest),
        "pretrain_root": str(args.pretrain_root),
        "parallel_models": args.parallel_models,
        "per_model_workers": args.per_model_workers,
        "total_manifest_models": len(rows),
        "linked_existing_count": len(linked),
        "linked_existing": linked,
    }
    write_json(args.log_dir / "download_plan.json", summary)

    token: bool | None = True if args.token == "auto" else None
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_models)) as pool:
        futures = [
            pool.submit(
                download_one,
                row,
                status_path=status_path,
                events_path=events_path,
                lock_path=lock_path,
                pretrain_root=args.pretrain_root,
                per_model_workers=max(1, args.per_model_workers),
                verify_existing=args.verify_existing,
                token=token,
            )
            for row in planned
        ]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    final = {
        **summary,
        "finished_at_utc": utc_now(),
        "status_counts": counts,
        "results": results,
    }
    write_json(args.log_dir / "download_final_summary.json", final)
    print(json.dumps({"event": "all_done", "status_counts": counts}, ensure_ascii=False), flush=True)
    return 0 if not counts.get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
