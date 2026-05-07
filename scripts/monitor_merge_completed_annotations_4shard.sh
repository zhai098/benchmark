#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaipengxiang/benchmark

unset LD_LIBRARY_PATH
unset PYTHONSTARTUP
unset PYTHON_BASIC_REPL

PY=/home/zhaipengxiang/miniconda3/envs/vllm/bin/python3.12
BASE=/home/zhaipengxiang/benchmark/artifacts/model_outputs/completed_annotations
MODEL_NAME=granite-4.1-8b
COMBINED_TAG=completed_annotations_full_granite_4_1_8b_4shard_combined
COMBINED_DIR="$BASE/${MODEL_NAME}_${COMBINED_TAG}"
STATUS=logs/completed_annotations_combined_status.json

write_status() {
  local phase="$1"
  local extra="${2:-{}}"
  "$PY" - "$STATUS" "$phase" "$extra" <<'PY'
import json, sys
from datetime import datetime, timezone
path, phase, extra_raw = sys.argv[1:4]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except Exception:
    data = {}
try:
    extra = json.loads(extra_raw)
except Exception:
    extra = {"note": extra_raw}
data.update(extra)
data["phase"] = phase
data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
open(path, "w", encoding="utf-8").write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY
}

write_status "waiting_for_shards" '{"expected_rows":317}'

while true; do
  failed=0
  completed=0
  for s in 00 01 02 03; do
    if [ ! -f "logs/completed_annotations_shard_${s}_status.json" ]; then
      continue
    fi
    phase=$("$PY" - "logs/completed_annotations_shard_${s}_status.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("phase", ""))
PY
)
    if [ "$phase" = "failed" ]; then
      failed=1
    fi
    if [ "$phase" = "completed" ]; then
      completed=$((completed + 1))
    fi
  done
  write_status "waiting_for_shards" "{\"completed_shards\":$completed,\"failed_shards\":$failed}"
  if [ "$failed" -ne 0 ]; then
    write_status "failed" '{"error":"at least one shard failed"}'
    exit 1
  fi
  if [ "$completed" -eq 4 ]; then
    break
  fi
  sleep 60
done

write_status "merging" '{}'
mkdir -p "$COMBINED_DIR"
rm -f "$COMBINED_DIR/gen_only.jsonl"
for s in 00 01 02 03; do
  cat "$BASE/${MODEL_NAME}_completed_annotations_full_granite_4_1_8b_shard${s}/gen_only.jsonl" >> "$COMBINED_DIR/gen_only.jsonl"
done

"$PY" - "$COMBINED_DIR/gen_only.jsonl" "$COMBINED_DIR/merge_validation.json" <<'PY'
import json, sys
from collections import Counter
gen_file, out_file = sys.argv[1:3]
rows = []
ids = []
case_ids = []
bad = []
with open(gen_file, encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        try:
            row = json.loads(line)
        except Exception as exc:
            bad.append({"line": idx, "error": str(exc)})
            continue
        rows.append(row)
        ids.append(row.get("id") or row.get("annotation_uid") or row.get("case_id"))
        case_ids.append(row.get("case_id"))
counts = Counter(ids)
dup_ids = sorted([k for k, v in counts.items() if k and v > 1])
result = {
    "rows": len(rows),
    "unique_ids": len(set(ids)),
    "unique_case_ids": len(set(case_ids)),
    "duplicate_ids": dup_ids[:20],
    "duplicate_id_count": len(dup_ids),
    "json_errors": bad[:20],
    "json_error_count": len(bad),
}
ok = result["rows"] == 317 and result["unique_ids"] == 317 and result["duplicate_id_count"] == 0 and result["json_error_count"] == 0
result["ok"] = ok
open(out_file, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
if not ok:
    raise SystemExit(f"merge validation failed: {result}")
PY

write_status "packing_prompts" "{\"gen_file\":\"$COMBINED_DIR/gen_only.jsonl\"}"
"$PY" tools/prompts/pack_prompt.py \
  --gen_file "$COMBINED_DIR/gen_only.jsonl" \
  --out_dir "$COMBINED_DIR/packed_prompts" \
  --write_all

"$PY" - "$COMBINED_DIR" "$STATUS" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
run_dir = pathlib.Path(sys.argv[1])
status = pathlib.Path(sys.argv[2])
cache = run_dir / "packed_prompts" / "cache_prompts"
case_files = sorted(cache.glob("case_*_cache.jsonl")) if cache.exists() else []
all_cache = cache / "ALL_cache.jsonl"
validation = json.loads((run_dir / "merge_validation.json").read_text(encoding="utf-8"))
data = {
    "phase": "completed",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "combined_dir": str(run_dir),
    "gen_file": str(run_dir / "gen_only.jsonl"),
    "merge_validation": validation,
    "all_cache": str(all_cache) if all_cache.exists() else None,
    "packed_case_files": len(case_files),
}
status.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not all_cache.exists() or len(case_files) != 317:
    raise SystemExit(f"pack validation failed: all_cache={all_cache.exists()} case_files={len(case_files)}")
PY
