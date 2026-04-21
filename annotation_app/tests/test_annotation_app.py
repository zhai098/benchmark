import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from annotation_app.app import (
    app,
    ensure_dirs,
    progress_detail_path,
    progress_path,
    progress_summary_path,
    split_by_cut_points,
)


def test_split_by_cut_points():
    assert split_by_cut_points("abcde", [2, 4]) == ["ab", "cd", "e"]


def test_one_file_per_annotator_device_case(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    p1 = progress_path("ann", "dev", "case-1")
    p2 = progress_path("ann", "dev", "case-1")
    s1 = progress_summary_path("ann", "dev", "case-1")
    d1 = progress_detail_path("ann", "dev", "case-1")
    assert p1 == p2
    assert s1.name == "case-1.summary.json"
    assert d1.name == "case-1.detail.json"


def test_save_and_restore_progress_writes_summary_and_detail(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    payload = {
        "annotator_id": "u1",
        "device_id": "d1",
        "case_id": "c1",
        "status": "in_progress",
        "current_step": 3,
        "current_workflow_state": {"active_sample_idx": 0, "sample_cursor": 0, "workflow_state": "claims_assigned"},
        "current_annotations": {
            "selected_solution_text": "$x^2$",
            "steps": ["a"],
            "claims": [{"step_id": "s1", "claims": ["c1"]}],
            "dependencies": {"s1c1": ["s0c1"]},
        },
        "sample_decisions": [{"is_correct": True, "pipeline_status": "in_progress"}],
        "correct_solutions": [],
    }
    save = client.put("/api/save_progress", json=payload)
    assert save.status_code == 200

    summary_path = progress_summary_path("u1", "d1", "c1")
    detail_path = progress_detail_path("u1", "d1", "c1")
    assert summary_path.exists()
    assert detail_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert detail["schema_version"] == 2
    assert "sample_annotations" in detail
    assert "current_annotations" not in detail
    assert "selected_solution_text" not in summary
    assert summary["annotation_stats"]["total_steps"] == 1
    assert summary["annotation_stats"]["total_claims"] == 1
    assert summary["annotation_stats"]["total_dependencies"] == 1

    restored = client.get("/api/load_progress", query_string={"annotator_id": "u1", "device_id": "d1", "case_id": "c1"})
    assert restored.status_code == 200
    body = restored.get_json()
    assert body["found"] is True
    assert body["progress"]["current_step"] == 3
    assert body["progress"]["current_annotations"]["selected_solution_text"] == "$x^2$"
    assert body["progress"]["current_annotations"]["sample_annotations"]["0"]["selected_solution_text"] == "$x^2$"


def test_save_progress_accepts_sendbeacon_text_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()
    payload = {
        "annotator_id": "u2",
        "device_id": "d2",
        "case_id": "c2",
        "status": "in_progress",
        "current_annotations": {"selected_solution_text": "$y$"},
    }
    res = client.post("/api/save_progress", data=json.dumps(payload), content_type="text/plain")
    assert res.status_code == 200
    got = client.get("/api/load_progress", query_string={"annotator_id": "u2", "device_id": "d2", "case_id": "c2"})
    assert got.status_code == 200
    assert got.get_json()["progress"]["current_annotations"]["selected_solution_text"] == "$y$"


def test_save_progress_rejects_missing_required_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    res = client.put("/api/save_progress", json={"device_id": "d0", "current_annotations": {"selected_solution_text": "$x$"}})
    assert res.status_code == 400
    assert "annotator_id" in res.get_json()["error"]


def test_save_record_forces_completed_status(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()
    payload = {"annotator": "u3", "device_id": "d3", "case_id": "c3"}
    res = client.post("/api/save_record", json=payload)
    assert res.status_code == 200
    got = client.get("/api/load_progress", query_string={"annotator_id": "u3", "device_id": "d3", "case_id": "c3"})
    assert got.status_code == 200
    assert got.get_json()["progress"]["status"] == "completed"


def test_save_progress_reports_unchanged_for_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()
    payload = {
        "annotator_id": "u4",
        "device_id": "d4",
        "case_id": "c4",
        "current_annotations": {"selected_solution_text": "$z$"},
    }
    first = client.put("/api/save_progress", json=payload)
    assert first.status_code == 200
    second = client.put("/api/save_progress", json=payload)
    assert second.status_code == 200
    assert second.get_json()["unchanged"] is True


def test_ensure_dirs_creates_logs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.DATA_DIR", tmp_path)
    monkeypatch.setattr("annotation_app.app.GUIDE_PATH", tmp_path / "guideline.md")
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    app.config.pop("_logging_configured", None)

    ensure_dirs()

    assert (tmp_path / "logs").exists()


def test_save_progress_writes_local_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.DATA_DIR", tmp_path)
    monkeypatch.setattr("annotation_app.app.GUIDE_PATH", tmp_path / "guideline.md")
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    app.config.pop("_logging_configured", None)
    client = app.test_client()

    payload = {
        "annotator_id": "u5",
        "device_id": "d5",
        "case_id": "c5",
        "current_annotations": {"selected_solution_text": "$w$"},
    }
    res = client.put("/api/save_progress", json=payload)
    assert res.status_code == 200

    access_log = (tmp_path / "logs" / "access.log").read_text(encoding="utf-8")
    app_log = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    assert "request.completed" in access_log
    assert "progress.saved" in app_log


def test_review_records_reads_new_layout_and_aggregates_sample_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    payload = {
        "annotator_id": "u1",
        "device_id": "d1",
        "case_id": "c1",
        "status": "completed",
        "current_step": 5,
        "current_workflow_state": {
            "active_sample_idx": None,
            "sample_cursor": 0,
            "workflow_state": "completed",
            "problem_quality_screening": {"decision": "pass"},
        },
        "current_annotations": {
            "sample_annotations": {
                "0": {
                    "selected_solution_text": "sol",
                    "steps": ["s1", "s2"],
                    "claims": [{"step_id": "s1", "claims": ["c1", "c2"]}],
                    "dependencies": {"s1c1": ["d1"]},
                    "workflow_state": "completed",
                }
            }
        },
        "sample_decisions": [{"is_correct": True, "pipeline_status": "completed"}],
        "correct_solutions": [],
    }
    assert client.put("/api/save_progress", json=payload).status_code == 200

    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200
    res = client.get("/api/review_records")
    assert res.status_code == 200
    row = next(r for r in res.get_json()["records"] if r["case_id"] == "c1")
    assert row["sample_valid_count"] == 1
    assert row["total_samples"] == 1
    assert row["completed_samples"] == 1
    assert row["step_count"] == 2
    assert row["claim_count"] == 2
    assert row["dependency_count"] == 1
    assert row["load_error"] is False


def test_review_records_falls_back_to_detail_when_summary_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    payload = {
        "annotator_id": "u6",
        "device_id": "d6",
        "case_id": "c6",
        "current_workflow_state": {"workflow_state": "completed"},
        "current_annotations": {
            "sample_annotations": {
                "0": {
                    "steps": ["s1"],
                    "claims": [{"step_id": "s1", "claims": ["c1"]}],
                    "dependencies": {"s1c1": ["d0"]},
                    "workflow_state": "completed",
                }
            }
        },
        "sample_decisions": [{"is_correct": True, "pipeline_status": "completed"}],
    }
    assert client.put("/api/save_progress", json=payload).status_code == 200
    progress_summary_path("u6", "d6", "c6").unlink()

    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200
    res = client.get("/api/review_records")
    assert res.status_code == 200
    row = next(r for r in res.get_json()["records"] if r["case_id"] == "c6")
    assert row["step_count"] == 1
    assert row["claim_count"] == 1
    assert row["dependency_count"] == 1


def test_review_records_isolates_bad_pair_files(tmp_path, monkeypatch):
    ann = tmp_path / "annotations/u1/d1"
    ann.mkdir(parents=True)
    (tmp_path / "records").mkdir(parents=True)
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")

    (ann / "broken.summary.json").write_text("{bad", encoding="utf-8")
    (ann / "broken.detail.json").write_text("{bad", encoding="utf-8")

    client = app.test_client()
    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200
    res = client.get("/api/review_records")
    assert res.status_code == 200
    row = next(r for r in res.get_json()["records"] if "broken.summary.json" in r["file"])
    assert row["load_error"] is True


def test_review_records_includes_legacy_records_dir_entries(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    (tmp_path / "records").mkdir(parents=True)
    legacy_record = {
        "annotator": "legacy-reviewer",
        "case_id": "records-case",
        "status": "completed",
        "saved_at_utc": "2026-01-01T00:00:00+00:00",
        "sample_validation": [{"is_correct": True, "pipeline_status": "completed"}],
        "steps": ["s1"],
        "claims": [{"claims": ["c1", "c2"]}],
        "dependencies": {"s1c1": ["d0"]},
    }
    (tmp_path / "records" / "records-case.json").write_text(json.dumps(legacy_record), encoding="utf-8")

    client = app.test_client()
    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200
    res = client.get("/api/review_records")
    assert res.status_code == 200
    row = next(r for r in res.get_json()["records"] if r["case_id"] == "records-case")
    assert row["file"] == "legacy/records-case.json"
    assert row["sample_valid_count"] == 1
    assert row["step_count"] == 1
    assert row["claim_count"] == 2


def test_review_records_skips_legacy_annotation_json_when_pair_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    payload = {
        "annotator_id": "u8",
        "device_id": "d8",
        "case_id": "c8",
        "current_workflow_state": {"workflow_state": "completed"},
        "current_annotations": {
            "sample_annotations": {
                "0": {"steps": ["s1"], "claims": [{"step_id": "s1", "claims": ["c1"]}], "workflow_state": "completed"}
            }
        },
        "sample_decisions": [{"is_correct": True, "pipeline_status": "completed"}],
    }
    assert client.put("/api/save_progress", json=payload).status_code == 200

    legacy = {
        "annotator_id": "u8",
        "device_id": "d8",
        "case_id": "c8",
        "status": "in_progress",
        "current_annotations": {"selected_solution_text": "$stale$"},
    }
    progress_path("u8", "d8", "c8").write_text(json.dumps(legacy), encoding="utf-8")

    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200
    res = client.get("/api/review_records")
    assert res.status_code == 200
    rows = [r for r in res.get_json()["records"] if r["case_id"] == "c8"]
    assert len(rows) == 1
    assert rows[0]["file"].endswith("c8.summary.json")


def test_load_progress_reads_legacy_single_file(tmp_path, monkeypatch):
    ann = tmp_path / "annotations/u7/d7"
    ann.mkdir(parents=True)
    (tmp_path / "records").mkdir(parents=True)
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")

    legacy = {
        "annotator_id": "u7",
        "device_id": "d7",
        "case_id": "c7",
        "status": "in_progress",
        "current_step": 3,
        "current_workflow_state": {"active_sample_idx": 0, "workflow_state": "claims_assigned"},
        "current_annotations": {"selected_solution_text": "$legacy$"},
        "sample_decisions": [{"is_correct": True, "pipeline_status": "in_progress"}],
        "correct_solutions": [],
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:01+00:00",
    }
    progress_path("u7", "d7", "c7").write_text(json.dumps(legacy), encoding="utf-8")

    client = app.test_client()
    res = client.get("/api/load_progress", query_string={"annotator_id": "u7", "device_id": "d7", "case_id": "c7"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["source"] == "exact_device:legacy"
    assert body["progress"]["current_annotations"]["selected_solution_text"] == "$legacy$"


def test_load_progress_can_fallback_across_devices_for_same_annotator(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    payload = {
        "annotator_id": "u9",
        "device_id": "device-a",
        "case_id": "c9",
        "current_step": 4,
        "client_revision": 101,
        "current_workflow_state": {"active_sample_idx": 0, "sample_cursor": 0, "workflow_state": "claims_checked"},
        "current_annotations": {"selected_solution_text": "$fallback$"},
        "sample_decisions": [{"is_correct": True, "pipeline_status": "in_progress"}],
    }
    assert client.put("/api/save_progress", json=payload).status_code == 200

    res = client.get("/api/load_progress", query_string={"annotator_id": "u9", "device_id": "device-b", "case_id": "c9"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["found"] is True
    assert body["source"].startswith("annotator_fallback:")
    assert body["progress"]["current_annotations"]["selected_solution_text"] == "$fallback$"
    assert body["progress"]["client_revision"] == 101


def test_stale_client_revision_cannot_overwrite_newer_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")
    client = app.test_client()

    newer = {
        "annotator_id": "u10",
        "device_id": "d10",
        "case_id": "c10",
        "client_revision": 200,
        "current_workflow_state": {"active_sample_idx": 0, "sample_cursor": 0, "workflow_state": "claims_checked"},
        "current_annotations": {"selected_solution_text": "$new$"},
        "sample_decisions": [{"is_correct": True, "pipeline_status": "in_progress"}],
    }
    older = {
        "annotator_id": "u10",
        "device_id": "d10",
        "case_id": "c10",
        "client_revision": 150,
        "current_workflow_state": {"active_sample_idx": 0, "sample_cursor": 0, "workflow_state": "steps_segmented"},
        "current_annotations": {"selected_solution_text": "$old$"},
        "sample_decisions": [{"is_correct": True, "pipeline_status": "in_progress"}],
    }

    first = client.put("/api/save_progress", json=newer)
    assert first.status_code == 200
    second = client.put("/api/save_progress", json=older)
    assert second.status_code == 200
    assert second.get_json()["ignored_stale"] is True

    got = client.get("/api/load_progress", query_string={"annotator_id": "u10", "device_id": "d10", "case_id": "c10"})
    assert got.status_code == 200
    assert got.get_json()["progress"]["current_annotations"]["selected_solution_text"] == "$new$"
    assert got.get_json()["progress"]["client_revision"] == 200


def test_root_route_shows_home_entry():
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Annotation Workspace" in body
    assert "JSONL 路径" in body


def test_frontend_has_katex_and_copy_ui():
    tpl = Path("annotation_app/templates/annotator.html").read_text(encoding="utf-8")
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "vendor/katex/katex.min.css" in tpl
    assert "vendor/katex/katex.min.js" in tpl
    assert "vendor/katex/auto-render.min.js" in tpl
    assert "jsdelivr" not in tpl
    assert "copySolutionRaw" in js
    assert "已复制" in js


def test_frontend_pipeline_isolation_rules_present():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "delete st.sample_annotations[i]" in js
    assert "st.correct_solutions.push" in js
    assert "wa.workflow_state = 'completed'" in js
    assert "Step 1：单样本验证入口（严格串行）" in js


def test_frontend_claim_preview_uses_math_fallback_rendering():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    css = Path("annotation_app/static/styles.css").read_text(encoding="utf-8")
    assert "function renderMathPreviewBlock" in js
    assert "${renderMathPreviewBlock(cl.text || '', '当前 Claim 为空')}" in js
    assert "${renderMathPreviewBlock(step.text || '', '当前 Step 暂无内容')}" in js
    assert ".compact-rendered-math" in css


def test_frontend_restore_logic_uses_local_draft_fallback_before_resetting_case_state():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "const draftCachePrefix = 'annotation_draft_v1'" in js
    assert "function writeDraftCache" in js
    assert "function readDraftCache" in js
    assert "function hasMeaningfulProgress" in js
    assert "applyRestoredProgress(caseId, cachedDraft.progress, 'local_draft:not_found')" in js
    assert "applyRestoredProgress(caseId, cachedDraft.progress, 'local_draft:error')" in js
    assert "已恢复本地草稿（服务器无记录）" in js
    assert "已恢复本地草稿（服务器恢复失败）" in js


def test_frontend_case_list_shows_progress_percent_and_status_colors():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    css = Path("annotation_app/static/styles.css").read_text(encoding="utf-8")
    assert "function getTaskProgressSummary" in js
    assert "task-nav-percent" in js
    assert "task-progress-fill" in js
    assert "task-status-done" in css
    assert "task-status-active" in css
    assert "task-status-rejected" in css


def test_annotator_cannot_access_reviewer_apis():
    client = app.test_client()
    res = client.get("/api/review_records")
    assert res.status_code == 403


def test_reviewer_can_edit_guideline_and_read_it_back(tmp_path, monkeypatch):
    monkeypatch.setattr("annotation_app.app.DATA_DIR", tmp_path)
    monkeypatch.setattr("annotation_app.app.GUIDE_PATH", tmp_path / "guideline.md")
    monkeypatch.setattr("annotation_app.app.ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr("annotation_app.app.RECORDS_DIR", tmp_path / "records")

    client = app.test_client()
    bad = client.put("/api/guideline", json={"content": "# x"})
    assert bad.status_code == 403

    login = client.post("/api/session/role", json={"role": "reviewer", "access_key": "reviewer"})
    assert login.status_code == 200

    updated = client.put("/api/guideline", json={"content": "# 新说明\n- A"})
    assert updated.status_code == 200

    got = client.get("/api/guideline")
    assert got.status_code == 200
    assert "# 新说明" in got.get_json()["content"]
