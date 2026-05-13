import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from annotation_app.app import app


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DATASET_FIXTURE = FIXTURES_DIR / "annotator_e2e_dataset.jsonl"


def test_load_jsonl_accepts_fixture_dataset():
    client = app.test_client()
    res = client.post("/api/load_jsonl", json={"path": str(DATASET_FIXTURE)})
    assert res.status_code == 200
    items = res.get_json()["items"]
    assert len(items) == 3
    assert [item["id"] for item in items] == ["case-valid", "case-bad-latex", "case-empty-claims"]
    assert len(items[0]["samples"]) == 2
    assert len(items[1]["samples"][0]["claims"]) == 5
    assert items[2]["samples"][0]["claims"] == []


def test_load_jsonl_requires_path():
    client = app.test_client()
    res = client.post("/api/load_jsonl", json={})
    assert res.status_code == 400
    assert "JSONL 文件路径" in res.get_json()["error"]


def test_load_jsonl_reports_missing_file():
    client = app.test_client()
    res = client.post("/api/load_jsonl", json={"path": str(FIXTURES_DIR / "missing.jsonl")})
    assert res.status_code == 404
    assert "文件不存在" in res.get_json()["error"]


def test_load_jsonl_rejects_directory_path(tmp_path):
    client = app.test_client()
    res = client.post("/api/load_jsonl", json={"path": str(tmp_path)})
    assert res.status_code == 404
    assert "文件不存在" in res.get_json()["error"]


def test_load_jsonl_reports_bad_jsonl(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")
    client = app.test_client()
    res = client.post("/api/load_jsonl", json={"path": str(broken)})
    assert res.status_code == 400
    assert "JSONL 解析失败" in res.get_json()["error"]


def test_split_steps_filters_invalid_cut_points():
    client = app.test_client()
    res = client.post(
        "/api/split_steps",
        json={"solution": "abcde", "cut_points": [-1, 0, 2, 2, 4, 99, "bad"]},
    )
    assert res.status_code == 200
    assert res.get_json()["steps"] == ["ab", "cd", "e"]


def test_frontend_math_render_helper_uses_safe_katex_options():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "throwOnError: false" in js
    assert "strict: 'ignore'" in js
    assert "function renderLatexWithFallback" in js
    assert "return `<pre>${htmlSafe}</pre>`;" in js


def test_frontend_claim_views_define_empty_state_messages():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "当前 solution 未提供预切分 claim" in js
    assert "当前 Claim 为空" in js
    assert "当前 Step 暂无内容" in js
    assert "renderMathPreviewBlock(targetStep.text || '', '当前 Step 暂无内容')" in js
    assert "renderMathPreviewBlock(cand.text, '当前 Claim 为空')" in js
    assert "renderMathPreviewBlock(cl.text || '', '当前 Claim 为空')" in js
    assert "renderMathPreviewBlock(step.text || '', '当前 Step 暂无内容')" in js


def test_frontend_problem_reject_and_step_range_guards_exist():
    js = Path("annotation_app/static/app.js").read_text(encoding="utf-8")
    assert "请先选择拒绝原因。" in js
    assert "选择 Other 时请填写简短说明。" in js
    assert "题目 ${rejectedCaseId} 已按低质量筛除，已自动跳到下一题" in js
    assert "的边界无效，请重新选择起止 Claim。" in js
    assert "必须从 Claim #" in js


def test_fixture_dataset_is_valid_annotation_jsonl():
    rows = [json.loads(line) for line in DATASET_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    for row in rows:
        assert {"id", "question", "reference_answer", "known_solutions", "samples"} <= set(row.keys())
        assert isinstance(row["samples"], list)


def test_fixture_dataset_covers_claim_visibility_edge_cases():
    rows = [json.loads(line) for line in DATASET_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    broken = next(row for row in rows if row["id"] == "case-bad-latex")
    empty = next(row for row in rows if row["id"] == "case-empty-claims")

    broken_claims = broken["samples"][0]["claims"]
    assert len(broken_claims) == 5
    assert any("$x^{2" in claim for claim in broken_claims)
    assert any("\\badcommand" in claim for claim in broken_claims)
    assert empty["samples"][0]["claims"] == []
