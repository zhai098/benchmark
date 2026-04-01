import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import json

from annotation_app.app import app, progress_path, split_by_cut_points



def test_split_by_cut_points():
    assert split_by_cut_points("abcde", [2, 4]) == ["ab", "cd", "e"]


def test_one_file_per_annotator_device_case(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    p1 = progress_path('ann', 'dev', 'case-1')
    p2 = progress_path('ann', 'dev', 'case-1')
    assert p1 == p2


def test_save_and_restore_progress(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')
    client = app.test_client()

    payload = {
        'annotator_id': 'u1',
        'device_id': 'd1',
        'case_id': 'c1',
        'status': 'in_progress',
        'current_step': 3,
        'current_workflow_state': {'active_sample_idx': 0, 'workflow_state': 'claims_assigned'},
        'current_annotations': {
            'selected_solution_text': '$x^2$',
            'steps': ['a'],
            'claims': [],
            'dependencies': {},
        },
        'sample_decisions': [{'is_correct': True}],
        'correct_solutions': [],
    }
    save = client.put('/api/save_progress', json=payload)
    assert save.status_code == 200

    restored = client.get('/api/load_progress', query_string={'annotator_id': 'u1', 'device_id': 'd1', 'case_id': 'c1'})
    assert restored.status_code == 200
    body = restored.get_json()
    assert body['found'] is True
    assert body['progress']['current_step'] == 3
    assert body['progress']['current_annotations']['selected_solution_text'] == '$x^2$'


def test_save_progress_accepts_sendbeacon_text_payload(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')
    client = app.test_client()
    payload = {
        'annotator_id': 'u2',
        'device_id': 'd2',
        'case_id': 'c2',
        'status': 'in_progress',
        'current_annotations': {'selected_solution_text': '$y$'},
    }
    res = client.post('/api/save_progress', data=json.dumps(payload), content_type='text/plain')
    assert res.status_code == 200
    got = client.get('/api/load_progress', query_string={'annotator_id': 'u2', 'device_id': 'd2', 'case_id': 'c2'})
    assert got.status_code == 200
    assert got.get_json()['progress']['current_annotations']['selected_solution_text'] == '$y$'


def test_save_record_forces_completed_status(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')
    client = app.test_client()
    payload = {'annotator': 'u3', 'device_id': 'd3', 'case_id': 'c3'}
    res = client.post('/api/save_record', json=payload)
    assert res.status_code == 200
    got = client.get('/api/load_progress', query_string={'annotator_id': 'u3', 'device_id': 'd3', 'case_id': 'c3'})
    assert got.status_code == 200
    assert got.get_json()['progress']['status'] == 'completed'


def test_save_progress_reports_unchanged_for_noop(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')
    client = app.test_client()
    payload = {
        'annotator_id': 'u4',
        'device_id': 'd4',
        'case_id': 'c4',
        'current_annotations': {'selected_solution_text': '$z$'},
    }
    first = client.put('/api/save_progress', json=payload)
    assert first.status_code == 200
    second = client.put('/api/save_progress', json=payload)
    assert second.status_code == 200
    assert second.get_json()['unchanged'] is True


def test_review_records_reads_new_layout(tmp_path, monkeypatch):
    ann = tmp_path / 'annotations/u1/d1'
    ann.mkdir(parents=True)
    (tmp_path / 'records').mkdir(parents=True)
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')

    data = {
        'annotator_id': 'u1',
        'device_id': 'd1',
        'case_id': 'c1',
        'status': 'completed',
        'updated_at_utc': '2026-01-01T00:00:00+00:00',
        'current_annotations': {'steps': [], 'claims': [], 'dependencies': {}},
        'sample_decisions': [{'is_correct': False}],
    }
    (ann / 'c1.json').write_text(json.dumps(data), encoding='utf-8')

    client = app.test_client()
    login = client.post('/api/session/role', json={'role': 'reviewer', 'access_key': 'reviewer'})
    assert login.status_code == 200
    res = client.get('/api/review_records')
    assert res.status_code == 200
    rows = res.get_json()['records']
    assert any(r['file'] == 'u1/d1/c1.json' for r in rows)


def test_root_route_uses_annotator_workspace():
    client = app.test_client()
    res = client.get('/')
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert 'Annotation Workspace' in body
    assert 'JSONL 路径' in body


def test_frontend_has_katex_and_copy_ui():
    tpl = Path('annotation_app/templates/annotator.html').read_text(encoding='utf-8')
    js = Path('annotation_app/static/app.js').read_text(encoding='utf-8')
    assert 'katex' in tpl.lower()
    assert 'copySolutionRaw' in js
    assert 'Copied' in js

def test_frontend_pipeline_isolation_rules_present():
    js = Path('annotation_app/static/app.js').read_text(encoding='utf-8')
    assert 'delete st.sample_annotations[i]' in js
    assert 'st.correct_solutions.push' in js
    assert "wa.workflow_state = 'completed'" in js
    assert 'Step 1：单样本验证入口（严格串行）' in js


def test_annotator_cannot_access_reviewer_apis():
    client = app.test_client()
    res = client.get('/api/review_records')
    assert res.status_code == 403


def test_reviewer_can_edit_guideline_and_read_it_back(tmp_path, monkeypatch):
    monkeypatch.setattr('annotation_app.app.DATA_DIR', tmp_path)
    monkeypatch.setattr('annotation_app.app.GUIDE_PATH', tmp_path / 'guideline.md')
    monkeypatch.setattr('annotation_app.app.ANNOTATIONS_DIR', tmp_path / 'annotations')
    monkeypatch.setattr('annotation_app.app.RECORDS_DIR', tmp_path / 'records')

    client = app.test_client()
    bad = client.put('/api/guideline', json={'content': '# x'})
    assert bad.status_code == 403

    login = client.post('/api/session/role', json={'role': 'reviewer', 'access_key': 'reviewer'})
    assert login.status_code == 200

    updated = client.put('/api/guideline', json={'content': '# 新说明\n- A'})
    assert updated.status_code == 200

    got = client.get('/api/guideline')
    assert got.status_code == 200
    assert '# 新说明' in got.get_json()['content']
