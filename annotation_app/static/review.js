async function loadGuideline() {
  const res = await fetch('/api/guideline');
  const data = await res.json();
  document.getElementById('guidelineEditor').value = data.content || '';
}

async function saveGuideline() {
  const content = document.getElementById('guidelineEditor').value;
  const res = await fetch('/api/guideline', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || '保存失败');
    return;
  }
  alert('说明已保存并生效');
}

async function loadRecords() {
  const res = await fetch('/api/review_records');
  if (res.status === 403) {
    alert('无 reviewer 权限，请从首页重新登录');
    location.href = '/';
    return;
  }
  const data = await res.json();
  const records = data.records || [];
  const tb = document.querySelector('#recordsTable tbody');
  tb.innerHTML = '';

  const summary = {
    total: records.length,
    completed: records.filter(r => r.status === 'completed').length,
    annotators: new Set(records.map(r => r.annotator)).size,
    completedSamples: records.reduce((acc, r) => acc + (r.completed_samples || 0), 0),
  };
  document.getElementById('summary').textContent = `记录数: ${summary.total}，已完成 case: ${summary.completed}，已完成样本: ${summary.completedSamples}，标注者数: ${summary.annotators}`;

  records.forEach((r) => {
    const tr = document.createElement('tr');
    if (r.load_error) tr.style.background = '#fff0f0';
    const progress = `${r.completed_samples || 0}/${r.total_samples || 0}`;
    const status = r.load_error
      ? `异常: ${r.error_type || 'ReadError'}`
      : (r.latest_workflow_state || r.status || '');
    tr.innerHTML = `<td>${r.file}</td><td>${r.annotator}</td><td>${r.case_id}</td><td>${r.sample_valid_count}</td><td>${progress}</td><td>${r.step_count}</td><td>${r.claim_count}</td><td>${r.dependency_count}</td><td>${status}</td><td>${r.saved_at_utc}</td>`;
    tr.onclick = () => { document.getElementById('detail').textContent = JSON.stringify(r.raw, null, 2); };
    tb.appendChild(tr);
  });
}

loadGuideline();
loadRecords();
