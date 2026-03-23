const $ = (id) => document.getElementById(id);

async function api(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'API error');
  return data;
}

async function refreshSummary() {
  const data = await api('/api/review/summary');
  $('annotatorList').innerHTML = '';
  data.annotators.forEach(row => {
    const div = document.createElement('div');
    div.className = 'sample-card';
    div.innerHTML = `
      <strong>${row.annotator}</strong>
      <small class="muted">提交 ${row.submitted_tasks}/${row.total_tasks}</small>
      <small class="muted">数据集: ${row.dataset_path || '-'}</small>
      <small class="muted">更新时间: ${row.updated_at || '-'}</small>
    `;
    div.onclick = () => $('annotatorInput').value = row.annotator;
    $('annotatorList').appendChild(div);
  });
}

async function loadDetail() {
  const annotator = $('annotatorInput').value.trim();
  const taskId = $('taskInput').value;
  if (!annotator || !taskId) return alert('请填写标注者与任务ID');
  const data = await api(`/api/review/task?annotator=${encodeURIComponent(annotator)}&task_id=${taskId}`);
  $('detail').innerHTML = `
    <h4>题目</h4><pre>${data.task.question || ''}</pre>
    <h4>标准答案</h4><pre>${data.task.standard_answer || ''}</pre>
    <h4>样本审核</h4><pre>${JSON.stringify(data.state.sample_reviews, null, 2)}</pre>
    <h4>Step 切分</h4><pre>${JSON.stringify(data.state.step_segments, null, 2)}</pre>
    <h4>Claim 切分</h4><pre>${JSON.stringify(data.state.claims, null, 2)}</pre>
    <h4>依赖关系</h4><pre>${JSON.stringify(data.state.dependencies, null, 2)}</pre>
    <h4>状态</h4><pre>${JSON.stringify({status: data.state.status, saved_at: data.state.saved_at, submitted_at: data.state.submitted_at}, null, 2)}</pre>
  `;
}

$('refreshBtn').onclick = refreshSummary;
$('loadDetailBtn').onclick = loadDetail;
refreshSummary();
