const state = {
  annotator: "",
  datasetPath: "",
  taskId: null,
  task: null,
  local: {
    sample_reviews: [],
    method_categories: ["与已知解同方法", "等价变体"],
    selected_primary_sample: null,
    step_segments: [],
    claims: [],
    dependencies: {}
  }
};

const $ = (id) => document.getElementById(id);
const tabs = [...document.querySelectorAll('.tab')];

function switchStep(idx) {
  tabs.forEach(t => t.classList.toggle('active', Number(t.dataset.step) === idx));
  for (let i = 0; i <= 4; i++) {
    $("stage" + i).classList.toggle('hidden', i !== idx);
  }
}

tabs.forEach(t => t.addEventListener('click', () => switchStep(Number(t.dataset.step))));

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'API Error');
  return data;
}

function renderTaskList(items = []) {
  const wrap = $('taskList');
  wrap.innerHTML = '';
  items.forEach(item => {
    const btn = document.createElement('button');
    btn.textContent = `题目 ${item.task_id} · ${item.status}`;
    btn.onclick = () => loadTask(item.task_id);
    wrap.appendChild(btn);
  });
}

function renderMeta() {
  if (!state.task) return;
  $('taskMeta').innerHTML = `
    <h3>题目 ${state.task.task_id}</h3>
    <label>题目内容<pre>${state.task.question || ''}</pre></label>
    <label>标准答案<pre>${state.task.standard_answer || ''}</pre></label>
  `;
}

function renderSamples() {
  const box = $('sampleContainer');
  box.innerHTML = '';
  const categories = state.local.method_categories;

  state.task.samples.forEach(sample => {
    const row = document.createElement('div');
    row.className = 'sample-card';
    const review = state.local.sample_reviews.find(r => r.sample_id === sample.sample_id) || { sample_id: sample.sample_id, is_correct: null, method: '', new_method_summary: '' };
    row.innerHTML = `
      <strong>样本 #${sample.sample_id}</strong>
      <pre>${sample.solution}</pre>
      <div class="flex">
        <button data-c="yes">正确</button>
        <button data-c="no" class="secondary">错误</button>
      </div>
      <label>解法分类
        <select>
          <option value="">-- 若判定正确请选择 --</option>
          ${categories.map(c => `<option ${review.method === c ? 'selected' : ''}>${c}</option>`).join('')}
          <option value="__new__">+ 新建分类</option>
        </select>
      </label>
      <label class="new-method ${review.method === '__new__' ? '' : 'hidden'}">新方法概述
        <textarea rows="3">${review.new_method_summary || ''}</textarea>
      </label>
    `;
    const [yesBtn, noBtn] = row.querySelectorAll('button[data-c]');
    yesBtn.onclick = () => { review.is_correct = true; upsertReview(review); renderSamples(); };
    noBtn.onclick = () => { review.is_correct = false; review.method = ''; review.new_method_summary = ''; upsertReview(review); renderSamples(); };

    const sel = row.querySelector('select');
    sel.onchange = () => {
      if (sel.value === '__new__') {
        const newName = prompt('请输入新分类名称');
        if (newName) {
          state.local.method_categories.push(newName.trim());
          review.method = newName.trim();
        }
      } else {
        review.method = sel.value;
      }
      upsertReview(review);
      renderSamples();
    };

    const ta = row.querySelector('textarea');
    ta.oninput = () => { review.new_method_summary = ta.value; upsertReview(review); };
    box.appendChild(row);
  });

  $('primarySampleId').value = state.local.selected_primary_sample || '';
}

function upsertReview(review) {
  const idx = state.local.sample_reviews.findIndex(r => r.sample_id === review.sample_id);
  if (idx >= 0) state.local.sample_reviews[idx] = review;
  else state.local.sample_reviews.push(review);
}

function renderSteps() {
  const box = $('stepEditor');
  box.innerHTML = '';
  state.local.step_segments.forEach((step, idx) => {
    const div = document.createElement('div');
    div.className = 'claim-row';
    div.innerHTML = `<label>Step ${idx + 1}<textarea rows="3">${step}</textarea></label><button class="secondary">删除</button>`;
    div.querySelector('textarea').oninput = (e) => state.local.step_segments[idx] = e.target.value;
    div.querySelector('button').onclick = () => { state.local.step_segments.splice(idx, 1); renderSteps(); };
    box.appendChild(div);
  });
}

function renderClaims() {
  const box = $('claimEditor');
  box.innerHTML = '';
  state.local.claims.forEach((c, idx) => {
    const div = document.createElement('div');
    div.className = 'claim-row';
    div.innerHTML = `<small>Claim ${c.claim_id} (Step ${c.step_index + 1})</small><textarea rows="2">${c.text}</textarea><button class="secondary">删除</button>`;
    div.querySelector('textarea').oninput = (e) => state.local.claims[idx].text = e.target.value;
    div.querySelector('button').onclick = () => { state.local.claims.splice(idx, 1); renderClaims(); renderDependencies(); };
    box.appendChild(div);
  });
  const add = document.createElement('button');
  add.textContent = '新增 claim';
  add.className = 'ghost';
  add.onclick = () => {
    state.local.claims.push({ claim_id: `custom-${Date.now()}`, step_index: 0, text: '' });
    renderClaims();
  };
  box.appendChild(add);
}

function renderDependencies() {
  const box = $('dependencyEditor');
  box.innerHTML = '';
  state.local.claims.forEach((claim, idx) => {
    const div = document.createElement('div');
    div.className = 'claim-row';
    const allowed = state.local.claims.slice(0, idx);
    const selected = new Set(state.local.dependencies[claim.claim_id] || []);
    div.innerHTML = `<strong>${claim.claim_id}</strong><pre>${claim.text}</pre>`;
    const list = document.createElement('div');
    allowed.forEach(prev => {
      const id = `${claim.claim_id}_${prev.claim_id}`;
      const checked = selected.has(prev.claim_id) ? 'checked' : '';
      const row = document.createElement('label');
      row.innerHTML = `<input type="checkbox" id="${id}" ${checked}/> 依赖 ${prev.claim_id}`;
      row.querySelector('input').onchange = (e) => {
        if (!state.local.dependencies[claim.claim_id]) state.local.dependencies[claim.claim_id] = [];
        if (e.target.checked) state.local.dependencies[claim.claim_id].push(prev.claim_id);
        else state.local.dependencies[claim.claim_id] = state.local.dependencies[claim.claim_id].filter(x => x !== prev.claim_id);
      };
      list.appendChild(row);
    });
    if (allowed.length === 0) list.innerHTML = '<small class="muted">无可依赖前序 claim</small>';
    div.appendChild(list);
    box.appendChild(div);
  });
}

async function loadTask(taskId) {
  const data = await api(`/api/task?annotator=${encodeURIComponent(state.annotator)}&task_id=${taskId}`);
  state.taskId = taskId;
  state.task = data.task;
  state.local = { ...state.local, ...data.state };
  renderMeta();
  renderSamples();
  renderSteps();
  renderClaims();
  renderDependencies();
}

$('initBtn').onclick = async () => {
  try {
    state.annotator = $('annotatorInput').value.trim();
    state.datasetPath = $('datasetInput').value.trim();
    await api('/api/session/init', { method: 'POST', body: JSON.stringify({ annotator: state.annotator, dataset_path: state.datasetPath }) });
    const session = await api(`/api/session?annotator=${encodeURIComponent(state.annotator)}`);
    renderTaskList(session.tasks);
  } catch (e) { alert(e.message); }
};

$('manualSaveBtn').onclick = async () => {
  if (!state.taskId) return alert('请先选择任务');
  state.local.selected_primary_sample = Number($('primarySampleId').value) || null;
  try {
    await api('/api/task/save', { method: 'POST', body: JSON.stringify({ annotator: state.annotator, task_id: state.taskId, ...state.local }) });
    alert('已手动保存。');
  } catch (e) { alert(e.message); }
};

$('autoSplitBtn').onclick = () => {
  const primary = state.task.samples.find(s => s.sample_id === Number($('primarySampleId').value)) || state.task.samples[0];
  const text = primary?.solution || '';
  state.local.step_segments = text.split(/\n+/).map(x => x.trim()).filter(Boolean);
  renderSteps();
};

$('addStepBtn').onclick = () => { state.local.step_segments.push(''); renderSteps(); };

$('genClaimsBtn').onclick = async () => {
  try {
    const res = await api('/api/task/generate_claims', {
      method: 'POST',
      body: JSON.stringify({ annotator: state.annotator, task_id: state.taskId, step_segments: state.local.step_segments })
    });
    state.local.claims = res.claims;
    renderClaims();
    renderDependencies();
  } catch (e) { alert(e.message); }
};

$('submitBtn').onclick = async () => {
  state.local.selected_primary_sample = Number($('primarySampleId').value) || null;
  try {
    const res = await api('/api/task/submit', { method: 'POST', body: JSON.stringify({ annotator: state.annotator, task_id: state.taskId, ...state.local }) });
    $('submitResult').textContent = `提交成功：${res.saved_at}`;
  } catch (e) { $('submitResult').textContent = e.message; }
};

$('openGuidelineBtn').onclick = async () => {
  const res = await api('/api/guideline');
  $('guidelineContent').textContent = res.content;
  $('guidelineDialog').showModal();
};
