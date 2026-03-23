let dataset = [];
let currentCaseIndex = -1;
let currentStep = 1;
const stateByCase = {};

function getCaseState(caseId) {
  if (!stateByCase[caseId]) {
    stateByCase[caseId] = {
      sample_validation: [],
      selected_solution_idx: 0,
      selected_solution_text: '',
      cut_points: [],
      steps: [],
      claims: [],
      dependencies: {},
      claim_generation_source: '',
    };
  }
  return stateByCase[caseId];
}

function selectedCase() { return dataset[currentCaseIndex]; }

function toggleCasePanel() {
  document.getElementById('casePanel').classList.toggle('collapsed');
}

async function loadDataset() {
  const path = document.getElementById('jsonlPath').value.trim();
  const res = await fetch('/api/load_jsonl', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.error || '加载失败');
  dataset = data.items;
  renderCaseList();
  if (dataset.length) selectCase(0);
}

function renderCaseList() {
  const ul = document.getElementById('caseList');
  ul.innerHTML = '';
  dataset.forEach((c, i) => {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.textContent = c.id;
    btn.onclick = () => selectCase(i);
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

function selectCase(idx) {
  currentCaseIndex = idx;
  currentStep = 1;
  const c = selectedCase();
  const st = getCaseState(c.id);
  if (!st.selected_solution_text && (c.samples || []).length) {
    st.selected_solution_idx = 0;
    st.selected_solution_text = c.samples[0].solution || '';
  }
  renderCurrentCase();
}

function goStep(s) {
  currentStep = s;
  renderStepContent();
}

function renderCurrentCase() {
  const c = selectedCase();
  if (!c) return;
  document.getElementById('caseTitle').textContent = `当前问题：${c.id}`;
  document.getElementById('qAndA').textContent = `题目:\n${c.question}\n\n标准答案:\n${c.reference_answer}`;
  document.getElementById('known').textContent = JSON.stringify(c.known_solutions || [], null, 2);
  renderStepContent();
}

function sampleRecord(st, i) {
  st.sample_validation[i] = st.sample_validation[i] || { is_correct: null, class_name: '', is_new_class: false, summary: '', translation: '' };
  return st.sample_validation[i];
}

function chooseSampleStatus(i, status) {
  const st = getCaseState(selectedCase().id);
  const rec = sampleRecord(st, i);
  rec.is_correct = rec.is_correct === status ? null : status;
  renderStepContent();
}

function setSampleField(i, k, v) {
  const st = getCaseState(selectedCase().id);
  sampleRecord(st, i)[k] = v;
}

async function translateSample(i) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const text = ((c.samples || [])[i] || {}).solution || '';
  const res = await fetch('/api/translate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, target: 'zh-CN' }),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.error || '翻译失败');
  sampleRecord(st, i).translation = data.translated;
  renderStepContent();
}

function selectSolution(i) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  st.selected_solution_idx = i;
  st.selected_solution_text = (c.samples[i] || {}).solution || '';
  st.cut_points = [];
  st.steps = [];
  renderStepContent();
}

function addCutPoint() {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const ta = document.getElementById('solutionText');
  if (!ta) return;
  const pos = ta.selectionStart;
  if (pos > 0 && pos < (st.selected_solution_text || '').length && !st.cut_points.includes(pos)) {
    st.cut_points.push(pos);
    st.cut_points.sort((a, b) => a - b);
    updateSplitPreview();
  }
}

function removeCutPoint(p) {
  const st = getCaseState(selectedCase().id);
  st.cut_points = st.cut_points.filter(x => x !== p);
  updateSplitPreview();
}

async function updateSplitPreview() {
  const st = getCaseState(selectedCase().id);
  const res = await fetch('/api/split_steps', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ solution: st.selected_solution_text, cut_points: st.cut_points }),
  });
  const data = await res.json();
  st.steps = (data.steps || []).map((text, i) => ({ id: `s${i + 1}`, text }));
  const box = document.getElementById('splitPreview');
  if (box) box.textContent = JSON.stringify(st.steps, null, 2);
  const cp = document.getElementById('cutPointList');
  if (cp) {
    cp.innerHTML = st.cut_points.map(x => `<button onclick="removeCutPoint(${x})">位置 ${x} ×</button>`).join(' ');
  }
}

async function generateClaims() {
  const st = getCaseState(selectedCase().id);
  const model = document.getElementById('claimModel').value.trim() || 'gpt-4o-mini';
  const temperatureRaw = document.getElementById('claimTemp').value.trim();
  const maxTokensRaw = document.getElementById('claimMaxTokens').value.trim();
  const temperature = temperatureRaw === '' ? 0 : Number(temperatureRaw);
  const max_tokens = maxTokensRaw === '' ? 1500 : Number(maxTokensRaw);
  const res = await fetch('/api/generate_claims', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steps: st.steps || [], model, temperature, max_tokens, allow_fallback: true }),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.error || 'claim 生成失败');

  st.claim_generation_source = data.source ? `${data.source}${data.model ? `:${data.model}` : ''}` : 'unknown';
  st.claims = (data.claims_by_step || []).map((x, i) => ({
    step_id: x.step_id || `s${i + 1}`,
    claims: (x.claims || []).map(t => (t || '').trim()).filter(Boolean),
  }));
  renderStepContent();
}

function editClaim(stepIdx, claimIdx, v) {
  getCaseState(selectedCase().id).claims[stepIdx].claims[claimIdx] = v;
}

function addClaim(stepIdx) {
  getCaseState(selectedCase().id).claims[stepIdx].claims.push('');
  renderStepContent();
}

function flattenClaimsByStep(claims) {
  return claims.map((s, si) => ({
    stepIdx: si,
    stepId: s.step_id || `s${si + 1}`,
    claims: (s.claims || []).map((text, ci) => ({ id: `s${si + 1}c${ci + 1}`, text, stepIdx: si, claimIdx: ci })),
  }));
}

function toggleDep(currId, depId, checked) {
  const st = getCaseState(selectedCase().id);
  st.dependencies[currId] = st.dependencies[currId] || [];
  if (checked) {
    if (!st.dependencies[currId].includes(depId)) st.dependencies[currId].push(depId);
  } else {
    st.dependencies[currId] = st.dependencies[currId].filter(x => x !== depId);
  }
}

function buildDependencyView() {
  const st = getCaseState(selectedCase().id);
  const grouped = flattenClaimsByStep(st.claims);
  let html = '<h3>Step 5：依赖关系（按当前 Step 逐条标注）</h3>';

  grouped.forEach((currStep, sIdx) => {
    html += `<section class="dep-section"><h4>当前 Step ${sIdx + 1}</h4>`;
    currStep.claims.forEach(curr => {
      html += `<div class="dep-card"><div class="curr-claim"><b>${curr.id}</b> ${curr.text}</div>`;
      html += '<div class="prev-steps">';
      for (let ps = sIdx; ps >= 0; ps--) {
        const prev = grouped[ps];
        const candidates = prev.claims.filter(c => (c.stepIdx < curr.stepIdx) || (c.stepIdx === curr.stepIdx && c.claimIdx < curr.claimIdx));
        if (!candidates.length) continue;
        html += `<details><summary>前序 Step ${ps + 1}（${candidates.length}条）</summary>`;
        candidates.forEach(cand => {
          const deps = st.dependencies[curr.id] || [];
          const checked = deps.includes(cand.id) ? 'checked' : '';
          html += `<label class="dep-option"><input type="checkbox" ${checked} onchange="toggleDep('${curr.id}','${cand.id}',this.checked)"> <span>${cand.id}</span> ${cand.text}</label>`;
        });
        html += '</details>';
      }
      html += '</div></div>';
    });
    html += '</section>';
  });
  return html;
}

function buildSummaryView() {
  const c = selectedCase();
  const st = getCaseState(c.id);
  return `
    <h3>Step 6：提交前总览</h3>
    <p>请检查以下结果无误后提交：</p>
    <h4>多采样验证</h4><pre>${JSON.stringify(st.sample_validation, null, 2)}</pre>
    <h4>Step切分（来自完整 solution 的切分点）</h4><pre>${JSON.stringify({ solution_index: st.selected_solution_idx, cut_points: st.cut_points, steps: st.steps }, null, 2)}</pre>
    <h4>Claim切分（来源：${st.claim_generation_source || '未生成'}）</h4><pre>${JSON.stringify(st.claims, null, 2)}</pre>
    <h4>依赖关系</h4><pre>${JSON.stringify(st.dependencies, null, 2)}</pre>
    <button class="primary" onclick="submitCase()">确认提交当前问题</button>
  `;
}

function renderStepContent() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const root = document.getElementById('stepContent');

  if (currentStep === 1) {
    let html = '<h3>Step 1：多采样验证（可回退）</h3><p>点击“正确/错误”可切换，再次点击可撤销为未判定。</p>';
    (c.samples || []).forEach((s, i) => {
      const rec = sampleRecord(st, i);
      const clsCorrect = rec.is_correct === true ? 'tag active ok' : 'tag';
      const clsWrong = rec.is_correct === false ? 'tag active bad' : 'tag';
      const clsUnset = rec.is_correct === null ? 'tag active' : 'tag';
      html += `
      <div class="card">
        <div class="card-head">
          <h4>sample-${i + 1}</h4>
          <button onclick="translateSample(${i})">翻译</button>
          <button onclick="selectSolution(${i})">设为Step切分对象</button>
        </div>
        <pre>${s.solution || ''}</pre>
        ${rec.translation ? `<details open><summary>翻译结果</summary><pre>${rec.translation}</pre></details>` : ''}
        <div class="row">
          <button class="${clsCorrect}" onclick="chooseSampleStatus(${i}, true)">正确</button>
          <button class="${clsWrong}" onclick="chooseSampleStatus(${i}, false)">错误</button>
          <button class="${clsUnset}" onclick="chooseSampleStatus(${i}, null)">未判定</button>
        </div>
        <div class="row">
          <label>分类 <input value="${rec.class_name || ''}" oninput="setSampleField(${i}, 'class_name', this.value)"></label>
          <label><input type="checkbox" ${rec.is_new_class ? 'checked' : ''} onchange="setSampleField(${i}, 'is_new_class', this.checked)"> 新分类</label>
          <label>新方法概述 <input value="${rec.summary || ''}" oninput="setSampleField(${i}, 'summary', this.value)"></label>
        </div>
      </div>`;
    });
    root.innerHTML = html;
    return;
  }

  if (currentStep === 2) {
    root.innerHTML = `
      <h3>Step 2：Step切分（在完整 solution 上打点）</h3>
      <p>当前切分对象：sample-${(st.selected_solution_idx || 0) + 1}。在下方文本中将光标移动到切分位置后点击“添加切分点”。</p>
      <textarea id="solutionText" class="full-solution">${st.selected_solution_text || ''}</textarea>
      <div class="row">
        <button onclick="addCutPoint()">添加切分点</button>
        <button onclick="updateSplitPreview()">刷新预览</button>
      </div>
      <div id="cutPointList" class="row"></div>
      <h4>切分结果（可回退：删除切分点后刷新）</h4>
      <pre id="splitPreview">${JSON.stringify(st.steps, null, 2)}</pre>
    `;
    return;
  }

  if (currentStep === 3) {
    root.innerHTML = `
      <h3>Step 3：Claim切分（服务端 OpenAI SDK）</h3>
      <p>逻辑说明：后端使用 OpenAI 官方 SDK 调用你配置的模型生成 claim；失败时可按开关退化到本地分句。</p>
      <button onclick="generateClaims()">调用 Claim API</button>
      <p>当前来源：${st.claim_generation_source || '未生成'}</p>
      <pre>${JSON.stringify(st.claims, null, 2)}</pre>
    `;
    return;
  }

  if (currentStep === 4) {
    let html = '<h3>Step 4：Claim校验与修正</h3>';
    (st.claims || []).forEach((cs, si) => {
      html += `<h4>Step ${si + 1}</h4>`;
      (cs.claims || []).forEach((claim, ci) => {
        html += `<div><input class="claim-input" value="${claim}" oninput="editClaim(${si}, ${ci}, this.value)"></div>`;
      });
      html += `<button onclick="addClaim(${si})">+ 添加 claim</button>`;
    });
    root.innerHTML = html;
    return;
  }

  if (currentStep === 5) {
    root.innerHTML = buildDependencyView();
    return;
  }

  if (currentStep === 6) {
    root.innerHTML = buildSummaryView();
  }
}

async function saveProgress() {
  const c = selectedCase(); if (!c) return alert('先选择问题');
  const annotator = document.getElementById('annotator').value.trim() || 'unknown';
  const st = getCaseState(c.id);
  const payload = {
    annotator,
    case_id: c.id,
    sample_validation: st.sample_validation,
    selected_solution_idx: st.selected_solution_idx,
    selected_solution_text: st.selected_solution_text,
    cut_points: st.cut_points,
    steps: st.steps,
    claims: st.claims,
    dependencies: st.dependencies,
    status: 'in_progress',
  };
  const res = await fetch('/api/save_record', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.error || '保存失败');
  alert(`已保存: ${data.path}`);
}

async function submitCase() {
  const c = selectedCase(); if (!c) return alert('先选择问题');
  const annotator = document.getElementById('annotator').value.trim() || 'unknown';
  const st = getCaseState(c.id);
  const payload = {
    annotator,
    case_id: c.id,
    sample_validation: st.sample_validation,
    selected_solution_idx: st.selected_solution_idx,
    selected_solution_text: st.selected_solution_text,
    cut_points: st.cut_points,
    steps: st.steps,
    claims: st.claims,
    dependencies: st.dependencies,
    status: 'completed',
  };
  const res = await fetch('/api/save_record', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) return alert(data.error || '提交失败');
  alert(`题目 ${c.id} 标注完成并自动保存`);
}

async function openGuide() {
  const res = await fetch('/api/guideline');
  const data = await res.json();
  document.getElementById('guideText').textContent = data.content || '暂无';
}
