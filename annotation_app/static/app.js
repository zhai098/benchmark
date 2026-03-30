let dataset = [];
let currentCaseIndex = -1;
let currentStep = 1;
const stateByCase = {};
let deviceId = '';
let autosaveTimer = null;
let saveBadgeTimer = null;

function initDeviceId() {
  const key = 'annotation_device_id';
  const existing = localStorage.getItem(key);
  if (existing) {
    deviceId = existing;
    return;
  }
  deviceId = `dev-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  localStorage.setItem(key, deviceId);
}

function annotatorId() {
  return document.getElementById('annotator').value.trim() || 'unknown';
}

function getSaveBadge() {
  return document.getElementById('saveState');
}

function setSaveState(text, cls = '') {
  const badge = getSaveBadge();
  if (!badge) return;
  badge.textContent = text;
  badge.className = `save-state ${cls}`.trim();
}

function getDefaultWorkingAnnotation(sample = {}) {
  return {
    selected_solution_text: sample.solution || '',
    cut_points: [],
    steps: [],
    presegmented_claims: extractPresegmentedClaims(sample),
    claims: [],
    claim_checks: {},
    dependencies: {},
    workflow_state: 'sample_selected',
  };
}

function getCaseState(caseId) {
  if (!stateByCase[caseId]) {
    stateByCase[caseId] = {
      current_step: 1,
      active_sample_idx: null,
      sample_validation: [],
      sample_annotations: {},
      correct_solutions: [],
    };
  }
  return stateByCase[caseId];
}

function getWorkingAnnotation(st) {
  if (st.active_sample_idx === null || st.active_sample_idx === undefined) {
    return getDefaultWorkingAnnotation();
  }
  if (!st.sample_annotations[st.active_sample_idx]) {
    const sample = (selectedCase()?.samples || [])[st.active_sample_idx] || {};
    st.sample_annotations[st.active_sample_idx] = getDefaultWorkingAnnotation(sample);
  }
  return st.sample_annotations[st.active_sample_idx];
}

function selectedCase() { return dataset[currentCaseIndex]; }
function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function renderLatexWithFallback(rawText) {
  const htmlSafe = escapeHtml(rawText || '');
  if (!window.renderMathInElement || !window.katex) {
    return `<pre>${htmlSafe}</pre>`;
  }
  const temp = document.createElement('div');
  temp.innerHTML = `<div class="math-content">${htmlSafe}</div>`;
  try {
    renderMathInElement(temp, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
      strict: 'ignore',
    });
    return temp.innerHTML;
  } catch (e) {
    return `<pre>${htmlSafe}</pre>`;
  }
}

function renderSolutionCard(solution, sampleIdx = null) {
  const copyArg = sampleIdx === null ? 'null' : sampleIdx;
  return `
    <div class="solution-render-card" data-raw-solution="${escapeHtml(solution || '')}">
      <div class="row card-actions">
        <button onclick="copySolutionRaw(${copyArg})">复制原始解答</button>
        <span id="copyStatus_${sampleIdx === null ? 'active' : sampleIdx}" class="copy-status"></span>
      </div>
      <div class="rendered-math">${renderLatexWithFallback(solution || '')}</div>
    </div>
  `;
}

function getClaimCheckStats(st) {
  const wa = getWorkingAnnotation(st);
  const total = (wa.claims || []).reduce((acc, x) => acc + (x.claims || []).length, 0);
  const checks = wa.claim_checks || {};
  let correct = 0;
  let incorrect = 0;
  Object.values(checks).forEach(v => {
    if (v === 'correct') correct += 1;
    else if (v === 'incorrect') incorrect += 1;
  });
  return { total, correct, incorrect, checked: correct + incorrect, unchecked: Math.max(0, total - correct - incorrect) };
}

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

async function selectCase(idx) {
  if (idx === currentCaseIndex) return;
  await flushAutosave();
  currentCaseIndex = idx;
  currentStep = 1;
  const c = selectedCase();
  const st = getCaseState(c.id);
  st.current_step = currentStep;
  await restoreProgress(c.id);
  if (st.active_sample_idx === null && (c.samples || []).length) {
    st.active_sample_idx = 0;
    if (!st.sample_annotations[0]) {
      st.sample_annotations[0] = getDefaultWorkingAnnotation(c.samples[0]);
    }
  }
  renderCurrentCase();
}

function goStep(s) {
  currentStep = s;
  const c = selectedCase();
  if (c) {
    getCaseState(c.id).current_step = s;
    scheduleAutosave();
  }
  renderStepContent();
}

function renderCurrentCase() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const stats = getClaimCheckStats(st);
  const activeSample = st.active_sample_idx === null ? '-' : `sample-${st.active_sample_idx + 1}`;
  document.getElementById('caseTitle').innerHTML = `当前问题：${escapeHtml(c.id)} <span class="pill">samples ${(c.samples || []).length}</span> <span class="pill">active ${activeSample}</span> <span class="pill">steps ${(wa.steps || []).length}</span> <span class="pill">claims ${stats.total}</span>`;
  document.getElementById('qAndA').textContent = `题目:\n${c.question}\n\n标准答案:\n${c.reference_answer}`;
  document.getElementById('known').textContent = JSON.stringify([...(c.known_solutions || []), ...(st.correct_solutions || [])], null, 2);
  renderStepContent();
}

function sampleRecord(st, i) {
  st.sample_validation[i] = st.sample_validation[i] || { is_correct: null, class_name: '', is_new_class: false, summary: '', translation: '' };
  return st.sample_validation[i];
}

function chooseSampleStatus(i, status) {
  const st = getCaseState(selectedCase().id);
  const rec = sampleRecord(st, i);
  rec.is_correct = status;

  if (status === false) {
    delete st.sample_annotations[i];
    st.correct_solutions = (st.correct_solutions || []).filter(x => x.sample_idx !== i);
    if (st.active_sample_idx === i) {
      st.active_sample_idx = null;
    }
  }

  if (status === true && !st.sample_annotations[i]) {
    st.sample_annotations[i] = getDefaultWorkingAnnotation((selectedCase().samples || [])[i] || {});
  }

  scheduleAutosave();
  renderStepContent();
}

function setSampleField(i, k, v) {
  const st = getCaseState(selectedCase().id);
  sampleRecord(st, i)[k] = v;
  scheduleAutosave();
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
  scheduleAutosave();
  renderStepContent();
}

function selectSolution(i) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const rec = sampleRecord(st, i);
  if (rec.is_correct !== true) {
    alert('仅可选择“判定为正确”的 sample 进入完整工作流');
    return;
  }
  st.active_sample_idx = i;
  if (!st.sample_annotations[i]) {
    st.sample_annotations[i] = getDefaultWorkingAnnotation((c.samples || [])[i] || {});
  }
  scheduleAutosave();
  renderStepContent();
}

function extractPresegmentedClaims(sample) {
  const raw = sample?.claims_by_step || sample?.step_claims || sample?.claims || [];
  if (!Array.isArray(raw)) return [];
  const out = [];
  raw.forEach((item, i) => {
    if (typeof item === 'string') {
      const text = item.trim();
      if (text) out.push({ id: `p${i + 1}`, text, step_idx: null });
      return;
    }
    if (item && typeof item === 'object' && Array.isArray(item.claims)) {
      const step_idx = Number.isInteger(item.step_index)
        ? item.step_index
        : parseInt(String(item.step_id || '').replace(/[^\d]/g, ''), 10) - 1;
      (item.claims || []).forEach((c, ci) => {
        const text = String(c || '').trim();
        if (text) out.push({ id: `p${i + 1}_${ci + 1}`, text, step_idx: Number.isFinite(step_idx) ? step_idx : null });
      });
      return;
    }
    const text = String(item.text || item.claim || '').trim();
    if (!text) return;
    const step_idx = Number.isInteger(item.step_index)
      ? item.step_index
      : parseInt(String(item.step_id || '').replace(/[^\d]/g, ''), 10) - 1;
    out.push({ id: `p${i + 1}`, text, step_idx: Number.isFinite(step_idx) ? step_idx : null });
  });
  return out;
}

function addCutPoint() {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const ta = document.getElementById('solutionText');
  if (!ta) return;
  const pos = ta.selectionStart;
  if (pos > 0 && pos < (wa.selected_solution_text || '').length && !wa.cut_points.includes(pos)) {
    wa.cut_points.push(pos);
    wa.cut_points.sort((a, b) => a - b);
    scheduleAutosave();
    updateSplitPreview();
  }
}

function removeCutPoint(p) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.cut_points = wa.cut_points.filter(x => x !== p);
  scheduleAutosave();
  updateSplitPreview();
}

async function updateSplitPreview() {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  const res = await fetch('/api/split_steps', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ solution: wa.selected_solution_text, cut_points: wa.cut_points }),
  });
  const data = await res.json();
  wa.steps = (data.steps || []).map((text, i) => ({ id: `s${i + 1}`, text }));
  wa.workflow_state = 'steps_segmented';
  const box = document.getElementById('splitPreview');
  if (box) box.textContent = JSON.stringify(wa.steps, null, 2);
  const cp = document.getElementById('cutPointList');
  if (cp) {
    cp.innerHTML = wa.cut_points.map(x => `<button onclick="removeCutPoint(${x})">位置 ${x} ×</button>`).join(' ');
  }
  scheduleAutosave();
}

function organizeClaimsBySteps() {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  const stepCount = (wa.steps || []).length;
  wa.claims = Array.from({ length: stepCount }, (_, i) => ({ step_id: `s${i + 1}`, claims: [] }));
  (wa.presegmented_claims || []).forEach((claim, i) => {
    let stepIdx = Number(document.getElementById(`claimStepSel_${i}`)?.value ?? -1);
    if (!Number.isInteger(stepIdx) || stepIdx < 0 || stepIdx >= stepCount) return;
    claim.step_idx = stepIdx;
    wa.claims[stepIdx].claims.push((claim.text || '').trim());
  });
  wa.workflow_state = 'claims_assigned';
  scheduleAutosave();
  renderStepContent();
}

function updateClaimCheck(claimId, status) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.claim_checks[claimId] = status;
  wa.workflow_state = 'claims_checked';
  scheduleAutosave();
}

function claimCheckTag(claimId, current, expected, label) {
  return `<button class="${current === expected ? 'tag active' : 'tag'}" onclick="updateClaimCheckAndRender('${expected}', '${claimId}')">${label}</button>`;
}

function updateClaimCheckAndRender(status, claimId) {
  updateClaimCheck(claimId, status);
  renderStepContent();
}

function editClaim(stepIdx, claimIdx, v) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.claims[stepIdx].claims[claimIdx] = v;
  scheduleAutosave();
}

function addClaim(stepIdx) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.claims[stepIdx].claims.push('');
  scheduleAutosave();
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
  const wa = getWorkingAnnotation(st);
  wa.dependencies[currId] = wa.dependencies[currId] || [];
  if (checked) {
    if (!wa.dependencies[currId].includes(depId)) wa.dependencies[currId].push(depId);
  } else {
    wa.dependencies[currId] = wa.dependencies[currId].filter(x => x !== depId);
  }
  wa.workflow_state = 'dependencies_labeled';
  scheduleAutosave();
}

function buildDependencyView() {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  const grouped = flattenClaimsByStep(wa.claims);
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
          const deps = wa.dependencies[curr.id] || [];
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
  const wa = getWorkingAnnotation(st);
  const stats = getClaimCheckStats(st);
  return `
    <h3>Step 6：提交前总览</h3>
    <p>请检查以下结果无误后提交：</p>
    <div class="kpi-grid">
      <div class="kpi"><small>Step 数</small><b>${(wa.steps || []).length}</b></div>
      <div class="kpi"><small>Claim 总数</small><b>${stats.total}</b></div>
      <div class="kpi"><small>已检查</small><b>${stats.checked}</b></div>
      <div class="kpi"><small>未检查</small><b>${stats.unchecked}</b></div>
    </div>
    <h4>多采样验证</h4><pre>${JSON.stringify(st.sample_validation, null, 2)}</pre>
    <h4>当前工作流状态</h4><pre>${JSON.stringify(wa.workflow_state || '', null, 2)}</pre>
    <h4>Step切分（当前样本）</h4><pre>${JSON.stringify({ active_sample_idx: st.active_sample_idx, cut_points: wa.cut_points, steps: wa.steps }, null, 2)}</pre>
    <h4>Claim整理结果（按 step）</h4><pre>${JSON.stringify(wa.claims, null, 2)}</pre>
    <h4>Claim正确性检查</h4><pre>${JSON.stringify(wa.claim_checks, null, 2)}</pre>
    <h4>依赖关系</h4><pre>${JSON.stringify(wa.dependencies, null, 2)}</pre>
    <h4>已完成正确解参考</h4><pre>${JSON.stringify(st.correct_solutions, null, 2)}</pre>
    <button class="primary" onclick="submitCase()">确认提交当前问题</button>
  `;
}

function renderStepContent() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const root = document.getElementById('stepContent');
  const navButtons = document.querySelectorAll('.step-btn');
  navButtons.forEach((btn) => btn.classList.toggle('active', Number(btn.dataset.step) === currentStep));

  if (currentStep === 1) {
    let html = '<h3>Step 1：多采样验证（可回退）</h3><p>点击“正确/错误”后将触发自动保存；错误样本会立即从当前工作流中丢弃。</p>';
    (c.samples || []).forEach((s, i) => {
      const rec = sampleRecord(st, i);
      const clsCorrect = rec.is_correct === true ? 'tag active ok' : 'tag';
      const clsWrong = rec.is_correct === false ? 'tag active bad' : 'tag';
      const clsUnset = rec.is_correct === null ? 'tag active' : 'tag';
      html += `
      <div class="card">
        <div class="card-head">
          <h4>sample-${i + 1}</h4>
          <div class="row">
            <button onclick="translateSample(${i})">翻译</button>
            <button onclick="selectSolution(${i})">设为当前工作样本</button>
          </div>
        </div>
        ${renderSolutionCard(s.solution || '', i)}
        ${rec.translation ? `<details open><summary>翻译结果</summary><pre>${escapeHtml(rec.translation)}</pre></details>` : ''}
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

  if (st.active_sample_idx === null) {
    root.innerHTML = '<div class="card"><h3>请先在 Step 1 中选择一个判定为“正确”的 sample 作为当前工作样本。</h3></div>';
    return;
  }

  if (currentStep === 2) {
    root.innerHTML = `
      <h3>Step 2：Step切分（在完整 solution 上打点）</h3>
      <p>当前切分对象：sample-${st.active_sample_idx + 1}</p>
      ${renderSolutionCard(wa.selected_solution_text || '', null)}
      <textarea id="solutionText" class="full-solution" oninput="updateWorkingSolution(this.value)">${escapeHtml(wa.selected_solution_text || '')}</textarea>
      <div class="row">
        <button onclick="addCutPoint()">添加切分点</button>
        <button onclick="updateSplitPreview()">刷新预览</button>
      </div>
      <div id="cutPointList" class="row"></div>
      <h4>切分结果（可回退：删除切分点后刷新）</h4>
      <pre id="splitPreview">${JSON.stringify(wa.steps, null, 2)}</pre>
    `;
    return;
  }

  if (currentStep === 3) {
    const stepCount = (wa.steps || []).length;
    const claims = wa.presegmented_claims || [];
    const unassigned = claims.filter(x => !Number.isInteger(x.step_idx) || x.step_idx < 0).length;
    let rows = '';
    claims.forEach((cl, i) => {
      const defaultStepIdx = (Number.isInteger(cl.step_idx) && cl.step_idx >= 0 && cl.step_idx < stepCount) ? cl.step_idx : -1;
      const options = ['<option value="-1">未分配</option>']
        .concat(Array.from({ length: stepCount }, (_, si) => `<option value="${si}" ${defaultStepIdx === si ? 'selected' : ''}>Step ${si + 1}</option>`))
        .join('');
      rows += `
        <tr>
          <td>${cl.id}</td>
          <td>${escapeHtml(cl.text)}</td>
          <td><select id="claimStepSel_${i}">${options}</select></td>
        </tr>
      `;
    });
    root.innerHTML = `
      <h3>Step 3：整理每个 Step 对应的 Claim（使用预切分 claim）</h3>
      <div class="kpi-grid">
        <div class="kpi"><small>Step 数</small><b>${stepCount}</b></div>
        <div class="kpi"><small>预切分 Claim</small><b>${claims.length}</b></div>
        <div class="kpi"><small>未分配</small><b>${unassigned}</b></div>
      </div>
      <table>
        <thead><tr><th>Claim</th><th>文本</th><th>归属 Step</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="3">当前 solution 未提供预切分 claim</td></tr>'}</tbody>
      </table>
      <div class="row">
        <button class="primary" onclick="organizeClaimsBySteps()">保存并生成 Step-Claim 结构</button>
      </div>
      <pre>${JSON.stringify(wa.claims, null, 2)}</pre>
    `;
    return;
  }

  if (currentStep === 4) {
    const checkStats = getClaimCheckStats(st);
    let html = '<h3>Step 4：Claim正确性检查与修正</h3>';
    html += `
      <div class="kpi-grid">
        <div class="kpi"><small>Claim 总数</small><b>${checkStats.total}</b></div>
        <div class="kpi"><small>正确</small><b>${checkStats.correct}</b></div>
        <div class="kpi"><small>错误</small><b>${checkStats.incorrect}</b></div>
        <div class="kpi"><small>未检查</small><b>${checkStats.unchecked}</b></div>
      </div>
    `;
    (wa.claims || []).forEach((cs, si) => {
      html += `<h4>Step ${si + 1}</h4>`;
      (cs.claims || []).forEach((claim, ci) => {
        const claimId = `s${si + 1}c${ci + 1}`;
        const current = wa.claim_checks[claimId] || 'unchecked';
        html += `
          <div class="card">
            <div><input class="claim-input" value="${escapeHtml(claim)}" oninput="editClaim(${si}, ${ci}, this.value)"></div>
            <div class="row">
              ${claimCheckTag(claimId, current, 'correct', '正确')}
              ${claimCheckTag(claimId, current, 'incorrect', '错误')}
              ${claimCheckTag(claimId, current, 'unchecked', '未检查')}
            </div>
          </div>
        `;
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

function updateWorkingSolution(value) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.selected_solution_text = value;
  scheduleAutosave();
}

function buildProgressPayload(status = 'in_progress') {
  const c = selectedCase();
  if (!c) return null;
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  return {
    annotator_id: annotatorId(),
    device_id: deviceId,
    case_id: c.id,
    status,
    current_step: st.current_step || currentStep,
    current_workflow_state: {
      active_sample_idx: st.active_sample_idx,
      workflow_state: wa.workflow_state || 'sample_selected',
    },
    current_annotations: {
      selected_solution_text: wa.selected_solution_text,
      cut_points: wa.cut_points,
      steps: wa.steps,
      presegmented_claims: wa.presegmented_claims,
      claims: wa.claims,
      claim_checks: wa.claim_checks,
      dependencies: wa.dependencies,
      sample_annotations: st.sample_annotations,
    },
    sample_decisions: st.sample_validation,
    correct_solutions: st.correct_solutions || [],
  };
}

async function persistProgress(status = 'in_progress', silent = true) {
  const payload = buildProgressPayload(status);
  if (!payload) return;
  setSaveState('保存中…', 'saving');
  const res = await fetch('/api/save_progress', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    setSaveState('保存失败', 'error');
    if (!silent) alert(data.error || '保存失败');
    return;
  }
  setSaveState(`已保存 ${new Date().toLocaleTimeString()}`, 'saved');
}

function scheduleAutosave() {
  setSaveState('待保存', 'pending');
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    persistProgress('in_progress', true);
  }, 900);
}

async function flushAutosave() {
  clearTimeout(autosaveTimer);
  await persistProgress('in_progress', true);
}

async function restoreProgress(caseId) {
  if (!caseId) return;
  const st = getCaseState(caseId);
  const params = new URLSearchParams({
    annotator_id: annotatorId(),
    device_id: deviceId,
    case_id: caseId,
  });
  const res = await fetch(`/api/load_progress?${params.toString()}`);
  const data = await res.json();
  if (!res.ok || !data.found) {
    setSaveState('未发现历史进度', 'pending');
    return;
  }
  const progress = data.progress || {};
  st.current_step = progress.current_step || 1;
  currentStep = st.current_step;
  st.sample_validation = progress.sample_decisions || [];
  st.correct_solutions = progress.correct_solutions || [];
  st.active_sample_idx = progress.current_workflow_state?.active_sample_idx ?? null;
  const savedAnnotations = progress.current_annotations || {};
  st.sample_annotations = savedAnnotations.sample_annotations || st.sample_annotations || {};
  if (st.active_sample_idx !== null && !st.sample_annotations[st.active_sample_idx]) {
    st.sample_annotations[st.active_sample_idx] = {
      selected_solution_text: savedAnnotations.selected_solution_text || '',
      cut_points: savedAnnotations.cut_points || [],
      steps: savedAnnotations.steps || [],
      presegmented_claims: savedAnnotations.presegmented_claims || [],
      claims: savedAnnotations.claims || [],
      claim_checks: savedAnnotations.claim_checks || {},
      dependencies: savedAnnotations.dependencies || {},
      workflow_state: progress.current_workflow_state?.workflow_state || 'sample_selected',
    };
  }
  setSaveState('已恢复历史进度', 'saved');
}

async function saveProgress() {
  await flushAutosave();
  alert('已保存当前进度');
}

async function submitCase() {
  const c = selectedCase(); if (!c) return alert('先选择问题');
  const st = getCaseState(c.id);
  if (st.active_sample_idx !== null) {
    const sample = (c.samples || [])[st.active_sample_idx] || {};
    const wa = getWorkingAnnotation(st);
    const existing = (st.correct_solutions || []).some(x => x.sample_idx === st.active_sample_idx);
    if (!existing) {
      st.correct_solutions.push({
        sample_idx: st.active_sample_idx,
        solution: sample.solution || wa.selected_solution_text || '',
        completed_at: new Date().toISOString(),
      });
    }
    wa.workflow_state = 'completed';
  }
  await persistProgress('completed', false);
  alert(`题目 ${c.id} 标注完成并保存`);
}

async function copySolutionRaw(sampleIdx) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const raw = sampleIdx === null
    ? (wa.selected_solution_text || '')
    : (((c.samples || [])[sampleIdx] || {}).solution || '');
  await navigator.clipboard.writeText(raw);
  const nodeId = sampleIdx === null ? 'copyStatus_active' : `copyStatus_${sampleIdx}`;
  const statusNode = document.getElementById(nodeId);
  if (!statusNode) return;
  statusNode.textContent = 'Copied';
  clearTimeout(saveBadgeTimer);
  saveBadgeTimer = setTimeout(() => {
    statusNode.textContent = '';
  }, 1200);
}

async function openGuide() {
  const res = await fetch('/api/guideline');
  const data = await res.json();
  document.getElementById('guideText').textContent = data.content || '暂无';
}

window.addEventListener('beforeunload', () => {
  const payload = buildProgressPayload('in_progress');
  if (!payload) return;
  navigator.sendBeacon('/api/save_progress', JSON.stringify(payload));
});

document.getElementById('annotator').addEventListener('change', async () => {
  const c = selectedCase();
  if (c) {
    await restoreProgress(c.id);
    renderCurrentCase();
  }
});

initDeviceId();
setSaveState('未保存', 'pending');
