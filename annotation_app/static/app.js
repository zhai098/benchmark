let dataset = [];
let currentCaseIndex = -1;
let currentStep = 1;
const stateByCase = {};
let toastTimer = null;
let autoSaveTimer = null;

function getCaseState(caseId) {
  if (!stateByCase[caseId]) {
    stateByCase[caseId] = {
      sample_validation: [],
      selected_solution_idx: 0,
      sample_pipelines: {},
    };
  }
  return stateByCase[caseId];
}

function selectedCase() { return dataset[currentCaseIndex]; }
function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function typesetMath(root = document.body) {
  if (window.MathJax?.typesetPromise) window.MathJax.typesetPromise([root]).catch(() => {});
}

function copyText(text) {
  navigator.clipboard.writeText(String(text || '')).then(
    () => notify('已复制到剪贴板', 'success'),
    () => notify('复制失败，请手动复制', 'error'),
  );
}

function localProgressKey(annotator) {
  return `annotator_progress_${annotator || 'unknown'}`;
}

function scheduleAutoSave() {
  if (autoSaveTimer) clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(() => {
    const annotator = document.getElementById('annotator')?.value.trim() || 'unknown';
    const payload = {
      datasetPath: document.getElementById('jsonlPath')?.value.trim() || '',
      currentCaseIndex,
      currentStep,
      stateByCase,
      savedAt: new Date().toISOString(),
    };
    localStorage.setItem(localProgressKey(annotator), JSON.stringify(payload));
  }, 250);
}

function ensureSamplePipeline(st, sampleIdx, sample) {
  const key = String(sampleIdx);
  if (!st.sample_pipelines[key]) {
    st.sample_pipelines[key] = {
      selected_solution_text: sample?.solution || '',
      cut_points: [],
      steps: [],
      presegmented_claims: extractPresegmentedClaims(sample || {}),
      claims: [],
      dependencies: {},
    };
  }
  return st.sample_pipelines[key];
}

function currentPipeline() {
  const c = selectedCase();
  if (!c) return null;
  const st = getCaseState(c.id);
  const idx = st.selected_solution_idx || 0;
  return ensureSamplePipeline(st, idx, (c.samples || [])[idx] || {});
}

function getClaimCheckStats(st) {
  const reviewed = (st.presegmented_claims || []).filter(x => x.review_status && x.review_status !== 'unchecked').length;
  const edited = (st.presegmented_claims || []).filter(x => x.review_status === 'edited').length;
  const totalClaims = (st.presegmented_claims || []).length;
  const mapped = (st.presegmented_claims || []).filter(x => Number.isInteger(x.step_idx) && x.step_idx >= 0).length;

  return {
    reviewed,
    edited,
    totalClaims,
    mapped,
    unmapped: Math.max(0, totalClaims - mapped),
  };
}

function getCaseCompletion(st) {
  const pipelineList = Object.values(st.sample_pipelines || {});
  const current = pipelineList[st.selected_solution_idx || 0] || pipelineList[0] || {};
  const sampleDone = (st.sample_validation || []).some(x => x && x.is_correct !== null) ? 1 : 0;
  const stepDone = (current.steps || []).length > 0 ? 1 : 0;
  const claimMapped = (current.claims || []).some(x => (x.claims || []).length > 0) ? 1 : 0;
  const check = getClaimCheckStats(current);
  const claimChecked = check.totalClaims > 0 && check.reviewed === check.totalClaims ? 1 : 0;
  const depDone = Object.keys(current.dependencies || {}).length > 0 ? 1 : 0;
  return Math.round(((sampleDone + stepDone + claimMapped + claimChecked + depDone) / 5) * 100);
}

function notify(msg, level = 'info') {
  let box = document.getElementById('toast');
  if (!box) {
    box = document.createElement('div');
    box.id = 'toast';
    box.className = 'toast';
    document.body.appendChild(box);
  }
  box.className = `toast show ${level}`;
  box.textContent = msg;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove('show'), 2200);
}

function toggleCasePanel() {
  document.getElementById('casePanel').classList.toggle('collapsed');
}

function openContextModal() {
  const modal = document.getElementById('contextModal');
  if (!modal) return;
  modal.classList.add('show');
}

function closeContextModal() {
  const modal = document.getElementById('contextModal');
  if (!modal) return;
  modal.classList.remove('show');
}

async function loadDataset() {
  const path = document.getElementById('jsonlPath').value.trim();
  const annotator = document.getElementById('annotator').value.trim() || 'unknown';
  const res = await fetch('/api/load_jsonl', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
  });
  const data = await res.json();
  if (!res.ok) return notify(data.error || '加载失败', 'error');
  dataset = data.items;
  try {
    const local = JSON.parse(localStorage.getItem(localProgressKey(annotator)) || '{}');
    if (local && local.stateByCase) {
      Object.assign(stateByCase, local.stateByCase);
      if (typeof local.currentCaseIndex === 'number') currentCaseIndex = local.currentCaseIndex;
      if (typeof local.currentStep === 'number') currentStep = local.currentStep;
    }
  } catch (_) {}
  try {
    const remoteRes = await fetch(`/api/load_progress?annotator=${encodeURIComponent(annotator)}`);
    const remote = await remoteRes.json();
    if (remoteRes.ok && remote.cases) {
      Object.entries(remote.cases).forEach(([caseId, caseData]) => {
        stateByCase[caseId] = {
          ...(stateByCase[caseId] || {}),
          ...caseData,
        };
      });
    }
  } catch (_) {}
  renderCaseList();
  if (dataset.length) {
    selectCase(Math.max(0, Math.min(currentCaseIndex, dataset.length - 1)));
    notify(`已加载 ${dataset.length} 条任务`, 'success');
  } else {
    notify('数据集为空', 'warn');
  }
}

function renderCaseList() {
  const ul = document.getElementById('caseList');
  ul.innerHTML = '';
  dataset.forEach((c, i) => {
    const st = getCaseState(c.id);
    const progress = getCaseCompletion(st);
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.className = 'case-item';
    btn.innerHTML = `<span>${escapeHtml(c.id)}</span><span class="case-progress">${progress}%</span>`;
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
  if ((c.samples || []).length) st.selected_solution_idx = st.selected_solution_idx || 0;
  (c.samples || []).forEach((sample, i) => ensureSamplePipeline(st, i, sample));
  renderCurrentCase();
  scheduleAutoSave();
}

function goStep(s) {
  const c = selectedCase();
  if (!c) return notify('请先加载并选择题目', 'warn');
  const pipeline = currentPipeline();
  if (s >= 3 && (pipeline?.steps || []).length === 0) {
    notify('请先完成 Step 2 的步骤切分', 'warn');
    currentStep = 2;
    renderStepContent();
    return;
  }
  currentStep = s;
  renderStepContent();
  scheduleAutoSave();
}

function renderCurrentCase() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const pipeline = currentPipeline() || {};
  const stats = getClaimCheckStats(pipeline);
  const completion = getCaseCompletion(st);
  document.getElementById('caseTitle').innerHTML = `当前问题：${escapeHtml(c.id)} <span class="pill">samples ${(c.samples || []).length}</span> <span class="pill">当前sample ${Number(st.selected_solution_idx || 0) + 1}</span> <span class="pill">steps ${(pipeline.steps || []).length}</span> <span class="pill">claims ${stats.totalClaims}</span> <span class="pill">progress ${completion}%</span>`;
  document.getElementById('qAndA').innerHTML = `题目:\n${escapeHtml(c.question)}\n\n标准答案:\n${escapeHtml(c.reference_answer)}`;
  document.getElementById('known').innerHTML = escapeHtml(JSON.stringify(c.known_solutions || [], null, 2));
  typesetMath(document.getElementById('contextModal'));
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
  renderCaseList();
  scheduleAutoSave();
}

function setSampleField(i, k, v) {
  const st = getCaseState(selectedCase().id);
  sampleRecord(st, i)[k] = v;
  scheduleAutoSave();
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
  if (!res.ok) return notify(data.error || '翻译失败', 'error');
  sampleRecord(st, i).translation = data.translated;
  renderStepContent();
}

function selectSolution(i) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  st.selected_solution_idx = i;
  ensureSamplePipeline(st, i, (c.samples || [])[i] || {});
  renderStepContent();
  renderCaseList();
  scheduleAutoSave();
}

function extractPresegmentedClaims(sample) {
  const raw = sample?.claims_by_step || sample?.step_claims || sample?.claims || [];
  if (!Array.isArray(raw)) return [];
  const out = [];
  raw.forEach((item, i) => {
    if (typeof item === 'string') {
      const text = item.trim();
      if (text) out.push({ id: `p${i + 1}`, text, edited_text: text, review_status: 'unchecked', step_idx: null });
      return;
    }
    if (item && typeof item === 'object' && Array.isArray(item.claims)) {
      const step_idx = Number.isInteger(item.step_index)
        ? item.step_index
        : parseInt(String(item.step_id || '').replace(/[^\d]/g, ''), 10) - 1;
      (item.claims || []).forEach((c, ci) => {
        const text = String(c || '').trim();
        if (text) out.push({ id: `p${i + 1}_${ci + 1}`, text, edited_text: text, review_status: 'unchecked', step_idx: Number.isFinite(step_idx) ? step_idx : null });
      });
      return;
    }
    const text = String(item?.text || item?.claim || '').trim();
    if (!text) return;
    const step_idx = Number.isInteger(item.step_index)
      ? item.step_index
      : parseInt(String(item.step_id || '').replace(/[^\d]/g, ''), 10) - 1;
    out.push({ id: `p${i + 1}`, text, edited_text: text, review_status: 'unchecked', step_idx: Number.isFinite(step_idx) ? step_idx : null });
  });
  return out;
}

function addCutPoint() {
  const p = currentPipeline();
  if (!p) return;
  const ta = document.getElementById('solutionText');
  if (!ta) return;
  const pos = ta.selectionStart;
  p.selected_solution_text = ta.value;
  if (pos > 0 && pos < (p.selected_solution_text || '').length && !p.cut_points.includes(pos)) {
    p.cut_points.push(pos);
    p.cut_points.sort((a, b) => a - b);
    updateSplitPreview();
  }
  scheduleAutoSave();
}

function removeCutPoint(p) {
  const pipeline = currentPipeline();
  if (!pipeline) return;
  pipeline.cut_points = pipeline.cut_points.filter(x => x !== p);
  updateSplitPreview();
  scheduleAutoSave();
}

async function updateSplitPreview() {
  const pipeline = currentPipeline();
  if (!pipeline) return;
  const ta = document.getElementById('solutionText');
  if (ta) pipeline.selected_solution_text = ta.value;
  const res = await fetch('/api/split_steps', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ solution: pipeline.selected_solution_text, cut_points: pipeline.cut_points }),
  });
  const data = await res.json();
  pipeline.steps = (data.steps || []).map((text, i) => ({ id: `s${i + 1}`, text }));
  const box = document.getElementById('splitPreview');
  if (box) box.textContent = JSON.stringify(pipeline.steps, null, 2);
  const cp = document.getElementById('cutPointList');
  if (cp) cp.innerHTML = pipeline.cut_points.map(x => `<button onclick="removeCutPoint(${x})">位置 ${x} ×</button>`).join(' ');
  renderCaseList();
  scheduleAutoSave();
}

function setClaimStep(claimIdx, stepIdx) {
  const pipeline = currentPipeline();
  const claim = (pipeline?.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.step_idx = stepIdx >= 0 ? stepIdx : null;
  scheduleAutoSave();
}

function organizeClaimsBySteps() {
  const pipeline = currentPipeline();
  if (!pipeline) return;
  const stepCount = (pipeline.steps || []).length;
  pipeline.claims = Array.from({ length: stepCount }, (_, i) => ({ step_id: `s${i + 1}`, claims: [] }));
  (pipeline.presegmented_claims || []).forEach((claim, i) => {
    const stepIdx = Number(document.getElementById(`claimStepSel_${i}`)?.value ?? -1);
    if (!Number.isInteger(stepIdx) || stepIdx < 0 || stepIdx >= stepCount) return;
    claim.step_idx = stepIdx;
    const finalText = (claim.review_status === 'edited' ? claim.edited_text : claim.text) || '';
    pipeline.claims[stepIdx].claims.push(finalText.trim());
  });
  renderStepContent();
  renderCaseList();
  scheduleAutoSave();
}

function setClaimReviewStatus(claimIdx, status) {
  const pipeline = currentPipeline();
  const claim = (pipeline?.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.review_status = status;
  if (status === 'ok') claim.edited_text = claim.text;
  scheduleAutoSave();
}

function setClaimReviewStatusAndRender(claimIdx, status) {
  setClaimReviewStatus(claimIdx, status);
  renderStepContent();
  renderCaseList();
}

function setClaimEditedText(claimIdx, v) {
  const pipeline = currentPipeline();
  const claim = (pipeline?.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.edited_text = v;
  scheduleAutoSave();
}

function editClaim(stepIdx, claimIdx, v) {
  const pipeline = currentPipeline();
  if (!pipeline?.claims?.[stepIdx]) return;
  pipeline.claims[stepIdx].claims[claimIdx] = v;
}

function addClaim(stepIdx) {
  const pipeline = currentPipeline();
  if (!pipeline?.claims?.[stepIdx]) return;
  pipeline.claims[stepIdx].claims.push('');
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
  const pipeline = currentPipeline();
  if (!pipeline) return;
  pipeline.dependencies[currId] = pipeline.dependencies[currId] || [];
  if (checked) {
    if (!pipeline.dependencies[currId].includes(depId)) pipeline.dependencies[currId].push(depId);
  } else {
    pipeline.dependencies[currId] = pipeline.dependencies[currId].filter(x => x !== depId);
  }
  renderCaseList();
  scheduleAutoSave();
}

function buildDependencyView() {
  const pipeline = currentPipeline();
  const grouped = flattenClaimsByStep(pipeline?.claims || []);
  let html = '<h3>Step 4：依赖关系（按当前 Step 逐条标注）</h3>';

  grouped.forEach((currStep, sIdx) => {
    html += `<section class="dep-section"><h4>当前 Step ${sIdx + 1}</h4>`;
    currStep.claims.forEach(curr => {
      html += `<div class="dep-card"><div class="curr-claim"><b>${curr.id}</b> ${escapeHtml(curr.text)}</div>`;
      html += '<div class="prev-steps">';
      for (let ps = sIdx; ps >= 0; ps--) {
        const prev = grouped[ps];
        const candidates = prev.claims.filter(c => (c.stepIdx < curr.stepIdx) || (c.stepIdx === curr.stepIdx && c.claimIdx < curr.claimIdx));
        if (!candidates.length) continue;
        html += `<details><summary>前序 Step ${ps + 1}（${candidates.length}条）</summary>`;
        candidates.forEach(cand => {
          const deps = pipeline.dependencies[curr.id] || [];
          const checked = deps.includes(cand.id) ? 'checked' : '';
          html += `<label class="dep-option"><input type="checkbox" ${checked} onchange="toggleDep('${curr.id}','${cand.id}',this.checked)"> <span>${cand.id}</span> ${escapeHtml(cand.text)}</label>`;
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
  const pipeline = currentPipeline() || {};
  const stats = getClaimCheckStats(pipeline);
  return `
    <h3>Step 5：提交前总览</h3>
    <p>请检查以下结果无误后提交：</p>
    <div class="kpi-grid">
      <div class="kpi"><small>当前 sample</small><b>${Number(st.selected_solution_idx || 0) + 1}</b></div>
      <div class="kpi"><small>Step 数</small><b>${(pipeline.steps || []).length}</b></div>
      <div class="kpi"><small>Claim 总数</small><b>${stats.totalClaims}</b></div>
      <div class="kpi"><small>已校验 Claim</small><b>${stats.reviewed}</b></div>
      <div class="kpi"><small>待归属 Claim</small><b>${stats.unmapped}</b></div>
    </div>
    <h4>多采样验证</h4><pre>${JSON.stringify(st.sample_validation, null, 2)}</pre>
    <h4>Step切分（当前 sample）</h4><pre>${JSON.stringify({ solution_index: st.selected_solution_idx, cut_points: pipeline.cut_points, steps: pipeline.steps }, null, 2)}</pre>
    <h4>Claim整理结果（当前 sample）</h4><pre>${JSON.stringify(pipeline.claims, null, 2)}</pre>
    <h4>Claim检查（当前 sample）</h4><pre>${JSON.stringify(pipeline.presegmented_claims, null, 2)}</pre>
    <h4>依赖关系（当前 sample）</h4><pre>${JSON.stringify(pipeline.dependencies, null, 2)}</pre>
    <h4>全部 sample pipeline</h4><pre>${JSON.stringify(st.sample_pipelines || {}, null, 2)}</pre>
    <button class="primary" onclick="submitCase()">确认提交当前问题</button>
  `;
}

function buildStepSplitCompact(pipeline) {
  return (pipeline.steps || []).map((step, i) => `
    <details class="step-mini" ${i === 0 ? 'open' : ''}>
      <summary>Step ${i + 1}</summary>
      <pre>${escapeHtml(step.text)}</pre>
    </details>
  `).join('') || '<p class="muted">尚未生成 Step 切分。</p>';
}

function renderStepContent() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const pipeline = currentPipeline() || ensureSamplePipeline(st, 0, (c.samples || [])[0] || {});
  const root = document.getElementById('stepContent');
  const navButtons = document.querySelectorAll('.step-btn');
  navButtons.forEach((btn) => btn.classList.toggle('active', Number(btn.dataset.step) === currentStep));

  const sampleSelector = (currentStep >= 2) ? `
    <div class="row">
      <label>当前 pipeline sample：
        <select onchange="selectSolution(Number(this.value))">
          ${(c.samples || []).map((_, i) => `<option value="${i}" ${Number(st.selected_solution_idx || 0) === i ? 'selected' : ''}>sample-${i + 1}</option>`).join('')}
        </select>
      </label>
      <small class="muted">每个 sample 的 Step/Claim/依赖会独立保存。</small>
    </div>
  ` : '';
  const verifiedSamples = (c.samples || [])
    .map((s, i) => ({ idx: i, solution: s.solution || '', rec: sampleRecord(st, i) }))
    .filter(x => x.rec?.is_correct === true);

  if (currentStep === 1) {
    let html = '<h3>Step 1：多采样验证（可回退）</h3><p>点击“正确/错误”可切换，再次点击可撤销为未判定。</p>';
    html += `
      <div class="verified-panel">
        <h4>已验证为正确的样本（实时）</h4>
        ${verifiedSamples.length ? verifiedSamples.map(v => `
          <div class="verified-item">
            <b>sample-${v.idx + 1}</b>
            <button class="copy-btn" onclick="copyText(decodeURIComponent('${encodeURIComponent(v.solution)}'))">一键复制</button>
          </div>
        `).join('') : '<p class="muted">暂无已验证正确样本</p>'}
      </div>
    `;
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
          <button class="copy-btn" onclick="copyText(decodeURIComponent('${encodeURIComponent(s.solution || '')}'))">复制solution</button>
        </div>
        <div class="math-text">${escapeHtml(s.solution || '')}</div>
        ${rec.translation ? `<details open><summary>翻译结果</summary><pre>${escapeHtml(rec.translation)}</pre></details>` : ''}
        <div class="row">
          <button class="${clsCorrect}" onclick="chooseSampleStatus(${i}, true)">正确</button>
          <button class="${clsWrong}" onclick="chooseSampleStatus(${i}, false)">错误</button>
          <button class="${clsUnset}" onclick="chooseSampleStatus(${i}, null)">未判定</button>
        </div>
        <div class="row">
          <label>分类 <input value="${escapeHtml(rec.class_name || '')}" oninput="setSampleField(${i}, 'class_name', this.value)"></label>
          <label><input type="checkbox" ${rec.is_new_class ? 'checked' : ''} onchange="setSampleField(${i}, 'is_new_class', this.checked)"> 新分类</label>
          <label>新方法概述 <input value="${escapeHtml(rec.summary || '')}" oninput="setSampleField(${i}, 'summary', this.value)"></label>
        </div>
      </div>`;
    });
    root.innerHTML = html;
    typesetMath(root);
    return;
  }

  if (currentStep === 2) {
    root.innerHTML = `
      <h3>Step 2：Step切分（在完整 solution 上打点）</h3>
      ${sampleSelector}
      <p>当前切分对象：sample-${(st.selected_solution_idx || 0) + 1}。请基于该 sample 原文进行切分。</p>
      <button class="copy-btn" onclick="copyText(decodeURIComponent('${encodeURIComponent(pipeline.selected_solution_text || '')}'))">一键复制当前solution</button>
      <textarea id="solutionText" class="full-solution">${escapeHtml(pipeline.selected_solution_text || '')}</textarea>
      <div class="row">
        <button onclick="addCutPoint()">添加切分点</button>
        <button onclick="updateSplitPreview()">刷新预览</button>
      </div>
      <div id="cutPointList" class="row"></div>
      <h4>切分结果（可回退：删除切分点后刷新）</h4>
      <pre id="splitPreview">${escapeHtml(JSON.stringify(pipeline.steps, null, 2))}</pre>
    `;
    typesetMath(root);
    return;
  }

  if (currentStep === 3) {
    const stepCount = (pipeline.steps || []).length;
    const claims = pipeline.presegmented_claims || [];
    const unassigned = claims.filter(x => !Number.isInteger(x.step_idx) || x.step_idx < 0).length;
    const unchecked = claims.filter(x => !x.review_status || x.review_status === 'unchecked').length;
    let rows = '';
    claims.forEach((cl, i) => {
      const status = cl.review_status || 'unchecked';
      const defaultStepIdx = (Number.isInteger(cl.step_idx) && cl.step_idx >= 0 && cl.step_idx < stepCount) ? cl.step_idx : -1;
      const options = ['<option value="-1">未分配</option>']
        .concat(Array.from({ length: stepCount }, (_, si) => `<option value="${si}" ${defaultStepIdx === si ? 'selected' : ''}>Step ${si + 1}</option>`))
        .join('');
      rows += `
        <tr>
          <td>${cl.id}</td>
          <td>${escapeHtml(cl.text)}</td>
          <td>
            <div class="row">
              <label><input type="radio" name="claimReview_${i}" value="ok" ${status === 'ok' ? 'checked' : ''} onchange="setClaimReviewStatusAndRender(${i}, 'ok')"> 正确</label>
              <label><input type="radio" name="claimReview_${i}" value="edited" ${status === 'edited' ? 'checked' : ''} onchange="setClaimReviewStatusAndRender(${i}, 'edited')"> 需修改</label>
            </div>
            <input value="${escapeHtml(cl.edited_text || cl.text || '')}" ${status === 'edited' ? '' : 'disabled'} oninput="setClaimEditedText(${i}, this.value)">
          </td>
          <td><select id="claimStepSel_${i}" onchange="setClaimStep(${i}, Number(this.value))">${options}</select></td>
        </tr>
      `;
    });
    root.innerHTML = `
      <h3>Step 3-4：Claim检查与Step归属（合并）</h3>
      ${sampleSelector}
      <p>先判断 claim 是否正确；如不正确先修改文本，再进行 Step 归属。</p>
      <div class="split-compare">
        <aside class="split-reference">
          <h4>Step 切分参考</h4>
          ${buildStepSplitCompact(pipeline)}
        </aside>
        <section>
          <div class="kpi-grid">
            <div class="kpi"><small>Step 数</small><b>${stepCount}</b></div>
            <div class="kpi"><small>预切分 Claim</small><b>${claims.length}</b></div>
            <div class="kpi"><small>未检查</small><b>${unchecked}</b></div>
            <div class="kpi"><small>未分配</small><b>${unassigned}</b></div>
          </div>
          <table>
            <thead><tr><th>Claim</th><th>原始文本</th><th>检查与修改</th><th>归属 Step</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="4">当前 solution 未提供预切分 claim</td></tr>'}</tbody>
          </table>
          <div class="row">
            <button class="primary" onclick="organizeClaimsBySteps()">保存并生成 Step-Claim 结构</button>
          </div>
          <pre>${escapeHtml(JSON.stringify(pipeline.claims, null, 2))}</pre>
        </section>
      </div>
    `;
    typesetMath(root);
    return;
  }

  if (currentStep === 4) {
    root.innerHTML = `${sampleSelector}${buildDependencyView()}`;
    typesetMath(root);
    return;
  }

  if (currentStep === 5) {
    root.innerHTML = `${sampleSelector}${buildSummaryView()}`;
    typesetMath(root);
  }
}

async function saveProgress() {
  const c = selectedCase(); if (!c) return notify('先选择问题', 'warn');
  const annotator = document.getElementById('annotator').value.trim() || 'unknown';
  const st = getCaseState(c.id);
  const payload = {
    annotator,
    case_id: c.id,
    sample_validation: st.sample_validation,
    selected_solution_idx: st.selected_solution_idx,
    sample_pipelines: st.sample_pipelines,
    status: 'in_progress',
  };
  const res = await fetch('/api/save_record', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) return notify(data.error || '保存失败', 'error');
  notify(`已保存: ${data.path}`, 'success');
  renderCaseList();
}

async function submitCase() {
  const c = selectedCase(); if (!c) return notify('先选择问题', 'warn');
  const annotator = document.getElementById('annotator').value.trim() || 'unknown';
  const st = getCaseState(c.id);
  const payload = {
    annotator,
    case_id: c.id,
    sample_validation: st.sample_validation,
    selected_solution_idx: st.selected_solution_idx,
    sample_pipelines: st.sample_pipelines,
    status: 'completed',
  };
  const res = await fetch('/api/save_record', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) return notify(data.error || '提交失败', 'error');
  notify(`题目 ${c.id} 标注完成并自动保存`, 'success');
  renderCaseList();
}

async function openGuide() {
  const res = await fetch('/api/guideline');
  const data = await res.json();
  document.getElementById('guideText').textContent = data.content || '暂无';
}

window.addEventListener('beforeunload', () => {
  const annotator = document.getElementById('annotator')?.value.trim() || 'unknown';
  const payload = {
    datasetPath: document.getElementById('jsonlPath')?.value.trim() || '',
    currentCaseIndex,
    currentStep,
    stateByCase,
    savedAt: new Date().toISOString(),
  };
  localStorage.setItem(localProgressKey(annotator), JSON.stringify(payload));
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeContextModal();
});
