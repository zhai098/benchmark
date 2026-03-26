let dataset = [];
let currentCaseIndex = -1;
let currentStep = 1;
const stateByCase = {};
let toastTimer = null;

function getCaseState(caseId) {
  if (!stateByCase[caseId]) {
    stateByCase[caseId] = {
      sample_validation: [],
      selected_solution_idx: 0,
      selected_solution_text: '',
      cut_points: [],
      steps: [],
      presegmented_claims: [],
      claims: [],
      dependencies: {},
    };
  }
  return stateByCase[caseId];
}

function selectedCase() { return dataset[currentCaseIndex]; }
function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
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
  const sampleDone = (st.sample_validation || []).some(x => x && x.is_correct !== null) ? 1 : 0;
  const stepDone = (st.steps || []).length > 0 ? 1 : 0;
  const claimMapped = (st.claims || []).some(x => (x.claims || []).length > 0) ? 1 : 0;
  const check = getClaimCheckStats(st);
  const claimChecked = check.totalClaims > 0 && check.reviewed === check.totalClaims ? 1 : 0;
  const depDone = Object.keys(st.dependencies || {}).length > 0 ? 1 : 0;
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
  const res = await fetch('/api/load_jsonl', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
  });
  const data = await res.json();
  if (!res.ok) return notify(data.error || '加载失败', 'error');
  dataset = data.items;
  renderCaseList();
  if (dataset.length) {
    selectCase(0);
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
  if (!st.selected_solution_text && (c.samples || []).length) {
    st.selected_solution_idx = 0;
    st.selected_solution_text = c.samples[0].solution || '';
    st.presegmented_claims = extractPresegmentedClaims(c.samples[0] || {});
  }
  renderCurrentCase();
}

function goStep(s) {
  const c = selectedCase();
  if (!c) return notify('请先加载并选择题目', 'warn');
  const st = getCaseState(c.id);
  if (s >= 3 && (st.steps || []).length === 0) {
    notify('请先完成 Step 2 的步骤切分', 'warn');
    currentStep = 2;
    renderStepContent();
    return;
  }
  currentStep = s;
  renderStepContent();
}

function renderCurrentCase() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const stats = getClaimCheckStats(st);
  const completion = getCaseCompletion(st);
  document.getElementById('caseTitle').innerHTML = `当前问题：${escapeHtml(c.id)} <span class="pill">samples ${(c.samples || []).length}</span> <span class="pill">steps ${(st.steps || []).length}</span> <span class="pill">claims ${stats.totalClaims}</span> <span class="pill">progress ${completion}%</span>`;
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
  renderCaseList();
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
  if (!res.ok) return notify(data.error || '翻译失败', 'error');
  sampleRecord(st, i).translation = data.translated;
  renderStepContent();
}

function selectSolution(i) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const sample = (c.samples || [])[i] || {};
  st.selected_solution_idx = i;
  st.selected_solution_text = sample.solution || '';
  st.cut_points = [];
  st.steps = [];
  st.presegmented_claims = extractPresegmentedClaims(sample);
  st.claims = [];
  st.dependencies = {};
  renderStepContent();
  renderCaseList();
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
  if (cp) cp.innerHTML = st.cut_points.map(x => `<button onclick="removeCutPoint(${x})">位置 ${x} ×</button>`).join(' ');
  renderCaseList();
}

function setClaimStep(claimIdx, stepIdx) {
  const st = getCaseState(selectedCase().id);
  const claim = (st.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.step_idx = stepIdx >= 0 ? stepIdx : null;
}

function organizeClaimsBySteps() {
  const st = getCaseState(selectedCase().id);
  const stepCount = (st.steps || []).length;
  st.claims = Array.from({ length: stepCount }, (_, i) => ({ step_id: `s${i + 1}`, claims: [] }));
  (st.presegmented_claims || []).forEach((claim, i) => {
    const stepIdx = Number(document.getElementById(`claimStepSel_${i}`)?.value ?? -1);
    if (!Number.isInteger(stepIdx) || stepIdx < 0 || stepIdx >= stepCount) return;
    claim.step_idx = stepIdx;
    const finalText = (claim.review_status === 'edited' ? claim.edited_text : claim.text) || '';
    st.claims[stepIdx].claims.push(finalText.trim());
  });
  renderStepContent();
  renderCaseList();
}

function setClaimReviewStatus(claimIdx, status) {
  const st = getCaseState(selectedCase().id);
  const claim = (st.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.review_status = status;
  if (status === 'ok') claim.edited_text = claim.text;
}

function setClaimEditedText(claimIdx, v) {
  const st = getCaseState(selectedCase().id);
  const claim = (st.presegmented_claims || [])[claimIdx];
  if (!claim) return;
  claim.edited_text = v;
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
  renderCaseList();
}

function buildDependencyView() {
  const st = getCaseState(selectedCase().id);
  const grouped = flattenClaimsByStep(st.claims);
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
          const deps = st.dependencies[curr.id] || [];
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
  const stats = getClaimCheckStats(st);
  return `
    <h3>Step 5：提交前总览</h3>
    <p>请检查以下结果无误后提交：</p>
    <div class="kpi-grid">
      <div class="kpi"><small>Step 数</small><b>${(st.steps || []).length}</b></div>
      <div class="kpi"><small>Claim 总数</small><b>${stats.totalClaims}</b></div>
      <div class="kpi"><small>已校验 Claim</small><b>${stats.reviewed}</b></div>
      <div class="kpi"><small>待归属 Claim</small><b>${stats.unmapped}</b></div>
    </div>
    <h4>多采样验证</h4><pre>${JSON.stringify(st.sample_validation, null, 2)}</pre>
    <h4>Step切分（来自完整 solution 的切分点）</h4><pre>${JSON.stringify({ solution_index: st.selected_solution_idx, cut_points: st.cut_points, steps: st.steps }, null, 2)}</pre>
    <h4>Claim整理结果（按 step）</h4><pre>${JSON.stringify(st.claims, null, 2)}</pre>
    <h4>Claim检查（原始/修改）</h4><pre>${JSON.stringify(st.presegmented_claims, null, 2)}</pre>
    <h4>依赖关系</h4><pre>${JSON.stringify(st.dependencies, null, 2)}</pre>
    <button class="primary" onclick="submitCase()">确认提交当前问题</button>
  `;
}

function buildStepSplitCompact(st) {
  return (st.steps || []).map((step, i) => `
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
  const root = document.getElementById('stepContent');
  const navButtons = document.querySelectorAll('.step-btn');
  navButtons.forEach((btn) => btn.classList.toggle('active', Number(btn.dataset.step) === currentStep));

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
        <pre>${escapeHtml(s.solution || '')}</pre>
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
    return;
  }

  if (currentStep === 2) {
    root.innerHTML = `
      <h3>Step 2：Step切分（在完整 solution 上打点）</h3>
      <p>当前切分对象：sample-${(st.selected_solution_idx || 0) + 1}。在下方文本中将光标移动到切分位置后点击“添加切分点”。</p>
      <textarea id="solutionText" class="full-solution">${escapeHtml(st.selected_solution_text || '')}</textarea>
      <div class="row">
        <button onclick="addCutPoint()">添加切分点</button>
        <button onclick="updateSplitPreview()">刷新预览</button>
      </div>
      <div id="cutPointList" class="row"></div>
      <h4>切分结果（可回退：删除切分点后刷新）</h4>
      <pre id="splitPreview">${escapeHtml(JSON.stringify(st.steps, null, 2))}</pre>
    `;
    return;
  }

  if (currentStep === 3) {
    const stepCount = (st.steps || []).length;
    const claims = st.presegmented_claims || [];
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
              <label><input type="radio" name="claimReview_${i}" value="ok" ${status === 'ok' ? 'checked' : ''} onchange="setClaimReviewStatus(${i}, 'ok')"> 正确</label>
              <label><input type="radio" name="claimReview_${i}" value="edited" ${status === 'edited' ? 'checked' : ''} onchange="setClaimReviewStatus(${i}, 'edited')"> 需修改</label>
            </div>
            <input value="${escapeHtml(cl.edited_text || cl.text || '')}" ${status === 'edited' ? '' : 'disabled'} oninput="setClaimEditedText(${i}, this.value)">
          </td>
          <td><select id="claimStepSel_${i}" onchange="setClaimStep(${i}, Number(this.value))">${options}</select></td>
        </tr>
      `;
    });
    root.innerHTML = `
      <h3>Step 3-4：Claim检查与Step归属（合并）</h3>
      <p>先判断 claim 是否正确；如不正确先修改文本，再进行 Step 归属。</p>
      <div class="split-compare">
        <aside class="split-reference">
          <h4>Step 切分参考</h4>
          ${buildStepSplitCompact(st)}
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
          <pre>${escapeHtml(JSON.stringify(st.claims, null, 2))}</pre>
        </section>
      </div>
    `;
    return;
  }

  if (currentStep === 4) {
    root.innerHTML = buildDependencyView();
    return;
  }

  if (currentStep === 5) {
    root.innerHTML = buildSummaryView();
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
    selected_solution_text: st.selected_solution_text,
    cut_points: st.cut_points,
    steps: st.steps,
    presegmented_claims: st.presegmented_claims,
    claims: st.claims,
    dependencies: st.dependencies,
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
    selected_solution_text: st.selected_solution_text,
    cut_points: st.cut_points,
    steps: st.steps,
    presegmented_claims: st.presegmented_claims,
    claims: st.claims,
    dependencies: st.dependencies,
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

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeContextModal();
});
