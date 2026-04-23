let dataset = [];
let currentCaseIndex = -1;
let currentStep = 0;
const stateByCase = {};
let deviceId = '';
let autosaveTimer = null;
let saveBadgeTimer = null;
let saveRequestSeq = 0;
let referenceTab = 'problem';
const draftCacheVersion = 1;
const draftCachePrefix = 'annotation_draft_v1';
const layoutPrefs = {
  leftWidth: 280,
  rightWidth: 360,
  leftCollapsed: false,
  rightCollapsed: false,
};
const layoutStorageKey = 'annotation_layout_prefs_v2';

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

function showToast(message, type = 'success') {
  const region = document.getElementById('toastRegion');
  if (!region) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  region.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

async function copyTextRobust(rawText) {
  const text = String(rawText ?? '');
  if (!text) throw new Error('没有可复制的文本');
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (err) {
      // fallback below
    }
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  ta.setSelectionRange(0, ta.value.length);
  const ok = document.execCommand('copy');
  document.body.removeChild(ta);
  if (!ok) throw new Error('浏览器禁止复制，请手动复制');
}

function getDefaultWorkingAnnotation(sample = {}) {
  return {
    selected_solution_text: sample.solution || '',
    cut_points: [],
    steps: [],
    presegmented_claims: normalizePresegmentedClaims(extractPresegmentedClaims(sample), sample),
    claims: [],
    claim_checks: {},
    dependencies: {},
    step_dependencies: {},
    workflow_state: 'sample_selected',
  };
}

function createDefaultCaseState() {
  return {
    current_step: 0,
    active_sample_idx: null,
    sample_cursor: 0,
    client_revision: 0,
    problem_quality_screening: {
      decision: null,
      reason: '',
      other_text: '',
      rejected_at: '',
    },
    sample_validation: [],
    sample_annotations: {},
    correct_solutions: [],
    ui: {
      showRawText: false,
      pinRawText: false,
      rawPanelWidth: 360,
      claimPreviewWidth: 360,
      stepContextWidth: 360,
      depStepIdx: 0,
    },
    last_applied_save_seq: 0,
    last_saved_hash: '',
    last_saved_fingerprint: '',
    last_saved_at_utc: '',
    draft_cached_at_utc: '',
    restore_source: '',
  };
}

function resetCaseState(caseId) {
  stateByCase[caseId] = createDefaultCaseState();
  return stateByCase[caseId];
}

function getCaseState(caseId) {
  if (!stateByCase[caseId]) {
    stateByCase[caseId] = createDefaultCaseState();
  }
  return stateByCase[caseId];
}

function deriveCaseWorkflowState(st) {
  const order = {
    sample_selected: 0,
    steps_segmented: 1,
    claims_assigned: 2,
    claims_checked: 3,
    dependencies_labeled: 4,
    completed: 5,
  };
  const states = [];
  if (st.active_sample_idx !== null) {
    states.push(getWorkingAnnotation(st).workflow_state || 'sample_selected');
  }
  Object.values(st.sample_annotations || {}).forEach((ann) => {
    if (ann?.workflow_state) states.push(ann.workflow_state);
  });
  const statuses = (st.sample_validation || []).map((item) => item?.pipeline_status);
  if (statuses.length && statuses.every((status) => status === 'completed' || status === 'discarded')) {
    states.push('completed');
  }
  if (!states.length) return 'sample_selected';
  return states.reduce((best, state) => ((order[state] ?? -1) > (order[best] ?? -1) ? state : best), 'sample_selected');
}

function nextClientRevision(st) {
  const now = Date.now();
  st.client_revision = Math.max(now, (st.client_revision || 0) + 1);
  return st.client_revision;
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

function getWorkspaceMainWidth() {
  return document.querySelector('.workspace-main')?.clientWidth || window.innerWidth || 0;
}

function shouldStackContextPanels(auxCount = 1) {
  const mainWidth = getWorkspaceMainWidth();
  const minMainWidth = auxCount > 1 ? 360 : 420;
  const minSideWidth = 260;
  const handleWidth = auxCount * 10;
  return mainWidth > 0 && mainWidth < (minMainWidth + handleWidth + (minSideWidth * auxCount));
}

function clampContextSideWidth(desiredWidth, auxCount = 1) {
  const mainWidth = getWorkspaceMainWidth();
  if (!mainWidth) return desiredWidth;
  const minMainWidth = auxCount > 1 ? 360 : 420;
  const handleWidth = auxCount * 10;
  const maxByLayout = Math.floor((mainWidth - minMainWidth - handleWidth) / auxCount);
  return Math.max(260, Math.min(desiredWidth, maxByLayout > 0 ? maxByLayout : desiredWidth));
}

function draftCacheKey(caseId, annotator = annotatorId()) {
  return `${draftCachePrefix}:${annotator}:${caseId}`;
}

function buildDraftCacheEnvelope(caseId, payload) {
  return {
    schema_version: draftCacheVersion,
    annotator_id: annotatorId(),
    device_id: deviceId,
    case_id: caseId,
    cached_at_utc: new Date().toISOString(),
    progress: payload,
  };
}

function writeDraftCache(caseId, payload) {
  if (!caseId || !payload) return;
  try {
    const envelope = buildDraftCacheEnvelope(caseId, payload);
    localStorage.setItem(draftCacheKey(caseId), JSON.stringify(envelope));
    const st = getCaseState(caseId);
    st.draft_cached_at_utc = envelope.cached_at_utc;
  } catch (_) {
    // Ignore quota/private-mode issues; server persistence remains primary.
  }
}

function readDraftCache(caseId) {
  if (!caseId) return null;
  try {
    const raw = localStorage.getItem(draftCacheKey(caseId));
    if (!raw) return null;
    const envelope = JSON.parse(raw);
    if (!envelope || typeof envelope !== 'object') return null;
    if (String(envelope.annotator_id || '') !== annotatorId()) return null;
    if (String(envelope.case_id || '') !== String(caseId)) return null;
    if (!envelope.progress || typeof envelope.progress !== 'object') return null;
    return envelope;
  } catch (_) {
    return null;
  }
}

function hasMeaningfulProgress(progress) {
  if (!progress || typeof progress !== 'object') return false;
  const workflow = progress.current_workflow_state || {};
  const annotations = progress.current_annotations || {};
  const screening = workflow.problem_quality_screening || {};
  const decisions = Array.isArray(progress.sample_decisions) ? progress.sample_decisions : [];
  const correctSolutions = Array.isArray(progress.correct_solutions) ? progress.correct_solutions : [];
  const sampleAnnotations = annotations.sample_annotations && typeof annotations.sample_annotations === 'object'
    ? annotations.sample_annotations
    : {};
  return Boolean(
    (progress.current_step || 0) > 0
    || workflow.active_sample_idx !== null
    || (workflow.sample_cursor || 0) > 0
    || screening.decision
    || decisions.some((item) => item && (
      item.is_correct !== null
      || item.pipeline_status && item.pipeline_status !== 'not_started'
      || item.class_name
      || item.summary
    ))
    || correctSolutions.length > 0
    || Object.keys(sampleAnnotations).length > 0
    || annotations.selected_solution_text
    || (Array.isArray(annotations.steps) && annotations.steps.length > 0)
    || (Array.isArray(annotations.claims) && annotations.claims.length > 0)
  );
}

function applyRestoredProgress(caseId, progress, source = '') {
  const st = resetCaseState(caseId);
  const c = selectedCase();
  const caseSamples = Array.isArray(c?.samples) ? c.samples : [];
  st.restore_source = source;
  st.current_step = progress.current_step || 0;
  currentStep = st.current_step;
  st.problem_quality_screening = progress.current_workflow_state?.problem_quality_screening || {
    decision: null, reason: '', other_text: '', rejected_at: '',
  };
  st.sample_validation = progress.sample_decisions || [];
  st.correct_solutions = progress.correct_solutions || [];
  st.active_sample_idx = progress.current_workflow_state?.active_sample_idx ?? null;
  st.client_revision = Number(progress.client_revision || 0) || 0;
  st.last_saved_at_utc = String(progress.updated_at_utc || '');
  const savedAnnotations = progress.current_annotations || {};
  const rawSampleAnnotations = savedAnnotations.sample_annotations && typeof savedAnnotations.sample_annotations === 'object'
    ? savedAnnotations.sample_annotations
    : {};
  st.sample_annotations = {};
  Object.entries(rawSampleAnnotations).forEach(([sampleIdx, ann]) => {
    const sample = caseSamples[Number(sampleIdx)] || {};
    const savedAnn = ann && typeof ann === 'object' ? ann : {};
    st.sample_annotations[sampleIdx] = {
      ...savedAnn,
      presegmented_claims: normalizePresegmentedClaims(savedAnn.presegmented_claims, sample),
    };
  });
  if (st.active_sample_idx !== null && !st.sample_annotations[st.active_sample_idx]) {
    const sample = caseSamples[st.active_sample_idx] || {};
    st.sample_annotations[st.active_sample_idx] = {
      selected_solution_text: savedAnnotations.selected_solution_text || '',
      cut_points: savedAnnotations.cut_points || [],
      steps: savedAnnotations.steps || [],
      presegmented_claims: normalizePresegmentedClaims(savedAnnotations.presegmented_claims || [], sample),
      claims: savedAnnotations.claims || [],
      claim_checks: savedAnnotations.claim_checks || {},
      dependencies: savedAnnotations.dependencies || {},
      step_dependencies: savedAnnotations.step_dependencies || {},
      workflow_state: progress.current_workflow_state?.workflow_state || 'sample_selected',
    };
  }
  st.sample_cursor = Number.isInteger(progress.current_workflow_state?.sample_cursor)
    ? progress.current_workflow_state.sample_cursor
    : (st.sample_cursor || 0);
  const qualityPassed = st.problem_quality_screening?.decision === 'pass';
  if (!qualityPassed) {
    st.current_step = 0;
    currentStep = 0;
  }
  return st;
}
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

function renderMathPreviewBlock(rawText, emptyText = '暂无内容') {
  const text = String(rawText || '').trim();
  if (!text) {
    return `<div class="muted-note">${escapeHtml(emptyText)}</div>`;
  }
  return `<div class="rendered-math compact-rendered-math">${renderLatexWithFallback(text)}</div>`;
}

function getClaimCheckStats(st) {
  const wa = getWorkingAnnotation(st);
  const total = (wa.claims || []).reduce((acc, x) => acc + (x.claims || []).length, 0);
  const checks = wa.claim_checks || {};
  let correct = 0;
  let incorrect = 0;
  let deleted = 0;
  Object.values(checks).forEach(v => {
    if (v === 'correct') correct += 1;
    else if (v === 'incorrect') incorrect += 1;
    else if (v === 'delete') deleted += 1;
  });
  return {
    total, correct, incorrect, deleted,
    checked: correct + incorrect + deleted,
    unchecked: Math.max(0, total - correct - incorrect - deleted),
  };
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
  const filter = (document.getElementById('taskFilter')?.value || '').trim().toLowerCase();
  ul.innerHTML = '';
  dataset.forEach((c, i) => {
    if (filter && !String(c.id || '').toLowerCase().includes(filter)) return;
    const st = getCaseState(c.id);
    const progress = getTaskProgressSummary(c, st);
    const li = document.createElement('li');
    const btn = document.createElement('button');
    const tooltipLines = getTaskProgressDetailLines(c, st, progress);
    btn.title = tooltipLines.join('\n');
    btn.classList.toggle('active-task', i === currentCaseIndex);
    btn.classList.add('task-nav-btn', `task-status-${progress.status}`);
    btn.innerHTML = `
      <span class="task-nav-top">
        <span class="task-nav-id">${escapeHtml(c.id || '')}</span>
        <span class="task-nav-percent">${progress.percent}%</span>
      </span>
      <span class="task-nav-meta">
        <span class="task-nav-label">${escapeHtml(progress.label)}</span>
        <span class="task-nav-ratio">${progress.ratioText}</span>
      </span>
      <span class="task-progress-bar" aria-hidden="true">
        <span class="task-progress-fill" style="width:${progress.percent}%"></span>
      </span>
      <span class="task-nav-tooltip" role="tooltip">
        ${tooltipLines.map((line) => `<span>${escapeHtml(line)}</span>`).join('')}
      </span>
    `;
    btn.onclick = () => selectCase(i);
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

function getTaskProgressSummary(c, st) {
  const totalSamplesRaw = Array.isArray(c?.samples) ? c.samples.length : 0;
  const totalSamples = Math.max(1, totalSamplesRaw);
  const screening = ensureProblemQualityScreening(st);
  const resolvedSamples = (st.sample_validation || []).filter((item) => {
    const status = item?.pipeline_status;
    return status === 'completed' || status === 'discarded';
  }).length;
  const completedSamples = (st.sample_validation || []).filter((item) => item?.pipeline_status === 'completed').length;
  const discardedSamples = (st.sample_validation || []).filter((item) => item?.pipeline_status === 'discarded').length;
  const activeStep = Number(st.current_step || 0);
  let status = 'idle';
  let label = '未开始';
  let ratioText = totalSamplesRaw > 0 ? `${resolvedSamples}/${totalSamplesRaw}` : '0/0';
  let percent = 0;

  if (screening.decision === 'reject') {
    status = 'rejected';
    label = '已拒绝';
    percent = 100;
    ratioText = totalSamplesRaw > 0 ? `${discardedSamples}/${totalSamplesRaw}` : '0/0';
  } else if (screening.decision === 'pass') {
    const activeSampleProgress = st.active_sample_idx !== null
      ? Math.max(0, Math.min(1, (Math.max(2, activeStep) - 1) / 5))
      : 0;
    const progressUnits = 1 + resolvedSamples + activeSampleProgress;
    percent = Math.max(5, Math.min(100, Math.round((progressUnits / (1 + totalSamples)) * 100)));
    if (resolvedSamples >= totalSamplesRaw && totalSamplesRaw > 0) {
      status = 'done';
      label = '已完成';
      percent = 100;
    } else if (st.active_sample_idx !== null) {
      status = 'active';
      label = `进行中 · Step ${Math.max(2, activeStep)}`;
    } else if (resolvedSamples > 0) {
      status = 'reviewed';
      label = '样本处理中';
    } else {
      status = 'screened';
      label = '已通过质检';
    }
  }

  if (completedSamples > 0 && status !== 'done') {
    ratioText = totalSamplesRaw > 0 ? `${completedSamples}+${discardedSamples}/${totalSamplesRaw}` : ratioText;
  }

  return { percent, status, label, ratioText };
}

function normalizedMethodName(value) {
  return String(value || '').trim();
}

function getOptimalSampleEntries(st) {
  const entries = [];
  const seen = new Set();
  (st.correct_solutions || []).forEach((item) => {
    const idx = Number(item?.sample_idx);
    if (!Number.isInteger(idx) || seen.has(idx)) return;
    seen.add(idx);
    entries.push({
      sample_idx: idx,
      completed_at: String(item?.completed_at || ''),
      solution: String(item?.solution || ''),
    });
  });
  return entries;
}

function getMethodLockOwners(st) {
  const owners = {};
  getOptimalSampleEntries(st).forEach((entry) => {
    const rec = sampleRecord(st, entry.sample_idx);
    const method = normalizedMethodName(rec.class_name);
    if (!method || owners[method] !== undefined) return;
    owners[method] = entry.sample_idx;
  });
  return owners;
}

function getSampleMethodLockInfo(st, sampleIdx) {
  const rec = sampleRecord(st, sampleIdx);
  const method = normalizedMethodName(rec.class_name);
  if (!method) return { locked: false, ownerIdx: null, method: '' };
  const ownerIdx = getMethodLockOwners(st)[method];
  if (ownerIdx === undefined || ownerIdx === sampleIdx) {
    return { locked: false, ownerIdx: ownerIdx ?? null, method };
  }
  return { locked: true, ownerIdx, method };
}

function getSampleStageLabel(st, sampleIdx) {
  const rec = sampleRecord(st, sampleIdx);
  const workflowState = st.sample_annotations?.[sampleIdx]?.workflow_state || '';
  if (rec.pipeline_status === 'completed') return '已完成';
  if (rec.pipeline_status === 'discarded') return '已丢弃';
  if (sampleIdx === st.active_sample_idx && rec.pipeline_status === 'in_progress') {
    return `进行中 · Step ${Math.max(2, Number(st.current_step || 2))}`;
  }
  if (rec.pipeline_status === 'ready') return '待进入主流程';
  if (workflowState === 'claims_checked') return '已做 Step 4';
  if (workflowState === 'claims_assigned') return '已做 Step 3';
  if (workflowState === 'steps_segmented') return '已做 Step 2';
  if (workflowState === 'sample_selected') return '待做 Step 2';
  return '未开始';
}

function buildSelectedSampleSummary(st) {
  const parts = getOptimalSampleEntries(st).map((entry) => {
    const rec = sampleRecord(st, entry.sample_idx);
    const method = normalizedMethodName(rec.class_name) || '未命名方法';
    return `sample-${entry.sample_idx + 1} (${method})`;
  });
  return parts.length ? parts.join('；') : '无';
}

function getTaskProgressDetailLines(c, st, progress = getTaskProgressSummary(c, st)) {
  const totalSamples = Array.isArray(c?.samples) ? c.samples.length : 0;
  const screening = ensureProblemQualityScreening(st);
  const validations = Array.isArray(st.sample_validation) ? st.sample_validation : [];
  const completedSamples = validations.filter((item) => item?.pipeline_status === 'completed').length;
  const discardedSamples = validations.filter((item) => item?.pipeline_status === 'discarded').length;
  const inProgressSamples = validations.filter((item) => item?.pipeline_status === 'in_progress').length;
  const latestWorkflow = deriveCaseWorkflowState(st);
  const activeSample = st.active_sample_idx === null || st.active_sample_idx === undefined
    ? '无'
    : `sample-${Number(st.active_sample_idx) + 1}`;
  const latestSaved = st.last_saved_hash || st.last_saved_at_utc
    ? formatUtcToLocal(st.last_saved_at_utc || '')
    : '未检测到已保存记录';
  const screeningLabel = screening.decision === 'pass'
    ? '已通过'
    : screening.decision === 'reject'
      ? `已拒绝${screening.reason ? ` · ${screening.reason}` : ''}`
      : '未质检';
  return [
    `标注者：${annotatorId()}`,
    `任务：${String(c?.id || '')}`,
    `总体进度：${progress.percent}% · ${progress.label}`,
    `已选最优样本：${buildSelectedSampleSummary(st)}`,
    `题目质检：${screeningLabel}`,
    `样本进度：完成 ${completedSamples} / 丢弃 ${discardedSamples} / 处理中 ${inProgressSamples} / 总计 ${totalSamples}`,
    `当前活跃样本：${activeSample}`,
    `当前步骤：Step ${Math.max(0, Number(st.current_step || 0))}`,
    `工作流状态：${latestWorkflow}`,
    `最近本地缓存：${st.draft_cached_at_utc ? formatUtcToLocal(st.draft_cached_at_utc) : '暂无'}`,
    `恢复来源：${st.restore_source || '初始数据'}`,
    `最近同步状态：${latestSaved}`,
  ];
}

async function selectCase(idx) {
  if (idx === currentCaseIndex) return;
  await flushAutosave();
  currentCaseIndex = idx;
  const c = selectedCase();
  await restoreProgress(c.id);
  const st = getCaseState(c.id);
  const qualityPassed = st.problem_quality_screening?.decision === 'pass';
  currentStep = qualityPassed ? Math.max(1, st.current_step || 1) : 0;
  st.current_step = currentStep;
  if (!Number.isInteger(st.sample_cursor)) st.sample_cursor = 0;
  renderCurrentCase();
}

function goStep(s) {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const qualityPassed = st.problem_quality_screening?.decision === 'pass';
  if (s > 0 && !qualityPassed) {
    currentStep = 0;
    st.current_step = 0;
    showToast('请先完成题目质量筛查（通过后才能进入后续流程）', 'error');
    renderStepContent();
    return;
  }
  currentStep = s;
  st.current_step = s;
  scheduleAutosave();
  renderStepContent();
}

function renderCurrentCase() {
  const c = selectedCase();
  if (!c) return;
  renderReferencePanel(c);
  renderCaseList();
  renderStepContent();
}

function renderReferencePanel(c) {
  const problemNode = document.getElementById('problemContent');
  const solutionNode = document.getElementById('solutionContent');
  if (!problemNode || !solutionNode) return;
  problemNode.innerHTML = renderLatexWithFallback(c?.question || '未加载');
  solutionNode.innerHTML = renderLatexWithFallback(c?.reference_answer || '未加载');
}

function sampleRecord(st, i) {
  st.sample_validation[i] = st.sample_validation[i] || {
    is_correct: null,
    class_name: '',
    is_new_class: false,
    summary: '',
    pipeline_status: 'not_started',
  };
  return st.sample_validation[i];
}

function formatUtcToLocal(iso) {
  if (!iso) return '未保存';
  const d = new Date(iso);
  return `${d.toLocaleTimeString()}.${String(d.getMilliseconds()).padStart(3, '0')}`;
}

function findNextSampleCursor(st, start) {
  const c = selectedCase();
  const total = (c?.samples || []).length;
  for (let i = start; i < total; i += 1) {
    const rec = sampleRecord(st, i);
    if (rec.pipeline_status !== 'completed' && rec.pipeline_status !== 'discarded') return i;
  }
  return Math.max(0, total - 1);
}

function setActiveSampleFromCursor() {
  const c = selectedCase();
  const st = getCaseState(c.id);
  if (st.problem_quality_screening?.decision !== 'pass') {
    alert('请先在题目质量筛查步骤完成通过后再开始样本流程。');
    currentStep = 0;
    st.current_step = 0;
    renderStepContent();
    return;
  }
  const idx = st.sample_cursor || 0;
  const rec = sampleRecord(st, idx);
  if (rec.is_correct !== true) {
    alert('当前样本尚未判定为正确，不能进入主工作流。');
    return;
  }
  const lockInfo = getSampleMethodLockInfo(st, idx);
  if (lockInfo.locked) {
    alert(`该方法已由 sample-${lockInfo.ownerIdx + 1} 入选最优样本，不能再进入同方法其他样本的主流程。`);
    return;
  }
  st.active_sample_idx = idx;
  rec.pipeline_status = 'in_progress';
  if (!st.sample_annotations[idx]) {
    st.sample_annotations[idx] = getDefaultWorkingAnnotation((c.samples || [])[idx] || {});
  }
  currentStep = 2;
  st.current_step = currentStep;
  scheduleAutosave();
  renderCurrentCase();
}

function chooseSampleStatus(i, status) {
  const st = getCaseState(selectedCase().id);
  const rec = sampleRecord(st, i);
  if (status === true) {
    const lockInfo = getSampleMethodLockInfo(st, i);
    if (lockInfo.locked) {
      showToast(`同方法已由 sample-${lockInfo.ownerIdx + 1} 入选最优样本，请不要再选择该方法的其他样本。`, 'error');
      renderStepContent();
      return;
    }
  }
  rec.is_correct = status;

  if (status === false) {
    rec.pipeline_status = 'discarded';
    delete st.sample_annotations[i];
    st.correct_solutions = (st.correct_solutions || []).filter(x => x.sample_idx !== i);
    if (st.active_sample_idx === i) {
      st.active_sample_idx = null;
    }
    if (st.sample_cursor === i) {
      st.sample_cursor = findNextSampleCursor(st, i + 1);
    }
  }

  if (status === true && !st.sample_annotations[i]) {
    rec.pipeline_status = rec.pipeline_status === 'completed' ? 'completed' : 'ready';
    st.sample_annotations[i] = getDefaultWorkingAnnotation((selectedCase().samples || [])[i] || {});
  }
  if (status === null) rec.pipeline_status = 'not_started';

  scheduleAutosave();
  renderStepContent();
}

function ensureProblemQualityScreening(st) {
  st.problem_quality_screening = st.problem_quality_screening || {
    decision: null, reason: '', other_text: '', rejected_at: '',
  };
  if (typeof st.problem_quality_screening !== 'object') {
    st.problem_quality_screening = { decision: null, reason: '', other_text: '', rejected_at: '' };
  }
  return st.problem_quality_screening;
}

function setProblemQualityRejectReason(reason) {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const screening = ensureProblemQualityScreening(st);
  screening.reason = reason;
  if (reason !== 'Other') screening.other_text = '';
  scheduleAutosave();
  renderStepContent();
}

function setProblemQualityRejectOtherText(value) {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const screening = ensureProblemQualityScreening(st);
  screening.other_text = value;
  scheduleAutosave();
}

function setProblemQualityDecision(decision) {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const screening = ensureProblemQualityScreening(st);
  screening.decision = decision;
  if (decision !== 'reject') {
    screening.reason = '';
    screening.other_text = '';
    screening.rejected_at = '';
  }
  scheduleAutosave();
  renderStepContent();
}

async function passProblemQualityCheck() {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const screening = ensureProblemQualityScreening(st);
  screening.decision = 'pass';
  screening.reason = '';
  screening.other_text = '';
  screening.rejected_at = '';
  currentStep = 1;
  st.current_step = 1;
  scheduleAutosave();
  renderStepContent();
}

async function rejectProblemQualityAndSkip() {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const screening = ensureProblemQualityScreening(st);
  if (!screening.reason) {
    alert('请先选择拒绝原因。');
    return;
  }
  if (screening.reason === 'Other' && !String(screening.other_text || '').trim()) {
    alert('选择 Other 时请填写简短说明。');
    return;
  }
  screening.decision = 'reject';
  screening.rejected_at = new Date().toISOString();
  st.active_sample_idx = null;
  st.current_step = 0;
  currentStep = 0;
  await persistProgress('in_progress', true);
  const rejectedCaseId = c.id;
  const nextIdx = currentCaseIndex + 1;
  if (nextIdx < dataset.length) {
    await selectCase(nextIdx);
    showToast(`题目 ${rejectedCaseId} 已按低质量筛除，已自动跳到下一题`, 'success');
    return;
  }
  renderCurrentCase();
  showToast(`题目 ${rejectedCaseId} 已按低质量筛除（当前已是最后一题）`, 'success');
}

function setSampleField(i, k, v) {
  const st = getCaseState(selectedCase().id);
  sampleRecord(st, i)[k] = v;
  scheduleAutosave();
  refreshSampleOverviewPanel();
}

function selectSolution(i) {
  const st = getCaseState(selectedCase().id);
  st.sample_cursor = i;
  setActiveSampleFromCursor();
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

function isLikelySerializedClaimRecord(value) {
  const text = String(value || '').trim();
  return text.startsWith('{') && text.includes('id') && text.includes('text');
}

function claimNeedsSourceRepair(claim, source, sourceClaimsPresent) {
  if (!sourceClaimsPresent) return false;
  if (!source) return !claim.text;
  if (!claim.text) return true;
  if (claim.serialized_like) return true;
  return false;
}

function normalizePresegmentedClaims(rawClaims, sample = {}) {
  const sourceClaims = extractPresegmentedClaims(sample);
  if (!Array.isArray(rawClaims) || !rawClaims.length) return sourceClaims;

  const normalized = rawClaims.map((item, index) => {
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      const stepIdx = Number.isInteger(item.step_idx)
        ? item.step_idx
        : Number.isInteger(item.step_index)
          ? item.step_index
          : null;
      return {
        id: String(item.id || `p${index + 1}`),
        text: String(item.text || item.claim || '').trim(),
        step_idx: stepIdx,
        serialized_like: false,
      };
    }
    if (typeof item === 'string') {
      const text = item.trim();
      return {
        id: `p${index + 1}`,
        text,
        step_idx: null,
        serialized_like: isLikelySerializedClaimRecord(text),
      };
    }
    return {
      id: `p${index + 1}`,
      text: '',
      step_idx: null,
      serialized_like: false,
    };
  });

  const sourceClaimsPresent = sourceClaims.length > 0;
  const emptyTextCount = normalized.filter((claim) => !claim.text).length;
  const serializedCount = normalized.filter((claim) => claim.serialized_like).length;
  const mismatchedCount = sourceClaimsPresent && normalized.length !== sourceClaims.length;
  const needsSourceRepair = sourceClaimsPresent && (
    mismatchedCount
    || serializedCount > 0
    || emptyTextCount > 0
    || normalized.some((claim, index) => claimNeedsSourceRepair(claim, sourceClaims[index], sourceClaimsPresent))
  );

  const repaired = (needsSourceRepair ? (
    mismatchedCount
      ? sourceClaims.map((source, index) => {
        const claim = normalized[index] || {};
        return {
          id: source.id || claim.id || `p${index + 1}`,
          text: String(source.text || '').trim(),
          step_idx: Number.isInteger(claim.step_idx) ? claim.step_idx : (Number.isInteger(source.step_idx) ? source.step_idx : null),
        };
      })
      : Array.from({ length: sourceClaims.length }, (_, index) => {
    const claim = normalized[index] || {};
    const source = sourceClaims[index] || {};
    return {
      id: claim.id || source.id || `p${index + 1}`,
      text: (claim.text && !claim.serialized_like) ? claim.text : String(source.text || '').trim(),
      step_idx: Number.isInteger(claim.step_idx) ? claim.step_idx : (Number.isInteger(source.step_idx) ? source.step_idx : null),
    };
  })
  ) : normalized.map(({ serialized_like, ...claim }) => claim))
    .filter((claim) => claim.text);

  return repaired.length ? repaired : sourceClaims;
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
  const claims = wa.presegmented_claims || [];
  wa.claims = Array.from({ length: stepCount }, (_, i) => ({ step_id: `s${i + 1}`, claims: [] }));
  claims.forEach((claim) => {
    claim.step_idx = -1;
  });
  let prevEnd = -1;
  for (let si = 0; si < stepCount; si += 1) {
    const start = Number(document.getElementById(`stepRangeStart_${si}`)?.value ?? -1);
    const end = Number(document.getElementById(`stepRangeEnd_${si}`)?.value ?? -1);
    if (start < 0 && end < 0) continue;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < 0 || start > end || end >= claims.length) {
      alert(`Step ${si + 1} 的边界无效，请重新选择起止 Claim。`);
      return;
    }
    if (start !== prevEnd + 1) {
      alert(`Step ${si + 1} 必须从 Claim #${prevEnd + 2} 开始，保证按顺序连续分段。`);
      return;
    }
    for (let ci = start; ci <= end; ci += 1) {
      claims[ci].step_idx = si;
      wa.claims[si].claims.push((claims[ci].text || '').trim());
    }
    prevEnd = end;
  }
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
  return `<button class="${current === expected ? `tag active ${expected}` : 'tag'}" onclick="updateClaimCheckAndRender('${expected}', '${claimId}')">${label}</button>`;
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

function getClaimsForLaterStages(wa) {
  const filtered = (wa.claims || []).map((s, si) => ({
    step_id: s.step_id || `s${si + 1}`,
    claims: (s.claims || []).filter((_, ci) => (wa.claim_checks[`s${si + 1}c${ci + 1}`] || '') !== 'delete'),
  }));
  const filteredChecks = {};
  Object.entries(wa.claim_checks || {}).forEach(([claimId, status]) => {
    if (status !== 'delete') filteredChecks[claimId] = status;
  });
  return { claims: filtered, claim_checks: filteredChecks };
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
  const grouped = flattenClaimsByStep(getClaimsForLaterStages(wa).claims);
  if (!grouped.length) return '<div class="card"><h3>请先在 Step 3 完成 claim 整理。</h3></div>';
  st.ui.depStepIdx = Math.max(0, Math.min(grouped.length - 1, st.ui.depStepIdx || 0));
  const targetStep = (wa.steps || [])[st.ui.depStepIdx] || {};
  const targetKey = `s${st.ui.depStepIdx + 1}`;
  wa.step_dependencies = wa.step_dependencies || {};
  const selectedDeps = wa.step_dependencies[targetKey] || [];
  const options = grouped.map((step, i) => `<option value="${i}" ${i === st.ui.depStepIdx ? 'selected' : ''}>Step ${i + 1}</option>`).join('');
  let candidateHtml = '';
  for (let si = 0; si < st.ui.depStepIdx; si += 1) {
    const step = grouped[si];
    if (!step.claims.length) continue;
    candidateHtml += `<details open><summary>Step ${si + 1}</summary>`;
    step.claims.forEach((cand) => {
      const checked = selectedDeps.includes(cand.id) ? 'checked' : '';
      candidateHtml += `<label class="dep-option"><input type="checkbox" ${checked} onchange="updateStepDependency(${st.ui.depStepIdx}, '${cand.id}', this.checked)"> <span>${cand.id}</span>${renderMathPreviewBlock(cand.text, '当前 Claim 为空')}</label>`;
    });
    candidateHtml += '</details>';
  }
  return `
    <h3>Step 5：依赖关系（按 Step 标注）</h3>
    <div class="card">
      <div class="row">
        <label>当前目标 Step
          <select onchange="setDependencyTargetStep(this.value)">${options}</select>
        </label>
        <span class="pill">已选依赖 ${selectedDeps.length}</span>
      </div>
      <h4>当前 Step 内容</h4>
      <div class="curr-claim">${renderMathPreviewBlock(targetStep.text || '', '当前 Step 暂无内容')}</div>
    </div>
    <div class="dep-section">
      <h4>可选前序 claims（仅来自之前的 Step）</h4>
      ${candidateHtml || '<p class="muted-note">当前为第 1 个 Step，没有前序 claims。</p>'}
    </div>
  `;
}

function buildSummaryView() {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const stats = getClaimCheckStats(st);
  const laterStageData = getClaimsForLaterStages(wa);
  return `
    <h3>Step 6：提交前总览</h3>
    <p>请检查以下结果无误后提交：</p>
    <div class="kpi-grid">
      <div class="kpi"><small>Step 数</small><b>${(wa.steps || []).length}</b></div>
      <div class="kpi"><small>Claim 总数</small><b>${stats.total}</b></div>
      <div class="kpi"><small>已检查</small><b>${stats.checked}</b></div>
      <div class="kpi"><small>删除</small><b>${stats.deleted}</b></div>
      <div class="kpi"><small>未检查</small><b>${stats.unchecked}</b></div>
    </div>
    <h4>多采样验证</h4><pre>${JSON.stringify(st.sample_validation, null, 2)}</pre>
    <h4>当前工作流状态</h4><pre>${JSON.stringify(wa.workflow_state || '', null, 2)}</pre>
    <h4>Step切分（当前样本）</h4><pre>${JSON.stringify({ active_sample_idx: st.active_sample_idx, cut_points: wa.cut_points, steps: wa.steps }, null, 2)}</pre>
    <h4>Claim整理结果（按 step，已排除 Delete）</h4><pre>${JSON.stringify(laterStageData.claims, null, 2)}</pre>
    <h4>Claim正确性检查（已排除 Delete）</h4><pre>${JSON.stringify(laterStageData.claim_checks, null, 2)}</pre>
    <h4>依赖关系</h4><pre>${JSON.stringify(wa.dependencies, null, 2)}</pre>
    <h4>Step 依赖关系（简化标注）</h4><pre>${JSON.stringify(wa.step_dependencies || {}, null, 2)}</pre>
    <h4>已完成正确解参考</h4><pre>${JSON.stringify(st.correct_solutions, null, 2)}</pre>
    <button class="primary" onclick="submitCase()">完成当前样本并保存</button>
  `;
}

function buildWorkspaceHeader(c, st) {
  const wa = getWorkingAnnotation(st);
  const stats = getClaimCheckStats(st);
  const totalSamples = (c.samples || []).length;
  const completed = st.sample_validation.filter(x => x?.pipeline_status === 'completed' || x?.pipeline_status === 'discarded').length;
  const activeSample = st.active_sample_idx === null ? '-' : `sample-${st.active_sample_idx + 1}`;
  return `
    <div class="card">
      <div class="row">
        <h3 style="margin:0;">${escapeHtml(c.id)}</h3>
        <span class="pill">samples ${totalSamples}</span>
        <span class="pill">progress ${completed}/${totalSamples}</span>
        <span class="pill">active ${activeSample}</span>
        <span class="pill">steps ${(wa.steps || []).length}</span>
        <span class="pill">claims ${stats.total}</span>
      </div>
    </div>
  `;
}

function buildSampleOverviewPanel(c, st) {
  const samples = Array.isArray(c?.samples) ? c.samples : [];
  const lockOwners = getMethodLockOwners(st);
  const selectedSamples = buildSelectedSampleSummary(st);
  const rows = samples.map((_, i) => {
    const rec = sampleRecord(st, i);
    const method = normalizedMethodName(rec.class_name);
    const lockInfo = getSampleMethodLockInfo(st, i);
    const isSelected = getOptimalSampleEntries(st).some((entry) => entry.sample_idx === i);
    const notes = [];
    if (isSelected) notes.push('已入选最优样本');
    if (lockInfo.locked) {
      const owner = sampleRecord(st, lockInfo.ownerIdx);
      const ownerSummary = normalizedMethodName(owner.summary);
      notes.push(`同方法已由 sample-${lockInfo.ownerIdx + 1} 锁定${ownerSummary ? `：${ownerSummary}` : ''}`);
    } else if (method && lockOwners[method] === i) {
      notes.push('该方法已锁定其他同分类样本');
    }
    if (i === st.active_sample_idx) notes.push('当前活跃样本');
    return `
      <tr class="${i === st.sample_cursor ? 'sample-overview-current' : ''}">
        <td>sample-${i + 1}</td>
        <td>${escapeHtml(getSampleStageLabel(st, i))}</td>
        <td>${method ? escapeHtml(method) : '<span class="muted-note">未填写</span>'}</td>
        <td>${rec.is_new_class ? '是' : '否'}</td>
        <td>${rec.summary ? escapeHtml(rec.summary) : '<span class="muted-note">暂无</span>'}</td>
        <td>${notes.length ? notes.map((note) => `<span class="sample-note">${escapeHtml(note)}</span>`).join('') : '<span class="muted-note">-</span>'}</td>
      </tr>
    `;
  }).join('');

  return `
    <div class="card sample-overview">
      <div class="row">
        <h4 style="margin:0;">当前题目 sample 总览</h4>
        <span class="pill">已选最优样本 ${getOptimalSampleEntries(st).length}</span>
        <span class="pill">完成 ${st.sample_validation.filter((item) => item?.pipeline_status === 'completed').length}</span>
        <span class="pill">丢弃 ${st.sample_validation.filter((item) => item?.pipeline_status === 'discarded').length}</span>
      </div>
      <p class="muted-note">同方法按分类名精确匹配；某个 sample 完成并入选最优后，会锁定其他同分类样本。</p>
      <div class="sample-overview-selected"><strong>已选最优样本：</strong>${escapeHtml(selectedSamples)}</div>
      <div class="context-panel-scroll">
        <table class="compact-table sample-overview-table">
          <thead>
            <tr><th>样本</th><th>进度</th><th>分类</th><th>新分类</th><th>简介</th><th>说明</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  `;
}

function refreshSampleOverviewPanel() {
  if (currentStep !== 1) return;
  const c = selectedCase();
  if (!c) return;
  const mount = document.getElementById('sampleOverviewMount');
  if (!mount) return;
  mount.innerHTML = buildSampleOverviewPanel(c, getCaseState(c.id));
}

function getRawTextForSample(c, idx) {
  const sample = (c.samples || [])[idx] || {};
  return String(
    sample.solution
    || sample.raw_solution
    || sample.generated_solution
    || sample.raw_text
    || sample.input
    || sample.problem
    || c.question
    || '',
  ).trim();
}

function toggleRawTextPanel() {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  st.ui.showRawText = !st.ui.showRawText;
  renderStepContent();
}

function togglePinRawText() {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  st.ui.pinRawText = !st.ui.pinRawText;
  if (st.ui.pinRawText) st.ui.showRawText = true;
  renderStepContent();
}

async function copyCurrentRawText() {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  const raw = getRawTextForSample(c, st.sample_cursor || 0);
  try {
    await copyTextRobust(raw);
    showToast('已复制原始文本', 'success');
  } catch (err) {
    showToast(`复制失败：${err.message}`, 'error');
  }
}

function setDependencyTargetStep(v) {
  const c = selectedCase(); if (!c) return;
  const st = getCaseState(c.id);
  st.ui.depStepIdx = Math.max(0, Number(v) || 0);
  renderStepContent();
}

function updateStepDependency(stepIdx, depId, checked) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  const key = `s${stepIdx + 1}`;
  wa.step_dependencies = wa.step_dependencies || {};
  wa.step_dependencies[key] = wa.step_dependencies[key] || [];
  if (checked) {
    if (!wa.step_dependencies[key].includes(depId)) wa.step_dependencies[key].push(depId);
  } else {
    wa.step_dependencies[key] = wa.step_dependencies[key].filter((x) => x !== depId);
  }
  wa.workflow_state = 'dependencies_labeled';
  scheduleAutosave();
}

function buildStepContextPanel(steps = []) {
  const cards = (steps || []).map((step, i) => `
    <article class="step-context-card">
      <details open>
        <summary>Step ${i + 1}</summary>
        ${renderMathPreviewBlock(step.text || '', '当前 Step 暂无内容')}
      </details>
    </article>
  `).join('') || '<p class="muted-note">尚未生成 Step 内容。</p>';
  return `
    <aside class="context-panel-body">
      <div class="context-panel-head">
        <h4>Step 上下文</h4>
        <span class="pill">共 ${(steps || []).length} 条</span>
      </div>
      <div class="context-panel-scroll">${cards}</div>
    </aside>
  `;
}

function buildClaimPreviewPanel(claims = []) {
  const rows = (claims || []).map((cl, i) => `
    <tr>
      <td>${cl.id || `p${i + 1}`}</td>
      <td>${renderMathPreviewBlock(cl.text || '', '当前 Claim 为空')}</td>
      <td>#${i + 1}</td>
    </tr>
  `).join('');
  return `
    <aside class="context-panel-body">
      <div class="context-panel-head">
        <h4>Claim 顺序预览</h4>
        <span class="pill">共 ${(claims || []).length} 条</span>
      </div>
      <div class="context-panel-scroll">
        <table class="compact-table">
          <thead><tr><th>Claim</th><th>文本</th><th>顺序</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="3">当前 solution 未提供预切分 claim</td></tr>'}</tbody>
        </table>
      </div>
    </aside>
  `;
}

function withContextSplit(mainHtml, contextHtml, panelType = 'step') {
  const c = selectedCase(); if (!c) return mainHtml;
  const st = getCaseState(c.id);
  const compact = shouldStackContextPanels(1);
  const desiredWidth = panelType === 'raw'
    ? Math.max(300, Math.min(680, st.ui.rawPanelWidth || 360))
    : Math.max(300, Math.min(620, st.ui.stepContextWidth || 360));
  const width = compact ? null : clampContextSideWidth(desiredWidth, 1);
  const splitId = panelType === 'raw' ? 'rawSplit' : 'stepSplit';
  const handleId = panelType === 'raw' ? 'rawSplitHandle' : 'stepSplitHandle';
  return `
    <div id="${splitId}" class="context-split${compact ? ' context-compact' : ''}">
      <section class="context-main">${mainHtml}</section>
      <div id="${handleId}" class="inline-resize-handle" aria-hidden="true"></div>
      <section class="context-side"${width ? ` style="width:${width}px"` : ''}>${contextHtml}</section>
    </div>
  `;
}

function withDualContextSplit(mainHtml, middleHtml, rightHtml) {
  const c = selectedCase(); if (!c) return mainHtml;
  const st = getCaseState(c.id);
  const compact = shouldStackContextPanels(2);
  const middleWidth = compact ? null : clampContextSideWidth(Math.max(300, Math.min(640, st.ui.claimPreviewWidth || 360)), 2);
  const rightWidth = compact ? null : clampContextSideWidth(Math.max(300, Math.min(620, st.ui.stepContextWidth || 360)), 2);
  return `
    <div id="stepClaimSplit" class="context-split context-split-3${compact ? ' context-compact' : ''}">
      <section class="context-main">${mainHtml}</section>
      <div id="claimSplitHandle" class="inline-resize-handle" aria-hidden="true"></div>
      <section class="context-side"${middleWidth ? ` style="width:${middleWidth}px"` : ''}>${middleHtml}</section>
      <div id="stepSplitHandleDual" class="inline-resize-handle" aria-hidden="true"></div>
      <section class="context-side"${rightWidth ? ` style="width:${rightWidth}px"` : ''}>${rightHtml}</section>
    </div>
  `;
}

function initInlineResizer(panelType = 'step') {
  const handleId = panelType === 'raw' ? 'rawSplitHandle' : 'stepSplitHandle';
  const splitId = panelType === 'raw' ? 'rawSplit' : 'stepSplit';
  const handle = document.getElementById(handleId);
  const split = document.getElementById(splitId);
  if (!handle || !split) return;
  handle.onmousedown = (event) => {
    event.preventDefault();
    const onMove = (e) => {
      const rect = split.getBoundingClientRect();
      const nextWidth = rect.right - e.clientX;
      const c = selectedCase(); if (!c) return;
      const st = getCaseState(c.id);
      if (panelType === 'raw') st.ui.rawPanelWidth = Math.max(300, Math.min(680, nextWidth));
      else st.ui.stepContextWidth = Math.max(300, Math.min(620, nextWidth));
      const side = split.querySelector('.context-side');
      if (side) side.style.width = `${panelType === 'raw' ? st.ui.rawPanelWidth : st.ui.stepContextWidth}px`;
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };
}

function initDualInlineResizer() {
  const split = document.getElementById('stepClaimSplit');
  const claimHandle = document.getElementById('claimSplitHandle');
  const stepHandle = document.getElementById('stepSplitHandleDual');
  if (!split || !claimHandle || !stepHandle) return;

  const bindHandle = (handle, panelKey, min, max, sideIndex) => {
    handle.onmousedown = (event) => {
      event.preventDefault();
      const onMove = (e) => {
        const rect = split.getBoundingClientRect();
        const nextWidth = rect.right - e.clientX;
        const c = selectedCase(); if (!c) return;
        const st = getCaseState(c.id);
        st.ui[panelKey] = Math.max(min, Math.min(max, nextWidth));
        const side = split.querySelectorAll('.context-side')[sideIndex];
        if (side) side.style.width = `${st.ui[panelKey]}px`;
      };
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    };
  };
  bindHandle(claimHandle, 'claimPreviewWidth', 300, 640, 0);
  bindHandle(stepHandle, 'stepContextWidth', 300, 620, 1);
}

function renderStepContent() {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const root = document.getElementById('stepContent');
  const header = buildWorkspaceHeader(c, st);
  const navButtons = document.querySelectorAll('.step-btn');
  navButtons.forEach((btn) => btn.classList.toggle('active', Number(btn.dataset.step) === currentStep));

  if (currentStep === 0) {
    const screening = ensureProblemQualityScreening(st);
    const passCls = screening.decision === 'pass' ? 'tag active ok' : 'tag';
    const rejectCls = screening.decision === 'reject' ? 'tag active bad' : 'tag';
    const rejectReasons = [
      { key: 'Wrong', label: 'Wrong' },
      { key: 'Contradictory', label: 'Contradictory' },
      { key: 'Too Simple', label: 'Too Simple' },
      { key: 'Ambiguous', label: 'Ambiguous' },
      { key: 'Other', label: 'Other' },
    ];
    const rejectReasonButtons = rejectReasons.map((x) => (
      `<button class="${screening.reason === x.key ? 'tag active' : 'tag'}" onclick="setProblemQualityRejectReason('${x.key}')">${x.label}</button>`
    )).join('');
    root.innerHTML = `
      ${header}
      <h3>Step 0：题目质量筛查（独立前置步骤）</h3>
      <p>该步骤独立于后续标注流程：通过后进入 Step 1；拒绝则自动跳过当前题目。</p>
      <div class="card sample-focus">
        <div class="row">
          <button class="${passCls}" onclick="passProblemQualityCheck()">Pass quality check</button>
          <button class="${rejectCls}" onclick="setProblemQualityDecision('reject')">Reject as low-quality problem</button>
        </div>
        <div class="card">
          <h4 style="margin:0 0 8px;">问题陈述</h4>
          <div class="rendered-math">${renderLatexWithFallback(c.question || '未提供题目')}</div>
        </div>
        <div class="card">
          <h4 style="margin:0 0 8px;">标准/参考答案</h4>
          <div class="rendered-math">${renderLatexWithFallback(c.reference_answer || '未提供标准答案')}</div>
        </div>
        ${screening.decision === 'reject' ? `
          <div class="row">
            <span>拒绝原因：</span>
            ${rejectReasonButtons}
          </div>
          ${screening.reason === 'Other' ? `
            <div class="row">
              <label style="width:100%;">Other 说明
                <input value="${escapeHtml(screening.other_text || '')}" maxlength="200" oninput="setProblemQualityRejectOtherText(this.value)" placeholder="请简要说明其他质量问题（必填）" />
              </label>
            </div>
          ` : ''}
          <div class="row">
            <button class="primary" onclick="rejectProblemQualityAndSkip()">确认拒绝并自动跳过当前题目</button>
          </div>
        ` : ''}
      </div>
    `;
    return;
  }

  if (currentStep === 1) {
    const sampleCount = (c.samples || []).length;
    const i = Math.min(st.sample_cursor || 0, Math.max(0, sampleCount - 1));
    st.sample_cursor = i;
    const s = (c.samples || [])[i] || {};
    const rec = sampleRecord(st, i);
    const lockInfo = getSampleMethodLockInfo(st, i);
    const clsCorrect = rec.is_correct === true ? 'tag active ok' : 'tag';
    const clsWrong = rec.is_correct === false ? 'tag active bad' : 'tag';
    const clsUnset = rec.is_correct === null ? 'tag active' : 'tag';
    let html = `${header}<h3>Step 1：单样本验证入口（严格串行）</h3><p>一次只处理一个样本：判定后进入完整流程，完成后再转到下一样本。</p>`;
    html += `<div id="sampleOverviewMount">${buildSampleOverviewPanel(c, st)}</div>`;
    const rawText = getRawTextForSample(c, i);
    const rawVisible = st.ui.showRawText || st.ui.pinRawText;
    html += `
      <div class="card sample-focus">
        <div class="card-head">
          <h4>sample-${i + 1} / ${sampleCount}</h4>
          <div class="row">
            <span class="pill">状态 ${rec.pipeline_status || 'not_started'}</span>
            <button class="ghost" onclick="toggleRawTextPanel()">${rawVisible ? '隐藏原始文本' : '查看原始文本'}</button>
            <button class="ghost" onclick="togglePinRawText()">${st.ui.pinRawText ? '取消置顶' : '置顶原始文本'}</button>
          </div>
        </div>
        ${renderSolutionCard(s.solution || '', i)}
        <div class="row">
          <button class="${clsCorrect}" ${lockInfo.locked ? `disabled title="${escapeHtml(`同方法已由 sample-${lockInfo.ownerIdx + 1} 入选最优样本`)}"` : ''} onclick="chooseSampleStatus(${i}, true)">正确</button>
          <button class="${clsWrong}" onclick="chooseSampleStatus(${i}, false)">错误</button>
          <button class="${clsUnset}" onclick="chooseSampleStatus(${i}, null)">未判定</button>
        </div>
        ${lockInfo.locked ? `<div class="sample-lock-warning">同方法已由 sample-${lockInfo.ownerIdx + 1} 入选最优样本，当前样本不能再标记为正确。请修改分类或改为错误/未判定。</div>` : ''}
        <div class="row">
          <label>分类 <input value="${rec.class_name || ''}" oninput="setSampleField(${i}, 'class_name', this.value)"></label>
          <label><input type="checkbox" ${rec.is_new_class ? 'checked' : ''} onchange="setSampleField(${i}, 'is_new_class', this.checked)"> 新分类</label>
          <label>新方法概述 <input value="${rec.summary || ''}" oninput="setSampleField(${i}, 'summary', this.value)"></label>
        </div>
        <div class="row">
          <button class="primary" onclick="setActiveSampleFromCursor()">开始当前样本流程</button>
          <button onclick="moveSampleCursor(-1)">上一样本</button>
          <button onclick="moveSampleCursor(1)">下一样本</button>
        </div>
      </div>
    `;
    if (rawVisible) {
      const rawPanel = `
        <aside class="context-panel-body">
          <div class="context-panel-head"><h4>原始文本</h4><button class="ghost" onclick="copyCurrentRawText()">复制</button></div>
          <div class="context-panel-scroll"><pre>${escapeHtml(rawText || '当前样本未提供 solution/raw_solution，已回退显示题目。')}</pre></div>
        </aside>
      `;
      root.innerHTML = withContextSplit(html, rawPanel, 'raw');
      initInlineResizer('raw');
      return;
    }
    root.innerHTML = html;
    return;
  }

  if (st.active_sample_idx === null) {
    root.innerHTML = `${header}<div class="card"><h3>请先在 Step 1 中选择一个判定为“正确”的 sample 作为当前工作样本。</h3></div>`;
    return;
  }

  if (currentStep === 2) {
    root.innerHTML = `${header}
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
    const claimOptions = ['<option value="-1">未设置</option>']
      .concat(claims.map((cl, i) => `<option value="${i}">${cl.id || `p${i + 1}`}</option>`))
      .join('');
    let prevEnd = -1;
    let rangeRows = '';
    for (let si = 0; si < stepCount; si += 1) {
      let start = -1;
      let end = -1;
      for (let ci = 0; ci < claims.length; ci += 1) {
        if ((claims[ci]?.step_idx ?? -1) === si) {
          if (start < 0) start = ci;
          end = ci;
        }
      }
      if (start < 0 && claims.length > 0 && prevEnd + 1 < claims.length) {
        start = prevEnd + 1;
        end = prevEnd + 1;
      }
      if (start >= 0) prevEnd = Math.max(prevEnd, end);
      rangeRows += `
        <tr>
          <td>Step ${si + 1}</td>
          <td><select id="stepRangeStart_${si}">${claimOptions.replace(`value="${start}"`, `value="${start}" selected`)}</select></td>
          <td><select id="stepRangeEnd_${si}">${claimOptions.replace(`value="${end}"`, `value="${end}" selected`)}</select></td>
        </tr>
      `;
    }
    const unassigned = claims.filter(x => !Number.isInteger(x.step_idx) || x.step_idx < 0).length;
    const mainHtml = `
      ${header}
      <h3>Step 3：按顺序为每个 Step 标注 Claim 连续区间（边界选择）</h3>
      <p class="muted-note">Claim 已按原始顺序排列。请仅为每个 Step 选择起始/结束 Claim，系统会自动按连续区间归属。</p>
      <div class="kpi-grid">
        <div class="kpi"><small>Step 数</small><b>${stepCount}</b></div>
        <div class="kpi"><small>预切分 Claim</small><b>${claims.length}</b></div>
        <div class="kpi"><small>未分配</small><b>${unassigned}</b></div>
      </div>
      <table>
        <thead><tr><th>Step</th><th>起始 Claim</th><th>结束 Claim</th></tr></thead>
        <tbody>${rangeRows || '<tr><td colspan="3">请先在 Step 2 生成 Step</td></tr>'}</tbody>
      </table>
      <div class="row">
        <button class="primary" onclick="organizeClaimsBySteps()">按边界保存并生成 Step-Claim 结构</button>
      </div>
      <pre>${JSON.stringify(wa.claims, null, 2)}</pre>
    `;
    root.innerHTML = withDualContextSplit(mainHtml, buildClaimPreviewPanel(claims), buildStepContextPanel(wa.steps || []));
    initDualInlineResizer();
    return;
  }

  if (currentStep === 4) {
    const checkStats = getClaimCheckStats(st);
    let html = `${header}<h3>Step 4：Claim正确性检查与修正</h3>`;
    html += `
      <div class="kpi-grid">
        <div class="kpi"><small>Claim 总数</small><b>${checkStats.total}</b></div>
        <div class="kpi"><small>正确</small><b>${checkStats.correct}</b></div>
        <div class="kpi"><small>错误</small><b>${checkStats.incorrect}</b></div>
        <div class="kpi"><small>删除</small><b>${checkStats.deleted}</b></div>
        <div class="kpi"><small>未检查</small><b>${checkStats.unchecked}</b></div>
      </div>
    `;
    (wa.claims || []).forEach((cs, si) => {
      html += `<h4>Step ${si + 1}</h4>`;
      (cs.claims || []).forEach((claim, ci) => {
        const claimId = `s${si + 1}c${ci + 1}`;
        const current = wa.claim_checks[claimId] || 'unchecked';
        const cardCls = current === 'correct' ? 'claim-card claim-correct'
          : current === 'incorrect' ? 'claim-card claim-incorrect'
            : current === 'delete' ? 'claim-card claim-delete' : 'claim-card';
        html += `
          <div class="card ${cardCls}">
            <div><input class="claim-input" value="${escapeHtml(claim)}" oninput="editClaim(${si}, ${ci}, this.value)"></div>
            <div class="row">
              ${claimCheckTag(claimId, current, 'correct', '正确')}
              ${claimCheckTag(claimId, current, 'incorrect', '修改')}
              ${claimCheckTag(claimId, current, 'delete', '删除')}
            </div>
          </div>
        `;
      });
      html += `<button onclick="addClaim(${si})">+ 添加 claim</button>`;
    });
    root.innerHTML = withContextSplit(html, buildStepContextPanel(wa.steps || []), 'step');
    initInlineResizer('step');
    return;
  }

  if (currentStep === 5) {
    root.innerHTML = `${header}${buildDependencyView()}`;
    return;
  }

  if (currentStep === 6) {
    root.innerHTML = `${header}${buildSummaryView()}`;
  }
}

function updateWorkingSolution(value) {
  const st = getCaseState(selectedCase().id);
  const wa = getWorkingAnnotation(st);
  wa.selected_solution_text = value;
  scheduleAutosave();
}

function moveSampleCursor(delta) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  if (st.active_sample_idx !== null) {
    alert('当前样本流程尚未完成，请先完成或丢弃当前样本。');
    return;
  }
  const total = (c.samples || []).length;
  const next = Math.max(0, Math.min(total - 1, (st.sample_cursor || 0) + delta));
  st.sample_cursor = next;
  st.current_step = 1;
  currentStep = 1;
  renderStepContent();
}

function buildProgressPayload(status = 'in_progress', clientRevision = null) {
  const c = selectedCase();
  if (!c) return null;
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  Object.entries(st.sample_annotations || {}).forEach(([sampleIdx, ann]) => {
    if (!ann || typeof ann !== 'object') return;
    const sample = (c.samples || [])[Number(sampleIdx)] || {};
    ann.presegmented_claims = normalizePresegmentedClaims(ann.presegmented_claims, sample);
  });
  const laterStageData = getClaimsForLaterStages(wa);
  const workflowState = deriveCaseWorkflowState(st);
  return {
    annotator_id: annotatorId(),
    device_id: deviceId,
    case_id: c.id,
    client_revision: clientRevision ?? st.client_revision ?? 0,
    status,
    current_step: st.current_step ?? currentStep,
    current_workflow_state: {
      active_sample_idx: st.active_sample_idx,
      sample_cursor: st.sample_cursor || 0,
      workflow_state: workflowState,
      problem_quality_screening: st.problem_quality_screening || { decision: null, reason: '', other_text: '', rejected_at: '' },
    },
    current_annotations: {
      selected_solution_text: wa.selected_solution_text,
      cut_points: wa.cut_points,
      steps: wa.steps,
      presegmented_claims: normalizePresegmentedClaims(
        wa.presegmented_claims,
        st.active_sample_idx !== null ? ((c.samples || [])[st.active_sample_idx] || {}) : {},
      ),
      claims: laterStageData.claims,
      claim_checks: laterStageData.claim_checks,
      dependencies: wa.dependencies,
      step_dependencies: wa.step_dependencies,
      sample_annotations: st.sample_annotations,
    },
    sample_decisions: st.sample_validation,
    correct_solutions: st.correct_solutions || [],
  };
}

function payloadFingerprint(payload) {
  return JSON.stringify({
    annotator_id: payload.annotator_id,
    device_id: payload.device_id,
    case_id: payload.case_id,
    status: payload.status,
    current_step: payload.current_step,
    current_workflow_state: payload.current_workflow_state,
    current_annotations: payload.current_annotations,
    sample_decisions: payload.sample_decisions,
    correct_solutions: payload.correct_solutions,
  });
}

async function persistProgress(status = 'in_progress', silent = true) {
  const c = selectedCase();
  if (!c) return { ok: false, skipped: true };
  const st = getCaseState(c.id);
  const clientRevision = nextClientRevision(st);
  const payload = buildProgressPayload(status, clientRevision);
  if (!payload) return { ok: false, skipped: true };
  writeDraftCache(c.id, payload);
  const fingerprint = payloadFingerprint(payload);
  if (fingerprint === st.last_saved_fingerprint && status !== 'completed') {
    setSaveState('无变更', 'saved');
    return { ok: true, unchanged: true };
  }
  const requestSeq = ++saveRequestSeq;
  setSaveState('保存中…', 'saving');
  let res;
  let data;
  try {
    res = await fetch('/api/save_progress', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    data = await res.json();
  } catch (error) {
    setSaveState('保存失败（网络）', 'error');
    if (!silent) alert(`保存失败：${error.message || '网络异常'}`);
    return { ok: false, error: error.message || '网络异常' };
  }
  if (requestSeq < st.last_applied_save_seq) return { ok: true, skipped: true };
  st.last_applied_save_seq = requestSeq;
  if (!res.ok) {
    setSaveState('保存失败', 'error');
    if (!silent) alert(data.error || '保存失败');
    return { ok: false, error: data.error || '保存失败' };
  }
  if (data.client_revision) {
    st.client_revision = Math.max(st.client_revision || 0, Number(data.client_revision) || 0);
  }
  st.last_saved_hash = data.content_hash || st.last_saved_hash;
  st.last_saved_fingerprint = fingerprint;
  st.last_saved_at_utc = String(data.updated_at_utc || st.last_saved_at_utc || '');
  const ts = formatUtcToLocal(data.updated_at_utc);
  if (data.ignored_stale) {
    setSaveState(`已忽略旧保存 ${ts}`, 'saved');
  } else if (data.unchanged) {
    setSaveState(`无变更 ${ts}`, 'saved');
  } else {
    setSaveState(`已保存 ${ts}`, 'saved');
  }
  return { ok: true, unchanged: Boolean(data.unchanged), ignored_stale: Boolean(data.ignored_stale) };
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
  return persistProgress('in_progress', true);
}

async function restoreProgress(caseId) {
  if (!caseId) return;
  let st = getCaseState(caseId);
  const cachedDraft = readDraftCache(caseId);
  const params = new URLSearchParams({
    annotator_id: annotatorId(),
    device_id: deviceId,
    case_id: caseId,
  });
  let res;
  let data;
  try {
    res = await fetch(`/api/load_progress?${params.toString()}`);
    data = await res.json();
  } catch (error) {
    if (cachedDraft && hasMeaningfulProgress(cachedDraft.progress)) {
      applyRestoredProgress(caseId, cachedDraft.progress, 'local_draft:error');
      setSaveState('已恢复本地草稿（网络异常）', 'pending');
      showToast(`服务器恢复失败，已保留本地草稿：${error.message || '网络异常'}`, 'error');
      return;
    }
    resetCaseState(caseId);
    currentStep = 0;
    setSaveState('恢复失败', 'error');
    showToast(`恢复失败：${error.message || '网络异常'}`, 'error');
    return;
  }
  if (!res.ok) {
    if (cachedDraft && hasMeaningfulProgress(cachedDraft.progress)) {
      applyRestoredProgress(caseId, cachedDraft.progress, 'local_draft:error');
      setSaveState('已恢复本地草稿（服务器恢复失败）', 'pending');
      showToast(data.error || '服务器恢复失败，已回退到本地草稿', 'error');
      return;
    }
    resetCaseState(caseId);
    currentStep = 0;
    setSaveState('恢复失败', 'error');
    showToast(data.error || '恢复失败', 'error');
    return;
  }
  if (!data.found) {
    if (cachedDraft && hasMeaningfulProgress(cachedDraft.progress)) {
      applyRestoredProgress(caseId, cachedDraft.progress, 'local_draft:not_found');
      setSaveState('已恢复本地草稿（服务器无记录）', 'pending');
      showToast('服务器未发现历史进度，已恢复本地草稿', 'error');
      return;
    }
    resetCaseState(caseId);
    currentStep = 0;
    setSaveState('未发现历史进度', 'pending');
    return;
  }
  const progress = data.progress || {};
  st = applyRestoredProgress(caseId, progress, String(data.source || ''));
  writeDraftCache(caseId, progress);
  st.last_saved_fingerprint = payloadFingerprint(progress);
  st.last_saved_hash = progress.content_hash || st.last_saved_hash;
  st.last_saved_at_utc = String(progress.updated_at_utc || st.last_saved_at_utc || '');
  const mode = String(data.source || '');
  const ts = formatUtcToLocal(progress.updated_at_utc);
  setSaveState(`已恢复 ${ts}`, 'saved');
}

async function saveProgress() {
  const result = await flushAutosave();
  if (result?.ok) {
    alert('已保存当前进度');
    return;
  }
  alert(`保存失败：${result?.error || '请检查网络或服务器状态'}`);
}

async function submitCase() {
  const c = selectedCase(); if (!c) return alert('先选择问题');
  const st = getCaseState(c.id);
  if (st.active_sample_idx !== null) {
    const activeIdx = st.active_sample_idx;
    const sample = (c.samples || [])[activeIdx] || {};
    const wa = getWorkingAnnotation(st);
    const existing = (st.correct_solutions || []).some(x => x.sample_idx === activeIdx);
    if (!existing) {
      st.correct_solutions.push({
        sample_idx: activeIdx,
        solution: sample.solution || wa.selected_solution_text || '',
        completed_at: new Date().toISOString(),
      });
    }
    sampleRecord(st, activeIdx).pipeline_status = 'completed';
    wa.workflow_state = 'completed';
    st.active_sample_idx = null;
    st.sample_cursor = findNextSampleCursor(st, activeIdx + 1);
  }
  const allDone = (c.samples || []).every((_, idx) => {
    const status = sampleRecord(st, idx).pipeline_status;
    return status === 'completed' || status === 'discarded';
  });
  await persistProgress(allDone ? 'completed' : 'in_progress', false);
  currentStep = 1;
  st.current_step = 1;
  renderCurrentCase();
  alert(allDone ? `题目 ${c.id} 所有样本已完成` : `sample-${st.sample_cursor + 1} 已切换到下一样本`);
}

async function copySolutionRaw(sampleIdx) {
  const c = selectedCase();
  const st = getCaseState(c.id);
  const wa = getWorkingAnnotation(st);
  const raw = sampleIdx === null
    ? (wa.selected_solution_text || '')
    : (((c.samples || [])[sampleIdx] || {}).solution || '');
  try {
    await copyTextRobust(raw);
  } catch (err) {
    showToast(`复制失败：${err.message}`, 'error');
    const nodeIdFail = sampleIdx === null ? 'copyStatus_active' : `copyStatus_${sampleIdx}`;
    const statusNodeFail = document.getElementById(nodeIdFail);
    if (statusNodeFail) statusNodeFail.textContent = '复制失败，请手动复制';
    return;
  }
  const nodeId = sampleIdx === null ? 'copyStatus_active' : `copyStatus_${sampleIdx}`;
  const statusNode = document.getElementById(nodeId);
  if (!statusNode) return;
  statusNode.textContent = '已复制';
  showToast('已复制解答文本', 'success');
  clearTimeout(saveBadgeTimer);
  saveBadgeTimer = setTimeout(() => {
    statusNode.textContent = '';
  }, 1200);
}

async function copyReferenceSection(section) {
  const c = selectedCase();
  if (!c) return showToast('未加载题目，无法复制', 'error');
  const raw = section === 'problem' ? (c.question || '') : (c.reference_answer || '');
  try {
    await copyTextRobust(raw);
    showToast(section === 'problem' ? '题目已复制' : '标准答案已复制', 'success');
  } catch (err) {
    showToast(`复制失败：${err.message}`, 'error');
  }
}

function setReferenceTab(tab) {
  referenceTab = tab;
  document.getElementById('tabProblem')?.classList.toggle('active', tab === 'problem');
  document.getElementById('tabSolution')?.classList.toggle('active', tab === 'solution');
  document.getElementById('problemSection')?.classList.toggle('active', tab === 'problem');
  document.getElementById('solutionSection')?.classList.toggle('active', tab === 'solution');
}

function loadLayoutPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(layoutStorageKey) || '{}');
    Object.assign(layoutPrefs, saved || {});
    const left = Number(layoutPrefs.leftWidth);
    const right = Number(layoutPrefs.rightWidth);
    layoutPrefs.leftWidth = Number.isFinite(left) ? left : 280;
    layoutPrefs.rightWidth = Number.isFinite(right) ? right : 380;
    layoutPrefs.leftCollapsed = Boolean(layoutPrefs.leftCollapsed);
    layoutPrefs.rightCollapsed = Boolean(layoutPrefs.rightCollapsed);
  } catch (_) {}
}

function persistLayoutPrefs() {
  localStorage.setItem(layoutStorageKey, JSON.stringify(layoutPrefs));
}

function applyLayoutPrefs() {
  const layout = document.getElementById('workspaceLayout');
  if (!layout) return;
  const viewport = window.innerWidth || 1440;
  const left = Math.max(240, Math.min(360, layoutPrefs.leftWidth || 280));
  const right = Math.max(320, Math.min(700, layoutPrefs.rightWidth || 380));
  const reserved = (layoutPrefs.leftCollapsed ? 28 : left) + (layoutPrefs.rightCollapsed ? 28 : right) + 20;
  const minCenter = 540;
  if (viewport - reserved < minCenter) {
    layoutPrefs.leftCollapsed = true;
    layoutPrefs.rightCollapsed = false;
  }
  const leftCol = layoutPrefs.leftCollapsed ? '0px' : `${left}px`;
  const leftRestoreCol = layoutPrefs.leftCollapsed ? '28px' : '0px';
  const rightCol = layoutPrefs.rightCollapsed ? '0px' : `${right}px`;
  const rightRestoreCol = layoutPrefs.rightCollapsed ? '28px' : '0px';
  layout.style.gridTemplateColumns = `${leftCol} 10px ${leftRestoreCol} minmax(0, 1fr) ${rightRestoreCol} 10px ${rightCol}`;
  const casePanel = document.getElementById('casePanel');
  const refPanel = document.getElementById('referencePanel');
  const leftHandle = document.getElementById('leftResizeHandle');
  const rightHandle = document.getElementById('rightResizeHandle');
  const leftRestoreBtn = document.getElementById('leftRestoreBtn');
  const rightRestoreBtn = document.getElementById('rightRestoreBtn');
  casePanel.classList.toggle('panel-collapsed', !!layoutPrefs.leftCollapsed);
  refPanel.classList.toggle('panel-collapsed', !!layoutPrefs.rightCollapsed);
  casePanel.classList.toggle('is-collapsed', !!layoutPrefs.leftCollapsed);
  refPanel.classList.toggle('is-collapsed', !!layoutPrefs.rightCollapsed);
  leftRestoreBtn?.classList.toggle('is-collapsed', !layoutPrefs.leftCollapsed);
  rightRestoreBtn?.classList.toggle('is-collapsed', !layoutPrefs.rightCollapsed);
  leftHandle.classList.toggle('is-collapsed', !!layoutPrefs.leftCollapsed);
  rightHandle.classList.toggle('is-collapsed', !!layoutPrefs.rightCollapsed);
  const leftBtn = document.getElementById('toggleLeftPanel');
  const rightBtn = document.getElementById('toggleRightPanel');
  if (leftBtn) leftBtn.textContent = layoutPrefs.leftCollapsed ? '任务' : '◧';
  if (rightBtn) rightBtn.textContent = layoutPrefs.rightCollapsed ? '参考' : '◨';
}

function togglePanel(side) {
  if (side === 'left') layoutPrefs.leftCollapsed = !layoutPrefs.leftCollapsed;
  if (side === 'right') layoutPrefs.rightCollapsed = !layoutPrefs.rightCollapsed;
  applyLayoutPrefs();
  persistLayoutPrefs();
}

function bindResize(handleId, side) {
  const handle = document.getElementById(handleId);
  if (!handle) return;
  handle.addEventListener('mousedown', (event) => {
    event.preventDefault();
    handle.classList.add('dragging');
    const onMove = (e) => {
      const total = window.innerWidth;
      if (side === 'left') layoutPrefs.leftWidth = Math.max(240, Math.min(360, e.clientX - 20));
      if (side === 'right') layoutPrefs.rightWidth = Math.max(320, Math.min(700, total - e.clientX - 20));
      applyLayoutPrefs();
    };
    const onUp = () => {
      handle.classList.remove('dragging');
      persistLayoutPrefs();
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });
}

function initLayoutControls() {
  loadLayoutPrefs();
  if ((window.innerWidth || 0) < 1360) {
    layoutPrefs.leftCollapsed = true;
  }
  applyLayoutPrefs();
  bindResize('leftResizeHandle', 'left');
  bindResize('rightResizeHandle', 'right');
  document.getElementById('toggleLeftPanel')?.addEventListener('click', () => {
    togglePanel('left');
  });
  document.getElementById('toggleRightPanel')?.addEventListener('click', () => {
    togglePanel('right');
  });
}

window.addEventListener('beforeunload', () => {
  const c = selectedCase();
  if (!c) return;
  const st = getCaseState(c.id);
  const payload = buildProgressPayload('in_progress', nextClientRevision(st));
  if (!payload) return;
  writeDraftCache(c.id, payload);
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  navigator.sendBeacon('/api/save_progress', blob);
});

document.getElementById('annotator').addEventListener('change', async () => {
  const c = selectedCase();
  if (c) {
    await restoreProgress(c.id);
    renderCurrentCase();
  }
});

initDeviceId();
initLayoutControls();
setReferenceTab('problem');
setSaveState('未保存', 'pending');
