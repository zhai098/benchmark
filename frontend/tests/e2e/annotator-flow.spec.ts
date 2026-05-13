import fs from 'fs/promises';
import path from 'path';
import { expect, test } from '@playwright/test';
import {
  draftCacheKey,
  datasetPath,
  e2eDataDir,
  generateStepPreview,
  loadFixtureDataset,
  openCase,
  passQualityCheck,
  startFirstCorrectSample,
} from './helpers';

test.describe('annotator flow', () => {
  test('restores progress after reload for a partially completed case', async ({ page }) => {
    const annotatorId = `ann-flow-${Date.now()}`;
    await loadFixtureDataset(page, annotatorId);
    await openCase(page, 'case-valid');

    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);

    await page.getByRole('button', { name: '3 Claim整理' }).click();
    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.getByText('Claim 顺序预览')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(3);

    await page.getByRole('button', { name: '手动保存' }).click();
    await expect(page.locator('#saveState')).toContainText(/已保存|保存中|待保存/);

    await page.evaluate((key) => window.localStorage.removeItem(key), draftCacheKey(annotatorId, 'case-valid'));

    await page.reload();
    await page.locator('#annotator').fill(annotatorId);
    await page.locator('#jsonlPath').fill(datasetPath);
    await page.getByRole('button', { name: '加载数据' }).click();
    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(3);
    await expect(page.locator('#saveState')).toContainText('已恢复');
  });

  test('blocks jumping ahead before quality check and before selecting a correct sample', async ({ page }) => {
    page.once('dialog', (dialog) => dialog.dismiss());
    await loadFixtureDataset(page, `ann-gate-${Date.now()}`);
    await openCase(page, 'case-valid');

    await page.getByRole('button', { name: '3 Claim整理' }).click();
    await expect(page.getByText('Step 0：题目质量筛查')).toBeVisible();

    await passQualityCheck(page);
    await page.getByRole('button', { name: '开始当前样本流程' }).click();
    await expect(page.getByText('Step 1：单样本验证入口（严格串行）')).toBeVisible();
  });

  test('requires a reject reason before skipping and moves to the next case after confirmation', async ({ page }) => {
    await loadFixtureDataset(page, `ann-reject-${Date.now()}`);
    await openCase(page, 'case-valid');

    await page.getByRole('button', { name: 'Reject as low-quality problem' }).click();
    let missingReasonMessage = '';
    page.once('dialog', async (dialog) => {
      missingReasonMessage = dialog.message();
      await dialog.dismiss();
    });
    await page.getByRole('button', { name: '确认拒绝并自动跳过当前题目' }).click();
    await expect.poll(() => missingReasonMessage).toContain('请先选择拒绝原因');

    await page.getByRole('button', { name: 'Other' }).click();
    let missingOtherMessage = '';
    page.once('dialog', async (dialog) => {
      missingOtherMessage = dialog.message();
      await dialog.dismiss();
    });
    await page.getByRole('button', { name: '确认拒绝并自动跳过当前题目' }).click();
    await expect.poll(() => missingOtherMessage).toContain('选择 Other 时请填写简短说明');

    await page.getByPlaceholder('请简要说明其他质量问题（必填）').fill('synthetic test rejection');
    await page.getByRole('button', { name: '确认拒绝并自动跳过当前题目' }).click();
    await expect(page.getByRole('heading', { name: 'case-bad-latex' })).toBeVisible();
    await expect(page.getByText('题目 case-valid 已按低质量筛除')).toBeVisible();
    await expect(page.getByText('Step 0：题目质量筛查')).toBeVisible();
  });

  test('advances the sample cursor after marking the current sample wrong', async ({ page }) => {
    await loadFixtureDataset(page, `ann-sample-nav-${Date.now()}`);
    await openCase(page, 'case-valid');
    await passQualityCheck(page);

    await expect(page.getByText('sample-1 / 2')).toBeVisible();
    await page.getByRole('button', { name: '错误' }).click();
    await expect(page.getByText('sample-2 / 2')).toBeVisible();
    await expect(page.getByText('Step 1：单样本验证入口（严格串行）')).toBeVisible();
  });

  test('rejects invalid step-to-claim ranges and keeps step 3 visible', async ({ page }) => {
    await loadFixtureDataset(page, `ann-step-range-${Date.now()}`);
    await openCase(page, 'case-valid');
    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);
    await page.getByRole('button', { name: '3 Claim整理' }).click();

    await page.locator('#stepRangeStart_0').selectOption('1');
    await page.locator('#stepRangeEnd_0').selectOption('0');
    let invalidRangeMessage = '';
    page.once('dialog', async (dialog) => {
      invalidRangeMessage = dialog.message();
      await dialog.dismiss();
    });
    await page.getByRole('button', { name: '按边界保存并生成 Step-Claim 结构' }).click();
    await expect.poll(() => invalidRangeMessage).toContain('边界无效');

    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(3);
  });

  test('shows the no-prior-claims message for the first dependency step', async ({ page }) => {
    await loadFixtureDataset(page, `ann-dep-empty-${Date.now()}`);
    await openCase(page, 'case-valid');
    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);
    await page.getByRole('button', { name: '3 Claim整理' }).click();
    await page.getByRole('button', { name: '按边界保存并生成 Step-Claim 结构' }).click();

    await page.getByRole('button', { name: '5 依赖关系' }).click();
    await expect(page.getByText('Step 5：依赖关系（按 Step 标注）')).toBeVisible();
    await expect(page.getByText('当前为第 1 个 Step，没有前序 claims。')).toBeVisible();
  });

  test('restores a local draft when the server has no saved progress for the case', async ({ page }) => {
    const annotatorId = `ann-local-draft-${Date.now()}`;
    const draftEnvelope = {
      schema_version: 1,
      annotator_id: annotatorId,
      device_id: 'browser',
      case_id: 'case-valid',
      cached_at_utc: '2026-04-21T00:00:00Z',
      progress: {
        annotator_id: annotatorId,
        device_id: 'browser',
        case_id: 'case-valid',
        client_revision: 42,
        status: 'in_progress',
        current_step: 3,
        current_workflow_state: {
          active_sample_idx: 0,
          sample_cursor: 0,
          workflow_state: 'claims_assigned',
          problem_quality_screening: { decision: 'pass' },
        },
        current_annotations: {
          selected_solution_text: 'Locally cached solution',
          cut_points: [12],
          steps: [{ id: 's1', text: 'Cached step one' }],
          presegmented_claims: [
            { id: 'p1', text: 'Cached claim A' },
            { id: 'p2', text: 'Cached claim B' },
          ],
          claims: [{ step_id: 's1', claims: ['Cached claim A', 'Cached claim B'] }],
          claim_checks: {},
          dependencies: {},
          step_dependencies: {},
          sample_annotations: {
            '0': {
              selected_solution_text: 'Locally cached solution',
              cut_points: [12],
              steps: [{ id: 's1', text: 'Cached step one' }],
              presegmented_claims: [
                { id: 'p1', text: 'Cached claim A' },
                { id: 'p2', text: 'Cached claim B' },
              ],
              claims: [{ step_id: 's1', claims: ['Cached claim A', 'Cached claim B'] }],
              claim_checks: {},
              dependencies: {},
              step_dependencies: {},
              workflow_state: 'claims_assigned',
            },
          },
        },
        sample_decisions: [{ is_correct: true, pipeline_status: 'in_progress' }],
        correct_solutions: [],
      },
    };

    await page.goto('/annotator');
    await page.locator('#annotator').fill(annotatorId);
    await page.evaluate(([key, value]) => window.localStorage.setItem(key, value), [
      draftCacheKey(annotatorId, 'case-valid'),
      JSON.stringify(draftEnvelope),
    ]);
    await page.locator('#jsonlPath').fill(datasetPath);
    await page.getByRole('button', { name: '加载数据' }).click();

    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.getByText('Cached claim A', { exact: true })).toBeVisible();
    await expect(page.locator('#saveState')).toContainText('已恢复本地草稿（服务器无记录）');
  });

  test('falls back to a local draft when the server-side record is unreadable', async ({ page }) => {
    const annotatorId = `ann-broken-restore-${Date.now()}`;
    const deviceId = 'browser-draft';
    const caseId = 'case-valid';
    const detailDir = path.join(e2eDataDir, 'annotations', annotatorId, deviceId);
    await fs.mkdir(detailDir, { recursive: true });
    await fs.writeFile(path.join(detailDir, `${caseId}.summary.json`), '{bad json', 'utf-8');
    await fs.writeFile(path.join(detailDir, `${caseId}.detail.json`), '{bad json', 'utf-8');

    const draftEnvelope = {
      schema_version: 1,
      annotator_id: annotatorId,
      device_id: deviceId,
      case_id: caseId,
      cached_at_utc: '2026-04-21T00:00:00Z',
      progress: {
        annotator_id: annotatorId,
        device_id: deviceId,
        case_id: caseId,
        client_revision: 50,
        status: 'in_progress',
        current_step: 3,
        current_workflow_state: {
          active_sample_idx: 0,
          sample_cursor: 0,
          workflow_state: 'claims_assigned',
          problem_quality_screening: { decision: 'pass' },
        },
        current_annotations: {
          selected_solution_text: 'Draft after broken restore',
          cut_points: [],
          steps: [{ id: 's1', text: 'Draft step' }],
          presegmented_claims: [{ id: 'p1', text: 'Draft claim after broken restore' }],
          claims: [{ step_id: 's1', claims: ['Draft claim after broken restore'] }],
          claim_checks: {},
          dependencies: {},
          step_dependencies: {},
          sample_annotations: {
            '0': {
              selected_solution_text: 'Draft after broken restore',
              cut_points: [],
              steps: [{ id: 's1', text: 'Draft step' }],
              presegmented_claims: [{ id: 'p1', text: 'Draft claim after broken restore' }],
              claims: [{ step_id: 's1', claims: ['Draft claim after broken restore'] }],
              claim_checks: {},
              dependencies: {},
              step_dependencies: {},
              workflow_state: 'claims_assigned',
            },
          },
        },
        sample_decisions: [{ is_correct: true, pipeline_status: 'in_progress' }],
        correct_solutions: [],
      },
    };

    await page.addInitScript((savedDeviceId) => {
      window.localStorage.setItem('annotation_device_id', savedDeviceId);
    }, deviceId);
    await page.goto('/annotator');
    await page.locator('#annotator').fill(annotatorId);
    await page.evaluate(([key, value]) => window.localStorage.setItem(key, value), [
      draftCacheKey(annotatorId, caseId),
      JSON.stringify(draftEnvelope),
    ]);
    await page.locator('#jsonlPath').fill(datasetPath);
    await page.getByRole('button', { name: '加载数据' }).click();

    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.getByText('Draft claim after broken restore', { exact: true })).toBeVisible();
    await expect(page.locator('#saveState')).toContainText('已恢复本地草稿（服务器恢复失败）');
  });

  test('repairs corrupted restored presegmented claims from the original sample source', async ({ page }) => {
    const annotatorId = `ann-corrupt-claims-${Date.now()}`;
    const deviceId = 'browser-corrupt';
    const caseId = 'case-valid';
    const detailDir = path.join(e2eDataDir, 'annotations', annotatorId, deviceId);
    await fs.mkdir(detailDir, { recursive: true });

    const detailPayload = {
      schema_version: 2,
      annotator_id: annotatorId,
      device_id: deviceId,
      case_id: caseId,
      client_revision: 77,
      status: 'in_progress',
      current_step: 3,
      current_workflow_state: {
        active_sample_idx: 0,
        sample_cursor: 0,
        workflow_state: 'claims_assigned',
        problem_quality_screening: { decision: 'pass' },
      },
      sample_decisions: [{ is_correct: true, pipeline_status: 'in_progress' }],
      correct_solutions: [],
      sample_annotations: {
        '0': {
          selected_solution_text: 'Recovered broken solution',
          cut_points: [],
          steps: [{ id: 's1', text: 'Recovered broken step' }],
          presegmented_claims: [
            "{'id': 'p1', 'text': 'The equation $x+y=3$ is given.', 'step_idx': None}",
            '',
            "{'id': 'p3', 'text': 'Therefore the value is $3$.', 'step_idx': None}",
          ],
          claims: [],
          claim_checks: {},
          dependencies: {},
          step_dependencies: {},
          workflow_state: 'claims_assigned',
        },
      },
      created_at_utc: '2026-04-22T00:00:00Z',
      updated_at_utc: '2026-04-22T00:00:00Z',
      detail_content_hash: 'corrupt-test',
    };

    await fs.writeFile(path.join(detailDir, `${caseId}.detail.json`), JSON.stringify(detailPayload), 'utf-8');

    await page.addInitScript((savedDeviceId) => {
      window.localStorage.setItem('annotation_device_id', savedDeviceId);
    }, deviceId);
    await loadFixtureDataset(page, annotatorId);
    await openCase(page, caseId);

    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(3);
    await expect(page.getByText('The equation', { exact: false })).toBeVisible();
    await expect(page.getByText('The requested value is', { exact: false })).toBeVisible();
    await expect(page.getByText('Therefore the value is', { exact: false })).toBeVisible();
    await expect(page.getByText('当前 Claim 为空')).toHaveCount(0);
  });

  test('surfaces save failures without crashing the annotator flow', async ({ page }) => {
    await page.route('**/api/save_progress', async (route) => {
      await route.abort('failed');
    });

    await loadFixtureDataset(page, `ann-save-fail-${Date.now()}`);
    await openCase(page, 'case-valid');
    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);

    let saveFailureMessage = '';
    page.once('dialog', async (dialog) => {
      saveFailureMessage = dialog.message();
      await dialog.dismiss();
    });
    await page.getByRole('button', { name: '手动保存' }).click();
    await expect.poll(() => saveFailureMessage).toContain('保存失败');
    await expect(page.locator('#saveState')).toContainText('保存失败（网络）');
    await expect(page.getByText('Step 2：Step切分')).toBeVisible();
  });
});
