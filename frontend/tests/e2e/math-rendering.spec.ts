import { expect, test } from '@playwright/test';
import { datasetPath, generateStepPreview, loadFixtureDataset, openCase, passQualityCheck, startFirstCorrectSample } from './helpers';

test.describe('annotator math rendering and claim visibility', () => {
  test('keeps claim preview visible when one claim contains broken latex', async ({ page }) => {
    const annotatorId = `ann-math-${Date.now()}`;
    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));

    await loadFixtureDataset(page, annotatorId);
    await openCase(page, 'case-bad-latex');
    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);

    await page.getByRole('button', { name: '3 Claim整理' }).click();
    await expect(page.getByText('Claim 顺序预览')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(5);
    await expect(page.getByText('We are given')).toBeVisible();
    await expect(page.getByText('Broken inline formula')).toBeVisible();
    await expect(pageErrors).toHaveLength(0);
  });

  test('shows explicit empty-state text when a sample has no pre-segmented claims', async ({ page }) => {
    await loadFixtureDataset(page, `ann-empty-${Date.now()}`);
    await openCase(page, 'case-empty-claims');
    await passQualityCheck(page);
    await startFirstCorrectSample(page);
    await generateStepPreview(page);

    await page.getByRole('button', { name: '3 Claim整理' }).click();
    await expect(page.getByText('当前 solution 未提供预切分 claim')).toBeVisible();
  });

  test('restores directly into dependency view without losing broken-claim text', async ({ page, request }) => {
    const annotatorId = `ann-deps-${Date.now()}`;
    const payload = {
      annotator_id: annotatorId,
      device_id: 'browser',
      case_id: 'case-valid',
      status: 'in_progress',
      current_step: 5,
      current_workflow_state: {
        active_sample_idx: 0,
        sample_cursor: 0,
        workflow_state: 'dependencies_labeled',
        problem_quality_screening: { decision: 'pass' },
      },
      current_annotations: {
        sample_annotations: {
          '0': {
            selected_solution_text: 'Recovered sample with broken latex $x^{2',
            cut_points: [12],
            steps: [
              { id: 's1', text: 'Recovered step $x^{2' },
              { id: 's2', text: 'Next step with \\badcommand{y}' },
            ],
            presegmented_claims: [
              { text: 'Safe claim' },
              { text: 'Broken inline formula: $x^{2' },
              { text: 'Bad command: \\badcommand{y}' },
            ],
            claims: [
              { step_id: 's1', claims: ['Safe claim', 'Broken inline formula: $x^{2'] },
              { step_id: 's2', claims: ['Bad command: \\badcommand{y}'] },
            ],
            claim_checks: {},
            dependencies: {},
            step_dependencies: { s2: ['s1c1'] },
            workflow_state: 'dependencies_labeled',
            updated_at_utc: '2026-04-20T00:00:00Z',
          },
        },
      },
      sample_decisions: [{ is_correct: true, pipeline_status: 'in_progress' }],
      correct_solutions: [],
    };

    const save = await request.post('/api/save_progress', { data: payload });
    expect(save.ok()).toBeTruthy();

    const pageErrors: Error[] = [];
    page.on('pageerror', (error) => pageErrors.push(error));
    await page.addInitScript(() => {
      window.localStorage.setItem('annotation_device_id', 'browser');
    });
    await page.goto('/annotator');
    await page.locator('#annotator').fill(annotatorId);
    await page.locator('#jsonlPath').fill(datasetPath);
    await page.getByRole('button', { name: '加载数据' }).click();

    await expect(page.getByText('Step 5：依赖关系（按 Step 标注）')).toBeVisible();
    await expect(page.getByText('当前 Step 暂无内容')).toBeVisible();
    await page.getByRole('combobox', { name: '当前目标 Step' }).selectOption({ label: 'Step 2' });
    await expect(page.getByText('Safe claim')).toBeVisible();
    await expect(page.getByText('Broken inline formula')).toBeVisible();
    await expect(pageErrors).toHaveLength(0);
  });
});
