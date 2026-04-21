import { expect, test } from '@playwright/test';
import fs from 'fs/promises';
import path from 'path';
import { e2eDataDir, loginReviewer } from './helpers';

async function seedProgressRecord(request: import('@playwright/test').APIRequestContext, caseId: string) {
  const payload = {
    annotator_id: 'reviewer-seed',
    device_id: 'browser',
    case_id: caseId,
    status: 'completed',
    current_step: 6,
    current_workflow_state: {
      active_sample_idx: null,
      sample_cursor: 0,
      workflow_state: 'completed',
      problem_quality_screening: { decision: 'pass' },
    },
    current_annotations: {
      sample_annotations: {
        '0': {
          selected_solution_text: 'A complete solution with $x^2+y^2$.',
          steps: ['Step one', 'Step two'],
          claims: [
            { step_id: 's1', claims: ['Claim A', 'Claim B'] },
            { step_id: 's2', claims: ['Claim C'] },
          ],
          dependencies: { c1: ['c0'] },
          workflow_state: 'completed',
        },
      },
    },
    sample_decisions: [{ is_correct: true, pipeline_status: 'completed', summary: 'ok' }],
    correct_solutions: [{ sample_idx: 0, solution: 'accepted', completed_at: '2026-04-20T00:00:00Z' }],
  };
  const res = await request.post('/api/save_progress', { data: payload });
  expect(res.ok()).toBeTruthy();
}

test.describe('reviewer panel', () => {
  test('lists summary stats and opens detail payloads', async ({ page, request }) => {
    await seedProgressRecord(request, 'review-case-visible');
    await loginReviewer(page);
    await expect(page.getByText('记录数:')).toBeVisible();

    const row = page.locator('#recordsTable tbody tr').filter({ hasText: 'review-case-visible' }).first();
    await expect(row).toBeVisible();
    await expect(row).toContainText('1/1');
    await expect(row).toContainText('3');
    await row.click();
    await expect(page.locator('#detail')).toContainText('review-case-visible');
    await expect(page.locator('#detail')).toContainText('"summary"');
    await expect(page.locator('#detail')).toContainText('"detail"');
  });

  test('falls back from detail-only records and surfaces broken records without killing the page', async ({ page, request }) => {
    const detailDir = path.join(e2eDataDir, 'annotations', 'fallback-ann', 'fallback-device');
    const brokenDir = path.join(e2eDataDir, 'annotations', 'broken-ann', 'broken-device');
    await fs.mkdir(detailDir, { recursive: true });
    await fs.mkdir(brokenDir, { recursive: true });

    const detailOnlyPayload = {
      schema_version: 2,
      annotator_id: 'fallback-ann',
      device_id: 'fallback-device',
      case_id: 'detail-only-case',
      status: 'in_progress',
      created_at_utc: '2026-04-20T00:00:00Z',
      updated_at_utc: '2026-04-20T00:00:00Z',
      current_workflow_state: { workflow_state: 'claims_assigned', active_sample_idx: 0, sample_cursor: 0 },
      sample_decisions: [{ is_correct: true, pipeline_status: 'in_progress' }],
      correct_solutions: [],
      sample_annotations: {
        '0': {
          selected_solution_text: 'Detail-only solution',
          steps: ['Only step'],
          claims: [{ step_id: 's1', claims: ['Only claim'] }],
          dependencies: {},
          workflow_state: 'claims_assigned',
        },
      },
    };
    await fs.writeFile(
      path.join(detailDir, 'detail-only-case.detail.json'),
      JSON.stringify(detailOnlyPayload, null, 2),
      'utf-8',
    );
    await fs.writeFile(
      path.join(brokenDir, 'broken-case.summary.json'),
      '{bad json',
      'utf-8',
    );

    await loginReviewer(page);
    const fallbackRow = page.locator('#recordsTable tbody tr').filter({ hasText: 'detail-only-case' }).first();
    await expect(fallbackRow).toBeVisible();
    await expect(fallbackRow).toContainText('1');
    await expect(fallbackRow).toContainText('claims_assigned');

    const brokenRow = page.locator('#recordsTable tbody tr').filter({ hasText: 'broken-case' }).first();
    await expect(brokenRow).toContainText('异常:');
    await brokenRow.click();
    await expect(page.locator('#detail')).toContainText('"summary": null');
  });

  test('edits guideline content and keeps it after reload', async ({ page }) => {
    await loginReviewer(page);
    const editor = page.locator('#guidelineEditor');
    const updated = `# Reviewer E2E ${Date.now()}\n- keep me`;
    await editor.fill(updated);

    const saved = page.waitForEvent('dialog');
    await page.getByRole('button', { name: '保存说明' }).click();
    const savedDialog = await saved;
    expect(savedDialog.message()).toContain('说明已保存并生效');
    await savedDialog.dismiss();

    await page.reload();
    await expect(page.locator('#guidelineEditor')).toHaveValue(`${updated}\n`);
  });
});
