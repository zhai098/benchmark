import { expect, Page } from '@playwright/test';
import path from 'path';

export const datasetPath = path.resolve(
  __dirname,
  '../../../annotation_app/tests/fixtures/annotator_e2e_dataset.jsonl',
);
export const e2eDataDir = path.resolve(__dirname, '../../../.tmp/annotation-app-e2e-data');

export async function loadFixtureDataset(page: Page, annotatorId: string) {
  await page.goto('/annotator');
  await page.locator('#annotator').fill(annotatorId);
  await page.locator('#jsonlPath').fill(datasetPath);
  await page.getByRole('button', { name: '加载数据' }).click();
  await expect(page.getByRole('heading', { name: 'case-valid' })).toBeVisible();
}

export async function openCase(page: Page, caseId: string) {
  const currentHeading = page.getByRole('heading', { name: caseId });
  if (await currentHeading.isVisible().catch(() => false)) return;
  await page.evaluate((targetCaseId) => {
    const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>('#caseList button'));
    const target = buttons.find((button) => (button.textContent || '').trim() === targetCaseId);
    if (!target) throw new Error(`Case button not found: ${targetCaseId}`);
    target.click();
  }, caseId);
  await expect(page.getByRole('heading', { name: caseId })).toBeVisible();
}

export async function passQualityCheck(page: Page) {
  await page.getByRole('button', { name: 'Pass quality check' }).click();
  await expect(page.getByText('Step 1：单样本验证入口（严格串行）')).toBeVisible();
}

export async function startFirstCorrectSample(page: Page) {
  await page.getByRole('button', { name: '正确' }).click();
  await page.getByRole('button', { name: '开始当前样本流程' }).click();
  await expect(page.getByText('Step 2：Step切分')).toBeVisible();
}

export async function generateStepPreview(page: Page) {
  await page.getByRole('button', { name: '刷新预览' }).click();
  await expect(page.locator('#splitPreview')).toContainText('s1');
}

export async function loginReviewer(page: Page, accessKey = 'reviewer') {
  await page.goto('/annotator');
  const res = await page.evaluate(async (key) => {
    const response = await fetch('/api/session/role', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: 'reviewer', access_key: key }),
    });
    return { ok: response.ok, status: response.status, body: await response.json() };
  }, accessKey);
  if (!res.ok) {
    throw new Error(`Reviewer login failed: ${res.status} ${JSON.stringify(res.body)}`);
  }
  await page.goto('/review');
  await expect(page).toHaveURL(/\/review$/);
}
