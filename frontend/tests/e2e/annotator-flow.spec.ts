import { expect, test } from '@playwright/test';
import {
  datasetPath,
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

    await page.reload();
    await page.locator('#annotator').fill(annotatorId);
    await page.locator('#jsonlPath').fill(datasetPath);
    await page.getByRole('button', { name: '加载数据' }).click();
    await expect(page.getByText('Step 3：按顺序为每个 Step 标注 Claim 连续区间')).toBeVisible();
    await expect(page.locator('.compact-table tbody tr')).toHaveCount(3);
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
});
