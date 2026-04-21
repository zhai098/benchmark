# Annotation Web Manual Checklist

## Goal
Use this checklist for failure injection and edge cases that are either hard to automate or worth validating by hand after large workflow changes.

Each item should be marked with:
- `pass`: behavior matches the expectation below
- `fail`: user-visible regression or data loss
- `n/a`: not relevant to the current change

## Annotator: lifecycle and recovery
- Load a normal JSONL file, enter one case, and complete Step 0 through Step 6 once end to end.
  - Expected: every step renders, no blank panels, and submit returns to Step 1 for the next sample or next case state.
- Reload the page after each individual step.
  - Expected: the same case, same active sample, and same current step are restored.
- Close the tab after making edits but before clicking `手动保存`, then reopen and reload the dataset.
  - Expected: autosave or unload save restores the latest persisted state; no corrupted record is created.
- Restart the backend while a case is mid-progress, then reload the page.
  - Expected: before the backend returns, the UI shows failure without hanging forever; after restart, progress can be restored.
- Change the browser profile or clear `localStorage`, then reopen the same dataset.
  - Expected: old progress is not silently loaded under a different `device_id`.

## Annotator: workflow guards and wrong operations
- Before Step 0 passes, click Step 2-6 repeatedly.
  - Expected: the page stays on Step 0 or explicitly returns there.
- In Step 0, choose `Reject as low-quality problem` and click confirm without a reason.
  - Expected: alert asks for a reject reason; no case switch happens.
- In Step 0, choose `Other` and click confirm without filling the free-text box.
  - Expected: alert asks for an explanation; no case switch happens.
- In Step 0, reject the current case with a valid reason.
  - Expected: the next case opens automatically and the toast explains what happened.
- In Step 1, do not mark the sample `正确`, then click `开始当前样本流程`.
  - Expected: the UI blocks the transition to Step 2.
- In Step 1, mark the current sample `错误`.
  - Expected: the sample becomes discarded and the cursor advances to the next unfinished sample.
- While one sample is active, try to switch to another sample or another case quickly.
  - Expected: the app does not enter an impossible mixed state or lose the active sample unexpectedly.
- Double-click `手动保存`, `刷新预览`, `按边界保存并生成 Step-Claim 结构`, and `完成当前样本并保存`.
  - Expected: no duplicate rows, duplicate accepted solutions, or JS crashes.

## Step 2 and Step 3: segmentation edge cases
- Add cut points in reverse order, add duplicate cut points, and add cut points near the beginning and end.
  - Expected: the split preview is stable, duplicates are ignored, and empty segments do not create garbage steps.
- Enter Step 3 before generating any Step preview.
  - Expected: explicit empty/table fallback, not a blank panel.
- In Step 3, set `start > end`.
  - Expected: alert for invalid boundaries; Step 3 remains visible.
- In Step 3, skip a claim so ranges are non-contiguous.
  - Expected: alert that ranges must remain continuous; no broken claim structure is committed.
- In Step 3, use a case with zero pre-segmented claims.
  - Expected: explicit empty-state row `当前 solution 未提供预切分 claim`.

## Claim visibility and LaTeX rendering
- Use a case whose problem and reference answer contain valid inline and block LaTeX.
  - Expected: formulas render in the left and right panels.
- Use a sample whose solution contains broken inline LaTeX, broken block delimiters, and invalid commands.
  - Expected: the affected block falls back to raw text or `<pre>`, not an empty region.
- In Step 3, verify claim preview with:
  - one normal claim
  - one claim containing broken inline LaTeX
  - one claim containing an invalid command
  - one long plain-text claim
  - one claim containing HTML-like characters `< > &`
  - Expected: every row remains visible; a bad row does not hide other rows.
- In Step 4, edit a claim to an empty string, then mark another claim `删除`.
  - Expected: inputs remain editable, status chips still work, and later-step previews do not disappear.
- In Step 5, verify both cases:
  - first step with no prior claims
  - later step with bad-LaTeX claims as dependency candidates
  - Expected: either the empty-state message appears or all candidates remain visible; no blank dependency pane.
- After refreshing on a bad-LaTeX case, confirm Step 3 and Step 5 still show the relevant claim text.
  - Expected: restore path preserves visibility; no panel disappears after reload.

## Large-content and layout resilience
- Use very long solutions, very long steps, and very long claims.
  - Expected: the page remains scrollable; panels do not overlap into unreadable blanks.
- Resize the browser to a narrow width and then back to desktop width.
  - Expected: no content vanishes permanently; toggle buttons and restore controls still work.
- Collapse and restore the left/right panels repeatedly.
  - Expected: panels restore to a usable width and step content still renders.

## Network and save-failure behavior
- Throttle the network or disconnect it during `手动保存`.
  - Expected: `保存失败` or equivalent status appears; the page stays interactive.
- Throttle or interrupt the request during `完成当前样本并保存`.
  - Expected: the UI does not claim success if the request failed; reloading does not create a half-completed phantom state.
- Trigger multiple saves in quick succession while editing.
  - Expected: the latest successful state wins; no corrupted summary/detail files.

## Reviewer
- Log in as reviewer and load the review page with normal records present.
  - Expected: summary counts, record rows, and detail drawer all work.
- Verify a detail-only record.
  - Expected: reviewer shows a synthesized fallback row instead of losing the record.
- Corrupt a `.summary.json` file while leaving `.detail.json` intact.
  - Expected: reviewer keeps loading; the broken row is marked as an error and other rows still work.
- Corrupt both summary and detail for one record.
  - Expected: only that row is flagged; the whole page does not fail.
- Verify legacy `data/records/*.json` still appears.
  - Expected: legacy rows are visible.
- Verify a migrated pair plus stale legacy single-file copy.
  - Expected: only one row appears for that case.
- Edit and save the guideline, then reload.
  - Expected: the saved text persists exactly.

## Multi-tab behavior
- Open the same annotator/case in two tabs.
  - In tab A, make progress and save.
  - In tab B, make conflicting edits and save.
  - Expected: the app may be last-write-wins, but it must stay readable and recoverable; no invalid JSON or impossible workflow state should be produced.

## Record damage tests
- Delete only `.summary.json`.
  - Expected: annotator can still restore through detail; reviewer can still show a fallback summary.
- Delete only `.detail.json`.
  - Expected: reviewer can still show the summary; annotator does not pretend restore succeeded.
- Write invalid JSON into one file.
  - Expected: the failure is isolated and surfaced clearly.
