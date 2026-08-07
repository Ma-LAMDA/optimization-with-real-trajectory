# Recoverable final-answer scoring rescore (2026-08-07)

## Decision

The normal answer protocol remains one `<result>...</result>` JSON string list. A
model answer is additionally accepted when all of the following hold:

1. the `<result>` wrapper is completely absent;
2. the final answer contains exactly one unique, non-conflicting fenced code
   block that can be parsed as a complete fault list; and
3. that complete list exactly matches one accepted answer option.

The recovered parse is recorded as `recovered_fenced_exact_match`. Prose
mentions, partial or approximate matches, multiple conflicting candidates, and
malformed explicit `<result>` wrappers are not recovered. Training-data
conversion and validation remain strict about the wrapper; the relaxation is a
scoring-only rule.

The existing q73-q86 inclusive-OR rule remains in force: A, B, A+B, and B+A are
accepted when A and B are the two documented VRRP alternatives.

## Result

Only one unique historical trajectory is newly recovered by the format rule:
the current 0805 `q100-r5`. Its fenced answer exactly matches the expected
`Core_SW_01;VRRP工作在非抢占模式`; the model turn completed normally, but the old
runner marked it failed solely because `<result>` was missing.

The current 0805 epoch-3 final validation becomes **35/60 (58.33%)**. The raw
archived strict flag total was 30/60; four corrections come from q73-q86
inclusive-OR and one from the new fenced exact-match recovery. Compared with the
corrected 0804 P1 result of 25/60 (41.67%), 0805 is **+16.67 percentage points**.

## Archived Agent experiments

The table uses effective terminal attempts only. Infrastructure failures and
unfinished attempts remain outside the denominator.

| Experiment | Archived correct | Unified-rule correct | Accuracy | Change |
|:--|--:|--:|--:|--:|
| 0731 base deployment A/B | 8/38 | 8/38 | 21.05% | 0 |
| 0731 base full eval | 2/49 | 12/49 | 24.49% | +10 |
| 0802 heldout6 base | 7/27 | 7/27 | 25.93% | 0 |
| 0802 heldout6 LoRA | 12/26 | 15/26 | 57.69% | +3 |
| 0804 best1 partial validation | 8/33 | 10/33 | 30.30% | +2 |
| 0805 q12/q100 reasoning-prefix validation | 2/10 | 2/10 | 20.00% | 0 |
| 0804 new P1 final validation | 23/60 | 25/60 | 41.67% | +2 |
| 0805 epoch-3 final validation | 30/60 | 35/60 | 58.33% | +5 |

All changes except the single 0805 `q100-r5` recovery are q73-q86 inclusive-OR
corrections.

## Other scored artifacts

- SFT deterministic validation outputs are unchanged. Base repeats remain
  5/60, 4/60, 4/60, 5/60, and 4/60; checkpoints 500/600/700/760 remain
  47/60, 48/60, 49/60, and 49/60; the five 0731 LoRA repeats remain 49/60;
  BB2 remains 10/10; +100 and +200 remain 52/60.
- The 2026-07-28 distillation judgments remain 819/1292 (63.39%).
- The 2026-08-02 GPT-5.6 distillation judgments remain 840/840 (100%).

## Audit scope

The rescore scanned 44 report groups and 4,175 recorded rows. These rows include
overlapping intermediate and final reports, so the aggregate 2,503/4,175 is an
audit occurrence count and must not be interpreted as one pooled benchmark.
The format recovery appears twice only because the same q100-r5 trajectory is
referenced by both an intermediate and the final report; it is one unique
trajectory.

Original `events`, `manifest`, `run.json`, trajectories, and historical reports
were not rewritten. The complete file-by-file machine-readable result, source
commit, and scorer hashes are in
[`2026-08-07_RECOVERABLE_FINAL_ANSWER_RESCORE.json`](2026-08-07_RECOVERABLE_FINAL_ANSWER_RESCORE.json).
