# 0809 five-epoch SFT and temperature-zero Agent result

This report records the completed 0809 error-aware SFT workflow and its fixed
temperature-zero Agent validation. The canonical scoring policy is
`agent-final-answer.v4.2026-08-12-incomplete-result-exact-recovery`.

The training source was commit
`f8db4abbc7891f43164c8849004c6b57b613b859`. Epoch 3/checkpoint 432 was
declared as the primary checkpoint before validation. Epochs 4 and 5 are
comparative checkpoints only and must not replace the primary retrospectively.

## Training

All five full-state SFT epochs completed. Each epoch exposed 1,151 rows and
performed 144 optimizer steps.

| Epoch | Checkpoint | Mean logged train loss | Last logged loss |
| ---: | ---: | ---: | ---: |
| 1 | 144 | 0.234749 | 0.243645 |
| 2 | 288 | 0.086537 | 0.163109 |
| 3 | 432 | 0.066639 | 0.004671 |
| 4 | 576 | 0.053631 | 0.006158 |
| 5 | 720 | 0.044804 | 0.010928 |

The mean logged training loss fell by 80.91% from epoch 1 to epoch 5. The last
single logged loss is noisier than the per-epoch mean and is included only as
diagnostic evidence.

## Fixed Agent protocol

- Cases: q2, q12, q19, q20, q29, q38, q65, q71, q85, q86, q99, q100.
- Target: five effective terminals per case, subject to the declared early-stop
  rule.
- Serving: one TP=2 vLLM instance, two Agent runners, total concurrency two.
- Maximum cell time: 3,600 seconds.
- Reasoning: `high`, Qwen3 parser, raw reasoning archived.
- Temperature: zero, enforced by both the vLLM generation override and an ASGI
  request-rewrite/audit middleware.
- Timeouts, pure infrastructure failures, and interruptions are archived and
  excluded from effective denominators.
- Early stop: at least three complete effective rounds and at least 30
  effective model errors.

The earlier epoch-3 temperature-one launch was user-terminated at 0/60. Its two
interrupted attempts remain preserved and are excluded from every result below.

## Aggregate result

| Epoch/checkpoint | Role | Correct/effective | Accuracy | Effective wrong | Full effective rounds | Final ceiling | Mean minutes | Mean input+output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3/432 | Predeclared primary | 20/57 | 35.09% | 37 | 3 | 23/60 (38.33%) | 17.13 | 3,535,239 |
| 4/576 | Comparative | 26/58 | **44.83%** | 32 | 3 | 28/60 (46.67%) | 12.99 | 1,826,771 |
| 5/720 | Comparative | 24/58 | 41.38% | 34 | 3 | 26/60 (43.33%) | 12.38 | 2,218,846 |

All three phases met the declared early-stop rule. Epoch 4 has the highest
observed score and lowest average token use, but remains a comparative result.
Relative to the predeclared epoch-3 primary, epoch 4 is +9.74 percentage points
and epoch 5 is +6.29 percentage points.

## Per-case result

| Case | Epoch 3 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: | ---: |
| q2 | 1/5 (20%) | 1/5 (20%) | 2/5 (40%) |
| q12 | 0/4 (0%) | 0/4 (0%) | 0/5 (0%) |
| q19 | 1/5 (20%) | 3/5 (60%) | 1/5 (20%) |
| q20 | 3/5 (60%) | 3/5 (60%) | 3/5 (60%) |
| q29 | 0/5 (0%) | 0/5 (0%) | 0/4 (0%) |
| q38 | 1/5 (20%) | 1/5 (20%) | 0/5 (0%) |
| q65 | 1/4 (25%) | 3/5 (60%) | 2/5 (40%) |
| q71 | 1/4 (25%) | 1/4 (25%) | 2/4 (50%) |
| q85 | 4/5 (80%) | 5/5 (100%) | 4/5 (80%) |
| q86 | 4/5 (80%) | 4/5 (80%) | 4/5 (80%) |
| q99 | 3/5 (60%) | 1/5 (20%) | 3/5 (60%) |
| q100 | 1/5 (20%) | 4/5 (80%) | 3/5 (60%) |

The epoch-4 improvement is concentrated in q19, q65, q85, and q100, while q99
regresses. q12 and q29 remain persistent zero-score cases across checkpoints.
Thus lower SFT loss does not translate monotonically into Agent accuracy.

## Final evidence audit

| Epoch | Accepted temperature audits | Nonzero / malformed | Reasoning evidence | Token/timing | Timeout archives | Infra archives | Preemption / OOM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 7,575 | 0 / 0 | 57/57 | 57/57 | 81 | 0 | 0 / 0 |
| 4 | 7,713 | 0 / 0 | 58/58 | 58/58 | 62 | 0 | 0 / 0 |
| 5 | 7,050 | 0 / 0 | 58/58 | 58/58 | 59 | 0 | 0 / 0 |

Epoch-4 q12-r04 had a telemetry-only reasoning gap in its original event file.
Reasoning was recovered non-destructively from the preserved Codex rollout into
a separate evidence directory; the original event hash and score were not
changed.

Final report SHA-256 values:

- Epoch 3: `87bccf16e4fe511edf62a99005f36ea0fb6ec1dee244c93b99aa33e73feca2b1`
- Epoch 4: `9bc87bae52aa18cbb1c0b3abfe28f50abbb522a8752a014d2e557449dcf6f124`
- Epoch 5: `ad879ea91b7996f45eb9b4a3686f2980ca6de264c86a8acacba0d947dd4dc415`

The final audit passed scoring-policy, all-zero temperature, completion,
reasoning, token/timing, preemption/OOM, process, GPU, and global-lock gates.
The machine-readable audit is
[`2026-08-19_0809_SFT_AGENT_TEMPERATURE0_RESULT.json`](2026-08-19_0809_SFT_AGENT_TEMPERATURE0_RESULT.json).

## Interpretation limit

These 12 cases were used for error mining and development. They are not an
independent held-out set, so the comparison does not establish independent
generalization. The epoch-3 checkpoint remains the predeclared primary even
though epoch 4 has the highest observed comparative score.
