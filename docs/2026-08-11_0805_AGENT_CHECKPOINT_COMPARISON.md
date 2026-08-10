# 0805 Agent checkpoint 统一比较（2026-08-11）

## 口径

- 计分版本：`agent-final-answer.v3.2026-08-10-final-answer-only`。
- 唯一入口：`scripts/final_answer_scoring.py`；所有结果均从最终答案重算或验证相同 `scoring_policy_version`。
- 固定单元：q2、q12、q19、q20、q29、q38、q65、q71、q85、q86、q99、q100，各 5 次，TP=2、concurrency=2、单次上限 3,600 秒。
- 只看最终答案；工具调用、router、协议和归因仅作诊断。超时、基础设施失败和人为中断不进入有效分母。
- 提前终止仅在至少 3 个完整有效轮次且累计有效模型错误达到 30 时触发。提前终止的观察准确率是部分值，不能伪装为 60/60 完整结果。

## 结果

| 实验 | checkpoint | 正确/有效 | 准确率 | 状态 | 60 格理论上限 |
|---|---:|---:|---:|---|---:|
| 0805 epoch-1 | 144 | 18/48 | 37.50% | 四轮后 30 错，合规提前终止；12 个第五轮槽位未运行 | 30/60（50.00%） |
| 0805 epoch-2 | 288 | 36/60 | 60.00% | 完整 | 36/60（60.00%） |
| 0805 epoch-3 | 432 | 35/60 | 58.33% | 完整，v3 重算 | 35/60（58.33%） |
| 0805 epoch-4 | 576 | 37/60 | 61.67% | 完整 | 37/60（61.67%） |
| 0805 epoch-5 | 720 | 35/60 | 58.33% | 完整 | 35/60（58.33%） |
| 0804 固定参考 | — | 25/60 | 41.67% | 固定参考 | 25/60（41.67%） |

补充状态：0807 epoch-5 为 11/41（26.83%）的提前终止部分值，理论上限 50.00%；0807 epoch-3 由用户终止且不完整，未恢复、未补跑。

## 结论

- 在四个完整的 60/60 checkpoint 中，epoch-4 最高，为 37/60（61.67%）。
- epoch-2 为 36/60（60.00%），比 epoch-4 低 1.67 个百分点。
- epoch-3 与 epoch-5 均为 35/60（58.33%），比 epoch-4 低 3.34 个百分点。
- epoch-1 的观察部分值为 18/48（37.50%）；其最终理论上限仅 50.00%，因此不可能达到 epoch-2/3/4/5 的完整准确率，但不得把 18/48 换算成 60 格最终值。

## epoch-1 提前终止明细

| 题目 | 五个固定槽位 | 正确/有效 | 部分准确率 |
|---|---|---:|---:|
| q2 | ❌ ❌ ❌ ❌ ⬜ | 0/4 | 0.00% |
| q12 | ❌ ❌ ❌ ❌ ⬜ | 0/4 | 0.00% |
| q19 | ✅ ✅ ❌ ❌ ⬜ | 2/4 | 50.00% |
| q20 | ❌ ✅ ✅ ✅ ⬜ | 3/4 | 75.00% |
| q29 | ❌ ❌ ❌ ❌ ⬜ | 0/4 | 0.00% |
| q38 | ❌ ❌ ❌ ❌ ⬜ | 0/4 | 0.00% |
| q65 | ❌ ❌ ❌ ✅ ⬜ | 1/4 | 25.00% |
| q71 | ✅ ✅ ✅ ❌ ⬜ | 3/4 | 75.00% |
| q85 | ✅ ✅ ✅ ✅ ⬜ | 4/4 | 100.00% |
| q86 | ✅ ✅ ✅ ✅ ⬜ | 4/4 | 100.00% |
| q99 | ❌ ❌ ❌ ✅ ⬜ | 1/4 | 25.00% |
| q100 | ❌ ❌ ❌ ❌ ⬜ | 0/4 | 0.00% |

epoch-1 共 48 个有效终态，thinking 48/48，2,703 条 reasoning；没有超时或基础设施失败占位。独立监控器在第四轮完整后写入决策并停止新调度；第五轮 12 个槽位均未运行。

## SeetaCloud 证据

- epoch-1 决策：`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0805/0805-formal-5epoch-2gpu-20260807T103558+0800/epoch1_final_validation/control/early_stop_decision.json`
- epoch-1 v3 重算：`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0805/0805-formal-5epoch-2gpu-20260807T103558+0800/epoch1_final_validation/report_early_stop_20260811T0710+0800/validation_summary.json`
- epoch-1 Markdown 报告：`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0805/0805-formal-5epoch-2gpu-20260807T103558+0800/epoch1_final_validation/report_early_stop_20260811T0710+0800/report.md`
- epoch-2 完整报告：`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0805/0805-formal-5epoch-2gpu-20260807T103558+0800/epoch2_final_validation/report_recovery_20260810T1611+0800/validation_summary.json`
- epoch-5 完整报告：`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0805/0805-formal-5epoch-2gpu-20260807T103558+0800/epoch5_final_validation/report_recovery_20260810T1643+0800/validation_summary.json`

提前终止后，epoch-1 launcher、runner、vLLM 和监控器进程均已退出，两张 GPU 为 0 MiB / 0%，锁已清除；死亡 PID 标记保存在阶段内归档目录，原始 events、manifest、attempt、轨迹和日志未被覆盖或删除。
