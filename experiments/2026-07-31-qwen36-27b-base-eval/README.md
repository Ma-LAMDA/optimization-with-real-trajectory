# Qwen3.6-27B 基座评测（部署 A/B 与全量运行）

本目录独立归档 2026-07-30 至 2026-07-31 的 Qwen3.6-27B 基座模型评测结果，
避免与早期 14 题 × 10 次轨迹生成实验混放。

## 目录结构

```text
2026-07-31-qwen36-27b-base-eval/
├── README.md
├── deployment-ab/
│   ├── attempts.csv
│   ├── deployment-decision.json
│   ├── report.md
│   └── summary.json
└── full-eval/
    ├── attempts.csv
    ├── report.md
    └── summary.json
```

`deployment-ab/` 保存单实例 TP=2 与双实例 TP=1×2 的部署对比；
`full-eval/` 保存按用户要求终止全量运行时，对已经结束轨迹生成的冻结统计。

## 统一判分口径

从最终答案中提取 `<result>...</result>`，将其中内容解析为 JSON，并与
`json.loads(source_record.answer)` 做整体精确相等比较。超时和 runner 失败均计错；
不能把“60 分钟内完成”当作“回答正确”。

## 部署 A/B

两组均使用题 4、5、20、89，每题 5 次，共 20 次；单次上限 60 分钟，总并发均为 2。

- A：单个 vLLM 实例，TP=2。完成 20/20、超时 0、严格正确 3/20（15%），
  平均/中位封顶耗时 26.66/26.72 分钟。
- B：两个 vLLM 实例，每实例 TP=1 且固定一个 worker。完成 20/20、超时 2、
  严格正确 5/20（25%），平均/中位封顶耗时 34.37/30.55 分钟。
- 速度选择规则要求 B 的超时数不高于 A，且平均封顶耗时低于 A；因此最终选择 A，
  即 `tp2x1`。

### 后续运行约束

后续运行固定沿用 A 拓扑：只启动 1 个 `tp2x1` vLLM 实例，使用 2 个 runner
worker，总请求并发固定为 2。禁止恢复旧的高并发方案，不得启动 8 个 worker、
8 路请求或让 runner 自动扩容；重试必须复用两个现有并发槽位。本约束只适用于
Qwen3.6-27B 基座/LoRA eval，不限制 Codex 轨迹生成等数据采集任务。

文件说明：

- [`deployment-ab/report.md`](deployment-ab/report.md)：便于阅读的逐题五次结果和汇总；
- [`deployment-ab/summary.json`](deployment-ab/summary.json)：完整结构化汇总及逐次预测；
- [`deployment-ab/attempts.csv`](deployment-ab/attempts.csv)：40 条 A/B 逐次明细；
- [`deployment-ab/deployment-decision.json`](deployment-ab/deployment-decision.json)：最终部署选择及规则。

## 全量运行终止快照

全量运行原计划覆盖其余 92 题、每题 5 次，共 460 次。实验在用户要求下停止，
watchdog、当时使用的 8 个 worker、活动任务和专用 TP2 vLLM 服务均已终止。
这里的 worker 数仅用于历史审计，不是后续运行配置；后续固定使用上述单实例双并发。

停止时已结束 381 次：成功输出 45、runner 失败 4、60 分钟超时 332；严格正确
2/381（0.52%），成功输出中的命中率为 2/45（4.44%）。超时按 60 分钟封顶后，
平均/中位耗时为 57.39/60.00 分钟。停止时的 8 个进行中样本作为未完成样本排除，
不进入准确率或耗时统计。

文件说明：

- [`full-eval/report.md`](full-eval/report.md)：总体、逐题结果和终止说明；
- [`full-eval/summary.json`](full-eval/summary.json)：结构化总体、逐题、终止状态及 A/B 参考；
- [`full-eval/attempts.csv`](full-eval/attempts.csv)：381 条已结束样本的逐次明细。

## 100×5 完整评测的组合规则

正式完整报告采用可审计的 500 槽位口径：题 1–88、91–94 的 460 次由当前单实例 TP=2、
双 runner 运行；题 89、90、95–100 复用旧留出实验的 37 次非超时结果。旧实验中的
`q89-r3`、`q90-r3`、`q99-r2` 三次 60 分钟超时不复用，改由单实例 TP=2、双 runner
各补跑一次并映射回原槽位。因此最终仍是每题 5 次、共 500 次，而不是额外增加到 503 次。

旧 37 次来自 8 个题目 worker 同时启动的历史运行，最终报告必须披露这一并发差异；该例外
只用于本次经确认的历史非超时结果复用，后续新运行仍固定单实例双并发。补跑和组合分别由
`scripts/run_seetacloud_base_full500_followup.sh` 与 `scripts/compose_base_full500_eval.py` 完成，
组合器会拒绝错误题号、重复槽位、缺失槽位、非预期超时集合或错误拓扑。

Base 全量运行允许在一对已启动样本自然结束后暂停，以便优先执行 LoRA 的最新 6 题 × 5 次
完整 Agent 留出验证。切换由 `scripts/run_seetacloud_lora_heldout_then_resume_base.sh` 管理：
先关闭 Base 服务，单独启动 checkpoint-760 +100 LoRA 服务，完成并汇总后关闭 LoRA，最后
使用同一 Base 前缀恢复。暂停和恢复不得并存两个模型实例，也不得把正在运行的两个样本截断。
恢复后首先补齐题 12、24、40、72、86、100 各 5 次；已有终态槽位复用，缺失槽位优先运行，
并生成独立的 `priority-heldout6-report`，供 LoRA 最新留出 6×5 完整 Agent 做严格同口径对比。
只有该 Base 对照报告完成后，才继续普通全量队列。
监督器会识别冻结控制器下已经终态的 zombie runner；该状态表示子任务已完成、只待父进程
回收，不会被误判成仍在请求模型，从而能够安全进入 Base 服务关闭和 LoRA 切换步骤。

## 数据边界

本目录只归档统计产物，不复制仓库外的完整轨迹、服务日志或离线配置。CSV 和 JSON
中的远端绝对路径用于历史审计，不表示克隆仓库后这些路径仍然存在。
