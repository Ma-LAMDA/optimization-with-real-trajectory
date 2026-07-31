# Qwen3.6-27B checkpoint-760 +100 完整 Agent A/B

本目录归档推荐 LoRA checkpoint 与历史 base-eval 的同条件端到端 A/B。LoRA 侧于
2026-07-31 至 2026-08-01 在 SeetaCloud 实跑；base 侧直接复用已经归档的 20 次
`tp2x1` 结果，没有重复推理。

## 实验口径

- 候选模型：`Qwen3.6-27B-trained`，即 checkpoint-760 之后以 `1e-5` 独立续训
  100 steps 的 LoRA checkpoint；远端路径为
  `/root/autodl-tmp/optimization-with-real-trajectory/output/qwen36-27b-lora-0731-step760-plus200-v2/train/v0-20260731-203846/checkpoint-100`。
- 代码提交：`ee025b2e96197d7a1cc85fac462ca69e1afb9aaf`。
- 题目：4、5、20、89，每题 5 次，共 20 次。
- Agent：复用 `2026-07-27-ip_codex_train0629_14x10` 的 Codex CLI runner、完整调查
  prompt、离线 `saved_configs` 工具和严格 `<result>` 判分。
- 部署：单个 vLLM 实例、TP=2、两个 Agent worker、总并发 2；没有使用 8 并发。
- 单次硬上限：3600 秒；超时和 runner 失败均按错误计。
- base：复用
  `experiments/2026-07-31-qwen36-27b-base-eval/deployment-ab/summary.json`
  中相同四题、相同重复数、相同 TP2 拓扑的结果。

## 结果

| 条件 | 严格正确 | 准确率 | 超时 | 平均封顶耗时/分 | P95/分 |
|:---|---:|---:|---:|---:|---:|
| Base（复用） | 3/20 | 15.00% | 0 | 26.66 | 34.94 |
| checkpoint-760 +100 | 15/20 | 75.00% | 0 | 13.95 | 21.12 |
| LoRA - Base | +12 | +60.00 pp | +0 | -12.71 | -13.82 |

LoRA 分题结果为：题 4、5、20 均 4/5，题 89 为 3/5。20 次全部在 60 分钟内
完成，没有 runner 失败；false positive / false negative 为 5 / 2。完整事件、命令和
token 汇总见 `summary.json`。

该结果说明推荐 checkpoint 在历史 Agent 任务上的端到端正确率和运行耗时均明显优于
base，但四道题都已经进入当前 SFT 训练集，因此它只能作为历史 base-eval 的兼容性
A/B，不能作为留出集泛化结论。正式工作流仍从当前 manifest 读取题 12、24、40、
72、86、100，并在相同完整 Agent、单实例 TP2、双并发、60 分钟口径下每题运行 5 次。

## 文件

- `report.md`：总体、分题和 base A/B 的可读摘要。
- `summary.json`：20 次运行的结构化结果、预测、遥测和基线差异。
- `attempts.csv`：逐次预测、期望答案、严格判分、耗时和事件计数。

原始事件流和服务日志保留在 SeetaCloud 的
`/root/autodl-tmp/qwen-codex-eval/agent-ab-step760-plus100-20260731T143800Z-*`；仓库只提交
紧凑统计产物，避免复制大体积轨迹和配置快照。
