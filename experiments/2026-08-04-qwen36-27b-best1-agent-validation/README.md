# 0804 best1 LoRA SFT 与部分 Agent 验证

本目录归档 2026-08-04 每题最佳一条原生多轮 SFT 的 SeaTACLOUD 快跑，以及按用户
指令提前停止后的有效 Agent 验证结果。训练完整结束；Agent 验证原计划 12 题各 5 次，
在 39 个 attempt 完成后停止，剩余 21 个未启动 attempt 不计失败。

## 训练结果

- 基座：Qwen3.6-27B；LoRA 可训练参数 58,363,904。
- 数据：训练 318 个节点样本、验证 53 个节点样本，最大长度 16,384 token。
- 训练：1 epoch、159 step；train loss `0.26836527`。
- eval loss：step 40=`0.3065788`、80=`0.1956932`、120=`0.1823089`、
  159=`0.1806803`。
- 最低 eval loss 位于 `checkpoint-159`，该 checkpoint 用于 Agent 验证。

## Agent 验证结果

- 运行：Codex CLI + LoRA，单个 vLLM TP=2 实例、两个 Agent runner，总并发 2。
- Thinking：显式设置 `reasoning_effort=high`。本地 provider 的独立 reasoning token
  计数能力有限，不能用该计数反推 thinking 未开启。
- 已执行 39/60，严格正确 8/39（20.51%）。
- 模型硬超时 6 次；基础设施失败 0；中断不入表；未启动 21 次不计失败。
- q85、q86 最好，各为 2/3；主要错误是过度搜索、停止判断不足、标签映射错误和设备
  集合多报。

用户要求等待当时在途的 q12、q19 第 4 次完成后停止。主调度器先暂停以禁止补位；
q12 以模型硬超时结束，q19 正常返回后，调度器、外层工作流和 vLLM 均已退出，GPU
计算进程已释放。整个停止过程没有把基础设施失败或人为中断写成评测样本。

## 文件

- [`training_summary.json`](training_summary.json)：训练与最低 eval-loss checkpoint 摘要。
- [`validation_summary.json`](validation_summary.json)：39 个有效槽位的完整机器可读汇总。
- [`attempts.csv`](attempts.csv)：逐 attempt 预测、答案、耗时、事件与 token 指标。
- [`report.md`](report.md)：逐题可读报告与主要错误模式。
- [`workflow_summary.json`](workflow_summary.json)：训练和部分验证的端到端状态。
- [`fragments/`](fragments/)：完整前三轮与第 4 轮三题的原始汇总片段。
- [`compose_partial_summary.py`](compose_partial_summary.py)：合并不等长重复轮次的复现脚本。

从本目录重新合并：

```powershell
python compose_partial_summary.py `
  fragments/repeats1_3_validation_summary.json@0 `
  fragments/repeat4_q2_q12_q19_validation_summary.json@3 `
  --training-summary training_summary.json `
  --output-dir . `
  --scheduled-attempts 60 `
  --stop-note "用户要求当前在途 q12/q19 第4次完成后停止；调度器已提前暂停，未再启动后续题次。"
```
