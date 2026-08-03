# GPT-5.6 Sol IP 轨迹蒸馏（100 题 × 每题 10 条正确轨迹）

本实验使用本地 Codex CLI 和 `gpt-5.6-sol` 覆盖 `train_0629.jsonl` 的全部
100 题。每题只收录经独立判题严格正确的轨迹，目标为 10 条；某题连续错误达到
10 次，或累计错误达到 20 次时停止该题并继续其他题。基础设施、额度、认证、模型
可用性和超时问题不计入题目错误次数。

## 输入与数据边界

- 原始提示词副本：`inputs/IP user prompt by text.original.txt`。
- 优化提示词：`inputs/IP user prompt by text.txt`。
- 题目源：仓库不可变原始文件 `data/simulation/train_0629.jsonl`。
- 配置根目录：仓库根目录 `saved_configs/`。
- 生成器只能通过只读本地服务查询 `saved_configs/` 的固定快照；它看不到源数据中的
  标准答案。判题器在每次 Codex 进程退出后，以单独进程读取标准答案并执行严格故障
  集合匹配。

优化提示词保留题目和输出格式两个占位符，删除重复说明，并明确项目锁定、按假设取证、
交叉验证、证据不足处理与最终 JSON 格式。运行时只把本地服务的监听端口适配到当前端口，
不改变其余提示词。

## 运行

在仓库根目录执行：

```powershell
python experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py
```

脚本支持原地恢复。重复执行同一命令时，已收录的正确轨迹不会重复，未终态题目从单调
递增的 attempt 编号继续。基础设施停机断点在重新执行并通过 CLI、登录和本地服务预检后
自动恢复；网络/DNS 故障单独归类、退避重试且不计入题目错误次数。最多并发 4 个 Codex
进程；每个生成进程显式关闭 Fast mode，按 Standard（正常）速度运行；遇到速率限制会
自动降低并发和退避。

## 产物

`results/` 在实验运行期间暂不提交；实验完成、独立审计通过并完成归档整理后，将轨迹、
判题记录和最终报告一并提交到 `2026-07-31-sft`。临时锁文件和本地 runtime 不提交。

```text
results/
├── questions/qXXXX/{prompt.txt,source_record.json}
├── runs/qXXXX_rYY/attempt_ZZZ/
│   ├── events.jsonl
│   ├── final_answer.txt
│   ├── judgment.json
│   └── metadata.json
└── report/
    ├── state.json
    ├── accepted_index.json
    ├── heartbeat.json
    ├── summary.json
    ├── FINAL_REPORT.md
    └── final_audit.json
```

只有 `accepted_index.json` 中列出的 attempt 属于正确轨迹。错误、格式错误、中断和基础
设施失败 attempt 仍完整保留，以便复核停止条件和运行过程。

## 独立审计

实验到达终态后，生成器会运行独立审计。也可手工复核：

```powershell
python experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/final_audit.py
```

审计会复查 100 题调度、模型标识、每条 accepted 轨迹的独立严格匹配、停止阈值、
临时工作区隔离、只读查询策略、事件流完整性和敏感凭据模式。
