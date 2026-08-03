# GPT-5.6 Sol IP 轨迹蒸馏（100 题 × 每题 10 条正确轨迹）

本实验使用本地 Codex CLI 和 `gpt-5.6-sol` 覆盖 `train_0629.jsonl` 的全部
100 题。每题只收录经独立判题严格正确的轨迹，目标为 10 条；某题连续错误达到
10 次，或累计错误达到 20 次时停止该题并继续其他题。基础设施、额度、认证、模型
可用性和超时问题不计入题目错误次数。

## 当前交接断点（2026-08-03 16:26，北京时间）

实验已为切换 Codex CLI 账号主动停止并完成断点整理：控制器和活跃 attempt 均为 0，
当前 accepted 为 18 / 1,000，已分配 82 个 attempt，其中 40 个答案错误、24 个基础设施
失败；尚无题目达到 10 条或触发跳过阈值。恢复配置为 Standard（正常）速度、最大并发 10。

完整任务说明、逐题进度、账号切换和恢复步骤见 [`HANDOFF.md`](HANDOFF.md)。恢复时以
`results/report/state.json` 和 `results/report/accepted_index.json` 为权威断点。

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
自动恢复；网络/DNS 故障单独归类、退避重试且不计入题目错误次数。最多并发 10 个 Codex
进程；每个生成进程显式关闭 Fast mode，按 Standard（正常）速度运行；遇到速率限制会
自动降低并发和退避。

## 产物

`results/` 已在 2026-08-03 的账号切换点作为中间检查点提交，以支持换号或新任务接手后
原地恢复；后续完成时还需重写最终报告并执行独立审计。逐命令 hook 临时碎片已经合并到
每个 attempt 的 `hook_audit.jsonl`，不重复提交；临时锁文件和本地 runtime 也不提交。

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
