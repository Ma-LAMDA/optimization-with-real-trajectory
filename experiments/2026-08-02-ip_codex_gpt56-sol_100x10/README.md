# GPT-5.6 Sol IP 轨迹蒸馏（100 题 × 每题 10 条正确轨迹）

本实验使用本地 Codex CLI 和 `gpt-5.6-sol` 覆盖 `train_0629.jsonl` 的全部
100 题。每题只收录经独立判题严格正确的轨迹，目标为 10 条；某题连续错误达到
10 次，或累计错误达到 20 次时停止该题并继续其他题。基础设施、额度、认证、模型
可用性和超时问题不计入题目错误次数。

## 当前状态：结果已重置

2026-08-03，用户确认实际 user prompt 存在问题，因此此前的 82 个 attempt 和全部
中间状态均已作废并删除。当前有效 accepted 和 attempt 都是 0，`results/` 不存在，
控制器、Codex 子进程及本地配置服务均已停止。现在应先修改
`inputs/IP user prompt by text.txt`，经用户确认后再从 q0001 attempt 1 开始全新采集；
不得恢复旧结果。

完整任务、固定配置、prompt 复核清单和重新启动步骤见 [`HANDOFF.md`](HANDOFF.md)。

## 输入与数据边界

- 原始提示词副本：`inputs/IP user prompt by text.original.txt`。
- 实际运行提示词：`inputs/IP user prompt by text.txt`，当前等待用户修改。
- 题目源：仓库不可变原始文件 `data/simulation/train_0629.jsonl`。
- 配置根目录：仓库根目录 `saved_configs/`。
- 生成器只能通过只读本地服务查询 `saved_configs/` 的固定快照；它看不到源数据中的
  标准答案。判题器在每次 Codex 进程退出后，以单独进程读取标准答案并执行严格故障
  集合匹配。

实际运行提示词必须保留 `{original_query}` 和 `{output_format}` 两个唯一占位符，并明确
项目锁定、按假设取证、交叉验证、证据不足处理与最终 JSON 格式。运行时只能适配本地
服务监听端口，不得改变提示词其余内容。

## 运行配置

- 模型：`gpt-5.6-sol`。
- 速度：Standard（正常），显式关闭 Fast/priority。
- 最大并发：10；遇到速率限制时自动降并发和退避。
- attempt 超时：2,700 秒。
- 会话：每个 attempt 使用全新 CLI 进程和 `--ephemeral`。

Prompt 修改并确认后，从仓库根目录执行：

```powershell
python -B experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py
```

控制器会重新创建 `results/`，从零初始化 manifest、state、题目输入副本和 accepted 索引。

## Accepted-only 保留策略

后续采集只长期保留 accepted 正确轨迹的完整 attempt 目录。错误答案、格式错误和基础设施
失败会先写入 `state.json` 的计数，再删除其 attempt 目录；运行中断留下的目录会在下次
启动完成中断记账后删除。删除不会回退或复用 attempt 编号，连续错误和累计错误阈值仍按
状态计数执行。

最终 `results/` 的主要结构为：

```text
results/
├── questions/qXXXX/{prompt.txt,source_record.json}
├── runs/qXXXX_rYY/attempt_ZZZ/   # 仅 accepted
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

最终审计会验证磁盘上的 attempt 目录全部且仅由 `accepted_index.json` 引用，同时复查
100 题调度、模型标识、每条 accepted 轨迹的独立严格匹配、停止阈值、临时工作区隔离、
只读查询策略、事件流完整性和敏感凭据模式。
