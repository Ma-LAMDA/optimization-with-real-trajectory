# GPT-5.6 Sol IP 轨迹蒸馏（100 题 × 每题 10 条正确轨迹）

本实验使用本地 Codex CLI 和 `gpt-5.6-sol` 覆盖 `train_0629.jsonl` 的全部
100 题。每题只收录经独立判题严格正确的轨迹，目标为 10 条；某题连续错误达到
10 次，或累计错误达到 20 次时停止该题并继续其他题。基础设施、额度、认证、模型
可用性和超时问题不计入题目错误次数。

## 当前状态：结果已重置

2026-08-03，用户确认实际 user prompt 存在问题，因此此前的 82 个 attempt 和全部
中间状态均已作废并删除。当前有效 accepted 和 attempt 都是 0，`results/` 不存在，
控制器和 Codex 子进程均已停止。实际运行 prompt 已于 2026-08-03 改为直接读取
`saved_configs/` 本地文件，并简化了读取方式和快照属性的限制性说明；当前等待用户复核。
确认后再从 q0001 attempt 1 开始全新采集，不得恢复旧结果。

完整任务、固定配置、prompt 复核清单和重新启动步骤见 [`HANDOFF.md`](HANDOFF.md)。

## 输入与数据边界

- 原始提示词副本：`inputs/IP user prompt by text.original.txt`。
- 实际运行提示词：`inputs/IP user prompt by text.txt`，已重新优化，当前等待用户确认。
- 题目源：仓库不可变原始文件 `data/simulation/train_0629.jsonl`。
- 配置根目录：仓库根目录 `saved_configs/`。
- 生成器只能直接列出、搜索和读取 `saved_configs/` 下的本地 `.txt` 固定快照；禁止
  通过 HTTP、API、浏览器或其他网络服务读取配置，也禁止访问该根目录之外的文件。
  它看不到源数据中的标准答案。判题器在每次 Codex 进程退出后，以单独进程读取标准
  答案并执行严格故障集合匹配。

实际运行提示词保留 `{original_query}`、`{output_format}` 两个唯一占位符，并包含
`{saved_configs_root}` 路径占位符。控制器只把后者替换为仓库 `saved_configs/` 的绝对路径，
不改写其余内容。Prompt 明确了目录为 `saved_configs/<项目>/<节点>/<命令回显>.txt`：先从
题目精确锁定项目，再列出真实节点和文件；命令文件名中非字母、数字、下划线、点或连字符
的每个字符替换为下划线。文件名转换只作定位线索，结论仍需读取实际文件内容核验。

运行时 hook 只允许 `Get-ChildItem`、`Get-Content`、`Select-String` 和 `Test-Path` 对上述
配置根目录执行只读操作；API、网络、写入、删除、移动、命令串联及越界路径均被拒绝。

## 运行配置

- 模型：`gpt-5.6-sol`。
- 速度：Standard（正常），显式关闭 Fast/priority。
- 初始/最大并发：10；遇到速率限制时自动降并发和退避。
- attempt 超时：2,700 秒。
- 会话：每个 attempt 使用全新 CLI 进程和 `--ephemeral`。

Prompt 确认后，从仓库根目录执行：

```powershell
python -B experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py
```

控制器会重新创建 `results/`，从零初始化 manifest、state、题目输入副本和 accepted 索引。

## 强制保留策略：只保留 accepted

> **失败或中断的结果一律不保留。只有 accepted 正确结果可以长期保存完整轨迹。**

这里的“不保留”包括：不归档、不提交，也不长期保留错误答案、格式错误、基础设施失败、
超时或中断 attempt 的事件流、回答、日志、判题文件、审计文件和目录。非 accepted attempt
只允许在执行及记账期间短暂存在；控制器必须先把错误/基础设施计数和停止阈值写入
`state.json`，随后删除整个 attempt 目录。运行中断留下的目录在下次启动完成中断记账后
删除。删除不会回退或复用 attempt 编号，连续错误和累计错误阈值仍按状态计数执行。

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
直接只读文件策略、事件流完整性和敏感凭据模式。
