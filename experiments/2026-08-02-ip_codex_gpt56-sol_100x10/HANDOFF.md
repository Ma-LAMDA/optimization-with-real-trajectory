# 0802 GPT-5.6-Sol IP 轨迹蒸馏：重置交接

## 当前状态

2026-08-03，用户确认当前实际 user prompt 存在问题，因此此前采集结果全部作废。
实验目录下的 `results/` 已整体删除，不再保留任何 accepted、错误、基础设施失败或中断
attempt，也不应从历史提交恢复这些旧结果。控制器、Codex 子进程和本地
`saved_configs_service` 均已停止，旧 attempt 监听已停用。

当前实验处于“等待修改 prompt”状态：

| 指标 | 当前值 |
|---|---:|
| 有效 accepted | 0 |
| 有效 attempt | 0 |
| 活跃控制器 / Codex attempt | 0 / 0 |
| `results/` | 不存在；下次运行时重新创建 |
| 是否可以立即恢复旧断点 | 否 |

旧结果仍可从 Git 提交 `6d7cc72c908f462cddccf2898607d8e7fe630932` 技术性找回，
但该提交只用于追溯，不能作为本实验的新起点。

## 任务目标与固定配置

- 数据源：`data/simulation/train_0629.jsonl`，共 100 题。
- 生成器：本地 Codex CLI，模型固定为 `gpt-5.6-sol`。
- 每题目标：10 条经独立严格判题正确的轨迹，总目标 1,000 条。
- 某题连续错误达到 10 次，或累计错误达到 20 次时停止该题并继续其他题。
- 基础设施、认证、额度、网络、模型不可用、服务不可用和超时不计入题目错误阈值。
- 配置根目录必须是仓库根目录 `saved_configs/`；生成器只能经只读本地 HTTP 服务查询。
- 运行速度固定为 Standard（正常），显式关闭 Fast/priority。
- 最大并发为 10；发生 429 时允许控制器自动降并发和退避。
- 每次 attempt 使用新的 CLI 进程和 `--ephemeral` 会话。

## Prompt 修改位置

- 原始副本：[`inputs/IP user prompt by text.original.txt`](inputs/IP%20user%20prompt%20by%20text.original.txt)
- 实际运行 prompt：[`inputs/IP user prompt by text.txt`](inputs/IP%20user%20prompt%20by%20text.txt)

重新采集前必须先修改并复核实际运行 prompt。至少确认：

1. `{original_query}` 和 `{output_format}` 各出现且只出现一次。
2. 配置目录明确写为 `saved_configs/`。
3. 生成器不能看到标准答案，也不能直接读取源数据或配置文件。
4. 本地只读 API、最终 `<result>...</result>` 和 JSON 字符串数组格式说明正确。
5. 不要先运行控制器；prompt 经用户确认后再从零启动。

## 强制产物保留策略：只保留 accepted

> **失败或中断的结果不用保留，也不得进入长期归档或提交。磁盘上长期存在的完整
> attempt 必须全部是 accepted。**

下次从零采集时按以下规则执行：

- accepted：保留完整 attempt 目录、事件流、最终答案、判题和审计证据。
- 答案错误或格式错误：先更新 `state.json` 中的错误计数和停止阈值，然后删除 attempt 目录。
- 基础设施失败：更新基础设施计数后删除 attempt 目录，不影响题目错误阈值。
- 中断：下次启动时完成中断记账，然后删除残留 attempt 目录。
- attempt 编号仍单调递增；删除失败目录不会造成编号复用。
- 最终审计要求磁盘上的 attempt 目录全部且仅来自 `accepted_index.json`。
- `events.jsonl`、最终回答、判题、stderr、hook 审计等失败/中断产物均随目录删除；不得以
  “便于排查”为由长期保留。排查只使用运行期间日志，必要计数以 `state.json` 为准。

## Prompt 确认后的启动步骤

1. 确认 Codex CLI 已切换到可使用 `gpt-5.6-sol` 的账号。
2. 确认 user prompt 已修改并完成复核。
3. 确认没有 Fast/priority 全局配置。
4. 从仓库根目录执行：

```powershell
python -B experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py
```

控制器会创建全新的 `results/`、manifest、state 和 accepted 索引，从 q0001 的
attempt 1 开始采集。不要恢复或复制旧 `results/`。

## 不可破坏的约束

- 不得把旧提交中的结果恢复为新实验断点。
- 不得长期保留或提交任何非 accepted（包括失败和中断）attempt 的结果或目录。
- 不得直接修改 accepted 或错误计数；只能由控制器和独立判题器更新。
- 不得把基础设施失败计入 `consecutive_wrong` 或 `total_wrong`。
- 不得让生成器读取标准答案或绕过只读本地 API。
- 不得重新开启 Fast/priority。
- 提交或推送新的运行结果时，同步更新本实验 README 和仓库根 README。
