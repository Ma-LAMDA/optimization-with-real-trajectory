# 0802 GPT-5.6-Sol IP 轨迹蒸馏：任务与断点交接

本文档用于在 Codex CLI 账号切换或新 Codex 任务接手后，继续同一个 100 题轨迹蒸馏实验。
状态快照时间为 **2026-08-03 16:26:47（北京时间）**。本文档、实验脚本、提示词和
`results/` 中的中间轨迹/断点共同构成可恢复检查点。

为控制 Git 文件数，每个 attempt 的逐命令 `hook_audit_parts/` 临时碎片不进入检查点；
其内容在 attempt 结束时已合并到同目录 `hook_audit.jsonl`，恢复和最终审计均使用合并文件。

## 1. 任务目标

- 数据源：仓库 `data/simulation/train_0629.jsonl`，共 100 题。
- 生成器：本地 Codex CLI，模型固定为 `gpt-5.6-sol`。
- 每题目标：收录 10 条经独立严格判题正确的完整轨迹，总目标 1,000 条。
- 低正确率停止：某题连续错误达到 10 次，或累计错误达到 20 次时停止该题并继续其他题。
- 基础设施、认证、额度、网络、模型可用性、服务不可用和超时不计入题目错误次数。
- 配置数据必须来自仓库根目录 `saved_configs/`；生成器只能通过只读本地 HTTP 服务查询，
  不得直接读取配置文件或源数据中的标准答案。

## 2. 固定运行配置

| 配置项 | 当前值 |
|---|---|
| 仓库/分支 | `optimization-with-real-trajectory` / `2026-07-31-sft` |
| 实验目录 | `experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` |
| 模型 | `gpt-5.6-sol`（GPT-5.6-Sol） |
| 推理强度 | 本机 CLI 默认 `xhigh` |
| 运行速度 | Standard（正常）；每个 attempt 显式 `--disable fast_mode` |
| 初始并发 | 4 |
| 当前/最大并发 | 10 |
| attempt 超时 | 2,700 秒（45 分钟） |
| 会话隔离 | 每次使用全新 CLI 进程和 `--ephemeral` 会话 |
| Web 搜索 | 禁用 |
| 配置服务 | `http://127.0.0.1:3080`，只读 `saved_configs/` |
| 判题 | CLI 退出后由独立 `judge_attempt.py` 严格比较故障集合 |
| accepted 判定 | 与平铺标准答案或显式备选答案精确相等；忽略顺序，不允许缺失或多余项 |

提示词与脚本：

- 原始提示词：[`inputs/IP user prompt by text.original.txt`](inputs/IP%20user%20prompt%20by%20text.original.txt)
- 实际优化提示词：[`inputs/IP user prompt by text.txt`](inputs/IP%20user%20prompt%20by%20text.txt)
- 主控制器：[`scripts/run_experiment.py`](scripts/run_experiment.py)
- 独立判题器：[`scripts/judge_attempt.py`](scripts/judge_attempt.py)
- API 访问约束：[`scripts/api_only_hook.py`](scripts/api_only_hook.py)

## 3. 当前断点

实验因 CLI 账号额度即将不足而主动停止；控制器及所有 Codex 子进程均已终止，未完成的
10 个 attempt 已按基础设施中断处理，不增加答错次数。

| 指标 | 快照值 |
|---|---:|
| 状态 | `interrupted` |
| 活跃控制器/attempt | 0 / 0 |
| accepted 正确轨迹 | 18 / 1,000（1.8%） |
| 达到每题 10 条的题目 | 0 / 100 |
| 连续错误阈值跳过 | 0 |
| 累计错误阈值跳过 | 0 |
| 已分配 attempt | 82 |
| 已启动模型 attempt | 64 |
| 答案错误 | 40 |
| 基础设施失败 | 24 |
| 待继续题目 | 100 |

当前仅 q0001–q0010 产生过 attempt；q0011–q0100 尚未开始。

| 题目 | accepted | 连续错 | 累计错 | attempt | 模型调用 | 基础设施失败 | 下一个 attempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| q0001 | 4 | 2 | 5 | 13 | 10 | 4 | 14 |
| q0002 | 4 | 1 | 7 | 16 | 13 | 5 | 17 |
| q0003 | 4 | 1 | 6 | 14 | 11 | 4 | 15 |
| q0004 | 2 | 4 | 8 | 15 | 12 | 5 | 16 |
| q0005 | 1 | 0 | 2 | 4 | 3 | 1 | 5 |
| q0006 | 1 | 2 | 2 | 4 | 3 | 1 | 5 |
| q0007 | 0 | 3 | 3 | 4 | 3 | 1 | 5 |
| q0008 | 0 | 3 | 3 | 4 | 3 | 1 | 5 |
| q0009 | 1 | 0 | 2 | 4 | 3 | 1 | 5 |
| q0010 | 1 | 0 | 2 | 4 | 3 | 1 | 5 |

权威断点文件：

- [`results/report/state.json`](results/report/state.json)：题目计数、下一 attempt 编号和运行状态。
- [`results/report/accepted_index.json`](results/report/accepted_index.json)：18 条 accepted 的唯一映射。
- [`results/report/manifest.json`](results/report/manifest.json)：模型、输入哈希、并发和判题配置。
- [`results/report/heartbeat.json`](results/report/heartbeat.json)：停止后的最后心跳。
- [`results/report/runner.log`](results/report/runner.log)：调度历史。

本机若仍存在 `FINAL_REPORT.md`、`summary.json` 或 `final_audit.json`，它们是较早一次基础设施
停机时生成的阶段报告，不代表本快照，因此未纳入本次中间检查点提交。恢复时应以
`state.json`、`accepted_index.json` 和本文档为准，最终完成后控制器会重新生成这些报告。

## 4. 已处理问题和关键决策

1. 2026-08-02 首轮运行因 `chatgpt.com` DNS 解析失败停止。旧分类器把设备配置中的
   `authentication` 字样误当成 CLI 认证失败；现已只从 stderr/错误事件分类，并新增
   `network_failure`。
2. 已修复 `stopped_by_infrastructure_blocker` 断点不能自动恢复的问题；重新运行会恢复非终态题目。
3. Fast/priority 已关闭，后续固定 Standard（正常）速度；模型和 `xhigh` 推理强度不变。
4. 最大并发由 4 提升到 10；遇到 429 时控制器仍会自动降并发和退避。
5. 账号切换前已停止新 attempt 并整理断点。中断只增加基础设施失败，不改变题目错误阈值。

## 5. 切换 CLI 账号

当前检查点创建时 CLI 仍登录旧 ChatGPT 账号。必须在恢复实验前完成切换：

```powershell
cd D:\myWork\github\optimization-with-real-trajectory\2026-07-31-sft

experiments\2026-08-02-ip_codex_gpt56-sol_100x10\runtime\codex.exe logout
experiments\2026-08-02-ip_codex_gpt56-sol_100x10\runtime\codex.exe login
experiments\2026-08-02-ip_codex_gpt56-sol_100x10\runtime\codex.exe login status
```

在浏览器中登录新账号，并确认新账号可使用 `gpt-5.6-sol`。不要提交、复制或发送
`~/.codex/auth.json`；它包含可刷新访问令牌。若本地 runtime 不存在，可直接使用 PATH 中的
`codex` 执行相同命令，运行器会在恢复时重新复制 CLI。

## 6. 恢复步骤

1. 拉取包含本文档和 `results/` 检查点的 `origin/2026-07-31-sft`。
2. 确认 `codex login status` 显示新账号已登录。
3. 确认没有全局 `service_tier = "priority"`；运行器也会显式关闭 Fast mode。
4. 确认 `results/report/state.json` 为 `interrupted`、accepted 为 18、没有 `current_attempt`。
5. 从仓库根目录执行：

```powershell
python -B experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py
```

运行器会验证数据哈希、CLI、模型缓存和登录状态，复用或启动 `saved_configs_service`，从
q0001 的 attempt 14、q0002 的 attempt 17 等单调递增编号继续；已 accepted 的 18 条不会重跑。

恢复后检查：

```powershell
Get-Content experiments/2026-08-02-ip_codex_gpt56-sol_100x10/results/report/heartbeat.json
Get-Content experiments/2026-08-02-ip_codex_gpt56-sol_100x10/results/report/runner.log -Tail 30
```

预期 `state_status` 为 `running`、`current_concurrency` 为 10，并出现 10 个新的运行中 attempt。

## 7. 不可破坏的约束

- 不得删除、覆盖或重编号已有 `results/runs/**/attempt_*`。
- 不得直接修改 accepted 计数；只能由独立判题器更新。
- 不得把基础设施失败计入 `consecutive_wrong` 或 `total_wrong`。
- 不得让生成器读取 `train_0629.jsonl` 中的标准答案或绕过只读本地 API。
- 不得重新开启 Fast/priority；账号切换只改变认证，不改变模型、提示词、判题或并发配置。
- 提交/推送新的运行结果时，同步更新本实验 README 和仓库根 README。
