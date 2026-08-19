# Codex IP 蒸馏实验（100 题 × 每题最多 10 条正确轨迹）

本目录是 `data/simulation/train_0629.jsonl` 的只读派生实验。生成器固定使用本机
Codex CLI 与模型 `gpt-5.6-sol`，并且只能通过本地 `saved_configs_service` HTTP API
查询仿真配置。原始数据、旧实验、仓库公共脚本和仓库根 README 均不得修改。

数据行实际没有 `metadata` 字段；依据仓库根 README 和已验证的 2026-07-28 10×10
实验，本实验将 `data/simulation/IP user prompt.txt` 视为这 100 行共享的原始 user
prompt 模板。唯一内容适配是把远程服务地址换成本次本地服务地址，再机械代入当前行
的 `question` 与 `output_format`。

## 目录

```text
scripts/                 输入代理、生成调度、独立判题和验证脚本
runtime/                 本地运行时目录；codex.exe 启动时生成且不提交
results/report/          manifest、状态、heartbeat、服务日志和最终报告
results/questions/       每题唯一的 prompt 和脱敏 source_record
results/runs/            每题的 success slot、所有 attempt 和原始事件流
config/hooks.json        历史运行共享的唯一 hooks 配置副本
```

每个成功槽位沿用旧实验名称 `qXXXX_rYY`；同一道题的 attempt 编号跨槽位单调递增。
每个 attempt 保留唯一原始流 `events.jsonl`，以及 `stderr.log`、`final_answer.txt`、
`metadata.json`、`timing.json`、`exit_code.txt` 和判题记录。原先与 `events.jsonl`
逐字节相同的 `stdout.log` 已删除；metadata 的 `stdout` 字段兼容性指向
`events.jsonl`。prompt 与脱敏 source record 按题号集中保存，临时 hooks workspace
在 attempt 结束后清理。

## 运行与恢复

从仓库根目录执行：

```powershell
py -3 -B experiments/2026-07-28-ip_codex_train0629_100x10/scripts/run_experiment.py
```

## 失败轨迹紧凑归档

历史实验的 819 条 `accepted` 轨迹继续完整保留。494 条未成功 attempt（473 条
`rejected`、10 条 `infrastructure_failure`、11 条 `interrupted`）仅保留
`metadata.json`、`timing.json`、`judgment.json`、`final_answer.txt`、退出码及恢复记录等
紧凑证据；其 `events.jsonl`、hook 明细和诊断日志已经裁剪。裁剪前每个文件或目录的
大小、SHA-256、事件行数及 hook 计数保存在
`results/report/failed_trajectory_pruning.json`，因此逐题 attempt 数、成功数、状态和平均
耗时仍可由保留的 metadata 复核。原始失败轨迹正文不再位于当前 Git 树中。

裁剪操作可先预览，再显式执行；当前归档只需运行检查：

```powershell
py -3 -B experiments/2026-07-28-ip_codex_train0629_100x10/scripts/prune_failed_trajectory_payloads.py
py -3 -B experiments/2026-07-28-ip_codex_train0629_100x10/scripts/prune_failed_trajectory_payloads.py --apply
py -3 -B experiments/2026-07-28-ip_codex_train0629_100x10/scripts/prune_failed_trajectory_payloads.py --check
```

`final_audit.py` 会同时验证完整 accepted 轨迹和失败轨迹裁剪清单；裁剪清单缺项、保留
文件哈希变化或应删除载荷重新出现都会使审计失败。

本目录属于数据采集实验，worker 数、并发调度和退避策略以本实验脚本及 manifest
为准，不受 Qwen3.6-27B eval 单实例双并发策略约束。

相同命令可断点续跑。已登记的正确 attempt 不会重跑；中断中的 attempt 会保留并标记
为 `interrupted`，下一次使用更大的 attempt 编号。最终或停止时会写入
`results/report/FINAL_REPORT.md`。

## 已归档结果

> 2026-08-07 标签修正：q73-q86 现在接受 A、B 或 A+B（任意顺序）。本实验这14题各保留
> 10条 accepted 轨迹，共140条；另有6次基础设施失败（q73/q74各3次），不进入计分。
> 可恢复证据中没有因该修正而由错转对的模型失败，因此总 accepted 数仍为819。原始轨迹和
> 旧 judgment 保持不变，修正审计见 `../../docs/2026-08-07_Q73_Q86_INCLUSIVE_OR_RESCORE.md`。

- 输入 100 题，全部进入调度；
- 79 题完成每题 10 条正确轨迹，21 题在连续 10 次错误后停止；
- 共保留 819 条 accepted 正确轨迹；
- 共记录 1,313 次 attempt、1,302 次 Codex CLI 调用和 21 次基础设施失败；
- 完整统计见 `results/report/FINAL_REPORT.md` 和 `results/report/summary.json`。
