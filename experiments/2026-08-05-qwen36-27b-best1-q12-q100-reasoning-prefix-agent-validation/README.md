# 0804 best1：q12 / q100 Agent 复测

本目录归档 2026-08-05 对 0804 best1 checkpoint 的 q12、q100 并发复测结果。每题独立运行 5 次，使用 Codex CLI Agent、高 thinking、两个 worker 和单个 TP=2 vLLM 服务。完整逐步事件与原始日志保留在远端实验目录，本仓库保存汇总表和可复核指标。

## 结论

| 题号 | 本轮严格正确率 | 上轮严格正确率 | 本轮平均耗时 | 上轮平均封顶耗时 |
|---:|---:|---:|---:|---:|
| 12 | 0/5（0%） | 0/4（0%，含 1 次超时） | 17.04 分钟 | 40.07 分钟 |
| 100 | 2/5（40%） | 1/3（33.3%） | 11.42 分钟 | 14.22 分钟 |
| 合计 | 2/10（20%） | 1/7（14.3%） | 14.23 分钟 | 28.99 分钟 |

准确率没有呈现足够明确的提升：q12 依然完全失败；q100 虽由 1/3 变为 2/5，但样本很小，且错误仍集中在“已找到 VRRP 非抢占证据，却映射到错误 label”或被旁路 STP 证据带偏。两轮重复数不同，因此合计准确率只用于描述，不能视为显著性结论。

耗时则明显下降：q12 平均封顶耗时下降 57.5%，q100 下降 19.7%，合计下降 50.9%，本轮 10 次均在 60 分钟内完成。该差异同时受到 prefix cache、模型元数据修复、输入截断单位修复、并行工具调用等变更影响，不能仅归因于 prefix cache。

## 本轮配置

- checkpoint：`checkpoint-159`（0804 best1）
- 题目：q12、q100；每题 5 次
- 推理：`reasoning_effort=high`，原始 reasoning 强制捕获
- 并发：一个 vLLM TP=2 实例，两个 Agent worker
- 超时：每次 3600 秒；超时和 runner 失败均按错误计
- 判分：最终 `<result>` JSON 列表与标准 label 完全相等
- prefix cache：显式启用；Qwen3.6 混合 Mamba 缓存使用 `align` 模式
- 远端完整目录：`/root/autodl-tmp/optimization-with-real-trajectory/output/qwen36-27b-lora-0804-best1-reasoning-prefix-q12-q100x5-20260805T095337Z`

## 逐次结果

q12 的标准答案为 `Core_SW_02;全局STP未使能`。5 次均错误，分别输出：

1. `PE2;ISIS配置错误`、`PE3;ISIS配置错误`
2. `PE2;BGP配置错误`、`PE3;BGP配置错误`
3. `PE3;L3VPN配置错误`
4. `PE2;ISIS配置错误`、`PE3;ISIS配置错误`
5. `PE2;ISIS配置错误`、`PE3;ISIS配置错误`

q100 的标准答案为 `Core_SW_01;VRRP工作在非抢占模式`。第 2、3 次正确；其余输出：

1. `Core_SW_01;VRRP Master角色规划不合理`
2. 正确
3. 正确
4. `Core_SW_01;STP 端口下实例根路径开销规划不合理`
5. `Core_SW_02;VRRP Master角色规划不合理`

特别需要关注 q100 第 1 次：推理中已经识别到 `preempt disable`，但最终选用了“Master 角色规划不合理”label。这说明主要短板不只在证据发现，还在“证据到标准 label”的受控映射。

## Thinking 与 events.jsonl

Codex CLI 的 `exec --json` 输出本身未包含模型已生成的 reasoning，但对应 Codex session rollout 保留了原始 reasoning。runner 现在会在每次任务结束后按 thread id 找到 rollout，将 reasoning 与下一条实际消息或命令精确对齐，再原子写回 `events.jsonl`；若请求了 thinking 却没有捕获到 reasoning，该次运行会以 capture failure 结束。

本轮 10/10 次均完成捕获，共回填 392 个 reasoning 节点、847,307 个字符；392 个节点全部与实际动作精确对齐（212 个命令、180 条消息），没有使用顺序兜底。每条 reasoning 记录都保存来源 rollout、原始 item id、文本 SHA-256 和对齐方式，`metadata.json` 同步更新事件计数与文件摘要。注意 reasoning 在单次任务结束后的后处理阶段写入，因此运行中途查看 `events.jsonl` 仍可能暂时看不到。

provider 统计仍显示 `reasoning_output_tokens=0`，这是当前接口的统计缺口，不代表没有 thinking；应以实际 reasoning 节点及其摘要校验为准。

## Prefix cache 与告警审计

本轮缓存输入 token 为 21,064,512 / 21,743,492，聚合命中率 96.88%，vLLM 结束时显示约 96.9%。Qwen3.6 在当前 vLLM 中属于混合 Mamba 架构，`align` 模式支持仍被标记为 experimental；本轮未观察到缓存错误、HTTP 5xx 或结果串扰。

启动日志中的高频 LoRA 告警来自视觉模块没有匹配的 Punica wrapper；当前任务为纯文本，且 LoRA 训练目标不包含这些视觉层，因此按无害告警记录。另有 Blackwell `SymmMemCommunicator` 不支持而回退、首次推理 JIT 编译和退出清理告警，均未导致运行失败。脚本已避免向 vLLM 子进程泄漏自定义 `VLLM_LOG` 环境变量，并显式设置 `OMP_NUM_THREADS=1`，消除两条可直接修复的启动告警；模型 `generation_config.json` 覆盖采样参数为有意保留。

## CUDA 评估

远端为 RTX PRO 6000 Blackwell Server Edition（compute capability 12.0）、驱动 580.95.05、驱动支持 CUDA 13.0，运行环境为 PyTorch 2.11.0+cu130 和 vLLM 0.25.1。当前软件栈已经是 CUDA 13 wheel，驱动也满足 CUDA 13 的最低要求；系统中另存的 `/usr/local/cuda-12.8` 不在本次 Python wheel 推理链路中。因此不建议为了本实验临时升级 CUDA。后续可在独立维护窗口评估驱动补丁升级，但不是本轮性能或准确率问题的修复项。

## 文件

- `attempts.csv`：10 次运行的逐次指标、答案和错误类型
- `validation_summary.json`：结构化配置、总体/分题统计和逐次结果
- `report.md`：由汇总脚本生成的简表
