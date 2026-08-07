# 0804 best1 五 epoch SFT 与完整 Agent 验证

本目录归档 2026-08-05 至 2026-08-06 在 SeaTACLOUD 完成的 0804 best1 后续实验。
实验使用 84 条最佳成功轨迹转换出的原生多轮 SFT：训练 318 个节点样本、验证 53 个节点
样本；训练五个 epoch，并在每个 epoch 结束保存 checkpoint。随后先用固定六题各两次
Agent 结果选择 checkpoint，再在既定十二题上补齐每题五次。

原始 checkpoint、完整 events、Codex session、reasoning 和 vLLM/GPU 日志保留在远端：

`/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0804/0804-5epoch-20260805T114046Z`

本目录只提交紧凑的机器可读结果、逐次指标、LR 审计和复现控制脚本。

## 结论

- 五轮训练共 200 optimizer step，有效 batch 为 8；epoch 内学习率固定，依次为
  `2e-5`、`1.5e-5`、`1e-5`、`6e-6`、`3e-6`。
- 最低 SFT eval loss 位于 epoch 4 / checkpoint-160（`0.14917336`）；但固定六题的
  Agent 选择结果以 epoch 3 / checkpoint-120 最好（6/12，50%），因此最终部署 epoch 3。
- 最终十二题各五次，共 60 次：按题85包含式 OR 修正口径严格正确 24/60（40.00%）；
  原始旧标签报告为 23/60（38.33%）。无模型硬超时、无进入分母的基础设施失败；
  平均/中位/P95 耗时为 16.65/14.63/33.10 分钟。
- q19 的第 1、3 次是“模型 turn 正常完成，但没有可解析最终答案”，两次均按错误答案
  计分，不归类为基础设施失败，也不重试成更有利样本。
- 60/60 均从事件与 session 中捕获非空 thinking，共 2,443 个 reasoning item、
  5,083,176 个字符。`reasoning_output_tokens=0` 是当前本地 provider 的计数限制，不能据此
  推断没有 thinking。
- 输入 token 共 160,928,942，其中 155,484,448 为 cached input，聚合缓存占比约
  96.62%；输出 token 为 1,862,728。

## 五个 checkpoint

选择题固定为 q12、q20、q38、q71、q86、q100，每个 checkpoint 每题两次。

| Epoch | Step | 固定 LR | Eval loss | Agent 严格正确 | 平均耗时（分钟） | 选择 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 40 | `2.0e-5` | 0.23613213 | 3/12（25.00%） | 23.27 |  |
| 2 | 80 | `1.5e-5` | 0.16515934 | 4/12（33.33%） | 21.12 |  |
| 3 | 120 | `1.0e-5` | 0.15305212 | 6/12（50.00%） | 19.30 | 是 |
| 4 | 160 | `6.0e-6` | 0.14917336 | 4/12（33.33%） | 13.47 |  |
| 5 | 200 | `3.0e-6` | 0.14932011 | 4/12（33.33%） | 13.50 |  |

五个 checkpoint 均无模型硬超时，60/60 checkpoint 选择 attempt 均捕获 thinking。
该结果明确说明最低 eval loss 与最佳 Agent 准确率并不一致。

## 最终逐题结果

| 题号 | 严格正确 | 准确率 |
| ---: | ---: | ---: |
| q2 | 2/5 | 40% |
| q12 | 0/5 | 0% |
| q19 | 0/5 | 0% |
| q20 | 5/5 | 100% |
| q29 | 1/5 | 20% |
| q38 | 0/5 | 0% |
| q65 | 1/5 | 20% |
| q71 | 1/5 | 20% |
| q85 | 5/5 | 100% |
| q86 | 4/5 | 80% |
| q99 | 2/5 | 40% |
| q100 | 3/5 | 60% |

用于 checkpoint 选择的六题最终为 13/30（43.33%），未参与选择的六题按修正口径为
11/30（36.67%）。正式结论采用全部 60 次；分组数值仅用于暴露 checkpoint 选择偏差。

## 重试和计分口径

基础设施失败会先被移动到带 `.infra_failed_<timestamp>` 后缀的隔离目录，再对同一题次
重试；隔离目录不进入 SFT、最终轨迹或准确率分母。本次共保留六个基础设施失败归档，
最终 60 个有效槽位中基础设施失败为 0。

vLLM Responses API 日志出现五次 `JSONDecodeError`。控制器没有把这些调用直接判错，
而是根据 manifest、exit code、事件完整性和 reasoning 捕获状态判定为基础设施问题后，
在同一题次重新开始干净 attempt。与此不同，q19 两次已产生 `turn.completed` 且没有基础
设施错误，只是缺少有效最终答案，因此保留并按错误计分。

## Thinking、缓存与 warning

- Codex 固定 `reasoning_effort=high`、`show_raw_agent_reasoning=true`；Qwen reasoning
  parser 固定为 `qwen3`。
- vLLM 开启 automatic prefix caching；末段日志命中率约 97%，汇总 token 口径为
  96.62%。未发现 KV preemption、CUDA OOM 或 NaN。
- vLLM 启动时报告 GPU capability 12.0 不支持 SymmMem communicator；运行自动使用其余
  通信路径，没有观察到功能失败。
- LoRA manager 对视觉模块报告 Punica wrapper 缺失并忽略视觉层。本实验是纯文本 Agent
  评测，LoRA 的文本层正常加载；该 warning 未观察到对本实验的直接影响，但后续多模态
  任务不能沿用这一结论。

## 对比边界

- 旧 0804 一 epoch 运行的 8/39（20.51%）使用了 fallback model metadata，因此不能与
  本轮修正后的 24/60 直接作为能力增益比较。
- 修正 metadata 后的一 epoch checkpoint-159 在 q12、q100 各五次为 2/10；本轮相同
  两题为 3/10。表面提升 10 个百分点，但样本只有十次，应视为小样本信号。
- 0731 LoRA 在另一组六题上为 12/30；本轮 checkpoint 选择六题为 13/30，但仅三题题号
  重合，不能把 1/30 的差异归因于本轮训练策略。

## 环境与溯源

- 训练数据 SHA-256 见 [`data_sha256.txt`](data_sha256.txt)。
- 训练环境：ms-swift 4.4.2、PyTorch 2.8.0+cu128、Transformers 5.12.1、PEFT 0.19.1、
  Accelerate 1.14.0，详见 [`environment.txt`](environment.txt)。
- 训练启动提交为 `a4f22e13a10e2b8d32868999f811f53997a18995`；Agent 评测使用已包含
  reasoning 归档修正的 `aa29f1430782b90b28944170e4009db6bfbcc803`。
- P2“0805 SFT 实验”没有启动：截至归档时，任何已 fetch 的远端分支均不存在已提交的
  `data/2026-08-05`、对应 README 或启动入口。为避免混用 0804 产物，保持等待规范。

## 文件

- [`training_summary.json`](training_summary.json)：五轮 eval loss 与训练完成状态。
- [`checkpoint_selection_summary.json`](checkpoint_selection_summary.json)：五个 checkpoint
  的固定六题 Agent 指标和选择键。
- [`validation_summary.json`](validation_summary.json)：最终 60 次按运行时旧标签生成的原始汇总。
- [`validation_summary_q85_inclusive_or.json`](validation_summary_q85_inclusive_or.json)：
  保留原始轨迹后，按题85包含式 OR 规则生成的修正汇总。
- [`attempts.csv`](attempts.csv)：60 次逐题准确率、耗时、token、thinking 和选择复用标记。
- [`p1_complete.json`](p1_complete.json)：P1 终态摘要。
- [`epoch_lr_audit.jsonl`](epoch_lr_audit.jsonl)：逐 step 实际 optimizer LR 审计。
- [`control/`](control/)：训练、checkpoint 选择、最终合并、GPU 空闲检查和故障恢复脚本。

可复用的 Agent 容错入口位于
[`scripts/run_agent_validation_resilient.sh`](../../scripts/run_agent_validation_resilient.sh)；
汇总器会把模型正常完成但无有效答案与真实基础设施失败分开统计。
