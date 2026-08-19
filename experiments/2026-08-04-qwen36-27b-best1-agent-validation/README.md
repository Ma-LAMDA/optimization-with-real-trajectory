# 0804 best1 LoRA SFT 与部分 Agent 验证

本目录归档 2026-08-04 每题最佳一条原生多轮 SFT 的 SeaTACLOUD 快跑，以及按用户
指令提前停止后的有效 Agent 验证结果。训练完整结束；Agent 验证原计划 12 题各 5 次，
在 39 个 attempt 完成后停止，剩余 21 个未启动 attempt 不计失败。

> 2026-08-05事后复核：39/39个attempt的事件流开头均出现
> `Model metadata ... not found. Defaulting to fallback metadata`。原因是served model
> `Qwen3.6-27B-0804-best1`未登记到Codex model catalog，而0731模型已登记。以下结果
> 仍按原样保留，但只能表示错误Agent metadata条件下的历史运行，不能用于0804/0731
> 能力结论；修正控制器并通过无warning冒烟后需重新验证。

## Metadata修正冒烟

2026-08-05使用原`checkpoint-159`启动单个TP=2 vLLM实例，并通过修正后的验证控制器
执行一个不调用工具的最小Codex turn。运行内catalog成功加入
`Qwen3.6-27B-0804-best1`，SHA-256为
`32d0b87d93b49db9d446cb5767466b2beb65de1023b0224d7bfe0aab98316c48`；事件顺序为
`thread.started`、`turn.started`、返回`OK`、`turn.completed`，未出现fallback metadata
warning，Responses API返回HTTP 200。该冒烟只验证启动与metadata，不计入准确率，也没有
继续执行正式验证题。

## 训练结果

- 基座：Qwen3.6-27B；LoRA 可训练参数 58,363,904。
- 数据：训练 318 个节点样本、验证 53 个节点样本，最大长度 16,384 token。
- 训练：1 epoch、159 step；train loss `0.26836527`。
- eval loss：step 40=`0.3065788`、80=`0.1956932`、120=`0.1823089`、
  159=`0.1806803`。
- 最低 eval loss 位于 `checkpoint-159`，该 checkpoint 用于 Agent 验证。

## 后续5 epoch实验方案（已执行）

本节记录2026-08-05确认、随后于2026-08-06完成的方案，不追溯改写上面的1 epoch历史
快跑结果。五轮训练、五个checkpoint选择和12题×5次最终验证的独立归档见
[`experiments/2026-08-06-qwen36-27b-0804-best1-5epoch-agent-validation/`](../2026-08-06-qwen36-27b-0804-best1-5epoch-agent-validation/)；
最终选择epoch 3 / checkpoint-120，严格正确23/60（38.33%）。

### 训练参数

- 训练5个epoch；单卡`per_device_train_batch_size=1`，
  `gradient_accumulation_steps=8`，因此有效batch为8。
- 使用固定`seed=42`和`data_seed=42`，每个epoch重新shuffle；micro batch保持1以控制
  Qwen3.6-27B、16K上下文的显存占用。
- 不使用单个epoch内的step级cosine衰减和10% warmup。每个epoch内部学习率固定，
  epoch之间按下表阶梯衰减：

| Epoch | 固定learning rate |
| ---: | ---: |
| 1 | `2.0e-5` |
| 2 | `1.5e-5` |
| 3 | `1.0e-5` |
| 4 | `6.0e-6` |
| 5 | `3.0e-6` |

- 每个epoch结束计算SFT eval loss并保存一个checkpoint，保留全部5个epoch checkpoint；
  不再让Trainer仅凭最低eval loss自动加载最终模型。

### checkpoint Agent选择

固定从12题验证集中选择每个label一题：

| Label | 选择题 | 与0731重合 |
| --- | ---: | :---: |
| 全局STP未使能 | q12 | 是 |
| STP BPDU被过滤 | q20 | 否 |
| 存在IP路由环路 | q38 | 否 |
| 存在MPLS标签环路 | q71 | 否 |
| VRRP Master角色规划不合理 | q86 | 是 |
| VRRP工作在非抢占模式 | q100 | 是 |

这是当前0804验证划分在不引入训练题和不破坏既有筛选规则的前提下，与0731能够达到的
最大重合（3/6）。每个checkpoint在上述6题上各运行2次，共12个Agent attempt；5个
checkpoint合计60个挑选attempt。选择顺序固定为：

1. 严格准确率更高；
2. 模型硬超时更少；
3. 平均运行时间更短；
4. SFT eval loss更低；
5. 仍相同时选择更早的epoch。

所有运行显式使用`reasoning_effort=high`。正式题目开始前必须通过无fallback warning
的model metadata冒烟；模型硬超时计错，基础设施失败和人为中断不进入样本或分母。

### 最终12题验证与复用

入选checkpoint最终在q2、q12、q19、q20、q29、q38、q65、q71、q85、q86、q99、
q100上各有5次结果，共60个attempt。为减少重复计算：

- q12、q20、q38、q71、q86、q100复用该入选checkpoint在挑选阶段的2次，并各补跑
  3次，共补18次；
- q2、q19、q29、q65、q85、q99各运行5次，共30次；
- 因此选定checkpoint后新增48次，连同复用的12次组成最终60次；其他未入选
  checkpoint的挑选运行不进入最终结果。

最终报告必须标记12个`used_for_checkpoint_selection=true`的attempt，并同时报告全部
12题、参与选择的6题和未参与选择的6题三种汇总，以显式呈现checkpoint选择偏差。

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
