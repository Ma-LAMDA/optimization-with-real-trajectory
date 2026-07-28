# Qwen3.6-27B epoch-10 LoRA：题 94 Codex CLI 五次验证报告

## 1. 结论

2026-07-28 使用完整题 94 user prompt、Codex CLI 和本地部署的
Qwen3.6-27B epoch-10 LoRA 连续串行运行 5 次：

- runner 成功率：5/5；
- 最终答案格式通过率：5/5；
- 与标准 label 严格集合匹配：4/5，准确率 80%；
- Run 1 命中正确项，但额外输出 `Core_SW_02;VRRP工作在非抢占模式`，
  因此按最小根因集合标准判错；
- Run 2–5 均只输出标准 label。

本轮没有执行原始基座模型的同条件 5 次 A/B 对照，所以只能报告训练后模型的
绝对结果，不能据此量化 LoRA 相对基座的提升。

## 2. 标准 label

标准答案来自题 94 的 `source_record.json`：

```json
[
  "Core_SW_01;VRRP工作在非抢占模式"
]
```

`source_record.json` SHA-256：

```text
40e39dbc66602e52baaa3c6aa2dc3c127de37fad13d786259f9eadb54610b4b4
```

## 3. 训练信息

| 项目 | 值 |
| --- | --- |
| 基座模型 | `/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B` |
| 训练方式 | LoRA SFT，BF16，单卡训练 |
| LoRA rank / alpha / dropout | 8 / 32 / 0.05 |
| target modules | `all-linear` |
| 训练样本 | 90 条，题 13、14、17、18、87、88、91、92、93 |
| 验证样本 | 10 条，全部为留出的题 94 |
| max length | 4096 |
| batch / gradient accumulation | 1 / 2 |
| learning rate | `5e-5`，cosine，warmup ratio 0.1 |
| epoch / step | 10 / 450 |
| seed | 42 |
| 训练耗时 | 949.8 秒 |
| 最终 train loss | 0.07722 |
| epoch-10 validation loss | `5.2280279305705335e-06` |
| epoch-10 validation token accuracy | 1.0 |
| 最佳 checkpoint | epoch 7、step 315，validation loss `3.884348643623525e-06` |
| 本次 Codex 验证模型 | epoch 10、step 450 |

训练集 SHA-256：

```text
54cf41db1519a60a17c69ea276c0a5f9ff132c175968e4306cfbeccec224c7f5
```

验证集 SHA-256：

```text
6c4639010442aa186029216556dbda2493ffe14910fb1f2c70cfd8251893be95
```

训练输出：

```text
/root/autodl-tmp/optimization-with-real-trajectory/output/qwen36-27b-reasoning-lora-0728-10epoch-20260728T041717Z
```

## 4. Codex 与推理服务配置

| 项目 | 值 |
| --- | --- |
| Codex CLI | 0.145.0 |
| 模型名 | `Qwen3.6-27B-trained` |
| API | OpenAI-compatible Responses API |
| sandbox | `danger-full-access` |
| vLLM | 0.25.1 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| tensor parallel | 2 |
| dtype | BF16 |
| max model length | 262,144 |
| GPU memory utilization | 0.90 |
| reasoning parser | `qwen3` |
| tool-call parser | `qwen3_coder` |
| LoRA | `checkpoint-450`，max rank 8 |
| 执行方式 | 5 次串行，每次 1 个 attempt |

训练和验证开始时的仓库提交为 `0f1441bb`。该提交之后的实验目录重构不改变
本报告引用的外部验证产物。

## 5. 逐次结果

“工具 loop”按已完成的 `command_execution` 计数；Agent 消息段按
`agent_message` 计数。Input token 是 Codex 多轮模型调用的累计输入，会重复
计算历史上下文，不能解释为原始 user prompt 的单次长度。

| Run | 严格匹配 | 耗时 | Input token | Output token | 端到端输出速度 | 工具 loop（成功/失败） | Agent 消息段 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 否，多报 Core_SW_02 | 458.126 秒 | 488,044 | 14,485 | 31.62 token/s | 22（21/1） | 17 |
| 2 | 是 | 449.790 秒 | 970,742 | 11,562 | 25.71 token/s | 31（30/1） | 17 |
| 3 | 是 | 686.751 秒 | 1,124,521 | 19,801 | 28.83 token/s | 33（32/1） | 7 |
| 4 | 是 | 580.252 秒 | 601,702 | 18,470 | 31.83 token/s | 18（18/0） | 11 |
| 5 | 是 | 440.225 秒 | 565,020 | 13,514 | 30.70 token/s | 24（24/0） | 16 |

汇总：

- 总耗时：2,615.144 秒，即 43 分 35.144 秒；
- 平均耗时：523.029 秒；中位数：458.126 秒；
- 耗时标准差：107.833 秒；变异系数：20.62%；
- 累计 input/output token：3,750,029 / 77,832；
- 加权端到端输出速度：29.76 token/s；
- 工具 loop：128，其中成功 125、失败 3；
- 平均工具 loop：25.6；平均 Agent 消息段：13.6；
- 5 次均无 error event、非法 JSONL、重试或 credit-limit 事件。

3 个失败命令均为读取或 grep 不存在的 `Vlanif100` 配置，未导致 runner 失败。

## 6. vLLM 周期日志

以下速度来自 vLLM 每 10 秒一次的周期日志，不等同于逐请求精确 TTFT/TPOT：

| Run | vLLM 生成均速 | 生成峰值 | Prompt 吞吐峰值 | KV cache 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 31.49 token/s | 38.9 token/s | 5,598.3 token/s | 2.7% |
| 2 | 25.69 token/s | 38.8 token/s | 4,944.1 token/s | 3.0% |
| 3 | 28.70 token/s | 39.5 token/s | 5,105.3 token/s | 3.1% |
| 4 | 31.84 token/s | 38.6 token/s | 5,740.1 token/s | 3.5% |
| 5 | 30.71 token/s | 38.9 token/s | 7,293.3 token/s | 2.6% |

5 次运行的 prefix cache 命中率均为 0%，`cached_input_tokens` 也均为 0。
`reasoning_output_tokens=0` 表示本地 provider 没有单独上报 reasoning token，
不能解释为模型没有推理。

## 7. 原始最终输出

### Run 1

```xml
<result>
[
"Core_SW_01;VRRP工作在非抢占模式",
"Core_SW_02;VRRP工作在非抢占模式"
]
</result>
```

`final_answer.txt` SHA-256：
`35219ac5a68dcc35af97047a7eb4cb53ac00716b149d4a696c046373ecd26b97`

### Run 2

```xml
<result>
[
"Core_SW_01;VRRP工作在非抢占模式"
]
</result>
```

`final_answer.txt` SHA-256：
`986d9023b85d6cb1883795c46bc78734f07a6b53ca3f4dea8f35349dc05a9751`

### Run 3

```xml
<result>
[
"Core_SW_01;VRRP工作在非抢占模式"
]
</result>
```

`final_answer.txt` SHA-256：
`6610d6dee08c31186be4cdec6d8ee8aeff80934daeb617bb43786555c9761557`

### Run 4

```xml
<result>
[
"Core_SW_01;VRRP工作在非抢占模式"
]
</result>
```

`final_answer.txt` SHA-256：
`92ba676b75285d184160f1dce30a8d3b86ed22482df7243ed0b8943fbfe7fa3d`

### Run 5

```xml
<result>
[
"Core_SW_01;VRRP工作在非抢占模式"
]
</result>
```

`final_answer.txt` SHA-256：
`9ee67cb8a287f46928a96f197b44e8021bc8e8250c904be6daea4ca9a4862139`

## 8. 产物与复现信息

验证目录：

```text
/root/autodl-tmp/qwen-codex-eval/2026-07-28/q94-0728-epoch10-20260728T043741Z
```

manifest：

```text
/root/autodl-tmp/qwen-codex-eval/2026-07-28/q94-0728-epoch10-20260728T043741Z/manifest.json
```

manifest 记录的数据集和模板 SHA-256：

```text
dataset:  79f961a2ce788fa2219e8ee5343b7fa87ca8d79ed3f3dec6049dca0ff7514ad9
template: 16cf68369b2f0ae4df90b35a89a6c07846d539cebeaf78f4700eb1ec2d02d7ce
prompt:   4d4a5f580e4f9709c90303332286831fe963c66b880ed835d7184ff4cd1fe0f1
```

当前仓库重构后的对应实验输入位于：

```text
experiments/2026-07-27-ip_codex_train0629_14x10/inputs/
```

## 9. 局限与后续对照

- 题 94 未进入 90 条训练集，但作为 10 条 validation 样本参与训练期评估；
- Codex CLI 验证使用完整调查 prompt 和离线配置工具，和直接 SFT
  validation loss 的测试口径不同；
- 样本仍为 `draft`，不能替代领域专家审核；
- 当前缺少逐请求 TTFT、TPOT、prefill/decode 拆分、每 loop token、
  GPU 时间序列和实际 sampling 参数；
- 下一轮应按 `docs/TRAINING_PLAN.md` 的遥测规范，让原始基座和 LoRA
  使用相同 prompt、版本、硬件和采样参数交错运行，各不少于 5 次。
