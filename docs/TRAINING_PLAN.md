# Qwen3.6-27B 推理、计划与决策 SFT 训练方案

## 1. 目标与边界

本方案默认使用
`data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_train.jsonl` 训练，并使用
`data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_validation.jsonl` 验证。
`data/2026-07-27/` 的 12 条人工策展多阶段样本继续作为独立基线保留。
目标是对 `Qwen3.6-27B` 进行 LoRA SFT，使模型学习：

- 在证据不足时提出下一步需要核验的事实；
- 根据新增证据形成阶段判断；
- 在证据充分时输出最小故障根因集合；
- 保持 `<think>`、普通响应和 `<result>` 的既定格式；
- 不把工具名、命令、API 路径或工具调用协议学入回答。

当前数据仅用于训练链路冒烟测试，不作为可证明能力提升的正式训练集。

## 2. 已核对的基线

### 数据

- 最新完整实验运行包含 14 道题、每题 10 条，共 140 条原始轨迹；
- 题 25、26、27、28 因准确率未达到 100% 而整题排除；
- 剩余 10 道题均为 10/10 正确，共形成 100 条 `decision` 样本；
- 训练集：题 13、14、17、18、87、88、91、92、93，共 90 条；
- 验证集：留出题 94，共 10 条；
- 划分键为 `case_id`，训练与验证题号交集为 0；
- 100 条样本均为 `draft`，正式训练前需要领域审核；
- 0728 样本完整消息文本为 1568–3401 个字符，均值约 2314 个字符；
- 0727 基线仍包含 7 条 `planning`、2 条 `reasoning` 和 3 条 `decision`。

训练脚本暂用 `max_length=4096`。开始训练前仍必须使用目标模型 tokenizer
重新统计 token 长度，并确认 ms-swift 没有删除或截断样本。

### 服务器

- GPU：2×NVIDIA RTX PRO 6000 Blackwell Server Edition，每卡约 96 GiB；
- 基座模型：`/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B`；
- 仓库：`/root/autodl-tmp/optimization-with-real-trajectory`；
- 训练框架：独立环境中的 `ms-swift==4.4.2`；
- 精度：BF16；
- 当前 vLLM 使用两张 GPU，训练前必须先安排停机并确认显存释放。

## 3. 阶段一：数据准入

冒烟训练前：

1. 运行 `scripts/convert_codex_run_trajectories.py` 重新生成 0728 数据；
2. 运行 `scripts/validate_codex_run_sft.py`，要求全部检查通过；
3. 记录 Git 提交、数据文件 SHA-256、模型路径和软件版本；
4. 检查 ms-swift 预处理后的有效样本数仍为训练 90、验证 10；
5. 如果 `max_length=4096` 删除任何样本，停止训练并重新检查模板与 tokenizer 统计。

正式训练前还必须：

1. 由网络领域专家审核标注，将 `review_status` 改为 `reviewed`；
2. 补充不同设备、故障类型、正确配置与难负例；
3. 保持按 `case_id` 分组的训练、验证划分；
4. 禁止把同一道题的重复运行随机分到不同集合；
5. 冻结独立评测集及其哈希，再开始调参。

## 4. 阶段二：环境准备

训练环境与 vLLM 推理环境隔离，避免依赖互相覆盖。建议位置：

```text
/root/autodl-tmp/envs/qwen36-sft
```

环境至少需要：

- `ms-swift==4.4.2`；
- 可支持 Blackwell GPU 的 PyTorch/CUDA 组合；
- Transformers 5.x；
- PEFT、Accelerate；
- `qwen_vl_utils>=0.0.14` 和 `decord`。
- `flash-linear-attention==0.5.1`，用于 Qwen3.6/Qwen3.5 的线性注意力层；
- Triton 3.3 或更高版本。

`flash-linear-attention` 官方要求 PyTorch 2.7 或更高版本及 Triton 3.3
或更高版本。缺少该依赖时，模型可以完成加载，但会在首个 forward
开始前退出，因而不会产生 loss。

训练使用本地模型，不重复下载基座权重。开始前记录：

```bash
swift --version
python -c "import torch, transformers, fla, triton; print(torch.__version__, transformers.__version__, fla.__version__, triton.__version__)"
nvidia-smi
```

## 5. 阶段三：单卡 LoRA 冒烟训练

首轮只使用一张 GPU，另一张保持空闲。27B BF16 基座在单张 96 GiB
GPU 上执行 LoRA；不使用 QLoRA，不启用双卡 DDP，也不启用 packing。

| 参数 | 值 |
| --- | --- |
| tuner | LoRA |
| dtype | BF16 |
| LoRA rank / alpha | 8 / 32 |
| LoRA dropout | 0.05 |
| target modules | `all-linear` |
| ViT / aligner | 冻结 |
| max length | 4096 |
| per-device batch | 1 |
| gradient accumulation | 2 |
| global batch | 2 |
| learning rate | `5e-5` |
| scheduler | cosine |
| warmup ratio | 0.1 |
| epochs | 1 |
| gradient checkpointing | 开启 |
| validation split | 固定题 94，共 10 条 |
| seed | 42 |

执行入口：

```bash
cd /root/autodl-tmp/optimization-with-real-trajectory
source /root/autodl-tmp/envs/qwen36-sft/bin/activate
bash scripts/train_qwen36_lora_smoke.sh
```

90 条训练样本、梯度累积 2 时，预计每个 epoch 约 45 个优化步骤。这一轮只验证：

- 模型、模板和数据能够正确加载；
- 有效训练样本仍为 90，验证样本仍为 10；
- loss 为有限值并能正常反向传播；
- 没有 OOM、NaN 或进程异常；
- checkpoint 和训练日志能够正常保存。

## 6. 每分钟监控

训练应通过独立日志文件启动。监控程序每 60 秒记录：

- 进程是否仍存活；
- 最新 step、epoch、loss、grad_norm 和 learning_rate；
- 两张 GPU 的显存与利用率；
- 是否出现 `CUDA out of memory`、`NaN`、`Traceback` 或训练结束标志。

若出现以下任一情况立即停止并保留日志：

- loss 为 NaN/Inf；
- OOM；
- 有效训练样本数小于 90，或验证样本数小于 10；
- 模型或模板识别错误；
- 非训练 GPU 出现意外占用；
- 连续多个监控周期没有 step 推进且进程无有效计算。

## 7. 冒烟训练后的评估

冒烟训练不以训练 loss 作为能力结论。至少比较基座和 LoRA adapter：

- 输出结构通过率；
- 工具名、命令和 API 泄漏率；
- planning 的信息需求覆盖率；
- reasoning 与证据的一致性；
- decision 根因集合的正确性与最小性；
- 未见样本上的领域专家盲评结果。

验收底线：

- 数据校验 100% 通过；
- 样本删除和截断为 0；
- 输出格式通过率 100%；
- 工具/API 泄漏率为 0；
- 独立评测不得比基座模型明显退化。

## 8. 推理生成参数

后续离线评估、交互推理和服务请求的默认输出上限统一为 8,000 个新
token。该值是允许生成的最大长度，不是要求模型必须生成的目标长度；模型
输出 EOS 时应立即结束，不强制补足到 8,000 token。

ms-swift/Transformers 推理使用：

```bash
--max_new_tokens 8000
```

vLLM 的 OpenAI 兼容请求使用：

```json
{
  "max_tokens": 8000
}
```

温度、top-p、top-k 等采样参数必须由具体任务显式指定，不与 8,000 token
上限绑定。确定性评估建议使用 `temperature=0`；需要观察随机采样行为时再
单独调整温度。

当前模型配置的上下文窗口为 262,144 token。每次请求仍须保证输入 token、
新生成 token 和模板/特殊 token 的总和不超过上下文窗口，并为模板与边界
token 预留余量。长输出会增加延迟和 KV cache 占用。

2026-07-27 在当前单卡 Transformers 环境中，以原始基座模型、
`temperature=0.7` 和 5,000-token 上限进行一次单样本测试：解码后输出约
4,860 token，生成阶段耗时 229 秒，约 21.2 token/s；包含模型加载的端到端
耗时约 257.6 秒。该结果仅作为当前硬件与软件环境的速度基线，不代表 8,000
token 输出会线性耗时。

### Qwen3.6-27B 评测运行拓扑

自 2026-07-31 起，凡调用本地 Qwen3.6-27B 基座或 LoRA adapter 服务进行的
eval，包括通过 Codex CLI 发起的 27B eval，一律采用单实例双并发策略：

- 只启动 1 个 vLLM 实例；当前双卡部署保持 `tp2x1`，即该实例使用
  `tensor_parallel_size=2`；
- 固定启动 2 个 eval runner worker，总请求并发固定为 2；
- 待运行样本进入队列，由两个 worker 消费；重试必须复用原有并发槽位，
  不得额外启动 worker 或产生槽位外请求；
- 禁止配置 8 个 worker、8 路请求或任何等效的 8 并发运行方式，也不允许
  runner 根据积压任务自动扩容；
- 每次运行的元数据必须明确记录 `instance_count=1`、`worker_count=2` 和
  `request_concurrency=2`，启动前后均需核对实际进程数与在途请求数。

除非后续计划被明确修订，否则不得偏离上述实例数和并发数。
该约束不适用于 Codex 轨迹生成或其他数据采集任务；数据采集的 worker 数、
并发调度和退避策略由对应实验计划与采集脚本独立规定。

## 9. Codex CLI 多次验证遥测规范

使用完整 user prompt、Codex CLI 和本地模型进行多次验证时，不能只记录最终
答案和总耗时。每次运行必须保存可复现标识、模型调用、Agent 循环、工具执行、
延迟、token、缓存、GPU 资源和质量判定。原始事件流、服务日志和汇总指标必须
同时保留，避免只有汇总值而无法追溯。

### 9.1 统一计数口径

以下四个计数不得混用：

- **Codex turn**：一次用户输入触发的外层 turn；一次 `codex exec` 通常为 1；
- **Responses API 调用数**：Codex 与模型服务之间实际完成的 HTTP 请求数；
- **Agent 消息段数**：事件流中的 `agent_message` 数；一个模型调用可能产生零个
  或多个消息段；
- **工具 loop 数**：事件流中的已完成 `command_execution` 数，必须进一步拆为
  成功、失败、超时和取消。

如果 runner 无法直接提供 Responses API 调用数，应将该字段记为 `null`，不得
用 Agent 消息段数或工具 loop 数替代。

### 9.2 每次运行必须记录的字段

#### 可复现标识

- run ID、case ID、repeat index、attempt index、Codex thread ID；
- 开始和结束时间，统一使用带时区的 ISO 8601；
- Git 分支、完整 commit SHA、工作区是否干净；
- 数据集、user prompt 模板、source record 和期望 label 的 SHA-256；
- 基座模型路径、adapter 路径、训练 epoch/step 和 checkpoint SHA-256；
- Codex CLI、vLLM、PyTorch、CUDA、驱动和 tokenizer 版本；
- API 类型、sandbox、tool parser、reasoning parser 和工作目录。

#### 采样与上下文

- `temperature`、`top_p`、`top_k`、`seed`、`max_tokens` 和停止条件；
- 模型上下文上限、首次请求输入 token、峰值上下文 token；
- 累计 input、cached input、cache-write input、output 和 reasoning token；
- 每个 Responses API 调用和每个工具 loop 前后的 token 增量；
- 是否发生上下文截断、自动摘要、重试或重新编码。

采样参数若由 Codex 或 provider 使用默认值，也必须解析并落盘实际值；确实无法
获取时写 `null`，不能只写“默认”。累计 input token 会重复计算多轮历史，不能
把它解释为原始 user prompt 长度。

#### 延迟与吞吐

- 端到端总耗时；
- TTFT（请求发出到首 token 的时间）；
- TPOT（首 token 后相邻输出 token 的平均时间）；
- prefill、decode、排队和模型服务时间；
- Shell/工具执行时间、模型等待时间以及二者占总耗时的比例；
- 每个模型请求和工具 loop 的开始、结束、耗时；
- prompt throughput、generation throughput 及其均值、中位数、P95 和峰值。

统一计算公式：

```text
end_to_end_output_tps = output_tokens / duration_seconds
effective_total_tps = (input_tokens + output_tokens) / duration_seconds
TPOT = (last_token_time - first_token_time) / (output_tokens - 1)
decode_tps = 1 / TPOT
```

`end_to_end_output_tps` 包含工具和文件 I/O 等待，`decode_tps` 才表示纯解码速度。
vLLM 周期日志给出的吞吐是时间窗口采样值，必须与逐请求精确计时分列展示。

#### Agent 与工具行为

- Codex turn、Responses API 调用、Agent 消息段和工具 loop 四类计数；
- 每个工具命令的类别、目标、退出码、耗时和输出字节数；
- 成功、失败、超时、取消、无匹配和重复命令数；
- 首次定位正确根因所在的 loop，以及定位后继续执行的冗余 loop；
- runner attempt、重试原因、error event、非法 JSONL 和 credit-limit 事件；
- 最终答案前是否已形成正确证据链。

#### GPU、缓存与服务资源

- 每张 GPU 的型号、显存总量、显存占用、利用率、温度、功耗和时钟；
- tensor parallel、dtype、最大上下文、GPU memory utilization 和 LoRA rank；
- GPU 指标至少每 1 秒采样一次，并与模型请求时间戳对齐；
- KV cache 峰值/均值，prefix cache 命中率及 cached token 数；
- vLLM running/waiting request 数、请求队列时长、OOM 和 engine reset；
- 并发数以及同一 GPU 上是否存在其他训练或推理任务。

#### 质量与稳定性

- runner 状态、退出码、最终答案格式是否有效；
- 解析后的预测 label、期望 label、严格集合匹配结果；
- false positive、false negative、重复 label 和非法 label；
- 根因集合最小性、证据一致性和人工复核结论；
- 多次运行的准确率、均值、中位数、最小/最大值、标准差、P95 和变异系数；
- 基座与 LoRA 的同条件 A/B 差值，禁止只用 LoRA 单组结果宣称提升。

### 9.3 最小落盘格式

每个 `attempt_XXX/` 除现有 `metadata.json`、`events.jsonl`、`stderr.log` 和
`final_answer.txt` 外，应新增 `telemetry.json`。建议最小结构如下：

```json
{
  "schema_version": "codex-sft-eval-telemetry.v1",
  "run_id": "q94-example/run_01/attempt_001",
  "model": {"base": "Qwen3.6-27B", "adapter": "checkpoint-450"},
  "sampling": {"temperature": null, "top_p": null, "seed": null, "max_tokens": 8000},
  "counts": {"codex_turns": 1, "responses_requests": null, "agent_messages": 0,
             "tool_loops": 0, "tool_success": 0, "tool_failed": 0},
  "tokens": {"input": 0, "cached_input": 0, "output": 0,
             "reasoning_output": null, "peak_context": null},
  "latency_ms": {"total": 0, "ttft": null, "tpot": null,
                 "prefill": null, "decode": null, "tools": null},
  "throughput_tps": {"end_to_end_output": 0, "decode": null,
                     "prompt_mean": null, "generation_mean": null},
  "cache": {"kv_peak_pct": null, "prefix_hit_pct": null},
  "quality": {"final_valid": false, "exact_match": false,
              "false_positive": [], "false_negative": []}
}
```

所有时间统一以毫秒落盘，吞吐统一为 token/s，比例统一为百分数。缺失指标使用
`null`，只有实测为零时才写 `0`。汇总 CSV 可以从 JSON 生成，但 JSON、事件流
和原始服务日志应作为事实来源保留。

### 9.4 A/B 验证要求

训练效果验证至少应让原始基座和 LoRA adapter 使用完全相同的 prompt、数据、
采样参数、Codex/服务版本与硬件配置，各运行不少于 5 次。两组运行顺序应交错或
随机化，避免温度、缓存和服务预热造成固定顺序偏差。报告必须同时给出：

- 严格准确率、false positive/negative 和输出格式通过率；
- 运行耗时、TTFT、TPOT、token、工具 loop 与失败命令；
- 均值、中位数、P95、标准差和变异系数；
- prefix cache 开启/关闭状态及命中率；
- 原始逐次结果和聚合结果，不能只报告最优一次。

### 9.5 已完成的题 94 epoch-10 验证

2026-07-28 已使用完整题 94 user prompt、Codex CLI 0.145.0 和本地部署的
Qwen3.6-27B epoch-10 LoRA `checkpoint-450` 串行运行 5 次。逐次原始输出、
标准 label、文件哈希、训练配置和完整遥测见
[`2026-07-28_Q94_EPOCH10_VALIDATION.md`](2026-07-28_Q94_EPOCH10_VALIDATION.md)。

| 指标 | 实测 |
| --- | --- |
| runner / 格式成功 | 5/5 / 5/5 |
| 严格 label 匹配 | 4/5，80% |
| 唯一错误 | Run 1 多报 `Core_SW_02;VRRP工作在非抢占模式` |
| 总计 / 平均耗时 | 2,615.144 / 523.029 秒 |
| 累计 input / output token | 3,750,029 / 77,832 |
| 加权端到端输出速度 | 29.76 token/s |
| 工具 loop | 128；成功 125、失败 3 |
| prefix cache | 5 次均为 0% 命中 |

本轮已经记录总耗时、累计 token、Agent 消息段、工具 loop、错误事件、
vLLM 周期吞吐、KV cache 峰值、最终输出和严格判分；尚未记录逐请求 TTFT、
TPOT、prefill/decode 拆分、Responses API 请求数、每 loop token、GPU 时间序列
和实际 sampling 参数。这些缺失字段不得事后推算为 0。

题 94 未进入 90 条训练集，但作为 10 条 validation 样本参与训练期评估。本次
Codex CLI 使用完整调查 prompt 和离线配置工具，与直接 validation loss 的口径
不同。由于没有执行原始基座的同条件 5 次对照，80% 只能作为 LoRA 绝对结果，
不能用于宣称相对基座提升。下一轮必须按 9.4 的要求补跑基座，并将两组顺序交错。

## 10. 正式训练扩展

数据完成审核并扩充后，再考虑：

- 3 epochs 起步并按验证集早停；
- 使用 42、43、44 三个种子评估波动；
- 双卡 DDP；
- 按 `case_id` 控制 0728 重复轨迹的采样权重；混合 0727 数据时再按 `target_type` 控制阶段样本权重；
- 根据验证集重新选择学习率、LoRA rank、全局 batch 和 max length。

任何训练脚本、数据或使用方式的变更在推送 GitHub 时，都必须在同一提交
中同步更新 `README.md`。
