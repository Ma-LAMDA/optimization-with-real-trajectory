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

## 9. 正式训练扩展

数据完成审核并扩充后，再考虑：

- 3 epochs 起步并按验证集早停；
- 使用 42、43、44 三个种子评估波动；
- 双卡 DDP；
- 按 `case_id` 控制 0728 重复轨迹的采样权重；混合 0727 数据时再按 `target_type` 控制阶段样本权重；
- 根据验证集重新选择学习率、LoRA rank、全局 batch 和 max length。

任何训练脚本、数据或使用方式的变更在推送 GitHub 时，都必须在同一提交
中同步更新 `README.md`。
