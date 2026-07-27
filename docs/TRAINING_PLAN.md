# Qwen3.6-27B 推理、计划与决策 SFT 训练方案

## 1. 目标与边界

本方案使用 `data/sft/qwen3_6_27b_reasoning_decision_sft.jsonl` 对
`Qwen3.6-27B` 进行 LoRA SFT，使模型学习：

- 在证据不足时提出下一步需要核验的事实；
- 根据新增证据形成阶段判断；
- 在证据充分时输出最小故障根因集合；
- 保持 `<think>`、普通响应和 `<result>` 的既定格式；
- 不把工具名、命令、API 路径或工具调用协议学入回答。

当前数据仅用于训练链路冒烟测试，不作为可证明能力提升的正式训练集。

## 2. 已核对的基线

### 数据

- 来源轨迹：3 条，分别为 `q0014`、`q0017`、`q0018`；
- 阶段样本：12 条；
- 类型分布：7 条 `planning`、2 条 `reasoning`、3 条 `decision`；
- 审核状态：12 条均为 `draft`；
- 划分：训练 12 条，验证 0 条；
- 三条来源轨迹的最终标签相同。

使用本地 `Qwen3.6-27B` tokenizer 统计，完整聊天序列为 980–1976
tokens，均值约 1513 tokens；assistant 监督部分为 95–133 tokens。
因此冒烟训练先使用 `max_length=2048`，但开始训练前必须确认
ms-swift 没有删除或截断样本。

### 服务器

- GPU：2×NVIDIA RTX PRO 6000 Blackwell Server Edition，每卡约 96 GiB；
- 基座模型：`/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B`；
- 仓库：`/root/autodl-tmp/optimization-with-real-trajectory`；
- 训练框架：独立环境中的 `ms-swift==4.4.2`；
- 精度：BF16；
- 当前 vLLM 使用两张 GPU，训练前必须先安排停机并确认显存释放。

## 3. 阶段一：数据准入

冒烟训练前：

1. 运行 `scripts/convert_trajectories.py` 重新生成数据；
2. 运行 `scripts/validate_sft.py`，要求全部检查通过；
3. 记录 Git 提交、数据文件 SHA-256、模型路径和软件版本；
4. 检查 ms-swift 预处理后的有效样本数仍为 12；
5. 如果 `max_length=2048` 删除任何样本，停止训练并改为 2560。

正式训练前还必须：

1. 由网络领域专家审核标注，将 `review_status` 改为 `reviewed`；
2. 补充不同设备、故障类型、正确配置与难负例；
3. 按 `source_id` 分组划分训练、验证和测试集；
4. 禁止把同一来源轨迹拆出的阶段样本随机分到不同集合；
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

训练使用本地模型，不重复下载基座权重。开始前记录：

```bash
swift --version
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
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
| max length | 2048 |
| per-device batch | 1 |
| gradient accumulation | 2 |
| global batch | 2 |
| learning rate | `5e-5` |
| scheduler | cosine |
| warmup ratio | 0.1 |
| epochs | 1 |
| gradient checkpointing | 开启 |
| validation split | 0 |
| seed | 42 |

执行入口：

```bash
cd /root/autodl-tmp/optimization-with-real-trajectory
source /root/autodl-tmp/envs/qwen36-sft/bin/activate
bash scripts/train_qwen36_lora_smoke.sh
```

12 条样本、梯度累积 2 时，预计每个 epoch 约 6 个优化步骤。这一轮只验证：

- 模型、模板和数据能够正确加载；
- 有效训练样本仍为 12；
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
- 有效样本数小于 12；
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

## 8. 正式训练扩展

数据完成审核并扩充后，再考虑：

- 3 epochs 起步并按验证集早停；
- 使用 42、43、44 三个种子评估波动；
- 双卡 DDP；
- 按 `source_id` 与 `target_type` 控制采样权重；
- 根据验证集重新选择学习率、LoRA rank、全局 batch 和 max length。

任何训练脚本、数据或使用方式的变更在推送 GitHub 时，都必须在同一提交
中同步更新 `README.md`。
