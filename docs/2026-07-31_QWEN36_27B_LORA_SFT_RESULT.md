# 2026-07-31 Qwen3.6-27B LoRA SFT 结果

## 1. 结论

SeetaCloud 端到端工作流已完成。训练以验证集 `eval_loss` 为选择指标，
最低点出现在 step 600（epoch 1.4821），loss 为 `0.005798716563731432`。
训练按 patience 继续观察到 step 900 后早停，推理评测加载的不是最后一个
checkpoint，而是最低 loss 对应的 checkpoint-600。

checkpoint-600 在固定验证集的 10 条样本上全部请求成功，输出格式、严格
`<result>` 集合匹配和无泄漏检查均为 10/10。

## 2. 数据

| 项目 | 数量 |
| --- | ---: |
| 原始 attempt | 1,313 |
| 判题 accepted | 819 |
| 进入 SFT 的完全正确轨迹 | 819 |
| 训练样本 | 809 |
| 训练 case | 83 |
| 验证样本 | 10 |
| 验证 case | 题 100 |
| 训练/验证 case 交集 | 0 |

数据转换会重新扫描原始运行，过滤非 accepted attempt，并执行独立校验。训练和
验证按 case 分组，避免同题重复运行跨集合泄漏。manifest 的 LF 归一化 SHA-256
为 `eb1dda44a15f63ea39c2c80d7f6b9fd61eb1bb2fa4f853c869b0517cb23766f7`。

## 3. 训练配置

| 项目 | 配置 |
| --- | --- |
| 基座模型 | Qwen3.6-27B |
| 训练源码提交 | `bb2a819ea` |
| 训练设备 | 单张 NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 精度 | BF16 |
| LoRA | rank 8、alpha 32、dropout 0.05、all-linear |
| 可训练参数 | 58.3639M / 27,415.0925M（0.2129%） |
| 最大序列长度 | 4,096 |
| batch | per-device 1，gradient accumulation 2 |
| 学习率 | `5e-5`，cosine，warmup ratio 0.1 |
| 最大 epoch | 3 |
| 验证/保存间隔 | 100 step |
| 早停 | 连续 3 次验证无改进 |
| checkpoint 选择 | `eval_loss` 越低越好，训练结束加载最佳模型 |
| seed / data seed | 42 / 42 |

训练实际执行 900/1,215 step、2.2225 epoch，耗时 1,997.6478 秒（33 分
18 秒），汇总 train loss 为 `0.06577042`。训练环境为 ms-swift 4.4.2、
PyTorch 2.8.0+cu128、Transformers 5.12.1、PEFT 0.19.1 和
flash-linear-attention 0.5.1。

## 4. 验证 loss

| Step | Epoch | eval_loss |
| ---: | ---: | ---: |
| 100 | 0.2472 | 0.0158160273 |
| 200 | 0.4944 | 0.0357796326 |
| 300 | 0.7417 | 0.0116868522 |
| 400 | 0.9889 | 0.0214598905 |
| 500 | 1.2349 | 0.0090409992 |
| **600** | **1.4821** | **0.0057987166** |
| 700 | 1.7293 | 0.0276175644 |
| 800 | 1.9765 | 0.0059553948 |
| 900 | 2.2225 | 0.0088952258 |

早停无法在首次遇到最低点时预知后续结果，因此在 step 600 后继续执行三个验证
区间。step 700、800、900 均未低于 step 600，patience 耗尽后停止，并加载
checkpoint-600。

## 5. 生成评测

评测使用 vLLM 0.25.1，只启动一个服务实例；两张 GPU 组成 TP=2，服务的
`max_num_seqs=2`，评测脚本固定两个 worker 和总请求并发 2。没有启动 8
并发。Blackwell sm_120 环境设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，绕过
FlashInfer 0.6.13 的架构识别问题，不改变实例数、TP 或并发数。

采样固定为 `temperature=0`、`top_p=1`、`seed=42`、`max_tokens=8000`。

| 指标 | 结果 |
| --- | ---: |
| 请求完成 | 10/10 |
| 请求失败 | 0/10 |
| 格式正确 | 10/10 |
| 严格集合匹配 | 10/10 |
| 无泄漏 | 10/10 |
| 评测墙钟时间 | 49.2061 秒 |
| 单请求平均延迟 | 9.8404 秒 |
| 单请求中位延迟 | 2.4053 秒 |
| 单请求 P95 延迟 | 38.9758 秒 |

服务在评测完成后自动退出，两张 GPU 已释放。

## 6. 远端产物

运行根目录：

```text
/root/autodl-tmp/optimization-with-real-trajectory/output/qwen36-27b-lora-0731-20260731-bb2a819e
```

关键产物：

```text
train/v0-20260731-123211/checkpoint-600/
training_summary.json
validation_eval/validation_predictions.jsonl
validation_eval/validation_summary.json
workflow_summary.json
workflow-resume-flashinfer.log
vllm.log
```

这些文件包含模型权重、逐条输出和运行日志，保留在 SeetaCloud 的 `output/`
目录，不推送到 GitHub。

## 7. 工作流复现

新训练：

```bash
cd /root/autodl-tmp/optimization-with-real-trajectory
RUN_ID=<唯一运行编号> \
  bash scripts/run_seetacloud_lora_workflow.sh
```

若训练已经完成，只恢复摘要、部署或评测：

```bash
cd /root/autodl-tmp/optimization-with-real-trajectory
RUN_ID=20260731-bb2a819e \
REUSE_COMPLETED_TRAINING=1 \
TRAINING_GIT_COMMIT=bb2a819ea \
  bash scripts/run_seetacloud_lora_workflow.sh
```

新工作流会在训练输出目录自动保存训练源码提交；只有复用未带该记录的旧训练时
才需要手工给出 `TRAINING_GIT_COMMIT`。本次权重由 `bb2a819ea` 训练，最终验证
与工作流汇总由 `79dffb1d0` 执行，两者应分开解释。

## 8. 适用边界

题 100 的 10 条数据同时承担训练过程的早停/选点和最终生成验证，因此结果回答的
是“所选 checkpoint 在该验证集上的表现”，不是独立测试集上的泛化结论。后续若
要比较基座与 LoRA 的真实提升，应另行冻结不参与早停和调参的测试集，并执行同条件
A/B 评测。
