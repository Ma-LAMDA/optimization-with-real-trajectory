# Qwen3.6-27B 网络故障轨迹与 SFT

本仓库用于构造、训练和评测网络故障诊断轨迹数据。当前保留三批日期数据：

- `2026-07-27`：人工策展的多阶段小样本基线；
- `2026-07-31`：首轮 100×10 正确轨迹决策 SFT，也是现有训练方案的默认数据；
- `2026-08-04`：GPT-5.6-Sol accepted-only 轨迹及每题最佳一条的原生多轮 SFT 快跑集。

旧版 decision SFT 只保留可复核的规划、推理或最终决策；0804 原生多轮 SFT 另在独立
`tool_call`/`tool_response` 角色中保留对归因有价值的真实命令和结果。所有新 SFT 样本
当前均为 `draft`，正式使用前仍需领域审核。

> **Thinking 强制策略**：自 2026-08-04 起，所有新的 Base/LoRA Agent 评测与
> 对比实验必须显式开启 thinking，并在结果中记录 `reasoning_output_tokens`。
> 完整规则见 [`docs/THINKING_POLICY.md`](docs/THINKING_POLICY.md)；未观察到可见
> thinking 输出的历史运行不得与 thinking-on 运行混合为同一能力结论。

> **Codex 模型 metadata 前置校验**：Agent 验证会从
> [`config/codex_qwen_model_catalog.json`](config/codex_qwen_model_catalog.json) 为当前
> vLLM/LoRA 模型名生成隔离的运行内 catalog，并显式传给 Codex CLI。未知模型名不得
> 使用 fallback metadata；启动事件中出现 `Defaulting to fallback metadata` 时，该次
> 能力评测无效，必须修正后重跑。
> 可在vLLM就绪后设置 `MODEL_METADATA_SMOKE_ONLY=1` 调用
> `scripts/run_agent_validation.sh`；控制器只执行一个不调用工具的最小Codex turn，确认
> catalog被加载且事件流无fallback warning，不会启动正式题目验证。

## 当前数据

| 目录 | 用途 | 规模与划分 | 状态 |
| --- | --- | --- | --- |
| [`data/2026-07-27/`](data/2026-07-27/) | 多阶段策展基线 | 3 条原始轨迹；7 planning、2 reasoning、3 decision | 保留 |
| [`data/2026-07-31/`](data/2026-07-31/) | 当前 LoRA 训练基线 | 819 decision；训练 759、验证 60 | 已训练、已评测 |
| [`data/2026-08-04/`](data/2026-08-04/) | accepted-only 归档及 best1 多轮快跑集 | 814 decision；best1 84 轨迹、371 节点（训练 318、验证 53） | 数据已校验、GPU 快跑已归档 |
| [`data/simulation/`](data/simulation/) | 原始仿真资料 | prompt、JSONL、配置与评测轨迹 | 不可变来源 |

### 2026-07-31 划分

来源为
[`experiments/2026-07-28-ip_codex_train0629_100x10/`](experiments/2026-07-28-ip_codex_train0629_100x10/)。
819 条严格正确轨迹按 `case_id` 整题隔离；六种故障类型各留出一题
（12、24、40、72、86、100），形成 759/60 训练验证划分。完整筛选口径见
[`data/2026-07-31/README.md`](data/2026-07-31/README.md)。

### 2026-08-04 划分

来源为
[`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/)。
日期归档只统计 1,254 个模型有效 attempt：814 accepted、440 incorrect、0 format error；
基础设施失败与中断不进入日期归档。814 条 accepted 轨迹全部通过二次答案、事件、哈希和
证据清洁检查。

验证集按六种故障类型各留两道完整题。五类只选择成功率 100% 且有 10 条 accepted
轨迹的题；`全局STP未使能` 没有合格题，按显式回退规则选择成功率最高的 q12、q2。
完整候选、回退规则和逐题统计见
[`data/2026-08-04/README.md`](data/2026-08-04/README.md)。

0804 快跑版暂不对同题的 10 条轨迹聚类，而是在每个训练题和验证题中各选择一条证据
最充分、路径较短的最佳成功轨迹，再把每个有价值的推理节点生成一条原生多轮 SFT。
共选择 84 条轨迹，得到训练 318、验证 53 个节点样本。reconstructed `<think>` 的 token
loss 权重为 0.4，阶段结论、实际工具调用和最终结果为 1.0，历史轮次为 0；工具结果仅作
上下文。绕路、重复、失败和无关命令被删除，证据已收敛的无调用节点保留为
`decision_ready`，不会补造工具调用。该规则只作用于 0804，不修改 0731 数据与记录。

## 数据规则

- `data/simulation/` 是不可变来源，只允许读取或复制，不得编辑、覆盖、移动或删除。
- 新的日期归档只记录模型有效结果：`accepted`、`incorrect` 和 `format_error`。
- 基础设施失败与中断可供 runner 临时控制流程，但不进入日期归档、报表或训练数据。
- 训练/验证必须按 `case_id` 整题隔离，禁止把同题重复轨迹随机分到两侧。
- accepted 样本必须通过参考答案、独立判题、最终事件、文件哈希和证据清洁检查。
- 旧版 decision SFT 的 assistant 输出不得包含工具协议、工具名、命令、URL、API 路径
  或文件路径；0804 原生轨迹 SFT 只允许在独立 `tool_call`/`tool_response` 角色中保留
  对最终归因有因果价值的真实命令和结果，且工具结果不参与 loss。

## 常用命令

### 校验现有数据

```powershell
python scripts/validate_sft.py
python scripts/validate_100x10_sft.py
python -B scripts/validate_accepted_only_100x10_sft.py
python -B scripts/validate_0804_best_trajectory_reasoning_sft.py
```

### 重新生成日期数据

```powershell
python scripts/convert_trajectories.py
python scripts/convert_100x10_accepted_to_sft.py
python -B scripts/convert_accepted_only_100x10_to_sft.py
python -B scripts/convert_0804_best_trajectory_reasoning_sft.py
```

已删除的 2026-07-28 历史留一数据仍可从保留的 14×10 来源实验重建：

```powershell
python scripts/convert_codex_run_trajectories.py
python scripts/validate_codex_run_sft.py
```

旧版 SFT 校验兼容 Git 工作区中的 LF/CRLF 换行差异；如果 2026-07-28 数据尚未重建，
Codex-run 校验器会明确要求先运行转换器或通过 `--data-root` 指定数据目录。

### 训练

默认训练方案使用 2026-07-31 的 759/60 划分：

```bash
bash scripts/run_seetacloud_lora_workflow.sh
```

0804 每题最佳一条的 16K、1 epoch 快跑使用独立入口，不读取或改写 0731：

```bash
bash scripts/train_qwen36_0804_best1_quick.sh
```

该入口在启动训练前会重新生成数据、执行独立静态校验，并使用训练机上的目标 tokenizer
逐条确认没有样本超过 16,384 token；未通过预检时会直接退出。

SeaTACLOUD 上的端到端入口会在 GPU 空闲检查通过后完成同一训练，读取全部 validation
history 选择 `eval_loss` 最低且 checkpoint 仍存在的步，然后以单个 TP=2 vLLM 实例部署
该 LoRA。最终使用 Codex CLI 完整 Agent 工具循环，在 12 道整题隔离验证题上各运行 5 次，
固定 `REASONING_EFFORT=high` 并记录 `reasoning_output_tokens`：

```bash
bash scripts/run_seetacloud_0804_best1_workflow.sh
```

环境、LoRA 参数、早停、部署和恢复流程统一记录在
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)，根 README 不再重复维护服务器路径和
逐步操作说明。

## 当前结果

### LoRA SFT

2026-07-31 的 759/60 数据从基座模型重新训练 2 epochs，最低验证 loss 出现在最终
`checkpoint-760`。确定性生成验证连续执行五次，每次均为 49/60 严格匹配、60/60
格式正确且无工具信息泄漏；合计 245/300（81.67%）。该结果用于训练流程和稳定性检查，
不是独立随机测试集结论。

详细训练参数和逐类错误见
[`docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md`](docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md)。

### 完整 Agent A/B

当前完整 Agent 对比在题 12、24、40、72、86、100 上各运行五次：

| 指标 | Base | LoRA checkpoint-760 +100 |
| --- | ---: | ---: |
| 严格正确 | 7/30（23.33%） | 12/30（40.00%） |
| 平均封顶耗时 | 32.37 分钟 | 24.21 分钟 |
| 超时 | 3 | 4 |

LoRA 严格准确率提高 16.67 个百分点且典型耗时下降，但超时没有改善。逐题结果、运行拓扑
和原始汇总见
[`experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/`](experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/)。

### 0804 best1 快跑

0804 best1 原生多轮数据完成 1 epoch、159 step LoRA SFT；eval loss 从 step 40 的
`0.3065788` 持续下降到 step 159 的 `0.1806803`，因此选择最终
`checkpoint-159`。Codex CLI Agent 验证显式使用 `reasoning_effort=high`，原计划 12 题
各 5 次；按用户指令，在当时在途的 q12、q19 第 4 次完成后停止，最终执行 39/60，
严格正确 8/39（20.51%），模型硬超时 6 次，基础设施失败 0，剩余 21 次未启动且不计
失败。2026-08-05复核发现39/39个Codex事件流均因served model名称未登记而使用fallback
metadata；这些数值只保留为原运行记录，不能再作为0804与0731的有效能力对比，修正后需
重新验证。完整逐题结果和可复现合并脚本见
[`experiments/2026-08-04-qwen36-27b-best1-agent-validation/`](experiments/2026-08-04-qwen36-27b-best1-agent-validation/)。

下一轮0804 best1实验已固定为5 epochs：单卡micro batch为1、梯度累积为8（有效batch
为8），每个epoch内部使用固定学习率，五轮依次为`2e-5`、`1.5e-5`、`1e-5`、
`6e-6`、`3e-6`。每轮结束保存checkpoint；固定使用q12、q20、q38、q71、q86、
q100（每个label一题，其中q12、q86、q100与0731重合）各运行2次完整Agent来选择
checkpoint。入选checkpoint随后在全部12题上达到每题5次：上述6题复用挑选阶段的2次
并各补3次，其余6题各运行5次，最终仍汇总60次。完整选择与计数规则见实验目录README
及[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md#11-0804-best1下一轮5-epoch实验已确认未执行)。

## 评测约定

- 最终答案必须能解析为题目要求的 `<result>...</result>` JSON 列表。
- 严格正确要求预测与一个完整可接受答案精确匹配；漏报、多报和错报均计错。
- 模型未能在硬上限内完成时按错误计；基础设施失败和人为中断不进入评测报表，未启动
  槽位也不计失败。不能把“请求完成”当作“回答正确”。
- Qwen3.6-27B Base/LoRA Agent eval 固定单个 vLLM TP=2 实例、两个 runner、总并发 2。
- 默认单次上限 3,600 秒，最大生成长度 8,000 个新 token。
- validation loss 用于选点，不单独作为能力提升结论；正式对比必须使用相同题目、prompt、
  工具链、并发、超时和判分口径。

基座部署 A/B 与历史全量评测保存在
[`experiments/2026-07-31-qwen36-27b-base-eval/`](experiments/2026-07-31-qwen36-27b-base-eval/)。

## 目录导航

```text
.
├── data/
│   ├── 2026-07-27/
│   ├── 2026-07-31/
│   ├── 2026-08-04/
│   └── simulation/
├── docs/
│   ├── TRAINING_PLAN.md
│   └── 训练与评测结果报告
├── experiments/
│   ├── 2026-07-27-ip_codex_train0629_14x10/
│   ├── 2026-07-28-ip_codex_train0629_10x10/
│   ├── 2026-07-28-ip_codex_train0629_100x10/
│   ├── 2026-07-31-qwen36-27b-base-eval/
│   ├── 2026-08-02-ip_codex_gpt56-sol_100x10/
│   ├── 2026-08-02-qwen36-27b-heldout6-agent-ab/
│   └── 2026-08-04-qwen36-27b-best1-agent-validation/
└── scripts/
    ├── 数据转换与校验
    ├── LoRA 训练
    └── Base/LoRA Agent 评测
```

## 归档与清理状态

- `data/2026-07-27/` 保留：它是唯一包含 planning/reasoning 目标的人工策展基线。
- `data/2026-07-28/` 已删除：100 条 decision 样本已被更大数据替代，并可从来源实验重建。
- `experiments/2026-07-31-qwen36-27b-agent-ab/` 已删除：其四道题进入过训练集，不能作为
  泛化结论，已由六题完整 Agent A/B 替代。
- 07-27 14×10、07-28 10×10、两轮 100×10 来源实验继续保留，用于来源审计和复现。
- `2026-07-31-qwen36-27b-base-eval` 继续保留，因为部署决策和后续脚本仍引用该基线。

## 维护规则

凡调用本地 Qwen3.6-27B 基座或 LoRA adapter 服务进行的 eval，固定使用单实例
双并发：只启动 1 个 vLLM 实例，当前双卡部署采用 `tp2x1`；固定 2 个 eval
runner worker，总请求并发为 2。所有评测样本采用连续补位调度：任一 runner 结束后立即从
队列启动下一个样本，重试也必须复用已有槽位。禁止在 27B eval 中启动 8 个 worker、8 路
请求或自动扩容。该约束不适用于
Codex 轨迹生成及其他数据采集任务；数据采集策略由各实验独立配置。完整约束见
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)。

## Codex CLI 验证遥测

使用完整 user prompt、Codex CLI 和本地模型进行多次验证时，应同时保留原始事件流、服务日志和逐次遥测。必须区分 Codex turn、Responses API 调用、Agent 消息段与工具 loop，并记录 TTFT、TPOT、每轮 token、采样参数、上下文峰值、工具耗时、GPU 时间序列、KV cache、prefix cache、错误重试和严格 label 判定。缺失指标统一写为 `null`，不得以 0 或其他计数替代。

训练效果结论必须来自原始基座与 LoRA adapter 的同条件 A/B；两组各运行不少于 5 次，并报告逐次结果、准确率、false positive/negative、均值、中位数、P95、标准差和变异系数。字段定义、计算公式和推荐的 `telemetry.json` schema 见 [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md#9-codex-cli-多次验证遥测规范)。

2026-07-28 使用 epoch-10 LoRA、完整题 94 prompt 和 Codex CLI 串行验证 5 次：runner 与格式均为 5/5 成功，严格 label 匹配为 4/5；其中 Run 1 多报 Core_SW_02。本轮逐次最终输出、label、文件哈希、耗时、token、工具 loop、vLLM 吞吐、训练配置和适用边界见 [`docs/2026-07-28_Q94_EPOCH10_VALIDATION.md`](docs/2026-07-28_Q94_EPOCH10_VALIDATION.md)。由于未执行原始基座同条件对照，该结果不用于量化 LoRA 相对提升。

## 数据说明

- 0727 数据只有 3 条来源轨迹、12 条阶段样本，不单独划分验证集。
- 0728 数据包含 90 条训练样本和 10 条验证样本；按题号整组留一，禁止将同题重复运行拆到两个集合。
- 0731 数据包含 759 条训练样本和 60 条验证样本；默认作为当前 SFT 输入，按 6 种
  合并故障类型分别留出题 12、24、40、72、86、100。
- 题 25、26、27、28 的原始运行只用于审计，不进入训练或验证数据。
- 三条来源轨迹的最终标签相同。正式数据集必须补充不同设备、故障类型、正确配置和难负例，避免模型记忆固定答案。
- 同一轨迹拆出的阶段样本高度相关，训练时应按来源轨迹控制采样权重，避免长轨迹过度影响模型。
- 推送到远端 GitHub 前，应检查配置、地址和内部系统信息是否已经完成脱敏与授权。

### IP 故障分析仿真资料

`data/simulation/` 保存 IP 故障分析仿真的原始提示词、两组 GPT 评测轨迹、配置归档和训练 JSONL。资料包括：

- `ChatGPT system prompt.txt`、`Claude Code system prompt.txt`、`IP user prompt.txt` 和 `IP user prompt by text.txt`；
- `myf-ip评测0725-GPT_轨迹.zip`、`myf-ip评测GPT-0725-2_轨迹.zip` 和 `saved_configs.rar`；
- `train_data_0610.jsonl`（350 条记录）和 `train_0629.jsonl`（100 条记录）；后者与本次 Codex 实验归档的输入副本字节一致。

`Claude Code system prompt.txt` 的来源文件当前为空，仓库按来源状态原样保留。

`data/simulation/` 中的文件均视为不可变原始资料：只允许读取或复制到其他位置，不得编辑、覆盖、重命名、移动或删除。具体协作约束见 [`AGENTS.md`](AGENTS.md)。

### Codex 批量轨迹生成

`experiments/2026-07-27-ip_codex_train0629_14x10/` 集中保存本次实验的输入、生成脚本、完整运行，以及耗时与准确率统计。完整运行直接位于 `results/runs/`，包含题号 13、14、17、18、25、26、27、28、87、88、91、92、93、94 各 10 条成功轨迹，共 140 条。

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py
```

实验仍从仓库根目录的共享 `saved_configs/` 快照读取离线配置。完整目录结构、轨迹文件说明、重试规则、恢复命令和耗时 CSV 生成方式见 `experiments/2026-07-27-ip_codex_train0629_14x10/README.md`。

`experiments/2026-07-28-ip_codex_train0629_10x10/` 保存使用本地 Codex CLI、`gpt-5.6-sol`
和 `saved_configs_service` 本地 HTTP API 重新生成的 10×10 实验。题号为
13、14、17、18、87、88、91、92、93、94，每题保留 10 条有效成功轨迹，共 100 条；
故障集合精确匹配标准答案后为 96/100 正确，准确率 96%。该目录包含完整事件流、最终回答、
运行/策略审计、逐题准确率 CSV、逐轨迹判分明细和审计工作簿；具体结构与复核命令见
`experiments/2026-07-28-ip_codex_train0629_10x10/README.md`。

`experiments/2026-07-28-ip_codex_train0629_100x10/` 保存覆盖 100 条输入、每题最多
10 条正确轨迹的完整实验。历史运行共保留 819 条 accepted 正确轨迹：79 题完成
10 条正确轨迹，21 题在连续 10 次错误后停止；全部 attempt、事件流、回答、判题结果
和运行审计均原样归档。该实验属于数据采集，历史运行及后续恢复均使用独立采集策略，
不受 Qwen3.6-27B eval 的单实例双并发约束。819 条 accepted 轨迹已经转换为
`data/2026-07-31/` 下的严格正确 SFT 数据。

`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` 是基于
`IP user prompt by text.txt` 的新一轮本地 Codex CLI + `gpt-5.6-sol` 全量蒸馏。
该目录同时保留原提示词副本和优化提示词；优化版将配置根目录明确为
`saved_configs/`，说明 `<项目>/<节点>/<命令回显>.txt` 的三级目录与文件名转换规则，
并要求生成器直接列目录、搜索和读取本地文件；HTTP/API 读取被禁止，标准答案仍由安全
输入边界隔离。实验覆盖全部 100 题，每题只收录
独立严格判题正确的 10 条轨迹；连续错误达到 10 次或累计错误达到 20 次时停止该题，
基础设施失败不计入这两个阈值。运行状态、accepted 唯一映射和恢复方法见该实验的
[`README.md`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/README.md)。2026-08-03 账号切换
检查点曾归档 accepted 18 / 1,000；随后确认实际 user prompt 存在问题，旧 `results/`
已整体作废并删除。实际 prompt 已在 2026-08-03 完成本地文件读取版优化，之后从 q0001
attempt 1 全新启动，不恢复旧断点。实验已于 2026-08-04 完成，100 道题全部到达终态，
共保留 814 条 accepted 轨迹：79 题收齐 10 条正确轨迹，19 题因连续 10 次错误停止，
2 题因累计 20 次错误停止；最终完整性审计通过。新运行强制采用 accepted-only 保留策略：
**失败或中断结果一律不保留**；错误、格式错误、基础设施失败、超时和中断只保留必要的
状态计数，不归档、不提交、不长期保留其事件流、回答、日志或 attempt 目录。重置状态、
固定的 Standard 速度/初始及最大并发 10 配置及启动清单见
[`HANDOFF.md`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/HANDOFF.md)。
2026-08-05 又仅对已有部分成功的题 3、7、21、22、23 以并发 4 补跑，零成功题不再尝试；
五题分别新增 4、5、1、9、7 条 accepted 后全部达到 10 条，使全实验 accepted 总数增至
840，完成 10 条的题目增至 84 道，其余 16 道零成功题保持原终态。补跑结束时剩余额度
38%，最终审计再次通过。

三个 Codex 轨迹实验已采用统一的紧凑归档：`prompt.txt` 和
`source_record.json` 按“实验 + 题号”各保留一份；100×10 实验只保留
`events.jsonl` 作为 Codex 原始标准输出流，并将共享 hooks 配置集中到
`config/hooks.json`。迁移删除 7,412 个重复文件、新建 249 个规范文件，净减少
7,163 个文件项和 971,494,909 字节（926.49 MiB），不修改事件、答案或判题证据。
逐实验统计见
[`experiments/ARCHIVE_COMPACTION_REPORT.json`](experiments/ARCHIVE_COMPACTION_REPORT.json)；
可用 `python scripts/compact_experiment_archives.py` 只读复核。

2026-07-30 至 2026-07-31 的 Qwen3.6-27B 基座部署 A/B 和全量评测已作为独立实验
归档到 [`experiments/2026-07-31-qwen36-27b-base-eval/`](experiments/2026-07-31-qwen36-27b-base-eval/)。
目录将部署对比与全量结果分开保存，并提供总体、逐题和逐次明细。

checkpoint-760 +100 与历史 base 的完整 Agent A/B 已归档到
[`experiments/2026-07-31-qwen36-27b-agent-ab/`](experiments/2026-07-31-qwen36-27b-agent-ab/)。
该实验复用 base 的 20 次 TP2 结果，只实跑 LoRA 侧；所有运行固定单实例 TP2、双并发
和 60 分钟上限。

checkpoint-760 +100 与 Base 在最新留出题 12、24、40、72、86、100 上的同条件完整 Agent
A/B 已归档到 [`experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/`](experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/)：
LoRA 严格正确 12/30（40.00%），Base 为 7/30（23.33%），提升 16.67 个百分点；两侧分别有
4/3 次超时，均无非超时 runner failure。LoRA 平均/中位封顶耗时为 24.21/13.74 分钟，Base
为 32.37/26.81 分钟。两侧题目、prompt、工具链、次数、3600 秒上限与单实例双并发拓扑一致，
该结果作为当前最新留出划分的正式端到端泛化 A/B 结论。

## 提交维护规则

每次创建并推送 GitHub 提交时，必须在同一个提交中同步更新本 README，记录该次变更对项目内容、数据、脚本或使用方式的影响。

### 更新记录

- 2026-08-05：修正0804 Agent验证的Codex模型metadata缺失问题；验证控制器现在为
  当前served model生成运行内catalog，避免未知LoRA名称回退到10,000-byte工具输出截断、
  非并行工具调用和通用基础指令。
- 2026-08-05：完成 0802 GPT-5.6-Sol 100×10 实验中题 3、7、21、22、23 的定向补跑，
  五题全部补满 10 条 accepted；新增 26 条轨迹后全实验 accepted 总数为 840，84 道题
  达到 10 条，最终完整性与输入隔离审计通过。
- 2026-08-03：清理 0802 实验中未被引用的 smoke 输入、Python 字节码、空目录和
  约 337 MiB 的可重建 Codex CLI 副本，并合并 `.gitignore` 中已被 `/runtime/` 覆盖的
  重复规则；正式运行会自动重建所需 runtime 和输入索引。
- 2026-08-03：进一步简化 0802 实验 prompt，删除对其他读取方式和只读快照属性的
  重复强调，保留 `saved_configs/` 路径、三级目录解析、文件名转换和必要调查步骤；
  运行侧的文件访问边界保持不变。
- 2026-08-03：重写 0802 GPT-5.6-Sol 100×10 实验 prompt，将配置访问从本地 API
  改为直接只读 `saved_configs/` 文件，补充项目、节点、命令回显文件的目录解析规则，
  并同步切换输入边界、运行 hook、最终审计和交接文档；尚未启动新一轮采集。
- 2026-08-01：增加 27B base 全量 Agent eval 的单实例 TP2、双并发重跑入口；重新
  覆盖历史相同的 92 题×5 次范围，明确与旧 8-worker 轨迹隔离，并保留断点恢复能力。
- 2026-08-01：完成 checkpoint-760 +100 与历史 base-eval 的同条件完整 Agent A/B；
  LoRA 严格正确率 15/20（75%），较复用 base 的 3/20（15%）提升 60 个百分点，
  平均耗时由 26.66 分钟降至 13.95 分钟，20 次均无超时；归档逐次结果并明确该四题
  已进入训练集，正式泛化验收仍使用当前六题留出集。
- 2026-07-31：将训练工作流的最终验证改为复用 base-eval 的完整 Codex Agent
  runner、调查 prompt、离线工具和严格判分；最新 6 个留出题默认各跑 5 次，固定
  单实例 TP2、双并发、单次 60 分钟，并增加复用历史 base 20 次结果的 checkpoint
  端到端 A/B 入口。训练期 SFT validation loss 仍仅用于早停和 checkpoint 选择。
- 2026-07-31：在当前 60 条验证集上完成原始 27B 基座的 5 次双并发验证，
  扫描原训练 step500/600/700/760，并从 checkpoint-760 独立续训
  +100/+200 steps；严格正确率从基座均值 7.33% 提升至 86.67%，推荐 +100。
- 2026-07-31：训练脚本支持只加载既有 LoRA 权重并以独立优化器执行指定
  `MAX_STEPS` 的额外训练，用于在原调度器已经衰减结束后复现实验性 step 扩展。
- 2026-07-31：增加原始 27B 基座与多个 LoRA checkpoint 的统一验证扫描脚本；
  同一单实例 TP=2 服务中，基座默认按双并发重复 5 次，checkpoint 默认各验证
  1 次，并汇总严格正确率及相对基座变化。
- 2026-07-31：按最新 759/60 分层划分完成 2-epoch LoRA SFT，最低验证 loss
  位于 checkpoint-760；在同一单实例 TP=2 服务中完成 5 次双并发验证，每次严格
  匹配 49/60，并归档按题号、故障类型和稳定错误模式的分析。
- 2026-07-31：压缩三个 Codex 轨迹实验归档，删除与 `events.jsonl` 完全相同的
  1,313 份 `stdout.log`，并按题号集中 prompt/source record、集中共享 hooks；
  同步更新 metadata、转换器、校验器和后续 runner，净释放 926.49 MiB。
- 2026-07-31：将 0731 的 819 条严格正确轨迹重新按故障类型分层并以题号整组划分，
  每种故障类型确定性留出 1 题，生成 759 条训练样本和 60 条验证样本；同步固化
  六类覆盖、题号隔离、选择规则和报告校验。
- 2026-07-31：完成 Qwen3.6-27B LoRA SFT 实跑，选定最低验证 loss 的
  checkpoint-600，并在单实例 TP=2、双并发验证中取得严格匹配 10/10；固化实际
  参数、结果、远端路径和训练/后处理提交分离的溯源规则。
- 2026-07-31：固化 SeetaCloud LoRA SFT 端到端工作流，增加按 `eval_loss`
  早停与最佳 checkpoint 校验，并以单实例双并发在固定验证集上执行格式、严格集合
  匹配及泄漏评测；支持安全复用已经完成的训练状态继续后处理。
- 2026-07-31：从 100×10 实验的 1,313 个 attempt 中过滤 819 条独立判题完全正确轨迹，生成 809 条训练和 10 条题 100 验证 SFT 数据，并增加可复现转换与独立校验脚本。
- 2026-07-31：合并 `taowen` 的 `saved_configs_service`、10×10 与 100×10 实验，保留完整轨迹和审计产物，并保持数据采集策略独立配置。
- 2026-07-31：将 Qwen3.6-27B eval 策略固定为单个 vLLM 实例、2 个 eval runner worker、总请求并发 2；该约束不适用于轨迹生成等数据采集任务。
- 2026-07-31：建立独立的 Qwen3.6-27B 基座评测实验目录，分开归档部署 A/B 与终止时的 381 个全量已结束样本，并提供 JSON、CSV、Markdown 统计。
- 2026-07-28：将题 94 epoch-10 LoRA 的 5 次实测结果同步到 Training Plan 和实验 README，补充运行命令、严格 4/5 判定、耗时/token/工具 loop、外部产物路径及基座 A/B 缺口。
- 2026-07-28：归档 Qwen3.6-27B epoch-10 LoRA 在题 94 上的 5 次 Codex CLI 验证报告，记录原始 label/输出、严格 4/5 结果、耗时、token、工具 loop、vLLM 指标、哈希与评测局限。
- 2026-07-28：增加 Codex CLI 多次验证遥测规范，统一 turn、API 调用、Agent 消息和工具 loop 口径，并规定 TTFT、TPOT、token、缓存、GPU、质量判定及基座/LoRA A/B 的记录要求。
- 2026-07-28：新增 `experiments/2026-07-28-ip_codex_train0629_10x10/`，归档通过本地 API 仿真环境重新生成的 100 条有效 Codex 轨迹、运行审计及 96% 准确率统计。
- 2026-07-28：将实验运行压缩为 `results/runs/fullaccess/q<题号>_r<轮次>/attempt_<序号>/`，合并重复的 case/run 层级，同时保留额度重试所需的 attempt 记录。
- 2026-07-28：将实验目录按“日期-实验名”合并命名为 `experiments/2026-07-27-ip_codex_train0629_14x10/`，移除 `results/runs/` 下的日期层，并同步适配生成、统计和 SFT 转换脚本。
- 2026-07-28：将最新 140 条 Codex 运行规范化到 `data/2026-07-28/`；排除准确率未达 100% 的题 25、26、27、28，并按题号留出题 94，生成 90 条训练和 10 条验证样本。
- 2026-07-28：将本次 Codex 实验使用的 `train_0629.jsonl` 原样复制到 `data/simulation/`，并增加仿真原始资料只读、只允许复制的保护规则。
- 2026-07-28：为 `raw`、`curation` 和 `sft` 增加统一的 `data/2026-07-27/` 日期层；`data/simulation/` 作为原始仿真资料保持原位不变。
- 2026-07-28：将 Train 0629 Codex 轨迹实验的输入、脚本、140 条完整轨迹、runner 日志和统计 CSV 统一整理到 `experiments/2026-07-27-ip_codex_train0629_14x10/`。
- 2026-07-27：新增指定 IP 题目的 Codex 批量执行脚本，每题执行 10 个成功轮次，共生成 140 条完整 JSONL 轨迹；额度不足时保留失败尝试，并每隔 30 分钟无限重试同一槽位。
- 2026-07-27：新增根目录 `saved_configs` 离线组网配置快照和 `IP user prompt with saved configs skills.txt`，用于按项目、节点与命令文件查询故障证据。
- 2026-07-27：向 `data/simulation/` 补充 IP 故障分析提示词、GPT 评测轨迹、配置归档及两份训练 JSONL，并记录文件清单与可解析记录数。
- 2026-07-27：将后续推理默认输出上限统一为 8,000 个新 token，并记录 ms-swift、vLLM 参数写法及原始基座模型的单样本速度基线。
- 2026-07-27：根据首次冒烟执行结果补充 `flash-linear-attention==0.5.1` 环境要求，并在训练脚本中增加启动前依赖检查。
- 2026-07-27：新增 Qwen3.6-27B LoRA 训练方案和单卡冒烟训练脚本，补充数据准入、逐分钟 loss 监控与验收要求，并将 ms-swift 4.x 参数更正为 `--tuner_type lora`。
- 2026-07-27：将一轨迹一决策样本扩展为多阶段样本；保留抽象的下一步核验计划，删除具体工具与执行方式，共整理 7 条 planning、2 条 reasoning 和 3 条 decision 样本。
- 2026-07-27：将生成目标升级为单一 `reasoning_decision` SFT；新增显式策展证据和推理标注，移除工具调用训练格式。
- 2026-07-27：建立 README 同步维护规则，并增加仓库级协作说明。

每次推送 GitHub 的提交都必须同步更新本 README，确保数据、脚本、实验和当前结论一致。
