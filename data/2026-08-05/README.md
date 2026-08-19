# 2026-08-05 因果路径聚类原生多轮 SFT

本目录基于补跑后的
`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` 新建，不修改
`data/2026-08-04/`。来源共有 84 道完成题，每题 10 条严格正确轨迹，共 840 条；
其余 q41–q56 没有正确轨迹，不进入本目录。

基础设施失败和人为中断不记录、不统计、不聚类，也不进入 SFT。来源中的模型有效
attempt 为 1342 条，其中 accepted 840、incorrect 502、format error 0。

本README用于概览；权威的逐步生成流程、固定参数、输入/输出、哈希、排错信息覆盖率、
loss占比、复现命令和变更要求见[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。以后对0805
来源、划分、转换、loss、采样、协议、tokenizer、训练入口或生成文件的任何修改，都必须
在同一变更中更新两份文档、根README和manifest，重生成全部派生文件并通过独立校验。

## 训练/验证划分

题目级划分完全复用 0804，禁止把同题轨迹拆到两侧：

| 集合 | 题数 | 来源成功轨迹 | SFT 节点 |
| --- | ---: | ---: | ---: |
| 训练 | 72 | 720 | 1430 |
| 验证 | 12 | 120 | 245 |
| 合计 | 84 | 840 | 1675 |

语义训练池包含1430个唯一节点。719个planning、reasoning和真实错误候选排除节点组成固定
core；711个路径端点组成全量endpoint pool。每个epoch对72个训练query各选2条路径，并把
每条路径的`evidence_summary → decision_ready → decision`三节点作为不可拆分的端点组，
共采样144个路径组、432条端点；与core合并后每轮有效曝光1151条。五轮按路径round-robin
轮换，覆盖全部237条路径；同题各路径五轮累计曝光次数最多相差1。验证集保留全部40条路径
的120条端点，不参与训练采样。

验证题仍为 q2、q12、q19、q20、q29、q38、q65、q71、q85、q86、q99、q100。
六个 label 各保留两题；`全局STP未使能` 没有成功率 100% 的题，继续使用 0804
已经确认的 q12、q2 回退规则。

## 因果路径聚类与节点选择

0804 快跑版每题只选择一条 best-1 轨迹；0805 改为同时分析同题的 10 条成功轨迹：

1. 从真实事件中提取“设备 × 协议/证据类型”的有序取证路径。
2. 删除重复、失败、基础设施、目录 housekeeping、低价值 CPU/内存/告警和与最终归因
   弱相关的动作；每个调查节点最多保留 2 个最有区分度的真实成功命令。
3. 按因果路径相似度聚类。优先保留有多个成员的路径；单例只有质量不低于本题中位数且
   与已保留路径不冗余时才可进入，每题最多保留 4 类路径。
4. 每类选择证据落地质量最高的代表轨迹，再对不同代表轨迹的等价推理节点去重。
5. 在每个保留路径簇的全部成员中查找可见原始排错句；仅当该句之前已有1–2个成功工具结果
   能作为证据锚点时，保留至多1个`hypothesis_elimination`节点。结论逐字包含来源排错句，
   metadata记录来源消息、候选排除句、证据action ID、实际观察和哈希；没有真实案例时不补造。
6. 每条保留路径都增加 1 个证据归纳节点和 1 个显式停止判断节点。归纳节点继承该路径最后
   一批工具结果，停止节点继承归纳结果；不跨路径合并这两类终点。
7. 每条保留路径都连接 1 个经严格判题验证的 decision，使各种正确证据历史都能学习
   “归纳 → 停止 → 回答”；训练时不直接加载全量端点池，而由epoch采样表保证每个query
   每轮恰好选择2条路径并同步曝光三类端点，避免路径多的题获得更大总权重。

全量3505个原始可见推理节点形成433个路径簇；保留277个非冗余路径，并生成1675个
SFT 节点。每条路径各有一个证据归纳和停止节点；节点类型如下：

| 集合 | planning | reasoning | hypothesis_elimination | evidence_summary | decision_ready | decision | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练语义池 | 167 | 449 | 103 | 237 | 237 | 237 | 1430 |
| 验证 | 29 | 79 | 17 | 40 | 40 | 40 | 245 |

120条真实排错节点覆盖67题、120条路径；训练103条覆盖58/72题，验证17条覆盖9/12题，
训练排错结论103/103逐字唯一。每轮实际训练为719条core加432条均衡端点，三类端点各144条。
所有120条排错节点都能在来源可见消息中逐字找到排错句，并引用当前节点之前已返回的成功
工具证据；校验器禁止未来证据、虚构排错句和无证据排除。

q0001 的 10 条成功轨迹共有 47 个原始可见节点，聚为 6 类路径，最终保留 4 类、24 个
代表调查/端点节点并新增2个真实候选排除节点，共26个；其中“否定已验证 STP 根因后继续
转向 OSPF”的绕路节点已被删除。

聚类清单
[`curation/causal_path_clusters_per_case.json`](curation/causal_path_clusters_per_case.json)
记录每题全部簇、成员、代表轨迹、被合并节点和排除原因，便于后续领域审核。

## 消息与 loss 规则

- 每条样本的 `system` 消息不再使用转换器自拟的网络诊断提示词，而是逐字读取
  [`config/codex_qwen_model_catalog.json`](../../config/codex_qwen_model_catalog.json) 中
  `Qwen3.6-27B-trained.base_instructions`，与训练后 LoRA 在 Codex CLI Agent 评测时收到的
  system prompt 保持一致；manifest 和逐条 metadata 同时记录来源、模型 slug 与内容哈希。
- 每个有价值的当前推理节点生成一条原生多轮 SFT，而不是一条完整轨迹只生成一条数据。
- 历史阶段、真实工具调用和结果摘要会作为当前节点上下文继承；当前调用的结果只在下一
  节点输入中出现，避免未来证据泄漏。
- 每条保留路径的历史均按“调查及工具结果 → 证据归纳 → 停止判断 → 最终回答”闭合；训练
  使用按query均衡、跨epoch轮换的完整路径端点组，而不是固定重复归纳/停止或只轮换decision。
- `hypothesis_elimination`只来自原始可见Agent排错句，并要求已有工具回显支撑；其thinking
  保留来源文本，结论只显式组织为“候选排除 + 来源原句”；真实工具结果留在输入历史，证据
  action、观察和哈希留在metadata反查，不把自动摘取的回显行再次写进监督结论。
- 可见 Agent 分析只保留与证据路径对齐的句子；必要的桥接 thinking 明确标记为重建，
  不冒充模型不可见的原始 chain-of-thought。
- 当前`<think>`按来源分级监督：裁剪后的原始可见Agent分析权重0.60，自动生成的路径端点
  证据桥接权重0.20；固定调查桥接模板和固定最终桥接均为0，只作上下文。结论也按来源
  分级：原始可见结论和严格验证最终答案为1.0，真实来源排错结论为0.60，自动重建阶段结论
  为0.40，路径证据归纳和停止判断为0.20，当前工具调用为0.10；历史assistant/tool call、
  system、user、tool response均为0。按中英文混合文本近似token口径及第一轮端点组采样估算，
  有效loss约为thinking48.1%、结论/答案46.2%、工具调用5.7%。自动证据归纳与停止判断
  合计占25.0%的行、40.5%的当前目标token和23.8%的加权loss，低于旧版51.6%；真实候选
  排除节点占8.9%的行和约11.7%的加权loss。五轮自动归纳+停止范围为23.64%–23.89%，
  校验器以30%为硬上限。
- 证据归纳必须点名当前路径的设备/配置、引用实际工具回显并列出已检查的排除项；停止判断
  必须复述决定性证据与排除证据，明确说明继续取证不会改变最小根因集合。训练集归纳结论
  230/237 唯一，停止结论 210/237 唯一，任一结论最多逐字重复 4 次；出现 10 次以上的固定
  thinking 模板均由校验器强制要求 `loss=0`。
- 工具监督协议与 LoRA 的 Codex CLI Agent 评测对齐：工具名使用 `exec_command`，监督参数只含
  `cmd`，命令使用评测机已有的 Linux `cat`、`grep`、`find`、`head`、`tail`、`test`，路径统一为仓库内
  `saved_configs/...` 相对路径。`justification` 仅保留在 metadata 中用于审计，不参与工具调用
  loss，从而进一步降低工具文本权重。
- 转换器只接受已盘点的只读 PowerShell 形态；无法可靠映射的命令会直接终止生成，不会以
  错误协议或零权重回退静默混入。metadata 同时保存完整原始 PowerShell、去外层包装后的
  PowerShell、Linux `cmd`、转换类型及转换前后哈希。
- 历史工具结果保留相关原始证据摘录；目录/搜索结果会移除 PowerShell 表格、控制台折行和
  Windows 绝对路径，规范为 `find`/`grep` 的 Linux 相对路径结果，外层再改为 Codex CLI 的
  `Chunk ID / Wall time / Process exited / Original token count / Output` 结构。transport token 数
  为可复核的近似值，整条 tool response 只作上下文、不计入 loss。
- 训练必须使用 `--loss_scale default --is_binary_loss_scale false`。

## 0805正式实验方案（双卡DDP）

正式训练使用Qwen3.6-27B、LoRA rank 8 / alpha 32 / dropout 0.05和16,384 token。
训练由GPU 0、1上的两个DDP进程共同执行：每卡micro batch为1、梯度累积4步，因此全局
有效batch为`2 × 1 × 4 = 8`，与旧单卡累计8步的优化器更新语义一致；seed和data seed均为42，
每轮重新shuffle。五轮内部使用固定学习率，依次为`2e-5`、`1.5e-5`、`1e-5`、
`6e-6`、`3e-6`。每轮结束只计算一次eval loss并保留checkpoint，eval loss仅作诊断，
不参与checkpoint选择。

机器可读权威配置为`config/qwen36_0805_formal_training.json`，正式入口为
`scripts/train_qwen36_0805_causal_path_formal.sh`。入口固定
`CUDA_VISIBLE_DEVICES=0,1`与`NPROC_PER_NODE=2`，并把五轮拆成五个连续阶段：stage 1从基座
开始，stage 2–5用`resume_from_checkpoint`和`resume_only_model=false`恢复上一阶段完整Trainer
状态，同时把累计`num_train_epochs`设为2–5，使每次只新增一个epoch并加载对应
`train_endpoint_epoch_01..05`采样表。所有阶段使用constant scheduler和零warmup。

完整resume会恢复旧optimizer/scheduler状态，因此仅传新的`--learning_rate`并不充分。
`scripts/qwen36_0805_fixed_stage_lr_plugin.py`会在所有DDP rank的train/epoch/step开始时把
optimizer参数组、scheduler的`base_lrs`和`_last_lr`重设为当轮目标值；所有rank都执行
数值核验，但只有rank 0写入`control/learning_rate_audit.jsonl`，避免并发写文件。任一实际值
与目标不符都会立即终止训练。

训练完成后固定使用第3个epoch结束时的checkpoint，不运行6题×2次或任何Agent选点流程，
也不改选eval loss最低的epoch。该epoch-3 checkpoint直接在全部12道验证题上各运行5次，
形成60次全新结果。基础设施失败和人为中断不进入样本或分母；模型无有效答案、错误工具调用
或错误归因按模型失败计错。

`scripts/train_qwen36_0805_causal_path_quick.sh`仅保留与0804历史1 epoch快跑一致的
数据、静态校验和 tokenizer 冒烟入口，不替代上述正式 5 epoch 对比方案。训练前必须
通过目标 tokenizer 的 16,384 token 长度预检。

## 目录

```text
data/2026-08-05/
├── README.md
├── raw/                                      # 840 条规范化严格正确轨迹
├── curation/
│   ├── accepted_trajectory_selection.json
│   ├── causal_path_clusters_per_case.json
│   └── FILTER_REPORT.md
└── sft/
    ├── reasoning_causal_path_manifest.json
    ├── qwen3_6_27b_reasoning_causal_path_train.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_core.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_01.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_02.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_03.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_04.jsonl
    ├── qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_05.jsonl
    └── qwen3_6_27b_reasoning_causal_path_validation.jsonl
```

## 重新生成与校验

```powershell
python -B scripts/convert_accepted_only_100x10_to_sft.py `
  --output-root data/2026-08-05 --archive-only
python -B scripts/convert_0805_causal_path_reasoning_sft.py
python -B scripts/validate_0805_causal_path_reasoning_sft.py
```

在训练机上执行正式五阶段训练：

```bash
bash scripts/train_qwen36_0805_causal_path_formal.sh
```

可以用`TRAIN_ENV`、`MODEL_PATH`和`OUTPUT_DIR`覆盖默认训练环境、基座和输出目录。同一
`OUTPUT_DIR`存在合法epoch checkpoint时，入口验证`trainer_state.json`中的epoch边界后从
下一阶段继续；如果旧运行已经启动但没有形成checkpoint，则拒绝混写，必须改用新的输出目录。
入口会固化Git提交、环境版本以及配置、正式入口、manifest、core、五轮端点表、验证数据、
LR插件和模型目录全部文件的SHA256；resume前会逐项重新校验。

以下命令只执行与0804历史快跑相同的16K单轮冒烟；`TRAIN_EPOCH_INDEX`选择要冒烟的完整
路径端点组采样表：

```bash
TRAIN_EPOCH_INDEX=1 bash scripts/train_qwen36_0805_causal_path_quick.sh
```

quick入口一次只允许训练1个epoch，不读取上一checkpoint，也不会自动切换学习率；不得通过
手工连续运行`TRAIN_EPOCH_INDEX=1..5`来冒充正式续训。正式实验必须使用formal入口。

当前数据状态为 `auto_clustered_draft_requires_domain_review`；静态一致性检查已经通过，
正式训练结论仍应在领域抽检和目标 tokenizer 预检通过后产生。
