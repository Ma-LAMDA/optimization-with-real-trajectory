# 2026-08-05 SFT 生成与复现规范

本文档是0805数据的权威复现记录。`README.md`只提供概览；当数据、脚本与本文冲突时，
必须停止训练，修正不一致后重新生成和校验。

## 1. 当前版本与状态

- 数据范围：仅`data/2026-08-05/`，不修改0804和0731。
- manifest schema：`qwen36-0805-causal-path-reasoning-sft.v9`。
- cluster selection schema：`0805-causal-path-cluster-selection.v4`。
- 状态：`auto_clustered_draft_requires_domain_review`。
- 当前代码尚在本地工作树；在形成Git提交前，以本文列出的脚本、manifest文件哈希和独立
  校验结果共同定义版本。提交或推送时必须把脚本、数据和文档放在同一提交。
- 基础设施失败和人为中断不归档、不计数、不聚类，也不进入SFT或评测分母。

## 2. 不可变来源与输入

原始实验为：

```text
experiments/2026-08-02-ip_codex_gpt56-sol_100x10/
```

日期归档使用84道完成题，每题10条严格正确轨迹，共840条。模型有效attempt为1342条：
accepted 840、incorrect 502、format error 0；q41–q56没有正确轨迹，不进入0805。

关键输入：

```text
data/2026-08-05/curation/accepted_trajectory_selection.json
data/2026-08-04/curation/accepted_trajectory_selection.json
config/codex_qwen_model_catalog.json
data/2026-08-05/raw/
对应轨迹的原始event JSONL
```

当前输入标识：

```text
accepted_trajectory_selection SHA256(LF-normalized):
fc64bf0ca77e028eebc66415f9e84b46d9863cea8e7c4f698fc5264039c5c7ef

Codex CLI system prompt content SHA256:
f54052796d1557c843d758a5d8ea5a948205c10f7a4a6368b4d40e72ce4cb80b
```

`data/simulation/`以及归档后的来源文件是不可变输入，只能读取或复制。

## 3. 训练/验证划分

按`case_id`整题隔离，完全复用0804冻结划分：训练72题，验证12题。验证题为：

```text
q2 q12 q19 q20 q29 q38 q65 q71 q85 q86 q99 q100
```

六个故障label各保留两题。`全局STP未使能`没有成功率100%的题，沿用0804已记录的q12、
q2回退规则。同一道题的10条轨迹不得拆到训练和验证两侧。

这只是题目隔离，不是拓扑隔离；七个CampusNetwork配置项目在训练和验证两侧均有出现。
当前冻结验证集用于checkpoint横向比较，不能单独证明新拓扑泛化。真正的拓扑泛化结论需要
额外的topology-heldout测试集。

## 4. 生成流水线

### 4.1 归档严格正确轨迹

```powershell
python -B scripts/convert_accepted_only_100x10_to_sft.py `
  --output-root data/2026-08-05 --archive-only
```

只接纳同时通过参考答案匹配、独立判题、最终事件、来源文件哈希和证据清洁检查的轨迹。

### 4.2 解析可见消息和真实工具事件

转换器读取原始可见`agent_message`、真实工具命令、退出状态和结果。隐藏chain-of-thought
不可见，不得声称恢复。工具结果只作上下文，不参与loss。

### 4.3 清理和选择调查动作

固定参数：

| 参数 | 值 |
| --- | ---: |
| `MAX_RETAINED_PATHS_PER_CASE` | 4 |
| `MAX_ACTIONS_PER_STAGE` | 2 |
| `TRAJECTORY_CLUSTER_THRESHOLD` | 0.46 |
| `NODE_DUPLICATE_THRESHOLD` | 0.76 |

删除重复、失败、基础设施、目录housekeeping、低价值CPU/内存/告警和与最终归因弱相关的
动作。每个调查节点最多保留2个能够区分候选根因的真实成功命令。原始PowerShell命令和
哈希保留在metadata用于审计。

### 4.4 转换到评测工具协议

只允许已盘点的只读命令形态。监督协议统一为：

```text
tool name: exec_command
arguments: {"cmd": "..."}
OS: Linux
path: repository-relative saved_configs/...
```

支持的基础命令为`cat`、`grep`、`find`、`head`、`tail`、`test`。无法可靠转换的命令必须
使生成失败，不能静默回退。结果规范成Codex CLI的
`Chunk ID / Wall time / Process exited / Original token count / Output`外形，去除PowerShell
表格、折行和Windows绝对路径。

### 4.5 因果路径聚类和节点去重

以“设备 × 协议/证据类型”的有序序列作为路径特征。同题10条成功轨迹先聚类；优先保留
多成员簇，单例只有质量不低于本题中位数且不与已保留路径冗余时才保留。每题最多4条路径。
每簇选择质量最高的代表轨迹，再对跨路径等价调查节点去重。

当前3505个原始可见节点形成433个路径簇，保留277条非冗余路径。

### 4.6 thinking和阶段结论

原始可见thinking按设备、协议、证据、假设、排除和路径相关句子裁剪。没有可保留原句时，
固定桥接只作上下文，loss为0。结论按来源分级：

| 内容来源 | loss scale |
| --- | ---: |
| 裁剪后的原始可见thinking | 0.60 |
| 自动路径端点证据桥接thinking | 0.20 |
| 固定调查桥接 | 0 |
| 固定最终桥接 | 0 |
| 原始可见结论 | 1.00 |
| 自动重建阶段结论 | 0.40 |
| 真实来源错误候选排除结论 | 0.60 |
| 路径证据归纳结论 | 0.20 |
| 路径停止判断结论 | 0.20 |
| 严格验证最终答案 | 1.00 |
| 当前工具调用 | 0.10 |
| 历史assistant/tool call | 0 |
| system/user/tool response | 0 |

训练必须使用：

```text
--loss_scale default --is_binary_loss_scale false
```

### 4.7 错误候选排除信息

生成器在每个保留路径簇的全部真实成员中查找含“排除、不支持、不能解释、不成立、不是
根因、并非根因、暂不支持、可排除、反证”等标记的可见原始thinking句。一个路径只有同时
满足以下条件时才新增1条`hypothesis_elimination`：

1. 排错句能够在来源`agent_message`中逐字找到，且thinking来源标记为
   `pruned_original_visible_agent_message`。
2. 当前节点之前已有1–2个非目录发现类的成功工具结果；证据协议类型或真实回显关键词必须
   与排错句重合，结果必须含可复核观察，不能只是“命令成功返回”。
3. 证据action ID能够反查到同一原始轨迹的更早阶段，不能引用当前或未来工具结果。
4. 每个保留路径至多选择1条；优先证据更完整、排错表达更具体、轨迹质量更高的真实节点。
5. 没有满足条件的真实排错案例时不补造，不用最终答案反向生成一个“错误候选”。

监督thinking保留来源可见文本，结论只组织为“候选排除 + 来源排错原句”。真实工具回显
已经位于该节点输入历史中；metadata另行记录`rejected_candidate_statements`、
`elimination_evidence_action_ids`、证据描述、实际观察、来源消息位置与哈希，用于验证其确实
来自更早证据。这样不会把自动摘取的单行回显再次写入loss。真实来源排错结论loss为0.60。

当前全量120条，覆盖67/84题和120/277条路径；训练103条覆盖58/72题，验证17条覆盖
9/12题。训练排错结论103/103逐字唯一。其余157条路径没有满足上述严格条件的真实排错
节点，因此保持缺失而不合成。该规则提高的是“如何用反证排除错误根因”的真实监督覆盖，
不代表已经穷举10条轨迹中的所有候选假设。

### 4.8 路径终点

每条保留路径依次增加：

```text
evidence_summary → decision_ready → decision
```

证据归纳必须包含当前设备/配置项、实际工具回显和排除项。停止判断必须说明决定性证据、
排除证据以及继续取证为何不会改变最小根因集合。每条路径都连接严格验证的最终答案。

当前全量277条路径各有一个归纳、停止和decision；训练237条，验证40条。训练集归纳结论
230/237唯一，停止结论210/237唯一，任一归纳/停止结论最多逐字重复4次。出现10次以上的
固定thinking模板必须由校验器确认loss为0。归纳与停止的自动thinking和结论均使用0.20，
避免自动端点文本在训练loss中压过原始推理与严格答案。

### 4.9 query均衡路径端点组采样

训练语义池保留全部237条路径的`evidence_summary`、`decision_ready`和`decision`，共711条
端点，但训练不直接加载全量端点池。固定core只包含167条planning、449条reasoning和103条
`hypothesis_elimination`，共719条。每个epoch的采样表对72个query各选择2条路径，并把所选
路径的三类端点作为不可拆分组全部加入：每轮144个路径组、432条端点、1151条实际曝光。

路径按确定性round-robin跨五轮轮换：全部237条路径至少出现一次，同题路径累计曝光最多
相差1。每个query每轮2个路径组、五轮总计10个路径组；同一slot中的归纳、停止和decision
必须具有相同`case_id`与`path_cluster_id`。这样三类端点具有完全相同的query权重和路径分布，
不会出现归纳/停止固定重复而decision单独轮换的偏差。

文件对应关系：

```text
qwen3_6_27b_reasoning_causal_path_train.jsonl                 # 1430条语义池
qwen3_6_27b_reasoning_causal_path_train_core.jsonl            # 719条固定core
qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl   # 711条全路径端点池
qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_01..05.jsonl
                                                               # 每轮432条/144路径组
qwen3_6_27b_reasoning_causal_path_validation.jsonl             # 245条
```

## 5. 当前规模和训练信号审计

| 集合 | planning | reasoning | hypothesis_elimination | evidence_summary | decision_ready | decision | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练语义池 | 167 | 449 | 103 | 237 | 237 | 237 | 1430 |
| 验证 | 29 | 79 | 17 | 40 | 40 | 40 | 245 |

每轮实际训练为167 planning、449 reasoning、103 hypothesis_elimination、144
evidence_summary、144 decision_ready和144 decision，共1151条。

按“每个CJK字符约1 token、其余字符约4字符/token”的启发式口径，每轮加权loss约为：

| 类型 | 比例 |
| --- | ---: |
| thinking | 48.1% |
| 结论/答案 | 46.2% |
| 工具调用 | 5.7% |

上表第一轮精确启发式结果为thinking48.10%、结论/答案46.17%、工具5.73%。
`evidence_summary + decision_ready`占288/1151=25.0%的样本、约40.5%的当前目标token和
23.8%的加权loss，已从旧版51.6%显著降低。`hypothesis_elimination`占103/1151=8.9%的
样本和约11.7%的加权loss；严格decision占约6.3%的加权loss。这里是启发式文本token口径，
不替代目标tokenizer报告。manifest逐轮记录同一审计；五轮自动归纳+停止的估算加权loss
范围为23.64%–23.89%，独立
校验器设置30%硬上限，超过即拒绝数据。

启发式最大样本长度约5063 token，0条超过16384。这个数字不是目标Qwen tokenizer结果。

## 6. 重新生成和静态校验

在仓库根目录执行：

```powershell
python -B scripts/convert_accepted_only_100x10_to_sft.py `
  --output-root data/2026-08-05 --archive-only
python -B scripts/convert_0805_causal_path_reasoning_sft.py
python -B scripts/validate_0805_causal_path_reasoning_sft.py
git diff --check
```

当前生成器应输出：

```text
Clustered 840 trajectories across 84 cases
Retained paths=277; semantic rows: train=1430, validation=245; per-epoch train exposures=1151
q0001: raw checkpoints=47, clusters=6, retained paths=4, selected nodes=26
```

当前校验器必须通过以下约束：来源840条且每题10条；0804冻结划分不变；每条路径各有一个
归纳、停止和decision；每条真实排错句逐字存在于来源可见消息，其证据action来自同轨迹更早
阶段且观察可复核；工具协议及来源哈希正确；未来结果不泄漏；固定模板loss为0；五个epoch
每题各选2条完整路径端点组，同一slot三类端点路径一致；五轮覆盖全部训练路径且同题路径
曝光差不超过1；输出行数、字节数和SHA256与manifest一致。

## 7. 目标tokenizer、正式训练与运行时审计

正式训练前可以先在实际训练机使用目标模型环境执行独立quick预检：

```bash
TRAIN_EPOCH_INDEX=1 bash scripts/train_qwen36_0805_causal_path_quick.sh
```

该入口会重新生成、静态校验，并用目标Qwen tokenizer和ms-swift template检查固定core及
全量唯一端点池的16K长度。它只执行一个独立冒烟epoch，梯度累积2、cosine scheduler、
10% warmup，不读取上一checkpoint，也不属于正式五轮resume链。

正式训练的权威机器可读配置与入口为：

```text
config/qwen36_0805_formal_training.json
scripts/train_qwen36_0805_causal_path_formal.sh
scripts/qwen36_0805_fixed_stage_lr_plugin.py
```

启动命令：

```bash
bash scripts/train_qwen36_0805_causal_path_formal.sh
```

正式入口必须保持以下状态机，任何一点不满足都不得称为0805正式对比训练：

1. stage 1加载固定719条core与`train_endpoint_epoch_01`，从Qwen3.6-27B基座开始。
2. stage 2–5分别加载对应`train_endpoint_epoch_02..05`，并用
   `resume_from_checkpoint`、`resume_only_model=false`恢复上一stage的模型、optimizer、
   scheduler、global step、随机数与Trainer状态。
3. 五个stage的累计`num_train_epochs`依次为1、2、3、4、5；由于每轮数据长度相同，恢复后
   每个进程只新增一个完整epoch，不重复已经完成的epoch。
4. micro batch为1、梯度累积8、有效batch为8；constant scheduler、warmup为0；五轮目标
   学习率依次为`2e-5`、`1.5e-5`、`1e-5`、`6e-6`、`3e-6`。
5. 每轮末尾执行validation并保存checkpoint，保留全部五个，不启用early stopping，也不让
   Trainer自动加载最低eval loss。
6. 普通完整resume会恢复上一轮optimizer/scheduler学习率，命令行新LR可能被覆盖。因此
   LR callback在`on_train_begin`、`on_epoch_begin`和每个`on_step_begin`重新设置optimizer
   参数组及scheduler状态；`on_log`同时核验Trainer记录。任一值不等于当轮目标立即失败。
7. `control/learning_rate_audit.jsonl`必须包含每轮train_begin、epoch_begin、step_begin和log
   事件；入口在接受stage checkpoint前再次读取该文件并检查全部optimizer LR。
8. 首次启动归档Git提交、工作树状态、训练环境与模型绝对路径、Python/ms-swift/PyTorch/
   Transformers/PEFT/Accelerate版本，以及配置、正式入口、manifest、core、五轮端点表、
   验证数据、LR插件和模型目录全部文件的SHA256。resume前逐项重新校验；中断后只允许从
   相同Git提交、相同Python/训练包版本和完整epoch边界继续，已经开始但尚无checkpoint的
   目录不得混写。

当前尚未归档目标tokenizer的真实预检报告。正式训练前必须新增带以下内容的归档：

- 模型和tokenizer绝对路径、revision或目录哈希。
- Python、PyTorch、Transformers和ms-swift版本。
- template名称与关键参数。
- 数据文件SHA256。
- min、median、p95、p99、max长度和全部超长样本ID。
- `labels != -100`的token数、各`loss_scale`的逐token数量和权重和。
- system、user、tool response全部为0；固定桥接为0；最终答案为1.0。

没有该报告时只能称“静态校验通过”，不能称“正式训练前检查完成”。

## 8. 当前输出哈希

以下哈希由当前manifest记录；任何文件变化都必须重新生成并同步本文：

```text
train 1430 f30834daed453a6bd1b2b9ffffc26db3d387bf65d607f1b640cace044fc1a651
train_core 719 b7bfa0c544b2484dff9b14942274a9b502b4f17142ce16de8a6f671ecf825554
train_endpoint_pool 711 5362a87e65083204b6690a47c0aa03cd08a31f007af51204054754dace25f7be
train_endpoint_epoch_01 432 cb8aa371415964fd89d2c174b8490d328f0a9f6ca7d8944efac53a295192acf9
train_endpoint_epoch_02 432 e6f2eb98a9e6a903ae784b0fada69f7d742e87c0b6ecc9f02ae2b409e91bd3d7
train_endpoint_epoch_03 432 2b77d8d0f64374638d0827059d51be9cd1781b380dce0d9a13a942d66c181add
train_endpoint_epoch_04 432 ae188a9ab856377647ee8e64a66765c425471ee28c5752ebe7c7eac6527d4e85
train_endpoint_epoch_05 432 b357863765f9ad95fb1e5fcce73e9ed6e8be31bed3984c743baabc82be0709d4
validation 245 d7fe066d56f545116d799e55207e14381ca426228088d7a2b9b9a4db1d66f76a
```

## 9. v9变更记录

2026-08-06相对v8修复正式训练入口与文档/manifest不一致：

- 新增`config/qwen36_0805_formal_training.json`作为机器可读权威配置，固定五阶段、梯度累积8、
  constant scheduler、零warmup、逐epoch固定LR以及完整状态resume规则。
- 新增`train_qwen36_0805_causal_path_formal.sh`；五个stage使用各自端点采样表，stage 2–5验证
  上一checkpoint处于正确epoch边界后完整恢复，累计epoch目标依次为1–5。
- 新增`qwen36_0805_fixed_stage_lr_plugin.py`，在完整resume恢复optimizer/scheduler后强制覆盖
  当轮目标LR，逐step核验并归档。正式入口会验证LR事件完整性后才接受新checkpoint。
- quick入口继续保留，但manifest和文档明确它只负责独立单轮冒烟，不能替代正式resume链。
- 独立校验器增加正式配置关键值、入口resume/训练参数和LR插件强制点检查；manifest升级为v9，
  固化配置、正式入口、quick入口和插件的路径与哈希。

## 10. v8变更记录

2026-08-06相对v7完成三项关联修改：

- 新增真实错误候选排除监督：从保留路径簇的全部来源成员中抽取120条可见排错案例，要求
  更早成功工具证据与排错协议/回显相关，并记录结构化来源，不生成没有来源的候选排除文本。
- 将证据归纳、停止判断和最终回答改成同一路径三节点采样单元；固定core从1128条改为719条，
  每轮端点由144条单独decision改为144个路径组/432条完整端点，总曝光由1272改为1151。
- 自动路径端点thinking由0.30降至0.20，归纳/停止结论由0.40降至0.20；归纳+停止估算
  有效loss占比由51.6%降至23.8%。原始可见thinking、真实来源排错和严格最终答案保持较高
  权重，工具仍为0.10。

旧`train_decision_pool`和`train_decision_epoch_01..05`文件由生成器删除，替换为对应的
`train_endpoint_pool`和`train_endpoint_epoch_01..05`，避免旧采样表被误用。

## 11. 后续修改的强制流程

任何来源、筛选、划分、聚类、节点、文本、排错信息、loss、采样、system prompt、工具协议、
tokenizer、训练入口、验证器或生成文件修改，都必须在同一变更中完成：

1. 说明修改目的、旧规则、新规则和预期影响。
2. 修改生成器/验证器，并在语义变化时提升schema版本。
3. 更新本文、`data/2026-08-05/README.md`和根`README.md`更新记录。
4. 重新生成全部派生JSON/JSONL和manifest，禁止手工修改生成文件。
5. 运行独立验证、`git diff --check`以及适用的shell语法检查。
6. 更新行数、目标类型、loss占比、长度、重复率、排错覆盖率和全部输出哈希。
7. 正式训练前归档目标tokenizer长度及逐token loss mask结果。
8. 对训练入口修改必须同时验证机器可读配置、resume链、实际optimizer LR审计点和quick/formal
   边界，不能只比较命令行表面参数。
9. 提交时把代码、文档、manifest和生成数据放在同一提交；没有同步README不得推送。

仅修改生成文件而不修改生成器，或仅修改生成器而不更新本文和manifest，均视为不可复现，
不得用于训练、比较、提交或推送。
