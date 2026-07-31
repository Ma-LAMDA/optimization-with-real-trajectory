# Train 0629 Codex IP 轨迹实验（14 题 × 10 次）

本目录集中保存本次 Codex IP 故障分析轨迹实验的输入、生成代码、完整运行结果和统计报表。

## 目录结构

```text
2026-07-27-ip_codex_train0629_14x10/
├── README.md
├── inputs/
│   ├── IP user prompt with saved configs skills.txt
│   └── train_0629.jsonl
├── scripts/
│   ├── run_codex_ip_trajectories.py
│   └── summarize_durations.py
└── results/
    ├── reports/
    │   ├── 各题准确率统计.csv
    │   └── 各题耗时统计.csv
    └── runs/
        └── fullaccess/
            ├── manifest.json
            ├── q0013_r01/
            │   ├── prompt.txt
            │   ├── source_record.json
            │   ├── run.json
            │   └── attempt_001/
            └── ...
```

`fullaccess` 是完成的 full-access 运行，包含 14 个题号各 10 条成功轨迹，共
140 条。后续新运行会直接写入
`results/runs/<运行目录>/`，日期由实验目录名和运行目录名记录。

每个运行目录内同时保存 `manifest.json`、对应的 `runner.stdout.log`/`runner.stderr.log`，以及按 `q<题号>_r<轮次>/attempt_<序号>/` 组织的完整事件、最终答案、元数据和哈希。

目录级 `.gitattributes` 禁止 Git 改写输入、轨迹和报表文件的换行符，以保证 metadata 中保存的 SHA-256 在克隆或切换分支后仍可复验。

## 外部只读依赖

实验执行时从仓库根目录的 `saved_configs/` 查询离线组网配置。该目录包含 99,031 个共享配置文件，因此不在本实验目录内重复复制。运行脚本会自动定位仓库根目录，并将其作为 Codex 工作目录。

运行结果中的 manifest 和 metadata 保留了执行发生时的原始路径，用于审计历史；文件移动后这些字段是历史记录，不表示当前存放位置。

## 重新生成轨迹

在仓库根目录执行：

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py
```

脚本默认选择题号 13、14、17、18、25、26、27、28、87、88、91、92、93、94，每题串行执行 10 次。只有成功执行才计入目标数量；额度不足时保留失败 attempt，等待 1,800 秒后重试同一槽位，其他错误会停止整批任务。

运行前可只实例化并校验 140 份 prompt：

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py --dry-run
```

从未完成运行恢复：

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py `
  --resume-run experiments/2026-07-27-ip_codex_train0629_14x10/results/runs/<运行目录>
```

## 重新生成耗时统计

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/summarize_durations.py
```

脚本自动选择最新的成功运行，校验 14 × 10 条成功 metadata，并生成 UTF-8 BOM 编码的 `results/reports/各题耗时统计.csv`。CSV 每题一行，包含 10 次轨迹耗时，以及最短、平均、中位数、最长和总耗时。

## 准确率统计

`results/reports/各题准确率统计.csv` 将 140 个 `final_answer.txt` 与 `inputs/train_0629.jsonl` 中的标准答案进行比较。判分采用故障集合精确匹配：忽略列表顺序，但漏报、多报或错报均判错。报表包含每题的正确数、错误数、准确率、正确/错误轮次、解析失败数和标准答案；总计为 117/140，准确率 83.57%。

## 题 94 epoch-10 LoRA 验证（2026-07-28）

后续验证复用了本实验 `inputs/train_0629.jsonl` 中题 94 的 source record 和
`inputs/IP user prompt with saved configs skills.txt` 模板，使用完整 prompt、
Codex CLI 0.145.0 和本地 `Qwen3.6-27B-trained` 串行运行 5 次。模型是从原始
Qwen3.6-27B 训练 10 个 epoch 后的 LoRA `checkpoint-450`。

重构后的等价执行入口为：

```bash
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py \
  --case-ids 94 \
  --repeats 5 \
  --sandbox danger-full-access \
  --model Qwen3.6-27B-trained \
  --output-root /root/autodl-tmp/qwen-codex-eval \
  --run-name q94-0728-epoch10-20260728T043741Z \
  --credit-retry-seconds 0
```

标准 label：

```json
[
  "Core_SW_01;VRRP工作在非抢占模式"
]
```

| Run | 最终预测 | 严格匹配 | 耗时 | 工具 loop |
| --- | --- | --- | ---: | ---: |
| 1 | Core_SW_01、Core_SW_02 均为 VRRP 非抢占 | 否，多报 Core_SW_02 | 458.126 秒 | 22 |
| 2 | 仅 Core_SW_01 VRRP 非抢占 | 是 | 449.790 秒 | 31 |
| 3 | 仅 Core_SW_01 VRRP 非抢占 | 是 | 686.751 秒 | 33 |
| 4 | 仅 Core_SW_01 VRRP 非抢占 | 是 | 580.252 秒 | 18 |
| 5 | 仅 Core_SW_01 VRRP 非抢占 | 是 | 440.225 秒 | 24 |

runner 和最终格式均为 5/5 成功，严格集合匹配为 4/5。5 次累计 input/output
token 为 3,750,029 / 77,832，总耗时 2,615.144 秒，加权端到端输出速度为
29.76 token/s；共执行 128 个工具 loop，其中成功 125、失败 3。所有失败命令
均为读取或 grep 不存在的 `Vlanif100` 配置，不影响 runner 完成。

本次运行产物保存在仓库外：

```text
/root/autodl-tmp/qwen-codex-eval/2026-07-28/q94-0728-epoch10-20260728T043741Z
```

逐次原始 `<result>`、SHA-256、vLLM 指标、训练参数和适用边界见
[`../../docs/2026-07-28_Q94_EPOCH10_VALIDATION.md`](../../docs/2026-07-28_Q94_EPOCH10_VALIDATION.md)。
本轮没有原始基座同条件 A/B，因此不得用 4/5 量化 LoRA 相对提升。

## 相关基座评测

2026-07-30 至 2026-07-31 的 Qwen3.6-27B 基座部署 A/B 和全量评测属于独立实验，
已统一归档到 [`../2026-07-31-qwen36-27b-base-eval/`](../2026-07-31-qwen36-27b-base-eval/)。
