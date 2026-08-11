# 2026-08-09 Agent 错误感知 SFT v7 审计报告

## 结论

v7 针对 2026-08-11 20:02 深审将 action 门禁升级为完整结构门禁：从首个正 loss 前的真实工具回显重算同题所有严格事实 binding，不再信任 metadata。完整 binding 一律停止，未完整 binding 一律只监督已匹配事实、缺失条件、待验证假设和下一步只读动作。endpoint 也从“正证据顺带排除邻近标签”改为候选范围校准：正证据只确认当前标签/当前证据集合，未经候选专属反证的邻近标签保持未验证。训练/验证划分、每轮 1151 行、144 optimizer step、固定 epoch 3 / global step 432 均保持不变。

## 数据结构

- 语义池 1151 行：父级 core 719、可达 action 308、明确标注的同目标错误候选排除 replay 8、真实证据路径 bundle 116。
- endpoint 是 116 个 objective bundle，也是 116 条不同原始证据路径，覆盖 72/72 题。每题真实路径数分布为 {1: 47, 2: 9, 3: 13, 4: 3}；用 `1 / 本题真实路径数` 归一化，故每题 endpoint 总权重均为 1。
- 每个 endpoint completion 在同一可达历史后依次监督：证据归纳、正向标签边界、候选范围校准、当前证据最小集合、停止判断和最终 `<result>`；不再把未检查候选声明为已排除。
- 每轮均完整曝光 719 个父级节点、308 个 action、8 个 elimination replay 和 116 个 endpoint；固定 checkpoint-432 之前已经看到全部新增目标，不再把 boundary/minimal 延迟到第 4–5 轮。

## 复审核心指标

1. **不可达控制句为 0。** marker 行数 0。剥离 marker 后不再需要另一套统计，因为发布数据本身就是原始 Agent 可见上下文。
2. **输入路径数与输出目标多样性分开报告。** objective bundle=116，unique raw evidence path=116；但精确正目标只有 16/116 唯一，机械条件归一化后也是 16/116。116 表示不同输入证据路径，不表示 116 份不同输出措辞；不为追求唯一数而机械改写教师目标。
3. **如实报告自然多目标。** 整个语义池共有 784 个原始首个正 loss 前缀；其中 116 组、325 行存在多个兼容正目标，组合为 {('counterfactual_action_selection', 'reasoning'): 61, ('counterfactual_action_selection', 'planning'): 35, ('planning',): 20}。这些主要是原始 reasoning/planning 与同状态可达 action 的兼容延续；校验器不再用人工 marker 掩盖它们。
4. **错误根因排除与候选校准分开。** 原始可见、证据支持的排除节点 103 行/58 题，另有 8 条透明的同目标采样 replay；endpoint 不再补造排除，116 条均只说明邻近候选保持未验证，unsupported exclusion=0。
5. **恢复证据可回查。** 40 条恢复型真实路径包含 216 个事实，回指 45 个成功 `events.jsonl`，保留 item、命令、输出哈希和逐字 span。
6. **监督比重。** 启发式估算：thinking 58.32%、结论/回答 37.14%、工具调用 4.54%；endpoint bundle 7.94%，真实排除 7.21%，排除 replay 0.55%。工具 call loss 已从 0.1 降到 0.05。
7. **action 完整结构门禁。** 全量从 messages 重算 308 条 action：完整 binding 0 条，明确待验证 308 条，matched/expected 分布 {'0/1': 103, '0/4': 41, '0/6': 128, '1/4': 15, '1/6': 7, '2/4': 5, '2/6': 7, '4/6': 2}。深审点名的 17 条均存在且不完整；父级完整证据后仍调用工具 0 条。q0023 原 1/1 完整节点已删除 2 个正 loss 工具调用并改为停止。
8. **候选排除不做闭世界推断。** endpoint rejected-label 声明总数 0，候选范围校准 bundle 116/116。没有候选专属负证据时只监督“保持未验证”。
9. **目标表达多样性。** endpoint 精确正目标 16/116 唯一；只去除 snapshot/query/config path 等机械条件后仍有 16/116 唯一，设备、事实、标签和结果均保留在统计中。

## 发布门禁

- 目标 tokenizer 预检状态：`passed`。正式训练前必须归档真实 Qwen 16K 编码、逐 token loss mask、模型/tokenizer 文件哈希和 ms-swift/transformers 版本。
- 还必须在异根目录 Linux checkout 运行两套 validator、通过 runtime identity gate，并完成双卡 1–2 step full-state resume/LR smoke 和小规模 Agent canary；这些运行门禁不应被静态数据校验替代。
- malformed tool call、错误集合等 hard negative 仍仅保存在 metadata；SFT 不把坏 JSON 当正目标。显式负例学习应使用 grammar eval 或 preference/ranking 数据。
- 12 道现有验证题只用于 error-mining/dev 与横向比较；正式泛化结论仍需要未参与反推的 topology-heldout 集。
