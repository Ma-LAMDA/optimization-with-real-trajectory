# Qwen3.6-27B 完整 Agent 验证

- 模型：`Qwen3.6-27B-0804-best1`
- 方法：原始 Codex CLI Agent runner，允许读取离线 `saved_configs` 并执行完整工具循环。
- 部署：单个 vLLM TP=2 实例，2 个 Agent worker，总并发 2。
- Thinking：已显式请求，reasoning effort=`high`；原始 reasoning 已回填 10/10 次、共 392 个节点 / 847307 字符；provider token 字段另报 0 tokens。
- 单次硬上限：3600 秒；超时和 runner 失败均按错误计。
- 严格判分：最终 `<result>` 中的 JSON 列表必须与独立 label 完全一致。

| 题号 | 运行结果 | 严格正确 | 准确率 | 平均封顶耗时/分 | 超时 |
|---:|:---:|---:|---:|---:|---:|
| 12 | ❌ ❌ ❌ ❌ ❌ | 0/5 | 0.00% | 17.04 | 0 |
| 100 | ❌ ✅ ✅ ❌ ❌ | 2/5 | 40.00% | 11.42 | 0 |

| 汇总 | 数值 |
|:---|---:|
| 严格正确率 | 2/10 (20.00%) |
| 60 分钟内完成 | 10/10 |
| 超时 / runner 失败 | 0 / 0 |
| false positive / false negative | 12 / 8 |
| 耗时均值 / 中位数 / P95 | 14.23 / 13.79 / 22.47 分钟 |
