# saved_configs 本地查询服务

该目录把仓库根目录下的 `saved_configs/` 以只读 HTTP API 暴露出来。接口兼容原 IP user prompt 的项目、节点和命令查询格式，并额外提供命令发现与快照内容搜索，供 GPT 在不直接访问文件夹的情况下查询离线数据。

服务仅返回已有 `.txt` 快照，不连接设备、不执行新的 CLI 命令，也不会修改 `saved_configs/`。

代理范围仅限 `saved_configs/`。仓库根目录不是 Web 静态目录，`README.md`、`data/`、`experiments/` 等其他仓库文件和目录不会通过本服务暴露；根路径 `/` 只是调用 API 的固定浏览器界面。

## 启动

要求 Python 3.8 或更高版本，不需要安装第三方依赖。在仓库根目录运行：

```bash
python saved_configs_service/serve_saved_configs.py
```

Windows 也可以使用：

```powershell
py -3 saved_configs_service/serve_saved_configs.py
```

默认设置为：

- 数据目录：仓库根目录下的 `saved_configs/`
- 监听地址：`127.0.0.1`
- 端口：`3080`

启动成功后，在浏览器打开：

```text
http://127.0.0.1:3080/
```

首页会自动显示 `saved_configs` 下的项目文件夹。依次点击项目、节点和命令快照，即可查看对应 `.txt` 文件的完整内容。页面还提供项目、节点、命令文件筛选和内容复制功能。

也可以通过健康检查确认服务状态：

```bash
curl http://127.0.0.1:3080/healthz
```

预期响应为 `{"status":"ok"}`。

如果修改了服务脚本或网页文件，需要先在原终端按 `Ctrl+C` 停止旧进程，再重新运行启动命令；已运行的 Python 进程不会自动加载代码更新。

如需从其他机器或容器访问，可显式监听所有网卡，并按实际情况设置防火墙：

```bash
python saved_configs_service/serve_saved_configs.py \
  --host 0.0.0.0 \
  --port 3080 \
  --configs-root /path/to/repository/saved_configs
```

该服务不包含身份认证。`saved_configs` 可能含有网络配置敏感信息，因此不要把端口直接暴露到不受信任的网络。

也可以用环境变量 `SAVED_CONFIGS_SERVICE_HOST`、`SAVED_CONFIGS_SERVICE_PORT` 和 `SAVED_CONFIGS_ROOT` 设置这三个参数；命令行参数优先。

## API

| 用途 | 方法与路径 |
| --- | --- |
| 浏览器界面 | `GET /` |
| 健康检查 | `GET /healthz` |
| 项目列表 | `GET /v3/projects` |
| 节点列表 | `GET /v3/projects/{project_id}/nodes` |
| 命令发现 | `GET /v3/projects/{project_id}/nodes/{node_id}/commands?keyword=ip&limit=200&offset=0` |
| 命令回显 | `GET /v3/projects/{project_id}/nodes/{node_id}/command?cmd=display%20version` |
| 项目内搜索 | `GET /v3/projects/{project_id}/search?q=10.0.0.1&node_id=Core_SW_01&file_keyword=route&limit=20` |

所有路径参数和查询参数都应进行 URL 编码。

命令发现接口始终返回真实 `command_key`。只有快照首个非空行能够与该 key 严格对应时，`command` 才返回识别出的 CLI，否则返回 `null`，避免把输出正文误认成命令。命令回显接口的 `cmd` 参数既可以传 CLI 命令，也可以传 `command_key`；调用方应优先传精确 `command_key`。找到后返回与旧接口一致的 `{"command":"...","output":"..."}`。未找到命令时仍返回 HTTP 200，并在 `output` 中返回旧接口格式的错误信息。

搜索接口必须至少提供 `q` 或 `file_keyword`。`q` 搜索快照内容，`file_keyword` 筛选命令名/文件标识，`node_id` 可将搜索限制在一个精确节点；响应只给出命中行，取得完整证据时应再调用命令回显接口。

## GPT prompt

使用同目录下的 `IP user prompt with local saved configs service.txt`。它保留 `{original_query}` 和 `{output_format}` 占位符，但将数据查询流程改为仅通过本服务的 URL 完成。原有 prompt 文件没有改动。

如果服务不在 GPT 调用环境的 `127.0.0.1:3080`，使用前将新 prompt 中的基础 URL 替换成可访问地址。

## 测试

测试只使用临时数据，不会读取或修改真实 `saved_configs/`：

```bash
python -m unittest discover -s saved_configs_service/tests -v
```
