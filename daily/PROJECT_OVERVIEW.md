# Hermes Agent — 项目功能与技术栈概览

Hermes 是一个功能丰富的多接口 AI 助手框架，具备工具调用、技能系统、记忆管理、任务调度等核心能力。它支持 CLI、TUI、Web 仪表盘及 20+ 消息平台，可部署在本地、Docker、SSH、云端等多种环境中。

---

## 目录

- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [支持的消息平台](#支持的消息平台)
- [模型提供商](#模型提供商)
- [内置工具](#内置工具)
- [插件系统](#插件系统)
- [执行环境](#执行环境)
- [技能系统](#技能系统)

---

## 核心功能

### 多模态交互界面

| 界面 | 说明 |
|------|------|
| **CLI** | 基于 prompt_toolkit + Rich 的经典交互式命令行 |
| **TUI** | 基于 Ink (React) 的现代终端 UI，通过 `hermes --tui` 启动 |
| **Web 仪表盘** | `hermes dashboard` 启动 FastAPI 后端 + React SPA，内嵌 TUI (PTY + xterm.js) |
| **编辑器集成** | ACP 适配器支持 VS Code / Zed / JetBrains |

### 消息网关 (Gateway)

统一的消息网关进程可同时接入多个聊天平台，所有平台共享一套斜杠命令和工具调用体系。支持后台进程通知、多会话管理和平台特定适配。

### 工具系统

Agent 在对话循环中自动调用工具完成任务：Web 搜索/提取、终端执行、文件操作、浏览器自动化、图像生成/分析、代码执行等。工具基于注册表机制自动发现，支持运行时按需启用/禁用。

### 子代理委派 (Delegation)

通过 `delegate_task` 工具生成独立子代理，支持单任务和批量并行两种模式。可配置角色（叶子节点 / 编排者）、最大深度、并发上限和超时时间。

### 技能系统

- **内置技能** (`skills/`)：默认可用
- **可选技能** (`optional-skills/`)：按需安装，覆盖 AI 代理、区块链、DevOps、安全等十余个分类
- **Curator 管理器**：自动追踪技能使用情况，归档不活跃技能，支持备份与恢复

### 记忆管理

可插拔的记忆后端架构，内置 Honcho、mem0、supermemory、byterover、hindsight 等多种提供商。支持跨会话记忆持久化。

### 会话管理

基于 SQLite + FTS5 的会话存储，支持全文搜索和会话摘要。通过 `session_search` 工具实现跨历史会话的语义检索。

### 定时任务 (Cron)

内置调度器支持多种时间格式（duration、cron 表达式、ISO 时间戳），可按计划触发 Agent 会话并将结果投递到指定平台。具备 3 分钟硬中断保护和文件锁防重复。

### 看板系统 (Kanban)

基于 SQLite 的持久化多代理工作队列，支持多 Profile / 多 Worker 协作。内置调度器自动分配任务、回收超时认领、促进就绪任务。

### Profile 多实例

支持多个完全隔离的实例（独立配置、密钥、记忆、会话、技能、网关）。通过 `HERMES_HOME` 环境变量实现路径隔离。

---

## 技术栈

### 语言

| 语言 | 用途 |
|------|------|
| **Python ≥3.11** | 核心运行时、Agent 循环、工具、网关、调度器 |
| **TypeScript / React** | TUI (Ink)、Web 仪表盘 (Vite + React)、文档站点 (Docusaurus) |
| **Node.js ≥20** | 浏览器自动化依赖、WhatsApp Bridge |

### Python 核心依赖

| 类别 | 关键库 |
|------|--------|
| LLM 客户端 | `openai`, `anthropic` |
| HTTP / 网络 | `httpx[socks]`, `requests`, `aiohttp` |
| 数据验证 | `pydantic` |
| 配置管理 | `pyyaml`, `ruamel.yaml`, `python-dotenv` |
| CLI / 显示 | `rich`, `prompt_toolkit`, `fire` |
| 搜索 / 爬取 | `exa-py`, `firecrawl-py`, `parallel-web` |
| 调度 | `croniter` |
| 模板 | `jinja2` |
| 重试 | `tenacity` |
| 进程管理 | `psutil` |
| TTS | `edge-tts`, `elevenlabs`（可选） |
| 认证 | `PyJWT[crypto]` |

### 前端依赖

| 项目 | 关键库 |
|------|--------|
| **TUI** (`ui-tui/`) | React 19, Ink 6, Nanostores, TypeScript, esbuild, vitest |
| **Web 仪表盘** (`web/`) | Vite 7, React 19, Tailwind 4, React Router 7, xterm.js, R3F, GSAP |
| **文档站点** (`website/`) | Docusaurus 3.9, React 19, Mermaid, MDX |

### 存储

- **SQLite**：会话存储（FTS5 全文索引）、看板系统
- **YAML / dotenv**：用户配置 (`config.yaml`) 和密钥 (`.env`)

### 测试

- **pytest** + **pytest-xdist**（4 worker 并行）+ **pytest-asyncio**
- 通过 `scripts/run_tests.sh` 保证与 CI 环境一致（隔离环境变量、UTC 时区）

### 包管理与构建

- Python: `uv` / `pip`，`pyproject.toml` 驱动，`uv.lock` 锁定
- Node: `npm`
- Docker: 多阶段构建，`tini` 进程管理，`gosu` UID 映射

---

## 系统架构

```
                         ┌─────────────────────┐
                         │   用户 / 消息平台     │
                         └─────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
        ┌─────┴─────┐    ┌───────┴────────┐    ┌──────┴──────┐
        │   CLI/TUI   │    │  消息网关       │    │ Web 仪表盘   │
        │ (cli.py     │    │ (gateway/)     │    │ (web/)      │
        │  ui-tui/)   │    │ 20+ 平台适配    │    │ PTY 桥接    │
        └─────┬──────┘    └───────┬────────┘    └──────┬──────┘
              │                    │                     │
              └────────────────────┼────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │   AIAgent 核心   │
                          │ (run_agent.py)   │
                          └────────┬────────┘
                                   │
           ┌───────────┬───────────┼───────────┬────────────┐
           │           │           │           │            │
     ┌─────┴──┐  ┌────┴────┐ ┌───┴───┐  ┌───┴────┐  ┌───┴────┐
     │ 工具系统 │  │模型提供商│ │记忆管理│  │技能系统 │  │调度/Cron│
     │(tools/) │  │(plugins/│ │(agent/│  │(skills/│  │(cron/) │
     │         │  │ model-  │ │memory)│  │        │  │        │
     └────────┘  │providers)│ └───────┘  └────────┘  └────────┘
                  └─────────┘
```

### Agent 核心循环

```python
while iterations < max_iterations and budget.remaining > 0:
    response = llm.chat(messages, tools=tool_schemas)
    if response.tool_calls:
        for call in response.tool_calls:
            result = handle_function_call(call.name, call.args)
            messages.append(tool_result(result))
    else:
        return response.content
```

消息格式遵循 OpenAI 标准：`system` / `user` / `assistant` / `tool`。

---

## 支持的消息平台

### 内置平台适配器

| 分类 | 平台 |
|------|------|
| **即时通讯** | Telegram, Discord, Slack, Signal, Mattermost, Matrix, WhatsApp, DingTalk, 飞书 (Feishu), 企业微信 (WeCom), 微信 (Weixin), QQ, 腾讯元宝 (Yuanbao), BlueBubbles (iMessage) |
| **邮件 / 短信** | Email (IMAP/SMTP), SMS (Twilio) |
| **智能家居** | Home Assistant |
| **HTTP 接口** | Webhook, OpenAI 兼容 API Server, Microsoft Graph Webhook |

### 插件平台（可扩展）

IRC, LINE, Google Chat, Microsoft Teams

---

## 模型提供商

通过 `plugins/model-providers/` 插件化管理，支持的提供商包括：

| 类别 | 提供商 |
|------|--------|
| **主流** | OpenAI, Anthropic, Google Gemini, DeepSeek, xAI |
| **云服务** | AWS Bedrock, Azure Foundry, NVIDIA, OpenRouter, Vercel AI Gateway |
| **国产** | 阿里通义 (Alibaba), Kimi/Moonshot, MiniMax, 阶跃星辰 (StepFun), 千问 OAuth (Qwen), 小米 (Xiaomi), 智谱 (z.ai/GLM) |
| **开源/自托管** | Ollama Cloud, Hugging Face, Nous Portal, GMI |
| **其他** | Arcee, Copilot, Kilocode, OpenCode Zen |

用户插件同名可覆盖内置提供商（last-writer-wins），无需修改源码。

---

## 内置工具

### 核心工具集 (`_HERMES_CORE_TOOLS`)

| 分类 | 工具 |
|------|------|
| **Web** | `web_search`, `web_extract` |
| **终端** | `terminal`, `process` |
| **文件** | `read_file`, `write_file`, `patch`, `search_files` |
| **浏览器** | `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_get_images`, `browser_vision`, `browser_console`, `browser_cdp`, `browser_dialog` |
| **视觉 / 图像** | `vision_analyze`, `image_generate` |
| **技能** | `skills_list`, `skill_view`, `skill_manage` |
| **TTS** | `text_to_speech` |
| **规划 / 记忆** | `todo`, `memory` |
| **会话** | `session_search` |
| **交互** | `clarify` |
| **执行 / 委派** | `execute_code`, `delegate_task` |
| **调度** | `cronjob` |
| **消息** | `send_message` |
| **智能家居** | `ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service` |
| **看板** | `kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`, `kanban_unblock` |
| **计算机操控** | `computer_use` (macOS) |

### 扩展工具集

`video_analyze`, `mixture_of_agents`, Discord 管理, Spotify 控制, 飞书文档/云盘, RL 训练, MCP 工具代理等。

---

## 插件系统

Hermes 提供多层插件架构：

| 插件类型 | 目录 | 能力 |
|---------|------|------|
| **通用插件** | `~/.hermes/plugins/` | 生命周期钩子、注册工具、注册 CLI 子命令、注册平台 |
| **记忆提供商** | `plugins/memory/` | 实现 `MemoryProvider` ABC（sync_turn, prefetch, shutdown） |
| **模型提供商** | `plugins/model-providers/` | 注册 `ProviderProfile`，自定义推理后端 |
| **图像生成** | `plugins/image_gen/` | OpenAI, xAI 等图像生成器 |
| **平台插件** | `plugins/platforms/` | 扩展消息网关平台 |
| **可观测性** | `plugins/observability/` | Langfuse 集成 |
| **功能插件** | `plugins/kanban/`, `plugins/hermes-achievements/` 等 | 看板、成就系统、Spotify 等 |

### 生命周期钩子

```
pre_tool_call → post_tool_call
pre_llm_call  → post_llm_call
on_session_start → on_session_end
```

---

## 执行环境

Agent 的终端命令可在以下环境中运行：

| 环境 | 说明 |
|------|------|
| **Local** | 本地系统直接执行 |
| **Docker** | 容器化执行，提供沙箱隔离 |
| **SSH** | 远程服务器执行 |
| **Singularity** | HPC 容器化执行 |
| **Modal** | 云端 Serverless 执行 |
| **Daytona** | 云开发环境 |
| **Vercel Sandbox** | Vercel 沙箱环境 |

Docker 部署使用多阶段构建镜像（Debian 基础 + Node.js + Playwright Chromium），`tini` 作为 PID 1，`gosu` 处理 UID 映射。

---

## 技能系统

### 内置技能 (`skills/`)

按分类组织（如 `github/`, `mlops/`），默认可用。

### 可选技能 (`optional-skills/`)

通过 `hermes skills install official/<category>/<skill>` 安装，覆盖以下分类：

`autonomous-ai-agents`, `blockchain`, `communication`, `creative`, `devops`, `email`, `health`, `mcp`, `migration`, `mlops`, `productivity`, `research`, `security`, `web-development`

### Curator 生命周期管理

- 自动追踪 Agent 创建的技能使用率
- 闲置技能 → 标记为 stale → 自动归档
- 支持 pin（免于自动操作）、备份、恢复
- 绝不删除，最大破坏性操作为归档

---

> 本文档基于 2026 年 5 月的代码库状态编写。项目持续迭代中，具体细节请以源码为准。
