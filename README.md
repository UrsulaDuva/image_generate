# AI 图像工坊

一个基于 FastAPI + 原生 HTML/Tailwind 的轻量级 AI 图像/视频/对话一体化工坊框架，
支持图片生成、图片编辑、图片反推（多模态理解）、视频生成、文案对话、提示词模板与
积分管理等场景。所有上游模型均通过 OpenAI 兼容协议接入，不绑定具体供应商。

> 本仓库仅包含框架代码，不附带任何账号、密钥、模型清单或运行时数据。

## 功能模块

- **图片生成**：OpenAI 兼容 `/images/generations`，支持文生图与图生图（edits）
- **图片反推**：OpenAI 兼容多模态 `/chat/completions`，从图片反推自然语言描述
- **视频生成**：兼容 Vertex AI 协议（如 doubao-seedance），异步轮询下载
- **文案对话**：OpenAI 兼容 `/chat/completions`，支持流式 SSE，可配 Agent 工具调用
- **提示词模板库**：内置常用提示词模板，可分类检索
- **登录与积分**：双角色（管理员 / 用户），积分申请、审核、流水、余额
- **管理后台**：用户管理、积分调整、申请审批

## 目录结构

```
.
├── app/                    # FastAPI 应用 + 静态前端
│   ├── app.py              # 主入口（路由、SSR HTML）
│   ├── image_client.py     # 图像生成客户端
│   ├── video_client.py     # 视频生成客户端（Vertex 协议）
│   ├── llm/                # Agent + Prompt 编排 + MCP 桥接
│   ├── *.html              # 各页面（chat/analyze/video/admin/login...）
│   └── start.sh            # 启动脚本
├── data/                   # 运行时数据（用户、积分、会话）— 已剥离
├── docs/                   # 设计文档
├── resources/              # 内置资源（提示词样例等）
├── skills/                 # 可选技能模块
├── requirements.txt
└── .env.example            # 环境变量示例（复制为 .env 后填值）
```

## 快速开始

### 1. 准备环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，按需填入：
#   API_URL / API_KEY / MODEL                  - 图片生成
#   VISION_API_URL / VISION_API_KEY / VISION_MODEL  - 多模态理解
#   CHAT_API_URL / CHAT_API_KEY / CHAT_MODEL   - 文案对话
#   INITIAL_ADMIN_PASSWORD                     - 首次启动管理员密码
```

### 3. 启动服务

```bash
cd app
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

或使用脚本 `bash app/start.sh`。

浏览器访问 `http://127.0.0.1:8000`，使用 `admin` + `INITIAL_ADMIN_PASSWORD` 登录管理后台。

## 配置说明

所有上游 API 均使用 OpenAI 兼容协议（`Authorization: Bearer ...`）。常见可接入的网关包括：

- 图片生成：自建 / 第三方 OpenAI 兼容图像网关
- 文案对话：自建 vLLM / 第三方 LLM 聚合网关（zhipu/deepseek/qwen/kimi 等）
- 多模态理解：阿里云 DashScope、第三方多模态网关
- 视频生成：兼容 Vertex AI 协议的网关（doubao-seedance 等）

仅需保证上游遵循 OpenAI/Vertex 协议即可接入，无需修改业务代码。

## 数据持久化

- `data/user_passwords.json` — 用户名 → PBKDF2 密码哈希
- `data/user_credits.json`   — 用户名 → 积分余额
- `data/sessions.json`       — 会话 token → 用户名/过期时间
- `data/credits_applications.json` — 积分申请单
- `data/credits_ledger.json` — 积分变动流水
- `data/chat_sessions/`      — 用户对话历史（JSON）
- `history/`                 — 生成产物（图片/视频）输出目录

## 安全注意事项

- **不要提交 `.env`**：仓库 `.gitignore` 已默认排除
- **不要提交 `data/` 与 `history/` 中的真实运行数据**
- 生产部署请配置反向代理（nginx）+ HTTPS + 速率限制
- 密码以 PBKDF2-SHA256 + 260000 轮 + 随机 salt 存储

## 许可证

本项目代码以 MIT 许可证发布。第三方上游 API 的使用须遵循各自服务条款。
