# 孔春春AI工坊 1.0

## 项目概述
AI 图像生成 Web 应用，支持文生图、图生图、图片反推、视频生成和文案/智能助手对话，包含提示词模板库、作品历史记录管理和积分审核系统。

## 功能特性
- 🎨 **图像生成**: 支持多种模型和纵横比选择
- 🔎 **图片反推**: 调用 Qwen 多模态模型分析图片并反推提示词
- 📝 **提示词模板**: 12个精选提示词模板，一键复制使用
- 📚 **作品库**: 历史记录管理，收藏功能
- 🔐 **登录验证**: 双角色系统（管理员/用户）
- 💝 **积分系统**: 积分余额、申请兑换、管理员审核
- 🔧 **密码管理**: 密码哈希存储，用户可通过旧密码修改

## 角色说明
| 角色 | 登录后页面 | 功能 |
|------|-----------|------|
| 管理员账号 | 管理员审核页面 | 审核积分申请、管理用户与积分 |
| 普通用户账号 | 用户使用页面 | 图像生成、积分申请、修改密码 |

首次启动时不会使用代码内置固定密码。请在 `.env` 中配置 `INITIAL_ADMIN_PASSWORD`；
如需同时初始化普通用户，可配置 `INITIAL_USER_USERNAME` 和 `INITIAL_USER_PASSWORD`。

## 目录结构
```
version1.0/
├── app/                    # 应用主目录
│   ├── app.py              # FastAPI后端服务
│   ├── image_client.py     # 图片生成核心模块
│   ├── code.html           # 用户前端页面
│   ├── admin.html          # 管理员审核页面
│   ├── templates.html      # 提示词模板页面
│   ├── login.html          # 登录页面
│   ├── start.sh            # 一键启动脚本
│   ├── template_images/    # 模板图片 (12张)
│   └── uploads/            # 上传文件临时目录
├── config/                 # 配置目录
│   └── .env.example        # 配置示例文件
├── data/                   # 数据目录（运行时生成）
│   ├── user_passwords.json # 用户密码
│   ├── user_credits.json   # 用户积分
│   └── credits_applications.json # 积分申请记录
├── resources/              # 资源文件目录
│   ├── images/             # 提示词模版原图 (12张)
│   └── prompts/            # 提示词txt文件 (12个)
├── history/                # 历史记录目录
│   └── outputs/            # 生成图片保存目录
├── docs/                   # 文档目录
│   └── DESIGN.md           # 设计系统文档
└── README.md               # 本文件
```

## 安装与运行

### 1. 环境要求
- Python 3.8+
- pip

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置API
复制配置示例并修改：
```bash
cp config/.env.example .env
# 编辑 .env 文件，填入实际的 API_URL、API_KEY 和首次初始化账号密码
```

### 4. 启动服务
```bash
cd app
./start.sh
# 或
python3 app.py
```

### 5. 访问应用
- 登录页面: http://localhost:8000/login

## API端点
| 端点 | 方法 | 说明 |
|------|------|------|
| /api/login | POST | 登录验证 |
| /api/verify | GET | 验证session |
| /api/logout | GET | 退出登录 |
| /api/change-password | POST | 修改密码（仅用户） |
| /api/credits | GET | 获取用户积分 |
| /api/credits/apply | POST | 提交积分申请 |
| /api/generate | POST | 图片生成 |
| /api/analyze-image | POST | 图片分析反推提示词 |
| /api/history | GET | 获取历史记录 |
| /api/models | GET | 获取模型列表 |
| /api/admin/user-password | GET | 兼容旧接口，仅返回密码加密状态 |
| /api/admin/user-credits | GET | 获取用户积分（管理员） |
| /api/credits/pending | GET | 待审核申请（管理员） |
| /api/credits/approve/{id} | POST | 批准申请（管理员） |
| /api/credits/reject/{id} | POST | 拒绝申请（管理员） |

## 积分兑换类型
| 类型 | 积分 |
|------|------|
| 亲亲 | 50 |
| 拥抱 | 100 |
| 牵手 | 30 |
| 陪伴聊天 | 20 |

## 模型支持
- Gemini Flash 0.5K (快速)
- Gemini Flash 2K (标准)
- Gemini Flash 4K (高清)
- GPT Image 1.5 (专业)

## 技术栈
- 后端: FastAPI + Python
- 前端: HTML + TailwindCSS + JavaScript
- 存储: 本地JSON文件

---
© 孔春春AI工坊 2026
