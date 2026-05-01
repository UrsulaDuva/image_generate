#!/usr/bin/env python3
"""
AI图像工坊 - FastAPI后端服务
整合 image_client.py 的图片生成功能
带登录验证 + 积分审核系统
"""

import os
import sys
import json
import base64
import hashlib
import time
import secrets
import threading
import uuid
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import requests
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 添加当前目录到路径以导入 image_client
sys.path.insert(0, str(Path(__file__).parent))
from image_client import (
    read_env_from_file,
    compress_image,
    fetch_with_retry,
    extract_image_from_response,
    infer_mime_type,
    response_json_or_raise,
)
from video_client import generate_video as call_generate_video_api

# ============================================
# 配置
# ============================================

BASE_DIR = Path(__file__).parent.parent  # version1.0目录
ENV_PATH = BASE_DIR / '.env'
HISTORY_DIR = BASE_DIR / 'history'
OUTPUTS_DIR = HISTORY_DIR / 'outputs'
VIDEOS_DIR = HISTORY_DIR / 'videos'
UPLOADS_DIR = Path(__file__).parent / 'uploads'
TEMPLATE_IMAGES_DIR = Path(__file__).parent / 'template_images'
HISTORY_FILE = HISTORY_DIR / 'history.json'
VIDEO_HISTORY_FILE = HISTORY_DIR / 'video_history.json'

# 视频生成配置
VIDEO_CREDITS_COST = 100
VIDEO_DEFAULT_MODEL = "bytedance/doubao-seedance-2.0"  # client 内部会归一化为 zenmux 官方名
VIDEO_AVAILABLE_MODELS = [
    {"id": "bytedance/doubao-seedance-2.0", "name": "Doubao Seedance 2.0", "description": "字节豆包 Seedance 2.0"},
]

# 数据文件
USER_PASSWORDS_FILE = BASE_DIR / 'data' / 'user_passwords.json'
USER_CREDITS_FILE = BASE_DIR / 'data' / 'user_credits.json'
CREDITS_APPLICATIONS_FILE = BASE_DIR / 'data' / 'credits_applications.json'
CREDITS_LEDGER_FILE = BASE_DIR / 'data' / 'credits_ledger.json'  # 积分变动流水
CHAT_SESSIONS_DIR = BASE_DIR / 'data' / 'chat_sessions'  # 按用户存储 chat 会话
SESSIONS_FILE = BASE_DIR / 'data' / 'sessions.json'

# Session存储 (token -> {username, expires_at})，运行时内存缓存 + 本地持久化
SESSIONS = {}
SESSIONS_LOADED = False

# 视频生成异步任务表 (task_id -> dict)
# status: pending / running / done / failed
VIDEO_TASKS: dict = {}
VIDEO_TASKS_LOCK = threading.Lock()

# ============================================
# 积分兑换配置
# ============================================
EXCHANGE_RATES = {
    "亲亲": 50,
    "拥抱": 100,
    "牵手": 30,
    "陪伴聊天": 20
}

# ============================================
# 后处理标签 -> 提示词映射
# ============================================
POSTPROCESS_PROMPTS = {
    "2倍放大": "upscale 2x, high resolution, detailed",
    "面部增强": "enhance facial details, detailed face, refined features",
    "去除背景": "remove background, transparent background, isolated subject",
    "色彩校正": "color correction, enhanced colors, vibrant tones",
    "锐化": "sharpen details, crisp edges, clear definition"
}

# 纵横比 -> 尺寸映射
ASPECT_RATIO_SIZES = {
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "16:9": "1024x576"
}

# 模型分辨率限制
MODEL_SIZE_LIMITS = {
    "gemini-3.1-flash-image-preview-0.5k": {"max": 512, "sizes": {"1:1": "512x512", "4:3": "512x384", "16:9": "512x288"}},
    "gemini-3.1-flash-image-preview-2k": {"max": 1024, "sizes": {"1:1": "1024x1024", "4:3": "1024x768", "16:9": "1024x576"}},
    "gemini-3.1-flash-image-preview-4k": {"max": 2048, "sizes": {"1:1": "2048x2048", "4:3": "2048x1536", "16:9": "2048x1152"}},
    "gpt-image-1.5": {"max": 1024, "sizes": {"1:1": "1024x1024", "4:3": "1024x768", "16:9": "1024x576"}},
}

# 默认模型列表
DEFAULT_MODELS = [
    {"id": "gemini-3.1-flash-image-preview-0.5k", "name": "Gemini Flash 0.5K", "description": "快速"},
    {"id": "gemini-3.1-flash-image-preview-2k", "name": "Gemini Flash 2K", "description": "标准"},
    {"id": "gemini-3.1-flash-image-preview-4k", "name": "Gemini Flash 4K", "description": "高清"},
    {"id": "gpt-image-1.5", "name": "GPT Image 1.5", "description": "专业"},
]

# ============================================
# 数据模型
# ============================================

class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    username: str
    old_password: Optional[str] = None
    new_password: str

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gemini-3.1-flash-image-preview-2k"
    aspect_ratio: str = "1:1"
    postprocess: List[str] = []
    input_image_base64: Optional[str] = None
    opacity: int = 100

class VideoGenerateRequest(BaseModel):
    prompt: str
    model: str = VIDEO_DEFAULT_MODEL
    aspect_ratio: Optional[str] = "16:9"      # 16:9 / 9:16 / 1:1
    resolution: Optional[str] = "720p"        # 720p / 1080p
    duration_seconds: Optional[int] = 5       # 5 / 8 / 10
    generate_audio: Optional[bool] = False
    input_image_base64: Optional[str] = None  # 可选首帧图

class CreditsApplyRequest(BaseModel):
    type: str
    count: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None  # 不传则使用模型上限
    temperature: float = 0.7
    model: Optional[str] = None

class ChatSessionSaveRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[dict]
    model: Optional[str] = None
    usage: Optional[dict] = None
    session_type: str = "chat"

class AddUserRequest(BaseModel):
    username: str
    password: str
    credits: int = 50

class SetCreditsRequest(BaseModel):
    username: str
    credits: int
    reason: Optional[str] = ''  # 可选备注，写入流水

# ============================================
# 数据加载/保存函数
# ============================================

DATA_LOCK = threading.RLock()
PASSWORD_HASH_PREFIX = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 260000
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_UPLOAD_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
DEFAULT_SESSION_MAX_AGE_SECONDS = 86400 * 7

def write_json_atomic(path: Path, data):
    """原子写 JSON，降低并发/中断时写坏文件的概率。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=str(path.parent), delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_name = f.name
    os.replace(tmp_name, path)

def read_int_setting(name: str, default: int, min_value: int, max_value: int) -> int:
    env_config = read_env_from_file(str(ENV_PATH))
    raw = os.environ.get(name) or env_config.get(name)
    try:
        value = int(raw) if raw else default
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(value, max_value))

def get_session_max_age_seconds() -> int:
    return read_int_setting(
        "SESSION_MAX_AGE_SECONDS",
        DEFAULT_SESSION_MAX_AGE_SECONDS,
        3600,
        86400 * 30,
    )

def is_password_hash(value: str) -> bool:
    return isinstance(value, str) and value.startswith(f"{PASSWORD_HASH_PREFIX}$")

def hash_password(password: str) -> str:
    salt = secrets.token_urlsafe(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return f"{PASSWORD_HASH_PREFIX}${PASSWORD_HASH_ITERATIONS}${salt}${digest}"

def verify_password(password: str, stored: str) -> bool:
    if not isinstance(stored, str):
        return False
    if not is_password_hash(stored):
        return secrets.compare_digest(password, stored)
    try:
        _, iterations, salt, digest = stored.split("$", 3)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        ).hex()
        return secrets.compare_digest(actual, digest)
    except Exception:
        return False

def invalidate_user_sessions(username: str):
    ensure_sessions_loaded()
    changed = False
    for token, session in list(SESSIONS.items()):
        if session.get("username") == username:
            del SESSIONS[token]
            changed = True
    if changed:
        save_sessions()

def load_sessions() -> dict:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {}

    now = int(time.time())
    sessions = {}
    changed = False
    for token, value in data.items():
        if isinstance(value, str):
            session = {
                "username": value,
                "expires_at": now + get_session_max_age_seconds(),
            }
            changed = True
        elif isinstance(value, dict):
            session = {
                "username": str(value.get("username") or ""),
                "expires_at": int(value.get("expires_at") or 0),
            }
        else:
            changed = True
            continue

        if not session["username"] or session["expires_at"] <= now:
            changed = True
            continue
        sessions[token] = session

    if changed:
        write_json_atomic(SESSIONS_FILE, sessions)
    return sessions

def save_sessions():
    with DATA_LOCK:
        write_json_atomic(SESSIONS_FILE, SESSIONS)

def ensure_sessions_loaded():
    global SESSIONS_LOADED, SESSIONS
    if SESSIONS_LOADED:
        return
    with DATA_LOCK:
        if not SESSIONS_LOADED:
            SESSIONS = load_sessions()
            SESSIONS_LOADED = True

def create_session(username: str) -> tuple[str, int]:
    ensure_sessions_loaded()
    max_age = get_session_max_age_seconds()
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": username,
        "expires_at": int(time.time()) + max_age,
    }
    save_sessions()
    return token, max_age

def get_session_username(token: str) -> Optional[str]:
    if not token:
        return None
    ensure_sessions_loaded()
    session = SESSIONS.get(token)
    if not session:
        return None
    if int(session.get("expires_at") or 0) <= int(time.time()):
        del SESSIONS[token]
        save_sessions()
        return None
    return session.get("username")

def delete_session(token: str):
    ensure_sessions_loaded()
    if token in SESSIONS:
        del SESSIONS[token]
        save_sessions()

def get_initial_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    env_config = read_env_from_file(str(ENV_PATH))
    value = os.environ.get(name) or env_config.get(name) or default
    if isinstance(value, str):
        value = value.strip()
    return value or None

def load_initial_passwords() -> dict:
    admin_username = get_initial_setting("INITIAL_ADMIN_USERNAME", "admin")
    admin_password = get_initial_setting("INITIAL_ADMIN_PASSWORD")
    placeholder_passwords = {
        "change-this-admin-password",
        "change-this-user-password",
        "your-admin-password",
        "your-user-password",
    }

    if not admin_password or admin_password in placeholder_passwords:
        raise RuntimeError(
            "首次初始化账号需要配置 INITIAL_ADMIN_PASSWORD。"
            "请在 .env 中设置一个真实强密码后再启动服务。"
        )

    users = {admin_username: hash_password(admin_password)}

    user_username = get_initial_setting("INITIAL_USER_USERNAME")
    user_password = get_initial_setting("INITIAL_USER_PASSWORD")
    if user_username or user_password:
        if not user_username or not user_password:
            raise RuntimeError("INITIAL_USER_USERNAME 和 INITIAL_USER_PASSWORD 必须同时配置。")
        if user_password in placeholder_passwords:
            raise RuntimeError("INITIAL_USER_PASSWORD 不能使用 .env.example 中的占位密码。")
        if user_username in users:
            raise RuntimeError("INITIAL_USER_USERNAME 不能与管理员账号相同。")
        users[user_username] = hash_password(user_password)

    return users

def sanitize_filename(filename: str) -> str:
    name = Path(filename or "upload.png").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload.png"

def load_user_passwords() -> dict:
    """加载用户密码"""
    with DATA_LOCK:
        if not USER_PASSWORDS_FILE.exists():
            # 首次初始化必须从环境变量读取密码，避免代码中出现固定默认密码。
            default = load_initial_passwords()
            save_user_passwords(default)
            return default
        with open(USER_PASSWORDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        migrated = False
        for username, stored in list(data.items()):
            if isinstance(stored, str) and not is_password_hash(stored):
                data[username] = hash_password(stored)
                migrated = True
        if migrated:
            save_user_passwords(data)
        return data

def save_user_passwords(data: dict):
    """保存用户密码"""
    with DATA_LOCK:
        write_json_atomic(USER_PASSWORDS_FILE, data)

def load_user_credits() -> dict:
    """加载用户积分"""
    if not USER_CREDITS_FILE.exists():
        passwords = load_user_passwords()
        admin_username = get_initial_setting("INITIAL_ADMIN_USERNAME", "admin")
        admin_credits = read_int_setting("INITIAL_ADMIN_CREDITS", 1000, 0, 1000000)
        user_credits = read_int_setting("INITIAL_USER_CREDITS", 50, 0, 1000000)
        default = {
            username: admin_credits if username == admin_username else user_credits
            for username in passwords
        }
        save_user_credits(default)
        return default
    with open(USER_CREDITS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_user_credits(data: dict):
    """保存用户积分"""
    with DATA_LOCK:
        write_json_atomic(USER_CREDITS_FILE, data)

def load_credits_applications() -> List[dict]:
    """加载积分申请"""
    if not CREDITS_APPLICATIONS_FILE.exists():
        return []
    with open(CREDITS_APPLICATIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_credits_applications(data: List[dict]):
    """保存积分申请"""
    with DATA_LOCK:
        write_json_atomic(CREDITS_APPLICATIONS_FILE, data)

def load_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_history(history: List[dict]):
    with DATA_LOCK:
        write_json_atomic(HISTORY_FILE, history)

# ---- 积分流水 ----
def load_credits_ledger() -> List[dict]:
    if not CREDITS_LEDGER_FILE.exists():
        return []
    try:
        with open(CREDITS_LEDGER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_credits_ledger(data: List[dict]):
    with DATA_LOCK:
        write_json_atomic(CREDITS_LEDGER_FILE, data)

def add_credits_ledger(username: str, change: int, balance: int, source: str, detail: str = ''):
    ledger = load_credits_ledger()
    ledger.insert(0, {
        "id": f"cl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "username": username,
        "change": change,        # 正数=收入，负数=支出
        "balance": balance,      # 变动后余额
        "source": source,        # 来源类型: generate_image / generate_chat / approve / admin_grant
        "detail": detail,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    save_credits_ledger(ledger)

# ---- Chat 会话存储（按用户ID独立文件） ----
def normalize_session_type(session_type: str) -> str:
    return "agent" if session_type == "agent" else "chat"

def user_chat_file(username: str, session_type: str = "chat") -> Path:
    safe = ''.join(c for c in username if c.isalnum() or c in '_-')
    suffix = "_agent" if normalize_session_type(session_type) == "agent" else ""
    return CHAT_SESSIONS_DIR / f"{safe}{suffix}.json"

def load_user_chat_sessions(username: str, session_type: str = "chat") -> List[dict]:
    fp = user_chat_file(username, session_type)
    if not fp.exists():
        return []
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_user_chat_sessions(username: str, sessions: List[dict], session_type: str = "chat"):
    fp = user_chat_file(username, session_type)
    with DATA_LOCK:
        write_json_atomic(fp, sessions)

# ============================================
# 认证函数
# ============================================

def verify_session(request: Request) -> Optional[str]:
    """验证session，返回username或None"""
    token = request.cookies.get('auth_token')
    return get_session_username(token)

def require_admin(request: Request):
    """要求管理员身份"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    if username != 'admin':
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return username

# ============================================
# 辅助函数
# ============================================

def load_env_config() -> dict:
    return read_env_from_file(str(ENV_PATH))

def get_api_config() -> tuple:
    env_config = load_env_config()
    api_url = os.environ.get('API_URL') or env_config.get('API_URL')
    api_key = os.environ.get('API_KEY') or env_config.get('API_KEY')
    return api_url, api_key

def get_vision_config() -> tuple[str, str, str]:
    env_config = load_env_config()
    api_url = (
        os.environ.get('VISION_API_URL')
        or env_config.get('VISION_API_URL')
        or os.environ.get('QWEN_VISION_API_URL')
        or env_config.get('QWEN_VISION_API_URL')
        or ''
    )
    api_key = (
        os.environ.get('VISION_API_KEY')
        or env_config.get('VISION_API_KEY')
        or os.environ.get('QWEN_VISION_API_KEY')
        or env_config.get('QWEN_VISION_API_KEY')
        or ''
    )
    model = (
        os.environ.get('VISION_MODEL')
        or env_config.get('VISION_MODEL')
        or os.environ.get('QWEN_VISION_MODEL')
        or env_config.get('QWEN_VISION_MODEL')
        or 'qwen3.6-plus'
    )
    if api_url and not api_url.rstrip('/').endswith('/chat/completions'):
        api_url = api_url.rstrip('/') + '/chat/completions'
    return api_url, api_key, model

def get_image_request_options() -> tuple[int, int]:
    """网页端图片生成的请求参数：减少失败场景的长时间无效等待。"""
    env_config = load_env_config()

    def read_int(name: str, default: int, min_value: int, max_value: int) -> int:
        raw = os.environ.get(name) or env_config.get(name)
        try:
            value = int(raw) if raw else default
        except (TypeError, ValueError):
            value = default
        return max(min_value, min(value, max_value))

    timeout = read_int('IMAGE_API_TIMEOUT_SECONDS', 120, 30, 300)
    attempts = read_int('IMAGE_API_MAX_ATTEMPTS', 2, 1, 4)
    return timeout, attempts

def safe_http_error(status_code: int, message: str) -> HTTPException:
    """返回给前端的错误只保留稳定文案，详细异常写服务端日志。"""
    return HTTPException(status_code=status_code, detail=message)

def get_video_api_key() -> str:
    """视频生成走 ZenMux Vertex AI 协议，复用 CHAT_API_KEY。
    可通过 VIDEO_API_KEY 单独覆盖。
    """
    env_config = load_env_config()
    return (
        os.environ.get('VIDEO_API_KEY')
        or env_config.get('VIDEO_API_KEY')
        or os.environ.get('CHAT_API_KEY')
        or env_config.get('CHAT_API_KEY')
        or ''
    )

def load_video_history() -> List[dict]:
    if not VIDEO_HISTORY_FILE.exists():
        return []
    try:
        with open(VIDEO_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_video_history(history: List[dict]):
    VIDEO_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VIDEO_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def generate_video_filename(prompt: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{timestamp}_{prompt_hash}.mp4"

def ensure_png_format(image_bytes: bytes) -> bytes:
    """将任意格式的图片转换为真正的 PNG 格式"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format and img.format.lower() == 'png':
            return image_bytes
        # 转换为 RGBA 以确保透明度支持
        if img.mode in ('RGBA', 'LA', 'P'):
            png_bytes = io.BytesIO()
            img.save(png_bytes, format='PNG')
            return png_bytes.getvalue()
        # RGB 转 RGBA
        if img.mode == 'RGB':
            img = img.convert('RGBA')
        else:
            img = img.convert('RGBA')
        png_bytes = io.BytesIO()
        img.save(png_bytes, format='PNG')
        return png_bytes.getvalue()
    except Exception as e:
        print(f"PNG 转换警告: {e}，使用原始数据")
        return image_bytes


def generate_image_filename(prompt: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:6]
    return f"{timestamp}_{prompt_hash}.png"

def enhance_prompt_with_postprocess(prompt: str, postprocess: List[str], opacity: int = 100) -> str:
    enhancements = []
    if opacity != 100:
        enhancements.append(f"修改图片不透明度为{opacity}%")
    for tag in postprocess:
        if tag in POSTPROCESS_PROMPTS:
            enhancements.append(POSTPROCESS_PROMPTS[tag])
    if enhancements:
        return f"{prompt}, {', '.join(enhancements)}"
    return prompt

def call_generate_api(api_url: str, api_key: str, model: str, prompt: str, size: str, input_image_path: Optional[str] = None) -> bytes:
    timeout_seconds, max_attempts = get_image_request_options()
    if input_image_path:
        edit_url = api_url.replace('/images/generations', '/images/edits')
        with open(input_image_path, 'rb') as f:
            image_data = f.read()
        mime_type = infer_mime_type(input_image_path)
        files = {'image': (Path(input_image_path).name, image_data, mime_type)}
        headers = {'Authorization': f'Bearer {api_key}'}
        data = {'model': model, 'prompt': prompt, 'size': size}
        response = requests.post(edit_url, headers=headers, data=data, files=files, timeout=timeout_seconds)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code} {response.text[:200]}")
        return extract_image_from_response(response_json_or_raise(response))
    else:
        payload = {'model': model, 'prompt': prompt, 'size': size, 'n': 1}
        response_json = fetch_with_retry(api_url, api_key, payload, max_attempts=max_attempts, timeout=timeout_seconds)
        return extract_image_from_response(response_json)

def analyze_image_with_qwen(image_bytes: bytes, mime_type: str, instruction: str) -> dict:
    api_url, api_key, model = get_vision_config()
    if not api_url or not api_key:
        raise HTTPException(status_code=500, detail="图片分析 API 配置缺失")

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    user_prompt = instruction.strip() or (
        "请仔细分析这张图片，并反推出可用于 AI 生图的高质量提示词。"
        "输出包含：1. 画面描述；2. 可直接复制的中文提示词；3. 英文提示词；"
        "4. 构图、镜头、光线、风格、材质、色彩等关键词。"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是专业的 AI 图像提示词反推助手。"
                    "你会从主体、场景、构图、镜头、光线、材质、色彩、风格和负面约束等维度分析图片，"
                    "并给出可直接用于文生图或图生图的提示词。"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"vision upstream HTTP {response.status_code}: {response.text[:300]}")
    data = response_json_or_raise(response)
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    return {
        "content": content,
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
    }

# ============================================
# FastAPI应用
# ============================================

app = FastAPI(title="AI图像工坊", version="1.0")

# 启动前确保静态目录存在（避免 StaticFiles 找不到目录）
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 静态文件服务
app.mount("/history/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/history/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/template_images", StaticFiles(directory=str(TEMPLATE_IMAGES_DIR)), name="template_images")

# ============================================
# 登录相关API
# ============================================

@app.get("/login")
async def login_page():
    """返回登录页面"""
    html_path = Path(__file__).parent / 'login.html'
    return FileResponse(html_path)

@app.post("/api/login")
async def login(login_request: LoginRequest, request: Request):
    """登录验证"""
    username = login_request.username
    password = login_request.password
    users = load_user_passwords()

    if username in users and verify_password(password, users[username]):
        token, max_age = create_session(username)
        response = JSONResponse({"success": True, "token": token, "username": username})
        response.set_cookie(
            "auth_token",
            token,
            max_age=max_age,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
        return response

    return JSONResponse({"success": False, "message": "账号或密码错误"}, status_code=401)

@app.get("/api/verify")
async def verify_token(request: Request):
    """验证token有效性"""
    username = verify_session(request)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False}

@app.get("/api/logout")
async def logout(request: Request):
    """退出登录"""
    token = request.cookies.get('auth_token')
    if token:
        delete_session(token)
    response = RedirectResponse(url="/login")
    response.delete_cookie("auth_token")
    return response

@app.post("/api/change-password")
async def change_password(request: ChangePasswordRequest):
    """修改密码：必须提供旧密码。"""
    username = request.username.strip()

    if len(request.new_password) < 6:
        return JSONResponse({"success": False, "message": "密码至少6位"}, status_code=400)

    users = load_user_passwords()
    if username not in users:
        return JSONResponse({"success": False, "message": "账号不存在"}, status_code=404)

    if not request.old_password or not verify_password(request.old_password, users[username]):
        return JSONResponse({"success": False, "message": "旧密码不正确"}, status_code=403)

    users[username] = hash_password(request.new_password)
    save_user_passwords(users)
    invalidate_user_sessions(username)

    return {"success": True, "message": "密码修改成功"}

# ============================================
# 页面路由
# ============================================

@app.get("/")
async def index(request: Request):
    """用户主页"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")

    html_path = Path(__file__).parent / 'code.html'
    return FileResponse(html_path)

@app.get("/admin")
async def admin_page(request: Request):
    """管理员页面"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username != 'admin':
        return RedirectResponse(url="/")

    html_path = Path(__file__).parent / 'admin.html'
    return FileResponse(html_path)

@app.get("/templates")
async def templates(request: Request):
    """提示词模板页面"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")

    html_path = Path(__file__).parent / 'templates.html'
    return FileResponse(html_path)

@app.get("/chat")
async def chat_page(request: Request):
    """文案生成（聊天）页面"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")

    html_path = Path(__file__).parent / 'chat.html'
    return FileResponse(html_path)

@app.get("/video")
async def video_page(request: Request):
    """视频生成页面"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")

    html_path = Path(__file__).parent / 'video.html'
    return FileResponse(html_path)

@app.get("/analyze")
async def analyze_page(request: Request):
    """图片反推页面"""
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")

    html_path = Path(__file__).parent / 'analyze.html'
    return FileResponse(html_path)

# ============================================
# 用户API
# ============================================

@app.get("/api/credits")
async def get_user_credits(request: Request):
    """获取用户积分"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    credits = load_user_credits()
    return {"credits": credits.get(username, 0)}

@app.post("/api/credits/apply")
async def apply_credits(request: Request, apply_request: CreditsApplyRequest):
    """提交积分申请"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    if apply_request.type not in EXCHANGE_RATES:
        return JSONResponse({"success": False, "message": "无效的兑换类型"}, status_code=400)
    if apply_request.count < 1 or apply_request.count > 100:
        return JSONResponse({"success": False, "message": "兑换数量必须在 1 到 100 之间"}, status_code=400)

    credits_to_add = EXCHANGE_RATES[apply_request.type] * apply_request.count

    application = {
        "id": f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "username": username,
        "type": apply_request.type,
        "count": apply_request.count,
        "credits_to_add": credits_to_add,
        "status": "pending",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    applications = load_credits_applications()
    applications.insert(0, application)
    save_credits_applications(applications)

    return {"success": True, "message": "申请已提交，等待管理员审核"}

# ============================================
# 管理员API
# ============================================

@app.get("/api/admin/user-password")
async def get_user_password(request: Request):
    """兼容旧前端：不再返回明文密码。"""
    require_admin(request)
    return {"success": True, "password": "已加密保存", "password_visible": False}

@app.get("/api/admin/user-credits")
async def get_user_credits_admin(request: Request):
    """兼容旧前端：返回一个普通用户的积分概览（仅管理员）"""
    require_admin(request)
    users = load_user_passwords()
    credits = load_user_credits()
    admin_username = get_initial_setting("INITIAL_ADMIN_USERNAME", "admin")
    preferred_user = get_initial_setting("INITIAL_USER_USERNAME")
    candidates = [u for u in users if u != admin_username]
    username = preferred_user if preferred_user in users else (candidates[0] if candidates else None)
    return {
        "success": True,
        "username": username,
        "credits": credits.get(username, 0) if username else 0,
    }

@app.get("/api/credits/pending")
async def get_pending_applications(request: Request):
    """获取待审核申请（仅管理员）"""
    require_admin(request)
    applications = load_credits_applications()
    pending = [a for a in applications if a['status'] == 'pending']
    return {"applications": pending}

@app.get("/api/credits/processed")
async def get_processed_applications(request: Request):
    """获取已处理申请（仅管理员）"""
    require_admin(request)
    applications = load_credits_applications()
    processed = [a for a in applications if a['status'] != 'pending']
    return {"applications": processed}

@app.post("/api/credits/approve/{app_id}")
async def approve_application(request: Request, app_id: str):
    """批准申请（仅管理员）"""
    require_admin(request)

    applications = load_credits_applications()
    for app in applications:
        if app['id'] == app_id and app['status'] == 'pending':
            app['status'] = 'approved'
            app['approved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_credits_applications(applications)

            # 增加用户积分
            credits = load_user_credits()
            user = app['username']
            credits[user] = credits.get(user, 0) + app['credits_to_add']
            save_user_credits(credits)
            add_credits_ledger(user, app['credits_to_add'], credits[user], "approve",
                               f"管理员批准: {app.get('type','')} x{app.get('count',1)}")

            return {"success": True, "message": "已批准"}

    return JSONResponse({"success": False, "message": "申请不存在或已处理"}, status_code=404)

@app.post("/api/credits/reject/{app_id}")
async def reject_application(request: Request, app_id: str):
    """拒绝申请（仅管理员）"""
    require_admin(request)

    applications = load_credits_applications()
    for app in applications:
        if app['id'] == app_id and app['status'] == 'pending':
            app['status'] = 'rejected'
            app['rejected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_credits_applications(applications)
            return {"success": True, "message": "已拒绝"}

    return JSONResponse({"success": False, "message": "申请不存在或已处理"}, status_code=404)

@app.get("/api/admin/all-users")
async def get_all_users(request: Request):
    """获取所有账号列表（仅管理员）"""
    require_admin(request)

    passwords = load_user_passwords()
    credits = load_user_credits()

    users = []
    for username in passwords:
        users.append({
            "username": username,
            "password_status": "已加密",
            "credits": credits.get(username, 0)
        })

    return {"success": True, "users": users}

@app.post("/api/admin/add-user")
async def add_user(request: Request, body: AddUserRequest):
    """新增账号（仅管理员）"""
    require_admin(request)

    if len(body.username) < 3:
        return JSONResponse({"success": False, "message": "账号至少3位"}, status_code=400)

    if len(body.password) < 6:
        return JSONResponse({"success": False, "message": "密码至少6位"}, status_code=400)

    passwords = load_user_passwords()
    if body.username in passwords:
        return JSONResponse({"success": False, "message": "账号已存在"}, status_code=400)

    passwords[body.username] = hash_password(body.password)
    save_user_passwords(passwords)

    credits = load_user_credits()
    credits[body.username] = body.credits
    save_user_credits(credits)

    return {"success": True, "message": "账号添加成功"}

@app.post("/api/admin/set-user-credits")
async def set_user_credits(request: Request, body: SetCreditsRequest):
    """管理员直接修改某用户积分（绝对值，不是增量）"""
    require_admin(request)

    if body.credits < 0:
        return JSONResponse({"success": False, "message": "积分不能为负数"}, status_code=400)

    passwords = load_user_passwords()
    if body.username not in passwords:
        return JSONResponse({"success": False, "message": "账号不存在"}, status_code=404)

    credits = load_user_credits()
    old_value = credits.get(body.username, 0)
    delta = body.credits - old_value
    credits[body.username] = body.credits
    save_user_credits(credits)

    if delta != 0:
        detail = f"管理员调整积分 {old_value} -> {body.credits}"
        if body.reason:
            detail += f"（备注：{body.reason}）"
        add_credits_ledger(body.username, delta, body.credits, "admin_grant", detail)

    return {"success": True, "message": "积分已更新", "username": body.username,
            "old_credits": old_value, "new_credits": body.credits, "delta": delta}

@app.post("/api/admin/delete-user/{username}")
async def delete_user(request: Request, username: str):
    """删除账号（仅管理员）"""
    require_admin(request)

    if username == 'admin':
        return JSONResponse({"success": False, "message": "不能删除管理员账号"}, status_code=403)

    passwords = load_user_passwords()
    if username not in passwords:
        return JSONResponse({"success": False, "message": "账号不存在"}, status_code=404)

    del passwords[username]
    save_user_passwords(passwords)

    credits = load_user_credits()
    if username in credits:
        del credits[username]
    save_user_credits(credits)

    return {"success": True, "message": "账号已删除"}

# ============================================
# 图像生成API
# ============================================

@app.get("/api/models")
async def get_models(request: Request):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    env_config = load_env_config()
    models = DEFAULT_MODELS.copy()
    default_model = env_config.get('MODEL')
    if default_model and default_model not in [m['id'] for m in models]:
        models.insert(0, {"id": default_model, "name": default_model, "description": "当前配置"})
    return {"models": models}

@app.get("/api/history")
async def get_history(request: Request, limit: int = 50):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    history = load_history()
    if username != 'admin':
        history = [h for h in history if h.get('user') == username]
    return {"history": history[:limit]}

@app.post("/api/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    if file.content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPEG/WebP 图片")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 8MB")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"upload_{timestamp}_{sanitize_filename(file.filename)}"
    file_path = UPLOADS_DIR / filename

    with open(file_path, 'wb') as f:
        f.write(content)

    return {"success": True, "filename": filename, "path": f"/uploads/{filename}"}

@app.post("/api/analyze-image")
async def analyze_image_endpoint(
    request: Request,
    file: UploadFile = File(...),
    instruction: str = Form("")
):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    if file.content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 PNG/JPEG/WebP 图片")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 8MB")

    started_at = time.time()
    try:
        result = analyze_image_with_qwen(content, file.content_type or "image/png", instruction)
        elapsed = round(time.time() - started_at, 2)
        print(f"图片反推完成: user={username}, model={result.get('model')}, elapsed={elapsed}s")
        return {
            "success": True,
            "content": result["content"],
            "model": result.get("model"),
            "usage": result.get("usage", {}),
            "elapsed_seconds": elapsed,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"图片反推失败: user={username}, elapsed={round(time.time() - started_at, 2)}s, error={e}")
        raise safe_http_error(500, "图片反推失败，请稍后重试或检查图片格式")

@app.post("/api/generate")
async def generate_image(request: Request, gen_request: GenerateRequest):
    started_at = time.time()
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    # 检查积分
    credits = load_user_credits()
    user_credits = credits.get(username, 0)
    if user_credits < 10:
        raise HTTPException(status_code=403, detail="积分不足")

    api_url, api_key = get_api_config()
    if not api_url or not api_key:
        raise HTTPException(status_code=500, detail="API配置缺失")

    model_sizes = MODEL_SIZE_LIMITS.get(gen_request.model, {})
    if model_sizes and "sizes" in model_sizes:
        size = model_sizes["sizes"].get(gen_request.aspect_ratio, "1024x1024")
    else:
        size = ASPECT_RATIO_SIZES.get(gen_request.aspect_ratio, "1024x1024")

    enhanced_prompt = enhance_prompt_with_postprocess(gen_request.prompt, gen_request.postprocess, gen_request.opacity)

    input_image_path = None
    if gen_request.input_image_base64:
        temp_path = UPLOADS_DIR / f"temp_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        input_b64 = gen_request.input_image_base64
        if ',' in input_b64 and input_b64.strip().startswith('data:'):
            input_b64 = input_b64.split(',', 1)[1]
        try:
            image_bytes = base64.b64decode(input_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"参考图 base64 解码失败: {e}")
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        input_image_path = str(temp_path)

    try:
        print(f"生成图片: model={gen_request.model}, size={size}, user={username}")
        api_started_at = time.time()
        image_bytes = call_generate_api(api_url, api_key, gen_request.model, enhanced_prompt, size, input_image_path)
        api_elapsed = round(time.time() - api_started_at, 2)

        # 确保输出为真正的 PNG 格式
        image_bytes = ensure_png_format(image_bytes)

        filename = generate_image_filename(gen_request.prompt)
        output_path = OUTPUTS_DIR / filename

        with open(output_path, 'wb') as f:
            f.write(image_bytes)

        history_item = {
            "id": filename.replace('.png', ''),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prompt": gen_request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "model": gen_request.model,
            "size": size,
            "aspect_ratio": gen_request.aspect_ratio,
            "postprocess": gen_request.postprocess,
            "image_path": f"/history/outputs/{filename}",
            "input_image": gen_request.input_image_base64 is not None,
            "user": username
        }

        history = load_history()
        history.insert(0, history_item)
        save_history(history)

        # 扣除积分
        credits[username] = user_credits - 10
        save_user_credits(credits)
        add_credits_ledger(username, -10, credits[username], "generate_image",
                           f"生成图片 [{gen_request.model}] {gen_request.prompt[:30]}")

        total_elapsed = round(time.time() - started_at, 2)
        print(f"生成图片完成: user={username}, api_elapsed={api_elapsed}s, total_elapsed={total_elapsed}s")

        return {
            "success": True,
            "image_path": f"/history/outputs/{filename}",
            "history_item": history_item,
            "remaining_credits": credits[username],
            "elapsed_seconds": total_elapsed,
            "api_elapsed_seconds": api_elapsed,
        }

    except Exception as e:
        print(f"生成图片失败: user={username}, elapsed={round(time.time() - started_at, 2)}s, error={e}")
        raise safe_http_error(500, "图片生成失败，请稍后重试或切换模型")

    finally:
        if input_image_path and Path(input_image_path).exists():
            Path(input_image_path).unlink()

@app.delete("/api/history/{item_id}")
async def delete_history_item(request: Request, item_id: str):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    history = load_history()
    for i, item in enumerate(history):
        if item['id'] == item_id:
            if username != 'admin' and item.get('user') != username:
                raise HTTPException(status_code=403, detail="无权删除")
            image_path = OUTPUTS_DIR / f"{item_id}.png"
            if image_path.exists():
                image_path.unlink()
            history.pop(i)
            save_history(history)
            return {"success": True, "message": "已删除"}

    raise HTTPException(status_code=404, detail="记录不存在")

# ============================================
# 视频生成 API
# ============================================

@app.get("/api/video-models")
async def get_video_models(request: Request):
    """视频生成可用模型列表"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    return {
        "default": VIDEO_DEFAULT_MODEL,
        "models": VIDEO_AVAILABLE_MODELS,
        "credits_cost": VIDEO_CREDITS_COST,
    }

@app.get("/api/video-history")
async def get_video_history(request: Request, limit: int = 50):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    history = load_video_history()
    # 仅返回当前用户的视频（admin 看全部）
    if username != 'admin':
        history = [h for h in history if h.get('user') == username]
    return {"history": history[:limit]}

@app.delete("/api/video-history/{item_id}")
async def delete_video_history(request: Request, item_id: str):
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    history = load_video_history()
    for i, item in enumerate(history):
        if item['id'] == item_id:
            if username != 'admin' and item.get('user') != username:
                raise HTTPException(status_code=403, detail="无权删除")
            video_path = VIDEOS_DIR / f"{item_id}.mp4"
            if video_path.exists():
                video_path.unlink()
            history.pop(i)
            save_video_history(history)
            return {"success": True}
    raise HTTPException(status_code=404, detail="记录不存在")

@app.post("/api/generate-video")
async def generate_video_endpoint(request: Request, gen_request: VideoGenerateRequest):
    """提交视频生成任务（立即返回 task_id，后台线程实际生成）"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    # 检查积分
    credits = load_user_credits()
    user_credits = credits.get(username, 0)
    if user_credits < VIDEO_CREDITS_COST:
        raise HTTPException(
            status_code=403,
            detail=f"积分不足，生成视频需要 {VIDEO_CREDITS_COST} 积分（当前 {user_credits}）"
        )

    api_key = get_video_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="视频生成 API Key 未配置（CHAT_API_KEY 或 VIDEO_API_KEY）")

    # 处理首帧图（在提交前完成解码 + 临时落盘）
    input_image_path = None
    if gen_request.input_image_base64:
        b64 = gen_request.input_image_base64
        if ',' in b64 and b64.strip().startswith('data:'):
            b64 = b64.split(',', 1)[1]
        try:
            image_bytes = base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"首帧图片 base64 解码失败: {e}")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_path = UPLOADS_DIR / f"video_input_{ts}.png"
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        input_image_path = str(temp_path)

    # 创建任务
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "user": username,
        "status": "pending",
        "stage": "queued",
        "created_at": time.time(),
        "elapsed": 0,
        "model": gen_request.model,
        "prompt": gen_request.prompt,
        "aspect_ratio": gen_request.aspect_ratio,
        "resolution": gen_request.resolution,
        "duration_seconds": gen_request.duration_seconds,
        "generate_audio": gen_request.generate_audio,
        "history_item": None,
        "remaining_credits": None,
        "error": None,
    }
    with VIDEO_TASKS_LOCK:
        VIDEO_TASKS[task_id] = task

    # 启动后台线程
    t = threading.Thread(
        target=_run_video_task,
        args=(task_id, api_key, gen_request, input_image_path, username),
        daemon=True,
    )
    t.start()

    return {"success": True, "task_id": task_id, "status": "pending"}


def _run_video_task(task_id: str, api_key: str, gen_request: "VideoGenerateRequest",
                    input_image_path: Optional[str], username: str):
    """后台线程：实际跑生成、落盘、扣分、写历史"""
    def _update(**kwargs):
        with VIDEO_TASKS_LOCK:
            t = VIDEO_TASKS.get(task_id)
            if t:
                t.update(kwargs)
                t["elapsed"] = int(time.time() - t["created_at"])

    _update(status="running", stage="submitting")
    try:
        print(
            f"[video] 任务 {task_id} 开始: model={gen_request.model} "
            f"ratio={gen_request.aspect_ratio} res={gen_request.resolution} "
            f"dur={gen_request.duration_seconds}s user={username}"
        )

        def _on_status(s: str):
            print(f"[video] {task_id[:8]} {s}")
            _update(stage=s)

        result = call_generate_video_api(
            api_key=api_key,
            model=gen_request.model,
            prompt=gen_request.prompt,
            image_path=input_image_path,
            aspect_ratio=gen_request.aspect_ratio,
            resolution=gen_request.resolution,
            duration_seconds=gen_request.duration_seconds,
            generate_audio=gen_request.generate_audio,
            on_status=_on_status,
        )

        video_bytes = result.get("video_bytes")
        video_uri = result.get("video_uri")
        if not video_bytes:
            raise RuntimeError(f"视频生成失败：未拿到视频数据 uri={video_uri}")

        _update(stage="saving")
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        filename = generate_video_filename(gen_request.prompt)
        output_path = VIDEOS_DIR / filename
        with open(output_path, 'wb') as f:
            f.write(video_bytes)

        item_id = filename.replace('.mp4', '')
        history_item = {
            "id": item_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prompt": gen_request.prompt,
            "model": result.get("model", gen_request.model),
            "model_requested": gen_request.model,
            "aspect_ratio": gen_request.aspect_ratio,
            "resolution": gen_request.resolution,
            "duration_seconds": gen_request.duration_seconds,
            "generate_audio": gen_request.generate_audio,
            "video_path": f"/history/videos/{filename}",
            "input_image": gen_request.input_image_base64 is not None,
            "user": username,
            "remote_uri": video_uri,
        }

        history = load_video_history()
        history.insert(0, history_item)
        save_video_history(history)

        # 扣积分
        credits = load_user_credits()
        cur = credits.get(username, 0)
        credits[username] = max(0, cur - VIDEO_CREDITS_COST)
        save_user_credits(credits)
        add_credits_ledger(
            username, -VIDEO_CREDITS_COST, credits[username], "generate_video",
            f"生成视频 [{gen_request.model}] {gen_request.prompt[:30]}"
        )

        _update(
            status="done",
            stage="completed",
            history_item=history_item,
            remaining_credits=credits[username],
        )
        print(f"[video] 任务 {task_id} 完成 -> {filename}")

    except Exception as e:
        print(f"[video] 任务 {task_id} 失败: {e}")
        _update(status="failed", stage="failed", error=str(e))
    finally:
        if input_image_path and Path(input_image_path).exists():
            try:
                Path(input_image_path).unlink()
            except Exception:
                pass


@app.get("/api/video-task/{task_id}")
async def get_video_task(request: Request, task_id: str):
    """查询视频生成任务状态"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    with VIDEO_TASKS_LOCK:
        task = VIDEO_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        # 仅本人可查
        if task.get("user") != username:
            raise HTTPException(status_code=403, detail="无权查看该任务")
        # 实时刷新 elapsed
        elapsed = int(time.time() - task["created_at"])
        return {
            "task_id": task_id,
            "status": task["status"],
            "stage": task.get("stage"),
            "elapsed": elapsed,
            "history_item": task.get("history_item"),
            "remaining_credits": task.get("remaining_credits"),
            "error": task.get("error"),
        }

# ============================================
# 文案生成（Chat）API
# ============================================

# Chat 模型信息（展示用）
CHAT_MODEL_INFO = {
    "id": "deepseek/deepseek-v4-flash-free",
    "display_name": "DeepSeek V4 Flash (Free)",
    "owned_by": "DeepSeek",
    "context_length": 1000000,
    "input_modalities": ["text"],
    "output_modalities": ["text"],
    "capabilities": {"reasoning": True},
    "pricing": "免费",
    "endpoint": "ZenMux (https://zenmux.ai)",
    "description": "支持 100 万 tokens 超长上下文，具备思维链推理能力，适合长文创作、代码生成、复杂推理。"
}

def get_chat_config() -> tuple:
    """读取 chat API 配置"""
    env_config = load_env_config()
    api_url = os.environ.get('CHAT_API_URL') or env_config.get('CHAT_API_URL')
    api_key = os.environ.get('CHAT_API_KEY') or env_config.get('CHAT_API_KEY')
    model = os.environ.get('CHAT_MODEL') or env_config.get('CHAT_MODEL') or CHAT_MODEL_INFO['id']
    return api_url, api_key, model

@app.get("/api/chat/model-info")
async def chat_model_info(request: Request):
    """获取当前 chat 模型的元信息"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    _, _, model = get_chat_config()
    info = dict(CHAT_MODEL_INFO)
    info['id'] = model
    return info

_FREE_MODELS_CACHE = {"ts": 0.0, "models": []}

def _fetch_free_models() -> list:
    """从 zenmux /models 拉取免费模型清单（id 以 -free 结尾），10 分钟缓存。"""
    import time as _time
    if _FREE_MODELS_CACHE["models"] and (_time.time() - _FREE_MODELS_CACHE["ts"] < 600):
        return _FREE_MODELS_CACHE["models"]
    env_config = load_env_config()
    api_url = os.environ.get('CHAT_API_URL') or env_config.get('CHAT_API_URL') or ''
    api_key = os.environ.get('CHAT_API_KEY') or env_config.get('CHAT_API_KEY') or ''
    # /chat/completions -> /models
    base = api_url.rsplit('/chat/completions', 1)[0]
    models_url = base + '/models'
    items = []
    try:
        ca_bundle = True
        for p in ['/etc/ssl/cert.pem']:
            if os.path.exists(p):
                ca_bundle = p
                break
        resp = requests.get(models_url, headers={"Authorization": f"Bearer {api_key}"},
                            timeout=15, proxies={"http": None, "https": None}, verify=ca_bundle)
        if resp.status_code == 200:
            data = resp.json()
            for m in (data.get('data') or data.get('models') or []):
                mid = m.get('id') or ''
                if not mid.endswith('-free'):
                    continue
                label = m.get('display_name') or m.get('name') or mid
                # 简化 label：去掉 (Free) 之类
                items.append({"id": mid, "label": label})
    except Exception as e:
        print(f"[chat_models] fetch free models failed: {e}")
    if items:
        _FREE_MODELS_CACHE["ts"] = _time.time()
        _FREE_MODELS_CACHE["models"] = items
    return items


@app.get("/api/chat/models")
async def chat_models(request: Request):
    """返回当前账号可用的免费模型清单（自动从上游拉取并过滤）。"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    _, _, default_model = get_chat_config()
    items = _fetch_free_models()
    if not items:
        # 回退：默认模型
        items = [{"id": default_model, "label": default_model}]
    # 确保 default 在第一位（若 default 在列表中）
    items.sort(key=lambda x: 0 if x["id"] == default_model else 1)
    # 若 default 不在免费列表里，把列表第一个作为 default
    if not any(x["id"] == default_model for x in items):
        default_model = items[0]["id"]
    return {"default": default_model, "models": items}

@app.post("/api/chat")
async def chat_completion(request: Request, body: ChatRequest):
    """文案生成 - OpenAI 兼容 chat completion"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    api_url, api_key, model = get_chat_config()
    if not api_url or not api_key:
        raise HTTPException(status_code=500, detail="Chat API 配置缺失")

    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    # 若没有 system 消息，注入一条带当前北京时间的 system，便于模型回答“今天/最新”类问题
    if not any(m.get("role") == "system" for m in msgs):
        from datetime import datetime, timezone, timedelta
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        wd = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
        now_str = now.strftime(f"%Y年%m月%d日 星期{wd} %H:%M (UTC+8)")
        msgs.insert(0, {
            "role": "system",
            "content": f"你是「孔春春AI工坊」的文案助手。当前时间：{now_str}。如用户提到“今天/最新/近期”等时间相关内容，请基于此时间回答。"
        })

    payload = {
        "model": body.model or model,
        "messages": msgs,
        "temperature": body.temperature,
    }
    if body.max_tokens:
        payload["max_tokens"] = body.max_tokens
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # 选择一个可用的 CA bundle（Homebrew certifi 软链经常失效）
        ca_bundle = True
        for p in ['/etc/ssl/cert.pem', '/opt/homebrew/lib/python3.13/site-packages/pip/_vendor/certifi/cacert.pem']:
            if os.path.exists(p):
                ca_bundle = p
                break
        resp = requests.post(api_url, headers=headers, json=payload, timeout=180,
                             proxies={"http": None, "https": None}, verify=ca_bundle)
        if resp.status_code != 200:
            print(f"[chat] upstream HTTP {resp.status_code}: {resp.text[:300]}")
            raise safe_http_error(502, "文案生成服务暂时不可用，请稍后重试")
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning", "") or ""
        usage = data.get("usage", {})
        return {
            "success": True,
            "content": content,
            "reasoning": reasoning,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "model": data.get("model", model),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[chat] request failed: {e}")
        raise safe_http_error(500, "文案生成失败，请稍后重试")

# ---- Chat 会话持久化 API ----

def _summary_from_messages(messages: List[dict]) -> str:
    for m in messages:
        if m.get('role') == 'user' and m.get('content'):
            return (m['content'][:40] + '…') if len(m['content']) > 40 else m['content']
    return '(空对话)'

@app.post("/api/chat/sessions")
async def save_chat_session(request: Request, body: ChatSessionSaveRequest):
    """保存/更新一个 chat 会话（按当前登录用户隔离）"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    session_type = normalize_session_type(body.session_type)
    sessions = load_user_chat_sessions(username, session_type)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sid = body.session_id

    if sid:
        # 更新
        for s in sessions:
            if s.get('id') == sid:
                s['messages'] = body.messages
                s['updated_at'] = now
                s['model'] = body.model or s.get('model')
                s['usage'] = body.usage or s.get('usage')
                s['type'] = session_type
                s['title'] = _summary_from_messages(body.messages)
                s['msg_count'] = len(body.messages)
                save_user_chat_sessions(username, sessions, session_type)
                return {"success": True, "session_id": sid}

    # 新建
    sid = sid or f"{session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    new_session = {
        "id": sid,
        "type": session_type,
        "title": _summary_from_messages(body.messages),
        "messages": body.messages,
        "model": body.model,
        "usage": body.usage or {},
        "created_at": now,
        "updated_at": now,
        "msg_count": len(body.messages),
    }
    sessions.insert(0, new_session)
    save_user_chat_sessions(username, sessions, session_type)
    return {"success": True, "session_id": sid}

@app.get("/api/chat/sessions")
async def list_chat_sessions(request: Request, type: str = "chat"):
    """列出当前用户的所有 chat 会话（按更新时间倒序）"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    session_type = normalize_session_type(type)
    sessions = load_user_chat_sessions(username, session_type)
    # 列表只返回元数据
    items = [{
        "id": s['id'],
        "type": s.get('type', session_type),
        "title": s.get('title', '(无标题)'),
        "model": s.get('model'),
        "msg_count": s.get('msg_count', len(s.get('messages', []))),
        "created_at": s.get('created_at'),
        "updated_at": s.get('updated_at'),
    } for s in sessions]
    items.sort(key=lambda x: x.get('updated_at') or '', reverse=True)
    return {"sessions": items}

@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session(request: Request, session_id: str, type: str = "chat"):
    """获取指定会话的完整消息"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    session_type = normalize_session_type(type)
    sessions = load_user_chat_sessions(username, session_type)
    for s in sessions:
        if s.get('id') == session_id:
            return {"session": s}
    raise HTTPException(status_code=404, detail="会话不存在")

@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(request: Request, session_id: str, type: str = "chat"):
    """删除指定会话"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    session_type = normalize_session_type(type)
    sessions = load_user_chat_sessions(username, session_type)
    new_sessions = [s for s in sessions if s.get('id') != session_id]
    if len(new_sessions) == len(sessions):
        raise HTTPException(status_code=404, detail="会话不存在")
    save_user_chat_sessions(username, new_sessions, session_type)
    return {"success": True}

# ---- 个人中心 API ----

@app.get("/api/profile")
async def api_profile(request: Request):
    """返回当前用户的个人中心数据"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    credits = load_user_credits().get(username, 0)

    # 积分流水
    ledger = [x for x in load_credits_ledger() if x.get('username') == username]

    # 生图明细（仅当前用户）
    history = [h for h in load_history() if h.get('user') == username]
    image_records = [{
        "id": h.get('id'),
        "timestamp": h.get('timestamp'),
        "prompt": h.get('prompt'),
        "model": h.get('model'),
        "size": h.get('size'),
        "image_path": h.get('image_path'),
    } for h in history]

    # 文案明细
    chat_sessions = load_user_chat_sessions(username)
    chat_records = [{
        "id": s.get('id'),
        "title": s.get('title'),
        "model": s.get('model'),
        "msg_count": s.get('msg_count'),
        "created_at": s.get('created_at'),
        "updated_at": s.get('updated_at'),
    } for s in chat_sessions]

    features = [
        {"name": "AI 图像生成", "desc": "文生图 / 图生图，多模型多尺寸", "icon": "image", "url": "/"},
        {"name": "图片反推", "desc": "Qwen 多模态图片分析与提示词反推", "icon": "auto_awesome", "url": "/analyze"},
        {"name": "AI 文案生成", "desc": "DeepSeek V4 Flash 长上下文对话", "icon": "chat", "url": "/chat?mode=chat"},
        {"name": "文案历史", "desc": "按用户隔离的文案对话记录", "icon": "history", "url": "/chat/history?type=chat"},
        {"name": "助手历史", "desc": "智能助手独立对话记录", "icon": "smart_toy", "url": "/chat/history?type=agent"},
        {"name": "提示词模板", "desc": "12 个精选提示词快速生成", "icon": "menu_book", "url": "/templates"},
        {"name": "积分申请", "desc": "提交兑换申请，管理员审核", "icon": "stars", "url": "/"},
    ]

    return {
        "username": username,
        "is_admin": username == 'admin',
        "credits": credits,
        "features": features,
        "credits_ledger": ledger[:200],
        "image_records": image_records[:200],
        "chat_records": chat_records[:200],
        "stats": {
            "image_count": len(image_records),
            "chat_count": len(chat_records),
            "ledger_count": len(ledger),
        }
    }

# ---- 新页面路由 ----

@app.get("/chat/history")
async def chat_history_page(request: Request):
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")
    return FileResponse(Path(__file__).parent / 'chat_history.html')

@app.get("/profile")
async def profile_page(request: Request):
    username = verify_session(request)
    if not username:
        return RedirectResponse(url="/login")
    if username == 'admin':
        return RedirectResponse(url="/admin")
    return FileResponse(Path(__file__).parent / 'profile.html')

# ============================================
# Prompt-Agent 路由（DeepSeek + 本地工具 + MCP，ReAct 风格）
# ============================================

class AgentChatRequest(BaseModel):
    messages: List[dict]   # [{"role": "user"|"assistant", "content": "..."}]
    stream: bool = True
    temperature: Optional[float] = 0.5
    model: Optional[str] = None

@app.post("/api/agent/chat")
async def agent_chat(request: Request, body: AgentChatRequest):
    """ReAct 风格智能体聊天端点。
    - stream=True: SSE
    - stream=False: JSON {content, trace}
    """
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")

    try:
        from llm.prompt_agent import run_once, run_stream
    except Exception as e:
        print(f"[agent] module load failed: {e}")
        raise safe_http_error(500, "智能助手暂时不可用")

    if not body.stream:
        try:
            result = await run_once(body.messages, temperature=body.temperature or 0.5, model=body.model)
            return {"success": True, **result}
        except Exception as e:
            print(f"[agent] non-stream failed: {e}")
            raise safe_http_error(500, "智能助手请求失败，请稍后重试")

    from fastapi.responses import StreamingResponse

    async def sse_gen():
        try:
            async for ev in run_stream(body.messages, temperature=body.temperature or 0.5, model=body.model):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[agent] stream failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': '智能助手请求失败，请稍后重试'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


@app.get("/api/agent/tools")
async def agent_tools(request: Request):
    """列出当前可用的工具（本地 + MCP）"""
    username = verify_session(request)
    if not username:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        from llm.prompt_agent import list_tool_summaries, ensure_tools_loaded_async, reset_tools
        reset_tools()
        await ensure_tools_loaded_async()
        tools = list_tool_summaries()
        return {"count": len(tools), "tools": tools}
    except Exception as e:
        print(f"[agent_tools] load failed: {e}")
        raise safe_http_error(500, "工具列表加载失败")


# ============================================
# 启动服务
# ============================================

if __name__ == '__main__':
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化数据目录
    (BASE_DIR / 'data').mkdir(parents=True, exist_ok=True)
    CHAT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("AI图像工坊服务启动...")
    print(f"历史记录目录: {HISTORY_DIR}")
    print(f"输出图片目录: {OUTPUTS_DIR}")
    print("")
    print("角色说明:")
    print("  - 管理员账号由 INITIAL_ADMIN_USERNAME 配置，登录进入审核页面")
    print("  - 普通账号可由 INITIAL_USER_USERNAME 初始化，或由管理员在后台新增")

    uvicorn.run(app, host="0.0.0.0", port=8000)
