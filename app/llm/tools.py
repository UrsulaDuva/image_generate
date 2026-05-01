"""本地工具集合 - prompt-式 agent 使用

每个工具是一个 dict：
    {
        "name": str,
        "description": str,
        "args_schema": {arg_name: {"type": str, "desc": str, "required": bool}},
        "fn": callable(**kwargs) -> str,
    }
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
HISTORY_DIR = PROJECT_DIR / "history" / "outputs"


def _safe_read_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


# ---------- 工具实现 ----------

def query_credits(username: str) -> str:
    credits = _safe_read_json(DATA_DIR / "user_credits.json", {})
    if username in credits:
        return f"用户 {username} 当前积分：{credits[username]}"
    return f"未找到用户 {username} 的积分记录"


def list_history(limit: int = 10) -> str:
    if not HISTORY_DIR.exists():
        return json.dumps({"items": [], "note": "history 目录尚未创建"}, ensure_ascii=False)
    files = sorted(
        [p for p in HISTORY_DIR.iterdir() if p.is_file() and not p.name.startswith(".")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[: max(1, min(int(limit or 10), 100))]
    items = [
        {
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
        }
        for f in files
    ]
    return json.dumps({"count": len(items), "items": items}, ensure_ascii=False)


def generate_image(prompt: str, size: str = "1024x1024") -> str:
    try:
        from dotenv import dotenv_values
        env = dotenv_values(PROJECT_DIR / ".env")
        api_url = env.get("API_URL")
        api_key = env.get("API_KEY")
        model = env.get("MODEL", "gemini-3.1-flash-image-preview-2k")
        if not api_url or not api_key:
            return "图像生成失败：API_URL 或 API_KEY 未配置"

        import requests
        payload = {
            "model": model, "prompt": prompt, "size": size, "n": 1,
            "response_format": "b64_json",
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120,
                             proxies={"http": None, "https": None})
        if resp.status_code != 200:
            return f"图像生成失败：{resp.status_code} {resp.text[:200]}"
        data = resp.json()
        b64 = (data.get("data") or [{}])[0].get("b64_json")
        if not b64:
            return f"图像生成失败：上游未返回图片数据"
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        fname = f"agent_{int(time.time())}_{h}.png"
        (HISTORY_DIR / fname).write_bytes(base64.b64decode(b64))
        return f"已生成图片：{fname}（保存于 history/outputs/）"
    except Exception as e:
        return f"图像生成异常：{e}"


def web_search_ddg(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max(1, min(int(max_results or 5), 10))))
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"DuckDuckGo 搜索失败：{e}"


# ---------- 工具注册表 ----------

LOCAL_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "query_credits",
        "description": "查询指定用户名的积分余额",
        "args_schema": {
            "username": {"type": "string", "desc": "用户名，例如当前登录账号", "required": True},
        },
        "fn": query_credits,
    },
    {
        "name": "list_history",
        "description": "列出最近生成的图片文件名与时间",
        "args_schema": {
            "limit": {"type": "integer", "desc": "返回数量，默认 10，最大 100", "required": False},
        },
        "fn": list_history,
    },
    {
        "name": "generate_image",
        "description": "根据提示词生成一张图片，保存到历史目录",
        "args_schema": {
            "prompt": {"type": "string", "desc": "图像提示词（中文或英文）", "required": True},
            "size": {"type": "string", "desc": "尺寸，可选 512x512 / 1024x1024 / 2048x2048", "required": False},
        },
        "fn": generate_image,
    },
    {
        "name": "web_search_ddg",
        "description": "通过 DuckDuckGo 进行联网搜索（备用，海外资讯优先）",
        "args_schema": {
            "query": {"type": "string", "desc": "搜索关键词", "required": True},
            "max_results": {"type": "integer", "desc": "结果数，默认 5", "required": False},
        },
        "fn": web_search_ddg,
    },
]
