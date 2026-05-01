"""ReAct 风格 Prompt Agent

工作流：
    1. 把工具说明 + skill 注入到 system prompt
    2. 模型输出形如：
           Thought: 我需要查询积分
           Action: query_credits
           Action Input: {"username": "当前登录账号"}
       或最终回复：
           Final Answer: ...
    3. 后端解析 Action 块、调用工具、把结果作为 Observation 续接
    4. 循环直到 Final Answer 或步数上限
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx
from dotenv import dotenv_values

from .mcp_bridge import call_mcp_tool_async, list_all_mcp_tools_async
from .tools import LOCAL_TOOLS

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------- 配置 ----------

def _load_chat_env() -> dict:
    env = dotenv_values(PROJECT_DIR / ".env")
    return {
        "api_key": os.environ.get("CHAT_API_KEY") or env.get("CHAT_API_KEY"),
        "api_url": os.environ.get("CHAT_API_URL") or env.get("CHAT_API_URL"),
        "model": os.environ.get("AGENT_MODEL") or env.get("AGENT_MODEL")
        or os.environ.get("CHAT_MODEL") or env.get("CHAT_MODEL")
        or "deepseek/deepseek-v4-flash-free",
    }


# ---------- 工具表（本地 + MCP）----------

_TOOLS_CACHE: Dict[str, dict] = {}


def _register_local_tools():
    for t in LOCAL_TOOLS:
        _TOOLS_CACHE[t["name"]] = {
            "kind": "local",
            "name": t["name"],
            "description": t["description"],
            "args_schema": t["args_schema"],
            "fn": t["fn"],
        }


async def _register_mcp_tools_async():
    try:
        for t in await list_all_mcp_tools_async():
            _TOOLS_CACHE[t["name"]] = {
                "kind": "mcp",
                "name": t["name"],
                "description": t["description"],
                "input_schema": t.get("input_schema") or {},
                "_server": t["_server"],
            }
    except Exception as e:
        print(f"[prompt_agent] 注册 MCP 工具失败: {e}")


async def ensure_tools_loaded_async():
    if _TOOLS_CACHE:
        return
    _register_local_tools()
    await _register_mcp_tools_async()


def ensure_tools_loaded():
    """同步版本：仅注册本地工具（FastAPI 内不要调）"""
    if _TOOLS_CACHE:
        return
    _register_local_tools()


def list_tool_summaries() -> List[dict]:
    out = []
    for name, t in _TOOLS_CACHE.items():
        out.append({
            "name": name,
            "kind": t["kind"],
            "description": t["description"][:200],
        })
    return out


def reset_tools():
    _TOOLS_CACHE.clear()


# ---------- Skill 加载 ----------

SKILLS_DIR = PROJECT_DIR / "skills"


def load_skills() -> str:
    """读取 skills/ 目录下所有 SKILL.md，拼成段落注入 system prompt。"""
    if not SKILLS_DIR.exists():
        return ""
    chunks = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        md = skill_dir / "SKILL.md"
        if md.exists():
            try:
                chunks.append(f"### Skill: {skill_dir.name}\n{md.read_text(encoding='utf-8').strip()}")
            except Exception:
                pass
    return "\n\n".join(chunks)


# ---------- Prompt 构建 ----------

def _format_tool_for_prompt(t: dict) -> str:
    if t["kind"] == "local":
        args_lines = []
        for k, v in t["args_schema"].items():
            req = "必填" if v.get("required") else "可选"
            args_lines.append(f"  - {k} ({v.get('type','string')}, {req}): {v.get('desc','')}")
        return f"- {t['name']}: {t['description']}\n" + "\n".join(args_lines)
    else:
        # MCP 工具：把 inputSchema.properties 摘要出来
        schema = t.get("input_schema") or {}
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required") or [])
        args_lines = []
        for k, v in props.items():
            req = "必填" if k in required else "可选"
            args_lines.append(f"  - {k} ({v.get('type','string')}, {req}): {v.get('description','')}")
        body = f"- {t['name']} [MCP/{t['_server']}]: {t['description']}"
        if args_lines:
            body += "\n" + "\n".join(args_lines)
        return body


REACT_INSTRUCTIONS = """你是一个能调用工具的智能体。每一步严格按以下格式输出：

Thought: 你的推理
Action: 工具名
Action Input: 一个 JSON 对象，包含工具参数

工具会被执行，结果以 Observation: ... 形式返回。然后你可以继续：

Thought: ...
Action: ...
Action Input: ...

或者，当你已经能给出最终回答时，输出：

Thought: 我已经知道答案了
Final Answer: 给用户的中文回答

严格规则：
1. 每一步只能输出一个 Action 或一个 Final Answer，不要混合
2. Action Input 必须是合法 JSON
3. 没必要调用工具时直接给 Final Answer
4. 不要编造工具返回，等待真正的 Observation
5. 涉及实时/外部信息时优先使用搜索工具（bing_search 或 web_search_ddg）
6. 涉及用户数据（积分、历史）时使用本地工具
"""


def _now_cn_str() -> str:
    """返回东八区当前时间字符串（含星期）。"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    weekday_cn = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    return now.strftime(f"%Y年%m月%d日 星期{weekday_cn} %H:%M (UTC+8)")


def build_system_prompt() -> str:
    tools_text = "\n".join(_format_tool_for_prompt(t) for t in _TOOLS_CACHE.values())
    skills_text = load_skills()
    parts = [
        "你是「孔春春AI工坊」的智能助手。",
        f"## 当前时间\n{_now_cn_str()}\n（如果用户问“今天”“最新”“近期”等时间相关问题，请基于此时间作答；联网搜索时也以此为准。）",
        REACT_INSTRUCTIONS,
        "## 可用工具\n" + tools_text,
    ]
    if skills_text:
        parts.append("## Skill 指引\n" + skills_text)
    return "\n\n".join(parts)


# ---------- 解析 ----------

_RE_ACTION = re.compile(r"Action\s*:\s*(.+?)\s*\nAction Input\s*:\s*(.+?)(?=\n(?:Thought|Action|Observation|Final Answer)\s*:|\Z)", re.S)
_RE_FINAL = re.compile(r"Final Answer\s*:\s*(.+?)\Z", re.S)


def parse_step(text: str) -> Tuple[str, dict]:
    """返回 (kind, payload):
        kind="action": payload = {"name", "args", "raw"}
        kind="final":  payload = {"text"}
        kind="raw":    payload = {"text"}  无法解析
    """
    final_m = _RE_FINAL.search(text)
    action_m = _RE_ACTION.search(text)
    # Final Answer 优先（且需要在 Action 之后）
    if final_m and (not action_m or final_m.start() > action_m.start()):
        return "final", {"text": final_m.group(1).strip()}
    if action_m:
        name = action_m.group(1).strip().strip("`'\"")
        args_raw = action_m.group(2).strip()
        # 去除可能的代码栅栏
        args_raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", args_raw.strip(), flags=re.M)
        try:
            args = json.loads(args_raw)
            if not isinstance(args, dict):
                args = {"_raw": args}
        except Exception:
            args = {"_raw": args_raw}
        return "action", {"name": name, "args": args, "raw": text}
    return "raw", {"text": text}


# ---------- LLM 调用（直连 zenmux）----------

async def _call_llm(messages: List[dict], stream: bool = False, temperature: float = 0.5, model: Optional[str] = None) -> dict:
    cfg = _load_chat_env()
    if not cfg["api_key"] or not cfg["api_url"]:
        raise RuntimeError("CHAT_API_KEY / CHAT_API_URL 未配置")

    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        # 让模型在看到 Observation 前停下来
        "stop": ["\nObservation:", "Observation:"],
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        resp = await client.post(cfg["api_url"], headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"上游错误 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        return {
            "content": msg.get("content", "") or "",
            "reasoning": msg.get("reasoning", "") or "",
            "usage": data.get("usage", {}),
        }


async def _call_llm_stream(messages: List[dict], temperature: float = 0.5, model: Optional[str] = None) -> AsyncIterator[str]:
    """流式：yield 文本片段"""
    cfg = _load_chat_env()
    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        "stop": ["\nObservation:", "Observation:"],
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        async with client.stream("POST", cfg["api_url"], headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                raise RuntimeError(f"上游错误 {resp.status_code}: {err.decode('utf-8', 'ignore')[:300]}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_str)
                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                    text = delta.get("content") or ""
                    if text:
                        yield text
                except Exception:
                    continue


# ---------- 工具调度 ----------

async def _exec_tool(name: str, args: dict) -> str:
    t = _TOOLS_CACHE.get(name)
    if not t:
        return f"[错误] 未知工具：{name}。可用工具：{list(_TOOLS_CACHE.keys())}"
    clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
    try:
        if t["kind"] == "local":
            return str(t["fn"](**clean_args))
        else:
            return await call_mcp_tool_async(t["_server"], t["name"], clean_args)
    except TypeError as e:
        return f"[参数错误] {e}"
    except Exception as e:
        return f"[执行错误] {e}"


# ---------- 主循环 ----------

MAX_STEPS = 6


async def run_once(messages: List[dict], temperature: float = 0.5, model: Optional[str] = None) -> dict:
    """非流式：跑完整轮，返回 {content, trace}"""
    await ensure_tools_loaded_async()
    system = build_system_prompt()
    chat: List[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            chat.append({"role": m["role"], "content": m.get("content", "")})

    trace: List[dict] = []
    for step in range(MAX_STEPS):
        out = await _call_llm(chat, stream=False, temperature=temperature, model=model)
        text = out["content"]
        kind, payload = parse_step(text)
        trace.append({"type": "model_output", "step": step, "text": text})

        if kind == "final":
            return {"content": payload["text"], "trace": trace, "steps": step + 1}
        if kind == "raw":
            return {"content": text, "trace": trace, "steps": step + 1}

        tool_name = payload["name"]
        tool_args = payload["args"]
        trace.append({"type": "tool_call", "name": tool_name, "args": tool_args})
        observation = await _exec_tool(tool_name, tool_args)
        trace.append({"type": "tool_result", "name": tool_name, "content": observation[:2000]})

        chat.append({"role": "assistant", "content": text})
        chat.append({"role": "user", "content": f"Observation: {observation}"})

    return {"content": "（达到最大步数仍未完成）", "trace": trace, "steps": MAX_STEPS}


async def run_stream(messages: List[dict], temperature: float = 0.5, model: Optional[str] = None) -> AsyncIterator[dict]:
    """流式：边推理边输出。"""
    await ensure_tools_loaded_async()
    system = build_system_prompt()
    chat: List[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            chat.append({"role": m["role"], "content": m.get("content", "")})

    for step in range(MAX_STEPS):
        buf = []
        async for piece in _call_llm_stream(chat, temperature=temperature, model=model):
            buf.append(piece)
            yield {"type": "token", "text": piece}
        text = "".join(buf)

        kind, payload = parse_step(text)
        if kind == "final":
            yield {"type": "final", "text": payload["text"]}
            yield {"type": "done"}
            return
        if kind == "raw":
            yield {"type": "final", "text": text}
            yield {"type": "done"}
            return

        tool_name = payload["name"]
        tool_args = payload["args"]
        yield {"type": "tool_call", "name": tool_name, "args": tool_args}
        observation = await _exec_tool(tool_name, tool_args)
        yield {"type": "tool_result", "name": tool_name, "content": observation[:2000]}

        chat.append({"role": "assistant", "content": text})
        chat.append({"role": "user", "content": f"Observation: {observation}"})

    yield {"type": "final", "text": "（达到最大步数仍未完成）"}
    yield {"type": "done"}
