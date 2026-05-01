#!/usr/bin/env python3
"""
视频生成客户端 - ZenMux Vertex AI 协议封装

仅使用模型: bytedance/doubao-seedance-2.0

调用流程: submit -> poll(15s) -> download
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Callable, Any

ZENMUX_VERTEX_BASE_URL = "https://zenmux.ai/api/vertex-ai"
DEFAULT_POLL_INTERVAL = 15
DEFAULT_TIMEOUT_SECONDS = 600  # 10 分钟


def _normalize_model(model: str) -> str:
    """zenmux 实际只接受 bytedance/doubao-seedance-2.0；映射常见别名"""
    if not model:
        return model
    m = model.strip()
    aliases = {
        "volcengine/doubao-seedance-2": "bytedance/doubao-seedance-2.0",
        "bytedance/doubao-seedance-2": "bytedance/doubao-seedance-2.0",
        "doubao-seedance-2": "bytedance/doubao-seedance-2.0",
        "doubao-seedance-2.0": "bytedance/doubao-seedance-2.0",
    }
    return aliases.get(m, m)


def _build_client(api_key: str):
    """构造 google-genai 客户端（指向 zenmux）"""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "缺少依赖 google-genai，请运行: pip install google-genai"
        ) from e

    client = genai.Client(
        api_key=api_key,
        vertexai=True,
        http_options=types.HttpOptions(
            api_version="v1",
            base_url=ZENMUX_VERTEX_BASE_URL,
        ),
    )
    return client, types


def generate_video(
    api_key: str,
    model: str,
    prompt: str,
    *,
    image_path: Optional[str] = None,
    aspect_ratio: Optional[str] = None,        # "16:9" / "9:16" / "1:1"
    resolution: Optional[str] = None,           # "720p" / "1080p"
    duration_seconds: Optional[int] = None,     # 5 / 8 / 10
    generate_audio: Optional[bool] = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    生成视频（同步阻塞，内部轮询）。
    返回: {"video_bytes": bytes | None, "video_uri": str | None, "model": str, "raw": Any}

    优先返回 video_bytes（已下载的二进制），URI 作为后备。
    """
    if not api_key:
        raise ValueError("api_key 不能为空")
    if not model:
        raise ValueError("model 不能为空")
    if not prompt:
        raise ValueError("prompt 不能为空")

    real_model = _normalize_model(model)
    client, types = _build_client(api_key)

    # 构造 config
    config_kwargs = {}
    if aspect_ratio:
        config_kwargs["aspectRatio"] = aspect_ratio
    if resolution:
        config_kwargs["resolution"] = resolution
    if duration_seconds is not None:
        config_kwargs["durationSeconds"] = int(duration_seconds)
    if generate_audio is not None:
        config_kwargs["generateAudio"] = bool(generate_audio)

    config_obj = types.GenerateVideosConfig(**config_kwargs) if config_kwargs else None

    # 构造 image
    image_obj = None
    if image_path:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"首帧图片不存在: {image_path}")
        suffix = p.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        image_bytes = p.read_bytes()
        image_obj = types.Image(image_bytes=image_bytes, mime_type=mime)

    # 提交
    if on_status:
        on_status(f"提交视频生成请求 model={real_model}")

    submit_kwargs: dict = {"model": real_model, "prompt": prompt}
    if image_obj is not None:
        submit_kwargs["image"] = image_obj
    if config_obj is not None:
        submit_kwargs["config"] = config_obj

    operation = client.models.generate_videos(**submit_kwargs)

    # 轮询
    start = time.time()
    while not getattr(operation, "done", False):
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            raise TimeoutError(f"视频生成超时（{timeout}s 内未完成）")
        if on_status:
            on_status(f"生成中... 已等待 {elapsed}s")
        time.sleep(poll_interval)
        operation = client.operations.get(operation)

    # 取结果
    response = getattr(operation, "response", None)
    if response is None:
        raise RuntimeError(f"视频生成失败（无 response）：{operation}")

    generated_videos = getattr(response, "generated_videos", None) or []
    if not generated_videos:
        raise RuntimeError(f"视频生成结果为空: {response}")

    video_obj = generated_videos[0]

    # 尝试取 bytes / uri
    video_bytes = None
    video_uri = None

    # google-genai 的 Video 对象常见字段：video.uri / video.video_bytes
    inner = getattr(video_obj, "video", video_obj)

    candidate_bytes_attrs = ("video_bytes", "data", "bytes_value")
    for attr in candidate_bytes_attrs:
        v = getattr(inner, attr, None)
        if v:
            video_bytes = v if isinstance(v, (bytes, bytearray)) else None
            if video_bytes:
                break

    candidate_uri_attrs = ("uri", "url", "video_uri")
    for attr in candidate_uri_attrs:
        v = getattr(inner, attr, None)
        if v:
            video_uri = str(v)
            break

    # 若有 uri 但没 bytes，用 SDK 的 download 方法
    if video_bytes is None and video_uri:
        try:
            # SDK 提供的下载（部分版本接口为 client.files.download / videos.download）
            if hasattr(client, "files") and hasattr(client.files, "download"):
                downloaded = client.files.download(file=inner)
                if isinstance(downloaded, (bytes, bytearray)):
                    video_bytes = bytes(downloaded)
        except Exception:
            pass

        # 仍然没拿到则用 requests 直拉
        if video_bytes is None:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"}
            r = requests.get(video_uri, headers=headers, timeout=180)
            if r.status_code == 200:
                video_bytes = r.content
            else:
                raise RuntimeError(
                    f"下载视频失败 HTTP {r.status_code}: {r.text[:200]}"
                )

    if not video_bytes and not video_uri:
        raise RuntimeError(f"未能从响应中提取视频: {response}")

    return {
        "video_bytes": video_bytes,
        "video_uri": video_uri,
        "model": real_model,
        "raw": response,
    }
