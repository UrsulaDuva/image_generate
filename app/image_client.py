#!/usr/bin/env python3
"""
图片生成与分析客户端
参考 image-optimizer-client.mjs 的逻辑，支持从 .env 文件读取配置

功能：
1. 图片生成 (text -> image)
2. 图片分析 (image + text -> text)

使用方法：
    # 图片生成
    python image_client.py --mode generate --prompt "描述内容"
    python image_client.py --mode generate --prompt-file prompt.txt

    # 图片分析
    python image_client.py --mode analyze --input image.png --prompt "分析这张图片"
"""

import os
import sys
import argparse
import base64
import io
import json
import re
import time
import requests
from pathlib import Path
from PIL import Image

def read_env_from_file(env_path: str) -> dict:
    """从 .env 文件读取配置"""
    result = {}

    if not os.path.exists(env_path):
        return result

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行、注释、无等号的行
            if not line or line.startswith('#') or '=' not in line:
                continue

            index = line.index('=')
            key = line[:index].strip()

            if not key or key in result:
                continue

            value = line[index + 1:].strip()

            # 去除引号
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            result[key] = value

    return result


def pick_first_non_empty(*values):
    """选择第一个非空值"""
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def compress_image(image_path: Path, max_size: int = 5 * 1024 * 1024) -> str:
    """压缩图片到指定大小以下，返回 base64 编码"""
    img = Image.open(image_path)

    # 转换为 RGB
    if img.mode in ('RGBA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    quality = 85
    scale = 1.0

    while True:
        if scale < 1.0:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            resized = img

        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=quality, optimize=True)
        size = buffer.tell()

        if size <= max_size:
            buffer.seek(0)
            return base64.standard_b64encode(buffer.read()).decode('utf-8')

        if quality > 30:
            quality -= 10
        elif scale > 0.3:
            scale -= 0.1
            quality = 70
        else:
            buffer.seek(0)
            return base64.standard_b64encode(buffer.read()).decode('utf-8')


def infer_mime_type(file_path: str) -> str:
    """根据文件扩展名推断 MIME 类型"""
    ext = Path(file_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    if ext == '.png':
        return 'image/png'
    if ext == '.webp':
        return 'image/webp'
    return 'application/octet-stream'


def response_text_preview(text: str, limit: int = 220) -> str:
    """把 HTML/长文本响应压成适合错误提示的一小段。"""
    plain = re.sub(r'<script[\s\S]*?</script>', ' ', text or '', flags=re.I)
    plain = re.sub(r'<style[\s\S]*?</style>', ' ', plain, flags=re.I)
    plain = re.sub(r'<[^>]+>', ' ', plain)
    plain = re.sub(r'\s+', ' ', plain).strip()
    return (plain or (text or '').strip())[:limit]


def response_json_or_raise(response: requests.Response) -> dict:
    """解析 JSON，并在上游返回 HTML 时给出可读错误。"""
    content_type = response.headers.get('content-type', '')
    text = response.text
    if 'application/json' not in content_type.lower():
        raise Exception(
            f"上游接口返回非 JSON 响应: HTTP {response.status_code}, "
            f"Content-Type={content_type or '-'}, Body={response_text_preview(text)}"
        )
    try:
        return response.json()
    except ValueError as e:
        raise Exception(f"上游接口返回的 JSON 无法解析: {e}; Body={response_text_preview(text)}")


def fetch_with_retry(api_url: str, api_key: str, payload: dict, max_attempts: int = 6, timeout: int = 180) -> dict:
    """带重试的 API 请求"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    base_delay_ms = 2000

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=True
            )

            if response.status_code == 200:
                return response_json_or_raise(response)

            status = response.status_code

            # 可重试的状态码
            retryable = status in (429, 503, 502, 504)

            if not retryable or attempt == max_attempts:
                raise Exception(f"HTTP {status} {response.text[:200]}")

            jitter = int(time.time() * 1000) % 250
            delay_ms = min(base_delay_ms * (2 ** (attempt - 1)) + jitter, 30000)
            print(f"请求失败 (HTTP {status}), {delay_ms}ms 后重试 ({attempt}/{max_attempts})...")
            time.sleep(delay_ms / 1000)

        except requests.exceptions.Timeout:
            if attempt == max_attempts:
                raise Exception("请求超时")
            delay_ms = min(base_delay_ms * (2 ** (attempt - 1)), 30000)
            print(f"请求超时, {delay_ms}ms 后重试 ({attempt}/{max_attempts})...")
            time.sleep(delay_ms / 1000)

    raise Exception("请求重试次数已用尽")


def extract_image_from_response(response_json: dict) -> bytes:
    """从 API 响应中提取图片数据"""
    # 查找 base64 图片
    b64_candidates = [
        response_json.get('data', [{}])[0].get('b64_json'),
        response_json.get('output', [{}])[0].get('b64_json'),
        response_json.get('image', {}).get('b64_json'),
        response_json.get('b64_json'),
    ]

    for candidate in b64_candidates:
        if candidate:
            return base64.b64decode(candidate)

    # 查找 URL
    url_candidates = [
        response_json.get('data', [{}])[0].get('url'),
        response_json.get('output', [{}])[0].get('url'),
        response_json.get('image', {}).get('url'),
        response_json.get('url'),
    ]

    for url in url_candidates:
        if url:
            response = requests.get(url, timeout=60, verify=True)
            if response.status_code == 200:
                return response.content
            raise Exception(f"下载图片失败: HTTP {response.status_code}")

    raise Exception("API 响应中未找到图片数据 (b64_json/url)")


def generate_image(api_url: str, api_key: str, model: str, prompt: str, size: str, input_path: str = None) -> bytes:
    """生成图片"""

    if input_path:
        # 带参考图：使用 images/edits 端点
        edit_url = api_url.replace('/images/generations', '/images/edits')

        # 读取图片
        with open(input_path, 'rb') as f:
            image_data = f.read()

        # 这里需要 multipart/form-data，Python requests 支持
        mime_type = infer_mime_type(input_path)

        files = {
            'image': (Path(input_path).name, image_data, mime_type)
        }

        headers = {
            'Authorization': f'Bearer {api_key}'
        }

        data = {
            'model': model,
            'prompt': prompt,
            'size': size
        }

        response = requests.post(
            edit_url,
            headers=headers,
            data=data,
            files=files,
            timeout=180,
            verify=True
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code} {response.text[:200]}")

        return extract_image_from_response(response.json())

    else:
        # 纯文本生成
        payload = {
            'model': model,
            'prompt': prompt,
            'size': size,
            'n': 1
        }

        response_json = fetch_with_retry(api_url, api_key, payload)
        return extract_image_from_response(response_json)


def analyze_image(api_url: str, api_key: str, model: str, image_path: str, prompt: str) -> str:
    """分析图片（多模态理解）"""

    # 压缩图片
    image_base64 = compress_image(Path(image_path))

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            }
        ],
        'max_tokens': 4096
    }

    response_json = fetch_with_retry(api_url, api_key, payload)

    # 提取文本内容
    choices = response_json.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('content', '')

    return ''


def main():
    parser = argparse.ArgumentParser(
        description='图片生成与分析客户端',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 图片生成
  python image_client.py --mode generate --prompt "一只可爱的猫咪"
  python image_client.py --mode generate --prompt-file prompt.txt --output generated.png

  # 带参考图生成
  python image_client.py --mode generate --input ref.png --prompt "修改风格"

  # 图片分析
  python image_client.py --mode analyze --input image.png --prompt "描述这张图片"
        """
    )

    parser.add_argument('--env', default='.env', help='配置文件路径 (默认: .env)')
    parser.add_argument('--mode', choices=['generate', 'analyze'], default='generate', help='模式: generate(生成) 或 analyze(分析)')

    # API 配置参数
    parser.add_argument('--api-url', help='API 地址')
    parser.add_argument('--api-key', help='API Key')
    parser.add_argument('--model', help='模型名称')
    parser.add_argument('--size', default='1024x1024', help='图片尺寸')

    # 提示词参数
    parser.add_argument('--prompt', help='提示词')
    parser.add_argument('--prompt-file', help='提示词文件路径')

    # 输入输出参数
    parser.add_argument('--input', help='输入图片路径 (用于分析或参考图生成)')
    parser.add_argument('--output', help='输出图片路径 (默认: output.png)')
    parser.add_argument('--output-dir', help='输出目录')

    args = parser.parse_args()

    # 从 .env 文件读取配置
    env_path = Path(args.env)
    if not env_path.is_absolute():
        # 相对于脚本目录查找
        script_dir = Path(__file__).parent
        env_path = script_dir / args.env

    env_config = read_env_from_file(str(env_path))

    # 合并配置：命令行参数 > 环境变量 > .env 文件
    api_url = pick_first_non_empty(args.api_url, os.environ.get('API_URL'), env_config.get('API_URL'))
    api_key = pick_first_non_empty(args.api_key, os.environ.get('API_KEY'), env_config.get('API_KEY'))
    model = pick_first_non_empty(args.model, os.environ.get('MODEL'), env_config.get('MODEL'))
    size = pick_first_non_empty(args.size, os.environ.get('SIZE'), env_config.get('SIZE'), '1024x1024')
    output_dir = pick_first_non_empty(args.output_dir, os.environ.get('OUTPUT_DIR'), env_config.get('OUTPUT_DIR'))
    default_prompt = pick_first_non_empty(os.environ.get('PROMPT'), env_config.get('PROMPT'))

    # 获取提示词
    prompt = args.prompt
    if not prompt and args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    if not prompt:
        prompt = default_prompt

    # 检查必要参数
    missing = []
    if not api_url:
        missing.append('API_URL (--api-url)')
    if not api_key:
        missing.append('API_KEY (--api-key)')
    if not model:
        missing.append('MODEL (--model)')

    if args.mode == 'generate' and not prompt:
        missing.append('PROMPT (--prompt 或 --prompt-file)')

    if args.mode == 'analyze' and not args.input:
        missing.append('--input (输入图片路径)')

    if missing:
        print(f"缺少必要配置: {', '.join(missing)}")
        print("\n使用 --help 查看帮助")
        sys.exit(1)

    # 设置输出路径
    output_path = args.output
    if not output_path and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'generated_{timestamp}.png')
    if not output_path:
        output_path = 'output.png'

    print(f"模式: {args.mode}")
    print(f"模型: {model}")
    print(f"API: {api_url}")

    try:
        if args.mode == 'generate':
            print(f"尺寸: {size}")
            print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            print("\n生成图片...")
            image_bytes = generate_image(api_url, api_key, model, prompt, size, args.input)

            # 保存图片
            with open(output_path, 'wb') as f:
                f.write(image_bytes)

            # 显示图片信息
            img = Image.open(io.BytesIO(image_bytes))
            print(f"\n✓ 成功!")
            print(f"  尺寸: {img.width}x{img.height}")
            print(f"  大小: {len(image_bytes) / 1024:.1f} KB")
            print(f"  保存: {output_path}")

        elif args.mode == 'analyze':
            print(f"输入: {args.input}")
            print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            # 分析时使用 Vision API 配置
            vision_api_url = pick_first_non_empty(
                os.environ.get('VISION_API_URL'),
                env_config.get('VISION_API_URL'),
                api_url
            )
            vision_model = pick_first_non_empty(
                os.environ.get('VISION_MODEL'),
                env_config.get('VISION_MODEL'),
                model
            )

            print(f"\n分析图片...")
            result = analyze_image(vision_api_url, api_key, vision_model, args.input, prompt)

            print(f"\n✓ 分析结果:")
            print(result)

            # 可选：保存分析结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"\n保存到: {args.output}")

    except Exception as e:
        print(f"\n✗ 失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
