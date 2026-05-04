#!/bin/bash
# AI图像工坊一键启动脚本

cd "$(dirname "$0")"

echo "========================================"
echo "  AI图像工坊 - 启动服务"
echo "========================================"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3，请先安装"
    exit 1
fi

# 创建必要目录
mkdir -p ../history/outputs
mkdir -p uploads

echo "✓ 目录准备完成"

# 检查.env文件
if [ ! -f "../.env" ]; then
    echo "⚠ 未找到配置文件 ../.env"
    echo "请复制 config/.env.example 到上级目录并重命名为 .env"
    echo "然后填写API配置"
    echo ""
fi

# 启动服务
echo "✓ 启动 FastAPI 服务..."
echo ""
echo "登录页面: http://localhost:8000/login"
echo "主页面: http://localhost:8000/"
echo "提示词模板: http://localhost:8000/templates"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"

python3 app.py