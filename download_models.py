#!/usr/bin/env python3
"""
Funasr 语音识别模型下载脚本
运行此脚本下载 Paraformer 中文语音识别模型到本地缓存
"""

import os
import sys

def download_funasr_model():
    """从 ModelScope 下载 Funasr Paraformer 模型"""
    try:
        from funasr import AutoModel
    except ImportError:
        print("错误: 需要安装 funasr")
        print("请运行: pip install funasr")
        return False

    print("=" * 50)
    print("Funasr Paraformer 语音识别模型下载")
    print("=" * 50)
    print()
    print("正在从 ModelScope 下载模型（约 944MB）...")
    print("下载位置: ~/.cache/modelscope/hub/")
    print()
    print("这可能需要几分钟时间，请耐心等待...")
    print()

    try:
        # 使用 funasr 的 AutoModel，会自动下载
        model = AutoModel(
            model='paraformer-zh',
            model_revision='v2.0.4',
            disable_update=True,
        )

        print()
        print("=" * 50)
        print("下载成功!")
        print("=" * 50)
        print()
        print("模型信息:")
        print("  - 名称: Paraformer 中文语音识别")
        print("  - 来源: ModelScope")
        print("  - 缓存: ~/.cache/modelscope/hub/")
        print()
        print("环境变量配置:")
        print("  SPEECH_ENGINE=funasr")
        print()
        print("启动服务后，语音识别将自动使用 Funasr Paraformer")
        print()

        return True

    except Exception as e:
        print(f"下载失败: {e}")
        return False


if __name__ == "__main__":
    success = download_funasr_model()
    sys.exit(0 if success else 1)
