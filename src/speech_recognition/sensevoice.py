# Funasr 语音识别服务 - 使用 ModelScope Paraformer 模型
# 支持中文语音识别，模型下载自 ModelScope

import os
import logging
import tempfile
import traceback
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FunasrService:
    def __init__(self, model_name: str = "paraformer-zh"):
        """
        初始化 Funasr 语音识别服务
        :param model_name: 模型名称，默认使用 paraformer-zh（支持中文）
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 Funasr 模型"""
        try:
            from funasr import AutoModel

            logger.info(f"正在加载 Funasr 模型: {self.model_name}")

            # 使用 funasr 的 AutoModel，会自动从 ModelScope 下载
            self.model = AutoModel(
                model=self.model_name,
                model_revision="v2.0.4",
                disable_update=True,  # 禁用自动检查更新
            )

            logger.info("Funasr 模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Funasr 模型加载失败: {str(e)}")

    async def convert_audio_to_text(self, audio_file) -> str:
        """
        将音频文件转换为文字
        :param audio_file: 音频文件对象
        :return: 转换后的文字
        """
        temp_files = []
        try:
            logger.info("开始 Funasr 语音转文字处理")

            # 获取文件扩展名
            content_type = getattr(audio_file, 'content_type', 'audio/wav')
            logger.info(f"接收到的音频内容类型: {content_type}")

            # 保存上传的音频文件到临时文件
            suffix = '.wav'
            if 'mp3' in content_type.lower():
                suffix = '.mp3'
            elif 'm4a' in content_type.lower():
                suffix = '.m4a'
            elif 'ogg' in content_type.lower():
                suffix = '.ogg'

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name
                await audio_file.seek(0)
                file_content = await audio_file.read()

                if len(file_content) == 0:
                    raise Exception("读取到的音频文件内容为空")

                logger.info(f"读取到的文件内容大小: {len(file_content)} bytes")
                temp_file.write(file_content)
            temp_files.append(temp_file_path)

            # 调用 Funasr 进行识别
            result = self._recognize(temp_file_path)

            text = result.strip()

            if not text:
                logger.warning("Funasr 无法识别音频内容")
                return "无法识别语音"

            logger.info(f"语音转文字成功: {text}")
            return text

        except Exception as e:
            logger.error(f"语音转换失败: {str(e)}")
            logger.error(traceback.format_exc())
            return f"语音转换失败: {str(e)}"
        finally:
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

    def _recognize(self, audio_path: str) -> str:
        """
        使用 Funasr 识别音频
        :param audio_path: 音频文件路径
        :return: 识别文本
        """
        try:
            if self.model is None:
                self._load_model()

            logger.info(f"开始识别音频: {audio_path}")

            # Funasr 的 generate 方法返回列表
            result = self.model.generate(
                input=audio_path,
                batch_size_s=300,
                merge_vad=True,
                merge_length_s=15,
            )

            if not result:
                return ""

            # 提取文本
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, tuple):
                    return first_result[0]
                elif isinstance(first_result, str):
                    return first_result
                elif isinstance(first_result, dict):
                    return first_result.get('text', '')
                elif isinstance(first_result, list):
                    # 可能是 [(text, ...), ...] 格式
                    return str(first_result[0]) if first_result else ""

            return str(result)

        except Exception as e:
            logger.error(f"Funasr 识别失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise Exception(f"Funasr 识别失败: {str(e)}")

    def convert_audio_path_to_text(self, audio_path: str) -> str:
        """
        从文件路径转换音频为文字
        :param audio_path: 音频文件路径
        :return: 转换后的文字
        """
        try:
            if not os.path.exists(audio_path):
                raise Exception(f"音频文件不存在: {audio_path}")

            logger.info(f"开始处理音频文件: {audio_path}")
            return self._recognize(audio_path)

        except Exception as e:
            logger.error(f"语音转换失败: {str(e)}", exc_info=True)
            return f"语音转换失败: {str(e)}"


# 为了兼容旧代码，保留 WhisperService 别名
WhisperService = FunasrService


if __name__ == "__main__":
    print("Funasr 语音识别服务")
    print(f"模型名称: paraformer-zh")
    service = FunasrService()
    print("服务初始化完成")
