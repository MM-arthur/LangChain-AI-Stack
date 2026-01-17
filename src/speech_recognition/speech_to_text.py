# 语音转文字服务

import os
import logging
import tempfile
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextService:
    def __init__(self, model_path=None):
        """
        初始化语音转文字服务
        :param model_path: 模型路径，PaddleSpeech不需要本地模型文件，此参数仅为保持兼容性
        """
        self.asr = None
        self.model_path = model_path
        self.asr_executor = None
        
    def _init_asr(self):
        """
        延迟初始化PaddleSpeech ASR执行器
        """
        if self.asr_executor is None:
            try:
                # 动态导入，避免启动时失败
                from paddlespeech.cli.asr.infer import ASRExecutor
                
                # 初始化PaddleSpeech ASR执行器
                logger.info("正在加载PaddleSpeech ASR模型")
                self.asr_executor = ASRExecutor()
                logger.info("PaddleSpeech ASR模型加载成功")
            except Exception as e:
                logger.error(f"PaddleSpeech模型初始化失败: {str(e)}")
                logger.error(traceback.format_exc())
                # 不抛出异常，而是返回友好错误信息
                raise Exception(f"PaddleSpeech模型初始化失败: {str(e)}")
    
    async def convert_audio_to_text(self, audio_file):
        """
        将音频文件转换为文字
        :param audio_file: 音频文件对象
        :return: 转换后的文字
        """
        temp_files = []
        try:
            logger.info("开始语音转文字处理")
            
            # 获取文件扩展名和实际内容类型
            content_type = getattr(audio_file, 'content_type', 'audio/wav')
            logger.info(f"接收到的音频内容类型: {content_type}")
            
            # 保存上传的音频文件到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                await audio_file.seek(0)  # 确保从头开始读取
                file_content = await audio_file.read()
                
                # 检查读取到的文件内容是否为空
                if len(file_content) == 0:
                    raise Exception("读取到的音频文件内容为空")
                
                logger.info(f"读取到的文件内容大小: {len(file_content)} bytes")
                temp_file.write(file_content)
            temp_files.append(temp_file_path)
            
            # 验证文件是否成功保存
            if not os.path.exists(temp_file_path):
                raise Exception(f"无法保存音频文件到临时路径: {temp_file_path}")
            
            saved_size = os.path.getsize(temp_file_path)
            if saved_size == 0:
                raise Exception(f"保存的音频文件大小为0字节: {temp_file_path}")
            
            logger.info(f"音频文件已保存到临时路径: {temp_file_path}")
            logger.info(f"保存的音频文件大小: {saved_size} bytes")
            
            # 使用PaddleSpeech处理音频文件
            logger.info("使用PaddleSpeech处理音频文件")
            
            # 延迟初始化ASR执行器
            self._init_asr()
            
            # 调用PaddleSpeech ASR进行识别
            logger.info(f"调用PaddleSpeech ASR识别音频文件: {temp_file_path}")
            result = self.asr_executor(audio_file=temp_file_path)
            
            text = result.strip()
            
            if not text:
                logger.warning("PaddleSpeech无法识别音频内容")
                return "无法识别语音"
            
            logger.info(f"语音转文字成功: {text}")
            return text
            
        except Exception as e:
            logger.error(f"语音转换失败: {str(e)}")
            logger.error(traceback.format_exc())
            return f"语音转换失败: {str(e)}"
        finally:
            # 清理所有临时文件
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    try:
                        logger.info(f"正在清理临时文件: {temp_path}")
                        os.unlink(temp_path)
                        logger.info(f"临时文件已清理: {temp_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {temp_path}, 错误: {str(e)}")
    
    def convert_audio_path_to_text(self, audio_path):
        """
        从文件路径转换音频为文字
        :param audio_path: 音频文件路径
        :return: 转换后的文字
        """
        try:
            logger.info(f"开始处理音频文件: {audio_path}")
            
            # 延迟初始化ASR执行器
            self._init_asr()
            # 调用PaddleSpeech ASR进行识别
            result = self.asr_executor(audio_file=audio_path)
            
            text = result.strip()
            
            if not text:
                logger.warning("PaddleSpeech无法识别音频内容")
                return "无法识别语音"
            
            logger.info(f"语音转文字成功: {text}")
            return text
            
        except Exception as e:
            logger.error(f"语音转换失败: {str(e)}", exc_info=True)
            return f"语音转换失败: {str(e)}"

# 测试代码
if __name__ == "__main__":
    stt_service = SpeechToTextService()
    print("语音转文字服务已初始化")
