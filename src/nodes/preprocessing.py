# Tool nodes: pre_router, ocr_processing, document_parsing, process_speech_to_text

from typing import Dict, Any
import os

try:
    from src.ocr.ocr_service import OCRService
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from src.document_parser.document_parser_service import DocumentParserService
    DOCUMENT_PARSER_AVAILABLE = True
except ImportError:
    DOCUMENT_PARSER_AVAILABLE = False


def pre_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 前置路由：判断是否有文件需要上传或视频帧数据
    """
    input_text = state.get("input_text", "")
    file_path = state.get("file_path", "")
    video_frame_data = state.get("video_frame_data", "")

    # 优先处理视频帧（行为分析）
    if video_frame_data:
        return {"pre_route": "video_input"}

    # 文件路由
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".gif"]:
            return {"pre_route": "file_ocr"}
        elif ext in [".docx", ".doc", ".xlsx", ".xls"]:
            return {"pre_route": "file_document"}

    # 默认文本输入
    return {"pre_route": "text_input"}


def ocr_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🩵 Local Node - PaddleOCR 提取图片/PDF 文字
    """
    file_path = state.get("file_path", "")
    if not file_path or not OCR_AVAILABLE:
        return {
            "ocr_result": {"error": "OCR not available"},
            "transcript": ""
        }

    try:
        ocr_service = OCRService()
        result = ocr_service.process_image(file_path)
        return {
            "ocr_result": result,
            "transcript": result.get("text", "")
        }
    except Exception as e:
        return {
            "ocr_result": {"error": str(e)},
            "transcript": ""
        }


def document_parsing(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 解析 Excel/Word 文档
    """
    file_path = state.get("file_path", "")
    if not file_path or not DOCUMENT_PARSER_AVAILABLE:
        return {
            "document_content": "",
            "transcript": ""
        }

    try:
        parser = DocumentParserService()
        result = parser.parse(file_path)
        return {
            "document_content": result.get("content", ""),
            "transcript": result.get("content", "")
        }
    except Exception as e:
        return {
            "document_content": "",
            "transcript": ""
        }


def process_speech_to_text(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    🟢 Tool Node - 统一处理文本输入（不做语音识别，直接透传）
    """
    input_text = state.get("input_text", "")
    return {"transcript": input_text}