# 面试官行为分析服务 - YOLO + 姿态检测
# 支持：人体检测、姿态估计、视线分析、表情识别
# 帮助你（面试者）分析面试官的行为、表情、注意力

import os
import logging
import tempfile
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 本地模型路径
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "BehaviorAnalysis")

@dataclass
class DetectionResult:
    """面试官检测结果"""
    has_person: bool = False
    face_detected: bool = False
    gaze_direction: str = "unknown"  # left, right, center, down, up
    gaze_score: float = 0.0
    posture: str = "unknown"  # sitting, standing, leaning_forward, leaning_back
    posture_score: float = 0.0
    expression: str = "neutral"  # happy, sad, angry, surprised, fearful, disgusted, neutral
    expression_score: float = 0.0
    attention_level: str = "normal"  # focused, distracted, away, thinking
    attention_score: float = 0.0
    warnings: List[str] = None
    raw_output: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class BehaviorAnalyzer:
    def __init__(self, use_ultra_light: bool = True):
        """
        初始化行为分析器
        :param use_ultra_light: 是否使用超轻量模式（优先速度）
        """
        self.use_ultra_light = use_ultra_light
        self.yolo_model = None
        self._load_models()

    def _load_models(self):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            logger.info("正在加载 YOLOv8n 模型...")

            # YOLOv8n 会自动从官方下载（约 6MB）
            # ultralytics 在国内可通过阿里云镜像加速
            self.yolo_model = YOLO("yolov8n.pt")

            logger.info("YOLOv8n 模型加载成功")

        except ImportError:
            logger.warning("ultralytics 未安装，请运行: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            logger.warning(f"YOLOv8n 模型加载失败: {e}，将使用简化模式")
            self.yolo_model = None

    def analyze_frame(self, frame) -> DetectionResult:
        """
        分析单帧图像
        :param frame: numpy array (BGR) 或 图像路径
        :return: DetectionResult
        """
        result = DetectionResult()

        try:
            import cv2

            if isinstance(frame, str):
                # 图像路径
                frame = cv2.imread(frame)

            if frame is None:
                result.warnings.append("无法读取图像")
                return result

            h, w = frame.shape[:2]

            # 使用 YOLO 进行人体检测
            if self.yolo_model is not None:
                yolo_results = self.yolo_model(frame, verbose=False)

                # 检测人 (COCO class 0 = person)
                person_boxes = []
                for r in yolo_results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # person
                            person_boxes.append(box)

                result.has_person = len(person_boxes) > 0

                if person_boxes:
                    # 取最大的人体框
                    largest_box = max(person_boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
                    x, y, bw, bh = largest_box.xywh[0]
                    conf = float(largest_box.conf[0])

                    # 判断姿态（基于人体框比例）
                    aspect_ratio = bh / bw if bw > 0 else 1
                    if aspect_ratio > 1.5:
                        result.posture = "standing"
                    else:
                        result.posture = "sitting"
                    result.posture_score = conf

                    # 简化视线分析（基于面部在上半区域）
                    if y / h < 0.4:
                        result.gaze_direction = "up"  # 可能看屏幕
                        result.gaze_score = 0.6
                    elif y / h > 0.6:
                        result.gaze_direction = "down"  # 可能看手机/笔记
                        result.gaze_score = 0.5
                    else:
                        result.gaze_direction = "center"  # 看向镜头
                        result.gaze_score = 0.7

            # 简化表情分析（基于图像亮度特征，实际应该用专门的表情模型）
            # 这里返回 neutral 作为占位
            result.expression = "neutral"
            result.expression_score = 0.5

            # 综合注意力分析
            result.attention_level = self._calculate_attention(result)
            result.attention_score = 0.8 if result.attention_level == "focused" else 0.5

            result.raw_output = {
                "frame_shape": frame.shape,
                "person_count": len(person_boxes) if self.yolo_model and 'person_boxes' in dir() else 0
            }

            logger.info(f"分析完成: 姿势={result.posture}, 视线={result.gaze_direction}, 注意力={result.attention_level}")

        except Exception as e:
            logger.error(f"帧分析失败: {e}")
            result.warnings.append(f"分析错误: {str(e)}")

        return result

    def _calculate_attention(self, result: DetectionResult) -> str:
        """计算注意力水平"""
        score = 0.5

        if result.posture == "sitting":
            score += 0.15

        if result.gaze_direction == "center":
            score += 0.25
        elif result.gaze_direction == "up":
            score += 0.15

        if result.expression == "neutral":
            score += 0.1

        if score >= 0.7:
            return "focused"
        elif score >= 0.4:
            return "normal"
        else:
            return "distracted"

    def analyze_video_frame(self, frame_data: str) -> Dict[str, Any]:
        """
        分析 base64 编码的帧
        :param frame_data: base64 编码的图像数据
        :return: 检测结果字典
        """
        try:
            import cv2
            import numpy as np

            # 解码 base64
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"error": "无法解码图像"}

            # 分析
            result = self.analyze_frame(frame)

            return {
                "success": True,
                "has_person": result.has_person,
                "face_detected": result.face_detected,
                "gaze": {
                    "direction": result.gaze_direction,
                    "confidence": result.gaze_score
                },
                "posture": {
                    "state": result.posture,
                    "confidence": result.posture_score
                },
                "expression": {
                    "state": result.expression,
                    "confidence": result.expression_score
                },
                "attention": {
                    "level": result.attention_level,
                    "score": result.attention_score
                },
                "warnings": result.warnings
            }

        except Exception as e:
            logger.error(f"视频帧分析失败: {e}")
            return {"success": False, "error": str(e)}

    def analyze_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        分析图像文件
        :param image_path: 图像文件路径
        :return: 检测结果字典
        """
        try:
            result = self.analyze_frame(image_path)

            return {
                "success": True,
                "has_person": result.has_person,
                "face_detected": result.face_detected,
                "gaze": {
                    "direction": result.gaze_direction,
                    "confidence": result.gaze_score
                },
                "posture": {
                    "state": result.posture,
                    "confidence": result.posture_score
                },
                "expression": {
                    "state": result.expression,
                    "confidence": result.expression_score
                },
                "attention": {
                    "level": result.attention_level,
                    "score": result.attention_score
                },
                "warnings": result.warnings
            }

        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return {"success": False, "error": str(e)}


def download_yolo_models():
    """预下载 YOLO 模型（可选）"""
    try:
        from ultralytics import YOLO
        logger.info("正在下载 YOLOv8n 模型...")
        model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n 模型下载完成")
        return True
    except Exception as e:
        logger.error(f"YOLOv8n 模型下载失败: {e}")
        return False


if __name__ == "__main__":
    print("面试官行为分析服务")
    print("YOLOv8n 会在首次使用时自动下载")

    # 测试
    print("\n下载/加载模型...")
    download_yolo_models()

    analyzer = BehaviorAnalyzer()
    print("\n服务初始化完成，等待分析请求...")
