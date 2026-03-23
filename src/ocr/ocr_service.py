import os
import logging
import tempfile
import numpy as np
import cv2
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF未安装，PDF处理功能将受限。请安装: pip install pymupdf")

try:
    from paddleocr import PaddleOCR, PPStructure
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR未安装，OCR功能将不可用。请安装: pip install paddleocr paddlepaddle")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, use_gpu: bool = False, lang: str = 'ch'):
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr_engine = None
        self.structure_engine = None
        
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR未安装，无法初始化OCR服务。请安装: pip install paddleocr paddlepaddle")
    
    def _init_ocr_engine(self):
        if self.ocr_engine is None:
            try:
                logger.info("正在初始化PaddleOCR引擎...")
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
                logger.info("PaddleOCR引擎初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {str(e)}")
                raise Exception(f"OCR引擎初始化失败: {str(e)}")
    
    def _init_structure_engine(self):
        if self.structure_engine is None:
            try:
                logger.info("正在初始化PPStructure引擎...")
                self.structure_engine = PPStructure(
                    use_gpu=self.use_gpu,
                    show_log=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False
                )
                logger.info("PPStructure引擎初始化成功")
            except Exception as e:
                logger.error(f"PPStructure初始化失败: {str(e)}")
                raise Exception(f"结构化引擎初始化失败: {str(e)}")
    
    def _render_pdf_page_to_image(self, page, scale: float = 1.5) -> np.ndarray:
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF未安装，无法渲染PDF页面。请安装: pip install pymupdf")
        
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            (pix.height, pix.width, pix.n)
        )
        
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        pix = None
        return img_np
    
    def _check_pdf_has_text(self, pdf_path: Union[str, Path]) -> bool:
        if not PYMUPDF_AVAILABLE:
            return False
        
        try:
            doc = fitz.open(pdf_path)
            total_text = ""
            
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                total_text += page.get_text()
            
            doc.close()
            
            has_text = len(total_text.strip()) > 100
            logger.info(f"PDF文本检测: {'有文字层' if has_text else '扫描版/图片版'}")
            return has_text
            
        except Exception as e:
            logger.warning(f"PDF文本检测失败: {str(e)}")
            return False
    
    def extract_text_from_image(
        self, 
        image_input: Union[str, np.ndarray, Path],
        enable_structure: bool = False
    ) -> Dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"无法读取图片: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("不支持的输入类型，请提供文件路径或numpy数组")
        
        self._init_ocr_engine()
        
        try:
            result = self.ocr_engine.ocr(image, cls=True)
            
            text_lines = []
            all_text = []
            
            if result and result[0]:
                for line in result[0]:
                    box = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    text_lines.append({
                        "text": text,
                        "confidence": float(confidence),
                        "box": box
                    })
                    all_text.append(text)
            
            full_text = "\n".join(all_text)
            
            response = {
                "success": True,
                "text": full_text,
                "text_lines": text_lines,
                "total_lines": len(text_lines)
            }
            
            if enable_structure:
                structure_result = self._extract_structure(image)
                response["structure"] = structure_result
            
            return response
            
        except Exception as e:
            logger.error(f"图片OCR失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "text_lines": []
            }
    
    def _extract_structure(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            self._init_structure_engine()
            result = self.structure_engine(image)
            
            tables = []
            texts = []
            
            for region in result:
                region_type = region.get('type', '')
                
                if region_type == 'table':
                    table_data = region.get('res', {})
                    tables.append({
                        "type": "table",
                        "data": table_data
                    })
                else:
                    text_content = region.get('res', [])
                    if text_content:
                        texts.append({
                            "type": region_type,
                            "content": text_content
                        })
            
            return {
                "tables": tables,
                "texts": texts,
                "has_tables": len(tables) > 0
            }
            
        except Exception as e:
            logger.error(f"结构化提取失败: {str(e)}")
            return {
                "tables": [],
                "texts": [],
                "error": str(e)
            }
    
    def extract_text_from_pdf_smart(
        self, 
        pdf_path: Union[str, Path],
        scale: float = 1.5,
        enable_structure: bool = False,
        force_ocr: bool = False
    ) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            raise ImportError("请安装PyMuPDF: pip install pymupdf")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        logger.info(f"开始智能处理PDF: {pdf_path}")
        
        use_ocr = force_ocr or not self._check_pdf_has_text(pdf_path)
        
        if use_ocr:
            logger.info("使用OCR方式处理PDF（扫描版或强制OCR）")
            return self._extract_text_from_pdf_via_ocr(pdf_path, scale, enable_structure)
        else:
            logger.info("使用文本提取方式处理PDF（文字版）")
            return self._extract_text_from_pdf_direct(pdf_path, enable_structure)
    
    def _extract_text_from_pdf_direct(
        self, 
        pdf_path: Union[str, Path],
        enable_structure: bool = False
    ) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "total_pages": len(doc)
            }
            
            pages_content = []
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                page_data = {
                    "page_number": page_num + 1,
                    "text": text,
                    "method": "direct_extraction"
                }
                
                if enable_structure:
                    blocks = page.get_text("dict")["blocks"]
                    structured_blocks = []
                    
                    for block in blocks:
                        if "lines" in block:
                            block_text = "\n".join([
                                " ".join([span["text"] for span in line["spans"]])
                                for line in block["lines"]
                            ])
                            structured_blocks.append({
                                "type": "text",
                                "bbox": block["bbox"],
                                "text": block_text
                            })
                    
                    page_data["blocks"] = structured_blocks
                
                pages_content.append(page_data)
                all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
            
            doc.close()
            
            return {
                "success": True,
                "text": "\n\n".join(all_text),
                "pages": pages_content,
                "total_pages": len(pages_content),
                "method": "direct_extraction",
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"PDF直接提取失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "pages": []
            }
    
    def _extract_text_from_pdf_via_ocr(
        self, 
        pdf_path: Union[str, Path],
        scale: float = 1.5,
        enable_structure: bool = False
    ) -> Dict[str, Any]:
        try:
            doc = fitz.open(pdf_path)
            all_pages_result = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                logger.info(f"正在OCR处理第 {page_num + 1} 页...")
                
                img_np = self._render_pdf_page_to_image(page, scale)
                
                page_result = self.extract_text_from_image(
                    img_np, 
                    enable_structure=enable_structure
                )
                page_result["page_number"] = page_num + 1
                page_result["method"] = "ocr"
                all_pages_result.append(page_result)
            
            doc.close()
            
            full_text = "\n\n".join([
                f"--- 第 {page['page_number']} 页 ---\n{page['text']}"
                for page in all_pages_result
            ])
            
            return {
                "success": True,
                "text": full_text,
                "pages": all_pages_result,
                "total_pages": len(all_pages_result),
                "method": "ocr"
            }
            
        except Exception as e:
            logger.error(f"PDF OCR处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "pages": []
            }
    
    def extract_text_from_pdf(
        self, 
        pdf_path: Union[str, Path],
        scale: float = 1.5,
        enable_structure: bool = False
    ) -> Dict[str, Any]:
        return self.extract_text_from_pdf_smart(
            pdf_path, 
            scale=scale, 
            enable_structure=enable_structure,
            force_ocr=False
        )
    
    def process_file(
        self, 
        file_path: Union[str, Path],
        enable_structure: bool = False
    ) -> Dict[str, Any]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf_smart(
                file_path, 
                enable_structure=enable_structure
            )
        elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
            return self.extract_text_from_image(
                file_path, 
                enable_structure=enable_structure
            )
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

if __name__ == "__main__":
    ocr_service = OCRService()
    
    test_image = "test.png"
    if os.path.exists(test_image):
        result = ocr_service.extract_text_from_image(test_image)
        print(f"识别结果: {result['text']}")
    
    test_pdf = "test.pdf"
    if os.path.exists(test_pdf):
        result = ocr_service.extract_text_from_pdf_smart(test_pdf)
        print(f"PDF识别结果 (方法: {result.get('method', 'unknown')}): {result['text'][:200]}...")
