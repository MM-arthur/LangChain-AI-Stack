import os
import fitz  # PyMuPDF
import numpy as np
import cv2  # 用于图像处理，如颜色空间转换

from paddleocr import PPStructureV3

def render_pdf_pages_to_numpy(pdf_path, scale=1.5):
    """
    将PDF页面渲染为NumPy数组列表（BGR格式）。
    这避免了将中间图像保存到磁盘。
    
    参数:
        pdf_path (str): 输入PDF文件的路径。
        scale (float): 渲染分辨率的缩放因子。例如, scale=3 通常提供约300 DPI的效果。
                       更高的值会生成更高质量的图像, 但会消耗更多内存和处理时间。
    
    返回:
        list: 包含每个页面图像的NumPy数组列表。
    """
    pdf_document = fitz.open(pdf_path)
    page_images_np = []
    
    print(f"开始渲染PDF '{os.path.basename(pdf_path)}' 的页面到内存...")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # 渲染页面到 pixmap
        # 使用更高的缩放因子以获得更好的OCR效果
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # 将 pixmap 转换为 NumPy 数组
        # pix.samples 是一个字节对象，将其重塑为 (height, width, channels)
        # fitz.Pixmap 通常是 RGB 格式，而 PaddleOCR/PPStructure 期望 BGR 格式
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        
        # 如果是 RGBA 格式 (4 通道)，先转换为 RGB (3 通道)
        if img_np.shape[2] == 4: # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # 将 RGB 转换为 BGR，以符合 PaddleOCR/PPStructure 的输入要求
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        page_images_np.append(img_np)
        print(f'  已将PDF第 {page_num+1} 页渲染到内存。')
        
        pix = None # 释放 pixmap 内存
    
    pdf_document.close()
    return page_images_np

# 初始化 PPStructureV3 引擎
# 按照示例进行初始化
structure_engine = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# 设置输入PDF和输出文件夹路径
pdf_path = "../path"
output_folder = "../path"

# 确保主输出结果的目录存在
base_result_folder = os.path.join(output_folder, "ppstructure_results")
if not os.path.exists(base_result_folder):
    os.makedirs(base_result_folder)
    print(f"创建主输出目录: {base_result_folder}")

# 渲染PDF页面到内存中的NumPy数组
page_numpy_arrays = render_pdf_pages_to_numpy(pdf_path, scale=1.5) # 使用较高的缩放因子以获得更好的质量

# 使用PPStructureV3处理每个页面
print("\n开始使用PP-StructureV3处理每个页面...")
for i, img_np in enumerate(page_numpy_arrays):
    page_num = i + 1
    print(f"\n正在处理第 {page_num} 页...")
    
    # 调用 PPStructureV3 的 predict 方法，传入图像 NumPy 数组
    # predict 方法返回一个列表的 StructureResult 对象
    page_results = structure_engine.predict(input=img_np)

    # 为当前页面创建独立的输出目录，所有该页面的结构化结果将保存到此目录
    current_page_output_dir = os.path.join(base_result_folder, f"page_{page_num:03d}")
    if not os.path.exists(current_page_output_dir):
        os.makedirs(current_page_output_dir)
        print(f"  创建页面 {page_num} 的输出目录: {current_page_output_dir}")
    
    # 遍历当前页面的所有结构化结果对象并保存
    for res in page_results:
        res.print() # 打印结果到控制台，与官方示例一致
        
        # 保存为 JSON 文件，save_path 是目录
        res.save_to_json(save_path=current_page_output_dir)
        # 保存为 Markdown 文件，save_path 是目录
        res.save_to_markdown(save_path=current_page_output_dir)

print("\n所有处理完成！")