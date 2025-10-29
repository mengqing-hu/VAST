"""
scene_analyzer.py
基于图像字幕生成模型（BLIP）对关键帧进行场景分析。
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch

# 全局缓存模型（避免重复加载）
_processor = None
_model = None

def load_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    """加载 BLIP 模型和处理器"""
    global _processor, _model
    if _processor is None or _model is None:
        print(f"正在加载模型: {model_name} ...")
        _processor = BlipProcessor.from_pretrained(model_name)
        _model = BlipForConditionalGeneration.from_pretrained(model_name)
        _model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("模型加载完成")
    return _processor, _model


def analyze_scene(image_path: Path) -> str:
    """
    对单张关键帧图像生成场景描述。

    参数:
        image_path (Path): 图像路径。
    返回:
        str: 图像场景描述文本。
    """
    processor, model = load_model()
    img = Image.open(image_path).convert("RGB")

    inputs = processor(img, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def analyze_directory(image_dir: Path, output_dir: Path):
    """
    对整个关键帧目录进行批量场景分析，结果保存为 .txt 文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model()

    images = sorted(image_dir.glob("*.jpg"))
    for img_path in images:
        caption = analyze_scene(img_path)
        out_path = output_dir / f"{img_path.stem}.txt"
        out_path.write_text(caption, encoding="utf-8")
        print(f"{img_path.name} → {caption}")


# 模块自测入口
if __name__ == "__main__":
    img_dir = Path("data/keyframes")
    out_dir = Path("data/scene_descriptions")
    analyze_directory(img_dir, out_dir)
