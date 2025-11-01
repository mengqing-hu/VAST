"""
scene_analyzer.py
Perform scene analysis using the BLIP model.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch

# Global variables to cache the model and processor
_processor = None
_model = None
_loaded_model_name = None 


def load_model(model_name):
    """
    Load the BLIP image captioning model and processor.

    Args:
        model_name: The name of the pretrained BLIP model from Hugging Face.

    Returns:
        (processor, model)
    """
    global _processor, _model, _loaded_model_name


    if _processor is not None and _model is not None and _loaded_model_name == model_name:
        return _processor, _model

    print(f"Loading model: {model_name} ...")
    _processor = BlipProcessor.from_pretrained(model_name)
    _model = BlipForConditionalGeneration.from_pretrained(model_name)
    _model.to("cuda" if torch.cuda.is_available() else "cpu")
    _loaded_model_name = model_name
    print("Model loaded successfully.")
    return _processor, _model


def analyze_scene(image_path, model_name):
    processor, model = load_model(model_name)
    img = Image.open(image_path).convert("RGB")

    inputs = processor(img, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def analyze_directory(image_dir, output_dir, model_name):
    """
    Analyze all images in a directory and generate captions for each.

    Args:
        image_dir: Directory containing images (.jpg files).
        output_dir: Directory to save generated captions (.txt files).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(model_name)

    images = sorted(image_dir.glob("*.jpg"))
    for img_path in images:
        caption = analyze_scene(img_path, model_name)
        out_path = output_dir / f"{img_path.stem}.txt"
        out_path.write_text(caption, encoding="utf-8")
        print(f"{img_path.name} → {caption}")
