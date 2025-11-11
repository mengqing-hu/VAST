
import json
import random
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    pipeline
)


_processor = None
_model = None
_loaded_model_name = None



def load_blip_model(model_name):
    """Load BLIP image captioning model (only once)."""
    global _processor, _model, _loaded_model_name

    if _processor is not None and _model is not None and _loaded_model_name == model_name:
        return _processor, _model

    print(f"Loading BLIP model: {model_name} ...")
    _processor = BlipProcessor.from_pretrained(model_name)
    _model = BlipForConditionalGeneration.from_pretrained(model_name)
    _model.to("cuda" if torch.cuda.is_available() else "cpu")
    _loaded_model_name = model_name
    print("BLIP model loaded successfully.")
    return _processor, _model



def generate_scene_description(image_path, model_name):
    """Generate a scene caption using BLIP."""
    processor, model = load_blip_model(model_name)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption



def detect_speaker_position(image_path):
    """
    Simulate speaker position detection.
    In a full system, use pose estimation (MediaPipe/OpenPose) here.
    """

    return random.choice(["Standing", "Sitting", "Unknown"])



def detect_emotion_from_caption(caption):
    """
    Simple emotion estimation based on caption semantics.
    """
    caption_lower = caption.lower()
    if any(word in caption_lower for word in ["smile", "laugh", "happy", "cheerful"]):
        return "Happy"
    elif any(word in caption_lower for word in ["angry", "shouting", "furious"]):
        return "Angry"
    elif any(word in caption_lower for word in ["sad", "cry", "tearful"]):
        return "Sad"
    else:
        return random.choice(["Neutral", "Calm", "Serious"])



_sentiment_classifier = None

def load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    """Load sentiment analysis model (cached)."""
    global _sentiment_classifier
    if _sentiment_classifier is None:
        print(f"Loading sentiment model: {model_name}")
        _sentiment_classifier = pipeline("sentiment-analysis", model=model_name)
    return _sentiment_classifier


def analyze_sentiment(caption, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    """Analyze sentiment from generated caption."""
    classifier = load_sentiment_model(model_name)
    result = classifier(caption[:512])[0]
    return result["label"]



def analyze_directory(image_dir, output_dir, model_name="Salesforce/blip-image-captioning-base"):
    """
    Analyze all .jpg images in a directory.
    For each image, detect scene description, speaker position, emotion, and sentiment.
    Save results to JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(Path(image_dir).glob("*.jpg"))
    if not images:
        print(f"No .jpg files found in {image_dir}")
        return None

    results = []

    for img_path in images:
        print(f"Analyzing: {img_path.name}")
        scene_desc = generate_scene_description(img_path, model_name)
        position = detect_speaker_position(img_path)
        emotion = detect_emotion_from_caption(scene_desc)
        sentiment = analyze_sentiment(scene_desc)

        result = {
            "image": str(img_path),
            "speaker_position": position,
            "emotion": emotion,
            "sentiment": sentiment,
            "scene_description": scene_desc
        }
        results.append(result)

    # Save results to JSON
    output_json = Path(output_dir) / "scene_analysis.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f" Scene analysis JSON saved to {output_json}")
    return output_json
