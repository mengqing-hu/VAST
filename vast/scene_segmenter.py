import cv2
import json
import ffmpeg
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from sentence_transformers import SentenceTransformer, util


def load_model(method="ssim", model_name="clip-ViT-B-32"):
    """Load CLIP model if needed."""
    if method == "clip":
        print(f"Loading CLIP model: {model_name}")
        return SentenceTransformer(model_name)
    return None


def compute_similarity(img1, img2, method="ssim", model=None):
    """Compute similarity between two frames."""
    if method == "ssim":
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score = ssim(gray1, gray2)
        return score

    elif method == "clip":
        if model is None:
            raise ValueError("CLIP model required for 'clip' method.")
        emb1 = model.encode(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), convert_to_tensor=True)
        emb2 = model.encode(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return score

    else:
        raise ValueError(f"Unsupported method: {method}")


def detect_scenes(keyframes, interval=60.0, method="ssim", threshold=0.6, model_name="clip-ViT-B-32"):
    """Detect scene boundaries based on keyframe similarity."""
    model = load_model(method, model_name)
    scenes = []
    start_time = 0.0
    prev_img = cv2.imread(str(keyframes[0]))
    total_frames = len(keyframes)

    for i in tqdm(range(1, total_frames), desc="Detecting scene boundaries"):
        curr_img = cv2.imread(str(keyframes[i]))
        sim = compute_similarity(prev_img, curr_img, method, model)
        diff = 1 - sim

        if diff > threshold:
            end_time = i * interval
            scenes.append((start_time, end_time))
            start_time = end_time
        prev_img = curr_img

    # Add the final segment
    scenes.append((start_time, total_frames * interval))
    print(f"Detected {len(scenes)} scenes.")
    return scenes


def export_scenes(video_path, scenes, output_dir):
    """
    Export segmented video clips using FFmpeg
    and automatically save scene_segments.json in the same folder.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path.resolve()}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metadata container
    scene_metadata = []

    for i, (start, end) in enumerate(tqdm(scenes, desc="Exporting scenes")):
        output_file = output_dir / f"scene_{i:03d}.mp4"
        duration = end - start

        # Export each scene clip
        (
            ffmpeg
            .input(str(video_path), ss=start, t=duration)
            .output(str(output_file), c='copy', loglevel="error")
            .overwrite_output()
            .run()
        )

        # Add scene metadata
        scene_metadata.append({
            "scene_id": i,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(duration, 2),
            "video_path": str(video_path),
            "output_file": str(output_file)
        })

    print(f"Export complete! {len(scene_metadata)} scenes saved to: {output_dir}")

    # Save scene_segments.json inside the same folder
    json_path = output_dir / "scene_segments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene_metadata, f, indent=2, ensure_ascii=False)
    print(f"Scene metadata saved to {json_path}")

    return scene_metadata
