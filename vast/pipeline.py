from pathlib import Path
from vast.video_downloader import download_video
from vast.subtitle_generator import generate_subtitle
from vast.keyframe_extractor import extract_keyframes
from vast.scene_analyzer import analyze_scene
from vast.utils import setup_logger

logger = setup_logger()


def run_pipeline(url):
    """
    Run the full video summarization pipeline.

    Steps:
        1. Download the video from the given URL.
        2. Generate subtitles using the Whisper model.
        3. Extract keyframes from the video using FFmpeg.
        4. Analyze each keyframe using the BLIP model for scene descriptions.

    Args:
        url: The video URL to process.
    """
    # Define base data directories
    base_dir = Path("data")
    dirs = {
        "raw": base_dir / "raw_videos",
        "sub": base_dir / "subtitles",
        "kf": base_dir / "keyframes",
        "desc": base_dir / "scene_descriptions"
    }

    # Create directories if they don't exist
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the video
    logger.info(f"Downloading video from: {url}")
    video_path = download_video(url, dirs["raw"])

    # Step 2: Generate subtitles
    logger.info("Generating subtitles using Whisper...")
    generate_subtitle(video_path, dirs["sub"])

    # Step 3: Extract keyframes
    logger.info("Extracting keyframes with FFmpeg...")
    extract_keyframes(video_path, dirs["kf"])

    # Step 4: Analyze each scene
    logger.info("Analyzing scenes with BLIP model...")
    for img in dirs["kf"].glob(f"{video_path.stem}_*.jpg"):
        caption = analyze_scene(img)
        (dirs["desc"] / f"{img.stem}.txt").write_text(caption, encoding="utf-8")

    logger.info("Pipeline completed successfully.")
