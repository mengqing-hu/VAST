import ffmpeg
from pathlib import Path


def extract_keyframes(video_path, output_dir, interval=2.0):
    """
    Extract keyframes from a video at the specified time interval (in seconds).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted keyframes will be saved.
        interval: Time interval (in seconds) between keyframes.

    Returns:
        A list of file paths for the extracted keyframe images.
    """
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output image filename pattern
    output_pattern = str(output_dir / f"{video_path.stem}_%04d.jpg")

    # Use FFmpeg to extract frames at the specified interval
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=1 / interval)
        .output(output_pattern, qscale=2, start_number=0)
        .overwrite_output()
        .run(quiet=True)
    )

    # Collect all generated keyframes
    keyframes = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
    print(f"Extraction completed: {len(keyframes)} keyframes generated.")
    return keyframes



