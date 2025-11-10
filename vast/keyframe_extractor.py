import ffmpeg
import json
from pathlib import Path


def extract_keyframes(video_path, output_dir, interval=2.0):

    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output image filename pattern
    output_pattern = str(output_dir / f"{video_path.stem}_%04d.jpg")

    # Extract frames using FFmpeg
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=1 / interval)
        .output(output_pattern, qscale=2, start_number=0)
        .overwrite_output()
        .run(quiet=True)
    )

    # Collect generated keyframes
    keyframes = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
    print(f"Extraction completed: {len(keyframes)} keyframes generated.")

    # Build keyframe metadata list
    keyframe_metadata = []
    for i, frame in enumerate(keyframes):
        start_time = i * interval
        end_time = (i + 1) * interval
        keyframe_metadata.append({
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "frame": str(frame)
        })

    # Save keyframe metadata to JSON
    json_path = output_dir / f"{video_path.stem}_keyframes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(keyframe_metadata, f, indent=2, ensure_ascii=False)

    print(f"Keyframe metadata saved to {json_path}")
    return keyframes
