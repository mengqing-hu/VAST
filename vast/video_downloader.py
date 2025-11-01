from pathlib import Path
import yt_dlp
import subprocess
import json


def get_video_codec(video_path):
    """
    Detect the video codec format using ffprobe.
    Returns the codec name as a string, or "unknown" if detection fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "json",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return data["streams"][0]["codec_name"]
    except Exception:
        return "unknown"


def convert_to_vscode_compatible(input_path):
    """
    Convert a video to a VS Code compatible format (H.264 + AAC).
    Returns the path of the converted file.
    """
    output_path = input_path.with_name(input_path.stem + "_vscode.mp4")
    print("Converting video to VS Code compatible format (H.264 + AAC)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Conversion complete: {output_path}")
    return output_path


def download_video(url, output_dir):
    """
    Download a video and ensure it is playable in VS Code (H.264 + AAC).
    If the downloaded video uses a non-compatible codec, it will be re-encoded.
    """
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt_dlp download options
    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        # Prefer mp4 + H.264 + AAC to avoid AV1 / VP9 codecs
        "format": "bestvideo[vcodec*=avc1][ext=mp4]+bestaudio[acodec*=mp4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }

    print(f"Downloading video: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_file = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    codec = get_video_codec(video_file)
    print(f"Detected video codec: {codec}")

    # Re-encode if the codec is not H.264
    if codec not in ("h264", "avc1"):
        video_file = convert_to_vscode_compatible(video_file)

    print(f"Final file: {video_file}")
    return video_file
