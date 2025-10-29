from pathlib import Path
import yt_dlp
import subprocess
import json


def _get_video_codec(video_path: Path) -> str:
    """检测视频编码格式"""
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


def _convert_to_vscode_compatible(input_path: Path) -> Path:
    """将视频转码为 VS Code 可播放格式 (H.264 + AAC)"""
    output_path = input_path.with_name(input_path.stem + "_vscode.mp4")
    print(f"正在转码为 VS Code 兼容格式 (H.264 + AAC)...")
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
    print(f"转码完成：{output_path}")
    return output_path


def download_video(url: str, output_dir: Path) -> Path:
    """
    下载视频并确保在 VS Code 可播放（H.264 + AAC）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        # 优先选择 mp4 + H.264 + AAC，避免 AV1 / VP9
        "format": "bestvideo[vcodec*=avc1][ext=mp4]+bestaudio[acodec*=mp4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }

    print(f"正在下载视频：{url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_file = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    codec = _get_video_codec(video_file)
    print(f"当前视频编码：{codec}")

    # 如果检测到不是 H.264，则转码
    if codec not in ("h264", "avc1"):
        video_file = _convert_to_vscode_compatible(video_file)

    print(f"最终文件：{video_file}")
    return video_file
