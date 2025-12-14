from pathlib import Path
import yt_dlp
import subprocess
import json


def get_video_codec(video_path):

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

    output_path = input_path.with_name(input_path.stem + "_vscode.mp4")
    print("Converting video to VS Code compatible format (H.264 + AAC)")

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


def extract_wav_audio(video_path, audio_dir):

    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{video_path.stem}.wav"

    if wav_path.exists():
        print(f"WAV already exists: {wav_path}")
        return wav_path

    print(f"Extracting WAV audio: {wav_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    subprocess.run(cmd, check=True)
    print(f"WAV audio saved: {wav_path}")

    return wav_path


def download_video(url, output_dir, audio_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "format": (
            "bestvideo[vcodec*=avc1][ext=mp4]+"
            "bestaudio[acodec*=mp4a]/best[ext=mp4]"
        ),
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

    if codec not in ("h264", "avc1"):
        video_file = convert_to_vscode_compatible(video_file)

    # extract wav audio
    wav_file = extract_wav_audio(video_file, audio_dir)

    print(f"Final video file: {video_file}")
    print(f"Final audio file: {wav_file}")

    return video_file, wav_file
