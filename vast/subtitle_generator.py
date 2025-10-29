"""
subtitle_generator.py
使用 OpenAI Whisper 将视频音频转录为字幕 (.srt)。
"""

import whisper
from pathlib import Path

def generate_subtitle(video_path: Path, output_dir: Path, model_size: str = "small") -> Path:
    """
    使用 OpenAI Whisper 模型为视频生成字幕文件（.srt）。

    参数:
        video_path (Path): 视频文件路径。
        output_dir (Path): 输出字幕目录。
        model_size (str): Whisper 模型尺寸，可选 tiny/small/medium/large。

    返回:
        Path: 生成的字幕文件路径。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_size)

    print(f"正在转录音频（模型：{model_size}）...")
    result = model.transcribe(str(video_path))

    srt_path = output_dir / f"{video_path.stem}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        write_srt(result["segments"], f)

    print(f"字幕生成完成: {srt_path}")
    return srt_path


def write_srt(segments, file_obj):
    """将 Whisper 的转录结果写入 .srt 文件"""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip().replace("-->", "->")

        file_obj.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_timestamp(seconds: float) -> str:
    """将时间格式化为 00:00:00,000"""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


# 模块自测入口
if __name__ == "__main__":
    test_video = Path("data/raw_videos/test_video.mp4")
    out_dir = Path("data/subtitles")
    generate_subtitle(test_video, out_dir, model_size="small")
