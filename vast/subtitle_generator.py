import whisper
from pathlib import Path
from box import Box

def generate_subtitle(video_path, output_dir, model_cfg):
    """
    Generate subtitle (.srt) file from video using OpenAI Whisper.
    model_cfg:
        whisper_size: model name (e.g. 'tiny', 'base', 'small', 'medium', 'large-v3')
        language: (optional) target language (e.g. 'de', 'en', or omit for auto-detect)
    """

    if isinstance(model_cfg, Box):
        model_cfg = model_cfg.to_dict()

    print(f"------------------: {model_cfg}")

    whisper_size = model_cfg.get("whisper_size", "base")
    print(f"Resolved Whisper model size: {whisper_size}")
    language = model_cfg.get("language", None)  

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model: {whisper_size}")
    model = whisper.load_model(whisper_size)

    print(f"Transcribing audio... (language={language or 'auto'})")
    result = model.transcribe(str(video_path), language=language)

    srt_path = output_dir / f"{video_path.stem}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        write_srt(result["segments"], f)

    print(f"Subtitle file created: {srt_path}")
    return srt_path


def write_srt(segments, file_obj):
    """Write Whisper transcription results into an .srt subtitle file."""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip().replace("-->", "->")
        file_obj.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format: 00:00:00,000"""
    milliseconds = int(seconds * 1000)
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1000
    milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
