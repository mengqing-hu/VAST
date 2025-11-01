import whisper
from pathlib import Path


def generate_subtitle(video_path, output_dir, model_size):
    """
    Generate a subtitle (.srt) file for a video using the OpenAI Whisper model.

    Args:
        video_path: Path to the video file.
        output_dir: Directory where the subtitle file will be saved.
        model_size: Whisper model size, e.g. 'tiny', 'small', 'medium', 'large'.

    Returns:
        The path of the generated subtitle (.srt) file.
    """
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the Whisper model
    print(f"Loading Whisper model: {model_size} ...")
    model = whisper.load_model(model_size)

    print(f"Transcribing audio (model: {model_size})...")
    result = model.transcribe(str(video_path))

    # Write the transcription results to an .srt file
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


if __name__ == "__main__":
    # Here we can still allow manual testing, but with an explicit argument
    test_video = Path("data/raw_videos/test_video.mp4")
    out_dir = Path("data/subtitles")
    # User can manually decide model_size for test
    generate_subtitle(test_video, out_dir, model_size="small")
