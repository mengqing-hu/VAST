import json
import re
from pathlib import Path
from pyannote.audio import Pipeline

from secret_key import hf_token
from vast.utils import get_device


def parse_rttm(rttm_path):

    # Parse RTTM file into JSON-friendly speaker segments.
    segments = []

    with open(rttm_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            # RTTM format:
            # SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA>
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            segments.append({
                "index": i,
                "speaker": speaker,
                "start": start,
                "end": start + duration,
            })

    return segments


def extract_speaker_diarization(wav_path, output_dir):
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    print("Running speaker diarization (pyannote.audio 3.4 / 4.x)")
    print("Audio:", wav_path)

    device = get_device()
    print("Device:", device)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    ).to(device)

    diarization = pipeline(str(wav_path))

    # 取出 Annotation（pyannote.audio >= 3.3）
    annotation = diarization.speaker_diarization

    # RTTM URI 清洗（正则）
    # 只保留：字母、数字、下划线、连字符
    # 其他全部替换为 "_"
    safe_uri = re.sub(r"[^\w\-]", "_", wav_path.stem)
    annotation.uri = safe_uri

    # 导出 RTTM
    rttm_path = output_dir / f"{safe_uri}.rttm"
    print("Exporting RTTM to:", rttm_path)

    with open(rttm_path, "w") as f:
        annotation.write_rttm(f)

    print(f"RTTM saved to {rttm_path}")

    # RTTM -> JSON
    segments = parse_rttm(rttm_path)

    json_path = output_dir / "speaker_diarization.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4, ensure_ascii=False)

    print(f"Speaker diarization JSON saved to {json_path}")
    print(f"Detected {len(segments)} segments")

    return segments
