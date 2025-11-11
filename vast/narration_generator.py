"""

Generate narration audio files from text summaries.
Each summary is converted to speech (mp3/wav).

"""

from gtts import gTTS
from pathlib import Path
import json
from tqdm import tqdm


def generate_narration_from_summaries(summaries_json, output_dir="data/audio", lang="de"):
    """
    Generate narration (text-to-speech) audio for each summarized scene.

    Args:
        summaries_json (Path): JSON file containing scene summaries
        output_dir (str | Path): output folder for audio files
        lang (str): narration language (e.g. 'de' for German, 'en' for English)
    Returns:
        list[dict]: updated metadata with audio paths
    """
    summaries_json = Path(summaries_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary data
    with open(summaries_json, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    results = []
    print(f"Generating narration for {len(summaries)} sections...")

    for i, item in enumerate(tqdm(summaries, desc="Generating narration")):
        text = item.get("summary", "").strip()
        if not text:
            print(f"Section {i} has no summary, skipping.")
            continue

        audio_path = output_dir / f"scene_{i:03d}.mp3"

        try:
            tts = gTTS(text, lang=lang)
            tts.save(str(audio_path))
            print(f"Saved narration: {audio_path.name}")
        except Exception as e:
            print(f"Failed to generate narration for section {i}: {e}")
            audio_path = None

        item["narration_audio"] = str(audio_path) if audio_path else None
        results.append(item)

    # Save updated metadata
    output_json = output_dir / "narration_metadata.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Narration metadata saved to {output_json}")
    return results
