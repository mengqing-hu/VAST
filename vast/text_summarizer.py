"""
text_summarizer.py
----------------------------------------
Generate text summaries for segmented video sections
based on their subtitles (timestamp + text).
----------------------------------------
"""

import json
from pathlib import Path
from transformers import pipeline


def summarize_sections(subtitles_json, segments_json, output_json,
                       summarizer_model="facebook/bart-large-cnn",
                       max_length=120, min_length=25, language="en"):
    """
    Perform text summarization for each segmented video section.

    Args:
        subtitles_json (Path): Path to Whisper subtitle JSON
        segments_json (Path): Path to scene segments JSON (start, end)
        output_json (Path): Output path for summaries
        summarizer_model (str): Hugging Face summarization model
        max_length (int): Maximum tokens for summary
        min_length (int): Minimum tokens for summary
        language (str): "en", "de" etc. for model adaptation
    """

    subtitles_json = Path(subtitles_json)
    segments_json = Path(segments_json)
    output_json = Path(output_json)

    # 读取字幕和场景分段
    with open(subtitles_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)
    with open(segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Loaded {len(subtitles)} subtitles and {len(segments)} segments")

    # 加载摘要模型（多语言可切换）
    if language == "de":
        model_name = "ml6team/mt5-small-german-finetune-mlsum"
    elif language == "en":
        model_name = summarizer_model
    else:
        model_name = "google/mt5-small"  # fallback multilingual model

    print(f"Loading summarization model: {model_name}")
    summarizer = pipeline("summarization", model=model_name)

    results = []
    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]

        # 聚合该时间段的所有字幕文本
        texts = [
            s["text"] for s in subtitles
            if s["start"] >= start and s["end"] <= end
        ]
        full_text = " ".join(texts).strip()

        if not full_text:
            summary = ""
        else:
            # 执行摘要
            result = summarizer(full_text, max_length=max_length,
                                min_length=min_length, do_sample=False)
            summary = result[0]["summary_text"]

        section = {
            "start": start,
            "end": end,
            "text": full_text,
            "summary": summary
        }
        results.append(section)
        print(f"Section {i}: summarized {len(full_text.split())} words.")

    # 保存 JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summaries saved to {output_json}")
    return results
