"""
keyframe_extractor.py
使用 FFmpeg 从视频中提取关键帧或定时间隔帧。
"""

import ffmpeg
from pathlib import Path

def extract_keyframes(video_path: Path, output_dir: Path, interval: float = 2.0) -> list[Path]:
    """
    从视频中提取关键帧（或定间隔帧）。

    参数:
        video_path (Path): 视频路径。
        output_dir (Path): 输出图像目录。
        interval (float): 每隔多少秒提取一帧（默认 2 秒）。

    返回:
        list[Path]: 生成的关键帧图像路径列表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / f"{video_path.stem}_%04d.jpg")

    # FFmpeg 命令：每 interval 秒取一帧，保存为 JPG
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=1/interval)
        .output(output_pattern, qscale=2, start_number=0)
        .overwrite_output()
        .run(quiet=True)
    )

    keyframes = sorted(output_dir.glob(f"{video_path.stem}_*.jpg"))
    print(f"提取完成，共生成 {len(keyframes)} 张关键帧。")
    return keyframes


# 模块自测入口
if __name__ == "__main__":
    test_video = Path("data/raw_videos/test_video.mp4")
    out_dir = Path("data/keyframes")
    frames = extract_keyframes(test_video, out_dir, interval=2)
    print(f"示例帧: {frames[:5]}")
