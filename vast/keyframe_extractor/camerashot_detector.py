import os
import cv2
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def extract_visual_keyframes(video_path, output_dir):

    print("Running camera shot detection (PySceneDetect, CPU)...")

    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    video_manager.set_downscale_factor()
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    print(f"Detected {len(scene_list)} camera shots")

    visual_frames = []
    cap = cv2.VideoCapture(video_path)

    for i, (start, end) in enumerate(scene_list):
        frame_num = start.get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        if not ret:
            continue

        fname = output_dir / f"scene_{i}.jpg"
        cv2.imwrite(str(fname), frame)

        visual_frames.append({
            "scene_index": i,
            "frame": frame_num,
            "file": str(fname)
        })

    cap.release()
    return visual_frames
