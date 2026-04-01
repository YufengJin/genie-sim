"""
SAM3 Real-time Text-Guided Tracker
===================================
Usage:
    python -m sam3_tracker.app --text "a coffee mug" --camera 0

Keys:
    q  — quit
    r  — reset and re-detect
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _open_camera(cfg: dict) -> cv2.VideoCapture:
    cam_cfg = cfg["camera"]
    cap = cv2.VideoCapture(cam_cfg["index"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_cfg['index']}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
    return cap


def main():
    parser = argparse.ArgumentParser(description="SAM3 real-time text tracker")
    parser.add_argument(
        "--text", required=True, nargs="+",
        help='Object descriptions, e.g. --text "person" "red mug"'
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg["camera"]["index"] = args.camera

    # --- Load SAM3 ---
    print(f"[app] Loading SAM3 on {args.device} ...")
    from .tracker import SAM3Tracker
    from .visualizer import Visualizer

    tracker = SAM3Tracker(device=args.device)
    vis = Visualizer(
        mask_alpha=cfg["tracker"].get("mask_alpha", 0.4),
        show_fps=cfg.get("display", {}).get("show_fps", True),
        show_score=cfg.get("display", {}).get("show_score", True),
    )

    cap = _open_camera(cfg)
    window = cfg.get("display", {}).get("window_name", "SAM3 Tracker")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    score_threshold = cfg["tracker"]["score_threshold"]
    labels = args.text  # list of text prompts

    print(f'[app] Tracking {labels} — press r to reset, q to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            print("[app] Reset.")
            tracker.reset()

        # --- SAM3 text-prompted segmentation (every frame, all prompts) ---
        annotated = frame.copy()
        found_any = False
        color_idx = 0

        for label in labels:
            masks, boxes, scores = tracker.detect(frame, label)
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if score >= score_threshold:
                    annotated = vis.draw(annotated, mask, label, score, color_idx=color_idx, box=box)
                    found_any = True
                color_idx += 1

        if not found_any:
            annotated = vis.draw_waiting(frame, " / ".join(labels))

        vis.draw_overlay(annotated)
        cv2.imshow(window, annotated)

    cap.release()
    cv2.destroyAllWindows()
    print("[app] Done.")


if __name__ == "__main__":
    main()
