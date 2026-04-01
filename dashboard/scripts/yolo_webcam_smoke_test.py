#!/usr/bin/env python3
"""Smoke-test YOLO-World + webcam inside the dashboard Docker image (headless or with GUI).

Aligns with others/yolo-world/detect_webcam.py: VideoCapture -> resize -> YOLOWorld.predict.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

DEFAULT_CLASSES = [
    "person", "face", "hand",
    "cup", "bottle", "glass",
    "phone", "laptop", "keyboard", "mouse", "monitor",
    "chair", "table", "book", "pen",
    "bag", "backpack", "hat",
    "cat", "dog",
    "car", "bicycle",
]


def _resolve_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO-World webcam smoke test (container-friendly)")
    parser.add_argument(
        "--model",
        default=os.environ.get("YOLO_MODEL_PATH", "yolov8x-worldv2"),
        help="Model path or name (default: YOLO_MODEL_PATH or yolov8x-worldv2)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Custom class names (space-separated)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Max side for inference resize")
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Headless mode: number of frames to process (default: 30). Ignored with --show.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open GUI window (needs DISPLAY and X11 socket). Press 'q' to quit.",
    )
    args = parser.parse_args()

    classes = list(args.classes) if args.classes else list(DEFAULT_CLASSES)

    try:
        import torch
    except ImportError:
        torch = None

    cuda_avail = bool(torch and torch.cuda.is_available())
    print(f"CUDA available: {cuda_avail}", flush=True)

    from ultralytics import YOLOWorld

    dev = _resolve_device()
    print(f"Loading model: {args.model} (device={dev})", flush=True)
    t_load = time.perf_counter()
    model = YOLOWorld(args.model)
    if dev != "cpu":
        model.to(dev)
    model.set_classes(classes)
    print(
        f"Model ready in {time.perf_counter() - t_load:.2f}s; classes: {classes}",
        flush=True,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}", file=sys.stderr, flush=True)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if args.show:
        print("Press 'q' to quit", flush=True)
        fps_smooth = 0.0
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End of stream or read failed", flush=True)
                break

            h, w = frame.shape[:2]
            scale = args.imgsz / max(h, w)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame

            results = model.predict(small, conf=args.conf, verbose=False, device=dev)
            annotated = results[0].plot()

            if scale < 1.0:
                annotated = cv2.resize(annotated, (w, h))

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            cv2.putText(
                annotated,
                f"FPS: {fps_smooth:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLO-World Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        latencies: list[float] = []
        ok = 0
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {i}: read failed", flush=True)
                break

            h, w = frame.shape[:2]
            scale = args.imgsz / max(h, w)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame

            t0 = time.perf_counter()
            results = model.predict(small, conf=args.conf, verbose=False, device=dev)
            _ = results[0].plot()
            latencies.append(time.perf_counter() - t0)
            ok += 1

        if not latencies:
            print("No frames processed", file=sys.stderr, flush=True)
            cap.release()
            return 1

        avg_ms = sum(latencies) / len(latencies) * 1000.0
        min_ms = min(latencies) * 1000.0
        max_ms = max(latencies) * 1000.0
        print(
            f"Headless done: {ok} frames, infer time ms: avg={avg_ms:.1f} min={min_ms:.1f} max={max_ms:.1f}",
            flush=True,
        )

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
