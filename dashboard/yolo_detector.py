"""YOLO-World open-vocabulary object detector for dashboard camera frames."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger("dashboard.yolo_detector")

DEFAULT_CLASSES = [
    "person", "hand", "cup", "bottle", "phone",
    "laptop", "keyboard", "mouse", "chair", "table",
]


class YoloDetector:
    """Thread-safe wrapper around ultralytics YOLOWorld.

    The model is lazy-loaded on first detection call to avoid slowing dashboard
    startup and consuming GPU memory when detection is not in use.
    """

    def __init__(
        self,
        model_path: str = "yolov8x-worldv2",
        conf: float = 0.3,
        default_classes: list[str] | None = None,
    ):
        self._model_path = model_path
        self._conf = conf
        self._model = None
        self._lock = threading.Lock()
        self._enabled: dict[str, bool] = {}
        self._classes: list[str] = list(default_classes or DEFAULT_CLASSES)
        self._classes_dirty = True
        self._detect_count = 0
        self._device: str | None = None

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch

            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        logger.info("YOLO inference device: %s", self._device)
        return self._device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLOWorld

        dev = self._resolve_device()
        logger.info("Loading YOLO-World model: %s (device=%s)", self._model_path, dev)
        t0 = time.perf_counter()
        self._model = YOLOWorld(self._model_path)
        if dev != "cpu":
            self._model.to(dev)
        self._model.set_classes(self._classes)
        self._classes_dirty = False
        logger.info(
            "YOLO-World model loaded in %.2fs, classes=%s, device=%s",
            time.perf_counter() - t0,
            self._classes,
            dev,
        )

    def set_enabled(self, camera: str, enabled: bool) -> None:
        self._enabled[camera] = enabled
        logger.info("YOLO enabled for camera %r: %s", camera, enabled)

    def is_enabled(self, camera: str) -> bool:
        return self._enabled.get(camera, False)

    def set_classes(self, classes: list[str]) -> None:
        with self._lock:
            self._classes = list(classes)
            self._classes_dirty = True
        logger.info("YOLO classes updated: %s", self._classes)

    def set_confidence(self, conf: float) -> None:
        self._conf = max(0.05, min(0.95, conf))
        logger.info("YOLO confidence set to %.3f", self._conf)

    def detect(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Run detection on an RGB frame and return the annotated RGB frame."""
        with self._lock:
            self._ensure_model()
            if self._classes_dirty:
                self._model.set_classes(self._classes)
                self._classes_dirty = False
            dev = self._resolve_device()
            # YOLO expects BGR input (OpenCV convention)
            bgr_in = np.ascontiguousarray(rgb_frame[:, :, ::-1])
            t0 = time.perf_counter()
            results = self._model.predict(
                bgr_in, conf=self._conf, verbose=False, device=dev
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._detect_count += 1
            if self._detect_count == 1:
                logger.info(
                    "First YOLO inference: shape=%s conf=%.3f classes=%s took %.1fms",
                    getattr(rgb_in, "shape", None),
                    self._conf,
                    self._classes,
                    elapsed_ms,
                )
            elif self._detect_count % 60 == 0:
                logger.info(
                    "YOLO inference #%d: %.1fms (conf=%.3f)",
                    self._detect_count,
                    elapsed_ms,
                    self._conf,
                )
            else:
                logger.debug(
                    "YOLO inference #%d: %.1fms shape=%s",
                    self._detect_count,
                    elapsed_ms,
                    getattr(rgb_in, "shape", None),
                )
            annotated_bgr = results[0].plot()
            return annotated_bgr[:, :, ::-1].copy()

    def get_status(self) -> dict:
        return {
            "enabled": dict(self._enabled),
            "classes": list(self._classes),
            "confidence": self._conf,
            "model_loaded": self._model is not None,
            "device": self._resolve_device() if self._model is not None else None,
        }
