"""
SAM3 real-time text-guided segmentation and tracking.

Uses SAM3's native text prompt API:
  - Sam3Processor for per-frame text → segmentation
  - build_sam3_video_predictor for video tracking (when using frame buffer)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class SAM3Tracker:
    """
    Real-time tracker using SAM3 native text prompt.

    For webcam streaming, uses Sam3Processor per-frame (text → masks).
    Maintains the last known masks for smooth tracking.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._processor = None
        self._last_masks = None
        self._last_boxes = None
        self._last_scores = None
        self._load_model()

    def _load_model(self):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("[tracker] Loading SAM3 model...")
        self._model = build_sam3_image_model()
        self._processor = Sam3Processor(self._model)
        print("[tracker] SAM3 model loaded.")

    def detect(
        self, frame: np.ndarray, text: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """
        Run text-prompted segmentation on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)
            text:  natural language description, e.g. "a coffee mug"

        Returns:
            masks:  list of (H, W) bool arrays — one per detected instance
            boxes:  list of [x1, y1, x2, y2] arrays
            scores: list of float confidences
        """
        rgb = frame[..., ::-1]  # BGR → RGB
        pil_image = Image.fromarray(rgb)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(
                state=inference_state, prompt=text
            )

        raw_masks = output["masks"]   # tensor or list
        raw_boxes = output["boxes"]   # tensor or list
        raw_scores = output["scores"] # tensor or list

        h, w = frame.shape[:2]
        masks = self._to_mask_list(raw_masks, h, w)
        boxes = self._to_box_list(raw_boxes)
        scores = self._to_score_list(raw_scores)

        self._last_masks = masks
        self._last_boxes = boxes
        self._last_scores = scores

        return masks, boxes, scores

    def best_result(
        self, frame: np.ndarray, text: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """
        Return the highest-confidence (mask, box, score) for text prompt.
        Returns (None, None, 0.0) if nothing detected.
        """
        masks, boxes, scores = self.detect(frame, text)
        if not scores:
            return None, None, 0.0
        idx = int(np.argmax(scores))
        return masks[idx], boxes[idx], scores[idx]

    @property
    def last_masks(self) -> list[np.ndarray] | None:
        return self._last_masks

    @property
    def last_boxes(self) -> list[np.ndarray] | None:
        return self._last_boxes

    @property
    def last_scores(self) -> list[float] | None:
        return self._last_scores

    def reset(self):
        """Clear cached state."""
        self._last_masks = None
        self._last_boxes = None
        self._last_scores = None

    # ------------------------------------------------------------------
    # Internal converters
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mask_list(raw, h: int, w: int) -> list[np.ndarray]:
        """Convert SAM3 mask output to list of (H, W) bool arrays."""
        import cv2

        if isinstance(raw, torch.Tensor):
            raw = raw.cpu().numpy()

        if isinstance(raw, np.ndarray):
            if raw.ndim == 2:
                masks_np = [raw]
            elif raw.ndim == 3:
                masks_np = [raw[i] for i in range(raw.shape[0])]
            elif raw.ndim == 4:
                # (N, 1, H, W) or (N, C, H, W)
                masks_np = [raw[i, 0] for i in range(raw.shape[0])]
            else:
                masks_np = [raw]
        elif isinstance(raw, (list, tuple)):
            masks_np = []
            for m in raw:
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                if m.ndim >= 3:
                    m = m.squeeze()
                masks_np.append(m)
        else:
            return []

        result = []
        for m in masks_np:
            m_bool = (m > 0.5) if m.dtype != bool else m
            if m_bool.shape[:2] != (h, w):
                m_bool = cv2.resize(
                    m_bool.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            result.append(m_bool)
        return result

    @staticmethod
    def _to_box_list(raw) -> list[np.ndarray]:
        if isinstance(raw, torch.Tensor):
            raw = raw.cpu().numpy()
        if isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                return [raw.astype(np.float32)]
            return [raw[i].astype(np.float32) for i in range(raw.shape[0])]
        if isinstance(raw, (list, tuple)):
            result = []
            for b in raw:
                if isinstance(b, torch.Tensor):
                    b = b.cpu().numpy()
                result.append(np.asarray(b, dtype=np.float32))
            return result
        return []

    @staticmethod
    def _to_score_list(raw) -> list[float]:
        if isinstance(raw, torch.Tensor):
            raw = raw.float().cpu().numpy()
        if isinstance(raw, np.ndarray):
            return raw.flatten().tolist()
        if isinstance(raw, (list, tuple)):
            return [float(s) for s in raw]
        return [float(raw)]
