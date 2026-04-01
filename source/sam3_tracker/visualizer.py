"""
Overlay segmentation mask, bounding box, label and FPS onto a BGR frame.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


# Colour palette for objects (BGR)
COLORS = [
    (0, 200, 255),   # amber
    (0, 255, 128),   # green
    (255, 80, 80),   # blue
    (200, 0, 255),   # purple
]


class Visualizer:
    def __init__(self, mask_alpha: float = 0.4, show_fps: bool = True, show_score: bool = True):
        self.mask_alpha = mask_alpha
        self.show_fps = show_fps
        self.show_score = show_score
        self._fps_timer = time.time()
        self._fps = 0.0
        self._frame_count = 0

    def draw(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None,
        label: str,
        score: float,
        color_idx: int = 0,
        box: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Overlay mask + label onto frame and return the annotated frame.

        Draws directly on the passed-in frame (no copy) so that multiple
        calls can layer annotations. Caller should pass a copy if needed.
        """
        color = COLORS[color_idx % len(COLORS)]

        if mask is not None and mask.any():
            self._draw_mask(frame, mask, color)
            self._draw_contour(frame, mask, color)

        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if mask is not None and mask.any():
            self._draw_label(frame, mask, label, score, color)

        return frame

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS and controls. Call once after all draw() calls."""
        self._update_fps()
        if self.show_fps:
            self._draw_fps(frame)
        self._draw_controls(frame)
        return frame

    def draw_waiting(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Shown while waiting for an initial detection."""
        out = frame.copy()
        msg = f'Searching: "{text}" ...'
        cv2.putText(out, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        self._draw_controls(out)
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _draw_mask(self, frame: np.ndarray, mask: np.ndarray, color: tuple):
        overlay = frame.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, self.mask_alpha, frame, 1 - self.mask_alpha, 0, frame)

    def _draw_contour(self, frame: np.ndarray, mask: np.ndarray, color: tuple):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 2)

    def _draw_label(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        label: str,
        score: float,
        color: tuple,
    ):
        # Place label above the topmost mask pixel
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return
        x_center = int(xs.mean())
        y_top = int(ys.min()) - 8

        text = label
        if self.show_score:
            text = f"{label}  {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x0 = max(0, x_center - tw // 2)
        y0 = max(th + baseline, y_top)

        # Background pill
        cv2.rectangle(frame, (x0 - 4, y0 - th - 4), (x0 + tw + 4, y0 + baseline), color, -1)
        cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

    def _draw_fps(self, frame: np.ndarray):
        h = frame.shape[0]
        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_controls(self, frame: np.ndarray):
        h = frame.shape[0]
        cv2.putText(
            frame,
            "q: quit   r: reset",
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
