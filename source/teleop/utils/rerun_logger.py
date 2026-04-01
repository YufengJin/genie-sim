#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Rerun trajectory visualizer for Meta Quest teleoperation debugging.

Logs into three sub-trees, all expressed in base_link (robot) frame:

  world/quest/...   Quest hand poses & trails
                    (absolute VR position after Pico→Robot axis reorder)
  world/robot/...   Robot EE target poses & trails
                    (target_pos/quat from P-controller, before IK / compute_target_pose)
  world/delta/...   Relative offset comparison: VR offset vs EE offset
                    (should match when tracking is correct)

Base frames labelled:
  world/robot/base_link       robot base (fixed at origin)
  world/robot/arm_base_link   robot arm base (updates with waist)
  world/quest/base_frame      Quest tracking origin with vr_to_global orientation

Usage:
    python teleop.py --device_type meta_quest --rerun [--rerun_trail 400]
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import rerun as rr

    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False


# Axis-reorder matrix T: Pico/aligned [x,y,z] → Robot [fwd,left,up]
# robot_pos = [aligned_z, -aligned_x, aligned_y]
_T = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=float)

# ── colours ──────────────────────────────────────────────────────────────────
_C = {
    "quest_left": [100, 150, 255],  # blue
    "quest_right": [255, 140, 80],  # orange
    "ee_left": [70, 220, 110],  # green
    "ee_right": [220, 210, 60],  # yellow
    "actual_left": [160, 200, 255],  # light blue
    "actual_right": [255, 200, 140],  # light orange
    "origin": [210, 210, 210],  # grey  (grip-press markers)
    "error": [220, 80, 80],  # red   (tracking error lines)
    "body_chain": [180, 180, 180],  # grey  (body links)
    "arm_left": [100, 200, 255],  # cyan  (left arm chain)
    "arm_right": [255, 180, 100],  # warm orange (right arm chain)
    "head": [200, 150, 255],  # purple (head chain)
    "tf_line": [120, 120, 120],  # dim grey (connection lines)
}


def _rgba(key, alpha=255):
    return _C[key] + [alpha]


class TeleopRerunLogger:
    """Call from teleop.py when --rerun is set.

    Minimal API surface:
        logger.tick()                                  — advance timeline (once per main loop)
        logger.log_quest_base(vr_to_global_mat)        — when orientation is reset
        logger.log_quest_hand(pos, quat, side, active) — Quest hand in robot frame
        logger.log_grip_origins(vr_pos, ee_pos, side)  — when grip is first pressed
        logger.log_ee_target(pos, quat, side)          — EE target in base_link frame
        logger.log_current_ee(pos, quat, side)         — actual EE from TF
        logger.log_arm_base(pos, quat)                 — arm_base_link in base_link frame
        logger.log_delta(vr_offset, ee_offset, side)   — relative movement comparison
    """

    def __init__(self, trail_len: int = 400):
        if not _RERUN_AVAILABLE:
            raise ImportError(
                "rerun-sdk not installed.\n"
                "  cd source/teleop && uv sync   # adds rerun-sdk from pyproject.toml"
            )

        rr.init("genie_sim_teleop", spawn=True)
        self._frame = 0
        self._trail_len = trail_len

        self._trails: dict[str, deque] = {
            k: deque(maxlen=trail_len)
            for k in (
                "quest_left", "quest_right",
                "ee_left", "ee_right",
                "dvr_left", "dvr_right",
                "dee_left", "dee_right",
            )
        }

        # Robot frame is right-handed X=fwd Y=left Z=up → FLU
        # Rerun default is RDF; tell viewer that "world" uses FLU so axes are sensible.
        rr.log("world", rr.ViewCoordinates.FLU, static=True)

        # Static: robot base_link at origin
        self._axes("world/robot/base_link", np.zeros(3), np.eye(3), size=0.20, static=True)
        rr.log(
            "world/robot/base_link/label",
            rr.Points3D([[0.22, 0.0, 0.0]], colors=[[255, 255, 255]], labels=["robot/base_link"]),
            static=True,
        )

        # Initial Quest base frame placeholder (updated on first orientation reset)
        self._axes("world/quest/base_frame", np.zeros(3), np.eye(3), size=0.20)
        rr.log(
            "world/quest/base_frame/label",
            rr.Points3D([[0.22, 0.0, 0.0]], colors=[[255, 255, 255]], labels=["quest/base_frame"]),
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    def _axes(self, path: str, pos, rot_mat, size: float = 0.10, static: bool = False):
        """Draw RGB = XYZ axes at given pose."""
        pos = np.asarray(pos, dtype=float)
        origins = np.array([pos, pos, pos])
        vectors = np.array(
            [
                rot_mat @ [size, 0.0, 0.0],  # X – red
                rot_mat @ [0.0, size, 0.0],  # Y – green
                rot_mat @ [0.0, 0.0, size],  # Z – blue
            ]
        )
        colors = np.array([[200, 50, 50], [50, 200, 50], [50, 50, 200]], dtype=np.uint8)
        rr.log(f"{path}/axes", rr.Arrows3D(origins=origins, vectors=vectors, colors=colors), static=static)

    def _trail(self, path: str, key: str, pos, color_key: str):
        """Append pos to trail and log as LineStrips3D."""
        self._trails[key].append(np.asarray(pos, dtype=float).copy())
        if len(self._trails[key]) > 1:
            pts = np.array(self._trails[key])
            rr.log(
                path,
                rr.LineStrips3D([pts], colors=[_rgba(color_key, 180)], radii=0.003),
            )

    # ── public API ───────────────────────────────────────────────────────────

    def tick(self):
        """Advance the Rerun timeline. Call once per main loop iteration."""
        rr.set_time_sequence("frame", self._frame)
        self._frame += 1

    def log_quest_base(self, vr_to_global_mat: np.ndarray):
        """Update Quest reference frame when vr_to_global_mat is reset.

        vr_to_global_mat is the inverse of the controller pose at calibration time.
        Supports both 3×3 (rotation-only, legacy) and 4×4 (rotation+translation).
        The Quest base frame axes expressed in *robot* frame are:
            T @ rot.T @ T.T
        where T is the Pico→Robot axis-reorder matrix.
        """
        if vr_to_global_mat.shape == (4, 4):
            rot = vr_to_global_mat[:3, :3]
            pos = vr_to_global_mat[:3, 3]
        else:
            rot = vr_to_global_mat
            pos = np.zeros(3)
        quest_rot_in_robot = _T @ rot.T @ _T.T
        quest_pos_in_robot = _T @ pos
        self._axes("world/quest/base_frame", quest_pos_in_robot, quest_rot_in_robot, size=0.20)

    def log_quest_hand(self, pos, quat, side: str, is_active: bool):
        """Log Quest hand pose in robot frame (X=fwd, Y=left, Z=up).

        pos / quat come from pico_device output: already axis-reordered to robot frame.
        """
        pos = np.asarray(pos, dtype=float)
        rot = R.from_quat(quat).as_matrix()
        key = f"quest_{side}"
        path = f"world/quest/{side}_hand"
        size = 0.07 if is_active else 0.04
        alpha = 255 if is_active else 120
        radius = 0.018 if is_active else 0.010

        self._axes(path, pos, rot, size=size)
        rr.log(f"{path}/point", rr.Points3D([pos], colors=[_rgba(key, alpha)], radii=radius))
        self._trail(f"{path}/trail", key, pos, key)

    def log_grip_origins(self, vr_pos, ee_pos, side: str):
        """Mark the VR and EE positions captured at grip-press."""
        rr.log(
            f"world/quest/{side}_hand/grip_origin",
            rr.Points3D(
                [vr_pos],
                colors=[_rgba("origin", 220)],
                radii=0.025,
                labels=[f"{side} VR origin"],
            ),
        )
        rr.log(
            f"world/robot/ee_{side}_target/grip_origin",
            rr.Points3D(
                [ee_pos],
                colors=[_rgba("origin", 220)],
                radii=0.025,
                labels=[f"{side} EE origin"],
            ),
        )

    def log_ee_target(self, pos_base, quat_base, side: str):
        """Log robot EE target pose in base_link frame (BEFORE IK / compute_target_pose).

        pos_base / quat_base: target_pos / target_quat computed by the P-controller
        in teleop.parse_arm_control(), expressed in base_link frame.
        """
        pos = np.asarray(pos_base, dtype=float)
        rot = R.from_quat(quat_base).as_matrix()
        key = f"ee_{side}"
        path = f"world/robot/ee_{side}_target"

        self._axes(path, pos, rot, size=0.07)
        rr.log(f"{path}/point", rr.Points3D([pos], colors=[_rgba(key)], radii=0.018))
        self._trail(f"{path}/trail", key, pos, key)

    def log_current_ee(self, pos, quat, side: str):
        """Log actual current EE pose from TF (for visual comparison with target)."""
        pos = np.asarray(pos, dtype=float)
        rot = R.from_quat(quat).as_matrix()
        key = f"actual_{side}"
        path = f"world/robot/ee_{side}_actual"

        self._axes(path, pos, rot, size=0.05)
        rr.log(f"{path}/point", rr.Points3D([pos], colors=[_rgba(key, 200)], radii=0.012))

    def log_arm_base(self, pos, quat):
        """Log arm_base_link frame in base_link frame (updates when waist moves)."""
        rot = R.from_quat(quat).as_matrix()
        pos = np.asarray(pos, dtype=float)
        self._axes("world/robot/arm_base_link", pos, rot, size=0.15)
        rr.log(
            "world/robot/arm_base_link/label",
            rr.Points3D(
                [[pos[0] + 0.17, pos[1], pos[2]]],
                colors=[[255, 255, 255]],
                labels=["arm_base_link"],
            ),
        )

    def log_delta(self, vr_offset, ee_offset, side: str):
        """Log relative movements for tracking quality debugging.

        Both should overlap perfectly when the P-controller has converged.

        vr_offset  = spatial_coeff * (vr_pos  - vr_origin)   [robot frame]
        ee_offset  = ee_pos - robot_origin                    [base_link frame]
        """
        vr_off = np.asarray(vr_offset, dtype=float)
        ee_off = np.asarray(ee_offset, dtype=float)
        vr_key = f"quest_{side}"
        ee_key = f"ee_{side}"

        rr.log(f"world/delta/{side}/vr_offset", rr.Points3D([vr_off], colors=[_rgba(vr_key)], radii=0.012))
        rr.log(f"world/delta/{side}/ee_offset", rr.Points3D([ee_off], colors=[_rgba(ee_key)], radii=0.012))

        self._trail(f"world/delta/{side}/vr_trail", f"dvr_{side}", vr_off, vr_key)
        self._trail(f"world/delta/{side}/ee_trail", f"dee_{side}", ee_off, ee_key)

        # tracking error line (vr_offset → ee_offset)
        rr.log(
            f"world/delta/{side}/error_line",
            rr.LineStrips3D([[vr_off, ee_off]], colors=[_rgba("error", 200)]),
        )

    # ── TF tree visualization ────────────────────────────────────────────────

    # Key frames get larger axes for emphasis
    _KEY_FRAMES = {"arm_base_link", "arm_l_end_link", "arm_r_end_link", "body_link5"}

    @staticmethod
    def _frame_color_key(frame_name: str) -> str:
        if "arm_l" in frame_name:
            return "arm_left"
        if "arm_r" in frame_name:
            return "arm_right"
        if "head" in frame_name:
            return "head"
        return "body_chain"

    def log_tf_tree(self, frames: dict, edges: list):
        """Log all TF frames and parent-child connection lines.

        Args:
            frames: dict mapping frame_name -> (pos, quat_xyzw) or None.
            edges: list of (parent_name, child_name) pairs.
        """
        positions = {"base_link": np.zeros(3)}

        for name, data in frames.items():
            if data is None:
                continue
            pos = np.asarray(data[0], dtype=float)
            rot = R.from_quat(data[1]).as_matrix()
            positions[name] = pos

            path = f"world/robot/tf/{name}"
            size = 0.08 if name in self._KEY_FRAMES else 0.04
            self._axes(path, pos, rot, size=size)

            color_key = self._frame_color_key(name)
            label_offset = pos + rot @ [size + 0.02, 0.0, 0.0]
            rr.log(
                f"{path}/label",
                rr.Points3D(
                    [label_offset], colors=[_rgba(color_key, 200)],
                    radii=0.0, labels=[name],
                ),
            )

        # Parent-child connection lines
        segments = []
        for parent, child in edges:
            p_pos = positions.get(parent)
            c_pos = positions.get(child)
            if p_pos is not None and c_pos is not None:
                segments.append([p_pos.tolist(), c_pos.tolist()])
        if segments:
            rr.log(
                "world/robot/tf/edges",
                rr.LineStrips3D(
                    segments,
                    colors=[_rgba("tf_line", 150)] * len(segments),
                    radii=0.002,
                ),
            )

    def log_control_mode(self, mode: str):
        """Log current control mode (absolute or relative) as text."""
        rr.log("control_mode", rr.TextLog(f"Control mode: {mode}"), static=True)

    # ── Legend mode helpers ──────────────────────────────────────────────────

    def log_legend_progress(self, elapsed: float, duration: float):
        """Show calibration timer progress during dual-grip hold."""
        pct = min(elapsed / duration * 100, 100)
        rr.log(
            "legend/status",
            rr.TextLog(f"Calibrating: {elapsed:.1f}s / {duration:.1f}s ({pct:.0f}%)"),
        )

    def log_legend_calibration(
        self,
        vr_l: np.ndarray,
        vr_r: np.ndarray,
        ee_l: np.ndarray,
        ee_r: np.ndarray,
        scale: float,
        t_calib: np.ndarray,
    ):
        """Visualize calibration result: VR↔EE correspondence lines and info."""
        # VR and EE hand markers
        rr.log(
            "world/quest/calib/vr_hands",
            rr.Points3D(
                [vr_l, vr_r],
                colors=[_rgba("quest_left"), _rgba("quest_right")],
                radii=0.025,
                labels=["VR L", "VR R"],
            ),
        )
        rr.log(
            "world/robot/calib/ee_hands",
            rr.Points3D(
                [ee_l, ee_r],
                colors=[_rgba("ee_left"), _rgba("ee_right")],
                radii=0.025,
                labels=["EE L", "EE R"],
            ),
        )

        # Correspondence lines: VR ↔ EE for each hand
        rr.log(
            "world/calib/correspondence",
            rr.LineStrips3D(
                [[vr_l.tolist(), ee_l.tolist()], [vr_r.tolist(), ee_r.tolist()]],
                colors=[_rgba("error", 180), _rgba("error", 180)],
                radii=0.004,
            ),
        )

        # Midpoint markers
        vr_mid = (vr_l + vr_r) / 2
        ee_mid = (ee_l + ee_r) / 2
        rr.log(
            "world/calib/midpoints",
            rr.Points3D(
                [vr_mid, ee_mid],
                colors=[_rgba("origin"), _rgba("origin")],
                radii=0.018,
                labels=["VR mid", "EE mid"],
            ),
        )

        rr.log(
            "legend/status",
            rr.TextLog(
                f"Legend Calibrated! scale={scale:.3f}, "
                f"t_calib=[{t_calib[0]:.3f}, {t_calib[1]:.3f}, {t_calib[2]:.3f}]"
            ),
        )
