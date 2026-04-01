#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Meta Quest Bridge for Genie Sim Teleoperation.

Reads Meta Quest controller data via oculus_reader (USB/ADB) on the host machine,
converts it to Pico-compatible UDP JSON, and forwards to the Genie Sim teleop
process running inside a Docker container.

Architecture note
-----------------
This bridge runs on the HOST machine (uv environment). Its only coordinate-system
responsibility is the Quest → Pico handedness flip (see quest_pose_to_pico_format).
All subsequent Pico → Robot transformations and IK solving happen INSIDE the
container (pico_device.py, teleop.py, genie_motion_control).

Usage (run on host machine, NOT inside Docker):
    cd source/teleop
    uv run python utils/quest_bridge.py --target_ip <CONTAINER_IP> --target_port 8081

    # Debug mode (full terminal display, no UDP):
    uv run python utils/quest_bridge.py --debug

    # Rerun visualization (send UDP + open Rerun viewer):
    uv run python utils/quest_bridge.py --rerun --target_ip <CONTAINER_IP>
"""

import argparse
import json
import socket
import time
from collections import deque

import numpy as np
from oculus_reader.reader import OculusReader
from scipy.spatial.transform import Rotation as R


def quest_pose_to_pico_format(mat, side, buttons):
    """Convert a single Quest controller 4x4 pose matrix + buttons to Pico JSON dict.

    Why this conversion is needed
    ------------------------------
    Quest uses a right-handed coordinate system:  X=right, Y=up, Z=backward
    Pico/Unity uses a left-handed coordinate system: X=right, Y=up, Z=forward

    The Z axes point in opposite directions, so we apply a handedness flip:
        F = diag(1, 1, -1)
        position:  pos' = F @ pos          (negate Z)
        rotation:  R'   = F @ R @ F        (similarity transform preserves meaning)

    This is the ONLY coordinate-system work done on the host side.
    All further Pico → Robot transformations happen inside the container
    in pico_device.py (axis reordering + orientation alignment).

    Args:
        mat: 4x4 numpy array from oculus_reader (controller pose in Quest frame)
        side: "l" or "r"
        buttons: dict from oculus_reader.get_transformations_and_buttons()

    Returns:
        dict matching Pico AIDEA App UDP JSON format
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        mat = np.eye(4)

    pos = mat[:3, 3].copy()
    rot_mat = mat[:3, :3].copy()

    # Validate rotation matrix; fall back to identity if degenerate
    det = np.linalg.det(rot_mat)
    if det < 0.5 or det > 1.5:
        rot_mat = np.eye(3)

    # Quest (right-handed, +Z=backward) → Pico/Unity (left-handed, +Z=forward)
    # Handedness flip: F = diag(1,1,-1), R' = F R F, pos' = F pos
    flip = np.diag([1.0, 1.0, -1.0])
    pos = flip @ pos
    rot_mat = flip @ rot_mat @ flip
    quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]

    if side == "l":
        # Grip trigger: prefer analog leftGrip (float tuple) over bool LG
        grip_raw = buttons.get("leftGrip", (0.0,))
        hand_trig = float(grip_raw[0]) if isinstance(grip_raw, (list, tuple)) else float(grip_raw)

        # Index trigger: OculusReader returns (float,) for leftTrig
        index_trig_raw = buttons.get("leftTrig", (0.0,))
        index_trig = float(index_trig_raw[0]) if isinstance(index_trig_raw, (list, tuple)) else float(index_trig_raw)

        # Buttons
        key_one = "true" if buttons.get("X", False) else "false"
        key_two = "true" if buttons.get("Y", False) else "false"

        # Joystick
        js = buttons.get("leftJS", [0.0, 0.0])
        axis_x = float(js[0]) if isinstance(js, (list, tuple)) else 0.0
        axis_y = float(js[1]) if isinstance(js, (list, tuple)) else 0.0
        axis_click = "true" if buttons.get("LJ", False) else "false"
    else:
        # Grip trigger: prefer analog rightGrip (float tuple) over bool RG
        grip_raw = buttons.get("rightGrip", (0.0,))
        hand_trig = float(grip_raw[0]) if isinstance(grip_raw, (list, tuple)) else float(grip_raw)

        index_trig_raw = buttons.get("rightTrig", (0.0,))
        index_trig = float(index_trig_raw[0]) if isinstance(index_trig_raw, (list, tuple)) else float(index_trig_raw)

        key_one = "true" if buttons.get("A", False) else "false"
        key_two = "true" if buttons.get("B", False) else "false"

        js = buttons.get("rightJS", [0.0, 0.0])
        axis_x = float(js[0]) if isinstance(js, (list, tuple)) else 0.0
        axis_y = float(js[1]) if isinstance(js, (list, tuple)) else 0.0
        axis_click = "true" if buttons.get("RJ", False) else "false"

    return {
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "rotation": {"x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]), "w": float(quat[3])},
        "handTrig": hand_trig,
        "indexTrig": index_trig,
        "keyOne": key_one,
        "keyTwo": key_two,
        "axisX": axis_x,
        "axisY": axis_y,
        "axisClick": axis_click,
    }


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"


def _trig_bar(val, width=10):
    filled = int(max(0.0, min(1.0, val)) * width)
    bar = "█" * filled + "░" * (width - filled)
    color = _RED if val > 0.8 else _YELLOW if val > 0.4 else _RESET
    return f"{color}[{bar}]{_RESET} {val:.2f}"


def _render_debug(left_mat, right_mat, left_data, right_data, buttons, frame_count, hz_actual):
    """Full-information terminal display (Pico coordinate frame only)."""
    lines = ["\033[2J\033[H"]  # clear screen
    lines.append(f"{_BOLD}{'=' * 72}{_RESET}")
    lines.append(f"  {_CYAN}Quest Bridge — Debug Mode{_RESET}   frame #{frame_count}   {hz_actual:.1f} Hz")
    lines.append(f"{_BOLD}{'=' * 72}{_RESET}")
    lines.append("")

    # Raw 4x4 matrices
    lines.append(f"  {_BOLD}RAW MATRICES (Quest frame: X=right Y=up Z=backward){_RESET}")
    for label, mat in [("Left ", left_mat), ("Right", right_mat)]:
        lines.append(f"    {label}:")
        for row in mat:
            lines.append(f"      [{row[0]:+8.4f}  {row[1]:+8.4f}  {row[2]:+8.4f}  {row[3]:+8.4f}]")
    lines.append("")

    # Per-hand Pico data
    for label, data, k1_name, k2_name in [
        ("LEFT ", left_data, "X(reset)", "Y(against)"),
        ("RIGHT", right_data, "A(reset)", "B(waist)"),
    ]:
        p = data["position"]
        q = data["rotation"]
        grip = data["handTrig"]
        index_t = data["indexTrig"]
        arm_on = grip > 0.8

        arm_tag = f"  {_RED}{_BOLD}<< ARM ON{_RESET}" if arm_on else ""
        lines.append(f"  {_BOLD}--- {label} HAND (Pico: X=right Y=up Z=fwd) ---{_RESET}{arm_tag}")
        lines.append(f"    pos : x={p['x']:+.4f}  y={p['y']:+.4f}  z={p['z']:+.4f}")
        lines.append(f"    quat: x={q['x']:+.4f}  y={q['y']:+.4f}  z={q['z']:+.4f}  w={q['w']:+.4f}")

        lines.append(f"    grip    {_trig_bar(grip)}{arm_tag}")
        gripper_val = 1.0 - index_t
        lines.append(f"    index   {_trig_bar(index_t)}   gripper={gripper_val:.2f}")

        # Buttons
        btns = []
        if data["keyOne"] == "true":
            btns.append(f"{_GREEN}{_BOLD}{k1_name}{_RESET}")
        if data["keyTwo"] == "true":
            btns.append(f"{_GREEN}{_BOLD}{k2_name}{_RESET}")
        js_click = data.get("axisClick", "false")
        if js_click == "true":
            btns.append(f"{_YELLOW}JS-click{_RESET}")
        btn_str = "  ".join(btns) if btns else "(none)"
        lines.append(f"    buttons {btn_str}")

        js_x = data.get("axisX", 0.0)
        js_y = data.get("axisY", 0.0)
        lines.append(f"    joystick x={js_x:+.3f}  y={js_y:+.3f}")
        lines.append("")

    lines.append(f"  {_BOLD}Raw buttons dict:{_RESET} {buttons}")
    lines.append("")
    lines.append(f"  Press Ctrl+C to exit")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

_AXIS_LEN_WORLD = 0.15  # reference frame arrow length (metres)
_AXIS_LEN_HAND = 0.06   # hand pose frame arrow length (metres)

# X=red  Y=green  Z=blue  (RGB ↔ XYZ convention)
_AXIS_COLORS = [[220, 50, 50, 255], [50, 200, 50, 255], [50, 100, 220, 255]]
_LEFT_TRAIL_COLOR = [100, 160, 255, 200]   # blue-ish
_RIGHT_TRAIL_COLOR = [255, 130, 80, 200]   # orange-ish


def _init_rerun():
    import rerun as rr

    rr.init("quest_bridge", spawn=True)

    # Set world coordinate convention: Pico/Unity frame (X=right, Y=up, Z=fwd)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Static reference frame at world origin showing Pico coordinate axes
    origins = np.zeros((3, 3), dtype=np.float32)
    vectors = np.eye(3, dtype=np.float32) * _AXIS_LEN_WORLD
    rr.log(
        "world/pico_frame",
        rr.Arrows3D(origins=origins, vectors=vectors, colors=_AXIS_COLORS, labels=["X (right)", "Y (up)", "Z (fwd)"]),
        static=True,
    )

    return rr


def _log_rerun(rr, left_data, right_data, frame_count, left_trail, right_trail, prev_buttons):
    """Log one frame of Quest data to Rerun (Pico coordinate frame)."""
    rr.set_time_sequence("frame", frame_count)

    for side, data, trail, trail_color in [
        ("left", left_data, left_trail, _LEFT_TRAIL_COLOR),
        ("right", right_data, right_trail, _RIGHT_TRAIL_COLOR),
    ]:
        p = data["position"]
        q = data["rotation"]
        pos_arr = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)

        # Rotate the three unit axes by the hand's quaternion into world space.
        # This is computed explicitly rather than relying on Rerun's entity-hierarchy
        # transform inheritance, which does not reliably propagate to Arrows3D.
        rot = R.from_quat([q["x"], q["y"], q["z"], q["w"]])
        axis_vectors = rot.apply(np.eye(3, dtype=np.float32) * _AXIS_LEN_HAND)
        origins = np.tile(pos_arr, (3, 1))
        rr.log(f"pico/{side}/axes", rr.Arrows3D(origins=origins, vectors=axis_vectors, colors=_AXIS_COLORS))

        # Motion trail
        trail.append(pos_arr.tolist())
        if len(trail) >= 2:
            rr.log(
                f"pico/{side}/trail",
                rr.LineStrips3D([list(trail)], colors=[trail_color]),
            )

        # Trigger scalars
        rr.log(f"pico/{side}/grip", rr.Scalar(data["handTrig"]))
        rr.log(f"pico/{side}/index", rr.Scalar(data["indexTrig"]))

    # Button events (log only on state change)
    all_btns = {
        "L_X": left_data["keyOne"] == "true",
        "L_Y": left_data["keyTwo"] == "true",
        "L_JS": left_data.get("axisClick", "false") == "true",
        "R_A": right_data["keyOne"] == "true",
        "R_B": right_data["keyTwo"] == "true",
        "R_JS": right_data.get("axisClick", "false") == "true",
    }
    for name, state in all_btns.items():
        if state != prev_buttons.get(name, False):
            rr.log("pico/buttons", rr.TextLog(f"{name} {'PRESS' if state else 'RELEASE'}"))
    prev_buttons.update(all_btns)


# ---------------------------------------------------------------------------
# Main bridge loop
# ---------------------------------------------------------------------------

def run_bridge(target_ip, target_port, hz, debug, use_rerun):
    """Main bridge loop.

    Args:
        target_ip: IP of the Genie Sim teleop process (container IP)
        target_port: UDP port to send to (default 8081)
        hz: Send frequency
        debug: If True, show full terminal display instead of sending UDP
        use_rerun: If True, open Rerun viewer + send UDP
    """
    print(f"[QuestBridge] Initializing OculusReader...")
    reader = OculusReader()

    rr = None
    left_trail = deque(maxlen=200)
    right_trail = deque(maxlen=200)
    prev_buttons = {}

    sock = None
    if not debug:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[QuestBridge] Sending to {target_ip}:{target_port} at {hz} Hz")

    if use_rerun:
        rr = _init_rerun()
        print(f"[QuestBridge] Rerun viewer launched")
    elif debug:
        print(f"[QuestBridge] Debug mode — full terminal display at {hz} Hz (no UDP)")

    period = 1.0 / hz
    frame_count = 0
    t_start = time.time()

    while True:
        loop_start = time.time()

        poses, buttons = reader.get_transformations_and_buttons()
        if poses == {}:
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            continue

        left_raw = poses.get("l", np.eye(4))
        right_raw = poses.get("r", np.eye(4))
        left_mat = np.asarray(left_raw, dtype=np.float64)
        right_mat = np.asarray(right_raw, dtype=np.float64)

        if frame_count == 0:
            print(f"[QuestBridge] First frame raw data:")
            print(f"  Left  shape={left_mat.shape}\n  {left_mat}")
            print(f"  Right shape={right_mat.shape}\n  {right_mat}")
            print(f"  Buttons: {buttons}")

        left_data = quest_pose_to_pico_format(left_mat, "l", buttons)
        right_data = quest_pose_to_pico_format(right_mat, "r", buttons)
        payload = json.dumps([left_data, right_data])

        if debug:
            hz_actual = frame_count / (time.time() - t_start) if frame_count > 0 else 0.0
            _render_debug(left_mat, right_mat, left_data, right_data, buttons, frame_count, hz_actual)
        else:
            sock.sendto(payload.encode("utf-8"), (target_ip, target_port))
            if use_rerun:
                _log_rerun(rr, left_data, right_data, frame_count, left_trail, right_trail, prev_buttons)

        frame_count += 1

        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)


def main():
    parser = argparse.ArgumentParser(description="Meta Quest → Pico-compatible UDP bridge for Genie Sim")
    parser.add_argument("--target_ip", type=str, default="127.0.0.1", help="Target IP (container or localhost)")
    parser.add_argument("--target_port", type=int, default=8081, help="Target UDP port (default 8081)")
    parser.add_argument("--hz", type=int, default=30, help="Send frequency in Hz (default 30)")
    parser.add_argument("--debug", action="store_true", help="Full terminal display (no UDP send)")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun visualization: send UDP + log 3D poses/triggers/buttons to Rerun viewer",
    )
    args = parser.parse_args()
    run_bridge(args.target_ip, args.target_port, args.hz, args.debug, args.rerun)


if __name__ == "__main__":
    main()
