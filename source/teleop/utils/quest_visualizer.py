#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Open3D real-time visualizer for Meta Quest controllers.

Shows left hand (blue) and right hand (red) in Pico/Unity coordinate frame
(what quest_bridge.py actually sends to the container). A large coordinate
frame at the origin represents the head position.

Pico/Unity frame: X=right (red), Y=up (green), Z=forward (blue)

Usage (run on host machine with Quest USB-connected):
    cd source/teleop
    uv run python utils/quest_visualizer.py

    # Without Quest hardware (static demo):
    uv run python utils/quest_visualizer.py --demo
"""

import argparse
import sys
import time
import threading

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# ── Quest→Pico transform (same as quest_bridge.py) ──

FLIP = np.diag([1.0, 1.0, -1.0])


def quest_to_pico(mat4):
    """Quest 4x4 → Pico (pos, rot_mat)."""
    pos = FLIP @ mat4[:3, 3]
    rot = FLIP @ mat4[:3, :3] @ FLIP
    det = np.linalg.det(rot)
    if det < 0.5 or det > 1.5:
        rot = np.eye(3)
    return pos, rot


# ── Open3D helpers ──

def update_geometry(geoms, pos_new, rot_new, pos_old, rot_old):
    """Move a list of meshes from old pose to new pose."""
    T_old = np.eye(4)
    T_old[:3, :3] = rot_old
    T_old[:3, 3] = pos_old
    T_new = np.eye(4)
    T_new[:3, :3] = rot_new
    T_new[:3, 3] = pos_new
    delta = np.linalg.inv(T_old) @ T_new
    for g in geoms:
        g.transform(delta)


def make_controller(side):
    """Create controller mesh: a box (body) + coordinate frame (axes)."""
    # Body
    sx, sy, sz = 0.04, 0.025, 0.10
    box = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
    box.translate([-sx / 2, -sy / 2, -sz / 2])
    box.compute_vertex_normals()
    color = [0.25, 0.45, 0.95] if side == "l" else [0.95, 0.35, 0.25]
    box.paint_uniform_color(color)

    # Axes
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)

    return box, frame


class QuestVisualizer:
    def __init__(self, demo=False):
        self.demo = demo
        self.running = True
        self.left_mat = np.eye(4)
        self.right_mat = np.eye(4)
        self.lock = threading.Lock()
        if demo:
            self._set_demo_poses()

    def _set_demo_poses(self):
        """Natural holding position in Quest frame (hand relative to head).

        Quest: +X=right, +Y=up, +Z=backward.
        Natural hold: hands ~25cm to each side, 20cm below, 30cm forward of head.
        """
        # Left hand
        self.left_mat = np.eye(4)
        self.left_mat[:3, 3] = [-0.25, -0.20, -0.30]
        self.left_mat[:3, :3] = R.from_euler("xyz", [0, 15, 0], degrees=True).as_matrix()

        # Right hand
        self.right_mat = np.eye(4)
        self.right_mat[:3, 3] = [0.25, -0.20, -0.30]
        self.right_mat[:3, :3] = R.from_euler("xyz", [0, -15, 0], degrees=True).as_matrix()

    def start_reader(self):
        """Background thread reading Quest data."""
        try:
            from oculus_reader.reader import OculusReader
        except ImportError:
            print("[Visualizer] oculus_reader not available, falling back to --demo")
            self.demo = True
            self._set_demo_poses()
            return

        reader = OculusReader()
        print("[Visualizer] OculusReader started, reading at 30 Hz...")
        while self.running:
            poses, _ = reader.get_transformations_and_buttons()
            if poses:
                with self.lock:
                    l = poses.get("l")
                    r = poses.get("r")
                    if l is not None:
                        self.left_mat = np.asarray(l, dtype=np.float64)
                    if r is not None:
                        self.right_mat = np.asarray(r, dtype=np.float64)
            time.sleep(1.0 / 30)

    def run(self):
        if not self.demo:
            threading.Thread(target=self.start_reader, daemon=True).start()

        vis = o3d.visualization.Visualizer()
        vis.create_window("Quest Teleop — Pico Frame (X=right, Y=up, Z=forward)", width=1024, height=768)

        # Head origin (large frame)
        head_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20)
        vis.add_geometry(head_frame)

        # Ground grid at y=-0.5
        pts, lns = [], []
        for i in range(-5, 6):
            s = 0.1
            idx = len(pts)
            pts += [[i * s, -0.5, -0.5], [i * s, -0.5, 0.5]]
            lns.append([idx, idx + 1])
            idx = len(pts)
            pts += [[-0.5, -0.5, i * s], [0.5, -0.5, i * s]]
            lns.append([idx, idx + 1])
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(pts)
        grid.lines = o3d.utility.Vector2iVector(lns)
        grid.paint_uniform_color([0.4, 0.4, 0.4])
        vis.add_geometry(grid)

        # Controllers
        l_box, l_frame = make_controller("l")
        r_box, r_frame = make_controller("r")
        for g in [l_box, l_frame, r_box, r_frame]:
            vis.add_geometry(g)

        # Track previous poses
        prev_l = (np.zeros(3), np.eye(3))
        prev_r = (np.zeros(3), np.eye(3))

        # Camera
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.3, -1.0])
        ctr.set_lookat([0.0, -0.1, 0.2])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.5)

        print_info()

        frame_count = 0
        last_print = 0.0
        while self.running:
            if not vis.poll_events():
                break
            vis.update_renderer()

            with self.lock:
                lm = self.left_mat.copy()
                rm = self.right_mat.copy()

            l_pos, l_rot = quest_to_pico(lm)
            r_pos, r_rot = quest_to_pico(rm)

            # Update left hand
            update_geometry([l_box, l_frame], l_pos, l_rot, *prev_l)
            for g in [l_box, l_frame]:
                vis.update_geometry(g)
            prev_l = (l_pos.copy(), l_rot.copy())

            # Update right hand
            update_geometry([r_box, r_frame], r_pos, r_rot, *prev_r)
            for g in [r_box, r_frame]:
                vis.update_geometry(g)
            prev_r = (r_pos.copy(), r_rot.copy())

            # Terminal output at 2 Hz
            now = time.time()
            if now - last_print >= 0.5:
                l_euler = R.from_matrix(l_rot).as_euler("xyz", degrees=True)
                r_euler = R.from_matrix(r_rot).as_euler("xyz", degrees=True)
                sys.stdout.write(
                    f"\r  L: pos=({l_pos[0]:+.3f}, {l_pos[1]:+.3f}, {l_pos[2]:+.3f})"
                    f"  rot=({l_euler[0]:+5.1f}, {l_euler[1]:+5.1f}, {l_euler[2]:+5.1f})deg"
                    f"  |  R: pos=({r_pos[0]:+.3f}, {r_pos[1]:+.3f}, {r_pos[2]:+.3f})"
                    f"  rot=({r_euler[0]:+5.1f}, {r_euler[1]:+5.1f}, {r_euler[2]:+5.1f})deg   "
                )
                sys.stdout.flush()
                last_print = now

            frame_count += 1
            time.sleep(1.0 / 30)

        vis.destroy_window()
        self.running = False


def print_info():
    print("""
================================================================================
  Quest Teleop Visualizer — Pico/Unity Coordinate Frame
================================================================================

  3D Window:
    Large axes at origin = HEAD position  (RGB = XYZ)
    Blue box  = LEFT hand controller
    Red box   = RIGHT hand controller
    Each has small RGB axes showing orientation

  Pico/Unity frame (what the bridge sends):
    Red   axis = +X = RIGHT
    Green axis = +Y = UP
    Blue  axis = +Z = FORWARD

  NOTE: Large euler angles at rest are NORMAL.
    pico_device.py has vr_to_global_mat reset: on first grip-press,
    it captures current rotation as reference and subtracts it.
    Only relative changes from grip-press moment affect the robot.

  Direction check — move controller and verify in 3D:
    Push FORWARD → blue axis direction (+Z)
    Move RIGHT   → red axis direction (+X)
    Lift UP      → green axis direction (+Y)

  Press Ctrl+C to exit
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Open3D visualizer for Quest controllers (Pico frame)")
    parser.add_argument("--demo", action="store_true", help="Static demo without Quest hardware")
    args = parser.parse_args()

    viz = QuestVisualizer(demo=args.demo)
    try:
        viz.run()
    except KeyboardInterrupt:
        viz.running = False
        print("\n[Visualizer] Stopped.")


if __name__ == "__main__":
    main()
