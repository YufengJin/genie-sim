#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R

# from utils.ik_utils import IKSolver
from utils.logger import Logger
from utils.ros_utils import RosUtils
from utils.name_utils import *
from devices.pico_device import PicoDevice
from devices.meta_quest_device import MetaQuestDevice
from pynput import keyboard
from geometry_msgs.msg import Pose

import os, sys, argparse, math
from copy import deepcopy
import json
import time, subprocess, shutil, signal
import yaml

from config.robot_interface import *
from utils.ros_nodes import TF_TREE_EDGES

logger = Logger()


class TeleOp(object):
    def __init__(self, args, ik_version="0.4.3"):
        self.args = args
        self.port = args.port
        self.reset_flg = False
        self.switch_flg = False
        self.robot_cfg = args.robot_cfg
        self.last_eef_pub = [None, None]
        self.last_on = [False, False]
        self.last_eef_control_on = [0.0, 0.0]
        # --- Closed-loop motion planner state (Role-ROS2 style) ---
        self.vr_origin_pos = [None, None]
        self.vr_origin_quat = [None, None]
        self.robot_origin_pos = [None, None]
        self.robot_origin_quat = [None, None]
        self.spatial_coeff = getattr(args, "spatial_coeff", 1.0)
        self.pos_gain = getattr(args, "pos_gain", 5.0)
        self.rot_gain = getattr(args, "rot_gain", 2.0)
        self.max_pos_step = getattr(args, "max_pos_step", 0.02)
        self.max_rot_step = getattr(args, "max_rot_step", 0.1)
        self.legend_mode = getattr(args, "legend", False)
        self.loop_hz = 30.0
        self.ee_pub = [None, None]
        self.waist_pub = None
        self.waist_update = False
        self.update_wait_num = [0, 0]
        self.iskeyboard = False
        self.keyboard_pos = [None, None]
        self.ik_version = ik_version
        self.current_mode = "realtime"
        self.is_recording = False
        self.host_ip = self.get_local_ip()
        self.test_num = 0
        self.process_pid = []
        self.waist_angle = 0
        if self.host_ip == None:
            print("======>Can't get host_ip!!!! Please enter the ip address manually !!!")
            self.host_ip = args.host_ip
        self.setup_robot()
        self.count = 0
        self.pub_mc_count = 0
        self.waist_control_count = 0

        self.ros_utils = RosUtils(self.robot_name)
        self.device_type = args.device_type
        self.setup_device()
        if self.legend_mode:
            self.device.skip_vr_calibration = True
            self.device.reset_orientation = False

        # Legend mode state
        self.legend_calibrated = False
        self.legend_scale = 1.0
        self.legend_offset = np.zeros(3)     # position offset: ee_mid - scale * vr_mid
        self._legend_grip_start = None
        self._legend_calib_samples = []
        self._LEGEND_CALIB_DURATION = 3.0
        self._legend_ema_pos = [None, None]    # [left, right] smoothed position
        self._legend_ema_alpha = 0.8           # EMA weight for new value
        self._legend_vr_origin = [None, None]  # VR pos when trigger first pressed
        self._legend_ee_origin = [None, None]  # EE pos when trigger first pressed
        # Fixed EE rotation in base_link frame (rotation locked after calibration)
        # Left:  end_link Z=base X, end_link X=-base Y
        # Right: end_link Z=base X, end_link X=+base Y
        self._legend_fixed_quat = [
            R.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).as_quat(),  # left
            R.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).as_quat(),    # right
        ]

        self.robot_init_body_state = None
        self.robot_init_head_state = None
        self.robot_init_arm = None
        self.robot_init_hand = None

        # Rerun visualizer (optional, enabled by --rerun)
        self.rerun_logger = None
        if getattr(args, "rerun", False):
            try:
                from utils.rerun_logger import TeleopRerunLogger
                trail_len = getattr(args, "rerun_trail", 400)
                self.rerun_logger = TeleopRerunLogger(trail_len=trail_len)
            except ImportError as e:
                print(f"Warning: Rerun visualizer disabled — {e}")
                print("  To enable: cd source/teleop && uv sync  (then run from host with display)")
        if self.rerun_logger:
            if self.legend_mode:
                mode = "legend (absolute position, dual-hand calibration)"
            else:
                mode = "absolute (closed-loop P)"
            self.rerun_logger.log_control_mode(mode)
        self._prev_vr_to_global_mat = None

    def get_local_ip(self):
        try:
            # same as bash command
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
            # get first IP (same as awk '{print $1}')
            ip = result.stdout.strip().split()[0]
            return ip
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError) as e:
            return None

    def _extract_pose_from_joint_state(self, state):
        names = list(state.name)
        positions = list(state.position)

        def get_positions(joint_names):
            out = []
            for n in joint_names:
                if n in names:
                    out.append(positions[names.index(n)])
                else:
                    return None
            return out

        return (
            get_positions(BODY_JOINT_NAMES),
            get_positions(HEAD_JOINT_NAMES),
            get_positions(LEFT_ARM_JOINT_NAMES),
            get_positions(RIGHT_ARM_JOINT_NAMES),
        )

    def setup_robot(self):
        self.robot_eef = "omnipicker"
        self.robot_name = self.robot_cfg.split(".")[0]
        if "omnipicker" in self.robot_cfg:
            self.robot_eef = "omnipicker"
        elif "120s" in self.robot_cfg:
            self.robot_eef = "120s"
        else:
            raise ValueError(f"Invalid robot_cfg {self.robot_cfg}")

    def setup_device(self):
        if self.device_type == "pico":
            self.device = PicoDevice(self.host_ip, self.port, self.robot_cfg)
        elif self.device_type == "meta_quest":
            self.device = MetaQuestDevice(self.host_ip, self.port, self.robot_cfg)
        else:
            raise ValueError(f"Unsupported device_type {self.device_type}")

    def reset_command(self):
        self.command_ = np.array([0.0, 0.0, 0.0])
        self.rotation_command = np.array([0.0, 0.0, 0.0])
        self.robot_command = np.array([0.0, 0.0, 0.0])
        self.robot_rotation_command = np.array([0.0, 0.0, 0.0])
        self.waist_command = np.array([0.0, 0.0])
        self.head_command = np.array([0.0, 0.0])

    def initialize(self):
        # To be optimized
        init_marker = False
        while not init_marker:
            init_marker = self.ros_utils.sim_ros_node.initialize()
            time.sleep(0.1)
            print("waiting for initialize")
        print("initialize success")

        self.reset_command()
        self.current_step = 0
        self.eval_interval = 30

        self.device.initialize()
        self.joint_cmd = {}

        self._load_robot_init_states()

        # Log initial arm_base_link frame to Rerun
        if self.rerun_logger:
            arm_part = self.ros_utils.sim_ros_node.parts[2]
            if arm_part["init"]:
                p = arm_part["pose"]
                self.rerun_logger.log_arm_base(
                    [p.position.x, p.position.y, p.position.z],
                    [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
                )

    def _load_robot_init_states(self):
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _source_dir = os.path.normpath(os.path.join(_this_dir, ".."))
        teleop_yaml_path = os.path.join(_source_dir, "geniesim", "config", "teleop.yaml")
        if not os.path.isfile(teleop_yaml_path):
            logger.warning(f"teleop.yaml not found: {teleop_yaml_path}, skip loading robot init states")
            return
        with open(teleop_yaml_path, "r", encoding="utf-8") as f:
            teleop_cfg = yaml.safe_load(f)
        sub_task_name = (teleop_cfg or {}).get("benchmark", {}).get("sub_task_name")
        if not sub_task_name:
            logger.warning("benchmark.sub_task_name not found in teleop.yaml, skip loading robot init states")
            return
        if _source_dir not in sys.path:
            sys.path.insert(0, _source_dir)
        try:
            from geniesim.benchmark.config.robot_init_states import TASK_INFO_DICT
        except Exception as e:
            logger.warning(f"failed to import TASK_INFO_DICT: {e}, skip loading robot init states")
            return
        task_states = TASK_INFO_DICT.get(sub_task_name)
        if not task_states:
            logger.warning(f"sub_task_name '{sub_task_name}' not in TASK_INFO_DICT, skip loading robot init states")
            return
        robot_state = task_states.get("G2_omnipicker")
        if not robot_state:
            logger.warning(
                f"G2_omnipicker not found for sub_task_name '{sub_task_name}', skip loading robot init states"
            )
            return
        self.robot_init_body_state = deepcopy(robot_state.get("body_state"))
        self.robot_init_head_state = deepcopy(robot_state.get("head_state"))
        self.robot_init_arm = deepcopy(robot_state.get("init_arm"))
        self.robot_init_hand = deepcopy(robot_state.get("init_hand"))
        self.waist_yaw = self.robot_init_body_state[0]
        self.waist_pitch = self.robot_init_body_state[2]
        logger.info(f"loaded robot init states for sub_task_name='{sub_task_name}' (G2_omnipicker)")

    def _parse_arm_control_legend(self):
        """Legend mode: absolute position control with dual-hand grip calibration."""
        xyzxyzw_l = self.input.get("left")
        xyzxyzw_r = self.input.get("right")
        on_l = self.input.get("l_on")
        on_r = self.input.get("r_on")

        both_grip = on_l and on_r and xyzxyzw_l and xyzxyzw_r

        if both_grip and not self.legend_calibrated:
            # --- Calibration: both grips held for 3 seconds ---
            if self._legend_grip_start is None:
                self._legend_grip_start = time.time()
                self._legend_calib_samples = []
                logger.info("[Legend] Dual grip detected, hold 3s to calibrate...")

            # Collect samples
            vr_l = np.array(xyzxyzw_l[0:3])
            vr_r = np.array(xyzxyzw_r[0:3])
            vr_quat_l = np.array(xyzxyzw_l[3:7])
            vr_quat_r = np.array(xyzxyzw_r[3:7])
            self.ros_utils.sim_ros_node.update_without_judge()
            ee_l_pos, ee_l_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking("left")
            ee_r_pos, ee_r_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking("right")

            if ee_l_pos is not None and ee_r_pos is not None:
                self._legend_calib_samples.append({
                    "vr_l": vr_l, "vr_r": vr_r,
                    "ee_l": np.array(ee_l_pos), "ee_r": np.array(ee_r_pos),
                })
                if len(self._legend_calib_samples) % 10 == 1:
                    logger.debug(
                        f"[Legend] Sample #{len(self._legend_calib_samples)}: "
                        f"vr_l={vr_l}, vr_r={vr_r}, "
                        f"ee_l={np.array(ee_l_pos)}, ee_r={np.array(ee_r_pos)}"
                    )
            else:
                logger.warning(
                    f"[Legend] TF not ready during calibration: "
                    f"ee_l={'OK' if ee_l_pos is not None else 'None'}, "
                    f"ee_r={'OK' if ee_r_pos is not None else 'None'}"
                )

            elapsed = time.time() - self._legend_grip_start

            # Rerun: show calibration progress
            if self.rerun_logger:
                self.rerun_logger.log_legend_progress(elapsed, self._LEGEND_CALIB_DURATION)
                self.rerun_logger.log_quest_hand(vr_l, vr_quat_l, "left", is_active=True)
                self.rerun_logger.log_quest_hand(vr_r, vr_quat_r, "right", is_active=True)
                if ee_l_pos is not None:
                    self.rerun_logger.log_current_ee(ee_l_pos, ee_l_quat, "left")
                if ee_r_pos is not None:
                    self.rerun_logger.log_current_ee(ee_r_pos, ee_r_quat, "right")

            if elapsed >= self._LEGEND_CALIB_DURATION and len(self._legend_calib_samples) > 0:
                # Average samples
                n = len(self._legend_calib_samples)
                avg_vr_l = np.mean([s["vr_l"] for s in self._legend_calib_samples], axis=0)
                avg_vr_r = np.mean([s["vr_r"] for s in self._legend_calib_samples], axis=0)
                avg_ee_l = np.mean([s["ee_l"] for s in self._legend_calib_samples], axis=0)
                avg_ee_r = np.mean([s["ee_r"] for s in self._legend_calib_samples], axis=0)

                vr_dist = np.linalg.norm(avg_vr_r - avg_vr_l)
                ee_dist = np.linalg.norm(avg_ee_r - avg_ee_l)
                if vr_dist < 0.05:
                    logger.warning("[Legend] VR hands too close, calibration failed")
                    self._legend_grip_start = None
                    self._legend_calib_samples = []
                    return

                # Calibrate: scale + position offset only (rotation uses fixed per-hand transform)
                self.legend_scale = ee_dist / vr_dist
                vr_mid = (avg_vr_l + avg_vr_r) / 2
                ee_mid = (avg_ee_l + avg_ee_r) / 2
                self.legend_offset = ee_mid - self.legend_scale * vr_mid

                self.legend_calibrated = True
                self._legend_grip_start = None
                self._legend_calib_samples = []
                self._legend_ema_pos = [None, None]

                # Detailed calibration log
                logger.info(
                    f"[Legend] ===== Calibrated! samples={n} =====\n"
                    f"  avg_vr_l={avg_vr_l}, avg_vr_r={avg_vr_r}\n"
                    f"  avg_ee_l={avg_ee_l}, avg_ee_r={avg_ee_r}\n"
                    f"  vr_dist={vr_dist:.4f}, ee_dist={ee_dist:.4f}\n"
                    f"  scale={self.legend_scale:.4f}\n"
                    f"  offset={self.legend_offset}"
                )
                # Verify: apply transform to calibration VR poses, should match EE
                verify_l = self.legend_scale * avg_vr_l + self.legend_offset
                verify_r = self.legend_scale * avg_vr_r + self.legend_offset
                logger.info(
                    f"[Legend] Verification (should match avg_ee):\n"
                    f"  mapped_vr_l={verify_l} vs ee_l={avg_ee_l} err={np.linalg.norm(verify_l - avg_ee_l):.4f}\n"
                    f"  mapped_vr_r={verify_r} vs ee_r={avg_ee_r} err={np.linalg.norm(verify_r - avg_ee_r):.4f}"
                )

                if self.rerun_logger:
                    self.rerun_logger.log_legend_calibration(
                        avg_vr_l, avg_vr_r, avg_ee_l, avg_ee_r,
                        self.legend_scale, self.legend_offset,
                    )
            return  # During calibration, don't control

        # --- Not both grips: reset calibration timer ---
        if self._legend_grip_start is not None:
            elapsed = time.time() - self._legend_grip_start
            if elapsed < self._LEGEND_CALIB_DURATION:
                logger.info(f"[Legend] Grip released after {elapsed:.1f}s, calibration cancelled")
            self._legend_grip_start = None
            self._legend_calib_samples = []

        if not self.legend_calibrated:
            return

        # --- Control: directly follow scaled VR pose ---
        if not hasattr(self, "_legend_ctrl_count"):
            self._legend_ctrl_count = 0
        sides = [
            (xyzxyzw_l, on_l, "left", 0),
            (xyzxyzw_r, on_r, "right", 1),
        ]
        for xyzxyzw, on, side, idx in sides:
            if not on or not xyzxyzw:
                # Trigger released: reset origins so next press re-captures
                self._legend_vr_origin[idx] = None
                self._legend_ee_origin[idx] = None
                self._legend_ema_pos[idx] = None
                continue

            vr_pos = np.array(xyzxyzw[0:3])

            # First frame with trigger on: record VR origin + current EE origin
            if self._legend_vr_origin[idx] is None:
                self._legend_vr_origin[idx] = vr_pos.copy()
                ee_pos, _ = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking(side)
                self._legend_ee_origin[idx] = np.array(ee_pos) if ee_pos is not None else None
                if self._legend_ee_origin[idx] is None:
                    logger.warning(f"[Legend] {side} trigger on but no EE pose, skipping")
                    continue
                logger.info(f"[Legend] {side} trigger on: vr_origin={vr_pos}, ee_origin={self._legend_ee_origin[idx]}")

            if self._legend_ee_origin[idx] is None:
                continue

            # Relative position: EE origin + scaled delta from VR origin
            delta = vr_pos - self._legend_vr_origin[idx]
            target_pos = self._legend_ee_origin[idx] + self.legend_scale * self.spatial_coeff * delta
            alpha = self._legend_ema_alpha
            if self._legend_ema_pos[idx] is None:
                self._legend_ema_pos[idx] = target_pos.copy()
            else:
                self._legend_ema_pos[idx] = alpha * target_pos + (1 - alpha) * self._legend_ema_pos[idx]
            smoothed_pos = self._legend_ema_pos[idx]

            # Rotation: fixed per-hand (no VR rotation tracking)
            fixed_quat = self._legend_fixed_quat[idx]

            # Send pose to IK
            pose = self.ros_utils.sim_ros_node.compute_target_pose(
                smoothed_pos.tolist(), fixed_quat.tolist(), side
            )
            if pose is not None:
                self.ee_pub[idx] = pose
            else:
                logger.warning(f"[Legend] compute_target_pose returned None for {side}")

            # Debug log every 30 frames
            if self._legend_ctrl_count % 30 == 0:
                ee_pos, ee_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking(side)
                ee_arr = np.array(ee_pos) if ee_pos is not None else None
                err = np.linalg.norm(smoothed_pos - ee_arr) if ee_arr is not None else -1
                logger.debug(
                    f"[Legend] {side} ctrl: vr={vr_pos} → smoothed={smoothed_pos}, "
                    f"ee={ee_arr}, pos_err={err:.4f}, rot=fixed"
                )

            # Rerun: VR hand, smoothed target, actual EE (for visual comparison only)
            if self.rerun_logger:
                self.rerun_logger.log_quest_hand(vr_pos, fixed_quat, side, is_active=True)
                self.rerun_logger.log_ee_target(smoothed_pos, fixed_quat, side)
                ee_pos, ee_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking(side)
                if ee_pos is not None:
                    self.rerun_logger.log_current_ee(ee_pos, ee_quat, side)

        self._legend_ctrl_count += 1
        self.last_on = [on_l, on_r]

    def parse_arm_control(self):
        if self.current_mode == "playback":
            return
        if self.iskeyboard:
            self.ee_pub = self.ros_utils.sim_ros_node.parse_keyboard_pose(self.keyboard_pos)
            self.iskeyboard = False
            return

        if self.legend_mode:
            self._parse_arm_control_legend()
            return

        xyzxyzw_l = self.input.get("left")
        xyzxyzw_r = self.input.get("right")
        on_l = self.input.get("l_on")
        on_r = self.input.get("r_on")
        extra_r = self.input.get("r_b")

        sides = [
            (xyzxyzw_l, on_l, "left", 0),
            (xyzxyzw_r, on_r and (not extra_r), "right", 1),
        ]

        for xyzxyzw, on, side, idx in sides:
            # grip released → clear origins for re-calibration on next press
            if not on:
                if self.last_on[idx]:
                    self.vr_origin_pos[idx] = None
                    self.vr_origin_quat[idx] = None
                    self.robot_origin_pos[idx] = None
                    self.robot_origin_quat[idx] = None
                continue
            if not xyzxyzw:
                continue

            vr_pos = np.array(xyzxyzw[0:3])
            vr_quat = np.array(xyzxyzw[3:7])

            # grip just pressed or origin not yet captured → record dual origins
            if not self.last_on[idx] or self.vr_origin_pos[idx] is None:
                self.ros_utils.sim_ros_node.update_without_judge()
                ee_pos, ee_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking(side)
                if ee_pos is None:
                    continue
                self.vr_origin_pos[idx] = vr_pos.copy()
                self.vr_origin_quat[idx] = vr_quat.copy()
                self.robot_origin_pos[idx] = np.array(ee_pos)
                self.robot_origin_quat[idx] = np.array(ee_quat)
                if self.rerun_logger:
                    self.rerun_logger.log_grip_origins(vr_pos, np.array(ee_pos), side)

            if self.vr_origin_pos[idx] is None:
                continue

            # --- Read current EE pose every frame (closed-loop key) ---
            ee_pos, ee_quat = self.ros_utils.sim_ros_node.get_ee_pose_nonblocking(side)
            if ee_pos is None:
                continue
            ee_pos = np.array(ee_pos)
            ee_quat = np.array(ee_quat)

            # --- Position: closed-loop P control ---
            vr_offset = self.spatial_coeff * (vr_pos - self.vr_origin_pos[idx])
            robot_offset = ee_pos - self.robot_origin_pos[idx]
            pos_error = vr_offset - robot_offset
            pos_step = self.pos_gain * pos_error / self.loop_hz  # gain * error * dt
            norm = np.linalg.norm(pos_step)
            if norm > self.max_pos_step:
                pos_step = pos_step * (self.max_pos_step / norm)
            target_pos = ee_pos + pos_step

            # --- Rotation: closed-loop P control ---
            vr_rot_offset = R.from_quat(self.vr_origin_quat[idx]).inv() * R.from_quat(vr_quat)
            robot_rot_offset = R.from_quat(self.robot_origin_quat[idx]).inv() * R.from_quat(ee_quat)
            rot_error = robot_rot_offset.inv() * vr_rot_offset
            euler_error = rot_error.as_euler("xyz")
            euler_step = self.rot_gain * euler_error / self.loop_hz
            euler_step = np.clip(euler_step, -self.max_rot_step, self.max_rot_step)
            target_rot = R.from_quat(ee_quat) * R.from_euler("xyz", euler_step)
            target_quat = target_rot.as_quat()

            # --- Convert to arm_base_link frame → send to IK ---
            pose = self.ros_utils.sim_ros_node.compute_target_pose(
                target_pos.tolist(), target_quat.tolist(), side
            )
            if pose is not None:
                self.ee_pub[idx] = pose

            # --- Rerun logging ---
            if self.rerun_logger:
                self.rerun_logger.log_quest_hand(vr_pos, vr_quat, side, is_active=True)
                self.rerun_logger.log_current_ee(ee_pos, ee_quat, side)
                # target_pos / target_quat are in base_link frame (before compute_target_pose)
                self.rerun_logger.log_ee_target(target_pos, target_quat, side)
                self.rerun_logger.log_delta(vr_offset, robot_offset, side)

        self.last_on = [on_l, on_r]

    def parse_eef_control(self):
        cmd_l, cmd_r = self.input.get("l_eef"), self.input.get("r_eef")
        if cmd_l is None or cmd_r is None:
            return
        output_l = 0.65 * cmd_l
        output_r = 0.65 * cmd_r
        self.ros_utils.set_joint_state(
            ["idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"],
            [output_l, output_r],
        )
        self.last_eef_control_on = [output_l, output_r]

    def parse_waist_control(self):
        if not (self.input.get("r_axisX") or self.input.get("r_axisY")):
            return
        print(f"waist control: {self.input.get('r_axisX')} {self.input.get('r_axisY')}")
        self.waist_control_count += 1
        if self.waist_control_count < 10:
            return
        self.waist_control_count = 0
        waist_angle_limit = 1.67
        self.waist_yaw -= self.input.get("r_axisX") * 0.1
        self.waist_pitch += self.input.get("r_axisY") * 0.1
        print(f"=====>waist control: {self.waist_yaw} {self.waist_pitch}")
        self.waist_yaw = max(-waist_angle_limit, min(waist_angle_limit, self.waist_yaw))
        self.waist_pitch = max(-waist_angle_limit, min(waist_angle_limit, self.waist_pitch))
        self.ros_utils.sim_ros_node.pub_waist_pose(self.robot_init_body_state, self.waist_yaw, self.waist_pitch)

    def parse_body_control(self):
        if not (self.input.get("l_axisX") or self.input.get("l_axisY")):
            return
        name = [
            "idx111_chassis_lwheel_front_joint1",
            "idx112_chassis_lwheel_front_joint2",
            "idx131_chassis_rwheel_front_joint1",
            "idx132_chassis_rwheel_front_joint2",
            "idx121_chassis_lwheel_rear_joint1",
            "idx122_chassis_lwheel_rear_joint2",
            "idx141_chassis_rwheel_rear_joint1",
            "idx142_chassis_rwheel_rear_joint2",
        ]
        position = [0.0] * 8
        velocity = [0.0] * 8
        wheel_yaw = self.input.get("l_axisX")
        wheel_velocity = self.input.get("l_axisY")
        position[0] = -wheel_yaw
        position[2] = -wheel_yaw
        position[4] = -wheel_yaw
        position[6] = -wheel_yaw
        velocity[1] = 2.0 * math.pi * wheel_velocity
        velocity[3] = 2.0 * math.pi * wheel_velocity
        velocity[5] = 2.0 * math.pi * wheel_velocity
        velocity[7] = 2.0 * math.pi * wheel_velocity
        self.ros_utils.set_joint_position_and_velocity(name, position, velocity)

    def send_command(self):
        if self.ee_pub[0] != None or self.ee_pub[1] != None:
            self.ros_utils.sim_ros_node.pub_mc(self.ee_pub)

    def on_playback(self):
        if self.device.extra_l():
            print("Pub sig")
            self.ros_utils.sim_ros_node.pub_playback(True)
            self.current_mode = "playback"
            state = self.ros_utils.sim_ros_node.get_joint_state()
            body_position, head_position, left_arm_position, right_arm_position = self._extract_pose_from_joint_state(
                state
            )
            self.ros_utils.sim_ros_node.pub_robot_pose(
                body_position, head_position, left_arm_position, right_arm_position
            )
        else:
            self.ros_utils.sim_ros_node.pub_playback(False)
            if self.current_mode == "playback":
                self.current_mode = "realtime"

    def is_start_recording(self):
        if self.is_recording == True:
            return
        if self.device.extra_r():
            self.ros_utils.sim_ros_node.pub_recording(True)
            self.is_recording = True

    def sub_keyboard_event(self):
        self.pressed_keys = set()

        def on_press(key):
            try:
                if key.char == "d":
                    self.waist_angle -= 0.01
                    self.iskeyboard = True
                elif key.char == "a":
                    self.waist_angle += 0.01
                    self.iskeyboard = True
            except AttributeError:
                pass

        def on_release(key):
            self.pressed_keys.discard(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def reset_whole_robot(self):
        left_reset = self.input.get("l_x")
        right_reset = False
        body_reset = self.input.get("r_a")

        left_arm_position = None
        right_arm_position = None
        body_position = None
        head_position = None

        if left_reset and self.robot_init_arm:
            left_arm_position = list(self.robot_init_arm[:7])
            right_arm_position = list(self.robot_init_arm[7:])
        # if right_reset and self.robot_init_arm:
        #     right_arm_position = list(self.robot_init_arm[7:])
        if body_reset:
            if self.robot_init_body_state is not None:
                body_position = list(reversed(self.robot_init_body_state))
            if self.robot_init_head_state is not None:
                head_position = list(self.robot_init_head_state)

        self.ros_utils.sim_ros_node.pub_robot_pose(body_position, head_position, left_arm_position, right_arm_position)

    def run(self):
        self.ros_utils.start_ros_node()
        self.initialize()
        # self.sub_keyboard_event()

        target_period = 1.0 / self.loop_hz
        while rclpy.ok():
            loop_start = time.time()

            self.input = self.device.update()
            self.count += 1
            if self.count <= 3 or self.count % 300 == 0:
                logger.info(f"[TeleOp] loop #{self.count}, input={'has_data' if self.input else 'empty'}")
            if self.input and self.count <= 5:
                logger.info(f"[TeleOp] input keys={list(self.input.keys())}, l_on={self.input.get('l_on')}, r_on={self.input.get('r_on')}")
            if self.input:
                self.is_start_recording()
                self.ee_pub = [None, None]
                self.parse_arm_control()
                self.parse_waist_control()
                self.send_command()
                self.parse_eef_control()
                self.parse_body_control()
                self.reset_whole_robot()
                self.on_playback()

            # --- Rerun: TF tree + quest base + tick (runs every frame, both modes) ---
            if self.rerun_logger:
                tf_frames = self.ros_utils.sim_ros_node.get_all_tf_frames()
                if tf_frames:
                    self.rerun_logger.log_tf_tree(tf_frames, TF_TREE_EDGES)
                # Quest base frame (log when vr_to_global_mat changes)
                current_mat = self.device.vr_to_global_mat
                if (
                    self._prev_vr_to_global_mat is None
                    or not np.allclose(current_mat, self._prev_vr_to_global_mat)
                ):
                    self.rerun_logger.log_quest_base(current_mat)
                    self._prev_vr_to_global_mat = current_mat.copy()
                self.rerun_logger.tick()

            elapsed = time.time() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        for process_pid in self.process_pid:
            os.kill(process_pid, signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    parser.add_argument("--host_ip", type=str, default="", help="Set vr host ip")
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    parser.add_argument("--robot_cfg", type=str, default="G2_omnipicker.json", help="Set robot config")
    parser.add_argument("--device_type", type=str, default="pico", help="Set device type (pico or meta_quest)")
    parser.add_argument("--spatial_coeff", type=float, default=1.0, help="VR to robot spatial scale")
    parser.add_argument("--pos_gain", type=float, default=5.0, help="Position P-gain")
    parser.add_argument("--rot_gain", type=float, default=2.0, help="Rotation P-gain")
    parser.add_argument("--max_pos_step", type=float, default=0.02, help="Max position step per frame (m)")
    parser.add_argument("--max_rot_step", type=float, default=0.1, help="Max rotation step per frame (rad)")
    parser.add_argument("--legend", action="store_true", help="Legend absolute position control with dual-hand grip calibration")
    parser.add_argument("--rerun", action="store_true", help="Launch Rerun visualizer for trajectory debugging")
    parser.add_argument("--rerun_trail", type=int, default=400, help="Trail length (frames) in Rerun (default 400)")
    # fmt: on
    args = parser.parse_args()

    teleop = TeleOp(args)
    try:
        teleop.run()
    except KeyboardInterrupt:
        teleop.ros_utils.stop_ros_node()
        pass


if __name__ == "__main__":
    main()
