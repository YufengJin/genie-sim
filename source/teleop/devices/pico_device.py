#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
from .teleop_device import TeleopDevice
from utils.vr_server import VRServer
import numpy as np
from utils.logger import logger
from scipy.spatial.transform import Rotation as R
import math


def scale_quat_to_euler(q, s):
    rotation = R.from_quat(q)
    rpy = rotation.as_euler("xzy", degrees=False)
    r, p, y = rpy
    if r >= 0:
        r_trans = math.pi - r
    else:
        r_trans = -math.pi - r
    p_trans = p
    y_trans = y
    rpy_trans = np.array([-r_trans, p_trans, y_trans])
    rpy_scaled = rpy_trans * s

    return rpy_scaled


def scale_quat(q, s):
    return q * s


def round_v(val):
    PRECISION = 3
    return [round(v, PRECISION) for v in val]


class PicoDevice(TeleopDevice):
    def __init__(self, host_ip=None, port=8080, robot_cfg="G2_omnipicker"):
        self.host_ip = host_ip
        self.port = port
        self.robot_cfg = robot_cfg
        self.output = {"left": None, "right": None}
        self.coef_pos = 1.0
        self.coef_rot = 1.0
        self.vr_to_global_mat = np.eye(4)
        self.reset_orientation = True
        self.skip_vr_calibration = False  # Legend mode: skip VR origin calibration

    def initialize(self):
        self.vr_server = VRServer(host=self.host_ip, port=self.port)

    def update(self, debug=False):
        self.content = self.vr_server.on_update()
        if self.content:
            self.pico_command = self.parse_pico_command(robot_type=self.robot_cfg)
            dpos_l = self.coef_pos * self.pico_command["l"]["position"]
            dpos_r = self.coef_pos * self.pico_command["r"]["position"]
            drot_l = scale_quat(self.pico_command["l"]["quaternion"], self.coef_rot)
            drot_r = scale_quat(self.pico_command["r"]["quaternion"], self.coef_rot)

            # Organize Output
            self.output["left"] = list(round_v(dpos_l)) + list(round_v(drot_l))
            self.output["right"] = list(round_v(dpos_r)) + list(round_v(drot_r))
            self.output["l_eef"] = self.pico_command["l"]["gripper"]
            self.output["r_eef"] = self.pico_command["r"]["gripper"]
            self.output["l_on"] = self.pico_command["l"]["On"]
            self.output["r_on"] = self.pico_command["r"]["On"]
            self.output["r_b"] = self.pico_command["r"]["resetbh"]
            self.output["r_a"] = self.pico_command["r"]["reset"]
            self.output["l_x"] = self.pico_command["l"]["reset"]

            self.output["r_axisX"] = self.pico_command["r"]["axisX"]
            self.output["r_axisY"] = self.pico_command["r"]["axisY"]

            self.output["l_axisX"] = self.pico_command["l"]["axisX"]
            self.output["l_axisY"] = self.pico_command["l"]["axisY"]

            if debug:
                logger.debug(f"[PICO]\n {self.output['left']},\n, {self.output['right']}")
            return self.output
        else:
            # logger.warning("[PICO] no command")
            return {}

    def reset(self):
        self.output = {"left": None, "right": None}

    def parse_pico_command(self, robot_type="G2_omnipicker"):
        """
        Args:
            content (list(dict)): parse pico control command to local frame
        """
        [l_sig, r_sig] = self.content
        ret = {"l": {}, "r": {}}

        # --- Orientation reset (Role-ROS2 style): joystick click starts,
        #     continuously updates vr_to_global_mat, grip press locks ---
        if not self.skip_vr_calibration and (l_sig["axisClick"] == "true" or r_sig["axisClick"] == "true"):
            self.reset_orientation = True
        if self.reset_orientation and not self.skip_vr_calibration:
            ref_pos = np.array([l_sig["position"]["x"], l_sig["position"]["y"], l_sig["position"]["z"]])
            ref_rot = R.from_quat(
                [l_sig["rotation"]["x"], l_sig["rotation"]["y"], l_sig["rotation"]["z"], l_sig["rotation"]["w"]]
            ).as_matrix()
            ref_pose = np.eye(4)
            ref_pose[:3, :3] = ref_rot
            ref_pose[:3, 3] = ref_pos
            self.vr_to_global_mat = np.linalg.inv(ref_pose)
            # grip press → lock
            if l_sig["handTrig"] > 0.8 or r_sig["handTrig"] > 0.8:
                self.reset_orientation = False

        # --- Left arm: apply 4×4 vr_to_global then axis reorder ---
        raw_pos_l = np.array([l_sig["position"]["x"], l_sig["position"]["y"], l_sig["position"]["z"]])
        l_original_quat = np.array(
            [l_sig["rotation"]["x"], l_sig["rotation"]["y"], l_sig["rotation"]["z"], l_sig["rotation"]["w"]]
        )
        l_original_rot = R.from_quat(l_original_quat).as_matrix()
        raw_pose_l = np.eye(4)
        raw_pose_l[:3, :3] = l_original_rot
        raw_pose_l[:3, 3] = raw_pos_l
        aligned_pose_l = self.vr_to_global_mat @ raw_pose_l
        aligned_pos_l = aligned_pose_l[:3, 3]
        aligned_rot_l = aligned_pose_l[:3, :3]

        l_transform_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        ret["l"]["position"] = l_transform_matrix @ aligned_pos_l
        l_new_matrix = l_transform_matrix @ aligned_rot_l @ l_transform_matrix.T
        ret["l"]["quaternion"] = R.from_matrix(l_new_matrix).as_quat()

        ret["l"]["axisMode"] = "reset" if l_sig["axisClick"] == "true" else "move"
        ret["l"]["axisX"] = l_sig["axisX"]
        ret["l"]["axisY"] = l_sig["axisY"]
        ret["l"]["gripper"] = 1 - l_sig["indexTrig"]
        ret["l"]["On"] = l_sig["handTrig"] > 0.8
        ret["l"]["reset"] = l_sig["keyOne"] == "true"
        ret["l"]["against"] = l_sig["keyTwo"] == "true"

        # --- Right arm: apply 4×4 vr_to_global then axis reorder ---
        raw_pos_r = np.array([r_sig["position"]["x"], r_sig["position"]["y"], r_sig["position"]["z"]])
        r_original_quat = np.array(
            [r_sig["rotation"]["x"], r_sig["rotation"]["y"], r_sig["rotation"]["z"], r_sig["rotation"]["w"]]
        )
        r_original_rot = R.from_quat(r_original_quat).as_matrix()
        raw_pose_r = np.eye(4)
        raw_pose_r[:3, :3] = r_original_rot
        raw_pose_r[:3, 3] = raw_pos_r
        aligned_pose_r = self.vr_to_global_mat @ raw_pose_r
        aligned_pos_r = aligned_pose_r[:3, 3]
        r_aligned_matrix = aligned_pose_r[:3, :3]

        # right-side control: keyTwo switches between arm and waist control
        if r_sig["keyTwo"] == "true":
            # waist control
            ret["r"]["position"] = np.array([aligned_pos_r[2], -aligned_pos_r[0], aligned_pos_r[1]])
            r_transform_matrix = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
            r_new_matrix = r_transform_matrix @ r_aligned_matrix @ r_transform_matrix.T
            r_new_rot = R.from_matrix(r_new_matrix)
            ret["r"]["quaternion"] = r_new_rot.as_quat()

            euler = r_new_rot.as_euler("xyz", degrees=False)  # [roll, pitch, yaw]
            euler = euler * 0.5
            eps = np.deg2rad(10.0)
            step = np.deg2rad(5.0)
            roll = 0.0 if abs(euler[0]) < eps else np.round(euler[0] / step) * step
            pitch = 0.0 if abs(euler[1]) < eps else euler[1]
            yaw = 0.0 if abs(euler[2]) < eps else euler[2]
            clamped_rot = R.from_euler("xyz", [roll, pitch, yaw])
            ret["r"]["quaternion"] = clamped_rot.as_quat()

            ret["r"]["position"] = ret["r"]["position"] * 0.5
        else:
            ret["r"]["position"] = np.array([aligned_pos_r[2], -aligned_pos_r[0], aligned_pos_r[1]])
            r_transform_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
            r_new_matrix = r_transform_matrix @ r_aligned_matrix @ r_transform_matrix.T
            ret["r"]["quaternion"] = R.from_matrix(r_new_matrix).as_quat()

        ret["r"]["axisX"] = r_sig["axisX"]
        ret["r"]["axisY"] = r_sig["axisY"]
        ret["r"]["axisMode"] = "head" if r_sig["axisClick"] == "true" else "waist"
        ret["r"]["gripper"] = 1 - r_sig["indexTrig"]
        ret["r"]["On"] = r_sig["handTrig"] > 0.8
        ret["r"]["reset"] = r_sig["keyOne"] == "true"
        ret["r"]["resetbh"] = r_sig["keyTwo"] == "true"

        return ret

    def is_on_l(self):
        return self.pico_command["l"]["On"]

    def is_on_r(self):
        return self.pico_command["r"]["On"]

    def reset_l(self):
        return self.pico_command["l"]["reset"]

    def reset_r(self):
        return self.pico_command["r"]["reset"]

    def extra_l(self):
        if hasattr(self, "pico_command"):
            return self.pico_command["l"]["against"]
        else:
            return False

    def extra_r(self):
        if hasattr(self, "pico_command"):
            return self.pico_command["r"]["resetbh"]
        else:
            return False
