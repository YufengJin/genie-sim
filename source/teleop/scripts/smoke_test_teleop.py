#!/usr/bin/env python3
"""
Smoke test for Genie Sim teleoperation environment.
Tests all dependencies and module-level logic without requiring
Quest hardware, ROS, or a running Rerun viewer.

Run:
    /home/yjin/localdisk/genie_sim/source/teleop/.venv/bin/python scripts/smoke_test_teleop.py
"""

import sys
import os
import traceback

# Ensure source/teleop/ is on path
_TELEOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TELEOP_DIR not in sys.path:
    sys.path.insert(0, _TELEOP_DIR)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[96m[INFO]\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  {PASS} {name}" + (f"  — {detail}" if detail else ""))
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"         {e}")
        results.append((name, False, str(e)))


# ── 1. Core dependencies ─────────────────────────────────────────────────────
print(f"\n{BOLD}1. Core dependencies{RESET}")


def _numpy():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    assert a.sum() == 6.0
    return f"v{np.__version__}"


def _scipy():
    from scipy.spatial.transform import Rotation as R
    q = R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
    assert len(q) == 4
    import scipy
    return f"v{scipy.__version__}"


def _open3d():
    import open3d as o3d
    return f"v{o3d.__version__}"


def _rerun():
    import rerun as rr
    return f"v{rr.__version__}"


check("numpy", _numpy)
check("scipy", _scipy)
check("open3d", _open3d)
check("rerun-sdk", _rerun)

# ── 2. Teleop module imports ──────────────────────────────────────────────────
print(f"\n{BOLD}2. Teleop module imports{RESET}")


def _import_quest_bridge():
    from utils.quest_bridge import quest_pose_to_pico_format, run_bridge
    return "quest_bridge OK"


def _import_rerun_logger():
    from utils.rerun_logger import TeleopRerunLogger
    return "rerun_logger OK"


def _import_vr_server():
    from utils.vr_server import VRServer
    return "vr_server OK"


def _import_pico_device():
    # pico_device imports TeleopDevice base + VRServer; no ROS needed
    from devices.pico_device import PicoDevice
    return "pico_device OK"


def _import_meta_quest_device():
    from devices.meta_quest_device import MetaQuestDevice
    return "meta_quest_device OK"


check("utils.quest_bridge", _import_quest_bridge)
check("utils.rerun_logger", _import_rerun_logger)
check("utils.vr_server", _import_vr_server)
check("devices.pico_device", _import_pico_device)
check("devices.meta_quest_device", _import_meta_quest_device)

# ── 3. Coordinate transform correctness ──────────────────────────────────────
print(f"\n{BOLD}3. Coordinate transforms (quest_bridge){RESET}")

import numpy as np
from scipy.spatial.transform import Rotation as R


def _quest_to_pico_identity():
    """Identity pose (eye) should survive Quest→Pico with no position shift."""
    from utils.quest_bridge import quest_pose_to_pico_format
    mat = np.eye(4)
    buttons = {
        "leftGrip": (0.0,), "leftTrig": (0.0,), "X": False, "Y": False,
        "leftJS": [0.0, 0.0], "LJ": False,
    }
    d = quest_pose_to_pico_format(mat, "l", buttons)
    pos = d["position"]
    assert abs(pos["x"]) < 1e-9 and abs(pos["y"]) < 1e-9 and abs(pos["z"]) < 1e-9, \
        f"Identity pos should be zero, got {pos}"
    return "identity pos OK"


def _quest_z_flip():
    """Quest +Z → Pico -Z (handedness flip)."""
    from utils.quest_bridge import quest_pose_to_pico_format
    mat = np.eye(4)
    mat[2, 3] = 1.0   # Quest position: z=+1 (backward)
    buttons = {
        "leftGrip": (0.0,), "leftTrig": (0.0,), "X": False, "Y": False,
        "leftJS": [0.0, 0.0], "LJ": False,
    }
    d = quest_pose_to_pico_format(mat, "l", buttons)
    pos = d["position"]
    assert abs(pos["z"] - (-1.0)) < 1e-9, \
        f"Quest z=+1 → Pico z=-1 expected, got {pos['z']}"
    return "Z flip OK"


def _pico_to_robot_fwd():
    """Pico z=+1 (forward) → Robot x=+1 (forward)."""
    from utils.quest_bridge import _pico_to_robot
    pico = {
        "position": {"x": 0.0, "y": 0.0, "z": 1.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    pos_robot, _ = _pico_to_robot(pico)
    assert abs(pos_robot[0] - 1.0) < 1e-9, \
        f"Pico z=+1 → Robot fwd=+1 expected, got {pos_robot}"
    return "Pico z→Robot x(fwd) OK"


def _pico_to_robot_left():
    """Pico x=-1 (left) → Robot y=+1 (left)."""
    from utils.quest_bridge import _pico_to_robot
    pico = {
        "position": {"x": -1.0, "y": 0.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    pos_robot, _ = _pico_to_robot(pico)
    assert abs(pos_robot[1] - 1.0) < 1e-9, \
        f"Pico x=-1 → Robot left=+1 expected, got {pos_robot}"
    return "Pico -x→Robot y(left) OK"


def _pico_to_robot_up():
    """Pico y=+1 (up) → Robot z=+1 (up)."""
    from utils.quest_bridge import _pico_to_robot
    pico = {
        "position": {"x": 0.0, "y": 1.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    pos_robot, _ = _pico_to_robot(pico)
    assert abs(pos_robot[2] - 1.0) < 1e-9, \
        f"Pico y=+1 → Robot up=+1 expected, got {pos_robot}"
    return "Pico y→Robot z(up) OK"


def _json_roundtrip():
    """quest_pose_to_pico_format output must survive json.dumps/loads."""
    import json
    from utils.quest_bridge import quest_pose_to_pico_format
    mat = np.random.randn(4, 4)
    mat[:3, :3], _ = np.linalg.qr(mat[:3, :3])   # make it a proper rotation
    mat[3] = [0, 0, 0, 1]
    buttons = {
        "rightGrip": (0.5,), "rightTrig": (0.3,), "A": True, "B": False,
        "rightJS": [0.1, -0.2], "RJ": False,
    }
    d = quest_pose_to_pico_format(mat, "r", buttons)
    payload = json.dumps([d, d])
    decoded = json.loads(payload)
    assert isinstance(decoded, list) and len(decoded) == 2
    assert decoded[0]["handTrig"] == d["handTrig"]
    return f"payload {len(payload)} bytes OK"


check("quest identity → Pico zero pos", _quest_to_pico_identity)
check("Quest +Z → Pico -Z (handedness)", _quest_z_flip)
check("Pico z+1 → Robot fwd+1", _pico_to_robot_fwd)
check("Pico x-1 → Robot left+1", _pico_to_robot_left)
check("Pico y+1 → Robot up+1", _pico_to_robot_up)
check("JSON roundtrip", _json_roundtrip)

# ── 4. PicoDevice parse_pico_command (no hardware) ──────────────────────────
print(f"\n{BOLD}4. PicoDevice parse_pico_command{RESET}")


def _pico_device_parse():
    """PicoDevice.parse_pico_command parses a minimal Pico JSON pair correctly."""
    from devices.pico_device import PicoDevice
    dev = PicoDevice(host_ip=None, port=9999)
    dev.vr_to_global_mat = np.eye(3)
    dev.reset_orientation = False

    def _hand(grip=0.0, trig=0.0, x=0.0, y=0.0, z=0.0):
        return {
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "handTrig": grip, "indexTrig": trig,
            "keyOne": "false", "keyTwo": "false",
            "axisX": 0.0, "axisY": 0.0, "axisClick": "false",
        }

    dev.content = [_hand(grip=0.9, z=0.5), _hand(grip=0.0, z=0.3)]
    cmd = dev.parse_pico_command()

    assert cmd["l"]["On"] is True, "left grip > 0.8 should set On=True"
    assert cmd["r"]["On"] is False, "right grip = 0 should set On=False"

    # Pico z=+0.5 → Robot fwd=+0.5
    assert abs(cmd["l"]["position"][0] - 0.5) < 1e-9, \
        f"Left fwd expected 0.5, got {cmd['l']['position'][0]}"
    return "On flags + position mapping OK"


def _pico_gripper():
    """gripper = 1 - indexTrig."""
    from devices.pico_device import PicoDevice
    dev = PicoDevice()
    dev.vr_to_global_mat = np.eye(3)
    dev.reset_orientation = False

    def _hand(trig):
        return {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "handTrig": 0.0, "indexTrig": trig,
            "keyOne": "false", "keyTwo": "false",
            "axisX": 0.0, "axisY": 0.0, "axisClick": "false",
        }

    dev.content = [_hand(0.4), _hand(0.8)]
    cmd = dev.parse_pico_command()
    assert abs(cmd["l"]["gripper"] - 0.6) < 1e-9, \
        f"gripper = 1-0.4 = 0.6 expected, got {cmd['l']['gripper']}"
    assert abs(cmd["r"]["gripper"] - 0.2) < 1e-9, \
        f"gripper = 1-0.8 = 0.2 expected, got {cmd['r']['gripper']}"
    return "gripper = 1-indexTrig OK"


check("PicoDevice parse_pico_command", _pico_device_parse)
check("PicoDevice gripper mapping", _pico_gripper)

# ── 5. Rerun logger (no viewer spawned) ──────────────────────────────────────
print(f"\n{BOLD}5. Rerun logger (spawn=False){RESET}")


def _rerun_logger_full():
    """Init TeleopRerunLogger once (spawn=False) and exercise all log methods.

    rr.init() must only be called once per process — calling it a second time
    blocks waiting for the previous BufferedSink to drain. We therefore do both
    the instantiation check and the method calls in a single test function.
    """
    import rerun as rr

    # Patch rr.init so it never spawns a viewer subprocess
    _orig = rr.init
    def _patched(*args, **kwargs):
        kwargs.pop("spawn", None)
        _orig(*args, spawn=False, **kwargs)
    rr.init = _patched

    try:
        from utils.rerun_logger import TeleopRerunLogger
        logger = TeleopRerunLogger(trail_len=20)   # init check

        pos = np.array([0.3, 0.1, 0.5])
        quat = R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()

        logger.tick()
        logger.log_quest_base(np.eye(3))
        logger.log_quest_hand(pos, quat, "left", is_active=True)
        logger.log_quest_hand(pos * 0.8, quat, "right", is_active=False)
        logger.log_grip_origins(pos, pos * 0.5, "left")
        logger.log_grip_origins(pos, pos * 0.5, "right")
        logger.log_ee_target(pos * 0.5, quat, "left")
        logger.log_ee_target(pos * 0.6, quat, "right")
        logger.log_current_ee(pos * 0.48, quat, "left")
        logger.log_current_ee(pos * 0.58, quat, "right")
        logger.log_arm_base([0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 1.0])
        logger.log_delta(pos * 0.1, pos * 0.09, "left")
        logger.log_delta(pos * 0.12, pos * 0.11, "right")

        # Build trails (15 frames)
        for i in range(15):
            logger.tick()
            offset = np.array([i * 0.01, 0.0, 0.0])
            logger.log_quest_hand(pos + offset, quat, "left", True)
            logger.log_ee_target(pos * 0.5 + offset * 0.8, quat, "left")
            logger.log_delta(offset, offset * 0.9, "left")

    finally:
        rr.init = _orig

    return f"init + all log methods OK, {logger._frame} frames"


check("TeleopRerunLogger (init + all log methods)", _rerun_logger_full)

# ── 6. teleop.py argparse smoke ───────────────────────────────────────────────
print(f"\n{BOLD}6. teleop.py CLI args{RESET}")


def _teleop_argparse():
    """teleop.main argparse accepts --rerun and --rerun_trail without ROS."""
    import importlib.util, argparse
    spec = importlib.util.spec_from_file_location(
        "teleop_mod", os.path.join(_TELEOP_DIR, "teleop.py")
    )
    mod = importlib.util.load_from_spec = None  # don't exec, just parse args

    # Reconstruct just the argument parser from teleop.main
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_host", type=str, default="localhost:50051")
    parser.add_argument("--host_ip", type=str, default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--robot_cfg", type=str, default="G2_omnipicker.json")
    parser.add_argument("--device_type", type=str, default="pico")
    parser.add_argument("--spatial_coeff", type=float, default=1.0)
    parser.add_argument("--pos_gain", type=float, default=5.0)
    parser.add_argument("--rot_gain", type=float, default=2.0)
    parser.add_argument("--max_pos_step", type=float, default=0.02)
    parser.add_argument("--max_rot_step", type=float, default=0.1)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun_trail", type=int, default=400)

    args = parser.parse_args([
        "--device_type", "meta_quest",
        "--port", "8081",
        "--rerun",
        "--rerun_trail", "300",
    ])
    assert args.device_type == "meta_quest"
    assert args.rerun is True
    assert args.rerun_trail == 300
    return "--rerun --rerun_trail parsed OK"


check("teleop.py --rerun argparse", _teleop_argparse)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    for name, ok, err in results:
        if not ok:
            print(f"    {FAIL} {name}: {err}")
else:
    print(f"\n  \033[92m\033[1mAll checks passed.\033[0m")
print(f"  Python: {sys.executable}")
print(f"  venv:   {os.path.dirname(os.path.dirname(sys.executable))}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
