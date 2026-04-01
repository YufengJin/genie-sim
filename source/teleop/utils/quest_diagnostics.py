#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Meta Quest Diagnostics Tool for Genie Sim Teleoperation.

Three-stage validation to minimize integration errors:

  Stage 1: Quest connection & raw data inspection
  Stage 2: Pico-format conversion validation (field types, value ranges)
  Stage 3: UDP loopback test (send + receive, simulates container-side VRServer)

Usage (run on host machine with Quest connected via USB):
    cd source/teleop
    uv run python utils/quest_diagnostics.py                  # all stages
    uv run python utils/quest_diagnostics.py --stage 1        # raw data only
    uv run python utils/quest_diagnostics.py --stage 2        # raw + conversion
    uv run python utils/quest_diagnostics.py --stage 3        # full loopback
    uv run python utils/quest_diagnostics.py --duration 30    # run for 30 seconds
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

# Ensure source/teleop/ is on sys.path so "from utils.xxx" works
# regardless of which directory the script is invoked from.
_TELEOP_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _TELEOP_DIR not in sys.path:
    sys.path.insert(0, _TELEOP_DIR)

import numpy as np

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def fail(msg):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def info(msg):
    print(f"  {CYAN}[INFO]{RESET} {msg}")


def header(title):
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}\n")


# ---------------------------------------------------------------------------
# Expected data contract (what Genie Sim's PicoDevice expects)
# ---------------------------------------------------------------------------
PICO_FIELDS = {
    "position": {"type": dict, "keys": {"x": float, "y": float, "z": float}},
    "rotation": {"type": dict, "keys": {"x": float, "y": float, "z": float, "w": float}},
    "handTrig": {"type": float, "min": 0.0, "max": 1.0},
    "indexTrig": {"type": float, "min": 0.0, "max": 1.0},
    "keyOne": {"type": str, "values": ["true", "false"]},
    "keyTwo": {"type": str, "values": ["true", "false"]},
    "axisX": {"type": float, "min": -1.0, "max": 1.0},
    "axisY": {"type": float, "min": -1.0, "max": 1.0},
    "axisClick": {"type": str, "values": ["true", "false"]},
}

# Expected button keys from oculus_reader
EXPECTED_BUTTONS_LEFT = ["LG", "leftTrig", "X", "Y", "leftJS", "LJ"]
EXPECTED_BUTTONS_RIGHT = ["RG", "rightTrig", "A", "B", "rightJS", "RJ"]


# ---------------------------------------------------------------------------
# Stage 1: Raw Quest data
# ---------------------------------------------------------------------------
def stage1_check_connection(reader, duration):
    """Verify Quest is connected and inspect raw data format."""
    header("Stage 1: Quest Connection & Raw Data")

    errors = 0
    info("Waiting for Quest controller data (move the controllers)...")

    deadline = time.time() + min(duration, 10)
    poses, buttons = {}, {}
    while time.time() < deadline:
        poses, buttons = reader.get_transformations_and_buttons()
        if poses:
            break
        time.sleep(0.05)

    if not poses:
        fail("No pose data received from Quest. Check USB/ADB connection.")
        return False

    ok("Quest connected - receiving pose data")

    # Check pose keys
    for side, label in [("l", "Left"), ("r", "Right")]:
        if side in poses:
            mat = np.asarray(poses[side])
            if mat.shape == (4, 4):
                ok(f"{label} controller pose: shape={mat.shape}")
                pos = mat[:3, 3]
                info(f"  Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
            else:
                fail(f"{label} controller pose: expected (4,4), got {mat.shape}")
                errors += 1
        else:
            warn(f"{label} controller not detected (key '{side}' missing)")

    # Check button keys
    info(f"Button keys received: {sorted(buttons.keys())}")
    for key in EXPECTED_BUTTONS_LEFT + EXPECTED_BUTTONS_RIGHT:
        if key in buttons:
            val = buttons[key]
            ok(f"Button '{key}': {val} (type={type(val).__name__})")
        else:
            warn(f"Button '{key}' not found")

    # Continuous monitoring
    info(f"Monitoring raw data for {duration}s (press Ctrl+C to stop)...")
    print()
    frame_count = 0
    t0 = time.time()
    last_print = 0
    try:
        while time.time() - t0 < duration:
            poses, buttons = reader.get_transformations_and_buttons()
            if poses:
                frame_count += 1
            now = time.time()
            if now - last_print >= 1.0:
                hz = frame_count / (now - t0) if now > t0 else 0
                # Show live position of right controller
                rpos = ""
                if "r" in poses:
                    p = np.asarray(poses["r"])[:3, 3]
                    rpos = f"  R_pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"
                # Show trigger values
                rt = buttons.get("rightTrig", [0])[0] if isinstance(buttons.get("rightTrig"), list) else 0
                rg = buttons.get("RG", False)
                sys.stdout.write(
                    f"\r  {CYAN}[LIVE]{RESET} {hz:.1f} Hz | frames={frame_count}"
                    f"{rpos} | RG={rg} rightTrig={rt:.2f}    "
                )
                sys.stdout.flush()
                last_print = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    print()

    elapsed = time.time() - t0
    avg_hz = frame_count / elapsed if elapsed > 0 else 0
    if avg_hz > 20:
        ok(f"Average rate: {avg_hz:.1f} Hz ({frame_count} frames in {elapsed:.1f}s)")
    elif avg_hz > 5:
        warn(f"Low rate: {avg_hz:.1f} Hz (expected >20 Hz)")
    else:
        fail(f"Very low rate: {avg_hz:.1f} Hz - check USB connection")
        errors += 1

    return errors == 0


# ---------------------------------------------------------------------------
# Stage 2: Pico format conversion
# ---------------------------------------------------------------------------
def validate_pico_dict(data, label):
    """Validate a single controller dict against the Pico format contract."""
    errors = 0
    for field, spec in PICO_FIELDS.items():
        if field not in data:
            fail(f"{label}: missing field '{field}'")
            errors += 1
            continue

        val = data[field]

        # Type check for nested dicts (position, rotation)
        if spec["type"] == dict:
            if not isinstance(val, dict):
                fail(f"{label}.{field}: expected dict, got {type(val).__name__}")
                errors += 1
                continue
            for k, expected_type in spec["keys"].items():
                if k not in val:
                    fail(f"{label}.{field}: missing key '{k}'")
                    errors += 1
                elif not isinstance(val[k], (int, float)):
                    fail(f"{label}.{field}.{k}: expected number, got {type(val[k]).__name__}")
                    errors += 1
        elif spec["type"] == str:
            if not isinstance(val, str):
                fail(f"{label}.{field}: expected str, got {type(val).__name__}")
                errors += 1
            elif "values" in spec and val not in spec["values"]:
                fail(f"{label}.{field}: '{val}' not in {spec['values']}")
                errors += 1
        elif spec["type"] == float:
            if not isinstance(val, (int, float)):
                fail(f"{label}.{field}: expected number, got {type(val).__name__}")
                errors += 1
            else:
                lo, hi = spec.get("min", -999), spec.get("max", 999)
                if not (lo - 0.01 <= val <= hi + 0.01):
                    warn(f"{label}.{field}: value {val:.4f} outside [{lo}, {hi}]")

    return errors


def stage2_check_conversion(reader, duration):
    """Validate that quest_bridge conversion produces valid Pico format."""
    header("Stage 2: Pico Format Conversion Validation")

    from utils.quest_bridge import quest_pose_to_pico_format

    info("Reading Quest data and converting to Pico format...")

    poses, buttons = {}, {}
    deadline = time.time() + 5
    while time.time() < deadline:
        poses, buttons = reader.get_transformations_and_buttons()
        if poses:
            break
        time.sleep(0.05)

    if not poses:
        fail("No data to convert")
        return False

    errors = 0
    for side, label in [("l", "Left"), ("r", "Right")]:
        if side not in poses:
            warn(f"{label} controller not available, skipping")
            continue

        mat = np.asarray(poses[side])
        pico_data = quest_pose_to_pico_format(mat, side, buttons)

        info(f"{label} controller converted output:")
        for k, v in pico_data.items():
            info(f"  {k}: {v}")

        e = validate_pico_dict(pico_data, label)
        if e == 0:
            ok(f"{label} controller: all fields valid")
        else:
            errors += e

    # JSON serialization test (must survive json.dumps/loads roundtrip)
    info("Testing JSON serialization roundtrip...")
    for side in ["l", "r"]:
        if side not in poses:
            continue
        mat = np.asarray(poses[side])
        pico_data = quest_pose_to_pico_format(mat, side, buttons)
        try:
            payload = json.dumps([pico_data, pico_data])
            decoded = json.loads(payload)
            assert isinstance(decoded, list) and len(decoded) == 2
            ok(f"JSON roundtrip: OK (payload size: {len(payload)} bytes)")
        except Exception as exc:
            fail(f"JSON roundtrip failed: {exc}")
            errors += 1

    # Continuous conversion check
    info(f"Running continuous conversion for {duration}s...")
    frame_count = 0
    conv_errors = 0
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            poses, buttons = reader.get_transformations_and_buttons()
            if not poses:
                time.sleep(0.02)
                continue
            for side in ["l", "r"]:
                if side in poses:
                    mat = np.asarray(poses[side])
                    pico_data = quest_pose_to_pico_format(mat, side, buttons)
                    e = validate_pico_dict(pico_data, side)
                    if e > 0:
                        conv_errors += e
            frame_count += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    if conv_errors == 0:
        ok(f"Continuous validation: {frame_count} frames, 0 errors")
    else:
        fail(f"Continuous validation: {frame_count} frames, {conv_errors} errors")

    return errors == 0 and conv_errors == 0


# ---------------------------------------------------------------------------
# Stage 3: UDP loopback
# ---------------------------------------------------------------------------
def stage3_udp_loopback(reader, duration):
    """Send converted data via UDP and verify it's receivable by VRServer logic."""
    header("Stage 3: UDP Loopback Test")

    from utils.quest_bridge import quest_pose_to_pico_format

    PORT = 18999  # ephemeral port for loopback test
    received = {"data": None, "count": 0, "errors": 0}

    # Receiver thread (simulates container-side VRServer)
    def receiver():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", PORT))
        sock.settimeout(1.0)
        while received.get("running", True):
            try:
                data, _ = sock.recvfrom(2048)
                message = data.decode("utf-8")
                _new_message = message.replace("False", "false")
                parsed = json.loads(_new_message)
                if isinstance(parsed, list) and len(parsed) == 2:
                    received["data"] = parsed
                    received["count"] += 1
                else:
                    received["errors"] += 1
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                received["errors"] += 1
        sock.close()

    received["running"] = True
    rx_thread = threading.Thread(target=receiver, daemon=True)
    rx_thread.start()
    info(f"Loopback receiver listening on 127.0.0.1:{PORT}")

    # Sender (simulates quest_bridge)
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    info("Sending converted Quest data via UDP...")

    send_count = 0
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            poses, buttons = reader.get_transformations_and_buttons()
            if not poses:
                time.sleep(0.02)
                continue

            left_mat = np.asarray(poses.get("l", np.eye(4)))
            right_mat = np.asarray(poses.get("r", np.eye(4)))
            left_data = quest_pose_to_pico_format(left_mat, "l", buttons)
            right_data = quest_pose_to_pico_format(right_mat, "r", buttons)
            payload = json.dumps([left_data, right_data])
            tx_sock.sendto(payload.encode("utf-8"), ("127.0.0.1", PORT))
            send_count += 1
            time.sleep(1.0 / 30)
    except KeyboardInterrupt:
        pass

    # Wait for receiver to catch up
    time.sleep(0.5)
    received["running"] = False
    rx_thread.join(timeout=2)
    tx_sock.close()

    info(f"Sent: {send_count} packets")
    info(f"Received: {received['count']} packets")
    info(f"Parse errors: {received['errors']}")

    if send_count == 0:
        fail("No packets sent (no Quest data)")
        return False

    loss = 1.0 - received["count"] / send_count if send_count > 0 else 1.0
    if loss < 0.05:
        ok(f"Packet loss: {loss * 100:.1f}%")
    elif loss < 0.2:
        warn(f"Packet loss: {loss * 100:.1f}%")
    else:
        fail(f"Packet loss: {loss * 100:.1f}%")

    if received["errors"] > 0:
        fail(f"{received['errors']} JSON parse errors on receiver side")
        return False

    # Validate last received packet
    if received["data"]:
        info("Last received packet (as container would see it):")
        for i, label in enumerate(["Left", "Right"]):
            info(f"  {label}: pos=({received['data'][i]['position']['x']:+.3f}, "
                 f"{received['data'][i]['position']['y']:+.3f}, "
                 f"{received['data'][i]['position']['z']:+.3f})")
        errors = 0
        errors += validate_pico_dict(received["data"][0], "rx_left")
        errors += validate_pico_dict(received["data"][1], "rx_right")
        if errors == 0:
            ok("Received data passes Pico format validation")
        else:
            fail(f"Received data has {errors} validation errors")
            return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Meta Quest diagnostics for Genie Sim teleoperation")
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1/2/3), 0=all (default)")
    parser.add_argument("--duration", type=int, default=10, help="Monitoring duration per stage in seconds (default 10)")
    args = parser.parse_args()

    header("Meta Quest Diagnostics for Genie Sim")
    info("Initializing OculusReader (ensure Quest is connected via USB)...")

    try:
        from oculus_reader.reader import OculusReader
        reader = OculusReader()
        ok("OculusReader initialized")
    except Exception as e:
        fail(f"Failed to initialize OculusReader: {e}")
        info("Ensure: 1) Quest connected via USB  2) ADB debugging enabled  3) oculus_reader installed")
        sys.exit(1)

    results = {}

    if args.stage in (0, 1):
        results["stage1"] = stage1_check_connection(reader, args.duration)

    if args.stage in (0, 2):
        results["stage2"] = stage2_check_conversion(reader, args.duration)

    if args.stage in (0, 3):
        results["stage3"] = stage3_udp_loopback(reader, args.duration)

    # Summary
    header("Summary")
    all_pass = True
    for stage, passed in results.items():
        if passed:
            ok(f"{stage}: PASSED")
        else:
            fail(f"{stage}: FAILED")
            all_pass = False

    if all_pass:
        print(f"\n  {GREEN}{BOLD}All checks passed! Ready to connect to Genie Sim.{RESET}")
        print(f"\n  Next steps:")
        print(f"    1. Start bridge:  cd source/teleop && uv run python utils/quest_bridge.py --target_ip <CONTAINER_IP> --target_port 8081")
        print(f"    2. In container:  python3 teleop.py --device_type meta_quest --port 8081")
    else:
        print(f"\n  {RED}{BOLD}Some checks failed. Fix issues above before connecting.{RESET}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
