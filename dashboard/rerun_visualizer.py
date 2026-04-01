# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Rerun visualizer for the Genie Sim dashboard container.

Subscribes to ROS 2 camera topics published by Isaac Sim and logs
RGB + depth images into the Rerun web viewer (~10 Hz).

Camera topics (must match Isaac Sim config):
  RGB:   /genie_sim/head_front_camera_rgb
         /genie_sim/left_camera_rgb
         /genie_sim/right_camera_rgb
  Depth: /genie_sim/head_front_Camera_depth   (32FC1, meters)
         /genie_sim/Left_Camera_depth
         /genie_sim/Right_Camera_depth

Usage (inside dashboard container, separate terminal from the FastAPI server):
    python3 -m dashboard.rerun_visualizer

Then open http://localhost:9090 in your browser to view the Rerun web viewer.

Environment variables:
  RERUN_WEB_PORT   — Rerun web viewer port (default: 9090)
  RERUN_LOG_HZ     — Log rate in Hz (default: 10)
"""

import argparse
import logging
import os
import threading
import time

import numpy as np

try:
    import rerun as rr

    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False

logger = logging.getLogger("dashboard.rerun_visualizer")

# --------------------------------------------------------------------------- #
# Camera topic mapping — RGB (RGBA8) and depth (32FC1, meters)
# --------------------------------------------------------------------------- #
_CAMERAS: dict = {
    "head": {
        "rgb_topic": "/genie_sim/head_front_camera_rgb",
        "depth_topic": "/genie_sim/head_front_camera_depth",
        "label": "Head Front Camera",
    },
    "left": {
        "rgb_topic": "/genie_sim/left_camera_rgb",
        "depth_topic": "/genie_sim/left_camera_depth",
        "label": "Left Gripper Camera",
    },
    "right": {
        "rgb_topic": "/genie_sim/right_camera_rgb",
        "depth_topic": "/genie_sim/right_camera_depth",
        "label": "Right Gripper Camera",
    },
}

_DEFAULT_WEB_PORT = int(os.environ.get("RERUN_WEB_PORT", "9090"))
_DEFAULT_GRPC_PORT = int(os.environ.get("RERUN_GRPC_PORT", "9876"))
_DEFAULT_LOG_HZ = float(os.environ.get("RERUN_LOG_HZ", "10"))


class RerunVisualizer:
    """ROS 2 → Rerun bridge for the dashboard container.

    Two background threads are started on `start()`:
      - ROS spin thread: calls rclpy.spin_once() in a loop
      - Log thread: pushes buffered images into Rerun at _log_hz

    Thread safety: image buffers are protected by a single lock.

    Rerun 0.31+ API:
      serve_grpc(grpc_port)       — data/recording server
      serve_web_viewer(web_port)  — browser-accessible web viewer
    """

    def __init__(
        self,
        web_port: int = _DEFAULT_WEB_PORT,
        grpc_port: int = _DEFAULT_GRPC_PORT,
        log_hz: float = _DEFAULT_LOG_HZ,
    ):
        if not _RERUN_AVAILABLE:
            raise ImportError("rerun-sdk not installed. Run: pip install rerun-sdk")

        self._web_port = web_port
        self._grpc_port = grpc_port
        self._interval = 1.0 / max(log_hz, 0.1)
        self._lock = threading.Lock()
        # Per-camera buffers: None = not received yet
        self._latest: dict = {name: {"rgb": None, "depth": None} for name in _CAMERAS}
        self._frame = 0
        self._node = None
        self._running = False
        self._warned: set = set()

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def start(self) -> None:
        """Initialize Rerun, subscribe to ROS topics, start background threads."""
        rr.init("genie_sim_dashboard", spawn=False)
        # serve_grpc: holds recording data; serve_web_viewer: serves the browser UI
        grpc_addr = rr.serve_grpc(grpc_port=self._grpc_port)
        rr.serve_web_viewer(web_port=self._web_port, open_browser=False, connect_to=grpc_addr)
        logger.info(f"[Rerun] gRPC server → {grpc_addr}")
        logger.info(f"[Rerun] Web viewer  → http://localhost:{self._web_port}")
        rr.log("status", rr.TextLog("Genie Sim Rerun Visualizer started", level=rr.TextLogLevel.INFO))

        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        try:
            rclpy.init()
        except RuntimeError:
            pass  # already initialized (e.g. shared process with dashboard)

        self._node = rclpy.create_node("geniesim_rerun_visualizer")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        from sensor_msgs.msg import Image as RosImage

        for cam_name, cam_cfg in _CAMERAS.items():
            self._node.create_subscription(
                RosImage,
                cam_cfg["rgb_topic"],
                lambda msg, n=cam_name: self._on_rgb(n, msg),
                qos,
            )
            self._node.create_subscription(
                RosImage,
                cam_cfg["depth_topic"],
                lambda msg, n=cam_name: self._on_depth(n, msg),
                qos,
            )
            self._node.get_logger().info(
                f"Subscribed: {cam_cfg['rgb_topic']} | {cam_cfg['depth_topic']}"
            )

        self._running = True

        threading.Thread(target=self._spin, name="rerun_ros_spin", daemon=True).start()
        threading.Thread(target=self._log_loop, name="rerun_log", daemon=True).start()
        threading.Thread(target=self._watchdog, name="rerun_watchdog", daemon=True).start()

        logger.info("[Rerun] Visualizer running — waiting for ROS data...")

    def stop(self) -> None:
        self._running = False
        if self._node:
            self._node.destroy_node()
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    # ---------------------------------------------------------------------- #
    # ROS callbacks
    # ---------------------------------------------------------------------- #

    def _watchdog(self) -> None:
        time.sleep(5.0)
        if not self._running:
            return
        for cam_name, cam_cfg in _CAMERAS.items():
            if self._latest[cam_name]["rgb"] is None:
                logger.warning(
                    f"[Rerun] No RGB data from '{cam_cfg['rgb_topic']}' after 5s. "
                    "Check topic is published and QoS matches."
                )
            if self._latest[cam_name]["depth"] is None:
                logger.warning(
                    f"[Rerun] No depth data from '{cam_cfg['depth_topic']}' after 5s."
                )

    def _spin(self) -> None:
        import rclpy

        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _on_rgb(self, cam_name: str, msg) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            rgb = arr[:, :, :3].copy()  # RGBA → RGB
        except Exception as e:
            key = f"rgb_err_{cam_name}"
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(f"RGB decode error ({cam_name}): {e}")
            return
        with self._lock:
            self._latest[cam_name]["rgb"] = rgb

    def _on_depth(self, cam_name: str, msg) -> None:
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            elif msg.encoding == "16UC1":
                # millimetres → metres
                depth = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(msg.height, msg.width)
                    .astype(np.float32)
                    / 1000.0
                )
            else:
                key = f"depth_enc_{cam_name}"
                if key not in self._warned:
                    self._warned.add(key)
                    logger.warning(f"Unsupported depth encoding '{msg.encoding}' ({cam_name})")
                return
        except Exception as e:
            key = f"depth_err_{cam_name}"
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(f"Depth decode error ({cam_name}): {e}")
            return
        with self._lock:
            self._latest[cam_name]["depth"] = depth

    # ---------------------------------------------------------------------- #
    # Rerun log loop
    # ---------------------------------------------------------------------- #

    def _log_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            self._log_frame()
            sleep = self._interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _log_frame(self) -> None:
        rr.set_time("frame", sequence=self._frame)
        self._frame += 1

        with self._lock:
            snapshot = {k: dict(v) for k, v in self._latest.items()}

        logged = 0
        for cam_name, data in snapshot.items():
            if data["rgb"] is not None:
                rr.log(f"cameras/{cam_name}/rgb", rr.Image(data["rgb"]))
                logged += 1
            if data["depth"] is not None:
                rr.log(f"cameras/{cam_name}/depth", rr.DepthImage(data["depth"], meter=1.0))
                logged += 1

        if self._frame % 100 == 0:
            logger.info(f"[Rerun] frame={self._frame} logged_streams={logged}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Genie Sim Rerun Visualizer")
    parser.add_argument(
        "--web-port",
        type=int,
        default=_DEFAULT_WEB_PORT,
        help=f"Rerun web viewer port (default: {_DEFAULT_WEB_PORT}, env: RERUN_WEB_PORT)",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=_DEFAULT_GRPC_PORT,
        help=f"Rerun gRPC data port (default: {_DEFAULT_GRPC_PORT}, env: RERUN_GRPC_PORT)",
    )
    parser.add_argument(
        "--log-hz",
        type=float,
        default=_DEFAULT_LOG_HZ,
        help=f"Log rate in Hz (default: {_DEFAULT_LOG_HZ}, env: RERUN_LOG_HZ)",
    )
    args = parser.parse_args()

    viz = RerunVisualizer(web_port=args.web_port, grpc_port=args.grpc_port, log_hz=args.log_hz)
    viz.start()

    print(f"\n[Rerun] Open http://localhost:{args.web_port} to view the Rerun web viewer.")
    print("[Rerun] Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[Rerun] Stopping...")
        viz.stop()


if __name__ == "__main__":
    main()
