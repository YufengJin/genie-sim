import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

try:
    from turbojpeg import TurboJPEG

    _tj = TurboJPEG()

    def _encode_jpeg(rgb_array: np.ndarray, quality: int) -> bytes:
        return _tj.encode(rgb_array, quality=quality)

except ImportError:
    from PIL import Image
    import io

    _tj = None

    def _encode_jpeg(rgb_array: np.ndarray, quality: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb_array).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


logger = logging.getLogger("dashboard.ros_subscriber")


class CameraSubscriber:
    """Subscribes to ROS 2 camera and joint_states topics.

    Camera images (RGBA8) are converted to JPEG and cached per camera.
    Joint states are cached in a thread-safe deque for dashboard consumption.
    """

    def __init__(
        self,
        config: dict,
        yolo_detector=None,
        on_yolo_error: Callable[[str, str], None] | None = None,
    ):
        self._config = config
        self._yolo = yolo_detector
        self._on_yolo_error = on_yolo_error
        self._jpeg_quality = config["jpeg_quality"]
        self._target_interval = 1.0 / config["target_fps"]
        self._latest_frame: dict[str, bytes | None] = {}
        self._new_frame: dict[str, bool] = {}
        self._last_capture_time: dict[str, float] = {}
        self._joint_state_queue: deque = deque(maxlen=1)
        self._node = None
        self._thread = None
        self._running = False
        self._data_received: dict[str, bool] = {}

        for cam_name in config["cameras"]:
            self._latest_frame[cam_name] = None
            self._new_frame[cam_name] = False
            self._last_capture_time[cam_name] = 0.0
            self._data_received[cam_name] = False

    def start(self):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        try:
            rclpy.init()
        except RuntimeError:
            pass  # already initialized

        # Log ROS_DOMAIN_ID for debugging connectivity issues
        domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
        logger.info(f"ROS_DOMAIN_ID={domain_id}")
        if domain_id != "0":
            logger.warning(
                f"ROS_DOMAIN_ID is set to {domain_id}. "
                "Ensure the simulator uses the same domain ID, or set ROS_DOMAIN_ID=0."
            )

        self._node = rclpy.create_node("geniesim_dashboard")

        # Use RELIABLE to match Isaac Sim's publisher QoS.
        # In CycloneDDS with shared memory transport (used inside the same container),
        # BEST_EFFORT subscribers do not receive data from RELIABLE publishers reliably.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        from sensor_msgs.msg import Image as RosImage
        from sensor_msgs.msg import JointState

        # Subscribe to camera topics
        for cam_name, cam_cfg in self._config["cameras"].items():
            self._node.create_subscription(
                RosImage,
                cam_cfg["topic"],
                lambda msg, name=cam_name: self._on_image(name, msg),
                qos,
            )
            self._node.get_logger().info(f"Subscribed to {cam_cfg['topic']}")

        # Subscribe to joint_states
        joint_topic = self._config.get("joint_states_topic", "/joint_states")
        self._node.create_subscription(
            JointState,
            joint_topic,
            self._on_joint_state,
            qos,
        )
        self._node.get_logger().info(f"Subscribed to {joint_topic}")

        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        # Start a watchdog to warn about missing data (possible QoS mismatch)
        self._watchdog_thread = threading.Thread(target=self._qos_watchdog, daemon=True)
        self._watchdog_thread.start()

    def _spin(self):
        import rclpy

        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def _qos_watchdog(self):
        """After 5 seconds, warn if no data has been received on any subscribed topic."""
        time.sleep(5.0)
        if not self._running:
            return

        # Check cameras
        for cam_name in self._config["cameras"]:
            if not self._data_received.get(cam_name, False):
                topic = self._config["cameras"][cam_name]["topic"]
                logger.warning(
                    f"No data received on '{topic}' after 5s. "
                    "Possible causes: topic not published, QoS mismatch "
                    "(try switching RELIABLE/BEST_EFFORT), or wrong ROS_DOMAIN_ID."
                )

        # Check joint_states
        if not self._data_received.get("joint_states", False):
            joint_topic = self._config.get("joint_states_topic", "/joint_states")
            logger.warning(
                f"No data received on '{joint_topic}' after 5s. "
                "Possible causes: topic not published, QoS mismatch, or wrong ROS_DOMAIN_ID."
            )

    def _on_image(self, camera_name: str, msg):
        self._data_received[camera_name] = True
        now = time.monotonic()
        if now - self._last_capture_time[camera_name] < self._target_interval:
            return
        self._last_capture_time[camera_name] = now

        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            # RGBA8 -> RGB
            rgb = arr[:, :, :3]
            # Downscale to max width 640
            h, w = rgb.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / w
                new_h, new_w = int(h * scale), max_w
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # YOLO detection (if enabled for this camera)
            if self._yolo and self._yolo.is_enabled(camera_name):
                try:
                    rgb = self._yolo.detect(rgb)
                except Exception as e:
                    logger.exception("YOLO detection error (%s)", camera_name)
                    if self._on_yolo_error:
                        try:
                            self._on_yolo_error(camera_name, str(e)[:500])
                        except Exception:
                            logger.debug("on_yolo_error callback failed", exc_info=True)
            # RGB→BGR for TurboJPEG (expects BGR pixel format)
            bgr = rgb[:, :, ::-1].copy()
            jpeg_bytes = _encode_jpeg(bgr, self._jpeg_quality)
            self._latest_frame[camera_name] = jpeg_bytes
            self._new_frame[camera_name] = True
        except Exception as e:
            if self._node:
                self._node.get_logger().warning(f"Frame encode error ({camera_name}): {e}")

    def _on_joint_state(self, msg):
        self._data_received["joint_states"] = True
        try:
            state: dict[str, Any] = {
                "names": list(msg.name),
                "positions": list(msg.position),
                "velocities": list(msg.velocity) if msg.velocity else [],
                "efforts": list(msg.effort) if msg.effort else [],
                "timestamp": time.time(),
            }
            self._joint_state_queue.append(state)
        except Exception as e:
            if self._node:
                self._node.get_logger().warning(f"Joint state parse error: {e}")

    def get_latest_frame(self, camera_name: str) -> bytes | None:
        """Return the latest JPEG frame if a new one is available (marks as consumed)."""
        if self._new_frame.get(camera_name):
            self._new_frame[camera_name] = False
            return self._latest_frame.get(camera_name)
        return None

    def peek_latest_frame(self, camera_name: str) -> bytes | None:
        """Return the latest JPEG frame without marking it as consumed."""
        return self._latest_frame.get(camera_name)

    def get_joint_state(self) -> dict[str, Any] | None:
        """Return the latest joint state dict, or None if no data received yet.

        Returns a dict with keys: names, positions, velocities, efforts, timestamp.
        """
        try:
            # Peek at the latest without removing it (so status broadcast can re-read)
            return self._joint_state_queue[-1]
        except IndexError:
            return None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._node:
            self._node.destroy_node()
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()
