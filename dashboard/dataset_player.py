"""Offline replay provider — reads LeRobot datasets and serves frames like CameraSubscriber.

Drop-in replacement for ``ros_subscriber.CameraSubscriber`` when
``DASHBOARD_MODE=offline``.  No ROS dependency required.

Includes monkey-patches for Python 3.12 + lerobot v2.1 compatibility:
  - torch.stack patch for HuggingFace datasets Column type
  - Pure PyAV video decoder replacing torchcodec/torchvision
"""

from __future__ import annotations

import io
import logging
import os
import random
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

logger = logging.getLogger("dashboard.dataset_player")

# ---------------------------------------------------------------------------
# JPEG encoding (same approach as ros_subscriber.py)
# ---------------------------------------------------------------------------

try:
    from turbojpeg import TurboJPEG, TJPF_RGB

    _tj = TurboJPEG()

    def _encode_jpeg(rgb_array: np.ndarray, quality: int) -> bytes:
        return _tj.encode(rgb_array, quality=quality, pixel_format=TJPF_RGB)

except ImportError:
    from PIL import Image

    _tj = None

    def _encode_jpeg(rgb_array: np.ndarray, quality: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb_array).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Monkey-patches for lerobot + Python 3.12 compatibility
# ---------------------------------------------------------------------------


def _install_lerobot_patches() -> None:
    """Apply compatibility patches before importing lerobot.

    1. torch.stack: newer HuggingFace `datasets` returns Column objects
       instead of list[Tensor], causing torch.stack to fail.
    2. Video decoder: replace torchcodec/torchvision with pure PyAV,
       since geniesim-dashboard container lacks CUDA runtime libs for torchcodec.
    """
    import av

    # Patch torch.stack to handle Column objects
    _original_stack = torch.stack

    def _safe_stack(tensors, *args, **kwargs):
        if not isinstance(tensors, (list, tuple)):
            return torch.tensor(list(tensors))
        return _original_stack(tensors, *args, **kwargs)

    torch.stack = _safe_stack

    # Patch video decoder to use pure PyAV with seeking (not full-file decode)
    def _decode_pyav(video_path, timestamps, tolerance_s, backend=None):
        video_path = str(video_path)
        container = av.open(video_path)
        stream = container.streams.video[0]
        time_base = float(stream.time_base)

        result = []
        for qt in timestamps:
            # Seek to nearest keyframe before target timestamp
            target_pts = int(qt / time_base) if time_base > 0 else 0
            container.seek(target_pts, stream=stream)
            best_frame = None
            best_diff = float("inf")
            for frame in container.decode(video=0):
                ts = float(frame.pts * time_base)
                diff = abs(ts - qt)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = frame
                # Stop once we've passed the target (no need to decode further)
                if ts > qt + tolerance_s:
                    break
            if best_frame is not None:
                img = best_frame.to_ndarray(format="rgb24")
                result.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

        container.close()
        if not result:
            raise RuntimeError(f"No frames decoded from {video_path}")
        return _original_stack(result)

    try:
        import lerobot.common.datasets.video_utils as vu
        vu.decode_video_frames = _decode_pyav
        logger.info("Patched lerobot video decoder to use PyAV")
    except ImportError:
        pass  # lerobot not yet installed


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

try:
    import rerun as rr
    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False

_RERUN_WEB_PORT = int(os.environ.get("RERUN_WEB_PORT", "9090"))
_RERUN_GRPC_PORT = int(os.environ.get("RERUN_GRPC_PORT", "9876"))

# Map dashboard camera names to Rerun entity paths (match rerun_visualizer.py)
_RERUN_CAM_MAP = {
    "head_front_camera": "head",
    "left_camera": "left",
    "right_camera": "right",
}

# Depth camera key suffixes (dataset feature keys)
_DEPTH_SUFFIXES = {
    "head": "observation.images.head_depth",
    "left": "observation.images.hand_left_depth",
    "right": "observation.images.hand_right_depth",
}


def _to_depth_hw(img: Any) -> np.ndarray | None:
    """Convert depth tensor to HW float32 array, or None if not available."""
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img[0]
    elif img.ndim == 3:
        img = img[:, :, 0]
    return img


# ---------------------------------------------------------------------------
# Image helpers (ported from lerobot_visualize_common.py)
# ---------------------------------------------------------------------------


def _to_hwc_uint8(img: Any) -> np.ndarray:
    """Convert a LeRobot image tensor/array to HWC uint8 numpy array."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img = np.asarray(img)
    # CHW → HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    # float [0,1] → uint8
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Per-robot player state
# ---------------------------------------------------------------------------


class _RobotPlayer:
    """Manages dataset loading and episode iteration for a single robot."""

    def __init__(self, robot_id: str, repo_id: str, task_name: str, camera_mapping: dict[str, str], max_episodes: int = 0):
        self.robot_id = robot_id
        self.repo_id = repo_id
        self.task_name = task_name
        self.camera_mapping = camera_mapping  # dashboard_cam_name → dataset feature key
        self._max_episodes = max_episodes  # 0 = no limit

        self.dataset = None  # loaded in load()
        self.episode_to_frames: dict[int, tuple[int, int]] = {}  # ep → (start, end) global idx
        self.episode_ids: list[int] = []
        self.current_episode: int = -1
        self.frame_start: int = 0
        self.frame_end: int = 0
        self.local_frame_pos: int = 0

    def load(self, dataset_cache: dict | None = None) -> None:
        """Load the LeRobot dataset and build episode index."""
        # Reuse cached dataset if same repo_id was already loaded
        if dataset_cache is not None and self.repo_id in dataset_cache:
            self.dataset = dataset_cache[self.repo_id]
            logger.info("Loading dataset for %s: %s (cached)", self.robot_id, self.repo_id)
        else:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # v2.x
            except ImportError:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v0.5+

            logger.info("Loading dataset for %s: %s", self.robot_id, self.repo_id)
            repo_path = Path(self.repo_id)
            if repo_path.is_dir():
                # Local dataset — root IS the dataset directory in this lerobot version
                self.dataset = LeRobotDataset(
                    repo_id=repo_path.name,
                    root=repo_path,
                    delta_timestamps=None,
                    image_transforms=None,
                )
            else:
                # HuggingFace Hub dataset
                self.dataset = LeRobotDataset(
                    self.repo_id,
                    delta_timestamps=None,
                    image_transforms=None,
                )
            if dataset_cache is not None:
                dataset_cache[self.repo_id] = self.dataset

        total = len(self.dataset)
        logger.info("  %s: %d frames loaded", self.robot_id, total)

        # Build episode index
        self._build_episode_index()

        # Limit to max_episodes if configured
        if self._max_episodes > 0 and len(self.episode_ids) > self._max_episodes:
            sampled = sorted(random.sample(self.episode_ids, self._max_episodes))
            self.episode_to_frames = {k: self.episode_to_frames[k] for k in sampled}
            self.episode_ids = sampled
            logger.info("  %s: limited to %d episodes", self.robot_id, len(self.episode_ids))

        if not self.episode_ids:
            logger.warning("  %s: no episodes found!", self.robot_id)
            return

        # Start on a random episode
        self._pick_random_episode()

    def _build_episode_index(self) -> None:
        """Build mapping from episode_index → (start_frame, end_frame)."""
        ds = self.dataset
        # Try LeRobot v2 metadata first
        if hasattr(ds, "episode_data_index") and ds.episode_data_index is not None:
            try:
                starts = ds.episode_data_index["from"]
                ends = ds.episode_data_index["to"]
                for ep_idx in range(len(starts)):
                    s = int(starts[ep_idx]) if hasattr(starts[ep_idx], "item") else int(starts[ep_idx])
                    e = int(ends[ep_idx]) if hasattr(ends[ep_idx], "item") else int(ends[ep_idx])
                    self.episode_to_frames[ep_idx] = (s, e)
                # Filter out empty episodes
                self.episode_to_frames = {k: v for k, v in self.episode_to_frames.items() if v[0] < v[1]}
                self.episode_ids = sorted(self.episode_to_frames.keys())
                logger.info("  %s: %d episodes (from metadata)", self.robot_id, len(self.episode_ids))
                return
            except Exception as exc:
                logger.warning("  episode_data_index parse failed, falling back to scan: %s", exc)

        # Fallback: scan the dataset (slow for large datasets)
        logger.info("  %s: scanning episode indices...", self.robot_id)
        ep_ranges: dict[int, list[int]] = {}
        for i in range(len(ds)):
            sample = ds[i]
            ep = sample.get("episode_index", 0)
            ep_val = int(ep.item()) if hasattr(ep, "item") else int(ep)
            if ep_val not in ep_ranges:
                ep_ranges[ep_val] = [i, i + 1]
            else:
                ep_ranges[ep_val][1] = i + 1
        for ep_val, (s, e) in ep_ranges.items():
            if s < e:  # skip empty episodes
                self.episode_to_frames[ep_val] = (s, e)
        self.episode_ids = sorted(self.episode_to_frames.keys())
        logger.info("  %s: %d episodes (from scan)", self.robot_id, len(self.episode_ids))

    def _pick_random_episode(self) -> None:
        """Select a random episode and start from a random frame within it."""
        if not self.episode_ids:
            return
        self.current_episode = random.choice(self.episode_ids)
        self.frame_start, self.frame_end = self.episode_to_frames[self.current_episode]
        ep_len = self.frame_end - self.frame_start
        self.local_frame_pos = random.randint(0, max(0, ep_len - 1))

    def advance(self) -> dict | None:
        """Advance one frame and return the sample dict, or None if not loaded."""
        if self.dataset is None or not self.episode_ids:
            return None

        global_idx = self.frame_start + self.local_frame_pos
        if global_idx >= self.frame_end:
            # Episode finished — pick a new random one
            self._pick_random_episode()
            global_idx = self.frame_start

        sample = self.dataset[global_idx]
        self.local_frame_pos += 1
        return sample

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ids)

    @property
    def episode_length(self) -> int:
        return self.frame_end - self.frame_start


# ---------------------------------------------------------------------------
# Main DatasetPlayer — drop-in for CameraSubscriber
# ---------------------------------------------------------------------------


class DatasetPlayer:
    """Replays LeRobot datasets, exposing the same interface as CameraSubscriber."""

    def __init__(
        self,
        fleet_config_path: str,
        jpeg_quality: int = 75,
        target_fps: int = 10,
        yolo_detector=None,
        on_yolo_error=None,
    ):
        self._jpeg_quality = jpeg_quality
        self._target_fps = target_fps
        self._yolo = yolo_detector
        self._on_yolo_error = on_yolo_error

        # Parse config
        cfg_path = Path(fleet_config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Fleet config not found: {cfg_path}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        self._target_fps = cfg.get("fps", target_fps)
        default_max_ep = cfg.get("max_episodes", 0)
        default_cam_map = cfg.get("default_camera_mapping", {
            "head_front_camera": "observation.images.top_head",
            "left_camera": "observation.images.hand_left",
            "right_camera": "observation.images.hand_right",
        })

        # Build per-robot players
        self._robots: dict[str, _RobotPlayer] = {}
        robots_cfg = cfg.get("robots", {})
        for rid, rcfg in robots_cfg.items():
            cam_map = rcfg.get("camera_mapping", default_cam_map)
            max_ep = rcfg.get("max_episodes", default_max_ep)
            player = _RobotPlayer(
                robot_id=rid,
                repo_id=rcfg["repo_id"],
                task_name=rcfg.get("task_name", "unknown"),
                camera_mapping=cam_map,
                max_episodes=max_ep,
            )
            self._robots[rid] = player

        if not self._robots:
            raise ValueError("No robots defined in fleet config")

        # Active robot (first in config)
        self._active_robot_id: str = next(iter(self._robots))
        self._lock = threading.Lock()

        # Randomly assign ~10% of robots as idle (at least 1)
        self._idle_robots: set[str] = set()
        all_rids = list(self._robots.keys())
        n_idle = max(1, round(len(all_rids) * 0.10))
        candidates = [r for r in all_rids if r != self._active_robot_id]
        for rid in random.sample(candidates, min(n_idle, len(candidates))):
            self._idle_robots.add(rid)

        # Frame cache (keyed by dashboard camera name)
        self._latest_frames: dict[str, bytes | None] = {}
        self._new_frame: dict[str, bool] = {}
        self._joint_state: dict[str, Any] | None = None

        # Rerun state
        self._rerun_enabled = _RERUN_AVAILABLE
        self._rerun_frame = 0

        # Thread control
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Load all datasets, init Rerun, and start the playback thread."""
        # Install lerobot compatibility patches before loading datasets
        _install_lerobot_patches()

        dataset_cache: dict = {}
        for rid, player in self._robots.items():
            try:
                player.load(dataset_cache=dataset_cache)
            except Exception:
                logger.exception("Failed to load dataset for %s (%s)", rid, player.repo_id)

        # Position idle robots at ~70% through their episode
        for rid in self._idle_robots:
            player = self._robots.get(rid)
            if player and player.dataset and player.episode_ids:
                ep_len = player.frame_end - player.frame_start
                player.local_frame_pos = int(ep_len * 0.70)

        # Initialize Rerun (offline mode — no ROS, we log directly)
        if self._rerun_enabled:
            # Pre-check ports to avoid Rust panic (unrecoverable) on bind failure
            for label, port in [("gRPC", _RERUN_GRPC_PORT), ("web", _RERUN_WEB_PORT)]:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        logger.warning("[Rerun] %s port %d in use, disabling Rerun", label, port)
                        self._rerun_enabled = False
                        break
        if self._rerun_enabled:
            try:
                rr.init("genie_sim_dashboard", spawn=False)
                grpc_addr = rr.serve_grpc(grpc_port=_RERUN_GRPC_PORT)
                rr.serve_web_viewer(web_port=_RERUN_WEB_PORT, open_browser=False, connect_to=grpc_addr)
                rr.log("status", rr.TextLog("Offline Replay Rerun Visualizer started", level=rr.TextLogLevel.INFO))
                logger.info("[Rerun] gRPC → %s | Web → http://localhost:%d", grpc_addr, _RERUN_WEB_PORT)
            except Exception:
                logger.exception("[Rerun] Failed to initialize")
                self._rerun_enabled = False

        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info(
            "DatasetPlayer started: %d robots, active=%s, fps=%d",
            len(self._robots), self._active_robot_id, self._target_fps,
        )

    def stop(self) -> None:
        """Stop the playback thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("DatasetPlayer stopped")

    def select_robot(self, robot_id: str) -> None:
        """Switch the active robot (thread-safe). Grabs an instant preview frame."""
        if robot_id not in self._robots:
            logger.warning("select_robot: unknown robot_id %s", robot_id)
            return
        with self._lock:
            if self._active_robot_id != robot_id:
                self._active_robot_id = robot_id
                # Clear cached frames and grab a preview immediately
                self._latest_frames.clear()
                self._new_frame.clear()
                logger.info("Switched to robot %s (dataset: %s, idle=%s)",
                            robot_id, self._robots[robot_id].repo_id,
                            robot_id in self._idle_robots)
        # Grab one preview frame outside the lock (don't advance for idle robots)
        self._grab_preview(robot_id)

    def takeover_robot(self, robot_id: str) -> None:
        """Resume an idle robot — remove from idle set and make it active."""
        if robot_id not in self._robots:
            return
        with self._lock:
            self._idle_robots.discard(robot_id)
            self._active_robot_id = robot_id
        logger.info("Takeover: robot %s resumed from idle", robot_id)

    def _grab_preview(self, robot_id: str) -> None:
        """Decode the current frame of *robot_id* and push it into the frame cache.

        For idle robots this peeks at the current position without advancing.
        For active robots this also peeks (the playback loop will advance normally).
        """
        player = self._robots.get(robot_id)
        if player is None or player.dataset is None or not player.episode_ids:
            return
        try:
            global_idx = player.frame_start + player.local_frame_pos
            if global_idx >= player.frame_end:
                global_idx = player.frame_start
            sample = player.dataset[global_idx]
            frames: dict[str, bytes] = {}
            for dash_cam, ds_key in player.camera_mapping.items():
                raw = sample.get(ds_key)
                if raw is None:
                    continue
                img = _to_hwc_uint8(raw)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                h, w = img.shape[:2]
                if h < 1 or w < 1:
                    continue
                max_w = 640
                if w > max_w:
                    scale = max_w / w
                    img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
                if self._yolo and self._yolo.is_enabled(dash_cam):
                    try:
                        img = self._yolo.detect(img)
                    except Exception:
                        pass
                frames[dash_cam] = _encode_jpeg(img, self._jpeg_quality)
            with self._lock:
                if self._active_robot_id == robot_id:
                    self._latest_frames.update(frames)
                    for cam in frames:
                        self._new_frame[cam] = True
        except Exception:
            logger.debug("Preview grab failed for %s", robot_id, exc_info=True)

    # -- CameraSubscriber-compatible interface ---------------------------------

    def get_latest_frame(self, camera_name: str) -> bytes | None:
        """Return the latest JPEG frame if new (marks as consumed)."""
        with self._lock:
            if self._new_frame.get(camera_name):
                self._new_frame[camera_name] = False
                return self._latest_frames.get(camera_name)
        return None

    def peek_latest_frame(self, camera_name: str) -> bytes | None:
        """Return the latest JPEG frame without consuming it."""
        with self._lock:
            return self._latest_frames.get(camera_name)

    def get_joint_state(self) -> dict[str, Any] | None:
        """Return the latest joint state dict."""
        with self._lock:
            return self._joint_state

    # -- Offline-specific interface --------------------------------------------

    def get_fleet_info(self) -> dict[str, dict]:
        """Return fleet status for all configured robots (for _build_status)."""
        fleet: dict[str, dict] = {}
        for rid, player in self._robots.items():
            state_data = []
            with self._lock:
                if rid == self._active_robot_id and self._joint_state:
                    state_data = self._joint_state.get("positions", [])
                is_idle = rid in self._idle_robots

            if is_idle:
                phase = "IDLE"
            else:
                phase = "ACTIVE"

            fleet[rid] = {
                "phase": phase,
                "state": "replaying" if rid == self._active_robot_id and not is_idle else "idle" if is_idle else "running",
                "products_placed": player.current_episode,
                "total_failures": 0,
                "paused": False,
                "target_product": player.task_name,
                "joint_angles": list(state_data)[:7] if state_data else [0.0] * 7,
                "joint_names": [f"joint_{j}" for j in range(7)],
                "joint_velocities": [0.0] * 7,
                "gripper_open": 0.04,
                "replay_info": {
                    "episode": player.current_episode,
                    "frame": player.local_frame_pos,
                    "total_frames": player.episode_length,
                    "total_episodes": player.num_episodes,
                    "repo_id": player.repo_id,
                },
            }
        return fleet

    # -- Playback thread -------------------------------------------------------

    def _playback_loop(self) -> None:
        """Background thread: advance frames at target FPS for the active robot."""
        interval = 1.0 / self._target_fps

        while self._running:
            t0 = time.monotonic()

            with self._lock:
                active_id = self._active_robot_id
                is_idle = active_id in self._idle_robots

            if is_idle:
                time.sleep(interval)
                continue

            player = self._robots.get(active_id)
            if player is None or player.dataset is None:
                time.sleep(interval)
                continue

            sample = player.advance()
            if sample is None:
                time.sleep(interval)
                continue

            # Decode camera images → RGB arrays (for YOLO/Rerun) and JPEG (for WebSocket)
            frames: dict[str, bytes] = {}
            rgb_arrays: dict[str, np.ndarray] = {}  # for Rerun logging
            for dash_cam, ds_key in player.camera_mapping.items():
                raw = sample.get(ds_key)
                if raw is None:
                    continue
                try:
                    img = _to_hwc_uint8(raw)
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    h, w = img.shape[:2]
                    if h < 1 or w < 1:
                        continue

                    # Keep full-res RGB for Rerun
                    rgb_arrays[dash_cam] = img.copy()

                    # Downscale for WebSocket streaming
                    max_w = 640
                    if w > max_w:
                        scale = max_w / w
                        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

                    # YOLO detection (if enabled for this camera)
                    if self._yolo and self._yolo.is_enabled(dash_cam):
                        try:
                            img = self._yolo.detect(img)
                        except Exception as e:
                            logger.debug("YOLO detection error (%s): %s", dash_cam, e)
                            if self._on_yolo_error:
                                try:
                                    self._on_yolo_error(dash_cam, str(e)[:500])
                                except Exception:
                                    pass

                    # Encode RGB → JPEG
                    frames[dash_cam] = _encode_jpeg(img, self._jpeg_quality)
                except Exception:
                    logger.debug("Frame decode error for %s/%s", active_id, ds_key, exc_info=True)

            # Extract joint state
            joint_state = self._extract_joint_state(sample)

            # Log to Rerun
            if self._rerun_enabled:
                self._log_rerun(sample, rgb_arrays, joint_state, player)

            # Update cache under lock
            with self._lock:
                # Only update if this is still the active robot
                if self._active_robot_id == active_id:
                    for cam_name, jpeg in frames.items():
                        self._latest_frames[cam_name] = jpeg
                        self._new_frame[cam_name] = True
                    if joint_state:
                        self._joint_state = joint_state

            # Sleep for remaining interval
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _log_rerun(
        self,
        sample: dict,
        rgb_arrays: dict[str, np.ndarray],
        joint_state: dict[str, Any] | None,
        player: _RobotPlayer,
    ) -> None:
        """Log current frame data to Rerun viewer."""
        try:
            rr.set_time("frame", sequence=self._rerun_frame)
            self._rerun_frame += 1

            # RGB cameras
            for dash_cam, rgb in rgb_arrays.items():
                cam_name = _RERUN_CAM_MAP.get(dash_cam, dash_cam)
                rr.log(f"cameras/{cam_name}/rgb", rr.Image(rgb))

            # Depth cameras
            for cam_name, ds_key in _DEPTH_SUFFIXES.items():
                raw_depth = sample.get(ds_key)
                if raw_depth is not None:
                    depth_hw = _to_depth_hw(raw_depth)
                    if depth_hw is not None:
                        rr.log(f"cameras/{cam_name}/depth", rr.DepthImage(depth_hw, meter=1.0))

            # Joint state
            if joint_state:
                positions = joint_state.get("positions", [])
                for i, pos in enumerate(positions):
                    rr.log(f"joints/joint_{i}", rr.Scalar(pos))

            # Episode info (every 30 frames to avoid spam)
            if self._rerun_frame % 30 == 0:
                rr.log(
                    "replay/info",
                    rr.TextLog(
                        f"robot={player.robot_id} ep={player.current_episode} "
                        f"frame={player.local_frame_pos}/{player.episode_length} "
                        f"task={player.task_name}",
                        level=rr.TextLogLevel.INFO,
                    ),
                )
        except Exception:
            logger.debug("Rerun log error", exc_info=True)

    def _decode_frame(self, raw: Any) -> bytes | None:
        """Convert a dataset image value to JPEG bytes."""
        img = _to_hwc_uint8(raw)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        if h < 1 or w < 1:
            return None
        max_w = 640
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        return _encode_jpeg(img, self._jpeg_quality)

    @staticmethod
    def _extract_joint_state(sample: dict) -> dict[str, Any] | None:
        """Extract joint state from a dataset sample."""
        state = sample.get("observation.state")
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        positions = np.asarray(state, dtype=np.float64).flatten().tolist()
        n = len(positions)
        return {
            "names": [f"joint_{j}" for j in range(n)],
            "positions": positions,
            "velocities": [],
            "efforts": [],
            "timestamp": time.time(),
        }
