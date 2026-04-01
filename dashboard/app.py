"""Genie Sim Dashboard -- FastAPI + WebSocket with camera streaming and fleet management.

Adapted from Chaser Fleet Command Center v3.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from .config import DASHBOARD_CONFIG

# Load .env from chaser_brain if available
_env_path = Path(__file__).resolve().parent.parent / "chaser_brain" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

openai_client: OpenAI | None = None
DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o")
if os.environ.get("OPENAI_API_KEY"):
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

logger = logging.getLogger("dashboard")

app = FastAPI(title="Genie Sim Dashboard")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Mode-dependent provider: online (ROS) vs offline (dataset replay)
# ---------------------------------------------------------------------------
_OFFLINE_MODE = os.environ.get("DASHBOARD_MODE", "").lower() == "offline"

from .yolo_detector import YoloDetector

_yolo_cfg = DASHBOARD_CONFIG["yolo"]
yolo_detector = YoloDetector(
    model_path=_yolo_cfg["model_path"],
    conf=_yolo_cfg["default_confidence"],
    default_classes=_yolo_cfg["default_classes"],
)
YOLO_HEAD_CAMERA = "head_front_camera"


def _notify_yolo_error(camera_name: str, message: str) -> None:
    logger.warning("YOLO error [%s]: %s", camera_name, message)
    _add_event("yolo_error", {"camera": camera_name, "message": message[:500]}, source="yolo")


if _OFFLINE_MODE:
    from .dataset_player import DatasetPlayer

    _fleet_config = os.environ.get(
        "FLEET_CONFIG", str(Path(__file__).parent / "fleet_config.yaml")
    )
    camera_sub = DatasetPlayer(
        fleet_config_path=_fleet_config,
        jpeg_quality=DASHBOARD_CONFIG["jpeg_quality"],
        target_fps=DASHBOARD_CONFIG["target_fps"],
        yolo_detector=yolo_detector,
        on_yolo_error=_notify_yolo_error,
    )
else:
    from .ros_subscriber import CameraSubscriber

    camera_sub = CameraSubscriber(
        DASHBOARD_CONFIG,
        yolo_detector=yolo_detector,
        on_yolo_error=_notify_yolo_error,
    )
connected_clients: set[WebSocket] = set()

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------
fleet_status: dict[str, Any] = {}
event_log: list[dict] = []
brain_status: dict[str, Any] = {
    "autonomous_running": False,
    "total_decisions": 0,
    "tokens": {"input": 0},
}

MAX_EVENT_LOG = 100

# Offline mode: success rate slowly climbs from 81% → 92-95%
_offline_success_rate: float = 0.81
_offline_success_target: float = round(random.uniform(0.92, 0.95), 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _normalize_yolo_classes(raw: Any) -> list[str]:
    """Coerce WebSocket `classes` to list[str] (supports JSON list or comma-separated string)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [c.strip() for c in raw.split(",") if c.strip()]
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif x is not None:
                s = str(x).strip()
                if s:
                    out.append(s)
        return out
    return []


# ---------------------------------------------------------------------------
# Fake fleet simulation
# ---------------------------------------------------------------------------

_FAKE_ROBOT_COUNT = 10
_FAKE_TASK_NAMES = [
    "pick_and_place", "sort_objects", "stack_blocks", "grasp_targets",
    "organize_items", "select_color", "size_recognize", "shelf_arrange",
    "bin_packing", "object_handover",
]
_fake_robots: dict[str, dict] = {}
_fake_total_episodes: int = 0
_fake_success_rate: float = 0.56
_fake_tick: int = 0
_fake_stopped_count: int = 0


def _init_fake_fleet() -> None:
    global _fake_robots, _fake_total_episodes, _fake_success_rate, _fake_tick, _fake_stopped_count
    _fake_total_episodes = 127
    _fake_success_rate = 0.56
    _fake_tick = 0
    _fake_stopped_count = 0
    for i in range(_FAKE_ROBOT_COUNT):
        rid = f"robot_{i:02d}"
        _fake_robots[rid] = {
            "phase": "ACTIVE",
            "state": "running",
            "products_placed": random.randint(8, 25),
            "total_failures": random.randint(2, 10),
            "paused": False,
            "target_product": _FAKE_TASK_NAMES[i],
            "joint_angles": [round(random.uniform(-1.5, 1.5), 2) for _ in range(7)],
            "joint_names": [f"joint_{j}" for j in range(7)],
            "joint_velocities": [0.0] * 7,
            "gripper_open": random.choice([0.0, 0.04]),
        }


def _update_fake_fleet() -> None:
    global _fake_total_episodes, _fake_success_rate, _fake_tick, _fake_stopped_count
    _fake_tick += 1

    # Every 5 ticks (~5s), increase episodes
    if _fake_tick % 5 == 0:
        _fake_total_episodes += random.randint(1, 3)

    # Slowly increase success rate (cap at ~92%)
    if _fake_success_rate < 0.92:
        _fake_success_rate += random.uniform(0.0001, 0.0005)
        _fake_success_rate = min(_fake_success_rate, 0.92)

    for rid, r in _fake_robots.items():
        # Skip already stopped/idle robots
        if r["phase"] in ("IDLE", "STOPPED"):
            continue

        # Small chance to go Idle or Stop (max 2 non-active total, irreversible)
        if _fake_stopped_count < 2 and _fake_tick > 10 and random.random() < 0.02:
            r["phase"] = random.choice(["IDLE", "STOPPED"])
            r["state"] = "idle" if r["phase"] == "IDLE" else "stopped"
            _fake_stopped_count += 1
            continue

        # Active robots: slowly accumulate stats
        if random.random() < 0.15:
            if random.random() < _fake_success_rate:
                r["products_placed"] += 1
            else:
                r["total_failures"] += 1

        # Jitter joint angles slightly for liveliness
        r["joint_angles"] = [round(a + random.uniform(-0.02, 0.02), 2) for a in r["joint_angles"]]


_init_fake_fleet()


def _build_status() -> dict:
    """Build a status snapshot with simulated fleet data."""
    if _OFFLINE_MODE:
        global _offline_success_rate
        # Slowly climb from 81% to 92-95%
        if _offline_success_rate < _offline_success_target:
            _offline_success_rate += random.uniform(0.0002, 0.0008)
            _offline_success_rate = min(_offline_success_rate, _offline_success_target)
        fleet_info = camera_sub.get_fleet_info()
        # Token throughput: random 200-5000 up/down
        tk_up = random.randint(200, 5000)
        tk_down = random.randint(200, 5000)
        return {
            "fleet": fleet_info,
            "total_episodes": sum(
                r.get("replay_info", {}).get("total_episodes", 0)
                for r in fleet_info.values()
            ),
            "policy_version": "π-next",
            "buffer_stats": {"success_rate": _offline_success_rate},
            "eval_scores": {},
            "brain": {
                "autonomous_running": True,
                "tokens_up": tk_up,
                "tokens_down": tk_down,
            },
        }

    _update_fake_fleet()

    # Merge real joint state into robot_00 if available
    joint_state = camera_sub.get_joint_state()
    if joint_state and "robot_00" in _fake_robots:
        _fake_robots["robot_00"]["joint_angles"] = list(joint_state.get("positions", []))
        _fake_robots["robot_00"]["joint_names"] = list(joint_state.get("names", []))
        _fake_robots["robot_00"]["joint_velocities"] = list(joint_state.get("velocities", []))

    return {
        "fleet": {rid: dict(r) for rid, r in _fake_robots.items()},
        "total_episodes": _fake_total_episodes,
        "policy_version": "v2.1",
        "buffer_stats": {"success_rate": _fake_success_rate},
        "eval_scores": {},
        "brain": brain_status,
    }


def _add_event(event_type: str, data: dict | None = None, source: str = "dashboard") -> dict:
    """Record an event and schedule its broadcast to all connected clients."""
    event = {
        "type": "event",
        "event_type": event_type,
        "data": _sanitize(data or {}),
        "timestamp": time.time(),
        "source": source,
    }
    event_log.append(event)
    if len(event_log) > MAX_EVENT_LOG:
        event_log[:] = event_log[-MAX_EVENT_LOG:]
    # Schedule async broadcast (fire-and-forget from sync context)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_event(event))
    except RuntimeError:
        pass  # no event loop running yet
    return event


async def _broadcast_event(event: dict) -> None:
    """Send an event dict to all connected WebSocket clients."""
    payload = json.dumps(event)
    dead: list[WebSocket] = []
    for ws in list(connected_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.discard(ws)


def _pack_frame(camera_name: str, jpeg_bytes: bytes) -> bytes:
    """Pack binary WebSocket frame: [1B name_len][name][JPEG bytes]."""
    name_bytes = camera_name.encode("ascii")
    return bytes([len(name_bytes)]) + name_bytes + jpeg_bytes


# ---------------------------------------------------------------------------
# Broadcast loops
# ---------------------------------------------------------------------------


async def _status_broadcast_loop() -> None:
    """Broadcast fleet status + joint states every second."""
    while True:
        await asyncio.sleep(1.0)
        if not connected_clients:
            continue
        try:
            status = _build_status()
            payload = json.dumps({"type": "status", "data": _sanitize(status)})
            dead: list[WebSocket] = []
            for ws in list(connected_clients):
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                connected_clients.discard(ws)
        except Exception as e:
            logger.debug("Status broadcast error: %s", e)


async def frame_broadcast_loop() -> None:
    """Stream camera frames as binary WebSocket messages at target FPS."""
    interval = 1.0 / DASHBOARD_CONFIG["target_fps"]
    cameras = list(DASHBOARD_CONFIG["cameras"].keys())

    while True:
        if connected_clients:
            for cam_name in cameras:
                jpeg = camera_sub.get_latest_frame(cam_name)
                if jpeg is None:
                    continue
                frame = _pack_frame(cam_name, jpeg)
                dead: list[WebSocket] = []
                for ws in list(connected_clients):
                    try:
                        await ws.send_bytes(frame)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    connected_clients.discard(ws)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    logging.getLogger("dashboard.yolo_detector").setLevel(logging.INFO)
    if _OFFLINE_MODE:
        logging.getLogger("dashboard.dataset_player").setLevel(logging.INFO)
    else:
        logging.getLogger("dashboard.ros_subscriber").setLevel(logging.INFO)
    camera_sub.start()
    asyncio.create_task(_status_broadcast_loop())
    asyncio.create_task(frame_broadcast_loop())
    logger.info("Dashboard started on %s:%d", DASHBOARD_CONFIG["host"], DASHBOARD_CONFIG["port"])


@app.on_event("shutdown")
async def shutdown() -> None:
    camera_sub.stop()


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(_sanitize(_build_status()))


@app.get("/api/logs")
async def api_logs(count: int = 50) -> JSONResponse:
    recent = event_log[-count:] if count > 0 else event_log
    return JSONResponse(_sanitize(recent))


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    connected_clients.add(ws)
    logger.info("Client connected (%d total)", len(connected_clients))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
            await _handle_ws_message(ws, msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        connected_clients.discard(ws)
        logger.info("Client disconnected (%d total)", len(connected_clients))


async def _handle_ws_message(ws: WebSocket, msg: dict) -> None:
    """Route incoming WebSocket messages by action type."""
    action = msg.get("action")
    robot_id = msg.get("robot_id", "robot_0")

    # ── Select Robot ────────────────────────────────────────────
    if action == "select_robot":
        if _OFFLINE_MODE and hasattr(camera_sub, "select_robot"):
            camera_sub.select_robot(robot_id)
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "select_robot",
            "data": {"robot_id": robot_id, "status": "selected"},
        }))

    # ── Takeover ────────────────────────────────────────────────
    elif action == "takeover":
        if _OFFLINE_MODE and hasattr(camera_sub, "takeover_robot"):
            camera_sub.takeover_robot(robot_id)
        _add_event("takeover", {"robot_id": robot_id}, source="operator")
        fleet_status.setdefault(robot_id, {})["mode"] = "teleop"
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "takeover",
            "data": {"robot_id": robot_id, "status": "takeover_active"},
        }))

    # ── Release ─────────────────────────────────────────────────
    elif action == "release":
        _add_event("release", {"robot_id": robot_id}, source="operator")
        fleet_status.setdefault(robot_id, {})["mode"] = "autonomous"
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "release",
            "data": {"robot_id": robot_id, "status": "released"},
        }))

    # ── Joint Delta (keyboard teleop) ───────────────────────────
    elif action == "joint_delta":
        deltas = msg.get("deltas", [])
        gripper = msg.get("gripper")
        logger.debug("joint_delta robot=%s deltas=%s gripper=%s", robot_id, deltas, gripper)
        # TODO: publish joint deltas to ROS topic when rclpy publisher is added
        _add_event("joint_delta", {"robot_id": robot_id, "deltas": deltas, "gripper": gripper}, source="operator")
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "joint_delta",
            "data": {"status": "received"},
        }))

    # ── Brain Chat ──────────────────────────────────────────────
    elif action == "brain_chat":
        await _handle_brain_chat(ws, msg)

    # ── Brain Think (capture head camera → OpenAI) ─────────────
    elif action == "brain_think":
        await _handle_brain_think(ws)

    # ── Reset Robot ─────────────────────────────────────────────
    elif action == "reset_robot":
        _add_event("reset_robot", {"robot_id": robot_id}, source="operator")
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "reset_robot",
            "data": {"robot_id": robot_id, "status": "reset_requested"},
        }))

    # ── Emergency Stop ──────────────────────────────────────────
    elif action == "emergency_stop":
        _add_event("emergency_stop", {"robot_id": robot_id}, source="operator")
        logger.warning("EMERGENCY STOP requested for %s", robot_id)
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "emergency_stop",
            "data": {"robot_id": robot_id, "status": "stopped"},
        }))

    # ── Trigger Training ────────────────────────────────────────
    elif action == "trigger_training":
        _add_event("trigger_training", msg.get("params", {}), source="operator")
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "trigger_training",
            "data": {"status": "training_requested"},
        }))

    # ── Get Status ──────────────────────────────────────────────
    elif action == "get_status":
        status = _build_status()
        await ws.send_text(json.dumps({"type": "status", "data": _sanitize(status)}))

    # ── Teleop Mode ─────────────────────────────────────────────
    elif action == "teleop_mode":
        mode = msg.get("mode", "keyboard")
        _add_event("teleop_mode", {"robot_id": robot_id, "mode": mode}, source="operator")
        fleet_status.setdefault(robot_id, {})["teleop_mode"] = mode
        await ws.send_text(json.dumps({
            "type": "response",
            "action": "teleop_mode",
            "data": {"robot_id": robot_id, "mode": mode, "status": "set"},
        }))

    # ── YOLO Toggle (head camera only in UI; "all" maps to head-only) ─
    elif action == "yolo_toggle":
        camera = msg.get("camera", YOLO_HEAD_CAMERA)
        enabled = msg.get("enabled", False)
        if camera in ("all", "head", YOLO_HEAD_CAMERA):
            for cam_name in DASHBOARD_CONFIG["cameras"]:
                yolo_detector.set_enabled(cam_name, bool(enabled and cam_name == YOLO_HEAD_CAMERA))
        else:
            yolo_detector.set_enabled(camera, enabled)
        _add_event(
            "yolo_toggle",
            {"camera": camera, "enabled": enabled, "head_only": True},
            source="operator",
        )
        await ws.send_text(json.dumps({
            "type": "response", "action": "yolo_toggle",
            "data": yolo_detector.get_status(),
        }))

    # ── YOLO Set Classes ─────────────────────────────────────────
    elif action == "yolo_set_classes":
        classes = _normalize_yolo_classes(msg.get("classes", []))
        if classes:
            yolo_detector.set_classes(classes)
            _add_event("yolo_set_classes", {"classes": classes}, source="operator")
        merged = dict(_sanitize(yolo_detector.get_status()))
        merged["classes_requested"] = classes
        await ws.send_text(json.dumps({
            "type": "response", "action": "yolo_set_classes",
            "data": merged,
        }))

    # ── YOLO Set Confidence ──────────────────────────────────────
    elif action == "yolo_set_confidence":
        conf = float(msg.get("confidence", 0.3))
        yolo_detector.set_confidence(conf)
        merged = dict(_sanitize(yolo_detector.get_status()))
        await ws.send_text(json.dumps({
            "type": "response", "action": "yolo_set_confidence",
            "data": merged,
        }))

    # ── YOLO Status ──────────────────────────────────────────────
    elif action == "yolo_status":
        await ws.send_text(json.dumps({
            "type": "response", "action": "yolo_status",
            "data": yolo_detector.get_status(),
        }))

    # ── Unknown Action ──────────────────────────────────────────
    else:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown action: {action}",
        }))


# ---------------------------------------------------------------------------
# Brain Chat / Think helpers
# ---------------------------------------------------------------------------


async def _stream_openai(ws: WebSocket, messages: list[dict]) -> None:
    """Call OpenAI chat completions with streaming and push chunks via WS."""
    if not openai_client:
        await ws.send_text(json.dumps({
            "type": "brain_chat", "event": "error",
            "content": "OpenAI not configured. Set OPENAI_API_KEY in chaser_brain/.env",
        }))
        await ws.send_text(json.dumps({"type": "brain_chat", "event": "done"}))
        return

    try:
        stream = openai_client.chat.completions.create(
            model=DEFAULT_MODEL, messages=messages, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                await ws.send_text(json.dumps({
                    "type": "brain_chat", "event": "text", "content": delta,
                }))
    except Exception as e:
        logger.exception("OpenAI error")
        await ws.send_text(json.dumps({
            "type": "brain_chat", "event": "error", "content": str(e),
        }))

    await ws.send_text(json.dumps({"type": "brain_chat", "event": "done"}))


async def _handle_brain_chat(ws: WebSocket, msg: dict) -> None:
    """Process a brain_chat message — text and optional image."""
    message = msg.get("message", "")
    image = msg.get("image")  # base64 string or None

    content: list[dict] = []
    if message:
        content.append({"type": "text", "text": message})
    if image:
        if not image.startswith("data:"):
            image = f"data:image/jpeg;base64,{image}"
        content.append({"type": "image_url", "image_url": {"url": image}})

    if not content:
        await ws.send_text(json.dumps({
            "type": "brain_chat", "event": "error", "content": "Empty message",
        }))
        await ws.send_text(json.dumps({"type": "brain_chat", "event": "done"}))
        return

    messages = [{"role": "user", "content": content}]
    await _stream_openai(ws, messages)


async def _handle_brain_think(ws: WebSocket) -> None:
    """Capture head camera frame and ask OpenAI to describe it."""
    # Try all camera names that contain 'head'
    jpeg = None
    for cam_name in DASHBOARD_CONFIG["cameras"]:
        if "head" in cam_name.lower():
            jpeg = camera_sub.peek_latest_frame(cam_name)
            if jpeg:
                break

    if not jpeg:
        await ws.send_text(json.dumps({
            "type": "brain_chat", "event": "error",
            "content": "No head camera frame available",
        }))
        await ws.send_text(json.dumps({"type": "brain_chat", "event": "done"}))
        return

    image_b64 = base64.b64encode(jpeg).decode()

    # Send preview to frontend so user sees the captured frame
    await ws.send_text(json.dumps({
        "type": "brain_chat", "event": "think_preview", "image": image_b64,
    }))

    # Compress image for GPT (max 512px wide, quality 60)
    from PIL import Image as PILImage
    import io

    pil_img = PILImage.open(io.BytesIO(jpeg))
    max_w = 512
    if pil_img.width > max_w:
        ratio = max_w / pil_img.width
        pil_img = pil_img.resize((max_w, int(pil_img.height * ratio)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=60)
    compressed_b64 = base64.b64encode(buf.getvalue()).decode()

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the picture."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{compressed_b64}"}},
        ],
    }]
    await _stream_openai(ws, messages)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import subprocess
    import sys

    if os.environ.get("LAUNCH_RERUN", "0").lower() in ("1", "true", "yes") and not _OFFLINE_MODE:
        log_path = "/tmp/rerun_viz.log"
        proc = subprocess.Popen(
            [sys.executable, "-m", "dashboard.rerun_visualizer"],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        logger.info(f"[Rerun] visualizer started (PID {proc.pid}) → http://localhost:9090")

    import uvicorn

    uvicorn.run(
        app,
        host=DASHBOARD_CONFIG["host"],
        port=DASHBOARD_CONFIG["port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
