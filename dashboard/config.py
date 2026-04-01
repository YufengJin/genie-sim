import os

DASHBOARD_CONFIG = {
    "host": os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
    "port": int(os.environ.get("DASHBOARD_PORT", "8200")),
    "jpeg_quality": int(os.environ.get("DASHBOARD_JPEG_QUALITY", "75")),
    "target_fps": int(os.environ.get("DASHBOARD_TARGET_FPS", "10")),
    "cameras": {
        "head_front_camera": {
            "topic": "genie_sim/head_front_camera_rgb",
            "label": "Head Front Camera",
        },
        "left_camera": {
            "topic": "genie_sim/left_camera_rgb",
            "label": "Left Gripper Camera",
        },
        "right_camera": {
            "topic": "genie_sim/right_camera_rgb",
            "label": "Right Gripper Camera",
        },
    },
    "joint_states_topic": "/joint_states",
    "yolo": {
        "model_path": os.environ.get("YOLO_MODEL_PATH", "yolov8x-worldv2"),
        "default_confidence": float(os.environ.get("YOLO_DEFAULT_CONF", "0.3")),
        "default_classes": [
            c.strip()
            for c in os.environ.get(
                "YOLO_DEFAULT_CLASSES",
                "milk,bottle,can,screws,tools",
            ).split(",")
            if c.strip()
        ],
    },
}
