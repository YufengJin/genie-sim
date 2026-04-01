FROM ros:jazzy-ros-base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libturbojpeg \
    ros-jazzy-rmw-cyclonedds-cpp \
    git \
    libgl1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dashboard/requirements.txt /app/requirements.txt

RUN uv pip install --system --break-system-packages --no-cache \
    "torch>=2.7.0" "torchvision" \
    --index-url https://download.pytorch.org/whl/cu128

RUN uv pip install --system --break-system-packages --no-cache -r /app/requirements.txt

COPY dashboard /app/dashboard
COPY dashboard/scripts /app/dashboard/scripts
COPY source/sam3_tracker /app/sam3_tracker
COPY source/chaser_brain /app/chaser_brain
COPY others/yolo-world/yolov8x-worldv2.pt /app/yolov8x-worldv2.pt

WORKDIR /app

ENV ROS_LOCALHOST_ONLY=1
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV YOLO_MODEL_PATH=/app/yolov8x-worldv2.pt

EXPOSE 8200 8765 9090

CMD ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && python3 -m dashboard"]
