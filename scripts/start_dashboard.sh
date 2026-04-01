#!/bin/bash
# Launch Genie Sim Camera Dashboard
# Connects to Isaac Sim camera ROS 2 topics and displays in browser at http://localhost:8200

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_DIR/dashboard"
DASHBOARD_PORT="${DASHBOARD_PORT:-8200}"
DASHBOARD_IMAGE="${DASHBOARD_IMAGE:-geniesim-dashboard:latest}"

MODE="${1:-docker}"
SIM_CONTAINER="${2:-genie_sim_benchmark}"

# Auto-kill anything occupying the dashboard port
_free_port() {
    local port="$1"
    # Stop any docker container bound to this port
    local cid
    cid=$(docker ps --filter "publish=${port}" -q 2>/dev/null)
    if [ -n "$cid" ]; then
        echo "[Dashboard] Stopping container using port ${port}..."
        docker stop "$cid" >/dev/null 2>&1 || true
    fi
    # Kill any local process on the port
    local pid
    pid=$(lsof -ti :"${port}" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[Dashboard] Killing process ${pid} on port ${port}..."
        kill "$pid" 2>/dev/null || true
        sleep 0.5
    fi
}

_free_port "$DASHBOARD_PORT"

case "$MODE" in
    sim)
        echo "[Dashboard] DEPRECATED: 'sim' mode is deprecated. Use 'docker' mode instead (now the default)."
        echo "[Dashboard] Cross-container DDS communication is now supported via CycloneDDS UDP."
        exit 1
        # --- Deprecated code below, kept for reference ---
        echo "[Dashboard] Starting inside Isaac Sim container '$SIM_CONTAINER' (http://localhost:8200)..."
        echo "[Dashboard] This avoids DDS shared memory isolation between containers."

        # Install dashboard dependencies inside the sim container (idempotent)
        docker exec --user root "$SIM_CONTAINER" bash -c "
            apt-get install -y -qq libturbojpeg 2>/dev/null
            pip3 install --break-system-packages --ignore-installed typing-extensions 2>/dev/null
            pip3 install --break-system-packages fastapi uvicorn websockets 'PyTurboJPEG>=1.7.0,<2.0.0' 2>/dev/null
        " || echo '[Dashboard] Warning: dependency install failed, continuing...'

        # Run dashboard inside the sim container
        docker exec -d "$SIM_CONTAINER" bash -c "
            source /opt/ros/jazzy/setup.bash 2>/dev/null || true
            export ROS_LOCALHOST_ONLY=1
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export CYCLONEDDS_URI=/geniesim/main/config/cyclonedds.xml
            export DASHBOARD_PORT=${DASHBOARD_PORT:-8200}
            export DASHBOARD_JPEG_QUALITY=${DASHBOARD_JPEG_QUALITY:-75}
            export DASHBOARD_TARGET_FPS=${DASHBOARD_TARGET_FPS:-10}
            cd /geniesim/main
            python3 -m dashboard
        "
        echo "[Dashboard] Started in background. Open http://localhost:8200"
        ;;

    docker)
        if [ "${DASHBOARD_BUILD:-0}" = "1" ]; then
            echo "[Dashboard] Building image ${DASHBOARD_IMAGE} (repo root)..."
            docker build -f "$DASHBOARD_DIR/Dockerfile" -t "${DASHBOARD_IMAGE}" "$PROJECT_DIR"
        elif ! docker image inspect "${DASHBOARD_IMAGE}" >/dev/null 2>&1; then
            echo "[Dashboard] Image '${DASHBOARD_IMAGE}' not found, building..."
            docker build -f "$DASHBOARD_DIR/Dockerfile" -t "${DASHBOARD_IMAGE}" "$PROJECT_DIR"
        else
            echo "[Dashboard] Using existing image '${DASHBOARD_IMAGE}' (set DASHBOARD_BUILD=1 to rebuild)."
        fi

        echo "[Dashboard] Starting container (http://localhost:${DASHBOARD_PORT})..."
        docker rm -f geniesim_dashboard 2>/dev/null || true
        docker run -d --name geniesim_dashboard \
            --network=host \
            -e ROS_LOCALHOST_ONLY=1 \
            -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
            -e CYCLONEDDS_URI=/geniesim/main/config/cyclonedds.xml \
            -e DASHBOARD_PORT="${DASHBOARD_PORT}" \
            -e DASHBOARD_JPEG_QUALITY="${DASHBOARD_JPEG_QUALITY:-75}" \
            -e DASHBOARD_TARGET_FPS="${DASHBOARD_TARGET_FPS:-10}" \
            -e LAUNCH_RERUN="${LAUNCH_RERUN:-0}" \
            -e RERUN_WEB_PORT="${RERUN_WEB_PORT:-9090}" \
            -v "$PROJECT_DIR/source/chaser_brain/.env:/app/chaser_brain/.env:ro" \
            -v "$PROJECT_DIR/config/cyclonedds.xml:/geniesim/main/config/cyclonedds.xml:ro" \
            "${DASHBOARD_IMAGE}"
        echo "[Dashboard] Started in background. Open http://localhost:${DASHBOARD_PORT}"
        ;;

    local)
        echo "[Dashboard] Starting locally (http://localhost:8200)..."
        echo "[Dashboard] Requires ROS 2 (Jazzy/Humble) sourced in current shell."

        export ROS_LOCALHOST_ONLY=1
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

        cd "$PROJECT_DIR"
        python3 -m dashboard
        ;;

    *)
        echo "Usage: $0 [docker|local|sim] [container_name]"
        echo "  docker  - Run in separate Docker container (default, recommended)"
        echo "  local   - Run directly (requires ROS 2 installed)"
        echo "  sim     - (DEPRECATED) Run inside Isaac Sim container"
        echo ""
        echo "  container_name - Isaac Sim container name (only for deprecated 'sim' mode)"
        echo ""
        echo "Docker mode env:"
        echo "  DASHBOARD_IMAGE   - Image tag (default: geniesim-dashboard:latest)"
        echo "  DASHBOARD_BUILD=1 - Force rebuild before run"
        echo "  (If image is missing, build runs once automatically.)"
        exit 1
        ;;
esac
