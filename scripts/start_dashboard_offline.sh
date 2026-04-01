#!/bin/bash
# Launch Genie Sim Dashboard in offline replay mode (no ROS / Isaac Sim required).
# Reads LeRobot datasets from disk and replays camera frames through the dashboard UI.
#
# Usage:
#   ./scripts/start_dashboard_offline.sh           # docker mode (default)
#   ./scripts/start_dashboard_offline.sh local      # host mode (requires deps installed)
#
# Env vars:
#   DASHBOARD_PORT          (default: 8200)
#   DASHBOARD_TARGET_FPS    (default: 10)
#   DATASETS_HOST           (default: /mnt/hdd18T/datasets/AgiBotWorld/Reasoning2Action-Sim)
#   DASHBOARD_IMAGE         (default: geniesim-dashboard-offline:latest)
#   DASHBOARD_BUILD=1       Force rebuild image before run
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_DIR/dashboard"

DASHBOARD_PORT="${DASHBOARD_PORT:-8200}"
DASHBOARD_TARGET_FPS="${DASHBOARD_TARGET_FPS:-10}"
DATASETS_HOST="${DATASETS_HOST:-/mnt/hdd18T/datasets/AgiBotWorld/Reasoning2Action-Sim}"
CONTAINER_NAME="${CONTAINER_NAME:-dashboard-offline}"
DASHBOARD_IMAGE="${DASHBOARD_IMAGE:-geniesim-dashboard-offline:latest}"

MODE="${1:-docker}"

# Auto-kill anything occupying the dashboard port
_free_port() {
    local port="$1"
    local cid
    cid=$(docker ps --filter "publish=${port}" -q 2>/dev/null)
    if [ -n "$cid" ]; then
        echo "[Dashboard-Offline] Stopping container using port ${port}..."
        docker stop "$cid" >/dev/null 2>&1 || true
    fi
    local pid
    pid=$(lsof -ti :"${port}" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[Dashboard-Offline] Killing process ${pid} on port ${port}..."
        kill "$pid" 2>/dev/null || true
        sleep 0.5
    fi
}

case "$MODE" in
    docker)
        _free_port "$DASHBOARD_PORT"

        if [ "${DASHBOARD_BUILD:-0}" = "1" ]; then
            echo "[Dashboard-Offline] Building image ${DASHBOARD_IMAGE}..."
            docker build -f "$DASHBOARD_DIR/Dockerfile.offline" -t "${DASHBOARD_IMAGE}" "$PROJECT_DIR"
        elif ! docker image inspect "${DASHBOARD_IMAGE}" >/dev/null 2>&1; then
            echo "[Dashboard-Offline] Image '${DASHBOARD_IMAGE}' not found, building..."
            docker build -f "$DASHBOARD_DIR/Dockerfile.offline" -t "${DASHBOARD_IMAGE}" "$PROJECT_DIR"
        else
            echo "[Dashboard-Offline] Using existing image '${DASHBOARD_IMAGE}' (set DASHBOARD_BUILD=1 to rebuild)."
        fi

        echo "=== Genie Sim Dashboard (Offline Replay — Docker) ==="
        echo "Image        : $DASHBOARD_IMAGE"
        echo "Datasets     : $DATASETS_HOST → /datasets"
        echo "Port         : $DASHBOARD_PORT"
        echo "Container    : $CONTAINER_NAME"
        echo "====================================================="

        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

        docker run -d --name "$CONTAINER_NAME" \
            --gpus all \
            --network=host \
            -v "${DATASETS_HOST}:/datasets:ro" \
            -v "${PROJECT_DIR}/dashboard:/app/dashboard" \
            -e DASHBOARD_MODE=offline \
            -e FLEET_CONFIG=/app/dashboard/fleet_config.yaml \
            -e DASHBOARD_PORT="$DASHBOARD_PORT" \
            -e DASHBOARD_TARGET_FPS="$DASHBOARD_TARGET_FPS" \
            -e LAUNCH_RERUN=1 \
            -e RERUN_WEB_PORT="${RERUN_WEB_PORT:-9090}" \
            "$DASHBOARD_IMAGE"

        echo "[Dashboard-Offline] Started in background. Open http://localhost:${DASHBOARD_PORT}"
        echo "[Dashboard-Offline] Rerun viewer: http://localhost:${RERUN_WEB_PORT:-9090}"
        echo "[Dashboard-Offline] Logs: docker logs -f $CONTAINER_NAME"
        ;;

    local)
        echo "=== Genie Sim Dashboard (Offline Replay — Local) ==="
        echo "Port         : $DASHBOARD_PORT"
        echo "FPS          : $DASHBOARD_TARGET_FPS"
        echo "==================================================="

        export DASHBOARD_MODE=offline
        export FLEET_CONFIG="${FLEET_CONFIG:-$PROJECT_DIR/dashboard/fleet_config.yaml}"
        export DASHBOARD_PORT
        export DASHBOARD_TARGET_FPS
        export LAUNCH_RERUN=1

        cd "$PROJECT_DIR"
        python3 -m dashboard
        ;;

    *)
        echo "Usage: $0 [docker|local]"
        echo "  docker  - Run in Docker container (default, recommended)"
        echo "  local   - Run directly (requires lerobot, torch, etc. installed)"
        exit 1
        ;;
esac
