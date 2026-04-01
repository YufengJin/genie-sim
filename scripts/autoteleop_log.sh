#!/bin/bash
CONTAINER_NAME="genie_sim_benchmark"
START_SCRIPT="$PWD/scripts/start_gui.sh"
PROCESS_CLIENT="teleop|ros"
LOG_DIR="$PWD/logs"
CLEANUP_DONE=0

# VR device configuration (pico or meta_quest)
DEVICE_TYPE="pico"
VR_PORT=8080
TELEOP_EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE_TYPE="$2"
            shift 2
            ;;
        --port)
            VR_PORT="$2"
            shift 2
            ;;
        --legend)
            TELEOP_EXTRA_ARGS="$TELEOP_EXTRA_ARGS --legend"
            shift
            ;;
        --rerun)
            TELEOP_EXTRA_ARGS="$TELEOP_EXTRA_ARGS --rerun"
            shift
            ;;
        --rerun_trail)
            TELEOP_EXTRA_ARGS="$TELEOP_EXTRA_ARGS --rerun_trail $2"
            shift 2
            ;;
        --pos_gain|--rot_gain|--spatial_coeff|--max_pos_step|--max_rot_step)
            TELEOP_EXTRA_ARGS="$TELEOP_EXTRA_ARGS $1 $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set default port by device type if not explicitly provided
if [[ "$DEVICE_TYPE" == "meta_quest" && "$VR_PORT" == "8080" ]]; then
    VR_PORT=8081
fi

echo "Info: Device type=$DEVICE_TYPE, VR port=$VR_PORT"

# Cleanup function: kill container processes and host-side PIDs
cleanup() {
    if [[ "$CLEANUP_DONE" -eq 1 ]]; then
        return
    fi
    CLEANUP_DONE=1
    echo ""
    echo "Info: Cleaning up all teleop processes..."

    # Kill child processes first (genie_motion_control), then parents (start_mc.sh)
    # This avoids zombies from killing parent before child
    docker exec "$CONTAINER_NAME" bash -c "
        pkill -SIGTERM -f 'genie_motion_control' 2>/dev/null || true
        pkill -SIGTERM -f 'teleop.py' 2>/dev/null || true
        pkill -SIGTERM -f 'bridge.py' 2>/dev/null || true
        sleep 1
        pkill -SIGTERM -f 'start_mc.sh' 2>/dev/null || true
    " 2>/dev/null || true

    # Wait and verify, escalate to SIGKILL if needed
    sleep 2
    local remaining
    remaining=$(docker exec "$CONTAINER_NAME" bash -c "pgrep -f 'teleop.py|bridge.py|genie_motion_control|start_mc.sh' 2>/dev/null" 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
        echo "Warning: Processes still alive after SIGTERM, sending SIGKILL..."
        docker exec "$CONTAINER_NAME" bash -c "
            pkill -9 -f 'genie_motion_control' 2>/dev/null || true
            pkill -9 -f 'teleop.py' 2>/dev/null || true
            pkill -9 -f 'bridge.py' 2>/dev/null || true
            pkill -9 -f 'start_mc.sh' 2>/dev/null || true
        " 2>/dev/null || true
        sleep 1
    fi

    # Clean up DDS shared memory to prevent participant exhaustion on next start
    docker exec "$CONTAINER_NAME" bash -c "rm -f /dev/shm/iceoryx_* /dev/shm/cyclonedds* /dev/shm/fastrtps_* 2>/dev/null" 2>/dev/null || true

    # Kill host-side docker exec PIDs
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    echo "Info: Cleanup complete."
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap '' TSTP   # ignore Ctrl+Z

# If the pinocchio library does not exist in vendors/lib, extract it
PINOCCHIO_LIB="$PWD/source/teleop/app/vendors/lib/libpinocchio_casadi.so.3.7.0"
PINOCCHIO_TAR="$PWD/source/teleop/app/vendors/lib/libpinocchio.tar.gz"
if [ ! -f "$PINOCCHIO_LIB" ]; then
    echo "Extracting libpinocchio.tar.gz to vendors/lib ..."
    tar -xzvf "$PINOCCHIO_TAR" -C "$PWD/source/teleop/app/vendors/lib"
fi

if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    echo "Warning: Container $CONTAINER_NAME not running, try to start..."

    if [ -x "$START_SCRIPT" ]; then
        echo "Executing script: $START_SCRIPT (in background)"
        "$START_SCRIPT" &
        START_PID=$!
        MAX_WAIT=60
        ELAPSED=0
        while [ $ELAPSED -lt $MAX_WAIT ]; do
            sleep 3
            ELAPSED=$((ELAPSED + 3))
            if docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
                echo "Info: Container $CONTAINER_NAME started (after ${ELAPSED}s)"
                break
            fi
        done
        if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
            echo "Error: Container failed to start within ${MAX_WAIT}s"
            kill $START_PID 2>/dev/null || true
            exit 1
        fi
    else
        echo "Error: Start script $START_SCRIPT not exist or not executable"
        exit 1
    fi
else
    echo "Info: Container $CONTAINER_NAME already running"
fi

# Kill any leftover teleop-related processes and clean DDS shared memory
echo "Info: Cleaning up old teleop processes in container..."
docker exec "$CONTAINER_NAME" bash -c "pkill -f 'teleop.py' 2>/dev/null; pkill -f 'bridge.py' 2>/dev/null; pkill -f 'start_mc.sh' 2>/dev/null; pkill -f 'genie_motion_control' 2>/dev/null; rm -f /dev/shm/iceoryx_* /dev/shm/cyclonedds* /dev/shm/fastrtps_* 2>/dev/null" || true
sleep 2

# Prepare log directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG_DIR="$LOG_DIR/$TIMESTAMP"
mkdir -p "$SESSION_LOG_DIR"

echo "Info: Logs will be saved to $SESSION_LOG_DIR"

# Inner commands to run inside the container
declare -a INNER_CMDS=(
    "omni_python ./source/geniesim/app/app.py --config ./source/geniesim/config/teleop.yaml"
    "source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && python3 ./source/teleop/bridge.py"
    "source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && /geniesim/main/source/teleop/app/bin/start_mc.sh --no-tool"
    "source /geniesim/teleop_env/bin/activate && source /opt/ros/jazzy/setup.bash && source /geniesim/main/source/teleop/app/bin/env.sh && python3 ./source/teleop/teleop.py --device_type $DEVICE_TYPE --port $VR_PORT $TELEOP_EXTRA_ARGS"
)
declare -a LOG_NAMES=(
    "isaac_sim"
    "ros_bridge"
    "motion_control"
    "teleop"
)
declare -a DELAYS=(1 15 3 5 5)

declare -a PIDS=()

for i in "${!INNER_CMDS[@]}"; do
    sleep "${DELAYS[$i]}"
    LOG_FILE="$SESSION_LOG_DIR/${LOG_NAMES[$i]}.log"
    echo "Starting ${LOG_NAMES[$i]} → $LOG_FILE"
    docker exec "$CONTAINER_NAME" bash -ic "${INNER_CMDS[$i]}" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo -e "\nAll processes started (PIDs: ${PIDS[*]})"
echo "Logs: $SESSION_LOG_DIR"
echo "  tail -f $SESSION_LOG_DIR/isaac_sim.log"
echo "  tail -f $SESSION_LOG_DIR/ros_bridge.log"
echo "  tail -f $SESSION_LOG_DIR/motion_control.log"
echo "  tail -f $SESSION_LOG_DIR/teleop.log"
echo ""
TIMEOUT=3600  # Auto-cleanup after 1 hour if no input
echo "Press 'y' = teleoperation succeeded, keep data; 'n' = failed, discard ..."
echo "  (auto-cleanup after ${TIMEOUT}s if no input)"

while true; do
    if read -t "$TIMEOUT" -n 1 -s input; then
        if [[ "$input" == "Y" || "$input" == "y" ]]; then
            echo ""
            echo "Save the remote operation data.....Congratulations!"
            cleanup
            echo "Patching recording_info.json: add teleop_result"
            docker exec "$CONTAINER_NAME" python3 /geniesim/main/source/teleop/data_recording/patch_recording_info.py \
                --config /geniesim/main/source/geniesim/config/teleop.yaml \
                --base /geniesim/main/output/recording_data \
                || true
            break
        elif [[ "$input" == "N" || "$input" == "n" ]]; then
            echo ""
            cleanup
            echo "Patching recording_info.json: add teleop_result=false"
            docker exec "$CONTAINER_NAME" python3 /geniesim/main/source/teleop/data_recording/patch_recording_info.py \
                --config /geniesim/main/source/geniesim/config/teleop.yaml \
                --base /geniesim/main/output/recording_data \
                --teleop-result false \
                || true
            break
        fi
    else
        echo ""
        echo "Warning: Timeout (${TIMEOUT}s) reached, auto-cleaning up..."
        break
    fi
done

# trap EXIT will call cleanup() automatically
echo "Done. Logs saved in: $SESSION_LOG_DIR"
reset
