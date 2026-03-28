#!/bin/bash
# GenieSim Docker Smoke Test
# Usage: ./docker/smoke-test.sh
# Requires: images already built (see docker/README.md)

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

run_check() {
    local label="$1"
    local container="$2"
    local cmd="$3"
    local check_output="${4:-}"  # optional: grep pattern to match in output

    printf "  %-35s " "$label"

    output=$(docker exec "$container" bash -c "$cmd" 2>&1)
    exit_code=$?

    if [ -n "$check_output" ]; then
        if echo "$output" | grep -q "$check_output"; then
            printf "${GREEN}PASS${NC}\n"
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS+=("PASS: $label")
            return 0
        else
            printf "${RED}FAIL${NC} (expected output containing '$check_output')\n"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            RESULTS+=("FAIL: $label — expected '$check_output' in output")
            return 1
        fi
    elif [ $exit_code -eq 0 ]; then
        printf "${GREEN}PASS${NC}\n"
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("PASS: $label")
        return 0
    else
        printf "${RED}FAIL${NC} (exit code: $exit_code)\n"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("FAIL: $label — exit $exit_code: $(echo "$output" | tail -1)")
        return 1
    fi
}

cleanup() {
    echo ""
    echo "Cleaning up containers..."
    docker compose -f docker/docker-compose.headless.yaml down --timeout 5 2>/dev/null
    docker compose -f docker/docker-compose.data-collection.yaml down --timeout 5 2>/dev/null
}

trap cleanup EXIT

# ── Phase 1: geniesim-dev ────────────────────────────────────────────────────
echo ""
echo "${YELLOW}=== Phase 1: geniesim-dev ===${NC}"
echo "Starting container..."
docker compose -f docker/docker-compose.headless.yaml up -d --no-build 2>/dev/null

# Wait for entrypoint to finish
echo "Waiting for entrypoint to complete (editable install)..."
sleep 10

CONTAINER="geniesim-headless"

echo "Running checks:"
run_check "Isaac Sim Python"       "$CONTAINER" "/isaac-sim/python.sh -c 'import isaacsim'"
run_check "GenieSim package"       "$CONTAINER" "/isaac-sim/python.sh -c 'import geniesim'"
run_check "ROS 2 Jazzy"            "$CONTAINER" "source /opt/ros/jazzy/setup.bash && ros2 --help"
run_check "Generator venv"         "$CONTAINER" "/geniesim/generator_env/bin/python -c 'import networkx'"
run_check "Record venv"            "$CONTAINER" "/geniesim/record_env/bin/python -c 'import rosbags'"
run_check "Teleop venv"            "$CONTAINER" "/geniesim/teleop_env/bin/python -c 'import scipy'"
run_check "Claude Code CLI"        "$CONTAINER" "\${HOME}/.local/bin/claude --version"

echo "Stopping geniesim-dev..."
docker compose -f docker/docker-compose.headless.yaml down --timeout 5 2>/dev/null

# ── Phase 2: geniesim-data-collection ────────────────────────────────────────
echo ""
echo "${YELLOW}=== Phase 2: geniesim-data-collection ===${NC}"
echo "Starting container..."
docker compose -f docker/docker-compose.data-collection.yaml up -d --no-build 2>/dev/null

echo "Waiting for entrypoint to complete..."
sleep 10

CONTAINER="geniesim-data-collection"

echo "Running checks:"
run_check "CUDA 12.8 toolkit"      "$CONTAINER" "/usr/local/cuda-12.8/bin/nvcc --version" "12.8"
run_check "cuRobo import"          "$CONTAINER" "/isaac-sim/python.sh -c 'import curobo'"
run_check "DC deps (grpc, h5py)"   "$CONTAINER" "/isaac-sim/python.sh -c 'import grpc; import h5py'"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Results: ${GREEN}${PASS_COUNT} passed${NC}, ${RED}${FAIL_COUNT} failed${NC}"
echo "============================================"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed checks:"
    for r in "${RESULTS[@]}"; do
        if [[ "$r" == FAIL* ]]; then
            echo "  ${RED}✗${NC} ${r#FAIL: }"
        fi
    done
    echo ""
    exit 1
fi

echo ""
echo "${GREEN}All smoke tests passed!${NC}"
exit 0
