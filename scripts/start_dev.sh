#!/bin/bash
# Launch genie-sim-dev container with Claude Code, GPU access, and project mounted

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p ~/docker/isaac-sim/cache/{main,computecache}
mkdir -p ~/docker/isaac-sim/{config,data,logs,pkg}
mkdir -p ~/.claude

docker run -it --name genie_sim_dev \
    --user 1234:1234 \
    --entrypoint ./scripts/entrypoint.sh \
    --rm \
    --gpus all --network=host --privileged \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v /dev/input:/dev/input:rw \
    -v $CURRENT_DIR:/geniesim/main:rw \
    -v ~/.claude:/home/isaac-sim/.claude:rw \
    genie-sim-dev:latest bash
