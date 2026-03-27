# GenieSim Docker Environment

## Prerequisites

- Docker Engine 24+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (RTX 40 series recommended)
- GenieSim assets downloaded (`SIM_ASSETS` env var pointing to the directory)

## Environment Setup

Copy the example env file and fill in your values:

```bash
cp docker/.env.example docker/.env
# Edit docker/.env with your SIM_ASSETS path and optional settings
```

## Image Hierarchy

```
nvcr.io/nvidia/isaac-sim:5.1.0
  └── geniesim-dev:latest              (Dockerfile)
        └── geniesim-data-collection   (Dockerfile.data-collection)

python:3.11
  └── geniesim-generator-server        (in docker-compose.full.yaml)
```

## Build

```bash
# From repo root:

# 1. Base dev image (Isaac Sim + ROS 2 + Claude Code)
docker compose -f docker/docker-compose.headless.yaml build

# 2. Data collection image (adds CUDA 12.8 + cuRobo, requires step 1)
docker compose -f docker/docker-compose.data-collection.yaml build

# 3. Full stack with generator server
docker compose -f docker/docker-compose.full.yaml build
```

To change GPU architecture (default RTX4090D = 8.9):
```bash
docker compose -f docker/docker-compose.data-collection.yaml build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0"
```

## Usage

### Headless (benchmark / training)

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it geniesim-headless bash

# Inside container:
omni_python source/geniesim/benchmark/task_benchmark.py
```

### GUI (Isaac Sim visualization)

```bash
xhost +local:
docker compose -f docker/docker-compose.x11.yaml up -d
docker exec -it geniesim-gui bash

# Inside container:
geniesim  # alias for omni_python source/geniesim/app/app.py
```

### Data Collection

```bash
docker compose -f docker/docker-compose.data-collection.yaml up -d
docker exec -it geniesim-data-collection bash

# Terminal 1: server
omni_python source/data_collection/scripts/data_collector_server.py \
  --enable_physics --enable_curobo --publish_ros

# Terminal 2: client
omni_python source/data_collection/scripts/run_data_collection.py \
  --task_template tasks/geniesim_2025/<task>.json --use_recording
```

### Full Stack (sim + generator server)

```bash
docker compose -f docker/docker-compose.full.yaml up -d
docker exec -it geniesim-full bash        # Isaac Sim
docker exec -it geniesim-generator bash   # Generator server
```

## Volume Mounts

| Host | Container | Purpose |
|------|-----------|---------|
| `<repo-root>` | `/geniesim/main` | Project source (editable) |
| `~/docker/isaac-sim/cache/main` | `/isaac-sim/.cache` | Isaac Sim cache |
| `~/docker/isaac-sim/cache/computecache` | `/isaac-sim/.nv/ComputeCache` | Shader cache |
| `~/docker/isaac-sim/logs` | `/isaac-sim/.nvidia-omniverse/logs` | Omniverse logs |
| `~/docker/isaac-sim/config` | `/isaac-sim/.nvidia-omniverse/config` | Omniverse config |
| `~/.claude` | `/root/.claude` | Claude Code settings (ro) |
| `~/.claude.json` | `/root/.claude.json` | Claude Code auth token (rw) |
| `$SIM_ASSETS` | `/geniesim/assets` | GenieSim assets (ro) |

## Entrypoint

On container start, `entrypoint.sh` performs:
1. Sets ACL permissions for user 1234 on project & Isaac Sim dirs
2. Configures ROS 2 Jazzy environment in `.bashrc`
3. Installs `ik_solver` wheel and GenieSim package in editable mode
4. Sets up aliases: `omni_python`, `isaacsim`, `geniesim`

## Smoke Test

After building all images, run the automated smoke test:

```bash
./docker/smoke-test.sh
```

This verifies 10 checks across both images:
- **geniesim-dev** (7 checks): Isaac Sim Python, GenieSim import, ROS 2 Jazzy, 3 venvs, Claude Code
- **geniesim-data-collection** (3 checks): CUDA 12.8, cuRobo import, data collection deps

The script starts containers, runs checks via `docker exec`, prints colored PASS/FAIL results, and cleans up.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (unset) | Claude Code API key |
| `TORCH_CUDA_ARCH_LIST` | `8.9` | GPU SM version for cuRobo build |
| `ACCEPT_EULA` | `Y` | Isaac Sim EULA acceptance |
| `DISPLAY` | from host | X11 display (GUI mode only) |
