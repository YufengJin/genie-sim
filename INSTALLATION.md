# Genie Sim — Installation Guide

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Assets Setup](#2-assets-setup)
3. [Docker Images](#3-docker-images)
   - [Main Benchmark Image](#31-main-benchmark-image)
   - [Data Collection Image](#32-data-collection-image)
   - [Scene Reconstruction Image](#33-scene-reconstruction-image)
   - [Scene Generator Server Image](#34-scene-generator-server-image)
   - [Dashboard Image](#35-dashboard-image)
   - [Development Image (Claude Code)](#36-development-image-claude-code)
4. [Running Containers](#4-running-containers)
   - [Interactive GUI](#41-interactive-gui)
   - [Headless / Batch](#42-headless--batch)
5. [Dashboard](#5-dashboard)
   - [Online Mode (live simulation)](#51-online-mode-live-simulation)
   - [Offline Mode (dataset replay)](#52-offline-mode-dataset-replay)
6. [Volume Mounts Reference](#6-volume-mounts-reference)
7. [GPU Configuration](#7-gpu-configuration)
8. [Environment Variables](#8-environment-variables)
9. [Port Reference](#9-port-reference)

---

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA RTX 40 series (RTX 4080+ recommended) |
| Driver | ≥ 550.120 |
| CUDA | 12.4 + |
| RAM | 32 GB minimum, 64 GB recommended |
| Storage | 50 GB SSD minimum |
| Docker | Docker Engine + NVIDIA Container Toolkit |

Install NVIDIA Container Toolkit if not already installed:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 2. Assets Setup

Download simulation assets from ModelScope and set the environment variable **before running anything**:

```bash
# Download from:
# https://modelscope.cn/datasets/agibot_world/GenieSimAssets

# Place into:
genie_sim/source/geniesim/assets/

# Export the path (add to ~/.bashrc for persistence):
export SIM_ASSETS=$(pwd)/source/geniesim/assets
```

---

## 3. Docker Images

### 3.1 Main Benchmark Image

The primary runtime image. Includes Isaac Sim 5.1.0, ROS 2 Jazzy, CycloneDDS, and all benchmark dependencies.

**Option A — Pull pre-built image (recommended):**

```bash
docker pull registry.agibot.com/genie-sim/open_source:latest
```

**Option B — Build locally:**

```bash
docker build -f ./scripts/dockerfile \
    -t registry.agibot.com/genie-sim/open_source:latest \
    .
```

Build time: ~30–60 minutes. The Dockerfile installs ROS 2 Jazzy, CycloneDDS, Vulkan, FFmpeg, Git LFS, and sets up three Python virtual environments (generator, record, teleop).

---

### 3.2 Data Collection Image

Extends the main image with cuRobo motion planning support.

> **Important:** cuRobo requires knowing your GPU's CUDA compute capability. Edit the Dockerfile before building.

```bash
# Check your GPU's SM version:
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# e.g. RTX 4090 → 8.9, RTX 3090 → 8.6

# Edit source/data_collection/dockerfile:
#   ENV TORCH_CUDA_ARCH_LIST="8.9"   ← change to match your GPU

cd source/data_collection
docker build -f dockerfile \
    -t registry.agibot.com/genie-sim/open_source-data-collection:latest \
    .
```

Build time: ~60–90 minutes (compiles cuRobo from source).

---

### 3.3 Scene Reconstruction Image

Multi-stage image for the 3DGS + Difix3D reconstruction pipeline. Includes CloudCompare, COLMAP, PGSR, gsplat, and Hierarchical Localization.

```bash
cd source/scene_reconstruction
docker build -t genie-sim-reconstruction:latest .
```

Run:

```bash
docker run --rm --gpus all -it --shm-size=32g \
    -v $(pwd):/House \
    genie-sim-reconstruction:latest

# Inside container:
cd /root/third_party/gsplat/examples
sh real2sim_environment_entrypoint.sh /mnt 1   # 1 = enable Difix3D
```

---

### 3.4 Scene Generator Server Image

Lightweight FastAPI server powering the LLM-driven scene generation pipeline (Open WebUI + MCP server).

```bash
cd source/geniesim/generator/server
docker build -t genie-sim-generator:latest .
```

Or use the launch script which starts both Open WebUI and the MCP server together:

```bash
./scripts/start_generator.sh
# Open WebUI: http://localhost:8080
# MCP server: http://localhost:8765
```

Required environment variables before running:

```bash
export OPENAI_API_BASE_URL=https://...
export OPENAI_API_KEY=sk-...
```

---

### 3.5 Dashboard Image

Web dashboard for monitoring robot camera feeds, joint states, and fleet status. Includes YOLO object detection and Rerun 3D visualization.

**Online mode image** (connects to live Isaac Sim via ROS 2):

```bash
docker build -f dashboard/Dockerfile \
    -t geniesim-dashboard:latest \
    .
```

**Offline replay image** (standalone dataset replay, no ROS/Isaac Sim required):

```bash
# Build online image first, then:
docker build -f dashboard/Dockerfile.offline \
    -t geniesim-dashboard-offline:latest \
    .
```

Both images are also built automatically by `start_dashboard.sh` when `DASHBOARD_BUILD=1` is set.

---

### 3.6 Development Image (Claude Code)

Extends the main benchmark image with Node.js 22 and Claude Code CLI for AI-assisted development.

```bash
docker build -f ./scripts/dockerfile.dev \
    -t genie-sim-dev:latest \
    .
```

Run with your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/start_dev.sh
```

---

## 4. Running Containers

All container launch scripts perform the following automatically:
- Configure ROS 2 Jazzy + CycloneDDS environment
- Set up file ACL permissions for user 1234
- Install the `ik_solver` wheel and editable `geniesim` package
- Create convenience aliases: `omni_python`, `isaacsim`, `geniesim`

### 4.1 Interactive GUI

Starts a long-running background container with X11 display forwarding. Use this for interactive development and testing with a visible simulation window.

```bash
# Start the container (stays running in background):
./scripts/start_gui.sh

# Open a new terminal into the running container:
./scripts/into.sh

# Inside container, run a demo task:
omni_python source/geniesim/app/app.py --conf
```

### 4.2 Headless / Batch

Starts a single interactive session that auto-removes on exit. Use this for scripted runs.

```bash
./scripts/start_headless.sh

# Inside container:
omni_python source/geniesim/app/app.py --headless --conf
```

---

## 5. Dashboard

The dashboard streams robot camera feeds (head, left-wrist, right-wrist) and displays joint states and fleet status in a web browser at `http://localhost:8200`.

### 5.1 Online Mode (live simulation)

Requires a running Genie Sim container and ROS 2 topics being published.

**Docker mode** (recommended — runs in an isolated ROS Jazzy container):

```bash
./scripts/start_dashboard.sh docker
```

**Local mode** (if ROS 2 Jazzy is installed on the host):

```bash
./scripts/start_dashboard.sh local
```

Configuration via environment variables:

```bash
DASHBOARD_PORT=8200          # web port (default: 8200)
DASHBOARD_JPEG_QUALITY=75    # JPEG compression quality (default: 75)
DASHBOARD_TARGET_FPS=10      # streaming frame rate (default: 10)
DASHBOARD_BUILD=1            # force rebuild before run
```

Open the dashboard at `http://localhost:8200`.

---

### 5.2 Offline Mode (dataset replay)

Replays recorded LeRobot HDF5 datasets without needing Isaac Sim or ROS 2. Supports 3D visualization via Rerun.

**Docker mode:**

```bash
DATASETS_HOST=/path/to/datasets ./scripts/start_dashboard_offline.sh docker
```

**Local mode:**

```bash
DASHBOARD_MODE=offline \
FLEET_CONFIG=./dashboard/fleet_config.yaml \
python3 -m dashboard
```

Configuration via environment variables:

```bash
DATASETS_HOST=/mnt/datasets    # path to LeRobot dataset directory
LAUNCH_RERUN=1                 # enable Rerun 3D visualizer (default: off)
RERUN_WEB_PORT=9090            # Rerun viewer port (default: 9090)
```

Access:
- Dashboard: `http://localhost:8200`
- Rerun 3D viewer: `http://localhost:9090` (if `LAUNCH_RERUN=1`)

---

## 6. Volume Mounts Reference

All Isaac Sim containers mount the following host directories:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `~/docker/isaac-sim/cache/main` | `/isaac-sim/.cache` | Shader / asset cache |
| `~/docker/isaac-sim/cache/computecache` | `/isaac-sim/.nv/ComputeCache` | GPU compute cache |
| `~/docker/isaac-sim/logs` | `/isaac-sim/.nvidia-omniverse/logs` | Isaac Sim logs |
| `~/docker/isaac-sim/config` | `/isaac-sim/.nvidia-omniverse/config` | Isaac Sim config |
| `~/docker/isaac-sim/data` | `/isaac-sim/.local/share/ov/data` | Omniverse data |
| `~/docker/isaac-sim/pkg` | `/isaac-sim/.local/share/ov/pkg` | Omniverse packages |
| `/dev/input` | `/dev/input` | Gamepad / joystick input |
| `$PWD` (project root) | `/geniesim/main` | Source code (read/write) |

The start scripts create the cache directories automatically on first run.

Dashboard additional mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `source/chaser_brain/.env` | `/app/chaser_brain/.env` | OpenAI API key (optional) |
| `config/cyclonedds.xml` | `/geniesim/main/config/cyclonedds.xml` | DDS network config |
| `$DATASETS_HOST` | `/datasets` | Offline dataset replay |

---

## 7. GPU Configuration

**CUDA Compute Capability by GPU:**

| GPU | SM Version | `TORCH_CUDA_ARCH_LIST` |
|-----|-----------|------------------------|
| RTX 4090 / 4090D | 8.9 | `"8.9"` |
| RTX 4080 | 8.9 | `"8.9"` |
| RTX 3090 / 3080 | 8.6 | `"8.6"` |
| RTX 5090 | 12.0 | `"12.0"` ⚠ cuRobo may not support yet |

> **Note:** RTX 50 series (SM 12.0) — Isaac Sim 5.1.0 is supported, but cuRobo compilation may fail pending upstream support. The main benchmark image does not require cuRobo.

Edit `source/data_collection/dockerfile` before building the data collection image:

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="8.9"   # ← change to your GPU's SM version
```

---

## 8. Environment Variables

Variables that must be set on the **host** before launching containers:

| Variable | Required By | Description |
|----------|-------------|-------------|
| `SIM_ASSETS` | All containers | Absolute path to `source/geniesim/assets/` |
| `ANTHROPIC_API_KEY` | Dev container | Claude Code API key |
| `OPENAI_API_KEY` | Dashboard / Chaser Brain | OpenAI API key for VLM features |
| `OPENAI_API_BASE_URL` | Scene generator | API base URL (supports compatible endpoints) |
| `BASE_URL` | Benchmark scoring | LLM endpoint for evaluation generation |
| `API_KEY` | Benchmark scoring | LLM API key |
| `MODEL` | Benchmark scoring | Model name, e.g. `qwen3-max` |
| `VL_MODEL` | Benchmark scoring | Vision model name, e.g. `qwen3-vl-plus` |

Add persistent variables to `~/.bashrc`:

```bash
export SIM_ASSETS=/path/to/genie_sim/source/geniesim/assets
```

---

## 9. Port Reference

| Port | Service | Image | Notes |
|------|---------|-------|-------|
| 50051 | gRPC API (Isaac Sim) | Main benchmark | Default; set via `benchmark.client_host` in config.yaml |
| 8200 | Dashboard web UI | Dashboard | Camera feeds + fleet status |
| 8765 | MCP server / Chaser Brain | Generator server / Dashboard | LLM scene gen or VLM API |
| 8080 | Open WebUI | Generator server | LLM scene generation UI |
| 8999 | Inference service | External (OpenPI) | Policy inference endpoint |
| 9090 | Rerun 3D viewer | Dashboard offline | 3D point cloud + trajectory |
