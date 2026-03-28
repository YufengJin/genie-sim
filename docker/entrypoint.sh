#!/bin/bash
set -e

# ── Environment ──────────────────────────────────────────────────────────────
export ROS_DISTRO=jazzy
export ISAACSIM_HOME=/isaac-sim
export PATH="${HOME}/.local/bin:${HOME}/.bun/bin:${PATH}"

# ── Permissions (user 1234) ──────────────────────────────────────────────────
if [ -d "/geniesim/main" ]; then
    sudo setfacl -m u:1234:rwX /geniesim/main
    sudo setfacl -m u:1234:rwX /geniesim/main/source 2>/dev/null || true
    sudo setfacl -m u:1234:rwX /geniesim/main/source/geniesim/benchmark/saved_task 2>/dev/null || true

    # Teleop dirs
    if [ -d "/geniesim/main/source/teleop" ]; then
        sudo setfacl -m u:1234:rwX /geniesim/main/source/teleop
        sudo mkdir -p /geniesim/main/source/teleop/app/bin/.cache
        sudo mkdir -p /geniesim/main/source/teleop/app/bin/logs/dylog
        sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/bin/.cache
        sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/bin/logs
        sudo chown -R 1234:1234 /geniesim/main/source/teleop/app/share 2>/dev/null || true
    fi
fi

# Isaac Sim cache dirs
sudo setfacl -m u:1234:rwX /isaac-sim/.cache 2>/dev/null || true
sudo setfacl -m u:1234:rwX /isaac-sim/.nv/ComputeCache 2>/dev/null || true
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/logs 2>/dev/null || true
sudo setfacl -m u:1234:rwX /isaac-sim/.nvidia-omniverse/config 2>/dev/null || true
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/data 2>/dev/null || true
sudo setfacl -m u:1234:rwX /isaac-sim/.local/share/ov/pkg 2>/dev/null || true

# ── Shell config ─────────────────────────────────────────────────────────────
cat >> ~/.bashrc << 'BASHEOF'
export SIM_REPO_ROOT=/geniesim/main
export ENABLE_SIM=1
export ROS_DISTRO=jazzy
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ISAACSIM_HOME=/isaac-sim
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ISAACSIM_HOME}/exts/isaacsim.ros2.bridge/${ROS_DISTRO}/lib"
export ROS_CMD_DISTRO=jazzy
export PATH="${HOME}/.local/bin:${HOME}/.bun/bin:${PATH}"
source ${ISAACSIM_HOME}/setup_ros_env.sh

alias omni_python='${ISAACSIM_HOME}/python.sh'
alias isaacsim='${ISAACSIM_HOME}/runapp.sh'
alias geniesim='omni_python /geniesim/main/source/geniesim/app/app.py'
BASHEOF

# ── Editable install ────────────────────────────────────────────────────────
if [ -d "/geniesim/main/source" ]; then
    sudo rm -rf /geniesim/main/source/GenieSim.egg-info
    /isaac-sim/python.sh -m pip install /geniesim/main/3rdparty/ik_solver-0.4.3-cp311-cp311-linux_x86_64.whl 2>/dev/null || true
    /isaac-sim/python.sh -m pip install -e /geniesim/main/source
fi

echo ">> GenieSim dev environment ready."
exec "$@"
