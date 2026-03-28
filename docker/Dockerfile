# GenieSim 3.0 Development Image
# Build: docker build -f docker/Dockerfile -t geniesim-dev:latest .
# Build context must be the repo root.

FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive

# ── 1. System deps ───────────────────────────────────────────────────────────
USER root
RUN apt-get update && apt-get install -y \
    bash wget curl git git-lfs sudo lsb-release unzip xterm \
    build-essential cmake ninja-build libeigen3-dev \
    ffmpeg jq vim acl tmux \
    python3 python3-pip python3-venv \
    libvulkan1 vulkan-tools \
    libx11-6 libxrandr2 libxss1 libxcursor1 libxi6 libglu1-mesa \
    libgl1 libegl1 libnss3 libxcomposite1 libxdamage1 libxext6 \
    liburing2 liburing-dev \
    software-properties-common locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── 2. ROS 2 Jazzy ──────────────────────────────────────────────────────────
RUN add-apt-repository universe -y \
    && curl -L -o /tmp/ros2-apt-source.deb \
       "https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb" \
    && dpkg -i /tmp/ros2-apt-source.deb && rm /tmp/ros2-apt-source.deb \
    && apt-get update && apt-get install -y \
       ros-jazzy-desktop \
       ros-jazzy-rmw-cyclonedds-cpp \
       ros-jazzy-vision-msgs-rviz-plugins ros-jazzy-joy \
       ros-dev-tools graphviz libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Isaac Sim Python deps (benchmark) ────────────────────────────────────
COPY ./requirements.txt /tmp/requirements.txt
RUN /isaac-sim/python.sh -m pip install pip -U \
    && /isaac-sim/python.sh -m pip install -r /tmp/requirements.txt

# ── 4. Generator venv ────────────────────────────────────────────────────────
RUN mkdir -p /geniesim && chown 1234:1234 /geniesim
COPY ./source/geniesim/generator/requirements.txt /tmp/requirements.generator.txt
RUN python3 -m venv /geniesim/generator_env \
    && /geniesim/generator_env/bin/pip install pip -U \
    && /geniesim/generator_env/bin/pip install -r /tmp/requirements.generator.txt

# ── 5. Record venv ───────────────────────────────────────────────────────────
RUN python3 -m venv /geniesim/record_env \
    && /geniesim/record_env/bin/pip install pip -U \
    && /geniesim/record_env/bin/pip install rosbags pyyaml opencv-python rosbags.image "numpy<2.0"

# ── 6. Teleop venv ──────────────────────────────────────────────────────────
COPY ./source/teleop/requirements.txt /tmp/requirements.teleop.txt
RUN python3 -m venv /geniesim/teleop_env \
    && /geniesim/teleop_env/bin/pip install pip -U \
    && /geniesim/teleop_env/bin/pip install -r /tmp/requirements.teleop.txt

# ── 7. Sudo access for user 1234 ────────────────────────────────────────────
RUN usermod -aG sudo isaac-sim \
    && echo "isaac-sim ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/isaac-sim \
    && chmod 0440 /etc/sudoers.d/isaac-sim

USER 1234:1234

# ── 8. Claude Code (bun + native installer, as user 1234) ──────────────────
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/home/isaac-sim/.bun/bin:${PATH}"
WORKDIR /tmp
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/isaac-sim/.local/bin:${PATH}"

# ── 9. Entrypoint ───────────────────────────────────────────────────────────
WORKDIR /geniesim/main
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sudo chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
