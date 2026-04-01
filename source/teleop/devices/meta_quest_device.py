#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .pico_device import PicoDevice


class MetaQuestDevice(PicoDevice):
    """Meta Quest VR device for Genie Sim teleoperation.

    The external quest_bridge.py process reads Quest controller data via
    oculus_reader and translates it to Pico-compatible UDP JSON format.
    This class therefore reuses PicoDevice's parsing logic entirely,
    with only a different default port (8081 vs 8080).

    Usage:
        1. On host machine: uv run python utils/quest_bridge.py --target_ip <IP> --target_port 8081
        2. In container:    python teleop.py --device_type meta_quest --port 8081
    """

    def __init__(self, host_ip=None, port=8081, robot_cfg="G2_omnipicker"):
        super().__init__(host_ip=host_ip, port=port, robot_cfg=robot_cfg)
