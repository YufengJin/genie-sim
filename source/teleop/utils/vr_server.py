# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import socket
import json
import threading

from utils.logger import logger
import socket


class VRServer:
    def __init__(self, host=None, port=8080):
        self.data = None
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_addr = "0.0.0.0"
        logger.info(f"[VRServer] Binding UDP socket to ({bind_addr!r}, {port}) (host_ip={host!r})")
        self.sock.bind((bind_addr, port))
        logger.info(f"[VRServer] UDP socket bound successfully, waiting for data...")
        listener_thread = threading.Thread(target=self.udp_listener)
        listener_thread.daemon = True
        listener_thread.start()
        self.counter = 0
        self.recv_count = 0

    def udp_listener(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                message = data.decode("utf-8")
                _new_message = message.replace("False", "false")
                json_data = json.loads(_new_message)
                self.data = json_data
                self.recv_count += 1
                if self.recv_count <= 3 or self.recv_count % 300 == 0:
                    logger.info(f"[VRServer] UDP recv #{self.recv_count} from {addr}, len={len(data)}")
                if self.recv_count == 1:
                    logger.info(f"[VRServer] First packet data: {_new_message[:500]}")
            except json.JSONDecodeError:
                logger.info(f"[VRServer] Receive {addr} NON-JSON data: {_new_message[:200]}")
            except Exception as e:
                logger.error(f"[VRServer] UDP listener error: {e}")

    def on_update(self):
        self.counter += 1
        if self.counter % 300 == 0:
            logger.info(f"[VRServer] on_update called {self.counter} times, recv_count={self.recv_count}, has_data={self.data is not None}")
        if self.data is not None:
            return self.data
        return None


if __name__ == "__main__":
    vr_server = VRServer(host="", port=8080)
    while True:
        vr_server.on_update()
