"""
MORAI UDP 송신기 — Ctrl Cmd / Traffic Light Set.

MORAI SIM:Drive 24.R2 UDP 바이너리 프로토콜.
참고: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1
"""
import socket
import struct

from .protocol import (
    HEADER_CTRL, HEADER_TL,
    CTRL_MODE_AUTO, GEAR_DRIVE, CMD_TYPE_THROTTLE,
)


# ═══════════════════════════════════════════════════════════════════
# 기본 송신기
# ═══════════════════════════════════════════════════════════════════

class _BaseSender:
    """UDP 소켓 + 목적지 주소를 보관하는 기본 송신기."""

    def __init__(self, dst_ip: str, dst_port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._address = (dst_ip, dst_port)

    def send_data(self, data):
        """서브클래스에서 구현. data 형식은 각 송신기마다 다름."""
        raw = self._pack(data)
        self._sock.sendto(raw, self._address)

    def _pack(self, data) -> bytes:
        raise NotImplementedError

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Ego Ctrl Cmd 송신기
# ═══════════════════════════════════════════════════════════════════

class CtrlCmdSender(_BaseSender):
    """
    차량 제어 명령 전송 (User → SIM).

    send_data([accel, brake, steer]):
        accel : float, 0 ~ 1
        brake : float, 0 ~ 1
        steer : float, -1 ~ 1  (정규화된 조향값)

    패킷 구조 (59 bytes):
        Header "#MoraiCtrlCmd$"  (14 bytes)
        data_length = 27         (int32, 4 bytes)
        aux_data                 (int32×3 = 12 bytes)
        ───── payload (27 bytes) ─────
        mode       (int8)  = 2  (AutoMode)
        gear       (int8)  = 4  (Drive)
        cmd_type   (int8)  = 1  (Throttle)
        velocity   (float) = 0
        acceleration(float)= 0
        accel_cmd  (float) = data[0]
        brake_cmd  (float) = data[1]
        front_steer(float) = data[2]
        rear_steer (float) = 0
        ───── tail ─────
        \\r\\n  (2 bytes)
    """

    _HEADER = HEADER_CTRL   # b'#MoraiCtrlCmd$'

    def _pack(self, data) -> bytes:
        accel_cmd  = float(data[0])
        brake_cmd  = float(data[1])
        front_steer = float(data[2])

        # ── Header ──
        header = self._HEADER
        header += struct.pack('<i', 27)        # data_length
        header += struct.pack('<iii', 0, 0, 0) # aux_data (12 bytes)

        # ── Payload (27 bytes) ──
        payload = struct.pack(
            '<bbb f f f f f f',
            CTRL_MODE_AUTO,     # mode = 2
            GEAR_DRIVE,         # gear = 4
            CMD_TYPE_THROTTLE,  # cmd_type = 1
            0.0,                # velocity (미사용)
            0.0,                # acceleration (미사용)
            accel_cmd,          # accel
            brake_cmd,          # brake
            front_steer,        # front_steer
            0.0,                # rear_steer
        )

        # ── Tail ──
        tail = b'\r\n'

        return header + payload + tail


# ═══════════════════════════════════════════════════════════════════
# Traffic Light Set 송신기
# ═══════════════════════════════════════════════════════════════════

class TrafficLightSender(_BaseSender):
    """
    신호등 상태 강제 변경 (User → SIM).

    send_data([traffic_index, traffic_status]):
        traffic_index  : str (신호등 ID)
        traffic_status : int (TL_* 상수, 예: 16 = GREEN)
    """

    _HEADER = HEADER_TL  # b'#TrafficLight$'

    def _pack(self, data) -> bytes:
        tl_index  = str(data[0])
        tl_status = int(data[1])

        # ── Header ──
        header = self._HEADER
        header += struct.pack('<i', 14)        # data_length
        header += struct.pack('<iii', 0, 0, 0) # aux_data (12 bytes)

        # ── Payload (14 bytes) ──
        # traffic_index: 12 bytes (ASCII, null-padded)
        index_bytes = tl_index.encode('ascii')[:12]
        index_bytes = index_bytes.ljust(12, b'\x00')

        payload = index_bytes + struct.pack('<h', tl_status)

        # ── Tail ──
        tail = b'\r\n'

        return header + payload + tail
