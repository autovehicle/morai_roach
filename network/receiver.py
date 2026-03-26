"""
MORAI UDP 수신기 — Ego / Object / TrafficLight.

MORAI SIM:Drive 24.R2 UDP 바이너리 프로토콜을 파싱한다.
각 수신기는 데몬 스레드에서 recvfrom 루프를 돌며,
패킷이 도착할 때마다 콜백 함수에 파싱된 데이터를 전달한다.

프로토콜 공통 구조:
  [Header ASCII] [data_length: int32] [aux: 12 bytes] [payload ...]

참고: https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1
"""
import socket
import struct
import threading

from .protocol import (
    HEADER_EGO, HEADER_OBJ, HEADER_TL,
    EgoState, ObjectData, TrafficLightData,
)

# ═══════════════════════════════════════════════════════════════════
# 기본 수신기
# ═══════════════════════════════════════════════════════════════════

class _BaseReceiver:
    """UDP 소켓 바인드 + 데몬 스레드 수신 루프.

    서브클래스는 _parse(raw_bytes) 를 구현해야 한다.
    파싱 결과가 None 이 아니면 callback(result) 을 호출한다.
    """

    def __init__(self, ip: str, port: int, callback):
        self._callback = callback
        self._running  = True

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((ip, port))
        self._sock.settimeout(1.0)        # 1초 타임아웃 (종료 체크용)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── 수신 루프 ──────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            result = self._parse(raw)
            if result is not None:
                self._callback(result)

    # ── 서브클래스 구현 ────────────────────────────────────────────

    def _parse(self, raw: bytes):
        raise NotImplementedError

    # ── 종료 ──────────────────────────────────────────────────────

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
# Ego Vehicle Status 수신기
# ═══════════════════════════════════════════════════════════════════

class EgoReceiver(_BaseReceiver):
    """
    Ego Vehicle Status (SIM → User).

    패킷 바이트 레이아웃 (24.R2, data_length=216):
    ┌──────────┬──────┬────────────────────────────────────────────┐
    │ Offset   │ Size │ Field                                      │
    ├──────────┼──────┼────────────────────────────────────────────┤
    │  0 ~ 10  │  11  │ Header  "#MoraiInfo$"                      │
    │ 11 ~ 14  │   4  │ data_length (int32)                        │
    │ 15 ~ 26  │  12  │ aux_data (reserved)                        │
    │ 27 ~ 30  │   4  │ timestamp_sec   (float)                    │
    │ 31 ~ 34  │   4  │ timestamp_nsec  (float)                    │
    │ 35       │   1  │ ctrl_mode       (int8)                     │
    │ 36       │   1  │ gear            (int8)                     │
    │ 37 ~ 40  │   4  │ signed_vel      (float, km/h)              │
    │ 41 ~ 44  │   4  │ map_id          (int32)                    │
    │ 45 ~ 48  │   4  │ accel           (float, 0-1)               │
    │ 49 ~ 52  │   4  │ brake           (float, 0-1)               │
    │ 53 ~ 64  │  12  │ size_x/y/z      (float×3, m)               │
    │ 65 ~ 76  │  12  │ overhang/wheelbase/rear_overhang (float×3) │
    │ 77 ~ 88  │  12  │ pos_x/y/z       (float×3, m)               │
    │ 89 ~100  │  12  │ roll/pitch/yaw  (float×3, deg)             │
    │101 ~112  │  12  │ vel_x/y/z       (float×3, km/h)            │
    │113 ~124  │  12  │ ang_vel_x/y/z   (float×3, deg/s) [24.R1+] │
    │125 ~136  │  12  │ acc_x/y/z       (float×3, m/s²)            │
    │137 ~140  │   4  │ front_steer     (float, deg)               │
    │141 ~144  │   4  │ rear_steer      (float, deg)               │
    │145 ~182  │  38  │ link_id         (string)                   │
    │  ...     │      │ (추가 필드: tire force, lane dist 등)       │
    └──────────┴──────┴────────────────────────────────────────────┘
    """

    _HEADER     = HEADER_EGO         # b'#MoraiInfo$'
    _HEADER_LEN = len(_HEADER)       # 11
    _DATA_START = 27                 # header(11) + data_len(4) + aux(12)
    _MIN_PKT    = 141                # front_steer(137~140)까지 최소 크기

    def _parse(self, raw: bytes):
        if len(raw) < self._MIN_PKT:
            return None
        if raw[:self._HEADER_LEN] != self._HEADER:
            return None

        # data_length 로 프로토콜 버전 판별 (ang_vel 유무)
        data_len = struct.unpack_from('<i', raw, self._HEADER_LEN)[0]

        def f(offset):
            """offset 에서 little-endian float 1개 읽기"""
            return struct.unpack_from('<f', raw, offset)[0]

        def b(offset):
            """offset 에서 signed int8 1개 읽기"""
            return struct.unpack_from('<b', raw, offset)[0]

        ego = EgoState()

        # ── 제어 ──
        ego.ctrl_mode     = b(35)
        ego.gear          = b(36)
        ego.signed_vel    = f(37)
        ego.accel         = f(45)
        ego.brake         = f(49)

        # ── 차량 치수 ──
        ego.size_x        = f(53)
        ego.size_y        = f(57)
        ego.size_z        = f(61)
        ego.overhang      = f(65)
        ego.wheelbase     = f(69)
        ego.rear_overhang = f(73)

        # ── 위치 ──
        ego.pos_x         = f(77)
        ego.pos_y         = f(81)
        ego.pos_z         = f(85)

        # ── 자세 ──
        ego.roll          = f(89)
        ego.pitch         = f(93)
        ego.yaw           = f(97)

        # ── 속도 ──
        ego.vel_x         = f(101)
        ego.vel_y         = f(105)
        ego.vel_z         = f(109)

        # ── 조향각 (ang_vel 유무에 따라 오프셋 상이) ──
        if data_len >= 216:
            ego.front_steer = f(137)    # ang_vel 있음 (24.R1+)
        else:
            ego.front_steer = f(125)    # ang_vel 없음 (구버전)

        return ego


# ═══════════════════════════════════════════════════════════════════
# Object Info 수신기  (주변 차량 · 보행자 · 장애물)
# ═══════════════════════════════════════════════════════════════════

class ObjectReceiver(_BaseReceiver):
    """
    Object Info (SIM → User).

    ego 기준 가까운 순으로 최대 20개 객체 전송.
    obj_id == 0 인 슬롯은 빈 슬롯 → 필터링.

    ── Per-Object 바이트 레이아웃 (68 bytes 숫자 + 38 bytes link_id) ──
    ┌────────┬──────┬──────────────────────────────┐
    │ Offset │ Size │ Field                        │
    ├────────┼──────┼──────────────────────────────┤
    │  +0    │   2  │ obj_id       (int16)         │
    │  +2    │   2  │ obj_type     (int16)         │
    │  +4    │  12  │ pos_x/y/z   (float×3, m)    │
    │ +16    │   4  │ heading      (float, deg)    │
    │ +20    │  12  │ size_x/y/z  (float×3, m)    │
    │ +32    │   4  │ overhang     (float, m)      │
    │ +36    │   4  │ wheelbase    (float, m)      │
    │ +40    │   4  │ rear_overhang(float, m)      │
    │ +44    │  12  │ vel_x/y/z   (float×3, km/h) │
    │ +56    │  12  │ acc_x/y/z   (float×3, m/s²) │
    │ +68    │  38  │ link_id     (string) [opt]   │
    └────────┴──────┴──────────────────────────────┘
    """

    _HEADER       = HEADER_OBJ          # b'#MoraiObjInfo$'
    _HEADER_LEN   = len(_HEADER)        # 14
    _OBJ_START    = 38                  # header(14)+data_len(4)+aux(12)+ts(8)
    _NUMERIC_SIZE = 68                  # 숫자 데이터 부분
    _LINK_ID_SIZE = 38                  # link_id 문자열
    _OBJ_FULL     = _NUMERIC_SIZE + _LINK_ID_SIZE  # 106 (link_id 포함)
    _MAX_OBJECTS  = 20

    # Per-object struct 포맷 (68 bytes 숫자 부분)
    _OBJ_FMT = '<hh fff f fff f f f fff fff'

    def _parse(self, raw: bytes):
        if len(raw) < self._OBJ_START:
            return None
        if raw[:self._HEADER_LEN] != self._HEADER:
            return None

        remaining = len(raw) - self._OBJ_START

        # ── per-object 크기 판별: 106(link_id 포함) vs 68(숫자만) ──
        # \r\n tail 가능성 고려
        effective = remaining
        if effective >= 2 and raw[-2:] == b'\r\n':
            effective -= 2

        if effective > 0 and effective % self._OBJ_FULL == 0:
            stride = self._OBJ_FULL          # 106 bytes
        elif effective > 0 and effective % self._NUMERIC_SIZE == 0:
            stride = self._NUMERIC_SIZE      # 68 bytes
        else:
            stride = self._OBJ_FULL          # 기본값: 최신 포맷 가정

        num_objects = min(effective // stride, self._MAX_OBJECTS)

        objects = []
        for i in range(num_objects):
            off = self._OBJ_START + i * stride
            if off + self._NUMERIC_SIZE > len(raw):
                break

            vals = struct.unpack_from(self._OBJ_FMT, raw, off)

            obj_id = vals[0]
            if obj_id == 0:                  # 빈 슬롯 무시
                continue

            objects.append(ObjectData(
                obj_id   = obj_id,
                obj_type = vals[1],
                pos_x    = vals[2],
                pos_y    = vals[3],
                pos_z    = vals[4],
                heading  = vals[5],          # deg
                size_x   = vals[6],          # m  (전체 길이)
                size_y   = vals[7],          # m  (전체 너비)
                size_z   = vals[8],          # m  (전체 높이)
                overhang = vals[9],
                wheelbase     = vals[10],
                rear_overhang = vals[11],
                vel_x    = vals[12],         # km/h
                vel_y    = vals[13],
                vel_z    = vals[14],
                acc_x    = vals[15],         # m/s²
                acc_y    = vals[16],
                acc_z    = vals[17],
            ))

        return objects


# ═══════════════════════════════════════════════════════════════════
# Traffic Light 수신기
# ═══════════════════════════════════════════════════════════════════

class TrafficLightReceiver(_BaseReceiver):
    """
    Traffic Light Status (SIM → User).

    바이트 레이아웃:
    ┌──────────┬──────┬──────────────────────────────┐
    │ Offset   │ Size │ Field                        │
    ├──────────┼──────┼──────────────────────────────┤
    │  0 ~ 13  │  14  │ Header "#TrafficLight$"      │
    │ 14 ~ 17  │   4  │ data_length (int32, =16)     │
    │ 18 ~ 29  │  12  │ aux_data (reserved)          │
    │ 30 ~ 41  │  12  │ traffic_index (string)       │
    │ 42 ~ 43  │   2  │ traffic_type  (int16)        │
    │ 44 ~ 45  │   2  │ traffic_status (int16)       │
    └──────────┴──────┴──────────────────────────────┘
    """

    _HEADER     = HEADER_TL            # b'#TrafficLight$'
    _HEADER_LEN = len(_HEADER)         # 14
    _MIN_PKT    = 46

    def _parse(self, raw: bytes):
        if len(raw) < self._MIN_PKT:
            return None
        if raw[:self._HEADER_LEN] != self._HEADER:
            return None

        # traffic_index: 12-byte ASCII (공백/null 패딩)
        raw_index = raw[30:42]
        index_str = raw_index.decode('ascii', errors='ignore').rstrip('\x00 ')

        tl_type, tl_status = struct.unpack_from('<hh', raw, 42)

        return TrafficLightData(
            index   = index_str,
            tl_type = tl_type,
            status  = tl_status,
        )
