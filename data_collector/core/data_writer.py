"""
core/data_writer.py
-------------------
센서 스냅샷을 타임스탬프 동기화 후 .npz 프레임 파일로 저장.

폴더 구조:
  {root}/{zone}/{scenario}/episode_{NNN}/
      frames/
          000001.npz
          000002.npz
          ...
      episode_params.json
      episode_summary.json
"""

"""
1. 데이터 컨테이너 정의
CameraFrame     → 카메라 이미지 1장
GNSSData        → GPS 위치/속도
IMUData         → 가속도/각속도
EgoState        → 자차 위치/속도/조향
GTObject        → NPC 차량/보행자/장애물
GTLane          → 차선 정보
TrafficLightState → 신호등 상태
SensorSnapshot  → 위 전부를 한 타임스탬프에 묶은 것

2. BEV 맵 생성기
자차 중심 기준으로 30m x 20m 영역을 0.2m
해상도로 8채널 BEV 맵 생성.

3. DataWriter 클래스
- begin_episode(params) → 에피소드 폴더 생성 및 episode_params.json 저장
- write_frame(snapshot) → SensorSnapshot을 .npz 파일로 저장
- end_episode(success, reason) → episode_summary.json 저장 및 수집 현황 업데이트
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.scenario_params import EpisodeParams


# ─────────────────────────────────────────────
# 데이터 컨테이너
# ─────────────────────────────────────────────

@dataclass
class CameraFrame:
    """카메라 1개 뷰 RGB 이미지 (H, W, 3) uint8"""
    timestamp_ns: int
    image: np.ndarray          # shape: (H, W, 3)


@dataclass
class GNSSData:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float
    velocity_x: float          # m/s, 차량 좌표계
    velocity_y: float
    velocity_z: float


@dataclass
class IMUData:
    timestamp_ns: int
    accel_x: float             # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float              # rad/s
    gyro_y: float
    gyro_z: float


@dataclass
class EgoState:
    timestamp_ns: int
    x: float
    y: float
    z: float
    yaw: float                 # rad
    speed: float               # m/s
    steer: float               # rad


@dataclass
class GTObject:
    obj_id: int
    obj_type: str              # "vehicle" | "pedestrian" | "static"
    x: float
    y: float
    z: float
    vel_x: float
    vel_y: float
    heading: float


@dataclass
class GTLane:
    lane_id: str
    boundary_left: np.ndarray  # shape: (N, 2)  x,y 좌표 배열
    boundary_right: np.ndarray
    lane_type: str             # "solid" | "dashed" | "center_yellow"


@dataclass
class TrafficLightState:
    tl_id: str
    state: int                 # 0~9 클래스


@dataclass
class SensorSnapshot:
    """매 타임스텝에 동기화된 전체 센서 묶음"""
    frame_id: int
    timestamp_ns: int

    cameras: Dict[str, CameraFrame]     # key: "front", "front_left", ...
    gnss: Optional[GNSSData]
    imu: Optional[IMUData]
    ego: Optional[EgoState]

    gt_objects: List[GTObject] = field(default_factory=list)
    gt_lanes: List[GTLane]     = field(default_factory=list)
    tl_states: List[TrafficLightState] = field(default_factory=list)

    expert_steer: float    = 0.0
    expert_throttle: float = 0.0
    expert_brake: float    = 0.0

    nav_waypoints: Optional[np.ndarray] = None   # shape: (N, 2)
    nav_link_ids:  Optional[List[str]]  = None

    bev_map: Optional[np.ndarray] = None         # shape: (H, W, 8)

    is_longtail: bool = False


# ─────────────────────────────────────────────
# BEV 맵 생성기
# ─────────────────────────────────────────────

class BEVMapGenerator:
    """
    자차 중심 기준으로 8채널 BEV 맵을 생성.

    영역: 전방 30m / 후방 10m / 좌우 20m
    해상도: 0.2m/px → 200px(세로) × 200px(가로)

    채널:
        0: background
        1: center yellow line
        2: solid line
        3: dashed line
        4: stop line
        5: dynamic objects (vehicles, pedestrians)
        6: drivable area
        7: crosswalk
    """

    FRONT_M  = 30.0
    REAR_M   = 10.0
    SIDE_M   = 20.0
    RES_M    = 0.2     # 픽셀당 m

    def __init__(self):
        self.h = int((self.FRONT_M + self.REAR_M) / self.RES_M)   # 200
        self.w = int((self.SIDE_M * 2) / self.RES_M)              # 200

    def generate(self, ego: EgoState,
                 gt_objects: List[GTObject],
                 gt_lanes: List[GTLane]) -> np.ndarray:
        """
        반환: (H, W, 8) float32 배열 (각 채널 0 또는 1)
        """
        bev = np.zeros((self.h, self.w, 8), dtype=np.float32)

        # 채널 6: 전체 주행 가능 영역 (기본값으로 중앙 영역 fill)
        bev[:, :, 6] = 1.0

        # 채널 1~4: 차선 그리기
        for lane in gt_lanes:
            self._draw_lane(bev, ego, lane)

        # 채널 5: 동적 객체
        for obj in gt_objects:
            if obj.obj_type in ("vehicle", "pedestrian"):
                self._draw_object(bev, ego, obj)

        # 채널 0: 나머지 background (아무 채널도 없는 픽셀)
        occupied = bev[:, :, 1:].sum(axis=2) > 0
        bev[~occupied, 0] = 1.0

        return bev

    def _world_to_bev(self, ego: EgoState,
                      wx: float, wy: float):
        """월드 좌표 → BEV 픽셀 좌표 (행, 열)"""
        dx = wx - ego.x
        dy = wy - ego.y

        # 자차 yaw 기준으로 로컬 변환
        cos_y = np.cos(-ego.yaw)
        sin_y = np.sin(-ego.yaw)
        local_x =  cos_y * dx - sin_y * dy   # 전방 방향
        local_y =  sin_y * dx + cos_y * dy   # 좌측 방향

        # 픽셀 인덱스 (전방이 행 0, 후방이 행 h-1)
        row = int((self.FRONT_M - local_x) / self.RES_M)
        col = int((self.SIDE_M  + local_y) / self.RES_M)
        return row, col

    def _draw_lane(self, bev: np.ndarray, ego: EgoState, lane: GTLane):
        lane_type_to_ch = {
            "center_yellow": 1,
            "solid":         2,
            "dashed":        3,
            "stop_line":     4,
            "crosswalk":     7,
        }
        ch = lane_type_to_ch.get(lane.lane_type, 2)

        for pts in [lane.boundary_left, lane.boundary_right]:
            if pts is None or len(pts) == 0:
                continue
            for pt in pts:
                r, c = self._world_to_bev(ego, pt[0], pt[1])
                if 0 <= r < self.h and 0 <= c < self.w:
                    bev[r, c, ch] = 1.0

    def _draw_object(self, bev: np.ndarray, ego: EgoState, obj: GTObject):
        r, c = self._world_to_bev(ego, obj.x, obj.y)
        # 객체 중심 ±2픽셀 박스
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    bev[rr, cc, 5] = 1.0


# ─────────────────────────────────────────────
# DataWriter
# ─────────────────────────────────────────────

class DataWriter:
    """
    에피소드 시작/종료 및 프레임 단위 .npz 저장을 담당.

    사용 예시:
        writer = DataWriter(config)
        writer.begin_episode(params)
        writer.write_frame(snapshot)
        ...
        summary = writer.end_episode(success=True, reason="goal_reached")
    """

    CAMERA_KEYS = ["front", "front_left", "front_right", "rear_left", "rear_right"]

    def __init__(self, config: dict):
        self.root      = Path(config["dataset"]["root_dir"])
        self.bev_gen   = BEVMapGenerator()

        self._ep_dir: Optional[Path] = None
        self._frame_dir: Optional[Path] = None
        self._frame_id: int = 0
        self._ep_params: Optional[EpisodeParams] = None
        self._ep_start_time: float = 0.0

        # 수집 현황 (zone/scenario별 카운트)
        self._summary: Dict[str, Dict[str, int]] = {}

    # ── 에피소드 관리 ───────────────────────────────────────

    def begin_episode(self, params: EpisodeParams) -> Path:
        ep_str   = f"episode_{params.episode_id:03d}"
        zone_str = f"zone_{params.zone}"
        scen_str = f"scenario_{params.scenario}"

        self._ep_dir    = self.root / zone_str / scen_str / ep_str
        self._frame_dir = self._ep_dir / "frames"
        self._frame_dir.mkdir(parents=True, exist_ok=True)

        self._frame_id      = 0
        self._ep_params     = params
        self._ep_start_time = time.time()

        # episode_params.json 저장
        params_path = self._ep_dir / "episode_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)

        return self._ep_dir

    def end_episode(self, success: bool, reason: str) -> dict:
        elapsed = time.time() - self._ep_start_time
        summary = {
            "success":     success,
            "reason":      reason,
            "total_frames": self._frame_id,
            "duration_sec": round(elapsed, 2),
            "zone":        self._ep_params.zone,
            "scenario":    self._ep_params.scenario,
            "episode_id":  self._ep_params.episode_id,
        }

        summary_path = self._ep_dir / "episode_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 전체 현황 업데이트
        key = f"{self._ep_params.zone}/{self._ep_params.scenario}"
        self._summary.setdefault(key, {"total": 0, "success": 0, "frames": 0})
        self._summary[key]["total"]   += 1
        self._summary[key]["success"] += int(success)
        self._summary[key]["frames"]  += self._frame_id

        return summary

    def episode_exists(self, zone: str, scenario: str, episode_id: int) -> bool:
        """--resume 옵션 시 이미 완료된 에피소드 스킵용"""
        ep_str   = f"episode_{episode_id:03d}"
        zone_str = f"zone_{zone}"
        scen_str = f"scenario_{scenario}"
        summary  = self.root / zone_str / scen_str / ep_str / "episode_summary.json"
        return summary.exists()

    # ── 프레임 저장 ─────────────────────────────────────────

    def write_frame(self, snap: SensorSnapshot):
        """SensorSnapshot 하나를 .npz 1파일로 저장"""
        self._frame_id += 1
        fname = self._frame_dir / f"{self._frame_id:06d}.npz"

        # ── BEV 맵 생성 (ego + GT 정보 필요) ──────────────────
        if snap.ego is not None:
            snap.bev_map = self.bev_gen.generate(
                snap.ego, snap.gt_objects, snap.gt_lanes
            )

        # ── 카메라 이미지 묶기 ─────────────────────────────────
        cameras = {}
        for key in self.CAMERA_KEYS:
            cam = snap.cameras.get(key)
            cameras[key] = cam.image if cam is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # ── GT 객체 배열 변환 (타입별 분리) ───────────────────────
        # gt_vehicles    : (N, 6)  [id, x, y, z, vel_x, vel_y]
        # gt_pedestrians : (M, 6)  [id, x, y, z, vel_x, vel_y]
        # gt_static      : (K, 4)  [id, x, y, z]  <- 속도 없음

        vehicles    = [o for o in snap.gt_objects if o.obj_type == "vehicle"]
        pedestrians = [o for o in snap.gt_objects if o.obj_type == "pedestrian"]
        statics     = [o for o in snap.gt_objects if o.obj_type == "static"]

        gt_vehicles = np.array([
            [o.obj_id, o.x, o.y, o.z, o.vel_x, o.vel_y]
            for o in vehicles
        ], dtype=np.float32) if vehicles else np.zeros((0, 6), dtype=np.float32)

        gt_pedestrians = np.array([
            [o.obj_id, o.x, o.y, o.z, o.vel_x, o.vel_y]
            for o in pedestrians
        ], dtype=np.float32) if pedestrians else np.zeros((0, 6), dtype=np.float32)

        gt_static = np.array([
            [o.obj_id, o.x, o.y, o.z]
            for o in statics
        ], dtype=np.float32) if statics else np.zeros((0, 4), dtype=np.float32)

        # ── 신호등 상태 배열 ───────────────────────────────────
        # shape: (M, 2)  [tl_id_hash, state]
        if snap.tl_states:
            tl_arr = np.array([[
                hash(tl.tl_id) & 0xFFFF,
                tl.state
            ] for tl in snap.tl_states], dtype=np.int32)
        else:
            tl_arr = np.zeros((0, 2), dtype=np.int32)

        # ── ego state 배열 ─────────────────────────────────────
        ego = snap.ego
        ego_arr = np.array([
            ego.x, ego.y, ego.z, ego.yaw, ego.speed, ego.steer
        ], dtype=np.float32) if ego else np.zeros(6, dtype=np.float32)

        # ── GNSS 배열 ──────────────────────────────────────────
        gnss = snap.gnss
        gnss_arr = np.array([
            gnss.latitude, gnss.longitude, gnss.altitude,
            gnss.velocity_x, gnss.velocity_y, gnss.velocity_z
        ], dtype=np.float32) if gnss else np.zeros(6, dtype=np.float32)

        # ── IMU 배열 ───────────────────────────────────────────
        imu = snap.imu
        imu_arr = np.array([
            imu.accel_x, imu.accel_y, imu.accel_z,
            imu.gyro_x,  imu.gyro_y,  imu.gyro_z
        ], dtype=np.float32) if imu else np.zeros(6, dtype=np.float32)

        # ── nav waypoints ──────────────────────────────────────
        nav_wp = snap.nav_waypoints if snap.nav_waypoints is not None \
                 else np.zeros((0, 2), dtype=np.float32)

        # ── expert 제어 ────────────────────────────────────────
        expert_arr = np.array([
            snap.expert_steer,
            snap.expert_throttle,
            snap.expert_brake
        ], dtype=np.float32)

        # ── .npz 저장 ──────────────────────────────────────────
        save_dict = {
            "timestamp_ns":   np.array([snap.timestamp_ns], dtype=np.int64),
            "frame_id":       np.array([snap.frame_id],     dtype=np.int32),
            "is_longtail":    np.array([snap.is_longtail],  dtype=bool),
            # 카메라
            "cam_front":        cameras["front"],
            "cam_front_left":   cameras["front_left"],
            "cam_front_right":  cameras["front_right"],
            "cam_rear_left":    cameras["rear_left"],
            "cam_rear_right":   cameras["rear_right"],
            # 센서
            "ego":     ego_arr,
            "gnss":    gnss_arr,
            "imu":     imu_arr,
            # GT
            "gt_vehicles":     gt_vehicles,
            "gt_pedestrians":  gt_pedestrians,
            "gt_static":       gt_static,
            "tl_states":   tl_arr,
            # 경로
            "nav_waypoints": nav_wp,
            # expert 제어
            "expert": expert_arr,
            # BEV
            "bev_map": snap.bev_map if snap.bev_map is not None
                       else np.zeros((200, 200, 8), dtype=np.float32),
        }

        np.savez_compressed(str(fname), **save_dict)

    # ── 수집 현황 출력 ──────────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 55)
        print(f"{'Zone/Scenario':<35} {'에피소드':>6} {'성공':>5} {'프레임':>8}")
        print("-" * 55)
        for key, val in self._summary.items():
            print(f"{key:<35} {val['total']:>6} {val['success']:>5} {val['frames']:>8}")
        print("=" * 55)
