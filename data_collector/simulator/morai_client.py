"""
simulator/morai_client.py
--------------------------
MORAI 시뮬레이터 클라이언트.

현재는 Mock(더미) 구현만 있음.
gRPC 실제 구현은 별도 파일을 이 파일과 교체하는 방식으로 연동.

다른 팀이 구현할 때 지켜야 할 함수 목록 및 입출력 타입:

    connect()                                              -> bool
    disconnect()                                           -> None
    reset_map(map_name: str)                               -> bool
    spawn_ego(x, y, yaw, z=0.0)                            -> bool
    spawn_npcs(count, spawn_x_range, spawn_y_range)        -> bool
    spawn_pedestrians(count, spawn_x_range, spawn_y_range) -> bool
    set_weather(weather: str, time_of_day: str)            -> bool
    send_control(steer, throttle, brake)                   -> bool
    get_global_path(start_x, start_y, goal_x, goal_y)     -> (np.ndarray shape(N,2), List[str])
    setup_episode(params: EpisodeParams, map_name: str)    -> (np.ndarray shape(N,2), List[str])

weather 가능값    : "clear" | "rain" | "fog" | "snow"
time_of_day 가능값: "day"   | "dusk" | "night"
steer    범위     : [-1.0, 1.0]
throttle 범위     : [0.0,  1.0]
brake    범위     : [0.0,  1.0]
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from core.scenario_params import EpisodeParams


class MoraiClient:
    """
    Mock 구현체.
    실제 시뮬레이터 없이도 collect.py / episode_manager.py가 동작하도록
    모든 함수가 적절 반환값을 돌려줌.
    """

    def __init__(self, config: dict):
        self.host      = config["morai"]["host"]
        self.port      = config["morai"]["grpc_port"]
        self.ego_model = config["morai"]["ego_vehicle"]

    # ── 연결 관리 ──────────────────────────────────────────────

    def connect(self) -> bool:
        print(f"[MoraiClient] Mock 모드 → {self.host}:{self.port}")
        return True

    def disconnect(self):
        pass

    # ── 시뮬레이터 제어 ────────────────────────────────────────

    def reset_map(self, map_name: str) -> bool:
        print(f"[MoraiClient] reset_map: {map_name}")
        time.sleep(0.5)
        return True

    def spawn_ego(self, x: float, y: float, yaw: float,
                  z: float = 0.0) -> bool:
        print(f"[MoraiClient] spawn_ego: ({x:.1f}, {y:.1f}, yaw={yaw:.1f})")
        return True

    def spawn_npcs(self, count: int,
                   spawn_x_range: Tuple[float, float],
                   spawn_y_range: Tuple[float, float]) -> bool:
        print(f"[MoraiClient] spawn_npcs: {count}대")
        return True

    def spawn_pedestrians(self, count: int,
                          spawn_x_range: Tuple[float, float],
                          spawn_y_range: Tuple[float, float]) -> bool:
        print(f"[MoraiClient] spawn_pedestrians: {count}명")
        return True

    def set_weather(self, weather: str, time_of_day: str) -> bool:
        print(f"[MoraiClient] set_weather: {weather}, {time_of_day}")
        return True

    def send_control(self, steer: float,
                     throttle: float, brake: float) -> bool:
        """
        steer    : [-1.0, 1.0]
        throttle : [0.0,  1.0]
        brake    : [0.0,  1.0]
        """
        return True

    def get_global_path(self, start_x: float, start_y: float,
                        goal_x: float, goal_y: float
                        ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        반환:
            waypoints : np.ndarray shape (N, 2)  ← x, y 좌표
            link_ids  : List[str]                ← 도로 링크 ID
        """
        n         = 50
        xs        = np.linspace(start_x, goal_x, n)
        ys        = np.linspace(start_y, goal_y, n)
        waypoints = np.stack([xs, ys], axis=1).astype(np.float32)
        link_ids  = [f"link_{i:03d}" for i in range(n)]
        return waypoints, link_ids

    # ── 에피소드 전체 세팅 (편의 함수) ─────────────────────────

    def setup_episode(self, params: EpisodeParams,
                      map_name: str
                      ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        에피소드 시작 전 시뮬레이터 전체 세팅.
            1. 맵 리셋
            2. 날씨 / 시간대 설정
            3. 자차 스폰
            4. NPC / 보행자 스폰
            5. 전역 경로 요청
        반환: (waypoints, link_ids)  ← 실패 시 (None, None)
        """
        print(f"[MoraiClient] 시뮬 세팅 → "
              f"zone={params.zone}, scenario={params.scenario}, ep={params.episode_id}")

        if not self.reset_map(map_name):
            print("[MoraiClient] 맵 리셋 실패")
            return None, None

        time.sleep(1.0)

        self.spawn_ego(params.start_x, params.start_y, params.start_yaw)

        npc_x = (params.start_x - 50, params.start_x + 50)
        npc_y = (params.start_y - 50, params.start_y + 50)

        if params.npc_count > 0:
            self.spawn_npcs(params.npc_count, npc_x, npc_y)
        if params.pedestrian_count > 0:
            self.spawn_pedestrians(params.pedestrian_count, npc_x, npc_y)

        return self.get_global_path(
            params.start_x, params.start_y,
            params.goal_x,  params.goal_y,
        )
