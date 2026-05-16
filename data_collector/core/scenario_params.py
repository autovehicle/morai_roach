"""
core/scenario_params.py
-----------------------
에피소드마다 랜덤 파라미터(EpisodeParams)를 생성한다.
config의 zone/scenario 설정을 읽어 spawn 위치, NPC 수 등을 샘플링.
"""

"""
1. 에피소드 1개의 설정값을 담는 EpisodeParams dataclass 정의
- zone, scenario, episode_id
- 자차 시작 위치/방향
- 목표 지점
- NPC/보행자 수
- longtail 여부 + 트리거 상황 리스트
- 시뮬 최대 스텝 수

=> 이게 채워지면 episode_manager.py에서 시뮬 세팅, expert 초기화 등에 활용

2. ScenarioParamGenerator 클래스 정의
- collection_config.yaml의 zones/scenarios 설정을 읽어 EpisodeParams를 랜덤 생성
- generate(zone, scenario, episode_id) 메서드에서 zone/scenario에 맞는 파라미터 범위를 읽어 샘플링 후 EpisodeParams 반환

"""

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeParams:
    zone: str
    scenario: str
    episode_id: int

    # 자차 시작
    start_x: float
    start_y: float
    start_yaw: float

    # 목표 지점
    goal_x: float
    goal_y: float

    # 환경
    npc_count: int
    pedestrian_count: int

    # 저장 Hz 결정
    is_longtail: bool
    longtail_triggers: List[str] = field(default_factory=list)

    # 시뮬 설정
    max_steps: int = 1000

    def to_dict(self) -> dict:
        return {
            "zone": self.zone,
            "scenario": self.scenario,
            "episode_id": self.episode_id,
            "start": {"x": self.start_x, "y": self.start_y, "yaw": self.start_yaw},
            "goal":  {"x": self.goal_x,  "y": self.goal_y},
            "npc_count": self.npc_count,
            "pedestrian_count": self.pedestrian_count,
            "is_longtail": self.is_longtail,
            "longtail_triggers": self.longtail_triggers,
            "max_steps": self.max_steps,
        }


class ScenarioParamGenerator:
    """
    config dict를 받아 EpisodeParams를 랜덤 생성.

    사용 예시:
        generator = ScenarioParamGenerator(config)
        params = generator.generate("urban", "traffic_jam", episode_id=1)
    """

    def __init__(self, config: dict):
        self.config = config

    def generate(self, zone: str, scenario: str, episode_id: int) -> EpisodeParams:
        zone_cfg     = self.config["zones"][zone]
        scenario_cfg = self.config["scenarios"][scenario]

        # ── 자차 시작 위치 (랜덤) ───────────────────────────────────────────
        start_x   = random.uniform(*zone_cfg["spawn_ranges"]["x"])
        start_y   = random.uniform(*zone_cfg["spawn_ranges"]["y"])
        start_yaw = random.uniform(*zone_cfg["spawn_ranges"]["yaw"])

        # ── 목표 지점 (랜덤) ────────────────────────────────────────────────
        goal_x = random.uniform(*zone_cfg["goal_ranges"]["x"])
        goal_y = random.uniform(*zone_cfg["goal_ranges"]["y"])

        # ── NPC / 보행자 수 (랜덤) ──────────────────────────────────────────
        npc_count        = random.randint(*scenario_cfg["npc_count"])
        pedestrian_count = random.randint(*scenario_cfg["pedestrian_count"])

        # ── longtail 여부 ─────────────────────────────────────────────────
        triggers    = scenario_cfg.get("longtail_triggers", [])
        is_longtail = len(triggers) > 0

        return EpisodeParams(
            zone=zone,
            scenario=scenario,
            episode_id=episode_id,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            npc_count=npc_count,
            pedestrian_count=pedestrian_count,
            is_longtail=is_longtail,
            longtail_triggers=triggers,
            max_steps=zone_cfg.get("max_steps", 1000),
        )
