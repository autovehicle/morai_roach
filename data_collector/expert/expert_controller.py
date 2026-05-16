"""
expert/expert_controller.py
----------------------------
Rule-based Expert Controller — stub 상태.

실제 expert 로직은 추후 구현 예정.
현재는 step()이 항상 [0.0, 0.0, 0.0] 을 반환하며,
.npz 저장 시 expert 필드는 빈 값(0)으로 기록됨.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ControlOutput:
    steer:    float  # [-1.0, 1.0]
    throttle: float  # [0.0,  1.0]
    brake:    float  # [0.0,  1.0]


class ExpertController:

    def __init__(self, config: dict = None):
        pass

    def reset(self, waypoints: np.ndarray):
        # TODO: expert 구현 시 경로 초기화 로직 추가
        pass

    def step(self, ego, tl_states: List, dt: float = 0.1) -> ControlOutput:
        # TODO: rule-based expert 구현
        return ControlOutput(steer=0.0, throttle=0.0, brake=0.0)

    @staticmethod
    def detect_longtail(ego, prev_speed: float, dt: float,
                        tl_states: List, triggers: List[str]) -> bool:
        # TODO: longtail 감지 로직 구현
        return False
