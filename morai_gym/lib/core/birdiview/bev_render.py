"""
BEV (Bird's Eye View) 동적 객체 렌더러.

MORAI 시뮬레이터에서 UDP로 수신한 동적 GT 데이터(차량, 보행자, 신호등)를
BEV 이미지에 마스킹한다.

Roach 논문 (carla-roach)의 chauffeurnet.py BEV 표현 방식을 그대로 따름:
  - 192×192 픽셀, 5 pixels/meter
  - Ego 차량: 이미지 하단에서 40px 위(y=152), 좌우 중앙(x=96)
  - Ego heading → 이미지 상단 방향
  - 차량: 5각형(pointed front) filled polygon, 히스토리 포함
  - 보행자: 4각형 filled polygon, bbox 2× 스케일, 히스토리 포함
  - 신호등: 상태별 밝기값 (green=80, yellow=170, red=255), 히스토리 포함

채널 구성 (Roach 원본과 동일):
  masks shape: (3 * len(history_idx), H, W)
    = [vehicle_t-3, vehicle_t-2, vehicle_t-1, vehicle_t-0,
       walker_t-3,  walker_t-2,  walker_t-1,  walker_t-0,
       tl_t-3,      tl_t-2,      tl_t-1,      tl_t-0]

좌표계 가정 (MORAI):
  - 세계 좌표: x=동쪽(East), y=북쪽(North)  [ENU 좌표계]
  - yaw/heading: 동쪽(East)축 기준, 반시계 방향 양수 (degrees)
  - 이 가정이 맞지 않으면 _get_warp_transform() 수정 필요

참고:
  - 정적 데이터(도로, 경로, 차선)는 현재 미구현 (시뮬레이션 측 이슈)
  - 신호등 stopline 위치 정보는 MORAI에서 미제공 → BEV 상단 바(bar)로 표시
"""

import numpy as np
import cv2 as cv
from collections import deque
from dataclasses import replace
from typing import List, Optional

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from network.UDP.protocol import (
    ObjectData, TrafficLightData, EgoState,
    OBJ_TYPE_VEHICLE, OBJ_TYPE_PEDESTRIAN,
    TL_RED, TL_YELLOW, TL_GREEN,
    TL_RED_YELLOW, TL_YELLOW_GREEN,
    TL_GREEN_LEFT, TL_YELLOW_GREEN_LEFT,
    TL_GREEN_GREEN_LEFT,
)

# ═══════════════════════════════════════════════════════════════════
# 시각화용 색상 (Roach 원본과 동일, RGB 순서)
# ═══════════════════════════════════════════════════════════════════
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)        # 차량
COLOR_CYAN = (0, 255, 255)      # 보행자
COLOR_GREEN = (0, 255, 0)       # 신호등 - 녹색
COLOR_YELLOW = (255, 255, 0)    # 신호등 - 황색
COLOR_RED = (255, 0, 0)         # 신호등 - 적색
COLOR_WHITE = (255, 255, 255)   # Ego 차량


def _tint(color, factor):
    """색상을 밝게(tint) 만든다. factor=0이면 원색, factor=1이면 흰색."""
    r = min(int(color[0] + (255 - color[0]) * factor), 255)
    g = min(int(color[1] + (255 - color[1]) * factor), 255)
    b = min(int(color[2] + (255 - color[2]) * factor), 255)
    return (r, g, b)


class BEVDynamicRenderer:
    """동적 객체(차량, 보행자, 신호등)를 BEV 이미지에 마스킹하는 렌더러.

    사용법:
        from morai_gym.lib.core.birdiview.bev_render import BEVDynamicRenderer

        renderer = BEVDynamicRenderer()   # 기본값 사용
        # 또는
        renderer = BEVDynamicRenderer.from_config(config)  # Config 객체 사용

        # 매 프레임마다:
        result = renderer.update(ego_state, vehicle_list, pedestrian_list, traffic_light)
        rendered_image = result['rendered']   # (192, 192, 3) uint8 RGB
        mask_channels  = result['masks']      # (12, 192, 192) uint8
    """

    def __init__(
        self,
        width: int = 192,
        pixels_ev_to_bottom: int = 40,
        pixels_per_meter: float = 5.0,
        history_idx: list = None,
        scale_bbox: bool = False,
        walker_bbox_scale: float = 2.0,
        vehicle_distance: float = 20.0,
        pedestrian_distance: float = 15.0,
        tl_green_val: int = 80,
        tl_yellow_val: int = 170,
        tl_red_val: int = 255,
        default_veh_size: tuple = (4.5, 2.0),
        default_ped_size: tuple = (0.5, 0.5),
    ):
        """
        Args:
            width: BEV 이미지 가로/세로 크기 (pixels).
            pixels_ev_to_bottom: Ego 차량 위치 — 이미지 하단으로부터의 거리 (pixels).
            pixels_per_meter: 미터당 픽셀 수.
            history_idx: 히스토리 큐에서 가져올 프레임 인덱스 리스트.
                         [-16, -11, -6, -1] → 10Hz 기준 1.6s, 1.1s, 0.6s, 0.1s 전.
            scale_bbox: True면 차량 bbox를 1.0× 스케일링 (CARLA 호환).
            walker_bbox_scale: 보행자 bbox 확대 계수 (Roach 기본값: 2.0).
            vehicle_distance: 차량 감지 범위 (meters).
            pedestrian_distance: 보행자 감지 범위 (meters).
            tl_green_val: 녹색 신호등 채널 밝기값 (Roach: 80 ≈ 0.3137).
            tl_yellow_val: 황색 신호등 채널 밝기값 (Roach: 170 ≈ 0.6667).
            tl_red_val: 적색 신호등 채널 밝기값 (Roach: 255 = 1.0).
            default_veh_size: MORAI에서 크기 미제공 시 기본 차량 크기 (length, width) meters.
            default_ped_size: MORAI에서 크기 미제공 시 기본 보행자 크기 (length, width) meters.
        """
        self._width = width
        self._ev_to_bottom = pixels_ev_to_bottom
        self._ppm = pixels_per_meter
        self._history_idx = history_idx if history_idx is not None else [-16, -11, -6, -1]
        self._scale_bbox = scale_bbox
        self._walker_scale = walker_bbox_scale
        self._veh_dist = vehicle_distance
        self._ped_dist = pedestrian_distance

        self._tl_green_val = tl_green_val
        self._tl_yellow_val = tl_yellow_val
        self._tl_red_val = tl_red_val

        self._default_veh_size = default_veh_size   # (length, width) m
        self._default_ped_size = default_ped_size

        # 히스토리 큐 — 최대 20프레임 보관 (10Hz × 2초)
        self._history_queue: deque = deque(maxlen=20)

    # ───────────────────────────────────────────────────────────────
    # 팩토리 메서드
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config):
        """map_to_h5.py의 Config 객체로부터 렌더러를 생성한다."""
        return cls(
            width=config.width,
            pixels_ev_to_bottom=config.ev_to_bottom,
            pixels_per_meter=config.ppm,
            history_idx=config.history_idx,
            scale_bbox=config.scale_bbox,
            walker_bbox_scale=config.walker_scale,
            vehicle_distance=config.veh_dist,
            pedestrian_distance=config.ped_dist,
            tl_green_val=config.tl_green,
            tl_yellow_val=config.tl_yellow,
            tl_red_val=config.tl_red,
            default_veh_size=config.veh_size,
            default_ped_size=config.ped_size,
        )

    # ═══════════════════════════════════════════════════════════════
    # 공개 API
    # ═══════════════════════════════════════════════════════════════

    def update(
        self,
        ego_state: EgoState,
        vehicle_list: List[ObjectData],
        pedestrian_list: List[ObjectData],
        traffic_light: Optional[TrafficLightData],
    ) -> dict:
        """현재 프레임 데이터를 히스토리에 추가하고 BEV를 렌더링한다.

        Args:
            ego_state: Ego 차량 상태 (UdpManager.ego_state).
            vehicle_list: 주변 차량 리스트 (UdpManager.vehicle_list).
            pedestrian_list: 보행자 리스트 (UdpManager.pedestrian_list).
            traffic_light: 신호등 상태 (UdpManager.traffic_light). None 가능.

        Returns:
            dict:
                'rendered': (H, W, 3) uint8 — RGB 시각화 이미지.
                'masks': (C, H, W) uint8 — 마스크 채널.
                    C = 3 × len(history_idx).
                    [vehicle_history..., walker_history..., tl_history...].
                    vehicle/walker: 0 또는 255.
                    tl: 0 / 80(green) / 170(yellow) / 255(red).
        """
        # ── 1) 거리 기반 필터링 ──
        vehicles = self._filter_by_distance(
            vehicle_list, ego_state, self._veh_dist)
        pedestrians = self._filter_by_distance(
            pedestrian_list, ego_state, self._ped_dist)

        # ── 2) 보행자 bbox 스케일링 (Roach 원본: 2.0×) ──
        if self._walker_scale != 1.0:
            pedestrians = self._scale_objects(pedestrians, self._walker_scale)

        # ── 3) 히스토리 큐에 저장 ──
        self._history_queue.append((vehicles, pedestrians, traffic_light))

        # ── 4) Affine 변환 행렬 생성 (세계 좌표 → BEV 픽셀) ──
        M_warp = self._get_warp_transform(
            ego_state.pos_x, ego_state.pos_y, ego_state.yaw)

        # ── 5) 히스토리별 마스크 렌더링 ──
        vehicle_masks, walker_masks, tl_masks = self._get_history_masks(M_warp)

        # ── 6) RGB 시각화 이미지 ──
        rendered = self._render_rgb(
            vehicle_masks, walker_masks, tl_masks, ego_state, M_warp)

        # ── 7) 출력 마스크 채널 조합 ──
        c_vehicle = [m.astype(np.uint8) * 255 for m in vehicle_masks]
        c_walker = [m.astype(np.uint8) * 255 for m in walker_masks]
        c_tl = tl_masks  # 이미 uint8 밝기값

        masks = np.stack((*c_vehicle, *c_walker, *c_tl), axis=0)

        return {'rendered': rendered, 'masks': masks}

    def reset(self):
        """히스토리 큐를 초기화한다. 에피소드 시작 시 호출."""
        self._history_queue.clear()

    # ═══════════════════════════════════════════════════════════════
    # 좌표 변환
    # ═══════════════════════════════════════════════════════════════

    def _get_warp_transform(self, ego_x: float, ego_y: float, ego_yaw_deg: float):
        """세계 좌표(m) → BEV 픽셀 아핀 변환 행렬 (2×3)을 생성한다.

        Roach chauffeurnet.py의 _get_warp_transform()과 동일한 원리:
          1. Ego 위치를 기준으로 BEV 영역의 세 꼭짓점을 세계 좌표로 계산.
          2. 대응하는 이미지 픽셀 좌표와 매핑하여 아핀 행렬 생성.

        결과: Ego가 이미지 하단 중앙에 위치하고, heading이 위를 가리킴.
        """
        yaw = np.deg2rad(ego_yaw_deg)

        # Ego의 전방 벡터, 우측 벡터 (세계 좌표)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw - 0.5 * np.pi),
                              np.sin(yaw - 0.5 * np.pi)])

        mpp = 1.0 / self._ppm  # meters per pixel
        ego_pos = np.array([ego_x, ego_y])

        # BEV 이미지의 세 꼭짓점에 해당하는 세계 좌표 (m)
        #
        # bottom_left:  ego에서 뒤로 ev_to_bottom px, 왼쪽으로 width/2 px
        # top_left:     ego에서 앞으로 (width - ev_to_bottom) px, 왼쪽으로 width/2 px
        # top_right:    ego에서 앞으로 (width - ev_to_bottom) px, 오른쪽으로 width/2 px
        bottom_left = (ego_pos
                       - self._ev_to_bottom * mpp * forward_vec
                       - 0.5 * self._width * mpp * right_vec)
        top_left = (ego_pos
                    + (self._width - self._ev_to_bottom) * mpp * forward_vec
                    - 0.5 * self._width * mpp * right_vec)
        top_right = (ego_pos
                     + (self._width - self._ev_to_bottom) * mpp * forward_vec
                     + 0.5 * self._width * mpp * right_vec)

        # 소스: 세계 좌표의 세 점 → 목적: BEV 이미지 픽셀의 세 점
        src_pts = np.stack(
            (bottom_left, top_left, top_right), axis=0
        ).astype(np.float32)

        dst_pts = np.array(
            [[0, self._width - 1],   # bottom_left  → 이미지 좌하단
             [0, 0],                  # top_left     → 이미지 좌상단
             [self._width - 1, 0]],   # top_right    → 이미지 우상단
            dtype=np.float32
        )

        return cv.getAffineTransform(src_pts, dst_pts)

    # ═══════════════════════════════════════════════════════════════
    # 객체 마스킹
    # ═══════════════════════════════════════════════════════════════

    def _get_mask_from_objects(self, obj_list: List[ObjectData],
                              M_warp, is_vehicle: bool):
        """객체 리스트를 BEV 마스크에 렌더링한다.

        Roach chauffeurnet.py의 _get_mask_from_actor_list()와 동일한 방식.
          - 차량: 5각형 (pointed front — 전방이 뾰족한 형태)
          - 보행자: 4각형 (단순 직사각형)

        Args:
            obj_list: ObjectData 리스트.
            M_warp: 세계 좌표 → BEV 픽셀 아핀 행렬 (2×3).
            is_vehicle: True면 차량(5각형), False면 보행자(4각형).

        Returns:
            (H, W) bool 마스크.
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        for obj in obj_list:
            # 바운딩 박스 half-extent (미터)
            if obj.size_x > 0 and obj.size_y > 0:
                half_x = obj.size_x / 2.0   # 전방/후방 half-length
                half_y = obj.size_y / 2.0   # 좌/우 half-width
            else:
                # MORAI에서 크기 미제공 시 기본값 사용
                default = self._default_veh_size if is_vehicle else self._default_ped_size
                half_x = default[0] / 2.0
                half_y = default[1] / 2.0

            # 바운딩 박스 코너 — 객체 로컬 좌표
            # x축 = 객체 전방, y축 = 객체 좌측
            # Roach 원본: corners = [(-ext.x,-ext.y), (ext.x,-ext.y),
            #                        (ext.x, 0), (ext.x, ext.y), (-ext.x, ext.y)]
            if is_vehicle:
                # 5각형: 전방이 뾰족한 형태 (Roach 스타일)
                corners_local = np.array([
                    [-half_x, -half_y],   # 후방-좌
                    [ half_x, -half_y],   # 전방-좌
                    [ half_x,  0.0   ],   # 전방-중앙 (노즈)
                    [ half_x,  half_y],   # 전방-우
                    [-half_x,  half_y],   # 후방-우
                ], dtype=np.float32)
            else:
                # 보행자: 4각형
                corners_local = np.array([
                    [-half_x, -half_y],
                    [ half_x, -half_y],
                    [ half_x,  half_y],
                    [-half_x,  half_y],
                ], dtype=np.float32)

            # 객체 heading으로 회전 후 세계 좌표로 변환
            heading_rad = np.deg2rad(obj.heading)
            cos_h = np.cos(heading_rad)
            sin_h = np.sin(heading_rad)

            # 2D 회전 행렬 적용: [cos -sin; sin cos] · [lx; ly]
            corners_world = np.empty_like(corners_local)
            corners_world[:, 0] = (obj.pos_x
                                   + cos_h * corners_local[:, 0]
                                   - sin_h * corners_local[:, 1])
            corners_world[:, 1] = (obj.pos_y
                                   + sin_h * corners_local[:, 0]
                                   + cos_h * corners_local[:, 1])

            # 세계 좌표 → BEV 픽셀 (아핀 변환)
            corners_px = corners_world.reshape(-1, 1, 2).astype(np.float32)
            corners_bev = cv.transform(corners_px, M_warp)

            # 다각형 채우기
            cv.fillConvexPoly(
                mask,
                np.round(corners_bev).astype(np.int32),
                1
            )

        return mask.astype(bool)

    # ═══════════════════════════════════════════════════════════════
    # 신호등 마스킹
    # ═══════════════════════════════════════════════════════════════

    def _classify_traffic_light(self, tl_data: Optional[TrafficLightData]):
        """신호등 상태를 green / yellow / red 로 분류한다.

        MORAI 신호등 상태 코드 → Roach BEV 채널 밝기값 매핑:
          GREEN (16), GREEN_LEFT (32), GREEN_GREEN_LEFT (48) → 'green'
          YELLOW (4), YELLOW_GREEN (20), YELLOW_GREEN_LEFT (36) → 'yellow'
          RED (1), RED_YELLOW (5) → 'red'

        Returns:
            'green' | 'yellow' | 'red' | None
        """
        if tl_data is None:
            return None

        status = tl_data.status
        if status in (TL_GREEN, TL_GREEN_LEFT, TL_GREEN_GREEN_LEFT):
            return 'green'
        elif status in (TL_YELLOW, TL_YELLOW_GREEN, TL_YELLOW_GREEN_LEFT):
            return 'yellow'
        elif status in (TL_RED, TL_RED_YELLOW):
            return 'red'
        return None

    def _render_traffic_light_mask(self, tl_data: Optional[TrafficLightData]):
        """신호등 상태를 BEV 마스크로 렌더링한다.

        MORAI는 신호등의 stopline 위치 정보를 제공하지 않으므로,
        Roach 원본의 stopline 렌더링 대신 BEV 이미지 상단에
        가로 바(bar)로 신호등 상태를 표시한다.

        바(bar) 위치: BEV 이미지 상단 10~16px 영역
        → Ego 전방 약 (192-40-10)/5 ≈ 28.4m ~ (192-40-16)/5 ≈ 27.2m 지점

        채널 밝기값 (Roach 원본과 동일):
          Green:  80  (80/255 ≈ 0.3137)
          Yellow: 170 (170/255 ≈ 0.6667)
          Red:    255 (255/255 = 1.0)

        Returns:
            (H, W) uint8 마스크. 해당 상태의 밝기값 또는 0.
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        state = self._classify_traffic_light(tl_data)
        if state is None:
            return mask

        val_map = {
            'green':  self._tl_green_val,    # 80
            'yellow': self._tl_yellow_val,   # 170
            'red':    self._tl_red_val,       # 255
        }
        val = val_map[state]

        # Roach 원본의 stopline 두께: thickness=6
        # 여기서도 6px 높이의 바(bar)로 표시
        bar_y_start = 10
        bar_y_end = 16
        mask[bar_y_start:bar_y_end, :] = val

        return mask

    # ═══════════════════════════════════════════════════════════════
    # 히스토리 처리
    # ═══════════════════════════════════════════════════════════════

    def _get_history_masks(self, M_warp):
        """히스토리 인덱스에 따라 과거 프레임들의 마스크를 생성한다.

        Roach chauffeurnet.py의 _get_history_masks()와 동일한 로직.
        history_idx = [-16, -11, -6, -1] 이면 4개의 시점에 대해 마스크 생성.

        Returns:
            vehicle_masks: list[bool mask], len = len(history_idx)
            walker_masks: list[bool mask], len = len(history_idx)
            tl_masks: list[uint8 mask], len = len(history_idx)
        """
        qsize = len(self._history_queue)
        vehicle_masks = []
        walker_masks = []
        tl_masks = []

        for idx in self._history_idx:
            # 큐 크기보다 과거를 요청하면 가장 오래된 프레임 사용
            actual_idx = max(idx, -qsize)
            vehicles, pedestrians, tl = self._history_queue[actual_idx]

            vehicle_masks.append(
                self._get_mask_from_objects(vehicles, M_warp, is_vehicle=True))
            walker_masks.append(
                self._get_mask_from_objects(pedestrians, M_warp, is_vehicle=False))
            tl_masks.append(
                self._render_traffic_light_mask(tl))

        return vehicle_masks, walker_masks, tl_masks

    # ═══════════════════════════════════════════════════════════════
    # 유틸리티
    # ═══════════════════════════════════════════════════════════════

    def _filter_by_distance(self, obj_list: List[ObjectData],
                            ego: EgoState, max_dist: float):
        """Ego로부터 max_dist(m) 이내 객체만 필터링한다.

        Roach chauffeurnet.py의 is_within_distance()와 동일한 로직:
          - x, y 각각 독립적으로 거리 체크 (맨해튼 거리)
          - Ego 자신은 제외 (1m 이내)
        """
        filtered = []
        for obj in obj_list:
            dx = abs(obj.pos_x - ego.pos_x)
            dy = abs(obj.pos_y - ego.pos_y)
            # 거리 이내이고, ego 자신이 아닌 경우
            if dx < max_dist and dy < max_dist and not (dx < 1.0 and dy < 1.0):
                filtered.append(obj)
        return filtered

    def _scale_objects(self, obj_list: List[ObjectData], scale: float):
        """객체의 바운딩 박스를 스케일링한다 (복사본 생성).

        Roach chauffeurnet.py의 _get_surrounding_actors()에서
        보행자 bbox를 2.0× 확대하는 것과 동일.
        최소 크기: 0.8m (각 축)
        """
        scaled = []
        for obj in obj_list:
            new_size_x = max(obj.size_x * scale, 0.8) if obj.size_x > 0 else self._default_ped_size[0] * scale
            new_size_y = max(obj.size_y * scale, 0.8) if obj.size_y > 0 else self._default_ped_size[1] * scale
            scaled.append(replace(obj, size_x=new_size_x, size_y=new_size_y))
        return scaled

    # ═══════════════════════════════════════════════════════════════
    # RGB 시각화 렌더링
    # ═══════════════════════════════════════════════════════════════

    def _render_ego_mask(self, ego_state: EgoState, M_warp):
        """Ego 차량 바운딩 박스를 BEV 마스크에 렌더링한다."""
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        half_x = ego_state.size_x / 2.0 if ego_state.size_x > 0 else 2.25
        half_y = ego_state.size_y / 2.0 if ego_state.size_y > 0 else 1.0

        # 5각형 (차량과 동일한 pointed front)
        corners_local = np.array([
            [-half_x, -half_y],
            [ half_x, -half_y],
            [ half_x,  0.0   ],
            [ half_x,  half_y],
            [-half_x,  half_y],
        ], dtype=np.float32)

        heading_rad = np.deg2rad(ego_state.yaw)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        corners_world = np.empty_like(corners_local)
        corners_world[:, 0] = (ego_state.pos_x
                               + cos_h * corners_local[:, 0]
                               - sin_h * corners_local[:, 1])
        corners_world[:, 1] = (ego_state.pos_y
                               + sin_h * corners_local[:, 0]
                               + cos_h * corners_local[:, 1])

        corners_px = corners_world.reshape(-1, 1, 2).astype(np.float32)
        corners_bev = cv.transform(corners_px, M_warp)
        cv.fillConvexPoly(mask, np.round(corners_bev).astype(np.int32), 1)

        return mask.astype(bool)

    def _render_rgb(self, vehicle_masks, walker_masks, tl_masks,
                    ego_state, M_warp):
        """디버깅/시각화용 RGB 이미지를 렌더링한다.

        Roach chauffeurnet.py의 렌더링 순서와 색상을 그대로 따름:
          - 신호등: green / yellow / red (히스토리: 점점 밝게)
          - 차량: 파란색 (히스토리: 점점 밝게)
          - 보행자: 시안색 (히스토리: 점점 밝게)
          - Ego: 흰색
        """
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)

        h_len = len(self._history_idx) - 1

        # 신호등 (밝기값으로 상태 구분)
        for i, tl_mask in enumerate(tl_masks):
            factor = (h_len - i) * 0.2
            green_area = tl_mask == self._tl_green_val
            yellow_area = tl_mask == self._tl_yellow_val
            red_area = tl_mask == self._tl_red_val
            if np.any(green_area):
                image[green_area] = _tint(COLOR_GREEN, factor)
            if np.any(yellow_area):
                image[yellow_area] = _tint(COLOR_YELLOW, factor)
            if np.any(red_area):
                image[red_area] = _tint(COLOR_RED, factor)

        # 차량 (파란색, 히스토리: 과거→현재 점점 진해짐)
        for i, mask in enumerate(vehicle_masks):
            if np.any(mask):
                image[mask] = _tint(COLOR_BLUE, (h_len - i) * 0.2)

        # 보행자 (시안색)
        for i, mask in enumerate(walker_masks):
            if np.any(mask):
                image[mask] = _tint(COLOR_CYAN, (h_len - i) * 0.2)

        # Ego 차량 (흰색)
        ego_mask = self._render_ego_mask(ego_state, M_warp)
        image[ego_mask] = COLOR_WHITE

        return image
