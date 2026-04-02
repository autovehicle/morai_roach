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
  - 신호등: stopline을 BEV에 렌더링, 상태별 밝기값 (green=80, yellow=170, red=255)

신호등 stopline 매핑:
  MORAI는 신호등과 정지선이 1:1 대응되지 않으므로,
  traffic_light_set.json과 stoplane_marking_set.json을 로드하여
  각 정지선에서 가장 가까운 신호등을 매핑한다.
  Roach 원본의 _get_mask_from_stopline_vtx()와 동일하게
  stopline을 cv.line()으로 BEV에 렌더링한다.

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
"""

import json
import numpy as np
import cv2 as cv
from collections import deque
from dataclasses import replace
from typing import List, Optional, Dict, Tuple

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
COLOR_WHITE = (255, 255, 255)   # 실선(Solid)
COLOR_GRAY = (120, 120, 120)    # 점선(Broken/Dashed)
COLOR_EGO = (200, 255, 200)     # Ego 차량 (연한 초록색, 실선과 구분)


def _tint(color, factor):
    """색상을 밝게(tint) 만든다. factor=0이면 원색, factor=1이면 흰색."""
    r = min(int(color[0] + (255 - color[0]) * factor), 255)
    g = min(int(color[1] + (255 - color[1]) * factor), 255)
    b = min(int(color[2] + (255 - color[2]) * factor), 255)
    return (r, g, b)


# ═══════════════════════════════════════════════════════════════════
# 신호등 ↔ 정지선 매핑
# ═══════════════════════════════════════════════════════════════════

class TrafficLightStoplineMapper:
    """신호등과 정지선의 최근접 매핑을 관리한다.

    MORAI의 traffic_light_set.json과 stoplane_marking_set.json을 로드하여
    각 정지선에서 가장 가까운 신호등을 찾아 매핑한다.

    매핑 결과:
      tl_idx → list of stopline_vtx
      각 stopline_vtx = [(x1, y1), (x2, y2)]  (정지선의 시작점과 끝점)

    Roach 원본에서 TrafficLightHandler.get_stopline_vtx()가 반환하는 것과
    동일한 형태의 데이터를 제공한다.
    """

    def __init__(self, tl_json_path: str, stopline_json_path: str,
                 max_match_distance: float = 30.0):
        """
        Args:
            tl_json_path: traffic_light_set.json 경로.
            stopline_json_path: stoplane_marking_set.json 경로.
            max_match_distance: 매칭 최대 거리 (m). 이 거리 초과 시 매핑 안 함.
        """
        with open(tl_json_path, 'r', encoding='utf-8') as f:
            tl_data = json.load(f)
        with open(stopline_json_path, 'r', encoding='utf-8') as f:
            stopline_data = json.load(f)

        # 신호등 위치: {idx: (x, y)}
        tl_positions: Dict[str, Tuple[float, float]] = {}
        for tl in tl_data:
            tl_positions[tl['idx']] = (tl['point'][0], tl['point'][1])

        # 정지선 → 시작/끝 꼭짓점 (Roach stopline_vtx 형태)
        # 각 정지선의 첫 점과 마지막 점을 stopline 양 끝으로 사용
        stopline_vtx_list: List[Tuple[str, List[Tuple[float, float]]]] = []
        for sl in stopline_data:
            pts = sl['points']
            if len(pts) < 2:
                continue
            # 정지선의 시작점과 끝점
            p_start = (pts[0][0], pts[0][1])
            p_end = (pts[-1][0], pts[-1][1])
            # 정지선 중심 (매칭용)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            stopline_vtx_list.append((sl['idx'], cx, cy, [p_start, p_end]))

        # 각 정지선에서 가장 가까운 신호등 찾기 → tl_idx별로 그룹핑
        # 결과: {tl_idx: [stopline_vtx, ...]}
        self._tl_to_stoplines: Dict[str, List[List[Tuple[float, float]]]] = {}
        self._all_tl_positions = tl_positions

        for sl_idx, cx, cy, vtx in stopline_vtx_list:
            min_dist = float('inf')
            best_tl_idx = None
            for tl_idx, (tx, ty) in tl_positions.items():
                d = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                if d < min_dist:
                    min_dist = d
                    best_tl_idx = tl_idx
            if best_tl_idx is not None and min_dist <= max_match_distance:
                if best_tl_idx not in self._tl_to_stoplines:
                    self._tl_to_stoplines[best_tl_idx] = []
                self._tl_to_stoplines[best_tl_idx].append(vtx)

        n_matched = sum(len(v) for v in self._tl_to_stoplines.values())
        n_tl_matched = len(self._tl_to_stoplines)
        print(f'[TrafficLightStoplineMapper] '
              f'{n_tl_matched}/{len(tl_positions)} 신호등에 '
              f'{n_matched}/{len(stopline_vtx_list)} 정지선 매핑 완료 '
              f'(max_dist={max_match_distance}m)')

    def get_stopline_vtx(self, tl_idx: str) -> List[List[Tuple[float, float]]]:
        """신호등 ID에 매핑된 정지선 꼭짓점 리스트를 반환한다.

        Roach 원본의 TrafficLightHandler.get_stopline_vtx()와 동일한 형태.

        Args:
            tl_idx: 신호등 ID 문자열.

        Returns:
            list of stopline_vtx.
            각 stopline_vtx = [(x1, y1), (x2, y2)] — 정지선의 양 끝점.
            매핑이 없으면 빈 리스트.
        """
        return self._tl_to_stoplines.get(tl_idx, [])

    def get_nearby_stoplines(self, ego_x: float, ego_y: float,
                             max_dist: float = 50.0
                             ) -> Dict[str, List[List[Tuple[float, float]]]]:
        """Ego 주변의 신호등 → 정지선 매핑을 반환한다.

        Args:
            ego_x, ego_y: Ego 차량 세계 좌표 (m).
            max_dist: 검색 반경 (m).

        Returns:
            {tl_idx: [stopline_vtx, ...]} — ego 주변 신호등만 포함.
        """
        nearby = {}
        for tl_idx, (tx, ty) in self._all_tl_positions.items():
            if abs(tx - ego_x) < max_dist and abs(ty - ego_y) < max_dist:
                vtx_list = self._tl_to_stoplines.get(tl_idx)
                if vtx_list:
                    nearby[tl_idx] = vtx_list
        return nearby


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
        tl_distance: float = 18.0,
        tl_green_val: int = 80,
        tl_yellow_val: int = 170,
        tl_red_val: int = 255,
        default_veh_size: tuple = (4.5, 2.0),
        default_ped_size: tuple = (0.5, 0.5),
        tl_mapper: Optional['TrafficLightStoplineMapper'] = None,
        lane_markings: Optional[List[Dict]] = None,
        lane_max_range: float = 50.0,
        lane_thickness: int = 2,
        lane_solid_value: int = 255,
        lane_broken_value: int = 120,
        link_data: Optional[List[Dict]] = None,
        link_max_range: float = 50.0,
        link_thickness: int = 1,
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
            tl_distance: 신호등 감지 범위 (meters).
            tl_green_val: 녹색 신호등 채널 밝기값 (Roach: 80 ≈ 0.3137).
            tl_yellow_val: 황색 신호등 채널 밝기값 (Roach: 170 ≈ 0.6667).
            tl_red_val: 적색 신호등 채널 밝기값 (Roach: 255 = 1.0).
            default_veh_size: MORAI에서 크기 미제공 시 기본 차량 크기 (length, width) meters.
            default_ped_size: MORAI에서 크기 미제공 시 기본 보행자 크기 (length, width) meters.
            tl_mapper: 신호등 ↔ 정지선 매퍼. None이면 stopline 렌더링 비활성화.
        """
        self._width = width
        self._ev_to_bottom = pixels_ev_to_bottom
        self._ppm = pixels_per_meter
        self._history_idx = history_idx if history_idx is not None else [-16, -11, -6, -1]
        self._scale_bbox = scale_bbox
        self._walker_scale = walker_bbox_scale
        self._veh_dist = vehicle_distance
        self._ped_dist = pedestrian_distance
        self._tl_dist = tl_distance

        self._tl_green_val = tl_green_val
        self._tl_yellow_val = tl_yellow_val
        self._tl_red_val = tl_red_val

        self._default_veh_size = default_veh_size   # (length, width) m
        self._default_ped_size = default_ped_size

        # 신호등 ↔ 정지선 매퍼
        self._tl_mapper = tl_mapper

        # 신호등 상태 캐시: {tl_idx: 'green'|'yellow'|'red'}
        # MORAI UDP는 한 번에 하나의 신호등 상태만 전송하므로
        # 수신된 상태를 캐시하여 여러 신호등의 상태를 유지한다.
        self._tl_state_cache: Dict[str, str] = {}

        # 히스토리 큐 — 최대 20프레임 보관 (10Hz × 2초)
        self._history_queue: deque = deque(maxlen=20)

        # 차선 마스크
        self._lane_markings = lane_markings if lane_markings is not None else []
        self._lane_max_range = lane_max_range
        self._lane_thickness = lane_thickness
        self._lane_solid_value = lane_solid_value
        self._lane_broken_value = lane_broken_value

        # 도로 링크 (link_set.json)
        self._link_data = link_data if link_data is not None else []
        self._link_max_range = link_max_range
        self._link_thickness = link_thickness

    # ───────────────────────────────────────────────────────────────
    # 팩토리 메서드
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config):
        """map_to_h5.py의 Config 객체로부터 렌더러를 생성한다."""
        # 신호등 ↔ 정지선 매퍼 생성 (JSON 파일이 존재할 때만)
        map_dir = Path(config.root) / 'morai_gym' / 'lib' / 'core' / 'birdiview' / 'map'
        tl_json = map_dir / 'traffic_light_set.json'
        sl_json = map_dir / 'stoplane_marking_set.json'

        tl_mapper = None
        if tl_json.exists() and sl_json.exists():
            tl_mapper = TrafficLightStoplineMapper(str(tl_json), str(sl_json))
            print(f'[BEVDynamicRenderer] Stopline mapper loaded.')
        else:
            print(f'[BEVDynamicRenderer] WARNING: JSON files not found, '
                  f'stopline rendering disabled.')

        lane_json = map_dir / 'lane_marking_set.json'
        lane_markings = []
        if lane_json.exists():
            lane_markings = cls._load_lane_markings(str(lane_json))
            print(f'[BEVDynamicRenderer] Lane markings loaded: {len(lane_markings)} elements')
        else:
            print(f'[BEVDynamicRenderer] WARNING: lane_marking_set.json not found, lane rendering disabled.')

            # ── link_set.json 로드 추가 ──
        link_json = map_dir / 'link_set.json'
        link_data = []
        if link_json.exists():
            link_data = cls._load_link_set(str(link_json))
            print(f'[BEVDynamicRenderer] Link set loaded: {len(link_data)} links')
        else:
            print(f'[BEVDynamicRenderer] WARNING: link_set.json not found, link rendering disabled.')


        return cls(
            width=config.width,
            pixels_ev_to_bottom=config.ev_to_bottom,
            pixels_per_meter=config.ppm,
            history_idx=config.history_idx,
            scale_bbox=config.scale_bbox,
            walker_bbox_scale=config.walker_scale,
            vehicle_distance=config.veh_dist,
            pedestrian_distance=config.ped_dist,
            tl_distance=config.tl_dist,
            tl_green_val=config.tl_green,
            tl_yellow_val=config.tl_yellow,
            tl_red_val=config.tl_red,
            default_veh_size=config.veh_size,
            default_ped_size=config.ped_size,
            tl_mapper=tl_mapper,
            lane_markings=lane_markings,
            lane_max_range=50.0,
            lane_thickness=int(getattr(config, 'route_thick', 2)),
            lane_solid_value=config.lane_solid,
            lane_broken_value=config.lane_broken,
            link_data=link_data,
            link_max_range=50.0,
            link_thickness=1,
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

        # ── 3) 신호등 상태 캐시 업데이트 ──
        #   MORAI UDP는 한 번에 하나의 신호등만 전송하므로
        #   수신할 때마다 해당 신호등의 상태를 캐시에 저장한다.
        if traffic_light is not None:
            state = self._classify_traffic_light(traffic_light)
            if state is not None:
                self._tl_state_cache[str(traffic_light.index)] = state

        # ── 4) 히스토리 큐에 저장 (신호등은 현재 캐시 스냅샷) ──
        tl_cache_snapshot = dict(self._tl_state_cache)
        self._history_queue.append(
            (vehicles, pedestrians, tl_cache_snapshot, ego_state))

        # ── 5) Affine 변환 행렬 생성 (세계 좌표 → BEV 픽셀) ──
        M_warp = self._get_warp_transform(
            ego_state.pos_x, ego_state.pos_y, ego_state.yaw)

        # ── 6) 히스토리별 마스크 렌더링 ──
        vehicle_masks, walker_masks, tl_masks = self._get_history_masks(
            M_warp, ego_state)

        # ── 7) 차선 마스크 생성 및 RGB에 오버레이 ──
        lane_mask = self._get_lane_mask(ego_state)

        # ── 7-1) 도로 링크 마스크 생성 ── 
        link_mask = self._get_link_mask(ego_state)  

        rendered = self._render_rgb(
            vehicle_masks, walker_masks, tl_masks, lane_mask, link_mask, ego_state, M_warp)

        # ── 8) 출력 마스크 채널 조합 ──
        c_vehicle = [m.astype(np.uint8) * 255 for m in vehicle_masks]
        c_walker = [m.astype(np.uint8) * 255 for m in walker_masks]
        c_tl = tl_masks  # 이미 uint8 밝기값
        c_lane = lane_mask.astype(np.uint8)
        c_link = link_mask.astype(np.uint8)  


        masks = np.stack((*c_vehicle, *c_walker, *c_tl, c_lane, c_link), axis=0)

        return {'rendered': rendered, 'masks': masks}

    def reset(self):
        """히스토리 큐와 신호등 캐시를 초기화한다. 에피소드 시작 시 호출."""
        self._history_queue.clear()
        self._tl_state_cache.clear()

    @staticmethod
    def _load_lane_markings(lane_json_path: str):
        if lane_json_path is None:
            return []
        path = Path(lane_json_path)
        if not path.exists():
            print(f"[BEVDynamicRenderer] WARNING: lane json not found: {lane_json_path}")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[BEVDynamicRenderer] WARNING: lane json load fail: {e}")
            return []

        lanes = []
        for item in data:
            pts = np.asarray(item.get('points', []), dtype=np.float32)
            lane_type = item.get('lane_type', -1)
            if lane_type == 530:
                continue
            if pts.ndim == 2 and pts.shape[1] >= 2 and pts.shape[0] >= 2:
                # lane_shape: ['Solid'], ['Broken'] 등
                lane_shape = item.get('lane_shape', [])
                shape_str = lane_shape[0].lower() if lane_shape else 'solid'
                lanes.append({
                    'idx': item.get('idx', ''),
                    'points': pts,
                    'lane_type': lane_type,
                    'lane_color': item.get('lane_color', ''),
                    'lane_shape': shape_str,  # 'solid', 'broken' 등
                    'line_width': item.get('line_width', 0.15),  # 기본값 0.15m
                })
        return lanes
    
    @staticmethod
    def _load_link_set(link_json_path: str):
        """link_set.json을 로드하여 BEV 렌더링용 데이터로 변환한다."""
        if link_json_path is None:
            return []
        path = Path(link_json_path)
        if not path.exists():
            print(f"[BEVDynamicRenderer] WARNING: link json not found: {link_json_path}")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[BEVDynamicRenderer] WARNING: link json load fail: {e}")
            return []

        links = []
        for item in data:
            pts = np.asarray(item.get('points', []), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 2 and pts.shape[0] >= 2:
                links.append({
                    'idx': item.get('idx', ''),
                    'points': pts,
                    'link_type': item.get('link_type'),       # '1', '6', None
                    'road_type': item.get('road_type'),
                    'width_start': item.get('width_start', 3.5),
                })
        return links

    def _get_lane_mask(self, ego_state: EgoState):
        """Ego 주변의 차선을 BEV 마스크에 렌더링한다.
        
        마스크 값:
          - 255 (흰색): 실선(Solid)
          - 120 (회색): 점선(Broken/Dashed)
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not self._lane_markings or ego_state is None:
            return mask.astype(bool)

        M_warp = self._get_warp_transform(
            ego_state.pos_x, ego_state.pos_y, ego_state.yaw)

        for lane in self._lane_markings:
            pts = lane['points']
            dist = np.sqrt((pts[:, 0] - ego_state.pos_x) ** 2 +
                           (pts[:, 1] - ego_state.pos_y) ** 2)
            pts_in_range = pts[dist <= self._lane_max_range]
            if pts_in_range.shape[0] < 2:
                continue

            pts_xy = pts_in_range[:, :2].reshape(-1, 1, 2).astype(np.float32)
            pts_bev = cv.transform(pts_xy, M_warp)
            pts_bev = np.ascontiguousarray(np.round(pts_bev).astype(np.int32))

            # lane_shape: 'solid', 'broken', 'dashed' 등
            lane_shape = str(lane.get('lane_shape', 'solid')).lower()
            if lane_shape in ('broken', 'dashed', 'dash'):
                val = self._lane_broken_value  # 120 (회색)
            else:
                val = self._lane_solid_value   # 255 (흰색)

            # 동적 두께 계산: line_width (m) * pixels_per_meter
            line_width = lane.get('line_width', 0.15)
            thickness = max(1, int(line_width * self._ppm))

            # 차선을 마스크에 그리기
            cv.polylines(
                mask,
                [pts_bev],
                isClosed=False,
                color=int(val),
                thickness=thickness,
            )

        return mask
    
    def _get_link_mask(self, ego_state: EgoState):

        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not self._link_data or ego_state is None:
            return mask

        M_warp = self._get_warp_transform(
            ego_state.pos_x, ego_state.pos_y, ego_state.yaw)

        for link in self._link_data:
            pts = link['points']
            dist = np.sqrt((pts[:, 0] - ego_state.pos_x) ** 2 +
                        (pts[:, 1] - ego_state.pos_y) ** 2)
            pts_in_range = pts[dist <= self._link_max_range]
            if pts_in_range.shape[0] < 2:
                continue

            pts_xy = pts_in_range[:, :2].reshape(-1, 1, 2).astype(np.float32)
            pts_bev = cv.transform(pts_xy, M_warp)
            pts_bev = np.ascontiguousarray(np.round(pts_bev).astype(np.int32))

            # link_type별 밝기값 구분
            lt = link.get('link_type')
            if lt == '1':
                val = 200
            elif lt == '6':
                val = 140
            else:
                val = 100

            cv.polylines(
                mask,
                [pts_bev],
                isClosed=False,
                color=int(val),
                thickness=self._link_thickness,
            )

        return mask

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

    def _get_mask_from_stopline_vtx(self, stopline_vtx: list, M_warp):
        """정지선 꼭짓점을 BEV 마스크에 렌더링한다.

        Roach chauffeurnet.py의 _get_mask_from_stopline_vtx()와 동일.
        각 정지선의 양 끝점을 세계 좌표에서 BEV 픽셀로 변환 후
        cv.line()으로 그린다.

        Args:
            stopline_vtx: [[(x1,y1), (x2,y2)], ...] — 정지선 양 끝점 리스트.
            M_warp: 세계 좌표 → BEV 픽셀 아핀 변환 행렬 (2×3).

        Returns:
            (H, W) bool 마스크.
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        for sp_locs in stopline_vtx:
            # sp_locs = [(x1, y1), (x2, y2)]
            stopline_in_pixel = np.array(
                [[list(x)] for x in sp_locs], dtype=np.float32)
            stopline_warped = cv.transform(stopline_in_pixel, M_warp)
            cv.line(mask,
                    tuple(np.round(stopline_warped[0, 0]).astype(int)),
                    tuple(np.round(stopline_warped[1, 0]).astype(int)),
                    color=1, thickness=6)
        return mask.astype(bool)

    def _render_tl_stopline_mask(self, tl_state_cache: dict,
                                 ego_state: EgoState, M_warp):
        """Roach 원본 방식: 신호등 상태별 stopline을 BEV에 렌더링한다.

        Roach chauffeurnet.py의 _get_history_masks() 내부 신호등 처리와 동일:
          1. Ego 주변 신호등 → 매핑된 정지선 찾기
          2. 신호등 상태별로 stopline을 분류
          3. 각 상태의 stopline을 _get_mask_from_stopline_vtx()로 렌더링
          4. 마스크에 해당 상태의 밝기값 적용

        채널 밝기값:
          Green:  80, Yellow: 170, Red: 255

        Args:
            tl_state_cache: {tl_idx: 'green'|'yellow'|'red'} — 신호등 상태 캐시.
            ego_state: Ego 차량 상태.
            M_warp: 세계 좌표 → BEV 아핀 변환 행렬.

        Returns:
            (H, W) uint8 마스크 (밝기값으로 상태 구분).
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        if self._tl_mapper is None:
            return mask

        # Ego 주변 신호등의 매핑된 정지선 가져오기
        nearby = self._tl_mapper.get_nearby_stoplines(
            ego_state.pos_x, ego_state.pos_y, self._tl_dist)

        val_map = {
            'green':  self._tl_green_val,    # 80
            'yellow': self._tl_yellow_val,   # 170
            'red':    self._tl_red_val,       # 255
        }

        # 상태별 stopline 분류
        stoplines_by_state: Dict[str, list] = {
            'green': [], 'yellow': [], 'red': []
        }
        for tl_idx, vtx_list in nearby.items():
            state = tl_state_cache.get(tl_idx)
            if state is not None and state in stoplines_by_state:
                stoplines_by_state[state].extend(vtx_list)

        # 각 상태별로 stopline 렌더링
        for state, vtx_list in stoplines_by_state.items():
            if not vtx_list:
                continue
            state_mask = self._get_mask_from_stopline_vtx(vtx_list, M_warp)
            mask[state_mask] = val_map[state]

        return mask

    # ═══════════════════════════════════════════════════════════════
    # 히스토리 처리
    # ═══════════════════════════════════════════════════════════════

    def _get_history_masks(self, M_warp, ego_state: EgoState):
        """히스토리 인덱스에 따라 과거 프레임들의 마스크를 생성한다.

        Roach chauffeurnet.py의 _get_history_masks()와 동일한 로직.
        history_idx = [-16, -11, -6, -1] 이면 4개의 시점에 대해 마스크 생성.

        신호등은 Roach 원본 방식으로 stopline을 렌더링한다:
          각 시점의 tl_state_cache를 사용하여 해당 시점의 신호등 상태에
          맞는 stopline을 BEV에 그린다.

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
            vehicles, pedestrians, tl_cache, _ = \
                self._history_queue[actual_idx]

            vehicle_masks.append(
                self._get_mask_from_objects(vehicles, M_warp, is_vehicle=True))
            walker_masks.append(
                self._get_mask_from_objects(pedestrians, M_warp, is_vehicle=False))
            tl_masks.append(
                self._render_tl_stopline_mask(tl_cache, ego_state, M_warp))

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
                    lane_mask, link_mask, ego_state, M_warp):
        """디버깅/시각화용 RGB 이미지를 렌더링한다.

        Roach chauffeurnet.py의 렌더링 순서와 색상을 그대로 따름:
          - 신호등: green / yellow / red (히스토리: 점점 밝게)
          - 차량: 파란색 (히스토리: 점점 밝게)
          - 보행자: 시안색 (히스토리: 점점 밝게)
          - Ego: 흰색
        """
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)

        h_len = len(self._history_idx) - 1

            # ── 도로 링크 (가장 먼저 = 배경) ──  ← 추가
        if link_mask is not None and np.any(link_mask):
            # link_type별 색상 구분
            type1_mask = (link_mask == 200)
            type6_mask = (link_mask == 140)
            other_mask = (link_mask == 100)
            image[type1_mask] = (255, 105, 180)   # 일반 도로: 핫핑크
            image[type6_mask] = (255, 182, 193)   # 교차로: 라이트핑크
            image[other_mask] = (219, 112, 147)   # 미분류: 팔레 바이올렛 레드

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

        # 차선 (CARLA Roach 명세: 실선=흰색, 점선=회색)
        if lane_mask is not None and np.any(lane_mask):
            solid_mask = (lane_mask == self._lane_solid_value)
            broken_mask = (lane_mask == self._lane_broken_value)
            image[solid_mask] = COLOR_WHITE       # Solid: 255, 255, 255
            image[broken_mask] = COLOR_GRAY       # Broken: 120, 120, 120

        # Ego 차량 (연한 초록색) - 최상단
        # 실선(흰색)과 구분하기 위해 다른 색 사용
        ego_mask = self._render_ego_mask(ego_state, M_warp)
        image[ego_mask] = COLOR_EGO

        return image
