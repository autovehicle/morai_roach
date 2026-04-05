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
  traffic_light_set / stoplane / link_set 사용. 신호등별 수동 맵 →
  link_id_list+링크 시작점 근접 정지선 → 거리 fallback 순 매칭.
  Roach와 동일하게 stopline은 cv.line()으로 BEV에 렌더링한다.

채널 구성 — carla-roach chauffeurnet.py와 동일 순서 (총 3 + 3*K, K=len(history_idx), 기본 15채널):
  masks shape: (3 + 3*K, H, W) uint8
    [road, route, lane,
     vehicle 히스토리 K장,
     walker 히스토리 K장,
     traffic_light(+stop) 히스토리 K장]

  - road: KATRI 맵 H5(morai_katri_map.h5 등)에 'road' 레이어가 있으면 CARLA와 동일 방식으로 워핑.
          H5가 없으면 해당 채널은 0.
  - route: update(..., route_world_xy=) 로 주입한 웨이포인트(세계 좌표) 폴리라인. 없으면 0.
  - lane: JSON 차선(실선 255 / 점선 120).
  - rendered RGB: chauffeurnet.py와 동일 — road(ALUMINIUM_5) → route(ALUMINIUM_3) →
    lane(실선 MAGENTA / 점선 MAGENTA_2) → TL → 차량 → 보행자 → ego(흰색).
    link_set 핑크 배경은 CARLA에 없어 시각화에서 제외.

좌표계 가정 (MORAI):
  - 세계 좌표: x=동쪽(East), y=북쪽(North)  [ENU 좌표계]
  - yaw/heading: 동쪽(East)축 기준, 반시계 방향 양수 (degrees)
  - H5 road 워핑은 CARLA와 같이 맵 픽셀 + world_offset_in_meters 를 사용한다.

설치: 저장소 루트에서 pip install -e . (network, morai_gym import)

디버그 (상태 공유용):
  MORAI_BEV_DEBUG=1 — bev_debug.dlog / Config 로드 / renderer_init / H5 / update 요약 출력
  MORAI_BEV_DEBUG_EVERY=N — update 요약을 N 프레임마다 (기본 1)

H5 road 베이킹: ``birdview_map.world_to_pixel_carla`` 수정 후에는 ``morai_katri_map.h5`` 를 반드시 다시 생성한다.
"""

from __future__ import annotations

import json
import h5py
import numpy as np
import cv2 as cv
from collections import deque
from dataclasses import replace
from typing import List, Optional, Dict, Tuple

from pathlib import Path

from morai_gym.core.obs_manager.birdview.bev_debug import (
    morai_bev_dlog,
    morai_bev_debug_enabled,
    morai_bev_debug_every_n,
)

from network.UDP.protocol import (
    ObjectData, TrafficLightData, EgoState,
    OBJ_TYPE_VEHICLE, OBJ_TYPE_PEDESTRIAN,
    TL_RED, TL_YELLOW, TL_GREEN,
    TL_RED_YELLOW, TL_YELLOW_GREEN,
    TL_GREEN_LEFT, TL_YELLOW_GREEN_LEFT,
    TL_GREEN_GREEN_LEFT,
)

# ═══════════════════════════════════════════════════════════════════
# 시각화용 색상 — carla_gym/.../chauffeurnet.py 와 동일 (RGB 튜플)
# ═══════════════════════════════════════════════════════════════════
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_3 = (136, 138, 133)   # route
COLOR_ALUMINIUM_5 = (46, 52, 54)      # road


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
    """신호등 → 정지선 매핑 (main 브랜치 3단계 로직, carla stopline_vtx 형식 유지).

    매핑 순서 (신호등 단위):
      1) 수동: ``link_id_list``가 비어 삼거리 등에서 링크 기준이 안 맞는 TL.
      2) ``link_id_list`` + ``link_set.json``: 각 링크의 시작점(첫 point)에 가장 가까운
         정지선 중심에 매칭 (여러 링크 → 여러 정지선).
      3) fallback: 신호등 좌표에 가장 가까운 정지선 중심 (기존 거리 기반).

    매핑 결과: tl_idx → [stopline_vtx, ...], stopline_vtx = [(x1,y1),(x2,y2)].
    """

    def __init__(self, tl_json_path: str, stopline_json_path: str,
                 max_match_distance: float = 100.0):
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

        tl_positions: Dict[str, Tuple[float, float]] = {}
        for tl in tl_data:
            tl_positions[tl['idx']] = (tl['point'][0], tl['point'][1])

        link_json_path = Path(tl_json_path).parent / 'link_set.json'
        link_endpoints: Dict[str, Tuple[float, float]] = {}
        if link_json_path.is_file():
            with open(link_json_path, 'r', encoding='utf-8') as f:
                link_data = json.load(f)
            for link in link_data:
                pts = link.get('points')
                if not pts or len(pts) < 2:
                    continue
                lid = link.get('idx')
                if lid is None:
                    continue
                link_endpoints[str(lid)] = (float(pts[0][0]), float(pts[0][1]))

        stopline_vtx_list: List[Tuple[str, float, float, List[Tuple[float, float]]]] = []
        for sl in stopline_data:
            pts = sl['points']
            if len(pts) < 2:
                continue
            p_start = (pts[0][0], pts[0][1])
            p_end = (pts[-1][0], pts[-1][1])
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            stopline_vtx_list.append((sl['idx'], cx, cy, [p_start, p_end]))

        # KATRI 삼거리 등: link_id 비어 있거나 링크 매칭만으로 부족한 TL (main과 동일)
        MANUAL_TL_TO_STOPLANE: Dict[str, List[str]] = {
            'C119BS010063': ['B219BS010022'],
        }

        self._tl_to_stoplines: Dict[str, List[List[Tuple[float, float]]]] = {}
        self._all_tl_positions = tl_positions

        sl_vtx_map: Dict[str, List[Tuple[float, float]]] = {
            sl_idx: vtx for sl_idx, _cx, _cy, vtx in stopline_vtx_list}

        for tl in tl_data:
            tl_idx = tl['idx']
            link_id_list = tl.get('link_id_list') or []

            if tl_idx in MANUAL_TL_TO_STOPLANE:
                matched: List[List[Tuple[float, float]]] = []
                for sl_idx in MANUAL_TL_TO_STOPLANE[tl_idx]:
                    vtx = sl_vtx_map.get(sl_idx)
                    if vtx is not None:
                        matched.append(vtx)
                if matched:
                    self._tl_to_stoplines[tl_idx] = matched
                continue

            if link_id_list and link_endpoints:
                matched = []
                for link_id in link_id_list:
                    lid = str(link_id)
                    if lid not in link_endpoints:
                        continue
                    ex, ey = link_endpoints[lid]
                    min_d = float('inf')
                    best_vtx: Optional[List[Tuple[float, float]]] = None
                    for _sl_idx, cx, cy, vtx in stopline_vtx_list:
                        d = np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
                        if d < min_d:
                            min_d = d
                            best_vtx = vtx
                    if best_vtx is not None and min_d <= max_match_distance:
                        matched.append(best_vtx)
                if matched:
                    self._tl_to_stoplines[tl_idx] = matched
                    continue

            tx, ty = tl_positions[tl_idx]
            min_dist = float('inf')
            best_vtx = None
            for _sl_idx, cx, cy, vtx in stopline_vtx_list:
                d = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                if d < min_dist:
                    min_dist = d
                    best_vtx = vtx
            if best_vtx is not None and min_dist <= max_match_distance:
                self._tl_to_stoplines[tl_idx] = [best_vtx]

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
    """동적·정적 BEV 마스크 (carla-roach chauffeurnet 관측 contract 정합).

    사용법:
        from morai_gym.core.obs_manager.birdview.bev_render import BEVDynamicRenderer

        renderer = BEVDynamicRenderer.from_config(config)
        result = renderer.update(
            ego_state, vehicle_list, pedestrian_list, traffic_light,
            route_world_xy=optional_Nx2_array,
        )
        result['masks']   # (15, H, W) uint8 — K=4 기본
        result['collision_px']  # bool — chauffeurnet과 동일 의미
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
        tl_distance: float = 100.0,
        tl_green_val: int = 80,
        tl_yellow_val: int = 170,
        tl_red_val: int = 255,
        default_veh_size: tuple = (4.5, 2.0),
        default_ped_size: tuple = (0.5, 0.5),
        tl_mapper: Optional[TrafficLightStoplineMapper] = None,
        lane_markings: Optional[List[Dict]] = None,
        lane_max_range: float = 50.0,
        lane_thickness: int = 2,
        lane_solid_value: int = 255,
        lane_broken_value: int = 120,
        link_data: Optional[List[Dict]] = None,
        link_max_range: float = 50.0,
        link_thickness: int = 1,
        static_h5_path: Optional[str] = None,
        max_route_waypoints: int = 80,
        route_line_thickness: int = 16,
        scale_mask_col: float = 1.0,
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
            scale_mask_col: chauffeurnet collision_px용 ego bbox 확대 (기본 1.0, carla ppo yaml).
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

        self._max_route_wps = int(max_route_waypoints)
        self._route_thickness = int(route_line_thickness)

        self._h5_road: Optional[np.ndarray] = None
        self._h5_world_offset: Optional[np.ndarray] = None
        self._road_ppm: float = float(pixels_per_meter)
        self._scale_mask_col = float(scale_mask_col)
        self._debug_update_i = 0
        self._try_load_static_h5(static_h5_path)

        morai_bev_dlog(
            'renderer_init',
            'w=%d ev_bottom=%d ppm=%.4f road_ppm=%.4f hist=%s scale_bbox=%s '
            'scale_mask_col=%s walker_scale(when not scale_bbox)=%s '
            'h5_ok=%s h5_shape=%s offset=%s',
            self._width,
            self._ev_to_bottom,
            self._ppm,
            self._road_ppm,
            self._history_idx,
            self._scale_bbox,
            self._scale_mask_col,
            self._walker_scale,
            self._h5_road is not None,
            None if self._h5_road is None else repr(self._h5_road.shape),
            None if self._h5_world_offset is None else self._h5_world_offset.tolist(),
        )

    def _try_load_static_h5(self, static_h5_path: Optional[str]) -> None:
        if not static_h5_path:
            return
        p = Path(static_h5_path)
        if not p.is_file():
            print(f'[BEVDynamicRenderer] WARNING: H5 없음 — road 채널은 0: {static_h5_path}')
            return
        try:
            with h5py.File(str(p), 'r', libver='latest', swmr=True) as hf:
                if 'road' not in hf:
                    print(f'[BEVDynamicRenderer] WARNING: H5에 road 키 없음: {p}')
                    return
                self._h5_road = np.array(hf['road'], dtype=np.uint8)
                self._h5_world_offset = np.array(
                    hf.attrs['world_offset_in_meters'], dtype=np.float32)
                self._road_ppm = float(hf.attrs.get('pixels_per_meter', self._ppm))
                wpx = hf.attrs.get('width_in_pixels', '?')
                wm = hf.attrs.get('width_in_meters', '?')
            if not np.isclose(self._road_ppm, self._ppm, rtol=0.01, atol=0.01):
                print(
                    f'[BEVDynamicRenderer] WARNING: H5 pixels_per_meter={self._road_ppm} '
                    f'!= config {self._ppm} — road 워핑은 H5 기준 ppm 사용'
                )
            print(f'[BEVDynamicRenderer] Static road H5 로드: {p}')
            morai_bev_dlog(
                'h5',
                'road nonzero=%d / %d attrs ppm=%s width_px=%s width_m=%s offset=%s',
                int(np.count_nonzero(self._h5_road)),
                int(self._h5_road.size),
                str(self._road_ppm),
                str(wpx),
                str(wm),
                self._h5_world_offset.tolist(),
            )
        except Exception as e:
            print(f'[BEVDynamicRenderer] WARNING: H5 로드 실패 — road=0: {e}')
            self._h5_road = None
            self._h5_world_offset = None

    def _affine_bev_from_ego_map_px(
            self, ego_x: float, ego_y: float, ego_yaw_deg: float) -> np.ndarray:
        """CARLA chauffeurnet._get_warp_transform — 맵 픽셀 좌표 기준 BEV 아핀 (2×3)."""
        ox, oy = float(self._h5_world_offset[0]), float(self._h5_world_offset[1])
        ppm = self._road_ppm
        ev_px = np.array(
            [ppm * (ego_x - ox), ppm * (ego_y - oy)], dtype=np.float32)
        yaw = np.deg2rad(ego_yaw_deg)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        right_vec = np.array(
            [np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)],
            dtype=np.float32)
        w = float(self._width)
        ev_b = float(self._ev_to_bottom)
        bottom_left = ev_px - ev_b * forward_vec - 0.5 * w * right_vec
        top_left = ev_px + (w - ev_b) * forward_vec - 0.5 * w * right_vec
        top_right = ev_px + (w - ev_b) * forward_vec + 0.5 * w * right_vec
        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array(
            [[0, self._width - 1], [0, 0], [self._width - 1, 0]],
            dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)

    def _get_road_mask(self, ego_x: float, ego_y: float, ego_yaw_deg: float) -> np.ndarray:
        out = np.zeros((self._width, self._width), dtype=np.uint8)
        if self._h5_road is None or self._h5_world_offset is None:
            return out
        m_map = self._affine_bev_from_ego_map_px(ego_x, ego_y, ego_yaw_deg)
        warped = cv.warpAffine(self._h5_road, m_map, (self._width, self._width))
        out[warped > 0] = np.uint8(255)
        return out

    def _get_route_mask_world(
            self,
            _ego_state: EgoState,
            M_warp: np.ndarray,
            route_world_xy: Optional[np.ndarray],
    ) -> np.ndarray:
        """세계 좌표 웨이포인트 경로 — 차선과 동일 아핀(M_warp, 미터 기준)."""
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if route_world_xy is None or len(route_world_xy) < 2:
            return mask
        rw = np.asarray(route_world_xy, dtype=np.float32)
        if rw.ndim != 2 or rw.shape[1] < 2:
            return mask
        n = min(rw.shape[0], self._max_route_wps)
        pts = rw[:n, :2].reshape(-1, 1, 2).astype(np.float32)
        pts_bev = cv.transform(pts, M_warp)
        pts_bev = np.ascontiguousarray(np.round(pts_bev).astype(np.int32))
        cv.polylines(
            mask, [pts_bev], isClosed=False,
            color=255, thickness=self._route_thickness)
        return mask

    # ───────────────────────────────────────────────────────────────
    # 팩토리 메서드
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config):
        """map_to_h5.Config 로부터 렌더러를 생성한다."""
        map_dir = Path(config.root) / 'morai_gym' / 'core' / 'obs_manager' / 'birdview' / 'map'
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
            lane_thickness=2,
            lane_solid_value=config.lane_solid,
            lane_broken_value=config.lane_broken,
            link_data=link_data,
            link_max_range=50.0,
            link_thickness=1,
            static_h5_path=getattr(config, 'static_h5', None),
            max_route_waypoints=config.max_wps,
            route_line_thickness=config.route_thick,
            scale_mask_col=config.scale_mask_col,
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
        route_world_xy: Optional[np.ndarray] = None,
    ) -> dict:
        """현재 프레임 데이터를 히스토리에 추가하고 BEV를 렌더링한다.

        Args:
            ego_state: Ego 차량 상태 (UdpManager.ego_state).
            vehicle_list: 주변 차량 리스트 (UdpManager.vehicle_list).
            pedestrian_list: 보행자 리스트 (UdpManager.pedestrian_list).
            traffic_light: 신호등 상태 (UdpManager.traffic_light). None 가능.
            route_world_xy: (N, 2) 이상 — 세계 좌표 경로 웨이포인트. 없으면 route 채널은 0.

        Returns:
            dict:
                'rendered': (H, W, 3) uint8 — RGB 시각화 이미지.
                'masks': (3 + 3*K, H, W) uint8 — K=len(history_idx).
                    순서: road, route, lane, vehicle×K, walker×K, tl×K.
                'collision_px': bool — chauffeurnet: ev_mask_col & walker_masks[-1].
        """
        # ── 1) 거리 기반 필터링 (chauffeurnet is_within_distance와 동일 박스) ──
        vehicles = self._filter_by_distance(
            vehicle_list, ego_state, self._veh_dist)
        pedestrians = self._filter_by_distance(
            pedestrian_list, ego_state, self._ped_dist)

        # ── 2) bbox 스케일 — chauffeurnet get_observation
        #   scale_bbox True: 차량 scale 1.0, 보행자 2.0 + half-extent 최소 0.8m
        #   scale_bbox False: 보행자만 walker_scale (레거시 MORAI)
        if self._scale_bbox:
            vehicles = [self._carla_scale_actor_bbox(o, 1.0) for o in vehicles]
            pedestrians = [self._carla_scale_actor_bbox(o, 2.0) for o in pedestrians]
        elif self._walker_scale != 1.0:
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

        # ── 7) 정적 채널 (carla chauffeurnet 순서: road, route, lane) ──
        c_road = self._get_road_mask(
            ego_state.pos_x, ego_state.pos_y, ego_state.yaw)
        c_route = self._get_route_mask_world(
            ego_state, M_warp, route_world_xy)
        c_lane = self._get_lane_mask(ego_state)

        rendered = self._render_rgb(
            vehicle_masks, walker_masks, tl_masks,
            c_road, c_route, c_lane,
            ego_state=ego_state, M_warp=M_warp)

        # ── 8) masks: road, route, lane, vehicles..., walkers..., tls...
        c_vehicle = [m.astype(np.uint8) * 255 for m in vehicle_masks]
        c_walker = [m.astype(np.uint8) * 255 for m in walker_masks]
        c_tl = tl_masks
        k = len(self._history_idx)
        masks = np.stack(
            (c_road, c_route, c_lane, *c_vehicle, *c_walker, *c_tl),
            axis=0)
        assert masks.shape[0] == 3 + 3 * k, (masks.shape[0], k)

        ego_mask_col = self._render_ego_mask(
            ego_state, M_warp, bbox_scale=self._scale_mask_col)
        collision_px = False
        if walker_masks:
            collision_px = bool(np.any(ego_mask_col & walker_masks[-1]))

        self._debug_update_i += 1
        if morai_bev_debug_enabled() and (
                self._debug_update_i % morai_bev_debug_every_n() == 0):
            morai_bev_dlog(
                'update',
                'frame=%d ego=(%.2f,%.2f) yaw=%.2f n_veh=%d n_ped=%d '
                'sum_road=%d sum_route=%d sum_lane=%d collision_px=%s '
                'tl_cache_keys=%d',
                self._debug_update_i,
                ego_state.pos_x,
                ego_state.pos_y,
                ego_state.yaw,
                len(vehicles),
                len(pedestrians),
                int(np.sum(c_road > 0)),
                int(np.sum(c_route > 0)),
                int(np.sum(c_lane > 0)),
                collision_px,
                len(self._tl_state_cache),
            )

        return {
            'rendered': rendered,
            'masks': masks,
            'collision_px': collision_px,
        }

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

    def _get_lane_mask(self, ego_state: EgoState) -> np.ndarray:
        """Ego 주변의 차선을 BEV 마스크에 렌더링한다.

        마스크 값 (uint8):
          - 255: 실선(Solid)
          - 120: 점선(Broken/Dashed)
        """
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not self._lane_markings or ego_state is None:
            return mask

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

        # Ego의 전방·우측 단위벡터 — chauffeurnet._get_warp_transform 과 동일 (yaw + pi/2)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi),
                              np.sin(yaw + 0.5 * np.pi)])

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

    def _carla_scale_actor_bbox(self, obj: ObjectData, scale: float) -> ObjectData:
        """chauffeurnet._get_surrounding_actors(..., scale): half-extent에 scale 후 min 0.8m."""
        d = (self._default_veh_size if obj.obj_type == OBJ_TYPE_VEHICLE
             else self._default_ped_size)
        if obj.size_x > 0 and obj.size_y > 0:
            hx = obj.size_x / 2.0
            hy = obj.size_y / 2.0
        else:
            hx = d[0] / 2.0
            hy = d[1] / 2.0
        hx = max(hx * scale, 0.8)
        hy = max(hy * scale, 0.8)
        return replace(obj, size_x=hx * 2.0, size_y=hy * 2.0)

    def _filter_by_distance(self, obj_list: List[ObjectData],
                            ego: EgoState, max_dist: float):
        """chauffeurnet is_within_distance: |dx|,|dy|<max_dist, |dz|<8, ego 1m 박스 제외."""
        filtered = []
        for obj in obj_list:
            dx = abs(obj.pos_x - ego.pos_x)
            dy = abs(obj.pos_y - ego.pos_y)
            dz = abs(obj.pos_z - ego.pos_z)
            if dz >= 8.0:
                continue
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

    def _render_ego_mask(
            self, ego_state: EgoState, M_warp,
            bbox_scale: float = 1.0):
        """Ego 차량 BEV 마스크. bbox_scale=scale_mask_col 시 chauffeurnet ev_mask_col."""
        mask = np.zeros([self._width, self._width], dtype=np.uint8)

        half_x = ego_state.size_x / 2.0 if ego_state.size_x > 0 else 2.25
        half_y = ego_state.size_y / 2.0 if ego_state.size_y > 0 else 1.0
        half_x *= float(bbox_scale)
        half_y *= float(bbox_scale)

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
                    road_u8, route_u8, lane_u8, ego_state, M_warp):
        """디버깅/시각화용 RGB. carla chauffeurnet.py render 블록과 동일 순서·색.

        순서: road → route → lane(all/broken) → TL → vehicles → walkers → ego.
        (CARLA의 stop_sign 마스크는 MORAI에 없어 생략.)
        """
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)

        if road_u8 is not None:
            image[road_u8 > 0] = COLOR_ALUMINIUM_5
        if route_u8 is not None:
            image[route_u8 > 0] = COLOR_ALUMINIUM_3

        if lane_u8 is not None and np.any(lane_u8):
            solid_mask = (lane_u8 == self._lane_solid_value)
            broken_mask = (lane_u8 == self._lane_broken_value)
            image[solid_mask] = COLOR_MAGENTA
            image[broken_mask] = COLOR_MAGENTA_2

        h_len = len(self._history_idx) - 1

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

        for i, mask in enumerate(vehicle_masks):
            if np.any(mask):
                image[mask] = _tint(COLOR_BLUE, (h_len - i) * 0.2)

        for i, mask in enumerate(walker_masks):
            if np.any(mask):
                image[mask] = _tint(COLOR_CYAN, (h_len - i) * 0.2)

        ego_mask = self._render_ego_mask(ego_state, M_warp, bbox_scale=1.0)
        image[ego_mask] = COLOR_WHITE

        return image
