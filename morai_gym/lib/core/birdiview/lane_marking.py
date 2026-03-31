import json
import numpy as np
import cv2 as cv
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from network.UDP.protocol import EgoState
from morai_gym.lib.core.birdiview.bev_render import BEVDynamicRenderer


class LaneMarkingLoader:
    def __init__(self, json_path):
        self.raw_data = self._load_json(json_path)
        if self.raw_data is None:
            print("❌ [LaneLoader] JSON 로드 실패")
            return

        self.lanes = self._parse_lanes(self.raw_data)
        print(f"✅ [LaneLoader] 총 {len(self.lanes)}개 차선 로드 완료")
        self._print_enu_range()

    def _load_json(self, path: str):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"❌ [LaneLoader] 파일 없음: {abs_path}")
            return None
        with open(abs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📂 [LaneLoader] JSON 로드 성공: {abs_path}")
        return data

    def _parse_lanes(self, raw: list) -> list:
        lanes = []
        for item in raw:
            pts = np.array(item['points'], dtype=np.float64)
            lane = {
                'idx': item.get('idx', ''),
                'points': pts,
                'lane_type': item.get('lane_type', -1),
                'lane_color': item.get('lane_color', 'unknown'),
            }
            lanes.append(lane)
        return lanes

    def _print_enu_range(self):
        if not self.lanes:
            return
        all_pts = np.vstack([l['points'] for l in self.lanes])
        print("-" * 40)
        for axis, name in enumerate(['X(East)', 'Y(North)', 'Z(Up)']):
            lo, hi = all_pts[:, axis].min(), all_pts[:, axis].max()
            print(f"📍 ENU {name}: [{lo:.2f}, {hi:.2f}] (범위: {hi-lo:.2f}m)")
        print("-" * 40)

    def _get_warp_transform(
        self,
        ego_x: float, ego_y: float, ego_yaw_deg: float,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
    ) -> np.ndarray:
        yaw = np.deg2rad(ego_yaw_deg)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec   = np.array([np.cos(yaw - 0.5 * np.pi),
                                 np.sin(yaw - 0.5 * np.pi)])

        mpp = 1.0 / pixels_per_meter
        ego_pos = np.array([ego_x, ego_y])

        bottom_left = (ego_pos
                       - pixels_ev_to_bottom * mpp * forward_vec
                       - 0.5 * width * mpp * right_vec)
        top_left    = (ego_pos
                       + (width - pixels_ev_to_bottom) * mpp * forward_vec
                       - 0.5 * width * mpp * right_vec)
        top_right   = (ego_pos
                       + (width - pixels_ev_to_bottom) * mpp * forward_vec
                       + 0.5 * width * mpp * right_vec)

        src_pts = np.stack(
            (bottom_left, top_left, top_right), axis=0
        ).astype(np.float32)

        dst_pts = np.array(
            [[0,         width - 1],
             [0,         0        ],
             [width - 1, 0        ]],
            dtype=np.float32
        )

        return cv.getAffineTransform(src_pts, dst_pts)

    def get_lane_bev_mask(
        self,
        ego_x: float, ego_y: float, ego_yaw_deg: float,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        max_range: float = 50.0,
        line_thickness: int = 2,
    ) -> np.ndarray:
        M_warp = self._get_warp_transform(
            ego_x, ego_y, ego_yaw_deg,
            width, pixels_per_meter, pixels_ev_to_bottom
        )

        mask = np.zeros((width, width), dtype=np.uint8)

        for lane in self.lanes:
            pts_enu = lane['points']

            dist = np.sqrt(
                (pts_enu[:, 0] - ego_x) ** 2 +
                (pts_enu[:, 1] - ego_y) ** 2
            )
            pts_in_range = pts_enu[dist <= max_range]
            if len(pts_in_range) < 2:
                continue

            pts_xy = pts_in_range[:, :2].astype(np.float32).reshape(-1, 1, 2)
            pts_bev = cv.transform(pts_xy, M_warp)
            pts_bev = np.ascontiguousarray(
                np.round(pts_bev).astype(np.int32)
            )

            cv.polylines(
                mask,
                [pts_bev],
                isClosed=False,
                color=255,
                thickness=line_thickness,
            )

        return mask

    def get_lane_bev_mask_from_ego(
        self,
        ego_state: EgoState,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        max_range: float = 50.0,
        line_thickness: int = 2,
    ) -> np.ndarray:
        if ego_state is None:
            print("⚠️  [LaneLoader] ego_state가 None — 빈 마스크 반환")
            return np.zeros((width, width), dtype=np.uint8)

        return self.get_lane_bev_mask(
            ego_x               = ego_state.pos_x,
            ego_y               = ego_state.pos_y,
            ego_yaw_deg         = ego_state.yaw,
            width               = width,
            pixels_per_meter    = pixels_per_meter,
            pixels_ev_to_bottom = pixels_ev_to_bottom,
            max_range           = max_range,
            line_thickness      = line_thickness,
        )

    def render_bev_with_lanes(
        self,
        ego_x: float, ego_y: float, ego_yaw_deg: float,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        max_range: float = 50.0,
        lane_color: tuple = (255, 255, 255),
        bg_color: tuple = (30, 30, 30),
    ) -> np.ndarray:
        mask = self.get_lane_bev_mask(
            ego_x, ego_y, ego_yaw_deg,
            width, pixels_per_meter, pixels_ev_to_bottom,
            max_range,
        )

        image = np.full((width, width, 3), bg_color, dtype=np.uint8)
        image[mask == 255] = lane_color

        ego_px = width // 2
        ego_py = width - pixels_ev_to_bottom
        cv.circle(image, (ego_px, ego_py), radius=5,
                  color=(255, 50, 50), thickness=-1)
        cv.arrowedLine(
            image,
            (ego_px, ego_py),
            (ego_px, ego_py - 20),
            color=(255, 50, 50),
            thickness=2,
            tipLength=0.3,
        )

        return image


# ═══════════════════════════════════════════════════════════════════
# ✅ BEV 관측값 통합 — lane + dynamic
# ═══════════════════════════════════════════════════════════════════

class BEVObservationBuilder:
    """
    차선(정적) + 동적 객체(차량/보행자/신호등) BEV 마스크를 합쳐
    RL 에이전트용 최종 관측값을 생성한다.

    최종 관측값 shape: (13, 192, 192) uint8
        ch  0~ 3 : vehicle    × 4 히스토리   (0 or 255)
        ch  4~ 7 : walker     × 4 히스토리   (0 or 255)
        ch  8~11 : tl stopline× 4 히스토리   (0/80/170/255)
        ch 12    : lane marking               (0 or 255)
    """

    def __init__(
        self,
        lane_loader: LaneMarkingLoader,
        bev_renderer: BEVDynamicRenderer,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        lane_max_range: float = 50.0,
    ):
        """
        Args:
            lane_loader  : 초기화된 LaneMarkingLoader 인스턴스.
            bev_renderer : 초기화된 BEVDynamicRenderer 인스턴스.
            width, pixels_per_meter, pixels_ev_to_bottom: BEV 파라미터.
                두 렌더러가 동일한 값을 써야 픽셀이 일치함.
            lane_max_range: 차선 렌더링 범위 (m).
        """
        self._lane_loader  = lane_loader
        self._bev_renderer = bev_renderer
        self._width        = width
        self._ppm          = pixels_per_meter
        self._ev_to_bottom = pixels_ev_to_bottom
        self._lane_range   = lane_max_range

    def build(
        self,
        ego_state: EgoState,
        vehicle_list:    list,
        pedestrian_list: list,
        traffic_light,
    ) -> dict:
        """
        매 프레임 호출 — 최종 BEV 관측값을 반환한다.

        Args:
            ego_state       : UdpManager.ego_state
            vehicle_list    : UdpManager.vehicle_list
            pedestrian_list : UdpManager.pedestrian_list
            traffic_light   : UdpManager.traffic_light

        Returns:
            dict:
                'obs'      : (13, H, W) uint8  ← RL 에이전트 입력
                'rendered' : (H, W, 3)  uint8  ← 디버깅용 RGB 이미지
        """
        # ── 1) 동적 객체 마스크 (12채널) + RGB 시각화 ──
        dynamic_result = self._bev_renderer.update(
            ego_state,
            vehicle_list,
            pedestrian_list,
            traffic_light,
        )
        dynamic_masks = dynamic_result['masks']    # (12, 192, 192) uint8
        rendered      = dynamic_result['rendered'] # (192, 192, 3)  uint8

        # ── 2) 차선 마스크 (1채널) ──
        lane_mask = self._lane_loader.get_lane_bev_mask_from_ego(
            ego_state,
            width               = self._width,
            pixels_per_meter    = self._ppm,
            pixels_ev_to_bottom = self._ev_to_bottom,
            max_range           = self._lane_range,
        )                                          # (192, 192) uint8

        # ── 3) 채널 축 추가 후 결합 ──
        lane_channel = lane_mask[np.newaxis, :, :] # (1, 192, 192)
        obs = np.concatenate(
            [dynamic_masks, lane_channel], axis=0  # (13, 192, 192)
        )

        # ── 4) 디버깅용 RGB에 차선 오버레이 ──
        # rendered는 BEVDynamicRenderer가 만든 RGB 위에
        # 차선을 흰색으로 추가로 그려서 한 화면에 확인
        rendered_with_lanes = rendered.copy()
        rendered_with_lanes[lane_mask == 255] = (200, 200, 200)  # 밝은 회색

        return {
            'obs'      : obs,                  # (13, 192, 192) — RL 입력
            'rendered' : rendered_with_lanes,  # (192, 192, 3)  — 시각화
        }

    def reset(self):
        """에피소드 시작 시 히스토리 초기화."""
        self._bev_renderer.reset()


# ═══════════════════════════════════════════════════════════════════
# 사용 예시 — UdpManager + BEVObservationBuilder 통합 루프
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    from network.UDP.udp_manager import UdpManager
    from morai_gym.lib.core.birdiview.bev_render import (
        BEVDynamicRenderer, TrafficLightStoplineMapper
    )

    TARGET_PATH  = r'C:\Users\DELL\Desktop\morai_roach\morai_gym\lib\core\birdiview\map\lane_marking_set.json'
    TL_JSON      = r'C:\Users\DELL\Desktop\morai_roach\morai_gym\lib\core\birdiview\map\traffic_light_set.json'
    STOPLINE_JSON= r'C:\Users\DELL\Desktop\morai_roach\morai_gym\lib\core\birdiview\map\stoplane_marking_set.json'

    # ── 1) 초기화 ──
    lane_loader = LaneMarkingLoader(TARGET_PATH)

    tl_mapper = TrafficLightStoplineMapper(TL_JSON, STOPLINE_JSON)
    bev_renderer = BEVDynamicRenderer(tl_mapper=tl_mapper)

    obs_builder = BEVObservationBuilder(
        lane_loader  = lane_loader,
        bev_renderer = bev_renderer,
    )

    # ── 2) UDP 수신 시작 ──
    manager = UdpManager()
    manager.start()

    print("⏳ ego_state 수신 대기 중...")
    while not manager.is_ready:
        time.sleep(0.05)
    print("✅ ego_state 수신 완료")

    obs_builder.reset()  # 히스토리 초기화

    # ── 3) 매 프레임 루프 ──
    try:
        while True:
            result = obs_builder.build(
                ego_state       = manager.ego_state,
                vehicle_list    = manager.vehicle_list,
                pedestrian_list = manager.pedestrian_list,
                traffic_light   = manager.traffic_light,
            )

            obs      = result['obs']       # (13, 192, 192) → RL 에이전트로
            rendered = result['rendered']  # (192, 192, 3)  → 시각화

            print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")

            cv.imshow('BEV', cv.cvtColor(rendered, cv.COLOR_RGB2BGR))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)  # 10Hz

    finally:
        manager.stop()
        cv.destroyAllWindows()
# ```

# ---

# ## 구조 요약
# ```
# BEVObservationBuilder.build()
#     ├─ BEVDynamicRenderer.update()  → dynamic_masks (12, 192, 192)
#     │       차량 × 4 히스토리
#     │       보행자 × 4 히스토리
#     │       신호등 × 4 히스토리
#     │
#     ├─ LaneMarkingLoader.get_lane_bev_mask_from_ego()  → lane_mask (192, 192)
#     │
#     └─ np.concatenate([dynamic_masks, lane_channel])
#             → obs (13, 192, 192)  ← RL 에이전트 입력