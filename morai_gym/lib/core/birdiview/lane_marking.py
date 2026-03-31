import json
import numpy as np
import cv2 as cv
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from network.UDP.protocol import EgoState


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
                'points': pts,              # ENU [x, y, z] — 원본 보존
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

    # ═══════════════════════════════════════════════════════════════
    # BEV 렌더러와 동일한 M_warp 생성
    # ═══════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════
    # 차선 BEV 렌더링
    # ═══════════════════════════════════════════════════════════════

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

            # ── 점별 거리 필터링 → 범위 안 점만 사용 ──
            dist = np.sqrt(
                (pts_enu[:, 0] - ego_x) ** 2 +
                (pts_enu[:, 1] - ego_y) ** 2
            )
            pts_in_range = pts_enu[dist <= max_range]
            if len(pts_in_range) < 2:
                continue

            # ── ENU (X, Y) → BEV 픽셀 ──
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

    # ───────────────────────────────────────────────────────────────
    # ✅ UdpManager.ego_state 연동 인터페이스
    # ───────────────────────────────────────────────────────────────

    def get_lane_bev_mask_from_ego(
        self,
        ego_state: EgoState,
        width: int = 192,
        pixels_per_meter: float = 5.0,
        pixels_ev_to_bottom: int = 40,
        max_range: float = 50.0,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        UdpManager.ego_state를 직접 받아서 차선 BEV 마스크를 반환한다.

        get_lane_bev_mask()의 래퍼 — ego_state 필드를 직접 언패킹해서 전달.

        Args:
            ego_state: UdpManager.ego_state (MORAI UDP 수신값).
                       None이면 빈 마스크 반환.
            width, pixels_per_meter, pixels_ev_to_bottom: BEV 파라미터.
            max_range: 차선 렌더링 범위 (m).
            line_thickness: 차선 선 두께 (pixels).

        Returns:
            (H, W) uint8 — 차선 마스크 (차선=255, 배경=0).
            ego_state가 None이면 zeros 반환.
        """
        # ego_state가 아직 수신되지 않은 경우 (UdpManager.is_ready == False)
        if ego_state is None:
            print("⚠️  [LaneLoader] ego_state가 None — 빈 마스크 반환")
            return np.zeros((width, width), dtype=np.uint8)

        return self.get_lane_bev_mask(
            ego_x           = ego_state.pos_x,
            ego_y           = ego_state.pos_y,
            ego_yaw_deg     = ego_state.yaw,   # MORAI: degree, ENU 기준 CCW+
            width           = width,
            pixels_per_meter = pixels_per_meter,
            pixels_ev_to_bottom = pixels_ev_to_bottom,
            max_range       = max_range,
            line_thickness  = line_thickness,
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
# 사용 예시 — UdpManager 연동
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    from network.UDP.udp_manager import UdpManager

    TARGET_PATH = r'C:\Users\DELL\Desktop\morai_roach\morai_gym\lib\core\birdiview\map\lane_marking_set.json'

    # ── 1) 차선 데이터 로드 (초기화 1회) ──
    loader = LaneMarkingLoader(TARGET_PATH)

    # ── 2) UDP 수신 시작 ──
    manager = UdpManager()
    manager.start()

    # ── 3) ego_state 수신 대기 ──
    print("⏳ ego_state 수신 대기 중...")
    while not manager.is_ready:
        time.sleep(0.05)
    print("✅ ego_state 수신 완료")

    # ── 4) 매 프레임 루프 ──
    try:
        while True:
            ego_state = manager.ego_state  # UdpManager가 항상 최신값 유지

            # EgoState → 차선 BEV 마스크
            lane_mask = loader.get_lane_bev_mask_from_ego(
                ego_state,
                width=192,
                pixels_per_meter=5.0,
                pixels_ev_to_bottom=40,
                max_range=50.0,
            )

            # 시각화 (디버깅용)
            bev_image = loader.render_bev_with_lanes(
                ego_state.pos_x,
                ego_state.pos_y,
                ego_state.yaw,
                max_range=50.0,
            )

            cv.imshow('Lane BEV', cv.cvtColor(bev_image, cv.COLOR_RGB2BGR))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)  # 10Hz

    finally:
        manager.stop()
        cv.destroyAllWindows()