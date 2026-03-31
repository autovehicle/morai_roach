"""
BEV 동적 객체 렌더링 테스트 스크립트.

MORAI 시뮬레이터에 연결하여 차량, 보행자, 신호등 데이터를
실시간으로 BEV 이미지에 마스킹하고 시각화한다.

사용법:
    1. MORAI 시뮬레이터를 실행하고 시나리오를 로드한다.
    2. 이 스크립트를 실행한다:
       python run/test_bev_dynamic.py

    키보드:
       q / ESC: 종료
       s: 현재 프레임 저장 (rendered + masks)
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import cv2 as cv

from network.UDP.udp_manager import UdpManager
from morai_gym.lib.core.birdiview.map_to_h5 import Config
from morai_gym.lib.core.birdiview.bev_render import BEVDynamicRenderer


def test_lane_rendering():
    """MORAI 없이 차선 렌더링만 테스트."""
    
    # 더미 ego_state (원점 근처)
    class DummyEgo:
        def __init__(self):
            self.pos_x = 0.0
            self.pos_y = 0.0
            self.yaw = 0.0
    
    ego = DummyEgo()
    
    # Config 로드
    config = Config(project_root)
    
    # BEV renderer 생성 (lane markings 포함)
    renderer = BEVDynamicRenderer.from_config(config)
    
    # 차선 마스크 생성 (renderer의 lane markings 사용)
    lane_mask = renderer._get_lane_mask(ego)
    
    print(f'Lane mask shape: {lane_mask.shape}, dtype: {lane_mask.dtype}')
    print(f'Lane pixels: {np.sum(lane_mask > 0)}')
    print(f'Sample line_width: {renderer._lane_markings[0]["line_width"] if renderer._lane_markings else "none"}')
    
    # 시각화
    display_size = 512
    lane_bgr = cv.cvtColor(lane_mask, cv.COLOR_GRAY2BGR)
    display = cv.resize(lane_bgr, (display_size, display_size), interpolation=cv.INTER_NEAREST)
    
    cv.imshow('Lane Mask Test', display)
    cv.waitKey(1000)  # 1초 대기 후 자동 종료
    cv.destroyAllWindows()


def main():
    # ── config 경로 로드 (birdview.yaml) ──
    config = Config(project_root)

    # ── 1) UDP 매니저 시작 ──
    manager = UdpManager()
    manager.start()

    # ── 2) BEV 렌더러 생성 (birdview.yaml 설정 반영) ──
    renderer = BEVDynamicRenderer.from_config(config)

    print('\n[test_bev_dynamic] MORAI 시뮬레이터에서 데이터 대기 중...')
    print('  q/ESC: 종료, s: 현재 프레임 저장\n')

    # Ego 데이터가 수신될 때까지 대기
    while not manager.is_ready:
        time.sleep(0.1)
    print('[test_bev_dynamic] Ego 데이터 수신 완료. 렌더링 시작.\n')

    frame_count = 0
    save_dir = project_root / 'collected_data' / 'bev_test'

    try:
        while True:
            ego = manager.ego_state
            if ego is None:
                time.sleep(0.01)
                continue

            # ── BEV 렌더링 ──
            result = renderer.update(
                ego_state=ego,
                vehicle_list=manager.vehicle_list,
                pedestrian_list=manager.pedestrian_list,
                traffic_light=manager.traffic_light,
            )

            rendered = result['rendered']   # (192, 192, 3) RGB
            masks = result['masks']         # (12, 192, 192) uint8

            # ── 시각화 (확대 표시) ──
            display_size = 512 
            
            rendered_bgr = cv.cvtColor(rendered, cv.COLOR_RGB2BGR)
            cv.imshow('BEV Original (192x192)', rendered_bgr)
            display = cv.resize(rendered_bgr, (display_size, display_size),
                                interpolation=cv.INTER_NEAREST)

            # 상태 텍스트 오버레이
            n_veh = len(manager.vehicle_list)
            n_ped = len(manager.pedestrian_list)
            tl = manager.traffic_light
            tl_str = f'{tl.status}' if tl else 'N/A'

            info_lines = [
                f'Frame: {frame_count}',
                f'Ego: ({ego.pos_x:.1f}, {ego.pos_y:.1f}) yaw={ego.yaw:.1f}',
                f'Vehicles: {n_veh}  Peds: {n_ped}  TL: {tl_str}',
            ]
            for i, line in enumerate(info_lines):
                cv.putText(display, line, (10, 20 + i * 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.imshow('BEV Dynamic Objects', display)

            # ── 개별 채널 시각화 (선택적) ──
            n_hist = len(renderer._history_idx)

            # 최신 프레임의 차량/보행자/신호등/차선 마스크
            veh_mask = masks[n_hist - 1]        # 가장 최근 차량
            ped_mask = masks[2 * n_hist - 1]    # 가장 최근 보행자
            tl_mask = masks[3 * n_hist - 1]     # 가장 최근 신호등
            lane_mask = masks[3 * n_hist] if masks.shape[0] > 3 * n_hist else np.zeros_like(veh_mask)

            channel_display = np.hstack([veh_mask, ped_mask, tl_mask, lane_mask])
            channel_display = cv.resize(
                channel_display, (display_size * 4, display_size),
                interpolation=cv.INTER_NEAREST)
            cv.putText(channel_display, 'Vehicle', (10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
            cv.putText(channel_display, 'Pedestrian', (display_size + 10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
            cv.putText(channel_display, 'Traffic Light', (display_size * 2 + 10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
            cv.putText(channel_display, 'Lane', (display_size * 3 + 10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
            cv.imshow('BEV Channels (latest)', channel_display)

            # ── 키 입력 처리 ──
            key = cv.waitKey(100) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):
                save_dir.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(save_dir / f'rendered_{frame_count:05d}.png'),
                           rendered_bgr)
                np.save(str(save_dir / f'masks_{frame_count:05d}.npy'), masks)
                print(f'  [saved] frame {frame_count} → {save_dir}')

            frame_count += 1

    except KeyboardInterrupt:
        print('\n[test_bev_dynamic] 중단됨.')
    finally:
        cv.destroyAllWindows()
        manager.stop()
        print(f'[test_bev_dynamic] 총 {frame_count} 프레임 처리 완료.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test-lane':
        test_lane_rendering()
    else:
        main()
