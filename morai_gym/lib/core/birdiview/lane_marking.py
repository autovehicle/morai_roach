#!/usr/bin/env python3
"""
lane_loader.py
Morai lane_marking_set.json 을 로드하여 ENU 좌표 기반 차선 데이터를 파싱하는 ROS Node.
이후 단계: ENU -> base_link 좌표 변환 -> pixel 변환 -> BEV image 생성
"""

import rospy
import json
import numpy as np
import os


class LaneMarkingLoader:
    def __init__(self):
        rospy.init_node('lane_marking_loader', anonymous=True)

        # -------------------------------------------------------
        # 파라미터: rosrun 실행 시 _json_path:=... 로 덮어쓸 수 있음
        # -------------------------------------------------------
        json_path = '/home/wonjung/alpa_ws/morai_roach/morai_gym/lib/core/birdiview/map/lane_marking_set.json'

        # -------------------------------------------------------
        # 1) JSON 로드
        # -------------------------------------------------------
        raw = self._load_json(json_path)
        if raw is None:
            rospy.signal_shutdown("JSON 로드 실패")
            return

        # -------------------------------------------------------
        # 2) 차선 파싱  →  self.lanes : list of dict
        #    각 dict = {
        #      'idx'        : str            차선 ID
        #      'points'     : np.ndarray     shape (N, 3)  [x, y, z] ENU
        #      'lane_type'  : int
        #      'lane_color' : str
        #      'lane_shape' : list[str]
        #      'lane_width' : float
        #    }
        # -------------------------------------------------------
        self.lanes = self._parse_lanes(raw)
        rospy.loginfo(f"[LaneLoader] 총 {len(self.lanes)}개 차선 로드 완료")

        # -------------------------------------------------------
        # 3) 전체 ENU 범위 확인 (BEV 이미지 크기 설계에 활용)
        # -------------------------------------------------------
        self._print_enu_range()

    # ===========================================================
    # Private methods
    # ===========================================================

    def _load_json(self, path: str):
        """JSON 파일을 읽어 Python 객체로 반환"""
        if not os.path.exists(path):
            rospy.logerr(f"[LaneLoader] 파일 없음: {path}")
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        rospy.loginfo(f"[LaneLoader] JSON 로드: {path}  ({len(data)}개 항목)")
        return data

    def _parse_lanes(self, raw: list) -> list:
        """
        raw  : JSON 최상위 list (각 원소가 차선 하나)
        반환 : 파싱된 차선 dict 리스트
        """
        lanes = []
        for item in raw:
            pts = np.array(item['points'], dtype=np.float64)  # (N, 3)

            lane = {
                'idx'        : item.get('idx', ''),
                'points'     : pts,               # ENU [x, y, z]
                'lane_type'  : item.get('lane_type', -1),
                'lane_color' : item.get('lane_color', 'unknown'),
                'lane_shape' : item.get('lane_shape', []),
                'lane_width' : item.get('lane_width', 0.0),
            }
            lanes.append(lane)
        return lanes

    def _print_enu_range(self):
        """전체 포인트의 ENU 범위를 로그로 출력 (BEV 설계용)"""
        all_pts = np.vstack([l['points'] for l in self.lanes])  # (TotalN, 3)
        for axis, name in enumerate(['X(East)', 'Y(North)', 'Z(Up)']):
            lo, hi = all_pts[:, axis].min(), all_pts[:, axis].max()
            rospy.loginfo(f"  ENU {name}: [{lo:.2f}, {hi:.2f}]  범위={hi-lo:.2f} m")


# ===========================================================
# 실행 진입점
# ===========================================================
if __name__ == '__main__':
    try:
        node = LaneMarkingLoader()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass