#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MORAI K-city (및 동일 포맷) link_set.json 으로부터 Roach/CARLA 호환 **road** 래스터를
**오프라인에서 한 번만** 생성해 HDF5로 저장한다.

설계 철학 (논문 코드와의 정합)
----------------------------
- CARLA 쪽은 시뮬 가동 후 ``carla_gym/utils/birdview_map.py`` 의 ``MapImage.draw_map_image`` 로
  맵 전체를 pygame 에 그린 뒤 ``*.h5`` 로 굽는다.
- MORAI 에서는 동일 정보가 MGeo보내기 JSON(``link_set.json``)에 담기므로,
  **ROS/UDP 없이** JSON 만 읽어 동일한 메타데이터 규약으로 ``road`` 레이어를 만든다.

H5 출력 규약 (``bev_render.BEVDynamicRenderer._try_load_static_h5`` / carla chauffeurnet)
----------------------------------------------------------------------------------------
- 데이터셋 ``road``: ``uint8`` 2D, 값 0/255 (주행 가능 영역=255).
- 속성 ``pixels_per_meter``: float, 기본 **5.0** (요구사항).
- 속성 ``world_offset_in_meters``: float32 shape ``(2,)`` — 맵 좌하단(최소 x, 최소 y) 월드 좌표(m).
- 속성 ``width_in_meters``, ``width_in_pixels``: CARLA 스크립트와 동일 의미(정사각 캔버스).

좌표 매핑 (CARLA ``MapImage.world_to_pixel`` 와 동일)
------------------------------------------------------
CARLA 는 pygame 에 점을 넣을 때 ``[round(y_map), round(x_map)]`` 형태를 사용한다
(``x_map = ppm * (world.x - ox)``, ``y_map = ppm * (world.y - oy)``).
OpenCV ``polylines`` 는 ``(x, y) = (열, 행)`` 이므로 동일 숫자를 그대로 사용한다.

MGeo / link_set 참고
--------------------
- 링크 필드 의미는 MORAI-MGeoModule ``save_load/subproc_load_link_ver2.py`` 의
  ``line_save_info_list`` 파싱과 동일하게 ``points``, ``width_start`` 등을 사용한다.
- ``points`` 가 문자열이면 float 로 변환한다 (MGeo 호환).

본 모듈은 **통신/ROS/토픽을 전혀 사용하지 않는다.**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import h5py
import numpy as np


def _parse_points(raw: Any) -> np.ndarray:
    """link 항목의 points 를 (N, 2+) float 배열로 만든다."""
    if raw is None or len(raw) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    first = raw[0][0]
    if isinstance(first, str):
        arr = np.array([[float(c) for c in row] for row in raw], dtype=np.float32)
    else:
        arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr


def load_link_entries(link_json_path: Path) -> List[Dict[str, Any]]:
    """link_set.json 전체를 리스트[dict] 로 로드한다."""
    with open(link_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'link_set.json 최상위는 리스트여야 합니다: {link_json_path}')
    return data


def _link_lane_width_m(entry: Dict[str, Any], default_m: float) -> float:
    """차로 폭(m). MGeo 의 width_start 등을 우선."""
    for key in ('width_start', 'width_end'):
        v = entry.get(key)
        if v is not None:
            try:
                w = float(v)
                if w > 0.5:
                    return w
            except (TypeError, ValueError):
                pass
    return default_m


def compute_world_bounds(
    entries: List[Dict[str, Any]],
    margin_m: float,
) -> Tuple[float, float, float, float]:
    """모든 링크 포인트를 포함하는 축 정렬 AABB + margin (미터)."""
    xs: List[float] = []
    ys: List[float] = []
    for e in entries:
        pts = _parse_points(e.get('points'))
        if pts.shape[0] == 0:
            continue
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
    if not xs:
        raise ValueError('유효한 link points 가 없습니다.')
    min_x, max_x = min(xs) - margin_m, max(xs) + margin_m
    min_y, max_y = min(ys) - margin_m, max(ys) + margin_m
    return min_x, min_y, max_x, max_y


def world_to_pixel_carla(wx: float, wy: float, ppm: float, world_offset: np.ndarray) -> Tuple[int, int]:
    """
    CARLA MapImage.world_to_pixel 과 동일한 스케일·축 순서.

    pygame 점 (x, y) = (round(ppm*(wy-oy)), round(ppm*(wx-ox))).
    OpenCV polylines 의 (x, y)에 그대로 사용.
    """
    xm = ppm * (wx - float(world_offset[0]))
    ym = ppm * (wy - float(world_offset[1]))
    return int(round(ym)), int(round(xm))


def bake_road_mask_from_links(
    entries: List[Dict[str, Any]],
    pixels_per_meter: float = 5.0,
    margin_m: float = 100.0,
    default_lane_width_m: float = 3.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    link 중심선을 두꺼운 폴리라인으로 그려 주행 가능(road) 마스크를 만든다.

    Returns:
        road: uint8 (H, H), 0 또는 255
        meta: world_offset, width_in_meters, width_in_pixels, pixels_per_meter
    """
    min_x, min_y, max_x, max_y = compute_world_bounds(entries, margin_m)
    world_offset = np.array([min_x, min_y], dtype=np.float32)
    width_m = max(max_x - min_x, max_y - min_y)
    width_px = int(round(pixels_per_meter * width_m))
    if width_px < 16:
        raise ValueError(f'맵 픽셀 크기가 너무 작습니다: {width_px}')

    road = np.zeros((width_px, width_px), dtype=np.uint8)

    for entry in entries:
        pts = _parse_points(entry.get('points'))
        if pts.shape[0] < 2:
            continue
        w_m = _link_lane_width_m(entry, default_lane_width_m)
        thickness = max(1, int(round(w_m * pixels_per_meter)))

        cv_pts = np.array(
            [
                world_to_pixel_carla(float(px), float(py), pixels_per_meter, world_offset)
                for px, py in pts[:, :2]
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        cv.polylines(road, [cv_pts], isClosed=False, color=255, thickness=thickness)

    meta = {
        'world_offset_in_meters': world_offset,
        'width_in_meters': float(width_m),
        'width_in_pixels': float(width_px),
        'pixels_per_meter': float(pixels_per_meter),
    }
    return road, meta


def save_roach_h5(
    output_path: Path,
    road: np.ndarray,
    meta: Dict[str, Any],
    compression_level: int = 9,
) -> None:
    """
    carla birdview_map.py 와 동일한 gzip 압축으로 ``road`` 저장.

    chauffeurnet 은 ``lane_marking_*`` 등도 읽지만, morai 측 BEV 로더는
    현재 ``road`` 만 필수이므로 나머지는 생략한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as hf:
        hf.attrs['pixels_per_meter'] = meta['pixels_per_meter']
        hf.attrs['world_offset_in_meters'] = meta['world_offset_in_meters']
        hf.attrs['width_in_meters'] = meta['width_in_meters']
        hf.attrs['width_in_pixels'] = meta['width_in_pixels']
        hf.create_dataset(
            'road',
            data=road,
            compression='gzip',
            compression_opts=compression_level,
        )


def bake_from_cli(
    link_json: Path,
    output_h5: Path,
    pixels_per_meter: float,
    margin_m: float,
    default_lane_width_m: float,
) -> None:
    entries = load_link_entries(link_json)
    road, meta = bake_road_mask_from_links(
        entries,
        pixels_per_meter=pixels_per_meter,
        margin_m=margin_m,
        default_lane_width_m=default_lane_width_m,
    )
    save_roach_h5(output_h5, road, meta)
    print(f'[birdview_map] wrote {output_h5}')
    print(
        f'  shape={road.shape}, ppm={meta["pixels_per_meter"]}, '
        f'offset={meta["world_offset_in_meters"].tolist()}, '
        f'width_m={meta["width_in_meters"]:.1f}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MORAI link_set.json → Roach 호환 road H5 (오프라인, ROS/UDP 없음)',
    )
    parser.add_argument(
        '--link_json',
        type=Path,
        default=None,
        help='link_set.json 경로 (기본: morai_gym/.../birdview/map/link_set.json)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='출력 H5 (기본: morai_gym/.../birdview/map/morai_kcity_map.h5)',
    )
    parser.add_argument('--pixels_per_meter', type=float, default=5.0)
    parser.add_argument('--margin_m', type=float, default=100.0)
    parser.add_argument('--lane_width_m', type=float, default=3.5)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    default_link = (
        repo / 'morai_gym' / 'core' / 'obs_manager' / 'birdview' / 'map' / 'link_set.json'
    )
    default_out = (
        repo / 'morai_gym' / 'core' / 'obs_manager' / 'birdview' / 'map' / 'morai_kcity_map.h5'
    )

    link_json = args.link_json or default_link
    output_h5 = args.output or default_out

    if not link_json.is_file():
        raise SystemExit(f'link_set.json 없음: {link_json}')

    bake_from_cli(
        link_json,
        output_h5,
        pixels_per_meter=args.pixels_per_meter,
        margin_m=args.margin_m,
        default_lane_width_m=args.lane_width_m,
    )


if __name__ == '__main__':
    main()
