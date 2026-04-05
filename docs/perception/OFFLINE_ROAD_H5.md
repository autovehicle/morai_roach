# 오프라인 Road H5 베이킹 (Perception)

## 목적

- **실시간**에 `link_set.json` 을 매 프레임 파싱해 그리는 비용을 없애고, CARLA Roach와 같이 **시뮬 전에 한 번** 맵 래스터를 굽는다.
- 산출물은 `bev_render` 가 읽는 **`morai_kcity_map.h5`** 의 `road` 데이터셋과 HDF5 attrs (`pixels_per_meter`, `world_offset_in_meters`, …).

## 실행 (Windows CMD, conda)

```cmd
conda activate morai
cd C:\path\to\morai-roach
python -m morai_gym.utils.birdview_map
```

기본 입력: `morai_gym/core/obs_manager/birdview/map/link_set.json`  
기본 출력: `morai_gym/core/obs_manager/birdview/map/morai_kcity_map.h5`

옵션 예:

```cmd
python -m morai_gym.utils.birdview_map --pixels_per_meter 5.0 --margin_m 100 --lane_width_m 3.5
```

`conda activate morai` 후에는 해당 환경의 **pip** 로 패키지 설치가 가능하다 (`pip install -e .`, `pip install -r requirements.txt`).

## MORAI 통신·토픽과의 관계

- 이 파이프라인은 **JSON → NumPy → HDF5** 만 다룬다. **UDP/ROS 토픽명·포트·메시지 정의를 변경하지 않는다.**
- MORAI SIM ↔ 사용자 코드 통신은 기존 `network/UDP` 및 공식 문서의 키(`ego_info_dst_port` 등)를 그대로 쓴다.

## 참고 소스

| 자료 | 용도 |
|------|------|
| carla-roach `carla_gym/utils/birdview_map.py` | H5 attrs, gzip, `road` 레이어 의미 |
| MORAI-MGeoModule `subproc_load_link_ver2.py` | `link_set` 필드(`points`, `width_start`, …) |
| morai-roach `link_set.json` | K-city 링크 geometry |

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-04-05 | `morai_gym/utils/birdview_map.py` 최초 추가, 문서 작성 |
