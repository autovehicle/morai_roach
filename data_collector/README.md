# MORAI 데이터 수집 자동화

## 폴더 구조

```
data_collector/
├── collect.py                        ← 메인 실행 (여기서 실행)
├── requirements.txt
├── config/
│   └── collection_config.yaml        ← Zone / Scenario / 토픽 이름 설정
├── core/
│   ├── scenario_params.py            ← EpisodeParams 랜덤 생성
│   ├── episode_manager.py            ← 에피소드 1개 실행 루프
│   └── data_writer.py                ← 타임스탬프 동기화 + .npz 저장 + BEV 생성
├── ros/
│   └── ros_manager.py                ← ROS 토픽 구독 (Windows ROS1 Noetic)
├── simulator/
│   └── morai_client.py               ← MORAI gRPC 통신 (리셋/스폰/제어/경로)
└── expert/
    └── expert_controller.py          ← Pure Pursuit + PID rule-based expert
```

## 저장 폴더 구조

```
D:/morai_dataset/
└── zone_urban/
    └── scenario_stop_and_go/
        ├── episode_001/
        │   ├── frames/
        │   │   ├── 000001.npz
        │   │   ├── 000002.npz
        │   │   └── ...
        │   ├── episode_params.json    ← 랜덤 파라미터 (spawn 위치, NPC 수 등)
        │   └── episode_summary.json  ← 결과 요약 (성공 여부, 프레임 수 등)
        └── episode_002/
```

## .npz 파일 구조 (프레임 1개)

| 키 | shape | dtype | 설명 |
|---|---|---|---|
| `timestamp_ns` | (1,) | int64 | 타임스탬프 |
| `cam_front` | (H, W, 3) | uint8 | 전방 카메라 RGB |
| `cam_front_left` | (H, W, 3) | uint8 | 전방 좌측 |
| `cam_front_right` | (H, W, 3) | uint8 | 전방 우측 |
| `cam_rear_left` | (H, W, 3) | uint8 | 후방 좌측 |
| `cam_rear_right` | (H, W, 3) | uint8 | 후방 우측 |
| `ego` | (6,) | float32 | x, y, z, yaw, speed, steer |
| `gnss` | (6,) | float32 | lat, lon, alt, vel_x, vel_y, vel_z |
| `imu` | (6,) | float32 | accel xyz, gyro xyz |
| `gt_objects` | (N, 7) | float32 | id, type, x, y, z, vel_x, vel_y |
| `tl_states` | (M, 2) | int32 | tl_id_hash, state(0~9) |
| `nav_waypoints` | (N, 2) | float32 | 전역 경로 x, y |
| `expert` | (3,) | float32 | steer, throttle, brake |
| `bev_map` | (200, 200, 8) | float32 | 8채널 BEV 맵 |
| `is_longtail` | (1,) | bool | longtail 프레임 여부 |

## 실행 방법

```bash
# 설치
pip install -r requirements.txt

# urban / stop_and_go 10개 수집
python collect.py --zone urban --scenario stop_and_go --episodes 10

# urban 전체 시나리오 수집
python collect.py --zone urban --all-scenarios

# 전체 수집
python collect.py --all

# 중단 후 이어서
python collect.py --zone urban --scenario stop_and_go --resume

# 현황 확인
python collect.py --status
```

## 실제 MORAI 연동 전 수정 필요 항목

### 1. ros/ros_manager.py
- `_cb_gnss_morai()` : `morai_msgs/GPSMessage` 실제 필드명 확인
- `_cb_objects()` : `morai_msgs/ObjectStatusList` npc_list / obstacle_list 필드명 확인
- `_cb_traffic_light()` : `morai_msgs/TrafficLightStatus` 필드명 확인
- `_cb_ego()` : `morai_msgs/EgoVehicleStatus` position / heading / velocity 필드명 확인

### 2. simulator/morai_client.py
- proto 파일 컴파일 후 `morai_sim_pb2.py`, `morai_sim_pb2_grpc.py` 생성
- `simulator/` 폴더에 위치시키면 자동 import

### 3. config/collection_config.yaml
- `zones.*.spawn_ranges` : 실제 K-City 맵 좌표로 교체
- `zones.*.goal_ranges` : 실제 목표 구역 좌표로 교체
- `ros.topics.*` : 실제 ROS 토픽 이름 확인/수정

## Hz 구조

```
카메라 10Hz (ROS 토픽)
GNSS/IMU 50Hz (ROS 토픽)
        ↓
  타임스탬프 동기화 (±20ms 윈도우)
        ↓
  일반 주행 구간 → 2Hz 저장
  longtail 구간 → 10Hz 저장
  (급제동, 빨간불, 보행자 횡단 등)
```
