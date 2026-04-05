# morai-roach

**알파프로젝트1** — [Roach (ICCV 2021)](https://arxiv.org/abs/2108.08265)의 CARLA 기반 강화학습 교사·학생 파이프라인을 **MORAI SIM (Windows)** 으로 옮기는 팀 저장소입니다.

- 논문 공식 코드: [zhejz/carla-roach](https://github.com/zhejz/carla-roach)
- **범위**: **K-city** 단일 맵, Python **3.8** 고정
- **팀**: 인지 3명 / 판단·제어 3명 (역할은 아래 표 참고)

---

## Cursor에서 AI가 파일을 직접 수정하게 하려면

1. 채팅을 **Agent 모드**로 두고 질문합니다.
2. 터미널/저장소 쓰기가 막히면(샌드박스) 해당 명령에 **전체 권한(all)** 이 필요합니다. Cursor에서 터미널/도구 실행 시 뜨는 권한 승인을 허용하면 이동·설치·대량 편집이 가능합니다.

---

## 개발 환경 (Windows + Python 3.8)

> 팀 공식 가이드와 동일한 전제입니다.

### 필수 주의: 경로·계정

- **사용자명·프로젝트 경로에 한글·공백 금지** — MORAI, ROS 연동, Python에서 인코딩 오류가 납니다.
- 권장: `C:\workspace\morai-roach` 같은 **영문 경로**에 클론합니다. (현재 `26-1 FVE9001` 등 공백 경로는 이상 징후가 나면 이전 권장)

### 시스템 사전 작업 (MORAI 안정화)

1. **가상 메모리**: Windows 성능 옵션 → 고급 → 가상 메모리 **자동 관리** 켜기 후 재부팅.
2. **런타임**: [All in One Runtimes](https://www.computerbase.de/downloads/systemtools/all-in-one-runtimes/) 등으로 VC++ / .NET 등 누락 패키지 설치.
3. **MORAI**: [MoraiLauncher_Win.zip](https://develop-morai-s3-bucket.s3.ap-northeast-2.amazonaws.com/Asset/Launcher/Release/MoraiLauncher_Win.zip) 로 설치.

### Git, VS Code, Miniconda

- [Git for Windows](https://gitforwindows.org/), [VS Code](https://code.visualstudio.com/)
- [Miniconda Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html) — 설치 시 **Add Miniconda3 to PATH** 체크 권장.

### VS Code 터미널: PowerShell 대신 **CMD**

`conda activate` 가 PowerShell 정책에 막히는 경우가 많으므로, 기본 터미널을 **Command Prompt** 로 바꿉니다. (`Select Default Profile` → Command Prompt)

### Conda 환경 (Python 3.8)

```cmd
conda create -n morai python=3.8 -y
conda activate morai
cd C:\path\to\morai-roach
pip install -e .
pip install -r requirements.txt
```

VS Code: `Python: Select Interpreter` → `Python 3.8.x ('morai': conda)`.

**왜 3.8인가**: CARLA 0.9.10.1 / 논문 코드·ROS1 Noetic 계열과의 호환을 위해 버전을 고정합니다.

**pip / python**: `conda activate morai` 를 한 뒤에는 해당 가상환경의 `pip`·`python` 을 쓰면 됩니다. 시스템 PATH에 전역 `pip`가 없어도 문제 없습니다.

---

## 오프라인 Road H5 베이킹 (주행가능 영역)

`link_set.json` 만으로 Roach 호환 `road` 래스터를 한 번 생성합니다 (**ROS/UDP 없음**).

```cmd
conda activate morai
cd C:\path\to\morai-roach
python -m morai_gym.utils.birdview_map
```

상세: [docs/perception/OFFLINE_ROAD_H5.md](docs/perception/OFFLINE_ROAD_H5.md)

---

## 저장소 구조와 carla-roach 대응

| carla-roach | morai-roach |
|-------------|-------------|
| `carla_gym/utils/birdview_map.py` (맵 H5 생성) | `morai_gym/utils/birdview_map.py` (MGeo `link_set.json` → H5) |
| `carla_gym/core/obs_manager/birdview/chauffeurnet.py` | `morai_gym/core/obs_manager/birdview/bev_render.py` |
| `carla_gym/envs/`, `task_actor/` (보상·종료) | (예정) `morai_gym/envs/` 등 |
| `agents/rl_birdview`, `agents/cilrs` | (예정) `agents/` |
| `config/` Hydra | `config/birdview.yaml` + 향후 `train_rl` / `train_il` |

### 팀 역할 가이드

| 파트 | 담당 |
|------|------|
| **인지 (3)** | `morai_gym/core/obs_manager/birdview/`, `network/`, 맵 JSON·H5, UDP→텐서 |
| **판단·제어 (3)** | (예정) `gym.Env`, 보상/종료, PPO/CILRS, 제어 publish |

---

## BEV 관측 (Roach RL 정책과 채널 정합)

- **`masks`**: shape **`(15, 192, 192)`**, `uint8`, **carla `chauffeurnet` 과 동일 순서**  
  `[road, route, lane, vehicle×K, walker×K, tl×K]` — 기본 `K=4` (`history_idx` 길이).
- **`rendered`**: `(192, 192, 3)` RGB (디버그용).
- **road**: `config/birdview.yaml` 의 `static_map.h5_path` 가 가리키는 H5에 **`road`** 데이터셋과 `world_offset_in_meters`, `pixels_per_meter` attrs 가 있으면 CARLA와 같은 방식으로 워핑합니다. **파일이 없으면 road 채널은 0**.
- **route**: `BEVDynamicRenderer.update(..., route_world_xy=)` 에 `(N,2)` 세계 좌표를 넘기면 그립니다. 없으면 0.
- **lane**: `map/*.json` 의 차선(실선 255 / 점선 120).

---

## 네트워크

- MORAI IP/포트: [`network/UDP/ipconfig.json`](network/UDP/ipconfig.json)

---

## 빠른 실행 (BEV 테스트)

1. MORAI SIM 실행, **K-city** 시나리오 로드.  
2. 저장소 루트에서 (conda `morai` 활성화 후):

```cmd
python run\test_bev_dynamic.py
```

차선만 오프라인 확인:

```cmd
python run\test_bev_dynamic.py --test-lane
```

---

## K-city 맵 에셋

| 종류 | 경로 |
|------|------|
| JSON | `morai_gym/core/obs_manager/birdview/map/` (`lane_marking_set.json`, `link_set.json`, `traffic_light_set.json`, `stoplane_marking_set.json`) |
| H5 (선택) | `morai_gym/core/obs_manager/birdview/map/morai_kcity_map.h5` — `birdview.yaml` 의 `h5_path` 와 일치 |

[MORAI-MGeoModule](https://github.com/MORAI-Autonomous/MORAI-MGeoModule) 등은 **참고만** 하고, 필요한 코드만 이 저장소로 옮기고 **출처·라이선스**를 명시합니다.

---

## 로드맵 (한 학기)

- [x] UDP 브리지, BEV 렌더, **masks 15채널·CARLA 순서**
- [x] `morai_gym/core/obs_manager/birdview/` 로 carla `obs_manager/birdview` 에 대응
- [ ] `gym.Env` + 제어 루프 + 보상/종료
- [ ] `agents/rl_birdview` 이식 및 PPO 학습
- [ ] (선택) CILRS / DAGGER

---

## 트러블슈팅

- BEV가 틀어지면 `bev_render._get_warp_transform` 의 좌표·yaw 정의를 MORAI 문서와 대조합니다.
- UDP가 안 오면 방화벽·`ipconfig.json` 의 IP/포트를 확인합니다.
- **`pip` / `python` 인식 안 됨**: CMD에서 `conda activate morai` 후 다시 시도하거나, 전체 경로로 `conda run -n morai python ...` 사용.

---

## 라이선스

- carla-roach / Roach 저자 라이선스를 준수합니다. 외부 코드 복사 시 해당 레포 조건을 따릅니다.
