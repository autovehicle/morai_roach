"""
collect.py
----------
데이터 수집 자동화 메인 실행 스크립트.

사용법:
    어떤 zone/scenario를 몇 개 수집할지 명령행 인자로 지정 가능.
    # 특정 zone + 특정 scenario 10개 수집
    python collect.py --zone urban --scenario stop_and_go --episodes 10

    # 특정 zone의 전체 scenario 수집
    python collect.py --zone urban --all-scenarios

    # 전체 zone + 전체 scenario 수집
    python collect.py --all

    # 중단 후 이어서 수집 (완료된 에피소드는 스킵)
    python collect.py --zone urban --scenario stop_and_go --resume

    # 현황 확인
    python collect.py --status
"""

"""
실행하면 다음과 같은 흐름으로 데이터 수집이 진행됨
ROSManager  → ROS 토픽 구독 시작
MoraiClient → 시뮬레이터 연결
ExpertController → expert 초기화
DataWriter  → 저장 경로 준비
EpisodeManager → 에피소드 실행기 준비
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ── Windows ROS1 환경변수 설정 ────────────────────────────────────────
def _setup_windows_ros(setup_bat: str):
    """
    Windows ROS1 Noetic setup.bat을 실행해 환경변수를 현재 프로세스에 적용.
    ROS_ROOT, PYTHONPATH, PATH 등이 설정됨.
    """
    if not sys.platform.startswith("win"):
        return  # Linux에서는 불필요

    if not os.path.exists(setup_bat):
        print(f"[WARNING] ROS setup.bat 없음: {setup_bat}")
        return

    try:
        # cmd /c "setup.bat && set" 으로 환경변수 덤프
        result = subprocess.run(
            f'cmd /c "{setup_bat} && set"',
            capture_output=True, text=True, shell=True
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip()
        print(f"[ROS] 환경변수 설정 완료: {setup_bat}")
    except Exception as e:
        print(f"[WARNING] ROS 환경변수 설정 실패: {e}")


# ── 로거 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("collect")


# ── 설정 파일 로드 ────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 수집 실행 함수 ────────────────────────────────────────────────────
def collect_scenario(zone: str, scenario: str, n_episodes: int,
                     manager, generator, writer, resume: bool):
    """단일 zone/scenario에 대해 n_episodes개 수집"""
    logger.info(f"  ▶ zone={zone}  scenario={scenario}  n={n_episodes}")

    for ep_id in range(1, n_episodes + 1):
        # --resume: 이미 완료된 에피소드 스킵
        if resume and writer.episode_exists(zone, scenario, ep_id):
            logger.info(f"    episode_{ep_id:03d} 이미 완료 → 스킵")
            continue

        params  = generator.generate(zone, scenario, episode_id=ep_id)
        summary = manager.run(params)

        status = "✓" if summary.get("success") else "✗"
        logger.info(
            f"    [{status}] ep={ep_id:03d}  "
            f"frames={summary.get('frames', 0)}  "
            f"reason={summary.get('reason', '?')}"
        )


# ── 데이터셋 현황 출력 ────────────────────────────────────────────────
def print_status(config: dict):
    root = Path(config["dataset"]["root_dir"])
    if not root.exists():
        print(f"데이터셋 경로 없음: {root}")
        return

    print(f"\n{'='*60}")
    print(f"{'Zone/Scenario':<40} {'에피소드':>8} {'프레임':>8}")
    print(f"{'-'*60}")
    total_ep = total_fr = 0

    for zone_dir in sorted(root.iterdir()):
        if not zone_dir.is_dir():
            continue
        for scen_dir in sorted(zone_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            ep_dirs = [d for d in scen_dir.iterdir() if d.is_dir()]
            frames  = sum(
                len(list((d / "frames").glob("*.npz")))
                for d in ep_dirs if (d / "frames").exists()
            )
            key = f"{zone_dir.name}/{scen_dir.name}"
            print(f"{key:<40} {len(ep_dirs):>8} {frames:>8}")
            total_ep += len(ep_dirs)
            total_fr += frames

    print(f"{'─'*60}")
    print(f"{'합계':<40} {total_ep:>8} {total_fr:>8}")
    print(f"{'='*60}\n")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MORAI 데이터 수집 자동화",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--zone",          type=str, help="Zone 이름 (예: urban)")
    parser.add_argument("--scenario",      type=str, help="Scenario 이름 (예: stop_and_go)")
    parser.add_argument("--episodes",      type=int, default=None, help="수집할 에피소드 수")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="지정 zone의 전체 scenario 수집")
    parser.add_argument("--all",           action="store_true",
                        help="전체 zone + 전체 scenario 수집")
    parser.add_argument("--resume",        action="store_true",
                        help="완료된 에피소드 스킵하고 이어서 수집")
    parser.add_argument("--status",        action="store_true",
                        help="현재 수집 현황만 출력")
    parser.add_argument("--config",        type=str,
                        default="config/collection_config.yaml",
                        help="설정 파일 경로")
    args = parser.parse_args()

    # ── 설정 로드 ────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── --status 단독 실행 ───────────────────────────────────────────
    if args.status:
        print_status(config)
        return

    # ── Windows ROS 환경변수 설정 ────────────────────────────────────
    setup_bat = config.get("ros", {}).get("setup_bat", "")
    if setup_bat:
        _setup_windows_ros(setup_bat)

    # ── 모듈 import (ROS 환경변수 설정 후 해야 함) ───────────────────
    from core.data_writer      import DataWriter
    from core.episode_manager  import EpisodeManager
    from core.scenario_params  import ScenarioParamGenerator
    from expert.expert_controller import ExpertController
    from ros.ros_manager       import ROSManager
    from simulator.morai_client import MoraiClient

    # ── 컴포넌트 초기화 ──────────────────────────────────────────────
    logger.info("컴포넌트 초기화 중...")

    ros_mgr   = ROSManager(config)
    ros_mgr.start()

    client    = MoraiClient(config)
    if not client.connect():
        logger.error("MORAI gRPC 연결 실패 → 종료")
        ros_mgr.stop()
        sys.exit(1)

    expert    = ExpertController(config)
    writer    = DataWriter(config)
    generator = ScenarioParamGenerator(config)
    manager   = EpisodeManager(ros_mgr, client, expert, writer, config)

    n_episodes = args.episodes or config.get("collection", {}).get("default_episodes", 10)

    # ── 수집 실행 ────────────────────────────────────────────────────
    try:
        if args.all:
            logger.info("전체 Zone + 전체 Scenario 수집 모드")
            for zone, zone_cfg in config["zones"].items():
                for scenario in zone_cfg["scenarios"]:
                    collect_scenario(zone, scenario, n_episodes,
                                     manager, generator, writer, args.resume)

        elif args.zone and args.all_scenarios:
            zone = args.zone
            if zone not in config["zones"]:
                logger.error(f"정의되지 않은 zone: {zone}")
                sys.exit(1)
            logger.info(f"zone={zone} 전체 Scenario 수집 모드")
            for scenario in config["zones"][zone]["scenarios"]:
                collect_scenario(zone, scenario, n_episodes,
                                 manager, generator, writer, args.resume)

        elif args.zone and args.scenario:
            collect_scenario(args.zone, args.scenario, n_episodes,
                             manager, generator, writer, args.resume)

        else:
            parser.print_help()
            logger.error("\n--zone + --scenario 또는 --all-scenarios 또는 --all 옵션이 필요합니다.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n수집 중단됨 (Ctrl+C)")

    finally:
        client.disconnect()
        ros_mgr.stop()
        logger.info("\n최종 수집 현황:")
        writer.print_summary()


if __name__ == "__main__":
    main()
