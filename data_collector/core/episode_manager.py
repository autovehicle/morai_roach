"""
core/episode_manager.py
------------------------
에피소드 1개의 전체 실행 루틴을 담당.

흐름:
    0. 에피소드 시작 → 저장 폴더 생성
    1. 시뮬레이터 맵/자차/NPC 세팅 (morai_client.setup_episode) & expert 경로 초기화
    2. 센서 데이터 수신 대기
    3. 매 스텝: 센서 스냅샷 수신 → expert 제어값 계산 → 시뮬에 제어 전송 → 특이상황(longtail)dlswl 판정 
    -> Hz 판정 후 저장 (일반 2Hz, longtail 10Hz) → 목표 도달/타임아웃 체크
    4. 종료 조건 도달 시 에피소드 종료 or timeout or sensor_failure → summary 반환
"""

import math
import time
from typing import Dict, Optional

from core.data_writer import DataWriter, SensorSnapshot
from core.scenario_params import EpisodeParams
from expert.expert_controller import ExpertController
from ros.ros_manager import ROSManager
from simulator.morai_client import MoraiClient


class EpisodeManager:
    """
    에피소드 실행기.

    사용 예시:
        manager = EpisodeManager(ros_mgr, morai_client, expert, writer, config)
        summary = manager.run(params)
    """

    def __init__(self,
                 ros_mgr:   ROSManager,
                 client:    MoraiClient,
                 expert:    ExpertController,
                 writer:    DataWriter,
                 config:    dict):

        self.ros    = ros_mgr
        self.client = client
        self.expert = expert
        self.writer = writer
        self.cfg    = config

        coll_cfg = config.get("collection", {})
        self.normal_hz    = coll_cfg.get("normal_hz",    2)
        self.longtail_hz  = coll_cfg.get("longtail_hz", 10)
        self.goal_thresh  = coll_cfg.get("goal_threshold_m", 5.0)
        self.sync_win_ms  = coll_cfg.get("sync_window_ms", 20)

    # ── 메인 실행 ─────────────────────────────────────────────────

    def run(self, params: EpisodeParams) -> dict:
        """에피소드 1개 실행. 종료 후 summary dict 반환."""

        # ── 1. 저장 폴더 생성 ─────────────────────────────────────
        ep_dir = self.writer.begin_episode(params)
        print(f"\n[Episode {params.episode_id:03d}] "
              f"zone={params.zone}  scenario={params.scenario}  "
              f"longtail={params.is_longtail}")
        print(f"  저장 경로: {ep_dir}")

        # ── 2. 시뮬레이터 세팅 (리셋/스폰/경로) ───────────────────
        map_name  = self.cfg["zones"][params.zone]["map_name"]
        waypoints, link_ids = self.client.setup_episode(params, map_name)

        if waypoints is None:
            print("[EpisodeManager] 시뮬 세팅 실패 → 에피소드 스킵")
            return self.writer.end_episode(False, "setup_failed")

        # ── 3. Expert 경로 초기화 ─────────────────────────────────
        self.expert.reset(waypoints)

        # ── 4. 일정 데이터 대기 ───────────────────────────────────
        print("[EpisodeManager] 일정 데이터 대기 중...")
        if not self.ros.wait_for_data(timeout_sec=5.0):
            print("[EpisodeManager] 일정 수신 타임아웃 → 에피소드 스킵")
            return self.writer.end_episode(False, "sensor_timeout")

        # ── 5. 메인 루프 ─────────────────────────────────────────
        max_steps        = params.max_steps
        step             = 0
        frame_id         = 0

        # Hz 제어용
        normal_period    = 1.0 / self.normal_hz     # 0.5s (2Hz)
        longtail_period  = 1.0 / self.longtail_hz   # 0.1s (10Hz)
        control_period   = longtail_period           # 제어 루프는 항상 10Hz
        last_save_time   = 0.0

        prev_speed       = 0.0
        termination      = "timeout"

        print(f"[EpisodeManager] 루프 시작 (max_steps={max_steps})")

        while step < max_steps:
            loop_start = time.time()
            step += 1

            # ── 5-1. 일정 스냅샷 수신 ────────────────────────────
            snap = self.ros.get_snapshot(frame_id, params.is_longtail)
            if snap is None or snap.ego is None:
                time.sleep(control_period)
                continue

            snap.nav_waypoints = waypoints
            snap.nav_link_ids  = link_ids

            # ── 5-2. Expert 제어 계산 ────────────────────────────
            output = self.expert.step(
                ego       = snap.ego,
                tl_states = snap.tl_states,
                dt        = control_period,
            )
            snap.expert_steer    = output.steer
            snap.expert_throttle = output.throttle
            snap.expert_brake    = output.brake

            # ── 5-3. 시뮬레이터에 제어 전송 ─────────────────────
            self.client.send_control(
                output.steer, output.throttle, output.brake
            )

            # ── 5-4. longtail 실시간 감지 ────────────────────────
            is_longtail_frame = ExpertController.detect_longtail(
                ego        = snap.ego,
                prev_speed = prev_speed,
                dt         = control_period,
                tl_states  = snap.tl_states,
                triggers   = params.longtail_triggers,
            )
            snap.is_longtail = is_longtail_frame or params.is_longtail
            prev_speed = snap.ego.speed

            # ── 5-5. Hz 제한 후 저장 ─────────────────────────────
            save_period = longtail_period if snap.is_longtail else normal_period
            now = time.time()
            if now - last_save_time >= save_period:
                frame_id      += 1
                snap.frame_id  = frame_id
                self.writer.write_frame(snap)
                last_save_time = now

            # ── 5-6. 종료 조건 체크 ──────────────────────────────
            if self._check_goal(snap.ego, params):
                termination = "goal_reached"
                print(f"[EpisodeManager] 목표 도달 (step={step}, frames={frame_id})")
                break

            # ── 5-7. 루프 주기 유지 ──────────────────────────────
            elapsed = time.time() - loop_start
            sleep_t = control_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # ── 6. 에피소드 종료 ──────────────────────────────────────
        summary = self.writer.end_episode(
            success    = (termination == "goal_reached"),
            reason     = termination,
        )
        summary["frames"] = frame_id

        print(f"[Episode {params.episode_id:03d}] 종료: {termination}  "
              f"frames={frame_id}  "
              f"elapsed={summary.get('duration_sec', 0):.1f}s")
        return summary

    # ── 헬퍼 ─────────────────────────────────────────────────────

    def _check_goal(self, ego, params: EpisodeParams) -> bool:
        dist = math.hypot(ego.x - params.goal_x,
                          ego.y - params.goal_y)
        return dist < self.goal_thresh
