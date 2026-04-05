"""
BEV / chauffeurnet 정합 디버그 로깅.

환경 변수:
  MORAI_BEV_DEBUG=1|true|yes|on  — 디버그 출력 활성화
  MORAI_BEV_DEBUG_EVERY=N        — N 프레임마다 한 번만 update() 요약 (기본 1)

에이전트/사용자가 상태를 공유할 때 콘솔에 붙여 넣기 좋게 태그를 고정한다.
"""
from __future__ import annotations

import os
from typing import Any


def morai_bev_debug_enabled() -> bool:
    v = os.environ.get('MORAI_BEV_DEBUG', '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


def morai_bev_debug_every_n() -> int:
    try:
        n = int(os.environ.get('MORAI_BEV_DEBUG_EVERY', '1'))
        return max(1, n)
    except ValueError:
        return 1


def morai_bev_dlog(tag: str, msg: str, *args: Any) -> None:
    """디버그가 켜져 있을 때만 print. msg는 % 포맷을 사용할 수 있다."""
    if not morai_bev_debug_enabled():
        return
    if args:
        print(f'[MORAI_BEV_DEBUG:{tag}] {msg % args}')
    else:
        print(f'[MORAI_BEV_DEBUG:{tag}] {msg}')
