import time
import sys
import os

# ------------------------------
# 테스트 실행 방법:
# python test_traffic_light.py
# ------------------------------

# run 폴더 기준으로 프로젝트 루트(morai_roach) 를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from network.UDP.receiver import TrafficLightReceiver

def on_traffic_light(data):
    print(f"[신호등]")
    print(f"  ID     : {data.index}")
    print(f"  타입   : {data.tl_type}")
    print(f"  상태   : {data.status}")
    print("-" * 30)
    print(repr(data.index))  # repr로 정확한 값 확인


# 모라이 네트워크 설정에서 TrafficLight 포트 확인 후 변경
receiver = TrafficLightReceiver(
    ip="127.0.0.1",
    port=7502,
    callback=on_traffic_light
)

print("수신 대기 중... (Ctrl+C 로 종료)")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    receiver.close()
    print("종료")