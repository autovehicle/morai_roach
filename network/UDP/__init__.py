from .protocol import EgoState, ObjectData, TrafficLightData
from .protocol import OBJ_TYPE_EGO, OBJ_TYPE_PEDESTRIAN, OBJ_TYPE_VEHICLE, OBJ_TYPE_OBSTACLE
from .protocol import TL_RED, TL_YELLOW, TL_GREEN, TL_GREEN_LEFT
from .receiver import EgoReceiver, ObjectReceiver, TrafficLightReceiver
from .sender import CtrlCmdSender, TrafficLightSender


## 전체 데이터 흐름 요약
"""
MORAI SIM → UDP 패킷 → receiver.py 파싱 → protocol.py 데이터 객체 → udp_manager.py 저장
                                                                         ↓
                                                              BEV 렌더러가 여기서 가져감
                                                              manager.vehicle_list
                                                              manager.pedestrian_list
                                                              manager.traffic_light
"""
"""
BEV renderer/RL agent → 제어 명령 → udp_manager.send_ctrl() → sender.py 패킹 → UDP 패킷 → MORAI SIM
"""