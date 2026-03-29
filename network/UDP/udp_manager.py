"""
MORAI UDP 통신 매니저.

MORAI SIM:Drive 시뮬레이터와 UDP 통신으로:
  - Ego 차량 상태 수신       (EgoState)
  - 주변 객체 정보 수신       (ObjectData × N)
  - 신호등 상태 수신          (TrafficLightData)
  - 차량 제어 명령 송신       (accel, brake, steer)
  - 신호등 상태 강제 변경     (선택적)

사용법:
    from network.udp_manager import UdpManager

    manager = UdpManager()
    manager.start()

    # 메인 루프에서:
    ego   = manager.ego_state       # EgoState | None
    objs  = manager.object_list     # list[ObjectData]
    tl    = manager.traffic_light   # TrafficLightData | None

    manager.send_ctrl(accel=0.5, brake=0.0, steer=0.1)
"""
import json
from pathlib import Path

from .protocol import (
    EgoState, ObjectData, TrafficLightData,
    OBJ_TYPE_VEHICLE, OBJ_TYPE_PEDESTRIAN,
    TL_GREEN,
)
from .receiver import EgoReceiver, ObjectReceiver, TrafficLightReceiver
from .sender import CtrlCmdSender, TrafficLightSender


class UdpManager:
    """MORAI ↔ User UDP 통신을 관리하는 중앙 클래스."""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: ipconfig.json 경로.
                         None이면 network/ipconfig.json 자동 탐색.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / 'ipconfig.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            net = json.load(f)['network']

        # UDP 통신 하기 위한 user_ip, host_ip, 포트 번호 (ipconfig.json에서 읽음)
        self._user_ip = net['user_ip'] # ._ (언더스코어 네이밍): C++ 에서 private과 같은 접근 제어자 키워드 python에는 X, 하지만 관례적으로 _ 붙이면 "이건 내부용이니 외부에서 직접 건드리지 마세요" 라는 의미. 실제로는 외부에서 manager._user_ip 이렇게 접근 가능하지만, 관례적으로는 manager.user_ip 이런 식으로 공개 프로퍼티를 통해 접근하도록 유도. 
        self._host_ip = net['host_ip']

        # 수신 포트 (bind — 우리 쪽)
        self._ego_port = int(net['ego_info_dst_port'])        # 909
        self._obj_port = int(net['object_info_dst_port'])     # 7505
        self._tl_port  = int(net['get_traffic_dst_port'])     # 7502

        # 송신 포트 (시뮬레이터 쪽)
        self._ctrl_port   = int(net['ctrl_cmd_host_port'])     # 9095
        self._tl_set_port = int(net['set_traffic_host_port'])  # 7607

        # ── 상태 저장소 ────────────────────────────────────────────
        self._ego_state:     EgoState           = None
        self._object_list:   list[ObjectData]   = []
        self._traffic_light: TrafficLightData   = None

        # ── 수신기 / 송신기 (start() 에서 생성) ────────────────────
        self._ego_rx:  EgoReceiver          = None
        self._obj_rx:  ObjectReceiver       = None
        self._tl_rx:   TrafficLightReceiver = None
        self._ctrl_tx: CtrlCmdSender        = None
        self._tl_tx:   TrafficLightSender   = None

        # ── 옵션 ──────────────────────────────────────────────────
        self.traffic_light_control = False  # True면 신호등 강제 녹색

    # ═══════════════════════════════════════════════════════════════
    # 공개 프로퍼티
    # ═══════════════════════════════════════════════════════════════

    @property # 읽기 전용 속성: self._ego_state 를 외부에서 수정하지 못하게 막음. 하지만 외부에서는 manager.ego_state 로 쉽게 읽을 수 있음. (상태 변조 막음)
    def ego_state(self) -> EgoState:
        """최신 ego 차량 상태. 아직 수신 전이면 None."""
        return self._ego_state

    @property
    def object_list(self) -> list:
        """최신 주변 객체 리스트. 빈 리스트일 수 있음."""
        return self._object_list

    @property
    def vehicle_list(self) -> list:
        """object_list 중 차량(obj_type=1)만 필터링."""
        return [o for o in self._object_list if o.obj_type == OBJ_TYPE_VEHICLE] # python의 간결한 반복문: self._object_list를 돌면서 특정 조건(o.obj_type == OBJ_TYPE_VEHICLE)을 만족하는 요소만 골라서 새 리스트로 반환. (OBJ_TYPE_VEHICLE에 대한 정의는 protocol.py 참고)
        
    @property
    def pedestrian_list(self) -> list:
        """object_list 중 보행자(obj_type=0)만 필터링."""
        return [o for o in self._object_list if o.obj_type == OBJ_TYPE_PEDESTRIAN]

    @property
    def traffic_light(self) -> TrafficLightData:
        """최신 신호등 상태. 아직 수신 전이면 None."""
        return self._traffic_light

    @property
    def is_ready(self) -> bool:
        """ego 데이터가 한 번 이상 수신됐으면 True."""
        return self._ego_state is not None

    # ═══════════════════════════════════════════════════════════════
    # 시작 / 종료
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """수신기·송신기 소켓을 열고 데몬 스레드를 시작한다."""
        # ── Receivers ──
        self._ego_rx = EgoReceiver(
            self._host_ip, self._ego_port, self._on_ego
        )
        self._obj_rx = ObjectReceiver(
            self._host_ip, self._obj_port, self._on_objects
        )
        self._tl_rx = TrafficLightReceiver(
            self._host_ip, self._tl_port, self._on_traffic_light
        )

        # ── Senders  ──
        self._ctrl_tx = CtrlCmdSender(
            self._user_ip, self._ctrl_port
        )
        self._tl_tx = TrafficLightSender(
            self._user_ip, self._tl_set_port
        )

        print('[UdpManager] started')
        print(f'  ego  recv  @ {self._host_ip}:{self._ego_port}')
        print(f'  obj  recv  @ {self._host_ip}:{self._obj_port}')
        print(f'  tl   recv  @ {self._host_ip}:{self._tl_port}')
        print(f'  ctrl send  → {self._user_ip}:{self._ctrl_port}')
        print(f'  tl   send  → {self._user_ip}:{self._tl_set_port}')

    def stop(self):
        """모든 소켓을 닫는다."""
        for rx in (self._ego_rx, self._obj_rx, self._tl_rx):
            if rx is not None:
                rx.close()
        for tx in (self._ctrl_tx, self._tl_tx):
            if tx is not None:
                tx.close()
        print('[UdpManager] stopped')

    # ═══════════════════════════════════════════════════════════════
    # 제어 명령 전송
    # ═══════════════════════════════════════════════════════════════

    def send_ctrl(self, accel: float, brake: float, steer: float):
        """
        차량 제어 명령 전송.

        Args:
            accel: 가속 (0 ~ 1)
            brake: 제동 (0 ~ 1)
            steer: 조향 (-1 ~ 1, 정규화)
        """
        if self._ctrl_tx is not None:
            self._ctrl_tx.send_data([accel, brake, steer])

    def force_green(self, tl_index: str):
        """특정 신호등을 녹색(16)으로 강제 변경."""
        if self._tl_tx is not None:
            self._tl_tx.send_data([tl_index, TL_GREEN])

    # ═══════════════════════════════════════════════════════════════
    # 수신 콜백 (데몬 스레드에서 호출됨)
    # ═══════════════════════════════════════════════════════════════

    def _on_ego(self, ego: EgoState):
        self._ego_state = ego

    def _on_objects(self, objects: list):
        self._object_list = objects if objects else []

    def _on_traffic_light(self, tl: TrafficLightData):
        self._traffic_light = tl

        # 신호등 강제 녹색 모드
        if self.traffic_light_control and tl is not None:
            self.force_green(tl.index)

    # ═══════════════════════════════════════════════════════════════
    # 디버그 출력
    # ═══════════════════════════════════════════════════════════════

    def print_status(self):
        """현재 수신 상태를 콘솔에 출력한다."""
        ego = self._ego_state
        if ego is None:
            print('[UdpManager] ego 데이터 대기 중...')
            return

        print('──────────── Ego ────────────')
        print(f'  pos   : ({ego.pos_x:.2f}, {ego.pos_y:.2f}, {ego.pos_z:.2f})')
        print(f'  yaw   : {ego.yaw:.2f} deg')
        print(f'  vel   : {ego.vel_x:.2f} km/h')
        print(f'  steer : {ego.front_steer:.2f} deg')
        print(f'  size  : ({ego.size_x:.2f}, {ego.size_y:.2f}, {ego.size_z:.2f})')

        objs = self._object_list
        if objs:
            print(f'──────────── Objects ({len(objs)}) ────────────')
            for i, o in enumerate(objs):
                type_name = {-1: 'Ego', 0: 'Ped', 1: 'Veh', 2: 'Obs'}.get(o.obj_type, '?')
                print(
                    f'  #{i} [{type_name}] '
                    f'pos=({o.pos_x:.1f}, {o.pos_y:.1f}) '
                    f'hdg={o.heading:.1f}° '
                    f'size=({o.size_x:.1f}×{o.size_y:.1f}) '
                    f'vel=({o.vel_x:.1f}, {o.vel_y:.1f}) km/h'
                )

        tl = self._traffic_light
        if tl is not None:
            print(f'──────────── Traffic Light ────────────')
            print(f'  index  : {tl.index}')
            print(f'  type   : {tl.tl_type}')
            print(f'  status : {tl.status}')
