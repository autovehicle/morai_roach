# protocl.py

해당 파일은 실제 통신하는 코드가 아님. MORAI에서 주고받는 MSG의 형태만 정의한 것.

1. # Header 상수

   HEADER_EGO = b'#MoraiInfo$'       # 11 bytes  — Ego Vehicle Status
HEADER_OBJ    = b'#MoraiObjInfo$' # 14 bytes — Object Info
   HEADER_TL = b'#TrafficLight$'    # 14 bytes  — Traffic Light (Get/Set 공용)
HEADER_CTRL   = b'#MoraiCtrlCmd$' # 14 bytes — Ego Ctrl Cmd
   ==================================================================================
   MORAI UDP 패킷은 항상 이 ASCII 문자열로 시작.
   Receiver가 패킷을 받으면 맨 앞 바이트를 확인해서 패킷을 구분

2. @dataclass
   class의 데이터를 담는 그릇. 일반 class에서는 **init** 함수도 만들고, 출력 함수도 만들고 값 비교도 해야하는데 @dataclass 사용하면 그냥 선언 해주면 끝

3. MSG를 통해 받아오는 변수 저장
   dataclass에 변수 타입, 명칭 선언 후 초기화. UDP 통신으로 받아오는 타겟 MSG들을 저장하는 변수들을 이 class에 저장

# receiver.py

해당 파일은 실제 통신이 receive 되는 핵심 코드이다. MORAI SIM -> USER 바이너리 패킷을 받아서 python 객체로 변환하는 코드

1. class BaseReceiver
   socket을 열고 thread에서 무한 loop를 돌며 패킷을 기다림.
   기본 상태는 thread가 무한 loop를 돌며 msg가 들어오길 기다리는 중 이라고 생각하면 됨
   패킷이 들어오면 \_parse() 로 데이터 파싱하고, 성공하면 callback 함수를 호출함.
   통신이 이루어지지 않더라도 thread는 블로킹 되지 않아 통신이 비정상적으로 끊기지 않는다.

# sender.py

receiver.py의 반대

# udp_manager.py
