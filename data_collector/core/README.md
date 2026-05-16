/data_collector/core code 3개 관계

[ 실행 순서 ]
scenario_params.py
    → episode_manager.py
        → data_writer.py

[ 역할 ]
1. scenario_params.py — 에피소드 설정값 생성
이번 에피소드에서 자차를 어디에 소환할지, NPC는 몇 대인지, longtail인지 등을 랜덤으로 뽑아서 EpisodeParams 객체 하나를 만들어요.

2. episode_manager.py — 에피소드 실행
EpisodeParams를 받아서 실제로 에피소드를 돌려요. 시뮬 세팅하고, 매 스텝마다 센서 받고, expert 제어 계산하고, 저장 타이밍 판단해요.

3. data_writer.py — 데이터 저장
episode_manager.py가 매 스텝마다 SensorSnapshot을 넘겨주면, BEV 맵 생성하고 GT 분리하고 .npz로 저장해요.