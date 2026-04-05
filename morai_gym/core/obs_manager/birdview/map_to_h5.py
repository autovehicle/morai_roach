import os
import sys
import json
import yaml
import numpy as np
import cv2
import h5py
import struct
import socket
import time
from collections import deque
from pathlib import Path

# ======================
# config
# ======================

class Config:
    """birdview.yaml + ipconfig.json을 로드하여 전역 설정으로 사용"""
 
    def __init__(self, workspace_root):
        self.root = Path(workspace_root)
 
        # --------- birdview.yaml ----------
        bev_path = self.root / 'config' / 'birdview.yaml'
        with open(bev_path, 'r', encoding='utf-8') as f:
            bev_cfg = yaml.safe_load(f)
 
        bv = bev_cfg['birdview']
        self.width = int(bv['width_in_pixels'])         # BEV Image 크기 192 x 192
        self.ev_to_bottom = int(bv['pixels_ev_to_bottom'])  # BEV Image에서 Ego Vehicle 위치가 아래에서 얼마나 떨어져 있는지 (40px)
        self.ppm = float(bv['pixels_per_meter'])         # 5.0
        self.history_idx = list(bv['history_idx'])        # [-16,-11,-6,-1] 과거 4프레임의 BEV를 사용 (각각 0.5s, 0.75s, 0.9s, 0.99s 시점)
        self.scale_bbox = bool(bv.get('scale_bbox', False))
        self.scale_mask_col = float(bv.get('scale_mask_col', 1.1))
 
        cv = bev_cfg['channel_values']
        self.lane_solid = int(cv['lane_solid'])
        self.lane_broken = int(cv['lane_broken'])
        self.tl_green = int(cv['tl_green'])
        self.tl_yellow = int(cv['tl_yellow'])
        self.tl_red = int(cv['tl_red'])
        self.stop_sign = int(cv['stop_sign'])
 
        db = bev_cfg['default_bbox']
        self.veh_size = (float(db['vehicle_length']), float(db['vehicle_width']))
        self.ped_size = (float(db['pedestrian_length']), float(db['pedestrian_width']))
        self.walker_scale = float(db['walker_bbox_scale'])
 
        det = bev_cfg['detection']
        self.veh_dist = float(det['vehicle_distance'])
        self.ped_dist = float(det['pedestrian_distance'])
        self.tl_dist = float(det['traffic_light_distance'])
 
        rt = bev_cfg['route']
        self.max_wps = int(rt['max_waypoints'])
        self.route_thick = int(rt['line_thickness'])
 
        sm = bev_cfg['static_map']
        self.static_h5 = str(self.root / sm['h5_path'])
 
        col = bev_cfg['collection']
        self.output_dir = str(self.root / col['output_dir'])
        self.max_steps = int(col['max_steps_per_episode'])
        self.sim_hz = int(col['sim_hz'])
 
        # --------- ipconfig.json ----------
        net_path = self.root / 'network' / 'ipconfig.json'
        if not net_path.exists():
            net_path = self.root / 'network' / 'UDP' / 'ipconfig.json'

        if not net_path.exists():
            raise FileNotFoundError(
                f"ipconfig.json not found in '{self.root / 'network'}' or '{self.root / 'network' / 'UDP'}'"
            )

        with open(net_path, 'r', encoding='utf-8') as f:
            net_cfg = json.load(f)['network']

        self.user_ip = net_cfg['user_ip']
        self.host_ip = net_cfg['host_ip']
 
        # 수신 포트 (bind) — dst_port가 우리 쪽
        self.ego_rx_port = int(net_cfg['ego_info_dst_port'])          # 909
        self.obj_rx_port = int(net_cfg['object_info_dst_port'])       # 7505
        self.tl_rx_port = int(net_cfg['get_traffic_dst_port'])        # 7502
 
        # 송신 포트 — host_port가 시뮬레이터 쪽
        self.ctrl_tx_port = int(net_cfg['ctrl_cmd_host_port'])        # 9095
 

class MoraiReceiver:
    """UdpManager를 감싸는 래퍼. Config 객체로 초기화."""

    def __init__(self, config):
        from network.UDP.udp_manager import UdpManager
        net_path = str(config.root / 'network' / 'UDP' / 'ipconfig.json')
        self.manager = UdpManager(config_path=net_path)

    def start(self):
        self.manager.start()

    def stop(self):
        self.manager.stop()

    @property
    def ego_state(self):
        return self.manager.ego_state

    @property
    def vehicle_list(self):
        return self.manager.vehicle_list

    @property
    def pedestrian_list(self):
        return self.manager.pedestrian_list

    @property
    def traffic_light(self):
        return self.manager.traffic_light

    @property
    def is_ready(self):
        return self.manager.is_ready


class BEVRender:
    """BEV 렌더러. 동적 객체 마스킹은 BEVDynamicRenderer에 위임."""

    def __init__(self, config):
        from morai_gym.core.obs_manager.birdview.bev_render import BEVDynamicRenderer
        self.config = config
        self.dynamic_renderer = BEVDynamicRenderer.from_config(config)