"""Policies: abstract base class and concrete implementations."""

import torch as th
import torch.nn as nn
import numpy as np

from . import torch_util as tu # 사용 안하면 삭제하기


class BirdviewExtractor(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        # 채널에 맞게 조정 필요함(observation_space['birdview']의 shape에 맞게 n_input_channels 조정)
        n_input_channels = observation_space['birdview'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 3. CNN 출력 차원 자동 계산
        with th.no_grad():
            sample_input = th.as_tensor(observation_space['birdview'].sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        # 4. 수치 데이터 처리용 MLP (상태 정보 추출)
        state_input_dim = observation_space['state'].shape[0]
        state_layers = []
        in_dim = state_input_dim
        for out_dim in states_neurons:
            state_layers.append(nn.Linear(in_dim, out_dim))
            state_layers.append(nn.ReLU())
            in_dim = out_dim
        self.state_linear = nn.Sequential(*state_layers)

        # 5. CNN 특징 + 상태 특징을 합쳐서 최종 특징(features_dim)으로 투영
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + states_neurons[-1], 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

        # 가중치 초기화 적용
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        # 이미지 데이터 통과
        cnn_features = self.cnn(birdview)
        # 수치 데이터 통과
        state_features = self.state_linear(state)

        # 두 특징을 옆으로 합치기 (Concatenate)
        combined = th.cat((cnn_features, state_features), dim=1)
        
        # 최종 특징 벡터 반환
        return self.linear(combined)
