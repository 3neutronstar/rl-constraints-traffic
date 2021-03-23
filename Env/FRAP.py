import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy

'''
state는 movement를 
action은 phase를 하겠다
'''


class TL3x3Env(baseEnv):
    def __init__(self, configs):
        self.configs = configs
        self.tl_rl_list = self.configs['tl_rl_list']
        self.tl_list = traci.trafficlight.getIDList()
        self.edge_list = traci.edge.getIDList()
        self.pressure = 0
        self.reward = 0
        self.phase_list = self._phase_list()
        self.left_lane_num = self.configs['num_lanes']-1
        '''
        up right down left 순서대로 저장

        '''
        # grid_num 3일 때
        self.interest_list = [
            {
                'id': 'u_1_1',
                'inflow': 'n_1_0_to_n_1_1',
                'outflow': 'n_1_1_to_n_1_2',
            },
            {
                'id': 'r_1_1',
                'inflow': 'n_2_1_to_n_1_1',
                'outflow': 'n_1_1_to_n_0_1',
            },
            {
                'id': 'd_1_1',
                'inflow': 'n_1_2_to_n_1_1',
                'outflow': 'n_1_1_to_n_1_0',
            },
            {
                'id': 'l_1_1',
                'inflow': 'n_0_1_to_n_1_1',
                'outflow': 'n_1_1_to_n_2_1',
            }
        ]

        self.phase_size = len(
            traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))

    def get_state(self):
        phase = list()
        state = torch.zeros(
            (1, self.configs['state_space']), device=self.configs['device'], dtype=torch.float)  # 기준
        vehicle_state = torch.zeros(
            (int(self.configs['state_space']/2), 1), device=self.configs['device'], dtype=torch.float)  # -8은 phase크기
        # 변환
        for _, tl_rl in enumerate(self.tl_rl_list):
            phase.append(traci.trafficlight.getRedYellowGreenState(tl_rl))

        # 1교차로용 n교차로는 추가요망
        # phase state
        phase_state = self._toState(phase).view(
            1, -1).to(self.configs['device'])

        # vehicle state
        for i, interest in enumerate(self.interest_list):
            left_movement = traci.lane.getLastStepHaltingNumber(
                interest['inflow']+'_{}'.format(self.left_lane_num))
            # 직진
            vehicle_state[i*2] = traci.edge.getLastStepHaltingNumber(
                interest['inflow'])-left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
            # 좌회전
            vehicle_state[i*2] = left_movement

        vehicle_state = torch.transpose(vehicle_state, 0, 1)
        state = torch.cat((vehicle_state, phase_state),
                          dim=1)  # 여기 바꿨다 문제 생기면 여기임 암튼 그럼

        return state

    def collect_state(self):
        '''
        갱신 및 점수 확정용 함수
        '''
        inflow_rate = 0
        outflow_rate = 0
        for _, interest in enumerate(self.interest_list):
            inflow_rate += traci.edge.getLastStepHaltingNumber(
                interest['inflow'])
            outflow_rate += traci.edge.getLastStepHaltingNumber(
                interest['outflow'])
        self.pressure += (outflow_rate-inflow_rate)

    def step(self, action):
        '''
        agent 의 action 적용 및 reward 계산
        '''
        phase = self._toPhase(action)  # action을 분해

        # action을 environment에 등록 후 상황 살피기
        for _, tl_rl in enumerate(self.tl_rl_list):
            traci.trafficlight.setRedYellowGreenState(tl_rl, phase)

        # reward calculation and save

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        self.reward += self.pressure
        self.pressure = 0
        return self.reward

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0

    def _toPhase(self, action):  # action을 해석가능한 phase로 변환
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        return self.phase_list[action]

    def _toState(self, phase_set):  # env의 phase를 해석불가능한 state로 변환
        state_set = tuple()
        for i, phase in enumerate(phase_set):
            state = torch.zeros(8, dtype=torch.int)
            for i in range(4):  # 4차로
                phase = phase[1:]  # 우회전
                state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
                phase = phase[self.configs['num_lanes']-1:]  # 직전
                state[i+1] = self._mappingMovement(phase[0])  # 좌회전신호 추출
                phase = phase[1:]  # 좌회전
                phase = phase[1:]  # 유턴
            state_set += tuple(state.view(1, -1))
        state_set = torch.cat(state_set, 0)
        return state_set

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'G' or movement == 'g':
            return 1
        elif movement == 'r' or movement == 'R':
            return 0
        else:
            return -1  # error

    def _phase_list(self):
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        phase_list = [
            'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{3}rG{0}{1}gr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),  # current
            'r{2}{3}rG{0}{1}gr{2}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{3}rr{2}{3}rG{0}{1}g'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(
                g*num_lanes, g, r*num_lanes, r),  # current
        ]
        return phase_list
