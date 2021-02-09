import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class Memory():
    def __init__(self, configs):
        self.reward = torch.zeros(1, dtype=torch.int, device=configs['device'])
        self.state = torch.zeros(
            (1, len(configs['tl_rl_list']), 8), dtype=torch.float, device=configs['device'])
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(
            (1, 2), dtype=torch.int, device=configs['device'])


class CityEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_list = traci.trafficlight.getIDList()
        self.tl_rl_list = self.configs['tl_rl_list']
        self.num_agent = len(self.tl_rl_list)
        self.side_list = ['u', 'r', 'd', 'l']
        self.interest_list = self.configs['interest_list']
        self.node_interest_pair = self.configs['node_interest_pair']

        self.reward = 0
        self.state_space = self.configs['state_space']
        self.action_size = self.configs['action_size']
        self.left_lane_num = self.configs['num_lanes']-1
        self.traffic_node_info = self.configs['traffic_node_info']
        self.vehicle_state_space = 8
        self.nodes = self.configs['node_info']

        self.before_action_update_mask = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.configs['device'])
        self.before_action_index_matrix = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.configs['device'])
        self.tl_rl_memory = list()
        for _ in range(self.num_agent):
            self.tl_rl_memory.append(Memory(self.configs))

        # action의 mapping을 위한 matrix
        self.min_phase = torch.tensor(
            self.configs['min_phase'], dtype=torch.int, device=self.configs['device'])
        self.max_phase = torch.tensor(
            self.configs['max_phase'], dtype=torch.int, device=self.configs['device'])
        self.common_phase = torch.tensor(
            self.configs['common_phase'], dtype=torch.int, device=self.configs['device'])
        self.matrix_actions = torch.tensor(
            self.configs['matrix_actions'], dtype=torch.int, device=self.configs['device'])
        # phase 갯수 list 생성
        self.num_phase_list = list()
        for phase in self.common_phase:
            self.num_phase_list.append(len(phase))

        # traci 내의 정보받아오기
        for key in self.traffic_node_info.keys():
            self.traffic_node_info[key]['program'] = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                key)

    def get_state(self, mask):
        '''
        매 주기마다 매 주기 이전의 state, 현재 state, reward를 반환하는 함수
        reward,next_state<-state 초기화 해줘야됨
        '''

        state = torch.zeros(
            (1, self.num_agent, self.num_agent, self.state_space), dtype=torch.float, device=self.configs['device'])
        next_state = torch.zeros_like(state)
        action = torch.zeros(
            (1, self.num_agent, self.num_agent, 2), dtype=torch.int, device=self.configs['device'])
        reward = torch.zeros((1, self.num_agent),
                             dtype=torch.int, device=self.configs['device'])
        for index in torch.nonzero(mask):
            state[0, index, :] = deepcopy(self.tl_rl_memory[index].state)
            action[0, index, :] = deepcopy(self.tl_rl_memory[index].action)
            next_state[0, index] = deepcopy(
                self.tl_rl_memory[index].next_state)
            reward[0, index] = deepcopy(self.tl_rl_memory[index].reward)
            # reward clear

        return state, action, reward, next_state

    def collect_state(self, action_update_mask, action_index_matrix):
        '''
        매초 마다 update할 것이 있는지 확인해야함
        전과 비교해서 인덱스가 늘어나고 그 인덱스가

        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''

        # next state 저장 여부 # before가 필요한 이유는 이전 step 행동에 대해서 state를 저장시켜줘야하기 때문
        need_state_mask = torch.bitwise_or(
            self.before_action_update_mask, action_update_mask)
        action_change_mask = torch.bitwise_or(
            self.before_action_update_mask, action_update_mask)
        next_state = torch.zeros(
            (1, self.num_agent, self.vehicle_state_space), dtype=torch.float, device=self.configs['device'])
        if need_state_mask.sum() != 0:  # 검색의 필요가 없다면 검색x
            next_state = tuple()
            # 모든 rl node에 대해서
            # vehicle state
            for interest in self.node_interest_pair:
                veh_state = torch.zeros(
                    (self.vehicle_state_space, 1), dtype=torch.float, device=self.configs['device'])
                # 모든 inflow에 대해서
                for j, pair in enumerate(self.node_interest_pair[interest]):
                    left_movement = traci.lane.getLastStepHaltingNumber(
                        pair['inflow']+'_{}'.format(self.left_lane_num))  # 멈춘애들 계산
                    # 직진
                    veh_state[j*2] = traci.edge.getLastStepHaltingNumber(
                        pair['inflow'])-left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
                    # 좌회전
                    veh_state[j*2+1] = left_movement
                veh_state = torch.transpose(veh_state, 0, 1)
                next_state += tuple(veh_state)
            next_state = torch.cat(next_state, dim=0).view(
                1, self.num_agent, self.vehicle_state_space)
            # 각 agent env에 state,next_state 저장
            for state_index in torch.nonzero(self.before_action_update_mask):
                self.tl_rl_memory[state_index].state = self.tl_rl_memory[state_index].next_state
                self.tl_rl_memory[state_index].next_state = next_state

        return next_state.view(1, -1)

    def step(self, action, mask, action_index_matrix, action_update_mask):
        '''
        매 초마다 action을 적용하고, next_state를 반환하는 역할
        yellow mask가 True이면 해당 agent reward저장
        '''
        # action update
        for index in torch.nonzero(mask):
            # action의 변환 -> 각 phase의 길이
            phase_length_set = self._toPhaseLength(
                self.tl_rl_list[index], action[index])
            # tls재설정
            tls = self.traffic_node_info[self.tl_rl_list[index]]['program']
            for phase_idx in self.traffic_node_info[self.tl_rl_list[index]]['phase_index']:
                tls.phase[phase_idx].duration = phase_length_set[phase_idx]

            traci.trafficlight.setProgramLogic(self.tl_rl_list[index], tls)
            self.tl_rl_memory[i].action = action.int()
        # action을 environment에 등록 후 상황 살피기,action을 저장

        # step
        traci.simulationStep()
        # next state 받아오기, reward저장
        next_state = self.collect_state(
            action_update_mask, action_index_matrix)
        self.before_action_index_matrix = action_index_matrix
        self.before_action_update_mask = action_update_mask
        return next_state

    def calc_action(self, action_matrix, actions, mask_matrix):
        for index in torch.nonzero(mask_matrix):
            actions = actions.long()
            action_matrix[index] = self.traffic_node_info[self.tl_rl_list[index]]['phase_duration']*actions[0, index, 1].int() + \
                self.common_phase[index]  # action으로 분배하는 공식 필요
        # 누적 합산
            for j in range(self.num_phase_list[index]):
                if j >= 1:
                    action_matrix[index, j] += action_matrix[index, j-1]
        return action_matrix.int()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0

    def _toPhaseLength(self, tl_rl, action):  # action을 해석가능한 phase로 변환
        tl_dict = deepcopy(self.traffic_node_info[tl_rl])
        for j, idx in enumerate(tl_dict['phase_index']):
            tl_dict['phase_duration'][idx] = tl_dict['phase_duration'][idx] + \
                action[j][0]*action[j][1]
        phase_set_length = tl_dict['phase_duration']
        return phase_length_set
