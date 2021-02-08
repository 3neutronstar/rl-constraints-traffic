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


class GridEnv(baseEnv):
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
        self.action_space = self.configs['action_space']
        self.action_size = self.configs['action_size']
        self.left_lane_num = self.configs['num_lanes']-1
        self.phase_dict = self.configs['phase_dict']
        self.vehicle_state_space = 8
        self.nodes = self.configs['node_info']

        self.before_action_change_mask = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.configs['device'])
        self.before_index_mask = torch.zeros(
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

    def collect_state(self, action_change_mask, yellow_mask):
        '''
        매 초마다 reward update할 것이 있는지 확인하는 함수
        action change mask가 True라면 update를 해줘야함
        action_change_mask가 all False면 동작 x, all zero
        yellow_mask가 False이면 reward 검색 x

        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        # outflow확인,reward 저장
        # 값이 0인 경우 == all False
        for index in torch.nonzero(yellow_mask):
            outflow = 0
            inflow = 0
            interests = self.node_interest_pair[self.tl_rl_list[index]]
            for interest in interests:
                outflow += traci.edge.getLastStepVehicleNumber(
                    interest['outflow'])
                inflow += traci.edge.getLastStepHaltingNumber(
                    interest['inflow'])
            # pressure=inflow-outflow
            # reward cumulative sum
            pressure = torch.tensor(
                -(inflow-outflow), dtype=torch.int, device=self.configs['device'])
            self.tl_rl_memory[index].reward += pressure
            self.reward += pressure

        # next state 저장
        need_state_mask = torch.bitwise_and(
            self.before_action_change_mask, action_change_mask)
        # print(need_state_mask)
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
            for state_index in torch.nonzero(self.before_action_change_mask):
                self.tl_rl_memory[state_index].state = self.tl_rl_memory[state_index].next_state
                self.tl_rl_memory[state_index].next_state = next_state

        return next_state.view(1, -1)

    def step(self, action, index_mask, yellow_mask):
        '''
        매 초마다 action을 적용하고, next_state를 반환하는 역할
        yellow mask가 True이면 해당 agent reward저장
        '''
        # action 적용
        action_change_mask = ~torch.eq(
            self.before_index_mask, index_mask)  # True가 하나라도 존재시 실행
        # action을 environment에 등록 후 상황 살피기,action을 저장
        for i in torch.nonzero(action_change_mask):  # 노란 신호 초기화는 어떻게 할까요
            phase = self._toPhase(
                self.tl_rl_list[i], index_mask[i])  # action을 분해
            traci.trafficlight.setRedYellowGreenState(
                self.tl_rl_list[i], phase)
            self.tl_rl_memory[i].action = action.int()

        # step
        traci.simulationStep()
        # next state 받아오기, reward저장
        next_state = self.collect_state(action_change_mask, yellow_mask)

        # index mask,action change mask -> update
        self.before_index_mask = deepcopy(index_mask)
        self.before_action_change_mask = deepcopy(action_change_mask)
        return next_state

    def calc_action(self, action_matrix, actions, mask_matrix):
        for index in torch.nonzero(mask_matrix):
            actions = actions.long()
            action_matrix[index] = self.matrix_actions[actions[0, index, 0]]*actions[0, index, 1].int() + \
                self.common_phase[index]  # action으로 분배하는 공식 필요
        # 누적 합산
            for j in range(self.num_phase_list[index]):
                if j >= 1:
                    action_matrix[index, j] += action_matrix[index, j-1]
                action_matrix[index, j] += 3
        return action_matrix.int()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0

    def _toPhase(self, tl_rl, action):  # action을 해석가능한 phase로 변환
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        return self.phase_dict[tl_rl][action]
