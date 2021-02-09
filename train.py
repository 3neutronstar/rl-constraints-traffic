import json
import os
import sys
import time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import update_tensorboard
from Agent.base import merge_dict


def city_dqn_train(configs, time_data, sumoCmd):
    '''
    여기서 configs 값 조정 금지
    '''
    from Agent.super_dqn import Trainer
    if configs['model'] == 'city':
        from Env.CityEnv import CityEnv

    phase_num_matrix = torch.tensor(  # 각 tl이 갖는 최대 phase갯수
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    NUM_AGENT = configs['num_agent']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],  # i*10
                          device=configs['device'], dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=configs['device'], dtype=torch.int)
    epoch = 0
    while epoch < configs['num_epochs']:
        step = 0
        traci.start(sumoCmd)
        env = CityEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=configs['device'])
        # Mask Matrix : TL_Period가 끝나면 True
        mask_matrix = torch.ones(
            (NUM_AGENT), dtype=torch.bool, device=configs['device'])

        # MAX Period까지만 증가하는 t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=configs['device'])
        t_agent -= OFFSET

        # Action configs['offset']on Matrix : 비교해서 동일할 때 collect_state, 없는 state는 zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=configs['device'])  # 노란불 3초 해줘야됨
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=configs['device'])  # 현재 몇번째 phase인지
        action_update_mask = torch.eq(   # action이 지금 update해야되는지 확인
            t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 0,인 이유는 인덱싱

        # state initialization
        state = env.collect_state(
            action_update_mask, action_index_matrix, mask_matrix)
        total_reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()
        while step < MAX_STEPS:

            # action 을 정하고
            actions = agent.get_action(state, mask_matrix)
            # action형태로 변환 # 다음으로 넘어가야할 시점에 대한 matrix
            action_matrix = env.calc_action(
                action_matrix, actions, mask_matrix)
            # 누적값으로 나타남

            # 전체 1초증가 # traci는 env.step에
            step += 1
            t_agent += 1

            # environment에 적용
            # action 적용함수, traci.simulationStep 있음
            next_state = env.step(
                actions, mask_matrix, action_index_matrix, action_update_mask)

            # env속에 agent별 state를 꺼내옴, max_offset+period 이상일 때 시작
            if step >= int(torch.max(OFFSET)+torch.max(TL_PERIOD)):
                rep_state, rep_action, rep_reward, rep_next_state = env.get_state(
                    mask_matrix)
                agent.save_replay(rep_state, rep_action, rep_reward,
                                  rep_next_state, mask_matrix)  # dqn
                total_reward += rep_reward.sum()
            # update
            agent.update(mask_matrix)

            # 모두 하고 나서

            # 넘어가야된다면 action index증가 (by tensor slicing)
            action_update_mask = torch.eq(  # update는 단순히 진짜 현시만 받아서 결정해야됨
                t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 0,인 이유는 인덱싱
            # print(t_agent, "time")

            # 최대에 도달하면 0으로 초기화 (offset과 비교)
            update_matrix = torch.eq(t_agent % TL_PERIOD, 0)
            t_agent[update_matrix] = 0
            # print(update_matrix, "update")

            action_index_matrix[action_update_mask] += 1
            # agent의 최대 phase를 넘어가면 해당 agent의 action index 0으로 초기화
            clear_matrix = torch.ge(action_index_matrix, phase_num_matrix)
            action_index_matrix[clear_matrix] = 0
            # mask update, matrix True로 전환
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False

            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % agent.configs['target_update_period'] == 0:
            agent.target_update()  # dqn
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward.sum(), arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def super_dqn_train(configs, time_data, sumoCmd):
    '''
    mask
    If some agents' time step are over their period, then mask True.
    Other's matrix element continue False.
    '''
    from Agent.super_dqn import Trainer
    if configs['model'] == 'base':
        from Env.MultiEnv import GridEnv

    phase_num_matrix = torch.tensor(
        [len(phase) for i, phase in enumerate(configs['max_phase'])])
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    NUM_AGENT = configs['num_agent']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']

    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],  # i*10
                          device=configs['device'], dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=configs['device'], dtype=torch.int)
    epoch = 0
    while epoch < configs['num_epochs']:
        step = 0
        traci.start(sumoCmd)
        env = GridEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=configs['device'])
        # Mask Matrix
        mask_matrix = torch.ones(
            (NUM_AGENT), dtype=torch.bool, device=configs['device'])

        # MAX Period까지만 증가하는 t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=configs['device'])
        t_agent -= OFFSET

        # Acticonfigs['offset']on Matrix : 비교해서 동일할 때 collect_state, 없는 state는 zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=configs['device'])  # 노란불 3초 해줘야됨
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=configs['device'])  # 현재 몇번째 phase인지
        yellow_mask = torch.zeros(
            (NUM_AGENT), dtype=torch.bool, device=configs['device'])  # 현재 몇번째 phase인지

        # state initialization
        state = env.collect_state(mask_matrix, yellow_mask)
        total_reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()
        while step < MAX_STEPS:

            # action 을 정하고
            actions = agent.get_action(state, mask_matrix)
            action_matrix = env.calc_action(action_matrix,
                                            actions, mask_matrix)  # action형태로 변환 # 다음으로 넘어가야할 시점에 대한 matrix
            # 누적값으로 나타남

            # 전체 1초증가 # traci는 env.step에
            step += 1
            t_agent += 1
            # 넘어가야된다면 action index증가 (by tensor slicing+yellow signal)
            action_update_mask = torch.eq(
                t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 0,인 이유는 인덱싱

            # 최대에 도달하면 0으로 초기화 (offset과 비교)
            update_matrix = torch.eq(t_agent % TL_PERIOD, 0)
            t_agent[update_matrix] = 0

            action_index_matrix[action_update_mask] += 1
            # agent의 최대 phase를 넘어가면 해당 agent의 action index 0으로 초기화
            clear_matrix = torch.ge(action_index_matrix, phase_num_matrix)
            action_index_matrix[clear_matrix] = 0
            # mask update, matrix True로 전환
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False

            # 만약 action이 끝나기 3초전이면 yellow signal 적용, reward 갱신
            yellow_mask = torch.eq(
                t_agent, action_matrix[0, action_index_matrix]-3)  # 3초먼저 yellow로 바꿈
            for y in torch.nonzero(yellow_mask):
                traci.trafficlight.setRedYellowGreenState(
                    TL_RL_LIST[y], 'y'*(12+4*configs['num_lanes']))
            # environment에 적용
            # action 적용함수, traci.simulationStep 있음
            next_state = env.step(actions, action_index_matrix, yellow_mask)

            # env속에 agent별 state를 꺼내옴, max_offset+period 이상일 때 시작
            if step >= int(torch.max(OFFSET)+torch.max(TL_PERIOD)):
                rep_state, rep_action, rep_reward, rep_next_state = env.get_state(
                    mask_matrix)
                agent.save_replay(rep_state, rep_action, rep_reward,
                                  rep_next_state, mask_matrix)  # dqn
                total_reward += rep_reward.sum()
            # print(mask_matrix)

            agent.update(mask_matrix)
            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % agent.configs['target_update_period'] == 0:
            agent.target_update()  # dqn
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward.sum(), arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()