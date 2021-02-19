import torch
import traci
import time
from utils import load_params
from Agent.base import merge_dict_non_conflict


def city_dqn_test(flags, sumoCmd, configs):
    # Environment Setting
    from Agent.super_dqn import Trainer
    from Env.CityEnv import CityEnv
    # init test setting
    if flags.replay_name is not None:
        # 여기앞에 configs 설정해도 의미 없음
        configs = load_params(configs, flags.replay_name)
        configs['replay_epoch'] = str(flags.replay_epoch)
        configs['mode'] = 'test'

    phase_num_matrix = torch.tensor(  # 각 tl이 갖는 최대 phase갯수
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])

    agent = Trainer(configs)
    agent.save_params(configs['time_data'])
    agent.load_weights(flags.replay_name)
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
    # state initialization
    # agent setting
    # check performance
    avg_waiting_time = 0
    avg_part_velocity = 0
    total_reward = 0
    avg_velocity = 0
    arrived_vehicles = 0
    part_velocity = list()
    with torch.no_grad():
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
            # check performance
            for _, interests in enumerate(configs['interest_list']):
                for interest in interests:
                    # 신호군 흐름
                    avg_inEdge_velocity = list()
                    if interest['inflow'] != None:
                        inflow_vehicle_list = traci.edge.getLastStepVehicleIDs(
                            interest['inflow'])
                        for inflow_vehicle_id in inflow_vehicle_list:
                            avg_inEdge_velocity.append(
                                traci.vehicle.getSpeed(inflow_vehicle_id))
                        if len(avg_inEdge_velocity) != 0:
                            part_velocity.append(torch.tensor(
                                avg_inEdge_velocity, dtype=torch.float).mean())
                        # 차량의 대기시간
                        if traci.edge.getLastStepVehicleNumber(interest['inflow']) != 0:
                            avg_waiting_time += traci.edge.getWaitingTime(interest['inflow'])/float(
                                traci.edge.getLastStepVehicleNumber(interest['inflow']))
                    avg_outEdge_velocity = list()
                    if interest['outflow'] != None:
                        outflow_vehicle_list = traci.edge.getLastStepVehicleIDs(
                            interest['outflow'])
                        for outflow_vehicle_id in outflow_vehicle_list:
                            avg_outEdge_velocity.append(
                                traci.vehicle.getSpeed(outflow_vehicle_id))
                        if len(avg_inEdge_velocity) != 0:
                            part_velocity.append(torch.tensor(
                                avg_inEdge_velocity, dtype=torch.float).mean())

            # # 전체 흐름
            # vehicle_list = traci.vehicle.getIDList()
            # for i, vehicle in enumerate(vehicle_list):
            #     speed = traci.vehicle.getSpeed(vehicle)
            #     avg_velocity = float((i)*avg_velocity+speed) / \
            #         float(i+1)

            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        avg_part_velocity = torch.tensor(
            part_velocity, dtype=torch.float).mean()
        b = time.time()
        traci.close()
        print("time:", b-a)
        print('======== return: {} arrived number:{}, avg_waiting_time:{}, avg_velocity:{}, avg_part_velocity: {}'.format(
            total_reward, arrived_vehicles, avg_waiting_time/MAX_STEPS, avg_velocity, avg_part_velocity))
