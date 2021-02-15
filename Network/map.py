
import os
import torch
from xml.etree.ElementTree import parse
from gen_net import Network
from configs import EXP_CONFIGS


class MapNetwork(Network):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_rl_list = list()
        self.offset_list = list()
        self.phase_list = list()
        self.common_phase = list()
        self.net_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.net.xml')
        self.rou_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.rou.xml')

    def specify_traffic_light(self):
        self.traffic_light = traffic_light

        return traffic_light

    def get_tl_from_xml(self):
        NET_CONFIGS = dict()
        NET_CONFIGS['phase_num_actions'] = {2: [[0, 0], [1, -1]],
                                            3: [[0, 0, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [-1, 1, 0]],
                                            4: [[0, 0, 0, 0], [1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1],
                                                [1, 0, 0, -1], [1, 0, -1, 0], [1, 0, 0, -1], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1]],
                                            5: [[0,0,0,0,0]],
                                            6:[[0,0,0,0,0,0]],}

        NET_CONFIGS['rate_action_space'] = {2: len(NET_CONFIGS['phase_num_actions'][2]), 3: len(
            NET_CONFIGS['phase_num_actions'][3]), 4: len(NET_CONFIGS['phase_num_actions'][4])}
        NET_CONFIGS['tl_period'] = list()
        traffic_info = dict()
        net_tree = parse(self.net_file_path)
        tlLogicList = net_tree.findall('tlLogic')
        NET_CONFIGS['time_action_space'] = list()

        #traffic info 저장
        for tlLogic in tlLogicList:
            tl_id = tlLogic.attrib['id']
            traffic_info[tl_id] = dict()
            traffic_node_info = traffic_info[tl_id]
            traffic_node_info['min_phase'] = list()
            traffic_node_info['phase_duration'] = list()
            traffic_node_info['max_phase'] = list()
            traffic_node_info['min_phase'] = list()
            traffic_node_info['min_phase'] = list()

            # rl agent 갯수 정리
            self.tl_rl_list.append(tlLogic.attrib['id'])  # rl 조종할 tl_rl추가
            # offset 저장
            traffic_node_info['offset'] = int(tlLogic.attrib['offset'])
            self.offset_list.append(traffic_node_info['offset'])

            # phase전체 찾기
            phaseList = tlLogic.findall('phase')
            phase_state_list = list()
            phase_duration_list = list()
            common_phase_list = list()
            phase_index_list = list()
            min_duration_list = list()
            max_duration_list = list()
            tl_period = 0  # phase set의 전체 길이
            # 각 phase에 대해서 길이 찾기 등등
            num_phase = 0  # phase갯수 filtering
            for i, phase in enumerate(phaseList):
                phase_state_list.append(phase.attrib['state'])
                phase_duration_list.append(int(phase.attrib['duration']))
                tl_period += int(phase.attrib['duration'])
                if int(phase.attrib['duration']) > 5:  # Phase 로 간주할 숫자
                    num_phase += 1
                    min_duration_list.append(int(phase.attrib['minDuration']))
                    max_duration_list.append(int(phase.attrib['maxDuration']))
                    phase_index_list.append(i)
                    common_phase_list.append(int(phase.attrib['duration']))

            # dictionary에 담기
            traffic_node_info['phase_list'] = phase_state_list
            traffic_node_info['phase_duration'] = phase_duration_list
            traffic_node_info['common_phase'] = common_phase_list
            traffic_node_info['phase_index'] = phase_index_list
            # 각 신호별 길이
            traffic_node_info['period'] = tl_period
            NET_CONFIGS['tl_period'].append(tl_period)
            traffic_node_info['matrix_actions'] = NET_CONFIGS['phase_num_actions'][num_phase]
            traffic_node_info['min_phase'] = min_duration_list
            traffic_node_info['max_phase'] = max_duration_list
            traffic_node_info['num_phase'] = num_phase
            # 각 tl_rl의 time_action_space지정
            NET_CONFIGS['time_action_space'].append(round((torch.min(torch.tensor(traffic_node_info['max_phase'])-torch.tensor(
                traffic_node_info['common_phase']), torch.tensor(traffic_node_info['common_phase'])-torch.tensor(traffic_node_info['min_phase']))/2).mean().item()))

            self.phase_list.append(phase_state_list)
            self.common_phase.append(phase_duration_list)

        # TODO  node interest pair 계산기 network base에 생성
        maximum = 0
        for key in traffic_info.keys():
            if maximum < len(traffic_info[key]['phase_duration']):
                maximum = len(traffic_info[key]['phase_duration'])
        NET_CONFIGS['max_phase_num'] = maximum

        # road용
        # edge info 저장
        self.configs['edge_info']=list()
        edge_list=list() # edge존재 확인용
        edges=net_tree.findall('edge')
        for edge in edges:
            if 'function' not in edge.attrib.keys():
                self.configs['edge_info'].append({
                    'id':edge.attrib['id'],
                    'from':edge.attrib['from'],
                    'to':edge.attrib['to'],
                })
                edge_list.append(edge.attrib['id'])
        # node info 저장
        self.configs['node_info'] = list()
        # interest list
        interest_list = list()
        # node interest pair
        node_interest_pair = dict()
        junctions = net_tree.findall('junction')

        # network용
        if self.configs['network']!='3x3grid':
            for junction in junctions:
                node_id=junction.attrib['id']
                if junction.attrib['type'] == "traffic_light": # 정상 node만 분리, 신호등 노드
                    self.configs['node_info'].append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                    # node 결정 완료
                    # edge는?
                    for edge in self.configs['edge_info']:
                        i=0
                        interest=dict()
                        if edge['to']==node_id: # inflow
                            interest['id']=node_id+'_{}'.format(i)
                            interest['inflow']=edge['id']
                            tmp_edge=str(-int(edge['id']))
                            if tmp_edge in edge_list:
                                interest['outflow']=tmp_edge
                            else:
                                interest['outflow']=None
                            interest_list.append(interest)

                            i+=1 # index표기용
                            
                        elif edge['from']==node_id:
                            interest['id']=node_id+'_{}'.format(i)
                            interest['outflow']=edge['id']
                            tmp_edge=str(-int(edge['id']))
                            if tmp_edge in edge_list:
                                interest['inflow']=tmp_edge
                            else:
                                interest['inflow']=None
                            interest_list.append(interest)
                            i+=1 # index표기용

                        # outflow가 존재하는 지 확인 후 list에 삽입
                        
                    node_interest_pair[node_id]=interest_list


                elif junction.attrib['type'] == "priority": # 정상 node만 분리
                    self.configs['node_info'].append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                else:
                    pass



        # 임시 3x3 grid 용
        if self.configs['network']=='3x3grid':
            side_list = ['u', 'r', 'd', 'l']
            self.configs['grid_num'] = 3
            x_y_end = self.configs['grid_num']-1
            # grid junction
            junctions = net_tree.findall('junction')
            for junction in junctions:
                if junction.attrib['type'] != "internal":
                    self.configs['node_info'].append({
                        'id': junction.attrib['id'],
                        'type': junction.attrib['type'],
                    })

            for _, node in enumerate(self.configs['node_info']):
                if node['id'][-1] not in side_list:
                    x = int(node['id'][-3])
                    y = int(node['id'][-1])
                    left_x = x-1
                    left_y = y
                    right_x = x+1
                    right_y = y
                    down_x = x
                    down_y = y+1  # 아래로가면 y는 숫자가 늘어남
                    up_x = x
                    up_y = y-1  # 위로가면 y는 숫자가 줄어듦

                    if x == 0:
                        left_y = 'l'
                        left_x = y
                    if y == 0:
                        up_y = 'u'
                    if x == x_y_end:
                        right_y = 'r'
                        right_x = y
                    if y == x_y_end:
                        down_y = 'd'
                    # up
                    interest_list.append(
                        {
                            'id': 'u_{}'.format(node['id'][2:]),
                            'inflow': 'n_{}_{}_to_n_{}_{}'.format(up_x, up_y, x, y),
                            'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, up_x, up_y),
                        }
                    )
                    # right
                    interest_list.append(
                        {
                            'id': 'r_{}'.format(node['id'][2:]),
                            'inflow': 'n_{}_{}_to_n_{}_{}'.format(right_x, right_y, x, y),
                            'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, right_x, right_y),
                        }
                    )
                    # down
                    interest_list.append(
                        {
                            'id': 'd_{}'.format(node['id'][2:]),
                            'inflow': 'n_{}_{}_to_n_{}_{}'.format(down_x, down_y, x, y),
                            'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, down_x, down_y),
                        }
                    )
                    # left
                    interest_list.append(
                        {
                            'id': 'l_{}'.format(node['id'][2:]),
                            'inflow': 'n_{}_{}_to_n_{}_{}'.format(left_x, left_y, x, y),
                            'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, left_x, left_y),
                        }
                    )
            for _, node in enumerate(self.configs['node_info']):
                if node['id'][-1] not in side_list:
                    node_interest_pair[node['id']] = list()
                    for _, interest in enumerate(interest_list):
                        if node['id'][-3:] == interest['id'][-3:]:  # 좌표만 받기
                            node_interest_pair[node['id']].append(interest)

        #정리
        NET_CONFIGS['traffic_node_info'] = traffic_info
        NET_CONFIGS['interest_list'] = interest_list
        NET_CONFIGS['node_interest_pair'] = node_interest_pair
        NET_CONFIGS['tl_rl_list'] = self.tl_rl_list
        NET_CONFIGS['offset'] = self.offset_list
        NET_CONFIGS['phase_list'] = self.phase_list
        NET_CONFIGS['common_phase'] = self.common_phase

        return NET_CONFIGS

    def gen_net_from_xml(self):
        net_tree = parse(self.net_file_path)
        if self.configs['mode']=='train' or self.configs['mode']=='test':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                            self.configs['time_data'], 'net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)
        else: #simulate
            gen_file_name = str(os.path.join(self.configs['current_path'], 'Net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)

    def gen_rou_from_xml(self):
        net_tree = parse(self.rou_file_path)
        if self.configs['mode']=='train' or self.configs['mode']== 'test':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                            self.configs['time_data'], 'net_data', self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)
        else:
            gen_file_name = str(os.path.join(self.configs['current_path'], 'Net_data',
                                             self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)
