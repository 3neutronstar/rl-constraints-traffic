from gen_net import Network
from configs import EXP_CONFIGS
import math


class GridNetwork(Network):
    def __init__(self, configs, grid_num):
        self.grid_num = grid_num
        self.configs = configs
        self.tl_rl_list = list()
        super().__init__(self.configs)

    def specify_node(self):
        nodes = list()
        # inNode
        #
        #   . .
        #   | |
        # .-*-*-.
        #   | |
        #   . .
        center = float(self.grid_num)/2.0
        for x in range(self.grid_num):
            for y in range(self.grid_num):
                node_info = dict()
                node_info = {
                    'id': 'n_'+str(x)+'_'+str(y),
                    'type': 'traffic_light',
                    'tl': 'n_'+str(x)+'_'+str(y),
                }
                # if self.grid_num % 2==0: # odd due to index rule
                #     grid_x=self.configs['laneLength']*(x-center_x)
                #     grid_x=self.configs['laneLength']*(center_y-y)

                # else: # even due to index rule
                grid_x = self.configs['laneLength']*(x-center)
                grid_y = self.configs['laneLength']*(center-y)

                node_info['x'] = str('%.1f' % grid_x)
                node_info['y'] = str('%.1f' % grid_y)
                nodes.append(node_info)
                self.tl_rl_list.append(node_info)

        # outNode
        #   * *
        #   | |
        # *-.-.-*
        #   | |
        for i in range(self.grid_num):
            grid_y = (center-i)*self.configs['laneLength']
            grid_x = (i-center)*self.configs['laneLength']
            node_information = [{
                'id': 'n_'+str(i)+'_u',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (-center*self.configs['laneLength']+(self.grid_num+1)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_r',
                'x': str('%.1f' % (-center*self.configs['laneLength']+(self.grid_num)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            },
                {
                'id': 'n_'+str(i)+'_d',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (+center*self.configs['laneLength']-(self.grid_num)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_l',
                'x': str('%.1f' % (+center*self.configs['laneLength']-(self.grid_num+1)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            }]
            for _, node_info in enumerate(node_information):
                nodes.append(node_info)
        self.configs['node_info'] = nodes
        self.nodes = nodes
        return nodes

    def specify_edge(self):
        edges = list()
        edges_dict = dict()
        for i in range(self.grid_num):
            edges_dict['n_{}_l'.format(i)] = list()
            edges_dict['n_{}_r'.format(i)] = list()
            edges_dict['n_{}_u'.format(i)] = list()
            edges_dict['n_{}_d'.format(i)] = list()

        for y in range(self.grid_num):
            for x in range(self.grid_num):
                edges_dict['n_{}_{}'.format(x, y)] = list()

                # outside edge making
                if x == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_l'.format(y))
                    edges_dict['n_{}_l'.format(y)].append(
                        'n_{}_{}'.format(x, y))
                if y == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_u'.format(x))
                    edges_dict['n_{}_u'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if y == self.grid_num-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_d'.format(x))
                    edges_dict['n_{}_d'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if x == self.grid_num-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_r'.format(y))
                    edges_dict['n_{}_r'.format(y)].append(
                        'n_{}_{}'.format(x, y))

                # inside edge making
                if x+1 < self.grid_num:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x+1, y))

                if y+1 < self.grid_num:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y+1))
                if x-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x-1, y))
                if y-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y-1))

        for _, dict_key in enumerate(edges_dict.keys()):
            for i, _ in enumerate(edges_dict[dict_key]):
                edge_info = dict()
                edge_info = {
                    'from': dict_key,
                    'id': "{}_to_{}".format(dict_key, edges_dict[dict_key][i]),
                    'to': edges_dict[dict_key][i],
                    'numLanes': self.num_lanes
                }
                edges.append(edge_info)
        self.edges = edges
        self.configs['edge_info'] = edges
        return edges

    def specify_flow(self):
        flows = list()
        direction_list = ['l', 'u', 'd', 'r']
        # via 제작
        # self.grid_num
        # for i,direction in enumerate(direction_list):
        #     if direction=='l':

        # 삽입
        for _, edge in enumerate(self.edges):
            for i, _ in enumerate(direction_list):
                if direction_list[i] in edge['from']:
                    for _, checkEdge in enumerate(self.edges):
                        if edge['from'][-3] == checkEdge['to'][-3] and checkEdge['to'][-1] == direction_list[3-i] and direction_list[i] in edge['from']:

                            # 위 아래
                            if checkEdge['to'][-1] == direction_list[1] or checkEdge['to'][-1] == direction_list[2]:
                                self.configs['probability'] = '0.133'
                            else:
                                self.configs['probability'] = '0.388'
                            via_string = str()
                            node_x_y = edge['id'][2]  # 끝에서 사용하는 기준 x나 y
                            if 'r' in edge['id']:
                                for i in range(self.configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i-1, node_x_y)
                            elif 'l' in edge['id']:
                                for i in range(self.configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i+1, node_x_y)
                            elif 'u' in edge['id']:
                                for i in range(self.configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i+1)
                            elif 'd' in edge['id']:
                                for i in range(self.configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i-1)

                            flows.append({
                                'from': edge['id'],
                                'to': checkEdge['id'],
                                'id': edge['from'],
                                'begin': str(self.configs['flow_start']),
                                'end': str(self.configs['flow_end']),
                                'probability': self.configs['probability'],
                                'reroute': 'false',
                                'via': edge['id']+" "+via_string+" "+checkEdge['id'],
                                'departPos': "base",
                                'departLane': 'best',
                            })

        self.flows = flows
        self.configs['vehicle_info'] = flows
        return flows

    def specify_connection(self):
        connections = list()

        self.connections = connections
        return connections

    def specify_traffic_light(self):
        traffic_lights = []
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                # 4행시
                phase_set = [
                    {'duration': '37',  # 1
                     'state': 'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
                         g*num_lanes, g, r*num_lanes, r),
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 2
                     'state': 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
                         g*num_lanes, g, r*num_lanes, r),  # current
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 1
                     'state': 'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
                         g*num_lanes, g, r*num_lanes, r),
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 1
                     'state': 'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
                         g*num_lanes, g, r*num_lanes, r),  # current
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                ]
                # 2행시
                # phase_set = [
                #     {'duration': '42',
                #      'state': 'G{}ggr{}rrG{}ggr{}rr'.format('G'*num_lanes, 'r'*num_lanes, 'G'*num_lanes, 'r'*num_lanes),
                #      },
                #     {'duration': '3',
                #      'state': 'y{}yyr{}rry{}yyr{}rr'.format('y'*num_lanes, 'r'*num_lanes, 'y'*num_lanes, 'r'*num_lanes),
                #      },
                #     {'duration': '42',
                #      'state': 'r{}rrG{}ggr{}rrG{}gg'.format('r'*num_lanes, 'G'*num_lanes, 'r'*num_lanes, 'G'*num_lanes),
                #      },
                #     {'duration': '3',
                #      'state': 'r{}rry{}yyr{}rry{}yy'.format('r'*num_lanes, 'y'*num_lanes, 'r'*num_lanes, 'y'*num_lanes),
                #      },
                # ]
                traffic_lights.append({
                    'id': 'n_{}_{}'.format(i, j),
                    'type': 'static',
                    'programID': 'n_{}_{}'.format(i, j),
                    'offset': '0',
                    'phase': phase_set,
                })
        # rl_phase_set = [
        #     {'duration': '35',  # 1
        #      'state': 'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
        #          g*num_lanes, g, r*num_lanes, r),
        #      },
        #     {'duration': '5',
        #      'state': 'y'*20,
        #      },
        #     {'duration': '35',  # 2
        #      'state': 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
        #          g*num_lanes, g, r*num_lanes, r),  # current
        #      },
        #     {'duration': '5',
        #      'state': 'y'*20,
        #      },
        #     {'duration': '35',  # 1
        #      'state': 'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
        #          g*num_lanes, g, r*num_lanes, r),
        #      },
        #     {'duration': '5',
        #      'state': 'y'*20,
        #      },
        #     {'duration': '35',  # 1
        #      'state': 'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
        #          g*num_lanes, g, r*num_lanes, r),  # current
        #      },
        #     {'duration': '5',
        #      'state': 'y'*20,
        #      },
        # ]
        # traffic_lights.append({
        #     'id': 'n_1_1',
        #     'type': 'static',
        #     'programID': 'n_1_1',
        #     'offset': '0',
        #     'phase': rl_phase_set,
        # })
        return traffic_lights

    def generate_cfg(self, route_exist, mode='simulate'):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_add_xml()
        self._generate_net_xml()
        super().generate_cfg(route_exist, mode)

    def get_configs(self):
        side_list = ['u', 'r', 'd', 'l']
        NET_CONFIGS = dict()
        interest_list = list()
        phase_dict = dict()
        node_list = self.configs['node_info']
        # grid에서는 자동 생성기 따라서 사용해도 무방함 #map완성되면 통일 가능
        x_y_end = self.configs['grid_num']-1
        for _, node in enumerate(node_list):
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
        # phase 생성
        for tl_rl in self.tl_rl_list:
            phase_dict[tl_rl['id']] = self._phase_list()
        # agent별 reward,state,next_state,action저장용
        # 관심 노드와 interest inflow or outflow edge 정렬
        node_interest_pair = dict()
        for _, node in enumerate(node_list):
            if node['id'][-1] not in side_list:
                node_interest_pair['{}'.format(
                    node['id'])] = list()
                for _, interest in enumerate(interest_list):
                    if node['id'][-3:] == interest['id'][-3:]:  # 좌표만 받기
                        node_interest_pair['{}'.format(
                            node['id'])].append(interest)

        NET_CONFIGS['interest_list'] = interest_list
        NET_CONFIGS['node_interest_pair'] = node_interest_pair
        NET_CONFIGS['phase_dict'] = phase_dict
        return NET_CONFIGS

    def _phase_list(self):
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        phase_list = [
            'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
                g*num_lanes, g, r*num_lanes, r),
            'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
                g*num_lanes, g, r*num_lanes, r),  # current
            'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
                g*num_lanes, g, r*num_lanes, r),  # current
        ]
        return phase_list

    def _get_tl_rl_list(self):
        return self.tl_rl_list


if __name__ == "__main__":
    grid_num = 3
    configs = EXP_CONFIGS
    configs['grid_num'] = grid_num
    configs['file_name'] = '{}x{}grid'.format(grid_num, grid_num)
    a = GridNetwork(configs, grid_num)
    a.sumo_gui()
