from gen_net import Network
from configs import EXP_CONFIGS
import math
import argparse
import json
import os
import sys
import time
from xml.etree.ElementTree import parse


class MapNetwork(Network):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs=configs
        self.tl_rl_list = list()
        self.offset_list = list()
        self.phase_list = list()
        self.common_phase = list()
        self.net_file_path = os.path.join(self.configs['current_path'],'Network',self.configs['load_file_name']+'.net.xml')
        self.rou_file_path = os.path.join(self.configs['current_path'],'Network',self.configs['load_file_name']+'.rou.xml')

    def specify_traffic_light(self):
        self.traffic_light = traffic_light

        return traffic_light

    def get_tl_from_xml(self):
        # , 'Network') # 가동시
        tl_tree = parse(self.net_file_path)
        tlLogicList = tl_tree.findall('tlLogic')
        for tlLogic in tlLogicList:
            self.offset_list.append(tlLogic.attrib['offset'])
            self.tl_rl_list.append(tlLogic.attrib['id'])  # rl 조종할 tl_rl추가
            phaseList = tlLogic.findall('phase')
            phase_state_list = list()
            phase_duration_list = list()
            phase_period = 0
            for phase in phaseList:
                phase_state_list.append(phase.attrib['state'])
                phase_duration_list.append(int(phase.attrib['duration']))
                phase_period += int(phase.attrib['duration'])
            self.phase_list.append(phase_state_list)
            self.common_phase.append(phase_duration_list)

        configs = {
            'tl_rl_list': self.tl_rl_list,
            'offset': self.offset_list,
            'phase_list': self.phase_list,
            'common_phase': self.common_phase,
        }

        return configs

    def gen_net_from_xml(self):
        net_tree=parse(self.net_file_path)
        gen_file_name=str(os.path.join(self.configs['current_path'],'training_data',self.configs['time_data'],'net_data',self.configs['time_data']+'.net.xml'))
        net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)

    def gen_rou_from_xml(self):
        net_tree=parse(self.rou_file_path)
        gen_file_name=str(os.path.join(self.configs['current_path'],'training_data',self.configs['time_data'],'net_data',self.configs['time_data']+'.rou.xml'))
        net_tree.write(gen_file_name, encoding='UTF-8', xml_declaration=True)


    def print(self):
        return 123

