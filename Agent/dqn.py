import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random
import os
from collections import namedtuple
from copy import deepcopy
from Agent.base import RLAlgorithm, ReplayMemory, merge_dict
from torch.utils.tensorboard import SummaryWriter

DEFAULT_CONFIG = {
    'gamma': 0.99,
    'tau': 0.995,
    'batch_size': 32,
    'experience_replay_size': 1e5,
    'epsilon': 0.9,
    'epsilon_decay_rate': 0.98,
    'fc_net': [16, 16, 16],
    'lr': 0.00005,  # no via
    # 'lr': 0.001, # via
    'target_update_period': 5,
    'lr_decay_rate': 0.98,
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(QNetwork, self).__init__()
        self.configs = configs
        self.state_space = input_size
        self.action_space = output_size
        self.time_size = configs['time_size']

        # build nn
        self.fc1 = nn.Linear(self.state_space, self.configs['fc_net'][0])
        self.fc2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc3 = nn.Linear(self.configs['fc_net'][1], self.action_space)

        # seconde network
        self.fc_y1 = nn.Linear(
            self.state_space+1, self.configs['fc_net'][0])  # action_space내의 max값만 들어감
        self.fc_y2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc_y3 = nn.Linear(self.configs['fc_net'][1], self.time_size)
        # self.fc4 = nn.Linear(30, self.action_space)

    def forward(self, input_x):
        x = f.leaky_relu(self.fc1(input_x))
        x = f.dropout(x, 0.4)
        x = f.leaky_relu(self.fc2(x))
        x = f.dropout(x, 0.3)
        # x = f.softmax(self.fc3(x))
        x = self.fc3(x)
        y = torch.cat((x.max(1)[1].detach().view(-1, 1),
                       input_x), dim=1)  # 최댓위치를 넣기위함
        y = f.leaky_relu(self.fc_y1(y))
        y = f.dropout(y, 0.4)
        y = f.leaky_relu(self.fc_y2(y))
        y = f.leaky_relu(self.fc_y3(y))
        #x = f.softmax(self.fc4(x), dim=0)
        return x, y  # q value


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        print("Current_Mode:",configs['mode'])
        if configs['mode']=='train':
            os.mkdir(os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model'))
            configs = merge_dict(configs, DEFAULT_CONFIG)
        self.configs=configs
        self.state_space = self.configs['state_space']
        self.action_space = self.configs['action_space']
        self.action_size = self.configs['action_size']
        self.gamma = self.configs['gamma']
        self.epsilon = self.configs['epsilon']
        self.criterion = nn.MSELoss()
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.experience_replay = ReplayMemory(
            self.configs['experience_replay_size'])
        self.batch_size = self.configs['batch_size']
        self.num_agent = len(self.configs['tl_rl_list'])
        if self.configs['model'].lower() == 'frap':
            from Agent.Model.FRAP import FRAP
            model = FRAP(self.state_space*self.num_agent*self.configs['num_phase'], self.action_space*self.num_agent,
                         self.configs['device'])
            # model.add_module('QNetwork',
            #                  QNetwork(self.state_space, self.action_space, self.configs))
        else:
            model = QNetwork(self.state_space*self.configs['num_phase']*self.num_agent, self.action_space*self.num_agent,
                             self.configs)  # 1개 네트워크용
        model.to(self.configs['device'])
        self.mainQNetwork = deepcopy(model).to(self.configs['device'])
        print("========NETWORK==========\n", self.mainQNetwork)
        self.targetQNetwork = deepcopy(model).to(self.configs['device'])
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())
        self.optimizer = optim.Adam(
            self.mainQNetwork.parameters(), lr=self.lr)
        self.action = tuple()
        self.running_loss = 0
        if self.configs['mode'] == 'train':
            self.mainQNetwork.train()
        elif self.configs['mode'] == 'test':
            self.mainQNetwork.eval()
        self.targetQNetwork.eval()

    def get_action(self, state):

        if random.random() > self.epsilon:  # epsilon greedy
            with torch.no_grad():
                rate_Q, time_Q = self.mainQNetwork(state)
                rate_action = torch.max(rate_Q, dim=1)[1].view(
                    1, 1)  # 가로로 # action 수가 늘어나면 view(1,action_size)
                time_action = torch.max(time_Q, dim=1)[1].view(
                    1, 1)  # 가로로 # action 수가 늘어나면 view(1,action_size)
                # agent가 늘어나면 view(agents,action_size)
                action = torch.cat((rate_action, time_action), dim=1)
                self.action += tuple(action)  # 기록용
            return action
        else:
            rate_action = torch.tensor([random.randint(0, self.configs['action_space']-1)  # 여기서 3일 때, phase 4 7일때 phase8
                                        for i in range(self.num_agent)], device=self.configs['device']).view(1, -1)
            time_action = torch.tensor(
                [random.randint(0, self.configs['time_size']-1) for i in range(self.num_agent)]).view(1, -1)
            action = torch.cat((rate_action, time_action), dim=1)
            self.action += tuple(action)  # 기록용

            return action

    def target_update(self):
        # soft update
        # for target, source in zip(self.targetQNetwork.parameters(), self.mainQNetwork.parameters()):
        #     target.data.copy_(
        #         target.data*(1-self.configs['tau']), source.data*self.configs['tau'])
        # Hard Update
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)

    def update(self, done=False):
        if len(self.experience_replay) < self.configs['batch_size']:
            return

        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.configs['device'], dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=0)

        # dim=0인 이유는 batch 끼리 cat 하는 것이기 때문임
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        reward_batch = torch.tensor(batch.reward).to(self.configs['device'])

        # Q(s_t, a) 계산 - 모델이 action batch의 a'일때의 Q(s_t,a')를 계산할때, 취한 행동 a'의 column 선택(column이 Q)

        rate_state_action_values, time_state_action_values = self.mainQNetwork(
            state_batch)
        rate_state_action_values = rate_state_action_values.gather(
            1, action_batch[:, 0].view(-1, 1))
        time_state_action_values = time_state_action_values.gather(
            1, action_batch[:, 1].view(-1, 1))
        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        rate_next_state_values = torch.zeros(
            self.configs['batch_size'], device=self.configs['device'], dtype=torch.float)
        time_next_state_values = torch.zeros(
            self.configs['batch_size'], device=self.configs['device'], dtype=torch.float)
        rate_Q, time_Q = self.targetQNetwork(non_final_next_states)
        rate_next_state_values[non_final_mask] = rate_Q.max(
            1)[0].detach().to(self.configs['device'])
        time_next_state_values[non_final_mask] = time_Q.max(1)[0].detach().to(
            self.configs['device'])  # .to(self.configs['device'])  # 자신의 Q value 중에서max인 value를 불러옴

        # 기대 Q 값 계산
        rate_expected_state_action_values = (
            rate_next_state_values * self.configs['gamma']) + reward_batch
        time_expected_state_action_values = (
            time_next_state_values * self.configs['gamma']) + reward_batch

        # loss 계산
        rate_loss = self.criterion(rate_state_action_values,
                                   rate_expected_state_action_values.unsqueeze(1))
        time_loss = self.criterion(time_state_action_values,
                                   time_expected_state_action_values.unsqueeze(1))
        self.running_loss += rate_loss/self.configs['batch_size']
        self.running_loss += time_loss/self.configs['batch_size']
        # 모델 최적화
        self.optimizer.zero_grad()
        rate_loss.backward()
        time_loss.backward()
        for param in self.mainQNetwork.parameters():
            param.grad.data.clamp_(-1, 1)  # 값을 -1과 1로 한정시켜줌 (clipping)
        self.optimizer.step()

    def update_hyperparams(self, epoch):
        # decay rate (epsilon greedy)
        if self.epsilon > 0.005:
            self.epsilon *= self.epsilon_decay_rate

        # decay learning rate
        if self.lr > 0.05*self.configs['lr']:
            self.lr = self.lr_decay_rate*self.lr

    def save_weights(self, name):
        torch.save(self.mainQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'.h5'))
        torch.save(self.targetQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_target.h5'))

    def load_weights(self, name):
        print(self.configs['replay_epoch'])
        self.mainQNetwork.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_'+self.configs['replay_epoch']+'.h5')))
        self.mainQNetwork.eval()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)
        writer.add_scalar('hyperparameter/epsilon',
                          self.epsilon, self.configs['max_steps']*epoch)

        action_distribution = torch.cat(self.action, 0)
        writer.add_histogram('hist/episode/action_distribution', action_distribution,
                             epoch)  # 1 epoch마다
        self.action = tuple()
        # clear
        self.running_loss = 0
