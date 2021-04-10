import torch
import torch.nn as nn
import torch.nn.functional as F


class FRAP(nn.Module):
    def __init__(self, input_size, output_size,device):
        self.device=device
        phase_input_size = 1
        vehicle_input_size = 1
        super(FRAP, self).__init__()
        # A
        self.phase_competition_mask = torch.tensor([
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            [0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],  # d
            [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]],device=device).view(1, 1, 7,8)  # 완전 겹치면 1, 겹치다 말면 0.5 자기자신은 0
        self.demand_model_phase = [nn.Sequential(
            nn.Linear(phase_input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
        ).to(device) for _ in range(8)]
        self.demand_model_vehicle = [nn.Sequential(
            nn.Linear(vehicle_input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
        ).to(device) for _ in range(8)]
        self.embedding = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU()
        ).to(device)
        self.conv_pair = nn.Sequential(
            nn.Conv2d(32, 20, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(1, 1)),
            nn.ReLU(),
        ).to(device)
        self.conv_mask_pair = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 20, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(1, 1)),
            nn.ReLU(),
        ).to(device)
        self.conv_competition = nn.Sequential(
            nn.Conv2d(20, 8, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(1, 1)),
            nn.ReLU(), #여기까지 끝남
        ).to(device)
        self.Qnetwork=nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,output_size),
            nn.ReLU(),
        )

    def forward(self, state):
        for k in range(state.size()[0]):
            demand = list()
            phase_demand = list()
            for i in range(8):
                state=state.detach()
                x_phase = self.demand_model_vehicle[i](state[k][i].view(1, 1))
                x_vehicle = self.demand_model_phase[i](state[k][8+i].view(1, 1))
                x = torch.cat((x_phase, x_vehicle), dim=1)
                x = self.embedding(x)
                demand.append(x)
            # element wise sum
            phase_demand.append(torch.add(demand[0], demand[4]))  # a
            phase_demand.append(torch.add(demand[0], demand[1]))  # b
            phase_demand.append(torch.add(demand[4], demand[5]))  # c
            phase_demand.append(torch.add(demand[1], demand[5]))  # d
            phase_demand.append(torch.add(demand[2], demand[6]))  # e
            phase_demand.append(torch.add(demand[2], demand[3]))  # f
            phase_demand.append(torch.add(demand[6], demand[7]))  # g
            phase_demand.append(torch.add(demand[3], demand[7]))  # h
            
            # phase pair representation
            z = torch.zeros((state.size()[0],7,8, 32), dtype=torch.float,device=self.device)
            # print("list_len",len(phase_demand))
            for j, phase_j in enumerate(phase_demand):
                for i, phase_i in enumerate(phase_demand):
                    # print(i," ",j)
                    if i == j:
                        continue
                    elif i > j:
                        # print(torch.cat((phase_i, phase_j), dim=1))
                        z[k][i-1, j] = torch.cat((phase_i, phase_j), dim=1)
                    else:  # y>x
                        # print(torch.cat((phase_i, phase_j), dim=1))
                        z[k][i, j] = torch.cat((phase_i, phase_j), dim=1)
        z = z.view(state.size()[0], 32, 7,8) # 7x8 : height,width
        z = self.conv_pair(z)

        # phase competition mask
        y = self.conv_mask_pair(self.phase_competition_mask)
        results = (z*y)  # element-wise multiplication -> grad?
        results = self.conv_competition(results)  # size(1,1,1,8)
        results = torch.sum(results, 2).view(state.size()[0], 8)  # size (1,8)
        Qvalue = self.Qnetwork(results)
        return Qvalue
