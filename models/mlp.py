import torch
from torch import nn

# 全链接神经网络
class MlpNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim = 5):
        super(MlpNet,self).__init__()
        self.an1 = nn.Linear(state_dim,hidden_dim)
        self.an2 = nn.Linear(hidden_dim,hidden_dim)
        self.an3 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = torch.relu(self.an1(x))
        x = torch.relu(self.an2(x))
        return self.an3(x)