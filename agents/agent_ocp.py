import torch
import numpy as np
from torch import nn
from models import ActorNet,CriticNet
from utils import setup_code_environment,load_config,get_three_lane_paths
from random import sample

class AgentOcp:
    def __init__(self,env,state_dim,
                 hidden_dim = 256,action_dim = 2,actor_lr = 1e-3,critic_lr = 1e-4):
        """
        智能体
        :param state_dim: 状态空间维度
        :param env: 环境
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作维度
        :param actor_lr: 策略学习率
        :param critic_lr: 评论家学习率
        """
        self.config = load_config('configs/default.yaml')
        # 获取环境动作信息
        self.env_acceleration_range = env.unwrapped.config["action"]['acceleration_range']
        self.env_steering_range = env.unwrapped.config["action"]['steering_range']

        # 获取缓冲区最大数量
        self.buffer_size = self.config.train.buffer_size


        self.advantage = None
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.batch_size = self.config.train.batch_size

        # 初始化函数
        self._setup()
        # 初始化设备
        self.device = self.config.device

        # 初始化神经网络
        self.actor_model = ActorNet(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(),lr = actor_lr)

        self.critic_model = CriticNet(self.state_dim,self.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(),lr=critic_lr)

        self.loss_func = nn.MSELoss()

        self.gamma = 0.99
        self.memory = []

        # 惩罚放大系数
        self.penalty = 1000.0
        self.penalty_amplifier = 1.1
        self.penalty_max = 1000

        # 正定矩阵
        self.Q_matrix = np.diag([0.04, 0.04, 0.01, 0.01, 0.1, 0.02])
        self.R_matrix = np.diag([0.1, 0.005])
        self.M_matrix = np.diag([1,1,0,0,0,0])
        # 严格使用s^ref = [δp, δφ, δv ]状态时候的Q
        self.Q_matrix_ref = np.diag([0.04,0.1,0.01])


    def _setup(self):
        setup_code_environment(self.config)

    # 存储记录
    def store_transition(self, state_info):
        self.memory.append(state_info)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)


    def store_len(self):
        return len(self.memory)

    def select_action(self,state: np.ndarray,grad: bool = False):
        """
        根据当前策略选择动作（用于与环境交互）。
        :param state: 状态数组，shape=[state_dim,]
        :param grad: 是否需要梯度
        :return: (action, log_prob)
        """
        if not grad:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor_model(state_tensor)
            return action.cpu().detach().numpy().flatten()
        else:
            state_tensor = torch.from_numpy(state).squeeze(0).to(self.device).float()
            action = self.actor_model(state_tensor)
            return action

    # 更新critic
    def update_critic(self):
        """
        更新评论家网络ω
        """
        # 1.更新评论家网络
        # 从缓冲区采样
        states,actions = zip(*sample(self.memory, self.batch_size))
        # 转为tensor
        state_x_refs = torch.from_numpy(np.array([[s['state_x_ref'] for s in states]])).squeeze(0).to(self.device).float()
        state_ego = torch.from_numpy(np.array([[s['state_ego'] for s in states]])).squeeze(0).to(self.device).float()
        state_all_np = np.array([[s['state'] for s in states]]).squeeze(0)
        state_all = torch.from_numpy(state_all_np).to(self.device).float()
        Q_matrix_tensor = torch.from_numpy(self.Q_matrix).to(self.device).float()
        R_matrix_tensor = torch.from_numpy(self.R_matrix).to(self.device).float()
        # 跟踪误差
        actions = self.select_action(state_all_np,grad=True)
        tracking_error = ((state_x_refs - state_ego) @ Q_matrix_tensor) * (state_x_refs - state_ego)
        control_energy = (actions @ R_matrix_tensor) * actions
        # 文章中的l(si|t, πθ(si|t))
        l_critic = torch.mean(tracking_error) + torch.mean(control_energy)

        # 文章中的Vw(s0|t)
        v_w = self.critic_model(state_all)
        v_w = torch.mean(v_w)
        loss_critic = self.loss_func(l_critic,v_w)
        #更新ω
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        return loss_critic.detach().item()

    def update_actor(self):
        """
        更新策略网络θ
        """
        # 从缓冲区采样
        states, actions = zip(*sample(self.memory, self.batch_size))
        # 转为tensor
        state_x_refs = torch.from_numpy(np.array([[s['state_x_ref'] for s in states]])).squeeze(0).to(self.device).float()
        state_ego = torch.from_numpy(np.array([[s['state_ego'] for s in states]])).squeeze(0).to(self.device).float()
        state_all_np = np.array([[s['state'] for s in states]]).squeeze(0)
        x_roads = torch.from_numpy(np.array([[s['x_road'] for s in states]])).squeeze(0).to(self.device).float()
        Q_matrix_tensor = torch.from_numpy(self.Q_matrix).to(self.device).float()
        R_matrix_tensor = torch.from_numpy(self.R_matrix).to(self.device).float()
        M_matrix_tensor = torch.from_numpy(self.M_matrix).to(self.device).float()
        # 跟踪误差
        actions = self.select_action(state_all_np, grad=True)
        tracking_error = ((state_x_refs - state_ego) @ Q_matrix_tensor) * (state_x_refs - state_ego)
        control_energy = (actions @ R_matrix_tensor) * actions
        # 文章中的l(si|t, πθ(si|t))
        l_actor = torch.mean(tracking_error) + torch.mean(control_energy)

        # 周车约束
        ge_car = torch.max(torch.tensor(0.0),(state_ego - state_x_refs) @ M_matrix_tensor * (state_ego - state_x_refs))
        # 道路约束
        ge_road = torch.max(torch.tensor(0.0), (state_ego - x_roads) @ M_matrix_tensor * (state_ego - x_roads))
        constraint = self.penalty * torch.mean(ge_car + ge_road)

        loss_actor = l_actor + constraint
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        return loss_actor.detach().item()

    def static_road_plan(self,env):
        """
        静态路径规划
        :param env: 环境
        :return: 规划结果，其中规划的是当前车道和旁边两车道的点状信息
        """
        return get_three_lane_paths(env)

    def clean_mem(self):
        self.memory.clear()

    def update_penalty(self):
        """
        更新惩罚参数
        """
        self.penalty = min(self.penalty * self.penalty_amplifier,self.penalty_max)