import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from models import ActorNet,CriticNet
from utils import setup_code_environment,load_config,normalize_action,get_three_lane_paths
from buffer import IDCBuffer
# # 设备选择
# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"使用设备: {device}")

class AgentOcp:
    def __init__(self,env,state_dim,
                 hidden_dim = 256,action_dim = 2,actor_lr = 0.9,critic_lr = 0.999):
        """
        智能体
        :param state_dim: 状态空间维度
        :param env: 环境
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作维度
        :param actor_lr: 策略学习率
        :param critic_lr: 评论家学习率
        """
        self.config = load_config('../configs/default.yaml')
        # 获取环境动作信息
        self.env_acceleration_range = env.unwrapped.config["action"]['acceleration_range']
        self.env_steering_range = env.unwrapped.config["action"]['steering_range']

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
        self.actor_model = ActorNet(self.state_dim,self.hidden_dim,2 * self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(),lr = actor_lr)

        self.critic_model = CriticNet(self.state_dim,self.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(),lr=critic_lr)

        self.gamma = 0.99
        self.memory = []

        # 惩罚放大系数
        self.penalty = 1.0
        self.penalty_amplifier = 1.1

        # 正定矩阵
        self.Q_matrix = [0.04, 0.04, 0.01, 0.1]
        self.R_matrix = [0.1, 0.005]
        self.M_matrix = [1,1,0,0,0,0]


    def _setup(self):
        setup_code_environment(self.config)

    # 存储记录
    def store_transition(self, state_info):
        self.memory.append(state_info)

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """
        根据当前策略选择动作（用于与环境交互）。
        :param state: 状态数组，shape=[state_dim,]
        :return: (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 前向传播策略网络
        mu, log_std = self.actor_model(state_tensor)  # [1, 2*action_dim]
        std = torch.exp(log_std.clamp(-20, 2))  # 防止爆炸

        # 构建高斯分布并采样
        # 对转向角采样
        dist = Normal(mu, std)
        raw_action = dist.sample()
        # 动作映射到tan
        action = torch.tanh(raw_action)
        value = self.critic_model(state_tensor)
        return (
            action.cpu().detach().numpy().flatten(),
            value.cpu().detach().numpy().flatten()
        )

    def calculate_j_critic(self, x_others, xs, value_t, actions):
        """
        计算Jcritic   x_refs,xs,actions的len()必须相等
        :param x_others: 周车状态信息
        :param xs: 自车信息
        :param value_t: t时刻critic网络计算的value
        :param actions: 每个时刻的动作
        :return: 计算值
        """
        length = len(actions)
        total_cost = 0.0
        for i in range(length):
            dx = x_others[i] - xs[i]
            cost_state = dx.T @ self.Q_matrix @ dx
            cost_action = actions[i].T @ self.R_matrix @ actions[i]

            l = cost_state + cost_action
            total_cost += (l - value_t) **2

        return total_cost

    def calculate_j_p(self, x_others, xs,x_refs, actions):

        length = len(actions)
        total_l = 0.0
        for i in range(length):
            dx = x_others[i] - xs[i]
            cost_state = dx.T @ self.Q_matrix @ dx
            cost_action = actions[i].T @ self.R_matrix @ actions[i]

            total_l += cost_state + cost_action

        total_ge = 0.0
        for i in range(length):
            # 因为目前环境没有红绿灯约束，所以少一些约束
            ge_other = self.penalty * max(-((xs[i] - x_others[i]).T @ self.M_matrix @ (xs[i] - x_others[i])),0)
            ge_ref = self.penalty * max(-((xs[i] - x_refs[i]).T @ self.M_matrix @ (xs[i] - x_refs[i])),0)
            # 目前没有红绿灯约束
            total_l += (ge_other + ge_ref)**2

        return  total_l + total_ge

    # 使用A2C算法更新策略
    def update_critic(self,buffer:IDCBuffer):
        """
        使用收集到的整条轨迹更新共享主干的 Actor-Critic 网络（on-policy A2C）。
        假设动作已通过 tanh + 线性缩放映射到物理空间：
            acc ∈ [-5.0, 5.0]
            steer ∈ [-0.785, 0.785]
        """
        if not buffer.size() <=0:
            return {}

        # 从缓冲区采样
        states, actions, rewards, values, next_states, dones, infos = zip(*buffer.sample_batch(self.batch_size))
        # 计算J_critic
        j_critic = self.calculate_j_critic(states['state_other'],states['state_ego'],values[0],actions)
        j_critic = torch.from_numpy(j_critic).to(self.device).float()
        self.critic_model.zero_grad()
        j_critic.backward()
        self.policy_optimizer.step()

        # 计算J_p
        j_p = self.calculate_j_p(states['state_other'], states['state_ego'],states['state_ref'], actions)


        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': (policy_loss + value_loss).item(),
            'avg_return': returns.mean().item()
        }

    def static_road_plan(self,env):
        """
        静态路径规划
        :param env: 环境
        :return: 规划结果，其中规划的是当前车道和旁边两车道的点状信息
        """
        return get_three_lane_paths(env)

    def clean_mem(self):
        self.memory.clear()