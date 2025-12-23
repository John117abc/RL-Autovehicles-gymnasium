import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from models import ActorCritic
from utils import setup_code_environment,load_config
# # 设备选择
# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"使用设备: {device}")

class AgentHighWayContinuous:
    def __init__(self,state_dim,env,
                 hidden_dim = 256,action_dim = 5,lr = 0.01):
        """
        智能体
        :param state_dim: 状态空间维度
        :param env: 环境
        :param hidden_dim: 隐藏层维度
        :param action_dim: 动作维度
        :param lr: 学习率
        """
        self.config = load_config('../configs/default.yaml')
        # 获取环境动作信息
        self.env_acceleration_range = env.unwrapped.config["action"]['acceleration_range']
        self.env_steering_range = env.unwrapped.config["action"]['steering_range']

        self.advantage = None
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr

        # 初始化函数
        self._setup()
        # 初始化设备
        self.device = self.config.device

        # 初始化神经网络
        self.model = ActorCritic(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.lr)

        self.gamma = 0.99
        self.memory = []

    def _setup(self):
        setup_code_environment(self.config)

    # 存储记录
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.memory.append((state,action,reward,log_prob,value,done))

    def calculate_returns(self):
        returns = []
        G = 0.0

        # 如果最后一步不是终止状态，用 critic 估计 V(last_state)
        if not self.memory[-1][5]:  # 第6个元素是 done
            last_state = torch.FloatTensor(self.memory[-1][0]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, G = self.model(last_state)
                G = G.item()

        # 从后往前计算
        for i in reversed(range(len(self.memory))):
            reward = self.memory[i][2]
            G = reward + self.gamma * G
            returns.insert(0, G)

        return returns

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """
        根据当前策略选择动作（用于与环境交互）。
        :param state: 状态数组，shape=[state_dim,]
        :return: (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 前向传播策略网络
        mu, log_std, value = self.model(state_tensor)  # [1, 2*action_dim]
        std = torch.exp(log_std.clamp(-20, 2))  # 防止爆炸

        # 构建高斯分布并采样
        dist = Normal(mu, std)
        raw_action = dist.sample()
        # 动作映射到tan
        action = torch.tanh(raw_action)
        # 减去 tanh 的 Jacobian 行列式对数（每个维度独立）
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)     # 变量变换公式
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 把action映射到真实的转向角和加速度上
        action = self.denormalize_action(action,self.env_acceleration_range,self.env_steering_range)
        return (
            action.cpu().detach().numpy().flatten(),
            log_prob.cpu().detach().numpy().flatten(),
            value.cpu().detach().numpy().flatten()
        )

    # 使用A2C算法更新策略
    def update(self):
        """
        使用收集到的整条轨迹更新共享主干的 Actor-Critic 网络（on-policy A2C）。
        假设动作已通过 tanh + 线性缩放映射到物理空间：
            acc ∈ [-5.0, 5.0]
            steer ∈ [-0.785, 0.785]
        """
        if len(self.memory) == 0:
            return {}

        # 解包经验
        states, actions, rewards, old_log_probs, values, dones = zip(*self.memory)

        returns = torch.FloatTensor(self.calculate_returns()).to(self.device)
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        actions = torch.from_numpy(np.array(actions)).to(self.device).float()  # [T, 2]
        values = torch.from_numpy(np.array(values)).to(self.device).float()  # [T]

        mu, log_std, current_values = self.model(states)
        current_values = current_values.squeeze(-1)

        # 反推 tanh_action 和 raw_action
        tanh_action = self.normalize_action(actions,self.env_acceleration_range,self.env_steering_range)

        eps = 1e-6
        clipped_tanh = torch.clamp(tanh_action, -1 + eps, 1 - eps)
        raw_action = torch.atanh(clipped_tanh)

        # 计算修正后的 log_prob
        std = torch.exp(log_std.clamp(-20, 2))
        dist = Normal(mu, std)

        log_prob_raw = dist.log_prob(raw_action)
        log_prob_correction = torch.log(1 - clipped_tanh.pow(2) + eps)
        new_log_probs = (log_prob_raw - log_prob_correction).sum(dim=-1, keepdim=True)

        # 计算损失
        value_loss = nn.MSELoss()(current_values, returns)

        advantages = (returns - values).detach().unsqueeze(-1)
        policy_loss = -(new_log_probs * advantages).mean()

        total_loss = policy_loss + 0.5 * value_loss  # 加系数平衡

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.memory.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': (policy_loss + value_loss).item(),
            'avg_return': returns.mean().item()
        }

    def clean_mem(self):
        self.memory.clear()

    def denormalize_action(self,action_norm, acceleration_range, steering_range):
        """
        将 [-1, 1] 范围内的归一化动作（PyTorch Tensor）映射到实际的 steering 和 acceleration 范围。

        参数:
            action_norm (torch.Tensor):
                - shape: [..., 2]，最后一维为 [steering_norm, acceleration_norm]，值 ∈ [-1, 1]
            acceleration_range (tuple/list): [min_acc, max_acc]，例如 [-5.0, 5.0]
            steering_range (tuple/list): [min_steering, max_steering]，例如 [-0.785, 0.785]

        返回:
            torch.Tensor: [..., 2]，最后一维为 [steering_real, acceleration_real]
        """
        # 确保输入是 Tensor
        if not isinstance(action_norm, torch.Tensor):
            action_norm = torch.as_tensor(action_norm, dtype=torch.float32)

        action_norm = action_norm.detach()

        # 分离 steering 和 acceleration（保留所有前导维度）
        s_norm = action_norm[..., 0]  # [...]
        a_norm = action_norm[..., 1]  # [...]

        # 转换范围为 Tensor（支持 GPU / 梯度无关）
        steer_min, steer_max = float(steering_range[0]), float(steering_range[1])
        acc_min, acc_max = float(acceleration_range[0]), float(acceleration_range[1])

        # 线性映射：x = x_min + (x_norm + 1) * (x_max - x_min) / 2
        steering_real = steer_min + (s_norm + 1.0) * (steer_max - steer_min) / 2.0
        acceleration_real = acc_min + (a_norm + 1.0) * (acc_max - acc_min) / 2.0

        # 合并回 [..., 2]
        return torch.stack([steering_real, acceleration_real], dim=-1)


    def normalize_action(self,action_real, acceleration_range, steering_range):
        """
        将真实动作（steering, acceleration）线性映射回 [-1, 1] 的归一化空间。

        参数:
            action_real (torch.Tensor):
                - shape: [..., 2]，最后一维为 [steering_real, acceleration_real]
            acceleration_range (tuple/list): [min_acc, max_acc]，例如 [-5.0, 5.0]
            steering_range (tuple/list): [min_steering, max_steering]，例如 [-0.785, 0.785]

        返回:
            torch.Tensor: [..., 2]，最后一维为 [steering_norm, acceleration_norm] ∈ [-1, 1]
        """
        if not isinstance(action_real, torch.Tensor):
            action_real = torch.as_tensor(action_real, dtype=torch.float32)

        action_real = action_real.detach()

        a_real = action_real[..., 0]  # acceleration
        s_real = action_real[..., 1]  # steering

        acc_min, acc_max = float(acceleration_range[0]), float(acceleration_range[1])
        steer_min, steer_max = float(steering_range[0]), float(steering_range[1])

        acceleration_norm = 2.0 * (a_real - acc_min) / (acc_max - acc_min) - 1.0
        steering_norm = 2.0 * (s_real - steer_min) / (steer_max - steer_min) - 1.0

        return torch.stack([acceleration_norm, steering_norm], dim=-1)