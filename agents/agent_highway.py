import torch
import numpy as np
from torch import nn
from torch.distributions import Normal
from models import MlpNet
from utils import setup_code_environment,load_config
# # 设备选择
# device = (
#     "cuda" if torch.cuda.is_available()
#     else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"使用设备: {device}")

class AgentHighWayContinuous:
    def __init__(self,value_output_dim,state_dim,
                 hidden_dim = 256,action_dim = 5,policy_lr = 0.01,value_lr = 0.01):
        self.config = load_config('../configs/default.yaml')

        self.advantage = None
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.policy_lr = policy_lr

        self.value_input_dim = state_dim
        self.value_output_dim = value_output_dim
        self.value_lr = value_lr

        # 初始化函数
        self._setup()
        # 初始化设备
        self.device = self.config.device

        # 初始化神经网络
        self.policy_model = MlpNet(self.state_dim,self.hidden_dim,2 * self.action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(),lr = self.policy_lr)

        self.value_model = MlpNet(self.value_input_dim,self.hidden_dim,self.value_output_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(),lr = self.value_lr)

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
                G = self.value_model(last_state).item()

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
        policy_out = self.policy_model(state_tensor)  # [1, 2*action_dim]
        mu = policy_out[:, :self.action_dim]  # 均值
        log_std = policy_out[:, self.action_dim:]  # 对数标准差
        std = torch.exp(log_std.clamp(-20, 2))  # 防止数值爆炸

        # 构建高斯分布并采样
        dist = Normal(mu, std)
        action = dist.sample()
        value = self.value_model(state_tensor)
        log_prob = dist.log_prob(action)    # 计算刚刚采样出的 action 在该分布下的对数概率
        return (
            action.cpu().detach().numpy().flatten(), log_prob.cpu().detach().numpy().flatten(), value.cpu().detach().numpy().flatten()
        )

    # 使用A2C算法更新策略
    def update(self):
        """
        使用收集到的整条轨迹更新策略和价值网络（on-policy）。
        :return: 训练指标字典
        """
        if len(self.memory) == 0:
            return {}

        # 解包内存
        states,actions,_,_,values,_ = zip(*self.memory)
        returns = torch.FloatTensor(self.calculate_returns()).to(self.device).float()
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        actions = torch.from_numpy(np.array(actions)).to(self.device).float()
        values = torch.from_numpy(np.array(values)).to(self.device).float()

        # 更新价值网络
        current_values = self.value_model(states).squeeze(-1)
        value_loss = nn.MSELoss()(current_values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 更新策略网络
        policy_out = self.policy_model(states)
        mu = policy_out[:, :self.action_dim]
        log_std = policy_out[:, self.action_dim:]
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        advantages = (returns - values).detach()  # A(s,a) = G_t - V(s)
        policy_loss = -(new_log_probs * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 清空内存
        self.memory.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_return': returns.mean().item()
        }

    def clean_mem(self):
        self.memory.clear()


class AgentHighWayDiscrete:
    def __init__(self,value_output_dim,state_dim,
                 hidden_dim = 256,action_dim = 5,policy_lr = 0.01,value_lr = 0.01):
        self.advantage = None
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.policy_lr = policy_lr

        self.value_input_dim = state_dim
        self.value_output_dim = value_output_dim
        self.value_lr = value_lr

        # 初始化神经网络
        self.policy_model = MlpNet(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(),lr = self.policy_lr)

        self.value_model = MlpNet(self.value_input_dim,self.hidden_dim,self.value_output_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(),lr = self.value_lr)

        self.gamma = 0.99
        self.memory = []

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
                G = self.value_model(last_state).item()

        # 从后往前计算
        for i in reversed(range(len(self.memory))):
            reward = self.memory[i][2]
            G = reward + self.gamma * G
            returns.insert(0, G)

        return returns

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_model(state)
            value = self.value_model(state)
        dist = torch.distributions.Categorical(logits=logits)   # 直接传 logits，如果 logits = [0.1, 0.7, 0.2]，那么 dist 就是一个“以10%概率选0、70%选1、20%选2”的分布。
        action = dist.sample()   # 从 dist 中随机采样一个动作。
        log_prob = dist.log_prob(action)    # 计算刚刚采样出的 action 在该分布下的对数概率
        return action.item(), log_prob.item(), value.item()

    # 使用A2C算法更新策略
    def update_policy(self):
        if not self.memory:
            return
        states,actions,rewards,log_probs,values,dones = zip(*self.memory)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        states = torch.from_numpy(np.array(states)).to(self.device).float()
        actions = torch.tensor(actions,dtype=torch.long).to(self.device)
        # 重新计算当前策略下的 log_prob
        logits = self.policy_model(states)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions).to(self.device)

        # 计算优势函数
        returns = self.calculate_returns()
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values

        loss_func = -(current_log_probs * advantages.detach()).mean()

        self.policy_optimizer.zero_grad()
        loss_func.backward()
        self.policy_optimizer.step()

    def update_value(self):
        if not self.memory:
            return

        states,actions,rewards,log_probs,values,dones = zip(*self.memory)
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        # 重新计算values
        current_values = self.value_model(states).squeeze(-1)

        returns = self.calculate_returns()  # 应该返回 list of floats
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        loss_func = nn.MSELoss()(current_values, returns)

        self.value_optimizer.zero_grad()
        loss_func.backward()
        self.value_optimizer.step()

    def clean_mem(self):
        self.memory.clear()