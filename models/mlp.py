from torch import nn

class ActorCritic(nn.Module):
    """
    AC/A2C算法的神经网络，使用共享主干参数，来获得policy和value
    """
    def __init__(self, state_dim, hidden_dim, action_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)      # 均值
        self.log_std_head = nn.Linear(hidden_dim, action_dim) # 标准差
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        value = self.value_head(h)
        return mu, log_std, value