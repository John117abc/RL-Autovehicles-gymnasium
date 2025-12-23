import numpy as np

from agents import AgentHighWayContinuous
from custom_env import get_highway_discrete_env
from utils import normalize_Kinematics_obs
from utils import load_config,load_checkpoint,Plotter

# 读取配置文件
config = load_config('../configs/default.yaml')
max_episode = config.train.epochs   # 最大回合数
max_step = config.train.max_step    # 每回合最大步数
checkpoint_dir = config.checkpoints
# 初始化环境
env = get_highway_discrete_env()
episode_rewards = []
rollout_steps = 20
action_dim = 2

# 初始化智能体
agent = AgentHighWayContinuous(state_dim=7*10,action_dim=action_dim, value_output_dim=1,policy_lr=0.0001,value_lr=0.001)

# 读取参数
policy_check_point = load_checkpoint(agent.policy_model,
                                  f'{checkpoint_dir}/20251223/mlp-policy_highway-v0-ca_170743_max_avg_reward=-9169.7_train_step=1000.pth',
                                  agent.policy_optimizer,
                                  agent.device)

policy_check_value = load_checkpoint(agent.value_model,
                                     f'{checkpoint_dir}/20251223/mlp-value_highway-v0-ca_170743_max_avg_reward=-9169.7_train_step=1000.pth',
                                     agent.value_optimizer,
                                     agent.device)

# 画图
plotter = Plotter(history=policy_check_point['history'])
plotter.plot_training_metrics()


# 使用模型进行验证
done = False
step_count = 2000
obs, _ = env.reset()
state = normalize_Kinematics_obs(obs)
while not done and step_count < 5000:  # 加个上限防死循环
    env.render()
    action, log_prob, value = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
