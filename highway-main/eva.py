import numpy as np

from agents import AgentHighWayContinuous
from custom_env import get_highway_discrete_env
from utils import normalize_Kinematics_obs
from utils import load_config,load_checkpoint,Plotter

# 读取配置文件
config = load_config('configs/default.yaml')
checkpoint_dir = config.checkpoints
# 初始化环境
env = get_highway_discrete_env()
action_dim = 2

# 初始化智能体
agent = AgentHighWayContinuous(state_dim=7*10,action_dim=action_dim,env=env,lr=0.0001)
# 读取参数
checkpoint_info = load_checkpoint(agent.model,
                                  f'{checkpoint_dir}/20251224/a2c-mlp_highway-v0_120412_max_avg_reward=-483.1_train_step=18456.pth',
                                  agent.optimizer,
                                  agent.device)

# 画图
plotter = Plotter(history=checkpoint_info['history'])
plotter.plot_training_metrics()

agent.model.eval()
# 使用模型进行验证
done = False
step_count = 2000
obs, _ = env.reset()
state = normalize_Kinematics_obs(obs)
while not done and step_count < 5000:  # 加个上限防死循环
    env.render()
    action, log_prob, value = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
