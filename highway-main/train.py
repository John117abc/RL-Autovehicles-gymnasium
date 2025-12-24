import numpy as np

from agents import AgentHighWayContinuous
from custom_env import get_highway_discrete_env
from utils import checkpoint,load_config,get_kinematics_state

# 读取配置文件
config = load_config('../configs/default.yaml')
max_episode = config.train.epochs   # 最大回合数
max_step = config.train.max_step    # 每回合最大步数

# 初始化环境
env = get_highway_discrete_env()
episode_rewards = []
action_dim = 2

# 使用策略梯度法进行训练
agent = AgentHighWayContinuous(state_dim=7*10,action_dim=action_dim,env=env,lr=0.0001)
agent.model.train()
# 初始化参数
max_avg_reward = float("-inf")
train_step = 0


# 初始化历史记录
history = {
    'episode': [],
    'total_reward': [],
    'avg_loss':[],
    'avg_return':[]
}

# 开始训练
for episode in range(max_episode):
    obs, _ = env.reset()
    # 静态路径信息
    state = get_kinematics_state(obs)

    done = False
    total_reward = 0
    step_count = 0
    actions_taken = []
    update_count = 0
    loss = 0
    returns = 0
    # 每一个回合的训练
    while not done and step_count < max_step:  # 加个上限防死循环
        env.render()
        action, value = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # if len(agent.memory) == rollout_steps or done:


    # 打印终止原因
    if done:
        print(f"回合结束于第 {step_count} 步:")
        print(f"  terminated={terminated}, truncated={truncated}")
        print(f"  终止信息={info}")
        print("Actions:", actions_taken[:20])

    # 打印进度
    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        max_avg_reward = max(avg_reward,max_avg_reward)
        train_step += step_count
        print(f"回合 {episode}, 平均奖励: {avg_reward:.2f}")

env.close()
# 存储训练参数

print('开始存储训练参数')
# 存储策略参数
metrics = {'max_avg_reward':max_avg_reward,'train_step':train_step}

# 存储历史数据
extra_info = {'history':history}
checkpoint.save_checkpoint(model= agent.model,
                           model_name='a2c-mlp',
                           env_name='highway-v0',
                           file_dir=config.checkpoints,
                           metrics = metrics,
                           optimizer=agent.optimizer,
                           epoch = max_episode,
                           extra_info = extra_info)