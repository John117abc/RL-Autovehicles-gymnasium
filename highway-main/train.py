import numpy as np
from collections import namedtuple
from agents import AgentOcp
from custom_env import get_highway_discrete_env
from utils import checkpoint, load_config, get_kinematics_state_current,get_kinematics_state_static, get_logger,get_complete_lane_references
from buffer import IDCBuffer

# 初始化日志系统
logger = get_logger()

# 读取配置文件
logger.info('开始读取配置文件')
config = load_config('configs/default.yaml')
max_episode = config.train.epochs
max_step = config.train.max_step
batch_size = config.train.batch_size
buffer_size = config.train.buffer_size
amplifier_c = config.train.amplifier_c
start_count = config.train.start_count
amplifier_m = config.train.amplifier_m

# 初始化环境
logger.info('开始初始化环境')
env = get_highway_discrete_env()
action_dim = 2  # 转向角 + 加速度

# 初始化智能体
logger.info('开始初始化智能体')
obs, _ = env.reset()
all_state = get_kinematics_state_current(obs, env)
state = all_state['state']
agent = AgentOcp(env, state_dim=len(state))
agent.actor_model.train()
agent.critic_model.train()

# 初始化参数
max_avg_reward = float("-inf")
train_step = 0

# 初始化历史记录
history_critic = {
    'episode': [],
    'loss_critic': [],
    'reward':[]
}

history_actor = {
    'episode': [],
    'loss_actor': [],
    'reward':[]
}

# 开始训练
episode_rewards = []
# 论文中的Dynamic Optimal Tracking-Offline Training算法
for episode in range(max_episode):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    constraint_violations = 0.0
    # 从环境中进行采样,选择当前车道(Sampling (from environment))
    # 选择一条路径为参考路径
    path = get_complete_lane_references(env, horizon=1000)[1]
    # 初始化xt,xjt
    state = get_kinematics_state_static(env,obs,path)
    while not done and step_count < max_step:
        env.render()
        # 用当前策略生成动作，并且观察
        action = agent.select_action(state['state'])
        # 保存state ={τ, xt, xj  t, j ∈ I}
        agent.store_transition([state,action])
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # 下一个状态信息
        state = get_kinematics_state_static(env,next_obs, path)
        # 步数+1
        step_count+=1

    # 检查是否可以开始训练
    if agent.store_len() > start_count:
        # 更新 Critic
        loss_critic = agent.update_critic()
        history_critic['loss_critic'].append(loss_critic)
        history_critic['episode'].append(episode + 1)
        history_critic['reward'].append(total_reward)

        # 更新 Actor
        loss = agent.update_actor()
        history_actor['episode'].append(episode + 1)
        history_actor['loss_actor'].append(loss)
        history_actor['reward'].append(total_reward)

        # 更新 ρ (惩罚系数)
        if episode % amplifier_m == 0:
            agent.update_penalty()

        # 打印日志
        log_msg = f"第{episode}回合 | 训练步数 | Critic Loss: {loss_critic:.5f}  | Actor Loss: {loss:.5f} | Penalty: {agent.penalty:.3f}"
        logger.info(log_msg)

    # 可选：保存最佳模型
    if total_reward > max_avg_reward:
        max_avg_reward = total_reward
        # 这里可以加 best model 保存逻辑

# 训练结束
env.close()
logger.info("训练完成！")

# 保存最终模型和历史
metrics = {'max_avg_reward': max_avg_reward, 'train_step': train_step}

extra_info_actor = {'history': history_actor}
extra_info_critic = {'history': history_critic}

# 参数保存路径


# 保存 Actor
checkpoint.save_checkpoint(
    model=agent.actor_model,
    model_name='a-a2c-mlp',
    env_name='highway-v0',
    file_dir=config.checkpoints,
    metrics=metrics,
    optimizer=agent.actor_optimizer,
    extra_info=extra_info_actor
)

# 保存 Critic
checkpoint.save_checkpoint(
    model=agent.critic_model,
    model_name='c-a2c-mlp',
    env_name='highway-v0',
    file_dir=config.checkpoints,
    metrics=metrics,
    optimizer=agent.critic_optimizer,
    extra_info=extra_info_critic
)

logger.info("模型与训练历史已保存。")