import numpy as np
from collections import namedtuple
from agents import AgentOcp
from custom_env import get_highway_discrete_env
from utils import checkpoint, load_config, get_kinematics_state, get_logger
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
min_start_train = config.train.min_start_train
actor_interval = config.train.actor_interval

# 初始化环境
logger.info('开始初始化环境')
env = get_highway_discrete_env()
action_dim = 2  # 转向角 + 加速度

# 初始化智能体
logger.info('开始初始化智能体')
obs, _ = env.reset()
all_state = get_kinematics_state(obs, env)
state = all_state['state']
agent = AgentOcp(env, state_dim=len(state))
agent.actor_model.train()
agent.critic_model.train()

# 初始化缓冲区
logger.info('开始初始化缓冲区')
buffer_manage = IDCBuffer(buffer_size)

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

StateInfo = namedtuple('StateInfo', ['state_ego', 'state_other', 'state_s_ref', 'state_x_ref'])

# 开始训练
episode_rewards = []
for episode in range(max_episode):
    obs, _ = env.reset()
    state = get_kinematics_state(obs, env)
    done = False
    total_reward = 0.0
    step_count = 0
    episode_buffer = []

    # 每一个回合的交互
    while not done and step_count < max_step:
        env.render()
        action, value = agent.select_action(state['state'])
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 构建当前状态快照
        state_info = StateInfo(
            state['state_ego'],
            state['state_other'],
            state['state_s_ref'],
            state['state_x_ref']
        )

        # 获取下一个状态
        next_state = get_kinematics_state(next_obs, env)

        # 存储经验（注意：value 是当前状态的 critic 估计）
        experience = state_info, action, reward, value, next_state, done, info
        episode_buffer.append(experience)

        total_reward += reward
        state = next_state
        step_count += 1

        if done:
            logger.info(f"回合数 {episode + 1} 步数结束于 {step_count}: "
                        f"terminated={terminated}, truncated={truncated}, 最终获得奖励={reward:.3f}")

    # 回合结束，记录总奖励
    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
    logger.info(f"回合进度： {episode + 1}/{max_episode} | 总奖励: {total_reward:.3f} | 平均回报: {avg_reward:.3f}")

    # 存入缓冲区
    buffer_manage.add_trajectory(episode_buffer)

    # 检查是否可以开始训练
    if buffer_manage.size_trajectory() >= min_start_train:
        # 更新 Critic（每轮都更新）
        loss_critic = agent.update_critic(buffer_manage)
        # 确保 loss 是标量（如果是 tensor，取 item）
        if isinstance(loss_critic, (float, int)):
            critic_loss_val = loss_critic
        else:
            critic_loss_val = loss_critic.item()

        history_critic['episode'].append(episode + 1)
        history_critic['loss_critic'].append(critic_loss_val)
        history_critic['reward'].append(total_reward)

        # 更新 Actor（按间隔）
        loss_actor = None
        if episode % actor_interval == 0:
            loss_actor = agent.update_actor(buffer_manage)
            if isinstance(loss_actor, (float, int)):
                actor_loss_val = loss_actor
            else:
                actor_loss_val = loss_actor.item()

            history_actor['episode'].append(episode + 1)
            history_actor['loss_actor'].append(actor_loss_val)
            history_actor['reward'].append(total_reward)
        else:
            actor_loss_val = None

        # 打印训练损失（可选：只在有 actor 更新时打印）
        log_msg = f"训练步数 | Critic Loss: {critic_loss_val:.5f}"
        if actor_loss_val is not None:
            log_msg += f" | Actor Loss: {actor_loss_val:.5f}"
        logger.info(log_msg)

    # 可选：保存最佳模型
    if total_reward > max_avg_reward:
        max_avg_reward = total_reward
        # 这里可以加 best model 保存逻辑（略）

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