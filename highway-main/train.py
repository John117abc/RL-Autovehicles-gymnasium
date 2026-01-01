import numpy as np
from collections import namedtuple
from agents import AgentOcp
from custom_env import get_highway_discrete_env
from utils import checkpoint, load_config, get_kinematics_state_current,get_kinematics_state_static, get_logger,get_complete_lane_references,compute_reward_IDC
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
    path = get_complete_lane_references(env, horizon=20)[1]
    # 初始化xt,xjt
    state = get_kinematics_state_static(env,obs,path)
    loss_critic_one_eps = []
    loss_actor_one_eps = []
    reward_one_eps = []
    while not done and step_count < max_step:
        env.render()
        path = get_complete_lane_references(env, horizon=20)[1]
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
        # 计算奖励
        reward = compute_reward_IDC(env, action)
        # 检查是否可以开始训练
        if agent.store_len() > start_count:
            # 更新 Critic
            loss_critic = agent.update_critic()
            loss_critic_one_eps.append(loss_critic)

            # 更新 Actor
            loss_actor = agent.update_actor()
            loss_actor_one_eps.append(loss_actor)

            reward_one_eps.append(reward)

            # 存储模型参数
            history_critic['loss_critic'].append(np.mean(loss_critic_one_eps))
            history_critic['episode'].append(episode + 1)
            history_critic['reward'].append(np.mean(reward_one_eps))

            history_actor['episode'].append(episode + 1)
            history_actor['loss_actor'].append(np.mean(loss_actor_one_eps))
            history_actor['reward'].append(np.mean(reward_one_eps))

    # 打印日志
    if len(loss_critic_one_eps) > 0 and len(loss_critic_one_eps) > 0:
        log_msg = (f"第{episode}回合 | 训练步数{step_count} | Critic Loss: {np.mean(loss_critic_one_eps):.5f}  "
                   f"| Actor Loss: {np.mean(loss_actor_one_eps):.5f} | Penalty: {agent.penalty:.3f}  |  Reward:{np.mean(reward_one_eps):.3f}")
        logger.info(log_msg)

    # 更新 ρ (惩罚系数)
    if episode % amplifier_m == 0:
        agent.update_penalty()
        agent.clean_mem()

    # 保存最佳模型
    if len(reward_one_eps) > 0 and np.mean(reward_one_eps) > max_avg_reward and episode/max_episode > 0.1:
        max_avg_reward = np.mean(reward_one_eps)
        # 保存模型
        metrics = {'max_avg_reward': max_avg_reward, 'episode': episode+1}

        extra_info_actor = {'history': history_actor}
        extra_info_critic = {'history': history_critic}

        # 保存 Actor
        checkpoint.save_checkpoint(
            model=agent.actor_model,
            model_name='ac-actor',
            env_name='highway-v0',
            file_dir=config.checkpoints,
            metrics=metrics,
            optimizer=agent.actor_optimizer,
            extra_info=extra_info_actor
        )

        # 保存 Critic
        checkpoint.save_checkpoint(
            model=agent.critic_model,
            model_name='ac-critic',
            env_name='highway-v0',
            file_dir=config.checkpoints,
            metrics=metrics,
            optimizer=agent.critic_optimizer,
            extra_info=extra_info_critic
        )
        logger.info("模型与训练历史已保存。")


# max_avg_reward = np.mean(reward_one_eps)
# # 保存模型
# metrics = {'max_avg_reward': max_avg_reward, 'episode': episode+1}

extra_info_actor = {'history': history_actor}
extra_info_critic = {'history': history_critic}

# 保存 Actor
checkpoint.save_checkpoint(
    model=agent.actor_model,
    model_name='ac-actor',
    env_name='highway-v0',
    file_dir=config.checkpoints,
    # metrics=metrics,
    optimizer=agent.actor_optimizer,
    extra_info=extra_info_actor
)

# 保存 Critic
checkpoint.save_checkpoint(
    model=agent.critic_model,
    model_name='ac-critic',
    env_name='highway-v0',
    file_dir=config.checkpoints,
    # metrics=metrics,
    optimizer=agent.critic_optimizer,
    extra_info=extra_info_critic
)
logger.info("模型与训练历史已保存。")

# 训练结束
env.close()
logger.info("训练完成！")