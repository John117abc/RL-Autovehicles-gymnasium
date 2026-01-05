import numpy as np
from agents import AgentOcp
from custom_env import get_highway_discrete_env
from utils import (
    checkpoint, load_config, get_kinematics_state_current,
    get_kinematics_state_static, get_logger,
    get_complete_lane_references, compute_reward_IDC,load_checkpoint
)


def train_agent(check_point_path = None):
    """主训练函数，封装完整训练流程"""
    logger = get_logger()
    logger.info('开始读取配置文件')
    config = load_config('configs/default.yaml')
    checkpoint_dir = config.checkpoints

    # 配置参数提取
    train_cfg = config.train
    max_episode = train_cfg.epochs
    max_step = train_cfg.max_step
    start_count = train_cfg.start_count
    amplifier_m = train_cfg.amplifier_m

    # 初始化环境
    logger.info('开始初始化环境')
    env = get_highway_discrete_env()

    # 初始化智能体
    logger.info('开始初始化智能体')
    obs, _ = env.reset()
    all_state = get_kinematics_state_current(obs, env)
    state = all_state['state']
    agent = AgentOcp(env, state_dim=len(state))
    agent.actor_model.train()
    agent.critic_model.train()
    if check_point_path is not None:
        # 读取参数
        load_checkpoint(agent.actor_model,
                                           f'{checkpoint_dir}/{check_point_path['actor']}',
                                           agent.actor_optimizer,
                                           agent.device)

        load_checkpoint(agent.critic_model,
                                            f'{checkpoint_dir}/{check_point_path['critic']}',
                                            agent.critic_optimizer,
                                            agent.device)

    # 初始化训练状态
    max_avg_reward = float("-inf")

    # 历史记录
    history_critic = {'episode': [], 'loss_critic': [], 'reward': []}
    history_actor = {'episode': [], 'loss_actor': [], 'reward': []}

    # 开始训练循环
    for episode in range(max_episode):
        obs, _ = env.reset()
        done = False
        step_count = 0
        path = get_complete_lane_references(env, horizon=20)[1]
        state = get_kinematics_state_static(env, obs, path)

        loss_critic_one_eps = []
        loss_actor_one_eps = []
        reward_one_eps = []

        while not done and step_count < max_step:
            env.render()
            path = get_complete_lane_references(env, horizon=20)[1]

            action = agent.select_action(state['state'])
            agent.store_transition([state, action])

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = get_kinematics_state_static(env, next_obs, path)
            step_count += 1

            reward = compute_reward_IDC(env, action)
            reward_one_eps.append(reward)

            if agent.store_len() > start_count:
                loss_critic = agent.update_critic()
                loss_actor = agent.update_actor()

                loss_critic_one_eps.append(loss_critic)
                loss_actor_one_eps.append(loss_actor)

                history_critic['loss_critic'].append(np.mean(loss_critic_one_eps))
                history_critic['episode'].append(episode + 1)
                history_critic['reward'].append(np.mean(reward_one_eps))

                history_actor['episode'].append(episode + 1)
                history_actor['loss_actor'].append(np.mean(loss_actor_one_eps))
                history_actor['reward'].append(np.mean(reward_one_eps))

        # 日志输出
        if loss_critic_one_eps:  # 避免空列表
            log_msg = (
                f"第{episode}回合 | 训练步数{step_count} | "
                f"Critic Loss: {np.mean(loss_critic_one_eps):.5f} | "
                f"Actor Loss: {np.mean(loss_actor_one_eps):.5f} | "
                f"Penalty: {agent.penalty:.3f} | "
                f"Reward: {np.mean(reward_one_eps):.3f}"
            )
            logger.info(log_msg)

        # 定期更新惩罚系数并清空记忆
        if episode % amplifier_m == 0:
            agent.update_penalty()
            # agent.clean_mem()

        # 保存最佳模型
        current_avg_reward = np.mean(reward_one_eps) if reward_one_eps else -np.inf
        if (reward_one_eps and
                current_avg_reward > max_avg_reward and
                episode / max_episode > 0.1):
            max_avg_reward = current_avg_reward
            metrics = {'max_avg_reward': max_avg_reward, 'episode': episode + 1}
            extra_info_actor = {'history': history_actor}
            extra_info_critic = {'history': history_critic}

            _save_model(
                agent=agent,
                config=config,
                metrics=metrics,
                extra_info_actor=extra_info_actor,
                extra_info_critic=extra_info_critic,
                logger=logger
            )

    # 最终保存（无论是否最佳）
    extra_info_actor = {'history': history_actor}
    extra_info_critic = {'history': history_critic}
    _save_model(
        agent=agent,
        config=config,
        metrics=None,  # 不传 metrics 表示非最佳模型
        extra_info_actor=extra_info_actor,
        extra_info_critic=extra_info_critic,
        logger=logger
    )

    env.close()
    logger.info("训练完成！")


def _save_model(agent, config, metrics, extra_info_actor, extra_info_critic, logger):
    common_args = {
        'env_name': 'highway-v0',
        'file_dir': config.checkpoints,
    }

    checkpoint.save_checkpoint(
        model=agent.actor_model,
        model_name='ac-actor',
        optimizer=agent.actor_optimizer,
        extra_info=extra_info_actor,
        metrics=metrics,
        **common_args
    )

    checkpoint.save_checkpoint(
        model=agent.critic_model,
        model_name='ac-critic',
        optimizer=agent.critic_optimizer,
        extra_info=extra_info_critic,
        metrics=metrics,
        **common_args
    )

    if metrics is not None:
        logger.info("最佳模型与训练历史已保存。")
    else:
        logger.info("最终模型与训练历史已保存。")



if __name__ == "__main__":
    checkpoint_pth = {'actor':'20260105/ac-actor_highway-v0_213949_max_avg_reward=0.9019502401351929_episode=1009.pth',
                      'critic':'20260105/ac-critic_highway-v0_213949_max_avg_reward=0.9019502401351929_episode=1009.pth'}
    # checkpoint_pth = None
    train_agent(check_point_path = checkpoint_pth)