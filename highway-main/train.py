import numpy as np

from collections import namedtuple
from agents import AgentOcp
from custom_env import get_highway_discrete_env
from utils import checkpoint,load_config,get_kinematics_state,get_logger
from buffer import IDCBuffer

# 初始化日志系统
logger = get_logger()

# 读取配置文件
logger.info('开始读取配置文件')
config = load_config('../configs/default.yaml')
max_episode = config.train.epochs   # 最大回合数
max_step = config.train.max_step    # 每回合最大步数
batch_size = config.train.batch_size    # 每批次训练大小
buffer_size = config.train.buffer_size  # 经验回放缓冲区大小
min_start_train = config.train.min_start_train # 最小训练启动样本量
actor_interval = config.train.actor_interval    # 对actor的更新，每x次更新一次

# 初始化环境
logger.info('开始初始化环境')
env = get_highway_discrete_env()
episode_rewards = []
# 因为是连续动作，动作输出为转向角和加速度，所以dim = 2
action_dim = 2

# 初始化智能体
logger.info('开始初始化智能体')
obs, _ = env.reset()
all_state = get_kinematics_state(obs,env)
state = all_state['state']
agent = AgentOcp(env,state_dim=len(state))
agent.actor_model.train()
agent.critic_model.train()

# 初始化缓冲区
logger.info('开始初始化缓冲区')
buffer_manage = IDCBuffer(buffer_size)

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

StateInfo = namedtuple('StateInfo', ['state_ego', 'state_other', 'state_ref'])
# 开始训练
for episode in range(max_episode):
    obs, _ = env.reset()
    # 获取状态信息
    state = get_kinematics_state(obs,env)
    done = False
    total_reward = 0
    step_count = 0
    actions_taken = []
    update_count = 0
    episode_buffer = []
    # 每一个回合的训练
    while not done and step_count < max_step:  # 加个上限防死循环
        env.render()
        action, value = agent.select_action(state['state'])
        next_state, reward, terminated, truncated, info = env.step(action)
        state_info = StateInfo(
            state['state_ego'],
            state['state_other'],
            state['state_ref']
        )
        # 暂存单条轨迹储经验
        state = get_kinematics_state(next_state,env)
        experience = state_info, action, reward, value, state,done, info
        episode_buffer.append(experience)
        step_count+=1

    logger.info(f'第{episode+1}次回合结束')
    # 存储经验
    buffer_manage.add_trajectory(episode_buffer)
    # 如果经验缓冲区的数据够训练，则开始采样训练
    if buffer_manage.size_trajectory() >= min_start_train:
        agent.update_critic(buffer_manage)
        if episode % actor_interval ==0:
            # 进行actor更新
            agent.update_actor(buffer_manage)

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