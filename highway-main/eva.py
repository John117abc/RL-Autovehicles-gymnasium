
from utils import normalize_Kinematics_obs
from utils import load_checkpoint,Plotter
from custom_env import get_highway_discrete_env
from utils import load_config, get_kinematics_state_current,get_logger,get_complete_lane_references,get_kinematics_state_static
from agents import AgentOcp
# 初始化日志系统
logger = get_logger()

# 读取配置文件
config = load_config('configs/default.yaml')
checkpoint_dir = config.checkpoints
# 初始化环境
env = get_highway_discrete_env()
action_dim = 2

# 初始化智能体
logger.info('开始初始化智能体')
obs, _ = env.reset()
all_state = get_kinematics_state_current(obs, env)
state = all_state['state']
agent = AgentOcp(env, state_dim=len(state))
agent.actor_model.train()
agent.critic_model.train()
# 读取参数
checkpoint_actor = load_checkpoint(agent.actor_model,
                                  f'{checkpoint_dir}/20251231/ac-actor_highway-v0_195500.pth',
                                  agent.actor_optimizer,
                                  agent.device)

checkpoint_critic = load_checkpoint(agent.critic_model,
                                  f'{checkpoint_dir}/20251231/ac-critic_highway-v0_195501.pth',
                                  agent.critic_optimizer,
                                  agent.device)

# 画图
plotter = Plotter(history=checkpoint_critic['history'])
plotter.plot_training_metrics()

agent.actor_model.eval()
agent.critic_model.eval()
# 使用模型进行验证
done = False
step_count = 2000
obs, _ = env.reset()
path = get_complete_lane_references(env, horizon=20)[1]
state = get_kinematics_state_static(env,obs,path)
while not done and step_count < 5000:  # 加个上限防死循环
    env.render()
    path = get_complete_lane_references(env, horizon=20)[1]
    action = agent.select_action(state['state'])
    next_state, reward, terminated, truncated, info = env.step(action)
    state = get_kinematics_state_static(env, next_state, path)
