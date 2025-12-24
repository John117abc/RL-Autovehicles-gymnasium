import gymnasium as gym
import highway_env
from utils import load_config_json
from .utils.custom_observation import YawRateObservationWrapper
# render_mode="rgb_array",

def get_highway_discrete_env():
    config = load_config_json('/home/jiangchengxuan/PycharmProjects/RL-Autovehicles-gymnasium/configs/highway-discrete-env-config.yaml')
    env = gym.make("highway-v0", render_mode="rgb_array", config=config)
    # 把角速度添加到观察里面
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "heading", "cos_h", "sin_h"],
            "vehicles_count": 5,
            "absolute": False,
            "order": "sorted"
        }
    })
    env = YawRateObservationWrapper(env)
    return env