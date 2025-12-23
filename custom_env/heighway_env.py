import gymnasium as gym
import highway_env
from utils import load_config_json,load_config
# render_mode="rgb_array",


def get_highway_discrete_env():
    config = load_config_json('../configs/highway-discrete-env-config.yaml')
    return  gym.make("highway-v0", render_mode="rgb_array", config=config)
