import gymnasium as gym
import numpy as np

class YawRateObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, vehicles_count=5):
        super().__init__(env)
        self.vehicles_count = vehicles_count
        self.prev_headings = None

        # 获取时间步长
        config = env.unwrapped.config
        sim_freq = config.get("simulation_frequency", 15)
        policy_freq = config.get("policy_frequency", 1)
        self.dt = 1.0 / (sim_freq / policy_freq)

        # 假设原始观测包含 'heading'，我们需要知道它的索引
        # 默认 Kinematics features: ['presence', 'x', 'y', 'vx', 'vy', 'heading', 'cos_h', 'sin_h']
        # 但我们先检查实际特征
        self.heading_index = None
        self.presence_index = None

        # 修改 observation space: 原始 + 1 (yaw_rate)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[0], old_shape[1] + 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32
        )

    def observation(self, obs):
        # 第一次运行时确定 heading 和 presence 的位置
        if self.heading_index is None:
            # 尝试从环境配置中获取 features
            obs_config = self.env.unwrapped.config.get("observation", {})
            features = obs_config.get("features",
                ['presence', 'x', 'y', 'vx', 'vy', 'heading', 'cos_h', 'sin_h'])
            try:
                self.heading_index = features.index('heading')
                self.presence_index = features.index('presence')
            except ValueError:
                raise ValueError("Observation must include 'heading' and 'presence' to compute yaw_rate.")

        N = obs.shape[0]
        if self.prev_headings is None:
            self.prev_headings = np.zeros(N, dtype=np.float32)

        current_headings = obs[:, self.heading_index]
        presence = obs[:, self.presence_index]

        # 计算偏航率
        delta_heading = current_headings - self.prev_headings
        delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi
        yaw_rates = delta_heading / self.dt

        # 更新 prev_headings（仅对存在的车辆）
        self.prev_headings = np.where(presence > 0.5, current_headings, self.prev_headings)

        # 拼接 yaw_rate 到最后一列
        yaw_rates = yaw_rates.reshape(-1, 1)
        obs_with_yaw = np.concatenate([obs, yaw_rates], axis=1)
        return obs_with_yaw.astype(np.float32)




if __name__ == '__main__':
    # main.py
    import gymnasium as gym
    import highway_env

    # 创建基础环境，并配置为包含 'heading'
    env = gym.make("highway-v0")
    env.unwrapped.configure({
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "heading", "cos_h", "sin_h"],
            "vehicles_count": 5,
            "absolute": False,
            "order": "sorted"
        }
    })

    # 用 wrapper 添加 yaw_rate
    env = YawRateObservationWrapper(env)

    obs, info = env.reset()

    print("Obs shape:", obs.shape)  # 应为 (5, 9)
    print("Yaw rate (ego):", obs[0, -1])


