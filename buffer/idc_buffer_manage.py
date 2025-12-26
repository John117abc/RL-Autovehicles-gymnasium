import threading
import numpy as np
from collections import deque
import random


class IDCBuffer:
    """随机选择轨迹的IDC缓冲区，但保持轨迹内部连续性"""

    def __init__(self, max_trajectories=100):
        """初始化缓冲区"""
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        self.lock = threading.Lock()

    def add_trajectory(self, trajectory):
        """添加一条完整轨迹"""
        with self.lock:
            self.trajectories.append(trajectory)

    def sample_batch(self, batch_size):
        """随机选择一条足够长的轨迹，然后采样连续批次

        Args:
            batch_size: 批次大小

        Returns:
            (states, actions, rewards, next_states, dones, trajectory_id) 或 None
        """
        with self.lock:
            if len(self.trajectories) == 0:
                return None

            # 收集所有足够长的轨迹索引
            valid_indices = [i for i, traj in enumerate(self.trajectories)
                             if len(traj) >= batch_size]

            if not valid_indices:
                return None  # 没有足够长的轨迹

            # 随机选择一条足够长的轨迹
            traj_id = random.choice(valid_indices)
            trajectory = self.trajectories[traj_id]

            # 在这条轨迹中随机选择一个起点
            max_start = len(trajectory) - batch_size
            start_idx = random.randint(0, max_start)
            batch = trajectory[start_idx:start_idx + batch_size]

            # 转换为批次格式
            states, actions, rewards, values, next_states, dones, infos = zip(*batch)
            state_infos = [exp[0] for exp in batch]
            return (state_infos, np.array(actions),
                    np.array(rewards), np.array(values), np.array(next_states),
                    np.array(dones),np.array(infos), traj_id)

    def sample_full_trajectory(self):
        """随机采样一条完整轨迹"""
        with self.lock:
            if len(self.trajectories) == 0:
                return None

            # 随机选择一条轨迹
            traj_id = random.randint(0, len(self.trajectories) - 1)
            trajectory = self.trajectories[traj_id]

            states, actions, rewards, values, next_states, dones, infos = zip(*trajectory)
            return (np.array(states), np.array(actions),
                    np.array(rewards), np.array(next_states),
                    np.array(dones), traj_id)

    def size(self):
        """返回总样本数"""
        with self.lock:
            return sum(len(traj) for traj in self.trajectories)

    def size_trajectory(self):
        """
        返回轨迹数
        :return: 轨迹数
        """
        with self.lock:
            return len(self.trajectories)

    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.trajectories.clear()