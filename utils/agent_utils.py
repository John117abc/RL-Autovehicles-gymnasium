import numpy as np
import torch


def get_three_lane_paths(env, horizon=50, longitudinal_step=5.0):
    """
    获取自车所在车道 + 左右相邻车道的静态参考路径（车道中心线）

    Args:
        env: 已 reset 的 highway-env 环境
        horizon: 路径点数量
        longitudinal_step: 纵向采样间隔（米）

    Returns:
        paths: list of 3 elements [left_path, current_path, right_path]
               每个元素是 shape=(horizon, 2) 的 np.array 或 None（如果车道不存在）
    """
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road

    # 获取当前车道索引
    current_index = ego.lane_index
    route, lane_id = current_index[0], current_index[1]
    current_lane_idx = current_index[2]

    # 获取总车道数（假设所有路段车道数一致）
    total_lanes = len(road.network.graph[route][lane_id])

    paths = [None, None, None]  # [left, current, right]

    # 定义要采样的三个车道索引
    lane_indices_to_sample = [
        current_lane_idx - 1,  # left
        current_lane_idx,  # current
        current_lane_idx + 1  # right
    ]

    s0 = ego.position[0]  # 当前纵向位置

    for i, lane_idx in enumerate(lane_indices_to_sample):
        if 0 <= lane_idx < total_lanes:
            # 构造完整的 lane_index
            lane_index = (route, lane_id, lane_idx)
            lane = road.network.get_lane(lane_index)

            # 采样路径点
            points = []
            for j in range(horizon):
                s = s0 + j * longitudinal_step
                xy = lane.position(s, 0.0)  # 横向偏移为 0（车道中心）
                points.append(xy)
            paths[i] = np.array(points)
        else:
            paths[i] = None  # np.full((horizon, 2), np.nan) 占位也行

    return paths  # [left_path, current_path, right_path]



def denormalize_action(action_norm, acceleration_range, steering_range):
    """
    将 [-1, 1] 范围内的归一化动作（PyTorch Tensor）映射到实际的 steering 和 acceleration 范围。

    参数:
        action_norm (torch.Tensor):
            - shape: [..., 2]，最后一维为 [steering_norm, acceleration_norm]，值 ∈ [-1, 1]
        acceleration_range (tuple/list): [min_acc, max_acc]，例如 [-5.0, 5.0]
        steering_range (tuple/list): [min_steering, max_steering]，例如 [-0.785, 0.785]

    返回:
        torch.Tensor: [..., 2]，最后一维为 [steering_real, acceleration_real]
    """
    # 确保输入是 Tensor
    if not isinstance(action_norm, torch.Tensor):
        action_norm = torch.as_tensor(action_norm, dtype=torch.float32)

    action_norm = action_norm.detach()

    # 分离 steering 和 acceleration（保留所有前导维度）
    s_norm = action_norm[..., 0]  # [...]
    a_norm = action_norm[..., 1]  # [...]

    # 转换范围为 Tensor（支持 GPU / 梯度无关）
    steer_min, steer_max = float(steering_range[0]), float(steering_range[1])
    acc_min, acc_max = float(acceleration_range[0]), float(acceleration_range[1])

    # 线性映射：x = x_min + (x_norm + 1) * (x_max - x_min) / 2
    steering_real = steer_min + (s_norm + 1.0) * (steer_max - steer_min) / 2.0
    acceleration_real = acc_min + (a_norm + 1.0) * (acc_max - acc_min) / 2.0

    # 合并回 [..., 2]
    return torch.stack([steering_real, acceleration_real], dim=-1)


def normalize_action(action_real, acceleration_range, steering_range):
    """
    将真实动作（steering, acceleration）线性映射回 [-1, 1] 的归一化空间。

    参数:
        action_real (torch.Tensor):
            - shape: [..., 2]，最后一维为 [steering_real, acceleration_real]
        acceleration_range (tuple/list): [min_acc, max_acc]，例如 [-5.0, 5.0]
        steering_range (tuple/list): [min_steering, max_steering]，例如 [-0.785, 0.785]

    返回:
        torch.Tensor: [..., 2]，最后一维为 [steering_norm, acceleration_norm] ∈ [-1, 1]
    """
    if not isinstance(action_real, torch.Tensor):
        action_real = torch.as_tensor(action_real, dtype=torch.float32)

    action_real = action_real.detach()

    a_real = action_real[..., 0]  # acceleration
    s_real = action_real[..., 1]  # steering

    acc_min, acc_max = float(acceleration_range[0]), float(acceleration_range[1])
    steer_min, steer_max = float(steering_range[0]), float(steering_range[1])

    acceleration_norm = 2.0 * (a_real - acc_min) / (acc_max - acc_min) - 1.0
    steering_norm = 2.0 * (s_real - steer_min) / (steer_max - steer_min) - 1.0

    return torch.stack([acceleration_norm, steering_norm], dim=-1)


def get_kinematics_ego(obs):
    """获取kinematics观察模式下自车的状态（x坐标，y坐标，纵向速度，横向速度，航向角，偏航率"""
    ego_state = obs[0]
    ego_state_x = ego_state[1]
    ego_state_y = ego_state[2]
    ego_state_vy = ego_state[4]
    ego_state_vx = ego_state[3]
    ego_state_heading = ego_state[5]
    ego_state_yaw_rate = ego_state[8]

    return [ego_state_x,ego_state_y,ego_state_vy,ego_state_vx,ego_state_heading,ego_state_yaw_rate]

def get_kinematics_surround(obs):
    """获取kinematics观察模式下周车的状态（x坐标，y坐标，纵向速度，横向速度，航向角，偏航率"""
    surround_state = obs[1:,:]
    surround_state_x = surround_state[:,1]
    surround_state_y = surround_state[:,2]
    surround_state_vy = surround_state[:,4]
    surround_state_vx = np.zeros_like(surround_state[:,3])
    surround_state_heading = surround_state[:,5]
    surround_state_yaw_rate = np.zeros_like(surround_state[:,8])

    return [surround_state_x,surround_state_y,surround_state_vy,surround_state_vx,surround_state_heading,surround_state_yaw_rate]

def get_kinematics_state(obs):
    """获取kinematics观察模式下的按照论文说的，道路信息，自车信息，周车信息集合 state"""
    # 静态路径信息
    state_lane_state = get_three_lane_paths(obs)
    # 自车信息
    ego_state = get_kinematics_ego(obs)
    # 周车信息
    surround_state = get_kinematics_surround(obs)

    return np.concatenate([state_lane_state.flatten(), ego_state.flatten(), surround_state.flatten()])



