import math


def compute_reward(env):
    # 车道居中奖励
    ego = env.unwrapped.vehicle
    lane_index = ego.lane_index[2]  # 当前车道索引
    lateral_dist = get_lateral_distance_to_center(ego.position[1], lane_index)
    r_center = -0.5 * (lateral_dist ** 2) # 二次惩罚

    # 速度奖励
    speed = ego.speed
    r_speed = 0.1 * speed

    # 碰撞惩罚
    r_collision = -1000.0 if ego.crashed else 0.0

    # 车道变更惩罚
    r_lane_change = -0.1 if ego.action['steering'] > 0.3 else 0.0

    total_reward = r_center + r_speed + r_collision + r_lane_change

    return total_reward


def compute_reward_IDC(env, action):
    # 主任务奖励（越大越好）
    speed_reward = env.unwrapped.vehicle.speed / env.unwrapped.config["reward_speed_range"][1]  # 归一化到 [0,1]
    smooth_reward = -0.05 * abs(action[0])  # 惩罚转向幅度（若 action 连续）

    # 安全约束惩罚（违反则负大值）
    collision_penalty = -10.0 if env.unwrapped.vehicle.crashed else 0.0
    off_road_penalty = -5.0 if not env.unwrapped.vehicle.on_road else 0.0

    # 可选：激进变道惩罚
    lane_change_penalty = -0.1 if action[0] != 0 else 0.0  # 鼓励少变道

    total_reward = (
            speed_reward
            + smooth_reward
            + collision_penalty
            + off_road_penalty
            + lane_change_penalty
    )
    return total_reward


def get_lateral_distance_to_center(ego_y, lane_index, lane_width=4.0):
    """
    计算 ego 车辆到其当前车道中心线的横向距离。

    Args:
        ego_y (float): 车辆 y 坐标
        lane_index (int): 当前车道索引（0, 1, 2, ...）
        lane_width (float): 车道宽度（默认 4.0 米）

    Returns:
        float: 到中心线的距离（绝对值，单位：米）
    """
    lane_center_y = (lane_index + 0.5) * lane_width
    return abs(ego_y - lane_center_y)