import math


def compute_reward(env):
    # 1. 车道居中奖励（主项）
    ego = env.unwrapped.vehicle
    lane_index = ego.lane_index[2]  # 当前车道索引
    lateral_dist = get_lateral_distance_to_center(ego.position[1], lane_index)
    r_center = -0.5 * (lateral_dist ** 2) # 二次惩罚

    # 速度奖励
    speed = ego.speed
    r_speed = 0.1 * speed

    # 碰撞惩罚
    r_collision = -10.0 if ego.crashed else 0.0

    # 车道变更惩罚
    r_lane_change = -0.1 if ego.action['steering'] > 0.3 else 0.0

    total_reward = r_center + r_speed + r_collision + r_lane_change
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