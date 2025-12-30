import numpy as np
import torch
import itertools

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
            paths[i] = np.full((horizon, 2), np.nan)  # None也行

    return paths  # [left_path, current_path, right_path]

def calculate_x_road(env):
    """
    计算自车与道路边缘最近的点的坐标
    """
    # 自车（ego vehicle）
    ego = env.unwrapped.vehicle
    # 道路网络
    road = env.unwrapped.road
    # 自车当前所在的车道
    lane_index = road.network.get_closest_lane_index(ego.position)
    lane = road.network.get_lane(lane_index)
    # 获取自车在车道上的纵向坐标s
    s, d = lane.local_coordinates(ego.position)
    lane_width = lane.width  # 通常为 4.0

    # 自车当前位置的 s 值
    s_ego = s
    # 左右边缘在 s_ego 处的坐标
    left_edge = lane.position(s_ego, lane_width / 2)  # 左侧边缘
    right_edge = lane.position(s_ego, -lane_width / 2)  # 右侧边缘

    if d >= 0:
        nearest_edge = left_edge  # 更靠近左边缘
    else:
        nearest_edge = right_edge  # 更靠近右边缘
    # print("自车位置:", ego.position)
    # print("左边缘坐标:", left_edge)
    # print("右边缘坐标:", right_edge)

    return [nearest_edge[0]/100,nearest_edge[1]/100,0,0,0,0]

def calculate_state_error(obs, reference):
    """
    计算状态误差向量 s^ref = [delta_p, delta_phi, delta_v] 和静态路径的x，y，v_l,φ

    Args:
        obs: 观察状态
            {
                'position': np.array([x, y]),
                'heading': float,  # 当前航向角
                'speed': float     # 当前纵向速度
            }
        reference: 选定的参考轨迹（来自get_three_lane_references的输出）

    Returns:
        s_ref: np.array([delta_p, delta_phi, delta_v])
    """
    # 查找自车信息
    ego = get_kinematics_ego(obs)
    ego_state = {'position':np.array([ego[1],ego[2]]),
                 'heading':ego[5],
                 'speed':ego[4]}
    # 1. 位置误差 delta_p
    # 找到最近的参考点
    distances = np.linalg.norm(reference['positions'] - ego_state['position'], axis=1)
    nearest_idx = np.argmin(distances)
    nearest_ref_point = reference['positions'][nearest_idx]

    # 计算到参考路径的垂直距离
    if nearest_idx < len(reference['positions']) - 1:
        # 使用前后点计算路径切线
        next_point = reference['positions'][nearest_idx + 1]
        path_tangent = next_point - nearest_ref_point
        path_tangent = path_tangent / np.linalg.norm(path_tangent)

        # 从最近点到自车的向量
        ego_to_ref = ego_state['position'] - nearest_ref_point

        # 投影到法线方向得到横向误差
        normal = np.array([-path_tangent[1], path_tangent[0]])  # 90度旋转
        lateral_error = np.dot(ego_to_ref, normal)

        # 确定符号：左侧为正，右侧为负
        delta_p = lateral_error
    else:
        delta_p = distances[nearest_idx]  # 简化处理

    # 2. 航向角误差 delta_phi
    ref_heading = reference['headings'][nearest_idx]
    delta_phi = ego_state['heading'] - ref_heading
    # 规范化到[-π, π]
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi

    # 3. 速度误差 delta_v
    ref_speed = reference['velocities'][nearest_idx]
    delta_v = ego_state['speed'] - ref_speed

    return np.array([delta_p, delta_phi, delta_v]),np.array([nearest_ref_point[0],nearest_ref_point[1],ref_speed,0.0,ref_heading,0])


def get_complete_lane_references(env, horizon=50, longitudinal_step=1.0):
    """
    获取 highway-env 中三条车道的完整参考信息
    """
    ego = env.unwrapped.vehicle
    road = env.unwrapped.road

    # 获取当前车道信息
    current_index = ego.lane_index
    route, lane_id = current_index[0], current_index[1]
    current_lane_idx = current_index[2]

    # 获取总车道数
    total_lanes = len(road.network.graph[route][lane_id])

    # 定义行为类型
    behavior_types = ["change_to_left", "lane_keep", "change_to_right"]
    lane_indices = [current_lane_idx - 1, current_lane_idx, current_lane_idx + 1]

    references = []

    for i, lane_idx in enumerate(lane_indices):
        ref = {
            'valid': False,
            'positions': np.full((horizon, 2), np.nan),
            'headings': np.full(horizon, np.nan),
            'velocities': np.full(horizon, np.nan),
            'behavior': behavior_types[i],
            'lane_index': None
        }

        # 检查车道是否存在
        if 0 <= lane_idx < total_lanes:
            lane_index = (route, lane_id, lane_idx)
            lane = road.network.get_lane(lane_index)
            ref['valid'] = True
            ref['lane_index'] = lane_index

            # 1. 采样位置和航向角
            positions = []
            headings = []
            s0 = ego.position[0]  # 当前纵向位置

            for j in range(horizon):
                s = s0 + j * longitudinal_step
                positions.append(lane.position(s, 0.0))  # 车道中心线
                headings.append(lane.heading_at(s))

            ref['positions'] = np.array(positions)
            ref['headings'] = np.array(headings)

            # 2. 生成速度剖面
            ref['velocities'] = generate_velocity_profile(
                env, lane_index, s0, horizon, longitudinal_step, behavior_types[i]
            )

        references.append(ref)

    return references


def generate_velocity_profile(env, lane_index, s_start, horizon=50, step=5.0, behavior_type="lane_keep"):
    """
    为 highway-env 生成完整的速度剖面

    Args:
        env: highway-env 环境
        lane_index: 目标车道索引
        s_start: 起始纵向位置
        horizon: 路径点数量
        step: 纵向步长 (米)
        behavior_type: 行为类型

    Returns:
        velocity_profile: 速度剖面数组 (m/s)
    """
    ego_vehicle = env.unwrapped.vehicle
    road = env.unwrapped.road
    lane = road.network.get_lane(lane_index)

    velocity_profile = []

    for i in range(horizon):
        s_position = s_start + i * step

        # 1. 计算基础速度 (考虑道路几何)
        base_speed = calculate_base_speed(env, lane_index, s_position)

        # 2. 计算交通约束速度
        traffic_speed = calculate_traffic_speed(env, ego_vehicle, lane_index, s_position)

        # 3. 考虑行为调整
        behavior_speed = adjust_speed_for_behavior(behavior_type, base_speed, ego_vehicle.speed)

        # 4. 综合确定最终速度
        # 优先考虑安全约束
        final_speed = min(base_speed, traffic_speed, behavior_speed)

        # 5. 平滑处理 (避免速度突变)
        if i > 0:
            max_accel = 2.0  # m/s²
            max_decel = 3.0  # m/s²
            prev_speed = velocity_profile[-1]

            # 限制加速度
            if final_speed > prev_speed:
                max_possible_speed = prev_speed + max_accel * (step / max(prev_speed, 5.0))
                final_speed = min(final_speed, max_possible_speed)
            else:
                # 限制减速度
                min_possible_speed = prev_speed - max_decel * (step / max(prev_speed, 5.0))
                final_speed = max(final_speed, min_possible_speed)

        # 6. 确保速度在合理范围内
        final_speed = np.clip(final_speed, 0.0, env.unwrapped.config.get("speed_limit", 30.0))

        velocity_profile.append(final_speed)

    return np.array(velocity_profile)


def calculate_curvature_at_point(lane, s_position):
    """
    计算 highway-env 中 lane 在指定位置的曲率

    Args:
        lane: highway-env 的 AbstractLane 对象
        s_position: 纵向位置 (米)

    Returns:
        curvature: 曲率 (1/米)
    """
    # 使用数值微分计算曲率：通过相邻点的航向角变化
    delta_s = 1.0  # 1米的步长用于数值微分

    # 获取当前点和前后点的航向角
    heading_current = lane.heading_at(s_position)
    heading_next = lane.heading_at(s_position + delta_s)

    # 计算航向角变化率 (弧度/米)
    heading_diff = (heading_next - heading_current + np.pi) % (2 * np.pi) - np.pi

    # 曲率 = 航向角变化率
    curvature = heading_diff / delta_s

    # 确保曲率在合理范围内
    return np.clip(curvature, -0.1, 0.1)  # highway-env 中典型曲率范围


def calculate_speed_limit_by_curvature(curvature, friction_coefficient=0.7):
    """
    根据曲率计算最大安全速度

    Args:
        curvature: 曲率 (1/米)
        friction_coefficient: 摩擦系数 (默认0.7，干燥沥青路面)

    Returns:
        max_safe_speed: 最大安全速度 (m/s)
    """
    if abs(curvature) < 1e-5:  # 直道
        return float('inf')

    # 物理公式：v = sqrt(μ * g * r)，其中 r = 1/|curvature|
    radius = 1.0 / abs(curvature)
    gravity = 9.81  # m/s²

    # 计算理论最大速度
    theoretical_max_speed = np.sqrt(friction_coefficient * gravity * radius)

    # 添加安全裕度 (80%)
    safe_speed = theoretical_max_speed * 0.8

    # highway-env 通常速度范围 0-30 m/s
    return np.clip(safe_speed, 0, 30.0)


def calculate_base_speed(env, lane_index, s_position, horizon_points=10):
    """
    计算 highway-env 中的基础速度

    Args:
        env: highway-env 环境
        lane_index: 车道索引 (route, lane_id, lane_idx)
        s_position: 当前纵向位置
        horizon_points: 前瞻点数，用于评估前方道路

    Returns:
        base_speed: 基础速度 (m/s)
    """
    road = env.unwrapped.road
    lane = road.network.get_lane(lane_index)

    # 1. 获取环境配置的速度限制
    config = env.unwrapped.config
    default_speed_limit = config.get("speed_limit", 30)  # 默认30 m/s

    # 2. 考虑前方道路曲率
    min_safe_speed = float('inf')
    for i in range(horizon_points):
        lookahead_s = s_position + i * 5.0  # 每5米检查一次
        curvature = calculate_curvature_at_point(lane, lookahead_s)
        speed_limit = calculate_speed_limit_by_curvature(curvature)
        min_safe_speed = min(min_safe_speed, speed_limit)

    # 3. 考虑道路类型
    road_type_speed = default_speed_limit
    if hasattr(lane, 'speed_limit'):
        road_type_speed = lane.speed_limit

    # 4. 综合确定基础速度
    base_speed = min(road_type_speed, min_safe_speed)

    # 5. highway-env 特定调整
    # 在环形交叉路口或复杂路段降低速度
    if "roundabout" in str(lane_index[0]).lower():
        base_speed *= 0.7

    return np.clip(base_speed, 5.0, 30.0)  # 限制在合理范围


def calculate_traffic_speed(env, ego_vehicle, lane_index, s_position, lookahead_distance=50.0):
    """
    计算 highway-env 中基于交通的约束速度

    Args:
        env: highway-env 环境
        ego_vehicle: 自车对象
        lane_index: 目标车道索引
        s_position: 自车在目标车道的纵向位置
        lookahead_distance: 前瞻距离 (米)

    Returns:
        traffic_speed: 交通约束速度 (m/s)
    """
    road = env.unwrapped.road
    vehicles = road.vehicles

    # 1. IDM (Intelligent Driver Model) 参数
    desired_time_headway = 1.5  # 期望时距 (秒)
    safe_distance = 4.0  # 最小安全距离 (米)
    max_acceleration = 2.0  # 最大加速度 (m/s²)
    comfortable_braking = 3.0  # 舒适制动减速度 (m/s²)

    # 2. 检查目标车道的前方车辆
    front_vehicle = None
    min_distance = float('inf')

    for vehicle in vehicles:
        if vehicle is ego_vehicle:
            continue

        # 检查是否在同一车道
        if vehicle.lane_index == lane_index:
            # 计算纵向距离
            vehicle_s = vehicle.position[0]  # highway-env 中 position[0] 通常是纵向位置

            # 检查是否在前方
            if vehicle_s > s_position and (vehicle_s - s_position) < lookahead_distance:
                distance = vehicle_s - s_position
                if distance < min_distance:
                    min_distance = distance
                    front_vehicle = vehicle

    # 3. 没有前方车辆，返回高速
    if front_vehicle is None:
        return ego_vehicle.speed * 1.2  # 允许轻微加速

    # 4. 有前方车辆，应用 IDM 模型
    front_speed = front_vehicle.speed
    relative_speed = ego_vehicle.speed - front_speed
    distance = min_distance

    # IDM 公式
    if distance < safe_distance:
        # 紧急制动
        return max(0, ego_vehicle.speed - comfortable_braking * 2.0)

    # 计算期望速度
    time_headway = distance / max(ego_vehicle.speed, 1.0)

    if time_headway < desired_time_headway:
        # 需要减速
        speed_reduction = (desired_time_headway - time_headway) * max_acceleration
        return max(0, front_speed - speed_reduction)
    else:
        # 可以保持或轻微加速
        return min(front_speed * 1.1, ego_vehicle.speed + max_acceleration * 0.5)


def adjust_speed_for_behavior(behavior_type, base_speed, ego_speed):
    """
    根据驾驶行为调整速度

    Args:
        behavior_type: 行为类型 ("lane_keep", "change_to_left", "change_to_right", "emergency_brake")
        base_speed: 基础速度 (m/s)
        ego_speed: 自车当前速度 (m/s)

    Returns:
        adjusted_speed: 调整后的速度 (m/s)
    """
    if behavior_type == "lane_keep":
        # 车道保持：接近基础速度
        return base_speed * 0.95  # 保持95%的基础速度

    elif behavior_type in ["change_to_left", "change_to_right"]:
        # 变道行为：需要减速以确保安全
        # 变道速度通常为基础速度的70-80%
        change_speed = base_speed * 0.75

        # 但不能比当前速度低太多（避免突然制动）
        min_speed = max(ego_speed * 0.6, 5.0)  # 不低于当前速度的60%或5m/s

        return max(change_speed, min_speed)

    elif behavior_type == "emergency_brake":
        # 紧急制动：大幅减速
        return max(ego_speed * 0.3, 2.0)  # 降至当前速度的30%或最低2m/s

    elif behavior_type == "overtake":
        # 超车行为：适当加速
        return min(base_speed * 1.1, ego_speed + 3.0)  # 增加10%或3m/s

    else:
        # 默认行为
        return base_speed * 0.9


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


def get_kinematics_ego(obs, out=None):
    """获取kinematics观察模式下自车的状态（x坐标，y坐标，纵向速度，横向速度，航向角，偏航率"""
    """
    获取 kinematics 观察模式下自车的状态：
    [x, y, vy, vx, heading, yaw_rate]

    Parameters:
        obs (np.ndarray): 形状为 (N, D) 的观测数组，obs[0] 为自车状态
        out (np.ndarray, optional): 预分配的输出数组，形状应为 (6,)

    Returns:
        np.ndarray: 形状为 (6,) 的自车状态数组
    """
    ego = obs[0]

    if out is None:
        out = np.empty(6, dtype=ego.dtype)

    # 按顺序填充：x, y, vy, vx, heading, yaw_rate
    out[0] = ego[1]/100  # x
    out[1] = ego[2]/100  # y
    out[2] = ego[4]/20  # vy
    out[3] = ego[3]/20  # vx
    out[4] = ego[5]  # heading
    out[5] = ego[8]  # yaw_rate

    return out

def get_kinematics_surround(obs, out=None):
    """获取kinematics观察模式下周车的状态（x坐标，y坐标，纵向速度，横向速度，航向角，偏航率"""
    s = obs[1:]
    M = s.shape[0]
    if out is None:
        out = np.empty((M, 6), dtype=s.dtype)
    out[:, 0] = s[:, 1]/100      # x
    out[:, 1] = s[:, 2]/100      # y
    out[:, 2] = s[:, 4]/20      # vy
    out[:, 3] = 0            # vx
    out[:, 4] = s[:, 5]      # heading
    out[:, 5] = 0            # yaw_rate
    return out

def get_kinematics_state_current(obs,env):
    """获取kinematics观察模式下的当前车道，道路信息，自车信息，周车信息集合 state"""
    # 自车信息
    state_ego = get_kinematics_ego(obs)
    # 周车信息
    state_other = get_kinematics_surround(obs)
    # 参考信息,1是自车道
    state_s_ref,state_x_ref = calculate_state_error(obs,get_complete_lane_references(env)[1])
    return {'state':np.concatenate([state_ego, list(itertools.chain.from_iterable(state_other)) , state_s_ref]),
            'state_ego':state_ego,
            'state_other':state_other,
            'state_s_ref':state_s_ref,
            'state_x_ref':state_x_ref
            }


def get_kinematics_state_static(env,obs,references_road):
    """获取kinematics观察模式下的参考车道，道路信息，自车信息，周车信息集合 state"""
    # 自车信息
    state_ego = get_kinematics_ego(obs)
    # 周车信息
    state_other = get_kinematics_surround(obs)
    # 参考信息,1是参考车道
    state_s_ref,state_x_ref = calculate_state_error(obs,references_road)
    # x_road信息，自车与道路两边最近的点的坐标
    x_road = calculate_x_road(env)
    return {'state':np.concatenate([state_ego, list(itertools.chain.from_iterable(state_other)) , state_s_ref]),
            'state_ego':state_ego,
            'state_other':state_other,
            'state_s_ref':state_s_ref,
            'state_x_ref':state_x_ref,
            'x_road':x_road
            }