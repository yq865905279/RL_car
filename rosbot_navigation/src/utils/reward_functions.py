"""
ROSbot导航环境的奖励函数模块
包含各种奖励计算逻辑，用于强化学习训练
"""

from doctest import debug
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Set
from collections import deque
    
class RewardFunctions:
    """
    奖励函数类，包含各种奖励计算逻辑
    """
    
    def __init__(self):
        """初始化奖励函数类"""
        # 奖励计算状态
        self.prev_distance_to_target = None
        self.reward_previous_position = (0.0, 0.0)
        self.current_rotation = 0.0
        self.visited_locations = set()
        self.same_spot_steps = 0
        self.prev_controls = []
        # 角度变化跟踪（用于“更靠近目标角度”奖励）
        self.prev_abs_angle_to_target = None
        
        # 原地打转检测变量
        self.previous_yaw_for_spin_check = 0.0
        self.total_rotation_in_place = 0.0
        self.stuck_steps_for_spin_check = 5       # 卡住超过5个动作开始检测 (更敏感)
        self.spin_termination_threshold = 4*math.pi    # 累计旋转半圈就终止 (更严格)
        
        # 卡住检测变量
        self.consecutive_stuck_steps = 0
        self.stuck_termination_threshold = 40      # 连续卡住超过15个动作就终止
        self.previous_position = None
        self.stuck_position_threshold = 0.05     # 单个动作位置变化小于此值认为卡住
        self.debug = False
        self.turn_steps = 0
        # 连续同方向旋转跟踪（用于靠墙惩罚叠加）
        self.last_rotation_sign = 0   # -1: 顺时针(示意)，+1: 逆时针，0: 无明显旋转
        self.rotation_streak = 0      # 连续相同方向旋转的步数
        # 速度稳定性统计（线速度方差）
        self._vel_hist = deque(maxlen=50)

    def _debug(self, msg: str):
        """条件调试输出"""
        try:
            if getattr(self, 'debug', False):
                step = int(getattr(self, '_global_step', 0))
                print(f"[DEBUG][inst {self.instance_id}][step {step}] {msg}")
        except Exception:
            pass
    
    def reset(self, position: Tuple[float, float, float], orientation: Tuple[float, float, float]):
        """重置奖励状态"""
        self.prev_distance_to_target = None
        self.reward_previous_position = (float(position[0]), float(position[1]))
        self.current_rotation = 0.0
        self.visited_locations.clear()
        self.same_spot_steps = 0
        self.prev_controls.clear()
        
        # 重置原地打转检测变量
        self.previous_yaw_for_spin_check = float(orientation[2])
        self.total_rotation_in_place = 0.0
        
        # 重置卡住检测变量
        self.consecutive_stuck_steps = 0
        self.previous_position = np.array([float(position[0]), float(position[1])])
        # 重置角度变化跟踪
        self.prev_abs_angle_to_target = None
        self.turn_steps = 0
        # 重置连续同方向旋转跟踪
        self.last_rotation_sign = 0
        self.rotation_streak = 0
        # 重置速度历史（用于稳定性方差统计）
        try:
            self._vel_hist.clear()
        except Exception:
            self._vel_hist = deque(maxlen=50)

    def calculate_reward(self, 
                            action: np.ndarray, 
                            observation: np.ndarray, 
                            env_state: Dict[str, Any]) -> float:
            """
            综合奖励函数：碰撞、到达、距离、靠近、时间、探索、移动、贴墙、反空转、角度与一致转向
            
            参数:
                action: 当前执行的动作
                observation: 当前观察
                env_state: 环境状态，包含以下键:
                    - get_sup_position: 获取机器人位置的函数
                    - get_sup_orientation: 获取机器人朝向的函数
                    - calculate_distance_to_target: 计算到目标距离的函数
                    - get_lidar_features: 获取LiDAR特征的函数
                    - detect_collision_simple: 检测碰撞的函数
                    - episode_steps: 当前回合步数
                    - cargo_type: 货物类型
                    - success_threshold: 成功阈值
                    - task_info: 任务信息
            
            返回:
                float: 计算得到的奖励值
            """
            rewards = {}
            # 从环境状态获取可选的参数对象（args），并提供安全的默认值
            args= env_state.get('args')


            # 计算距离与目标
            distance, closest_target = env_state['calculate_distance_to_target']()
            
            # 根据货物类型调整奖励
            if env_state['cargo_type'] == 'fragile':
                rewards['cargo_specific'] = self._fragile_cargo_reward(action, observation, env_state)
            elif env_state['cargo_type'] == 'dangerous':
                rewards['cargo_specific'] = self._dangerous_cargo_reward(action, observation, env_state)
            
            # 1) 碰撞惩罚（强惩罚）
            pose=env_state['get_sup_position']()
            collision_terminate = False
            try:
                if env_state['detect_collision_simple']():
                    if -2 <= pose[0] <= 2 and -2.7 <= pose[1] <= 2.7:
                        rewards['collision_penalty'] = -1000.0
                        collision_terminate = True
                    else:
                        rewards['collision_penalty'] = -1000.0
                        collision_terminate = True
                        # 不提前返回，终止标志将在后续逻辑中处理
                else:
                    rewards['collision_penalty'] = 0.0
                    collision_terminate = False
            except Exception:
                collision_terminate = False
                
            # 2) 卡住检测
            stuck_terminate = False
            current_position = np.array(pose[:2])  # 只考虑x, y坐标
            if self.previous_position is not None:
                position_change = np.linalg.norm(current_position - self.previous_position)
                if position_change < self.stuck_position_threshold:
                    self.consecutive_stuck_steps += 1
                    rewards['stuck_penalty'] = (-1)*self.consecutive_stuck_steps*self.consecutive_stuck_steps
                    if self.consecutive_stuck_steps >= self.stuck_termination_threshold:
                        rewards['stuck_penalty'] = -1000.0
                        stuck_terminate = True
                        print(f"[REWARD] 检测到卡住: 连续{self.consecutive_stuck_steps}步位置变化小于{self.stuck_position_threshold}")
                else:
                    self.consecutive_stuck_steps = 0
            self.previous_position = current_position.copy()
            


            # 3) 渐进式到达目标奖励，增加停车奖励
            DIST_THRESHOLD = env_state['success_threshold']
            # 获取当前速度（兼容旧版向量obs与新版字典obs）
            try:
                if isinstance(observation, dict):
                    odom = env_state['get_odometry_data']()
                    # 线速度与角速度（需要有符号的角速度用于判断旋转方向）
                    linear_vel = abs(float(odom['linear_velocity']))
                    angular_vel_signed = float(odom['angular_velocity'])
                    angular_vel = abs(angular_vel_signed)
                else:
                    linear_vel = abs(float(observation[23]))
                    angular_vel_signed = float(observation[40])
                    angular_vel = abs(angular_vel_signed)
            except Exception:
                linear_vel = 0.0
                angular_vel = 0.0
                angular_vel_signed = 0.0
            lw, rw = env_state.get('get_last_wheel_speeds', lambda: (None, None))()

            # 3) 渐进式到达目标奖励，强制要求停车
            goal_reached = False
            if distance < DIST_THRESHOLD:
                # 到达目标区域：必须停车才能获得完整奖励
                stop_bonus = (1.0 - min(1.0, linear_vel + angular_vel)) * 10
                base_goal_reward = 4000.0
                # 如果速度足够小（接近停车），给予巨大奖励并终止
                if (linear_vel < 0.1 and angular_vel < 0.1):
                    rewards['goal_reward'] = base_goal_reward + stop_bonus * getattr(args, 'stop_bonus_k', 100.0)
                    goal_reached = True  # 成功到达且停车，设置终止标志
                else:
                    # 在目标区域但未停车：只给部分奖励，鼓励停车
                    rewards['goal_reward'] = base_goal_reward * 0.6 + stop_bonus * getattr(args, 'stop_bonus_k', 60.0)
                    goal_reached = True
                    # 在目标附近打转给予惩罚
            elif distance < 1:  # 50cm内
                if angular_vel > 0.1:
                    rewards['goal_spin_penalty'] = -50.0 * angular_vel *(1/distance) 
                slow_down_reward = (1.0 - (linear_vel+angular_vel)) * 10
                rewards['slow_down_reward'] = slow_down_reward * getattr(args, 'slow_down_reward_k', 5.0)
                approach_reward = (1.0 - distance) * 10  # 接近奖励
                rewards['approach_goal'] = approach_reward * getattr(args, 'approach_reward_k', 5.0)

            # 设置终止标志（碰撞、卡住、成功到达且停车）
            env_state['terminate'] = collision_terminate or stuck_terminate or goal_reached

            # 4) 时间惩罚（随步数递增）
            time_penalty = 1 + (float(env_state['episode_steps']))
            rewards['time_penalty'] = -time_penalty*args.time_k

            # 5) 距离基奖励
            liner_distance_reward = args.liner_distance_reward
            if liner_distance_reward == 0:
                dist_reward = min(125.0, ((-1)*distance+125) if distance > 1e-6 else 125)
            elif liner_distance_reward == 1:
                dist_reward = (-0.3)*(distance)*(distance)+50
            elif liner_distance_reward == 2:
                dist_reward = 4/(distance+1)*50

            # 6) 距离变化（靠近奖励）
            prev_d = self.prev_distance_to_target if self.prev_distance_to_target is not None else distance
            dist_change = prev_d - distance
            if dist_change < 0 and distance<2:
                rewards['distance_change_reward'] = dist_change*args.delta_distance_k
            else:
                rewards['distance_change_reward'] = dist_change*args.delta_distance_k*((distance+1)/4)
            # print(f"Distance change reward: {rewards['distance_change_reward']}")
            
            if dist_change >0:
                rewards['distance_reward'] = dist_reward*args.distance_k
            else:
                rewards['distance_reward'] = dist_reward*args.distance_k*0.5
            
            # 7) 探索奖励（新位置）
            current_pos = env_state['get_sup_position']()
            robot_position = (float(current_pos[0]), float(current_pos[1]))
            # current_position = (round(robot_position[0], 2), round(robot_position[1], 2))
            # if current_position not in self.visited_locations:
            #     self.visited_locations.add(current_position)
            #     new_position_reward = 1
            #     if self.same_spot_steps > 2:
            #         new_position_reward *= 2.5
            #     rewards['exploration_reward'] = new_position_reward

            # 9) 移动奖励（位移）
            distance_moved = round(float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position))), 3)
            #print(f"Distance moved:{distance_moved},{args.movement_reward_k},{self.reward_previous_position}")
            movement_reward = distance_moved*args.movement_reward_k*(self.consecutive_stuck_steps/3 if self.consecutive_stuck_steps>3 else 1)
            #print(f"Movement reward: {movement_reward}")
            rewards['movement_reward'] = movement_reward

            # 9) 精确的原地停留检测：参考成功代码的逻辑
            #distance_moved = float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position)))
            
            # 参考成功代码：使用0.005的更严格阈值
            # if distance_moved < 0.1:  # 单步移动小于0.2m，视为在原地
            #     self.same_spot_steps += 1
            #     self.consecutive_stuck_steps += 1
            #     # 检查是否连续卡住超过阈值
            #     if self.consecutive_stuck_steps >= self.stuck_termination_threshold:
            #         rewards['stuck_penalty'] = -1000.0
            #         env_state['terminate'] = True
            #         _debug(self, f"Terminating episode due to being stuck for {self.consecutive_stuck_steps} steps")
            # else:
            #     # 一旦有明显移动，重置所有计数器
            #     self.same_spot_steps = 0
            #     self.consecutive_stuck_steps = 0
            #     self.total_rotation_in_place = 0.0
                
            # 参考成功代码：额外的原地停留惩罚机制
            # same_spot_penalty = 0.0
            # if self.same_spot_steps > 3:  # 参考成功代码的阈值
            #     same_spot_penalty = self.same_spot_steps * 0.25
            #     # 靠近墙壁时惩罚减半，鼓励转向
            #     close_to_wall = any(feature <= 0.2 for feature in lidar_features)
            #     if close_to_wall:episode_reward
            #         same_spot_penalty *= 0.5
            #     # 距离目标近时惩罚加重
            #     if distance < 0.3:
            #         same_spot_penalty *= 1.5
            # rewards['same_spot_penalty'] = -same_spot_penalty

            # 10) 原地打转惩罚：
            if abs(lw-rw) > 20:
                self.turn_steps += 1
                #print(f"Turn steps: {self.turn_steps}")
            else:
                self.turn_steps = 0

            if self.turn_steps > self.stuck_steps_for_spin_check:
                rewards['spin_penalty'] = (-1)*(self.turn_steps-3)*(self.turn_steps-3)*args.early_spin_penalty_k
                current_yaw = env_state['get_sup_orientation']()[2]
                delta_yaw = current_yaw - self.previous_yaw_for_spin_check
                # 处理角度环绕问题
                delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
                self.total_rotation_in_place += delta_yaw
                # 实时更新上一步的朝向，用于计算下一步的角度变化
                self.previous_yaw_for_spin_check = env_state['get_sup_orientation']()[2]
                
                # 如果累计旋转超过阈值，给予一个惩罚
                if self.total_rotation_in_place > self.spin_termination_threshold:
                    rewards['spin_penalty'] = -500.0
                

            else:
                rewards['spin_penalty'] = 0
            
            # 连续同方向旋转时提高靠墙惩罚
            # 仅当“旋转明显且线速度较小”才计入旋转方向
            # rot_ang_thr = getattr(args, 'rotation_ang_threshold', 0.3)
            # lin_thr = getattr(args, 'rotation_lin_threshold', 0.2)
            # curr_sign = 0
            # if angular_vel > rot_ang_thr and linear_vel < lin_thr:
            #     curr_sign = 1 if angular_vel_signed > 0 else -1
            # # 更新旋转连续计数
            # if curr_sign == 0:
            #     self.rotation_streak = 0
            #     self.last_rotation_sign = 0
            # else:
            #     if curr_sign == self.last_rotation_sign:
            #         self.rotation_streak += 1
            #     else:
            #         self.rotation_streak = 1
            #         self.last_rotation_sign = curr_sign
            # # 近墙时放大靠墙惩罚：mult = 1 + beta * streak
            beta = float(getattr(args, 'same_dir_spin_wall_penalty_k', 1))
               # 8) 靠近墙壁惩罚（LiDAR特征）
            lidar_features = env_state['get_lidar_features']()
            wall_prox_penalty = 0.0
            if -2 <= pose[0] <= 2 and -2.7 <= pose[1] <= 2.7:
                for feature in lidar_features:
                    if feature <= 1:
                        wall_prox_penalty += (1 - float(feature))
            else:
                for feature in lidar_features:
                    if feature <= 1:
                        wall_prox_penalty += (1 - float(feature))

            # if self.turn_steps >= 3:
            #     wall_prox_penalty *= (1.0 + beta * self.turn_steps)
            rewards['wall_proximity_penalty'] = (-1)*wall_prox_penalty*args.wall_proximity_penalty_k
            #((distance+1)/3)

            # === 暴露用于统计的原始安全度量与加速度（不乘系数） ===
            # 使用 env_state['step_metrics'] 传递给环境，供 info 与训练回调统计
            try:
                if 'step_metrics' not in env_state or not isinstance(env_state.get('step_metrics'), dict):
                    env_state['step_metrics'] = {}
                # 1) 原始靠墙度量（墙面接近惩罚的原始值，不乘以 wall_proximity_penalty_k）
                env_state['step_metrics']['wall_proximity_raw'] = float(wall_prox_penalty)
                # 2) 线性加速度（幅值），从环境提供的估计接口获取
                if 'estimate_linear_acceleration' in env_state and callable(env_state['estimate_linear_acceleration']):
                    acc = env_state['estimate_linear_acceleration']()
                    if hasattr(acc, '__len__'):
                        lin_acc = float(acc[0])
                    else:
                        lin_acc = float(acc)
                    env_state['step_metrics']['linear_acc'] = float(abs(lin_acc))
            except Exception:
                # 统计数据非关键，出错时静默忽略
                pass

            
            # 11) 角度奖励：仅在“前方有路径”且“与目标夹角小于90°”时给予奖励
            angle_to_target = self._calculate_angle_to_target(env_state)
            abs_angle = abs(angle_to_target)
            
            # 安全获取 LiDAR 特征

            lidar_features = env_state['get_lidar_features']()

            clear_path_to_target = False
            front_clear = False

            num_feats = len(lidar_features) if hasattr(lidar_features, '__len__') else 0
            # # 将角度(-pi, pi)映射到索引[0, N-1]
            # target_angle_index = int(((angle_to_target + math.pi) / (2.0 * math.pi)) * num_feats)
            # target_angle_index = max(0, min(target_angle_index, num_feats - 1))
            # clear_path_to_target = float(lidar_features[target_angle_index]) > 0.3

            # 判断“前方是否有路”：检查正前方附近的一小段扇区
            center_idx = int((0.5) * num_feats)
            half_width = 3
            start_idx = int(center_idx - half_width)
            end_idx = int(center_idx + half_width)
            front_max = max(float(lidar_features[i]) for i in range(start_idx, end_idx + 1))
            front_clear = front_max > 0.4
            front_sum = sum(float(lidar_features[i]) for i in range(start_idx, end_idx + 1))
            if front_clear:
                rewards['front_clear'] = front_sum*0.1*args.front_clear_k*(((-1)*(distance-5)**2*0.1+3))
            else:
                rewards['front_clear'] = 0.0

            # # 无条件给予角度奖励，确保机器人始终有朝向目标的引导信号
            # # 即使前方无路或背离目标，也应该知道需要转向
            # angle_reward = (-5)*abs_angle*abs_angle + 10
            # # 当前方有明显空间时，放大角度奖励
            # if front_clear:
            #     angle_reward *= 1.5  # 前方有路时奖励增强50%
            # rewards['angle_reward'] = angle_reward*args.angle_reward_k*(3/distance)


            # # 12) 角度变化奖励：若本步与目标的夹角较上一步更小，则给予正向奖励
            # try:
            #     prev_abs = self.prev_abs_angle_to_target
            #     angle_delta = (prev_abs - abs_angle) if (prev_abs is not None) else 0.0
            #     # 仅对“更接近目标方向”的变化奖励，负向变化不额外惩罚（已有其他项约束）
            #     angle_change_k = getattr(args, 'angle_change_k', getattr(args, 'angle_reward_k', 1.0))
            #     rewards['angle_change_reward'] = max(0.0, angle_delta)* angle_change_k
            # except Exception as e:
            #     # 兜底，避免日志噪音
            #     print(f"Error in angle change reward calculation: {e}")
            #     self._debug(f"Error in angle change reward calculation: {e}")
            #     rewards['angle_change_reward'] = 0.0
            # # 更新上一时刻的角度绝对值
            # self.prev_abs_angle_to_target = abs_angle
            
            # 13) 方向性移动奖励：如果移动方向与目标方向一致，给予额外奖励
            # 这确保机器人不仅转向正确，还要朝正确方向移动
            directional_reward = 0.0
            if distance_moved > 0.10:  # 确实在移动
                try:
                    # 计算实际移动方向
                    movement_angle = math.atan2(
                        robot_position[1] - self.reward_previous_position[1],
                        robot_position[0] - self.reward_previous_position[0]
                    )
                    # 计算目标方向
                    target_pos = env_state['task_info']['target_pos']
                    target_angle = math.atan2(
                        target_pos[1] - robot_position[1],
                        target_pos[0] - robot_position[0]
                    )
                    # 计算方向差异
                    angle_diff = abs((target_angle - movement_angle + math.pi) % (2*math.pi) - math.pi)
                    # # 角度差越小，奖励越大（从1.0到0）
                    # if distance > 8:
                    #     alignment = 1.0 - angle_diff/(((-0.017)*(distance-8)**2+0.8)*math.pi)
                    # if distance > 2:
                    #     alignment = 1.0 - angle_diff/(((-0.01)*(distance-5.5)**2+1)*math.pi)
                    # elif distance < 2:
                    #     alignment = 1.0 - angle_diff/(((-0.017)*(distance-6)**2+0.8)*math.pi)

                    #alignment = 1.0 - angle_diff/(((-0.025)*(distance-5)**2+1)*math.pi) # 最开始约束约为70度，放大到180再缩小到70
                    alignment = 1.0 - angle_diff/(((0.0055)*(distance)**2+0.35)*math.pi) # 最开始约束约为70度，放大到180再缩小到70
                    # 奖励 = 对齐度 × 移动距离 × 系数 X 转圈衰减 X 距离目标加成
                    if alignment < 0:
                        directional_reward = alignment * abs(dist_change) * getattr(args, 'directional_movement_k', 15.0)*((distance-5)*(distance-5)*0.2+1)
                    else:
                        directional_reward = alignment * dist_change * getattr(args, 'directional_movement_k', 15.0)*((distance-5)*(distance-5)*0.2+1)
                except Exception as e:
                    print(f"Error in directional movement reward calculation: {e}")
                    directional_reward = 0.0
            rewards['directional_movement'] = directional_reward

            # # 14) 角速度惩罚
            # angular_velocity_penalty = 0.0
            # if angular_vel > 0.3 and distance < 3:
            #     angular_velocity_penalty = angular_vel * getattr(args, 'angular_velocity_penalty_k', 10)
            # rewards['angular_velocity_penalty'] = angular_velocity_penalty
            # === 调试输出：左右轮速度、目标角度与距离 ===
            try:
                dbg = bool(getattr(args, 'debug', False))
            except Exception:
                dbg = False
            if dbg:
                try:
                    dist_dbg, _ = env_state['calculate_distance_to_target']()
                except Exception:
                    dist_dbg = None
                try:
                    ang_dbg = self._calculate_angle_to_target(env_state)
                except Exception:
                    ang_dbg = None
                # 从观测中读取导航信息（若有），便于对比
                obs_nav = None
                try:
                    if isinstance(observation, dict):
                        if 'navigation_info' in observation:
                            obs_nav = observation['navigation_info']
                        elif 'robot_state' in observation and observation['robot_state'] is not None:
                            # 旧结构兜底，不强制
                            obs_nav = None
                except Exception:
                    obs_nav = None
                if obs_nav is not None:
                    obs_dist = float(obs_nav[0])
                    obs_ang = float(obs_nav[1]) if len(obs_nav) > 1 else None
                    print(f"[DEBUG][wheels] L={lw}, R={rw} | [nav_int] dist={dist_dbg}, angle={ang_dbg} | [nav_obs] dist={obs_dist}, angle={obs_ang}")
                else:
                    print(f"[DEBUG][wheels] L={lw}, R={rw} | [nav_int] dist={dist_dbg}, angle={ang_dbg}")
                for key, value in rewards.items():
                    print(f"[DEBUG][reward] {key}: {value}")

            
            # 更新缓存
            self.prev_distance_to_target = distance
            self.reward_previous_position = robot_position
            total_reward = sum(rewards.values())

            # === 奖励缩放到 -1 ~ 1 区间 ===
            # 使用对称线性缩放，以 |reward| = 300 时达到饱和；超出范围则对称裁剪
            #scale_limit = 300.0
            #scaled_reward = total_reward / scale_limit
            # 防止数值爆炸，保持对称裁剪
            #scaled_reward = float(np.clip(scaled_reward, -1.0, 1.0))
            scaled_reward = total_reward

            # print(f"""
            # Total reward: {total_reward},\n
            # collision_penalty: {rewards['collision_penalty']},\n
            # time_penalty: {rewards['time_penalty']},\n
            # distance_reward: {rewards['distance_reward']},\n
            # distance_change_reward: {rewards['distance_change_reward']},\n
            # movement_reward: {rewards['movement_reward']},\n
            # spin_penalty: {rewards['spin_penalty']},\n
            # angle_reward: {rewards['angle_reward']},\n
            # angle_change_reward: {rewards['angle_change_reward']}  
            # """)
            # wall_proximity_penalty: {rewards['wall_proximity_penalty']},\n
            # same_spot_penalty: {rewards['same_spot_penalty']},\n
            return scaled_reward
    
    def _calculate_angle_to_target(self, env_state: Dict[str, Any]) -> float:
        """
        计算当前航向与目标方向的角度偏差（-π, π）
        
        参数:
            env_state: 环境状态
            
        返回:
            float: 角度偏差
        """
        current_pos = env_state['get_sup_position']()
        target_pos = env_state['task_info']['target_pos']
        
        vec_to_target = target_pos - current_pos
        target_heading = math.atan2(float(vec_to_target[1]), float(vec_to_target[0]))

        # 使用线速度方向（世界坐标中的运动方向）而非机体朝向
        motion_heading = None
        try:
            if 'get_odometry_data' in env_state and callable(env_state['get_odometry_data']):
                odom = env_state['get_odometry_data']()
                vx = float(odom.get('dx', 0.0))
                vy = float(odom.get('dy', 0.0))
                if abs(vx) + abs(vy) > 1e-9:
                    motion_heading = math.atan2(vy, vx)
        except Exception:
            motion_heading = None
        # 若无法从线速度确定方向（例如速度过小或无里程计），回退到机体朝向
        if motion_heading is None:
            print("motion_heading is None")
            try:
                current_orient = env_state['get_sup_orientation']()
                motion_heading = float(current_orient[2])
            except Exception:
                motion_heading = 0.0

        angle = target_heading - motion_heading
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        #angle = math.atan2(math.sin(angle), math.cos(angle))
        return angle
    
    def _fragile_cargo_reward(self, action: np.ndarray, observation: np.ndarray, env_state: Dict[str, Any]) -> float:
        """
        易碎品专用奖励
        
        参数:
            action: 当前执行的动作
            observation: 当前观察
            env_state: 环境状态（用于获取加速度与里程计）
        
        返回:
            float: 奖励值
        """
        reward = 0.0
        args = env_state.get('args', None)
        # 1) 高加速度限制：对超出阈值的线性加速度进行惩罚
        try:
            acc_arr = env_state.get('estimate_linear_acceleration', lambda: np.array([0.0]))()
            linear_acc = float(acc_arr[0] if hasattr(acc_arr, '__len__') else float(acc_arr))
        except Exception:
            linear_acc = 0.0
        acc_thr = float(getattr(args, 'fragile_acc_threshold', 2) if args is not None else 2)  # m/s^2
        acc_k = float(getattr(args, 'fragile_acc_penalty_k', 3.0) if args is not None else 3.0)
        acc_excess = max(0.0, abs(linear_acc) - acc_thr)
        accel_penalty = -acc_k * (acc_excess ** 2)
        reward += accel_penalty

        # 2) 去除每步的速度方差计算以降低开销；方差改为在训练回调的滑动窗口阶段统计

        # 3) 可选：角速度平滑性（保持轻微约束）
        try:
            ang_vel = float(env_state.get('get_odometry_data', lambda: {'angular_velocity': 0.0})().get('angular_velocity', 0.0))
        except Exception:
            ang_vel = 0.0
        ang_penalty_k = float(getattr(args, 'fragile_ang_vel_penalty_k', 0.3) if args is not None else 0.3)
        angular_penalty = -ang_penalty_k * max(0.0, abs(ang_vel) - 1.0)
        reward += angular_penalty

        # 不在奖励函数内暴露速度方差（避免重复与性能开销），在训练回调中基于线速度窗口统计

        return reward
    
    def _dangerous_cargo_reward(self, action: np.ndarray, observation: np.ndarray, env_state: Dict[str, Any]) -> float:
        """
        危险品专用奖励
        
        参数:
            action: 当前执行的动作
            observation: 当前观察
            
        返回:
            float: 奖励值
        """
        reward = 0.0

        args = env_state.get('args', None)
        # 1) 高加速度限制：对超出阈值的线性加速度进行惩罚
        try:
            acc_arr = env_state.get('estimate_linear_acceleration', lambda: np.array([0.0]))()
            linear_acc = float(acc_arr[0] if hasattr(acc_arr, '__len__') else float(acc_arr))
        except Exception:
            linear_acc = 0.0
        acc_thr = float(getattr(args, 'fragile_acc_threshold', 1) if args is not None else 1)
        acc_k = float(getattr(args, 'fragile_acc_penalty_k', 4.0) if args is not None else 4.0)
        acc_excess = max(0.0, abs(linear_acc) - acc_thr)
        accel_penalty = -acc_k * (acc_excess ** 2)
        reward += accel_penalty

        # 2) 去除每步的速度方差计算以降低开销；方差改为在训练回调的滑动窗口阶段统计

        # 3) 可选：角速度平滑性（保持轻微约束）
        try:
            ang_vel = float(env_state.get('get_odometry_data', lambda: {'angular_velocity': 0.0})().get('angular_velocity', 0.0))
        except Exception:
            ang_vel = 0.0
        ang_penalty_k = float(getattr(args, 'fragile_ang_vel_penalty_k', 0.3) if args is not None else 0.3)
        angular_penalty = -ang_penalty_k * max(0.0, abs(ang_vel) - 1.0)
        reward += angular_penalty
        
        # 安全性奖励 - 保持安全距离
        # 安全性奖励 - 保持安全距离
        # try:
        #     # 优先通过环境的雷达特征接口（更一致）
        #     if 'get_lidar_features' in env_state and callable(env_state['get_lidar_features']):
        #         lidar_data = env_state['get_lidar_features']()
        #     else:
        #         # 回退：从观测中提取或默认安全值
        #         if isinstance(observation, dict):
        #             lidar_data = np.array([1.0]*20, dtype=np.float32)
        #         else:
        #             lidar_data = observation[0:20]
        # except Exception:
        #     lidar_data = np.array([1.0]*20, dtype=np.float32)
        # min_obstacle_dist = min(lidar_data)
        
        # if min_obstacle_dist < 0.8:  # 距离障碍物小于0.7m
        #     safety_penalty = -20.0 * (0.8 - min_obstacle_dist)
        #     reward += safety_penalty
        
        # 保守速度奖励
        try:
            if isinstance(observation, dict):
                odom = observation.get('_odometry_cache_', None)
                if odom is None:
                    odom = {'linear_velocity': 0.0}
                linear_vel = float(odom['linear_velocity'])
            else:
                linear_vel = float(observation[23])
        except Exception:
            linear_vel = 0.0
        conservative_reward = -(linear_vel-0.4)* 70
        reward += conservative_reward
        
        return reward
    
    def update_action_history(self, action: np.ndarray) -> None:
        """
        更新动作历史
        
        参数:
            action: 当前执行的动作
        """
        try:
            self.prev_controls.append([float(action[0]), float(action[1])])
            if len(self.prev_controls) > 10:
                self.prev_controls.pop(0)
        except Exception:
            pass
        
    def update_current_rotation(self, angular_vel: float) -> None:
        """
        更新当前旋转
        
        参数:
            angular_vel: 当前角速度
        """
        self.current_rotation = float(angular_vel)
    
    def is_excessive_spin(self) -> bool:
        """
        检查是否过度旋转
        
        返回:
            bool: 是否过度旋转
        """
        return self.total_rotation_in_place > self.spin_termination_threshold
