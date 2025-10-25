"""
ROSbot导航环境 - 支持AMCL定位的42维状态空间
基于Webots仿真环境和真实定位算法
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import os
from typing import Optional, Tuple, Dict, Any
from controller import Supervisor, GPS, InertialUnit, Gyro, Compass
from pathlib import Path

# 导入自定义模块
from ..utils.navigation_utils import NavigationUtils
from ..utils.reward_functions import RewardFunctions
from .local_map_obs import LocalMapObservation
from . import navigation_env_obstacles as obstacles
WEBOTS_AVAILABLE = True


class ROSbotNavigationEnv(gym.Env):
    """
    ROSbot导航环境类
    
    状态空间：42维
    - LiDAR数据 (20维)
    - 机器人位姿状态 (12维) 
    - 目标导航信息 (6维)
    - 航向控制信息 (4维)
    
    动作空间：2维（一次控制：左右轮速度百分比）
    - 动作为 [左轮速度百分比, 右轮速度百分比]
    - 范围均为 [0.0, 1.0]，对应实际速度 [0.0, 26.0] rad/s
    """
    @staticmethod
    def get_spaces(include_robot_state: bool = False,
                   include_navigation_info: bool = True,
                   nav_info_mode: str = 'minimal',
                   macro_action_steps: int = 1,
                   action_mode: str = 'wheels',
                   obs_mode: str = 'local_map'):
        """无需连接 Webots，返回与环境一致的多输入观测/动作空间定义
        参数:
          include_robot_state: 是否包含机器人状态向量
          include_navigation_info: 是否包含导航信息向量
          nav_info_mode: 'minimal' -> [distance_to_target, angle_to_target] 2维,
                         'full' -> 10维(与原先实现一致)
          obs_mode: 'local_map' -> 多输入字典（局部地图）；'lidar' -> 单一Box（20维LiDAR精选向量）
        """
        pi = math.pi
        
        # 观测空间定义：根据 obs_mode 决定
        if obs_mode == 'lidar':
            # 仅使用20维 LiDAR 精选向量（不包含地图）
            base_low = np.zeros(10, dtype=np.float32)
            base_high = np.ones(10, dtype=np.float32) * 10.0  # 距离上限按10m
            parts = [spaces.Box(low=base_low, high=base_high, dtype=np.float32)]
            # 可选：导航信息
            if include_navigation_info:
                if nav_info_mode == 'minimal':
                    nav_low = np.array([0.0, -pi], dtype=np.float32)
                    nav_high = np.array([20.0, pi], dtype=np.float32)
                else:
                    nav_low = np.array([
                        -20.0, -20.0, -2.0,
                        -20.0, -20.0, -2.0,
                        -pi,
                        -pi,
                        -1.0
                    ], dtype=np.float32)
                    nav_high = np.array([
                        20.0, 20.0, 2.0,
                        20.0, 20.0, 2.0,
                        pi,
                        pi,
                        1.0
                    ], dtype=np.float32)
                parts.append(spaces.Box(low=nav_low, high=nav_high, dtype=np.float32))
            # 可选：机器人状态，这里若开启则把它也拼接
            if include_robot_state:
                parts.append(spaces.Box(
                    low=np.array([
                        -20.0, -20.0, -2.0,
                        -pi, -pi, -pi,
                        -2.0, -2.0, -2.0,
                        -2.0, -2.0,
                        -1.0
                    ], dtype=np.float32),
                    high=np.array([
                        20.0, 20.0, 2.0,
                        pi, pi, pi,
                        2.0, 2.0, 2.0,
                        2.0, 2.0,
                        1.0
                    ], dtype=np.float32),
                    dtype=np.float32
                ))
            # 组合为单一Box（MlpPolicy需要向量）
            total_dim = int(np.sum([np.prod(p.shape) for p in parts]))
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        else:
            # 多输入字典：包含局部地图
            obs_dict = {}
            obs_dict['local_map'] = spaces.Box(
                low=-1.0, high=100.0,
                shape=(1, 200, 200),
                dtype=np.float32
            )
            if include_robot_state:
                obs_dict['robot_state'] = spaces.Box(
                    low=np.array([
                        -20.0, -20.0, -2.0,
                        -pi, -pi, -pi,
                        -2.0, -2.0, -2.0,
                        -2.0, -2.0,
                        -1.0
                    ], dtype=np.float32),
                    high=np.array([
                        20.0, 20.0, 2.0,
                        pi, pi, pi,
                        2.0, 2.0, 2.0,
                        2.0, 2.0,
                        1.0
                    ], dtype=np.float32),
                    dtype=np.float32
                )
            if include_navigation_info:
                if nav_info_mode == 'minimal':
                    obs_dict['navigation_info'] = spaces.Box(
                        low=np.array([0.0, -pi], dtype=np.float32),
                        high=np.array([20.0, pi], dtype=np.float32),
                        dtype=np.float32
                    )
                else:
                    obs_dict['navigation_info'] = spaces.Box(
                        low=np.array([
                            -20.0, -20.0, -2.0,
                            -20.0, -20.0, -2.0,
                            -pi,
                            -pi,
                            -1.0
                        ], dtype=np.float32),
                        high=np.array([
                            20.0, 20.0, 2.0,
                            20.0, 20.0, 2.0,
                            pi,
                            pi,
                            1.0
                        ], dtype=np.float32),
                        dtype=np.float32
                    )
            obs_space = spaces.Dict(obs_dict)
        
        # 动作空间：根据 macro_action_steps 决定维度
        steps = int(max(1, macro_action_steps))
        if action_mode == 'twist':
            # [linear_percent (0..1), angular_percent (-1..1)] per step
            low_pair = [0.0, -1.0]
            high_pair = [1.0, 1.0]
        else:
            # wheels: [left_percent (0..1), right_percent (0..1)] per step
            low_pair = [0.0, 0.0]
            high_pair = [1.0, 1.0]
        if steps == 1:
            act_low = np.array(low_pair, dtype=np.float32)
            act_high = np.array(high_pair, dtype=np.float32)
        else:
            act_low = np.array(low_pair * steps, dtype=np.float32)
            act_high = np.array(high_pair * steps, dtype=np.float32)
        act_space = spaces.Box(
            low=act_low,
            high=act_high,
            dtype=np.float32
        )
        return obs_space, act_space
    
    def __init__(self, 
                  cargo_type: str = 'normal', 
                  instance_id: Optional[int] = None,
                  controller_url: Optional[str] = None,
                  fast_mode: bool = True,
                  control_period_ms: int = 200,
                  headless: bool = True,
                  no_rendering: bool = True,
                  batch: bool = True,
                  minimize: bool = True,
                  include_robot_state: bool = False,
                  include_navigation_info: bool = True,
                  nav_info_mode: str = 'minimal',
                  macro_action_steps: int = 1,
                  action_mode: str = 'wheels',
                  obs_mode: str = 'local_map',
                  debug: bool = False,
                  training_mode: str = 'vertical_curriculum',
                  **kwargs):
        # 货物类型
        self.cargo_type = cargo_type
        self.is_training = True
        self.instance_id = instance_id if instance_id is not None else 0
        self.control_period_ms = int(control_period_ms) if control_period_ms and control_period_ms > 0 else 200
        # 调试模式
        self.debug = bool(debug)
        self.training_mode = str(training_mode)
        # 启用初始朝向目标
        self._rotate_to_target_on_reset = True
        
        # 多输入观测空间定义 - 使用Dict格式支持MultiInputPolicy
        pi = math.pi
        
        # 初始化局部地图观测器
        self.local_map_obs = LocalMapObservation(
            map_size=200,
            resolution=0.1,  # 10cm分辨率
            max_range=10.0
        )
        
        # 观测空间配置开关
        self.include_robot_state = bool(include_robot_state)
        self.include_navigation_info = bool(include_navigation_info)
        self.nav_info_mode = str(nav_info_mode)
        # 定义多输入观测空间（根据开关动态构造）
        self.macro_action_steps = int(max(1, macro_action_steps))
        self.action_mode = str(action_mode)
        self.observation_space, _ = ROSbotNavigationEnv.get_spaces(
            include_robot_state=self.include_robot_state,
            include_navigation_info=self.include_navigation_info,
            nav_info_mode=self.nav_info_mode,
            macro_action_steps=self.macro_action_steps,
            action_mode=self.action_mode,
            obs_mode=str(obs_mode)
        )
        self.obs_mode = str(obs_mode)
        # 速度平滑（单步限幅）开关，默认开启，可由上层通过 kwargs 传入
        self.enable_speed_smoothing = bool(kwargs.get('enable_speed_smoothing', True))
        
        # 动作空间 - 根据宏动作步数配置
        if self.action_mode == 'twist':
            low_pair = [0.0, -1.0]
            high_pair = [1.0, 1.0]
        else:
            low_pair = [0.0, 0.0]
            high_pair = [1.0, 1.0]
        if self.macro_action_steps == 1:
            self.action_space = spaces.Box(
                low=np.array(low_pair, dtype=np.float32),
                high=np.array(high_pair, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array(low_pair * self.macro_action_steps, dtype=np.float32),
                high=np.array(high_pair * self.macro_action_steps, dtype=np.float32),
                dtype=np.float32
            )

        # 由电机最大速度推导线速度/角速度上限（用于 twist 模式的缩放）
        self.max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
        _wr = getattr(self, 'wheel_radius', 0.043)
        _wb = getattr(self, 'wheel_base', 0.22)
        self.max_linear_speed = float(self.max_motor_speed * _wr)
        self.max_angular_speed = float(2.0 * self.max_motor_speed * _wr / max(_wb, 1e-6))
        
        # Webots控制器 - 使用兼容性层
        # 允许通过 controller_url 连接到特定的 Webots 实例（外部控制器）
        
        # 设置最大连接重试次数（在并行实例较多时适当增加重试和间隔）
        max_tries = 5
        retry_delay = 3  # 秒
        connected = False
        import time
        
        for try_num in range(max_tries):
            try:
                if controller_url:
                    print(f"🔌 实例 {self.instance_id} 连接到 Webots ({try_num+1}/{max_tries}): {controller_url}")
                    # 解析URL并设置环境变量
                    if controller_url.startswith('tcp://'):
                        # TCP连接格式：tcp://localhost:1234
                        import urllib.parse
                        parsed = urllib.parse.urlparse(controller_url)
                        host = parsed.hostname or 'localhost'
                        port = parsed.port or 10000 + (self.instance_id * 100)
                        
                        # 设置Webots的连接参数
                        os.environ['WEBOTS_SERVER'] = host
                        os.environ['WEBOTS_PORT'] = str(port)
                        # 同时设置标准的控制器URL，确保选择到正确的机器人（包含?name=...时生效）
                        os.environ['WEBOTS_CONTROLLER_URL'] = str(controller_url)
                        print(f"   设置连接: {host}:{port}")
                    else:
                        # 其他格式直接设置
                        os.environ['WEBOTS_CONTROLLER_URL'] = str(controller_url)
                else:
                    print(f"🔌 实例 {self.instance_id} 使用默认 Webots 连接 ({try_num+1}/{max_tries})")
                
                print(f"🤖 实例 {self.instance_id} 初始化 Supervisor...")
                self.robot = Supervisor()
                self.supervisor = self.robot
                print(f"✅ 实例 {self.instance_id} Supervisor 初始化成功")
                connected = True
                break
                
            except Exception as e:
                print(f"⚠️ 实例 {self.instance_id} 连接失败 ({try_num+1}/{max_tries}): {e}")
                if try_num < max_tries - 1:
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    # 尝试更改URL格式
                    if controller_url and '?name=' not in controller_url and 'tcp://' in controller_url:
                        controller_url = f"{controller_url}?name=rosbot"
                        print(f"🔄 修改URL格式: {controller_url}")
        
        if not connected:
            print(f"❌ 实例 {self.instance_id} 多次尝试后仍无法连接")
            raise ConnectionError(f"无法连接到 Webots 实例 {self.instance_id}")
            
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # 切换到 FAST 模式（若可用）
        try:
            if fast_mode and hasattr(self.supervisor, 'simulationSetMode') and hasattr(Supervisor, 'SIMULATION_MODE_FAST'):
                self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        except Exception:
            pass
        
        # 传感器设备
        self._setup_sensors()
        
        # 执行器设备  
        self._setup_actuators()
        
        # AMCL定位器
        # self.amcl_localizer = AMCLLocalizer(
        #     num_particles=800,
        #     initial_std=[0.2, 0.2, 0.15]
        # )
        # 导航工具
        self.nav_utils = NavigationUtils()
        
        # 状态变量
        self.state_buffer = {
            'previous_position': np.zeros(3),
            'previous_velocity': np.zeros(3),
            'previous_orientation': np.zeros(3),
            'previous_time': 0.0
        }
        # 最近一步的统计指标（用于info输出）
        self._last_step_metrics = {}
        
        # 初始化AMCL结果
        self.amcl_result = {
            'position_estimated': np.zeros(3),
            'orientation_estimated': np.zeros(3),
            'velocity_estimated': np.zeros(3),
            'position_uncertainty': 0.1
        }
        
        # 任务信息
        self.task_info = {
            'start_pos': np.zeros(3),
            'target_pos': np.zeros(3),
            'cargo_type': cargo_type
        }
        
        # 训练参数 - 随课程难度动态设置
        # 默认参数将由 _apply_curriculum_params 基于 NavigationUtils.curriculum_stage 设置
        self._apply_curriculum_params()
        
        # 轨迹跟踪
        self.trajectory = []
        self.min_obstacle_distance = float('inf')

        # 反打转与卡滞检测状态
        self.spin_steps = 0
        self.no_progress_steps = 0
        self.last_distance_to_target = None
        self._stuck = False
        # # 旋转/前进阈值（单位：m/s 与 rad/s）
        self.spin_linear_speed_threshold = 1.0
        self.spin_angular_speed_threshold = 1.0

        # 探索/移动奖励与角度相关状态
        self.episode_steps = 0
        
        # 奖励函数实例
        self.reward_functions = RewardFunctions()
        
        # 渐进式障碍物课程学习配置
        # enable_obstacle_curriculum 为首选开关；兼容旧参数 enable_obstacle_randomization
        self.enable_obstacle_curriculum = bool(kwargs.get(
            'enable_obstacle_curriculum',
            kwargs.get('enable_obstacle_randomization', True)
        ))
        # 兼容旧代码：保持同名属性，值与新开关一致
        self.enable_obstacle_randomization = self.enable_obstacle_curriculum
        self.obstacle_x_range = [-3.0, 3.0]  # x 范围（用于旧的随机位置逻辑）
        self.obstacle_y_range = [-3.5, 3.5]  # y 范围（用于旧的随机位置逻辑）
        self.obstacle_z_height = 0.3  # 障碍物高度（中心点）
        self.max_obstacles = 14  # 最大障碍物数量
        # 课程学习参数：优先从kwargs获取，否则使用默认值
        default_steps = [0,10000,18000,26000,34000,43000,53000,64000,76000,89000,110000,133000,156000]
        self.obstacle_curriculum_steps = kwargs.get('obstacle_curriculum_steps', default_steps)
        if self.obstacle_curriculum_steps is None:
            self.obstacle_curriculum_steps = default_steps

        default_counts = [2,3,4,5,6,7,8,9,10,11,12,13]
        self.obstacle_curriculum_counts = kwargs.get('obstacle_curriculum_counts', default_counts)
        if self.obstacle_curriculum_counts is None:
            self.obstacle_curriculum_counts = default_counts
        # self.obstacle_curriculum_steps = [0,10000,16000,22000,29000,37000,46000,56000,67000,79000]  # 每个阶段的步数
        # self.obstacle_curriculum_counts = [4,5,6,7,8,9,10,11,12,13]  # 对应的障碍物数量
        # self.obstacle_curriculum_steps = [0,5000,8000,12000,17000,26000,34000,46000,55000,65000]  # 每个阶段的步数
        # self.obstacle_curriculum_counts = [4,5,6,7,8,9,10,11,12,13]  # 对应的障碍物数量
        # self.obstacle_curriculum_steps = [0,5000,7000,9000,12000,16000,21000,27000,34000,42000]  # 每个阶段的步数
        # self.obstacle_curriculum_counts = [4,5,6,7,8,9,10,11,12,13]  # 对应的障碍物数量
        # self.obstacle_curriculum_steps = [0, 1000,2000,3000,4000,5000,6000,7000,8000]  # 每个阶段的步数
        # self.obstacle_curriculum_counts = [1,1,2,4,6,8,10,12,14]  # 对应的障碍物数量
        self._global_training_step = 0  # 全局训练步数（由 train_single.py 更新）
        self.obstacle_nodes = []  # 缓存障碍物节点
        self._obstacle_safe_zones = []  # 安全区域（避免随机到这些区域）
        
        # 新增：预定义障碍物位置列表（从这些位置中随机选择）
        self.use_predefined_positions = kwargs.get('use_predefined_positions', False)  # 是否使用预定义位置
        self.fixed_obstacle_count = kwargs.get('fixed_obstacle_count', 5)  # 固定激活的障碍物数量（仅当禁用课程学习时使用）
        # 新增：阶段锁定模式（每个课程阶段内障碍物集合不变；进入下一阶段在原有基础上新增一个）
        self.lock_obstacles_per_stage = bool(kwargs.get('lock_obstacles_per_stage', False))
        # 预定义位置：当use_predefined_positions=True时，会从 world 文件中自动读取 WoodenBox 的初始位置
        # 下面是默认值（仅当world文件中没有WoodenBox时使用）
        self.predefined_obstacle_positions = kwargs.get('predefined_obstacle_positions', [
            # 14个预设位置（备用）
            (-2.5, -2.0), (-1.5, -2.5), (-0.5, -2.0), (0.5, -2.5), (1.5, -2.0), (2.5, -2.5),  # 下方区域
            (-2.5, 2.0), (-1.5, 2.5), (-0.5, 2.0), (0.5, 2.5), (1.5, 2.0), (2.5, 2.5),      # 上方区域
            (-2.0, 0.0), (2.0, 0.0)  # 中间两侧
        ])

    def _apply_curriculum_params(self):
        """根据课程学习阶段设置训练参数。
        - start/easy 阶段更宽松（步数更多、成功阈值更大）
        - medium 适中
        - hard/end/all 更严格
        """
        try:
            stage = getattr(self.nav_utils, 'curriculum_stage', 'end')
        except Exception:
            stage = 'end'

        # (max_steps_per_episode, collision_threshold, success_threshold)
        params_by_stage = {
            'start':  (100, 0.05, 0.05),
            'easy':   (120,  0.05, 0.25),
            'medium': (140,  0.05, 0.25),
            'hard':   (160,  0.05, 0.25),
            'hard2':   (180,  0.05, 0.25),
            'end':    (250,  0.05, 0.30),
            'all':    (250,  0.05, 0.50),
        }
        max_steps, coll_th, succ_th = params_by_stage.get(stage, params_by_stage['end'])
        self.max_steps_per_episode = int(max_steps)
        self.collision_threshold = float(coll_th)
        self.success_threshold = float(succ_th)
        self._debug(f"Apply curriculum params: stage={stage}, max_steps={self.max_steps_per_episode}, collision_th={self.collision_threshold}, success_th={self.success_threshold}")
    
    def _setup_sensors(self):
        """初始化传感器设备"""        # 重置AMCL定位器
        # self.amcl_localizer.reset()
    
        # LiDAR传感器 - 在proto文件中名称为"laser"
        self.lidar = self.supervisor.getDevice('laser')
        if self.lidar:
            self.lidar.enablePointCloud()
            self.lidar.enable(self.timestep)
        
        # 彩色相机 - 用于图像识别
        # 根据 Astra PROTO 文件，RGB 相机设备名称是 'camera color' (带空格)
        self.camera_color = None
        possible_camera_names = ['camera color', 'camera rgb', 'camera', 'Camera', 'rgb_camera', 'color_camera']
        
        for camera_name in possible_camera_names:
            try:
                self.camera_color = self.supervisor.getDevice(camera_name)
                if self.camera_color:
                    self.camera_color.enable(self.timestep)
                    print(f"[Navigation Env] 彩色相机已启用: {camera_name}")
                    break
            except Exception as e:
                continue
        
        if not self.camera_color:
            print("[Navigation Env] 警告: 未找到彩色相机设备")
        
        # 直接通过supervisor获取机器人位置
        self.robot_node = self.supervisor.getSelf()
        
        # # IMU/姿态传感器 - 在proto文件中名称为"imu"
        # self.inertial_unit = self.supervisor.getDevice('imu')
        # if self.inertial_unit:
        #     self.inertial_unit.enable(self.timestep)
        # IMU设备无法获取，使用supervisor功能代替
        self.inertial_unit = None
        
        # 使用IMU内置的陀螺仪功能
        # self.gyro = self.inertial_unit  # IMU已经包含陀螺仪功能        
        # 使用supervisor功能代替陀螺仪
        self.gyro = None
        
        # 使用距离传感器进行碰撞检测
        # 获取前方距离传感器
        self.collision_sensors = []
        sensor_names = ['fl_range', 'fr_range', 'rl_range', 'rr_range']
        for name in sensor_names:
            sensor = self.supervisor.getDevice(name)
            if sensor:
                sensor.enable(self.timestep)
                self.collision_sensors.append(sensor)
        
        # 设置碰撞检测阈值（单位：米）
        self.collision_distance_threshold = 0.05

        # 用于基于节点识别的碰撞检测
        # 从 rosbot.proto 和 warehouse2.wbt 文件中提取的名称
        self.wheel_defs = {'front left wheel', 'front right wheel', 'rear left wheel', 'rear right wheel'}
        self.ground_defs = {'floor'}
    
    def _setup_actuators(self):
        """初始化执行器设备"""
        # 获取机器人节点
        self.robot_node = self.supervisor.getFromDef('rosbot')
        
        self.fl_motor = self.supervisor.getDevice('fl_wheel_joint')
        self.fr_motor = self.supervisor.getDevice('fr_wheel_joint')
        self.rl_motor = self.supervisor.getDevice('rl_wheel_joint')
        self.rr_motor = self.supervisor.getDevice('rr_wheel_joint')
        
        # 为了与原代码兼容，定义左右电机（使用前轮）
        self.left_motor = self.fl_motor
        self.right_motor = self.fr_motor
        
        # 设置电机模式
        for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
        
        # 读取电机最大速度以进行限幅（若获取失败则回退到26.0rad/s）
        try:
            speeds = []
            for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
                if motor:
                    speeds.append(motor.getMaxVelocity())
            self.max_motor_speed = float(min(speeds)) if speeds else 26.0
        except Exception:
            self.max_motor_speed = 26.0

        # 保存底盘参数，避免硬编码分散
        self.wheel_base = 0.22
        self.wheel_radius = 0.043

        # 轮速平滑缓存，避免瞬时大跃迁导致不稳定
        self._prev_left_speed = 0.0
        self._prev_right_speed = 0.0
    
    def test_reset(self,seed=None):
        """重置环境和状态"""
        super().reset(seed=seed)

        self._angle_mode_on_reset = 'axis'
        # 重置状态缓存
        self._reset_state_buffer()
        # 重置轨迹记录
        self.trajectory = []
        self.min_obstacle_distance = float('inf')
        
        # 初始化速度变量，用于平滑控制
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0
        self._last_cmd_linear_vel = 0.0
        self._last_cmd_angular_vel = 0.0
        self.spin_steps = 0
        self.no_progress_steps = 0
        self.last_distance_to_target = None
        self._stuck = False
        self.episode_steps = 0

        # 重置奖励函数状态
        _pos = self._get_sup_position()
        _orient = self._get_sup_orientation()
        self.reward_functions.reset(_pos, _orient)
        
        # 重置终止标志
        self._reward_terminate_flag = False
        
        # 如果电机存在，重置电机速度为0
        if hasattr(self, 'fl_motor') and self.fl_motor:
            self.fl_motor.setVelocity(0.0)
        if hasattr(self, 'fr_motor') and self.fr_motor:
            self.fr_motor.setVelocity(0.0)
        if hasattr(self, 'rl_motor') and self.rl_motor:
            self.rl_motor.setVelocity(0.0)
        if hasattr(self, 'rr_motor') and self.rr_motor:
            self.rr_motor.setVelocity(0.0)
        
        # 等待一小段时间，确保机器人完全停止
        # 增加等待时间，确保初始化完全
        for _ in range(5):
            self.supervisor.step(self.timestep)
        
        # 机器人初始朝向目标 (可选，按课程阶段角度模式)
        if hasattr(self, '_rotate_to_target_on_reset') and self._rotate_to_target_on_reset:
            angle_mode = getattr(self, '_angle_mode_on_reset', 'axis')
            if angle_mode == 'align':
                self._rotate_to_target_exact()
            elif angle_mode == 'random':
                self._set_random_yaw()
            elif angle_mode == 'exact_noise':
                self._rotate_to_target_exact_noise()
            else:
                # 'axis' 模式：使用简化到主方向
                self._rotate_to_target()

    def reset(self, seed=None, options=None):
        """重置环境和状态"""
        super().reset(seed=seed)
        
        # 设置新的导航任务
        self._set_navigation_task()
        # 重置AMCL定位器
        # self.amcl_localizer.reset()
        
        # 重置状态缓存
        self._reset_state_buffer()
        
        # 重置轨迹记录
        self.trajectory = []
        self.min_obstacle_distance = float('inf')
        
        # 初始化速度变量，用于平滑控制
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0
        self._last_cmd_linear_vel = 0.0
        self._last_cmd_angular_vel = 0.0
        self.spin_steps = 0
        self.no_progress_steps = 0
        self.last_distance_to_target = None
        self._stuck = False
        self.episode_steps = 0

        # 重置奖励函数状态
        _pos = self._get_sup_position()
        _orient = self._get_sup_orientation()
        self.reward_functions.reset(_pos, _orient)
        
        # 重置终止标志
        self._reward_terminate_flag = False
        
        # 如果电机存在，重置电机速度为0
        if hasattr(self, 'fl_motor') and self.fl_motor:
            self.fl_motor.setVelocity(0.0)
        if hasattr(self, 'fr_motor') and self.fr_motor:
            self.fr_motor.setVelocity(0.0)
        if hasattr(self, 'rl_motor') and self.rl_motor:
            self.rl_motor.setVelocity(0.0)
        if hasattr(self, 'rr_motor') and self.rr_motor:
            self.rr_motor.setVelocity(0.0)
        
        # 等待一小段时间，确保机器人完全停止
        # 增加等待时间，确保初始化完全
        for _ in range(5):
            self.supervisor.step(self.timestep)
        
        # 机器人初始朝向目标 (可选，按课程阶段角度模式)
        if hasattr(self, '_rotate_to_target_on_reset') and self._rotate_to_target_on_reset:
            angle_mode = getattr(self, '_angle_mode_on_reset', 'axis')
            if angle_mode == 'align':
                self._rotate_to_target_exact()
            elif angle_mode == 'random':
                self._set_random_yaw()
            elif angle_mode == 'exact_noise':
                self._rotate_to_target_exact_noise()
            else:
                # 'axis' 模式：使用简化到主方向
                self._rotate_to_target()
        
        # 获取初始观察
        observation, lidar_data = self._get_observation()
        
        # 设置初始AMCL状态
        # self._initialize_amcl_state()
        
        # 渐进式障碍物随机化（根据训练步数动态调整数量）
        if self.enable_obstacle_randomization:
            self._randomize_obstacles()
        
        info = {
            'start_position': self.task_info['start_pos'].copy(),
            'target_position': self.task_info['target_pos'].copy(),
            'cargo_type': self.cargo_type
        }
        
        # 仅返回 Gymnasium 规范的二元组
        return observation, info

    
    def _debug(self, msg: str):
        """条件调试输出"""
        try:
            if getattr(self, 'debug', False):
                step = int(getattr(self, '_global_step', 0))
                print(f"[DEBUG][inst {self.instance_id}][step {step}] {msg}")
        except Exception:
            pass
    
    def step(self, action):
        """执行动作并返回新状态
        
        参数:
            action: 当 macro_action_steps==1 时为 2维 [左轮速度百分比, 右轮速度百分比]；
                    当 macro_action_steps>1 时为 2*steps 维，按步拆分执行。
                - 两者范围均为 [0.0, 1.0]，对应实际速度 [0.0, 26.0] rad/s
                - 环境内部将百分比转换为实际轮速，并进行限幅与单步平滑
        """
        total_reward = 0
        observation = None
        steps = int(self.macro_action_steps)
        if steps <= 1:
            # 单步控制

            current_action = np.asarray(action, dtype=np.float32).reshape(2,)

            if self.action_mode == 'twist':
                self._debug(f"action (twist): v%={float(current_action[0]):.4f}, w%={float(current_action[1]):.4f}")
            else:
                self._debug(f"action (wheels): L%={float(current_action[0]):.4f}, R%={float(current_action[1]):.4f}")
            self._execute_action(current_action)
            self.reward_functions.update_action_history(current_action)
            self.reward_functions.update_current_rotation(float(getattr(self, '_last_cmd_angular_vel', 0.0)))
            self.episode_steps = int(self.episode_steps) + 1
            steps_per_control = max(1, int(self.control_period_ms / self.timestep))
            for _ in range(steps_per_control):
                self.supervisor.step(self.timestep)

            observation, lidar_data = self._get_observation()
            action_reward = self._calculate_reward(current_action, observation)
            total_reward += action_reward
            terminated = self._check_termination()
            truncated = self._check_truncation()

            d, _ = self._calculate_distance_to_target()
            self._debug(f"reward={action_reward:+.4f}, dist_to_target={d:.3f}, terminated={terminated}, truncated={truncated}")
            
        else:
            # 宏动作：按子步顺序执行
            actions = np.asarray(action, dtype=np.float32).reshape(steps, 2)
            terminated = False
            truncated = False
            for i in range(steps):
                current_action = actions[i]
                if self.action_mode == 'twist':
                    self._debug(f"substep {i+1}/{steps} action (twist): v%={float(current_action[0]):.4f}, w%={float(current_action[1]):.4f}")
                else:
                    self._debug(f"substep {i+1}/{steps} action (wheels): L%={float(current_action[0]):.4f}, R%={float(current_action[1]):.4f}")
                self._execute_action(current_action)
                self.reward_functions.update_action_history(current_action)
                self.reward_functions.update_current_rotation(float(getattr(self, '_last_cmd_angular_vel', 0.0)))
                self.episode_steps = int(self.episode_steps) + 1
                steps_per_control = max(1, int(self.control_period_ms / self.timestep))
                for _ in range(steps_per_control):
                    self.supervisor.step(self.timestep)
                observation, lidar_data = self._get_observation()
                action_reward = self._calculate_reward(current_action, observation)
                total_reward += action_reward
                terminated = self._check_termination()
                truncated = self._check_truncation()
                try:
                    d, _ = self._calculate_distance_to_target()
                    self._debug(f"reward={action_reward:+.4f}, dist_to_target={d:.3f}, terminated={terminated}, truncated={truncated}")
                except Exception:
                    self._debug(f"reward={action_reward:+.4f}, terminated={terminated}, truncated={truncated}")
                if terminated or truncated:
                    break
        
        # 更新状态信息
        info = self._get_step_info()
        
        # 记录机器人位置到轨迹
        if self.robot_node:
            try:
                current_pos = self._get_sup_position()
                self.trajectory.append(current_pos)
                # 打印当前位置，用于调试
                # print(f"机器人位置: {current_pos}")
            except Exception as e:
                print(f"记录轨迹错误: {e}")
        
        # 仅返回 Gymnasium 规范的五元组
        return observation, total_reward, terminated, truncated, info

    def _calculate_distance_to_target(self):
        """计算到当前目标的距离及目标位置"""
        current_pos = self._get_sup_position()
        target_pos = self.task_info['target_pos']
        distance = float(np.linalg.norm(current_pos - target_pos))
        return distance, target_pos

    def _calculate_angle_to_target(self, target_pos):
        """
        计算带符号的相对偏航角（-π, π）：
        使用实际移动方向（而非机器人朝向）与目标方向的夹角。
        如果机器人未移动，则回退到使用机器人朝向。
        移动方向与目标方向一致时为0，左偏为正，右偏为负。
        """
        current_pos = self._get_sup_position()
        previous_pos = self.state_buffer.get('previous_position', current_pos)
        
        # 计算目标方向
        vec_to_target = target_pos - current_pos
        target_heading = math.atan2(float(vec_to_target[1]), float(vec_to_target[0]))
        
        # 计算实际移动方向
        movement_vec = current_pos - previous_pos
        movement_distance = float(np.linalg.norm(movement_vec[:2]))
        
        # 如果移动距离足够大，使用移动方向；否则使用机器人朝向
        if movement_distance > 0.01:  # 1cm 阈值，避免噪声
            movement_heading = math.atan2(float(movement_vec[1]), float(movement_vec[0]))
            angle = target_heading - movement_heading
        else:
            # 回退：使用机器人朝向
            current_orient = self._get_sup_orientation()
            current_heading = float(current_orient[2])
            angle = target_heading - current_heading
        
        # 归一化到 [-π, π]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        # print(f"target_heading={target_heading}, current_heading={current_heading}, angle={angle}")
        return angle

    def _get_lidar_features(self):
        """获取用于奖励的LiDAR特征（0-1，约对应0-10m归一化）"""
        data,_ = self._get_lidar_data()
        return data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    
    def _set_navigation_task(self,test_id = None):
        """设置导航任务（集成课程学习阶段）"""
        if self.training_mode == 'horizontal_curriculum':
            start_pos, target_pos, angle_mode = self.nav_utils.get_curriculum_task()
            # 保存角度模式以便 reset 时设置朝向
            self._angle_mode_on_reset = angle_mode
            # 每次根据课程阶段应用对应训练参数
            self._apply_curriculum_params()

        elif self.training_mode == 'vertical_curriculum':
            start_pos, target_pos = self.nav_utils.get_navigation_task(self.cargo_type)
            self._angle_mode_on_reset = 'axis'
        
        self.task_info['start_pos'] = np.array(start_pos, dtype=np.float32)
        self.task_info['target_pos'] = np.array(target_pos, dtype=np.float32)
        self._reset_robot_position(start_pos)
                

    def _randomize_obstacles(self):
        """障碍物随机化：委托到 navigation_env_obstacles 模块实现。"""
        try:
            obstacles._randomize_obstacles(self)
        
        except Exception as e:
            print(f"[Obstacle] 随机化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_global_training_step(self, step: int):
        """
        更新全局训练步数（由训练脚本调用）
        
        参数:
            step: 当前全局训练步数
        """
        self._global_training_step = int(step)
    
    def _reset_robot_position(self, position):
        """重置机器人位置"""
        if self.robot_node:
            self.robot_node.getField('translation').setSFVec3f(list(position))
            self.robot_node.getField('rotation').setSFRotation([0, 1, 0, 0])
            # 清零机器人及其子节点的物理状态，避免残余速度导致起跳/漂移
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            # 额外执行几个周期以稳定着地
            for _ in range(2):
                self.supervisor.step(self.timestep)
                
    def _rotate_to_target(self):
        """将机器人朝向目标位置"""
        if self.robot_node and 'target_pos' in self.task_info:
            # 获取当前位置和目标位置
            current_pos = self._get_sup_position()
            target_pos = self.task_info['target_pos']

            
            # 计算朝向目标的方向向量
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            
            # 计算目标朝向角度（偏航角）
            target_yaw = math.atan2(dy, dx)

            # 将角度简化为四个主要方向（0, π/2, π, -π/2）
            # 根据角度所在的象限确定大致朝向
            if -math.pi/4 <= target_yaw < math.pi/4:
                # 朝向右侧 (东)
                simplified_yaw = 0
            elif math.pi/4 <= target_yaw < 3*math.pi/4:
                # 朝向上方 (北)
                simplified_yaw = math.pi/2
            elif target_yaw >= 3*math.pi/4 or target_yaw < -3*math.pi/4:
                # 朝向左侧 (西)
                simplified_yaw = math.pi
            else:
                # 朝向下方 (南)
                simplified_yaw = -math.pi/2

            # 加入20%-30%的扰动
            yaw_range = math.pi/2  # 每个方向覆盖π/2弧度
            perturb_ratio = random.uniform(0.0, 0.0)
            perturb = (yaw_range * perturb_ratio) * random.choice([-1, 1])
            simplified_yaw += perturb
            simplified_yaw = math.atan2(math.sin(simplified_yaw), math.cos(simplified_yaw))
            
            self._debug(f"目标朝向角度: {target_yaw}, 简化朝向: {simplified_yaw}")

            # 2. 重置物理状态，确保没有残余动量
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            
            # 3. 直接使用平移和旋转重置机器人，确保正确的姿态
            # 获取当前位置
            current_translation = self.robot_node.getField('translation').getSFVec3f()
            
            # 设置新的旋转 - 只改变Y轴旋转（偏航角），保持其他轴为0
            # Webots中，机器人应该是平放在地面上的，所以我们需要保持X和Z轴的旋转为0
            # 标准姿态是[0, 1, 0, angle]，表示绕Y轴旋转angle角度
            # 使用简化的朝向角度而不是精确角度
            new_rotation = [0, 0, 1, simplified_yaw]
            
            # 重新设置机器人的位置和姿态
            self.robot_node.getField('translation').setSFVec3f(current_translation)
            self.robot_node.getField('rotation').setSFRotation(new_rotation)

            # 等待物理引擎稳定
            for _ in range(5):
                self.supervisor.step(self.timestep)
    
    def _rotate_to_target_exact(self):
        """将机器人精确朝向目标（yaw = 指向目标的角度），并加入5%随机扰动"""
        if self.robot_node and 'target_pos' in self.task_info:
            current_pos = self._get_sup_position()
            target_pos = self.task_info['target_pos']
            dx = float(target_pos[0] - current_pos[0])
            dy = float(target_pos[1] - current_pos[1])
            target_yaw = math.atan2(dy, dx)
            # 添加5%扰动（正负方向均可）
            perturb_percent = random.uniform(0, 0.05)
            #perturb_percent = 0 
            perturb_direction = random.choice([-1, 1])
            yaw_perturb = perturb_direction * perturb_percent * math.pi  # 最大扰动为±18°~±36°
            target_yaw += yaw_perturb
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            current_translation = self.robot_node.getField('translation').getSFVec3f()
            new_rotation = [0, 0, 1, target_yaw]
            self.robot_node.getField('translation').setSFVec3f(current_translation)
            self.robot_node.getField('rotation').setSFRotation(new_rotation)
            for _ in range(5):
                self.supervisor.step(self.timestep)
    
    def _rotate_to_target_exact_noise(self):
        """将机器人精确朝向目标（yaw = 指向目标的角度），并加入5%随机扰动"""
        if self.robot_node and 'target_pos' in self.task_info:
            current_pos = self._get_sup_position()
            target_pos = self.task_info['target_pos']
            dx = float(target_pos[0] - current_pos[0])
            dy = float(target_pos[1] - current_pos[1])
            target_yaw = math.atan2(dy, dx)
            # 添加5%扰动（正负方向均可）
            perturb_percent = random.uniform(0.05, 0.2)
            #perturb_percent = 0 
            perturb_direction = random.choice([-1, 1])
            yaw_perturb = perturb_direction * perturb_percent * math.pi  # 最大扰动为±18°~±36°
            target_yaw += yaw_perturb
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            current_translation = self.robot_node.getField('translation').getSFVec3f()
            new_rotation = [0, 0, 1, target_yaw]
            self.robot_node.getField('translation').setSFVec3f(current_translation)
            self.robot_node.getField('rotation').setSFRotation(new_rotation)
            for _ in range(5):
                self.supervisor.step(self.timestep)


    def _set_random_yaw(self):
        """随机设置机器人yaw角"""
        if self.robot_node:
            rand_yaw = random.uniform(-math.pi, math.pi)
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            current_translation = self.robot_node.getField('translation').getSFVec3f()
            new_rotation = [0, 0, 1, rand_yaw]
            self.robot_node.getField('translation').setSFVec3f(current_translation)
            self.robot_node.getField('rotation').setSFRotation(new_rotation)
            for _ in range(3):
                self.supervisor.step(self.timestep)
    
    def _get_sup_position(self):
        """获取机器人位置（使用supervisor功能获取位置）"""
        if self.robot_node:
            # 使用supervisor API获取机器人位置
            position = self.robot_node.getPosition()
            return np.array(position, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def _get_sup_orientation(self):
        """获取机器人朝向（使用supervisor API）"""
        # 直接使用supervisor API获取朝向
        if self.robot_node:
            # 从机器人节点获取旋转矩阵
            rotation = self.robot_node.getOrientation()
            #print("Rotation:", rotation)
            # 转换为欧拉角 (roll, pitch, yaw)
            # 从旋转矩阵提取欧拉角
            # 矩阵格式为[r11 r21 r31 r12 r22 r32 r13 r23 r33]
            if len(rotation) >= 9:
                # 简化计算，只关注yaw角（绕z轴旋转）
                # yaw = atan2(r21, r11)
                yaw = math.atan2(rotation[1], rotation[0])
                # 简化roll和pitch计算
                roll = 0.0
                pitch = 0.0
                return np.array([roll, pitch, yaw], dtype=np.float32)
        
        # 如果无法获取，返回零向量
        return np.zeros(3, dtype=np.float32)
    
    def _reset_state_buffer(self):
        """重置状态缓存"""
        # 获取当前位置作为初始状态
        current_pos = self._get_sup_position()
        current_orient = self._get_sup_orientation()
        current_vel = np.zeros(3, dtype=np.float32)
        current_time = self.supervisor.getTime()
        
        self.state_buffer = {
            'previous_position': current_pos.copy(),
            'previous_velocity': current_vel.copy(),
            'previous_orientation': current_orient.copy(),
            'previous_time': current_time
        }
    
    def _execute_action(self, action):
        """执行动作"""
        # 防御式：将动作中的 NaN/Inf 替换为 0，并限定范围
        try:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            print(f"动作转换异常: {action}")
            action = np.array([0.0, 0.0], dtype=np.float32)
        if action.size < 2:
            action = np.pad(action, (0, max(0, 2 - action.size)), constant_values=0.0)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
        wheel_radius = getattr(self, 'wheel_radius', 0.043)
        wheel_base = getattr(self, 'wheel_base', 0.22)

        if getattr(self, 'action_mode', 'wheels') == 'twist':
            # action = [linear_percent (0..1), angular_percent (-1..1)]
            linear_percent = float(action[0])
            angular_percent = float(action[1])
            # 限幅，防止异常
            if not np.isfinite(linear_percent):
                linear_percent = 0.0
            if not np.isfinite(angular_percent):
                angular_percent = 0.0
            max_linear = float(getattr(self, 'max_linear_speed', max_motor_speed * wheel_radius))
            max_angular = float(getattr(self, 'max_angular_speed', 2.0 * max_motor_speed * wheel_radius / max(wheel_base, 1e-6)))
            linear_vel = np.clip(linear_percent, 0.0, 1.0) * max_linear
            angular_vel = np.clip(angular_percent, -1.0, 1.0) * max_angular
            # 反解为左右轮角速度(rad/s)
            cmd_left_speed = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
            cmd_right_speed = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
            self._debug(f"cmd_twist_in: v={linear_vel:.4f} m/s, w={angular_vel:.4f} rad/s -> L={cmd_left_speed:.4f}, R={cmd_right_speed:.4f}")
            # 允许双向旋转，剪裁到 [-max, max]
            left_speed = float(np.clip(cmd_left_speed, -max_motor_speed, max_motor_speed))
            right_speed = float(np.clip(cmd_right_speed, -max_motor_speed, max_motor_speed))
        else:
            # wheels 百分比：action = [left_percent (0..1), right_percent (0..1)]
            cmd_left_percent = float(action[0])
            cmd_right_percent = float(action[1])
            if not np.isfinite(cmd_left_percent):
                cmd_left_percent = 0.0
            if not np.isfinite(cmd_right_percent):
                cmd_right_percent = 0.0
            cmd_left_speed = cmd_left_percent * max_motor_speed
            cmd_right_speed = cmd_right_percent * max_motor_speed
            self._debug(f"cmd_wheels_in: L={cmd_left_percent:.4f}->{cmd_left_speed:.4f}, R={cmd_right_percent:.4f}->{cmd_right_speed:.4f}")
            left_speed = float(np.clip(cmd_left_speed, 0.0, max_motor_speed))
            right_speed = float(np.clip(cmd_right_speed, 0.0, max_motor_speed))

        # 平滑限速：限制单步变化，避免瞬时大扭矩引发不稳定（可选）
        if bool(getattr(self, 'enable_speed_smoothing', True)):
            max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
            # 每步允许的最大变化（与设备能力成比例）
            max_delta = max_motor_speed * 0.6  # 例如 60%/step
            prev_left = float(getattr(self, '_prev_left_speed', 0.0))
            prev_right = float(getattr(self, '_prev_right_speed', 0.0))
            # 限制变化范围；twist 模式允许负向
            if getattr(self, 'action_mode', 'wheels') == 'twist':
                left_speed = float(np.clip(left_speed, prev_left - max_delta, prev_left + max_delta))
                right_speed = float(np.clip(right_speed, prev_right - max_delta, prev_right + max_delta))
            else:
                left_speed = float(np.clip(left_speed, max(0.0, prev_left - max_delta), prev_left + max_delta))
                right_speed = float(np.clip(right_speed, max(0.0, prev_right - max_delta), prev_right + max_delta))
            self._debug(f"cmd_wheels_post_smooth: L={left_speed:.4f} (prev {prev_left:.4f}), R={right_speed:.4f} (prev {prev_right:.4f}), max_delta={max_delta:.3f}")
        else:
            # 未启用平滑时，仅打印一次性调试信息（保持接口一致）
            self._debug("speed smoothing disabled: using raw wheel speeds")
        
        # 设置四个电机速度 - 左侧两个轮子相同速度，右侧两个轮子相同速度
        if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
            # 左侧电机
            if not (np.isfinite(left_speed)):
                left_speed = 0.0
            if not (np.isfinite(right_speed)):
                right_speed = 0.0
            self.fl_motor.setVelocity(left_speed)
            self.rl_motor.setVelocity(left_speed)
            
            # 右侧电机
            self.fr_motor.setVelocity(right_speed)
            self.rr_motor.setVelocity(right_speed)
            
            # 打印调试信息
            self._debug(f"设置电机速度 - 左: {left_speed:.4f}, 右: {right_speed:.4f}")

        # 记录本次轮速用于下次平滑
        self._prev_left_speed = left_speed
        self._prev_right_speed = right_speed
        # 记录命令速度用于反打转检测（由轮速反推出等效线/角速度）
        wheel_base = getattr(self, 'wheel_base', 0.22)
        wheel_radius = getattr(self, 'wheel_radius', 0.043)
        linear_vel_est = wheel_radius * (left_speed + right_speed) / 2.0
        angular_vel_est = wheel_radius * (right_speed - left_speed) / max(wheel_base, 1e-6)
        if not np.isfinite(linear_vel_est):
            linear_vel_est = 0.0
        if not np.isfinite(angular_vel_est):
            angular_vel_est = 0.0
        self._last_cmd_linear_vel = float(linear_vel_est)
        self._last_cmd_angular_vel = float(angular_vel_est)
        self._debug(f"vel_est: lin={linear_vel_est:+.3f} m/s, ang={angular_vel_est:+.3f} rad/s")
    
    def _diff_drive_kinematics(self, linear_vel, angular_vel):
        """差速驱动运动学计算"""
        # 使用与模型一致的底盘与设备参数
        wheel_base = getattr(self, 'wheel_base', 0.22)
        wheel_radius = getattr(self, 'wheel_radius', 0.043)
        max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
        
        # 逆运动学计算
        left_wheel_speed = (linear_vel - angular_vel * wheel_base / 2) / wheel_radius
        right_wheel_speed = (linear_vel + angular_vel * wheel_base / 2) / wheel_radius
        
        # 速度限制
        left_wheel_speed = np.clip(left_wheel_speed, -max_motor_speed, max_motor_speed)
        right_wheel_speed = np.clip(right_wheel_speed, -max_motor_speed, max_motor_speed)
        
        return left_wheel_speed, right_wheel_speed
    
    def _get_observation(self):
        """获取观察值
        - 当 obs_mode == 'local_map' 时，返回字典{'local_map', 可选'res'}
        - 当 obs_mode == 'lidar' 时，返回拼接后的向量（20维LiDAR + 可选信息）
        """
        # 1. 获取LiDAR数据
        lidar_normalized, lidar_data = self._get_lidar_data()
        
        if getattr(self, 'obs_mode', 'local_map') == 'lidar':
            # 构造20维LiDAR子集：从[8,150]与[250,400]各取10个等间距索引
            n = len(lidar_data)
            idx_band1 = np.clip(np.linspace(8, 80, num=5, dtype=int), 0, n-1)
            idx_band2 = np.clip(np.linspace(320, 400, num=5, dtype=int), 0, n-1)
            idx = np.unique(np.concatenate([idx_band1, idx_band2]))
            # 若去重后不足20，重复采样补齐
            if idx.size < 10:
                extra = np.tile(idx, int(np.ceil(10/idx.size)))[:10]
                idx = extra
            lidar_vec = np.asarray([lidar_data[i] for i in idx[:10]], dtype=np.float32)
            lidar_vec = np.clip(lidar_vec, 0.1, 10.0)
            lidar_vec = np.nan_to_num(lidar_vec, nan=0.1)
            parts = [lidar_vec]
            if getattr(self, 'include_navigation_info', True):
                if getattr(self, 'nav_info_mode', 'minimal') == 'minimal':
                    dist, _ = self._calculate_distance_to_target()
                    ang = self._calculate_angle_to_target(self.task_info['target_pos'])
                    navigation_info = np.array([float(dist), float(ang)], dtype=np.float32)
                else:
                    nav_info = self._get_navigation_info()
                    heading_info = self._get_heading_info()
                    navigation_info = np.concatenate([nav_info, heading_info]).astype(np.float32)
                parts.append(navigation_info)
            if getattr(self, 'include_robot_state', False):
                parts.append(self._get_supervisor_pose_state().astype(np.float32))
            vector_obs = np.concatenate(parts).astype(np.float32)
            return vector_obs, lidar_data
        else:
            # 2. 生成局部地图
            robot_pos = self._get_sup_position()
            robot_orientation = self._get_sup_orientation()
            robot_pose = (robot_pos[0], robot_pos[1], robot_orientation[2])
            local_map = self.local_map_obs.lidar_to_local_map(lidar_data, robot_pose)
            if local_map.ndim == 2:
                local_map = local_map[np.newaxis, :, :]
            observation = {'local_map': local_map.astype(np.float32)}
            if getattr(self, 'include_robot_state', False):
                observation['robot_state'] = self._get_supervisor_pose_state().astype(np.float32)
            if getattr(self, 'include_navigation_info', True):
                if getattr(self, 'nav_info_mode', 'minimal') == 'minimal':
                    dist, _ = self._calculate_distance_to_target()
                    ang = self._calculate_angle_to_target(self.task_info['target_pos'])
                    navigation_info = np.array([float(dist), float(ang)], dtype=np.float32)
                else:
                    nav_info = self._get_navigation_info()
                    heading_info = self._get_heading_info()
                    navigation_info = np.concatenate([nav_info, heading_info]).astype(np.float32)
                observation['navigation_info'] = navigation_info
            return observation, lidar_data
    
    def _get_supervisor_pose_state(self):
        """获取基于supervisor的位姿状态 (12维), 模拟amcl_state的输出"""
        position = self._get_sup_position()
        orientation = self._get_sup_orientation()

        # 计算速度
        current_time = self.supervisor.getTime()
        dt = current_time - self.state_buffer['previous_time']
        if dt > 0:
            velocity = (position - self.state_buffer['previous_position']) / dt
        else:
            velocity = self.state_buffer['previous_velocity']

        # 更新状态缓存，用于下次计算速度
        self.state_buffer['previous_position'] = position.copy()
        self.state_buffer['previous_orientation'] = orientation.copy()
        self.state_buffer['previous_velocity'] = velocity.copy()
        self.state_buffer['previous_time'] = current_time

        # 使用odometry估计的加速度
        acceleration = self._estimate_acceleration() # 1维

        # 模拟amcl_state的12维输出
        # 格式: est_pos(3), est_orient(3), vel(3), prev_vel(2), accel(1)
        amcl_state_replacement = np.concatenate([
            position,       # 3维 - 估计位置 (使用真实位置)
            orientation,    # 3维 - 估计姿态 (使用真实姿态)
            velocity,       # 3维 - 估计速度
            self.state_buffer['previous_velocity'][:2],  # 2维 - 历史速度（只取xy）
            acceleration   # 1维 - 加速度（只取线性加速度）
        ])

        return amcl_state_replacement
    
    def _get_lidar_data(self,norm_lidar=False):
        """获取LiDAR数据"""
        try:
            # 真实Webots模式
            ranges = self.lidar.getRangeImage()
            lidar_data = ranges
            if ranges and len(ranges) >= 20:
                # 从8开始每距离6个下标选一个，选10个
                indices1 = np.arange(8, 8+60, 6)
                # 从339开始，每隔6个下标选一个，一共选择20个点作为激光雷达数据输入
                indices2 = np.arange(339, 339+60, 6)
                indices = np.concatenate([indices2,indices1])
                data = np.array([ranges[i] for i in indices])
                # 仅限制上限为10米，不做归一化
                data = np.clip(data, 0.1, 10.0)
                data = np.nan_to_num(data, nan=0.1)
                # 维护最近一次的障碍物最近距离（米）供信息输出/调试
                try:
                    self.min_obstacle_distance = float(np.min(data))
                except Exception:
                    pass
                return data.astype(np.float32),lidar_data
        except AttributeError:
            raise Exception(AttributeError)
    
    def _get_odometry_data(self):
        """获取里程计数据"""
        # 基于轮速计算里程计
        if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
            # 获取四个轮子的速度
            fl_speed = self.fl_motor.getVelocity()
            fr_speed = self.fr_motor.getVelocity()
            rl_speed = self.rl_motor.getVelocity()
            rr_speed = self.rr_motor.getVelocity()
            
            # 计算左右侧平均速度
            left_speed = (fl_speed + rl_speed) / 2.0
            right_speed = (fr_speed + rr_speed) / 2.0
            
            # 逆运动学计算车体速度
            linear_vel, angular_vel = self._calculate_body_velocity(left_speed, right_speed)
            
            # 计算位姿变化（基于上一时刻）
            dt = self.timestep / 1000.0  # 转换为秒
            
            if self.state_buffer['previous_time'] > 0:
                dx = linear_vel * dt * math.cos(self.state_buffer['previous_orientation'][2])
                dy = linear_vel * dt * math.sin(self.state_buffer['previous_orientation'][2])
                dyaw = angular_vel * dt
            else:
                dx, dy, dyaw = 0.0, 0.0, 0.0
            
            return {
                'dx': dx, 'dy': dy, 'dyaw': dyaw,
                'linear_velocity': linear_vel,
                'angular_velocity': angular_vel,
                'left_speed': left_speed,
                'right_speed': right_speed
            }
        
        return {'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0, 'linear_velocity': 0.0, 'angular_velocity': 0.0}
    
    def _calculate_body_velocity(self, left_speed, right_speed):
        """计算车体速度（正运动学）"""
        wheel_radius = 0.043  # 轮径 - 与_diff_drive_kinematics中保持一致
        wheel_base = 0.22   # 轴距 - 与_diff_drive_kinematics中保持一致
        
        linear_vel = (left_speed + right_speed) * wheel_radius / 2.0
        angular_vel = (right_speed - left_speed) * wheel_radius / wheel_base
        
        return linear_vel, angular_vel
    
    def _estimate_acceleration(self):
        """估计加速度（只返回线性加速度）"""
        dt = self.timestep / 1000.0
        
        if dt > 0 and self.state_buffer['previous_time'] > 0:
            # 使用里程计的线速度作为当前速度，状态缓存中的为上一时刻速度
            try:
                odom = self._get_odometry_data()
                current_vel = float(odom.get('linear_velocity', 0.0))
            except Exception:
                current_vel = float(np.linalg.norm(self.state_buffer['previous_velocity'][:2]))
            prev_vel = float(np.linalg.norm(self.state_buffer['previous_velocity'][:2]))
            linear_acc = (current_vel - prev_vel) / dt
        else:
            linear_acc = 0.0
        
        # 只返回线性加速度（1维）
        return np.array([linear_acc], dtype=np.float32)
    
    def _get_navigation_info(self):
        """获取导航目标信息 (6维)"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']  # 使用AMCL位置
        target_pos = self.task_info['target_pos']
        start_pos = self.task_info['start_pos']
        
        # 目标相对位置
        relative_target = target_pos - current_pos
        
        return np.concatenate([
            relative_target,  # 3维
            start_pos,        # 3维
        ]).astype(np.float32)
    
    def _calculate_angular_acceleration(self):
        """计算角加速度"""
        dt = self.timestep / 1000.0  # 转换为秒
        
        if dt > 0 and self.state_buffer['previous_time'] > 0:
            # 当前角速度
            # current_angular_vel = self.amcl_result['velocity_estimated'][2]
            odometry_data = self._get_odometry_data()
            current_angular_vel = odometry_data['angular_velocity']
            # 上一时刻角速度
            prev_angular_vel = self.state_buffer['previous_velocity'][2]
            # 角加速度
            angular_acc = (current_angular_vel - prev_angular_vel) / dt
            return angular_acc
        
        return 0.0
    
    def _get_heading_info(self):
        """获取航向控制信息 (3维: 航向误差, 目标航向角, 角加速度)"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        current_orient = self._get_sup_orientation() # self.amcl_result['orientation_estimated']
        target_pos = self.task_info['target_pos']
        
        # 计算当前到目标的向量
        target_vector = target_pos - current_pos
        target_heading = math.atan2(target_vector[1], target_vector[0])
        
        # 当前航向角（使用估计姿态）
        current_heading = current_orient[2]  # yaw角
        
        # 航向偏差
        heading_error = target_heading - current_heading
        # 归一化到[-π, π]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        print(f"current_heading: {current_heading}, target_heading: {target_heading}, heading_error: {heading_error}")
        # 角加速度（保留），角速度不再纳入观测
        angular_acceleration = self._calculate_angular_acceleration()
        
        return np.array([
            heading_error,
            target_heading, 
            angular_acceleration
        ], dtype=np.float32)
    

    def _calculate_reward(self, action, observation):
        """使用RewardFunctions类计算奖励"""
        # 在本步开始时仅计算一次里程计、线速度与加速度，后续复用
        try:
            odom = self._get_odometry_data()
        except Exception:
            odom = {'linear_velocity': 0.0, 'angular_velocity': 0.0}
        # 线速度（m/s）
        try:
            current_linear_vel = float(odom.get('linear_velocity', 0.0))
        except Exception:
            current_linear_vel = 0.0
        # 角速度（rad/s）
        try:
            current_angular_vel = float(odom.get('angular_velocity', 0.0))
        except Exception:
            current_angular_vel = 0.0
        dt = max(1e-6, self.timestep / 1000.0)
        prev_linear_vel = float(getattr(self, '_prev_linear_velocity_scalar', 0.0))
        current_linear_acc = (current_linear_vel - prev_linear_vel) / dt
        # 缓存以便 info 与下次计算
        self._prev_linear_velocity_scalar = current_linear_vel
        self._current_linear_vel = current_linear_vel
        self._current_linear_acc = current_linear_acc
        self._current_angular_vel = current_angular_vel

        # 准备环境状态字典，传递给奖励函数
        env_state = {
            'get_sup_position': self._get_sup_position,
            'get_sup_orientation': self._get_sup_orientation,
            'calculate_distance_to_target': self._calculate_distance_to_target,
            'get_lidar_features': self._get_lidar_features,
            # 返回本步缓存的里程计，避免重复查询
            'get_odometry_data': (lambda od=odom: od),
            'calculate_angular_acceleration': self._calculate_angular_acceleration,
            # 返回本步预计算的线性加速度（不再重复计算）
            'estimate_linear_acceleration': (lambda acc=self._current_linear_acc: np.array([acc], dtype=np.float32)),
            'detect_collision_simple': self._detect_collision_simple,
            'episode_steps': self.episode_steps,
            'cargo_type': self.cargo_type,
            'success_threshold': self.success_threshold,
            'task_info': self.task_info,
            'terminate': False,  # 初始化终止标志
            'args': getattr(self, 'args', None),  # 传递训练时的参数对象
            # 暴露上一次生效的左右轮角速度（由执行器估计/限幅后）
            'get_last_wheel_speeds': lambda: (
                float(getattr(self, '_prev_left_speed', 0.0)),
                float(getattr(self, '_prev_right_speed', 0.0))
            )
        }
        
        # 调用奖励函数计算奖励
        reward = self.reward_functions.calculate_reward(action, observation, env_state)
        
        # 检查奖励函数是否设置了终止标志
        self._reward_terminate_flag = env_state.get('terminate', False)
        # 捕获奖励函数暴露的步级统计指标
        try:
            metrics = env_state.get('step_metrics', None)
            if isinstance(metrics, dict):
                self._last_step_metrics = metrics.copy()
            else:
                self._last_step_metrics = {}
        except Exception:
            self._last_step_metrics = {}
        
        return reward
    
    # 货物专用奖励函数已移至reward_functions.py
    
    def _check_termination(self):
        """检查终止条件"""
        # 首先检查奖励函数是否已经决定终止（碰撞或卡住检测）
        if hasattr(self, '_reward_terminate_flag') and self._reward_terminate_flag:
            return True
            
        collision_detected = False

        # 全局步计数与朝向历史初始化
        try:
            # 记录当前步数
            self._global_step = getattr(self, '_global_step', 0) + 1
            # 步数统计变量
            setattr(self, '_global_step', self._global_step)
            # 记录当前步数
            # 滑动窗口用于记录最近几个步骤的偏航角
            if not hasattr(self, '_heading_hist'):
                self._heading_hist = []  # 简单列表作为滑窗
            # 记录当前位置
            if not hasattr(self, '_last_pos'):
                self._last_pos = self._get_sup_position().copy()
            # 上一次联系检查的步数
            if not hasattr(self, '_last_contact_check_step'):
                self._last_contact_check_step = -999999
        except Exception:
            # 若异常，继续执行但不启用条件触发
            self._heading_hist = []
            self._last_pos = self._get_sup_position().copy()
            self._last_contact_check_step = -999999

        # 基于平面运动估计朝向，并维护滑动窗口
        try:
            curr_pos = self._get_sup_position()
            dx = float(curr_pos[0] - self._last_pos[0])
            dy = float(curr_pos[1] - self._last_pos[1])
            self._last_pos = curr_pos.copy()
            import math
            # 仅在移动幅度超过极小阈值时更新朝向估计
            if (dx * dx + dy * dy) > 1e-6:
                heading = math.atan2(dy, dx)
                self._heading_hist.append(heading)
                if len(self._heading_hist) > 10:
                    self._heading_hist = self._heading_hist[-10:]
        except Exception:
            pass

        # # 简化：固定频率检查接触点，避免漏检（每步最多检查一次）
        # should_check_contacts = True
        # try:
        #     # 防止同一时间步重复查询
        #     if (self._global_step - self._last_contact_check_step) < 1:
        #         should_check_contacts = False
        # except Exception:
        #     should_check_contacts = True

        # # 1. selfCollision检测 (Webots) - 仅在触发条件满足时进行昂贵查询
        # if should_check_contacts and self.robot_node:
        #     try:
        #         contact_points = self.robot_node.getContactPoints(includeDescendants=True)
        #         self._last_contact_check_step = self._global_step
        #         for cp in contact_points:
        #             other_node_id = cp.getNodeId()
        #             contact_z_height = cp.getPoint()[2]
        #             other_node = self.supervisor.getFromId(other_node_id)
        #             if other_node is None:
        #                 continue
        #             other_node_name = ""
        #             name_field = other_node.getField("name")
        #             if name_field:
        #                 other_node_name = name_field.getSFString()
        #             # 若缺省名称，尝试使用 DEF 名称作为回退
        #             if not other_node_name:
        #                 try:
        #                     other_node_name = other_node.getDef() or ""
        #                 except Exception:
        #                     other_node_name = ""
        #             # 正常车轮-地面接触判定（名称统一为小写比较）
        #             other_label = other_node_name.strip().lower()
        #             ground_set = set(s.strip().lower() for s in getattr(self, 'ground_defs', {'floor'}))
        #             is_ground_contact = other_label in ground_set
        #             # 放宽地面高度容差，避免由于数值抖动导致误判
        #             is_at_floor_level = abs(contact_z_height) < 0.02
        #             if is_ground_contact and is_at_floor_level:
        #                 continue
        #             # 其余接触一律视为碰撞
        #             try:
        #                 self._last_collision_info = {
        #                     'type': 'contact_point',
        #                     'node': other_label,
        #                     'z': float(contact_z_height),
        #                     'step': int(self._global_step)
        #                 }
        #             except Exception:
        #                 self._last_collision_info = {'type': 'contact_point'}
        #             collision_detected = True
        #             break
        #     except Exception as e:
        #         # 静默或降频打印
        #         # print(f"通过getContactPoints检测碰撞时发生错误: {e}")
        #         pass

        # # 回退：若接触点未检测到碰撞，使用距离传感器阈值作为补充
        # if not collision_detected:
        #     try:
        #         threshold = getattr(self, 'collision_distance_threshold', 0.05)
        #         for sensor in getattr(self, 'collision_sensors', []) or []:
        #             value = sensor.getValue()
        #             if value is not None and value < threshold:
        #                 try:
        #                     self._last_collision_info = {
        #                         'type': 'proximity',
        #                         'sensor': getattr(sensor, 'getName', lambda: 'unknown')(),
        #                         'value': float(value),
        #                         'threshold': float(threshold),
        #                         'step': int(self._global_step)
        #                     }
        #                 except Exception:
        #                     self._last_collision_info = {'type': 'proximity'}
        #                 collision_detected = True
        #                 break
        #     except Exception:
        #         pass

        # if collision_detected:
        #     try:
        #         self._debug(f"termination: collision detected, info={getattr(self, '_last_collision_info', None)}")
        #     except Exception:
        #         pass
        #     return True
        
        # 2. 到达目标
        current_pos = self._get_sup_position()
        current_distance = np.linalg.norm(current_pos - self.task_info['target_pos'])
        if current_distance < self.success_threshold:
            try:
                self._debug(f"termination: success, distance={current_distance:.4f} < {self.success_threshold}")
            except Exception:
                pass
            return True
        
        # 3. 新增：原地打转终止条件
        if self.reward_functions.is_excessive_spin():
            try:
                self._debug("termination: excessive spin detected")
            except Exception:
                pass
            return True
        
        # 4. 超时/步数限制
        if len(self.trajectory) > self.max_steps_per_episode:
            try:
                self._debug(f"termination: max steps exceeded ({len(self.trajectory)} > {self.max_steps_per_episode})")
            except Exception:
                pass
            return True
            
        return False


    def _check_truncation(self):
        """检查截断条件"""
        # 位置边界检查
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        if abs(current_pos[0]) > 20 or abs(current_pos[1]) > 20:
            try:
                self._debug(f"truncation: out_of_bounds pos=({current_pos[0]:.2f},{current_pos[1]:.2f})")
            except Exception:
                pass
            return True
            
        return False
    
    def _get_step_info(self):
        """获取步信息"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        target_pos = self.task_info['target_pos']
        
        # 计算距离信息
        distance_to_target = float(np.linalg.norm(current_pos - target_pos))
        
        info = {
            'position': current_pos.tolist(),
            'target_position': target_pos.tolist(),
            'distance_to_target': distance_to_target,
            'amcl_uncertainty': 0.0, # float(self.amcl_result.get('position_uncertainty', 0.1)),
            'min_obstacle_distance': float(self.min_obstacle_distance),
            'trajectory_length': len(self.trajectory),
            'close_to_target': distance_to_target < 0.5,  # 接近目标标志
            'very_close_to_target': distance_to_target < 0.25,  # 非常接近目标
            'success': distance_to_target < self.success_threshold,  # 目标成功标志（按课程参数第三项）
            'cargo_type': self.cargo_type,
            'last_collision': getattr(self, '_last_collision_info', None),
            'collision': self._detect_collision_simple
        }
        # 合并奖励函数传回的步级统计指标
        try:
            if isinstance(getattr(self, '_last_step_metrics', None), dict):
                # 仅拷贝关心的字段，避免污染info
                if 'linear_acc' in self._last_step_metrics:
                    info['linear_acc'] = float(self._last_step_metrics['linear_acc'])
                if 'wall_proximity_raw' in self._last_step_metrics:
                    info['wall_proximity_raw'] = float(self._last_step_metrics['wall_proximity_raw'])
        except Exception:
            pass
        # 将本步缓存的线速度/加速度直接写入info，供回调高效使用
        try:
            info['linear_vel'] = float(getattr(self, '_current_linear_vel', 0.0))
            if 'linear_acc' not in info:
                info['linear_acc'] = float(abs(getattr(self, '_current_linear_acc', 0.0)))
            info['angular_vel'] = float(getattr(self, '_current_angular_vel', 0.0))
        except Exception:
            pass
        
        return info

    def _detect_collision_simple(self) -> bool:
        """轻量碰撞检测：
        1) 使用接触点排除正常的轮-地面接触
        2) 回退使用红外距离传感器阈值
        """
        # 方法1：接触点检测
        try:
            if self.robot_node:
                contact_points = self.robot_node.getContactPoints(includeDescendants=True)
                for cp in contact_points:
                    other_node_id = cp.getNodeId()
                    contact_z_height = cp.getPoint()[2]
                    other_node = self.supervisor.getFromId(other_node_id)
                    if other_node is None:
                        continue
                    other_node_name = ""
                    name_field = other_node.getField("name")
                    if name_field:
                        other_node_name = name_field.getSFString()
                    if not other_node_name:
                        try:
                            other_node_name = other_node.getDef() or ""
                        except Exception:
                            other_node_name = ""
                    other_label = other_node_name.strip().lower()
                    ground_set = set(s.strip().lower() for s in getattr(self, 'ground_defs', {'floor'}))
                    is_ground_contact = other_label in ground_set
                    is_at_floor_level = abs(contact_z_height) < 0.02
                    # 过滤正常的轮-地面接触
                    if is_ground_contact and is_at_floor_level:
                        continue
                    return True
        except Exception:
            pass

        # 方法2：距离传感器阈值
        try:
            threshold = getattr(self, 'collision_distance_threshold', 0.05)
            for sensor in getattr(self, 'collision_sensors', []) or []:
                value = sensor.getValue()
                if value is not None and value < threshold:
                    return True
        except Exception:
            pass

        return False
    
    def get_current_pose(self):
        """获取当前位姿"""
        # 使用supervisor数据
        pos = self._get_sup_position()
        orient = self._get_sup_orientation()
        # 从里程计获取速度
        odometry_data = self._get_odometry_data()
        vel = np.array([odometry_data['linear_velocity'], 0, odometry_data['angular_velocity']])

        return {
            'position': pos,
            'orientation': orient,
            'velocity': vel
        }
        # return {
        #     'position': self.amcl_result['position_estimated'].copy(),
        #     'orientation': self.amcl_result['orientation_estimated'].copy(),
        #     'velocity': self.amcl_result['velocity_estimated'].copy()
        # }
    
    def set_training_mode(self, mode: bool):
        """设置训练模式"""
        self.is_training = mode
        # self.amcl_localizer.set_training_mode(mode)

    def close(self):
        """释放资源并尽量优雅地停止控制器"""
        try:
            print("关闭环境...")
            if hasattr(self, 'supervisor') and self.supervisor and hasattr(self.supervisor, 'simulationSetMode') and hasattr(Supervisor, 'SIMULATION_MODE_PAUSE'):
                self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            print("环境关闭完成")
        except Exception:
            pass