"""
局部地图观测空间类
将激光雷达信息转换为局部地图，地图分辨率5cm
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Tuple, Optional
import cv2


class LocalMapObservation:
    """
    局部地图观测空间类
    将激光雷达数据转换为局部地图表示
    """
    
    def __init__(self, 
                 map_size: int = 200,  # 地图大小 (像素)
                 resolution: float = 0.1,  # 分辨率 (米/像素)
                 max_range: float = 10.0,  # 最大探测距离 (米)
                 align_with_robot_pose: bool = False):  # 是否按机器人位姿对齐
        """
        初始化局部地图观测空间
        
        Args:
            map_size: 地图大小 (像素)，默认200x200
            resolution: 分辨率 (米/像素)，默认0.05m
            max_range: 最大探测距离 (米)，默认10m
        """
        self.map_size = map_size
        self.resolution = resolution
        self.max_range = max_range
        self.align_with_robot_pose = bool(align_with_robot_pose)
        
        # 地图物理尺寸 (米)
        self.map_physical_size = map_size * resolution  # 10m x 10m
        
        # 创建观测空间 - 与训练时保持一致，使用ROS风格占用栅格值范围
        self.observation_space = spaces.Box(
            low=-1.0, high=100.0, 
            shape=(1, map_size, map_size), 
            dtype=np.float32
        )
        
        # 为了兼容性，也提供扁平化的观测空间
        self.flat_observation_space = spaces.Box(
            low=-1.0, high=100.0,
            shape=(map_size * map_size,),
            dtype=np.float32
        )
    
    def lidar_to_local_map(self, 
                           lidar_ranges: np.ndarray, 
                           robot_pose: Tuple[float, float, float]) -> np.ndarray:
        """
        将激光雷达数据转换为局部地图
        
        Args:
            lidar_ranges: 激光雷达距离数据 (米为单位，范围8-399)
            robot_pose: 机器人位姿 (x, y, theta)
            
        Returns:
            local_map: 局部地图 (1, map_size, map_size)
        """
        # 初始化地图 (0表示未知，1表示占用)
        local_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # 机器人在地图中的位置 (中心)
        robot_x_pixel = self.map_size // 2
        robot_y_pixel = self.map_size // 2
        
        # 防御性编程：处理空或无效的激光数据
        if lidar_ranges is None:
            return local_map.reshape(1, self.map_size, self.map_size)
        
        # 确保为一维数组
        lidar_ranges = np.asarray(lidar_ranges).reshape(-1)
        
        # 激光雷达参数 - 8-399范围，逆时针旋转一圈
        num_rays = len(lidar_ranges)
        if num_rays == 0:
            return local_map.reshape(1, self.map_size, self.map_size)
        angle_min = -math.pi  # 起始角度
        angle_max = math.pi   # 结束角度
        angle_increment = (angle_max - angle_min) / num_rays
        
        # 处理每条激光束
        for i, range_val in enumerate(lidar_ranges):
            # 跳过非有限值 (NaN, inf)
            if not np.isfinite(range_val):
                continue
            # 无效数据或超出范围
            if range_val < 0.01 or range_val > self.max_range:
                continue
                
            # 计算激光束角度 (相对于机器人朝向)
            angle = angle_min + i * angle_increment
            if self.align_with_robot_pose:
                # 使用机器人朝向旋转射线
                world_angle = angle + robot_pose[2]
                dx = range_val * math.cos(world_angle)
                dy = range_val * math.sin(world_angle)
            else:
                # 忽略机器人朝向与位置，固定以机器人中心为原点，角度不随姿态变化
                dx = range_val * math.cos(angle)
                dy = range_val * math.sin(angle)
            
            # 如果由于数值问题导致dx/dy为非有限值，直接跳过
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            
            # 转换为相对于机器人中心的地图像素坐标（四舍五入以减少系统性偏差）
            end_x_pixel = int(round((dx + self.map_physical_size/2) / self.resolution))
            end_y_pixel = int(round((dy + self.map_physical_size/2) / self.resolution))
            
            # 确保坐标在地图范围内
            end_x_pixel = np.clip(end_x_pixel, 0, self.map_size - 1)
            end_y_pixel = np.clip(end_y_pixel, 0, self.map_size - 1)
            
            # 使用Bresenham算法绘制激光束路径
            self._draw_line(local_map, robot_x_pixel, robot_y_pixel, 
                          end_x_pixel, end_y_pixel, range_val)
        
        # 在地图中心标记机器人位置
        local_map[robot_x_pixel-1:robot_x_pixel+2, robot_y_pixel-1:robot_y_pixel+2] = 0.5
        
        return local_map.reshape(1, self.map_size, self.map_size)
    
    def _draw_line(self, 
                   map_array: np.ndarray, 
                   x0: int, y0: int, 
                   x1: int, y1: int, 
                   range_val: float):
        """
        使用Bresenham算法在地图上绘制直线
        """
        # 计算距离衰减因子
        distance_factor = min(1.0, range_val / self.max_range)
        
        # Bresenham算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # 在地图上标记占用 (距离越近，占用值越高)
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                # 使用距离衰减，距离越近占用值越高
                occupancy_value = (1.0 - distance_factor) * 0.8 + 0.2
                map_array[y, x] = max(map_array[y, x], occupancy_value)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def create_observation(self, 
                          lidar_data: np.ndarray,
                          robot_pose: Tuple[float, float, float]) -> np.ndarray:
        """
        创建完整的观测
        
        Args:
            lidar_data: 激光雷达数据
            robot_pose: 机器人位姿
            
        Returns:
            observation: 局部地图观测
        """
        # 生成局部地图
        local_map = self.lidar_to_local_map(lidar_data, robot_pose)
        
        return local_map
    
    def flatten_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        将观测扁平化为向量
        
        Args:
            observation: 地图观测
            
        Returns:
            flattened_obs: 扁平化的观测向量
        """
        return observation.flatten().astype(np.float32)
    
    def unflatten_observation(self, flattened_obs: np.ndarray) -> np.ndarray:
        """
        将扁平化的观测恢复为地图格式
        
        Args:
            flattened_obs: 扁平化的观测向量
            
        Returns:
            observation: 地图观测
        """
        return flattened_obs.reshape(1, self.map_size, self.map_size)
    
    def render_map_to_cv_image(self, local_map: np.ndarray) -> np.ndarray:
        """
        将局部地图转换为OpenCV图像格式
        
        Args:
            local_map: 局部地图 (1, H, W) 或 (H, W) - ROS风格占用栅格
            
        Returns:
            cv_image: OpenCV图像 (H, W, 3) BGR格式
        """
        # 确保地图是2D的
        if len(local_map.shape) == 3:
            map_2d = local_map[0]  # 取第一个通道
        else:
            map_2d = local_map
        
        # 将ROS风格占用栅格转换为灰度图像
        # -1(未知) -> 128, 0(自由) -> 255, 100(占用) -> 0
        gray_map = np.zeros_like(map_2d, dtype=np.uint8)
        gray_map[map_2d == -1] = 128  # 未知区域为灰色
        gray_map[map_2d == 0] = 255   # 自由空间为白色
        gray_map[map_2d == 100] = 0   # 占用区域为黑色
        
        # 转换为3通道BGR图像
        cv_image = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)
        
        return cv_image


class LocalMapNavigationEnv:
    """
    局部地图导航环境包装器
    将原有的导航环境包装为使用局部地图观测
    """
    
    @staticmethod
    def get_spaces(map_size: int = 200):
        """
        无需连接 Webots，返回与环境一致的观测/动作空间定义
        
        Args:
            map_size: 地图大小 (像素)
            
        Returns:
            obs_space: 观测空间 (局部地图)
            act_space: 动作空间 (与原环境相同)
        """
        # 局部地图观测空间 - 图像格式 (1, H, W) 用于CNN，ROS风格占用栅格
        obs_space = spaces.Box(
            low=-1.0, 
            high=100.0, 
            shape=(1, map_size, map_size), 
            dtype=np.float32
        )
        
        # 动作空间保持不变 - 10维（5个子动作，每个子动作2维：左右轮速度百分比，范围0-1）
        act_space = spaces.Box(
            low=np.array([0.0, 0.0] * 5, dtype=np.float32),
            high=np.array([1.0, 1.0] * 5, dtype=np.float32),
            dtype=np.float32
        )
        
        return obs_space, act_space
    
    def __init__(self, 
                 base_env,
                 map_size: int = 200,
                 resolution: float = 0.05,
                 max_range: float = 10.0):
        """
        初始化局部地图导航环境
        
        Args:
            base_env: 基础导航环境
            map_size: 地图大小
            resolution: 分辨率
            max_range: 最大探测距离
        """
        self.base_env = base_env
        self.local_map_obs = LocalMapObservation(
            map_size=map_size,
            resolution=resolution,
            max_range=max_range
        )
        
        # 使用图像格式的观测空间以兼容CNN策略
        self.observation_space = self.local_map_obs.observation_space
        self.action_space = base_env.action_space
        
        # 保存原始环境的方法
        self._get_sup_position = base_env._get_sup_position
        self._get_sup_orientation = base_env._get_sup_orientation
        self._get_lidar_data = base_env._get_lidar_data
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, info,lidar_data = self.base_env.reset(seed=seed, options=options)
        return self._convert_observation(lidar_data), info,lidar_data
    
    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info,lidar_data = self.base_env.step(action)
        return self._convert_observation(lidar_data), reward, terminated, truncated, info
    
    def _convert_observation(self, lidar_data: np.ndarray) -> np.ndarray:
        """
        将原始观测转换为局部地图观测
        
            
        Returns:
            local_map_obs: 局部地图观测
        """
        # 获取原始激光雷达数据 (米为单位，范围8-399)
        raw_lidar_data = lidar_data
        
        # 获取机器人位姿
        robot_pos = self._get_sup_position()
        robot_orient = self._get_sup_orientation()
        robot_pose = (robot_pos[0], robot_pos[1], robot_orient[2])
        
        # 创建局部地图观测
        observation = self.local_map_obs.create_observation(
            raw_lidar_data, robot_pose
        )
        
        # 返回图像格式的观测 (1, H, W)
        return observation
    
    def render_map(self) -> np.ndarray:
        """
        渲染当前局部地图为OpenCV图像
        
        Returns:
            cv_image: OpenCV图像 (H, W, 3) BGR格式
        """
        # 获取原始激光雷达数据
        _,raw_lidar_data = self._get_lidar_data()
        
        # 获取机器人位姿
        robot_pos = self._get_sup_position()
        robot_orient = self._get_sup_orientation()
        robot_pose = (robot_pos[0], robot_pos[1], robot_orient[2])
        
        # 创建局部地图
        local_map = self.local_map_obs.create_observation(raw_lidar_data, robot_pose)
        
        # 转换为OpenCV图像
        return self.local_map_obs.render_map_to_cv_image(local_map)
    
    def close(self):
        """关闭环境"""
        return self.base_env.close()
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.base_env.render(mode)
    
    def seed(self, seed=None):
        """设置随机种子"""
        return self.base_env.seed(seed)
    
    def __getattr__(self, name):
        """代理其他属性到基础环境"""
        return getattr(self.base_env, name)
