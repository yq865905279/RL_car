"""
导航工具类 - 提供任务配置和导航辅助功能
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import random
import json
from pathlib import Path
import math


class NavigationUtils:
    """导航工具类"""
    
    def __init__(self):
        # 固定的位置坐标
        self.fixed_positions = {
            'normal_start': [3.0, 0.2, 0.0],
            'unload_start': [-3.0, -2.0, 0.0],
            'start': [-5.0, 3.0, 0.0],       # 起点
            'unload': [-5.0, -2.0, 0.0],     # 卸货点
            'dangerous': [3.0, 3.0, 0.0],    # 危险货物点
            'fragile': [3.0, 1.7, 0.0],      # 易碎货物点
            'normal': [3.0, 0.2, 0.0],       # 普通货物点
            'smaller': [-0.75, -0.55, 0.0]   # 小环境卸货点
        }

        # 区域定义（与现有代码兼容，提供别名）
        self.difficulty_position_areas = {
            'start_area': {
                'x_min': -5.5, 'x_max': -4.5,
                'y_min': 2.5, 'y_max': 3.5,
                'z': 0.0
            },
            'unload_area': {
                'x_min': -5.5, 'x_max': -4.5,
                'y_min': -2.5, 'y_max': -1.5,
                'z': 0.0
            },
            'dangerous_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': 2.5, 'y_max': 3.5,
                'z': 0.0
            },
            'fragile_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': 1.2, 'y_max': 2.2,
                'z': 0.0
            },
            'normal_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': -0.3, 'y_max': 0.7,
                'z': 0.0
            }
        }

        # 兼容别名，避免KeyError
        self.position_areas = self.difficulty_position_areas

        self.woodenbox_area_env1 = {
            'forbidden_zones': [
                {'center': [-0.531454, 1.8261, 0.0], 'radius': 0.45},
                {'center': [0.80243, 0.972897, 0.0], 'radius': 0.45},
                {'center': [1.75897, -1.68178, 0.0], 'radius': 0.45},
                {'center': [2.11678, 1.97471, 0.0], 'radius': 0.45},
                {'center': [1.00292, 2.79978, 0.0], 'radius': 0.45},
                {'center': [2.31996, 0.102915, 0.0], 'radius': 0.45},
                {'center': [-0.875248, 0.292079, 0.0], 'radius': 0.45},
                {'center': [0.824845, -0.581012, 0.0], 'radius': 0.45},
                {'center': [-2.57865, 0.177985, 0.0], 'radius': 0.45},
                {'center': [-1.57535, 3.02372, 0.0], 'radius': 0.45},
                {'center': [-2.44712, 1.63647, 0.0], 'radius': 0.45},
                {'center': [-0.331992, -1.4555, 0.0], 'radius': 0.45},
                {'center': [-2.03615, -1.49171, 0.0], 'radius': 0.45},
                # WoodenPalletStack (from warehouse5_env1.wbt)
                {'center': [5.00669, -3.36607, 0.0], 'radius': 0.8},
                {'center': [3.55506, -3.30287, 0.0], 'radius': 0.8}
            ]
        }

        self.woodenbox_area_env2 = {
            'forbidden_zones': [
                {'center': [-0.691454, 2.6961, 0.0], 'radius': 0.45},
                {'center': [0.99243, 0.132897, 0.0], 'radius': 0.45},
                {'center': [0.22897, -1.86178, 0.0], 'radius': 0.45},
                {'center': [1.37678, 1.97471, 0.0], 'radius': 0.45},
                {'center': [-0.34708, 0.96978, 0.0], 'radius': 0.45},
                {'center': [2.35996, -0.757085, 0.0], 'radius': 0.45},
                {'center': [-2.42525, 0.182079, 0.0], 'radius': 0.45},
                {'center': [-0.81516, -0.751012, 0.0], 'radius': 0.45},
                {'center': [-2.20865, 2.00798, 0.0], 'radius': 0.45},
                {'center': [-3.21535, 3.35372, 0.0], 'radius': 0.45},
                {'center': [-3.85712, 1.68647, 0.0], 'radius': 0.45},
                {'center': [-3.66199, -1.3955, 0.0], 'radius': 0.45},
                {'center': [-4.60615, -0.041712, 0.0], 'radius': 0.45},
                # WoodenPalletStack (from warehouse5_env1.wbt)
                {'center': [5.00669, -3.36607, 0.0], 'radius': 0.8},
                {'center': [3.55506, -3.30287, 0.0], 'radius': 0.8}
            ]
        }


        
        # 额外矩形启动禁区（两环境一致）：x∈[4.9, 6.0], y∈[-0.9, 3.9]
        self.forbidden_rects_env1 = {
            'rects': [
                {'x_min': 3.7, 'x_max': 6.0, 'y_min': -0.5, 'y_max': 3.9}
            ]
        }
        self.forbidden_rects_env2 = {
            'rects': [
                {'x_min': 3.7, 'x_max': 6.0, 'y_min': -0.5, 'y_max': 3.9}
            ]
        }

        # 导航任务配置 - 根据货物类型分不同的取货点和目的地
        self.navigation_config = {
            'normal': self._get_normal_cargo_tasks(),
            'fragile': self._get_fragile_cargo_tasks(),
            'dangerous': self._get_dangerous_cargo_tasks()
        }
        
        # 环境边界
        self.env_bounds = {
            'x_min': -6.0, 'x_max': 6.0,
            'y_min': -4.0, 'y_max': 4.0,
            'z_min': 0, 'z_max': 0
        }
        
        # 是否使用随机目标点（end阶段时仍可生效）
        self.use_random_targets = True

        # 课程学习阶段：start/easy/medium/hard/end/all
        self.curriculum_stage: str = 'end'
    
    def _generate_random_position_in_area(self, area_name: str) -> List[float]:
        """从指定区域生成随机位置"""
        if area_name not in self.position_areas:
            # 如果区域不存在，返回对应的固定位置
            area_name = area_name.replace('_area', '')
            return self.fixed_positions.get(area_name, [0, 0, 0])
        
        area = self.position_areas[area_name]
        x = random.uniform(area['x_min'], area['x_max'])
        y = random.uniform(area['y_min'], area['y_max'])
        z = area['z']
        
        return [x, y, z]
    
    def get_navigation_task(self, cargo_type: str) -> Tuple[List[float], List[float]]:
        """
        根据货物类型和任务阶段获取导航任务
        
        参数:
            cargo_type: 货物类型 ('normal', 'fragile', 'dangerous')
            difficulty_type: 任务难易程度 ('start', 'easy', 'medium', 'hard', 'hard2')
            task_stage: 任务阶段
                - 'base': 基础模型训练，从固定起点到各个货物点，再从普通货物点到卸货点
                - 'dangerous_to_unload': 从危险货物点到卸货点
                - 'fragile_to_unload': 从易碎货物点到卸货点
                - 'normal_to_unload': 从普通货物点到卸货点
        
        返回:
            起点和终点坐标元组
        """
        
        if cargo_type not in self.navigation_config:
            cargo_type = 'normal'  # 默认为普通货物

        self.use_random_targets = False
        all_start=True

        # 根据任务阶段选择起点和终点
        if cargo_type == 'normal':
            if random.random() < 0.5:
                # 从起点到各个货物点
                if self.use_random_targets:
                    start_pos = self._generate_random_position_in_area('start_area')
                    # 随机选择一个货物点类型作为终点
                    target_type = random.choice(['dangerous', 'fragile', 'normal'])
                    target_pos = self._generate_random_position_in_area(f'{target_type}_area')
                else:
                    start_pos = self.fixed_positions['start']
                    target_type = random.choice(['dangerous', 'fragile', 'normal'])
                    target_pos = self.fixed_positions[target_type]
            else:
                # 从普通货物点到卸货点
                if self.use_random_targets:
                    start_pos = self._generate_random_position_in_area('normal_area')
                    target_pos = self.fixed_positions['unload']
                    #target_pos = self._generate_random_position_in_area('unload_area')
                else:
                    start_type = random.choice(['dangerous', 'fragile', 'normal'])
                    start_pos = self.fixed_positions[start_type]
                    target_pos = self.fixed_positions['unload']
        
        elif cargo_type == 'dangerous':
            # 从危险货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('dangerous_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['dangerous']
                target_pos = self.fixed_positions['unload']
        
        elif cargo_type == 'fragile':
            # 从易碎货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('fragile_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['fragile']
                target_pos = self.fixed_positions['unload']
        
        elif cargo_type == 'normal_to_unload':
            # 从普通货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('normal_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['normal']
                target_pos = self.fixed_positions['unload']
        
        else:
            # 默认：从固定起点到普通货物点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('start_area')
                target_pos = self._generate_random_position_in_area('normal_area')
            else:
                start_pos = self.fixed_positions['start']
                target_pos = self.fixed_positions['normal']
        
        return start_pos, target_pos
    
    def _get_normal_cargo_tasks(self) -> List[Dict]:
        """普通货物导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['normal'],
                'destination': self.fixed_positions['unload'],
                'description': '普通货物：从固定起点到普通货物点'
            },
            {
                'start_pos': self.fixed_positions['normal'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '普通货物：从普通货物点到卸货点'
            }
        ]
    
    def _get_fragile_cargo_tasks(self) -> List[Dict]:
        """易碎品导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['fragile'],
                'destination': self.fixed_positions['unload'],
                'description': '易碎品：从固定起点到易碎货物点'
            },
            {
                'start_pos': self.fixed_positions['fragile'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '易碎品：从易碎货物点到卸货点'
            }
        ]
    
    def _get_dangerous_cargo_tasks(self) -> List[Dict]:
        """危险品导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['dangerous'],
                'destination': self.fixed_positions['unload'],
                'description': '危险品：从固定起点到危险货物点'
            },
            {
                'start_pos': self.fixed_positions['dangerous'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '危险品：从危险货物点到卸货点'
            }
        ]
    
    def is_position_valid(self, position: List[float]) -> bool:
        """检查位置是否在有效范围内"""
        x, y, z = position
        return (self.env_bounds['x_min'] <= x <= self.env_bounds['x_max'] and
                self.env_bounds['y_min'] <= y <= self.env_bounds['y_max'] and
                self.env_bounds['z_min'] <= z <= self.env_bounds['z_max'])

    # ===== WoodenBox/桌子/货架 启动禁区支持 =====
    # 当前环境，默认 env1，可在外部根据使用的 world 切换为 'env2'
    current_env: str = 'env1'

    def _get_forbidden_zones(self) -> List[Dict[str, List[float]]]:
        """根据当前环境返回禁区列表"""
        if getattr(self, 'current_env', 'env1') == 'env2':
            data = getattr(self, 'woodenbox_area_env2', None)
        else:
            data = getattr(self, 'woodenbox_area_env1', None)
        if isinstance(data, dict):
            return data.get('forbidden_zones', [])
        return []

    def _get_forbidden_rects(self) -> List[Dict[str, float]]:
        """矩形禁区（桌子、货架）"""
        if getattr(self, 'current_env', 'env1') == 'env2':
            data = getattr(self, 'forbidden_rects_env2', None)
        else:
            data = getattr(self, 'forbidden_rects_env1', None)
        if isinstance(data, dict):
            return data.get('rects', [])
        return []

    def _is_in_forbidden_zone(self, position: List[float]) -> bool:
        """判断给定位置是否落入任一禁区（WoodenBox 圆形禁区 + 桌子/货架矩形禁区）"""
        x, y, _ = position
        for zone in self._get_forbidden_zones():
            cx, cy, _ = zone['center']
            r = zone['radius']
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True
        for rect in self._get_forbidden_rects():
            if rect['x_min'] <= x <= rect['x_max'] and rect['y_min'] <= y <= rect['y_max']:
                return True
        return False

    # ===== 采样辅助 =====
    def _sample_in_circle_avoid(self, center: List[float], radius: float, align_to_target: bool = False) -> List[float]:
        """在圆内采样，避开禁区与边界"""
        cx, cy, cz = center
        for _ in range(200):
            angle = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.uniform(0, 1))  # 均匀分布
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            candidate = [x, y, cz]
            if self.is_position_valid(candidate) and not self._is_in_forbidden_zone(candidate):
                return candidate
        # 回退：返回靠近边界的安全点或目标附近点
        return [cx, cy, cz]

    def _sample_in_rect_avoid(self, rect: Dict[str, float]) -> List[float]:
        """在矩形区域内均匀采样且避开禁区"""
        z = 0.0
        for _ in range(200):
            x = random.uniform(rect['x_min'], rect['x_max'])
            y = random.uniform(rect['y_min'], rect['y_max'])
            candidate = [x, y, z]
            if self.is_position_valid(candidate) and not self._is_in_forbidden_zone(candidate):
                return candidate
        # 回退：返回矩形中心
        cx = 0.5 * (rect['x_min'] + rect['x_max'])
        cy = 0.5 * (rect['y_min'] + rect['y_max'])
        return [cx, cy, z]

    def _sample_in_annulus_avoid(self, center: List[float], r_min: float, r_max: float) -> List[float]:
        """在环形区域(r_min, r_max)内采样，避开禁区与边界"""
        cx, cy, cz = center
        r_min = max(0.0, float(r_min))
        r_max = max(r_min, float(r_max))
        for _ in range(300):
            angle = random.uniform(0, 2 * math.pi)
            # 面积均匀的半径采样：sqrt(uniform(r_min^2, r_max^2))
            radius = math.sqrt(random.uniform(r_min * r_min, r_max * r_max))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            candidate = [x, y, cz]
            if self.is_position_valid(candidate) and not self._is_in_forbidden_zone(candidate):
                return candidate
        # 回退：在最大半径圆内采样
        return self._sample_in_circle_avoid(center=center, radius=r_max)

    # ===== 课程学习任务生成 =====
    def set_stage(self, stage: str):
        self.curriculum_stage = stage

    def _choose_target_type(self) -> str:
        # 仅使用普通货物点和卸货点
        return random.choice(['normal', 'unload'])

    def _get_target_pos(self, target_type: str) -> List[float]:
        return self.fixed_positions['normal'] if target_type == 'normal' else self.fixed_positions['unload']

    def get_curriculum_task(self) -> Tuple[List[float], List[float], str]:
        """
        返回(起点, 目标点, 角度模式) 按课程学习阶段生成：
        - start: 从 normal_start 或 unload_start 固定点出发，角度对准目标
        - easy: 目标为 normal/unload 二选一；起点在目标为中心半径4m内，避开禁区，角度对准目标
        - medium: 同 easy，但半径5m，角度随机
        - hard: 同 easy，但半径7m，角度随机
        - end: 使用现有指定区域到目标点，角度固定为“正对目标点的那个90度”（轴向对齐）
        - all: 在环境范围内随机点，避开禁区，角度随机
        angle_mode: 'align' | 'random' | 'axis' | 'exact_noise'
        """
        stage = getattr(self, 'curriculum_stage', 'end')
        # 选择目标类型
        target_type = self._choose_target_type()
        target_pos = self._get_target_pos(target_type)

        if stage == 'small':
            start_pos = self.fixed_positions['start']
            angle_mode = 'exact_noise'
            target_pos = self.fixed_positions['smaller']
            return start_pos, target_pos, angle_mode

        if stage == 'start':
            start_key = 'normal_start' if target_type == 'normal' else 'unload_start'
            start_pos = self.fixed_positions[start_key]
            return start_pos, target_pos, 'exact_noise'

        if stage in ('easy', 'medium', 'hard', 'hard2'):
            # 分阶段半径范围：easy(2,4), medium(3,5), hard(4,7)
            r_ranges = {
                'easy': (2.0, 7.0),
                'medium': (4.0,9.0),
                'hard': (5.0, 10.0),
                'hard2': (7.0, 10.0)
            }
            r_min, r_max = r_ranges[stage]
            start_pos = self._sample_in_annulus_avoid(center=target_pos, r_min=r_min, r_max=r_max)
            if stage == 'easy':
                angle_mode = 'exact_noise' 
            elif stage == 'medium':
                angle_mode = 'exact_noise'
            elif stage == 'hard':
                angle_mode = 'axis'
            elif stage == 'hard2':
                angle_mode = 'axis'
            return start_pos, target_pos, angle_mode

        if stage == 'all':
            # 全域随机，避开禁区
            for _ in range(300):
                candidate = [
                    random.uniform(self.env_bounds['x_min'], self.env_bounds['x_max']),
                    random.uniform(self.env_bounds['y_min'], self.env_bounds['y_max']),
                    0.0
                ]
                if self.is_position_valid(candidate) and not self._is_in_forbidden_zone(candidate):
                    return candidate, target_pos, 'axis'
            # 回退
            return self.fixed_positions['start'], target_pos, 'axis'

        # 默认 end：使用预定义区域 -> 目标（矩形内采样且避开禁区），角度按轴向
        # 起点区域：目标为 normal 则从 start_area；目标为 unload 则从 normal_area
        depart_area_name = 'start_area' if target_type == 'normal' else 'normal_area'
        rect = self.position_areas[depart_area_name]
        start_pos = self._sample_in_rect_avoid(rect)
        return start_pos, target_pos, 'axis'
    
    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间欧氏距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def calculate_manhattan_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算曼哈顿距离"""
        return sum(abs(a - b) for a, b in zip(pos1, pos2))
    
    def calculate_heading(self, from_pos: List[float], to_pos: List[float]) -> float:
        """计算从from_pos到to_pos的航向角（偏航角）"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return math.atan2(dy, dx)
    
    def normalize_heading(self, heading: float) -> float:
        """归一化航向角到[-π, π]"""
        while heading > math.pi:
            heading -= 2 * math.pi
        while heading < -math.pi:
            heading += 2 * math.pi
        return heading
    
    def generate_random_position(self) -> List[float]:
        """生成随机有效位置"""
        return [
            random.uniform(self.env_bounds['x_min'], self.env_bounds['x_max']),
            random.uniform(self.env_bounds['y_min'], self.env_bounds['y_max']),
            random.uniform(self.env_bounds['z_min'], self.env_bounds['z_max'])
        ]
    
    def generate_nearby_position(self, center: List[float], radius: float) -> List[float]:
        """在以center为中心、radius为半径的圆内生成随机位置"""
        while True:
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            
            new_pos = [
                center[0] + r * math.cos(angle),
                center[1] + r * math.sin(angle),
                center[2]  # z保持不变
            ]
            
            if self.is_position_valid(new_pos):
                return new_pos

class NavigationTaskGenerator:
    """导航任务生成器 - 用于训练和测试"""
    
    def __init__(self, nav_utils: NavigationUtils):
        self.nav_utils = nav_utils
        self.task_pool = []
    
    def get_navigation_task_test(test_id:str):
        """
        返回:起点和终点坐标元组
        """
        fixed_positions = {
            'start': [-5.0, 3.0, 0.0],       # 起点
            'unload': [-5.0, -2.0, 0.0],     # 卸货点
            'dangerous': [5.0, 3.2, 0.0],    # 危险货物点
            'fragile': [5.0, 1.7, 0.0],      # 易碎货物点
            'normal': [5.0, 0.2, 0.0],       # 普通货物点
        }

        if test_id == '1':
            start_pos = fixed_positions['start']
            target_pos = fixed_positions['normal']
        
        if test_id == '2':
            start_pos = fixed_positions['start']
            target_pos = fixed_positions['fragile']
        
        if test_id == '3':
            start_pos = fixed_positions['start']
            target_pos = fixed_positions['dangerous']
        
        if test_id == '4':
            start_pos = fixed_positions['normal']
            target_pos = fixed_positions['unload']
        
        if test_id == '5':
            start_pos = fixed_positions['fragile']
            target_pos = fixed_positions['unload']
        
        if test_id == '6':
            start_pos = fixed_positions['dangerous']
            target_pos = fixed_positions['unload']

        return start_pos, target_pos