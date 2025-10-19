def _determine_obstacle_count(self):
    """
    按照以下优先级确定当次应激活的障碍物数量：
    1) 若 fixed_obstacle_count >= 0，则使用固定数量（优先级最高）
    2) 若 fixed_obstacle_count == -1 且启用课程学习(enable_obstacle_curriculum=True)，使用课程学习表
    3) 否则，使用保守默认值 1

    返回值已被限制在 [0, min(max_obstacles, len(obstacle_nodes))]
    """
    try:
        max_allowed = min(getattr(self, 'max_obstacles', 0), len(getattr(self, 'obstacle_nodes', [])))
        max_allowed = int(max(0, max_allowed))
        # 读取固定数量
        fixed = int(getattr(self, 'fixed_obstacle_count', -1))
        if fixed >= 0:
            count = fixed
            if getattr(self, 'debug', False):
                print(f"[Obstacle] 数量策略: 固定数量 {fixed}")
        else:
            # -1 表示不生效 -> 走课程学习或默认值
            enable_curr = bool(getattr(self, 'enable_obstacle_curriculum', getattr(self, 'enable_obstacle_randomization', False)))
            if enable_curr:
                current_step = int(getattr(self, '_global_training_step', 0))
                count = 0
                steps = getattr(self, 'obstacle_curriculum_steps', [])
                counts = getattr(self, 'obstacle_curriculum_counts', [])
                for i, step_threshold in enumerate(steps):
                    if current_step >= step_threshold and i < len(counts):
                        count = counts[i]
                if getattr(self, 'debug', False):
                    print(f"[Obstacle] 数量策略: 课程学习 Step {current_step} -> {count}")
            else:
                count = 1
                if getattr(self, 'debug', False):
                    print(f"[Obstacle] 数量策略: 默认数量 {count}")
        # 约束
        original = count
        count = max(0, min(count, max_allowed))
        if getattr(self, 'debug', False) and original != count:
            print(f"[Obstacle] 数量被限制: {original} -> {count} (max_allowed={max_allowed})")
        return count
    except Exception as e:
        print(f"[Obstacle] 确定障碍物数量失败: {e}")
        return 0
"""
障碍物随机化功能 - 用于 navigation_env.py
"""
import random
import numpy as np


def _randomize_obstacles(self):
    """
    障碍物随机化
    - 支持两种模式：
      1. 旧模式（use_predefined_positions=False）：根据训练步数渐进式增加数量，位置随机生成
      2. 新模式（use_predefined_positions=True）：从预定义位置中随机选择固定数量的障碍物
    """
    try:
        # 1. 初始化障碍物节点列表（仅第一次）
        if not self.obstacle_nodes:
            # 从场景中查找所有 WoodenBox 并记录其初始位置
            root = self.supervisor.getRoot()
            children_field = root.getField('children')
            num_children = children_field.getCount()
            
            initial_positions = []  # 记录初始位置
            
            for i in range(num_children):
                node = children_field.getMFNode(i)
                if node and node.getTypeName() == 'WoodenBox':
                    self.obstacle_nodes.append(node)
                    
                    # 读取初始位置
                    translation_field = node.getField('translation')
                    if translation_field is not None:
                        pos = translation_field.getSFVec3f()
                        # 只记录 x, y 坐标（忽略 z 高度）
                        initial_positions.append((pos[0], pos[1]))
            
            if self.obstacle_nodes:
                print(f"[Obstacle] 找到 {len(self.obstacle_nodes)} 个 WoodenBox 障碍物")
                
                # 无论使用哪种模式，都从 world 文件读取初始位置并覆盖默认配置
                if initial_positions:
                    self.predefined_obstacle_positions = initial_positions
                    print(f"[Obstacle] 从world文件读取到 {len(initial_positions)} 个初始位置")
                    if self.debug:
                        print(f"[Obstacle] 位置集合: {initial_positions}")
                    
                    # 显示当前模式
                    if getattr(self, 'use_predefined_positions', False):
                        print(f"[Obstacle] 模式：从预定义位置集合中随机选择")
                    else:
                        print(f"[Obstacle] 模式：在范围内随机生成坐标")
            else:
                print(f"[Obstacle] 警告：未找到任何 WoodenBox 障碍物")
                return
        
        # 2. 判断使用哪种模式
        if getattr(self, 'use_predefined_positions', False):
            # === 新模式：从预定义位置中随机选择固定数量 ===
            _randomize_obstacles_from_predefined(self)
        else:
            # === 旧模式：渐进式课程学习 + 随机位置 ===
            _randomize_obstacles_legacy(self)
    
    except Exception as e:
        print(f"[Obstacle] 随机化失败: {e}")
        import traceback
        traceback.print_exc()


def _randomize_obstacles_from_predefined(self):
    """
    新模式：从预定义位置列表中随机选择障碍物位置
    - 支持课程学习：数量根据训练步数变化
    - 位置从预定义的位置集合中随机选择（而非随机生成坐标）
    """
    try:
        # 确定本次应激活数量
        current_step = int(getattr(self, '_global_training_step', 0))
        num_obstacles = _determine_obstacle_count(self)
        
        # 确保预定义位置数量足够
        predefined_positions = self.predefined_obstacle_positions[:]
        if len(predefined_positions) < len(self.obstacle_nodes):
            print(f"[Obstacle] 警告：预定义位置数量({len(predefined_positions)}) < 障碍物节点数量({len(self.obstacle_nodes)})")
            # 补充位置到足够数量（使用随机位置）
            while len(predefined_positions) < len(self.obstacle_nodes):
                x = random.uniform(self.obstacle_x_range[0], self.obstacle_x_range[1])
                y = random.uniform(self.obstacle_y_range[0], self.obstacle_y_range[1])
                predefined_positions.append((x, y))
        
        # 定义安全区域（起点、终点周围）
        safe_radius = 0.8  # 安全半径（米）
        safe_zones = []
        
        # 起点安全区
        if hasattr(self, 'task_info') and 'start_pos' in self.task_info:
            start_pos = self.task_info['start_pos']
            if start_pos is not None and len(start_pos) >= 2:
                safe_zones.append((float(start_pos[0]), float(start_pos[1]), safe_radius))
        
        # 终点安全区
        if hasattr(self, 'task_info') and 'target_pos' in self.task_info:
            target_pos = self.task_info['target_pos']
            if target_pos is not None and len(target_pos) >= 2:
                safe_zones.append((float(target_pos[0]), float(target_pos[1]), safe_radius))
        
        # 过滤掉在安全区内的位置
        valid_positions = []
        for pos in predefined_positions:
            x, y = pos
            in_safe_zone = False
            for sx, sy, sr in safe_zones:
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                if dist < sr:
                    in_safe_zone = True
                    break
            if not in_safe_zone:
                valid_positions.append(pos)
        
        # 如果有效位置不足，使用所有预定义位置（忽略安全区检查）
        if len(valid_positions) < num_obstacles:
            print(f"[Obstacle] 警告：有效位置不足({len(valid_positions)} < {num_obstacles})，使用所有预定义位置")
            valid_positions = predefined_positions[:]
        
        # 从有效位置中选择 num_obstacles 个
        if getattr(self, 'lock_obstacles_per_stage', False):
            # 在课程阶段内锁定障碍物集合：仅在数量增长时新增一个，其余保持不变
            if not hasattr(self, '_locked_predef_indices') or not isinstance(self._locked_predef_indices, list):
                self._locked_predef_indices = []

            # 构建位置->索引映射（基于本次的预定义位置列表）
            index_map = {pos: idx for idx, pos in enumerate(predefined_positions)}
            valid_indices = [index_map[pos] for pos in valid_positions if pos in index_map]

            # 先保留仍然有效的已锁定索引
            keep_indices = [i for i in self._locked_predef_indices if i in valid_indices]

            # 若目标数量更大，则从其余有效候选中补充
            if len(keep_indices) < num_obstacles:
                needed = num_obstacles - len(keep_indices)
                candidates = [i for i in valid_indices if i not in keep_indices]
                add_count = min(needed, len(candidates))
                if add_count > 0:
                    keep_indices.extend(random.sample(candidates, add_count))
                # 如果仍不足（有效位置不足），则退化为使用当前所有有效位置
                if len(keep_indices) < num_obstacles:
                    # 尽力而为：可能因为安全区过滤导致不足
                    pass

            # 若目标数量更小（极少发生），则截断
            if len(keep_indices) > num_obstacles:
                keep_indices = keep_indices[:num_obstacles]

            # 更新锁定集合
            self._locked_predef_indices = keep_indices[:]
            selected_positions = [predefined_positions[i] for i in keep_indices]

            if self.debug:
                print(f"[Obstacle] 锁定模式: 已锁定索引 {self._locked_predef_indices}, 选取 {len(selected_positions)} 个")
        else:
            if len(valid_positions) >= num_obstacles:
                selected_positions = random.sample(valid_positions, num_obstacles)
            else:
                selected_positions = valid_positions[:]
        
        # 将所有障碍物先移到场景外
        for obstacle_node in self.obstacle_nodes:
            translation_field = obstacle_node.getField('translation')
            if translation_field is not None:
                translation_field.setSFVec3f([100.0, 100.0, 0.3])
        
        # 为选中的位置激活障碍物
        for idx, pos in enumerate(selected_positions):
            if idx < len(self.obstacle_nodes):
                obstacle_node = self.obstacle_nodes[idx]
                translation_field = obstacle_node.getField('translation')
                if translation_field is not None:
                    x, y = pos
                    translation_field.setSFVec3f([x, y, self.obstacle_z_height])
        # 记录当次激活的障碍物位置，便于可视化
        try:
            self._active_obstacle_positions = [tuple(p) for p in selected_positions]
        except Exception:
            self._active_obstacle_positions = selected_positions
        
        # 输出调试信息
        if self.debug or current_step % 10000 == 0:
            print(f"[Obstacle] 新模式 Step {current_step}: 从 {len(predefined_positions)} 个预定义位置中随机选择 {len(selected_positions)} 个 (目标 {num_obstacles})")
        if self.debug:
            print(f"[Obstacle] 激活的障碍物位置: {selected_positions}")
    
    except Exception as e:
        print(f"[Obstacle] 新模式随机化失败: {e}")
        import traceback
        traceback.print_exc()


def _randomize_obstacles_legacy(self):
    """
    旧模式：渐进式障碍物课程学习 + 随机位置生成
    """
    try:
        # 1. 确定障碍物数量（遵循 fixed >=0 覆盖；否则课程或默认）
        current_step = int(getattr(self, '_global_training_step', 0))
        num_obstacles = _determine_obstacle_count(self)
        
        # 2. 定义安全区域（起点、终点周围）
        safe_radius = 0.8  # 安全半径（米）
        safe_zones = []
        
        # 起点安全区
        if hasattr(self, 'task_info') and 'start_pos' in self.task_info:
            start_pos = self.task_info['start_pos']
            if start_pos is not None and len(start_pos) >= 2:
                safe_zones.append((float(start_pos[0]), float(start_pos[1]), safe_radius))
        
        # 终点安全区
        if hasattr(self, 'task_info') and 'target_pos' in self.task_info:
            target_pos = self.task_info['target_pos']
            if target_pos is not None and len(target_pos) >= 2:
                safe_zones.append((float(target_pos[0]), float(target_pos[1]), safe_radius))
        
        # 3. 随机化障碍物位置
        active_positions = []
        for idx, obstacle_node in enumerate(self.obstacle_nodes):
            translation_field = obstacle_node.getField('translation')
            if translation_field is None:
                continue
            
            if idx < num_obstacles:
                # 激活并随机化位置
                max_attempts = 50
                for attempt in range(max_attempts):
                    # 生成随机位置
                    x = random.uniform(self.obstacle_x_range[0], self.obstacle_x_range[1])
                    y = random.uniform(self.obstacle_y_range[0], self.obstacle_y_range[1])
                    
                    # 检查是否在安全区内
                    in_safe_zone = False
                    for sx, sy, sr in safe_zones:
                        dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                        if dist < sr:
                            in_safe_zone = True
                            break
                    
                    # 如果不在安全区，使用这个位置
                    if not in_safe_zone:
                        translation_field.setSFVec3f([x, y, self.obstacle_z_height])
                        active_positions.append((x, y))
                        break
                else:
                    # 如果多次尝试失败，使用最后一次生成的位置
                    translation_field.setSFVec3f([x, y, self.obstacle_z_height])
                    active_positions.append((x, y))
            else:
                # 移到场景外（禁用）
                translation_field.setSFVec3f([100.0, 100.0, 0.3])
        # 记录当次激活的障碍物位置，便于可视化
        try:
            self._active_obstacle_positions = [tuple(p) for p in active_positions]
        except Exception:
            self._active_obstacle_positions = active_positions
        
        # 4. 输出调试信息
        if self.debug or current_step % 10000 == 0:
            print(f"[Obstacle] 旧模式 Step {current_step}: 激活 {num_obstacles}/{len(self.obstacle_nodes)} 个")
    
    except Exception as e:
        print(f"[Obstacle] 旧模式随机化失败: {e}")
        import traceback
        traceback.print_exc()


def update_global_training_step(self, step: int):
    """
    更新全局训练步数（由训练脚本调用）
    
    参数:
        step: 当前全局训练步数
    """
    self._global_training_step = int(step)
