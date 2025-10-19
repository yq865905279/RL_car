"""
手动控制模式模块
支持手动选择任务、图形化展示、模型导航、图像识别和状态发送
"""
import numpy as np
import time
from typing import Tuple, Optional
import torch


class ManualTaskSelector:
    """手动任务选择器"""
    
    def __init__(self):
        # 定义所有可用的位置点（来自navigation_utils.py）
        self.available_points = {
            'start': {'pos': np.array([-5.0, 3.0, 0.0]), 'name': '起点', 'type': 'waypoint'},
            'unload': {'pos': np.array([-5.0, -2.0, 0.0]), 'name': '卸货点', 'type': 'waypoint'},
            'dangerous': {'pos': np.array([5.0, 3.0, 0.0]), 'name': '危险货物点', 'type': 'pickup'},
            'fragile': {'pos': np.array([5.0, 1.7, 0.0]), 'name': '易碎货物点', 'type': 'pickup'},
            'normal': {'pos': np.array([5.0, 0.2, 0.0]), 'name': '普通货物点', 'type': 'pickup'},
            'smaller': {'pos': np.array([-0.75, -0.55, 0.0]), 'name': '小环境卸货点', 'type': 'waypoint'},
        }
    
    def display_available_points(self):
        """显示所有可用点位"""
        print("\n" + "="*60)
        print("可用的点位：")
        print("="*60)
        
        # 分类显示
        print("\n【路径点】")
        for key, info in self.available_points.items():
            if info['type'] == 'waypoint':
                pos = info['pos']
                print(f"  {key:12s} - {info['name']:15s} 位置: ({pos[0]:+.1f}, {pos[1]:+.1f})")
        
        print("\n【取货点】")
        for key, info in self.available_points.items():
            if info['type'] == 'pickup':
                pos = info['pos']
                print(f"  {key:12s} - {info['name']:15s} 位置: ({pos[0]:+.1f}, {pos[1]:+.1f})")
        
        print("="*60)
    
    def select_task(self) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
        """交互式选择任务
        
        Returns:
            (start_pos, target_pos, start_key, target_key) 或 None（取消）
        """
        self.display_available_points()
        
        print("\n请选择任务：")
        
        # 选择起点
        while True:
            print("\n选择起点 (输入点位名称，如 'start', 或输入 'q' 退出):")
            start_key = input("> ").strip().lower()
            
            if start_key == 'q':
                return None
            
            if start_key in self.available_points:
                start_pos = self.available_points[start_key]['pos'].copy()
                start_name = self.available_points[start_key]['name']
                print(f"✓ 已选择起点: {start_name} {start_pos[:2]}")
                break
            else:
                print(f"✗ 无效的点位名称: {start_key}")
        
        # 选择终点
        while True:
            print("\n选择终点 (输入点位名称，如 'normal', 或输入 'q' 退出):")
            target_key = input("> ").strip().lower()
            
            if target_key == 'q':
                return None
            
            if target_key in self.available_points:
                target_pos = self.available_points[target_key]['pos'].copy()
                target_name = self.available_points[target_key]['name']
                print(f"✓ 已选择终点: {target_name} {target_pos[:2]}")
                break
            else:
                print(f"✗ 无效的点位名称: {target_key}")
        
        # 确认任务
        print(f"\n任务确认:")
        print(f"  起点: {start_name} → 终点: {target_name}")
        print("开始执行？(y/n)")
        confirm = input("> ").strip().lower()
        
        if confirm == 'y':
            return start_pos, target_pos, start_key, target_key
        else:
            print("任务已取消")
            return None


class ManualControlMode:
    """手动控制模式主类"""
    
    def __init__(self, env, model, args, pickup_handler, backend_comm=None):
        self.env = env
        self.model = model
        self.args = args
        self.pickup_handler = pickup_handler
        self.backend_comm = backend_comm
        self.task_selector = ManualTaskSelector()
        
        # 启用键盘（用于图像识别等）
        try:
            self.keyboard = self.env.robot.getKeyboard()
            self.keyboard.enable(self.env.timestep)
            print("键盘已启用（按 'C' 进行图像识别，'Q' 退出当前任务）")
        except Exception as e:
            print(f"键盘初始化失败: {e}")
            self.keyboard = None
    
    def check_keyboard_interrupt(self) -> bool:
        """检查键盘中断
        
        Returns:
            True: 继续运行, False: 用户要求中断
        """
        if not self.keyboard:
            return True
        
        key = self.keyboard.getKey()
        
        if key == ord('C') or key == ord('c'):  # 图像识别
            if self.backend_comm:
                print("\n[图像识别] 正在捕获和识别图像...")
                self.backend_comm.send_image_for_recognition()
            else:
                print("[图像识别] 后端通信未启用")
        elif key == ord('Q') or key == ord('q'):  # 中断任务
            print("\n[用户中断] 收到中断信号")
            return False
        
        return True
    
    def execute_task(self, start_pos: np.ndarray, target_pos: np.ndarray,
                    start_key: str, target_key: str) -> bool:
        """执行一个手动任务
        
        Returns:
            bool: 任务是否成功完成
        """
        print("\n" + "="*60)
        print(f"开始执行任务: {start_key} → {target_key}")
        print("="*60)
        
        # 设置任务信息（手动设置环境的任务）
        self.env.task_info = {
            'start_pos': start_pos,
            'target_pos': target_pos,
            'cargo_type': self.args.cargo_type,
        }
        
        # 重置环境到指定起点
        try:
            # 手动设置机器人位置到起点
            self.env.robot.getFromDef('rosbot').getField('translation').setSFVec3f([
                float(start_pos[0]),
                float(start_pos[1]),
                float(start_pos[2])
            ])
            # 重置朝向为0
            self.env.robot.getFromDef('rosbot').getField('rotation').setSFRotation([0, 0, 1, 0])
            # 步进一次使设置生效
            self.env.robot.step(self.env.timestep)
            
            # 获取初始观测
            obs = self.env._get_observation()
            
            print(f"✓ 机器人已移动到起点: {start_pos[:2]}")
        except Exception as e:
            print(f"✗ 设置起点失败: {e}")
            # 使用标准reset
            obs, info = self.env.reset()
        
        # 判断是否从取货点出发
        if self.args.enable_pickup_handling and self.pickup_handler.is_pickup_point(start_pos):
            print(f"\n[取货点] 从取货点出发，先向前开2米")
            if not self.pickup_handler.leave_pickup_point():
                print("[警告] 离开取货点序列失败")
        
        # 判断目标是否是取货点
        is_target_pickup = self.args.enable_pickup_handling and self.pickup_handler.is_pickup_point(target_pos)
        if is_target_pickup:
            print(f"[取货点] 目标是取货点，将在接近时启动特殊处理")
        
        # 计算初始距离
        try:
            dist0 = np.linalg.norm(self.env._get_sup_position()[:2] - target_pos[:2])
            print(f"初始距离目标: {dist0:.3f} m")
        except:
            dist0 = np.nan
        
        # 开始导航循环
        done = False
        steps = 0
        total_reward = 0.0
        collision_count = 0
        status_send_counter = 0
        
        while not done and steps < self.args.max_episode_steps:
            # 检查键盘中断
            if not self.check_keyboard_interrupt():
                print("\n任务已被用户中断")
                self.env._send_wheel_velocities(0, 0)
                return False
            
            # 模型推理
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            steps += 1
            total_reward += float(reward)
            
            # 发送状态到后端（每10步发送一次，避免过于频繁）
            status_send_counter += 1
            if self.backend_comm and status_send_counter >= 10:
                try:
                    # 从info中获取实际的轮速
                    left_vel = info.get('left_wheel_velocity', 0.0)
                    right_vel = info.get('right_wheel_velocity', 0.0)
                    self.backend_comm.send_robot_status(left_vel, right_vel)
                except:
                    pass
                status_send_counter = 0
            
            # 检查碰撞
            collision = info.get('collision', False)
            if collision:
                collision_count += 1
                print(f"\n[碰撞] Step {steps}: 检测到碰撞 (第{collision_count}次)")
                
                # 碰撞恢复
                if self.args.collision_recovery and collision_count < self.args.max_collision_retries:
                    from test_enhanced_pickup import handle_collision_recovery
                    handle_collision_recovery(self.env, self.args.collision_backup_distance)
                    obs = next_obs
                    continue
                else:
                    print(f"[碰撞] 达到最大重试次数或未启用恢复，任务失败")
                    terminated = True
            
            # 检查是否接近取货点目标
            if is_target_pickup and not terminated:
                current_pos = self.env._get_sup_position()
                current_dist = np.linalg.norm(current_pos[:2] - target_pos[:2])
                
                if current_dist < 2.5:  # 接近到2.5米时触发特殊处理
                    print(f"\n[取货点] 接近目标取货点，当前距离: {current_dist:.3f}m")
                    self.env._send_wheel_velocities(0, 0)  # 停止
                    
                    # 执行取货点接近序列
                    if self.pickup_handler.approach_pickup_point(target_pos):
                        terminated = True
                        current_pos = self.env._get_sup_position()
                        final_dist = np.linalg.norm(current_pos[:2] - target_pos[:2])
                        print(f"[取货点] 接近序列完成，最终距离: {final_dist:.3f}m")
                    else:
                        print(f"[取货点] 接近序列失败")
                        terminated = True
                    break
            
            obs = next_obs
            done = bool(terminated or truncated)
            
            # 打印进度（每10步）
            if steps % 10 == 0 or done:
                try:
                    current_pos = self.env._get_sup_position()
                    d = np.linalg.norm(current_pos[:2] - target_pos[:2])
                    print(f"step={steps:03d} reward={reward:+.4f} dist={d:.3f} "
                          f"term={terminated} trunc={truncated} coll={collision_count}")
                except:
                    print(f"step={steps:03d} reward={reward:+.4f} term={terminated} trunc={truncated}")
        
        # 任务完成
        self.env._send_wheel_velocities(0, 0)  # 停止
        
        try:
            current_pos = self.env._get_sup_position()
            dist_final = np.linalg.norm(current_pos[:2] - target_pos[:2])
            success = (dist_final < 0.5)
        except:
            dist_final = np.nan
            success = False
        
        print("\n" + "="*60)
        print(f"任务完成:")
        print(f"  总步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  碰撞次数: {collision_count}")
        print(f"  终点距离: {dist_final:.3f} m")
        print(f"  任务状态: {'✓ 成功' if success else '✗ 失败'}")
        print("="*60)
        
        return success
    
    def run(self):
        """运行手动控制模式主循环"""
        print("\n" + "="*60)
        print("手动控制模式")
        print("支持手动选择任务、模型自动导航、图像识别")
        print("="*60)
        
        if self.backend_comm:
            print("✓ 后端通信已启用（图像识别和状态发送）")
        
        task_count = 0
        success_count = 0
        
        while True:
            print("\n" + "-"*60)
            print(f"已完成 {task_count} 个任务，成功 {success_count} 个")
            print("-"*60)
            
            # 选择任务
            task = self.task_selector.select_task()
            
            if task is None:
                print("\n退出手动控制模式")
                break
            
            start_pos, target_pos, start_key, target_key = task
            
            # 执行任务
            success = self.execute_task(start_pos, target_pos, start_key, target_key)
            
            task_count += 1
            if success:
                success_count += 1
            
            # 询问是否继续
            print("\n是否继续下一个任务？(y/n)")
            continue_choice = input("> ").strip().lower()
            
            if continue_choice != 'y':
                break
        
        # 最终统计
        print("\n" + "="*60)
        print("手动控制模式统计")
        print("="*60)
        print(f"总任务数: {task_count}")
        print(f"成功任务: {success_count}")
        if task_count > 0:
            print(f"成功率: {success_count/task_count*100:.1f}%")
        print("="*60)
