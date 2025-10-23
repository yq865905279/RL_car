"""
基于 Webots GUI 的模型测试脚本
"""
from __future__ import annotations

import argparse
from re import T
import sys
import time
from pathlib import Path
from typing import Optional
import random
import numpy as np
import torch
import math
import keyboard
import requests
import cv2
# 确保可以从本目录下的 src/ 导入
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 项目内导入
from src.environments.navigation_env import ROSbotNavigationEnv
from src.utils.navigation_utils import NavigationTaskGenerator
from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor

#global controller
#controller = test_utilis.RosbotController()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webots GUI 模型测试")
    parser.add_argument("--model", type=str, default="../models/R4.6.10_W2E4_T6.1_normal_20251009_221340/td3_R4.6.10_W2E4_T6.1_normal_20251009_221340_normal_400000_autosave_0.94.zip", help="已训练模型的 .zip 路径 (SB3 格式)")
    parser.add_argument("--cargo_type", type=str, default="normal", choices=["normal", "fragile", "dangerous"], help="货物类型")
    parser.add_argument("--episodes", type=int, default=20, help="测试 episode 数")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略动作")
    parser.add_argument("--device", type=str, default="cuda", help="模型推理设备：auto/cpu/cuda")
    parser.add_argument("--world", type=str, default="../warehouse/worlds/warehouse2_end5.wbt", help="可选，自定义 world 文件路径")
    parser.add_argument("--control_period_ms", type=int, default=200, help="控制周期 (ms)，与训练一致")
    parser.add_argument("--fast_mode", action="store_true", help="以 fast 模式运行 Webots (默认关闭以更贴近真实速度)")
    parser.add_argument("--debug", action="store_true", help="启用环境与脚本调试输出")
    # 添加与训练一致的环境参数
    parser.add_argument("--obs_mode", type=str, default="lidar", choices=["lidar", "local_map"], help="观测模式（需与模型训练时一致）")
    parser.add_argument("--action_mode", type=str, default="wheels", choices=["wheels", "twist"], help="动作模式")
    parser.add_argument("--macro_action_steps", type=int, default=1, help="宏动作步数")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="最大episode步数")
    parser.add_argument("--show_map", action="store_true", help="显示地图（调试用）")
    parser.add_argument("--enable_speed_smoothing", action="store_true", default=False, help="启用速度平滑")
    parser.add_argument("--training_mode", type=str, default="vertical_curriculum", help="训练模式")
    
    # 奖励函数参数（需要与训练时一致）
    parser.add_argument('--delta_distance_k', type=float, default=30.0, help='距离变化奖励系数')
    parser.add_argument('--movement_reward_k', type=float, default=0.5, help='移动奖励系数')
    parser.add_argument('--distance_k', type=float, default=0.7, help='基础距离奖励系数')
    parser.add_argument('--time_k', type=float, default=0.4, help='时间惩罚系数')
    parser.add_argument('--wall_proximity_penalty_k', type=float, default=3, help='墙壁接近惩罚系数')
    parser.add_argument('--angle_reward_k', type=float, default=5, help='角度奖励系数')
    parser.add_argument('--angle_change_k', type=float, default=15.0, help='角度变化奖励')
    parser.add_argument('--directional_movement_k', type=float, default=50.0, help='方向性移动奖励系数')
    parser.add_argument('--early_spin_penalty_k', type=float, default=1.0, help='早期原地打转惩罚系数')
    parser.add_argument('--front_clear_k', type=float, default=1.0, help='前方有路奖励系数')
    parser.add_argument('--liner_distance_reward', type=float, default=0, help='线性距离奖励')
    parser.add_argument('--stop_bonus_k', type=float, default=0, help='停车奖励系数')
    parser.add_argument('--approach_reward_k', type=float, default=0, help='接近奖励系数')
    parser.add_argument('--slow_down_reward_k', type=float, default=0, help='接近目标减速奖励系数')
    
    return parser.parse_args()


def load_model(model_path: str, env, device: str = "auto"):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    # SB3 的 .load 可指定 device；加载后设置 env 用于 predict
    model = TD3.load(str(model_file), env=env, device=device)
    return model


def send_image_for_recognition(env, backend_url="http://127.0.0.1:8008/recognize"):
    """捕获图像，发送到后端，并打印识别结果。"""
    # 检查相机设备
    if not hasattr(env, 'camera_color'):
        print("错误: 环境没有 camera_color 属性。")
        print("请确保使用最新版本的 navigation_env.py，并重启环境。")
        return
    
    if not env.camera_color:
        print("错误: 彩色相机设备未初始化。")
        print("可能原因:")
        print("  1. 机器人 proto 文件中没有相机设备")
        print("  2. 相机设备名称不匹配")
        print("  3. 需要重启 Webots 以加载新配置")
        print("建议: 关闭 Webots 和脚本，然后重新启动。")
        return

    print("正在捕获图像...")
    
    # 确保相机数据已准备好：执行几个仿真步骤
    # 相机需要至少一个仿真步骤才能捕获图像
    for _ in range(3):
        env.robot.step(env.timestep)
    
    image_data = env.camera_color.getImage()
    if not image_data:
        print("警告: 无法获取彩色相机图像。")
        print("请确保:")
        print("  1. 相机已启用 (env.camera_color.enable())")
        print("  2. 已执行至少一个仿真步骤")
        print("  3. 相机设备正常工作")
        return

    width = env.camera_color.getWidth()
    height = env.camera_color.getHeight()

    # 转换为 OpenCV 格式 (BGRA 到 BGR)
    image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # 编码为 JPEG 以便 multipart 上传
    _, buffer = cv2.imencode('.jpg', image_bgr)
    
    print(f"正在发送图像到 {backend_url}...")
    try:
        # 以 multipart/form-data 格式发送（文件上传）而不是 JSON
        files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
        response = requests.post(backend_url, files=files, timeout=10)
        response.raise_for_status()  # 对于坏的状态码（4xx 或 5xx）抛出异常

        # 打印识别结果
        result = response.json()
        print(f"识别结果: {result}")
        
        # 如果有详细的 OCR 结果，打印它们
        if result.get('success') and result.get('texts'):
            print(f"✓ OCR 识别成功 - 总共识别到 {len(result['texts'])} 行文本:")
            for i, (text, conf) in enumerate(zip(result['texts'], result.get('confidences', []))):
                print(f"  {i+1}. {text} (置信度: {conf:.2f})")
            print(f"完整文本: {result.get('total_text', '')}")
            print(f"处理时间: {result.get('processing_time', 0):.2f} 秒")
        elif result.get('success') == False:
            print(f"✗ OCR 识别失败: {result.get('message', '未知错误')}")

    except requests.exceptions.RequestException as e:
        print(f"发送图像错误: {e}")


def send_robot_status(env, action, status_backend_url="http://127.0.0.1:8009/status"):
    """发送机器人状态信息到监控后端。"""
    try:
        # 获取底盘参数用于计算速度
        wheel_radius = getattr(env, 'wheel_radius', 0.043)  # 轮子半径（米）
        wheel_base = getattr(env, 'wheel_base', 0.22)  # 轮距（米）
        
        # 计算线速度和角速度（基于轮速 action）
        # action 是轮速百分比 [left, right]，需要转换为实际速度
        left_wheel_speed = float(action[0])  # 左轮速度百分比
        right_wheel_speed = float(action[1])  # 右轮速度百分比
        
        # 线速度 (m/s) = 平均轮速 * 轮子半径
        linear_velocity_x = (left_wheel_speed + right_wheel_speed) / 2.0 * wheel_radius
        
        # 角速度 (rad/s) = (右轮速度 - 左轮速度) * 轮子半径 / 轮距
        angular_velocity_z = (right_wheel_speed - left_wheel_speed) * wheel_radius / wheel_base
        
        # 构建状态数据（包含必需字段）
        status_data = {
            "robot_id": "rosbot_01",
            "status_message": f"running - left:{action[0]:.1f}, right:{action[1]:.1f}",
            "linear_velocity": {
                "x": linear_velocity_x,
                "y": 0.0,
                "z": 0.0
            },
            "angular_velocity": {
                "x": 0.0,
                "y": 0.0,
                "z": angular_velocity_z
            }
        }
        
        # 添加位置信息
        try:
            gps_values = env._get_sup_position()
            if gps_values and len(gps_values) >= 3:
                status_data["position"] = {
                    "x": float(gps_values[0]),
                    "y": float(gps_values[1]), 
                    "z": float(gps_values[2])
                }
        except Exception:
            pass
        
        # 添加朝向信息
        try:
            orientation = env._get_sup_orientation()
            if orientation and len(orientation) >= 3:
                status_data["orientation"] = {
                    "x": float(orientation[0]),  # roll
                    "y": float(orientation[1]),  # pitch
                    "z": float(orientation[2]),  # yaw
                    "w": 1.0  # 简化的四元数w分量
                }
        except Exception:
            pass
        
        # 发送状态数据
        # 调试：显示发送的数据
        if not hasattr(send_robot_status, 'debug_printed'):
            print(f"[DEBUG] 发送的状态数据: {status_data}")
            send_robot_status.debug_printed = True
        
        response = requests.post(status_backend_url, json=status_data, timeout=2)
        
        if response.status_code == 200:
            result = response.json()
            if not hasattr(send_robot_status, 'success_count'):
                send_robot_status.success_count = 0
            send_robot_status.success_count += 1
            
            # 每100次成功发送打印一次确认
            # if send_robot_status.success_count % 100 == 0:
            #     print(f"发送状态 {send_robot_status.success_count}, {result.get('message', '')}")
        else:
            print(f"发送状态失败: HTTP {response.status_code}")
            try:
                error_detail = response.json()
                print(f"错误详情: {error_detail}")
            except:
                print(f"响应内容: {response.text}")
            
    except requests.exceptions.RequestException as e:
        # 只在第一次失败或每50次失败时打印错误，避免刷屏
        if not hasattr(send_robot_status, 'error_count'):
            send_robot_status.error_count = 0
        send_robot_status.error_count += 1
        
        if send_robot_status.error_count == 1 or send_robot_status.error_count % 50 == 0:
            print(f"发送状态错误 (#{send_robot_status.error_count}): {e}")
    except Exception as e:
        print(f"发送状态错误: {e}")


# 取货点坐标（仅用于测试脚本内的特殊处理，不修改训练/环境逻辑）
PICKUP_POINTS = {
    "dangerous": np.array([5.0, 3.0, 0.0], dtype=float),
    "fragile": np.array([5.0, 1.7, 0.0], dtype=float),
    "normal": np.array([5.0, 0.2, 0.0], dtype=float),
}

def _get_yaw(env) -> float:
    """获取机器人当前朝向（yaw, 弧度）"""
    try:
        ori = env._get_sup_orientation()
        if isinstance(ori, (list, tuple, np.ndarray)) and len(ori) >= 3:
            return float(ori[2])
    except Exception:
        pass
    return 0.0

def _normalize_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def _set_raw_wheel_velocities(env, left_speed: float, right_speed: float):
    """直接设置四个轮子的速度（rad/s），用于测试脚本中的手动控制。"""
    try:
        if hasattr(env, 'fl_motor') and env.fl_motor:
            env.fl_motor.setVelocity(float(left_speed))
        if hasattr(env, 'rl_motor') and env.rl_motor:
            env.rl_motor.setVelocity(float(left_speed))
        if hasattr(env, 'fr_motor') and env.fr_motor:
            env.fr_motor.setVelocity(float(right_speed))
        if hasattr(env, 'rr_motor') and env.rr_motor:
            env.rr_motor.setVelocity(float(right_speed))
    except Exception:
        pass

def turn_to_angle(env, target_angle: float, tolerance: float = 0.05, max_speed: float = 4.0, timeout_s: float = 3.0, debug: bool = False):
    """原地转向到目标角度（使用直驱轮速度），不依赖环境动作空间
    
    转向逻辑：
    - 正误差 (err > 0): 需要逆时针转 (左转)
      -> 左轮正转速, 右轮负转速
    - 负误差 (err < 0): 需要顺时针转 (右转)
      -> 左轮负转速, 右轮正转速
    """
    start_t = time.time()
    iteration = 0
    while True:
        if time.time() - start_t > timeout_s:
            if debug:
                print(f"    [turn_to_angle] 超时退出")
            break
        current = _get_yaw(env)
        err = _normalize_angle(target_angle - current)
        if abs(err) < tolerance:
            if debug:
                print(f"    [turn_to_angle] 达到目标角度, 误差: {math.degrees(err):.2f}°")
            break
        
        # 计算转向速度 (rad/s)
        turn_speed = float(np.clip(err * 3.0, -max_speed, max_speed))
        
        # 左右轮相反速度以原地转向
        # 正误差(需要左转): 左轮+, 右轮-
        # 负误差(需要右转): 左轮-, 右轮+
        _set_raw_wheel_velocities(env, turn_speed, -turn_speed)
        
        if debug and iteration % 5 == 0:
            print(f"    [turn_to_angle] iter={iteration}, current={math.degrees(current):.1f}°, "
                  f"target={math.degrees(target_angle):.1f}°, err={math.degrees(err):.1f}°, "
                  f"turn_speed={turn_speed:.2f}")
        
        env.robot.step(env.timestep)
        send_robot_status(env, np.array([turn_speed, -turn_speed]))
        iteration += 1
    
    # 停止
    _set_raw_wheel_velocities(env, 0.0, 0.0)
    env.robot.step(env.timestep)
    send_robot_status(env, np.array([0.0, 0.0]))

def move_forward_distance(env, distance_m: float, speed: float = None, timeout_s: float = 3.0):
    """沿当前朝向前进指定距离（基于Supervisor位姿测距）。
    直接使用底层轮速控制，使用平滑的速度曲线在1秒（5次控制）内完成移动，防止翘头。
    
    Args:
        env: 环境实例
        distance_m: 目标距离（米）
        speed: 最大转速 (rad/s)。如果为None，则自动计算在1秒内完成所需的转速
        timeout_s: 超时时间（秒）
    
    速度曲线：使用梯形速度曲线（加速-匀速-减速）
      - 加速阶段：1步（0-0.2秒）
      - 匀速阶段：3步（0.2-0.8秒）
      - 减速阶段：1步（0.8-1.0秒）
    """
    # 获取底盘参数
    wheel_radius = getattr(env, 'wheel_radius', 0.043)  # 轮子半径（米）
    max_motor_speed = getattr(env, 'max_motor_speed', 26.0)  # 最大转速 (rad/s)
    
    # 每步时间（通常是0.2秒）
    dt = env.timestep / 1000.0  # 转换为秒
    total_steps = 5  # 1秒内的控制步数
    
    # 如果未指定转速，则根据距离和时间计算所需的转速
    if speed is None:
        # 梯形速度曲线：加速(1步) + 匀速(3步) + 减速(1步)
        # 总距离 = 0.5*v_linear*dt + 3*v_linear*dt + 0.5*v_linear*dt = 4*v_linear*dt
        # 因此 v_linear = distance_m / (4 * dt)  [线速度 m/s]
        # 转速 (rad/s) = 线速度 (m/s) / 轮子半径 (m)
        v_linear_max = distance_m / (4.0 * dt)  # 所需的最大线速度 (m/s)
        speed = v_linear_max / wheel_radius  # 转换为转速 (rad/s)
        # 限制最大转速，防止超过电机极限或过快
        #speed = min(speed, max_motor_speed * 0.8)  # 限制在60%最大转速
    
    # 限制转速范围
    speed = float(np.clip(speed, 0.0, max_motor_speed))
    
    # 定义平滑的转速曲线（梯形）(rad/s)
    speed_profile = [
        speed * 0.3,   # 第1步：加速到50%
        speed * 1.0,   # 第2步：达到最大转速
        speed * 1.4,   # 第3步：保持最大转速
        speed * 1.0,   # 第4步：保持最大转速
        speed * 0.3,   # 第5步：减速到50%
    ]
    
    start_pos = env._get_sup_position()
    start_t = time.time()
    step_idx = 0
    
    while True:
        if time.time() - start_t > timeout_s:
            break

        cur = env._get_sup_position()
        moved = float(np.linalg.norm((np.array(cur) - np.array(start_pos))[:2]))
        if moved >= float(distance_m):
            break
        
        # 根据速度曲线选择当前转速 (rad/s)
        if step_idx < len(speed_profile):
            current_speed = speed_profile[step_idx]
        else:
            # 如果超过预定步数，使用最小转速缓慢前进
            current_speed = speed * 0.3
        
        # 直接设置底层轮速 (rad/s)
        _set_raw_wheel_velocities(env, float(current_speed), float(current_speed))
        env.robot.step(env.timestep)
        send_robot_status(env, np.array([current_speed, current_speed]))
        step_idx += 1
    
    # 停止：设置轮速为0
    _set_raw_wheel_velocities(env, 0.0, 0.0)
    env.robot.step(env.timestep)
    send_robot_status(env, np.array([0.0, 0.0]))

def go_to_target(env, model,deterministic: bool = True) -> dict:
    done = False
    steps = 0
    total_reward = 0.0

    # 新增统计变量
    collision_count = 0
    # velocities = []
    # accelerations = []
    # last_velocity = 0.0
    success = False
    env_success_last = False
    
    # ✅ 关键修复：不要在这里重新设置 task_info！
    # task_info 已经在 run_episode 中设置并通过 test_reset() 初始化过了
    # 重复设置会导致奖励函数状态与实际任务不一致
    #print(f"开始导航: {start_pose} -> {target_pose}")
    
    # 环境已经在 run_episode 中通过 test_reset() 初始化
    # 这里只需要获取当前观测
    obs, _ = env._get_observation()
    # obs = first_obs
    while not done:
        action, _ = model.policy.predict(obs, deterministic=True)
        #controller.send_robot_status_env(action)
        obs, reward, terminated, truncated, info = env.step(action)
        send_robot_status(env, action)
        # 环境info提供 'last_collision' 字段
        collision_info = info.get('last_collision')
        if collision_info:
            collision_count += 1
        # 统一使用环境info中的success
        try:
            env_success_last = bool(info.get('success'))
        except Exception:
            env_success_last = False


        # # 碰撞处理（依据 info['_last_collision_info']）
        # collision_info = info.get('_last_collision_info')
        # if collision_info:
        #     collision_count += 1
        #     print(f"💥 检测到碰撞! 第 {collision_count} 次. 开始执行后退操作...")
        #     # 后退操作：发送负向速度指令
        #     backup_speed = -5.0  # 后退速度
        #     backup_duration_ms = 1000  # 后退持续时间 (ms)
        #     backup_steps = int(backup_duration_ms / env.control_period_ms)

        #     for _ in range(backup_steps):
        #         _set_raw_wheel_velocities(env, backup_speed, backup_speed)
        #         env.robot.step(env.timestep)
            
        #     # 后退后停止
        #     _set_raw_wheel_velocities(env, 0, 0)
        #     env.robot.step(env.timestep)
        #     print("后退完成，恢复模型控制.")

        #     # 由于我们不希望碰撞终止episode，重置终止状态
        #     # 注意：这会覆盖掉环境因碰撞返回的 terminated=True
        #     terminated = False

        steps += 1
        total_reward += float(reward)
        done = bool(terminated or truncated)
    dist_final, _ = env._calculate_distance_to_target()
    # 统一依据环境info中的success
    success = bool(env_success_last)
    return {"success": success, "final_distance": float(dist_final), "collision_count": collision_count, "total_reward": total_reward, "steps": steps}

def correct_position(env, target_pos: list):
    print("矫正位置...")
    # 1) 转向最终目标点
    # 计算世界坐标系下的绝对角度（从当前位置指向目标）
    current_pos = env._get_sup_position()
    current_yaw = _get_yaw(env)
    dx = float(target_pos[0] - current_pos[0])
    dy = float(target_pos[1] - current_pos[1])
    distance_to_target = math.sqrt(dx**2 + dy**2)
    
    # Webots 坐标系检查：如果 Y 轴方向相反，需要反转 dy
    # 标准数学坐标系: atan2(dy, dx)
    # 如果 Webots Y 轴向下: atan2(-dy, dx)
    absolute_angle = math.atan2(-dy, dx)  # 尝试反转 Y 轴
    
    # print(f"  [DEBUG] dx={dx:.3f}, dy={dy:.3f}, -dy={-dy:.3f}")
    # print(f"  [DEBUG] atan2(dy,dx)={math.degrees(math.atan2(dy,dx)):.1f}°, atan2(-dy,dx)={math.degrees(math.atan2(-dy,dx)):.1f}°")
    
    # print(f"  当前位置: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
    # print(f"  目标位置: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
    # print(f"  当前朝向: {math.degrees(current_yaw):.1f}°")
    # print(f"  目标朝向: {math.degrees(absolute_angle):.1f}°")
    # print(f"  需要转向: {math.degrees(_normalize_angle(absolute_angle - current_yaw)):.1f}°")
    # print(f"  距离: {distance_to_target:.3f}m")
    
    # 使用较低的转速和更高的精度要求
    turn_to_angle(env, absolute_angle, tolerance=0.005, max_speed=3, timeout_s=5.0, debug=False)
    
    # 转向后稳定：确保完全停止并更新状态
    _set_raw_wheel_velocities(env, 0.0, 0.0)
    for _ in range(3):
        env.robot.step(env.timestep)
    
    # # 验证转向结果
    final_yaw = _get_yaw(env)
    yaw_error = _normalize_angle(absolute_angle - final_yaw)
    # print(f"  转向后朝向: {math.degrees(final_yaw):.1f}°, 误差: {math.degrees(yaw_error):.1f}°")
    
    # # 如果误差较大，再次微调
    if abs(yaw_error) > 0.01:  # 大于0.57°
        print(f"  误差较大，进行微调...")
        turn_to_angle(env, absolute_angle, tolerance=0.001, max_speed=1, timeout_s=3.0, debug=False)
        _set_raw_wheel_velocities(env, 0.0, 0.0)
        for _ in range(3):
            env.robot.step(env.timestep)
        final_yaw = _get_yaw(env)
        yaw_error = _normalize_angle(absolute_angle - final_yaw)
        print(f"  微调后朝向: {math.degrees(final_yaw):.1f}°, 误差: {math.degrees(yaw_error):.1f}°")
    
    # 2) 直行到最终目标
    cur_after_turn = np.array(env._get_sup_position(), dtype=float)
    dist_to_final = float(np.linalg.norm(target_pos[:2] - cur_after_turn[:2]))
    
    # 再次验证朝向，确保前进方向正确
    verify_yaw = _get_yaw(env)
    verify_error = _normalize_angle(absolute_angle - verify_yaw)
    #print(f"  前进前最终验证朝向: {math.degrees(verify_yaw):.1f}°, 误差: {math.degrees(verify_error):.1f}°")
    #print(f"  开始前进 {dist_to_final:.3f}m...")
    move_forward_distance(env, dist_to_final)
    
    # 验证前进结果
    final_pos = env._get_sup_position()
    final_dist = math.sqrt((target_pos[0]-final_pos[0])**2 + (target_pos[1]-final_pos[1])**2)
    #print(f"  前进后位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f}), 距目标: {final_dist:.3f}m")
    
    # 3) 转到正对的角度（这里采用 0°）
    #print(f"  转向0°...")
    # 使用较低的转速和更高的精度要求
    turn_to_angle(env, 0, tolerance=0.001, max_speed=3, timeout_s=5.0, debug=False)
    
    # 稳定后验证
    _set_raw_wheel_velocities(env, 0.0, 0.0)
    for _ in range(3):
        env.robot.step(env.timestep)
    
    final_yaw_2 = _get_yaw(env)
    #print(f"  最终朝向: {math.degrees(final_yaw_2):.1f}°")

    _set_raw_wheel_velocities(env, 0.0, 0.0)
    env.robot.step(env.timestep)

def run_episode(env, model, deterministic: bool = True, episode_index: int = 0,
                debug: bool = False,test_id: str = '0') -> dict:
    done = False
    steps = 0
    total_reward = 0.0

    # 新增统计变量
    collision_count = 0
    velocities = []
    accelerations = []
    last_velocity = 0.0
    success = False
    dist_final = np.inf

    # 设置新的导航任务
    start_pos, target_pos = NavigationTaskGenerator.get_navigation_task_test(test_id)
    
    
    if test_id in ['1','2','3']:
        start_pos = np.array(start_pos, dtype=np.float32)
        target_pos = np.array(target_pos, dtype=np.float32)
        target_pos_1 = np.array([target_pos[0]-2,target_pos[1],target_pos[2]], dtype=np.float32)
        env._angle_mode_on_reset = 'axis'
        env._reset_robot_position(start_pos)
        env._rotate_to_target()

        env.task_info['start_pos'] = np.array(start_pos, dtype=np.float32)
        env.task_info['target_pos'] = np.array(target_pos_1, dtype=np.float32)
        env.test_reset() 
        result = go_to_target(env, model,deterministic)
        steps+=result['steps']
        collision_count+=result['collision_count']
        total_reward+=result['total_reward']

        # 二阶段
        # print("二阶段...")
        # env.task_info['start_pos'] = np.array(env._get_sup_position(), dtype=np.float32) # 二阶段当前位置为起点
        # env.task_info['target_pos'] = np.array([target_pos[0]+2, target_pos[1], target_pos[2]], dtype=np.float32)
        # result = go_to_target(env, model, env.task_info['start_pos'], env.task_info['target_pos'], deterministic)
        # steps+=result['steps']
        # collision_count+=result['collision_count']
        # total_reward+=result['total_reward']

        # 统一使用 go_to_target 返回（来源于 info['success']）
        dist_final = float(result.get('final_distance', np.inf))
        success = bool(result.get('success', False))
        if success:
            correct_position(env, target_pos)        

    elif test_id in ['4','5','6']:
        # 先从取货点前移2米，再返回取货点或进入下一阶段
        # env.task_info['start_pos'] = np.array([start_pos[0]+2, start_pos[1], start_pos[2]], dtype=np.float32)
        # env.task_info['target_pos'] = np.array(start_pos, dtype=np.float32)
        # env._reset_robot_position(env.task_info['start_pos'])
        # result = go_to_target(env, model, env.task_info['start_pos'], env.task_info['target_pos'], deterministic)
        # steps+=result['steps']
        # collision_count+=result['collision_count']
        # total_reward+=result['total_reward']

        # env.task_info['start_pos'] = np.array(env._get_sup_position(), dtype=np.float32) # 二阶段当前位置为起点
        start_pos = np.array(start_pos, dtype=np.float32)
        start_pos_1 = np.array([start_pos[0]-2,start_pos[1],start_pos[2]], dtype=np.float32)
        target_pos= np.array(target_pos, dtype=np.float32)

        env._angle_mode_on_reset = 'axis'
        env._reset_robot_position(start_pos)
        _set_raw_wheel_velocities(env, 0.0, 0.0)
        for _ in range(3):
            env.robot.step(env.timestep)

        env._rotate_to_target()
        env.robot.step(env.timestep)
        move_forward_distance(env, 2.0)
        env.robot.step(env.timestep)
        
        env.task_info['start_pos'] = np.array(start_pos_1, dtype=np.float32)
        env.task_info['target_pos'] = np.array(target_pos, dtype=np.float32)
        env.test_reset()
        result = go_to_target(env, model,deterministic)
        steps+=result['steps']
        collision_count+=result['collision_count']
        total_reward+=result['total_reward']

        dist_final = float(result.get('final_distance', np.inf))
        success = bool(result.get('success', False))
           
    # 统计速度和加速度（当前未在循环中累计，先置为0，避免未定义变量）
    avg_velocity = 0.0
    avg_acceleration = 0.0

    # # 可选：打印部分关键信息
    # try:
    #     d, _ = env._calculate_distance_to_target()

    # 成功判定统一由 info['success'] 决定（来自 go_to_target 的返回）

    print(f"Episode {episode_index+1} 结束: 总奖励={total_reward:.4f}, 步数={steps}, 终止距离={(dist_final if np.isfinite(dist_final) else float('nan')):.3f} m, 碰撞次数={collision_count}, 是否成功={success}")

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": float(dist_final) if np.isfinite(dist_final) else None,
        "collisions": collision_count,
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "success": success
    }


def main():
    args = parse_args()

    # 1) 启动 Webots（带 GUI）
    print("启动 Webots 实例 (GUI 模式)...")
    proc = None
    url: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            proc, url = start_webots_instance(
                instance_id=0,
                world_path=args.world,
                headless=False,          # 强制 GUI
                fast_mode=False,
                no_rendering=False,
                batch=False,
                minimize=False,
                stdout=True,
                stderr=True,
            )
            if proc != -2 and url:
                break
        except Exception as e:
            last_err = e
            print(f"启动 Webots 失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(1.0)
    if proc in (-2, None) or not url:
        raise RuntimeError(f"无法启动 Webots 或解析 URL: {last_err}")

    print(f"Webots extern controller URL: {url}")

    # 2) 构造环境 - 与训练保持一致
    print("构建导航环境...")
    env = ROSbotNavigationEnv(
        extern_controller_url=url,
        cargo_type=args.cargo_type,
        show_map=args.show_map,
        control_period_ms=args.control_period_ms,
        max_episode_steps=args.max_episode_steps,
        seed=0,
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        macro_action_steps=args.macro_action_steps,
        enable_speed_smoothing=args.enable_speed_smoothing,
        training_mode=args.training_mode,
        debug=args.debug,
        enable_obstacle_curriculum=False,
        enable_obstacle_randomization=False,
        use_predefined_positions=False,
        fixed_obstacle_count=-1,
        lock_obstacles_per_stage=False,
    )
    
    # 设置args属性，供奖励函数使用（与训练保持一致）
    env.args = args
    
    # 注册进程清理
    attach_process_cleanup_to_env(env, proc)
    
    # 包装环境用于监控（可选，便于统计）
    # env = Monitor(env)
    
    print("环境创建完成")

    # 3) 加载模型
    model = load_model(args.model, env, device=args.device)
    print("模型加载完成，开始测试...")

    while True:
        controller_type=input("请输入控制器类型 (1:auto,2:manual,q:quit): ")    
        if controller_type == '1':
            # 4) 运行若干 episodes
            results = []
            try:
                for ep in range(int(args.episodes)):
                    test_id=random.randint(1,6)
                    test_id=str(test_id)
                    ep_res = run_episode(env, model, deterministic=True, episode_index=ep, debug=False, test_id=test_id)
                    results.append(ep_res)
            except KeyboardInterrupt:
                print("\n收到中断信号，提前结束测试...")
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            # 5) 汇总
            if results:
                total_episodes = len(results)
                avg_reward = float(np.mean([r["total_reward"] for r in results]))
                avg_steps = float(np.mean([r["steps"] for r in results]))
                
                # 成功率
                successes = sum([1 for r in results if r["success"]])
                success_rate = (successes / total_episodes) * 100 if total_episodes > 0 else 0
                
                # 碰撞率
                total_collisions = sum([r["collisions"] for r in results])
                avg_collisions = total_collisions / total_episodes if total_episodes > 0 else 0

                # 平均速度和加速度
                avg_velocity = float(np.mean([r["avg_velocity"] for r in results]))
                avg_acceleration = float(np.mean([r["avg_acceleration"] for r in results]))

                final_dists = [r["final_distance"] for r in results if r["final_distance"] is not None]
                avg_final_dist = float(np.mean(final_dists)) if final_dists else None
                
                print("\n==== 测试汇总 ====")
                print(f"货物类型: {args.cargo_type}")
                print(f"总 Episodes: {total_episodes}")
                print(f"成功率: {success_rate:.2f}% ({successes}/{total_episodes})")
                print(f"平均碰撞次数: {avg_collisions:.2f}")
                print(f"平均奖励: {avg_reward:.4f}")
                print(f"平均步数: {avg_steps:.1f}")
                print(f"平均线速度: {avg_velocity:.3f} m/s")
                print(f"平均加速度: {avg_acceleration:.3f} m/s²")
                if avg_final_dist is not None:
                    print(f"平均终止距离: {avg_final_dist:.3f} m")
        
        elif controller_type == '2':
            print("手动选择任务ID")
            print("1:start to normal")
            print("2:start to fragile")
            print("3:start to cargo")
            print("4:normal to unload")
            print("5:fragile to unload")
            print("6:cargo to unload")
            print("C:识别图像")
            print("S:发送状态")
            print("Q:退出")
            ep = 1
            while True:
                test_id=input("请输入任务ID:(1,2,3,4,5,6,C,S,Q) ")
                
                if test_id in ['1', '2', '3', '4', '5', '6']:
                    ep_res = run_episode(env, model, deterministic=True, episode_index=ep, debug=False, test_id=test_id)
                    ep+=1
                elif test_id == 'C' or test_id == 'c':
                    # 图像识别功能
                    send_image_for_recognition(env)
                elif test_id == 'S' or test_id == 's':
                    # 手动发送一次状态
                    current_pos = env._get_sup_position()
                    print(f"当前位置: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
                    send_robot_status(env, np.array([0.0, 0.0]))
                elif test_id == 'Q' or test_id == 'q':
                    break
        
        elif controller_type == 'q':
            break

if __name__ == "__main__":
    main()
