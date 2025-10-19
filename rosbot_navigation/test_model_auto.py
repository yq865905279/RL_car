"""
基于 Webots GUI 的模型测试脚本
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# 确保可以从本目录下的 src/ 导入
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 项目内导入
from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3
from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
from stable_baselines3 import TD3
import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webots GUI 模型测试")
    parser.add_argument("--model", type=str,default="/home/dji/RL_car/models/R4.6.10_W2E4_T6.1_normal_20251009_221340/td3_R4.6.10_W2E4_T6.1_normal_20251009_221340_normal_400000_autosave_0.94.zip", help="已训练模型的 .zip 路径 (SB3 格式)")
    parser.add_argument("--cargo_type", type=str, default="normal", choices=["normal", "fragile", "dangerous"], help="货物类型")
    parser.add_argument("--episodes", type=int, default=20, help="测试 episode 数")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略动作")
    parser.add_argument("--device", type=str, default="cuda", help="模型推理设备：auto/cpu/cuda")
    parser.add_argument("--world", type=str, default="/home/dji/RL_car/warehouse/worlds/warehouse2_end4.wbt", help="可选，自定义 world 文件路径")
    parser.add_argument("--control_period_ms", type=int, default=200, help="控制周期 (ms)，与训练一致")
    parser.add_argument("--fast_mode", action="store_true", help="以 fast 模式运行 Webots (默认关闭以更贴近真实速度)")
    parser.add_argument("--debug", action="store_true", help="启用环境与脚本调试输出")
    # 观测/动作配置（需与训练一致）
    parser.add_argument("--obs_mode", type=str, default="lidar", choices=["local_map", "lidar"], help="观测模式：local_map=多输入字典，lidar=向量")
    parser.add_argument("--nav_info_mode", type=str, default="minimal", choices=["minimal", "full"], help="导航信息模式")
    parser.add_argument("--macro_action_steps", type=int, default=1, help="宏动作步数（=1表示每步2维动作）")
    parser.add_argument("--max_episode_steps", type=int, default=500, help="最大 episode 步数")
    parser.add_argument("--action_mode", type=str, default="wheels", choices=["wheels", "twist"], help="动作模式：wheels(左右轮百分比) 或 twist(线/角速度百分比)")
    parser.add_argument("--include_robot_state", action="store_true", help="在观测中加入机器人状态向量")
    parser.add_argument("--no_navigation_info", action="store_true", help="不包含导航信息向量")
    parser.add_argument("--enable_speed_smoothing", action="store_true", help="启用单步速度平滑（训练默认开启）")

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




def run_episode(env, model, deterministic: bool = True, episode_index: int = 0,
                debug: bool = False) -> dict:
    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    # 距离信息（可选）
    try:
        dist0, _ = env._calculate_distance_to_target()
    except Exception:
        dist0 = np.nan

    print(f"\n==== 开始 Episode {episode_index+1} ====")
    print(f"初始距离目标: {dist0:.3f} m" if np.isfinite(dist0) else "初始距离目标: N/A")

    while not done:
        with torch.no_grad():
            # 在分布式训练中我们用 policy.predict() 得到已缩放动作，这里保持一致
            action, _ = model.policy.predict(obs, deterministic=deterministic)
        if debug:
            try:
                # 动作对打印（支持宏动作）
                if hasattr(action, 'shape') and action is not None:
                    flat = np.asarray(action).reshape(-1)
                    if flat.size % 2 == 0:
                        a_pairs = flat.reshape(-1, 2)
                        preview_n = min(5, a_pairs.shape[0])
                        print("[DEBUG][model_out] pairs(0..n): " + ", ".join([f"({a_pairs[i,0]:+.3f},{a_pairs[i,1]:+.3f})" for i in range(preview_n)]))
                    else:
                        print(f"[DEBUG][model_out] action: {flat}")
                else:
                    print(f"[DEBUG][model_out] action type: {type(action)}")
                # 观测预览（兼容字典与ndarray）
                if isinstance(obs, dict):
                    parts = []
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            parts.append(f"{k}: shape={v.shape} range=[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]")
                        else:
                            parts.append(f"{k}: type={type(v)}")
                    print("[DEBUG][obs] {" + ", ".join(parts) + "}")
                elif isinstance(obs, np.ndarray):
                    print(f"[DEBUG][obs] shape: {obs.shape}, dtype: {obs.dtype}, range: [{np.nanmin(obs):.3f}, {np.nanmax(obs):.3f}]")
                else:
                    print(f"[DEBUG][obs] type: {type(obs)}")
            except Exception as e:
                print(f"[DEBUG] 打印调试信息失败: {e}")
        # 新模型直接输出正确的动作格式，无需转换
        next_obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)
        obs = next_obs
        done = bool(terminated or truncated)

        # 可选：打印部分关键信息
        # try:
        #     d, _ = env._calculate_distance_to_target()
        #     print(f"step={steps:03d} reward={reward:+.4f} dist={d:.3f} terminated={terminated} truncated={truncated}")
        # except Exception:
        #     print(f"step={steps:03d} reward={reward:+.4f} terminated={terminated} truncated={truncated}")

    # 结束统计
    try:
        dist_final, _ = env._calculate_distance_to_target()
    except Exception:
        dist_final = np.nan

    print(f"Episode {episode_index+1} 结束: 总奖励={total_reward:.4f}, 步数={steps}, 终止距离={dist_final:.3f} m" if np.isfinite(dist_final) else f"Episode {episode_index+1} 结束: 总奖励={total_reward:.4f}, 步数={steps}")

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": float(dist_final) if np.isfinite(dist_final) else None,
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
                fast_mode=bool(args.fast_mode),
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

    # 2) 构造环境（任务发布与训练一致：reset() 内部会调用 _set_navigation_task）
    env = ROSbotNavigationEnv(
        extern_controller_url=url,
        cargo_type=args.cargo_type,
        show_map= False,
        control_period_ms=200,
        max_episode_steps=500,
        seed=0,
        obs_mode='lidar',
        action_mode='wheels',
        macro_action_steps=1,
        enable_speed_smoothing=False,
        training_mode='vertical_curriculum',
        debug=False,
        enable_obstacle_curriculum=False,
        enable_obstacle_randomization=False,
        use_predefined_positions=False,
        fixed_obstacle_count=-1,
        lock_obstacles_per_stage=False,
    )
    env.args = args
    attach_process_cleanup_to_env(env, proc)

    # 3) 加载模型
    model = load_model(args.model, env, device=args.device)
    print("模型加载完成，开始测试...")

    # 4) 运行若干 episodes
    results = []
    try:
        for ep in range(int(args.episodes)):
            ep_res = run_episode(env, model, deterministic=bool(args.deterministic), episode_index=ep,
                                 debug=bool(args.debug))
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
        avg_reward = float(np.mean([r["total_reward"] for r in results]))
        avg_steps = float(np.mean([r["steps"] for r in results]))
        final_dists = [r["final_distance"] for r in results if r["final_distance"] is not None]
        avg_final_dist = float(np.mean(final_dists)) if final_dists else None
        print("\n==== 测试汇总 ====")
        print(f"货物类型: {args.cargo_type}")
        print(f"Episodes: {len(results)}  平均奖励: {avg_reward:.4f}  平均步数: {avg_steps:.1f}")
        if avg_final_dist is not None:
            print(f"平均终止距离: {avg_final_dist:.3f} m")


if __name__ == "__main__":
    main()
 