"""
多线程训练
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
import cv2
from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3

import mlflow
import mlflow.pytorch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.utils import configure_logger as sb3_configure_logger

from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
from functools import partial
import os
import signal
import multiprocessing as mp
import time
from queue import Empty, Full
from datetime import datetime
import yaml
from collections import deque
import cv2
from src.environments.local_map_obs import LocalMapObservation
import math
try:
    import matplotlib
    # 使用无交互后端，确保在服务器/无显示环境下也能保存图像
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as _matplot_err:
    matplotlib = None
    plt = None
    try:
        print(f"[WARN] matplotlib 不可用，调试图像将不会生成: {_matplot_err}")
    except Exception:
        pass
# ================= 分布式训练实现 ==================

def plot_episode_positions_and_rewards(
    positions: List[tuple],
    rewards: List[float],
    env: ROSbotNavigationEnv = None,
    actor_rank: int = 0,
    episode_index: int = 0,
    out_dir: str = "debug_plots",
    save_root: Optional[str] = None,
    display: bool = False,
    window_name: str = "Episode Debug (Actor 0)",
    color_mode: str = "continuous",   # "continuous" 或 "discrete"
    vmin: float = -0.5,
    vmax: float = 0.5,
    bins: int = 100,
    cmap_name: str = "turbo",
    draw_heading: bool = False
) -> None:
    """将本回合内每一步车辆的位置与奖励绘制到图像中并保存。

    参数:
      positions: [(x, y), ...]，每一步的平面坐标
      rewards: [r1, r2, ...]，每一步奖励
      env: 可选，用于获取起点/终点等信息
      actor_rank: 当前actor编号（用于文件命名）
      episode_index: 回合编号（用于文件命名）
      out_dir: 输出目录，相对于本脚本位置
    """
    try:
        # 如未安装matplotlib，直接返回
        if 'plt' not in globals() or plt is None:
            return
        if not positions or not rewards or len(positions) != len(rewards):
            return
        # 选择保存根目录：优先使用训练的模型目录（save_root -> env.args.model_dir -> parent(model_path) -> 脚本目录）
        if save_root:
            base_dir = Path(save_root)
        else:
            base_dir = None
            try:
                if env is not None and hasattr(env, 'args'):
                    md = getattr(env.args, 'model_dir', None)
                    if md:
                        base_dir = Path(md)
                    else:
                        mpth = getattr(env.args, 'model_path', None)
                        if mpth:
                            base_dir = Path(mpth).parent
            except Exception:
                pass
            if base_dir is None:
                base_dir = Path(__file__).parent
        save_dir = base_dir / out_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        xs = [float(p[0]) for p in positions]
        ys = [float(p[1]) for p in positions]
        total_reward = float(np.sum(rewards)) if len(rewards) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        # 路径线
        ax.plot(xs, ys, color='gray', linewidth=1.0, alpha=0.6, label='trajectory')
        # 颜色表示奖励：可选离散分级或连续渐变，颜色范围 [vmin, vmax]
        # 默认连续渐变，范围更大（-2.0~2.0），并使用高对比色图
        try:
            bins = int(max(3, bins))
        except Exception:
            bins = 50
        rclipped = np.clip(rewards, float(vmin), float(vmax))
        cmap = plt.get_cmap(cmap_name)
        # 选择归一化方式
        try:
            if color_mode == "discrete":
                bounds = np.linspace(vmin, vmax, bins + 1)
                norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N)
            else:
                if float(vmin) < 0.0 < float(vmax):
                    norm = mcolors.TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))
                else:
                    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
        except Exception:
            norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
        # 提升点的可见度：稍大尺寸+细描边
        sc = ax.scatter(xs, ys, c=rclipped, cmap=cmap, norm=norm, s=20, alpha=0.98,
                        edgecolors='k', linewidths=0.1)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        try:
            tick_vals = np.linspace(float(vmin), float(vmax), 5)
            cbar.set_ticks(tick_vals)
        except Exception:
            pass
        cbar.set_label(f'step reward [{vmin}, {vmax}]')

        # 标注起点与终点
        try:
            if env is not None and hasattr(env, 'task_info'):
                sp = env.task_info.get('start_pos', None)
                tp = env.task_info.get('target_pos', None)
                if sp is not None and len(sp) >= 2:
                    ax.scatter([float(sp[0])], [float(sp[1])], marker='*', s=120, c='green', label='start')
                if tp is not None and len(tp) >= 2:
                    ax.scatter([float(tp[0])], [float(tp[1])], marker='*', s=120, c='red', label='target')
        except Exception:
            pass

        # 可选：当前步方向（最后一步姿态）
        if draw_heading:
            try:
                if env is not None and hasattr(env, '_get_sup_orientation') and len(xs) > 0:
                    orient = env._get_sup_orientation()
                    yaw = float(orient[2]) if isinstance(orient, (list, tuple, np.ndarray)) and len(orient) >= 3 else None
                    if yaw is not None:
                        # 以轨迹范围自适应的箭头尺度
                        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
                        y_min, y_max = np.nanmin(ys), np.nanmax(ys)
                        scale = max(0.2, 0.1 * (abs(x_max - x_min) + abs(y_max - y_min)))
                        dx = scale * math.cos(yaw)
                        dy = scale * math.sin(yaw)
                        ax.arrow(xs[-1], ys[-1], dx, dy, head_width=0.1 * scale, head_length=0.1 * scale, fc='k', ec='k', length_includes_head=True)
            except Exception:
                pass

        # 边界框：以(0,0)为中心，(-4,-6)~(4,6) 对角线
        try:
            from matplotlib.patches import Rectangle
            boundary = Rectangle((-6.0, -4.0), 12.0, 8.0,
                                 linewidth=1.2, edgecolor='orange', facecolor='none',
                                 linestyle='--', alpha=0.9, label='boundary')
            ax.add_patch(boundary)
        except Exception:
            pass

        ax.set_title(f"Actor{actor_rank} Episode {episode_index} | Steps: {len(xs)} | SumR: {total_reward:.2f}")
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best')

        out_path = save_dir / f"actor{actor_rank}_episode_{episode_index:05d}.png"
        fig.tight_layout()
        fig.savefig(str(out_path))

        # 如果需要实时显示，则将matplotlib图像转换为OpenCV可显示的图像
        if display and 'cv2' in globals() and cv2 is not None:
            try:
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(h, w, 3)
                # 转为BGR
                img_bgr = img[:, :, ::-1]
                cv2.imshow(window_name, img_bgr)
                cv2.waitKey(1)
            except Exception:
                pass

        plt.close(fig)
    except Exception as e:
        # 避免影响训练流程
        print(f"[ACTOR{actor_rank}] 绘图保存失败: {e}")

def _actor_process(rank: int, cargo_type: str, args: argparse.Namespace, obs_q: mp.Queue, act_q: mp.Queue, exp_q: mp.Queue, stats_q: mp.Queue, stop_event: mp.Event):
    print(f"[ACTOR{rank}] 进程启动...")
    # 提前初始化，避免 finally 中引用未定义
    start_time = time.time()
    metrics = {
        "episodes_completed": 0,
        "steps_completed": 0,
        "total_reward": 0.0,
        "avg_episode_reward": 0.0,
        "max_episode_reward": float('-inf'),
        "min_episode_reward": float('inf'),
        "last_log_time": time.time()
    }

    # 初始MLflow跟踪
    if args.actor_mlflow_tracking:
        experiment_name = getattr(args, 'experiment_name', None) or f"distributed_training_{cargo_type}"
        run_name = f"actor{rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 设置MLflow实验
        try:
            mlflow.set_experiment(experiment_name)
            # 开始一个新的MLflow运行
            mlflow.start_run(run_name=run_name)
            
            # 记录Actor参数
            mlflow.log_params({
                "actor_id": rank,
                "cargo_type": cargo_type,
                "fast_mode": getattr(args, 'fast_mode', True),
                "control_period_ms": getattr(args, 'control_period_ms', 200)
            })
            print(f"[ACTOR{rank}] MLflow 跟踪已初始化: {experiment_name}/{run_name}")
        except Exception as e:
            print(f"[ACTOR{rank}] MLflow 初始化失败: {e}")
    # 选择本Actor使用的world及环境编号（env1/env2对半分配；只有1个actor则使用env1）
    env1_world = str(Path(args.world_1))
    env2_world = str(Path(args.world_2))
    use_env2 = False
    try:
        n_actors = int(getattr(args, 'num_actors', 1))
        if n_actors <= 1:
            use_env2 = False
        else:
            use_env2 = (rank % 2 == 1)
    except Exception:
        use_env2 = (rank % 2 == 1)
    world_to_use = env2_world if use_env2 else env1_world

    # 启动单实例 Webots 并连接
    for i in range(3): 
        proc, url = start_webots_instance(
            instance_id=rank,
            world_path=world_to_use,
            headless=args.headless,
            fast_mode=getattr(args, 'fast_mode', True),
            no_rendering=args.no_rendering,
            batch=args.batch,
            minimize=args.minimize,
            stdout=True,
            stderr=True
        )
        if proc != -2:
            break
    
    print(f"[ACTOR{rank}] Webots 启动完成，URL: {url}")
    
    base_env = ROSbotNavigationEnv(
        cargo_type=cargo_type,
        instance_id=rank,
        controller_url=url,
        fast_mode=(args.fast_mode if args else True),
        control_period_ms=(args.control_period_ms if args else 200),
        debug=bool(getattr(args, 'debug', False)),
        include_robot_state=False,
        include_navigation_info=True,
        nav_info_mode='minimal',
        macro_action_steps=int(getattr(args, 'macro_action_steps', 1)),
        action_mode=str(getattr(args, 'action_mode', 'wheels')),
        obs_mode=str(getattr(args, 'obs_mode', 'local_map')),
        enable_speed_smoothing=bool(getattr(args, 'enable_speed_smoothing', True))
    )
    # 将训练参数对象传递给环境，供奖励函数使用
    base_env.args = args


    # 设置环境选择与课程阶段
    try:
        env_tag = 'env2' if use_env2 else 'env1'
        if hasattr(base_env, 'nav_utils') and base_env.nav_utils:
            base_env.nav_utils.current_env = env_tag
            # 配置课程阶段（start/easy/medium/hard/end/all）
            stage = getattr(args, 'curriculum_stage', 'end')
            if hasattr(base_env.nav_utils, 'set_stage'):
                base_env.nav_utils.set_stage(stage)
            else:
                base_env.nav_utils.curriculum_stage = stage
        print(f"[ACTOR{rank}] 使用 {env_tag} ({world_to_use}), 课程阶段: {getattr(args, 'curriculum_stage', 'end')}")
    except Exception as e:
        print(f"[ACTOR{rank}] 配置环境标签/阶段失败: {e}")

    # 直接使用多输入基础环境（Dict obs: local_map 固定 + 可选向量）
    env = base_env

    attach_process_cleanup_to_env(env, proc)
    print(f"[ACTOR{rank}] 环境初始化完成")

    obs, info = env.reset()
    print(f"[ACTOR{rank}] 开始采样循环")
    
    episode_steps = 0
    episode_reward = 0.0
    episode_count = 0
    start_time = time.time()
    # actor0 专用：收集本回合每一步的位置与奖励
    episode_positions: List[tuple] = []
    episode_step_rewards: List[float] = []
    
    # 指标已提前初始化
    
    # 初始化episode buffer
    episode_buffer = []
    # 早停相关：记录最近N次任务是否成功
    early_stop_window = int(getattr(args, 'early_stop_window', 50))
    early_stop_threshold = float(getattr(args, 'early_stop_success', 0.90))
    recent_episode_success = deque(maxlen=early_stop_window)
    
    while not stop_event.is_set():
        # 1) 将观测放入队列，请求一个动作
        try:
            obs_q.put((rank, obs), timeout=0.1)
            # 仅0号环境进行可视化（有图形化界面）
            if rank == 0 and args.show_map:
                try:
                    # 仅当观测为多输入字典且包含 local_map 时渲染
                    if args.debug:
                        for key,value in obs.items():
                            print(f"{key}: {value}")
                    if isinstance(obs, dict) and ('local_map' in obs):
                        lmo = LocalMapObservation()
                        img = lmo.render_map_to_cv_image(obs['local_map'])
                        cv2.imshow("Local Map (Actor 0)", img)
                        cv2.waitKey(1)
                    else:
                        # lidar 模式：按与模型输入一致的20维向量做柱状图可视化
                        try:
                            if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 20 and cv2 is not None:
                                vec = np.array(obs[:20], dtype=np.float32)
                                h, w = 200, 400
                                img = np.ones((h, w, 3), dtype=np.uint8) * 255
                                maxv = float(np.max(vec)) if np.max(vec) > 0 else 1.0
                                bar_w = w // 20
                                for i, v in enumerate(vec):
                                    bh = int((v / maxv) * (h - 20))
                                    x1 = i * bar_w
                                    y1 = h - 10
                                    x2 = (i + 1) * bar_w - 2
                                    y2 = h - 10 - bh
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
                                cv2.imshow("LiDAR Vector (Actor 0)", img)
                                cv2.waitKey(1)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[ACTOR{rank}] 观测渲染异常: {e}")
                    pass
        except Full:
            # 观测队列满了，Learner 暂未处理，稍后再试，避免阻塞
            time.sleep(0.001)
            continue
        except Exception as e:
            print(f"[ACTOR{rank}] 发送观测异常: {e}")
            raise e

        # 2) 等待 Learner 返回动作（只在成功发送观测后才等待）
        try:
            action = act_q.get(timeout=2.0)
        except Empty:
            # 未及时获得动作，重试
            # 不退出循环，避免 actor 过早终止
            continue
        except Exception as e:
            print(f"[ACTOR{rank}] 获取动作异常: {e}")
            continue

        # 3) 执行动作并写入经验

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        
        # 跟踪奖励和指标
        episode_reward += reward
        metrics["steps_completed"] += 1
        metrics["total_reward"] += reward

        # 仅在actor0记录位置与奖励
        if rank == 0:
            try:
                pos = env._get_sup_position()
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    episode_positions.append((float(pos[0]), float(pos[1])))
                else:
                    episode_positions.append((np.nan, np.nan))
                episode_step_rewards.append(float(reward))
            except Exception:
                # 若读取失败，保持长度一致
                episode_positions.append((np.nan, np.nan))
                episode_step_rewards.append(float(reward))
        
        # 添加距离信息到info
        distance, _ = env._calculate_distance_to_target()
        info['distance_to_target'] = distance
        
        # 根据配置选择写入时机：逐步写入或按回合结束再写入
        exp_item = {
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_obs': next_obs.copy(),
            'done': done,
            'info': info.copy() if info else {}
        }
        if bool(getattr(args, 'enqueue_at_episode_end', True)):
            # 先缓存，等待回合结束后统一写入
            episode_buffer.append(exp_item)
        else:
            # 立即写入经验队列（不做episode级整形）
            try:
                exp_q.put((exp_item['obs'], exp_item['action'], exp_item['reward'], exp_item['next_obs'], exp_item['done']), timeout=0.1)
            except Full:
                pass
            except Exception as e:
                print(f"[ACTOR{rank}] 写入经验异常: {e}")
        
        # 降频记录指标到MLflow
        if args.actor_mlflow_tracking:
            current_time = time.time()
            if current_time - metrics["last_log_time"] >= 10.0:  # 每10秒记录一次
                try:
                    if mlflow.active_run():
                        mlflow.log_metrics({
                        "actor_steps": metrics["steps_completed"],
                        "actor_episodes": metrics["episodes_completed"],
                        "actor_total_reward": metrics["total_reward"]
                    }, step=metrics["steps_completed"])
                except Exception as e:
                    print(f"[ACTOR{rank}] MLflow 记录指标失败: {e}")
                metrics["last_log_time"] = current_time

        obs = next_obs
        episode_steps += 1
            
        if done:
            try:
                # 记录完成的episode指标
                metrics["episodes_completed"] += 1
                metrics["avg_episode_reward"] = metrics["total_reward"] / metrics["episodes_completed"]
                metrics["max_episode_reward"] = max(metrics["max_episode_reward"], episode_reward)
                metrics["min_episode_reward"] = min(metrics["min_episode_reward"], episode_reward)
                
                # 记录到MLflow
                if args.actor_mlflow_tracking:
                    try:
                        if mlflow.active_run():
                            mlflow.log_metrics({
                                "actor_avg_episode_reward": metrics["avg_episode_reward"],
                                # "actor_episode_reward": episode_reward,
                                # "actor_episode_steps": episode_steps,
                                "actor_max_episode_reward": metrics["max_episode_reward"],
                                "actor_min_episode_reward": metrics["min_episode_reward"]
                            }, step=metrics["episodes_completed"])
                    except Exception as e:
                        print(f"[ACTOR{rank}] MLflow 记录episode指标失败: {e}")
                if rank == 0:
                    print(f"[ACTOR{rank}] Episode {metrics['episodes_completed']} 完成, 奖励: {episode_reward:.4f}, 步数: {episode_steps}")

                    # 绘制并保存本回合轨迹与奖励图
                    try:
                        plot_episode_positions_and_rewards(
                            positions=episode_positions,
                            rewards=episode_step_rewards,
                            env=env,
                            actor_rank=rank,
                            episode_index=metrics["episodes_completed"],
                            out_dir="debug_plots",
                            save_root=(getattr(args, 'model_dir', None) or (Path(getattr(args, 'model_path', ''))).parent if hasattr(args, 'model_path') else None),
                            display=bool(getattr(args, 'show_ui', False)),
                            window_name="Episode Debug (Actor 0)",
                            color_mode=str(getattr(args, 'plot_color_mode', 'continuous')),
                            vmin=float(getattr(args, 'plot_vmin', -0.25)),
                            vmax=float(getattr(args, 'plot_vmax', 0.25)),
                            bins=int(getattr(args, 'plot_bins', 50)),
                            cmap_name=str(getattr(args, 'plot_cmap', 'turbo')),
                            draw_heading=bool(getattr(args, 'plot_draw_heading', False))
                        )
                    except Exception as e:
                        print(f"[ACTOR{rank}] 回合调试可视化失败: {e}")

                # 计算本集是否成功：基于可用的 info 字段
                episode_success = False
                try:
                    for exp in reversed(episode_buffer):
                        info_dict = exp.get('info', {}) if isinstance(exp, dict) else {}
                        if info_dict.get('success', False):
                            episode_success = True
                            break
                except Exception as e:
                    print(f"[ACTOR{rank}] 计算episode成功标志失败，{e}")
                    episode_success = False
                recent_episode_success.append(bool(episode_success))

                # 将成功标志发送给 Learner 统计成功率
                try:
                    if stats_q is not None:
                        stats_q.put({
                            'actor': int(rank),
                            'episode_success': bool(episode_success)
                        }, timeout=0.01)
                except Full:
                    pass
                except Exception:
                    pass

                # 处理episode buffer中的经验（可选episode级奖励整形）
                if bool(getattr(args, 'enqueue_at_episode_end', True)) and bool(getattr(args, 'enable_episode_shaping', True)):
                    # 1. 计算episode总奖励
                    episode_total_reward = episode_reward
                    # 调整episode总奖励的影响
                    if episode_success:
                        episode_reward_factor = 0.01
                    else:
                        episode_reward_factor = 0.0
                    # 2. 计算episode-level距离奖励（基于最后一步与目标的距离）
                    if 'distance_to_target' in episode_buffer[-1]['info']:
                        final_distance = episode_buffer[-1]['info']['distance_to_target']
                        episode_distance_reward = 50.0 / (1.0 + final_distance)
                    else:
                        episode_distance_reward = 0.0
                    # 平均分配到每一步
                    per_step_episode_reward = episode_distance_reward / len(episode_buffer)
                    # 3. 修改每一步的奖励并发送到经验队列
                    for i, exp in enumerate(episode_buffer):
                        modified_reward = exp['reward'] + episode_total_reward * episode_reward_factor
                        modified_reward += per_step_episode_reward
                        if 'distance_to_target' in exp['info']:
                            step_distance = exp['info']['distance_to_target']
                            step_distance_reward = 10.0 / (1.0 + step_distance)
                            modified_reward += step_distance_reward
                        try:
                            exp_q.put((exp['obs'], exp['action'], modified_reward, exp['next_obs'], exp['done']), timeout=0.1)
                        except Full:
                            pass
                        except Exception as e:
                            print(f"[ACTOR{rank}] 写入经验异常: {e}")
                elif bool(getattr(args, 'enqueue_at_episode_end', True)) and not bool(getattr(args, 'enable_episode_shaping', True)):
                    # 回合结束写入，但不进行episode级整形，直接发送原始经验
                    for i, exp in enumerate(episode_buffer):
                        try:
                            exp_q.put((exp['obs'], exp['action'], exp['reward'], exp['next_obs'], exp['done']), timeout=0.1)
                        except Full:
                            pass
                        except Exception as e:
                            print(f"[ACTOR{rank}] 写入经验异常: {e}")
                
                # 清空episode buffer
                episode_buffer = []
                
                # 重置环境和计数器
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                # 重置actor0的采样缓存
                if rank == 0:
                    episode_positions = []
                    episode_step_rewards = []
            except Exception as e:
                print(f"[ACTOR{rank}] reset 异常: {e}")
                continue
    if args.actor_mlflow_tracking:
        try:
            if mlflow.active_run():
                # 记录最终指标
                duration = time.time() - start_time
                mlflow.log_metrics({
                    "actor_final_episodes": metrics["episodes_completed"],
                    "actor_final_steps": metrics["steps_completed"],
                    "actor_runtime_seconds": duration,
                    "actor_steps_per_second": metrics["steps_completed"] / max(1.0, duration)
                })
                mlflow.end_run()
                print(f"[ACTOR{rank}] MLflow 运行已结束")
        except Exception as e:
            print(f"[ACTOR{rank}] MLflow 结束运行失败: {e}")
            raise e
    print(f"[ACTOR{rank}] 进程退出")
    try:
        env.close()
    except Exception:
        pass
    # 关闭调试窗口
    try:
        if rank == 0 and bool(getattr(args, 'show_ui', False)) and 'cv2' in globals() and cv2 is not None:
            cv2.destroyWindow("Episode Debug (Actor 0)")
    except Exception:
        pass


def run_distributed_training(args: argparse.Namespace, model_path: str):
    """多Actor+单Learner 训练主控"""
    mp.set_start_method('spawn', force=True)
    num_actors = int(getattr(args, 'num_actors', max(1, args.num_envs if hasattr(args, 'num_envs') else 1)))
    # 将模型保存目录传递给子进程，便于actor绘图保存
    try:
        args.model_path = model_path
        args.model_dir = str(Path(model_path).parent)
    except Exception:
        pass
    
    # 初始化MLflow主运行
    experiment_name = args.experiment_name or f"distributed_training_{args.cargo_type}"
    run_name = f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # try:
    #     mlflow.set_experiment(experiment_name)
    #     mlflow.start_run(run_name=run_name)
        
    #     # 记录分布式训练参数
    #     mlflow.log_params({
    #         "cargo_type": args.cargo_type,
    #         "num_actors": num_actors,
    #         "total_steps": getattr(args, 'total_steps', 50000),
    #         "device": args.device if hasattr(args, 'device') else 'auto',
    #         "distributed_mode": True,
    #         "model_path": model_path
    #     })
    #     print(f"[MAIN] MLflow 跟踪已初始化: {experiment_name}/{run_name}")
    # except Exception as e:
    #     print(f"[MAIN] MLflow 初始化失败: {e}")

    # 队列：每个 actor 配一对 obs/act 队列；经验共享一个队列
    obs_queues = [mp.Queue(maxsize=8) for _ in range(num_actors)]
    act_queues = [mp.Queue(maxsize=8) for _ in range(num_actors)]
    exp_queue = mp.Queue(maxsize=4096)
    stats_queue = mp.Queue(maxsize=1024)
    stop_event = mp.Event()
    init_event = mp.Event()  # 初始化完成信号

    # Learner 进程
    learner_proc = mp.Process(target=_learner_process, args=(args, model_path, obs_queues, act_queues, exp_queue, stats_queue, stop_event, init_event), daemon=True)
    learner_proc.start()
    
    # 等待 Learner 初始化完成
    print("[MAIN] 等待 Learner 初始化完成...")
    init_event.wait()  # 阻塞直到 learner 设置 init_event
    print("[MAIN] Learner 初始化完成，开始启动 Actors...")
    # 启动 Actors
    actors = []
    # 第一个actor启动图形化界面，其余均为无头无渲染
    for i in range(num_actors):
        if i == 0 and args.show_ui:
            args.headless = False
            args.no_rendering = False
        else:
            args.headless = True
            args.no_rendering = True
        p = mp.Process(target=_actor_process, args=(i, args.cargo_type, args, obs_queues[i], act_queues[i], exp_queue, stats_queue, stop_event), daemon=True)
        p.start()
        actors.append(p)

    try:
        # 主进程只需等待 Learner 结束或 KeyboardInterrupt
        learner_proc.join()
    except KeyboardInterrupt:
        print("\n分布式训练被中断，正在清理...")
    finally:
        stop_event.set()
        for p in actors:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        try:
            learner_proc.join(timeout=5)
        except Exception:
            pass
        
        # 结束MLflow主运行
        # try:
        #     if mlflow.active_run():
        #         # 记录训练结束状态
        #         #mlflow.log_param("training_completed", "interrupted" if sys.exc_info()[0] else "completed")
        #         mlflow.end_run()
        #         print("[MAIN] MLflow 运行已结束")
        # except Exception as e:
        #     print(f"[MAIN] MLflow 结束运行失败: {e}")


def _learner_process(args: argparse.Namespace, model_path: str, obs_queues: list, act_queues: list, exp_queue: mp.Queue, stats_queue: mp.Queue, stop_event: mp.Event, init_event: mp.Event):
    """Learner：集中选择动作并训练TD3，异步消费经验。"""
    print("[LEARNER] 进程启动...")
    
    # 初始化MLflow跟踪
    experiment_name = args.experiment_name
    run_name = f"learner_VerticalTest_{args.curriculum_stage}_{args.total_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 设置MLflow实验
    try:
        mlflow.set_experiment(experiment_name)
        # 开始一个新的MLflow运行（作为主运行的子运行，便于在同一实验下聚合）
        mlflow.start_run(run_name=run_name, nested=True)
        
        # 记录训练参数
        mlflow.log_params({
            "cargo_type": args.cargo_type,
            "learning_rate": getattr(args, 'learning_rate', 3e-4),
            "buffer_size": getattr(args, 'buffer_size', 100000),
            "learning_starts": getattr(args, 'learning_starts', 10000),
            "batch_size": getattr(args, 'batch_size', 256),
            "gamma": getattr(args, 'gamma', 0.98),
            "tau": getattr(args, 'tau', 0.005),
            "gradient_steps": getattr(args, 'gradient_steps', 1),
            "train_freq": getattr(args, 'train_freq', 1),
            "policy_delay": getattr(args, 'policy_delay', 2),
            "target_policy_noise": getattr(args, 'target_noise', 0.2),
            "target_noise_clip": getattr(args, 'noise_clip', 0.5),
            "num_actors": len(obs_queues),
            "total_steps": getattr(args, 'total_steps', 50000),
            "macro_action_steps": getattr(args, 'macro_action_steps', 1),
            "action_mode": getattr(args, 'action_mode', 'wheels'),
            "obs_mode": getattr(args, 'obs_mode', 'local_map'),
            "device": getattr(args, 'device', 'cuda'),
            "enqueue_at_episode_end": getattr(args, 'enqueue_at_episode_end', False),
            "enable_episode_shaping": getattr(args, 'enable_episode_shaping', False),
            "delta_distance_k": getattr(args, 'delta_distance_k', 30.0),
            "movement_reward_k": getattr(args, 'movement_reward_k', 0.5),
            "liner_distance_reward": getattr(args, 'liner_distance_reward', False),
            "distance_k": getattr(args, 'distance_k', 1.0),
            "time_k": getattr(args, 'time_k', 0.2),
            "wall_proximity_penalty_k": getattr(args, 'wall_proximity_penalty_k', 2.0),
            "angle_reward_k": getattr(args, 'angle_reward_k', 2.0),
            "angle_change_k": getattr(args, 'angle_change_k', 10.0),
            "early_spin_penalty_k": getattr(args, 'early_spin_penalty_k', 1.0),
            "front_clear_k": getattr(args, 'front_clear_k', 1.0),
            "stop_bonus_k": getattr(args, 'stop_bonus_k', 0),
            "approach_reward_k": getattr(args, 'approach_reward_k', 0),
            "slow_down_reward_k": getattr(args, 'slow_down_reward_k', 0),
            # 动态超参（学习率与噪声）
            "lr_schedule_type": getattr(args, 'lr_schedule_type', 'linear'),
            "lr_final": (getattr(args, 'lr_final', None) if getattr(args, 'lr_final', None) is not None else getattr(args, 'learning_rate', 3e-4)),
            "exploration_noise_type": getattr(args, 'exploration_noise_type', 'linear'),
            "exploration_noise_init": getattr(args, 'exploration_noise_init', 0.3),
            "exploration_noise_final": getattr(args, 'exploration_noise_final', 0.05),
            "target_policy_noise_init": getattr(args, 'target_policy_noise_init', 0.2),
            "target_policy_noise_final": getattr(args, 'target_policy_noise_final', 0.05),
            "target_noise_clip_init": getattr(args, 'target_noise_clip_init', 0.5),
            "target_noise_clip_final": getattr(args, 'target_noise_clip_final', 0.1)
        })
        print(f"[LEARNER] MLflow 跟踪已初始化: {experiment_name}/{run_name}")
    except Exception as e:
        print(f"[LEARNER] MLflow 初始化失败: {e}")
    
    device = args.device if hasattr(args, 'device') else 'auto'

    # 直接通过静态方法获取空间定义，避免启动临时 Webots 实例

    try:
        # 使用多输入观测空间：固定 local_map，关闭 robot_state，开启最小导航信息
        obs_space, act_space = ROSbotNavigationEnv.get_spaces(
            include_robot_state=False,
            include_navigation_info=True,
            nav_info_mode='minimal',
            macro_action_steps=getattr(args, 'macro_action_steps', 1),
            action_mode=getattr(args, 'action_mode', 'wheels'),
            obs_mode=str(getattr(args, 'obs_mode', 'local_map'))
        )
        print("[LEARNER] 获取环境空间完成(静态)")
    except Exception as e:
        print(f"[LEARNER] 获取环境空间失败: {e}")
        raise

    # 创建一个 Dummy 环境供 SB3 初始化（最小实现）
    import gymnasium as gym
    class _FakeEnv(gym.Env):
        def __init__(self, obs_space, act_space):
            self.observation_space = obs_space
            self.action_space = act_space
        def reset(self, *, seed=None, options=None):
            obs = self.observation_space.sample()
            return obs, {}
        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, True, False, {}

    fake_env = _FakeEnv(obs_space, act_space)
    print(f"{obs_space}, {act_space}")
    # TensorBoard logging root and run name
    log_root = getattr(args, 'tensorboard_log', '/root/workspace/RL_car2/rosbot_navigation/logs')
    _tb_arg = getattr(args, 'tb_log_name', None)
    tb_log_name = _tb_arg if _tb_arg else f"stage1_{getattr(args,'cargo_type','normal')}_{getattr(args,'curriculum_stage','start')}"
    
    # ===== 课程链式加载：若提供上一课程模型或可自动发现，则优先加载 =====
    model = None
    try:
        prev_path = getattr(args, 'prev_model_path', None)
        if prev_path and Path(prev_path).exists():
            print(f"[LEARNER] 从提供的上一课程模型加载: {prev_path}")
            model = TD3.load(prev_path, env=fake_env, device=device)
            try:
                model.set_logger(
                    sb3_configure_logger(
                        verbose=int(getattr(args, 'verbose', 1)),
                        tensorboard_log=log_root,
                        tb_log_name=tb_log_name,
                        reset_num_timesteps=True,
                    )
                )
            except Exception:
                pass
        else:
            # 自动发现：基于课程阶段顺序查找上一阶段模型
            stage_order = ['start','easy','medium','hard','end','all']
            cur_stage = getattr(args, 'curriculum_stage', 'start')
            if cur_stage in stage_order and cur_stage != 'start':
                prev_stage = stage_order[max(0, stage_order.index(cur_stage)-1)]
                pattern = f"td3_{getattr(args,'cargo_type','normal')}_stage1_{prev_stage}_*.zip"
                candidates = list(Path('./results').rglob(pattern))
                if candidates:
                    # 选择最近修改的模型
                    best = max(candidates, key=lambda p: p.stat().st_mtime)
                    print(f"[LEARNER] 自动发现上一课程模型: {best}")
                    model = TD3.load(str(best), env=fake_env, device=device)
                    try:
                        model.set_logger(
                            sb3_configure_logger(
                                verbose=int(getattr(args, 'verbose', 1)),
                                tensorboard_log=log_root,
                                tb_log_name=tb_log_name,
                                reset_num_timesteps=True,
                            )
                        )
                    except Exception:
                        pass
    except Exception as e:
        print(f"[LEARNER] 加载上一课程模型失败，使用新模型初始化: {e}")
        model = None

    if model is None :
        policy_name = 'MlpPolicy' if getattr(args, 'obs_mode', 'local_map') == 'lidar' else 'MultiInputPolicy'
        model = TD3(
                    policy_name,
                    fake_env,
                    learning_rate=getattr(args, 'learning_rate', 3e-4),
                    buffer_size=getattr(args, 'buffer_size', 100000),
                    learning_starts=getattr(args, 'learning_starts', 10000),
                    batch_size=getattr(args, 'batch_size', 256),
                    gamma=getattr(args, 'gamma', 0.98),
                    tau=getattr(args, 'tau', 0.005),
                    gradient_steps=getattr(args, 'gradient_steps', 1),
                    train_freq=getattr(args, 'train_freq', 1),
                    policy_delay=getattr(args, 'policy_delay', 2),
                    # target_policy_noise=getattr(args, 'target_noise', 0.2),
                    # target_noise_clip=getattr(args, 'noise_clip', 0.5),
                    verbose=int(getattr(args, 'verbose', 1)),
                    device=device,
                    tensorboard_log=log_root
        )
        # Configure SB3 logger so that manual training (model.train) has a logger
        tb_root_for_logger = None if (log_root is None or str(log_root).strip()=="") else log_root
        model.set_logger(
            sb3_configure_logger(
                verbose=int(getattr(args, 'verbose', 1)),
                tensorboard_log=tb_root_for_logger,
                tb_log_name=tb_log_name,
                reset_num_timesteps=True,
            )
        )
    
    # Ensure logger is set even when model was loaded (avoid AttributeError: _logger)
    try:
        _ = model.logger  # access to verify presence
    except Exception:
        tb_root_for_logger = None if (log_root is None or str(log_root).strip()=="") else log_root
        model.set_logger(
            sb3_configure_logger(
                verbose=int(getattr(args, 'verbose', 1)),
                tensorboard_log=tb_root_for_logger,
                tb_log_name=tb_log_name,
                reset_num_timesteps=True,
            )
        )

    print("[LEARNER] TD3 模型初始化完成")
    # 向主进程发送初始化完成的信号
    init_event.set()
    # 分布式主循环
    total_steps = args.total_steps
    collected = 0
    last_print = 0
    actions_served = 0  
    
    # 奖励跟踪变量
    reward_history = []  # 存储所有奖励
    episode_count = 0
    last_reward_print = 0
    last_mlflow_log = 0  # MLflow日志记录时间点
    last_step_log = 0    # 基于步数的MLflow记录节流
    # 成功率统计（来自各个 actor 的episode结束报告）
    learner_success_count = 0
    learner_episode_count = 0
    # 早停窗口统计
    early_stop_window = int(getattr(args, 'early_stop_window', 100))
    early_stop_threshold = float(getattr(args, 'early_stop_success', 0.9))
    recent_episode_success_global = deque(maxlen=early_stop_window)
    add_new_exp = 0
    # 训练指标缓存（在降频打印/记录时使用）
    actor_loss_last = None
    critic_loss_last = None
    learning_rate_last = None
    exploration_sigma_last = None
    target_policy_noise_last = None
    n_updates_last = None
    sb3_learning_rate_last = None  # 仅记录 SB3 logger 报告的学习率，不覆盖我们动态学习率

    # ===== 动态调度函数（学习率与噪声）=====
    def _progress(collected_steps: int, total: int) -> float:
        total = max(1, int(total))
        p = max(0.0, min(1.0, float(collected_steps) / float(total)))
        return p

    def _interp(kind: str, start: float, end: float, t: float) -> float:
        if kind == 'cosine':
            import math
            c = (1.0 - math.cos(math.pi * max(0.0, min(1.0, t)))) * 0.5
            return start + (end - start) * c
        # default linear
        return start + (end - start) * max(0.0, min(1.0, t))

    def get_current_lr(collected_steps: int, total: int) -> float:
        sched = getattr(args, 'lr_schedule_type', 'linear')
        lr_start = float(getattr(args, 'learning_rate', 3e-4))
        _lr_end = getattr(args, 'lr_final', None)
        lr_end = lr_start if (_lr_end is None) else float(_lr_end)
        t = _progress(collected_steps, total)
        val = _interp(sched, lr_start, lr_end, t)
        return max(1e-8, float(val))

    def get_exploration_sigma(collected_steps: int, total: int) -> float:
        sched = getattr(args, 'exploration_noise_type', 'linear')
        n0 = float(getattr(args, 'exploration_noise_init', 0.3))
        n1 = float(getattr(args, 'exploration_noise_final', 0.05))
        t = _progress(collected_steps, total)
        return max(0.0, float(_interp(sched, n0, n1, t)))

    def get_target_policy_noise(collected_steps: int, total: int) -> float:
        sched = getattr(args, 'exploration_noise_type', 'linear')
        n0 = float(getattr(args, 'target_policy_noise_init', 0.2))
        n1 = float(getattr(args, 'target_policy_noise_final', 0.05))
        t = _progress(collected_steps, total)
        return max(0.0, float(_interp(sched, n0, n1, t)))

    def get_target_noise_clip(collected_steps: int, total: int) -> float:
        sched = getattr(args, 'exploration_noise_type', 'linear')
        c0 = float(getattr(args, 'target_noise_clip_init', 0.5))
        c1 = float(getattr(args, 'target_noise_clip_final', 0.1))
        t = _progress(collected_steps, total)
        return max(0.0, float(_interp(sched, c0, c1, t)))

    def _set_optimizer_lr(opt, lr: float):
        try:
            for pg in opt.param_groups:
                pg['lr'] = float(lr)
        except Exception:
            pass

    print(f"[LEARNER] 开始主循环，目标步数: {total_steps}")

    try:
        while not stop_event.is_set() and collected < total_steps:
            # 优先处理所有动作请求（避免 actor 超时）
            # 先消费来自actors的成功统计
            try:
                while True:
                    stat = stats_queue.get_nowait()
                    try:
                        if isinstance(stat, dict) and 'episode_success' in stat:
                            learner_episode_count += 1
                            success_flag = bool(stat.get('episode_success', False))
                            if success_flag:
                                learner_success_count += 1
                            # 将成功标志加入滚动窗口，供后续窗口成功率计算
                            try:
                                recent_episode_success_global.append(success_flag)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Empty:
                pass

            for i, (oq, aq) in enumerate(zip(obs_queues, act_queues)):
                # 避免使用不可靠的 empty()，直接使用 get_nowait 循环取尽
                while True:
                    try:
                        aid, obs = oq.get_nowait()
                    except Exception:
                        break  # 队列暂时无数据，处理下一个 actor
                    try:
                        # 预处理观测：确保为 dict[np.float32 ndarray]
                        if isinstance(obs, dict):
                            obs_proc = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
                        else:
                            obs_proc = np.asarray(obs, dtype=np.float32)
                        # 通过 policy.predict 获得动作
                        with torch.no_grad():
                            #print("obs_proc", obs_proc)
                            action, _ = model.predict(obs_proc, deterministic=True)
                        # 动态探索噪声（按步数衰减）
                        try:
                            sigma = get_exploration_sigma(collected, total_steps)
                            exploration_sigma_last = float(sigma)
                            if sigma > 0.0:
                                noise = np.random.normal(loc=0.0, scale=sigma, size=np.shape(action)).astype(np.float32)
                                action = np.asarray(action, dtype=np.float32) + noise
                            # 动作裁剪到动作空间范围
                            if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                                action = np.clip(action, act_space.low, act_space.high)
                        except Exception:
                            action = np.asarray(action, dtype=np.float32)
                        aq.put(action, timeout=1.0)
                        actions_served += 1
                    except Exception as e:
                        print(f"[LEARNER] 生成/发送动作异常: {e}")
                        # 当前 actor 发送失败则跳出其处理，避免死循环
                        break

            #experiences_added = 0
            
            for _ in range(64):
                try:
                    obs, action, reward, next_obs, done = exp_queue.get_nowait()
                    # 记录奖励
                    reward_history.append(reward)
                    # 加入回放缓冲
                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=next_obs,
                        action=action,
                        reward=reward,
                        done=done,
                        infos=[{}]
                    )
                    add_new_exp += 1
                    collected += 1

                    # 每新增一条经验，尝试立即训练一次（达到学习启动阈值后）
                    if collected >= getattr(args, 'learning_starts', 10000) and collected % getattr(args, 'train_freq', 4) == 0:
                        # 动态学习率与目标策略噪声（按收集步数实时更新）
                        try:
                            current_lr = get_current_lr(collected, total_steps)
                            learning_rate_last = float(current_lr)
                            # 更新优化器学习率（兼容不同SB3结构）
                            try:
                                if hasattr(model, 'policy') and hasattr(model.policy, 'actor') and hasattr(model.policy.actor, 'optimizer'):
                                    _set_optimizer_lr(model.policy.actor.optimizer, current_lr)
                            except Exception:
                                pass
                            try:
                                if hasattr(model, 'policy') and hasattr(model.policy, 'critic') and hasattr(model.policy.critic, 'optimizer'):
                                    _set_optimizer_lr(model.policy.critic.optimizer, current_lr)
                            except Exception:
                                pass
                            try:
                                if hasattr(model, 'actor') and hasattr(model.actor, 'optimizer'):
                                    _set_optimizer_lr(model.actor.optimizer, current_lr)
                            except Exception:
                                pass
                            try:
                                if hasattr(model, 'critic') and hasattr(model.critic, 'optimizer'):
                                    _set_optimizer_lr(model.critic.optimizer, current_lr)
                            except Exception:
                                print("[LEARNER] 更新优化器学习率失败")
                                pass
                            # 更新 TD3 目标策略噪声（用于目标值平滑）
                            tp_noise = get_target_policy_noise(collected, total_steps)
                            tp_clip = get_target_noise_clip(collected, total_steps)
                            target_policy_noise_last = float(tp_noise)
                            if hasattr(model, 'target_policy_noise'):
                                model.target_policy_noise = float(tp_noise)
                            if hasattr(model, 'target_noise_clip'):
                                model.target_noise_clip = float(tp_clip)
                        except Exception:
                            print("[LEARNER] 更新 TD3 目标策略噪声/学习率失败")
                            pass

                        # 单步训练
                        model.train(gradient_steps=1, batch_size=model.batch_size)

                        # 抓取本次训练产生的关键指标，缓存起来，按 log_interval 再统一输出/记录
                        try:
                            nv = getattr(model, 'logger').name_to_value if hasattr(model, 'logger') else {}
                            if isinstance(nv, dict):
                                if "train/actor_loss" in nv:
                                    actor_loss_last = float(nv["train/actor_loss"])  # type: ignore[arg-type]
                                if "train/critic_loss" in nv:
                                    critic_loss_last = float(nv["train/critic_loss"])  # type: ignore[arg-type]
                                if "train/learning_rate" in nv:
                                    sb3_learning_rate_last = float(nv["train/learning_rate"])  # type: ignore[arg-type]
                                if "train/n_updates" in nv:
                                    try:
                                        n_updates_last = int(nv["train/n_updates"])  # type: ignore[arg-type]
                                    except Exception:
                                        n_updates_last = int(float(nv["train/n_updates"]))  # type: ignore[arg-type]
                        except Exception:
                            pass
                except Empty:
                    # 队列为空，退出循环
                    break
                except Exception as e:
                    # 其他异常，记录后跳过当前条目，继续尝试获取更多经验
                    print(f"[LEARNER] 读取经验异常: {e}")
                    raise e

            # 训练已在每次写入经验后进行（见上方循环），此处无需再次批量训练

            # 降频打印
            if collected - last_print >= getattr(args, 'log_interval', 1000) and collected > 1000:
                last_print = collected
                # 计算最近100个奖励的平均值
                recent_rewards = reward_history[-1000:] if len(reward_history) > 1000 else reward_history
                recent_avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                # 打印核心训练与采样信息
                al = f"{actor_loss_last:.6f}" if actor_loss_last is not None else "NA"
                cl = f"{critic_loss_last:.6f}" if critic_loss_last is not None else "NA"
                lr = f"{learning_rate_last:.2e}" if learning_rate_last is not None else "NA"
                nu = f"{n_updates_last}" if n_updates_last is not None else "NA"
                print(
                    f"[LEARNER] 收集样本: {collected}, 缓冲区大小: {model.replay_buffer.size()}, "
                    f"最近1000步平均奖励: {recent_avg_reward:.4f}, actor_loss: {al}, critic_loss: {cl}, lr: {lr}, n_updates: {nu}"
                )
                # 奖励统计改为使用滚动窗口（与 recent_rewards 一致）
                window_avg_reward = recent_avg_reward
                window_max_reward = max(recent_rewards) if recent_rewards else 0.0
                window_min_reward = min(recent_rewards) if recent_rewards else 0.0
                # 在此处不再修改成功率计数与窗口，仅在消费 stats_queue 处更新
                # 仅用于后续日志：当窗口已满时计算 recent_sr，否则置为 None
                recent_sr = None
                if len(recent_episode_success_global) == early_stop_window:
                    recent_sr = sum(1 for s in recent_episode_success_global if s) / float(early_stop_window)
                try:
                    if mlflow.active_run():
                        metrics = {
                            "recent_1000_avg_reward": recent_avg_reward,
                            "critic_loss": critic_loss_last if critic_loss_last is not None else 0.0,
                            "actor_loss": actor_loss_last if actor_loss_last is not None else 0.0,
                            "n_updates": n_updates_last if n_updates_last is not None else 0,
                            "learning_rate": learning_rate_last if learning_rate_last is not None else get_current_lr(collected, total_steps),
                            "sb3_learning_rate": sb3_learning_rate_last if sb3_learning_rate_last is not None else 0.0,
                            "exploration_sigma": exploration_sigma_last if exploration_sigma_last is not None else get_exploration_sigma(collected, total_steps),
                            "target_policy_noise": target_policy_noise_last if target_policy_noise_last is not None else get_target_policy_noise(collected, total_steps),
                            "target_noise_clip": get_target_noise_clip(collected, total_steps),
                            "learner_episode_count": learner_episode_count,
                            "learner_success_count": learner_success_count,
                            "average_reward": window_avg_reward,
                            "max_reward": window_max_reward,
                            "min_reward": window_min_reward,
                        }
                        if recent_sr is not None:
                            metrics["recent_success_rate"] = recent_sr
                        mlflow.log_metrics(metrics, step=collected)
                except Exception as e:
                    print(f"[LEARNER] MLflow 记录指标失败: {e}")                 
                
                if (recent_sr is not None) and (recent_sr >= early_stop_threshold):
                    print(f"[LEARNER] 早停触发: 最近{early_stop_window}集成功率 {recent_sr*100:.1f}% >= {early_stop_threshold*100:.1f}% ，请求停止训练")
                    try:
                        stop_event.set()
                    except Exception:
                        pass
            # 基于步数的 MLflow 记录
                # # 将关键指标也同步到 SB3 Logger，并在该频率下 flush 到各输出（如 stdout / TB）
                # try:
                #     if actor_loss_last is not None:
                #         model.logger.record("train/actor_loss", float(actor_loss_last))
                #     if critic_loss_last is not None:
                #         model.logger.record("train/critic_loss", float(critic_loss_last))
                #     if learning_rate_last is not None:
                #         model.logger.record("train/learning_rate", float(learning_rate_last))
                #     if n_updates_last is not None:
                #         model.logger.record("train/n_updates", int(n_updates_last))
                #     model.logger.record("train/collected_steps", float(collected), exclude=("stdout",))
                #     model.logger.dump(step=int(collected))
                # except Exception:
                #     pass
            
    except KeyboardInterrupt:
        print("[LEARNER] 收到中断信号")
    except Exception as e:
        print(f"[LEARNER] 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[LEARNER] 保存模型...")
        try:
            model.save(model_path)
            print(f"[LEARNER] 模型已保存到: {model_path}")
            
            # 将模型保存到MLflow
            try:
                mlflow.pytorch.log_model(model.policy, "policy_model")
                # 记录最终指标
                if reward_history:
                    mlflow.log_metrics({
                        "final_avg_reward": sum(reward_history) / len(reward_history),
                        "total_steps": collected,
                    })
                # 结束MLflow运行
                mlflow.end_run()
                print("[LEARNER] MLflow 运行已完成并记录")
            except Exception as e:
                print(f"[LEARNER] MLflow 记录模型失败: {e}")
        except Exception as e:
            print(f"[LEARNER] 保存模型失败: {e}")

def main():
    now = datetime.now().replace(microsecond=0)
    """主函数"""
    parser = argparse.ArgumentParser(description='ROSbot第一阶段训练脚本')

    #调试参数    
    parser.add_argument('--debug', type=bool, default=False, help='调试模式')
    parser.add_argument('--show_map', type=bool, default=False, help='是否显示地图')
    parser.add_argument('--show_ui', type=bool, default=True, help='是否显示UI')

    # 日志
    parser.add_argument('--log_interval', type=int, default=1000, help='控制台打印间隔步数')
    parser.add_argument('--mlflow_log_interval', type=int, default=500, help='MLflow 记录间隔步数')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='SB3 日志详细程度 (0=无,1=信息,2=调试)')
    parser.add_argument('--tensorboard_log', type=str, default='/root/workspace/RL_car2/rosbot_navigation/logs', help='TensorBoard 日志根目录')
    parser.add_argument('--tb_log_name', type=str, default=None, help='TensorBoard 运行名称 (默认根据阶段和货物自动生成)')
    parser.add_argument('--actor_mlflow_tracking', type=bool, default=False, help='是否启用Actor的MLflow跟踪')
    parser.add_argument('--experiment_name', type=str, default='101_vertical', help='MLflow 实验名称')
    parser.add_argument('--remark', type=str, default='vertical_test', help='备注')
    parser.add_argument('--plot_vmin', type=float, default=-150, help='绘图最小值')
    parser.add_argument('--plot_vmax', type=float, default=150, help='绘图最大值')

    # 分布式训练参数
    parser.add_argument('--distributed', type=bool, default=True, help='启用多Actor+单Learner分布式训练')
    parser.add_argument('--num_actors', type=int, default=4, help='Actor 数量')
    parser.add_argument('--num_envs', type=int, default=4, help='并行环境数量（>1启用多进程并行）')

    # Webots参数
    # world参数将被每个actor覆盖为 env1/env2 对半分配，这里保留但不使用
    parser.add_argument('--world_1', type=str, default='/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env1.wbt', help='默认 Webots world 文件路径（分布式模式下按actor覆盖）')
    parser.add_argument('--world_2', type=str, default='/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env1.wbt', help='默认 Webots world 文件路径（分布式模式下按actor覆盖）')
    parser.add_argument('--headless', type=bool,default=True, help='以无渲染/批处理模式启动 Webots')
    parser.add_argument('--fast_mode', type=bool,default=True, help='使用Webots FAST模式')
    parser.add_argument('--no-rendering',type=bool,default=True, help='渲染模式')
    parser.add_argument('--batch', type=bool,default=True, help='批处理模式')
    parser.add_argument('--minimize', type=bool,default=True, help='最小化模式')
    parser.add_argument('--control_period_ms', type=int, default=200, help='控制周期(ms)，用于减少控制往返')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')

    # 宏动作步数：1 表示一次控制，5 表示 5×2 宏动作
    parser.add_argument('--macro_action_steps', type=int, default=1, help='每步执行的子动作数量（1 表示一次控制，>1 表示宏动作，例如 5 表示5×2）')
    # 动作模式：wheels 表示左右轮百分比；twist 表示线速度/角速度百分比
    parser.add_argument('--action_mode', type=str, default='wheels', choices=['wheels','twist'], help='动作模式：wheels=左右轮百分比，twist=线速度/角速度百分比')
    # 观测模式：local_map 使用局部栅格地图（MultiInputPolicy）；lidar 使用20维LiDAR向量（MlpPolicy）
    parser.add_argument('--obs_mode', type=str, default='lidar', choices=['local_map','lidar'], help='观测模式：local_map=局部地图，lidar=20维激光向量')
    # 动作平滑（限幅）开关
    parser.add_argument('--enable_speed_smoothing', type=bool, default=False, help='是否启用速度平滑（单步限幅）')

    # 课程学习阶段
    parser.add_argument('--curriculum_stage', type=str, default='end', choices=['start','easy','medium','hard','hard2','end','all','small'], help='课程学习阶段')
    parser.add_argument('--prev_model_path', type=str, default='', help='上一课程模型路径（若不提供则自动搜索）')
    parser.add_argument('--cargo_type', type=str, default='normal', choices=['normal', 'fragile', 'dangerous'],help='货物类型')
    parser.add_argument('--models_dir', type=str, default=None,help='模型保存路径')

    # TD3算法相关参数
    parser.add_argument('--total_steps', type=int, default=150000,help='总训练步数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备 (e.g., "cpu", "cuda", "auto")')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--buffer_size', type=int, default=50000, help='经验回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=2000, help='预热步数，开始学习前收集的样本数量')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数')
    parser.add_argument('--gradient_steps', type=int, default=1, help='每步梯度更新次数')
    parser.add_argument('--train_freq', type=int, default=1, help='训练频率')
    parser.add_argument('--policy_delay', type=int, default=2, help='策略延迟更新步数')
    parser.add_argument('--step_per_train', type=int, default=4, help='每次训练次相隔步数')
    # parser.add_argument('--target_noise', type=float, default=0.05, help='目标策略噪声')
    # parser.add_argument('--noise_clip', type=float, default=0.2, help='噪声裁剪范围')
    
    # 动态学习率与噪声调度
    parser.add_argument('--lr_schedule_type', type=str, default='linear', choices=['linear', 'cosine'], help='学习率调度策略')
    parser.add_argument('--lr_final', type=float, default=5e-5, help='最终学习率（为空则等于初始学习率）')
    parser.add_argument('--exploration_noise_type', type=str, default='linear', choices=['linear', 'cosine'], help='动作探索噪声调度策略')
    parser.add_argument('--exploration_noise_init', type=float, default=0.1, help='初始动作探索噪声 sigma')
    parser.add_argument('--exploration_noise_final', type=float, default=0.05, help='最终动作探索噪声 sigma')
    parser.add_argument('--target_policy_noise_init', type=float, default=0.05, help='TD3 目标策略噪声初始值')
    parser.add_argument('--target_policy_noise_final', type=float, default=0.05, help='TD3 目标策略噪声最终值')
    parser.add_argument('--target_noise_clip_init', type=float, default=0.2, help='TD3 目标噪声裁剪初始值')
    parser.add_argument('--target_noise_clip_final', type=float, default=0.2, help='TD3 目标噪声裁剪最终值')
        
    # 早停相关：最近N集成功率超过阈值则停止
    parser.add_argument('--early_stop_window', type=int, default=100, help='早停窗口大小（最近N集）')
    parser.add_argument('--early_stop_success', type=float, default=0.8, help='早停成功率阈值（0-1）')

    # 经验写入时机：True=回合结束再写入（可配合episode整形）；False=逐步立即写入
    parser.add_argument('--enqueue_at_episode_end', type=bool, default=True, help='是否在回合结束后再写入经验（否则逐步写入）')
    # 是否启用 episode 级别奖励整形（在每集结束后对整集经验的奖励进行再分配）
    parser.add_argument('--enable_episode_shaping', type=bool, default=False, help='是否启用 episode 级别奖励整形')

    # 奖励系数参数
    parser.add_argument('--delta_distance_k', type=float, default=50.0, help='距离变化奖励系数，以最大速度1.18m/s计，基础最大值约为0.7，线性')
    parser.add_argument('--movement_reward_k', type=float, default=0.5, help='移动奖励系数，基础最大值为10，线性')

    parser.add_argument('--liner_distance_reward', type=int, default=2,choices=[0, 1, 2], help='0:线性 1：负二次型 2：反比例')
    parser.add_argument('--distance_k', type=float, default=0.5, help='基础距离奖励系数，基础最大值为50')

    parser.add_argument('--time_k', type=float, default=0.02, help='时间惩罚系数，线性')   
    parser.add_argument('--wall_proximity_penalty_k', type=float, default=5, help='墙壁接近惩罚系数，每条线最大0.8，一共二十线，基础最大值14，线性')

    parser.add_argument('--angle_reward_k', type=float, default=5, help='角度奖励系数，基础最大值为10，二次型')
    parser.add_argument('--angle_change_k', type=float, default=15.0, help='角度变化奖励,以单步最大角度变化1计，基础最大值1，线性')
    parser.add_argument('--directional_movement_k', type=float, default=35.0, help='方向性移动奖励系数，鼓励朝目标方向移动而非随机移动')
    parser.add_argument('--early_spin_penalty_k', type=float, default=1.0, help='早期原地打转惩罚系数，负二次型')
    parser.add_argument('--front_clear_k', type=float, default=3.0, help='前方有路奖励系数，基础最大值约为7，七条线求和，线性')
    
    
    # 停用奖励
    parser.add_argument('--stop_bonus_k', type=float, default=0, help='停车奖励系数，基础最大值为10')   
    parser.add_argument('--approach_reward_k', type=float, default=0, help='接近奖励系数，基础最大值为10')  
    parser.add_argument('--slow_down_reward_k', type=float, default=0, help='接近目标减速奖励系数，基础最大值为10')
    args = parser.parse_args()
    
    
    # 设置调试模式
    if args.debug:
        print("启用调试模式")
        torch.autograd.set_detect_anomaly(True)
    # 模型文件路径
    
    # 将课程阶段加入模型文件名，便于课程链式加载
    if args.models_dir is None:
        name = f"{args.curriculum_stage}_{now}"
        # 创建模型保存目录
        results_dir = Path(f"/root/workspace/RL_car2/rosbot_navigation/results/test_vertical/{name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        models_dir = results_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 确保为 Path 对象
        models_dir = Path(args.models_dir)

    model_path = models_dir / f"td3_{args.curriculum_stage}_{args.total_steps}_{now}.zip"

    # 创建参数文件
    config_file_path = models_dir / f"config_{args.cargo_type}_{args.total_steps}_{now}.yaml"
    with open(config_file_path, 'w') as f:
        config_dict = vars(args)
        # 加入启动时间和备注信息
        config_dict['start_time'] = now.strftime("%Y-%m-%d %H:%M:%S")
        config_dict['remark'] = args.remark
        yaml.dump(config_dict, f)
    
    try:
        if args.distributed is True:
            print("\n开始分布式训练...")
            print("."*60)
            run_distributed_training(args, str(model_path))
            print(f"\n分布式训练完成！模型保存到: {model_path}")
        else:
            print("\n开始单进程训练...")
            print("."*60)
            
            model, env = train_single_cargo_model(
                cargo_type=args.cargo_type,
                total_steps=args.total_steps,
                model_save_path=str(model_path),
                device=args.device,
                args=args
            )
            
            print(f"\n训练完成！模型保存到: {model_path}")
            
            # 输出训练统计
            if hasattr(model, 'num_timesteps'):
                print(f"总训练步数: {model.num_timesteps}")
            
            # 创建训练完成标记
            completion_file = Path(model_path).parent / f"{args.cargo_type}_stage1_completed.txt"
            with open(completion_file, 'w') as f:
                f.write(f"Cargo type: {args.cargo_type}\n")
                f.write(f"Total steps: {args.total_steps}\n")
                f.write(f"Completed at: {datetime.now()}\n")
    except Exception as e:
        raise e
    finally:
        # 预启动实例清理（父进程兜底）
        if hasattr(args, '_prelaunch_pids') and args._prelaunch_pids:
            for pid in args._prelaunch_pids:
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
if __name__ == "__main__":
    # 将项目路径添加到Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main()


