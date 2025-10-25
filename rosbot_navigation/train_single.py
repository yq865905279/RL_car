"""
单线程标准训练脚本
使用 SB3 的 model.learn() 方法进行标准采样和训练
不使用分布式架构，由算法自行控制采样和训练流程
"""

import os
import sys
import signal
import argparse
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
import math
import cv2
import json

import mlflow
import mlflow.pytorch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3 import TD3
from src.environments.navigation_env import ROSbotNavigationEnv
from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
import matplotlib
matplotlib.use('Agg')  # 使用无交互后端
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

def plot_episode_positions_and_rewards(
    positions: List[tuple],
    rewards: List[float],
    env: ROSbotNavigationEnv = None,
    episode_index: int = 0,
    out_dir: str = "debug_plots",
    save_root: Optional[str] = None,
    display_plot: bool = False,
    window_name: str = "Episode Debug",
    color_mode: str = "continuous",
    vmin: float = -0.5,
    vmax: float = 0.5,
    bins: int = 100,
    cmap_name: str = "turbo",
    draw_heading: bool = False,
    draw_line: bool = False,
    aggregate_csv: Optional[str] = "trajectories.csv",
    aggregate_jsonl: Optional[str] = "trajectories.jsonl",
    draw_trajectory: bool = False,
    episode_task_info: Optional[Dict] = None,
    terminal_global_step: Optional[int] = None
) -> None:
    """
    将本回合内每一步车辆的位置与奖励绘制到图像中并保存（基于train_stage1.py实现）
    
    参数:
        positions: [(x, y), ...]，每一步的平面坐标
        rewards: [r1, r2, ...]，每一步奖励
        env: 可选，用于获取起点/终点等信息
        episode_index: 回合编号（用于文件命名）
        out_dir: 输出目录
        save_root: 保存根目录
        display_plot: 是否实时显示
        window_name: 显示窗口名称
        color_mode: 颜色模式 ("continuous" 或 "discrete")
        vmin: 奖励最小值
        vmax: 奖励最大值
        bins: 离散颜色分级数
        cmap_name: 颜色映射名称
        draw_heading: 是否绘制朝向箭头
    """
    try:
        if draw_trajectory:
            
            if plt is None:
                return
            if not positions or not rewards or len(positions) != len(rewards):
                return
            
            # 确定保存目录
            if save_root:
                base_dir = Path(save_root)
            else:
                base_dir = Path.cwd()
            
            save_dir = base_dir / out_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            xs = [float(p[0]) for p in positions]
            ys = [float(p[1]) for p in positions]
            total_reward = float(np.sum(rewards)) if len(rewards) > 0 else 0.0
            
            # 使用更宽的画布以匹配可行区域的长方形比例
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
            # 可选：绘制轨迹线（默认关闭，避免视觉上连接起点与终点）
            if draw_line:
                ax.plot(xs, ys, color='gray', linewidth=1.0, alpha=0.6, label='trajectory')
            
            # 颜色表示奖励：可选离散分级或连续渐变
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
            # 将颜色条移动到上方，使用水平布局；不设置标签，避免与 x 轴标签重叠
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.10)
            try:
                tick_vals = np.linspace(float(vmin), float(vmax), 5)
                cbar.set_ticks(tick_vals)
            except Exception:
                pass
            # 不设置 colorbar 标签，避免文字与 x 轴标签重合
            try:
                cbar.ax.xaxis.set_ticks_position('top')
            except Exception:
                pass
            
            # 标注起点与终点（优先使用传入的 episode_task_info）
            try:
                sp = None
                tp = None
                if episode_task_info is not None:
                    sp = episode_task_info.get('start_pos', None)
                    tp = episode_task_info.get('target_pos', None)
                elif env is not None and hasattr(env, 'task_info'):
                    sp = env.task_info.get('start_pos', None)
                    tp = env.task_info.get('target_pos', None)
                
                if sp is not None and len(sp) >= 2:
                    ax.scatter([float(sp[0])], [float(sp[1])], marker='*', s=120, c='green', label='start')
                if tp is not None and len(tp) >= 2:
                    ax.scatter([float(tp[0])], [float(tp[1])], marker='*', s=120, c='red', label='target')
            except Exception:
                pass

            # 绘制本回合激活的障碍物为正方形（边长0.6m，细线）
            try:
                obstacle_positions = None
                if episode_task_info is not None and 'obstacle_positions' in episode_task_info:
                    obstacle_positions = episode_task_info.get('obstacle_positions', None)
                elif env is not None and hasattr(env, '_active_obstacle_positions'):
                    obstacle_positions = getattr(env, '_active_obstacle_positions', None)
                if obstacle_positions:
                    added_label = False
                    for (ox, oy) in obstacle_positions:
                        try:
                            ox = float(ox); oy = float(oy)
                        except Exception:
                            continue
                        rect = Rectangle((ox - 0.3, oy - 0.3), 0.6, 0.6,
                                         linewidth=0.8, edgecolor='black', facecolor='none', alpha=0.9,
                                         label=('obstacle' if not added_label else None))
                        ax.add_patch(rect)
                        added_label = True
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

            # 边界框（保持大的约束框）
            try:
                boundary = Rectangle((-6.0, -4.0), 12.0, 8.0,
                                     linewidth=1.2, edgecolor='orange', facecolor='none',
                                     linestyle='--', alpha=0.9, label='boundary')
                ax.add_patch(boundary)
                # 设置矩形轴范围，便于观察细节
                ax.set_xlim(-6.2, 6.2)
                ax.set_ylim(-4.2, 4.2)
            except Exception:
                pass

            # 在标题标注回合步数与终止的全局步数
            term_global = int(terminal_global_step) if terminal_global_step is not None else None
            if term_global is not None:
                title_str = f"Episode {episode_index} | EpisodeSteps: {len(xs)} | GlobalStep: {term_global} | SumR: {total_reward:.2f}"
            else:
                title_str = f"Episode {episode_index} | EpisodeSteps: {len(xs)} | SumR: {total_reward:.2f}"
            ax.set_title(title_str)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.axis('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='best')

            # 在终止位置附近添加文本标注（显示全局步数）
            try:
                if len(xs) > 0:
                    note = f"term@{term_global}" if term_global is not None else f"term {len(xs)}"
                    ax.annotate(note, (xs[-1], ys[-1]), textcoords='offset points',
                                xytext=(5, 5), fontsize=8, color='k')
            except Exception:
                pass
            
            out_path = save_dir / f"episode_{episode_index:05d}.png"
            # 留出顶部空间以容纳水平 colorbar 与标题
            fig.tight_layout()
            fig.savefig(str(out_path))
        
        if aggregate_csv or aggregate_jsonl:
            # 保存CSV/JSON（支持聚合到单文件，或按回合单文件）
            try:
                # 组织通用数据
                episode_data = {
                    "episode_index": int(episode_index),
                    "total_steps": int(len(xs)),
                    "total_reward": float(total_reward),
                    "positions": [[float(x), float(y)] for x, y in zip(xs, ys)],
                    "rewards": [float(r) for r in rewards],
                }
                # 优先使用传入的 episode_task_info
                try:
                    if episode_task_info is not None:
                        episode_data["start_pos"] = episode_task_info.get('start_pos', None)
                        episode_data["target_pos"] = episode_task_info.get('target_pos', None)
                    elif env is not None and hasattr(env, 'task_info'):
                        episode_data["start_pos"] = env.task_info.get('start_pos', None)
                        episode_data["target_pos"] = env.task_info.get('target_pos', None)
                except Exception:
                    pass

                # 1) 聚合到单CSV
                if aggregate_csv:
                    import csv as _csv
                    agg_csv_path = save_dir / aggregate_csv
                    need_header = not agg_csv_path.exists()
                    with open(agg_csv_path, 'a', newline='') as fcsv:
                        writer = _csv.writer(fcsv)
                        if need_header:
                            writer.writerow(["episode", "step", "x", "y", "reward"])  # 简洁字段
                        for i, (x, y, r) in enumerate(zip(xs, ys, rewards)):
                            writer.writerow([int(episode_index), i, float(x), float(y), float(r)])

                # 2) 聚合为 JSONL（每行一个 episode）
                if aggregate_jsonl:
                    jsonl_path = save_dir / aggregate_jsonl
                    with open(jsonl_path, 'a') as fjsonl:
                        fjsonl.write(json.dumps(episode_data, ensure_ascii=False) + "\n")
            except Exception:
                pass
            
        # 实时显示
        if display_plot and draw_trajectory:
            try:
                fig.canvas.draw()
                # 使用兼容的 buffer_rgba() 方法（适用于 Agg 后端）
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                # RGBA 格式，需要转换为 RGB
                img = buf.reshape(h, w, 4)[:, :, :3]  # 去掉 alpha 通道
                img_bgr = img[:, :, ::-1]  # RGB 转 BGR
                cv2.imshow(window_name, img_bgr)
                cv2.waitKey(1)
            except Exception as e:
                if verbose > 0:
                    print(f"[WARN] 实时显示失败: {e}")
        
        plt.close(fig)
    except Exception as e:
        # 避免影响训练流程
        print(f"[WARN] 轨迹可视化失败: {e}")


class EnhancedTrainingCallback(BaseCallback):
    """增强的训练回调 - 包含成功率统计、轨迹可视化等功能"""
    
    def __init__(self, 
                 env: ROSbotNavigationEnv,
                 log_interval: int = 100,
                 save_root: Optional[str] = None,
                 show_ui: bool = False,
                 draw_trajectory: bool = False,
                 plot_vmin: float = -0.5,
                 plot_vmax: float = 0.5,
                 step_avg_window: int = 100,
                 display_plot: bool = False,
                 enable_obstacle_curriculum: bool = False,
                 verbose: int = 0,
                 model_save_path: Optional[str] = None):
        super().__init__(verbose)
        self.env = env
        self.model_save_path = model_save_path
        self.interrupted = False
        # 兼容 Monitor 包装，保留对底层环境的引用以用于调试信息/可视化
        try:
            self.base_env = env.env if hasattr(env, 'env') else env
        except Exception:
            self.base_env = env
        self.log_interval = log_interval
        self.enable_obstacle_curriculum = enable_obstacle_curriculum
        self.save_root = save_root
        self.draw_trajectory = draw_trajectory
        self.show_ui = show_ui
        self.plot_vmin = plot_vmin
        self.plot_vmax = plot_vmax
        self.display_plot = display_plot
        
        # 单步奖励滑动平均窗口
        self.step_avg_window = int(max(1, step_avg_window))
        self.step_reward_window = deque(maxlen=self.step_avg_window)
        # 线性加速度与靠墙原始度量的滑动窗口
        self.linear_acc_window = deque(maxlen=self.step_avg_window)
        self.wall_proximity_window = deque(maxlen=self.step_avg_window)
        # 线速度窗口（用于在记录时计算方差，降低每步开销）
        self.linear_vel_window = deque(maxlen=self.step_avg_window)
        # 角速度窗口（用于在记录时计算平均值与方差）
        self.angular_vel_window = deque(maxlen=self.step_avg_window)
        
        # 训练损失（若可用）占位，避免未定义报错
        self.critic_loss_last = None
        self.actor_loss_last = None
        
        # Episode统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []  # 成功标志
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # 轨迹收集（当前episode）
        self.current_episode_positions = []
        self.current_episode_step_rewards = []
        # 记录当前回合的起点和终点（避免绘图时使用下一回合的信息）
        self.current_episode_start_pos = None
        self.current_episode_target_pos = None
        
        # 成功率统计
        self.recent_success_window = deque(maxlen=100)  # 最近100个episode的成功情况
        self.total_episodes = 0
        self.total_successes = 0
        
        # 接近率统计（终点在目标2米范围内）
        self.recent_approach_window = deque(maxlen=100)  # 最近100个episode的接近情况
        self.total_approaches = 0
        
        # 自动保存设置（用于应对手动中断）
        self.autosave_interval = 10000  # 每10000步自动保存一次
        self.last_autosave_step = 0
        
    def _on_step(self) -> bool:
        # 更新环境的全局训练步数（用于渐进式障碍物课程）
        if self.enable_obstacle_curriculum:
            # 优先使用 base_env（绕过 Monitor 等包装器）
            target_env = self.base_env
            if hasattr(target_env, 'update_global_training_step'):
                target_env.update_global_training_step(self.num_timesteps)
            elif hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                # VecEnv 包装的情况
                base_env = self.env.envs[0]
                if hasattr(base_env, 'update_global_training_step'):
                    base_env.update_global_training_step(self.num_timesteps)

        
        # 获取当前步的奖励和信息
        reward = self.locals.get('rewards', [0])[0]
        info = self.locals.get('infos', [{}])[0]
        
        # 累积奖励和步数
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # 更新滑动窗口（内存累积，减少频繁统计开销）
        try:
            self.step_reward_window.append(float(reward))
            la = info.get('linear_acc', None)
            if la is not None and np.isfinite(la):
                self.linear_acc_window.append(float(la))
            wp = info.get('wall_proximity_raw', None)
            if wp is not None and np.isfinite(wp):
                self.wall_proximity_window.append(float(wp))
            lv = info.get('linear_vel', None)
            if lv is not None and np.isfinite(lv):
                self.linear_vel_window.append(float(lv))
            av = info.get('angular_vel', None)
            if av is not None and np.isfinite(av):
                self.angular_vel_window.append(float(av))
        except Exception:
            pass
        
        # 收集位置信息（用于轨迹可视化）
        try:
            env_for_info = getattr(self, 'base_env', self.env)
            if hasattr(env_for_info, '_get_sup_position'):
                pos = env_for_info._get_sup_position()
                if pos is not None and len(pos) >= 2:
                    self.current_episode_positions.append((float(pos[0]), float(pos[1])))
                    # 只有成功获取位置时，才记录对应的奖励，确保长度一致
                    self.current_episode_step_rewards.append(float(reward))
                    
                    # 在回合第一步记录起点和终点（避免绘图时使用下一回合的信息）
                    if len(self.current_episode_positions) == 1:
                        try:
                            if hasattr(env_for_info, 'task_info'):
                                self.current_episode_start_pos = env_for_info.task_info.get('start_pos', None)
                                self.current_episode_target_pos = env_for_info.task_info.get('target_pos', None)
                        except Exception:
                            pass
        except Exception:
            pass
        
        # 检查episode是否结束
        done = self.locals.get('dones', [False])[0]
        if done:
            self.total_episodes += 1
            
            # 判断是否成功
            episode_success = info.get('success', False)
            self.episode_successes.append(episode_success)
            self.recent_success_window.append(episode_success)
            if episode_success:
                self.total_successes += 1
            
            # 判断是否接近目标（终点在目标2米范围内）
            final_distance = info.get('distance_to_target', float('inf'))
            episode_approach = final_distance < 2.0  # 2米阈值
            self.recent_approach_window.append(episode_approach)
            if episode_approach:
                self.total_approaches += 1
            
            # 记录episode指标
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 打印episode信息
            # if self.verbose > 0:
            #     success_str = "✓ 成功" if episode_success else "✗ 失败"
            #     print(f"Episode {self.total_episodes} 完成 [{success_str}] - "
            #           f"步数: {self.num_timesteps}, 奖励: {self.current_episode_reward:.2f}, "
            #           f"长度: {self.current_episode_length}, "
            #           f"成功率(近100): {success_rate_recent*100:.1f}%, "
            #           f"成功率(总): {success_rate_total*100:.1f}%")
            
            # 绘制轨迹图
            try:
                if len(self.current_episode_positions) > 0 and len(self.current_episode_step_rewards) > 0:
                    # 使用记录的当前回合起点和终点，而不是 env.task_info（已更新为下一回合）
                    episode_task_info = {
                        'start_pos': self.current_episode_start_pos,
                        'target_pos': self.current_episode_target_pos,
                        'obstacle_positions': getattr(getattr(self, 'base_env', self.env), '_active_obstacle_positions', None)
                    }
                    plot_episode_positions_and_rewards(
                        positions=self.current_episode_positions,
                        rewards=self.current_episode_step_rewards,
                        env=getattr(self, 'base_env', self.env),
                        episode_index=self.total_episodes,
                        out_dir="debug_plots",
                        save_root=self.save_root,
                        display_plot=self.display_plot,
                        window_name="Episode Trajectory",
                        color_mode="continuous",
                        vmin=self.plot_vmin,
                        vmax=self.plot_vmax,
                        cmap_name="turbo",
                        draw_heading=False,
                        draw_line=False,
                        draw_trajectory=self.draw_trajectory,
                        aggregate_csv="trajectories.csv",
                        aggregate_jsonl="trajectories.jsonl",
                        terminal_global_step=self.num_timesteps,
                        episode_task_info=episode_task_info
                    )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[WARN] 轨迹可视化失败: {e}")
            
            # 重置episode计数器
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_positions = []
            self.current_episode_step_rewards = []
            self.current_episode_start_pos = None
            self.current_episode_target_pos = None

        # 定期记录平均指标（合并写入，降低MLflow写频率）
        if self.num_timesteps % self.log_interval == 0:
            # 计算成功率和接近率
            success_rate_recent = sum(self.recent_success_window) / len(self.recent_success_window) if len(self.recent_success_window) > 0 else 0.0
            approach_rate_recent = sum(self.recent_approach_window) / len(self.recent_approach_window) if len(self.recent_approach_window) > 0 else 0.0
            #success_rate_total = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0
            #approach_rate_total = self.total_approaches / self.total_episodes if self.total_episodes > 0 else 0.0
            avg_step_reward = float(np.mean(self.step_reward_window)) if len(self.step_reward_window) > 0 else 0.0
            avg_linear_acc = float(np.mean(self.linear_acc_window)) if len(self.linear_acc_window) > 0 else 0.0
            avg_wall_proximity = float(np.mean(self.wall_proximity_window)) if len(self.wall_proximity_window) > 0 else 0.0
            # 线速度/角速度统计仅在记录时计算，避免逐步重复计算
            avg_linear_vel = float(np.mean(self.linear_vel_window)) if len(self.linear_vel_window) > 0 else 0.0
            linear_velocity_var = float(np.var(self.linear_vel_window)) if len(self.linear_vel_window) > 1 else 0.0
            avg_angular_vel = float(np.mean(self.angular_vel_window)) if len(self.angular_vel_window) > 0 else 0.0
            angular_velocity_var = float(np.var(self.angular_vel_window)) if len(self.angular_vel_window) > 1 else 0.0
            # 回合统计滑动平均（最近100个回合）
            avg_reward = float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) > 0 else 0.0
            avg_length = float(np.mean(self.episode_lengths[-100:])) if len(self.episode_lengths) > 0 else 0.0

            # 仅在写入时读取一次最新的训练损失，避免每步读取带来的开销
            actor_loss_to_log = 0.0
            critic_loss_to_log = 0.0
            try:
                if hasattr(self, 'model') and hasattr(self.model, 'logger') and self.model.logger is not None:
                    actor_loss_val = self.model.logger.name_to_value.get('train/actor_loss', None)
                    critic_loss_val = self.model.logger.name_to_value.get('train/critic_loss', None)
                    if actor_loss_val is not None:
                        actor_loss_to_log = float(actor_loss_val)
                    if critic_loss_val is not None:
                        critic_loss_to_log = float(critic_loss_val)
            except Exception:
                pass

            # 记录到MLflow（合并写入，减少调用次数）
            if mlflow.active_run():
                mlflow.log_metrics({
                    "critic_loss": critic_loss_to_log,
                    "actor_loss": actor_loss_to_log,
                    "avg_reward_1000": avg_step_reward,
                    "avg_episode_length_100": float(self.current_episode_length),
                    "success_rate_recent_100": float(success_rate_recent),
                    "approach_rate_recent_100": float(approach_rate_recent),
                    "avg_linear_acc_window": avg_linear_acc,
                    "avg_wall_proximity_window": avg_wall_proximity,
                    "avg_linear_vel_window": avg_linear_vel,
                    "linear_velocity_var_window": linear_velocity_var,
                    "avg_angular_vel_window": avg_angular_vel,
                    "angular_velocity_var_window": angular_velocity_var,
                }, step=self.num_timesteps)

            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"步数 {self.num_timesteps} - 统计信息:")
                print(f"  单步平均奖励(窗口{self.step_avg_window}): {avg_step_reward:.2f}")
                print(f"  平均线性加速度(窗口{self.step_avg_window}): {avg_linear_acc:.4f} m/s^2")
                print(f"  平均靠墙原始度量(窗口{self.step_avg_window}): {avg_wall_proximity:.4f}")
                print(f"  线速度均值(窗口{self.step_avg_window}): {avg_linear_vel:.4f} m/s | 方差: {linear_velocity_var:.6f}")
                print(f"  角速度均值(窗口{self.step_avg_window}): {avg_angular_vel:.4f} rad/s | 方差: {angular_velocity_var:.6f}")
                print(f"  总Episodes: {self.total_episodes}")
                if len(self.episode_rewards) > 0:
                    print(f"  平均奖励(最近100): {avg_reward:.2f}")
                    print(f"  平均长度(最近100): {avg_length:.1f}")
                    print(f"  成功率(最近100): {success_rate_recent*100:.1f}%")
                    print(f"  接近率(最近100,<2m): {approach_rate_recent*100:.1f}%")
                    print(f"  成功率(总体): {(self.total_successes/self.total_episodes*100) if self.total_episodes > 0 else 0:.1f}%")
                    print(f"  接近率(总体,<2m): {(self.total_approaches/self.total_episodes*100) if self.total_episodes > 0 else 0:.1f}%")
                print(f"{'='*80}\n")
        
        # 自动保存机制（应对手动中断）
        if self.model_save_path and (self.num_timesteps - self.last_autosave_step) >= self.autosave_interval:
            try:
                autosave_path = str(self.model_save_path).replace('.zip', '_autosave.zip')
                self.model.save(autosave_path)
                self.last_autosave_step = self.num_timesteps
                if self.verbose > 0:
                    print(f"\u2713 自动保存模型到: {autosave_path} (步数: {self.num_timesteps})")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[WARN] 自动保存失败: {e}")
        
        return True


def train_single_cargo_model(
    cargo_type: str,
    total_steps: int,
    model_save_path: str,
    device: str = 'cuda',
    args: Optional[argparse.Namespace] = None
):
    """
    单线程标准训练函数
    
    Args:
        cargo_type: 货物类型 ('normal', 'fragile', 'dangerous')
        total_steps: 总训练步数
        model_save_path: 模型保存路径
        device: 计算设备
        args: 命令行参数对象
    
    Returns:
        model: 训练好的TD3模型
        env: 环境实例
    """
    print(f"\n{'='*60}")
    print(f"开始单线程标准训练")
    print(f"货物类型: {cargo_type}")
    print(f"总步数: {total_steps}")
    print(f"设备: {device}")
    print(f"观测模式: {args.obs_mode if args else 'lidar'}")
    print(f"动作模式: {args.action_mode if args else 'wheels'}")
    print(f"{'='*60}\n")
    
    # 1. 启动Webots实例
    print("启动Webots实例...")
    world_path = args.world_1 if args and hasattr(args, 'world_1') else None
    
    if args.show_ui:
        args.headless = False
        args.no_rendering = False
    else:
        args.headless = True
        args.no_rendering = True
    webots_pid, webots_url = start_webots_instance(
        instance_id=0,  # 单线程训练使用实例ID 0
        world_path=world_path,
        headless=args.headless if args else True,
        fast_mode=args.fast_mode if args else True,
        no_rendering=args.no_rendering if args else True,
        batch=args.batch if args else True,
        minimize=args.minimize if args else True,
        stdout=True,
        stderr=True
    )
    print(f"Webots实例已启动 - PID: {webots_pid}, URL: {webots_url}")
    
    # 2. 构建环境
    print("构建导航环境...")
    
    env = ROSbotNavigationEnv(
        extern_controller_url=webots_url,
        cargo_type=cargo_type,
        show_map=args.show_map if args else False,
        control_period_ms=args.control_period_ms if args else 200,
        max_episode_steps=500,
        seed=args.seed if args else 0,
        obs_mode=args.obs_mode if args else 'lidar',
        action_mode=args.action_mode if args else 'wheels',
        macro_action_steps=args.macro_action_steps if args else 1,
        enable_speed_smoothing=args.enable_speed_smoothing if args else False,
        training_mode=args.training_mode if args else 'vertical_curriculum',
        # Flags for obstacle logic
        enable_obstacle_curriculum=args.enable_obstacle_curriculum if args and hasattr(args, 'enable_obstacle_curriculum') else True,
        enable_obstacle_randomization=args.enable_obstacle_curriculum if args and hasattr(args, 'enable_obstacle_curriculum') else True,  # backward compat
        use_predefined_positions=args.use_predefined_positions if args and hasattr(args, 'use_predefined_positions') else False,
        fixed_obstacle_count=args.fixed_obstacle_count if args and hasattr(args, 'fixed_obstacle_count') else 5,
        lock_obstacles_per_stage=args.lock_obstacles_per_stage if args and hasattr(args, 'lock_obstacles_per_stage') else False,
        obstacle_curriculum_steps=args.obstacle_curriculum_steps if args and hasattr(args, 'obstacle_curriculum_steps') else None,
        obstacle_curriculum_counts=args.obstacle_curriculum_counts if args and hasattr(args, 'obstacle_curriculum_counts') else None,
        debug=args.debug
    )
    
    # 设置args属性，供奖励函数使用
    env.args = args
    
    # 注册进程清理
    attach_process_cleanup_to_env(env, webots_pid)
    
    # 3. 包装环境用于监控
    model_dir = Path(model_save_path).parent
    env = Monitor(env, str(model_dir / "monitor_logs"))
    
    print("环境创建完成")
    
    # 4. 根据观测模式选择策略
    obs_mode = args.obs_mode if args else 'lidar'
    if obs_mode == 'local_map':
        policy_type = "MultiInputPolicy"
        print("使用 MultiInputPolicy（字典观测空间 - local_map）")
    else:
        policy_type = "MlpPolicy"
        print("使用 MlpPolicy（向量观测空间 - lidar）")
    
    # 5. 设置动作噪声（用于探索）
    n_actions = env.action_space.shape[0]
    action_noise_sigma = args.exploration_noise_init if args and hasattr(args, 'exploration_noise_init') else 0.1
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_sigma * np.ones(n_actions)
    )
    print(f"动作噪声: sigma={action_noise_sigma}")
    
    # 6. 创建或加载TD3模型（支持在已有模型基础上继续训练）
    print("\n创建/加载TD3模型...")
    
    # TD3超参数
    learning_rate = args.learning_rate if args else 3e-4
    buffer_size = args.buffer_size if args else 50000
    learning_starts = args.learning_starts if args else 2000
    batch_size = args.batch_size if args else 256
    gamma = args.gamma if args else 0.99
    tau = args.tau if args else 0.005
    train_freq = args.train_freq if args else 1
    gradient_steps = args.gradient_steps if args else 1
    policy_delay = args.policy_delay if args else 2
    
    # TD3目标噪声参数
    target_policy_noise = args.target_policy_noise_init if args and hasattr(args, 'target_policy_noise_init') else 0.05
    target_noise_clip = args.target_noise_clip_init if args and hasattr(args, 'target_noise_clip_init') else 0.2
    
    print(f"  学习率: {learning_rate}")
    print(f"  回放缓冲区大小: {buffer_size}")
    print(f"  学习开始步数: {learning_starts}")
    print(f"  批大小: {batch_size}")
    print(f"  折扣因子 gamma: {gamma}")
    print(f"  软更新系数 tau: {tau}")
    print(f"  目标策略噪声: {target_policy_noise}")
    print(f"  目标噪声裁剪: {target_noise_clip}")
    
    resume_training = False
    pretrained_path = ''
    try:
        pretrained_path = getattr(args, 'pretrained_model_path', '') if args is not None else ''
    except Exception:
        pretrained_path = ''
    if pretrained_path and Path(pretrained_path).exists():
        # 从已有模型加载，继续训练
        print(f"检测到预训练模型，加载并继续训练: {pretrained_path}")
        try:
            model = TD3.load(pretrained_path, env=env, device=device)
            # 确保继续训练时仍然有探索噪声
            model.action_noise = action_noise
            # 设置logger（使TensorBoard/控制台输出正常工作）
            try:
                tb_root = args.tensorboard_log if args and hasattr(args, 'tensorboard_log') else None
                if tb_root:
                    model.set_logger(sb3_configure(tb_root, ["stdout", "tensorboard"]))
            except Exception:
                pass
            resume_training = True
            print("预训练模型加载完成，将在其基础上继续训练")
        except Exception as e:
            print(f"[WARN] 加载预训练模型失败，改为新建模型: {e}")
            resume_training = False
    
    if not resume_training:
        model = TD3(
            policy=policy_type,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            device=device,
            verbose=args.verbose if args else 1,
            tensorboard_log=args.tensorboard_log if args and hasattr(args, 'tensorboard_log') else None
        )
    
    print("模型就绪（创建或加载）")
    
    # 7. 设置MLflow（如果需要）
    if args and hasattr(args, 'experiment_name') and args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=f"single_{args.leaner_name}")
        
        # 设置运行描述
        if args and hasattr(args, 'remark') and args.remark:
            mlflow.set_tag("mlflow.note.content", args.remark)
        
        # 记录参数
        # 记录所有args参数到MLflow
        try:
            if args:
                # 获取所有参数的字典
                all_params = vars(args).copy()
                # 添加额外的运行时参数
                all_params.update({
                    'policy_type': policy_type,
                    'model_save_path': str(model_save_path),
                })
                # 过滤掉None值和过长的字符串（MLflow有长度限制）
                filtered_params = {}
                for key, value in all_params.items():
                    if value is not None:
                        # 转换为字符串并限制长度
                        str_value = str(value)
                        if len(str_value) > 250:
                            str_value = str_value[:247] + '...'
                        filtered_params[key] = str_value
                mlflow.log_params(filtered_params)
        except Exception as e:
            print(f"[WARN] 记录args参数到MLflow失败: {e}")
            # 兜底：只记录核心参数
            mlflow.log_params({
                'cargo_type': cargo_type,
                'total_steps': total_steps,
                'learning_rate': learning_rate,
                'buffer_size': buffer_size,
                'batch_size': batch_size,
                'gamma': gamma,
                'obs_mode': obs_mode,
                'action_mode': args.action_mode if args else 'wheels',
                'policy_type': policy_type,
                'device': device,
                'model_save_path': model_save_path,
                'step_avg_window': (args.step_avg_window if args and hasattr(args, 'step_avg_window') else 100)
            })
        print("MLflow运行已启动")
        
        # 记录代码到MLflow（便于实验追溯）
        try:
            # 1. 记录当前运行的训练脚本
            current_script = Path(__file__)
            if current_script.exists():
                mlflow.log_artifact(str(current_script), artifact_path='code')
                print(f"✓ 已记录训练脚本: {current_script.name}")
            
            # 2. 记录整个src目录
            src_dir = Path(__file__).parent / 'src'
            if src_dir.exists() and src_dir.is_dir():
                mlflow.log_artifacts(str(src_dir), artifact_path='code/src')
                print(f"✓ 已记录整个src目录: {src_dir}")
            else:
                print(f"[WARN] src目录不存在: {src_dir}")
                
        except Exception as e:
            print(f"[WARN] 记录代码文件失败: {e}")
    
    # 8. 创建增强的训练回调（包含成功率统计和轨迹可视化）
    callbacks = []
    enhanced_callback = EnhancedTrainingCallback(
        env=env,
        log_interval=args.mlflow_log_interval if args and hasattr(args, 'mlflow_log_interval') else 500,
        save_root=str(model_dir),
        show_ui=args.show_ui if args and hasattr(args, 'show_ui') else False,
        plot_vmin=args.plot_vmin if args and hasattr(args, 'plot_vmin') else -150,
        plot_vmax=args.plot_vmax if args and hasattr(args, 'plot_vmax') else 150,
        step_avg_window=(args.step_avg_window if args and hasattr(args, 'step_avg_window') else 100),
        verbose=args.verbose if args and hasattr(args, 'verbose') else 1,
        draw_trajectory=args.draw_trajectory if args and hasattr(args, 'draw_trajectory') else False,
        display_plot=args.display_plot if args and hasattr(args, 'display_plot') else False,
        enable_obstacle_curriculum=args.enable_obstacle_curriculum if args and hasattr(args, 'enable_obstacle_curriculum') else False,
        model_save_path=model_save_path  # 传递模型保存路径用于自动保存
    )
    callbacks.append(enhanced_callback)
    
    # 9. 开始训练
    print(f"\n{'='*60}")
    print(f"开始训练 - 目标步数: {total_steps}")
    print(f"{'='*60}\n")
    
    try:
        # 继续训练时通常不重置时间步计数，使日志/学习率调度等连续
        reset_counter = getattr(args, 'reset_num_timesteps', False) if args is not None else False
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks if callbacks else None,
            log_interval=args.log_interval if args and hasattr(args, 'log_interval') else 100,
            reset_num_timesteps=bool(reset_counter) if resume_training else True,
            progress_bar=True
        )
        
        print(f"\n{'='*60}")
        print("训练完成！")
        print(f"{'='*60}\n")
        
        # 10. 保存模型
        print(f"保存模型到: {model_save_path}")
        model.save(model_save_path)
        print("模型保存成功")
        
        # 11. 结束MLflow运行
        if mlflow.active_run():
            mlflow.end_run()
            print("MLflow运行已结束")
        
        return model, env
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存当前模型
        interrupted_path = str(model_save_path).replace('.zip', '_interrupted.zip')
        model.save(interrupted_path)
        print(f"中断模型已保存到: {interrupted_path}")
        
        if mlflow.active_run():
            mlflow.end_run(status='KILLED')
        
        raise
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        if mlflow.active_run():
            mlflow.end_run(status='FAILED')
        raise


def main():
    """主函数 - 用于独立运行此脚本"""
    parser = argparse.ArgumentParser(description='ROSbot单线程标准训练脚本')
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 常用训练参数
    parser.add_argument('--tensorboard_log', type=str, default='./logs', help='TensorBoard日志目录')
    parser.add_argument('--experiment_name', type=str, default='Single-Vertical-R4.6.10', help='MLflow实验名称')
    parser.add_argument('--model_dir', type=str, 
                        default=f'/root/workspace/RL_car2/rosbot_navigation/results/Single-Vertical-R4.6.10', 
                        help='模型保存目录')
    
    parser.add_argument('--cargo_type', type=str, default='dangerous', choices=['normal', 'fragile', 'dangerous'], help='货物类型')
    cargo_type = parser.parse_known_args()[0].cargo_type
    parser.add_argument('--leaner_name', type=str, default=f'R4.6.10_W2E4_T5.2_{cargo_type}_{now}', help='leaner名称')
    parser.add_argument('--remark', type=str, default='T5.2.0 全任务微调到dangerous，提高各种限制系数', help='remark')
    
    parser.add_argument('--total_steps', type=int, default=80000, help='总训练步数')

    parser.add_argument('--pretrained_model_path', type=str, 
                        default='/root/workspace/RL_car2/rosbot_navigation/results/Single-Vertical-R4.6.10/R4.6.10_W2E4_T6.1_normal_20251009_221340/td3_R4.6.10_W2E4_T6.1_normal_20251009_221340_normal_400000_autosave_0.94.zip', 
                        help='预训练模型路径(.zip)，若提供则在其基础上继续训练')
    
    parser.add_argument('--world_1', type=str, 
                        default='/root/workspace/RL_car2/warehouse/worlds/warehouse2_end4.wbt', 
                        help='Webots world文件路径')
    # 障碍物参数
    parser.add_argument('--enable_obstacle_curriculum', type=bool, default=False, help='是否启用渐进式障碍物数量课程学习')
    parser.add_argument('--use_predefined_positions', type=bool, default=False, help='True: 从当前world文件的WoodenBox初始位置集合中选择；False: 在范围内随机生成坐标')
    parser.add_argument('--fixed_obstacle_count', type=int, default=-1, help='障碍物固定数量: >=0时生效并覆盖课程学习；-1时不生效（遵循课程或默认）')
    parser.add_argument('--lock_obstacles_per_stage', type=bool, default=True, help='是否启用阶段锁定模式')

    # 课程学习参数
    parser.add_argument('--obstacle_curriculum_steps', type=int, nargs='+', 
                        default=[0,10000,18000,26000,34000,43000,53000,64000,76000,89000,110000,133000,156000], 
                        help='渐进式障碍物课程学习的步数阈值列表')
    parser.add_argument('--obstacle_curriculum_counts', type=int, nargs='+', 
                        default=[2,3,4,5,6,7,8,9,10,11,12,13], 
                        help='渐进式障碍物课程学习对应的障碍物数量列表')

    parser.add_argument('--training_mode', type=str, default='vertical_curriculum', help='训练模式')
    # 绘图参数
    parser.add_argument('--plot_vmin', type=float, default=-100, help='轨迹图奖励最小值')
    parser.add_argument('--plot_vmax', type=float, default=100, help='轨迹图奖励最大值')

    # 基本参数
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    # 环境参数
    parser.add_argument('--obs_mode', type=str, default='lidar', choices=['local_map', 'lidar'], help='观测模式')
    parser.add_argument('--action_mode', type=str, default='wheels', choices=['wheels', 'twist'], help='动作模式')
    parser.add_argument('--macro_action_steps', type=int, default=1, help='宏动作步数')
    parser.add_argument('--enable_speed_smoothing', type=bool, default=False, help='是否启用速度平滑')
    parser.add_argument('--control_period_ms', type=int, default=200, help='控制周期(ms)')
    
    # Webots参数
    
    parser.add_argument('--headless', type=bool, default=True, help='无头模式')
    parser.add_argument('--fast_mode', type=bool, default=True, help='快速模式')
    parser.add_argument('--no_rendering', type=bool, default=True, help='禁用渲染')
    parser.add_argument('--batch', type=bool, default=True, help='批处理模式')
    parser.add_argument('--minimize', type=bool, default=True, help='最小化窗口')
    
    # UI参数
    parser.add_argument('--show_ui', type=bool, default=False, help='显示UI')
    parser.add_argument('--show_map', type=bool, default=False, help='显示地图')
    parser.add_argument('--display_plot', type=bool, default=False, help='显示轨迹图')
    parser.add_argument('--debug', type=bool, default=False, help='调试模式')
    
    # TD3参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--buffer_size', type=int, default=50000, help='回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=2000, help='开始学习步数')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
    parser.add_argument('--train_freq', type=int, default=1, help='训练频率')
    parser.add_argument('--gradient_steps', type=int, default=1, help='梯度步数')
    parser.add_argument('--policy_delay', type=int, default=2, help='策略延迟')
    parser.add_argument('--exploration_noise_init', type=float, default=0.1, help='探索噪声')
    parser.add_argument('--target_policy_noise_init', type=float, default=0.05, help='目标策略噪声')
    parser.add_argument('--target_noise_clip_init', type=float, default=0.2, help='目标噪声裁剪')
    
    # 日志参数
    parser.add_argument('--verbose', type=int, default=1, help='详细程度')
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--mlflow_log_interval', type=int, default=500, help='MLflow记录间隔')
    parser.add_argument('--step_avg_window', type=int, default=1000, help='单步奖励滑动平均窗口大小（步数）')
    parser.add_argument('--draw_trajectory', type=bool, default=True, help='绘制轨迹')
    
    # 继续训练参数
    parser.add_argument('--reset_num_timesteps', type=bool, default=False, help='继续训练时是否重置时间步计数到0（默认False表示连续计数）')
    
    # 课程学习参数（兼容性）
    parser.add_argument('--curriculum_stage', type=str, default='end', help='课程阶段')
    
    # 奖励系数参数
    parser.add_argument('--delta_distance_k', type=float, default=30.0, help='距离变化奖励系数，以最大速度1.18m/s计，基础最大值约为0.7，线性')
    parser.add_argument('--movement_reward_k', type=float, default=0.5, help='移动奖励系数，基础最大值为10，线性')

    parser.add_argument('--liner_distance_reward', type=int, default=0,choices=[0, 1, 2], help='0:线性 1：负二次型 2：反比例')
    parser.add_argument('--distance_k', type=float, default=0.7, help='基础距离奖励系数，基础最大值为50')

    parser.add_argument('--time_k', type=float, default=0.4, help='时间惩罚系数，线性')   
    parser.add_argument('--wall_proximity_penalty_k', type=float, default=3, help='墙壁接近惩罚系数，每条线最大0.8，一共二十线，基础最大值14，线性')

    parser.add_argument('--angle_reward_k', type=float, default=5, help='角度奖励系数，基础最大值为10，二次型')
    parser.add_argument('--angle_change_k', type=float, default=15.0, help='角度变化奖励,以单步最大角度变化1计，基础最大值1，线性')
    parser.add_argument('--directional_movement_k', type=float, default=50.0, help='方向性移动奖励系数，鼓励朝目标方向移动而非随机移动')
    parser.add_argument('--early_spin_penalty_k', type=float, default=1.0, help='早期原地打转惩罚系数，负二次型')
    parser.add_argument('--front_clear_k', type=float, default=1.0, help='前方有路奖励系数，基础最大值约为7，七条线求和，线性')
    
    # 停用奖励
    parser.add_argument('--stop_bonus_k', type=float, default=0, help='停车奖励系数，基础最大值为10')   
    parser.add_argument('--approach_reward_k', type=float, default=0, help='接近奖励系数，基础最大值为10')  
    parser.add_argument('--slow_down_reward_k', type=float, default=0, help='接近目标减速奖励系数，基础最大值为10')
    args = parser.parse_args()
    # 创建模型保存路径
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_dir is not None:
        models_dir = Path(args.model_dir+"/"+args.leaner_name)
    else:   
        models_dir = Path(f"./models/single_train_{now}")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"td3_{args.leaner_name}_{args.cargo_type}_{args.total_steps}.zip"
    
    print(f"\n开始单线程标准训练")
    print(f"模型保存路径: {model_path}")
    
    try:
        model = None
        env = None
        model, env = train_single_cargo_model(
            cargo_type=args.cargo_type,
            total_steps=args.total_steps,
            model_save_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            args=args
        )
        
        print("\n训练成功完成！")
        print(f"模型已保存到: {model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        raise
    finally:
        # 清理资源
        print("\n清理资源...")
        if env is not None:
            try:
                print("正在关闭环境...")
                env.close()
                print("环境已关闭")
            except Exception as e:
                print(f"关闭环境时出错: {e}")
        
        # 等待一小段时间确保所有进程完全结束
        import time
        time.sleep(2)
        print("清理完成")


if __name__ == "__main__":
    # 添加项目路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main()
