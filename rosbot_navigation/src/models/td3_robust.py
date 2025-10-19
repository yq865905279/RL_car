"""
改进的TD3模型 - 适配ROSbot导航任务
包含鲁棒性增强和AMCL不确定性建模
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple, Union

from stable_baselines3 import TD3 
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# MLflow导入
import mlflow
import mlflow.pytorch
# 直接使用TD3的标准分布，无需额外导入


class RobustTD3Policy(BasePolicy):
    """鲁棒的TD3策略网络"""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: type = nn.ReLU,
        features_extractor_class: type = BaseFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        normalize_images: bool = True,
        optimizer_class: type = optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
            **kwargs
        )
        
        # 网络架构
        self.net_arch = net_arch or [512, 256, 128]
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.share_features_extractor = share_features_extractor
        
        # 特征提取器
        self.features_extractor = features_extractor_class(
            self.observation_space, **features_extractor_kwargs
        )
        
        # 鲁棒性增强层
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.features_extractor.features_dim)
        
        # 策略网络
        self.actor = self._build_actor()
        
        # Q网络（双Q网络）
        self.critics = nn.ModuleList([
            self._build_critic() for _ in range(n_critics)
        ])
        
        # 设置优化器
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_schedule = lr_schedule
        
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(), 
            lr=self.lr_schedule(1),
            **self.optimizer_kwargs
        )
        
        self.critic_optimizer = self.optimizer_class(
            self.critics.parameters(),
            lr=self.lr_schedule(1),
            **self.optimizer_kwargs
        )
        
    def _build_actor(self) -> nn.Module:
        """构建策略网络"""
        layers = []
        
        layers.append(nn.Linear(self.features_extractor.features_dim, self.net_arch[0]))
        layers.append(self.activation_fn())
        layers.append(nn.Dropout(0.1))
        
        for i in range(len(self.net_arch) - 1):
            layers.append(nn.Linear(self.net_arch[i], self.net_arch[i+1]))
            layers.append(self.activation_fn())
            layers.append(nn.LayerNorm(self.net_arch[i+1]))
            layers.append(nn.Dropout(0.1))
        
        # 输出层 - 确定性和噪声分支
        layers.append(nn.Linear(self.net_arch[-1], self.net_arch[-1]))
        
        # 策略头和噪声头 - 增强长期规划能力
        # 使用更大的中间层来处理多个动作的输出
        self.policy_head = nn.Sequential(
            nn.Linear(self.net_arch[-1], self.net_arch[-1] * 2),
            self.activation_fn(),
            nn.LayerNorm(self.net_arch[-1] * 2),
            nn.Linear(self.net_arch[-1] * 2, self.action_space.shape[0])
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(self.net_arch[-1], self.net_arch[-1] * 2),
            self.activation_fn(),
            nn.LayerNorm(self.net_arch[-1] * 2),
            nn.Linear(self.net_arch[-1] * 2, self.action_space.shape[0])
        )
        
        return nn.Sequential(*layers)
    
    def _build_critic(self) -> nn.Module:
        """构建Q网络"""
        layers = []
        
        # 输入：观测 + 动作
        input_dim = self.features_extractor.features_dim + self.action_space.shape[0]
        
        layers.append(nn.Linear(input_dim, self.net_arch[0]))
        layers.append(self.activation_fn())
        layers.append(nn.Dropout(0.1))
        
        for i in range(len(self.net_arch) - 1):
            layers.append(nn.Linear(self.net_arch[i], self.net_arch[i+1]))
            layers.append(self.activation_fn())
            layers.append(nn.LayerNorm(self.net_arch[i+1]))
            layers.append(nn.Dropout(0.1))
        
        # Q值输出
        layers.append(nn.Linear(self.net_arch[-1], 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 增强长期规划能力
        
        输出10维动作空间，包含5个连续的动作，每个动作2维[线速度, 角速度]
        """
        # 特征提取
        features = self.extract_features(obs)
        
        # 策略网络
        core_features = self.actor(features)
        
        # 主策略输出
        mean_actions = self.policy_head(core_features)
        
        # 将动作重塑为(batch_size, 5, 2)以便于处理时序依赖关系
        batch_size = mean_actions.shape[0]
        mean_actions_reshaped = mean_actions.view(batch_size, 5, 2)
        
        # 应用时序平滑化，确保相邻动作之间的连续性
        # 对于第一个动作保持原样，后续动作与前一个动作保持一定的相关性
        for i in range(1, 5):
            # 应用平滑因子，使当前动作部分依赖于前一个动作
            smoothing_factor = 0.3  # 可调整的平滑因子
            mean_actions_reshaped[:, i] = mean_actions_reshaped[:, i] * (1 - smoothing_factor) + \
                                        mean_actions_reshaped[:, i-1] * smoothing_factor
        
        # 重新展平为10维动作
        mean_actions = mean_actions_reshaped.reshape(batch_size, -1)
        
        if deterministic:
            actions = torch.tanh(mean_actions)
        else:
            # 噪声输出
            noise_std = torch.sigmoid(self.noise_head(core_features))
            
            # 将噪声重塑为(batch_size, 5, 2)以应用不同的噪声策略
            noise_std_reshaped = noise_std.view(batch_size, 5, 2)
            
            # 远期动作的噪声逐渐增大，表示长期不确定性增加
            for i in range(1, 5):
                # 每个时间步噪声系数增加
                noise_factor = 1.0 + i * 0.15  # 每个时间步增加15%的噪声
                noise_std_reshaped[:, i] = noise_std_reshaped[:, i] * noise_factor
            
            # 重新展平
            noise_std = noise_std_reshaped.reshape(batch_size, -1)
            
            # 应用噪声
            noise = torch.randn_like(mean_actions) * noise_std * 0.1
            actions = torch.tanh(mean_actions + noise)
        
        return actions, mean_actions
    
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """特征提取，包含不确定性处理"""
        base_features = self.features_extractor(obs)
        
        # 如果输入包含不确定性信息，进行处理
        if obs.shape[-1] > 40:  # 42维状态
            # 提取不确定性相关的特征
            uncertainty_features = obs[:, -4:]  # 后面几维可能包含不确定性
            uncertainty_weight = torch.sigmoid(uncertainty_features).mean(dim=-1, keepdim=True)
            
            # 根据不确定性调整特征权重
            base_features = base_features * (1 - uncertainty_weight * 0.2)
        
        # 应用归一化和dropout
        base_features = self.layer_norm(base_features)
        base_features = self.dropout(base_features)
        
        return base_features
    
    def q_value_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        """Q值网络前向传播"""
        features = self.extract_features(obs)
        
        # 连接观测和动作
        q_input = torch.cat([features, actions], dim=1)
        
        # 计算多个Q值（双Q网络）
        q_values = []
        for critic in self.critics:
            q_value = critic(q_input)
            q_values.append(q_value)
        
        return q_values


class RosbotFeaturesExtractor(BaseFeaturesExtractor):
    """ROSbot专用的特征提取器"""
    
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # 特征提取网络
        # 注意：feature_fusion 输出为 128 维，因此此处改为从 128 输入开始
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # 特殊特征处理层（缩小规模）
        self.lidar_processor = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.amcl_processor = nn.Sequential(
            nn.Linear(12, 32),  # AMCL特征
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.navigation_processor = nn.Sequential(
            nn.Linear(10, 32),   # 导航信息
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 特征融合层（缩小规模）
        self.feature_fusion = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """特征提取前向传播"""
        batch_size = observations.shape[0]
        
        # 提取各部分特征
        #lidar_features = self.lidar_processor(observations[:, 0:20])
        #pose_features = self.amcl_processor(observations[:, 20:32])
        #nav_features = self.navigation_processor(observations[:, 32:42])
        
        # 其他特征直接处理
        #other_features = observations[:, 20:42]  # 其他状态信息
        
        # 特征融合
        # combined_features = torch.cat([
        #     lidar_features,    # 16维
        #     pose_features,     # 16维 
        #     nav_features      # 16维
        #     # other_features     
        # ], dim=1)  # 总共48维
        # fused_features = self.feature_fusion(combined_features)
        # 不进行特征提取，直接将42维观测输入融合层
        fused_features = self.feature_fusion(observations)
        
        # 最终特征提取
        final_features = self.net(fused_features)
        
        return final_features


class AdaptiveRewardModel(nn.Module):
    """自适应奖励模型 - 根据货物类型调整"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3种货物类型的奖励分量
        )
    
    def forward(self, state: torch.Tensor, cargo_type: str) -> torch.Tensor:
        """计算自适应奖励分量"""
        base_rewards = self.net(state)
        
        # 根据货物类型选择奖励分量
        if cargo_type == 'fragile':
            return base_rewards[:, 0:1]
        elif cargo_type == 'dangerous':
            return base_rewards[:, 1:2]
        else:
            return base_rewards[:, 2:3]


class ImprovedTD3(TD3):
    """改进的TD3算法 - 适配ROSbot导航"""
    
    def __init__(
        self,
        policy: Union[str, type[BasePolicy]] = RobustTD3Policy,
        env: Optional[Any] = None,
        learning_rate: Union[float, Any] = 5e-4,
        buffer_size: int = 500000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.98,
        train_freq: Union[int, Any] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[Any] = None,
        replay_buffer_class = None,
        replay_buffer_kwargs: Optional[Dict] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.3,
        target_noise_clip: float = 0.5,
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs
    ):
        # 初始化日志记录器
        self._default_logger = configure(None, ["stdout"])
        # 改进的默认参数
        policy_kwargs = policy_kwargs or {}
        policy_kwargs.update({
            'net_arch': [512, 256, 128],
            'features_extractor_class': RosbotFeaturesExtractor,
            'features_extractor_kwargs': {'features_dim': 512}
        })
        
        super().__init__(
            policy=policy,
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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            **kwargs
        )
    
    def get_model_architecture_info(self) -> Dict[str, Any]:
        """获取模型架构的详细信息，用于记录到MLflow"""
        # 获取策略网络信息
        if not hasattr(self, "policy") or self.policy is None:
            return {"error": "Policy not initialized"}
        
        # 基本架构信息
        architecture_info = {
            "model_type": "ImprovedTD3",
            "policy_type": self.policy.__class__.__name__,
            "policy_kwargs": self.policy_kwargs,
        }
        
        # 如果策略已初始化，获取更详细的网络结构
        if hasattr(self.policy, "actor") and self.policy.actor is not None:
            # 获取策略网络结构
            actor_layers = []
            for name, module in self.policy.actor.named_children():
                actor_layers.append(f"{name}: {module.__class__.__name__}")
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    actor_layers[-1] += f" ({module.in_features} -> {module.out_features})"
            
            # 获取策略头结构
            policy_head_layers = []
            for name, module in self.policy.policy_head.named_children():
                policy_head_layers.append(f"{name}: {module.__class__.__name__}")
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    policy_head_layers[-1] += f" ({module.in_features} -> {module.out_features})"
            
            # 获取噪声头结构
            noise_head_layers = []
            for name, module in self.policy.noise_head.named_children():
                noise_head_layers.append(f"{name}: {module.__class__.__name__}")
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    noise_head_layers[-1] += f" ({module.in_features} -> {module.out_features})"
            
            # 获取特征提取器结构
            features_extractor_layers = []
            if hasattr(self.policy, "features_extractor") and self.policy.features_extractor is not None:
                for name, module in self.policy.features_extractor.named_children():
                    features_extractor_layers.append(f"{name}: {module.__class__.__name__}")
                    # 如果是Sequential，进一步获取其子模块
                    if isinstance(module, nn.Sequential):
                        for i, submodule in enumerate(module):
                            features_extractor_layers.append(f"  {i}: {submodule.__class__.__name__}")
                            if hasattr(submodule, "in_features") and hasattr(submodule, "out_features"):
                                features_extractor_layers[-1] += f" ({submodule.in_features} -> {submodule.out_features})"
            
            # 获取评论家网络结构
            critic_layers = []
            if hasattr(self.policy, "critics") and len(self.policy.critics) > 0:
                for i, critic in enumerate(self.policy.critics):
                    critic_layers.append(f"Critic {i}:")
                    for j, module in enumerate(critic):
                        critic_layers.append(f"  {j}: {module.__class__.__name__}")
                        if hasattr(module, "in_features") and hasattr(module, "out_features"):
                            critic_layers[-1] += f" ({module.in_features} -> {module.out_features})"
            
            # 计算参数总量
            total_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            
            # 更新架构信息
            architecture_info.update({
                "actor_layers": actor_layers,
                "policy_head_layers": policy_head_layers,
                "noise_head_layers": noise_head_layers,
                "features_extractor_layers": features_extractor_layers,
                "critic_layers": critic_layers,
                "total_trainable_parameters": total_params,
                "observation_space": str(self.observation_space),
                "action_space": str(self.action_space),
                "learning_rate": self.learning_rate,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "tau": self.tau,
                "gamma": self.gamma,
                "policy_delay": self.policy_delay,
                "target_policy_noise": self.target_policy_noise,
                "target_noise_clip": self.target_noise_clip
            })
        
        return architecture_info
    
    def log_model_architecture_to_mlflow(self):
        """将模型架构记录到MLflow"""
        if not mlflow.active_run():
            print("没有活跃的MLflow运行，无法记录模型架构")
            return
        
        try:
            # 获取模型架构信息
            architecture_info = self.get_model_architecture_info()
            
            # 将架构信息转换为字符串格式，便于MLflow记录
            architecture_str = "\n".join([
                "# 模型架构详情",
                f"## 基本信息",
                f"- 模型类型: {architecture_info['model_type']}",
                f"- 策略类型: {architecture_info['policy_type']}",
                f"- 可训练参数总量: {architecture_info.get('total_trainable_parameters', 'N/A')}",
                f"- 观测空间: {architecture_info.get('observation_space', 'N/A')}",
                f"- 动作空间: {architecture_info.get('action_space', 'N/A')}",
                
                f"\n## 超参数",
                f"- 学习率: {architecture_info.get('learning_rate', 'N/A')}",
                f"- 缓冲区大小: {architecture_info.get('buffer_size', 'N/A')}",
                f"- 批次大小: {architecture_info.get('batch_size', 'N/A')}",
                f"- Tau: {architecture_info.get('tau', 'N/A')}",
                f"- Gamma: {architecture_info.get('gamma', 'N/A')}",
                f"- 策略延迟: {architecture_info.get('policy_delay', 'N/A')}",
                f"- 目标策略噪声: {architecture_info.get('target_policy_noise', 'N/A')}",
                f"- 目标噪声裁剪: {architecture_info.get('target_noise_clip', 'N/A')}",
                
                f"\n## 特征提取器",
                *[f"- {layer}" for layer in architecture_info.get('features_extractor_layers', ['N/A'])],
                
                f"\n## Actor网络",
                *[f"- {layer}" for layer in architecture_info.get('actor_layers', ['N/A'])],
                
                f"\n### 策略头",
                *[f"- {layer}" for layer in architecture_info.get('policy_head_layers', ['N/A'])],
                
                f"\n### 噪声头",
                *[f"- {layer}" for layer in architecture_info.get('noise_head_layers', ['N/A'])],
                
                f"\n## Critic网络",
                *[f"- {layer}" for layer in architecture_info.get('critic_layers', ['N/A'])],
            ])
            
            # 记录架构信息到MLflow
            mlflow.log_text(architecture_str, "model_architecture.md")
            
            # 记录策略参数
            mlflow.log_dict(architecture_info, "model_architecture.json")
            
            # 记录PyTorch模型结构图 (可选，需要graphviz支持)
            try:
                # 尝试使用torchviz记录模型结构图
                import torch
                from torchviz import make_dot
                
                # 创建一个示例输入
                dummy_input = torch.zeros((1, *self.observation_space.shape), 
                                         dtype=torch.float32, 
                                         device=self.device)
                
                # 获取模型输出
                with torch.no_grad():
                    actions, _ = self.policy(dummy_input)
                
                # 创建计算图
                dot = make_dot(actions, params=dict(self.policy.named_parameters()))
                
                # 保存为临时文件
                import tempfile
                import os
                
                with tempfile.TemporaryDirectory() as tmpdirname:
                    dot_path = os.path.join(tmpdirname, "model_graph")
                    dot.render(dot_path, format="png")
                    mlflow.log_artifact(f"{dot_path}.png", "model_architecture")
            except ImportError:
                print("torchviz未安装，跳过模型结构图记录")
            except Exception as e:
                print(f"记录模型结构图失败: {e}")
            
            print("✅ 模型架构已成功记录到MLflow")
            
        except Exception as e:
            print(f"记录模型架构到MLflow失败: {e}")
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
        use_mlflow: bool = True,
        mlflow_run_name: str = None,
        mlflow_experiment_name: str = None
    ):
        """训练学习 - 增强版本"""
        print(f"开始ROSbot导航训练 - TD3算法")
        print(f"总步数: {total_timesteps}")
        print(f"状态空间: 42维 (LiDAR+AMCL+导航信息)")
        print(f"动作空间: 2维连续 [线速度, 角速度]")
        print(f"AMCL粒子数: 800")
        print(f"✅ 已启用梯度裁剪 (max_norm=0.5)")
        
        # 确保logger可用
        if self._logger is None:
            self._logger = configure(None, ["stdout"])
        
        # 如果启用MLflow且没有活跃的运行，则创建一个新的运行
        if use_mlflow and mlflow.active_run() is None:
            try:
                # 设置实验名称（如果提供）
                if mlflow_experiment_name:
                    mlflow.set_experiment(mlflow_experiment_name)
                
                # 开始一个新的MLflow运行
                run_name = mlflow_run_name or f"td3_{tb_log_name}_{total_timesteps}"
                mlflow.start_run(run_name=run_name)
                
                # 记录模型超参数
                mlflow.log_params({
                    "learning_rate": self.learning_rate,
                    "buffer_size": self.buffer_size,
                    "learning_starts": self.learning_starts,
                    "batch_size": self.batch_size,
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "train_freq": self.train_freq,
                    "gradient_steps": self.gradient_steps,
                    "policy_delay": self.policy_delay,
                    "target_policy_noise": self.target_policy_noise,
                    "target_noise_clip": self.target_noise_clip,
                    "total_timesteps": total_timesteps
                })
                
                # 记录模型架构信息
                self.log_model_architecture_to_mlflow()
                
                print(f"MLflow跟踪已初始化: {mlflow.get_artifact_uri()}")
            except Exception as e:
                print(f"MLflow初始化失败: {e}")
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
    
    def _update_policy(self, gradient_steps: int, batch_size: int = 64):
        """重写策略更新以添加梯度裁剪和学习率衰减"""
        # 确保日志记录器已初始化
        if self._logger is None:
            self._logger = configure(None, ["stdout"])
        
        # 调用父类方法前收集指标
        policy_losses = []
        critic_losses = []
        
        # 执行策略更新
        for _ in range(gradient_steps):
            # 更新critic
            critic_loss = self._update_critic(batch_size)
            if critic_loss is not None:
                critic_losses.append(critic_loss.item())
            
            # 延迟更新actor
            if self._n_updates % self.policy_delay == 0:
                # 计算actor loss
                actor_loss = self._update_actor(batch_size)
                if actor_loss is not None:
                    policy_losses.append(actor_loss.item())
                
                # 更新目标网络
                self._update_target_networks()
            
            self._n_updates += 1
        
        # 记录平均策略损失
        if len(policy_losses) > 0:
            policy_loss_mean = np.mean(policy_losses)
            self.logger.record("train/policy_loss", policy_loss_mean)
            
            # 记录到MLflow（如果有活跃的运行）
            if mlflow.active_run() is not None:
                try:
                    mlflow.log_metric("policy_loss", policy_loss_mean, step=self._n_updates)
                except Exception as e:
                    print(f"MLflow记录策略损失失败: {e}")
        
        # 记录平均critic损失
        if len(critic_losses) > 0:
            critic_loss_mean = np.mean(critic_losses)
            self.logger.record("train/critic_loss", critic_loss_mean)
            
            # 记录到MLflow（如果有活跃的运行）
            if mlflow.active_run() is not None:
                try:
                    mlflow.log_metric("critic_loss", critic_loss_mean, step=self._n_updates)
                except Exception as e:
                    print(f"MLflow记录critic损失失败: {e}")
        
        # 学习率衰减
        if hasattr(self.policy.optimizer, "param_groups"):
            current_lr = self.policy.optimizer.param_groups[0]["lr"]
            self.logger.record("train/learning_rate", current_lr)
            
            # 记录到MLflow（如果有活跃的运行）
            if mlflow.active_run() is not None:
                try:
                    mlflow.log_metric("learning_rate", current_lr, step=self._n_updates)
                except Exception as e:
                    print(f"MLflow记录学习率失败: {e}")
        
        # 添加梯度裁剪
        if hasattr(self.policy, 'actor'):
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
        if hasattr(self.policy, 'critics'):
            for critic in self.policy.critics:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        
        # 学习率衰减 (训练超过5000步后开始) - 暂时禁用
        if self.num_timesteps > 5000 and self.num_timesteps % 5000 == 0:
            decay_factor = max(0.1, 1.0 - (self.num_timesteps - 5000) / 45000)  # 5k-50k线性衰减到10%
            current_lr = 3e-4 * decay_factor
            print(f"📉 Step {self.num_timesteps}: 学习率衰减计划 - 目标LR: {current_lr:.2e} (衰减因子: {decay_factor:.3f})")
            print(f"   注意: 动态学习率调整暂时禁用，使用固定学习率训练")
        
        return result
    
    def _setup_model(self) -> None:
        """模型设置增强"""
        super()._setup_model()
        
        # AMCL不确定性日志记录 - AMCL已停用，暂时注释
        # self.logger.record("amcl/effective_particles", -1)  # 占位符
        # self.logger.record("amcl/position_uncertainty", -1)
        # self.logger.record("amcl/convergence_score", -1)
    
    def _excluded_save_params(self) -> List[str]:
        """排除保存的参数"""
        excluded = super()._excluded_save_params()
        return excluded
        
    @property
    def logger(self) -> Logger:
        """确保始终有可用的日志记录器"""
        if not hasattr(self, "_logger") or self._logger is None:
            return self._default_logger
        return self._logger