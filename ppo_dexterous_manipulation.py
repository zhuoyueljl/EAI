from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class MultiModalEncoder(nn.Module):
    """
    为具身智能灵巧操作预留的多模态编码器:
    - proprio: 关节角、关节速度、末端位姿、夹爪/手指状态
    - object: 物体位姿、速度、接触状态、目标姿态
    - visual: 可选的视觉特征，默认假设已由外部 backbone 提前提取
    - tactile: 可选的触觉特征
    """

    def __init__(
        self,
        proprio_dim: int,
        object_dim: int,
        visual_dim: int = 0,
        tactile_dim: int = 0,
        hidden_dim: int = 256,
        fused_dim: int = 256,
    ) -> None:
        super().__init__()
        self.proprio_encoder = build_mlp(proprio_dim, [hidden_dim], hidden_dim)
        self.object_encoder = build_mlp(object_dim, [hidden_dim], hidden_dim)
        self.visual_encoder = build_mlp(visual_dim, [hidden_dim], hidden_dim) if visual_dim > 0 else None
        self.tactile_encoder = build_mlp(tactile_dim, [hidden_dim], hidden_dim) if tactile_dim > 0 else None

        num_streams = 2 + int(self.visual_encoder is not None) + int(self.tactile_encoder is not None)
        self.fusion = build_mlp(hidden_dim * num_streams, [hidden_dim, hidden_dim], fused_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = [
            self.proprio_encoder(obs["proprio"]),
            self.object_encoder(obs["object"]),
        ]
        if self.visual_encoder is not None:
            features.append(self.visual_encoder(obs["visual"]))
        if self.tactile_encoder is not None:
            features.append(self.tactile_encoder(obs["tactile"]))
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)


class ActorCritic(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        object_dim: int,
        action_dim: int,
        visual_dim: int = 0,
        tactile_dim: int = 0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = MultiModalEncoder(
            proprio_dim=proprio_dim,
            object_dim=object_dim,
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            fused_dim=hidden_dim,
        )
        self.actor_mean = build_mlp(hidden_dim, [hidden_dim, hidden_dim], action_dim)
        self.critic = build_mlp(hidden_dim, [hidden_dim, hidden_dim], 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[Normal, torch.Tensor]:
        latent = self.encoder(obs)
        mean = self.actor_mean(latent)
        std = self.log_std.exp().clamp(min=1e-4, max=10.0)
        dist = Normal(mean, std)
        value = self.critic(latent).squeeze(-1)
        return dist, value

    def act(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class PPOConfig:
    device: str = "cpu"
    rollout_steps: int = 1024
    num_envs: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    epochs: int = 10
    minibatch_size: int = 1024
    normalize_advantage: bool = True
    target_kl: Optional[float] = 0.02
    action_limit: float = 1.0


class RolloutBuffer:
    def __init__(self) -> None:
        self.obs: Dict[str, List[torch.Tensor]] = {}
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        for key, tensor in obs.items():
            self.obs.setdefault(key, []).append(tensor.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())
        self.values.append(value.detach())

    def stack(self) -> Dict[str, torch.Tensor]:
        batch = {key: torch.stack(value, dim=0) for key, value in self.obs.items()}
        batch["actions"] = torch.stack(self.actions, dim=0)
        batch["log_probs"] = torch.stack(self.log_probs, dim=0)
        batch["rewards"] = torch.stack(self.rewards, dim=0)
        batch["dones"] = torch.stack(self.dones, dim=0)
        batch["values"] = torch.stack(self.values, dim=0)
        return batch

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    def __init__(self, model: ActorCritic, config: PPOConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def select_action(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = {k: v.to(self.config.device) for k, v in obs.items()}
        with torch.no_grad():
            action, log_prob, value = self.model.act(obs)
            action = action.clamp(-self.config.action_limit, self.config.action_limit)
        return action, log_prob, value

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)

        for t in reversed(range(rewards.shape[0])):
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_values * next_non_terminal - values[t]
            last_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: Dict[str, torch.Tensor], next_obs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        next_obs = {k: v.to(self.config.device) for k, v in next_obs.items()}
        with torch.no_grad():
            _, next_value = self.model.forward(next_obs)

        rewards = rollout["rewards"].to(self.config.device)
        dones = rollout["dones"].to(self.config.device)
        values = rollout["values"].to(self.config.device)
        old_actions = rollout["actions"].to(self.config.device)
        old_log_probs = rollout["log_probs"].to(self.config.device)
        obs = {k: v.to(self.config.device) for k, v in rollout.items() if k not in {"actions", "log_probs", "rewards", "dones", "values"}}

        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_steps, num_envs = rewards.shape
        batch_size = num_steps * num_envs

        flat_obs = {k: v.reshape(batch_size, *v.shape[2:]) for k, v in obs.items()}
        flat_actions = old_actions.reshape(batch_size, -1)
        flat_log_probs = old_log_probs.reshape(batch_size)
        flat_values = values.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)

        indices = torch.arange(batch_size, device=self.config.device)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        num_updates = 0

        for _ in range(self.config.epochs):
            permutation = indices[torch.randperm(batch_size)]
            for start in range(0, batch_size, self.config.minibatch_size):
                batch_idx = permutation[start : start + self.config.minibatch_size]
                minibatch_obs = {k: v[batch_idx] for k, v in flat_obs.items()}
                minibatch_actions = flat_actions[batch_idx]
                minibatch_old_log_probs = flat_log_probs[batch_idx]
                minibatch_old_values = flat_values[batch_idx]
                minibatch_advantages = flat_advantages[batch_idx]
                minibatch_returns = flat_returns[batch_idx]

                new_log_probs, entropy, new_values = self.model.evaluate_actions(minibatch_obs, minibatch_actions)
                ratio = (new_log_probs - minibatch_old_log_probs).exp()
                unclipped = ratio * minibatch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * minibatch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred_clipped = minibatch_old_values + (new_values - minibatch_old_values).clamp(
                    -self.config.value_clip,
                    self.config.value_clip,
                )
                value_loss_unclipped = (new_values - minibatch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - minibatch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (minibatch_old_log_probs - new_log_probs).mean().abs().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                num_updates += 1

                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break

        if num_updates > 0:
            stats = {k: v / num_updates for k, v in stats.items()}
        return stats


class DummyDexterousEnv:
    """
    一个演示接口，方便直接替换成真实环境:
    - reset() -> obs 字典
    - step(action) -> next_obs, reward, done, info
    真实项目中可以接入 Isaac Gym、Mujoco、ManiSkill、RLBench 等环境。
    """

    def __init__(
        self,
        num_envs: int,
        proprio_dim: int,
        object_dim: int,
        action_dim: int,
        visual_dim: int = 0,
        tactile_dim: int = 0,
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.object_dim = object_dim
        self.visual_dim = visual_dim
        self.tactile_dim = tactile_dim
        self.action_dim = action_dim
        self.device = device
        self.step_count = torch.zeros(num_envs, device=device)

    def _sample_obs(self) -> Dict[str, torch.Tensor]:
        obs = {
            "proprio": torch.randn(self.num_envs, self.proprio_dim, device=self.device),
            "object": torch.randn(self.num_envs, self.object_dim, device=self.device),
        }
        if self.visual_dim > 0:
            obs["visual"] = torch.randn(self.num_envs, self.visual_dim, device=self.device)
        if self.tactile_dim > 0:
            obs["tactile"] = torch.randn(self.num_envs, self.tactile_dim, device=self.device)
        return obs

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count.zero_()
        return self._sample_obs()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        self.step_count += 1
        next_obs = self._sample_obs()

        action_penalty = 0.01 * action.pow(2).sum(dim=-1)
        task_progress = torch.randn(self.num_envs, device=self.device) * 0.1 + 1.0
        reward = task_progress - action_penalty

        done = (self.step_count >= 200).float()
        reset_mask = done.bool()
        if reset_mask.any():
            self.step_count[reset_mask] = 0

        info = {
            "success": (reward > 0.8).float(),
        }
        return next_obs, reward, done, info


def train_ppo(
    env: DummyDexterousEnv,
    agent: PPOAgent,
    total_updates: int = 100,
) -> None:
    obs = env.reset()

    for update_idx in range(total_updates):
        buffer = RolloutBuffer()
        episode_rewards = []
        episode_success = []

        for _ in range(agent.config.rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, log_prob, reward, done, value)
            obs = next_obs
            episode_rewards.append(reward.mean().item())
            if "success" in info:
                episode_success.append(info["success"].mean().item())

        stats = agent.update(buffer.stack(), obs)
        mean_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
        mean_success = sum(episode_success) / max(len(episode_success), 1) if episode_success else 0.0

        print(
            f"[update {update_idx:04d}] reward={mean_reward:.4f} "
            f"success={mean_success:.4f} policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
            f"approx_kl={stats['approx_kl']:.5f}"
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    config = PPOConfig(
        device="cpu",
        rollout_steps=256,
        num_envs=32,
        epochs=4,
        minibatch_size=512,
        learning_rate=3e-4,
    )

    env = DummyDexterousEnv(
        num_envs=config.num_envs,
        proprio_dim=48,
        object_dim=24,
        visual_dim=128,
        tactile_dim=32,
        action_dim=20,
        device=config.device,
    )

    model = ActorCritic(
        proprio_dim=48,
        object_dim=24,
        visual_dim=128,
        tactile_dim=32,
        action_dim=20,
        hidden_dim=256,
    )
    agent = PPOAgent(model, config)
    train_ppo(env, agent, total_updates=10)
