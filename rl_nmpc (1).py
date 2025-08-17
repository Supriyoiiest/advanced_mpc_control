"""
Reinforcement Learning Nonlinear Model Predictive Control (RL-NMPC)

This module implements the RL-NMPC algorithm that combines the optimization-based
approach of MPC with the learning capabilities of reinforcement learning.

Mathematical Foundation:
The RL-NMPC combines MPC optimization with RL policy learning:

MPC Objective:
J(x₀, U) = Σᵢ₌₀^{N-1} l(xᵢ, uᵢ) + Vf(xₙ)

where:
- l(xᵢ, uᵢ) is the stage cost
- Vf(xₙ) is the terminal cost
- N is the prediction horizon

RL Component:
The actor-critic framework is used to:
1. Actor π_θ(u|x): Policy network that suggests control actions
2. Critic V_φ(x): Value network that estimates state values
3. Advantage A(x,u) = Q(x,u) - V(x): Used for policy improvement

Integration:
The RL policy provides warm-start solutions for the MPC optimization,
while MPC provides high-quality training data for the RL components.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import deque
import scipy.optimize
from scipy.integrate import odeint
import warnings

from lsq_model import LSQModel, LSQConfig


@dataclass
class RLNMPCConfig:
    """Configuration for RL-NMPC controller."""
    state_dim: int
    action_dim: int
    horizon: int = 10
    dt: float = 0.1

    # MPC parameters
    Q: Optional[np.ndarray] = None  # State cost matrix
    R: Optional[np.ndarray] = None  # Input cost matrix
    P: Optional[np.ndarray] = None  # Terminal cost matrix
    action_bounds: Optional[Tuple[float, float]] = None
    state_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # RL parameters
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update rate
    batch_size: int = 256
    buffer_size: int = 100000

    # Neural network architectures
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"

    # Training parameters
    exploration_noise: float = 0.1
    warmup_episodes: int = 10
    mpc_weight: float = 0.5  # Weight for MPC vs RL action


class Actor(nn.Module):
    """Actor network for RL-NMPC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        action_bounds: Optional[Tuple[float, float]] = None
    ):
        super().__init__()

        self.action_bounds = action_bounds

        # Build network
        layers = []
        dims = [state_dim] + list(hidden_dims)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

        layers.append(nn.Linear(dims[-1], action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.network(state)

        # Scale to action bounds if provided
        if self.action_bounds is not None:
            low, high = self.action_bounds
            action = low + 0.5 * (high - low) * (action + 1)

        return action


class Critic(nn.Module):
    """Critic network for RL-NMPC (Q-function)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu"
    ):
        super().__init__()

        # Build Q-network
        layers = []
        dims = [state_dim + action_dim] + list(hidden_dims) + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Replay buffer for RL training."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.choice(self.size, batch_size, replace=False)

        return {
            "states": torch.FloatTensor(self.states[idx]),
            "actions": torch.FloatTensor(self.actions[idx]),
            "rewards": torch.FloatTensor(self.rewards[idx]),
            "next_states": torch.FloatTensor(self.next_states[idx]),
            "dones": torch.BoolTensor(self.dones[idx])
        }


class RLNMPC:
    """
    Reinforcement Learning Nonlinear Model Predictive Control.

    This controller combines MPC optimization with RL learning to achieve
    robust and adaptive control performance.
    """

    def __init__(
        self,
        config: Union[RLNMPCConfig, dict],
        model: Optional[LSQModel] = None
    ):
        if isinstance(config, dict):
            config = RLNMPCConfig(**config)

        self.config = config

        # Initialize cost matrices if not provided
        if config.Q is None:
            self.Q = np.eye(config.state_dim)
        else:
            self.Q = config.Q

        if config.R is None:
            self.R = np.eye(config.action_dim)
        else:
            self.R = config.R

        if config.P is None:
            self.P = np.eye(config.state_dim)
        else:
            self.P = config.P

        # Initialize model
        if model is None:
            model_config = LSQConfig(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                use_neural_augmentation=True
            )
            self.model = LSQModel(model_config)
        else:
            self.model = model

        # Initialize RL components
        self.actor = Actor(
            config.state_dim,
            config.action_dim,
            config.actor_hidden_dims,
            config.activation,
            config.action_bounds
        )

        self.critic = Critic(
            config.state_dim,
            config.action_dim,
            config.critic_hidden_dims,
            config.activation
        )

        self.target_actor = Actor(
            config.state_dim,
            config.action_dim,
            config.actor_hidden_dims,
            config.activation,
            config.action_bounds
        )

        self.target_critic = Critic(
            config.state_dim,
            config.action_dim,
            config.critic_hidden_dims,
            config.activation
        )

        # Copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, config.state_dim, config.action_dim
        )

        # Training statistics
        self.episode_count = 0
        self.training_stats = {
            "actor_loss": [],
            "critic_loss": [],
            "q_values": [],
            "mpc_cost": [],
            "episode_rewards": []
        }

    def mpc_step(
        self,
        state: np.ndarray,
        warm_start: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve MPC optimization problem.

        Args:
            state: Current state
            warm_start: Initial guess for optimization (from RL actor)

        Returns:
            Tuple of (optimal_action, mpc_info)
        """
        horizon = self.config.horizon
        action_dim = self.config.action_dim

        # Initial guess
        if warm_start is not None:
            u0 = warm_start.flatten()
        else:
            u0 = np.zeros(horizon * action_dim)

        # Bounds
        bounds = None
        if self.config.action_bounds is not None:
            low, high = self.config.action_bounds
            bounds = [(low, high)] * (horizon * action_dim)

        # Define cost function
        def cost_function(u_flat: np.ndarray) -> float:
            u_seq = u_flat.reshape(horizon, action_dim)

            # Simulate forward
            x = state.copy()
            total_cost = 0.0

            for i in range(horizon):
                # Stage cost
                stage_cost = x.T @ self.Q @ x + u_seq[i].T @ self.R @ u_seq[i]
                total_cost += stage_cost

                # Predict next state using model
                x_next, _ = self.model.predict(x, u_seq[i])
                x = x_next

            # Terminal cost
            terminal_cost = x.T @ self.P @ x
            total_cost += terminal_cost

            return total_cost

        # Solve optimization
        try:
            result = scipy.optimize.minimize(
                cost_function,
                u0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-4}
            )

            optimal_u = result.x.reshape(horizon, action_dim)
            success = result.success
            cost = result.fun

        except Exception as e:
            warnings.warn(f"MPC optimization failed: {e}")
            optimal_u = u0.reshape(horizon, action_dim)
            success = False
            cost = float("inf")

        info = {
            "success": success,
            "cost": cost,
            "iterations": getattr(result, "nit", 0),
            "control_sequence": optimal_u
        }

        return optimal_u[0], info  # Return first control action

    def rl_step(self, state: np.ndarray) -> np.ndarray:
        """Get action from RL policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
        return action

    def step(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get control action combining MPC and RL.

        Args:
            state: Current state
            training: Whether to add exploration noise

        Returns:
            Control action
        """
        state = np.atleast_1d(state)

        # Get RL action (for warm-start and blending)
        rl_action = self.rl_step(state)

        # Add exploration noise during training
        if training and self.episode_count > self.config.warmup_episodes:
            noise = np.random.normal(0, self.config.exploration_noise, rl_action.shape)
            rl_action += noise

            # Clip to bounds
            if self.config.action_bounds is not None:
                low, high = self.config.action_bounds
                rl_action = np.clip(rl_action, low, high)

        # Get MPC action with RL warm-start
        horizon = self.config.horizon
        warm_start = np.tile(rl_action, horizon)  # Repeat RL action
        mpc_action, mpc_info = self.mpc_step(state, warm_start)

        # Blend MPC and RL actions
        alpha = self.config.mpc_weight
        if self.episode_count < self.config.warmup_episodes:
            # Use more MPC during warmup
            alpha = 0.8

        final_action = alpha * mpc_action + (1 - alpha) * rl_action

        # Clip to bounds
        if self.config.action_bounds is not None:
            low, high = self.config.action_bounds
            final_action = np.clip(final_action, low, high)

        return final_action

    def update_rl(self) -> Dict[str, float]:
        """Update RL networks using replay buffer."""
        if self.replay_buffer.size < self.config.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "q_value": 0.0}

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q[dones] = 0.0
            target_q = rewards.unsqueeze(1) + self.config.gamma * target_q

        # Update critic
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        tau = self.config.tau
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value": current_q.mean().item()
        }

    def train(
        self,
        env,
        episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        eval_freq: int = 50,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the RL-NMPC controller.

        Args:
            env: Training environment
            episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            eval_freq: Frequency of evaluation episodes
            verbose: Whether to print training progress

        Returns:
            Training statistics dictionary
        """
        for episode in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gymnasium API

            episode_reward = 0.0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                # Get action
                action = self.step(state, training=True)

                # Environment step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Update model
                model_stats = self.model.update(state, action, next_state)

                # Store experience
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update RL networks
                if episode > self.config.warmup_episodes:
                    rl_stats = self.update_rl()

                    # Store statistics
                    self.training_stats["actor_loss"].append(rl_stats["actor_loss"])
                    self.training_stats["critic_loss"].append(rl_stats["critic_loss"])
                    self.training_stats["q_values"].append(rl_stats["q_value"])

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            self.episode_count += 1
            self.training_stats["episode_rewards"].append(episode_reward)

            # Evaluation and logging
            if verbose and episode % eval_freq == 0:
                avg_reward = np.mean(self.training_stats["episode_rewards"][-eval_freq:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Model Updates: {self.model.update_count}")

        return self.training_stats

    def save(self, filepath: str) -> None:
        """Save controller state."""
        torch.save({
            "config": self.config,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "target_actor_state": self.target_actor.state_dict(),
            "target_critic_state": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "episode_count": self.episode_count,
            "training_stats": self.training_stats
        }, filepath)

        # Save model separately
        model_path = filepath.replace(".pth", "_model.npz")
        self.model.save_model(model_path)

    def load(self, filepath: str) -> None:
        """Load controller state."""
        checkpoint = torch.load(filepath, map_location="cpu")

        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.target_actor.load_state_dict(checkpoint["target_actor_state"])
        self.target_critic.load_state_dict(checkpoint["target_critic_state"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.episode_count = checkpoint["episode_count"]
        self.training_stats = checkpoint["training_stats"]

        # Load model
        model_path = filepath.replace(".pth", "_model.npz")
        try:
            self.model.load_model(model_path)
        except FileNotFoundError:
            print(f"Model file {model_path} not found, using default model")
