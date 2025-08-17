# Let's create a comprehensive test script for CartPole environment
cartpole_test_code = '''"""
Test script for RL-NMPC controller on CartPole environment.

This script demonstrates the usage of the advanced control library
on the CartPole environment, showing training and evaluation.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from typing import Dict, List
import os
import sys
import time

# Import our controllers and models
try:
    from lsq_model import LSQModel, LSQConfig
    from rl_nmpc import RLNMPC, RLNMPCConfig
    from ampc import AMPC, AMPCConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the controller modules are in the same directory")
    sys.exit(1)


class CartPoleWrapper:
    """
    Wrapper for CartPole environment to ensure compatibility
    with our control library.
    """
    
    def __init__(self, render_mode: str = None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.state_dim = 4  # [cart_pos, cart_vel, pole_angle, pole_vel]
        self.action_dim = 1  # force (discretized as -1 or +1)
        self.max_steps = 500
        
    def reset(self):
        """Reset environment and return initial state."""
        state, info = self.env.reset()
        return state
    
    def step(self, action):
        """
        Step environment with continuous action, convert to discrete.
        
        Args:
            action: Continuous action in [-1, 1]
            
        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Convert continuous action to discrete
        discrete_action = 0 if action[0] < 0 else 1
        
        next_state, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # Custom reward shaping for better learning
        # Penalize large angles and positions
        angle_penalty = -10.0 * abs(next_state[2])  # pole angle
        position_penalty = -1.0 * abs(next_state[0])  # cart position
        
        # Bonus for staying upright
        upright_bonus = 10.0 if abs(next_state[2]) < 0.1 else 0.0
        
        shaped_reward = reward + angle_penalty + position_penalty + upright_bonus
        
        return next_state, shaped_reward, terminated, truncated, info
    
    def close(self):
        """Close environment."""
        self.env.close()


def test_lsq_model():
    """Test the LSQ model with random data."""
    print("Testing LSQ Model...")
    
    config = LSQConfig(
        state_dim=4,
        action_dim=1,
        memory_length=100,
        use_neural_augmentation=True
    )
    
    model = LSQModel(config)
    
    # Generate random training data
    np.random.seed(42)
    
    prediction_errors = []
    for i in range(200):
        state = np.random.randn(4) * 0.1
        action = np.random.randn(1) * 0.5
        next_state = 0.95 * state + 0.1 * np.random.randn(4) + 0.1 * action[0] * np.ones(4)
        
        # Update model
        stats = model.update(state, action, next_state)
        prediction_errors.append(stats["prediction_mse"])
        
        # Test prediction
        if i % 50 == 0:
            pred_state, pred_info = model.predict(state, action)
            error = np.linalg.norm(pred_state - next_state)
            print(f"Step {i}: Prediction error = {error:.4f}, Confidence = {pred_info['confidence']:.4f}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(prediction_errors)
    plt.title("LSQ Model Learning Curve")
    plt.xlabel("Update Step")
    plt.ylabel("Prediction MSE")
    plt.yscale('log')
    
    # Test model parameters
    A, B = model.get_linearization(np.zeros(4))
    plt.subplot(1, 2, 2)
    plt.imshow(A, cmap='viridis')
    plt.title("Learned A Matrix")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("lsq_model_test.png", dpi=150, bbox_inches='tight')
    print("✓ LSQ Model test completed. Results saved to lsq_model_test.png")
    
    return model


def test_rl_nmpc_cartpole():
    """Test RL-NMPC controller on CartPole."""
    print("\\nTesting RL-NMPC on CartPole...")
    
    # Create environment
    env = CartPoleWrapper()
    
    # Create controller configuration
    config = RLNMPCConfig(
        state_dim=4,
        action_dim=1,
        horizon=8,
        action_bounds=(-1.0, 1.0),
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=64,
        warmup_episodes=20,
        mpc_weight=0.6
    )
    
    # Initialize controller
    controller = RLNMPC(config)
    
    print("Training RL-NMPC controller...")
    
    # Training loop
    training_stats = controller.train(
        env=env,
        episodes=200,
        max_steps_per_episode=500,
        eval_freq=25,
        verbose=True
    )
    
    print("\\nEvaluating trained controller...")
    
    # Evaluation
    eval_rewards = []
    eval_episode_lengths = []
    
    for eval_ep in range(10):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(500):
            action = controller.step(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        eval_episode_lengths.append(episode_length)
        print(f"Eval Episode {eval_ep + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Training rewards
    plt.subplot(1, 3, 1)
    episode_rewards = training_stats["episode_rewards"]
    window_size = 20
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-window_size):i+1]) 
                       for i in range(len(episode_rewards))]
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    plt.plot(smoothed_rewards, label='Smoothed')
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # Actor/Critic losses
    plt.subplot(1, 3, 2)
    if training_stats["actor_loss"]:
        plt.plot(training_stats["actor_loss"], label='Actor Loss', alpha=0.7)
        plt.plot(training_stats["critic_loss"], label='Critic Loss', alpha=0.7)
        plt.title("Training Losses")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
    
    # Evaluation results
    plt.subplot(1, 3, 3)
    plt.bar(range(len(eval_rewards)), eval_rewards)
    plt.title("Evaluation Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("rl_nmpc_cartpole_results.png", dpi=150, bbox_inches='tight')
    
    # Print summary statistics
    print(f"\\nRL-NMPC CartPole Results:")
    print(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(eval_episode_lengths):.1f} ± {np.std(eval_episode_lengths):.1f}")
    print(f"Success Rate (>= 450 steps): {sum(1 for l in eval_episode_lengths if l >= 450) / len(eval_episode_lengths) * 100:.1f}%")
    
    # Save controller
    controller.save("rl_nmpc_cartpole.pth")
    print("✓ RL-NMPC test completed. Results saved to rl_nmpc_cartpole_results.png")
    
    env.close()
    return controller, training_stats


def test_ampc_cartpole():
    """Test AMPC controller on CartPole."""
    print("\\nTesting AMPC on CartPole...")
    
    # Create environment
    env = CartPoleWrapper()
    
    # Create controller configuration
    config = AMPCConfig(
        state_dim=4,
        action_dim=1,
        horizon=6,
        action_bounds=(-1.0, 1.0),
        adaptation_rate=0.05,
        forgetting_factor=0.98,
        uncertainty_weight=0.1
    )
    
    # Initialize controller
    controller = AMPC(config)
    
    print("Testing AMPC adaptation...")
    
    # Test adaptation
    episode_rewards = []
    adaptation_stats = []
    
    for episode in range(100):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(300):
            # Get control action
            action = controller.step(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Adapt model
            adapt_stats = controller.adapt(state, action, next_state)
            adaptation_stats.append(adapt_stats)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 20 == 0:
            model_stats = controller.get_adaptation_statistics()
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Pred Error = {model_stats['prediction_error_mean']:.4f}, "
                  f"Param Stability = {np.mean(model_stats['parameter_stability']):.4f}")
    
    # Plot adaptation results
    plt.figure(figsize=(15, 5))
    
    # Episode rewards
    plt.subplot(1, 3, 1)
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-10):i+1]) 
                       for i in range(len(episode_rewards))]
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    plt.plot(smoothed_rewards, label='Smoothed', linewidth=2)
    plt.title("AMPC Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # Prediction errors
    plt.subplot(1, 3, 2)
    prediction_errors = [stats["prediction_error"] for stats in adaptation_stats]
    window_size = 50
    smoothed_errors = [np.mean(prediction_errors[max(0, i-window_size):i+1]) 
                      for i in range(len(prediction_errors))]
    plt.plot(smoothed_errors)
    plt.title("Model Prediction Error")
    plt.xlabel("Step")
    plt.ylabel("Prediction Error")
    plt.yscale('log')
    plt.grid(True)
    
    # Parameter stability
    plt.subplot(1, 3, 3)
    param_changes = [stats["parameter_change"] for stats in adaptation_stats]
    smoothed_changes = [np.mean(param_changes[max(0, i-window_size):i+1]) 
                       for i in range(len(param_changes))]
    plt.plot(smoothed_changes)
    plt.title("Parameter Change Rate")
    plt.xlabel("Step")
    plt.ylabel("Parameter Change")
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("ampc_cartpole_results.png", dpi=150, bbox_inches='tight')
    
    # Final evaluation
    print("\\nFinal AMPC evaluation...")
    eval_rewards = []
    
    for eval_ep in range(5):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(500):
            action = controller.step(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        print(f"Eval Episode {eval_ep + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\\nAMPC CartPole Results:")
    print(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    # Save controller
    controller.save("ampc_cartpole.npz")
    print("✓ AMPC test completed. Results saved to ampc_cartpole_results.png")
    
    env.close()
    return controller


def install_mujoco_colab():
    """
    Installation instructions for MuJoCo on Google Colab.
    """
    colab_install_code = '''
    # Google Colab MuJoCo Installation
    # Run these commands in a Colab cell:
    
    !apt-get install -y \\
        libosmesa6-dev \\
        libgl1-mesa-glx \\
        libglfw3 \\
        libgl1-mesa-dev \\
        libglew-dev \\
        patchelf
    
    # Install MuJoCo
    !pip install mujoco gymnasium[mujoco]
    
    # Test installation
    import mujoco
    import gymnasium as gym
    
    # Create a simple MuJoCo environment
    env = gym.make("HalfCheetah-v4")
    print("MuJoCo installation successful!")
    '''
    
    print("Google Colab MuJoCo Installation Instructions:")
    print("=" * 50)
    print(colab_install_code)


def main():
    """Main test function."""
    print("Advanced Control Library Test Suite")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test LSQ Model
    lsq_model = test_lsq_model()
    
    # Test RL-NMPC on CartPole
    rl_nmpc_controller, rl_training_stats = test_rl_nmpc_cartpole()
    
    # Test AMPC on CartPole
    ampc_controller = test_ampc_cartpole()
    
    # Print Colab installation instructions
    print("\\n" + "=" * 50)
    install_mujoco_colab()
    
    print("\\n" + "=" * 50)
    print("All tests completed successfully!")
    print("\\nGenerated files:")
    print("- lsq_model_test.png")
    print("- rl_nmpc_cartpole_results.png")  
    print("- ampc_cartpole_results.png")
    print("- rl_nmpc_cartpole.pth")
    print("- ampc_cartpole.npz")
    
    return {
        "lsq_model": lsq_model,
        "rl_nmpc_controller": rl_nmpc_controller,
        "ampc_controller": ampc_controller,
        "rl_training_stats": rl_training_stats
    }


if __name__ == "__main__":
    results = main()
'''

print("CartPole Test Script:")
print("=" * 50)
print(cartpole_test_code)

# Save the test script
with open("test_cartpole.py", "w") as f:
    f.write(cartpole_test_code)

print("\n✓ CartPole test script saved to test_cartpole.py")