# Advanced MPC Control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Colab-lightgrey)](https://github.com/advanced-mpc-control)

A comprehensive Python library for advanced Model Predictive Control (MPC) with Reinforcement Learning (RL) integration and Least-Squares (LSQ) modeling. Implements state-of-the-art control algorithms with neural network enhancement and cross-platform compatibility.

## 🚀 Features

### Control Algorithms
- **RL-NMPC**: Reinforcement Learning-assisted Nonlinear Model Predictive Control
- **RL+CMPC**: RL-informed Constrained MPC with critic-driven robustness
- **AMPC**: Adaptive MPC with online recursive least-squares updates
- **Attention-MPC**: Attention-layer-enhanced MPC (Afty-layer augmentation)

### Core Capabilities
- 🔧 **Unified API**: Single interface for all controllers via `BaseMPC`
- 📊 **LSQ Modeling**: Linear, polynomial, ridge/lasso, and neural hybrid models
- 🧠 **Neural Networks**: MLP, RNN, Attention, and Actor-Critic architectures
- 🎮 **MuJoCo Integration**: CartPole, HalfCheetah, and custom environment support
- 📈 **Visualization**: Trajectory plotting, training curves, and video generation
- 🎥 **Video Export**: MP4 and GIF rendering of simulation episodes
- 📚 **Documentation**: Complete LaTeX manual with mathematical derivations
- ✅ **Cross-Platform**: Windows, Linux, macOS, and Google Colab support

## 📦 Installation

### Quick Install
```bash
pip install -e .
```

### With Optional Dependencies
```bash
pip install torch gymnasium[mujoco] mujoco moviepy matplotlib seaborn
pip install -e .
```

### Google Colab Setup
```python
# Run this in a Colab cell
!python -m advanced_mpc_control.examples.colab_setup
```

### Prerequisites
- Python 3.9-3.12
- NumPy, SciPy (required)
- PyTorch (optional, for neural networks)
- MuJoCo + Gymnasium (optional, for simulation)

## 🏗️ Architecture

```
advanced_mpc_control/
├── 🔧 core/                    # Core building blocks
│   ├── base_mpc.py            # Generic MPC solver with SciPy SLSQP
│   ├── lsq_modeling.py        # LSQ variants + Neural-LSQ
│   └── neural_networks.py     # PyTorch networks + training utilities
├── 🎯 controllers/            # Ready-to-use MPC controllers
│   ├── rl_nmpc.py            # RL-assisted NMPC
│   ├── rl_cmpc.py            # RL + constrained MPC
│   ├── ampc.py               # Adaptive MPC
│   └── attention_mpc.py      # Attention-enhanced MPC
├── 🎮 simulation/            # MuJoCo environment wrappers
│   ├── mujoco_envs.py
│   ├── cartpole_env.py
│   └── halfcheetah_env.py
├── 📊 utils/                 # Plotting and video utilities
├── 🧪 tests/                 # Unit tests and benchmarks
├── 📚 examples/              # Usage examples and demos
└── 📖 docs/                  # LaTeX documentation
```

## 🎯 Quick Start

### Basic Usage
```python
from advanced_mpc_control import CartPoleEnv, RLNMPC, create_network

# Create environment
env = CartPoleEnv()
state_dim = env.observation_space.shape[0]  # 4
action_dim = env.action_space.shape[0]      # 1

# Initialize RL-NMPC controller
mpc = RLNMPC(
    prediction_horizon=25,
    control_horizon=10, 
    state_dim=state_dim,
    control_dim=action_dim,
    dt=0.05,
    actor_critic=create_network(
        'actor_critic', 
        state_dim, 
        action_dim,
        action_bound=float(env.action_space.high[0])
    )
)

# Control loop
state, _ = env.reset(seed=42)
for step in range(500):
    # MPC computes optimal action
    action = mpc.control_step(state)
    
    # Apply action to environment  
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

# Get performance metrics
metrics = mpc.get_performance_metrics()
print(f"Mean cost: {metrics['mean_cost']:.3f}")
print(f"Control effort: {metrics['control_effort']:.3f}")
```

### System Identification with LSQ
```python
from advanced_mpc_control.core.lsq_modeling import SystemIdentificationLSQ
import numpy as np

# Generate or load trajectory data
states = np.random.randn(1000, 4)        # State trajectory
inputs = np.random.randn(1000, 1)        # Input trajectory  
next_states = np.random.randn(1000, 4)   # Next state trajectory

# Fit linear system: x[k+1] = A*x[k] + B*u[k] + bias
sysid = SystemIdentificationLSQ(state_dim=4, input_dim=1, model_type='linear')
training_info = sysid.fit(states, inputs, next_states)

print(f"Training MSE: {training_info['training_mse']:.6f}")
print(f"System stability (spectral radius): {training_info['spectral_radius']:.3f}")

# Use identified system for prediction
predicted_states = sysid.predict_step(states, inputs)
simulation_trajectory = sysid.simulate(initial_state=states[0], 
                                     input_sequence=inputs[:100])
```

### Neural Network Training
```python
from advanced_mpc_control import NeuralLSQModel
import torch

# Create hybrid neural-LSQ model
model = NeuralLSQModel(
    input_dim=5, 
    output_dim=1,
    hidden_dims=[64, 32],
    use_lsq_init=True  # Initialize with LSQ solution
)

# Generate training data
X_train = np.random.randn(1000, 5)
y_train = np.random.randn(1000, 1)

# Train with both LSQ initialization and gradient descent
training_info = model.fit(
    X_train, y_train,
    epochs=500,
    learning_rate=1e-3,
    batch_size=32,
    validation_split=0.2
)

# Make predictions
X_test = np.random.randn(100, 5)
predictions = model.predict(X_test)
```

## 🎮 Simulation Environments

### CartPole (Classic Control)
```python
from advanced_mpc_control import CartPoleEnv

env = CartPoleEnv()
# 4-dimensional state: [position, velocity, angle, angular_velocity]
# 1-dimensional action: [force]
```

### HalfCheetah (MuJoCo Locomotion)
```python
from advanced_mpc_control import HalfCheetahEnv

env = HalfCheetahEnv(max_steps=1000)
# 17-dimensional state space
# 6-dimensional action space (joint torques)
```

### Custom Environment
```python
from advanced_mpc_control.simulation import MuJoCoEnvironment

env = MuJoCoEnvironment(
    xml_path="path/to/custom.xml",
    max_steps=500,
    render_mode="rgb_array"
)
```

## 📊 Visualization and Analysis

### Trajectory Plotting
```python
from advanced_mpc_control import plot_trajectory, plot_control_inputs

# Plot state trajectories
states = np.array(mpc.state_history)
plot_trajectory(states, labels=["x", "ẋ", "θ", "θ̇"], 
               title="CartPole State Evolution")

# Plot control inputs
controls = np.array(mpc.control_history)  
plot_control_inputs(controls, labels=["Force (N)"],
                   title="Control Effort")
```

### Video Generation
```python
from advanced_mpc_control import create_gif, create_video

# Record environment frames during simulation
env = CartPoleEnv(render_mode="rgb_array")
frames = []

for step in range(200):
    action = mpc.control_step(state)
    state, _, done, _, _ = env.step(action)
    frames.append(env.render())
    if done:
        break

# Generate video outputs
create_gif("cartpole_episode.gif", frames, fps=30)
create_video("cartpole_episode.mp4", frames, fps=30)
```

### Training Analysis
```python
from advanced_mpc_control.utils import TrainingPlotter

plotter = TrainingPlotter()
plotter.plot_losses(train_losses, val_losses, save_path="training_curves.png")
plotter.plot_metrics(metrics_dict, save_path="performance_metrics.png")
```

## 🎯 Advanced Usage

### Custom Controller Implementation
```python
from advanced_mpc_control.core import BaseMPC
import numpy as np

class CustomMPC(BaseMPC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
        
    def predict_trajectory(self, x0, u_sequence):
        # Implement custom dynamics prediction
        x_traj = np.zeros((self.prediction_horizon + 1, self.state_dim))
        x_traj[0] = x0
        
        for k in range(self.prediction_horizon):
            u_k = u_sequence[min(k, len(u_sequence) - 1)]
            # Custom dynamics: x[k+1] = f(x[k], u[k])  
            x_traj[k + 1] = self.custom_dynamics(x_traj[k], u_k)
            
        return x_traj
    
    def custom_dynamics(self, x, u):
        # Implement your system dynamics
        return x  # Placeholder
```

### Multi-Environment Benchmarking
```python
from advanced_mpc_control import RLNMPC, AMPC, AttentionMPC

controllers = {
    'RL-NMPC': RLNMPC(state_dim=4, control_dim=1, prediction_horizon=20),
    'AMPC': AMPC(state_dim=4, control_dim=1, prediction_horizon=20), 
    'Attention-MPC': AttentionMPC(state_dim=4, control_dim=1, prediction_horizon=20)
}

environments = [CartPoleEnv(), HalfCheetahEnv()]

results = {}
for ctrl_name, controller in controllers.items():
    for env in environments:
        # Run benchmark
        metrics = run_benchmark(controller, env, episodes=10)
        results[f"{ctrl_name}_{env.__class__.__name__}"] = metrics

print_benchmark_results(results)
```

## 📚 Examples and Demos

### Run Built-in Examples
```bash
# Basic usage demonstration
python -m advanced_mpc_control.examples.basic_usage

# Full simulation demo with all controllers
python -m advanced_mpc_control.examples.simulation_demo  

# Google Colab setup script
python -m advanced_mpc_control.examples.colab_setup
```

### Example Scripts Include:
- `basic_usage.py`: LSQ fitting + simple MPC loop
- `simulation_demo.py`: All controllers on CartPole and HalfCheetah
- `neural_training.py`: Neural network training workflows
- `video_generation.py`: Create MP4/GIF outputs
- `benchmarking.py`: Performance comparison across controllers

## 🧪 Testing

Run the test suite:
```bash
# Quick tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=advanced_mpc_control --cov-report=html

# Specific test categories
pytest tests/test_controllers.py -v  # Controller tests
pytest tests/test_lsq.py -v          # LSQ model tests
pytest tests/test_simulation.py -v   # Environment tests
```

## 📖 Documentation

### LaTeX Manual
Comprehensive mathematical documentation with derivations:

```bash
cd docs/
latexmk -pdf advanced_mpc_control.tex
```

**Contents:**
- Mathematical formulation of each controller
- LSQ theory and numerical stability analysis  
- Neural network architectures and training
- Attention mechanism derivation
- Complete API reference
- UML diagrams and class relationships

### API Reference
Generate HTML docs:
```bash
cd docs/
make html  # Requires sphinx
```

## ⚙️ Configuration

### Global Configuration
```python
from advanced_mpc_control import get_config, set_config

# View current config
config = get_config()
print(config)

# Update settings
set_config(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    default_dtype='float32',
    torch_available=True
)
```

### Controller-Specific Options
```python
mpc = RLNMPC(
    # MPC parameters
    prediction_horizon=30,
    control_horizon=10,
    dt=0.05,
    
    # Cost function weights
    Q=np.diag([10, 1, 10, 1]),  # State weights
    R=np.array([[0.1]]),        # Control weights  
    P=np.diag([10, 1, 10, 1]),  # Terminal weights
    
    # Constraints
    u_min=np.array([-10]),
    u_max=np.array([10]),
    x_min=np.array([-5, -10, -1, -10]),
    x_max=np.array([5, 10, 1, 10]),
    
    # Solver options
    solver_options={
        'method': 'SLSQP',
        'options': {'maxiter': 200, 'ftol': 1e-9}
    }
)
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality
4. **Run** tests and linting (`pytest && ruff check`)
5. **Commit** changes (`git commit -am 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`) 
7. **Open** a Pull Request

### Development Setup
```bash
git clone https://github.com/Supriyoiiest/advanced-mpc-control.git
cd advanced-mpc-control
pip install -e .[dev]
pre-commit install
```

### Code Style
- Follow **PEP 8** 
- Use **type hints** for all public APIs
- Add **docstrings** with examples
- Run **ruff** for linting
- Achieve **>90% test coverage**

## 🗺️ Roadmap

### Near Term (v1.1)
- [ ] CasADi backend option for faster QP solving
- [ ] Robust/tube MPC variants
- [ ] Extended Kalman Filter integration
- [ ] MATLAB/Simulink interface

### Medium Term (v1.2)  
- [ ] Distributed MPC for multi-agent systems
- [ ] Stochastic MPC with uncertainty quantification
- [ ] Real-time performance profiling and optimization
- [ ] Industrial control examples (chemical plants, robotics)

### Long Term (v2.0)
- [ ] GPU-accelerated batch MPC for RL training
- [ ] Differentiable MPC layers for end-to-end learning
- [ ] Integration with modern RL libraries (Stable Baselines3, RLLib)
- [ ] Cloud deployment utilities and APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Citation

If you use this library in your research, please cite:

```bibtex
@software{advanced_mpc_control,
  title={Advanced MPC Control: A Python Library for RL-Enhanced Model Predictive Control},
  author={Advanced MPC Control Development Team},
  year={2025},
  url={https://github.com/Supriyoiiest/advanced_mpc_control},
  version={0.0.1e}
}
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Supriyoiiest/advanced-mpc-control/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Supriyoiiest/advanced-mpc-control/discussions)  
- **Documentation**: [Online Docs](https://advanced-mpc-control.readthedocs.io)


*Star ⭐ this repo if you find it helpful!*
