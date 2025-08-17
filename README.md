Advanced MPC Control
A cross-platform Python library for modern Model Predictive Control (MPC) with Reinforcement Learning (RL) and Least-Squares (LSQ) modeling. It implements:

RL-NMPC: Reinforcement Learning–assisted Nonlinear MPC

RL+CMPC: RL-informed Constrained MPC with critic shaping

AMPC: Adaptive MPC with online LSQ system identification

Afty-MPC: Attention-layer–enhanced MPC (Afty = Attention layer)

Works on Windows, Linux, macOS, and Google Colab. Neural components (PyTorch) are optional.

Features
Unified BaseMPC interface for all controllers

LSQ modeling: linear, polynomial, ridge/lasso, and neural hybrids

Neural networks for dynamics, cost shaping, policy/value (MLP/RNN/Attention)

MuJoCo + Gymnasium simulation wrappers (CartPole, HalfCheetah)

Plotting utilities: trajectories, inputs, training curves

Video utilities: MP4 and GIF rendering of episodes

LaTeX documentation with derivations, diagrams, and API reference

Clean modular package, tested across platforms

Directory Structure
text
advanced_mpc_control/
├── core/
│   ├── base_mpc.py             # Base MPC implementation (SciPy SLSQP)
│   ├── lsq_modeling.py         # LSQ, nonlinear LSQ, system-ID, Neural-LSQ
│   └── neural_networks.py      # MLP, RNN, Attention, Actor-Critic, trainer
├── controllers/
│   ├── rl_nmpc.py              # RL-assisted NMPC
│   ├── rl_cmpc.py              # RL + constrained MPC (critic shaping)
│   ├── ampc.py                 # Adaptive MPC (online LSQ updates)
│   └── attention_mpc.py        # Attention-augmented MPC (Afty-layer)
├── simulation/
│   ├── mujoco_envs.py          # Generic MuJoCo wrapper
│   ├── cartpole_env.py
│   └── halfcheetah_env.py
├── utils/
│   ├── plotting.py             # Trajectories, inputs, training curves
│   ├── visualization.py
│   └── video_utils.py          # MP4 / GIF rendering
├── examples/
│   ├── basic_usage.py
│   ├── simulation_demo.py
│   └── colab_setup.py
├── tests/
└── docs/
    └── advanced_mpc_control.tex
Installation
Prerequisites:

Python 3.9–3.12

NumPy, SciPy

Optional: PyTorch (CPU/GPU)

Gymnasium + MuJoCo for simulations

Install editable:

bash
pip install -e .
Or minimal (without MuJoCo):

bash
pip install -e .[minimal]
Recommended extras:

bash
pip install torch gymnasium[mujoco] mujoco moviepy matplotlib
On Google Colab:

Run examples/colab_setup.py to install MuJoCo, Gymnasium, and set up rendering.

Quick Start
Basic MPC loop on CartPole:

python
from advanced_mpc_control import CartPoleEnv, RLNMPC, create_network

env = CartPoleEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape

mpc = RLNMPC(
    prediction_horizon=25,
    control_horizon=10,
    state_dim=state_dim,
    control_dim=action_dim,
    actor_critic=create_network(
        'actor_critic', state_dim, action_dim,
        action_bound=float(env.action_space.high)
    ),
)

state, _ = env.reset(seed=0)
for t in range(500):
    action = mpc.control_step(state)
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

metrics = mpc.get_performance_metrics()
print(metrics)
Controllers
All controllers extend BaseMPC and share:

python
u = controller.control_step(x0, x_ref=None, u_ref=None)
metrics = controller.get_performance_metrics()
RL-NMPC (controllers/rl_nmpc.py)

Uses an actor-critic to warm start and shape NMPC cost

Neural dynamics or cost approximation optional

RL+CMPC (controllers/rl_cmpc.py)

Constrained MPC with critic-based safety/robustness shaping

Enforces input/state bounds via SLSQP bounds and constraints

AMPC (controllers/ampc.py)

Online recursive LSQ to adapt A, B (or nonlinear basis) each step

Robust to slow drifts and mild unmodeled dynamics

Afty-MPC (controllers/attention_mpc.py)

Adds attention-based sequence penalty to the cost functional

Encourages trajectory-level structure and smoothness

Modeling: LSQ and Neural-LSQ
Use core/lsq_modeling.py:

LinearLSQModel: standard least-squares (+ Ridge/Lasso)

NonlinearLSQModel: polynomial + interaction features

SystemIdentificationLSQ: fit A, B (+ bias) from rollouts

NeuralLSQModel: PyTorch MLP with LSQ pretraining and gradient descent

Example (system ID):

python
from advanced_mpc_control import LSQModel
from advanced_mpc_control.core.lsq_modeling import SystemIdentificationLSQ

sysid = SystemIdentificationLSQ(state_dim=4, input_dim=1, model_type='linear')
info = sysid.fit(states, inputs, next_states)
x_rollout = sysid.simulate(initial_state, input_seq)
Simulation
simulation/ exposes Gymnasium-compatible envs:

python
from advanced_mpc_control import CartPoleEnv, HalfCheetahEnv

env = CartPoleEnv()        # fast CPU classic control
env = HalfCheetahEnv()     # MuJoCo locomotion; requires mujoco installed
Recording:

python
from advanced_mpc_control import create_gif, create_video
# after running an episode with env.frame_buffer filled:
create_gif("episode.gif", frames=env.frame_buffer, fps=30)
create_video("episode.mp4", frames=env.frame_buffer, fps=30)
Plotting
python
from advanced_mpc_control import plot_trajectory, plot_control_inputs, TrainingPlotter

plot_trajectory(states, labels=["x", "xdot", "theta", "thetadot"])
plot_control_inputs(actions, labels=["force"])
TrainingPlotter().plot_losses(train_losses, val_losses)
Examples
examples/basic_usage.py: LSQ fit + simple MPC loop

examples/simulation_demo.py: Full controller running in CartPole/HalfCheetah

examples/colab_setup.py: Installs MuJoCo + Gymnasium + sets up codecs on Colab

Run:

bash
python -m advanced_mpc_control.examples.simulation_demo
LaTeX Documentation
A full derivation and API reference is provided at docs/advanced_mpc_control.tex:

NMPC objective and constraints

RL integration (policy warm-start, critic penalty)

AMPC parameter updates via least squares

Attention augmentation term and implementation details

UML diagrams of module and class relationships

Compile:

bash
cd docs
latexmk -pdf advanced_mpc_control.tex
Configuration
Top-level config:

python
from advanced_mpc_control import get_config, set_config

cfg = get_config()
set_config(device='cuda', default_dtype='float32')
Graceful degradation:

If PyTorch is missing, neural features are disabled with warnings.

If MuJoCo/Gymnasium is missing, simulation-dependent demos are skipped.

Testing
Run unit tests:

bash
pytest -q
Covers:

LSQ fitting accuracy and stability

Base MPC optimization step and constraints

Controller smoke tests

Plotting and video utils

Roadmap
CasADi/QP backends (optional) for faster constrained solves

Tube MPC and distributionally robust shaping

Multi-environment benchmark harness and leaderboards

TorchScript export for controllers

Contributing
Fork, create a feature branch, add tests, open a PR

Follow PEP8; run ruff and pytest before submitting

Add docstrings and examples for any public API changes

License
MIT License. See LICENSE.

Citation
If this library is useful in research, consider citing the project (BibTeX template to be added).

Contact
Issues and feature requests via the repository issue tracker.
