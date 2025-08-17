"""
Adaptive Model Predictive Control (AMPC)

This module implements adaptive MPC that continuously updates its internal model
based on real-time system identification and parameter estimation.

Mathematical Foundation:
AMPC combines MPC with recursive parameter estimation:

System Model:
x(k+1) = f(x(k), u(k), θ(k)) + w(k)

where θ(k) are time-varying parameters estimated online using:
θ̂(k) = θ̂(k-1) + Γ(k)[y(k) - ŷ(k|k-1)]

MPC Optimization (solved at each time step):
min  Σᵢ₌₀^{N-1} [x(i)ᵀQx(i) + u(i)ᵀRu(i)] + x(N)ᵀPx(N)
u    
s.t. x(i+1) = f(x(i), u(i), θ̂(k))  ∀i = 0,...,N-1
     u_min ≤ u(i) ≤ u_max           ∀i = 0,...,N-1
     x(0) = x_current

The key innovation is that θ̂(k) is updated in real-time based on model prediction errors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import scipy.optimize
from scipy.linalg import cholesky, solve_triangular
import warnings

from lsq_model import LSQModel, LSQConfig


@dataclass 
class AMPCConfig:
    """Configuration for Adaptive MPC controller."""
    state_dim: int
    action_dim: int
    horizon: int = 10
    dt: float = 0.1

    # MPC cost matrices
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None  
    P: Optional[np.ndarray] = None

    # Constraints
    action_bounds: Optional[Tuple[float, float]] = None
    state_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # Adaptation parameters
    adaptation_rate: float = 0.01
    forgetting_factor: float = 0.95
    minimum_excitation: float = 1e-6
    parameter_bounds: Optional[Tuple[float, float]] = None

    # Optimization parameters
    max_iterations: int = 100
    tolerance: float = 1e-4
    solver_method: str = "SLSQP"

    # Robustness parameters
    uncertainty_weight: float = 0.1
    constraint_tightening: float = 0.05


class AMPC:
    """
    Adaptive Model Predictive Control.

    This controller adapts its internal model in real-time using recursive 
    parameter estimation while solving the MPC optimization problem.

    Attributes:
        config (AMPCConfig): Controller configuration
        model (LSQModel): Adaptive system model
        parameter_history (List): History of parameter estimates
        covariance_history (List): History of parameter covariances
        adaptation_gains (np.ndarray): Current adaptation gains
    """

    def __init__(
        self,
        config: Union[AMPCConfig, dict], 
        model: Optional[LSQModel] = None
    ):
        if isinstance(config, dict):
            config = AMPCConfig(**config)

        self.config = config

        # Initialize cost matrices
        if config.Q is None:
            self.Q = np.eye(config.state_dim)
        else:
            self.Q = config.Q

        if config.R is None:
            self.R = np.eye(config.action_dim) 
        else:
            self.R = config.R

        if config.P is None:
            self.P = self.Q.copy()
        else:
            self.P = config.P

        # Initialize model
        if model is None:
            model_config = LSQConfig(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                forgetting_factor=config.forgetting_factor,
                use_neural_augmentation=False  # AMPC focuses on parametric adaptation
            )
            self.model = LSQModel(model_config)
        else:
            self.model = model

        # Adaptation tracking
        self.parameter_history = []
        self.covariance_history = []
        self.prediction_errors = []
        self.adaptation_gains = np.ones(config.state_dim + config.action_dim)

        # Performance monitoring
        self.step_count = 0
        self.mpc_solve_times = []
        self.adaptation_stats = {
            "parameter_updates": [],
            "prediction_errors": [],
            "covariance_trace": [],
            "control_effort": []
        }

    def adapt(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray
    ) -> Dict[str, Any]:
        """
        Adapt the internal model based on observed system behavior.

        Args:
            state: Current state observation
            action: Applied control action
            next_state: Observed next state

        Returns:
            Dictionary with adaptation statistics
        """
        # Update model with new data
        model_stats = self.model.update(state, action, next_state)

        # Compute prediction error
        predicted_next_state, pred_info = self.model.predict(state, action)
        prediction_error = next_state - predicted_next_state
        error_magnitude = np.linalg.norm(prediction_error)

        # Adaptive gain adjustment based on prediction performance
        if error_magnitude > 0:
            # Increase adaptation rate if prediction error is large
            error_factor = min(2.0, 1.0 + error_magnitude / np.linalg.norm(next_state))
            self.adaptation_gains *= (1.0 + self.config.adaptation_rate * error_factor)
        else:
            # Decrease adaptation rate if prediction is accurate
            self.adaptation_gains *= (1.0 - 0.1 * self.config.adaptation_rate)

        # Ensure gains stay within reasonable bounds
        self.adaptation_gains = np.clip(self.adaptation_gains, 0.01, 10.0)

        # Store parameter estimates and covariance
        A, B = self.model.get_linearization(state)
        self.parameter_history.append(np.concatenate([A.flatten(), B.flatten()]))
        self.covariance_history.append(np.trace(self.model.P))

        # Update statistics
        self.adaptation_stats["parameter_updates"].append(model_stats["update_count"])
        self.adaptation_stats["prediction_errors"].append(error_magnitude)
        self.adaptation_stats["covariance_trace"].append(np.trace(self.model.P))

        return {
            "prediction_error": error_magnitude,
            "model_confidence": pred_info["confidence"],
            "adaptation_gains": self.adaptation_gains.copy(),
            "parameter_change": np.linalg.norm(
                self.parameter_history[-1] - self.parameter_history[-2] 
                if len(self.parameter_history) > 1 else np.zeros_like(self.parameter_history[-1])
            )
        }

    def step(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control action using adaptive MPC.

        Args:
            state: Current system state

        Returns:
            Optimal control action
        """
        state = np.atleast_1d(state)

        # Solve MPC optimization with current model
        action, solve_info = self._solve_mpc(state)

        # Store control effort statistic
        control_effort = np.linalg.norm(action)
        self.adaptation_stats["control_effort"].append(control_effort)

        self.step_count += 1
        return action

    def _solve_mpc(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the MPC optimization problem.

        Args:
            state: Current state

        Returns:
            Tuple of (optimal_action, solve_info)
        """
        horizon = self.config.horizon
        action_dim = self.config.action_dim

        # Get current model linearization
        A, B = self.model.get_linearization(state)

        # Initial guess (zero control sequence)
        u0 = np.zeros(horizon * action_dim)

        # Set up bounds
        bounds = None
        if self.config.action_bounds is not None:
            low, high = self.config.action_bounds
            bounds = [(low, high)] * (horizon * action_dim)

        # Robust MPC cost function with uncertainty compensation
        def robust_cost_function(u_flat: np.ndarray) -> float:
            u_seq = u_flat.reshape(horizon, action_dim)

            # Forward simulation
            x = state.copy()
            total_cost = 0.0
            uncertainty_penalty = 0.0

            for i in range(horizon):
                # Stage cost
                stage_cost = x.T @ self.Q @ x + u_seq[i].T @ self.R @ u_seq[i]
                total_cost += stage_cost

                # Uncertainty penalty based on parameter covariance
                if len(self.covariance_history) > 0:
                    uncertainty = self.covariance_history[-1]
                    uncertainty_penalty += self.config.uncertainty_weight * uncertainty * np.linalg.norm(x)**2

                # Predict next state
                x_next, pred_info = self.model.predict(x, u_seq[i])

                # Add prediction uncertainty
                if "prediction_variance" in pred_info:
                    pred_uncertainty = np.sum(pred_info["prediction_variance"])
                    uncertainty_penalty += self.config.uncertainty_weight * pred_uncertainty

                x = x_next

            # Terminal cost
            terminal_cost = x.T @ self.P @ x
            total_cost += terminal_cost + uncertainty_penalty

            return total_cost

        # Add constraint function for state bounds (if specified)
        constraints = []
        if self.config.state_bounds is not None:
            def state_constraint(u_flat: np.ndarray) -> np.ndarray:
                u_seq = u_flat.reshape(horizon, action_dim)
                violations = []

                x = state.copy()
                for i in range(horizon):
                    # Check state bounds
                    for state_idx, (low, high) in self.config.state_bounds.items():
                        idx = int(state_idx) if isinstance(state_idx, str) and state_idx.isdigit() else state_idx
                        if isinstance(idx, int) and 0 <= idx < len(x):
                            # Constraint: low <= x[idx] <= high
                            violations.extend([x[idx] - low, high - x[idx]])

                    # Predict next state
                    x, _ = self.model.predict(x, u_seq[i])

                return np.array(violations)

            constraints.append({
                "type": "ineq",
                "fun": state_constraint
            })

        # Solve optimization
        start_time = time.time() if 'time' in globals() else 0

        try:
            result = scipy.optimize.minimize(
                robust_cost_function,
                u0,
                method=self.config.solver_method,
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.tolerance
                }
            )

            if result.success:
                optimal_u = result.x.reshape(horizon, action_dim)
                success = True
                cost = result.fun
            else:
                warnings.warn(f"MPC optimization failed: {result.message}")
                optimal_u = u0.reshape(horizon, action_dim)
                success = False
                cost = float("inf")

        except Exception as e:
            warnings.warn(f"MPC optimization exception: {e}")
            optimal_u = u0.reshape(horizon, action_dim)
            success = False
            cost = float("inf")

        solve_time = time.time() - start_time if 'time' in globals() else 0
        self.mpc_solve_times.append(solve_time)

        solve_info = {
            "success": success,
            "cost": cost,
            "solve_time": solve_time,
            "iterations": getattr(result, "nit", 0) if 'result' in locals() else 0
        }

        return optimal_u[0], solve_info

    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        """Get current model parameters."""
        A, B = self.model.get_linearization(np.zeros(self.config.state_dim))
        return {
            "A_matrix": A,
            "B_matrix": B,
            "parameter_vector": np.concatenate([A.flatten(), B.flatten()]),
            "covariance_matrix": self.model.P
        }

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        return {
            "step_count": self.step_count,
            "parameter_history": self.parameter_history.copy(),
            "covariance_history": self.covariance_history.copy(),
            "adaptation_gains": self.adaptation_gains.copy(),
            "avg_solve_time": np.mean(self.mpc_solve_times) if self.mpc_solve_times else 0.0,
            "prediction_error_mean": np.mean(self.adaptation_stats["prediction_errors"]) if self.adaptation_stats["prediction_errors"] else 0.0,
            "parameter_stability": np.std(self.parameter_history, axis=0) if len(self.parameter_history) > 1 else np.zeros(self.config.state_dim * (self.config.state_dim + self.config.action_dim))
        }

    def reset_adaptation(self) -> None:
        """Reset adaptation history and parameters."""
        self.model.reset()
        self.parameter_history = []
        self.covariance_history = []
        self.adaptation_gains = np.ones(self.config.state_dim + self.config.action_dim)
        self.step_count = 0

        # Reset statistics
        self.adaptation_stats = {
            "parameter_updates": [],
            "prediction_errors": [],
            "covariance_trace": [],
            "control_effort": []
        }

    def save(self, filepath: str) -> None:
        """Save controller state."""
        save_dict = {
            "config": self.config,
            "parameter_history": self.parameter_history,
            "covariance_history": self.covariance_history,
            "adaptation_gains": self.adaptation_gains,
            "step_count": self.step_count,
            "adaptation_stats": self.adaptation_stats
        }

        # Save main data
        np.savez(filepath, **save_dict)

        # Save model separately
        model_path = filepath.replace(".npz", "_model.npz")
        self.model.save_model(model_path)

    def load(self, filepath: str) -> None:
        """Load controller state."""
        data = np.load(filepath, allow_pickle=True)

        # Load main data
        self.parameter_history = data["parameter_history"].tolist()
        self.covariance_history = data["covariance_history"].tolist()
        self.adaptation_gains = data["adaptation_gains"]
        self.step_count = int(data["step_count"])
        self.adaptation_stats = data["adaptation_stats"].item()

        # Load model
        model_path = filepath.replace(".npz", "_model.npz")
        try:
            self.model.load_model(model_path)
        except FileNotFoundError:
            print(f"Model file {model_path} not found, using default model")
