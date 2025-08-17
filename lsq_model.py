"""
Least Squares Predictive Model for Advanced Control Systems

This module implements LSQ-based predictive models that can be optionally 
augmented with neural networks for enhanced dynamics modeling.

Mathematical Foundation:
The LSQ model assumes a linear relationship of the form:
x(k+1) = Ax(k) + Bu(k) + w(k)

where:
- x(k) ∈ ℝⁿ is the state vector at time k
- u(k) ∈ ℝᵐ is the control input at time k  
- A ∈ ℝⁿˣⁿ is the state transition matrix
- B ∈ ℝⁿˣᵐ is the input matrix
- w(k) ∈ ℝⁿ is the process noise

The parameters θ = [A, B] are estimated using recursive least squares:
θ̂(k) = θ̂(k-1) + P(k)φ(k)[y(k) - φᵀ(k)θ̂(k-1)]

where P(k) is the covariance matrix and φ(k) is the regressor vector.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from scipy.linalg import pinv


@dataclass
class LSQConfig:
    """Configuration for LSQ model parameters."""
    state_dim: int
    action_dim: int
    memory_length: int = 50
    forgetting_factor: float = 0.99
    regularization: float = 1e-6
    use_neural_augmentation: bool = False
    neural_hidden_dims: Tuple[int, ...] = (64, 32)
    neural_activation: str = "relu"


class NeuralAugmentation(nn.Module):
    """Neural network for augmenting LSQ predictions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        activation: str = "relu"
    ):
        super().__init__()

        # Build network architecture
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output layer
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSQModel:
    """
    Least Squares Predictive Model with optional neural network augmentation.

    This class implements a recursive least squares estimator for system
    identification, with optional neural network residual learning.

    Attributes:
        config (LSQConfig): Model configuration
        A (np.ndarray): State transition matrix estimate
        B (np.ndarray): Input matrix estimate  
        P (np.ndarray): Covariance matrix
        data_buffer (list): Historical data for training
        neural_net (Optional[NeuralAugmentation]): Neural augmentation network
    """

    def __init__(self, config: Union[LSQConfig, dict]):
        if isinstance(config, dict):
            config = LSQConfig(**config)

        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        # Initialize parameter estimates
        self.A = np.eye(self.state_dim) * 0.95  # Stable initial guess
        self.B = np.random.normal(0, 0.1, (self.state_dim, self.action_dim))

        # Initialize covariance matrix
        param_dim = self.state_dim * (self.state_dim + self.action_dim)
        self.P = np.eye(param_dim) / config.regularization

        # Data buffer for batch updates
        self.data_buffer = []

        # Neural network augmentation
        self.neural_net: Optional[NeuralAugmentation] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        if config.use_neural_augmentation:
            input_dim = self.state_dim + self.action_dim
            self.neural_net = NeuralAugmentation(
                input_dim=input_dim,
                output_dim=self.state_dim,
                hidden_dims=config.neural_hidden_dims,
                activation=config.neural_activation
            )
            self.optimizer = torch.optim.Adam(
                self.neural_net.parameters(), lr=0.001
            )

        # Statistics tracking
        self.update_count = 0
        self.prediction_errors = []

    def update(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray
    ) -> Dict[str, float]:
        """
        Update model parameters using recursive least squares.

        Args:
            state: Current state vector (n,)
            action: Applied action vector (m,)
            next_state: Resulting next state vector (n,)

        Returns:
            Dictionary with update statistics
        """
        # Ensure proper shapes
        state = np.atleast_1d(state)
        action = np.atleast_1d(action) 
        next_state = np.atleast_1d(next_state)

        # Store data in buffer
        self.data_buffer.append((state, action, next_state))
        if len(self.data_buffer) > self.config.memory_length:
            self.data_buffer.pop(0)

        # Build regressor vector φ(k) = [x(k)ᵀ, u(k)ᵀ]ᵀ
        phi = np.concatenate([state, action])

        # Prediction with current parameters
        predicted = self.A @ state + self.B @ action
        prediction_error = next_state - predicted

        # Recursive least squares update
        self._rls_update(phi, next_state, prediction_error)

        # Neural network residual learning
        neural_loss = 0.0
        if self.neural_net is not None:
            neural_loss = self._update_neural_network()

        # Track statistics
        mse = np.mean(prediction_error**2)
        self.prediction_errors.append(mse)
        if len(self.prediction_errors) > 1000:
            self.prediction_errors.pop(0)

        self.update_count += 1

        return {
            "prediction_mse": mse,
            "neural_loss": neural_loss,
            "update_count": self.update_count,
            "condition_number": np.linalg.cond(self.P)
        }

    def _rls_update(
        self, 
        phi: np.ndarray, 
        y: np.ndarray, 
        error: np.ndarray
    ) -> None:
        """Perform recursive least squares parameter update."""
        # Forgetting factor
        lambda_f = self.config.forgetting_factor

        # Build full regressor for vectorized parameters
        # For x(k+1) = Ax(k) + Bu(k), we need Kronecker products
        Phi = np.kron(phi.T, np.eye(self.state_dim))

        # Vectorized parameters: θ = vec([A, B])  
        theta_vec = np.concatenate([self.A.flatten(), self.B.flatten()])

        # RLS update equations
        P_phi = self.P @ Phi
        denominator = lambda_f + Phi.T @ P_phi

        if np.abs(denominator) > 1e-10:  # Avoid division by zero
            K = P_phi / denominator  # Kalman gain

            # Parameter update
            innovation = y - Phi.T @ theta_vec
            theta_vec += K @ innovation

            # Covariance update (Joseph form for numerical stability)
            I_KPhi = np.eye(len(theta_vec)) - np.outer(K, Phi)
            self.P = (I_KPhi @ self.P @ I_KPhi.T) / lambda_f + \
                     np.outer(K, K) * self.config.regularization

        # Reshape back to matrices
        n_A_params = self.state_dim * self.state_dim
        self.A = theta_vec[:n_A_params].reshape(self.state_dim, self.state_dim)
        self.B = theta_vec[n_A_params:].reshape(self.state_dim, self.action_dim)

    def _update_neural_network(self) -> float:
        """Update neural network using recent data."""
        if len(self.data_buffer) < 10:
            return 0.0

        # Prepare training data
        states, actions, next_states = zip(*self.data_buffer[-20:])  # Use recent data

        X = np.array([np.concatenate([s, a]) for s, a in zip(states, actions)])
        Y_lsq = np.array([self.A @ s + self.B @ a for s, a in zip(states, actions)])
        Y_true = np.array(next_states)

        # Residual learning: neural network predicts Y_true - Y_lsq
        residuals = Y_true - Y_lsq

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        residual_tensor = torch.FloatTensor(residuals)

        # Training step
        self.optimizer.zero_grad()
        predicted_residuals = self.neural_net(X_tensor)
        loss = nn.MSELoss()(predicted_residuals, residual_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(
        self, 
        state: np.ndarray, 
        action: np.ndarray,
        use_neural_augmentation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict next state given current state and action.

        Args:
            state: Current state vector
            action: Action to apply
            use_neural_augmentation: Whether to use neural network correction

        Returns:
            Tuple of (predicted_next_state, prediction_info)
        """
        state = np.atleast_1d(state)
        action = np.atleast_1d(action)

        # LSQ prediction
        lsq_prediction = self.A @ state + self.B @ action

        # Neural augmentation
        neural_correction = np.zeros_like(lsq_prediction)
        if (self.neural_net is not None and 
            use_neural_augmentation and 
            len(self.data_buffer) > 0):

            input_tensor = torch.FloatTensor(np.concatenate([state, action]))
            with torch.no_grad():
                neural_correction = self.neural_net(input_tensor).numpy()

        final_prediction = lsq_prediction + neural_correction

        # Prediction uncertainty (diagonal of covariance)
        phi = np.concatenate([state, action])
        Phi = np.kron(phi.T, np.eye(self.state_dim))
        prediction_var = np.diag(Phi.T @ self.P @ Phi)

        info = {
            "lsq_prediction": lsq_prediction,
            "neural_correction": neural_correction,
            "prediction_variance": prediction_var,
            "confidence": np.exp(-np.mean(prediction_var))
        }

        return final_prediction, info

    def get_linearization(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearized dynamics around a state.

        Args:
            state: State to linearize around

        Returns:
            Tuple of (A_matrix, B_matrix) for linearized dynamics
        """
        # For LSQ model, linearization is just the learned matrices
        return self.A.copy(), self.B.copy()

    def save_model(self, filepath: str) -> None:
        """Save model parameters to file."""
        save_dict = {
            "A": self.A,
            "B": self.B, 
            "P": self.P,
            "config": self.config.__dict__,
            "update_count": self.update_count
        }

        if self.neural_net is not None:
            save_dict["neural_net_state"] = self.neural_net.state_dict()

        np.savez(filepath, **save_dict)

    def load_model(self, filepath: str) -> None:
        """Load model parameters from file."""
        data = np.load(filepath, allow_pickle=True)

        self.A = data["A"]
        self.B = data["B"]
        self.P = data["P"] 
        self.update_count = int(data["update_count"])

        # Reconstruct config
        config_dict = data["config"].item()
        self.config = LSQConfig(**config_dict)

        # Load neural network if present
        if "neural_net_state" in data and self.neural_net is not None:
            self.neural_net.load_state_dict(data["neural_net_state"].item())

    def reset(self) -> None:
        """Reset model to initial state."""
        self.__init__(self.config)
