"""Federated learning client implementation using Flower.

Each client represents a vehicle/node in the fleet with its own
local dataset. Supports both standard training and DP-SGD training
through Opacus.

Design: Client-side DP during local training
    Local data → Local DP-SGD → Private local update →
    Federated aggregation → Global model
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from numpy.typing import NDArray
from opacus.accountants import RDPAccountant
from torch.utils.data import DataLoader

from fleet_dp.dp import attach_privacy_engine
from fleet_dp.model import create_model

logger = logging.getLogger(__name__)


class FleetClient(NumPyClient):
    """Flower NumPyClient for a fleet vehicle node.

    Supports standard SGD training and DP-SGD training via Opacus.

    Args:
        client_id: Unique identifier for this client/vehicle.
        train_loader: DataLoader for local training data.
        model_name: Name of the model architecture.
        device: Torch device to use.
        local_epochs: Number of local training epochs per round.
        learning_rate: Learning rate for the optimizer.
        dp_enabled: Whether to use differential privacy.
        target_epsilon: Target privacy budget (if DP enabled).
        target_delta: Privacy delta parameter.
        max_grad_norm: Maximum gradient norm for DP clipping.
        noise_multiplier: Explicit noise multiplier (auto-calibrated if None).
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        model_name: str = "mnist_cnn",
        device: str = "cpu",
        local_epochs: int = 1,
        learning_rate: float = 0.01,
        dp_enabled: bool = False,
        target_epsilon: Optional[float] = None,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        clipping_only: bool = False,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.train_loader = train_loader
        self.model_name = model_name
        self.device = torch.device(device)
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dp_enabled = dp_enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.clipping_only = clipping_only

        # These are set fresh each round
        self.model: Optional[nn.Module] = None
        self.privacy_engine = None
        self._current_epsilon: Optional[float] = None
        self._accountant = RDPAccountant()
        self._accounted_steps = 0
        self._last_round = 0

    def get_parameters(self, config: dict[str, Any]) -> list[NDArray]:
        """Return model parameters as a list of NumPy arrays."""
        if self.model is None:
            self.model = create_model(self.model_name).to(self.device)
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: list[NDArray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        model = create_model(self.model_name).to(self.device)
        keys = list(model.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        model.load_state_dict(state_dict, strict=True)
        self.model = model

    def fit(
        self,
        parameters: list[NDArray],
        config: dict[str, Any],
    ) -> tuple[list[NDArray], int, dict[str, Any]]:
        """Train the model on local data.

        Args:
            parameters: Global model parameters from the server.
            config: Configuration from the server.

        Returns:
            Tuple of (updated parameters, num samples, metrics dict).
        """
        self.set_parameters(parameters)

        # Set up optimizer and training
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )

        train_loader = self.train_loader
        self.privacy_engine = None
        self._current_epsilon = None
        round_num = int(config.get("server_round", self._last_round + 1))
        self._last_round = round_num
        noise_norms: list[float] = []
        clipped_grad_norms: list[float] = []
        preclip_sample_norms: list[float] = []
        clipped_sample_norms: list[float] = []

        if self.dp_enabled or self.clipping_only:
            # Opacus handles per-sample clipping for both DP-SGD and the
            # clipping-only control; the latter uses exactly zero noise.
            self.model, optimizer, train_loader, self.privacy_engine = (
                attach_privacy_engine(
                    model=self.model,
                    optimizer=optimizer,
                    data_loader=self.train_loader,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    epochs=self.local_epochs,
                    max_grad_norm=self.max_grad_norm,
                    noise_multiplier=(
                        0.0 if self.clipping_only else self.noise_multiplier
                    ),
                )
            )
            if not self.clipping_only:
                self.noise_multiplier = float(optimizer.noise_multiplier)
                sample_rate = getattr(train_loader, "sample_rate", None)
                if sample_rate is None:
                    dataset_size = max(len(self.train_loader.dataset), 1)
                    sample_rate = min(
                        1.0,
                        (self.train_loader.batch_size or 1) / dataset_size,
                    )
                steps_per_round = len(train_loader) * self.local_epochs
                expected_prior_steps = max(0, round_num - 1) * steps_per_round
                while self._accounted_steps < expected_prior_steps:
                    self._accountant.step(
                        noise_multiplier=self.noise_multiplier,
                        sample_rate=sample_rate,
                    )
                    self._accounted_steps += 1

            original_clip = optimizer.clip_and_accumulate

            def measure_clipping() -> None:
                grad_samples = optimizer.grad_samples
                if grad_samples and len(grad_samples[0]) > 0:
                    per_parameter_norms = [
                        grad.reshape(len(grad), -1).norm(2, dim=1)
                        for grad in grad_samples
                    ]
                    sample_norms = torch.stack(
                        per_parameter_norms, dim=1
                    ).norm(2, dim=1)
                    preclip_sample_norms.append(
                        float(sample_norms.mean().item())
                    )
                    clipped_sample_norms.append(
                        float(
                            sample_norms.clamp(max=self.max_grad_norm)
                            .mean()
                            .item()
                        )
                    )
                original_clip()
                clipped_grad_norms.append(
                    float(
                        torch.sqrt(
                            sum(
                                parameter.summed_grad.float().pow(2).sum()
                                for parameter in optimizer.params
                            )
                        ).item()
                    )
                )

            optimizer.clip_and_accumulate = measure_clipping
            original_add_noise = optimizer.add_noise

            def measure_noise() -> None:
                clipped_grads = [
                    parameter.summed_grad.detach().clone()
                    for parameter in optimizer.params
                ]
                original_add_noise()
                noise_norms.append(
                    float(
                        torch.sqrt(
                            sum(
                                (parameter.grad - clipped_grad)
                                .float()
                                .pow(2)
                                .sum()
                                for parameter, clipped_grad in zip(
                                    optimizer.params, clipped_grads
                                )
                            )
                        ).item()
                    )
                )

            optimizer.add_noise = measure_noise

        # Local training
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                if self.dp_enabled and self.privacy_engine is not None:
                    self._accountant.step(
                        noise_multiplier=self.noise_multiplier,
                        sample_rate=sample_rate,
                    )
                    self._accounted_steps += 1

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        dataset_size = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") and self.train_loader.dataset is not None else total

        # Get privacy accounting
        metrics: dict[str, Any] = {
            "client_id": self.client_id,
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "num_samples": dataset_size,
            "processed_samples": total,
            "dp_enabled": self.dp_enabled,
            "clipping_only": self.clipping_only,
            "server_round": round_num,
        }

        if clipped_grad_norms:
            metrics["clipped_grad_norm"] = float(np.mean(clipped_grad_norms))
            metrics["noise_norm"] = float(np.mean(noise_norms))
            metrics["preclip_sample_grad_norm"] = float(
                np.mean(preclip_sample_norms)
            )
            metrics["clipped_sample_grad_norm"] = float(
                np.mean(clipped_sample_norms)
            )

        if self.privacy_engine is not None:
            if self.dp_enabled:
                eps_spent = self._accountant.get_epsilon(
                    delta=self.target_delta
                )
                self._current_epsilon = eps_spent
                metrics["epsilon_spent"] = float(eps_spent)
            metrics["noise_multiplier"] = float(optimizer.noise_multiplier)

        logger.info(
            f"  Vehicle {self.client_id}: loss={avg_loss:.4f}, "
            f"acc={accuracy:.4f}, dataset_samples={dataset_size}"
            + (f", ε_spent={self._current_epsilon:.2f}"
               if self._current_epsilon is not None else "")
            + (
                f", clipped_grad_norm={metrics['clipped_grad_norm']:.4f}, "
                f"noise_norm={metrics['noise_norm']:.4f}"
                if "clipped_grad_norm" in metrics else ""
            )
        )

        # Extract parameters from possibly-wrapped model
        updated_params = self._extract_parameters()

        return updated_params, dataset_size, metrics

    def evaluate(
        self,
        parameters: list[NDArray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the model on local data.

        Args:
            parameters: Global model parameters.
            config: Configuration from the server.

        Returns:
            Tuple of (loss, num samples, metrics dict).
        """
        self.set_parameters(parameters)
        self.model.eval()

        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += loss_fn(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(avg_loss), total, {"accuracy": float(accuracy)}

    def _extract_parameters(self) -> list[NDArray]:
        """Extract parameters from model, handling Opacus wrapper."""
        # Opacus wraps the model in GradSampleModule
        # We need to get the original model's state_dict keys
        if self.privacy_engine is not None:
            # Get the underlying model from Opacus wrapper
            model_to_extract = self.model
            if hasattr(model_to_extract, '_module'):
                model_to_extract = model_to_extract._module

            # Build a reference model to get the expected keys
            ref_model = create_model(self.model_name)
            ref_keys = list(ref_model.state_dict().keys())

            state = model_to_extract.state_dict()
            params = []
            for key in ref_keys:
                if key in state:
                    params.append(state[key].cpu().numpy())
                else:
                    raise KeyError(
                        f"Expected key '{key}' not found in model state. "
                        f"Available: {list(state.keys())}"
                    )
            return params
        else:
            return [
                val.cpu().numpy()
                for _, val in self.model.state_dict().items()
            ]
