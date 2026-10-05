"""Differential privacy utilities using Opacus.

Handles DP-SGD setup, privacy engine attachment, noise multiplier
calibration, and privacy accounting for federated clients.

Design: Client-side DP during local training
    Local data → Local DP-SGD → Private local update →
    Federated aggregation → Global model

This provides record-level DP for local training data. It does NOT
automatically provide client-level DP, secure aggregation, or encryption.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def validate_and_fix_model(model: nn.Module) -> nn.Module:
    """Validate model compatibility with Opacus and fix if possible.

    Opacus requires specific layer types (e.g., no BatchNorm by default).
    This function checks and auto-fixes the model.

    Args:
        model: PyTorch model to validate.

    Returns:
        Opacus-compatible model (may be a modified copy).

    Raises:
        RuntimeError: If model cannot be made compatible.
    """
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info("Model not Opacus-compatible. Attempting to fix...")
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            raise RuntimeError(
                f"Cannot make model Opacus-compatible. Errors: {errors}"
            )
        logger.info("Model fixed for Opacus compatibility.")
    return model


def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    max_grad_norm: float = 1.0,
) -> float:
    """Find noise multiplier that achieves target epsilon via binary search.

    Uses Opacus privacy accounting to determine the noise multiplier
    that yields the target epsilon budget for the given training setup.

    Args:
        target_epsilon: Desired privacy budget epsilon.
        target_delta: Privacy delta parameter.
        sample_rate: Fraction of dataset sampled per step (batch_size / dataset_size).
        epochs: Number of training epochs.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Calibrated noise multiplier.
    """
    from opacus.accountants.utils import get_noise_multiplier

    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )
        logger.info(
            f"Calibrated noise_multiplier={noise_multiplier:.4f} for "
            f"epsilon={target_epsilon}, delta={target_delta}, "
            f"sample_rate={sample_rate:.4f}, epochs={epochs}"
        )
        return noise_multiplier
    except Exception as e:
        logger.warning(
            f"Opacus get_noise_multiplier failed: {e}. "
            f"Falling back to manual binary search."
        )
        return _binary_search_noise_multiplier(
            target_epsilon, target_delta, sample_rate, epochs
        )


def _binary_search_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    low: float = 0.01,
    high: float = 100.0,
    tolerance: float = 0.01,
    max_iterations: int = 100,
) -> float:
    """Binary search for noise multiplier using RDP accounting.

    Args:
        target_epsilon: Desired epsilon.
        target_delta: Privacy delta.
        sample_rate: Batch sampling rate.
        epochs: Number of training epochs.
        low: Lower bound for search.
        high: Upper bound for search.
        tolerance: Acceptable error margin.
        max_iterations: Maximum search iterations.

    Returns:
        Noise multiplier achieving approximately target epsilon.
    """
    from opacus.accountants import RDPAccountant

    steps = int(epochs / sample_rate)

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(
                noise_multiplier=mid,
                sample_rate=sample_rate,
            )
        epsilon = accountant.get_epsilon(delta=target_delta)

        if abs(epsilon - target_epsilon) < tolerance:
            return mid
        elif epsilon > target_epsilon:
            low = mid  # Need more noise
        else:
            high = mid  # Can reduce noise

    logger.warning(
        f"Binary search did not converge. Using noise_multiplier={mid:.4f} "
        f"(achieved epsilon={epsilon:.2f} vs target={target_epsilon})"
    )
    return mid


def attach_privacy_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float = 1.0,
    noise_multiplier: Optional[float] = None,
) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    """Attach Opacus PrivacyEngine to model, optimizer, and dataloader.

    Args:
        model: Neural network model.
        optimizer: PyTorch optimizer.
        data_loader: Training data loader.
        target_epsilon: Target privacy budget.
        target_delta: Privacy delta.
        epochs: Number of local training epochs.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: If provided, use directly; otherwise calibrate.

    Returns:
        Tuple of (private_model, private_optimizer, private_loader, privacy_engine).
    """
    model = validate_and_fix_model(model)

    dataset_size = len(data_loader.dataset)
    batch_size = data_loader.batch_size or 1
    sample_rate = batch_size / dataset_size

    if noise_multiplier is None:
        noise_multiplier = calibrate_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    logger.info(
        f"PrivacyEngine attached: noise_multiplier={noise_multiplier:.4f}, "
        f"max_grad_norm={max_grad_norm}, "
        f"sample_rate={sample_rate:.4f}"
    )

    return model, optimizer, data_loader, privacy_engine


def get_privacy_spent(
    privacy_engine: PrivacyEngine,
    delta: float,
) -> tuple[float, float]:
    """Query the current privacy budget spent.

    Args:
        privacy_engine: Opacus PrivacyEngine instance.
        delta: Privacy delta for epsilon computation.

    Returns:
        Tuple of (epsilon, delta).
    """
    epsilon = privacy_engine.get_epsilon(delta=delta)
    return epsilon, delta
