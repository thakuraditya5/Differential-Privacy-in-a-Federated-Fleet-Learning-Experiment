"""Unit tests for differential privacy mechanisms using Opacus."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fleet_dp.dp import (
    attach_privacy_engine,
    calibrate_noise_multiplier,
    get_privacy_spent,
    validate_and_fix_model,
)
from fleet_dp.model import MnistCNN


def test_model_opacus_compatibility():
    model = MnistCNN()
    fixed_model = validate_and_fix_model(model)
    assert fixed_model is not None


def test_noise_multiplier_calibration():
    noise = calibrate_noise_multiplier(
        target_epsilon=5.0,
        target_delta=1e-5,
        sample_rate=0.01,
        epochs=1,
    )
    assert noise > 0.0
    assert isinstance(noise, float)


def test_attach_privacy_engine_and_accounting():
    model = MnistCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 100 samples, batch_size=20
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=20, shuffle=True)

    priv_model, priv_opt, priv_loader, engine = attach_privacy_engine(
        model=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=10.0,
        target_delta=1e-5,
        epochs=1,
        max_grad_norm=1.0,
    )

    # Do a training step
    loss_fn = nn.CrossEntropyLoss()
    for batch_x, batch_y in priv_loader:
        priv_opt.zero_grad()
        out = priv_model(batch_x)
        loss = loss_fn(out, batch_y)
        loss.backward()
        priv_opt.step()

    eps, delta = get_privacy_spent(engine, delta=1e-5)
    assert eps > 0.0
    assert delta == 1e-5
