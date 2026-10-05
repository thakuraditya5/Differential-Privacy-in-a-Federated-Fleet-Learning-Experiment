"""Unit tests for neural network architectures."""

import pytest
import torch

from fleet_dp.model import Cifar10CNN, MnistCNN, create_model


def test_mnist_cnn_forward():
    model = MnistCNN()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


def test_cifar10_cnn_forward():
    model = Cifar10CNN()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_create_model():
    m1 = create_model("mnist_cnn")
    assert isinstance(m1, MnistCNN)

    m2 = create_model("cifar10_cnn")
    assert isinstance(m2, Cifar10CNN)

    with pytest.raises(ValueError):
        create_model("invalid_model_name")
