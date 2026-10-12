"""Configuration management for Fleet DP experiments.

Loads YAML configuration files and provides typed access to experiment
parameters including dataset, model, federated learning, and DP settings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "mnist"
    num_clients: int = 8
    iid: bool = False
    dirichlet_alpha: float = 0.3


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "mnist_cnn"


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_rounds: int = 20
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01


@dataclass
class DPConfig:
    """Differential privacy configuration."""
    enabled: bool = False
    clipping_only: bool = False
    target_epsilon: Optional[float] = None
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None  # Set by calibration if needed
    current_epsilon: Optional[float] = None  # Privacy spent, reported by clients


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    device: str = "cpu"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    output_dir: str = "results"
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from a dictionary."""
        cfg = cls()
        if "seed" in d:
            cfg.seed = d["seed"]
        if "seeds" in d:
            cfg.seeds = d["seeds"]
        if "device" in d:
            cfg.device = d["device"]
        if "output_dir" in d:
            cfg.output_dir = d["output_dir"]
        if "experiment_name" in d:
            cfg.experiment_name = d["experiment_name"]

        if "dataset" in d:
            ds = d["dataset"]
            cfg.dataset = DatasetConfig(
                name=ds.get("name", "mnist"),
                num_clients=ds.get("num_clients", 8),
                iid=ds.get("iid", False),
                dirichlet_alpha=ds.get("dirichlet_alpha", 0.3),
            )

        if "model" in d:
            cfg.model = ModelConfig(name=d["model"].get("name", "mnist_cnn"))

        if "federated" in d:
            fed = d["federated"]
            cfg.federated = FederatedConfig(
                num_rounds=fed.get("num_rounds", 20),
                fraction_fit=fed.get("fraction_fit", 1.0),
                fraction_evaluate=fed.get("fraction_evaluate", 1.0),
                local_epochs=fed.get("local_epochs", 1),
                batch_size=fed.get("batch_size", 64),
                learning_rate=fed.get("learning_rate", 0.01),
            )

        if "dp" in d:
            dp = d["dp"]
            cfg.dp = DPConfig(
                enabled=dp.get("enabled", False),
                clipping_only=dp.get("clipping_only", False),
                target_epsilon=dp.get("target_epsilon"),
                target_delta=dp.get("target_delta", 1e-5),
                max_grad_norm=dp.get("max_grad_norm", 1.0),
                noise_multiplier=dp.get("noise_multiplier"),
                current_epsilon=dp.get("current_epsilon"),
            )

        return cfg

    def record_calibrated_privacy(
        self,
        noise_multiplier: Optional[float] = None,
        current_epsilon: Optional[float] = None,
    ) -> None:
        """Store runtime DP values reported by clients after calibration.

        Opacus calibrates ``noise_multiplier`` on each client. The server
        records the aggregated values so evaluate-time metrics and the
        saved config include the actual privacy parameters used.
        """
        if noise_multiplier is not None:
            self.dp.noise_multiplier = float(noise_multiplier)
        if current_epsilon is not None:
            self.dp.current_epsilon = float(current_epsilon)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return {
            "seed": self.seed,
            "seeds": self.seeds,
            "device": self.device,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "dataset": {
                "name": self.dataset.name,
                "num_clients": self.dataset.num_clients,
                "iid": self.dataset.iid,
                "dirichlet_alpha": self.dataset.dirichlet_alpha,
            },
            "model": {"name": self.model.name},
            "federated": {
                "num_rounds": self.federated.num_rounds,
                "fraction_fit": self.federated.fraction_fit,
                "fraction_evaluate": self.federated.fraction_evaluate,
                "local_epochs": self.federated.local_epochs,
                "batch_size": self.federated.batch_size,
                "learning_rate": self.federated.learning_rate,
            },
            "dp": {
                "enabled": self.dp.enabled,
                "clipping_only": self.dp.clipping_only,
                "target_epsilon": self.dp.target_epsilon,
                "target_delta": self.dp.target_delta,
                "max_grad_norm": self.dp.max_grad_norm,
                "noise_multiplier": self.dp.noise_multiplier,
                "current_epsilon": self.dp.current_epsilon,
            },
        }

    def copy(self) -> "ExperimentConfig":
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)
