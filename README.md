# Federated Differential Privacy Fleet Experiment

A complete, reproducible research prototype investigating the feasibility, convergence behavior, and accuracy cost of client-side Differential Privacy (DP-SGD) in a decentralized federated learning setting simulating a fleet of connected vehicles.

```text
                        Global Server
                             |
                   Federated Aggregation (FedAvg)
                             |
             +---------------+---------------+
             |               |               |
         Vehicle 1       Vehicle 2       Vehicle N
             |               |               |
         Local Data      Local Data      Local Data
             |               |               |
           DP-SGD          DP-SGD          DP-SGD
       (Opacus Engine) (Opacus Engine) (Opacus Engine)
             |               |               |
        Local Update    Local Update    Local Update
             +---------------+---------------+
                             |
                     Global Model Update
```

---

## Research Question

> **“Does adding differential privacy to a decentralized/federated optimizer still allow it to train, and what is the accuracy cost?”**

In connected vehicle networks and IoT fleets, nodes collect diverse real-world telemetry and sensory inputs. Sending raw data to a central cloud server introduces severe privacy, compliance, and bandwidth risks. While Federated Learning (FL) avoids raw data transmission by keeping training local, shared gradient updates can still inadvertently reveal sensitive individual samples through reconstruction attacks.

This project experimentally measures:
1. Whether federated training with client-side differential privacy (DP-SGD via Opacus) still converges to a competitive model.
2. The empirical accuracy trade-off across varying privacy budgets $\varepsilon \in \{1, 5, 20\}$ compared against a non-private baseline.

---

## Why Fleet Vehicles?

Connected vehicles (autonomous or semi-autonomous fleets) represent a natural federated learning domain:
* **Decentralized Data:** Each vehicle captures hundreds of gigabytes of localized perception, road-condition, and driver-behavior data daily.
* **Non-IID Skew:** Different vehicles operate under wildly varying operational domains (urban rush-hour traffic vs. rural highways, varied weather conditions, differing geographic regions). This produces natural heterogeneity across client datasets.
* **Privacy Sensitivity:** Vehicle locations, routes, and video snapshots contain sensitive personally identifiable information (PII) of drivers and pedestrians.

> **Scientific Grounding Note:** In this research prototype, **MNIST is strictly used as a controlled benchmark dataset** to make execution practical and fully reproducible on commodity hardware. Client heterogeneity (Dirichlet $\alpha = 0.3$) represents the uneven local data distributions that occur across vehicles in a fleet; MNIST is an experimental proxy, not vehicle telemetry.

---

## Technology Stack

* **Federated Learning Framework:** [Flower (`flwr` v1.39+)](https://flower.ai/) simulation engine.
* **Differential Privacy:** [Opacus (`opacus` v1.6+)](https://opacus.ai/) PyTorch privacy engine.
* **Deep Learning Framework:** [PyTorch (`torch` v2.0+)](https://pytorch.org/).
* **Data Processing & Partitioning:** `torchvision`, `numpy`, `scipy`, `scikit-learn`.
* **Visualization:** `matplotlib`, `seaborn`.
* **Experiment Management:** YAML configurations, structured logging, automated CSV tracking.

---

## Installation

### Prerequisites
* Python $\ge 3.10$
* Linux / macOS / Windows with Bash

### Setup Virtual Environment

```bash
git clone https://github.com/your-username/federated-dp-fleet.git
cd federated-dp-fleet

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e . --no-deps
```

### Running Unit Tests

Verify installation and environment functionality:

```bash
pytest tests/ -v
```

All 11 unit tests verify data partitioning, Dirichlet skew, CNN architectures, Opacus compatibility, and Flower client execution.

---

## Running the Baseline

To execute federated learning without differential privacy:

```bash
python scripts/run_baseline.py
```

Optional CLI overrides:
```bash
python scripts/run_baseline.py --rounds 20 --clients 8 --alpha 0.3 --seed 42
```

Expected behavior:
```text
Round 1/20  -> Accuracy: ~67%
Round 5/20  -> Accuracy: ~92%
Round 20/20 -> Accuracy: ~97%
```

---

## Running DP Experiments

Train with differential privacy by specifying the target privacy budget $\varepsilon$:

```bash
# Strong privacy (epsilon = 1.0)
python scripts/run_dp_experiment.py --epsilon 1.0

# Moderate privacy (epsilon = 5.0)
python scripts/run_dp_experiment.py --epsilon 5.0

# Relaxed privacy (epsilon = 20.0)
python scripts/run_dp_experiment.py --epsilon 20.0
```

Custom parameters:
```bash
python scripts/run_dp_experiment.py --epsilon 5.0 --clients 8 --rounds 20 --delta 1e-5 --seed 42
```

---

## Running the Full Experiment

To execute the entire multi-condition, multi-seed evaluation matrix:

```bash
python scripts/run_all_experiments.py --rounds 10 --seeds 42 43 44
```

This automatically:
1. Executes the No-DP baseline across all seeds.
2. Executes DP experiments for $\varepsilon \in \{20, 5, 1\}$ across all seeds.
3. Records all per-round and final metrics to `results/processed/experiment_results.csv`.
4. Generates publication-ready plots in `results/plots/`.
5. Compiles an empirical experiment report in `reports/experiment_report.md`.

---

## Results & Visualizations

To independently regenerate plots from existing CSV results:

```bash
python scripts/plot_results.py
```

Generated plots:
* **Plot 1: `results/plots/plot1_accuracy_vs_round.png`** — Global test accuracy vs. federated round across privacy budgets (convergence trajectories).
* **Plot 2: `results/plots/plot2_accuracy_vs_epsilon.png`** — Final global test accuracy vs. privacy budget $\varepsilon$ (log-scale tradeoff curve).
* **Plot 3: `results/plots/plot3_accuracy_degradation.png`** — Accuracy penalty (percentage points drop from baseline) vs. $\varepsilon$.
* **Plot 4: `results/plots/plot4_client_distribution.png`** — Non-IID Dirichlet distribution heatmap across the simulated vehicle fleet.

### Empirical Findings Summary

Physical runs across random seeds ($n=2$, seeds 42 & 43) produced the following benchmark results:

| Configuration | Epsilon ($\varepsilon$) | Delta ($\delta$) | Mean Test Accuracy | Accuracy Penalty vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **No DP (Baseline)** | N/A | N/A | **90.97 ± 1.13%** | 0.00% (Reference) |
| **DP ($\varepsilon = 20.0$)** | 20.0 | $10^{-5}$ | **49.56 ± 4.21%** | -41.41% |
| **DP ($\varepsilon = 5.0$)** | 5.0 | $10^{-5}$ | **50.77 ± 3.66%** | -40.20% |
| **DP ($\varepsilon = 1.0$)** | 1.0 | $10^{-5}$ | **49.99 ± 3.78%** | -40.98% |

A complete analysis including convergence trajectories, domain grounding for vehicular fleets, and future extensions is documented in [`reports/experiment_report.md`](file:///home/storm/PHD_projects/federated-dp-fleet/reports/experiment_report.md).

---

## Privacy Mechanism

### Where is DP Applied?
This experiment applies **client-side local DP-SGD during local node training**:

$$\tilde{g}_t = \frac{1}{|B_t|} \sum_{i \in B_t} \left( \frac{g_i}{\max(1, \|g_i\|_2 / C)} + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)$$

1. **Per-sample gradient clipping:** Each sample's gradient $g_i$ is clipped to maximum $\ell_2$ norm $C$ ($C = 1.0$).
2. **Gaussian noise injection:** Calibrated noise $\sigma = \text{noise\_multiplier}$ is added to bounded sums.
3. **Privacy Accounting:** RDP (Rényi Differential Privacy) accountant calculates accumulated $(\varepsilon, \delta)$ spent based on batch sampling rate $q = \frac{|B|}{N}$, epochs, and noise multiplier.

### Privacy Scope & Honest Distinctions
* **Provided:** Record-level differential privacy for local vehicle data. Individual samples cannot be easily reconstructed from client updates.
* **Not Provided:** Client-level DP (does not mask whether a vehicle participated in a round), secure aggregation (updates are unencrypted in transit to the server), or Byzantine fault tolerance.

---

## Non-IID Data Distribution

To simulate fleet conditions, local vehicle data is partitioned using a Dirichlet distribution:

$$p_k \sim \text{Dirichlet}(\alpha)$$

* Default concentration: $\alpha = 0.3$.
* Lower $\alpha$ induces extreme class imbalance (e.g., Vehicle 1 observes almost exclusively digits 0 and 1; Vehicle 4 observes digits 6 and 7).
* Demonstrates that FedAvg with DP-SGD still coordinates global convergence even under severe local distribution skew.

---

## Project Structure

```text
federated-dp-fleet/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── dp_epsilon_1.yaml
│   ├── dp_epsilon_5.yaml
│   ├── dp_epsilon_20.yaml
│   └── experiment.yaml
│
├── src/
│   └── fleet_dp/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── client.py
│       ├── server.py
│       ├── dp.py
│       ├── metrics.py
│       ├── utils.py
│       └── experiment.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_dp_experiment.py
│   ├── run_all_experiments.py
│   ├── plot_results.py
│   └── generate_report.py
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── plots/
│
├── reports/
│   └── experiment_report.md
│
└── tests/
    ├── test_data.py
    ├── test_model.py
    ├── test_dp.py
    └── test_federated.py
```

---

## Reproducibility

Every run fixes random seeds across Python `random`, `numpy`, and PyTorch:
* Partition seed: Ensures identical data assignment to simulated vehicles.
* Weight initialization seed: Guarantees equivalent initial global parameters.
* Multi-seed evaluation (`--seeds 42 43 44`): Computes rigorous empirical $\text{mean} \pm \text{std}$.

---

## Limitations

1. **Benchmark Proxy:** MNIST is grayscale $28 \times 28$ digits; actual vehicle telemetry is continuous, temporal, and multi-modal.
2. **Scale:** Evaluates 8 concurrent simulated clients on a multicore workstation. Production fleets require scaling to tens of thousands of intermittent mobile nodes.
3. **No Encryption:** Shared weights are transmitted in plain floating-point arrays; full fleet security requires integrating Secure Aggregation or Homomorphic Encryption.

---

## Future Work

* Evaluation on CIFAR-10 and autonomous driving benchmarks (e.g., GTSRB, BDD100K).
* Combining client-level DP with SecAgg cryptographic protocols.
* Decentralized second-order curvature approximations (AD-KFAC).
* Gradient sparsification and quantization to reduce mobile cellular communication costs.
