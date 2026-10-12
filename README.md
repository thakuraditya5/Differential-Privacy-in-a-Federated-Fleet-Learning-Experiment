# Federated Differential Privacy Fleet Experiment

A reproducible research prototype investigating the feasibility, convergence behavior, and accuracy cost of client-side Differential Privacy (DP-SGD) in a federated learning setting that simulates a fleet of connected vehicles.

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

> **"Does adding differential privacy to a decentralized/federated optimizer still allow it to train, and what is the accuracy cost?"**

In connected vehicle networks and IoT fleets, nodes collect diverse real-world telemetry and sensory inputs. Sending raw data to a central cloud server introduces privacy, compliance, and bandwidth risks. Federated Learning (FL) avoids raw data transmission by keeping training local, but shared model updates can still leak information about individual training samples (for example through reconstruction attacks).

This project experimentally measures:

1. Whether federated training with client-side differential privacy (DP-SGD via Opacus) still converges to a competitive model.
2. The empirical accuracy trade-off across target privacy budgets $\varepsilon \in \{0.1, 0.5, 1, 5, 20, 100\}$, compared against a non-private baseline and a clipping-only control that separates the cost of gradient clipping from the cost of added noise.

---

## Why Fleet Vehicles?

Connected vehicles (autonomous or semi-autonomous fleets) are a natural federated learning domain:

* **Decentralized Data:** Each vehicle can capture large volumes of localized perception, road-condition, and driver-behavior data.
* **Non-IID Skew:** Vehicles operate in very different domains (urban vs. rural, varied weather, differing regions), producing heterogeneous client datasets.
* **Privacy Sensitivity:** Vehicle locations, routes, and video snapshots can contain personally identifiable information (PII) of drivers and pedestrians.

> **Scientific Grounding Note:** **MNIST is used strictly as a controlled benchmark** so that execution is practical and reproducible on commodity hardware. Client heterogeneity (Dirichlet $\alpha = 0.3$) is a stand-in for uneven local data distributions across vehicles. MNIST is an experimental proxy, not vehicle telemetry.

---

## Technology Stack

* **Federated Learning Framework:** [Flower (`flwr`)](https://flower.ai/) simulation engine.
* **Differential Privacy:** [Opacus (`opacus`)](https://opacus.ai/) PyTorch privacy engine.
* **Deep Learning Framework:** [PyTorch (`torch`)](https://pytorch.org/).
* **Data Processing & Partitioning:** `torchvision`, `numpy`, `scipy`, `scikit-learn`.
* **Visualization:** `matplotlib`, `seaborn`.
* **Experiment Management:** YAML configurations, structured logging, automated CSV tracking.

Exact version requirements are declared in `pyproject.toml` and `requirements.txt`.

---

## Installation

### Prerequisites

* Python $\ge 3.10$
* Linux / macOS / Windows with Bash

### Setup Virtual Environment

```bash
git clone https://github.com/<your-username>/federated-dp-fleet.git
cd federated-dp-fleet

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e . --no-deps
```

### Running Unit Tests

```bash
pytest tests/ -v
```

The 17 unit tests cover data partitioning, Dirichlet skew, CNN architectures, Opacus compatibility, Flower client execution, cumulative privacy accounting, clipping-only behavior, and CSV metric recording.

---

## Running the Baseline

To run federated learning without differential privacy:

```bash
python scripts/run_baseline.py
```

Optional CLI overrides:

```bash
python scripts/run_baseline.py --rounds 50 --clients 8 --alpha 0.3 --seed 42
```

Standalone scripts take their hyperparameters from `configs/*.yaml`. The expected behavior below corresponds to the robust-sweep settings (3 local epochs, learning rate 0.05; see "Running the Full Experiment"):

```text
Round 1/50  -> Accuracy: ~88%
Round 5/50  -> Accuracy: ~98%
Round 50/50 -> Accuracy: ~99%
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
python scripts/run_dp_experiment.py --epsilon 5.0 --clients 8 --rounds 50 --delta 1e-5 --seed 42
```

---

## Running the Full Experiment

To execute the full multi-condition, multi-seed evaluation matrix:

```bash
python scripts/run_all_experiments.py
```

The default robust sweep uses 50 federated rounds, 3 local epochs per round, learning rate 0.05, clipping norm 0.5, five seeds (42-46), $\delta = 10^{-5}$, and target $\varepsilon \in \{0.1, 0.5, 1, 5, 20, 100\}$. It also runs a standard no-DP baseline and a clipping-only control (Opacus clipping with zero added noise). To override the sweep:

```bash
python scripts/run_all_experiments.py --epsilons 0.1 0.5 1 5 20 100 \
    --seeds 42 43 44 45 46 --rounds 50 --local-epochs 3 \
    --learning-rate 0.05 --max-grad-norm 0.5 --output-dir results_robust
```

Use `--skip-clipping-only` to omit the clipping control. The results CSV includes cumulative per-client epsilon, the actual noise multiplier, train accuracy, mean clipped-gradient norm, injected-noise norm, and per-sample pre/post-clipping norms. The generated report and plots summarize the five-seed mean and standard deviation.

The runner automatically:

1. Executes the no-DP baseline and clipping-only control across all seeds.
2. Calibrates one noise multiplier for each requested epsilon over the full configured run and executes the DP-SGD runs across all seeds.
3. Records per-round metrics to `<output-dir>/processed/experiment_results.csv`.
4. Generates plots in `<output-dir>/plots/` and a report in `reports/`.

---

## Results & Visualizations

Generated plots (written to `<output-dir>/plots/`; the completed sweep is in `results_robust/plots/`):

* **Plot 1: `plot1_accuracy_vs_round.png`**: Global test accuracy vs. federated round for every condition (convergence trajectories, mean with shaded standard deviation).
* **Plot 2: `plot2_accuracy_vs_epsilon.png`**: Final global test accuracy vs. target privacy budget $\varepsilon$ (log scale), with the no-DP baseline and clipping-only reference lines.
* **Plot 3: `plot3_accuracy_degradation.png`**: Accuracy drop (percentage points) from the no-DP baseline vs. $\varepsilon$. This drop includes the clipping cost, which is paid at every $\varepsilon$.
* **Plot 4: `plot4_client_distribution.png`**: Heatmap of the Dirichlet non-IID partition across the eight simulated vehicles (one partition).

To regenerate plots and the report from existing results without retraining:

```bash
python scripts/plot_results.py \
    --results-file results_robust/processed/experiment_results.csv \
    --output-dir results_robust/plots
python scripts/generate_report.py \
    --results-file results_robust/processed/experiment_results.csv \
    --output-file reports/experiment_report_robust.md
```

### Completed Robust Sweep

The sweep contains 40 runs: a no-DP baseline, a clipping-only control, and six DP targets, each over five seeds (42-46) and 50 rounds. The processed CSV contains 2,040 round records (rounds 0-50 for each run). Final test accuracy (round 50) is reported as mean ± sample standard deviation across the five seeds. Noise multipliers are those used for seed 42; calibration is performed separately for each seed's client partition. All DP runs use $\delta = 10^{-5}$.

| Condition | Target ε | Noise Multiplier (seed 42) | Final Test Accuracy | Drop vs. No-DP (pts) | Gap to Clipping-Only (pts) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| No-DP baseline | N/A | N/A | 99.04 ± 0.09% | 0.00 | N/A |
| Clipping only, no noise | N/A | 0.0000 | 92.23 ± 0.41% | 6.81 | 0.00 |
| DP-SGD | 100 | 0.4907 | 92.23 ± 0.39% | 6.81 | 0.00 |
| DP-SGD | 20 | 0.8586 | 92.08 ± 0.45% | 6.96 | 0.15 |
| DP-SGD | 5 | 1.9580 | 90.95 ± 0.98% | 8.09 | 1.28 |
| DP-SGD | 1 | 7.7734 | 78.07 ± 4.07% | 20.97 | 14.16 |
| DP-SGD | 0.5 | 14.6875 | 67.54 ± 3.57% | 31.50 | 24.69 |
| DP-SGD | 0.1 | 100.0000 | 25.91 ± 3.89% | 73.13 | 66.32 |

The processed data, per-seed raw CSV/config files, and plots are in [`results_robust/`](results_robust/). The full generated analysis is in [`reports/experiment_report_robust.md`](reports/experiment_report_robust.md).

### Interpretation

* **Federated DP-SGD trains under severe non-IID skew.** Every DP configuration improves steadily over rounds. Whether the final accuracy is competitive depends strongly on $\varepsilon$.
* **Clipping is a fixed cost, separate from privacy budget.** With the settings used here, clipping alone reduces accuracy from about 99.0% to about 92.2% (≈6.8 points), even with no noise. The clipping-only run is the ceiling for DP-SGD in this configuration.
* **Large budgets add essentially nothing beyond clipping.** At $\varepsilon \ge 20$ the mean accuracy is within 0.15 points of clipping-only, well inside the seed-to-seed standard deviation (≈0.4 points). With five seeds, $\varepsilon = 20$, $\varepsilon = 100$, and clipping-only should be treated as indistinguishable.
* **The cost grows sharply below $\varepsilon \approx 5$.** Relative to clipping-only, the additional loss is about 1.3 points at $\varepsilon = 5$, 14 points at $\varepsilon = 1$, 25 points at $\varepsilon = 0.5$, and 66 points at $\varepsilon = 0.1$, where the model is only modestly better than chance.
* **Variance grows as $\varepsilon$ shrinks.** The standard deviation across seeds rises from ≈0.4 points (high $\varepsilon$) to ≈4 points ($\varepsilon \le 1$), so strong-privacy results are less predictable.
* **Low-$\varepsilon$ runs may not be converged.** Curves for $\varepsilon \le 0.5$ are still rising at round 50. Because noise is calibrated to the full 50-round horizon, simply running more rounds is not a free improvement: the budget is fixed, so more rounds means more noise per round.
* **$\varepsilon = 100$ provides negligible formal protection.** It is included as an upper anchor, not as a recommended operating point.

All of these statements apply to this specific setup (MNIST, a small CNN, hyperparameters fixed across all $\varepsilon$ rather than tuned per $\varepsilon$). Tuning the learning rate and clipping norm per privacy budget may narrow the gaps.

### Changes Since the Initial Version

* **Full-run noise calibration:** The server calibrates a separate noise multiplier for each target epsilon over the complete training horizon (federated rounds × local epochs). The smallest client partition is used for a conservative (highest) sampling-rate estimate, and the calibrated multiplier is passed to clients and saved in each run's config. Runtime privacy state is reset between independent sweep conditions.
* **Cumulative privacy accounting:** Each client retains an RDP accountant across federated rounds. The CSV reports the maximum cumulative epsilon across clients for each round. Round 0 is the pre-training evaluation, before privacy budget is spent.
* **Noise and clipping diagnostics:** Client training records the mean per-sample gradient norm before clipping, the mean clipped per-sample norm, the aggregate clipped-gradient norm, and the injected-noise norm. These are aggregated per round alongside train loss and train accuracy.
* **Clipping-only control:** Opacus per-sample clipping with a zero noise multiplier isolates clipping effects from added noise. It is reported separately from both noisy DP-SGD and the unmodified no-DP baseline, and it is not a differentially private condition.
* **Robust sweep defaults:** The runner defaults to $\varepsilon \in \{0.1, 0.5, 1, 5, 20, 100\}$, five seeds, 50 rounds, three local epochs, learning rate 0.05, and clipping norm 0.5. All can be overridden on the command line.
* **Results and reporting:** CSV output includes train/test metrics, actual noise multipliers, cumulative epsilon, and clipping/noise diagnostics. Reports derive their epsilon columns from the input CSV, distinguish the clipping-only control from the true baseline, and resolve plot paths from the supplied results location.
* **Regression tests:** The suite has 17 tests, including checks for cumulative epsilon across rounds, zero-noise clipping, fresh calibration state for each epsilon, and privacy fields written to CSV.

---

## Privacy Mechanism

### Where is DP Applied?

This experiment applies **client-side DP-SGD during local training**. For a batch $B_t$, per-sample gradients $g_i$ are clipped, summed, perturbed once with Gaussian noise, and averaged:

$$
\tilde{g}_t = \frac{1}{|B_t|} \left( \sum_{i \in B_t} \frac{g_i}{\max(1, \|g_i\|_2 / C)} + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}) \right)
$$

1. **Per-sample gradient clipping:** Each sample's gradient $g_i$ is clipped to a maximum $\ell_2$ norm $C$ (the robust-sweep default is $C = 0.5$).
2. **Gaussian noise injection:** Noise with standard deviation $\sigma C$ is added once to the sum of clipped gradients, where $\sigma$ is the noise multiplier.
3. **Privacy accounting:** A persistent per-client RDP (Rényi Differential Privacy) accountant tracks the cumulative $(\varepsilon, \delta)$ spent across federated rounds, using the client's sampling rate, the actual noise multiplier, and the number of optimization steps. Noise is calibrated so that the target $\varepsilon$ is met over the full configured run.

### How the Reported ε Should Be Read

* **Per-client, record-level guarantee.** The reported $\varepsilon$ is the maximum cumulative value across the eight clients, for a single client's local dataset (adding or removing one record).
* **Conservative calibration.** Because the noise multiplier is calibrated using the smallest client partition (highest sampling rate), clients with more data accumulate a lower $\varepsilon$ than the target. They are effectively over-protected, which costs some accuracy.
* **Target vs. achieved.** The CSV records the achieved cumulative $\varepsilon$ per round. Confirm that the value at the final round does not exceed the target for each condition.

### Privacy Scope & Honest Distinctions

* **Provided:** Formal $(\varepsilon, \delta)$-DP at the record level for each client's local data, applied to the model updates that clients send. This bounds what the server or any observer of the updates can infer about any individual record.
* **Not provided:** Client-level DP (a vehicle's participation in a round is not hidden; all clients participate in every round), secure aggregation (updates are unencrypted in transit to the server), or Byzantine fault tolerance.
* **Not evaluated:** No empirical attacks (reconstruction or membership inference) were run. The guarantees are formal accounting results, not measured attack resistance.

---

## Non-IID Data Distribution

Local vehicle data is partitioned across 8 clients using a Dirichlet distribution:

$$
p_k \sim \text{Dirichlet}(\alpha)
$$

* Default concentration: $\alpha = 0.3$.
* The partition covers the full 60,000-sample MNIST training set.
* Skew affects both **label distribution** and **client dataset size**. In the partition shown in Plot 4, client sizes range from about 2,660 samples (Vehicle 6) to about 11,640 (Vehicle 4), roughly a 4.4× spread.
* Label skew is strong. For example, Vehicle 4 is dominated by digits 0 and 3, Vehicle 3 by digits 7 and 8, Vehicle 7 by digits 1 and 6, and Vehicle 8 by digits 2 and 9. Many clients hold almost none of some classes (single-digit counts).
* The exact partition depends on the seed, so Plot 4 shows one realization.
* The no-DP baseline reaches about 99% despite this skew, so FedAvg tolerates this partition on MNIST. No IID control was run, so the effect of skew itself on accuracy is not isolated.

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
├── results_robust/          # completed 40-run sweep
│   ├── raw/                 # per-seed CSV and config files
│   ├── processed/           # experiment_results.csv
│   └── plots/
│
├── reports/
│   ├── experiment_report.md
│   └── experiment_report_robust.md
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

* **Partition seed:** Within a given seed, all conditions (baseline, clipping-only, every $\varepsilon$) use the same data assignment. Different seeds produce different partitions, so the reported standard deviations include partition variance as well as training variance.
* **Weight initialization seed:** Within a given seed, all conditions start from identical initial global parameters.
* **Multi-seed evaluation:** The sweep uses five seeds (42-46) and reports mean ± sample standard deviation. With $n = 5$, small differences (a few tenths of a point) should not be interpreted as ordering.

---

## Limitations

1. **Benchmark Proxy:** MNIST is grayscale $28 \times 28$ digits; real vehicle telemetry is continuous, temporal, and multi-modal. Results will not transfer directly.
2. **Scale:** Eight simulated clients, all participating in every round. Production fleets involve tens of thousands of intermittent nodes with client sampling.
3. **Untuned DP hyperparameters:** Learning rate, clipping norm, and local epochs are fixed across all $\varepsilon$. Clipping alone costs about 6.8 points in this configuration, and tuning per privacy budget could reduce the DP gap.
4. **Aggregate accuracy only:** Per-class accuracy was not analysed. Strong DP may hurt rare classes disproportionately.
5. **No IID control:** The contribution of non-IID skew to the observed accuracy was not separated out.
6. **No attack evaluation:** Privacy is reported through formal accounting only.
7. **No Encryption:** Shared weights are transmitted as plain floating-point arrays; full fleet security requires Secure Aggregation or Homomorphic Encryption.

---

## Future Work

* Evaluation on CIFAR-10 and driving benchmarks (e.g., GTSRB, BDD100K).
* Per-class accuracy analysis and an IID-partition control.
* Per-budget hyperparameter tuning (learning rate, clipping norm, local epochs).
* Empirical privacy audits (membership inference, gradient reconstruction).
* Combining client-level DP with secure aggregation protocols.
* Decentralized second-order curvature approximations (AD-KFAC).
* Gradient sparsification and quantization to reduce cellular communication costs.
