# Low-Rank SVD Neural Compression & Hardware Profiling

An empirical and theoretical research project investigating post-training Singular Value Decomposition (SVD) on deep neural network projection layers, characterizing the divergence between theoretical FLOP reductions and actual wall-clock execution latency on modern hardware accelerators.

---

## 1. Core Research Thesis

When a dense linear projection layer $W \in \mathbb{R}^{M \times N}$ is factorized into cascaded low-rank linear layers $L_1 \in \mathbb{R}^{M \times K}$ and $L_2 \in \mathbb{R}^{K \times N}$ ($K \ll \min(M, N)$ without an intermediate activation function), theoretical floating-point operations scale down linearly:
$$\text{FLOPs}_{\text{decomp}} = 2 \cdot B_{sz} \cdot K \cdot (M + N)$$

However, actual wall-clock execution time frequently stagnates—or even regresses—due to hardware constraints: memory bandwidth ceilings, low arithmetic intensity, kernel launch latency, and intermediate tensor memory round-trips.

> **Capstone Thesis:** By compressing dense projection layers via truncated SVD, we systematically alter operational arithmetic intensity and memory traffic per forward pass, empirically modeling the memory-bandwidth hardware constraints found in autoregressive LLM KV-cache retrieval.

---

## 2. Repository Architecture

```text
├── models/
│   ├── baseline.py          # BaselineMLP (4-layer dense) & SmallCNN definitions
│   └── decomposed.py        # DecomposedLinear (L1/L2) & SVD layer-swapping mechanics
├── profiling/
│   ├── harness.py           # Jitter-free hardware timing engine (CUDA Events / MPS / CPU)
│   ├── metrics.py           # Theoretical FLOPs, MACs, breakeven rank, and Roofline math
│   └── memory.py            # Parameter footprint & peak VRAM allocation tracker
├── scripts/
│   ├── image_svd.py         # Phase 1: 2D image SVD intuition builder & spectral decay
│   ├── train_finetune.py    # Phase 2 & 3: Model training, SVD decomposition, fine-tuning loop
│   └── run_sweep.py         # Systematic multi-variable rank and batch size hardware sweeps
├── visualizations/
│   ├── pareto.py            # Publication-quality Pareto curves & empirical Roofline plots
├── data/
│   ├── dataset_loader.py    # High-dimensional calibrated dataset split generator
├── tests/
│   └── test_svd_core.py     # Comprehensive automated test suite
└── results/
    ├── checkpoints/         # Saved baseline and fine-tuned .pt model weights
    ├── plots/               # High-resolution generated empirical figures
    ├── accuracy_pareto_data.csv # Quantified accuracy retention across ranks
    └── benchmark_data.csv   # Unified benchmarking metrics (70 configurations)
```

---

## 3. Quick Start & Reproduction

### Prerequisites
Setup the Python 3.12 virtual environment:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision numpy pandas matplotlib scipy pillow tqdm
```

### Run Unit Tests
```bash
python3 -m unittest discover tests
```

### 1. Phase 1 Image SVD Intuition Builder
```bash
PYTHONPATH=. python3 scripts/image_svd.py
```
Outputs reconstruction grids and singular value energy decay curves to `results/plots/`.

### 2. End-to-End Training & Post-SVD Fine-Tuning
```bash
PYTHONPATH=. python3 scripts/train_finetune.py --epochs 3
```
Trains `BaselineMLP` to ~86.2% convergence, performs SVD layer decomposition across ranks $k \in \{8, 16, 32, 64, 128, 256\}$, and runs fine-tuning loops saving checkpoints to `results/checkpoints/`.

### 3. Hardware Profiling Sweeps & Roofline Analysis
```bash
PYTHONPATH=. python3 scripts/run_sweep.py
```
Sweeps ranks and batch sizes $B_{sz} \in \{1, 8, 32, 64, 128\}$, logging to `results/benchmark_data.csv` and generating publication plots.

### 4. Interactive Results Dashboard & Summary
```bash
PYTHONPATH=. python3 scripts/show_results.py --open
```
Displays terminal summary tables and opens the interactive results dashboard (`results/dashboard.html`).

---

## 4. Key Empirical Findings

### 1. The $B_{sz}=1$ Latency Stagnation (The Memory Wall)
* For a $1024 \to 512$ projection layer, rank $k=8$ achieves a **$41.0\times$ parameter compression** and **$22.7\times$ theoretical FLOP reduction**.
* At batch size $B_{sz}=1$, wall-clock latency remains virtually identical to the uncompressed baseline (~$0.23\text{ ms}$).
* **Why?** Arithmetic intensity at $B_{sz}=1$ is pinned below $0.50\text{ FLOP/Byte}$. Execution is strictly memory-bandwidth bound (Williams et al., 2009). The factorized layer introduces a second GEMM kernel launch overhead and intermediate tensor memory round-trips.

### 2. Compute-Bound Scaling at High Batch Sizes
* At $B_{sz}=128$, arithmetic intensity rises up to $38.4\text{ FLOP/Byte}$, crossing into compute-bound operational space where arithmetic FLOP savings translate to throughput gains.

### 3. Accuracy Recovery (Sainath et al., 2013)
* Truncating to rank $k=8$ zero-shot reduces parameter count by $39.25\times$.
* Fine-tuning with AdamW and cosine annealing recovers accuracy back to **$87.29\%$**, exceeding the original uncompressed model ($86.18\%$) via low-rank regularization.

---

## 5. Research Grounding & Citations
1. **Williams et al. (2009)**: *Roofline: An Insightful Visual Performance Model for Multicore Architectures.*
2. **Hooker (2020)**: *The Hardware Lottery.*
3. **Denton et al. (2014)**: *Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation.*
4. **Sainath et al. (2013)**: *Low-rank matrix factorization for Deep Neural Network training with application to speech recognition.*
5. **Zhang et al. (2024)**: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.*
